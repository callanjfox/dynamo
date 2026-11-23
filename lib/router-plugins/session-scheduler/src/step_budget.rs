// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! f49i: the per-worker step-time budget (`admission.worker_step_budget`), off (absent) by default.
//!
//! `{classes: [priority, standard], margin: 0.9}`. A worker keeps the budget for a turn when, with
//! the turn's prefill added, the projected mean ITL of every listed class decoding there (and of the
//! turn's own class when it is listed) is at most `margin` times that class's service-level ITL.
//!
//! The projection ([`crate::forecast::Forecast::step_budget_verdict`]) is the forecast's: its step
//! model (`step_table_v2` when given, else `step_table` or AISimulate, through
//! [`crate::forecast::StepModel::mixed_s`]) and K's mixed-step decode window. The decode batch is
//! the running batch plus the turn, at their mean context, as the forecast reads it (the engine's
//! gauges with `class_backlog`, else the router's booked sample), every class's decodes together
//! since they share the step. Over a class's decode window (its mean output tokens at the bare
//! step) the prefill beside it is the worker's queued and in-flight prefill, plus the turn's
//! computed tokens, plus the v2.2 arrival term (the computed prefill sent to the worker over the
//! trailing 60 s, as a rate, times the window). The turn's own class, when it is not decoding there
//! yet, decodes after the queue and its own prefill: only the arrival term runs beside it.
//!
//! Where it acts:
//! - The gate (`hooks.rs` `check_wake`, enforce mode): a turn every other gate rule lets go waits
//!   while no worker keeps the budget, at most its class's `max_lobby_wait_s` since it was queued
//!   (then it goes, and the forecast route prefers the worker over least). It never waits while a
//!   worker keeps it. It is asked again at every completion (the gate's notify) and at least every
//!   [`RECHECK_S`] (a prefill drain lowers the projection without an event). A gate target that
//!   breaks the budget is dropped; with no forecast route installed, the fitting worker that
//!   computes least becomes the target.
//! - The forecast route (`route.rs`): a candidate that breaks the budget is excluded while another
//!   keeps it, in the route's own order (the ITL limit's hard rule, [`crate::route::itl_limit_costs`],
//!   on the budget's ratios); when none keeps it, the one over least wins.
//!
//! Diagnostics, only with `DYNAMO_WSB_LOG=<path>`: one JSON line per refused (worker, turn) pair,
//! once per pair per simulated minute and at most [`MAX_LINES_PER_MINUTE`] a minute, and one line
//! of counts per simulated minute.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Write;

use dynamo_kv_router::plugins::prelude::WorkerWithDpRank;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::clock::Time;
use crate::config::{ConfigError, ServicePolicy};

/// Longest a held turn waits before the gate asks again without an event.
pub const RECHECK_S: f64 = 1.0;
/// Requests whose per-worker computed tokens are kept (oldest dropped beyond).
const MAX_TRACKED: usize = 65_536;
/// Refused-pair lines written per simulated minute at most; the rest are counted as suppressed.
pub const MAX_LINES_PER_MINUTE: u64 = 2_000;

/// `admission.worker_step_budget`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct WorkerStepBudgetConfig {
    /// Classes whose decodes the budget protects; each needs a `service_level`.
    pub classes: Vec<String>,
    /// The budget as a share of each class's ITL limit, in (0, 1].
    pub margin: f64,
}

impl Default for WorkerStepBudgetConfig {
    fn default() -> Self {
        Self {
            classes: vec!["priority".to_owned(), "standard".to_owned()],
            margin: 0.9,
        }
    }
}

impl WorkerStepBudgetConfig {
    pub fn validate(&self, policy: &ServicePolicy) -> Result<(), ConfigError> {
        let invalid = |reason: String| Err(ConfigError::WorkerStepBudget(reason));
        if self.classes.is_empty() {
            return invalid("classes must name at least one class".to_owned());
        }
        for name in &self.classes {
            match policy.classes.iter().find(|c| &c.name == name) {
                None => return invalid(format!("class {name:?} is not one of the classes")),
                Some(c) if c.service_level.is_none() => {
                    return invalid(format!(
                        "class {name:?} has no service_level (its itl_s is the budget)"
                    ));
                }
                Some(_) => {}
            }
        }
        if !(self.margin > 0.0 && self.margin <= 1.0) {
            return Err(ConfigError::PolicyOutOfRange {
                field: "admission.worker_step_budget.margin",
                expected: "in (0, 1]",
                value: self.margin,
            });
        }
        if !policy.forecast.enabled {
            return invalid("needs forecast.enabled (the projection is the forecast's)".to_owned());
        }
        Ok(())
    }
}

/// One worker judged for one turn.
#[derive(Clone, Debug, PartialEq)]
pub struct Check {
    pub worker: WorkerWithDpRank,
    /// The tokens the turn computes there.
    pub new_tokens: u64,
    /// Projected mean ITL, seconds, of each listed class judged there (decoding there, or the
    /// turn's own).
    pub itls: Vec<(String, f64)>,
    /// The largest projected ITL over its class's budget (`margin` x `itl_s`); 0 with none judged.
    pub ratio: f64,
}

impl Check {
    pub fn fits(&self) -> bool {
        self.ratio <= 1.0
    }
}

/// Every sampled worker judged for one turn, in worker order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Verdict {
    pub checks: Vec<Check>,
}

impl Verdict {
    pub fn any_fits(&self) -> bool {
        self.checks.iter().any(Check::fits)
    }

    /// Whether `worker` keeps the budget; a worker not judged (no sample) does.
    pub fn fits(&self, worker: WorkerWithDpRank) -> bool {
        self.checks
            .iter()
            .find(|c| c.worker == worker)
            .is_none_or(Check::fits)
    }

    pub fn refused(&self) -> impl Iterator<Item = &Check> {
        self.checks.iter().filter(|c| !c.fits())
    }

    pub fn fitting(&self) -> Vec<WorkerWithDpRank> {
        self.checks
            .iter()
            .filter(|c| c.fits())
            .map(|c| c.worker)
            .collect()
    }

    /// The fitting worker that computes least (then the lowest ratio, then the lowest worker).
    pub fn best_fitting(&self) -> Option<WorkerWithDpRank> {
        self.checks
            .iter()
            .filter(|c| c.fits())
            .min_by(|a, b| {
                a.new_tokens
                    .cmp(&b.new_tokens)
                    .then_with(|| a.ratio.total_cmp(&b.ratio))
                    .then_with(|| a.worker.cmp(&b.worker))
            })
            .map(|c| c.worker)
    }
}

/// What the gate did with a turn it judged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateOutcome {
    /// No worker fits: held.
    Held,
    /// Some worker fits: let go, with the gate's target when it set one.
    Go { chosen: Option<WorkerWithDpRank> },
    /// No worker fits but the turn has waited its class's wait limit: let go.
    Overdue,
}

#[derive(Default)]
struct Computed {
    map: HashMap<String, Vec<(WorkerWithDpRank, u64)>>,
    order: VecDeque<String>,
}

/// The budget, held by the forecast (built from the policy with it).
pub struct StepBudget {
    classes: Vec<String>,
    /// `margin` x `itl_s` per listed class.
    budget: HashMap<String, f64>,
    computed: Mutex<Computed>,
    log: Option<Mutex<WsbLog>>,
}

impl StepBudget {
    /// The budget `policy` configures; `None` when `admission.worker_step_budget` is absent.
    pub fn from_policy(policy: &ServicePolicy) -> Option<Self> {
        let config = policy.admission.worker_step_budget.as_ref()?;
        let budget = config
            .classes
            .iter()
            .filter_map(|name| {
                let level = policy
                    .classes
                    .iter()
                    .find(|c| &c.name == name)?
                    .service_level
                    .as_ref()?;
                Some((name.clone(), config.margin * level.itl_s))
            })
            .collect();
        Some(Self {
            classes: config.classes.clone(),
            budget,
            computed: Mutex::new(Computed::default()),
            log: std::env::var_os("DYNAMO_WSB_LOG")
                .map(|path| Mutex::new(WsbLog::new(std::path::PathBuf::from(path)))),
        })
    }

    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    /// `margin` x `itl_s` of `class`, when it is listed.
    pub fn budget(&self, class: &str) -> Option<f64> {
        self.budget.get(class).copied()
    }

    /// The largest projected ITL over its class's budget; 0 with none.
    pub fn ratio(&self, itls: &[(String, f64)]) -> f64 {
        itls.iter()
            .filter_map(|(class, itl)| Some(itl / self.budget(class)?.max(1e-9)))
            .fold(0.0, f64::max)
    }

    /// Keep the tokens `request_id` computes on each worker (noted at classify).
    pub fn note_compute(&self, request_id: &str, per_worker: Vec<(WorkerWithDpRank, u64)>) {
        let mut c = self.computed.lock();
        if c.map.insert(request_id.to_owned(), per_worker).is_none() {
            c.order.push_back(request_id.to_owned());
        }
        while c.order.len() > MAX_TRACKED {
            if let Some(old) = c.order.pop_front() {
                c.map.remove(&old);
            }
        }
    }

    /// The tokens `request_id` computes on each worker, when noted.
    pub fn computed(&self, request_id: &str) -> Option<Vec<(WorkerWithDpRank, u64)>> {
        self.computed.lock().map.get(request_id).cloned()
    }

    /// Diagnostics: the gate's verdict on one turn and what it did.
    pub fn record_gate(
        &self,
        t: Time,
        request_id: &str,
        class: &str,
        verdict: &Verdict,
        outcome: GateOutcome,
    ) {
        let Some(log) = &self.log else {
            return;
        };
        let mut log = log.lock();
        let t = t.as_secs_f64();
        log.roll(t);
        let refused: Vec<&Check> = verdict.refused().collect();
        log.counts.checks += 1;
        log.counts.refused_pairs += refused.len() as u64;
        let decision = match outcome {
            GateOutcome::Held => {
                log.counts.held += 1;
                "held"
            }
            GateOutcome::Go { .. } if refused.is_empty() => {
                log.counts.go_clean += 1;
                "go"
            }
            GateOutcome::Go { .. } => {
                log.counts.go_steered += 1;
                "go"
            }
            GateOutcome::Overdue => {
                log.counts.overdue += 1;
                "overdue"
            }
        };
        if refused.is_empty() {
            return;
        }
        let chosen = match outcome {
            GateOutcome::Go {
                chosen: Some(worker),
            } => json_str(&worker_name(worker)),
            _ => "null".to_owned(),
        };
        let fitting = verdict
            .fitting()
            .into_iter()
            .map(|w| json_str(&worker_name(w)))
            .collect::<Vec<_>>()
            .join(",");
        for check in refused {
            if !log.seen.insert((request_id.to_owned(), check.worker)) {
                continue;
            }
            if log.lines >= MAX_LINES_PER_MINUTE {
                log.counts.suppressed += 1;
                continue;
            }
            log.lines += 1;
            let projected = check
                .itls
                .iter()
                .map(|(c, s)| format!("{}:{}", json_str(c), json_num(*s)))
                .collect::<Vec<_>>()
                .join(",");
            let limits = check
                .itls
                .iter()
                .filter_map(|(c, _)| Some(format!("{}:{}", json_str(c), json_num(self.budget(c)?))))
                .collect::<Vec<_>>()
                .join(",");
            let line = format!(
                "{{\"t\":{},\"ev\":\"refused\",\"request\":{},\"worker\":{},\"class\":{},\"projected_itl\":{{{projected}}},\"limit\":{{{limits}}},\"ratio\":{},\"decision\":\"{decision}\",\"chosen\":{chosen},\"fitting\":[{fitting}]}}",
                json_num(t),
                json_str(request_id),
                json_str(&worker_name(check.worker)),
                json_str(class),
                json_num(check.ratio),
            );
            log.write(&line);
        }
    }

    /// Diagnostics: the forecast route ruled one selection; `kept` whether a candidate kept the
    /// budget, `excluded` how many did not.
    pub fn record_route(&self, t: Time, kept: bool, excluded: u64) {
        let Some(log) = &self.log else {
            return;
        };
        let mut log = log.lock();
        log.roll(t.as_secs_f64());
        log.counts.route_ruled += 1;
        log.counts.route_kept += u64::from(kept);
        log.counts.route_excluded += excluded;
    }
}

#[derive(Debug, Default)]
struct MinuteCounts {
    checks: u64,
    held: u64,
    go_clean: u64,
    go_steered: u64,
    overdue: u64,
    refused_pairs: u64,
    route_ruled: u64,
    route_kept: u64,
    route_excluded: u64,
    suppressed: u64,
}

/// `DYNAMO_WSB_LOG`: opened at the first line, so a forecast rebuilt before any decision leaves
/// no stale file behind.
struct WsbLog {
    path: std::path::PathBuf,
    out: Option<std::io::BufWriter<std::fs::File>>,
    failed: bool,
    minute: Option<u64>,
    counts: MinuteCounts,
    /// (request, worker) pairs already written this minute.
    seen: HashSet<(String, WorkerWithDpRank)>,
    lines: u64,
}

impl WsbLog {
    fn new(path: std::path::PathBuf) -> Self {
        Self {
            path,
            out: None,
            failed: false,
            minute: None,
            counts: MinuteCounts::default(),
            seen: HashSet::new(),
            lines: 0,
        }
    }

    fn write(&mut self, line: &str) {
        if self.out.is_none() && !self.failed {
            match std::fs::File::create(&self.path) {
                Ok(f) => self.out = Some(std::io::BufWriter::new(f)),
                Err(e) => {
                    self.failed = true;
                    eprintln!("DYNAMO_WSB_LOG {}: {e}", self.path.display());
                }
            }
        }
        if let Some(out) = self.out.as_mut() {
            let _ = writeln!(out, "{line}");
            let _ = out.flush();
        }
    }

    /// Close the minutes before `t`'s with one line of counts.
    fn roll(&mut self, t: f64) {
        let minute = (t.max(0.0) / 60.0).floor() as u64;
        match self.minute {
            None => self.minute = Some(minute),
            Some(prev) if minute > prev => {
                let c = std::mem::take(&mut self.counts);
                let line = format!(
                    "{{\"t\":{},\"ev\":\"minute\",\"minute\":{prev},\"checks\":{},\"held\":{},\"go_clean\":{},\"go_steered\":{},\"overdue\":{},\"refused_pairs\":{},\"route_ruled\":{},\"route_kept\":{},\"route_excluded\":{},\"suppressed\":{}}}",
                    json_num(t),
                    c.checks,
                    c.held,
                    c.go_clean,
                    c.go_steered,
                    c.overdue,
                    c.refused_pairs,
                    c.route_ruled,
                    c.route_kept,
                    c.route_excluded,
                    c.suppressed,
                );
                self.write(&line);
                self.seen.clear();
                self.lines = 0;
                self.minute = Some(minute);
            }
            Some(_) => {}
        }
    }
}

fn worker_name(w: WorkerWithDpRank) -> String {
    format!("{}:{}", w.worker_id, w.dp_rank)
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_num(x: f64) -> String {
    if x.is_finite() {
        format!("{x:.6}")
    } else {
        "null".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use dynamo_kv_router::plugins::prelude::*;
    use dynamo_kv_router::plugins::request_classifier::{WorkerCacheView, WorkerLoadView};
    use tokio::time::Instant;

    use super::*;
    use crate::admit::LoggedDecision;
    use crate::clock::ManualClock;
    use crate::config::Mode;
    use crate::hooks::SessionSchedulerClassifier;
    use crate::route::{Row, itl_limit_costs, route_costs};

    fn w(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(id, 0)
    }

    /// Workers with a load (prefill tokens, decode blocks, running requests) and no cache rows.
    struct Fleet(Vec<(WorkerWithDpRank, usize, usize, usize)>);

    impl WorkerView for Fleet {
        fn len(&self) -> usize {
            self.0.len()
        }

        fn info(&self, row: usize) -> Option<WorkerInfo> {
            self.0
                .get(row)
                .map(|r| WorkerInfo::new(r.0, WorkerCapacity::new(Some(4096), Some(1000))))
        }

        fn load(&self, row: usize) -> Option<WorkerLoadView> {
            self.0.get(row).map(|r| WorkerLoadView {
                available: true,
                active_prefill_tokens: r.1,
                active_decode_blocks: r.2,
                decode_cost_blocks: 0.0,
                active_requests: r.3,
            })
        }

        fn cache(&self, _row: usize) -> Option<WorkerCacheView> {
            None
        }
    }

    fn heavy(ids: &[u64]) -> Fleet {
        Fleet(ids.iter().map(|&i| (w(i), 100_000, 64, 2)).collect())
    }

    fn policy(classes: Option<&[&str]>) -> ServicePolicy {
        let mut p = ServicePolicy {
            mode: Mode::Enforce,
            ..Default::default()
        };
        p.forecast.enabled = true;
        p.forecast.snapshot_interval_s = 0.0;
        p.admission.worker_step_budget = classes.map(|c| WorkerStepBudgetConfig {
            classes: c.iter().map(|s| (*s).to_owned()).collect(),
            margin: 0.9,
        });
        p
    }

    fn classifier(policy: ServicePolicy) -> SessionSchedulerClassifier {
        policy.validate().unwrap();
        SessionSchedulerClassifier::new(
            policy,
            Arc::new(ManualClock::new()),
            Arc::new(|| (1_000_000_000, 2)),
        )
        .with_block_size(16)
    }

    fn request(id: &str, class: &str, tokens: usize) -> ClassifyRequest {
        let mut r = ClassifyRequest::for_test(
            tokens,
            Some(id),
            Some(SessionContext::new(format!("s-{id}"), None, None, None)),
        );
        r.set_policy_class(class);
        r
    }

    /// A turn's gate answer (its session is `s-<id>`).
    fn check(gate: &dyn DispatchGate, id: &str, tokens: usize) -> DispatchDecision {
        let session = SessionContext::new(format!("s-{id}"), None, None, None);
        gate.check(&DispatchHead::new(
            Some(id),
            Some(&session),
            tokens,
            None,
            "default",
            Instant::now(),
        ))
    }

    /// A turn of `class` let go, sent to `worker` and decoding there.
    async fn decoding(
        c: &mut SessionSchedulerClassifier,
        gate: &dyn DispatchGate,
        id: &str,
        class: &str,
        worker: WorkerWithDpRank,
        fleet: &Fleet,
    ) {
        c.classify(request(id, class, 1000), fleet).unwrap();
        assert!(
            matches!(check(gate, id, 1000), DispatchDecision::Go { .. }),
            "{id}"
        );
        let session = SessionContext::new(format!("s-{id}"), None, None, None);
        gate.dispatched(
            &DispatchHead::new(
                Some(id),
                Some(&session),
                1000,
                None,
                "default",
                Instant::now(),
            ),
            Some(worker),
        );
        c.on_event(ClassifyEvent::Sent {
            request_id: id.into(),
            worker,
        })
        .await;
        c.on_event(ClassifyEvent::Responding {
            request_id: id.into(),
            worker,
        })
        .await;
    }

    #[tokio::test]
    async fn a_worker_over_budget_is_skipped_and_the_next_fitting_one_is_chosen() {
        let fleet = Fleet(vec![
            (w(1), 100_000, 64, 2),
            (w(2), 0, 64, 2),
            (w(3), 0, 64, 2),
        ]);
        let mut c = classifier(policy(Some(&["priority"])));
        let gate = c.dispatch_gate().unwrap();
        decoding(&mut c, gate.as_ref(), "p1", "priority", w(1), &fleet).await;
        c.classify(request("s1", "standard", 2000), &fleet).unwrap();
        let forecast = c.forecast().unwrap();
        let v = forecast
            .step_budget_verdict("s1", "standard", 2000)
            .unwrap();
        // Worker 1's queued prefill plus this turn's stretches priority's decode window there.
        assert_eq!(v.fitting(), vec![w(2), w(3)]);
        let refused: Vec<&Check> = v.refused().collect();
        assert_eq!(refused.len(), 1);
        assert_eq!(refused[0].worker, w(1));
        assert_eq!(refused[0].itls[0].0, "priority");
        assert!(refused[0].itls[0].1 > 0.9 * 0.05, "{:?}", refused[0]);
        // Some worker fits: the gate lets the turn go, never to worker 1.
        match check(gate.as_ref(), "s1", 2000) {
            DispatchDecision::Go { target } => assert_ne!(target, Some(w(1))),
            other => panic!("{other:?}"),
        }
        // The route: worker 1 is first in the router's order; the budget's rule skips it for the
        // next one, worker 2.
        let rows: Vec<Row> = [1u64, 2, 3]
            .iter()
            .map(|&i| Row {
                worker: w(i),
                hit_tokens: 0.0,
                base_cost: i as f64,
                active_tokens: 0.0,
                over: false,
                room: true,
            })
            .collect();
        let (costs, _) = route_costs(&rows, None, 3, 20_000.0, true, None, |_: &Row| {
            Some((0.0, 0.0))
        });
        let ratios: Vec<f64> = rows
            .iter()
            .map(|r| {
                forecast
                    .step_budget_check(r.worker, "standard", 2000, 0)
                    .unwrap()
                    .ratio
            })
            .collect();
        let (ruled, kept) = itl_limit_costs(&costs, &ratios);
        assert!(kept);
        let pick = |cs: &[f64]| {
            (0..rows.len())
                .min_by(|a, b| {
                    (cs[*a] + rows[*a].base_cost).total_cmp(&(cs[*b] + rows[*b].base_cost))
                })
                .map(|i| rows[i].worker)
        };
        assert_eq!(pick(&costs), Some(w(1)));
        assert_eq!(pick(&ruled), Some(w(2)));
    }

    #[tokio::test]
    async fn every_worker_over_budget_holds_until_a_completion_lowers_the_projection() {
        let fleet = heavy(&[1, 2]);
        let mut c = classifier(policy(Some(&["priority"])));
        let gate = c.dispatch_gate().unwrap();
        let wakes = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&wakes);
        gate.attach(DispatchRecheck::new(Arc::new(move || {
            counter.fetch_add(1, Ordering::Relaxed);
        })));
        decoding(&mut c, gate.as_ref(), "p1", "priority", w(1), &fleet).await;
        // Worker 2 has no priority decode yet: priority's own turn fits there.
        decoding(&mut c, gate.as_ref(), "p2", "priority", w(2), &fleet).await;
        c.classify(request("s1", "standard", 2000), &fleet).unwrap();
        // Both workers would push priority over 0.9 x 50 ms: the turn waits at the router.
        for _ in 0..2 {
            assert!(matches!(
                check(gate.as_ref(), "s1", 2000),
                DispatchDecision::Hold { .. }
            ));
        }
        let before = wakes.load(Ordering::Relaxed);
        c.on_event(ClassifyEvent::Completed {
            request_id: "p1".into(),
            worker: w(1),
            context_tokens: None,
            output_tokens: None,
        })
        .await;
        assert_eq!(
            wakes.load(Ordering::Relaxed),
            before + 1,
            "a completion asks held heads again"
        );
        // Worker 1 decodes no priority now: the turn goes.
        let v = c
            .forecast()
            .unwrap()
            .step_budget_verdict("s1", "standard", 2000)
            .unwrap();
        assert_eq!(v.fitting(), vec![w(1)]);
        assert!(matches!(
            check(gate.as_ref(), "s1", 2000),
            DispatchDecision::Go { .. }
        ));
    }

    #[tokio::test]
    async fn an_unlisted_class_is_not_checked() {
        let fleet = heavy(&[1]);
        for (classes, held) in [
            (&["priority"][..], false),
            (&["priority", "standard"][..], true),
        ] {
            let mut p = policy(Some(classes));
            p.classes[1].service_level.as_mut().unwrap().itl_s = 0.05;
            let mut c = classifier(p);
            let gate = c.dispatch_gate().unwrap();
            decoding(&mut c, gate.as_ref(), "d1", "standard", w(1), &fleet).await;
            c.classify(request("s1", "standard", 2000), &fleet).unwrap();
            let answer = check(gate.as_ref(), "s1", 2000);
            assert_eq!(
                matches!(answer, DispatchDecision::Hold { .. }),
                held,
                "{classes:?}: {answer:?}"
            );
            let v = c
                .forecast()
                .unwrap()
                .step_budget_verdict("s1", "standard", 2000)
                .unwrap();
            // Standard decodes on worker 1; only a budget that lists it judges it.
            assert_eq!(v.checks[0].itls.is_empty(), !held);
        }
    }

    #[tokio::test]
    async fn the_arrival_term_raises_the_projection_of_a_class_not_decoding_yet() {
        let fleet = Fleet(vec![(w(1), 0, 64, 2)]);
        let mut c = classifier(policy(Some(&["priority"])));
        let forecast = c.forecast().unwrap();
        c.classify(request("x0", "batch", 10), &fleet).unwrap();
        let bare = forecast.step_budget_check(w(1), "priority", 0, 0).unwrap();
        assert_eq!(bare.itls.len(), 1, "the turn's own listed class is judged");
        // 600,000 computed tokens sent in the last minute: 10,000 tokens a second arrive.
        c.classify(request("x1", "batch", 600_000), &fleet).unwrap();
        c.on_event(ClassifyEvent::Sent {
            request_id: "x1".into(),
            worker: w(1),
        })
        .await;
        let loaded = forecast.step_budget_check(w(1), "priority", 0, 0).unwrap();
        assert!(loaded.itls[0].1 > bare.itls[0].1, "{bare:?} {loaded:?}");
        // A batch turn is not listed: nothing is judged for it on an idle worker.
        let batch = forecast.step_budget_check(w(1), "batch", 0, 0).unwrap();
        assert!(batch.itls.is_empty() && batch.fits());
    }

    /// The answers of a fixed scenario and the decision log it leaves.
    async fn scenario(
        policy: ServicePolicy,
        fleet: &Fleet,
    ) -> (Vec<DispatchDecision>, Vec<LoggedDecision>) {
        let mut c = classifier(policy);
        let gate = c.dispatch_gate().unwrap();
        decoding(&mut c, gate.as_ref(), "p1", "priority", w(1), fleet).await;
        decoding(&mut c, gate.as_ref(), "p2", "priority", w(2), fleet).await;
        let mut answers = Vec::new();
        // Small turns: on an idle worker their prefill keeps priority's decodes under budget.
        for id in ["s1", "s2", "s3"] {
            c.classify(request(id, "standard", 200), fleet).unwrap();
            answers.push(check(gate.as_ref(), id, 200));
            answers.push(check(gate.as_ref(), id, 200));
        }
        let log = c.with_scheduler(|s| s.decisions().entries().cloned().collect());
        (answers, log)
    }

    #[tokio::test]
    async fn off_or_never_refusing_leaves_the_answers_and_the_decision_log_unchanged() {
        let light = Fleet(vec![(w(1), 0, 64, 2), (w(2), 0, 64, 2)]);
        let on = || policy(Some(&["priority", "standard"]));
        let off_light = scenario(policy(None), &light).await;
        assert_eq!(off_light, scenario(on(), &light).await);
        assert!(
            off_light
                .0
                .iter()
                .any(|a| matches!(a, DispatchDecision::Go { target: None }))
        );
        // Under load the option is the only difference: off answers as on an idle fleet, on holds
        // every turn.
        let off_heavy = scenario(policy(None), &heavy(&[1, 2])).await;
        assert_eq!(off_heavy.0, off_light.0);
        let on_heavy = scenario(on(), &heavy(&[1, 2])).await;
        assert!(
            on_heavy
                .0
                .iter()
                .all(|a| matches!(a, DispatchDecision::Hold { .. }))
        );
        // Absent from a policy file: none, never serialized, and the forecast has no budget.
        let p: ServicePolicy = serde_json::from_str(r#"{"mode": "enforce"}"#).unwrap();
        assert!(p.admission.worker_step_budget.is_none());
        assert!(
            !serde_json::to_string(&p.admission)
                .unwrap()
                .contains("worker_step_budget")
        );
        let c = classifier(policy(None));
        assert!(c.forecast().unwrap().step_budget().is_none());
    }

    #[test]
    fn validation_needs_listed_classes_with_service_levels_and_a_margin_in_zero_one() {
        let p: ServicePolicy = serde_json::from_str(
            r#"{"forecast": {"enabled": true},
                "admission": {"worker_step_budget": {"classes": ["priority", "standard"], "margin": 0.9}}}"#,
        )
        .unwrap();
        p.validate().unwrap();
        assert_eq!(
            p.admission.worker_step_budget,
            Some(WorkerStepBudgetConfig::default())
        );
        let defaults: ServicePolicy = serde_json::from_str(
            r#"{"forecast": {"enabled": true}, "admission": {"worker_step_budget": {}}}"#,
        )
        .unwrap();
        assert_eq!(
            defaults.admission.worker_step_budget,
            Some(WorkerStepBudgetConfig::default())
        );
        assert!(
            serde_json::from_str::<WorkerStepBudgetConfig>(r#"{"clases": []}"#).is_err(),
            "unknown keys are refused"
        );
        let with = |f: &dyn Fn(&mut WorkerStepBudgetConfig)| {
            let mut q = p.clone();
            f(q.admission.worker_step_budget.as_mut().unwrap());
            q.validate()
        };
        let msg = |r: Result<(), ConfigError>| r.unwrap_err().to_string();
        assert!(msg(with(&|c| c.classes = vec!["nope".into()])).contains("not one of the classes"));
        assert!(msg(with(&|c| c.classes = vec!["batch".into()])).contains("no service_level"));
        assert!(msg(with(&|c| c.classes.clear())).contains("at least one class"));
        for bad in [0.0, -0.5, 1.01, f64::NAN] {
            assert!(msg(with(&|c| c.margin = bad)).contains("margin"), "{bad}");
        }
        with(&|c| c.margin = 1.0).unwrap();
        let mut no_forecast = p.clone();
        no_forecast.forecast.enabled = false;
        assert!(msg(no_forecast.validate()).contains("forecast.enabled"));
    }

    #[test]
    fn the_log_writes_refused_pairs_once_a_minute_and_counts_per_minute() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wsb.jsonl");
        let mut p = policy(Some(&["priority"]));
        p.admission.worker_step_budget.as_mut().unwrap().margin = 1.0;
        let budget = StepBudget {
            log: Some(Mutex::new(WsbLog::new(path.clone()))),
            ..StepBudget::from_policy(&p).unwrap()
        };
        let verdict = Verdict {
            checks: vec![
                Check {
                    worker: w(1),
                    new_tokens: 10,
                    itls: vec![("priority".into(), 0.1)],
                    ratio: 2.0,
                },
                Check {
                    worker: w(2),
                    new_tokens: 10,
                    itls: vec![],
                    ratio: 0.0,
                },
            ],
        };
        let t = Time::from_secs_f64;
        let go = GateOutcome::Go { chosen: Some(w(2)) };
        budget.record_gate(t(1.0), "r\"1", "standard", &verdict, go);
        budget.record_gate(t(2.0), "r\"1", "standard", &verdict, go);
        budget.record_route(t(3.0), true, 1);
        budget.record_gate(t(61.0), "r\"1", "standard", &verdict, GateOutcome::Held);
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3, "{text}");
        assert!(lines[0].starts_with(r#"{"t":1.000000,"ev":"refused","request":"r\"1","worker":"1:0","class":"standard","projected_itl":{"priority":0.100000},"limit":{"priority":0.050000},"ratio":2.000000,"decision":"go","chosen":"2:0","fitting":["2:0"]}"#), "{}", lines[0]);
        assert!(lines[1].contains(r#""ev":"minute","minute":0,"checks":2,"held":0,"go_clean":0,"go_steered":2,"overdue":0,"refused_pairs":2,"route_ruled":1,"route_kept":1,"route_excluded":1,"suppressed":0"#), "{}", lines[1]);
        assert!(
            lines[2].contains(r#""decision":"held","chosen":null"#),
            "{}",
            lines[2]
        );
    }
}
