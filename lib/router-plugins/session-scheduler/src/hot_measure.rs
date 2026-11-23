// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! f49k `admission.hot_budget: measured`: the hot budget's use from the engines' device KV
//! instead of the session table's bookings. Default `booked` (the table, as before).
//!
//! Hot use = the sum over workers of `min(measured, 0.95 x the worker's HBM tokens)`, plus the
//! input tokens of the turns the gate admitted that no measurement shows yet:
//! - measured, per worker: its engine's gauges (`ClassifyEvent::EngineGauge`, offline replay with
//!   `DYNAMO_REPLAY_ENGINE_GAUGES=1`), `active_blocks x block_size + waiting_tokens`; a worker
//!   with no gauge yet, the router's LOAD row, `active_prefill_tokens + active_decode_blocks x
//!   block_size`;
//! - a turn counts its input tokens from the host taking it (`Scheduler::on_dispatched`) until the
//!   first refresh of the worker it was sent to after it was sent (or until it completes or
//!   aborts), so it is counted in the gap between the gate's `Go` and the measurement showing it.
//!
//! A session between turns holds no running request, so its KV (prefix cache in HBM or offloaded
//! to DRAM) counts nothing. Before the first measurement the booked hot use stands.
//!
//! The queue-0 guard: while a worker's engine waiting queue is empty (gauges: `waiting_requests`
//! and `waiting_tokens` 0; LOAD rows: no active prefill tokens), the gate admits at most one new
//! session the hot budget holds per refresh of that worker (a new gauge, or a changed LOAD row
//! for a worker without gauges), even though the measured budget says no: an engine with room is
//! not left idle by an over-count, and a burst cannot all pass before the measurement catches up.
//! Only a session held by the hot budget alone (its total budget fits) with no session of another
//! class ahead of it in the lobby.
//!
//! The signal (`engine_gauges` when `DYNAMO_REPLAY_ENGINE_GAUGES=1`, else `load_rows`) is written
//! once to stderr when the scheduler is built.

use std::collections::{BTreeMap, HashMap};

use dynamo_kv_router::plugins::prelude::WorkerWithDpRank;

use crate::config::ServicePolicy;

/// The share of a worker's HBM tokens its measured use counts at most.
pub const WORKER_CAP_FRAC: f64 = 0.95;

/// The env var with which offline replay sends the engines' gauges to the classifier.
pub const GAUGES_ENV: &str = "DYNAMO_REPLAY_ENGINE_GAUGES";

/// Most admitted turns waiting to be measured; the table is cleared when it fills.
const MAX_PENDING: usize = 65_536;

/// Which signal measures the hot use in this run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Signal {
    EngineGauges,
    LoadRows,
}

impl Signal {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EngineGauges => "engine_gauges",
            Self::LoadRows => "load_rows",
        }
    }
}

/// One worker's measurement, in tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reading {
    /// Device KV in use, the waiting queue's reservation included.
    pub used_tokens: u64,
    /// The worker's HBM KV tokens (0: unknown, no cap).
    pub hbm_tokens: u64,
    /// The engine's waiting queue is empty.
    pub queue_empty: bool,
}

impl Reading {
    /// From an engine gauge.
    pub fn from_gauge(
        active_blocks: u64,
        waiting_requests: u64,
        waiting_tokens: u64,
        total_blocks: u64,
        block_size: u64,
    ) -> Self {
        Self {
            used_tokens: active_blocks
                .saturating_mul(block_size)
                .saturating_add(waiting_tokens),
            hbm_tokens: total_blocks.saturating_mul(block_size),
            queue_empty: waiting_requests == 0 && waiting_tokens == 0,
        }
    }

    /// From the router's LOAD row: active prefill tokens plus the active decode blocks' tokens.
    pub fn from_load(
        active_prefill_tokens: u64,
        active_decode_blocks: u64,
        total_blocks: u64,
        block_size: u64,
    ) -> Self {
        Self {
            used_tokens: active_prefill_tokens
                .saturating_add(active_decode_blocks.saturating_mul(block_size)),
            hbm_tokens: total_blocks.saturating_mul(block_size),
            queue_empty: active_prefill_tokens == 0,
        }
    }

    /// The use counted: at most [`WORKER_CAP_FRAC`] of the worker's HBM tokens.
    pub fn counted(&self) -> u64 {
        if self.hbm_tokens == 0 {
            self.used_tokens
        } else {
            self.used_tokens
                .min((WORKER_CAP_FRAC * self.hbm_tokens as f64) as u64)
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct WorkerState {
    gauge: Option<Reading>,
    load: Option<Reading>,
    /// Refreshes of the reading in use (gauges once any came, else changed LOAD rows).
    refresh: u64,
    /// The refresh whose guard admission was used.
    probe_used: Option<u64>,
}

impl WorkerState {
    fn reading(&self) -> Option<Reading> {
        self.gauge.or(self.load)
    }
}

#[derive(Clone, Copy, Debug)]
struct Pending {
    tokens: u64,
    /// The worker it was sent to and that worker's refresh count then.
    sent: Option<(WorkerWithDpRank, u64)>,
}

/// The measured hot use (`admission.hot_budget: measured`).
#[derive(Debug)]
pub struct HotMeasure {
    signal: Signal,
    workers: BTreeMap<WorkerWithDpRank, WorkerState>,
    pending: HashMap<String, Pending>,
    probes: u64,
}

impl HotMeasure {
    /// On with `admission.hot_budget: measured`; writes the run's signal once.
    pub fn from_policy(policy: &ServicePolicy) -> Option<Self> {
        policy.admission.hot_budget.is_measured().then(|| {
            let gauges = std::env::var(GAUGES_ENV).is_ok_and(|v| v == "1");
            let measure = Self::new(if gauges {
                Signal::EngineGauges
            } else {
                Signal::LoadRows
            });
            eprintln!(
                "f49k admission.hot_budget: measured, signal {} (LOAD rows for a worker until its first gauge)",
                measure.signal.as_str()
            );
            measure
        })
    }

    pub fn new(signal: Signal) -> Self {
        Self {
            signal,
            workers: BTreeMap::new(),
            pending: HashMap::new(),
            probes: 0,
        }
    }

    /// The signal this run measures with.
    pub fn signal(&self) -> Signal {
        self.signal
    }

    /// An engine gauge of `worker`: a refresh. Returns whether its room may have grown (its use
    /// fell or its queue is empty).
    pub fn gauge(&mut self, worker: WorkerWithDpRank, reading: Reading) -> bool {
        let w = self.workers.entry(worker).or_default();
        let before = w.reading();
        w.gauge = Some(reading);
        w.refresh += 1;
        let refresh = w.refresh;
        self.settle(worker, refresh);
        reading.queue_empty || before.is_none_or(|b| reading.used_tokens < b.used_tokens)
    }

    /// The router's LOAD rows: a changed row of a worker without gauges is a refresh. Returns
    /// whether a worker's room may have grown.
    pub fn load_rows(&mut self, rows: &[(WorkerWithDpRank, Reading)]) -> bool {
        let mut grew = false;
        for (worker, reading) in rows {
            let w = self.workers.entry(*worker).or_default();
            let before = w.load;
            w.load = Some(*reading);
            if before == Some(*reading) || w.gauge.is_some() {
                continue;
            }
            w.refresh += 1;
            let refresh = w.refresh;
            grew |=
                reading.queue_empty || before.is_none_or(|b| reading.used_tokens < b.used_tokens);
            self.settle(*worker, refresh);
        }
        grew
    }

    /// Forget the turns sent to `worker` before its refresh `refresh`: the measurement shows them.
    fn settle(&mut self, worker: WorkerWithDpRank, refresh: u64) {
        self.pending
            .retain(|_, p| !p.sent.is_some_and(|(w, at)| w == worker && at < refresh));
    }

    /// The host took `request` (its turn admitted): its `tokens` count until measured.
    pub fn admitted(&mut self, request: &str, tokens: u64) {
        if self.pending.len() >= MAX_PENDING && !self.pending.contains_key(request) {
            self.pending.clear();
        }
        self.pending
            .insert(request.to_owned(), Pending { tokens, sent: None });
    }

    /// `request` was sent to `worker`: it counts until that worker's next refresh.
    pub fn sent(&mut self, request: &str, worker: WorkerWithDpRank) {
        let refresh = self.workers.get(&worker).map_or(0, |w| w.refresh);
        if let Some(p) = self.pending.get_mut(request) {
            p.sent = Some((worker, refresh));
        }
    }

    /// `request` completed or aborted.
    pub fn finished(&mut self, request: &str) {
        self.pending.remove(request);
    }

    /// Tokens of the admitted turns no measurement shows yet.
    pub fn pending_tokens(&self) -> u64 {
        self.pending.values().map(|p| p.tokens).sum()
    }

    /// The measured hot use; `None` before any worker is measured.
    pub fn hot_tokens(&self) -> Option<u64> {
        let mut measured = None::<u64>;
        for r in self.workers.values().filter_map(WorkerState::reading) {
            measured = Some(measured.unwrap_or(0).saturating_add(r.counted()));
        }
        measured.map(|m| m.saturating_add(self.pending_tokens()))
    }

    /// The worker whose queue-0 guard admission is free now (its queue empty and its admission
    /// since its last refresh unused): the least used, then the lowest id.
    pub fn probe(&self) -> Option<WorkerWithDpRank> {
        self.workers
            .iter()
            .filter_map(|(id, w)| {
                let r = w.reading()?;
                (r.queue_empty && w.probe_used != Some(w.refresh)).then_some((r.counted(), *id))
            })
            .min()
            .map(|(_, id)| id)
    }

    /// Use `worker`'s guard admission until its next refresh.
    pub fn use_probe(&mut self, worker: WorkerWithDpRank) {
        if let Some(w) = self.workers.get_mut(&worker) {
            w.probe_used = Some(w.refresh);
            self.probes += 1;
        }
    }

    /// Admissions by the queue-0 guard so far.
    pub fn probes(&self) -> u64 {
        self.probes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admit::{AdmitReason, Decision, GateAnswer, HoldReason, KvBudget};
    use crate::clock::Time;
    use crate::config::{ConfigError, HotBudget, HotDemand, Mode};
    use crate::scheduler::{RequestAttributes, Scheduler};
    use crate::session::Signals;

    fn t(s: f64) -> Time {
        Time::from_secs_f64(s)
    }

    fn w(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(id, 0)
    }

    fn policy(f: impl FnOnce(&mut ServicePolicy)) -> ServicePolicy {
        let mut p = ServicePolicy {
            mode: Mode::Enforce,
            ..Default::default()
        };
        f(&mut p);
        p
    }

    fn measured() -> Scheduler {
        Scheduler::new(policy(|p| p.admission.hot_budget = HotBudget::Measured))
    }

    fn row(used: u64, hbm: u64, queue_empty: bool) -> Reading {
        Reading {
            used_tokens: used,
            hbm_tokens: hbm,
            queue_empty,
        }
    }

    fn last_decision(s: &Scheduler, session: &str) -> Decision {
        s.decisions()
            .entries()
            .filter(|e| e.session == session)
            .last()
            .expect("logged")
            .decision
            .clone()
    }

    #[test]
    fn f49k_class_overcommit_lifts_only_the_classes_below_the_top() {
        let mut p = policy(|p| {
            p.admission
                .class_overcommit
                .extend([("standard".to_owned(), 1.3), ("batch".to_owned(), 1.6)]);
        });
        p.validate().unwrap();
        // f49k2: f49k's caps are `class_overcommit_mode: legacy` (the default `max` lifts
        // priority to batch's 1200).
        p.admission.class_overcommit_mode = Some(crate::config::ClassOvercommitMode::Legacy);
        let kv = KvBudget {
            hbm_tokens: 1000,
            dram_tokens: 100_000,
            ..Default::default()
        };
        let room = |p: &ServicePolicy, c: &str| crate::admit::room(p, p.class(Some(c)), &kv);
        // Priority (the top rank) keeps 1.0 x 1000; standard 0.9 x 1.3; batch 0.75 x 1.6.
        assert_eq!(room(&p, "priority"), 1000);
        assert!(room(&p, "standard").abs_diff(1170) <= 1);
        assert!(room(&p, "batch").abs_diff(1200) <= 1);
        // The step target's scale composes with it (room_scaled is the gate's fit's room).
        let scaled = crate::admit::room_scaled(&p, p.class(Some("standard")), &kv, 0.5);
        assert!(scaled.abs_diff(585) <= 1);
        // A top-ranked entry set past validation is still ignored.
        p.admission.class_overcommit.insert("priority".into(), 2.0);
        assert_eq!(room(&p, "priority"), 1000);
        // The gate's fit: standard with 1000 hot fits 100 more only with the overcommit.
        let mut s = Scheduler::new(policy(|p| {
            p.admission.class_overcommit.insert("standard".into(), 1.3);
        }));
        let mut off = Scheduler::new(policy(|_| {}));
        for s in [&mut s, &mut off] {
            s.on_classify(Some("r1"), "a", None, Some("priority"), 1000, t(0.0));
            s.on_dispatched("r1", "a", t(0.0));
            s.on_classify(Some("r2"), "b", None, Some("standard"), 100, t(1.0));
        }
        let kv = s.kv_budget(1000, 100_000);
        assert_eq!(s.gate_check("b", t(1.0), &kv), GateAnswer::Go);
        assert!(matches!(
            off.gate_check("b", t(1.0), &kv),
            GateAnswer::Hold { .. }
        ));
    }

    #[test]
    fn f49k_validation_and_parsing() {
        let ok = policy(|p| {
            p.admission.class_overcommit.insert("standard".into(), 1.3);
            p.admission.class_overcommit.insert("batch".into(), 0.5);
        });
        assert!(ok.validate().is_ok());
        for (class, value) in [
            ("priority", 1.3),
            ("standard", 0.0),
            ("standard", -1.0),
            ("standard", f64::NAN),
            ("standard", f64::INFINITY),
            ("nobody", 1.3),
        ] {
            let bad = policy(|p| {
                p.admission.class_overcommit.insert(class.into(), value);
            });
            assert!(
                matches!(bad.validate(), Err(ConfigError::ClassOvercommit(_))),
                "{class} {value}"
            );
        }
        let parsed: ServicePolicy = serde_json::from_str(
            r#"{"admission": {"hot_budget": "measured",
                              "class_overcommit": {"standard": 1.3, "batch": 1.6}}}"#,
        )
        .unwrap();
        parsed.validate().unwrap();
        assert_eq!(parsed.admission.hot_budget, HotBudget::Measured);
        assert_eq!(parsed.admission.class_overcommit["batch"], 1.6);
        for (text, want) in [
            ("true", HotBudget::Booked),
            ("false", HotBudget::Off),
            ("\"booked\"", HotBudget::Booked),
            ("\"off\"", HotBudget::Off),
        ] {
            let p: ServicePolicy =
                serde_json::from_str(&format!(r#"{{"admission": {{"hot_budget": {text}}}}}"#))
                    .unwrap();
            assert_eq!(p.admission.hot_budget, want, "{text}");
        }
        assert!(
            serde_json::from_str::<ServicePolicy>(r#"{"admission": {"hot_budget": "guess"}}"#)
                .is_err()
        );
        // Off by default, and the default serializes as before: `hot_budget: true`, no new key.
        let json = serde_json::to_string(&ServicePolicy::default()).unwrap();
        assert!(json.contains("\"hot_budget\":true"));
        assert!(!json.contains("class_overcommit"));
        let back: ServicePolicy =
            serde_json::from_str(&serde_json::to_string(&parsed).unwrap()).unwrap();
        assert_eq!(back, parsed);
    }

    #[test]
    fn f49k_measured_hot_excludes_idle_sessions_and_caps_each_worker() {
        let think = RequestAttributes {
            signals: Signals {
                next_think_s: Some(10.0),
                osl_tokens: Some(100),
                ..Default::default()
            },
            ..Default::default()
        };
        let run = |hot_budget: HotBudget| {
            let mut s = Scheduler::new(policy(|p| {
                p.admission.hot_demand = HotDemand::NextTurn;
                p.admission.hot_budget = hot_budget;
            }));
            s.on_classify_with(Some("r1"), "a", None, None, &think, 600, t(0.0));
            s.on_dispatched("r1", "a", t(0.0));
            s.on_completed_with_output("r1", Some(700), Some(100), t(5.0));
            s
        };
        // Booked (`next_turn`): the idle session counts its duty cycle of its next turn.
        let booked = run(HotBudget::Booked);
        assert!(booked.kv_budget(1000, 0).hot_tokens > 0);
        // Measured: before any measurement the booked use stands; then the engines' (none here).
        let mut m = run(HotBudget::Measured);
        assert_eq!(
            m.kv_budget(1000, 0).hot_tokens,
            booked.kv_budget(1000, 0).hot_tokens
        );
        m.hot_load_rows(&[(w(1), row(0, 500, true)), (w(2), row(0, 500, true))]);
        assert_eq!(m.kv_budget(1000, 0).hot_tokens, 0);
        // Each worker counts at most 0.95 of its HBM.
        m.hot_load_rows(&[(w(1), row(900, 500, false)), (w(2), row(100, 500, false))]);
        assert_eq!(m.kv_budget(1000, 0).hot_tokens, 475 + 100);
    }

    #[test]
    fn f49k_admitted_turns_count_until_their_worker_refreshes() {
        let mut s = measured();
        s.hot_load_rows(&[(w(1), row(100, 1000, false))]);
        s.on_classify(Some("r1"), "a", None, Some("standard"), 300, t(0.0));
        s.on_dispatched("r1", "a", t(0.0));
        assert_eq!(s.kv_budget(1000, 0).hot_tokens, 400);
        s.hot_measure_sent("r1", w(1));
        // An unchanged row is no refresh: still counted.
        s.hot_load_rows(&[(w(1), row(100, 1000, false))]);
        assert_eq!(s.kv_budget(1000, 0).hot_tokens, 400);
        // The next refresh shows it.
        s.hot_load_rows(&[(w(1), row(400, 1000, false))]);
        assert_eq!(s.kv_budget(1000, 0).hot_tokens, 400);
        // A completion forgets a turn not yet shown.
        s.on_classify(Some("r2"), "b", None, Some("standard"), 50, t(1.0));
        s.on_dispatched("r2", "b", t(1.0));
        assert_eq!(s.kv_budget(1000, 0).hot_tokens, 450);
        s.on_completed("r2", Some(60), t(2.0));
        assert_eq!(s.kv_budget(1000, 0).hot_tokens, 400);
    }

    #[test]
    fn f49k_queue_zero_guard_admits_one_per_refresh() {
        let mut s = measured();
        // One worker over standard's 900 hot budget (counted 950), its queue empty.
        s.hot_load_rows(&[(w(1), row(990, 1000, true))]);
        for (i, session) in ["b", "c", "d"].into_iter().enumerate() {
            let request = format!("r-{session}");
            s.on_classify(
                Some(&request),
                session,
                None,
                Some("standard"),
                100,
                t(i as f64),
            );
        }
        let kv = |s: &Scheduler| s.kv_budget(1000, 100_000);
        // b goes on the guard; c, next in line, waits: one per refresh.
        let k = kv(&s);
        assert_eq!(s.gate_check("b", t(3.0), &k), GateAnswer::Go);
        s.on_dispatched("r-b", "b", t(3.0));
        let k = kv(&s);
        assert!(matches!(
            s.gate_check("c", t(3.0), &k),
            GateAnswer::Hold { .. }
        ));
        assert!(matches!(
            last_decision(&s, "c"),
            Decision::Hold {
                reason: HoldReason::KvBudget { .. },
                ..
            }
        ));
        // Asking again without a refresh changes nothing.
        let k = kv(&s);
        assert!(matches!(
            s.gate_check("c", t(4.0), &k),
            GateAnswer::Hold { .. }
        ));
        // A refresh (a changed row) frees one more admission: c, and not d.
        s.hot_load_rows(&[(w(1), row(980, 1000, true))]);
        let k = kv(&s);
        assert_eq!(s.gate_check("c", t(5.0), &k), GateAnswer::Go);
        s.on_dispatched("r-c", "c", t(5.0));
        let k = kv(&s);
        assert!(matches!(
            s.gate_check("d", t(5.0), &k),
            GateAnswer::Hold { .. }
        ));
        assert_eq!(s.hot_measure().unwrap().probes(), 2);
        // A worker whose queue is not empty gives none.
        s.hot_load_rows(&[(w(1), row(980, 1000, false))]);
        let k = kv(&s);
        assert!(matches!(
            s.gate_check("d", t(6.0), &k),
            GateAnswer::Hold { .. }
        ));
        // With gauges, a changed LOAD row of that worker is no refresh; a new gauge is.
        s.hot_gauge(w(1), row(985, 1000, true));
        let k = kv(&s);
        assert_eq!(s.gate_check("d", t(7.0), &k), GateAnswer::Go);
        s.on_dispatched("r-d", "d", t(7.0));
        s.on_classify(Some("r-e"), "e", None, Some("standard"), 100, t(7.0));
        s.hot_load_rows(&[(w(1), row(970, 1000, true))]);
        let k = kv(&s);
        assert!(matches!(
            s.gate_check("e", t(8.0), &k),
            GateAnswer::Hold { .. }
        ));
        s.hot_gauge(w(1), row(985, 1000, true));
        let k = kv(&s);
        assert_eq!(s.gate_check("e", t(9.0), &k), GateAnswer::Go);
    }

    #[test]
    fn f49k_measured_charges_context_plus_output() {
        let osl = |n: u32| RequestAttributes {
            signals: Signals {
                osl_tokens: Some(n),
                ..Default::default()
            },
            ..Default::default()
        };
        let check = |hot_budget: HotBudget, out: u32| {
            let mut s = Scheduler::new(policy(|p| p.admission.hot_budget = hot_budget));
            // 300 measured (queue not empty: no guard): standard's room is 900 - 300 = 600.
            s.hot_load_rows(&[(w(1), row(300, 1000, false))]);
            s.on_classify_with(
                Some("r1"),
                "a",
                None,
                Some("standard"),
                &osl(out),
                400,
                t(0.0),
            );
            let kv = KvBudget {
                hot_tokens: 300,
                ..s.kv_budget(1000, 100_000)
            };
            let answer = s.gate_check("a", t(1.0), &kv);
            (answer, last_decision(&s, "a"))
        };
        // Measured: 400 + 300 = 700 over 600 waits; 400 + 100 = 500 fits.
        let (answer, decision) = check(HotBudget::Measured, 300);
        assert!(matches!(answer, GateAnswer::Hold { .. }));
        assert!(matches!(
            decision,
            Decision::Hold {
                reason: HoldReason::KvBudget { needed: 700, .. },
                ..
            }
        ));
        assert_eq!(
            check(HotBudget::Measured, 100).1,
            Decision::Admit(AdmitReason::Fits)
        );
        // Booked: the input alone (400) as before.
        assert_eq!(
            check(HotBudget::Booked, 300).1,
            Decision::Admit(AdmitReason::Fits)
        );
    }

    #[test]
    fn f49k_falls_back_to_load_rows_without_gauges() {
        let mut m = HotMeasure::new(Signal::EngineGauges);
        assert_eq!(m.hot_tokens(), None);
        m.load_rows(&[(w(1), row(200, 1000, false)), (w(2), row(300, 1000, false))]);
        assert_eq!(m.hot_tokens(), Some(500));
        // Worker 1's gauge replaces its LOAD row; worker 2 stays on its row.
        m.gauge(w(1), Reading::from_gauge(10, 1, 64, 100, 16));
        assert_eq!(m.hot_tokens(), Some(160 + 64 + 300));
        assert_eq!(Reading::from_gauge(10, 0, 0, 100, 16), row(160, 1600, true));
        assert_eq!(Reading::from_load(32, 2, 100, 16), row(64, 1600, false));
        // The scheduler: measured without gauges uses the rows.
        let mut s = measured();
        s.hot_load_rows(&[(w(1), Reading::from_load(0, 10, 100, 16))]);
        assert_eq!(s.kv_budget(1600, 0).hot_tokens, 160);
        assert_eq!(s.hot_measure().unwrap().signal(), Signal::LoadRows);
    }

    #[test]
    fn f49k_options_off_change_nothing() {
        // Three schedulers on one event sequence: options absent; absent with the measure feeds
        // called anyway (no-ops when off); `class_overcommit` of 1.0 for the lower classes.
        let mut plain = Scheduler::new(policy(|_| {}));
        let mut fed = Scheduler::new(policy(|_| {}));
        let mut ones = Scheduler::new(policy(|p| {
            p.admission.class_overcommit.insert("standard".into(), 1.0);
            p.admission.class_overcommit.insert("batch".into(), 1.0);
        }));
        assert!(!fed.hot_measured_on());
        let classes = ["priority", "standard", "batch"];
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        // Requests classified and not yet checked; requests dispatched and not yet completed.
        let mut queued: Vec<(String, String)> = Vec::new();
        let mut running: Vec<String> = Vec::new();
        for step in 0..600u64 {
            let now = t(step as f64 * 0.5);
            let action = next(3);
            if action == 0 || queued.is_empty() {
                let session = format!("s{}", next(30));
                let request = format!("r{step}");
                let class = classes[next(3) as usize];
                let isl = 100 + next(900);
                for s in [&mut plain, &mut fed, &mut ones] {
                    s.on_classify(Some(&request), &session, None, Some(class), isl, now);
                }
                queued.push((request, session));
            } else if action == 1 {
                let (request, session) = queued.remove(next(queued.len() as u64) as usize);
                let mut each = Vec::new();
                for s in [&mut plain, &mut fed, &mut ones] {
                    let kv = s.kv_budget(4000, 8000);
                    let answer = s.gate_check(&session, now, &kv);
                    if answer == GateAnswer::Go {
                        s.on_dispatched(&request, &session, now);
                    }
                    each.push(answer);
                }
                assert!(each.windows(2).all(|p| p[0] == p[1]), "{each:?}");
                if each[0] == GateAnswer::Go {
                    running.push(request);
                } else {
                    queued.push((request, session));
                }
            } else if !running.is_empty() {
                let request = running.remove(next(running.len() as u64) as usize);
                let context = 200 + next(2000) as usize;
                for s in [&mut plain, &mut fed, &mut ones] {
                    s.on_completed(&request, Some(context), now);
                }
            }
            fed.hot_load_rows(&[(w(1), row(step * 7, 4000, true))]);
            fed.hot_gauge(w(1), row(step, 4000, true));
            if let Some(request) = running.last() {
                fed.hot_measure_sent(request, w(1));
            }
        }
        let log = |s: &Scheduler| s.decisions().entries().cloned().collect::<Vec<_>>();
        assert!(!log(&plain).is_empty());
        assert_eq!(log(&plain), log(&fed));
        assert_eq!(log(&plain), log(&ones));
        assert_eq!(plain.kv_budget(4000, 8000), fed.kv_budget(4000, 8000));
    }
}
