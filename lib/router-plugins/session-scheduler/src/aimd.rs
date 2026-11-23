// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! f49h `admission.aimd`: a per-worker limit on one class's active turns (priority by default),
//! moved by additive increase and multiplicative decrease on each worker's measured service.
//!
//! Each worker has a limit `L`, starting at `start`. Every `interval_s`, a worker whose turns of the
//! class had a TTFT p90 under `slack_frac` times the class's TTFT limit and a mean ITL under
//! `slack_frac` times its ITL limit gets `L + 1` (at most `max`); one whose TTFT p90 or mean ITL
//! passed the limit itself gets `max(min, L / 2)`; otherwise, or with no samples, `L` stays.
//!
//! The samples are each worker's own, from the lifecycle events the host reports with the worker
//! (`Sent`, `Responding`, `Completed`): TTFT from the turn's dispatch to the worker to its first
//! token (the router wait is left out, so a hold here never feeds back into `L`), and mean ITL from
//! the first token to completion over output tokens minus one.
//!
//! A turn the gate would let go is bound for its session's last worker (from `Sent`), or, for a
//! session with none yet, for the worker with the most room under its limit. It goes while that
//! room exceeds the turns of the same binding waiting ahead of it, ordered by session start (the
//! root session's first offer): the newest sessions' turns wait first, so sessions already mid-run
//! go first and stay whole. Nothing is reserved and no other class is touched. In enforce mode a
//! turn of the class gets, at classification, a deadline `wait_bound_s` after its arrival (the
//! router expires it there) and a Retry-After of its place among the waiting turns over the fleet's
//! measured departure rate (turns of the class finishing per second), within
//! `admission.retry_after_estimate`'s `min_s` and `max_s` when set.
//!
//! `DYNAMO_AIMD_LOG=<path>`: one JSON line per worker per interval (limit, TTFT p90, mean ITL,
//! sample counts, active and waiting turns) and one for the fleet. Nothing is written when unset.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::Write;
use std::time::Duration;

use dynamo_kv_router::plugins::prelude::*;
use serde::{Deserialize, Serialize};

use crate::clock::Time;
use crate::config::{ConfigError, ServicePolicy};
use crate::metrics::percentile;

/// The env var naming the AIMD diagnostics log.
pub const AIMD_LOG_ENV: &str = "DYNAMO_AIMD_LOG";

/// Most sessions whose last worker is remembered; the table is cleared when it fills.
const MAX_HOMES: usize = 65_536;

/// Longest Retry-After when `admission.retry_after_estimate` sets no `max_s`, seconds.
const DEFAULT_MAX_RETRY_S: f64 = 300.0;

/// Shortest Retry-After when `admission.retry_after_estimate` sets no `min_s`, seconds.
const DEFAULT_MIN_RETRY_S: f64 = 1.0;

/// `admission.aimd` (f49h). Off (absent) by default.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AimdConfig {
    /// The class whose turns are limited per worker. Default `priority`.
    pub class: String,
    /// Seconds between limit updates. Default 30.
    pub interval_s: f64,
    /// Each worker's limit at the start. Default 8.
    pub start: u32,
    /// Lowest limit. Default 1.
    pub min: u32,
    /// Highest limit. Default 64.
    pub max: u32,
    /// The share of the class's TTFT and ITL limits under which the limit grows. Default 0.8.
    pub slack_frac: f64,
    /// Longest a turn of the class waits before it is refused, seconds. Default: the class's
    /// `max_lobby_wait_s`.
    pub wait_bound_s: Option<f64>,
}

impl Default for AimdConfig {
    fn default() -> Self {
        Self {
            class: "priority".to_owned(),
            interval_s: 30.0,
            start: 8,
            min: 1,
            max: 64,
            slack_frac: 0.8,
            wait_bound_s: None,
        }
    }
}

impl AimdConfig {
    pub fn validate(&self, policy: &ServicePolicy) -> Result<(), ConfigError> {
        let class = policy
            .classes
            .iter()
            .find(|c| c.name == self.class)
            .ok_or_else(|| ConfigError::AimdClass {
                name: self.class.clone(),
                reason: "is not one of the classes",
            })?;
        if class.service_level.is_none() {
            return Err(ConfigError::AimdClass {
                name: self.class.clone(),
                reason: "has no service_level (its TTFT and ITL limits)",
            });
        }
        let wait_bound = self.wait_bound_s.unwrap_or(1.0);
        for (field, expected, value, ok) in [
            (
                "admission.aimd.interval_s",
                "positive",
                self.interval_s,
                self.interval_s.is_finite() && self.interval_s > 0.0,
            ),
            (
                "admission.aimd.min",
                "at least 1",
                f64::from(self.min),
                self.min >= 1,
            ),
            (
                "admission.aimd.start",
                "within [min, max]",
                f64::from(self.start),
                self.min <= self.start && self.start <= self.max,
            ),
            (
                "admission.aimd.slack_frac",
                "in (0, 1]",
                self.slack_frac,
                self.slack_frac > 0.0 && self.slack_frac <= 1.0,
            ),
            (
                "admission.aimd.wait_bound_s",
                "positive",
                wait_bound,
                wait_bound.is_finite() && wait_bound > 0.0,
            ),
        ] {
            if !ok {
                return Err(ConfigError::PolicyOutOfRange {
                    field,
                    expected,
                    value,
                });
            }
        }
        Ok(())
    }
}

/// The gate's answer for a turn of the AIMD class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AimdAnswer {
    /// Not a tracked turn (another class, or already dispatched): the gate answers as before.
    Untracked,
    /// Go, to this worker when one is known.
    Go(Option<WorkerWithDpRank>),
    /// Wait; ask again by this time at the latest.
    Hold(Time),
}

#[derive(Debug)]
struct WorkerState {
    limit: u32,
    active: u32,
    ttft: Vec<f64>,
    itl: Vec<f64>,
}

impl WorkerState {
    fn new(limit: u32) -> Self {
        Self {
            limit,
            active: 0,
            ttft: Vec::new(),
            itl: Vec::new(),
        }
    }

    fn room(&self) -> u32 {
        self.limit.saturating_sub(self.active)
    }
}

#[derive(Debug)]
struct Turn {
    session: String,
    /// The session start the waiting order reads.
    start: Time,
    arrived: Time,
    worker: Option<WorkerWithDpRank>,
    sent_at: Option<Time>,
    first_token: Option<Time>,
}

#[derive(Debug)]
struct Waiter {
    start: Time,
    since: Time,
    bound: Option<WorkerWithDpRank>,
}

#[derive(Debug)]
struct RetryBounds {
    min_s: f64,
    max_s: f64,
    fallback_s: f64,
}

/// The AIMD state: per-worker limits and samples, the class's turns, and the waiting turns.
#[derive(Debug)]
pub struct Aimd {
    cfg: AimdConfig,
    ttft_limit_s: f64,
    itl_limit_s: f64,
    wait_bound_s: f64,
    retry: RetryBounds,
    workers: BTreeMap<WorkerWithDpRank, WorkerState>,
    turns: HashMap<String, Turn>,
    homes: HashMap<String, WorkerWithDpRank>,
    waiting: HashMap<String, Waiter>,
    departures: VecDeque<Time>,
    next_step: Option<Time>,
    log: Option<std::io::BufWriter<std::fs::File>>,
}

impl Aimd {
    /// The AIMD state for `policy`, or `None` when `admission.aimd` is absent (the default).
    pub fn from_policy(policy: &ServicePolicy) -> Option<Self> {
        let cfg = policy.admission.aimd.clone()?;
        let class = policy.classes.iter().find(|c| c.name == cfg.class)?;
        let level = class.service_level.as_ref()?;
        let wait_bound_s = cfg.wait_bound_s.unwrap_or(class.admission.max_lobby_wait_s);
        let estimate = policy.admission.retry_after_estimate.as_ref();
        let min_s = estimate
            .and_then(|e| e.min_s)
            .unwrap_or(DEFAULT_MIN_RETRY_S);
        let max_s = estimate
            .and_then(|e| e.max_s)
            .unwrap_or(DEFAULT_MAX_RETRY_S)
            .max(min_s);
        let fallback_s = estimate
            .and_then(|e| e.fallback_s)
            .or(policy.admission.retry_after_s)
            .unwrap_or(wait_bound_s);
        let log = std::env::var_os(AIMD_LOG_ENV)
            .and_then(|path| std::fs::File::create(path).ok())
            .map(std::io::BufWriter::new);
        Some(Self {
            ttft_limit_s: level.ttft_s,
            itl_limit_s: level.itl_s,
            wait_bound_s,
            retry: RetryBounds {
                min_s,
                max_s,
                fallback_s,
            },
            cfg,
            workers: BTreeMap::new(),
            turns: HashMap::new(),
            homes: HashMap::new(),
            waiting: HashMap::new(),
            departures: VecDeque::new(),
            next_step: None,
            log,
        })
    }

    /// Seconds a turn of the class may wait before it is refused.
    pub fn wait_bound_s(&self) -> f64 {
        self.wait_bound_s
    }

    /// A worker's current limit, once the worker is known.
    pub fn limit(&self, worker: WorkerWithDpRank) -> Option<u32> {
        self.workers.get(&worker).map(|w| w.limit)
    }

    /// A worker's active turns of the class.
    pub fn active(&self, worker: WorkerWithDpRank) -> u32 {
        self.workers.get(&worker).map_or(0, |w| w.active)
    }

    /// Turns of the class waiting at the gate.
    pub fn waiting(&self) -> usize {
        self.waiting.len()
    }

    /// A request of `class` was classified: track it when it is the AIMD class. `start` is its
    /// session's start. Returns whether it is tracked.
    pub fn arrived(
        &mut self,
        request_id: &str,
        session: &str,
        class: &str,
        start: Time,
        now: Time,
    ) -> bool {
        self.tick(now);
        if class != self.cfg.class {
            return false;
        }
        // A retry of the same request starts over.
        self.forget(request_id);
        self.turns.insert(
            request_id.to_owned(),
            Turn {
                session: session.to_owned(),
                start,
                arrived: now,
                worker: None,
                sent_at: None,
                first_token: None,
            },
        );
        true
    }

    /// Retry-After for a refused turn, seconds: its place among the waiting turns over the
    /// fleet's departure rate, within the bounds; the fallback while no departure is measured.
    pub fn retry_after_s(&self, request_id: &str, now: Time) -> f64 {
        let start = self.turns.get(request_id).map_or(now, |t| t.start);
        let ahead = self
            .waiting
            .iter()
            .filter(|(id, w)| {
                id.as_str() != request_id && (w.start, id.as_str()) < (start, request_id)
            })
            .count();
        let rate = self.departure_rate(now);
        let estimate = if rate > 0.0 {
            (ahead + 1) as f64 / rate
        } else {
            self.retry.fallback_s
        };
        estimate.clamp(self.retry.min_s, self.retry.max_s)
    }

    /// Turns of the class finishing per second on the fleet, over the trailing window.
    pub fn departure_rate(&self, now: Time) -> f64 {
        let window = self.rate_window_s();
        let n = self
            .departures
            .iter()
            .filter(|&&at| now.since(at).as_secs_f64() <= window)
            .count();
        n as f64 / window
    }

    fn rate_window_s(&self) -> f64 {
        (2.0 * self.cfg.interval_s).max(60.0)
    }

    /// The workers the host lists now: a new one starts at the configured limit; one no longer
    /// listed is forgotten once none of its turns is active.
    pub fn observe_workers(&mut self, workers: &[WorkerWithDpRank]) {
        if workers.is_empty() {
            return;
        }
        let initial = self.cfg.start;
        for &worker in workers {
            self.workers
                .entry(worker)
                .or_insert_with(|| WorkerState::new(initial));
        }
        self.workers
            .retain(|worker, state| state.active > 0 || workers.contains(worker));
    }

    /// The gate would let `request_id` go: whether it goes now, and where.
    pub fn check(&mut self, request_id: &str, session: &str, now: Time) -> AimdAnswer {
        self.tick(now);
        let Some(turn) = self.turns.get(request_id) else {
            return AimdAnswer::Untracked;
        };
        if turn.worker.is_some() {
            return AimdAnswer::Untracked;
        }
        let start = turn.start;
        let initial = self.cfg.start;
        let bound = self.homes.get(session).copied();
        let since = self.waiting.get(request_id).map_or(now, |w| w.since);
        self.waiting.insert(
            request_id.to_owned(),
            Waiter {
                start,
                since,
                bound,
            },
        );
        let ahead = |bound: Option<WorkerWithDpRank>| {
            self.waiting
                .iter()
                .filter(|(id, w)| w.bound == bound && (w.start, id.as_str()) < (start, request_id))
                .count()
        };
        let target = match bound {
            Some(home) => {
                let room = self.workers.get(&home).map_or(initial, WorkerState::room);
                (room as usize > ahead(Some(home))).then_some(home)
            }
            None => {
                if self.workers.is_empty() {
                    // Nothing known yet: no limit to apply.
                    return AimdAnswer::Go(None);
                }
                let room: usize = self.workers.values().map(|w| w.room() as usize).sum();
                if room > ahead(None) {
                    // The most room; the lowest worker on a tie (the map is ordered).
                    self.workers
                        .iter()
                        .filter(|(_, w)| w.room() > 0)
                        .max_by(|a, b| a.1.room().cmp(&b.1.room()).then_with(|| b.0.cmp(a.0)))
                        .map(|(&worker, _)| worker)
                } else {
                    None
                }
            }
        };
        match target {
            Some(worker) => AimdAnswer::Go(Some(worker)),
            None => {
                let next = self.next_step.unwrap_or(now);
                let deadline = since.after(Duration::from_secs_f64(self.wait_bound_s));
                let until = if deadline > now && deadline < next {
                    deadline
                } else {
                    next
                };
                AimdAnswer::Hold(until)
            }
        }
    }

    /// The host dispatched `request_id` toward `target`: it counts there until `Sent` says
    /// otherwise.
    pub fn dispatched(&mut self, request_id: &str, target: Option<WorkerWithDpRank>, now: Time) {
        self.tick(now);
        let Some(turn) = self.turns.get_mut(request_id) else {
            return;
        };
        self.waiting.remove(request_id);
        if turn.worker.is_some() {
            return;
        }
        if let Some(worker) = target {
            turn.worker = Some(worker);
            let initial = self.cfg.start;
            self.workers
                .entry(worker)
                .or_insert_with(|| WorkerState::new(initial))
                .active += 1;
        }
    }

    /// A lifecycle event of any request; only tracked turns change anything.
    pub fn on_event(&mut self, event: &ClassifyEvent, now: Time) {
        self.tick(now);
        match event {
            ClassifyEvent::Sent { request_id, worker } => self.sent(request_id, *worker, now),
            ClassifyEvent::Responding { request_id, .. } => {
                if let Some(turn) = self.turns.get_mut(request_id.as_str())
                    && turn.first_token.is_none()
                {
                    turn.first_token = Some(now);
                    if let (Some(worker), Some(sent)) = (turn.worker, turn.sent_at)
                        && let Some(state) = self.workers.get_mut(&worker)
                    {
                        state.ttft.push(now.since(sent).as_secs_f64());
                    }
                }
            }
            ClassifyEvent::Completed {
                request_id,
                output_tokens,
                ..
            } => {
                let Some(turn) = self.turns.remove(request_id.as_str()) else {
                    return;
                };
                self.waiting.remove(request_id.as_str());
                let Some(worker) = turn.worker else {
                    return;
                };
                if let Some(state) = self.workers.get_mut(&worker) {
                    state.active = state.active.saturating_sub(1);
                    if let (Some(first), Some(out)) = (turn.first_token, *output_tokens)
                        && out > 1
                    {
                        state
                            .itl
                            .push(now.since(first).as_secs_f64() / (out - 1) as f64);
                    }
                }
                if turn.sent_at.is_some() {
                    self.departures.push_back(now);
                    self.trim_departures(now);
                }
            }
            ClassifyEvent::Aborted { request_id, .. } => self.forget(request_id),
            _ => {}
        }
    }

    fn sent(&mut self, request_id: &str, worker: WorkerWithDpRank, now: Time) {
        let Some(turn) = self.turns.get_mut(request_id) else {
            return;
        };
        self.waiting.remove(request_id);
        let initial = self.cfg.start;
        if turn.worker != Some(worker) {
            if let Some(old) = turn.worker
                && let Some(state) = self.workers.get_mut(&old)
            {
                state.active = state.active.saturating_sub(1);
            }
            self.workers
                .entry(worker)
                .or_insert_with(|| WorkerState::new(initial))
                .active += 1;
            turn.worker = Some(worker);
        }
        turn.sent_at = Some(now);
        if self.homes.len() >= MAX_HOMES && !self.homes.contains_key(&turn.session) {
            self.homes.clear();
        }
        self.homes.insert(turn.session.clone(), worker);
    }

    /// Drop a turn (aborted, expired, or retried), freeing its worker's count.
    fn forget(&mut self, request_id: &str) {
        self.waiting.remove(request_id);
        if let Some(turn) = self.turns.remove(request_id)
            && let Some(worker) = turn.worker
            && let Some(state) = self.workers.get_mut(&worker)
        {
            state.active = state.active.saturating_sub(1);
        }
    }

    fn trim_departures(&mut self, now: Time) {
        let window = self.rate_window_s();
        while self
            .departures
            .front()
            .is_some_and(|&at| now.since(at).as_secs_f64() > window)
        {
            self.departures.pop_front();
        }
    }

    /// Update every worker's limit once per interval.
    pub fn tick(&mut self, now: Time) {
        let interval = Duration::from_secs_f64(self.cfg.interval_s);
        let Some(next) = self.next_step else {
            self.next_step = Some(now.after(interval));
            return;
        };
        if now < next {
            return;
        }
        self.prune(now);
        let (min, max) = (self.cfg.min, self.cfg.max);
        let slack = self.cfg.slack_frac;
        let (ttft_limit, itl_limit) = (self.ttft_limit_s, self.itl_limit_s);
        let mut waiting_on: BTreeMap<WorkerWithDpRank, usize> = BTreeMap::new();
        let mut unbound = 0usize;
        for waiter in self.waiting.values() {
            match waiter.bound {
                Some(worker) => *waiting_on.entry(worker).or_default() += 1,
                None => unbound += 1,
            }
        }
        let logging = self.log.is_some();
        let at = now.as_secs_f64();
        let mut lines = Vec::new();
        for (&worker, state) in &mut self.workers {
            let ttft_n = state.ttft.len();
            let itl_n = state.itl.len();
            let ttft_p90 = percentile(&mut state.ttft, 0.9);
            let itl_mean = (itl_n > 0).then(|| state.itl.iter().sum::<f64>() / itl_n as f64);
            let over =
                ttft_p90.is_some_and(|p| p > ttft_limit) || itl_mean.is_some_and(|m| m > itl_limit);
            let under = (ttft_p90.is_some() || itl_mean.is_some())
                && ttft_p90.is_none_or(|p| p < slack * ttft_limit)
                && itl_mean.is_none_or(|m| m < slack * itl_limit);
            let previous = state.limit;
            if over {
                state.limit = (state.limit / 2).max(min);
            } else if under {
                state.limit = state.limit.saturating_add(1).min(max);
            }
            if logging {
                lines.push(format!(
                    "{{\"t\":{at:.3},\"worker\":{},\"dp_rank\":{},\"limit\":{},\"previous\":{previous},\"ttft_p90_s\":{},\"itl_mean_s\":{},\"ttft_n\":{ttft_n},\"itl_n\":{itl_n},\"active\":{},\"waiting\":{}}}",
                    worker.worker_id,
                    worker.dp_rank,
                    state.limit,
                    json_f64(ttft_p90),
                    json_f64(itl_mean),
                    state.active,
                    waiting_on.get(&worker).copied().unwrap_or(0),
                ));
            }
            state.ttft.clear();
            state.itl.clear();
        }
        if let Some(log) = self.log.as_mut() {
            lines.push(format!(
                "{{\"t\":{at:.3},\"fleet\":true,\"waiting\":{},\"waiting_unbound\":{unbound},\"departures_per_s\":{:.6}}}",
                self.waiting.len(),
                {
                    let window = (2.0 * self.cfg.interval_s).max(60.0);
                    self.departures
                        .iter()
                        .filter(|&&d| now.since(d).as_secs_f64() <= window)
                        .count() as f64
                        / window
                },
            ));
            for line in &lines {
                let _ = writeln!(log, "{line}");
            }
            let _ = log.flush();
        }
        let after = next.after(interval);
        self.next_step = Some(if now >= after {
            now.after(interval)
        } else {
            after
        });
    }

    /// Forget what no event will end: waiters past their bound (the router has refused them) and
    /// turns never dispatched long after they arrived.
    fn prune(&mut self, now: Time) {
        let keep_waiter = self.wait_bound_s + self.cfg.interval_s;
        self.waiting
            .retain(|_, w| now.since(w.since).as_secs_f64() <= keep_waiter);
        let keep_turn = 4.0 * self.wait_bound_s + 2.0 * self.cfg.interval_s;
        self.turns
            .retain(|_, t| t.worker.is_some() || now.since(t.arrived).as_secs_f64() <= keep_turn);
        self.trim_departures(now);
    }
}

fn json_f64(value: Option<f64>) -> String {
    value.map_or_else(|| "null".to_owned(), |v| format!("{v:.6}"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::time::Instant;

    use super::*;
    use crate::clock::ManualClock;
    use crate::config::{Mode, RetryAfterEstimate};
    use crate::hooks::SessionSchedulerClassifier;

    fn t(seconds: f64) -> Time {
        Time::from_secs_f64(seconds)
    }

    fn w(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(id, 0)
    }

    fn policy(cfg: AimdConfig) -> ServicePolicy {
        let mut policy = ServicePolicy {
            mode: Mode::Enforce,
            ..Default::default()
        };
        policy.admission.aimd = Some(cfg);
        policy
    }

    fn aimd(cfg: AimdConfig) -> Aimd {
        Aimd::from_policy(&policy(cfg)).expect("aimd is configured")
    }

    /// One turn of `session` on `worker` from `at`: the given TTFT, then 11 tokens at `itl`.
    fn serve(a: &mut Aimd, id: &str, worker: WorkerWithDpRank, at: f64, ttft: f64, itl: f64) {
        assert!(a.arrived(id, id, "priority", t(at), t(at)));
        a.dispatched(id, Some(worker), t(at));
        a.on_event(
            &ClassifyEvent::Sent {
                request_id: id.into(),
                worker,
            },
            t(at),
        );
        a.on_event(
            &ClassifyEvent::Responding {
                request_id: id.into(),
                worker,
            },
            t(at + ttft),
        );
        a.on_event(
            &ClassifyEvent::Completed {
                request_id: id.into(),
                worker,
                context_tokens: None,
                output_tokens: Some(11),
            },
            t(at + ttft + 10.0 * itl),
        );
    }

    #[test]
    fn additive_increase_under_slack() {
        let mut a = aimd(AimdConfig::default());
        a.tick(t(0.0));
        // Priority's limits are 3 s TTFT and 0.05 s ITL: 0.5 s and 0.01 s are under 0.8 of them.
        for i in 0..3 {
            serve(
                &mut a,
                &format!("a{i}"),
                w(1),
                1.0 + f64::from(i),
                0.5,
                0.01,
            );
        }
        assert_eq!(a.limit(w(1)), Some(8));
        a.tick(t(30.0));
        assert_eq!(a.limit(w(1)), Some(9));
        serve(&mut a, "b", w(1), 31.0, 0.5, 0.01);
        a.tick(t(60.0));
        assert_eq!(a.limit(w(1)), Some(10));
        // An interval without samples leaves it.
        a.tick(t(90.0));
        assert_eq!(a.limit(w(1)), Some(10));
        // Between the slack and the limit it stays too.
        serve(&mut a, "c", w(1), 91.0, 2.8, 0.01);
        a.tick(t(120.0));
        assert_eq!(a.limit(w(1)), Some(10));
        assert_eq!(a.active(w(1)), 0);
    }

    #[test]
    fn halving_on_a_ttft_or_itl_miss() {
        let mut a = aimd(AimdConfig::default());
        a.tick(t(0.0));
        serve(&mut a, "slow", w(1), 1.0, 4.0, 0.01);
        a.tick(t(30.0));
        assert_eq!(a.limit(w(1)), Some(4));
        serve(&mut a, "choppy", w(1), 31.0, 0.5, 0.06);
        a.tick(t(60.0));
        assert_eq!(a.limit(w(1)), Some(2));
        // Another worker's miss does not move this one.
        serve(&mut a, "other", w(2), 61.0, 4.0, 0.01);
        a.tick(t(90.0));
        assert_eq!((a.limit(w(1)), a.limit(w(2))), (Some(2), Some(4)));
    }

    #[test]
    fn the_limit_stays_within_min_and_max() {
        let mut a = aimd(AimdConfig {
            start: 64,
            ..AimdConfig::default()
        });
        a.tick(t(0.0));
        serve(&mut a, "fast", w(1), 1.0, 0.5, 0.01);
        a.tick(t(30.0));
        assert_eq!(a.limit(w(1)), Some(64));

        let mut a = aimd(AimdConfig {
            start: 5,
            min: 3,
            ..AimdConfig::default()
        });
        a.tick(t(0.0));
        for (i, at) in [1.0, 31.0, 61.0].into_iter().enumerate() {
            serve(&mut a, &format!("slow{i}"), w(1), at, 4.0, 0.01);
            a.tick(t(at + 29.0));
            assert_eq!(a.limit(w(1)), Some(3), "interval {i}");
        }
    }

    #[test]
    fn waiting_turns_go_oldest_session_first() {
        let mut a = aimd(AimdConfig {
            start: 1,
            ..AimdConfig::default()
        });
        a.tick(t(0.0));
        // One slot on the only worker, taken.
        a.observe_workers(&[w(1)]);
        assert!(a.arrived("x", "sx", "priority", t(0.0), t(1.0)));
        assert_eq!(a.check("x", "sx", t(1.0)), AimdAnswer::Go(Some(w(1))));
        a.dispatched("x", Some(w(1)), t(1.0));
        // A new session asks first, then a session that started earlier.
        assert!(a.arrived("new", "s_new", "priority", t(50.0), t(50.0)));
        assert!(a.arrived("old", "s_old", "priority", t(5.0), t(51.0)));
        assert!(matches!(
            a.check("new", "s_new", t(50.0)),
            AimdAnswer::Hold(_)
        ));
        assert!(matches!(
            a.check("old", "s_old", t(51.0)),
            AimdAnswer::Hold(_)
        ));
        assert_eq!(a.waiting(), 2);
        a.on_event(
            &ClassifyEvent::Completed {
                request_id: "x".into(),
                worker: w(1),
                context_tokens: None,
                output_tokens: Some(1),
            },
            t(52.0),
        );
        // The slot is free: the newest session still waits, the older one goes.
        assert!(matches!(
            a.check("new", "s_new", t(52.0)),
            AimdAnswer::Hold(_)
        ));
        assert_eq!(a.check("old", "s_old", t(52.0)), AimdAnswer::Go(Some(w(1))));
        a.dispatched("old", Some(w(1)), t(52.0));
        assert!(matches!(
            a.check("new", "s_new", t(53.0)),
            AimdAnswer::Hold(_)
        ));
        assert_eq!(a.active(w(1)), 1);
        // A session's later turn is bound to its worker.
        a.on_event(
            &ClassifyEvent::Sent {
                request_id: "old".into(),
                worker: w(1),
            },
            t(52.0),
        );
        // A second worker with room does not take it.
        a.observe_workers(&[w(1), w(2)]);
        assert!(a.arrived("old2", "s_old", "priority", t(5.0), t(54.0)));
        assert!(matches!(
            a.check("old2", "s_old", t(54.0)),
            AimdAnswer::Hold(_)
        ));
        // Another class is not tracked at all.
        assert!(!a.arrived("std", "s_std", "standard", t(0.0), t(54.0)));
        assert_eq!(a.check("std", "s_std", t(54.0)), AimdAnswer::Untracked);
    }

    #[test]
    fn retry_after_is_place_over_departure_rate_within_bounds() {
        let mut p = policy(AimdConfig {
            start: 1,
            ..AimdConfig::default()
        });
        p.admission.retry_after_estimate = Some(RetryAfterEstimate {
            min_s: Some(5.0),
            max_s: Some(120.0),
            fallback_s: Some(30.0),
            ..RetryAfterEstimate::default()
        });
        let mut a = Aimd::from_policy(&p).unwrap();
        a.tick(t(0.0));
        // No departures yet: the fallback.
        assert!(a.arrived("q0", "s0", "priority", t(0.0), t(1.0)));
        assert_eq!(a.retry_after_s("q0", t(1.0)), 30.0);
        // Six departures in the 60 s window: 0.1 per second.
        for i in 0..6 {
            serve(
                &mut a,
                &format!("d{i}"),
                w(1),
                1.0 + f64::from(i),
                0.1,
                0.01,
            );
        }
        assert!((a.departure_rate(t(10.0)) - 0.1).abs() < 1e-9);
        // First in line: 1 / 0.1 = 10 s.
        assert!((a.retry_after_s("q0", t(10.0)) - 10.0).abs() < 1e-9);
        // Behind two older sessions' waiting turns: 30 s.
        a.observe_workers(&[w(1)]);
        assert_eq!(a.check("q0", "s0", t(10.0)), AimdAnswer::Go(Some(w(1))));
        a.dispatched("q0", Some(w(1)), t(10.0));
        for (id, start) in [("o1", 1.0), ("o2", 2.0)] {
            assert!(a.arrived(id, id, "priority", t(start), t(10.0)));
            assert!(matches!(a.check(id, id, t(10.0)), AimdAnswer::Hold(_)));
        }
        assert!(a.arrived("late", "late", "priority", t(9.0), t(10.0)));
        assert!((a.retry_after_s("late", t(10.0)) - 30.0).abs() < 1e-9);
        // Far back in line: capped at max_s.
        for i in 0..30 {
            let id = format!("o{}", i + 10);
            assert!(a.arrived(&id, &id, "priority", t(3.0), t(10.0)));
            let _ = a.check(&id, &id, t(10.0));
        }
        assert_eq!(a.retry_after_s("late", t(10.0)), 120.0);
        // Without estimate bounds: at least 1 s, at most 300 s.
        let b = aimd(AimdConfig::default());
        assert_eq!(b.retry_after_s("none", t(0.0)), 30.0);
    }

    fn turn(id: &str, session: &str, class: &str) -> ClassifyRequest {
        let mut request = ClassifyRequest::for_test(
            10,
            Some(id),
            Some(SessionContext::new(session.into(), None, None, None)),
        );
        request.set_policy_class(class);
        request
    }

    fn sent(id: &str, worker: WorkerWithDpRank) -> ClassifyEvent {
        ClassifyEvent::Sent {
            request_id: id.into(),
            worker,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_waiting_turn_is_refused_at_the_bound_with_a_retry_after() {
        let mut p = policy(AimdConfig {
            wait_bound_s: Some(20.0),
            ..AimdConfig::default()
        });
        p.admission.retry_after_estimate = Some(RetryAfterEstimate {
            min_s: Some(5.0),
            max_s: Some(120.0),
            fallback_s: Some(30.0),
            ..RetryAfterEstimate::default()
        });
        let clock = Arc::new(crate::clock::HostClock::new());
        let mut classifier = SessionSchedulerClassifier::new(p, clock, Arc::new(|| (1000, 1)));
        let ingress = Instant::now();
        let out = classifier
            .classify(turn("p1", "p", "priority"), &NoWorkers)
            .unwrap();
        let due = out
            .due_at()
            .expect("a priority turn is refused at the bound");
        assert_eq!(due.duration_since(ingress), Duration::from_secs(20));
        // No departure measured yet: the fallback, within the bounds.
        assert_eq!(out.retry_after(), Some(Duration::from_secs(30)));
        // Standard is untouched: its lobby deadline (60 s) as before.
        let std = classifier
            .classify(turn("s1", "s", "standard"), &NoWorkers)
            .unwrap();
        assert_eq!(
            std.due_at().map(|d| d.duration_since(ingress)),
            Some(Duration::from_secs(60))
        );
    }

    fn gate_head<'a>(id: &'a str, session: &'a SessionContext) -> DispatchHead<'a> {
        DispatchHead::new(
            Some(id),
            Some(session),
            10,
            None,
            "priority",
            Instant::now(),
        )
    }

    #[tokio::test]
    async fn the_gate_holds_a_turn_bound_for_a_worker_at_its_limit() {
        let clock = Arc::new(ManualClock::new());
        let mut classifier = SessionSchedulerClassifier::new(
            policy(AimdConfig {
                start: 1,
                ..AimdConfig::default()
            }),
            clock.clone(),
            Arc::new(|| (1_000_000, 1)),
        );
        let gate = classifier.dispatch_gate().unwrap();
        let (a, b) = (
            SessionContext::new("a".into(), None, None, None),
            SessionContext::new("b".into(), None, None, None),
        );
        classifier
            .classify(turn("a1", "a", "priority"), &NoWorkers)
            .unwrap();
        // Nothing known about the workers yet: it goes as before.
        assert_eq!(
            gate.check(&gate_head("a1", &a)),
            DispatchDecision::Go { target: None }
        );
        gate.dispatched(&gate_head("a1", &a), None);
        classifier.on_event(sent("a1", w(1))).await;
        clock.advance(Duration::from_secs(1));
        classifier
            .classify(turn("b1", "b", "priority"), &NoWorkers)
            .unwrap();
        assert!(matches!(
            gate.check(&gate_head("b1", &b)),
            DispatchDecision::Hold { .. }
        ));
        classifier
            .on_event(ClassifyEvent::Completed {
                request_id: "a1".into(),
                worker: w(1),
                context_tokens: Some(10),
                output_tokens: Some(2),
            })
            .await;
        assert_eq!(
            gate.check(&gate_head("b1", &b)),
            DispatchDecision::Go { target: Some(w(1)) }
        );
    }

    #[tokio::test]
    async fn options_off_leave_decisions_as_they_were() {
        assert!(Aimd::from_policy(&ServicePolicy::default()).is_none());
        let parsed: crate::config::AdmissionPolicy = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed.aimd, None);
        // Off, and on for a class the traffic does not use: the same answers for every turn.
        let run = |cfg: Option<AimdConfig>| {
            let clock = Arc::new(ManualClock::new());
            let mut p = ServicePolicy {
                mode: Mode::Enforce,
                ..Default::default()
            };
            p.admission.aimd = cfg;
            let mut classifier =
                SessionSchedulerClassifier::new(p, clock.clone(), Arc::new(|| (100, 1)));
            let gate = classifier.dispatch_gate().unwrap();
            let mut answers = Vec::new();
            for (id, session, class) in [
                ("s1", "a", "standard"),
                ("s2", "b", "standard"),
                ("b1", "c", "batch"),
                ("s3", "a", "standard"),
            ] {
                let out = classifier.classify(turn(id, session, class), &NoWorkers);
                let out = out.map(|o| (o.due_at().is_some(), o.retry_after(), o.order_key()));
                let context = SessionContext::new(session.into(), None, None, None);
                let decision = gate.check(&DispatchHead::new(
                    Some(id),
                    Some(&context),
                    10,
                    None,
                    class,
                    Instant::now(),
                ));
                answers.push(format!("{out:?} {decision:?}"));
                clock.advance(Duration::from_secs(1));
            }
            answers
        };
        let off = run(None);
        assert_eq!(
            off,
            run(Some(AimdConfig {
                class: "priority".into(),
                start: 1,
                ..AimdConfig::default()
            }))
        );
    }
}
