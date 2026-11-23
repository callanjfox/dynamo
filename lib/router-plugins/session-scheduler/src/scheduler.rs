// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The scheduler's state and the operations the router hooks drive. No router types appear here,
//! so the state machine is tested on its own with a manual clock.
//!
//! The router contract this follows (PR #18443, `dynamo-admission` classifier module):
//!
//! - `classify` and lifecycle events share one lock: [`Scheduler::on_classify`] and the
//!   `on_*` event handlers are O(1) amortized, with bounded memory, and do no forecasting.
//! - The gate's `check` is a read and may be asked many times about one head:
//!   [`Scheduler::gate_check`] changes nothing that later answers depend on (it only writes the
//!   decision log). State changes when the host takes the head: [`Scheduler::on_dispatched`].
//! - `Completed` arrives asynchronously, so a session's next turn can be classified (and even
//!   dispatched) before the previous turn's completion lands. Each request keeps its own record,
//!   and a session counts turns in flight, so a late completion settles the right turn.

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use crate::admit::{
    Decision, DecisionLog, GateAnswer, HoldReason, KvBudget, LobbyEntry, LoggedDecision,
    decide_indexed_classes, gate_answer,
};
use crate::clock::Time;
use crate::config::{LobbyOrder, Mode, ServicePolicy};
use crate::metrics::{ClassSnapshot, Completion, Metrics};
use crate::session::{LobbyOrderKeys, SessionState, SessionTable, Signals};

#[derive(Clone, Debug)]
struct RequestRecord {
    session: String,
    class: String,
    arrived: Time,
    first_token: Option<Time>,
    /// The gate let it go and the host took it; only then does it count as in flight.
    dispatched: bool,
    /// When the host took it (`admission.hot_demand: next_turn` reads the service time).
    dispatched_at: Option<Time>,
    /// The client marked this the session's last request (`nvext.agent_context.session_final`).
    ends_session: bool,
    /// Prompt tokens, for the requests still waiting at the router (`adaptive.queue_load`).
    isl: u64,
    /// The router sent it to a worker.
    sent: bool,
    /// The top-level session it counts for in the session log (itself, or its subagent chain's
    /// root), F35.
    root: String,
    /// Prompt tokens with the cached prefix, when the host said (`set_request_prompt`; f36b).
    prompt: Option<u64>,
    /// The session's context before this request (its previous request's input and output), the
    /// prefix an idle worker would hold (uncontended time, f36b).
    prev_context: u64,
    /// The ordering key given to the router queue (`turn_order: run_las` or
    /// `server_wait_deficit`; lower first), f36b.
    order_key: Option<f64>,
}

/// What the frontend attached to one request beyond its session: the tenant, the client's
/// signals about the next turn, and whether this request ends the session.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RequestAttributes {
    pub tenant: Option<String>,
    pub signals: Signals,
    pub session_final: bool,
    /// `x-dynamo-hint-session-kind` (F35).
    pub session_kind: Option<crate::session::SessionKind>,
    /// `x-dynamo-user-id` (F35).
    pub user: Option<String>,
    /// `x-dynamo-hint-last-role: user`: this turn's last message is a new user message, so the
    /// previous turn handed back to the user (F35).
    pub next_is_user: bool,
    /// The frontend (`input_trigger`) or the client said what this request's last message is (a
    /// user message or a tool result), so the stop gap does not decide (f36b).
    pub next_known: bool,
}

/// Bounded map from request id to its record, oldest dropped first.
#[derive(Debug)]
struct Requests {
    capacity: usize,
    map: HashMap<String, RequestRecord>,
    order: VecDeque<String>,
    /// Prompt tokens of the records still waiting at the router ([`RequestRecord::waiting`]).
    queued: u64,
}

impl Requests {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            map: HashMap::new(),
            order: VecDeque::new(),
            queued: 0,
        }
    }

    fn insert(&mut self, id: String, record: RequestRecord) {
        if record.waiting() {
            self.queued += record.isl;
        }
        match self.map.insert(id.clone(), record) {
            None => self.order.push_back(id),
            Some(old) if old.waiting() => self.queued -= old.isl,
            Some(_) => {}
        }
        while self.map.len() > self.capacity {
            if let Some(old) = self.order.pop_front() {
                self.remove(&old);
            } else {
                break;
            }
        }
        // Ids already removed leave stale entries in `order`; keep it from growing without bound.
        if self.order.len() > self.capacity * 2 {
            self.order.retain(|id| self.map.contains_key(id));
        }
    }

    fn remove(&mut self, id: &str) -> Option<RequestRecord> {
        let record = self.map.remove(id)?;
        if record.waiting() {
            self.queued -= record.isl;
        }
        Some(record)
    }

    /// Change the record of `id`, keeping the waiting count.
    fn update<R>(&mut self, id: &str, f: impl FnOnce(&mut RequestRecord) -> R) -> Option<R> {
        let record = self.map.get_mut(id)?;
        let was = record.waiting();
        let out = f(record);
        if was && !record.waiting() {
            self.queued -= record.isl;
        }
        Some(out)
    }
}

impl RequestRecord {
    /// Still waiting at the router: neither taken by the host nor sent to a worker.
    fn waiting(&self) -> bool {
        !self.dispatched && !self.sent && self.first_token.is_none()
    }
}

/// What the gate knows about the queue head beyond its session.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HeadFacts {
    /// When the head entered the router queue.
    pub queued_at: Option<Time>,
    /// Whether some worker has KV room for the head beside the work active there, from the
    /// classifier's latest worker sample; `None` when unknown (the shed never holds then).
    pub kv_room: Option<bool>,
    /// Whether some worker is below its running-request limit, from the same sample; `None`
    /// when unknown or no worker reports a limit.
    pub seq_room: Option<bool>,
    /// With `admission.shed_forecast_harm`: the misses the forecast predicts the head's best
    /// placement adds to each class (running requests pushed over their ITL limit); empty when
    /// unknown.
    pub harm: Vec<(String, f64)>,
    /// With `admission.surge`: seconds the fleet's queued prefill takes to clear, and the head's
    /// own prefill seconds, from the step model; `None` when unknown (surge never holds then).
    pub surge: Option<(f64, f64)>,
    /// With `admission.prefill_budget` (f42): the fleet's prefill budget and its measured prefill
    /// rate, tokens per second; `None` when unknown (the budget never holds then).
    pub prefill: Option<(f64, f64)>,
    /// With `admission.forecast_harm_new` (ks2b): the weighted misses the forecast predicts a new
    /// session's best placement adds to its class and the classes ranked above it; `None` when
    /// unknown (the check never holds then).
    pub new_harm: Option<f64>,
}

/// The target loop's state (`admission.target_loop`, f42).
#[derive(Debug)]
struct LoopState {
    scale: f64,
    last: Option<Time>,
    /// Smoothed attainment per class with a service level.
    att: HashMap<String, f64>,
    /// Lobby checks the prefill budget judged and held since the last read.
    judged: u64,
    held: u64,
    /// Target changes applied (`admission.target_schedule`).
    applied: usize,
    /// Since when a higher class has been short with the scale at its floor.
    short_since: Option<Time>,
    /// The higher class whose target the loop cannot meet now, if any.
    infeasible: Option<String>,
    /// Seconds spent infeasible, and holds for it, since the last read.
    infeasible_s: f64,
    infeasible_held: u64,
    /// The look-ahead's predicted prefill rate, tokens per second, and when it was computed.
    predicted: Option<f64>,
    predicted_at: Option<Time>,
    /// The budget's calibration factor (`prefill_budget.calibrate`) and when it last moved; the top
    /// class's ITL miss share it last read.
    cal: f64,
    cal_at: Option<Time>,
    itl_miss: Option<f64>,
    /// Lost sessions of the loop's class seen at the last step (yield to demand).
    lost_seen: u64,
    /// Since when the scale has sat at its floor.
    floor_since: Option<Time>,
    /// The guard could not be bought with the scale at its floor: it stays off until a higher class
    /// falls under its own target.
    guard_released: bool,
    /// Best effort (`infeasible_hold: false`): the higher class's attainment when its target was
    /// declared infeasible, the floor its guard now protects.
    floor_att: Option<f64>,
    /// When the current infeasible state was declared (for `reprobe_s`).
    declared_at: Option<Time>,

    /// A session of a class ranked above the loop's class was waiting in the lobby at the last
    /// step (`target_loop.priority_wait_cut`).
    higher_waiting: bool,

    /// f49n `target_loop.rebalance`: the per-class scales.
    rb: LoopRb,
}

/// f49n `target_loop.rebalance`: the per-class scales' state.
#[derive(Debug, Default)]
struct LoopRb {
    /// Each class's scale at or below the loop's class (absent: 1).
    scale: HashMap<String, f64>,
    /// Lost sessions per class seen at the last step (yield to demand).
    lost_seen: HashMap<String, u64>,
    /// Classes past their warm-up (latched).
    warm: std::collections::HashSet<String>,
    /// The latest step's mode (empty before one) and the top class's head.
    mode: &'static str,
    head: Option<f64>,
}

/// The target loop's counters for the run log ([`Scheduler::take_loop_stats`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LoopStats {
    pub scale: f64,
    pub att: HashMap<String, f64>,
    pub targets: HashMap<String, f64>,
    pub judged: u64,
    pub held: u64,
    pub infeasible: Option<String>,
    pub infeasible_s: f64,
    pub infeasible_held: u64,
    pub predicted: Option<f64>,
    pub cal: f64,
    pub itl_miss: Option<f64>,
    /// f49n: with `target_loop.rebalance`, its figures.
    pub rb: Option<LoopRbStats>,
}

/// f49n `target_loop.rebalance`'s figures for the run log.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LoopRbStats {
    /// The latest step's mode: `release`, `pull`, `hold`, `floor` or `warmup`.
    pub mode: String,
    /// The top class's head (smoothed attainment less its reference).
    pub head: Option<f64>,
    /// By rank: (class, loop scale on its gate share, smoothed attainment, target). A class ranked
    /// above the loop's class shows its no-inversion lift.
    pub classes: Vec<(String, f64, Option<f64>, Option<f64>)>,
}

impl LoopRbStats {
    /// The `DYNAMO_F42_LOG` line's fields: `"rb_mode"`, `"rb_head"` and `"rb"` (per class
    /// `scale`, `att`, `target`).
    pub fn log_fields(&self) -> String {
        let num = |x: Option<f64>| x.map_or("null".to_owned(), |v| format!("{v:.4}"));
        let classes = self
            .classes
            .iter()
            .map(|(c, scale, att, target)| {
                format!(
                    "{c:?}:{{\"scale\":{scale:.4},\"att\":{},\"target\":{}}}",
                    num(*att),
                    num(*target)
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "\"rb_mode\":{:?},\"rb_head\":{},\"rb\":{{{classes}}}",
            self.mode,
            num(self.head)
        )
    }
}

impl Default for LoopState {
    fn default() -> Self {
        Self {
            scale: 1.0,
            last: None,
            att: HashMap::new(),
            judged: 0,
            held: 0,
            applied: 0,
            short_since: None,
            infeasible: None,
            infeasible_s: 0.0,
            infeasible_held: 0,
            predicted: None,
            predicted_at: None,
            cal: 1.0,
            cal_at: None,
            itl_miss: None,
            lost_seen: 0,
            floor_since: None,
            guard_released: false,
            higher_waiting: false,
            floor_att: None,
            declared_at: None,
            rb: LoopRb::default(),
        }
    }
}

/// What can change a held head's answer before its `until` ([`Scheduler::gate_check_head_wake`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GateWake {
    /// Anything: ask again at every recheck.
    Any,
    /// The head's session waits behind `ahead` whatever else changes: only `ahead` or the
    /// session itself leaving or moving in the lobby ([`SessionTable::take_lobby_changes`]), or
    /// the adaptive rule switching the gate, can change the answer before its `until`.
    Behind { ahead: String },
}

/// Gate-share pacing state (`admission.pace`).
#[derive(Debug)]
struct PaceState {
    standard: f64,
    batch: f64,
    priority_att: Option<f64>,
    standard_att: Option<f64>,
    last: Option<Time>,
    lost_standard_seen: u64,
}

impl Default for PaceState {
    fn default() -> Self {
        Self {
            standard: 1.0,
            batch: 1.0,
            priority_att: None,
            standard_att: None,
            last: None,
            lost_standard_seen: 0,
        }
    }
}

/// Target-aware shedding and surplus release state (`admission.shed_target`,
/// `admission.release`).
#[derive(Debug, Default)]
struct TargetState {
    /// Smoothed windowed attainment per class with a service level.
    att: HashMap<String, f64>,
    /// Shed switch per class: (on, when it last changed). Absent: on.
    shed: HashMap<String, (bool, Option<Time>)>,
    /// Release multiple of each class's gate share (absent: 1).
    release: HashMap<String, f64>,
    lost_seen: HashMap<String, u64>,
    last: Option<Time>,
    /// Shed switch changes, for reporting.
    flips: u64,
}

/// Sessions rotation may pause for one waiting class ([`Scheduler::rotation_candidates`]).
#[derive(Clone, Debug, PartialEq)]
pub struct RotationCandidates {
    /// The class whose oldest waiter triggered it.
    pub for_class: String,
    /// That waiter's wait, seconds.
    pub waited_s: f64,
    /// (session, its class, seconds since its latest admission), in pause order.
    pub sessions: Vec<(String, String, f64)>,
}

/// One class's lobby at a moment ([`Scheduler::lobby_stats`]).
#[derive(Clone, Debug, PartialEq)]
pub struct LobbyStats {
    pub class: String,
    /// Sessions waiting (new and resuming).
    pub waiting: usize,
    /// The longest current wait, seconds ([`crate::session::Session::lobby_wait_s`]).
    pub oldest_wait_s: f64,
    /// Admissions out of the lobby a second over the class's estimate window, when known.
    pub admit_rate: Option<f64>,
}

/// The scheduler.
#[derive(Debug)]
pub struct Scheduler {
    policy: ServicePolicy,
    sessions: SessionTable,
    requests: Requests,
    metrics: Metrics,
    log: DecisionLog,
    pace: PaceState,
    targets: TargetState,
    /// Shed-by-slack diagnostics: [admitted turns checked, no worker had KV room, room unknown,
    /// a higher class had no slack, holds].
    shed_counts: [std::sync::atomic::AtomicU64; 5],
    /// New sessions turned away by class (expired in the lobby or throttled).
    lost: HashMap<String, u64>,
    /// Sessions rotated out (`admission.rotate`).
    rotations: u64,
    /// Per class, the times of recent admissions out of the lobby (for the Retry-After
    /// estimate and the lobby stats), trimmed to the class's estimate window.
    admits: HashMap<String, VecDeque<Time>>,
    /// Per class, the times of recent ends of admitted sessions (`retry_after_estimate.count_ends`,
    /// f40), trimmed like `admits`.
    ends: HashMap<String, VecDeque<Time>>,
    /// The adaptive rule (`adaptive.rs`), when configured.
    adaptive: Option<crate::adaptive::AdaptiveState>,
    /// Surge admission's counts by class since the last [`Self::take_surge_counts`]: (lobby
    /// checks it judged, of those held by it).
    surge_counts: std::collections::BTreeMap<String, (u64, u64)>,
    /// Sessions the client ended since the last [`Self::take_ended`], with their context tokens,
    /// for the KV-action check (`admission.accounting.verify`, F34); kept only when it is on.
    ended: Vec<(String, u64)>,
    /// What each session gets (`experience.rs`, F35).
    experience: crate::experience::Experience,
    /// Onboarding pace (`admission.onboard_pace`, f39): the bucket's tokens (negative in debt) and
    /// when it was last refilled.
    onboard_level: f64,
    onboard_at: Option<Time>,
    /// Finished runs per class for the class's run attainment (f36b, `runs.signal: slowdown`).
    runs: RunWindows,
    /// Per class, added delay and allowance over the trailing `run_level.delay_window_s` (f36b,
    /// `runs.signal: delay`).
    delay: DelayWindows,
    /// The step model that prices a request's uncontended time (f36b), and the worker's batch
    /// token budget its prefill is chunked at.
    step: Option<StepHandle>,
    chunk_tokens: u64,
    /// The target loop and the prefill budget's counts (f42).
    lp: LoopState,
    /// f47: the Retry-After calibration factor by class, and the sessions refused with one that
    /// have not been checked since.
    ra_factor: HashMap<String, f64>,
    /// Per refused session: (returns so far, refused since its last check).
    ra_waiting: HashMap<String, (u32, bool)>,
    /// f47: refusals so far per session (with `admission.returnee_guarantee`).
    refusals: HashMap<String, u32>,
    /// f47h (`prefill_budget.meter: uncached`): per request, the tokens it will compute (its prompt
    /// less its prefix cached in any tier) at best and on each worker; per session, its pending turn's.
    compute: HashMap<
        String,
        (
            u64,
            Vec<(dynamo_kv_router::plugins::prelude::WorkerWithDpRank, u64)>,
        ),
    >,
    session_compute: HashMap<String, u64>,
    /// f47k: the host's cached prompt tokens on the worker each request was sent to (`Placed`).
    placed: HashMap<String, u64>,
    /// f49c: the admission tickets (`admission.tickets`; empty without it).
    tickets: crate::tickets::Tickets,
    /// f49m `admission.honest_retry` (`None`: off).
    honest: Option<crate::honest_retry::HonestRetry>,
    /// f49k `admission.hot_budget: measured`: the measured hot use (`None`: off).
    hot_measure: Option<crate::hot_measure::HotMeasure>,
    /// f49d `DYNAMO_HARM_DEBUG=1`: the forecast-harm check's counters (`None`: off).
    harm_debug: Option<crate::harm_debug::HarmDebug>,
    /// f49j `admission.step_target`: the per-worker step-target controller (`None`: off).
    step_target: Option<crate::step_target::StepTarget>,
    /// f49l `admission.sla_budget`: the outer SLA loop (`None`: off).
    sla_budget: Option<crate::sla_budget::SlaBudget>,
    /// f49q: the session signal's state (`None`: neither `starvation_override` nor `lobby_order:
    /// signal`), and its settings (`admission.session_signal` or the defaults).
    signals: Option<crate::session_signal::SignalState>,
    signal_cfg: crate::session_signal::SessionSignalConfig,
    /// f49o `admission.batch_yield`: the yield controller (`None`: off).
    batch_yield: Option<crate::batch_yield::BatchYield>,
    /// f50 `mix_select`: the selector and its two mode policies (`None`: off).
    mix: Option<crate::mix_select::MixState>,
}

/// The lobby's order keys for `policy`.
fn lobby_keys(policy: &ServicePolicy) -> LobbyOrderKeys {
    LobbyOrderKeys {
        ranks: policy
            .classes
            .iter()
            .map(|c| (c.name.clone(), c.rank))
            .collect(),
        default_rank: policy.class(None).rank,
        fair: policy.admission.lobby_order == LobbyOrder::Fair,
        aging: policy.admission.lobby_order == LobbyOrder::Aging,
        session_age: policy.admission.turn_order == crate::config::TurnOrder::SessionAge,
    }
}

/// f49c: how far the tickets' Retry-After forecast looks without `retry_after_estimate.max_s`.
const TICKET_HORIZON_S: f64 = 600.0;

/// The step model behind an `Arc`, for [`Scheduler`]'s `Debug`.
#[derive(Clone)]
struct StepHandle(std::sync::Arc<dyn crate::forecast::StepModel>);

impl std::fmt::Debug for StepHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("StepModel")
    }
}

/// Most finished runs kept per class.
const MAX_RUNS_PER_CLASS: usize = 65_536;

/// Per class, the finished runs of the trailing run window: (end, within the slowdown target).
#[derive(Debug, Default)]
struct RunWindows {
    by_class: HashMap<String, VecDeque<(Time, bool)>>,
}

impl RunWindows {
    fn record(&mut self, class: &str, end: Time, met: bool, window_s: f64, now: Time) {
        let q = self.by_class.entry(class.to_owned()).or_default();
        // Runs are recorded when their stop point is known, so ends arrive nearly in order.
        let at = q.partition_point(|(t, _)| *t <= end);
        q.insert(at, (end, met));
        while q
            .front()
            .is_some_and(|(t, _)| now.since(*t).as_secs_f64() >= window_s)
            || q.len() > MAX_RUNS_PER_CLASS
        {
            q.pop_front();
        }
    }

    /// Share of the runs that ended in the trailing `window_s` within their target; `None` under
    /// [`crate::metrics::MIN_WINDOW_COMPLETIONS`] runs.
    fn share(&self, class: &str, now: Time, window_s: f64) -> Option<f64> {
        let q = self.by_class.get(class)?;
        let (mut n, mut met) = (0usize, 0usize);
        for (_, ok) in q
            .iter()
            .rev()
            .take_while(|(t, _)| now.since(*t).as_secs_f64() < window_s)
        {
            n += 1;
            met += usize::from(*ok);
        }
        (n >= crate::metrics::MIN_WINDOW_COMPLETIONS).then(|| met as f64 / n as f64)
    }
}

/// Most delay samples kept per class.
const MAX_DELAY_SAMPLES: usize = 1 << 18;

/// Per class, (time, added delay, allowance) samples of the trailing delay window: added delay
/// as it is known (a request's waiting and prefill excess at completion, a refused attempt at its
/// end, retry backoff at the retry) and allowance as it accrues (the base at a run's first
/// request, the per-reference share at each completion).
#[derive(Debug, Default)]
struct DelayWindows {
    by_class: HashMap<String, VecDeque<(Time, f64, f64)>>,
}

impl DelayWindows {
    fn add(&mut self, class: &str, now: Time, delay: f64, allowance: f64, window_s: f64) {
        let q = self.by_class.entry(class.to_owned()).or_default();
        q.push_back((now, delay, allowance));
        while q
            .front()
            .is_some_and(|(t, _, _)| now.since(*t).as_secs_f64() >= window_s)
            || q.len() > MAX_DELAY_SAMPLES
        {
            q.pop_front();
        }
    }

    /// 1 - delay / allowance over the trailing `window_s`, within [-1, 1]; `None` without any
    /// allowance in it.
    fn headroom(&self, class: &str, now: Time, window_s: f64) -> Option<f64> {
        let q = self.by_class.get(class)?;
        let (mut d, mut a) = (0.0, 0.0);
        for (_, delay, allow) in q
            .iter()
            .rev()
            .take_while(|(t, _, _)| now.since(*t).as_secs_f64() < window_s)
        {
            d += delay;
            a += allow;
        }
        (a > 0.0).then(|| (1.0 - d / a).clamp(-1.0, 1.0))
    }
}

impl Scheduler {
    pub fn new(policy: ServicePolicy) -> Self {
        // f50 `mix_select`: run the start mode's policy; the base is the union of both modes.
        let mix = policy
            .mix_select
            .clone()
            .filter(|m| m.enabled)
            .map(|cfg| crate::mix_select::MixState::new(&policy, cfg));
        crate::mix_select::set_current(mix.as_ref().map(|m| m.selector.mode));
        if let Some(m) = &mix {
            crate::mix_select::apply_router_busy(m.config.parts(m.selector.mode));
        }
        // The session signal's state is kept in every mode, so a switch to a mode that reads
        // it finds it current.
        let keep_signals = mix.is_some();
        let policy = match &mix {
            Some(m) => m.policy(m.selector.mode).clone(),
            None => policy,
        };
        let policy_adaptive = policy
            .adaptive
            .clone()
            .map(crate::adaptive::AdaptiveState::new);
        let mut sessions = SessionTable::new(policy.session_table_capacity);
        let (policy_capacity, policy_stop_gap, policy_max_runs) = (
            policy.session_table_capacity,
            policy.session_log.stop_gap_s,
            policy.session_log.max_runs,
        );
        let floor_window_s = policy.runs.floor_window_s;
        let allowances: HashMap<String, (f64, f64)> = policy
            .classes
            .iter()
            .filter_map(|c| {
                let l = c.run_level.as_ref()?;
                Some((c.name.clone(), (l.delay_base_s, l.delay_per_ref)))
            })
            .collect();
        let chunk_tokens = u64::from(policy.forecast.chunk_tokens.max(1));
        let tickets = crate::tickets::Tickets::new(policy.admission.tickets.is_some());
        // f49q: kept only when the override or the signal order is on.
        let signals = (policy.admission.starvation_override.is_some()
            || policy.admission.lobby_order == LobbyOrder::Signal
            || keep_signals)
            .then(|| crate::session_signal::SignalState::new(tickets.logging()));
        let signal_cfg = policy.admission.session_signal.clone().unwrap_or_default();
        let honest = policy
            .admission
            .honest_retry
            .as_ref()
            .map(|_| crate::honest_retry::HonestRetry::default());
        // f49o: batch yield tracks its class's KV apart too (counted only times the level).
        sessions.set_yield_class(policy.admission.yield_class.clone().or_else(|| {
            policy
                .admission
                .batch_yield
                .as_ref()
                .filter(|b| b.admit)
                .map(|b| b.class.clone())
        }));
        sessions.set_hot_exempt(policy.admission.hot_exempt_classes.clone());
        let onboard_level = policy
            .admission
            .onboard_pace
            .as_ref()
            .map_or(0.0, |p| p.burst_tokens);
        sessions.set_hot_demand(policy.admission.hot_demand);
        sessions.set_lifetime(policy.admission.lifetime);
        sessions.set_lobby_order(lobby_keys(&policy));
        Self {
            sessions,
            requests: Requests::new(policy.session_table_capacity.saturating_mul(4)),
            hot_measure: crate::hot_measure::HotMeasure::from_policy(&policy),
            metrics: Metrics::new(&policy),
            step_target: crate::step_target::StepTarget::from_policy(&policy),
            sla_budget: crate::sla_budget::SlaBudget::from_policy(&policy),
            batch_yield: crate::batch_yield::BatchYield::from_policy(&policy),
            log: DecisionLog::new(policy.decision_log_capacity),
            policy,
            pace: PaceState::default(),
            targets: TargetState::default(),
            shed_counts: Default::default(),
            lost: HashMap::new(),
            rotations: 0,
            admits: HashMap::new(),
            ends: HashMap::new(),
            adaptive: policy_adaptive,
            surge_counts: std::collections::BTreeMap::new(),
            ended: Vec::new(),
            onboard_level,
            onboard_at: None,
            experience: crate::experience::Experience::new(
                policy_capacity,
                policy_stop_gap,
                policy_max_runs,
            )
            .with_floor_window(floor_window_s)
            .with_allowances(allowances),
            runs: RunWindows::default(),
            delay: DelayWindows::default(),
            lp: LoopState::default(),
            ra_factor: HashMap::new(),
            ra_waiting: Default::default(),
            refusals: HashMap::new(),
            compute: HashMap::new(),
            placed: HashMap::new(),
            session_compute: HashMap::new(),
            tickets,
            honest,
            step: None,
            chunk_tokens,
            signals,
            signal_cfg,
            harm_debug: crate::harm_debug::enabled_by_env().then(Default::default),
            mix,
        }
    }

    /// The step model that prices each request's uncontended time (F1): the run log's
    /// uncontended time and run slowdown, run attainment and `run_las`'s attained service.
    /// Without one these are unknown.
    pub fn set_step_model(&mut self, model: std::sync::Arc<dyn crate::forecast::StepModel>) {
        self.step = Some(StepHandle(model));
    }

    /// The workers' batch token budget (`max_num_batched_tokens`), the prefill chunk of the
    /// uncontended time; default `forecast.chunk_tokens`.
    pub fn set_chunk_tokens(&mut self, tokens: u64) {
        if tokens > 0 {
            self.chunk_tokens = tokens;
        }
    }

    /// Whether anything reads runs: run ordering, run headroom or compute-headroom admission.
    pub fn uses_runs(&self) -> bool {
        use crate::config::{HeadroomSource, TurnOrder};
        matches!(
            self.policy.admission.turn_order,
            TurnOrder::RunLas | TurnOrder::ServerWaitDeficit
        ) || self.policy.runs.headroom != HeadroomSource::Request
            || self.policy.admission.compute_headroom.is_some()
    }

    /// The prompt tokens of `request_id` with its cached prefix (the classifier knows them; the
    /// scheduler is given the uncached part), for its uncontended time.
    pub fn set_request_prompt(&mut self, request_id: &str, prompt_tokens: u64) {
        self.requests
            .update(request_id, |r| r.prompt = Some(prompt_tokens));
    }

    /// A request's uncached input tokens (its prompt less the best cached prefix at
    /// classification) and its class, for the prefill meter (`admission.prefill_budget`, f42).
    /// Whether the prefill budget meters computed tokens only (`prefill_budget.meter: uncached`).
    fn meter_uncached(&self) -> bool {
        self.policy
            .admission
            .prefill_budget
            .as_ref()
            .is_some_and(|c| c.meter == crate::config::PrefillMeterKind::Uncached)
    }

    /// f47j: the tokens `request_id` will compute, at best and per worker (`meter: uncached`).
    pub fn request_compute(
        &self,
        request_id: &str,
    ) -> Option<(
        u64,
        Vec<(dynamo_kv_router::plugins::prelude::WorkerWithDpRank, u64)>,
    )> {
        self.compute.get(request_id).cloned()
    }

    /// f47h: record the tokens `request_id` (of `session`) will compute, at best and per worker.
    pub fn set_request_compute(
        &mut self,
        request_id: &str,
        session: &str,
        best: u64,
        per_worker: Vec<(dynamo_kv_router::plugins::prelude::WorkerWithDpRank, u64)>,
    ) {
        let cap = self
            .policy
            .session_table_capacity
            .saturating_mul(4)
            .max(1024);
        if self.compute.len() >= cap {
            self.compute.clear();
        }
        if self.session_compute.len() >= cap {
            self.session_compute.clear();
        }
        self.compute
            .insert(request_id.to_owned(), (best, per_worker));
        self.session_compute.insert(session.to_owned(), best);
    }

    /// The meter's tokens for `request_id` sent (or about to be sent) to `worker`: with
    /// `meter: uncached` its computed tokens there (the best worker's when unknown), else
    /// [`Self::request_prefill`]'s.
    /// f47k: the host's estimate of `request_id`'s cached prompt tokens on its selected worker.
    pub fn set_placed(&mut self, request_id: &str, cached_tokens: u64) {
        if self.meter_uncached() {
            if self.placed.len()
                >= self
                    .policy
                    .session_table_capacity
                    .saturating_mul(4)
                    .max(1024)
            {
                self.placed.clear();
            }
            self.placed.insert(request_id.to_owned(), cached_tokens);
        }
    }

    pub fn request_prefill_on(
        &self,
        request_id: &str,
        worker: Option<dynamo_kv_router::plugins::prelude::WorkerWithDpRank>,
    ) -> Option<(u64, String)> {
        let (isl, class) = self.request_prefill(request_id)?;
        if !self.meter_uncached() {
            return Some((isl, class));
        }
        // f47k: the host's own count on the worker it was sent to, when it gave one.
        if let (Some(cached), Some(prompt)) = (
            self.placed.get(request_id),
            self.requests.map.get(request_id).and_then(|r| r.prompt),
        ) {
            return Some((prompt.saturating_sub(*cached), class));
        }
        let tokens = self.compute.get(request_id).map_or(isl, |(best, per)| {
            worker
                .and_then(|w| per.iter().find(|(x, _)| *x == w).map(|(_, t)| *t))
                .unwrap_or(*best)
        });
        Some((tokens, class))
    }

    pub fn request_prefill(&self, request_id: &str) -> Option<(u64, String)> {
        self.requests
            .map
            .get(request_id)
            .map(|r| (r.isl, r.class.clone()))
    }

    /// The router queue ordering key given to `request_id` (lower first), with `turn_order:
    /// run_las` or `server_wait_deficit`.
    pub fn request_order_key(&self, request_id: &str) -> Option<f64> {
        self.requests.map.get(request_id).and_then(|r| r.order_key)
    }

    /// Whether sessions of `kind` are run-ordered, counted for run attainment and held by compute
    /// headroom (`runs.kind`; every session without it).
    fn run_kind(&self, kind: Option<&str>) -> bool {
        self.policy
            .runs
            .kind
            .as_deref()
            .is_none_or(|k| kind.is_some_and(|have| have.eq_ignore_ascii_case(k)))
    }

    /// Record the runs the session log finished since the last call.
    fn absorb_runs(&mut self, now: Time) {
        let finished = self.experience.take_finished();
        for run in finished {
            if !self.run_kind(run.kind) {
                continue;
            }
            let Some(level) = self.policy.class(Some(&run.class)).run_level.clone() else {
                continue;
            };
            let Some(slowdown) = run.slowdown() else {
                continue;
            };
            self.runs.record(
                &run.class,
                run.end,
                slowdown <= level.slowdown,
                level.window_s,
                now,
            );
        }
    }

    /// Share of `class`'s runs that ended in its trailing `run_level.window_s` within its slowdown
    /// target; `None` without a run level or under five runs.
    pub fn run_share(&self, class: &str, now: Time) -> Option<f64> {
        let level = self.policy.class(Some(class)).run_level.as_ref()?;
        self.runs.share(class, now, level.window_s)
    }

    /// Run headroom (`runs.signal`): with `delay`, 1 - added delay / allowance over the class's
    /// trailing `run_level.delay_window_s`, within [-1, 1]; with `slowdown`, run attainment less
    /// `run_level.attainment`.
    pub fn run_headroom(&self, class: &str, now: Time) -> Option<f64> {
        let level = self.policy.class(Some(class)).run_level.as_ref()?;
        match self.policy.runs.signal {
            crate::config::RunSignal::Delay => {
                self.delay.headroom(class, now, level.delay_window_s)
            }
            crate::config::RunSignal::Slowdown => {
                self.run_share(class, now).map(|a| a - level.attainment)
            }
        }
    }

    /// Run headroom in attainment units for the readers that compare an attainment with a
    /// target: a delay headroom h reads as h x (1 - target) above 0 and h x target below, so no
    /// delay reads as full attainment and delay at the allowance as the target.
    fn run_headroom_as_attainment(&self, class: &str, target: f64, now: Time) -> Option<f64> {
        let h = self.run_headroom(class, now)?;
        Some(match self.policy.runs.signal {
            crate::config::RunSignal::Delay if h >= 0.0 => h * (1.0 - target),
            crate::config::RunSignal::Delay => h * target,
            crate::config::RunSignal::Slowdown => h,
        })
    }

    /// Account added delay `delay` and allowance `allowance` to the root session's class, when it
    /// is a run-ordered kind with a run level.
    fn accrue_delay(&mut self, root: &str, now: Time, delay: f64, allowance: f64) {
        let Some(account) = self.experience.get(root) else {
            return;
        };
        if !self.run_kind(account.kind) {
            return;
        }
        let class = account.class.clone();
        let Some(window_s) = self
            .policy
            .class(Some(&class))
            .run_level
            .as_ref()
            .map(|l| l.delay_window_s)
        else {
            return;
        };
        if delay > 0.0 || allowance > 0.0 {
            self.delay.add(&class, now, delay, allowance, window_s);
        }
    }

    /// A request of `root` was refused before classification completed (throttled: the lobby
    /// was full), so the session log and the delay count the refusal and the retry backoff.
    pub fn note_throttled(&mut self, session: &str, class: Option<&str>, now: Time) {
        let class = self.policy.class(class).name.clone();
        let root = self.root_of(session);
        let (opened, backoff) = self.experience.start(&root, &class, now);
        self.experience.end(&root, now, None, false, true, None);
        let base = self.experience.allowance(&class).map_or(0.0, |a| a.0);
        self.accrue_delay(&root, now, backoff, if opened { base } else { 0.0 });
    }

    /// The attainment the parts that read a class's headroom see (`runs.headroom`): per request
    /// (the windowed share, as before), or, for a class with a service level, its target plus run
    /// headroom, or plus the tighter of the two.
    fn attainment(&self, class: &str, now: Time) -> Option<f64> {
        use crate::config::HeadroomSource;
        let request = self.metrics.window_share(class, now);
        let source = self.policy.runs.headroom;
        if source == HeadroomSource::Request {
            return request;
        }
        let Some(target) = self
            .policy
            .classes
            .iter()
            .find(|c| c.name == class)
            .and_then(|c| c.service_level.as_ref())
            .map(|l| l.attainment)
        else {
            return request;
        };
        let run = self.run_headroom_as_attainment(class, target, now);
        let headroom = match source {
            HeadroomSource::Request => unreachable!("returned above"),
            HeadroomSource::Run => run,
            HeadroomSource::Min => match (request.map(|a| a - target), run) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            },
        };
        headroom.map(|h| target + h)
    }

    /// The router queue and lobby ordering key of a request of `root`'s session (lower first),
    /// with `turn_order: run_las` or `server_wait_deficit`; `None` otherwise. `isl` uncached
    /// tokens after `prefix` cached ones.
    fn order_key(&self, root: &str, class: &str, isl: u64, prefix: u64, now: Time) -> Option<f64> {
        use crate::config::TurnOrder;
        let order = self.policy.admission.turn_order;
        if !matches!(order, TurnOrder::RunLas | TurnOrder::ServerWaitDeficit) {
            return None;
        }
        let account = self.experience.get(root)?;
        let eligible = self.run_kind(account.kind);
        let level = self.policy.class(Some(class)).run_level.as_ref();
        let now_s = now.as_secs_f64();
        if order == TurnOrder::ServerWaitDeficit {
            if !eligible {
                return Some(0.0);
            }
            if self.policy.runs.deficit == crate::config::DeficitMeasure::Delay {
                // The session furthest over its allowance goes first.
                let allowance = self.experience.allowance(&account.class);
                return Some(-account.delay_deficit_s(allowance));
            }
            let wall = now.since(account.started).as_secs_f64();
            let share = if wall > 0.0 {
                (account.server_s_at(now) / wall).min(1.0)
            } else {
                0.0
            };
            let target = level.map_or(1.0, |l| l.server_share);
            // The session furthest over its class's share goes first.
            return Some(target - share);
        }
        if !eligible {
            // First come, as a fresh run.
            return Some(now_s);
        }
        let cfg = &self.policy.runs;
        let run = &account.run;
        let mut key = now_s + cfg.las_weight * run.attained_s;
        let age = now.since(run.start).as_secs_f64();
        let promote_at = if age >= cfg.autonomous_after_s {
            // Autonomous: the floor promotes it once its progress over the trailing window would
            // fall under the floor share.
            let from = Time::from_secs_f64((now_s - cfg.floor_window_s).max(0.0));
            let (at, progress) = run.progress_at(from);
            Some(at.as_secs_f64() + (run.progress_s() - progress) / cfg.floor_share)
        } else {
            let own = self.step.as_ref().map_or(0.0, |m| {
                crate::uncontended::prefill_s(m.0.as_ref(), self.chunk_tokens, isl, prefix)
                    .unwrap_or(0.0)
            });
            level.map(|l| match cfg.signal {
                // Promoted once its added delay, growing as it waits, would pass its allowance
                // (this request's own prefill counted as reference service to come).
                crate::config::RunSignal::Delay => {
                    let allowance = l.delay_base_s + l.delay_per_ref * (run.s_ref_s + own);
                    now_s + (allowance - run.d1_s())
                }
                // Promoted once its slowdown passes the class's target.
                crate::config::RunSignal::Slowdown => {
                    run.start.as_secs_f64() + l.slowdown * (run.progress_s() + own)
                }
            })
        };
        if let Some(p) = promote_at {
            key = key.min(p);
        }
        Some(key)
    }

    /// Admission on compute headroom (`admission.compute_headroom`): hold a new session of `class`
    /// (of a run-ordered kind) while a class ranked at or above it has run headroom under
    /// `hold_below`. Until its wait limit, like any hold.
    fn compute_hold(&self, session: &crate::session::Session, now: Time) -> Option<Decision> {
        let cfg = self.policy.admission.compute_headroom.as_ref()?;
        if !self.run_kind(session.kind.map(crate::session::SessionKind::as_str)) {
            return None;
        }
        let me = self.policy.class(Some(&session.class));
        let short = self
            .policy
            .classes
            .iter()
            .filter(|c| c.rank <= me.rank && c.run_level.is_some())
            .filter_map(|c| Some((c, self.run_headroom(&c.name, now)?)))
            .filter(|(_, h)| *h < cfg.hold_below)
            .min_by(|a, b| a.0.rank.cmp(&b.0.rank))?;
        let until = session
            .arrived
            .after(Duration::from_secs_f64(me.admission.max_lobby_wait_s));
        (now < until).then(|| Decision::Hold {
            reason: HoldReason::ComputeHeadroom {
                class: short.0.name.clone(),
                headroom: short.1,
            },
            until,
        })
    }

    /// The session log's account of a top-level session (F35).
    pub fn experience(&self) -> &crate::experience::Experience {
        &self.experience
    }

    /// The top-level session `session` counts for: itself, or the root of its parent chain.
    pub fn root_of(&self, session: &str) -> String {
        let mut id = session;
        for _ in 0..16 {
            match self.sessions.get(id).and_then(|s| s.parent.as_deref()) {
                Some(parent) => id = parent,
                None => break,
            }
        }
        id.to_owned()
    }

    /// Sessions the client ended since the last call, with their context tokens (F34).
    pub fn take_ended(&mut self) -> Vec<(String, u64)> {
        std::mem::take(&mut self.ended)
    }

    pub fn policy(&self) -> &ServicePolicy {
        &self.policy
    }

    /// f50: the admission block with every mode's parts (the active policy's without
    /// `mix_select`). Bookkeeping that must stay current across a switch reads this.
    pub fn union_admission(&self) -> &crate::config::AdmissionPolicy {
        self.mix
            .as_ref()
            .map_or(&self.policy.admission, |m| &m.union)
    }

    /// f50: the mix selector's mode (`None` without `mix_select`).
    pub fn mix_mode(&self) -> Option<crate::mix_select::MixMode> {
        self.mix.as_ref().map(|m| m.selector.mode)
    }

    /// f50: one selector step when due: the class's share of the demand (admitted or waiting
    /// sessions), and a switch when the selector says so.
    fn mix_step(&mut self, now: Time) {
        let Some(mix) = self.mix.as_ref() else {
            return;
        };
        if !mix.selector.due(now) {
            return;
        }
        let (class, measure) = (mix.config.class.clone(), mix.config.measure);
        let mut demand = crate::mix_select::Demand::default();
        for s in self.sessions.iter() {
            if s.is_live() || s.state == SessionState::Lobby {
                demand.add(s.class == class, s.context_tokens.max(s.pending_isl));
            }
        }
        let mix = self.mix.as_mut().expect("checked");
        let to = mix
            .selector
            .step_with(now, demand.share(measure), demand.total_sessions);
        mix.log_step(now, &demand);
        if let Some(to) = to {
            mix.log_switch(now, to);
            self.apply_mix_mode(to, now);
        }
    }

    /// f50: run `to`'s policy from now on. Sessions, the lobby and every part's state are kept;
    /// the lobby is re-keyed for the new order, and the router's busy thresholds and the worker
    /// selection's mode follow.
    fn apply_mix_mode(&mut self, to: crate::mix_select::MixMode, now: Time) {
        let Some(mix) = self.mix.as_ref() else {
            return;
        };
        let policy = mix.policy(to).clone();
        crate::mix_select::apply_router_busy(mix.config.parts(to));
        crate::mix_select::set_current(Some(to));
        if let Some(mix) = self.mix.as_mut() {
            mix.selector.mode = to;
        }
        let signal = policy.admission.lobby_order == LobbyOrder::Signal;
        self.signal_cfg = policy.admission.session_signal.clone().unwrap_or_default();
        self.policy = policy;
        if !signal {
            // The signal order's keys would still order the lobby.
            let keyed: Vec<String> = self
                .sessions
                .iter()
                .filter(|s| s.order_key.is_some())
                .map(|s| s.id.clone())
                .collect();
            for id in keyed {
                self.sessions.update(&id, |s| s.order_key = None);
            }
        }
        self.sessions.set_lobby_order(lobby_keys(&self.policy));
        if signal {
            if let Some(signals) = self.signals.as_mut() {
                signals.force_refresh();
            }
            self.signal_step(now);
        }
    }

    pub fn mode(&self) -> Mode {
        self.policy.mode
    }

    pub fn sessions(&self) -> &SessionTable {
        &self.sessions
    }

    pub fn decisions(&self) -> &DecisionLog {
        &self.log
    }

    /// Whether a request of `session` arriving now should be throttled: it would open a new
    /// session while `max_lobby_sessions` are already waiting. A read apart from the log. O(1):
    /// the lobby's size is kept incrementally (ledger L65).
    pub fn throttle(&mut self, session: &str, class: Option<&str>, now: Time) -> Option<Decision> {
        let limit = self.policy.admission.max_lobby_sessions;
        if limit == 0 || self.policy.mode == Mode::Off {
            return None;
        }
        // f49c: a session holding a live ticket is never throttled.
        if self.tickets.is_live(session, now) {
            return None;
        }
        if self
            .sessions
            .get(session)
            .is_some_and(|s| s.state != SessionState::Ended)
        {
            return None;
        }
        let waiting = self.sessions.lobby_len();
        if waiting < limit {
            return None;
        }
        let decision = Decision::Throttle { waiting, limit };
        self.log.record(LoggedDecision {
            at: now,
            session: session.to_owned(),
            class: self.policy.class(class).name.clone(),
            mode: self.policy.mode,
            decision: decision.clone(),
        });
        Some(decision)
    }

    /// [`Self::throttle`] for a request of `session` with its `parent`: with `admission.tickets`
    /// `per_root` (f49e), a sub-session whose root holds a live ticket is never throttled.
    pub fn throttle_with_parent(
        &mut self,
        session: &str,
        parent: Option<&str>,
        class: Option<&str>,
        now: Time,
    ) -> Option<Decision> {
        self.throttle_with_parent_need(session, parent, class, None, now)
    }

    /// f49m: [`Self::throttle_with_parent`] with the request's KV `need` (its uncached input) for
    /// `admission.honest_retry`'s room check before `returnee_first` refuses it (`None`: none).
    pub fn throttle_with_parent_need(
        &mut self,
        session: &str,
        parent: Option<&str>,
        class: Option<&str>,
        need: Option<u64>,
        now: Time,
    ) -> Option<Decision> {
        if let Some(h) = self.honest.as_mut() {
            h.blocked = false;
        }
        if self.tickets_per_root()
            && self
                .tickets
                .is_live(&self.ticket_root(session, parent), now)
        {
            return None;
        }
        if let Some(decision) = self.throttle(session, class, now) {
            return Some(decision);
        }
        // f49f `admission.returnee_first`: a new session of a listed class waits behind its
        // class's live ticket holders (off by default).
        self.returnee_block(session, parent, class, need, now)
    }

    /// A request of `session` was classified. Returns the session's state and its class.
    pub fn on_classify(
        &mut self,
        request_id: Option<&str>,
        session: &str,
        parent: Option<&str>,
        class: Option<&str>,
        isl: u64,
        now: Time,
    ) -> (SessionState, String) {
        self.on_classify_with(
            request_id,
            session,
            parent,
            class,
            &RequestAttributes::default(),
            isl,
            now,
        )
    }

    /// [`Self::on_classify`] with what the frontend attached: the tenant, the client's signals
    /// (preferred over the history estimate), and the session end marker, which ends the session
    /// once this request completes.
    #[allow(clippy::too_many_arguments)]
    pub fn on_classify_with(
        &mut self,
        request_id: Option<&str>,
        session: &str,
        parent: Option<&str>,
        class: Option<&str>,
        attributes: &RequestAttributes,
        isl: u64,
        now: Time,
    ) -> (SessionState, String) {
        // Subagents count against their parent's class when they bring none of their own.
        let parent_class = parent
            .and_then(|p| self.sessions.get(p))
            .map(|s| s.class.clone());
        let class = self
            .policy
            .class(class.or(parent_class.as_deref()))
            .name
            .clone();
        let previous_end = self.sessions.get(session).and_then(|s| s.last_turn_end);
        let prev_context = self
            .sessions
            .get(session)
            .filter(|s| s.turns_served > 0)
            .map_or(0, |s| s.context_tokens);
        let state = self.sessions.on_turn(
            session,
            parent,
            &class,
            attributes.tenant.as_deref(),
            now,
            isl,
        );
        let root = self.root_of(session);
        if parent.is_none() {
            let kind = attributes.session_kind;
            let user = attributes.user.clone();
            self.sessions.update(session, |s| {
                if kind.is_some() {
                    s.kind = kind;
                }
                if user.is_some() {
                    s.user = user;
                }
            });
            self.experience.identify(
                &root,
                &class,
                kind.map(crate::session::SessionKind::as_str),
                attributes.user.as_deref(),
                attributes.tenant.as_deref(),
                now,
            );
        }
        // Only the root's own requests mark stop points: a subagent's first request starts with
        // its prompt as a user message.
        if parent.is_none() && (attributes.next_is_user || attributes.next_known) {
            self.experience
                .next_starts_with_user(&root, &class, now, attributes.next_is_user);
        }
        let mut order_key = None;
        if request_id.is_some() {
            self.experience.sweep(now);
            let (opened, backoff) = self.experience.start(&root, &class, now);
            let base = self
                .experience
                .get(&root)
                .and_then(|a| self.experience.allowance(&a.class))
                .map_or(0.0, |a| a.0);
            self.accrue_delay(&root, now, backoff, if opened { base } else { 0.0 });
            if state == SessionState::Lobby {
                self.experience.hold(&root);
            }
            self.absorb_runs(now);
            order_key = self.order_key(&root, &class, isl, prev_context, now);
            if let Some(key) = order_key
                && state == SessionState::Lobby
            {
                self.sessions.update(session, |s| s.order_key = Some(key));
            }
        }
        if let Some(end) = previous_end {
            // The observed think gap, the history estimate the forecast will start from.
            let gap = now.since(end).as_secs_f64();
            self.sessions
                .update(session, |s| s.signals.next_think_s = Some(gap));
        }
        let client = attributes.signals;
        if client != Signals::default() {
            // The client's own signals, when it sends them, replace the history estimate.
            self.sessions.update(session, |s| {
                s.signals.next_think_s = client.next_think_s.or(s.signals.next_think_s);
                s.signals.osl_tokens = client.osl_tokens.or(s.signals.osl_tokens);
                s.signals.next_turn_tokens = client.next_turn_tokens.or(s.signals.next_turn_tokens);
                s.signals.peak_tokens = client.peak_tokens.or(s.signals.peak_tokens);
            });
        }
        // f49q `lobby_order: signal`: a session entering the lobby takes its key now.
        if state == SessionState::Lobby && self.signal_orders(&class) {
            self.set_signal_key(session, now);
        }
        if let Some(id) = request_id {
            self.requests.insert(
                id.to_owned(),
                RequestRecord {
                    session: session.to_owned(),
                    class: class.clone(),
                    arrived: now,
                    first_token: None,
                    dispatched: false,
                    dispatched_at: None,
                    ends_session: attributes.session_final,
                    isl,
                    sent: false,
                    root,
                    prompt: None,
                    prev_context,
                    order_key,
                },
            );
        }
        // f49c: a ticket holder back in the lobby: one more return.
        if state == SessionState::Lobby && !self.tickets.is_empty() {
            // f49e `per_root`: a sub-session returns on its root's ticket.
            let key = self.ticket_key(session);
            self.tickets.returned_by(&key, session, now);
        }
        (state, class)
    }

    /// The lobby by a full scan of the table, in the shape [`crate::admit::decide`] takes. Tests
    /// use it to check the incremental lobby; the gate does not.
    pub fn lobby_scan(&self) -> Vec<LobbyEntry> {
        self.sessions
            .iter()
            .filter(|s| s.state == SessionState::Lobby)
            .map(|s| LobbyEntry {
                session: s.id.clone(),
                class: s.class.clone(),
                arrived: s.arrived,
                order_at: self.sessions.order_at(s),
                order: crate::session::order_units(s.order_key),
                isl: s.pending_isl,
                served: s.turns_served,
            })
            .collect()
    }

    /// The KV budget from the host's capacity and the session table's load.
    pub fn kv_budget(&self, hbm_tokens: u64, dram_tokens: u64) -> KvBudget {
        let (total_tokens, hot_tokens) = self.sessions.kv_load();
        let (yield_total_tokens, yield_hot_tokens) = self.sessions.yield_load();
        // f49k `admission.hot_budget: measured`: the engines' measured hot use, once measured (the
        // yield class's share of it is at most all of it).
        let (hot_tokens, yield_hot_tokens) =
            match self.hot_measure.as_ref().and_then(|m| m.hot_tokens()) {
                Some(hot) => (hot, yield_hot_tokens.min(hot)),
                None => (hot_tokens, yield_hot_tokens),
            };
        // f49o `admission.batch_yield`: the yield class's KV counts times the level (0 at rest).
        let (yield_hot_tokens, yield_total_tokens) = match self.batch_yield_discount() {
            Some(level) => (
                (yield_hot_tokens as f64 * level) as u64,
                (yield_total_tokens as f64 * level) as u64,
            ),
            None => (yield_hot_tokens, yield_total_tokens),
        };
        KvBudget {
            hbm_tokens,
            dram_tokens,
            hot_tokens,
            total_tokens,
            yield_hot_tokens,
            yield_total_tokens,
        }
    }

    /// The gate's answer for the queue head of `session`. A read: it only writes the decision log
    /// (and advances the gate-share pacing, which is a function of time).
    pub fn gate_check(&mut self, session: &str, now: Time, kv: &KvBudget) -> GateAnswer {
        self.gate_check_head(session, now, kv, HeadFacts::default())
    }

    /// [`Self::gate_check`] with what the host knows about the head (for shedding).
    pub fn gate_check_head(
        &mut self,
        session: &str,
        now: Time,
        kv: &KvBudget,
        head: HeadFacts,
    ) -> GateAnswer {
        self.gate_check_head_wake(session, now, kv, head).0
    }

    /// [`Self::gate_check_head`] with what can change a `Hold` before its `until`.
    pub fn gate_check_head_wake(
        &mut self,
        session: &str,
        now: Time,
        kv: &KvBudget,
        head: HeadFacts,
    ) -> (GateAnswer, GateWake) {
        let mode = self.policy.mode;
        if mode == Mode::Off {
            return (GateAnswer::Go, GateWake::Any);
        }
        // f50: the mix selector first, so this check runs under the mode it leaves.
        self.mix_step(now);
        self.pace_step(now);
        self.targets_step(now);
        self.loop_step(now);
        self.tickets_step(now, kv);
        self.step_target_step(now);
        self.sla_budget_step(now);
        // f49q: the signal order's keys and the override log's minute (nothing when off).
        self.signal_step(now);
        self.batch_yield_step(now);
        let Some(entry) = self.sessions.get(session) else {
            return (GateAnswer::Go, GateWake::Any);
        };
        let class = entry.class.clone();
        if entry.state != SessionState::Lobby {
            let Some(decision) = self.shed(&class, now, head) else {
                return (GateAnswer::Go, GateWake::Any);
            };
            let answer = gate_answer(mode, &decision);
            self.log.record(LoggedDecision {
                at: now,
                session: session.to_owned(),
                class,
                mode,
                decision,
            });
            return (answer, GateWake::Any);
        }
        if !self.policy.admission.session_gate || self.adaptive.as_ref().is_some_and(|a| !a.gate) {
            return (GateAnswer::Go, GateWake::Any);
        }
        let entry = self.sessions.get(session).expect("checked above");
        if self.policy.admission.hold_at_max_num_seqs && head.seq_room == Some(false) {
            let limit = self.policy.class(Some(&class)).admission.max_lobby_wait_s;
            let until = entry.arrived.after(Duration::from_secs_f64(limit));
            if now < until {
                let decision = Decision::Hold {
                    reason: HoldReason::NoSeqRoom,
                    until,
                };
                // f49m2 `honest_retry.by_reason`: the hold's reason (nothing when off).
                self.honest_note_hold(session, &decision, None);
                let answer = gate_answer(mode, &decision);
                self.log.record(LoggedDecision {
                    at: now,
                    session: session.to_owned(),
                    class,
                    mode,
                    decision,
                });
                return (answer, GateWake::Any);
            }
        }
        let scale = self.pace_scale(&class);
        // f49j `admission.step_target`: a class ranked below the target class has its hot budget
        // times the fleet's measured multiple (the HBM-weighted mean of the workers'); the total
        // budget and the classes at or above the target class are untouched.
        let scale = match &self.step_target {
            Some(step_target) => scale * step_target.scale(&class),
            None => scale,
        };
        // f49l `admission.sla_budget`: its share of the class's hot budget (exactly 1 without it).
        let scale = scale * self.sla_scale(&class);
        // f49o `admission.batch_yield`: the yield class's hot budget times 1 - level (1 at rest).
        let scale = scale * self.yield_scale(&class);
        // f49c: what the live tickets keep from this session (nothing without them).
        let ticket = self.ticket_terms(session, &class, now);
        // f49k2: the other classes' hot budgets at their own scales (`class_overcommit_mode`).
        let mut decision = decide_indexed_classes(
            &self.policy,
            now,
            entry,
            &self.sessions,
            kv,
            &|c: &crate::config::ClassPolicy| self.class_scale(c, &class, scale),
            ticket.held_kv,
            ticket.front,
        );
        // f49k `admission.hot_budget: measured`: the queue-0 guard admits a session the hot budget
        // alone holds while a worker's engine queue is empty, once per refresh of that worker.
        let hot_probe = self.hot_probe(entry, &decision);
        if hot_probe.is_some() {
            decision = Decision::Admit(crate::admit::AdmitReason::Fits);
        }
        let arrived = entry.arrived;
        // f49e `DYNAMO_TICKETS_LOG`: the step of the chain that held the session (the KV fit
        // itself, or the first hold after it), and the room the KV fit saw when it held it. Read by
        // the tickets log only.
        let held_at = |d: &Decision, step: &'static str| {
            (!matches!(d, Decision::Admit(crate::admit::AdmitReason::Fits))).then_some(step)
        };
        let mut ticket_step = match &decision {
            Decision::Admit(_) => None,
            Decision::Expire { .. } => Some("expire"),
            _ => Some("kv_fit"),
        };
        let ticket_room = match &decision {
            Decision::Hold {
                reason:
                    HoldReason::KvBudget {
                        hot_room,
                        total_room,
                        ..
                    },
                ..
            } => Some((*hot_room).min(*total_room)),
            _ => None,
        };
        // f49f `admission.returnee_first`: the live holders of a listed class are admitted in
        // the order of their first refusal (off by default).
        let mut returnee_held = false;
        if matches!(decision, Decision::Admit(crate::admit::AdmitReason::Fits))
            && let Some(hold) = self.returnee_fifo_hold(session, &class, now)
        {
            decision = hold;
            returnee_held = true;
        }
        ticket_step = ticket_step.or_else(|| held_at(&decision, "returnee_fifo"));
        if matches!(decision, Decision::Admit(crate::admit::AdmitReason::Fits))
            && let Some(hold) = self.surge_hold(&class, head.surge, arrived, now)
        {
            decision = hold;
        }
        ticket_step = ticket_step.or_else(|| held_at(&decision, "surge"));
        if matches!(decision, Decision::Admit(crate::admit::AdmitReason::Fits))
            && let Some(hold) = self
                .sessions
                .get(session)
                .and_then(|entry| self.compute_hold(entry, now))
        {
            decision = hold;
        }
        ticket_step = ticket_step.or_else(|| held_at(&decision, "compute"));
        if matches!(decision, Decision::Admit(crate::admit::AdmitReason::Fits))
            && let Some(hold) = self.onboard_hold(&class, arrived, now)
        {
            decision = hold;
        }
        ticket_step = ticket_step.or_else(|| held_at(&decision, "onboard"));
        if matches!(decision, Decision::Admit(crate::admit::AdmitReason::Fits))
            && let Some(hold) = self.infeasible_hold(&class, arrived, now)
        {
            decision = hold;
        }
        ticket_step = ticket_step.or_else(|| held_at(&decision, "infeasible"));
        if matches!(decision, Decision::Admit(crate::admit::AdmitReason::Fits))
            && let Some(hold) = self.ttft_hold(&class, arrived, now)
        {
            decision = hold;
        }
        ticket_step = ticket_step.or_else(|| held_at(&decision, "ttft"));
        if matches!(decision, Decision::Admit(crate::admit::AdmitReason::Fits)) {
            let hold = self.harm_hold(&class, head.new_harm, arrived, now);
            // f49d `DYNAMO_HARM_DEBUG`: the check reached with the decision still `Fits`.
            if self.policy.admission.forecast_harm_new.is_some()
                && let Some(d) = self.harm_debug.as_mut()
            {
                d.reached(&class, hold.as_ref());
            }
            if let Some(hold) = hold {
                decision = hold;
            }
        }
        ticket_step = ticket_step.or_else(|| held_at(&decision, "harm"));
        if matches!(decision, Decision::Admit(crate::admit::AdmitReason::Fits)) {
            let isl = self.sessions.get(session).map_or(0, |s| s.pending_isl);
            // f47h: with `meter: uncached` the onboarding term is the tokens the turn will compute.
            let isl = if self.meter_uncached() {
                self.session_compute.get(session).copied().unwrap_or(isl)
            } else {
                isl
            };
            // f49c: the onboarding of the live tickets ahead counts with its own.
            let isl = isl.saturating_add(ticket.held_prefill);
            if let Some(hold) = self.prefill_hold(&class, isl, head.prefill, arrived, now) {
                decision = hold;
            }
        }
        ticket_step = ticket_step.or_else(|| held_at(&decision, "prefill"));
        // f49q `admission.starvation_override`: a starved session of a listed class goes past its
        // class's own budgets and holds (nothing when off).
        if !matches!(decision, Decision::Admit(_))
            && let Some(d) =
                self.starvation_override(session, &class, kv, ticket.held_kv, head.seq_room, now)
        {
            decision = d;
        }
        let entry = self.sessions.get(session).expect("checked above");
        // A session with no deadline to expire it (classified while the adaptive rule had the gate
        // off) would wait for good at its wait limit: admit it instead.
        if matches!(decision, Decision::Expire { .. }) && entry.no_lobby_deadline {
            decision = Decision::Admit(crate::admit::AdmitReason::WaitLimit);
        }
        // f47: a refused session of a guaranteed class is not left behind: after enough refusals it
        // is admitted at its wait limit when it fits, whatever the order of earlier sessions.
        if matches!(decision, Decision::Expire { .. })
            && let Some(g) = self.policy.admission.returnee_guarantee.as_ref()
            && g.classes.contains(&class)
            && self.refusals.get(session).copied().unwrap_or(0) >= g.after_refusals
        {
            let d = crate::admit::fits_ignoring_order_classes(
                &self.policy,
                now,
                entry,
                &self.sessions,
                kv,
                &|c: &crate::config::ClassPolicy| self.class_scale(c, &class, scale),
                ticket.held_kv,
            );
            if matches!(d, Decision::Admit(_)) {
                decision = d;
            }
        }
        if self
            .policy
            .admission
            .retry_after_estimate
            .as_ref()
            .is_some_and(|e| e.calibrate)
            && self.ra_waiting.get(session).is_some_and(|(_, back)| *back)
        {
            // The first check since its last refusal: one more return.
            let admitted = matches!(decision, Decision::Admit(_));
            let returns = {
                let e = self.ra_waiting.get_mut(session).expect("checked");
                e.0 += 1;
                e.1 = false;
                e.0
            };
            let f = self.ra_factor.entry(class.clone()).or_insert(1.0);
            if admitted {
                if returns <= 3 {
                    *f = (*f * 0.95).max(1.0);
                }
                self.ra_waiting.remove(session);
            } else if returns == 3 {
                *f = (*f * 1.25).min(20.0);
            }
        }
        // f49k: an admission on the queue-0 guard uses up that worker's until its next refresh.
        if let Some(worker) = hot_probe
            && matches!(decision, Decision::Admit(_))
            && let Some(m) = self.hot_measure.as_mut()
        {
            m.use_probe(worker);
        }
        let answer = gate_answer(mode, &decision);
        // f49f: a hold behind an older ticket holder (`returnee_first`) can lift without a lobby
        // change (that ticket lapsing): it asks again at every recheck.
        let wake = match (&answer, &decision) {
            (
                GateAnswer::Hold { .. },
                Decision::Hold {
                    reason: HoldReason::BehindEarlier { .. },
                    ..
                },
            ) if !returnee_held => self.behind_for_good(entry),
            _ => GateWake::Any,
        };
        // f49e `DYNAMO_TICKETS_LOG`: a live ticket holder not admitted (log only).
        if self.tickets.logging() {
            self.ticket_hold_diag(
                session,
                &class,
                &decision,
                ticket_step,
                ticket_room,
                ticket.held_kv,
                now,
            );
        }
        // f49m2 `honest_retry.by_reason`: the session's latest hold and the step that held it
        // (nothing when off).
        self.honest_note_hold(session, &decision, ticket_step);
        self.log.record(LoggedDecision {
            at: now,
            session: session.to_owned(),
            class,
            mode,
            decision,
        });
        (answer, wake)
    }

    /// f49d `DYNAMO_HARM_DEBUG=1`: whether the forecast-harm check's counters are on.
    pub fn harm_debug_on(&self) -> bool {
        self.harm_debug.is_some()
    }

    /// Turn the forecast-harm check's counters on (the scheduler reads the environment when built).
    pub fn enable_harm_debug(&mut self) {
        self.harm_debug.get_or_insert_with(Default::default);
    }

    pub fn harm_debug(&self) -> Option<&crate::harm_debug::HarmDebug> {
        self.harm_debug.as_ref()
    }

    /// f49d: count one gate check's input to the forecast-harm check (the host's `new_harm`), and
    /// print the summary line to stderr when one is due. Nothing when the counters are off.
    pub fn harm_debug_check(
        &mut self,
        class: &str,
        check: crate::harm_debug::HarmCheck,
        now: Time,
    ) {
        let Some(d) = self.harm_debug.as_mut() else {
            return;
        };
        d.check(class, check);
        if let Some(line) = d.due(now) {
            eprintln!("{line}");
        }
    }

    /// The forecast-harm check on new sessions (`admission.forecast_harm_new`, ks2b; `web/sim.js`
    /// `admitLobby` with `gateHarm`): hold a new session of a class not exempt while its best
    /// placement adds more than `max_added_misses` weighted misses to its class and the classes
    /// above it (`harm`, from the host). Until its wait limit, like any hold; there the class's
    /// `on_wait_limit` decides, as the simulator admits priority and turns the others away.
    fn harm_hold(
        &self,
        class: &str,
        harm: Option<f64>,
        arrived: Time,
        now: Time,
    ) -> Option<Decision> {
        let cfg = self.policy.admission.forecast_harm_new.as_ref()?;
        if cfg.exempt_classes.iter().any(|c| c == class) {
            return None;
        }
        let added = harm?;
        if added <= cfg.max_added_misses {
            return None;
        }
        let me = self.policy.class(Some(class));
        let until = arrived.after(Duration::from_secs_f64(me.admission.max_lobby_wait_s));
        if now >= until {
            return match me.admission.on_wait_limit {
                crate::config::WaitLimitAction::Admit => None,
                crate::config::WaitLimitAction::Expire => Some(Decision::Expire {
                    waited: now.since(arrived),
                }),
            };
        }
        Some(Decision::Hold {
            reason: HoldReason::ForecastHarm {
                added,
                limit: cfg.max_added_misses,
            },
            until,
        })
    }

    /// The TTFT cap (`admission.ttft_cap`, f47): hold a new session of a listed class while the
    /// class's measured TTFT quantile is over its target; at its wait limit its `on_wait_limit`
    /// decides (expire: refused; admit: let in).
    fn ttft_hold(&self, class: &str, arrived: Time, now: Time) -> Option<Decision> {
        let cfg = self.policy.admission.ttft_cap.as_ref()?;
        if !cfg.classes.iter().any(|c| c == class) {
            return None;
        }
        let me = self.policy.class(Some(class));
        let target = me.service_level.as_ref()?.ttft_s;
        let q = self
            .metrics
            .ttft_quantile(class, now, cfg.window_s, cfg.quantile)?;
        if q <= target * cfg.factor {
            return None;
        }
        let until = arrived.after(Duration::from_secs_f64(me.admission.max_lobby_wait_s));
        if now >= until {
            return match me.admission.on_wait_limit {
                crate::config::WaitLimitAction::Admit => None,
                crate::config::WaitLimitAction::Expire => Some(Decision::Expire {
                    waited: now.since(arrived),
                }),
            };
        }
        Some(Decision::Hold {
            reason: HoldReason::ComputeHeadroom {
                class: class.to_owned(),
                headroom: target - q,
            },
            until,
        })
    }

    /// The two-resource gate (`admission.prefill_budget`, f42): hold a new session of a class
    /// below the top-ranked one while the fleet's measured prefill rate `used` plus its onboarding
    /// (`isl` over the window) is over the budget times the class's target-loop scale. Until its
    /// wait limit, like any hold.
    fn prefill_hold(
        &mut self,
        class: &str,
        isl: u64,
        prefill: Option<(f64, f64)>,
        arrived: Time,
        now: Time,
    ) -> Option<Decision> {
        let cfg_pb = self.policy.admission.prefill_budget.as_ref()?;
        let (window_s, include_top) = (cfg_pb.window_s, cfg_pb.include_top);
        let (budget, used) = prefill?;
        let used = match self.lookahead_rate(now) {
            Some(p) => used.max(p),
            None => used,
        };
        let me = self.policy.class(Some(class));
        let top = self
            .policy
            .classes
            .iter()
            .map(|c| c.rank)
            .min()
            .unwrap_or(0);
        let is_top = me.rank <= top;
        if is_top && !include_top {
            return None;
        }
        self.lp.judged += 1;
        let own = isl as f64 / window_s;
        let budget = budget * self.loop_scale(class);
        // Work-conserving: a session whose onboarding alone outsizes the room still starts once the
        // fleet's rate is under half the budget (one large onboarding at a time, not never).
        if used + own <= budget || used <= 0.5 * budget {
            return None;
        }
        let until = arrived.after(Duration::from_secs_f64(me.admission.max_lobby_wait_s));
        if now >= until {
            // The top class under its own budget (include_top) is refused at its wait limit while
            // still over the budget, rather than let in past it.
            if is_top {
                return Some(Decision::Expire {
                    waited: now.since(arrived),
                });
            }
            return None;
        }
        self.lp.held += 1;
        Some(Decision::Hold {
            reason: HoldReason::PrefillBudget { used, own, budget },
            until,
        })
    }

    /// The look-ahead's predicted prefill rate (`prefill_budget.lookahead_s`), recomputed every
    /// `lookahead_step_s`: the new tokens of the idle sessions' next turns due within the horizon
    /// (overdue ones count as due now), over the horizon. O(sessions) per step.
    fn lookahead_rate(&mut self, now: Time) -> Option<f64> {
        let cfg = self.policy.admission.prefill_budget.as_ref()?;
        let horizon = cfg.lookahead_s?;
        if self
            .lp
            .predicted_at
            .is_some_and(|at| now.since(at).as_secs_f64() < cfg.lookahead_step_s)
            && self.lp.predicted.is_some()
        {
            return self.lp.predicted;
        }
        // A next turn is due at the last turn's end plus the think estimate. Sessions with no think
        // estimate yet, and ones overdue by more than the horizon (between runs, or gone), are not
        // counted: counting every overdue idle session as due now overstated the rate several times
        // (f42 bench B1: predicted 100k to 200k tok/s against 10k to 50k measured).
        let (lo, hi) = (now.as_secs_f64() - horizon, now.as_secs_f64() + horizon);
        let mut tokens = 0.0;
        for s in self.sessions.iter() {
            if s.state != SessionState::Idle {
                continue;
            }
            let (Some(last), Some(think)) = (s.last_turn_end, s.signals.next_think_s) else {
                continue;
            };
            let due = last.as_secs_f64() + think;
            if !(lo..=hi).contains(&due) {
                continue;
            }
            let next = s
                .signals
                .next_turn_tokens
                .unwrap_or_else(|| s.context_tokens / u64::from(s.turns_served.max(1)));
            tokens += next as f64;
        }
        // Due within the horizon either side of now: spread over both.
        let rate = tokens / (2.0 * horizon.max(1.0));
        self.lp.predicted = Some(rate);
        self.lp.predicted_at = Some(now);
        Some(rate)
    }

    /// The prefill budget's calibration factor (1 without `prefill_budget.calibrate`), moved every
    /// `lookahead_step_s` by the top class's ITL miss share against what its target allows.
    pub fn prefill_cal(&mut self, now: Time) -> f64 {
        let Some(cfg) = self.policy.admission.prefill_budget.clone() else {
            return 1.0;
        };
        if !cfg.calibrate {
            return 1.0;
        }
        let Some(at) = self.lp.cal_at else {
            self.lp.cal_at = Some(now);
            return self.lp.cal;
        };
        let dt = now.since(at).as_secs_f64();
        if dt < cfg.lookahead_step_s {
            return self.lp.cal;
        }
        self.lp.cal_at = Some(now);
        let top = self
            .policy
            .classes
            .iter()
            .filter(|c| c.service_level.is_some())
            .min_by_key(|c| c.rank)
            .map(|c| {
                (
                    c.name.clone(),
                    c.service_level.as_ref().map_or(0.9, |l| l.attainment),
                )
            });
        let Some((class, target)) = top else {
            return self.lp.cal;
        };
        let miss = self.metrics.itl_miss_share(&class, now);
        self.lp.itl_miss = miss;
        if let Some(miss) = miss {
            let err = (1.0 - target) - miss;
            let gain = if err < 0.0 { cfg.cal_down } else { cfg.cal_up };
            self.lp.cal =
                (self.lp.cal * (gain * err * dt / 60.0).exp()).clamp(cfg.cal_min, cfg.cal_max);
        }
        self.lp.cal
    }

    /// The target loop's scale for `class` (`admission.target_loop`, f42): the loop's scale for
    /// the loop's class and the classes ranked below it, 1 otherwise and without the loop.
    pub fn loop_scale(&self, class: &str) -> f64 {
        let Some(cfg) = self.policy.admission.target_loop.as_ref() else {
            return 1.0;
        };
        let at = self.policy.class(Some(&cfg.class)).rank;
        if self.policy.class(Some(class)).rank >= at {
            let cut = if self.lp.higher_waiting {
                cfg.priority_wait_cut
            } else {
                1.0
            };
            // f49n `rebalance`: the class's own scale (f49q4 `demand_first`: ordered).
            let scale = match cfg.rebalance.as_ref() {
                Some(rc) if rc.demand_first() => self.loop_rb_bounded(cfg, class),
                Some(_) => self.lp.rb.scale.get(class).copied().unwrap_or(1.0),
                None => self.lp.scale,
            };
            scale * cut
        } else {
            1.0
        }
    }

    /// The target loop's state for the run log: (scale, smoothed attainment by class, lobby
    /// checks the prefill budget judged and held since the last call).
    pub fn take_loop_stats(&mut self) -> LoopStats {
        let out = LoopStats {
            scale: self.lp.scale,
            att: self.lp.att.clone(),
            targets: self
                .policy
                .classes
                .iter()
                .filter_map(|c| {
                    c.service_level
                        .as_ref()
                        .map(|l| (c.name.clone(), l.attainment))
                })
                .collect(),
            judged: self.lp.judged,
            held: self.lp.held,
            infeasible: self.lp.infeasible.clone(),
            infeasible_s: self.lp.infeasible_s,
            infeasible_held: self.lp.infeasible_held,
            predicted: self.lp.predicted,
            cal: self.lp.cal,
            itl_miss: self.lp.itl_miss,
            rb: self.loop_rb_stats(),
        };
        self.lp.judged = 0;
        self.lp.held = 0;
        self.lp.infeasible_s = 0.0;
        self.lp.infeasible_held = 0;
        out
    }

    /// The target loop cannot meet a higher class's target (`target_loop.infeasible_after_s`): hold
    /// a new session of the loop's class or a lower one until its wait limit.
    fn infeasible_hold(&mut self, class: &str, arrived: Time, now: Time) -> Option<Decision> {
        let cfg = self.policy.admission.target_loop.as_ref()?;
        if !cfg.infeasible_hold {
            return None;
        }
        let short = self.lp.infeasible.clone()?;
        let at = self.policy.class(Some(&cfg.class)).rank;
        let me = self.policy.class(Some(class));
        if me.rank < at {
            return None;
        }
        let until = arrived.after(Duration::from_secs_f64(me.admission.max_lobby_wait_s));
        if now >= until {
            return None;
        }
        self.lp.infeasible_held += 1;
        Some(Decision::Hold {
            reason: HoldReason::TargetInfeasible { class: short },
            until,
        })
    }

    /// One step of the target loop when `step_s` has passed: smooth each class's attainment,
    /// then move the scale by the loop class's error from its target plus `margin`, or by a
    /// higher class's shortfall from its target plus `guard` (times `guard_gain`) when that is
    /// lower.
    fn loop_step(&mut self, now: Time) {
        let Some(cfg) = self.policy.admission.target_loop.clone() else {
            return;
        };
        let Some(last) = self.lp.last else {
            self.lp.last = Some(now);
            return;
        };
        let dt = now.since(last).as_secs_f64();
        if dt < cfg.step_s {
            return;
        }
        self.lp.last = Some(now);
        if cfg.priority_wait_cut < 1.0 {
            let at = self.policy.class(Some(&cfg.class)).rank;
            let waiting = self.sessions.iter().any(|x| {
                x.state == SessionState::Lobby && self.policy.class(Some(&x.class)).rank < at
            });
            self.lp.higher_waiting = waiting;
        }
        let elapsed = now.since_origin().as_secs_f64();
        while let Some(change) = self
            .policy
            .admission
            .target_schedule
            .get(self.lp.applied)
            .cloned()
            && change.at_s <= elapsed
        {
            self.lp.applied += 1;
            if let Some(level) = self
                .policy
                .classes
                .iter_mut()
                .find(|c| c.name == change.class)
                .and_then(|c| c.service_level.as_mut())
            {
                level.attainment = change.attainment;
            }
        }
        let alpha = 1.0 - (-dt / cfg.smooth_s).exp();
        let levels: Vec<(String, u32, f64)> = self
            .policy
            .classes
            .iter()
            .filter_map(|c| {
                c.service_level
                    .as_ref()
                    .map(|l| (c.name.clone(), c.rank, l.attainment))
            })
            .collect();
        for (name, _, _) in &levels {
            if let Some(a) = self.attainment(name, now) {
                let e = self.lp.att.entry(name.clone()).or_insert(a);
                *e += alpha * (a - *e);
            }
        }
        let me = self.policy.class(Some(&cfg.class)).rank;
        if self.lp.infeasible.is_some() {
            self.lp.infeasible_s += dt;
        }
        // Best effort: the protected level ratchets up as the higher class recovers, and with
        // `reprobe_s` the state is cleared to test the real target again.
        if !cfg.infeasible_hold
            && let Some(class) = self.lp.infeasible.clone()
        {
            if let Some(a) = self.lp.att.get(&class).copied() {
                self.lp.floor_att = Some(self.lp.floor_att.map_or(a, |f| f.max(a)));
            }
            if cfg.reprobe_s.is_some_and(|r| {
                self.lp
                    .declared_at
                    .is_some_and(|at| now.since(at).as_secs_f64() >= r)
            }) {
                self.lp.infeasible = None;
                self.lp.floor_att = None;
                self.lp.declared_at = None;
                self.lp.short_since = None;
                self.lp.guard_released = false;
            }
        }
        // Yield to demand: grow only while a session of the loop's class waits or was turned away.
        let lost = self.lost.get(&cfg.class).copied().unwrap_or(0);
        let demand = self.sessions.lobby().has_class(&cfg.class) || lost != self.lp.lost_seen;
        self.lp.lost_seen = lost;
        let mut rate = levels
            .iter()
            .find(|(n, _, _)| *n == cfg.class)
            .and_then(|(n, _, t)| {
                match self.lp.att.get(n) {
                    Some(a) => {
                        let e = a - (t + cfg.margin);
                        Some(if e > 0.0 {
                            if demand {
                                cfg.gain * cfg.up_factor * e
                            } else {
                                0.0
                            }
                        } else {
                            cfg.gain * e
                        })
                    }
                    // Nothing served to judge while sessions wait: under-admitted, grow slowly.
                    None if demand => Some(cfg.gain * cfg.up_factor * 0.1),
                    None => None,
                }
            })
            .unwrap_or(0.0);
        // At the floor for longer than infeasible_after_s, a higher class that still meets its target
        // but not its guard is at what the lower classes can give: its guard stops pulling (it would
        // only starve the lower classes); under its target, the infeasible state below takes over.
        if self
            .lp
            .floor_since
            .is_some_and(|at| now.since(at).as_secs_f64() >= cfg.infeasible_after_s)
        {
            self.lp.guard_released = true;
        }
        // Released, the guard stays off until a higher class falls under its target itself.
        if levels
            .iter()
            .any(|(n, r, t)| *r < me && self.lp.att.get(n).is_some_and(|a| a < t))
        {
            self.lp.guard_released = false;
        }
        let floor_long = self.lp.guard_released;
        // A higher class under its target itself (not just its guard) with the scale at its floor.
        let mut below: Option<String> = None;
        for (name, rank, target) in &levels {
            if *rank < me
                && let Some(a) = self.lp.att.get(name)
            {
                let best_effort =
                    !cfg.infeasible_hold && self.lp.infeasible.as_deref() == Some(name);
                let short = match (best_effort, self.lp.floor_att) {
                    (true, Some(f)) => a - (f - 0.02),
                    _ => a - (target + cfg.guard),
                };
                if short < 0.0 && !(floor_long && *a >= *target) {
                    rate = rate.min(cfg.guard_gain * short);
                }
                if *a < *target && below.is_none() {
                    below = Some(name.clone());
                }
            }
        }
        self.lp.scale =
            (self.lp.scale * (rate * dt / 60.0).exp()).clamp(cfg.min_scale, cfg.max_scale);
        // f49n `rebalance`: the per-class scales step, and the floor the infeasible state reads is
        // theirs (every class at or below the loop's at its own floor).
        let floor = match cfg.rebalance.as_ref() {
            Some(rc) => self.loop_rebalance(&cfg, rc, &levels, now, dt),
            None => self.lp.scale <= cfg.min_scale * (1.0 + 1e-9),
        };
        if floor {
            self.lp.floor_since.get_or_insert(now);
        } else {
            self.lp.floor_since = None;
        }
        match below {
            // Best effort keeps the state while the higher class stays under its target, wherever
            // the scale goes (the lower classes are served again).
            Some(class)
                if !cfg.infeasible_hold && self.lp.infeasible.as_deref() == Some(&class) => {}
            Some(class) if floor => {
                let since = *self.lp.short_since.get_or_insert(now);
                if now.since(since).as_secs_f64() >= cfg.infeasible_after_s {
                    if self.lp.infeasible.is_none() {
                        self.lp.floor_att = self.lp.att.get(&class).copied();
                        self.lp.declared_at = Some(now);
                    }
                    self.lp.infeasible = Some(class);
                }
            }
            _ => {
                self.lp.short_since = None;
                self.lp.infeasible = None;
                self.lp.floor_att = None;
                self.lp.declared_at = None;
            }
        }
    }

    /// f49n `target_loop.rebalance`: one step of the per-class scales
    /// ([`crate::config::LoopRebalanceConfig`]) through f49l3's transfer step
    /// ([`crate::transfer`]); whether every class at or below the loop's class is at its floor.
    fn loop_rebalance(
        &mut self,
        cfg: &crate::config::TargetLoopConfig,
        rc: &crate::config::LoopRebalanceConfig,
        levels: &[(String, u32, f64)],
        now: Time,
        dt: f64,
    ) -> bool {
        use crate::transfer::{down, factor, give_way, share, up};
        let at = self.policy.class(Some(&cfg.class)).rank;
        // The classes it steers, by rank, with their own targets.
        let mut lower: Vec<(String, u32, Option<f64>)> = self
            .policy
            .classes
            .iter()
            .filter(|c| c.rank >= at)
            .map(|c| {
                (
                    c.name.clone(),
                    c.rank,
                    c.service_level.as_ref().map(|l| l.attainment),
                )
            })
            .collect();
        lower.sort_by_key(|(_, r, _)| *r);
        let top = levels
            .iter()
            .filter(|(_, r, _)| *r < at)
            .min_by_key(|(_, r, _)| *r)
            .map(|(n, _, t)| (n.clone(), *t));
        // Yield to demand, per class: a session waiting, or one turned away since the last step.
        let mut demand: HashMap<String, bool> = HashMap::new();
        for (name, _, _) in &lower {
            let lost = self.lost.get(name).copied().unwrap_or(0);
            let seen = self.lp.rb.lost_seen.insert(name.clone(), lost).unwrap_or(0);
            let waiting = self.sessions.lobby().has_class(name);
            demand.insert(name.clone(), waiting || lost != seen);
        }
        // f49l2's warm-up, latched.
        let t_s = now.since_origin().as_secs_f64();
        for name in top
            .iter()
            .map(|(n, _)| n)
            .chain(lower.iter().map(|(n, _, _)| n))
        {
            if self.lp.rb.warm.contains(name) {
                continue;
            }
            let has_level = self.policy.class(Some(name)).service_level.is_some();
            let scored = self.metrics.window_counts(name, now).map(|(n, _)| n);
            if rc.is_warm(t_s, has_level, scored) {
                self.lp.rb.warm.insert(name.clone());
            }
        }
        let warm = |c: &str| self.lp.rb.warm.contains(c);
        let top_warm = top.as_ref().is_some_and(|(n, _)| warm(n.as_str()));
        let may = |c: &str| top_warm && warm(c);
        let k = |c: &str| self.lp.rb.scale.get(c).copied().unwrap_or(1.0);
        let caps = rc.rate_caps();
        // f49q4 `release_order: demand_first`: a release's ceiling for `c` (rank `rank`) is also
        // the scale of every class above it down to the loop's.
        let demand_first = rc.demand_first();
        let ceiling = |c: &str, rank: u32| {
            if !demand_first {
                return rc.ceiling(c);
            }
            lower
                .iter()
                .filter(|(_, r, _)| *r < rank)
                .fold(rc.ceiling(c), |m, (o, _, _)| m.min(k(o.as_str())))
        };
        // The top class's head: against its target + margin, or in best effort once its target is
        // declared infeasible, against the guard's protected level.
        let head = top.as_ref().and_then(|(name, target)| {
            let reference = match self.lp.floor_att {
                Some(f)
                    if !cfg.infeasible_hold
                        && self.lp.infeasible.as_deref() == Some(name.as_str()) =>
                {
                    f - 0.02
                }
                _ => target + rc.margin,
            };
            self.lp.att.get(name).map(|a| a - reference)
        });
        let mut moved: Option<(String, f64)> = None;
        let mode = match head {
            None => "hold",
            Some(_) if !top_warm => "warmup",
            Some(h) if h > 0.0 => {
                // Release: the highest-ranked class with demand, under its ceiling and under its
                // own target + margin (f49q4 `demand_first`: whatever its own attainment).
                let wants = |c: &str, target: Option<f64>| {
                    demand_first
                        || target.is_none_or(|t| {
                            self.lp.att.get(c).is_none_or(|a| *a < t + rc.margin_of(c))
                        })
                };
                let taker = lower.iter().find(|(c, r, t)| {
                    let c = c.as_str();
                    may(c)
                        && demand.get(c).copied().unwrap_or(false)
                        && k(c) < ceiling(c, *r)
                        && wants(c, *t)
                });
                match taker {
                    Some((c, r, _)) => {
                        let c = c.as_str();
                        let f = factor(caps, rc.release_per_min, share(h, rc.full_at), dt);
                        moved = Some((c.to_owned(), up(k(c), f, ceiling(c, *r))));
                        "release"
                    }
                    None => "hold",
                }
            }
            Some(h) if h < 0.0 => {
                // Pull back: the lowest-ranked class above its floor gives way.
                let order: Vec<&str> = lower.iter().rev().map(|(c, _, _)| c.as_str()).collect();
                match give_way(&order, k, |c| rc.floor(c)) {
                    Some(c) if may(c) => {
                        let f = factor(caps, -rc.pull_per_min, share(h, rc.full_at), dt);
                        moved = Some((c.to_owned(), down(k(c), f, rc.floor(c))));
                        "pull"
                    }
                    Some(_) => "warmup",
                    None => "floor",
                }
            }
            Some(_) => "hold",
        };
        if let Some((c, v)) = moved {
            self.lp.rb.scale.insert(c, v);
        }
        // f49q4 `demand_first`: no class's scale over that of a class ranked above it (by rank,
        // so a pull of standard takes batch down with it).
        if demand_first {
            for (c, r, _) in &lower {
                let above = lower
                    .iter()
                    .filter(|(_, o, _)| o < r)
                    .map(|(o, _, _)| self.lp.rb.scale.get(o).copied().unwrap_or(1.0))
                    .fold(f64::INFINITY, f64::min);
                if self.lp.rb.scale.get(c).copied().unwrap_or(1.0) > above {
                    self.lp.rb.scale.insert(c.clone(), above);
                }
            }
        }
        self.lp.rb.mode = mode;
        self.lp.rb.head = head;
        lower.iter().all(|(c, _, _)| {
            self.lp.rb.scale.get(c).copied().unwrap_or(1.0) <= rc.floor(c) * (1.0 + 1e-9)
        })
    }

    /// f49n: `target_loop.rebalance`'s figures for the run log (`None` without it).
    fn loop_rb_stats(&self) -> Option<LoopRbStats> {
        let cfg = self.policy.admission.target_loop.as_ref()?;
        cfg.rebalance.as_ref()?;
        let mut classes: Vec<&crate::config::ClassPolicy> = self.policy.classes.iter().collect();
        classes.sort_by_key(|c| c.rank);
        let mode = if self.lp.rb.mode.is_empty() {
            "hold"
        } else {
            self.lp.rb.mode
        };
        Some(LoopRbStats {
            mode: mode.to_owned(),
            head: self.lp.rb.head,
            classes: classes
                .iter()
                .map(|c| {
                    (
                        c.name.clone(),
                        self.loop_gate_scale(&c.name),
                        self.lp.att.get(&c.name).copied(),
                        c.service_level.as_ref().map(|l| l.attainment),
                    )
                })
                .collect(),
        })
    }

    /// Whether onboarding pace (`admission.onboard_pace`) applies to `class`.
    fn onboard_paces(&self, class: &str) -> bool {
        self.policy
            .admission
            .onboard_pace
            .as_ref()
            .is_some_and(|p| p.classes.is_empty() || p.classes.iter().any(|c| c == class))
    }

    /// Refill the onboarding bucket up to `now`.
    fn onboard_refill(&mut self, now: Time) {
        let Some(cfg) = self.policy.admission.onboard_pace.as_ref() else {
            return;
        };
        if let Some(at) = self.onboard_at
            && now > at
        {
            let dt = now.since(at).as_secs_f64();
            self.onboard_level = (self.onboard_level + dt * cfg.tokens_per_s).min(cfg.burst_tokens);
        }
        if self.onboard_at.is_none_or(|at| now > at) {
            self.onboard_at = Some(now);
        }
    }

    /// Onboarding pace (`admission.onboard_pace`): hold a new session of a paced class while the
    /// onboarding bucket is in debt. Until its wait limit, like any hold.
    fn onboard_hold(&mut self, class: &str, arrived: Time, now: Time) -> Option<Decision> {
        if !self.onboard_paces(class) {
            return None;
        }
        self.onboard_refill(now);
        if self.onboard_level >= 0.0 {
            return None;
        }
        let me = self.policy.class(Some(class));
        let until = arrived.after(Duration::from_secs_f64(me.admission.max_lobby_wait_s));
        (now < until).then_some(Decision::Hold {
            reason: HoldReason::OnboardBudget {
                level: self.onboard_level,
            },
            until,
        })
    }

    /// A new session of a paced class left the lobby: its first turn's input tokens come out of
    /// the onboarding bucket.
    fn onboard_charge(&mut self, class: &str, tokens: u64, now: Time) {
        if !self.onboard_paces(class) {
            return;
        }
        self.onboard_refill(now);
        self.onboard_level -= tokens as f64;
    }

    /// The onboarding bucket's level in tokens (negative in debt), for tests and logs.
    pub fn onboard_level(&self) -> f64 {
        self.onboard_level
    }

    /// Surge admission (`admission.surge`): hold a new session of `class` while the fleet's
    /// queued prefill takes longer than the tightest TTFT limit among the classes ranked above it
    /// (times `factor`). The top-ranked class, and a class with no class above it
    /// that has a TTFT limit, is never held. Until its wait limit, like any hold.
    /// Surge admission's counts by class (lobby checks judged, held) since the last call.
    pub fn take_surge_counts(&mut self) -> std::collections::BTreeMap<String, (u64, u64)> {
        std::mem::take(&mut self.surge_counts)
    }

    fn surge_hold(
        &mut self,
        class: &str,
        surge: Option<(f64, f64)>,
        arrived: Time,
        now: Time,
    ) -> Option<Decision> {
        let cfg = self.policy.admission.surge.as_ref()?;
        if !crate::coordinator::surge_admission_enabled() {
            return None;
        }
        let (backlog_s, own_s) = surge?;
        let me = self.policy.class(Some(class));
        let tightest = self
            .policy
            .classes
            .iter()
            .filter(|c| c.rank < me.rank)
            .filter_map(|c| c.service_level.as_ref().map(|l| l.ttft_s))
            .fold(f64::INFINITY, f64::min);
        let limit_s = cfg.factor * tightest;
        let counts = self.surge_counts.entry(class.to_owned()).or_default();
        counts.0 += 1;
        // The queued prefill alone: a head's own prefill is not counted, or a session whose first
        // turn alone outlasts the limit (a long agentic context) would wait out every hold, be
        // refused, retry and wait again (F22's trace: 1,040 users, standard held all run).
        let _ = own_s;
        if !tightest.is_finite() || backlog_s <= limit_s {
            return None;
        }
        let until = arrived.after(Duration::from_secs_f64(me.admission.max_lobby_wait_s));
        if now < until {
            self.surge_counts.entry(class.to_owned()).or_default().1 += 1;
        }
        (now < until).then_some(Decision::Hold {
            reason: HoldReason::PrefillBacklog { backlog_s, limit_s },
            until,
        })
    }

    /// For a lobby session that waits behind an earlier one: the earlier session it waits behind
    /// whatever else changes before its wait limit. In first-come or fair order, the nearest
    /// session ahead in its class (the lobby is ordered by fixed keys, and the session stays
    /// behind it until it leaves or moves); in tenant order, the nearest one ahead of the same
    /// tenant (the tenants' loads order only the first session of each tenant). Other holds
    /// depend on budgets, loads or time ([`GateWake::Any`]).
    fn behind_for_good(&self, entry: &crate::session::Session) -> GateWake {
        let Some(key) = self.sessions.lobby_key(entry) else {
            return GateWake::Any;
        };
        let lobby = self.sessions.lobby();
        let ahead = if self.policy.admission.lobby_order == LobbyOrder::Tenant {
            lobby.earlier_in_tenant(&entry.class, entry.tenant.as_deref().unwrap_or(""), &key)
        } else {
            lobby.earlier_in_class(&entry.class, &key)
        };
        ahead.map_or(GateWake::Any, |ahead| GateWake::Behind {
            ahead: ahead.session.clone(),
        })
    }

    /// Record which sessions leave or move in the lobby, for [`GateWake::Behind`] holds.
    pub fn track_lobby_changes(&mut self) {
        self.sessions.track_lobby_changes();
    }

    /// The lobby changes since the last call (see [`SessionTable::take_lobby_changes`]).
    pub fn take_lobby_changes(&mut self) -> crate::session::LobbyChanges {
        self.sessions.take_lobby_changes()
    }

    /// Record that a lobby session waits without a deadline (see `Session::no_lobby_deadline`).
    pub fn mark_no_lobby_deadline(&mut self, session: &str) {
        self.sessions
            .update(session, |s| s.no_lobby_deadline = true);
    }

    /// Set the adaptive rule's gate group (tests).
    #[cfg(test)]
    pub(crate) fn set_adaptive_gate(&mut self, on: bool) {
        if let Some(a) = self.adaptive.as_mut() {
            a.gate = on;
        }
    }

    /// Whether the session gate is on (the adaptive rule can switch it).
    pub fn session_gate_on(&self) -> bool {
        self.policy.admission.session_gate && self.adaptive.as_ref().is_none_or(|a| a.gate)
    }

    /// Shed by slack (`admission.shed_by_slack`, the simulator's `holdReason` KV branch): hold a
    /// turn of an admitted session of `class`, up to its class's `shed_hold_s` after it was
    /// queued, while a higher class with a service level is not 5 points over its target in its
    /// trailing window and no worker has KV room for the turn. `None` lets it go.
    fn shed(&self, class: &str, now: Time, head: HeadFacts) -> Option<Decision> {
        if !self.policy.admission.shed_by_slack {
            return None;
        }
        use std::sync::atomic::Ordering::Relaxed;
        let n = self.shed_counts[0].fetch_add(1, Relaxed) + 1;
        match head.kv_room {
            Some(false) => {
                self.shed_counts[1].fetch_add(1, Relaxed);
            }
            None => {
                self.shed_counts[2].fetch_add(1, Relaxed);
            }
            Some(true) => {}
        }
        if n.is_power_of_two() || n.is_multiple_of(1_000_000) {
            let c: Vec<u64> = self.shed_counts.iter().map(|x| x.load(Relaxed)).collect();
            tracing::info!(
                checked = c[0],
                no_room = c[1],
                room_unknown = c[2],
                no_slack = c[3],
                holds = c[4],
                "shed_by_slack_stats"
            );
        }
        let own = self.policy.class(Some(class));
        let hold = own.admission.shed_hold_s;
        if hold <= 0.0 {
            return None;
        }
        let until = head
            .queued_at
            .unwrap_or(now)
            .after(Duration::from_secs_f64(hold));
        if now >= until {
            return None;
        }
        let no_slack = |c: &crate::config::ClassPolicy| {
            c.rank < own.rank
                && c.service_level.as_ref().is_some_and(|level| {
                    if self.policy.admission.shed_target.is_some() {
                        self.shed_switch_on(&c.name)
                    } else {
                        self.attainment(&c.name, now)
                            .is_none_or(|a| a < level.attainment + 0.05)
                    }
                })
        };
        let hold = |short: &str| Decision::Hold {
            reason: HoldReason::NoSlack {
                short: short.to_owned(),
            },
            until,
        };
        // KV first: no worker has room while a higher class has no slack.
        let short = self.policy.classes.iter().find(|c| no_slack(c));
        if short.is_some() {
            self.shed_counts[3].fetch_add(1, Relaxed);
        }
        if let Some(short) = short
            && head.kv_room == Some(false)
        {
            self.shed_counts[4].fetch_add(1, Relaxed);
            return Some(hold(&short.name));
        }
        // Forecast harm (the simulator's harmAbove): the best placement adds predicted misses to
        // a higher class that has no slack to spend.
        if self.policy.admission.shed_forecast_harm {
            for (class, misses) in &head.harm {
                if *misses <= 0.0 {
                    continue;
                }
                let c = self.policy.class(Some(class));
                if c.name == *class && no_slack(c) {
                    return Some(hold(class));
                }
            }
        }
        None
    }

    /// The multiple of `class`'s gate share pacing applies (1 without pacing).
    pub fn pace_scale(&self, class: &str) -> f64 {
        // f49n: the loop's gate scale (with `target_loop.rebalance`, a higher class's lift).
        self.release_scale(class) * self.loop_gate_scale(class) * self.pace_mult(class)
    }

    /// The pacing multiple of `class` alone (1 without `admission.pace`).
    fn pace_mult(&self, class: &str) -> f64 {
        if self.policy.admission.pace.is_none() {
            return 1.0;
        }
        match class {
            "standard" => self.pace.standard,
            "batch" => self.pace.batch,
            _ => 1.0,
        }
    }

    /// f49n: the target loop's multiple of `class`'s gate share: [`Self::loop_scale`], except that
    /// with `target_loop.rebalance` a class ranked above the loop's class is lifted so its hot cap
    /// is never under a lower class's ([`Self::lift_over_lower`], f49l3's rule). Its prefill budget
    /// share ([`Self::loop_scale`]) is not lifted.
    fn loop_gate_scale(&self, class: &str) -> f64 {
        let lifted = self.policy.admission.target_loop.as_ref().and_then(|cfg| {
            cfg.rebalance.as_ref()?;
            let at = self.policy.class(Some(&cfg.class)).rank;
            self.policy
                .classes
                .iter()
                .find(|c| c.name == class && c.rank < at)
        });
        let Some(me) = lifted else {
            return self.loop_scale(class);
        };
        // f49q4 `release_order: demand_first`: no lift (a lower class's hot cap may pass its).
        if self.loop_demand_first() {
            return 1.0;
        }
        // Its hot cap per unit of scale without the loop (its loop scale is 1).
        let own = me.admission.gate_share
            * self.release_scale(class)
            * self.pace_mult(class)
            * self.step_target_scale(class)
            * self.policy.class_overcommit(me);
        self.lift_over_lower(me, own, 1.0, &|_| 1.0)
    }

    /// f49l3: `c`'s hot cap at share `k`, in units of `hot_overcommit x` HBM: `gate_share x` pacing
    /// `x` step target `x class_overcommit x k`.
    fn cap_share(&self, c: &crate::config::ClassPolicy, k: f64) -> f64 {
        c.admission.gate_share
            * self.pace_scale(&c.name)
            * self.step_target_scale(&c.name)
            * self.policy.class_overcommit(c)
            * k
    }

    /// f49l3's no-inversion rule, shared with f49n: `scale` raised so the hot cap of `me` (`own`
    /// per unit of scale) is never under that of a class ranked below it at its share `k_of`
    /// ([`Self::cap_share`]), the hot-exempt classes aside.
    fn lift_over_lower(
        &self,
        me: &crate::config::ClassPolicy,
        own: f64,
        scale: f64,
        k_of: &dyn Fn(&str) -> f64,
    ) -> f64 {
        let mut scale = scale;
        if own > 0.0 {
            for c in &self.policy.classes {
                if c.rank <= me.rank || self.policy.admission.hot_exempt_classes.contains(&c.name) {
                    continue;
                }
                scale = scale.max(self.cap_share(c, k_of(&c.name)) / own);
            }
        }
        scale
    }

    /// f49q4: whether `target_loop.rebalance.release_order: demand_first` is set.
    fn loop_demand_first(&self) -> bool {
        self.policy
            .admission
            .target_loop
            .as_ref()
            .and_then(|l| l.rebalance.as_ref())
            .is_some_and(|rc| rc.demand_first())
    }

    /// f49q4 `release_order: demand_first`: `class`'s rebalance scale, at most that of every
    /// class ranked above it down to the loop's class.
    fn loop_rb_bounded(&self, cfg: &crate::config::TargetLoopConfig, class: &str) -> f64 {
        let at = self.policy.class(Some(&cfg.class)).rank;
        let me = self.policy.class(Some(class));
        self.policy
            .classes
            .iter()
            .filter(|c| c.rank >= at && (c.rank < me.rank || c.name == me.name))
            .map(|c| self.lp.rb.scale.get(&c.name).copied().unwrap_or(1.0))
            .fold(f64::INFINITY, f64::min)
    }

    /// f49j `admission.step_target`: the multiple of `hot_overcommit` for `class` in the gate's KV
    /// fit, which is fleet-aggregated: the HBM-weighted mean of the workers' multiples for a class
    /// ranked below the target class, else 1; 1 without the option.
    pub fn step_target_scale(&self, class: &str) -> f64 {
        self.step_target.as_ref().map_or(1.0, |st| st.scale(class))
    }

    /// f49k2: the share scale of class `c` in the gate's KV fit of a session of `class`: `own` (the
    /// scale the gate computed for `class`) for `class`, else `c`'s pacing, release, target loop
    /// and step target scales.
    fn class_scale(&self, c: &crate::config::ClassPolicy, class: &str, own: f64) -> f64 {
        if c.name == class {
            own
        } else {
            // f49l: with `admission.sla_budget`, its share too (1 without it).
            self.pace_scale(&c.name)
                * self.step_target_scale(&c.name)
                * self.sla_scale(&c.name)
                * self.yield_scale(&c.name)
        }
    }

    /// f49j: whether `admission.step_target` is on.
    pub fn step_target_on(&self) -> bool {
        self.step_target.is_some()
    }

    /// f49j: the step-target controller, when on.
    pub fn step_target(&self) -> Option<&crate::step_target::StepTarget> {
        self.step_target.as_ref()
    }

    /// f49j: the step-target controller, when on (for the host's feeds and tests).
    pub fn step_target_mut(&mut self) -> Option<&mut crate::step_target::StepTarget> {
        self.step_target.as_mut()
    }

    /// f49j: a lifecycle event of any request for the step target's measure; nothing when off.
    pub fn step_target_event(
        &mut self,
        event: &dynamo_kv_router::plugins::prelude::ClassifyEvent,
        now: Time,
    ) {
        use dynamo_kv_router::plugins::prelude::ClassifyEvent;
        let Some(st) = self.step_target.as_mut() else {
            return;
        };
        // f49j2: the request's class (its target and the decode cap) and session (its home worker).
        let id = match event {
            ClassifyEvent::Sent { request_id, .. }
            | ClassifyEvent::Responding { request_id, .. }
            | ClassifyEvent::Completed { request_id, .. }
            | ClassifyEvent::Aborted { request_id, .. } => Some(request_id.as_str()),
            _ => None,
        };
        let record = id.and_then(|id| self.requests.map.get(id));
        st.on_event_of(
            event,
            record.map(|r| r.class.as_str()),
            record.map(|r| r.session.as_str()),
            now,
        );
    }

    /// f49j: the workers the host lists now with their HBM tokens; nothing when off.
    pub fn step_target_workers(
        &mut self,
        listed: &[(dynamo_kv_router::plugins::prelude::WorkerWithDpRank, u64)],
        now: Time,
    ) {
        if let Some(st) = self.step_target.as_mut() {
            st.observe_workers(listed, now);
        }
    }

    /// f49j2: the host's context counter of classified request `request_id` (its decode progress
    /// for the step target's measure); nothing when off.
    pub fn step_target_progress(
        &mut self,
        request_id: &str,
        progress: &dynamo_kv_router::plugins::request_classifier::RequestProgress,
        now: Time,
    ) {
        if let Some(st) = self.step_target.as_mut() {
            st.note_progress(request_id, progress.clone(), now);
        }
    }

    /// f49j2 `step_target.decode_cap`: the gate's answer for turn `request_id` of `session`, which
    /// the gate lets go to `target`; [`crate::step_target::CapVerdict::Pass`] when off, outside
    /// enforce mode, or for a class not capped.
    pub fn step_target_cap(
        &mut self,
        request_id: &str,
        session: &str,
        target: Option<dynamo_kv_router::plugins::prelude::WorkerWithDpRank>,
        now: Time,
    ) -> crate::step_target::CapVerdict {
        if self.policy.mode != Mode::Enforce {
            return crate::step_target::CapVerdict::Pass;
        }
        let Some(st) = self.step_target.as_mut() else {
            return crate::step_target::CapVerdict::Pass;
        };
        let class = self
            .sessions
            .get(session)
            .map_or(self.policy.default_class.as_str(), |s| s.class.as_str());
        st.cap_verdict(request_id, class, Some(session), target, now)
    }

    /// f49j: one step of the step-target controller when its interval has passed.
    fn step_target_step(&mut self, now: Time) {
        if let Some(st) = self.step_target.as_mut() {
            st.tick(now);
        }
    }

    /// f49l `admission.sla_budget`: the multiple of `class`'s hot budget it sets; exactly 1
    /// without it. A class ranked below the top: its share K. A top-ranked class: the larger of
    /// its own share K_p (1 outside the fallback) and, for every lower class the hot budget gates,
    /// that class's cap over this one's (each `gate_share x` pacing `x` step target `x
    /// class_overcommit`, the lower one's `x K`), so its hot cap is never below a lower class's.
    pub fn sla_scale(&self, class: &str) -> f64 {
        let Some(sb) = self.sla_budget.as_ref() else {
            return 1.0;
        };
        // f49l3 `mode: rebalance`: every class has its own K, and no class's cap is under a lower
        // one's.
        if sb.rebalancing() {
            return self.sla_scale_rebalance(sb, class);
        }
        if !sb.is_top(class) {
            return sb.k(class);
        }
        // f49n: the shared hot cap.
        let cap = |c: &crate::config::ClassPolicy, k: f64| self.cap_share(c, k);
        let own = cap(self.policy.class(Some(class)), 1.0);
        let mut scale = sb.k_top();
        if own > 0.0 {
            for name in sb.lower() {
                if self.policy.admission.hot_exempt_classes.contains(name) {
                    continue;
                }
                scale = scale.max(cap(self.policy.class(Some(name.as_str())), sb.k(name)) / own);
            }
        }
        scale
    }

    /// f49l3 `sla_budget.mode: rebalance`: `class`'s own K, raised so its hot cap (`gate_share x`
    /// pacing `x` step target `x class_overcommit x K`) is never under that of a class ranked below
    /// it (the hot-exempt classes aside). An unknown name: its K (1).
    fn sla_scale_rebalance(&self, sb: &crate::sla_budget::SlaBudget, class: &str) -> f64 {
        let scale = sb.k(class);
        let Some(me) = self.policy.classes.iter().find(|c| c.name == class) else {
            return scale;
        };
        // f49n: the no-inversion rule shared with `target_loop.rebalance`.
        self.lift_over_lower(me, self.cap_share(me, 1.0), scale, &|c| sb.k(c))
    }

    /// f49o `admission.batch_yield`: the share of the yield class's KV higher classes do not count
    /// (the level), when the batch-yield gate is on and no static `yield_class` is set.
    fn batch_yield_discount(&self) -> Option<f64> {
        if self.policy.admission.yield_class.is_some() {
            return None;
        }
        let by = self.batch_yield.as_ref()?;
        by.config().admit.then(|| by.level())
    }

    /// f49o: the multiple of `class`'s hot budget batch yield sets: 1 - level for the yield class
    /// with the gate on; exactly 1 otherwise.
    pub fn yield_scale(&self, class: &str) -> f64 {
        match self.batch_yield.as_ref() {
            Some(by) if by.config().admit && by.config().class == class => 1.0 - by.level(),
            _ => 1.0,
        }
    }

    /// f49o: the batch-yield level (0 when off).
    pub fn batch_yield_level(&self) -> f64 {
        self.batch_yield.as_ref().map_or(0.0, |b| b.level())
    }

    /// f49o: one batch-yield step when due: the watched classes' trailing attainment and whether
    /// any of their sessions waits in the lobby; the engine signal when `engine` is on.
    fn batch_yield_step(&mut self, now: Time) {
        let Some(by) = self.batch_yield.as_ref() else {
            return;
        };
        if !by.due(now) {
            return;
        }
        let watch = by.config().watch.clone();
        let waiting = self
            .sessions
            .iter()
            .any(|s| s.state == SessionState::Lobby && watch.contains(&s.class));
        let inputs: Vec<crate::batch_yield::WatchInput> = watch
            .iter()
            .map(|c| crate::batch_yield::WatchInput {
                class: c.clone(),
                attainment: self.attainment(c, now),
                target: self
                    .policy
                    .classes
                    .iter()
                    .find(|p| &p.name == c)
                    .and_then(|p| p.service_level.as_ref())
                    .map(|l| l.attainment),
            })
            .collect();
        let by = self.batch_yield.as_mut().expect("checked above");
        by.step(now, &inputs, waiting);
        if by.config().engine {
            let (level, rank, finish, aside) = by.engine_signal();
            dynamo_kv_router::plugins::adaptive_switches::set_batch_yield(
                level, rank, finish, aside,
            );
        }
    }

    /// f49l: whether `admission.sla_budget` is on.
    pub fn sla_budget_on(&self) -> bool {
        self.sla_budget.is_some()
    }

    /// f49l: the SLA loop, when on.
    pub fn sla_budget(&self) -> Option<&crate::sla_budget::SlaBudget> {
        self.sla_budget.as_ref()
    }

    /// f49l: the SLA loop, when on (tests).
    #[cfg(test)]
    pub(crate) fn sla_budget_mut(&mut self) -> Option<&mut crate::sla_budget::SlaBudget> {
        self.sla_budget.as_mut()
    }

    /// f49l: one step of the SLA loop when its interval has passed, on each class's window
    /// counts, lobby waiters and refusals so far (no forecast: the plugin's forecast predicts no
    /// class attainment).
    fn sla_budget_step(&mut self, now: Time) {
        if !self.sla_budget.as_ref().is_some_and(|sb| sb.due(now)) {
            return;
        }
        let mut waiting = std::collections::HashSet::new();
        for s in self.sessions.iter() {
            if s.state == SessionState::Lobby {
                waiting.insert(s.class.as_str());
            }
        }
        let inputs: HashMap<String, crate::sla_budget::ClassInput> = self
            .policy
            .classes
            .iter()
            .map(|c| {
                let input = crate::sla_budget::ClassInput {
                    counts: self.metrics.window_counts(&c.name, now),
                    waiting: waiting.contains(c.name.as_str()),
                    lost: self.lost.get(&c.name).copied().unwrap_or(0),
                    forecast: None,
                };
                (c.name.clone(), input)
            })
            .collect();
        self.sla_budget_apply(now, &inputs);
    }

    /// f49l: one step of the SLA loop on `inputs`, and its setpoint handed to the step target.
    pub(crate) fn sla_budget_apply(
        &mut self,
        now: Time,
        inputs: &HashMap<String, crate::sla_budget::ClassInput>,
    ) {
        let Some(sb) = self.sla_budget.as_mut() else {
            return;
        };
        sb.step(&self.policy, now, inputs);
        if let (Some(sp), Some(st)) = (sb.setpoint(), self.step_target.as_mut()) {
            st.set_target_frac(sp);
        }
        // f49l3 `mode: rebalance`: each class's own setpoint.
        if let (Some(fracs), Some(st)) = (sb.class_setpoints(), self.step_target.as_mut()) {
            st.set_class_fracs(fracs);
        }
    }

    /// The KV room a new session of `class` has now ([`crate::admit::room`]), with f49j's
    /// multiple on its hot budget when `admission.step_target` is on.
    fn step_room(&self, class: &crate::config::ClassPolicy, kv: &KvBudget) -> u64 {
        if self.step_target.is_none() && self.sla_budget.is_none() {
            return crate::admit::room(&self.policy, class, kv);
        }
        // f49k2: every class at its step target scale.
        // f49l: with `admission.sla_budget`, its share too.
        crate::admit::room_classes(
            &self.policy,
            class,
            kv,
            &|c: &crate::config::ClassPolicy| {
                self.step_target_scale(&c.name) * self.sla_scale(&c.name)
            },
        )
    }

    /// The release multiple of `class`'s gate share (1 without `admission.release`).
    pub fn release_scale(&self, class: &str) -> f64 {
        self.targets.release.get(class).copied().unwrap_or(1.0)
    }

    /// Whether target-aware shedding counts `class` as having no slack now (on until its first
    /// reading turns it off).
    pub fn shed_switch_on(&self, class: &str) -> bool {
        self.targets.shed.get(class).is_none_or(|(on, _)| *on)
    }

    /// (shed switch changes, smoothed attainment per class) for reporting.
    pub fn target_stats(&self) -> (u64, HashMap<String, f64>) {
        (self.targets.flips, self.targets.att.clone())
    }

    /// One step of target-aware shedding and surplus release when their `step_s` has passed.
    fn targets_step(&mut self, now: Time) {
        let shed_cfg = self.policy.admission.shed_target.clone();
        let rel_cfg = self.policy.admission.release.clone();
        if shed_cfg.is_none() && rel_cfg.is_none() {
            return;
        }
        let step_s = shed_cfg
            .as_ref()
            .map(|c| c.step_s)
            .into_iter()
            .chain(rel_cfg.as_ref().map(|c| c.step_s))
            .fold(f64::INFINITY, f64::min);
        let smooth_s = shed_cfg
            .as_ref()
            .map(|c| c.smooth_s)
            .or(rel_cfg.as_ref().map(|c| c.smooth_s))
            .unwrap_or(300.0);
        let Some(last) = self.targets.last else {
            self.targets.last = Some(now);
            return;
        };
        let dt = now.since(last).as_secs_f64();
        if dt < step_s {
            return;
        }
        self.targets.last = Some(now);
        let alpha = 1.0 - (-dt / smooth_s.max(1e-9)).exp();
        let mut classes: Vec<(String, u32, Option<f64>)> = self
            .policy
            .classes
            .iter()
            .map(|c| {
                (
                    c.name.clone(),
                    c.rank,
                    c.service_level.as_ref().map(|l| l.attainment),
                )
            })
            .collect();
        classes.sort_by_key(|c| c.1);
        for (name, _, target) in &classes {
            if target.is_none() {
                continue;
            }
            if let Some(a) = self.attainment(name, now) {
                let e = self.targets.att.entry(name.clone()).or_insert(a);
                *e += alpha * (a - *e);
            }
        }
        if let Some(cfg) = &shed_cfg {
            for (name, _, target) in &classes {
                let (Some(t), Some(a)) = (target, self.targets.att.get(name).copied()) else {
                    continue;
                };
                let (on, since) = self.targets.shed.get(name).copied().unwrap_or((true, None));
                let held = since.is_none_or(|at| now.since(at).as_secs_f64() >= cfg.min_hold_s);
                let next = if on {
                    !(held && a > t + cfg.off_margin)
                } else {
                    held && a < t + cfg.on_margin
                };
                if next != on {
                    self.targets.flips += 1;
                    self.targets.shed.insert(name.clone(), (next, Some(now)));
                } else if !self.targets.shed.contains_key(name) {
                    self.targets.shed.insert(name.clone(), (on, since));
                }
            }
        }
        if let Some(cfg) = &rel_cfg {
            // Demand per class this step: a session waiting, or one turned away since last step.
            let demand: HashMap<String, bool> = classes
                .iter()
                .map(|(name, _, _)| {
                    let lost = self.lost.get(name).copied().unwrap_or(0);
                    let seen = self
                        .targets
                        .lost_seen
                        .insert(name.clone(), lost)
                        .unwrap_or(0);
                    (
                        name.clone(),
                        self.sessions.lobby().has_class(name) || lost != seen,
                    )
                })
                .collect();
            for (i, (name, _, _)) in classes.iter().enumerate() {
                if i == 0 || (!cfg.classes.is_empty() && !cfg.classes.contains(name)) {
                    continue;
                }
                let mut headroom = f64::INFINITY;
                let mut known = true;
                for (above, _, target) in &classes[..i] {
                    let Some(t) = target else { continue };
                    let Some(a) = self.targets.att.get(above).copied() else {
                        known = false;
                        break;
                    };
                    let m = cfg
                        .margins
                        .get(above)
                        .copied()
                        .unwrap_or(cfg.default_margin);
                    headroom = headroom.min(a - (t + m));
                }
                if !known || !headroom.is_finite() {
                    continue;
                }
                let own_demand = demand.get(name).copied().unwrap_or(false);
                if cfg.yield_to_demand
                    && classes[..i].iter().any(|(above, _, target)| {
                        demand.get(above).copied().unwrap_or(false)
                            && match (cfg.yield_margin, target) {
                                (Some(m), Some(t)) => {
                                    self.targets.att.get(above).is_none_or(|a| *a < t + m)
                                }
                                _ => true,
                            }
                    })
                {
                    // A class above has users waiting: never grow, and give back.
                    headroom = headroom.min(-1e-9);
                }
                if !crate::coordinator::release_enabled() {
                    // The coordinator holds release off (a surge): give the growth back.
                    headroom = headroom.min(-1e-9);
                }
                let rate = cfg.grow_per_min / 60.0 * dt;
                let scale = self.targets.release.entry(name.clone()).or_insert(1.0);
                if headroom > 0.0 && own_demand {
                    let k = (headroom / 0.05).min(2.0);
                    *scale = (*scale * (rate * k).exp()).min(cfg.max_scale);
                } else if headroom < 0.0 {
                    *scale = (*scale * (-rate * cfg.shrink_ratio).exp()).max(1.0);
                }
            }
        }
    }

    /// One pacing step when `step_s` has passed (the simulator's `paceGates`, mode `standard`):
    /// batch's share gives way first while priority is short of its margin, then standard's
    /// falls below the set share; standard's grows on demand (a standard session waiting or
    /// turned away) while priority and admitted standard both have their margins.
    fn pace_step(&mut self, now: Time) {
        let Some(cfg) = self.policy.admission.pace.clone() else {
            return;
        };
        let Some(last) = self.pace.last else {
            self.pace.last = Some(now);
            return;
        };
        let dt = now.since(last).as_secs_f64();
        if dt < cfg.step_s {
            return;
        }
        self.pace.last = Some(now);
        let alpha = 1.0 - (-dt / cfg.smooth_s.max(1e-9)).exp();
        let smooth = |e: Option<f64>, a: Option<f64>| match (e, a) {
            (Some(e), Some(a)) => Some(e + alpha * (a - e)),
            (None, a) => a,
            (e, None) => e,
        };
        self.pace.priority_att = smooth(self.pace.priority_att, self.attainment("priority", now));
        self.pace.standard_att = smooth(self.pace.standard_att, self.attainment("standard", now));
        let target = |name: &str| {
            self.policy
                .classes
                .iter()
                .find(|c| c.name == name)
                .and_then(|c| c.service_level.as_ref())
                .map(|l| l.attainment)
        };
        let (pt, st) = (target("priority"), target("standard"));
        let (pa, sa) = (self.pace.priority_att, self.pace.standard_att);
        let p_ok = match (pt, pa) {
            (Some(t), Some(a)) => a - t > cfg.head_priority,
            _ => true,
        };
        let p_low = matches!((pt, pa), (Some(t), Some(a)) if a - t < cfg.head_priority - 0.02);
        let s_ok = matches!((st, sa), (Some(t), Some(a)) if a - t > cfg.head_standard);
        let s_low = matches!((st, sa), (Some(t), Some(a)) if a - t < cfg.head_standard - 0.03);
        let lost = self.lost.get("standard").copied().unwrap_or(0);
        let demand =
            self.sessions.lobby().has_class("standard") || lost != self.pace.lost_standard_seen;
        self.pace.lost_standard_seen = lost;
        let shrink = (-0.05 / 60.0 * dt).exp();
        let grow = (0.01 / 60.0 * dt).exp();
        let k = &mut self.pace;
        if p_low && k.batch > cfg.batch_low {
            k.batch = (k.batch * shrink).max(cfg.batch_low);
        } else if !p_low && k.batch < 1.0 && p_ok {
            k.batch = (k.batch * grow).min(1.0);
        }
        if p_low && k.batch <= cfg.batch_low {
            k.standard = (k.standard * shrink).max(cfg.standard_low);
        } else if p_low {
            // Batch is still giving way.
        } else if s_low {
            k.standard = (k.standard * shrink).max(k.standard.min(1.0));
        } else if p_ok && s_ok && demand {
            k.standard = (k.standard * grow).min(cfg.standard_high);
        }
    }

    /// Rotation candidates (`admission.rotate`, the simulator's `rotateSessions`; F32): for each
    /// class other than the top-ranked whose oldest waiting new session has waited the trigger
    /// (from its first attempt: [`Session::lobby_wait_s`]), the admitted top-level sessions of that
    /// class or a lower one, between turns with no subagent in flight, admitted at least the least
    /// run ago ([`RotateConfig::min_run_for`]): the lowest class first, then the longest since its
    /// latest admission. The caller pauses the first one whose KV the host tier covers
    /// ([`Self::pause`]). O(sessions).
    pub fn rotation_candidates(&self, now: Time) -> Vec<RotationCandidates> {
        let Some(rotate) = &self.policy.admission.rotate else {
            return Vec::new();
        };
        let top = self
            .policy
            .classes
            .iter()
            .map(|c| c.rank)
            .min()
            .unwrap_or(0);
        let mut oldest: HashMap<String, f64> = HashMap::new();
        for s in self.sessions.iter() {
            if (s.state == SessionState::Lobby || s.away()) && !s.resumed && s.parent.is_none() {
                let e = oldest.entry(s.class.clone()).or_insert(0.0);
                *e = e.max(s.lobby_wait_s(now));
            }
        }
        let busy_parents: std::collections::HashSet<&str> = self
            .sessions
            .iter()
            .filter(|s| s.in_flight > 0 || s.state == SessionState::Active)
            .filter_map(|s| s.parent.as_deref())
            .collect();
        let mut out = Vec::new();
        let mut classes: Vec<_> = oldest.into_iter().collect();
        // Higher classes' waiters first; ties by name so a run repeats.
        classes.sort_by(|a, b| {
            let rank = |c: &str| self.policy.class(Some(c)).rank;
            rank(&a.0).cmp(&rank(&b.0)).then_with(|| a.0.cmp(&b.0))
        });
        for (class, waited) in classes {
            let policy = self.policy.class(Some(&class));
            let rank = policy.rank;
            let after = rotate.after_for(policy.admission.max_lobby_wait_s);
            if rank == top || waited < after {
                continue;
            }
            let min_run = rotate.min_run_for(after, waited);
            let mut ran: Vec<(u32, f64, String, String)> = self
                .sessions
                .iter()
                .filter(|s| {
                    s.parent.is_none()
                        && s.state == SessionState::Idle
                        && s.in_flight == 0
                        && !busy_parents.contains(s.id.as_str())
                })
                .filter_map(|s| {
                    let r = self.policy.class(Some(&s.class)).rank;
                    if r < rank {
                        return None;
                    }
                    let run = now.since(s.last_admitted.or(s.admitted)?).as_secs_f64();
                    (run >= min_run).then(|| (r, run, s.id.clone(), s.class.clone()))
                })
                .collect();
            ran.sort_by(|a, b| {
                b.0.cmp(&a.0)
                    .then_with(|| b.1.total_cmp(&a.1))
                    .then_with(|| a.2.cmp(&b.2))
            });
            if !ran.is_empty() {
                out.push(RotationCandidates {
                    for_class: class,
                    waited_s: waited,
                    sessions: ran
                        .into_iter()
                        .map(|(_, run, id, class)| (id, class, run))
                        .collect(),
                });
            }
        }
        out
    }

    /// Pause `session` (rotate it out at a stop point). Returns whether it was paused.
    pub fn pause(&mut self, session: &str, now: Time) -> bool {
        let paused = self
            .sessions
            .update(session, |s| {
                if s.state != SessionState::Idle || s.in_flight > 0 {
                    return false;
                }
                s.state = SessionState::Paused;
                s.paused_at = Some(now);
                true
            })
            .unwrap_or(false);
        if paused {
            self.rotations += 1;
        }
        paused
    }

    /// Sessions rotated out so far.
    pub fn rotations(&self) -> u64 {
        self.rotations
    }

    /// The adaptive rule's state, when configured.
    pub fn adaptive(&self) -> Option<&crate::adaptive::AdaptiveState> {
        self.adaptive.as_ref()
    }

    /// Feed the adaptive rule: a request carried a client signal.
    pub fn adaptive_hint_seen(&mut self) {
        if let Some(a) = &mut self.adaptive {
            a.hints_seen = true;
        }
    }

    /// One adaptive-rule sample when due: `load` is the mean of active tokens over KV capacity
    /// across workers; KV demand is every live session's context over `hbm + dram` tokens.
    /// Returns the place group's state (`None` without the rule).
    pub fn adaptive_step(&mut self, now: Time, load: f64, hbm: u64, dram: u64) -> Option<bool> {
        let total = self.sessions.kv_load().0;
        // The top class with a service level, for the preempt group.
        let top = self
            .policy
            .classes
            .iter()
            .filter(|c| c.service_level.is_some())
            .min_by_key(|c| c.rank)
            .map(|c| {
                (
                    c.name.clone(),
                    c.service_level.as_ref().map(|l| l.attainment),
                )
            });
        let top_attainment = top
            .as_ref()
            .and_then(|(name, _)| self.attainment(name, now));
        let top_target = top.and_then(|(_, target)| target);
        let a = self.adaptive.as_mut()?;
        if a.due(now) {
            let demand = total as f64 / (hbm + dram).max(1) as f64;
            a.step_with(now, load, demand, top_attainment, top_target);
        }
        Some(a.place)
    }

    /// The class's attainment over the trailing window (the simulator's `windowAtt`).
    /// With `runs.headroom` other than `request`, the class's target plus its run headroom (or
    /// the tighter of the two): what the soft reserve, release and the coordinator read.
    pub fn window_share(&self, class: &str, now: Time) -> Option<f64> {
        self.attainment(class, now)
    }

    /// Count a new session of `class` turned away (expired in the lobby or throttled).
    pub fn note_lost(&mut self, class: &str) {
        *self.lost.entry(class.to_owned()).or_default() += 1;
    }

    /// New sessions turned away so far, by class.
    pub fn lost(&self) -> &HashMap<String, u64> {
        &self.lost
    }

    /// The pacing multiples of standard's and batch's gate shares.
    pub fn pace_multiples(&self) -> (f64, f64) {
        (self.pace.standard, self.pace.batch)
    }

    /// The session request `request_id` belongs to, while the scheduler tracks it.
    pub fn session_of(&self, request_id: &str) -> Option<&str> {
        self.requests
            .map
            .get(request_id)
            .map(|r| r.session.as_str())
    }

    /// The host took the head of `session` (request `request_id`) out of the queue to admit it.
    pub fn on_dispatched(&mut self, request_id: &str, session: &str, now: Time) {
        let fresh = self.requests.update(request_id, |record| {
            if record.dispatched {
                return false;
            }
            record.dispatched = true;
            record.dispatched_at = Some(now);
            true
        });
        // Unknown or already counted: nothing is in flight that a completion will settle.
        if fresh != Some(true) {
            return;
        }
        // f49k `admission.hot_budget: measured`: the turn's input counts as hot until measured.
        if let Some(m) = self.hot_measure.as_mut()
            && let Some(isl) = self.requests.map.get(request_id).map(|r| r.isl)
        {
            m.admitted(request_id, isl);
        }
        let admitted = self
            .sessions
            .update(session, |s| {
                let was_lobby = s.state == SessionState::Lobby;
                if was_lobby {
                    if let Some(q) = s.queued_since.take() {
                        s.waited_s += now.since(q).as_secs_f64();
                    }
                    s.last_admitted = Some(now);
                }
                if was_lobby && s.resumed {
                    // A rotated session resumes: its time out counts as paused.
                    if let Some(at) = s.paused_at.take() {
                        s.paused_s += now.since(at).as_secs_f64();
                    }
                    s.resumed = false;
                } else if was_lobby {
                    s.admitted = Some(now);
                }
                s.state = SessionState::Active;
                s.in_flight += 1;
                was_lobby
            })
            .unwrap_or(false);
        if admitted {
            self.log.forget(session);
            // f49c: admission uses up the session's ticket (f49e `per_root`: its root's, once no
            // other session waits on it).
            if self.honest.is_some() {
                self.honest_admitted(session, now);
            } else {
                self.ticket_admitted(session, now);
            }
            if self.policy.admission.onboard_pace.is_some()
                && let Some((class, isl)) = self
                    .sessions
                    .get(session)
                    .map(|s| (s.class.clone(), s.pending_isl))
            {
                self.onboard_charge(&class, isl, now);
            }
            if let Some(class) = self.sessions.get(session).map(|s| s.class.clone()) {
                let window = self.estimate_window_s(&class);
                let q = self.admits.entry(class).or_default();
                q.push_back(now);
                while q
                    .front()
                    .is_some_and(|&t| now.since(t).as_secs_f64() > window)
                {
                    q.pop_front();
                }
            }
        }
    }

    /// The window the admission rate is measured over for `class`: `estimate_window_waits`
    /// (default 10) times the class's wait limit.
    fn estimate_window_s(&self, class: &str) -> f64 {
        let waits = self
            .policy
            .admission
            .retry_after_estimate
            .as_ref()
            .map_or(10.0, |e| e.estimate_window_waits);
        waits * self.policy.class(Some(class)).admission.max_lobby_wait_s
    }

    /// Admissions out of `class`'s lobby a second over its estimate window at `now`; `None`
    /// before a whole window has passed since the first admission.
    pub fn admit_rate(&self, class: &str, now: Time) -> Option<f64> {
        Self::rate_in(&self.admits, class, self.estimate_window_s(class), now)
    }

    /// Ends of `class`'s admitted sessions a second over its estimate window at `now` (f40);
    /// `None` before a whole window has passed or before the first end.
    pub fn end_rate(&self, class: &str, now: Time) -> Option<f64> {
        Self::rate_in(&self.ends, class, self.estimate_window_s(class), now)
    }

    fn rate_in(
        times: &HashMap<String, VecDeque<Time>>,
        class: &str,
        window: f64,
        now: Time,
    ) -> Option<f64> {
        let q = times.get(class)?;
        let n = q
            .iter()
            .filter(|&&t| now.since(t).as_secs_f64() <= window)
            .count();
        (window > 0.0 && now.as_secs_f64() >= window).then(|| n as f64 / window)
    }

    /// Retry-After for waiting session `session` (`admission.retry_after_estimate`): its place in
    /// its class's lobby over the class's admission rate, less the class's wait limit (the time
    /// it will already have waited when the limit expires it). `None` when the estimate is off,
    /// the session is not waiting, the rate is unknown or zero, or the estimate is not positive.
    pub fn retry_after_estimate(&self, session: &str, now: Time) -> Option<f64> {
        self.policy.admission.retry_after_estimate.as_ref()?;
        let s = self.sessions.get(session)?;
        let key = self.sessions.lobby_key(s)?;
        let place = self.sessions.lobby().place_in_class(&s.class, &key) + 1;
        let estimate = self.policy.admission.retry_after_estimate.as_ref()?;
        let admits = self.admit_rate(&s.class, now);
        let rate = if estimate.count_ends {
            match (admits, self.end_rate(&s.class, now)) {
                (Some(a), Some(e)) => Some(a.max(e)),
                (a, e) => a.or(e),
            }
        } else {
            admits
        };
        let Some(rate) = rate.filter(|r| *r > 0.0) else {
            return estimate.fallback_s;
        };
        let limit = self.policy.class(Some(&s.class)).admission.max_lobby_wait_s;
        let mut after = place as f64 / rate - limit;
        if estimate.calibrate {
            let f = self.ra_factor.get(&s.class).copied().unwrap_or(1.0);
            after = after.max(estimate.min_s.unwrap_or(0.0)) * f;
        }
        if let Some(max) = estimate.max_s {
            after = after.min(max);
        }
        if after > 0.0 {
            Some(after.max(estimate.min_s.unwrap_or(0.0)))
        } else {
            estimate.min_s
        }
    }

    /// Record the Retry-After `session`'s latest attempt was given (`None`: none).
    pub fn note_retry_after(&mut self, session: &str, seconds: Option<f64>) {
        self.sessions.update(session, |s| s.retry_after_s = seconds);
        if self.policy.admission.returnee_guarantee.is_some() {
            *self.refusals.entry(session.to_owned()).or_insert(0) += 1;
        }
        if seconds.is_some()
            && self
                .policy
                .admission
                .retry_after_estimate
                .as_ref()
                .is_some_and(|e| e.calibrate)
        {
            self.ra_waiting
                .entry(session.to_owned())
                .or_insert((0, false))
                .1 = true;
        }
    }

    /// The admission tickets (`admission.tickets`, f49c).
    pub fn tickets(&self) -> &crate::tickets::Tickets {
        &self.tickets
    }

    /// `admission.tickets` when `class` is one of its classes.
    fn ticket_class(&self, class: &str) -> Option<&crate::config::TicketsConfig> {
        self.policy
            .admission
            .tickets
            .as_ref()
            .filter(|c| c.classes.iter().any(|x| x == class))
    }

    /// f49e `admission.tickets.per_root`.
    fn tickets_per_root(&self) -> bool {
        self.policy
            .admission
            .tickets
            .as_ref()
            .is_some_and(|c| c.per_root)
    }

    /// f49e: the key of the ticket `session` holds or would hold: its root with `per_root`, else
    /// itself.
    fn ticket_key<'a>(&self, session: &'a str) -> std::borrow::Cow<'a, str> {
        if self.tickets_per_root() {
            std::borrow::Cow::Owned(self.root_of(session))
        } else {
            std::borrow::Cow::Borrowed(session)
        }
    }

    /// f49e: [`Self::ticket_key`] for a request of `session` with its `parent` (a session the table
    /// may not hold yet): with `per_root` the root of its parent chain, else `session`.
    pub fn ticket_root(&self, session: &str, parent: Option<&str>) -> String {
        if self.tickets_per_root() {
            self.root_of(parent.unwrap_or(session))
        } else {
            session.to_owned()
        }
    }

    /// Drop the expired tickets; with `hold_while_waiting` (f49e) keep a ticket past its expiry
    /// while a session that came back on it waits in the lobby.
    fn tickets_expire(&mut self, now: Time) {
        // f49m: a reservation also stays while its holder waits, and its lapse is logged.
        if self.honest.is_some() {
            self.honest_expire(now);
            return;
        }
        if !self
            .policy
            .admission
            .tickets
            .as_ref()
            .is_some_and(|c| c.hold_while_waiting)
        {
            self.tickets.expire(now);
            return;
        }
        let sessions = &self.sessions;
        self.tickets.expire_keeping(now, |t| {
            t.waiting.iter().any(|s| {
                sessions
                    .get(s)
                    .is_some_and(|x| x.state == SessionState::Lobby)
            })
        });
    }

    /// f49e `standard_cap_when_priority_below`: while the top-ranked class with a service level is
    /// under its attainment target, cap the holds of the tickets ranked below it at that share of
    /// the HBM budget. The attainment is the target loop's smoothed one (the f42 log's `att`), else
    /// the attainment the loop smooths ([`Self::attainment`]); unknown, nothing is capped.
    fn ticket_lower_cap_step(&mut self, now: Time) {
        let Some(frac) = self
            .policy
            .admission
            .tickets
            .as_ref()
            .and_then(|c| c.standard_cap_when_priority_below)
        else {
            return;
        };
        let below = self
            .policy
            .classes
            .iter()
            .filter_map(|c| Some((c, c.service_level.as_ref()?.attainment)))
            .min_by_key(|(c, _)| c.rank)
            .and_then(|(c, target)| {
                let att = self
                    .lp
                    .att
                    .get(&c.name)
                    .copied()
                    .or_else(|| self.attainment(&c.name, now))?;
                (att < target).then_some(c.rank)
            });
        let hbm = self.tickets.last_kv.map_or(0, |k| k.hbm_tokens);
        self.tickets.lower_cap = below.map(|rank| (rank, crate::tickets::Tickets::cap(frac, hbm)));
    }

    /// `session` was admitted: its ticket is used up. With `per_root` (f49e) the root's ticket,
    /// once no other session waiting on it is still in the lobby.
    fn ticket_admitted(&mut self, session: &str, now: Time) {
        if !self.tickets_per_root() {
            self.tickets.admitted(session, now);
            return;
        }
        let key = self.root_of(session);
        let others = self.tickets.get(&key).is_some_and(|t| {
            t.waiting.iter().any(|s| {
                s != session
                    && self
                        .sessions
                        .get(s)
                        .is_some_and(|x| x.state == SessionState::Lobby)
            })
        });
        if others {
            self.tickets.left_wait(&key, session);
        } else {
            self.tickets.admitted_by(&key, session, now);
        }
    }

    /// f49e `DYNAMO_TICKETS_LOG`: a gate decision other than an admission on `session` while its
    /// ticket (with `per_root` its root's) is live: `ticket_hold`, once per reason per wait.
    /// `step` is the step of the chain that held it, `room` the KV fit's room when the fit held it,
    /// `held_ahead` the holds it fitted around. Log only.
    #[allow(clippy::too_many_arguments)]
    fn ticket_hold_diag(
        &mut self,
        session: &str,
        class: &str,
        decision: &Decision,
        step: Option<&'static str>,
        room: Option<u64>,
        held_ahead: u64,
        now: Time,
    ) {
        if self.policy.admission.tickets.is_none() || matches!(decision, Decision::Admit(_)) {
            return;
        }
        let key = self.ticket_key(session).into_owned();
        if !self.tickets.is_live(&key, now) {
            return;
        }
        let reason = step.unwrap_or(match decision {
            Decision::Expire { .. } => "expire",
            _ => "kv_fit",
        });
        if !self.tickets.hold_seen(session, reason) {
            return;
        }
        let cap = self.ticket_cap();
        let rank = self.policy.class(Some(class)).rank;
        let (held_kv, kv_need) = self
            .tickets
            .holds(cap)
            .into_iter()
            .find(|(t, _)| t.session == key)
            .map_or((0, 0), |(t, h)| (h, t.kv_need));
        let fit_need = self
            .sessions
            .get(session)
            .map_or(0, |s| self.sessions.total_need(s));
        let held_others = self.tickets.held_by_others(&key, rank, cap);
        let room = room.map_or_else(|| "null".to_owned(), |r| r.to_string());
        let line = format!(
            "{{\"event\":\"ticket_hold\",\"t\":{:.3},\"session\":{:?},\"class\":{:?},\"reason\":{:?},\"held_kv\":{},\"kv_need\":{},\"fit_need\":{},\"room\":{},\"held_others\":{},\"held_ahead\":{}}}",
            now.as_secs_f64(),
            session,
            class,
            reason,
            held_kv,
            kv_need,
            fit_need,
            room,
            held_others,
            held_ahead
        );
        self.tickets.log_line(&line);
    }

    /// f49e `DYNAMO_TICKETS_LOG`: the lobby deadline refused `session`: `ticket_rerefused` when its
    /// ticket (with `per_root` its root's) is live, with the first hold reason of the wait. Log
    /// only.
    fn ticket_rerefused_diag(&mut self, session: &str, now: Time) {
        if !self.tickets.logging() {
            return;
        }
        let first = self.tickets.wait_over(session);
        let key = self.ticket_key(session).into_owned();
        if !self.tickets.is_live(&key, now) {
            return;
        }
        let Some((class, returns)) = self.tickets.get(&key).map(|t| (t.class.clone(), t.returns))
        else {
            return;
        };
        let first = first.map_or_else(|| "null".to_owned(), |r| format!("{r:?}"));
        let line = format!(
            "{{\"event\":\"ticket_rerefused\",\"t\":{:.3},\"session\":{:?},\"ticket\":{:?},\"class\":{:?},\"first_reason\":{},\"returns\":{}}}",
            now.as_secs_f64(),
            session,
            key,
            class,
            first,
            returns
        );
        self.tickets.log_line(&line);
    }

    /// f49f `admission.returnee_first` when it lists `class`.
    fn returnee_first(&self, class: &str) -> Option<&crate::config::ReturneeFirstConfig> {
        self.policy
            .admission
            .returnee_first
            .as_ref()
            .filter(|c| c.classes.iter().any(|x| x == class))
    }

    /// f49f: the live tickets of `class` (listed in `returnee_first`) that queue at `now`, oldest
    /// first (issue time, then issue order); with `max_block_s`, those issued at most that long
    /// ago. Empty for a class not listed.
    fn returnee_queue(&self, class: &str, now: Time) -> Vec<&crate::tickets::Ticket> {
        let Some(cfg) = self.returnee_first(class) else {
            return Vec::new();
        };
        let mut queue: Vec<&crate::tickets::Ticket> = self
            .tickets
            .iter()
            .filter(|t| t.class == class && self.tickets.is_live(&t.session, now))
            .filter(|t| {
                cfg.max_block_s
                    .is_none_or(|m| now.since(t.issued).as_secs_f64() <= m)
            })
            .collect();
        queue.sort_by(|a, b| a.issued.cmp(&b.issued).then_with(|| a.seq.cmp(&b.seq)));
        queue
    }

    /// f49f: a session (other than `except`) waiting in the lobby on ticket `t`: its holder, or
    /// with `per_root` a sub-session of its root that came back on it.
    fn returnee_waiter(&self, t: &crate::tickets::Ticket, except: Option<&str>) -> Option<String> {
        std::iter::once(&t.session)
            .chain(t.waiting.iter())
            .find(|s| {
                except != Some(s.as_str())
                    && self
                        .sessions
                        .get(s)
                        .is_some_and(|x| x.state == SessionState::Lobby)
            })
            .cloned()
    }

    /// f49f `admission.returnee_first`: a request of `session` (with `parent`, of `class`) that
    /// would open a new top-level session of a listed class (not in the table, or ended) without
    /// a live ticket of its own, while its class has a live ticket holder not yet admitted: a
    /// `Throttle` (`waiting` the class's live tickets, `limit` 0), recorded in the decision log,
    /// and `returnee_block` in the tickets log. The hooks refuse it like any throttle: it gets its
    /// own ticket and that ticket's Retry-After. Sub-sessions are never blocked.
    fn returnee_block(
        &mut self,
        session: &str,
        parent: Option<&str>,
        class: Option<&str>,
        need: Option<u64>,
        now: Time,
    ) -> Option<Decision> {
        if parent.is_some()
            || self.policy.mode == Mode::Off
            || self.policy.admission.returnee_first.is_none()
            || self.policy.admission.tickets.is_none()
            || self.tickets.is_empty()
        {
            return None;
        }
        let class = self.policy.class(class).name.clone();
        if self
            .sessions
            .get(session)
            .is_some_and(|s| s.state != SessionState::Ended)
            || self.tickets.is_live(session, now)
        {
            return None;
        }
        let (only, due_soon_s) = self
            .policy
            .admission
            .returnee_first
            .as_ref()
            .map_or((false, 0.0), |c| (c.only_when_waiting, c.due_soon_s));
        // The class's queue: its live tickets, those with a holder waiting in the lobby, and
        // (f49f2) those away and due back within `due_soon_s`.
        let (live, waiting, due_soon) = {
            let queue = self.returnee_queue(&class, now);
            let soon = now.after(Duration::from_secs_f64(due_soon_s.max(0.0)));
            let (mut waiting, mut due_soon) = (0, 0);
            for t in &queue {
                if self.returnee_waiter(t, None).is_some() {
                    waiting += 1;
                } else if t.due <= soon {
                    due_soon += 1;
                }
            }
            (queue.len(), waiting, due_soon)
        };
        if live == 0 {
            return None;
        }
        // f49f2 `only_when_waiting`: nobody waiting and nobody due soon: let it through.
        if only && waiting + due_soon == 0 {
            if self.tickets.logging()
                && let Some(count) = self.tickets.returnee_pass(&class, now)
            {
                let line = format!(
                    r#"{{"event":"returnee_pass","t":{:.3},"session":{:?},"class":{:?},"queue_live":{},"count":{},"room":{}"#,
                    now.as_secs_f64(),
                    session,
                    class,
                    live,
                    count,
                    self.returnee_room(&class)
                );
                self.tickets.returnee_line(session, line);
            }
            return None;
        }
        // f49m `admission.honest_retry`: refused only when the gate would not admit it now.
        if let Some(need) = need
            && self.honest.is_some()
            && self.honest_room_fits(&class, need, now)
        {
            return None;
        }
        if let Some(h) = self.honest.as_mut() {
            h.blocked = true;
        }
        let decision = Decision::Throttle {
            waiting: live,
            limit: 0,
        };
        self.log.record(LoggedDecision {
            at: now,
            session: session.to_owned(),
            class: class.clone(),
            mode: self.policy.mode,
            decision: decision.clone(),
        });
        if self.tickets.logging() {
            // f49f2: `due_soon` with `only_when_waiting`; `room` now, and `need` once known.
            let due = if only {
                format!(r#","due_soon":{due_soon}"#)
            } else {
                String::new()
            };
            let line = format!(
                r#"{{"event":"returnee_block","t":{:.3},"session":{:?},"class":{:?},"queue_live":{},"queue_waiting":{}{},"room":{}"#,
                now.as_secs_f64(),
                session,
                class,
                live,
                waiting,
                due,
                self.returnee_room(&class)
            );
            self.tickets.returnee_line(session, line);
        }
        Some(decision)
    }

    /// f49f2 `DYNAMO_TICKETS_LOG`: the KV room a new session of `class` has now with no tickets
    /// counted ([`crate::admit::room`]: the smaller of its hot and total room, from the gate's
    /// latest HBM and DRAM with the table's load); 0 before the first gate check.
    fn returnee_room(&self, class: &str) -> u64 {
        let me = self.policy.class(Some(class));
        self.tickets.last_kv.map_or(0, |k| {
            let kv = self.kv_budget(k.hbm_tokens, k.dram_tokens);
            self.step_room(me, &kv)
        })
    }

    /// f49f2 `DYNAMO_TICKETS_LOG`: the KV need of the request of `session` just checked at
    /// classification (its uncached input): writes its waiting `returnee_block` /
    /// `returnee_pass` line with it. Nothing without one.
    pub fn returnee_need(&mut self, session: &str, need: u64) {
        self.tickets.returnee_need(session, need);
    }

    /// f49f `admission.returnee_first`: a live ticket holder of a listed class that fits waits
    /// (`BehindEarlier`, ahead the older holder's waiting session) while an older live ticket of
    /// its class (earlier `issued`, then issue order) has a holder waiting in the lobby, so the
    /// holders get the freed room in the order of their first refusal. Older holders that are
    /// away do not hold it. Not applied to a sub-session whose root is admitted, a resumed
    /// session, or at or past the session's wait limit (its deadline decides then).
    fn returnee_fifo_hold(&self, session: &str, class: &str, now: Time) -> Option<Decision> {
        self.returnee_first(class)?;
        if self.tickets.is_empty() {
            return None;
        }
        let entry = self.sessions.get(session)?;
        if entry.resumed {
            return None;
        }
        let limit = self.policy.class(Some(class)).admission.max_lobby_wait_s;
        let until = entry.arrived.after(Duration::from_secs_f64(limit));
        if now >= until {
            return None;
        }
        let root = self.root_of(session);
        if self
            .sessions
            .get(&root)
            .is_some_and(|r| matches!(r.state, SessionState::Active | SessionState::Idle))
        {
            return None;
        }
        let key = self.ticket_key(session);
        if !self.tickets.is_live(&key, now) {
            return None;
        }
        // f49m: reservations first (oldest reservation first), then the order of first refusal.
        let order =
            |t: &crate::tickets::Ticket| (t.reserved.is_none(), t.reserved, t.issued, t.seq);
        let mine = self.tickets.get(&key).map(order)?;
        let ahead = self
            .returnee_queue(class, now)
            .into_iter()
            .filter(|t| t.session.as_str() != &*key && order(t) < mine)
            .find_map(|t| self.returnee_waiter(t, Some(session)))?;
        Some(Decision::Hold {
            reason: HoldReason::BehindEarlier { ahead },
            until,
        })
    }

    /// f49f `admission.returnee_first`: for the forecast Retry-After of the ticket keyed `key`
    /// (of `class`), the KV need of the live tickets of its class ahead of it (older than its own,
    /// or all of them without one) less what they hold already. 0 for a class not listed.
    fn returnee_ahead_need(&self, key: &str, class: &str, now: Time) -> u64 {
        if self.returnee_first(class).is_none() {
            return 0;
        }
        let mine = self
            .tickets
            .get(key)
            .filter(|_| self.tickets.is_live(key, now))
            .map(|t| (t.issued, t.seq));
        let holds: HashMap<&str, u64> = self
            .tickets
            .holds(self.ticket_cap())
            .into_iter()
            .map(|(t, h)| (t.session.as_str(), h))
            .collect();
        self.returnee_queue(class, now)
            .into_iter()
            .filter(|t| t.session != key && mine.is_none_or(|m| (t.issued, t.seq) < m))
            .map(|t| {
                t.kv_need
                    .saturating_sub(holds.get(t.session.as_str()).copied().unwrap_or(0))
            })
            .sum()
    }

    /// The cap on the KV the tickets hold: `max_hold_frac` of the HBM budget of the gate's latest
    /// check (0 before one).
    fn ticket_cap(&self) -> u64 {
        let Some(cfg) = self.policy.admission.tickets.as_ref() else {
            return 0;
        };
        crate::tickets::Tickets::cap(
            cfg.max_hold_frac,
            self.tickets.last_kv.map_or(0, |k| k.hbm_tokens),
        )
    }

    /// f49m: whether `admission.honest_retry` is on.
    pub fn honest_on(&self) -> bool {
        self.honest.is_some()
    }

    /// f49m `admission.honest_retry` (`None`: off).
    fn honest_cfg(&self) -> Option<&crate::honest_retry::HonestRetryConfig> {
        self.policy.admission.honest_retry.as_ref()
    }

    /// f49m: the reservations' pool, `reserve_frac` of the HBM budget of the gate's latest check
    /// (0 when off or before one).
    fn reserve_cap(&self) -> u64 {
        self.honest_cfg().map_or(0, |c| {
            crate::tickets::Tickets::cap(
                c.reserve_frac,
                self.tickets.last_kv.map_or(0, |k| k.hbm_tokens),
            )
        })
    }

    /// f49m: whether a session waits in the lobby on ticket `t`.
    fn ticket_waiting(&self, t: &crate::tickets::Ticket) -> bool {
        self.returnee_waiter(t, None).is_some()
    }

    /// f49m: `N` for the session keyed `key` (its ticket key) of `class`: its place in its class's
    /// lobby (`lobby`, the session when it waits there; otherwise every waiting session of the
    /// class is ahead), plus the live tickets of its class ahead of its own (all of them without
    /// one) whose holders are away, plus one.
    fn honest_place(&self, lobby: Option<&str>, key: &str, class: &str, now: Time) -> u64 {
        let index = self.sessions.lobby();
        let place = lobby
            .and_then(|s| self.sessions.get(s))
            .filter(|s| s.state == SessionState::Lobby)
            .and_then(|s| self.sessions.lobby_key(s))
            .map_or_else(
                || index.class_len(class),
                |k| index.place_in_class(class, &k),
            );
        let order =
            |t: &crate::tickets::Ticket| (t.reserved.is_none(), t.reserved, t.issued, t.seq);
        let mine = self
            .tickets
            .get(key)
            .filter(|_| self.tickets.is_live(key, now))
            .map(order);
        let away = self
            .tickets
            .iter()
            .filter(|t| t.class == class && t.session != key)
            .filter(|t| self.tickets.is_live(&t.session, now))
            .filter(|t| mine.is_none_or(|m| order(t) < m))
            .filter(|t| !self.ticket_waiting(t))
            .count();
        (place + away + 1) as u64
    }

    /// f49m: the advice for the session keyed `key` of `class`, `place` deep, refused
    /// `until_refusal_s` from `now`; `kv_need` for the ticket forecast (a listed class only).
    fn honest_advice(
        &self,
        key: &str,
        class: &str,
        place: u64,
        kv_need: Option<u64>,
        until_refusal_s: f64,
        now: Time,
    ) -> Option<crate::honest_retry::Advice> {
        let cfg = self.honest_cfg()?;
        let admits = self.honest.as_ref()?.admits(class, now, cfg.window_s);
        let forecast = || {
            let need = kv_need?;
            (self.ticket_class(class).is_some() && self.tickets.last_kv.is_some())
                .then(|| self.ticket_forecast(key, class, need, until_refusal_s, now))
        };
        crate::honest_retry::advise(
            cfg,
            place,
            admits,
            now.as_secs_f64(),
            until_refusal_s,
            forecast,
            self.policy.admission.retry_after_s,
        )
    }

    /// f49m2 `honest_retry.by_reason`: whether it is on.
    fn honest_by_reason(&self) -> bool {
        self.honest_cfg().is_some_and(|c| c.by_reason)
    }

    /// f49m2 `honest_retry.by_reason`: note `session`'s gate answer: a hold's reason (the step
    /// `step` that held it, or the KV fit's own), forgotten on an admission, kept on an expiry.
    /// Nothing when off.
    fn honest_note_hold(&mut self, session: &str, decision: &Decision, step: Option<&'static str>) {
        if !self.honest_by_reason() {
            return;
        }
        let Some(h) = self.honest.as_mut() else {
            return;
        };
        match decision {
            Decision::Hold { reason, .. } => {
                h.note_cause(session, crate::honest_retry::hold_reason(reason, step));
            }
            Decision::Admit(_) => h.clear_cause(session),
            _ => {}
        }
    }

    /// f49m2: the reason `session`'s refusal is priced on (`None` without `by_reason`): its latest
    /// gate hold; with none noted, with its KV need `kv_need` and the gate's KV picture, `kv_fits`
    /// (not KV: with the room there a refusal is another term's) or `kv_short` (KV); else
    /// `unknown` (priced as KV, as f49m). `key` is its ticket key, for the room.
    fn honest_reason(
        &self,
        session: &str,
        key: &str,
        class: &str,
        kv_need: Option<u64>,
        now: Time,
    ) -> Option<crate::honest_retry::Reason> {
        use crate::honest_retry::Reason;
        if !self.honest_by_reason() {
            return None;
        }
        if let Some(r) = self.honest.as_ref().and_then(|h| h.cause(session)) {
            return Some(r);
        }
        let fits = kv_need.and_then(|need| self.honest_kv_fits_now(key, class, need, now));
        Some(match fits {
            Some(true) => Reason {
                name: "kv_fits",
                kv: false,
            },
            Some(false) => Reason {
                name: "kv_short",
                kv: true,
            },
            None => Reason {
                name: "unknown",
                kv: true,
            },
        })
    }

    /// f49m2: whether `kv_need` fits `class`'s room on the gate's latest KV picture now, less the
    /// other live tickets' holds and the returnee queue ahead of ticket key `key` (the ticket
    /// forecast's room at 0 s). `None` before the gate's first check.
    fn honest_kv_fits_now(&self, key: &str, class: &str, kv_need: u64, now: Time) -> Option<bool> {
        let k = self.tickets.last_kv?;
        let me = self.policy.class(Some(class));
        let key = self.ticket_key(key);
        let held = self
            .tickets
            .held_by_others(&key, me.rank, self.ticket_cap())
            .saturating_add(self.returnee_ahead_need(&key, class, now));
        let kv = self.kv_budget(k.hbm_tokens, k.dram_tokens);
        Some(self.step_room(me, &kv).saturating_sub(held) >= kv_need)
    }

    /// f49m2: [`Self::honest_advice`] priced by `reason`
    /// ([`crate::honest_retry::advise_by_reason`]): a non-KV refusal never gets the ticket
    /// forecast; its last resort is `retry_after_s`, else the class's wait limit.
    #[allow(clippy::too_many_arguments)]
    fn honest_advice_by_reason(
        &self,
        reason: crate::honest_retry::Reason,
        key: &str,
        class: &str,
        place: u64,
        kv_need: Option<u64>,
        until_refusal_s: f64,
        now: Time,
    ) -> Option<crate::honest_retry::Advice> {
        let cfg = self.honest_cfg()?;
        let admits = self.honest.as_ref()?.admits(class, now, cfg.window_s);
        let forecast = || {
            let need = kv_need?;
            (self.ticket_class(class).is_some() && self.tickets.last_kv.is_some())
                .then(|| self.ticket_forecast(key, class, need, until_refusal_s, now))
        };
        let wait_limit_s = self.policy.class(Some(class)).admission.max_lobby_wait_s;
        crate::honest_retry::advise_by_reason(
            cfg,
            reason,
            place,
            admits,
            now.as_secs_f64(),
            until_refusal_s,
            forecast,
            self.policy.admission.retry_after_s,
            wait_limit_s,
        )
    }

    /// f49m `DYNAMO_TICKETS_LOG`: one refusal and its advice (`null`s without one).
    /// f49m2 `by_reason`: with `reason` (the reason priced on, and the gate's latest hold) the line
    /// gains `reason`, `reason_kv` and `last_hold`.
    #[allow(clippy::too_many_arguments)]
    fn honest_refusal_diag(
        &mut self,
        session: &str,
        class: &str,
        why: &str,
        advice: Option<crate::honest_retry::Advice>,
        refusals: u32,
        reason: Option<(
            crate::honest_retry::Reason,
            Option<crate::honest_retry::Reason>,
        )>,
        now: Time,
    ) {
        if !self.tickets.logging() {
            return;
        }
        let (seconds, source, place, rate) = match advice {
            Some(a) => (
                format!("{:.3}", a.seconds),
                format!("{:?}", a.source.as_str()),
                a.place.to_string(),
                a.rate
                    .map_or_else(|| "null".to_owned(), |r| format!("{r:.6}")),
            ),
            None => Default::default(),
        };
        let null = |s: String| if s.is_empty() { "null".to_owned() } else { s };
        let line = format!(
            r#"{{"event":"honest_refusal","t":{:.3},"session":{:?},"class":{:?},"why":{:?},"retry_after_s":{},"source":{},"place":{},"rate":{},"refusals":{}}}"#,
            now.as_secs_f64(),
            session,
            class,
            why,
            null(seconds),
            null(source),
            null(place),
            null(rate),
            refusals
        );
        // f49m2 `by_reason`: the reason it was priced on, and the gate's latest hold.
        let line = match reason {
            Some((r, last)) => format!(
                r#"{},"reason":{:?},"reason_kv":{},"last_hold":{}}}"#,
                &line[..line.len() - 1],
                r.name,
                r.kv,
                last.map_or_else(|| "null".to_owned(), |l| format!("{:?}", l.name))
            ),
            None => line,
        };
        self.tickets.log_line(&line);
    }

    /// f49m: a throttle (or `returnee_first` block) of a request whose ticket key is `key`, of
    /// `class`: the honest advice. With `need` (KV, prefill) and a listed class the refusal counts
    /// and issues the key's ticket with the advice. `None` without any advice (no ticket then).
    pub fn honest_throttle(
        &mut self,
        key: &str,
        class: &str,
        need: Option<(u64, u64)>,
        now: Time,
    ) -> Option<f64> {
        let blocked = self.honest.as_ref()?.blocked;
        let why = if blocked {
            "returnee_block"
        } else {
            "throttle"
        };
        let place = self.honest_place(None, key, class, now);
        // f49m2 `by_reason`: priced by the refusal's reason (f49m's advice without it).
        let reason = self.honest_reason(key, key, class, need.map(|n| n.0), now);
        let advice = match reason {
            Some(r) => {
                self.honest_advice_by_reason(r, key, class, place, need.map(|n| n.0), 0.0, now)
            }
            None => self.honest_advice(key, class, place, need.map(|n| n.0), 0.0, now),
        };
        let reason = reason.map(|r| (r, self.honest.as_ref().and_then(|h| h.cause(key))));
        let listed = need.is_some() && self.ticket_class(class).is_some();
        let refusals = match self.honest.as_mut() {
            Some(h) if listed => h.refused(key),
            _ => 0,
        };
        self.honest_refusal_diag(key, class, why, advice, refusals, reason, now);
        let advice = advice?;
        if listed && let Some((kv_need, prefill_need)) = need {
            self.ticket_issue(key, class, kv_need, prefill_need, advice.seconds, now);
        }
        Some(advice.seconds)
    }

    /// f49m: [`Self::honest_throttle`] for a request that issues no ticket (a sub-session without
    /// `tickets.per_root`): the advice alone. `None` when off.
    pub fn honest_throttle_untracked(
        &mut self,
        session: &str,
        class: Option<&str>,
        now: Time,
    ) -> Option<f64> {
        self.honest.as_ref()?;
        let class = self.policy.class(class).name.clone();
        self.honest_throttle(session, &class, None, now)
    }

    /// f49m: the Retry-After waiting `session`'s lobby deadline (`until_refusal_s` from `now`)
    /// carries, kept to issue its ticket (a listed class) and for its refusal's log line. `None`
    /// when off or without any advice.
    pub fn honest_lobby_retry_after(
        &mut self,
        session: &str,
        until_refusal_s: f64,
        now: Time,
    ) -> Option<f64> {
        self.honest.as_ref()?;
        let s = self.sessions.get(session)?;
        let class = s.class.clone();
        let need = self.sessions.total_need(s);
        let key = self.ticket_key(session).into_owned();
        let place = self.honest_place(Some(session), &key, &class, now);
        // f49m2 `by_reason`: priced by the refusal's reason (f49m's advice without it), kept for
        // the refusal's log line.
        let reason = self.honest_reason(session, &key, &class, Some(need), now);
        let advice = match reason {
            Some(r) => self.honest_advice_by_reason(
                r,
                &key,
                &class,
                place,
                Some(need),
                until_refusal_s,
                now,
            ),
            None => self.honest_advice(&key, &class, place, Some(need), until_refusal_s, now),
        };
        if let (Some(r), Some(h)) = (reason, self.honest.as_mut()) {
            h.set_priced(session, r);
        }
        let listed = self.ticket_class(&class).is_some();
        match advice {
            Some(a) => {
                if listed {
                    self.tickets.set_pending(session, a.seconds);
                }
                if let Some(h) = self.honest.as_mut() {
                    h.set_pending(session, a);
                }
            }
            None => {
                self.tickets.take_pending(session);
                if let Some(h) = self.honest.as_mut() {
                    h.take_pending(session);
                }
            }
        }
        advice.map(|a| a.seconds)
    }

    /// f49m: the lobby deadline refused `session`: its refusal counts (a listed class) and its
    /// advice goes to the log, before its ticket is issued ([`Self::ticket_refused`]).
    fn honest_lobby_refused(&mut self, session: &str, now: Time) {
        let Some(h) = self.honest.as_mut() else {
            return;
        };
        let advice = h.take_pending(session);
        // f49m2 `by_reason`: the reason its advice was priced on, and the gate's latest hold.
        let reason = h.take_priced(session).map(|r| (r, h.cause(session)));
        let Some(class) = self.sessions.get(session).map(|s| s.class.clone()) else {
            return;
        };
        let key = self.ticket_key(session).into_owned();
        let listed = self.ticket_class(&class).is_some();
        let refusals = match self.honest.as_mut() {
            Some(h) if listed => h.refused(&key),
            _ => 0,
        };
        self.honest_refusal_diag(session, &class, "lobby", advice, refusals, reason, now);
    }

    /// f49m: after a refusal issued (or kept) the ticket of `session` (of `class`), advised back
    /// `retry_after_s` from `now`: at `max_refusals` refusals of its key it becomes a reservation,
    /// or stays one with its lapse moved to the new advice plus `grace_s`.
    fn honest_reserve(&mut self, session: &str, class: &str, retry_after_s: f64, now: Time) {
        let Some((max, grace_s)) = self.honest_cfg().map(|c| (c.max_refusals, c.grace_s)) else {
            return;
        };
        if self.ticket_class(class).is_none() {
            return;
        }
        let key = self.ticket_key(session).into_owned();
        if self.honest.as_ref().map_or(0, |h| h.refusals(&key)) < max {
            return;
        }
        self.tickets.reserve_cap = self.reserve_cap();
        if self.tickets.reserve(&key, retry_after_s, grace_s, now)
            && let Some(line) = self.reserve_line("reserve_open", &key, now)
        {
            self.tickets.log_line(&line);
        }
    }

    /// f49m `DYNAMO_TICKETS_LOG`: a reservation event of the ticket keyed `key` with the KV it
    /// holds now. `None` without the log or when that ticket is not a reservation.
    fn reserve_line(&self, event: &str, key: &str, now: Time) -> Option<String> {
        if !self.tickets.logging() {
            return None;
        }
        let t = self.tickets.get(key).filter(|t| t.reserved.is_some())?;
        Some(format!(
            r#"{{"event":{:?},"t":{:.3},"session":{:?},"class":{:?},"kv_need":{},"reserved_kv":{},"expires":{:.3}}}"#,
            event,
            now.as_secs_f64(),
            key,
            t.class,
            t.kv_need,
            self.tickets.held(key, self.ticket_cap()),
            t.expires.as_secs_f64()
        ))
    }

    /// f49m: `session` was admitted: its ticket is used up as before ([`Self::ticket_admitted`]);
    /// a reservation it used up is logged, its key's refusals start over, and the admission counts
    /// for its class's rate.
    fn honest_admitted(&mut self, session: &str, now: Time) {
        let key = self.ticket_key(session).into_owned();
        let line = self.reserve_line("reserve_consume", &key, now);
        self.ticket_admitted(session, now);
        let used = self.tickets.get(&key).is_none();
        if used && let Some(line) = line {
            self.tickets.log_line(&line);
        }
        let class = self.sessions.get(session).map(|s| s.class.clone());
        let window = self.honest_cfg().map_or(0.0, |c| c.window_s);
        if let Some(h) = self.honest.as_mut() {
            h.take_pending(session);
            if used {
                h.forget(&key);
            }
            if let Some(class) = class {
                h.admitted(&class, now, window);
            }
        }
    }

    /// f49m: drop the expired tickets, keeping a reservation (with `hold_while_waiting`, any
    /// ticket) while a session that came back on it waits in the lobby; a reservation that lapses
    /// is logged with the KV it held.
    fn honest_expire(&mut self, now: Time) {
        let hold = self
            .policy
            .admission
            .tickets
            .as_ref()
            .is_some_and(|c| c.hold_while_waiting);
        let sessions = &self.sessions;
        let waits = |t: &crate::tickets::Ticket| {
            t.waiting.iter().any(|s| {
                sessions
                    .get(s)
                    .is_some_and(|x| x.state == SessionState::Lobby)
            })
        };
        let mut lapsing: Vec<&str> = self
            .tickets
            .iter()
            .filter(|t| t.reserved.is_some() && now > t.expires && !waits(t))
            .map(|t| t.session.as_str())
            .collect();
        lapsing.sort_unstable();
        let lines: Vec<String> = lapsing
            .into_iter()
            .filter_map(|key| self.reserve_line("reserve_lapse", key, now))
            .collect();
        for line in lines {
            self.tickets.log_line(&line);
        }
        let sessions = &self.sessions;
        self.tickets.expire_keeping(now, |t| {
            (hold || t.reserved.is_some())
                && t.waiting.iter().any(|s| {
                    sessions
                        .get(s)
                        .is_some_and(|x| x.state == SessionState::Lobby)
                })
        });
    }

    /// f49m `DYNAMO_TICKETS_LOG`: each gate check, the reserved KV on free HBM (the smaller of the
    /// reservations' holds and the HBM not in hot use) integrated over time; one `reserve_minute`
    /// line a simulated minute. Nothing without the log.
    fn honest_step(&mut self, now: Time, kv: &KvBudget) {
        if self.honest.is_none() || !self.tickets.logging() {
            return;
        }
        let (reserved, live) = self.tickets.reserved_held(self.ticket_cap());
        let unused = reserved.min(kv.hbm_tokens.saturating_sub(kv.hot_tokens));
        let Some((t_s, kv_s)) = self.honest.as_mut().and_then(|h| h.step(now, unused)) else {
            return;
        };
        let line = format!(
            r#"{{"event":"reserve_minute","t_s":{},"reserved_kv":{},"reservations":{},"unused_kv":{},"reserved_unused_kv_s":{:.0},"reserve_cap":{}}}"#,
            t_s, reserved, live, unused, kv_s, self.tickets.reserve_cap
        );
        self.tickets.log_line(&line);
    }

    /// f49m (c): whether a new session of `class` needing `need` KV tokens fits now: the class's
    /// room on the gate's latest KV picture ([`Self::step_room`]) less the tickets' holds of its
    /// class and above, the unheld need of its class's holders waiting in the lobby (they go
    /// first), and the input of the higher classes' waiting sessions. False before the first gate
    /// check.
    fn honest_room_fits(&self, class: &str, need: u64, now: Time) -> bool {
        let Some(k) = self.tickets.last_kv else {
            return false;
        };
        let me = self.policy.class(Some(class));
        let kv = self.kv_budget(k.hbm_tokens, k.dram_tokens);
        let room = self.step_room(me, &kv);
        let holds = self.tickets.holds(self.ticket_cap());
        let held: u64 = holds
            .iter()
            .filter(|(t, _)| t.rank <= me.rank)
            .map(|(_, h)| *h)
            .sum();
        let waiting: u64 = holds
            .iter()
            .filter(|(t, _)| t.class == class && self.tickets.is_live(&t.session, now))
            .filter(|(t, _)| self.ticket_waiting(t))
            .map(|(t, h)| t.kv_need.saturating_sub(*h))
            .sum();
        let above = self.sessions.lobby().isl_ranked_above(me.rank);
        held.saturating_add(waiting)
            .saturating_add(above)
            .saturating_add(need)
            <= room
    }

    /// Each gate check (with `admission.tickets`): note the KV budget, drop expired tickets, and
    /// write the tickets log's per-minute row.
    fn tickets_step(&mut self, now: Time, kv: &KvBudget) {
        if self.policy.admission.tickets.is_none() {
            return;
        }
        self.tickets.last_kv = Some(*kv);
        // f49m: the reservations' pool (0 off).
        self.tickets.reserve_cap = self.reserve_cap();
        // f49e: the lower ranks' cap, and expiry that keeps waiting holders (both off by default).
        self.ticket_lower_cap_step(now);
        self.tickets_expire(now);
        let cap = self.ticket_cap();
        self.tickets.minute(now, cap);
        self.honest_step(now, kv);
    }

    /// What the live tickets keep from `session` (of `class`) at its gate check.
    fn ticket_terms(&self, session: &str, class: &str, now: Time) -> crate::tickets::TicketTerms {
        if self.policy.admission.tickets.is_none() || self.tickets.is_empty() {
            return crate::tickets::TicketTerms::default();
        }
        let rank = self.policy.class(Some(class)).rank;
        // f49e `per_root`: a sub-session fits and goes to the front on its root's ticket.
        let key = self.ticket_key(session);
        self.tickets.terms(&key, rank, self.ticket_cap(), now)
    }

    /// Issue `session`'s ticket (a listed class only), advised back `retry_after_s` from `now`.
    /// f49m `admission.honest_retry`: at its key's `max_refusals`-th refusal it becomes a
    /// reservation.
    fn ticket_issue(
        &mut self,
        session: &str,
        class: &str,
        kv_need: u64,
        prefill_need: u64,
        retry_after_s: f64,
        now: Time,
    ) {
        self.ticket_issue_base(session, class, kv_need, prefill_need, retry_after_s, now);
        self.honest_reserve(session, class, retry_after_s, now);
    }

    /// [`Self::ticket_issue`] as f49c / f49e issue it.
    fn ticket_issue_base(
        &mut self,
        session: &str,
        class: &str,
        kv_need: u64,
        prefill_need: u64,
        retry_after_s: f64,
        now: Time,
    ) {
        let Some((grace_s, keep, per_root)) = self
            .ticket_class(class)
            .map(|c| (c.grace_s, c.keep_on_rerefusal, c.per_root))
        else {
            return;
        };
        let rank = self.policy.class(Some(class)).rank;
        self.ticket_lower_cap_step(now);
        self.tickets_expire(now);
        let cap = self.ticket_cap();
        // f49e `per_root`: one ticket per root session, keyed by the root.
        let key = self.ticket_key(session).into_owned();
        let live = (keep || per_root) && self.tickets.is_live(&key, now);
        // f49e: a live ticket is kept on a refusal of a sub-session of its root (`per_root`: no new
        // ticket) or of its holder (`keep_on_rerefusal`, which extends its expiry).
        if live && (keep || key != session) {
            self.tickets.keep(
                &key,
                session,
                per_root.then_some(kv_need),
                keep.then_some((retry_after_s, grace_s)),
                cap,
                now,
            );
            return;
        }
        let kv_need = match self.tickets.get(&key) {
            Some(t) if live && per_root => t.kv_need.max(kv_need),
            _ => kv_need,
        };
        self.tickets.issue_by(
            &key,
            session,
            class,
            rank,
            kv_need,
            prefill_need,
            retry_after_s,
            grace_s,
            cap,
            now,
        );
    }

    /// The lobby deadline refused `session` (its request expired in the queue): issue the ticket
    /// the Retry-After its deadline carried advised, if any.
    fn ticket_refused(&mut self, session: &str, now: Time) {
        // f49e `DYNAMO_TICKETS_LOG`: a live holder refused at its deadline (log only).
        self.ticket_rerefused_diag(session, now);
        // f49m: the refusal counts, and its advice goes to the log.
        self.honest_lobby_refused(session, now);
        let Some(retry_after_s) = self.tickets.take_pending(session) else {
            return;
        };
        let Some(s) = self.sessions.get(session) else {
            return;
        };
        let class = s.class.clone();
        let kv_need = self.sessions.total_need(s);
        let prefill_need = if self.meter_uncached() {
            self.session_compute
                .get(session)
                .copied()
                .unwrap_or(s.pending_isl)
        } else {
            s.pending_isl
        };
        self.ticket_issue(session, &class, kv_need, prefill_need, retry_after_s, now);
    }

    /// f49c: the Retry-After a waiting session of a listed class is given with its lobby deadline
    /// (`until_refusal_s` from `now`), kept to issue its ticket if the deadline refuses it.
    /// `fallback` is today's advice (the estimate, else `retry_after_s`). Returns the advice to give
    /// instead of today's: the forecast with `retry_after: forecast`; `None` with `rate` (today's
    /// stands, and is the ticket's) or for a class not listed.
    pub fn ticket_lobby_retry_after(
        &mut self,
        session: &str,
        until_refusal_s: f64,
        fallback: Option<f64>,
        now: Time,
    ) -> Option<f64> {
        let s = self.sessions.get(session)?;
        let class = s.class.clone();
        let need = self.sessions.total_need(s);
        let forecast =
            self.ticket_class(&class)?.retry_after == crate::config::TicketRetryAfter::Forecast;
        let advice = if forecast {
            Some(self.ticket_forecast(session, &class, need, until_refusal_s, now))
        } else {
            fallback
        };
        match advice {
            Some(r) => self.tickets.set_pending(session, r),
            None => {
                self.tickets.take_pending(session);
            }
        }
        advice.filter(|_| forecast)
    }

    /// f49c: a request of top-level `session` (class `class`) was throttled at classification:
    /// issue its ticket with the request's `kv_need` and `prefill_need`. Returns the Retry-After to
    /// refuse it with (the forecast, or `retry_after_s` with `rate`); `None` for a class not listed
    /// or without a Retry-After (no ticket then).
    pub fn ticket_on_throttle(
        &mut self,
        session: &str,
        class: Option<&str>,
        kv_need: u64,
        prefill_need: u64,
        now: Time,
    ) -> Option<f64> {
        // f49f2 `DYNAMO_TICKETS_LOG`: a `returnee_block` line of this arrival ends with its need.
        self.tickets.returnee_need(session, kv_need);
        let class = self.policy.class(class).name.clone();
        // f49m `admission.honest_retry`: the honest advice, and the ticket for a listed class.
        if self.honest.is_some() {
            return self.honest_throttle(session, &class, Some((kv_need, prefill_need)), now);
        }
        let advice = match self.ticket_class(&class)?.retry_after {
            crate::config::TicketRetryAfter::Forecast => {
                Some(self.ticket_forecast(session, &class, kv_need, 0.0, now))
            }
            crate::config::TicketRetryAfter::Rate => self.policy.admission.retry_after_s,
        }?;
        self.ticket_issue(session, &class, kv_need, prefill_need, advice, now);
        Some(advice)
    }

    /// f49c `retry_after: forecast`: seconds from its refusal (`until_refusal_s` from `now`) until
    /// `kv_need` tokens are projected free for a session of `class`. The room now (the gate's
    /// latest HBM and DRAM with the table's load) less the holds of the other live tickets of its
    /// class or above, plus the admitted sessions' predicted releases in order, until it covers the
    /// need; then the prefill window (`prefill_budget.window_s`, 30 s without it) on top, the
    /// calibration factor with `retry_after_estimate.calibrate`, and `min_s` / `max_s`. Never within
    /// `max_s` (600 s without it): `max_s`.
    pub fn ticket_forecast(
        &self,
        session: &str,
        class: &str,
        kv_need: u64,
        until_refusal_s: f64,
        now: Time,
    ) -> f64 {
        let estimate = self.policy.admission.retry_after_estimate.as_ref();
        let max_s = estimate.and_then(|e| e.max_s);
        let horizon = max_s.unwrap_or(TICKET_HORIZON_S);
        let window = self
            .policy
            .admission
            .prefill_budget
            .as_ref()
            .map_or(30.0, |p| p.window_s);
        let me = self.policy.class(Some(class));
        // f49e `per_root`: a sub-session's projection excludes its root's hold.
        let key = self.ticket_key(session);
        let held = self
            .tickets
            .held_by_others(&key, me.rank, self.ticket_cap());
        // f49f `returnee_first`: behind the queue of its class's live tickets ahead of it.
        let held = held + self.returnee_ahead_need(&key, class, now);
        let room = self.tickets.last_kv.map_or(0, |k| {
            let kv = self.kv_budget(k.hbm_tokens, k.dram_tokens);
            self.step_room(me, &kv)
        });
        let need = kv_need as f64;
        let mut free = room as f64 - held as f64;
        let mut at = (free >= need).then_some(0.0);
        if at.is_none() {
            for (dt, tokens) in self.predicted_releases(now) {
                if dt > horizon {
                    break;
                }
                free += tokens as f64;
                if free >= need {
                    at = Some(dt);
                    break;
                }
            }
        }
        let Some(at) = at else {
            return horizon;
        };
        let mut after = (at - until_refusal_s).max(0.0) + window;
        if estimate.is_some_and(|e| e.calibrate) {
            after *= self.ra_factor.get(class).copied().unwrap_or(1.0);
        }
        if let Some(min) = estimate.and_then(|e| e.min_s) {
            after = after.max(min);
        }
        if let Some(max) = max_s {
            after = after.min(max);
        }
        after
    }

    /// The admitted sessions' predicted KV releases, (seconds from `now`, tokens), soonest first:
    /// a session whose turn in flight ends it (`session_final`) at that turn's arrival plus its last
    /// service time; an idle session at its last turn's end plus its think time; overdue ones now.
    /// Sessions with neither are not counted.
    fn predicted_releases(&self, now: Time) -> Vec<(f64, u64)> {
        let finals: std::collections::HashSet<&str> = self
            .requests
            .map
            .values()
            .filter(|r| r.ends_session && r.dispatched)
            .map(|r| r.session.as_str())
            .collect();
        let now_s = now.as_secs_f64();
        let mut out: Vec<(f64, u64)> = self
            .sessions
            .iter()
            .filter(|s| s.is_live())
            .filter_map(|s| {
                let at = if finals.contains(s.id.as_str()) {
                    s.last_turn_arrival.as_secs_f64() + s.last_service_s.unwrap_or(0.0)
                } else if s.state == SessionState::Idle {
                    s.last_turn_end?.as_secs_f64() + s.signals.next_think_s?
                } else {
                    return None;
                };
                Some(((at - now_s).max(0.0), s.context_tokens.max(s.pending_isl)))
            })
            .collect();
        out.sort_by(|a, b| a.0.total_cmp(&b.0));
        out
    }

    /// Each class's lobby now: sessions waiting, the longest current wait and the admission
    /// rate. O(sessions); for the coordinator's log.
    pub fn lobby_stats(&self, now: Time) -> Vec<LobbyStats> {
        let mut by: HashMap<&str, (usize, f64)> = HashMap::new();
        for s in self.sessions.iter() {
            if s.state == SessionState::Lobby || s.away() {
                let e = by.entry(s.class.as_str()).or_insert((0, 0.0));
                e.0 += usize::from(s.state == SessionState::Lobby);
                e.1 = e.1.max(s.lobby_wait_s(now));
            }
        }
        self.policy
            .classes
            .iter()
            .map(|c| {
                let (waiting, oldest) = by.get(c.name.as_str()).copied().unwrap_or((0, 0.0));
                LobbyStats {
                    class: c.name.clone(),
                    waiting,
                    oldest_wait_s: oldest,
                    admit_rate: self.admit_rate(&c.name, now),
                }
            })
            .collect()
    }

    /// f49k: whether `admission.hot_budget: measured` is on.
    pub fn hot_measured_on(&self) -> bool {
        self.hot_measure.is_some()
    }

    /// f49k: the measured hot use, when on.
    pub fn hot_measure(&self) -> Option<&crate::hot_measure::HotMeasure> {
        self.hot_measure.as_ref()
    }

    /// f49k: an engine gauge of `worker` (a refresh of its measurement); whether its room may have
    /// grown. Nothing (false) when off.
    pub fn hot_gauge(
        &mut self,
        worker: dynamo_kv_router::plugins::prelude::WorkerWithDpRank,
        reading: crate::hot_measure::Reading,
    ) -> bool {
        self.hot_measure
            .as_mut()
            .is_some_and(|m| m.gauge(worker, reading))
    }

    /// f49k: the router's LOAD rows (the measurement of the workers without gauges); whether a
    /// worker's room may have grown. Nothing (false) when off.
    pub fn hot_load_rows(
        &mut self,
        rows: &[(
            dynamo_kv_router::plugins::prelude::WorkerWithDpRank,
            crate::hot_measure::Reading,
        )],
    ) -> bool {
        self.hot_measure.as_mut().is_some_and(|m| m.load_rows(rows))
    }

    /// f49k: `request_id` was sent to `worker`; nothing when off.
    pub fn hot_measure_sent(
        &mut self,
        request_id: &str,
        worker: dynamo_kv_router::plugins::prelude::WorkerWithDpRank,
    ) {
        if let Some(m) = self.hot_measure.as_mut() {
            m.sent(request_id, worker);
        }
    }

    /// f49k: `request_id` completed or aborted; nothing when off.
    fn hot_finished(&mut self, request_id: &str) {
        if let Some(m) = self.hot_measure.as_mut() {
            m.finished(request_id);
        }
    }

    /// f49k: the worker whose queue-0 guard admits `entry` now: `decision` holds it on the hot
    /// budget alone (its total budget fits), no session of another class is ahead of it, and a
    /// worker's engine queue is empty with its admission since its last refresh unused.
    fn hot_probe(
        &self,
        entry: &crate::session::Session,
        decision: &Decision,
    ) -> Option<dynamo_kv_router::plugins::prelude::WorkerWithDpRank> {
        let m = self.hot_measure.as_ref()?;
        let Decision::Hold {
            reason:
                HoldReason::KvBudget {
                    needed,
                    hot_room,
                    total_room,
                },
            ..
        } = decision
        else {
            return None;
        };
        if needed <= hot_room || self.sessions.total_need(entry) > *total_room {
            return None;
        }
        let key = self.sessions.lobby_key(entry)?;
        if self
            .sessions
            .lobby()
            .isl_ahead_in_other_classes(&entry.class, &key)
            > 0
        {
            return None;
        }
        m.probe()
    }

    /// The router sent `request_id` to a worker.
    pub fn on_sent(&mut self, request_id: &str) {
        self.requests
            .update(request_id, |record| record.sent = true);
    }

    /// Prompt tokens of the requests waiting at the router: classified, and neither taken by the
    /// host nor sent to a worker (`adaptive.queue_load`).
    pub fn queued_tokens(&self) -> u64 {
        self.requests.queued
    }

    /// The adaptive rule's load pressure from the workers' (`workers`, the mean of active tokens
    /// over KV capacity): with `adaptive.queue_load`, plus the prompt tokens waiting at the router
    /// over the fleet's KV tokens `kv_tokens`.
    pub fn adaptive_load(&self, workers: f64, kv_tokens: u64) -> f64 {
        if self.adaptive.as_ref().is_some_and(|a| a.config.queue_load) {
            workers + self.queued_tokens() as f64 / kv_tokens.max(1) as f64
        } else {
            workers
        }
    }

    /// The first token of `request_id` was produced.
    pub fn on_first_token(&mut self, request_id: &str, now: Time) {
        self.requests.update(request_id, |record| {
            record.first_token.get_or_insert(now);
        });
    }

    /// `request_id` finished. `context_tokens` is the context it leaves behind, when the host
    /// reports it. Mean ITL is not reported by the lifecycle events today, so per-request ITL is
    /// unknown and the windows are judged on TTFT alone (see the README).
    pub fn on_completed(&mut self, request_id: &str, context_tokens: Option<usize>, now: Time) {
        self.on_completed_with_output(request_id, context_tokens, None, now);
    }

    /// [`Self::on_completed`] with the request's output tokens when the host reports them: with
    /// `admission.judge_itl` the request's mean ITL (first token to completion over output tokens
    /// minus one) is judged with its TTFT.
    pub fn on_completed_with_output(
        &mut self,
        request_id: &str,
        context_tokens: Option<usize>,
        output_tokens: Option<usize>,
        now: Time,
    ) {
        self.hot_finished(request_id);
        self.compute.remove(request_id);
        self.placed.remove(request_id);
        let Some(record) = self.requests.remove(request_id) else {
            return;
        };
        if !record.dispatched {
            // Ended without being admitted (an attempt that never reached a worker).
            self.refused_attempt(&record, now);
            self.experience
                .end(&record.root, now, None, false, true, None);
            let refused = self
                .sessions
                .get(&record.session)
                .is_some_and(|s| s.state == SessionState::Lobby);
            self.leave_lobby_if_waiting(&record.session);
            if refused {
                self.ticket_refused(&record.session, now);
            }
            return;
        }
        let ttft = record
            .first_token
            .unwrap_or(now)
            .since(record.arrived)
            .as_secs_f64();
        let itl_s = match (
            self.policy.admission.judge_itl,
            record.first_token,
            output_tokens,
        ) {
            (true, Some(first), Some(out)) if out > 1 => {
                Some(now.since(first).as_secs_f64() / (out - 1) as f64)
            }
            _ => None,
        };
        self.metrics.record(
            &record.class,
            Completion {
                at: now,
                ttft_s: ttft,
                itl_s,
            },
        );
        // On target for the session log: TTFT and, when the host reports output tokens, mean ITL
        // within the class's limits (whatever `judge_itl` says for the windows).
        let on_target = self
            .policy
            .class(Some(&record.class))
            .service_level
            .as_ref()
            .map(|level| {
                let itl = match (record.first_token, output_tokens) {
                    (Some(first), Some(out)) if out > 1 => {
                        Some(now.since(first).as_secs_f64() / (out - 1) as f64)
                    }
                    _ => None,
                };
                ttft <= level.ttft_s && itl.is_none_or(|i| i <= level.itl_s)
            });
        // Uncontended time (F1): the cached prefix at no contention is the larger of what a worker
        // held at classification and the session's previous context.
        let uncontended = self.step.as_ref().and_then(|m| {
            let prompt = record.prompt.unwrap_or(record.isl).max(record.isl);
            let cached = (prompt - record.isl).max(record.prev_context);
            crate::uncontended::request_s(
                m.0.as_ref(),
                self.chunk_tokens,
                prompt,
                cached,
                output_tokens? as u64,
            )
        });
        // f49q `signal: slowdown`: the root's standalone service so far (nothing when off).
        if self.signals.is_some() {
            self.signal_note_service(&record, output_tokens, uncontended);
        }
        // Added delay, phase 1 (the main agent's own requests): router wait, and the time from
        // dispatch to first token over the reference prefill.
        if record.session == record.root
            && let Some(dispatched) = record.dispatched_at
        {
            let waiting = dispatched.since(record.arrived).as_secs_f64();
            let prefill_ref = self.step.as_ref().map_or(0.0, |m| {
                let prompt = record.prompt.unwrap_or(record.isl).max(record.isl);
                let cached = (prompt - record.isl).max(record.prev_context).min(prompt);
                crate::uncontended::prefill_s(
                    m.0.as_ref(),
                    self.chunk_tokens,
                    prompt - cached,
                    cached,
                )
                .unwrap_or(0.0)
            });
            let to_first = record
                .first_token
                .unwrap_or(now)
                .since(dispatched)
                .as_secs_f64();
            let excess = (to_first - prefill_ref).max(0.0);
            let s_ref = uncontended.unwrap_or(0.0);
            self.experience
                .note_delay(&record.root, waiting, excess, s_ref);
            let per_ref = self
                .experience
                .get(&record.root)
                .and_then(|a| self.experience.allowance(&a.class))
                .map_or(0.0, |a| a.1);
            self.accrue_delay(&record.root, now, waiting + excess, per_ref * s_ref);
        }
        self.experience
            .end(&record.root, now, on_target, true, false, uncontended);
        self.absorb_runs(now);
        self.sessions.update(&record.session, |s| {
            s.in_flight = s.in_flight.saturating_sub(1);
            s.turns_served += 1;
            s.last_turn_end = Some(now);
            if let Some(at) = record.dispatched_at {
                s.last_service_s = Some(now.since(at).as_secs_f64());
            }
            if let Some(tokens) = context_tokens {
                s.context_tokens = tokens as u64;
            }
            if s.in_flight == 0 && s.state == SessionState::Active {
                s.state = SessionState::Idle;
            }
        });
        if record.ends_session {
            self.on_session_end(&record.session);
        }
    }

    /// A main-agent attempt that never reached a worker (refused or expired): its whole time in
    /// the router is added delay.
    fn refused_attempt(&mut self, record: &RequestRecord, now: Time) {
        if record.session != record.root {
            return;
        }
        let waited = now.since(record.arrived).as_secs_f64();
        self.experience.note_delay(&record.root, waited, 0.0, 0.0);
        self.accrue_delay(&record.root, now, waited, 0.0);
    }

    /// `request_id` was aborted.
    pub fn on_aborted(&mut self, request_id: &str) {
        let at = self.requests.map.get(request_id).map(|r| r.arrived);
        self.on_aborted_at(request_id, at.unwrap_or_default());
    }

    /// [`Self::on_aborted`] at `now`, so the session log counts the attempt's time in the router
    /// (a lobby wait that ended in expiry).
    pub fn on_aborted_at(&mut self, request_id: &str, now: Time) {
        self.hot_finished(request_id);
        self.compute.remove(request_id);
        self.placed.remove(request_id);
        let Some(record) = self.requests.remove(request_id) else {
            return;
        };
        if !record.dispatched {
            self.refused_attempt(&record, now);
        }
        self.experience
            .end(&record.root, now, None, false, !record.dispatched, None);
        if !record.dispatched {
            // Expired in the queue (its `due_at` passed) or cancelled while waiting.
            let refused = self
                .sessions
                .get(&record.session)
                .is_some_and(|s| s.state == SessionState::Lobby);
            if refused {
                self.note_lost(&record.class);
            }
            self.leave_lobby_if_waiting(&record.session);
            // f49c: the refusal issues the ticket its deadline's Retry-After advised.
            if refused {
                self.ticket_refused(&record.session, now);
            }
            return;
        }
        self.sessions.update(&record.session, |s| {
            s.in_flight = s.in_flight.saturating_sub(1);
            if s.in_flight == 0 && s.state == SessionState::Active {
                s.state = SessionState::Idle;
            }
        });
    }

    fn leave_lobby_if_waiting(&mut self, session: &str) {
        let left = self
            .sessions
            .update(session, |s| {
                let waiting = s.state == SessionState::Lobby;
                if waiting {
                    s.state = SessionState::Ended;
                    s.lobby_expired = true;
                }
                waiting
            })
            .unwrap_or(false);
        if left {
            self.log.forget(session);
        }
    }

    /// The client said the session is over.
    pub fn on_session_end(&mut self, session: &str) {
        // `retry_after_estimate.count_ends` (f40): an admitted top-level session's end frees its
        // budget for a waiting one; it counts at its last turn's end.
        if self
            .union_admission()
            .retry_after_estimate
            .as_ref()
            .is_some_and(|e| e.count_ends)
            && let Some((class, now)) = self
                .sessions
                .get(session)
                .filter(|s| s.parent.is_none() && s.admitted.is_some())
                .and_then(|s| Some((s.class.clone(), s.last_turn_end?)))
        {
            let window = self.estimate_window_s(&class);
            let q = self.ends.entry(class).or_default();
            q.push_back(now);
            while q
                .front()
                .is_some_and(|&t| now.since(t).as_secs_f64() > window)
            {
                q.pop_front();
            }
        }
        if self
            .sessions
            .get(session)
            .is_some_and(|s| s.parent.is_none())
        {
            self.experience.finish(session);
            self.absorb_runs(
                self.sessions
                    .get(session)
                    .and_then(|s| s.last_turn_end)
                    .unwrap_or_default(),
            );
        }
        if self.policy.admission.accounting.verify.is_some() && self.ended.len() < 1 << 14 {
            let context = self.sessions.get(session).map_or(0, |s| s.context_tokens);
            self.ended.push((session.to_owned(), context));
        }
        self.tickets.ended(session);
        if let Some(h) = self.honest.as_mut() {
            h.forget(session);
        }
        self.sessions.update(session, |s| {
            s.state = SessionState::Ended;
            s.lobby_expired = false;
            s.queued_since = None;
        });
        self.log.forget(session);
    }

    /// Judge the windows at `now` and return the class's attainment. Called by whatever exports
    /// metrics, not on the classifier's lock path.
    pub fn snapshot(&mut self, class: &str, now: Time) -> Option<ClassSnapshot> {
        self.metrics.judge(now);
        self.metrics.snapshot(class)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admit::AdmitReason;
    use crate::admit::decide_indexed;

    fn t(s: f64) -> Time {
        Time::from_secs_f64(s)
    }

    fn enforce() -> Scheduler {
        let policy = ServicePolicy {
            mode: Mode::Enforce,
            ..Default::default()
        };
        Scheduler::new(policy)
    }

    #[test]
    fn a_session_waits_then_runs_then_idles() {
        let mut s = enforce();
        let (state, class) = s.on_classify(Some("r1"), "a", None, None, 400, t(0.0));
        assert_eq!((state, class.as_str()), (SessionState::Lobby, "standard"));
        // HBM 1000, nothing live: 400 fits standard's 900.
        let kv = s.kv_budget(1000, 0);
        assert_eq!(s.gate_check("a", t(0.0), &kv), GateAnswer::Go);
        // A check is a read: asking again changes nothing.
        assert_eq!(s.gate_check("a", t(0.0), &kv), GateAnswer::Go);
        assert_eq!(s.sessions().get("a").unwrap().state, SessionState::Lobby);
        s.on_dispatched("r1", "a", t(0.1));
        assert_eq!(s.sessions().get("a").unwrap().state, SessionState::Active);
        s.on_first_token("r1", t(1.0));
        s.on_completed("r1", Some(450), t(2.0));
        let a = s.sessions().get("a").unwrap();
        assert_eq!(
            (a.state, a.context_tokens, a.turns_served),
            (SessionState::Idle, 450, 1)
        );
    }

    #[test]
    fn judge_itl_counts_a_slow_decode_as_a_miss_only_when_switched_on() {
        for judge in [false, true] {
            let mut s = enforce();
            s.policy.admission.judge_itl = judge;
            for i in 0..10 {
                let (r, a) = (format!("r{i}"), format!("s{i}"));
                s.on_classify(Some(&r), &a, None, Some("priority"), 100, t(0.0));
                s.on_dispatched(&r, &a, t(0.0));
                // TTFT 0.5 s (on target), then 101 tokens in 10 s: mean ITL 100 ms, over 50 ms.
                s.on_first_token(&r, t(0.5));
                s.on_completed_with_output(&r, Some(200), Some(101), t(10.5));
            }
            let share = s.metrics.window_share("priority", t(11.0));
            assert_eq!(
                share,
                Some(if judge { 0.0 } else { 1.0 }),
                "judge_itl {judge}"
            );
        }
    }

    #[test]
    fn a_second_session_is_held_until_the_first_leaves_room() {
        let mut s = enforce();
        s.on_classify(Some("r1"), "a", None, None, 600, t(0.0));
        s.on_dispatched("r1", "a", t(0.0));
        s.on_classify(Some("r2"), "b", None, None, 600, t(1.0));
        let kv = s.kv_budget(1000, 0);
        // a is in flight with 600: standard's hot room is 900 - 600 = 300 < 600.
        assert!(
            matches!(s.gate_check("b", t(1.0), &kv), GateAnswer::Hold { until: Some(u) } if u == t(61.0))
        );
        s.on_completed("r1", Some(600), t(5.0));
        // a is idle now: hot room 900, total room 950 - 600 = 350 < 600, still held.
        let kv = s.kv_budget(1000, 0);
        assert!(matches!(
            s.gate_check("b", t(5.0), &kv),
            GateAnswer::Hold { .. }
        ));
        // With DRAM the total budget has room.
        let kv = s.kv_budget(1000, 1000);
        assert_eq!(s.gate_check("b", t(5.0), &kv), GateAnswer::Go);
        let logged: Vec<_> = s
            .decisions()
            .entries()
            .map(|d| d.decision.clone())
            .collect();
        assert!(matches!(
            logged.last(),
            Some(Decision::Admit(AdmitReason::Fits))
        ));
    }

    #[test]
    fn the_client_attributes_reach_the_session_and_session_final_ends_it() {
        let mut s = enforce();
        let attributes = RequestAttributes {
            tenant: Some("acme".into()),
            signals: Signals {
                next_think_s: Some(12.5),
                osl_tokens: Some(300),
                next_turn_tokens: Some(4_000),
                peak_tokens: None,
            },
            session_final: false,
            ..RequestAttributes::default()
        };
        s.on_classify_with(Some("r1"), "a", None, None, &attributes, 400, t(0.0));
        let session = s.sessions().get("a").unwrap();
        assert_eq!(session.tenant.as_deref(), Some("acme"));
        assert_eq!(session.signals, attributes.signals);
        s.on_dispatched("r1", "a", t(0.1));
        s.on_completed("r1", Some(700), t(2.0));
        assert_eq!(s.sessions().get("a").unwrap().state, SessionState::Idle);
        // The last turn says it ends the session: the session ends when that turn completes.
        let last = RequestAttributes {
            session_final: true,
            ..RequestAttributes::default()
        };
        s.on_classify_with(Some("r2"), "a", None, None, &last, 800, t(10.0));
        s.on_dispatched("r2", "a", t(10.0));
        assert_eq!(s.sessions().get("a").unwrap().state, SessionState::Active);
        s.on_completed("r2", Some(900), t(12.0));
        assert_eq!(s.sessions().get("a").unwrap().state, SessionState::Ended);
    }

    #[test]
    fn shadow_mode_logs_but_never_holds() {
        let mut s = Scheduler::new(ServicePolicy::default());
        s.on_classify(Some("r1"), "a", None, None, 5_000, t(0.0));
        let kv = s.kv_budget(1000, 0);
        assert_eq!(s.gate_check("a", t(0.0), &kv), GateAnswer::Go);
        let d = s.decisions().entries().next().unwrap();
        assert_eq!(d.mode, Mode::Shadow);
        assert!(matches!(d.decision, Decision::Hold { .. }));
    }

    #[test]
    fn next_turn_before_late_completion_settles_the_right_turn() {
        let mut s = enforce();
        s.on_classify(Some("r1"), "a", None, None, 100, t(0.0));
        s.on_dispatched("r1", "a", t(0.0));
        // The client sent turn 2 before the router delivered turn 1's Completed.
        s.on_classify(Some("r2"), "a", None, None, 150, t(3.0));
        s.on_dispatched("r2", "a", t(3.0));
        // A repeated `dispatched` for the same request does not count twice.
        s.on_dispatched("r2", "a", t(3.0));
        assert_eq!(s.sessions().get("a").unwrap().in_flight, 2);
        s.on_completed("r1", Some(200), t(3.5));
        let a = s.sessions().get("a").unwrap();
        assert_eq!((a.state, a.in_flight), (SessionState::Active, 1));
        s.on_first_token("r2", t(4.0));
        s.on_completed("r2", Some(350), t(5.0));
        assert_eq!(s.sessions().get("a").unwrap().state, SessionState::Idle);
        // TTFTs come from each request's own arrival: 3.5 s and 1 s.
        let snap = s.snapshot("standard", t(5.0)).unwrap();
        assert_eq!((snap.requests, snap.per_request), (2, Some(1.0)));
    }

    #[test]
    fn a_full_lobby_throttles_new_sessions_only() {
        let mut policy = ServicePolicy {
            mode: Mode::Enforce,
            ..Default::default()
        };
        policy.admission.max_lobby_sessions = 1;
        let mut s = Scheduler::new(policy);
        assert_eq!(s.throttle("a", None, t(0.0)), None);
        s.on_classify(Some("r1"), "a", None, None, 10, t(0.0));
        assert_eq!(
            s.throttle("b", None, t(1.0)),
            Some(Decision::Throttle {
                waiting: 1,
                limit: 1
            })
        );
        // A known session's next turn is never throttled.
        assert_eq!(s.throttle("a", None, t(1.0)), None);
    }

    #[test]
    fn a_turn_that_expires_in_the_lobby_leaves_it() {
        let mut s = enforce();
        s.on_classify(Some("r1"), "a", None, None, 100, t(0.0));
        s.on_aborted("r1");
        let a = s.sessions().get("a").unwrap();
        assert_eq!((a.state, a.in_flight), (SessionState::Ended, 0));
        assert_eq!(s.sessions().kv_load(), (0, 0));
    }

    /// A small deterministic generator, so the property test needs no extra crate.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n.max(1)
        }
    }

    /// L65: the incrementally kept KV totals, lobby size and gate decisions equal a full scan of
    /// the table after every event, across random sequences of new and returning turns, gate
    /// checks, dispatches, first tokens, late completions, expiries, session ends and evictions,
    /// in both lobby orders and with tied class ranks.
    #[test]
    fn indexed_matches_full_scan() {
        use crate::admit::decide;
        use crate::config::LobbyOrder;
        let (mut compared, mut held, mut behind) = (0usize, 0usize, 0usize);
        for seed in 1..=40u64 {
            let mut policy = ServicePolicy {
                mode: Mode::Enforce,
                session_table_capacity: 12,
                ..Default::default()
            };
            if seed % 2 == 0 {
                policy.admission.lobby_order = LobbyOrder::Fair;
            }
            if seed % 5 == 0 {
                // Tie standard's rank to batch's: the walk over a tied class is exercised.
                policy.classes[2].rank = policy.classes[1].rank;
            }
            policy.admission.dram_tokens_per_worker = 0;
            let mut s = Scheduler::new(policy);
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            let classes = [None, Some("priority"), Some("standard"), Some("batch")];
            let mut next_req = 0u64;
            let mut issued: Vec<(String, String)> = Vec::new();
            let mut now = 0.0;
            for _ in 0..400 {
                now += rng.below(5_000) as f64 / 1000.0;
                let at = t(now);
                match rng.below(8) {
                    0..=2 => {
                        let session = format!("s{}", rng.below(16));
                        next_req += 1;
                        let id = format!("r{next_req}");
                        let class = classes[rng.below(4) as usize];
                        s.on_classify(Some(&id), &session, None, class, 1 + rng.below(800), at);
                        issued.push((id, session));
                    }
                    3 => {
                        if let Some((id, session)) = issued
                            .get(rng.below(issued.len() as u64 + 1) as usize)
                            .cloned()
                        {
                            s.on_dispatched(&id, &session, at);
                        }
                    }
                    4 => {
                        if let Some((id, _)) = issued
                            .get(rng.below(issued.len() as u64 + 1) as usize)
                            .cloned()
                        {
                            s.on_first_token(&id, at);
                        }
                    }
                    5 => {
                        if !issued.is_empty() {
                            let (id, _) =
                                issued.swap_remove(rng.below(issued.len() as u64) as usize);
                            s.on_completed(&id, Some(1 + rng.below(1200) as usize), at);
                        }
                    }
                    6 => {
                        if !issued.is_empty() {
                            let (id, _) =
                                issued.swap_remove(rng.below(issued.len() as u64) as usize);
                            s.on_aborted(&id);
                        }
                    }
                    _ => {
                        if rng.below(3) == 0 {
                            s.on_session_end(&format!("s{}", rng.below(16)));
                        }
                    }
                }
                // Totals and lobby size.
                assert_eq!(
                    s.sessions.kv_load(),
                    s.sessions.kv_load_scan(),
                    "seed {seed}"
                );
                let scan = s.lobby_scan();
                assert_eq!(s.sessions.lobby_len(), scan.len(), "seed {seed}");
                // Every waiting session gets the same decision both ways.
                let kv = s.kv_budget(1000 + rng.below(4000), rng.below(3) * 1000);
                for entry in &scan {
                    let session = s.sessions.get(&entry.session).unwrap();
                    let fast = decide_indexed(&s.policy, at, session, &s.sessions, &kv);
                    let full = decide(&s.policy, at, &entry.session, &scan, &kv);
                    assert_eq!(fast, full, "seed {seed} session {}", entry.session);
                    compared += 1;
                    match full {
                        Decision::Hold {
                            reason: crate::admit::HoldReason::BehindEarlier { .. },
                            ..
                        } => behind += 1,
                        Decision::Hold { .. } => held += 1,
                        _ => {}
                    }
                }
            }
        }
        // Not vacuous: many lobby decisions compared, including both kinds of hold.
        assert!(
            compared > 2_000 && held > 100 && behind > 100,
            "compared {compared}, held {held}, behind {behind}"
        );
    }

    #[test]
    fn subagents_take_the_parent_class_and_off_does_nothing() {
        let mut s = enforce();
        s.on_classify(Some("r1"), "main", None, Some("priority"), 10, t(0.0));
        let (_, class) = s.on_classify(Some("r2"), "sub", Some("main"), None, 10, t(1.0));
        assert_eq!(class, "priority");
        let off = ServicePolicy {
            mode: Mode::Off,
            ..Default::default()
        };
        let mut s = Scheduler::new(off);
        s.on_classify(None, "a", None, None, 1_000_000, t(0.0));
        assert_eq!(
            s.gate_check("a", t(0.0), &s.kv_budget(1, 0)),
            GateAnswer::Go
        );
        assert!(s.decisions().is_empty());
    }

    #[test]
    fn placed_cached_tokens_replace_the_classify_estimate_with_the_uncached_meter() {
        let make = |kind| {
            let mut s = with_admission(|ad| {
                ad.prefill_budget = Some(crate::config::PrefillBudgetConfig {
                    meter: kind,
                    ..Default::default()
                })
            });
            s.on_classify(Some("r1"), "std", None, Some("standard"), 3_000, t(0.0));
            s.set_request_prompt("r1", 3_000);
            s.set_request_compute("r1", "std", 600, vec![]);
            s.set_placed("r1", 2_500);
            s
        };
        // uncached: prompt less the host's cached count on the chosen worker (500), not 600.
        let unc = make(crate::config::PrefillMeterKind::Uncached);
        assert_eq!(
            unc.request_prefill_on("r1", None).map(|(t, _)| t),
            Some(500)
        );
        // device (f42): ignored, the classify isl as before.
        let dev = make(crate::config::PrefillMeterKind::Device);
        assert_eq!(
            dev.request_prefill_on("r1", None).map(|(t, _)| t),
            Some(3_000)
        );
        // Completion drops it.
        let mut done = make(crate::config::PrefillMeterKind::Uncached);
        done.on_dispatched("r1", "std", t(0.0));
        done.on_completed("r1", Some(3_000), t(1.0));
        assert!(done.placed.is_empty());
    }

    #[test]
    fn request_prefill_on_uses_the_routed_workers_computed_tokens() {
        use dynamo_kv_router::plugins::prelude::WorkerWithDpRank;
        let (a, b, c) = (
            WorkerWithDpRank::new(1, 0),
            WorkerWithDpRank::new(2, 0),
            WorkerWithDpRank::new(3, 0),
        );
        let meter = |kind| {
            let mut s = with_admission(|ad| {
                ad.prefill_budget = Some(crate::config::PrefillBudgetConfig {
                    meter: kind,
                    ..Default::default()
                })
            });
            s.on_classify(Some("r1"), "std", None, Some("standard"), 3_000, t(0.0));
            s.set_request_compute("r1", "std", 400, vec![(a, 900), (b, 400)]);
            s
        };
        // device (the default, f42): the classify isl, whatever the worker.
        let dev = meter(crate::config::PrefillMeterKind::Device);
        let isl = dev.request_prefill("r1").unwrap();
        assert_eq!(isl, (3_000, "standard".to_owned()));
        for w in [Some(a), Some(b), Some(c), None] {
            assert_eq!(dev.request_prefill_on("r1", w), Some(isl.clone()));
        }
        // uncached: the routed worker's computed tokens; best when the worker is unknown or not listed.
        let unc = meter(crate::config::PrefillMeterKind::Uncached);
        let on = |w| unc.request_prefill_on("r1", w).map(|(t, _)| t);
        assert_eq!(on(Some(a)), Some(900));
        assert_eq!(on(Some(b)), Some(400));
        assert_eq!(on(Some(c)), Some(400));
        assert_eq!(on(None), Some(400));
        // A request never classified: nothing to meter.
        assert_eq!(unc.request_prefill_on("nope", Some(a)), None);
        // Completion drops the record.
        let mut done = meter(crate::config::PrefillMeterKind::Uncached);
        done.on_dispatched("r1", "std", t(0.0));
        done.on_completed("r1", Some(3_000), t(1.0));
        assert_eq!(done.request_prefill_on("r1", Some(a)), None);
        assert!(done.compute.is_empty());
    }

    #[test]
    fn the_onboarding_term_uses_computed_tokens_with_the_uncached_meter() {
        let make = |kind| {
            let mut s = with_admission(|ad| {
                ad.prefill_budget = Some(crate::config::PrefillBudgetConfig {
                    meter: kind,
                    ..Default::default()
                })
            });
            // 3,000 prompt tokens of which 600 will be computed (the rest cached in some tier).
            s.on_classify(Some("r1"), "std", None, Some("standard"), 3_000, t(0.0));
            s.set_request_compute("r1", "std", 600, vec![]);
            s
        };
        let head = |prefill| HeadFacts {
            prefill,
            ..HeadFacts::default()
        };
        // Budget 100 tok/s, 60 used (over half the budget, so the onboarding term decides). device:
        // 3,000 / 30 s = 100 tok/s of onboarding, 160 > 100: held. uncached: 600 / 30 s = 20, 80 <= 100: go.
        let mut dev = make(crate::config::PrefillMeterKind::Device);
        let kv = dev.kv_budget(1_000_000, 0);
        assert!(matches!(
            dev.gate_check_head("std", t(1.0), &kv, head(Some((100.0, 60.0)))),
            GateAnswer::Hold { .. }
        ));
        let mut unc = make(crate::config::PrefillMeterKind::Uncached);
        assert_eq!(
            unc.gate_check_head("std", t(1.0), &kv, head(Some((100.0, 60.0)))),
            GateAnswer::Go
        );
        // Still held when the computed tokens alone are over the room: 2,400 / 30 = 80, 140 > 100.
        let mut big = make(crate::config::PrefillMeterKind::Uncached);
        big.set_request_compute("r1", "std", 2_400, vec![]);
        assert!(matches!(
            big.gate_check_head("std", t(1.0), &kv, head(Some((100.0, 60.0)))),
            GateAnswer::Hold { .. }
        ));
    }

    #[test]
    fn prefill_budget_holds_lower_classes_over_the_budget_and_never_the_top_class() {
        let mut s = with_admission(|a| {
            a.prefill_budget = Some(crate::config::PrefillBudgetConfig::default())
        });
        s.on_classify(Some("r1"), "std", None, Some("standard"), 300, t(0.0));
        s.on_classify(Some("r2"), "pri", None, Some("priority"), 300, t(0.0));
        let kv = s.kv_budget(1_000_000, 0);
        let head = |prefill| HeadFacts {
            prefill,
            ..HeadFacts::default()
        };
        // 300 tokens over the 30 s window is 10 tok/s: 95 used + 10 is over a budget of 100.
        assert!(matches!(
            s.gate_check_head("std", t(1.0), &kv, head(Some((100.0, 95.0)))),
            GateAnswer::Hold { .. }
        ));
        assert_eq!(
            s.gate_check_head("pri", t(1.0), &kv, head(Some((100.0, 95.0)))),
            GateAnswer::Go
        );
        assert_eq!(
            s.gate_check_head("std", t(1.0), &kv, head(Some((100.0, 80.0)))),
            GateAnswer::Go
        );
        // Unknown: never held.
        assert_eq!(
            s.gate_check_head("std", t(1.0), &kv, head(None)),
            GateAnswer::Go
        );
        // Off: never held.
        let mut off = with_admission(|_| {});
        off.on_classify(Some("r1"), "std", None, Some("standard"), 300, t(0.0));
        assert_eq!(
            off.gate_check_head("std", t(1.0), &kv, head(Some((100.0, 95.0)))),
            GateAnswer::Go
        );
    }

    #[test]
    fn prefill_budget_include_top_holds_the_top_class_too() {
        let mut s = with_admission(|a| {
            a.prefill_budget = Some(crate::config::PrefillBudgetConfig {
                include_top: true,
                ..Default::default()
            })
        });
        s.on_classify(Some("r2"), "pri", None, Some("priority"), 300, t(0.0));
        let kv = s.kv_budget(1_000_000, 0);
        let head = |prefill| HeadFacts {
            prefill,
            ..HeadFacts::default()
        };
        assert!(matches!(
            s.gate_check_head("pri", t(1.0), &kv, head(Some((100.0, 95.0)))),
            GateAnswer::Hold { .. }
        ));
        assert_eq!(
            s.gate_check_head("pri", t(1.0), &kv, head(Some((100.0, 80.0)))),
            GateAnswer::Go
        );
    }

    #[test]
    fn the_lookahead_counts_idle_sessions_next_turns_due_within_the_horizon() {
        let mut s = with_admission(|a| {
            a.prefill_budget = Some(crate::config::PrefillBudgetConfig {
                lookahead_s: Some(60.0),
                ..Default::default()
            })
        });
        for (id, think) in [("a", 10.0), ("b", 500.0)] {
            s.on_classify(Some(id), id, None, Some("standard"), 600, t(0.0));
            s.on_dispatched(id, id, t(0.0));
            s.on_completed(id, Some(600), t(1.0));
            s.sessions.update(id, |x| {
                x.signals.next_think_s = Some(think);
                x.signals.next_turn_tokens = Some(1_200);
            });
        }
        // Only a's next turn is due within 60 s either side: 1,200 tokens over 120 s.
        assert_eq!(s.lookahead_rate(t(2.0)), Some(10.0));
        // Cached until the step passes.
        s.sessions
            .update("b", |x| x.signals.next_think_s = Some(1.0));
        assert_eq!(s.lookahead_rate(t(5.0)), Some(10.0));
        assert_eq!(s.lookahead_rate(t(20.0)), Some(20.0));
        // Overdue by more than the horizon: not counted.
        assert_eq!(s.lookahead_rate(t(200.0)), Some(0.0));
    }

    #[test]
    fn calibration_shrinks_the_budget_fast_on_top_class_itl_misses_and_grows_slowly() {
        let mut s = with_admission(|a| {
            a.prefill_budget = Some(crate::config::PrefillBudgetConfig {
                calibrate: true,
                ..Default::default()
            });
            a.judge_itl = true;
        });
        assert_eq!(s.prefill_cal(t(0.0)), 1.0);
        let itl = |s: &mut Scheduler, at: f64, n: usize, itl_s: f64| {
            for _ in 0..n {
                s.metrics.record(
                    "priority",
                    crate::metrics::Completion {
                        at: t(at),
                        ttft_s: 0.1,
                        itl_s: Some(itl_s),
                    },
                );
            }
        };
        // Half over the limit, against 10% allowed: down fast.
        itl(&mut s, 5.0, 10, 0.02);
        itl(&mut s, 5.0, 10, 0.2);
        let down = s.prefill_cal(t(10.0));
        assert!(down < 0.35, "{down}");
        // All within: up, slowly.
        itl(&mut s, 75.0, 50, 0.02);
        let up = s.prefill_cal(t(80.0));
        assert!(up > down && up < down * 3.0, "{down} {up}");
        // Off: always 1.
        let mut off = with_admission(|_| {});
        assert_eq!(off.prefill_cal(t(10.0)), 1.0);
    }

    #[test]
    fn the_loop_does_not_starve_the_lower_classes_for_a_guard_it_cannot_buy() {
        let mut s = with_admission(|a| {
            a.target_loop = Some(crate::config::TargetLoopConfig {
                guard: 0.02,
                min_scale: 0.5,
                guard_gain: 1_000.0,
                infeasible_after_s: 20.0,
                smooth_s: 1.0,
                ..Default::default()
            })
        });
        // Priority at 0.91: over its 0.90 target, under target + guard. Standard waits, never served.
        s.on_classify(Some("w"), "waiting", None, Some("standard"), 100, t(0.0));
        s.loop_step(t(0.0));
        let mut at = 0.0;
        for _ in 0..2 {
            at += 10.0;
            record(&mut s, "priority", at - 1.0, 91, 0.1);
            record(&mut s, "priority", at - 1.0, 9, 100.0);
            s.loop_step(t(at));
        }
        assert_eq!(s.loop_scale("standard"), 0.5);
        assert!(s.take_loop_stats().infeasible.is_none());
        // Past infeasible_after_s at the floor, the guard lets go and standard's demand grows it.
        for _ in 0..2 {
            at += 10.0;
            record(&mut s, "priority", at - 1.0, 91, 0.1);
            record(&mut s, "priority", at - 1.0, 9, 100.0);
            s.loop_step(t(at));
        }
        assert!(
            s.loop_scale("standard") > 0.5,
            "{}",
            s.loop_scale("standard")
        );
    }

    #[test]
    fn best_effort_serves_the_lower_classes_again_once_the_target_is_infeasible() {
        let mut s = with_admission(|a| {
            a.target_loop = Some(crate::config::TargetLoopConfig {
                min_scale: 0.5,
                guard_gain: 1_000.0,
                infeasible_after_s: 20.0,
                smooth_s: 1.0,
                infeasible_hold: false,
                ..Default::default()
            })
        });
        // Priority stuck at 0.80 (target 0.90) whatever standard does; standard waits.
        s.on_classify(Some("w"), "waiting", None, Some("standard"), 100, t(0.0));
        s.loop_step(t(0.0));
        let mut at = 0.0;
        let step = |s: &mut Scheduler, at: f64| {
            record(s, "priority", at - 1.0, 80, 0.1);
            record(s, "priority", at - 1.0, 20, 100.0);
            s.loop_step(t(at));
        };
        for _ in 0..4 {
            at += 10.0;
            step(&mut s, at);
        }
        let stats = s.take_loop_stats();
        assert_eq!(stats.infeasible.as_deref(), Some("priority"));
        // No hold for standard in best effort: the waiting session goes.
        let kv = s.kv_budget(1_000_000, 0);
        assert_eq!(
            s.gate_check_head("waiting", t(at + 2.0), &kv, HeadFacts::default()),
            GateAnswer::Go
        );
        // The guard protects only the floor (0.80 - 0.02): standard's demand grows the scale.
        for _ in 0..3 {
            at += 10.0;
            step(&mut s, at);
        }
        assert!(
            s.loop_scale("standard") > 0.5,
            "{}",
            s.loop_scale("standard")
        );
        assert_eq!(s.take_loop_stats().infeasible.as_deref(), Some("priority"));
    }

    #[test]
    fn best_effort_reprobes_the_real_target_and_ratchets_the_protected_level() {
        let mut s = with_admission(|a| {
            a.target_loop = Some(crate::config::TargetLoopConfig {
                min_scale: 0.5,
                guard_gain: 1_000.0,
                infeasible_after_s: 20.0,
                smooth_s: 1.0,
                infeasible_hold: false,
                reprobe_s: Some(30.0),
                ..Default::default()
            })
        });
        s.on_classify(Some("w"), "waiting", None, Some("standard"), 100, t(0.0));
        s.loop_step(t(0.0));
        let step = |s: &mut Scheduler, at: f64, met: usize| {
            record(s, "priority", at - 1.0, met, 0.1);
            record(s, "priority", at - 1.0, 100 - met, 100.0);
            s.loop_step(t(at));
        };
        let mut at = 0.0;
        for _ in 0..3 {
            at += 10.0;
            step(&mut s, at, 80);
        }
        assert_eq!(s.lp.infeasible.as_deref(), Some("priority"));
        assert_eq!(s.lp.floor_att, Some(0.8));
        // Priority recovers part way: the protected level follows it up.
        at += 10.0;
        step(&mut s, at, 85);
        assert!(s.lp.floor_att.unwrap() > 0.8, "{:?}", s.lp.floor_att);
        // Thirty seconds after the declaration the state is cleared and the target enforced again.
        for _ in 0..2 {
            at += 10.0;
            step(&mut s, at, 85);
        }
        assert!(s.lp.infeasible.is_none());
    }

    #[test]
    fn the_loop_grows_only_while_its_class_has_demand() {
        let mut s =
            with_admission(|a| a.target_loop = Some(crate::config::TargetLoopConfig::default()));
        s.loop_step(t(0.0));
        record(&mut s, "standard", 5.0, 20, 0.1);
        record(&mut s, "priority", 5.0, 20, 0.1);
        // Nobody waits: no growth however far standard is over its set point.
        s.loop_step(t(15.0));
        assert_eq!(s.loop_scale("standard"), 1.0);
        // A standard session waits in the lobby: it grows.
        s.on_classify(Some("w"), "waiting", None, Some("standard"), 100, t(16.0));
        s.loop_step(t(30.0));
        assert!(s.loop_scale("standard") > 1.0);
    }

    #[test]
    fn priority_waiting_cuts_the_lower_classes_scale() {
        let mut s = with_admission(|a| {
            a.target_loop = Some(crate::config::TargetLoopConfig {
                priority_wait_cut: 0.5,
                ..Default::default()
            })
        });
        s.loop_step(t(0.0));
        s.loop_step(t(15.0));
        let base = s.loop_scale("standard");
        // A priority session waits in the lobby: standard and batch get half their scale.
        s.on_classify(Some("p"), "waiting", None, Some("priority"), 100, t(16.0));
        s.loop_step(t(30.0));
        let now = s.lp.scale;
        assert_eq!(s.loop_scale("standard"), now * 0.5);
        assert_eq!(s.loop_scale("batch"), now * 0.5);
        assert_eq!(s.loop_scale("priority"), 1.0);
        assert!(base > 0.0);
    }

    fn record(s: &mut Scheduler, class: &str, at: f64, n: usize, ttft_s: f64) {
        for _ in 0..n {
            s.metrics.record(
                class,
                crate::metrics::Completion {
                    at: t(at),
                    ttft_s,
                    itl_s: None,
                },
            );
        }
    }

    #[test]
    fn target_loop_grows_on_standard_surplus_and_falls_when_priority_is_short() {
        let mut s =
            with_admission(|a| a.target_loop = Some(crate::config::TargetLoopConfig::default()));
        assert_eq!(s.loop_scale("standard"), 1.0);
        s.loop_step(t(0.0));
        record(&mut s, "priority", 5.0, 20, 0.1);
        record(&mut s, "standard", 5.0, 20, 0.1);
        s.on_classify(Some("w"), "waiting", None, Some("standard"), 100, t(1.0));
        s.loop_step(t(15.0));
        let grown = s.loop_scale("standard");
        assert!(grown > 1.0, "{grown}");
        // The loop's class and the classes below it follow the scale; priority does not.
        assert_eq!(s.loop_scale("batch"), grown);
        assert_eq!(s.loop_scale("priority"), 1.0);
        assert_eq!(s.pace_scale("standard"), grown);
        // Priority misses: the guard pulls the scale down although standard is over its target.
        record(&mut s, "priority", 20.0, 200, 100.0);
        s.loop_step(t(30.0));
        assert!(s.loop_scale("standard") < grown);
    }

    #[test]
    fn target_schedule_changes_a_class_target_at_its_time() {
        let mut s = with_admission(|a| {
            a.target_loop = Some(crate::config::TargetLoopConfig::default());
            a.target_schedule = vec![crate::config::TargetChange {
                at_s: 100.0,
                class: "priority".into(),
                attainment: 0.95,
            }];
        });
        let target = |s: &Scheduler| {
            s.policy()
                .class(Some("priority"))
                .service_level
                .as_ref()
                .unwrap()
                .attainment
        };
        s.loop_step(t(0.0));
        s.loop_step(t(50.0));
        assert_eq!(target(&s), 0.9);
        s.loop_step(t(150.0));
        assert_eq!(target(&s), 0.95);
    }

    #[test]
    fn an_infeasible_target_holds_new_lower_class_sessions_and_says_so() {
        let mut s = with_admission(|a| {
            a.target_loop = Some(crate::config::TargetLoopConfig {
                min_scale: 0.5,
                guard_gain: 1_000.0,
                infeasible_after_s: 20.0,
                smooth_s: 1.0,
                ..Default::default()
            })
        });
        s.loop_step(t(0.0));
        record(&mut s, "priority", 5.0, 50, 100.0);
        s.loop_step(t(10.0));
        assert_eq!(s.loop_scale("standard"), 0.5);
        record(&mut s, "priority", 25.0, 50, 100.0);
        s.loop_step(t(20.0));
        record(&mut s, "priority", 35.0, 50, 100.0);
        s.loop_step(t(40.0));
        s.loop_step(t(50.0));
        let stats = s.take_loop_stats();
        assert_eq!(stats.infeasible.as_deref(), Some("priority"));
        assert!(stats.infeasible_s > 0.0);
        s.on_classify(Some("r1"), "std", None, Some("standard"), 100, t(41.0));
        s.on_classify(Some("r2"), "pri", None, Some("priority"), 100, t(41.0));
        let kv = s.kv_budget(1_000_000, 0);
        assert!(matches!(
            s.gate_check_head("std", t(42.0), &kv, HeadFacts::default()),
            GateAnswer::Hold { .. }
        ));
        assert_eq!(
            s.gate_check_head("pri", t(42.0), &kv, HeadFacts::default()),
            GateAnswer::Go
        );
        // Priority recovers: the hold lifts.
        record(&mut s, "priority", 60.0, 2_000, 0.1);
        s.loop_step(t(65.0));
        assert!(s.take_loop_stats().infeasible.is_none());
    }

    fn tickets_policy(f: impl FnOnce(&mut crate::config::TicketsConfig)) -> Scheduler {
        with_admission(|a| {
            let mut cfg = crate::config::TicketsConfig::default();
            f(&mut cfg);
            a.tickets = Some(cfg);
        })
    }

    /// f49e (a) `hold_while_waiting`: a holder back in the lobby keeps its ticket past its expiry
    /// until its deadline refuses it; without the option the ticket lapses in its wait.
    #[test]
    fn f49e_a_waiting_holder_does_not_lapse() {
        let run = |hold: bool| {
            let mut s = tickets_policy(|c| {
                c.max_hold_frac = 1.0;
                c.hold_while_waiting = hold;
            });
            // Due at 30, expires at 60; the holder is back at its due time and waits.
            s.ticket_issue("w", "standard", 200, 0, 30.0, t(0.0));
            s.on_classify(Some("rw"), "w", None, Some("standard"), 200, t(30.0));
            let kv = s.kv_budget(1000, 0);
            s.gate_check("w", t(70.0), &kv);
            s
        };
        let off = run(false);
        assert!(off.tickets().is_empty(), "today: lapses 30 s into the wait");
        let mut on = run(true);
        let k = on.tickets().get("w").unwrap();
        assert!(k.waiting_hold && on.tickets().is_live("w", t(70.0)));
        assert_eq!((k.returns, k.due, k.expires), (1, t(30.0), t(60.0)));
        // Its terms: front of line, still.
        assert!(on.ticket_terms("w", "standard", t(70.0)).front);
        // The deadline refuses it (no advice pending: no re-issue): the next check drops it.
        on.on_aborted_at("rw", t(85.0));
        on.on_classify(Some("rn"), "n", None, Some("standard"), 10, t(86.0));
        let kv = on.kv_budget(1000, 0);
        on.gate_check("n", t(86.0), &kv);
        assert!(on.tickets().is_empty());
    }

    /// f49e (b) `per_root`: one ticket per root; a sub-session uses its root's live ticket (front,
    /// not throttled) and its refusal issues nothing new; with none live it issues the root's.
    #[test]
    fn f49e_b_sub_sessions_use_their_roots_ticket() {
        let mut s = tickets_policy(|c| {
            c.max_hold_frac = 1.0;
            c.per_root = true;
        });
        s.on_classify(Some("r1"), "r", None, Some("standard"), 300, t(0.0));
        let kv = s.kv_budget(1000, 0);
        s.gate_check("r", t(0.0), &kv);
        assert_eq!(
            s.ticket_lobby_retry_after("r", 61.0, None, t(0.0)),
            Some(30.0)
        );
        s.on_aborted_at("r1", t(61.0));
        let root = s.tickets().get("r").unwrap().clone();
        assert_eq!(
            (root.due, root.expires, root.kv_need),
            (t(91.0), t(121.0), 300)
        );
        // A sub-session waits (its root is not admitted): front, on the root's ticket.
        s.on_classify(
            Some("a1"),
            "r:a1",
            Some("r"),
            Some("standard"),
            100,
            t(62.0),
        );
        assert_eq!(s.sessions().get("r:a1").unwrap().state, SessionState::Lobby);
        let terms = s.ticket_terms("r:a1", "standard", t(62.0));
        assert!(terms.front);
        assert_eq!(
            terms.held_kv, 0,
            "its root's hold is not counted against it"
        );
        assert!(s.tickets().get("r").unwrap().waiting.contains("r:a1"));
        // A new sub-session is not throttled while its root's ticket is live; others are.
        s.policy.admission.max_lobby_sessions = 1;
        assert_eq!(
            s.throttle_with_parent("r:a2", Some("r"), Some("standard"), t(63.0)),
            None
        );
        assert!(matches!(
            s.throttle_with_parent("x", None, Some("standard"), t(63.0)),
            Some(Decision::Throttle { .. })
        ));
        s.policy.admission.max_lobby_sessions = 0;
        // Its refusal under the live root ticket issues nothing new.
        assert!(
            s.ticket_lobby_retry_after("r:a1", 60.0, None, t(62.0))
                .is_some()
        );
        s.on_aborted_at("a1", t(110.0));
        assert_eq!(s.tickets().len(), 1);
        assert!(s.tickets().get("r:a1").is_none());
        let k = s.tickets().get("r").unwrap();
        assert_eq!(
            (k.seq, k.issued, k.due, k.expires, k.kv_need),
            (root.seq, root.issued, root.due, root.expires, 300)
        );
        // The root's ticket lapsed: a sub-session refusal issues the root's, at the larger need.
        s.on_classify(
            Some("a2"),
            "r:a1",
            Some("r"),
            Some("standard"),
            500,
            t(200.0),
        );
        assert!(
            s.ticket_lobby_retry_after("r:a1", 60.0, None, t(200.0))
                .is_some()
        );
        s.on_aborted_at("a2", t(260.0));
        assert_eq!(s.tickets().len(), 1);
        let k = s.tickets().get("r").unwrap();
        assert_eq!((k.issued, k.kv_need), (t(260.0), 500));
        // Without `per_root` the same sub-session refusal issues its own ticket.
        let mut plain = tickets_policy(|c| c.max_hold_frac = 1.0);
        plain.on_classify(Some("r1"), "r", None, Some("standard"), 300, t(0.0));
        plain.on_aborted_at("r1", t(61.0));
        plain.on_classify(
            Some("a1"),
            "r:a1",
            Some("r"),
            Some("standard"),
            100,
            t(62.0),
        );
        assert!(
            plain
                .ticket_lobby_retry_after("r:a1", 60.0, None, t(62.0))
                .is_some()
        );
        plain.on_aborted_at("a1", t(110.0));
        assert!(plain.tickets().get("r:a1").is_some());
    }

    /// f49e (c) `keep_on_rerefusal`: a re-refusal keeps the ticket's age, due and returns and
    /// extends its expiry; without it the ticket is replaced.
    #[test]
    fn f49e_c_a_rerefusal_keeps_age_and_due_and_extends_expiry() {
        let run = |keep: bool| {
            let mut s = tickets_policy(|c| {
                c.max_hold_frac = 1.0;
                c.keep_on_rerefusal = keep;
            });
            s.ticket_issue("w", "standard", 200, 0, 30.0, t(0.0));
            s.on_classify(Some("rw"), "w", None, Some("standard"), 200, t(30.0));
            s.ticket_issue("w", "standard", 250, 0, 40.0, t(50.0));
            s.tickets().get("w").unwrap().clone()
        };
        let k = run(true);
        assert_eq!(
            (k.issued, k.due, k.returns, k.expires, k.seq, k.kv_need),
            (t(0.0), t(30.0), 1, t(120.0), 1, 200)
        );
        let k = run(false);
        assert_eq!(
            (k.issued, k.due, k.returns, k.expires, k.seq, k.kv_need),
            (t(50.0), t(90.0), 1, t(120.0), 2, 250)
        );
    }

    /// f49e (e) `standard_cap_when_priority_below`: while priority is under its target the
    /// standard tickets' holds are capped; priority's are not.
    #[test]
    fn f49e_e_the_standard_hold_is_capped_while_priority_is_below_target() {
        let mut s = tickets_policy(|c| {
            c.max_hold_frac = 1.0;
            c.standard_cap_when_priority_below = Some(0.1);
        });
        s.ticket_issue("st", "standard", 500, 0, 100.0, t(0.0));
        s.ticket_issue("pt", "priority", 400, 0, 100.0, t(0.5));
        s.on_classify(Some("rn"), "n", None, Some("batch"), 10, t(1.0));
        let holds = |s: &mut Scheduler, at: f64| {
            let kv = s.kv_budget(1000, 0);
            s.gate_check("n", t(at), &kv);
            s.tickets()
                .holds(s.ticket_cap())
                .into_iter()
                .map(|(k, h)| (k.session.clone(), h))
                .collect::<Vec<_>>()
        };
        // Priority's attainment unknown: nothing capped.
        assert_eq!(
            holds(&mut s, 1.0),
            vec![("pt".to_owned(), 400), ("st".to_owned(), 500)]
        );
        // Under its 0.9 target: standard holds at most 0.1 x 1000.
        s.lp.att.insert("priority".to_owned(), 0.5);
        assert_eq!(
            holds(&mut s, 2.0),
            vec![("pt".to_owned(), 400), ("st".to_owned(), 100)]
        );
        // Back over it: the cap lifts.
        s.lp.att.insert("priority".to_owned(), 0.95);
        assert_eq!(
            holds(&mut s, 3.0),
            vec![("pt".to_owned(), 400), ("st".to_owned(), 500)]
        );
    }

    /// f49e: with the options absent and no tickets log, the same answers and decision log as f49c
    /// on the ticket paths.
    #[test]
    fn f49e_options_off_change_nothing() {
        let run = |s: &mut Scheduler| {
            let mut out = Vec::new();
            s.on_classify(Some("r1"), "r", None, Some("standard"), 300, t(0.0));
            let kv = s.kv_budget(1000, 0);
            out.push(s.gate_check("r", t(0.0), &kv));
            s.ticket_lobby_retry_after("r", 61.0, None, t(0.0));
            s.on_aborted_at("r1", t(61.0));
            s.on_classify(
                Some("a1"),
                "r:a1",
                Some("r"),
                Some("standard"),
                100,
                t(62.0),
            );
            s.on_classify(Some("r2"), "r", None, Some("standard"), 300, t(91.0));
            for at in [92.0, 125.0, 150.0] {
                let kv = s.kv_budget(1000, 0);
                out.push(s.gate_check("r", t(at), &kv));
                out.push(s.gate_check("r:a1", t(at), &kv));
            }
            let tickets: Vec<_> = s
                .tickets()
                .holds(1000)
                .into_iter()
                .map(|(k, h)| (k.session.clone(), k.issued, k.due, k.expires, h))
                .collect();
            (
                out,
                tickets,
                s.decisions().entries().cloned().collect::<Vec<_>>(),
            )
        };
        let mut base = tickets_policy(|_| {});
        let mut explicit = tickets_policy(|c| {
            c.hold_while_waiting = false;
            c.per_root = false;
            c.keep_on_rerefusal = false;
            c.standard_cap_when_priority_below = None;
        });
        assert_eq!(run(&mut base), run(&mut explicit));
        assert!(!base.tickets().logging());
    }

    /// f49f: tickets as the policy pairs them with `returnee_first` (no KV hold, per root, kept on
    /// a re-refusal), and `returnee_first` set by `f` (absent with `None`).
    fn returnee_policy(f: Option<fn(&mut crate::config::ReturneeFirstConfig)>) -> Scheduler {
        with_admission(|a| {
            a.tickets = Some(crate::config::TicketsConfig {
                max_hold_frac: 0.0,
                per_root: true,
                keep_on_rerefusal: true,
                ..Default::default()
            });
            a.returnee_first = f.map(|f| {
                let mut cfg = crate::config::ReturneeFirstConfig::default();
                f(&mut cfg);
                cfg
            });
        })
    }

    /// f49f: the gate's answer and, when held, the hold reason it logged.
    fn returnee_gate(s: &mut Scheduler, session: &str, at: f64) -> (GateAnswer, Option<String>) {
        let kv = s.kv_budget(100_000, 0);
        let answer = s.gate_check(session, t(at), &kv);
        let ahead = match s.decisions().entries().last().map(|e| &e.decision) {
            Some(Decision::Hold {
                reason: HoldReason::BehindEarlier { ahead },
                ..
            }) => Some(ahead.clone()),
            _ => None,
        };
        (answer, ahead)
    }

    /// f49f (1, 2): a new top-level session of a listed class is throttled while its class has a
    /// live ticket, and gets its own ticket (a Retry-After behind the queue); once none is live it
    /// is not throttled. Cap 0: the tickets hold nothing and still go to the front.
    #[test]
    fn f49f_new_sessions_wait_behind_live_tickets() {
        let mut s = returnee_policy(Some(|_| {}));
        // `old` refused: due at 30, lapses at 60. It needs 90k of the 100k room.
        s.ticket_issue("old", "standard", 90_000, 0, 30.0, t(0.0));
        let kv = s.kv_budget(100_000, 0);
        s.gate_check("none", t(1.0), &kv);
        assert_eq!(s.ticket_cap(), 0);
        assert!(
            s.tickets()
                .holds(s.ticket_cap())
                .iter()
                .all(|(_, h)| *h == 0)
        );
        assert!(s.ticket_terms("old", "standard", t(1.0)).front);
        assert_eq!(s.ticket_terms("old", "standard", t(1.0)).held_kv, 0);
        assert!(matches!(
            s.throttle_with_parent("new", None, Some("standard"), t(2.0)),
            Some(Decision::Throttle {
                waiting: 1,
                limit: 0
            })
        ));
        assert!(matches!(
            s.decisions().entries().last().map(|e| &e.decision),
            Some(Decision::Throttle { limit: 0, .. })
        ));
        // Its Retry-After counts `old`'s need ahead of it: not the bare prefill window.
        let behind = s.ticket_forecast("new", "standard", 20_000, 0.0, t(2.0));
        let mut off = returnee_policy(None);
        off.ticket_issue("old", "standard", 90_000, 0, 30.0, t(0.0));
        off.gate_check("none", t(1.0), &kv);
        let bare = off.ticket_forecast("new", "standard", 20_000, 0.0, t(2.0));
        assert!(behind > bare, "{behind} > {bare}");
        // The hooks issue its ticket: it joins the queue, and is not throttled again.
        let advice = s.ticket_on_throttle("new", Some("standard"), 20_000, 0, t(2.0));
        assert_eq!(advice, Some(behind));
        assert!(s.tickets().is_live("new", t(2.0)));
        assert!(s.tickets().get("new").unwrap().issued > s.tickets().get("old").unwrap().issued);
        assert_eq!(
            s.throttle_with_parent("new", None, Some("standard"), t(3.0)),
            None
        );
        // A refused session back without a live ticket counts as new.
        s.ticket_issue("gone", "standard", 100, 0, 1.0, t(0.0));
        assert!(
            s.throttle_with_parent("gone", None, Some("standard"), t(40.0))
                .is_some()
        );
        // `old` is back and admitted, `new`'s and `gone`'s tickets lapse: not throttled.
        s.on_classify(Some("ro"), "old", None, Some("standard"), 100, t(30.0));
        assert_eq!(returnee_gate(&mut s, "old", 30.0).0, GateAnswer::Go);
        s.on_dispatched("ro", "old", t(30.0));
        assert!(s.tickets().get("old").is_none());
        let lapse = s.tickets().get("new").unwrap().expires.as_secs_f64() + 1.0;
        assert_eq!(
            s.throttle_with_parent("n2", None, Some("standard"), t(lapse)),
            None
        );
        // The live-ticket check alone: with one live ticket of its class, throttled.
        let mut one = returnee_policy(Some(|_| {}));
        one.ticket_issue("old", "standard", 100, 0, 30.0, t(0.0));
        assert!(
            one.throttle_with_parent("n3", None, Some("standard"), t(59.0))
                .is_some()
        );
        assert_eq!(
            one.throttle_with_parent("n3", None, Some("standard"), t(61.0)),
            None
        );
    }

    /// f49f (3): sub-sessions and the next turns of admitted sessions are continuation work: never
    /// blocked.
    #[test]
    fn f49f_continuation_work_is_never_blocked() {
        let mut s = returnee_policy(Some(|_| {}));
        admitted(&mut s, "r1", "root", "standard", 0.0);
        s.ticket_issue("old", "standard", 100, 0, 30.0, t(1.0));
        assert_eq!(
            s.throttle_with_parent("root:a1", Some("root"), Some("standard"), t(2.0)),
            None
        );
        assert_eq!(
            s.throttle_with_parent("root", None, Some("standard"), t(2.0)),
            None
        );
        assert!(
            s.throttle_with_parent("fresh", None, Some("standard"), t(2.0))
                .is_some()
        );
        // The admitted root's sub-session in the lobby, on the root's ticket (`per_root`), is not
        // held behind the older holder waiting.
        s.on_classify(
            Some("a1"),
            "root:a1",
            Some("root"),
            Some("standard"),
            100,
            t(5.0),
        );
        s.on_aborted_at("a1", t(6.0));
        s.ticket_issue("root:a1", "standard", 100, 0, 30.0, t(6.0));
        s.on_classify(Some("ro"), "old", None, Some("standard"), 100, t(30.0));
        s.on_classify(
            Some("a2"),
            "root:a1",
            Some("root"),
            Some("standard"),
            100,
            t(35.0),
        );
        assert!(s.tickets().is_live("root", t(35.0)));
        assert_eq!(returnee_gate(&mut s, "root:a1", 35.0).0, GateAnswer::Go);
    }

    /// f49f (4, 5): a younger waiting holder holds while an older holder waits in the lobby, and
    /// is admitted once the older one is admitted or leaves; an older holder away does not hold
    /// it. Without the rule the younger one goes first.
    #[test]
    fn f49f_holders_are_admitted_in_order_of_first_refusal() {
        let setup = |rule: bool| {
            let mut s = returnee_policy(if rule { Some(|_| {}) } else { None });
            s.ticket_issue("a", "standard", 300, 0, 10.0, t(0.0));
            s.ticket_issue("b", "standard", 300, 0, 10.0, t(1.0));
            s
        };
        // Both back and waiting: `b` (younger) holds behind `a`, `a` goes.
        for leaves in [false, true] {
            let mut s = setup(true);
            s.on_classify(Some("ra"), "a", None, Some("standard"), 300, t(10.0));
            s.on_classify(Some("rb"), "b", None, Some("standard"), 300, t(11.0));
            let (answer, ahead) = returnee_gate(&mut s, "b", 12.0);
            assert!(matches!(answer, GateAnswer::Hold { .. }));
            assert_eq!(ahead.as_deref(), Some("a"));
            assert_eq!(returnee_gate(&mut s, "a", 12.0).0, GateAnswer::Go);
            assert!(matches!(
                returnee_gate(&mut s, "b", 12.5).0,
                GateAnswer::Hold { .. }
            ));
            if leaves {
                // `a`'s deadline refuses it: away, it no longer holds `b`.
                s.on_aborted_at("ra", t(13.0));
                assert!(s.sessions().get("a").unwrap().away());
            } else {
                s.on_dispatched("ra", "a", t(13.0));
                assert!(s.tickets().get("a").is_none());
            }
            assert_eq!(returnee_gate(&mut s, "b", 13.0).0, GateAnswer::Go);
        }
        // (5) `a` away (not back yet): `b` is not held.
        let mut s = setup(true);
        s.on_classify(Some("rb"), "b", None, Some("standard"), 300, t(11.0));
        assert_eq!(returnee_gate(&mut s, "b", 12.0).0, GateAnswer::Go);
        // An older holder never holds a younger one from behind: `a` goes with `b` waiting.
        let mut s = setup(true);
        s.on_classify(Some("rb"), "b", None, Some("standard"), 300, t(11.0));
        s.on_classify(Some("ra"), "a", None, Some("standard"), 300, t(11.5));
        assert_eq!(returnee_gate(&mut s, "a", 12.0).0, GateAnswer::Go);
        // Without the rule: front of line for both, `b` goes with `a` waiting.
        let mut s = setup(false);
        s.on_classify(Some("ra"), "a", None, Some("standard"), 300, t(10.0));
        s.on_classify(Some("rb"), "b", None, Some("standard"), 300, t(11.0));
        assert_eq!(returnee_gate(&mut s, "b", 12.0).0, GateAnswer::Go);
        // `max_block_s`: a ticket issued longer ago no longer holds a younger holder.
        let mut s = returnee_policy(Some(|c| c.max_block_s = Some(5.0)));
        s.ticket_issue("a", "standard", 300, 0, 10.0, t(0.0));
        s.ticket_issue("b", "standard", 300, 0, 10.0, t(4.0));
        s.on_classify(Some("ra"), "a", None, Some("standard"), 300, t(4.5));
        s.on_classify(Some("rb"), "b", None, Some("standard"), 300, t(4.5));
        assert!(matches!(
            returnee_gate(&mut s, "b", 4.6).0,
            GateAnswer::Hold { .. }
        ));
        assert_eq!(returnee_gate(&mut s, "b", 5.5).0, GateAnswer::Go);
        assert!(
            s.throttle_with_parent("new", None, Some("standard"), t(5.5))
                .is_some(),
            "`b`'s ticket still blocks"
        );
        assert_eq!(
            s.throttle_with_parent("new", None, Some("standard"), t(9.5)),
            None
        );
    }

    /// f49f (6): a class not listed is unaffected: no throttle, no order among its holders.
    #[test]
    fn f49f_an_unlisted_class_is_unaffected() {
        let mut s = returnee_policy(Some(|c| c.classes = vec!["priority".into()]));
        s.ticket_issue("a", "standard", 300, 0, 10.0, t(0.0));
        s.ticket_issue("b", "standard", 300, 0, 10.0, t(1.0));
        assert_eq!(
            s.throttle_with_parent("new", None, Some("standard"), t(2.0)),
            None
        );
        s.on_classify(Some("ra"), "a", None, Some("standard"), 300, t(10.0));
        s.on_classify(Some("rb"), "b", None, Some("standard"), 300, t(11.0));
        assert_eq!(returnee_gate(&mut s, "b", 12.0).0, GateAnswer::Go);
        // A priority ticket does not block standard either; it blocks priority.
        s.ticket_issue("p", "priority", 300, 0, 10.0, t(12.0));
        assert_eq!(
            s.throttle_with_parent("new", None, Some("standard"), t(13.0)),
            None
        );
        assert!(
            s.throttle_with_parent("newp", None, Some("priority"), t(13.0))
                .is_some()
        );
    }

    /// f49f: with `returnee_first` absent, or listing no class, the same answers, tickets,
    /// Retry-After and decision log, and the f49e outcomes (the younger holder goes first, new
    /// sessions are not throttled); no tickets log.
    #[test]
    fn f49f_options_off_change_nothing() {
        let run = |s: &mut Scheduler| {
            let mut out = Vec::new();
            s.ticket_issue("a", "standard", 300, 0, 10.0, t(0.0));
            s.ticket_issue("b", "standard", 300, 0, 10.0, t(1.0));
            s.on_classify(Some("ra"), "a", None, Some("standard"), 300, t(10.0));
            s.on_classify(Some("rb"), "b", None, Some("standard"), 300, t(11.0));
            let kv = s.kv_budget(1000, 0);
            out.push(format!("{:?}", s.gate_check("b", t(12.0), &kv)));
            out.push(format!("{:?}", s.gate_check("a", t(12.0), &kv)));
            out.push(format!(
                "{:?}",
                s.throttle_with_parent("new", None, Some("standard"), t(12.0))
            ));
            out.push(format!(
                "{:?}",
                s.ticket_on_throttle("new", Some("standard"), 500, 0, t(12.0))
            ));
            out.push(format!(
                "{:?}",
                s.ticket_lobby_retry_after("b", 40.0, None, t(12.0))
            ));
            s.on_aborted_at("rb", t(52.0));
            let tickets: Vec<_> = s
                .tickets()
                .holds(1000)
                .into_iter()
                .map(|(k, h)| (k.session.clone(), k.issued, k.due, k.expires, h))
                .collect();
            (
                out,
                tickets,
                s.decisions().entries().cloned().collect::<Vec<_>>(),
            )
        };
        let mut base = returnee_policy(None);
        let mut empty = returnee_policy(Some(|c| c.classes.clear()));
        let off = run(&mut base);
        assert_eq!(off, run(&mut empty));
        assert_eq!(off.0[0], "Go");
        assert_eq!(off.0[2], "None");
        assert!(!base.tickets().logging());
        // The rule on changes the run.
        assert_ne!(off, run(&mut returnee_policy(Some(|_| {}))));
    }

    /// f49f2 `only_when_waiting`: a new session waits only behind a holder in the lobby or due
    /// back within `due_soon_s`; an away holder not due soon lets it through. FIFO among waiting
    /// holders is unchanged; without the option (f49f) the away holder blocks it.
    #[test]
    fn f49f2_only_when_waiting_blocks_for_a_waiting_or_due_soon_holder() {
        // `old` refused at 0: due at 30, lapses at 60; away.
        let mut s = returnee_policy(Some(|c| c.only_when_waiting = true));
        s.ticket_issue("old", "standard", 100, 0, 30.0, t(0.0));
        assert_eq!(
            s.throttle_with_parent("new", None, Some("standard"), t(2.0)),
            None
        );
        assert!(s.tickets().get("new").is_none());
        // Due back within 10 s: throttled.
        assert!(matches!(
            s.throttle_with_parent("new", None, Some("standard"), t(20.0)),
            Some(Decision::Throttle {
                waiting: 1,
                limit: 0
            })
        ));
        // f49f: throttled while `old` is away, whenever it is due.
        let mut f = returnee_policy(Some(|_| {}));
        f.ticket_issue("old", "standard", 100, 0, 30.0, t(0.0));
        assert!(
            f.throttle_with_parent("new", None, Some("standard"), t(2.0))
                .is_some()
        );
        // `due_soon_s` 2: not yet at 20 s, at 28.5 s.
        let mut d = returnee_policy(Some(|c| {
            c.only_when_waiting = true;
            c.due_soon_s = 2.0;
        }));
        d.ticket_issue("old", "standard", 100, 0, 30.0, t(0.0));
        assert_eq!(
            d.throttle_with_parent("new", None, Some("standard"), t(20.0)),
            None
        );
        assert!(
            d.throttle_with_parent("new", None, Some("standard"), t(28.5))
                .is_some()
        );
        // `old` back early and waiting in the lobby: throttled; a younger waiting holder still
        // holds behind it.
        let mut w = returnee_policy(Some(|c| c.only_when_waiting = true));
        w.ticket_issue("old", "standard", 100, 0, 30.0, t(0.0));
        w.ticket_issue("b", "standard", 100, 0, 30.0, t(1.0));
        w.on_classify(Some("ro"), "old", None, Some("standard"), 100, t(5.0));
        assert!(
            w.throttle_with_parent("new", None, Some("standard"), t(6.0))
                .is_some()
        );
        w.on_classify(Some("rb"), "b", None, Some("standard"), 100, t(6.0));
        let (answer, ahead) = returnee_gate(&mut w, "b", 7.0);
        assert!(matches!(answer, GateAnswer::Hold { .. }));
        assert_eq!(ahead.as_deref(), Some("old"));
        assert!(w.returnee_room("standard") > 0);
        // `max_block_s` unchanged: a ticket issued longer ago does not block.
        let mut m = returnee_policy(Some(|c| {
            c.only_when_waiting = true;
            c.max_block_s = Some(5.0);
        }));
        m.ticket_issue("old", "standard", 100, 0, 1.0, t(0.0));
        assert!(
            m.throttle_with_parent("new", None, Some("standard"), t(4.0))
                .is_some()
        );
        assert_eq!(
            m.throttle_with_parent("new", None, Some("standard"), t(6.0)),
            None
        );
    }

    /// f49f2: with `only_when_waiting` false, any `due_soon_s` decides as f49f.
    #[test]
    fn f49f2_option_false_is_f49f() {
        let run = |s: &mut Scheduler| {
            let mut out = Vec::new();
            s.ticket_issue("old", "standard", 100, 0, 30.0, t(0.0));
            s.ticket_issue("b", "standard", 100, 0, 30.0, t(1.0));
            for (n, at) in [("n1", 2.0), ("n2", 20.0)] {
                out.push(format!(
                    "{:?}",
                    s.throttle_with_parent(n, None, Some("standard"), t(at))
                ));
            }
            s.on_classify(Some("ro"), "old", None, Some("standard"), 100, t(25.0));
            s.on_classify(Some("rb"), "b", None, Some("standard"), 100, t(26.0));
            out.push(format!("{:?}", returnee_gate(s, "b", 27.0)));
            out.push(format!("{:?}", returnee_gate(s, "old", 27.0)));
            out.push(format!(
                "{:?}",
                s.throttle_with_parent("n3", None, Some("standard"), t(27.0))
            ));
            out.push(format!(
                "{:?}",
                s.ticket_on_throttle("n3", Some("standard"), 100, 0, t(27.0))
            ));
            (out, s.decisions().entries().cloned().collect::<Vec<_>>())
        };
        let f49f = run(&mut returnee_policy(Some(|_| {})));
        let off = run(&mut returnee_policy(Some(|c| {
            c.only_when_waiting = false;
            c.due_soon_s = 99.0;
        })));
        assert_eq!(f49f, off);
        assert!(f49f.0[0].starts_with("Some(Throttle"));
        let on = run(&mut returnee_policy(Some(|c| c.only_when_waiting = true)));
        assert_ne!(f49f, on);
        assert_eq!(on.0[0], "None");
    }

    #[test]
    fn tickets_off_or_empty_decide_as_before() {
        // The same events with tickets absent and with tickets on but none issued.
        let run = |s: &mut Scheduler| -> Vec<GateAnswer> {
            let mut out = Vec::new();
            s.on_classify(Some("r1"), "a", None, Some("standard"), 600, t(0.0));
            let kv = s.kv_budget(1000, 0);
            out.push(s.gate_check("a", t(0.0), &kv));
            s.on_dispatched("r1", "a", t(0.0));
            s.on_classify(Some("r2"), "b", None, Some("standard"), 600, t(1.0));
            s.on_classify(Some("r3"), "c", None, Some("priority"), 300, t(2.0));
            s.on_classify(Some("r4"), "d", None, Some("batch"), 100, t(3.0));
            for (id, at) in [("b", 4.0), ("c", 4.0), ("d", 4.0), ("b", 70.0)] {
                let kv = s.kv_budget(1000, 0);
                out.push(s.gate_check(id, t(at), &kv));
            }
            s.on_completed("r1", Some(600), t(80.0));
            for id in ["b", "c", "d"] {
                let kv = s.kv_budget(1000, 500);
                out.push(s.gate_check(id, t(81.0), &kv));
            }
            out
        };
        let mut off = enforce();
        let mut on = tickets_policy(|_| {});
        let answers = run(&mut off);
        assert_eq!(answers, run(&mut on));
        assert!(on.tickets().is_empty());
        assert_eq!(off.decisions().len(), on.decisions().len());
        // Nothing held and not in front: the held decision is the plain one.
        let kv = on.kv_budget(1000, 0);
        for id in ["b", "c", "d"] {
            let session = on.sessions().get(id).unwrap();
            assert_eq!(
                crate::admit::decide_indexed_scaled(
                    &on.policy,
                    t(5.0),
                    session,
                    on.sessions(),
                    &kv,
                    1.0
                ),
                crate::admit::decide_indexed_held(
                    &on.policy,
                    t(5.0),
                    session,
                    on.sessions(),
                    &kv,
                    1.0,
                    0,
                    false
                )
            );
        }
    }

    #[test]
    fn tickets_config_parses_with_defaults() {
        let c: crate::config::TicketsConfig =
            serde_json::from_str(r#"{"classes":["priority"],"retry_after":"rate"}"#).unwrap();
        assert_eq!(c.classes, vec!["priority".to_owned()]);
        assert_eq!(c.retry_after, crate::config::TicketRetryAfter::Rate);
        assert_eq!((c.grace_s, c.max_hold_frac), (30.0, 0.3));
        assert_eq!(crate::config::AdmissionPolicy::default().tickets, None);
    }

    #[test]
    fn tickets_issue_on_a_throttle_and_spare_the_holder_the_next() {
        let mut s = tickets_policy(|_| {});
        s.policy.admission.max_lobby_sessions = 1;
        s.on_classify(Some("r1"), "a", None, Some("standard"), 100, t(0.0));
        assert!(matches!(
            s.throttle("b", Some("standard"), t(1.0)),
            Some(Decision::Throttle { .. })
        ));
        let r = s
            .ticket_on_throttle("b", Some("standard"), 500, 400, t(1.0))
            .unwrap();
        // No gate check yet: no room known and nothing predicted to free, so the horizon.
        assert_eq!(r, 600.0);
        let k = s.tickets().get("b").unwrap();
        assert_eq!(
            (k.kv_need, k.prefill_need, k.due, k.expires),
            (500, 400, t(601.0), t(631.0))
        );
        // Back while its ticket is live: not throttled.
        assert_eq!(s.throttle("b", Some("standard"), t(10.0)), None);
        // A class not listed: no ticket.
        assert_eq!(
            s.ticket_on_throttle("c", Some("batch"), 500, 400, t(1.0)),
            None
        );
        // `rate`: `retry_after_s`, and no ticket without it.
        let mut rate = tickets_policy(|c| c.retry_after = crate::config::TicketRetryAfter::Rate);
        assert_eq!(
            rate.ticket_on_throttle("b", Some("standard"), 500, 400, t(1.0)),
            None
        );
        assert!(rate.tickets().is_empty());
        rate.policy.admission.retry_after_s = Some(45.0);
        assert_eq!(
            rate.ticket_on_throttle("b", Some("standard"), 500, 400, t(1.0)),
            Some(45.0)
        );
        assert_eq!(rate.tickets().get("b").unwrap().due, t(46.0));
    }

    #[test]
    fn tickets_issue_when_the_lobby_deadline_refuses() {
        let mut s = tickets_policy(|_| {});
        s.on_classify(Some("r1"), "w", None, Some("standard"), 300, t(0.0));
        let kv = s.kv_budget(1000, 0);
        s.gate_check("w", t(0.0), &kv);
        // Room now (900 hot, 950 total) covers 300: due at the refusal plus the 30 s window.
        assert_eq!(
            s.ticket_lobby_retry_after("w", 61.0, None, t(0.0)),
            Some(30.0)
        );
        assert!(s.tickets().is_empty(), "issued at the refusal, not before");
        s.on_aborted_at("r1", t(61.0));
        let k = s.tickets().get("w").unwrap();
        assert_eq!(
            (k.kv_need, k.prefill_need, k.due, k.class.as_str()),
            (300, 300, t(91.0), "standard")
        );
        // `rate`: today's advice stands and is the ticket's (a refusal reported as a completion
        // that never reached a worker too).
        let mut rate = tickets_policy(|c| c.retry_after = crate::config::TicketRetryAfter::Rate);
        rate.on_classify(Some("r1"), "w", None, Some("standard"), 300, t(0.0));
        assert_eq!(
            rate.ticket_lobby_retry_after("w", 61.0, Some(45.0), t(0.0)),
            None
        );
        rate.on_completed("r1", None, t(61.0));
        assert_eq!(rate.tickets().get("w").unwrap().due, t(106.0));
        // No advice: no ticket.
        rate.on_classify(Some("r2"), "v", None, Some("standard"), 300, t(0.0));
        assert_eq!(rate.ticket_lobby_retry_after("v", 61.0, None, t(0.0)), None);
        rate.on_aborted_at("r2", t(61.0));
        assert!(rate.tickets().get("v").is_none());
    }

    #[test]
    fn a_standard_ticket_never_reduces_prioritys_room_and_a_priority_ticket_reduces_standards() {
        let fits = |tickets: &[(&str, &str, u64)], class: &str, frac: f64| {
            let mut s = tickets_policy(|c| c.max_hold_frac = frac);
            for (id, c, need) in tickets {
                s.ticket_issue(id, c, *need, 0, 100.0, t(0.0));
            }
            s.on_classify(Some("r1"), "n", None, Some(class), 600, t(1.0));
            let kv = s.kv_budget(1000, 0);
            s.gate_check("n", t(1.0), &kv) == GateAnswer::Go
        };
        assert!(fits(&[], "standard", 1.0));
        assert!(fits(&[("st", "standard", 500)], "priority", 1.0));
        assert!(!fits(&[("st", "standard", 500)], "standard", 1.0));
        assert!(!fits(&[("pt", "priority", 500)], "standard", 1.0));
        assert!(fits(&[], "batch", 1.0));
        assert!(!fits(&[("st", "standard", 200)], "batch", 1.0));
        // Priority 600 with a priority ticket of 500: 1000 - 500 hot room.
        assert!(!fits(&[("pt", "priority", 500)], "priority", 1.0));
        // Under a cap, an older standard ticket never takes priority's share of it.
        let mut s = tickets_policy(|_| {});
        s.ticket_issue("st", "standard", 300, 0, 100.0, t(0.0));
        s.ticket_issue("pt", "priority", 300, 0, 100.0, t(0.5));
        let kv = s.kv_budget(1000, 0);
        s.on_classify(Some("r1"), "n", None, Some("batch"), 10, t(1.0));
        s.gate_check("n", t(1.0), &kv);
        let holds: Vec<(String, u64)> = s
            .tickets()
            .holds(s.ticket_cap())
            .into_iter()
            .map(|(k, h)| (k.session.clone(), h))
            .collect();
        assert_eq!(holds, vec![("pt".to_owned(), 300), ("st".to_owned(), 0)]);
    }

    #[test]
    fn ticket_holds_are_capped_at_max_hold_frac_of_hbm() {
        // Cap 0.3 x 1000 = 300: the priority ticket holds 300 of its 500, standard's nothing.
        let mut s = tickets_policy(|_| {});
        s.ticket_issue("st", "standard", 300, 0, 100.0, t(0.0));
        s.ticket_issue("pt", "priority", 500, 0, 100.0, t(0.5));
        s.on_classify(Some("r1"), "n", None, Some("standard"), 600, t(1.0));
        let kv = s.kv_budget(1000, 0);
        // 900 - 300 = 600 hot room, 950 - 300 = 650 total: fits.
        assert_eq!(s.gate_check("n", t(1.0), &kv), GateAnswer::Go);
        let held: u64 = s
            .tickets()
            .holds(s.ticket_cap())
            .into_iter()
            .map(|(_, h)| h)
            .sum();
        assert_eq!((s.ticket_cap(), held), (300, 300));
        // 601 would not.
        let mut s = tickets_policy(|_| {});
        s.ticket_issue("pt", "priority", 500, 0, 100.0, t(0.5));
        s.on_classify(Some("r1"), "n", None, Some("standard"), 601, t(1.0));
        assert!(matches!(
            s.gate_check("n", t(1.0), &kv),
            GateAnswer::Hold { .. }
        ));
    }

    #[test]
    fn a_ticket_holder_returns_to_the_front_and_its_admission_uses_the_ticket() {
        let mut s = tickets_policy(|c| c.max_hold_frac = 1.0);
        // x waits first in standard's lobby; w was refused at 0 with a ticket due at 30.
        s.on_classify(Some("rx"), "x", None, Some("standard"), 100, t(0.0));
        s.ticket_issue("w", "standard", 200, 0, 30.0, t(0.0));
        // Back early: in front of x, admitted at once.
        s.on_classify(Some("rw"), "w", None, Some("standard"), 200, t(20.0));
        let k = s.tickets().get("w").unwrap();
        assert_eq!((k.returns, k.early), (1, true));
        let kv = s.kv_budget(1000, 0);
        assert_eq!(s.gate_check("w", t(20.0), &kv), GateAnswer::Go);
        // x fits around w's hold (100 + 200 within 900).
        assert_eq!(s.gate_check("x", t(20.0), &kv), GateAnswer::Go);
        s.on_dispatched("rw", "w", t(20.0));
        assert!(
            s.tickets().get("w").is_none(),
            "admission uses up the ticket"
        );
        // Without a ticket the same return waits behind x.
        let mut plain = tickets_policy(|_| {});
        plain.on_classify(Some("rx"), "x", None, Some("standard"), 100, t(0.0));
        plain.on_classify(Some("rw"), "w", None, Some("standard"), 200, t(20.0));
        let kv = plain.kv_budget(1000, 0);
        assert!(matches!(
            plain.gate_check("w", t(20.0), &kv),
            GateAnswer::Hold { .. }
        ));
    }

    #[test]
    fn a_ticket_expires_after_its_grace_and_its_session_end_drops_it() {
        let mut s = tickets_policy(|c| c.max_hold_frac = 1.0);
        // Due at 30, expires at 60.
        s.ticket_issue("pt", "priority", 500, 0, 30.0, t(0.0));
        s.on_classify(Some("r1"), "n", None, Some("standard"), 600, t(1.0));
        let kv = s.kv_budget(1000, 0);
        assert!(matches!(
            s.gate_check("n", t(59.0), &kv),
            GateAnswer::Hold { .. }
        ));
        assert_eq!(s.tickets().len(), 1);
        assert_eq!(s.gate_check("n", t(60.5), &kv), GateAnswer::Go);
        assert!(s.tickets().is_empty());
        admitted(&mut s, "e1", "e", "standard", 70.0);
        s.ticket_issue("e", "standard", 100, 0, 30.0, t(70.0));
        s.on_session_end("e");
        assert!(s.tickets().is_empty());
    }

    #[test]
    fn the_ticket_forecast_walks_predicted_releases_less_earlier_holds() {
        let mut s = tickets_policy(|c| c.max_hold_frac = 1.0);
        admitted(&mut s, "a1", "a", "standard", 0.0);
        admitted(&mut s, "b1", "b", "standard", 0.0);
        s.on_completed("a1", Some(400), t(10.0));
        s.on_completed("b1", Some(300), t(20.0));
        // a frees at 10 + 20 = 30 s, b at 20 + 30 = 50 s.
        s.sessions
            .update("a", |x| x.signals.next_think_s = Some(20.0));
        s.sessions
            .update("b", |x| x.signals.next_think_s = Some(30.0));
        s.on_classify(Some("rw"), "w", None, Some("standard"), 700, t(20.0));
        let kv = s.kv_budget(1000, 0);
        s.gate_check("w", t(20.0), &kv);
        // Room now 950 - 700 = 250; + 400 at 30 s is 650 < 700; + 300 at 50 s is 950: 30 s from
        // now, plus the 30 s prefill window.
        let r = |s: &Scheduler, until| s.ticket_forecast("w", "standard", 700, until, t(20.0));
        assert!((r(&s, 0.0) - 60.0).abs() < 1e-9);
        // Refused 10 s from now: 20 s after it, plus the window.
        assert!((r(&s, 10.0) - 50.0).abs() < 1e-9);
        // An earlier priority ticket holding 200 leaves 750 at 50 s.
        s.ticket_issue("pt", "priority", 200, 0, 100.0, t(20.0));
        assert!((r(&s, 0.0) - 60.0).abs() < 1e-9);
        // A batch ticket holds nothing against standard's projection.
        s.ticket_issue("bt", "batch", 900, 0, 100.0, t(20.0));
        assert!((r(&s, 0.0) - 60.0).abs() < 1e-9);
        // Holding 300, 650 never covers 700: the horizon, or `max_s`.
        s.ticket_issue("pt", "priority", 300, 0, 100.0, t(20.0));
        assert_eq!(r(&s, 0.0), 600.0);
        s.policy.admission.retry_after_estimate = Some(crate::config::RetryAfterEstimate {
            max_s: Some(400.0),
            ..Default::default()
        });
        assert_eq!(r(&s, 0.0), 400.0);
        // A small need fits the room now: the window alone, at least `min_s`.
        s.policy.admission.retry_after_estimate = Some(crate::config::RetryAfterEstimate {
            min_s: Some(45.0),
            ..Default::default()
        });
        s.ticket_issue("pt", "priority", 0, 0, 100.0, t(20.0));
        assert_eq!(s.ticket_forecast("w", "standard", 100, 0.0, t(20.0)), 45.0);
    }

    fn with_admission(f: impl FnOnce(&mut crate::config::AdmissionPolicy)) -> Scheduler {
        let mut policy = ServicePolicy {
            mode: Mode::Enforce,
            ..Default::default()
        };
        f(&mut policy.admission);
        Scheduler::new(policy)
    }

    /// f50: a switch runs the other mode's policy and keeps the sessions and the lobby.
    #[test]
    fn mix_select_switch_keeps_sessions_and_the_lobby() {
        use crate::mix_select::{MixMode, MixSelectConfig, ModeParts};
        let mut policy = ServicePolicy {
            mode: Mode::Enforce,
            ..Default::default()
        };
        policy.admission.lobby_order = LobbyOrder::Signal;
        policy.admission.target_loop = Some(Default::default());
        policy.admission.session_signal = Some(Default::default());
        policy.admission.starvation_override = Some(Default::default());
        policy.mix_select = Some(MixSelectConfig {
            window_s: 30.0,
            min_dwell_s: 0.0,
            step_s: 30.0,
            start: MixMode::Pg,
            b0: ModeParts {
                lobby_order: LobbyOrder::Aging,
                target_loop: true,
                max_lobby_wait_s: [("priority".to_owned(), 25.0)].into_iter().collect(),
                ..ModeParts::default()
            },
            pg: ModeParts {
                lobby_order: LobbyOrder::Signal,
                session_signal: true,
                starvation_override: true,
                ..ModeParts::default()
            },
            ..MixSelectConfig::default()
        });
        assert_eq!(policy.validate(), Ok(()));
        let mut s = Scheduler::new(policy);
        assert_eq!(s.mix_mode(), Some(MixMode::Pg));
        assert_eq!(s.policy().admission.lobby_order, LobbyOrder::Signal);
        assert!(
            s.policy().admission.target_loop.is_none(),
            "pg runs without B0's loop"
        );
        assert!(s.union_admission().target_loop.is_some());
        for (i, (class, isl)) in [
            ("standard", 100),
            ("standard", 100),
            ("priority", 1000),
            ("priority", 1000),
        ]
        .into_iter()
        .enumerate()
        {
            s.on_classify(
                Some(&format!("r{i}")),
                &format!("s{i}"),
                None,
                Some(class),
                isl,
                t(i as f64),
            );
        }
        let sessions = s.sessions().iter().count();
        let lobby = s.sessions().lobby().ordered().len();
        assert!(lobby > 0);
        // Priority holds 2000 of 2200 tokens: over `high` at the first step (no dwell here).
        s.mix_step(t(10.0));
        assert_eq!(s.mix_mode(), Some(MixMode::B0));
        assert_eq!(s.policy().admission.lobby_order, LobbyOrder::Aging);
        assert!(s.policy().admission.target_loop.is_some());
        assert!(s.policy().admission.starvation_override.is_none());
        assert_eq!(
            s.policy()
                .class(Some("priority"))
                .admission
                .max_lobby_wait_s,
            25.0
        );
        assert_eq!(s.sessions().iter().count(), sessions, "no session dropped");
        assert_eq!(
            s.sessions().lobby().ordered().len(),
            lobby,
            "the lobby is re-keyed, not emptied"
        );
        assert!(
            s.sessions().iter().all(|x| x.order_key.is_none()),
            "no signal keys under aging"
        );
        // Not due again before step_s.
        s.mix_step(t(20.0));
        assert_eq!(s.mix_mode(), Some(MixMode::B0));
        // Back to pg: the signal order again, the lobby kept.
        s.apply_mix_mode(MixMode::Pg, t(40.0));
        assert_eq!(s.mix_mode(), Some(MixMode::Pg));
        assert_eq!(s.policy().admission.lobby_order, LobbyOrder::Signal);
        assert_eq!(s.sessions().lobby().ordered().len(), lobby);
        assert_eq!(s.sessions().iter().count(), sessions);
    }

    #[test]
    fn without_mix_select_the_union_is_the_policy() {
        let s = with_admission(|a| a.target_loop = Some(Default::default()));
        assert_eq!(s.mix_mode(), None);
        assert_eq!(s.union_admission(), &s.policy().admission);
    }

    fn admitted(s: &mut Scheduler, request: &str, session: &str, class: &str, at: f64) {
        s.on_classify(Some(request), session, None, Some(class), 100, t(at));
        s.on_dispatched(request, session, t(at));
    }

    #[test]
    fn rotation_pauses_the_longest_runner_and_its_next_turn_resumes_through_the_lobby() {
        let mut s = with_admission(|a| a.rotate = Some(crate::config::RotateConfig::default()));
        // Two standard sessions admitted at 0 and 100 s; each finishes its turn.
        admitted(&mut s, "a1", "a", "standard", 0.0);
        admitted(&mut s, "b1", "b", "standard", 100.0);
        s.on_completed("a1", Some(100), t(10.0));
        s.on_completed("b1", Some(100), t(110.0));
        // A new standard session waits from 400 s.
        s.on_classify(Some("c1"), "c", None, Some("standard"), 100, t(400.0));
        assert_eq!(s.sessions().get("c").unwrap().state, SessionState::Lobby);
        // Not yet: c has waited under 300 s.
        assert!(s.rotation_candidates(t(650.0)).is_empty());
        // At 750 s c has waited 350 s, so a candidate must have run 350 s: a (750 s) and b
        // (650 s) both have; the longer run first.
        let c = s.rotation_candidates(t(750.0));
        assert_eq!(
            c,
            vec![RotationCandidates {
                for_class: "standard".to_owned(),
                waited_s: 350.0,
                sessions: vec![
                    ("a".to_owned(), "standard".to_owned(), 750.0),
                    ("b".to_owned(), "standard".to_owned(), 650.0),
                ],
            }]
        );
        assert!(s.pause("a", t(750.0)));
        assert_eq!(s.rotations(), 1);
        assert!(!s.sessions().get("a").unwrap().is_live());
        // Its next turn waits in the lobby; past the wait limit it is not expired, and it waits
        // its turn (behind c, waiting since 400 s) like any other session (F32: it is not
        // forced in over the budget).
        s.on_classify(Some("a2"), "a", None, Some("standard"), 100, t(800.0));
        let a = s.sessions().get("a").unwrap().clone();
        assert!(a.resumed && a.state == SessionState::Lobby);
        let none = s.kv_budget(0, 0);
        assert!(matches!(
            decide_indexed(s.policy(), t(900.0), &a, s.sessions(), &none),
            Decision::Hold { reason: HoldReason::BehindEarlier { ref ahead }, until }
                if ahead == "c" && until == t(960.0)
        ));
        // c (waiting since 400 s) is ahead of a; once c is admitted, a fits.
        s.on_dispatched("c1", "c", t(890.0));
        let room = s.kv_budget(1_000_000, 0);
        assert_eq!(
            decide_indexed(s.policy(), t(900.0), &a, s.sessions(), &room),
            Decision::Admit(AdmitReason::Fits)
        );
        // Resuming counts the time out as paused and keeps the admission time.
        s.on_dispatched("a2", "a", t(900.0));
        let a = s.sessions().get("a").unwrap();
        assert_eq!(a.admitted, Some(t(0.0)));
        assert!((a.paused_s - 150.0).abs() < 1e-9 && !a.resumed);
        // Priority (the top rank) is never rotated, and off means no candidates.
        let mut off = with_admission(|_| {});
        admitted(&mut off, "x1", "x", "standard", 0.0);
        off.on_completed("x1", Some(100), t(10.0));
        off.on_classify(Some("y1"), "y", None, Some("standard"), 100, t(1.0));
        assert!(off.rotation_candidates(t(5000.0)).is_empty());
    }

    #[test]
    fn rotation_counts_the_wait_across_expiries_and_derives_its_thresholds() {
        let mut s = with_admission(|a| a.rotate = Some(crate::config::RotateConfig::default()));
        // a admitted at 0 s and between turns; batch session z admitted at 200 s.
        admitted(&mut s, "a1", "a", "standard", 0.0);
        s.on_completed("a1", Some(100), t(10.0));
        admitted(&mut s, "z1", "z", "batch", 200.0);
        s.on_completed("z1", Some(100), t(210.0));
        // c first waits at 100 s; its wait limit (60 s) expires it and its client retries.
        s.on_classify(Some("c1"), "c", None, Some("standard"), 100, t(100.0));
        s.on_aborted("c1");
        assert_eq!(s.sessions().get("c").unwrap().state, SessionState::Ended);
        s.on_classify(Some("c2"), "c", None, Some("standard"), 100, t(170.0));
        let c = s.sessions().get("c").unwrap();
        assert_eq!(c.state, SessionState::Lobby);
        assert_eq!(c.queued_since, Some(t(100.0)));
        // The trigger is 5 x standard's 60 s wait limit = 300 s, from the first attempt: at 450 s
        // c has waited 350 s although its latest attempt is 280 s old.
        assert!(s.rotation_candidates(t(390.0)).is_empty());
        let r = s.rotation_candidates(t(450.0));
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].for_class, "standard");
        assert!((r[0].waited_s - 350.0).abs() < 1e-9);
        // The lower class goes first (z, batch, ran 250 s, is short of 350 s: not a candidate
        // yet); a (standard, ran 450 s) is.
        assert_eq!(
            r[0].sessions,
            vec![("a".to_owned(), "standard".to_owned(), 450.0)]
        );
        // At 600 s c has waited 500 s, so the least run is 500 s: z (400 s) is still short.
        let r = s.rotation_candidates(t(600.0));
        assert_eq!(r[0].sessions[0].0, "a");
        assert_eq!(r[0].sessions.len(), 1);
        // At 1,000 s c has waited 900 s; z ran 800 s (short), a 1,000 s.
        assert_eq!(s.rotation_candidates(t(1000.0))[0].sessions.len(), 1);
        // With a fixed least run of 100 s, z (batch, the lower class) is first.
        let mut s2 = with_admission(|a| {
            a.rotate = Some(crate::config::RotateConfig {
                min_run_s: Some(100.0),
                ..Default::default()
            })
        });
        admitted(&mut s2, "a1", "a", "standard", 0.0);
        s2.on_completed("a1", Some(100), t(10.0));
        admitted(&mut s2, "z1", "z", "batch", 200.0);
        s2.on_completed("z1", Some(100), t(210.0));
        s2.on_classify(Some("c1"), "c", None, Some("standard"), 100, t(100.0));
        let r = s2.rotation_candidates(t(450.0));
        let order: Vec<&str> = r[0].sessions.iter().map(|x| x.0.as_str()).collect();
        assert_eq!(order, vec!["z", "a"]);
        // A batch waiter never pauses a standard session (a higher class).
        let mut s3 = with_admission(|a| {
            a.rotate = Some(crate::config::RotateConfig {
                after_s: Some(10.0),
                min_run_s: Some(1.0),
                ..Default::default()
            })
        });
        admitted(&mut s3, "a1", "a", "standard", 0.0);
        s3.on_completed("a1", Some(100), t(10.0));
        s3.on_classify(Some("b1"), "b", None, Some("batch"), 100, t(20.0));
        assert!(s3.rotation_candidates(t(100.0)).is_empty());
    }

    #[test]
    fn aging_keeps_a_retried_session_ahead_and_fifo_sends_it_to_the_back() {
        use crate::admit::decide;
        for (order, first) in [(LobbyOrder::Aging, "c"), (LobbyOrder::Fifo, "d")] {
            let mut s = with_admission(|a| a.lobby_order = order);
            // c waits from 0 s, expires and retries at 70 s; d arrived at 30 s.
            s.on_classify(Some("c1"), "c", None, Some("standard"), 100, t(0.0));
            s.on_classify(Some("d1"), "d", None, Some("standard"), 100, t(30.0));
            s.on_aborted("c1");
            s.on_classify(Some("c2"), "c", None, Some("standard"), 100, t(70.0));
            let kv = s.kv_budget(1_000_000, 0);
            let head = |id: &str| {
                let session = s.sessions().get(id).unwrap();
                decide_indexed(s.policy(), t(71.0), session, s.sessions(), &kv)
            };
            let other = if first == "c" { "d" } else { "c" };
            assert_eq!(head(first), Decision::Admit(AdmitReason::Fits), "{order:?}");
            assert!(
                matches!(
                    head(other),
                    Decision::Hold {
                        reason: HoldReason::BehindEarlier { .. },
                        ..
                    }
                ),
                "{order:?}"
            );
            // The full scan agrees with the index.
            let scan = s.lobby_scan();
            for id in ["c", "d"] {
                let session = s.sessions().get(id).unwrap();
                assert_eq!(
                    decide(s.policy(), t(71.0), id, &scan, &kv),
                    decide_indexed(s.policy(), t(71.0), session, s.sessions(), &kv)
                );
            }
            // The wait limit still counts from the latest attempt: c is not expired at 71 s.
            assert!(s.sessions().get("c").unwrap().lobby_wait_s(t(71.0)) > 70.0);
        }
    }

    #[test]
    fn aging_holds_an_expired_sessions_place_while_its_client_is_away() {
        let mut s = with_admission(|a| a.lobby_order = LobbyOrder::Aging);
        // c waits from 0 s and its wait limit (60 s) expires it; d arrives at 65 s while c's
        // client is away.
        s.on_classify(Some("c1"), "c", None, Some("standard"), 100, t(0.0));
        s.on_aborted("c1");
        assert!(s.sessions().get("c").unwrap().away());
        s.on_classify(Some("d1"), "d", None, Some("standard"), 100, t(65.0));
        let kv = s.kv_budget(1_000_000, 0);
        let d = s.sessions().get("d").unwrap().clone();
        assert_eq!(
            decide_indexed(s.policy(), t(66.0), &d, s.sessions(), &kv),
            Decision::Hold {
                reason: HoldReason::BehindEarlier { ahead: "c".into() },
                until: t(125.0)
            }
        );
        // Its retry is overdue after another wait limit (past 120 s): d goes.
        assert_eq!(
            decide_indexed(s.policy(), t(121.0), &d, s.sessions(), &kv),
            Decision::Admit(AdmitReason::Fits)
        );
        // Given a Retry-After of 500 s, c's place is held 500 s longer: e, new at 590 s, waits
        // behind it until 620 s (d expired too, and its retry is long overdue).
        s.note_retry_after("c", Some(500.0));
        s.on_aborted("d1");
        s.on_classify(Some("e1"), "e", None, Some("standard"), 100, t(590.0));
        let e = s.sessions().get("e").unwrap().clone();
        assert!(matches!(
            decide_indexed(s.policy(), t(600.0), &e, s.sessions(), &kv),
            Decision::Hold { reason: HoldReason::BehindEarlier { ref ahead }, .. } if ahead == "c"
        ));
        assert_eq!(
            decide_indexed(s.policy(), t(621.0), &e, s.sessions(), &kv),
            Decision::Admit(AdmitReason::Fits)
        );
        // First come does not hold the place.
        let mut f = with_admission(|_| {});
        f.on_classify(Some("c1"), "c", None, Some("standard"), 100, t(0.0));
        f.on_aborted("c1");
        f.on_classify(Some("d1"), "d", None, Some("standard"), 100, t(65.0));
        let d = f.sessions().get("d").unwrap().clone();
        assert_eq!(
            decide_indexed(f.policy(), t(66.0), &d, f.sessions(), &kv),
            Decision::Admit(AdmitReason::Fits)
        );
    }

    #[test]
    fn aging_counts_a_resumed_sessions_earlier_wait() {
        let mut s = with_admission(|a| {
            a.lobby_order = LobbyOrder::Aging;
            a.rotate = Some(crate::config::RotateConfig::default());
        });
        // a waited 50 s before admission, ran, and was paused at 1,000 s.
        s.on_classify(Some("a1"), "a", None, Some("standard"), 100, t(0.0));
        s.on_dispatched("a1", "a", t(50.0));
        s.on_completed("a1", Some(100), t(60.0));
        assert!((s.sessions().get("a").unwrap().waited_s - 50.0).abs() < 1e-9);
        assert!(s.pause("a", t(1000.0)));
        // b has waited since 990 s; a resumes at 1,020 s: aged from 970 s, ahead of b.
        s.on_classify(Some("b1"), "b", None, Some("standard"), 100, t(990.0));
        s.on_classify(Some("a2"), "a", None, Some("standard"), 100, t(1020.0));
        let a = s.sessions().get("a").unwrap();
        assert_eq!(a.aged_since(), t(970.0));
        let b = s.sessions().get("b").unwrap();
        assert_eq!(b.aged_since(), t(990.0));
        assert!(s.sessions().order_at(a) < s.sessions().order_at(b));
    }

    #[test]
    fn fair_order_is_first_come_among_new_sessions() {
        let mut fair = with_admission(|a| a.lobby_order = LobbyOrder::Fair);
        let mut fifo = with_admission(|_| {});
        for s in [&mut fair, &mut fifo] {
            s.on_classify(Some("x1"), "x", None, Some("standard"), 100, t(0.0));
            s.on_classify(Some("y1"), "y", None, Some("standard"), 100, t(5.0));
        }
        let kv = fair.kv_budget(1_000_000, 0);
        for id in ["x", "y"] {
            assert_eq!(
                decide_indexed(
                    fair.policy(),
                    t(6.0),
                    fair.sessions().get(id).unwrap(),
                    fair.sessions(),
                    &kv
                ),
                decide_indexed(
                    fifo.policy(),
                    t(6.0),
                    fifo.sessions().get(id).unwrap(),
                    fifo.sessions(),
                    &kv
                )
            );
        }
    }

    #[test]
    fn retry_after_estimate_is_place_over_admission_rate_less_the_wait_limit() {
        let mut s = with_admission(|a| {
            a.retry_after_estimate = Some(crate::config::RetryAfterEstimate::default())
        });
        // Standard's window is 10 x 60 s = 600 s. Twelve admissions in it: 0.02 a second.
        for i in 0..12 {
            let at = 100.0 + 40.0 * f64::from(i);
            admitted(&mut s, &format!("r{i}"), &format!("s{i}"), "standard", at);
        }
        assert!((s.admit_rate("standard", t(650.0)).unwrap() - 12.0 / 600.0).abs() < 1e-9);
        // Five sessions wait; the fifth is fifth in line: 5 / 0.02 = 250 s, less 60 s = 190 s.
        for i in 0..5 {
            s.on_classify(
                Some(&format!("w{i}")),
                &format!("w{i}"),
                None,
                Some("standard"),
                100,
                t(640.0 + f64::from(i)),
            );
        }
        let after = s.retry_after_estimate("w4", t(650.0)).unwrap();
        assert!((after - 190.0).abs() < 1e-6, "{after}");
        // The first in line: 1 / 0.02 = 50 s < 60 s: no Retry-After (its own backoff).
        assert_eq!(s.retry_after_estimate("w0", t(650.0)), None);
        // Before a whole window has passed the rate is unknown.
        assert_eq!(s.admit_rate("standard", t(500.0)), None);
        // Off: none.
        let off = with_admission(|_| {});
        assert_eq!(off.retry_after_estimate("w0", t(650.0)), None);
        // The lobby stats see the five, the oldest from 640 s.
        let stats = s.lobby_stats(t(650.0));
        let st = stats.iter().find(|l| l.class == "standard").unwrap();
        assert_eq!(st.waiting, 5);
        assert!((st.oldest_wait_s - 10.0).abs() < 1e-9);
    }

    #[test]
    fn retry_after_estimate_counts_session_ends_and_is_capped() {
        let estimate = |count_ends, max_s| {
            with_admission(move |a| {
                a.retry_after_estimate = Some(crate::config::RetryAfterEstimate {
                    count_ends,
                    max_s,
                    ..Default::default()
                })
            })
        };
        let run = |s: &mut Scheduler| {
            // Six standard sessions admitted at 100 to 150 s, before the 600 s window that ends at
            // 1,300 s; each finishes its turn at 800 to 1,050 s and then ends.
            for i in 0..6 {
                let at = 100.0 + 10.0 * f64::from(i);
                admitted(s, &format!("r{i}"), &format!("s{i}"), "standard", at);
                s.on_completed(&format!("r{i}"), Some(100), t(800.0 + 50.0 * f64::from(i)));
                s.on_session_end(&format!("s{i}"));
            }
            // Five sessions wait from 1,290 s.
            for i in 0..5 {
                s.on_classify(
                    Some(&format!("w{i}")),
                    &format!("w{i}"),
                    None,
                    Some("standard"),
                    100,
                    t(1290.0 + f64::from(i)),
                );
            }
        };
        // Admissions out of the lobby only: none in the window, so no rate and no Retry-After.
        let mut lobby_only = estimate(false, None);
        run(&mut lobby_only);
        assert_eq!(lobby_only.retry_after_estimate("w4", t(1300.0)), None);
        // With session ends: 6 in 600 s = 0.01 a second; the fifth in line: 500 s less 60 s.
        let mut ends = estimate(true, None);
        run(&mut ends);
        assert!((ends.end_rate("standard", t(1300.0)).unwrap() - 0.01).abs() < 1e-9);
        let after = ends.retry_after_estimate("w4", t(1300.0)).unwrap();
        assert!((after - 440.0).abs() < 1e-6, "{after}");
        // Capped at 300 s.
        let mut capped = estimate(true, Some(300.0));
        run(&mut capped);
        assert_eq!(capped.retry_after_estimate("w4", t(1300.0)), Some(300.0));
        // A fallback when the rate is unknown, and a floor when the estimate is not positive.
        let mut fallback = with_admission(|a| {
            a.retry_after_estimate = Some(crate::config::RetryAfterEstimate {
                fallback_s: Some(45.0),
                min_s: Some(15.0),
                ..Default::default()
            })
        });
        run(&mut fallback);
        assert_eq!(fallback.retry_after_estimate("w4", t(1300.0)), Some(45.0));
        let mut floor = with_admission(|a| {
            a.retry_after_estimate = Some(crate::config::RetryAfterEstimate {
                count_ends: true,
                min_s: Some(15.0),
                ..Default::default()
            })
        });
        run(&mut floor);
        // The first in line: 1 / 0.01 = 100 s, less 60 s = 40 s; above the floor.
        assert_eq!(floor.retry_after_estimate("w0", t(1300.0)), Some(40.0));
        // The default leaves the estimate as it was (no cap, lobby admissions only, no fallback).
        let d = crate::config::RetryAfterEstimate::default();
        assert!(d.max_s.is_none() && !d.count_ends && d.fallback_s.is_none() && d.min_s.is_none());
    }

    #[test]
    fn the_adaptive_gate_group_switches_the_session_gate() {
        let mut policy = ServicePolicy {
            mode: Mode::Enforce,
            ..Default::default()
        };
        policy.adaptive = Some(crate::adaptive::AdaptiveConfig {
            win_s: 10.0,
            sample_s: 10.0,
            ..Default::default()
        });
        let mut s = Scheduler::new(policy);
        s.adaptive_hint_seen();
        // No room: a new session waits while the gate group is on (it starts on).
        s.on_classify(Some("r1"), "a", None, Some("standard"), 100, t(0.0));
        let full = s.kv_budget(100, 0);
        assert!(matches!(
            s.gate_check("a", t(1.0), &full),
            GateAnswer::Hold { .. }
        ));
        // The first sample waits sample_s from the first ask (no load is booked at the start).
        assert_eq!(s.adaptive_step(t(1.0), 0.05, 1000, 0), Some(true));
        assert!(s.adaptive().unwrap().gate);
        // Low load: the gate group switches off and the session goes.
        assert_eq!(s.adaptive_step(t(11.0), 0.05, 1000, 0), Some(false));
        assert!(!s.adaptive().unwrap().gate);
        assert_eq!(s.gate_check("a", t(12.0), &full), GateAnswer::Go);
        // Load back up with memory room: on again, and it holds.
        assert_eq!(s.adaptive_step(t(31.0), 0.6, 1000, 0), Some(true));
        assert!(matches!(
            s.gate_check("a", t(32.0), &full),
            GateAnswer::Hold { .. }
        ));
        // Without the rule nothing changes.
        let mut off = enforce();
        assert_eq!(off.adaptive_step(t(1.0), 0.0, 1000, 0), None);
    }

    #[test]
    fn with_queue_load_a_flood_waiting_at_the_router_keeps_the_gate_on() {
        let rule = |queue_load| {
            let mut policy = ServicePolicy {
                mode: Mode::Enforce,
                ..Default::default()
            };
            policy.adaptive = Some(crate::adaptive::AdaptiveConfig {
                win_s: 10.0,
                sample_s: 10.0,
                queue_load,
                ..Default::default()
            });
            let mut s = Scheduler::new(policy);
            s.adaptive_hint_seen();
            s
        };
        // Workers nearly idle (load 0.05 of 1,000 KV tokens) while 2,000 prompt tokens wait at
        // the router.
        for (queue_load, gate_after) in [(false, false), (true, true)] {
            let mut s = rule(queue_load);
            s.on_classify(Some("r1"), "a", None, Some("standard"), 1200, t(0.0));
            s.on_classify(Some("r2"), "b", None, Some("priority"), 800, t(0.0));
            assert_eq!(s.queued_tokens(), 2000);
            let load = s.adaptive_load(0.05, 1000);
            s.adaptive_step(t(1.0), load, 1000, 0);
            s.adaptive_step(t(11.0), load, 1000, 0);
            assert_eq!(
                s.adaptive().unwrap().gate,
                gate_after,
                "queue_load {queue_load}"
            );
        }
        // A request leaves the count once the host takes it or the router sends it.
        let mut s = rule(true);
        s.on_classify(Some("r1"), "a", None, Some("standard"), 1200, t(0.0));
        s.on_classify(Some("r2"), "b", None, Some("priority"), 800, t(0.0));
        s.on_dispatched("r1", "a", t(1.0));
        assert_eq!(s.queued_tokens(), 800);
        s.on_sent("r2");
        assert_eq!(s.queued_tokens(), 0);
        assert_eq!(s.adaptive_load(0.05, 1000), 0.05);
        // Without the rule the load is the workers' alone.
        let mut off = enforce();
        off.on_classify(Some("r1"), "a", None, Some("standard"), 1200, t(0.0));
        assert_eq!(off.adaptive_load(0.05, 1000), 0.05);
    }

    #[test]
    fn forecast_harm_holds_a_lower_turn_that_would_push_a_higher_class_without_slack() {
        let mut s = with_admission(|a| {
            a.shed_by_slack = true;
            a.shed_forecast_harm = true;
        });
        admitted(&mut s, "r1", "std", "standard", 0.0);
        let kv = s.kv_budget(1000, 0);
        let head = |harm: Vec<(String, f64)>| HeadFacts {
            queued_at: Some(t(5.0)),
            kv_room: Some(true),
            seq_room: None,
            harm,
            surge: None,
            prefill: None,
            new_harm: None,
        };
        let pushes_priority = vec![("priority".to_owned(), 2.0)];
        // Room on a worker, but the best placement pushes priority (no slack yet) over its ITL.
        assert!(matches!(
            s.gate_check_head("std", t(5.0), &kv, head(pushes_priority.clone())),
            GateAnswer::Hold { until: Some(u) } if u == t(15.0)
        ));
        // No harm, or harm only to its own or a lower class: it goes.
        assert_eq!(
            s.gate_check_head("std", t(5.0), &kv, head(Vec::new())),
            GateAnswer::Go
        );
        assert_eq!(
            s.gate_check_head("std", t(5.0), &kv, head(vec![("batch".to_owned(), 3.0)])),
            GateAnswer::Go
        );
        // Priority well over target: the harm is let through.
        for i in 0..10 {
            s.metrics.record(
                "priority",
                Completion {
                    at: t(1.0 + f64::from(i) * 0.1),
                    ttft_s: 0.1,
                    itl_s: Some(0.01),
                },
            );
        }
        assert_eq!(
            s.gate_check_head("std", t(5.0), &kv, head(pushes_priority.clone())),
            GateAnswer::Go
        );
        // The switch off: harm is ignored.
        let mut off = with_admission(|a| a.shed_by_slack = true);
        admitted(&mut off, "r1", "std", "standard", 0.0);
        assert_eq!(
            off.gate_check_head("std", t(5.0), &kv, head(pushes_priority)),
            GateAnswer::Go
        );
    }

    #[test]
    fn shed_by_slack_holds_a_lower_turn_only_without_slack_and_without_room() {
        let mut s = with_admission(|a| a.shed_by_slack = true);
        admitted(&mut s, "r1", "std", "standard", 0.0);
        admitted(&mut s, "p1", "pri", "priority", 0.0);
        let kv = s.kv_budget(1000, 0);
        let head = |room| HeadFacts {
            queued_at: Some(t(5.0)),
            kv_room: room,
            seq_room: None,
            harm: Vec::new(),
            surge: None,
            prefill: None,
            new_harm: None,
        };
        // Priority has no window yet (no slack) and no worker has room: standard's turn waits
        // until its 10 s hold runs out.
        assert!(matches!(
            s.gate_check_head("std", t(5.0), &kv, head(Some(false))),
            GateAnswer::Hold { until: Some(u) } if u == t(15.0)
        ));
        // Room on some worker, or room unknown: it goes.
        assert_eq!(
            s.gate_check_head("std", t(5.0), &kv, head(Some(true))),
            GateAnswer::Go
        );
        assert_eq!(
            s.gate_check_head("std", t(5.0), &kv, head(None)),
            GateAnswer::Go
        );
        // Past its hold limit: it goes.
        assert_eq!(
            s.gate_check_head("std", t(15.0), &kv, head(Some(false))),
            GateAnswer::Go
        );
        // Priority is never shed.
        assert_eq!(
            s.gate_check_head("pri", t(5.0), &kv, head(Some(false))),
            GateAnswer::Go
        );
        // Priority 5 points over target in its window: standard has slack to spend.
        for i in 0..10 {
            s.metrics.record(
                "priority",
                Completion {
                    at: t(1.0 + f64::from(i) * 0.1),
                    ttft_s: 0.1,
                    itl_s: Some(0.01),
                },
            );
        }
        assert_eq!(
            s.gate_check_head("std", t(5.0), &kv, head(Some(false))),
            GateAnswer::Go
        );
    }

    /// Record `n` priority completions in the second before `at`, `met` of them on target.
    fn priority_window(s: &mut Scheduler, class: &str, at: f64, n: u32, met: u32) {
        for i in 0..n {
            let ok = i < met;
            s.metrics.record(
                class,
                Completion {
                    at: t(at - 1.0 + f64::from(i) / f64::from(n)),
                    ttft_s: if ok { 0.1 } else { 100.0 },
                    itl_s: Some(0.01),
                },
            );
        }
    }

    #[test]
    fn target_aware_shedding_stops_while_priority_is_well_over_target() {
        let mut s = with_admission(|a| {
            a.shed_by_slack = true;
            a.shed_target = Some(crate::config::ShedTargetConfig {
                min_hold_s: 0.0,
                smooth_s: 1.0,
                step_s: 1.0,
                ..Default::default()
            });
        });
        admitted(&mut s, "r1", "std", "standard", 0.0);
        let kv = s.kv_budget(1000, 0);
        let head = |q: f64| HeadFacts {
            queued_at: Some(t(q)),
            kv_room: Some(false),
            seq_room: None,
            harm: Vec::new(),
            surge: None,
            prefill: None,
            new_harm: None,
        };
        // No reading yet: the switch starts on and the turn waits.
        assert!(matches!(
            s.gate_check_head("std", t(1.0), &kv, head(1.0)),
            GateAnswer::Hold { .. }
        ));
        // Priority at 0.93 (between target + 0.02 and target + 0.05): still on.
        priority_window(&mut s, "priority", 10.0, 100, 93);
        assert!(matches!(
            s.gate_check_head("std", t(10.0), &kv, head(10.0)),
            GateAnswer::Hold { .. }
        ));
        // Priority at 1.0 for a while: over target + 0.05, the switch turns off and the turn goes,
        // although the old rule (window under target + 0.05) would no longer hold it either.
        for k in 0..20 {
            let at = 70.0 + f64::from(k) * 2.0;
            priority_window(&mut s, "priority", at, 50, 50);
            s.gate_check_head("std", t(at), &kv, head(at));
        }
        assert!(!s.shed_switch_on("priority"));
        assert_eq!(
            s.gate_check_head("std", t(110.0), &kv, head(110.0)),
            GateAnswer::Go
        );
        // Priority falls to 0.80: under target + 0.02, the switch turns back on.
        for k in 0..40 {
            let at = 180.0 + f64::from(k) * 2.0;
            priority_window(&mut s, "priority", at, 50, 40);
            s.gate_check_head("std", t(at), &kv, head(at));
        }
        assert!(s.shed_switch_on("priority"));
        assert!(s.target_stats().0 >= 2);
    }

    #[test]
    fn target_aware_switch_waits_its_minimum_hold() {
        let mut s = with_admission(|a| {
            a.shed_by_slack = true;
            a.shed_target = Some(crate::config::ShedTargetConfig {
                min_hold_s: 100.0,
                smooth_s: 1.0,
                step_s: 1.0,
                ..Default::default()
            });
        });
        let kv = s.kv_budget(1000, 0);
        let head = HeadFacts::default();
        for k in 0..10 {
            let at = 10.0 + f64::from(k) * 2.0;
            priority_window(&mut s, "priority", at, 50, 50);
            s.gate_check_head("none", t(at), &kv, head.clone());
        }
        assert!(!s.shed_switch_on("priority"));
        // Priority drops at once, but the switch changed under 100 s ago: it stays off.
        for k in 0..10 {
            let at = 40.0 + f64::from(k) * 2.0;
            priority_window(&mut s, "priority", at, 50, 10);
            s.gate_check_head("none", t(at), &kv, head.clone());
        }
        assert!(!s.shed_switch_on("priority"));
        for k in 0..10 {
            let at = 130.0 + f64::from(k) * 2.0;
            priority_window(&mut s, "priority", at, 50, 10);
            s.gate_check_head("none", t(at), &kv, head.clone());
        }
        assert!(s.shed_switch_on("priority"));
    }

    #[test]
    fn release_grows_a_class_share_only_with_surplus_above_and_demand() {
        let mut s = with_admission(|a| {
            a.release = Some(crate::config::ReleaseConfig {
                smooth_s: 1.0,
                step_s: 1.0,
                ..Default::default()
            });
        });
        let kv = s.kv_budget(1000, 0);
        let head = HeadFacts::default();
        let step = |s: &mut Scheduler, at: f64, p: u32, st: u32| {
            priority_window(s, "priority", at, 50, p);
            priority_window(s, "standard", at, 50, st);
            s.gate_check_head("none", t(at), &kv, head.clone());
        };
        // Both classes far over target, but nobody waiting: no growth.
        for k in 0..10 {
            step(&mut s, 10.0 + f64::from(k) * 2.0, 50, 50);
        }
        assert_eq!(s.release_scale("batch"), 1.0);
        assert_eq!(s.release_scale("standard"), 1.0);
        // A batch session turned away: batch grows (both above have surplus); standard has no
        // demand and stays.
        s.note_lost("batch");
        step(&mut s, 40.0, 50, 50);
        let grown = s.release_scale("batch");
        assert!(grown > 1.0, "{grown}");
        assert_eq!(s.release_scale("standard"), 1.0);
        assert!((s.pace_scale("batch") - grown).abs() < 1e-12);
        // Priority falls under its margin: batch shrinks back toward 1, never under.
        for k in 0..200 {
            step(&mut s, 100.0 + f64::from(k) * 2.0, 40, 50);
        }
        assert_eq!(s.release_scale("batch"), 1.0);
    }

    #[test]
    fn release_grows_only_the_listed_classes() {
        let mut s = with_admission(|a| {
            a.release = Some(crate::config::ReleaseConfig {
                classes: vec!["batch".to_owned()],
                smooth_s: 1.0,
                step_s: 1.0,
                ..Default::default()
            });
        });
        let kv = s.kv_budget(1000, 0);
        for k in 0..10 {
            let at = 10.0 + f64::from(k) * 2.0;
            priority_window(&mut s, "priority", at, 50, 50);
            priority_window(&mut s, "standard", at, 50, 50);
            s.note_lost("batch");
            s.note_lost("standard");
            s.gate_check_head("none", t(at), &kv, HeadFacts::default());
        }
        assert!(s.release_scale("batch") > 1.0);
        assert_eq!(s.release_scale("standard"), 1.0);
    }

    #[test]
    fn release_yields_to_waiting_sessions_of_a_class_above() {
        let mut s = with_admission(|a| {
            a.release = Some(crate::config::ReleaseConfig {
                classes: vec!["batch".to_owned()],
                yield_to_demand: true,
                smooth_s: 1.0,
                step_s: 1.0,
                ..Default::default()
            });
        });
        let kv = s.kv_budget(1000, 0);
        let step = |s: &mut Scheduler, at: f64, lost_standard: bool| {
            priority_window(s, "priority", at, 50, 50);
            priority_window(s, "standard", at, 50, 50);
            s.note_lost("batch");
            if lost_standard {
                s.note_lost("standard");
            }
            s.gate_check_head("none", t(at), &kv, HeadFacts::default());
        };
        for k in 0..10 {
            step(&mut s, 10.0 + f64::from(k) * 2.0, false);
        }
        let grown = s.release_scale("batch");
        assert!(grown > 1.0);
        // Standard sessions turned away: batch stops growing and gives back.
        for k in 0..10 {
            step(&mut s, 40.0 + f64::from(k) * 2.0, true);
        }
        assert!(s.release_scale("batch") < grown);
    }

    #[test]
    fn release_yields_only_while_the_waiting_class_is_near_its_target() {
        let run = |std_met: u32| {
            let mut s = with_admission(|a| {
                a.release = Some(crate::config::ReleaseConfig {
                    classes: vec!["batch".to_owned()],
                    yield_to_demand: true,
                    yield_margin: Some(0.10),
                    smooth_s: 1.0,
                    step_s: 1.0,
                    ..Default::default()
                });
            });
            let kv = s.kv_budget(1000, 0);
            for k in 0..10 {
                let at = 10.0 + f64::from(k) * 2.0;
                priority_window(&mut s, "priority", at, 50, 50);
                priority_window(&mut s, "standard", at, 50, std_met);
                s.note_lost("batch");
                s.note_lost("standard");
                s.gate_check_head("none", t(at), &kv, HeadFacts::default());
            }
            s.release_scale("batch")
        };
        // Standard waits but runs at 0.96, over 0.70 + 0.10: batch is released anyway.
        assert!(run(48) > 1.0);
        // Standard waits and runs at 0.78, under 0.80: batch yields.
        assert_eq!(run(39), 1.0);
    }

    #[test]
    fn release_is_capped() {
        let mut s = with_admission(|a| {
            a.release = Some(crate::config::ReleaseConfig {
                smooth_s: 1.0,
                step_s: 1.0,
                max_scale: 1.2,
                grow_per_min: 10.0,
                ..Default::default()
            });
        });
        let kv = s.kv_budget(1000, 0);
        for k in 0..50 {
            let at = 10.0 + f64::from(k) * 2.0;
            priority_window(&mut s, "priority", at, 50, 50);
            priority_window(&mut s, "standard", at, 50, 50);
            s.note_lost("batch");
            s.gate_check_head("none", t(at), &kv, HeadFacts::default());
        }
        assert!((s.release_scale("batch") - 1.2).abs() < 1e-12);
    }

    #[test]
    fn shedding_off_never_holds_an_admitted_turn() {
        let mut s = with_admission(|_| {});
        admitted(&mut s, "r1", "std", "standard", 0.0);
        let kv = s.kv_budget(1000, 0);
        let head = HeadFacts {
            queued_at: Some(t(5.0)),
            kv_room: Some(false),
            seq_room: None,
            harm: Vec::new(),
            surge: None,
            prefill: None,
            new_harm: None,
        };
        assert_eq!(s.gate_check_head("std", t(5.0), &kv, head), GateAnswer::Go);
    }

    #[test]
    fn without_the_session_gate_a_new_session_goes_at_once() {
        let mut s = with_admission(|a| a.session_gate = false);
        s.on_classify(Some("r1"), "a", None, None, 600, t(0.0));
        s.on_dispatched("r1", "a", t(0.0));
        s.on_classify(Some("r2"), "b", None, None, 600, t(1.0));
        let kv = s.kv_budget(1000, 0);
        assert_eq!(s.gate_check("b", t(1.0), &kv), GateAnswer::Go);
    }

    fn record_window(s: &mut Scheduler, class: &str, from: f64, met: usize, missed: usize) {
        for i in 0..met + missed {
            s.metrics.record(
                class,
                Completion {
                    at: t(from + i as f64 * 0.01),
                    ttft_s: if i < met { 0.1 } else { 100.0 },
                    itl_s: Some(0.01),
                },
            );
        }
    }

    #[test]
    fn pacing_grows_standard_on_demand_and_gives_batch_way_first() {
        let mut s = with_admission(|a| a.pace = Some(Default::default()));
        let kv = s.kv_budget(1_000_000, 0);
        // Both classes far over target, and a standard session waiting: standard's share grows.
        s.on_classify(Some("w"), "waiting", None, Some("standard"), 10, t(0.0));
        let mut now = 0.0;
        for _ in 0..50 {
            record_window(&mut s, "priority", now, 20, 0);
            record_window(&mut s, "standard", now, 20, 0);
            now += 12.0;
            s.gate_check("waiting", t(now), &kv);
        }
        let (standard, batch) = s.pace_multiples();
        assert!(standard > 1.05 && standard <= 2.5, "standard {standard}");
        assert_eq!(batch, 1.0);
        assert!((s.pace_scale("standard") - standard).abs() < 1e-12);
        assert_eq!(s.pace_scale("priority"), 1.0);
        // Priority falls far short: batch gives way down to a third, then standard drops.
        for _ in 0..400 {
            record_window(&mut s, "priority", now, 0, 20);
            now += 12.0;
            s.gate_check("waiting", t(now), &kv);
        }
        let (standard_after, batch_after) = s.pace_multiples();
        assert!(
            (batch_after - 1.0 / 3.0).abs() < 1e-9,
            "batch {batch_after}"
        );
        assert!(standard_after < standard, "standard {standard_after}");
        assert!(standard_after >= 0.8 - 1e-9);
    }

    #[test]
    fn pacing_off_keeps_the_set_shares() {
        let s = with_admission(|_| {});
        assert_eq!(s.pace_scale("standard"), 1.0);
        assert_eq!(s.pace_scale("batch"), 1.0);
    }

    #[test]
    fn a_new_session_waits_while_every_worker_is_at_its_running_request_limit() {
        let mut s = with_admission(|a| a.hold_at_max_num_seqs = true);
        s.on_classify(Some("r1"), "a", None, None, 100, t(0.0));
        let kv = s.kv_budget(1000, 0);
        let head = |seq_room| HeadFacts {
            queued_at: Some(t(0.0)),
            kv_room: None,
            seq_room,
            harm: Vec::new(),
            surge: None,
            prefill: None,
            new_harm: None,
        };
        assert!(matches!(
            s.gate_check_head("a", t(1.0), &kv, head(Some(false))),
            GateAnswer::Hold { until: Some(u) } if u == t(60.0)
        ));
        assert_eq!(
            s.gate_check_head("a", t(1.0), &kv, head(Some(true))),
            GateAnswer::Go
        );
        assert_eq!(
            s.gate_check_head("a", t(1.0), &kv, head(None)),
            GateAnswer::Go
        );
        // Off: the running-request fact is ignored.
        let mut off = with_admission(|_| {});
        off.on_classify(Some("r1"), "a", None, None, 100, t(0.0));
        assert_eq!(
            off.gate_check_head("a", t(1.0), &kv, head(Some(false))),
            GateAnswer::Go
        );
    }

    #[test]
    fn surge_holds_lower_classes_while_queued_prefill_outlasts_a_higher_class_ttft() {
        let mut s = with_admission(|a| a.surge = Some(crate::config::SurgeConfig::default()));
        s.on_classify(Some("r1"), "std", None, Some("standard"), 100, t(0.0));
        s.on_classify(Some("r2"), "pri", None, Some("priority"), 100, t(0.0));
        let kv = s.kv_budget(1_000_000, 0);
        let head = |surge| HeadFacts {
            surge,
            ..HeadFacts::default()
        };
        let ttft = s
            .policy()
            .class(Some("priority"))
            .service_level
            .as_ref()
            .unwrap()
            .ttft_s;
        // Queued prefill takes longer than priority's TTFT limit: standard waits, priority goes.
        let long = Some((ttft * 2.0, 0.1));
        assert!(matches!(
            s.gate_check_head("std", t(1.0), &kv, head(long)),
            GateAnswer::Hold { .. }
        ));
        assert_eq!(
            s.gate_check_head("pri", t(1.0), &kv, head(long)),
            GateAnswer::Go
        );
        // Short queue, or no facts: standard goes.
        assert_eq!(
            s.gate_check_head("std", t(1.0), &kv, head(Some((ttft * 0.2, 0.1)))),
            GateAnswer::Go
        );
        assert_eq!(
            s.gate_check_head("std", t(1.0), &kv, head(None)),
            GateAnswer::Go
        );
        // A long first turn alone never holds it: only the queue counts.
        assert_eq!(
            s.gate_check_head("std", t(1.0), &kv, head(Some((ttft * 0.2, ttft * 50.0)))),
            GateAnswer::Go
        );
        // Off: never held.
        let mut off = with_admission(|_| {});
        off.on_classify(Some("r1"), "std", None, Some("standard"), 100, t(0.0));
        assert_eq!(
            off.gate_check_head("std", t(1.0), &kv, head(long)),
            GateAnswer::Go
        );
    }

    #[test]
    fn session_kind_and_user_reach_the_session_and_the_session_log_counts_subagents_for_the_root() {
        let mut s = enforce();
        let hints = RequestAttributes {
            session_kind: Some(crate::session::SessionKind::Agent),
            user: Some("u7".into()),
            ..RequestAttributes::default()
        };
        s.on_classify_with(Some("r1"), "root", None, None, &hints, 400, t(0.0));
        let session = s.sessions().get("root").unwrap();
        assert_eq!(session.kind, Some(crate::session::SessionKind::Agent));
        assert_eq!(session.user.as_deref(), Some("u7"));
        s.on_dispatched("r1", "root", t(1.0));
        // A subagent's request counts for its root in the session log.
        s.on_classify_with(
            Some("r2"),
            "sub",
            Some("root"),
            None,
            &RequestAttributes::default(),
            100,
            t(2.0),
        );
        s.on_dispatched("r2", "sub", t(2.0));
        s.on_completed("r1", Some(500), t(4.0));
        s.on_completed("r2", Some(200), t(6.0));
        let account = s.experience().get("root").unwrap();
        assert_eq!(account.turns, 2);
        assert_eq!(account.kind, Some("agent"));
        assert_eq!(account.user.as_deref(), Some("u7"));
        assert!((account.server_s - 6.0).abs() < 1e-9);
        assert!(s.experience().get("sub").is_none());
        // The root ends: its account is written and dropped.
        let last = RequestAttributes {
            session_final: true,
            ..RequestAttributes::default()
        };
        s.on_classify_with(Some("r3"), "root", None, None, &last, 600, t(50.0));
        s.on_dispatched("r3", "root", t(50.0));
        s.on_completed("r3", Some(700), t(52.0));
        assert!(s.experience().get("root").is_none());
    }

    #[test]
    fn session_age_orders_the_lobby_by_first_offer_across_retries() {
        let mut policy = ServicePolicy::default();
        policy.admission.turn_order = crate::config::TurnOrder::SessionAge;
        let mut s = Scheduler::new(policy);
        // Session a offers first and expires; b offers later; a retries after b.
        s.on_classify_with(
            Some("a1"),
            "a",
            None,
            Some("standard"),
            &RequestAttributes::default(),
            100,
            t(0.0),
        );
        s.on_classify_with(
            Some("b1"),
            "b",
            None,
            Some("standard"),
            &RequestAttributes::default(),
            100,
            t(5.0),
        );
        s.on_aborted_at("a1", t(60.0));
        s.on_classify_with(
            Some("a2"),
            "a",
            None,
            Some("standard"),
            &RequestAttributes::default(),
            100,
            t(70.0),
        );
        let table = s.sessions();
        let (a, b) = (table.get("a").unwrap(), table.get("b").unwrap());
        assert_eq!(a.started, t(0.0));
        assert!(
            table.order_at(a) < table.order_at(b),
            "the older session goes first"
        );
        // First come by turn puts the retry behind b.
        let mut f = Scheduler::new(ServicePolicy::default());
        f.on_classify_with(
            Some("a1"),
            "a",
            None,
            Some("standard"),
            &RequestAttributes::default(),
            100,
            t(0.0),
        );
        f.on_classify_with(
            Some("b1"),
            "b",
            None,
            Some("standard"),
            &RequestAttributes::default(),
            100,
            t(5.0),
        );
        f.on_aborted_at("a1", t(60.0));
        f.on_classify_with(
            Some("a2"),
            "a",
            None,
            Some("standard"),
            &RequestAttributes::default(),
            100,
            t(70.0),
        );
        let table = f.sessions();
        assert!(table.order_at(table.get("a").unwrap()) > table.order_at(table.get("b").unwrap()));
    }

    #[test]
    fn lifetime_counts_sessions_at_their_hinted_peak_in_the_total_budget() {
        let hinted = |peak| RequestAttributes {
            signals: Signals {
                peak_tokens: Some(peak),
                ..Signals::default()
            },
            ..RequestAttributes::default()
        };
        for (lifetime, admitted) in [
            (crate::config::LifetimeBudget::Off, true),
            (crate::config::LifetimeBudget::Total, false),
        ] {
            let mut s = with_admission(|a| {
                a.lifetime = lifetime;
                a.total_headroom = 1.0;
            });
            // 1,000 tokens of HBM and DRAM in all; a 100-token first turn hinting a 5,000 peak.
            s.on_classify_with(Some("r1"), "a", None, None, &hinted(5_000), 100, t(0.0));
            let kv = s.kv_budget(500, 500);
            let answer = s.gate_check("a", t(1.0), &kv);
            assert_eq!(answer == GateAnswer::Go, admitted, "{lifetime:?}");
        }
    }

    // ---- Service class family (b): runs, run headroom, run orders, compute headroom (f36b) ----

    /// Prefill 1 s per 1,000 new tokens, decode 10 ms a token: a request of 1,000 uncached input
    /// tokens and 101 output tokens takes 2 s alone.
    fn unit_steps() -> std::sync::Arc<dyn crate::forecast::StepModel> {
        std::sync::Arc::new(crate::forecast::TableStepModel {
            prefill_tokens: vec![0.0, 1000.0],
            prefill_s: vec![0.0, 1.0],
            prefix_weight: 0.0,
            decode_batch: vec![1.0, 2.0],
            decode_context: vec![1000.0, 2000.0],
            decode_s: vec![0.01, 0.01, 0.01, 0.01],
        })
    }

    fn family_b(f: impl FnOnce(&mut ServicePolicy)) -> Scheduler {
        let mut policy = ServicePolicy {
            mode: Mode::Enforce,
            ..Default::default()
        };
        for c in &mut policy.classes {
            c.run_level = match c.name.as_str() {
                "priority" => Some(crate::config::RunLevel {
                    delay_base_s: 2.0,
                    delay_per_ref: 0.1,
                    slowdown: 1.5,
                    server_share: 0.5,
                    ..Default::default()
                }),
                "standard" => Some(crate::config::RunLevel {
                    slowdown: 3.0,
                    server_share: 0.5,
                    ..Default::default()
                }),
                _ => None,
            };
        }
        f(&mut policy);
        let mut s = Scheduler::new(policy);
        s.set_step_model(unit_steps());
        s.set_chunk_tokens(8192);
        s
    }

    /// [`family_b`] judged as the first f36b build: run headroom and promotion in slowdown, the
    /// deficit in server-wait share.
    fn slowdown_b(f: impl FnOnce(&mut ServicePolicy)) -> Scheduler {
        family_b(|p| {
            p.runs.signal = crate::config::RunSignal::Slowdown;
            p.runs.deficit = crate::config::DeficitMeasure::ServerShare;
            f(p);
        })
    }

    /// One request of `session` from `at` to `done` with 1,000 new input tokens and 101 output
    /// tokens (2 s alone), leaving no context behind.
    fn serve(s: &mut Scheduler, request: &str, session: &str, class: &str, at: f64, done: f64) {
        s.on_classify_with(
            Some(request),
            session,
            None,
            Some(class),
            &RequestAttributes::default(),
            1000,
            t(at),
        );
        s.on_dispatched(request, session, t(at));
        s.on_completed_with_output(request, Some(0), Some(101), t(done));
    }

    /// `met` runs of `class` within its slowdown target (2 s for 2 s alone) and `missed` over it
    /// (20 s for 2 s alone), each its own session, ended by the client.
    fn runs(s: &mut Scheduler, class: &str, from: f64, met: usize, missed: usize) {
        for i in 0..met + missed {
            let at = from + i as f64 * 30.0;
            let id = format!("{class}-run{i}-{from}");
            serve(
                s,
                &id,
                &id,
                class,
                at,
                at + if i < met { 2.0 } else { 20.0 },
            );
            s.on_session_end(&id);
        }
    }

    #[test]
    fn slowdown_run_attainment_feeds_the_headroom_readers_only_when_switched_on() {
        use crate::config::HeadroomSource;
        let mut s = slowdown_b(|p| p.runs.headroom = HeadroomSource::Run);
        runs(&mut s, "standard", 0.0, 4, 1);
        let now = t(200.0);
        assert_eq!(s.run_share("standard", now), Some(0.8));
        assert!((s.run_headroom("standard", now).unwrap() + 0.15).abs() < 1e-9);
        // What the soft reserve, release and the coordinator read: target 0.7 plus run headroom.
        assert!((s.window_share("standard", now).unwrap() - 0.55).abs() < 1e-9);
        // Under five runs, or past the window, nothing is known.
        assert_eq!(s.run_share("priority", now), None);
        assert_eq!(s.run_share("standard", t(5000.0)), None);
        // Off (the default): the request windows, as before.
        let mut off = slowdown_b(|_| {});
        runs(&mut off, "standard", 0.0, 4, 1);
        assert_eq!(
            off.window_share("standard", now),
            off.metrics.window_share("standard", now)
        );
        assert_eq!(off.run_share("standard", now), Some(0.8));
        // The tighter of the two: request windows have no completions here, so the run's.
        let mut min = slowdown_b(|p| p.runs.headroom = HeadroomSource::Min);
        runs(&mut min, "standard", 0.0, 4, 1);
        assert!((min.window_share("standard", now).unwrap() - 0.55).abs() < 1e-9);
    }

    #[test]
    fn the_run_log_prices_requests_alone_and_counts_the_previous_context_as_cached() {
        let mut s = family_b(|_| {});
        // 1,000 tokens alone: 1 s prefill and 100 decode steps of 10 ms; it leaves 1,101 tokens.
        s.on_classify(Some("r1"), "a", None, Some("standard"), 1000, t(0.0));
        s.on_dispatched("r1", "a", t(0.0));
        s.on_completed_with_output("r1", Some(1_101), Some(101), t(5.0));
        let a = s.experience().get("a").unwrap();
        assert!((a.run.attained_s - 2.0).abs() < 1e-9);
        // The next request carries the whole context (1,101 tokens) of which a worker held
        // 1,000; at no contention the session's previous context (1,101) is cached.
        s.on_classify_with(
            Some("r2"),
            "a",
            None,
            Some("standard"),
            &RequestAttributes::default(),
            101,
            t(6.0),
        );
        s.set_request_prompt("r2", 1_101);
        s.on_dispatched("r2", "a", t(6.0));
        s.on_completed_with_output("r2", Some(1_202), Some(1), t(7.0));
        // A request with nothing new and one output token: an empty prefill step.
        let a = s.experience().get("a").unwrap();
        assert!(
            (a.run.attained_s - 2.0).abs() < 1e-9,
            "{}",
            a.run.attained_s
        );
        // Without a step model nothing is priced.
        let mut bare = Scheduler::new(ServicePolicy {
            mode: Mode::Enforce,
            ..Default::default()
        });
        serve(&mut bare, "r1", "a", "standard", 0.0, 5.0);
        assert_eq!(bare.experience().get("a").unwrap().run.attained_s, 0.0);
    }

    #[test]
    fn run_las_orders_by_attained_service_and_promotes_a_run_past_its_target() {
        use crate::config::TurnOrder;
        let key = |s: &mut Scheduler, request: &str, session: &str, at: f64| {
            s.on_classify_with(
                Some(request),
                session,
                None,
                Some("standard"),
                &RequestAttributes::default(),
                1000,
                t(at),
            );
            s.request_order_key(request).unwrap()
        };
        for (slowdown, a_key) in [(100.0, 23.0), (3.0, 12.0)] {
            let mut s = slowdown_b(|p| {
                p.admission.turn_order = TurnOrder::RunLas;
                p.classes[1].run_level.as_mut().unwrap().slowdown = slowdown;
            });
            // a: 2 s of service in its run, then a 1 s tool gap.
            serve(&mut s, "a1", "a", "standard", 0.0, 2.0);
            let a = key(&mut s, "a2", "a", 3.0);
            // Arrival 3 plus 10 x 2 s attained = 23; at slowdown 3 the run is promoted at
            // 0 + 3 x (3 s progress + 1 s own prefill) = 12.
            assert!((a - a_key).abs() < 1e-9, "{slowdown}: {a}");
            // A fresh run keys at its arrival.
            assert!((key(&mut s, "b1", "b", 5.0) - 5.0).abs() < 1e-9);
            let c = key(&mut s, "c1", "c", 20.0);
            // Pure LAS lets a later fresh run go first; promotion puts a ahead of it.
            assert_eq!(c < a, slowdown > 10.0, "{slowdown}");
        }
        // Off: no key.
        let mut fifo = slowdown_b(|_| {});
        fifo.on_classify(Some("x"), "x", None, None, 100, t(0.0));
        assert_eq!(fifo.request_order_key("x"), None);
    }

    #[test]
    fn an_autonomous_run_is_promoted_by_its_floor_not_its_class_target() {
        use crate::config::TurnOrder;
        for (floor, promoted) in [(0.5, true), (0.1, false)] {
            let mut s = family_b(|p| {
                p.admission.turn_order = TurnOrder::RunLas;
                p.runs.autonomous_after_s = 50.0;
                p.runs.floor_share = floor;
                p.runs.floor_window_s = 20.0;
            });
            // Back to back, 2 s of service each taking 10 s: no stop point for 60 s.
            for i in 0..6 {
                let at = f64::from(i) * 10.0;
                serve(&mut s, &format!("d{i}"), "d", "standard", at, at + 10.0);
            }
            s.on_classify_with(
                Some("d6"),
                "d",
                None,
                Some("standard"),
                &RequestAttributes::default(),
                1000,
                t(60.0),
            );
            let k = s.request_order_key("d6").unwrap();
            // Progress 12 s, 8 s of it by 40 s: promoted at 40 + 4 / floor (48 at 0.5, 80 at
            // 0.1), not at its slowdown target (it is 5x over 3), and not by attained service
            // (60 + 10 x 12 = 180).
            let expect = 40.0 + 4.0 / floor;
            assert!((k - expect).abs() < 1e-9, "{floor}: {k}");
            assert_eq!(k < 60.0, promoted);
        }
    }

    #[test]
    fn server_wait_deficit_puts_the_session_furthest_over_its_share_first_and_a_retry_keeps_its_wait()
     {
        use crate::config::TurnOrder;
        let mut s = slowdown_b(|p| p.admission.turn_order = TurnOrder::ServerWaitDeficit);
        let attrs = RequestAttributes::default();
        // a waited on the server 10 s of its first 20; b 2 s.
        serve(&mut s, "a1", "a", "standard", 0.0, 10.0);
        serve(&mut s, "b1", "b", "standard", 0.0, 2.0);
        // c waited 15 s in the lobby, expired, and its client retries at 20 s.
        s.on_classify_with(Some("c1"), "c", None, Some("standard"), &attrs, 100, t(0.0));
        s.on_aborted_at("c1", t(15.0));
        // d arrives first but has waited nothing.
        s.on_classify_with(
            Some("d1"),
            "d",
            None,
            Some("standard"),
            &attrs,
            100,
            t(19.0),
        );
        for (r, session) in [("a2", "a"), ("b2", "b"), ("c2", "c")] {
            s.on_classify_with(
                Some(r),
                session,
                None,
                Some("standard"),
                &attrs,
                100,
                t(20.0),
            );
        }
        let k = |r: &str| s.request_order_key(r).unwrap();
        // Target share 0.5: a at 0.5 (0), b at 0.1 (0.4), c at 0.75 (-0.25), d new (0.5).
        assert!((k("a2") - 0.0).abs() < 1e-9);
        assert!((k("b2") - 0.4).abs() < 1e-9);
        assert!((k("c2") + 0.25).abs() < 1e-9);
        assert!((k("d1") - 0.5).abs() < 1e-9);
        // In the lobby, c (retried at 20 s) goes ahead of d (arrived at 19 s).
        let order: Vec<_> = s
            .sessions()
            .lobby()
            .ordered()
            .iter()
            .map(|(_, k, _)| k.session.clone())
            .collect();
        assert_eq!(order, vec!["c".to_string(), "d".to_string()]);
        let scan = s.lobby_scan();
        assert_eq!(
            scan.iter().find(|e| e.session == "c").unwrap().order,
            -250_000
        );
    }

    #[test]
    fn forecast_harm_new_holds_new_sessions_that_harm_their_class_or_above_and_spares_batch() {
        let make = |on: bool| {
            let mut policy = ServicePolicy {
                mode: Mode::Enforce,
                ..Default::default()
            };
            if on {
                policy.admission.forecast_harm_new =
                    Some(crate::config::ForecastHarmNewConfig::default());
            }
            let mut s = Scheduler::new(policy);
            for (id, session, class) in [
                ("p1", "p", "priority"),
                ("s1", "s", "standard"),
                ("b1", "b", "batch"),
            ] {
                s.on_classify(Some(id), session, None, Some(class), 100, t(0.0));
            }
            s
        };
        let head = |new_harm| HeadFacts {
            new_harm,
            ..HeadFacts::default()
        };
        // Off: the harm the host reports changes nothing (the answer without it), for every class.
        let mut off = make(false);
        let kv = off.kv_budget(1_000_000, 0);
        for session in ["p", "s", "b"] {
            let plain = off.gate_check_head(session, t(1.0), &kv, HeadFacts::default());
            assert_eq!(plain, GateAnswer::Go);
            assert_eq!(
                off.gate_check_head(session, t(1.0), &kv, head(Some(30.0))),
                plain
            );
        }
        // On: priority and standard placements adding over 0.01 weighted misses wait in the lobby.
        let mut on = make(true);
        let kv = on.kv_budget(1_000_000, 0);
        for session in ["p", "s"] {
            assert!(matches!(
                on.gate_check_head(session, t(1.0), &kv, head(Some(30.0))),
                GateAnswer::Hold { .. }
            ));
            assert!(matches!(
                on.decisions().entries().last().map(|d| d.decision.clone()),
                Some(Decision::Hold {
                    reason: HoldReason::ForecastHarm { added, limit },
                    ..
                }) if added == 30.0 && limit == 0.01
            ));
            // At the limit, and unknown, it goes.
            assert_eq!(
                on.gate_check_head(session, t(1.0), &kv, head(Some(0.01))),
                GateAnswer::Go
            );
            assert_eq!(
                on.gate_check_head(session, t(1.0), &kv, head(None)),
                GateAnswer::Go
            );
        }
        // Batch is exempt.
        assert_eq!(
            on.gate_check_head("b", t(1.0), &kv, head(Some(30.0))),
            GateAnswer::Go
        );
        // At its wait limit (30 s) priority is admitted anyway; standard's limit is 60 s.
        assert_eq!(
            on.gate_check_head("p", t(31.0), &kv, head(Some(30.0))),
            GateAnswer::Go
        );
        assert!(matches!(
            on.gate_check_head("s", t(31.0), &kv, head(Some(30.0))),
            GateAnswer::Hold { .. }
        ));
    }

    /// f49d `DYNAMO_HARM_DEBUG`: the counters classify the host's input on real session states,
    /// count the reach and the hold, and change no answer.
    #[test]
    fn harm_debug_counts_lobby_and_admitted_checks_and_changes_no_decision() {
        use crate::harm_debug::{HarmCheck, classify};
        let make = |debug: bool| {
            let mut policy = ServicePolicy {
                mode: Mode::Enforce,
                ..Default::default()
            };
            policy.admission.forecast_harm_new =
                Some(crate::config::ForecastHarmNewConfig::default());
            let mut s = Scheduler::new(policy);
            if debug {
                s.enable_harm_debug();
            }
            for (id, session, class) in [
                ("p1", "p", "priority"),
                ("s1", "s", "standard"),
                ("b1", "b", "batch"),
            ] {
                s.on_classify(Some(id), session, None, Some(class), 100, t(0.0));
            }
            s
        };
        let exempt = crate::config::ForecastHarmNewConfig::default().exempt_classes;
        let state =
            |s: &Scheduler, id: &str| s.sessions().get(id).map(|x| (x.class.clone(), x.state));
        let check = |s: &Scheduler, id: &str, rows_empty: bool, harm: Option<f64>| {
            let found = state(s, id);
            classify(
                found.as_ref().map(|(c, st)| (c.as_str(), *st)),
                &exempt,
                true,
                rows_empty,
                harm,
            )
        };
        let head = |new_harm| HeadFacts {
            new_harm,
            ..HeadFacts::default()
        };
        let mut on = make(true);
        let mut off = make(false);
        assert!(on.harm_debug_on() && !off.harm_debug_on());
        let kv = on.kv_budget(1_000_000, 0);
        // New sessions wait in the lobby; batch is exempt; unknown sessions are not found.
        assert_eq!(check(&on, "s", false, Some(0.5)), HarmCheck::Harm(0.5));
        assert_eq!(check(&on, "s", true, None), HarmCheck::NoRows);
        assert_eq!(check(&on, "b", false, None), HarmCheck::Exempt);
        assert_eq!(check(&on, "nobody", false, None), HarmCheck::NoSession);
        // The same checks on both: identical answers and decision logs.
        let steps = [
            ("s", 1.0, Some(0.5)),
            ("p", 1.0, Some(0.0)),
            ("b", 1.0, Some(9.0)),
            ("s", 2.0, None),
        ];
        for (session, at, harm) in steps {
            let c = check(&on, session, false, harm);
            let class = state(&on, session).map(|x| x.0).unwrap();
            on.harm_debug_check(&class, c, t(at));
            off.harm_debug_check(&class, c, t(at));
            assert_eq!(
                on.gate_check_head(session, t(at), &kv, head(harm)),
                off.gate_check_head(session, t(at), &kv, head(harm))
            );
        }
        assert!(on.decisions().entries().eq(off.decisions().entries()));
        assert!(off.harm_debug().is_none());
        let d = on.harm_debug().unwrap();
        let s = d.counts("standard").unwrap();
        // Held at 1 s on 0.5 added misses; at 2 s, unknown harm, it goes.
        assert_eq!((s.checks, s.lobby, s.harm_some, s.harm_none), (2, 2, 1, 1));
        assert_eq!((s.reached, s.held, s.expired), (2, 1, 0));
        assert_eq!(s.hist, [0, 0, 0, 0, 1, 0]);
        // Priority's 0 is reached and not held.
        let p = d.counts("priority").unwrap();
        assert_eq!((p.harm_some, p.reached, p.held, p.hist[0]), (1, 1, 0, 1));
        // Batch: exempt at the host; the hold is reached and lets it go.
        let b = d.counts("batch").unwrap();
        assert_eq!((b.exempt, b.reached, b.held), (1, 1, 0));
        // Dispatched, the session is admitted: no longer a lobby check.
        on.on_dispatched("p1", "p", t(3.0));
        assert_eq!(check(&on, "p", false, None), HarmCheck::NotLobby);
    }

    #[test]
    fn compute_headroom_holds_new_sessions_while_a_class_at_or_above_is_short_of_its_run_target() {
        let cfg = crate::config::ComputeHeadroomConfig::default();
        let mut s = slowdown_b(|p| p.admission.compute_headroom = Some(cfg.clone()));
        // Standard: 4 of 5 runs within target, 0.8 against 0.95.
        runs(&mut s, "standard", 0.0, 4, 1);
        s.on_classify(Some("n1"), "n", None, Some("standard"), 100, t(200.0));
        let kv = s.kv_budget(1_000_000, 0);
        assert!(matches!(
            s.gate_check("n", t(200.0), &kv),
            GateAnswer::Hold { .. }
        ));
        assert!(matches!(
            s.decisions().entries().last().map(|d| d.decision.clone()),
            Some(Decision::Hold {
                reason: HoldReason::ComputeHeadroom { ref class, .. },
                ..
            }) if class == "standard"
        ));
        // Priority ranks above standard: standard's shortfall does not hold it.
        s.on_classify(Some("p1"), "p", None, Some("priority"), 100, t(200.0));
        assert_eq!(s.gate_check("p", t(200.0), &kv), GateAnswer::Go);
        // Only agent sessions are held when the run family is limited to agents.
        let mut mixed = slowdown_b(|p| {
            p.admission.compute_headroom = Some(cfg.clone());
            p.runs.kind = Some("agent".into());
        });
        let agent = RequestAttributes {
            session_kind: Some(crate::session::SessionKind::Agent),
            ..RequestAttributes::default()
        };
        for i in 0..5 {
            let id = format!("r{i}");
            let at = f64::from(i) * 30.0;
            mixed.on_classify_with(Some(&id), &id, None, Some("standard"), &agent, 1000, t(at));
            mixed.on_dispatched(&id, &id, t(at));
            let done = at + if i < 4 { 2.0 } else { 20.0 };
            mixed.on_completed_with_output(&id, Some(0), Some(101), t(done));
            mixed.on_session_end(&id);
        }
        assert_eq!(mixed.run_share("standard", t(200.0)), Some(0.8));
        mixed.on_classify(Some("i1"), "i", None, Some("standard"), 100, t(200.0));
        assert_eq!(mixed.gate_check("i", t(200.0), &kv), GateAnswer::Go);
        mixed.on_classify_with(
            Some("g1"),
            "g",
            None,
            Some("standard"),
            &agent,
            100,
            t(200.0),
        );
        assert!(matches!(
            mixed.gate_check("g", t(200.0), &kv),
            GateAnswer::Hold { .. }
        ));
        // Interactive runs are not counted for run attainment.
        let mut only = slowdown_b(|p| p.runs.kind = Some("agent".into()));
        runs(&mut only, "standard", 0.0, 4, 1);
        assert_eq!(only.run_share("standard", t(200.0)), None);
    }

    #[test]
    fn without_the_hot_budget_only_the_total_budget_binds() {
        for (hot_budget, held) in [(true, true), (false, false)] {
            let mut s = with_admission(|a| a.hot_budget = hot_budget.into());
            s.on_classify(Some("r1"), "a", None, None, 600, t(0.0));
            s.on_dispatched("r1", "a", t(0.0));
            s.on_classify(Some("r2"), "b", None, None, 600, t(1.0));
            // a in flight with 600: standard's hot room 900 - 600 = 300 < 600; the total budget
            // (0.95 x 2,000 - 600) has room.
            let kv = s.kv_budget(1000, 1000);
            let answer = s.gate_check("b", t(1.0), &kv);
            assert_eq!(answer != GateAnswer::Go, held, "{hot_budget}");
        }
    }

    #[test]
    fn a_frontend_tool_result_keeps_the_run_and_only_the_roots_requests_mark_stops() {
        let mut s = family_b(|_| {});
        serve(&mut s, "r1", "a", "standard", 0.0, 2.0);
        // 300 s later, but the request continues with a tool result: same run.
        let tool = RequestAttributes {
            next_known: true,
            ..RequestAttributes::default()
        };
        s.on_classify_with(
            Some("r2"),
            "a",
            None,
            Some("standard"),
            &tool,
            100,
            t(302.0),
        );
        assert_eq!(s.experience().get("a").unwrap().stops, 0);
        s.on_dispatched("r2", "a", t(302.0));
        s.on_completed_with_output("r2", Some(0), Some(1), t(303.0));
        // A subagent's first request starts with a user message: not a stop point of the root.
        let user = RequestAttributes {
            next_is_user: true,
            next_known: true,
            ..RequestAttributes::default()
        };
        s.on_classify_with(Some("r3"), "sub", Some("a"), None, &user, 100, t(304.0));
        assert_eq!(s.experience().get("a").unwrap().stops, 0);
        // The root's own user message is one.
        s.on_dispatched("r3", "sub", t(304.0));
        s.on_completed_with_output("r3", Some(0), Some(1), t(305.0));
        s.on_classify_with(
            Some("r4"),
            "a",
            None,
            Some("standard"),
            &user,
            100,
            t(306.0),
        );
        assert_eq!(s.experience().get("a").unwrap().stops, 1);
    }

    // ---- Added delay D_1 (docs/test-plan-full.md, "The run metric: added delay") ----

    /// A main-agent request of `session` classified at `at`, dispatched at `sent`, first token at
    /// `first`, done at `done`: 1,000 new tokens and 101 out (2 s alone, 1 s of it prefill).
    fn timed(
        s: &mut Scheduler,
        request: &str,
        session: &str,
        at: f64,
        sent: f64,
        first: f64,
        done: f64,
    ) {
        s.on_classify_with(
            Some(request),
            session,
            None,
            Some("standard"),
            &RequestAttributes::default(),
            1000,
            t(at),
        );
        s.on_dispatched(request, session, t(sent));
        s.on_first_token(request, t(first));
        s.on_completed_with_output(request, Some(0), Some(101), t(done));
    }

    #[test]
    fn d1_is_router_wait_plus_prefill_excess_and_the_class_headroom_is_delay_over_allowance() {
        use crate::config::HeadroomSource;
        let mut s = family_b(|p| p.runs.headroom = HeadroomSource::Run);
        // a waits 30 s at the router, then 10 s to its first token against 1 s of reference
        // prefill: D_1 = 30 + 9. b is served at once.
        timed(&mut s, "a1", "a", 0.0, 30.0, 40.0, 41.0);
        timed(&mut s, "b1", "b", 0.0, 0.0, 1.0, 2.0);
        let a = &s.experience().get("a").unwrap().run;
        assert!((a.waiting_s - 30.0).abs() < 1e-9 && (a.prefill_excess_s - 9.0).abs() < 1e-9);
        assert!((a.s_ref_s - 2.0).abs() < 1e-9);
        // Standard's allowance: 10 s per run plus 25% of 2 s each: 21 s against 39 s of delay.
        let h = s.run_headroom("standard", t(50.0)).unwrap();
        assert!((h - (1.0 - 39.0 / 21.0)).abs() < 1e-9, "{h}");
        // The readers see target 0.7 moved down by h x 0.7.
        let read = s.window_share("standard", t(50.0)).unwrap();
        assert!((read - (0.7 + h * 0.7)).abs() < 1e-9, "{read}");
        // Past the window, nothing is known.
        assert_eq!(s.run_headroom("standard", t(1000.0)), None);
    }

    #[test]
    fn a_refusal_and_its_retry_backoff_are_delay_and_the_run_line_carries_d1_and_its_allowance() {
        let mut s = family_b(|_| {});
        let attrs = RequestAttributes::default();
        // The first attempt waits 60 s in the lobby and expires; the client retries 100 s later:
        // the backoff is delay, not client time. (A client silent for the stop gap gave up.)
        s.on_classify_with(
            Some("a1"),
            "a",
            None,
            Some("standard"),
            &attrs,
            1000,
            t(0.0),
        );
        s.on_aborted_at("a1", t(60.0));
        timed(&mut s, "a2", "a", 160.0, 160.0, 161.0, 162.0);
        let account = s.experience().get("a").unwrap();
        assert_eq!(account.stops, 0);
        assert!((account.client_s - 0.0).abs() < 1e-9);
        assert!((account.run.waiting_s - 160.0).abs() < 1e-9);
        // The session ends: one run, D_1 160 s against 10 + 0.25 x 2 s.
        s.on_session_end("a");
        let a = s.experience().get("a");
        assert!(a.is_none());
        let line = crate::experience::run_line(&crate::experience::FinishedRun {
            session: "a".into(),
            class: "standard".into(),
            kind: None,
            start: t(0.0),
            end: t(162.0),
            server_s: 62.0,
            uncontended_s: Some(2.0),
            requests: 1,
            ended_by: "end",
            waiting_s: 160.0,
            prefill_excess_s: 0.0,
            s_ref_s: 2.0,
            allowance_s: Some(10.5),
        });
        assert!(
            line.contains("\"d1_s\":160.00") && line.contains("\"allowance_s\":10.50"),
            "{line}"
        );
        // A throttled start counts the same way: its retry's backoff is delay.
        let mut th = family_b(|_| {});
        th.note_throttled("z", Some("standard"), t(0.0));
        th.on_classify_with(
            Some("z1"),
            "z",
            None,
            Some("standard"),
            &attrs,
            1000,
            t(30.0),
        );
        assert!((th.experience().get("z").unwrap().run.waiting_s - 30.0).abs() < 1e-9);
        assert_eq!(th.experience().get("z").unwrap().refusals, 1);
    }

    #[test]
    fn run_las_promotes_a_run_once_its_delay_would_pass_its_allowance() {
        use crate::config::TurnOrder;
        let mut s = family_b(|p| p.admission.turn_order = TurnOrder::RunLas);
        // 5 s of router wait so far, 2 s of reference service: allowance 10 + 0.25 x (2 + 1 s of
        // own prefill to come) = 10.75, slack 5.75 s.
        timed(&mut s, "a1", "a", 0.0, 5.0, 6.0, 7.0);
        s.on_classify_with(
            Some("a2"),
            "a",
            None,
            Some("standard"),
            &RequestAttributes::default(),
            1000,
            t(8.0),
        );
        let k = s.request_order_key("a2").unwrap();
        // Not 8 + 10 x 2 = 28: promoted at 8 + 5.75.
        assert!((k - 13.75).abs() < 1e-9, "{k}");
    }

    #[test]
    fn delay_deficit_orders_the_session_furthest_over_its_allowance_first() {
        use crate::config::TurnOrder;
        let mut s = family_b(|p| p.admission.turn_order = TurnOrder::ServerWaitDeficit);
        timed(&mut s, "a1", "a", 0.0, 30.0, 31.0, 32.0); // 30 s of delay against 10.5
        timed(&mut s, "b1", "b", 0.0, 0.0, 1.0, 2.0); // none
        let attrs = RequestAttributes::default();
        s.on_classify_with(
            Some("a2"),
            "a",
            None,
            Some("standard"),
            &attrs,
            1000,
            t(40.0),
        );
        s.on_classify_with(
            Some("b2"),
            "b",
            None,
            Some("standard"),
            &attrs,
            1000,
            t(40.0),
        );
        let (ka, kb) = (
            s.request_order_key("a2").unwrap(),
            s.request_order_key("b2").unwrap(),
        );
        assert!((ka - (10.5 - 30.0)).abs() < 1e-9, "{ka}");
        assert!((kb - 10.5).abs() < 1e-9, "{kb}");
    }

    #[test]
    fn compute_headroom_in_delay_holds_new_sessions_while_delay_exceeds_the_allowance() {
        let mut s = family_b(|p| {
            p.admission.compute_headroom = Some(crate::config::ComputeHeadroomConfig::default())
        });
        timed(&mut s, "a1", "a", 0.0, 30.0, 31.0, 32.0);
        s.on_classify(Some("n1"), "n", None, Some("standard"), 100, t(40.0));
        let kv = s.kv_budget(1_000_000, 0);
        assert!(matches!(
            s.gate_check("n", t(40.0), &kv),
            GateAnswer::Hold { .. }
        ));
        let mut fresh = family_b(|p| {
            p.admission.compute_headroom = Some(crate::config::ComputeHeadroomConfig::default())
        });
        timed(&mut fresh, "a1", "a", 0.0, 0.0, 1.0, 2.0);
        fresh.on_classify(Some("n1"), "n", None, Some("standard"), 100, t(40.0));
        assert_eq!(fresh.gate_check("n", t(40.0), &kv), GateAnswer::Go);
    }

    #[test]
    fn hot_exempt_classes_admit_on_the_total_budget_and_leave_the_hot_budget_to_the_others() {
        // HBM too small for any first turn, DRAM plenty: the hot budget alone holds a session.
        let mut gated = with_admission(|_| {});
        gated.on_classify(Some("r1"), "std", None, Some("standard"), 100, t(0.0));
        let kv = gated.kv_budget(50, 1_000_000);
        assert!(matches!(
            gated.gate_check_head("std", t(1.0), &kv, HeadFacts::default()),
            GateAnswer::Hold { .. }
        ));
        // Standard left out of the hot budget: admitted on the total budget.
        let mut s = with_admission(|a| a.hot_exempt_classes = vec!["standard".into()]);
        s.on_classify(Some("r1"), "std", None, Some("standard"), 100, t(0.0));
        let kv = s.kv_budget(50, 1_000_000);
        assert_eq!(
            s.gate_check_head("std", t(1.0), &kv, HeadFacts::default()),
            GateAnswer::Go
        );
        // In flight, it counts nothing toward the hot budget, but still toward the total.
        s.on_dispatched("r1", "std", t(1.0));
        let kv = s.kv_budget(50, 1_000_000);
        assert_eq!(kv.hot_tokens, 0);
        assert!(kv.total_tokens > 0);
        // Priority is still gated on HBM.
        s.on_classify(Some("r2"), "pri", None, Some("priority"), 100, t(2.0));
        let kv = s.kv_budget(50, 1_000_000);
        assert!(matches!(
            s.gate_check_head("pri", t(3.0), &kv, HeadFacts::default()),
            GateAnswer::Hold { .. }
        ));
    }

    #[test]
    fn onboard_pace_holds_new_sessions_while_the_bucket_is_in_debt() {
        let pace = crate::config::OnboardPaceConfig {
            tokens_per_s: 100.0,
            burst_tokens: 100.0,
            classes: vec!["standard".into()],
        };
        let mut s = with_admission(|a| a.onboard_pace = Some(pace.clone()));
        let kv = s.kv_budget(1_000_000, 0);
        // The bucket starts full: the first session goes, and its 300 tokens put it 200 in debt.
        s.on_classify(Some("a1"), "a", None, Some("standard"), 300, t(0.0));
        assert_eq!(
            s.gate_check_head("a", t(0.0), &kv, HeadFacts::default()),
            GateAnswer::Go
        );
        s.on_dispatched("a1", "a", t(0.0));
        assert!((s.onboard_level() + 200.0).abs() < 1e-6);
        // In debt at 1 s (-100): the next waits; by 2.5 s (+50) it goes.
        s.on_classify(Some("b1"), "b", None, Some("standard"), 100, t(0.5));
        let kv = s.kv_budget(1_000_000, 0);
        assert!(matches!(
            s.gate_check_head("b", t(1.0), &kv, HeadFacts::default()),
            GateAnswer::Hold { .. }
        ));
        assert_eq!(
            s.gate_check_head("b", t(2.5), &kv, HeadFacts::default()),
            GateAnswer::Go
        );
        // A class it does not pace is never held, and takes nothing from the bucket.
        s.on_classify(Some("p1"), "p", None, Some("priority"), 100, t(2.5));
        let kv = s.kv_budget(1_000_000, 0);
        assert_eq!(
            s.gate_check_head("p", t(2.5), &kv, HeadFacts::default()),
            GateAnswer::Go
        );
        // Off: the bucket never holds.
        let mut off = with_admission(|_| {});
        off.on_classify(Some("a1"), "a", None, Some("standard"), 300, t(0.0));
        off.on_dispatched("a1", "a", t(0.0));
        off.on_classify(Some("b1"), "b", None, Some("standard"), 100, t(0.5));
        let kv = off.kv_budget(1_000_000, 0);
        assert_eq!(
            off.gate_check_head("b", t(1.0), &kv, HeadFacts::default()),
            GateAnswer::Go
        );
    }
}

#[cfg(test)]
mod f49m_tests {
    //! f49m `admission.honest_retry`.
    use super::*;
    use crate::config::{ReturneeFirstConfig, TicketsConfig};
    use crate::honest_retry::HonestRetryConfig;

    fn t(s: f64) -> Time {
        Time::from_secs_f64(s)
    }

    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    /// Enforce mode with `tickets` and, with `honest`, `honest_retry`.
    fn sched(
        tickets: impl FnOnce(&mut TicketsConfig),
        honest: Option<fn(&mut HonestRetryConfig)>,
    ) -> Scheduler {
        let mut policy = ServicePolicy {
            mode: Mode::Enforce,
            ..Default::default()
        };
        let mut tc = TicketsConfig::default();
        tickets(&mut tc);
        policy.admission.tickets = Some(tc);
        policy.admission.honest_retry = honest.map(|f| {
            let mut hc = HonestRetryConfig::default();
            f(&mut hc);
            hc
        });
        policy.validate().unwrap();
        Scheduler::new(policy)
    }

    fn admit(s: &mut Scheduler, id: &str, class: &str, at: f64) {
        s.on_classify(Some(id), id, None, Some(class), 100, t(at));
        s.on_dispatched(id, id, t(at));
    }

    /// The gate's KV picture (`hbm` HBM, no DRAM), from a batch session's check.
    fn picture(s: &mut Scheduler, hbm: u64, at: f64) {
        s.on_classify(Some("rg"), "g", None, Some("batch"), 1, t(at));
        check(s, "g", hbm, at);
    }

    fn check(s: &mut Scheduler, session: &str, hbm: u64, at: f64) -> GateAnswer {
        let kv = s.kv_budget(hbm, 0);
        s.gate_check(session, t(at), &kv)
    }

    /// (a) Retry-After = N over the class's measured admission rate, N counting its lobby and its
    /// away ticket holders ahead; a waiting session's advice is less the wait it has left.
    #[test]
    fn f49m_a_retry_after_is_place_over_the_measured_admission_rate() {
        let mut s = sched(|_| {}, Some(|_| {}));
        for i in 0..6 {
            admit(&mut s, &format!("a{i}"), "standard", 40.0 + 10.0 * i as f64);
        }
        // Six admissions, the run 100 s old: 0.06 a second. Empty lobby, no ticket: N = 1.
        let r = s
            .ticket_on_throttle("n1", Some("standard"), 100, 100, t(100.0))
            .unwrap();
        assert!(near(r, 1.0 / 0.06), "{r}");
        let due = s.tickets().get("n1").unwrap().due.as_secs_f64();
        assert!(near(due, 100.0 + r), "the ticket carries it");
        // Two waiting and n1 away on its ticket: N = 2 + 1 + 1.
        s.on_classify(Some("rw1"), "w1", None, Some("standard"), 100, t(99.0));
        s.on_classify(Some("rw2"), "w2", None, Some("standard"), 100, t(100.0));
        let r = s
            .ticket_on_throttle("n2", Some("standard"), 100, 100, t(100.0))
            .unwrap();
        assert!(near(r, 4.0 / 0.06), "{r}");
        // w2 behind w1, n1 and n2 away: N = 4, advised 30 s before its deadline.
        let r = s.honest_lobby_retry_after("w2", 30.0, t(100.0)).unwrap();
        assert!(near(r, 4.0 / 0.06 - 30.0), "{r}");
        // Nothing measured, no KV picture, no `retry_after_s`: no advice and no ticket.
        assert_eq!(
            s.ticket_on_throttle("p", Some("priority"), 100, 100, t(100.0)),
            None
        );
        assert!(s.tickets().get("p").is_none());
    }

    /// (a) Fewer than five admissions: the ticket forecast (a listed class, once the gate has a KV
    /// picture), then `retry_after_s`, clamped; five or more: the rate.
    #[test]
    fn f49m_a_the_fallback_order_is_forecast_then_retry_after_s() {
        let mut s = sched(|_| {}, Some(|_| {}));
        assert_eq!(
            s.ticket_on_throttle("b", Some("batch"), 100, 100, t(1.0)),
            None
        );
        s.policy.admission.retry_after_s = Some(45.0);
        assert_eq!(
            s.ticket_on_throttle("b", Some("batch"), 100, 100, t(1.0)),
            Some(45.0)
        );
        assert!(s.tickets().get("b").is_none(), "batch is not listed");
        // Listed, but no KV picture yet: no forecast.
        assert_eq!(
            s.ticket_on_throttle("x", Some("standard"), 100, 100, t(1.0)),
            Some(45.0)
        );
        picture(&mut s, 1000, 2.0);
        let f = s
            .ticket_forecast("y", "standard", 300, 0.0, t(3.0))
            .clamp(5.0, 600.0);
        assert_eq!(
            s.ticket_on_throttle("y", Some("standard"), 300, 0, t(3.0)),
            Some(f)
        );
        // No ticket (a sub-session without `per_root`): no forecast.
        assert_eq!(
            s.honest_throttle_untracked("sub", Some("standard"), t(3.0)),
            Some(45.0)
        );
        for i in 0..5 {
            admit(&mut s, &format!("a{i}"), "standard", 10.0 + 10.0 * i as f64);
        }
        // Five in the 60 s since the start; x and y away: N = 3, 3 / (5 / 60).
        assert!(s.tickets().is_live("x", t(60.0)) && s.tickets().is_live("y", t(60.0)));
        let r = s
            .ticket_on_throttle("z", Some("standard"), 100, 0, t(60.0))
            .unwrap();
        assert!(near(r, 36.0), "{r}");
    }

    /// (b) The third refusal makes the ticket a reservation: its need is held for it (a standard
    /// session that would fit otherwise waits), it is admitted on its return, and its admission
    /// uses it up and starts the count over.
    #[test]
    fn f49m_b_the_third_refusal_reserves_and_the_return_is_admitted() {
        let run = |max_refusals: u32| {
            let honest: fn(&mut HonestRetryConfig) = if max_refusals == 3 {
                |c| c.reserve_frac = 0.5
            } else {
                |c| {
                    c.reserve_frac = 0.5;
                    c.max_refusals = 10;
                }
            };
            let mut s = sched(|c| c.max_hold_frac = 0.0, Some(honest));
            picture(&mut s, 1000, 0.0);
            s.ticket_on_throttle("w", Some("standard"), 400, 0, t(1.0))
                .unwrap();
            let mut at = 1.0;
            for n in 2..=3 {
                at = s.tickets().get("w").unwrap().due.as_secs_f64();
                let id = format!("rw{n}");
                s.on_classify(Some(&id), "w", None, Some("standard"), 400, t(at));
                s.honest_lobby_retry_after("w", 30.0, t(at)).unwrap();
                assert!(s.tickets().get("w").unwrap().reserved.is_none());
                at += 31.0;
                s.on_aborted_at(&id, t(at));
            }
            (s, at)
        };
        let (mut s, at) = run(3);
        let w = s.tickets().get("w").unwrap();
        assert!(near(w.reserved.unwrap().as_secs_f64(), at));
        assert_eq!(s.honest.as_ref().unwrap().refusals("w"), 3);
        assert_eq!(s.tickets().held("w", s.ticket_cap()), 400);
        // x (600) fits standard's 900 but for w's 400.
        let due = w.due.as_secs_f64();
        s.on_classify(Some("rx"), "x", None, Some("standard"), 600, t(due - 1.0));
        assert!(matches!(
            check(&mut s, "x", 1000, due - 1.0),
            GateAnswer::Hold { .. }
        ));
        // w back: ahead of x, admitted; the admission uses the reservation up.
        s.on_classify(Some("rw"), "w", None, Some("standard"), 400, t(due));
        assert_eq!(check(&mut s, "w", 1000, due), GateAnswer::Go);
        s.on_dispatched("rw", "w", t(due));
        assert!(s.tickets().get("w").is_none());
        assert_eq!(s.honest.as_ref().unwrap().refusals("w"), 0);
        // Not reserved (10 refusals needed): x is not held for w's ticket.
        let (mut plain, _) = run(10);
        let w = plain.tickets().get("w").unwrap();
        assert!(w.reserved.is_none());
        let due = w.due.as_secs_f64();
        plain.on_classify(Some("rx"), "x", None, Some("standard"), 600, t(due - 1.0));
        assert_eq!(check(&mut plain, "x", 1000, due - 1.0), GateAnswer::Go);
    }

    /// (b) A reservation whose holder is not back lapses `grace_s` after its due time (not the
    /// tickets' grace) and frees its room; one whose holder waits in the lobby stays.
    #[test]
    fn f49m_b_an_unclaimed_reservation_lapses_at_due_plus_grace() {
        let make = || {
            let mut s = sched(
                |c| {
                    c.max_hold_frac = 0.0;
                    c.grace_s = 50.0;
                },
                Some(|c| {
                    c.max_refusals = 1;
                    c.reserve_frac = 0.5;
                }),
            );
            picture(&mut s, 1000, 0.0);
            s.ticket_on_throttle("w", Some("standard"), 400, 0, t(1.0))
                .unwrap();
            let w = s.tickets().get("w").unwrap();
            assert!(w.reserved.is_some());
            let (due, expires) = (w.due.as_secs_f64(), w.expires.as_secs_f64());
            assert!(near(expires - due, 30.0), "{due} {expires}");
            (s, due, expires)
        };
        let (mut s, _, expires) = make();
        s.on_classify(
            Some("rx"),
            "x",
            None,
            Some("standard"),
            600,
            t(expires - 2.0),
        );
        assert!(matches!(
            check(&mut s, "x", 1000, expires - 1.0),
            GateAnswer::Hold { .. }
        ));
        assert_eq!(check(&mut s, "x", 1000, expires + 0.5), GateAnswer::Go);
        assert!(s.tickets().get("w").is_none());
        // Back at its due time and still waiting past the lapse time: it stays.
        let (mut s, due, expires) = make();
        s.on_classify(Some("rw"), "w", None, Some("standard"), 400, t(due));
        check(&mut s, "g", 1000, expires + 5.0);
        assert!(s.tickets().is_live("w", t(expires + 5.0)));
    }

    /// (b) Reservations hold at most `reserve_frac` x HBM together: by rank, then oldest first.
    #[test]
    fn f49m_b_reservations_are_capped_at_reserve_frac() {
        let mut s = sched(
            |c| c.max_hold_frac = 0.0,
            Some(|c| {
                c.max_refusals = 1;
                c.reserve_frac = 0.4;
            }),
        );
        picture(&mut s, 1000, 0.0);
        for (i, (id, class)) in [("s1", "standard"), ("s2", "standard"), ("p1", "priority")]
            .into_iter()
            .enumerate()
        {
            s.ticket_on_throttle(id, Some(class), 300, 0, t(1.0 + i as f64))
                .unwrap();
        }
        let holds: Vec<(String, u64)> = s
            .tickets()
            .holds(s.ticket_cap())
            .into_iter()
            .map(|(k, h)| (k.session.clone(), h))
            .collect();
        assert_eq!(
            holds,
            vec![
                ("p1".to_owned(), 300),
                ("s1".to_owned(), 100),
                ("s2".to_owned(), 0)
            ]
        );
        assert_eq!(s.tickets().reserved_held(s.ticket_cap()), (400, 3));
    }

    /// (b) No inversion: a standard reservation never reduces priority's room; a priority
    /// reservation reduces standard's (and priority's own).
    #[test]
    fn f49m_b_a_lower_class_reservation_never_reduces_prioritys_room() {
        let fits = |reserved: &str, class: &str| {
            let mut s = sched(
                |c| c.max_hold_frac = 0.0,
                Some(|c| {
                    c.max_refusals = 1;
                    c.reserve_frac = 1.0;
                }),
            );
            picture(&mut s, 1000, 0.0);
            s.ticket_on_throttle("r", Some(reserved), 500, 0, t(1.0))
                .unwrap();
            assert!(s.tickets().get("r").unwrap().reserved.is_some());
            s.on_classify(Some("rn"), "n", None, Some(class), 600, t(2.0));
            check(&mut s, "n", 1000, 2.0) == GateAnswer::Go
        };
        assert!(fits("standard", "priority"));
        assert!(!fits("standard", "standard"));
        assert!(!fits("priority", "standard"));
        assert!(!fits("priority", "priority"));
    }

    /// (c) With `honest_retry`, `returnee_first` refuses a new session only when it would not fit
    /// now: around the holds, its class's waiting holders' needs and the higher classes' lobby.
    #[test]
    fn f49m_c_returnee_first_admits_a_new_session_that_fits() {
        let run = |honest: bool, old_waits: bool, need: Option<u64>| {
            let mut s = sched(
                |c| {
                    c.max_hold_frac = 0.0;
                    c.per_root = true;
                    c.keep_on_rerefusal = true;
                },
                if honest { Some(|_| {}) } else { None },
            );
            s.policy.admission.returnee_first = Some(ReturneeFirstConfig::default());
            picture(&mut s, 100_000, 0.0);
            s.ticket_issue("old", "standard", 80_000, 0, 30.0, t(1.0));
            if old_waits {
                s.on_classify(Some("ro"), "old", None, Some("standard"), 80_000, t(2.0));
            }
            s.throttle_with_parent_need("new", None, Some("standard"), need, t(3.0))
                .is_some()
        };
        // Off: refused while `old` is live, room or not.
        assert!(run(false, false, Some(20_000)));
        // On: 20k fits the 90k room with `old` away (it holds nothing at max_hold_frac 0).
        assert!(!run(true, false, Some(20_000)));
        assert!(run(true, false, Some(95_000)));
        assert!(run(true, false, None), "no need known: no check");
        // `old` waits: its 80k goes first.
        assert!(run(true, true, Some(20_000)));
        assert!(!run(true, true, Some(5_000)));
    }

    /// Off: no state, no reservation however often refused, no pool, today's advice, and
    /// nothing serialized.
    #[test]
    fn f49m_options_off_change_nothing() {
        let mut s = sched(|_| {}, None);
        assert!(s.honest.is_none() && !s.honest_on());
        for i in 0..5 {
            s.ticket_on_throttle("w", Some("standard"), 400, 0, t(i as f64));
        }
        assert!(s.tickets().get("w").unwrap().reserved.is_none());
        // f49c's forecast with no KV picture: the 600 s horizon.
        assert_eq!(
            s.ticket_on_throttle("z", Some("standard"), 400, 0, t(5.0)),
            Some(600.0)
        );
        picture(&mut s, 1000, 6.0);
        assert_eq!(s.tickets.reserve_cap, 0);
        assert_eq!(s.honest_lobby_retry_after("g", 30.0, t(6.0)), None);
        assert_eq!(
            s.honest_throttle_untracked("q", Some("standard"), t(6.0)),
            None
        );
        let json = serde_json::to_string(s.policy()).unwrap();
        assert!(!json.contains("honest_retry"));
    }
}

#[cfg(test)]
mod f49m2_tests {
    //! f49m2 `admission.honest_retry.by_reason`.
    use super::*;
    use crate::config::TicketsConfig;
    use crate::honest_retry::{HonestRetryConfig, Reason};

    fn t(s: f64) -> Time {
        Time::from_secs_f64(s)
    }

    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    /// Enforce mode, `tickets` (priority and standard listed, `retry_after: forecast`) and
    /// `honest_retry` with `by_reason`.
    fn sched(by_reason: bool) -> Scheduler {
        let mut policy = ServicePolicy {
            mode: Mode::Enforce,
            ..Default::default()
        };
        policy.admission.tickets = Some(TicketsConfig::default());
        policy.admission.honest_retry = Some(HonestRetryConfig {
            by_reason,
            ..HonestRetryConfig::default()
        });
        policy.validate().unwrap();
        Scheduler::new(policy)
    }

    fn admit(s: &mut Scheduler, id: &str, class: &str, at: f64) {
        s.on_classify(Some(id), id, None, Some(class), 100, t(at));
        s.on_dispatched(id, id, t(at));
    }

    /// The gate's KV picture (`hbm` HBM, no DRAM), from a batch session's check.
    fn picture(s: &mut Scheduler, hbm: u64, at: f64) {
        s.on_classify(Some("rg"), "g", None, Some("batch"), 1, t(at));
        check(s, "g", hbm, at);
    }

    fn check(s: &mut Scheduler, session: &str, hbm: u64, at: f64) -> GateAnswer {
        let kv = s.kv_budget(hbm, 0);
        s.gate_check(session, t(at), &kv)
    }

    fn prefill_hold() -> Decision {
        Decision::Hold {
            reason: HoldReason::PrefillBudget {
                used: 2.0,
                own: 1.0,
                budget: 2.5,
            },
            until: t(1e6),
        }
    }

    /// A session held by a non-KV term with KV room to spare gets the rate's advice (or, with too
    /// few admissions, the wait limit), never the forecast's prefill window of 30 s.
    #[test]
    fn f49m2_a_non_kv_hold_with_kv_room_is_not_advised_the_forecast() {
        let run = |by_reason: bool, admissions: bool, held: bool| {
            let mut s = sched(by_reason);
            picture(&mut s, 1000, 0.0);
            if admissions {
                for i in 0..6 {
                    admit(&mut s, &format!("a{i}"), "standard", 40.0 + 10.0 * i as f64);
                }
            }
            s.on_classify(Some("rw"), "w", None, Some("standard"), 100, t(99.0));
            if held {
                // The gate's hold of w on the prefill budget (as `gate_check_head_wake` notes it).
                s.honest_note_hold("w", &prefill_hold(), Some("prefill"));
            }
            let r = s.honest_lobby_retry_after("w", 5.0, t(100.0)).unwrap();
            let forecast = s
                .ticket_forecast("w", "standard", 100, 5.0, t(100.0))
                .clamp(5.0, 600.0);
            (s, r, forecast)
        };
        // The KV room fits w: the forecast is the prefill window (30 s) less nothing.
        let (_, off, forecast) = run(false, false, true);
        assert_eq!((off, forecast), (30.0, 30.0), "f49m: the forecast");
        // On, too few admissions: neither noted hold nor KV short: retry_after_s is unset, so
        // standard's wait limit (60 s).
        for held in [false, true] {
            let (mut s, on, _) = run(true, false, held);
            assert_eq!(on, 60.0, "{held}");
            let want = if held { "prefill" } else { "kv_fits" };
            let priced = s.honest.as_mut().unwrap().take_priced("w");
            assert_eq!(priced.map(|r| (r.name, r.kv)), Some((want, false)));
        }
        // On, six admissions in 100 s (0.06/s), w first in its lobby: N = 1, 1 / 0.06 - 5. KV
        // would have taken the forecast (25 s) as the larger.
        let (_, on, forecast) = run(true, true, true);
        assert!(near(on, 1.0 / 0.06 - 5.0), "{on}");
        assert!(forecast > on, "{forecast} {on}");
        // With retry_after_s set, too few admissions: retry_after_s.
        let mut s = sched(true);
        s.policy.admission.retry_after_s = Some(45.0);
        picture(&mut s, 1000, 0.0);
        s.on_classify(Some("rw"), "w", None, Some("standard"), 100, t(99.0));
        assert_eq!(s.honest_lobby_retry_after("w", 5.0, t(100.0)), Some(45.0));
        // The throttle path too: a recorded compute hold, too few admissions: 60 s, not 30.
        let mut s = sched(true);
        picture(&mut s, 1000, 0.0);
        let compute = Decision::Hold {
            reason: HoldReason::ComputeHeadroom {
                class: "priority".into(),
                headroom: -0.1,
            },
            until: t(1e6),
        };
        s.honest_note_hold("n", &compute, Some("compute"));
        assert_eq!(
            s.ticket_on_throttle("n", Some("standard"), 100, 100, t(1.0)),
            Some(60.0)
        );
        let mut off = sched(false);
        picture(&mut off, 1000, 0.0);
        assert_eq!(
            off.ticket_on_throttle("n", Some("standard"), 100, 100, t(1.0)),
            Some(30.0)
        );
    }

    /// A session the gate holds for KV (`KvBudget`, noted by the gate itself) gets the larger of
    /// the rate's advice and the forecast's.
    #[test]
    fn f49m2_a_kv_hold_gets_the_larger_of_rate_and_forecast() {
        let run = |by_reason: bool| {
            let mut s = sched(by_reason);
            picture(&mut s, 1000, 0.0);
            for i in 0..6 {
                admit(&mut s, &format!("a{i}"), "standard", 40.0 + 10.0 * i as f64);
            }
            s.on_classify(Some("rk"), "k", None, Some("standard"), 2000, t(100.0));
            assert!(matches!(
                check(&mut s, "k", 1000, 100.0),
                GateAnswer::Hold { .. }
            ));
            let r = s.honest_lobby_retry_after("k", 5.0, t(100.0)).unwrap();
            let need = s.sessions.total_need(s.sessions.get("k").unwrap());
            let forecast = s
                .ticket_forecast("k", "standard", need, 5.0, t(100.0))
                .clamp(5.0, 600.0);
            (s, r, forecast)
        };
        let rate = 1.0 / 0.06 - 5.0;
        let (s, on, forecast) = run(true);
        assert_eq!(
            s.honest.as_ref().unwrap().cause("k"),
            Some(Reason {
                name: "kv_budget",
                kv: true
            })
        );
        assert!(forecast > rate, "{forecast}");
        assert!(near(on, rate.max(forecast)), "{on} {forecast}");
        // f49m: the rate alone.
        let (s, off, _) = run(false);
        assert!(near(off, rate), "{off}");
        assert_eq!(
            s.honest.as_ref().unwrap().cause("k"),
            None,
            "off: nothing noted"
        );
        // Admitted: the noted hold is forgotten.
        let (mut s, _, _) = run(true);
        s.honest_note_hold("k", &Decision::Admit(crate::admit::AdmitReason::Fits), None);
        assert_eq!(s.honest.as_ref().unwrap().cause("k"), None);
    }

    /// Off (`by_reason` false or absent): f49m's advice, nothing noted, nothing serialized.
    #[test]
    fn f49m2_options_off_change_nothing() {
        let mut s = sched(false);
        assert_eq!(
            s.honest_reason("w", "w", "standard", Some(100), t(0.0)),
            None
        );
        picture(&mut s, 1000, 0.0);
        s.on_classify(Some("rw"), "w", None, Some("standard"), 100, t(1.0));
        s.honest_note_hold("w", &prefill_hold(), Some("prefill"));
        assert_eq!(s.honest.as_ref().unwrap().cause("w"), None);
        let place = s.honest_place(Some("w"), "w", "standard", t(1.0));
        let want = s.honest_advice("w", "standard", place, Some(100), 5.0, t(1.0));
        let got = s.honest_lobby_retry_after("w", 5.0, t(1.0));
        assert_eq!(got, want.map(|a| a.seconds));
        assert_eq!(s.honest.as_mut().unwrap().take_priced("w"), None);
        let json = serde_json::to_string(s.policy()).unwrap();
        assert!(
            json.contains("honest_retry") && !json.contains("by_reason"),
            "{json}"
        );
        let on = sched(true);
        assert!(
            serde_json::to_string(on.policy())
                .unwrap()
                .contains("\"by_reason\":true")
        );
    }
}

#[cfg(test)]
mod f49n_tests {
    use super::*;
    use crate::admit::GateAnswer;
    use crate::config::{LoopRebalanceConfig, TargetLoopConfig};
    use crate::sla_budget::WarmupConfig;

    fn t(s: f64) -> Time {
        Time::from_secs_f64(s)
    }

    /// The default classes (priority 0.90, standard 0.70, batch without a level), enforced, the
    /// target loop on standard (10 s steps, unsmoothed) with `rb`.
    fn policy(rb: Option<LoopRebalanceConfig>) -> ServicePolicy {
        let mut p = ServicePolicy {
            mode: Mode::Enforce,
            ..Default::default()
        };
        p.admission.target_loop = Some(TargetLoopConfig {
            smooth_s: 1.0,
            rebalance: rb,
            ..Default::default()
        });
        p
    }

    fn fast() -> LoopRebalanceConfig {
        LoopRebalanceConfig {
            release_per_min: 0.6,
            pull_per_min: 0.6,
            ..LoopRebalanceConfig::default()
        }
    }

    /// `met` of 100 completions of `class` at `at` met its level.
    fn record(s: &mut Scheduler, class: &str, at: f64, met: usize) {
        for i in 0..100 {
            s.metrics.record(
                class,
                crate::metrics::Completion {
                    at: t(at),
                    ttft_s: if i < met { 0.1 } else { 100.0 },
                    itl_s: None,
                },
            );
        }
    }

    /// Steps of 10 s from `from` to `until`, priority and standard at `p` and `st` met (of 100).
    fn run(s: &mut Scheduler, from: f64, until: f64, p: usize, st: usize) -> f64 {
        let mut at = from;
        while at < until {
            at += 10.0;
            record(s, "priority", at - 1.0, p);
            record(s, "standard", at - 1.0, st);
            s.loop_step(t(at));
        }
        at
    }

    /// A session of `class` waits in the lobby.
    fn wait(s: &mut Scheduler, class: &str, at: f64) {
        let (r, w) = (format!("r-{class}"), format!("w-{class}"));
        s.on_classify(Some(r.as_str()), &w, None, Some(class), 100, t(at));
    }

    fn both_waiting(rb: LoopRebalanceConfig) -> Scheduler {
        let mut s = Scheduler::new(policy(Some(rb)));
        wait(&mut s, "standard", 0.0);
        wait(&mut s, "batch", 0.0);
        s.loop_step(t(0.0));
        s
    }

    #[test]
    fn f49n_release_goes_to_standard_first_then_batch() {
        let mut s = both_waiting(fast());
        // Priority at 0.95 (over 0.90 + 0.02), standard at 0.60 (under its 0.70 + 0.02).
        let at = run(&mut s, 0.0, 60.0, 95, 60);
        assert!(
            s.loop_scale("standard") > 1.0,
            "{}",
            s.loop_scale("standard")
        );
        assert_eq!(s.loop_scale("batch"), 1.0);
        assert_eq!(s.lp.rb.mode, "release");
        // Standard at 0.80, over its own target + margin: it keeps its scale, batch takes.
        let at = run(&mut s, at, at + 60.0, 95, 80);
        let (std_k, batch_k) = (s.loop_scale("standard"), s.loop_scale("batch"));
        let at = run(&mut s, at, at + 60.0, 95, 80);
        assert_eq!(s.loop_scale("standard"), std_k);
        assert!(s.loop_scale("batch") > batch_k, "{batch_k}");
        // Standard short again but at its ceiling: batch takes.
        s.lp.rb.scale.insert("standard".into(), 2.5);
        let batch_k = s.loop_scale("batch");
        run(&mut s, at, at + 60.0, 95, 60);
        assert_eq!(s.loop_scale("standard"), 2.5);
        assert!(s.loop_scale("batch") > batch_k);
        // Priority's loop scale (its prefill budget share) is not touched.
        assert_eq!(s.loop_scale("priority"), 1.0);
    }

    #[test]
    fn f49n_release_needs_demand() {
        let mut s = Scheduler::new(policy(Some(fast())));
        wait(&mut s, "batch", 0.0);
        s.loop_step(t(0.0));
        // Standard short but nobody of it waits: batch, which waits, takes.
        run(&mut s, 0.0, 60.0, 95, 60);
        assert_eq!(s.loop_scale("standard"), 1.0);
        assert!(s.loop_scale("batch") > 1.0);
    }

    #[test]
    fn f49n_pull_back_takes_batch_first_then_standard() {
        let rb = LoopRebalanceConfig {
            pull_per_min: 3.0,
            ..LoopRebalanceConfig::default()
        };
        let mut s = both_waiting(rb);
        s.lp.rb.scale.insert("standard".into(), 1.5);
        s.lp.rb.scale.insert("batch".into(), 1.5);
        // Priority at 0.85: under 0.90 + 0.02.
        let mut at = 0.0;
        let mut standard_moved = false;
        for _ in 0..40 {
            at = run(&mut s, at, at + 10.0, 85, 60);
            if s.loop_scale("batch") > 0.333 {
                assert_eq!(s.loop_scale("standard"), 1.5, "{at}");
                assert_eq!(s.lp.rb.mode, "pull");
            } else if s.loop_scale("standard") < 1.5 {
                standard_moved = true;
            }
        }
        assert!(standard_moved);
        assert_eq!(s.loop_scale("batch"), 0.333);
        assert_eq!(s.loop_scale("standard"), 0.5);
        assert_eq!(s.lp.rb.mode, "floor");
        // Every class at its floor with priority under its target: the infeasible state.
        assert_eq!(s.lp.infeasible.as_deref(), Some("priority"));
    }

    #[test]
    fn f49n_no_release_while_priority_is_under_target_plus_margin() {
        let mut s = both_waiting(fast());
        // 0.91: over priority's 0.90 target, under 0.90 + 0.02.
        let mut at = 0.0;
        for _ in 0..12 {
            at = run(&mut s, at, at + 10.0, 91, 60);
            assert!(s.loop_scale("standard") <= 1.0 && s.loop_scale("batch") <= 1.0);
            assert_eq!(s.lp.rb.mode, "pull");
        }
        assert!(s.loop_scale("batch") < 1.0);
        assert_eq!(s.loop_scale("standard"), 1.0);
    }

    #[test]
    fn f49n_each_class_steers_to_its_own_target() {
        // Standard at 0.80: a taker under 0.85 + 0.02 or 0.70 + 0.15, not under 0.70 + 0.02.
        for (target, margin, standard_takes) in [
            (0.70, None, false),
            (0.85, None, true),
            (0.70, Some(0.15), true),
        ] {
            let mut rb = fast();
            if let Some(m) = margin {
                rb.margins.insert("standard".into(), m);
            }
            let mut p = policy(Some(rb));
            for c in &mut p.classes {
                if c.name == "standard" {
                    c.service_level.as_mut().unwrap().attainment = target;
                }
            }
            p.validate().unwrap();
            let mut s = Scheduler::new(p);
            wait(&mut s, "standard", 0.0);
            wait(&mut s, "batch", 0.0);
            s.loop_step(t(0.0));
            run(&mut s, 0.0, 60.0, 95, 80);
            let (st, b) = (s.loop_scale("standard"), s.loop_scale("batch"));
            if standard_takes {
                assert!(st > 1.0 && b == 1.0, "{target} {margin:?}: {st} {b}");
            } else {
                assert!(st == 1.0 && b > 1.0, "{target} {margin:?}: {st} {b}");
            }
        }
    }

    #[test]
    fn f49n_rate_caps_and_warmup_hold_every_move() {
        let rb = LoopRebalanceConfig {
            release_per_min: 1.0,
            pull_per_min: 1.0,
            max_grow_per_min: Some(0.004),
            max_shrink_per_min: Some(0.005),
            ..LoopRebalanceConfig::default()
        };
        let mut p = policy(Some(rb));
        p.admission.target_loop.as_mut().unwrap().step_s = 60.0;
        let mut s = Scheduler::new(p);
        wait(&mut s, "standard", 0.0);
        wait(&mut s, "batch", 0.0);
        s.loop_step(t(0.0));
        record(&mut s, "priority", 59.0, 99);
        record(&mut s, "standard", 59.0, 60);
        s.loop_step(t(60.0));
        assert!((s.loop_scale("standard") - 0.004_f64.exp()).abs() < 1e-12);
        record(&mut s, "priority", 119.0, 80);
        s.loop_step(t(120.0));
        assert!((s.loop_scale("batch") - (-0.005_f64).exp()).abs() < 1e-12);
        // Warm-up: nothing moves before 1200 s, then it does.
        let rb = LoopRebalanceConfig {
            warmup: Some(WarmupConfig {
                min_scored: 50,
                after_s: 1200.0,
            }),
            ..fast()
        };
        let mut s = both_waiting(rb);
        let at = run(&mut s, 0.0, 1190.0, 95, 60);
        assert_eq!(s.loop_scale("standard"), 1.0);
        assert_eq!(s.loop_scale("batch"), 1.0);
        assert_eq!(s.lp.rb.mode, "warmup");
        run(&mut s, at, at + 30.0, 95, 60);
        assert!(s.loop_scale("standard") > 1.0);
    }

    /// The hot cap's multiple of `hot_overcommit x HBM` the gate uses for `class`.
    fn cap(s: &Scheduler, class: &str) -> f64 {
        let c = s.policy().class(Some(class));
        c.admission.gate_share
            * s.pace_scale(class)
            * s.step_target_scale(class)
            * s.policy().class_overcommit(c)
            * s.sla_scale(class)
    }

    #[test]
    fn f49n_no_inversion_of_the_top_class_hot_cap() {
        for overcommit in [None, Some(2.0)] {
            let mut p = policy(Some(LoopRebalanceConfig::default()));
            if let Some(o) = overcommit {
                p.admission.class_overcommit.insert("standard".into(), o);
            }
            let mut s = Scheduler::new(p);
            for (ks, kb) in [(1.0, 1.0), (2.5, 1.0), (1.0, 2.0), (2.5, 2.0), (0.5, 0.333)] {
                s.lp.rb.scale.insert("standard".into(), ks);
                s.lp.rb.scale.insert("batch".into(), kb);
                let (pc, sc, bc) = (cap(&s, "priority"), cap(&s, "standard"), cap(&s, "batch"));
                assert!(
                    pc + 1e-9 >= sc && pc + 1e-9 >= bc,
                    "{ks} {kb}: {pc} {sc} {bc}"
                );
                assert!((pc - 1.0_f64.max(sc).max(bc)).abs() < 1e-9, "{pc}");
                let scale = |c: &crate::config::ClassPolicy| {
                    s.pace_scale(&c.name) * s.step_target_scale(&c.name) * s.sla_scale(&c.name)
                };
                let hot = |c: &str| {
                    crate::admit::hot_cap(s.policy(), s.policy().class(Some(c)), 1_000_000, &scale)
                };
                assert!(hot("priority") + 1 >= hot("standard").max(hot("batch")));
            }
            // Priority's prefill budget share is not lifted.
            assert_eq!(s.loop_scale("priority"), 1.0);
        }
        // Off: the f42 scale lets standard's cap pass priority's, as before.
        let mut off = Scheduler::new(policy(None));
        off.lp.scale = 2.0;
        assert_eq!(off.pace_scale("priority"), 1.0);
        assert!(cap(&off, "standard") > cap(&off, "priority"));
    }

    /// Standard session `a` (600 tokens) asks at 0 s and 70 s; priority `p` (500) at 71 s; HBM
    /// 1000. Answers and the decision log.
    fn scenario(p: ServicePolicy) -> (Vec<GateAnswer>, String) {
        let mut s = Scheduler::new(p);
        let kv = s.kv_budget(1_000, 100_000);
        s.on_classify(Some("r1"), "a", None, None, 600, t(0.0));
        let mut answers = vec![s.gate_check("a", t(0.0), &kv)];
        s.on_classify(Some("r2"), "b", None, None, 600, t(1.0));
        answers.push(s.gate_check("b", t(1.0), &kv));
        answers.push(s.gate_check("b", t(70.0), &kv));
        s.on_classify(Some("r3"), "p", None, Some("priority"), 500, t(71.0));
        answers.push(s.gate_check("p", t(71.0), &kv));
        answers.push(s.gate_check("b", t(140.0), &kv));
        let log = format!("{:?}", s.decisions().entries().collect::<Vec<_>>());
        (answers, log)
    }

    #[test]
    fn f49n_options_off_are_the_target_loop_unchanged() {
        // Absent by default, not serialized, no figures.
        assert_eq!(TargetLoopConfig::default().rebalance, None);
        let json = serde_json::to_string(&policy(None)).unwrap();
        assert!(json.contains("target_loop") && !json.contains("rebalance"));
        let mut off = Scheduler::new(policy(None));
        assert_eq!(off.take_loop_stats().rb, None);
        // The f42 loop: the same scale whether rebalance is on or off, and off it is the classes'.
        let mut on = Scheduler::new(policy(Some(fast())));
        for s in [&mut off, &mut on] {
            wait(s, "standard", 0.0);
            s.loop_step(t(0.0));
        }
        let mut at = 0.0;
        for (p, st) in [(95, 80), (95, 80), (85, 60), (99, 90), (91, 75)] {
            let next = run(&mut off, at, at + 20.0, p, st);
            run(&mut on, at, at + 20.0, p, st);
            at = next;
            assert_eq!(off.lp.scale, on.lp.scale);
            assert_eq!(off.loop_scale("standard"), off.lp.scale);
            assert_eq!(off.loop_scale("batch"), off.lp.scale);
            assert_eq!(off.pace_scale("priority"), 1.0);
        }
        // Gate answers and decisions: on with no step taken equals off.
        let still = |rb: Option<LoopRebalanceConfig>| {
            let mut p = policy(rb);
            p.admission.target_loop.as_mut().unwrap().step_s = 1e9;
            p
        };
        let (a_off, log_off) = scenario(still(None));
        let (a_on, log_on) = scenario(still(Some(LoopRebalanceConfig::default())));
        assert_eq!(a_off, a_on);
        assert_eq!(log_off, log_on);
        // Without the target loop at all, nothing changes either.
        let mut none = policy(None);
        none.admission.target_loop = None;
        let s = Scheduler::new(none);
        assert_eq!(s.pace_scale("priority"), 1.0);
        assert_eq!(s.pace_scale("standard"), 1.0);
    }

    #[test]
    fn f49n_the_f42_line_gains_per_class_fields() {
        let mut s = both_waiting(fast());
        run(&mut s, 0.0, 30.0, 95, 60);
        let rb = s.take_loop_stats().rb.unwrap();
        let v: serde_json::Value =
            serde_json::from_str(&format!("{{{}}}", rb.log_fields())).unwrap();
        assert_eq!(v["rb_mode"], "release");
        assert!(v["rb_head"].as_f64().unwrap() > 0.0);
        assert!(v["rb"]["standard"]["scale"].as_f64().unwrap() > 1.0);
        assert_eq!(v["rb"]["standard"]["target"].as_f64(), Some(0.7));
        assert!((v["rb"]["priority"]["att"].as_f64().unwrap() - 0.95).abs() < 1e-3);
        assert!(v["rb"]["batch"]["target"].is_null() && v["rb"]["batch"]["att"].is_null());
        // Priority's scale is its no-inversion lift over standard's cap (gate share 0.9).
        let (ps, ss) = (
            v["rb"]["priority"]["scale"].as_f64().unwrap(),
            v["rb"]["standard"]["scale"].as_f64().unwrap(),
        );
        assert!((ps - (0.9 * ss).max(1.0)).abs() < 1e-3, "{ps} {ss}");
    }

    #[test]
    fn f49n_config_parses_and_validates() {
        let p: ServicePolicy = serde_json::from_str(
            r#"{"admission": {"target_loop": {"class": "standard", "margin": 0.03, "guard": 0.02,
                "rebalance": {"margin": 0.02, "release_per_min": 0.01, "pull_per_min": 0.05,
                    "full_at": 0.05, "floors": {"standard": 0.5, "batch": 0.333},
                    "ceilings": {"standard": 2.5, "batch": 2.0}}}}}"#,
        )
        .unwrap();
        p.validate().unwrap();
        let l = p.admission.target_loop.as_ref().unwrap();
        assert_eq!(l.rebalance, Some(LoopRebalanceConfig::default()));
        let p: ServicePolicy = serde_json::from_str(
            r#"{"admission": {"target_loop": {"rebalance": {"max_shrink_per_min": 0.02,
                "warmup": {"min_scored": 20, "after_s": 600}}}}}"#,
        )
        .unwrap();
        p.validate().unwrap();
        assert!(
            serde_json::from_str::<ServicePolicy>(
                r#"{"admission": {"target_loop": {"rebalance": {"gain": 1}}}}"#
            )
            .is_err()
        );
        type Edit = fn(&mut LoopRebalanceConfig);
        let check = |f: Edit, class: &str| {
            let mut rb = LoopRebalanceConfig::default();
            f(&mut rb);
            let mut p = policy(Some(rb));
            p.admission.target_loop.as_mut().unwrap().class = class.to_owned();
            p.validate().map_err(|e| e.to_string())
        };
        check(|_| {}, "standard").unwrap();
        let cases: [(Edit, &str); 7] = [
            (
                |r| {
                    r.floors.insert("priority".into(), 0.5);
                },
                "ranked above",
            ),
            (|r| r.full_at = 0.0, "full_at"),
            (|r| r.pull_per_min = -1.0, "pull_per_min"),
            (
                |r| {
                    r.ceilings.insert("standard".into(), 0.9);
                },
                "ceilings.standard",
            ),
            (
                |r| {
                    r.floors.insert("nope".into(), 0.5);
                },
                "not a configured class",
            ),
            (|r| r.max_shrink_per_min = Some(0.0), "max_shrink_per_min"),
            (
                |r| {
                    r.warmup = Some(WarmupConfig {
                        min_scored: 0,
                        after_s: 1.0,
                    })
                },
                "min_scored",
            ),
        ];
        for (bad, want) in cases {
            let err = check(bad, "standard").unwrap_err();
            assert!(
                err.starts_with("admission.target_loop.rebalance: "),
                "{err}"
            );
            assert!(err.contains(want), "{err}");
        }
        // The loop on the top class has no class above it to rebalance for.
        let err = check(
            |r| {
                r.floors.clear();
                r.ceilings.clear();
            },
            "priority",
        )
        .unwrap_err();
        assert!(err.contains("needs a class with a service_level"), "{err}");
    }

    /// f49q4: `fast()` with `release_order: demand_first`.
    fn demand_first() -> LoopRebalanceConfig {
        LoopRebalanceConfig {
            release_order: crate::config::ReleaseOrder::DemandFirst,
            ..fast()
        }
    }

    /// f49q4: a scheduler on `p` with `classes` waiting and one step taken at 0 s.
    fn waiting(p: ServicePolicy, classes: &[&str]) -> Scheduler {
        let mut s = Scheduler::new(p);
        for c in classes {
            wait(&mut s, c, 0.0);
        }
        s.loop_step(t(0.0));
        s
    }

    #[test]
    fn f49q4_demand_first_releases_to_standard_first_whatever_its_attainment() {
        let mut s = waiting(policy(Some(demand_first())), &["standard", "batch"]);
        // Standard at 0.80, over its own 0.70 + 0.02: it still takes while it has demand.
        let mut at = 0.0;
        for _ in 0..6 {
            at = run(&mut s, at, at + 10.0, 95, 80);
            assert_eq!(s.lp.rb.mode, "release");
            assert_eq!(s.loop_scale("batch"), 1.0);
        }
        assert!(
            s.loop_scale("standard") > 1.0,
            "{}",
            s.loop_scale("standard")
        );
        // f49n's rule on the same run: batch takes instead.
        let mut f49n = waiting(policy(Some(fast())), &["standard", "batch"]);
        run(&mut f49n, 0.0, 60.0, 95, 80);
        assert_eq!(f49n.loop_scale("standard"), 1.0);
        assert!(f49n.loop_scale("batch") > 1.0);
        // Standard at its ceiling: batch takes, up to standard's scale at most.
        s.lp.rb.scale.insert("standard".into(), 2.5);
        let batch_k = s.loop_scale("batch");
        run(&mut s, at, at + 60.0, 95, 80);
        assert_eq!(s.loop_scale("standard"), 2.5);
        assert!(s.loop_scale("batch") > batch_k);
        // Standard without demand: batch takes, standard keeps its scale.
        let mut s = waiting(policy(Some(demand_first())), &["batch"]);
        s.lp.rb.scale.insert("standard".into(), 1.5);
        let mut at = 0.0;
        for _ in 0..30 {
            at = run(&mut s, at, at + 10.0, 95, 60);
            assert_eq!(s.loop_scale("standard"), 1.5);
            assert!(s.loop_scale("batch") <= 1.5);
        }
        assert_eq!(s.loop_scale("batch"), 1.5);
        // Batch at standard's scale counts as at its ceiling: nothing to release.
        assert_eq!(s.lp.rb.mode, "hold");
    }

    #[test]
    fn f49q4_demand_first_keeps_batch_at_most_standard() {
        let under = |s: &Scheduler| {
            let k = |c: &str| s.lp.rb.scale.get(c).copied().unwrap_or(1.0);
            k("batch") <= k("standard") && s.loop_scale("batch") <= s.loop_scale("standard")
        };
        // At start.
        let mut s = waiting(policy(Some(demand_first())), &["standard", "batch"]);
        assert!(under(&s));
        // An inverted state is set right at the next step.
        s.lp.rb.scale.insert("batch".into(), 2.0);
        assert!(s.loop_scale("batch") <= s.loop_scale("standard"));
        let mut at = run(&mut s, 0.0, 10.0, 95, 60);
        assert!(under(&s));
        // Releases, standard's and then batch's.
        for _ in 0..60 {
            at = run(&mut s, at, at + 10.0, 95, 60);
            assert!(under(&s), "{at}");
        }
        assert_eq!(s.loop_scale("standard"), 2.5);
        assert_eq!(s.loop_scale("batch"), 2.0);
        // A pull back of standard below batch's floor takes batch down with it.
        let mut rb = demand_first();
        rb.pull_per_min = 3.0;
        rb.floors.insert("batch".into(), 0.8);
        let mut s = waiting(policy(Some(rb)), &["standard", "batch"]);
        s.lp.rb.scale.insert("standard".into(), 1.5);
        s.lp.rb.scale.insert("batch".into(), 1.5);
        let mut at = 0.0;
        for _ in 0..40 {
            at = run(&mut s, at, at + 10.0, 85, 60);
            assert!(under(&s), "{at}");
        }
        assert_eq!(s.loop_scale("standard"), 0.5);
        assert_eq!(s.loop_scale("batch"), 0.5);
        assert_eq!(s.lp.rb.mode, "floor");
    }

    #[test]
    fn f49q4_demand_first_never_lifts_the_top_class() {
        for overcommit in [None, Some(2.0)] {
            let mut p = policy(Some(demand_first()));
            if let Some(o) = overcommit {
                p.admission.class_overcommit.insert("standard".into(), o);
            }
            let mut s = Scheduler::new(p);
            for (ks, kb) in [(1.0, 1.0), (2.5, 1.0), (1.0, 2.0), (2.5, 2.0), (0.5, 0.333)] {
                s.lp.rb.scale.insert("standard".into(), ks);
                s.lp.rb.scale.insert("batch".into(), kb);
                assert_eq!(s.pace_scale("priority"), 1.0, "{ks} {kb}");
                assert_eq!(s.loop_scale("priority"), 1.0, "{ks} {kb}");
                assert!(
                    s.loop_scale("batch") <= s.loop_scale("standard"),
                    "{ks} {kb}"
                );
            }
        }
        // A run of releases takes standard to its ceiling (its hot cap over priority's, which
        // stays at its configured share), then batch to its own, under standard's.
        let mut s = waiting(policy(Some(demand_first())), &["standard", "batch"]);
        let mut at = 0.0;
        for _ in 0..60 {
            at = run(&mut s, at, at + 10.0, 95, 60);
            assert_eq!(s.pace_scale("priority"), 1.0, "{at}");
        }
        assert_eq!(s.loop_scale("standard"), 2.5);
        assert_eq!(s.loop_scale("batch"), 2.0);
        assert!(cap(&s, "standard") > cap(&s, "priority"));
        let rb = s.take_loop_stats().rb.unwrap();
        let v: serde_json::Value =
            serde_json::from_str(&format!("{{{}}}", rb.log_fields())).unwrap();
        assert_eq!(v["rb"]["priority"]["scale"].as_f64(), Some(1.0));
        assert_eq!(v["rb"]["standard"]["scale"].as_f64(), Some(2.5));
    }

    #[test]
    fn f49q4_own_target_is_f49n_unchanged() {
        // Absent and `own_target` are the same config, not serialized.
        let own = LoopRebalanceConfig {
            release_order: crate::config::ReleaseOrder::OwnTarget,
            ..fast()
        };
        assert_eq!(own, fast());
        let json = serde_json::to_string(&policy(Some(fast()))).unwrap();
        assert!(json.contains("rebalance") && !json.contains("release_order"));
        // f49n_release_goes_to_standard_first_then_batch's sequence, explicitly `own_target`.
        let mut s = both_waiting(own);
        let at = run(&mut s, 0.0, 60.0, 95, 60);
        assert!(s.loop_scale("standard") > 1.0);
        assert_eq!(s.loop_scale("batch"), 1.0);
        assert_eq!(s.lp.rb.mode, "release");
        let at = run(&mut s, at, at + 60.0, 95, 80);
        let (std_k, batch_k) = (s.loop_scale("standard"), s.loop_scale("batch"));
        let at = run(&mut s, at, at + 60.0, 95, 80);
        assert_eq!(s.loop_scale("standard"), std_k);
        assert!(s.loop_scale("batch") > batch_k, "{batch_k}");
        s.lp.rb.scale.insert("standard".into(), 2.5);
        let batch_k = s.loop_scale("batch");
        run(&mut s, at, at + 60.0, 95, 60);
        assert_eq!(s.loop_scale("standard"), 2.5);
        assert!(s.loop_scale("batch") > batch_k);
        // The no-inversion lift is still there.
        assert!(s.pace_scale("priority") > 1.0);
        // Step by step, absent and `own_target` take the same scales and modes.
        let mut a = both_waiting(fast());
        let mut b = both_waiting(LoopRebalanceConfig {
            release_order: crate::config::ReleaseOrder::OwnTarget,
            ..fast()
        });
        let mut at = 0.0;
        for (p, st) in [(95, 60), (95, 80), (85, 60), (99, 90), (91, 75), (95, 60)] {
            let next = run(&mut a, at, at + 30.0, p, st);
            run(&mut b, at, at + 30.0, p, st);
            at = next;
            for c in ["priority", "standard", "batch"] {
                assert_eq!(a.loop_scale(c), b.loop_scale(c));
                assert_eq!(a.pace_scale(c), b.pace_scale(c));
            }
            assert_eq!(a.lp.rb.mode, b.lp.rb.mode);
        }
    }

    #[test]
    fn f49q4_release_order_parses_and_validates() {
        let parse = |order: &str| {
            serde_json::from_str::<ServicePolicy>(&format!(
                r#"{{"admission": {{"target_loop": {{"class": "standard",
                    "rebalance": {{"release_order": {order}}}}}}}}}"#
            ))
        };
        let order = |p: &ServicePolicy| {
            let l = p.admission.target_loop.as_ref().unwrap();
            l.rebalance.as_ref().unwrap().release_order
        };
        let p = parse("\"demand_first\"").unwrap();
        p.validate().unwrap();
        assert_eq!(order(&p), crate::config::ReleaseOrder::DemandFirst);
        assert!(
            serde_json::to_string(&p)
                .unwrap()
                .contains("\"release_order\":\"demand_first\"")
        );
        let p = parse("\"own_target\"").unwrap();
        p.validate().unwrap();
        assert_eq!(order(&p), crate::config::ReleaseOrder::OwnTarget);
        let p: ServicePolicy =
            serde_json::from_str(r#"{"admission": {"target_loop": {"rebalance": {}}}}"#).unwrap();
        assert_eq!(order(&p), crate::config::ReleaseOrder::OwnTarget);
        for bad in ["\"nope\"", "\"DemandFirst\""] {
            let err = parse(bad).unwrap_err().to_string();
            assert!(err.contains("unknown variant"), "{bad}: {err}");
        }
        assert!(parse("1").is_err());
    }
}

/// f49q: the session signal ([`crate::session_signal`]): `admission.starvation_override` and
/// `admission.lobby_order: signal`.
impl Scheduler {
    /// f49q: whether the signal is `slowdown` with either use on (it needs the step model).
    pub fn uses_slowdown(&self) -> bool {
        self.signals.is_some()
            && self.signal_cfg.signal == crate::session_signal::SignalKind::Slowdown
    }

    /// f49q: the fleet's worker count, the override bucket's size.
    pub fn set_fleet_workers(&mut self, workers: u64) {
        if let Some(signals) = self.signals.as_mut() {
            signals.set_workers(workers);
        }
    }

    /// f49q: the session's signal at `now` (`None`: not in the table).
    pub fn session_reading(
        &self,
        session: &str,
        now: Time,
    ) -> Option<crate::session_signal::Reading> {
        use crate::session_signal::{Reading, SignalKind};
        let s = self.sessions.get(session)?;
        Some(match self.signal_cfg.signal {
            SignalKind::HeldTime => Reading::HeldS(s.waited_s + s.lobby_wait_s(now)),
            SignalKind::SessionAttainment => {
                let a = self.experience.get(&self.root_of(session));
                Reading::Attainment {
                    on_target: a.map_or(0, |a| a.on_target),
                    judged: a.map_or(0, |a| a.judged),
                }
            }
            SignalKind::Slowdown => Reading::Slowdown(self.session_slowdown(session, now)),
        })
    }

    /// f49q: the session's (its root's) active time over its standalone service, its pending turn
    /// included; `None` without the step model.
    pub fn session_slowdown(&self, session: &str, now: Time) -> Option<f64> {
        let model = self.step.as_ref()?;
        let s = self.sessions.get(session)?;
        let account = self.experience.get(&self.root_of(session))?;
        let pending = if s.state == SessionState::Lobby || s.in_flight > 0 {
            let osl = s
                .signals
                .osl_tokens
                .map(u64::from)
                .or_else(|| self.signals.as_ref().and_then(|g| g.mean_osl(&s.class)))
                .unwrap_or(0);
            let prefix = if s.turns_served > 0 {
                s.context_tokens
            } else {
                0
            };
            crate::session_signal::expected_turn_s(
                model.0.as_ref(),
                self.chunk_tokens,
                s.pending_isl,
                prefix,
                osl,
            )
            .unwrap_or(0.0)
        } else {
            0.0
        };
        crate::session_signal::slowdown(account.active_s(now), account.service_s + pending)
    }

    /// f49q `signal: slowdown`: a completed request adds its standalone time to its root's service
    /// (`uncontended` when the step model priced it; at the class's mean output otherwise).
    fn signal_note_service(
        &mut self,
        record: &RequestRecord,
        output_tokens: Option<usize>,
        uncontended: Option<f64>,
    ) {
        if self.signal_cfg.signal != crate::session_signal::SignalKind::Slowdown {
            return;
        }
        let Some(signals) = self.signals.as_mut() else {
            return;
        };
        if let Some(out) = output_tokens {
            signals.note_output(&record.class, out as u64);
        }
        let mean = signals.mean_osl(&record.class).unwrap_or(0);
        let service = uncontended.or_else(|| {
            let m = self.step.as_ref()?;
            let prompt = record.prompt.unwrap_or(record.isl).max(record.isl);
            let cached = (prompt - record.isl).max(record.prev_context);
            crate::uncontended::request_s(m.0.as_ref(), self.chunk_tokens, prompt, cached, mean)
        });
        if let Some(service) = service {
            self.experience.add_service(&record.root, service);
        }
    }

    /// f49q: whether `lobby_order: signal` orders `class`'s lobby.
    fn signal_orders(&self, class: &str) -> bool {
        self.signals.is_some()
            && self.policy.admission.lobby_order == LobbyOrder::Signal
            && self.signal_cfg.order_classes.iter().any(|c| c == class)
    }

    /// f49q: set a waiting session's lobby key from its signal (only when it changes).
    fn set_signal_key(&mut self, session: &str, now: Time) {
        let Some(reading) = self.session_reading(session, now) else {
            return;
        };
        let key = self.signal_cfg.order_key(reading);
        if self
            .sessions
            .get(session)
            .is_some_and(|s| s.order_key != Some(key))
        {
            self.sessions.update(session, |s| s.order_key = Some(key));
        }
    }

    /// f49q, at each gate check: the override log's minute, and with `lobby_order: signal` the
    /// keys of every waiting session of an ordered class, at most every
    /// [`crate::session_signal::REFRESH_S`]. Nothing when off.
    fn signal_step(&mut self, now: Time) {
        let Some(signals) = self.signals.as_mut() else {
            return;
        };
        signals.roll(now);
        signals.flush(&mut self.tickets);
        if self.policy.admission.lobby_order != LobbyOrder::Signal || !signals.refresh_due(now) {
            return;
        }
        let ids: Vec<String> = self
            .sessions
            .lobby()
            .ordered()
            .into_iter()
            .filter(|(class, _, _)| {
                self.signal_cfg
                    .order_classes
                    .iter()
                    .any(|c| c.as_str() == *class)
            })
            .map(|(_, key, _)| key.session.clone())
            .collect();
        for id in ids {
            self.set_signal_key(&id, now);
        }
    }

    /// f49q `admission.starvation_override`: a starved waiting session of a listed class is
    /// admitted whatever its class's own budgets say, while some worker has a free sequence slot,
    /// it fits [`Self::starvation_fits`] and the bucket has an override left.
    #[allow(clippy::too_many_arguments)]
    fn starvation_override(
        &mut self,
        session: &str,
        class: &str,
        kv: &KvBudget,
        held_kv: u64,
        seq_room: Option<bool>,
        now: Time,
    ) -> Option<Decision> {
        let cfg = self.policy.admission.starvation_override.as_ref()?;
        if seq_room == Some(false) || !cfg.classes.iter().any(|c| c == class) {
            return None;
        }
        let rate = cfg.per_worker_per_min;
        let entry = self
            .sessions
            .get(session)
            .filter(|s| s.state == SessionState::Lobby)?;
        let reading = self.session_reading(session, now)?;
        let me = self.policy.class(Some(class));
        let target = me.service_level.as_ref().map(|l| l.attainment);
        if !self.signal_cfg.starved(reading, target)
            || !self.starvation_fits(entry, me, kv, held_kv)
        {
            return None;
        }
        let signal = self.signal_cfg.signal;
        let signals = self.signals.as_mut()?;
        let admitted = signals.take(now, rate);
        if admitted {
            signals.record(now, session, class, signal, reading.value());
        } else {
            signals.denied(now);
        }
        signals.flush(&mut self.tickets);
        admitted.then_some(Decision::Admit(crate::admit::AdmitReason::Starved))
    }

    /// f49q: whether a starved session fits what the classes ranked above it keep: the total
    /// budget and the top class's hot budget (at its own scale; skipped without the hot budget or
    /// for a class it leaves out), less the input of the waiting sessions ranked above it and the
    /// KV the live tickets hold from it. Its own class's share and scales do not count.
    fn starvation_fits(
        &self,
        entry: &crate::session::Session,
        me: &crate::config::ClassPolicy,
        kv: &KvBudget,
        held_kv: u64,
    ) -> bool {
        let policy = &self.policy;
        let reserved = self
            .sessions
            .lobby()
            .isl_ranked_above(me.rank)
            .saturating_add(held_kv);
        let total_cap =
            (policy.admission.total_headroom * (kv.hbm_tokens + kv.dram_tokens) as f64) as u64;
        let total_room = total_cap
            .saturating_sub(kv.total_tokens)
            .saturating_sub(reserved);
        if self.sessions.total_need(entry) > total_room {
            return false;
        }
        if !policy.admission.hot_budget.on()
            || policy.admission.hot_exempt_classes.contains(&me.name)
        {
            return true;
        }
        let top = policy.classes.iter().min_by_key(|c| c.rank).unwrap_or(me);
        // Every class at its own scale ("" is no class's name).
        let scale = |c: &crate::config::ClassPolicy| self.class_scale(c, "", 1.0);
        let cap = crate::admit::hot_cap(policy, top, kv.hbm_tokens, &scale);
        let hot_room = cap.saturating_sub(kv.hot_tokens).saturating_sub(reserved);
        crate::admit::hot_charge(policy, entry) <= hot_room
    }
}

#[cfg(test)]
mod f49q_tests {
    use super::*;
    use crate::admit::AdmitReason;
    use crate::session_signal::{
        Reading, SessionSignalConfig, SignalKind, StarvationOverrideConfig,
    };

    fn t(s: f64) -> Time {
        Time::from_secs_f64(s)
    }

    /// Prefill 1 s per 1,000 new tokens, decode 10 ms a token: 1,000 new input tokens and 101
    /// output tokens take 2 s alone.
    fn unit_steps() -> std::sync::Arc<dyn crate::forecast::StepModel> {
        std::sync::Arc::new(crate::forecast::TableStepModel {
            prefill_tokens: vec![0.0, 1000.0],
            prefill_s: vec![0.0, 1.0],
            prefix_weight: 0.0,
            decode_batch: vec![1.0, 2.0],
            decode_context: vec![1000.0, 2000.0],
            decode_s: vec![0.01, 0.01, 0.01, 0.01],
        })
    }

    /// The default classes, enforced, with wait limits out of the way, then `f`.
    fn sched(f: impl FnOnce(&mut ServicePolicy)) -> Scheduler {
        let mut p = ServicePolicy {
            mode: Mode::Enforce,
            ..Default::default()
        };
        for c in &mut p.classes {
            c.admission.max_lobby_wait_s = 100_000.0;
        }
        f(&mut p);
        p.validate().unwrap();
        let mut s = Scheduler::new(p);
        s.set_step_model(unit_steps());
        s.set_chunk_tokens(8192);
        s
    }

    fn with_signal(p: &mut ServicePolicy, signal: SignalKind) {
        p.admission.session_signal = Some(SessionSignalConfig {
            signal,
            ..Default::default()
        });
    }

    fn arrive(s: &mut Scheduler, request: &str, session: &str, class: &str, isl: u64, at: f64) {
        s.on_classify_with(
            Some(request),
            session,
            None,
            Some(class),
            &RequestAttributes::default(),
            isl,
            t(at),
        );
    }

    fn held(answer: GateAnswer) -> bool {
        matches!(answer, GateAnswer::Hold { .. })
    }

    /// The latest logged decision for `session`.
    fn last(s: &Scheduler, session: &str) -> Decision {
        s.decisions()
            .entries()
            .filter(|d| d.session == session)
            .last()
            .unwrap()
            .decision
            .clone()
    }

    /// The waiting sessions in service order.
    fn order(s: &Scheduler) -> Vec<String> {
        s.sessions()
            .lobby()
            .ordered()
            .into_iter()
            .map(|(_, k, _)| k.session.clone())
            .collect()
    }

    #[test]
    fn f49q_slowdown_excludes_client_gaps_and_prices_with_the_step_model() {
        let mut s = sched(|p| {
            with_signal(p, SignalKind::Slowdown);
            p.admission.lobby_order = LobbyOrder::Signal;
        });
        assert!(s.uses_slowdown());
        let sd = |s: &Scheduler, at: f64| match s.session_reading("a", t(at)) {
            Some(Reading::Slowdown(Some(v))) => v,
            r => panic!("{r:?}"),
        };
        // a1: 2 s alone, served from 0 to 4.
        arrive(&mut s, "a1", "a", "standard", 1000, 0.0);
        s.on_dispatched("a1", "a", t(0.0));
        s.on_completed_with_output("a1", Some(0), Some(101), t(4.0));
        // At 14, idle since 4: the open gap is client time. 4 s active over 2 s alone.
        assert!((sd(&s, 14.0) - 2.0).abs() < 1e-9, "{}", sd(&s, 14.0));
        // a2 at 14, no OSL hint: priced at the class's mean output (101), 2 s alone. At 20: 4 + 6
        // active (the 10 s gap excluded) over 2 + 2.
        arrive(&mut s, "a2", "a", "standard", 1000, 14.0);
        s.on_dispatched("a2", "a", t(14.0));
        assert!((sd(&s, 20.0) - 2.5).abs() < 1e-9, "{}", sd(&s, 20.0));
        // Without a step model the slowdown is unknown.
        let mut bare = Scheduler::new(s.policy().clone());
        arrive(&mut bare, "b1", "b", "standard", 1000, 0.0);
        assert_eq!(
            bare.session_reading("b", t(5.0)),
            Some(Reading::Slowdown(None))
        );
        // The other signals do not need it.
        let off = sched(|_| {});
        assert!(!off.uses_slowdown());
    }

    #[test]
    fn f49q_default_signal_is_held_time_and_the_override_admits_past_held_s() {
        let mut s =
            sched(|p| p.admission.starvation_override = Some(StarvationOverrideConfig::default()));
        assert_eq!(s.signal_cfg.signal, SignalKind::HeldTime);
        assert!(!s.uses_slowdown());
        arrive(&mut s, "n1", "n", "standard", 1000, 0.0);
        // Standard's hot budget (0.9 x 1,000) holds it; the total budget and priority's hot
        // budget (all 1,000) have room.
        let kv = s.kv_budget(1000, 100_000);
        assert!(held(s.gate_check("n", t(100.0), &kv)));
        assert_eq!(
            s.session_reading("n", t(100.0)),
            Some(Reading::HeldS(100.0))
        );
        assert!(held(s.gate_check("n", t(300.0), &kv)));
        assert_eq!(s.gate_check("n", t(301.0), &kv), GateAnswer::Go);
        assert_eq!(last(&s, "n"), Decision::Admit(AdmitReason::Starved));
    }

    #[test]
    fn f49q_the_override_needs_the_total_budget_a_listed_class_and_a_free_sequence() {
        let on = |classes: &[&str]| {
            sched(|p| {
                p.admission.starvation_override = Some(StarvationOverrideConfig {
                    classes: classes.iter().map(|c| (*c).to_owned()).collect(),
                    ..Default::default()
                })
            })
        };
        // The total budget (0.95 x 1,000) has no room for 1,000 tokens.
        let mut s = on(&["standard"]);
        arrive(&mut s, "n1", "n", "standard", 1000, 0.0);
        let kv = s.kv_budget(1000, 0);
        assert!(held(s.gate_check("n", t(400.0), &kv)));
        // Not a listed class.
        let mut s = on(&["batch"]);
        arrive(&mut s, "n1", "n", "standard", 1000, 0.0);
        let kv = s.kv_budget(1000, 100_000);
        assert!(held(s.gate_check("n", t(400.0), &kv)));
        // Every worker at its max_num_seqs.
        let mut s = on(&["standard"]);
        arrive(&mut s, "n1", "n", "standard", 1000, 0.0);
        let full = HeadFacts {
            seq_room: Some(false),
            ..HeadFacts::default()
        };
        assert!(held(s.gate_check_head("n", t(400.0), &kv, full)));
        let free = HeadFacts {
            seq_room: Some(true),
            ..HeadFacts::default()
        };
        assert_eq!(s.gate_check_head("n", t(400.0), &kv, free), GateAnswer::Go);
    }

    #[test]
    fn f49q_the_override_is_rate_limited_per_worker_per_minute() {
        let mut s = sched(|p| {
            p.admission.starvation_override = Some(StarvationOverrideConfig {
                per_worker_per_min: 2.0,
                ..Default::default()
            })
        });
        s.set_fleet_workers(1);
        for i in 0..3 {
            arrive(
                &mut s,
                &format!("r{i}"),
                &format!("n{i}"),
                "standard",
                1000,
                0.0,
            );
        }
        let kv = s.kv_budget(1000, 100_000);
        assert_eq!(s.gate_check("n0", t(301.0), &kv), GateAnswer::Go);
        // Behind n0 in its class: the override passes the class's order too.
        assert_eq!(s.gate_check("n1", t(301.0), &kv), GateAnswer::Go);
        // Two a minute: the third waits for the bucket (one more after 30 s).
        assert!(held(s.gate_check("n2", t(302.0), &kv)));
        assert!(held(s.gate_check("n2", t(330.0), &kv)));
        assert_eq!(s.gate_check("n2", t(332.0), &kv), GateAnswer::Go);
        assert_eq!(
            s.signals
                .as_ref()
                .unwrap()
                .minute_overrides()
                .get("standard"),
            Some(&3)
        );
    }

    #[test]
    fn f49q_the_override_never_takes_what_priority_waits_for() {
        let run = |priority_waiting: bool| {
            let mut s = sched(|p| {
                p.admission.starvation_override = Some(StarvationOverrideConfig::default())
            });
            if priority_waiting {
                arrive(&mut s, "p1", "p", "priority", 500, 0.0);
            }
            arrive(&mut s, "n1", "n", "standard", 1000, 0.0);
            // Standard's hot budget is 900 of the 1,000; priority's is all 1,000, and the total
            // budget has room for both.
            let kv = s.kv_budget(1000, 1000);
            s.gate_check("n", t(400.0), &kv)
        };
        assert_eq!(run(false), GateAnswer::Go);
        // Priority waits with 500 tokens: 500 of priority's hot budget left, under 1,000.
        assert!(held(run(true)));
    }

    #[test]
    fn f49q_held_time_orders_standard_and_batch_longest_held_first_and_leaves_priority_fifo() {
        let mut s = sched(|p| p.admission.lobby_order = LobbyOrder::Signal);
        for (session, class, at) in [
            ("p1", "priority", 0.0),
            ("p2", "priority", 5.0),
            ("a", "standard", 0.0),
            ("b", "standard", 10.0),
            ("x", "batch", 0.0),
            ("y", "batch", 10.0),
        ] {
            arrive(&mut s, &format!("{session}-r"), session, class, 2000, at);
        }
        assert_eq!(order(&s), ["p1", "p2", "a", "b", "x", "y"]);
        // b, y and p2 waited before (500 s): only standard and batch reorder, at the next refresh.
        for id in ["b", "y", "p2"] {
            s.sessions.update(id, |x| x.waited_s = 500.0);
        }
        let kv = s.kv_budget(500, 0);
        s.gate_check("a", t(20.0), &kv);
        assert_eq!(order(&s), ["p1", "p2", "b", "a", "y", "x"]);
        assert_eq!(s.sessions().get("p2").unwrap().order_key, None);
        // The incremental lobby agrees with a full sort of it.
        let mut scan = s.lobby_scan();
        crate::admit::order_lobby(s.policy(), &mut scan);
        let scanned: Vec<String> = scan.into_iter().map(|e| e.session).collect();
        assert_eq!(scanned, order(&s));
        // Keys move only every 10 s.
        s.sessions.update("a", |x| x.waited_s = 5000.0);
        s.gate_check("a", t(25.0), &kv);
        assert_eq!(order(&s)[2], "b");
        s.gate_check("a", t(30.0), &kv);
        assert_eq!(order(&s)[2], "a");
        // Off: arrival order whatever the waits.
        let mut fifo = sched(|_| {});
        arrive(&mut fifo, "a-r", "a", "standard", 2000, 0.0);
        arrive(&mut fifo, "b-r", "b", "standard", 2000, 10.0);
        fifo.sessions.update("b", |x| x.waited_s = 500.0);
        fifo.gate_check("a", t(20.0), &kv);
        assert_eq!(order(&fifo), ["a", "b"]);
    }

    #[test]
    fn f49q_slowdown_orders_highest_first_as_it_grows() {
        let mut s = sched(|p| {
            with_signal(p, SignalKind::Slowdown);
            p.admission.lobby_order = LobbyOrder::Signal;
        });
        // a: 10 s alone; b, later, 0.1 s alone.
        arrive(&mut s, "a-r", "a", "standard", 10_000, 0.0);
        arrive(&mut s, "b-r", "b", "standard", 100, 0.5);
        assert_eq!(order(&s), ["a", "b"]);
        // At 20: a 20 / 10 = 2, b 19.5 / 0.1 = 195.
        let kv = s.kv_budget(500, 0);
        s.gate_check("a", t(20.0), &kv);
        assert_eq!(order(&s), ["b", "a"]);
    }

    #[test]
    fn f49q_session_attainment_orders_lowest_first_and_overrides_under_target_less_gap() {
        let mut s = sched(|p| {
            with_signal(p, SignalKind::SessionAttainment);
            p.admission.lobby_order = LobbyOrder::Signal;
            p.admission.starvation_override = Some(StarvationOverrideConfig::default());
        });
        // x: three turns, each 20 s to complete (over standard's 8 s TTFT).
        for i in 0..3 {
            let at = f64::from(i) * 30.0;
            let r = format!("x{i}");
            arrive(&mut s, &r, "x", "standard", 1000, at);
            s.on_dispatched(&r, "x", t(at));
            s.on_completed_with_output(&r, Some(0), Some(101), t(at + 20.0));
        }
        assert_eq!(
            s.session_reading("x", t(100.0)),
            Some(Reading::Attainment {
                on_target: 0,
                judged: 3
            })
        );
        // x rotated out, back in the lobby to resume; n, new, waited longer.
        arrive(&mut s, "n0", "n", "standard", 1000, 90.0);
        s.sessions.update("x", |x| {
            x.state = SessionState::Paused;
            x.paused_at = Some(t(95.0));
        });
        arrive(&mut s, "x3", "x", "standard", 1000, 100.0);
        assert_eq!(s.sessions().get("x").unwrap().state, SessionState::Lobby);
        // Lowest attainment first (no judged turn counts as 1).
        assert_eq!(order(&s), ["x", "n"]);
        let kv = s.kv_budget(1000, 100_000);
        assert_eq!(s.gate_check("x", t(101.0), &kv), GateAnswer::Go);
        assert_eq!(last(&s, "x"), Decision::Admit(AdmitReason::Starved));
        // n has no judged turn: never starved by attainment.
        assert!(held(s.gate_check("n", t(5000.0), &kv)));
    }

    #[test]
    fn f49q_off_answers_and_decision_logs_are_unchanged() {
        // Off, and on for a class none of these sessions has: the same answers and log.
        let script = |s: &mut Scheduler| {
            for i in 0..3 {
                arrive(
                    s,
                    &format!("r{i}"),
                    &format!("n{i}"),
                    "standard",
                    1000,
                    f64::from(i),
                );
            }
            let kv = s.kv_budget(1000, 100_000);
            let mut answers = Vec::new();
            for at in [10.0, 301.0, 400.0, 1000.0] {
                for i in 0..3 {
                    answers.push(s.gate_check(&format!("n{i}"), t(at), &kv));
                }
            }
            let log: Vec<String> = s.decisions().entries().map(|d| format!("{d:?}")).collect();
            (answers, log)
        };
        let mut off = sched(|_| {});
        assert!(off.signals.is_none());
        let mut inert = sched(|p| {
            p.admission.starvation_override = Some(StarvationOverrideConfig {
                classes: vec!["batch".to_owned()],
                ..Default::default()
            })
        });
        let (a, la) = script(&mut off);
        let (b, lb) = script(&mut inert);
        assert_eq!(a, b);
        assert_eq!(la, lb);
        assert!(a.iter().all(|x| held(*x)));
        assert!(off.sessions().iter().all(|s| s.order_key.is_none()));
        // The policy writes nothing new when off.
        let json = serde_json::to_string(&ServicePolicy::default()).unwrap();
        assert!(!json.contains("starvation_override"), "{json}");
        assert!(!json.contains("session_signal"), "{json}");
        assert!(json.contains("\"lobby_order\":\"fifo\""), "{json}");
    }

    #[test]
    fn f49q_policy_files_and_validation() {
        for (signal, extra) in [
            ("held_time", r#""held_s": 300"#),
            (
                "session_attainment",
                r#""attainment_gap": 0.3, "min_turns": 3"#,
            ),
            ("slowdown", r#""threshold": 5.0"#),
        ] {
            let text = format!(
                r#"{{"admission": {{"lobby_order": "signal",
                    "session_signal": {{"signal": "{signal}", {extra},
                                        "order_classes": ["standard", "batch"]}},
                    "starvation_override": {{"per_worker_per_min": 2,
                                             "classes": ["standard", "batch"]}}}}}}"#
            );
            let policy: ServicePolicy = serde_json::from_str(&text).unwrap();
            policy.validate().unwrap();
            assert_eq!(policy.admission.lobby_order, LobbyOrder::Signal);
            let cfg = policy.admission.session_signal.as_ref().unwrap();
            assert_eq!(cfg.signal.as_str(), signal);
            assert_eq!(
                policy.admission.starvation_override,
                Some(StarvationOverrideConfig::default())
            );
        }
        // Absent session_signal: the defaults (held_time).
        let policy: ServicePolicy =
            serde_json::from_str(r#"{"admission": {"starvation_override": {}}}"#).unwrap();
        policy.validate().unwrap();
        assert_eq!(
            Scheduler::new(policy).signal_cfg,
            SessionSignalConfig::default()
        );
        let check = |f: &dyn Fn(&mut ServicePolicy)| {
            let mut p = ServicePolicy::default();
            f(&mut p);
            p.validate().map_err(|e| e.to_string())
        };
        let err = check(&|p: &mut ServicePolicy| {
            p.admission.lobby_order = LobbyOrder::Signal;
            p.admission.turn_order = crate::config::TurnOrder::RunLas;
        })
        .unwrap_err();
        assert!(err.starts_with("admission.lobby_order: "), "{err}");
        let err = check(&|p: &mut ServicePolicy| {
            p.admission.starvation_override = Some(StarvationOverrideConfig {
                classes: vec!["gold".to_owned()],
                ..Default::default()
            })
        })
        .unwrap_err();
        assert!(err.starts_with("admission.starvation_override: "), "{err}");
        let err = check(&|p: &mut ServicePolicy| {
            p.admission.session_signal = Some(SessionSignalConfig {
                held_s: -1.0,
                ..Default::default()
            })
        })
        .unwrap_err();
        assert!(err.starts_with("admission.session_signal: "), "{err}");
        assert!(
            serde_json::from_str::<ServicePolicy>(
                r#"{"admission": {"session_signal": {"signal": "vibes"}}}"#
            )
            .is_err()
        );
    }
}
