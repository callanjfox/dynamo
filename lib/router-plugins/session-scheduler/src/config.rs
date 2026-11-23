// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The service policy: the `parameters` payload of the router policy file's
//! `request_classifier: { type: session_scheduler, parameters: ... }` block.
//!
//! Defaults match the admission simulator (`dynamo-admission-sim`): priority p90 TTFT 3 s and
//! ITL 50 ms in 90% of windows, standard 8 s and 100 ms in 70%, batch with no service level.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::forecast::{ForecastConfig, StepModelError};

/// Whether the scheduler acts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Pass every request through; record nothing.
    Off,
    /// Decide and log every decision, but let every head go.
    #[default]
    Shadow,
    /// Act on the decisions.
    Enforce,
}

/// What happens to a session that has waited its class's lobby limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WaitLimitAction {
    /// Admit it anyway (the simulator's priority class).
    Admit,
    /// Turn it away. The router expires it through `due_at`; see the README for the status code.
    Expire,
}

/// Whether the gate budgets a session at its projected size (`admission.lifetime`, F22): the peak
/// context its client hints it will reach (`x-dynamo-hint-peak-tokens`), not only its context
/// now. A session without the hint counts as before.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifetimeBudget {
    /// Context now (and, for the hot budget, the next turn): the gate as before.
    #[default]
    Off,
    /// The total (HBM plus DRAM) budget counts every live session, and a new one, at the larger
    /// of its context and its hinted peak.
    Total,
    /// `Total`, and the hot budget sizes a session's next turn from its hinted peak too.
    Both,
}

/// Surge admission (`admission.surge`, F22): while the fleet's queued prefill would take longer
/// to clear than the tightest TTFT limit of the classes ranked above a session's class, new
/// sessions of that class wait in the lobby, so a warm start or ramp does not bury the higher
/// classes' first tokens under lower classes' prefill. The top-ranked class is never held by it
/// (the KV budget still applies). Seconds to clear come from the step model and each worker's
/// chunk, so nothing here is tied to a model or hardware.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SurgeConfig {
    /// Multiplies the TTFT limit the queued prefill is compared with.
    pub factor: f64,
}

impl Default for SurgeConfig {
    fn default() -> Self {
        Self { factor: 1.0 }
    }
}

/// What a live session counts toward the gate's hot (HBM) budget (`admission.hot_demand`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HotDemand {
    /// A session with a turn in flight counts its current context (or the pending turn's input,
    /// if larger); a session between turns counts nothing.
    #[default]
    InFlight,
    /// As the simulator's gate counts it (`web/sim.js` `sessionDemand`): a session with a turn in
    /// flight counts its next size (context plus the turn's new tokens and output, from the
    /// client's signals), and a session between turns counts that size times its duty cycle,
    /// service / (service + think), from its last turn's observed service time (dispatch to
    /// completion) and its think signal. A session between turns with no observed turn or no
    /// think signal counts nothing.
    NextTurn,
}

/// f49k: what the HBM hot budget counts (`admission.hot_budget`). `true` (default) or `booked`:
/// the session table's bookings (`hot_demand`); `measured`: the engines' device KV per worker, capped
/// at 0.95 of the worker's HBM, plus the input of admitted turns not yet measured
/// ([`crate::hot_measure`]); `false` or `off`: no hot budget. Serialized as before: `true`, `false`,
/// or `measured`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HotBudget {
    Off,
    #[default]
    Booked,
    Measured,
}

impl HotBudget {
    /// Whether the hot budget binds at all.
    pub fn on(self) -> bool {
        self != Self::Off
    }

    /// f49k: whether the hot use is measured.
    pub fn is_measured(self) -> bool {
        self == Self::Measured
    }
}

impl From<bool> for HotBudget {
    fn from(on: bool) -> Self {
        if on { Self::Booked } else { Self::Off }
    }
}

impl Serialize for HotBudget {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Off => s.serialize_bool(false),
            Self::Booked => s.serialize_bool(true),
            Self::Measured => s.serialize_str("measured"),
        }
    }
}

impl<'de> Deserialize<'de> for HotBudget {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Flag(bool),
            Name(String),
        }
        match Raw::deserialize(d)? {
            Raw::Flag(on) => Ok(on.into()),
            Raw::Name(name) => match name.as_str() {
                "booked" => Ok(Self::Booked),
                "measured" => Ok(Self::Measured),
                "off" => Ok(Self::Off),
                other => Err(serde::de::Error::custom(format!(
                    "admission.hot_budget must be true, false, booked, measured or off, got {other:?}"
                ))),
            },
        }
    }
}

/// f49k2: how `admission.class_overcommit` treats the classes ranked above an overcommitted class
/// (`admission.class_overcommit_mode`). A class's hot room is its hot budget minus the fleet's hot
/// use of every class, so under f49k (`legacy`) a lower class lifted past a higher class's budget
/// fills hot past it and leaves the higher class no room. `max` (default): a class's hot budget is
/// at least the largest of the classes ranked at or below it (monotone in rank). `separate`: a
/// class's hot budget adds the overcommitted part (the budget past `class_overcommit` 1) of each
/// class ranked below it, so their overcommitted use is not charged to it (the hot use is not split
/// by class: under `hot_budget: measured` it is the engines' device KV). Only with
/// `class_overcommit` set ([`crate::admit::hot_cap`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClassOvercommitMode {
    #[default]
    Max,
    Separate,
    Legacy,
}

/// Which session's turn goes first within a class (F35, f36b). The lobby is ordered by it; with
/// `run_las` and `server_wait_deficit` the admitted requests are ordered by it too, through the
/// router queue's ordering key (`ClassifyRequest::set_order_key`), which never expires a request
/// (a `due_at` would). The engines still order by class.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnOrder {
    /// First in, first out by turn (and by `lobby_order` in the lobby).
    #[default]
    Fifo,
    /// The oldest session first: by the session's first offer to the router, which retries,
    /// rotations and waits never reset. Favours sessions finishing (end-to-end session time)
    /// over fairness between turns, with no lifetime hint.
    SessionAge,
    /// The session furthest behind first (goal 21). With `runs.deficit: delay` (default) the
    /// deficit is the session's added delay (D_1, over all its runs) less its allowance (each
    /// run's `run_level.delay_base_s` plus `delay_per_ref` times its reference service); with
    /// `server_share` it is its server time over its wall time less `run_level.server_share`.
    /// Per session (a subagent counts for its root), so a retry keeps its earlier wait.
    ServerWaitDeficit,
    /// Least attained service within the current run (goal 22; `runs` settings): the request of
    /// the run that has received the least service (its completed requests' uncontended time)
    /// goes first, reset at each stop point. A run is promoted once its added delay would pass
    /// its allowance (`runs.signal: delay`, default) or its slowdown its class's
    /// `run_level.slowdown` (`slowdown`); an autonomous run (no stop point for
    /// `runs.autonomous_after_s`) is promoted only when its progress over the trailing
    /// `runs.floor_window_s` falls under `runs.floor_share` of its uncontended rate.
    RunLas,
}

/// The session log (`DYNAMO_SESSION_LOG`, F35): what each session got, and its runs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SessionLogConfig {
    /// A gap with nothing of the session in flight longer than this is a stop point (a person
    /// reading, or an agent waiting on feedback) when neither the frontend's `input_trigger` nor
    /// the client (`x-dynamo-hint-last-role`) says whether a request starts with a user message.
    /// 180 s (owner, 2026-11-16): the AgentX gaps put most tool gaps under 10 s and a bump of
    /// human gaps at 3 to 5 minutes; 30 s split tool gaps into stops.
    pub stop_gap_s: f64,
    /// Runs listed per session in its summary (the rest are only counted).
    #[serde(alias = "max_bursts")]
    pub max_runs: usize,
}

impl Default for SessionLogConfig {
    fn default() -> Self {
        Self {
            stop_gap_s: 180.0,
            max_runs: 64,
        }
    }
}

/// Which headroom the parts that read a class's attainment (shedding, pacing, release, the soft
/// reserve, the adaptive rule and the coordinator) see (`runs.headroom`, f36b).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadroomSource {
    /// Per-request attainment (TTFT and ITL windows) against `service_level.attainment`, as before.
    #[default]
    Request,
    /// Run headroom only (`runs.signal`), mapped onto the class's request target: headroom h
    /// in [-1, 1] reads as target + h x (1 - target) above 0 and target + h x target below.
    Run,
    /// The tighter of the two (each where known).
    Min,
}

/// A class's run-level targets (service class family (b), goal 22; `docs/test-plan-full.md`
/// "The run metric: added delay").
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RunLevel {
    /// A run's added delay allowance is `delay_base_s + delay_per_ref x S_ref`, S_ref its
    /// reference service (the main agent's requests alone, from the step model).
    pub delay_base_s: f64,
    pub delay_per_ref: f64,
    /// Seconds over which the class's added delay and allowance are summed for its headroom
    /// (`runs.signal: delay`): a fast signal, not a percentile.
    pub delay_window_s: f64,
    /// Run slowdown target: hand-back time over the run's uncontended time (`runs.signal:
    /// slowdown`, and reported).
    pub slowdown: f64,
    /// Share of the class's runs that must be within `slowdown` (`runs.signal: slowdown`).
    pub attainment: f64,
    /// Target share of a session's wall time spent waiting on the server
    /// (`turn_order: server_wait_deficit`).
    pub server_share: f64,
    /// Runs counted for the class's run attainment (`runs.signal: slowdown`): those that ended
    /// within this many seconds.
    pub window_s: f64,
}

impl Default for RunLevel {
    fn default() -> Self {
        Self {
            delay_base_s: 10.0,
            delay_per_ref: 0.25,
            delay_window_s: 300.0,
            slowdown: 3.0,
            attainment: 0.95,
            server_share: 1.0,
            window_s: 1800.0,
        }
    }
}

/// What a class's run headroom is measured in (`runs.signal`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunSignal {
    /// Added delay D_1 (waiting plus prefill excess, refusals and retry backoff counted) against
    /// the allowance `delay_base_s + delay_per_ref x S_ref`, summed over the class's trailing
    /// `delay_window_s`: headroom is 1 - delay / allowance, within [-1, 1].
    #[default]
    Delay,
    /// The share of the class's runs ended in its trailing `window_s` within `slowdown`, less
    /// `attainment` (the first f36b build).
    Slowdown,
}

/// What `turn_order: server_wait_deficit` measures (`runs.deficit`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeficitMeasure {
    /// The session's added delay less its allowance, seconds.
    #[default]
    Delay,
    /// The session's server-wait share less its class's `run_level.server_share`.
    ServerShare,
}

/// Run-level settings shared by every class (service class family (b), f36b). They change
/// nothing unless `admission.turn_order` is `run_las` or `server_wait_deficit`,
/// `admission.compute_headroom` is set, or `headroom` is not `request`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RunConfig {
    /// Which headroom the attainment readers see. Default `request` (as before).
    pub headroom: HeadroomSource,
    /// What run headroom (and `run_las` promotion) is measured in. Default `delay`.
    pub signal: RunSignal,
    /// What `server_wait_deficit` orders by. Default `delay`.
    pub deficit: DeficitMeasure,
    /// Only sessions of this kind (`interactive` or `agent`, from `x-dynamo-hint-session-kind`)
    /// are ordered by run (or deficit), held by compute headroom and counted for run attainment;
    /// the others keep per-request targets and first-come order (the mixed fleet). Absent: every
    /// session.
    pub kind: Option<String>,
    /// `run_las`: seconds of queue order one second of attained service costs. A run's ordering
    /// key is its request's arrival plus `las_weight` times its attained service, capped at the
    /// time it is promoted; large weights approach pure least attained service.
    pub las_weight: f64,
    /// A run with no stop point for this long is autonomous: the class's slowdown target no
    /// longer promotes it; the floor does.
    pub autonomous_after_s: f64,
    /// An autonomous run is promoted while its uncontended progress over the trailing
    /// `floor_window_s` is under this share of the window.
    pub floor_share: f64,
    pub floor_window_s: f64,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            headroom: HeadroomSource::Request,
            signal: RunSignal::Delay,
            deficit: DeficitMeasure::Delay,
            kind: None,
            las_weight: 10.0,
            autonomous_after_s: 3600.0,
            floor_share: 0.1,
            floor_window_s: 600.0,
        }
    }
}

/// Onboarding pace (`admission.onboard_pace`, f39): a new session of a paced class is admitted only
/// while a token bucket of onboarding prefill is not in debt. The bucket refills at `tokens_per_s`
/// (the fleet's, set to a share of its prefill capacity) up to `burst_tokens`; each admitted new
/// session takes its first turn's input tokens (its cold prefill), and may take it into debt, so a
/// first turn larger than the burst still gets in. Until the class's wait limit, like any hold.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct OnboardPaceConfig {
    pub tokens_per_s: f64,
    pub burst_tokens: f64,
    /// Classes it paces; empty: every class.
    pub classes: Vec<String>,
}

impl Default for OnboardPaceConfig {
    fn default() -> Self {
        Self {
            tokens_per_s: 0.0,
            burst_tokens: 0.0,
            classes: Vec::new(),
        }
    }
}

/// The two-resource gate's compute side (`admission.prefill_budget`, f42): a new session is admitted
/// only while the fleet's measured prefill rate leaves room for its onboarding prefill under the
/// workers' prefill budgets. A worker's budget is the uncached prefill rate it can mix into its
/// decode steps and keep the mean inter-token latency of its strictest class in flight under that
/// class's ITL limit: with decode step `d` (the step model at the worker's running requests and
/// mean context), per-token prefill cost `u` (the step model's chunk time over the chunk) and limit
/// `L`, mean ITL is about `d / (1 - R u)`, so the budget is `margin (1 - d / L) / u`. A worker
/// with no class that has an ITL limit in flight gets `margin / u`. A new session counts as its
/// first turn's uncached input spread over `window_s`. The top-ranked class is never held; the
/// others wait, until their wait limit, like any hold. The budget follows the class targets: a
/// tighter ITL limit lowers it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PrefillBudgetConfig {
    /// Share of each worker's budget planned to, in (0, 1].
    pub margin: f64,
    /// The window the prefill rate is measured over and a new session's onboarding is spread
    /// over, seconds.
    pub window_s: f64,
    /// Send an admitted new session's first turn to the worker with the most prefill room among
    /// those with no class in flight whose ITL limit is tighter than the session's own (any
    /// worker when none), when it has room for the onboarding. Off: the router places it.
    pub place_onboard: bool,
    /// The look-ahead (the slow path): every `lookahead_step_s` predict the prefill the admitted
    /// sessions will bring over the next `lookahead_s` seconds (each idle session's next turn, due
    /// at its last turn's end plus its think time, at its hinted next-turn tokens or its mean
    /// growth per turn), and judge new sessions against the larger of that rate and the measured
    /// one. Off (absent) by default: the measured rate alone.
    pub lookahead_s: Option<f64>,
    pub lookahead_step_s: f64,
    /// Calibrate the budget online against the top class's quantile (f42 review): every
    /// `lookahead_step_s` a factor on every worker's budget falls at `cal_down` per minute per unit
    /// of excess while the share of the top class's completions over its ITL limit in its window is
    /// over the share its target allows (1 - attainment), and rises at `cal_up` while under it,
    /// between `cal_min` and `cal_max` (the formula sets the mean ITL at the limit; the quantile
    /// needs less). Off by default: factor 1.
    pub calibrate: bool,
    pub cal_down: f64,
    pub cal_up: f64,
    pub cal_min: f64,
    pub cal_max: f64,
    /// Judge new sessions of the top-ranked class against the budget too (f47), refusing them at
    /// their wait limit (expire, with the Retry-After estimate) while still over it. Off (the
    /// default): the top class is never held by the budget.
    pub include_top: bool,
    /// What the meter and a new session's onboarding term count (bug #12): `device` (the default,
    /// f42) the prompt less the best device-cached prefix, so host and disk hits count as prefill;
    /// `uncached` the prompt less its prefix cached in any tier on the worker it is sent to.
    pub meter: PrefillMeterKind,
    /// f47j: per-worker placement of large turns (see [`PlaceTurns`]). Off (absent) by default.
    pub place_turns: Option<PlaceTurns>,
}

/// `admission.prefill_budget.place_turns` (f47j): a turn computing at least `min_tokens` goes to a
/// worker whose metered prefill rate plus its own stays at or under `cap_tok_s`, away from workers
/// decoding the tightest-ITL class when one qualifies, preferring the most cached; needs
/// `meter: uncached`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PlaceTurns {
    pub min_tokens: u64,
    pub cap_tok_s: f64,
}

impl Default for PlaceTurns {
    fn default() -> Self {
        Self {
            min_tokens: 4096,
            cap_tok_s: 3000.0,
        }
    }
}

/// `admission.prefill_budget.meter` (f47h).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefillMeterKind {
    #[default]
    Device,
    Uncached,
}

impl Default for PrefillBudgetConfig {
    fn default() -> Self {
        Self {
            margin: 0.85,
            window_s: 30.0,
            place_onboard: true,
            lookahead_s: None,
            lookahead_step_s: 10.0,
            calibrate: false,
            cal_down: 20.0,
            cal_up: 1.0,
            cal_min: 0.3,
            cal_max: 1.5,
            include_top: false,
            meter: PrefillMeterKind::Device,
            place_turns: None,
        }
    }
}

/// The target loop (`admission.target_loop`, f42): a continuous controller on how much the classes
/// at or below `class` may admit. Every `step_s` it compares the class's smoothed attainment with
/// its target plus `margin` and moves a scale on the log scale by `gain` per minute per unit of
/// error; while a class ranked above runs under its own target plus `guard`, the scale falls at
/// least by that shortfall's rate. The scale multiplies the class's (and lower classes') HBM gate
/// share and their share of the prefill budget. A small margin admits more sessions at an
/// attainment just over the target (even spread); a large margin admits fewer and serves them
/// better.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct TargetLoopConfig {
    pub class: String,
    pub margin: f64,
    pub guard: f64,
    /// The guard's gain, per minute per unit of shortfall (a higher class's misses are small in
    /// attainment terms, so it acts faster than `gain`).
    pub guard_gain: f64,
    /// The scale grows at `gain` times this (shrink fast, grow slow), and only while the loop's
    /// class has demand: a session waiting in the lobby or one turned away since the last step
    /// (yield to demand, F18; no wind-up while nobody waits).
    pub up_factor: f64,
    pub step_s: f64,
    pub gain: f64,
    pub smooth_s: f64,
    pub min_scale: f64,
    pub max_scale: f64,
    /// A higher class short of its target for this long with the scale at `min_scale` is a target
    /// the loop cannot meet: every new session of the loop's classes is then held (and refused
    /// with Retry-After at its wait limit) until it recovers, and the time is counted.
    pub infeasible_after_s: f64,
    /// While infeasible, hold every new session of the loop's classes (default, as before). Off
    /// (best effort, f42 v3): holding the lower classes did not lift the higher class (that is
    /// what infeasible means), so they are served to their own target again, and the higher
    /// class's guard pulls only if it falls more than 0.02 under the attainment it had when the
    /// target was declared infeasible. The state is kept until the higher class meets its target.
    pub infeasible_hold: bool,
    /// Best effort only: re-probe an infeasible target every this many seconds (f42 v3r). The state
    /// is cleared and the real target (target + guard) enforced again; if it is still out of reach
    /// with the scale at its floor, it is declared again after `infeasible_after_s`. None (default):
    /// no re-probe. In best effort the protected level also ratchets up with the higher class's
    /// attainment either way.
    pub reprobe_s: Option<f64>,

    /// While a session of a class ranked above the loop's class waits in the lobby (checked every
    /// `step_s`), the loop's scale for the loop's class and the classes below it is multiplied by
    /// this (f47): their HBM gate share and prefill budget share shrink, leaving room for the
    /// higher class, and come back when its lobby clears. 1 (the default) is off.
    pub priority_wait_cut: f64,

    /// f49n: per-class scales for the loop's class and the classes below it, in place of the one
    /// scale ([`LoopRebalanceConfig`]). Absent (default): the loop above exactly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rebalance: Option<LoopRebalanceConfig>,
}

/// A class's attainment target changed at a time (`admission.target_schedule`, f42's tracking
/// test): from `at_s` seconds of scheduler time the class's `service_level.attainment` is
/// `attainment`. Applied by the target loop.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetChange {
    pub at_s: f64,
    pub class: String,
    pub attainment: f64,
}

impl Default for TargetLoopConfig {
    fn default() -> Self {
        Self {
            class: "standard".to_owned(),
            margin: 0.05,
            guard: 0.0,
            guard_gain: 10.0,
            up_factor: 0.25,
            step_s: 10.0,
            gain: 2.0,
            smooth_s: 120.0,
            min_scale: 0.2,
            max_scale: 2.0,
            infeasible_after_s: 120.0,
            infeasible_hold: true,
            reprobe_s: None,
            priority_wait_cut: 1.0,
            rebalance: None,
        }
    }
}

/// f49n `admission.target_loop.rebalance`: each class at or below the loop's class (standard,
/// batch) has its own scale, from 1 within its floor and ceiling, in place of the loop's one
/// scale (its gate share and prefill budget share; `priority_wait_cut` still multiplies it). Off
/// (absent) by default.
///
/// ```yaml
/// target_loop:
///   class: standard
///   rebalance:
///     margin: 0.02
///     release_per_min: 0.01
///     pull_per_min: 0.05
///     full_at: 0.05
///     floors: {standard: 0.5, batch: 0.333}
///     ceilings: {standard: 2.5, batch: 2.0}
///     release_order: own_target   # f49q4: own_target (default) | demand_first
/// ```
///
/// Every `step_s`, on the top class's head `h` (the highest-ranked class with a service level
/// above the loop's class: its smoothed attainment less its target + `margin`; in best effort,
/// once its target is declared infeasible, less the guard's protected level):
/// 1. Release (`h > 0`): the highest-ranked class with demand (a session waiting, or one turned
///    away since the last step), under its ceiling and under its OWN target + its margin (or with
///    no service level or no reading yet) grows `x exp(release_per_min / 60 x min(h / full_at, 1)
///    x dt)`: standard first, until it reaches its own target + margin or its ceiling, then batch.
/// 2. Pull back (`h < 0`): the lowest-ranked class above its floor shrinks `x exp(-pull_per_min /
///    60 x min(-h / full_at, 1) x dt)`: batch first, to its floor, then standard.
/// 3. The moves are f49l3's transfer step ([`crate::transfer`]) within f49l2's rate caps
///    (`max_shrink_per_min`, `max_grow_per_min`); nothing moves until the top class and the
///    moving class are warm (`warmup`).
/// 4. The guard, `guard_gain`, `min_scale` and `max_scale` still move the f42 scale, kept and
///    logged as `scale`. The infeasible state (held or best effort, `reprobe_s`) is declared on
///    the per-class floors: every class at or below the loop's at its floor with the top class
///    under its target for `infeasible_after_s`.
/// 5. No inversion: a class ranked above the loop's class has its gate share lifted so its hot cap
///    is never under a lower class's (f49l3's rule).
/// 6. With `DYNAMO_F42_LOG` the per-minute line gains `rb_mode` (`release`, `pull`, `hold`,
///    `floor` or `warmup`), `rb_head` and `rb` (per class: `scale`, `att`, `target`).
/// 7. f49q4 `release_order: demand_first` (default `own_target`: 1 and 5 above) replaces 1 and 5
///    ([`ReleaseOrder::DemandFirst`]): a release grows the highest-ranked class with demand under
///    its ceiling whatever its own attainment; no class's scale is over a higher one's (batch at
///    most standard, after every step); and no class above the loop's is lifted (a lower class's
///    hot cap may pass its, as without rebalance).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct LoopRebalanceConfig {
    /// Points over its target the top class keeps before it releases (under it, the pull back
    /// starts), and a lower class's margin unless `margins` lists it. Default 0.02.
    pub margin: f64,
    /// A lower class's own margin over its own target, by class (unlisted: `margin`). Default
    /// none.
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub margins: std::collections::BTreeMap<String, f64>,
    /// Release rate per minute at `full_at` of surplus (proportionally under it). Default 0.01.
    pub release_per_min: f64,
    /// Pull back rate per minute at `full_at` of shortfall (proportionally under it). Default 0.05.
    pub pull_per_min: f64,
    /// The head from which a move runs at its full rate. Default 0.05.
    pub full_at: f64,
    /// Lowest scale by class, in (0, 1] (unlisted: 0.5). Default standard 0.5, batch 0.333.
    pub floors: std::collections::BTreeMap<String, f64>,
    /// Highest scale by class, at least 1 (unlisted: 2). Default standard 2.5, batch 2.
    pub ceilings: std::collections::BTreeMap<String, f64>,
    /// f49l2's cap on a step's shrink, per minute, in (0, 1]. Absent (default): no cap.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_shrink_per_min: Option<f64>,
    /// f49l2's cap on a step's growth, per minute, positive. Absent (default): no cap.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_grow_per_min: Option<f64>,
    /// f49l2's warm-up. Absent (default): none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmup: Option<crate::sla_budget::WarmupConfig>,
    /// f49q4: which class a release grows, whether the lower classes stay ordered and whether the
    /// classes above the loop's are lifted ([`ReleaseOrder`]). Default `own_target` (f49n), not
    /// serialized.
    #[serde(skip_serializing_if = "ReleaseOrder::is_default")]
    pub release_order: ReleaseOrder,
}

/// f49q4 `target_loop.rebalance.release_order`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReleaseOrder {
    /// f49n: a release grows the highest-ranked class with demand, under its ceiling and under
    /// its own target + its margin; a class above the loop's is lifted over the lower ones.
    #[default]
    OwnTarget,
    /// A release grows the highest-ranked class with demand under its ceiling, whatever its own
    /// attainment (standard first while it has demand, then batch). No class's scale is ever over
    /// that of a class ranked above it (batch at most standard), and no class above the loop's is
    /// lifted (a lower class's hot cap may pass its, as without rebalance). The pull back is
    /// unchanged.
    DemandFirst,
}

impl ReleaseOrder {
    /// Whether it is the default (`own_target`).
    pub fn is_default(&self) -> bool {
        *self == Self::OwnTarget
    }
}

impl Default for LoopRebalanceConfig {
    fn default() -> Self {
        let map = |pairs: &[(&str, f64)]| {
            pairs
                .iter()
                .map(|(k, v)| ((*k).to_owned(), *v))
                .collect::<std::collections::BTreeMap<_, _>>()
        };
        Self {
            margin: 0.02,
            margins: std::collections::BTreeMap::new(),
            release_per_min: 0.01,
            pull_per_min: 0.05,
            full_at: 0.05,
            floors: map(&[("standard", 0.5), ("batch", 0.333)]),
            ceilings: map(&[("standard", 2.5), ("batch", 2.0)]),
            max_shrink_per_min: None,
            max_grow_per_min: None,
            warmup: None,
            release_order: ReleaseOrder::OwnTarget,
        }
    }
}

impl LoopRebalanceConfig {
    /// f49q4: whether `release_order: demand_first` is set.
    pub fn demand_first(&self) -> bool {
        self.release_order == ReleaseOrder::DemandFirst
    }

    /// A lower class's margin over its own target.
    pub fn margin_of(&self, class: &str) -> f64 {
        self.margins.get(class).copied().unwrap_or(self.margin)
    }

    /// A lower class's lowest scale.
    pub fn floor(&self, class: &str) -> f64 {
        self.floors.get(class).copied().unwrap_or(0.5)
    }

    /// A lower class's highest scale.
    pub fn ceiling(&self, class: &str) -> f64 {
        self.ceilings.get(class).copied().unwrap_or(2.0)
    }

    /// f49l2's rate caps.
    pub fn rate_caps(&self) -> crate::sla_budget::RateCaps {
        crate::sla_budget::RateCaps {
            max_shrink_per_min: self.max_shrink_per_min,
            max_grow_per_min: self.max_grow_per_min,
        }
    }

    /// f49l2: whether a class is past its warm-up (always without `warmup`).
    pub fn is_warm(&self, t_s: f64, has_service_level: bool, scored: Option<usize>) -> bool {
        self.warmup
            .as_ref()
            .is_none_or(|w| w.is_warm(t_s, has_service_level, scored))
    }

    pub fn validate(&self, policy: &ServicePolicy, loop_class: &str) -> Result<(), ConfigError> {
        let fail = |reason: String| Err(ConfigError::TargetLoopRebalance(reason));
        let positive = |x: f64| x.is_finite() && x > 0.0;
        let rate = |x: f64| x.is_finite() && x >= 0.0;
        for (field, expected, value, ok) in [
            ("margin", "finite", self.margin, self.margin.is_finite()),
            (
                "release_per_min",
                "finite and non-negative",
                self.release_per_min,
                rate(self.release_per_min),
            ),
            (
                "pull_per_min",
                "finite and non-negative",
                self.pull_per_min,
                rate(self.pull_per_min),
            ),
            ("full_at", "positive", self.full_at, positive(self.full_at)),
        ] {
            if !ok {
                return fail(format!("{field} must be {expected}, got {value}"));
            }
        }
        if let Some(m) = self.max_shrink_per_min
            && !(positive(m) && m <= 1.0)
        {
            return fail(format!("max_shrink_per_min must be in (0, 1], got {m}"));
        }
        if let Some(m) = self.max_grow_per_min
            && !positive(m)
        {
            return fail(format!(
                "max_grow_per_min must be positive and finite, got {m}"
            ));
        }
        if let Some(w) = &self.warmup {
            if w.min_scored == 0 {
                return fail("warmup.min_scored must be positive, got 0".to_owned());
            }
            if !positive(w.after_s) {
                return fail(format!(
                    "warmup.after_s must be positive and finite, got {}",
                    w.after_s
                ));
            }
        }
        let at = policy.class(Some(loop_class)).rank;
        if !policy
            .classes
            .iter()
            .any(|c| c.rank < at && c.service_level.is_some())
        {
            return fail(format!(
                "needs a class with a service_level ranked above target_loop.class {loop_class:?}"
            ));
        }
        let lower = |field: &str, name: &str| -> Result<(), ConfigError> {
            match policy.classes.iter().find(|c| c.name == name) {
                None => fail(format!("{field}: {name:?} is not a configured class")),
                Some(c) if c.rank < at => fail(format!(
                    "{field}: {name:?} is ranked above target_loop.class; only it and the classes below it have one"
                )),
                Some(_) => Ok(()),
            }
        };
        for (name, value) in &self.margins {
            lower("margins", name)?;
            if !(value.is_finite() && (0.0..1.0).contains(value)) {
                return fail(format!("margins.{name} must be in [0, 1), got {value}"));
            }
        }
        for (name, value) in &self.floors {
            lower("floors", name)?;
            if !(positive(*value) && *value <= 1.0) {
                return fail(format!("floors.{name} must be in (0, 1], got {value}"));
            }
        }
        for (name, value) in &self.ceilings {
            lower("ceilings", name)?;
            if !(value.is_finite() && *value >= 1.0) {
                return fail(format!(
                    "ceilings.{name} must be finite and at least 1, got {value}"
                ));
            }
        }
        Ok(())
    }
}

/// Admission on compute headroom (`admission.compute_headroom`, goal 22): a new session waits
/// while a class ranked at or above its own, with a `run_level`, has run headroom (run
/// attainment less its target) under `hold_below`. Until the class's wait limit, like any hold.
/// The KV budgets still apply; with `admission.hot_budget: false` this replaces the HBM hot
/// budget, so admission binds on whichever is tighter of measured KV (the total budget, from the
/// session index with `accounting.mode: index`) and compute headroom.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ComputeHeadroomConfig {
    pub hold_below: f64,
}

impl Default for ComputeHeadroomConfig {
    fn default() -> Self {
        Self { hold_below: 0.0 }
    }
}

/// Order of the lobby within a class rank.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LobbyOrder {
    /// First come, first served.
    #[default]
    Fifo,
    /// The least-served session first (fewest turns served so far), then first come. A new
    /// session has served none, so among new sessions this is first come; it differs only for
    /// sessions back in the lobby after serving turns (resuming after a rotation).
    Fair,
    /// First come by how long the session has waited in all (F32): a session keeps its place
    /// when its wait limit expires it and its client retries, and a session back from a rotation
    /// counts the lobby time it had before admission. First come (`fifo`) orders by the latest
    /// attempt's arrival, so a retry rejoins at the back.
    Aging,
    /// Tenant fair share (the simulator's `lobbyOrder: 'tenant'`): within a class, the session
    /// whose tenant is furthest under its share of the class's admitted sessions first (live
    /// top-level sessions of the class and tenant over `tenant_shares`, default 1), then first
    /// come. The tenant is the `x-dynamo-tenant` agent header; a session without one is a
    /// tenant of its own name "".
    Tenant,
    /// f49q: by the session signal (`admission.session_signal`), most starved first, for the
    /// classes in its `order_classes` (the others first come); then first come. Not with
    /// `turn_order: run_las` or `server_wait_deficit` (the same ordering key).
    Signal,
}

/// A class's service level: p-th percentile limits judged over a sliding window.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceLevel {
    pub ttft_s: f64,
    pub itl_s: f64,
    #[serde(default = "default_percentile")]
    pub percentile: f64,
    /// Share of judged windows that must pass.
    pub attainment: f64,
    #[serde(default = "default_window_s")]
    pub window_s: f64,
}

fn default_percentile() -> f64 {
    0.9
}
fn default_window_s() -> f64 {
    60.0
}

/// A class's admission settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassAdmission {
    /// Share of the HBM KV budget the class's sessions mid-turn may fill (the simulator's `gate`).
    pub gate_share: f64,
    /// Longest wait in the lobby before `on_wait_limit` applies.
    pub max_lobby_wait_s: f64,
    pub on_wait_limit: WaitLimitAction,
    /// Longest the gate holds a turn of this class's admitted sessions when it sheds by slack
    /// (`admission.shed_by_slack`; the simulator's `holdWait`). 0 never holds.
    #[serde(default)]
    pub shed_hold_s: f64,
    /// The class's prefill busy threshold, as a fraction of a worker's batched-token budget (the
    /// router's `prefill_busy_threshold_frac` for the same class): which workers have room for
    /// the gate's targeted `Go` (`admission.go_target`). Absent: 1.0.
    #[serde(default)]
    pub busy_frac: Option<f64>,
}

/// One service class.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassPolicy {
    /// Must match a router `policy_classes` name when the router has policy classes.
    pub name: String,
    /// Lower is served first.
    pub rank: u32,
    /// Relative value of a request, for trading between classes.
    pub value: f64,
    #[serde(default)]
    pub service_level: Option<ServiceLevel>,
    pub admission: ClassAdmission,
    /// Run-level targets (service class family (b), f36b); absent: none.
    #[serde(default)]
    pub run_level: Option<RunLevel>,
}

/// Settings shared by every class.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AdmissionPolicy {
    pub lobby_order: LobbyOrder,
    /// Which session's turn goes first within a class (F35). Default `fifo`.
    pub turn_order: TurnOrder,
    /// Share of HBM plus DRAM all live sessions' context may fill (the simulator's 0.95).
    pub total_headroom: f64,
    /// The hot (HBM) budget as a multiple of each class's `gate_share` of HBM (F32): over 1 the
    /// gate admits sessions whose turns in flight would together exceed HBM, counting on the
    /// engines to keep the rest in DRAM and reload it, while the total budget (HBM plus DRAM,
    /// `total_headroom`) stays the hard limit. Default 1.0 (the gate as before).
    pub hot_overcommit: f64,
    /// DRAM KV per worker in tokens. The classifier's worker view has no host-tier capacity
    /// today, so it is configured; 0 means HBM only.
    pub dram_tokens_per_worker: u64,
    /// The DRAM tier keeps a copy of every HBM block it was given (write-through offload), so a
    /// worker holds the larger of its HBM and DRAM, not their sum (the simulator's write-through
    /// capacity). Off: the sum (write-back or an exclusive tier).
    pub dram_write_through: bool,
    /// The KV tiers below HBM for the total budget's unique capacity; empty: the DRAM tier from
    /// `dram_tokens_per_worker` and `dram_write_through`.
    pub tiers: Vec<KvTier>,
    /// How the total budget counts usage: the session table (default) or the router's session
    /// index (F34).
    pub accounting: AccountingConfig,
    /// The class the engines make yield (batch yield: its prefill pauses and its decodes step
    /// aside for the higher classes, `class_scheduling.protect_max_rank` in AISimulate). Its live
    /// sessions' KV does not count against a higher class's budgets, as the simulator's gate
    /// counts it under `batchYield` (`web/sim.js` `hotFor`, `totFor`). None (default): every
    /// class counts.
    pub yield_class: Option<String>,
    /// What a live session counts toward the hot budget. Default `in_flight`.
    pub hot_demand: HotDemand,
    /// Most sessions waiting at once; a new session beyond it is throttled (rejected at
    /// classification in enforce mode). 0 means no limit.
    pub max_lobby_sessions: usize,
    /// The KV-budget session gate (the simulator's `gate`). Off, a new session is admitted at
    /// once and is never expired, while the turn-level switches below still apply.
    pub session_gate: bool,
    /// Shed by slack (the simulator's `holdsShed`, `slackShed`): hold a lower-class turn of an
    /// admitted session, up to its class's `shed_hold_s`, while a higher class has no slack and
    /// no worker has KV room for it. Off by default.
    pub shed_by_slack: bool,
    /// With `shed_by_slack` and the forecast on: also hold a lower-class turn while the
    /// forecast predicts its best placement pushes running requests of a higher class without
    /// slack over their ITL limit (the simulator's `holdReason` forecast-harm branch). Off by
    /// default.
    pub shed_forecast_harm: bool,
    /// Gate shares paced by attainment (the simulator's `pace`, `slaPaced: 'standard'`). Off
    /// (absent) by default.
    pub pace: Option<PaceConfig>,
    /// Retry-After for refused new sessions (the simulator's `retryAfter`, `retryAfterS`): a new
    /// session the gate turns away (its wait limit passed, or the lobby was full) is refused with
    /// a rate-limit error (HTTP 429) carrying this many seconds as Retry-After. Off (absent) by
    /// default: the wait limit expires the request through `due_at` and a throttle is an
    /// untyped error.
    pub retry_after_s: Option<f64>,
    /// Retry-After estimated per session (F32) instead of the fixed `retry_after_s`: a session
    /// that waits in the lobby is given, with the deadline that expires it at its class's wait
    /// limit, Retry-After = (its place in its class's lobby) / (its class's measured admission
    /// rate) minus the wait limit, when that is positive; otherwise none (the client's own
    /// backoff). The rate is admissions out of the class's lobby over the trailing
    /// `estimate_window_waits` times the class's wait limit. Off (absent) by default.
    pub retry_after_estimate: Option<RetryAfterEstimate>,
    /// Ask the host to queue every class so the gate sees all of them, including classes with no
    /// busy threshold (`DispatchGate::gates_every_class`). Off by default.
    pub queue_all_classes: bool,
    /// Answer `Go` with a worker that has room for the class (`busy_frac`): the session's last
    /// worker when it has room, else the least prefill-loaded worker with room, else none. Off
    /// by default (`Go` without a target).
    pub go_target: bool,
    /// Hold a new session while every worker is at its running-request limit (`max_num_seqs`,
    /// carried when `DYN_ROUTER_BUSY_AT_MAX_NUM_SEQS=1`). Off by default.
    pub hold_at_max_num_seqs: bool,
    /// Each tenant's share for `lobby_order: tenant` (the simulator's `cfg.tenants` shares); a
    /// tenant not listed has share 1.
    pub tenant_shares: HashMap<String, f64>,
    /// Rotate long-running sessions out to let waiting ones in (the simulator's `rotate`). Off
    /// (absent) by default.
    pub rotate: Option<RotateConfig>,
    /// Surge admission by the fleet's prefill capacity (F22). Off (absent) by default.
    pub surge: Option<SurgeConfig>,
    /// Budget sessions at their hinted peak context (F22). Off by default.
    pub lifetime: LifetimeBudget,
    /// Target-aware shedding (`admission.shed_target`): with `shed_by_slack`, a higher class
    /// counts as having no slack only while a hysteresis switch on its smoothed attainment is
    /// on, instead of whenever its last 60 s window is under its target plus 5 points. Off
    /// (absent) by default.
    pub shed_target: Option<ShedTargetConfig>,
    /// Surplus release (`admission.release`): while every class ranked above a class runs over
    /// its target by its margin, that class's gate share grows over its set share; it shrinks
    /// back three times as fast when a class above falls under its margin. Off (absent) by
    /// default.
    pub release: Option<ReleaseConfig>,
    /// Admission on compute headroom in run slowdown (f36b). Off (absent) by default.
    pub compute_headroom: Option<ComputeHeadroomConfig>,
    /// The HBM hot budget (each class's `gate_share` of HBM for sessions mid-turn). Default on
    /// (`true`, `booked`); off (`false`), only the total budget and the other holds bind (with
    /// `compute_headroom`, f36b). f49k `measured`: the hot use is the engines' measured device KV
    /// ([`HotBudget`], [`crate::hot_measure`]).
    pub hot_budget: HotBudget,
    /// f49k: a multiple of the hot budget per class (`gate_share x hot_overcommit x` HBM, times
    /// the pacing and step-target scales), for classes ranked below the top rank only; it may take
    /// a class's effective share past 1. Empty (absent) by default.
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub class_overcommit: std::collections::BTreeMap<String, f64>,
    /// f49k2: how `class_overcommit` treats the classes ranked above an overcommitted class
    /// ([`ClassOvercommitMode`]). Absent: `max`. Only with `class_overcommit` set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub class_overcommit_mode: Option<ClassOvercommitMode>,
    /// Classes the HBM hot budget leaves out (f39, the turn queue): their sessions are admitted on
    /// the total budget alone, and their sessions' KV does not count in the other classes' hot
    /// budget (the engines' per-request KV occupancy limits queue their turns instead). Empty
    /// (default): every class is gated as before.
    pub hot_exempt_classes: Vec<String>,
    /// Pace new sessions by their onboarding prefill (f39). Off (absent) by default.
    pub onboard_pace: Option<OnboardPaceConfig>,
    /// Admission on the fleet's prefill budget (f42, the two-resource gate). Off (absent) by
    /// default.
    pub prefill_budget: Option<PrefillBudgetConfig>,
    /// The target loop on the lower classes' admission (f42). Off (absent) by default.
    pub target_loop: Option<TargetLoopConfig>,
    /// Target changes over the run (f42 tracking test), applied in order by the target loop.
    /// Empty by default.
    pub target_schedule: Vec<TargetChange>,
    /// Judge each completed request's inter-token latency too, when the host reports its output
    /// tokens (`ClassifyEvent::Completed::output_tokens`; offline replay does): mean ITL is the
    /// time from first token to completion over output tokens minus one. Off by default: windows
    /// are judged on TTFT alone, as before.
    pub judge_itl: bool,
    /// Classes admitted at their wait limit when they fit the budgets, ignoring the order of earlier
    /// sessions of their class (f47); refused otherwise. Only with `on_wait_limit: expire`. Empty
    /// by default.
    pub fit_at_limit_classes: Vec<String>,
    /// f47g: seconds past the wait limit at which a session of a class listed in
    /// `fit_at_limit_classes` or `returnee_guarantee` expires, so the lobby check at the limit (which
    /// may admit it) runs before the host's queue deadline refuses it. Other classes expire at the
    /// limit.
    pub at_limit_grace_s: f64,
    /// Hold new sessions of the listed classes while the class's measured TTFT is over its target
    /// (f47). Off (absent) by default.
    pub ttft_cap: Option<TtftCapConfig>,
    /// ks2b: the session gate's forecast-harm check on new sessions (the simulator's `gateHarm`,
    /// `web/sim.js` `admitLobby`). A new session of a class not in `exempt_classes` is admitted only
    /// when its best placement adds at most `max_added_misses` weighted predicted misses to its own
    /// class and the classes ranked above it ([`crate::forecast::Forecast::new_session_harm`]);
    /// otherwise it waits in the lobby until its wait limit, like the gate's other holds. Needs
    /// `forecast.enabled`; without a worker sample nothing is held. Off (absent) by default.
    pub forecast_harm_new: Option<ForecastHarmNewConfig>,
    /// f49h: AIMD per-worker admission of one class's turns ([`crate::aimd`]): each worker's limit
    /// on the class's active turns grows by one per interval under slack and halves on a TTFT or
    /// ITL miss; a turn bound for a worker at its limit waits, oldest sessions first, and is
    /// refused at `wait_bound_s` with a Retry-After from the measured departure rate. Off (absent)
    /// by default.
    pub aimd: Option<crate::aimd::AimdConfig>,
    /// A refused session of a listed class is not left behind (f47): once refused
    /// `after_refusals` times it is admitted at its wait limit when it fits the budgets,
    /// ignoring the order of earlier sessions of its class. Off (absent) by default.
    pub returnee_guarantee: Option<ReturneeGuarantee>,
    /// Admission tickets (f49c, SLA "Must 1b"; [`crate::tickets`]): a refused session of a listed
    /// class gets a ticket that holds its KV while it is away, puts it at the front of its class when
    /// it returns, and (`retry_after: forecast`) is advised back when its KV is projected to free.
    /// f49e options under it: `hold_while_waiting`, `per_root`, `keep_on_rerefusal`,
    /// `standard_cap_when_priority_below` (all off by default). Off (absent) by default.
    pub tickets: Option<TicketsConfig>,
    /// Returnee first (f49f, SLA "Must 1b" option A): while a listed class has a live ticket
    /// holder not yet admitted, its new top-level sessions are refused (with their own ticket and
    /// its Retry-After) and its holders are admitted in the order of their first refusal. Needs
    /// `tickets` for the same classes. Off (absent) by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub returnee_first: Option<ReturneeFirstConfig>,
    /// f49m: honest Retry-After ([`crate::honest_retry`]): every refusal advises the session's
    /// place over its class's measured admission rate; at `max_refusals` refusals its ticket
    /// becomes a reservation; `returnee_first` refuses a new session only when it does not fit.
    /// Needs `tickets`. Off (absent) by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub honest_retry: Option<crate::honest_retry::HonestRetryConfig>,
    /// f49i: a per-worker step-time budget ([`crate::step_budget`]): a turn goes only to a worker
    /// where the forecast's projected mean ITL of each listed class decoding there (its own class
    /// included when listed), with the turn added, stays at most `margin` x the class's ITL limit;
    /// the route skips workers that break it, and the gate holds the turn while every worker does.
    /// Needs `forecast.enabled`. Off (absent) by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_step_budget: Option<crate::step_budget::WorkerStepBudgetConfig>,
    /// f49j: the per-worker step-target controller ([`crate::step_target`]): each worker's measured
    /// mean decode ITL (every class, trailing `window_s`) against `target_frac` x `class`'s ITL
    /// limit steers a multiple of `hot_overcommit` (PI with anti-windup, clamped to [min, max],
    /// slew-limited) for the classes ranked below `class`; the gate's fleet KV fit takes the
    /// HBM-weighted mean of the workers' multiples. Off (absent) by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step_target: Option<crate::step_target::StepTargetConfig>,
    /// f49l: the outer SLA loop ([`crate::sla_budget`]): per class with a service level, the
    /// smoothed attainment (the top class on all offered work, the others on admitted) against
    /// target plus margin steers a share of each lower class's hot budget, the step target's
    /// `target_frac`, and as a fallback the top class's own share. Off (absent) by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sla_budget: Option<crate::sla_budget::SlaBudgetConfig>,
    /// f49q: the session signal ([`crate::session_signal`]) `starvation_override` and
    /// `lobby_order: signal` read. Absent: its defaults (`held_time`, 300 s).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_signal: Option<crate::session_signal::SessionSignalConfig>,
    /// f49q: admit a starved session of a listed class past its class's own budgets, within the
    /// total and the higher classes' budgets and `per_worker_per_min` per worker. Off (absent) by
    /// default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub starvation_override: Option<crate::session_signal::StarvationOverrideConfig>,
    /// f49o: batch yield ([`crate::batch_yield`]): the yield class gives way to the watched classes as
    /// far as their attainment needs it (gate discount and scale, engine preemption and step-aside
    /// through the router's switches). Off (absent) by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_yield: Option<crate::batch_yield::BatchYieldConfig>,
}

/// `admission.returnee_first` (f49f).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ReturneeFirstConfig {
    /// Classes the rule applies to; each must be in `admission.tickets.classes`. Default
    /// `[priority, standard]`.
    pub classes: Vec<String>,
    /// Seconds after its issue (its first refusal under `keep_on_rerefusal`) a ticket stops
    /// blocking new sessions and younger holders of its class; it keeps its own front of line.
    /// Default absent: it blocks while it is live.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_block_s: Option<f64>,
    /// `only_when_waiting` (f49f2): refuse a new session only while a live holder of its class
    /// waits in the lobby or is away and due back within `due_soon_s`; otherwise let it through,
    /// so no room stays idle while the rule refuses. Default false (refuse while any holder of its
    /// class is live).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub only_when_waiting: bool,
    /// `due_soon_s` (f49f2, with `only_when_waiting`): an away holder whose ticket is due back
    /// within this many seconds counts as waiting. Default 10.
    #[serde(skip_serializing_if = "ReturneeFirstConfig::due_soon_is_default")]
    pub due_soon_s: f64,
}

impl ReturneeFirstConfig {
    /// f49f2: the default `due_soon_s`.
    pub const DUE_SOON_S: f64 = 10.0;

    fn due_soon_is_default(v: &f64) -> bool {
        v.to_bits() == Self::DUE_SOON_S.to_bits()
    }
}

impl Default for ReturneeFirstConfig {
    fn default() -> Self {
        Self {
            classes: vec!["priority".to_owned(), "standard".to_owned()],
            max_block_s: None,
            only_when_waiting: false,
            due_soon_s: Self::DUE_SOON_S,
        }
    }
}

/// `admission.tickets` (f49c).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct TicketsConfig {
    /// Classes whose refusals issue tickets, ranked by their class rank. Default `[priority,
    /// standard]`.
    pub classes: Vec<String>,
    /// Seconds after its advised return (`due`) a ticket lapses. Default 30.
    pub grace_s: f64,
    /// Most KV the live tickets hold together, as a share of the HBM budget. Default 0.3.
    pub max_hold_frac: f64,
    /// The Retry-After a ticketed refusal carries. Default `forecast`.
    pub retry_after: TicketRetryAfter,
    /// `hold_while_waiting` (f49e): a live ticket whose holder came back and waits in the lobby does
    /// not lapse at its expiry; it ends at admission, at session end, or when the holder's lobby
    /// deadline refuses it. Default false.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub hold_while_waiting: bool,
    /// `per_root` (f49e): one ticket per root session. A sub-session gets no ticket of its own: it
    /// uses its root's live ticket (front of line, the root's hold not counted against it, never
    /// throttled); its refusal issues the root's ticket when none is live, and otherwise issues
    /// nothing new (the KV need becomes the larger of the two). Default false.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub per_root: bool,
    /// `keep_on_rerefusal` (f49e): a refusal of a session whose ticket is live keeps the ticket (its
    /// issue time, due and returns) and only extends its expiry to the new due plus `grace_s`.
    /// Default false (the ticket is replaced).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub keep_on_rerefusal: bool,
    /// `standard_cap_when_priority_below` (f49e): while the top-ranked class with a service level is
    /// under its attainment target (the target loop's smoothed attainment, else the attainment it
    /// smooths), the tickets of the classes ranked below it hold at most this share of the HBM
    /// budget together (`max_hold_frac` still caps all tickets). Default absent (off).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub standard_cap_when_priority_below: Option<f64>,
}

impl Default for TicketsConfig {
    fn default() -> Self {
        Self {
            classes: vec!["priority".to_owned(), "standard".to_owned()],
            grace_s: 30.0,
            max_hold_frac: 0.3,
            retry_after: TicketRetryAfter::Forecast,
            hold_while_waiting: false,
            per_root: false,
            keep_on_rerefusal: false,
            standard_cap_when_priority_below: None,
        }
    }
}

/// `admission.tickets.retry_after` (f49c).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TicketRetryAfter {
    /// When the session's KV need is projected to free (the admitted sessions' predicted releases,
    /// less the holds of earlier tickets of its class or above), plus the prefill window.
    #[default]
    Forecast,
    /// Today's advice (`retry_after_estimate`, else `retry_after_s`).
    Rate,
}

/// `admission.returnee_guarantee` (f47).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ReturneeGuarantee {
    pub classes: Vec<String>,
    pub after_refusals: u32,
}

impl Default for ReturneeGuarantee {
    fn default() -> Self {
        Self {
            classes: vec!["priority".to_owned()],
            after_refusals: 2,
        }
    }
}

/// `admission.forecast_harm_new` (ks2b): the forecast-harm check on new sessions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ForecastHarmNewConfig {
    /// Most weighted misses (each `1 / (1 - attainment)` of its class) the session's best placement
    /// may add to its own class and the classes ranked above it. Default 0.01 (the simulator's).
    pub max_added_misses: f64,
    /// Classes never held by the check. Default `[batch]`.
    pub exempt_classes: Vec<String>,
}

impl Default for ForecastHarmNewConfig {
    fn default() -> Self {
        Self {
            max_added_misses: 0.01,
            exempt_classes: vec!["batch".to_owned()],
        }
    }
}

/// A class's overload cap on measured TTFT (`admission.ttft_cap`, f47): a new session of a listed
/// class waits while the class's TTFT `quantile` over the last `window_s` seconds of completions
/// is over `factor` times its TTFT target; at its wait limit the class's `on_wait_limit` decides
/// (expire: refused, with the Retry-After estimate; admit: let in anyway), the customer's choice.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct TtftCapConfig {
    pub classes: Vec<String>,
    pub quantile: f64,
    pub window_s: f64,
    pub factor: f64,
}

impl Default for TtftCapConfig {
    fn default() -> Self {
        Self {
            classes: vec!["priority".to_owned()],
            quantile: 0.5,
            window_s: 60.0,
            factor: 1.0,
        }
    }
}

/// Target-aware shedding: the switch for a class with a service level turns on when its
/// smoothed attainment falls under `target + on_margin` and off when it rises over `target +
/// off_margin`, and stays in a state at least `min_hold_s` once it changes. It starts on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ShedTargetConfig {
    pub on_margin: f64,
    pub off_margin: f64,
    pub min_hold_s: f64,
    /// Smoothing of the windowed attainment the switch reads, seconds.
    pub smooth_s: f64,
    /// Seconds between two readings.
    pub step_s: f64,
}

impl Default for ShedTargetConfig {
    fn default() -> Self {
        Self {
            on_margin: 0.02,
            off_margin: 0.05,
            min_hold_s: 300.0,
            smooth_s: 300.0,
            step_s: 30.0,
        }
    }
}

/// Surplus release. For each class below the top: headroom is the least, over the classes
/// ranked above it that have a service level, of smoothed attainment minus (target + margin).
/// While the headroom is positive and the class has demand (a session of it waiting in the
/// lobby, or one turned away since the last step), its gate-share multiple grows at
/// `grow_per_min` per minute times headroom / 0.05 (capped at twice that), up to `max_scale`;
/// while it is negative it shrinks `shrink_ratio` times as fast, down to 1.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ReleaseConfig {
    /// The classes whose gate share may grow; empty: every class below the top.
    pub classes: Vec<String>,
    /// Grow a class's share only while no class ranked above it has demand of its own (a session
    /// waiting in the lobby, or one turned away since the last step): a class over its target on
    /// admitted work can still have users waiting, and releasing to the class below would take
    /// their room. Off by default (the first version).
    pub yield_to_demand: bool,
    /// With `yield_to_demand`: yield only to a class above whose smoothed attainment is under its
    /// target plus this; above it, release even while that class has users waiting. Absent:
    /// always yield.
    pub yield_margin: Option<f64>,
    /// Margin over target per class name; a class not listed uses `default_margin`.
    pub margins: HashMap<String, f64>,
    pub default_margin: f64,
    pub grow_per_min: f64,
    pub shrink_ratio: f64,
    pub max_scale: f64,
    pub smooth_s: f64,
    pub step_s: f64,
}

impl Default for ReleaseConfig {
    fn default() -> Self {
        Self {
            classes: Vec::new(),
            yield_to_demand: false,
            yield_margin: None,
            margins: HashMap::from([("priority".to_owned(), 0.02), ("standard".to_owned(), 0.05)]),
            default_margin: 0.05,
            grow_per_min: 0.02,
            shrink_ratio: 3.0,
            max_scale: 1.5,
            smooth_s: 300.0,
            step_s: 30.0,
        }
    }
}

/// Rotation (the simulator's `rotate: {afterS, minRunS}`, `web/sim.js` `rotateSessions`; F32): when
/// a class's oldest waiting new session has waited `after_s` (counted from its first attempt, so
/// retries after a wait-limit expiry do not reset it), pause one admitted session of that class or
/// a lower one (never for the top-ranked class's waiters, never a higher class for a lower one):
/// the lowest class first, then the longest run, that has run at least `min_run_s` (admitted time
/// less time paused), is between turns with no subagent in flight, and whose KV the router's
/// session index shows in the host tier (at least `host_cover` of its context). Its next turn goes
/// back through the lobby, where it is never refused. Checked at most every `check_s`, one pause
/// per waiting class per check, and at most `max_per_min` pauses a minute over all classes.
///
/// Absent values are derived: `after_s` is `after_waits` times the waiting class's wait limit
/// (`max_lobby_wait_s`); `min_run_s` is the larger of `after_s` and the oldest waiter's wait, so a
/// swap never parks a session that has run less than the waiter has waited.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RotateConfig {
    pub after_s: Option<f64>,
    /// With `after_s` absent: the trigger as a multiple of the waiting class's wait limit.
    pub after_waits: f64,
    pub min_run_s: Option<f64>,
    pub host_cover: f64,
    pub check_s: f64,
    /// Most pauses a minute over all classes; absent: one per waiting class per check.
    pub max_per_min: Option<f64>,
}

impl Default for RotateConfig {
    fn default() -> Self {
        Self {
            after_s: None,
            after_waits: 5.0,
            min_run_s: None,
            host_cover: 0.9,
            check_s: 5.0,
            max_per_min: None,
        }
    }
}

impl RotateConfig {
    /// Seconds the oldest waiter of a class with wait limit `wait_limit_s` must have waited.
    pub fn after_for(&self, wait_limit_s: f64) -> f64 {
        self.after_s.unwrap_or(self.after_waits * wait_limit_s)
    }

    /// Least run of a session paused for a waiter that has waited `waited_s`.
    pub fn min_run_for(&self, after_s: f64, waited_s: f64) -> f64 {
        self.min_run_s.unwrap_or(after_s.max(waited_s))
    }
}

/// The per-session Retry-After estimate (`admission.retry_after_estimate`, F32).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RetryAfterEstimate {
    /// The admission rate's window as a multiple of the class's wait limit.
    pub estimate_window_waits: f64,
    /// Longest Retry-After given, seconds (f40). None (default): no cap. Without a cap the
    /// estimate can feed on itself: clients sent away leave the lobby empty, admissions out of
    /// the lobby stop, the measured rate falls and the next estimates grow (F32's `agra` gave a
    /// median of 80 h at 1,664 users while the budgets had room).
    pub max_s: Option<f64>,
    /// Also count the class's sessions that end (each frees its budget for a waiting session)
    /// in the rate (f40): the rate is the larger of admissions out of the lobby and session ends
    /// over the window, so it does not fall to zero when the waiting clients are away. Off
    /// (default): admissions out of the lobby only, as before.
    pub count_ends: bool,
    /// Retry-After when the rate is not known yet (before a whole window, or no admission or end
    /// in it), seconds (f40). None (default): no Retry-After then (the client's own backoff), as
    /// before. Set, every refusal carries an explicit Retry-After.
    pub fallback_s: Option<f64>,
    /// Shortest Retry-After, seconds (f40), also given when the estimate is not positive (the
    /// session is near the front of the line). None (default): no Retry-After then, as before.
    pub min_s: Option<f64>,
    /// Calibrate the advice on returns (f47): a per-class factor multiplies the advice (the larger
    /// of the estimate and `min_s`) before `max_s`. It is lowered x0.95 (floor 1) when a refused
    /// session is admitted within 3 returns and raised x1.25 (cap 20) at a session's 3rd return
    /// without admission: the aim is admission within about 3 returns, not on the first. Off
    /// (default): the estimate as is.
    pub calibrate: bool,
}

impl Default for RetryAfterEstimate {
    fn default() -> Self {
        Self {
            estimate_window_waits: 10.0,
            max_s: None,
            count_ends: false,
            fallback_s: None,
            min_s: None,
            calibrate: false,
        }
    }
}

/// Gate shares paced by attainment: standard's share moves on demand and batch's gives way to
/// priority, as the simulator's `slaPaced: 'standard'` does (`web/sim.js` `paceGates`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PaceConfig {
    /// Priority's margin over its target standard's share needs to grow.
    pub head_priority: f64,
    /// Admitted standard's margin over its target standard's share needs to grow.
    pub head_standard: f64,
    /// Range of standard's share, as multiples of the set share.
    pub standard_low: f64,
    pub standard_high: f64,
    /// Lowest multiple batch's share falls to before standard's goes below the set share.
    pub batch_low: f64,
    /// Smoothing of the attainments the pacing reads, seconds.
    pub smooth_s: f64,
    /// Seconds between two pacing steps.
    pub step_s: f64,
}

impl Default for PaceConfig {
    fn default() -> Self {
        Self {
            head_priority: 0.03,
            head_standard: 0.15,
            standard_low: 0.8,
            standard_high: 2.5,
            batch_low: 1.0 / 3.0,
            smooth_s: 600.0,
            step_s: 12.0,
        }
    }
}

impl Default for AdmissionPolicy {
    fn default() -> Self {
        Self {
            lobby_order: LobbyOrder::Fifo,
            turn_order: TurnOrder::Fifo,
            compute_headroom: None,
            hot_budget: HotBudget::Booked,
            class_overcommit: std::collections::BTreeMap::new(),
            class_overcommit_mode: None,
            hot_exempt_classes: Vec::new(),
            onboard_pace: None,
            prefill_budget: None,
            target_loop: None,
            target_schedule: Vec::new(),
            total_headroom: 0.95,
            hot_overcommit: 1.0,
            dram_tokens_per_worker: 0,
            dram_write_through: false,
            tiers: Vec::new(),
            accounting: AccountingConfig::default(),
            yield_class: None,
            hot_demand: HotDemand::InFlight,
            max_lobby_sessions: 0,
            session_gate: true,
            shed_by_slack: false,
            shed_forecast_harm: false,
            pace: None,
            retry_after_s: None,
            retry_after_estimate: None,
            queue_all_classes: false,
            go_target: false,
            hold_at_max_num_seqs: false,
            tenant_shares: HashMap::new(),
            rotate: None,
            surge: None,
            lifetime: LifetimeBudget::Off,
            shed_target: None,
            release: None,
            judge_itl: false,
            fit_at_limit_classes: Vec::new(),
            at_limit_grace_s: 1.0,
            ttft_cap: None,
            forecast_harm_new: None,
            aimd: None,
            returnee_guarantee: None,
            tickets: None,
            returnee_first: None,
            honest_retry: None,
            worker_step_budget: None,
            step_target: None,
            sla_budget: None,
            session_signal: None,
            starvation_override: None,
            batch_yield: None,
        }
    }
}

impl AdmissionPolicy {
    /// The KV tokens the tiers below HBM add over `hbm_tokens` across `workers` workers, as
    /// unique capacity. With `tiers` set: the largest of HBM and the inclusive tiers, plus every
    /// exclusive tier, less HBM. Otherwise the DRAM tier alone: all of it, or with
    /// `dram_write_through` only what it holds beyond HBM's copies (the same rule with one tier).
    pub fn dram_beyond_hbm(&self, hbm_tokens: u64, workers: u64) -> u64 {
        if !self.tiers.is_empty() {
            let total = |t: &KvTier| t.tokens_per_worker.saturating_mul(workers);
            let inclusive = self
                .tiers
                .iter()
                .filter(|t| t.inclusive)
                .map(total)
                .fold(hbm_tokens, u64::max);
            let exclusive: u64 = self.tiers.iter().filter(|t| !t.inclusive).map(total).sum();
            return inclusive
                .saturating_add(exclusive)
                .saturating_sub(hbm_tokens);
        }
        let dram = self.dram_tokens_per_worker.saturating_mul(workers);
        if self.dram_write_through {
            dram.saturating_sub(hbm_tokens)
        } else {
            dram
        }
    }
}

/// A KV tier below HBM (`admission.tiers`, F32): host DRAM, a peer's memory, SSD, a shared store.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvTier {
    pub name: String,
    /// KV tokens this tier holds per worker (a shared tier: its size over the workers).
    pub tokens_per_worker: u64,
    /// Inclusive (write-through): it keeps a copy of what the tiers above hold, so it adds only
    /// what it holds beyond them. Exclusive: its blocks are not held above, so it adds all of it.
    pub inclusive: bool,
    /// Which tier of the router's session index reports this tier's blocks (F34,
    /// `admission.accounting.mode: index`). A tier with none cannot be measured, and index
    /// accounting then falls back to the table.
    #[serde(default)]
    pub index_tier: Option<IndexTier>,
}

/// A storage tier as the router's session index reports it (from the KV events' medium).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexTier {
    /// Host memory (vLLM medium `CPU_TIER1`, KVBM G2).
    Host,
    /// Local disk (`CPU_TIER2`, KVBM G3).
    Disk,
    /// A shared store outside the workers (Mooncake and the like).
    External,
}

/// How the gate's total budget counts what is in use (F34).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccountingMode {
    /// Each live session's context, summed (a shared prefix counts once per session).
    #[default]
    Table,
    /// Unique resident blocks per tier from the router's session prefix index, combined by the
    /// tiers' inclusive and exclusive rules, plus the table's growth since the measurement.
    Index,
}

/// `admission.accounting` (F34).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AccountingConfig {
    pub mode: AccountingMode,
    /// Seconds between measurements of the index (each walks the index once).
    pub refresh_s: f64,
    /// A measurement older than this many refreshes is stale: the gate counts the table.
    pub stale_after_refreshes: f64,
    /// Check that the engines honour the KV actions a session end implies. Off (absent) by
    /// default.
    pub verify: Option<VerifyActions>,
}

impl Default for AccountingConfig {
    fn default() -> Self {
        Self {
            mode: AccountingMode::Table,
            refresh_s: 30.0,
            stale_after_refreshes: 3.0,
            verify: None,
        }
    }
}

/// `admission.accounting.verify` (F34): after a session ends, check in the session index that its
/// blocks at each holder dropped. Per worker, the share of checks that did is the honoured rate;
/// under `honoured_below` (after `min_checks`), the table stops crediting a session end on that
/// worker and keeps counting the session's context until the index shows it gone or `max_hold_s`
/// passes. Index accounting needs no credit (it measures), so there the rate is only reported.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct VerifyActions {
    /// Seconds after the end before the check.
    pub window_s: f64,
    /// The share of a holder's blocks that must be gone for the action to count as honoured
    /// (a shared prefix other sessions hold may stay).
    pub min_drop: f64,
    pub min_checks: u64,
    pub honoured_below: f64,
    pub max_hold_s: f64,
}

impl Default for VerifyActions {
    fn default() -> Self {
        Self {
            window_s: 60.0,
            min_drop: 0.5,
            min_checks: 20,
            honoured_below: 0.5,
            max_hold_s: 600.0,
        }
    }
}

/// The plugin's parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ServicePolicy {
    pub mode: Mode,
    pub default_class: String,
    pub classes: Vec<ClassPolicy>,
    pub admission: AdmissionPolicy,
    /// Sessions tracked at once (least recently used beyond this are dropped).
    pub session_table_capacity: usize,
    /// Decisions kept for inspection.
    pub decision_log_capacity: usize,
    /// The forecast (off by default).
    pub forecast: ForecastConfig,
    /// The adaptive rule (`adaptive.rs`); off (absent) by default.
    pub adaptive: Option<crate::adaptive::AdaptiveConfig>,
    /// The signal-driven coordinator (`coordinator.rs`); off (absent) by default.
    pub controller: Option<crate::coordinator::ControllerConfig>,
    /// The session log (F35); written only when `DYNAMO_SESSION_LOG` names a file.
    pub session_log: SessionLogConfig,
    /// Run-level settings (service class family (b), f36b).
    pub runs: RunConfig,
    /// f50: switch between two policies by one class's share of the load
    /// (`mix_select.rs`); absent: off.
    pub mix_select: Option<crate::mix_select::MixSelectConfig>,
}

impl Default for ServicePolicy {
    fn default() -> Self {
        let class = |name: &str,
                     rank,
                     value,
                     service_level,
                     gate_share,
                     wait,
                     on_wait_limit,
                     shed_hold_s| ClassPolicy {
            name: name.to_owned(),
            rank,
            value,
            service_level,
            admission: ClassAdmission {
                gate_share,
                max_lobby_wait_s: wait,
                on_wait_limit,
                shed_hold_s,
                busy_frac: None,
            },
            run_level: None,
        };
        let level = |ttft_s, itl_s, attainment| ServiceLevel {
            ttft_s,
            itl_s,
            percentile: 0.9,
            attainment,
            window_s: 60.0,
        };
        Self {
            mode: Mode::Shadow,
            default_class: "standard".to_owned(),
            classes: vec![
                class(
                    "priority",
                    1,
                    2.0,
                    Some(level(3.0, 0.05, 0.9)),
                    1.0,
                    30.0,
                    WaitLimitAction::Admit,
                    0.0,
                ),
                class(
                    "standard",
                    2,
                    1.0,
                    Some(level(8.0, 0.1, 0.7)),
                    0.9,
                    60.0,
                    WaitLimitAction::Expire,
                    10.0,
                ),
                class(
                    "batch",
                    3,
                    0.5,
                    None,
                    0.75,
                    3600.0,
                    WaitLimitAction::Expire,
                    120.0,
                ),
            ],
            admission: AdmissionPolicy::default(),
            session_table_capacity: 16_384,
            decision_log_capacity: 4_096,
            forecast: ForecastConfig::default(),
            adaptive: None,
            controller: None,
            session_log: SessionLogConfig::default(),
            runs: RunConfig::default(),
            mix_select: None,
        }
    }
}

/// A configuration that cannot be used, and why.
#[derive(Debug, Error, PartialEq)]
pub enum ConfigError {
    #[error("at least one class is required")]
    NoClasses,
    #[error("runs.kind must be interactive or agent, got {kind:?}")]
    UnknownRunKind { kind: String },
    #[error("class {index} has an empty name")]
    EmptyName { index: usize },
    #[error("class {name:?} is defined twice")]
    DuplicateClass { name: String },
    #[error("default_class {name:?} is not one of the classes ({known})")]
    UnknownDefaultClass { name: String, known: String },
    #[error("class {class:?}: {field} must be {expected}, got {value}")]
    OutOfRange {
        class: String,
        field: &'static str,
        expected: &'static str,
        value: f64,
    },
    #[error("{field} must be {expected}, got {value}")]
    PolicyOutOfRange {
        field: &'static str,
        expected: &'static str,
        value: f64,
    },
    #[error("admission.aimd.class {name:?} {reason}")]
    AimdClass { name: String, reason: &'static str },
    #[error("forecast: {0}")]
    Forecast(StepModelError),
    #[error("admission.returnee_first: {0}")]
    ReturneeFirst(String),
    #[error("admission.honest_retry: {0}")]
    HonestRetry(String),
    #[error("admission.class_overcommit: {0}")]
    ClassOvercommit(String),
    #[error("admission.worker_step_budget: {0}")]
    WorkerStepBudget(String),
    #[error("admission.step_target: {0}")]
    StepTarget(String),
    #[error("admission.sla_budget: {0}")]
    SlaBudget(String),
    #[error("admission.session_signal: {0}")]
    SessionSignal(String),
    #[error("admission.starvation_override: {0}")]
    StarvationOverride(String),
    #[error("admission.lobby_order: {0}")]
    LobbyOrder(String),
    #[error("admission.batch_yield: {0}")]
    BatchYield(String),
    #[error("admission.target_loop.rebalance: {0}")]
    TargetLoopRebalance(String),
    #[error("mix_select: {0}")]
    MixSelect(String),
}

impl ServicePolicy {
    pub fn validate(&self) -> Result<(), ConfigError> {
        // f50: the base and each mode's policy must be valid.
        if let Some(m) = &self.mix_select {
            m.validate(self).map_err(ConfigError::MixSelect)?;
            crate::mix_select::mode_policy(self, &m.b0).validate()?;
            crate::mix_select::mode_policy(self, &m.pg).validate()?;
        }
        if self.classes.is_empty() {
            return Err(ConfigError::NoClasses);
        }
        let mut seen = std::collections::HashSet::new();
        for (index, class) in self.classes.iter().enumerate() {
            if class.name.trim().is_empty() {
                return Err(ConfigError::EmptyName { index });
            }
            if !seen.insert(class.name.as_str()) {
                return Err(ConfigError::DuplicateClass {
                    name: class.name.clone(),
                });
            }
            let range = |field, expected, value: f64, ok: bool| {
                if ok {
                    Ok(())
                } else {
                    Err(ConfigError::OutOfRange {
                        class: class.name.clone(),
                        field,
                        expected,
                        value,
                    })
                }
            };
            range(
                "value",
                "positive",
                class.value,
                class.value.is_finite() && class.value > 0.0,
            )?;
            let a = &class.admission;
            range(
                "admission.gate_share",
                "in (0, 1]",
                a.gate_share,
                a.gate_share > 0.0 && a.gate_share <= 1.0,
            )?;
            range(
                "admission.max_lobby_wait_s",
                "positive",
                a.max_lobby_wait_s,
                a.max_lobby_wait_s.is_finite() && a.max_lobby_wait_s > 0.0,
            )?;
            if let Some(level) = &class.service_level {
                range(
                    "service_level.ttft_s",
                    "positive",
                    level.ttft_s,
                    level.ttft_s.is_finite() && level.ttft_s > 0.0,
                )?;
                range(
                    "service_level.itl_s",
                    "positive",
                    level.itl_s,
                    level.itl_s.is_finite() && level.itl_s > 0.0,
                )?;
                range(
                    "service_level.percentile",
                    "in (0, 1]",
                    level.percentile,
                    level.percentile > 0.0 && level.percentile <= 1.0,
                )?;
                range(
                    "service_level.attainment",
                    "in [0, 1]",
                    level.attainment,
                    (0.0..=1.0).contains(&level.attainment),
                )?;
                range(
                    "service_level.window_s",
                    "at least 1",
                    level.window_s,
                    level.window_s >= 1.0,
                )?;
            }
            if let Some(run) = &class.run_level {
                range(
                    "run_level.delay_base_s",
                    "non-negative",
                    run.delay_base_s,
                    run.delay_base_s.is_finite() && run.delay_base_s >= 0.0,
                )?;
                range(
                    "run_level.delay_per_ref",
                    "non-negative",
                    run.delay_per_ref,
                    run.delay_per_ref.is_finite() && run.delay_per_ref >= 0.0,
                )?;
                range(
                    "run_level.delay_window_s",
                    "at least 1",
                    run.delay_window_s,
                    run.delay_window_s.is_finite() && run.delay_window_s >= 1.0,
                )?;
                range(
                    "run_level.slowdown",
                    "at least 1",
                    run.slowdown,
                    run.slowdown.is_finite() && run.slowdown >= 1.0,
                )?;
                range(
                    "run_level.attainment",
                    "in [0, 1]",
                    run.attainment,
                    (0.0..=1.0).contains(&run.attainment),
                )?;
                range(
                    "run_level.server_share",
                    "in [0, 1]",
                    run.server_share,
                    (0.0..=1.0).contains(&run.server_share),
                )?;
                range(
                    "run_level.window_s",
                    "at least 1",
                    run.window_s,
                    run.window_s.is_finite() && run.window_s >= 1.0,
                )?;
            }
        }
        if !self.classes.iter().any(|c| c.name == self.default_class) {
            return Err(ConfigError::UnknownDefaultClass {
                name: self.default_class.clone(),
                known: self
                    .classes
                    .iter()
                    .map(|c| c.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
            });
        }
        let stop_gap = self.session_log.stop_gap_s;
        if !(stop_gap.is_finite() && stop_gap > 0.0) {
            return Err(ConfigError::PolicyOutOfRange {
                field: "session_log.stop_gap_s",
                expected: "positive",
                value: stop_gap,
            });
        }
        let runs = &self.runs;
        for (field, expected, value, ok) in [
            (
                "runs.las_weight",
                "non-negative",
                runs.las_weight,
                runs.las_weight.is_finite() && runs.las_weight >= 0.0,
            ),
            (
                "runs.autonomous_after_s",
                "positive",
                runs.autonomous_after_s,
                runs.autonomous_after_s > 0.0,
            ),
            (
                "runs.floor_share",
                "in (0, 1]",
                runs.floor_share,
                runs.floor_share > 0.0 && runs.floor_share <= 1.0,
            ),
            (
                "runs.floor_window_s",
                "positive",
                runs.floor_window_s,
                runs.floor_window_s.is_finite() && runs.floor_window_s > 0.0,
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
        if let Some(kind) = &runs.kind
            && crate::session::SessionKind::parse(kind).is_none()
        {
            return Err(ConfigError::UnknownRunKind { kind: kind.clone() });
        }
        if let Some(c) = &self.admission.compute_headroom
            && !c.hold_below.is_finite()
        {
            return Err(ConfigError::PolicyOutOfRange {
                field: "admission.compute_headroom.hold_below",
                expected: "finite",
                value: c.hold_below,
            });
        }
        if let Some(b) = &self.admission.prefill_budget {
            if !(b.margin > 0.0 && b.margin <= 1.0) {
                return Err(ConfigError::PolicyOutOfRange {
                    field: "admission.prefill_budget.margin",
                    expected: "in (0, 1]",
                    value: b.margin,
                });
            }
            if b.lookahead_s.is_some_and(|h| !(h.is_finite() && h >= 1.0))
                || !(b.lookahead_step_s.is_finite() && b.lookahead_step_s > 0.0)
            {
                return Err(ConfigError::PolicyOutOfRange {
                    field: "admission.prefill_budget.lookahead_s",
                    expected: "at least 1 (and lookahead_step_s over 0)",
                    value: b.lookahead_s.unwrap_or(b.lookahead_step_s),
                });
            }
            if !(b.window_s.is_finite() && b.window_s >= 1.0) {
                return Err(ConfigError::PolicyOutOfRange {
                    field: "admission.prefill_budget.window_s",
                    expected: "at least 1",
                    value: b.window_s,
                });
            }
        }
        if let Some(l) = &self.admission.target_loop {
            let ok = |v: f64| v.is_finite();
            let checks: [(&'static str, &'static str, f64, bool); 7] = [
                (
                    "admission.target_loop.guard_gain",
                    "non-negative",
                    l.guard_gain,
                    ok(l.guard_gain) && l.guard_gain >= 0.0,
                ),
                (
                    "admission.target_loop.step_s",
                    "over 0",
                    l.step_s,
                    ok(l.step_s) && l.step_s > 0.0,
                ),
                (
                    "admission.target_loop.gain",
                    "non-negative",
                    l.gain,
                    ok(l.gain) && l.gain >= 0.0,
                ),
                (
                    "admission.target_loop.smooth_s",
                    "over 0",
                    l.smooth_s,
                    ok(l.smooth_s) && l.smooth_s > 0.0,
                ),
                (
                    "admission.target_loop.min_scale",
                    "over 0",
                    l.min_scale,
                    ok(l.min_scale) && l.min_scale > 0.0,
                ),
                (
                    "admission.target_loop.max_scale",
                    "at least min_scale",
                    l.max_scale,
                    ok(l.max_scale) && l.max_scale >= l.min_scale,
                ),
                (
                    "admission.target_loop.margin",
                    "finite",
                    l.margin,
                    ok(l.margin) && ok(l.guard),
                ),
            ];
            for (field, expected, value, good) in checks {
                if !good {
                    return Err(ConfigError::PolicyOutOfRange {
                        field,
                        expected,
                        value,
                    });
                }
            }
            if !self.classes.iter().any(|c| c.name == l.class) {
                return Err(ConfigError::PolicyOutOfRange {
                    field: "admission.target_loop.class",
                    expected: "a configured class",
                    value: f64::NAN,
                });
            }
            // f49n.
            if let Some(rb) = &l.rebalance {
                rb.validate(self, &l.class)?;
            }
        }
        let overcommit = self.admission.hot_overcommit;
        if !(overcommit.is_finite() && overcommit >= 1.0) {
            return Err(ConfigError::PolicyOutOfRange {
                field: "admission.hot_overcommit",
                expected: "at least 1",
                value: overcommit,
            });
        }
        // f49k: `class_overcommit` names configured classes ranked below the top rank, each with
        // a finite multiple over 0.
        if !self.admission.class_overcommit.is_empty() {
            let top = self.classes.iter().map(|c| c.rank).min().unwrap_or(0);
            for (name, value) in &self.admission.class_overcommit {
                let Some(class) = self.classes.iter().find(|c| &c.name == name) else {
                    return Err(ConfigError::ClassOvercommit(format!(
                        "{name:?} is not a configured class"
                    )));
                };
                if class.rank <= top {
                    return Err(ConfigError::ClassOvercommit(format!(
                        "{name:?} is a top-ranked class; only classes ranked below it take one"
                    )));
                }
                if !(value.is_finite() && *value > 0.0) {
                    return Err(ConfigError::ClassOvercommit(format!(
                        "{name:?} must be finite and over 0, got {value}"
                    )));
                }
            }
        }
        // f49k2: the mode does nothing without `class_overcommit`.
        if self.admission.class_overcommit_mode.is_some()
            && self.admission.class_overcommit.is_empty()
        {
            return Err(ConfigError::ClassOvercommit(
                "class_overcommit_mode is set without class_overcommit".into(),
            ));
        }
        let refresh = self.admission.accounting.refresh_s;
        if !(refresh.is_finite() && refresh > 0.0) {
            return Err(ConfigError::PolicyOutOfRange {
                field: "admission.accounting.refresh_s",
                expected: "over 0",
                value: refresh,
            });
        }
        if let Some(v) = &self.admission.accounting.verify
            && !(v.min_drop > 0.0 && v.min_drop <= 1.0)
        {
            return Err(ConfigError::PolicyOutOfRange {
                field: "admission.accounting.verify.min_drop",
                expected: "in (0, 1]",
                value: v.min_drop,
            });
        }
        let headroom = self.admission.total_headroom;
        if !(headroom > 0.0 && headroom <= 1.0) {
            return Err(ConfigError::PolicyOutOfRange {
                field: "admission.total_headroom",
                expected: "in (0, 1]",
                value: headroom,
            });
        }
        for (field, value) in [
            ("session_table_capacity", self.session_table_capacity),
            ("decision_log_capacity", self.decision_log_capacity),
        ] {
            if value == 0 {
                return Err(ConfigError::PolicyOutOfRange {
                    field,
                    expected: "at least 1",
                    value: 0.0,
                });
            }
        }
        if let Some(h) = &self.admission.forecast_harm_new
            && !(h.max_added_misses.is_finite() && h.max_added_misses >= 0.0)
        {
            return Err(ConfigError::PolicyOutOfRange {
                field: "admission.forecast_harm_new.max_added_misses",
                expected: "finite and non-negative",
                value: h.max_added_misses,
            });
        }
        // f49f: `returnee_first` needs `tickets` for the same classes.
        if let Some(r) = &self.admission.returnee_first {
            let Some(tickets) = &self.admission.tickets else {
                return Err(ConfigError::ReturneeFirst(
                    "needs admission.tickets for the same classes, and admission.tickets is absent"
                        .to_owned(),
                ));
            };
            for class in &r.classes {
                if !self.classes.iter().any(|c| &c.name == class) {
                    return Err(ConfigError::ReturneeFirst(format!(
                        "class {class:?} is not one of the classes"
                    )));
                }
                if !tickets.classes.contains(class) {
                    return Err(ConfigError::ReturneeFirst(format!(
                        "class {class:?} is not in admission.tickets.classes ({}); returnee_first needs admission.tickets for the same classes",
                        tickets.classes.join(", ")
                    )));
                }
            }
            if let Some(m) = r.max_block_s
                && !(m.is_finite() && m > 0.0)
            {
                return Err(ConfigError::PolicyOutOfRange {
                    field: "admission.returnee_first.max_block_s",
                    expected: "positive",
                    value: m,
                });
            }
            if !(r.due_soon_s.is_finite() && r.due_soon_s >= 0.0) {
                return Err(ConfigError::PolicyOutOfRange {
                    field: "admission.returnee_first.due_soon_s",
                    expected: "finite and non-negative",
                    value: r.due_soon_s,
                });
            }
        }
        // f49m: `honest_retry` needs `tickets`, and its ranges.
        if let Some(honest) = &self.admission.honest_retry {
            honest.validate(self)?;
        }
        // f49i: the step budget's classes need service levels, and the forecast must be on.
        if let Some(budget) = &self.admission.worker_step_budget {
            budget.validate(self)?;
        }
        if let Some(aimd) = &self.admission.aimd {
            aimd.validate(self)?;
        }
        // f49j: the step target's class needs a service level; the gains and bounds their ranges.
        if let Some(step_target) = &self.admission.step_target {
            step_target.validate(self)?;
        }
        // f49l: the SLA loop's rates, ranges and class names.
        if let Some(sla_budget) = &self.admission.sla_budget {
            sla_budget.validate(self)?;
        }
        // f49o: batch yield's classes and ranges.
        if let Some(batch_yield) = &self.admission.batch_yield {
            batch_yield.validate(self)?;
        }
        // f49q: the session signal's ranges and classes, the override's, and `lobby_order: signal`
        // not with the turn orders that set the same key.
        if let Some(signal) = &self.admission.session_signal {
            signal.validate(self)?;
        }
        if let Some(over) = &self.admission.starvation_override {
            over.validate(self)?;
        }
        if self.admission.lobby_order == LobbyOrder::Signal
            && matches!(
                self.admission.turn_order,
                TurnOrder::RunLas | TurnOrder::ServerWaitDeficit
            )
        {
            return Err(ConfigError::LobbyOrder(
                "signal orders by the key turn_order run_las and server_wait_deficit set; use one"
                    .to_owned(),
            ));
        }
        self.forecast.validate().map_err(ConfigError::Forecast)?;
        Ok(())
    }

    /// f49k `admission.class_overcommit`: the multiple of `class`'s hot budget; 1 when it is not
    /// listed or is ranked at the top. The cap the gate uses is [`crate::admit::hot_cap`] (f49k2).
    pub fn class_overcommit(&self, class: &ClassPolicy) -> f64 {
        if self.admission.class_overcommit.is_empty() {
            return 1.0;
        }
        let top = self.classes.iter().map(|c| c.rank).min().unwrap_or(0);
        if class.rank <= top {
            return 1.0;
        }
        self.admission
            .class_overcommit
            .get(&class.name)
            .copied()
            .unwrap_or(1.0)
    }

    /// The class named `name`, or the default class.
    pub fn class(&self, name: Option<&str>) -> &ClassPolicy {
        name.and_then(|n| self.classes.iter().find(|c| c.name == n))
            .or_else(|| self.classes.iter().find(|c| c.name == self.default_class))
            .unwrap_or(&self.classes[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_simulator_and_validate() {
        let policy = ServicePolicy::default();
        policy.validate().unwrap();
        let p = policy.class(Some("priority"));
        let level = p.service_level.as_ref().unwrap();
        assert_eq!(
            (
                level.ttft_s,
                level.itl_s,
                level.attainment,
                level.percentile
            ),
            (3.0, 0.05, 0.9, 0.9)
        );
        let s = policy
            .class(Some("standard"))
            .service_level
            .as_ref()
            .unwrap();
        assert_eq!((s.ttft_s, s.itl_s, s.attainment), (8.0, 0.1, 0.7));
        assert!(policy.class(Some("batch")).service_level.is_none());
        assert_eq!(policy.class(Some("nope")).name, "standard");
        assert_eq!(policy.mode, Mode::Shadow);
    }

    #[test]
    fn write_through_dram_adds_only_what_it_holds_beyond_hbm() {
        let mut a = AdmissionPolicy {
            dram_tokens_per_worker: 50,
            ..AdmissionPolicy::default()
        };
        // 4 workers, 80 HBM tokens in all, 200 DRAM tokens in all.
        assert_eq!(a.dram_beyond_hbm(80, 4), 200);
        a.dram_write_through = true;
        assert_eq!(a.dram_beyond_hbm(80, 4), 120);
        assert_eq!(a.dram_beyond_hbm(300, 4), 0);
        // The tier list: an inclusive DRAM tier of 50 a worker (200) over HBM 80 adds 120, as
        // write-through does; an exclusive SSD tier of 100 a worker adds all 400; a second
        // inclusive tier larger than DRAM (a shared store of 75 a worker, 300) replaces it.
        let mut t = AdmissionPolicy {
            tiers: vec![KvTier {
                name: "dram".into(),
                tokens_per_worker: 50,
                inclusive: true,
                index_tier: None,
            }],
            ..Default::default()
        };
        assert_eq!(t.dram_beyond_hbm(80, 4), 120);
        t.tiers.push(KvTier {
            name: "ssd".into(),
            tokens_per_worker: 100,
            inclusive: false,
            index_tier: None,
        });
        assert_eq!(t.dram_beyond_hbm(80, 4), 520);
        t.tiers.push(KvTier {
            name: "store".into(),
            tokens_per_worker: 75,
            inclusive: true,
            index_tier: None,
        });
        assert_eq!(t.dram_beyond_hbm(80, 4), 620);
    }

    /// f49f: `returnee_first` parses, defaults off, and needs `tickets` for its classes.
    #[test]
    fn returnee_first_needs_tickets_for_its_classes() {
        let policy: ServicePolicy = serde_json::from_str(r#"{"mode": "enforce"}"#).unwrap();
        assert!(policy.admission.returnee_first.is_none());
        let policy: ServicePolicy = serde_json::from_str(
            r#"{"admission": {"returnee_first": {"classes": ["priority", "standard"]},
                "tickets": {"max_hold_frac": 0.0, "per_root": true, "keep_on_rerefusal": true}}}"#,
        )
        .unwrap();
        let r = policy.admission.returnee_first.as_ref().unwrap();
        assert_eq!((r.classes.len(), r.max_block_s), (2, None));
        policy.validate().unwrap();
        let mut bad = policy.clone();
        bad.admission.tickets = None;
        assert_eq!(
            bad.validate().unwrap_err().to_string(),
            "admission.returnee_first: needs admission.tickets for the same classes, and admission.tickets is absent"
        );
        let mut bad = policy.clone();
        bad.admission.tickets.as_mut().unwrap().classes = vec!["priority".into()];
        assert_eq!(
            bad.validate().unwrap_err().to_string(),
            "admission.returnee_first: class \"standard\" is not in admission.tickets.classes (priority); returnee_first needs admission.tickets for the same classes"
        );
        let mut bad = policy.clone();
        bad.admission.returnee_first.as_mut().unwrap().classes = vec!["gold".into()];
        assert!(matches!(bad.validate(), Err(ConfigError::ReturneeFirst(_))));
        let mut bad = policy;
        bad.admission.returnee_first.as_mut().unwrap().max_block_s = Some(0.0);
        assert!(matches!(
            bad.validate(),
            Err(ConfigError::PolicyOutOfRange { .. })
        ));
        assert!(
            serde_json::from_str::<ServicePolicy>(
                r#"{"admission": {"returnee_first": {"clases": []}}}"#
            )
            .is_err()
        );
    }

    /// f49f2: `only_when_waiting` and `due_soon_s` parse, default off and 10, stay out of a
    /// serialized default, and `due_soon_s` is checked.
    #[test]
    fn returnee_first_only_when_waiting_parses_and_validates() {
        let policy: ServicePolicy = serde_json::from_str(
            r#"{"admission": {"returnee_first": {"classes": ["priority", "standard"],
                "max_block_s": 180, "only_when_waiting": true, "due_soon_s": 10},
                "tickets": {"max_hold_frac": 0.0, "per_root": true, "keep_on_rerefusal": true}}}"#,
        )
        .unwrap();
        let r = policy.admission.returnee_first.as_ref().unwrap();
        assert_eq!(
            (r.max_block_s, r.only_when_waiting, r.due_soon_s),
            (Some(180.0), true, 10.0)
        );
        policy.validate().unwrap();
        let d = ReturneeFirstConfig::default();
        assert_eq!((d.only_when_waiting, d.due_soon_s), (false, 10.0));
        assert_eq!(
            serde_json::to_string(&d).unwrap(),
            r#"{"classes":["priority","standard"]}"#
        );
        let mut bad = policy;
        bad.admission.returnee_first.as_mut().unwrap().due_soon_s = -1.0;
        assert!(matches!(
            bad.validate(),
            Err(ConfigError::PolicyOutOfRange { .. })
        ));
    }

    #[test]
    fn errors_say_what_is_wrong() {
        let policy = ServicePolicy {
            default_class: "gold".into(),
            ..Default::default()
        };
        assert_eq!(
            policy.validate().unwrap_err().to_string(),
            "default_class \"gold\" is not one of the classes (priority, standard, batch)"
        );
        let mut policy = ServicePolicy::default();
        policy.classes[1].admission.gate_share = 1.5;
        assert_eq!(
            policy.validate().unwrap_err().to_string(),
            "class \"standard\": admission.gate_share must be in (0, 1], got 1.5"
        );
        let mut policy = ServicePolicy::default();
        policy.classes.push(policy.classes[0].clone());
        assert!(matches!(
            policy.validate(),
            Err(ConfigError::DuplicateClass { .. })
        ));
    }

    #[test]
    fn partial_parameters_fill_from_defaults_and_unknown_keys_fail() {
        let policy: ServicePolicy = serde_json::from_str(r#"{"mode": "enforce"}"#).unwrap();
        assert_eq!(policy.mode, Mode::Enforce);
        assert_eq!(policy.classes.len(), 3);
        assert!(serde_json::from_str::<ServicePolicy>(r#"{"mod": "enforce"}"#).is_err());
    }

    #[test]
    fn run_level_settings_parse_default_off_and_validate() {
        let off = ServicePolicy::default();
        assert_eq!(off.admission.turn_order, TurnOrder::Fifo);
        assert_eq!(off.runs.headroom, HeadroomSource::Request);
        assert!(off.admission.hot_budget.on() && off.admission.compute_headroom.is_none());
        assert!(off.classes.iter().all(|c| c.run_level.is_none()));
        assert_eq!(off.session_log.stop_gap_s, 180.0);
        let policy: ServicePolicy = serde_json::from_str(
            r#"{"mode": "enforce",
                "classes": [{"name": "standard", "rank": 1, "value": 1.0,
                             "admission": {"gate_share": 0.9, "max_lobby_wait_s": 60,
                                           "on_wait_limit": "expire", "shed_hold_s": 0},
                             "run_level": {"slowdown": 3, "attainment": 0.95, "server_share": 0.6}}],
                "admission": {"turn_order": "run_las", "hot_budget": false,
                              "compute_headroom": {"hold_below": 0.02}},
                "runs": {"headroom": "min", "kind": "agent", "las_weight": 5},
                "session_log": {"max_bursts": 8}}"#,
        )
        .unwrap();
        policy.validate().unwrap();
        assert_eq!(policy.admission.turn_order, TurnOrder::RunLas);
        assert_eq!(policy.runs.headroom, HeadroomSource::Min);
        assert_eq!(
            policy.classes[0].run_level.as_ref().unwrap().window_s,
            1800.0
        );
        assert_eq!(policy.session_log.max_runs, 8);
        let mut bad = policy.clone();
        bad.runs.kind = Some("robot".into());
        assert!(matches!(
            bad.validate(),
            Err(ConfigError::UnknownRunKind { .. })
        ));
        let mut bad = policy.clone();
        bad.classes[0].run_level.as_mut().unwrap().slowdown = 0.5;
        assert!(matches!(
            bad.validate(),
            Err(ConfigError::OutOfRange { .. })
        ));
        let mut bad = policy;
        bad.runs.floor_share = 0.0;
        assert!(matches!(
            bad.validate(),
            Err(ConfigError::PolicyOutOfRange { .. })
        ));
    }
}
