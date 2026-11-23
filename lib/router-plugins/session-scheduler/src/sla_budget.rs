// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! f49l `admission.sla_budget`: an outer SLA control loop on the gate's hot budget, ported from the
//! admission simulator's `slaPaced: 'budget'` (`web/sim.js` `budgetPace`). Off (absent) by default.
//!
//! ```yaml
//! admission:
//!   sla_budget:
//!     interval_s: 60
//!     smooth_s: 300
//!     blend: 0.5
//!     margin: {priority: 0.02, standard: 0.03}   # a class not listed: 0.03
//!     grow_per_min: 0.01
//!     shrink_per_min: 0.05
//!     shrink_gain: 2
//!     floors: {standard: 0.5, batch: 0.333}       # a lower class not listed: 0.5
//!     ceilings: {standard: 2.5, batch: 2.0}       # a lower class not listed: 2.0
//!     max_headroom: 0.15
//!     setpoint: {min: 0.6, max: 0.95, grow_per_min: 0.01, shrink_per_min: 0.05}
//!     fallback: {after_s: 600, floor: 0.3}
//! ```
//!
//! Every `interval_s` (stepped from the gate's checks, like the other admission loops):
//!
//! 1. Measure, per class with a service level, the share of requests meeting both limits (TTFT
//!    within `ttft_s` and mean ITL within `itl_s`: the scorer's per-request rule, as
//!    [`crate::metrics::Metrics::window_share`]) over the class's trailing `window_s`. The
//!    top-ranked classes are judged on all offered work: the new sessions refused in that window
//!    (expired in the lobby, [`crate::scheduler::Scheduler::note_lost`]) count as misses. The
//!    classes below are judged on admitted work only. Under 5 counted: no measurement.
//!    Forecast: the plugin's forecast prices single decisions (misses added), it predicts no class
//!    attainment, so the scheduler passes none ([`ClassInput::forecast`]) and the signal is the
//!    measurement alone. With both: `signal = blend x forecast + (1 - blend) x measured`, each an
//!    EMA with time constant `smooth_s`.
//! 2. `head(c) = signal(c) - (target(c) + margin(c))`. A lower class's headroom `h(c)` is the
//!    smallest head over the classes ranked at or above it (0.02 when none has one, as the
//!    simulator).
//! 3. Share `K[c]` per class ranked below the top, from 1, multiplying its hot budget (the gate's
//!    share scale, so it enters `admit.rs` `fits` / `room_scaled` with pacing, the step target and
//!    `class_overcommit`):
//!    - any head under 0 (`short`, the smallest head): the lowest class with `K` over its floor
//!      gives way, `K *= exp(shrink_gain x shrink_per_min / 60 x short / 0.10 x dt)`, at least its
//!      floor. The simulator's `exp(2 x 2 / 60 x short x dt)` gives 20 %/min at short -0.05; here
//!      the defaults give 5 %/min at short -0.05 and 10 %/min at -0.10 (`SHRINK_REF`).
//!    - otherwise each lower class with demand (a session of it in the lobby, or one of it refused
//!      since the last step) whose own head is at most `max_headroom` grows,
//!      `K *= exp(grow_per_min / 60 x h / 0.01 x dt)`, at most its ceiling: `grow_per_min` (1 %/min)
//!      per point of headroom (the simulator's is 2 %/min per point).
//!    - no inversion: a top class's hot budget is never under a lower class's
//!      ([`crate::scheduler::Scheduler::sla_scale`]).
//! 4. With `admission.step_target`, its `target_frac` (the setpoint) is owned here, within
//!    `[setpoint.min, setpoint.max]` from the configured value: `x exp(setpoint.grow_per_min / 60 x
//!    dt)` while the top class's head is over 0, `x exp(-setpoint.shrink_per_min / 60 x dt)` while
//!    it is short and every lower `K` is at its floor. Shrink order: the lowest class's `K`, up to
//!    the highest lower class's, then the setpoint.
//! 5. Fallback: the top class short for `fallback.after_s` on end with every lower `K` at its floor
//!    and the setpoint at its min (or no step target) enters priority control: the top class's own
//!    share `K_p` (from 1, at least `fallback.floor`) shrinks and grows by the same rules on its
//!    head (growing needs no demand: it gives back the configured share). Its sessions that do not
//!    fit wait and are refused at the class's wait limit as usual (with `on_wait_limit: expire`).
//!    The lower `K` and the setpoint hold meanwhile. It ends once `K_p` is back at 1 and the head
//!    is over 0.
//! 6. `DYNAMO_SLA_LOG=<path>`: one JSON line per class per step, `{t, class, measured, forecast,
//!    signal, head, K, setpoint, mode}`: `measured` and `forecast` the step's raw inputs, `signal`
//!    the smoothed blend, `K` the class's share (`K_p` for a top class), `mode` `normal` or
//!    `priority`, and (f49l2) `warm`: whether the class's lever may move at that step (8).
//!
//! f49l2 (absent, the default: f49l exactly):
//!
//! ```yaml
//! admission:
//!   sla_budget:
//!     max_shrink_per_min: 0.10                   # absent: no cap
//!     max_grow_per_min: 0.10                     # absent: no cap
//!     warmup: {min_scored: 50, after_s: 1200}    # absent: no warm-up
//! ```
//!
//! 7. Rate caps. `max_shrink_per_min` bounds every shrink, whatever the shortfall: a lower class's
//!    `K` (3), the setpoint (4) and `K_p` (5). A step's multiplier is at least
//!    `exp(-max_shrink_per_min / 60 x dt)`: 0.10 is about 10 %/min, where 75 points short is about
//!    75 %/min uncapped (`shrink_gain x shrink_per_min x 7.5`). `max_grow_per_min` bounds growth
//!    the same way: at most `exp(max_grow_per_min / 60 x dt)`.
//! 8. Warm-up. A class is warm once BOTH hold: it has at least `min_scored` scored turns
//!    (completions, [`ClassInput::counts`]) in its measurement window, and the scheduler time is
//!    at least `after_s`. So warm-up lasts until `min_scored` is reached or `after_s` has passed,
//!    whichever is later. A class without a service level has no turns to score and needs only
//!    the time. Warm is latched: a class once warm stays warm (a start-up rule). Until then:
//!    - its head steers nothing (it is still measured, smoothed and logged: the EMAs go on);
//!    - a lower class's `K` moves only with it and the top classes warm; the setpoint and `K_p`
//!      follow the top class's head, so they hold too. While the top class warms up nothing moves
//!      (no shrink, no growth, no fallback clock);
//!    - a held lower class that is next to give way holds the step's shrink (the class above it
//!      does not give way in its place, and the setpoint does not move).
//!
//! f49l3 `mode: rebalance` (absent, the default: all of the above exactly):
//!
//! ```yaml
//! admission:
//!   sla_budget:
//!     mode: rebalance
//!     floors: {priority: 0.3, standard: 0.5, batch: 0.333}   # priority unlisted: fallback.floor
//!     ceilings: {priority: 1.5, standard: 2.5, batch: 2.0}   # priority unlisted: 1.5
//!     rebalance: {shrink_per_min: 0.02, grow_per_min: 0.01, full_at: 0.05, exit_after_s: 300}
//! ```
//!
//! 9. Every class has its own `K` (the top classes share `K_p`), from 1, within its floor and
//!    ceiling. Heads are judged per step among the warm classes (8): `clear` (none short),
//!    `feasible` (some short, some over target) or `infeasible` (every class with a head short).
//! 10. Feasible: each class over target shrinks `x exp(-rebalance.shrink_per_min / 60 x
//!     min(head / full_at, 1) x dt)`, each short class grows `x exp(rebalance.grow_per_min / 60 x
//!     min(-head / full_at, 1) x dt)`, whatever its rank; a class without a service level gives
//!     way at the worst shortfall's scale. A short class never shrinks.
//! 11. Infeasible: rank order. The lowest-ranked class above its floor gives way (the shrink
//!     above, at the scale of the worst head ranked above it, the top classes' on admitted work)
//!     while one ranked above it is short, then the next. The top class is no donor: `K_p` moves
//!     on admitted priority alone, up while over target (to its ceiling), down while short but not
//!     under 1. Clear: (3)'s growth with demand.
//! 12. With `admission.step_target` each class with a service level has its own setpoint in
//!     `[setpoint.min, setpoint.max]` from `target_frac`: a short class tightens its own, one over
//!     target loosens its own (the setpoint rates, the caps of 7). A worker's target is the
//!     tightest `setpoint[c] x itl_s` among the classes decoding on it
//!     ([`crate::step_target::StepTarget::set_class_fracs`]); `setpoint` is the top class's.
//! 13. No inversion for every class: its hot cap is never under a lower-ranked one's
//!     ([`crate::scheduler::Scheduler::sla_scale`]).
//! 14. Fallback: infeasible with every lower `K` at its floor and the top setpoint at its min, and
//!     priority short on ADMITTED work, for `fallback.after_s`. `K_p` then steps on admitted
//!     priority: down while short (to its floor), up while over (to its ceiling). The top class's
//!     refusals made while it holds `K_p` under 1 are not misses for its control measure. It ends
//!     once admitted priority has been over target for `rebalance.exit_after_s`; `K_p` then
//!     recovers to 1 at the capped grow rate (`max_grow_per_min`, else `rebalance.grow_per_min`)
//!     while admitted priority is not short.
//! 15. The log line gains `measured_admitted`, `measured_offered`, `head_admitted`, `case` and
//!     `recovering`; `measured` is the control measure, `setpoint` the class's own.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::io::Write;

use serde::{Deserialize, Serialize};

use crate::clock::Time;
use crate::config::{ConfigError, ServicePolicy};
use crate::metrics::MIN_WINDOW_COMPLETIONS;

/// f49l3 `mode: rebalance`.
#[path = "sla_rebalance.rs"]
mod rebalance;
pub use rebalance::Case;

/// The environment variable naming the step log's file.
pub const SLA_LOG_ENV: &str = "DYNAMO_SLA_LOG";

/// Headroom at which a share grows at `grow_per_min`: one point.
const GROW_REF: f64 = 0.01;

/// Shortfall at which a share shrinks at `shrink_gain x shrink_per_min`: ten points (with the
/// default gain of 2, 5 %/min at five points short).
const SHRINK_REF: f64 = 0.10;

/// A lower class's headroom when no class at or above it has one (the simulator's).
const NO_HEAD: f64 = 0.02;

/// `admission.sla_budget` (f49l). Off (absent) by default.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SlaBudgetConfig {
    /// Seconds between steps. Default 60.
    pub interval_s: f64,
    /// Time constant of the measurement's and the forecast's EMAs, seconds. Default 300.
    pub smooth_s: f64,
    /// The forecast's weight in the signal, in [0, 1]. Default 0.5.
    pub blend: f64,
    /// Points over target a class keeps, by class (not listed: 0.03). Default priority 0.02,
    /// standard 0.03.
    pub margin: BTreeMap<String, f64>,
    /// Growth per minute per point of headroom. Default 0.01.
    pub grow_per_min: f64,
    /// Shrink per minute at `SHRINK_REF / shrink_gain` short. Default 0.05.
    pub shrink_per_min: f64,
    /// Default 2.
    pub shrink_gain: f64,
    /// Lowest share by lower class, in (0, 1] (not listed: 0.5). Default standard 0.5, batch 0.333.
    pub floors: BTreeMap<String, f64>,
    /// Highest share by lower class, at least 1 (not listed: 2). Default standard 2.5, batch 2.
    pub ceilings: BTreeMap<String, f64>,
    /// A lower class whose own head is over this does not grow. Default 0.15.
    pub max_headroom: f64,
    pub setpoint: SetpointConfig,
    pub fallback: FallbackConfig,
    /// f49l2: the most a step shrinks a lower class's `K`, the setpoint or `K_p`, per minute, in
    /// (0, 1]: the step's multiplier is at least `exp(-max_shrink_per_min / 60 x dt)`. Absent
    /// (default): no cap.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_shrink_per_min: Option<f64>,
    /// f49l2: the most a step grows them, per minute, positive. Absent (default): no cap.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_grow_per_min: Option<f64>,
    /// f49l2: hold the levers while their classes warm up. Absent (default): no warm-up.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmup: Option<WarmupConfig>,
    /// f49l3: `rebalance` steers every class's share, priority's too. Absent (default): f49l.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<BudgetMode>,
    /// f49l3: the rebalance mode's rates (needs `mode: rebalance`). Absent: the defaults.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rebalance: Option<RebalanceConfig>,
}

impl Default for SlaBudgetConfig {
    fn default() -> Self {
        let map = |pairs: &[(&str, f64)]| {
            pairs
                .iter()
                .map(|(k, v)| ((*k).to_owned(), *v))
                .collect::<BTreeMap<_, _>>()
        };
        Self {
            interval_s: 60.0,
            smooth_s: 300.0,
            blend: 0.5,
            margin: map(&[("priority", 0.02), ("standard", 0.03)]),
            grow_per_min: 0.01,
            shrink_per_min: 0.05,
            shrink_gain: 2.0,
            floors: map(&[("standard", 0.5), ("batch", 0.333)]),
            ceilings: map(&[("standard", 2.5), ("batch", 2.0)]),
            max_headroom: 0.15,
            setpoint: SetpointConfig::default(),
            fallback: FallbackConfig::default(),
            max_shrink_per_min: None,
            max_grow_per_min: None,
            warmup: None,
            mode: None,
            rebalance: None,
        }
    }
}

/// `sla_budget.setpoint`: the step target's `target_frac` range and rates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SetpointConfig {
    pub min: f64,
    pub max: f64,
    pub grow_per_min: f64,
    pub shrink_per_min: f64,
}

impl Default for SetpointConfig {
    fn default() -> Self {
        Self {
            min: 0.6,
            max: 0.95,
            grow_per_min: 0.01,
            shrink_per_min: 0.05,
        }
    }
}

/// `sla_budget.fallback`: priority control.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct FallbackConfig {
    /// Seconds the top class stays short with everything else pinned before priority control.
    pub after_s: f64,
    /// Lowest share of the top class's hot budget, in (0, 1].
    pub floor: f64,
}

impl Default for FallbackConfig {
    fn default() -> Self {
        Self {
            after_s: 600.0,
            floor: 0.3,
        }
    }
}

/// `sla_budget.warmup` (f49l2): a class is warm once BOTH hold (warm-up lasts until `min_scored`
/// is reached or `after_s` has passed, whichever is later). Both keys are required.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WarmupConfig {
    /// Scored turns (completions) the class needs in its measurement window, at least 1.
    pub min_scored: u32,
    /// Scheduler seconds that must have passed, positive.
    pub after_s: f64,
}

impl WarmupConfig {
    /// f49l2: whether a class is past its warm-up at scheduler time `t_s` (seconds from the
    /// origin): BOTH `t_s >= after_s` and at least `min_scored` scored turns (`scored`, the
    /// window's completions), so whichever comes later. A class without a service level has no
    /// turns to score and needs only the time.
    pub fn is_warm(&self, t_s: f64, has_service_level: bool, scored: Option<usize>) -> bool {
        t_s >= self.after_s
            && (!has_service_level
                || scored.is_some_and(|n| u32::try_from(n).unwrap_or(u32::MAX) >= self.min_scored))
    }
}

/// `sla_budget.mode` (f49l3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetMode {
    /// Every class's share, by surplus and shortfall, rank order when nothing is spare.
    Rebalance,
}

/// A top class's ceiling when `ceilings` does not list it (f49l3).
pub const TOP_CEILING: f64 = 1.5;

/// `sla_budget.rebalance` (f49l3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RebalanceConfig {
    /// A class over target shrinks this much a minute at `full_at` over. Default 0.02.
    pub shrink_per_min: f64,
    /// A short class grows this much a minute at `full_at` under. Default 0.01.
    pub grow_per_min: f64,
    /// The head from which a class moves at the full rate (proportionally under it). Default 0.05.
    pub full_at: f64,
    /// The fallback ends once admitted priority has been over target this long, s. Default 300.
    pub exit_after_s: f64,
}

impl Default for RebalanceConfig {
    fn default() -> Self {
        Self {
            shrink_per_min: 0.02,
            grow_per_min: 0.01,
            full_at: 0.05,
            exit_after_s: 300.0,
        }
    }
}

/// f49l2: the per-minute rate caps of [`SlaBudgetConfig`] (`None`: no cap).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RateCaps {
    pub max_shrink_per_min: Option<f64>,
    pub max_grow_per_min: Option<f64>,
}

impl RateCaps {
    /// A step's multiplier `mult` over `dt` seconds, within `[exp(-max_shrink_per_min / 60 x dt),
    /// exp(max_grow_per_min / 60 x dt)]`; without caps, `[0, inf]`: `mult` itself.
    pub fn cap(self, mult: f64, dt: f64) -> f64 {
        let lo = self
            .max_shrink_per_min
            .map_or(0.0, |m| (-m / 60.0 * dt).exp());
        let hi = self
            .max_grow_per_min
            .map_or(f64::INFINITY, |m| (m / 60.0 * dt).exp());
        mult.clamp(lo, hi)
    }
}

impl SlaBudgetConfig {
    /// f49l2: the rate caps.
    pub fn rate_caps(&self) -> RateCaps {
        RateCaps {
            max_shrink_per_min: self.max_shrink_per_min,
            max_grow_per_min: self.max_grow_per_min,
        }
    }

    /// f49l2: whether a class is past its warm-up (always without `warmup`); see
    /// [`WarmupConfig::is_warm`].
    pub fn is_warm(&self, t_s: f64, has_service_level: bool, scored: Option<usize>) -> bool {
        self.warmup
            .as_ref()
            .is_none_or(|w| w.is_warm(t_s, has_service_level, scored))
    }

    /// f49l3: whether `mode: rebalance` is set.
    pub fn rebalancing(&self) -> bool {
        self.mode == Some(BudgetMode::Rebalance)
    }

    /// f49l3: the rebalance rates (the defaults when `rebalance` is absent).
    pub fn rebalance_config(&self) -> RebalanceConfig {
        self.rebalance.clone().unwrap_or_default()
    }

    /// f49l3: `class`'s lowest share in the rebalance mode (a top class unlisted: `fallback.floor`).
    pub fn rb_floor(&self, class: &str, top: bool) -> f64 {
        match self.floors.get(class) {
            Some(v) => *v,
            None if top => self.fallback.floor,
            None => self.floor(class),
        }
    }

    /// f49l3: `class`'s highest share in the rebalance mode (a top class unlisted: [`TOP_CEILING`]).
    pub fn rb_ceiling(&self, class: &str, top: bool) -> f64 {
        match self.ceilings.get(class) {
            Some(v) => *v,
            None if top => TOP_CEILING,
            None => self.ceiling(class),
        }
    }

    /// Points over target `class` keeps.
    pub fn margin(&self, class: &str) -> f64 {
        self.margin.get(class).copied().unwrap_or(0.03)
    }

    /// Lowest share of lower class `class`.
    pub fn floor(&self, class: &str) -> f64 {
        self.floors.get(class).copied().unwrap_or(0.5)
    }

    /// Highest share of lower class `class`.
    pub fn ceiling(&self, class: &str) -> f64 {
        self.ceilings.get(class).copied().unwrap_or(2.0)
    }

    pub fn validate(&self, policy: &ServicePolicy) -> Result<(), ConfigError> {
        let fail = |reason: String| Err(ConfigError::SlaBudget(reason));
        let positive = |x: f64| x.is_finite() && x > 0.0;
        let rate = |x: f64| x.is_finite() && x >= 0.0;
        let (sp, fb) = (&self.setpoint, &self.fallback);
        for (field, expected, value, ok) in [
            (
                "interval_s",
                "positive",
                self.interval_s,
                positive(self.interval_s),
            ),
            (
                "smooth_s",
                "positive",
                self.smooth_s,
                positive(self.smooth_s),
            ),
            (
                "blend",
                "in [0, 1]",
                self.blend,
                (0.0..=1.0).contains(&self.blend),
            ),
            (
                "grow_per_min",
                "finite and non-negative",
                self.grow_per_min,
                rate(self.grow_per_min),
            ),
            (
                "shrink_per_min",
                "finite and non-negative",
                self.shrink_per_min,
                rate(self.shrink_per_min),
            ),
            (
                "shrink_gain",
                "positive",
                self.shrink_gain,
                positive(self.shrink_gain),
            ),
            (
                "max_headroom",
                "finite",
                self.max_headroom,
                self.max_headroom.is_finite(),
            ),
            (
                "setpoint.min",
                "in (0, 1]",
                sp.min,
                positive(sp.min) && sp.min <= 1.0,
            ),
            (
                "setpoint.max",
                "in [setpoint.min, 1]",
                sp.max,
                sp.max >= sp.min && sp.max <= 1.0,
            ),
            (
                "setpoint.grow_per_min",
                "finite and non-negative",
                sp.grow_per_min,
                rate(sp.grow_per_min),
            ),
            (
                "setpoint.shrink_per_min",
                "finite and non-negative",
                sp.shrink_per_min,
                rate(sp.shrink_per_min),
            ),
            (
                "fallback.after_s",
                "finite and non-negative",
                fb.after_s,
                rate(fb.after_s),
            ),
            (
                "fallback.floor",
                "in (0, 1]",
                fb.floor,
                positive(fb.floor) && fb.floor <= 1.0,
            ),
        ] {
            if !ok {
                return fail(format!("{field} must be {expected}, got {value}"));
            }
        }
        // f49l2.
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
        // f49l3.
        if self.rebalance.is_some() && !self.rebalancing() {
            return fail("rebalance needs mode: rebalance".to_owned());
        }
        if self.rebalancing() {
            let r = self.rebalance_config();
            for (field, expected, value, ok) in [
                (
                    "shrink_per_min",
                    "finite and non-negative",
                    r.shrink_per_min,
                    rate(r.shrink_per_min),
                ),
                (
                    "grow_per_min",
                    "finite and non-negative",
                    r.grow_per_min,
                    rate(r.grow_per_min),
                ),
                ("full_at", "positive", r.full_at, positive(r.full_at)),
                (
                    "exit_after_s",
                    "finite and non-negative",
                    r.exit_after_s,
                    rate(r.exit_after_s),
                ),
            ] {
                if !ok {
                    return fail(format!("rebalance.{field} must be {expected}, got {value}"));
                }
            }
        }
        if !policy.classes.iter().any(|c| c.service_level.is_some()) {
            return fail("no class has a service_level to steer by".to_owned());
        }
        for (name, value) in &self.margin {
            if !policy.classes.iter().any(|c| &c.name == name) {
                return fail(format!("margin: {name:?} is not a configured class"));
            }
            if !(value.is_finite() && (0.0..1.0).contains(value)) {
                return fail(format!("margin.{name} must be in [0, 1), got {value}"));
            }
        }
        let top = policy.classes.iter().map(|c| c.rank).min().unwrap_or(0);
        let lower = |field: &str, name: &str| -> Result<(), ConfigError> {
            match policy.classes.iter().find(|c| c.name == name) {
                None => fail(format!("{field}: {name:?} is not a configured class")),
                // f49l3: in the rebalance mode a top class has a floor and a ceiling too.
                Some(c) if c.rank <= top && !self.rebalancing() => fail(format!(
                    "{field}: {name:?} is a top-ranked class; only classes ranked below it have one"
                )),
                Some(_) => Ok(()),
            }
        };
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
        if let Some(st) = &policy.admission.step_target
            && !(sp.min..=sp.max).contains(&st.target_frac)
        {
            return fail(format!(
                "step_target.target_frac {} is outside [setpoint.min, setpoint.max] = [{}, {}]",
                st.target_frac, sp.min, sp.max
            ));
        }
        Ok(())
    }
}

/// What the loop is steering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlaMode {
    /// The lower classes' shares and the setpoint.
    Normal,
    /// The top class's own share too (the fallback).
    Priority,
}

impl SlaMode {
    pub fn as_str(self) -> &'static str {
        match self {
            SlaMode::Normal => "normal",
            SlaMode::Priority => "priority",
        }
    }
}

/// One class's inputs to a step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClassInput {
    /// (completions, met) in the class's trailing window
    /// ([`crate::metrics::Metrics::window_counts`]); `None` without a service level.
    pub counts: Option<(usize, usize)>,
    /// A session of the class is waiting in the lobby.
    pub waiting: bool,
    /// The class's new sessions refused so far ([`crate::scheduler::Scheduler::lost`]).
    pub lost: u64,
    /// The forecast's predicted attainment of the class, when there is one.
    pub forecast: Option<f64>,
}

/// The step log's writer.
struct LogSink(Box<dyn Write + Send + Sync>);

impl std::fmt::Debug for LogSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LogSink")
    }
}

/// The loop's state.
#[derive(Debug)]
pub struct SlaBudget {
    cfg: SlaBudgetConfig,
    /// The top-ranked classes (the lowest rank number).
    top: Vec<String>,
    /// The classes below the top, highest ranked first.
    lower: Vec<String>,
    k: HashMap<String, f64>,
    k_top: f64,
    mode: SlaMode,
    /// The step target's `target_frac`, with `admission.step_target`.
    setpoint: Option<f64>,
    meas: HashMap<String, f64>,
    fcst: HashMap<String, f64>,
    lost_seen: HashMap<String, u64>,
    /// Per class, (step, new sessions refused since the step before).
    refused: HashMap<String, VecDeque<(Time, u64)>>,
    /// Since when the top class has been short with everything else pinned.
    short_since: Option<Time>,
    last: Option<Time>,
    log: Option<LogSink>,
    /// f49l2: the classes past their warm-up (latched; every class without `warmup`).
    warm: HashSet<String>,
    /// f49l3: the rebalance mode's state (`None`: f49l).
    rb: Option<rebalance::RbState>,
}

impl SlaBudget {
    /// The loop for `policy`, or `None` when `admission.sla_budget` is absent (the default).
    pub fn from_policy(policy: &ServicePolicy) -> Option<Self> {
        let cfg = policy.admission.sla_budget.clone()?;
        let top_rank = policy.classes.iter().map(|c| c.rank).min()?;
        let top = policy
            .classes
            .iter()
            .filter(|c| c.rank == top_rank)
            .map(|c| c.name.clone())
            .collect();
        let mut below: Vec<_> = policy
            .classes
            .iter()
            .filter(|c| c.rank > top_rank)
            .collect();
        below.sort_by_key(|c| c.rank);
        let lower: Vec<String> = below.into_iter().map(|c| c.name.clone()).collect();
        let log = std::env::var_os(SLA_LOG_ENV)
            .and_then(|path| std::fs::File::create(path).ok())
            .map(|f| LogSink(Box::new(std::io::BufWriter::new(f))));
        Some(Self {
            k: lower.iter().map(|c| (c.clone(), 1.0)).collect(),
            top,
            lower,
            k_top: 1.0,
            mode: SlaMode::Normal,
            setpoint: policy.admission.step_target.as_ref().map(|s| s.target_frac),
            meas: HashMap::new(),
            fcst: HashMap::new(),
            lost_seen: HashMap::new(),
            refused: HashMap::new(),
            short_since: None,
            last: None,
            log,
            warm: HashSet::new(),
            rb: cfg.rebalancing().then(|| rebalance::RbState::new(policy)),
            cfg,
        })
    }

    /// Write the step log to `sink` (tests; the scheduler reads [`SLA_LOG_ENV`]).
    pub fn set_log(&mut self, sink: Box<dyn Write + Send + Sync>) {
        self.log = Some(LogSink(sink));
    }

    pub fn config(&self) -> &SlaBudgetConfig {
        &self.cfg
    }

    /// Whether `class` is top-ranked.
    pub fn is_top(&self, class: &str) -> bool {
        self.top.iter().any(|c| c == class)
    }

    /// The classes ranked below the top, highest first.
    pub fn lower(&self) -> &[String] {
        &self.lower
    }

    /// `class`'s share: `K` for a lower class, `K_p` for a top class, 1 for any other name.
    pub fn k(&self, class: &str) -> f64 {
        if self.is_top(class) {
            return self.k_top;
        }
        self.k.get(class).copied().unwrap_or(1.0)
    }

    /// The top classes' own share `K_p` (1 outside priority control).
    pub fn k_top(&self) -> f64 {
        self.k_top
    }

    pub fn mode(&self) -> SlaMode {
        self.mode
    }

    /// The step target's `target_frac` as set here, with `admission.step_target`.
    pub fn setpoint(&self) -> Option<f64> {
        self.setpoint
    }

    /// Set `class`'s share (tests).
    pub fn set_k(&mut self, class: &str, k: f64) {
        if self.is_top(class) {
            self.k_top = k;
        } else if let Some(v) = self.k.get_mut(class) {
            *v = k;
        }
    }

    /// Whether a step is due at `now` (always before the first, which only arms the clock).
    pub fn due(&self, now: Time) -> bool {
        self.last
            .is_none_or(|last| now.since(last).as_secs_f64() >= self.cfg.interval_s)
    }

    fn at_floor(&self, class: &str) -> bool {
        self.k(class) <= self.cfg.floor(class)
    }

    /// One step at `now` when its interval has passed (the first call only arms the clock), on
    /// `inputs` by class. The log lines it wrote (none when not due).
    pub fn step(
        &mut self,
        policy: &ServicePolicy,
        now: Time,
        inputs: &HashMap<String, ClassInput>,
    ) -> Vec<String> {
        let Some(last) = self.last else {
            self.last = Some(now);
            for (class, input) in inputs {
                self.lost_seen.insert(class.clone(), input.lost);
            }
            return Vec::new();
        };
        let dt = now.since(last).as_secs_f64();
        if dt < self.cfg.interval_s {
            return Vec::new();
        }
        self.last = Some(now);
        let t_now = now.since_origin().as_secs_f64();
        // f49l3 `mode: rebalance`: its own step (the one below is f49l's, untouched without it).
        if self.rb.is_some() {
            return self.step_rebalance(policy, now, dt, t_now, inputs);
        }
        let alpha = 1.0 - (-dt / self.cfg.smooth_s).exp();
        let blend = self.cfg.blend;
        let mut heads: HashMap<String, f64> = HashMap::new();
        let mut demand: HashMap<String, bool> = HashMap::new();
        let mut rows = Vec::with_capacity(policy.classes.len());
        for class in &policy.classes {
            let name = class.name.as_str();
            let input = inputs.get(name).copied().unwrap_or_default();
            let seen = self
                .lost_seen
                .insert(name.to_owned(), input.lost)
                .unwrap_or(input.lost);
            let fresh = input.lost.saturating_sub(seen);
            let window_s = class
                .service_level
                .as_ref()
                .map_or(self.cfg.interval_s, |l| l.window_s);
            let top = self.is_top(name);
            let q = self.refused.entry(name.to_owned()).or_default();
            if fresh > 0 {
                q.push_back((now, fresh));
            }
            while q
                .front()
                .is_some_and(|(t, _)| now.since(*t).as_secs_f64() >= window_s)
            {
                q.pop_front();
            }
            let refused: u64 = q.iter().map(|(_, n)| n).sum();
            demand.insert(name.to_owned(), input.waiting || fresh > 0);
            // The top classes on all offered work (refusals are misses), the others on admitted.
            let measured = input.counts.and_then(|(n, met)| {
                let total = n + if top { refused as usize } else { 0 };
                (total >= MIN_WINDOW_COMPLETIONS).then(|| met as f64 / total as f64)
            });
            ema(&mut self.meas, name, measured, alpha);
            ema(&mut self.fcst, name, input.forecast, alpha);
            let signal = match (self.fcst.get(name), self.meas.get(name)) {
                (Some(f), Some(m)) => Some(blend * f + (1.0 - blend) * m),
                (f, m) => f.or(m).copied(),
            };
            let head = class
                .service_level
                .as_ref()
                .zip(signal)
                .map(|(l, s)| s - (l.attainment + self.cfg.margin(name)));
            // f49l2: warm once both the time and the class's scored turns are reached (latched;
            // always without `warmup`). A cold class's head steers nothing.
            let warm = self.warm.contains(name)
                || self.cfg.is_warm(
                    t_now,
                    class.service_level.is_some(),
                    input.counts.map(|(n, _)| n),
                );
            if warm && !self.warm.contains(name) {
                self.warm.insert(name.to_owned());
            }
            if let Some(h) = head
                && warm
            {
                heads.insert(name.to_owned(), h);
            }
            rows.push((name.to_owned(), measured, input.forecast, signal, head));
        }
        // f49l2: a class's lever moves only with it and the top classes warm (every class without
        // `warmup`).
        let top_warm = self.top.iter().all(|c| self.warm.contains(c));
        let gate: HashMap<String, bool> = policy
            .classes
            .iter()
            .map(|c| (c.name.clone(), top_warm && self.warm.contains(&c.name)))
            .collect();
        let may = |c: &str| gate.get(c).copied().unwrap_or(false);
        let head_top = self
            .top
            .iter()
            .filter_map(|c| heads.get(c).copied())
            .reduce(f64::min);
        let short = heads.values().fold(0.0_f64, |a, &h| a.min(h));
        let rank = |c: &str| policy.class(Some(c)).rank;
        // f49l2: every step multiplier within the rate caps (the identity without them).
        let caps = self.cfg.rate_caps();
        let bound = |f: f64| caps.cap(f, dt);
        let shrink = |cfg: &SlaBudgetConfig, by: f64| {
            bound((cfg.shrink_gain * cfg.shrink_per_min / 60.0 * by / SHRINK_REF * dt).exp())
        };
        let grow = |cfg: &SlaBudgetConfig, h: f64| {
            bound((cfg.grow_per_min / 60.0 * h / GROW_REF * dt).exp())
        };
        match self.mode {
            SlaMode::Normal => {
                if short < 0.0 {
                    // Someone is short: the lowest class with room to give gives way first, then
                    // (for the top class only) the setpoint.
                    let f = shrink(&self.cfg, short);
                    let mut gave = false;
                    for c in self.lower.iter().rev() {
                        let lo = self.cfg.floor(c);
                        let k = self.k.get_mut(c).expect("every lower class has a share");
                        if *k > lo {
                            // f49l2: held in warm-up, it holds the step's shrink.
                            if may(c.as_str()) {
                                *k = (*k * f).max(lo);
                            }
                            gave = true;
                            break;
                        }
                    }
                    if !gave
                        && head_top.is_some_and(|h| h < 0.0)
                        && let Some(sp) = self.setpoint.as_mut()
                    {
                        let per = self.cfg.setpoint.shrink_per_min / 60.0;
                        *sp = (*sp * bound((-per * dt).exp())).max(self.cfg.setpoint.min);
                    }
                } else {
                    for c in &self.lower {
                        let at = rank(c.as_str());
                        let h = heads
                            .iter()
                            .filter(|(j, _)| rank(j.as_str()) <= at)
                            .map(|(_, h)| *h)
                            .reduce(f64::min)
                            .unwrap_or(NO_HEAD);
                        let own_ok = heads.get(c).is_none_or(|own| *own <= self.cfg.max_headroom);
                        // f49l2: and out of warm-up.
                        if demand.get(c).copied().unwrap_or(false) && own_ok && may(c.as_str()) {
                            let hi = self.cfg.ceiling(c);
                            let k = self.k.get_mut(c).expect("every lower class has a share");
                            *k = (*k * grow(&self.cfg, h)).min(hi);
                        }
                    }
                }
                if head_top.is_some_and(|h| h > 0.0)
                    && let Some(sp) = self.setpoint.as_mut()
                {
                    let per = self.cfg.setpoint.grow_per_min / 60.0;
                    *sp = (*sp * bound((per * dt).exp())).min(self.cfg.setpoint.max);
                }
                // Fallback: the top class short with every other lever at its stop.
                let pinned = self.lower.iter().all(|c| self.at_floor(c))
                    && self.setpoint.is_none_or(|sp| sp <= self.cfg.setpoint.min);
                if pinned && head_top.is_some_and(|h| h < 0.0) {
                    let since = *self.short_since.get_or_insert(now);
                    if now.since(since).as_secs_f64() >= self.cfg.fallback.after_s {
                        self.mode = SlaMode::Priority;
                        self.k_top = 1.0;
                    }
                } else {
                    self.short_since = None;
                }
            }
            SlaMode::Priority => {
                match head_top {
                    Some(h) if h < 0.0 => {
                        self.k_top =
                            (self.k_top * shrink(&self.cfg, h)).max(self.cfg.fallback.floor);
                    }
                    Some(h) if h > 0.0 => {
                        self.k_top = (self.k_top * grow(&self.cfg, h)).min(1.0);
                    }
                    _ => {}
                }
                if self.k_top >= 1.0 && head_top.is_some_and(|h| h > 0.0) {
                    self.k_top = 1.0;
                    self.mode = SlaMode::Normal;
                    self.short_since = None;
                }
            }
        }
        let t = now.since_origin().as_secs_f64();
        let lines: Vec<String> = rows
            .into_iter()
            .map(|(class, measured, forecast, signal, head)| {
                format!(
                    "{{\"t\":{t:.3},\"class\":{class:?},\"measured\":{},\"forecast\":{},\"signal\":{},\"head\":{},\"K\":{:.6},\"setpoint\":{},\"mode\":\"{}\",\"warm\":{}}}",
                    num(measured),
                    num(forecast),
                    num(signal),
                    num(head),
                    self.k(&class),
                    num(self.setpoint),
                    self.mode.as_str(),
                    may(class.as_str())
                )
            })
            .collect();
        if let Some(LogSink(w)) = self.log.as_mut() {
            for line in &lines {
                let _ = writeln!(w, "{line}");
            }
            let _ = w.flush();
        }
        lines
    }
}

/// Fold `value` into `class`'s EMA (the first value seeds it); nothing without a value.
fn ema(map: &mut HashMap<String, f64>, class: &str, value: Option<f64>, alpha: f64) {
    if let Some(v) = value {
        let e = map.entry(class.to_owned()).or_insert(v);
        *e += alpha * (v - *e);
    }
}

/// A JSON number with 6 decimals, or `null`.
fn num(x: Option<f64>) -> String {
    x.map_or_else(|| "null".to_owned(), |v| format!("{v:.6}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admit::GateAnswer;
    use crate::config::Mode;
    use crate::scheduler::Scheduler;
    use crate::step_target::StepTargetConfig;

    fn t(s: f64) -> Time {
        Time::from_secs_f64(s)
    }

    fn policy(sla: Option<SlaBudgetConfig>, step_target: bool) -> ServicePolicy {
        let mut p = ServicePolicy {
            mode: Mode::Enforce,
            ..Default::default()
        };
        p.admission.sla_budget = sla;
        if step_target {
            p.admission.step_target = Some(StepTargetConfig::default());
        }
        p
    }

    fn controller(step_target: bool) -> (SlaBudget, ServicePolicy) {
        controller_with(SlaBudgetConfig::default(), step_target)
    }

    fn controller_with(cfg: SlaBudgetConfig, step_target: bool) -> (SlaBudget, ServicePolicy) {
        let p = policy(Some(cfg), step_target);
        p.validate().unwrap();
        (SlaBudget::from_policy(&p).unwrap(), p)
    }

    /// Inputs: priority's and standard's attainment over 1000 completions (`None`: none), who
    /// waits, and the lost counts (standard, batch).
    fn inputs(
        p: Option<f64>,
        s: Option<f64>,
        waiting: &[&str],
        lost: (u64, u64),
    ) -> HashMap<String, ClassInput> {
        let counts = |a: Option<f64>| a.map(|a| (1000_usize, (a * 1000.0).round() as usize));
        [
            ("priority", counts(p), 0),
            ("standard", counts(s), lost.0),
            ("batch", None, lost.1),
        ]
        .into_iter()
        .map(|(c, counts, lost)| {
            (
                c.to_owned(),
                ClassInput {
                    counts,
                    waiting: waiting.contains(&c),
                    lost,
                    forecast: None,
                },
            )
        })
        .collect()
    }

    const EPS: f64 = 1e-9;

    #[test]
    fn grows_only_with_demand_and_under_max_headroom() {
        let (mut sb, p) = controller(false);
        assert!(
            sb.step(&p, t(0.0), &inputs(None, None, &[], (0, 0)))
                .is_empty()
        );
        // Priority 7 points over 0.92, standard 7 over 0.73; only standard waits.
        sb.step(
            &p,
            t(60.0),
            &inputs(Some(0.99), Some(0.80), &["standard"], (0, 0)),
        );
        // 1 %/min per point: 7 points for one minute.
        assert!(
            (sb.k("standard") - 0.07_f64.exp()).abs() < 1e-6,
            "{}",
            sb.k("standard")
        );
        assert_eq!(sb.k("batch"), 1.0);
        assert_eq!(sb.k("priority"), 1.0);
        // Not due before the interval: nothing moves.
        assert!(
            sb.step(
                &p,
                t(90.0),
                &inputs(Some(0.99), Some(0.80), &["standard", "batch"], (0, 9))
            )
            .is_empty()
        );
        // A batch refusal since the last step is demand too; standard no longer waits.
        let before = sb.k("standard");
        sb.step(&p, t(120.0), &inputs(Some(0.99), Some(0.80), &[], (0, 1)));
        assert_eq!(sb.k("standard"), before);
        assert!(
            (sb.k("batch") - 0.07_f64.exp()).abs() < 1e-6,
            "{}",
            sb.k("batch")
        );
        // The same refusal count again is no demand.
        let batch = sb.k("batch");
        sb.step(&p, t(180.0), &inputs(Some(0.99), Some(0.80), &[], (0, 1)));
        assert_eq!(sb.k("batch"), batch);
        // Standard's own head over max_headroom (0.15): it does not grow, though it waits.
        let (mut sb, p) = controller(false);
        sb.step(&p, t(0.0), &inputs(None, None, &[], (0, 0)));
        sb.step(
            &p,
            t(60.0),
            &inputs(Some(0.99), Some(0.95), &["standard"], (0, 0)),
        );
        assert_eq!(sb.k("standard"), 1.0);
        // Growth stops at the ceiling.
        let (mut sb, p) = controller(false);
        sb.step(&p, t(0.0), &inputs(None, None, &[], (0, 0)));
        for i in 1..=2000 {
            sb.step(
                &p,
                t(60.0 * f64::from(i)),
                &inputs(Some(0.99), Some(0.80), &["standard", "batch"], (0, 0)),
            );
        }
        assert_eq!(sb.k("standard"), 2.5);
        assert_eq!(sb.k("batch"), 2.0);
    }

    #[test]
    fn shrinks_batch_then_standard_then_the_setpoint() {
        // Five points short: 5 %/min with the defaults.
        let (mut sb, p) = controller(true);
        sb.step(&p, t(0.0), &inputs(None, None, &[], (0, 0)));
        sb.step(&p, t(60.0), &inputs(Some(0.87), Some(0.80), &[], (0, 0)));
        assert!(
            (sb.k("batch") - (-0.05_f64).exp()).abs() < 1e-6,
            "{}",
            sb.k("batch")
        );
        assert_eq!(sb.k("standard"), 1.0);
        // Priority 7 points short for good.
        let (mut sb, p) = controller(true);
        sb.step(&p, t(0.0), &inputs(None, None, &[], (0, 0)));
        let mut seen = Vec::new();
        for i in 1..=200 {
            sb.step(
                &p,
                t(60.0 * f64::from(i)),
                &inputs(Some(0.85), Some(0.80), &["standard", "batch"], (0, 0)),
            );
            seen.push((
                sb.k("batch"),
                sb.k("standard"),
                sb.setpoint().unwrap(),
                sb.mode(),
            ));
        }
        for &(b, s, sp, mode) in &seen {
            if b > 0.333 {
                assert_eq!((s, sp), (1.0, 0.85), "{seen:?}");
            }
            if s > 0.5 {
                assert_eq!(sp, 0.85, "{seen:?}");
            }
            if mode == SlaMode::Priority {
                assert_eq!((b, s, sp), (0.333, 0.5, 0.6), "{seen:?}");
            }
        }
        let &(b, s, sp, _) = seen.last().unwrap();
        assert_eq!((b, s, sp), (0.333, 0.5, 0.6));
        // A lower class short alone (priority fine) never moves the setpoint down.
        let (mut sb, p) = controller(true);
        sb.step(&p, t(0.0), &inputs(None, None, &[], (0, 0)));
        for i in 1..=200 {
            sb.step(
                &p,
                t(60.0 * f64::from(i)),
                &inputs(Some(0.95), Some(0.60), &[], (0, 0)),
            );
        }
        assert_eq!((sb.k("batch"), sb.k("standard")), (0.333, 0.5));
        assert!(sb.setpoint().unwrap() >= 0.85 - EPS);
        assert_eq!(sb.mode(), SlaMode::Normal);
    }

    #[test]
    fn setpoint_stays_within_its_bounds() {
        let (mut sb, p) = controller(true);
        sb.step(&p, t(0.0), &inputs(None, None, &[], (0, 0)));
        sb.step(&p, t(60.0), &inputs(Some(0.99), Some(0.80), &[], (0, 0)));
        assert!((sb.setpoint().unwrap() - 0.85 * 0.01_f64.exp()).abs() < 1e-9);
        for i in 2..=100 {
            sb.step(
                &p,
                t(60.0 * f64::from(i)),
                &inputs(Some(0.99), Some(0.80), &[], (0, 0)),
            );
            assert!(sb.setpoint().unwrap() <= 0.95);
        }
        assert_eq!(sb.setpoint(), Some(0.95));
        for i in 101..=400 {
            sb.step(
                &p,
                t(60.0 * f64::from(i)),
                &inputs(Some(0.80), Some(0.80), &[], (0, 0)),
            );
            assert!(sb.setpoint().unwrap() >= 0.6);
        }
        assert_eq!(sb.setpoint(), Some(0.6));
        // Without a step target there is no setpoint.
        let (sb, _) = controller(false);
        assert_eq!(sb.setpoint(), None);
    }

    #[test]
    fn fallback_enters_after_its_wait_and_exits_at_full_share() {
        let (mut sb, p) = controller(false);
        sb.step(&p, t(0.0), &inputs(None, None, &[], (0, 0)));
        let (mut pinned_at, mut entered_at) = (None, None);
        for i in 1..=400 {
            let now = 60.0 * f64::from(i);
            sb.step(&p, t(now), &inputs(Some(0.80), Some(0.80), &[], (0, 0)));
            if pinned_at.is_none() && sb.k("batch") == 0.333 && sb.k("standard") == 0.5 {
                pinned_at = Some(now);
            }
            if entered_at.is_none() && sb.mode() == SlaMode::Priority {
                entered_at = Some(now);
            }
            if sb.mode() == SlaMode::Normal {
                assert_eq!(sb.k_top(), 1.0);
            }
            assert!(sb.k_top() >= 0.3);
        }
        let (pinned_at, entered_at) = (pinned_at.unwrap(), entered_at.unwrap());
        assert_eq!(entered_at - pinned_at, 600.0);
        assert_eq!(sb.k_top(), 0.3);
        assert_eq!(sb.k("priority"), 0.3);
        // Priority recovers: K_p grows back, then the mode ends.
        let mut exited = false;
        for i in 401..=1400 {
            sb.step(
                &p,
                t(60.0 * f64::from(i)),
                &inputs(Some(0.99), Some(0.80), &[], (0, 0)),
            );
            if sb.mode() == SlaMode::Normal {
                assert_eq!(sb.k_top(), 1.0);
                exited = true;
                break;
            }
            assert!(sb.k_top() <= 1.0);
        }
        assert!(exited);
        // Short stretches under after_s (7 minutes, then a minute over target) never enter it
        // (the signal unsmoothed here, so the minute over target resets the clock).
        let raw = SlaBudgetConfig {
            smooth_s: 1e-3,
            ..SlaBudgetConfig::default()
        };
        let (mut sb, p) = controller_with(raw, false);
        sb.step(&p, t(0.0), &inputs(None, None, &[], (0, 0)));
        for i in 1..=60 {
            let att = if i % 8 == 0 { 0.99 } else { 0.80 };
            sb.step(
                &p,
                t(60.0 * f64::from(i)),
                &inputs(Some(att), Some(0.80), &[], (0, 0)),
            );
        }
        assert_eq!((sb.k("batch"), sb.k("standard")), (0.333, 0.5));
        assert_eq!(sb.mode(), SlaMode::Normal);
    }

    #[test]
    fn priority_counts_refusals_as_misses_and_lower_classes_do_not() {
        let (mut sb, p) = controller(false);
        sb.step(&p, t(0.0), &inputs(None, None, &[], (0, 0)));
        let mut i = inputs(Some(0.99), Some(0.80), &[], (100, 0));
        i.get_mut("priority").unwrap().lost = 10;
        // priority: 990 met of 1000 + 10 refused; standard's 100 refusals do not count.
        let lines = sb.step(&p, t(60.0), &i);
        assert!(lines[0].contains("\"measured\":0.980198"), "{}", lines[0]);
        assert!(lines[1].contains("\"measured\":0.800000"), "{}", lines[1]);
        // Out of the window a minute later.
        let lines = sb.step(&p, t(120.0), &i);
        assert!(lines[0].contains("\"measured\":0.990000"), "{}", lines[0]);
    }

    #[derive(Clone, Default)]
    struct Shared(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl Write for Shared {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn log_lines_have_the_documented_fields() {
        let (mut sb, p) = controller(true);
        let buf = Shared::default();
        sb.set_log(Box::new(buf.clone()));
        sb.step(&p, t(0.0), &inputs(None, None, &[], (0, 0)));
        let lines = sb.step(&p, t(60.0), &inputs(Some(0.99), Some(0.80), &[], (0, 0)));
        let sp = 0.85 * 0.01_f64.exp();
        assert_eq!(
            lines,
            vec![
                format!(
                    "{{\"t\":60.000,\"class\":\"priority\",\"measured\":0.990000,\"forecast\":null,\"signal\":0.990000,\"head\":0.070000,\"K\":1.000000,\"setpoint\":{sp:.6},\"mode\":\"normal\",\"warm\":true}}"
                ),
                format!(
                    "{{\"t\":60.000,\"class\":\"standard\",\"measured\":0.800000,\"forecast\":null,\"signal\":0.800000,\"head\":0.070000,\"K\":1.000000,\"setpoint\":{sp:.6},\"mode\":\"normal\",\"warm\":true}}"
                ),
                format!(
                    "{{\"t\":60.000,\"class\":\"batch\",\"measured\":null,\"forecast\":null,\"signal\":null,\"head\":null,\"K\":1.000000,\"setpoint\":{sp:.6},\"mode\":\"normal\",\"warm\":true}}"
                ),
            ]
        );
        let written = String::from_utf8(buf.0.lock().unwrap().clone()).unwrap();
        assert_eq!(written, lines.join("\n") + "\n");
        for line in written.lines() {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            for key in [
                "t", "class", "measured", "forecast", "signal", "head", "K", "setpoint", "mode",
                "warm",
            ] {
                assert!(v.get(key).is_some(), "{key} in {line}");
            }
        }
        // With a forecast, the signal blends the two.
        let (mut sb, p) = controller(false);
        sb.step(&p, t(0.0), &inputs(None, None, &[], (0, 0)));
        let mut i = inputs(Some(0.90), Some(0.80), &[], (0, 0));
        i.get_mut("priority").unwrap().forecast = Some(1.0);
        let lines = sb.step(&p, t(60.0), &i);
        assert!(
            lines[0].contains("\"forecast\":1.000000,\"signal\":0.950000"),
            "{}",
            lines[0]
        );
    }

    #[test]
    fn validation() {
        let base = policy(Some(SlaBudgetConfig::default()), true);
        base.validate().unwrap();
        let msg = |f: &dyn Fn(&mut ServicePolicy)| {
            let mut p = base.clone();
            f(&mut p);
            p.validate().unwrap_err().to_string()
        };
        let cfg = |f: &dyn Fn(&mut SlaBudgetConfig)| {
            msg(&|p: &mut ServicePolicy| f(p.admission.sla_budget.as_mut().unwrap()))
        };
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(cfg(&|c| c.interval_s = bad).contains("interval_s"), "{bad}");
            assert!(cfg(&|c| c.smooth_s = bad).contains("smooth_s"), "{bad}");
            assert!(
                cfg(&|c| c.shrink_gain = bad).contains("shrink_gain"),
                "{bad}"
            );
        }
        for bad in [-0.1, 1.1, f64::NAN] {
            assert!(cfg(&|c| c.blend = bad).contains("blend"), "{bad}");
        }
        assert!(cfg(&|c| c.grow_per_min = -0.01).contains("grow_per_min"));
        assert!(cfg(&|c| c.shrink_per_min = f64::NAN).contains("shrink_per_min"));
        assert!(cfg(&|c| c.max_headroom = f64::INFINITY).contains("max_headroom"));
        assert!(cfg(&|c| c.setpoint.min = 0.0).contains("setpoint.min"));
        assert!(cfg(&|c| c.setpoint.max = 0.5).contains("setpoint.max"));
        assert!(cfg(&|c| c.setpoint.max = 1.1).contains("setpoint.max"));
        assert!(cfg(&|c| c.setpoint.grow_per_min = -1.0).contains("setpoint.grow_per_min"));
        assert!(cfg(&|c| c.fallback.after_s = -1.0).contains("fallback.after_s"));
        assert!(cfg(&|c| c.fallback.floor = 0.0).contains("fallback.floor"));
        assert!(cfg(&|c| c.fallback.floor = 1.5).contains("fallback.floor"));
        assert!(
            cfg(&|c| {
                c.margin.insert("nope".into(), 0.02);
            })
            .contains("not a configured class")
        );
        assert!(
            cfg(&|c| {
                c.margin.insert("batch".into(), 1.0);
            })
            .contains("margin.batch")
        );
        assert!(
            cfg(&|c| {
                c.floors.insert("priority".into(), 0.5);
            })
            .contains("top-ranked")
        );
        assert!(
            cfg(&|c| {
                c.ceilings.insert("nope".into(), 2.0);
            })
            .contains("not a configured class")
        );
        assert!(
            cfg(&|c| {
                c.floors.insert("batch".into(), 0.0);
            })
            .contains("floors.batch")
        );
        assert!(
            cfg(&|c| {
                c.floors.insert("batch".into(), 1.2);
            })
            .contains("floors.batch")
        );
        assert!(
            cfg(&|c| {
                c.ceilings.insert("standard".into(), 0.9);
            })
            .contains("ceilings.standard")
        );
        // The step target's target_frac must start inside the setpoint's range.
        assert!(
            msg(&|p| p.admission.step_target.as_mut().unwrap().target_frac = 0.5)
                .contains("outside [setpoint.min, setpoint.max]")
        );
        let mut p = policy(Some(SlaBudgetConfig::default()), false);
        for c in &mut p.classes {
            c.service_level = None;
        }
        let err = p.validate().unwrap_err().to_string();
        assert!(err.contains("no class has a service_level"), "{err}");
        // Every message names the option.
        assert!(cfg(&|c| c.blend = 2.0).starts_with("admission.sla_budget: "));
        // The approved design parses to the defaults; an unknown key is refused.
        let p: ServicePolicy = serde_json::from_str(
            r#"{"admission": {"sla_budget": {"interval_s": 60, "smooth_s": 300, "blend": 0.5,
                "margin": {"priority": 0.02, "standard": 0.03}, "grow_per_min": 0.01,
                "shrink_per_min": 0.05, "shrink_gain": 2,
                "floors": {"standard": 0.5, "batch": 0.333},
                "ceilings": {"standard": 2.5, "batch": 2.0}, "max_headroom": 0.15,
                "setpoint": {"min": 0.6, "max": 0.95, "grow_per_min": 0.01, "shrink_per_min": 0.05},
                "fallback": {"after_s": 600, "floor": 0.3}}}}"#,
        )
        .unwrap();
        assert_eq!(p.admission.sla_budget, Some(SlaBudgetConfig::default()));
        let p: ServicePolicy =
            serde_json::from_str(r#"{"admission": {"sla_budget": {}}}"#).unwrap();
        assert_eq!(p.admission.sla_budget, Some(SlaBudgetConfig::default()));
        assert!(
            serde_json::from_str::<ServicePolicy>(r#"{"admission": {"sla_budget": {"gain": 1}}}"#)
                .is_err()
        );
        // Absent by default, and not serialized then.
        assert_eq!(ServicePolicy::default().admission.sla_budget, None);
        assert!(
            !serde_json::to_string(&ServicePolicy::default())
                .unwrap()
                .contains("sla_budget")
        );
    }

    /// The hot cap's multiple of `gate_share x hot_overcommit x HBM` the gate uses for `class`.
    fn cap(s: &Scheduler, class: &str) -> f64 {
        let c = s.policy().class(Some(class));
        c.admission.gate_share
            * s.pace_scale(class)
            * s.step_target_scale(class)
            * s.policy().class_overcommit(c)
            * s.sla_scale(class)
    }

    #[test]
    fn no_inversion_of_the_top_class_hot_budget() {
        let mut p = policy(Some(SlaBudgetConfig::default()), false);
        p.admission.class_overcommit.insert("standard".into(), 2.0);
        // Off: f49k alone lets standard's cap (0.9 x 2) pass priority's (1).
        let off = Scheduler::new(policy(None, false));
        assert_eq!(off.sla_scale("priority"), 1.0);
        let mut off_co = policy(None, false);
        off_co
            .admission
            .class_overcommit
            .insert("standard".into(), 2.0);
        let off_co = Scheduler::new(off_co);
        assert!(cap(&off_co, "standard") > cap(&off_co, "priority"));
        // On: priority's cap is the larger of its own and every lower class's.
        let mut s = Scheduler::new(p);
        for (ks, kb, kp) in [
            (1.0, 1.0, 1.0),
            (2.5, 1.0, 1.0),
            (1.0, 2.0, 1.0),
            (0.5, 0.333, 0.3),
            (0.5, 0.333, 1.0),
        ] {
            let sb = s.sla_budget_mut().unwrap();
            sb.set_k("standard", ks);
            sb.set_k("batch", kb);
            sb.set_k("priority", kp);
            let (pc, sc, bc) = (cap(&s, "priority"), cap(&s, "standard"), cap(&s, "batch"));
            assert!(
                pc + EPS >= sc && pc + EPS >= bc,
                "{ks} {kb} {kp}: {pc} {sc} {bc}"
            );
            assert!((pc - (kp * 1.0_f64).max(sc).max(bc)).abs() < EPS, "{pc}");
        }
        // The room the tickets read (the KV room of a new session) follows the same caps.
        let kv = s.kv_budget(1_000, 1_000_000);
        let room = |c: &str| {
            crate::admit::room_scaled(
                s.policy(),
                s.policy().class(Some(c)),
                &kv,
                s.step_target_scale(c) * s.sla_scale(c),
            )
        };
        assert!(room("priority") >= room("standard"));
    }

    /// Standard session `a` (600 tokens) asks at 0 s and 70 s; priority `p` (500) at 71 s; HBM
    /// 1000. Answers and the decision log.
    fn scenario(p: ServicePolicy, k_standard: Option<f64>) -> (Vec<GateAnswer>, String) {
        let mut s = Scheduler::new(p);
        if let Some(k) = k_standard {
            s.sla_budget_mut().unwrap().set_k("standard", k);
        }
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
    fn options_off_is_the_identity_and_the_share_reaches_the_gate() {
        let (off, off_log) = scenario(policy(None, false), None);
        let off_s = Scheduler::new(policy(None, false));
        for c in ["priority", "standard", "batch", "nope"] {
            assert_eq!(off_s.sla_scale(c), 1.0);
        }
        assert!(!off_s.sla_budget_on());
        // On, with every share at 1 and no step taken: the same answers and decisions.
        let still = SlaBudgetConfig {
            interval_s: 1e9,
            ..SlaBudgetConfig::default()
        };
        let (on, on_log) = scenario(policy(Some(still.clone()), false), None);
        assert_eq!(off, on);
        assert_eq!(off_log, on_log);
        // Standard's share at its floor halves its hot cap (900 to 450): `a` (600) waits.
        let (held, _) = scenario(policy(Some(still), false), Some(0.5));
        assert_eq!(off[0], GateAnswer::Go, "{off:?}");
        assert!(matches!(held[0], GateAnswer::Hold { .. }), "{held:?}");
    }

    /// f49l2: `i` with `n` scored turns for each measured class (about the same attainment).
    fn scored(mut i: HashMap<String, ClassInput>, n: usize) -> HashMap<String, ClassInput> {
        for input in i.values_mut() {
            input.counts = input.counts.map(|(total, met)| (n, met * n / total));
        }
        i
    }

    /// A log line's `key`.
    fn field(line: &str, key: &str) -> serde_json::Value {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        v[key].clone()
    }

    fn capped(shrink: Option<f64>, grow: Option<f64>) -> SlaBudgetConfig {
        SlaBudgetConfig {
            max_shrink_per_min: shrink,
            max_grow_per_min: grow,
            ..SlaBudgetConfig::default()
        }
    }

    fn warming() -> SlaBudgetConfig {
        SlaBudgetConfig {
            warmup: Some(WarmupConfig {
                min_scored: 50,
                after_s: 1200.0,
            }),
            ..SlaBudgetConfig::default()
        }
    }

    #[test]
    fn f49l2_shrink_cap_limits_a_75_point_shortfall() {
        // Priority at 0.17 against 0.90 + 0.02: 75 points short, for one minute.
        let short = |cfg: SlaBudgetConfig| {
            let (mut sb, p) = controller_with(cfg, false);
            sb.step(&p, t(0.0), &inputs(None, None, &[], (0, 0)));
            sb.step(&p, t(60.0), &inputs(Some(0.17), Some(0.80), &[], (0, 0)));
            (sb, p)
        };
        // f49l: about 75 %/min, batch's K from 1 to 0.47 in one step.
        let (sb, _) = short(SlaBudgetConfig::default());
        assert!(
            (sb.k("batch") - (-0.75_f64).exp()).abs() < 1e-9,
            "{}",
            sb.k("batch")
        );
        // Capped at 0.10: 10 %/min, whatever the shortfall.
        let (mut sb, p) = short(capped(Some(0.10), None));
        assert!(
            (sb.k("batch") - (-0.10_f64).exp()).abs() < 1e-9,
            "{}",
            sb.k("batch")
        );
        // Two minutes in one step: twice that.
        sb.step(&p, t(180.0), &inputs(Some(0.17), Some(0.80), &[], (0, 0)));
        assert!(
            (sb.k("batch") - (-0.30_f64).exp()).abs() < 1e-9,
            "{}",
            sb.k("batch")
        );
        // Under the cap the rate is f49l's: five points short, 5 %/min.
        let (mut sb, p) = controller_with(capped(Some(0.10), None), false);
        sb.step(&p, t(0.0), &inputs(None, None, &[], (0, 0)));
        sb.step(&p, t(60.0), &inputs(Some(0.87), Some(0.80), &[], (0, 0)));
        assert!((sb.k("batch") - (-0.05_f64).exp()).abs() < 1e-9);
        // max_grow_per_min caps growth the same way: 7 points over is 7 %/min, capped at 2.
        let grow = |cfg: SlaBudgetConfig| {
            let (mut sb, p) = controller_with(cfg, false);
            sb.step(&p, t(0.0), &inputs(None, None, &[], (0, 0)));
            sb.step(
                &p,
                t(60.0),
                &inputs(Some(0.99), Some(0.80), &["standard"], (0, 0)),
            );
            sb.k("standard")
        };
        assert!((grow(SlaBudgetConfig::default()) - 0.07_f64.exp()).abs() < 1e-9);
        assert!((grow(capped(None, Some(0.02))) - 0.02_f64.exp()).abs() < 1e-9);
        assert!((grow(capped(Some(0.10), None)) - 0.07_f64.exp()).abs() < 1e-9);
    }

    #[test]
    fn f49l2_caps_apply_to_the_setpoint_and_k_p() {
        // The setpoint's own rates at 50 %/min, so the caps bind.
        let fast = |shrink: Option<f64>, grow: Option<f64>| SlaBudgetConfig {
            setpoint: SetpointConfig {
                grow_per_min: 0.5,
                shrink_per_min: 0.5,
                ..SetpointConfig::default()
            },
            ..capped(shrink, grow)
        };
        // Every lower share at its floor: priority's shortfall moves the setpoint.
        let setpoint = |cfg: SlaBudgetConfig, att: f64| {
            let (mut sb, p) = controller_with(cfg, true);
            sb.set_k("standard", 0.5);
            sb.set_k("batch", 0.333);
            sb.step(&p, t(0.0), &inputs(None, None, &[], (0, 0)));
            sb.step(&p, t(60.0), &inputs(Some(att), Some(0.80), &[], (0, 0)));
            sb.setpoint().unwrap()
        };
        // 0.85 x exp(-0.5) is under the min.
        assert_eq!(setpoint(fast(None, None), 0.17), 0.6);
        let sp = setpoint(fast(Some(0.10), None), 0.17);
        assert!((sp - 0.85 * (-0.10_f64).exp()).abs() < 1e-12, "{sp}");
        // 0.85 x exp(0.5) is over the max.
        assert_eq!(setpoint(fast(None, None), 0.99), 0.95);
        let sp = setpoint(fast(None, Some(0.05)), 0.99);
        assert!((sp - 0.85 * 0.05_f64.exp()).abs() < 1e-12, "{sp}");
        // K_p: priority control at once (fallback.after_s 0), unsmoothed; a minute 75 points
        // short, then a minute 7 points over.
        let k_p = |cfg: SlaBudgetConfig| {
            let cfg = SlaBudgetConfig {
                smooth_s: 1e-3,
                fallback: FallbackConfig {
                    after_s: 0.0,
                    ..FallbackConfig::default()
                },
                ..cfg
            };
            let (mut sb, p) = controller_with(cfg, false);
            sb.set_k("standard", 0.5);
            sb.set_k("batch", 0.333);
            sb.step(&p, t(0.0), &inputs(None, None, &[], (0, 0)));
            sb.step(&p, t(60.0), &inputs(Some(0.17), Some(0.80), &[], (0, 0)));
            assert_eq!((sb.mode(), sb.k_top()), (SlaMode::Priority, 1.0));
            sb.step(&p, t(120.0), &inputs(Some(0.17), Some(0.80), &[], (0, 0)));
            let shrunk = sb.k_top();
            sb.step(&p, t(180.0), &inputs(Some(0.99), Some(0.80), &[], (0, 0)));
            (shrunk, sb.k_top())
        };
        let (s, g) = k_p(SlaBudgetConfig::default());
        assert!((s - (-0.75_f64).exp()).abs() < 1e-9, "{s}");
        assert!((g - (-0.68_f64).exp()).abs() < 1e-9, "{g}");
        let (s, g) = k_p(capped(Some(0.10), Some(0.02)));
        assert!((s - (-0.10_f64).exp()).abs() < 1e-9, "{s}");
        assert!((g - (-0.08_f64).exp()).abs() < 1e-9, "{g}");
    }

    #[test]
    fn f49l2_warmup_holds_until_both_conditions_are_met() {
        let short = |n: usize| {
            scored(
                inputs(Some(0.17), Some(0.80), &["standard", "batch"], (0, 0)),
                n,
            )
        };
        let unmoved = |sb: &SlaBudget| {
            assert_eq!(
                (sb.k("batch"), sb.k("standard"), sb.setpoint(), sb.mode()),
                (1.0, 1.0, Some(0.85), SlaMode::Normal)
            );
        };
        // Turns first: enough scored turns throughout; nothing moves before 1200 s, while the
        // smoothing goes on (the same signal as f49l's, which has moved).
        let (mut sb, p) = controller_with(warming(), true);
        let (mut free, fp) = controller(true);
        sb.step(&p, t(0.0), &short(1000));
        free.step(&fp, t(0.0), &short(1000));
        for i in 1..20 {
            let now = t(60.0 * f64::from(i));
            let lines = sb.step(&p, now, &short(1000));
            let free_lines = free.step(&fp, now, &short(1000));
            unmoved(&sb);
            assert_eq!(lines.len(), 3);
            for (held, moved) in lines.iter().zip(&free_lines) {
                assert_eq!(field(held, "warm"), serde_json::json!(false), "{held}");
                assert_eq!(field(moved, "warm"), serde_json::json!(true), "{moved}");
                assert_eq!(field(held, "signal"), field(moved, "signal"));
            }
        }
        assert!(free.k("batch") < 1.0);
        // At 1200 s both hold: it moves.
        let lines = sb.step(&p, t(1200.0), &short(1000));
        assert!(
            (sb.k("batch") - (-0.75_f64).exp()).abs() < 1e-9,
            "{}",
            sb.k("batch")
        );
        for line in &lines {
            assert_eq!(field(line, "warm"), serde_json::json!(true), "{line}");
        }
        // Time first: past 1200 s under 50 scored turns, nothing moves until they reach 50.
        let (mut sb, p) = controller_with(warming(), true);
        sb.step(&p, t(0.0), &short(40));
        for i in 1..=30 {
            sb.step(&p, t(60.0 * f64::from(i)), &short(40));
            unmoved(&sb);
        }
        sb.step(&p, t(1860.0), &short(50));
        let k = sb.k("batch");
        assert!(k < 1.0, "{k}");
        // Latched: fewer turns later do not hold it again.
        sb.step(&p, t(1920.0), &short(10));
        assert!(sb.k("batch") < k, "{} {k}", sb.k("batch"));
        // A lower class waits for its own turns: priority warm and over target, standard cold
        // and waiting. Standard holds; batch (no service level: the time alone) grows on
        // priority's head alone.
        let (mut sb, p) = controller_with(warming(), false);
        let mut i = inputs(Some(0.99), Some(0.80), &["standard", "batch"], (0, 0));
        i.get_mut("standard").unwrap().counts = Some((40, 32));
        sb.step(&p, t(0.0), &i);
        let lines = sb.step(&p, t(1200.0), &i);
        assert_eq!(sb.k("standard"), 1.0);
        assert!(sb.k("batch") > 1.0, "{}", sb.k("batch"));
        let warm: Vec<_> = lines.iter().map(|l| field(l, "warm")).collect();
        assert_eq!(warm, [true, false, true].map(serde_json::Value::Bool));
    }

    #[test]
    fn f49l2_absent_keys_are_f49l() {
        // Absent by default, and not serialized.
        let d = SlaBudgetConfig::default();
        assert_eq!(
            (d.max_shrink_per_min, d.max_grow_per_min, d.warmup.clone()),
            (None, None, None)
        );
        let json = serde_json::to_string(&d).unwrap();
        for key in ["max_shrink_per_min", "max_grow_per_min", "warmup"] {
            assert!(!json.contains(key), "{json}");
        }
        // No caps: any multiplier as is; no warm-up: always warm.
        for mult in [0.0, 1e-300, 0.47, 1.0, 7.0] {
            assert_eq!(d.rate_caps().cap(mult, 60.0), mult);
        }
        assert!(d.is_warm(0.0, true, None));
        // Keys that never bind leave every step as f49l's (steps of 75 points short and 7 over,
        // priority control and back): the same shares, setpoint, mode and log lines.
        let slack = SlaBudgetConfig {
            max_shrink_per_min: Some(1.0),
            max_grow_per_min: Some(1.0),
            warmup: Some(WarmupConfig {
                min_scored: 1,
                after_s: 1e-3,
            }),
            ..SlaBudgetConfig::default()
        };
        for step_target in [false, true] {
            let (mut a, pa) = controller(step_target);
            let (mut b, pb) = controller_with(slack.clone(), step_target);
            let mut modes = HashSet::new();
            for i in 0..=600_usize {
                let att = [0.99, 0.85, 0.17, 0.95, 0.80][(i / 40) % 5];
                let lost = u64::try_from(i / 7).unwrap();
                let input = inputs(Some(att), Some(0.80), &["standard", "batch"], (lost, lost));
                let now = t(60.0 * f64::from(u32::try_from(i).unwrap()));
                assert_eq!(a.step(&pa, now, &input), b.step(&pb, now, &input), "{i}");
                for c in ["priority", "standard", "batch"] {
                    assert_eq!(a.k(c), b.k(c), "{i} {c}");
                }
                assert_eq!((a.setpoint(), a.mode()), (b.setpoint(), b.mode()), "{i}");
                modes.insert(a.mode().as_str());
            }
            assert_eq!(modes.len(), 2, "{modes:?}");
        }
    }

    #[test]
    fn f49l2_validation() {
        let base = policy(Some(SlaBudgetConfig::default()), true);
        let check = |cfg: SlaBudgetConfig| {
            let mut p = base.clone();
            p.admission.sla_budget = Some(cfg);
            p.validate().map_err(|e| e.to_string())
        };
        let warm = |min_scored: u32, after_s: f64| SlaBudgetConfig {
            warmup: Some(WarmupConfig {
                min_scored,
                after_s,
            }),
            ..SlaBudgetConfig::default()
        };
        for ok in [0.10, 1.0] {
            check(capped(Some(ok), Some(ok))).unwrap();
        }
        check(capped(None, Some(5.0))).unwrap();
        for bad in [0.0, -0.1, 1.1, f64::NAN, f64::INFINITY] {
            let err = check(capped(Some(bad), None)).unwrap_err();
            assert!(
                err.contains("max_shrink_per_min must be in (0, 1]"),
                "{err}"
            );
            assert!(err.starts_with("admission.sla_budget: "), "{err}");
        }
        for bad in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            let err = check(capped(None, Some(bad))).unwrap_err();
            assert!(err.contains("max_grow_per_min"), "{err}");
        }
        check(warm(50, 1200.0)).unwrap();
        check(warm(1, 1e-3)).unwrap();
        let err = check(warm(0, 1200.0)).unwrap_err();
        assert!(err.contains("warmup.min_scored"), "{err}");
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let err = check(warm(50, bad)).unwrap_err();
            assert!(err.contains("warmup.after_s"), "{err}");
        }
        // The YAML's keys parse; warmup needs both of its keys and no other.
        let p: ServicePolicy = serde_json::from_str(
            r#"{"admission": {"sla_budget": {"max_shrink_per_min": 0.1, "max_grow_per_min": 0.1,
                "warmup": {"min_scored": 50, "after_s": 1200}}}}"#,
        )
        .unwrap();
        assert_eq!(
            p.admission.sla_budget,
            Some(SlaBudgetConfig {
                max_shrink_per_min: Some(0.1),
                max_grow_per_min: Some(0.1),
                ..warming()
            })
        );
        for bad in [
            r#"{"min_scored": 50}"#,
            r#"{"after_s": 1200}"#,
            r#"{"min_scored": 50, "after_s": 1200, "any": true}"#,
            r#"{"min_scored": -1, "after_s": 1200}"#,
        ] {
            let json = format!(r#"{{"admission": {{"sla_budget": {{"warmup": {bad}}}}}}}"#);
            assert!(
                serde_json::from_str::<ServicePolicy>(&json).is_err(),
                "{json}"
            );
        }
    }

    #[test]
    fn the_scheduler_hands_the_setpoint_to_the_step_target() {
        let mut s = Scheduler::new(policy(Some(SlaBudgetConfig::default()), true));
        assert_eq!(s.step_target().unwrap().target_frac(), 0.85);
        s.sla_budget_apply(t(0.0), &inputs(None, None, &[], (0, 0)));
        for i in 1..=100 {
            s.sla_budget_apply(
                t(60.0 * f64::from(i)),
                &inputs(Some(0.99), Some(0.80), &[], (0, 0)),
            );
        }
        assert_eq!(s.sla_budget().unwrap().setpoint(), Some(0.95));
        assert_eq!(s.step_target().unwrap().target_frac(), 0.95);
        assert!((s.step_target().unwrap().target_s() - 0.95 * 0.05).abs() < 1e-12);
        // The configured policy is untouched.
        assert_eq!(
            s.policy()
                .admission
                .step_target
                .as_ref()
                .unwrap()
                .target_frac,
            0.85
        );
    }
}
