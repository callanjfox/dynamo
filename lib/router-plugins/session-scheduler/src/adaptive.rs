// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The adaptive rule (`adaptive` in the plugin's parameters; the simulator's `policy.adaptive`,
//! `web/sim.js` `adaptStep`, `docs/decision-rules.md`): switch groups of capabilities on and off
//! by what a router can see. Off (absent) by default.
//!
//! Signals, sampled at most every `sample_s` and smoothed with weight `min(1, sample_s / win_s)`:
//!
//! - **P**, load pressure: the mean over workers of active tokens (prefill plus the blocks active
//!   requests hold) over the worker's KV capacity;
//! - **Dt**, KV demand: every live session's context over HBM plus DRAM, which gating does not
//!   relieve;
//! - with `queue_load` (off by default), P also counts the prompt tokens of the requests waiting at
//!   the router over the fleet's KV capacity, so a flood held at the busy limits reads as load;
//! - client hints: whether any request carried a client signal (`x-dynamo-hint-*`).
//!
//! Groups, with hysteresis (the simulator's `ADAPT_DEFAULTS`):
//!
//! - **place** (worker-selection scorers of this crate: the priority pool, affinity, forecast
//!   routing): on above `load_on` while `Dt` is under `mem_lo` (`mem_hi` once on, with
//!   `mem_guard`), off below `load_off` or past it;
//! - **gate** (the new-session gate): on above `load_on` with `Dt` under `mem_lo` and hints; off
//!   below `load_off`, at `Dt` over `mem_hi`, or without hints.
//!
//! - **kv** (the router's KV actions, `kv_actions` in its policy file): follows place (`kv: true`,
//!   the simulator's default); with `kv: false` always on;
//! - **preempt** (class preemption in the engine, `preemptLower`): with place on, on while the
//!   top class's attainment over its trailing window is under its target plus `preempt_on`; off
//!   when place is off or it is over target plus `preempt_off`; a change holds `preempt_hold_s`.
//!
//! All start on (`start_on`), as the simulator's do. The kv and preempt groups reach the router
//! and the engines through `dynamo_kv_router::plugins::adaptive_switches`.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

use crate::clock::Time;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AdaptiveConfig {
    pub load_on: f64,
    pub load_off: f64,
    pub mem_lo: f64,
    pub mem_hi: f64,
    /// Smoothing window of the signals, seconds.
    pub win_s: f64,
    /// The gate needs client hints.
    pub hints: bool,
    /// Place stays off while KV demand is over `mem_lo` (`mem_hi` once on).
    pub mem_guard: bool,
    /// Least seconds between two samples.
    pub sample_s: f64,
    /// Groups start on.
    pub start_on: bool,
    /// Keep every group on and only record the signals.
    pub observe: bool,
    /// The kv group follows place; false keeps KV actions always on.
    pub kv: bool,
    pub preempt_on: f64,
    pub preempt_off: f64,
    pub preempt_hold_s: f64,
    /// Load pressure also counts the requests waiting at the router (classified, not yet taken
    /// by the host): their prompt tokens over the fleet's KV capacity. Off (the default), load is
    /// what the workers hold only, so a flood held at the busy limits reads as a light load and
    /// the rule switches every group off while the router queue fills.
    pub queue_load: bool,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            load_on: 0.3,
            load_off: 0.2,
            mem_lo: 1.0,
            mem_hi: 1.2,
            win_s: 300.0,
            hints: true,
            mem_guard: true,
            sample_s: 60.0,
            start_on: true,
            observe: false,
            kv: true,
            preempt_on: 0.03,
            preempt_off: 0.06,
            preempt_hold_s: 600.0,
            queue_load: false,
        }
    }
}

/// The rule's state: smoothed signals and the groups.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveState {
    pub config: AdaptiveConfig,
    pub load: Option<f64>,
    pub demand: Option<f64>,
    pub hints_seen: bool,
    pub gate: bool,
    pub place: bool,
    pub kv: bool,
    pub preempt: bool,
    pub flips: u64,
    last: Option<Time>,
    /// When the rule first asked for a sample: the first sample waits `sample_s` from it.
    first: Option<Time>,
    preempt_at: Option<Time>,
}

impl AdaptiveState {
    pub fn new(config: AdaptiveConfig) -> Self {
        let on = config.start_on || config.observe;
        let kv = on || !config.kv;
        Self {
            config,
            load: None,
            demand: None,
            hints_seen: false,
            gate: on,
            place: on,
            kv,
            preempt: on,
            flips: 0,
            last: None,
            first: None,
            preempt_at: None,
        }
    }

    /// Whether a sample is due at `now`. The first is due `sample_s` after the rule is first asked,
    /// not at once: the first request arrives before any load is booked, and an unsmoothed sample of
    /// that empty fleet switched every group off, so the warm sessions came in ungated (the
    /// simulator's rule first samples a running fleet; `docs/decision-rules.md`, start state).
    pub fn due(&mut self, now: Time) -> bool {
        let since = self.last.unwrap_or(*self.first.get_or_insert(now));
        now.since(since).as_secs_f64() >= self.config.sample_s
    }

    /// One sample: `load` (mean active tokens over capacity across workers) and `demand` (live
    /// session KV over HBM plus DRAM). Returns whether a group changed.
    pub fn step(&mut self, now: Time, load: f64, demand: f64) -> bool {
        self.step_with(now, load, demand, None, None)
    }

    /// [`Self::step`] with the top class's attainment over its trailing window and its target,
    /// for the preempt group.
    pub fn step_with(
        &mut self,
        now: Time,
        load: f64,
        demand: f64,
        top_attainment: Option<f64>,
        top_target: Option<f64>,
    ) -> bool {
        let dt = self
            .last
            .map_or(self.config.sample_s, |at| now.since(at).as_secs_f64());
        self.last = Some(now);
        let k = (dt / self.config.win_s.max(1e-9)).min(1.0);
        let smooth = |old: Option<f64>, x: f64| Some(old.map_or(x, |o| o + k * (x - o)));
        self.load = smooth(self.load, load);
        self.demand = smooth(self.demand, demand);
        tracing::debug!(
            t_s = now.as_secs_f64(),
            raw_load = load,
            raw_demand = demand,
            load = self.load,
            demand = self.demand,
            "adaptive_sample"
        );
        if self.config.observe {
            return false;
        }
        let (p, d) = (self.load.unwrap_or(0.0), self.demand.unwrap_or(0.0));
        let c = &self.config;
        let was = (self.gate, self.place);
        let mem_ok = !c.mem_guard
            || if self.place {
                d < c.mem_hi
            } else {
                d < c.mem_lo
            };
        if !self.place && p > c.load_on && mem_ok {
            self.place = true;
        } else if self.place && (p < c.load_off || !mem_ok) {
            self.place = false;
        }
        let hints_ok = !c.hints || self.hints_seen;
        let want = hints_ok && p > c.load_on && d < c.mem_lo;
        let stay = hints_ok && p > c.load_off && d < c.mem_hi;
        if !self.gate && want {
            self.gate = true;
        } else if self.gate && !stay {
            self.gate = false;
        }
        let was_kv = self.kv;
        self.kv = if c.kv { self.place } else { true };
        let was_preempt = self.preempt;
        let held = self
            .preempt_at
            .is_some_and(|at| now.since(at).as_secs_f64() < c.preempt_hold_s);
        let short = |margin: f64| match (top_attainment, top_target) {
            (Some(a), Some(t)) => Some(a < t + margin),
            _ => None,
        };
        if !self.preempt && self.place && short(c.preempt_on) == Some(true) && !held {
            self.preempt = true;
        } else if self.preempt
            && !held
            && (!self.place || top_attainment.is_none() || short(c.preempt_off) == Some(false))
        {
            self.preempt = false;
        }
        if was_preempt != self.preempt {
            self.preempt_at = Some(now);
        }
        let changed =
            was != (self.gate, self.place) || was_kv != self.kv || was_preempt != self.preempt;
        if changed {
            tracing::info!(
                t_s = now.as_secs_f64(),
                load = p,
                demand = d,
                gate = self.gate,
                place = self.place,
                kv = self.kv,
                preempt = self.preempt,
                "adaptive_switch"
            );
            self.flips += u64::from(was.0 != self.gate)
                + u64::from(was.1 != self.place)
                + u64::from(was_kv != self.kv)
                + u64::from(was_preempt != self.preempt);
        }
        changed
    }
}

/// Whether the place group is on, read by this crate's worker-selection scorers (on when no
/// adaptive rule runs). Set by the classifier.
pub fn place_enabled() -> bool {
    place_flag().load(Ordering::Relaxed)
}

pub fn set_place_enabled(on: bool) {
    place_flag().store(on, Ordering::Relaxed);
}

fn place_flag() -> &'static AtomicBool {
    static FLAG: OnceLock<AtomicBool> = OnceLock::new();
    FLAG.get_or_init(|| AtomicBool::new(true))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: f64) -> Time {
        Time::from_secs_f64(s)
    }

    fn quick() -> AdaptiveConfig {
        AdaptiveConfig {
            win_s: 60.0,
            ..Default::default()
        }
    }

    #[test]
    fn groups_start_on_and_switch_off_at_low_load() {
        let mut a = AdaptiveState::new(quick());
        a.hints_seen = true;
        assert!(a.gate && a.place);
        a.step(t(0.0), 0.1, 0.5);
        assert!(!a.gate && !a.place, "load 0.1 is under load_off");
        assert!(!a.kv && !a.preempt, "kv follows place; preempt needs place");
        assert_eq!(a.flips, 4);
        // Back over load_on with memory room: both on.
        a.step(t(60.0), 0.5, 0.5);
        assert!(a.gate && a.place);
        // Between load_off and load_on: hysteresis keeps them on.
        a.step(t(120.0), 0.25, 0.5);
        assert!(a.gate && a.place);
    }

    #[test]
    fn the_gate_needs_hints_and_memory_room() {
        let mut a = AdaptiveState::new(AdaptiveConfig {
            start_on: false,
            ..quick()
        });
        a.step(t(0.0), 0.5, 0.5);
        assert!(!a.gate, "no hints seen");
        assert!(a.place);
        a.hints_seen = true;
        a.step(t(60.0), 0.5, 0.5);
        assert!(a.gate);
        // KV demand over mem_hi: the gate cannot relieve it, so it switches off, and so does
        // place under the memory guard.
        a.step(t(120.0), 0.5, 1.3);
        assert!(!a.gate && !a.place);
    }

    #[test]
    fn preempt_follows_the_top_class_short_of_target_and_holds() {
        let mut a = AdaptiveState::new(AdaptiveConfig {
            start_on: false,
            preempt_hold_s: 100.0,
            ..quick()
        });
        a.hints_seen = true;
        // Place on (load 0.5); priority at 0.91 against 0.9: under target + 0.03, preempt on.
        a.step_with(t(0.0), 0.5, 0.5, Some(0.91), Some(0.9));
        assert!(a.place && a.kv && a.preempt);
        // Well over target, but within the hold: stays on.
        a.step_with(t(60.0), 0.5, 0.5, Some(0.99), Some(0.9));
        assert!(a.preempt);
        // After the hold: off.
        a.step_with(t(120.0), 0.5, 0.5, Some(0.99), Some(0.9));
        assert!(!a.preempt);
        // kv: false keeps KV actions on with place off.
        let mut b = AdaptiveState::new(AdaptiveConfig {
            kv: false,
            ..quick()
        });
        b.step(t(0.0), 0.0, 0.0);
        assert!(!b.place && b.kv);
    }

    #[test]
    fn observe_keeps_every_group_on() {
        let mut a = AdaptiveState::new(AdaptiveConfig {
            observe: true,
            start_on: false,
            ..quick()
        });
        a.step(t(0.0), 0.0, 5.0);
        assert!(a.gate && a.place);
        assert_eq!(a.load, Some(0.0));
    }

    #[test]
    fn the_first_sample_waits_for_load_so_the_groups_start_on() {
        // The first request finds no load booked yet; sampling it at once switched every group off
        // and let the warm sessions in ungated.
        let mut a = AdaptiveState::new(AdaptiveConfig::default());
        a.hints_seen = true;
        assert!(!a.due(t(0.0)));
        assert!(a.gate && a.place && a.kv && a.preempt);
        assert!(!a.due(t(59.0)));
        assert!(a.due(t(60.0)));
        a.step(t(60.0), 0.6, 0.5);
        assert!(a.gate && a.place, "loaded and with memory room: stays on");
    }

    #[test]
    fn samples_are_smoothed_and_spaced() {
        let mut a = AdaptiveState::new(AdaptiveConfig {
            win_s: 120.0,
            sample_s: 60.0,
            ..Default::default()
        });
        assert!(!a.due(t(0.0)), "the first sample waits sample_s");
        assert!(a.due(t(60.0)));
        a.step(t(0.0), 1.0, 0.0);
        assert!(!a.due(t(30.0)));
        a.step(t(60.0), 0.0, 0.0);
        assert!((a.load.unwrap() - 0.5).abs() < 1e-9);
    }
}
