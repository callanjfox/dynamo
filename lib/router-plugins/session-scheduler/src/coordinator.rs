// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The signal-driven coordinator (`controller` in the plugin's parameters; F25,
//! `docs/controller-config.md`). Off (absent) by default.
//!
//! Each step (every `step_s`, sampled with the adaptive rule's worker sample) it reads signals the
//! router has and decides, per part set to `auto`, whether that part may act:
//!
//! - **surge**: the fleet's queued prefill seconds (booked prefill tokens over the fleet's prefill
//!   rate from the step model) over the tightest TTFT limit of any class, smoothed over
//!   `smooth_s`; on above `surge_on`, off below `surge_off`, each change held `min_hold_s`.
//!   Thresholds are multiples of a class target, not a hardware number.
//! - **release** (`admission.release`) may grow a class's gate share only outside a surge: in the
//!   surge it cost standard (F18: 0.822 against 0.895 whole run at 1,664 users).
//! - **ITL limit** (`forecast.itl_limit`, meant with `itl_limit_lower_only`: lower classes' prefill
//!   kept off workers decoding a higher class, the higher class never steered): on while the
//!   top-ranked class's attainment (ITL-inclusive with `admission.judge_itl`) is under its target
//!   plus `itl_on`, or in a surge; off once it is over target plus `itl_off` outside a surge, so
//!   release can give the lower classes the capacity back (F22: priority 0.868 against 0.838 and
//!   ITL-only misses -21% at 1,040 users, at -4.5% on-target tokens when always on).
//! - **surge admission** (`admission.surge`) applies only in a surge.
//!
//! Self-gating parts are left to their own switches and only reported: the session gate (the
//! adaptive rule), the soft reserve (its own hysteresis on priority's attainment), and the release's
//! own headroom test. Deployment-layer parts (pool, engine priority, retain, KV move, exclusive
//! tiers) are plain configuration.
//!
//! In `shadow` mode every decision is computed and logged but no part is switched: a run must equal
//! the same policy without the controller. In `act` mode the decisions switch the parts. With
//! `DYNAMO_CONTROLLER_LOG=<path>`, one JSON line per step: the signals, the regime, each part's
//! state, what changed and why.

use std::io::Write;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

use crate::clock::Time;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControllerMode {
    /// Decide and log only.
    #[default]
    Shadow,
    /// Decide and switch the parts.
    Act,
}

/// How a part is set: decided by the coordinator, or fixed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartMode {
    #[default]
    Auto,
    On,
    Off,
}

impl PartMode {
    fn resolve(self, auto: bool) -> bool {
        match self {
            PartMode::Auto => auto,
            PartMode::On => true,
            PartMode::Off => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ControllerConfig {
    pub mode: ControllerMode,
    /// Seconds between steps.
    pub step_s: f64,
    /// Smoothing of the surge signal, seconds.
    pub smooth_s: f64,
    /// Least seconds between two changes of the surge regime.
    pub min_hold_s: f64,
    /// Surge on above this many times the tightest TTFT limit of queued prefill.
    pub surge_on: f64,
    /// Surge off below this many times it.
    pub surge_off: f64,
    /// The surge regime also needs the top class short: on only while its headroom is under
    /// `surge_short`, and it ends once the headroom is over `surge_clear` (as well as when queued
    /// prefill falls under `surge_off`). Queued prefill alone cannot tell a warm-start surge from
    /// steady heavy load (F25: 3x to 8x the TTFT limit all run at 1,664 users with priority on
    /// target). Off by default.
    pub surge_needs_short: bool,
    pub surge_short: f64,
    pub surge_clear: f64,
    /// ITL limit on below the top class's target plus this.
    pub itl_on: f64,
    /// ITL limit off above the top class's target plus this (outside a surge).
    pub itl_off: f64,
    pub release: PartMode,
    pub itl_limit: PartMode,
    pub surge_admission: PartMode,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            mode: ControllerMode::Shadow,
            step_s: 30.0,
            smooth_s: 120.0,
            min_hold_s: 300.0,
            surge_on: 2.0,
            surge_off: 1.0,
            surge_needs_short: false,
            surge_short: 0.0,
            surge_clear: 0.02,
            itl_on: 0.02,
            itl_off: 0.05,
            release: PartMode::Auto,
            itl_limit: PartMode::Auto,
            surge_admission: PartMode::Auto,
        }
    }
}

/// What the coordinator reads at a step.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Signals {
    /// Seconds the fleet's queued prefill takes to clear, when the step model knows.
    pub backlog_s: Option<f64>,
    /// The tightest TTFT limit of any class with a service level.
    pub ttft_limit_s: Option<f64>,
    /// Hot demand over the HBM budget.
    pub hot_frac: Option<f64>,
    /// Live sessions' context over HBM plus DRAM.
    pub total_frac: Option<f64>,
    /// The top-ranked class with a target: its attainment over the trailing window minus the
    /// target.
    pub top_headroom: Option<f64>,
    /// Per class with a target: attainment over the trailing window minus the target.
    pub headroom: Vec<(String, Option<f64>)>,
    /// Per class: a session waiting in the lobby.
    pub waiting: Vec<(String, bool)>,
    /// The session gate is on (the adaptive rule can switch it).
    pub gate_on: bool,
    /// Per class: sessions waiting, the longest current wait and the admission rate (F32;
    /// reported only: the lobby's order, rotation and Retry-After act inside the gate).
    pub lobby: Vec<crate::scheduler::LobbyStats>,
    /// Sessions rotated out so far (`admission.rotate`).
    pub rotations: u64,
    /// The total budget's accounting (F34, `admission.accounting`): measured against claimed
    /// usage per tier and the KV-action check, as one JSON object; `None` when it is off.
    pub accounting: Option<String>,
}

/// The parts' decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Parts {
    pub release: bool,
    pub itl_limit: bool,
    pub surge_admission: bool,
}

/// The coordinator's state.
#[derive(Debug)]
pub struct Coordinator {
    pub config: ControllerConfig,
    /// Smoothed queued prefill over the tightest TTFT limit.
    pub ratio: Option<f64>,
    pub surge: bool,
    surge_at: Option<Time>,
    pub parts: Parts,
    /// The ITL limit's own switch (headroom hysteresis), and when it last changed.
    itl_short: bool,
    itl_at: Option<Time>,
    last: Option<Time>,
    pub flips: u64,
    log: Option<std::io::BufWriter<std::fs::File>>,
}

impl Coordinator {
    pub fn new(config: ControllerConfig) -> Self {
        let parts = Self::resolve(&config, false, false);
        let log = std::env::var_os("DYNAMO_CONTROLLER_LOG")
            .and_then(|p| std::fs::File::create(p).ok())
            .map(std::io::BufWriter::new);
        let me = Self {
            config,
            ratio: None,
            surge: false,
            surge_at: None,
            parts,
            itl_short: false,
            itl_at: None,
            last: None,
            flips: 0,
            log,
        };
        me.apply();
        me
    }

    fn resolve(c: &ControllerConfig, surge: bool, itl_short: bool) -> Parts {
        Parts {
            release: c.release.resolve(!surge),
            itl_limit: c.itl_limit.resolve(surge || itl_short),
            surge_admission: c.surge_admission.resolve(surge),
        }
    }

    /// Switch the parts (act mode only; shadow leaves every part as configured).
    fn apply(&self) {
        let on = self.config.mode == ControllerMode::Act;
        set(release_flag(), !on || self.parts.release);
        set(itl_limit_flag(), !on || self.parts.itl_limit);
        set(surge_flag(), !on || self.parts.surge_admission);
    }

    /// Whether a step is due at `now`.
    pub fn due(&self, now: Time) -> bool {
        self.last
            .is_none_or(|at| now.since(at).as_secs_f64() >= self.config.step_s)
    }

    /// One step. Returns whether a part changed.
    pub fn step(&mut self, now: Time, s: &Signals) -> bool {
        let dt = self
            .last
            .map_or(self.config.step_s, |at| now.since(at).as_secs_f64());
        self.last = Some(now);
        let raw = match (s.backlog_s, s.ttft_limit_s) {
            (Some(b), Some(l)) if l > 0.0 => Some(b / l),
            _ => None,
        };
        if let Some(x) = raw {
            let k = 1.0 - (-dt / self.config.smooth_s.max(1e-9)).exp();
            self.ratio = Some(self.ratio.map_or(x, |r| r + k * (x - r)));
        }
        let held = self
            .surge_at
            .is_some_and(|at| now.since(at).as_secs_f64() < self.config.min_hold_s);
        let was_surge = self.surge;
        let mut why = Vec::new();
        if let Some(r) = self.ratio
            && !held
        {
            let c = &self.config;
            let short = !c.surge_needs_short || s.top_headroom.is_some_and(|h| h < c.surge_short);
            let clear = c.surge_needs_short && s.top_headroom.is_some_and(|h| h > c.surge_clear);
            if !self.surge && r > c.surge_on && short {
                self.surge = true;
                why.push(format!(
                    "surge on: queued prefill {r:.2}x the tightest TTFT limit"
                ));
            } else if self.surge && (r < c.surge_off || clear) {
                self.surge = false;
                why.push(if clear {
                    format!("surge off: top class over target, queued prefill {r:.2}x")
                } else {
                    format!("surge off: queued prefill {r:.2}x the tightest TTFT limit")
                });
            }
        }
        if was_surge != self.surge {
            self.surge_at = Some(now);
        }
        let itl_held = self
            .itl_at
            .is_some_and(|at| now.since(at).as_secs_f64() < self.config.min_hold_s);
        if let Some(h) = s.top_headroom
            && !itl_held
        {
            if !self.itl_short && h < self.config.itl_on {
                self.itl_short = true;
                self.itl_at = Some(now);
                why.push(format!("top class short: headroom {h:+.3}"));
            } else if self.itl_short && h > self.config.itl_off {
                self.itl_short = false;
                self.itl_at = Some(now);
                why.push(format!("top class has headroom {h:+.3}"));
            }
        }
        let before = self.parts;
        self.parts = Self::resolve(&self.config, self.surge, self.itl_short);
        let changed = before != self.parts;
        if changed {
            self.flips += u64::from(before.release != self.parts.release)
                + u64::from(before.itl_limit != self.parts.itl_limit)
                + u64::from(before.surge_admission != self.parts.surge_admission);
            self.apply();
        }
        self.write(now, s, raw, &before, &why);
        changed
    }

    fn write(&mut self, now: Time, s: &Signals, raw: Option<f64>, before: &Parts, why: &[String]) {
        let Some(out) = self.log.as_mut() else {
            return;
        };
        let num = |x: Option<f64>| x.map_or("null".to_owned(), |v| format!("{v:.4}"));
        let headroom: Vec<String> = s
            .headroom
            .iter()
            .map(|(c, h)| format!("{c:?}:{}", num(*h)))
            .collect();
        let waiting: Vec<String> = s
            .waiting
            .iter()
            .map(|(c, w)| format!("{c:?}:{w}"))
            .collect();
        let lobby: Vec<String> = s
            .lobby
            .iter()
            .map(|l| {
                format!(
                    "{:?}:{{\"waiting\":{},\"oldest_wait_s\":{:.1},\"admit_per_min\":{}}}",
                    l.class,
                    l.waiting,
                    l.oldest_wait_s,
                    num(l.admit_rate.map(|r| r * 60.0)),
                )
            })
            .collect();
        let mut changed = Vec::new();
        if before.release != self.parts.release {
            changed.push("\"release\"");
        }
        if before.itl_limit != self.parts.itl_limit {
            changed.push("\"itl_limit\"");
        }
        if before.surge_admission != self.parts.surge_admission {
            changed.push("\"surge_admission\"");
        }
        let why: Vec<String> = why.iter().map(|w| format!("{w:?}")).collect();
        let _ = writeln!(
            out,
            "{{\"t_s\":{:.1},\"mode\":\"{}\",\"signals\":{{\"backlog_s\":{},\"ttft_limit_s\":{},\"surge_ratio_raw\":{},\"surge_ratio\":{},\"top_headroom\":{},\"hot_frac\":{},\"total_frac\":{},\"gate_on\":{},\"headroom\":{{{}}},\"waiting\":{{{}}},\"lobby\":{{{}}},\"rotations\":{},\"accounting\":{}}},\"regime\":\"{}\",\"parts\":{{\"release\":{},\"itl_limit\":{},\"surge_admission\":{}}},\"changed\":[{}],\"why\":[{}]}}",
            now.as_secs_f64(),
            if self.config.mode == ControllerMode::Act {
                "act"
            } else {
                "shadow"
            },
            num(s.backlog_s),
            num(s.ttft_limit_s),
            num(raw),
            num(self.ratio),
            num(s.top_headroom),
            num(s.hot_frac),
            num(s.total_frac),
            s.gate_on,
            headroom.join(","),
            waiting.join(","),
            lobby.join(","),
            s.rotations,
            s.accounting.as_deref().unwrap_or("null"),
            if self.surge {
                "surge"
            } else if !s.gate_on {
                "light"
            } else {
                "steady"
            },
            self.parts.release,
            self.parts.itl_limit,
            self.parts.surge_admission,
            changed.join(","),
            why.join(","),
        );
        let _ = out.flush();
    }
}

fn set(flag: &AtomicBool, on: bool) {
    flag.store(on, Ordering::Relaxed);
}

fn release_flag() -> &'static AtomicBool {
    static FLAG: OnceLock<AtomicBool> = OnceLock::new();
    FLAG.get_or_init(|| AtomicBool::new(true))
}

fn itl_limit_flag() -> &'static AtomicBool {
    static FLAG: OnceLock<AtomicBool> = OnceLock::new();
    FLAG.get_or_init(|| AtomicBool::new(true))
}

fn surge_flag() -> &'static AtomicBool {
    static FLAG: OnceLock<AtomicBool> = OnceLock::new();
    FLAG.get_or_init(|| AtomicBool::new(true))
}

/// Whether surplus release may grow a class's share (on when no coordinator acts).
pub fn release_enabled() -> bool {
    release_flag().load(Ordering::Relaxed)
}

/// Whether the ITL placement limit applies (on when no coordinator acts).
pub fn itl_limit_enabled() -> bool {
    itl_limit_flag().load(Ordering::Relaxed)
}

/// Whether surge admission may hold sessions (on when no coordinator acts).
pub fn surge_admission_enabled() -> bool {
    surge_flag().load(Ordering::Relaxed)
}

/// Reset every part to on (a classifier without a coordinator, or before one starts).
pub fn reset() {
    set(release_flag(), true);
    set(itl_limit_flag(), true);
    set(surge_flag(), true);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: f64) -> Time {
        Time::from_secs_f64(s)
    }

    fn sig(backlog: f64) -> Signals {
        Signals {
            backlog_s: Some(backlog),
            ttft_limit_s: Some(3.0),
            gate_on: true,
            ..Default::default()
        }
    }

    fn quick(mode: ControllerMode) -> ControllerConfig {
        ControllerConfig {
            mode,
            smooth_s: 1e-6,
            min_hold_s: 100.0,
            ..Default::default()
        }
    }

    #[test]
    fn surge_switches_release_off_and_the_itl_limit_and_surge_admission_on_with_hysteresis() {
        let mut c = Coordinator::new(quick(ControllerMode::Shadow));
        assert!(!c.surge && c.parts.release && !c.parts.itl_limit && !c.parts.surge_admission);
        // 30 s of queued prefill against a 3 s limit: 10x, surge on.
        assert!(c.step(t(0.0), &sig(30.0)));
        assert!(c.surge && !c.parts.release && c.parts.itl_limit && c.parts.surge_admission);
        // Between surge_off and surge_on: stays in surge.
        assert!(!c.step(t(200.0), &sig(4.5)));
        assert!(c.surge);
        // Under surge_off, but inside the hold after the last change: stays.
        let mut d = Coordinator::new(quick(ControllerMode::Shadow));
        d.step(t(0.0), &sig(30.0));
        d.step(t(50.0), &sig(0.5));
        assert!(d.surge, "held for min_hold_s");
        // After the hold: out of the surge.
        assert!(c.step(t(400.0), &sig(0.5)));
        assert!(!c.surge && c.parts.release && !c.parts.itl_limit && !c.parts.surge_admission);
        assert_eq!(c.flips, 6);
    }

    #[test]
    fn the_itl_limit_follows_the_top_class_headroom_outside_a_surge() {
        let mut c = Coordinator::new(quick(ControllerMode::Shadow));
        let head = |h: f64| Signals {
            top_headroom: Some(h),
            ..sig(0.0)
        };
        // Priority 1 point under target + itl_on: on.
        assert!(c.step(t(0.0), &head(0.01)));
        assert!(c.parts.itl_limit && c.parts.release);
        // Between itl_on and itl_off: stays on.
        assert!(!c.step(t(200.0), &head(0.03)));
        // Over itl_off, but inside the hold: stays on; after it: off.
        let mut d = Coordinator::new(quick(ControllerMode::Shadow));
        d.step(t(0.0), &head(0.0));
        d.step(t(50.0), &head(0.2));
        assert!(d.parts.itl_limit, "held for min_hold_s");
        assert!(c.step(t(400.0), &head(0.08)));
        assert!(!c.parts.itl_limit);
        // Unknown headroom changes nothing.
        assert!(!c.step(t(800.0), &sig(0.0)));
    }

    #[test]
    fn fixed_parts_ignore_the_regime_and_unknown_signals_change_nothing() {
        let mut c = Coordinator::new(ControllerConfig {
            release: PartMode::On,
            surge_admission: PartMode::Off,
            ..quick(ControllerMode::Shadow)
        });
        c.step(t(0.0), &sig(30.0));
        assert!(c.surge && c.parts.release && !c.parts.surge_admission && c.parts.itl_limit);
        let mut d = Coordinator::new(quick(ControllerMode::Shadow));
        assert!(!d.step(t(0.0), &Signals::default()));
        assert!(!d.surge && d.ratio.is_none());
    }

    #[test]
    fn with_surge_needs_short_a_long_queue_is_a_surge_only_while_the_top_class_is_short() {
        let mut c = Coordinator::new(ControllerConfig {
            surge_needs_short: true,
            ..quick(ControllerMode::Shadow)
        });
        let s = |b: f64, h: f64| Signals {
            top_headroom: Some(h),
            ..sig(b)
        };
        // Long queue, priority on target: steady heavy load, not a surge.
        c.step(t(0.0), &s(30.0, 0.03));
        assert!(!c.surge);
        // Long queue and priority short: surge.
        c.step(t(200.0), &s(30.0, -0.2));
        assert!(c.surge);
        // Queue still long, priority back over target + surge_clear (after the hold): ends.
        c.step(t(600.0), &s(30.0, 0.05));
        assert!(!c.surge);
    }

    #[test]
    fn steps_are_spaced() {
        let mut c = Coordinator::new(quick(ControllerMode::Shadow));
        assert!(c.due(t(0.0)));
        c.step(t(0.0), &sig(0.0));
        assert!(!c.due(t(10.0)));
        assert!(c.due(t(30.0)));
    }
}
