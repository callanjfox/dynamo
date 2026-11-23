// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! f49o `admission.batch_yield`: lower-class work gives way to the higher classes when, and as far
//! as, their attainment needs it. Off (absent) by default; any policy may turn it on.
//!
//! A yield level `y` in [0, 1] is stepped every `step_s`:
//!
//! - `trigger: attainment` (default): per watched class with a service level, its attainment over
//!   the scheduler's trailing window, smoothed over `smooth_s`; the raw level is the largest
//!   `(target + margin - smoothed) / span`, clamped to [0, 1]. A class on target plus margin asks
//!   for nothing, whatever arrives.
//! - `trigger: strict` (the control): 1 while a session of a watched class waits in the lobby,
//!   else 0: a strict class hierarchy.
//!
//! `y` moves toward the raw level by at most `max_step` a step (the rate limit). It acts on:
//!
//! - the gate (`admit: true`): a class ranked above `class` counts `class`'s KV times `1 - y`
//!   (the yield class's share of the KV budget, `admission.yield_class` scaled by `y`), and
//!   `class`'s own hot budget is scaled by `1 - y`, so its new sessions wait;
//! - the engines (`engine: true`), through the router's adaptive switches: while `y > 0` the
//!   classes ranked at or above `protect_rank` in the engine's class order preempt running
//!   lower-class requests for KV or sequence room (recompute mode; their blocks stay in the
//!   prefix cache and host tier), sparing one whose remaining output (its OSL) is at most
//!   `finish_s / decode_s` tokens; with `step_aside` the lower classes' decodes and prefill also
//!   leave a pass that would put the protected decodes past 90 % of their ITL target.
//!
//! With `y` at 0 every term is the identity: the discount is 0, the scale 1, the engine switch off.

use std::collections::HashMap;
use std::io::Write as _;

use serde::{Deserialize, Serialize};

use crate::clock::Time;
use crate::config::{ConfigError, ServicePolicy};

/// What raises the yield level.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum YieldTrigger {
    /// The watched classes' smoothed attainment against target plus margin.
    #[default]
    Attainment,
    /// Any watched-class session waiting in the lobby (the strict-hierarchy control).
    Strict,
}

/// `admission.batch_yield`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct BatchYieldConfig {
    /// The class that gives way. Default `batch`.
    pub class: String,
    /// The classes whose attainment (or waiting sessions) drive the level. Default
    /// `[priority, standard]`.
    pub watch: Vec<String>,
    pub trigger: YieldTrigger,
    /// Attainment above target the level aims to keep. Default 0.02.
    pub margin: f64,
    /// Shortfall below target plus margin at which the level is 1. Default 0.10.
    pub span: f64,
    /// Smoothing time constant of the attainment, seconds. Default 300.
    pub smooth_s: f64,
    /// Seconds between steps. Default 15.
    pub step_s: f64,
    /// Most the level moves in a step. Default 0.25.
    pub max_step: f64,
    /// Act on the gate (discount and own scale). Default true.
    pub admit: bool,
    /// Signal the engines. Default true.
    pub engine: bool,
    /// Index in the engine's class order at or above which classes are protected. Default 1
    /// (priority and standard of `priority,standard,batch`).
    pub protect_rank: usize,
    /// A running lower-class request finishing within this many seconds is not preempted.
    /// Default 1.0 (a third of priority's 3 s TTFT target).
    pub finish_s: f64,
    /// Decode step time used to turn `finish_s` into tokens. Default 0.035.
    pub decode_s: f64,
    /// The lower classes' decodes and prefill step aside for the protected ITL. Default true.
    pub step_aside: bool,
    /// A watched class still short by a whole `span` after this many seconds at level 1 is flagged
    /// infeasible and stops driving the level (yielding has not fixed it), until its smoothed
    /// attainment comes back within `span` of target plus margin. 0 (default): never flagged.
    pub infeasible_after_s: f64,
}

impl Default for BatchYieldConfig {
    fn default() -> Self {
        Self {
            class: "batch".into(),
            watch: vec!["priority".into(), "standard".into()],
            trigger: YieldTrigger::Attainment,
            margin: 0.02,
            span: 0.10,
            smooth_s: 300.0,
            step_s: 15.0,
            max_step: 0.25,
            admit: true,
            engine: true,
            protect_rank: 1,
            finish_s: 1.0,
            decode_s: 0.035,
            step_aside: true,
            infeasible_after_s: 0.0,
        }
    }
}

impl BatchYieldConfig {
    pub fn validate(&self, policy: &ServicePolicy) -> Result<(), ConfigError> {
        let err = |m: String| Err(ConfigError::BatchYield(m));
        let known = |c: &str| policy.classes.iter().any(|p| p.name == c);
        if !known(&self.class) {
            return err(format!("class {} is not a policy class", self.class));
        }
        for c in &self.watch {
            if !known(c) {
                return err(format!("watched class {c} is not a policy class"));
            }
            if c == &self.class {
                return err(format!("class {c} cannot watch itself"));
            }
        }
        if self.watch.is_empty() {
            return err("watch is empty".into());
        }
        for (name, v) in [
            ("span", self.span),
            ("smooth_s", self.smooth_s),
            ("step_s", self.step_s),
            ("max_step", self.max_step),
            ("decode_s", self.decode_s),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return err(format!("{name} must be positive"));
            }
        }
        if !(self.infeasible_after_s.is_finite() && self.infeasible_after_s >= 0.0) {
            return err("infeasible_after_s must be finite and at least 0".into());
        }
        if !(self.margin.is_finite() && self.finish_s.is_finite() && self.finish_s >= 0.0) {
            return err("margin and finish_s must be finite, finish_s at least 0".into());
        }
        Ok(())
    }

    /// `finish_s` in output tokens.
    pub fn finish_tokens(&self) -> usize {
        (self.finish_s / self.decode_s).floor() as usize
    }
}

/// One watched class's input to a step: its trailing attainment (`None` before any sample) and
/// its target (`None` without a service level).
#[derive(Clone, Debug, PartialEq)]
pub struct WatchInput {
    pub class: String,
    pub attainment: Option<f64>,
    pub target: Option<f64>,
}

/// The controller.
#[derive(Clone, Debug)]
pub struct BatchYield {
    cfg: BatchYieldConfig,
    level: f64,
    raw: f64,
    smoothed: HashMap<String, f64>,
    /// Since when each watched class has been short by a whole span with the level at 1.
    pinned_since: HashMap<String, Time>,
    /// Watched classes flagged infeasible (`infeasible_after_s`).
    infeasible: std::collections::BTreeSet<String>,
    last: Option<Time>,
    steps: u64,
    active_steps: u64,
    log: Option<String>,
}

impl BatchYield {
    pub fn from_policy(policy: &ServicePolicy) -> Option<Self> {
        let cfg = policy.admission.batch_yield.clone()?;
        Some(Self {
            cfg,
            level: 0.0,
            raw: 0.0,
            smoothed: HashMap::new(),
            pinned_since: HashMap::new(),
            infeasible: Default::default(),
            last: None,
            steps: 0,
            active_steps: 0,
            log: std::env::var("DYNAMO_BATCH_YIELD_LOG")
                .ok()
                .filter(|p| !p.is_empty()),
        })
    }

    pub fn config(&self) -> &BatchYieldConfig {
        &self.cfg
    }

    /// The yield level now.
    pub fn level(&self) -> f64 {
        self.level
    }

    pub fn due(&self, now: Time) -> bool {
        self.last
            .is_none_or(|t| now.since(t).as_secs_f64() >= self.cfg.step_s)
    }

    /// One step: smooth the watched classes' attainment, set the raw level, move the level.
    pub fn step(&mut self, now: Time, inputs: &[WatchInput], watched_waiting: bool) -> f64 {
        let dt = self
            .last
            .map_or(self.cfg.step_s, |t| now.since(t).as_secs_f64());
        self.last = Some(now);
        let alpha = 1.0 - (-dt / self.cfg.smooth_s).exp();
        let mut raw: f64 = 0.0;
        for input in inputs {
            let Some(a) = input.attainment else {
                continue;
            };
            let s = self
                .smoothed
                .entry(input.class.clone())
                .and_modify(|s| *s += alpha * (a - *s))
                .or_insert(a);
            let s = *s;
            if let Some(target) = input.target {
                let deficit = (target + self.cfg.margin - s) / self.cfg.span;
                if self.cfg.infeasible_after_s > 0.0 {
                    if deficit < 1.0 {
                        self.pinned_since.remove(&input.class);
                        self.infeasible.remove(&input.class);
                    } else if self.level >= 1.0 {
                        let since = *self.pinned_since.entry(input.class.clone()).or_insert(now);
                        if now.since(since).as_secs_f64() >= self.cfg.infeasible_after_s {
                            self.infeasible.insert(input.class.clone());
                        }
                    }
                }
                if !self.infeasible.contains(&input.class) {
                    raw = raw.max(deficit);
                }
            }
        }
        let raw = match self.cfg.trigger {
            YieldTrigger::Attainment => raw.clamp(0.0, 1.0),
            YieldTrigger::Strict => {
                if watched_waiting {
                    1.0
                } else {
                    0.0
                }
            }
        };
        self.raw = raw;
        let step = (raw - self.level).clamp(-self.cfg.max_step, self.cfg.max_step);
        self.level = (self.level + step).clamp(0.0, 1.0);
        self.steps += 1;
        if self.level > 0.0 {
            self.active_steps += 1;
        }
        self.write_log(now, watched_waiting);
        self.level
    }

    fn write_log(&self, now: Time, waiting: bool) {
        let Some(path) = self.log.as_deref() else {
            return;
        };
        let mut smoothed: Vec<_> = self.smoothed.iter().collect();
        smoothed.sort_by(|a, b| a.0.cmp(b.0));
        let smoothed: Vec<String> = smoothed
            .iter()
            .map(|(c, v)| format!("\"{c}\": {v:.4}"))
            .collect();
        let line = format!(
            "{{\"t_s\": {:.1}, \"level\": {:.4}, \"raw\": {:.4}, \"waiting\": {}, \"smoothed\": {{{}}}, \"infeasible\": [{}]}}\n",
            now.as_secs_f64(),
            self.level,
            self.raw,
            waiting,
            smoothed.join(", "),
            self.infeasible
                .iter()
                .map(|c| format!("\"{c}\""))
                .collect::<Vec<_>>()
                .join(", ")
        );
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = f.write_all(line.as_bytes());
        }
    }

    /// The engine signal: (level, protected rank, finish tokens, step aside).
    pub fn engine_signal(&self) -> (f64, Option<usize>, usize, bool) {
        (
            self.level,
            Some(self.cfg.protect_rank),
            self.cfg.finish_tokens(),
            self.cfg.step_aside,
        )
    }

    /// Steps taken and steps with the level up.
    pub fn counts(&self) -> (u64, u64) {
        (self.steps, self.active_steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn t(s: f64) -> Time {
        Time::ZERO.after(Duration::from_secs_f64(s))
    }

    fn controller(trigger: YieldTrigger) -> BatchYield {
        BatchYield {
            cfg: BatchYieldConfig {
                trigger,
                ..BatchYieldConfig::default()
            },
            level: 0.0,
            raw: 0.0,
            smoothed: HashMap::new(),
            pinned_since: HashMap::new(),
            infeasible: Default::default(),
            last: None,
            steps: 0,
            active_steps: 0,
            log: None,
        }
    }

    fn prio(a: f64) -> Vec<WatchInput> {
        vec![WatchInput {
            class: "priority".into(),
            attainment: Some(a),
            target: Some(0.9),
        }]
    }

    #[test]
    fn on_target_does_not_yield_even_with_waiting_work() {
        let mut c = controller(YieldTrigger::Attainment);
        for i in 0..10 {
            c.step(t(15.0 * f64::from(i)), &prio(0.95), true);
        }
        assert_eq!(c.level(), 0.0);
    }

    #[test]
    fn the_level_scales_with_the_shortfall_and_is_rate_limited() {
        let mut c = controller(YieldTrigger::Attainment);
        // 0.87 against 0.9 + 0.02: shortfall 0.05 over span 0.10, raw 0.5; at most 0.25 a step.
        assert!((c.step(t(0.0), &prio(0.87), false) - 0.25).abs() < 1e-9);
        assert!((c.step(t(15.0), &prio(0.87), false) - 0.5).abs() < 1e-9);
        assert!((c.step(t(30.0), &prio(0.87), false) - 0.5).abs() < 1e-9);
        // Far below target: the level reaches 1 and stays there.
        let mut c = controller(YieldTrigger::Attainment);
        for i in 0..8 {
            c.step(t(15.0 * f64::from(i)), &prio(0.5), false);
        }
        assert_eq!(c.level(), 1.0);
    }

    #[test]
    fn strict_follows_waiting_only() {
        let mut c = controller(YieldTrigger::Strict);
        c.step(t(0.0), &prio(0.99), true);
        assert!((c.level() - 0.25).abs() < 1e-9);
        let mut c = BatchYield {
            cfg: BatchYieldConfig {
                trigger: YieldTrigger::Strict,
                max_step: 1.0,
                ..BatchYieldConfig::default()
            },
            ..controller(YieldTrigger::Strict)
        };
        assert_eq!(c.step(t(0.0), &prio(0.99), true), 1.0);
        assert_eq!(c.step(t(15.0), &prio(0.5), false), 0.0);
    }

    #[test]
    fn steps_follow_step_s() {
        let mut c = controller(YieldTrigger::Attainment);
        assert!(c.due(t(0.0)));
        c.step(t(0.0), &prio(0.9), false);
        assert!(!c.due(t(10.0)));
        assert!(c.due(t(15.0)));
    }

    #[test]
    fn an_infeasible_class_stops_driving_the_level() {
        let mut c = controller(YieldTrigger::Attainment);
        c.cfg.infeasible_after_s = 60.0;
        let both = |p: f64, s: f64| {
            vec![
                WatchInput {
                    class: "priority".into(),
                    attainment: Some(p),
                    target: Some(0.9),
                },
                WatchInput {
                    class: "standard".into(),
                    attainment: Some(s),
                    target: Some(0.7),
                },
            ]
        };
        // Standard far below target: the level goes to 1 and stays while standard is not flagged.
        for i in 0..6 {
            c.step(t(15.0 * f64::from(i)), &both(0.95, 0.1), false);
        }
        assert_eq!(c.level(), 1.0);
        // 60 s at level 1 still short: flagged, and priority on target asks for nothing.
        for i in 6..20 {
            c.step(t(15.0 * f64::from(i)), &both(0.95, 0.1), false);
        }
        assert_eq!(c.level(), 0.0);
        // Back within a span of target: it drives the level again.
        let mut c2 = c.clone();
        c2.smoothed.insert("standard".into(), 0.7);
        c2.step(t(400.0), &both(0.95, 0.7), false);
        assert!(c2.level() > 0.0);
        // Off (0): never flagged.
        let mut d = controller(YieldTrigger::Attainment);
        for i in 0..20 {
            d.step(t(15.0 * f64::from(i)), &both(0.95, 0.1), false);
        }
        assert_eq!(d.level(), 1.0);
    }

    #[test]
    fn finish_tokens_from_seconds() {
        assert_eq!(BatchYieldConfig::default().finish_tokens(), 28);
    }
}
