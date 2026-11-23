// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! f50 mix selector (`mix_select` in the plugin's parameters; off when absent).
//!
//! Two policies won in opposite regimes: `b0` (forecast placement under an ITL limit, priority
//! past the router busy queue, priority waits cutting lower classes' budgets) when priority is
//! half the load or more, and `pg` (a reserved priority pool with held-time fairness for the other
//! classes) when priority is a small share. The selector keeps a smoothed estimate of one class's
//! share of the offered load and runs one policy's parts at a time:
//!
//! - the share is the class's demand over all classes' demand, demand being the sessions that
//!   are admitted (active or idle) or waiting in the lobby, counted in context tokens or sessions
//!   (`measure`); smoothed as an exponential moving average over `window_s`, sampled every
//!   `step_s`;
//! - hysteresis: to `pg` once the share is under `low`, to `b0` once it is over `high`;
//! - a switch needs `min_dwell_s` since the last one (and since the first step).
//!
//! What a mode switches is set per mode ([`ModeParts`]); every part is configured once in the
//! base `admission` block (the union of both modes), and a mode turns off the parts it does not
//! list. Their state (sessions, the lobby, the target loop's scale, the prefill meter, the session
//! signal) is kept across a switch: only whether each part acts changes. The worker-selection
//! side (pool, forecast route settings) reads [`current`]; the router's per-class busy threshold
//! is set through `dynamo_kv_router::plugins::adaptive_switches`.
//!
//! `DYNAMO_MIXSEL_LOG=<path>`: one JSON line per step (`"ev":"step"`: shares in tokens and
//! sessions, the smoothed share, the mode) and one per switch (`"ev":"switch"`).

use std::collections::BTreeMap;
use std::io::Write;
use std::sync::atomic::{AtomicU8, Ordering};

use serde::{Deserialize, Serialize};

use crate::clock::Time;
use crate::config::{LobbyOrder, ServicePolicy};

/// The two policies the selector runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MixMode {
    /// B0 + lower-tier hits (`tr4f_s_pq_cw_np_itl_lt`): for priority-heavy load.
    B0,
    /// pg_lt_fh (`stack7_mu0_pg_lt_fh`): for standard-heavy load.
    Pg,
}

impl MixMode {
    pub fn as_str(self) -> &'static str {
        match self {
            MixMode::B0 => "b0",
            MixMode::Pg => "pg",
        }
    }
}

/// What the share counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MixMeasure {
    /// Context tokens (the larger of the session's context and its pending input).
    #[default]
    Tokens,
    /// Sessions.
    Sessions,
}

/// The admission parts one mode runs. A part listed `false` is turned off in that mode (it must
/// be configured in the base `admission` block to run in the other).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ModeParts {
    pub lobby_order: LobbyOrder,
    pub retry_after_estimate: bool,
    pub prefill_budget: bool,
    pub target_loop: bool,
    pub session_signal: bool,
    pub starvation_override: bool,
    /// Per class, the lobby wait limit in this mode (classes not listed keep the base value).
    pub max_lobby_wait_s: BTreeMap<String, f64>,
    /// Per class, the router's `prefill_busy_threshold_frac` in this mode (match the router's
    /// `policy_classes` for the start mode). Classes not listed keep the router's value.
    pub router_busy_frac: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct MixSelectConfig {
    pub enabled: bool,
    /// The class whose share selects the mode.
    pub class: String,
    /// To `pg` under this share.
    pub low: f64,
    /// To `b0` over this share.
    pub high: f64,
    /// Smoothing window of the share, seconds.
    pub window_s: f64,
    /// Least seconds between switches (and from the first step to the first switch).
    pub min_dwell_s: f64,
    /// f50g: the dwell before the first switch, counted from the first sample past the warm-up
    /// (`None`: `min_dwell_s`).
    pub first_dwell_s: Option<f64>,
    /// f50g: a sample counts only once the demand holds at least this many sessions (admitted or
    /// waiting, every class); until then the share is not seeded and no switch can happen. 0: off.
    pub warmup_min_sessions: u64,
    /// Seconds between samples.
    pub step_s: f64,
    pub measure: MixMeasure,
    /// The mode before the first switch.
    pub start: MixMode,
    pub b0: ModeParts,
    pub pg: ModeParts,
}

impl Default for MixSelectConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            class: "priority".to_owned(),
            low: 0.35,
            high: 0.45,
            window_s: 300.0,
            min_dwell_s: 600.0,
            first_dwell_s: None,
            warmup_min_sessions: 0,
            step_s: 30.0,
            measure: MixMeasure::Tokens,
            start: MixMode::Pg,
            b0: ModeParts::default(),
            pg: ModeParts::default(),
        }
    }
}

impl MixSelectConfig {
    pub fn parts(&self, mode: MixMode) -> &ModeParts {
        match mode {
            MixMode::B0 => &self.b0,
            MixMode::Pg => &self.pg,
        }
    }

    pub fn validate(&self, base: &ServicePolicy) -> Result<(), String> {
        let unit = |x: f64| x.is_finite() && (0.0..=1.0).contains(&x);
        if !(unit(self.low) && unit(self.high) && self.low <= self.high) {
            return Err(format!(
                "low and high must be in [0, 1] with low <= high, got {} and {}",
                self.low, self.high
            ));
        }
        for (name, v) in [("window_s", self.window_s), ("step_s", self.step_s)] {
            if !(v.is_finite() && v > 0.0) {
                return Err(format!("{name} must be positive, got {v}"));
            }
        }
        if !(self.min_dwell_s.is_finite() && self.min_dwell_s >= 0.0) {
            return Err(format!(
                "min_dwell_s must not be negative, got {}",
                self.min_dwell_s
            ));
        }
        if let Some(f) = self.first_dwell_s
            && (!f.is_finite() || f < 0.0)
        {
            return Err(format!("first_dwell_s must not be negative, got {f}"));
        }
        if !base.classes.iter().any(|c| c.name == self.class) {
            return Err(format!("class {:?} is not one of the classes", self.class));
        }
        let a = &base.admission;
        for (mode, p) in [(MixMode::B0, &self.b0), (MixMode::Pg, &self.pg)] {
            let need = |on: bool, configured: bool, part: &str| {
                if on && !configured {
                    Err(format!(
                        "{}: {part} is on but admission.{part} is not configured",
                        mode.as_str()
                    ))
                } else {
                    Ok(())
                }
            };
            need(
                p.retry_after_estimate,
                a.retry_after_estimate.is_some(),
                "retry_after_estimate",
            )?;
            need(
                p.prefill_budget,
                a.prefill_budget.is_some(),
                "prefill_budget",
            )?;
            need(p.target_loop, a.target_loop.is_some(), "target_loop")?;
            need(
                p.session_signal,
                a.session_signal.is_some(),
                "session_signal",
            )?;
            need(
                p.starvation_override,
                a.starvation_override.is_some(),
                "starvation_override",
            )?;
            for (class, w) in &p.max_lobby_wait_s {
                if !base.classes.iter().any(|c| &c.name == class) || !w.is_finite() || *w < 0.0 {
                    return Err(format!("{}: max_lobby_wait_s {class:?} {w}", mode.as_str()));
                }
            }
            for (class, f) in &p.router_busy_frac {
                if !f.is_finite() || *f < 0.0 {
                    return Err(format!("{}: router_busy_frac {class:?} {f}", mode.as_str()));
                }
            }
        }
        Ok(())
    }
}

/// The policy one mode runs: the base (the union of both modes' parts) with the parts the mode
/// does not list turned off, its lobby order and its wait limits. The result has no `mix_select`.
pub fn mode_policy(base: &ServicePolicy, parts: &ModeParts) -> ServicePolicy {
    let mut p = base.clone();
    p.mix_select = None;
    let a = &mut p.admission;
    a.lobby_order = parts.lobby_order;
    if !parts.retry_after_estimate {
        a.retry_after_estimate = None;
    }
    if !parts.prefill_budget {
        a.prefill_budget = None;
    }
    if !parts.target_loop {
        a.target_loop = None;
    }
    if !parts.session_signal {
        a.session_signal = None;
    }
    if !parts.starvation_override {
        a.starvation_override = None;
    }
    for c in &mut p.classes {
        if let Some(w) = parts.max_lobby_wait_s.get(&c.name) {
            c.admission.max_lobby_wait_s = *w;
        }
    }
    p
}

/// The selector: the smoothed share, the mode and its hysteresis and dwell.
#[derive(Debug)]
pub struct Selector {
    low: f64,
    high: f64,
    window_s: f64,
    min_dwell_s: f64,
    first_dwell_s: f64,
    warmup_min_sessions: f64,
    step_s: f64,
    pub mode: MixMode,
    /// The smoothed share; `None` before the first sample with demand.
    pub share: Option<f64>,
    last: Option<Time>,
    /// The last switch, or the first step.
    since: Option<Time>,
    pub switches: u64,
}

impl Selector {
    pub fn new(cfg: &MixSelectConfig) -> Self {
        Self {
            low: cfg.low,
            high: cfg.high,
            window_s: cfg.window_s,
            min_dwell_s: cfg.min_dwell_s,
            first_dwell_s: cfg.first_dwell_s.unwrap_or(cfg.min_dwell_s),
            warmup_min_sessions: cfg.warmup_min_sessions as f64,
            step_s: cfg.step_s,
            mode: cfg.start,
            share: None,
            last: None,
            since: None,
            switches: 0,
        }
    }

    /// Whether a step is due at `now`.
    pub fn due(&self, now: Time) -> bool {
        self.last
            .is_none_or(|at| now.since(at).as_secs_f64() >= self.step_s)
    }

    /// One step with the share now (`None`: no demand, the estimate holds). Returns the new mode
    /// when it switches.
    pub fn step(&mut self, now: Time, raw: Option<f64>) -> Option<MixMode> {
        self.step_with(now, raw, f64::INFINITY)
    }

    /// One step with the share now and the sessions it was measured over (f50g warm-up: a sample
    /// over fewer than `warmup_min_sessions` sessions is ignored, so it neither seeds the share nor
    /// starts the first dwell).
    pub fn step_with(&mut self, now: Time, raw: Option<f64>, sessions: f64) -> Option<MixMode> {
        let dt = self
            .last
            .map_or(self.step_s, |at| now.since(at).as_secs_f64());
        self.last = Some(now);
        if sessions < self.warmup_min_sessions {
            return None;
        }
        self.since.get_or_insert(now);
        if let Some(x) = raw.filter(|x| x.is_finite()) {
            let k = 1.0 - (-dt / self.window_s).exp();
            self.share = Some(self.share.map_or(x, |s| s + k * (x - s)));
        }
        let share = self.share?;
        let dwelt = self.since.is_some_and(|at| {
            let dwell = if self.switches == 0 {
                self.first_dwell_s
            } else {
                self.min_dwell_s
            };
            now.since(at).as_secs_f64() >= dwell
        });
        let to = match self.mode {
            MixMode::Pg if share > self.high => MixMode::B0,
            MixMode::B0 if share < self.low => MixMode::Pg,
            _ => return None,
        };
        if !dwelt {
            return None;
        }
        self.mode = to;
        self.since = Some(now);
        self.switches += 1;
        Some(to)
    }
}

/// The class's share of demand, in tokens and in sessions: `(class, total)` sums for each.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Demand {
    pub class_tokens: f64,
    pub total_tokens: f64,
    pub class_sessions: f64,
    pub total_sessions: f64,
}

impl Demand {
    pub fn add(&mut self, of_class: bool, tokens: u64) {
        let t = tokens as f64;
        self.total_tokens += t;
        self.total_sessions += 1.0;
        if of_class {
            self.class_tokens += t;
            self.class_sessions += 1.0;
        }
    }

    pub fn tokens_share(&self) -> Option<f64> {
        (self.total_tokens > 0.0).then(|| self.class_tokens / self.total_tokens)
    }

    pub fn sessions_share(&self) -> Option<f64> {
        (self.total_sessions > 0.0).then(|| self.class_sessions / self.total_sessions)
    }

    pub fn share(&self, measure: MixMeasure) -> Option<f64> {
        match measure {
            MixMeasure::Tokens => self.tokens_share(),
            MixMeasure::Sessions => self.sessions_share(),
        }
    }
}

/// The selector with its two mode policies and its log, kept by the scheduler.
#[derive(Debug)]
pub struct MixState {
    pub config: MixSelectConfig,
    pub selector: Selector,
    /// The policy of each mode ([`mode_policy`]).
    pub b0: ServicePolicy,
    pub pg: ServicePolicy,
    /// The base admission block (both modes' parts): what bookkeeping that must survive a switch
    /// reads ([`crate::scheduler::Scheduler::union_admission`]).
    pub union: crate::config::AdmissionPolicy,
    log: Option<std::io::BufWriter<std::fs::File>>,
}

impl MixState {
    pub fn new(base: &ServicePolicy, config: MixSelectConfig) -> Self {
        let log = std::env::var_os("DYNAMO_MIXSEL_LOG")
            .and_then(|p| std::fs::File::create(p).ok())
            .map(std::io::BufWriter::new);
        Self {
            selector: Selector::new(&config),
            b0: mode_policy(base, &config.b0),
            pg: mode_policy(base, &config.pg),
            union: base.admission.clone(),
            config,
            log,
        }
    }

    pub fn policy(&self, mode: MixMode) -> &ServicePolicy {
        match mode {
            MixMode::B0 => &self.b0,
            MixMode::Pg => &self.pg,
        }
    }

    pub fn log_step(&mut self, now: Time, d: &Demand) {
        let Some(out) = self.log.as_mut() else {
            return;
        };
        let num = |x: Option<f64>| x.map_or("null".to_owned(), |v| format!("{v:.4}"));
        let _ = writeln!(
            out,
            "{{\"ev\":\"step\",\"t_s\":{:.1},\"share_tokens\":{},\"share_sessions\":{},\"sessions\":{},\"share\":{},\"mode\":\"{}\"}}",
            now.as_secs_f64(),
            num(d.tokens_share()),
            num(d.sessions_share()),
            d.total_sessions,
            num(self.selector.share),
            self.selector.mode.as_str(),
        );
    }

    pub fn log_switch(&mut self, now: Time, to: MixMode) {
        let Some(out) = self.log.as_mut() else {
            return;
        };
        let _ = writeln!(
            out,
            "{{\"ev\":\"switch\",\"t_s\":{:.1},\"to\":\"{}\",\"share\":{:.4},\"switches\":{}}}",
            now.as_secs_f64(),
            to.as_str(),
            self.selector.share.unwrap_or(f64::NAN),
            self.selector.switches,
        );
        let _ = out.flush();
    }
}

/// The current mode, process-wide, for the worker-selection side (0: no selector).
static MODE: AtomicU8 = AtomicU8::new(0);

/// The selector's mode, or `None` when no selector runs.
pub fn current() -> Option<MixMode> {
    match MODE.load(Ordering::Relaxed) {
        1 => Some(MixMode::B0),
        2 => Some(MixMode::Pg),
        _ => None,
    }
}

pub fn set_current(mode: Option<MixMode>) {
    let v = match mode {
        None => 0,
        Some(MixMode::B0) => 1,
        Some(MixMode::Pg) => 2,
    };
    MODE.store(v, Ordering::Relaxed);
}

/// Set the router's per-class busy thresholds for `parts` (classes not listed are left alone).
pub fn apply_router_busy(parts: &ModeParts) {
    for (class, frac) in &parts.router_busy_frac {
        dynamo_kv_router::plugins::adaptive_switches::set_class_prefill_busy_frac(
            class,
            Some(*frac),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{PrefillBudgetConfig, TargetLoopConfig};

    fn t(s: f64) -> Time {
        Time::from_secs_f64(s)
    }

    fn cfg() -> MixSelectConfig {
        MixSelectConfig {
            window_s: 300.0,
            min_dwell_s: 600.0,
            step_s: 30.0,
            ..MixSelectConfig::default()
        }
    }

    /// Feed `share` every step from `from` to `to` seconds; the switches seen.
    fn run(s: &mut Selector, from: f64, to: f64, share: f64) -> Vec<(f64, MixMode)> {
        let mut out = Vec::new();
        let mut x = from;
        while x <= to {
            if let Some(m) = s.step(t(x), Some(share)) {
                out.push((x, m));
            }
            x += 30.0;
        }
        out
    }

    #[test]
    fn starts_in_the_configured_mode_and_stays_inside_the_band() {
        let mut s = Selector::new(&cfg());
        assert_eq!(s.mode, MixMode::Pg);
        assert!(
            run(&mut s, 0.0, 7200.0, 0.40).is_empty(),
            "0.40 is inside the band"
        );
        let mut c = cfg();
        c.start = MixMode::B0;
        let mut s = Selector::new(&c);
        assert!(run(&mut s, 0.0, 7200.0, 0.40).is_empty());
        assert_eq!(s.mode, MixMode::B0);
    }

    #[test]
    fn switches_to_b0_over_high_only_after_the_dwell() {
        let mut s = Selector::new(&cfg());
        let sw = run(&mut s, 0.0, 3600.0, 0.8);
        assert_eq!(sw.len(), 1);
        assert_eq!(sw[0].1, MixMode::B0);
        assert!(sw[0].0 >= 600.0, "not before the first dwell: {:?}", sw);
        assert_eq!(s.switches, 1);
    }

    #[test]
    fn switches_back_to_pg_under_low_with_dwell_between_switches() {
        let mut s = Selector::new(&cfg());
        let up = run(&mut s, 0.0, 1200.0, 0.8);
        assert_eq!(up.len(), 1);
        let at = up[0].0;
        let down = run(&mut s, 1230.0, 6000.0, 0.1);
        assert_eq!(down.len(), 1);
        assert_eq!(down[0].1, MixMode::Pg);
        assert!(down[0].0 - at >= 600.0, "dwell between switches");
    }

    #[test]
    fn the_average_smooths_a_short_spike() {
        let mut s = Selector::new(&cfg());
        run(&mut s, 0.0, 1800.0, 0.1);
        // 60 s at 0.9 moves a 300 s average from 0.1 by under (1 - e^-0.2) * 0.8 ~ 0.15.
        let sw = run(&mut s, 1830.0, 1860.0, 0.9);
        assert!(sw.is_empty());
        assert!(s.share.unwrap() < 0.35, "{:?}", s.share);
        assert!(run(&mut s, 1890.0, 2400.0, 0.1).is_empty());
    }

    #[test]
    fn no_demand_holds_the_estimate() {
        let mut s = Selector::new(&cfg());
        assert_eq!(s.step(t(0.0), None), None);
        assert_eq!(s.share, None);
        run(&mut s, 30.0, 300.0, 0.2);
        let before = s.share;
        assert_eq!(s.step(t(330.0), None), None);
        assert_eq!(s.share, before);
    }

    #[test]
    fn warmup_ignores_samples_until_enough_sessions_and_seeds_after() {
        let mut c = cfg();
        c.warmup_min_sessions = 20;
        c.min_dwell_s = 120.0;
        let mut s = Selector::new(&c);
        // A cold start: one priority session gives a share of 1.0; ignored, nothing seeded.
        for x in [0.0, 30.0, 60.0, 90.0, 120.0, 150.0] {
            assert_eq!(s.step_with(t(x), Some(1.0), 3.0), None);
        }
        assert_eq!(s.share, None, "not seeded before the warm-up passes");
        assert_eq!(s.mode, MixMode::Pg);
        // Warm from 180 s at a share of 0.3: seeded at 0.3, no switch.
        let mut x = 180.0;
        while x <= 1800.0 {
            assert_eq!(s.step_with(t(x), Some(0.3), 50.0), None);
            x += 30.0;
        }
        assert!((s.share.unwrap() - 0.3).abs() < 1e-9);
        assert_eq!(s.switches, 0);
    }

    #[test]
    fn warmup_starts_the_first_dwell_at_the_first_warm_sample() {
        let mut c = cfg();
        c.warmup_min_sessions = 20;
        c.min_dwell_s = 120.0;
        let mut s = Selector::new(&c);
        for x in [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0] {
            assert_eq!(s.step_with(t(x), Some(0.9), 5.0), None);
        }
        // Warm from 240 s: the first switch waits 120 s from there, not from 0 s.
        let mut first = None;
        let mut x = 240.0;
        while x <= 900.0 {
            if let Some(m) = s.step_with(t(x), Some(0.9), 40.0) {
                first.get_or_insert((x, m));
            }
            x += 30.0;
        }
        assert_eq!(first, Some((360.0, MixMode::B0)));
    }

    #[test]
    fn first_dwell_applies_before_the_first_switch_only() {
        let mut c = cfg();
        c.min_dwell_s = 120.0;
        c.first_dwell_s = Some(300.0);
        c.window_s = 30.0;
        let mut s = Selector::new(&c);
        let up = run(&mut s, 0.0, 600.0, 0.9);
        assert_eq!(up.len(), 1);
        assert!(up[0].0 >= 300.0 && up[0].0 < 330.0, "{up:?}");
        let at = up[0].0;
        let down = run(&mut s, at + 30.0, at + 2000.0, 0.05);
        assert_eq!(down.len(), 1);
        assert!(
            down[0].0 - at >= 120.0 && down[0].0 - at < 180.0,
            "later switches use min_dwell_s: {down:?} after {at}"
        );
    }

    #[test]
    fn warmup_and_first_dwell_off_match_the_plain_selector() {
        let mut c = cfg();
        c.min_dwell_s = 120.0;
        let mut plain = Selector::new(&c);
        let mut guarded = Selector::new(&MixSelectConfig {
            warmup_min_sessions: 0,
            first_dwell_s: None,
            ..c.clone()
        });
        let shares = [1.0, 0.9, 0.2, 0.1, 0.8, 0.9, 0.95, 0.1, 0.05, 0.6, 0.3, 0.2];
        for (i, sh) in shares.iter().cycle().take(240).enumerate() {
            let now = t(i as f64 * 30.0);
            assert_eq!(
                plain.step(now, Some(*sh)),
                guarded.step_with(now, Some(*sh), 1.0)
            );
            assert_eq!(plain.share, guarded.share);
        }
        assert_eq!(plain.switches, guarded.switches);
    }

    #[test]
    fn first_dwell_must_not_be_negative() {
        let mut c = parts_pair();
        c.first_dwell_s = Some(-1.0);
        assert!(
            c.validate(&union_base())
                .unwrap_err()
                .contains("first_dwell_s")
        );
    }

    #[test]
    fn due_every_step_s() {
        let mut s = Selector::new(&cfg());
        assert!(s.due(t(0.0)));
        s.step(t(0.0), Some(0.5));
        assert!(!s.due(t(29.0)));
        assert!(s.due(t(30.0)));
    }

    #[test]
    fn demand_shares_in_tokens_and_sessions() {
        let mut d = Demand::default();
        assert_eq!(d.share(MixMeasure::Tokens), None);
        d.add(true, 300);
        d.add(false, 100);
        d.add(false, 100);
        assert!((d.share(MixMeasure::Tokens).unwrap() - 0.6).abs() < 1e-12);
        assert!((d.share(MixMeasure::Sessions).unwrap() - 1.0 / 3.0).abs() < 1e-12);
    }

    fn union_base() -> ServicePolicy {
        let mut p = ServicePolicy::default();
        p.admission.prefill_budget = Some(PrefillBudgetConfig::default());
        p.admission.target_loop = Some(TargetLoopConfig::default());
        p.admission.session_signal = Some(Default::default());
        p.admission.starvation_override = Some(Default::default());
        p
    }

    fn parts_pair() -> MixSelectConfig {
        let mut c = cfg();
        c.b0 = ModeParts {
            lobby_order: LobbyOrder::Aging,
            prefill_budget: true,
            target_loop: true,
            max_lobby_wait_s: [("priority".to_owned(), 25.0)].into_iter().collect(),
            router_busy_frac: [("priority".to_owned(), 4.0)].into_iter().collect(),
            ..ModeParts::default()
        };
        c.pg = ModeParts {
            lobby_order: LobbyOrder::Signal,
            session_signal: true,
            starvation_override: true,
            max_lobby_wait_s: [("priority".to_owned(), 30.0)].into_iter().collect(),
            router_busy_frac: [("priority".to_owned(), 1.0)].into_iter().collect(),
            ..ModeParts::default()
        };
        c
    }

    #[test]
    fn mode_policies_turn_off_the_other_modes_parts() {
        let base = union_base();
        let c = parts_pair();
        assert_eq!(c.validate(&base), Ok(()));
        let b0 = mode_policy(&base, &c.b0);
        assert_eq!(b0.admission.lobby_order, LobbyOrder::Aging);
        assert!(b0.admission.prefill_budget.is_some() && b0.admission.target_loop.is_some());
        assert!(
            b0.admission.session_signal.is_none() && b0.admission.starvation_override.is_none()
        );
        assert_eq!(b0.class(Some("priority")).admission.max_lobby_wait_s, 25.0);
        assert!(b0.mix_select.is_none());
        let pg = mode_policy(&base, &c.pg);
        assert_eq!(pg.admission.lobby_order, LobbyOrder::Signal);
        assert!(pg.admission.prefill_budget.is_none() && pg.admission.target_loop.is_none());
        assert!(
            pg.admission.session_signal.is_some() && pg.admission.starvation_override.is_some()
        );
        assert_eq!(pg.class(Some("priority")).admission.max_lobby_wait_s, 30.0);
        // Classes not listed keep the base limits.
        assert_eq!(
            pg.class(Some("standard")).admission.max_lobby_wait_s,
            base.class(Some("standard")).admission.max_lobby_wait_s
        );
    }

    #[test]
    fn validate_needs_each_listed_part_in_the_base() {
        let mut base = union_base();
        base.admission.target_loop = None;
        assert!(
            parts_pair()
                .validate(&base)
                .unwrap_err()
                .contains("target_loop")
        );
        let mut c = parts_pair();
        c.low = 0.6;
        c.high = 0.4;
        assert!(c.validate(&union_base()).is_err());
        let mut c = parts_pair();
        c.class = "nobody".to_owned();
        assert!(c.validate(&union_base()).is_err());
    }

    #[test]
    fn parses_from_yaml_keys() {
        let c: MixSelectConfig = serde_json::from_str(
            r#"{"low":0.3,"high":0.5,"window_s":120,"min_dwell_s":60,"measure":"sessions","start":"b0",
                "b0":{"lobby_order":"aging","target_loop":true,"router_busy_frac":{"priority":4.0}},
                "pg":{"lobby_order":"signal","starvation_override":true}}"#,
        )
        .unwrap();
        assert_eq!(c.start, MixMode::B0);
        assert_eq!(c.measure, MixMeasure::Sessions);
        assert_eq!(c.b0.router_busy_frac.get("priority"), Some(&4.0));
        assert!(c.pg.starvation_override && !c.pg.target_loop);
        assert_eq!(c.step_s, 30.0, "default");
    }
}
