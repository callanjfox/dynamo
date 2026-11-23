// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! f49q: a per-session starvation signal, after Duru (Lee et al., arXiv 2611.23558), with two
//! uses, both off (absent) by default:
//!
//! - `admission.starvation_override`: a session waiting in the lobby, of a listed class, whose
//!   signal says it is starved is admitted whatever its class's budgets say (its KV share, the
//!   prefill share, the target loop, batch yield, the SLA scales, the order of its class), as long
//!   as some worker is under its `max_num_seqs` and it fits what the classes ranked above it keep:
//!   the total KV budget and the top class's hot budget (at its own scale), less the input of the
//!   higher-ranked sessions waiting and the KV the live tickets hold. At most `per_worker_per_min`
//!   overrides per worker per minute (a token bucket over the fleet).
//! - `admission.lobby_order: signal`: the lobby of each class in `session_signal.order_classes`
//!   (default `[standard, batch]`) is ordered by the signal, most starved first, instead of by
//!   arrival; the other classes keep arrival order. Class ranks still order the classes.
//!
//! The signal (`admission.session_signal.signal`):
//!
//! - `held_time` (default): seconds the session has spent held in the lobby, its earlier waits
//!   and its current one, a refused session's time away until it retries included. Starved over
//!   `held_s` (300); longest held first.
//! - `session_attainment`: the share of the session's judged turns (its root's, subagents
//!   included) that met their class's TTFT and ITL limits, the session log's per-request rule.
//!   Starved under its class's `service_level.attainment` less `attainment_gap` (0.3) once
//!   `min_turns` (3) turns are judged; a class without a service level is never starved by it.
//!   Lowest first (a session with no judged turn counts as 1).
//! - `slowdown`: Duru's normalized slowdown, the session's (root's) wall time less its client
//!   gaps (time with nothing of it in flight, a retry backoff excepted) over its standalone
//!   service: the uncontended time ([`crate::uncontended`], batch 1, the step model the forecast
//!   uses) of its completed requests, at their actual output (the class's mean output when not
//!   reported), plus its pending turn's at the client's OSL hint or the class's mean output.
//!   Starved over `threshold` (5); highest first.
//!
//! The lobby keys are the sessions' `order_key`, recomputed for every waiting session at most
//! every [`REFRESH_S`] seconds at a gate check, and for a session as it enters the lobby.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;

use serde::{Deserialize, Serialize};

use crate::clock::Time;
use crate::config::{ConfigError, ServicePolicy};
use crate::forecast::StepModel;

/// Seconds between recomputations of the lobby keys (`lobby_order: signal`).
pub const REFRESH_S: f64 = 10.0;

/// The environment variable naming the override log (`DYNAMO_TICKETS_LOG` serves when unset).
pub const STARVATION_LOG_ENV: &str = "DYNAMO_STARVATION_LOG";

/// Which measure says a session is starved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalKind {
    #[default]
    HeldTime,
    SessionAttainment,
    Slowdown,
}

impl SignalKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HeldTime => "held_time",
            Self::SessionAttainment => "session_attainment",
            Self::Slowdown => "slowdown",
        }
    }
}

/// `admission.session_signal` (f49q): the signal and its starvation lines, read by
/// `starvation_override` and `lobby_order: signal`. Absent: the defaults.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SessionSignalConfig {
    pub signal: SignalKind,
    /// `held_time`: starved over this many seconds held. Default 300.
    pub held_s: f64,
    /// `session_attainment`: starved under the class's target less this. Default 0.3.
    pub attainment_gap: f64,
    /// `session_attainment`: judged turns needed before it can be starved. Default 3.
    pub min_turns: u32,
    /// `slowdown`: starved over this slowdown. Default 5.
    pub threshold: f64,
    /// Classes `lobby_order: signal` orders by the signal. Default `[standard, batch]`.
    pub order_classes: Vec<String>,
}

impl Default for SessionSignalConfig {
    fn default() -> Self {
        Self {
            signal: SignalKind::default(),
            held_s: 300.0,
            attainment_gap: 0.3,
            min_turns: 3,
            threshold: 5.0,
            order_classes: vec!["standard".to_owned(), "batch".to_owned()],
        }
    }
}

/// `admission.starvation_override` (f49q).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct StarvationOverrideConfig {
    /// Overrides per worker per minute. Default 2.
    pub per_worker_per_min: f64,
    /// Classes it may admit. Default `[standard, batch]`.
    pub classes: Vec<String>,
}

impl Default for StarvationOverrideConfig {
    fn default() -> Self {
        Self {
            per_worker_per_min: 2.0,
            classes: vec!["standard".to_owned(), "batch".to_owned()],
        }
    }
}

fn positive(v: f64) -> bool {
    v.is_finite() && v > 0.0
}

impl SessionSignalConfig {
    pub fn validate(&self, policy: &ServicePolicy) -> Result<(), ConfigError> {
        let fail = |reason: String| Err(ConfigError::SessionSignal(reason));
        if !positive(self.held_s) {
            return fail(format!("held_s must be positive, got {}", self.held_s));
        }
        if !(self.attainment_gap.is_finite() && (0.0..=1.0).contains(&self.attainment_gap)) {
            return fail(format!(
                "attainment_gap must be in [0, 1], got {}",
                self.attainment_gap
            ));
        }
        if self.min_turns == 0 {
            return fail("min_turns must be at least 1".to_owned());
        }
        if !positive(self.threshold) {
            return fail(format!(
                "threshold must be positive, got {}",
                self.threshold
            ));
        }
        for (i, class) in self.order_classes.iter().enumerate() {
            if !policy.classes.iter().any(|c| &c.name == class) {
                return fail(format!(
                    "order_classes: {class:?} is not one of the classes"
                ));
            }
            if self.order_classes[..i].contains(class) {
                return fail(format!("order_classes: {class:?} is listed twice"));
            }
        }
        Ok(())
    }

    /// The lobby key of a reading (lower first): the most starved first.
    pub fn order_key(&self, reading: Reading) -> f64 {
        match reading {
            Reading::HeldS(held) => -held,
            Reading::Attainment { .. } => reading.value().unwrap_or(1.0),
            Reading::Slowdown(s) => -s.unwrap_or(0.0),
        }
    }

    /// Whether a reading says the session is starved; `target` is its class's attainment target.
    pub fn starved(&self, reading: Reading, target: Option<f64>) -> bool {
        match reading {
            Reading::HeldS(held) => held > self.held_s,
            Reading::Attainment { judged, .. } => {
                judged >= self.min_turns
                    && target.is_some_and(|t| {
                        reading.value().is_some_and(|a| a < t - self.attainment_gap)
                    })
            }
            Reading::Slowdown(s) => s.is_some_and(|s| s > self.threshold),
        }
    }
}

impl StarvationOverrideConfig {
    pub fn validate(&self, policy: &ServicePolicy) -> Result<(), ConfigError> {
        let fail = |reason: String| Err(ConfigError::StarvationOverride(reason));
        if !positive(self.per_worker_per_min) {
            return fail(format!(
                "per_worker_per_min must be positive, got {}",
                self.per_worker_per_min
            ));
        }
        if self.classes.is_empty() {
            return fail("classes must list at least one class".to_owned());
        }
        for (i, class) in self.classes.iter().enumerate() {
            if !policy.classes.iter().any(|c| &c.name == class) {
                return fail(format!("class {class:?} is not one of the classes"));
            }
            if self.classes[..i].contains(class) {
                return fail(format!("class {class:?} is listed twice"));
            }
        }
        Ok(())
    }
}

/// One session's signal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reading {
    /// Seconds held in the lobby so far.
    HeldS(f64),
    /// Judged turns, and of those on target.
    Attainment { on_target: u32, judged: u32 },
    /// Its slowdown; `None` when its standalone service is unknown (no step model).
    Slowdown(Option<f64>),
}

impl Reading {
    /// The signal's value (`None`: an attainment with no judged turn, or an unknown slowdown).
    pub fn value(self) -> Option<f64> {
        match self {
            Self::HeldS(held) => Some(held),
            Self::Attainment { on_target, judged } => {
                (judged > 0).then(|| f64::from(on_target) / f64::from(judged))
            }
            Self::Slowdown(s) => s,
        }
    }
}

/// Active time over standalone service; `None` without any standalone service.
pub fn slowdown(active_s: f64, expected_s: f64) -> Option<f64> {
    (expected_s > 0.0).then(|| active_s.max(0.0) / expected_s)
}

/// A turn's standalone time: `new` uncached input tokens after `prefix` cached ones, then `osl`
/// output tokens, alone on a worker ([`crate::uncontended::request_s`]).
pub fn expected_turn_s(
    model: &dyn StepModel,
    chunk: u64,
    new: u64,
    prefix: u64,
    osl: u64,
) -> Option<f64> {
    crate::uncontended::request_s(model, chunk, prefix.saturating_add(new), prefix, osl)
}

/// The signal's running state: the classes' mean output (for `slowdown`), the lobby keys'
/// refresh, the override's token bucket and its log.
#[derive(Debug, Default)]
pub struct SignalState {
    /// Per class, (output tokens, completions) of the completions that reported their output.
    osl: HashMap<String, (u64, u64)>,
    refreshed: Option<Time>,
    workers: u64,
    bucket: f64,
    bucket_at: Option<Time>,
    /// The minute the counts are for, and the overrides by class and the denials in it.
    minute: Option<u64>,
    overrides: BTreeMap<String, u64>,
    denied: u64,
    logging: bool,
    file: Option<std::fs::File>,
    lines: Vec<String>,
}

impl SignalState {
    /// With the override log open: `DYNAMO_STARVATION_LOG`, else the tickets log (written through
    /// the tickets when they have it open, `tickets_logging`; appended to otherwise).
    pub fn new(tickets_logging: bool) -> Self {
        let file = match std::env::var_os(STARVATION_LOG_ENV).filter(|p| !p.is_empty()) {
            Some(path) => std::fs::File::create(path).ok(),
            None if !tickets_logging => std::env::var_os("DYNAMO_TICKETS_LOG")
                .filter(|p| !p.is_empty())
                .and_then(|path| {
                    std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(path)
                        .ok()
                }),
            None => None,
        };
        Self {
            logging: tickets_logging || file.is_some(),
            file,
            ..Default::default()
        }
    }

    /// A completion of `class` reported `tokens` of output.
    pub fn note_output(&mut self, class: &str, tokens: u64) {
        let e = self.osl.entry(class.to_owned()).or_default();
        e.0 = e.0.saturating_add(tokens);
        e.1 += 1;
    }

    /// The mean output of `class`'s completions so far, rounded.
    pub fn mean_osl(&self, class: &str) -> Option<u64> {
        self.osl
            .get(class)
            .filter(|(_, n)| *n > 0)
            .map(|(sum, n)| (*sum as f64 / *n as f64).round() as u64)
    }

    /// Whether the lobby keys are due for a recomputation at `now` (it is then marked done).
    /// f50: make the next [`Self::refresh_due`] true (a mix selector switch to `lobby_order: signal`).
    pub fn force_refresh(&mut self) {
        self.refreshed = None;
    }

    pub fn refresh_due(&mut self, now: Time) -> bool {
        if self
            .refreshed
            .is_some_and(|at| now.since(at).as_secs_f64() < REFRESH_S)
        {
            return false;
        }
        self.refreshed = Some(now);
        true
    }

    /// The fleet's worker count (the bucket's size).
    pub fn set_workers(&mut self, workers: u64) {
        self.workers = workers;
    }

    /// Take one override from the bucket: `per_worker_per_min` x workers a minute, holding one
    /// minute's worth (at least one), full at the start.
    pub fn take(&mut self, now: Time, per_worker_per_min: f64) -> bool {
        let rate = self.workers.max(1) as f64 * per_worker_per_min;
        let size = rate.max(1.0);
        self.bucket = match self.bucket_at {
            None => size,
            Some(at) => (self.bucket + rate / 60.0 * now.since(at).as_secs_f64()).min(size),
        };
        self.bucket_at = Some(now);
        if self.bucket >= 1.0 {
            self.bucket -= 1.0;
            true
        } else {
            false
        }
    }

    /// Close the minute before `now`'s when it had overrides or denials (its line, when logging).
    pub fn roll(&mut self, now: Time) {
        let minute = (now.as_secs_f64() / 60.0).floor() as u64;
        match self.minute {
            Some(m) if m == minute => return,
            Some(m) if self.logging && (!self.overrides.is_empty() || self.denied > 0) => {
                let counts = self
                    .overrides
                    .iter()
                    .map(|(c, n)| format!("{c:?}:{n}"))
                    .collect::<Vec<_>>()
                    .join(",");
                self.lines.push(format!(
                    "{{\"event\":\"starvation_minute\",\"t\":{:.1},\"overrides\":{{{counts}}},\"denied\":{},\"workers\":{}}}",
                    (m + 1) as f64 * 60.0,
                    self.denied,
                    self.workers
                ));
            }
            _ => {}
        }
        self.minute = Some(minute);
        self.overrides.clear();
        self.denied = 0;
    }

    /// An override of `session` of `class` at `now`, its signal at `value`.
    pub fn record(
        &mut self,
        now: Time,
        session: &str,
        class: &str,
        signal: SignalKind,
        value: Option<f64>,
    ) {
        self.roll(now);
        *self.overrides.entry(class.to_owned()).or_default() += 1;
        if self.logging {
            let value = value.map_or_else(|| "null".to_owned(), |v| format!("{v:.3}"));
            self.lines.push(format!(
                "{{\"event\":\"starvation_override\",\"t\":{:.3},\"session\":{session:?},\"class\":{class:?},\"signal\":\"{}\",\"value\":{value}}}",
                now.as_secs_f64(),
                signal.as_str()
            ));
        }
    }

    /// A starved session the bucket held back at `now`.
    pub fn denied(&mut self, now: Time) {
        self.roll(now);
        self.denied += 1;
    }

    /// The overrides by class in the current minute.
    pub fn minute_overrides(&self) -> &BTreeMap<String, u64> {
        &self.overrides
    }

    /// Write the pending lines to the override log, or through `tickets` (its log).
    pub fn flush(&mut self, tickets: &mut crate::tickets::Tickets) {
        if self.lines.is_empty() {
            return;
        }
        let lines = std::mem::take(&mut self.lines);
        for line in lines {
            match self.file.as_mut() {
                Some(file) => {
                    let _ = writeln!(file, "{line}");
                }
                None => tickets.log_line(&line),
            }
        }
    }

    /// The pending log lines (tests).
    #[cfg(test)]
    pub fn take_lines(&mut self) -> Vec<String> {
        std::mem::take(&mut self.lines)
    }

    /// Turn the log lines on without a file (tests).
    #[cfg(test)]
    pub fn log_in_memory(&mut self) {
        self.logging = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecast::TableStepModel;

    fn t(s: f64) -> Time {
        Time::from_secs_f64(s)
    }

    /// Prefill 1 s per 1,000 new tokens, decode 10 ms a token.
    fn unit() -> TableStepModel {
        TableStepModel {
            prefill_tokens: vec![0.0, 1000.0],
            prefill_s: vec![0.0, 1.0],
            prefix_weight: 0.0,
            decode_batch: vec![1.0, 2.0],
            decode_context: vec![1000.0, 2000.0],
            decode_s: vec![0.01, 0.01, 0.01, 0.01],
        }
    }

    #[test]
    fn f49q_slowdown_is_active_time_over_the_step_models_standalone_time() {
        let m = unit();
        // 1,000 new tokens after a 5,000-token cached prefix (free at prefix weight 0) and 101
        // output tokens: 1 s of prefill and 100 decode steps of 10 ms.
        let e = expected_turn_s(&m, 8192, 1000, 5000, 101).unwrap();
        assert!((e - 2.0).abs() < 1e-9, "{e}");
        // Chunked at 500: the same prefill in two steps.
        let e = expected_turn_s(&m, 500, 1000, 0, 1).unwrap();
        assert!((e - 1.0).abs() < 1e-9, "{e}");
        assert_eq!(slowdown(10.0, 4.0), Some(2.5));
        assert_eq!(slowdown(10.0, 0.0), None);
        assert_eq!(slowdown(-1.0, 2.0), Some(0.0));
    }

    #[test]
    fn f49q_readings_order_the_most_starved_first_and_judge_starvation_per_signal() {
        let held = SessionSignalConfig::default();
        assert_eq!(held.signal, SignalKind::HeldTime);
        assert!(held.order_key(Reading::HeldS(400.0)) < held.order_key(Reading::HeldS(10.0)));
        assert!(held.starved(Reading::HeldS(301.0), None));
        assert!(!held.starved(Reading::HeldS(300.0), None));
        let att = SessionSignalConfig {
            signal: SignalKind::SessionAttainment,
            ..Default::default()
        };
        let a = |on_target, judged| Reading::Attainment { on_target, judged };
        // Lowest first; no judged turn counts as 1.
        assert!(att.order_key(a(1, 4)) < att.order_key(a(3, 4)));
        assert!(att.order_key(a(3, 4)) < att.order_key(a(0, 0)));
        // Under 0.7 - 0.3 = 0.4 with 3 judged turns.
        assert!(att.starved(a(1, 3), Some(0.7)));
        assert!(!att.starved(a(1, 2), Some(0.7)));
        assert!(!att.starved(a(2, 4), Some(0.7)));
        // A class without a service level is never starved by attainment.
        assert!(!att.starved(a(0, 10), None));
        let sd = SessionSignalConfig {
            signal: SignalKind::Slowdown,
            ..Default::default()
        };
        assert!(sd.order_key(Reading::Slowdown(Some(8.0))) < sd.order_key(Reading::Slowdown(None)));
        assert!(sd.starved(Reading::Slowdown(Some(5.5)), None));
        assert!(!sd.starved(Reading::Slowdown(Some(5.0)), None));
        assert!(!sd.starved(Reading::Slowdown(None), None));
    }

    #[test]
    fn f49q_the_bucket_allows_per_worker_per_min_and_refills() {
        let mut s = SignalState::default();
        s.set_workers(2);
        // 2 workers x 1 a minute: 2 at once, then one per 30 s (one more by 31 s).
        assert!(s.take(t(0.0), 1.0));
        assert!(s.take(t(0.0), 1.0));
        assert!(!s.take(t(10.0), 1.0));
        assert!(s.take(t(31.0), 1.0));
        assert!(!s.take(t(32.0), 1.0));
        // Never more than a minute's worth.
        assert!(s.take(t(1000.0), 1.0));
        assert!(s.take(t(1000.0), 1.0));
        assert!(!s.take(t(1000.0), 1.0));
        // Under one a minute in all, the bucket still holds one.
        let mut slow = SignalState::default();
        slow.set_workers(1);
        assert!(slow.take(t(0.0), 0.5));
        assert!(!slow.take(t(60.0), 0.5));
        assert!(slow.take(t(121.0), 0.5));
    }

    #[test]
    fn f49q_log_lines_per_override_and_per_minute() {
        let mut s = SignalState::default();
        s.log_in_memory();
        s.set_workers(4);
        s.record(t(10.0), "a", "standard", SignalKind::Slowdown, Some(6.25));
        s.record(t(20.0), "b", "batch", SignalKind::Slowdown, None);
        s.denied(t(30.0));
        assert_eq!(s.minute_overrides().get("standard"), Some(&1));
        s.roll(t(61.0));
        // An empty minute writes nothing.
        s.roll(t(130.0));
        let lines = s.take_lines();
        assert_eq!(
            lines,
            vec![
                r#"{"event":"starvation_override","t":10.000,"session":"a","class":"standard","signal":"slowdown","value":6.250}"#,
                r#"{"event":"starvation_override","t":20.000,"session":"b","class":"batch","signal":"slowdown","value":null}"#,
                r#"{"event":"starvation_minute","t":60.0,"overrides":{"batch":1,"standard":1},"denied":1,"workers":4}"#,
            ]
        );
        // Not logging: counts only.
        let mut quiet = SignalState::default();
        quiet.record(t(1.0), "a", "standard", SignalKind::HeldTime, Some(301.0));
        quiet.roll(t(70.0));
        assert!(quiet.take_lines().is_empty());
    }

    #[test]
    fn f49q_mean_output_and_refresh_period() {
        let mut s = SignalState::default();
        assert_eq!(s.mean_osl("standard"), None);
        s.note_output("standard", 100);
        s.note_output("standard", 201);
        assert_eq!(s.mean_osl("standard"), Some(151));
        assert!(s.refresh_due(t(0.0)));
        assert!(!s.refresh_due(t(9.9)));
        assert!(s.refresh_due(t(10.0)));
    }

    #[test]
    fn f49q_validation() {
        let policy = ServicePolicy::default();
        assert!(SessionSignalConfig::default().validate(&policy).is_ok());
        assert!(
            StarvationOverrideConfig::default()
                .validate(&policy)
                .is_ok()
        );
        let bad_signal = [
            SessionSignalConfig {
                held_s: 0.0,
                ..Default::default()
            },
            SessionSignalConfig {
                attainment_gap: 1.5,
                ..Default::default()
            },
            SessionSignalConfig {
                min_turns: 0,
                ..Default::default()
            },
            SessionSignalConfig {
                threshold: f64::NAN,
                ..Default::default()
            },
            SessionSignalConfig {
                order_classes: vec!["gold".to_owned()],
                ..Default::default()
            },
            SessionSignalConfig {
                order_classes: vec!["batch".to_owned(), "batch".to_owned()],
                ..Default::default()
            },
        ];
        for bad in bad_signal {
            let err = bad.validate(&policy).unwrap_err().to_string();
            assert!(err.starts_with("admission.session_signal: "), "{err}");
        }
        let bad_override = [
            StarvationOverrideConfig {
                per_worker_per_min: 0.0,
                ..Default::default()
            },
            StarvationOverrideConfig {
                classes: Vec::new(),
                ..Default::default()
            },
            StarvationOverrideConfig {
                classes: vec!["gold".to_owned()],
                ..Default::default()
            },
            StarvationOverrideConfig {
                classes: vec!["batch".to_owned(), "batch".to_owned()],
                ..Default::default()
            },
        ];
        for bad in bad_override {
            let err = bad.validate(&policy).unwrap_err().to_string();
            assert!(err.starts_with("admission.starvation_override: "), "{err}");
        }
    }
}
