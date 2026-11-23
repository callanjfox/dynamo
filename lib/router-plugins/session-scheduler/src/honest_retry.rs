// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! f49m `admission.honest_retry`: a Retry-After the session can trust, a cap on re-refusals, and a
//! room check before `returnee_first` refuses a new session. Off (absent) by default; needs
//! `admission.tickets`.
//!
//! (a) **Retry-After** at every refusal (lobby deadline, throttle, `returnee_first` block):
//!     `clamp(N / rate - until_refusal_s, min_s, max_s)`, with `N` the session's place in its
//!     class's lobby (all of the class's waiting sessions for a session not yet in it), plus the
//!     live ticket holders of its class ahead of it that are away (a holder waiting in the lobby
//!     is in the place already), plus one; `rate` the class's admissions out of the lobby a second
//!     over the trailing `window_s` (over the time since the start while the run is younger than
//!     the window); `until_refusal_s` the time left until the refusal (0 for a throttle, the
//!     time to the lobby deadline for a waiting session, which is advised when it enters the
//!     lobby). Fewer than [`MIN_ADMITS`] admissions in the window: the ticket forecast
//!     ([`crate::scheduler::Scheduler::ticket_forecast`], a class listed in `admission.tickets`
//!     once the gate has run), then `admission.retry_after_s`, both clamped the same way. The
//!     source (`rate`, `forecast`, `fallback`) goes to the tickets log.
//! (b) **Re-refusal cap**: refusals are counted per ticket key (the session, or its root with
//!     `tickets.per_root`); at the `max_refusals`-th the session's ticket becomes a
//!     **reservation**: its KV need is held out of the room of its class and the classes below it
//!     (never a higher class's: no inversion) from a pool of `reserve_frac` x HBM, filled in rank
//!     order then oldest reservation first. It goes ahead of the unreserved holders of its class
//!     (lobby fit and `returnee_first` order). It lapses `grace_s` after the latest advice's due
//!     time unless its holder is back and waiting; admission consumes it.
//! (c) **`returnee_first` room check**: a new session is refused for its class's live holders only
//!     when its need does not fit the room now (the gate's latest KV picture, the class's room)
//!     after the tickets' holds of its class and above, the unheld need of the holders of its class
//!     waiting in the lobby (they go first) and the input of the higher classes' waiting sessions.
//!
//! (d) f49m2 **`by_reason`** (default false): the advice follows the refusal's reason
//!     ([`Reason`]): the gate's latest hold of the session (KV only for the KV fit's own hold),
//!     else whether the KV room fits it now. Not KV: the rate, else `retry_after_s`, else the
//!     class's wait limit, never the ticket forecast (KV-only: with room it is the prefill window,
//!     30 s, whatever holds the session). KV: the larger of the rate and the forecast, else (a)'s
//!     chain ([`advise_by_reason`]). The refusal line gains `reason`, `reason_kv`, `last_hold`.
//!
//! With `DYNAMO_TICKETS_LOG`: `honest_refusal` per refusal, `reserve_open` / `reserve_consume` /
//! `reserve_lapse` per reservation event, and `reserve_minute` with the reserved KV that sat on free
//! HBM, in KV x seconds.

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::clock::Time;
use crate::config::{ConfigError, ServicePolicy};

/// Fewer admissions than this in the window: the rate is not trusted.
pub const MIN_ADMITS: usize = 5;

/// `admission.honest_retry` (f49m).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct HonestRetryConfig {
    /// Seconds of admissions the rate is measured over. Default 120.
    pub window_s: f64,
    /// Least Retry-After, seconds. Default 5.
    pub min_s: f64,
    /// Most Retry-After, seconds. Default 600.
    pub max_s: f64,
    /// Refusals after which a session's ticket becomes a reservation. Default 3.
    pub max_refusals: u32,
    /// Most KV the reservations hold together, as a share of the HBM budget. Default 0.2.
    pub reserve_frac: f64,
    /// Seconds after its advised return a reservation whose holder is not back lapses. Default 30.
    pub grace_s: f64,
    /// f49m2: price each refusal by its reason (a non-KV refusal never gets the ticket forecast).
    /// Default false (f49m).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub by_reason: bool,
}

impl Default for HonestRetryConfig {
    fn default() -> Self {
        Self {
            window_s: 120.0,
            min_s: 5.0,
            max_s: 600.0,
            max_refusals: 3,
            reserve_frac: 0.2,
            grace_s: 30.0,
            by_reason: false,
        }
    }
}

impl HonestRetryConfig {
    pub fn validate(&self, policy: &ServicePolicy) -> Result<(), ConfigError> {
        if policy.admission.tickets.is_none() {
            return Err(ConfigError::HonestRetry(
                "needs admission.tickets (a reservation is a ticket), and admission.tickets is absent"
                    .to_owned(),
            ));
        }
        let out = |field, expected, value| {
            Err(ConfigError::PolicyOutOfRange {
                field,
                expected,
                value,
            })
        };
        if !(self.window_s.is_finite() && self.window_s > 0.0) {
            return out("admission.honest_retry.window_s", "positive", self.window_s);
        }
        if !(self.min_s.is_finite() && self.min_s >= 0.0) {
            return out(
                "admission.honest_retry.min_s",
                "finite and non-negative",
                self.min_s,
            );
        }
        if !(self.max_s.is_finite() && self.max_s > 0.0 && self.max_s >= self.min_s) {
            return out(
                "admission.honest_retry.max_s",
                "positive and at least min_s",
                self.max_s,
            );
        }
        if self.max_refusals == 0 {
            return out("admission.honest_retry.max_refusals", "at least 1", 0.0);
        }
        if !(self.reserve_frac.is_finite() && (0.0..=1.0).contains(&self.reserve_frac)) {
            return out(
                "admission.honest_retry.reserve_frac",
                "in [0, 1]",
                self.reserve_frac,
            );
        }
        if !(self.grace_s.is_finite() && self.grace_s >= 0.0) {
            return out(
                "admission.honest_retry.grace_s",
                "finite and non-negative",
                self.grace_s,
            );
        }
        Ok(())
    }

    fn clamp(&self, seconds: f64) -> f64 {
        seconds.max(self.min_s).min(self.max_s)
    }
}

/// Where a Retry-After came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// Place over the measured admission rate.
    Rate,
    /// The ticket forecast (too few admissions measured).
    Forecast,
    /// `admission.retry_after_s` (too few admissions, no forecast).
    Fallback,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rate => "rate",
            Self::Forecast => "forecast",
            Self::Fallback => "fallback",
        }
    }
}

/// One Retry-After and how it was made.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Advice {
    pub seconds: f64,
    pub source: Source,
    /// `N`: place in the lobby plus away holders ahead plus one.
    pub place: u64,
    /// Admissions a second measured (`None` below [`MIN_ADMITS`]).
    pub rate: Option<f64>,
}

/// The Retry-After for a session `place` deep (`N`) whose class had `admits` admissions in the
/// trailing window at `now_s`, refused `until_refusal_s` from now: the rate's, else `forecast()`'s,
/// else `fallback`, clamped to `[min_s, max_s]`. `None` with none of them.
pub fn advise(
    cfg: &HonestRetryConfig,
    place: u64,
    admits: usize,
    now_s: f64,
    until_refusal_s: f64,
    forecast: impl FnOnce() -> Option<f64>,
    fallback: Option<f64>,
) -> Option<Advice> {
    let span = cfg.window_s.min(now_s);
    if admits >= MIN_ADMITS && span > 0.0 {
        let rate = admits as f64 / span;
        let seconds = place as f64 / rate - until_refusal_s.max(0.0);
        return Some(Advice {
            seconds: cfg.clamp(seconds),
            source: Source::Rate,
            place,
            rate: Some(rate),
        });
    }
    let (seconds, source) = match forecast() {
        Some(f) => (f, Source::Forecast),
        None => (fallback?, Source::Fallback),
    };
    Some(Advice {
        seconds: cfg.clamp(seconds),
        source,
        place,
        rate: None,
    })
}

/// f49m2: why a session is (or will be) refused, for its Retry-After.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reason {
    /// The gate step that held it (`kv_budget`, `prefill`, `compute`, `ttft`, `surge`, `harm`,
    /// `onboard`, `infeasible`, `returnee_fifo`, `behind_earlier`, `seq_room`, `no_slack`), or with
    /// no hold noted `kv_fits`, `kv_short` or `unknown`.
    pub name: &'static str,
    /// Whether the KV room is what holds it.
    pub kv: bool,
}

/// f49m2: the reason of a gate hold: `reason` itself for the KV fit's own hold (`KvBudget`, KV)
/// and when no later step held it, else `step` (f49e's `ticket_step`: the step that turned `Fits`
/// into a hold), never KV.
pub fn hold_reason(reason: &crate::admit::HoldReason, step: Option<&'static str>) -> Reason {
    use crate::admit::HoldReason as H;
    let (own, kv) = match reason {
        H::KvBudget { .. } => ("kv_budget", true),
        H::NoSlack { .. } => ("no_slack", true),
        H::BehindEarlier { .. } => ("behind_earlier", false),
        H::NoSeqRoom => ("seq_room", false),
        H::PrefillBacklog { .. } => ("surge", false),
        H::ComputeHeadroom { .. } => ("compute", false),
        H::OnboardBudget { .. } => ("onboard", false),
        H::PrefillBudget { .. } => ("prefill", false),
        H::TargetInfeasible { .. } => ("infeasible", false),
        H::ForecastHarm { .. } => ("harm", false),
    };
    match step {
        Some(s) if !kv && s != "kv_fit" && s != "expire" => Reason { name: s, kv: false },
        _ => Reason { name: own, kv },
    }
}

/// f49m2: [`advise`] by the refusal's `reason`. Not KV: the rate, else `fallback`, else
/// `wait_limit_s`, never `forecast` (it prices the KV room alone). KV: the larger of the rate's
/// and the forecast's (each clamped) when the rate is trusted, else [`advise`]'s chain.
#[allow(clippy::too_many_arguments)]
pub fn advise_by_reason(
    cfg: &HonestRetryConfig,
    reason: Reason,
    place: u64,
    admits: usize,
    now_s: f64,
    until_refusal_s: f64,
    forecast: impl FnOnce() -> Option<f64>,
    fallback: Option<f64>,
    wait_limit_s: f64,
) -> Option<Advice> {
    if !reason.kv {
        return advise(
            cfg,
            place,
            admits,
            now_s,
            until_refusal_s,
            || None,
            fallback.or(Some(wait_limit_s)),
        );
    }
    match advise(cfg, place, admits, now_s, until_refusal_s, || None, None) {
        Some(rate) => Some(match forecast().map(|f| cfg.clamp(f)) {
            Some(f) if f > rate.seconds => Advice {
                seconds: f,
                source: Source::Forecast,
                ..rate
            },
            _ => rate,
        }),
        None => advise(
            cfg,
            place,
            admits,
            now_s,
            until_refusal_s,
            forecast,
            fallback,
        ),
    }
}

/// The option's state: admissions per class, refusals per ticket key, the advice kept for a waiting
/// session until its deadline refuses it, and the reserved-but-unused KV x seconds for the log.
#[derive(Debug, Default)]
pub struct HonestRetry {
    admits: HashMap<String, VecDeque<Time>>,
    refusals: HashMap<String, u32>,
    pending: HashMap<String, Advice>,
    /// `returnee_block` refused the request being throttled (for the refusal's log line).
    pub(crate) blocked: bool,
    /// Reserved KV on free HBM, integrated since the last `reserve_minute` line.
    unused_kv_s: f64,
    /// The latest step's time and reserved-but-unused KV.
    last: Option<(Time, u64)>,
    minute: Option<u64>,
    /// f49m2 `by_reason`: each waiting session's latest gate hold.
    causes: HashMap<String, Reason>,
    /// f49m2 `by_reason`: the reason a waiting session's lobby advice was priced on.
    priced: HashMap<String, Reason>,
}

impl HonestRetry {
    /// An admission of `class` out of the lobby at `now`; keeps `window_s` of them.
    pub fn admitted(&mut self, class: &str, now: Time, window_s: f64) {
        let q = self.admits.entry(class.to_owned()).or_default();
        q.push_back(now);
        while q
            .front()
            .is_some_and(|&t| now.since(t).as_secs_f64() > window_s)
        {
            q.pop_front();
        }
    }

    /// Admissions of `class` within `window_s` of `now`.
    pub fn admits(&self, class: &str, now: Time, window_s: f64) -> usize {
        self.admits.get(class).map_or(0, |q| {
            q.iter()
                .filter(|&&t| now.since(t).as_secs_f64() <= window_s)
                .count()
        })
    }

    /// One more refusal of ticket key `key`: the count.
    pub fn refused(&mut self, key: &str) -> u32 {
        let n = self.refusals.entry(key.to_owned()).or_insert(0);
        *n += 1;
        *n
    }

    pub fn refusals(&self, key: &str) -> u32 {
        self.refusals.get(key).copied().unwrap_or(0)
    }

    /// `key` was admitted on (or without) its ticket, or ended: its count starts over.
    pub fn forget(&mut self, key: &str) {
        self.refusals.remove(key);
        self.pending.remove(key);
        self.causes.remove(key);
        self.priced.remove(key);
    }

    /// f49m2: `session`'s latest gate hold was for `reason`.
    pub fn note_cause(&mut self, session: &str, reason: Reason) {
        self.causes.insert(session.to_owned(), reason);
    }

    /// f49m2: `session` was admitted: its hold is over.
    pub fn clear_cause(&mut self, session: &str) {
        self.causes.remove(session);
    }

    /// f49m2: `session`'s latest gate hold, if one was noted.
    pub fn cause(&self, session: &str) -> Option<Reason> {
        self.causes.get(session).copied()
    }

    /// f49m2: `session`'s lobby advice was priced on `reason`.
    pub fn set_priced(&mut self, session: &str, reason: Reason) {
        self.priced.insert(session.to_owned(), reason);
    }

    pub fn take_priced(&mut self, session: &str) -> Option<Reason> {
        self.priced.remove(session)
    }

    pub fn set_pending(&mut self, session: &str, advice: Advice) {
        self.pending.insert(session.to_owned(), advice);
    }

    pub fn take_pending(&mut self, session: &str) -> Option<Advice> {
        self.pending.remove(session)
    }

    /// A gate check at `now` with `unused` reserved KV on free HBM: integrates the previous
    /// reading over the time since. Returns the `reserve_minute` line's integral when a new
    /// simulated minute starts (and starts it over), else `None`.
    pub fn step(&mut self, now: Time, unused: u64) -> Option<(u64, f64)> {
        if let Some((at, kv)) = self.last
            && now > at
        {
            self.unused_kv_s += kv as f64 * now.since(at).as_secs_f64();
        }
        self.last = Some((now, unused));
        let minute = (now.as_secs_f64() / 60.0) as u64;
        if self.minute == Some(minute) {
            return None;
        }
        let first = self.minute.is_none();
        self.minute = Some(minute);
        let kv_s = std::mem::take(&mut self.unused_kv_s);
        (!first).then_some((minute * 60, kv_s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: f64) -> Time {
        Time::from_secs_f64(s)
    }

    #[test]
    fn the_rate_advice_is_place_over_the_measured_rate_less_the_wait_left() {
        let cfg = HonestRetryConfig::default();
        // 12 admissions in the last 120 s: 0.1 a second; N = 3: 30 s.
        let a = advise(&cfg, 3, 12, 500.0, 0.0, || Some(99.0), Some(30.0)).unwrap();
        assert_eq!(
            (a.seconds, a.source, a.rate),
            (30.0, Source::Rate, Some(0.1))
        );
        // Advised on entering the lobby, 20 s before its deadline: 10 s after it.
        let a = advise(&cfg, 3, 12, 500.0, 20.0, || None, None).unwrap();
        assert_eq!(a.seconds, 10.0);
        // Clamped to [min_s, max_s].
        assert_eq!(
            advise(&cfg, 3, 12, 500.0, 60.0, || None, None)
                .unwrap()
                .seconds,
            5.0
        );
        assert_eq!(
            advise(&cfg, 300, 12, 500.0, 0.0, || None, None)
                .unwrap()
                .seconds,
            600.0
        );
        // A run younger than the window: over the time since its start (5 in 50 s).
        let a = advise(&cfg, 1, 5, 50.0, 0.0, || None, None).unwrap();
        assert_eq!((a.seconds, a.rate), (10.0, Some(0.1)));
    }

    #[test]
    fn too_few_admissions_fall_back_to_the_forecast_then_retry_after_s() {
        let cfg = HonestRetryConfig::default();
        let a = advise(
            &cfg,
            2,
            MIN_ADMITS - 1,
            500.0,
            0.0,
            || Some(42.0),
            Some(30.0),
        )
        .unwrap();
        assert_eq!(
            (a.seconds, a.source, a.rate),
            (42.0, Source::Forecast, None)
        );
        let a = advise(&cfg, 2, 0, 500.0, 0.0, || None, Some(30.0)).unwrap();
        assert_eq!((a.seconds, a.source), (30.0, Source::Fallback));
        // Each clamped too.
        let a = advise(&cfg, 2, 0, 500.0, 0.0, || Some(1.0), None).unwrap();
        assert_eq!(a.seconds, 5.0);
        assert_eq!(advise(&cfg, 2, 0, 500.0, 0.0, || None, None), None);
        assert_eq!(
            [Source::Rate, Source::Forecast, Source::Fallback].map(Source::as_str),
            ["rate", "forecast", "fallback"]
        );
    }

    #[test]
    fn f49m2_advice_by_reason() {
        let cfg = HonestRetryConfig {
            by_reason: true,
            ..HonestRetryConfig::default()
        };
        let not_kv = Reason {
            name: "prefill",
            kv: false,
        };
        let kv = Reason {
            name: "kv_budget",
            kv: true,
        };
        // Not KV: the rate; the forecast is never asked.
        let a = advise_by_reason(
            &cfg,
            not_kv,
            3,
            12,
            500.0,
            0.0,
            || panic!(),
            Some(30.0),
            60.0,
        )
        .unwrap();
        assert_eq!((a.seconds, a.source), (30.0, Source::Rate));
        // Not KV, too few admissions: retry_after_s, else the wait limit; never the forecast.
        let a = advise_by_reason(
            &cfg,
            not_kv,
            3,
            2,
            500.0,
            0.0,
            || panic!(),
            Some(45.0),
            60.0,
        )
        .unwrap();
        assert_eq!((a.seconds, a.source), (45.0, Source::Fallback));
        let a = advise_by_reason(&cfg, not_kv, 3, 2, 500.0, 0.0, || panic!(), None, 60.0).unwrap();
        assert_eq!((a.seconds, a.source), (60.0, Source::Fallback));
        // KV: the larger of the rate (30 s) and the forecast, each clamped.
        let a = advise_by_reason(&cfg, kv, 3, 12, 500.0, 0.0, || Some(90.0), None, 60.0).unwrap();
        assert_eq!(
            (a.seconds, a.source, a.rate),
            (90.0, Source::Forecast, Some(0.1))
        );
        let a = advise_by_reason(&cfg, kv, 3, 12, 500.0, 0.0, || Some(10.0), None, 60.0).unwrap();
        assert_eq!((a.seconds, a.source), (30.0, Source::Rate));
        let a = advise_by_reason(&cfg, kv, 3, 12, 500.0, 0.0, || Some(9e9), None, 60.0).unwrap();
        assert_eq!(a.seconds, 600.0);
        // KV, too few admissions: (a)'s chain.
        let a =
            advise_by_reason(&cfg, kv, 3, 2, 500.0, 0.0, || Some(42.0), Some(30.0), 60.0).unwrap();
        assert_eq!((a.seconds, a.source), (42.0, Source::Forecast));
        assert_eq!(
            advise_by_reason(&cfg, kv, 3, 2, 500.0, 0.0, || None, None, 60.0),
            None
        );
        // A gate hold's reason: the step after the KV fit, else the KV fit's own.
        use crate::admit::HoldReason as H;
        let budget = H::KvBudget {
            needed: 1,
            hot_room: 0,
            total_room: 0,
        };
        assert_eq!(hold_reason(&budget, Some("kv_fit")), kv);
        let prefill = H::PrefillBudget {
            used: 1.0,
            own: 1.0,
            budget: 1.0,
        };
        assert_eq!(hold_reason(&prefill, Some("prefill")), not_kv);
        let ahead = H::BehindEarlier { ahead: "a".into() };
        assert_eq!(
            hold_reason(&ahead, Some("returnee_fifo")),
            Reason {
                name: "returnee_fifo",
                kv: false
            }
        );
        assert_eq!(hold_reason(&ahead, Some("kv_fit")).name, "behind_earlier");
        assert_eq!(hold_reason(&H::NoSeqRoom, None).name, "seq_room");
        // State: noted, forgotten on admission and end.
        let mut h = HonestRetry::default();
        h.note_cause("s", not_kv);
        h.set_priced("s", kv);
        assert_eq!((h.cause("s"), h.take_priced("s")), (Some(not_kv), Some(kv)));
        h.clear_cause("s");
        assert_eq!(h.cause("s"), None);
        h.note_cause("s", kv);
        h.forget("s");
        assert_eq!(h.cause("s"), None);
        // Off by default, not serialized then.
        let d = HonestRetryConfig::default();
        assert!(!d.by_reason);
        assert!(!serde_json::to_string(&d).unwrap().contains("by_reason"));
        let c: HonestRetryConfig = serde_json::from_str(r#"{"by_reason": true}"#).unwrap();
        assert!(c.by_reason);
    }

    #[test]
    fn admissions_refusals_and_the_unused_integral() {
        let mut h = HonestRetry::default();
        for s in [0.0, 10.0, 100.0, 130.0] {
            h.admitted("standard", t(s), 120.0);
        }
        assert_eq!(h.admits("standard", t(130.0), 120.0), 3);
        assert_eq!(h.admits("standard", t(230.0), 120.0), 1);
        assert_eq!(h.admits("priority", t(130.0), 120.0), 0);
        assert_eq!((h.refused("a"), h.refused("a"), h.refusals("a")), (1, 2, 2));
        h.forget("a");
        assert_eq!(h.refusals("a"), 0);
        // 100 KV from 10 s to 70 s, then 0: 6000 KV s in the first minute line.
        assert_eq!(h.step(t(10.0), 100), None);
        assert_eq!(h.step(t(50.0), 100), None);
        assert_eq!(h.step(t(70.0), 0), Some((60, 6000.0)));
        assert_eq!(h.step(t(125.0), 0), Some((120, 0.0)));
    }

    #[test]
    fn the_config_parses_with_defaults_and_validates() {
        let c: HonestRetryConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(c, HonestRetryConfig::default());
        assert_eq!(
            (
                c.window_s,
                c.min_s,
                c.max_s,
                c.max_refusals,
                c.reserve_frac,
                c.grace_s
            ),
            (120.0, 5.0, 600.0, 3, 0.2, 30.0)
        );
        assert!(serde_json::from_str::<HonestRetryConfig>(r#"{"windows": 1}"#).is_err());
        let policy: ServicePolicy = serde_json::from_str(
            r#"{"admission": {"tickets": {}, "honest_retry": {"max_refusals": 2}}}"#,
        )
        .unwrap();
        assert_eq!(
            policy.admission.honest_retry.as_ref().unwrap().max_refusals,
            2
        );
        policy.validate().unwrap();
        let off: ServicePolicy = serde_json::from_str("{}").unwrap();
        assert!(off.admission.honest_retry.is_none());
        let json = serde_json::to_string(&off).unwrap();
        assert!(!json.contains("honest_retry"), "absent: not serialized");
        let mut bad = policy.clone();
        bad.admission.tickets = None;
        assert_eq!(
            bad.validate().unwrap_err().to_string(),
            "admission.honest_retry: needs admission.tickets (a reservation is a ticket), and admission.tickets is absent"
        );
        let field = |f: fn(&mut HonestRetryConfig)| {
            let mut p = policy.clone();
            f(p.admission.honest_retry.as_mut().unwrap());
            match p.validate() {
                Err(ConfigError::PolicyOutOfRange { field, .. }) => field,
                other => panic!("{other:?}"),
            }
        };
        assert_eq!(
            field(|c| c.window_s = 0.0),
            "admission.honest_retry.window_s"
        );
        assert_eq!(field(|c| c.min_s = -1.0), "admission.honest_retry.min_s");
        assert_eq!(
            field(|c| c.max_s = f64::NAN),
            "admission.honest_retry.max_s"
        );
        assert_eq!(
            field(|c| {
                c.min_s = 10.0;
                c.max_s = 5.0;
            }),
            "admission.honest_retry.max_s"
        );
        assert_eq!(
            field(|c| c.max_refusals = 0),
            "admission.honest_retry.max_refusals"
        );
        assert_eq!(
            field(|c| c.reserve_frac = 1.5),
            "admission.honest_retry.reserve_frac"
        );
        assert_eq!(
            field(|c| c.grace_s = f64::INFINITY),
            "admission.honest_retry.grace_s"
        );
    }
}
