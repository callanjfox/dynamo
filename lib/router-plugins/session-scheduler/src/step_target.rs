// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! f49j: the per-worker step-target controller (`admission.step_target`), off (absent) by default.
//!
//! `{target_frac: 0.85, class: priority, window_s: 45, interval_s: 10, kp: 0.5, ki: 0.05,
//! min: 0.7, max: 1.8, slew: 0.05, min_samples: 3, sample_running: true}`, and the optional
//! `decode_cap: {base: auto | N}` (f49j2). The hot admission headroom of the classes ranked below
//! `class` (standard and batch by default) follows each worker's measured decode speed instead of
//! the fixed `hot_overcommit`.
//!
//! Measure (not forecast): per worker, the mean ITL of the decode tokens produced over the trailing
//! `window_s`, every class together (decode time in the window over decode tokens in it, so a
//! token-weighted mean). Each request's tokens are spread evenly over its decode and the part
//! inside the window counts.
//! - A completed request: `n - 1` tokens from its first token (`Responding`) to its completion.
//! - f49j2, `sample_running` (default true): a request still decoding, from its first token to
//!   now. Its tokens so far are the host's own count when the host raises the request's progress
//!   counter (`ClassifyRequest::progress`; the live router does, per output token). Offline replay
//!   never raises it and sends no per-request token counts (its `EngineGauge` has running requests
//!   and KV blocks per rank, no tokens), so there the tokens are estimated: decode time over the
//!   worker's reference ITL (the completed requests' mean in the window, else the worker's target),
//!   at most the p90 of its class's recent completed decode lengths. While under that bound the
//!   request adds weight at the reference ITL and moves nothing; past it (a decode running longer
//!   than nine in ten of its class's at the reference speed) it counts as a sample at
//!   `time / p90`, a lower bound on its true ITL that rises the longer it runs. Without class
//!   lengths yet it is neutral.
//!
//! A worker with fewer than `min_samples` counted requests (completed, plus running ones with a
//! host count or past the bound) in the window holds its multiplier.
//!
//! Target (f49j2): per worker, `target_frac x` the tightest `itl_s` among the classes decoding on
//! it now (first token seen, not finished); with none of them decoding, `class`'s `itl_s`.
//!
//! Control, every `interval_s`, per worker with its own state (`u` 0 and `m` 1 at the start):
//! - `e = (target - measured) / target` (positive: slack);
//! - `u = clamp(u + ki e interval_s, ln min, ln max)`, not integrated while the output
//!   `kp e + u` is already past `ln max` with `e > 0` or past `ln min` with `e < 0` (anti-windup);
//! - `m_raw = clamp(exp(kp e + u), min, max)`; `m = clamp(m_raw, m_prev (1 - slew), m_prev (1 + slew))`.
//!
//! Act, KV: the gate's KV fit is fleet-aggregated (one hot budget, `gate_share x hot_overcommit x`
//! the fleet's HBM), so a class below `class` gets the HBM-weighted mean of the workers' `m` as a
//! further multiple of `hot_overcommit` ([`crate::scheduler::Scheduler::step_target_scale`]): the
//! lobby decision, the f47 fit at the wait limit, and the tickets' room. `class` and the classes
//! above it keep their fit and room; the total budget never changes.
//!
//! Act, decode cap (f49j2, `decode_cap`, absent by default): the KV multiple does nothing when KV is
//! not what binds (large HBM, compute-bound decode). With `decode_cap` each worker also has a cap
//! on its active requests (sent, not finished) of the classes below `class`: `max(1, round(m x
//! base))`. `base: N` is fixed and always applies. `base: auto` is the worker's own p90 of that
//! count over the last 10 minutes, sampled once per interval while its `m` is at least 1, and the
//! cap applies only while `m < 1`: a cap at the p90 would otherwise hold the top tenth of normal
//! load, and samples taken under the cap would teach it its own output and ratchet it down.
//! At the gate ([`StepTarget::cap_verdict`], enforce mode) a turn of a capped class whose worker
//! (the gate's target, else its session's last worker) is at its cap goes to the worker under its
//! cap with the most room (`Go` with a target); with no such worker it holds, rechecked at each
//! completion and at the latest after `interval_s`. A turn with no known worker is untouched while
//! no worker is at its cap. `class` and the classes above it are never capped.
//!
//! `DYNAMO_STEPTGT_LOG=<path>`: one JSON line per worker per interval `{t, worker, measured_itl_s,
//! samples, running, target_s, e, u, m}` (`measured_itl_s` and `e` null while held; `running` the
//! running requests among the samples), with `active` and `cap` added when `decode_cap` is set.
//! Nothing is written when unset.

use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::Write;
use std::time::Duration;

use dynamo_kv_router::plugins::prelude::*;
use dynamo_kv_router::plugins::request_classifier::RequestProgress;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::clock::Time;
use crate::config::{ConfigError, ServicePolicy};

/// The env var naming the step-target diagnostics log.
pub const STEPTGT_LOG_ENV: &str = "DYNAMO_STEPTGT_LOG";

/// Most requests tracked; a table is cleared when it fills.
const MAX_TRACKED: usize = 65_536;

/// Most samples kept per worker (the window trims them first).
const MAX_SAMPLES: usize = 16_384;

/// A request still decoding this long after it was sent is forgotten, seconds.
const TRACK_S: f64 = 3_600.0;

/// A worker neither listed nor heard from for this long, with no samples, is forgotten, seconds.
const STALE_S: f64 = 600.0;

/// Completed decode lengths kept per class (their p90 bounds a running request's estimate).
const LENGTHS_KEPT: usize = 256;

/// The quantile of the class's decode lengths that bounds a running request's estimated tokens.
const LENGTH_QUANTILE: f64 = 0.9;

/// `decode_cap.base: auto`: the trailing window of the active-count samples, seconds.
const AUTO_WINDOW_S: f64 = 600.0;

/// `decode_cap.base: auto`: the quantile of the active-count samples.
const AUTO_QUANTILE: f64 = 0.9;

/// A cap reservation (a turn let go to a worker, not sent yet) lapses after this long, seconds.
const RESERVE_S: f64 = 60.0;

/// `admission.step_target` (f49j). Off (absent) by default.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct StepTargetConfig {
    /// The target ITL as a share of the tightest decoding class's `itl_s` (of `class`'s with none
    /// decoding). Default 0.85.
    pub target_frac: f64,
    /// The class whose ITL limit sets the target on an idle worker; the classes ranked below it
    /// are scaled. Default `priority`.
    pub class: String,
    /// Seconds of decode the measurement reads. Default 45.
    pub window_s: f64,
    /// Seconds between controller steps. Default 10.
    pub interval_s: f64,
    /// Proportional gain on the relative error. Default 0.5.
    pub kp: f64,
    /// Integral gain per second. Default 0.05.
    pub ki: f64,
    /// Lowest multiple of `hot_overcommit`. Default 0.7.
    pub min: f64,
    /// Highest multiple of `hot_overcommit`. Default 1.8.
    pub max: f64,
    /// Largest relative change of the multiple per step. Default 0.05.
    pub slew: f64,
    /// Fewest counted requests in the window for a step; with fewer the worker's multiple holds.
    /// Default 3.
    pub min_samples: u32,
    /// f49j2: the measure also reads the requests still decoding. Default true.
    pub sample_running: bool,
    /// f49j2: a per-worker cap on the active requests of the scaled classes. Off (absent) by
    /// default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_cap: Option<DecodeCapConfig>,
}

impl Default for StepTargetConfig {
    fn default() -> Self {
        Self {
            target_frac: 0.85,
            class: "priority".to_owned(),
            window_s: 45.0,
            interval_s: 10.0,
            kp: 0.5,
            ki: 0.05,
            min: 0.7,
            max: 1.8,
            slew: 0.05,
            min_samples: 3,
            sample_running: true,
            decode_cap: None,
        }
    }
}

/// `step_target.decode_cap` (f49j2).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DecodeCapConfig {
    /// The active requests of the scaled classes a worker takes at `m` 1. Default `auto`.
    pub base: CapBase,
}

/// `decode_cap.base`: `auto` (the worker's own recent p90) or a fixed count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CapBase {
    #[default]
    Auto,
    Fixed(u32),
}

impl Serialize for CapBase {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            CapBase::Auto => serializer.serialize_str("auto"),
            CapBase::Fixed(n) => serializer.serialize_u32(*n),
        }
    }
}

impl<'de> Deserialize<'de> for CapBase {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl serde::de::Visitor<'_> for Visitor {
            type Value = CapBase;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("\"auto\" or a non-negative integer")
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<CapBase, E> {
                if v == "auto" {
                    Ok(CapBase::Auto)
                } else {
                    Err(E::invalid_value(serde::de::Unexpected::Str(v), &self))
                }
            }

            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<CapBase, E> {
                u32::try_from(v)
                    .map(CapBase::Fixed)
                    .map_err(|_| E::invalid_value(serde::de::Unexpected::Unsigned(v), &self))
            }

            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<CapBase, E> {
                match u64::try_from(v) {
                    Ok(v) => self.visit_u64(v),
                    Err(_) => Err(E::invalid_value(serde::de::Unexpected::Signed(v), &self)),
                }
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

impl StepTargetConfig {
    pub fn validate(&self, policy: &ServicePolicy) -> Result<(), ConfigError> {
        let fail = |reason: String| Err(ConfigError::StepTarget(reason));
        let Some(class) = policy.classes.iter().find(|c| c.name == self.class) else {
            return fail(format!("class {:?} is not one of the classes", self.class));
        };
        if class.service_level.is_none() {
            return fail(format!(
                "class {:?} has no service_level (its itl_s)",
                self.class
            ));
        }
        let positive = |x: f64| x.is_finite() && x > 0.0;
        let gain = |x: f64| x.is_finite() && x >= 0.0;
        for (field, expected, value, ok) in [
            (
                "target_frac",
                "in (0, 1]",
                self.target_frac,
                positive(self.target_frac) && self.target_frac <= 1.0,
            ),
            (
                "window_s",
                "positive",
                self.window_s,
                positive(self.window_s),
            ),
            (
                "interval_s",
                "positive",
                self.interval_s,
                positive(self.interval_s),
            ),
            ("kp", "finite and non-negative", self.kp, gain(self.kp)),
            ("ki", "finite and non-negative", self.ki, gain(self.ki)),
            (
                "min",
                "in (0, 1]",
                self.min,
                positive(self.min) && self.min <= 1.0,
            ),
            (
                "max",
                "finite and at least 1",
                self.max,
                self.max.is_finite() && self.max >= 1.0,
            ),
            (
                "slew",
                "in (0, 1)",
                self.slew,
                positive(self.slew) && self.slew < 1.0,
            ),
            (
                "min_samples",
                "at least 1",
                f64::from(self.min_samples),
                self.min_samples >= 1,
            ),
        ] {
            if !ok {
                return fail(format!("{field} must be {expected}, got {value}"));
            }
        }
        if let Some(cap) = &self.decode_cap {
            if cap.base == CapBase::Fixed(0) {
                return fail("decode_cap.base must be auto or at least 1, got 0".to_owned());
            }
            if !policy.classes.iter().any(|c| c.rank > class.rank) {
                return fail(format!(
                    "decode_cap caps the classes ranked below {:?}; there are none",
                    self.class
                ));
            }
        }
        Ok(())
    }
}

/// One controller step from `(u, m_prev)` on a measured mean ITL: `(e, u, m)`.
pub fn control(
    cfg: &StepTargetConfig,
    target_s: f64,
    u: f64,
    m_prev: f64,
    measured_s: f64,
) -> (f64, f64, f64) {
    let e = (target_s - measured_s) / target_s;
    let (lo, hi) = (cfg.min.ln(), cfg.max.ln());
    let out = cfg.kp * e + u;
    // Anti-windup: no further integration while the output is saturated in e's direction.
    let saturated = (e > 0.0 && out >= hi) || (e < 0.0 && out <= lo);
    let u = if saturated {
        u
    } else {
        (u + cfg.ki * e * cfg.interval_s).clamp(lo, hi)
    };
    let raw = (cfg.kp * e + u).exp().clamp(cfg.min, cfg.max);
    let m = raw
        .clamp(m_prev * (1.0 - cfg.slew), m_prev * (1.0 + cfg.slew))
        .clamp(cfg.min, cfg.max);
    (e, u, m)
}

/// A worker's decode in the window: the token-weighted mean ITL, the requests that count, and
/// how many of those are still running.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reading {
    pub measured_itl_s: Option<f64>,
    pub samples: u32,
    pub running: u32,
}

/// The gate's answer for a turn under `decode_cap`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapVerdict {
    /// No change to the gate's decision.
    Pass,
    /// Go to this worker (under its cap) instead.
    Go(WorkerWithDpRank),
    /// Every worker is at its cap: hold, at the latest until then.
    Hold(Time),
}

#[derive(Clone, Copy, Debug)]
struct Sample {
    first: Time,
    done: Time,
    tokens: u64,
}

#[derive(Debug)]
struct WorkerCtl {
    hbm_tokens: u64,
    seen: Time,
    u: f64,
    m: f64,
    samples: VecDeque<Sample>,
    /// `decode_cap.base: auto`: (when, active requests of the scaled classes), while `m >= 1`.
    active_hist: VecDeque<(Time, u32)>,
    /// `decode_cap.base: auto`: the p90 of `active_hist` (the last one while it is empty).
    base_auto: Option<u32>,
}

impl WorkerCtl {
    fn new(now: Time) -> Self {
        Self {
            hbm_tokens: 0,
            seen: now,
            u: 0.0,
            m: 1.0,
            samples: VecDeque::new(),
            active_hist: VecDeque::new(),
            base_auto: None,
        }
    }
}

/// A request sent and not finished.
#[derive(Debug)]
struct Live {
    worker: WorkerWithDpRank,
    class: String,
    /// When it was sent (or first heard of).
    since: Time,
    /// Its first token.
    first: Option<Time>,
    /// The host's context counter for it, when the host gave one at classify.
    progress: Option<RequestProgress>,
    /// The counter at the first token.
    base: Option<usize>,
}

impl Live {
    /// Decode tokens since the first token by the host's count, when the host has raised it.
    fn counted(&self) -> Option<f64> {
        let (progress, base) = (self.progress.as_ref()?, self.base?);
        let tokens = progress.context_tokens().saturating_sub(base);
        (tokens > 0).then_some(tokens as f64)
    }
}

/// Time and tokens in the window, the requests counted and the running ones among them.
#[derive(Clone, Copy, Debug, Default)]
struct Sums {
    time: f64,
    tokens: f64,
    n: u32,
    running: u32,
}

fn window_start(now: Time, window_s: f64) -> Time {
    Time::from_secs_f64((now.as_secs_f64() - window_s).max(0.0))
}

fn sum_completed(samples: &VecDeque<Sample>, start: Time) -> Sums {
    let mut sums = Sums::default();
    for s in samples {
        if s.done <= start || s.tokens == 0 {
            continue;
        }
        let span = s.done.since(s.first).as_secs_f64();
        if span <= 0.0 {
            continue;
        }
        let inside = s.done.since(s.first.max(start)).as_secs_f64();
        sums.time += inside;
        sums.tokens += s.tokens as f64 * inside / span;
        sums.n += 1;
    }
    sums
}

/// The nearest-rank `q` quantile.
fn quantile(mut values: Vec<u64>, q: f64) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    let rank = ((q * values.len() as f64).ceil() as usize).clamp(1, values.len());
    Some(values[rank - 1])
}

/// Empty a full table before it takes a new key.
fn bound<V>(map: &mut HashMap<String, V>, key: &str) {
    if map.len() >= MAX_TRACKED && !map.contains_key(key) {
        map.clear();
    }
}

/// The controller's state: per-worker samples and multiples, and the requests in flight.
#[derive(Debug)]
pub struct StepTarget {
    cfg: StepTargetConfig,
    /// `target_frac x class`'s `itl_s`: the target of a worker with no class decoding.
    target_s: f64,
    /// The classes ranked below `cfg.class`: the ones scaled and capped.
    below: Vec<String>,
    /// Each class's `itl_s`, for the classes with a service level.
    levels: HashMap<String, f64>,
    /// The class of a request the scheduler does not know.
    default_class: String,
    workers: BTreeMap<WorkerWithDpRank, WorkerCtl>,
    /// Requests sent and not finished.
    live: HashMap<String, Live>,
    /// The host's context counters of classified requests not sent yet, and when given.
    progress: HashMap<String, (RequestProgress, Time)>,
    /// Recent completed decode lengths per class.
    lengths: HashMap<String, VecDeque<u64>>,
    /// The worker each session's latest request was sent to.
    home: HashMap<String, WorkerWithDpRank>,
    /// `decode_cap`: turns let go to a worker and not sent yet.
    reserved: HashMap<String, (WorkerWithDpRank, Time)>,
    /// `decode_cap`: turns moved to another worker, and turns held.
    cap_moved: u64,
    cap_held: u64,
    next_step: Option<Time>,
    log: Option<std::io::BufWriter<std::fs::File>>,
    /// f49l3 `sla_budget.mode: rebalance`: each class's own `target_frac` (`None`: the one
    /// `target_frac` for every class).
    class_fracs: Option<HashMap<String, f64>>,
}

impl StepTarget {
    /// The controller for `policy`, or `None` when `admission.step_target` is absent (the default).
    pub fn from_policy(policy: &ServicePolicy) -> Option<Self> {
        let cfg = policy.admission.step_target.clone()?;
        let class = policy.classes.iter().find(|c| c.name == cfg.class)?;
        let level = class.service_level.as_ref()?;
        let below = policy
            .classes
            .iter()
            .filter(|c| c.rank > class.rank)
            .map(|c| c.name.clone())
            .collect();
        let levels = policy
            .classes
            .iter()
            .filter_map(|c| c.service_level.as_ref().map(|l| (c.name.clone(), l.itl_s)))
            .collect();
        let log = std::env::var_os(STEPTGT_LOG_ENV)
            .and_then(|path| std::fs::File::create(path).ok())
            .map(std::io::BufWriter::new);
        Some(Self {
            target_s: cfg.target_frac * level.itl_s,
            below,
            levels,
            default_class: policy.class(None).name.clone(),
            cfg,
            workers: BTreeMap::new(),
            live: HashMap::new(),
            progress: HashMap::new(),
            lengths: HashMap::new(),
            home: HashMap::new(),
            reserved: HashMap::new(),
            cap_moved: 0,
            cap_held: 0,
            next_step: None,
            log,
            class_fracs: None,
        })
    }

    /// The target mean ITL of a worker with no class decoding, seconds.
    pub fn target_s(&self) -> f64 {
        self.target_s
    }

    /// The target as a share of the ITL limit (`target_frac`; f49l's `admission.sla_budget` may
    /// move it from the configured value).
    pub fn target_frac(&self) -> f64 {
        self.cfg.target_frac
    }

    /// f49l `admission.sla_budget`: set `target_frac`; the targets follow at once (the next step
    /// reads them). The policy's `step_target` config is not edited.
    pub fn set_target_frac(&mut self, frac: f64) {
        self.cfg.target_frac = frac;
        if let Some(itl_s) = self.levels.get(&self.cfg.class) {
            self.target_s = frac * itl_s;
        }
    }

    /// f49l3 `sla_budget.mode: rebalance`: each class's own `target_frac` (`setpoint[c]`). A
    /// worker's target is then the tightest `setpoint[c] x itl_s` among the classes decoding on
    /// it, an idle worker's `class`'s (`target_frac` follows `class`'s). Without a call the one
    /// `target_frac` holds for every class, as before.
    pub fn set_class_fracs(&mut self, fracs: &BTreeMap<String, f64>) {
        if let (Some(f), Some(itl_s)) =
            (fracs.get(&self.cfg.class), self.levels.get(&self.cfg.class))
        {
            self.cfg.target_frac = *f;
            self.target_s = f * itl_s;
        }
        self.class_fracs = Some(fracs.iter().map(|(c, f)| (c.clone(), *f)).collect());
    }

    /// f49l3: `class`'s `target_frac` (the one `target_frac` without per-class setpoints).
    pub fn class_frac(&self, class: &str) -> f64 {
        self.class_fracs
            .as_ref()
            .and_then(|m| m.get(class).copied())
            .unwrap_or(self.cfg.target_frac)
    }

    /// `worker`'s target mean ITL: `target_frac x` the tightest `itl_s` among the classes decoding
    /// on it now, else [`Self::target_s`].
    pub fn worker_target_s(&self, worker: WorkerWithDpRank) -> f64 {
        // f49l3: per-class setpoints, the tightest `setpoint[c] x itl_s` decoding there.
        if self.class_fracs.is_some() {
            let tightest = self
                .live
                .values()
                .filter(|l| l.worker == worker && l.first.is_some())
                .filter_map(|l| {
                    self.levels
                        .get(&l.class)
                        .map(|itl_s| self.class_frac(&l.class) * itl_s)
                })
                .fold(f64::INFINITY, f64::min);
            return if tightest.is_finite() {
                tightest
            } else {
                self.target_s
            };
        }
        let tightest = self
            .live
            .values()
            .filter(|l| l.worker == worker && l.first.is_some())
            .filter_map(|l| self.levels.get(&l.class))
            .fold(f64::INFINITY, |a, &b| a.min(b));
        if tightest.is_finite() {
            self.cfg.target_frac * tightest
        } else {
            self.target_s
        }
    }

    /// Whether `class` is scaled and capped (ranked below the target class).
    pub fn applies_to(&self, class: &str) -> bool {
        self.below.iter().any(|c| c == class)
    }

    /// The multiple of `hot_overcommit` for `class` in the fleet-aggregated KV fit: the
    /// hot-budget-weighted mean of the workers' multiples for a scaled class, else 1.
    pub fn scale(&self, class: &str) -> f64 {
        if self.applies_to(class) {
            self.fleet_m()
        } else {
            1.0
        }
    }

    /// The multiple of `hot_overcommit` for `class` on `worker`'s own hot room.
    pub fn worker_scale(&self, worker: WorkerWithDpRank, class: &str) -> f64 {
        if self.applies_to(class) {
            self.m(worker)
        } else {
            1.0
        }
    }

    /// `worker`'s multiple (1 until it is stepped).
    pub fn m(&self, worker: WorkerWithDpRank) -> f64 {
        self.workers.get(&worker).map_or(1.0, |w| w.m)
    }

    /// `worker`'s integral term (0 until it is stepped).
    pub fn u(&self, worker: WorkerWithDpRank) -> f64 {
        self.workers.get(&worker).map_or(0.0, |w| w.u)
    }

    /// The workers' multiples weighted by their HBM (their hot budgets); equal weights while no
    /// HBM size is known; 1 with no worker known.
    pub fn fleet_m(&self) -> f64 {
        if self.workers.is_empty() {
            return 1.0;
        }
        let weight: f64 = self.workers.values().map(|w| w.hbm_tokens as f64).sum();
        if weight > 0.0 {
            self.workers
                .values()
                .map(|w| w.hbm_tokens as f64 * w.m)
                .sum::<f64>()
                / weight
        } else {
            self.workers.values().map(|w| w.m).sum::<f64>() / self.workers.len() as f64
        }
    }

    /// The p90 of `class`'s recent completed decode lengths, tokens.
    pub fn class_length(&self, class: &str) -> Option<u64> {
        let lengths = self.lengths.get(class)?;
        quantile(lengths.iter().copied().collect(), LENGTH_QUANTILE)
    }

    /// `worker`'s decode over the window ending `now`.
    pub fn reading(&self, worker: WorkerWithDpRank, now: Time) -> Reading {
        let sums = self.sums(worker, now);
        Reading {
            measured_itl_s: (sums.tokens > 0.0).then(|| sums.time / sums.tokens),
            samples: sums.n,
            running: sums.running,
        }
    }

    fn sums(&self, worker: WorkerWithDpRank, now: Time) -> Sums {
        let Some(w) = self.workers.get(&worker) else {
            return Sums::default();
        };
        let start = window_start(now, self.cfg.window_s);
        let mut sums = sum_completed(&w.samples, start);
        if !self.cfg.sample_running {
            return sums;
        }
        // The reference ITL of a running request with no host count: the completed requests'.
        let reference = if sums.tokens > 0.0 {
            sums.time / sums.tokens
        } else {
            self.worker_target_s(worker)
        };
        for live in self.live.values().filter(|l| l.worker == worker) {
            let Some(first) = live.first else {
                continue;
            };
            let span = now.since(first).as_secs_f64();
            let inside = now.since(first.max(start)).as_secs_f64();
            if span <= 0.0 || inside <= 0.0 {
                continue;
            }
            let (tokens, counted) = match live.counted() {
                Some(tokens) => (tokens, true),
                None => {
                    let expected = span / reference;
                    match self.class_length(&live.class) {
                        Some(bound) if expected > bound as f64 => (bound as f64, true),
                        _ => (expected, false),
                    }
                }
            };
            sums.time += inside;
            sums.tokens += tokens * inside / span;
            if counted {
                sums.n += 1;
                sums.running += 1;
            }
        }
        sums
    }

    /// The workers the host lists now with their HBM in tokens (their hot-budget weights).
    pub fn observe_workers(&mut self, listed: &[(WorkerWithDpRank, u64)], now: Time) {
        for &(worker, hbm_tokens) in listed {
            let w = self
                .workers
                .entry(worker)
                .or_insert_with(|| WorkerCtl::new(now));
            w.seen = now;
            if hbm_tokens > 0 {
                w.hbm_tokens = hbm_tokens;
            }
        }
    }

    fn touch(&mut self, worker: WorkerWithDpRank, now: Time) -> &mut WorkerCtl {
        let w = self
            .workers
            .entry(worker)
            .or_insert_with(|| WorkerCtl::new(now));
        w.seen = w.seen.max(now);
        w
    }

    /// A request on `worker` decoded `tokens` tokens from `first` to `done`.
    pub fn record(&mut self, worker: WorkerWithDpRank, first: Time, done: Time, tokens: u64) {
        let w = self.touch(worker, done);
        w.samples.push_back(Sample {
            first,
            done,
            tokens,
        });
        if w.samples.len() > MAX_SAMPLES {
            w.samples.pop_front();
        }
    }

    /// A completed request of `class` decoded `tokens` tokens after its first.
    pub fn record_length(&mut self, class: &str, tokens: u64) {
        let lengths = self.lengths.entry(class.to_owned()).or_default();
        lengths.push_back(tokens);
        if lengths.len() > LENGTHS_KEPT {
            lengths.pop_front();
        }
    }

    /// The host's context counter of a classified request (its decode progress once it runs).
    pub fn note_progress(&mut self, request_id: &str, progress: RequestProgress, now: Time) {
        bound(&mut self.progress, request_id);
        self.progress.insert(request_id.to_owned(), (progress, now));
    }

    /// A lifecycle event of any request, every class, with no class or session known.
    pub fn on_event(&mut self, event: &ClassifyEvent, now: Time) {
        self.on_event_of(event, None, None, now);
    }

    /// A lifecycle event of a request of `class` (the default class when unknown) in `session`.
    pub fn on_event_of(
        &mut self,
        event: &ClassifyEvent,
        class: Option<&str>,
        session: Option<&str>,
        now: Time,
    ) {
        self.tick(now);
        let class = class.unwrap_or(self.default_class.as_str()).to_owned();
        match event {
            ClassifyEvent::Sent { request_id, worker } => {
                // A (re)sent request decodes afresh.
                self.reserved.remove(request_id.as_str());
                let progress = self.progress.remove(request_id.as_str()).map(|(p, _)| p);
                bound(&mut self.live, request_id);
                self.live.insert(
                    request_id.clone(),
                    Live {
                        worker: *worker,
                        class,
                        since: now,
                        first: None,
                        progress,
                        base: None,
                    },
                );
                if let Some(session) = session {
                    bound(&mut self.home, session);
                    self.home.insert(session.to_owned(), *worker);
                }
                self.touch(*worker, now);
            }
            ClassifyEvent::Responding { request_id, worker } => {
                if !self.live.contains_key(request_id.as_str()) {
                    let progress = self.progress.remove(request_id.as_str()).map(|(p, _)| p);
                    bound(&mut self.live, request_id);
                    self.live.insert(
                        request_id.clone(),
                        Live {
                            worker: *worker,
                            class,
                            since: now,
                            first: None,
                            progress,
                            base: None,
                        },
                    );
                }
                if let Some(live) = self.live.get_mut(request_id.as_str())
                    && live.first.is_none()
                {
                    live.first = Some(now);
                    live.base = live.progress.as_ref().map(RequestProgress::context_tokens);
                }
                self.touch(*worker, now);
            }
            ClassifyEvent::Completed {
                request_id,
                worker,
                output_tokens,
                ..
            } => {
                self.reserved.remove(request_id.as_str());
                self.progress.remove(request_id.as_str());
                if let Some(live) = self.live.remove(request_id.as_str())
                    && let Some(first) = live.first
                    && let Some(out) = *output_tokens
                    && out > 1
                {
                    self.record(*worker, first, now, (out - 1) as u64);
                    self.record_length(&live.class, (out - 1) as u64);
                }
            }
            ClassifyEvent::Aborted { request_id, .. } => {
                self.reserved.remove(request_id.as_str());
                self.progress.remove(request_id.as_str());
                self.live.remove(request_id.as_str());
            }
            _ => {}
        }
    }

    /// `decode_cap`: active requests of the capped classes on `worker` (sent and not finished,
    /// plus turns let go to it and not sent yet).
    pub fn active(&self, worker: WorkerWithDpRank) -> u32 {
        let live = self
            .live
            .values()
            .filter(|l| l.worker == worker && self.applies_to(&l.class))
            .count();
        let reserved = self
            .reserved
            .iter()
            .filter(|(id, (w, _))| *w == worker && !self.live.contains_key(id.as_str()))
            .count();
        u32::try_from(live + reserved).unwrap_or(u32::MAX)
    }

    /// `decode_cap`: `worker`'s cap now, `None` when it has none (option off, `auto` with `m >= 1`
    /// or no base learned yet).
    pub fn cap(&self, worker: WorkerWithDpRank) -> Option<u32> {
        let cfg = self.cfg.decode_cap.as_ref()?;
        let m = self.m(worker);
        let base = match cfg.base {
            CapBase::Fixed(n) => n,
            CapBase::Auto if m >= 1.0 => return None,
            CapBase::Auto => self.workers.get(&worker)?.base_auto?,
        };
        Some(((m * f64::from(base)).round() as u32).max(1))
    }

    /// `decode_cap`: the learned `auto` base of `worker`.
    pub fn base_auto(&self, worker: WorkerWithDpRank) -> Option<u32> {
        self.workers.get(&worker)?.base_auto
    }

    /// `decode_cap`: (turns moved to another worker, turns held) so far.
    pub fn cap_counts(&self) -> (u64, u64) {
        (self.cap_moved, self.cap_held)
    }

    fn has_room(&self, worker: WorkerWithDpRank) -> bool {
        self.cap(worker).is_none_or(|cap| self.active(worker) < cap)
    }

    /// `decode_cap`: the gate's answer for turn `request_id` of `class` in `session`, which the
    /// gate lets go to `target` (`None`: the router picks). [`CapVerdict::Pass`] with the option
    /// off, for a class not capped, and when the turn's worker (`target`, else the session's last
    /// worker) is under its cap or, with no worker known, while no worker is at its cap.
    pub fn cap_verdict(
        &mut self,
        request_id: &str,
        class: &str,
        session: Option<&str>,
        target: Option<WorkerWithDpRank>,
        now: Time,
    ) -> CapVerdict {
        if self.cfg.decode_cap.is_none() || !self.applies_to(class) {
            return CapVerdict::Pass;
        }
        self.tick(now);
        let home = target.or_else(|| session.and_then(|s| self.home.get(s).copied()));
        if let Some(home) = home
            && self.has_room(home)
        {
            self.reserve(request_id, home, now);
            return CapVerdict::Pass;
        }
        if home.is_none() && self.workers.keys().all(|&w| self.has_room(w)) {
            return CapVerdict::Pass;
        }
        let best = self
            .workers
            .iter()
            .filter(|(_, c)| now.since(c.seen).as_secs_f64() <= STALE_S)
            .map(|(&w, _)| w)
            .filter(|&w| self.has_room(w))
            .min_by_key(|&w| {
                let active = self.active(w);
                let room = self.cap(w).map_or(u32::MAX, |c| c.saturating_sub(active));
                (Reverse(room), active, w)
            });
        match best {
            Some(worker) => {
                self.reserve(request_id, worker, now);
                self.cap_moved += 1;
                CapVerdict::Go(worker)
            }
            None => {
                self.cap_held += 1;
                CapVerdict::Hold(now.after(Duration::from_secs_f64(self.cfg.interval_s)))
            }
        }
    }

    fn reserve(&mut self, request_id: &str, worker: WorkerWithDpRank, now: Time) {
        if !self.live.contains_key(request_id) {
            bound(&mut self.reserved, request_id);
            self.reserved.insert(request_id.to_owned(), (worker, now));
        }
    }

    /// Step every worker once per interval.
    pub fn tick(&mut self, now: Time) {
        let interval = Duration::from_secs_f64(self.cfg.interval_s);
        let Some(next) = self.next_step else {
            self.next_step = Some(now.after(interval));
            return;
        };
        if now < next {
            return;
        }
        let window_s = self.cfg.window_s;
        let start = window_start(now, window_s);
        self.live
            .retain(|_, l| now.since(l.since).as_secs_f64() <= TRACK_S);
        self.progress
            .retain(|_, (_, at)| now.since(*at).as_secs_f64() <= TRACK_S);
        self.reserved
            .retain(|_, (_, at)| now.since(*at).as_secs_f64() <= RESERVE_S);
        for w in self.workers.values_mut() {
            while w.samples.front().is_some_and(|s| s.done <= start) {
                w.samples.pop_front();
            }
        }
        self.workers
            .retain(|_, w| !w.samples.is_empty() || now.since(w.seen).as_secs_f64() <= STALE_S);
        // Read every worker first (the readings look at the requests in flight), then step.
        let cap_on = self.cfg.decode_cap.is_some();
        let views: Vec<(WorkerWithDpRank, Reading, f64, u32)> = self
            .workers
            .keys()
            .map(|&w| {
                let active = if cap_on { self.active(w) } else { 0 };
                (w, self.reading(w, now), self.worker_target_s(w), active)
            })
            .collect();
        let auto = self
            .cfg
            .decode_cap
            .as_ref()
            .is_some_and(|c| c.base == CapBase::Auto);
        let at = now.as_secs_f64();
        let mut lines = Vec::new();
        for (worker, reading, target_s, active) in views {
            let Some(w) = self.workers.get_mut(&worker) else {
                continue;
            };
            // `auto`: learn the base from the intervals the cap was off (m at least 1).
            if auto {
                if w.m >= 1.0 {
                    w.active_hist.push_back((now, active));
                }
                while w
                    .active_hist
                    .front()
                    .is_some_and(|(t, _)| now.since(*t).as_secs_f64() > AUTO_WINDOW_S)
                {
                    w.active_hist.pop_front();
                }
                if let Some(p) = quantile(
                    w.active_hist.iter().map(|(_, n)| u64::from(*n)).collect(),
                    AUTO_QUANTILE,
                ) {
                    w.base_auto = Some(u32::try_from(p).unwrap_or(u32::MAX).max(1));
                }
            }
            let e = match reading.measured_itl_s {
                Some(measured)
                    if reading.samples >= self.cfg.min_samples && measured.is_finite() =>
                {
                    let (e, u, m) = control(&self.cfg, target_s, w.u, w.m, measured);
                    w.u = u;
                    w.m = m;
                    Some(e)
                }
                _ => None,
            };
            if self.log.is_some() {
                let mut line = format!(
                    "{{\"t\":{at:.3},\"worker\":\"{}:{}\",\"measured_itl_s\":{},\"samples\":{},\"running\":{},\"target_s\":{:.6},\"e\":{},\"u\":{:.6},\"m\":{:.6}",
                    worker.worker_id,
                    worker.dp_rank,
                    json_f64(reading.measured_itl_s),
                    reading.samples,
                    reading.running,
                    target_s,
                    json_f64(e),
                    w.u,
                    w.m,
                );
                if cap_on {
                    let cap = self.cap(worker);
                    line.push_str(&format!(
                        ",\"active\":{active},\"cap\":{}",
                        cap.map_or_else(|| "null".to_owned(), |c| c.to_string())
                    ));
                }
                line.push('}');
                lines.push(line);
            }
        }
        if let Some(log) = self.log.as_mut() {
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
}

fn json_f64(value: Option<f64>) -> String {
    value.map_or_else(|| "null".to_owned(), |v| format!("{v:.6}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admit::{GateAnswer, KvBudget, room, room_scaled};
    use crate::config::Mode;
    use crate::scheduler::Scheduler;

    fn t(s: f64) -> Time {
        Time::from_secs_f64(s)
    }

    fn w(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(id, 0)
    }

    fn policy(cfg: Option<StepTargetConfig>) -> ServicePolicy {
        let mut p = ServicePolicy {
            mode: Mode::Enforce,
            ..Default::default()
        };
        p.admission.step_target = cfg;
        p
    }

    fn controller(cfg: StepTargetConfig) -> StepTarget {
        StepTarget::from_policy(&policy(Some(cfg))).unwrap()
    }

    /// Five requests on `worker` decoding 100 tokens each at `itl_s`, all done by `at_s`.
    fn feed(st: &mut StepTarget, worker: WorkerWithDpRank, itl_s: f64, at_s: f64) {
        for i in 0..5 {
            let done = at_s - f64::from(i);
            st.record(worker, t(done - 100.0 * itl_s), t(done), 100);
        }
    }

    /// Run `steps` intervals from `start_s` (the first tick only arms the clock), each with
    /// samples at `itl_s` on every listed worker; the multiples after each step.
    fn run(
        st: &mut StepTarget,
        start_s: f64,
        steps: usize,
        load: &[(WorkerWithDpRank, f64)],
    ) -> Vec<Vec<f64>> {
        st.tick(t(start_s));
        let interval = st.cfg.interval_s;
        (1..=steps)
            .map(|k| {
                let at = start_s + interval * k as f64;
                for &(worker, itl) in load {
                    feed(st, worker, itl, at);
                }
                st.tick(t(at));
                load.iter().map(|&(worker, _)| st.m(worker)).collect()
            })
            .collect()
    }

    fn sent(id: &str, worker: WorkerWithDpRank) -> ClassifyEvent {
        ClassifyEvent::Sent {
            request_id: id.into(),
            worker,
        }
    }

    fn responding(id: &str, worker: WorkerWithDpRank) -> ClassifyEvent {
        ClassifyEvent::Responding {
            request_id: id.into(),
            worker,
        }
    }

    fn completed(id: &str, worker: WorkerWithDpRank, out: Option<usize>) -> ClassifyEvent {
        ClassifyEvent::Completed {
            request_id: id.into(),
            worker,
            context_tokens: None,
            output_tokens: out,
        }
    }

    /// Request `id` of `class` sent to `worker` at `at_s`, first token at once.
    fn start(st: &mut StepTarget, id: &str, class: &str, worker: WorkerWithDpRank, at_s: f64) {
        st.on_event_of(&sent(id, worker), Some(class), None, t(at_s));
        st.on_event_of(&responding(id, worker), Some(class), None, t(at_s));
    }

    #[test]
    fn the_multiple_rises_under_slack_and_falls_over_the_target() {
        let mut st = controller(StepTargetConfig::default());
        assert!((st.target_s() - 0.85 * 0.05).abs() < 1e-12);
        // 20 ms against 42.5 ms: slack. The multiple climbs, never down, to the cap.
        let up = run(&mut st, 0.0, 30, &[(w(1), 0.02)]);
        assert!(up[0][0] > 1.0, "{up:?}");
        assert!(up.windows(2).all(|p| p[1][0] >= p[0][0]), "{up:?}");
        assert!((up[29][0] - 1.8).abs() < 1e-12, "{up:?}");
        // 80 ms: over the target. It falls, never up, to the floor.
        let mut st = controller(StepTargetConfig::default());
        let down = run(&mut st, 0.0, 30, &[(w(1), 0.08)]);
        assert!(down[0][0] < 1.0, "{down:?}");
        assert!(down.windows(2).all(|p| p[1][0] <= p[0][0]), "{down:?}");
        assert!((down[29][0] - 0.7).abs() < 1e-12, "{down:?}");
    }

    #[test]
    fn the_slew_limit_bounds_each_step() {
        let cfg = StepTargetConfig::default();
        let target = 0.0425;
        // A huge slack or overload moves m by 5% of its last value per step, no more.
        let (_, _, m) = control(&cfg, target, 0.0, 1.0, 0.0001);
        assert_eq!(m, 1.0 + 0.05);
        let (_, _, m2) = control(&cfg, target, 0.0, m, 0.0001);
        assert_eq!(m2, m * (1.0 + 0.05));
        let (_, _, m) = control(&cfg, target, 0.0, 1.0, 1.0);
        assert_eq!(m, 1.0 - 0.05);
        // A small error inside the slew band is taken whole.
        let (e, u, m) = control(&cfg, target, 0.0, 1.0, 0.0425 * 0.99);
        assert!((m - (cfg.kp * e + u).exp()).abs() < 1e-12 && m > 1.0 && m < 1.05);
        // Through the controller: consecutive multiples never differ by more than the slew.
        let mut st = controller(StepTargetConfig::default());
        let path = run(&mut st, 0.0, 12, &[(w(1), 0.001)]);
        let mut prev = 1.0;
        for step in &path {
            assert!(step[0] <= prev * 1.05 + 1e-12 && step[0] >= prev * 0.95 - 1e-12);
            prev = step[0];
        }
    }

    #[test]
    fn the_clamps_hold_and_the_integral_does_not_wind_up() {
        let cfg = StepTargetConfig {
            slew: 0.5,
            ..StepTargetConfig::default()
        };
        let mut st = controller(cfg.clone());
        // A long, extreme slack: m sits at max, and u stops growing once the output saturates,
        // well under its own clamp ln(max).
        let up = run(&mut st, 0.0, 200, &[(w(1), 0.001)]);
        assert!(up.iter().all(|s| s[0] <= cfg.max && s[0] >= cfg.min));
        assert_eq!(up[199][0], cfg.max);
        let u_sat = st.u(w(1));
        assert!(u_sat < cfg.max.ln() - 0.05, "u {u_sat}");
        let u_mid = {
            let mut probe = controller(cfg.clone());
            run(&mut probe, 0.0, 5, &[(w(1), 0.001)]);
            probe.u(w(1))
        };
        assert_eq!(u_sat, u_mid, "no integration while saturated");
        // Then an overload: m leaves max at the first step and reaches the floor by the third
        // (the window still holds the earlier fast requests at first); no long recovery from the
        // saturation.
        let down = run(&mut st, 2_010.0, 40, &[(w(1), 0.2)]);
        assert!(down[0][0] < cfg.max, "{down:?}");
        assert!(down[2][0] <= cfg.min + 1e-12, "{down:?}");
        assert!(down.iter().all(|s| s[0] <= cfg.max && s[0] >= cfg.min));
        // And under a sustained overload the integral does not run below its floor either.
        assert!(st.u(w(1)) >= cfg.min.ln());
        // The pure step never leaves [min, max], whatever the error.
        for measured in [0.0, 1e-6, 0.01, 0.0425, 0.1, 10.0] {
            for (u, m) in [(-5.0f64, 0.7), (0.0, 1.0), (5.0, 1.8)] {
                let u = u.clamp(cfg.min.ln(), cfg.max.ln());
                let (_, u2, m2) = control(&cfg, 0.0425, u, m, measured);
                assert!(
                    (cfg.min..=cfg.max).contains(&m2),
                    "{measured} {u} {m} -> {m2}"
                );
                assert!((cfg.min.ln()..=cfg.max.ln()).contains(&u2));
            }
        }
    }

    #[test]
    fn workers_are_steered_independently_and_the_fleet_takes_the_hbm_weighted_mean() {
        let mut st = controller(StepTargetConfig::default());
        st.observe_workers(&[(w(1), 1_000), (w(2), 3_000), (w(3), 1_000)], t(0.0));
        let path = run(&mut st, 0.0, 10, &[(w(1), 0.02), (w(2), 0.08)]);
        let last = &path[9];
        assert!(last[0] > 1.0 && last[1] < 1.0, "{path:?}");
        // Worker 3 has no samples: it holds 1.
        assert_eq!((st.m(w(3)), st.u(w(3))), (1.0, 0.0));
        // Worker 1's path is the same as alone: worker 2's overload does not reach it.
        let mut alone = controller(StepTargetConfig::default());
        let solo = run(&mut alone, 0.0, 10, &[(w(1), 0.02)]);
        assert_eq!(solo[9][0], last[0]);
        let expected = (1_000.0 * last[0] + 3_000.0 * last[1] + 1_000.0) / 5_000.0;
        assert!((st.fleet_m() - expected).abs() < 1e-12);
        assert_eq!(st.scale("standard"), st.fleet_m());
        assert_eq!(st.scale("batch"), st.fleet_m());
        assert_eq!(st.scale("priority"), 1.0);
        assert_eq!(st.worker_scale(w(1), "standard"), last[0]);
        assert_eq!(st.worker_scale(w(1), "priority"), 1.0);
    }

    #[test]
    fn too_few_samples_hold_the_multiple() {
        let mut st = controller(StepTargetConfig::default());
        st.tick(t(0.0));
        st.record(w(1), t(5.0), t(6.0), 100);
        st.record(w(1), t(5.0), t(7.0), 100);
        st.tick(t(10.0));
        assert_eq!(st.reading(w(1), t(10.0)).samples, 2);
        assert_eq!((st.m(w(1)), st.u(w(1))), (1.0, 0.0));
        st.record(w(1), t(15.0), t(16.0), 100);
        st.tick(t(20.0));
        assert!(st.m(w(1)) > 1.0);
    }

    #[test]
    fn the_measure_is_the_decode_in_the_window_from_lifecycle_events() {
        let mut st = controller(StepTargetConfig {
            sample_running: false,
            ..StepTargetConfig::default()
        });
        let responding = |id: &str| responding(id, w(1));
        let completed = |id: &str, out| completed(id, w(1), out);
        // 101 output tokens: 100 decoded over 2 s, 20 ms each.
        st.on_event(&responding("a"), t(100.0));
        st.on_event(&completed("a", Some(101)), t(102.0));
        let r = st.reading(w(1), t(102.0));
        assert_eq!(r.samples, 1);
        assert!((r.measured_itl_s.unwrap() - 0.02).abs() < 1e-12);
        // A request decoding 90 s at 100 ms per token: only its last 45 s count, at its own ITL,
        // so the mean is token-weighted.
        st.on_event(&responding("b"), t(12.0));
        st.on_event(&completed("b", Some(901)), t(102.0));
        let r = st.reading(w(1), t(102.0));
        assert_eq!(r.samples, 2);
        let expected = (2.0 + 45.0) / (100.0 + 450.0);
        assert!(
            (r.measured_itl_s.unwrap() - expected).abs() < 1e-12,
            "{r:?}"
        );
        // No output count, a single token, an abort, or a resend before completion: no sample.
        st.on_event(&responding("c"), t(103.0));
        st.on_event(&completed("c", None), t(104.0));
        st.on_event(&responding("d"), t(103.0));
        st.on_event(&completed("d", Some(1)), t(104.0));
        st.on_event(&responding("e"), t(103.0));
        st.on_event(
            &ClassifyEvent::Aborted {
                request_id: "e".into(),
                worker: Some(w(1)),
                error: None,
            },
            t(104.0),
        );
        st.on_event(&completed("e", Some(50)), t(104.0));
        st.on_event(&responding("f"), t(50.0));
        st.on_event(&sent("f", w(1)), t(103.0));
        st.on_event(&completed("f", Some(50)), t(104.0));
        assert_eq!(st.reading(w(1), t(104.0)).samples, 2);
        // Past the window nothing counts.
        assert_eq!(st.reading(w(1), t(200.0)).samples, 0);
        // The completed decode lengths are kept per class (the default class here).
        assert_eq!(st.class_length("standard"), Some(900));
        assert_eq!(st.class_length("batch"), None);
    }

    #[test]
    fn running_decodes_are_counted_by_the_host_count_when_it_has_one() {
        let mut st = controller(StepTargetConfig::default());
        let (progress, updater) = RequestProgress::new(1_000);
        st.note_progress("r", progress, t(0.0));
        st.on_event_of(&sent("r", w(1)), Some("batch"), None, t(0.0));
        // The prompt and the first token are in the counter at the first token.
        updater.update_context_tokens(1_001);
        st.on_event_of(&responding("r", w(1)), Some("batch"), None, t(100.0));
        // Before any decode token: nothing counted (and no class lengths: neutral, not a sample).
        let r = st.reading(w(1), t(100.5));
        assert_eq!((r.samples, r.running), (0, 0));
        // 200 tokens over 10 s: 50 ms each, counted as a running sample.
        updater.update_context_tokens(1_201);
        let r = st.reading(w(1), t(110.0));
        assert_eq!((r.samples, r.running), (1, 1));
        assert!((r.measured_itl_s.unwrap() - 0.05).abs() < 1e-12, "{r:?}");
        // 100 s on, 900 more tokens: only the window's share counts, at the request's mean ITL.
        updater.update_context_tokens(2_101);
        let r = st.reading(w(1), t(200.0));
        assert!(
            (r.measured_itl_s.unwrap() - 0.1 / 1.1 * 1.0).abs() < 1e-9,
            "{r:?}"
        );
        // With sample_running off the same request is invisible until it completes (f49j).
        let mut off = controller(StepTargetConfig {
            sample_running: false,
            ..StepTargetConfig::default()
        });
        let (progress, updater) = RequestProgress::new(1_000);
        off.note_progress("r", progress, t(0.0));
        off.on_event_of(&sent("r", w(1)), Some("batch"), None, t(0.0));
        off.on_event_of(&responding("r", w(1)), Some("batch"), None, t(100.0));
        updater.update_context_tokens(1_201);
        assert_eq!(off.reading(w(1), t(110.0)).samples, 0);
        // Completion replaces the running sample with the completed one, never both.
        st.on_event_of(
            &completed("r", w(1), Some(1_101)),
            Some("batch"),
            None,
            t(200.0),
        );
        let r = st.reading(w(1), t(200.0));
        assert_eq!((r.samples, r.running), (1, 0));
    }

    #[test]
    fn long_running_decodes_without_a_host_count_count_past_their_class_length() {
        let mut st = controller(StepTargetConfig::default());
        // Standard requests decode 100 tokens (p90 100); three finish at 20 ms just now.
        for i in 0..3 {
            let id = format!("c{i}");
            let at = 1_000.0 + f64::from(i);
            start(&mut st, &id, "standard", w(1), at - 2.0);
            st.on_event_of(
                &completed(&id, w(1), Some(101)),
                Some("standard"),
                None,
                t(at),
            );
        }
        assert_eq!(st.class_length("standard"), Some(100));
        let fast = st.reading(w(1), t(1_003.0));
        assert_eq!((fast.samples, fast.running), (3, 0));
        assert!((fast.measured_itl_s.unwrap() - 0.02).abs() < 1e-12);
        // A standard request decoding for 1 s: at 20 ms it would be at 50 tokens, inside its
        // class's length: neutral weight at the reference ITL, not a sample.
        start(&mut st, "short", "standard", w(1), 1_002.0);
        let r = st.reading(w(1), t(1_003.0));
        assert_eq!((r.samples, r.running), (3, 0));
        assert!((r.measured_itl_s.unwrap() - 0.02).abs() < 1e-12, "{r:?}");
        // Two decoding for 300 s: at 20 ms they would have finished long ago. Each counts as at
        // most 100 tokens over 300 s (3 s each), 15 of them in the window: the mean rises.
        start(&mut st, "long1", "standard", w(1), 703.0);
        start(&mut st, "long2", "standard", w(1), 703.0);
        let r = st.reading(w(1), t(1_003.0));
        assert_eq!((r.samples, r.running), (5, 2));
        let completed_time = 6.0;
        let expected = (completed_time + 1.0 + 90.0) / (300.0 + 50.0 + 30.0);
        assert!((r.measured_itl_s.unwrap() - expected).abs() < 1e-9, "{r:?}");
        // Through the controller: the long decodes pull the worker's multiple down where the
        // completed requests alone (f49j's measure) would push it up.
        let drive = |sample_running: bool| {
            let mut st = controller(StepTargetConfig {
                sample_running,
                ..StepTargetConfig::default()
            });
            st.tick(t(0.0));
            for k in 0..20 {
                let at = 10.0 * f64::from(k);
                for i in 0..3 {
                    let id = format!("c{k}-{i}");
                    start(&mut st, &id, "standard", w(1), at + 1.0);
                    st.on_event_of(
                        &completed(&id, w(1), Some(101)),
                        Some("standard"),
                        None,
                        t(at + 3.0),
                    );
                }
                if k == 0 {
                    for i in 0..6 {
                        start(&mut st, &format!("long{i}"), "standard", w(1), 0.5);
                    }
                }
                st.tick(t(at + 10.0));
            }
            st.m(w(1))
        };
        let (f49j, f49j2) = (drive(false), drive(true));
        assert!(f49j > 1.0 && f49j2 < 1.0, "{f49j} {f49j2}");
    }

    #[test]
    fn f49l3_per_class_setpoints_set_each_workers_target() {
        let mut st = controller(StepTargetConfig::default());
        start(&mut st, "p", "priority", w(1), 0.0);
        start(&mut st, "s1", "standard", w(1), 0.0);
        start(&mut st, "s2", "standard", w(2), 0.0);
        start(&mut st, "b", "batch", w(3), 0.0);
        let fracs: BTreeMap<String, f64> =
            [("priority".to_owned(), 0.95), ("standard".to_owned(), 0.6)]
                .into_iter()
                .collect();
        st.set_class_fracs(&fracs);
        // Both decoding on 1: min(0.95 x 50 ms, 0.6 x 100 ms) = 47.5 ms.
        assert!((st.worker_target_s(w(1)) - 0.95 * 0.05).abs() < 1e-12);
        // Standard alone on 2: its own setpoint.
        assert!((st.worker_target_s(w(2)) - 0.6 * 0.1).abs() < 1e-12);
        // Batch has no ITL limit; idle 4: priority's setpoint.
        assert!((st.worker_target_s(w(3)) - 0.95 * 0.05).abs() < 1e-12);
        assert!((st.worker_target_s(w(4)) - 0.95 * 0.05).abs() < 1e-12);
        assert_eq!((st.target_frac(), st.class_frac("standard")), (0.95, 0.6));
        // Standard tighter than priority on worker 1.
        let fracs: BTreeMap<String, f64> =
            [("priority".to_owned(), 0.95), ("standard".to_owned(), 0.4)]
                .into_iter()
                .collect();
        st.set_class_fracs(&fracs);
        assert!((st.worker_target_s(w(1)) - 0.4 * 0.1).abs() < 1e-12);
        // Without the call: one target_frac for every class (f49j2).
        let plain = controller(StepTargetConfig::default());
        assert_eq!(plain.class_frac("standard"), 0.85);
    }

    #[test]
    fn the_target_is_the_tightest_class_decoding_on_the_worker() {
        let mut st = controller(StepTargetConfig::default());
        start(&mut st, "p", "priority", w(1), 0.0);
        start(&mut st, "s1", "standard", w(1), 0.0);
        start(&mut st, "s2", "standard", w(2), 0.0);
        start(&mut st, "b", "batch", w(3), 0.0);
        // Sent, no first token yet: not decoding.
        st.on_event_of(&sent("q", w(5)), Some("priority"), None, t(0.0));
        st.on_event_of(&sent("s3", w(5)), Some("standard"), None, t(0.0));
        assert!((st.worker_target_s(w(1)) - 0.85 * 0.05).abs() < 1e-12);
        assert!((st.worker_target_s(w(2)) - 0.85 * 0.1).abs() < 1e-12);
        // Batch has no ITL limit; with nothing else decoding the configured class's.
        assert!((st.worker_target_s(w(3)) - 0.85 * 0.05).abs() < 1e-12);
        assert!((st.worker_target_s(w(4)) - 0.85 * 0.05).abs() < 1e-12);
        assert!((st.worker_target_s(w(5)) - 0.85 * 0.05).abs() < 1e-12);
        // The priority request finishes: worker 1 is held to standard's limit from then on.
        st.on_event_of(&completed("p", w(1), None), Some("priority"), None, t(1.0));
        assert!((st.worker_target_s(w(1)) - 0.85 * 0.1).abs() < 1e-12);
        // Through the controller: 60 ms is over priority's 42.5 ms target but under standard's
        // 85 ms, so a worker decoding only standard rises where one decoding priority falls.
        let mut st = controller(StepTargetConfig::default());
        start(&mut st, "p", "priority", w(1), 0.0);
        start(&mut st, "s", "standard", w(2), 0.0);
        let path = run(&mut st, 0.0, 5, &[(w(1), 0.06), (w(2), 0.06)]);
        assert!(path[4][0] < 1.0 && path[4][1] > 1.0, "{path:?}");
    }

    fn capped(base: CapBase) -> StepTarget {
        controller(StepTargetConfig {
            decode_cap: Some(DecodeCapConfig { base }),
            ..StepTargetConfig::default()
        })
    }

    #[test]
    fn the_decode_cap_holds_at_the_cap_passes_under_it_and_never_caps_priority() {
        let mut st = capped(CapBase::Fixed(2));
        st.observe_workers(&[(w(1), 1_000), (w(2), 1_000)], t(0.0));
        // m 1: a cap of 2 per worker.
        assert_eq!((st.cap(w(1)), st.cap(w(2))), (Some(2), Some(2)));
        st.on_event_of(&sent("a1", w(1)), Some("standard"), Some("a"), t(0.0));
        assert_eq!(st.active(w(1)), 1);
        // Under the cap: the gate's decision stands (the session's home is worker 1).
        assert_eq!(
            st.cap_verdict("a2", "standard", Some("a"), None, t(1.0)),
            CapVerdict::Pass
        );
        // That turn holds worker 1's second place until it is sent.
        assert_eq!(st.active(w(1)), 2);
        st.on_event_of(&sent("a2", w(1)), Some("standard"), Some("a"), t(1.0));
        assert_eq!(st.active(w(1)), 2);
        // At the cap: a batch turn for worker 1 goes to worker 2 (work-conserving).
        assert_eq!(
            st.cap_verdict("b1", "batch", None, Some(w(1)), t(2.0)),
            CapVerdict::Go(w(2))
        );
        st.on_event_of(&sent("b1", w(2)), Some("batch"), Some("b"), t(2.0));
        st.on_event_of(&sent("c1", w(2)), Some("standard"), Some("c"), t(2.0));
        // Both at the cap: a standard turn holds until the next interval at the latest.
        assert_eq!(
            st.cap_verdict("a3", "standard", Some("a"), None, t(3.0)),
            CapVerdict::Hold(t(13.0))
        );
        // A new session (no home) holds too, now that every worker is at its cap.
        assert!(matches!(
            st.cap_verdict("n1", "standard", Some("new"), None, t(3.0)),
            CapVerdict::Hold(_)
        ));
        // Priority is never capped.
        assert_eq!(
            st.cap_verdict("p1", "priority", Some("a"), Some(w(1)), t(3.0)),
            CapVerdict::Pass
        );
        st.on_event_of(&sent("p1", w(1)), Some("priority"), Some("p"), t(3.0));
        assert_eq!(st.active(w(1)), 2);
        // A completion frees worker 1: the held turn goes home.
        st.on_event_of(
            &completed("a1", w(1), Some(10)),
            Some("standard"),
            None,
            t(4.0),
        );
        assert_eq!(
            st.cap_verdict("a3", "standard", Some("a"), None, t(4.0)),
            CapVerdict::Pass
        );
        assert_eq!(st.cap_counts(), (1, 2));
        // An abort and a lapsed reservation free their places too.
        st.on_event_of(
            &ClassifyEvent::Aborted {
                request_id: "a2".into(),
                worker: Some(w(1)),
                error: None,
            },
            None,
            None,
            t(5.0),
        );
        assert_eq!(st.active(w(1)), 1);
        st.tick(t(100.0));
        assert_eq!(st.active(w(1)), 0);
    }

    #[test]
    fn the_decode_cap_follows_the_multiple_and_auto_learns_its_base_while_uncapped() {
        // Fixed: round(m x base), at least 1.
        let mut st = capped(CapBase::Fixed(10));
        run(&mut st, 0.0, 30, &[(w(1), 0.08)]);
        assert!((st.m(w(1)) - 0.7).abs() < 1e-12);
        assert_eq!(st.cap(w(1)), Some(7));
        let mut st = capped(CapBase::Fixed(1));
        run(&mut st, 0.0, 30, &[(w(1), 0.08)]);
        assert_eq!(st.cap(w(1)), Some(1));
        // Auto: no cap while m >= 1; the base is the p90 of the worker's active count over the
        // last 10 minutes, sampled while m >= 1.
        let mut st = capped(CapBase::Auto);
        st.tick(t(0.0));
        for k in 1..=20u32 {
            // Active standard requests on worker 1 cycle 1..=10.
            let n = (k - 1) % 10 + 1;
            for i in 0..n {
                st.on_event_of(
                    &sent(&format!("{k}-{i}"), w(1)),
                    Some("standard"),
                    None,
                    t(f64::from(k) * 10.0 - 1.0),
                );
            }
            st.tick(t(f64::from(k) * 10.0));
            for i in 0..n {
                st.on_event_of(
                    &ClassifyEvent::Aborted {
                        request_id: format!("{k}-{i}"),
                        worker: Some(w(1)),
                        error: None,
                    },
                    None,
                    None,
                    t(f64::from(k) * 10.0),
                );
            }
        }
        assert_eq!(st.m(w(1)), 1.0);
        assert_eq!(st.base_auto(w(1)), Some(9));
        assert_eq!(st.cap(w(1)), None);
        // Over the target, m falls: the cap is round(m x 9), and the base no longer moves.
        run(&mut st, 200.0, 30, &[(w(1), 0.08)]);
        assert!((st.m(w(1)) - 0.7).abs() < 1e-12);
        assert_eq!(st.base_auto(w(1)), Some(9));
        assert_eq!(st.cap(w(1)), Some(6));
    }

    #[test]
    fn the_decode_cap_through_the_scheduler_is_enforce_only() {
        let cfg = StepTargetConfig {
            decode_cap: Some(DecodeCapConfig {
                base: CapBase::Fixed(1),
            }),
            ..StepTargetConfig::default()
        };
        for (mode, expect_hold) in [(Mode::Enforce, true), (Mode::Shadow, false)] {
            let mut s = Scheduler::new(ServicePolicy {
                mode,
                ..policy(Some(cfg.clone()))
            });
            s.on_classify(Some("r1"), "a", None, None, 10, t(0.0));
            s.step_target_event(&sent("r1", w(1)), t(0.0));
            s.on_classify(Some("r2"), "b", None, Some("batch"), 10, t(1.0));
            let verdict = s.step_target_cap("r2", "b", Some(w(1)), t(1.0));
            assert_eq!(
                matches!(verdict, CapVerdict::Hold(_)),
                expect_hold,
                "{mode:?}"
            );
            s.on_classify(Some("r3"), "p", None, Some("priority"), 10, t(1.0));
            assert_eq!(
                s.step_target_cap("r3", "p", Some(w(1)), t(1.0)),
                CapVerdict::Pass
            );
        }
    }

    /// The gate, end to end: a standard session on worker 1 under `decode_cap.base: 1`; the next
    /// standard session holds while worker 1 is the only one, goes to worker 2 once it is listed,
    /// priority is never held, and a completion frees worker 1 for the next.
    #[tokio::test]
    async fn the_decode_cap_at_the_gate_holds_moves_and_rechecks_on_completion() {
        use crate::clock::ManualClock;
        use crate::hooks::SessionSchedulerClassifier;
        use std::sync::Arc;

        let cfg = StepTargetConfig {
            decode_cap: Some(DecodeCapConfig {
                base: CapBase::Fixed(1),
            }),
            ..StepTargetConfig::default()
        };
        fn head<'a>(id: &'a str, session: &'a SessionContext) -> DispatchHead<'a> {
            DispatchHead::new(
                Some(id),
                Some(session),
                10,
                None,
                "default",
                tokio::time::Instant::now(),
            )
        }
        let run = |cfg: Option<StepTargetConfig>| async move {
            let clock = Arc::new(ManualClock::new());
            let mut classifier = SessionSchedulerClassifier::new(
                policy(cfg),
                clock.clone(),
                Arc::new(move || (1_000_000, 1)),
            );
            let gate = classifier.dispatch_gate().unwrap();
            let ctx = |s: &str| SessionContext::new(s.into(), None, None, None);
            let (a, b, c, p) = (ctx("a"), ctx("b"), ctx("c"), ctx("p"));
            let classify = |classifier: &mut SessionSchedulerClassifier,
                            id: &str,
                            session: &str,
                            class: Option<&str>| {
                let mut request = ClassifyRequest::for_test(10, Some(id), Some(ctx(session)));
                if let Some(class) = class {
                    request = request.with_initial_policy_class(class);
                }
                classifier.classify(request, &NoWorkers).unwrap();
            };
            classify(&mut classifier, "a1", "a", None);
            classify(&mut classifier, "b1", "b", None);
            classify(&mut classifier, "p1", "p", Some("priority"));
            let mut answers = Vec::new();
            answers.push(gate.check(&head("a1", &a)));
            gate.dispatched(&head("a1", &a), None);
            classifier.on_event(sent("a1", w(1))).await;
            answers.push(gate.check(&head("b1", &b)));
            classifier.with_scheduler(|s| s.step_target_workers(&[(w(2), 1_000)], t(0.0)));
            answers.push(gate.check(&head("b1", &b)));
            gate.dispatched(&head("b1", &b), Some(w(2)));
            classifier.on_event(sent("b1", w(2))).await;
            answers.push(gate.check(&head("p1", &p)));
            classify(&mut classifier, "c1", "c", None);
            answers.push(gate.check(&head("c1", &c)));
            classifier.on_event(completed("a1", w(1), Some(5))).await;
            answers.push(gate.check(&head("c1", &c)));
            answers
        };
        let on = run(Some(cfg)).await;
        let go = |target| DispatchDecision::Go { target };
        assert_eq!(on[0], go(None), "{on:?}");
        // (The manual clock has no host mapping, so a hold's deadline reads `None` here.)
        assert!(matches!(on[1], DispatchDecision::Hold { .. }), "{on:?}");
        assert_eq!(on[2], go(Some(w(2))), "{on:?}");
        assert_eq!(on[3], go(None), "{on:?}");
        assert!(matches!(on[4], DispatchDecision::Hold { .. }), "{on:?}");
        assert_eq!(on[5], go(Some(w(1))), "{on:?}");
        // Options off: the gate lets every one of them go, untargeted, as before f49j.
        let off = run(None).await;
        assert!(off.iter().all(|d| *d == go(None)), "{off:?}");
        // Step target on without decode_cap: the same as off here.
        let plain = run(Some(StepTargetConfig::default())).await;
        assert_eq!(plain, off);
    }

    /// A standard session `a` decoding 600 tokens; a standard session `b` (600) asks at 1 s and
    /// at 11 s, a priority session `p` (500) at 12 s. HBM 1000, DRAM ample. With `slack` (and the
    /// option on), three fast requests finish on worker 1 before 11 s.
    fn scenario(cfg: Option<StepTargetConfig>, slack: bool) -> (Vec<GateAnswer>, String, f64) {
        let mut s = Scheduler::new(policy(cfg));
        s.on_classify(Some("r1"), "a", None, None, 600, t(0.0));
        s.on_dispatched("r1", "a", t(0.0));
        s.on_classify(Some("r2"), "b", None, None, 600, t(1.0));
        s.step_target_workers(&[(w(1), 1_000)], t(1.0));
        let kv = s.kv_budget(1_000, 100_000);
        let mut answers = vec![s.gate_check("b", t(1.0), &kv)];
        if slack && let Some(st) = s.step_target_mut() {
            for i in 0..3 {
                let done = 4.0 + f64::from(i);
                st.record(w(1), t(done - 2.0), t(done), 200);
            }
        }
        answers.push(s.gate_check("b", t(11.0), &kv));
        s.on_classify(Some("r3"), "p", None, Some("priority"), 500, t(12.0));
        answers.push(s.gate_check("p", t(12.0), &kv));
        let log = format!("{:?}", s.decisions().entries().collect::<Vec<_>>());
        (answers, log, s.step_target_scale("standard"))
    }

    #[test]
    fn standard_room_follows_the_multiple_while_priority_room_does_not() {
        let fast = StepTargetConfig {
            slew: 0.5,
            ..StepTargetConfig::default()
        };
        let (off, _, _) = scenario(None, true);
        let (on, _, scale) = scenario(Some(fast.clone()), true);
        // Standard: 900 x m hot cap less a's 600; held at m 1, admitted at m 1.5.
        assert!((scale - 1.5).abs() < 1e-12, "{scale}");
        assert!(matches!(off[1], GateAnswer::Hold { .. }), "{off:?}");
        assert_eq!(on[1], GateAnswer::Go, "{on:?}");
        // Priority: 1000 less 600 either way, so its 500 waits in both.
        assert!(matches!(off[2], GateAnswer::Hold { .. }), "{off:?}");
        assert_eq!(on[2], off[2]);
        // The rooms themselves: priority's unchanged, standard's scaled, the total budget never.
        let policy = policy(Some(fast));
        let roomy = KvBudget {
            hbm_tokens: 1_000,
            dram_tokens: 100_000,
            hot_tokens: 600,
            total_tokens: 600,
            ..KvBudget::default()
        };
        let priority = policy.class(Some("priority"));
        let standard = policy.class(Some("standard"));
        assert_eq!(room(&policy, priority, &roomy), 400);
        assert_eq!(room(&policy, standard, &roomy), 300);
        assert_eq!(room_scaled(&policy, standard, &roomy, 1.5), 750);
        // The total room (0.95 x 1000 less 600) binds whatever the multiple.
        let tight = KvBudget {
            dram_tokens: 0,
            ..roomy
        };
        assert_eq!(room_scaled(&policy, standard, &tight, 1.8), 350);
    }

    #[test]
    fn options_off_is_identical() {
        // Absent from a policy file: none, never serialized, no controller, every scale 1.
        let p: ServicePolicy = serde_json::from_str(r#"{"mode": "enforce"}"#).unwrap();
        assert!(p.admission.step_target.is_none());
        assert!(
            !serde_json::to_string(&p.admission)
                .unwrap()
                .contains("step_target")
        );
        assert!(StepTarget::from_policy(&p).is_none());
        let mut s = Scheduler::new(p);
        assert!(s.step_target_mut().is_none());
        for class in ["priority", "standard", "batch"] {
            assert_eq!(s.step_target_scale(class), 1.0);
        }
        // f49j2's hooks do nothing when off: no cap, no state from events or progress.
        s.on_classify(Some("r1"), "a", None, None, 10, t(0.0));
        let (progress, _) = RequestProgress::new(10);
        s.step_target_progress("r1", &progress, t(0.0));
        s.step_target_event(&sent("r1", w(1)), t(0.0));
        s.step_target_event(&responding("r1", w(1)), t(1.0));
        for class in ["a", "b"] {
            assert_eq!(
                s.step_target_cap("r1", class, Some(w(1)), t(1.0)),
                CapVerdict::Pass
            );
        }
        assert!(s.step_target().is_none());
        // The same answers and the same decision log with the option off as with it on and no
        // measurement yet; the measured slack is what changes them.
        let (off, off_log, off_scale) = scenario(None, true);
        let (idle, idle_log, _) = scenario(Some(StepTargetConfig::default()), false);
        assert_eq!(off, idle);
        assert_eq!(off_log, idle_log);
        assert_eq!(off_scale, 1.0);
        let (fast, fast_log, _) = scenario(
            Some(StepTargetConfig {
                slew: 0.5,
                ..StepTargetConfig::default()
            }),
            true,
        );
        assert_ne!(off, fast);
        assert_ne!(off_log, fast_log);
        // The scaled room at 1 is the room.
        for (hot, total) in [(0, 0), (600, 600), (2_000, 2_500)] {
            let kv = KvBudget {
                hbm_tokens: 1_000,
                dram_tokens: 1_000,
                hot_tokens: hot,
                total_tokens: total,
                ..KvBudget::default()
            };
            let policy = ServicePolicy::default();
            for c in &policy.classes {
                assert_eq!(room(&policy, c, &kv), room_scaled(&policy, c, &kv, 1.0));
            }
        }
    }

    #[test]
    fn validation() {
        let base = policy(Some(StepTargetConfig::default()));
        base.validate().unwrap();
        let msg = |f: &dyn Fn(&mut StepTargetConfig)| {
            let mut p = base.clone();
            f(p.admission.step_target.as_mut().unwrap());
            p.validate().unwrap_err().to_string()
        };
        assert!(msg(&|c| c.class = "nope".into()).contains("not one of the classes"));
        assert!(msg(&|c| c.class = "batch".into()).contains("no service_level"));
        assert!(msg(&|c| c.min = 1.1).contains("min"));
        assert!(msg(&|c| c.min = 0.0).contains("min"));
        assert!(msg(&|c| c.max = 0.9).contains("max"));
        assert!(msg(&|c| c.max = f64::INFINITY).contains("max"));
        for bad in [0.0, -0.1, 1.0, f64::NAN] {
            assert!(msg(&|c| c.slew = bad).contains("slew"), "{bad}");
        }
        for bad in [0.0, -1.0, f64::INFINITY] {
            assert!(msg(&|c| c.window_s = bad).contains("window_s"), "{bad}");
            assert!(msg(&|c| c.interval_s = bad).contains("interval_s"), "{bad}");
        }
        assert!(msg(&|c| c.target_frac = 0.0).contains("target_frac"));
        assert!(msg(&|c| c.kp = -0.1).contains("kp"));
        assert!(msg(&|c| c.ki = f64::NAN).contains("ki"));
        assert!(msg(&|c| c.min_samples = 0).contains("min_samples"));
        // decode_cap: a fixed base of at least 1, and some class below `class` to cap.
        assert!(
            msg(&|c| c.decode_cap = Some(DecodeCapConfig {
                base: CapBase::Fixed(0)
            }))
            .contains("decode_cap.base")
        );
        let mut lowest = base.clone();
        let top = lowest.classes.iter().map(|c| c.rank).max().unwrap();
        for c in &mut lowest.classes {
            if c.name == "standard" {
                c.rank = top + 1;
            }
        }
        let st = lowest.admission.step_target.as_mut().unwrap();
        st.class = "standard".into();
        st.decode_cap = Some(DecodeCapConfig::default());
        assert!(
            lowest
                .validate()
                .unwrap_err()
                .to_string()
                .contains("there are none")
        );
        // min = 1 = max is allowed (a fixed multiple of 1).
        let mut p = base.clone();
        let c = p.admission.step_target.as_mut().unwrap();
        (c.min, c.max) = (1.0, 1.0);
        p.validate().unwrap();
        // The documented example parses, and an unknown key is refused.
        let p: ServicePolicy = serde_json::from_str(
            r#"{"admission": {"step_target": {"target_frac": 0.85, "class": "priority",
                "window_s": 45, "interval_s": 10, "kp": 0.5, "ki": 0.05, "min": 0.7, "max": 1.8,
                "slew": 0.05, "sample_running": true}}}"#,
        )
        .unwrap();
        assert_eq!(p.admission.step_target, Some(StepTargetConfig::default()));
        assert!(
            serde_json::from_str::<ServicePolicy>(r#"{"admission": {"step_target": {"gain": 1}}}"#)
                .is_err()
        );
        // decode_cap: absent by default and not serialized; auto or a count; nothing else.
        assert!(
            !serde_json::to_string(&StepTargetConfig::default())
                .unwrap()
                .contains("decode_cap")
        );
        let parse = |json: &str| {
            serde_json::from_str::<ServicePolicy>(&format!(
                r#"{{"admission": {{"step_target": {{"decode_cap": {json}}}}}}}"#
            ))
            .map(|p| p.admission.step_target.unwrap().decode_cap.unwrap().base)
        };
        assert_eq!(parse(r#"{"base": "auto"}"#).unwrap(), CapBase::Auto);
        assert_eq!(parse(r#"{}"#).unwrap(), CapBase::Auto);
        assert_eq!(parse(r#"{"base": 12}"#).unwrap(), CapBase::Fixed(12));
        for bad in [
            r#"{"base": "x"}"#,
            r#"{"base": -1}"#,
            r#"{"base": 1.5}"#,
            r#"{"cap": 3}"#,
        ] {
            assert!(parse(bad).is_err(), "{bad}");
        }
        let cap = DecodeCapConfig {
            base: CapBase::Fixed(12),
        };
        assert_eq!(serde_json::to_string(&cap).unwrap(), r#"{"base":12}"#);
        assert_eq!(
            serde_json::to_string(&DecodeCapConfig::default()).unwrap(),
            r#"{"base":"auto"}"#
        );
    }

    #[test]
    fn the_log_writes_one_line_per_worker_per_interval() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("steptgt.jsonl");
        let mut st = StepTarget {
            log: Some(std::io::BufWriter::new(
                std::fs::File::create(&path).unwrap(),
            )),
            ..controller(StepTargetConfig::default())
        };
        st.observe_workers(&[(w(1), 1_000), (w(2), 1_000)], t(0.0));
        run(&mut st, 0.0, 2, &[(w(1), 0.02)]);
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 4, "{text}");
        assert_eq!(
            lines[0],
            r#"{"t":10.000,"worker":"1:0","measured_itl_s":0.020000,"samples":5,"running":0,"target_s":0.042500,"e":0.529412,"u":0.264706,"m":1.050000}"#
        );
        assert_eq!(
            lines[1],
            r#"{"t":10.000,"worker":"2:0","measured_itl_s":null,"samples":0,"running":0,"target_s":0.042500,"e":null,"u":0.000000,"m":1.000000}"#
        );
        // With decode_cap the active count and the cap follow.
        let path = dir.path().join("steptgt-cap.jsonl");
        let mut st = StepTarget {
            log: Some(std::io::BufWriter::new(
                std::fs::File::create(&path).unwrap(),
            )),
            ..capped(CapBase::Fixed(4))
        };
        st.on_event_of(&sent("s", w(1)), Some("standard"), None, t(0.0));
        st.tick(t(10.0));
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            text.lines().next().unwrap(),
            r#"{"t":10.000,"worker":"1:0","measured_itl_s":null,"samples":0,"running":0,"target_s":0.042500,"e":null,"u":0.000000,"m":1.000000,"active":1,"cap":4}"#
        );
        // Unset, nothing is opened.
        if std::env::var_os(STEPTGT_LOG_ENV).is_none() {
            assert!(controller(StepTargetConfig::default()).log.is_none());
        }
    }
}
