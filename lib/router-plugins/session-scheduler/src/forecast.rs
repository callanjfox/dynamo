// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The forecast: one view of the near future, kept off the classifier lock.
//!
//! The classifier and its gate never read or wait on the forecast. They push [`Observation`]s
//! into a bounded channel with `try_send` ([`ForecastFeed`]); a full channel drops the
//! observation and counts it. The forecast owns its own lock and drains the channel when it is
//! queried ([`Forecast::pump`]), so a slow query delays only other queries.
//!
//! # The model
//!
//! Deliberately simple, and documented so its errors can be read:
//!
//! - **Session** ([`Forecast::session`]). The next turn arrives `gap` after the last turn
//!   completed, adds `new_tokens` to the context, and generates `output_tokens`. Each is an
//!   exponentially weighted mean (weight `alpha`) of the session's own history, falling back to
//!   its class's when the session has none. A client hint ([`Observation::Hint`]) overrides the
//!   next turn only. Output tokens are learned as `context_tokens` at completion minus the
//!   prompt, since lifecycle events carry no output length.
//! - **Worker** ([`Forecast::worker`]). From the latest load sample (taken from the classifier's
//!   `LOAD` worker view at most every `snapshot_interval_s`): the queued prefill drains in
//!   `chunk_tokens` chunks, then the running requests decode in one batch whose context grows by
//!   one token per step. Arrivals and completions within the horizon are not modelled.
//! - **What-if** ([`Forecast::what_if`]). Admitting a turn of `new_tokens` over `prefix_tokens`
//!   cached on a worker: its TTFT is the prefill backlog plus its own prefill; the batch's
//!   inter-token latency becomes the decode step at one more request. A miss is the candidate
//!   over its class's TTFT or ITL limit, plus each running request whose class ITL limit the
//!   larger batch crosses.
//!
//! Step times come from a [`StepModel`]: [`TableStepModel`] (interpolated grids, the default
//! and the test model) or, with the `aisimulate` feature, `AisimulateStepModel` (AISimulate's
//! `ForwardPassPerfModel`, the intended production model).
//!
//! Per class, [`ClassErrors`] compares each prediction with what happened (next arrival, new
//! tokens, output tokens, TTFT after dispatch), for shadow mode.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::time::Duration;

use dynamo_kv_router::plugins::prelude::WorkerWithDpRank;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::clock::{Clock, Time};
use crate::config::ServicePolicy;

/// Seconds for one engine step. `None` when the model cannot answer (outside its data).
pub trait StepModel: Send + Sync {
    /// One prefill step of `batch` requests, each `new_tokens` long over `prefix_tokens` cached.
    fn prefill_s(&self, batch: u32, new_tokens: u32, prefix_tokens: u32) -> Option<f64>;

    /// One decode step of `batch` requests, each at `context_tokens`.
    fn decode_s(&self, batch: u32, context_tokens: u32) -> Option<f64>;

    /// k (`forecast.step_table_v2`): one engine step computing a prefill chunk of `new_tokens` over
    /// `prefix_tokens` cached beside `batch` decodes at `context_tokens`. Default: the prefill pass
    /// plus the decode pass, which is how the replay's engine prices a mixed step (docs/bugs.md #16).
    fn mixed_s(
        &self,
        new_tokens: u32,
        prefix_tokens: u32,
        batch: u32,
        context_tokens: u32,
    ) -> Option<f64> {
        let prefill = if new_tokens > 0 {
            self.prefill_s(1, new_tokens, prefix_tokens)?
        } else {
            0.0
        };
        Some(prefill + self.decode_s(batch, context_tokens)?)
    }
}

/// A step model from grids.
///
/// Prefill is piecewise linear in the step's work, `batch × (new_tokens + prefix_weight ×
/// prefix_tokens)`, extrapolated past the last point with the last segment's slope. Decode is
/// bilinear over (batch, context), clamped to the grid's edges, so it under-predicts beyond
/// them. The default grids are placeholders of a plausible shape, not measurements: load an
/// AISimulate B300 grid (or use `AisimulateStepModel`) for any evaluation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableStepModel {
    /// Prefill work points (tokens), ascending.
    pub prefill_tokens: Vec<f64>,
    /// Seconds at each prefill point.
    pub prefill_s: Vec<f64>,
    /// A cached prefix token's cost relative to a new token.
    pub prefix_weight: f64,
    /// Decode batch sizes, ascending.
    pub decode_batch: Vec<f64>,
    /// Decode contexts (tokens per request), ascending.
    pub decode_context: Vec<f64>,
    /// Seconds per decode step, row-major: `decode_s[b * decode_context.len() + c]`.
    pub decode_s: Vec<f64>,
}

impl Default for TableStepModel {
    fn default() -> Self {
        let decode_batch = vec![1.0, 8.0, 32.0, 128.0];
        let decode_context = vec![1024.0, 8192.0, 65536.0, 131072.0];
        let decode_s = decode_batch
            .iter()
            .flat_map(|b| {
                decode_context
                    .iter()
                    .map(move |c| 0.007 + 4.0e-5 * b + 2.0e-10 * b * c)
            })
            .collect();
        Self {
            prefill_tokens: vec![0.0, 1024.0, 8192.0, 32768.0],
            prefill_s: vec![0.004, 0.03, 0.2, 0.85],
            prefix_weight: 0.02,
            decode_batch,
            decode_context,
            decode_s,
        }
    }
}

/// A step model that cannot be used, and why.
#[derive(Debug, Error, PartialEq)]
pub enum StepModelError {
    #[error("{0}")]
    Table(&'static str),
    #[error("AISimulate model: {0}")]
    Aisimulate(String),
}

fn ascending(points: &[f64]) -> bool {
    points.iter().all(|p| p.is_finite()) && points.windows(2).all(|w| w[0] < w[1])
}

impl TableStepModel {
    pub fn validate(&self) -> Result<(), StepModelError> {
        let finite = |values: &[f64]| values.iter().all(|v| v.is_finite() && *v >= 0.0);
        if self.prefill_tokens.is_empty() || self.prefill_tokens.len() != self.prefill_s.len() {
            return Err(StepModelError::Table(
                "prefill_tokens and prefill_s must be non-empty and the same length",
            ));
        }
        if !ascending(&self.prefill_tokens) || !finite(&self.prefill_s) {
            return Err(StepModelError::Table(
                "prefill_tokens must ascend and prefill_s be finite and non-negative",
            ));
        }
        if !(self.prefix_weight.is_finite() && self.prefix_weight >= 0.0) {
            return Err(StepModelError::Table("prefix_weight must be non-negative"));
        }
        if self.decode_batch.is_empty()
            || self.decode_context.is_empty()
            || self.decode_s.len() != self.decode_batch.len() * self.decode_context.len()
        {
            return Err(StepModelError::Table(
                "decode_s must hold decode_batch × decode_context values",
            ));
        }
        if !ascending(&self.decode_batch)
            || !ascending(&self.decode_context)
            || !finite(&self.decode_s)
        {
            return Err(StepModelError::Table(
                "decode axes must ascend and decode_s be finite and non-negative",
            ));
        }
        Ok(())
    }

    fn decode_at(&self, b: usize, c: usize) -> f64 {
        self.decode_s[b * self.decode_context.len() + c]
    }
}

/// The segment `x` falls in and its position in it, clamped to the axis.
fn bracket(axis: &[f64], x: f64) -> (usize, usize, f64) {
    let last = axis.len() - 1;
    if last == 0 || x <= axis[0] {
        return (0, 0, 0.0);
    }
    if x >= axis[last] {
        return (last, last, 0.0);
    }
    let hi = axis.partition_point(|p| *p <= x);
    let lo = hi - 1;
    (lo, hi, (x - axis[lo]) / (axis[hi] - axis[lo]))
}

impl StepModel for TableStepModel {
    fn prefill_s(&self, batch: u32, new_tokens: u32, prefix_tokens: u32) -> Option<f64> {
        let work = f64::from(batch)
            * (f64::from(new_tokens) + self.prefix_weight * f64::from(prefix_tokens));
        let (xs, ys) = (&self.prefill_tokens, &self.prefill_s);
        let last = xs.len() - 1;
        if last > 0 && work > xs[last] {
            let slope = (ys[last] - ys[last - 1]) / (xs[last] - xs[last - 1]);
            return Some(ys[last] + slope * (work - xs[last]));
        }
        let (lo, hi, t) = bracket(xs, work);
        Some(ys[lo] + (ys[hi] - ys[lo]) * t)
    }

    fn decode_s(&self, batch: u32, context_tokens: u32) -> Option<f64> {
        if batch == 0 {
            return Some(0.0);
        }
        let (b0, b1, tb) = bracket(&self.decode_batch, f64::from(batch));
        let (c0, c1, tc) = bracket(&self.decode_context, f64::from(context_tokens));
        let row = |b| self.decode_at(b, c0) + (self.decode_at(b, c1) - self.decode_at(b, c0)) * tc;
        Some(row(b0) + (row(b1) - row(b0)) * tb)
    }
}

/// k (`forecast.step_table_v2`): a step model from AISimulate grids with a 2-D prefill table.
///
/// Prefill is bilinear over (new tokens, cached prefix): the cost of a prefill chunk grows with
/// new tokens times prefix (attention), which a single `prefix_weight` cannot represent. A turn
/// longer than the last new-token point is priced chunk by chunk, each chunk over its own prefix;
/// `batch` prefills are priced as one request of `batch × new_tokens`. Decode is bilinear over
/// (batch, mean context). Both extrapolate linearly past each axis rather than clamp. A mixed step
/// is the two passes summed ([`StepModel::mixed_s`]'s default). Generated by
/// `scripts/validate/k/tables_v2.py` (`scripts/validate/step_tables/pro-tp4-v2.json`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableStepModelV2 {
    /// Where the grids come from (free text).
    #[serde(default, rename = "_source", skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// The engine's chunk (max batched tokens) the grids were made for (informational).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_tokens: Option<u32>,
    /// Prefill new-token points, ascending.
    pub prefill_new_tokens: Vec<f64>,
    /// Prefill cached-prefix points, ascending.
    pub prefill_prefix: Vec<f64>,
    /// Seconds, row-major: `prefill_s[n * prefill_prefix.len() + c]`.
    pub prefill_s: Vec<f64>,
    /// Decode batch sizes, ascending.
    pub decode_batch: Vec<f64>,
    /// Decode mean contexts (tokens per request), ascending.
    pub decode_context: Vec<f64>,
    /// Seconds, row-major: `decode_s[b * decode_context.len() + c]`.
    pub decode_s: Vec<f64>,
}

/// k `forecast.step_table_v2`: the key under which `stalled_itls` returns the candidate's own
/// (before, after) ITL (no class name starts with a control character).
const V2_CANDIDATE: &str = "\u{1}candidate";

/// k v2.2: the trailing window over which a worker's sent prefill gives its arrival rate.
pub const V2_ARRIVAL_WINDOW_S: f64 = 60.0;

/// k v2.2: computed prefill tokens per second over the trailing `window_s` before `now`, and the
/// tokens-weighted mean prefix of those turns, from `(sent at, computed tokens, prefix tokens)`.
pub fn v2_arrival_rate(
    arrivals: &VecDeque<(Time, u64, u64)>,
    now: Time,
    window_s: f64,
) -> (f64, f64) {
    let (mut tokens, mut prefix) = (0.0, 0.0);
    for &(at, t, p) in arrivals {
        let age = now.since_origin().as_secs_f64() - at.since_origin().as_secs_f64();
        if (0.0..=window_s).contains(&age) {
            tokens += t as f64;
            prefix += t as f64 * p as f64;
        }
    }
    let rate = if window_s > 0.0 {
        tokens / window_s
    } else {
        0.0
    };
    (rate, if tokens > 0.0 { prefix / tokens } else { 0.0 })
}

/// k v2.2: a decode step `d` stretched by prefill arriving at `rate` tokens per second, each
/// costing `u` seconds of step: `d / (1 - rate u)`, capped at `10 d`.
pub fn v2_arrival_itl(d: f64, rate: f64, u: f64) -> f64 {
    d / (1.0 - rate * u).max(0.1)
}

/// The segment `x` falls in and its position, extrapolating past either end of the axis.
fn bracket_extrapolated(axis: &[f64], x: f64) -> (usize, usize, f64) {
    let last = axis.len() - 1;
    if last == 0 {
        return (0, 0, 0.0);
    }
    if x <= axis[0] {
        return (0, 1, (x - axis[0]) / (axis[1] - axis[0]));
    }
    if x >= axis[last] {
        return (
            last - 1,
            last,
            (x - axis[last - 1]) / (axis[last] - axis[last - 1]),
        );
    }
    let hi = axis.partition_point(|p| *p <= x);
    let lo = hi - 1;
    (lo, hi, (x - axis[lo]) / (axis[hi] - axis[lo]))
}

impl TableStepModelV2 {
    pub fn validate(&self) -> Result<(), StepModelError> {
        let finite = |values: &[f64]| values.iter().all(|v| v.is_finite() && *v >= 0.0);
        let axes_ok = |a: &[f64]| a.len() >= 2 && ascending(a) && finite(a);
        if !axes_ok(&self.prefill_new_tokens) || !axes_ok(&self.prefill_prefix) {
            return Err(StepModelError::Table(
                "step_table_v2 prefill axes need two or more ascending non-negative points",
            ));
        }
        if self.prefill_s.len() != self.prefill_new_tokens.len() * self.prefill_prefix.len()
            || !finite(&self.prefill_s)
        {
            return Err(StepModelError::Table(
                "step_table_v2 prefill_s must hold new_tokens × prefix finite non-negative values",
            ));
        }
        if !axes_ok(&self.decode_batch) || !axes_ok(&self.decode_context) {
            return Err(StepModelError::Table(
                "step_table_v2 decode axes need two or more ascending non-negative points",
            ));
        }
        if self.decode_s.len() != self.decode_batch.len() * self.decode_context.len()
            || !finite(&self.decode_s)
        {
            return Err(StepModelError::Table(
                "step_table_v2 decode_s must hold batch × context finite non-negative values",
            ));
        }
        Ok(())
    }

    fn grid(xs: &[f64], ys: &[f64], z: &[f64], x: f64, y: f64) -> f64 {
        let (x0, x1, tx) = bracket_extrapolated(xs, x);
        let (y0, y1, ty) = bracket_extrapolated(ys, y);
        let at = |i: usize, j: usize| z[i * ys.len() + j];
        let row = |i: usize| at(i, y0) + (at(i, y1) - at(i, y0)) * ty;
        (row(x0) + (row(x1) - row(x0)) * tx).max(0.0)
    }

    fn chunk_s(&self, new_tokens: f64, prefix_tokens: f64) -> f64 {
        Self::grid(
            &self.prefill_new_tokens,
            &self.prefill_prefix,
            &self.prefill_s,
            new_tokens,
            prefix_tokens,
        )
    }
}

impl StepModel for TableStepModelV2 {
    fn prefill_s(&self, batch: u32, new_tokens: u32, prefix_tokens: u32) -> Option<f64> {
        let total = f64::from(batch) * f64::from(new_tokens);
        if total <= 0.0 {
            return Some(0.0);
        }
        let last = *self.prefill_new_tokens.last()?;
        let mut done = 0.0;
        let mut s = 0.0;
        while done < total {
            let c = (total - done).min(last);
            s += self.chunk_s(c, f64::from(prefix_tokens) + done);
            done += c;
        }
        Some(s)
    }

    fn decode_s(&self, batch: u32, context_tokens: u32) -> Option<f64> {
        if batch == 0 {
            return Some(0.0);
        }
        Some(Self::grid(
            &self.decode_batch,
            &self.decode_context,
            &self.decode_s,
            f64::from(batch),
            f64::from(context_tokens),
        ))
    }
}

#[cfg(feature = "aisimulate")]
pub use ais::AisimulateStepModel;

#[cfg(feature = "aisimulate")]
mod ais {
    //! AISimulate's forward-pass model as a [`StepModel`], built and queried the way the Python
    //! bindings' `ais_callback.rs` does for the Mocker (same metric packing, same readiness
    //! rule), so the forecast and the simulated engine price a step identically.

    use aisimulate_core::{
        ForwardPassMetrics, ForwardPassPerfModel, ForwardPassPerfModelConfig,
        ForwardPassPerfReadiness, ScheduledRequestMetrics,
    };

    use super::{StepModel, StepModelError};

    pub struct AisimulateStepModel {
        model: ForwardPassPerfModel,
        nextn: u32,
    }

    impl AisimulateStepModel {
        /// From a `ForwardPassPerfModelConfig` in JSON (the `--ais-config` the Mocker takes).
        pub fn from_json(config: &str) -> Result<Self, StepModelError> {
            let config: ForwardPassPerfModelConfig = serde_json::from_str(config)
                .map_err(|e| StepModelError::Aisimulate(format!("invalid config: {e}")))?;
            let nextn = match config.speculation.as_ref() {
                None => config.nextn,
                Some(spec) if spec.kind == "ngram" => spec
                    .params
                    .get("num_speculative_tokens")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|depth| u32::try_from(depth).ok())
                    .ok_or_else(|| {
                        StepModelError::Aisimulate(
                            "ngram num_speculative_tokens must be a u32".into(),
                        )
                    })?,
                Some(_) => {
                    return Err(StepModelError::Aisimulate(
                        "only ngram speculation or legacy nextn is supported".into(),
                    ));
                }
            };
            let model = ForwardPassPerfModel::best_available(config)
                .map_err(|e| StepModelError::Aisimulate(e.to_string()))?;
            if model.diagnostics().readiness != ForwardPassPerfReadiness::Ready {
                return Err(StepModelError::Aisimulate(
                    "estimator is not ready (no step data for this model and system)".into(),
                ));
            }
            Ok(Self { model, nextn })
        }

        fn estimate_s(&self, metrics: ForwardPassMetrics) -> Option<f64> {
            let ms = self
                .model
                .estimate_forward_pass_time_ms(&[metrics])
                .ok()??;
            (ms.is_finite() && ms >= 0.0).then_some(ms / 1000.0)
        }
    }

    impl StepModel for AisimulateStepModel {
        fn prefill_s(&self, batch: u32, new_tokens: u32, prefix_tokens: u32) -> Option<f64> {
            self.estimate_s(ForwardPassMetrics {
                scheduled_requests: ScheduledRequestMetrics {
                    num_prefill_requests: batch,
                    sum_prefill_tokens: batch.checked_mul(new_tokens)?,
                    sum_prefill_kv_tokens: batch.checked_mul(prefix_tokens)?,
                    ..Default::default()
                },
                ..Default::default()
            })
        }

        fn decode_s(&self, batch: u32, context_tokens: u32) -> Option<f64> {
            if batch == 0 {
                return Some(0.0);
            }
            let batch = batch.checked_mul(self.nextn + 1)?;
            self.estimate_s(ForwardPassMetrics {
                scheduled_requests: ScheduledRequestMetrics {
                    num_decode_requests: batch,
                    sum_decode_kv_tokens: batch.checked_mul(context_tokens)?,
                    ..Default::default()
                },
                ..Default::default()
            })
        }
    }
}

/// The forecast's parameters (`forecast` in the plugin's parameters).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ForecastConfig {
    /// Off by default: no observations are sent and no `LOAD` worker input is requested.
    pub enabled: bool,
    /// Observations buffered between queries; beyond it they are dropped and counted.
    pub channel_capacity: usize,
    /// Sessions tracked (least recently seen beyond this are dropped).
    pub session_capacity: usize,
    /// Requests in flight tracked (oldest beyond this are dropped).
    pub request_capacity: usize,
    /// Weight of the newest sample in each running mean, in (0, 1].
    pub alpha: f64,
    /// Least time between two worker load samples taken in `classify`.
    pub snapshot_interval_s: f64,
    /// Prefill chunk the engine schedules per step (when the worker does not advertise one).
    pub chunk_tokens: u32,
    /// The table step model. Ignored when `aisimulate_config` is set.
    pub step_table: TableStepModel,
    /// Path to an AISimulate `ForwardPassPerfModelConfig` (JSON). Needs the `aisimulate` feature.
    pub aisimulate_config: Option<String>,
    /// Planning window in seconds; 0 (the default) is off. When set, pricing a placement also
    /// counts the misses it adds to the turns expected back on that worker within the window:
    /// sessions whose last turn went there and whose next turn is forecast inside the window,
    /// queued first come first served behind the worker's prefill backlog, with and without the
    /// candidate ahead of them; each miss weighs `1 / (1 - attainment)` of its class, as the
    /// simulator's `forecast` and `evaluate` do (`web/sim.js`). Off, a placement is priced on the
    /// candidate's own TTFT and ITL and the running requests it pushes over, so routing cannot see
    /// that a worker whose queue already misses costs nothing more to load.
    pub window_s: f64,
    /// ks2b: a class-ordered backlog. On, a candidate queues behind the prefill of the turns of its
    /// own class and the classes ranked above it only (batch is out of a priority or standard
    /// turn's backlog; a batch turn queues behind everything), from the forecast's own record of
    /// the turns sent to the worker and not yet at their first token; a send also updates the
    /// worker's load sample until the next one; the decode batch leaves out the requests of
    /// classes ranked below the candidate, and comes from the engine's gauges when the host feeds
    /// them ([`Observation::Engine`]). As the simulator's forecast with `classQueue` and
    /// `batchPreempt` (`web/sim.js`). Off (the default): the latest load sample's whole queue.
    pub class_backlog: bool,
    /// ks2b: price a placement's harm to the decodes on the worker with the prefill stall. On,
    /// each class decoding there is judged by its mean ITL without and with the candidate on the
    /// stall model of [`Forecast::itl_ratio_for`] (the decode step stretched by the queued prefill
    /// it runs beside), so `what_if`'s misses include the ITL misses co-scheduled prefill causes,
    /// as the simulator's `forecast` does. Off (the default): the bare decode step at one more
    /// request.
    pub price_itl_stall: bool,
    /// k: the step table with a 2-D prefill (new tokens × cached prefix) and the mixed-step ITL
    /// ([`TableStepModelV2`]). Present, it replaces `step_table` as the step model (unless
    /// `aisimulate_config` is set), and `what_if`'s ITL terms use the mixed step: each class
    /// decoding on the worker and the candidate's own class get the mean ITL of their decode window
    /// when the queued prefill (and, after, the candidate's) runs in chunks beside them. Absent (the
    /// default): unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step_table_v2: Option<TableStepModelV2>,
    /// k v2.2 (with `step_table_v2`): the candidate's own ITL is the bare decode step stretched by
    /// the prefill expected to arrive during its decode, `d / (1 - R u)`: `R` the computed prefill
    /// tokens per second sent to the worker over the trailing [`V2_ARRIVAL_WINDOW_S`], `u` the
    /// step model's seconds per token of a full chunk over their mean prefix. Off (the default):
    /// unchanged.
    #[serde(default)]
    pub step_table_v2_arrival: bool,
}

impl Default for ForecastConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            channel_capacity: 65_536,
            session_capacity: 16_384,
            request_capacity: 65_536,
            alpha: 0.3,
            snapshot_interval_s: 1.0,
            chunk_tokens: 8192,
            step_table: TableStepModel::default(),
            aisimulate_config: None,
            window_s: 0.0,
            class_backlog: false,
            price_itl_stall: false,
            step_table_v2: None,
            step_table_v2_arrival: false,
        }
    }
}

impl ForecastConfig {
    pub fn validate(&self) -> Result<(), StepModelError> {
        if self.channel_capacity == 0 || self.session_capacity == 0 || self.request_capacity == 0 {
            return Err(StepModelError::Table(
                "forecast capacities must be positive",
            ));
        }
        if !(self.alpha > 0.0 && self.alpha <= 1.0) {
            return Err(StepModelError::Table("forecast.alpha must be in (0, 1]"));
        }
        if !(self.snapshot_interval_s.is_finite() && self.snapshot_interval_s >= 0.0) {
            return Err(StepModelError::Table(
                "forecast.snapshot_interval_s must be non-negative",
            ));
        }
        if !(self.window_s.is_finite() && self.window_s >= 0.0) {
            return Err(StepModelError::Table(
                "forecast.window_s must be non-negative",
            ));
        }
        if self.chunk_tokens == 0 {
            return Err(StepModelError::Table(
                "forecast.chunk_tokens must be positive",
            ));
        }
        if self.aisimulate_config.is_some() && !cfg!(feature = "aisimulate") {
            return Err(StepModelError::Aisimulate(
                "aisimulate_config needs the crate's `aisimulate` feature".into(),
            ));
        }
        if let Some(v2) = &self.step_table_v2 {
            v2.validate()?;
        }
        self.step_table.validate()
    }

    /// The configured step model: AISimulate when a config path is given, else the table.
    pub fn step_model(&self) -> Result<Arc<dyn StepModel>, StepModelError> {
        #[cfg(feature = "aisimulate")]
        if let Some(path) = &self.aisimulate_config {
            let json = std::fs::read_to_string(path)
                .map_err(|e| StepModelError::Aisimulate(format!("{path}: {e}")))?;
            return Ok(Arc::new(AisimulateStepModel::from_json(&json)?));
        }
        self.validate()?;
        if let Some(v2) = &self.step_table_v2 {
            return Ok(Arc::new(v2.clone()));
        }
        Ok(Arc::new(self.step_table.clone()))
    }
}

/// One worker rank's booked load, from the classifier's `LOAD` view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkerSample {
    pub worker: WorkerWithDpRank,
    pub active_prefill_tokens: u64,
    pub active_decode_blocks: u64,
    pub active_requests: u64,
    pub max_batched_tokens: Option<u64>,
}

/// One engine rank's own gauges (ks2b, `forecast.class_backlog`), from the host.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EngineGauge {
    pub worker: WorkerWithDpRank,
    pub running_requests: u64,
    pub waiting_requests: u64,
    /// KV blocks the running requests hold.
    pub active_blocks: u64,
    /// Tokens the waiting queue must reserve.
    pub waiting_tokens: u64,
}

/// Something the classifier saw, for the forecast.
#[derive(Clone, Debug, PartialEq)]
pub enum Observation {
    /// A turn was classified. `prompt_tokens` is its whole prompt.
    Arrived {
        request_id: String,
        session: String,
        class: String,
        at: Time,
        prompt_tokens: u64,
    },
    /// The router sent the turn to a worker.
    Sent {
        request_id: String,
        worker: WorkerWithDpRank,
        at: Time,
    },
    /// The first token came back.
    FirstToken {
        request_id: String,
        worker: WorkerWithDpRank,
        at: Time,
    },
    Completed {
        request_id: String,
        at: Time,
        context_tokens: Option<u64>,
    },
    Aborted {
        request_id: String,
    },
    /// A load sample of every worker.
    Workers {
        at: Time,
        rows: Vec<WorkerSample>,
    },
    /// One engine rank's own gauges (ks2b, read with `class_backlog`).
    Engine {
        at: Time,
        gauge: EngineGauge,
    },
    /// A client's own forecast of a session's next turn.
    Hint {
        session: String,
        next_turn_in_s: Option<f64>,
        next_new_tokens: Option<u64>,
        output_tokens: Option<u64>,
    },
}

/// The sending side, held by the classifier: never blocks, never takes the forecast's lock.
#[derive(Clone)]
pub struct ForecastFeed {
    tx: SyncSender<Observation>,
    dropped: Arc<AtomicU64>,
}

impl ForecastFeed {
    pub fn send(&self, observation: Observation) {
        if let Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) =
            self.tx.try_send(observation)
        {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

/// A running mean weighted toward recent samples.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Ewma(Option<f64>);

impl Ewma {
    fn add(&mut self, alpha: f64, x: f64) {
        self.0 = Some(match self.0 {
            Some(mean) => mean + alpha * (x - mean),
            None => x,
        });
    }
}

/// Predicted against actual, for one quantity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct ErrorStats {
    pub count: u64,
    pub sum_abs: f64,
    /// Sum of (predicted − actual): positive means the forecast runs high.
    pub sum_signed: f64,
}

impl ErrorStats {
    fn add(&mut self, predicted: f64, actual: f64) {
        self.count += 1;
        self.sum_abs += (predicted - actual).abs();
        self.sum_signed += predicted - actual;
    }

    pub fn mean_abs(&self) -> Option<f64> {
        (self.count > 0).then(|| self.sum_abs / self.count as f64)
    }

    pub fn bias(&self) -> Option<f64> {
        (self.count > 0).then(|| self.sum_signed / self.count as f64)
    }
}

/// One class's forecast errors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct ClassErrors {
    /// Seconds between the predicted and the actual arrival of a session's next turn.
    pub next_arrival_s: ErrorStats,
    /// Tokens a turn adds to its session's context.
    pub next_new_tokens: ErrorStats,
    pub output_tokens: ErrorStats,
    /// Seconds from send to first token.
    pub ttft_s: ErrorStats,
}

#[derive(Clone, Copy, Debug, Default)]
struct Means {
    gap_s: Ewma,
    new_tokens: Ewma,
    output_tokens: Ewma,
}

#[derive(Clone, Copy, Debug)]
struct NextTurn {
    at: Option<Time>,
    new_tokens: Option<f64>,
    output_tokens: Option<f64>,
}

#[derive(Debug)]
struct SessionStats {
    class: String,
    seen: (Time, u64),
    context_tokens: u64,
    last_done: Option<Time>,
    turns: u64,
    in_flight: u32,
    means: Means,
    hint: Option<NextTurn>,
    /// What [`Forecast::session`] said at the last completion, scored at the next arrival.
    predicted: Option<NextTurn>,
    /// The worker its last turn was sent to.
    worker: Option<WorkerWithDpRank>,
}

#[derive(Debug)]
struct RequestStats {
    session: String,
    class: String,
    prompt_tokens: u64,
    new_tokens: u64,
    predicted_output: Option<f64>,
    sent: Option<(Time, WorkerWithDpRank, Option<f64>)>,
    decoding_on: Option<WorkerWithDpRank>,
    /// With `class_backlog`: the worker whose queue holds its prefill (sent, no first token yet).
    queued_on: Option<WorkerWithDpRank>,
}

#[derive(Debug, Default)]
struct WorkerState {
    sample: Option<(Time, WorkerSample)>,
    /// Requests decoding on this worker, by class (from first token to completion).
    decoding: HashMap<String, u32>,
    /// Sessions whose last turn was sent to this worker (kept with `window_s` on).
    homed: BTreeSet<String>,
    /// With `class_backlog`: prefill tokens of the turns sent here and not yet at their first
    /// token, by class.
    queued: HashMap<String, u64>,
    /// The engine's latest gauges (`Observation::Engine`).
    gauge: Option<(Time, EngineGauge)>,
    /// k v2.2 (`step_table_v2_arrival`): turns sent here, `(at, computed tokens, prefix tokens)`,
    /// kept for the trailing [`V2_ARRIVAL_WINDOW_S`].
    arrivals: VecDeque<(Time, u64, u64)>,
}

/// A session's next turn as the forecast sees it.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionForecast {
    pub class: String,
    pub turns: u64,
    pub in_flight: bool,
    /// The context it holds now (its working set), in tokens.
    pub context_tokens: u64,
    /// `None` while a turn is running or before any history.
    pub next_turn_at: Option<Time>,
    pub next_new_tokens: Option<f64>,
    pub output_tokens: Option<f64>,
}

/// A worker's near future.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkerForecast {
    /// When the load sample behind this was taken.
    pub sampled_at: Time,
    /// Seconds until the queued prefill drains.
    pub prefill_backlog_s: f64,
    pub decode_batch: u64,
    /// Mean context of a decoding request, in tokens.
    pub decode_context_tokens: f64,
    /// (seconds from now, decode step seconds) at nine points across the horizon.
    pub decode_steps: Vec<(f64, f64)>,
}

/// A decision to price: admit a turn of `new_tokens` over `prefix_tokens` onto `worker`.
#[derive(Clone, Debug, PartialEq)]
pub struct WhatIf {
    pub class: String,
    pub worker: WorkerWithDpRank,
    pub new_tokens: u64,
    pub prefix_tokens: u64,
}

/// The decision's predicted consequences.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WhatIfOutcome {
    pub ttft_s: f64,
    /// The decode step before and after adding the turn.
    pub itl_before_s: f64,
    pub itl_after_s: f64,
    /// The candidate's own misses (TTFT, ITL): 0, 1 or 2.
    pub candidate_misses: u32,
    /// Running requests the larger batch pushes over their class ITL limit.
    pub pushed_over: u32,
    /// With `window_s`: the candidate's own misses and the running requests it pushes over, each
    /// weighing `1 / (1 - attainment)` of its class, plus the weighted misses it adds to the turns
    /// expected back on the worker within the window. `None` with the window off.
    pub weighted: Option<f64>,
}

impl WhatIfOutcome {
    pub fn misses(&self) -> u32 {
        self.candidate_misses + self.pushed_over
    }

    /// The misses routing prices: weighted with the planning window on, else counted.
    pub fn routing_misses(&self) -> f64 {
        self.weighted.unwrap_or_else(|| f64::from(self.misses()))
    }
}

/// f49d `DYNAMO_PREDICT_LOG`: one placement's price with its parts ([`Forecast::what_if_detail`]).
#[derive(Clone, Debug, PartialEq)]
pub struct PricedDetail {
    pub outcome: WhatIfOutcome,
    /// `forecast.window_s` (0: the planning window is off).
    pub window_s: f64,
    /// The decode batch the price stepped from (before the candidate).
    pub decode_batch: u64,
    /// Seconds of prefill the candidate queues behind.
    pub prefill_backlog_s: f64,
    /// Per class decoding on the worker: the mean ITL the price judged it by with the candidate.
    pub itl_by_class: Vec<(String, f64)>,
    /// Per class: requests decoding on the worker, as the forecast counts them.
    pub decoding_by_class: Vec<(String, u32)>,
}

struct State {
    rx: Receiver<Observation>,
    sessions: HashMap<String, SessionStats>,
    by_seen: BTreeMap<(Time, u64), String>,
    seq: u64,
    requests: HashMap<String, RequestStats>,
    request_order: VecDeque<String>,
    classes: HashMap<String, (Means, ClassErrors)>,
    workers: HashMap<WorkerWithDpRank, WorkerState>,
}

/// The forecast. Queries drain pending observations first.
pub struct Forecast {
    clock: Arc<dyn Clock>,
    config: ForecastConfig,
    model: Arc<dyn StepModel>,
    /// (TTFT limit, ITL limit) per class with a service level.
    limits: HashMap<String, (f64, f64)>,
    /// Weight of one miss per class with a service level: `1 / (1 - attainment)`.
    weights: HashMap<String, f64>,
    /// Each class's rank (lower is served first).
    ranks: HashMap<String, u32>,
    block_size: u32,
    dropped: Arc<AtomicU64>,
    state: Mutex<State>,
    /// f49i `admission.worker_step_budget`; `None` when absent.
    step_budget: Option<crate::step_budget::StepBudget>,
}

impl Forecast {
    /// A forecast and the feed that fills it.
    pub fn new(
        policy: &ServicePolicy,
        model: Arc<dyn StepModel>,
        clock: Arc<dyn Clock>,
        block_size: u32,
    ) -> (Arc<Self>, ForecastFeed) {
        let config = policy.forecast.clone();
        let (tx, rx) = sync_channel(config.channel_capacity.max(1));
        let dropped = Arc::new(AtomicU64::new(0));
        let limits = policy
            .classes
            .iter()
            .filter_map(|c| {
                c.service_level
                    .as_ref()
                    .map(|l| (c.name.clone(), (l.ttft_s, l.itl_s)))
            })
            .collect();
        let weights = policy
            .classes
            .iter()
            .filter_map(|c| {
                c.service_level
                    .as_ref()
                    .map(|l| (c.name.clone(), 1.0 / (1.0 - l.attainment).max(0.01)))
            })
            .collect();
        let ranks = policy
            .classes
            .iter()
            .map(|c| (c.name.clone(), c.rank))
            .collect();
        let forecast = Arc::new(Self {
            clock,
            config,
            model,
            limits,
            weights,
            ranks,
            block_size,
            dropped: Arc::clone(&dropped),
            step_budget: crate::step_budget::StepBudget::from_policy(policy),
            state: Mutex::new(State {
                rx,
                sessions: HashMap::new(),
                by_seen: BTreeMap::new(),
                seq: 0,
                requests: HashMap::new(),
                request_order: VecDeque::new(),
                classes: HashMap::new(),
                workers: HashMap::new(),
            }),
        });
        (forecast, ForecastFeed { tx, dropped })
    }

    pub fn config(&self) -> &ForecastConfig {
        &self.config
    }

    /// The forecast's clock (f49d: the prediction log's time).
    pub fn now(&self) -> Time {
        self.clock.now()
    }

    /// Observations dropped because the channel was full.
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Apply every pending observation.
    pub fn pump(&self) {
        let mut state = self.state.lock();
        while let Ok(observation) = state.rx.try_recv() {
            self.apply(&mut state, observation);
        }
    }

    pub fn session(&self, id: &str) -> Option<SessionForecast> {
        self.pump();
        let state = self.state.lock();
        let s = state.sessions.get(id)?;
        let next = Self::next_turn(&state, s);
        Some(SessionForecast {
            class: s.class.clone(),
            turns: s.turns,
            in_flight: s.in_flight > 0,
            context_tokens: s.context_tokens,
            next_turn_at: next.at,
            next_new_tokens: next.new_tokens,
            output_tokens: next.output_tokens,
        })
    }

    pub fn worker(&self, worker: WorkerWithDpRank, horizon: Duration) -> Option<WorkerForecast> {
        self.pump();
        let state = self.state.lock();
        let (sampled_at, sample) = state.workers.get(&worker)?.sample?;
        Some(self.worker_forecast(sampled_at, &sample, horizon))
    }

    /// The ITL limit a placement breaks (`forecast.itl_limit` on the session-placement policy):
    /// over the classes decoding on `worker` with an ITL limit, the largest ratio of predicted
    /// mean ITL to the limit once `new_tokens` of prefill (over `prefix_tokens` cached) join the
    /// worker's queued prefill. `None` when the worker has no sample; `Some(0.0)` when nothing
    /// with a limit decodes there.
    ///
    /// Mean ITL follows the prefill rate the decodes run beside (docs/test-results.md, "Step
    /// times and the priority ITL limit"): `d / (1 - R u)`, saturating at `d + u C` when every
    /// step carries a full chunk. `d` is the decode step at the worker's batch and context, `C`
    /// the worker's chunk, `u` the extra step seconds per prefill token at that chunk, and `R`
    /// the queued plus new prefill tokens spread over the class's expected decode time (its mean
    /// output tokens times `d`). Every number comes from the step model, the worker sample and
    /// the class's service level.
    pub fn itl_ratio(
        &self,
        worker: WorkerWithDpRank,
        new_tokens: u64,
        prefix_tokens: u64,
    ) -> Option<f64> {
        self.itl_ratio_for(worker, new_tokens, prefix_tokens, None)
    }

    /// [`Self::itl_ratio`] counting only the classes ranked above `class` when it is given
    /// (`forecast.itl_limit_lower_only`): a request's prefill is kept off the decodes of higher
    /// classes, never off its own class's or lower ones'.
    pub fn itl_ratio_for(
        &self,
        worker: WorkerWithDpRank,
        new_tokens: u64,
        prefix_tokens: u64,
        class: Option<&str>,
    ) -> Option<f64> {
        let rank_of = |c: &str| self.ranks.get(c).copied().unwrap_or(u32::MAX);
        let below = class.map(rank_of);
        self.pump();
        let state = self.state.lock();
        let w = state.workers.get(&worker)?;
        let (_, sample) = w.sample?;
        let tokens = |t: u64| u32::try_from(t).unwrap_or(u32::MAX);
        let chunk = sample
            .max_batched_tokens
            .filter(|c| *c > 0)
            .unwrap_or(u64::from(self.config.chunk_tokens))
            .max(1);
        let batch = sample.active_requests.max(1);
        let context =
            (sample.active_decode_blocks * u64::from(self.block_size)) as f64 / batch as f64;
        let d = self.model.decode_s(tokens(batch), context as u32)?;
        let full = self.model.prefill_s(1, tokens(chunk), 0)?;
        let base = self.model.prefill_s(1, 0, 0).unwrap_or(0.0);
        let u = (full - base).max(0.0) / chunk as f64;
        // The new turn's prefill work in new-token equivalents: its new tokens plus what its
        // cached prefix costs to attend over.
        let added = if u > 0.0 {
            let with = self
                .model
                .prefill_s(1, tokens(new_tokens), tokens(prefix_tokens))
                .unwrap_or(0.0);
            let without = self
                .model
                .prefill_s(1, tokens(new_tokens), 0)
                .unwrap_or(0.0);
            new_tokens as f64 + (with - without).max(0.0) / u
        } else {
            new_tokens as f64
        };
        let queued = sample.active_prefill_tokens as f64;
        let mut ratio: f64 = 0.0;
        for (class, n) in &w.decoding {
            if *n == 0 {
                continue;
            }
            if below.is_some_and(|r| rank_of(class) >= r) {
                continue;
            }
            let Some(&(_, limit)) = self.limits.get(class) else {
                continue;
            };
            if limit <= 0.0 {
                continue;
            }
            let osl = state
                .classes
                .get(class)
                .and_then(|(m, _)| m.output_tokens.0)
                .unwrap_or(1.0)
                .max(1.0);
            let window = osl * d;
            let r = (queued + added) / window.max(1e-9);
            let saturated = d + u * chunk as f64;
            let itl = if r * u < 1.0 {
                (d / (1.0 - r * u)).min(saturated)
            } else {
                saturated
            };
            ratio = ratio.max(itl / limit);
        }
        Some(ratio)
    }

    /// f49i `admission.worker_step_budget`, when configured.
    pub fn step_budget(&self) -> Option<&crate::step_budget::StepBudget> {
        self.step_budget.as_ref()
    }

    /// f49i: the step budget's check of `worker` for a turn of `class` computing `new_tokens` over
    /// `prefix_tokens` cached. `None` without the budget, a sample of the worker, or a step model
    /// answer.
    pub fn step_budget_check(
        &self,
        worker: WorkerWithDpRank,
        class: &str,
        new_tokens: u64,
        prefix_tokens: u64,
    ) -> Option<crate::step_budget::Check> {
        let budget = self.step_budget.as_ref()?;
        self.pump();
        let state = self.state.lock();
        self.step_budget_check_in(&state, budget, worker, class, new_tokens, prefix_tokens)
    }

    /// f49i: every sampled worker checked, in worker order, for the turn `request_id` of `class`
    /// with `isl` prompt tokens; on each worker the turn computes what the classifier noted for it
    /// ([`crate::step_budget::StepBudget::note_compute`]), else its whole prompt. A worker the step
    /// model cannot price counts as fitting. `None` without the budget or a sampled worker.
    pub fn step_budget_verdict(
        &self,
        request_id: &str,
        class: &str,
        isl: u64,
    ) -> Option<crate::step_budget::Verdict> {
        let budget = self.step_budget.as_ref()?;
        let noted = budget.computed(request_id);
        self.pump();
        let state = self.state.lock();
        let mut workers: Vec<WorkerWithDpRank> = state
            .workers
            .iter()
            .filter(|(_, w)| w.sample.is_some())
            .map(|(w, _)| *w)
            .collect();
        if workers.is_empty() {
            return None;
        }
        workers.sort();
        let checks = workers
            .into_iter()
            .map(|worker| {
                let new_tokens = noted
                    .as_ref()
                    .and_then(|per| per.iter().find(|(w, _)| *w == worker).map(|(_, t)| *t))
                    .unwrap_or(isl)
                    .min(isl);
                self.step_budget_check_in(
                    &state,
                    budget,
                    worker,
                    class,
                    new_tokens,
                    isl - new_tokens,
                )
                .unwrap_or(crate::step_budget::Check {
                    worker,
                    new_tokens,
                    itls: Vec::new(),
                    ratio: 0.0,
                })
            })
            .collect();
        Some(crate::step_budget::Verdict { checks })
    }

    /// f49i: the projected mean ITL of each listed class judged on `worker` (decoding there, or
    /// `class` itself) with the turn added, on K's mixed-step decode window
    /// ([`Self::stalled_itls`]): the batch is the running decodes plus the turn at their mean
    /// context (the engine's gauges with `class_backlog`, as [`Self::class_ordered`] reads them,
    /// else the sample; every class's decodes, which share the step); the window is the class's
    /// mean output tokens at that step; the prefill beside it is the worker's queued and in-flight
    /// prefill (with `class_backlog` the forecast's own record of every class's, else the sample's),
    /// plus the turn's `new_tokens`, plus the v2.2 arrival term (the trailing arrival rate times the
    /// window). The turn's own class when not decoding there yet decodes after the queue and its own
    /// prefill: only the arrival term runs beside it.
    fn step_budget_check_in(
        &self,
        state: &State,
        budget: &crate::step_budget::StepBudget,
        worker: WorkerWithDpRank,
        class: &str,
        new_tokens: u64,
        prefix_tokens: u64,
    ) -> Option<crate::step_budget::Check> {
        let w = state.workers.get(&worker)?;
        let (sampled_at, sample) = w.sample?;
        let mut base = self.worker_forecast(sampled_at, &sample, Duration::ZERO);
        if self.config.class_backlog
            && let Some((_, g)) = w.gauge
        {
            base.decode_batch = g.running_requests;
            base.decode_context_tokens = if g.running_requests > 0 {
                (g.active_blocks * u64::from(self.block_size)) as f64 / g.running_requests as f64
            } else {
                0.0
            };
        }
        let tokens = |t: u64| u32::try_from(t).unwrap_or(u32::MAX);
        let chunk = sample
            .max_batched_tokens
            .filter(|c| *c > 0)
            .unwrap_or(u64::from(self.config.chunk_tokens))
            .max(1);
        let queued = if self.config.class_backlog {
            w.queued.values().sum::<u64>() as f64
        } else {
            sample.active_prefill_tokens as f64
        };
        let batch = base.decode_batch + 1;
        let context = (base.decode_context_tokens * base.decode_batch as f64
            + (new_tokens + prefix_tokens) as f64)
            / batch as f64;
        let d = self.model.decode_s(tokens(batch), context as u32)?;
        let (rate, _) = v2_arrival_rate(&w.arrivals, self.clock.now(), V2_ARRIVAL_WINDOW_S);
        // K's `mixed`: `prefill` tokens in chunks of the room the decodes leave, `k` mixed steps of
        // `m` and the rest bare steps of `d`, over a window of `osl` steps.
        let room = (chunk as f64 - batch as f64).max(64.0);
        let mixed = |prefill: f64, osl: f64| {
            let steps = (prefill / room).ceil();
            if steps <= 0.0 {
                return d;
            }
            let per = (prefill / steps).max(1.0);
            let m = self
                .model
                .mixed_s(
                    per as u32,
                    tokens(prefix_tokens),
                    tokens(batch),
                    context as u32,
                )
                .unwrap_or(d);
            let k = steps.min(osl);
            (k * m + (osl - k) * d) / osl
        };
        let mut itls = Vec::new();
        for listed in budget.classes() {
            let decoding = w.decoding.get(listed).is_some_and(|n| *n > 0);
            if !decoding && listed != class {
                continue;
            }
            let osl = state
                .classes
                .get(listed)
                .and_then(|(m, _)| m.output_tokens.0)
                .unwrap_or(1.0)
                .max(1.0);
            let arriving = rate * osl * d;
            let prefill = if decoding {
                queued + new_tokens as f64 + arriving
            } else {
                arriving
            };
            itls.push((listed.clone(), mixed(prefill, osl)));
        }
        let ratio = budget.ratio(&itls);
        Some(crate::step_budget::Check {
            worker,
            new_tokens,
            itls,
            ratio,
        })
    }

    pub fn what_if(&self, decision: &WhatIf) -> Option<WhatIfOutcome> {
        self.pump();
        let state = self.state.lock();
        self.price(&state, decision, true)
    }

    /// [`Self::what_if`] with the running requests the larger batch pushes over their ITL limit
    /// counted per class (the harm a placement does to each class).
    pub fn what_if_by_class(
        &self,
        decision: &WhatIf,
    ) -> Option<(WhatIfOutcome, Vec<(String, u32)>)> {
        self.pump();
        let state = self.state.lock();
        let outcome = self.price(&state, decision, true)?;
        let worker = state.workers.get(&decision.worker)?;
        let stalled = if self.config.price_itl_stall || self.config.step_table_v2.is_some() {
            self.stalled_itls(&state, worker, decision)
        } else {
            None
        };
        let mut by_class: Vec<(String, u32)> = worker
            .decoding
            .iter()
            .filter_map(|(class, n)| {
                let (_, itl) = self.limits.get(class)?;
                let crossed = match stalled.as_ref().and_then(|m| m.get(class)) {
                    Some(&(before, after)) => before <= *itl && after > *itl,
                    None => outcome.itl_before_s <= *itl && outcome.itl_after_s > *itl,
                };
                (crossed && *n > 0).then(|| (class.clone(), *n))
            })
            .collect();
        by_class.sort();
        Some((outcome, by_class))
    }

    /// f49d `DYNAMO_PREDICT_LOG`: [`Self::what_if`] with the parts of the price, from the same
    /// `price` call on the same state (so `outcome` is exactly what `what_if` returns): the worker's
    /// decode batch and prefill backlog as the candidate's class sees them, and per class decoding
    /// there the mean ITL the price judges it by with the candidate (the stall model's with
    /// `price_itl_stall`, else the bare decode step at one more request). `None` exactly when
    /// `what_if` is.
    pub fn what_if_detail(&self, decision: &WhatIf) -> Option<PricedDetail> {
        self.pump();
        let state = self.state.lock();
        let outcome = self.price(&state, decision, true)?;
        let worker = state.workers.get(&decision.worker)?;
        let (sampled_at, sample) = worker.sample?;
        let base = self.worker_forecast(sampled_at, &sample, Duration::ZERO);
        let base = self.class_ordered(worker, &sample, base, &decision.class);
        let stalled = if self.config.price_itl_stall || self.config.step_table_v2.is_some() {
            self.stalled_itls(&state, worker, decision)
        } else {
            None
        };
        let mut itl_by_class: Vec<(String, f64)> = worker
            .decoding
            .iter()
            .filter(|(_, n)| **n > 0)
            .map(|(class, _)| {
                let after = stalled
                    .as_ref()
                    .and_then(|m| m.get(class))
                    .map_or(outcome.itl_after_s, |&(_, after)| after);
                (class.clone(), after)
            })
            .collect();
        itl_by_class.sort_by(|a, b| a.0.cmp(&b.0));
        let decoding_by_class = {
            let mut v: Vec<(String, u32)> = worker
                .decoding
                .iter()
                .filter(|(_, n)| **n > 0)
                .map(|(c, n)| (c.clone(), *n))
                .collect();
            v.sort();
            v
        };
        Some(PricedDetail {
            outcome,
            window_s: self.config.window_s,
            decode_batch: base.decode_batch,
            prefill_backlog_s: base.prefill_backlog_s,
            itl_by_class,
            decoding_by_class,
        })
    }

    /// The forecast-harm check on a new session (`admission.forecast_harm_new`, ks2b; `web/sim.js`
    /// `admitLobby` with `gateHarm`): the weighted misses placing `decision` adds to its own class
    /// and the classes ranked above it. Counted, each at `1 / (1 - attainment)` of its class: the
    /// running requests of those classes the placement pushes over their ITL limit
    /// ([`Self::what_if_by_class`]), the candidate's own ITL miss, and with `window_s` the turns
    /// of those classes expected back on the worker within the window. The candidate's own
    /// cold-TTFT miss is not held against it: its miss counts only for ITL with its TTFT met (the
    /// simulator adds one miss for TTFT or ITL and takes one back for TTFT). `None` when the
    /// worker has no sample.
    pub fn new_session_harm(&self, decision: &WhatIf) -> Option<f64> {
        let (outcome, by_class) = self.what_if_by_class(decision)?;
        let rank_of = |c: &str| self.ranks.get(c).copied().unwrap_or(u32::MAX);
        let weight = |c: &str| self.weights.get(c).copied().unwrap_or(0.0);
        let rank = rank_of(&decision.class);
        let own = self
            .limits
            .get(&decision.class)
            .map_or(0.0, |&(ttft, itl)| {
                if outcome.itl_after_s > itl && outcome.ttft_s <= ttft {
                    weight(&decision.class)
                } else {
                    0.0
                }
            });
        let pushed: f64 = by_class
            .iter()
            .filter(|(c, _)| rank_of(c) <= rank)
            .map(|(c, n)| weight(c) * f64::from(*n))
            .sum();
        let window = if self.config.window_s > 0.0 {
            let tokens = |t: u64| u32::try_from(t).unwrap_or(u32::MAX);
            let own_s = self
                .model
                .prefill_s(
                    1,
                    tokens(decision.new_tokens),
                    tokens(decision.prefix_tokens),
                )
                .unwrap_or(0.0);
            let state = self.state.lock();
            state.workers.get(&decision.worker).map_or(0.0, |w| {
                // The backlog the candidate queues behind, as `price` saw it.
                let backlog_s = (outcome.ttft_s - own_s).max(0.0);
                self.window_added(&state, w, backlog_s, own_s, Some(rank))
            })
        } else {
            0.0
        };
        Some(own + pushed + window)
    }

    /// Per-class prediction errors so far.
    pub fn errors(&self) -> BTreeMap<String, ClassErrors> {
        self.pump();
        let state = self.state.lock();
        state
            .classes
            .iter()
            .map(|(class, (_, errors))| (class.clone(), *errors))
            .collect()
    }

    fn next_turn(state: &State, s: &SessionStats) -> NextTurn {
        let class = state.classes.get(&s.class).map(|(m, _)| *m);
        let pick = |own: Ewma, of_class: fn(&Means) -> Ewma| {
            own.0.or_else(|| class.as_ref().and_then(|m| of_class(m).0))
        };
        let mut next = NextTurn {
            at: None,
            new_tokens: pick(s.means.new_tokens, |m| m.new_tokens),
            output_tokens: pick(s.means.output_tokens, |m| m.output_tokens),
        };
        if s.in_flight == 0
            && let Some(done) = s.last_done
            && let Some(gap) = pick(s.means.gap_s, |m| m.gap_s)
        {
            next.at = Some(done.after(Duration::from_secs_f64(gap.max(0.0))));
        }
        if let Some(hint) = s.hint {
            next.at = hint.at.or(next.at);
            next.new_tokens = hint.new_tokens.or(next.new_tokens);
            next.output_tokens = hint.output_tokens.or(next.output_tokens);
        }
        next
    }

    fn worker_forecast(
        &self,
        sampled_at: Time,
        sample: &WorkerSample,
        horizon: Duration,
    ) -> WorkerForecast {
        let chunk = sample
            .max_batched_tokens
            .filter(|c| *c > 0)
            .unwrap_or(u64::from(self.config.chunk_tokens));
        let queued = sample.active_prefill_tokens;
        let tokens = |t: u64| u32::try_from(t).unwrap_or(u32::MAX);
        let full = (queued / chunk) as f64;
        let prefill_backlog_s = full * self.model.prefill_s(1, tokens(chunk), 0).unwrap_or(0.0)
            + if !queued.is_multiple_of(chunk) {
                self.model
                    .prefill_s(1, tokens(queued % chunk), 0)
                    .unwrap_or(0.0)
            } else {
                0.0
            };
        let batch = sample.active_requests;
        let context = if batch > 0 {
            (sample.active_decode_blocks * u64::from(self.block_size)) as f64 / batch as f64
        } else {
            0.0
        };
        let batch32 = tokens(batch);
        let step0 = self.model.decode_s(batch32, context as u32).unwrap_or(0.0);
        let horizon_s = horizon.as_secs_f64();
        let decode_steps = (0..=8)
            .map(|k| {
                let t = horizon_s * f64::from(k) / 8.0;
                // One token per request per step; decode starts after the prefill backlog.
                let steps = if step0 > 0.0 {
                    ((t - prefill_backlog_s).max(0.0) / step0).floor()
                } else {
                    0.0
                };
                let step = self
                    .model
                    .decode_s(batch32, (context + steps) as u32)
                    .unwrap_or(step0);
                (t, step)
            })
            .collect();
        WorkerForecast {
            sampled_at,
            prefill_backlog_s,
            decode_batch: batch,
            decode_context_tokens: context,
            decode_steps,
        }
    }

    /// Price `decision` on the latest load sample; `window` adds the planning-window term when
    /// `window_s` is set (routing asks for it; the TTFT error score does not).
    fn price(&self, state: &State, decision: &WhatIf, window: bool) -> Option<WhatIfOutcome> {
        let worker = state.workers.get(&decision.worker)?;
        let (sampled_at, sample) = worker.sample?;
        // Priced on the latest sample as it stands; `WorkerForecast::sampled_at` gives its age.
        let base = self.worker_forecast(sampled_at, &sample, Duration::ZERO);
        let base = self.class_ordered(worker, &sample, base, &decision.class);
        let tokens = |t: u64| u32::try_from(t).unwrap_or(u32::MAX);
        let own = self.model.prefill_s(
            1,
            tokens(decision.new_tokens),
            tokens(decision.prefix_tokens),
        )?;
        let batch = base.decode_batch;
        let itl_before_s = self
            .model
            .decode_s(tokens(batch), base.decode_context_tokens as u32)?;
        let context_after = (base.decode_context_tokens * batch as f64
            + (decision.new_tokens + decision.prefix_tokens) as f64)
            / (batch + 1) as f64;
        let itl_after_s = self
            .model
            .decode_s(tokens(batch + 1), context_after as u32)?;
        let ttft_s = base.prefill_backlog_s + own;
        // ks2b `forecast.price_itl_stall`: each decoding class's ITL with the prefill stall,
        // without and with the candidate (`None` off: the bare decode steps). k
        // `forecast.step_table_v2`: the mixed-step ITL, which also prices the candidate's own.
        let stalled = if self.config.price_itl_stall || self.config.step_table_v2.is_some() {
            self.stalled_itls(state, worker, decision)
        } else {
            None
        };
        let (itl_before_s, itl_after_s) = match (
            self.config.step_table_v2.is_some(),
            stalled.as_ref().and_then(|m| m.get(V2_CANDIDATE)),
        ) {
            (true, Some(&(before, after))) => (before, after),
            _ => (itl_before_s, itl_after_s),
        };
        let candidate_misses = self.limits.get(&decision.class).map_or(0, |(ttft, itl)| {
            u32::from(ttft_s > *ttft) + u32::from(itl_after_s > *itl)
        });
        let crossed = |class: &str, itl: f64| match stalled.as_ref().and_then(|m| m.get(class)) {
            Some(&(before, after)) => before <= itl && after > itl,
            None => itl_before_s <= itl && itl_after_s > itl,
        };
        let pushed_over = worker
            .decoding
            .iter()
            .filter_map(|(class, n)| {
                let (_, itl) = self.limits.get(class)?;
                crossed(class, *itl).then_some(*n)
            })
            .sum();
        let weighted = (window && self.config.window_s > 0.0).then(|| {
            let w = |class: &str| self.weights.get(class).copied().unwrap_or(0.0);
            let own_w = w(&decision.class) * f64::from(candidate_misses);
            let pushed_w: f64 = worker
                .decoding
                .iter()
                .filter_map(|(class, n)| {
                    let (_, itl) = self.limits.get(class)?;
                    crossed(class, *itl).then(|| w(class) * f64::from(*n))
                })
                .sum();
            own_w + pushed_w + self.window_added(state, worker, base.prefill_backlog_s, own, None)
        });
        Some(WhatIfOutcome {
            ttft_s,
            itl_before_s,
            itl_after_s,
            candidate_misses,
            pushed_over,
            weighted,
        })
    }

    /// ks2b `forecast.price_itl_stall`: per class decoding on `worker` with an ITL limit, its mean
    /// ITL without and with `decision`'s prefill on the stall model of [`Self::itl_ratio_for`]
    /// (`d / (1 - R u)`, saturating at `d + u C`, `R` the prefill tokens over the class's decode
    /// window): without, `d` is the decode step at the worker's batch and `R` its queued prefill;
    /// with, `d` is the step at one more request and `R` adds the candidate's prefill. The
    /// simulator's `forecast` stretches each decode window over the prefill-busy intervals it
    /// overlaps (`web/sim.js`). `None` without a sample or a step model answer.
    fn stalled_itls(
        &self,
        state: &State,
        worker: &WorkerState,
        decision: &WhatIf,
    ) -> Option<HashMap<String, (f64, f64)>> {
        let (sampled_at, sample) = worker.sample?;
        let base = self.worker_forecast(sampled_at, &sample, Duration::ZERO);
        let base = self.class_ordered(worker, &sample, base, &decision.class);
        let tokens = |t: u64| u32::try_from(t).unwrap_or(u32::MAX);
        let chunk = sample
            .max_batched_tokens
            .filter(|c| *c > 0)
            .unwrap_or(u64::from(self.config.chunk_tokens))
            .max(1);
        let full = self.model.prefill_s(1, tokens(chunk), 0)?;
        let empty = self.model.prefill_s(1, 0, 0).unwrap_or(0.0);
        let u = (full - empty).max(0.0) / chunk as f64;
        // The candidate's prefill in new-token equivalents, as `itl_ratio_for` counts it.
        let added = if u > 0.0 {
            let with = self
                .model
                .prefill_s(
                    1,
                    tokens(decision.new_tokens),
                    tokens(decision.prefix_tokens),
                )
                .unwrap_or(0.0);
            let without = self
                .model
                .prefill_s(1, tokens(decision.new_tokens), 0)
                .unwrap_or(0.0);
            decision.new_tokens as f64 + (with - without).max(0.0) / u
        } else {
            decision.new_tokens as f64
        };
        let queued_tokens = if self.config.class_backlog {
            self.queued_ahead(worker, &decision.class) as f64
        } else {
            sample.active_prefill_tokens as f64
        };
        let batch = base.decode_batch;
        let d_before = self
            .model
            .decode_s(tokens(batch.max(1)), base.decode_context_tokens as u32)?;
        let context_after = (base.decode_context_tokens * batch as f64
            + (decision.new_tokens + decision.prefix_tokens) as f64)
            / (batch + 1) as f64;
        let d_after = self
            .model
            .decode_s(tokens(batch + 1), context_after as u32)?;
        let v2 = self.config.step_table_v2.is_some();
        let context_before = base.decode_context_tokens;
        // k `forecast.step_table_v2`: the decode window of `osl` tokens at `b` decodes of mean
        // context `ctx` while `prefill` tokens (over the candidate's prefix, the shape of this
        // traffic) run in chunks of the room the decodes leave: `k` mixed steps of `m`, the rest
        // bare steps of `d`, and the window's mean.
        let mixed = |b: u64, ctx: f64, d: f64, prefill: f64, osl: f64| {
            let room = (chunk as f64 - b as f64).max(64.0);
            let steps = (prefill / room).ceil();
            if steps <= 0.0 {
                return d;
            }
            let per = (prefill / steps).max(1.0);
            let m = self
                .model
                .mixed_s(
                    per as u32,
                    tokens(decision.prefix_tokens),
                    tokens(b.max(1)),
                    ctx as u32,
                )
                .unwrap_or(d);
            let k = steps.min(osl);
            (k * m + (osl - k) * d) / osl
        };
        let stall = |d: f64, prefill: f64, osl: f64| {
            let saturated = d + u * chunk as f64;
            let r = prefill / (osl * d).max(1e-9);
            if r * u < 1.0 {
                (d / (1.0 - r * u)).min(saturated)
            } else {
                saturated
            }
        };
        let mut out = HashMap::new();
        if v2 && self.limits.contains_key(&decision.class) {
            // The candidate's own decode window: its prefill (and the queue ahead of it) ends before
            // its first token, so it decodes beside the prefill that keeps arriving, for which the
            // queue now stands in. Before: the bare step at the current batch.
            let osl = state
                .classes
                .get(&decision.class)
                .and_then(|(m, _)| m.output_tokens.0)
                .unwrap_or(1.0)
                .max(1.0);
            let value = if self.config.step_table_v2_arrival {
                let (rate, prefix) =
                    v2_arrival_rate(&worker.arrivals, self.clock.now(), V2_ARRIVAL_WINDOW_S);
                let u = self
                    .model
                    .prefill_s(1, tokens(chunk), prefix as u32)
                    .map_or(0.0, |s| s / chunk as f64);
                (
                    v2_arrival_itl(d_before, rate, u),
                    v2_arrival_itl(d_after, rate, u),
                )
            } else {
                (
                    d_before,
                    mixed(batch + 1, context_after, d_after, queued_tokens, osl),
                )
            };
            out.insert(V2_CANDIDATE.to_owned(), value);
        }
        for (class, n) in &worker.decoding {
            if *n == 0 || !self.limits.contains_key(class) {
                continue;
            }
            let osl = state
                .classes
                .get(class)
                .and_then(|(m, _)| m.output_tokens.0)
                .unwrap_or(1.0)
                .max(1.0);
            let value = if v2 {
                (
                    mixed(batch.max(1), context_before, d_before, queued_tokens, osl),
                    mixed(
                        batch + 1,
                        context_after,
                        d_after,
                        queued_tokens + decision.new_tokens as f64,
                        osl,
                    ),
                )
            } else {
                (
                    stall(d_before, queued_tokens, osl),
                    stall(d_after, queued_tokens + added, osl),
                )
            };
            out.insert(class.clone(), value);
        }
        Some(out)
    }

    /// The weighted misses a placement adds to the turns expected back on `worker` within
    /// `window_s`: idle sessions homed there with a next turn forecast in the window, served first
    /// come first served after `backlog_s`, with and without the candidate's `own_s` of prefill
    /// ahead of them (`web/sim.js` `evaluate`: the forecast with the candidate less the forecast
    /// without it). With `max_rank` only the misses of classes ranked at or above it count; the
    /// other turns still queue (`new_session_harm`).
    fn window_added(
        &self,
        state: &State,
        worker: &WorkerState,
        backlog_s: f64,
        own_s: f64,
        max_rank: Option<u32>,
    ) -> f64 {
        let now = self.clock.now();
        let tokens = |t: f64| u32::try_from(t.max(0.0).round() as u64).unwrap_or(u32::MAX);
        // (seconds from now it arrives, prefill seconds, TTFT limit, weight)
        let mut items: Vec<(f64, f64, f64, f64)> = Vec::new();
        for id in &worker.homed {
            let Some(s) = state.sessions.get(id) else {
                continue;
            };
            if s.in_flight > 0 {
                continue;
            }
            let Some(&(ttft, _)) = self.limits.get(&s.class) else {
                continue;
            };
            let next = Self::next_turn(state, s);
            let Some(at) = next.at else {
                continue;
            };
            let a = if at >= now {
                at.since(now).as_secs_f64()
            } else {
                0.0
            };
            if a > self.config.window_s {
                continue;
            }
            let new = next.new_tokens.unwrap_or(0.0);
            let dur = self
                .model
                .prefill_s(1, tokens(new), tokens(s.context_tokens as f64))
                .unwrap_or(0.0);
            let below =
                max_rank.is_some_and(|r| self.ranks.get(&s.class).copied().unwrap_or(u32::MAX) > r);
            let weight = if below {
                0.0
            } else {
                self.weights.get(&s.class).copied().unwrap_or(0.0)
            };
            items.push((a, dur, ttft, weight));
        }
        if items.is_empty() {
            return 0.0;
        }
        items.sort_by(|x, y| x.0.total_cmp(&y.0));
        let missed = |start: f64| {
            let mut c = start;
            let mut m = 0.0;
            for &(a, dur, ttft, weight) in &items {
                c = c.max(a) + dur;
                if c - a > ttft {
                    m += weight;
                }
            }
            m
        };
        (missed(backlog_s + own_s) - missed(backlog_s)).max(0.0)
    }

    fn touch(&self, state: &mut State, session: &str, class: &str, at: Time) {
        state.seq += 1;
        let key = (at, state.seq);
        if let Some(s) = state.sessions.get_mut(session) {
            let old = std::mem::replace(&mut s.seen, key);
            s.class = class.to_owned();
            state.by_seen.remove(&old);
            state.by_seen.insert(key, session.to_owned());
            return;
        }
        while state.sessions.len() >= self.config.session_capacity {
            let Some((_, oldest)) = state.by_seen.pop_first() else {
                break;
            };
            if let Some(gone) = state.sessions.remove(&oldest)
                && let Some(w) = gone.worker
                && let Some(ws) = state.workers.get_mut(&w)
            {
                ws.homed.remove(&oldest);
            }
        }
        state.by_seen.insert(key, session.to_owned());
        state.sessions.insert(
            session.to_owned(),
            SessionStats {
                class: class.to_owned(),
                seen: key,
                context_tokens: 0,
                last_done: None,
                turns: 0,
                in_flight: 0,
                means: Means::default(),
                hint: None,
                predicted: None,
                worker: None,
            },
        );
    }

    fn apply(&self, state: &mut State, observation: Observation) {
        let alpha = self.config.alpha;
        match observation {
            Observation::Arrived {
                request_id,
                session,
                class,
                at,
                prompt_tokens,
            } => {
                self.touch(state, &session, &class, at);
                let class_entry = state.classes.entry(class.clone()).or_default();
                let s = state.sessions.get_mut(&session).expect("touched");
                let new_tokens = prompt_tokens.saturating_sub(s.context_tokens);
                if let Some(predicted) = s.predicted.take() {
                    let errors = &mut class_entry.1;
                    if let (Some(p), Some(done)) = (predicted.at, s.last_done) {
                        errors.next_arrival_s.add(
                            p.as_secs_f64(),
                            done.as_secs_f64() + at.since(done).as_secs_f64(),
                        );
                    }
                    if let Some(p) = predicted.new_tokens {
                        errors.next_new_tokens.add(p, new_tokens as f64);
                    }
                }
                if let Some(done) = s.last_done {
                    let gap = at.since(done).as_secs_f64();
                    s.means.gap_s.add(alpha, gap);
                    class_entry.0.gap_s.add(alpha, gap);
                }
                if s.turns > 0 {
                    s.means.new_tokens.add(alpha, new_tokens as f64);
                    class_entry.0.new_tokens.add(alpha, new_tokens as f64);
                }
                s.turns += 1;
                s.in_flight += 1;
                s.hint = None;
                let predicted_output = s.means.output_tokens.0.or(class_entry.0.output_tokens.0);
                if state.requests.len() >= self.config.request_capacity
                    && let Some(oldest) = state.request_order.pop_front()
                    && let Some(gone) = state.requests.remove(&oldest)
                    && let Some(w) = gone.queued_on
                {
                    Self::release_queued(state, w, &gone.class, gone.new_tokens);
                }
                state.request_order.push_back(request_id.clone());
                state.requests.insert(
                    request_id,
                    RequestStats {
                        session,
                        class,
                        prompt_tokens,
                        new_tokens,
                        predicted_output,
                        sent: None,
                        decoding_on: None,
                        queued_on: None,
                    },
                );
            }
            Observation::Sent {
                request_id,
                worker,
                at,
            } => {
                let Some(r) = state.requests.get(&request_id) else {
                    return;
                };
                let decision = WhatIf {
                    class: r.class.clone(),
                    worker,
                    new_tokens: r.new_tokens,
                    prefix_tokens: r.prompt_tokens.saturating_sub(r.new_tokens),
                };
                let predicted = self.price(state, &decision, false).map(|o| o.ttft_s);
                let session = r.session.clone();
                // f49i: the step budget's arrival term reads the same trailing window.
                if self.config.step_table_v2_arrival || self.step_budget.is_some() {
                    let a = &mut state.workers.entry(worker).or_default().arrivals;
                    a.push_back((at, decision.new_tokens, decision.prefix_tokens));
                    while a.front().is_some_and(|&(t, _, _)| {
                        at.since_origin().as_secs_f64() - t.since_origin().as_secs_f64()
                            > V2_ARRIVAL_WINDOW_S
                    }) {
                        a.pop_front();
                    }
                }
                if let Some(r) = state.requests.get_mut(&request_id) {
                    r.sent = Some((at, worker, predicted));
                }
                // ks2b `class_backlog`: the turn is on the worker's queue now, not at the next sample.
                if self.config.class_backlog {
                    Self::queue_on(state, &request_id, worker);
                }
                if self.config.window_s > 0.0
                    && let Some(s) = state.sessions.get_mut(&session)
                {
                    let old = s.worker.replace(worker);
                    if old != Some(worker) {
                        if let Some(o) = old
                            && let Some(ws) = state.workers.get_mut(&o)
                        {
                            ws.homed.remove(&session);
                        }
                        state
                            .workers
                            .entry(worker)
                            .or_default()
                            .homed
                            .insert(session);
                    }
                }
            }
            Observation::FirstToken {
                request_id,
                worker,
                at,
            } => {
                let Some(r) = state.requests.get_mut(&request_id) else {
                    return;
                };
                if r.decoding_on.is_some() {
                    return;
                }
                r.decoding_on = Some(worker);
                let queued = r.queued_on.take().map(|w| (w, r.new_tokens));
                let class = r.class.clone();
                if let Some((sent_at, _, Some(predicted))) = r.sent {
                    let actual = at.since(sent_at).as_secs_f64();
                    state
                        .classes
                        .entry(class.clone())
                        .or_default()
                        .1
                        .ttft_s
                        .add(predicted, actual);
                }
                if let Some((w, tokens)) = queued {
                    Self::release_queued(state, w, &class, tokens);
                }
                *state
                    .workers
                    .entry(worker)
                    .or_default()
                    .decoding
                    .entry(class)
                    .or_default() += 1;
            }
            Observation::Completed {
                request_id,
                at,
                context_tokens,
            } => {
                let Some(r) = self.finish(state, &request_id) else {
                    return;
                };
                let class_entry = state.classes.entry(r.class.clone()).or_default();
                if let Some(context) = context_tokens {
                    let output = context.saturating_sub(r.prompt_tokens) as f64;
                    if let Some(p) = r.predicted_output {
                        class_entry.1.output_tokens.add(p, output);
                    }
                    class_entry.0.output_tokens.add(alpha, output);
                }
                let Some(s) = state.sessions.get_mut(&r.session) else {
                    return;
                };
                s.in_flight = s.in_flight.saturating_sub(1);
                s.last_done = Some(at);
                if let Some(context) = context_tokens {
                    s.context_tokens = context;
                    s.means
                        .output_tokens
                        .add(alpha, context.saturating_sub(r.prompt_tokens) as f64);
                }
                let s = state.sessions.get(&r.session).expect("present");
                let next = Self::next_turn(state, s);
                if let Some(s) = state.sessions.get_mut(&r.session) {
                    s.predicted = Some(next);
                }
            }
            Observation::Aborted { request_id } => {
                if let Some(r) = self.finish(state, &request_id)
                    && let Some(s) = state.sessions.get_mut(&r.session)
                {
                    s.in_flight = s.in_flight.saturating_sub(1);
                }
            }
            Observation::Workers { at, rows } => {
                for row in rows {
                    state.workers.entry(row.worker).or_default().sample = Some((at, row));
                }
            }
            Observation::Engine { at, gauge } => {
                state.workers.entry(gauge.worker).or_default().gauge = Some((at, gauge));
            }
            Observation::Hint {
                session,
                next_turn_in_s,
                next_new_tokens,
                output_tokens,
            } => {
                let now = self.clock.now();
                if let Some(s) = state.sessions.get_mut(&session) {
                    s.hint = Some(NextTurn {
                        at: next_turn_in_s
                            .filter(|s| s.is_finite() && *s >= 0.0)
                            .map(|s| now.after(Duration::from_secs_f64(s))),
                        new_tokens: next_new_tokens.map(|t| t as f64),
                        output_tokens: output_tokens.map(|t| t as f64),
                    });
                }
            }
        }
    }

    /// ks2b `class_backlog`: put `request_id`'s prefill on `worker`'s queue, by class, and count it
    /// in the worker's load sample until the next one.
    fn queue_on(state: &mut State, request_id: &str, worker: WorkerWithDpRank) {
        let Some(r) = state.requests.get_mut(request_id) else {
            return;
        };
        let old = r.queued_on.replace(worker);
        let (class, tokens) = (r.class.clone(), r.new_tokens);
        if let Some(o) = old {
            Self::release_queued(state, o, &class, tokens);
        }
        let ws = state.workers.entry(worker).or_default();
        *ws.queued.entry(class).or_default() += tokens;
        if let Some((_, sample)) = ws.sample.as_mut() {
            sample.active_prefill_tokens += tokens;
            sample.active_requests += 1;
        }
    }

    /// Take `tokens` of `class` off `worker`'s queue (`class_backlog`).
    fn release_queued(state: &mut State, worker: WorkerWithDpRank, class: &str, tokens: u64) {
        if let Some(ws) = state.workers.get_mut(&worker)
            && let Some(q) = ws.queued.get_mut(class)
        {
            *q = q.saturating_sub(tokens);
            if *q == 0 {
                ws.queued.remove(class);
            }
        }
    }

    /// Prefill tokens queued on `worker` (`class_backlog`) of `class` and the classes ranked
    /// above it: what a turn of `class` waits behind when prefill is served in class order.
    fn queued_ahead(&self, worker: &WorkerState, class: &str) -> u64 {
        let rank_of = |c: &str| self.ranks.get(c).copied().unwrap_or(u32::MAX);
        let rank = rank_of(class);
        worker
            .queued
            .iter()
            .filter(|(c, _)| rank_of(c) <= rank)
            .map(|(_, t)| *t)
            .sum()
    }

    /// Seconds to prefill `queued` tokens in `chunk`-token steps (as [`Self::worker_forecast`]).
    fn prefill_seconds(&self, queued: u64, chunk: u64) -> f64 {
        let tokens = |t: u64| u32::try_from(t).unwrap_or(u32::MAX);
        let chunk = chunk.max(1);
        (queued / chunk) as f64 * self.model.prefill_s(1, tokens(chunk), 0).unwrap_or(0.0)
            + if queued.is_multiple_of(chunk) {
                0.0
            } else {
                self.model
                    .prefill_s(1, tokens(queued % chunk), 0)
                    .unwrap_or(0.0)
            }
    }

    /// ks2b `forecast.class_backlog`: `base` as a turn of `class` sees it (`web/sim.js` `forecast`
    /// with `classQueue`, and `evaluate`'s `noBatch` under `batchPreempt`): the prefill backlog is
    /// the queue of `class` and the classes above it ([`Self::queued_ahead`]); the decode batch
    /// and context are the engine's running requests and their blocks when its gauges are in,
    /// less the requests of classes ranked below `class` decoding there. Off: `base` as it is.
    fn class_ordered(
        &self,
        worker: &WorkerState,
        sample: &WorkerSample,
        mut base: WorkerForecast,
        class: &str,
    ) -> WorkerForecast {
        if !self.config.class_backlog {
            return base;
        }
        let rank_of = |c: &str| self.ranks.get(c).copied().unwrap_or(u32::MAX);
        let rank = rank_of(class);
        let chunk = sample
            .max_batched_tokens
            .filter(|c| *c > 0)
            .unwrap_or(u64::from(self.config.chunk_tokens));
        base.prefill_backlog_s = self.prefill_seconds(self.queued_ahead(worker, class), chunk);
        if let Some((_, g)) = worker.gauge {
            gauge_debug_priced(base.decode_batch, g.running_requests);
            base.decode_batch = g.running_requests;
            base.decode_context_tokens = if g.running_requests > 0 {
                (g.active_blocks * u64::from(self.block_size)) as f64 / g.running_requests as f64
            } else {
                0.0
            };
        }
        let below: u64 = worker
            .decoding
            .iter()
            .filter(|(c, _)| rank_of(c) > rank)
            .map(|(_, n)| u64::from(*n))
            .sum();
        base.decode_batch = base.decode_batch.saturating_sub(below);
        base
    }

    /// Remove a finished request and release its decode slot.
    fn finish(&self, state: &mut State, request_id: &str) -> Option<RequestStats> {
        let r = state.requests.remove(request_id)?;
        if let Some(w) = r.queued_on {
            Self::release_queued(state, w, &r.class, r.new_tokens);
        }
        if let Some(worker) = r.decoding_on
            && let Some(w) = state.workers.get_mut(&worker)
            && let Some(n) = w.decoding.get_mut(&r.class)
        {
            *n = n.saturating_sub(1);
            if *n == 0 {
                w.decoding.remove(&r.class);
            }
        }
        Some(r)
    }
}

/// f49b: stderr counters for the engine-gauge path in the plugin (`DYNAMO_REPLAY_GAUGE_DEBUG=1`); off by default.
fn gauge_debug_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DYNAMO_REPLAY_GAUGE_DEBUG").is_ok_and(|v| v == "1"))
}

/// A gauge event handed to the forecast (counted per worker, printed every 1,000).
pub(crate) fn gauge_debug_received(worker: WorkerWithDpRank) {
    if !gauge_debug_on() {
        return;
    }
    static COUNTS: std::sync::OnceLock<
        std::sync::Mutex<(u64, std::collections::BTreeMap<String, u64>)>,
    > = std::sync::OnceLock::new();
    let mut c = COUNTS.get_or_init(Default::default).lock().unwrap();
    c.0 += 1;
    *c.1.entry(format!("{worker:?}")).or_insert(0) += 1;
    if c.0.is_multiple_of(1000) {
        eprintln!(
            "gauge-debug plugin: received {} gauge events, per worker {:?}",
            c.0, c.1
        );
    }
}

/// A priced candidate that read a gauge: every 100th, the router-booked decode batch next to the gauge's.
fn gauge_debug_priced(router_batch: u64, engine_running: u64) {
    if !gauge_debug_on() {
        return;
    }
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if n.is_multiple_of(100) {
        eprintln!(
            "gauge-debug plugin: priced {n} with gauges; router-booked decode batch {router_batch}, engine running_requests {engine_running}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::ManualClock;

    fn secs(s: f64) -> Time {
        Time::from_secs_f64(s)
    }

    fn setup(config: ForecastConfig) -> (Arc<Forecast>, ForecastFeed, Arc<ManualClock>) {
        let clock = Arc::new(ManualClock::new());
        let policy = ServicePolicy {
            forecast: ForecastConfig {
                enabled: true,
                ..config
            },
            ..Default::default()
        };
        let model = policy.forecast.step_model().unwrap();
        let (forecast, feed) = Forecast::new(&policy, model, clock.clone(), 16);
        (forecast, feed, clock)
    }

    fn arrive(feed: &ForecastFeed, id: &str, session: &str, at: f64, prompt: u64) {
        feed.send(Observation::Arrived {
            request_id: id.into(),
            session: session.into(),
            class: "standard".into(),
            at: secs(at),
            prompt_tokens: prompt,
        });
    }

    fn complete(feed: &ForecastFeed, id: &str, at: f64, context: u64) {
        feed.send(Observation::Completed {
            request_id: id.into(),
            at: secs(at),
            context_tokens: Some(context),
        });
    }

    #[test]
    fn table_interpolates_extrapolates_and_validates() {
        let table = TableStepModel::default();
        table.validate().unwrap();
        // On the grid and between points.
        assert!((table.prefill_s(1, 1024, 0).unwrap() - 0.03).abs() < 1e-12);
        let mid = table.prefill_s(1, 4608, 0).unwrap();
        assert!((mid - (0.03 + 0.17 * 0.5)).abs() < 1e-9);
        // Past the last point, the last slope continues.
        let far = table.prefill_s(2, 32768, 0).unwrap();
        assert!((far - (0.85 + (0.65 / 24576.0) * 32768.0)).abs() < 1e-9);
        // A cached prefix costs prefix_weight of a new token.
        assert!(table.prefill_s(1, 1024, 10_000).unwrap() > 0.03);
        // Decode: grid corner, clamped beyond, monotone in batch and context.
        let corner = 0.007 + 4.0e-5 * 8.0 + 2.0e-10 * 8.0 * 8192.0;
        assert!((table.decode_s(8, 8192).unwrap() - corner).abs() < 1e-12);
        assert_eq!(table.decode_s(1000, 1 << 20), table.decode_s(128, 131072));
        assert!(table.decode_s(16, 4096).unwrap() < table.decode_s(16, 40_000).unwrap());
        assert!(table.decode_s(4, 4096).unwrap() < table.decode_s(64, 4096).unwrap());
        assert_eq!(table.decode_s(0, 4096), Some(0.0));

        let mut bad = TableStepModel::default();
        bad.decode_context.swap(0, 1);
        assert!(bad.validate().is_err());
        let mut bad = TableStepModel::default();
        bad.prefill_s.pop();
        assert!(bad.validate().is_err());
    }

    #[test]
    fn session_forecast_learns_gaps_sizes_and_outputs_and_scores_itself() {
        let (forecast, feed, _clock) = setup(ForecastConfig {
            alpha: 0.5,
            ..Default::default()
        });
        // Turn 1: 1000-token prompt, 200 out; done at 5.
        arrive(&feed, "r1", "s", 0.0, 1000);
        let first = forecast.session("s").unwrap();
        assert!(first.in_flight && first.next_turn_at.is_none());
        complete(&feed, "r1", 5.0, 1200);
        // No gap history yet: no arrival forecast, but the output is known.
        let after1 = forecast.session("s").unwrap();
        assert_eq!(after1.next_turn_at, None);
        assert_eq!(after1.output_tokens, Some(200.0));
        assert_eq!(after1.context_tokens, 1200);

        // Turn 2 after a 10 s gap adds 300 tokens; 100 out.
        arrive(&feed, "r2", "s", 15.0, 1500);
        complete(&feed, "r2", 20.0, 1600);
        let after2 = forecast.session("s").unwrap();
        assert_eq!(after2.next_turn_at, Some(secs(30.0)));
        assert_eq!(after2.next_new_tokens, Some(300.0));
        assert_eq!(after2.output_tokens, Some(150.0));

        // Turn 3 comes after 20 s (predicted 10) and adds 500 (predicted 300).
        arrive(&feed, "r3", "s", 40.0, 2100);
        let errors = forecast.errors()["standard"];
        assert_eq!(errors.next_arrival_s.count, 1);
        assert!((errors.next_arrival_s.mean_abs().unwrap() - 10.0).abs() < 1e-9);
        assert!((errors.next_arrival_s.bias().unwrap() + 10.0).abs() < 1e-9);
        assert_eq!(errors.next_new_tokens.mean_abs(), Some(200.0));
        // Output predicted 200 for turn 2, actual 100.
        assert_eq!(errors.output_tokens.count, 1);
        assert_eq!(errors.output_tokens.bias(), Some(100.0));

        // A new session borrows its class's means.
        arrive(&feed, "q1", "t", 41.0, 800);
        complete(&feed, "q1", 42.0, 900);
        let borrowed = forecast.session("t").unwrap();
        assert_eq!(borrowed.next_turn_at, Some(secs(42.0 + 15.0)));
        assert_eq!(borrowed.next_new_tokens, Some(400.0));

        // A hint overrides the next turn only.
        feed.send(Observation::Hint {
            session: "t".into(),
            next_turn_in_s: Some(3.0),
            next_new_tokens: Some(50),
            output_tokens: None,
        });
        let hinted = forecast.session("t").unwrap();
        assert_eq!(hinted.next_turn_at, Some(secs(3.0)));
        assert_eq!(hinted.next_new_tokens, Some(50.0));
    }

    #[test]
    fn planning_window_counts_misses_added_to_turns_expected_back() {
        let run = |window_s: f64| {
            let (forecast, feed, clock) = setup(ForecastConfig {
                window_s,
                ..Default::default()
            });
            let (home, other) = (WorkerWithDpRank::new(1, 0), WorkerWithDpRank::new(2, 0));
            // Session s (standard: TTFT 8 s, attainment 0.7) runs two turns on `home`, 10 s apart;
            // its next turn is due at 30.
            for (id, at, prompt, done, context) in
                [("r1", 0.0, 1000, 5.0, 1200), ("r2", 15.0, 1500, 20.0, 1600)]
            {
                arrive(&feed, id, "s", at, prompt);
                feed.send(Observation::Sent {
                    request_id: id.into(),
                    worker: home,
                    at: secs(at),
                });
                complete(&feed, id, done, context);
            }
            clock.set(secs(25.0));
            let idle = |worker| WorkerSample {
                worker,
                active_prefill_tokens: 0,
                active_decode_blocks: 0,
                active_requests: 0,
                max_batched_tokens: None,
            };
            feed.send(Observation::Workers {
                at: secs(25.0),
                rows: vec![idle(home), idle(other)],
            });
            // A standard turn so long that s, due 5 s later, would wait past its 8 s limit.
            let on = |worker| {
                forecast
                    .what_if(&WhatIf {
                        class: "standard".into(),
                        worker,
                        new_tokens: 600_000,
                        prefix_tokens: 0,
                    })
                    .unwrap()
            };
            (on(home), on(other))
        };
        let (home, other) = run(60.0);
        assert!(home.ttft_s > 13.0);
        assert_eq!(home.candidate_misses, 1);
        let w = 1.0 / 0.3;
        // Its own TTFT miss on either worker; on home also the miss it adds to s.
        assert!((other.weighted.unwrap() - w).abs() < 1e-9);
        assert!((home.weighted.unwrap() - 2.0 * w).abs() < 1e-9);
        assert!((home.routing_misses() - 2.0 * w).abs() < 1e-9);
        // Due after the window: nothing added.
        let (home, _) = run(4.0);
        assert!((home.weighted.unwrap() - w).abs() < 1e-9);
        // Off: counted misses, no weights, no window.
        let (home, other) = run(0.0);
        assert_eq!(home.weighted, None);
        assert_eq!(home.routing_misses(), 1.0);
        assert_eq!(other.routing_misses(), 1.0);
    }

    #[test]
    fn worker_forecast_and_what_if_price_backlog_batch_and_pushed_over() {
        let (forecast, feed, _clock) = setup(ForecastConfig::default());
        let w = WorkerWithDpRank::new(7, 0);
        feed.send(Observation::Workers {
            at: secs(1.0),
            rows: vec![WorkerSample {
                worker: w,
                active_prefill_tokens: 8192 * 2 + 1024,
                active_decode_blocks: 8 * 512,
                active_requests: 8,
                max_batched_tokens: None,
            }],
        });
        let table = TableStepModel::default();
        let f = forecast.worker(w, Duration::from_secs(4)).unwrap();
        assert_eq!(f.sampled_at, secs(1.0));
        assert!((f.prefill_backlog_s - (2.0 * 0.2 + 0.03)).abs() < 1e-9);
        assert_eq!(f.decode_batch, 8);
        assert_eq!(f.decode_context_tokens, 8192.0);
        assert_eq!(f.decode_steps.len(), 9);
        assert_eq!(f.decode_steps[0], (0.0, table.decode_s(8, 8192).unwrap()));
        assert!(f.decode_steps[8].1 > f.decode_steps[0].1);
        assert!(
            forecast
                .worker(WorkerWithDpRank::new(9, 0), Duration::ZERO)
                .is_none()
        );

        // Three priority requests decoding (ITL limit 50 ms) and the step near it.
        for i in 0..3 {
            let id = format!("p{i}");
            feed.send(Observation::Arrived {
                request_id: id.clone(),
                session: id.clone(),
                class: "priority".into(),
                at: secs(1.0),
                prompt_tokens: 100,
            });
            feed.send(Observation::FirstToken {
                request_id: id,
                worker: w,
                at: secs(1.5),
            });
        }
        let decision = WhatIf {
            class: "standard".into(),
            worker: w,
            new_tokens: 4096,
            prefix_tokens: 0,
        };
        let o = forecast.what_if(&decision).unwrap();
        let expected_ttft = f.prefill_backlog_s + table.prefill_s(1, 4096, 0).unwrap();
        assert!((o.ttft_s - expected_ttft).abs() < 1e-9);
        assert!(o.itl_after_s > o.itl_before_s);
        assert_eq!(o.candidate_misses, 0);
        assert_eq!(o.pushed_over, 0);

        // A step table whose decode crosses 50 ms between batch 8 and 9 pushes the three over.
        let tight = TableStepModel {
            decode_batch: vec![8.0, 9.0],
            decode_context: vec![0.0, 1.0e6],
            decode_s: vec![0.049, 0.049, 0.051, 0.051],
            ..Default::default()
        };
        let (forecast, feed, _clock) = setup(ForecastConfig {
            step_table: tight,
            ..Default::default()
        });
        feed.send(Observation::Workers {
            at: secs(1.0),
            rows: vec![WorkerSample {
                worker: w,
                active_prefill_tokens: 0,
                active_decode_blocks: 8 * 512,
                active_requests: 8,
                max_batched_tokens: Some(4096),
            }],
        });
        for i in 0..3 {
            let id = format!("p{i}");
            feed.send(Observation::Arrived {
                request_id: id.clone(),
                session: id.clone(),
                class: "priority".into(),
                at: secs(1.0),
                prompt_tokens: 100,
            });
            feed.send(Observation::FirstToken {
                request_id: id,
                worker: w,
                at: secs(1.5),
            });
        }
        let o = forecast
            .what_if(&WhatIf {
                class: "priority".into(),
                ..decision.clone()
            })
            .unwrap();
        assert_eq!(o.pushed_over, 3);
        assert_eq!(o.candidate_misses, 1, "its own ITL is over 50 ms too");
        assert_eq!(o.misses(), 4);
        // Completing one frees its slot.
        complete(&feed, "p0", 2.0, 150);
        let o = forecast
            .what_if(&WhatIf {
                class: "priority".into(),
                ..decision
            })
            .unwrap();
        assert_eq!(o.pushed_over, 2);
    }

    /// ks2b `admission.forecast_harm_new`: the weighted misses a new session's placement adds to
    /// its class and the classes above it, without its own cold TTFT.
    #[test]
    fn new_session_harm_counts_its_class_and_above_but_not_its_own_cold_ttft() {
        // Decode crosses priority's 50 ms ITL limit between batch 8 and 9.
        let tight = TableStepModel {
            decode_batch: vec![8.0, 9.0],
            decode_context: vec![0.0, 1.0e6],
            decode_s: vec![0.049, 0.049, 0.051, 0.051],
            ..Default::default()
        };
        let (forecast, feed, _clock) = setup(ForecastConfig {
            step_table: tight,
            ..Default::default()
        });
        let w = WorkerWithDpRank::new(7, 0);
        feed.send(Observation::Workers {
            at: secs(1.0),
            rows: vec![WorkerSample {
                worker: w,
                active_prefill_tokens: 0,
                active_decode_blocks: 8 * 512,
                active_requests: 8,
                max_batched_tokens: Some(4096),
            }],
        });
        for i in 0..3 {
            let id = format!("p{i}");
            feed.send(Observation::Arrived {
                request_id: id.clone(),
                session: id.clone(),
                class: "priority".into(),
                at: secs(1.0),
                prompt_tokens: 100,
            });
            feed.send(Observation::FirstToken {
                request_id: id,
                worker: w,
                at: secs(1.5),
            });
        }
        let harm = |class: &str, new_tokens| {
            forecast
                .new_session_harm(&WhatIf {
                    class: class.into(),
                    worker: w,
                    new_tokens,
                    prefix_tokens: 0,
                })
                .unwrap()
        };
        let priority_w = 1.0 / 0.1;
        // Standard pushes three priority decodes over (3 x 10); its own 51 ms is under its 100 ms.
        assert!((harm("standard", 4096) - 3.0 * priority_w).abs() < 1e-9);
        // Priority: the same three plus its own ITL miss.
        assert!((harm("priority", 4096) - 4.0 * priority_w).abs() < 1e-9);
        // Its cold TTFT is not held against it: a prompt far over priority's 3 s TTFT adds the
        // same (its ITL miss is dropped with its TTFT miss, as the simulator's -1 does).
        assert!((harm("priority", 400_000) - 3.0 * priority_w).abs() < 1e-9);
        // No sample for the worker: unknown.
        assert!(
            forecast
                .new_session_harm(&WhatIf {
                    class: "priority".into(),
                    worker: WorkerWithDpRank::new(9, 0),
                    new_tokens: 1,
                    prefix_tokens: 0,
                })
                .is_none()
        );

        // Standard decodes are below a priority candidate: their misses do not count for it.
        let loose = TableStepModel {
            decode_batch: vec![8.0, 9.0],
            decode_context: vec![0.0, 1.0e6],
            decode_s: vec![0.04, 0.04, 0.2, 0.2],
            ..Default::default()
        };
        let (forecast, feed, _clock) = setup(ForecastConfig {
            step_table: loose,
            ..Default::default()
        });
        feed.send(Observation::Workers {
            at: secs(1.0),
            rows: vec![WorkerSample {
                worker: w,
                active_prefill_tokens: 0,
                active_decode_blocks: 8 * 512,
                active_requests: 8,
                max_batched_tokens: Some(4096),
            }],
        });
        for i in 0..3 {
            let id = format!("s{i}");
            arrive(&feed, &id, &id, 1.0, 100);
            feed.send(Observation::FirstToken {
                request_id: id,
                worker: w,
                at: secs(1.5),
            });
        }
        let h = forecast
            .new_session_harm(&WhatIf {
                class: "priority".into(),
                worker: w,
                new_tokens: 1024,
                prefix_tokens: 0,
            })
            .unwrap();
        // Only its own ITL miss (200 ms over 50 ms).
        assert!((h - priority_w).abs() < 1e-9);
        let standard_w = 1.0 / 0.3;
        let h = forecast
            .new_session_harm(&WhatIf {
                class: "standard".into(),
                worker: w,
                new_tokens: 1024,
                prefix_tokens: 0,
            })
            .unwrap();
        assert!((h - 4.0 * standard_w).abs() < 1e-9);
    }

    /// ks2b `forecast.price_itl_stall`: the candidate's prefill stalls priority decodes over
    /// their 50 ms limit; the bare decode step (off) does not see it.
    #[test]
    fn price_itl_stall_counts_decodes_the_candidates_prefill_stalls_over() {
        let w = WorkerWithDpRank::new(7, 0);
        let run = |stall: bool, queued: u64, new_tokens: u64| {
            let (forecast, feed, _clock) = setup(ForecastConfig {
                price_itl_stall: stall,
                window_s: 60.0,
                ..Default::default()
            });
            // Priority's mean output is 1,000 tokens (one finished turn).
            feed.send(Observation::Arrived {
                request_id: "h".into(),
                session: "h".into(),
                class: "priority".into(),
                at: secs(0.0),
                prompt_tokens: 100,
            });
            complete(&feed, "h", 0.5, 1100);
            feed.send(Observation::Workers {
                at: secs(1.0),
                rows: vec![WorkerSample {
                    worker: w,
                    active_prefill_tokens: queued,
                    active_decode_blocks: 8 * 512,
                    active_requests: 8,
                    max_batched_tokens: None,
                }],
            });
            for i in 0..3 {
                let id = format!("p{i}");
                feed.send(Observation::Arrived {
                    request_id: id.clone(),
                    session: id.clone(),
                    class: "priority".into(),
                    at: secs(1.0),
                    prompt_tokens: 100,
                });
                feed.send(Observation::FirstToken {
                    request_id: id,
                    worker: w,
                    at: secs(1.5),
                });
            }
            let decision = WhatIf {
                class: "standard".into(),
                worker: w,
                new_tokens,
                prefix_tokens: 0,
            };
            let outcome = forecast.what_if(&decision).unwrap();
            let by_class = forecast.what_if_by_class(&decision).unwrap().1;
            (outcome, by_class)
        };
        let table = TableStepModel::default();
        let (priority_w, standard_w) = (1.0 / (1.0 - 0.9), 1.0 / (1.0 - 0.7));
        // 400k tokens of prefill beside a 1,000-token decode window saturate it (about 0.2 s a
        // token); the bare step at batch 9 stays near 7 ms.
        let (off, off_by) = run(false, 0, 400_000);
        assert_eq!(off.itl_before_s, table.decode_s(8, 8192).unwrap());
        assert!(off.itl_after_s < 0.05);
        assert_eq!((off.pushed_over, off.candidate_misses), (0, 1));
        assert!(off_by.is_empty());
        assert!((off.weighted.unwrap() - standard_w).abs() < 1e-9);
        let (on, on_by) = run(true, 0, 400_000);
        // The candidate's own figures are the bare ones either way.
        let own = |o: &WhatIfOutcome| (o.ttft_s, o.itl_before_s, o.itl_after_s, o.candidate_misses);
        assert_eq!(own(&on), own(&off));
        assert_eq!(on.pushed_over, 3);
        assert_eq!(on_by, vec![("priority".to_owned(), 3)]);
        assert!((on.weighted.unwrap() - (standard_w + 3.0 * priority_w)).abs() < 1e-9);
        // Already stalled over without the candidate: it pushes nobody over.
        let (on, _) = run(true, 400_000, 400_000);
        assert_eq!(on.pushed_over, 0);
        // A small prefill barely stretches the step.
        let (on, on_by) = run(true, 0, 1024);
        assert_eq!(on.pushed_over, 0);
        assert!(on_by.is_empty());
        // Off, the queue changes only the TTFT, as before.
        let (off_q, _) = run(false, 400_000, 1024);
        let (off_0, _) = run(false, 0, 1024);
        assert_eq!(
            (off_q.itl_before_s, off_q.itl_after_s, off_q.pushed_over),
            (off_0.itl_before_s, off_0.itl_after_s, off_0.pushed_over)
        );
        assert!(off_q.ttft_s > off_0.ttft_s);
    }

    /// f49d: `what_if_detail` is `what_if`'s own price with its parts, and with `price_itl_stall`
    /// every decoding class with an ITL limit is judged on the stall model: standard decodes
    /// (100 ms) are pushed over and counted beside priority's, each weighted `1 / (1 - attainment)`.
    #[test]
    fn what_if_detail_is_what_ifs_price_and_the_stall_covers_every_class_with_a_limit() {
        let w = WorkerWithDpRank::new(5, 0);
        let run = |stall: bool| {
            let (forecast, feed, _clock) = setup(ForecastConfig {
                price_itl_stall: stall,
                window_s: 60.0,
                ..Default::default()
            });
            for class in ["priority", "standard"] {
                // Each class's mean output is 1,000 tokens (one finished turn).
                let id = format!("h-{class}");
                feed.send(Observation::Arrived {
                    request_id: id.clone(),
                    session: id.clone(),
                    class: class.into(),
                    at: secs(0.0),
                    prompt_tokens: 100,
                });
                complete(&feed, &id, 0.5, 1100);
            }
            feed.send(Observation::Workers {
                at: secs(1.0),
                rows: vec![WorkerSample {
                    worker: w,
                    active_prefill_tokens: 2048,
                    active_decode_blocks: 8 * 512,
                    active_requests: 8,
                    max_batched_tokens: None,
                }],
            });
            for (i, class) in ["priority", "priority", "standard", "standard", "standard"]
                .iter()
                .enumerate()
            {
                let id = format!("d{i}");
                feed.send(Observation::Arrived {
                    request_id: id.clone(),
                    session: id.clone(),
                    class: (*class).into(),
                    at: secs(1.0),
                    prompt_tokens: 100,
                });
                feed.send(Observation::FirstToken {
                    request_id: id,
                    worker: w,
                    at: secs(1.5),
                });
            }
            let decision = WhatIf {
                class: "standard".into(),
                worker: w,
                new_tokens: 400_000,
                prefix_tokens: 0,
            };
            let detail = forecast.what_if_detail(&decision).unwrap();
            let outcome = forecast.what_if(&decision).unwrap();
            let by_class = forecast.what_if_by_class(&decision).unwrap().1;
            let wf = forecast.worker(w, Duration::ZERO).unwrap();
            (detail, outcome, by_class, wf)
        };
        let (priority_w, standard_w) = (1.0 / (1.0 - 0.9), 1.0 / (1.0 - 0.7));
        for stall in [false, true] {
            let (detail, outcome, by_class, wf) = run(stall);
            assert_eq!(detail.outcome, outcome);
            assert_eq!(detail.window_s, 60.0);
            assert_eq!(detail.decode_batch, wf.decode_batch);
            assert_eq!(detail.prefill_backlog_s, wf.prefill_backlog_s);
            assert_eq!(
                detail.decoding_by_class,
                vec![("priority".to_owned(), 2), ("standard".to_owned(), 3)]
            );
            let classes: Vec<&str> = detail.itl_by_class.iter().map(|x| x.0.as_str()).collect();
            assert_eq!(classes, vec!["priority", "standard"]);
            if stall {
                // Both classes stalled over their limits: 2 priority and 3 standard decodes.
                assert!(detail.itl_by_class.iter().all(|(_, itl)| *itl > 0.1));
                assert_eq!(
                    by_class,
                    vec![("priority".to_owned(), 2), ("standard".to_owned(), 3)]
                );
                assert_eq!(outcome.pushed_over, 5);
                let own = standard_w * f64::from(outcome.candidate_misses);
                assert!(
                    (outcome.weighted.unwrap() - (own + 2.0 * priority_w + 3.0 * standard_w)).abs()
                        < 1e-9
                );
            } else {
                // Off: every class is judged by the bare step at one more request.
                assert!(
                    detail
                        .itl_by_class
                        .iter()
                        .all(|(_, itl)| *itl == outcome.itl_after_s)
                );
                assert!(by_class.is_empty());
                assert_eq!(outcome.pushed_over, 0);
            }
        }
    }

    /// ks2b `forecast.class_backlog`: the backlog by class, updated on send, and the engine's
    /// gauges for the decode batch.
    #[test]
    fn class_backlog_orders_the_queue_by_class_updates_on_send_and_reads_engine_gauges() {
        let w = WorkerWithDpRank::new(3, 0);
        let run = |on: bool| {
            let (forecast, feed, _clock) = setup(ForecastConfig {
                class_backlog: on,
                ..Default::default()
            });
            feed.send(Observation::Workers {
                at: secs(0.0),
                rows: vec![WorkerSample {
                    worker: w,
                    active_prefill_tokens: 0,
                    active_decode_blocks: 0,
                    active_requests: 0,
                    max_batched_tokens: None,
                }],
            });
            // Sent after the sample and not yet at their first token: batch 50k, standard 10k.
            for (id, class, prompt) in [("b", "batch", 50_000), ("s", "standard", 10_000)] {
                feed.send(Observation::Arrived {
                    request_id: id.into(),
                    session: id.into(),
                    class: class.into(),
                    at: secs(1.0),
                    prompt_tokens: prompt,
                });
                feed.send(Observation::Sent {
                    request_id: id.into(),
                    worker: w,
                    at: secs(1.0),
                });
            }
            (forecast, feed)
        };
        let table = TableStepModel::default();
        let backlog = |tokens: u64| {
            (tokens / 8192) as f64 * table.prefill_s(1, 8192, 0).unwrap()
                + table.prefill_s(1, (tokens % 8192) as u32, 0).unwrap()
        };
        let own = table.prefill_s(1, 1024, 0).unwrap();
        let price = |f: &Forecast, class: &str| {
            f.what_if(&WhatIf {
                class: class.into(),
                worker: w,
                new_tokens: 1024,
                prefix_tokens: 0,
            })
            .unwrap()
        };
        // Off: the sample as taken (nothing queued yet) for every class, as before.
        let (off, off_feed) = run(false);
        for class in ["priority", "standard", "batch"] {
            assert!((price(&off, class).ttft_s - own).abs() < 1e-12);
        }
        assert_eq!(off.worker(w, Duration::ZERO).unwrap().decode_batch, 0);
        // On: the sample counts both sends; priority queues behind neither, standard behind its
        // own class, batch behind both.
        let (on, on_feed) = run(true);
        let f = on.worker(w, Duration::ZERO).unwrap();
        assert_eq!(f.decode_batch, 2);
        assert!((f.prefill_backlog_s - backlog(60_000)).abs() < 1e-9);
        assert!((price(&on, "priority").ttft_s - own).abs() < 1e-9);
        assert!((price(&on, "standard").ttft_s - (backlog(10_000) + own)).abs() < 1e-9);
        assert!((price(&on, "batch").ttft_s - (backlog(60_000) + own)).abs() < 1e-9);
        // The standard turn's first token takes it off the queue.
        on_feed.send(Observation::FirstToken {
            request_id: "s".into(),
            worker: w,
            at: secs(2.0),
        });
        assert!((price(&on, "standard").ttft_s - own).abs() < 1e-9);
        assert!((price(&on, "batch").ttft_s - (backlog(50_000) + own)).abs() < 1e-9);
        // The engine's gauges: 4 running at 8,192 tokens of context each (16-token blocks); the
        // standard request decoding there is out of a priority turn's batch only.
        let gauge = EngineGauge {
            worker: w,
            running_requests: 4,
            waiting_requests: 1,
            active_blocks: 4 * 512,
            waiting_tokens: 50_000,
        };
        for feed in [&on_feed, &off_feed] {
            feed.send(Observation::Engine {
                at: secs(2.0),
                gauge,
            });
        }
        assert_eq!(
            price(&on, "priority").itl_before_s,
            table.decode_s(3, 8192).unwrap()
        );
        assert_eq!(
            price(&on, "standard").itl_before_s,
            table.decode_s(4, 8192).unwrap()
        );
        // Off, the gauges are ignored.
        assert_eq!(
            price(&off, "priority").itl_before_s,
            table.decode_s(0, 0).unwrap()
        );
        // Completion and abort release the queue too.
        on_feed.send(Observation::Aborted {
            request_id: "b".into(),
        });
        assert!((price(&on, "batch").ttft_s - own).abs() < 1e-9);
    }

    #[test]
    fn ttft_error_is_scored_from_send_to_first_token() {
        let (forecast, feed, _clock) = setup(ForecastConfig::default());
        let w = WorkerWithDpRank::new(1, 0);
        feed.send(Observation::Workers {
            at: secs(0.0),
            rows: vec![WorkerSample {
                worker: w,
                active_prefill_tokens: 0,
                active_decode_blocks: 0,
                active_requests: 0,
                max_batched_tokens: None,
            }],
        });
        arrive(&feed, "r", "s", 0.0, 1024);
        feed.send(Observation::Sent {
            request_id: "r".into(),
            worker: w,
            at: secs(2.0),
        });
        feed.send(Observation::FirstToken {
            request_id: "r".into(),
            worker: w,
            at: secs(2.5),
        });
        let errors = forecast.errors()["standard"];
        assert_eq!(errors.ttft_s.count, 1);
        assert!((errors.ttft_s.bias().unwrap() - (0.03 - 0.5)).abs() < 1e-9);
    }

    #[test]
    fn the_feed_never_blocks_and_tables_stay_bounded() {
        let (forecast, feed, _clock) = setup(ForecastConfig {
            channel_capacity: 4,
            session_capacity: 3,
            request_capacity: 5,
            ..Default::default()
        });
        for i in 0..10 {
            arrive(&feed, &format!("r{i}"), &format!("s{i}"), i as f64, 10);
        }
        assert_eq!(feed.dropped(), 6);
        assert_eq!(forecast.dropped(), 6);
        forecast.pump();
        for i in 10..40 {
            arrive(&feed, &format!("r{i}"), &format!("s{i}"), i as f64, 10);
            forecast.pump();
        }
        let state = forecast.state.lock();
        assert_eq!(state.sessions.len(), 3);
        assert_eq!(state.by_seen.len(), 3);
        assert!(state.requests.len() <= 5);
        assert!(state.request_order.len() <= 5);
        assert!(state.sessions.contains_key("s39"));
    }

    #[test]
    fn aisimulate_config_needs_the_feature() {
        let config = ForecastConfig {
            aisimulate_config: Some("/nonexistent.json".into()),
            ..Default::default()
        };
        if cfg!(feature = "aisimulate") {
            assert!(matches!(
                config.step_model(),
                Err(StepModelError::Aisimulate(_))
            ));
        } else {
            assert!(config.validate().is_err());
        }
    }

    /// Prices steps with a real AISimulate model when `AISIM_FORECAST_CONFIG` names a
    /// `ForwardPassPerfModelConfig` JSON file (for example DeepSeek-V4-Flash on `b300_sxm`);
    /// skipped otherwise, since the step data may not be installed.
    #[cfg(feature = "aisimulate")]
    #[test]
    fn aisimulate_model_prices_steps_when_configured() {
        let Ok(path) = std::env::var("AISIM_FORECAST_CONFIG") else {
            return;
        };
        let json = std::fs::read_to_string(&path).unwrap();
        let model = AisimulateStepModel::from_json(&json).unwrap();
        let small = model.prefill_s(1, 1024, 0).unwrap();
        let large = model.prefill_s(1, 16384, 0).unwrap();
        assert!(small > 0.0 && large > small, "{small} {large}");
        let d1 = model.decode_s(1, 8192).unwrap();
        let d64 = model.decode_s(64, 8192).unwrap();
        assert!(d1 > 0.0 && d64 >= d1, "{d1} {d64}");
        eprintln!("prefill 1k {small:.4}s 16k {large:.4}s; decode b1 {d1:.4}s b64 {d64:.4}s");
    }
}

#[cfg(test)]
mod k_step_table_v2_tests {
    //! k `forecast.step_table_v2`: the 2-D prefill table, chunking, extrapolation and the summed mixed step.
    use super::*;

    fn table() -> TableStepModelV2 {
        // prefill: 10 ms + 0.05 ms per new token + 1e-7 ms per new token x prefix token (attention)
        let n = vec![64.0, 1024.0, 8192.0];
        let c = vec![0.0, 65_536.0, 131_072.0];
        let prefill_s = n
            .iter()
            .flat_map(|x| {
                c.iter()
                    .map(move |y| (10.0 + 0.05 * x + 1e-7 * x * y) / 1000.0)
            })
            .collect();
        let b = vec![1.0, 64.0, 256.0];
        let l = vec![1024.0, 131_072.0];
        let decode_s = b
            .iter()
            .flat_map(|x| {
                l.iter()
                    .map(move |y| (10.0 + 0.1 * x + 1e-8 * x * y) / 1000.0)
            })
            .collect();
        TableStepModelV2 {
            source: None,
            chunk_tokens: Some(8192),
            prefill_new_tokens: n,
            prefill_prefix: c,
            prefill_s,
            decode_batch: b,
            decode_context: l,
            decode_s,
        }
    }

    #[test]
    fn prefill_follows_the_prefix_and_chunks_long_turns() {
        let t = table();
        t.validate().unwrap();
        let exact = |n: f64, c: f64| (10.0 + 0.05 * n + 1e-7 * n * c) / 1000.0;
        // on the grid
        assert!((t.prefill_s(1, 1024, 131_072).unwrap() - exact(1024.0, 131_072.0)).abs() < 1e-12);
        // a long prefix costs more on a small chunk than a linear prefix weight on new tokens would allow
        assert!(t.prefill_s(1, 64, 131_072).unwrap() > t.prefill_s(1, 64, 0).unwrap());
        // a turn of 16,384 new tokens is two chunks of 8,192, the second over 8,192 more prefix
        let two = exact(8192.0, 100_000.0) + exact(8192.0, 108_192.0);
        assert!((t.prefill_s(1, 16_384, 100_000).unwrap() - two).abs() < 1e-9);
        // prefix past the last point extrapolates (no clamp)
        assert!(t.prefill_s(1, 8192, 262_144).unwrap() > t.prefill_s(1, 8192, 131_072).unwrap());
    }

    #[test]
    fn decode_extrapolates_and_the_mixed_step_is_the_sum() {
        let t = table();
        // past the last context: grows (the v1 table clamps here)
        assert!(t.decode_s(256, 1_048_576).unwrap() > t.decode_s(256, 131_072).unwrap());
        let m = t.mixed_s(2048, 120_000, 64, 65_536).unwrap();
        let sum = t.prefill_s(1, 2048, 120_000).unwrap() + t.decode_s(64, 65_536).unwrap();
        assert!((m - sum).abs() < 1e-12);
        assert_eq!(t.decode_s(0, 1024), Some(0.0));
        assert_eq!(t.prefill_s(1, 0, 1024), Some(0.0));
    }

    #[test]
    fn v22_arrival_rate_keeps_the_trailing_window_and_stretches_the_step() {
        let t = |s: f64| Time::from_secs_f64(s);
        let mut a = VecDeque::new();
        a.push_back((t(10.0), 60_000, 0)); // outside the 60 s window at t = 100
        a.push_back((t(50.0), 30_000, 100_000));
        a.push_back((t(90.0), 30_000, 140_000));
        let (rate, prefix) = v2_arrival_rate(&a, t(100.0), 60.0);
        assert!((rate - 1000.0).abs() < 1e-9);
        assert!((prefix - 120_000.0).abs() < 1e-9);
        // 1,000 tok/s at 45 us per token: the step stretches by 1 / (1 - 0.045)
        assert!((v2_arrival_itl(0.030, 1000.0, 45e-6) - 0.030 / 0.955).abs() < 1e-12);
        // saturates at ten times the bare step
        assert!((v2_arrival_itl(0.030, 1e9, 1.0) - 0.300).abs() < 1e-12);
        assert_eq!(v2_arrival_rate(&VecDeque::new(), t(5.0), 60.0), (0.0, 0.0));
    }

    #[test]
    fn config_without_the_key_is_unchanged_and_with_it_parses() {
        let off: ForecastConfig = serde_json::from_str("{}").unwrap();
        assert!(off.step_table_v2.is_none());
        assert_eq!(off, ForecastConfig::default());
        let json = serde_json::to_string(&table()).unwrap();
        let on: ForecastConfig =
            serde_json::from_str(&format!("{{\"step_table_v2\": {json}}}")).unwrap();
        on.validate().unwrap();
        assert!(on.step_model().is_ok());
        assert!(!on.step_table_v2_arrival);
        let arrival: ForecastConfig = serde_json::from_str(&format!(
            "{{\"step_table_v2\": {json}, \"step_table_v2_arrival\": true}}"
        ))
        .unwrap();
        assert!(arrival.step_table_v2_arrival);
    }
}
