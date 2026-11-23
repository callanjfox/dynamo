// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Forecast routing: a worker-selection scorer that prices each candidate by the misses the
//! session scheduler's forecast predicts if the request goes there (`forecast.rs` `what_if`),
//! with the session's home from the router's session index (the simulator's `route: 'forecast'`,
//! `web/sim.js` `pickWorker`).
//!
//! Per request, as the simulator does:
//!
//! - the candidates are the three cheapest by the default cost (prefill blocks still to compute
//!   plus the worker's active blocks), plus the session's home; the rest are excluded;
//! - each is priced: the forecast's predicted misses (the candidate's own TTFT and ITL, and the
//!   running requests the larger batch pushes over their ITL limit), plus `(best hit - hit) /
//!   20000` for recomputing a prefix another worker holds, plus 1000 for leaving the home when
//!   that loses over half the home's cache while the home has KV room, plus 5 when the worker's
//!   KV would overflow; then the candidate's own TTFT (batch excluded), 1 s weighing like 1e4;
//! - the session stays home unless the forecast predicts misses there that a move avoids.
//!
//! The home is the worker holding the deepest device-tier prefix of the session in the router's
//! session index (`RequestClassifierContext::session_index`), else that of its parent (a
//! subagent's first turn). Without a session index the home is unknown and only the forecast and
//! cache terms decide. The forecast is the classifier's (`forecast.enabled`); the classifier hands
//! it and the index to this state when it is built. At `weight` 0, or without a forecast, the
//! scorer adds nothing.
//!
//! **Soft affinity instead of the index** (`home: last_sent`, f38): the home is the worker the
//! router last sent the session to (a subagent's first turn: its parent's), learned from the
//! classifier's `Sent` events, so the router need not keep a session prefix index at all. The home
//! is a soft target: it joins the candidates and keeps the stay-home and leave-home terms above,
//! and the forecast still moves the session when it predicts misses there. A home the KV router
//! reports no device-tier prefix of this request on is stale (its blocks were evicted, or the
//! session never got there) and is dropped, so the router's own overlap and load costs decide.
//! `home: none` drops the home altogether (attribution). The default `index` is the behaviour
//! above, unchanged.
//!
//! **Soft reserve** (`soft_reserve`, the simulator's `softReserve: 'auto'`, off by default):
//! while priority's attainment over the trailing window is within `on_below` of its target (on)
//! until it is `off_above` over it (off), the `k` lowest-numbered workers are softly reserved for
//! priority, `k = ceil(workers × priority share of live sessions × 1.25)` within [1, workers -
//! 1]. Priority's best reserved worker and a lower class's best unreserved worker join the
//! forecast's candidates, and a candidate on the wrong side costs 3 misses for priority and
//! `1 / (1 - priority target)` for the others. As in the simulator it acts only through forecast
//! routing. The classifier feeds the attainment and the session counts.
//!
//! Differences from the simulator, by design: the forecast prices one decision on the latest load
//! sample, not every request over a planning window; the KV overflow check reads the worker's
//! booked load, not the sessions expected back within the window.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use dynamo_kv_router::plugins::prelude::*;
use dynamo_kv_router::plugins::request_classifier::{
    SessionHolder, SessionIndexReader, SessionTier,
};
use parking_lot::Mutex;
use serde::Deserialize;

use crate::forecast::{Forecast, WhatIf};
use crate::pool::{Override, PoolSide, PoolState};

/// Requests tracked at most; beyond this new ones are not tracked (they score zero).
const MAX_TRACKED: usize = 1 << 20;

/// Cost of one predicted weighted miss (the simulator's 1e7).
const MISS_COST: f64 = 1e7;
/// Cost of one second of the candidate's own predicted TTFT (the simulator's 1e4).
const TTFT_COST: f64 = 1e4;
/// Cost that takes a candidate out of the running (not among the forecast's candidates).
const EXCLUDED: f64 = 1e15;
/// f68pool: the rule ratio of a row on the wrong side of a hard pool (never keeps a limit).
const MASKED_RATIO: f64 = 1e9;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SoftReserveConfig {
    /// The class the workers are reserved for.
    pub class: String,
    /// On when the class's attainment is below its target plus this.
    pub on_below: f64,
    /// Off when the class's attainment is above its target plus this.
    pub off_above: f64,
    /// Headroom over the class's share of live sessions.
    pub headroom: f64,
    /// Least seconds between two attainment readings (the simulator's sample interval).
    pub sample_s: f64,
    /// f48h: workers always left unreserved. The default 1 keeps one worker for the lower
    /// classes even when there are none; 0 lets the reserve cover every worker.
    pub min_unreserved: usize,
}

impl Default for SoftReserveConfig {
    fn default() -> Self {
        Self {
            class: "priority".into(),
            on_below: 0.03,
            off_above: 0.06,
            headroom: 1.25,
            sample_s: 60.0,
            min_unreserved: 1,
        }
    }
}

/// f49d `DYNAMO_PREDICT_LOG=<path>`: the JSON-lines file the forecast route's predictions go to
/// (appended, a line flushed at a time). `None` when the variable is unset or empty, or the file
/// cannot be opened (reported once on stderr).
type PredictSink = Mutex<std::io::LineWriter<std::fs::File>>;

fn predict_sink() -> Option<&'static PredictSink> {
    static SINK: OnceLock<Option<PredictSink>> = OnceLock::new();
    SINK.get_or_init(|| {
        let path = std::env::var_os("DYNAMO_PREDICT_LOG").filter(|p| !p.is_empty())?;
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            Ok(file) => Some(Mutex::new(std::io::LineWriter::new(file))),
            Err(error) => {
                eprintln!("DYNAMO_PREDICT_LOG: cannot open {path:?}: {error}");
                None
            }
        }
    })
    .as_ref()
}

/// f49d: what the forecast predicted for one candidate of a priced selection.
#[derive(Clone, Debug, PartialEq)]
pub struct PredictRow {
    pub worker: WorkerWithDpRank,
    /// The `WhatIf` the route priced: prompt less the candidate's cached prefix, and that prefix.
    pub new_tokens: u64,
    pub prefix_tokens: u64,
    /// Whether the route priced it (in its forecast set); otherwise priced for the log only.
    pub priced: bool,
    pub detail: Option<crate::forecast::PricedDetail>,
    /// The ITL-limit rule's ratio (`itl_limit`; `None` when the rule is off).
    pub itl_ratio: Option<f64>,
    /// The scorer's cost for the candidate (after `weight`).
    pub cost: f64,
}

/// f49d: one priced selection, kept until the router sends the request.
#[derive(Clone, Debug, PartialEq)]
pub struct PendingPrediction {
    /// The forecast clock's seconds when priced.
    pub t: f64,
    pub session: String,
    pub class: String,
    pub isl: u64,
    pub rows: Vec<PredictRow>,
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_num(x: f64) -> String {
    if x.is_finite() {
        format!("{x}")
    } else {
        "null".to_owned()
    }
}

/// f49d: the JSON line for `request_id` sent to `worker` after the selection `p`.
pub fn prediction_line(
    request_id: &str,
    p: &PendingPrediction,
    worker: WorkerWithDpRank,
) -> String {
    let row = p.rows.iter().find(|r| r.worker == worker);
    let detail = row.and_then(|r| r.detail.as_ref());
    let opt = |x: Option<f64>| x.map_or_else(|| "null".to_owned(), json_num);
    let classes = |v: Option<String>| v.unwrap_or_else(|| "null".to_owned());
    let itl = detail.map(|d| {
        let body: Vec<String> = d
            .itl_by_class
            .iter()
            .map(|(c, x)| format!("{}:{}", json_str(c), json_num(*x)))
            .collect();
        format!("{{{}}}", body.join(","))
    });
    let decoding = detail.map(|d| {
        let body: Vec<String> = d
            .decoding_by_class
            .iter()
            .map(|(c, n)| format!("{}:{n}", json_str(c)))
            .collect();
        format!("{{{}}}", body.join(","))
    });
    format!(
        "{{\"t\":{},\"request_id\":{},\"session\":{},\"class\":{},\"worker\":{},\"dp_rank\":{},\"isl\":{},\"candidates\":{},\"in_rows\":{},\"priced\":{},\"new_tokens\":{},\"prefix_tokens\":{},\"predicted_ttft_s\":{},\"predicted_itl_s\":{},\"candidate_itl_before_s\":{},\"candidate_itl_after_s\":{},\"decoding\":{},\"window_s\":{},\"decode_batch\":{},\"prefill_backlog_s\":{},\"candidate_misses\":{},\"pushed_over\":{},\"routing_misses\":{},\"itl_ratio\":{},\"cost\":{}}}",
        json_num(p.t),
        json_str(request_id),
        json_str(&p.session),
        json_str(&p.class),
        worker.worker_id,
        worker.dp_rank,
        p.isl,
        p.rows.len(),
        row.is_some(),
        row.is_some_and(|r| r.priced),
        row.map_or_else(|| "null".to_owned(), |r| r.new_tokens.to_string()),
        row.map_or_else(|| "null".to_owned(), |r| r.prefix_tokens.to_string()),
        opt(detail.map(|d| d.outcome.ttft_s)),
        classes(itl),
        opt(detail.map(|d| d.outcome.itl_before_s)),
        opt(detail.map(|d| d.outcome.itl_after_s)),
        classes(decoding),
        opt(detail.map(|d| d.window_s)),
        detail.map_or_else(|| "null".to_owned(), |d| d.decode_batch.to_string()),
        opt(detail.map(|d| d.prefill_backlog_s)),
        detail.map_or_else(
            || "null".to_owned(),
            |d| d.outcome.candidate_misses.to_string()
        ),
        detail.map_or_else(|| "null".to_owned(), |d| d.outcome.pushed_over.to_string()),
        opt(detail.map(|d| d.outcome.routing_misses())),
        opt(row.and_then(|r| r.itl_ratio)),
        opt(row.map(|r| r.cost)),
    )
}

/// What the classifier last told the soft reserve.
#[derive(Clone, Copy, Debug, Default)]
struct ReserveState {
    on: bool,
    /// Share of live top-level sessions in the reserved class.
    share: f64,
    target: f64,
    flips: u64,
}

/// Where forecast routing takes a session's home from.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HomeSource {
    /// The worker holding the session's deepest device-tier prefix in the router's session
    /// prefix index (needs `enable_session_prefix_index`).
    #[default]
    Index,
    /// Soft affinity: the worker the router last sent the session to, dropped when the KV router
    /// sees none of this request's prefix there. No session index is read.
    LastSent,
    /// No home.
    None,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ForecastRouteConfig {
    /// The shared-state name the classifier feeds.
    pub name: String,
    /// Multiplies every cost this scorer adds; 0 adds nothing.
    pub weight: f64,
    /// Candidates forecast: this many cheapest by the default cost, plus the home.
    pub candidates: usize,
    /// Tokens of recomputed prefix that weigh like one miss.
    pub hit_tokens_per_miss: f64,
    /// Soft reservation for priority; off (absent) by default.
    pub soft_reserve: Option<SoftReserveConfig>,
    /// Count the prefix a candidate holds below HBM: its own host (G2) tier and what KVCR can fetch
    /// from the source the router would name, at 90% (the transfer costs time), as the simulator's
    /// `pickWorker` does (`max(resident, 0.9 x farHit)`). Off by default: device-tier hits only,
    /// which with a host tier sees no cache where the router's own cost sees most of it, so the
    /// forecast moved sessions off the worker whose DRAM held them.
    pub lower_tier_hits: bool,
    /// The ITL limit as a hard rule (F22): a candidate whose predicted mean ITL for a class
    /// decoding there would pass that class's limit once this request's prefill joins it
    /// ([`Forecast::itl_ratio`]) is not picked while another candidate keeps every limit; when
    /// none does, the one breaking its limits least is preferred. Off by default.
    pub itl_limit: bool,
    /// With `itl_limit`: apply the rule only when some candidate keeps every limit; when none
    /// does, leave the route's costs as they are (no least-bad fallback). Off by default.
    pub itl_limit_only_when_kept: bool,
    /// With `itl_limit`: judge a request only against the decodes of classes ranked above its own
    /// (keep lower classes' prefill off higher classes' decodes; a top-class request is never
    /// steered by the rule). Off by default.
    pub itl_limit_lower_only: bool,
    /// Where the session's home comes from: `index` (default), `last_sent` or `none`.
    pub home: HomeSource,
    /// With `home: last_sent`: drop the soft home when its worker's active load (prefill tokens
    /// plus active decode blocks, in tokens) is over this many times the candidates' mean, so the
    /// stay-home and leave-home terms do not pin a session to the fleet's hot spot (f38b). Off
    /// (absent) by default.
    pub home_max_load: Option<f64>,
    /// With `home: last_sent`: drop the soft home when the forecast predicts this request misses
    /// its own limits there (TTFT or ITL: the queue ahead of it outlasts the class limit). Off by
    /// default.
    pub home_drop_if_late: bool,
    /// With `home_drop_if_late`: drop the late home only when another candidate is predicted to
    /// keep this request's limits; when every candidate would miss (a fleet-wide backlog, as under
    /// write-back), the home stays. Off by default.
    pub home_late_needs_alternative: bool,
}

impl Default for ForecastRouteConfig {
    fn default() -> Self {
        Self {
            name: "default".into(),
            weight: 1.0,
            candidates: 3,
            hit_tokens_per_miss: 20_000.0,
            soft_reserve: None,
            lower_tier_hits: false,
            itl_limit: false,
            itl_limit_only_when_kept: false,
            itl_limit_lower_only: false,
            home: HomeSource::Index,
            home_max_load: None,
            home_drop_if_late: false,
            home_late_needs_alternative: false,
        }
    }
}

struct Request {
    session: String,
    parent: Option<String>,
    class: String,
    session_final: bool,
}

/// Soft affinity's home for one request (`home: last_sent`): the session's last worker, else its
/// parent's, kept only while that worker is a candidate and the KV router sees a device-tier
/// prefix of this request there. Returns the home and whether a remembered worker was dropped as
/// stale.
pub fn soft_home(
    session_last: Option<WorkerWithDpRank>,
    parent_last: Option<WorkerWithDpRank>,
    rows: &[Row],
) -> (Option<WorkerWithDpRank>, bool) {
    let Some(last) = session_last.or(parent_last) else {
        return (None, false);
    };
    match rows.iter().find(|r| r.worker == last) {
        Some(r) if r.hit_tokens > 0.0 => (Some(last), false),
        _ => (None, true),
    }
}

/// Whether `home`'s active load is over `ratio` times the candidates' mean (`home_max_load`). A
/// home that is not a candidate, or an idle fleet, is never over.
pub fn home_overloaded(rows: &[Row], home: WorkerWithDpRank, ratio: f64) -> bool {
    if rows.is_empty() {
        return false;
    }
    let mean = rows.iter().map(|r| r.active_tokens).sum::<f64>() / rows.len() as f64;
    mean > 0.0
        && rows
            .iter()
            .find(|r| r.worker == home)
            .is_some_and(|r| r.active_tokens > ratio * mean)
}

/// Whether a soft home is dropped as late (`home_drop_if_late`): `late` says whether the forecast
/// predicts this request misses its limits on a candidate. The home is dropped when it is a
/// candidate and late there, and, with `needs_alternative`, only when another candidate is not.
pub fn late_home_dropped(
    rows: &[Row],
    home: WorkerWithDpRank,
    needs_alternative: bool,
    late: impl Fn(&Row) -> bool,
) -> bool {
    rows.iter().find(|r| r.worker == home).is_some_and(&late)
        && (!needs_alternative || rows.iter().any(|o| o.worker != home && !late(o)))
}

pub struct ForecastRouteState {
    config: ForecastRouteConfig,
    /// f50: the settings while the mix selector is in pg mode (`None`: `config` in every mode).
    pg: Option<ForecastRouteConfig>,
    requests: Mutex<HashMap<String, Request>>,
    forecast: OnceLock<Arc<Forecast>>,
    index: OnceLock<Arc<dyn SessionIndexReader>>,
    /// `home: last_sent`: the worker each session was last sent to.
    last_sent: Mutex<HashMap<String, WorkerWithDpRank>>,
    soft_homes: AtomicU64,
    stale_homes: AtomicU64,
    /// Soft homes dropped for load (`home_max_load`) and for a predicted miss (`home_drop_if_late`).
    loaded_homes: AtomicU64,
    late_homes: AtomicU64,
    reserve: Mutex<ReserveState>,
    last_reading: Mutex<Option<crate::clock::Time>>,
    reserve_on_selections: AtomicU64,
    scored: AtomicU64,
    priced: AtomicU64,
    stayed_home: AtomicU64,
    moved: AtomicU64,
    /// Selections under the ITL limit, and of those how many had a candidate keeping every limit.
    itl_ruled: AtomicU64,
    itl_kept: AtomicU64,
    /// f49d `DYNAMO_PREDICT_LOG`: each priced request's selection until it is sent.
    predictions: Mutex<HashMap<String, PendingPrediction>>,
    /// f68pool: the priority pool of the same placement policy (its `exclude_others` side).
    pool: OnceLock<Arc<PoolState>>,
}

impl ForecastRouteState {
    pub fn new(config: ForecastRouteConfig) -> Self {
        Self {
            config,
            pg: None,
            requests: Mutex::new(HashMap::new()),
            forecast: OnceLock::new(),
            index: OnceLock::new(),
            last_sent: Mutex::new(HashMap::new()),
            soft_homes: AtomicU64::new(0),
            stale_homes: AtomicU64::new(0),
            loaded_homes: AtomicU64::new(0),
            late_homes: AtomicU64::new(0),
            reserve: Mutex::new(ReserveState::default()),
            last_reading: Mutex::new(None),
            reserve_on_selections: AtomicU64::new(0),
            scored: AtomicU64::new(0),
            priced: AtomicU64::new(0),
            stayed_home: AtomicU64::new(0),
            moved: AtomicU64::new(0),
            itl_ruled: AtomicU64::new(0),
            itl_kept: AtomicU64::new(0),
            predictions: Mutex::new(HashMap::new()),
            pool: OnceLock::new(),
        }
    }

    /// f68pool: the priority pool placed beside this route (first call wins).
    pub fn attach_pool(&self, pool: Arc<PoolState>) {
        let _ = self.pool.set(pool);
    }

    /// f49d: keep a priced selection of `request_id` for the log line written when it is sent
    /// (the latest selection wins).
    pub fn stash_prediction(&self, request_id: &str, prediction: PendingPrediction) {
        let mut p = self.predictions.lock();
        if p.len() >= MAX_TRACKED && !p.contains_key(request_id) {
            return;
        }
        p.insert(request_id.to_owned(), prediction);
    }

    /// f49d: the log line of `request_id` sent to `worker`, taking its kept selection.
    pub fn take_prediction_line(
        &self,
        request_id: &str,
        worker: WorkerWithDpRank,
    ) -> Option<String> {
        let p = self.predictions.lock().remove(request_id)?;
        Some(prediction_line(request_id, &p, worker))
    }

    pub fn config(&self) -> &ForecastRouteConfig {
        &self.config
    }

    /// f50: the settings in force: the pg mode's while the mix selector is in pg, else `config`.
    fn cfg(&self) -> &ForecastRouteConfig {
        match (&self.pg, crate::mix_select::current()) {
            (Some(pg), Some(crate::mix_select::MixMode::Pg)) => pg,
            _ => &self.config,
        }
    }

    /// f50: whether any mode reads `home: last_sent` (it is then recorded in every mode).
    fn records_last_sent(&self) -> bool {
        self.config.home == HomeSource::LastSent
            || self
                .pg
                .as_ref()
                .is_some_and(|p| p.home == HomeSource::LastSent)
    }

    /// The classifier's forecast and the router's session index (first call wins).
    pub fn attach(
        &self,
        forecast: Option<Arc<Forecast>>,
        index: Option<Arc<dyn SessionIndexReader>>,
    ) {
        if let Some(f) = forecast {
            let _ = self.forecast.set(f);
        }
        if let Some(i) = index {
            let _ = self.index.set(i);
        }
    }

    pub fn arrived(
        &self,
        request_id: &str,
        session: &str,
        parent: Option<&str>,
        class: &str,
        session_final: bool,
    ) {
        let mut requests = self.requests.lock();
        if requests.len() >= MAX_TRACKED && !requests.contains_key(request_id) {
            return;
        }
        requests.insert(
            request_id.to_owned(),
            Request {
                session: session.to_owned(),
                parent: parent.map(str::to_owned),
                class: class.to_owned(),
                session_final,
            },
        );
    }

    /// The router sent the request to `worker` (`home: last_sent` remembers it as the session's
    /// home; the other sources ignore it).
    pub fn sent(&self, request_id: &str, worker: WorkerWithDpRank) {
        // f49d `DYNAMO_PREDICT_LOG`: the prediction on the worker the request went to.
        if let Some(sink) = predict_sink()
            && let Some(line) = self.take_prediction_line(request_id, worker)
        {
            use std::io::Write as _;
            let _ = writeln!(sink.lock(), "{line}");
        }
        if !self.records_last_sent() {
            return;
        }
        let Some(session) = self
            .requests
            .lock()
            .get(request_id)
            .map(|r| r.session.clone())
        else {
            return;
        };
        let mut last = self.last_sent.lock();
        if last.len() >= MAX_TRACKED && !last.contains_key(&session) {
            return;
        }
        last.insert(session, worker);
    }

    pub fn finished(&self, request_id: &str) {
        // f49d: a selection never sent is dropped with its request.
        self.predictions.lock().remove(request_id);
        let removed = self.requests.lock().remove(request_id);
        if let Some(r) = removed
            && r.session_final
        {
            self.last_sent.lock().remove(&r.session);
        }
    }

    /// The home for one request under the configured source, given its candidate rows.
    fn home_for(
        &self,
        session: &str,
        parent: Option<&str>,
        rows: &[Row],
    ) -> Option<WorkerWithDpRank> {
        match self.cfg().home {
            HomeSource::Index => self.home_of(session, parent),
            HomeSource::None => None,
            HomeSource::LastSent => {
                let (own, par) = {
                    let last = self.last_sent.lock();
                    (
                        last.get(session).copied(),
                        parent.and_then(|p| last.get(p).copied()),
                    )
                };
                let (mut home, stale) = soft_home(own, par, rows);
                if let (Some(h), Some(ratio)) = (home, self.cfg().home_max_load)
                    && home_overloaded(rows, h, ratio)
                {
                    self.loaded_homes.fetch_add(1, Ordering::Relaxed);
                    home = None;
                }
                if home.is_some() {
                    self.soft_homes.fetch_add(1, Ordering::Relaxed);
                }
                if stale {
                    self.stale_homes.fetch_add(1, Ordering::Relaxed);
                }
                home
            }
        }
    }

    /// `home: last_sent`: (selections with a soft home, remembered homes dropped as stale,
    /// sessions remembered now).
    pub fn soft_home_stats(&self) -> (u64, u64, usize) {
        (
            self.soft_homes.load(Ordering::Relaxed),
            self.stale_homes.load(Ordering::Relaxed),
            self.last_sent.lock().len(),
        )
    }

    /// `home: last_sent`: (soft homes dropped for load, soft homes dropped for a predicted miss).
    pub fn dropped_home_stats(&self) -> (u64, u64) {
        (
            self.loaded_homes.load(Ordering::Relaxed),
            self.late_homes.load(Ordering::Relaxed),
        )
    }

    /// Whether the soft reserve wants a reading at `now` (at most every `sample_s`).
    pub fn reading_due(&self, now: crate::clock::Time) -> bool {
        let Some(cfg) = &self.cfg().soft_reserve else {
            return false;
        };
        let mut last = self.last_reading.lock();
        if last.is_some_and(|at| now.since(at).as_secs_f64() < cfg.sample_s) {
            return false;
        }
        *last = Some(now);
        true
    }

    /// A reading for the soft reserve: the reserved class's attainment over the trailing window
    /// (`None` when too few completions), its target, and its share of live top-level sessions.
    pub fn reading(&self, attainment: Option<f64>, target: Option<f64>, share: f64) {
        let Some(cfg) = &self.cfg().soft_reserve else {
            return;
        };
        let mut r = self.reserve.lock();
        r.share = share;
        if let (Some(a), Some(t)) = (attainment, target) {
            r.target = t;
            let was = r.on;
            if !was && a < t + cfg.on_below {
                r.on = true;
            } else if was && a > t + cfg.off_above {
                r.on = false;
            }
            if was != r.on {
                r.flips += 1;
            }
        }
    }

    /// The soft reserve now: (reserved class, workers reserved out of `n`, priority target).
    fn reserved(&self, n: usize) -> Option<(String, usize, f64)> {
        let cfg = self.cfg().soft_reserve.as_ref()?;
        let r = *self.reserve.lock();
        if !r.on || n < 2 || r.share <= 0.0 {
            return None;
        }
        let k = ((n as f64) * r.share * cfg.headroom).ceil() as usize;
        Some((
            cfg.class.clone(),
            k.clamp(1, n.saturating_sub(cfg.min_unreserved).max(1)),
            r.target,
        ))
    }

    /// (selections with the soft reserve on, times it turned on or off).
    pub fn reserve_stats(&self) -> (u64, u64) {
        (
            self.reserve_on_selections.load(Ordering::Relaxed),
            self.reserve.lock().flips,
        )
    }

    /// The worker holding the deepest device-tier prefix of `session`, if the index knows one.
    fn deepest_device_holder(&self, session: &str) -> Option<WorkerWithDpRank> {
        let index = self.index.get()?;
        index
            .session_residency(session)
            .into_iter()
            .filter(|r| r.tier == SessionTier::Device && r.blocks > 0)
            .filter_map(|r| match r.holder {
                SessionHolder::Worker(w) => Some((r.blocks, w)),
                SessionHolder::SharedCache => None,
            })
            .max_by_key(|(blocks, w)| (*blocks, std::cmp::Reverse((w.worker_id, w.dp_rank))))
            .map(|(_, w)| w)
    }

    /// The request's home: its session's deepest holder, else its parent's.
    fn home_of(&self, session: &str, parent: Option<&str>) -> Option<WorkerWithDpRank> {
        self.deepest_device_holder(session)
            .or_else(|| parent.and_then(|p| self.deepest_device_holder(p)))
    }

    /// (selections scored, selections priced by the forecast, stayed home, moved off home).
    /// (selections under the ITL limit, those where some candidate kept every limit).
    pub fn itl_stats(&self) -> (u64, u64) {
        (
            self.itl_ruled.load(Ordering::Relaxed),
            self.itl_kept.load(Ordering::Relaxed),
        )
    }

    pub fn stats(&self) -> (u64, u64, u64, u64) {
        (
            self.scored.load(Ordering::Relaxed),
            self.priced.load(Ordering::Relaxed),
            self.stayed_home.load(Ordering::Relaxed),
            self.moved.load(Ordering::Relaxed),
        )
    }
}

fn states() -> &'static Mutex<HashMap<String, Arc<ForecastRouteState>>> {
    static STATES: OnceLock<Mutex<HashMap<String, Arc<ForecastRouteState>>>> = OnceLock::new();
    STATES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Install a fresh state under `config.name`, replacing an earlier one.
pub fn install(config: ForecastRouteConfig) -> Arc<ForecastRouteState> {
    let state = Arc::new(ForecastRouteState::new(config));
    states()
        .lock()
        .insert(state.cfg().name.clone(), Arc::clone(&state));
    state
}

/// f50: install with `b0`'s settings and `pg`'s for the mix selector's pg mode, under `b0.name`.
pub fn install_modes(b0: ForecastRouteConfig, pg: ForecastRouteConfig) -> Arc<ForecastRouteState> {
    let mut state = ForecastRouteState::new(b0);
    state.pg = Some(pg);
    let state = Arc::new(state);
    states()
        .lock()
        .insert(state.config.name.clone(), Arc::clone(&state));
    state
}

/// The state a forecast route installed under `name`, if any.
pub fn installed(name: &str) -> Option<Arc<ForecastRouteState>> {
    states().lock().get(name).cloned()
}

/// The soft reserve for one request: the reserved workers, whether the request belongs on them,
/// and the misses a candidate on the wrong side costs.
#[derive(Clone, Debug)]
pub struct Reserve {
    pub workers: Vec<WorkerWithDpRank>,
    pub mine: bool,
    pub wrong_side: f64,
}

/// One candidate as the forecast route sees it.
#[derive(Clone, Copy, Debug)]
pub struct Row {
    pub worker: WorkerWithDpRank,
    /// Prefix tokens of this request cached on the worker (device tier).
    pub hit_tokens: f64,
    /// Default cost, in blocks: prefill still to compute plus the worker's active blocks.
    pub base_cost: f64,
    /// The worker's active load in tokens: prefill tokens plus active decode blocks.
    pub active_tokens: f64,
    /// Whether the worker's KV would overflow with this request.
    pub over: bool,
    /// Whether the worker has KV room for this request.
    pub room: bool,
}

/// What the forecast predicts for one candidate: (weighted misses, own TTFT seconds).
pub type Price = Option<(f64, f64)>;

/// The extra cost per row (the scorer's output before `weight`), given the request's home and a
/// pricing function. Rows outside the candidate set get [`EXCLUDED`].
pub fn route_costs(
    rows: &[Row],
    home: Option<WorkerWithDpRank>,
    candidates: usize,
    hit_tokens_per_miss: f64,
    own_ttft_counts: bool,
    reserve: Option<&Reserve>,
    price: impl FnMut(&Row) -> Price,
) -> (Vec<f64>, Option<bool>) {
    let (costs, stayed, _) = route_costs_sided(
        rows,
        home,
        candidates,
        hit_tokens_per_miss,
        own_ttft_counts,
        reserve,
        None,
        price,
    );
    (costs, stayed)
}

/// [`route_costs`] with the request's side of the priority pool (f68pool `exclude_others`,
/// `pool.rs` [`PoolSide`]). With a hard side (standard and batch) only the request's own side
/// is a candidate: the cheapest set is taken from it, a home on the other side is dropped (the
/// session re-homes at this turn) and the cache term compares hits on that side. With a soft
/// side (priority) the cheapest worker on its side joins the candidates when none is among them.
/// Also returns the overrides that changed the candidate set.
#[allow(clippy::too_many_arguments)]
pub fn route_costs_sided(
    rows: &[Row],
    home: Option<WorkerWithDpRank>,
    candidates: usize,
    hit_tokens_per_miss: f64,
    own_ttft_counts: bool,
    reserve: Option<&Reserve>,
    side: Option<&PoolSide>,
    mut price: impl FnMut(&Row) -> Price,
) -> (Vec<f64>, Option<bool>, Vec<Override>) {
    let n = rows.len();
    let mut notes = Vec::new();
    let own = |i: usize| side.is_none_or(|s| s.own.get(i).copied().unwrap_or(true));
    let hard = side.is_some_and(|s| s.hard);
    // Equal default costs (idle workers, for one) are ordered by worker, not by the order the
    // router lists its candidates in, which need not repeat: the candidate set, and so the pick,
    // would change from run to run.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| {
        let key = |i: usize| (rows[i].worker.worker_id, rows[i].worker.dp_rank);
        rows[*a]
            .base_cost
            .total_cmp(&rows[*b].base_cost)
            .then_with(|| key(*a).cmp(&key(*b)))
    });
    if hard {
        let cut = &order[..candidates.max(1).min(n)];
        let wrong = cut.iter().filter(|&&i| !own(i)).count();
        if wrong > 0 && wrong == cut.len() {
            notes.push(Override::CutAllPool);
        } else if wrong > 0 {
            notes.push(Override::CutHadPool);
        }
        order.retain(|&i| own(i));
    }
    let mut chosen: Vec<usize> = order.iter().copied().take(candidates.max(1)).collect();
    let mut home_row = home.and_then(|h| rows.iter().position(|r| r.worker == h));
    if hard && home_row.is_some_and(|h| !own(h)) {
        notes.push(Override::HomeInPool);
        home_row = None;
    }
    if let Some(h) = home_row
        && !chosen.contains(&h)
    {
        chosen.push(h);
    }
    // Soft reserve: also the cheapest worker on the request's own side.
    if let Some(res) = reserve
        && let Some(mine) = order
            .iter()
            .copied()
            .find(|&i| res.workers.contains(&rows[i].worker) == res.mine)
        && !chosen.contains(&mine)
    {
        chosen.push(mine);
    }
    // f68pool, priority: its cheapest pool worker joins when the cheapest set holds none.
    if side.is_some()
        && !hard
        && !chosen.iter().any(|&i| own(i))
        && let Some(mine) = order.iter().copied().find(|&i| own(i))
    {
        notes.push(Override::PriorityPoolAdded);
        chosen.push(mine);
    }
    let best_hit = (0..n)
        .filter(|&i| !hard || own(i))
        .map(|i| rows[i].hit_tokens)
        .fold(0.0, f64::max);
    let home_hit = home_row.map_or(0.0, |h| rows[h].hit_tokens);
    let home_room = home_row.is_some_and(|h| rows[h].room);
    let mut costs = vec![EXCLUDED; n];
    let mut merit = vec![f64::INFINITY; n];
    for &i in &chosen {
        let r = &rows[i];
        let (misses, ttft) = price(r).unwrap_or((0.0, 0.0));
        let mut pen = (best_hit - r.hit_tokens).max(0.0) / hit_tokens_per_miss.max(1.0);
        if let Some(res) = reserve
            && res.workers.contains(&r.worker) != res.mine
        {
            pen += res.wrong_side;
        }
        if home_row.is_some_and(|h| h != i) && r.hit_tokens < 0.5 * home_hit && home_room {
            pen += 1e3;
        }
        merit[i] = misses + pen;
        let own = if own_ttft_counts {
            ttft.min(60.0) * TTFT_COST
        } else {
            0.0
        };
        costs[i] = (misses + pen + if r.over { 5.0 } else { 0.0 }) * MISS_COST + own;
    }
    // Stay home unless the forecast predicts misses there that a move avoids.
    let mut stayed = None;
    if let Some(h) = home_row {
        // Against the cheapest candidate overall (as the simulator compares with its first).
        let first = chosen
            .iter()
            .copied()
            .min_by(|a, b| {
                (costs[*a] + rows[*a].base_cost).total_cmp(&(costs[*b] + rows[*b].base_cost))
            })
            .unwrap_or(h);
        let at_home = merit[h] <= merit[first] + 1e-9;
        stayed = Some(at_home);
        if at_home {
            for (i, c) in costs.iter_mut().enumerate() {
                if i != h {
                    *c = EXCLUDED;
                }
            }
        }
    }
    (costs, stayed, notes)
}

/// f68pool: under a hard pool side, the ITL and step-budget rules ([`itl_limit_costs`]) must not
/// let a row on the other side back in: its ratio is made one that never keeps the limit. Returns
/// whether that changed the rule's outcome (an other-side row kept the limit while no routed row
/// on the request's side did).
pub fn mask_rule_ratios(ratios: &mut [f64], costs: &[f64], side: Option<&PoolSide>) -> bool {
    let Some(side) = side.filter(|s| s.hard) else {
        return false;
    };
    let own = |i: usize| side.own.get(i).copied().unwrap_or(true);
    let ok = |r: f64| r <= 1.0;
    let routed_ok = (0..ratios.len()).any(|i| own(i) && ok(ratios[i]) && costs[i] < EXCLUDED);
    let other_ok = (0..ratios.len()).any(|i| !own(i) && ok(ratios[i]));
    for (i, r) in ratios.iter_mut().enumerate() {
        if !own(i) {
            *r = MASKED_RATIO;
        }
    }
    other_ok && !routed_ok
}

/// The ITL limit as a hard rule over the route's costs (`itl_limit`). `ratios[i]` is candidate
/// `i`'s predicted mean ITL over its tightest decoding class's limit (0 with nothing decoding).
/// While some candidate keeps every limit (ratio at most 1), the others are excluded; a candidate
/// the route left out (outside its forecast set) that keeps every limit is let back in at no
/// extra cost when none of the route's own candidates does. When no candidate keeps every limit,
/// each is charged for how far it goes over, so the least-bad one wins. Returns the costs and
/// whether some candidate kept every limit.
pub fn itl_limit_costs(costs: &[f64], ratios: &[f64]) -> (Vec<f64>, bool) {
    let ok = |i: usize| ratios.get(i).copied().unwrap_or(0.0) <= 1.0;
    let n = costs.len();
    let any_ok = (0..n).any(ok);
    if !any_ok {
        let out = (0..n)
            .map(|i| {
                let over = ratios.get(i).copied().unwrap_or(0.0) - 1.0;
                if costs[i] >= EXCLUDED {
                    costs[i]
                } else {
                    costs[i] + over.max(0.0) * MISS_COST
                }
            })
            .collect();
        return (out, false);
    }
    let routed_ok = (0..n).any(|i| ok(i) && costs[i] < EXCLUDED);
    let out = (0..n)
        .map(|i| {
            if !ok(i) {
                EXCLUDED
            } else if costs[i] >= EXCLUDED && !routed_ok {
                0.0
            } else {
                costs[i]
            }
        })
        .collect();
    (out, true)
}

/// A candidate's prefix in tokens with `lower_tier_hits`: its device prefix, or 90% of the longer of
/// its device plus own host prefix and the prefix KVCR fetches from the named source, whichever is
/// more (`web/sim.js` `pickWorker`: `max(local, 0.9 x farHit)`).
pub fn candidate_hit_tokens(cache: Option<(f64, f64)>, fetch_blocks: Option<u32>, bs: f64) -> f64 {
    let (device, host) = cache.unwrap_or((0.0, 0.0));
    let far = (device + host).max(f64::from(fetch_blocks.unwrap_or(0)));
    device.max(0.9 * far) * bs
}

/// Prices each candidate by the forecast's predicted misses.
pub struct ForecastRouteScorer {
    state: Arc<ForecastRouteState>,
}

impl ForecastRouteScorer {
    pub fn new(state: Arc<ForecastRouteState>) -> Self {
        Self { state }
    }
}

impl WorkerScorer for ForecastRouteScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        if self.state.cfg().lower_tier_hits {
            WorkerInputs::CACHE | WorkerInputs::LOAD | WorkerInputs::KV_TRANSFER
        } else {
            WorkerInputs::CACHE | WorkerInputs::LOAD
        }
    }

    fn score(
        &self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        costs.fill(0.0);
        // The adaptive rule's place group (`adaptive.rs`) switched off: today's router decides.
        if !crate::adaptive::place_enabled() {
            return Ok(());
        }
        let state = &self.state;
        state.scored.fetch_add(1, Ordering::Relaxed);
        let weight = state.cfg().weight;
        let Some(forecast) = state.forecast.get() else {
            return Ok(());
        };
        if weight == 0.0 {
            return Ok(());
        }
        let Some((session, parent, class)) = context.request_id().and_then(|id| {
            state
                .requests
                .lock()
                .get(id)
                .map(|r| (r.session.clone(), r.parent.clone(), r.class.clone()))
        }) else {
            return Ok(());
        };
        let bs = f64::from(context.block_size().max(1));
        let isl = context.isl_tokens() as f64;
        let lower = state.cfg().lower_tier_hits;
        let rows: Vec<Row> = candidates
            .iter()
            .map(|c| {
                let hit = if lower {
                    candidate_hit_tokens(
                        c.cache()
                            .map(|k| (k.device_overlap_blocks(), k.host_overlap_blocks())),
                        c.kv_transfer().and_then(|t| t.source_prefix_blocks()),
                        bs,
                    )
                } else {
                    c.cache().map_or(0.0, |k| k.device_overlap_blocks() * bs)
                }
                .min(isl);
                let (active, prefill) = c.load().map_or((0.0, 0.0), |l| {
                    (
                        l.active_prefill_tokens as f64 + l.active_decode_blocks as f64 * bs,
                        l.active_prefill_tokens as f64,
                    )
                });
                let cap = c.info().total_kv_blocks().map(|b| b as f64 * bs);
                let need = active - prefill + isl;
                Row {
                    worker: c.worker(),
                    hit_tokens: hit,
                    base_cost: (isl - hit) / bs + active / bs,
                    active_tokens: active,
                    over: cap.is_some_and(|cap| need > cap),
                    room: cap.is_none_or(|cap| need <= cap),
                }
            })
            .collect();
        let mut home = state.home_for(&session, parent.as_deref(), &rows);
        // `home_drop_if_late`: a soft home where the forecast predicts this request misses its own
        // limits is dropped, so the forecast and cache terms pick among the candidates freely. With
        // `home_late_needs_alternative`, only when another candidate is predicted to keep them.
        if state.cfg().home == HomeSource::LastSent && state.cfg().home_drop_if_late {
            let late = |r: &Row| {
                forecast
                    .what_if(&WhatIf {
                        class: class.clone(),
                        worker: r.worker,
                        new_tokens: (isl - r.hit_tokens).max(0.0) as u64,
                        prefix_tokens: r.hit_tokens as u64,
                    })
                    .is_some_and(|o| o.candidate_misses > 0)
            };
            if let Some(h) = home
                && late_home_dropped(
                    rows.as_slice(),
                    h,
                    state.cfg().home_late_needs_alternative,
                    late,
                )
            {
                state.late_homes.fetch_add(1, Ordering::Relaxed);
                home = None;
            }
        }
        let own_ttft_counts = class != "batch";
        // f49d `DYNAMO_PREDICT_LOG`: keep what the forecast predicted for each candidate.
        let logging = predict_sink().is_some();
        let mut priced_details: Vec<(WorkerWithDpRank, crate::forecast::PricedDetail)> = Vec::new();
        let mut logged_ratios: Option<Vec<f64>> = None;
        let reserve = state
            .reserved(rows.len())
            .map(|(reserved_class, k, target)| {
                state.reserve_on_selections.fetch_add(1, Ordering::Relaxed);
                let mut ids: Vec<WorkerWithDpRank> = rows.iter().map(|r| r.worker).collect();
                ids.sort_by_key(|w| (w.worker_id, w.dp_rank));
                ids.truncate(k);
                let mine = class == reserved_class;
                Reserve {
                    workers: ids,
                    mine,
                    wrong_side: if mine {
                        3.0
                    } else {
                        1.0 / (1.0 - target).max(0.01)
                    },
                }
            });
        // f68pool `exclude_others`: the request's side of the priority pool (`None` when off).
        let pool = state.pool.get();
        let side = pool.and_then(|p| {
            let ids: Vec<u64> = rows.iter().map(|r| r.worker.worker_id).collect();
            p.side(context.request_id(), &ids)
        });
        let (extra, stayed, notes) = route_costs_sided(
            &rows,
            home,
            state.cfg().candidates,
            state.cfg().hit_tokens_per_miss,
            own_ttft_counts,
            reserve.as_ref(),
            side.as_ref(),
            |r| {
                let decision = WhatIf {
                    class: class.clone(),
                    worker: r.worker,
                    new_tokens: (isl - r.hit_tokens).max(0.0) as u64,
                    prefix_tokens: r.hit_tokens as u64,
                };
                // f49d: logging, the same price with its parts (`what_if_detail`'s outcome is
                // `what_if`'s).
                let outcome = if logging {
                    let detail = forecast.what_if_detail(&decision)?;
                    let outcome = detail.outcome;
                    priced_details.push((r.worker, detail));
                    outcome
                } else {
                    forecast.what_if(&decision)?
                };
                Some((outcome.routing_misses(), outcome.ttft_s))
            },
        );
        state.priced.fetch_add(1, Ordering::Relaxed);
        if let Some(p) = pool {
            for why in notes {
                p.record_override(why);
            }
        }
        let mut masked = false;
        let extra = if state.cfg().itl_limit && crate::coordinator::itl_limit_enabled() {
            let mut ratios: Vec<f64> = rows
                .iter()
                .map(|r| {
                    forecast
                        .itl_ratio_for(
                            r.worker,
                            (isl - r.hit_tokens).max(0.0) as u64,
                            r.hit_tokens as u64,
                            state.cfg().itl_limit_lower_only.then_some(class.as_str()),
                        )
                        .unwrap_or(0.0)
                })
                .collect();
            if logging {
                logged_ratios = Some(ratios.clone());
            }
            masked |= mask_rule_ratios(&mut ratios, &extra, side.as_ref());
            let (adjusted, kept) = itl_limit_costs(&extra, &ratios);
            let adjusted = if !kept && state.cfg().itl_limit_only_when_kept {
                extra.clone()
            } else {
                adjusted
            };
            state.itl_kept.fetch_add(u64::from(kept), Ordering::Relaxed);
            let ruled = state.itl_ruled.fetch_add(1, Ordering::Relaxed) + 1;
            // DYNAMO_FORECAST_ROUTE_STATS=1 prints the rule's counters to stderr every 20,000
            // selections.
            if ruled.is_multiple_of(20_000)
                && std::env::var_os("DYNAMO_FORECAST_ROUTE_STATS").is_some()
            {
                eprintln!(
                    "forecast_route_itl_limit ruled={ruled} kept={}",
                    state.itl_kept.load(Ordering::Relaxed)
                );
            }
            adjusted
        } else {
            extra
        };
        // f49i `admission.worker_step_budget`: a candidate that breaks the step budget is not picked
        // while another keeps it, in the route's own order (the ITL limit's hard rule on the
        // budget's ratios); when none keeps it (the gate let the turn go at its wait limit, or the
        // load moved since), the one over least. The turn computes what the classifier noted per
        // worker, as the gate judged it, else the prompt less the candidate's hit.
        let extra = match forecast.step_budget() {
            Some(budget) => {
                let noted = context.request_id().and_then(|id| budget.computed(id));
                let mut ratios: Vec<f64> = rows
                    .iter()
                    .map(|r| {
                        let new_tokens = noted
                            .as_ref()
                            .and_then(|per| {
                                per.iter().find(|(w, _)| *w == r.worker).map(|(_, t)| *t)
                            })
                            .unwrap_or((isl - r.hit_tokens).max(0.0) as u64)
                            .min(isl as u64);
                        forecast
                            .step_budget_check(
                                r.worker,
                                &class,
                                new_tokens,
                                (isl as u64).saturating_sub(new_tokens),
                            )
                            .map_or(0.0, |c| c.ratio)
                    })
                    .collect();
                masked |= mask_rule_ratios(&mut ratios, &extra, side.as_ref());
                let (adjusted, kept) = itl_limit_costs(&extra, &ratios);
                budget.record_route(
                    forecast.now(),
                    kept,
                    ratios.iter().filter(|x| **x > 1.0).count() as u64,
                );
                adjusted
            }
            None => extra,
        };
        if masked && let Some(p) = pool {
            p.record_override(Override::RuleMasked);
        }
        // Diagnostics: DYNAMO_FORECAST_ROUTE_TRACE=<substring>[,<substring>...] prints the scored rows of
        // matching request ids to stderr.
        if let Some(want) = std::env::var_os("DYNAMO_FORECAST_ROUTE_TRACE")
            && let Some(id) = context.request_id()
            && want
                .to_string_lossy()
                .split(',')
                .any(|w| !w.is_empty() && id.contains(w))
        {
            let reserve_dbg = reserve
                .as_ref()
                .map(|r| (r.workers.clone(), r.mine, r.wrong_side));
            eprintln!(
                "forecast_route_trace id={id} class={class} home={home:?} stayed={stayed:?} reserve={reserve_dbg:?} rows={rows:?} extra={extra:?}"
            );
        }
        match stayed {
            Some(true) => state.stayed_home.fetch_add(1, Ordering::Relaxed),
            Some(false) => state.moved.fetch_add(1, Ordering::Relaxed),
            None => 0,
        };
        for (cost, x) in costs.iter_mut().zip(extra) {
            *cost = weight * x;
        }
        if logging && let Some(id) = context.request_id() {
            let rows_out = rows
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    let decision = WhatIf {
                        class: class.clone(),
                        worker: r.worker,
                        new_tokens: (isl - r.hit_tokens).max(0.0) as u64,
                        prefix_tokens: r.hit_tokens as u64,
                    };
                    let priced = priced_details
                        .iter()
                        .find(|(w, _)| *w == r.worker)
                        .map(|(_, d)| d.clone());
                    PredictRow {
                        worker: r.worker,
                        new_tokens: decision.new_tokens,
                        prefix_tokens: decision.prefix_tokens,
                        priced: priced.is_some(),
                        detail: priced.or_else(|| forecast.what_if_detail(&decision)),
                        itl_ratio: logged_ratios.as_ref().and_then(|v| v.get(i).copied()),
                        cost: costs.get(i).copied().unwrap_or(0.0),
                    }
                })
                .collect();
            state.stash_prediction(
                id,
                PendingPrediction {
                    t: forecast.now().as_secs_f64(),
                    session: session.clone(),
                    class: class.clone(),
                    isl: isl as u64,
                    rows: rows_out,
                },
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(id, 0)
    }

    fn row(id: u64, hit: f64, base: f64) -> Row {
        Row {
            worker: w(id),
            hit_tokens: hit,
            base_cost: base,
            active_tokens: base,
            over: false,
            room: true,
        }
    }

    #[test]
    fn lower_tier_hits_count_own_host_and_fetch_at_ninety_percent() {
        // Device only.
        assert_eq!(candidate_hit_tokens(Some((10.0, 0.0)), None, 64.0), 640.0);
        // HBM evicted, own DRAM holds 100 blocks: 90 blocks' worth.
        assert_eq!(
            candidate_hit_tokens(Some((0.0, 100.0)), None, 64.0),
            0.9 * 100.0 * 64.0
        );
        // A peer holds more than device plus own host: the fetch decides.
        assert_eq!(
            candidate_hit_tokens(Some((5.0, 20.0)), Some(200), 64.0),
            0.9 * 200.0 * 64.0
        );
        // A full device prefix is never discounted below itself.
        assert_eq!(
            candidate_hit_tokens(Some((100.0, 1.0)), Some(100), 64.0),
            100.0 * 64.0
        );
        assert_eq!(candidate_hit_tokens(None, None, 64.0), 0.0);
    }

    #[test]
    fn only_the_cheapest_and_the_home_are_forecast() {
        let rows = [
            row(0, 0.0, 1.0),
            row(1, 0.0, 2.0),
            row(2, 0.0, 3.0),
            row(3, 0.0, 9.0),
        ];
        let mut priced = Vec::new();
        let (costs, _) = route_costs(&rows, None, 2, 20_000.0, true, None, |r| {
            priced.push(r.worker.worker_id);
            Some((0.0, 0.0))
        });
        assert_eq!(priced, vec![0, 1]);
        assert_eq!(costs[2], EXCLUDED);
        assert_eq!(costs[3], EXCLUDED);
    }

    #[test]
    fn equal_default_costs_pick_the_same_candidates_in_any_listing_order() {
        let rows = [
            row(3, 0.0, 1.0),
            row(1, 0.0, 1.0),
            row(2, 0.0, 1.0),
            row(0, 0.0, 1.0),
        ];
        let mut reversed = rows;
        reversed.reverse();
        let priced_ids = |rows: &[Row]| {
            let mut priced = Vec::new();
            route_costs(rows, None, 2, 20_000.0, true, None, |r| {
                priced.push(r.worker.worker_id);
                Some((0.0, 0.0))
            });
            priced
        };
        assert_eq!(priced_ids(&rows), vec![0, 1]);
        assert_eq!(priced_ids(&reversed), vec![0, 1]);
    }

    #[test]
    fn predicted_misses_dominate_then_own_ttft() {
        let rows = [row(0, 0.0, 1.0), row(1, 0.0, 2.0), row(2, 0.0, 3.0)];
        let (costs, stayed) = route_costs(&rows, None, 3, 20_000.0, true, None, |r| {
            match r.worker.worker_id {
                0 => Some((2.0, 0.1)),
                1 => Some((0.0, 5.0)),
                _ => Some((0.0, 1.0)),
            }
        });
        assert!(stayed.is_none());
        assert!(costs[2] < costs[1] && costs[1] < costs[0]);
    }

    #[test]
    fn the_session_stays_home_unless_a_move_avoids_misses() {
        let rows = [
            row(0, 50_000.0, 5.0),
            row(1, 0.0, 1.0),
            row(2, 0.0, 2.0),
            row(3, 0.0, 3.0),
        ];
        // No misses anywhere: home wins even though it is not among the cheapest.
        let (costs, stayed) = route_costs(&rows, Some(w(0)), 3, 20_000.0, true, None, |_| {
            Some((0.0, 0.0))
        });
        assert_eq!(stayed, Some(true));
        assert!(costs[0] < EXCLUDED && costs[1..].iter().all(|c| *c == EXCLUDED));
        // Home has KV room and the others lose all of its cache: moving costs 1000, so it stays.
        let (_, stayed) = route_costs(&rows, Some(w(0)), 3, 20_000.0, true, None, |r| {
            Some((if r.worker.worker_id == 0 { 3.0 } else { 0.0 }, 0.0))
        });
        assert_eq!(stayed, Some(true));
        // Home full: the move rule lifts, and the cheaper forecast wins.
        let mut full = rows;
        full[0].room = false;
        let (costs, stayed) = route_costs(&full, Some(w(0)), 3, 20_000.0, true, None, |r| {
            Some((if r.worker.worker_id == 0 { 3.0 } else { 0.0 }, 0.0))
        });
        assert_eq!(stayed, Some(false));
        assert!(costs[1] < costs[0]);
    }

    fn sided(
        rows: &[Row],
        home: Option<WorkerWithDpRank>,
        side: &PoolSide,
        misses: impl Fn(u64) -> f64,
    ) -> (Vec<f64>, Option<bool>, Vec<u64>, Vec<String>) {
        let mut priced = Vec::new();
        let (costs, stayed, notes) =
            route_costs_sided(rows, home, 3, 20_000.0, true, None, Some(side), |r| {
                priced.push(r.worker.worker_id);
                Some((misses(r.worker.worker_id), 0.0))
            });
        let notes = notes.iter().map(|n| format!("{n:?}")).collect();
        (costs, stayed, priced, notes)
    }

    #[test]
    fn a_hard_pool_side_prices_only_workers_outside_the_pool() {
        // Workers 0-2 are the pool and the cheapest; 3-5 are outside it.
        let rows: Vec<Row> = (0..6).map(|i| row(i, 0.0, 1.0 + i as f64)).collect();
        let side = PoolSide {
            own: vec![false, false, false, true, true, true],
            hard: true,
        };
        let (costs, stayed, priced, notes) = sided(&rows, None, &side, |_| 0.0);
        assert_eq!(priced, vec![3, 4, 5]);
        assert!(costs[..3].iter().all(|c| *c == EXCLUDED));
        assert!(costs[3..].iter().all(|c| *c < EXCLUDED));
        assert_eq!((stayed, notes), (None, vec!["CutAllPool".to_owned()]));
    }

    #[test]
    fn a_home_in_the_pool_is_left_and_misses_cannot_buy_the_pool() {
        // The home (0) is a pool worker holding the session's cache; outside, 2 and 3 predict
        // many misses. Without a side the session stays home; with a hard side it moves out.
        let rows = [
            row(0, 50_000.0, 1.0),
            row(1, 0.0, 2.0),
            row(2, 0.0, 3.0),
            row(3, 0.0, 4.0),
        ];
        let misses = |id: u64| if id >= 2 { 500.0 } else { 0.0 };
        let (_, stayed) = route_costs(&rows, Some(w(0)), 3, 20_000.0, true, None, |r| {
            Some((misses(r.worker.worker_id), 0.0))
        });
        assert_eq!(stayed, Some(true));
        let side = PoolSide {
            own: vec![false, false, true, true],
            hard: true,
        };
        let (costs, stayed, priced, notes) = sided(&rows, Some(w(0)), &side, misses);
        assert_eq!(stayed, None);
        assert_eq!(priced, vec![2, 3]);
        assert!(costs[0] == EXCLUDED && costs[1] == EXCLUDED && costs[2] < EXCLUDED);
        assert_eq!(
            notes,
            vec!["CutHadPool".to_owned(), "HomeInPool".to_owned()]
        );
    }

    #[test]
    fn a_soft_side_adds_priority_its_cheapest_pool_worker() {
        // Priority's cheapest three are outside the pool (3-5 cheapest); its pool is 0-1.
        let rows: Vec<Row> = (0..6).map(|i| row(i, 0.0, 10.0 - i as f64)).collect();
        let side = PoolSide {
            own: vec![true, true, false, false, false, false],
            hard: false,
        };
        let (costs, _, priced, notes) = sided(&rows, None, &side, |_| 0.0);
        assert_eq!(priced, vec![5, 4, 3, 1]);
        assert!(costs[1] < EXCLUDED && costs[0] == EXCLUDED);
        assert_eq!(notes, vec!["PriorityPoolAdded".to_owned()]);
        // Without a side nothing changes.
        let (costs_plain, _) =
            route_costs(&rows, None, 3, 20_000.0, true, None, |_| Some((0.0, 0.0)));
        assert_eq!(costs_plain[1], EXCLUDED);
    }

    #[test]
    fn rule_ratios_cannot_let_the_pool_back_in() {
        let side = PoolSide {
            own: vec![false, true, true],
            hard: true,
        };
        // Pool worker 0 keeps the limit, the routed own row 1 does not: masked, counted.
        let costs = [EXCLUDED, 5.0, EXCLUDED];
        let mut ratios = vec![0.5, 2.0, 3.0];
        assert!(mask_rule_ratios(&mut ratios, &costs, Some(&side)));
        let (out, kept) = itl_limit_costs(&costs, &ratios);
        assert!(!kept && out[0] >= EXCLUDED && out[1] < EXCLUDED);
        // A soft side (priority) or none leaves the ratios alone.
        let soft = PoolSide {
            hard: false,
            ..side.clone()
        };
        let mut ratios = vec![0.5, 2.0, 3.0];
        assert!(!mask_rule_ratios(&mut ratios, &costs, Some(&soft)));
        assert!(!mask_rule_ratios(&mut ratios, &costs, None));
        assert_eq!(ratios, vec![0.5, 2.0, 3.0]);
    }

    #[test]
    fn batch_ignores_its_own_ttft_and_overflow_costs_five_misses() {
        let mut rows = [row(0, 0.0, 1.0), row(1, 0.0, 2.0)];
        let (costs, _) = route_costs(&rows, None, 3, 20_000.0, false, None, |r| {
            Some((0.0, if r.worker.worker_id == 0 { 30.0 } else { 0.0 }))
        });
        assert_eq!(costs[0], costs[1]);
        rows[0].over = true;
        let (costs, _) = route_costs(&rows, None, 3, 20_000.0, true, None, |_| Some((0.0, 0.0)));
        assert_eq!(costs[0] - costs[1], 5.0 * MISS_COST);
    }

    #[test]
    fn the_soft_reserve_steers_each_class_to_its_side() {
        let rows = [
            row(0, 0.0, 1.0),
            row(1, 0.0, 2.0),
            row(2, 0.0, 3.0),
            row(3, 0.0, 9.0),
        ];
        let reserved = |mine| Reserve {
            workers: vec![w(2), w(3)],
            mine,
            wrong_side: if mine { 3.0 } else { 10.0 },
        };
        // Priority: its cheapest reserved worker (2) joins and wins.
        let r = reserved(true);
        let (costs, _) = route_costs(&rows, None, 2, 20_000.0, true, Some(&r), |_| {
            Some((0.0, 0.0))
        });
        let best = (0..4)
            .min_by(|a, b| costs[*a].total_cmp(&costs[*b]))
            .unwrap();
        assert_eq!(best, 2);
        assert_eq!(costs[3], EXCLUDED);
        // A lower class avoids the reserved workers.
        let r = reserved(false);
        let (costs, _) = route_costs(&rows, None, 3, 20_000.0, true, Some(&r), |_| {
            Some((0.0, 0.0))
        });
        assert!(costs[0] < costs[2] && costs[1] < costs[2]);
    }

    #[test]
    fn the_soft_reserve_turns_on_near_target_and_off_well_above() {
        let state = ForecastRouteState::new(ForecastRouteConfig {
            soft_reserve: Some(SoftReserveConfig::default()),
            ..Default::default()
        });
        state.reading(Some(0.95), Some(0.9), 0.3);
        assert!(state.reserved(8).is_none(), "0.95 is over target + 0.03");
        state.reading(Some(0.92), Some(0.9), 0.3);
        // ceil(8 × 0.3 × 1.25) = 3.
        assert_eq!(state.reserved(8).map(|r| r.1), Some(3));
        state.reading(Some(0.95), Some(0.9), 0.3);
        assert!(state.reserved(8).is_some(), "stays on inside the band");
        state.reading(Some(0.97), Some(0.9), 0.3);
        assert!(state.reserved(8).is_none());
        assert_eq!(state.reserve_stats().1, 2);
        // Off without a soft reserve configured.
        let off = ForecastRouteState::new(ForecastRouteConfig::default());
        off.reading(Some(0.5), Some(0.9), 0.3);
        assert!(off.reserved(8).is_none());
    }

    #[test]
    fn min_unreserved_zero_lets_the_soft_reserve_cover_every_worker() {
        let reserve = |min_unreserved| {
            let state = ForecastRouteState::new(ForecastRouteConfig {
                soft_reserve: Some(SoftReserveConfig {
                    min_unreserved,
                    ..SoftReserveConfig::default()
                }),
                ..Default::default()
            });
            // Priority-only traffic: ceil(4 x 1.0 x 1.25) = 5.
            state.reading(Some(0.5), Some(0.9), 1.0);
            state.reserved(4).map(|r| r.1)
        };
        assert_eq!(reserve(1), Some(3));
        assert_eq!(reserve(0), Some(4));
    }

    #[test]
    fn the_itl_limit_excludes_breaking_candidates_and_falls_back_to_the_least_bad() {
        // Candidate 1 breaks a limit, 0 keeps it: 1 is excluded.
        let (c, kept) = itl_limit_costs(&[5.0, 1.0], &[0.8, 1.4]);
        assert!(kept && c[0] == 5.0 && c[1] >= EXCLUDED);
        // The route's own candidates all break; one it left out keeps every limit: let it in.
        let (c, kept) = itl_limit_costs(&[0.0, EXCLUDED], &[1.2, 0.5]);
        assert!(kept && c[0] >= EXCLUDED && c[1] == 0.0);
        // All break: the one going over least wins; excluded rows stay excluded.
        let (c, kept) = itl_limit_costs(&[0.0, 0.0, EXCLUDED], &[1.5, 1.1, 0.9 + 9.0]);
        assert!(!kept && c[1] < c[0] && c[2] >= EXCLUDED);
    }

    #[test]
    fn requests_are_tracked_until_they_finish() {
        let state = ForecastRouteState::new(ForecastRouteConfig::default());
        state.arrived("r", "s", None, "standard", false);
        assert!(state.requests.lock().contains_key("r"));
        state.finished("r");
        assert!(!state.requests.lock().contains_key("r"));
        assert_eq!(state.home_of("s", None), None);
    }

    /// f49d `DYNAMO_PREDICT_LOG`: the line for the worker a request went to carries the forecast's
    /// own price there; without a kept selection there is no line; the env var unset, no sink.
    #[test]
    fn prediction_line_carries_the_forecasts_own_price_on_the_chosen_worker() {
        use crate::clock::{ManualClock, Time};
        use crate::config::ServicePolicy;
        use crate::forecast::{ForecastConfig, Observation, WorkerSample};
        if std::env::var_os("DYNAMO_PREDICT_LOG").is_none() {
            assert!(predict_sink().is_none());
        }
        let clock = Arc::new(ManualClock::new());
        let policy = ServicePolicy {
            forecast: ForecastConfig {
                enabled: true,
                window_s: 30.0,
                price_itl_stall: true,
                ..ForecastConfig::default()
            },
            ..Default::default()
        };
        let model = policy.forecast.step_model().unwrap();
        let (forecast, feed) = Forecast::new(&policy, model, clock.clone(), 16);
        let (a, b) = (w(1), w(2));
        feed.send(Observation::Workers {
            at: Time::from_secs_f64(1.0),
            rows: [(a, 4096, 4), (b, 0, 1)]
                .into_iter()
                .map(|(worker, queued, n)| WorkerSample {
                    worker,
                    active_prefill_tokens: queued,
                    active_decode_blocks: n * 256,
                    active_requests: n,
                    max_batched_tokens: None,
                })
                .collect(),
        });
        feed.send(Observation::Arrived {
            request_id: "d".into(),
            session: "d".into(),
            class: "priority".into(),
            at: Time::from_secs_f64(1.0),
            prompt_tokens: 100,
        });
        feed.send(Observation::FirstToken {
            request_id: "d".into(),
            worker: a,
            at: Time::from_secs_f64(1.5),
        });
        let decision = |worker| WhatIf {
            class: "standard".into(),
            worker,
            new_tokens: 3000,
            prefix_tokens: 1000,
        };
        let state = ForecastRouteState::new(ForecastRouteConfig::default());
        let rows: Vec<PredictRow> = [a, b]
            .into_iter()
            .map(|worker| PredictRow {
                worker,
                new_tokens: 3000,
                prefix_tokens: 1000,
                priced: worker == a,
                detail: forecast.what_if_detail(&decision(worker)),
                itl_ratio: Some(0.25),
                cost: 7.0,
            })
            .collect();
        state.stash_prediction(
            "r\"1",
            PendingPrediction {
                t: 2.5,
                session: "s".into(),
                class: "standard".into(),
                isl: 4000,
                rows,
            },
        );
        assert_eq!(state.take_prediction_line("other", a), None);
        let line = state.take_prediction_line("r\"1", a).unwrap();
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        let own = forecast.what_if(&decision(a)).unwrap();
        let wf = forecast.worker(a, std::time::Duration::ZERO).unwrap();
        let f = |k: &str| v[k].as_f64().unwrap();
        // The line's decimal text parses back to the same value (to the last bit or so).
        let same = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs().max(1.0);
        assert_eq!(v["request_id"], "r\"1");
        assert_eq!(
            (v["session"].as_str(), v["class"].as_str()),
            (Some("s"), Some("standard"))
        );
        assert_eq!(
            (v["worker"].as_u64(), v["dp_rank"].as_u64()),
            (Some(1), Some(0))
        );
        assert_eq!(
            (v["isl"].as_u64(), v["candidates"].as_u64()),
            (Some(4000), Some(2))
        );
        assert_eq!(
            (v["priced"].as_bool(), v["in_rows"].as_bool()),
            (Some(true), Some(true))
        );
        assert_eq!(f("t"), 2.5);
        assert!(same(f("predicted_ttft_s"), own.ttft_s));
        assert!(same(f("candidate_itl_after_s"), own.itl_after_s));
        assert!(same(f("routing_misses"), own.routing_misses()));
        assert_eq!(v["pushed_over"].as_u64(), Some(u64::from(own.pushed_over)));
        assert_eq!(f("window_s"), 30.0);
        assert_eq!(v["decode_batch"].as_u64(), Some(wf.decode_batch));
        assert!(same(f("prefill_backlog_s"), wf.prefill_backlog_s));
        assert_eq!((f("itl_ratio"), f("cost")), (0.25, 7.0));
        assert_eq!(v["decoding"]["priority"].as_u64(), Some(1));
        let detail = forecast.what_if_detail(&decision(a)).unwrap();
        assert_eq!(detail.itl_by_class[0].0, "priority");
        assert!(same(
            v["predicted_itl_s"]["priority"].as_f64().unwrap(),
            detail.itl_by_class[0].1
        ));
        // Taken once; a request finished without a send leaves nothing behind.
        assert_eq!(state.take_prediction_line("r\"1", a), None);
        state.stash_prediction(
            "x",
            PendingPrediction {
                t: 0.0,
                session: "s".into(),
                class: "standard".into(),
                isl: 1,
                rows: Vec::new(),
            },
        );
        state.finished("x");
        assert_eq!(state.take_prediction_line("x", a), None);
        // Sent to a worker outside the rows: the line says so, with nulls.
        state.stash_prediction(
            "y",
            PendingPrediction {
                t: 0.0,
                session: "s".into(),
                class: "standard".into(),
                isl: 1,
                rows: Vec::new(),
            },
        );
        let v: serde_json::Value =
            serde_json::from_str(&state.take_prediction_line("y", a).unwrap()).unwrap();
        assert_eq!(v["in_rows"].as_bool(), Some(false));
        assert!(v["predicted_ttft_s"].is_null());
    }

    fn last_sent_state() -> ForecastRouteState {
        ForecastRouteState::new(ForecastRouteConfig {
            home: HomeSource::LastSent,
            ..ForecastRouteConfig::default()
        })
    }

    #[test]
    fn home_source_defaults_to_the_index_and_parses() {
        assert_eq!(ForecastRouteConfig::default().home, HomeSource::Index);
        let c: ForecastRouteConfig = serde_json::from_str(r#"{"home": "last_sent"}"#).unwrap();
        assert_eq!(c.home, HomeSource::LastSent);
        let c: ForecastRouteConfig = serde_json::from_str(r#"{"home": "none"}"#).unwrap();
        assert_eq!(c.home, HomeSource::None);
        assert!(serde_json::from_str::<ForecastRouteConfig>(r#"{"home": "sticky"}"#).is_err());
    }

    #[test]
    fn soft_home_is_the_last_worker_while_it_holds_a_prefix() {
        let rows = [row(0, 0.0, 1.0), row(1, 640.0, 2.0), row(2, 0.0, 3.0)];
        // The session's own last worker, holding a prefix: home.
        assert_eq!(soft_home(Some(w(1)), None, &rows), (Some(w(1)), false));
        // No own worker yet (a subagent's first turn): the parent's.
        assert_eq!(soft_home(None, Some(w(1)), &rows), (Some(w(1)), false));
        // The session's own worker wins over the parent's.
        assert_eq!(
            soft_home(Some(w(1)), Some(w(2)), &rows),
            (Some(w(1)), false)
        );
        // Nothing remembered: no home, nothing stale.
        assert_eq!(soft_home(None, None, &rows), (None, false));
    }

    #[test]
    fn a_stale_soft_home_falls_back_to_the_router() {
        let rows = [row(0, 0.0, 1.0), row(1, 640.0, 2.0)];
        // The last worker holds none of this request's prefix (evicted): stale, no home.
        assert_eq!(soft_home(Some(w(0)), None, &rows), (None, true));
        // The last worker is not a candidate (removed or filtered out): stale, no home.
        assert_eq!(soft_home(Some(w(7)), None, &rows), (None, true));
        // With no home the route's costs are those of a request without one.
        let price = |_: &Row| Some((0.0, 0.0));
        let (with_none, stayed) = route_costs(&rows, None, 3, 20_000.0, true, None, price);
        assert_eq!(stayed, None);
        let (with_home, stayed) = route_costs(&rows, Some(w(1)), 3, 20_000.0, true, None, price);
        assert_eq!(stayed, Some(true));
        assert_ne!(with_none, with_home);
    }

    #[test]
    fn last_sent_remembers_the_worker_and_forgets_at_session_end() {
        let state = last_sent_state();
        let rows = [row(0, 0.0, 1.0), row(3, 64.0, 2.0)];
        state.arrived("r1", "s", None, "standard", false);
        state.sent("r1", w(3));
        state.finished("r1");
        assert_eq!(state.home_for("s", None, &rows), Some(w(3)));
        // A subagent's first turn follows its parent.
        state.arrived("c1", "s:a1", Some("s"), "standard", false);
        assert_eq!(state.home_for("s:a1", Some("s"), &rows), Some(w(3)));
        // The next turn goes elsewhere: that is the home now.
        state.arrived("r2", "s", None, "standard", false);
        state.sent("r2", w(0));
        assert_eq!(
            state.home_for("s", None, &rows),
            None,
            "w0 holds no prefix: stale"
        );
        assert_eq!(state.soft_home_stats().1, 1);
        // The session's final request forgets it.
        state.finished("r2");
        state.arrived("r3", "s", None, "standard", true);
        state.sent("r3", w(3));
        state.finished("r3");
        assert_eq!(state.home_for("s", None, &rows), None);
        assert_eq!(state.soft_home_stats().2, 0);
    }

    #[test]
    fn home_max_load_drops_a_soft_home_on_the_hot_spot() {
        // Active load (the helper's base cost): w0 9, w1 1, w2 2; mean 4.
        let rows = [row(0, 64.0, 9.0), row(1, 64.0, 1.0), row(2, 0.0, 2.0)];
        assert!(home_overloaded(&rows, w(0), 2.0), "9 > 2 x 4");
        assert!(!home_overloaded(&rows, w(0), 2.5), "9 < 2.5 x 4");
        assert!(!home_overloaded(&rows, w(1), 1.0));
        assert!(!home_overloaded(&rows, w(7), 1.0), "not a candidate");
        let idle = [row(0, 64.0, 0.0), row(1, 0.0, 0.0)];
        assert!(
            !home_overloaded(&idle, w(0), 1.0),
            "an idle fleet is never over"
        );
        assert!(!home_overloaded(&[], w(0), 1.0));
        let state = ForecastRouteState::new(ForecastRouteConfig {
            home: HomeSource::LastSent,
            home_max_load: Some(1.5),
            ..ForecastRouteConfig::default()
        });
        state.arrived("r", "s", None, "standard", false);
        state.sent("r", w(0));
        assert_eq!(
            state.home_for("s", None, &rows),
            None,
            "w0 over 1.5x: dropped"
        );
        state.sent("r", w(1));
        assert_eq!(state.home_for("s", None, &rows), Some(w(1)));
        assert_eq!(state.dropped_home_stats(), (1, 0));
    }

    #[test]
    fn a_late_home_is_dropped_only_with_an_alternative_when_asked() {
        let rows = [row(0, 64.0, 9.0), row(1, 64.0, 1.0), row(2, 0.0, 2.0)];
        let all_late = |_: &Row| true;
        let only_w0_late = |r: &Row| r.worker == w(0);
        let none_late = |_: &Row| false;
        // Late home, others fine: dropped either way.
        assert!(late_home_dropped(&rows, w(0), false, only_w0_late));
        assert!(late_home_dropped(&rows, w(0), true, only_w0_late));
        // Every candidate late (a fleet-wide backlog): dropped only without the guard.
        assert!(late_home_dropped(&rows, w(0), false, all_late));
        assert!(!late_home_dropped(&rows, w(0), true, all_late));
        // A home that is not late, or not a candidate, stays.
        assert!(!late_home_dropped(&rows, w(0), false, none_late));
        assert!(!late_home_dropped(&rows, w(7), false, all_late));
    }

    #[test]
    fn home_max_load_off_keeps_the_soft_home() {
        let rows = [row(0, 64.0, 9.0), row(1, 64.0, 1.0)];
        let state = last_sent_state();
        assert_eq!(state.config().home_max_load, None);
        assert!(!state.config().home_drop_if_late);
        state.arrived("r", "s", None, "standard", false);
        state.sent("r", w(0));
        assert_eq!(state.home_for("s", None, &rows), Some(w(0)));
        assert_eq!(state.dropped_home_stats(), (0, 0));
        let c: ForecastRouteConfig = serde_json::from_str(
            r#"{"home": "last_sent", "home_max_load": 2.0, "home_drop_if_late": true}"#,
        )
        .unwrap();
        assert_eq!(c.home_max_load, Some(2.0));
        assert!(c.home_drop_if_late);
        assert!(!c.home_late_needs_alternative, "off unless given");
        let c: ForecastRouteConfig = serde_json::from_str(
            r#"{"home": "last_sent", "home_drop_if_late": true, "home_late_needs_alternative": true}"#,
        )
        .unwrap();
        assert!(c.home_late_needs_alternative);
    }

    #[test]
    fn index_and_none_sources_ignore_sent_events() {
        let rows = [row(3, 64.0, 2.0)];
        for home in [HomeSource::Index, HomeSource::None] {
            let state = ForecastRouteState::new(ForecastRouteConfig {
                home,
                ..ForecastRouteConfig::default()
            });
            state.arrived("r", "s", None, "standard", false);
            state.sent("r", w(3));
            assert!(state.last_sent.lock().is_empty());
            // No index attached: the index source knows no home either.
            assert_eq!(state.home_for("s", None, &rows), None);
        }
    }
}
