// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin adapters from the router's plugin traits (PR #18443, `dynamo_kv_router::plugins::prelude`)
//! to the [`Scheduler`].
//!
//! - [`SessionSchedulerClassifier`] is the `RequestClassifier`: `classify` records the turn and
//!   returns at once; `on_event` settles completions.
//! - [`SessionSchedulerGate`] is its `DispatchGate`: `check` answers from the session table as a
//!   read; `dispatched` commits the admission.
//!
//! Only requests that carry both a session context and a request id are scheduled; the rest pass
//! through untouched.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use dynamo_kv_router::plugins::prelude::*;
use dynamo_kv_router::plugins::request_classifier::{SessionIndexReader, SessionTier};
use parking_lot::Mutex;
use thiserror::Error;

use crate::admit::{Decision, GateAnswer};
use crate::clock::{Clock, ContextClock, Time};
use crate::config::{Mode, ServicePolicy, WaitLimitAction};
use crate::forecast::{Forecast, ForecastFeed, Observation, StepModel, WorkerSample};
use crate::scheduler::{GateWake, HeadFacts, RequestAttributes, Scheduler};
use crate::session::{SessionState, Signals};

/// The `request_classifier.type` this crate registers.
pub const SESSION_SCHEDULER_TYPE: &str = "session-scheduler";

/// How long one HBM capacity reading serves gate checks before it is read again from the host's
/// worker list (which allocates).
const CAPACITY_TTL: Duration = Duration::from_millis(100);

/// Register the classifier provider.
pub fn register(registry: &mut RouterPluginRegistry) -> Result<(), RequestClassifierRegistryError> {
    registry.register_request_classifier(SESSION_SCHEDULER_TYPE, Arc::new(classifier_provider))
}

fn classifier_provider(
    parameters: &RequestClassifierParameters,
) -> Result<RequestClassifierFactory, RequestClassifierProviderError> {
    let policy: ServicePolicy = parameters.deserialize()?;
    policy
        .validate()
        .map_err(|error| RequestClassifierProviderError::new(error.to_string()))?;
    // `DYNAMO_SESSION_SCHEDULER_CLOCK_SCALE` (unset by default): scheduler time runs that many
    // times faster than the host's, for a live router in front of sped-up mock engines.
    let clock_scale =
        crate::clock::clock_scale_from_env().map_err(RequestClassifierProviderError::new)?;
    // Built once: an AISimulate model loads its step data at construction.
    let step_model = if policy.forecast.enabled {
        Some(
            policy
                .forecast
                .step_model()
                .map_err(|error| RequestClassifierProviderError::new(error.to_string()))?,
        )
    } else {
        None
    };
    Ok(Arc::new(move |context| {
        let block_size = context.block_size();
        let session_reader = context.session_index();
        // The context's clock: tokio time in a live router, virtual time in offline replay.
        let clock_context = context.clone();
        let host_clock = ContextClock::new(move || clock_context.now());
        let clock: Arc<dyn Clock> = match clock_scale {
            Some(scale) => Arc::new(crate::clock::ScaledClock::new(Box::new(host_clock), scale)),
            None => Arc::new(host_clock),
        };
        let mut classifier =
            SessionSchedulerClassifier::new(policy.clone(), clock, capacity_source(context));
        if let Some(model) = &step_model {
            classifier = classifier.with_step_model(Arc::clone(model));
        }
        if let Some(reader) = session_reader.clone() {
            classifier = classifier.with_session_index(Arc::new(ContextSessionIndex::new(reader)));
        }
        let classifier = classifier.with_block_size(block_size);
        // Forecast routing (`route.rs`) reads this classifier's forecast and the session index.
        if let Some(route) = crate::route::installed("default") {
            route.attach(classifier.forecast(), session_reader);
        }
        Box::new(classifier)
    }))
}

/// Agent header naming the tenant. Untrusted until the frontend sets it (ledger L54); replay
/// sets it from a seed.
pub const HEADER_TENANT: &str = "x-dynamo-tenant";
/// Client signal: expected output tokens of this request.
pub const HEADER_HINT_OSL: &str = "x-dynamo-hint-osl";
/// Client signal: seconds the client expects to think before its next turn.
pub const HEADER_HINT_NEXT_THINK_S: &str = "x-dynamo-hint-next-think-s";
/// Client signal: new prompt tokens the client expects its next turn to add.
pub const HEADER_HINT_NEXT_TOKENS: &str = "x-dynamo-hint-next-tokens";
/// Client signal: the largest context the client expects the session to reach from here (F22;
/// read by the gate only with `admission.lifetime`).
pub const HEADER_HINT_PEAK_TOKENS: &str = "x-dynamo-hint-peak-tokens";
/// Client signal: `interactive` (a person waits on the turns) or `agent` (F35; proposed as
/// `nvext.agent_hints.session_kind`). Not read by any policy yet; the session log records it.
pub const HEADER_HINT_SESSION_KIND: &str = "x-dynamo-hint-session-kind";
/// The user a session belongs to; a user can run several sessions at once (F35; proposed as
/// `nvext.agent_context.user_id`). Distinct from the tenant, which groups users for shares and
/// quotas. Untrusted until the frontend sets it from the credential (ledger L55).
pub const HEADER_USER_ID: &str = "x-dynamo-user-id";
/// Client signal: the role of this turn's last message, `user` (a person's new message: the
/// previous turn handed back to the user, a stop point) or `tool` (a tool result; F35).
pub const HEADER_HINT_LAST_ROLE: &str = "x-dynamo-hint-last-role";

/// The tenant, client signals and session end marker a request carries: the session end from
/// `nvext.agent_context.session_final` (header `x-dynamo-session-final`), the rest from agent
/// headers. Malformed values are ignored, as untrusted input must be.
fn request_attributes(session: &SessionContext) -> RequestAttributes {
    let headers = session.agent_headers();
    let first = |name: &str| headers.get(name).and_then(|values| values.first());
    let attributes = RequestAttributes {
        tenant: first(HEADER_TENANT).cloned(),
        signals: Signals {
            next_think_s: first(HEADER_HINT_NEXT_THINK_S)
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|v| v.is_finite() && *v >= 0.0),
            osl_tokens: first(HEADER_HINT_OSL).and_then(|v| v.parse().ok()),
            next_turn_tokens: first(HEADER_HINT_NEXT_TOKENS).and_then(|v| v.parse().ok()),
            peak_tokens: first(HEADER_HINT_PEAK_TOKENS)
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|v| v.is_finite() && *v >= 0.0)
                .map(|v| v as u64),
        },
        session_final: session.session_final() == Some(true),
        session_kind: first(HEADER_HINT_SESSION_KIND)
            .and_then(|v| crate::session::SessionKind::parse(v)),
        user: first(HEADER_USER_ID)
            .filter(|v| !v.is_empty() && v.len() <= 256)
            .cloned(),
        // The frontend already classifies each request by its last causal message
        // (`input_trigger`: a new user message, a tool result or other); the client header says
        // the same where the frontend cannot see the messages (replay).
        next_is_user: session.input_trigger() == Some(WorkerSelectionInputTrigger::UserMessage)
            || first(HEADER_HINT_LAST_ROLE).is_some_and(|v| v.trim().eq_ignore_ascii_case("user")),
        // A user message or a tool result, said by the frontend or the client: the stop gap
        // does not decide (an `Other` trigger leaves it to the gap).
        next_known: matches!(
            session.input_trigger(),
            Some(
                WorkerSelectionInputTrigger::UserMessage | WorkerSelectionInputTrigger::ToolResult
            )
        ) || first(HEADER_HINT_LAST_ROLE).is_some_and(|v| {
            let v = v.trim();
            v.eq_ignore_ascii_case("user") || v.eq_ignore_ascii_case("tool")
        }),
    };
    tracing::trace!(
        session = session.session_id(),
        parent = session.parent_session_id(),
        session_final = attributes.session_final,
        user = attributes.user.as_deref(),
        next_think_s = attributes.signals.next_think_s,
        osl_tokens = attributes.signals.osl_tokens,
        next_turn_tokens = attributes.signals.next_turn_tokens,
        "session scheduler request attributes"
    );
    attributes
}

/// A typed refusal the router passes to the client as HTTP 429, with `seconds` as Retry-After
/// (`admission.retry_after_s`).
fn rate_limited(waiting: usize, limit: usize, seconds: f64) -> DynamoError {
    DynamoError::builder()
        .class(ErrorClass::RateLimited)
        .reason(ErrorReason::new("request.rate_limited").expect("a valid reason"))
        .diagnostic(format!(
            "session scheduler refused a new session: {waiting} sessions waiting, limit {limit}"
        ))
        .public_details(PublicDetails::RateLimit {
            limit: Some(limit as u64),
            remaining: Some(0),
            retry_after_s: Some(seconds.max(0.0).ceil() as u64),
        })
        .build()
}

/// Total HBM KV in tokens, and the number of workers.
pub type CapacitySource = Arc<dyn Fn() -> (u64, u64) + Send + Sync>;

/// HBM from the host's cached worker list: the sum of `total_kv_blocks × block_size`.
pub fn capacity_source(context: RequestClassifierContext) -> CapacitySource {
    let block_size = u64::from(context.block_size());
    Arc::new(move || {
        let workers = context.workers();
        let hbm = workers
            .iter()
            .filter_map(|w| w.total_kv_blocks())
            .fold(0u64, |sum, blocks| {
                sum.saturating_add(blocks.saturating_mul(block_size))
            });
        (hbm, workers.len() as u64)
    })
}

/// What the scheduler learns from the router's session prefix index.
///
/// With our router patch (ledger L47) `RequestClassifierContext::session_index` gives a
/// read-only handle on the router's `SessionPrefixIndexer` when the router tracks one
/// (`enable_session_prefix_index`); the classifier then reads it through [`ContextSessionIndex`].
/// Without it, [`NoSessionIndex`] knows nothing and the classifier uses the per-request
/// `WorkerCacheView` overlaps and the session table's own context counts.
pub trait SessionIndexView: Send + Sync {
    /// Blocks of `session` resident on (device, host, disk), if known.
    fn resident_blocks(&self, session: &str) -> Option<(u64, u64, u64)>;
    /// `session`'s residency per holder and tier (F34); empty when unknown.
    fn session_rows(&self, _session: &str) -> Vec<crate::accounting::Row> {
        Vec::new()
    }
    /// Unique resident blocks per holder and tier over every session (F34). Walks the index:
    /// for a periodic step only. Empty when unknown.
    fn unique_rows(&self) -> Vec<crate::accounting::Row> {
        Vec::new()
    }
}

fn accounting_row(
    row: dynamo_kv_router::plugins::request_classifier::SessionResidency,
) -> crate::accounting::Row {
    use crate::accounting::Tier;
    use dynamo_kv_router::plugins::request_classifier::SessionHolder;
    crate::accounting::Row {
        holder: match row.holder {
            SessionHolder::Worker(worker) => Some(worker),
            SessionHolder::SharedCache => None,
        },
        tier: match row.tier {
            SessionTier::Device => Tier::Device,
            SessionTier::Host => Tier::Host,
            SessionTier::Disk => Tier::Disk,
            SessionTier::External => Tier::External,
        },
        blocks: row.blocks,
    }
}

/// The stub [`SessionIndexView`]: knows nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoSessionIndex;

impl SessionIndexView for NoSessionIndex {
    fn resident_blocks(&self, _session: &str) -> Option<(u64, u64, u64)> {
        None
    }
}

/// The router's session prefix index, from the classifier context. Per tier, the deepest
/// prefix any one holder has (a session's prefix on two workers is the same blocks twice, so
/// holders are not summed). The external tier (a shared cache) is reported with disk.
pub struct ContextSessionIndex {
    reader: Arc<dyn SessionIndexReader>,
}

impl ContextSessionIndex {
    pub fn new(reader: Arc<dyn SessionIndexReader>) -> Self {
        Self { reader }
    }
}

impl SessionIndexView for ContextSessionIndex {
    fn resident_blocks(&self, session: &str) -> Option<(u64, u64, u64)> {
        let rows = self.reader.session_residency(session);
        if rows.is_empty() {
            return None;
        }
        let (mut device, mut host, mut disk) = (0u64, 0u64, 0u64);
        for row in rows {
            let slot = match row.tier {
                SessionTier::Device => &mut device,
                SessionTier::Host => &mut host,
                SessionTier::Disk | SessionTier::External => &mut disk,
            };
            *slot = (*slot).max(row.blocks);
        }
        Some((device, host, disk))
    }

    fn session_rows(&self, session: &str) -> Vec<crate::accounting::Row> {
        self.reader
            .session_residency(session)
            .into_iter()
            .map(accounting_row)
            .collect()
    }

    fn unique_rows(&self) -> Vec<crate::accounting::Row> {
        self.reader
            .unique_residency()
            .into_iter()
            .map(accounting_row)
            .collect()
    }
}

/// A classification the scheduler turns away.
#[derive(Debug, Error)]
pub enum SchedulerReject {
    /// The router maps a classifier error to an unclassified failure today, not 429 with
    /// `Retry-After`; see the README.
    #[error("session scheduler throttled a new session: {waiting} sessions waiting, limit {limit}")]
    Throttled { waiting: usize, limit: usize },
}

struct Inner {
    clock: Arc<dyn Clock>,
    capacity: CapacitySource,
    cached_capacity: Mutex<Option<(Time, u64, u64)>>,
    scheduler: Mutex<Scheduler>,
    recheck: OnceLock<DispatchRecheck>,
    index: Arc<dyn SessionIndexView>,
    block_size: u32,
    /// The step model for the forecast; `None` when the forecast is off.
    step_model: Option<Arc<dyn StepModel>>,
    forecast: Option<(Arc<Forecast>, ForecastFeed)>,
    /// When `classify` last sampled the workers' load for the forecast.
    last_sample: Mutex<Option<Time>>,
    /// The priority pool's shared state (`pool.rs`), when a `priority-pool` worker-selection
    /// policy named "default" was installed before this classifier was built. Fed in every mode,
    /// so the pool works with the scheduler off or in shadow.
    pool: Option<Arc<crate::pool::PoolState>>,
    /// The session affinity policy's shared state (`affinity.rs`), likewise.
    affinity: Option<Arc<crate::affinity::AffinityState>>,
    /// Each worker's load and capacity at the latest classify, sampled only when a gate rule
    /// reads it (shed by slack, the targeted `Go`, the running-request hold).
    kv_rows: Mutex<Vec<WorkerRow>>,
    /// The class-busy policy's shared state (`busy.rs`), like `pool`.
    busy: Option<Arc<crate::busy::BusyState>>,
    /// The forecast route's shared state (`route.rs`), like `pool`.
    route: Option<Arc<crate::route::ForecastRouteState>>,
    /// When rotation (`admission.rotate`) last looked.
    last_rotate: Mutex<Option<Time>>,
    /// Times of recent pauses, for `admission.rotate.max_per_min`.
    recent_pauses: Mutex<std::collections::VecDeque<Time>>,
    /// `DYNAMO_LOBBY_LOG=<path>` (F32): one JSON line per rotation pause, with its reason.
    lobby_log: Mutex<Option<std::io::BufWriter<std::fs::File>>>,
    /// The worker each session's latest request was sent to (for the targeted `Go`).
    last_worker: Mutex<std::collections::HashMap<String, WorkerWithDpRank>>,
    /// f49h `admission.aimd`: per-worker limits on one class's turns; `None` when off.
    aimd: Option<Mutex<crate::aimd::Aimd>>,
    /// Whether the session gate was on at the last [`Inner::wake_changed`] (the adaptive rule
    /// can switch it), when the host honours keyed holds.
    gate_was_on: Mutex<Option<bool>>,
    /// `DYNAMO_SURGE_LOG=<path>` (F22 diagnostics): surge admission's view per simulated minute.
    surge_log: Mutex<Option<SurgeLog>>,
    /// The two-resource gate's prefill meter (`admission.prefill_budget`, f42).
    prefill_meter: Mutex<PrefillMeter>,
    /// `DYNAMO_F42_LOG=<path>`: the prefill budget and the target loop per simulated minute.
    f42_log: Mutex<Option<F42Log>>,
    /// The signal-driven coordinator (`controller`), when configured.
    controller: Mutex<Option<crate::coordinator::Coordinator>>,
    /// The total budget's accounting from the session index (`admission.accounting`, F34).
    accounting: Mutex<AccountingState>,
}

/// What [`Inner::accounting_step`] keeps between steps.
#[derive(Default)]
struct AccountingState {
    last: Option<Time>,
    snapshot: Option<crate::accounting::Snapshot>,
    check: crate::accounting::ActionCheck,
    /// The latest step's report (one JSON object), for the controller log.
    report: Option<String>,
    /// `DYNAMO_ACCOUNTING_LOG=<path>`: one JSON line per step.
    log: Option<std::io::BufWriter<std::fs::File>>,
}

/// Surge admission's per-minute trace (`DYNAMO_SURGE_LOG`): the fleet's queued prefill seconds
/// and tokens as the gate saw them, and the lobby checks surge judged and held, by class.
struct SurgeLog {
    out: std::io::BufWriter<std::fs::File>,
    minute: Option<u64>,
    checks: u64,
    sum_backlog_s: f64,
    max_backlog_s: f64,
    sum_queued: f64,
    rate: f64,
}

impl SurgeLog {
    fn open() -> Option<Self> {
        let path = std::env::var_os("DYNAMO_SURGE_LOG")?;
        let file = std::fs::File::create(path).ok()?;
        Some(Self {
            out: std::io::BufWriter::new(file),
            minute: None,
            checks: 0,
            sum_backlog_s: 0.0,
            max_backlog_s: 0.0,
            sum_queued: 0.0,
            rate: 0.0,
        })
    }

    /// Record one gate check at `now`; at a new minute, write the last one with `counts`.
    fn record(
        &mut self,
        now: Time,
        backlog_s: f64,
        rate: f64,
        counts: impl FnOnce() -> std::collections::BTreeMap<String, (u64, u64)>,
    ) {
        use std::io::Write;
        let minute = (now.as_secs_f64() / 60.0) as u64;
        if self.minute.is_some_and(|m| m != minute) && self.checks > 0 {
            let by: Vec<String> = counts()
                .into_iter()
                .map(|(c, (judged, held))| format!("{c:?}:{{\"judged\":{judged},\"held\":{held}}}"))
                .collect();
            let n = self.checks as f64;
            let _ = writeln!(
                self.out,
                "{{\"t_s\":{},\"checks\":{},\"backlog_s_mean\":{:.3},\"backlog_s_max\":{:.3},\"queued_tokens_mean\":{:.0},\"rate_tok_s\":{:.0},\"by_class\":{{{}}}}}",
                self.minute.unwrap_or(0) * 60,
                self.checks,
                self.sum_backlog_s / n,
                self.max_backlog_s,
                self.sum_queued / n,
                self.rate,
                by.join(",")
            );
            let _ = self.out.flush();
            self.checks = 0;
            self.sum_backlog_s = 0.0;
            self.max_backlog_s = 0.0;
            self.sum_queued = 0.0;
        }
        self.minute = Some(minute);
        self.checks += 1;
        self.sum_backlog_s += backlog_s;
        self.max_backlog_s = self.max_backlog_s.max(backlog_s);
        self.sum_queued += backlog_s * rate;
        self.rate = rate;
    }
}

/// The two-resource gate's prefill meter (`admission.prefill_budget`, f42): the uncached prefill
/// tokens the gate let go and the router sent, per worker over the window, and the classes in
/// flight on each worker.
#[derive(Debug, Default)]
struct PrefillMeter {
    /// Per worker, (when the gate let it go, uncached tokens) of requests sent there.
    sent: HashMap<WorkerWithDpRank, VecDeque<(Time, u64)>>,
    /// Requests the gate let go that the router has not sent yet: (when, tokens, target).
    pending: HashMap<String, (Time, u64, Option<WorkerWithDpRank>)>,
    /// Requests in flight: (worker, class).
    inflight: HashMap<String, (WorkerWithDpRank, String)>,
    /// Per worker, requests in flight by class.
    classes: HashMap<WorkerWithDpRank, HashMap<String, u32>>,
}

impl PrefillMeter {
    fn dispatched(&mut self, id: &str, tokens: u64, now: Time, target: Option<WorkerWithDpRank>) {
        self.pending.insert(id.to_owned(), (now, tokens, target));
    }

    fn sent(&mut self, id: &str, worker: WorkerWithDpRank, class: &str, tokens: u64, now: Time) {
        let at = match self.pending.remove(id) {
            Some((at, _, _)) => at,
            None => now,
        };
        self.sent.entry(worker).or_default().push_back((at, tokens));
        if let Some((w, c)) = self
            .inflight
            .insert(id.to_owned(), (worker, class.to_owned()))
        {
            self.release(w, &c);
        }
        *self
            .classes
            .entry(worker)
            .or_default()
            .entry(class.to_owned())
            .or_default() += 1;
    }

    fn finished(&mut self, id: &str) {
        self.pending.remove(id);
        if let Some((w, c)) = self.inflight.remove(id) {
            self.release(w, &c);
        }
    }

    fn release(&mut self, worker: WorkerWithDpRank, class: &str) {
        if let Some(by) = self.classes.get_mut(&worker)
            && let Some(n) = by.get_mut(class)
        {
            *n = n.saturating_sub(1);
            if *n == 0 {
                by.remove(class);
            }
        }
    }

    /// Drop what left the window (and requests never sent within ten windows).
    fn trim(&mut self, now: Time, window_s: f64) {
        for q in self.sent.values_mut() {
            while q
                .front()
                .is_some_and(|(at, _)| now.since(*at).as_secs_f64() > window_s)
            {
                q.pop_front();
            }
        }
        self.pending
            .retain(|_, (at, _, _)| now.since(*at).as_secs_f64() <= 10.0 * window_s);
    }

    /// Tokens in the window sent to `worker` or let go with it as the target.
    fn worker_tokens(&self, worker: WorkerWithDpRank) -> u64 {
        let sent: u64 = self
            .sent
            .get(&worker)
            .map_or(0, |q| q.iter().map(|(_, t)| *t).sum());
        let pending: u64 = self
            .pending
            .values()
            .filter(|(_, _, t)| *t == Some(worker))
            .map(|(_, n, _)| *n)
            .sum();
        sent + pending
    }

    /// Tokens in the window, fleet-wide (sent, plus let go and not sent yet).
    fn fleet_tokens(&self) -> u64 {
        let sent: u64 = self
            .sent
            .values()
            .map(|q| q.iter().map(|(_, t)| *t).sum::<u64>())
            .sum();
        sent + self.pending.values().map(|(_, n, _)| *n).sum::<u64>()
    }

    /// The tightest ITL limit among the classes in flight on `worker` (infinite when none has one).
    fn tightest_itl(&self, worker: WorkerWithDpRank, policy: &ServicePolicy) -> f64 {
        self.classes.get(&worker).map_or(f64::INFINITY, |by| {
            by.keys()
                .filter_map(|c| {
                    policy
                        .classes
                        .iter()
                        .find(|p| &p.name == c)
                        .and_then(|p| p.service_level.as_ref())
                        .map(|l| l.itl_s)
                })
                .fold(f64::INFINITY, f64::min)
        })
    }
}

/// One worker's prefill budget and room (f42), tokens per second.
#[derive(Clone, Copy, Debug)]
struct PrefillRoom {
    worker: WorkerWithDpRank,
    budget: f64,
    used: f64,
    /// The tightest ITL limit in flight there.
    tightest_itl: f64,
}

/// Each worker's prefill budget (`admission.prefill_budget`): `margin (1 - d / L) / u`, with
/// decode step `d` at its running requests and mean context, per-token prefill cost `u` at its
/// chunk, and `L` the tightest ITL limit in flight there (no limit: `margin / u`). `None`
/// without a step model answer.
fn prefill_rooms(
    model: &dyn StepModel,
    rows: &[WorkerRow],
    meter: &PrefillMeter,
    policy: &ServicePolicy,
    cfg: &crate::config::PrefillBudgetConfig,
) -> Option<Vec<PrefillRoom>> {
    let tokens = |t: u64| u32::try_from(t).unwrap_or(u32::MAX);
    let mut out = Vec::with_capacity(rows.len());
    for r in rows {
        let chunk = r
            .chunk
            .filter(|c| *c > 0)
            .unwrap_or(u64::from(policy.forecast.chunk_tokens))
            .max(1);
        let u = model.prefill_s(1, tokens(chunk), 0)? / chunk as f64;
        if u <= 0.0 {
            continue;
        }
        let d = if r.running > 0 {
            let decode = r.active_tokens.saturating_sub(r.prefill_tokens as u64);
            let ctx = decode / r.running as u64;
            model.decode_s(tokens(r.running as u64), tokens(ctx))?
        } else {
            0.0
        };
        let tightest = meter.tightest_itl(r.worker, policy);
        let factor = if tightest.is_finite() {
            (1.0 - d / tightest).max(0.0)
        } else {
            1.0
        };
        out.push(PrefillRoom {
            worker: r.worker,
            budget: cfg.margin * factor / u,
            used: meter.worker_tokens(r.worker) as f64 / cfg.window_s,
            tightest_itl: tightest,
        });
    }
    Some(out)
}

/// Where an admitted new session's first turn goes (`prefill_budget.place_onboard`): the worker
/// with the most room for `own` tokens per second among those with no class in flight whose ITL
/// limit is at or under `protect_itl` (the tightest class's: no decode of that class in flight),
/// else among all; none when no worker has room for it.
fn place_onboard(rooms: &[PrefillRoom], own: f64, protect_itl: f64) -> Option<WorkerWithDpRank> {
    let best = |clean: bool| {
        rooms
            .iter()
            .filter(|r| !clean || r.tightest_itl > protect_itl)
            .filter(|r| r.budget - r.used >= own)
            .max_by(|a, b| {
                (a.budget - a.used)
                    .total_cmp(&(b.budget - b.used))
                    .then_with(|| b.worker.cmp(&a.worker))
            })
            .map(|r| r.worker)
    };
    best(true).or_else(|| best(false))
}

/// f47j (`prefill_budget.place_turns`): where a large turn goes. `per` is the tokens it computes on
/// each worker; a worker qualifies when its metered rate plus the turn's own there (`tokens /
/// window_s`) is at or under `cap`. First among workers with no decode in flight of a class whose ITL
/// limit is at or under `protect_itl`, then among all; the least computed (most cached) wins, then
/// the least used, then the lower worker id.
fn place_turn(
    rooms: &[PrefillRoom],
    per: &[(WorkerWithDpRank, u64)],
    window_s: f64,
    cap: f64,
    protect_itl: f64,
) -> Option<WorkerWithDpRank> {
    let pick = |clean: bool| {
        per.iter()
            .filter_map(|(w, t)| {
                let r = rooms.iter().find(|r| r.worker == *w)?;
                if clean && r.tightest_itl <= protect_itl {
                    return None;
                }
                (r.used + *t as f64 / window_s <= cap).then_some((*w, *t, r.used))
            })
            .min_by(|a, b| {
                a.1.cmp(&b.1)
                    .then_with(|| a.2.total_cmp(&b.2))
                    .then_with(|| a.0.cmp(&b.0))
            })
            .map(|x| x.0)
    };
    pick(true).or_else(|| pick(false))
}

/// The prefill budget and the target loop per simulated minute (`DYNAMO_F42_LOG`).
struct F42Log {
    out: std::io::BufWriter<std::fs::File>,
    minute: Option<u64>,
    checks: u64,
    sum_budget: f64,
    sum_used: f64,
    max_used: f64,
}

impl F42Log {
    fn open() -> Option<Self> {
        let path = std::env::var_os("DYNAMO_F42_LOG")?;
        let file = std::fs::File::create(path).ok()?;
        Some(Self {
            out: std::io::BufWriter::new(file),
            minute: None,
            checks: 0,
            sum_budget: 0.0,
            sum_used: 0.0,
            max_used: 0.0,
        })
    }

    fn record(
        &mut self,
        now: Time,
        budget: f64,
        used: f64,
        stats: impl FnOnce() -> crate::scheduler::LoopStats,
    ) {
        use std::io::Write;
        let minute = (now.as_secs_f64() / 60.0) as u64;
        if self.minute.is_some_and(|m| m != minute) && self.checks > 0 {
            let st = stats();
            let fmt = |m: &HashMap<String, f64>| {
                let mut v: Vec<(&String, &f64)> = m.iter().collect();
                v.sort_by(|a, b| a.0.cmp(b.0));
                v.iter()
                    .map(|(c, a)| format!("{c:?}:{a:.4}"))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            let (scale, judged, held) = (st.scale, st.judged, st.held);
            let att = format!(
                "{}}},\"targets\":{{{}}},\"infeasible\":{},\"infeasible_s\":{:.0},\"infeasible_held\":{},\"predicted_tok_s\":{},\"cal\":{:.4},\"top_itl_miss\":{}",
                fmt(&st.att),
                fmt(&st.targets),
                st.infeasible
                    .as_ref()
                    .map_or("null".to_owned(), |c| format!("{c:?}")),
                st.infeasible_s,
                st.infeasible_held,
                st.predicted
                    .map_or("null".to_owned(), |p| format!("{p:.0}")),
                st.cal,
                st.itl_miss.map_or("null".to_owned(), |m| format!("{m:.4}"))
            );
            // f49n `target_loop.rebalance`: each class's scale, attainment and target, and the mode.
            let att = match &st.rb {
                Some(rb) => format!("{att},{}", rb.log_fields()),
                None => att,
            };
            let n = self.checks as f64;
            let _ = writeln!(
                self.out,
                "{{\"t_s\":{},\"checks\":{},\"budget_tok_s\":{:.0},\"used_tok_s_mean\":{:.0},\"used_tok_s_max\":{:.0},\"scale\":{:.4},\"judged\":{},\"held\":{},\"att\":{{{}}}",
                self.minute.unwrap_or(0) * 60,
                self.checks,
                self.sum_budget / n,
                self.sum_used / n,
                self.max_used,
                scale,
                judged,
                held,
                att
            );
            let _ = self.out.flush();
            self.checks = 0;
            self.sum_budget = 0.0;
            self.sum_used = 0.0;
            self.max_used = 0.0;
        }
        self.minute = Some(minute);
        self.checks += 1;
        self.sum_budget += budget;
        self.sum_used += used;
        self.max_used = self.max_used.max(used);
    }
}

/// The wake key of a session for keyed holds ([`HoldWake::Keys`]): a fixed hash of its id. Two
/// sessions sharing a key only cost an extra recheck.
pub fn session_wake_key(session: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    session.hash(&mut hasher);
    hasher.finish()
}

/// One worker in the classifier's sample.
#[derive(Clone, Copy, Debug)]
struct WorkerRow {
    worker: WorkerWithDpRank,
    /// Active prefill tokens plus the tokens of the KV blocks active requests hold.
    active_tokens: u64,
    kv_tokens: u64,
    prefill_tokens: usize,
    capacity: WorkerCapacity,
    running: usize,
    /// The worker's prefill chunk (`max_num_batched_tokens`), when it reports one.
    chunk: Option<u64>,
}

/// Sessions whose last worker is kept at most (the map is cleared when full).
const MAX_LAST_WORKERS: usize = 1 << 16;

impl Inner {
    /// One coordinator step when due (`controller`): gather the signals, decide, log.
    fn controller_step(&self, now: Time, rows: &[WorkerRow]) {
        if !self.controller.lock().as_ref().is_some_and(|c| c.due(now)) {
            return;
        }
        // The host's capacity directly, not through the cache, so a step never moves the cache's
        // refresh times (shadow mode must equal the run without the coordinator).
        let (hbm, workers) = (self.capacity)();
        let signals = {
            let scheduler = self.scheduler.lock();
            let policy = scheduler.policy();
            let backlog_s = self.step_model.as_ref().and_then(|model| {
                surge_facts(model.as_ref(), rows, 0, policy.forecast.chunk_tokens).map(|f| f.0)
            });
            let ttft_limit_s = policy
                .classes
                .iter()
                .filter_map(|c| c.service_level.as_ref().map(|l| l.ttft_s))
                .fold(None, |m: Option<f64>, x| Some(m.map_or(x, |m| m.min(x))));
            let dram = policy.admission.dram_beyond_hbm(hbm, workers);
            let mut kv = scheduler.kv_budget(hbm, dram);
            self.account(now, &policy.admission, &mut kv);
            let frac = |x: u64, cap: u64| (cap > 0).then(|| x as f64 / cap as f64);
            let top_headroom = policy
                .classes
                .iter()
                .filter(|c| c.service_level.is_some())
                .min_by_key(|c| c.rank)
                .and_then(|c| {
                    let t = c.service_level.as_ref()?.attainment;
                    scheduler.window_share(&c.name, now).map(|a| a - t)
                });
            crate::coordinator::Signals {
                backlog_s,
                top_headroom,
                ttft_limit_s,
                hot_frac: frac(kv.hot_tokens, kv.hbm_tokens),
                total_frac: frac(kv.total_tokens, kv.hbm_tokens + kv.dram_tokens),
                headroom: policy
                    .classes
                    .iter()
                    .filter_map(|c| {
                        let t = c.service_level.as_ref()?.attainment;
                        Some((
                            c.name.clone(),
                            scheduler.window_share(&c.name, now).map(|a| a - t),
                        ))
                    })
                    .collect(),
                waiting: policy
                    .classes
                    .iter()
                    .map(|c| {
                        (
                            c.name.clone(),
                            scheduler.sessions().lobby().has_class(&c.name),
                        )
                    })
                    .collect(),
                gate_on: scheduler.session_gate_on(),
                lobby: scheduler.lobby_stats(now),
                rotations: scheduler.rotations(),
                accounting: self.accounting.lock().report.clone(),
            }
        };
        if let Some(c) = self.controller.lock().as_mut() {
            c.step(now, &signals);
        }
    }

    fn notify(&self) {
        if let Some(recheck) = self.recheck.get() {
            recheck.notify();
        }
    }

    /// Whether the host answers keyed holds from their last answer.
    fn keyed(&self) -> bool {
        self.recheck
            .get()
            .is_some_and(DispatchRecheck::honours_keys)
    }

    /// Wake the keyed holds whose answer may have changed: those behind or of a session that left
    /// or moved in the lobby, or all of them when the lobby was rebuilt or the session gate
    /// switched. Called with the scheduler locked at the end of every entry point that can change
    /// either (classify, events, checks, dispatch).
    fn wake_changed(&self, scheduler: &mut Scheduler) {
        let Some(recheck) = self.recheck.get().filter(|r| r.honours_keys()) else {
            return;
        };
        let changes = scheduler.take_lobby_changes();
        let gate_on = scheduler.session_gate_on();
        let switched = {
            let mut was = self.gate_was_on.lock();
            let switched = was.is_some_and(|w| w != gate_on);
            *was = Some(gate_on);
            switched
        };
        if changes.reset || switched {
            recheck.wake_all();
        } else if !changes.sessions.is_empty() {
            let keys: Vec<u64> = changes
                .sessions
                .iter()
                .map(|id| session_wake_key(id))
                .collect();
            recheck.wake_keys(&keys);
        }
    }

    fn capacity(&self, now: Time) -> (u64, u64) {
        let mut cached = self.cached_capacity.lock();
        if let Some((at, hbm, workers)) = *cached
            && now.since(at) < CAPACITY_TTL
        {
            return (hbm, workers);
        }
        let (hbm, workers) = (self.capacity)();
        *cached = Some((now, hbm, workers));
        (hbm, workers)
    }

    /// The total budget's usage as `admission.accounting` counts it (F34): the index measurement
    /// plus the table's growth since, while fresh; otherwise the table, plus the session ends it
    /// does not credit (`verify`).
    fn account(
        &self,
        now: Time,
        admission: &crate::config::AdmissionPolicy,
        kv: &mut crate::admit::KvBudget,
    ) {
        use crate::config::AccountingMode;
        let a = &admission.accounting;
        if a.mode == AccountingMode::Table && a.verify.is_none() {
            return;
        }
        let state = self.accounting.lock();
        let stale_s = a.refresh_s * a.stale_after_refreshes;
        let measured = (a.mode == AccountingMode::Index)
            .then(|| {
                state
                    .snapshot
                    .and_then(|s| s.usage(now, kv.total_tokens, stale_s))
            })
            .flatten();
        match measured {
            Some(usage) => {
                kv.total_tokens = usage;
                kv.yield_total_tokens = kv.yield_total_tokens.min(usage);
            }
            None => {
                kv.total_tokens = kv
                    .total_tokens
                    .saturating_add(state.check.uncredited_tokens());
            }
        }
    }

    /// Record the sessions that just ended for the KV-action check (`verify`).
    fn note_ended(&self, now: Time, ended: Vec<(String, u64)>) {
        if ended.is_empty() {
            return;
        }
        let verify = self
            .scheduler
            .lock()
            .policy()
            .admission
            .accounting
            .verify
            .clone();
        let Some(verify) = verify else { return };
        let mut state = self.accounting.lock();
        for (session, context) in ended {
            let rows = self.index.session_rows(&session);
            state.check.ended(&verify, &session, now, context, &rows);
        }
    }

    /// One accounting step when due (`admission.accounting.refresh_s`): measure the index, run
    /// the KV-action check, and write the measured against the claimed usage per tier.
    fn accounting_step(&self, now: Time) {
        use crate::config::AccountingMode;
        let (admission, table_total) = {
            let scheduler = self.scheduler.lock();
            let admission = scheduler.policy().admission.clone();
            (admission, scheduler.kv_budget(0, 0).total_tokens)
        };
        let a = &admission.accounting;
        if a.mode == AccountingMode::Table && a.verify.is_none() {
            return;
        }
        {
            let state = self.accounting.lock();
            if state
                .last
                .is_some_and(|t| now.since(t).as_secs_f64() < a.refresh_s)
            {
                return;
            }
        }
        let (hbm, workers) = (self.capacity)();
        let below = admission.dram_beyond_hbm(hbm, workers);
        let block = u64::from(self.block_size.max(1));
        let measured = if a.mode == AccountingMode::Index {
            crate::accounting::measure(&self.index.unique_rows(), block, &admission)
        } else {
            None
        };
        let mut state = self.accounting.lock();
        state.last = Some(now);
        if let Some(m) = measured {
            state.snapshot = Some(crate::accounting::Snapshot {
                at: now,
                measured: m,
                table_total,
            });
        }
        if let Some(v) = &a.verify {
            state.check.step(v, now, |s| self.index.session_rows(s));
        }
        let source = if measured.is_some() { "index" } else { "table" };
        let m = measured.unwrap_or_default();
        let honoured: Vec<String> = state
            .check
            .counts()
            .into_iter()
            .map(|(w, c, h)| {
                format!(
                    "{{\"worker\":\"{}:{}\",\"checked\":{},\"honoured\":{}}}",
                    w.worker_id, w.dp_rank, c, h
                )
            })
            .collect();
        let report = format!(
            "{{\"mode\":\"{}\",\"source\":\"{}\",\"capacity\":{{\"hbm\":{},\"below_hbm\":{}}},\"measured\":{{\"device\":{},\"host\":{},\"disk\":{},\"external\":{},\"unique\":{}}},\"claimed\":{{\"table_total\":{},\"uncredited\":{}}},\"honoured\":[{}],\"pending_checks\":{}}}",
            match a.mode {
                AccountingMode::Table => "table",
                AccountingMode::Index => "index",
            },
            source,
            hbm,
            below,
            m.device,
            m.host,
            m.disk,
            m.external,
            m.unique,
            table_total,
            state.check.uncredited_tokens(),
            honoured.join(","),
            state.check.pending(),
        );
        if let Some(out) = state.log.as_mut() {
            use std::io::Write as _;
            let _ = writeln!(
                out,
                "{{\"t_s\":{:.1},\"accounting\":{}}}",
                now.as_secs_f64(),
                report
            );
            let _ = out.flush();
        }
        state.report = Some(report);
    }

    fn observe(&self, observation: Observation) {
        if let Some((_, feed)) = &self.forecast {
            feed.send(observation);
        }
    }

    /// Rotation (`admission.rotate`), at most every `check_s`: per class with a session waiting
    /// too long, pause the first candidate whose KV the session index shows in the host tier,
    /// within `max_per_min`. Each pause is logged to `DYNAMO_LOBBY_LOG` with its reason.
    fn maybe_rotate(&self, now: Time) {
        let mut scheduler = self.scheduler.lock();
        let Some(rotate) = scheduler.policy().admission.rotate.clone() else {
            return;
        };
        if scheduler.mode() != Mode::Enforce {
            return;
        }
        {
            let mut last = self.last_rotate.lock();
            if last.is_some_and(|at| now.since(at).as_secs_f64() < rotate.check_s) {
                return;
            }
            *last = Some(now);
        }
        let block = u64::from(self.block_size.max(1));
        let mut recent = self.recent_pauses.lock();
        while recent
            .front()
            .is_some_and(|&t| now.since(t).as_secs_f64() > 60.0)
        {
            recent.pop_front();
        }
        for c in scheduler.rotation_candidates(now) {
            if rotate.max_per_min.is_some_and(|m| recent.len() as f64 >= m) {
                break;
            }
            let mut uncovered = 0u32;
            let mut paused = None;
            for (id, class, run) in &c.sessions {
                let context = scheduler.sessions().get(id).map_or(0, |s| s.context_tokens);
                let covered = self.index.resident_blocks(id).is_some_and(|(_, host, _)| {
                    (host.saturating_mul(block)) as f64 >= rotate.host_cover * context as f64
                });
                if !covered {
                    uncovered += 1;
                    continue;
                }
                if scheduler.pause(id, now) {
                    paused = Some((id.clone(), class.clone(), *run, context));
                    break;
                }
            }
            if paused.is_none()
                && let Some(out) = self.lobby_log.lock().as_mut()
            {
                use std::io::Write as _;
                let _ = writeln!(
                    out,
                    "{{\"t_s\":{:.1},\"event\":\"no_park\",\"for_class\":{:?},\"oldest_wait_s\":{:.1},\"candidates\":{},\"skipped_not_in_host\":{}}}",
                    now.as_secs_f64(),
                    c.for_class,
                    c.waited_s,
                    c.sessions.len(),
                    uncovered,
                );
                let _ = out.flush();
            }
            if let Some((id, class, run, context)) = paused {
                recent.push_back(now);
                if let Some(out) = self.lobby_log.lock().as_mut() {
                    use std::io::Write as _;
                    let line = format!(
                        "{{\"t_s\":{:.1},\"event\":\"park\",\"session\":{:?},\"class\":{:?},\"run_s\":{:.1},\"context_tokens\":{},\"for_class\":{:?},\"oldest_wait_s\":{:.1},\"candidates\":{},\"skipped_not_in_host\":{}}}",
                        now.as_secs_f64(),
                        id,
                        class,
                        run,
                        context,
                        c.for_class,
                        c.waited_s,
                        c.sessions.len(),
                        uncovered,
                    );
                    let _ = writeln!(out, "{line}");
                    let _ = out.flush();
                }
            }
        }
    }

    /// Rebuild the forecast after its inputs change (only while constructing).
    fn rebuild_forecast(&mut self) {
        let policy = self.scheduler.get_mut().policy().clone();
        self.forecast = self.step_model.as_ref().map(|model| {
            Forecast::new(
                &policy,
                Arc::clone(model),
                Arc::clone(&self.clock),
                self.block_size,
            )
        });
    }

    /// A load sample of every worker, at most once per `snapshot_interval_s`. O(workers) when
    /// taken, so O(1) amortized per classify at any steady request rate.
    fn sample_workers(&self, now: Time, workers: &dyn WorkerView) {
        let Some((forecast, feed)) = &self.forecast else {
            return;
        };
        {
            let mut last = self.last_sample.lock();
            let interval = Duration::from_secs_f64(forecast.config().snapshot_interval_s);
            if last.is_some_and(|at| now.since(at) < interval) {
                return;
            }
            *last = Some(now);
        }
        let rows = (0..workers.len())
            .filter_map(|row| {
                let info = workers.info(row)?;
                let load = workers.load(row).filter(|l| l.available)?;
                Some(WorkerSample {
                    worker: info.worker(),
                    active_prefill_tokens: load.active_prefill_tokens as u64,
                    active_decode_blocks: load.active_decode_blocks as u64,
                    active_requests: load.active_requests as u64,
                    max_batched_tokens: info.max_num_batched_tokens(),
                })
            })
            .collect::<Vec<_>>();
        if !rows.is_empty() {
            feed.send(Observation::Workers { at: now, rows });
        }
    }
}

/// Runs [`Inner::wake_changed`] when dropped (the classifier's exits), with the scheduler unlocked
/// by then.
struct WakeChangedOnDrop<'a>(&'a Inner);

impl Drop for WakeChangedOnDrop<'_> {
    fn drop(&mut self) {
        if self.0.keyed() {
            let mut scheduler = self.0.scheduler.lock();
            self.0.wake_changed(&mut scheduler);
        }
    }
}

/// The scheduler as a `RequestClassifier`.
pub struct SessionSchedulerClassifier {
    inner: Arc<Inner>,
}

impl SessionSchedulerClassifier {
    /// With `forecast.enabled`, the forecast uses the configured step model, or the table when
    /// that cannot be built (the provider reports the error instead: see
    /// [`Self::with_step_model`]).
    pub fn new(policy: ServicePolicy, clock: Arc<dyn Clock>, capacity: CapacitySource) -> Self {
        let step_model = if policy.forecast.enabled {
            Some(policy.forecast.step_model().unwrap_or_else(|_| {
                Arc::new(policy.forecast.step_table.clone()) as Arc<dyn StepModel>
            }))
        } else {
            None
        };
        // The adaptive rule's groups start as configured; without the rule they are on.
        let start =
            crate::adaptive::AdaptiveState::new(policy.adaptive.clone().unwrap_or_default());
        let rule = policy.adaptive.is_some();
        crate::adaptive::set_place_enabled(!rule || start.place);
        dynamo_kv_router::plugins::adaptive_switches::set_kv_actions_enabled(!rule || start.kv);
        dynamo_kv_router::plugins::adaptive_switches::set_class_preemption_enabled(
            !rule || start.preempt,
        );
        let mut inner = Inner {
            clock,
            capacity,
            cached_capacity: Mutex::new(None),
            scheduler: Mutex::new(Scheduler::new(policy)),
            recheck: OnceLock::new(),
            index: Arc::new(NoSessionIndex),
            block_size: 0,
            step_model,
            forecast: None,
            last_sample: Mutex::new(None),
            pool: crate::pool::installed("default"),
            affinity: crate::affinity::installed("default"),
            kv_rows: Mutex::new(Vec::new()),
            busy: crate::busy::installed("default"),
            route: crate::route::installed("default"),
            last_rotate: Mutex::new(None),
            recent_pauses: Mutex::new(std::collections::VecDeque::new()),
            lobby_log: Mutex::new(
                std::env::var_os("DYNAMO_LOBBY_LOG")
                    .and_then(|p| std::fs::File::create(p).ok())
                    .map(std::io::BufWriter::new),
            ),
            last_worker: Mutex::new(std::collections::HashMap::new()),
            aimd: None,
            gate_was_on: Mutex::new(None),
            surge_log: Mutex::new(SurgeLog::open()),
            prefill_meter: Mutex::new(PrefillMeter::default()),
            f42_log: Mutex::new(F42Log::open()),
            controller: Mutex::new(None),
            accounting: Mutex::new(AccountingState {
                log: std::env::var_os("DYNAMO_ACCOUNTING_LOG")
                    .and_then(|p| std::fs::File::create(p).ok())
                    .map(std::io::BufWriter::new),
                ..AccountingState::default()
            }),
        };
        // f49h `admission.aimd`: off (absent) by default.
        inner.aimd =
            crate::aimd::Aimd::from_policy(inner.scheduler.lock().policy()).map(Mutex::new);
        // The run log's uncontended time (and the run orders) price requests with the forecast's
        // step model, or the configured table when the forecast is off but runs are read.
        {
            let mut scheduler = inner.scheduler.lock();
            let model = inner
                .step_model
                .clone()
                .or_else(|| {
                    // f49q `signal: slowdown`: the forecast's step model (v2 when present).
                    scheduler
                        .uses_slowdown()
                        .then(|| scheduler.policy().forecast.step_model().ok())
                        .flatten()
                })
                .or_else(|| {
                    scheduler.uses_runs().then(|| {
                        Arc::new(scheduler.policy().forecast.step_table.clone())
                            as Arc<dyn StepModel>
                    })
                });
            if let Some(model) = model {
                scheduler.set_step_model(model);
            }
        }
        // Every part on until a coordinator in act mode decides otherwise.
        crate::coordinator::reset();
        if let Some(cfg) = inner.scheduler.lock().policy().controller.clone() {
            *inner.controller.lock() = Some(crate::coordinator::Coordinator::new(cfg));
        }
        inner.rebuild_forecast();
        Self {
            inner: Arc::new(inner),
        }
    }

    /// The host's KV block size, to turn the worker view's overlap blocks into tokens. Without
    /// it (0) cached prefixes are not discounted.
    pub fn with_block_size(mut self, block_size: u32) -> Self {
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            inner.block_size = block_size;
            inner.rebuild_forecast();
        }
        self
    }

    /// The forecast's step model (when the forecast is enabled).
    pub fn with_step_model(mut self, model: Arc<dyn StepModel>) -> Self {
        if let Some(inner) = Arc::get_mut(&mut self.inner)
            && inner.step_model.is_some()
        {
            inner.step_model = Some(Arc::clone(&model));
            inner.scheduler.lock().set_step_model(model);
            inner.rebuild_forecast();
        }
        self
    }

    /// The forecast, when enabled. Querying it never blocks the classifier.
    pub fn forecast(&self) -> Option<Arc<Forecast>> {
        self.inner.forecast.as_ref().map(|(f, _)| Arc::clone(f))
    }

    /// Read the scheduler (for snapshots and tests). Do not hold it across awaits.
    pub fn with_scheduler<R>(&self, f: impl FnOnce(&mut Scheduler) -> R) -> R {
        f(&mut self.inner.scheduler.lock())
    }

    /// Read the session index through `index` instead of [`NoSessionIndex`].
    pub fn with_session_index(mut self, index: Arc<dyn SessionIndexView>) -> Self {
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            inner.index = index;
        }
        self
    }

    /// The session index in use: the router's when the host gives one, else [`NoSessionIndex`].
    pub fn session_index(&self) -> &dyn SessionIndexView {
        self.inner.index.as_ref()
    }
}

/// Blocks of this request's prefix already cached on its best worker, by tier, from the
/// classifier's worker view: the KV hint available to a plugin today. Only the device tier is
/// discounted from what admission adds; a host or disk hit still needs HBM room to run.
/// f47h (`prefill_budget.meter: uncached`): the tokens a request of `prompt` tokens will compute on each
/// worker (its prompt less the prefix cached there in any tier: device, host or disk), and at best (the
/// minimum over the workers; the whole prompt when no worker has a cache row).
fn computed_tokens(
    prompt: u64,
    block: u64,
    workers: &dyn WorkerView,
) -> (u64, Vec<(WorkerWithDpRank, u64)>) {
    let per: Vec<(WorkerWithDpRank, u64)> = (0..workers.len())
        .filter_map(|row| {
            let info = workers.info(row)?;
            let c = workers.cache(row)?;
            let hit = (c.device_overlap_blocks + c.host_overlap_blocks + c.disk_overlap_blocks)
                .max(0.0) as u64;
            Some((
                info.worker(),
                prompt.saturating_sub(hit.saturating_mul(block)),
            ))
        })
        .collect();
    let best = per.iter().map(|(_, t)| *t).min().unwrap_or(prompt);
    (best, per)
}

fn best_overlap(workers: &dyn WorkerView) -> (f64, f64, f64) {
    (0..workers.len())
        .filter_map(|row| workers.cache(row))
        .map(|c| {
            (
                c.device_overlap_blocks,
                c.host_overlap_blocks,
                c.disk_overlap_blocks,
            )
        })
        .fold((0.0, 0.0, 0.0), |best, row| {
            if row.0 + row.1 + row.2 > best.0 + best.1 + best.2 {
                row
            } else {
                best
            }
        })
}

#[async_trait]
impl RequestClassifier for SessionSchedulerClassifier {
    fn required_worker_inputs(&self) -> WorkerInputs {
        let scheduler = self.inner.scheduler.lock();
        if scheduler.mode() == Mode::Off {
            WorkerInputs::NONE
        } else if self.inner.forecast.is_some() || scheduler.policy().admission.shed_by_slack {
            WorkerInputs::CACHE | WorkerInputs::LOAD
        } else {
            WorkerInputs::CACHE
        }
    }

    fn classify(
        &mut self,
        mut request: ClassifyRequest,
        workers: &dyn WorkerView,
    ) -> Result<ClassifyRequest, Box<ClassifierError>> {
        // Wake the keyed holds this turn's lobby changes concern, on every way out.
        let _wake = WakeChangedOnDrop(&self.inner);
        let inner = &self.inner;
        if let (Some(pool), Some(request_id)) = (&inner.pool, request.request_id()) {
            let now = inner.clock.now();
            let workers = inner.capacity(now).1;
            let class = request
                .policy_class()
                .unwrap_or(&inner.scheduler.lock().policy().default_class)
                .to_owned();
            pool.arrived(
                request_id,
                &class,
                request.isl_tokens() as u64,
                now,
                workers,
            );
        }
        if let (Some(busy), Some(request_id)) = (&inner.busy, request.request_id()) {
            let class = request
                .policy_class()
                .unwrap_or(&inner.scheduler.lock().policy().default_class)
                .to_owned();
            busy.arrived(request_id, &class);
        }
        let (Some(request_id), Some(session)) = (request.request_id(), request.session_context())
        else {
            return Ok(request);
        };
        if let Some(affinity) = &inner.affinity {
            let fanout = session
                .agent_headers()
                .get(crate::affinity::HEADER_HINT_FANOUT)
                .and_then(|values| values.first())
                .and_then(|v| v.parse::<u32>().ok());
            affinity.arrived(
                request_id,
                session.session_id(),
                session.parent_session_id(),
                fanout,
                session.session_final() == Some(true),
            );
        }
        if let Some(route) = &inner.route
            && route.reading_due(inner.clock.now())
        {
            let scheduler = inner.scheduler.lock();
            let reserved = route
                .config()
                .soft_reserve
                .as_ref()
                .map(|r| r.class.clone());
            if let Some(class) = reserved {
                let now = inner.clock.now();
                let target = scheduler
                    .policy()
                    .class(Some(&class))
                    .service_level
                    .as_ref()
                    .map(|l| l.attainment);
                route.reading(
                    scheduler.window_share(&class, now),
                    target,
                    scheduler.sessions().live_share(&class),
                );
            }
        }
        if let Some(route) = &inner.route {
            let class = request
                .policy_class()
                .unwrap_or(&inner.scheduler.lock().policy().default_class)
                .to_owned();
            route.arrived(
                request_id,
                session.session_id(),
                session.parent_session_id(),
                &class,
                session.session_final() == Some(true),
            );
        }
        let now = inner.clock.now();
        let arrived = inner
            .clock
            .time_of_host(request.ingress_at())
            .unwrap_or(now);
        inner.maybe_rotate(now);
        let mut scheduler = inner.scheduler.lock();
        let mode = scheduler.mode();
        if mode == Mode::Off {
            return Ok(request);
        }
        let session_id = session.session_id().to_owned();
        let lobby_id = session_id.clone();
        // f49e `admission.tickets.per_root`: a sub-session of a root with a live ticket is not
        // throttled (without it, the same as `throttle`). f49f `admission.returnee_first`: a new
        // top-level session of a listed class is throttled while its class has a live ticket; it
        // gets its own ticket and that ticket's Retry-After below, like any throttle.
        // f49m `admission.honest_retry`: the request's KV need (its uncached input), for the room
        // check before `returnee_first` refuses it. Not computed when off.
        let honest_need = scheduler
            .policy()
            .admission
            .honest_retry
            .is_some()
            .then(|| {
                let (device, _, _) = best_overlap(workers);
                let cached = (device.max(0.0) as u64).saturating_mul(u64::from(inner.block_size));
                (request.isl_tokens() as u64).saturating_sub(cached)
            });
        if let Some(Decision::Throttle { waiting, limit }) = scheduler.throttle_with_parent_need(
            &session_id,
            session.parent_session_id(),
            request.policy_class(),
            honest_need,
            now,
        ) && mode == Mode::Enforce
        {
            if let Some(pool) = &inner.pool
                && let Some(id) = request.request_id()
            {
                pool.finished(id);
            }
            // The session log (and its added delay) counts the refusal and the retry after it.
            if session.parent_session_id().is_none() {
                scheduler.note_throttled(&session_id, request.policy_class(), now);
            }
            let class = scheduler
                .policy()
                .class(request.policy_class())
                .name
                .clone();
            scheduler.note_lost(&class);
            // f49c `admission.tickets`: a throttled top-level session of a listed class gets its
            // ticket (its KV need is the request's uncached input) and the ticket's Retry-After.
            // f49e `per_root`: a throttled sub-session issues its root's ticket.
            let per_root = scheduler
                .policy()
                .admission
                .tickets
                .as_ref()
                .is_some_and(|c| c.per_root);
            let ticket = if (session.parent_session_id().is_none() || per_root)
                && scheduler.policy().admission.tickets.is_some()
            {
                let (device, _, _) = best_overlap(workers);
                let cached = (device.max(0.0) as u64).saturating_mul(u64::from(inner.block_size));
                let isl = (request.isl_tokens() as u64).saturating_sub(cached);
                let uncached = scheduler
                    .policy()
                    .admission
                    .prefill_budget
                    .as_ref()
                    .is_some_and(|c| c.meter == crate::config::PrefillMeterKind::Uncached);
                let prefill = if uncached {
                    computed_tokens(
                        request.isl_tokens() as u64,
                        u64::from(inner.block_size),
                        workers,
                    )
                    .0
                } else {
                    isl
                };
                let key = scheduler.ticket_root(&session_id, session.parent_session_id());
                scheduler.ticket_on_throttle(&key, request.policy_class(), isl, prefill, now)
            } else {
                // f49m: the honest advice without a ticket (`None` when off).
                scheduler.honest_throttle_untracked(&session_id, request.policy_class(), now)
            };
            if let Some(seconds) = ticket.or(scheduler.policy().admission.retry_after_s) {
                return Err(Box::new(rate_limited(waiting, limit, seconds)) as Box<ClassifierError>);
            }
            return Err(
                Box::new(SchedulerReject::Throttled { waiting, limit }) as Box<ClassifierError>
            );
        }
        // Cached prefix tokens: not admitted twice against the budget.
        let (device, _, _) = best_overlap(workers);
        let cached = (device.max(0.0) as u64).saturating_mul(u64::from(inner.block_size));
        let isl = (request.isl_tokens() as u64).saturating_sub(cached);
        // f49f2 `DYNAMO_TICKETS_LOG`: a `returnee_pass` line of this arrival (or, in shadow mode,
        // its `returnee_block` line) ends with its KV need, as its ticket would count it.
        scheduler.returnee_need(&session_id, isl);
        // f49i `admission.worker_step_budget`: the tokens this request computes on each worker
        // (prompt less its prefix cached in any tier), for the budget's projections.
        if let Some((forecast, _)) = &inner.forecast
            && let Some(budget) = forecast.step_budget()
        {
            let (_, per) = computed_tokens(
                request.isl_tokens() as u64,
                u64::from(inner.block_size),
                workers,
            );
            budget.note_compute(request_id, per);
        }
        // f47h (`prefill_budget.meter: uncached`): the tokens this request will compute, prompt less
        // its prefix cached in any tier, per worker and at best.
        let compute = scheduler
            .policy()
            .admission
            .prefill_budget
            .as_ref()
            .is_some_and(|c| c.meter == crate::config::PrefillMeterKind::Uncached)
            .then(|| {
                computed_tokens(
                    request.isl_tokens() as u64,
                    u64::from(inner.block_size),
                    workers,
                )
            });
        let request_id = request_id.to_owned();
        let attributes = request_attributes(session);
        if attributes.signals != Signals::default() {
            scheduler.adaptive_hint_seen();
        }
        if let Some(chunk) = workers
            .info(0)
            .and_then(|info| info.max_num_batched_tokens())
        {
            scheduler.set_chunk_tokens(chunk);
        }
        let (state, class) = scheduler.on_classify_with(
            Some(&request_id),
            &session_id,
            session.parent_session_id(),
            request.policy_class(),
            &attributes,
            isl,
            arrived,
        );
        scheduler.set_request_prompt(&request_id, request.isl_tokens() as u64);
        // f49j2 `admission.step_target`: the host's progress counter (nothing when off).
        scheduler.step_target_progress(&request_id, request.progress(), now);
        if let Some((best, per)) = compute {
            scheduler.set_request_compute(&request_id, &session_id, best, per);
        }
        // Run or deficit order (f36b): the router queue orders the request by this key, which
        // never expires it.
        if let Some(key) = scheduler.request_order_key(&request_id) {
            request.set_order_key(key);
        }
        // A waiting session the router must expire: it is never classified again while queued,
        // so its deadline is set now. `due_at` also orders ahead of the policy score.
        let policy = scheduler.policy().class(Some(&class)).admission.clone();
        let session_gate = scheduler.policy().admission.session_gate;
        let retry_after_s = scheduler.policy().admission.retry_after_s;
        let admission = &scheduler.policy().admission;
        let adaptive = scheduler.policy().adaptive.is_some();
        let controller = scheduler.policy().controller.is_some();
        let sample = controller
            || admission.shed_by_slack
            || admission.go_target
            || admission.hold_at_max_num_seqs
            || admission.surge.is_some()
            || admission.prefill_budget.is_some()
            || adaptive;
        drop(scheduler);
        inner.accounting_step(now);
        // f49k `admission.hot_budget: measured`: the router's LOAD rows, the hot measurement of the
        // workers without engine gauges (a changed row is a refresh). Nothing when off.
        if inner.scheduler.lock().hot_measured_on() {
            let block_size = u64::from(inner.block_size.max(1));
            let rows: Vec<(WorkerWithDpRank, crate::hot_measure::Reading)> = (0..workers.len())
                .filter_map(|row| {
                    let info = workers.info(row)?;
                    let load = workers.load(row).filter(|l| l.available)?;
                    Some((
                        info.worker(),
                        crate::hot_measure::Reading::from_load(
                            load.active_prefill_tokens as u64,
                            load.active_decode_blocks as u64,
                            info.total_kv_blocks().unwrap_or(0),
                            block_size,
                        ),
                    ))
                })
                .collect();
            if !rows.is_empty() {
                inner.scheduler.lock().hot_load_rows(&rows);
            }
        }
        if sample {
            let block_size = u64::from(inner.block_size.max(1));
            let rows = (0..workers.len())
                .filter_map(|row| {
                    let info = workers.info(row)?;
                    let load = workers.load(row).filter(|l| l.available)?;
                    let active = (load.active_prefill_tokens as u64).saturating_add(
                        (load.active_decode_blocks as u64).saturating_mul(block_size),
                    );
                    Some(WorkerRow {
                        worker: info.worker(),
                        active_tokens: active,
                        kv_tokens: info
                            .total_kv_blocks()
                            .unwrap_or(0)
                            .saturating_mul(block_size),
                        prefill_tokens: load.active_prefill_tokens,
                        capacity: info.capacity(),
                        running: load.active_requests,
                        chunk: info.max_num_batched_tokens(),
                    })
                })
                .collect::<Vec<_>>();
            // In worker order, whatever order the router lists them in: ties among rows (the
            // forecast's best placement) and the float mean below then repeat run to run.
            let mut rows = rows;
            rows.sort_by_key(|r| r.worker);
            if adaptive && !rows.is_empty() {
                // The adaptive rule's load pressure: mean active tokens over KV capacity.
                let load = rows
                    .iter()
                    .map(|r| r.active_tokens as f64 / r.kv_tokens.max(1) as f64)
                    .sum::<f64>()
                    / rows.len() as f64;
                let (hbm, workers) = inner.capacity(now);
                let mut scheduler = inner.scheduler.lock();
                let load = scheduler.adaptive_load(load, rows.iter().map(|r| r.kv_tokens).sum());
                let dram = scheduler.policy().admission.dram_beyond_hbm(hbm, workers);
                if let Some(place) = scheduler.adaptive_step(now, load, hbm, dram) {
                    crate::adaptive::set_place_enabled(place);
                    if let Some(groups) = scheduler.adaptive() {
                        dynamo_kv_router::plugins::adaptive_switches::set_kv_actions_enabled(
                            groups.kv,
                        );
                        dynamo_kv_router::plugins::adaptive_switches::set_class_preemption_enabled(
                            groups.preempt,
                        );
                    }
                }
            }
            if controller && !rows.is_empty() {
                inner.controller_step(now, &rows);
            }
            if !rows.is_empty() {
                *inner.kv_rows.lock() = rows;
            }
        }
        inner.sample_workers(now, workers);
        // f49j `admission.step_target`: the workers listed now, with their HBM (the weights of the
        // fleet multiple). Nothing when off.
        {
            let mut scheduler = inner.scheduler.lock();
            if scheduler.step_target_on() {
                let block_size = u64::from(inner.block_size.max(1));
                let listed: Vec<(WorkerWithDpRank, u64)> = (0..workers.len())
                    .filter_map(|row| {
                        workers.info(row).map(|info| {
                            (
                                info.worker(),
                                info.total_kv_blocks()
                                    .unwrap_or(0)
                                    .saturating_mul(block_size),
                            )
                        })
                    })
                    .collect();
                scheduler.step_target_workers(&listed, now);
            }
        }
        // f49h `admission.aimd`: the workers listed now, and a turn of the AIMD class tracked from
        // here; in enforce mode it is refused (expired with a Retry-After) once it has waited
        // `wait_bound_s`. A lobby deadline set below for a new session replaces this one.
        if let Some(aimd) = &inner.aimd {
            let start = {
                let scheduler = inner.scheduler.lock();
                let root = scheduler.root_of(&session_id);
                scheduler
                    .sessions()
                    .get(&root)
                    .or_else(|| scheduler.sessions().get(&session_id))
                    .map_or(arrived, |s| s.started)
            };
            let listed: Vec<WorkerWithDpRank> = (0..workers.len())
                .filter_map(|row| workers.info(row).map(|info| info.worker()))
                .collect();
            let mut aimd = aimd.lock();
            aimd.observe_workers(&listed);
            if aimd.arrived(&request_id, &session_id, &class, start, now) && mode == Mode::Enforce {
                let deadline = arrived.after(Duration::from_secs_f64(aimd.wait_bound_s()));
                if let Some(due_at) = inner.clock.host_instant(deadline) {
                    request.set_due_at(due_at);
                    let seconds = aimd.retry_after_s(&request_id, now);
                    request.set_retry_after(Duration::from_secs_f64(seconds));
                }
            }
        }
        inner.observe(Observation::Arrived {
            request_id,
            session: session_id,
            class,
            at: arrived,
            prompt_tokens: request.isl_tokens() as u64,
        });
        // The lobby deadline applies while the session gate is on. With the adaptive rule's gate
        // off nothing holds the session, so no lobby wait limit expires it either.
        let gate_on = session_gate && inner.scheduler.lock().session_gate_on();
        if mode == Mode::Enforce
            && session_gate
            && !gate_on
            && state == SessionState::Lobby
            && policy.on_wait_limit == WaitLimitAction::Expire
        {
            inner.scheduler.lock().mark_no_lobby_deadline(&lobby_id);
        }
        // A session resuming after a rotation is never refused, so it gets no deadline.
        let resuming = state == SessionState::Lobby
            && inner
                .scheduler
                .lock()
                .sessions()
                .get(&lobby_id)
                .is_some_and(|s| s.resumed);
        if mode == Mode::Enforce
            && gate_on
            && state == SessionState::Lobby
            && !resuming
            && policy.on_wait_limit == WaitLimitAction::Expire
        {
            // f47g: a class the plugin may still admit at its wait limit expires a short grace after
            // it, so the lobby check at the limit runs before the queue's deadline refuses it.
            let grace = {
                let scheduler = inner.scheduler.lock();
                let adm = &scheduler.policy().admission;
                let listed = scheduler.sessions().get(&lobby_id).is_some_and(|s| {
                    adm.fit_at_limit_classes.contains(&s.class)
                        || adm
                            .returnee_guarantee
                            .as_ref()
                            .is_some_and(|g| g.classes.contains(&s.class))
                });
                if listed {
                    adm.at_limit_grace_s.max(0.0)
                } else {
                    0.0
                }
            };
            let deadline = arrived.after(Duration::from_secs_f64(policy.max_lobby_wait_s + grace));
            if let Some(due_at) = inner.clock.host_instant(deadline) {
                request.set_due_at(due_at);
                let estimate = {
                    let mut scheduler = inner.scheduler.lock();
                    let estimate = scheduler.retry_after_estimate(&lobby_id, now);
                    // f49c `admission.tickets`: a listed class's advice is kept for its ticket,
                    // and with `retry_after: forecast` it is the projected capacity's.
                    // f49m `admission.honest_retry`: the honest advice instead of both.
                    let estimate = if scheduler.honest_on() {
                        let until = deadline.as_secs_f64() - now.as_secs_f64();
                        scheduler.honest_lobby_retry_after(&lobby_id, until, now)
                    } else if scheduler.policy().admission.tickets.is_some() {
                        let until = deadline.as_secs_f64() - now.as_secs_f64();
                        scheduler
                            .ticket_lobby_retry_after(
                                &lobby_id,
                                until,
                                estimate.or(retry_after_s),
                                now,
                            )
                            .or(estimate)
                    } else {
                        estimate
                    };
                    if scheduler.policy().admission.retry_after_estimate.is_some() {
                        scheduler.note_retry_after(&lobby_id, estimate);
                    }
                    estimate
                };
                if let Some(seconds) = estimate.or(retry_after_s) {
                    request.set_retry_after(Duration::from_secs_f64(seconds.max(0.0)));
                }
            }
        }
        Ok(request)
    }

    async fn on_event(&mut self, event: ClassifyEvent) {
        let now = self.inner.clock.now();
        let inner = &self.inner;
        // ks2b `forecast.class_backlog`: an engine's own gauges (offline replay with
        // `DYNAMO_REPLAY_ENGINE_GAUGES=1`) go to the forecast and nowhere else.
        if let ClassifyEvent::EngineGauge {
            worker,
            running_requests,
            waiting_requests,
            active_blocks,
            waiting_tokens,
            ..
        } = &event
        {
            if inner
                .forecast
                .as_ref()
                .is_some_and(|(f, _)| f.config().class_backlog)
            {
                crate::forecast::gauge_debug_received(*worker);
                inner.observe(Observation::Engine {
                    at: now,
                    gauge: crate::forecast::EngineGauge {
                        worker: *worker,
                        running_requests: *running_requests,
                        waiting_requests: *waiting_requests,
                        active_blocks: *active_blocks,
                        waiting_tokens: *waiting_tokens,
                    },
                });
            }
            // f49k `admission.hot_budget: measured`: the engine's device KV and waiting queue, this
            // worker's hot measurement (a refresh). Nothing when off.
            if let ClassifyEvent::EngineGauge {
                worker,
                waiting_requests,
                active_blocks,
                total_blocks,
                waiting_tokens,
                ..
            } = &event
            {
                let grew = {
                    let mut scheduler = inner.scheduler.lock();
                    scheduler.hot_measured_on()
                        && scheduler.hot_gauge(
                            *worker,
                            crate::hot_measure::Reading::from_gauge(
                                *active_blocks,
                                *waiting_requests,
                                *waiting_tokens,
                                *total_blocks,
                                u64::from(inner.block_size.max(1)),
                            ),
                        )
                };
                if grew {
                    inner.notify();
                }
            }
            return;
        }
        // f49h `admission.aimd`: the class's turns per worker, their TTFT and ITL.
        if let Some(aimd) = &inner.aimd {
            aimd.lock().on_event(&event, now);
        }
        // f71pool `pool.headroom`: the pool's own measure of priority turns (first token, then
        // completion with its output tokens); nothing when that is off.
        if let Some(pool) = &inner.pool {
            match &event {
                ClassifyEvent::Responding { request_id, .. } => pool.first_token(request_id, now),
                ClassifyEvent::Completed {
                    request_id,
                    output_tokens,
                    ..
                } => pool.completed(request_id, *output_tokens, now),
                _ => {}
            }
        }
        match &event {
            ClassifyEvent::Completed { request_id, .. }
            | ClassifyEvent::Aborted { request_id, .. } => {
                if inner
                    .scheduler
                    .lock()
                    .union_admission()
                    .prefill_budget
                    .is_some()
                {
                    inner.prefill_meter.lock().finished(request_id);
                }
                if let Some(pool) = &inner.pool {
                    pool.finished(request_id);
                }
                if let Some(affinity) = &inner.affinity {
                    affinity.finished(request_id);
                }
                if let Some(busy) = &inner.busy {
                    busy.finished(request_id);
                }
                if let Some(route) = &inner.route {
                    route.finished(request_id);
                }
            }
            ClassifyEvent::Placed {
                request_id,
                cached_tokens,
                ..
            } => {
                inner
                    .scheduler
                    .lock()
                    .set_placed(request_id, *cached_tokens as u64);
            }
            ClassifyEvent::Sent { request_id, worker } => {
                let prefill = {
                    let scheduler = inner.scheduler.lock();
                    scheduler
                        .union_admission()
                        .prefill_budget
                        .is_some()
                        .then(|| scheduler.request_prefill_on(request_id, Some(*worker)))
                        .flatten()
                };
                if let Some((tokens, class)) = prefill {
                    inner
                        .prefill_meter
                        .lock()
                        .sent(request_id, *worker, &class, tokens, now);
                }
                if inner.scheduler.lock().policy().admission.go_target
                    && let Some(session) = inner
                        .scheduler
                        .lock()
                        .session_of(request_id)
                        .map(str::to_owned)
                {
                    let mut last = inner.last_worker.lock();
                    if last.len() >= MAX_LAST_WORKERS && !last.contains_key(&session) {
                        last.clear();
                    }
                    last.insert(session, *worker);
                }
                if let Some(affinity) = &inner.affinity {
                    affinity.sent(request_id, *worker);
                }
                if let Some(route) = &inner.route {
                    route.sent(request_id, *worker);
                }
                // f58pool `pool.demand: admitted`: the request's tokens count from its send.
                if let Some(pool) = &inner.pool {
                    pool.sent(request_id, now, worker.worker_id);
                }
            }
            _ => {}
        }
        let settled = {
            let mut scheduler = inner.scheduler.lock();
            // f49j `admission.step_target`: every class's decode per worker (nothing when off).
            scheduler.step_target_event(&event, now);
            match event {
                ClassifyEvent::Sent { request_id, worker } => {
                    scheduler.on_sent(&request_id);
                    scheduler.hot_measure_sent(&request_id, worker);
                    inner.observe(Observation::Sent {
                        request_id,
                        worker,
                        at: now,
                    });
                    false
                }
                ClassifyEvent::Responding { request_id, worker } => {
                    scheduler.on_first_token(&request_id, now);
                    inner.observe(Observation::FirstToken {
                        request_id,
                        worker,
                        at: now,
                    });
                    false
                }
                ClassifyEvent::Completed {
                    request_id,
                    context_tokens,
                    output_tokens,
                    ..
                } => {
                    tracing::trace!(
                        %request_id,
                        ?context_tokens,
                        ?output_tokens,
                        "session scheduler request completed"
                    );
                    scheduler.on_completed_with_output(
                        &request_id,
                        context_tokens,
                        output_tokens,
                        now,
                    );
                    inner.observe(Observation::Completed {
                        request_id,
                        at: now,
                        context_tokens: context_tokens.map(|t| t as u64),
                    });
                    true
                }
                ClassifyEvent::Aborted { request_id, .. } => {
                    scheduler.on_aborted_at(&request_id, now);
                    inner.observe(Observation::Aborted { request_id });
                    true
                }
                _ => false,
            }
        };
        if settled {
            let ended = inner.scheduler.lock().take_ended();
            inner.note_ended(now, ended);
        }
        if self.inner.keyed() {
            let mut scheduler = inner.scheduler.lock();
            inner.wake_changed(&mut scheduler);
        }
        if settled {
            self.inner.notify();
        }
    }

    fn dispatch_gate(&self) -> Option<Arc<dyn DispatchGate>> {
        if self.inner.scheduler.lock().mode() == Mode::Off {
            return None;
        }
        Some(Arc::new(SessionSchedulerGate {
            inner: Arc::clone(&self.inner),
        }))
    }
}

/// The harm per class the forecast predicts for a turn of `class` with `isl` new tokens at its
/// best placement among `rows` (fewest predicted misses, own and pushed over): the simulator's
/// `harmAbove` on the forecast's pick. Empty when the forecast has no sample.
fn best_placement_harm(
    forecast: &Forecast,
    class: &str,
    isl: u64,
    rows: &[WorkerRow],
) -> Vec<(String, f64)> {
    rows.iter()
        .filter_map(|r| {
            forecast.what_if_by_class(&crate::forecast::WhatIf {
                class: class.to_owned(),
                worker: r.worker,
                new_tokens: isl,
                prefix_tokens: 0,
            })
        })
        .min_by_key(|(outcome, _)| outcome.misses())
        .map(|(_, by_class)| {
            by_class
                .into_iter()
                .map(|(c, n)| (c, f64::from(n)))
                .collect()
        })
        .unwrap_or_default()
}

/// The forecast-harm check's input (`admission.forecast_harm_new`, ks2b): the least weighted harm
/// a new session of `class` with a first turn of `isl` tokens does to its class and the classes
/// above it over `rows` (`web/sim.js` `admitLobby`: the best `evaluate` over the eligible workers).
/// `None` when the forecast has no sample.
fn best_new_session_harm(
    forecast: &Forecast,
    class: &str,
    isl: u64,
    rows: &[WorkerRow],
) -> Option<f64> {
    rows.iter()
        .filter_map(|r| {
            forecast.new_session_harm(&crate::forecast::WhatIf {
                class: class.to_owned(),
                worker: r.worker,
                new_tokens: isl,
                prefix_tokens: 0,
            })
        })
        .min_by(f64::total_cmp)
}

/// Surge admission's facts (`admission.surge`): (seconds the fleet's queued prefill takes to
/// clear, the head's own prefill seconds, the fleet's prefill rate in tokens per second). Each worker clears its queue at its chunk over the step
/// model's time for one chunk; the fleet clears the sum of queues at the sum of rates. `None`
/// without a sample or a step model answer.
fn surge_facts(
    model: &dyn crate::forecast::StepModel,
    rows: &[WorkerRow],
    isl: u64,
    default_chunk: u32,
) -> Option<(f64, f64, f64)> {
    let tokens = |t: u64| u32::try_from(t).unwrap_or(u32::MAX);
    let mut queued = 0.0;
    let mut rate = 0.0;
    for r in rows {
        let chunk = r
            .chunk
            .filter(|c| *c > 0)
            .unwrap_or(u64::from(default_chunk))
            .max(1);
        let step = model.prefill_s(1, tokens(chunk), 0)?;
        if step <= 0.0 {
            continue;
        }
        queued += r.prefill_tokens as f64;
        rate += chunk as f64 / step;
    }
    if rate <= 0.0 {
        return None;
    }
    let own = model.prefill_s(1, tokens(isl), 0).unwrap_or(0.0);
    Some((queued / rate, own, rate))
}

/// The scheduler as the router's `DispatchGate`.
pub struct SessionSchedulerGate {
    inner: Arc<Inner>,
}

/// The gate's view for [`DispatchGate::report`]: the KV budget it decides on and its sessions per
/// class, as one JSON object. Class names are policy-file identifiers, written with `{:?}`.
pub fn gate_report(scheduler: &Scheduler, hbm: u64, workers: u64) -> String {
    let admission = &scheduler.policy().admission;
    let dram = admission.dram_beyond_hbm(hbm, workers);
    let kv = scheduler.kv_budget(hbm, dram);
    // Per class: active, idle, lobby, paused sessions; live context tokens; lobby input tokens.
    let mut per: std::collections::BTreeMap<&str, [u64; 6]> = scheduler
        .policy()
        .classes
        .iter()
        .map(|c| (c.name.as_str(), [0; 6]))
        .collect();
    for s in scheduler.sessions().iter() {
        let Some(row) = per.get_mut(s.class.as_str()) else {
            continue;
        };
        match s.state {
            SessionState::Active => row[0] += 1,
            SessionState::Idle => row[1] += 1,
            SessionState::Lobby => {
                row[2] += 1;
                row[5] += s.pending_isl;
            }
            SessionState::Paused => row[3] += 1,
            SessionState::Ended => {}
        }
        if s.is_live() {
            row[4] += s.context_tokens;
        }
    }
    let total_cap = (admission.total_headroom * (kv.hbm_tokens + kv.dram_tokens) as f64) as u64;
    let classes: Vec<String> = scheduler
        .policy()
        .classes
        .iter()
        .map(|c| {
            let r = per[c.name.as_str()];
            // f49k: with `admission.class_overcommit` or `admission.hot_budget: measured`, the cap
            // the gate's fit uses (pacing, release, target loop and step target scales, the
            // overcommits); otherwise the class's share of HBM, as before.
            let hot_cap = if !admission.class_overcommit.is_empty()
                || admission.hot_budget.is_measured()
                || scheduler.sla_budget_on()
            {
                // f49k2: as `admission.class_overcommit_mode` sets it, each class at its scale.
                // f49l: with `admission.sla_budget`, its share too (1 without it).
                let scale = |x: &crate::config::ClassPolicy| {
                    scheduler.pace_scale(&x.name)
                        * scheduler.step_target_scale(&x.name)
                        * scheduler.sla_scale(&x.name)
                };
                crate::admit::hot_cap(scheduler.policy(), c, kv.hbm_tokens, &scale)
            } else {
                (c.admission.gate_share * kv.hbm_tokens as f64) as u64
            };
            format!(
                "{:?}:{{\"active\":{},\"idle\":{},\"lobby\":{},\"paused\":{},\"live_context_tokens\":{},\"lobby_isl_tokens\":{},\"hot_cap_tokens\":{}}}",
                c.name, r[0], r[1], r[2], r[3], r[4], r[5], hot_cap
            )
        })
        .collect();
    format!(
        "{{\"hbm_tokens\":{},\"dram_tokens\":{},\"hot_tokens\":{},\"live_context_tokens\":{},\"total_cap_tokens\":{},\"classes\":{{{}}}}}",
        kv.hbm_tokens,
        kv.dram_tokens,
        kv.hot_tokens,
        kv.total_tokens,
        total_cap,
        classes.join(",")
    )
}

impl DispatchGate for SessionSchedulerGate {
    fn check(&self, head: &DispatchHead<'_>) -> DispatchDecision {
        self.check_wake(head).0
    }

    fn report(&self) -> Option<String> {
        // Read the host's capacity directly, not through the gate's cache, so a report never
        // moves the cache's refresh times.
        let (hbm, workers) = (self.inner.capacity)();
        let scheduler = self.inner.scheduler.lock();
        Some(gate_report(&scheduler, hbm, workers))
    }

    /// A head that waits behind an earlier lobby session ([`GateWake::Behind`]) is keyed on that
    /// session and its own; every other answer is [`HoldWake::Any`].
    fn check_wake(&self, head: &DispatchHead<'_>) -> (DispatchDecision, HoldWake) {
        let (Some(_), Some(session)) = (head.request_id(), head.session_context()) else {
            return (DispatchDecision::Go { target: None }, HoldWake::Any);
        };
        let now = self.inner.clock.now();
        self.inner.maybe_rotate(now);
        let (hbm, workers) = self.inner.capacity(now);
        let mut scheduler = self.inner.scheduler.lock();
        let dram = scheduler.policy().admission.dram_beyond_hbm(hbm, workers);
        let mut kv = scheduler.kv_budget(hbm, dram);
        let admission = scheduler.policy().admission.clone();
        self.inner.account(now, &admission, &mut kv);
        // f49q `admission.starvation_override`: the bucket's size is the fleet's workers.
        if admission.starvation_override.is_some() {
            scheduler.set_fleet_workers(workers);
        }
        let rows = self.inner.kv_rows.lock().clone();
        let kv_room = (admission.shed_by_slack && !rows.is_empty()).then(|| {
            let need = head.isl_tokens() as f64;
            rows.iter()
                .any(|r| r.active_tokens as f64 + need <= 0.95 * r.kv_tokens as f64)
        });
        let limited = rows.iter().any(|r| r.capacity.max_num_seqs().is_some());
        // f49q: the override reads the sequence room too.
        let seq_room = ((admission.hold_at_max_num_seqs
            || admission.starvation_override.is_some())
            && limited)
            .then(|| {
                rows.iter().any(|r| {
                    r.capacity
                        .max_num_seqs()
                        .is_none_or(|limit| (r.running as u64) < limit)
                })
            });
        let harm = if admission.shed_by_slack && admission.shed_forecast_harm {
            let class = scheduler
                .sessions()
                .get(session.session_id())
                .filter(|s| s.state != crate::session::SessionState::Lobby)
                .map(|s| s.class.clone());
            match (&self.inner.forecast, class) {
                (Some((forecast, _)), Some(class)) => {
                    best_placement_harm(forecast, &class, head.isl_tokens() as u64, &rows)
                }
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        };
        // ks2b `admission.forecast_harm_new`: the harm of a new session's best placement.
        let new_harm = match (&admission.forecast_harm_new, &self.inner.forecast) {
            (Some(cfg), Some((forecast, _))) => scheduler
                .sessions()
                .get(session.session_id())
                .filter(|s| {
                    s.state == SessionState::Lobby && !cfg.exempt_classes.contains(&s.class)
                })
                .and_then(|s| {
                    best_new_session_harm(forecast, &s.class, head.isl_tokens() as u64, &rows)
                }),
            _ => None,
        };
        // f49d `DYNAMO_HARM_DEBUG=1`: count what the harm check saw (no decision changes).
        if scheduler.harm_debug_on()
            && let Some(cfg) = &admission.forecast_harm_new
        {
            let found = scheduler
                .sessions()
                .get(session.session_id())
                .map(|s| (s.class.clone(), s.state));
            let check = crate::harm_debug::classify(
                found.as_ref().map(|(c, st)| (c.as_str(), *st)),
                &cfg.exempt_classes,
                self.inner.forecast.is_some(),
                rows.is_empty(),
                new_harm,
            );
            let class = found.map_or_else(|| "?".to_owned(), |(c, _)| c);
            scheduler.harm_debug_check(&class, check, now);
        }
        let surge_full = admission.surge.as_ref().and_then(|_| {
            let model = self.inner.step_model.as_ref()?;
            surge_facts(
                model.as_ref(),
                &rows,
                head.isl_tokens() as u64,
                scheduler.policy().forecast.chunk_tokens,
            )
        });
        let surge = surge_full.map(|(backlog, own, _)| (backlog, own));
        // The two-resource gate (f42): each worker's prefill budget and the fleet's.
        let rooms = admission.prefill_budget.as_ref().and_then(|cfg| {
            let model = self.inner.step_model.as_ref()?;
            let mut meter = self.inner.prefill_meter.lock();
            meter.trim(now, cfg.window_s);
            prefill_rooms(model.as_ref(), &rows, &meter, scheduler.policy(), cfg)
        });
        // The budget's calibration against the top class's ITL quantile.
        let cal = scheduler.prefill_cal(now);
        let rooms = rooms.map(|mut r| {
            for x in &mut r {
                x.budget *= cal;
            }
            r
        });
        let prefill = rooms.as_ref().filter(|r| !r.is_empty()).map(|r| {
            let budget: f64 = r.iter().map(|x| x.budget).sum();
            let window = admission
                .prefill_budget
                .as_ref()
                .map_or(1.0, |c| c.window_s);
            let used = self.inner.prefill_meter.lock().fleet_tokens() as f64 / window;
            (budget, used)
        });
        let was_lobby = scheduler
            .sessions()
            .get(session.session_id())
            .filter(|s| s.state == crate::session::SessionState::Lobby)
            .map(|s| (s.pending_isl, s.class.clone()));
        let facts = HeadFacts {
            queued_at: self.inner.clock.time_of_host(head.queued_at()),
            kv_room,
            seq_room,
            harm,
            surge,
            prefill,
            new_harm,
        };
        let class = scheduler
            .sessions()
            .get(session.session_id())
            .map(|s| s.class.clone());
        let busy_frac = scheduler
            .policy()
            .class(class.as_deref())
            .admission
            .busy_frac
            .unwrap_or(1.0);
        let (answer, wake) = scheduler.gate_check_head_wake(session.session_id(), now, &kv, facts);
        if let Some((budget, used)) = prefill
            && let Some(log) = self.inner.f42_log.lock().as_mut()
        {
            log.record(now, budget, used, || scheduler.take_loop_stats());
        }
        // An admitted new session's first turn goes where its onboarding fits best (f42).
        let onboard_target = match (&answer, &rooms, &was_lobby, &admission.prefill_budget) {
            (GateAnswer::Go, Some(rooms), Some((isl, class)), Some(cfg)) if cfg.place_onboard => {
                // Every class's onboarding stays off workers decoding the class with the tightest ITL
                // limit, priority's own included: its prefill is what pushes those decodes over.
                let _ = class;
                let protect = scheduler
                    .policy()
                    .classes
                    .iter()
                    .filter_map(|c| c.service_level.as_ref().map(|l| l.itl_s))
                    .fold(f64::INFINITY, f64::min);
                place_onboard(rooms, *isl as f64 / cfg.window_s, protect)
            }
            _ => None,
        };
        // f47j: a large turn goes where its prefill keeps the worker under the per-worker cap.
        let turn_target = match (
            &answer,
            &rooms,
            &admission.prefill_budget,
            head.request_id(),
        ) {
            (GateAnswer::Go, Some(rooms), Some(cfg), Some(id)) if onboard_target.is_none() => {
                cfg.place_turns.as_ref().and_then(|pt| {
                    let (best, per) = scheduler.request_compute(id)?;
                    if best < pt.min_tokens {
                        return None;
                    }
                    let protect = scheduler
                        .policy()
                        .classes
                        .iter()
                        .filter_map(|c| c.service_level.as_ref().map(|l| l.itl_s))
                        .fold(f64::INFINITY, f64::min);
                    place_turn(rooms, &per, cfg.window_s, pt.cap_tok_s, protect)
                })
            }
            _ => None,
        };
        let onboard_target = onboard_target.or(turn_target);
        // f49i `admission.worker_step_budget`: a turn every rule above lets go (enforce mode) waits
        // while no worker keeps the step budget, at most its class's `max_lobby_wait_s` since it was
        // queued; it is asked again at each completion (the notify in `on_event`) and at least every
        // `RECHECK_S`. A gate target that breaks the budget is dropped (the forecast route picks
        // among the workers that keep it); with no forecast route, the fitting worker computing
        // least becomes the target.
        let wsb = match (&self.inner.forecast, head.request_id()) {
            (Some((forecast, _)), Some(id))
                if forecast.step_budget().is_some()
                    && matches!(answer, GateAnswer::Go)
                    && scheduler.mode() == Mode::Enforce =>
            {
                let class = scheduler.policy().class(class.as_deref()).name.clone();
                forecast
                    .step_budget_verdict(id, &class, head.isl_tokens() as u64)
                    .map(|verdict| (Arc::clone(forecast), id, class, verdict))
            }
            _ => None,
        };
        let (answer, onboard_target) = match &wsb {
            None => (answer, onboard_target),
            Some((forecast, id, class, verdict)) => {
                let budget = forecast.step_budget().expect("checked above");
                let wait = scheduler
                    .policy()
                    .class(Some(class.as_str()))
                    .admission
                    .max_lobby_wait_s;
                let waited = self
                    .inner
                    .clock
                    .time_of_host(head.queued_at())
                    .map_or(0.0, |at| now.since(at).as_secs_f64());
                if !verdict.any_fits() && waited < wait {
                    budget.record_gate(
                        now,
                        id,
                        class,
                        verdict,
                        crate::step_budget::GateOutcome::Held,
                    );
                    let until = now.after(Duration::from_secs_f64(
                        (wait - waited).min(crate::step_budget::RECHECK_S),
                    ));
                    (GateAnswer::Hold { until: Some(until) }, None)
                } else {
                    let refused = verdict.refused().next().is_some();
                    let target = match onboard_target.filter(|t| verdict.fits(*t)) {
                        Some(t) => Some(t),
                        None if refused && self.inner.route.is_none() => verdict.best_fitting(),
                        None => None,
                    };
                    let outcome = if verdict.any_fits() {
                        crate::step_budget::GateOutcome::Go { chosen: target }
                    } else {
                        crate::step_budget::GateOutcome::Overdue
                    };
                    budget.record_gate(now, id, class, verdict, outcome);
                    (answer, target)
                }
            }
        };
        // f49i: the targeted `Go` chooses among the workers that keep the budget.
        let wsb_rows: Option<Vec<WorkerRow>> = wsb.as_ref().map(|(_, _, _, verdict)| {
            rows.iter()
                .filter(|r| verdict.fits(r.worker))
                .copied()
                .collect()
        });
        if let Some((backlog, _, rate)) = surge_full
            && let Some(log) = self.inner.surge_log.lock().as_mut()
        {
            log.record(now, backlog, rate, || scheduler.take_surge_counts());
        }
        if self.inner.keyed() {
            self.inner.wake_changed(&mut scheduler);
        }
        drop(scheduler);
        // f49h `admission.aimd`: a turn of the AIMD class the gate would let go waits while the
        // worker it is bound for is at its limit (oldest sessions first), and goes there otherwise.
        if matches!(answer, GateAnswer::Go)
            && let (Some(aimd), Some(request_id)) = (&self.inner.aimd, head.request_id())
            && self.inner.scheduler.lock().mode() == Mode::Enforce
        {
            let answer = aimd.lock().check(request_id, session.session_id(), now);
            match answer {
                crate::aimd::AimdAnswer::Untracked | crate::aimd::AimdAnswer::Go(None) => {}
                crate::aimd::AimdAnswer::Go(Some(worker)) => {
                    return (
                        DispatchDecision::Go {
                            target: Some(worker),
                        },
                        HoldWake::Any,
                    );
                }
                crate::aimd::AimdAnswer::Hold(until) => {
                    return (
                        DispatchDecision::Hold {
                            until: self.inner.clock.host_instant(until),
                        },
                        HoldWake::Any,
                    );
                }
            }
        }
        let out = match answer {
            GateAnswer::Go if admission.go_target => {
                let last = self
                    .inner
                    .last_worker
                    .lock()
                    .get(session.session_id())
                    .copied();
                let target = go_target(wsb_rows.as_deref().unwrap_or(&rows), last, busy_frac);
                (DispatchDecision::Go { target }, HoldWake::Any)
            }
            GateAnswer::Go => (
                DispatchDecision::Go {
                    target: onboard_target,
                },
                HoldWake::Any,
            ),
            GateAnswer::Hold { until } => {
                let decision = DispatchDecision::Hold {
                    until: until.and_then(|at| self.inner.clock.host_instant(at)),
                };
                let wake = match wake {
                    GateWake::Behind { ahead } if self.inner.keyed() => HoldWake::Keys(vec![
                        session_wake_key(&ahead),
                        session_wake_key(session.session_id()),
                    ]),
                    _ => HoldWake::Any,
                };
                (decision, wake)
            }
        };
        // f49j2 `step_target.decode_cap`: a standard or batch turn let go to a worker at its decode
        // cap goes to a worker under its cap, or holds (rechecked at each completion). Nothing
        // when off.
        if let (DispatchDecision::Go { target }, Some(request_id)) = (&out.0, head.request_id()) {
            let verdict = self.inner.scheduler.lock().step_target_cap(
                request_id,
                session.session_id(),
                *target,
                now,
            );
            match verdict {
                crate::step_target::CapVerdict::Pass => {}
                crate::step_target::CapVerdict::Go(worker) => {
                    return (
                        DispatchDecision::Go {
                            target: Some(worker),
                        },
                        HoldWake::Any,
                    );
                }
                crate::step_target::CapVerdict::Hold(until) => {
                    return (
                        DispatchDecision::Hold {
                            until: self.inner.clock.host_instant(until),
                        },
                        HoldWake::Any,
                    );
                }
            }
        }
        out
    }

    fn dispatched(&self, head: &DispatchHead<'_>, target: Option<WorkerWithDpRank>) {
        let (Some(request_id), Some(session)) = (head.request_id(), head.session_context()) else {
            return;
        };
        let now = self.inner.clock.now();
        if let Some(aimd) = &self.inner.aimd {
            aimd.lock().dispatched(request_id, target, now);
        }
        {
            let mut scheduler = self.inner.scheduler.lock();
            if scheduler.union_admission().prefill_budget.is_some()
                && let Some((tokens, _)) = scheduler.request_prefill_on(request_id, target)
            {
                self.inner
                    .prefill_meter
                    .lock()
                    .dispatched(request_id, tokens, now, target);
            }
            scheduler.on_dispatched(request_id, session.session_id(), now);
            if self.inner.keyed() {
                self.inner.wake_changed(&mut scheduler);
            }
        }
        // The lobby moved: a session behind this one in its class may fit now.
        self.inner.notify();
    }

    /// A host that honours keyed holds turns on the lobby change journal they are woken from.
    fn attach(&self, recheck: DispatchRecheck) {
        if recheck.honours_keys() {
            let mut scheduler = self.inner.scheduler.lock();
            scheduler.track_lobby_changes();
            // Changes before the host attached concern no hold.
            let _ = scheduler.take_lobby_changes();
            *self.inner.gate_was_on.lock() = Some(scheduler.session_gate_on());
        }
        let _ = self.inner.recheck.set(recheck);
    }

    fn gates_every_class(&self) -> bool {
        self.inner
            .scheduler
            .lock()
            .policy()
            .admission
            .queue_all_classes
    }
}

/// The targeted `Go` (`admission.go_target`): the session's last worker when it has room for
/// the class, else the least prefill-loaded worker with room, else none.
fn go_target(
    rows: &[WorkerRow],
    last: Option<WorkerWithDpRank>,
    busy_frac: f64,
) -> Option<WorkerWithDpRank> {
    let room = |r: &WorkerRow| {
        let load = dynamo_kv_router::plugins::request_classifier::WorkerLoadView {
            available: true,
            active_prefill_tokens: r.prefill_tokens,
            active_decode_blocks: 0,
            decode_cost_blocks: 0.0,
            active_requests: r.running,
        };
        !crate::busy::past_threshold(&load, r.capacity, busy_frac)
    };
    if let Some(home) = last
        && rows.iter().any(|r| r.worker == home && room(r))
    {
        return Some(home);
    }
    rows.iter()
        .filter(|r| room(r))
        .min_by(|a, b| {
            a.prefill_tokens
                .cmp(&b.prefill_tokens)
                .then_with(|| a.worker.cmp(&b.worker))
        })
        .map(|r| r.worker)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tokio::time::Instant;

    use super::*;
    use crate::clock::ManualClock;

    #[test]
    fn onboarding_goes_to_the_roomiest_worker_without_a_tighter_class_in_flight() {
        let w = |id| WorkerWithDpRank::new(id, 0);
        let room = |id, budget, used, tightest_itl| PrefillRoom {
            worker: w(id),
            budget,
            used,
            tightest_itl,
        };
        let rooms = [
            room(1, 10_000.0, 1_000.0, 0.05),
            room(2, 8_000.0, 2_000.0, 0.1),
            room(3, 9_000.0, 1_000.0, f64::INFINITY),
        ];
        // Worker 1 has the most room but decodes priority (ITL 0.05); worker 3 wins over 2.
        assert_eq!(place_onboard(&rooms, 5_000.0, 0.05), Some(w(3)));
        // Too big for the clean workers: the roomiest of all.
        assert_eq!(place_onboard(&rooms, 8_500.0, 0.05), Some(w(1)));
        // Too big for any: the router places it.
        assert_eq!(place_onboard(&rooms, 20_000.0, 0.05), None);
    }

    #[test]
    fn the_prefill_meter_counts_tokens_and_classes_per_worker() {
        let w = WorkerWithDpRank::new(1, 0);
        let mut m = PrefillMeter::default();
        let t = Time::from_secs_f64;
        m.dispatched("a", 1_000, t(0.0), Some(w));
        assert_eq!(m.worker_tokens(w), 1_000);
        assert_eq!(m.fleet_tokens(), 1_000);
        m.sent("a", w, "priority", 1_000, t(1.0));
        m.sent("b", w, "standard", 500, t(2.0));
        assert_eq!(m.worker_tokens(w), 1_500);
        let policy = ServicePolicy::default();
        assert_eq!(m.tightest_itl(w, &policy), 0.05);
        m.finished("a");
        assert_eq!(m.tightest_itl(w, &policy), 0.1);
        m.trim(t(31.0), 30.0);
        assert_eq!(m.worker_tokens(w), 500);
        m.trim(t(40.0), 30.0);
        assert_eq!(m.fleet_tokens(), 0);
    }

    #[test]
    fn the_gate_report_counts_sessions_per_class_and_the_kv_budget() {
        let mut s = Scheduler::new(ServicePolicy::default());
        let t0 = Time::from_secs_f64(0.0);
        s.on_classify(Some("r1"), "a", None, None, 5_000, t0);
        s.on_dispatched("r1", "a", t0);
        s.on_classify(Some("r2"), "b", None, None, 700, t0);
        let report: serde_json::Value = serde_json::from_str(&gate_report(&s, 100_000, 2)).unwrap();
        assert_eq!(report["hbm_tokens"], 100_000);
        let class = &report["classes"][s.policy().default_class.as_str()];
        assert_eq!(
            class["active"].as_u64().unwrap() + class["lobby"].as_u64().unwrap(),
            2
        );
        assert_eq!(
            class["lobby_isl_tokens"].as_u64().unwrap(),
            700 * class["lobby"].as_u64().unwrap()
        );
        assert_eq!(
            report["live_context_tokens"].as_u64().unwrap(),
            class["live_context_tokens"].as_u64().unwrap()
        );
    }

    #[test]
    fn request_attributes_come_from_session_final_and_agent_headers() {
        let headers: std::collections::BTreeMap<String, Vec<String>> = [
            (HEADER_TENANT, "acme"),
            (HEADER_HINT_OSL, "250"),
            (HEADER_HINT_NEXT_THINK_S, "7.5"),
            (HEADER_HINT_NEXT_TOKENS, "not-a-number"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), vec![v.to_owned()]))
        .collect();
        let context = SessionContext::new("s".into(), Some("p".into()), Some(true), None)
            .with_agent_headers(Arc::new(headers));
        let attributes = request_attributes(&context);
        assert_eq!(attributes.tenant.as_deref(), Some("acme"));
        assert_eq!(attributes.signals.osl_tokens, Some(250));
        assert_eq!(attributes.signals.next_think_s, Some(7.5));
        // Untrusted input: a malformed value is ignored, not an error.
        assert_eq!(attributes.signals.next_turn_tokens, None);
        assert!(attributes.session_final);
        let plain = SessionContext::new("s".into(), None, None, None);
        assert_eq!(request_attributes(&plain), RequestAttributes::default());
    }

    #[test]
    fn session_kind_user_and_the_user_turn_come_from_headers_and_the_input_trigger() {
        let headers =
            |pairs: &[(&str, &str)]| -> Arc<std::collections::BTreeMap<String, Vec<String>>> {
                Arc::new(
                    pairs
                        .iter()
                        .map(|(k, v)| ((*k).to_owned(), vec![(*v).to_owned()]))
                        .collect(),
                )
            };
        let hinted =
            SessionContext::new("s".into(), None, None, None).with_agent_headers(headers(&[
                (HEADER_HINT_SESSION_KIND, "Agent"),
                (HEADER_USER_ID, "u000007"),
                (HEADER_HINT_LAST_ROLE, "user"),
            ]));
        let a = request_attributes(&hinted);
        assert_eq!(a.session_kind, Some(crate::session::SessionKind::Agent));
        assert_eq!(a.user.as_deref(), Some("u000007"));
        assert!(a.next_is_user);
        // Unknown kinds are ignored; a tool result is not a user turn.
        let tool =
            SessionContext::new("s".into(), None, None, None).with_agent_headers(headers(&[
                (HEADER_HINT_SESSION_KIND, "robot"),
                (HEADER_HINT_LAST_ROLE, "tool"),
            ]));
        let b = request_attributes(&tool);
        assert_eq!((b.session_kind, b.next_is_user), (None, false));
        // The frontend's own classification of the last message counts without any header.
        let user_turn = SessionContext::new(
            "s".into(),
            None,
            None,
            Some(WorkerSelectionInputTrigger::UserMessage),
        );
        assert!(request_attributes(&user_turn).next_is_user);
    }

    #[test]
    fn a_tool_result_or_a_user_message_is_known_and_other_triggers_leave_it_to_the_gap() {
        let with = |trigger| {
            request_attributes(&SessionContext::new("s".into(), None, None, Some(trigger)))
        };
        let tool = with(WorkerSelectionInputTrigger::ToolResult);
        assert_eq!((tool.next_is_user, tool.next_known), (false, true));
        let user = with(WorkerSelectionInputTrigger::UserMessage);
        assert_eq!((user.next_is_user, user.next_known), (true, true));
        let other = with(WorkerSelectionInputTrigger::Other);
        assert_eq!((other.next_is_user, other.next_known), (false, false));
    }

    #[tokio::test]
    async fn run_order_sets_a_queue_ordering_key_and_fifo_sets_none() {
        for (order, keyed) in [
            (crate::config::TurnOrder::RunLas, true),
            (crate::config::TurnOrder::ServerWaitDeficit, true),
            (crate::config::TurnOrder::Fifo, false),
        ] {
            let mut policy = ServicePolicy {
                mode: Mode::Enforce,
                ..Default::default()
            };
            policy.admission.turn_order = order;
            let clock = Arc::new(ManualClock::new());
            let mut classifier =
                SessionSchedulerClassifier::new(policy, clock, Arc::new(|| (1_000_000, 1)));
            let out = classifier
                .classify(request("a1", "a", 10), &NoWorkers)
                .unwrap();
            assert_eq!(out.order_key().is_some(), keyed, "{order:?}");
        }
    }

    fn setup(mode: Mode, hbm: u64) -> (SessionSchedulerClassifier, Arc<ManualClock>) {
        let clock = Arc::new(ManualClock::new());
        let policy = ServicePolicy {
            mode,
            ..Default::default()
        };
        let classifier =
            SessionSchedulerClassifier::new(policy, clock.clone(), Arc::new(move || (hbm, 1)));
        (classifier, clock)
    }

    fn request(id: &str, session: &str, tokens: usize) -> ClassifyRequest {
        ClassifyRequest::for_test(
            tokens,
            Some(id),
            Some(SessionContext::new(session.into(), None, None, None)),
        )
    }

    fn head<'a>(id: &'a str, session: &'a SessionContext, tokens: usize) -> DispatchHead<'a> {
        DispatchHead::new(
            Some(id),
            Some(session),
            tokens,
            None,
            "default",
            Instant::now(),
        )
    }

    fn attach_counter(gate: &dyn DispatchGate) -> Arc<AtomicUsize> {
        let wakes = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&wakes);
        gate.attach(DispatchRecheck::new(Arc::new(move || {
            counter.fetch_add(1, Ordering::Relaxed);
        })));
        wakes
    }

    fn done(id: &str, context_tokens: usize) -> ClassifyEvent {
        ClassifyEvent::Completed {
            request_id: id.into(),
            worker: WorkerWithDpRank::new(1, 0),
            context_tokens: Some(context_tokens),
            output_tokens: None,
        }
    }

    #[tokio::test]
    async fn enforce_holds_a_session_that_does_not_fit_until_room_frees() {
        let (mut classifier, clock) = setup(Mode::Enforce, 1000);
        let gate = classifier.dispatch_gate().unwrap();
        let wakes = attach_counter(gate.as_ref());
        let a = SessionContext::new("a".into(), None, None, None);
        let b = SessionContext::new("b".into(), None, None, None);

        classifier
            .classify(request("a1", "a", 600), &NoWorkers)
            .unwrap();
        assert_eq!(
            gate.check(&head("a1", &a, 600)),
            DispatchDecision::Go { target: None }
        );
        gate.dispatched(&head("a1", &a, 600), None);

        clock.advance(Duration::from_secs(1));
        classifier
            .classify(request("b1", "b", 300), &NoWorkers)
            .unwrap();
        // Hot room for standard: 0.9 × 1000 − 600 = 300; total: 950 − 600 = 350. b fits.
        assert_eq!(
            gate.check(&head("b1", &b, 300)),
            DispatchDecision::Go { target: None }
        );
        // Checking is a read: the session is still waiting until `dispatched`.
        classifier.with_scheduler(|s| {
            assert_eq!(s.sessions().get("b").unwrap().state, SessionState::Lobby)
        });
        gate.dispatched(&head("b1", &b, 300), None);

        let c = SessionContext::new("c".into(), None, None, None);
        classifier
            .classify(request("c1", "c", 200), &NoWorkers)
            .unwrap();
        // ManualClock has no host mapping, so the hold carries no deadline.
        assert_eq!(
            gate.check(&head("c1", &c, 200)),
            DispatchDecision::Hold { until: None }
        );
        assert_eq!(
            gate.check(&head("c1", &c, 200)),
            DispatchDecision::Hold { until: None }
        );

        let before = wakes.load(Ordering::Relaxed);
        classifier.on_event(done("a1", 600)).await;
        assert_eq!(
            wakes.load(Ordering::Relaxed),
            before + 1,
            "a completion rechecks held heads"
        );
        // a is idle with 600, b in flight with 300: hot 300 → room 600; total 900 → room 50.
        assert!(matches!(
            gate.check(&head("c1", &c, 200)),
            DispatchDecision::Hold { .. }
        ));
        let logged = classifier.with_scheduler(|s| s.decisions().len());
        assert_eq!(logged, 4, "one entry per distinct decision, not per check");
    }

    #[tokio::test]
    async fn shadow_never_holds_and_off_installs_no_gate() {
        let (mut classifier, _) = setup(Mode::Shadow, 10);
        let gate = classifier.dispatch_gate().unwrap();
        let a = SessionContext::new("a".into(), None, None, None);
        classifier
            .classify(request("a1", "a", 600), &NoWorkers)
            .unwrap();
        assert_eq!(
            gate.check(&head("a1", &a, 600)),
            DispatchDecision::Go { target: None }
        );
        classifier.with_scheduler(|s| assert_eq!(s.decisions().len(), 1));

        let (off, _) = setup(Mode::Off, 10);
        assert!(off.dispatch_gate().is_none());
        assert_eq!(off.required_worker_inputs(), WorkerInputs::NONE);
    }

    #[tokio::test]
    async fn sessionless_requests_pass_through() {
        let (mut classifier, _) = setup(Mode::Enforce, 10);
        let plain = ClassifyRequest::for_test(5000, Some("r"), None);
        let out = classifier.classify(plain, &NoWorkers).unwrap();
        assert_eq!(out.due_at(), None);
        classifier.with_scheduler(|s| assert!(s.sessions().is_empty()));
        let gate = classifier.dispatch_gate().unwrap();
        let headless = DispatchHead::new(Some("r"), None, 5000, None, "default", Instant::now());
        assert_eq!(gate.check(&headless), DispatchDecision::Go { target: None });
    }

    #[tokio::test(start_paused = true)]
    async fn enforce_sets_due_at_for_classes_that_expire() {
        let policy = ServicePolicy {
            mode: Mode::Enforce,
            ..Default::default()
        };
        let clock = Arc::new(crate::clock::HostClock::new());
        let mut classifier = SessionSchedulerClassifier::new(policy, clock, Arc::new(|| (1000, 1)));
        let ingress = Instant::now();
        let out = classifier
            .classify(request("a1", "a", 10), &NoWorkers)
            .unwrap();
        let due = out
            .due_at()
            .expect("standard expires after 60 s in the lobby");
        assert_eq!(due.duration_since(ingress), Duration::from_secs(60));
        // Priority admits at its limit instead, so it gets no deadline.
        let mut prio = ClassifyRequest::for_test(
            10,
            Some("p1"),
            Some(SessionContext::new("p".into(), None, None, None)),
        );
        prio.set_policy_class("priority");
        assert_eq!(
            classifier.classify(prio, &NoWorkers).unwrap().due_at(),
            None
        );
    }

    /// Worker rows with a cache estimate per tier: (worker, device, host, disk overlap blocks).
    struct TieredWorkers(Vec<(WorkerWithDpRank, f64, f64, f64)>);

    impl WorkerView for TieredWorkers {
        fn len(&self) -> usize {
            self.0.len()
        }

        fn info(&self, row: usize) -> Option<WorkerInfo> {
            self.0
                .get(row)
                .map(|r| WorkerInfo::new(r.0, WorkerCapacity::new(Some(4096), Some(1000))))
        }

        fn load(
            &self,
            _row: usize,
        ) -> Option<dynamo_kv_router::plugins::request_classifier::WorkerLoadView> {
            None
        }

        fn cache(
            &self,
            row: usize,
        ) -> Option<dynamo_kv_router::plugins::request_classifier::WorkerCacheView> {
            self.0.get(row).map(|r| {
                dynamo_kv_router::plugins::request_classifier::WorkerCacheView {
                    effective_overlap_blocks: r.1,
                    estimated_cached_tokens: 0,
                    device_overlap_blocks: r.1,
                    host_overlap_blocks: r.2,
                    disk_overlap_blocks: r.3,
                }
            })
        }
    }

    #[test]
    fn place_turn_keeps_large_turns_under_the_cap_and_off_priority_decodes() {
        let w = |i| WorkerWithDpRank::new(i, 0);
        let room = |i, used, itl| PrefillRoom {
            worker: w(i),
            budget: 10_000.0,
            used,
            tightest_itl: itl,
        };
        // Worker 1 decodes priority (ITL 0.05); 2 and 3 do not. Window 30 s, cap 3,000 tok/s.
        let rooms = [
            room(1, 500.0, 0.05),
            room(2, 2_000.0, f64::INFINITY),
            room(3, 1_000.0, f64::INFINITY),
        ];
        // 30,000 tokens is 1,000 tok/s on any worker: 2 and 3 both fit; the most cached (least
        // computed) of the clean ones wins even though 1 computes least.
        let per = [(w(1), 6_000), (w(2), 24_000), (w(3), 30_000)];
        assert_eq!(place_turn(&rooms, &per, 30.0, 3_000.0, 0.05), Some(w(2)));
        // 2 over the cap with this turn (2,000 + 1,500): 3 (1,000 + 1,500).
        let per = [(w(1), 6_000), (w(2), 45_000), (w(3), 45_000)];
        assert_eq!(place_turn(&rooms, &per, 30.0, 3_000.0, 0.05), Some(w(3)));
        // No clean worker fits: the priority worker when it fits.
        let per = [(w(1), 60_000), (w(2), 90_000), (w(3), 90_000)];
        assert_eq!(place_turn(&rooms, &per, 30.0, 3_000.0, 0.05), Some(w(1)));
        // Nothing fits: the router places it.
        let per = [(w(1), 90_000), (w(2), 90_000), (w(3), 90_000)];
        assert_eq!(place_turn(&rooms, &per, 30.0, 3_000.0, 0.05), None);
    }

    #[test]
    fn computed_tokens_count_device_host_and_disk_hits_as_cached_per_worker() {
        let (a, b, c) = (
            WorkerWithDpRank::new(1, 0),
            WorkerWithDpRank::new(2, 0),
            WorkerWithDpRank::new(3, 0),
        );
        // 1,000-token prompt, 16-token blocks: a holds 10 blocks in HBM (840 to compute); b holds 5 in
        // HBM, 20 in host memory and 10 on disk (35 blocks: 440 to compute); c holds nothing.
        let view = TieredWorkers(vec![
            (a, 10.0, 0.0, 0.0),
            (b, 5.0, 20.0, 10.0),
            (c, 0.0, 0.0, 0.0),
        ]);
        let (best, per) = computed_tokens(1_000, 16, &view);
        assert_eq!(per, vec![(a, 840), (b, 440), (c, 1_000)]);
        assert_eq!(best, 440);
        // A hit longer than the prompt computes nothing; no cache rows: the whole prompt.
        let full = TieredWorkers(vec![(a, 50.0, 30.0, 0.0)]);
        assert_eq!(computed_tokens(1_000, 16, &full), (0, vec![(a, 0)]));
        assert_eq!(computed_tokens(1_000, 16, &NoWorkers), (1_000, vec![]));
    }

    /// One worker row with a load, as the host builds it when `LOAD` is declared.
    struct LoadedWorker(WorkerWithDpRank);

    impl WorkerView for LoadedWorker {
        fn len(&self) -> usize {
            1
        }

        fn info(&self, row: usize) -> Option<WorkerInfo> {
            (row == 0).then(|| WorkerInfo::new(self.0, WorkerCapacity::new(Some(4096), Some(1000))))
        }

        fn load(
            &self,
            row: usize,
        ) -> Option<dynamo_kv_router::plugins::request_classifier::WorkerLoadView> {
            (row == 0).then_some(
                dynamo_kv_router::plugins::request_classifier::WorkerLoadView {
                    available: true,
                    active_prefill_tokens: 4096,
                    active_decode_blocks: 64,
                    decode_cost_blocks: 0.0,
                    active_requests: 2,
                },
            )
        }

        fn cache(
            &self,
            _row: usize,
        ) -> Option<dynamo_kv_router::plugins::request_classifier::WorkerCacheView> {
            None
        }
    }

    #[tokio::test]
    async fn the_forecast_follows_classify_and_events_without_the_scheduler_lock() {
        let clock = Arc::new(ManualClock::new());
        let mut policy = ServicePolicy::default();
        policy.forecast.enabled = true;
        let mut classifier =
            SessionSchedulerClassifier::new(policy, clock.clone(), Arc::new(|| (100_000, 1)))
                .with_block_size(16);
        assert_eq!(
            classifier.required_worker_inputs(),
            WorkerInputs::CACHE | WorkerInputs::LOAD
        );
        let forecast = classifier.forecast().expect("enabled");
        let w = WorkerWithDpRank::new(3, 0);
        classifier
            .classify(request("r1", "s", 1000), &LoadedWorker(w))
            .unwrap();
        // The forecast is queried while the scheduler lock is held: it never takes it.
        classifier.with_scheduler(|_| {
            let f = forecast.worker(w, Duration::from_secs(1)).unwrap();
            assert_eq!(f.decode_batch, 2);
            assert_eq!(f.decode_context_tokens, 512.0);
            assert!(f.prefill_backlog_s > 0.0);
        });
        clock.advance(Duration::from_millis(100));
        classifier
            .on_event(ClassifyEvent::Sent {
                request_id: "r1".into(),
                worker: w,
            })
            .await;
        clock.advance(Duration::from_millis(400));
        classifier
            .on_event(ClassifyEvent::Responding {
                request_id: "r1".into(),
                worker: w,
            })
            .await;
        clock.advance(Duration::from_secs(1));
        classifier.on_event(done("r1", 1300)).await;
        let session = forecast.session("s").unwrap();
        assert_eq!(session.context_tokens, 1300);
        assert_eq!(session.output_tokens, Some(300.0));
        let errors = forecast.errors()["standard"];
        assert_eq!(errors.ttft_s.count, 1);
        assert_eq!(forecast.dropped(), 0);

        let (off, _) = setup(Mode::Enforce, 10);
        assert!(off.forecast().is_none());
        assert_eq!(off.required_worker_inputs(), WorkerInputs::CACHE);
    }

    #[tokio::test]
    async fn throttle_rejects_in_enforce_mode() {
        let clock = Arc::new(ManualClock::new());
        let mut policy = ServicePolicy {
            mode: Mode::Enforce,
            ..Default::default()
        };
        policy.admission.max_lobby_sessions = 1;
        let mut classifier = SessionSchedulerClassifier::new(policy, clock, Arc::new(|| (1000, 1)));
        classifier
            .classify(request("a1", "a", 10), &NoWorkers)
            .unwrap();
        let error = classifier
            .classify(request("b1", "b", 10), &NoWorkers)
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "session scheduler throttled a new session: 1 sessions waiting, limit 1"
        );
    }

    struct FakeIndex(Vec<dynamo_kv_router::plugins::request_classifier::SessionResidency>);

    impl SessionIndexReader for FakeIndex {
        fn session_residency(
            &self,
            session_id: &str,
        ) -> Vec<dynamo_kv_router::plugins::request_classifier::SessionResidency> {
            if session_id == "s" {
                self.0.clone()
            } else {
                Vec::new()
            }
        }
        fn totals(&self) -> dynamo_kv_router::plugins::request_classifier::SessionIndexTotals {
            Default::default()
        }
    }

    fn row(
        worker: u64,
        tier: SessionTier,
        blocks: u64,
    ) -> dynamo_kv_router::plugins::request_classifier::SessionResidency {
        use dynamo_kv_router::plugins::request_classifier::{SessionHolder, SessionResidency};
        SessionResidency {
            holder: SessionHolder::Worker(WorkerWithDpRank::new(worker, 0)),
            tier,
            blocks,
        }
    }

    #[test]
    fn context_session_index_takes_the_deepest_holder_per_tier() {
        let index = ContextSessionIndex::new(Arc::new(FakeIndex(vec![
            row(1, SessionTier::Device, 40),
            row(2, SessionTier::Device, 12),
            row(1, SessionTier::Host, 30),
            row(2, SessionTier::Disk, 5),
            row(2, SessionTier::External, 9),
        ])));
        assert_eq!(index.resident_blocks("s"), Some((40, 30, 9)));
        assert_eq!(index.resident_blocks("other"), None);
    }

    #[test]
    fn the_classifier_reads_the_index_it_is_given_and_none_by_default() {
        let (classifier, _) = setup(Mode::Shadow, 1000);
        assert_eq!(classifier.session_index().resident_blocks("s"), None);
        let classifier = classifier.with_session_index(Arc::new(ContextSessionIndex::new(
            Arc::new(FakeIndex(vec![row(3, SessionTier::Device, 7)])),
        )));
        assert_eq!(
            classifier.session_index().resident_blocks("s"),
            Some((7, 0, 0))
        );
    }

    #[tokio::test]
    async fn with_retry_after_a_throttle_is_a_typed_rate_limit() {
        let clock = Arc::new(ManualClock::new());
        let mut policy = ServicePolicy {
            mode: Mode::Enforce,
            ..Default::default()
        };
        policy.admission.max_lobby_sessions = 1;
        policy.admission.retry_after_s = Some(30.0);
        let mut classifier = SessionSchedulerClassifier::new(policy, clock, Arc::new(|| (1000, 1)));
        classifier
            .classify(request("a1", "a", 10), &NoWorkers)
            .unwrap();
        let error = classifier
            .classify(request("b1", "b", 10), &NoWorkers)
            .unwrap_err();
        let typed = error
            .downcast_ref::<DynamoError>()
            .expect("a typed refusal");
        assert_eq!(typed.class(), ErrorClass::RateLimited);
        assert!(matches!(
            typed.public_details(),
            Some(PublicDetails::RateLimit {
                retry_after_s: Some(30),
                ..
            })
        ));
    }

    #[tokio::test(start_paused = true)]
    async fn tickets_a_throttle_issues_a_ticket_with_its_retry_after() {
        let clock = Arc::new(crate::clock::HostClock::new());
        let mut policy = ServicePolicy {
            mode: Mode::Enforce,
            ..Default::default()
        };
        policy.admission.max_lobby_sessions = 1;
        policy.admission.tickets = Some(crate::config::TicketsConfig::default());
        let mut classifier = SessionSchedulerClassifier::new(policy, clock, Arc::new(|| (1000, 1)));
        let first = classifier
            .classify(request("a1", "a", 10), &NoWorkers)
            .unwrap();
        // The lobby deadline carries the forecast (nothing known yet: the 600 s horizon), kept
        // for a's ticket should the deadline refuse it.
        assert_eq!(first.retry_after(), Some(Duration::from_secs(600)));
        let error = classifier
            .classify(request("b1", "b", 10), &NoWorkers)
            .unwrap_err();
        let typed = error
            .downcast_ref::<DynamoError>()
            .expect("a typed refusal");
        assert!(matches!(
            typed.public_details(),
            Some(PublicDetails::RateLimit {
                retry_after_s: Some(600),
                ..
            })
        ));
        classifier.with_scheduler(|s| {
            let b = s.tickets().get("b").expect("the throttle issued a ticket");
            assert_eq!((b.kv_need, b.class.as_str()), (10, "standard"));
            assert!(s.tickets().get("a").is_none());
        });
    }

    fn worker_row(id: u64, prefill: usize, running: usize, max_seqs: Option<u64>) -> WorkerRow {
        WorkerRow {
            worker: WorkerWithDpRank::new(id, 0),
            active_tokens: 0,
            kv_tokens: 0,
            prefill_tokens: prefill,
            capacity: WorkerCapacity::new(Some(1000), Some(100)).with_max_num_seqs(max_seqs),
            running,
            chunk: None,
        }
    }

    #[test]
    fn the_targeted_go_prefers_the_last_worker_then_the_least_loaded_with_room() {
        let w = |id| WorkerWithDpRank::new(id, 0);
        let rows = [
            worker_row(0, 980, 1, None),
            worker_row(1, 500, 1, None),
            worker_row(2, 100, 1, None),
        ];
        // The last worker has room: keep it.
        assert_eq!(go_target(&rows, Some(w(1)), 0.95), Some(w(1)));
        // The last worker is past the class threshold: the least prefill-loaded with room.
        assert_eq!(go_target(&rows, Some(w(0)), 0.95), Some(w(2)));
        assert_eq!(go_target(&rows, None, 0.95), Some(w(2)));
        // At its running-request limit a worker has no room.
        let capped = [worker_row(0, 0, 8, Some(8)), worker_row(1, 900, 2, Some(8))];
        assert_eq!(go_target(&capped, None, 0.95), Some(w(1)));
        // None has room: no target.
        let full = [worker_row(0, 990, 1, None)];
        assert_eq!(go_target(&full, None, 0.95), None);
    }

    /// What a host that honours keyed holds keeps (offline replay with
    /// `DYNAMO_REPLAY_GATE_RECHECK=keyed`): each wake takes the next sequence number.
    #[derive(Default)]
    struct Ledger {
        seq: u64,
        all: u64,
        fired: std::collections::HashMap<u64, u64>,
    }

    impl Ledger {
        fn woken_since(&self, keys: &[u64], at: u64) -> bool {
            self.all > at
                || keys
                    .iter()
                    .any(|k| self.fired.get(k).is_some_and(|&f| f > at))
        }
    }

    fn attach_keyed(gate: &dyn DispatchGate) -> Arc<Mutex<Ledger>> {
        use dynamo_kv_router::plugins::request_classifier::WakeScope;
        let ledger = Arc::new(Mutex::new(Ledger::default()));
        let keys = Arc::clone(&ledger);
        gate.attach(DispatchRecheck::with_keys(
            Arc::new(|| {}),
            Arc::new(move |scope| {
                let mut l = keys.lock();
                l.seq += 1;
                let seq = l.seq;
                match scope {
                    WakeScope::All => l.all = seq,
                    WakeScope::Keys(keys) => {
                        for k in keys {
                            l.fired.insert(*k, seq);
                        }
                    }
                }
            }),
        ));
        ledger
    }

    #[test]
    fn a_host_without_keys_gets_no_keyed_hold() {
        let (mut classifier, _clock) = setup(Mode::Enforce, 1000);
        let gate = classifier.dispatch_gate().unwrap();
        attach_counter(gate.as_ref());
        let a = SessionContext::new("a".into(), None, None, None);
        let b = SessionContext::new("b".into(), None, None, None);
        classifier
            .classify(request("a1", "a", 2000), &NoWorkers)
            .unwrap();
        classifier
            .classify(request("b1", "b", 10), &NoWorkers)
            .unwrap();
        // b waits behind a, but the host re-asks every hold anyway.
        let (decision, wake) = gate.check_wake(&head("b1", &b, 10));
        assert!(matches!(decision, DispatchDecision::Hold { .. }));
        assert_eq!(wake, HoldWake::Any);
        assert_eq!(gate.check_wake(&head("a1", &a, 2000)).1, HoldWake::Any);
    }

    #[tokio::test]
    async fn a_session_behind_another_is_keyed_on_both_and_woken_when_the_first_leaves() {
        let (mut classifier, _clock) = setup(Mode::Enforce, 1000);
        let gate = classifier.dispatch_gate().unwrap();
        let ledger = attach_keyed(gate.as_ref());
        let a = SessionContext::new("a".into(), None, None, None);
        let b = SessionContext::new("b".into(), None, None, None);
        classifier
            .classify(request("a1", "a", 2000), &NoWorkers)
            .unwrap();
        classifier
            .classify(request("b1", "b", 10), &NoWorkers)
            .unwrap();
        // a does not fit (a budget, asked every time); b waits behind a.
        let (decision, wake) = gate.check_wake(&head("a1", &a, 2000));
        assert!(matches!(decision, DispatchDecision::Hold { .. }));
        assert_eq!(wake, HoldWake::Any);
        let at = ledger.lock().seq;
        let (decision, wake) = gate.check_wake(&head("b1", &b, 10));
        assert!(matches!(decision, DispatchDecision::Hold { .. }));
        let keys = vec![session_wake_key("a"), session_wake_key("b")];
        assert_eq!(wake, HoldWake::Keys(keys.clone()));
        assert!(!ledger.lock().woken_since(&keys, at));
        // a leaves the lobby (its request expires): b's keys wake, and b now fits.
        classifier
            .on_event(ClassifyEvent::Aborted {
                request_id: "a1".into(),
                worker: None,
                error: None,
            })
            .await;
        assert!(ledger.lock().woken_since(&keys, at));
        assert_eq!(
            gate.check_wake(&head("b1", &b, 10)),
            (DispatchDecision::Go { target: None }, HoldWake::Any)
        );
    }

    /// The wake contract on random runs: whenever no key of a keyed hold woke since it was
    /// answered, asking again gives the same answer. First come, fair and tenant order, with a
    /// small session table (evictions), sessions arriving, dispatching, completing, expiring.
    #[tokio::test]
    async fn keyed_holds_answer_as_a_fresh_check_until_a_key_wakes() {
        use crate::config::LobbyOrder;
        let mut verified = std::collections::HashMap::new();
        for order in [LobbyOrder::Fifo, LobbyOrder::Fair, LobbyOrder::Tenant] {
            for seed in 0..16u64 {
                *verified.entry(format!("{order:?}")).or_insert(0) +=
                    keyed_contract_run(order, seed).await;
            }
        }
        // Each order produced keyed holds that were checked against a fresh answer.
        assert!(verified.values().all(|&n| n > 50), "{verified:?}");
    }

    async fn keyed_contract_run(order: crate::config::LobbyOrder, seed: u64) -> u64 {
        struct Queued {
            id: String,
            ctx: SessionContext,
            tokens: usize,
            keyed: Option<(DispatchDecision, Vec<u64>, u64)>,
        }
        fn head_of(q: &Queued) -> DispatchHead<'_> {
            head(&q.id, &q.ctx, q.tokens)
        }
        let clock = Arc::new(ManualClock::new());
        let mut policy = ServicePolicy {
            mode: Mode::Enforce,
            session_table_capacity: 12,
            ..Default::default()
        };
        policy.admission.lobby_order = order;
        for class in &mut policy.classes {
            // Wait limits far away: a deadline ends a keyed hold anyway (the host's timer).
            class.admission.max_lobby_wait_s = 1e7;
        }
        let mut classifier =
            SessionSchedulerClassifier::new(policy, clock.clone(), Arc::new(|| (4000, 1)));
        let gate = classifier.dispatch_gate().unwrap();
        let ledger = attach_keyed(gate.as_ref());
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut queued: Vec<Queued> = Vec::new();
        let mut in_flight: Vec<(String, String, usize)> = Vec::new();
        let mut verified = 0;
        for step in 0..400 {
            match rng.u32(0..10) {
                0..=3 => {
                    let session = format!("s{}", rng.u32(0..18));
                    let busy = queued.iter().any(|q| q.ctx.session_id() == session)
                        || in_flight.iter().any(|f| f.1 == session);
                    if !busy {
                        let tenant = format!("t{}", rng.u32(0..3));
                        let headers: std::collections::BTreeMap<String, Vec<String>> =
                            [(HEADER_TENANT.to_owned(), vec![tenant])].into();
                        let ctx = SessionContext::new(session, None, None, None)
                            .with_agent_headers(Arc::new(headers));
                        let id = format!("r{step}");
                        let tokens = rng.usize(50..1500);
                        let mut req =
                            ClassifyRequest::for_test(tokens, Some(&id), Some(ctx.clone()));
                        req.set_policy_class(["priority", "standard", "batch"][rng.usize(0..3)]);
                        if classifier.classify(req, &NoWorkers).is_ok() {
                            queued.push(Queued {
                                id,
                                ctx,
                                tokens,
                                keyed: None,
                            });
                        }
                    }
                }
                4..=5 => {
                    let go = (0..queued.len()).find(|&i| {
                        matches!(
                            gate.check(&head_of(&queued[i])),
                            DispatchDecision::Go { .. }
                        )
                    });
                    if let Some(i) = go {
                        let q = queued.remove(i);
                        gate.dispatched(&head_of(&q), None);
                        in_flight.push((q.id, q.ctx.session_id().to_owned(), q.tokens));
                    }
                }
                6..=7 if !in_flight.is_empty() => {
                    let (id, _, tokens) = in_flight.remove(rng.usize(0..in_flight.len()));
                    classifier.on_event(done(&id, tokens + 100)).await;
                }
                8 if !queued.is_empty() => {
                    let q = queued.remove(rng.usize(0..queued.len()));
                    classifier
                        .on_event(ClassifyEvent::Aborted {
                            request_id: q.id,
                            worker: None,
                            error: None,
                        })
                        .await;
                }
                _ => clock.advance(Duration::from_millis(rng.u64(1..5000))),
            }
            // The host's recheck: ask every queued head; an unwoken keyed hold must not change.
            for q in &mut queued {
                let at = ledger.lock().seq;
                let (decision, wake) = gate.check_wake(&head_of(q));
                if let Some((held, keys, held_at)) = &q.keyed
                    && !ledger.lock().woken_since(keys, *held_at)
                {
                    assert_eq!(
                        decision, *held,
                        "{order:?} seed {seed} step {step}: {} changed without a wake",
                        q.id
                    );
                    verified += 1;
                }
                q.keyed = match wake {
                    HoldWake::Keys(keys) => Some((decision, keys, at)),
                    HoldWake::Any => None,
                };
            }
        }
        verified
    }

    /// With the adaptive rule's gate off, nothing holds a lobby session, so `classify` sets no
    /// lobby deadline (it used to, and the router expired sessions the gate never held). A session
    /// classified then has no deadline to expire it: when the gate comes back on, it is admitted at
    /// its wait limit instead of being held for good.
    #[test]
    fn with_the_adaptive_gate_off_no_lobby_deadline_and_no_hold_past_the_limit() {
        let base = Instant::now();
        let offset = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let o = Arc::clone(&offset);
        let clock = Arc::new(ContextClock::new(move || {
            base + Duration::from_millis(o.load(Ordering::Relaxed))
        }));
        let mut policy = ServicePolicy {
            mode: Mode::Enforce,
            adaptive: Some(crate::adaptive::AdaptiveConfig {
                start_on: false,
                ..Default::default()
            }),
            ..Default::default()
        };
        policy.classes[1].admission.max_lobby_wait_s = 60.0;
        let mut classifier = SessionSchedulerClassifier::new(policy, clock, Arc::new(|| (1000, 1)));
        let gate = classifier.dispatch_gate().unwrap();
        let a = SessionContext::new("a".into(), None, None, None);
        let b = SessionContext::new("b".into(), None, None, None);
        let req_a = classifier
            .classify(request("a1", "a", 2000), &NoWorkers)
            .unwrap();
        let req_b = classifier
            .classify(request("b1", "b", 10), &NoWorkers)
            .unwrap();
        // Gate off: no deadline, and the gate lets both go.
        assert_eq!((req_a.due_at(), req_b.due_at()), (None, None));
        assert_eq!(
            gate.check(&head("b1", &b, 10)),
            DispatchDecision::Go { target: None }
        );
        // The gate comes back on: b waits behind a, then past the wait limit it is admitted.
        classifier.with_scheduler(|s| s.set_adaptive_gate(true));
        assert!(matches!(
            gate.check(&head("b1", &b, 10)),
            DispatchDecision::Hold { .. }
        ));
        offset.store(61_000, Ordering::Relaxed);
        assert_eq!(
            gate.check(&head("a1", &a, 2000)),
            DispatchDecision::Go { target: None }
        );
        // A session classified with the gate on gets its deadline as before.
        let req_c = classifier
            .classify(request("c1", "c", 10), &NoWorkers)
            .unwrap();
        assert!(req_c.due_at().is_some());
    }
}
