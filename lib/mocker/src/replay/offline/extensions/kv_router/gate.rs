// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A request classifier and its dispatch gate inside offline replay (the "replay gate").
//!
//! The live router runs a classifier under its own clock and delivers lifecycle events from a
//! router task. Replay runs the same classifier on the replay thread: its context reads the
//! replay's virtual clock, classification and events happen in replay order, and a gate's
//! `Hold { until }` becomes a replay wake-up (`PlacementPolicy::next_wakeup_ms`). So a session
//! scheduler runs at simulated speed and gives the same answer for the same input.
//!
//! Requests carry what a production frontend would put on them: the session end marker
//! (`nvext.agent_context.session_final`, header `x-dynamo-session-final`), the parent session
//! of a subagent, the agent headers a plugin reads (tenant, client signals), and a policy class.
//! Replay rows cannot carry these yet, so they come from a side file keyed by the authored
//! request id ([`ReplayRequestFields`]), which the trace exporter writes from a seed.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use dynamo_admission::{
    ClassifyEvent, ClassifyRequest, DispatchGate, DispatchRecheck, RequestClassifier,
    RequestClassifierContext, SessionIndexReader, SessionLane, WakeScope, WorkerCacheView,
    WorkerLoadView, WorkerView,
};
use dynamo_kv_router::plugins::request_classifier::RequestClassifierFactory;
use dynamo_router_protocol::columns::{WorkerInfo, WorkerInputs};
use dynamo_router_protocol::session::SessionContext;
use futures::executor::block_on;
use rustc_hash::FxHashMap;
use serde::Deserialize;
use tokio::time::Instant;

/// What a production frontend attaches to one request, for a replayed request.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReplayRequestFields {
    /// The authored request id (`agentic.request_id` of the replay row).
    pub request_id: String,
    /// Overrides the replay session id (a subagent's own session), when set.
    #[serde(default)]
    pub session_id: Option<String>,
    /// The parent session of a subagent request.
    #[serde(default)]
    pub parent_session_id: Option<String>,
    /// `nvext.agent_context.session_final`: `true` on the last request of a session.
    #[serde(default)]
    pub session_final: Option<bool>,
    /// The policy class the frontend resolved (`x-dynamo-meta-policy-class`).
    #[serde(default)]
    pub policy_class: Option<String>,
    /// Agent headers as the HTTP ingress captures them (lowercase names): tenant
    /// (`x-dynamo-tenant`) and client signals (`x-dynamo-hint-*`).
    #[serde(default)]
    pub agent_headers: BTreeMap<String, Vec<String>>,
    /// The KV stream this request continues: the main conversation, a side
    /// chain (which shares the main session but not its KV) or a subagent.
    /// `kv_actions.invalidate` compares a request's blocks with the previous
    /// request of its stream.
    #[serde(default)]
    pub stream_id: Option<String>,
    /// The harness's compaction signal: this request's context was compacted,
    /// so blocks of the stream's previous request it no longer holds are dead.
    #[serde(default)]
    pub compaction: Option<bool>,
}

/// Request fields by authored request id.
pub type ReplayRequestAnnotations = HashMap<String, ReplayRequestFields>;

/// Reads a JSON Lines file of [`ReplayRequestFields`], one request per line.
pub fn load_request_annotations(path: &Path) -> Result<ReplayRequestAnnotations> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading request fields {}", path.display()))?;
    let mut fields = HashMap::new();
    for (number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: ReplayRequestFields = serde_json::from_str(line)
            .with_context(|| format!("{}:{}", path.display(), number + 1))?;
        fields.insert(row.request_id.clone(), row);
    }
    Ok(fields)
}

/// The fields of request `id`: its own row, or, for a copy in a closed-loop users run
/// (`cp<key>:<template request id>`, key = user then position, six digits each), the template
/// request's row with its session ids prefixed `cp<key>:` and the class and agent headers of the
/// user's row (`cp<user>`). A users run's fields file holds one row per template request and one
/// per user instead of one per copy.
pub fn resolve_request_fields<'a>(
    fields: &'a ReplayRequestAnnotations,
    id: &str,
) -> Option<Cow<'a, ReplayRequestFields>> {
    if let Some(row) = fields.get(id) {
        return Some(Cow::Borrowed(row));
    }
    let (key, template) = id.strip_prefix("cp")?.split_once(':')?;
    if key.len() != 12 || !key.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let base = fields.get(template)?;
    let user = fields.get(&format!("cp{}", &key[..6]))?;
    let prefixed = |session: &String| format!("cp{key}:{session}");
    let mut row = base.clone();
    row.request_id = id.to_owned();
    row.session_id = base.session_id.as_ref().map(prefixed);
    row.parent_session_id = base.parent_session_id.as_ref().map(prefixed);
    // A copy's KV is its own (its blocks are its own range), so its KV stream is too: without the
    // prefix every user replaying the same play shared one stream, and `kv_actions.invalidate`
    // and a KVCR move compared one user's request with another user's.
    row.stream_id = base.stream_id.as_ref().map(prefixed);
    if user.policy_class.is_some() {
        row.policy_class = user.policy_class.clone();
    }
    for (name, values) in &user.agent_headers {
        row.agent_headers.insert(name.clone(), values.clone());
    }
    // The copy's own row (`cp<key>`, 12 digits), when the export writes one: per-session hints
    // such as `x-dynamo-hint-session-kind` (F35), over the user's.
    if let Some(copy) = fields.get(&format!("cp{key}")) {
        for (name, values) in &copy.agent_headers {
            row.agent_headers.insert(name.clone(), values.clone());
        }
    }
    Some(Cow::Owned(row))
}

impl ReplayRequestFields {
    /// The session context a frontend would build from these fields.
    pub fn session_context(&self, replay_session: Option<&str>) -> Option<SessionContext> {
        let session = self.session_id.as_deref().or(replay_session)?;
        let context = SessionContext::new(
            session.to_owned(),
            self.parent_session_id.clone(),
            self.session_final,
            None,
        );
        Some(if self.agent_headers.is_empty() {
            context
        } else {
            context.with_agent_headers(Arc::new(self.agent_headers.clone()))
        })
    }
}

/// The replay's virtual clock as an `Instant`, readable from any closure.
#[derive(Clone)]
pub(crate) struct VirtualClock {
    epoch: Instant,
    nanos: Arc<AtomicU64>,
}

impl VirtualClock {
    pub(crate) fn new(epoch: Instant) -> Self {
        Self {
            epoch,
            nanos: Arc::new(AtomicU64::new(0)),
        }
    }

    pub(crate) fn set_ms(&self, now_ms: f64) {
        let nanos = (now_ms.max(0.0) * 1e6) as u64;
        // Replay time never goes back; a late callback keeps the later time.
        self.nanos.fetch_max(nanos, Ordering::Relaxed);
    }

    pub(crate) fn now(&self) -> Instant {
        self.epoch + Duration::from_nanos(self.nanos.load(Ordering::Relaxed))
    }

    /// Replay milliseconds of a virtual instant.
    pub(crate) fn ms_of(&self, instant: Instant) -> f64 {
        instant.saturating_duration_since(self.epoch).as_secs_f64() * 1000.0
    }
}

/// Counts of what the classifier and gate did, for reports and tests.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReplayGateStats {
    pub classified: u64,
    /// `classify` returned an error: the client is told to retry (429 with Retry-After).
    pub rejected: u64,
    /// A queued request passed its `due_at` (504) and the client retried.
    pub expired: u64,
    /// Gate answers of `Hold`.
    pub holds: u64,
    pub retries: u64,
    pub session_final_marks: u64,
    /// Distinct requests that got at least one `Hold` (a request held across many drains counts once).
    pub held_requests: u64,
    /// Classified requests whose session the session prefix index held somewhere.
    pub index_resident: u64,
    /// Drains run with the gate installed.
    pub drains: u64,
    /// Gate checks (`DispatchGate::check` calls) over the run, and the most in one drain.
    pub gate_checks: u64,
    pub max_checks_per_drain: u64,
    /// Drains that rechecked every held lane (all of them unless `DYNAMO_REPLAY_GATE_RECHECK`
    /// is `notify`).
    pub full_rechecks: u64,
    /// With `DYNAMO_REPLAY_GATE_RECHECK=keyed`: heads parked (the gate held them with keys), and
    /// lane visits of the full recheck that parking skipped (each a `Hold` the full recheck would
    /// have asked for), counted as parked heads per drain.
    pub parks: u64,
    pub keyed_answers: u64,
}

/// How offline replay asks the dispatch gate about held heads again: [`ReplayScope::gate_recheck`]
/// (crate::replay::ReplayScope), else `DYNAMO_REPLAY_GATE_RECHECK` (`notify`, `keyed`), else
/// `Full`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReplayGateRecheck {
    /// Every held head at every drain (the default).
    #[default]
    Full,
    /// `notify`: held heads only when the gate notified, a hold deadline came due or a request
    /// expired.
    Notify,
    /// `keyed`: as `Full`, but a head the gate held with [`HoldWake::Keys`] is parked until one
    /// of its keys is woken or its deadline comes. The same answers as `Full` when the gate keeps
    /// its wake contract; off by default.
    Keyed,
    /// `keyed-verify`: `Keyed`, and every drain also asks the gate about each parked head and logs
    /// (warn, `keyed_hold_stale`) any that would no longer hold: a broken wake contract. Asking
    /// perturbs the gate's upkeep, so the run itself is not the keyed run; for diagnosis only.
    KeyedVerify,
}

fn recheck_mode() -> ReplayGateRecheck {
    if let Some(mode) = crate::replay::replay_scope().gate_recheck {
        return mode;
    }
    match std::env::var("DYNAMO_REPLAY_GATE_RECHECK").as_deref() {
        Ok("notify") => ReplayGateRecheck::Notify,
        Ok("keyed") => ReplayGateRecheck::Keyed,
        Ok("keyed-verify") => ReplayGateRecheck::KeyedVerify,
        _ => ReplayGateRecheck::Full,
    }
}

/// The keys the gate woke ([`DispatchRecheck::wake_keys`]) since the last drain started.
#[derive(Debug, Default)]
pub(crate) struct WakeLedger {
    all: bool,
    fired: Vec<u64>,
}

impl WakeLedger {
    fn wake(&mut self, scope: WakeScope<'_>) {
        match scope {
            WakeScope::All => self.all = true,
            WakeScope::Keys(keys) => self.fired.extend_from_slice(keys),
        }
    }
}

/// A parked head: the gate held it with [`HoldWake::Keys`] and the queue no longer asks it.
#[derive(Debug)]
struct ParkedHold {
    class_index: usize,
    lane: SessionLane,
    until_ms: Option<f64>,
    keys: Vec<u64>,
}

/// Parked heads of `DYNAMO_REPLAY_GATE_RECHECK=keyed`. The queue skips a parked lane at every
/// recheck; each drain first unparks the lanes the full recheck could answer differently: those
/// whose key was woken, whose deadline came, all of them on [`DispatchRecheck::wake_all`], and one
/// more (the first parked request), so the gate is asked at least once per drain and its upkeep
/// (pacing, rotation) runs at the same drains as with the full recheck.
#[derive(Debug, Default)]
pub(crate) struct KeyedHolds {
    ledger: Arc<parking_lot::Mutex<WakeLedger>>,
    parked: BTreeMap<uuid::Uuid, ParkedHold>,
    by_key: FxHashMap<u64, Vec<uuid::Uuid>>,
    /// (deadline in ms as bits, request): non-negative floats order as their bits.
    by_until: std::collections::BTreeSet<(u64, uuid::Uuid)>,
}

impl KeyedHolds {
    /// Park `uuid`'s head, held until one of `keys` wakes or `until_ms`.
    pub(crate) fn park(
        &mut self,
        uuid: uuid::Uuid,
        class_index: usize,
        lane: SessionLane,
        until_ms: Option<f64>,
        keys: Vec<u64>,
    ) {
        self.forget(&uuid);
        for key in &keys {
            self.by_key.entry(*key).or_default().push(uuid);
        }
        if let Some(until) = until_ms {
            self.by_until.insert((until.max(0.0).to_bits(), uuid));
        }
        self.parked.insert(
            uuid,
            ParkedHold {
                class_index,
                lane,
                until_ms,
                keys,
            },
        );
    }

    /// `uuid` is no longer parked (unparked, or it left the queue).
    pub(crate) fn forget(&mut self, uuid: &uuid::Uuid) -> Option<(usize, SessionLane)> {
        let hold = self.parked.remove(uuid)?;
        if let Some(until) = hold.until_ms {
            self.by_until.remove(&(until.max(0.0).to_bits(), *uuid));
        }
        Some((hold.class_index, hold.lane))
    }

    /// The lanes to unpark as a drain at `now_ms` starts (see the type's description).
    pub(crate) fn due(&mut self, now_ms: f64) -> Vec<(usize, SessionLane)> {
        let (all, fired) = {
            let mut ledger = self.ledger.lock();
            (
                std::mem::take(&mut ledger.all),
                std::mem::take(&mut ledger.fired),
            )
        };
        let mut out = Vec::new();
        if all {
            self.by_key.clear();
            let parked: Vec<uuid::Uuid> = self.parked.keys().copied().collect();
            out.extend(parked.iter().filter_map(|uuid| self.forget(uuid)));
            return out;
        }
        for key in fired {
            for uuid in self.by_key.remove(&key).unwrap_or_default() {
                if self
                    .parked
                    .get(&uuid)
                    .is_some_and(|h| h.keys.contains(&key))
                {
                    out.extend(self.forget(&uuid));
                }
            }
        }
        while let Some(&(bits, uuid)) = self.by_until.first()
            && f64::from_bits(bits) <= now_ms
        {
            self.by_until.pop_first();
            out.extend(self.forget(&uuid));
        }
        if let Some(&uuid) = self.parked.keys().next() {
            out.extend(self.forget(&uuid));
        }
        out
    }

    /// Every parked head: (request, class, lane, keys).
    pub(crate) fn parked(&self) -> impl Iterator<Item = (&uuid::Uuid, usize, SessionLane, &[u64])> {
        self.parked
            .iter()
            .map(|(uuid, h)| (uuid, h.class_index, h.lane, h.keys.as_slice()))
    }

    /// The earliest deadline of a parked head.
    pub(crate) fn next_until_ms(&self) -> Option<f64> {
        self.by_until.first().map(|(bits, _)| f64::from_bits(*bits))
    }

    pub(crate) fn parked_count(&self) -> usize {
        self.parked.len()
    }
}

/// `DYNAMO_REPLAY_GATE_LOG=<path>`: write one JSON line per gate decision (first `hold`, its
/// `release` at dispatch, `expire`, `refuse`, client `retry`), with the replay time in seconds,
/// the request's replay uuid (the per-request capture's `uuid`), its policy class and, on
/// `release` and `expire`, how long it was held. Off by default; the counts in
/// `replay_gate_stats` do not depend on it.
fn open_decision_log() -> Option<std::io::BufWriter<std::fs::File>> {
    let path = std::env::var_os("DYNAMO_REPLAY_GATE_LOG")?;
    match std::fs::File::create(&path) {
        Ok(file) => Some(std::io::BufWriter::new(file)),
        Err(error) => {
            tracing::warn!(path = ?path, %error, "cannot open the replay gate decision log");
            None
        }
    }
}

/// One classifier installed in an offline replay router.
pub(crate) struct ReplayClassifier {
    classifier: Box<dyn RequestClassifier>,
    pub(crate) gate: Option<Arc<dyn DispatchGate>>,
    recheck: Arc<AtomicBool>,
    pub(crate) clock: VirtualClock,
    pub(crate) worker_inputs: WorkerInputs,
    pub(crate) stats: ReplayGateStats,
    /// Replay time (ms) of each held request's first `Hold`, until it leaves the queue.
    pub(crate) first_hold_ms: HashMap<uuid::Uuid, f64>,
    /// How long each held request waited from its first `Hold` until it was dispatched or expired.
    pub(crate) hold_durations_ms: Vec<f64>,
    /// The replay's session prefix index, read-only, when `enable_session_prefix_index` is on.
    session_index: Option<Arc<dyn SessionIndexReader>>,
    /// Per-decision log (`DYNAMO_REPLAY_GATE_LOG`), off by default.
    decision_log: Option<std::io::BufWriter<std::fs::File>>,
    /// With `DYNAMO_REPLAY_GATE_RECHECK=keyed`, the gate's keyed holds.
    pub(crate) keyed: Option<KeyedHolds>,
    /// How held heads are asked again.
    pub(crate) recheck_mode: ReplayGateRecheck,
}

impl ReplayClassifier {
    /// A request got a `Hold` at `now_ms`; only its first one starts the clock.
    pub(crate) fn note_hold(&mut self, uuid: uuid::Uuid, class: &str, now_ms: f64) {
        if let std::collections::hash_map::Entry::Vacant(e) = self.first_hold_ms.entry(uuid) {
            e.insert(now_ms);
            self.stats.held_requests += 1;
            self.log_decision(now_ms, "hold", &uuid, Some(class), None);
        }
    }

    /// A request left the queue at `now_ms`: dispatched (`release`) or past its `due_at`
    /// (`expire`). Only a request that was held is logged as released.
    pub(crate) fn note_left(
        &mut self,
        uuid: &uuid::Uuid,
        class: Option<&str>,
        now_ms: f64,
        expired: bool,
    ) {
        if let Some(keyed) = self.keyed.as_mut() {
            keyed.forget(uuid);
        }
        let held_ms = self
            .first_hold_ms
            .remove(uuid)
            .map(|t0| (now_ms - t0).max(0.0));
        if let Some(held) = held_ms {
            self.hold_durations_ms.push(held);
        }
        if expired {
            self.log_decision(now_ms, "expire", uuid, class, held_ms);
        } else if held_ms.is_some() {
            self.log_decision(now_ms, "release", uuid, class, held_ms);
        }
    }

    /// One line of the decision log, when `DYNAMO_REPLAY_GATE_LOG` is set.
    pub(crate) fn log_decision(
        &mut self,
        now_ms: f64,
        kind: &str,
        uuid: &uuid::Uuid,
        class: Option<&str>,
        held_ms: Option<f64>,
    ) {
        let Some(out) = self.decision_log.as_mut() else {
            return;
        };
        let line = serde_json::json!({
            "t": now_ms / 1000.0,
            "kind": kind,
            "uuid": uuid.to_string(),
            "class": class,
            "held_s": held_ms.map(|ms| ms / 1000.0),
        });
        let _ = writeln!(out, "{line}");
    }

    fn hold_percentile_ms(&self, q: f64) -> f64 {
        let mut v = self.hold_durations_ms.clone();
        if v.is_empty() {
            return 0.0;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let i = ((v.len() - 1) as f64 * q).round() as usize;
        v[i.min(v.len() - 1)]
    }
}

/// Logs what the classifier and gate did when the replay ends (the counts are not returned to
/// Python): grep the run's log for `replay_gate_stats`.
impl Drop for ReplayClassifier {
    fn drop(&mut self) {
        let totals = self
            .session_index
            .as_ref()
            .map(|index| index.totals())
            .unwrap_or_default();
        tracing::info!(
            classified = self.stats.classified,
            rejected = self.stats.rejected,
            expired = self.stats.expired,
            holds = self.stats.holds,
            retries = self.stats.retries,
            session_final_marks = self.stats.session_final_marks,
            held_requests = self.stats.held_requests,
            drains = self.stats.drains,
            gate_checks = self.stats.gate_checks,
            max_checks_per_drain = self.stats.max_checks_per_drain,
            full_rechecks = self.stats.full_rechecks,
            keyed_answers = self.stats.keyed_answers,
            parks = self.stats.parks,
            recheck_mode = ?self.recheck_mode,
            hold_released = self.hold_durations_ms.len(),
            hold_still_queued = self.first_hold_ms.len(),
            hold_total_s = self.hold_durations_ms.iter().sum::<f64>() / 1000.0,
            hold_p50_s = self.hold_percentile_ms(0.5) / 1000.0,
            hold_p90_s = self.hold_percentile_ms(0.9) / 1000.0,
            hold_p99_s = self.hold_percentile_ms(0.99) / 1000.0,
            hold_max_s = self.hold_percentile_ms(1.0) / 1000.0,
            session_index = self.session_index.is_some(),
            index_resident = self.stats.index_resident,
            index_sessions = totals.sessions,
            index_device_blocks = totals.device_blocks,
            index_host_blocks = totals.host_blocks,
            index_disk_blocks = totals.disk_blocks,
            index_external_blocks = totals.external_blocks,
            "replay_gate_stats"
        );
    }
}

impl ReplayClassifier {
    pub(crate) fn new(
        factory: &RequestClassifierFactory,
        block_size: u32,
        workers: impl Fn() -> Vec<WorkerInfo> + Send + Sync + 'static,
        clock: VirtualClock,
        session_index: Option<Arc<dyn SessionIndexReader>>,
    ) -> Self {
        let reader = clock.clone();
        let mut context =
            RequestClassifierContext::new(block_size, workers).with_clock(move || reader.now());
        if let Some(index) = &session_index {
            context = context.with_session_index(Arc::clone(index));
        }
        let classifier = factory(context);
        let worker_inputs = classifier.required_worker_inputs();
        let gate = classifier.dispatch_gate();
        let recheck = Arc::new(AtomicBool::new(false));
        let recheck_mode = recheck_mode();
        let mut keyed = None;
        if let Some(gate) = &gate {
            let flag = Arc::clone(&recheck);
            let wake: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
                flag.store(true, Ordering::Relaxed);
            });
            if matches!(
                recheck_mode,
                ReplayGateRecheck::Keyed | ReplayGateRecheck::KeyedVerify
            ) {
                let holds = KeyedHolds::default();
                let ledger = Arc::clone(&holds.ledger);
                gate.attach(DispatchRecheck::with_keys(
                    wake,
                    Arc::new(move |scope| ledger.lock().wake(scope)),
                ));
                keyed = Some(holds);
            } else {
                gate.attach(DispatchRecheck::new(wake));
            }
        }
        Self {
            classifier,
            gate,
            recheck,
            clock,
            worker_inputs,
            stats: ReplayGateStats::default(),
            first_hold_ms: HashMap::new(),
            hold_durations_ms: Vec::new(),
            session_index,
            decision_log: open_decision_log(),
            keyed,
            recheck_mode,
        }
    }

    pub(crate) fn classify(
        &mut self,
        request: ClassifyRequest,
        workers: &dyn WorkerView,
    ) -> std::result::Result<ClassifyRequest, (String, Option<std::time::Duration>)> {
        self.stats.classified += 1;
        if let (Some(index), Some(session)) = (&self.session_index, request.session_context())
            && !index.session_residency(session.session_id()).is_empty()
        {
            self.stats.index_resident += 1;
        }
        self.classifier
            .classify(request, workers)
            .map_err(|error| (error.to_string(), retry_after_hint(error.as_ref())))
    }

    /// Delivers one lifecycle event in replay order. The session scheduler's `on_event` never
    /// awaits, so this completes at once and replay stays single-threaded and deterministic.
    pub(crate) fn deliver(&mut self, event: ClassifyEvent) {
        block_on(self.classifier.on_event(event));
    }

    /// Whether the gate asked for its held heads to be rechecked since the last call.
    pub(crate) fn take_recheck(&self) -> bool {
        self.recheck.swap(false, Ordering::Relaxed)
    }
}

/// The worker table a replay classifier reads: one row per rank, with the groups it declared.
pub(crate) struct ReplayWorkerView {
    pub(crate) rows: Vec<(WorkerInfo, Option<WorkerLoadView>, Option<WorkerCacheView>)>,
}

impl WorkerView for ReplayWorkerView {
    fn len(&self) -> usize {
        self.rows.len()
    }

    fn info(&self, row: usize) -> Option<WorkerInfo> {
        self.rows.get(row).map(|r| r.0)
    }

    fn load(&self, row: usize) -> Option<WorkerLoadView> {
        self.rows.get(row).and_then(|r| r.1)
    }

    fn cache(&self, row: usize) -> Option<WorkerCacheView> {
        self.rows.get(row).and_then(|r| r.2)
    }
}

/// A refused request waiting to come back, as a client does after a 429 or a 504.
pub(crate) struct RetryEntry<T> {
    pub(crate) at_ms: f64,
    pub(crate) request: T,
}

/// Client backoff after a refusal: 2 s doubling to 30 s.
pub(crate) fn retry_delay_ms(attempt: u32) -> f64 {
    (2000.0 * 2f64.powi(attempt.min(4) as i32)).min(30_000.0)
}

/// The Retry-After a classifier's refusal carries: a typed rate-limit error (HTTP 429) with
/// `retry_after_s` in its public details, found anywhere in the error's source chain.
pub(crate) fn retry_after_hint(
    error: &(dyn std::error::Error + 'static),
) -> Option<std::time::Duration> {
    let mut current = Some(error);
    while let Some(e) = current {
        if let Some(typed) = e.downcast_ref::<dynamo_error::DynamoError>()
            && let Some(dynamo_error::PublicDetails::RateLimit {
                retry_after_s: Some(seconds),
                ..
            }) = typed.public_details()
        {
            return Some(std::time::Duration::from_secs(*seconds));
        }
        current = e.source();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(json: serde_json::Value) -> ReplayRequestFields {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn a_users_copy_takes_its_template_row_and_its_users_class() {
        let class = "x-dynamo-meta-policy-class";
        let fields: ReplayRequestAnnotations = [
            row(serde_json::json!({"request_id": "t:r1", "session_id": "s:a", "parent_session_id": "s:m",
                "session_final": true, "policy_class": "batch", "agent_headers": {"x-dynamo-meta-policy-class": ["batch"], "x-dynamo-tenant": ["t1"]}})),
            row(serde_json::json!({"request_id": "cp000007", "policy_class": "priority", "agent_headers": {"x-dynamo-meta-policy-class": ["priority"]}})),
        ]
        .into_iter()
        .map(|r| (r.request_id.clone(), r))
        .collect();
        let copy = resolve_request_fields(&fields, "cp000007000003:t:r1").unwrap();
        assert_eq!(copy.request_id, "cp000007000003:t:r1");
        assert_eq!(copy.session_id.as_deref(), Some("cp000007000003:s:a"));
        assert_eq!(
            copy.parent_session_id.as_deref(),
            Some("cp000007000003:s:m")
        );
        assert_eq!(copy.session_final, Some(true));
        assert_eq!(copy.policy_class.as_deref(), Some("priority"));
        assert_eq!(copy.agent_headers[class], vec!["priority".to_string()]);
        assert_eq!(
            copy.agent_headers["x-dynamo-tenant"],
            vec!["t1".to_string()]
        );
        // A copy row adds its own hints over the user's.
        let mut with_copy = fields.clone();
        let copy_row = row(serde_json::json!({"request_id": "cp000007000003",
            "agent_headers": {"x-dynamo-hint-session-kind": ["agent"]}}));
        with_copy.insert(copy_row.request_id.clone(), copy_row);
        let hinted = resolve_request_fields(&with_copy, "cp000007000003:t:r1").unwrap();
        assert_eq!(
            hinted.agent_headers["x-dynamo-hint-session-kind"],
            vec!["agent".to_string()]
        );
        assert_eq!(hinted.agent_headers[class], vec!["priority".to_string()]);
        assert!(
            !copy
                .agent_headers
                .contains_key("x-dynamo-hint-session-kind")
        );
        // Own rows win; unknown users, templates and malformed keys have no fields.
        assert!(matches!(
            resolve_request_fields(&fields, "t:r1"),
            Some(Cow::Borrowed(_))
        ));
        assert!(resolve_request_fields(&fields, "cp000008000003:t:r1").is_none());
        assert!(resolve_request_fields(&fields, "cp000007000003:t:r2").is_none());
        assert!(resolve_request_fields(&fields, "cp7:t:r1").is_none());
    }

    #[test]
    fn a_users_copy_has_its_own_kv_stream() {
        let fields: ReplayRequestAnnotations = [
            row(
                serde_json::json!({"request_id": "t:r1", "session_id": "p:m", "stream_id": "p:a3",
                "compaction": true}),
            ),
            row(serde_json::json!({"request_id": "t:r2", "session_id": "p:m"})),
            row(serde_json::json!({"request_id": "cp000007", "policy_class": "priority"})),
            row(serde_json::json!({"request_id": "cp000009", "policy_class": "batch"})),
        ]
        .into_iter()
        .map(|r| (r.request_id.clone(), r))
        .collect();
        let a = resolve_request_fields(&fields, "cp000007000003:t:r1").unwrap();
        let b = resolve_request_fields(&fields, "cp000009000001:t:r1").unwrap();
        // Two users replaying the same play: separate streams, each prefixed as its sessions.
        assert_eq!(a.stream_id.as_deref(), Some("cp000007000003:p:a3"));
        assert_eq!(b.stream_id.as_deref(), Some("cp000009000001:p:a3"));
        assert_eq!(a.compaction, Some(true));
        // A row without a stream id keeps none (the replay then uses the session id).
        let c = resolve_request_fields(&fields, "cp000007000003:t:r2").unwrap();
        assert_eq!(c.stream_id, None);
    }
}
