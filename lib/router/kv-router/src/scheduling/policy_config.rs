// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fs;
use std::num::NonZeroU32;
use std::path::Path;

use rustc_hash::FxHashMap;
use serde::Deserialize;
use thiserror::Error;

use crate::plugins::request_classifier::RawRequestClassifierConfig;
use crate::plugins::request_classifier::RequestClassifierConfig;
use crate::plugins::worker_selection::{
    RawWorkerSelectionConfig, WorkerCapacity, WorkerCapacityExt, WorkerSelectionConfig,
};
use dynamo_router_protocol::worker::{WorkerId, WorkerWithDpRank};
use dynamo_scheduler::protocols::WorkerConfigLike;
use dynamo_slot_tracker::WorkerLoadProjection;

#[derive(Debug, Error)]
pub enum RouterPolicyConfigError {
    #[error("failed to read router policy config {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse router policy config {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_yaml::Error,
    },
    #[error("invalid router policy config: {0}")]
    Validation(String),
}

/// Supplies the host's live reject thresholds. Selection skips a worker whose every rank is
/// past them, and rejects the request when no eligible worker remains.
pub use dynamo_scheduler::types::RejectThresholdsProvider;

pub use dynamo_slot_tracker::busy::BusyRanksExt;
pub use dynamo_slot_tracker::busy::{BusyThresholds, FullRankCounts};

pub use dynamo_admission::profile::{
    AdmissionConfigError, PolicyClassConfig, PolicyProfile, RouterQueuePolicy,
    SYNTHETIC_POLICY_CLASS,
};

impl From<AdmissionConfigError> for RouterPolicyConfigError {
    fn from(error: AdmissionConfigError) -> Self {
        match error {
            AdmissionConfigError::Validation(message) => Self::Validation(message),
        }
    }
}

/// Judges a whole worker against [`BusyThresholds`] from its runtime config. Lives beside
/// [`WorkerConfigLike`] because the slot tracker does not know the config trait.
pub trait BusyThresholdsExt {
    /// Whether no configured rank of `worker_id` has room, judged on the `LOAD` rows in `loads`.
    fn worker_is_full<C: WorkerConfigLike>(
        &self,
        worker_id: WorkerId,
        config: &C,
        loads: &FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
    ) -> bool;
}

impl BusyThresholdsExt for BusyThresholds {
    fn worker_is_full<C: WorkerConfigLike>(
        &self,
        worker_id: WorkerId,
        config: &C,
        loads: &FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
    ) -> bool {
        let start = config.data_parallel_start_rank();
        let end = start.saturating_add(config.data_parallel_size());
        self.worker_is_full_in_ranks(worker_id, start..end, WorkerCapacity::of(config), loads)
    }
}

/// Process-wide cache and tracking settings, separate from policy parameters.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(transparent)]
pub(super) struct RouterSettings(HashMap<String, serde_json::Value>);

impl RouterSettings {
    pub(super) fn apply(&self, config: &mut super::config::KvRouterConfig) -> Result<(), String> {
        // Deserialize into each field's existing type, including nullable fields.
        // Do not round-trip KvRouterConfig: it has process-local, non-wire fields.
        macro_rules! apply_fields {
            ($($field:ident),* $(,)?) => {
                for (name, value) in &self.0 {
                    match name.as_str() {
                        $(stringify!($field) => {
                            config.$field = serde_json::from_value(value.clone())
                                .map_err(|error| format!("router.{name}: {error}"))?;
                        })*
                        _ => return Err(format!("unknown router setting: {name}")),
                    }
                }
            };
        }
        apply_fields!(
            host_cache_hit_weight,
            disk_cache_hit_weight,
            use_kv_events,
            router_replica_sync,
            router_track_active_blocks,
            router_track_output_blocks,
            router_assume_kv_reuse,
            router_track_prefill_tokens,
            router_tracking_hash,
            router_tracking_key_file,
            router_tracking_key_id,
            router_prefill_load_model,
            router_ttl_secs,
            router_approximate_cache_policy,
            router_event_threads,
            use_remote_indexer,
            serve_indexer,
            enable_session_prefix_index,
            shared_cache_type,
            router_predicted_ttl_secs,
        );
        Ok(())
    }
}

/// Per-request KV actions the router attaches to a booking (`kv_actions` in the policy file):
/// `kv.release_source` on every `kv.fetch` (a move), and `kv.retain` with a TTL per policy
/// class. The offline replay router reads it; the live router carries the same actions as a
/// hint policy once the hint-policy stack lands (docs/kv-action-channel.md in the simulator).
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvActionsConfig {
    /// Every KV fetch also releases its source's copy once the pull lands.
    #[serde(default)]
    pub release_source: bool,
    /// With `release_source`, a move releases only the blocks no other stream's latest request
    /// holds: a prefix the stream shares (a parent's context a subagent read, a common system
    /// prompt) stays on the source for the streams that still read it, as the simulator's
    /// per-session copies do. Off by default: the whole fetched prefix is released.
    #[serde(default)]
    pub release_unshared: bool,
    /// With `release_source`, a fetch moves only while its source is under host-tier (G2)
    /// pressure: the source evicted a G2 block (a removal this router did not ask for) within
    /// this many seconds. Otherwise the fetch copies and the source keeps its blocks, so moves
    /// stop when G2 holds everything and their pulls would only cost replicas (as the
    /// simulator's `kvTransfer: 'pressure'`). Off (absent) by default: every own-stream fetch moves.
    #[serde(default)]
    pub move_under_pressure_s: Option<f64>,
    /// Keep a finished request's KV for a TTL, per policy class.
    #[serde(default)]
    pub retain: Option<KvRetainConfig>,
    /// Retention by value in the workers' host tier (G2): send each request's session value.
    #[serde(default)]
    pub retain_value: Option<KvRetainValueConfig>,
    /// Drop dead KV on every worker: the blocks a stream's previous request held and this
    /// one does not. `marker` only at requests the harness marks as a compaction; `oracle`
    /// at every request.
    #[serde(default)]
    pub invalidate: Option<KvInvalidateMode>,
    /// Demote a finished request's KV from G1 to G2 for a long think time, and optionally
    /// prefetch it back before the predicted return.
    #[serde(default)]
    pub demote: Option<KvDemoteConfig>,
    /// A KV fetch keeps no copy in the target's host tier (G2): only its G1 holds the
    /// fetched prefix once the request finishes.
    #[serde(default)]
    pub fetch_no_retain: bool,
    /// `engine.dequeue`: hand a request still waiting in its worker's queue after `after_s`
    /// back to the router, which places it on an idle worker (web/sim.js `dequeue`).
    #[serde(default)]
    pub dequeue: Option<KvDequeueConfig>,
    /// KVCR may also read a peer's device KV (G1), not only its host tier (web/sim.js
    /// `kvcrPeerHbm`): a fetch hint may name a worker that holds the prefix only in HBM.
    /// Needs the AISimulate KVCR `peer_hbm` engine switch.
    #[serde(default)]
    pub peer_hbm: bool,
    /// Fan-out replicas (web/sim.js `fanout: 'replicate'`): when a subagent of a fan-out of at
    /// least `fanout` arrives (client hint `x-dynamo-hint-fanout`), copy the prefix it reads into
    /// the host tier (G2) of up to `replicas` workers over KVCR, ahead of its siblings.
    #[serde(default)]
    pub replicate: Option<KvReplicateConfig>,
    /// Free a session's KV when it ends (web/sim.js `freeKV: 'session'`): when a stream's request
    /// marked `session_final` (`nvext.agent_context.session_final`, header
    /// `x-dynamo-session-final`) finishes, drop on every worker the blocks of its last request
    /// that no other live stream's last request holds (a parent's context its subagents still
    /// read, a prompt another session shares), with `kv.invalidate`. Off by default.
    #[serde(default)]
    pub free_at_session_end: bool,
    /// f48h: an HBM ledger per worker from the engines' own block counts (active blocks, what
    /// the waiting queue must reserve) plus the hinted output of in-flight requests; see
    /// [`KvEngineFitConfig`]. Off (absent) by default. The offline replay reads it.
    #[serde(default)]
    pub engine_fit: Option<KvEngineFitConfig>,
    /// f62: keep big standard/batch prefills off priority's workers; see
    /// [`KvPrefillIsolationConfig`]. Off (absent) by default. The offline replay reads it.
    #[serde(default)]
    pub prefill_isolation: Option<KvPrefillIsolationConfig>,
}

/// `kv_actions.engine_fit`: a turn fits on a worker when the worker's engine-reported active
/// blocks, plus its waiting queue's tokens, plus its in-flight requests' hinted output, plus the
/// turn's whole prefix, stay within `margin` of its G1 blocks (vLLM reserves the whole prefix
/// before it admits a request, and its queue stops behind one that cannot).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvEngineFitConfig {
    /// Share of the worker's G1 blocks the ledger may fill.
    #[serde(default = "default_engine_fit_margin")]
    pub margin: f64,
    /// Exclude the workers a turn does not fit on from its selection, while one fits.
    #[serde(default = "default_engine_fit_place")]
    pub place: bool,
    /// Keep a queued turn at the router while it fits on no worker (classes with queueing).
    #[serde(default)]
    pub hold: bool,
    /// With `hold`, TTFT slack: hold a turn only while it has waited at the router less than
    /// this many seconds (its TTFT target less its expected prefill), then place it as without
    /// the hold, so the hold never costs a turn its target on its own. Absent: no limit.
    #[serde(default)]
    pub hold_max_s: Option<f64>,
    /// `kv_actions.dequeue` hands a stuck turn to a worker it fits on by this ledger, instead
    /// of by the router's bookings.
    #[serde(default)]
    pub dequeue: bool,
    /// Step-time budget as a concurrency limit: a turn also fits on a worker only while the
    /// engine's running plus waiting requests, with this turn, stay within it, so decode batches
    /// and co-scheduled prefill keep ITL under the class limit (a pro-tp4 worker sustains about
    /// 11k prefill tok/s at batch 48 under 50 ms ITL, 5-6k at 128). Absent: HBM only.
    #[serde(default)]
    pub max_requests: Option<u64>,
    /// Spread by least committed KV: of the workers the turn may go to, only those whose
    /// committed blocks (engine active blocks, plus the waiting queue's tokens, plus in-flight
    /// hinted output) are within this share of a worker's G1 blocks of the least committed one
    /// stay eligible; the router's costs pick among them. 0 is the least committed worker only.
    /// Absent: no spread (the fit check alone, if `place`).
    #[serde(default)]
    pub spread: Option<f64>,
}

/// f62 `kv_actions.prefill_isolation`: turns of `classes` leave the selection of a protected
/// worker on which they would compute more than `max_prefill_tokens`, so their big chunked
/// prefills do not share a step with priority decodes. Protected: the priority pool
/// (`workers: pool`) or the workers running at least `min_priority` priority requests
/// (`workers: priority_active`). Never holds, never empties a selection. The offline replay
/// reads it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct KvPrefillIsolationConfig {
    /// Classes kept off protected workers for big prefills.
    pub classes: Vec<String>,
    /// The class whose workers are protected.
    pub priority_class: String,
    /// Which workers are protected.
    pub workers: PrefillIsolationWorkers,
    /// With `priority_active`, the priority requests that make a worker protected.
    pub min_priority: usize,
    /// Tokens a turn may compute on a protected worker.
    pub max_prefill_tokens: usize,
}

impl Default for KvPrefillIsolationConfig {
    fn default() -> Self {
        Self {
            classes: vec!["standard".into(), "batch".into()],
            priority_class: "priority".into(),
            workers: PrefillIsolationWorkers::Pool,
            min_priority: 1,
            max_prefill_tokens: 8192,
        }
    }
}

/// `kv_actions.prefill_isolation.workers`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefillIsolationWorkers {
    /// The priority pool's workers (the pool scorer publishes its size).
    Pool,
    /// The workers running priority requests.
    PriorityActive,
    /// f68pool: the pool's workers while the pool acts (it publishes a size), else the workers
    /// running priority requests (a mix selector mode without the pool, such as g120's b0).
    PoolElseActive,
}

fn default_engine_fit_margin() -> f64 {
    0.95
}

fn default_engine_fit_place() -> bool {
    true
}

/// `kv_actions.replicate`: at most once per parent session within `window_s`, the deepest holder
/// of the subagent's prefix is the source and the least-loaded workers that do not hold it, up
/// to `replicas` holders in all, receive a G2 copy (path B `kv.replicate`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvReplicateConfig {
    #[serde(default = "default_replicate_fanout")]
    pub fanout: u32,
    #[serde(default = "default_replicate_replicas")]
    pub replicas: usize,
    #[serde(default = "default_replicate_window_s")]
    pub window_s: f64,
}

fn default_replicate_fanout() -> u32 {
    3
}

fn default_replicate_replicas() -> usize {
    2
}

fn default_replicate_window_s() -> f64 {
    60.0
}

/// `kv_actions.dequeue`: every `check_s`, a request of a class other than `batch` dispatched at
/// least `after_s` ago whose prefill has not completed is asked back from its worker when another
/// worker has no prefill booked, KV room for it and a free running slot; the one holding most of
/// its prefix takes it. Each request is asked at most once.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvDequeueConfig {
    #[serde(default = "default_dequeue_after_s")]
    pub after_s: f64,
    #[serde(default = "default_dequeue_check_s")]
    pub check_s: f64,
}

fn default_dequeue_after_s() -> f64 {
    2.0
}

fn default_dequeue_check_s() -> f64 {
    1.0
}

/// `kv_actions.invalidate`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvInvalidateMode {
    /// At requests marked `compaction` (the harness's signal).
    Marker,
    /// At every request (an upper bound: the router knows every dead block).
    Oracle,
}

/// `kv_actions.demote`: at a request's end, when the client expects the next turn no sooner
/// than `after_think_s` (`x-dynamo-hint-next-think-s`) and the worker's G1 is more than
/// `g1_fraction` full (resident blocks from its KV events over its advertised capacity),
/// drop the request's G1 blocks that G2 holds.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvDemoteConfig {
    #[serde(default = "default_demote_after_think_s")]
    pub after_think_s: f64,
    #[serde(default = "default_demote_g1_fraction")]
    pub g1_fraction: f64,
    /// Prefetch the demoted blocks back into G1 ahead of the predicted return.
    #[serde(default)]
    pub prefetch: Option<KvPrefetchConfig>,
    /// Send the demote this many seconds after the request ends instead of at its end (0, the
    /// default). The engine drops only blocks G2 holds, from the chain's tail; at the end of a
    /// turn the write-through store of its newest blocks is still in flight, so a demote sent
    /// then found its tail missing from G2 and dropped nothing.
    #[serde(default)]
    pub delay_s: f64,
}

/// `kv_actions.demote.prefetch`: start the load `lead_s` plus its transfer time (blocks
/// times `block_bytes` over `load_gbs`) before the predicted return.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvPrefetchConfig {
    #[serde(default = "default_prefetch_lead_s")]
    pub lead_s: f64,
    #[serde(default)]
    pub load_gbs: Option<f64>,
    #[serde(default)]
    pub block_bytes: Option<f64>,
}

fn default_demote_after_think_s() -> f64 {
    20.0
}

fn default_demote_g1_fraction() -> f64 {
    0.8
}

fn default_prefetch_lead_s() -> f64 {
    1.0
}

/// `kv_actions.retain_value`: a session's KV value is its class weight discounted by how long
/// until its next turn; the workers' G2 evicts the least valuable first.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvRetainValueConfig {
    /// Retain by value only while a worker's G2 evicted within this many seconds
    /// (retain-short); absent means always.
    #[serde(default)]
    pub short_window_s: Option<f64>,
    /// Class weights (default: priority 10, standard 3.33, batch 0.3, from 1 / (1 - target)).
    #[serde(default = "default_class_weights")]
    pub class_weight: HashMap<String, f64>,
    /// Memory-pressure guard: off while live sessions' KV exceeds this share of capacity.
    #[serde(default)]
    pub guard: Option<KvRetainGuardConfig>,
    /// How a session's value is reckoned: `session` (the default) is its class weight
    /// discounted by the wait to its next turn, the same however many blocks it holds, so per
    /// block a long session is the cheapest to evict; `rebuild_cost` is the cost of losing the
    /// blocks, as the simulator's `dramLoss`: class weight times min(1, rebuild time over the
    /// class's TTFT target) plus `stall_weight` times the rebuild time, with the same discount.
    #[serde(default)]
    pub value: RetainValueModel,
    /// `rebuild_cost`: milliseconds to recompute one 64-token block (prefill rate).
    #[serde(default = "default_rebuild_ms_per_block")]
    pub rebuild_ms_per_block: f64,
    /// `rebuild_cost`: TTFT target per class in seconds (a class not listed: 30).
    #[serde(default = "default_class_ttft_s")]
    pub class_ttft_s: HashMap<String, f64>,
    /// `rebuild_cost`: weight of the rebuild time itself (the stall it causes).
    #[serde(default = "default_stall_weight")]
    pub stall_weight: f64,
    /// `rebuild_cost`: a session another rank's G2 also holds costs a KVCR fetch to lose
    /// (`fetch_latency_ms` plus `fetch_ms_per_block` per block), not a rebuild, as the
    /// simulator's `policy.retainPeers`. Off by default.
    #[serde(default)]
    pub peers: bool,
    #[serde(default = "default_fetch_latency_ms")]
    pub fetch_latency_ms: f64,
    #[serde(default = "default_fetch_ms_per_block")]
    pub fetch_ms_per_block: f64,
}

/// `kv_actions.retain_value.value`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetainValueModel {
    #[default]
    Session,
    RebuildCost,
}

fn default_rebuild_ms_per_block() -> f64 {
    // DeepSeek-V4-Pro TP4 on B300: about 0.35 s per 8,192-token prefill chunk.
    3.0
}

fn default_class_ttft_s() -> HashMap<String, f64> {
    HashMap::from([
        ("priority".to_string(), 3.0),
        ("standard".to_string(), 8.0),
        ("batch".to_string(), 30.0),
    ])
}

fn default_stall_weight() -> f64 {
    0.5
}

fn default_fetch_latency_ms() -> f64 {
    100.0
}

fn default_fetch_ms_per_block() -> f64 {
    // 284,416 bytes per 64-token block (V4-Pro) at 50 GB/s.
    0.0057
}

fn default_class_weights() -> HashMap<String, f64> {
    HashMap::from([
        ("priority".to_string(), 10.0),
        ("standard".to_string(), 1.0 / 0.3),
        ("batch".to_string(), 0.3),
    ])
}

/// Engage retention by value while live sessions' KV over `capacity_tokens` is below
/// `engage_below`; release it above `release_above` (hysteresis).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvRetainGuardConfig {
    /// HBM plus host (G2) KV capacity of all workers, in tokens.
    pub capacity_tokens: f64,
    #[serde(default = "default_engage_below")]
    pub engage_below: f64,
    #[serde(default = "default_release_above")]
    pub release_above: f64,
}

fn default_engage_below() -> f64 {
    1.0
}

fn default_release_above() -> f64 {
    1.2
}

/// `kv_actions.retain`: TTL in seconds per policy class (a class not listed, or 0, is not kept).
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvRetainConfig {
    #[serde(default)]
    pub ttl_seconds: HashMap<String, f64>,
    /// Keep only a request the client expects to follow within the TTL: its
    /// `x-dynamo-hint-next-think-s` header is present and no longer than the TTL, and it
    /// does not end its session.
    #[serde(default)]
    pub expected: bool,
    /// Keep a request's blocks until the client's expected return (`x-dynamo-hint-next-think-s`)
    /// plus this margin, at most the class TTL, instead of for the whole class TTL (unset, the
    /// default). With a fixed TTL every expected session is protected alike; with the return time
    /// the sessions not due back soon are evicted first, as the simulator's `policy.evictOrder`
    /// evicts by predicted return. Needs `expected`.
    #[serde(default)]
    pub until_return_margin_s: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RouterPolicyConfig {
    router: Option<RouterSettings>,
    root: Option<PolicyProfile>,
    models: HashMap<String, PolicyProfile>,
    worker_selection: Option<WorkerSelectionConfig>,
    request_classifier: Option<RequestClassifierConfig>,
    kv_actions: Option<KvActionsConfig>,
    engine_priority: Option<HashMap<String, i32>>,
}

impl RouterPolicyConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, RouterPolicyConfigError> {
        let path = path.as_ref();
        let contents =
            fs::read_to_string(path).map_err(|source| RouterPolicyConfigError::Read {
                path: path.display().to_string(),
                source,
            })?;
        Self::from_yaml(&contents).map_err(|error| match error {
            RouterPolicyConfigError::Parse { source, .. } => RouterPolicyConfigError::Parse {
                path: path.display().to_string(),
                source,
            },
            other => other,
        })
    }

    pub fn from_yaml(contents: &str) -> Result<Self, RouterPolicyConfigError> {
        let raw: RawRouterPolicyConfig =
            serde_yaml::from_str(contents).map_err(|source| RouterPolicyConfigError::Parse {
                path: "<inline>".to_string(),
                source,
            })?;
        raw.resolve()
    }

    pub fn resolve_profile(
        &self,
        model_name: Option<&str>,
        fallback_threshold: Option<f64>,
        fallback_policy: RouterQueuePolicy,
    ) -> PolicyProfile {
        // Model profiles replace the root wholesale; the synthetic profile is
        // constructed only when neither configured profile applies.
        model_name
            .and_then(|name| self.models.get(name))
            .or(self.root.as_ref())
            .cloned()
            .unwrap_or_else(|| PolicyProfile::synthetic(fallback_threshold, fallback_policy))
    }

    pub(super) fn router(&self) -> Option<&RouterSettings> {
        self.router.as_ref()
    }

    /// Returns the process-wide worker-selection policy configuration, if present.
    pub fn worker_selection(&self) -> Option<&WorkerSelectionConfig> {
        self.worker_selection.as_ref()
    }

    /// Returns the process-wide request-classifier plugin configuration, if present.
    pub fn request_classifier(&self) -> Option<&RequestClassifierConfig> {
        self.request_classifier.as_ref()
    }

    /// Returns the per-request KV actions configuration, if present.
    pub fn kv_actions(&self) -> Option<&KvActionsConfig> {
        self.kv_actions.as_ref()
    }

    /// The backend engine priority for a policy class (`engine_priority` in the policy file),
    /// if the file maps that class. See [`RawRouterPolicyConfig::engine_priority`].
    pub fn engine_priority(&self, policy_class: &str) -> Option<i32> {
        self.engine_priority.as_ref()?.get(policy_class).copied()
    }

    /// Whether this document configures queue policy profiles.
    pub fn has_routing_profiles(&self) -> bool {
        self.root.is_some() || !self.models.is_empty()
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRouterPolicyConfig {
    router: Option<RouterSettings>,
    default_policy_class: Option<String>,
    policy_classes: Option<Vec<RawPolicyClassConfig>>,
    #[serde(default)]
    models: HashMap<String, RawPolicyProfile>,
    worker_selection: Option<RawWorkerSelectionConfig>,
    request_classifier: Option<RawRequestClassifierConfig>,
    kv_actions: Option<KvActionsConfig>,
    /// The backend engine scheduling priority per policy class (ledger L53), for example
    /// `{priority: 2, standard: 1, batch: 0}`. The router writes it into a routed request's
    /// `routing.priority` when the client set none; the backend forwards it to the engine's
    /// generate call (the vLLM worker passes `-routing.priority`, and vLLM with
    /// `--scheduling-policy priority` serves the lower value first, so a higher
    /// `routing.priority` goes first). Absent by default: `routing.priority` stays as sent.
    engine_priority: Option<HashMap<String, i32>>,
}

impl RawRouterPolicyConfig {
    fn resolve(self) -> Result<RouterPolicyConfig, RouterPolicyConfigError> {
        if let Some(router) = &self.router {
            router
                .apply(&mut super::config::KvRouterConfig::default())
                .map_err(RouterPolicyConfigError::Validation)?;
        }
        let root = match (self.default_policy_class, self.policy_classes) {
            (default_policy_class, Some(policy_classes)) => Some(resolve_profile(
                default_policy_class,
                policy_classes,
                "root",
            )?),
            (Some(_), None) => {
                return Err(RouterPolicyConfigError::Validation(
                    "root default_policy_class requires policy_classes".to_string(),
                ));
            }
            (None, None) => None,
        };

        let mut models = HashMap::with_capacity(self.models.len());
        for (model_name, profile) in self.models {
            if model_name.is_empty() {
                return Err(RouterPolicyConfigError::Validation(
                    "model profile name must not be empty".to_string(),
                ));
            }
            let location = format!("model {model_name:?}");
            let resolved = resolve_profile(
                profile.default_policy_class,
                profile.policy_classes,
                &location,
            )?;
            models.insert(model_name, resolved);
        }

        let worker_selection = match self.worker_selection {
            Some(config) => Some(config.resolve()?),
            None => None,
        };

        let request_classifier = self
            .request_classifier
            .map(|config| config.resolve())
            .transpose()?;
        if self.router.is_none()
            && root.is_none()
            && models.is_empty()
            && worker_selection.is_none()
            && request_classifier.is_none()
            && self.kv_actions.is_none()
            && self.engine_priority.is_none()
        {
            return Err(RouterPolicyConfigError::Validation(
                "router policy config must define router settings, a root profile, at least one model profile, worker_selection, or request_classifier".to_string(),
            ));
        }

        Ok(RouterPolicyConfig {
            router: self.router,
            root,
            models,
            worker_selection,
            request_classifier,
            kv_actions: self.kv_actions,
            engine_priority: self.engine_priority,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPolicyProfile {
    default_policy_class: Option<String>,
    policy_classes: Vec<RawPolicyClassConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPolicyClassConfig {
    name: String,
    /// A configured class defaults to EDF; the synthetic class keeps the router-wide policy.
    #[serde(default = "default_class_queue_policy")]
    queue_policy: RouterQueuePolicy,
    quantum: usize,
    slo_ttft_ms: Option<u32>,
    #[serde(default)]
    controlled_delay: bool,
    prefill_busy_threshold: Option<usize>,
    prefill_busy_threshold_frac: Option<f64>,
    decode_busy_threshold_frac: Option<f64>,
    request_queue_limit_per_worker: Option<usize>,
    raw_isl_token_queue_limit_per_worker: Option<usize>,
    cached_token_queue_limit_per_worker: Option<usize>,
    /// f49h: per-class cap on active requests per worker rank (`PolicyClassConfig`). Absent: none.
    active_requests_per_worker: Option<usize>,
}

fn default_class_queue_policy() -> RouterQueuePolicy {
    RouterQueuePolicy::Edf
}

fn resolve_profile(
    default_policy_class: Option<String>,
    raw_classes: Vec<RawPolicyClassConfig>,
    location: &str,
) -> Result<PolicyProfile, RouterPolicyConfigError> {
    if raw_classes.is_empty() {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy_classes must not be empty"
        )));
    }

    let classes = raw_classes
        .into_iter()
        .map(|raw| resolve_policy_class(raw, location))
        .collect::<Result<Vec<_>, _>>()?;
    // The admission stage owns the class table's rules (unique names, a default class); this
    // file supplies where in the file they were broken.
    PolicyProfile::from_classes(classes, default_policy_class.as_deref()).map_err(
        |AdmissionConfigError::Validation(message)| {
            RouterPolicyConfigError::Validation(format!("{location} {message}"))
        },
    )
}

fn resolve_policy_class(
    raw: RawPolicyClassConfig,
    location: &str,
) -> Result<PolicyClassConfig, RouterPolicyConfigError> {
    validate_identifier(&raw.name, "policy class", location)?;
    if raw.quantum == 0 {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy class {:?} quantum must be greater than zero",
            raw.name
        )));
    }
    if raw.queue_policy == RouterQueuePolicy::Lcfs {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy class {:?} queue_policy must be edf, fcfs, or wspt",
            raw.name
        )));
    }
    if raw.controlled_delay && raw.queue_policy != RouterQueuePolicy::Edf {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy class {:?} controlled_delay requires queue_policy edf",
            raw.name
        )));
    }
    for (field, value) in [
        (
            "prefill_busy_threshold_frac",
            raw.prefill_busy_threshold_frac,
        ),
        ("decode_busy_threshold_frac", raw.decode_busy_threshold_frac),
    ] {
        if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return Err(RouterPolicyConfigError::Validation(format!(
                "{location} policy class {:?} {field} must be finite and non-negative",
                raw.name
            )));
        }
    }
    let slo_ttft_ms = match raw.slo_ttft_ms {
        None => None,
        Some(slo_ttft_ms) => Some(NonZeroU32::new(slo_ttft_ms).ok_or_else(|| {
            RouterPolicyConfigError::Validation(format!(
                "{location} policy class {:?} slo_ttft_ms must be greater than zero",
                raw.name
            ))
        })?),
    };

    if raw.active_requests_per_worker == Some(0) {
        return Err(RouterPolicyConfigError::Validation(format!(
            "{location} policy class {:?} active_requests_per_worker must be greater than zero",
            raw.name
        )));
    }
    if raw.prefill_busy_threshold.is_some() {
        tracing::warn!(
            policy_class = %raw.name,
            "prefill_busy_threshold is deprecated and will be removed in v1.7; use prefill_busy_threshold_frac"
        );
    }
    Ok(PolicyClassConfig {
        name: raw.name,
        queue_policy: raw.queue_policy,
        quantum: raw.quantum,
        slo_ttft_ms,
        controlled_delay: raw.controlled_delay,
        busy_thresholds: BusyThresholds {
            prefill_busy_threshold: raw.prefill_busy_threshold,
            prefill_busy_threshold_frac: raw.prefill_busy_threshold_frac,
            decode_busy_threshold_frac: raw.decode_busy_threshold_frac,
        },
        request_queue_limit_per_worker: raw.request_queue_limit_per_worker,
        raw_isl_token_queue_limit_per_worker: raw.raw_isl_token_queue_limit_per_worker,
        cached_token_queue_limit_per_worker: raw.cached_token_queue_limit_per_worker,
        active_requests_per_worker: raw.active_requests_per_worker,
    })
}

pub(crate) fn validate_identifier(
    name: &str,
    kind: &str,
    location: &str,
) -> Result<(), RouterPolicyConfigError> {
    if !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-'))
    {
        return Ok(());
    }

    Err(RouterPolicyConfigError::Validation(format!(
        "{location} {kind} name {name:?} must match [A-Za-z0-9_.-]+"
    )))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::time::Instant;

    use super::*;

    #[test]
    fn kv_actions_parse_and_default_off() {
        let config = RouterPolicyConfig::from_yaml(
            "kv_actions:\n  release_source: true\n  retain:\n    expected: true\n    ttl_seconds: {priority: 3600, standard: 3600}\n",
        )
        .unwrap();
        let actions = config.kv_actions().unwrap();
        assert!(actions.release_source);
        let retain = actions.retain.as_ref().unwrap();
        assert!(retain.expected);
        assert_eq!(retain.ttl_seconds.get("priority"), Some(&3600.0));
        let rejected = RouterPolicyConfig::from_yaml("kv_actions:\n  unknown: 1\n");
        assert!(rejected.is_err());
        assert_eq!(actions.invalidate, None);
        assert!(actions.demote.is_none() && !actions.fetch_no_retain);
    }

    #[test]
    fn kv_block_actions_parse_with_defaults() {
        let config = RouterPolicyConfig::from_yaml(
            "kv_actions:\n  invalidate: oracle\n  fetch_no_retain: true\n  demote:\n    prefetch: {load_gbs: 50}\n",
        )
        .unwrap();
        let actions = config.kv_actions().unwrap();
        assert_eq!(actions.invalidate, Some(KvInvalidateMode::Oracle));
        assert!(actions.fetch_no_retain);
        let demote = actions.demote.as_ref().unwrap();
        assert_eq!((demote.after_think_s, demote.g1_fraction), (20.0, 0.8));
        let prefetch = demote.prefetch.as_ref().unwrap();
        assert_eq!((prefetch.lead_s, prefetch.load_gbs), (1.0, Some(50.0)));
        assert!(RouterPolicyConfig::from_yaml("kv_actions:\n  invalidate: always\n").is_err());
    }
    use crate::plugins::worker_selection::WorkerSelectionPolicySelection;
    use crate::plugins::worker_selection::{WorkerLoadInput, WorkerLoadInputExt};
    use dynamo_admission::profile::UnknownPolicyClassError;
    use dynamo_scheduler::types::KvSchedulerError;
    use dynamo_slot_tracker::WorkerLoadProjection;

    fn prefill_load(active_prefill_tokens: usize) -> WorkerLoadInput {
        WorkerLoadInput::from_projection(WorkerLoadProjection {
            active_prefill_tokens,
            ..Default::default()
        })
    }

    #[test]
    fn worker_selection_only_config_preserves_parameter_mapping() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
worker_selection:
  aggregated:
    type: example-policy
    parameters:
      score_weight: 1.0
"#,
        )
        .unwrap();

        let selection = config.worker_selection().unwrap();
        let Some(WorkerSelectionPolicySelection::Inline(instance)) = selection.aggregated() else {
            panic!("an inline spec resolves to an inline selection");
        };
        assert_eq!(instance.policy_type(), "example-policy");
        assert_eq!(selection.aggregated().unwrap().label(), "example-policy");
        assert!(matches!(
            instance.parameters(),
            serde_yaml::Value::Mapping(_)
        ));
        assert!(selection.instance_names().is_empty());
        assert_eq!(
            config
                .resolve_profile(None, Some(2.0), RouterQueuePolicy::Wspt)
                .default_class()
                .queue_policy,
            RouterQueuePolicy::Wspt
        );
    }

    #[test]
    fn request_classifier_only_config_preserves_parameter_mapping() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
request_classifier:
  type: thunderagent
  parameters:
    pause_threshold: 0.9
"#,
        )
        .unwrap();

        let classifier = config.request_classifier().unwrap();
        assert_eq!(classifier.classifier_type(), "thunderagent");
        assert!(matches!(
            classifier.parameters(),
            serde_yaml::Value::Mapping(_)
        ));
    }

    #[test]
    fn rejects_invalid_request_classifier_config() {
        for yaml in [
            r#"
request_classifier:
  type: default
"#,
            r#"
request_classifier:
  type: thunderagent
  parameters: 1
"#,
            r#"
request_classifier:
  type: ""
"#,
        ] {
            assert!(
                RouterPolicyConfig::from_yaml(yaml).is_err(),
                "unexpectedly accepted {yaml}"
            );
        }
    }

    #[test]
    fn worker_selection_accepts_every_worker_type() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
worker_selection:
  prefill: cache-aware
  decode: default
  encode:
    type: media-aware
"#,
        )
        .unwrap();

        let selection = config.worker_selection().unwrap();
        assert_eq!(selection.aggregated(), None);
        assert_eq!(
            selection.prefill(),
            Some(&WorkerSelectionPolicySelection::Named(
                "cache-aware".to_string()
            ))
        );
        assert!(selection.decode().unwrap().is_default());
        assert_eq!(selection.encode().unwrap().label(), "media-aware");
        assert!(matches!(
            selection.encode(),
            Some(WorkerSelectionPolicySelection::Inline(instance))
                if instance.parameters() == &serde_yaml::Value::Mapping(Default::default())
        ));
    }

    /// The deprecated `instances:` list still resolves: roles reference entries by name.
    #[test]
    fn worker_selection_accepts_deprecated_named_instances() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
worker_selection:
  prefill: prefill-policy
  decode: default
  encode: encode-policy
  instances:
    - name: prefill-policy
      type: cache-aware
      parameters: {}
    - name: encode-policy
      type: media-aware
      parameters: {}
"#,
        )
        .unwrap();

        let selection = config.worker_selection().unwrap();
        assert_eq!(selection.aggregated(), None);
        assert_eq!(selection.prefill().unwrap().label(), "prefill-policy");
        assert!(selection.decode().unwrap().is_default());
        assert_eq!(selection.encode().unwrap().label(), "encode-policy");
        assert_eq!(
            selection.instance("prefill-policy").unwrap().policy_type(),
            "cache-aware"
        );
        assert_eq!(
            selection.instance_names(),
            vec!["encode-policy".to_string(), "prefill-policy".to_string()]
        );
    }

    #[test]
    fn rejects_invalid_worker_selection_config() {
        for yaml in [
            r#"
worker_selection: {}
"#,
            r#"
worker_selection:
  aggregated: ""
"#,
            r#"
worker_selection:
  aggregated: "two words"
"#,
            r#"
worker_selection:
  aggregated:
    type: alpha
    parameters: 1
"#,
            r#"
worker_selection:
  aggregated:
    type: default
"#,
            r#"
worker_selection:
  aggregated:
    name: alpha
    type: alpha
"#,
            r#"
worker_selection:
  aggregated:
    parameters: {}
"#,
            r#"
worker_selection:
  default: alpha
"#,
            r#"
worker_selection:
  instances:
    - name: default
      type: alpha
"#,
            r#"
worker_selection:
  instances:
    - name: alpha
      type: alpha
      parameters: 1
"#,
        ] {
            assert!(
                RouterPolicyConfig::from_yaml(yaml).is_err(),
                "unexpectedly accepted {yaml}"
            );
        }
    }

    #[test]
    fn model_profile_replaces_root_and_unmatched_model_uses_root() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
policy_classes:
  - name: root-default
    queue_policy: wspt
    quantum: 8
    prefill_busy_threshold: 100
models:
  exact-model:
    default_policy_class: model-direct
    policy_classes:
      - name: model-queued
        quantum: 4
        prefill_busy_threshold_frac: 0.0
      - name: model-direct
        quantum: 2
        request_queue_limit_per_worker: 0
"#,
        )
        .unwrap();

        let exact = config.resolve_profile(Some("exact-model"), Some(3.0), RouterQueuePolicy::Wspt);
        assert_eq!(exact.classes().len(), 2);
        assert_eq!(exact.default_class().name, "model-direct");
        assert_eq!(
            exact
                .default_class()
                .busy_thresholds
                .prefill_busy_threshold_frac,
            None
        );
        assert!(!exact.default_class().queueing_enabled());
        assert!(
            exact
                .class(exact.resolve_class_index(Some("model-queued")).unwrap())
                .queueing_enabled()
        );
        // A configured class without queue_policy is EDF, whatever the router-wide policy.
        assert_eq!(exact.default_class().queue_policy, RouterQueuePolicy::Edf);
        assert_eq!(
            exact.default_class().request_queue_limit_per_worker,
            Some(0)
        );
        assert_eq!(
            exact.class(exact.resolve_class_index(None).unwrap()).name,
            "model-direct"
        );
        for requested in ["unknown", "root-default"] {
            assert!(
                exact.resolve_class_index(Some(requested)).is_err(),
                "{requested:?} must not resolve in the model profile"
            );
        }

        let unmatched = config.resolve_profile(Some("other"), Some(3.0), RouterQueuePolicy::Fcfs);
        assert_eq!(unmatched.default_class().name, "root-default");
        assert_eq!(
            unmatched
                .default_class()
                .busy_thresholds
                .prefill_busy_threshold,
            Some(100)
        );
        assert_eq!(
            unmatched
                .default_class()
                .busy_thresholds
                .prefill_busy_threshold_frac,
            None
        );
    }

    #[test]
    fn has_room_without_limits_or_reported_budget() {
        let mut limits = BusyThresholds::default();
        assert!(!limits.is_configured());
        assert!(limits.has_room(&prefill_load(usize::MAX), WorkerCapacity::default()));

        // An unreported budget is taken as DEFAULT_MAX_BATCHED_TOKENS.
        limits.prefill_busy_threshold_frac = Some(0.5);
        let half_default =
            (dynamo_router_protocol::worker::DEFAULT_MAX_BATCHED_TOKENS / 2) as usize;
        assert!(limits.has_room(&prefill_load(half_default), WorkerCapacity::default()));
        assert!(!limits.has_room(&prefill_load(half_default + 1), WorkerCapacity::default()));
    }

    #[test]
    fn decode_limit_bounds_held_kv_blocks() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
policy_classes:
  - name: decode-bound
    quantum: 4
    decode_busy_threshold_frac: 0.5
"#,
        )
        .unwrap();
        let profile = config.resolve_profile(None, None, RouterQueuePolicy::Fcfs);
        let class = profile.default_class();
        assert!(class.queueing_enabled());

        let limits = class.busy_thresholds;
        let held = |active_decode_blocks| {
            WorkerLoadInput::from_projection(WorkerLoadProjection {
                active_decode_blocks,
                // The incoming request's blocks do not count toward the limit.
                additional_active_blocks: 1_000,
                ..Default::default()
            })
        };
        let kv = |total_kv_blocks| WorkerCapacity::new(None, total_kv_blocks);
        assert!(limits.has_room(&held(50), kv(Some(100))));
        assert!(!limits.has_room(&held(51), kv(Some(100))));
        // An unreported KV capacity skips the decode limit.
        assert!(limits.has_room(&held(1_000_000), kv(None)));

        let invalid = RouterPolicyConfig::from_yaml(
            r#"
policy_classes:
  - name: decode-bound
    quantum: 4
    decode_busy_threshold_frac: -0.1
"#,
        )
        .unwrap_err();
        assert!(
            invalid
                .to_string()
                .contains("decode_busy_threshold_frac must be finite and non-negative")
        );
    }

    #[test]
    fn rootless_model_config_falls_back_for_unmatched_model() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
models:
  exact-model:
    policy_classes:
      - name: absolute
        quantum: 4
        prefill_busy_threshold: 10
        prefill_busy_threshold_frac: 0.5
"#,
        )
        .unwrap();

        let exact = config.resolve_profile(Some("exact-model"), Some(7.0), RouterQueuePolicy::Wspt);
        let limits = exact.default_class().busy_thresholds;
        let budget = |tokens| WorkerCapacity::new(Some(tokens), None);
        assert!(!limits.has_room(&prefill_load(11), budget(10_000_000)));
        assert!(!limits.has_room(&prefill_load(6), budget(10)));
        assert!(limits.has_room(&prefill_load(5), budget(10)));

        let fallback = config.resolve_profile(Some("other"), Some(7.0), RouterQueuePolicy::Wspt);
        assert_eq!(fallback.default_class().name, SYNTHETIC_POLICY_CLASS);
        assert_eq!(
            fallback
                .default_class()
                .busy_thresholds
                .prefill_busy_threshold_frac,
            Some(7.0)
        );
        assert_eq!(
            fallback.default_class().queue_policy,
            RouterQueuePolicy::Wspt
        );
        // Only an omitted name or the synthetic class's own name selects it.
        for requested in [None, Some(SYNTHETIC_POLICY_CLASS)] {
            assert_eq!(fallback.resolve_class_index(requested).ok(), Some(0));
        }
        assert!(fallback.resolve_class_index(Some("absolute")).is_err());
        assert_eq!(
            fallback.resolve_class_index_strict(SYNTHETIC_POLICY_CLASS),
            Some(0)
        );
        assert_eq!(fallback.resolve_class_index_strict("absolute"), None);
    }

    #[test]
    fn classes_resolve_by_name_and_only_omitted_names_use_the_default() {
        let profile = RouterPolicyConfig::from_yaml(
            r#"
default_policy_class: standard
policy_classes:
  - name: priority
    quantum: 800
  - name: standard
    quantum: 200
  - name: batch
    quantum: 100
"#,
        )
        .unwrap()
        .resolve_profile(None, None, RouterQueuePolicy::Fcfs);

        // Every configured class is its own queue, selected by its exact name.
        for (index, name) in ["priority", "standard", "batch"].into_iter().enumerate() {
            assert_eq!(profile.class(index).name, name);
            assert_eq!(profile.resolve_class_index(Some(name)).ok(), Some(index));
            assert_eq!(profile.resolve_class_index_strict(name), Some(index));
        }
        // An omitted name uses the configured default, not the first class. Any other name
        // must match a configured class exactly.
        assert_eq!(profile.resolve_class_index(None).ok(), Some(1));
        for requested in ["", "unknown", "Priority"] {
            let error = profile
                .resolve_class_index(Some(requested))
                .expect_err(requested);
            assert_eq!(error, UnknownPolicyClassError(requested.to_string()));
            // The kernel refuses a request naming it at queue entry.
            assert!(
                matches!(
                    KvSchedulerError::from(error),
                    KvSchedulerError::UnknownPolicyClass(name) if name == requested
                ),
                "{requested:?}"
            );
            assert_eq!(profile.resolve_class_index_strict(requested), None);
        }

        // One class is the default without default_policy_class, and may also name itself.
        for default in ["", "default_policy_class: only\n"] {
            let single = RouterPolicyConfig::from_yaml(&format!(
                "{default}policy_classes:\n  - name: only\n    quantum: 1\n"
            ))
            .unwrap()
            .resolve_profile(None, None, RouterQueuePolicy::Fcfs);
            assert_eq!(single.default_class().name, "only");
            assert_eq!(single.resolve_class_index(Some("only")).ok(), Some(0));
            assert!(single.resolve_class_index(Some("unknown")).is_err());
        }
    }

    #[test]
    fn rejects_invalid_profiles() {
        for (yaml, expected) in [
            (
                "policy_classes: []\n",
                "root policy_classes must not be empty",
            ),
            (
                "policy_classes:\n  - name: first\n    quantum: 1\n  - name: second\n    quantum: 2\n",
                "root default_policy_class is required with more than one policy class",
            ),
            (
                "default_policy_class: missing\npolicy_classes:\n  - name: only\n    quantum: 1\n",
                "root default_policy_class \"missing\" does not name a configured policy class",
            ),
            (
                "default_policy_class: only\n",
                "root default_policy_class requires policy_classes",
            ),
            (
                "default_policy_class: same\npolicy_classes:\n  - name: same\n    quantum: 1\n  - name: same\n    quantum: 2\n",
                "root contains duplicate policy class \"same\"",
            ),
            (
                "policy_classes:\n  - name: invalid/name\n    quantum: 1\n",
                "root policy class name \"invalid/name\" must match [A-Za-z0-9_.-]+",
            ),
            (
                "policy_classes:\n  - name: zero\n    quantum: 0\n",
                "root policy class \"zero\" quantum must be greater than zero",
            ),
            (
                "policy_classes:\n  - name: lcfs\n    queue_policy: lcfs\n    quantum: 1\n",
                "root policy class \"lcfs\" queue_policy must be edf, fcfs, or wspt",
            ),
            (
                "models:\n  m:\n    policy_classes:\n      - name: a\n        quantum: 1\n      - name: b\n        quantum: 1\n",
                "model \"m\" default_policy_class is required with more than one policy class",
            ),
            (
                "models:\n  m:\n    default_policy_class: absent\n    policy_classes:\n      - name: a\n        quantum: 1\n",
                "model \"m\" default_policy_class \"absent\" does not name a configured policy class",
            ),
        ] {
            let error = RouterPolicyConfig::from_yaml(yaml).unwrap_err().to_string();
            assert!(error.contains(expected), "{yaml}: {error}");
        }
    }

    #[test]
    fn removed_hierarchy_keys_are_rejected_even_when_null_or_empty() {
        let assert_unknown_field = |yaml: String, key: &str| {
            let error = RouterPolicyConfig::from_yaml(&yaml).unwrap_err();
            assert!(
                matches!(error, RouterPolicyConfigError::Parse { .. })
                    && error
                        .to_string()
                        .contains(&format!("unknown field `{key}`")),
                "{yaml}: {error}"
            );
        };
        for value in ["", "null", "\"\"", "[]", "standard"] {
            for key in ["default_policy_family", "uncached_isl_buckets"] {
                assert_unknown_field(
                    format!("{key}: {value}\npolicy_classes:\n  - name: c\n    quantum: 1\n"),
                    key,
                );
                assert_unknown_field(
                    format!(
                        "models:\n  m:\n    {key}: {value}\n    policy_classes:\n      - name: c\n        quantum: 1\n"
                    ),
                    key,
                );
            }
            for key in ["policy_family", "cache_bucket"] {
                assert_unknown_field(
                    format!("policy_classes:\n  - name: c\n    quantum: 1\n    {key}: {value}\n"),
                    key,
                );
                assert_unknown_field(
                    format!(
                        "models:\n  m:\n    policy_classes:\n      - name: c\n        quantum: 1\n        {key}: {value}\n"
                    ),
                    key,
                );
            }
        }
    }

    #[test]
    fn controlled_delay_is_a_per_class_edf_opt_in() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
default_policy_class: omitted
policy_classes:
  - name: enabled
    quantum: 1
    slo_ttft_ms: 100
    controlled_delay: true
  - name: omitted
    quantum: 1
    slo_ttft_ms: 100
  - name: disabled
    quantum: 1
    controlled_delay: false
  - name: ordered
    queue_policy: wspt
    quantum: 1
    controlled_delay: false
models:
  m:
    policy_classes:
      - name: enabled
        quantum: 1
        slo_ttft_ms: 100
"#,
        )
        .unwrap();
        let enabled = |model| -> Vec<(String, bool)> {
            config
                .resolve_profile(model, None, RouterQueuePolicy::Fcfs)
                .classes()
                .iter()
                .map(|class| (class.name.clone(), class.controlled_delay))
                .collect()
        };
        let expected = |pairs: &[(&str, bool)]| -> Vec<(String, bool)> {
            pairs
                .iter()
                .map(|&(name, on)| (name.to_string(), on))
                .collect()
        };
        // Each class opts in alone; omitted is off, and any order may say false.
        let root = expected(&[
            ("enabled", true),
            ("omitted", false),
            ("disabled", false),
            ("ordered", false),
        ]);
        assert_eq!(enabled(None), root);
        assert_eq!(enabled(Some("other")), root);
        // A model profile replaces the root wholesale, settings included.
        assert_eq!(enabled(Some("m")), expected(&[("enabled", false)]));
        // Without a configured profile, the synthetic class is off.
        assert!(
            !PolicyProfile::synthetic(Some(0.5), RouterQueuePolicy::Edf)
                .default_class()
                .controlled_delay
        );

        // The setting is a YAML boolean, and only EDF can enforce it.
        let class = |setting: &str| {
            RouterPolicyConfig::from_yaml(&format!(
                "policy_classes:\n  - name: c\n    quantum: 1\n{setting}"
            ))
            .map(|_| ())
            .map_err(|error| error.to_string())
        };
        for setting in [
            "    controlled_delay: \"true\"\n",
            "    controlled_delay: 1\n",
            "    controlled_delay: null\n",
        ] {
            assert!(class(setting).is_err(), "{setting}");
        }
        for policy in ["fcfs", "wspt"] {
            let error = class(&format!(
                "    queue_policy: {policy}\n    controlled_delay: true\n"
            ))
            .unwrap_err();
            assert!(
                error
                    .contains("root policy class \"c\" controlled_delay requires queue_policy edf"),
                "{error}"
            );
        }
        class("    queue_policy: edf\n    controlled_delay: true\n").unwrap();

        // A queue expires a class budget only where the class opts in and orders by EDF.
        let base = PolicyProfile::synthetic(None, RouterQueuePolicy::Edf)
            .default_class()
            .clone();
        for (queue_policy, controlled_delay, expires) in [
            (RouterQueuePolicy::Edf, true, true),
            (RouterQueuePolicy::Edf, false, false),
            (RouterQueuePolicy::Wspt, true, false),
            (RouterQueuePolicy::Fcfs, true, false),
        ] {
            let class = PolicyClassConfig {
                queue_policy,
                controlled_delay,
                ..base.clone()
            };
            assert_eq!(
                class.expires_class_budget(),
                expires,
                "{queue_policy:?} {controlled_delay}"
            );
        }
    }

    #[test]
    fn slo_ttft_ms_is_an_optional_positive_millisecond_budget() {
        let slo_ttft_ms = |setting: &str| {
            RouterPolicyConfig::from_yaml(&format!(
                "policy_classes:\n  - name: c\n    quantum: 1\n{setting}"
            ))
            .map(|config| {
                config
                    .resolve_profile(None, None, RouterQueuePolicy::Fcfs)
                    .default_class()
                    .slo_ttft_ms
            })
        };

        assert_eq!(slo_ttft_ms("").unwrap(), None);
        assert_eq!(slo_ttft_ms("    slo_ttft_ms: null\n").unwrap(), None);
        assert_eq!(
            slo_ttft_ms("    slo_ttft_ms: 1\n").unwrap(),
            NonZeroU32::new(1)
        );
        // The maximum is u32::MAX milliseconds; one more is an ordinary parse error.
        assert_eq!(
            slo_ttft_ms(&format!("    slo_ttft_ms: {}\n", u32::MAX)).unwrap(),
            NonZeroU32::new(u32::MAX)
        );
        assert!(matches!(
            slo_ttft_ms(&format!("    slo_ttft_ms: {}\n", u64::from(u32::MAX) + 1)),
            Err(RouterPolicyConfigError::Parse { .. })
        ));

        let error = slo_ttft_ms("    slo_ttft_ms: 0\n").unwrap_err().to_string();
        assert!(
            error.contains("root policy class \"c\" slo_ttft_ms must be greater than zero"),
            "{error}"
        );
        for malformed in [
            "-1",
            "1.5",
            "\"3000\"",
            "3000ms",
            "18446744073709551616",
            "[]",
        ] {
            assert!(
                matches!(
                    slo_ttft_ms(&format!("    slo_ttft_ms: {malformed}\n")),
                    Err(RouterPolicyConfigError::Parse { .. })
                ),
                "unexpectedly accepted slo_ttft_ms {malformed}"
            );
        }
    }

    #[test]
    fn slo_ttft_due_at_counts_the_budget_from_ingress() {
        let class = |slo_ttft_ms| PolicyClassConfig {
            slo_ttft_ms: NonZeroU32::new(slo_ttft_ms),
            ..PolicyProfile::synthetic(None, RouterQueuePolicy::Edf)
                .default_class()
                .clone()
        };
        let ingress_at = Instant::now();
        assert_eq!(class(0).slo_ttft_due_at(ingress_at), None);
        assert_eq!(
            class(1500).slo_ttft_due_at(ingress_at),
            Some(ingress_at + Duration::from_millis(1500))
        );
        assert_eq!(
            class(u32::MAX).slo_ttft_due_at(ingress_at),
            Some(ingress_at + Duration::from_millis(u64::from(u32::MAX)))
        );
    }

    #[test]
    fn omitted_queue_policy_selects_edf_only_for_configured_classes() {
        let queue_policy = |setting: &str| {
            RouterPolicyConfig::from_yaml(&format!(
                "policy_classes:\n  - name: c\n    quantum: 1\n{setting}"
            ))
            .map(|config| {
                config
                    .resolve_profile(None, None, RouterQueuePolicy::Wspt)
                    .default_class()
                    .queue_policy
            })
        };

        assert_eq!(queue_policy("").unwrap(), RouterQueuePolicy::Edf);
        for policy in [
            RouterQueuePolicy::Edf,
            RouterQueuePolicy::Fcfs,
            RouterQueuePolicy::Wspt,
        ] {
            assert_eq!(
                queue_policy(&format!("    queue_policy: {policy}\n")).unwrap(),
                policy
            );
        }
        assert!(queue_policy("    queue_policy: lcfs\n").is_err());

        // The synthetic class keeps the router-wide policy: FCFS by default, LCFS included.
        assert_eq!(RouterQueuePolicy::default(), RouterQueuePolicy::Fcfs);
        for policy in [
            RouterQueuePolicy::default(),
            RouterQueuePolicy::Edf,
            RouterQueuePolicy::Lcfs,
        ] {
            assert_eq!(
                PolicyProfile::synthetic(None, policy)
                    .default_class()
                    .queue_policy,
                policy
            );
        }
    }

    #[test]
    fn documented_sample_exercises_root_and_model_profiles() {
        let config = RouterPolicyConfig::from_yaml(include_str!(
            "../../../../../examples/router/policy-class-queues.yaml"
        ))
        .unwrap();

        let root = config.resolve_profile(None, None, RouterQueuePolicy::Fcfs);
        let classes = root
            .classes()
            .iter()
            .map(|class| (class.name.as_str(), class.quantum, class.slo_ttft_ms))
            .collect::<Vec<_>>();
        assert_eq!(
            classes,
            [
                ("priority", 800, NonZeroU32::new(1000)),
                ("standard", 200, NonZeroU32::new(3000)),
            ]
        );
        assert!(root.classes().iter().all(|class| {
            class.queue_policy == RouterQueuePolicy::Edf && class.queueing_enabled()
        }));
        assert_eq!(
            root.class(root.resolve_class_index(Some("priority")).unwrap())
                .quantum,
            800
        );
        assert_eq!(
            root.class(root.resolve_class_index(None).unwrap()).name,
            "standard"
        );
        assert!(root.resolve_class_index(Some("unknown")).is_err());

        let model = config.resolve_profile(
            Some("example/large-model"),
            Some(3.0),
            RouterQueuePolicy::Wspt,
        );
        assert_eq!(model.classes().len(), 1);
        assert_eq!(model.default_class().name, "standard");
        assert_eq!(model.default_class().quantum, 1000);
        assert!(
            model.resolve_class_index(Some("priority")).is_err(),
            "model profiles must completely replace root classes"
        );
    }

    #[test]
    fn prefill_budget_defaults_when_unreported() {
        use dynamo_router_protocol::worker::DEFAULT_MAX_BATCHED_TOKENS;
        assert_eq!(
            WorkerCapacity::default().prefill_budget(),
            DEFAULT_MAX_BATCHED_TOKENS
        );
        assert_eq!(WorkerCapacity::new(Some(64), None).prefill_budget(), 64);
        assert_eq!(WorkerCapacity::default().max_num_batched_tokens(), None);
        // Every reported value round-trips; there is no sentinel.
        let max = WorkerCapacity::new(Some(u64::MAX), Some(u64::MAX));
        assert_eq!(max.max_num_batched_tokens(), Some(u64::MAX));
        assert_eq!(max.total_kv_blocks(), Some(u64::MAX));
    }

    #[test]
    fn unobserved_rank_row_is_unavailable_and_idle() {
        let seen = WorkerWithDpRank::new(1, 0);
        let unseen = WorkerWithDpRank::new(1, 1);
        let mut loads = FxHashMap::default();
        loads.insert(
            seen,
            WorkerLoadProjection {
                active_prefill_tokens: 7,
                ..Default::default()
            },
        );
        let unobserved = || WorkerLoadProjection {
            additional_active_blocks: 3,
            ..Default::default()
        };

        let row = WorkerLoadInput::for_worker(&loads, seen, unobserved);
        assert!(row.is_available());
        assert_eq!(row.active_prefill_tokens(), 7);

        let row = WorkerLoadInput::for_worker(&loads, unseen, unobserved);
        assert!(!row.is_available());
        assert_eq!(row.active_prefill_tokens(), 0);
        assert_eq!(row.decode_cost_blocks(), 3.0);
    }

    #[test]
    fn worker_is_full_only_when_every_rank_is_full() {
        let limits = BusyThresholds {
            prefill_busy_threshold: Some(10),
            ..Default::default()
        };
        let config = crate::test_utils::SimpleWorkerConfig {
            data_parallel_start_rank: 2,
            data_parallel_size: 2,
            ..Default::default()
        };
        let capacity = WorkerCapacity::of(&config);
        let busy = WorkerLoadProjection {
            active_prefill_tokens: 11,
            ..Default::default()
        };
        let mut loads = FxHashMap::default();

        loads.insert(WorkerWithDpRank::new(5, 2), busy);
        // Rank 3 has no projection and counts as idle.
        assert!(limits.rank_is_full(WorkerWithDpRank::new(5, 2), capacity, &loads));
        assert!(!limits.rank_is_full(WorkerWithDpRank::new(5, 3), capacity, &loads));
        assert!(!limits.worker_is_full(5, &config, &loads));

        loads.insert(WorkerWithDpRank::new(5, 3), busy);
        assert!(limits.worker_is_full(5, &config, &loads));

        // A rank outside the worker's range does not stand in for one inside it.
        loads.remove(&WorkerWithDpRank::new(5, 3));
        loads.insert(WorkerWithDpRank::new(5, 0), busy);
        assert!(!limits.worker_is_full(5, &config, &loads));
    }

    #[test]
    fn zero_capacity_is_unreported() {
        let zero = WorkerCapacity::new(Some(0), Some(0));
        assert_eq!(zero.max_num_batched_tokens(), None);
        assert_eq!(zero.total_kv_blocks(), None);
        assert_eq!(zero, WorkerCapacity::default());

        // vLLM under the Ray data-parallel backend registers `total_kv_blocks = 0` as a
        // placeholder; the decode limit skips such a rank instead of calling it full.
        let limits = BusyThresholds {
            decode_busy_threshold_frac: Some(0.5),
            ..Default::default()
        };
        let held = WorkerLoadInput::from_projection(WorkerLoadProjection {
            active_decode_blocks: 1_000,
            ..Default::default()
        });
        assert!(limits.has_room(&held, WorkerCapacity::new(None, Some(0))));
        assert!(!limits.has_room(&held, WorkerCapacity::new(None, Some(1_000))));

        // A zero prefill budget falls back to the default budget instead of being full at once.
        let limits = BusyThresholds {
            prefill_busy_threshold_frac: Some(0.5),
            ..Default::default()
        };
        assert!(limits.has_room(&prefill_load(1), WorkerCapacity::new(Some(0), None)));
    }

    #[test]
    fn engine_priority_maps_policy_classes_and_is_absent_by_default() {
        let config = RouterPolicyConfig::from_yaml(
            "default_policy_class: standard\npolicy_classes:\n  - name: priority\n    quantum: 8\n  - name: standard\n    quantum: 4\nengine_priority: {priority: 0, standard: 1}\n",
        )
        .unwrap();
        assert_eq!(config.engine_priority("priority"), Some(0));
        assert_eq!(config.engine_priority("standard"), Some(1));
        assert_eq!(config.engine_priority("batch"), None);
        let plain = RouterPolicyConfig::from_yaml(
            "default_policy_class: standard\npolicy_classes:\n  - name: standard\n    quantum: 4\n",
        )
        .unwrap();
        assert_eq!(plain.engine_priority("standard"), None);
        // It alone is a valid policy file.
        let only = RouterPolicyConfig::from_yaml("engine_priority: {batch: 5}\n").unwrap();
        assert_eq!(only.engine_priority("batch"), Some(5));
    }
}

#[cfg(test)]
mod active_requests_per_worker_tests {
    use super::*;

    #[test]
    fn active_requests_per_worker_is_an_optional_per_class_cap() {
        let config = RouterPolicyConfig::from_yaml(
            r#"
default_policy_class: standard
policy_classes:
  - name: priority
    quantum: 8
    prefill_busy_threshold_frac: 1.0
    active_requests_per_worker: 8
    request_queue_limit_per_worker: 16
  - name: standard
    quantum: 4
    prefill_busy_threshold_frac: 0.95
  - name: batch
    quantum: 1
    active_requests_per_worker: 2
"#,
        )
        .unwrap();
        let profile = config.resolve_profile(None, None, RouterQueuePolicy::Fcfs);
        let cap = |name: &str| {
            let index = profile.resolve_class_index(Some(name)).unwrap();
            profile.class(index).active_requests_per_worker
        };
        assert_eq!(cap("priority"), Some(8));
        assert_eq!(cap("standard"), None);
        assert_eq!(cap("batch"), Some(2));
        // A cap alone makes a class queue, as a busy threshold does.
        let batch = profile.class(profile.resolve_class_index(Some("batch")).unwrap());
        assert!(!batch.busy_thresholds.is_configured());
        assert!(batch.queueing_enabled());

        let zero = RouterPolicyConfig::from_yaml(
            "policy_classes:\n  - name: a\n    quantum: 1\n    active_requests_per_worker: 0\n",
        )
        .unwrap_err();
        assert!(
            zero.to_string()
                .contains("active_requests_per_worker must be greater than zero"),
            "{zero}"
        );
    }
}
