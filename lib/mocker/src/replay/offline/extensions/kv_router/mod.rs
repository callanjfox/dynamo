// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::replay::default_selector::{DefaultWorkerSelector, DefaultWorkerSelectorControl};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use dynamo_admission::host::{
    PolicyClassConfig, PolicyProfile, PolicyQueue, QueueMetadata, QueueSnapshot, RecheckScope,
    SessionLane,
};
use dynamo_admission::{
    BlockReason, ClassifyEvent, ClassifyRequest, DispatchDecision, DispatchHead, HoldWake,
    LaneCheck, WorkerCacheView, WorkerLoadView,
};
use dynamo_kv_events::event::{KvCacheEventData, RouterEvent};
use dynamo_kv_events::residency::StorageTier;
use dynamo_kv_hashing::LocalBlockHash;
use dynamo_kv_hashing::routing::{BlockHashOptions, compute_block_hash_for_seq};
use dynamo_kv_hints::KvTransferSourceChoice;
use dynamo_kv_indexer::host::RadixTree;
use dynamo_kv_indexer::host::{
    KvIndexerInterface, LowerTierIndexers, LowerTierQueryOptions, SessionPrefixIndexer,
    TieredMatchDetails, query_lower_tiers_with_options,
};
use dynamo_kv_indexer::kv_hints::{KvTransferCandidateSource, KvTransferCandidates};
pub(in crate::replay) use dynamo_kv_router::config::KvRouterConfig as ReplayKvRouterConfig;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::WorkerCapacity;
use dynamo_kv_router::scheduling::BusyThresholdsExt;
use dynamo_kv_router::tracking_hash::tracking_hash_context;
use dynamo_router_protocol::columns::{WorkerInfo, WorkerInputs};
use dynamo_router_protocol::identity::RoutingPartitionRef;
use dynamo_router_protocol::session::SessionContext;
use dynamo_router_protocol::worker::{WorkerId, WorkerWithDpRank};
use dynamo_scheduler::host::{
    CacheHitWeights, OverlapAnalysis, OverlapSignals, RoutingConstraints, ScheduleMode,
    SchedulingContext, SchedulingRequest, TrackingHashAlgorithm, TrackingHashContext,
    TrackingHashScope, WorkerCapacityExt, WorkerConfigLike, WorkerSelectionInput, WorkerSelector,
};
use dynamo_session_affinity::host::{
    AcquireStep, AffinityError, AffinityLease, AffinityTarget, Hold, SessionAffinity,
    SessionAffinityConfig, subagent_group_affinity_id,
};
use dynamo_slot_tracker::host::{
    ActiveSequencesMultiWorker, BusyRanksExt, PrefillLoadHint, SequenceRequest, WorkerDpRange,
    WorkerLoadProjection,
};
use dynamo_tokens::SequenceHash;
use futures::executor::block_on;
use futures::future::try_join_all;
use rustc_hash::FxHashMap;
use tokio::time::Instant;
use uuid::Uuid;

use crate::common::protocols::DirectRequest;
use crate::common::protocols::MockerConfig;
use crate::replay::offline::extensions::kv_events::RouterEventBatch;
use crate::replay::router_shared::{
    ReplayNoopPublisher, ReplayWorkerConfig, replay_router_config, replay_selector_with_plugins,
    replay_slots, replay_worker_config,
};
use crate::replay::{
    ConversationAffinityConfig, ConversationAffinityMode, ReplayPrefillLoadEstimator,
};
use aisimulate_core::replay::loadgen::{ReplayRequestHashes, ReplayRequestPayload};
use aisimulate_core::replay::{
    KvAction, KvActionMessage, KvActionRequest, KvActionResult, KvActionStatus, KvFetchAction,
    KvHints, KvRetainAction, KvRetainValue, Placement, PlacementCacheSample, PlacementDecision,
    PlacementEffects, PlacementPolicy, ProviderSpec, ReplayAdmissionMetadata, WorkerTopology,
};
use dynamo_kv_router::scheduling::{
    KvDemoteConfig, KvDequeueConfig, KvInvalidateMode, KvReplicateConfig,
};

mod class_cap;
mod composition;
mod engine_fit;
mod gate;
mod prefill_iso;
mod workset;
pub(in crate::replay) use composition::{KvReplayComposition, RoundRobinReplayComposition};
use gate::{
    KeyedHolds, ReplayClassifier, ReplayWorkerView, RetryEntry, VirtualClock, retry_delay_ms,
};
pub use gate::{
    ReplayGateRecheck, ReplayGateStats, ReplayRequestAnnotations, ReplayRequestFields,
    load_request_annotations,
};

#[derive(Clone, Copy)]
enum KvEventSummary {
    Stored {
        parent_hash: Option<dynamo_kv_hashing::ExternalSequenceBlockHash>,
        start_position: Option<u32>,
        count: usize,
        first: Option<dynamo_kv_hashing::ExternalSequenceBlockHash>,
        last: Option<dynamo_kv_hashing::ExternalSequenceBlockHash>,
    },
    Removed {
        count: usize,
        first: Option<dynamo_kv_hashing::ExternalSequenceBlockHash>,
        last: Option<dynamo_kv_hashing::ExternalSequenceBlockHash>,
    },
    Cleared,
}

impl KvEventSummary {
    fn from_data(data: &dynamo_kv_events::event::KvCacheEventData) -> Self {
        match data {
            dynamo_kv_events::event::KvCacheEventData::Stored(stored) => Self::Stored {
                parent_hash: stored.parent_hash,
                start_position: stored.start_position,
                count: stored.blocks.len(),
                first: stored.blocks.first().map(|block| block.block_hash),
                last: stored.blocks.last().map(|block| block.block_hash),
            },
            dynamo_kv_events::event::KvCacheEventData::Removed(removed) => Self::Removed {
                count: removed.block_hashes.len(),
                first: removed.block_hashes.first().copied(),
                last: removed.block_hashes.last().copied(),
            },
            dynamo_kv_events::event::KvCacheEventData::Cleared => Self::Cleared,
        }
    }
}

struct KvEventLabel {
    worker_id: WorkerId,
    dp_rank: u32,
    event_id: u64,
    summary: KvEventSummary,
}

impl KvEventLabel {
    fn new(event: &RouterEvent) -> Self {
        Self {
            worker_id: event.worker_id,
            dp_rank: event.event.dp_rank,
            event_id: event.event.event_id,
            summary: KvEventSummary::from_data(&event.event.data),
        }
    }
}

impl fmt::Display for KvEventLabel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "worker={} dp_rank={} event_id={} data={}",
            self.worker_id, self.dp_rank, self.event_id, self.summary
        )
    }
}

impl fmt::Display for KvEventSummary {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stored {
                parent_hash,
                start_position,
                count,
                first,
                last,
            } => write!(
                formatter,
                "stored parent={parent_hash:?} start={start_position:?} count={count} first={first:?} last={last:?}"
            ),
            Self::Removed { count, first, last } => write!(
                formatter,
                "removed count={count} first={first:?} last={last:?}"
            ),
            Self::Cleared => formatter.write_str("cleared"),
        }
    }
}

/// Serializable descriptor for the Dynamo-owned KV-router replay provider.
///
/// Keeping this value with the adapter prevents the neutral offline entrypoint
/// from naming or depending on the concrete Router crate.
pub(in crate::replay) fn provider_spec() -> ProviderSpec {
    ProviderSpec {
        provider: "dynamo_kv_router".to_string(),
        config: serde_json::Value::Null,
    }
}

/// Dynamo-owned metadata used by KV-aware placement. AISimulate only defines
/// the neutral admission-metadata contract and never imports Router types.
#[derive(Debug, Default)]
pub(in crate::replay) struct KvReplayMetadata {
    hashes: Option<ReplayRequestHashes>,
    max_output_tokens_override: Option<usize>,
}

impl ReplayAdmissionMetadata for KvReplayMetadata {
    fn from_hashes(hashes: Option<ReplayRequestHashes>) -> Self {
        Self {
            hashes,
            max_output_tokens_override: None,
        }
    }

    fn for_prefill(mut self) -> Self {
        self.max_output_tokens_override = Some(1);
        self
    }

    fn max_output_tokens_override(&self) -> Option<usize> {
        self.max_output_tokens_override
    }

    fn into_hashes(self) -> Option<ReplayRequestHashes> {
        self.hashes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorkerAdmission {
    uuid: Uuid,
    worker_idx: usize,
    overlap_blocks: u32,
    best_available_overlap_blocks: u32,
    isl_blocks: u32,
    /// Scheduler id of the peer to fetch the prompt prefix from.
    kv_fetch_source: Option<usize>,
    /// `kv.retain` TTL for this request's KV after it finishes.
    retain_ttl_ms: Option<u64>,
    /// `kv.retain` value of this request's session.
    retain_value: Option<KvRetainValue>,
    /// Live context of this request's session (prompt plus expected output), for the guard.
    context_tokens: u64,
}

#[derive(Debug, Default)]
pub(crate) struct RouterEffects {
    admissions: Vec<WorkerAdmission>,
}

/// Internal result of a successful ``admit_request`` call: the chosen
/// worker plus the router's view of prefix-cache overlap, so callers can
/// forward the overlap stats to the traffic accumulator.
#[derive(Debug, Clone, Copy)]
struct AdmitOutcome {
    worker_idx: usize,
    overlap_blocks: u32,
    best_available_overlap_blocks: u32,
    isl_blocks: u32,
    kv_fetch_source: Option<usize>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OfflinePendingRequestSnapshot {
    pub(crate) uuid: Uuid,
    pub(crate) expected_output_tokens: Option<u32>,
    pub(crate) overlap_blocks_by_worker: Vec<(usize, u32)>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OfflineIndexerSnapshot {
    pub(crate) total_cached_blocks: usize,
    pub(crate) cached_blocks_by_worker: Vec<(usize, usize)>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OfflineRouterSnapshot {
    pub(crate) pending: Vec<OfflinePendingRequestSnapshot>,
    pub(crate) active_blocks_by_worker: Vec<(usize, usize)>,
    pub(crate) active_tokens_by_worker: Vec<(usize, usize)>,
    pub(crate) indexer: OfflineIndexerSnapshot,
}

struct SyncReplayIndexer {
    block_size: u32,
    tree: RadixTree,
    /// Non-device (for example cluster-shared G2 `HostPinned`) residency,
    /// allocated on the first lower-tier event so G1-only replays pay nothing.
    lower_tier: Option<LowerTierIndexers>,
    /// Disaggregated decode placement never observes KV events, so its tree
    /// stays empty and hashing a prompt to query it would be wasted work.
    has_indexed_events: bool,
    /// Logical lineage uses the production index, while residency remains in `tree`.
    session_prefix_index: Option<Arc<SessionPrefixIndexer>>,
    /// Whether lookups keep the matched chain that KV fetch hints need.
    retain_kv_transfer_chain: bool,
}

impl SyncReplayIndexer {
    fn new(block_size: u32) -> Self {
        Self {
            block_size,
            tree: RadixTree::new(),
            lower_tier: None,
            has_indexed_events: false,
            session_prefix_index: None,
            retain_kv_transfer_chain: false,
        }
    }

    fn find_matches_for_request(
        &self,
        tokens: &[u32],
        lora_name: Option<&str>,
    ) -> TieredMatchDetails {
        if !self.has_indexed_events {
            return TieredMatchDetails::default();
        }
        let sequence = compute_block_hash_for_seq(
            tokens,
            self.block_size,
            BlockHashOptions {
                lora_name,
                ..Default::default()
            },
        );
        self.find_matches_for_hashes(sequence)
    }

    fn find_matches_for_hashes(&self, sequence: Vec<LocalBlockHash>) -> TieredMatchDetails {
        let retain = self.retain_kv_transfer_chain;
        let Some(lower_tier) = &self.lower_tier else {
            return TieredMatchDetails {
                device: self
                    .tree
                    .find_match_details_with_options(sequence, false, retain),
                lower_tier: Default::default(),
            };
        };
        let device = self
            .tree
            .find_match_details_with_options(sequence.clone(), false, retain);
        let lower_tier = query_lower_tiers_with_options(
            lower_tier,
            &sequence,
            &device,
            LowerTierQueryOptions {
                retain_kv_transfer_chain: retain,
            },
        );
        TieredMatchDetails { device, lower_tier }
    }

    /// Applies a batch in order. Lower-tier events are queued on their indexer
    /// threads without a per-event round trip; one wait at the end acks them
    /// all, so later placements still observe the whole batch.
    fn apply_events(&mut self, events: Vec<RouterEvent>) -> Result<()> {
        let mut lower_tier_acks = Vec::new();
        for event in events {
            let label = KvEventLabel::new(&event);
            self.has_indexed_events = true;
            if event.storage_tier.is_gpu() {
                // The synchronous physical update must succeed before logical lineage advances.
                // Clone only when opted in; events may already be delayed by the replay clock.
                let session_event = self.session_prefix_index.as_ref().map(|_| event.clone());
                self.tree
                    .apply_event(event)
                    .with_context(|| format!("failed to apply replay KV event {label}"))?;
                if let Some(event) = session_event {
                    self.apply_session_event(&event);
                }
                continue;
            }
            let block_size = self.block_size;
            let indexer = self
                .lower_tier
                .get_or_insert_with(|| LowerTierIndexers::new(1, block_size))
                .get_or_create(event.storage_tier);
            lower_tier_acks.push(async move {
                indexer
                    .apply_event_and_wait(event)
                    .await
                    .with_context(|| format!("failed to apply replay KV event {label}"))
            });
        }
        block_on(try_join_all(lower_tier_acks))?;
        Ok(())
    }

    #[cfg(test)]
    fn apply_event(&mut self, event: RouterEvent) -> Result<()> {
        self.apply_events(vec![event])
    }

    fn remove_worker(&mut self, worker_id: WorkerId) {
        self.tree.remove_worker(worker_id);
        if let Some(index) = &self.session_prefix_index {
            index.clear_worker_id_frontiers(worker_id);
        }
        if let Some(lower_tier) = &self.lower_tier {
            for indexer in lower_tier.all() {
                block_on(indexer.remove_worker(worker_id));
            }
        }
    }

    fn apply_session_event(&self, event: &RouterEvent) {
        let Some(index) = &self.session_prefix_index else {
            return;
        };
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        match &event.event.data {
            KvCacheEventData::Stored(stored) => {
                if let Some(session_id) = &event.session_id {
                    let hashes = stored
                        .blocks
                        .iter()
                        .map(|block| block.block_hash)
                        .collect::<Vec<_>>();
                    if let Err(error) = index.update_session_from_stored_blocks(
                        session_id,
                        worker,
                        stored.parent_hash,
                        &hashes,
                    ) {
                        // As in the live router, incomplete attribution must not fail routing.
                        tracing::warn!(%error, %session_id, ?worker, "failed to record stored session blocks");
                    }
                }
            }
            KvCacheEventData::Removed(removed) => {
                index.update_session_from_removed_blocks(worker, &removed.block_hashes);
            }
            KvCacheEventData::Cleared => {
                index.clear_worker_frontiers(worker);
            }
        }
    }

    #[cfg(test)]
    fn debug_snapshot(&self) -> OfflineIndexerSnapshot {
        let mut blocks_by_worker = HashMap::<usize, usize>::new();
        for event in self.tree.dump_tree_as_events() {
            *blocks_by_worker
                .entry(event.worker_id as usize)
                .or_default() += 1;
        }
        let mut cached_blocks_by_worker = blocks_by_worker.into_iter().collect::<Vec<_>>();
        cached_blocks_by_worker.sort_unstable_by_key(|(worker_id, _)| *worker_id);

        OfflineIndexerSnapshot {
            total_cached_blocks: self.tree.current_size(),
            cached_blocks_by_worker,
        }
    }
}

struct PendingRequest {
    uuid: Uuid,
    token_seq: Option<Vec<SequenceHash>>,
    isl_tokens: usize,
    /// Tiered overlap scored by the configured router (G1 plus weighted G2 hits).
    overlap: OverlapSignals,
    track_prefill_tokens: bool,
    expected_output_tokens: Option<u32>,
    priority_jump: f64,
    strict_priority: u32,
    policy_class: Option<String>,
    session_id: Option<String>,
    /// Authored local hashes for affinity refresh and session-prefix lookup.
    /// Active-load hashes can be keyed/randomized, so cannot substitute for these.
    block_hashes: Option<Vec<LocalBlockHash>>,
    affinity_id: Option<String>,
    affinity_hold: Option<Hold>,
    /// From arrival: unlike production, replay never refreshes a queued request's candidates.
    kv_transfer_candidates: Option<KvTransferCandidates>,
    /// The session context a frontend would build (parent session, session end, agent
    /// headers), from the replay's request fields; `None` builds one from `session_id`.
    session_context: Option<SessionContext>,
    /// The classifier's deadline (`ClassificationOverrides::due_at`): expired at it (504).
    due_at: Option<Instant>,
    /// The classifier's Retry-After for an expiry (`ClassificationOverrides::retry_after`):
    /// the client retries after it instead of its backoff.
    retry_after: Option<Duration>,
    /// The classifier's ordering key (`ClassificationOverrides::order_key`): orders it in its
    /// class's queue, lower first, and never expires it.
    order_key: Option<f64>,
    /// When the request entered the router queue (virtual time), for the gate's head view.
    queued_at: Option<Instant>,
    /// The dispatch gate's `Go` target, the soft affinity target unless an affinity hold binds.
    gate_target: Option<WorkerWithDpRank>,
    /// `kv.retain` TTL for this request's KV after it finishes (policy file `kv_actions`).
    retain_ttl_ms: Option<u64>,
    /// `kv.retain` value (policy file `kv_actions.retain_value`).
    retain_value: Option<KvRetainValue>,
    /// This request's blocks for `kv.invalidate` and `kv.demote` (policy file `kv_actions`).
    kv_blocks: Option<KvBlockPlan>,
    /// The KV stream this request continues (request field `stream_id`, else its session).
    stream_id: Option<String>,
    /// `engine.dequeue`: the worker a handed-back request is placed on.
    dequeue_target: Option<WorkerWithDpRank>,
    /// `kv.replicate`: a fan-out subagent's parent session and prompt blocks (sequence and
    /// token hashes), when replicas are on.
    replica: Option<(String, Vec<u64>, Vec<u64>)>,
}

/// A request's blocks as the KV block actions see them: by sequence hash (the engine's KV
/// event identity) and token hash, with the stream they continue.
#[derive(Debug, Clone)]
struct KvBlockPlan {
    stream_id: String,
    sequence: Vec<u64>,
    local: Vec<u64>,
    compaction: bool,
    think_ms: Option<u64>,
    /// `kv_actions.free_at_session_end`: this request ends its stream (`session_final`).
    session_final: bool,
}

impl PendingRequest {
    /// The copy kept for `engine.dequeue`: what placing it again needs. No affinity hold (the
    /// lease stays with the original), no KV block plan (its actions were planned once).
    fn copy_for_dequeue(&self) -> Self {
        Self {
            uuid: self.uuid,
            token_seq: self.token_seq.clone(),
            isl_tokens: self.isl_tokens,
            overlap: self.overlap.clone(),
            track_prefill_tokens: self.track_prefill_tokens,
            expected_output_tokens: self.expected_output_tokens,
            priority_jump: self.priority_jump,
            strict_priority: self.strict_priority,
            policy_class: self.policy_class.clone(),
            session_id: self.session_id.clone(),
            block_hashes: self.block_hashes.clone(),
            affinity_id: None,
            affinity_hold: None,
            kv_transfer_candidates: self.kv_transfer_candidates.clone(),
            session_context: self.session_context.clone(),
            due_at: None,
            retry_after: None,
            order_key: None,
            queued_at: self.queued_at,
            gate_target: None,
            retain_ttl_ms: self.retain_ttl_ms,
            retain_value: self.retain_value,
            kv_blocks: None,
            stream_id: self.stream_id.clone(),
            dequeue_target: None,
            replica: None,
        }
    }

    fn request_id(&self) -> String {
        self.uuid.to_string()
    }

    /// Build the selector input, moving `token_seq` and the KV fetch candidates
    /// into it; callers that still need them take them back from the result.
    fn scheduling_request(
        &mut self,
        worker_loads: FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
    ) -> SchedulingRequest {
        SchedulingRequest {
            accounting_attribution: Default::default(),
            mode: ScheduleMode::Tracked {
                request_id: self.request_id(),
            },
            token_seq: self.token_seq.take(),
            isl_tokens: self.isl_tokens,
            overlap: std::mem::take(&mut self.overlap),
            kv_transfer_candidates: self.kv_transfer_candidates.take(),
            retain_kv_transfer_chain: false,
            worker_loads,
            track_prefill_tokens: self.track_prefill_tokens,
            router_config_override: None,
            lora_name: None,
            priority_jump: self.priority_jump,
            strict_priority: self.strict_priority,
            policy_class: self.policy_class.clone(),
            session_context: self.session_context.clone().or_else(|| {
                self.session_id
                    .clone()
                    .map(|session_id| SessionContext::new(session_id, None, None, None))
            }),
            expected_output_tokens: self.expected_output_tokens,
            affinity_target: self
                .affinity_hold
                .as_ref()
                .and_then(Hold::target)
                .or_else(|| self.gate_target.map(Into::into)),
            pinned_worker: self
                .affinity_hold
                .as_ref()
                .and_then(Hold::target)
                .and_then(|target| {
                    target
                        .dp_rank
                        .map(|rank| WorkerWithDpRank::new(target.worker_id, rank))
                })
                .or(self.dequeue_target),
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: None,
        }
    }
}

pub(crate) struct OfflineReplayRouter {
    config: KvRouterConfig,
    block_size: u32,
    dp_size: u32,
    profile: PolicyProfile,
    worker_config_template: ReplayWorkerConfig,
    workers_with_configs: HashMap<WorkerId, ReplayWorkerConfig>,
    slots: Arc<ActiveSequencesMultiWorker<ReplayNoopPublisher>>,
    selector: DefaultWorkerSelector,
    pending: PolicyQueue<PendingRequest>,
    indexer: SyncReplayIndexer,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    decay_time_epoch: Instant,
    tracking_hash: TrackingHashContext,
    /// Simulated delay before KV events reach the indexer; `0.0` applies them synchronously.
    /// Completions never lag: a live router learns them in-band on the response path.
    kv_event_lag_ms: f64,
    /// Lagged KV event batches in arrival order, keyed by the replay time they become visible.
    lagged_kv_events: VecDeque<(f64, Vec<RouterEvent>)>,
    affinity: Option<(ConversationAffinityConfig, SessionAffinity)>,
    affinity_waiting: VecDeque<PendingRequest>,
    affinity_staged: HashMap<Uuid, (Hold, AffinityTarget)>,
    affinity_active: HashMap<Uuid, AffinityLease>,
    needs_drain: bool,
    /// Fixed at startup, like production's index; see `new_with_selector_seed`.
    kv_transfer_chain_supported: bool,
    kv_transfer_gate_warned: bool,
    /// The request classifier and dispatch gate, when the host supplied one (replay gate).
    classifier: Option<ReplayClassifier>,
    /// Per-request fields a frontend would attach, by authored request id.
    request_fields: Option<Arc<ReplayRequestAnnotations>>,
    /// A queue-limit rejection goes back to the client (`DYNAMO_REPLAY_QUEUE_REJECT_RETRY=1`)
    /// instead of ending the run; and how many did.
    queue_reject_retry: bool,
    queue_rejections: u64,
    /// `DYNAMO_REPLAY_REFUSAL_LOG=<path>` (off by default): one JSON line per request's first offer
    /// to the router and per refusal sent back to its client (queue-limit rejection, classifier
    /// refusal or gate expiry), so a run can count what refusals cost each client.
    refusal_log: Option<std::io::BufWriter<std::fs::File>>,
    /// `DYNAMO_REPLAY_WORKSET_LOG=<path>` (off by default): the live sessions' KV working set
    /// against what the workers hold, one JSON line per simulated minute (`workset.rs`).
    workset: Option<workset::WorksetLog>,
    /// `DYNAMO_REPLAY_BUSY_RANKED=1`: each class's busy test leaves out prefill booked by lower
    /// classes (the engines schedule by class); `None` when off.
    busy_ranked: Option<RankedBooking>,
    /// f49h `policy_classes[].active_requests_per_worker`: each class's requests admitted per rank
    /// (`class_cap.rs`), while any class sets a cap; `None` otherwise.
    class_active: Option<class_cap::ClassActive>,
    /// Path B (out of band): the KV action sender and the moves waiting for confirmation.
    kv_actions: KvActionSender,
    /// Refused requests (429 or 504) waiting for their client's retry, by retry time.
    retry_waiting: Vec<RetryEntry<PendingRequest>>,
    /// The earliest `at_ms` in `retry_waiting` (kept with it, so a visit with nothing due does
    /// not scan the list).
    retry_next_ms: Option<f64>,
    /// Refusals so far, per request: the client's backoff grows with them.
    retry_attempts: HashMap<Uuid, u32>,
    /// The earliest gate hold deadline still in the future, in replay ms.
    gate_wake_ms: Option<f64>,
    /// The worker each dispatched request went to and the context it leaves (prompt plus
    /// expected output), for its lifecycle events.
    dispatched_to: HashMap<Uuid, (WorkerWithDpRank, usize, usize)>,
    /// f47k (`DYNAMO_REPLAY_PLACED_CACHE`): the selected worker's cached prompt tokens, per request.
    placed_cached: HashMap<Uuid, usize>,
    /// ks2b (`DYNAMO_REPLAY_ENGINE_GAUGES`): the gauges last sent to the classifier, per rank.
    gauges_sent: HashMap<WorkerWithDpRank, [u64; 5]>,
    /// `engine.dequeue` (policy file `kv_actions.dequeue`).
    dequeue: DequeueState,
    /// f48h `kv_actions.engine_fit`: the engines' HBM ledger; `None` when off.
    engine_fit: Option<engine_fit::EngineFit>,
    /// f62 `kv_actions.prefill_isolation`; `None` when off.
    prefill_iso: Option<prefill_iso::PrefillIsolation>,
    /// `kv.replicate`: when each parent session's fan-out was last replicated (replay ms), and
    /// the replicas sent.
    replicated: HashMap<String, f64>,
    replicas_sent: u64,
}

/// A dispatched request kept for `engine.dequeue` until its prefill completes.
struct KeptRequest {
    request: PendingRequest,
    source: usize,
    worker: WorkerWithDpRank,
    at_ms: f64,
}

/// `engine.dequeue`: requests that may be handed back, those asked, and their new workers.
#[derive(Default)]
struct DequeueState {
    kept: HashMap<Uuid, KeptRequest>,
    asked: HashSet<Uuid>,
    targets: HashMap<Uuid, WorkerWithDpRank>,
    next_check_ms: f64,
    stats: DequeueStats,
}

/// `engine.dequeue` outcomes, logged when replay ends (`replay_dequeue_stats`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DequeueStats {
    pub(crate) asked: u64,
    pub(crate) requeued: u64,
    pub(crate) no_target: u64,
}

impl Drop for DequeueState {
    fn drop(&mut self) {
        let s = self.stats;
        if s.asked > 0 {
            tracing::info!(
                asked = s.asked,
                requeued = s.requeued,
                no_target = s.no_target,
                "replay_dequeue_stats"
            );
        }
    }
}

pub(in crate::replay) struct KvRouterPlacement {
    router: OfflineReplayRouter,
}

impl KvRouterPlacement {
    pub(in crate::replay) fn new_with_selector_seed(
        args: &MockerConfig,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        topology: &[WorkerTopology],
        selector_seed: Option<u64>,
    ) -> Result<Self> {
        // Composition checks that `topology[i]` is worker `i`.
        let router_hints: Vec<bool> = topology.iter().map(|worker| worker.router_hint).collect();
        let router = OfflineReplayRouter::new_with_selector_seed(
            args,
            router_config,
            prefill_load_estimator,
            &router_hints,
            selector_seed,
        )?;
        Ok(Self { router })
    }

    /// Runs `classifier` and its dispatch gate in this placement, with `fields` attached to
    /// requests. Built through [`OfflineReplayRouter::new_with_classifier`].
    #[allow(clippy::too_many_arguments)]
    pub(in crate::replay) fn new_with_classifier(
        args: &MockerConfig,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        topology: &[WorkerTopology],
        selector_seed: Option<u64>,
        classifier: Option<&crate::replay::ReplayClassifierPlugin>,
        fields: Option<Arc<ReplayRequestAnnotations>>,
        worker_selection: Option<&crate::replay::ReplayWorkerSelectionPlugin>,
    ) -> Result<Self> {
        let router_hints: Vec<bool> = topology.iter().map(|worker| worker.router_hint).collect();
        let mut router = OfflineReplayRouter::new_with_classifier(
            args,
            router_config,
            prefill_load_estimator,
            &router_hints,
            selector_seed,
            classifier,
            worker_selection,
        )?;
        router.request_fields = fields;
        Ok(Self { router })
    }

    pub(in crate::replay) fn with_kv_event_lag_ms(mut self, lag_ms: f64) -> Result<Self> {
        self.router.set_kv_event_lag_ms(lag_ms)?;
        Ok(self)
    }

    fn with_conversation_affinity(
        mut self,
        config: Option<ConversationAffinityConfig>,
    ) -> Result<Self> {
        if let Some(config) = config {
            config.validate()?;
            self.router.affinity = Some((
                config,
                SessionAffinity::with_manual_clock(
                    SessionAffinityConfig::new(Duration::from_secs_f64(config.ttl_seconds)),
                    self.router.decay_time_epoch,
                )?,
            ));
        }
        Ok(self)
    }

    /// Runtime parameter handle for this placement's worker selector.
    pub(in crate::replay) fn selector_control(&self) -> DefaultWorkerSelectorControl {
        self.router.selector.runtime_control()
    }

    fn placement(&self, admission: WorkerAdmission) -> Placement {
        Placement {
            request_id: admission.uuid,
            scheduler_id: admission.worker_idx,
            placement_replica_id: None,
            kv_hints: self.router.kv_hints(&admission),
            reported_overlap_tokens: admission.overlap_blocks as usize
                * self.router.block_size as usize,
            cache_sample: Some(PlacementCacheSample {
                overlap_blocks: admission.overlap_blocks,
                best_available_overlap_blocks: admission.best_available_overlap_blocks,
                isl_blocks: admission.isl_blocks,
            }),
        }
    }

    fn placements(&self, admissions: Vec<WorkerAdmission>) -> Vec<Placement> {
        admissions
            .into_iter()
            .map(|admission| self.placement(admission))
            .collect()
    }
}

trait PlacementRequestView {
    fn metadata(&self) -> &DirectRequest;
    fn input_length(&self) -> usize;
    fn prompt_tokens_for_placement(&self) -> Result<Cow<'_, [u32]>>;
}

impl PlacementRequestView for DirectRequest {
    fn metadata(&self) -> &DirectRequest {
        self
    }

    fn input_length(&self) -> usize {
        self.tokens.len()
    }

    fn prompt_tokens_for_placement(&self) -> Result<Cow<'_, [u32]>> {
        if !self.prompt_tokens_are_placement_safe() {
            return Err(anyhow!(
                "Dynamo KV Router placement requires authored prompt token IDs or replay hashes; length-only execution tokens are not valid KV identities"
            ));
        }
        Ok(Cow::Borrowed(&self.tokens))
    }
}

impl PlacementRequestView for ReplayRequestPayload {
    fn metadata(&self) -> &DirectRequest {
        self.metadata()
    }

    fn input_length(&self) -> usize {
        self.input_length()
    }

    fn prompt_tokens_for_placement(&self) -> Result<Cow<'_, [u32]>> {
        if !self.metadata().prompt_tokens_are_placement_safe() {
            return Err(anyhow!(
                "Dynamo KV Router placement requires authored prompt token IDs or replay hashes; length-only execution tokens are not valid KV identities"
            ));
        }
        match self.materialized_tokens() {
            Some(tokens) => Ok(Cow::Borrowed(tokens)),
            None => Ok(Cow::Owned(ReplayRequestPayload::prompt_tokens(self))),
        }
    }
}

impl<Request: PlacementRequestView> PlacementPolicy<Request> for KvRouterPlacement {
    type Metadata = KvReplayMetadata;
    type Observation = RouterEventBatch;

    fn place(
        &mut self,
        request: &Request,
        metadata: Self::Metadata,
        session_id: Option<String>,
        now_ms: f64,
    ) -> Result<PlacementEffects> {
        let request_metadata = request.metadata();
        let effective_max_output_tokens = request_metadata.effective_max_output_tokens();
        let max_output_tokens = metadata
            .max_output_tokens_override()
            .map_or(effective_max_output_tokens, |override_tokens| {
                effective_max_output_tokens.min(override_tokens)
            });
        let request_id = request_metadata
            .uuid
            .ok_or_else(|| anyhow!("KV placement requires a request UUID"))?;
        let admissions = self
            .router
            .on_compact_request_arrival_for_session(
                request,
                max_output_tokens,
                metadata.into_hashes(),
                session_id,
                now_ms,
            )?
            .admissions;
        let mut decision = PlacementDecision::Queued;
        let mut released = Vec::with_capacity(admissions.len().saturating_sub(1));
        for admission in admissions {
            let placement = self.placement(admission);
            if placement.request_id == request_id {
                decision = PlacementDecision::Immediate(placement);
            } else {
                released.push(placement);
            }
        }
        Ok(PlacementEffects { decision, released })
    }

    fn take_kv_actions(&mut self) -> Vec<KvActionMessage> {
        self.router.take_kv_actions()
    }

    fn kv_action_results(&mut self, results: Vec<KvActionResult>, _now_ms: f64) -> Result<()> {
        self.router.kv_action_results(results);
        Ok(())
    }

    fn observe(&mut self, observation: RouterEventBatch, now_ms: f64) -> Result<Vec<Placement>> {
        let effects = self.router.on_kv_events_at(observation.0, now_ms)?;
        Ok(self.placements(effects.admissions))
    }

    fn dispatch_committed(&mut self, request_id: Uuid, now_ms: f64) -> Result<()> {
        self.router.set_classifier_clock(now_ms);
        // The adaptive rule's preempt group reaches the engines (class preemption on or off).
        aisimulate_core::engine::set_class_preemption_enabled(
            dynamo_kv_router::plugins::adaptive_switches::class_preemption_enabled(),
        );
        // f49o: the session scheduler's batch-yield signal reaches the engines (level 0: off).
        {
            let (level, rank, finish, aside) =
                dynamo_kv_router::plugins::adaptive_switches::batch_yield();
            aisimulate_core::engine::set_batch_yield(level, rank, finish, aside);
        }
        // f47k: the engine's per-request KVCR reuse field (`DYNAMO_REPLAY_KVCR_FIELD=1`).
        aisimulate_core::engine::set_kvcr_reuse_field(kvcr_field());
        if let Some(&(worker, _, _)) = self.router.dispatched_to.get(&request_id) {
            if let Some(cached_tokens) = self.router.placed_cached.remove(&request_id) {
                self.router.deliver(ClassifyEvent::Placed {
                    request_id: request_id.to_string(),
                    worker,
                    cached_tokens,
                });
            }
            self.router.deliver(ClassifyEvent::Sent {
                request_id: request_id.to_string(),
                worker,
            });
        }
        self.router.advance_affinity_clock(now_ms)?;
        let Some((_, table)) = &self.router.affinity else {
            return Ok(());
        };
        if let Some((hold, target)) = self.router.affinity_staged.remove(&request_id) {
            self.router
                .affinity_active
                .insert(request_id, table.commit(hold, target)?);
        }
        Ok(())
    }

    fn dispatch_aborted(&mut self, request_id: Uuid, now_ms: f64) -> Result<()> {
        self.router.set_classifier_clock(now_ms);
        if let Some((worker, _, _)) = self.router.dispatched_to.remove(&request_id) {
            self.router.deliver(ClassifyEvent::Aborted {
                request_id: request_id.to_string(),
                worker: Some(worker),
                error: None,
            });
        }
        self.router.advance_affinity_clock(now_ms)?;
        self.router.affinity_staged.remove(&request_id);
        self.router.affinity_active.remove(&request_id);
        self.router.kv_actions.admitted.remove(&request_id);
        self.router.dequeue.kept.remove(&request_id);
        if let Some(fit) = self.router.engine_fit.as_mut() {
            fit.gone(request_id);
        }
        if let Some(iso) = self.router.prefill_iso.as_mut() {
            iso.gone(request_id);
        }
        if let Some(ranked) = self.router.busy_ranked.as_mut() {
            ranked.release(request_id);
        }
        if let Some(active) = self.router.class_active.as_mut() {
            active.release(request_id);
        }
        self.router.needs_drain |= self
            .router
            .slots
            .free(&request_id.to_string(), self.router.decay_now(now_ms))?
            .is_applied();
        Ok(())
    }

    fn retains_dispatched(&self) -> bool {
        self.router.dequeue_config().is_some()
    }

    fn wants_engine_gauges(&self) -> bool {
        // f49b: the env switch alone (AISimulate asks once, before the classifier is installed).
        self.router.engine_fit.is_some() || engine_gauges_to_classifier()
    }

    fn engine_gauges(
        &mut self,
        gauges: &[aisimulate_core::replay::ReplaySchedulerMetricsSnapshot],
        now_ms: f64,
    ) {
        // ks2b (`DYNAMO_REPLAY_ENGINE_GAUGES=1`): each rank's gauges to the classifier, when they
        // change, for its forecast (`forecast.class_backlog`).
        if engine_gauges_to_classifier() && self.router.classifier.is_some() {
            self.router.set_classifier_clock(now_ms);
            gauge_debug::first_snapshot(gauges.iter().map(|g| (g.worker_id as u64, g.dp_rank)));
            for g in gauges {
                let worker = WorkerWithDpRank::new(g.worker_id as WorkerId, g.dp_rank);
                let row = [
                    g.running_requests,
                    g.waiting_requests,
                    g.active_blocks,
                    g.total_blocks,
                    g.waiting_tokens,
                ];
                if self.router.gauges_sent.insert(worker, row) != Some(row) {
                    gauge_debug::forwarded(g.worker_id as u64, g.dp_rank);
                    self.router.deliver(ClassifyEvent::EngineGauge {
                        worker,
                        running_requests: g.running_requests,
                        waiting_requests: g.waiting_requests,
                        active_blocks: g.active_blocks,
                        total_blocks: g.total_blocks,
                        waiting_tokens: g.waiting_tokens,
                    });
                }
            }
        }
        let Some(fit) = self.router.engine_fit.as_mut() else {
            return;
        };
        // Room freed on a worker can release a turn `hold` keeps queued.
        let freed = fit.set_gauges(gauges);
        let timed = fit.config.hold_max_s.is_some();
        if fit.config.hold && (freed || timed) && self.router.pending_count() > 0 {
            self.router.needs_drain = true;
        }
    }

    fn requeued(&mut self, request_id: Uuid, now_ms: f64) -> Result<Vec<Placement>> {
        self.router.set_classifier_clock(now_ms);
        let admissions = self.router.requeue(request_id, now_ms)?;
        Ok(self.placements(admissions))
    }

    fn advance_clock(&mut self, now_ms: f64) -> Result<Vec<Placement>> {
        self.router.write_workset(now_ms);
        self.router.kv_actions.fire_prefetches(now_ms);
        self.router.dequeue_check(now_ms)?;
        if self.router.classifier.is_some() {
            self.router.set_classifier_clock(now_ms);
            let mut admissions = self.router.release_due_retries(now_ms)?;
            let woken = self
                .router
                .classifier
                .as_ref()
                .is_some_and(|classifier| classifier.take_recheck())
                || self
                    .router
                    .next_wakeup_ms()
                    .is_some_and(|wake_ms| wake_ms <= now_ms);
            if woken || self.router.needs_drain {
                self.router.apply_due_kv_events(now_ms)?;
                admissions.extend(self.router.drain_pending(self.router.decay_now(now_ms))?);
            }
            if self.router.affinity.is_none() {
                return Ok(self.placements(admissions));
            }
            let mut placements = self.placements(admissions);
            placements.extend(self.advance_affinity(now_ms)?);
            return Ok(placements);
        }
        // Without a classifier, only a queue-limit rejection's client retry waits here.
        let admissions = self.router.release_due_retries(now_ms)?;
        let mut placements = self.placements(admissions);
        placements.extend(self.advance_affinity(now_ms)?);
        Ok(placements)
    }

    fn next_wakeup_ms(&self) -> Option<f64> {
        self.router.next_wakeup_ms()
    }

    fn cancel_pending(&mut self, request_id: Uuid) -> bool {
        let canceled = self.router.cancel_pending(request_id);
        if canceled {
            self.router.deliver(ClassifyEvent::Aborted {
                request_id: request_id.to_string(),
                worker: None,
                error: None,
            });
        }
        canceled
    }

    fn request_terminal(&mut self, request_id: Uuid, now_ms: f64) -> Result<Vec<Placement>> {
        self.router.write_workset(now_ms);
        if let Some(log) = self.router.workset.as_mut() {
            log.on_terminal(request_id);
        }
        self.router.set_classifier_clock(now_ms);
        if let Some(fit) = self.router.engine_fit.as_mut() {
            fit.gone(request_id);
        }
        if let Some(iso) = self.router.prefill_iso.as_mut() {
            iso.gone(request_id);
        }
        if let Some((worker, context_tokens, output_tokens)) =
            self.router.dispatched_to.remove(&request_id)
        {
            self.router.deliver(ClassifyEvent::Completed {
                request_id: request_id.to_string(),
                worker,
                context_tokens: Some(context_tokens),
                output_tokens: Some(output_tokens),
            });
        }
        let effects = self.router.on_request_completed(request_id, now_ms)?;
        Ok(self.placements(effects.admissions))
    }

    fn prefill_completed(&mut self, request_id: Uuid, now_ms: f64) -> Result<Vec<Placement>> {
        self.router.set_classifier_clock(now_ms);
        if let Some(&(worker, _, _)) = self.router.dispatched_to.get(&request_id) {
            self.router.deliver(ClassifyEvent::Responding {
                request_id: request_id.to_string(),
                worker,
            });
        }
        let effects = self.router.on_prefill_completed(request_id, now_ms)?;
        Ok(self.placements(effects.admissions))
    }

    fn pending_count(&self) -> usize {
        self.router.pending_count()
    }

    fn worker_ready(&mut self, worker: WorkerTopology, _now_ms: f64) -> Result<Vec<Placement>> {
        self.router
            .add_worker(worker.worker_id, worker.router_hint)?;
        Ok(Vec::new())
    }

    fn worker_draining(&mut self, worker: WorkerTopology, _now_ms: f64) -> Result<Vec<Placement>> {
        self.router.remove_worker(worker.worker_id)?;
        Ok(Vec::new())
    }

    fn worker_removed(&mut self, worker: WorkerTopology, _now_ms: f64) -> Result<Vec<Placement>> {
        self.router.finalize_worker_removal(worker.worker_id)?;
        Ok(Vec::new())
    }

    fn topology_settled(&mut self, now_ms: f64) -> Result<Vec<Placement>> {
        let effects = self.router.on_topology_changed(now_ms)?;
        Ok(self.placements(effects.admissions))
    }
}

impl KvRouterPlacement {
    /// The affinity part of `advance_clock`, unchanged from before the replay gate.
    fn advance_affinity(&mut self, now_ms: f64) -> Result<Vec<Placement>> {
        if self.router.affinity.is_none() && !self.router.needs_drain {
            return Ok(Vec::new());
        }
        self.router.apply_due_kv_events(now_ms)?;
        self.router.advance_affinity_clock(now_ms)?;
        let waiting = std::mem::take(&mut self.router.affinity_waiting);
        let mut admissions = Vec::new();
        for mut request in waiting {
            if self.router.acquire_affinity(&mut request)? {
                if let Some(hashes) = &request.block_hashes {
                    let tiered = self.router.indexer.find_matches_for_hashes(hashes.clone());
                    request.overlap = OverlapAnalysis::new(
                        &CacheHitWeights::from(&self.router.config),
                        self.router.block_size,
                        &tiered,
                    )
                    .signals();
                    request.kv_transfer_candidates =
                        transfer_candidates(tiered, self.router.peer_hbm());
                }
                let effects = self.router.enqueue_or_admit(request, now_ms)?;
                admissions.extend(effects.admissions);
            } else {
                self.router.affinity_waiting.push_back(request);
            }
        }
        if self.router.needs_drain {
            admissions.extend(self.router.drain_pending(self.router.decay_now(now_ms))?);
        }
        Ok(self.placements(admissions))
    }
}

/// The KV transfer candidates for a request: the host tier's, else the device tier's, as the
/// router's indexer gives them; with `peer_hbm` (KVCR from a peer's HBM) the two merged, each
/// holder at its deepest prefix in either tier.
fn transfer_candidates(tiered: TieredMatchDetails, peer_hbm: bool) -> Option<KvTransferCandidates> {
    if !peer_hbm {
        return tiered.into_kv_transfer_candidates();
    }
    let device = tiered.device.kv_transfer_candidates.clone();
    let host = tiered.into_kv_transfer_candidates();
    merge_candidates(host, device)
}

fn merge_candidates(
    host: Option<KvTransferCandidates>,
    device: Option<KvTransferCandidates>,
) -> Option<KvTransferCandidates> {
    let (mut merged, other) = match (host, device) {
        (Some(h), Some(d)) => (h, d),
        (h, d) => return h.or(d),
    };
    if other.block_hashes.len() > merged.block_hashes.len() {
        merged.block_hashes = other.block_hashes;
    }
    for (owner, blocks) in other.owner_prefix_blocks {
        match merged
            .owner_prefix_blocks
            .iter_mut()
            .find(|(existing, _)| *existing == owner)
        {
            Some((_, have)) => *have = (*have).max(blocks),
            None => merged.owner_prefix_blocks.push((owner, blocks)),
        }
    }
    if merged.routing_snapshot.is_none() {
        merged.routing_snapshot = other.routing_snapshot;
    }
    Some(merged)
}

/// One row per worker rank, in worker and rank order, as the live router's classifier context
/// lists them.
fn worker_infos(workers: &HashMap<WorkerId, ReplayWorkerConfig>, dp_size: u32) -> Vec<WorkerInfo> {
    let mut ids: Vec<WorkerId> = workers.keys().copied().collect();
    ids.sort_unstable();
    ids.into_iter()
        .flat_map(|worker_id| {
            let capacity = WorkerCapacity::of(&workers[&worker_id]);
            (0..dp_size.max(1))
                .map(move |rank| WorkerInfo::new(WorkerWithDpRank::new(worker_id, rank), capacity))
        })
        .collect()
}

/// Path A: one booking as the hint policies see it (the `HintContext` of the hint-policy
/// stack, #17359, narrowed to what these policies read).
struct HintBooking<'a> {
    admission: &'a WorkerAdmission,
    /// `kv_actions.fetch_no_retain`.
    fetch_no_retain: bool,
}

/// Path A: a per-booking hint policy. Policies run in order and add their actions to one
/// envelope per booking, as #17359's `compose_kv_hint` does; `kv_fetch` (#17360's builtin)
/// runs first. Following that pattern, not its code: replay builds the envelope it delivers
/// to AISimulate's engine directly.
trait ReplayHintPolicy {
    fn build_hints(&self, booking: &HintBooking<'_>, envelope: &mut KvHints);
}

/// `kv.fetch@1.0` from the KVCR source the worker selection committed.
struct KvFetchHintPolicy;

impl ReplayHintPolicy for KvFetchHintPolicy {
    fn build_hints(&self, booking: &HintBooking<'_>, envelope: &mut KvHints) {
        envelope.fetch = booking
            .admission
            .kv_fetch_source
            .map(|source_scheduler_id| KvFetchAction {
                source_scheduler_id,
                retain_copy: !booking.fetch_no_retain,
            });
    }
}

/// The session scheduler's `session-kv` hint policy: `kv.retain@1.0` with the TTL the
/// policy file gives the request's class, for a request expected back within it.
struct SessionKvHintPolicy;

impl ReplayHintPolicy for SessionKvHintPolicy {
    fn build_hints(&self, booking: &HintBooking<'_>, envelope: &mut KvHints) {
        let admission = booking.admission;
        envelope.retain = (admission.retain_ttl_ms.is_some() || admission.retain_value.is_some())
            .then_some(KvRetainAction {
                ttl_ms: admission.retain_ttl_ms,
                value: admission.retain_value,
            });
    }
}

const HINT_POLICIES: &[&dyn ReplayHintPolicy] = &[&KvFetchHintPolicy, &SessionKvHintPolicy];

/// How long a move waits for the target's KV events before it gives up (frees nothing more).
const MOVE_CONFIRM_TIMEOUT_MS: f64 = 60_000.0;

/// A move in flight: `kv.fetch` booked to `target` from `source`; the source's copy of each
/// block is released only after the target's own KV events show it stored.
#[derive(Debug)]
struct PendingMove {
    source_scheduler_id: usize,
    target: WorkerWithDpRank,
    /// The blocks the fetch copies, by sequence hash (the source's prefix of the request).
    expected: HashSet<u64>,
    deadline_ms: f64,
}

/// Path B (out of band): the router side of the `kv_actions` channel. `send` only queues;
/// replay takes the queue after each callback and returns one result per action.
#[derive(Debug, Default)]
struct KvActionSender {
    queue: Vec<KvActionMessage>,
    next_message_id: u64,
    next_action_id: u64,
    moves: Vec<PendingMove>,
    stats: KvActionStats,
    /// Retention-by-value guard: each live session's context, their sum, and the mode sent.
    session_tokens: HashMap<u64, u64>,
    demand_tokens: u64,
    retain_disengaged: bool,
    /// `kv.invalidate`: each live stream's previous request's blocks.
    streams: HashMap<String, Vec<u64>>,
    /// `kv.demote`: admitted requests' rank and blocks, until they finish.
    admitted: HashMap<Uuid, (usize, WorkerWithDpRank, KvBlockPlan)>,
    /// `kv.demote`: G1 blocks each rank holds, from its device-tier KV events.
    g1_resident: HashMap<WorkerWithDpRank, u64>,
    /// `kv.prefetch` timers: (due, rank, stream, blocks).
    prefetches: Vec<(f64, usize, KvBlockPlan)>,
    /// `kv.demote` with `delay_s`: (due, rank, stream, blocks).
    demotes: Vec<(f64, usize, String, Vec<u64>)>,
    /// `release_source`: the rank each live stream's last request was placed on.
    stream_home: HashMap<String, usize>,
    /// `release_unshared`: each live stream's latest request's blocks, and for each block how
    /// many of those streams hold it.
    stream_blocks: HashMap<String, Vec<u64>>,
    block_streams: HashMap<u64, u32>,
    /// `move_under_pressure_s`: blocks each source rank was asked to release and has not yet
    /// removed, and when each rank last removed a G2 block of its own accord (an eviction).
    releasing: HashMap<usize, HashSet<u64>>,
    last_g2_eviction_ms: HashMap<usize, f64>,
    track_releases: bool,
    /// `free_at_session_end`: each live stream's last request's blocks, and how many live
    /// streams' last requests hold each block.
    free_streams: HashMap<String, Vec<u64>>,
    free_refs: HashMap<u64, u32>,
}

impl KvActionSender {
    /// `release_unshared`: `stream` now holds `blocks` (`None`: it ended).
    fn note_stream_blocks(&mut self, stream: &str, blocks: Option<&[u64]>) {
        let old = match blocks {
            Some(b) => self.stream_blocks.insert(stream.to_owned(), b.to_vec()),
            None => self.stream_blocks.remove(stream),
        };
        for hash in old.into_iter().flatten() {
            if let Some(n) = self.block_streams.get_mut(&hash) {
                *n -= 1;
                if *n == 0 {
                    self.block_streams.remove(&hash);
                }
            }
        }
        for hash in blocks.into_iter().flatten() {
            *self.block_streams.entry(*hash).or_default() += 1;
        }
    }

    /// Whether some stream other than one holding it also holds `hash`.
    fn shared(&self, hash: u64) -> bool {
        self.block_streams.get(&hash).is_some_and(|n| *n > 1)
    }

    /// `move_under_pressure_s`: a host-tier (G2) removal of `hashes` on `scheduler_id` at
    /// `now_ms`. A block this router asked the rank to release is a release; any other is an
    /// eviction and marks the rank as under pressure.
    fn note_g2_removed(&mut self, scheduler_id: usize, hashes: &[u64], now_ms: f64) {
        let releasing = self.releasing.entry(scheduler_id).or_default();
        let mut evicted = false;
        for hash in hashes {
            if !releasing.remove(hash) {
                evicted = true;
            }
        }
        if evicted {
            self.last_g2_eviction_ms.insert(scheduler_id, now_ms);
        }
    }

    /// Whether `scheduler_id` evicted from G2 within `window_s` before `now_ms`.
    fn under_pressure(&self, scheduler_id: usize, now_ms: f64, window_s: f64) -> bool {
        self.last_g2_eviction_ms
            .get(&scheduler_id)
            .is_some_and(|at| now_ms - at <= window_s * 1000.0)
    }
}

/// Path B outcomes, logged when replay ends.
#[derive(Debug, Default, Clone)]
pub(crate) struct KvActionStats {
    pub(crate) moves_started: u64,
    pub(crate) moves_timed_out: u64,
    /// Fetches from the stream's own previous rank (moved) and from another stream's copy
    /// (copied, the source keeps it).
    pub(crate) moves_own_stream: u64,
    pub(crate) copies_other_stream: u64,
    /// Blocks the target's KV events confirmed for a move.
    pub(crate) confirmed_blocks: u64,
    pub(crate) release_sent_blocks: u64,
    pub(crate) results_done: u64,
    pub(crate) results_partial: u64,
    pub(crate) results_failed: u64,
    pub(crate) results_unsupported: u64,
    /// Blocks the ranks report affected, over every action.
    pub(crate) released_blocks: u64,
    pub(crate) retain_mode_changes: u64,
    pub(crate) invalidate_sent_blocks: u64,
    pub(crate) demotes_sent: u64,
    pub(crate) prefetches_sent: u64,
    pub(crate) prefetches_cancelled: u64,
    /// `release_unshared`: fetched blocks a move kept on its source because another stream holds them.
    pub(crate) kept_shared_blocks: u64,
    /// `move_under_pressure_s`: own-stream fetches copied because their source was not under G2 pressure.
    pub(crate) moves_skipped_no_pressure: u64,
    /// `free_at_session_end`: streams ended, blocks freed, and blocks kept because another live
    /// stream holds them.
    pub(crate) session_end_frees: u64,
    pub(crate) session_end_freed_blocks: u64,
    pub(crate) session_end_shared_blocks: u64,
}

impl Drop for OfflineReplayRouter {
    fn drop(&mut self) {
        if self.queue_rejections > 0 {
            tracing::info!(
                queue_rejections = self.queue_rejections,
                "replay_queue_reject_stats"
            );
        }
    }
}

/// `DYNAMO_REPLAY_QUEUE_REJECT_RETRY=1`: a router queue-limit rejection goes back to the replay
/// client, which retries after its backoff, instead of ending the run. Off by default.
pub(crate) const REPLAY_QUEUE_REJECT_RETRY_ENV: &str = "DYNAMO_REPLAY_QUEUE_REJECT_RETRY";

/// `DYNAMO_REPLAY_REFUSAL_LOG=<path>`: write the client refusal log there (off by default).
pub(crate) const REPLAY_REFUSAL_LOG_ENV: &str = "DYNAMO_REPLAY_REFUSAL_LOG";

fn refusal_log() -> Option<std::io::BufWriter<std::fs::File>> {
    let path = std::env::var_os(REPLAY_REFUSAL_LOG_ENV)?;
    match std::fs::File::create(&path) {
        Ok(file) => Some(std::io::BufWriter::new(file)),
        Err(error) => {
            tracing::warn!(?path, %error, "cannot open the replay refusal log");
            None
        }
    }
}

fn queue_reject_retry() -> bool {
    std::env::var_os(REPLAY_QUEUE_REJECT_RETRY_ENV).is_some_and(|v| v == "1")
}

/// `DYNAMO_REPLAY_BUSY_RANKED=1` (off by default): a class's busy test counts only the prefill
/// booked by its own and higher classes. The router books a request's uncached prompt on its
/// worker until the first token; with engines that schedule by class and pause lower-class prefill
/// to protect a higher class's ITL (`class_scheduling.protect_max_rank`), a paused batch prefill
/// stays booked for as long as it is paused, and the busy test held priority and standard turns
/// at the router behind work the engine would not run before them.
pub(crate) const REPLAY_BUSY_RANKED_ENV: &str = "DYNAMO_REPLAY_BUSY_RANKED";

fn busy_ranked() -> Option<RankedBooking> {
    std::env::var_os(REPLAY_BUSY_RANKED_ENV)
        .is_some_and(|v| v == "1")
        .then(RankedBooking::default)
}

/// Prefill tokens booked per worker and class (class index, highest class first), for
/// `DYNAMO_REPLAY_BUSY_RANKED`.
#[derive(Debug, Default)]
pub(crate) struct RankedBooking {
    by_request: FxHashMap<Uuid, (WorkerWithDpRank, usize, usize)>,
    sums: FxHashMap<(WorkerWithDpRank, usize), usize>,
}

impl RankedBooking {
    fn book(&mut self, uuid: Uuid, worker: WorkerWithDpRank, class_index: usize, tokens: usize) {
        self.release(uuid);
        if tokens == 0 {
            return;
        }
        self.by_request.insert(uuid, (worker, class_index, tokens));
        *self.sums.entry((worker, class_index)).or_default() += tokens;
    }

    fn release(&mut self, uuid: Uuid) {
        if let Some((worker, class_index, tokens)) = self.by_request.remove(&uuid)
            && let Some(sum) = self.sums.get_mut(&(worker, class_index))
        {
            *sum = sum.saturating_sub(tokens);
            if *sum == 0 {
                self.sums.remove(&(worker, class_index));
            }
        }
    }

    /// `loads` with the prefill lower classes than `class_index` booked taken out.
    fn for_class(
        &self,
        loads: &FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
        class_index: usize,
    ) -> FxHashMap<WorkerWithDpRank, WorkerLoadProjection> {
        let mut out = loads.clone();
        for (&(worker, booked_class), &tokens) in &self.sums {
            if booked_class > class_index
                && let Some(load) = out.get_mut(&worker)
            {
                load.active_prefill_tokens = load.active_prefill_tokens.saturating_sub(tokens);
            }
        }
        out
    }
}

/// Logs what path B did when the replay ends (the counts are not returned to Python): grep
/// the run's log for `replay_kv_action_stats`. Retain counts are in the run's KVCR stats.
impl Drop for KvActionSender {
    fn drop(&mut self) {
        let s = &self.stats;
        if s.moves_started == 0
            && s.session_end_frees == 0
            && s.results_done + s.results_partial + s.results_failed + s.results_unsupported == 0
        {
            return;
        }
        tracing::info!(
            moves_started = s.moves_started,
            moves_timed_out = s.moves_timed_out,
            moves_own_stream = s.moves_own_stream,
            copies_other_stream = s.copies_other_stream,
            confirmed_blocks = s.confirmed_blocks,
            release_sent_blocks = s.release_sent_blocks,
            results_done = s.results_done,
            results_partial = s.results_partial,
            results_failed = s.results_failed,
            results_unsupported = s.results_unsupported,
            released_blocks = s.released_blocks,
            retain_mode_changes = s.retain_mode_changes,
            invalidate_sent_blocks = s.invalidate_sent_blocks,
            demotes_sent = s.demotes_sent,
            prefetches_sent = s.prefetches_sent,
            prefetches_cancelled = s.prefetches_cancelled,
            kept_shared_blocks = s.kept_shared_blocks,
            moves_skipped_no_pressure = s.moves_skipped_no_pressure,
            session_end_frees = s.session_end_frees,
            session_end_freed_blocks = s.session_end_freed_blocks,
            session_end_shared_blocks = s.session_end_shared_blocks,
            "replay_kv_action_stats"
        );
    }
}

impl KvActionSender {
    fn send(&mut self, scheduler_id: usize, actions: Vec<KvAction>) {
        let actions = actions
            .into_iter()
            .map(|action| {
                self.next_action_id += 1;
                KvActionRequest {
                    action_id: self.next_action_id,
                    action,
                }
            })
            .collect();
        self.next_message_id += 1;
        self.queue.push(KvActionMessage {
            scheduler_id,
            message_id: self.next_message_id,
            actions,
        });
    }

    /// Confirm moves from a target's stored blocks: release on each source exactly the
    /// blocks its target now holds. Expired moves are dropped and release nothing more.
    fn confirm(&mut self, events: &[RouterEvent], now_ms: f64) {
        if self.moves.is_empty() {
            return;
        }
        let mut stored: HashMap<WorkerWithDpRank, Vec<u64>> = HashMap::new();
        for event in events {
            if let KvCacheEventData::Stored(store) = &event.event.data {
                let target = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
                stored
                    .entry(target)
                    .or_default()
                    .extend(store.blocks.iter().map(|block| block.block_hash.0));
            }
        }
        let mut releases: Vec<(usize, Vec<u64>)> = Vec::new();
        for pending in &mut self.moves {
            let Some(hashes) = stored.get(&pending.target) else {
                continue;
            };
            let confirmed: Vec<u64> = hashes
                .iter()
                .copied()
                .filter(|hash| pending.expected.remove(hash))
                .collect();
            if !confirmed.is_empty() {
                self.stats.confirmed_blocks += confirmed.len() as u64;
                releases.push((pending.source_scheduler_id, confirmed));
            }
        }
        // A move whose every block is confirmed is complete.
        self.moves.retain(|pending| !pending.expected.is_empty());
        self.expire(now_ms);
        for (source, hashes) in releases {
            self.stats.release_sent_blocks += hashes.len() as u64;
            if self.track_releases {
                self.releasing
                    .entry(source)
                    .or_default()
                    .extend(hashes.iter().copied());
            }
            self.send(
                source,
                vec![KvAction::ReleaseSource {
                    block_hashes: hashes,
                }],
            );
        }
    }

    fn expire(&mut self, now_ms: f64) {
        let before = self.moves.len();
        self.moves.retain(|pending| pending.deadline_ms > now_ms);
        self.stats.moves_timed_out += (before - self.moves.len()) as u64;
    }

    fn results(&mut self, results: Vec<KvActionResult>) {
        for result in results {
            match result.status {
                KvActionStatus::Done => self.stats.results_done += 1,
                KvActionStatus::Partial => self.stats.results_partial += 1,
                KvActionStatus::Failed => self.stats.results_failed += 1,
                KvActionStatus::Unsupported => self.stats.results_unsupported += 1,
            }
            self.stats.released_blocks += result.blocks.len() as u64;
        }
    }

    /// Track each rank's G1 residency from its device-tier KV events.
    fn count_g1(&mut self, events: &[RouterEvent]) {
        for event in events {
            if event.storage_tier != StorageTier::Device {
                continue;
            }
            let rank = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
            let resident = self.g1_resident.entry(rank).or_default();
            match &event.event.data {
                KvCacheEventData::Stored(store) => *resident += store.blocks.len() as u64,
                KvCacheEventData::Removed(remove) => {
                    *resident = resident.saturating_sub(remove.block_hashes.len() as u64)
                }
                KvCacheEventData::Cleared => *resident = 0,
            }
        }
    }

    fn next_prefetch_ms(&self) -> Option<f64> {
        self.prefetches
            .iter()
            .map(|(due_ms, _, _)| *due_ms)
            .chain(self.demotes.iter().map(|(due_ms, _, _, _)| *due_ms))
            .min_by(f64::total_cmp)
    }

    /// Send the delayed demotes, then the prefetches, due by `now_ms`.
    fn fire_prefetches(&mut self, now_ms: f64) {
        if !self.demotes.is_empty() {
            let (due, waiting): (Vec<_>, Vec<_>) = std::mem::take(&mut self.demotes)
                .into_iter()
                .partition(|(due_ms, _, _, _)| *due_ms <= now_ms);
            self.demotes = waiting;
            for (_, scheduler_id, _, block_hashes) in due {
                self.stats.demotes_sent += 1;
                self.send(scheduler_id, vec![KvAction::Demote { block_hashes }]);
            }
        }
        if self.prefetches.is_empty() {
            return;
        }
        let (due, waiting): (Vec<_>, Vec<_>) = std::mem::take(&mut self.prefetches)
            .into_iter()
            .partition(|(due_ms, _, _)| *due_ms <= now_ms);
        self.prefetches = waiting;
        for (_, scheduler_id, plan) in due {
            self.stats.prefetches_sent += 1;
            self.send(
                scheduler_id,
                vec![KvAction::Prefetch {
                    block_hashes: plan.sequence,
                    local_hashes: plan.local,
                }],
            );
        }
    }
}

/// `kv_actions.retain_value.value: rebuild_cost`: the engine's cost model for one request's class
/// (`None` with the default `session` value).
fn retain_cost(
    config: &dynamo_kv_router::scheduling::KvRetainValueConfig,
    class: Option<&str>,
) -> Option<aisimulate_core::replay::KvRetainCost> {
    if config.value != dynamo_kv_router::scheduling::RetainValueModel::RebuildCost {
        return None;
    }
    let ttft_s = class
        .and_then(|class| config.class_ttft_s.get(class))
        .copied()
        .unwrap_or(30.0);
    let int = |x: f64| x.max(0.0).round().min(f64::from(u32::MAX)) as u32;
    Some(aisimulate_core::replay::KvRetainCost {
        rebuild_us_per_block: int(config.rebuild_ms_per_block * 1000.0),
        ttft_target_ms: int(ttft_s * 1000.0),
        stall_milli: int(config.stall_weight * 1000.0),
        peers: config.peers,
        fetch_latency_ms: int(config.fetch_latency_ms),
        fetch_ns_per_block: int(config.fetch_ms_per_block * 1e6),
    })
}

/// The client's expected think time to the stream's next request (`x-dynamo-hint-next-think-s`).
fn next_think_ms(fields: Option<&ReplayRequestFields>) -> Option<u64> {
    fields
        .and_then(|fields| fields.agent_headers.get("x-dynamo-hint-next-think-s"))
        .and_then(|values| values.first())
        .and_then(|value| value.parse::<f64>().ok())
        .map(|seconds| (seconds.max(0.0) * 1000.0).round() as u64)
}

/// KV residency probe (diagnostics only): with `DYNAMO_REPLAY_KV_DUP=<path>`, the replay keeps every
/// rank's resident blocks per tier from the KV events the router sees and writes one JSON line every
/// `DYNAMO_REPLAY_KV_DUP_S` seconds of replay time (default 300): per tier (G1 device, G2 host, and
/// either) the blocks summed over ranks (`total`) and the distinct blocks over the fleet (`unique`);
/// `total - unique` are the duplicate copies. Off, nothing is kept and the replay is unchanged.
mod kv_dup_probe {
    use std::collections::{HashMap, HashSet};
    use std::io::Write;
    use std::sync::{Mutex, OnceLock};

    use dynamo_kv_events::event::{KvCacheEventData, RouterEvent};

    struct State {
        out: std::io::LineWriter<std::fs::File>,
        every_ms: f64,
        next_ms: f64,
        /// (worker, dp rank, device tier) -> resident sequence hashes.
        held: HashMap<(u64, u32, bool), HashSet<u64>>,
    }

    static STATE: OnceLock<Option<Mutex<State>>> = OnceLock::new();

    fn state() -> &'static Option<Mutex<State>> {
        STATE.get_or_init(|| {
            let path = std::env::var_os("DYNAMO_REPLAY_KV_DUP")?;
            let file = std::fs::File::create(path).ok()?;
            let every_s: f64 = std::env::var("DYNAMO_REPLAY_KV_DUP_S")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(300.0);
            Some(Mutex::new(State {
                out: std::io::LineWriter::new(file),
                every_ms: every_s * 1000.0,
                next_ms: 0.0,
                held: HashMap::new(),
            }))
        })
    }

    fn tier(st: &State, device: Option<bool>) -> (usize, usize) {
        let mut total = 0;
        let mut all: HashSet<u64> = HashSet::new();
        let mut per_rank: HashMap<(u64, u32), HashSet<u64>> = HashMap::new();
        for ((w, r, dev), set) in &st.held {
            if device.is_some_and(|d| d != *dev) {
                continue;
            }
            if device.is_none() {
                per_rank
                    .entry((*w, *r))
                    .or_default()
                    .extend(set.iter().copied());
            } else {
                total += set.len();
            }
            all.extend(set.iter().copied());
        }
        if device.is_none() {
            total = per_rank.values().map(HashSet::len).sum();
        }
        (total, all.len())
    }

    pub(super) fn record(events: &[RouterEvent], now_ms: f64) {
        let Some(m) = state() else {
            return;
        };
        let mut st = m.lock().unwrap();
        for event in events {
            let key = (
                event.worker_id,
                event.event.dp_rank,
                event.storage_tier.is_gpu(),
            );
            match &event.event.data {
                KvCacheEventData::Stored(store) => {
                    let set = st.held.entry(key).or_default();
                    set.extend(store.blocks.iter().map(|b| b.block_hash.0));
                }
                KvCacheEventData::Removed(remove) => {
                    if let Some(set) = st.held.get_mut(&key) {
                        for h in &remove.block_hashes {
                            set.remove(&h.0);
                        }
                    }
                }
                KvCacheEventData::Cleared => {
                    st.held.remove(&key);
                }
            }
        }
        if now_ms < st.next_ms {
            return;
        }
        st.next_ms = now_ms + st.every_ms;
        let (g1t, g1u) = tier(&st, Some(true));
        let (g2t, g2u) = tier(&st, Some(false));
        let (at, au) = tier(&st, None);
        let line = format!(
            r#"{{"t":{:.1},"g1":{{"total":{g1t},"unique":{g1u}}},"g2":{{"total":{g2t},"unique":{g2u}}},"any":{{"total":{at},"unique":{au}}}}}"#,
            now_ms / 1000.0
        );
        let _ = writeln!(st.out, "{line}");
    }
}

#[cfg(test)]
mod busy_ranked_tests {
    use super::*;

    #[test]
    fn a_class_busy_test_leaves_out_lower_classes_booked_prefill() {
        let (w0, w1) = (WorkerWithDpRank::new(0, 0), WorkerWithDpRank::new(1, 0));
        let mut loads = FxHashMap::default();
        for (worker, tokens) in [(w0, 30_000), (w1, 5_000)] {
            loads.insert(
                worker,
                WorkerLoadProjection {
                    active_prefill_tokens: tokens,
                    ..Default::default()
                },
            );
        }
        let mut ranked = RankedBooking::default();
        // On w0: a paused batch prefill (class 2) of 25,000 tokens and a standard one of 5,000.
        ranked.book(Uuid::from_u128(1), w0, 2, 25_000);
        ranked.book(Uuid::from_u128(2), w0, 1, 5_000);
        ranked.book(Uuid::from_u128(3), w1, 0, 5_000);
        // Priority sees neither lower booking on w0, standard leaves out only batch, batch sees all.
        assert_eq!(ranked.for_class(&loads, 0)[&w0].active_prefill_tokens, 0);
        assert_eq!(
            ranked.for_class(&loads, 1)[&w0].active_prefill_tokens,
            5_000
        );
        assert_eq!(
            ranked.for_class(&loads, 2)[&w0].active_prefill_tokens,
            30_000
        );
        assert_eq!(
            ranked.for_class(&loads, 0)[&w1].active_prefill_tokens,
            5_000
        );
        // Its first token releases it, once.
        ranked.release(Uuid::from_u128(1));
        ranked.release(Uuid::from_u128(1));
        assert_eq!(
            ranked.for_class(&loads, 1)[&w0].active_prefill_tokens,
            30_000
        );
        assert!(!ranked.sums.contains_key(&(w0, 2)));
    }
}

#[cfg(test)]
mod kv_action_tests {
    use super::*;
    use dynamo_kv_events::event::{KvCacheEvent, KvCacheStoreData, KvCacheStoredBlockData};
    use dynamo_kv_hashing::ExternalSequenceBlockHash;

    fn stored(worker_id: u64, hashes: &[u64]) -> RouterEvent {
        RouterEvent::new(
            worker_id,
            KvCacheEvent {
                event_id: 0,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: hashes
                        .iter()
                        .map(|&hash| KvCacheStoredBlockData {
                            block_hash: ExternalSequenceBlockHash(hash),
                            tokens_hash: LocalBlockHash(hash),
                            mm_extra_info: None,
                        })
                        .collect(),
                }),
                dp_rank: 0,
            },
        )
    }

    #[test]
    fn release_unshared_counts_the_streams_holding_each_block() {
        let mut sender = KvActionSender::default();
        // A parent and its subagent share blocks 1 and 2; the subagent adds 3.
        sender.note_stream_blocks("parent", Some(&[1, 2]));
        sender.note_stream_blocks("child", Some(&[1, 2, 3]));
        assert!(sender.shared(1) && sender.shared(2));
        assert!(!sender.shared(3));
        assert!(!sender.shared(9));
        // The parent's next request grows its context: 1 and 2 stay shared, 4 is its own.
        sender.note_stream_blocks("parent", Some(&[1, 2, 4]));
        assert!(sender.shared(2) && !sender.shared(4));
        // The subagent ends: nothing is shared any more and its blocks are forgotten.
        sender.note_stream_blocks("child", None);
        assert!(!sender.shared(1) && !sender.shared(2));
        assert!(!sender.block_streams.contains_key(&3));
        assert_eq!(sender.block_streams.get(&1), Some(&1));
    }

    #[test]
    fn move_pressure_counts_evictions_not_releases() {
        let mut sender = KvActionSender::default();
        // Rank 3 was asked to release 7 and 8: removing them is not pressure.
        sender.releasing.entry(3).or_default().extend([7, 8]);
        sender.note_g2_removed(3, &[7, 8], 1_000.0);
        assert!(!sender.under_pressure(3, 1_000.0, 300.0));
        // A removal it was not asked for is an eviction: pressure for the window after it.
        sender.note_g2_removed(3, &[9], 2_000.0);
        assert!(sender.under_pressure(3, 2_000.0 + 300_000.0, 300.0));
        assert!(!sender.under_pressure(3, 2_000.0 + 300_001.0, 300.0));
        // Another rank has evicted nothing.
        assert!(!sender.under_pressure(4, 2_000.0, 300.0));
    }

    fn sender_with_move(expected: &[u64], deadline_ms: f64) -> KvActionSender {
        let mut sender = KvActionSender::default();
        sender.moves.push(PendingMove {
            source_scheduler_id: 3,
            target: WorkerWithDpRank::new(1, 0),
            expected: expected.iter().copied().collect(),
            deadline_ms,
        });
        sender
    }

    fn released(sender: &mut KvActionSender) -> Vec<(usize, Vec<u64>)> {
        std::mem::take(&mut sender.queue)
            .into_iter()
            .flat_map(|message| {
                let scheduler_id = message.scheduler_id;
                message
                    .actions
                    .into_iter()
                    .filter_map(move |request| match request.action {
                        KvAction::ReleaseSource { mut block_hashes } => {
                            block_hashes.sort();
                            Some((scheduler_id, block_hashes))
                        }
                        _ => None,
                    })
            })
            .collect()
    }

    #[test]
    fn a_move_releases_on_the_source_only_what_the_target_confirmed() {
        let mut sender = sender_with_move(&[10, 11, 12], 1_000.0);
        // Another worker storing the same blocks confirms nothing.
        sender.confirm(&[stored(2, &[10, 11, 12])], 1.0);
        assert!(released(&mut sender).is_empty());
        // The target stores two of three (a partial copy): release exactly those, on the source.
        sender.confirm(&[stored(1, &[10, 12, 99])], 2.0);
        assert_eq!(released(&mut sender), vec![(3, vec![10, 12])]);
        assert_eq!(sender.moves.len(), 1);
        // The last block lands: release it and the move is done.
        sender.confirm(&[stored(1, &[11])], 3.0);
        assert_eq!(released(&mut sender), vec![(3, vec![11])]);
        assert!(sender.moves.is_empty());
    }

    #[test]
    fn a_move_that_times_out_releases_nothing_more() {
        let mut sender = sender_with_move(&[10, 11], 50.0);
        sender.confirm(&[stored(1, &[10])], 10.0);
        assert_eq!(released(&mut sender), vec![(3, vec![10])]);
        sender.expire(60.0);
        assert_eq!(sender.stats.moves_timed_out, 1);
        sender.confirm(&[stored(1, &[11])], 61.0);
        assert!(released(&mut sender).is_empty());
    }

    #[test]
    fn results_are_counted_by_status() {
        let mut sender = KvActionSender::default();
        sender.results(vec![
            KvActionResult {
                scheduler_id: 3,
                message_id: 1,
                action_id: 1,
                status: KvActionStatus::Partial,
                blocks: vec![10],
            },
            KvActionResult {
                scheduler_id: 3,
                message_id: 2,
                action_id: 2,
                status: KvActionStatus::Unsupported,
                blocks: vec![],
            },
        ]);
        assert_eq!(
            (
                sender.stats.results_partial,
                sender.stats.results_unsupported
            ),
            (1, 1)
        );
        assert_eq!(sender.stats.released_blocks, 1);
    }

    #[test]
    fn hint_policies_compose_fetch_and_retain_into_one_envelope() {
        let admission = WorkerAdmission {
            uuid: Uuid::nil(),
            worker_idx: 0,
            overlap_blocks: 0,
            best_available_overlap_blocks: 0,
            isl_blocks: 0,
            kv_fetch_source: Some(2),
            retain_ttl_ms: Some(3_600_000),
            retain_value: None,
            context_tokens: 0,
        };
        let booking = HintBooking {
            admission: &admission,
            fetch_no_retain: true,
        };
        let mut envelope = KvHints::default();
        for policy in HINT_POLICIES {
            policy.build_hints(&booking, &mut envelope);
        }
        assert_eq!(
            envelope.fetch.map(|fetch| fetch.source_scheduler_id),
            Some(2)
        );
        assert_eq!(envelope.fetch.map(|fetch| fetch.retain_copy), Some(false));
        assert_eq!(
            envelope.retain.and_then(|retain| retain.ttl_ms),
            Some(3_600_000)
        );
    }
}

impl OfflineReplayRouter {
    /// Path A: the envelope for an admission, composed by the hint policies in order.
    fn kv_hints(&self, admission: &WorkerAdmission) -> Option<KvHints> {
        let fetch_no_retain = self
            .config
            .kv_actions_config()
            .ok()
            .flatten()
            .is_some_and(|actions| actions.fetch_no_retain);
        let booking = HintBooking {
            admission,
            fetch_no_retain,
        };
        let mut envelope = KvHints::default();
        for policy in HINT_POLICIES {
            policy.build_hints(&booking, &mut envelope);
        }
        (envelope.fetch.is_some() || envelope.retain.is_some()).then_some(envelope)
    }

    /// `kv.retain` value for a request (policy file `kv_actions.retain_value`): its session,
    /// class weight, the client's think time to the next turn, and whether it ends the session.
    fn retain_value(
        &self,
        class: Option<&str>,
        fields: Option<&ReplayRequestFields>,
        session_id: Option<&str>,
    ) -> Result<Option<KvRetainValue>> {
        let Some(actions) = self.config.kv_actions_config()? else {
            return Ok(None);
        };
        let (Some(config), Some(session_id)) = (&actions.retain_value, session_id) else {
            return Ok(None);
        };
        let weight = class
            .and_then(|class| config.class_weight.get(class))
            .copied()
            .unwrap_or(1.0);
        let think_ms = next_think_ms(fields);
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(session_id, &mut hasher);
        let session = std::hash::Hasher::finish(&hasher);
        // Probe (DYNAMO_REPLAY_RETAIN_PROBE set): one line per request with a retention value, to
        // join the engine's eviction log (AISIM_VALUE_PROBE, keyed by `session`) with sessions,
        // requests and the client's hint.
        if std::env::var_os("DYNAMO_REPLAY_RETAIN_PROBE").is_some() {
            eprintln!(
                "retain_probe {{\"key\":{session},\"session\":{:?},\"request\":{:?},\"think_ms\":{},\"final\":{}}}",
                session_id,
                fields.map_or("", |fields| fields.request_id.as_str()),
                think_ms.map_or(-1, |ms| ms as i64),
                fields.and_then(|fields| fields.session_final) == Some(true)
            );
        }
        Ok(Some(KvRetainValue {
            session,
            weight_milli: (weight * 1000.0).round() as u32,
            think_ms,
            session_final: fields.and_then(|fields| fields.session_final) == Some(true),
            short_window_ms: config
                .short_window_s
                .map(|seconds| (seconds * 1000.0).round() as u64),
            cost: retain_cost(config, class),
        }))
    }

    /// The memory-pressure guard: track live sessions' KV and switch retention by value off
    /// above `release_above` of capacity and back on below `engage_below` (path B, every rank).
    fn note_session_demand(&mut self, admission: &WorkerAdmission) -> Result<()> {
        let Some(value) = admission.retain_value else {
            return Ok(());
        };
        let Some(guard) = self
            .config
            .kv_actions_config()?
            .and_then(|actions| actions.retain_value.as_ref())
            .and_then(|config| config.guard.clone())
        else {
            return Ok(());
        };
        let sender = &mut self.kv_actions;
        let old = if value.session_final {
            sender.session_tokens.remove(&value.session)
        } else {
            sender
                .session_tokens
                .insert(value.session, admission.context_tokens)
        };
        sender.demand_tokens = sender.demand_tokens - old.unwrap_or(0)
            + if value.session_final {
                0
            } else {
                admission.context_tokens
            };
        let demand = sender.demand_tokens as f64 / guard.capacity_tokens.max(1.0);
        let disengage = !sender.retain_disengaged && demand > guard.release_above;
        let engage = sender.retain_disengaged && demand < guard.engage_below;
        if disengage || engage {
            sender.retain_disengaged = disengage;
            sender.stats.retain_mode_changes += 1;
            let ranks: Vec<usize> = self
                .workers_with_configs
                .keys()
                .flat_map(|worker_id| {
                    (0..self.dp_size).map(move |dp_rank| WorkerWithDpRank::new(*worker_id, dp_rank))
                })
                .map(|worker| self.scheduler_id(worker))
                .collect::<Result<_>>()?;
            for scheduler_id in ranks {
                self.kv_actions
                    .send(scheduler_id, vec![KvAction::RetainMode { enabled: engage }]);
            }
        }
        Ok(())
    }

    fn release_source_enabled(&self) -> bool {
        self.config
            .kv_actions_config()
            .ok()
            .flatten()
            .is_some_and(|actions| actions.release_source)
    }

    fn demote_config(&self) -> Option<&KvDemoteConfig> {
        self.config
            .kv_actions_config()
            .ok()
            .flatten()
            .and_then(|actions| actions.demote.as_ref())
    }

    fn replicate_config(&self) -> Option<KvReplicateConfig> {
        self.config
            .kv_actions_config()
            .ok()
            .flatten()
            .and_then(|actions| actions.replicate.clone())
    }

    /// `kv.replicate`: copy a fan-out subagent's prefix into the G2 of the least-loaded workers
    /// that do not hold it, up to `replicas` holders, from its deepest holder; at most once per
    /// parent session within `window_s`.
    fn plan_replicas(
        &mut self,
        parent: String,
        sequence: Vec<u64>,
        local: Vec<u64>,
        overlap: &OverlapSignals,
        decay_now: Instant,
    ) -> Result<()> {
        let Some(config) = self.replicate_config() else {
            return Ok(());
        };
        let now_ms = decay_now
            .saturating_duration_since(self.decay_time_epoch)
            .as_secs_f64()
            * 1000.0;
        if self
            .replicated
            .get(&parent)
            .is_some_and(|at| now_ms - at < config.window_s * 1000.0)
        {
            return Ok(());
        }
        let Some((source, depth)) = overlap
            .effective_overlap_blocks
            .iter()
            .map(|(worker, blocks)| (*worker, blocks.floor() as usize))
            .filter(|(_, blocks)| *blocks > 0)
            .max_by(|a, b| {
                a.1.cmp(&b.1)
                    .then_with(|| (b.0.worker_id, b.0.dp_rank).cmp(&(a.0.worker_id, a.0.dp_rank)))
            })
        else {
            return Ok(());
        };
        let depth = depth.min(sequence.len()).min(local.len());
        if depth == 0 {
            return Ok(());
        }
        let holders: HashSet<WorkerWithDpRank> = overlap
            .effective_overlap_blocks
            .iter()
            .filter(|(_, blocks)| blocks.floor() as usize >= depth)
            .map(|(worker, _)| *worker)
            .collect();
        let want = config.replicas.saturating_sub(holders.len());
        if want == 0 {
            return Ok(());
        }
        let loads = self.slots.project_worker_loads(None, decay_now);
        let block = self.block_size as usize;
        let mut targets: Vec<(usize, WorkerWithDpRank)> = loads
            .iter()
            .filter(|(worker, _)| !holders.contains(worker))
            .map(|(worker, load)| {
                (
                    load.active_prefill_tokens + load.active_decode_blocks * block,
                    *worker,
                )
            })
            .collect();
        targets.sort_by_key(|(load, worker)| (*load, worker.worker_id, worker.dp_rank));
        let source_scheduler_id = self.scheduler_id(source)?;
        self.replicated.insert(parent, now_ms);
        for (_, target) in targets.into_iter().take(want) {
            let target_id = self.scheduler_id(target)?;
            self.replicas_sent += 1;
            self.kv_actions.send(
                target_id,
                vec![KvAction::Replicate {
                    source_scheduler_id,
                    block_hashes: sequence[..depth].to_vec(),
                    local_hashes: local[..depth].to_vec(),
                }],
            );
        }
        Ok(())
    }

    /// `kv_actions.peer_hbm`: KVCR sources include peers' device KV.
    fn peer_hbm(&self) -> bool {
        self.config
            .kv_actions_config()
            .ok()
            .flatten()
            .is_some_and(|actions| actions.peer_hbm)
    }

    fn dequeue_config(&self) -> Option<KvDequeueConfig> {
        self.config
            .kv_actions_config()
            .ok()
            .flatten()
            .and_then(|actions| actions.dequeue.clone())
    }

    /// `engine.dequeue`, at most every `check_s`: ask back each kept request (not batch) that
    /// has waited `after_s` since dispatch, when another worker has no prefill booked, a free
    /// running slot and KV room for it; the one holding most of its prefix is the target.
    fn dequeue_check(&mut self, now_ms: f64) -> Result<()> {
        let Some(config) = self.dequeue_config() else {
            return Ok(());
        };
        if self.dequeue.kept.is_empty() || now_ms < self.dequeue.next_check_ms {
            return Ok(());
        }
        self.dequeue.next_check_ms = now_ms + config.check_s * 1000.0;
        let due: Vec<Uuid> = self
            .dequeue
            .kept
            .iter()
            .filter(|(uuid, kept)| {
                !self.dequeue.asked.contains(uuid)
                    && now_ms - kept.at_ms >= config.after_s * 1000.0
                    && kept.request.policy_class.as_deref() != Some("batch")
            })
            .map(|(uuid, _)| *uuid)
            .collect();
        if due.is_empty() {
            return Ok(());
        }
        let loads = self
            .slots
            .project_worker_loads(None, self.decay_now(now_ms));
        let block = u64::from(self.block_size.max(1));
        let mut due = due;
        due.sort();
        for uuid in due {
            let Some(kept) = self.dequeue.kept.get(&uuid) else {
                continue;
            };
            let need_blocks = (kept.request.isl_tokens as u64).div_ceil(block);
            let fit = self.engine_fit.as_ref().filter(|fit| fit.config.dequeue);
            let target = loads
                .iter()
                .filter(|(worker, load)| {
                    if let Some(fit) = fit {
                        // f48h `engine_fit.dequeue`: a worker the turn fits on by the engines' ledger.
                        return **worker != kept.worker
                            && fit.fits(**worker, kept.request.isl_tokens) == Some(true);
                    }
                    **worker != kept.worker
                        && load.active_prefill_tokens == 0
                        && self
                            .workers_with_configs
                            .get(&worker.worker_id)
                            .is_some_and(|config| {
                                config
                                    .max_num_seqs()
                                    .is_none_or(|max| (load.active_requests as u64) < max)
                                    && config.total_kv_blocks().is_none_or(|total| {
                                        load.active_decode_blocks as u64 + need_blocks <= total
                                    })
                            })
                })
                .map(|(worker, _)| {
                    let hit = kept
                        .request
                        .overlap
                        .effective_overlap_blocks
                        .get(worker)
                        .copied()
                        .unwrap_or(0.0);
                    (*worker, hit)
                })
                .max_by(|a, b| {
                    a.1.total_cmp(&b.1).then_with(|| {
                        (b.0.worker_id, b.0.dp_rank).cmp(&(a.0.worker_id, a.0.dp_rank))
                    })
                })
                .map(|(worker, _)| worker);
            let Some(target) = target else {
                self.dequeue.stats.no_target += 1;
                continue;
            };
            let source = kept.source;
            self.dequeue.asked.insert(uuid);
            self.dequeue.targets.insert(uuid, target);
            self.dequeue.stats.asked += 1;
            self.kv_actions
                .send(source, vec![KvAction::Dequeue { request_id: uuid }]);
        }
        Ok(())
    }

    /// The earliest time a kept request becomes due for `engine.dequeue`.
    fn dequeue_wakeup_ms(&self) -> Option<f64> {
        let config = self.dequeue_config()?;
        let first = self
            .dequeue
            .kept
            .iter()
            .filter(|(uuid, _)| !self.dequeue.asked.contains(uuid))
            .map(|(_, kept)| kept.at_ms + config.after_s * 1000.0)
            .min_by(f64::total_cmp)?;
        Some(first.max(self.dequeue.next_check_ms))
    }

    /// A request the engine handed back: free its booking on the old worker and place it on
    /// the target chosen when it was asked back.
    pub(crate) fn requeue(&mut self, uuid: Uuid, now_ms: f64) -> Result<Vec<WorkerAdmission>> {
        let Some(kept) = self.dequeue.kept.remove(&uuid) else {
            bail!("engine.dequeue handed back {uuid}, which the router did not keep");
        };
        let target = self.dequeue.targets.remove(&uuid);
        let decay_now = self.decay_now(now_ms);
        if let Some(ranked) = self.busy_ranked.as_mut() {
            ranked.release(uuid);
        }
        self.slots
            .free(&uuid.to_string(), decay_now)
            .map_err(anyhow::Error::from)?;
        if let Some(fit) = self.engine_fit.as_mut() {
            fit.gone(uuid);
        }
        if let Some(iso) = self.prefill_iso.as_mut() {
            iso.gone(uuid);
        }
        let mut request = kept.request;
        request.dequeue_target = target;
        let retain_ttl_ms = request.retain_ttl_ms;
        let retain_value = request.retain_value;
        let context_tokens = retain_value.map_or(0, |_| {
            (request.isl_tokens + request.expected_output_tokens.unwrap_or(0) as usize) as u64
        });
        let outcome = self.admit_request(request, decay_now)?;
        self.dequeue.stats.requeued += 1;
        Ok(vec![WorkerAdmission {
            uuid,
            worker_idx: outcome.worker_idx,
            overlap_blocks: outcome.overlap_blocks,
            best_available_overlap_blocks: outcome.best_available_overlap_blocks,
            isl_blocks: outcome.isl_blocks,
            kv_fetch_source: outcome.kv_fetch_source,
            retain_ttl_ms,
            retain_value,
            context_tokens,
        }])
    }

    fn invalidate_mode(&self) -> Option<KvInvalidateMode> {
        self.config
            .kv_actions_config()
            .ok()
            .flatten()
            .and_then(|actions| actions.invalidate)
    }

    fn free_at_session_end(&self) -> bool {
        self.config
            .kv_actions_config()
            .ok()
            .flatten()
            .is_some_and(|actions| actions.free_at_session_end)
    }

    /// Whether a KV block action (`invalidate`, `demote`, `free_at_session_end`) needs each
    /// request's blocks.
    fn kv_block_actions_on(&self) -> bool {
        self.invalidate_mode().is_some()
            || self.demote_config().is_some()
            || self.release_unshared()
            || self.free_at_session_end()
    }

    /// `kv_actions.move_under_pressure_s` (with `release_source`).
    fn move_pressure_window_s(&self) -> Option<f64> {
        if !self.release_source_enabled() {
            return None;
        }
        self.config
            .kv_actions_config()
            .ok()
            .flatten()
            .and_then(|actions| actions.move_under_pressure_s)
    }

    /// `kv_actions.release_unshared` (with `release_source`).
    fn release_unshared(&self) -> bool {
        self.release_source_enabled()
            && self
                .config
                .kv_actions_config()
                .ok()
                .flatten()
                .is_some_and(|actions| actions.release_unshared)
    }

    /// `free_at_session_end`: `sequence` is now `stream_id`'s last request (counted until the
    /// stream ends).
    fn free_track(&mut self, stream_id: &str, sequence: &[u64]) {
        let refs = &mut self.kv_actions.free_refs;
        for hash in sequence {
            *refs.entry(*hash).or_insert(0) += 1;
        }
        if let Some(previous) = self
            .kv_actions
            .free_streams
            .insert(stream_id.to_string(), sequence.to_vec())
        {
            Self::free_release(&mut self.kv_actions.free_refs, &previous);
        }
    }

    /// Drop one live stream's hold on `sequence`; return the blocks no live stream holds now.
    fn free_release(refs: &mut HashMap<u64, u32>, sequence: &[u64]) -> Vec<u64> {
        let mut unheld = Vec::new();
        for hash in sequence {
            if let Some(count) = refs.get_mut(hash) {
                *count -= 1;
                if *count == 0 {
                    refs.remove(hash);
                    unheld.push(*hash);
                }
            }
        }
        unheld
    }

    /// `free_at_session_end`: a stream's `session_final` request finished. Its blocks that no
    /// other live stream's last request holds are dropped on every rank (`kv.invalidate`; the
    /// engine drops only blocks without cached children, tail first).
    fn free_session_end(&mut self, plan: &KvBlockPlan) -> Result<()> {
        if !plan.session_final || !self.free_at_session_end() {
            return Ok(());
        }
        let Some(sequence) = self.kv_actions.free_streams.remove(&plan.stream_id) else {
            return Ok(());
        };
        let unheld = Self::free_release(&mut self.kv_actions.free_refs, &sequence);
        let stats = &mut self.kv_actions.stats;
        stats.session_end_frees += 1;
        stats.session_end_shared_blocks += (sequence.len() - unheld.len()) as u64;
        if unheld.is_empty() {
            return Ok(());
        }
        stats.session_end_freed_blocks += unheld.len() as u64;
        for scheduler_id in self.all_ranks()? {
            self.kv_actions.send(
                scheduler_id,
                vec![KvAction::Invalidate {
                    block_hashes: unheld.clone(),
                }],
            );
        }
        Ok(())
    }

    /// Every rank's scheduler id, in worker and rank order.
    fn all_ranks(&self) -> Result<Vec<usize>> {
        let mut workers: Vec<WorkerId> = self.workers_with_configs.keys().copied().collect();
        workers.sort_unstable();
        workers
            .into_iter()
            .flat_map(|worker_id| {
                (0..self.dp_size).map(move |dp_rank| WorkerWithDpRank::new(worker_id, dp_rank))
            })
            .map(|worker| self.scheduler_id(worker))
            .collect()
    }

    /// A stream's next request arrived: its pending prefetch is moot, and with
    /// `kv_actions.invalidate` the blocks its previous request held and this one does not are
    /// dropped on every rank (`marker`: only at a compaction).
    fn on_stream_request(&mut self, pending: &mut PendingRequest, _now_ms: f64) -> Result<()> {
        let Some(plan) = pending.kv_blocks.as_ref() else {
            return Ok(());
        };
        let before = self.kv_actions.prefetches.len();
        self.kv_actions
            .prefetches
            .retain(|(_, _, timer)| timer.stream_id != plan.stream_id);
        self.kv_actions.stats.prefetches_cancelled +=
            (before - self.kv_actions.prefetches.len()) as u64;
        // A delayed demote of this stream's previous request is moot once its next request is here.
        self.kv_actions
            .demotes
            .retain(|(_, _, stream, _)| *stream != plan.stream_id);
        if let Some(mode) = self.invalidate_mode() {
            let final_request = pending
                .session_context
                .as_ref()
                .is_some_and(|context| context.session_final() == Some(true));
            let previous = if final_request {
                self.kv_actions.streams.remove(&plan.stream_id)
            } else {
                self.kv_actions
                    .streams
                    .insert(plan.stream_id.clone(), plan.sequence.clone())
            };
            if let Some(previous) = previous
                && (mode == KvInvalidateMode::Oracle || plan.compaction)
            {
                let current: HashSet<u64> = plan.sequence.iter().copied().collect();
                let dead: Vec<u64> = previous
                    .into_iter()
                    .filter(|hash| !current.contains(hash))
                    .collect();
                if !dead.is_empty() {
                    self.kv_actions.stats.invalidate_sent_blocks += dead.len() as u64;
                    for scheduler_id in self.all_ranks()? {
                        self.kv_actions.send(
                            scheduler_id,
                            vec![KvAction::Invalidate {
                                block_hashes: dead.clone(),
                            }],
                        );
                    }
                }
            }
        }
        if self.free_at_session_end() {
            let final_request = pending
                .session_context
                .as_ref()
                .is_some_and(|context| context.session_final() == Some(true));
            if let Some(plan) = pending.kv_blocks.as_mut() {
                plan.session_final = final_request;
                let (stream_id, sequence) = (plan.stream_id.clone(), plan.sequence.clone());
                self.free_track(&stream_id, &sequence);
            }
        } else if self.demote_config().is_none() && !self.release_unshared() {
            pending.kv_blocks = None;
        }
        Ok(())
    }

    /// `kv.demote` at a request's end: for a think time of at least `after_think_s` on a rank
    /// whose G1 is more than `g1_fraction` full, drop its G1 blocks G2 holds; with `prefetch`,
    /// load them back ahead of the predicted return.
    fn demote_finished(&mut self, uuid: Uuid, now_ms: f64) -> Result<()> {
        let Some((scheduler_id, worker, plan)) = self.kv_actions.admitted.remove(&uuid) else {
            return Ok(());
        };
        self.free_session_end(&plan)?;
        let Some(config) = self.demote_config().cloned() else {
            return Ok(());
        };
        let Some(think_ms) = plan.think_ms else {
            return Ok(());
        };
        if (think_ms as f64) < config.after_think_s * 1000.0 || plan.sequence.is_empty() {
            return Ok(());
        }
        let capacity = self
            .workers_with_configs
            .get(&worker.worker_id)
            .map(|config| config.total_kv_blocks)
            .filter(|&blocks| blocks > 0);
        let resident = self
            .kv_actions
            .g1_resident
            .get(&worker)
            .copied()
            .unwrap_or(0);
        let Some(capacity) = capacity else {
            return Ok(());
        };
        if (resident as f64) / (capacity as f64) <= config.g1_fraction {
            return Ok(());
        }
        let demote_ms = now_ms + config.delay_s.max(0.0) * 1000.0;
        if config.delay_s > 0.0 {
            self.kv_actions.demotes.push((
                demote_ms,
                scheduler_id,
                plan.stream_id.clone(),
                plan.sequence.clone(),
            ));
        } else {
            self.kv_actions.stats.demotes_sent += 1;
            self.kv_actions.send(
                scheduler_id,
                vec![KvAction::Demote {
                    block_hashes: plan.sequence.clone(),
                }],
            );
        }
        if let Some(prefetch) = &config.prefetch {
            let transfer_s = match (prefetch.load_gbs, prefetch.block_bytes) {
                (Some(gbs), Some(bytes)) if gbs > 0.0 => {
                    plan.sequence.len() as f64 * bytes / (gbs * 1e9)
                }
                _ => 0.0,
            };
            // Never before the demote it undoes (`delay_s`).
            let due_ms =
                (now_ms + think_ms as f64 - (prefetch.lead_s + transfer_s) * 1000.0).max(demote_ms);
            self.kv_actions
                .prefetches
                .push((due_ms, scheduler_id, plan));
        }
        Ok(())
    }

    /// Path B: take the out-of-band KV actions queued so far.
    pub(crate) fn take_kv_actions(&mut self) -> Vec<KvActionMessage> {
        std::mem::take(&mut self.kv_actions.queue)
    }

    pub(crate) fn kv_action_results(&mut self, results: Vec<KvActionResult>) {
        self.kv_actions.results(results);
    }

    /// `kv.retain` TTL for a request of `class` (policy file `kv_actions.retain`). With
    /// `expected`, only a request whose client expects the next turn within the TTL
    /// (`x-dynamo-hint-next-think-s`) and that does not end its session.
    fn retain_ttl_ms(
        &self,
        class: Option<&str>,
        fields: Option<&ReplayRequestFields>,
    ) -> Result<Option<u64>> {
        let Some(actions) = self.config.kv_actions_config()? else {
            return Ok(None);
        };
        let (Some(retain), Some(class)) = (&actions.retain, class) else {
            return Ok(None);
        };
        let ttl_s = retain.ttl_seconds.get(class).copied().unwrap_or(0.0);
        if ttl_s.is_nan() || ttl_s <= 0.0 {
            return Ok(None);
        }
        if retain.expected {
            let Some(fields) = fields else {
                return Ok(None);
            };
            if fields.session_final == Some(true) {
                return Ok(None);
            }
            let think_s = fields
                .agent_headers
                .get("x-dynamo-hint-next-think-s")
                .and_then(|values| values.first())
                .and_then(|value| value.parse::<f64>().ok());
            if !think_s.is_some_and(|think_s| think_s <= ttl_s) {
                return Ok(None);
            }
            // `until_return_margin_s`: protect until the expected return plus the margin.
            if let (Some(margin_s), Some(think_s)) = (retain.until_return_margin_s, think_s) {
                let until_s = (think_s.max(0.0) + margin_s.max(0.0)).min(ttl_s);
                return Ok(Some((until_s * 1000.0).round() as u64));
            }
        }
        Ok(Some((ttl_s * 1000.0).round() as u64))
    }

    #[cfg(test)]
    pub(crate) fn new(
        args: &MockerConfig,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        num_workers: usize,
    ) -> Result<Self> {
        Self::new_with_selector_seed(
            args,
            router_config,
            prefill_load_estimator,
            &vec![false; num_workers],
            None,
        )
    }

    /// Worker `i` receives KV fetch hints and serves as their source when
    /// `router_hints[i]` is set.
    pub(crate) fn new_with_selector_seed(
        args: &MockerConfig,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        router_hints: &[bool],
        selector_seed: Option<u64>,
    ) -> Result<Self> {
        Self::new_with_classifier(
            args,
            router_config,
            prefill_load_estimator,
            router_hints,
            selector_seed,
            None,
            None,
        )
    }

    /// As [`Self::new_with_selector_seed`], running `classifier` and its dispatch gate on every
    /// request (the replay gate). Its context reads the replay's virtual clock.
    pub(crate) fn new_with_classifier(
        args: &MockerConfig,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        router_hints: &[bool],
        selector_seed: Option<u64>,
        classifier: Option<&crate::replay::ReplayClassifierPlugin>,
        worker_selection: Option<&crate::replay::ReplayWorkerSelectionPlugin>,
    ) -> Result<Self> {
        let config = replay_router_config(args, router_config);
        let mut indexer = SyncReplayIndexer::new(args.block_size as u32);
        if config.enable_session_prefix_index {
            config.validate().map_err(anyhow::Error::msg)?;
            indexer.session_prefix_index = Some(Arc::new(SessionPrefixIndexer::new()));
        }
        let tracking_hash = tracking_hash_context(&config)?;
        let worker_config_template = replay_worker_config(args);
        let workers_with_configs = router_hints
            .iter()
            .enumerate()
            .map(|(worker_id, &router_hint)| {
                (
                    worker_id as WorkerId,
                    worker_config_template.with_router_hint(router_hint),
                )
            })
            .collect();
        let slots = replay_slots(args, &workers_with_configs);
        let selector = replay_selector_with_plugins(
            &config,
            selector_seed,
            classifier.is_some(),
            worker_selection,
        )?;
        // As in production: the startup policy reads the KV index (a later retune
        // keeps the index it started with), and the index retains the chain.
        let kv_transfer_chain_supported =
            WorkerSelector::<ReplayWorkerConfig>::required_worker_inputs(&selector)
                .reads_kv_index()
                && config.supports_kv_transfer_chain_retention();
        let profile = config
            .configured_policy_profile()
            .map_err(anyhow::Error::from)?;

        let mut router = Self {
            config,
            block_size: args.block_size as u32,
            dp_size: args.dp_size.max(1),
            profile: profile.clone(),
            worker_config_template,
            workers_with_configs,
            slots,
            selector,
            pending: PolicyQueue::new(&profile),
            indexer,
            prefill_load_estimator,
            // This is only a base Instant for converting replay `now_ms` values into
            // synthetic `Instant`s. All subsequent decay/accounting uses virtual replay
            // time derived from this epoch, not wall-clock progression.
            decay_time_epoch: Instant::now(),
            tracking_hash,
            kv_event_lag_ms: 0.0,
            lagged_kv_events: VecDeque::new(),
            affinity: None,
            affinity_waiting: VecDeque::new(),
            affinity_staged: HashMap::new(),
            affinity_active: HashMap::new(),
            needs_drain: false,
            kv_transfer_chain_supported,
            kv_transfer_gate_warned: false,
            classifier: None,
            request_fields: None,
            queue_reject_retry: queue_reject_retry(),
            queue_rejections: 0,
            refusal_log: refusal_log(),
            workset: workset::workset_log(),
            busy_ranked: busy_ranked(),
            class_active: class_cap::ClassActive::for_profile(&profile),
            kv_actions: KvActionSender::default(),
            retry_waiting: Vec::new(),
            retry_next_ms: None,
            retry_attempts: HashMap::new(),
            gate_wake_ms: None,
            dispatched_to: HashMap::new(),
            placed_cached: HashMap::new(),
            gauges_sent: HashMap::new(),
            dequeue: DequeueState::default(),
            engine_fit: None,
            prefill_iso: None,
            replicated: HashMap::new(),
            replicas_sent: 0,
        };
        router.engine_fit = router
            .config
            .kv_actions_config()
            .ok()
            .flatten()
            .and_then(|actions| actions.engine_fit.clone())
            .map(|config| engine_fit::EngineFit::new(config, router.block_size));
        router.prefill_iso = router
            .config
            .kv_actions_config()
            .ok()
            .flatten()
            .and_then(|actions| actions.prefill_isolation.clone())
            .map(prefill_iso::PrefillIsolation::new);
        router.refresh_kv_transfer_chain();
        if let Some(plugin) = classifier {
            let workers = router.workers_with_configs.clone();
            let dp_size = router.dp_size;
            let clock = VirtualClock::new(router.decay_time_epoch);
            // Plugins read the session prefix index when the replay tracks one (L47).
            let session_index = router.indexer.session_prefix_index.as_ref().map(|index| {
                Arc::new(
                    dynamo_kv_router::plugins::request_classifier::SessionPrefixIndexReader::new(
                        Arc::clone(index),
                    ),
                ) as Arc<dyn dynamo_admission::SessionIndexReader>
            });
            router.classifier = Some(ReplayClassifier::new(
                &plugin.0,
                router.block_size,
                move || worker_infos(&workers, dp_size),
                clock,
                session_index,
            ));
        }
        Ok(router)
    }

    fn advance_affinity_clock(&self, now_ms: f64) -> Result<()> {
        if let Some((_, table)) = &self.affinity {
            table.advance_clock(self.decay_now(now_ms))?;
        }
        Ok(())
    }

    fn affinity_id(
        &self,
        request: &DirectRequest,
        session: Option<&str>,
    ) -> Result<Option<String>> {
        let Some((config, _)) = &self.affinity else {
            return Ok(None);
        };
        let agentic = request
            .replay_context
            .as_ref()
            .and_then(|context| context.agentic.as_ref());
        let session = agentic
            .map(|identity| identity.conversation_id.as_str())
            .or(session)
            .filter(|id| !id.trim().is_empty())
            .context("conversation affinity requires a session identity")?;
        let play = agentic.map(|identity| identity.play_id.as_str());
        let key = match config.mode {
            ConversationAffinityMode::Session => {
                serde_json::to_string(&("session", play, session))?
            }
            ConversationAffinityMode::SiblingGroup => {
                let lineage = agentic
                    .and_then(|identity| identity.lineage.as_ref())
                    .context("sibling affinity requires unambiguous conversation lineage")?;
                anyhow::ensure!(
                    lineage.schema
                        == aisimulate_core::replay::AGENTIC_CONVERSATION_LINEAGE_SCHEMA_V1
                        && !lineage.root_conversation_id.is_empty(),
                    "unsupported conversation lineage"
                );
                serde_json::to_string(&(
                    if lineage.parent_conversation_id.is_some() {
                        "siblings"
                    } else {
                        "root"
                    },
                    play,
                    &lineage.root_conversation_id,
                    lineage.parent_conversation_id.as_deref().unwrap_or(session),
                ))?
            }
        };
        Ok(Some(subagent_group_affinity_id(&key)))
    }

    fn acquire_affinity(&self, request: &mut PendingRequest) -> Result<bool> {
        if let (Some((_, table)), Some(id)) = (&self.affinity, &request.affinity_id) {
            match table.try_acquire(id, None) {
                Ok(AcquireStep::Held(hold)) => request.affinity_hold = Some(hold),
                Ok(AcquireStep::Wait(_)) => return Ok(false),
                Err(AffinityError::ResourceExhausted(_)) => {
                    tracing::debug!(
                        session_id = id,
                        "Affinity table full; routing without session affinity"
                    );
                }
                Err(error) => return Err(error.into()),
            }
        }
        Ok(true)
    }

    pub(crate) fn set_kv_event_lag_ms(&mut self, lag_ms: f64) -> Result<()> {
        if !lag_ms.is_finite() || lag_ms < 0.0 {
            return Err(anyhow!(
                "kv_event_lag_ms must be finite and non-negative, got {lag_ms}"
            ));
        }
        self.kv_event_lag_ms = lag_ms;
        Ok(())
    }

    /// Apply, in arrival order, every lagged KV event batch visible at `now_ms`. KV events change
    /// no load, so this never admits queued requests on its own.
    fn apply_due_kv_events(&mut self, now_ms: f64) -> Result<()> {
        while let Some((_, events)) = self
            .lagged_kv_events
            .pop_front_if(|(visible_ms, _)| *visible_ms <= now_ms)
        {
            self.kv_actions.confirm(&events, now_ms);
            self.on_kv_events(events)?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn on_request_arrival(
        &mut self,
        request: &DirectRequest,
        replay_hashes: Option<ReplayRequestHashes>,
        now_ms: f64,
    ) -> Result<RouterEffects> {
        self.on_request_arrival_for_session(request, replay_hashes, None, now_ms)
    }

    #[cfg(test)]
    pub(crate) fn on_request_arrival_for_session(
        &mut self,
        request: &DirectRequest,
        replay_hashes: Option<ReplayRequestHashes>,
        session_id: Option<String>,
        now_ms: f64,
    ) -> Result<RouterEffects> {
        self.on_compact_request_arrival_for_session(
            request,
            request.effective_max_output_tokens(),
            replay_hashes,
            session_id,
            now_ms,
        )
    }

    fn on_compact_request_arrival_for_session<Request: PlacementRequestView>(
        &mut self,
        request: &Request,
        max_output_tokens: usize,
        replay_hashes: Option<ReplayRequestHashes>,
        session_id: Option<String>,
        now_ms: f64,
    ) -> Result<RouterEffects> {
        self.apply_due_kv_events(now_ms)?;
        self.advance_affinity_clock(now_ms)?;
        self.set_classifier_clock(now_ms);
        let mut pending =
            self.build_pending_request(request, max_output_tokens, replay_hashes, session_id)?;
        self.on_stream_request(&mut pending, now_ms)?;
        if !self.acquire_affinity(&mut pending)? {
            self.affinity_waiting.push_back(pending);
            return Ok(RouterEffects::default());
        }
        self.classify_and_enqueue(pending, now_ms)
    }

    /// Classifies `pending` (when a classifier is installed) and queues or admits it. A refusal
    /// (classify error) sends the request back to its client, which retries after a backoff,
    /// as a client does with a 429 and its Retry-After.
    fn classify_and_enqueue(
        &mut self,
        mut pending: PendingRequest,
        now_ms: f64,
    ) -> Result<RouterEffects> {
        if self.refusal_log.is_some() && !self.retry_attempts.contains_key(&pending.uuid) {
            self.note_refusal_log(now_ms, "offer", &pending, None);
        }
        if self.classifier.is_none() {
            return self.enqueue_or_admit(pending, now_ms);
        }
        let decay_now = self.decay_now(now_ms);
        let view = self.worker_view(&pending, decay_now);
        let mut request = ClassifyRequest::with_timing(pending.isl_tokens, 0, decay_now)
            .with_request_id(pending.request_id())
            .with_sequence_hashes(pending.token_seq.as_deref());
        if let Some(class) = &pending.policy_class {
            request = request.with_initial_policy_class(class.clone());
        }
        let context = pending.session_context.clone().or_else(|| {
            pending
                .session_id
                .clone()
                .map(|session_id| SessionContext::new(session_id, None, None, None))
        });
        let session_final = context
            .as_ref()
            .is_some_and(|context| context.session_final() == Some(true));
        if let Some(context) = context {
            request = request.with_session_context(context);
        }
        let classifier = self.classifier.as_mut().expect("checked above");
        if session_final {
            classifier.stats.session_final_marks += 1;
        }
        match classifier.classify(request, &view) {
            Ok(classified) => {
                if let Some(class) = classified.policy_class() {
                    pending.policy_class = Some(class.to_owned());
                }
                pending.due_at = classified.due_at();
                pending.retry_after = classified.retry_after();
                pending.order_key = classified.order_key();
                self.enqueue_or_admit(pending, now_ms)
            }
            Err((reason, hint)) => {
                classifier.stats.rejected += 1;
                classifier.log_decision(
                    now_ms,
                    "refuse",
                    &pending.uuid,
                    pending.policy_class.as_deref(),
                    None,
                );
                tracing::trace!(request_id = %pending.uuid, %reason, "replay classifier refused a request");
                self.schedule_retry(pending, now_ms, hint);
                Ok(RouterEffects::default())
            }
        }
    }

    /// The client of a refused request sends it again after its backoff.
    /// `hint`: the server's Retry-After, which replaces the backoff for this attempt.
    /// One line of the refusal log (`DYNAMO_REPLAY_REFUSAL_LOG`), when it is on.
    /// `hint`: the Retry-After the refusal carried, if any (F32).
    fn note_refusal_log(
        &mut self,
        now_ms: f64,
        kind: &str,
        pending: &PendingRequest,
        hint: Option<Duration>,
    ) {
        if let Some(out) = self.refusal_log.as_mut() {
            use std::io::Write as _;
            let line = serde_json::json!({
                "t": now_ms / 1000.0,
                "kind": kind,
                "uuid": pending.uuid.to_string(),
                "class": pending.policy_class.as_deref(),
                "session": pending.session_id.as_deref(),
                "retry_after_s": hint.map(|d| d.as_secs_f64()),
            });
            let _ = writeln!(out, "{line}");
        }
    }

    fn schedule_retry(&mut self, pending: PendingRequest, now_ms: f64, hint: Option<Duration>) {
        self.note_refusal_log(now_ms, "refused", &pending, hint);
        let attempt = self.retry_attempts.entry(pending.uuid).or_insert(0);
        let at_ms =
            now_ms + hint.map_or_else(|| retry_delay_ms(*attempt), |d| d.as_secs_f64() * 1000.0);
        *attempt += 1;
        self.retry_waiting.push(RetryEntry {
            at_ms,
            request: pending,
        });
        self.retry_next_ms = Some(self.retry_next_ms.map_or(at_ms, |next| next.min(at_ms)));
    }

    fn set_classifier_clock(&self, now_ms: f64) {
        if let Some(classifier) = &self.classifier {
            classifier.clock.set_ms(now_ms);
        }
    }

    /// The worker table the classifier declared, for one request.
    fn worker_view(&self, pending: &PendingRequest, decay_now: Instant) -> ReplayWorkerView {
        let Some(classifier) = &self.classifier else {
            return ReplayWorkerView { rows: Vec::new() };
        };
        let inputs = classifier.worker_inputs;
        if inputs == WorkerInputs::NONE {
            return ReplayWorkerView { rows: Vec::new() };
        }
        let loads = inputs
            .contains(WorkerInputs::LOAD)
            .then(|| self.slots.project_worker_loads(None, decay_now));
        let rows = worker_infos(&self.workers_with_configs, self.dp_size)
            .into_iter()
            .map(|info| {
                let worker = info.worker();
                let load = loads.as_ref().map(|loads| {
                    let projection = loads.get(&worker).copied().unwrap_or_default();
                    WorkerLoadView {
                        available: loads.contains_key(&worker),
                        active_prefill_tokens: projection.active_prefill_tokens,
                        active_decode_blocks: projection.active_decode_blocks,
                        decode_cost_blocks: projection.active_decode_blocks as f64,
                        active_requests: projection.active_requests,
                    }
                });
                let cache = inputs.contains(WorkerInputs::CACHE).then(|| {
                    let blocks = pending
                        .overlap
                        .effective_overlap_blocks
                        .get(&worker)
                        .copied()
                        .unwrap_or(0.0);
                    // f47i: with DYNAMO_REPLAY_WORKER_VIEW_TIERS=1 the view reports the router's
                    // per-tier matches (device, host-pinned, disk) instead of the effective estimate
                    // as device and nothing in the other tiers.
                    let tiers = &pending.overlap.tier_overlap_blocks;
                    let any_tier = !tiers.device.is_empty()
                        || !tiers.host_pinned.is_empty()
                        || !tiers.disk.is_empty();
                    let (device, host, disk) = if worker_view_tiers() && any_tier {
                        let get = |m: &FxHashMap<WorkerWithDpRank, usize>| {
                            m.get(&worker).copied().unwrap_or(0) as f64
                        };
                        (
                            get(&tiers.device),
                            get(&tiers.host_pinned),
                            get(&tiers.disk),
                        )
                    } else {
                        (blocks, 0.0, 0.0)
                    };
                    WorkerCacheView {
                        effective_overlap_blocks: blocks,
                        estimated_cached_tokens: pending
                            .overlap
                            .effective_cached_tokens
                            .get(&worker)
                            .copied()
                            .unwrap_or(0),
                        device_overlap_blocks: device,
                        host_overlap_blocks: host,
                        disk_overlap_blocks: disk,
                    }
                });
                (info, load, cache)
            })
            .collect();
        ReplayWorkerView { rows }
    }

    /// Delivers a lifecycle event to the classifier, when one is installed.
    fn deliver(&mut self, event: ClassifyEvent) {
        if let Some(classifier) = &mut self.classifier {
            classifier.deliver(event);
        }
    }

    /// The next replay time this router must be visited without another event: a gate hold
    /// deadline, a client retry, or a queued request's `due_at`.
    fn next_wakeup_ms(&self) -> Option<f64> {
        let prefetch = [self.kv_actions.next_prefetch_ms(), self.dequeue_wakeup_ms()]
            .into_iter()
            .flatten()
            .min_by(f64::total_cmp);
        // Client retries wait here with or without a classifier (a queue-limit rejection, too).
        let retry = self.retry_next_ms;
        let Some(classifier) = self.classifier.as_ref() else {
            return [retry, prefetch]
                .into_iter()
                .flatten()
                .min_by(f64::total_cmp);
        };
        let due = (0..self.pending.class_count())
            .filter_map(|class| self.pending.next_due_at(class))
            .map(|due| classifier.clock.ms_of(due))
            .min_by(f64::total_cmp);
        [self.gate_wake_ms, retry, due, prefetch]
            .into_iter()
            .flatten()
            .min_by(f64::total_cmp)
    }

    /// Requests whose client retry is due, back through classification.
    fn release_due_retries(&mut self, now_ms: f64) -> Result<Vec<WorkerAdmission>> {
        if self.retry_next_ms.is_none_or(|next| next > now_ms) {
            return Ok(Vec::new());
        }
        let (due, waiting): (Vec<_>, Vec<_>) = std::mem::take(&mut self.retry_waiting)
            .into_iter()
            .partition(|entry| entry.at_ms <= now_ms);
        self.retry_waiting = waiting;
        self.retry_next_ms = self
            .retry_waiting
            .iter()
            .map(|entry| entry.at_ms)
            .min_by(f64::total_cmp);
        let mut admissions = Vec::new();
        for entry in due {
            if let Some(classifier) = &mut self.classifier {
                classifier.stats.retries += 1;
                classifier.log_decision(
                    now_ms,
                    "retry",
                    &entry.request.uuid,
                    entry.request.policy_class.as_deref(),
                    None,
                );
            }
            let mut request = entry.request;
            request.due_at = None;
            request.order_key = None;
            request.queued_at = None;
            request.gate_target = None;
            admissions.extend(self.classify_and_enqueue(request, now_ms)?.admissions);
        }
        Ok(admissions)
    }

    fn enqueue_or_admit(
        &mut self,
        mut pending: PendingRequest,
        now_ms: f64,
    ) -> Result<RouterEffects> {
        let decay_now = self.decay_now(now_ms);
        let class_index = self
            .profile
            .resolve_class_index(pending.policy_class.as_deref())
            .map_err(dynamo_scheduler::host::KvSchedulerError::from)?;
        let class = self.profile.class(class_index);
        let gated = self
            .classifier
            .as_ref()
            .is_some_and(|classifier| classifier.gate.is_some());
        // A gate that asks for every class (`DispatchGate::gates_every_class`) also sees the
        // heads of classes with no busy threshold, as the live router's queue does.
        let gate_every_class = self.classifier.as_ref().is_some_and(|classifier| {
            classifier
                .gate
                .as_ref()
                .is_some_and(|gate| gate.gates_every_class())
        });
        // As in the live router: with a gate installed every queueing request is queued, so the
        // gate sees every head.
        let should_queue = (gated && gate_every_class)
            || class.queueing_enabled()
                && (gated
                    || self.pending.has_backlog(class_index)
                    || self
                        .engine_fit
                        .as_ref()
                        .is_some_and(|fit| fit.holds(pending.isl_tokens, 0.0))
                    || Self::request_workers_busy(
                        &self.ranked_loads(
                            self.slots.project_worker_loads(None, decay_now),
                            class_index,
                        ),
                        &self.workers_with_configs,
                        class,
                        &pending,
                        self.class_cap(class_index),
                    ));

        if should_queue {
            let snapshot = self.snapshot_for(&mut pending);
            pending.queued_at = Some(decay_now);
            let metadata = QueueMetadata {
                class_index,
                snapshot,
                due_at: pending.due_at,
                // Replay requests carry no explicit deadline. A class budget counts from the
                // virtual time the request reaches the scheduler, as the live router counts it
                // from scheduler ingress, after any affinity wait.
                class_due_at: class.slo_ttft_due_at(decay_now),
                arrival_offset_secs: now_ms.max(0.0) / 1000.0,
                order_key: pending.order_key,
            };
            let priority_jump = pending.priority_jump;
            let strict_priority = pending.strict_priority;
            // As in the live router, a session's requests wait in their own lane, so a
            // session bound to a busy rank holds only itself.
            let lane = pending.session_id.as_deref().map(SessionLane::new);
            if let Err((rejection, pending)) = self.pending.enqueue_with_due_at(
                metadata,
                self.workers_with_configs
                    .len()
                    .saturating_mul(self.dp_size as usize),
                priority_jump,
                strict_priority,
                lane,
                pending,
            ) {
                // The live router answers a queue-limit rejection to its client (HTTP 503/429).
                // Offline replay ends the run on it unless DYNAMO_REPLAY_QUEUE_REJECT_RETRY is set;
                // then the client sends the request again after its backoff, as for a refusal.
                if !self.queue_reject_retry {
                    return Err(anyhow::Error::new(rejection));
                }
                if let Some(classifier) = self.classifier.as_mut() {
                    classifier.stats.rejected += 1;
                }
                self.queue_rejections += 1;
                self.schedule_retry(pending, now_ms, None);
                return Ok(RouterEffects::default());
            }
            self.needs_drain |= self.affinity.is_some();
            if gated {
                // The gate decides now whether this head may go.
                return Ok(RouterEffects {
                    admissions: self.drain_pending(decay_now)?,
                });
            }
            return Ok(RouterEffects::default());
        }

        let uuid = pending.uuid;
        let retain_ttl_ms = pending.retain_ttl_ms;
        let retain_value = pending.retain_value;
        // Only the retention-by-value guard reads it.
        let context_tokens = retain_value.map_or(0, |_| {
            (pending.isl_tokens + pending.expected_output_tokens.unwrap_or(0) as usize) as u64
        });
        let outcome = self.admit_request(pending, decay_now)?;
        Ok(RouterEffects {
            admissions: vec![WorkerAdmission {
                uuid,
                worker_idx: outcome.worker_idx,
                overlap_blocks: outcome.overlap_blocks,
                best_available_overlap_blocks: outcome.best_available_overlap_blocks,
                isl_blocks: outcome.isl_blocks,
                kv_fetch_source: outcome.kv_fetch_source,
                retain_ttl_ms,
                retain_value,
                context_tokens,
            }],
        })
        .and_then(|effects: RouterEffects| {
            for admission in &effects.admissions {
                self.note_session_demand(admission)?;
            }
            Ok(effects)
        })
    }

    pub(crate) fn on_kv_events_at(
        &mut self,
        events: Vec<RouterEvent>,
        now_ms: f64,
    ) -> Result<RouterEffects> {
        self.write_workset(now_ms);
        self.kv_actions.expire(now_ms);
        kv_dup_probe::record(&events, now_ms);
        if self.move_pressure_window_s().is_some() {
            self.kv_actions.track_releases = true;
            for event in &events {
                if event.storage_tier.is_gpu() {
                    continue;
                }
                if let KvCacheEventData::Removed(remove) = &event.event.data {
                    let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
                    let hashes: Vec<u64> = remove.block_hashes.iter().map(|h| h.0).collect();
                    if let Ok(scheduler_id) = self.scheduler_id(worker) {
                        self.kv_actions
                            .note_g2_removed(scheduler_id, &hashes, now_ms);
                    }
                }
            }
        }
        if self.kv_event_lag_ms == 0.0 {
            self.kv_actions.confirm(&events, now_ms);
            return self.on_kv_events(events);
        }
        self.apply_due_kv_events(now_ms)?;
        self.lagged_kv_events
            .push_back((now_ms + self.kv_event_lag_ms, events));
        Ok(RouterEffects::default())
    }

    pub(crate) fn on_kv_events(&mut self, events: Vec<RouterEvent>) -> Result<RouterEffects> {
        if let Some(log) = self.workset.as_mut() {
            log.on_events(&events);
        }
        if self.demote_config().is_some() {
            self.kv_actions.count_g1(&events);
        }
        self.indexer.apply_events(events)?;
        Ok(RouterEffects::default())
    }

    pub(crate) fn on_prefill_completed(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
    ) -> Result<RouterEffects> {
        self.apply_due_kv_events(now_ms)?;
        let decay_now = self.decay_now(now_ms);
        self.dequeue.kept.remove(&uuid);
        if let Some(ranked) = self.busy_ranked.as_mut() {
            ranked.release(uuid);
        }
        self.slots
            .mark_prefill_completed(&uuid.to_string(), decay_now)
            .map_err(anyhow::Error::from)?;
        Ok(RouterEffects {
            admissions: self.drain_pending(decay_now)?,
        })
    }

    pub(crate) fn on_request_completed(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
    ) -> Result<RouterEffects> {
        self.apply_due_kv_events(now_ms)?;
        let decay_now = self.decay_now(now_ms);
        self.advance_affinity_clock(now_ms)?;
        self.affinity_staged.remove(&uuid);
        self.affinity_active.remove(&uuid);
        self.dequeue.kept.remove(&uuid);
        self.dequeue.targets.remove(&uuid);
        if let Some(ranked) = self.busy_ranked.as_mut() {
            ranked.release(uuid);
        }
        if let Some(active) = self.class_active.as_mut() {
            active.release(uuid);
        }
        self.slots
            .free(&uuid.to_string(), decay_now)
            .map_err(anyhow::Error::from)?;
        self.demote_finished(uuid, now_ms)?;
        Ok(RouterEffects {
            admissions: self.drain_pending(decay_now)?,
        })
    }

    /// Cancel a request that has not yet been assigned to a worker.
    pub(crate) fn cancel_pending(&mut self, uuid: Uuid) -> bool {
        if let Some(keyed) = self.classifier.as_mut().and_then(|c| c.keyed.as_mut()) {
            keyed.forget(&uuid);
        }
        let before = self.pending_count();
        self.affinity_waiting.retain(|request| request.uuid != uuid);
        self.pending.retain(|request| request.uuid != uuid);
        self.pending_count() != before
    }

    pub(crate) fn pending_count(&self) -> usize {
        self.pending.pending_count() + self.affinity_waiting.len()
    }

    /// Register a new worker with the router without disturbing existing slot state.
    /// With `router_hint`, it receives KV fetch hints and serves as their source.
    pub(crate) fn add_worker(&mut self, worker_id: usize, router_hint: bool) -> Result<()> {
        let wid = worker_id as WorkerId;
        if self
            .workers_with_configs
            .insert(
                wid,
                self.worker_config_template.with_router_hint(router_hint),
            )
            .is_some()
        {
            return Err(anyhow!("router worker {worker_id} already exists"));
        }
        if let Err(error) = self
            .slots
            .upsert_worker(WorkerDpRange::new(wid, 0, self.dp_size))
        {
            self.workers_with_configs.remove(&wid);
            return Err(error.into());
        }
        self.refresh_kv_transfer_chain();
        Ok(())
    }

    /// Lookups retain the transfer chain that KV fetch hints need while the
    /// router supports it and a routable worker takes hints.
    fn refresh_kv_transfer_chain(&mut self) {
        let hinted = self
            .workers_with_configs
            .values()
            .any(ReplayWorkerConfig::is_hint_capable);
        self.indexer.retain_kv_transfer_chain = hinted && self.kv_transfer_chain_supported;
        if hinted && !self.kv_transfer_chain_supported && !self.kv_transfer_gate_warned {
            self.kv_transfer_gate_warned = true;
            tracing::warn!(
                "router hints need a local event-driven indexer with no approximate side indexer; workers advertise hint capability but no hints will be attached"
            );
        }
    }

    /// Remove a worker from routing eligibility.
    ///
    /// Only removes the worker from the config map so the selector won't
    /// pick it for new requests.  The radix tree and active-sequence slots
    /// are left intact so that in-flight requests on this worker can still
    /// complete (free / mark_prefill_completed) and KV events can still
    /// reference existing blocks without "parent block not found" errors.
    /// Stale slot and indexer state is harmless — the selector and
    /// `all_workers_busy` both skip workers absent from `workers_with_configs`.
    pub(crate) fn remove_worker(&mut self, worker_id: usize) -> Result<()> {
        let wid = worker_id as WorkerId;
        self.workers_with_configs.remove(&wid);
        self.refresh_kv_transfer_chain();
        Ok(())
    }

    /// Drop the retained topology/cache state after the engine confirms that
    /// the draining worker no longer owns any request lifecycle state.
    pub(crate) fn finalize_worker_removal(&mut self, worker_id: usize) -> Result<()> {
        let wid = worker_id as WorkerId;
        self.slots
            .unregister_worker(wid)
            .map_err(anyhow::Error::from)?;
        self.indexer.remove_worker(wid);
        // Lagged events from the removed worker would re-add blocks it no longer owns.
        self.lagged_kv_events.retain_mut(|(_, events)| {
            events.retain(|event| event.worker_id != wid);
            !events.is_empty()
        });
        Ok(())
    }

    pub(crate) fn on_topology_changed(&mut self, now_ms: f64) -> Result<RouterEffects> {
        self.apply_due_kv_events(now_ms)?;
        if self.workers_with_configs.is_empty() {
            return Ok(RouterEffects::default());
        }
        let decay_now = self.decay_now(now_ms);
        Ok(RouterEffects {
            admissions: self.drain_pending(decay_now)?,
        })
    }

    #[cfg(test)]
    pub(crate) fn debug_snapshot(&self, now_ms: f64) -> OfflineRouterSnapshot {
        let decay_now = self.decay_now(now_ms);
        let mut pending = self
            .pending
            .entries()
            .map(|entry| {
                let mut overlap_blocks_by_worker = entry
                    .payload()
                    .overlap
                    .tier_overlap_blocks
                    .device
                    .iter()
                    .map(|(worker, overlap)| (worker.worker_id as usize, *overlap as u32))
                    .collect::<Vec<_>>();
                overlap_blocks_by_worker.sort_unstable_by_key(|(worker_id, _)| *worker_id);

                (
                    entry,
                    OfflinePendingRequestSnapshot {
                        uuid: entry.payload().uuid,
                        expected_output_tokens: entry.payload().expected_output_tokens,
                        overlap_blocks_by_worker,
                    },
                )
            })
            .collect::<Vec<_>>();
        pending.sort_unstable_by(|(left_entry, _), (right_entry, _)| {
            left_entry.cmp(right_entry).reverse()
        });

        let mut active_blocks_by_worker = self
            .slots
            .active_blocks()
            .into_iter()
            .map(|(worker, blocks)| (worker.worker_id as usize, blocks))
            .collect::<Vec<_>>();
        active_blocks_by_worker.sort_unstable_by_key(|(worker_id, _)| *worker_id);

        let mut active_tokens_by_worker = self
            .slots
            .active_tokens(decay_now)
            .into_iter()
            .map(|(worker, tokens)| (worker.worker_id as usize, tokens))
            .collect::<Vec<_>>();
        active_tokens_by_worker.sort_unstable_by_key(|(worker_id, _)| *worker_id);

        OfflineRouterSnapshot {
            pending: pending.into_iter().map(|(_, snapshot)| snapshot).collect(),
            active_blocks_by_worker,
            active_tokens_by_worker,
            indexer: self.indexer.debug_snapshot(),
        }
    }

    fn decay_now(&self, now_ms: f64) -> Instant {
        self.decay_time_epoch + Duration::from_secs_f64(now_ms.max(0.0) / 1000.0)
    }

    fn build_pending_request<Request: PlacementRequestView>(
        &self,
        request_view: &Request,
        max_output_tokens: usize,
        replay_hashes: Option<ReplayRequestHashes>,
        session_id: Option<String>,
    ) -> Result<PendingRequest> {
        let request = request_view.metadata();
        anyhow::ensure!(
            request.kv_hints.is_none(),
            "Dynamo KV router replay does not support authored KV fetch hints"
        );
        anyhow::ensure!(
            request.images.is_empty(),
            "Dynamo KV router replay does not support image requests"
        );
        let input_length = request_view.input_length();
        let uuid = request
            .uuid
            .ok_or_else(|| anyhow!("offline replay requires requests to have stable UUIDs"))?;
        let (priority_jump, strict_priority) = request.router_priorities();
        let retain_hashes = refresh_overlap()
            || self.affinity.is_some()
            || (self.indexer.session_prefix_index.is_some() && session_id.is_some());
        let mut block_hashes = None;
        let kv_block_hashes = replay_hashes
            .as_ref()
            .filter(|_| self.kv_block_actions_on())
            .map(|hashes| {
                (
                    hashes.sequence_hashes.clone(),
                    hashes.local_block_hashes.clone(),
                )
            });
        let replica_hashes = replay_hashes
            .as_ref()
            .filter(|_| self.replicate_config().is_some())
            .map(|hashes| {
                (
                    hashes.sequence_hashes.clone(),
                    hashes.local_block_hashes.clone(),
                )
            });
        let (tiered, token_seq) = match replay_hashes {
            Some(replay_hashes) => {
                let local_hashes =
                    crate::loadgen::local_block_hashes(replay_hashes.local_block_hashes);
                if retain_hashes {
                    block_hashes = Some(local_hashes.clone());
                }
                let tiered = self.indexer.find_matches_for_hashes(local_hashes);
                let token_seq = if !self.config.router_track_active_blocks {
                    None
                } else if self.config.router_assume_kv_reuse
                    && self.tracking_hash.algorithm() == TrackingHashAlgorithm::PublicXxh3V1
                {
                    Some(replay_hashes.sequence_hashes)
                } else if !self.config.router_assume_kv_reuse {
                    self.config
                        .random_seq_hashes_for_tracking(input_length / self.block_size as usize)
                } else {
                    let tokens = request_view.prompt_tokens_for_placement()?;
                    self.config.compute_seq_hashes_for_tracking_with_context(
                        &self.tracking_hash,
                        self.tracking_hash_scope(),
                        &tokens,
                        None,
                        BlockHashOptions::default(),
                        None,
                    )
                };
                (tiered, token_seq)
            }
            None => {
                let tokens = request_view.prompt_tokens_for_placement()?;
                let tiered = if retain_hashes {
                    let hashes = compute_block_hash_for_seq(
                        &tokens,
                        self.block_size,
                        BlockHashOptions::default(),
                    );
                    block_hashes = Some(hashes.clone());
                    self.indexer.find_matches_for_hashes(hashes)
                } else {
                    self.indexer.find_matches_for_request(&tokens, None)
                };
                let token_seq = self.config.compute_seq_hashes_for_tracking_with_context(
                    &self.tracking_hash,
                    self.tracking_hash_scope(),
                    &tokens,
                    None,
                    BlockHashOptions::default(),
                    None,
                );
                (tiered, token_seq)
            }
        };
        let overlap = OverlapAnalysis::new(
            &CacheHitWeights::from(&self.config),
            self.block_size,
            &tiered,
        )
        .signals();
        // Lookups retain candidates only while hints are on.
        let kv_transfer_candidates = transfer_candidates(tiered, self.peer_hbm());

        let resolved_fields = self.request_fields.as_ref().and_then(|fields| {
            let replay_context = request.replay_context.as_ref()?;
            let id = replay_context
                .agentic
                .as_ref()
                .map(|agentic| agentic.request_id.as_str())
                .unwrap_or(replay_context.authored_id.as_str());
            gate::resolve_request_fields(fields, id)
        });
        let fields = resolved_fields.as_deref();
        let session_context = fields.and_then(|f| f.session_context(session_id.as_deref()));
        let session_id = session_context
            .as_ref()
            .map(|context| context.session_id().to_owned())
            .or(session_id);
        let policy_class = fields
            .and_then(|f| f.policy_class.clone())
            .or_else(|| request.policy_class.clone());
        let retain_ttl_ms = self.retain_ttl_ms(policy_class.as_deref(), fields)?;
        let retain_value =
            self.retain_value(policy_class.as_deref(), fields, session_id.as_deref())?;
        let stream_id = fields
            .and_then(|fields| fields.stream_id.clone())
            .or_else(|| session_id.clone());
        // `kv.replicate`: a subagent of a large enough fan-out (client hint).
        let replica = self.replicate_config().and_then(|config| {
            let fields = fields?;
            let parent = fields.parent_session_id.clone()?;
            let fanout: u32 = fields
                .agent_headers
                .get("x-dynamo-hint-fanout")?
                .first()?
                .parse()
                .ok()?;
            (fanout >= config.fanout).then_some(())?;
            let (sequence, local) = replica_hashes?;
            Some((parent, sequence, local))
        });
        let kv_blocks = kv_block_hashes.and_then(|(sequence, local)| {
            let stream_id = stream_id.clone()?;
            Some(KvBlockPlan {
                stream_id,
                sequence,
                local,
                compaction: fields.and_then(|fields| fields.compaction) == Some(true),
                think_ms: next_think_ms(fields),
                session_final: false,
            })
        });

        Ok(PendingRequest {
            uuid,
            token_seq,
            isl_tokens: input_length,
            overlap,
            track_prefill_tokens: self.config.router_track_prefill_tokens,
            expected_output_tokens: Some(
                u32::try_from(max_output_tokens)
                    .context("max_output_tokens does not fit into u32")?,
            ),
            priority_jump,
            strict_priority,
            policy_class,
            affinity_id: self.affinity_id(request, session_id.as_deref())?,
            affinity_hold: None,
            session_id,
            block_hashes,
            kv_transfer_candidates,
            session_context,
            due_at: None,
            retry_after: None,
            order_key: None,
            queued_at: None,
            gate_target: None,
            retain_ttl_ms,
            retain_value,
            kv_blocks,
            stream_id,
            dequeue_target: None,
            replica,
        })
    }

    fn tracking_hash_scope(&self) -> TrackingHashScope<'_> {
        TrackingHashScope {
            partition: RoutingPartitionRef::new("replay", "default"),
            block_size: self.block_size,
        }
    }

    /// The working-set log's view of an admitted request: its session, class and blocks.
    fn note_workset_admit(&mut self, request: &PendingRequest) {
        let session = request
            .session_context
            .as_ref()
            .map(|context| context.session_id().to_string())
            .or_else(|| request.session_id.clone())
            .unwrap_or_else(|| request.uuid.to_string());
        let ends = request
            .session_context
            .as_ref()
            .and_then(SessionContext::session_final)
            == Some(true);
        let class = request.policy_class.clone().unwrap_or_default();
        // The router's sequence hashes are already chained (one key per prefix); authored local
        // hashes, when only they are there, are chained here.
        let keys: Vec<u64> = match (&request.token_seq, &request.block_hashes) {
            (Some(seq), _) => seq.clone(),
            (None, Some(local)) => {
                let local: Vec<u64> = local.iter().map(|hash| hash.0).collect();
                workset::chained_keys(&local)
            }
            (None, None) => Vec::new(),
        };
        if let Some(log) = self.workset.as_mut() {
            log.on_admit(
                request.uuid,
                &session,
                &class,
                keys,
                request.isl_tokens as u64,
                u64::from(request.expected_output_tokens.unwrap_or(0)),
                ends,
            );
        }
    }

    /// Write the working-set lines due by `now_ms` (a read of the router's state).
    fn write_workset(&mut self, now_ms: f64) {
        if !self.workset.as_ref().is_some_and(|log| log.due(now_ms)) {
            return;
        }
        let mut waiting = workset::Waiting::default();
        if let Some(log) = self.workset.as_ref() {
            let mut seen = HashSet::new();
            for entry in self.pending.entries() {
                let request = entry.payload();
                let session = request
                    .session_context
                    .as_ref()
                    .map(|context| context.session_id().to_string())
                    .or_else(|| request.session_id.clone())
                    .unwrap_or_else(|| request.uuid.to_string());
                let new_session = !log.is_live(&session) && seen.insert(session);
                waiting.add(
                    request.policy_class.as_deref().unwrap_or(""),
                    new_session,
                    request.isl_tokens as u64,
                );
            }
        }
        waiting.retry_requests = self.retry_waiting.len() as u64;
        let capacity: std::collections::BTreeMap<WorkerWithDpRank, u64> =
            worker_infos(&self.workers_with_configs, self.dp_size)
                .into_iter()
                .map(|info| {
                    let worker = info.worker();
                    let blocks = self
                        .workers_with_configs
                        .get(&worker.worker_id)
                        .map_or(0, |config| config.total_kv_blocks);
                    (worker, blocks)
                })
                .collect();
        let report = self
            .classifier
            .as_ref()
            .and_then(|classifier| classifier.gate.as_ref())
            .and_then(|gate| gate.report());
        let block_size = self.block_size;
        if let Some(log) = self.workset.as_mut() {
            log.write_due(now_ms, block_size, &capacity, &waiting, report.as_deref());
        }
    }

    fn admit_request(
        &mut self,
        mut request: PendingRequest,
        decay_now: Instant,
    ) -> Result<AdmitOutcome> {
        if self.workset.is_some() {
            self.note_workset_admit(&request);
        }
        // `engine.dequeue`: keep a copy until the prefill completes (once per request).
        let kept = (self.dequeue_config().is_some() && request.dequeue_target.is_none())
            .then(|| request.copy_for_dequeue());
        if let Some((parent, sequence, local)) = request.replica.take() {
            let overlap = request.overlap.clone();
            self.plan_replicas(parent, sequence, local, &overlap, decay_now)?;
        }
        // f47k (`DYNAMO_REPLAY_REFRESH_OVERLAP`): the overlap as it stands at selection, not at arrival.
        if refresh_overlap()
            && let Some(hashes) = &request.block_hashes
        {
            let tiered = self.indexer.find_matches_for_hashes(hashes.clone());
            request.overlap = OverlapAnalysis::new(
                &CacheHitWeights::from(&self.config),
                self.block_size,
                &tiered,
            )
            .signals();
            request.kv_transfer_candidates = transfer_candidates(tiered, self.peer_hbm());
        }
        let worker_loads = self
            .slots
            .project_worker_loads(request.token_seq.as_deref(), decay_now);
        let mut scheduling_request = request.scheduling_request(worker_loads);
        // f48h `engine_fit.place`: the workers this turn's whole prefix does not fit on now, by the
        // engines' own block counts, leave its selection while one fits.
        let fit_excluded = match self.engine_fit.as_mut() {
            Some(fit) if scheduling_request.pinned_worker.is_none() => fit.excluded(
                self.workers_with_configs.keys().copied(),
                self.dp_size,
                request.isl_tokens,
            ),
            _ => None,
        };
        // f49h `active_requests_per_worker`: workers whose ranks are all at the class's cap leave
        // the selection while another worker has room (not for a pinned request).
        let cap_excluded = match (self.class_active.as_ref(), scheduling_request.pinned_worker) {
            (Some(active), None) => self
                .profile
                .resolve_class_index(request.policy_class.as_deref())
                .ok()
                .and_then(|class_index| active.cap(class_index, self.profile.class(class_index)))
                .and_then(|cap| {
                    class_cap::excluded(&self.workers_with_configs, cap, fit_excluded.as_ref())
                }),
            _ => None,
        };
        let fit_excluded = cap_excluded.or(fit_excluded);
        // f62 `kv_actions.prefill_isolation`: a listed class's turn leaves the protected workers
        // on which it would compute a big prefill (`prefill_iso.rs`; never empties the selection).
        let fit_excluded = match self.prefill_iso.as_mut() {
            Some(iso) if scheduling_request.pinned_worker.is_none() => {
                let class = self
                    .profile
                    .resolve_class_index(request.policy_class.as_deref())
                    .ok()
                    .map(|index| self.profile.class(index).name.clone());
                let mut ids: Vec<WorkerId> = self.workers_with_configs.keys().copied().collect();
                ids.sort_unstable();
                let protected = match class {
                    Some(class) if iso.applies_to(&class) => {
                        iso.stats.scored += 1;
                        iso.protected(&ids, self.dp_size)
                    }
                    _ => None,
                };
                match protected {
                    Some(protected) => {
                        let block = self.block_size as usize;
                        let tiers = &scheduling_request.overlap.tier_overlap_blocks;
                        // What a peer could hand the turn over KVCR instead of computing it.
                        let peer =
                            scheduling_request
                                .kv_transfer_candidates
                                .as_ref()
                                .map_or(0, |c| {
                                    c.owner_prefix_blocks
                                        .iter()
                                        .map(|(_, n)| *n)
                                        .max()
                                        .unwrap_or(0)
                                });
                        let rows: Vec<prefill_iso::Row> = ids
                            .iter()
                            .zip(&protected)
                            .map(|(&worker_id, &protected)| {
                                let local = (0..self.dp_size.max(1))
                                    .map(|rank| {
                                        let w = WorkerWithDpRank::new(worker_id, rank);
                                        tiers.device.get(&w).copied().unwrap_or(0)
                                            + tiers.host_pinned.get(&w).copied().unwrap_or(0)
                                    })
                                    .max()
                                    .unwrap_or(0);
                                let cached = (local.max(peer) * block).min(request.isl_tokens);
                                prefill_iso::Row {
                                    worker_id,
                                    protected,
                                    new_tokens: request.isl_tokens - cached,
                                    priority: iso.priority_on(worker_id, self.dp_size),
                                    already: fit_excluded
                                        .as_ref()
                                        .is_some_and(|set| set.contains(&worker_id)),
                                }
                            })
                            .collect();
                        if protected.iter().any(|p| *p) {
                            iso.stats.protected_seen += 1;
                        }
                        let before = fit_excluded.as_ref().map_or(0, |set| set.len());
                        let (out, fell) =
                            prefill_iso::exclude(iso.config.max_prefill_tokens, &rows);
                        iso.stats.fallback += u64::from(fell);
                        match out {
                            Some(out) => {
                                iso.stats.steered += 1;
                                iso.stats.excluded += out.len().saturating_sub(before) as u64;
                                Some(out)
                            }
                            None => fit_excluded,
                        }
                    }
                    None => fit_excluded,
                }
            }
            _ => fit_excluded,
        };
        let eligibility = match fit_excluded.as_ref() {
            Some(excluded) => scheduling_request.eligibility_with_overloaded(Some(excluded)),
            None => scheduling_request.eligibility(),
        };
        // Best and selected overlap use the same weighted, rounded blocks as
        // the production router's overlap metric.
        let best_available_overlap_blocks = scheduling_request
            .overlap
            .effective_overlap_blocks
            .iter()
            .filter(|(worker, _)| {
                self.workers_with_configs
                    .get(&worker.worker_id)
                    .is_some_and(|config| eligibility.allows_worker(worker.worker_id, config))
                    && worker.dp_rank < self.dp_size
            })
            .map(|(_, overlap)| overlap.round() as u32)
            .max()
            .unwrap_or(0);
        let selection = self
            .selector
            .select_worker(WorkerSelectionInput::configured(
                &self.workers_with_configs,
                &scheduling_request,
                eligibility,
                self.block_size,
            ))?;
        let worker_idx = self.scheduler_id(selection.worker)?;
        let kv_fetch_source = self.kv_fetch_source(selection.kv_transfer_source)?;
        // f47k (`DYNAMO_REPLAY_PLACED_CACHE`): what the selected worker will not compute.
        if placed_cache() && self.classifier.is_some() {
            let tiers = &scheduling_request.overlap.tier_overlap_blocks;
            let get = |m: &FxHashMap<WorkerWithDpRank, usize>| {
                m.get(&selection.worker).copied().unwrap_or(0)
            };
            let local = get(&tiers.device) + get(&tiers.host_pinned) + get(&tiers.disk);
            let peer = selection.kv_transfer_source.map_or(0, |c| c.prefix_blocks);
            let cached = (local.max(peer) * self.block_size as usize).min(request.isl_tokens);
            self.placed_cached.insert(request.uuid, cached);
        }
        if kvcr_probe::enabled() {
            let tiers = &scheduling_request.overlap.tier_overlap_blocks;
            let device = tiers.device.get(&selection.worker).copied().unwrap_or(0);
            let host = tiers
                .host_pinned
                .get(&selection.worker)
                .copied()
                .unwrap_or(0);
            let (index_best, index_sources) = scheduling_request
                .kv_transfer_candidates
                .as_ref()
                .map_or((0, 0), |c| {
                    (
                        c.owner_prefix_blocks
                            .iter()
                            .map(|(_, n)| *n)
                            .max()
                            .unwrap_or(0),
                        c.owner_prefix_blocks.len(),
                    )
                });
            let now_ms = decay_now
                .saturating_duration_since(self.decay_time_epoch)
                .as_secs_f64()
                * 1000.0;
            let opt = |v: Option<usize>| v.map_or_else(|| "null".to_string(), |n| n.to_string());
            kvcr_probe::write(&format!(
                r#"{{"ev":"route","uuid":"{}","t":{:.3},"w":{worker_idx},"device":{device},"host":{host},"src":{},"offer":{},"index_best":{index_best},"index_sources":{index_sources},"isl_blocks":{}}}"#,
                request.uuid,
                now_ms,
                opt(kv_fetch_source),
                opt(selection.kv_transfer_source.map(|c| c.prefix_blocks)),
                request.isl_tokens / self.block_size as usize,
            ));
        }
        // A move only migrates a stream's own KV: the source must be where this stream ran
        // last. A prefix another stream reads there (a subagent's parent prefix, a shared
        // system prompt) is copied, so the source's other sessions keep it.
        let own_stream_source = request.stream_id.as_ref().and_then(|stream| {
            let previous = self.kv_actions.stream_home.get(stream).copied();
            if self.release_source_enabled() {
                let finished = request
                    .session_context
                    .as_ref()
                    .is_some_and(|context| context.session_final() == Some(true));
                if finished {
                    self.kv_actions.stream_home.remove(stream);
                } else {
                    self.kv_actions
                        .stream_home
                        .insert(stream.clone(), worker_idx);
                }
            }
            previous
        });
        if self.release_unshared()
            && let Some(plan) = request.kv_blocks.as_ref()
        {
            let finished = request
                .session_context
                .as_ref()
                .is_some_and(|context| context.session_final() == Some(true));
            let blocks = (!finished).then_some(plan.sequence.as_slice());
            self.kv_actions.note_stream_blocks(&plan.stream_id, blocks);
        }
        if kv_fetch_source.is_some() && self.release_source_enabled() {
            if own_stream_source == kv_fetch_source {
                self.kv_actions.stats.moves_own_stream += 1;
            } else {
                self.kv_actions.stats.copies_other_stream += 1;
            }
        }
        if let (Some(source_scheduler_id), Some(choice)) =
            (kv_fetch_source, selection.kv_transfer_source)
            && self.release_source_enabled()
            && own_stream_source == Some(source_scheduler_id)
            && let Some(candidates) = scheduling_request.kv_transfer_candidates.as_ref()
            && self.move_pressure_window_s().is_none_or(|window_s| {
                let now_ms = decay_now
                    .saturating_duration_since(self.decay_time_epoch)
                    .as_secs_f64()
                    * 1000.0;
                let pressed = self
                    .kv_actions
                    .under_pressure(source_scheduler_id, now_ms, window_s);
                if !pressed {
                    self.kv_actions.stats.moves_skipped_no_pressure += 1;
                }
                pressed
            })
        {
            // A move: the fetch copies the source's prefix of this request; release it only
            // after the target's KV events confirm each block.
            let unshared = self.release_unshared();
            let fetched = candidates.block_hashes.iter().take(choice.prefix_blocks);
            let mut expected: HashSet<u64> = HashSet::new();
            for hash in fetched {
                if unshared && self.kv_actions.shared(hash.0) {
                    self.kv_actions.stats.kept_shared_blocks += 1;
                } else {
                    expected.insert(hash.0);
                }
            }
            if !expected.is_empty() {
                let now_ms = decay_now
                    .saturating_duration_since(self.decay_time_epoch)
                    .as_secs_f64()
                    * 1000.0;
                self.kv_actions.stats.moves_started += 1;
                self.kv_actions.moves.push(PendingMove {
                    source_scheduler_id,
                    target: selection.worker,
                    expected,
                    deadline_ms: now_ms + MOVE_CONFIRM_TIMEOUT_MS,
                });
            }
        }
        if let Some(plan) = request.kv_blocks.take() {
            self.kv_actions
                .admitted
                .insert(request.uuid, (worker_idx, selection.worker, plan));
        }
        let request_id = request.request_id();
        let prefill_load_hint = self.prefill_load_hint_for(
            request.isl_tokens,
            selection.cached_tokens,
            request.track_prefill_tokens,
        );
        if let Some(active) = self.class_active.as_mut()
            && let Ok(class_index) = self
                .profile
                .resolve_class_index(request.policy_class.as_deref())
        {
            active.book(request.uuid, selection.worker, class_index);
        }
        if self.busy_ranked.is_some() && request.track_prefill_tokens {
            let class_index = self
                .profile
                .resolve_class_index(request.policy_class.as_deref())
                .unwrap_or(usize::MAX);
            let tokens = request
                .isl_tokens
                .saturating_sub(selection.cached_tokens.min(request.isl_tokens));
            if let Some(ranked) = self.busy_ranked.as_mut() {
                ranked.book(request.uuid, selection.worker, class_index, tokens);
            }
        }

        let isl_blocks = u32::try_from(request.isl_tokens.div_ceil(self.block_size as usize))
            .unwrap_or(u32::MAX);
        let overlap_blocks = selection.effective_overlap_blocks.round() as u32;

        self.slots
            .add_request(
                SequenceRequest {
                    request_id,
                    token_sequence: scheduling_request.token_seq.take(),
                    track_prefill_tokens: request.track_prefill_tokens,
                    expected_output_tokens: request.expected_output_tokens,
                    prefill_load_hint,
                    worker: selection.worker,
                    lora_name: None,
                },
                decay_now,
            )
            .map_err(anyhow::Error::from)?;

        if let Some(index) = &self.indexer.session_prefix_index
            && let Some(session_id) = request.session_id.as_deref()
            && let Some(hashes) = request.block_hashes.take()
            && let Some(&matched_hash) = self
                .indexer
                .tree
                .find_match_details(hashes, false)
                .last_matched_hashes
                .get(&selection.worker)
        {
            index.update_session_from_match(session_id, selection.worker, matched_hash)?;
        }

        if let Some(hold) = request.affinity_hold.take() {
            self.affinity_staged
                .insert(request.uuid, (hold, selection.worker.into()));
        }
        if let Some(fit) = self.engine_fit.as_mut() {
            fit.dispatched(
                request.uuid,
                selection.worker,
                u64::from(request.expected_output_tokens.unwrap_or(0)),
            );
        }
        if let Some(iso) = self.prefill_iso.as_mut() {
            let priority = self
                .profile
                .resolve_class_index(request.policy_class.as_deref())
                .is_ok_and(|index| iso.is_priority(&self.profile.class(index).name));
            iso.dispatched(request.uuid, selection.worker, priority);
        }
        if self.classifier.is_some() {
            let output_tokens = request.expected_output_tokens.unwrap_or(0) as usize;
            let context_tokens = request.isl_tokens + output_tokens;
            self.dispatched_to.insert(
                request.uuid,
                (selection.worker, context_tokens, output_tokens),
            );
            self.retry_attempts.remove(&request.uuid);
        }
        if let Some(copy) = kept {
            let at_ms = decay_now
                .saturating_duration_since(self.decay_time_epoch)
                .as_secs_f64()
                * 1000.0;
            self.dequeue.kept.insert(
                copy.uuid,
                KeptRequest {
                    request: copy,
                    source: worker_idx,
                    worker: selection.worker,
                    at_ms,
                },
            );
        }
        Ok(AdmitOutcome {
            worker_idx,
            overlap_blocks,
            best_available_overlap_blocks,
            isl_blocks,
            kv_fetch_source,
        })
    }

    fn scheduler_id(&self, worker: WorkerWithDpRank) -> Result<usize> {
        usize::try_from(worker.worker_id)
            .ok()
            .and_then(|worker_id| worker_id.checked_mul(self.dp_size as usize))
            .and_then(|base| base.checked_add(worker.dp_rank as usize))
            .ok_or_else(|| anyhow!("scheduler id of {worker:?} does not fit into usize"))
    }

    /// Replay models no cache owners, so a cache-owner source gives no hint.
    fn kv_fetch_source(&self, source: Option<KvTransferSourceChoice>) -> Result<Option<usize>> {
        match source.map(|choice| choice.source) {
            Some(KvTransferCandidateSource::Worker(worker)) => self.scheduler_id(worker).map(Some),
            Some(KvTransferCandidateSource::CacheOwner(_)) | None => Ok(None),
        }
    }

    fn drain_pending(&mut self, decay_now: Instant) -> Result<Vec<WorkerAdmission>> {
        self.needs_drain = false;
        let mut admissions = Vec::new();
        let gate = self
            .classifier
            .as_ref()
            .and_then(|classifier| classifier.gate.clone());
        if gate.is_none() {
            // No dispatch gate: capacity is the only reason a lane blocks.
            self.pending.recheck_blocked_lanes(RecheckScope::Capacity);
        } else {
            let expired_before = self.classifier.as_ref().map_or(0, |c| c.stats.expired);
            self.expire_due(decay_now);
            // A gate hold ends on its `notify` or its deadline; replay asks every held lane
            // again at each drain, which is when either can have happened, and a hold that still
            // stands answers `Hold` again at no cost to correctness. With
            // `DYNAMO_REPLAY_GATE_RECHECK=notify` it asks them only when one of those happened.
            let notified = self
                .classifier
                .as_ref()
                .is_some_and(|classifier| classifier.take_recheck());
            let now_ms = decay_now
                .saturating_duration_since(self.decay_time_epoch)
                .as_secs_f64()
                * 1000.0;
            let deadline_due = self.gate_wake_ms.is_some_and(|wake| wake <= now_ms);
            let expired = self.classifier.as_ref().map_or(0, |c| c.stats.expired) > expired_before;
            let notify_only = self.classifier.as_ref().is_some_and(|classifier| {
                classifier.recheck_mode == gate::ReplayGateRecheck::Notify
            });
            let full = !notify_only || notified || deadline_due || expired;
            self.pending.recheck_blocked_lanes(if full {
                RecheckScope::All
            } else {
                RecheckScope::Capacity
            });
            if let Some(classifier) = &mut self.classifier {
                classifier.stats.drains += 1;
                if full {
                    classifier.stats.full_rechecks += 1;
                }
                // Keyed rechecks: parked lanes the full recheck could answer differently now are
                // asked again in this drain.
                if let Some(keyed) = classifier.keyed.as_mut() {
                    classifier.stats.keyed_answers += keyed.parked_count() as u64;
                    for (class_index, lane) in keyed.due(now_ms) {
                        self.pending.unpark_lane(class_index, lane);
                        classifier.stats.keyed_answers -= 1;
                    }
                }
                if classifier.recheck_mode == gate::ReplayGateRecheck::KeyedVerify
                    && let (Some(keyed), Some(gate)) = (classifier.keyed.as_ref(), gate.as_ref())
                {
                    for (uuid, class_index, lane, keys) in keyed.parked() {
                        let Some(request) = self.pending.lane_head(class_index, lane) else {
                            continue;
                        };
                        let request_id = request.request_id();
                        let context = request.session_context.clone().or_else(|| {
                            request
                                .session_id
                                .clone()
                                .map(|id| SessionContext::new(id, None, None, None))
                        });
                        let class_name = self.profile.class(class_index).name.clone();
                        let head = DispatchHead::new(
                            Some(&request_id),
                            context.as_ref(),
                            request.isl_tokens,
                            None,
                            &class_name,
                            request.queued_at.unwrap_or(decay_now),
                        );
                        if request.uuid != *uuid {
                            tracing::warn!(%uuid, head = %request.uuid, "keyed_hold_stale: parked lane head changed");
                            continue;
                        }
                        if let DispatchDecision::Go { .. } = gate.check(&head) {
                            tracing::warn!(
                                %uuid,
                                t_s = now_ms / 1000.0,
                                session = ?context.as_ref().map(|c| c.session_id().to_owned()),
                                keys = ?keys,
                                "keyed_hold_stale: a parked head would go"
                            );
                        }
                    }
                }
            }
        }
        let now_ms = decay_now
            .saturating_duration_since(self.decay_time_epoch)
            .as_secs_f64()
            * 1000.0;
        let wake = std::cell::Cell::new(None::<f64>);
        let holds = std::cell::Cell::new(0u64);
        let checks = std::cell::Cell::new(0u64);
        let parks = std::cell::Cell::new(0u64);
        // `DYNAMO_REPLAY_GATE_RECHECK=keyed`: the parked heads (taken out for the drain, put back
        // after).
        let keyed = std::cell::RefCell::new(
            self.classifier
                .as_mut()
                .and_then(|classifier| classifier.keyed.take()),
        );
        // Requests that got a `Hold` in this drain (distinct-hold accounting after the loop).
        let held_now = std::cell::RefCell::new(Vec::<(Uuid, String)>::new());
        // The gate's `Go` target per head, set by the predicate and taken when the head leaves.
        let targets = std::cell::RefCell::new(HashMap::<Uuid, Option<WorkerWithDpRank>>::new());
        let epoch = self.decay_time_epoch;
        // f48h `engine_fit.hold`: heads held because they fit on no worker, in this drain.
        let fit_held = std::cell::Cell::new(0u64);
        loop {
            // Most completions find an empty queue, which never consults the
            // predicate, so only snapshot worker loads once one is needed.
            let mut worker_loads = None;
            let slots = &self.slots;
            let ranked = self.busy_ranked.as_ref();
            let class_active = self.class_active.as_ref();
            let workers = &self.workers_with_configs;
            let fit = self.engine_fit.as_ref();
            let gate = gate.as_deref();
            let Some(popped) =
                self.pending
                    .pop_next(|class_index, class, request: &PendingRequest| {
                        if let Some(gate) = gate {
                            let request_id = request.request_id();
                            let context = request.session_context.clone().or_else(|| {
                                request
                                    .session_id
                                    .clone()
                                    .map(|id| SessionContext::new(id, None, None, None))
                            });
                            let head = DispatchHead::new(
                                Some(&request_id),
                                context.as_ref(),
                                request.isl_tokens,
                                None,
                                &class.name,
                                request.queued_at.unwrap_or(decay_now),
                            );
                            checks.set(checks.get() + 1);
                            let mut keyed = keyed.borrow_mut();
                            let (decision, hold_wake) = match keyed.as_ref() {
                                Some(_) => gate.check_wake(&head),
                                None => (gate.check(&head), HoldWake::Any),
                            };
                            // A keyed hold on a session lane parks the lane (the shared lane cannot park).
                            let park = match (
                                &decision,
                                hold_wake,
                                keyed.as_mut(),
                                request.session_id.as_deref(),
                            ) {
                                (
                                    DispatchDecision::Hold { until },
                                    HoldWake::Keys(keys),
                                    Some(keyed),
                                    Some(session),
                                ) => {
                                    let until_ms = until.map(|until| {
                                        until.saturating_duration_since(epoch).as_secs_f64()
                                            * 1000.0
                                    });
                                    keyed.park(
                                        request.uuid,
                                        class_index,
                                        SessionLane::new(session),
                                        until_ms,
                                        keys,
                                    );
                                    parks.set(parks.get() + 1);
                                    true
                                }
                                _ => false,
                            };
                            drop(keyed);
                            match decision {
                                DispatchDecision::Go { target } => {
                                    targets.borrow_mut().insert(request.uuid, target);
                                }
                                DispatchDecision::Hold { until } => {
                                    holds.set(holds.get() + 1);
                                    held_now
                                        .borrow_mut()
                                        .push((request.uuid, class.name.clone()));
                                    if let Some(until) = until {
                                        let until_ms =
                                            until.saturating_duration_since(epoch).as_secs_f64()
                                                * 1000.0;
                                        // A deadline not in the future arms nothing; it waits for notify.
                                        if until_ms > now_ms {
                                            wake.set(Some(
                                                wake.get().map_or(until_ms, |w| w.min(until_ms)),
                                            ));
                                        }
                                    }
                                    return LaneCheck::Blocked(if park {
                                        BlockReason::GateParked
                                    } else {
                                        BlockReason::Gate
                                    });
                                }
                            }
                        }
                        let worker_loads = worker_loads
                            .get_or_insert_with(|| slots.project_worker_loads(None, decay_now));
                        let ranked_loads = ranked.map(|r| r.for_class(worker_loads, class_index));
                        let waited_s = request.queued_at.map_or(0.0, |at| {
                            decay_now.saturating_duration_since(at).as_secs_f64()
                        });
                        if fit.is_some_and(|fit| fit.holds(request.isl_tokens, waited_s)) {
                            fit_held.set(fit_held.get() + 1);
                            return LaneCheck::from(false);
                        }
                        LaneCheck::from(!Self::request_workers_busy(
                            ranked_loads.as_ref().unwrap_or(worker_loads),
                            workers,
                            class,
                            request,
                            class_active.and_then(|active| active.cap(class_index, class)),
                        ))
                    })
            else {
                break;
            };
            let class_index = popped.class_index();
            let mut request = popped.into_payload();
            if let Some(gate) = gate {
                request.gate_target = targets.borrow_mut().remove(&request.uuid).flatten();
                let request_id = request.request_id();
                let context = request.session_context.clone().or_else(|| {
                    request
                        .session_id
                        .clone()
                        .map(|id| SessionContext::new(id, None, None, None))
                });
                let class_name = self.profile.class(class_index).name.clone();
                gate.dispatched(
                    &DispatchHead::new(
                        Some(&request_id),
                        context.as_ref(),
                        request.isl_tokens,
                        None,
                        &class_name,
                        request.queued_at.unwrap_or(decay_now),
                    ),
                    request.gate_target,
                );
            }
            let uuid = request.uuid;
            if let Some(keyed) = keyed.borrow_mut().as_mut() {
                keyed.forget(&uuid);
            }
            if let Some(classifier) = &mut self.classifier {
                let class_name = &self.profile.class(class_index).name;
                classifier.note_left(&uuid, Some(class_name), now_ms, false);
            }
            let retain_ttl_ms = request.retain_ttl_ms;
            let retain_value = request.retain_value;
            let context_tokens = retain_value.map_or(0, |_| {
                (request.isl_tokens + request.expected_output_tokens.unwrap_or(0) as usize) as u64
            });
            let outcome = self.admit_request(request, decay_now)?;
            admissions.push(WorkerAdmission {
                uuid,
                worker_idx: outcome.worker_idx,
                overlap_blocks: outcome.overlap_blocks,
                best_available_overlap_blocks: outcome.best_available_overlap_blocks,
                isl_blocks: outcome.isl_blocks,
                kv_fetch_source: outcome.kv_fetch_source,
                retain_ttl_ms,
                retain_value,
                context_tokens,
            });
            self.note_session_demand(admissions.last().unwrap())?;
        }
        // Parked heads are not asked: their deadlines arm the wake-up as an answer would.
        let keyed = keyed.into_inner();
        if let Some(until_ms) = keyed.as_ref().and_then(KeyedHolds::next_until_ms)
            && until_ms > now_ms
        {
            wake.set(Some(wake.get().map_or(until_ms, |w| w.min(until_ms))));
        }
        if let Some(fit) = self.engine_fit.as_mut() {
            fit.stats.held += fit_held.get();
        }
        if let Some(classifier) = &mut self.classifier {
            classifier.keyed = keyed;
            classifier.stats.parks += parks.get();
            classifier.stats.holds += holds.get();
            classifier.stats.gate_checks += checks.get();
            classifier.stats.max_checks_per_drain =
                classifier.stats.max_checks_per_drain.max(checks.get());
            // Lanes are asked in their map's order (a per-process hash); note them in request
            // order so the decision log repeats.
            let mut held_now = held_now.into_inner();
            held_now.sort_by_key(|(uuid, _)| *uuid);
            for (uuid, class) in held_now {
                classifier.note_hold(uuid, &class, now_ms);
            }
        }
        self.gate_wake_ms = wake.get();
        Ok(admissions)
    }

    /// Queued requests past their classifier `due_at` leave the queue (the live router answers
    /// 504) and their clients retry after a backoff.
    fn expire_due(&mut self, decay_now: Instant) {
        let now_ms = decay_now
            .saturating_duration_since(self.decay_time_epoch)
            .as_secs_f64()
            * 1000.0;
        for class_index in 0..self.pending.class_count() {
            // The queue returns them in its lane map's order, which follows a per-process hash
            // of the session ids; replay takes them in request order so a run repeats.
            let mut expired = self.pending.take_expired(class_index, decay_now);
            expired.sort_by_key(|entry| entry.payload().uuid);
            for entry in expired {
                let request = entry.into_payload();
                self.deliver(ClassifyEvent::Aborted {
                    request_id: request.request_id(),
                    worker: None,
                    error: None,
                });
                if let Some(classifier) = &mut self.classifier {
                    classifier.stats.expired += 1;
                    let class_name = &self.profile.class(class_index).name;
                    classifier.note_left(&request.uuid, Some(class_name), now_ms, true);
                }
                let hint = request.retry_after;
                self.schedule_retry(request, now_ms, hint);
            }
        }
    }

    /// The loads a class's busy test reads: as projected, or with lower classes' booked prefill
    /// taken out under `DYNAMO_REPLAY_BUSY_RANKED`.
    fn ranked_loads(
        &self,
        loads: FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
        class_index: usize,
    ) -> FxHashMap<WorkerWithDpRank, WorkerLoadProjection> {
        match &self.busy_ranked {
            Some(ranked) => ranked.for_class(&loads, class_index),
            None => loads,
        }
    }

    /// f49h: the cap of class `class_index`, when it sets one.
    fn class_cap(&self, class_index: usize) -> Option<class_cap::ClassCap<'_>> {
        self.class_active
            .as_ref()
            .and_then(|active| active.cap(class_index, self.profile.class(class_index)))
    }

    fn request_workers_busy(
        worker_loads: &FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
        workers: &HashMap<WorkerId, ReplayWorkerConfig>,
        class: &PolicyClassConfig,
        request: &PendingRequest,
        cap: Option<class_cap::ClassCap<'_>>,
    ) -> bool {
        // f49h: a class with a cap also counts its active requests per rank.
        if cap.is_some() {
            let target = request
                .affinity_hold
                .as_ref()
                .and_then(Hold::target)
                .map(|target| (target.worker_id, target.dp_rank));
            return class_cap::workers_busy(worker_loads, workers, class, target, cap);
        }
        match request.affinity_hold.as_ref().and_then(Hold::target) {
            Some(target) => {
                workers
                    .get(&target.worker_id)
                    .is_none_or(|config| match target.dp_rank {
                        // f50: with the class's run-time override (mix selector), as the router queue.
                        Some(rank) => class.busy_thresholds.for_class(&class.name).rank_is_full(
                            WorkerWithDpRank::new(target.worker_id, rank),
                            WorkerCapacity::of(config),
                            worker_loads,
                        ),
                        None => class.busy_thresholds.for_class(&class.name).worker_is_full(
                            target.worker_id,
                            config,
                            worker_loads,
                        ),
                    })
            }
            None => Self::all_workers_busy_with(worker_loads, workers, class),
        }
    }

    fn all_workers_busy_with(
        worker_loads: &FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
        workers_with_configs: &HashMap<WorkerId, ReplayWorkerConfig>,
        class: &PolicyClassConfig,
    ) -> bool {
        let limits = class.busy_thresholds.for_class(&class.name);
        workers_with_configs
            .iter()
            .all(|(&worker_id, config)| limits.worker_is_full(worker_id, config, worker_loads))
    }

    fn snapshot_for(&self, request: &mut PendingRequest) -> QueueSnapshot {
        let scheduling_request = request.scheduling_request(FxHashMap::default());
        let cached_tokens = SchedulingContext::new(&scheduling_request, &self.workers_with_configs)
            .best_cached_tokens();
        request.token_seq = scheduling_request.token_seq;
        request.overlap = scheduling_request.overlap;
        request.kv_transfer_candidates = scheduling_request.kv_transfer_candidates;
        QueueSnapshot::new(request.isl_tokens, cached_tokens)
    }

    fn prefill_load_hint_for(
        &self,
        isl_tokens: usize,
        cached_tokens: usize,
        track_prefill_tokens: bool,
    ) -> Option<PrefillLoadHint> {
        if !track_prefill_tokens {
            return None;
        }

        let prefix = cached_tokens.min(isl_tokens);
        let effective_isl = isl_tokens.saturating_sub(prefix);
        if effective_isl == 0 {
            return None;
        }

        let expected_prefill_duration = match &self.prefill_load_estimator {
            Some(estimator) => match estimator.predict_prefill_duration(1, effective_isl, prefix) {
                Ok(expected_prefill_duration) => Some(expected_prefill_duration),
                Err(error) => {
                    tracing::warn!(
                        effective_isl,
                        prefix,
                        "failed to predict replay prefill duration for active load tracking: {error}"
                    );
                    None
                }
            },
            None => None,
        };

        Some(PrefillLoadHint {
            initial_effective_prefill_tokens: effective_isl,
            expected_prefill_duration,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::{
        KvBlockPlan, KvTransferCandidates, OverlapSignals, ReplayRequestFields, merge_candidates,
        retain_cost,
    };
    use aisimulate_core::replay::KvAction;
    use std::sync::Arc;
    use std::time::Duration;

    use dynamo_admission::host::RouterQueuePolicy;
    use dynamo_kv_events::event::{
        KvCacheEvent, KvCacheEventData, KvCacheStoreData, KvCacheStoredBlockData, RouterEvent,
    };
    use dynamo_kv_events::residency::StorageTier;
    use dynamo_kv_hashing::routing::{BlockHashOptions, compute_block_hash_for_seq};
    use dynamo_kv_hashing::{ExternalSequenceBlockHash, LocalBlockHash};
    use dynamo_kv_router::config::KvRouterConfig;
    use dynamo_router_protocol::worker::{WorkerId, WorkerWithDpRank};
    use dynamo_scheduler::host::{
        PrefillLoadEstimator, RouterPrefillLoadModel, TrackingHashAlgorithm,
    };
    use rustc_hash::FxHashMap;
    use serde_json::Value;
    use tempfile::NamedTempFile;
    use uuid::Uuid;

    use super::{
        KvReplayMetadata, KvRouterPlacement, OfflineReplayRouter, ReplayRequestHashes,
        SyncReplayIndexer, WorkerAdmission,
    };
    use crate::common::protocols::{DirectRequest, MockerConfig};
    use crate::replay::ReplayPrefillLoadEstimator;
    use aisimulate_core::replay::{
        KvHints, Placement, PlacementDecision, PlacementPolicy, ReplayPromptTokenSource,
        ReplayRequestContext, WorkerTopology,
    };
    use dynamo_kv_indexer::kv_hints::KvTransferCandidateSource;

    #[rstest::rstest]
    #[case(2, 1)]
    #[case(1, 2)]
    fn native_affinity_waits_for_dispatch_and_queues_only_on_its_bound_rank(
        #[case] num_workers: usize,
        #[case] dp_size: u32,
    ) {
        use super::KvRouterPlacement as Policy;
        use aisimulate_core::replay::PlacementPolicy;
        let mut args = queueing_args();
        args.dp_size = dp_size;
        let mut policy = Policy::new_with_selector_seed(
            &args,
            Some(queueing_router_config()),
            None,
            &topology(num_workers, dp_size, false),
            Some(0),
        )
        .unwrap()
        .with_conversation_affinity(Some(crate::replay::ConversationAffinityConfig {
            ttl_seconds: 1.0,
            ..Default::default()
        }))
        .unwrap();
        let submit = |policy: &mut Policy, id, session: &str, now| {
            policy
                .router
                .on_request_arrival_for_session(
                    &request(id, id as u32),
                    None,
                    Some(session.into()),
                    now,
                )
                .unwrap()
                .admissions
        };
        let commit = |policy: &mut Policy, id, now| {
            <Policy as PlacementPolicy<DirectRequest>>::dispatch_committed(policy, id, now)
                .unwrap();
        };
        let advance = |policy: &mut Policy, now| {
            <Policy as PlacementPolicy<DirectRequest>>::advance_clock(policy, now).unwrap()
        };
        let terminal = |policy: &mut Policy, id, now| {
            <Policy as PlacementPolicy<DirectRequest>>::request_terminal(policy, id, now).unwrap()
        };
        let first = submit(&mut policy, 1, "a", 0.0)[0];
        assert!(submit(&mut policy, 2, "a", 0.0).is_empty());
        assert_eq!(policy.router.affinity_waiting.len(), 1);
        commit(&mut policy, first.uuid, 0.0);
        assert!(advance(&mut policy, 0.0).is_empty());
        // An idle unrelated worker cannot override a binding to a busy rank.
        assert_eq!(policy.router.pending_count(), 1);
        let other = submit(&mut policy, 3, "b", 0.0);
        let other = if other.is_empty() {
            advance(&mut policy, 0.0)
        } else {
            policy.placements(other)
        };
        assert_eq!(other.len(), 1);
        assert_ne!(other[0].scheduler_id, first.worker_idx);
        commit(&mut policy, other[0].request_id, 0.0);
        let released = terminal(&mut policy, first.uuid, 10.0);
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].scheduler_id, first.worker_idx);
        commit(&mut policy, released[0].request_id, 10.0);
        terminal(&mut policy, released[0].request_id, 20.0);
        let key = policy
            .router
            .affinity_id(&request(4, 4), Some("a"))
            .unwrap()
            .unwrap();
        for (now, bound) in [(1019.0, true), (1020.0, false)] {
            advance(&mut policy, now);
            assert_eq!(
                policy
                    .router
                    .affinity
                    .as_ref()
                    .unwrap()
                    .1
                    .query_target(&key, None)
                    .unwrap()
                    .is_some(),
                bound
            );
        }
        // Cancellation during tokenization must release an uncommitted initializer.
        let canceled = submit(&mut policy, 4, "a", 1020.0)[0];
        assert!(submit(&mut policy, 5, "a", 1020.0).is_empty());
        terminal(&mut policy, canceled.uuid, 1020.0);
        let retry = advance(&mut policy, 1020.0);
        assert_eq!(retry.len(), 1);
        assert_eq!(retry[0].request_id, Uuid::from_u128(5));
        <Policy as PlacementPolicy<DirectRequest>>::dispatch_aborted(
            &mut policy,
            retry[0].request_id,
            1020.0,
        )
        .unwrap();
        assert_eq!(policy.router.pending_count(), 0);
        assert!(policy.router.affinity_staged.is_empty());
    }

    #[test]
    fn unknown_policy_class_fails_at_class_entry_and_releases_session_affinity() {
        use super::KvRouterPlacement as Policy;
        use aisimulate_core::replay::PlacementPolicy;
        use dynamo_scheduler::host::KvSchedulerError;
        let mut policy = Policy::new_with_selector_seed(
            &queueing_args(),
            Some(queueing_router_config()),
            None,
            &topology(1, 1, false),
            Some(0),
        )
        .unwrap()
        .with_conversation_affinity(Some(Default::default()))
        .unwrap();
        let arrive = |policy: &mut Policy, id: u128, policy_class: Option<&str>| {
            let mut request = request(id, id as u32);
            request.policy_class = policy_class.map(str::to_string);
            policy
                .router
                .on_request_arrival_for_session(&request, None, Some("a".into()), 0.0)
        };
        let is_unknown = |error: anyhow::Error| {
            matches!(
                error.downcast_ref::<KvSchedulerError>(),
                Some(KvSchedulerError::UnknownPolicyClass(name)) if name == "unknown"
            )
        };

        // Entering its class at arrival, the request fails and releases its session hold,
        // so the next request for the session is not blocked by it.
        assert!(is_unknown(
            arrive(&mut policy, 1, Some("unknown")).unwrap_err()
        ));
        let first = arrive(&mut policy, 2, None).unwrap().admissions;
        assert_eq!(first.len(), 1);
        // Behind that uncommitted request it waits for the session, and fails only when it
        // enters its class.
        assert!(
            arrive(&mut policy, 3, Some("unknown"))
                .unwrap()
                .admissions
                .is_empty()
        );
        assert_eq!(policy.router.affinity_waiting.len(), 1);
        <Policy as PlacementPolicy<DirectRequest>>::dispatch_committed(
            &mut policy,
            first[0].uuid,
            0.0,
        )
        .unwrap();
        assert!(is_unknown(
            <Policy as PlacementPolicy<DirectRequest>>::advance_clock(&mut policy, 0.0)
                .unwrap_err()
        ));
        assert!(policy.router.affinity_waiting.is_empty());
        assert_eq!(policy.router.pending_count(), 0);
    }

    #[test]
    fn affinity_queue_cost_uses_only_the_bound_rank_cache() {
        use aisimulate_core::replay::PlacementPolicy;
        let mut args = queueing_args();
        args.dp_size = 2;
        let mut policy = super::KvRouterPlacement::new_with_selector_seed(
            &args,
            Some(queueing_router_config_with_policy(RouterQueuePolicy::Wspt)),
            None,
            &topology(1, 2, false),
            Some(0),
        )
        .unwrap()
        .with_conversation_affinity(Some(Default::default()))
        .unwrap();
        let first = policy
            .router
            .on_request_arrival_for_session(&request(1, 1), None, Some("a".into()), 0.0)
            .unwrap()
            .admissions[0];
        <super::KvRouterPlacement as PlacementPolicy<DirectRequest>>::dispatch_committed(
            &mut policy,
            first.uuid,
            0.0,
        )
        .unwrap();
        policy
            .router
            .on_kv_events(vec![store_event_for_rank(
                0,
                1 - first.worker_idx as u32,
                1,
                77,
                StorageTier::Device,
            )])
            .unwrap();
        for (id, tokens, hashes) in [
            (
                2,
                64,
                Some(ReplayRequestHashes {
                    local_block_hashes: vec![77],
                    sequence_hashes: vec![77],
                }),
            ),
            (3, 16, None),
        ] {
            assert!(
                policy
                    .router
                    .on_request_arrival_for_session(
                        &request_with_priorities(id, id as u32, tokens, 0, 0),
                        hashes,
                        Some("a".into()),
                        0.0,
                    )
                    .unwrap()
                    .admissions
                    .is_empty()
            );
        }
        let released = policy
            .router
            .on_request_completed(first.uuid, 1.0)
            .unwrap()
            .admissions;
        assert_eq!(released[0].uuid, Uuid::from_u128(3));
        assert_eq!(released[0].worker_idx, first.worker_idx);
    }

    #[rstest::rstest]
    #[case(false, false)]
    #[case(false, true)]
    #[case(true, false)]
    #[case(true, true)]
    fn affinity_waiter_refreshes_cache_before_admission(
        #[case] evict: bool,
        #[case] authored_hashes: bool,
    ) {
        use aisimulate_core::replay::PlacementPolicy;
        let mut policy = super::KvRouterPlacement::new_with_selector_seed(
            &queueing_args(),
            Some(queueing_router_config()),
            None,
            &topology(1, 1, false),
            Some(0),
        )
        .unwrap()
        .with_conversation_affinity(Some(Default::default()))
        .unwrap();
        let waiter = request(3, 7);
        let hash = if authored_hashes {
            77
        } else {
            compute_block_hash_for_seq(&waiter.tokens, 64, BlockHashOptions::default())[0].0
        };
        let stored = || store_event(0, 1, hash, StorageTier::Device);
        if evict {
            policy.router.on_kv_events(vec![stored()]).unwrap();
        }
        policy
            .router
            .on_request_arrival_for_session(&request(1, 1), None, Some("busy".into()), 0.0)
            .unwrap();
        assert!(
            policy
                .router
                .on_request_arrival_for_session(
                    &request_with_priorities(2, 2, 16, 0, 0),
                    None,
                    Some("a".into()),
                    0.0,
                )
                .unwrap()
                .admissions
                .is_empty()
        );
        let hashes = authored_hashes.then(|| ReplayRequestHashes {
            local_block_hashes: vec![hash],
            sequence_hashes: vec![hash],
        });
        assert!(
            policy
                .router
                .on_request_arrival_for_session(&waiter, hashes, Some("a".into()), 0.0,)
                .unwrap()
                .admissions
                .is_empty()
        );
        assert_eq!(policy.router.affinity_waiting.len(), 1);
        let mut event = stored();
        if evict {
            event.event.data = KvCacheEventData::Cleared;
        }
        policy.router.on_kv_events(vec![event]).unwrap();
        let initializer = policy
            .router
            .on_request_completed(Uuid::from_u128(1), 1.0)
            .unwrap()
            .admissions;
        assert_eq!(initializer[0].uuid, Uuid::from_u128(2));
        <super::KvRouterPlacement as PlacementPolicy<DirectRequest>>::dispatch_committed(
            &mut policy,
            initializer[0].uuid,
            1.0,
        )
        .unwrap();
        let released = <super::KvRouterPlacement as PlacementPolicy<DirectRequest>>::advance_clock(
            &mut policy,
            1.0,
        )
        .unwrap();
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].request_id, Uuid::from_u128(3));
        assert_eq!(
            released[0].reported_overlap_tokens,
            if evict { 0 } else { 64 }
        );
    }

    #[rstest::rstest]
    #[case(false)]
    #[case(true)]
    fn affinity_clock_does_not_drain_on_decay_and_abort_releases_capacity(#[case] affinity: bool) {
        use aisimulate_core::replay::PlacementPolicy;
        let mut config = queueing_router_config();
        config.router_prefill_load_model = RouterPrefillLoadModel::Ais;
        let mut policy = super::KvRouterPlacement::new_with_selector_seed(
            &queueing_args(),
            Some(config),
            Some(estimator(Duration::from_secs(10))),
            &topology(1, 1, false),
            Some(0),
        )
        .unwrap()
        .with_conversation_affinity(affinity.then(Default::default))
        .unwrap();
        for (id, session) in [(1, "a"), (2, "b")] {
            let result = policy
                .router
                .on_request_arrival_for_session(
                    &request(id, id as u32),
                    None,
                    Some(session.into()),
                    0.0,
                )
                .unwrap();
            assert_eq!(result.admissions.len(), usize::from(id == 1));
        }
        assert!(
            <super::KvRouterPlacement as PlacementPolicy<DirectRequest>>::advance_clock(
                &mut policy,
                0.0
            )
            .unwrap()
            .is_empty()
        );
        assert!(
            <super::KvRouterPlacement as PlacementPolicy<DirectRequest>>::advance_clock(
                &mut policy,
                10_000.0
            )
            .unwrap()
            .is_empty()
        );
        assert_eq!(policy.router.pending_count(), 1);
        <super::KvRouterPlacement as PlacementPolicy<DirectRequest>>::dispatch_aborted(
            &mut policy,
            Uuid::from_u128(1),
            10_000.0,
        )
        .unwrap();
        assert!(
            policy
                .router
                .debug_snapshot(10_000.0)
                .active_blocks_by_worker
                .iter()
                .all(|(_, blocks)| *blocks == 0)
        );
        let released = <super::KvRouterPlacement as PlacementPolicy<DirectRequest>>::advance_clock(
            &mut policy,
            10_000.0,
        )
        .unwrap();
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].request_id, Uuid::from_u128(2));
    }

    #[test]
    fn affinity_table_full_routes_without_affinity() {
        use super::{KvRouterPlacement as Policy, SessionAffinity, SessionAffinityConfig};
        use aisimulate_core::replay::PlacementPolicy;
        let mut policy = Policy::new_with_selector_seed(
            &replay_args(),
            None,
            None,
            &topology(1, 1, false),
            Some(0),
        )
        .unwrap();
        let mut config = SessionAffinityConfig::new(Duration::from_secs(60));
        config.max_entries = 1;
        policy.router.affinity = Some((
            Default::default(),
            SessionAffinity::with_manual_clock(config, policy.router.decay_time_epoch).unwrap(),
        ));
        for (id, session) in [(1, "a"), (2, "b"), (3, "a")] {
            let admitted = policy
                .router
                .on_request_arrival_for_session(
                    &request(id, id as u32),
                    None,
                    Some(session.into()),
                    id as f64,
                )
                .unwrap()
                .admissions;
            assert_eq!(admitted.len(), 1);
            assert_eq!(
                policy
                    .router
                    .affinity_staged
                    .contains_key(&admitted[0].uuid),
                session == "a"
            );
            <Policy as PlacementPolicy<DirectRequest>>::dispatch_committed(
                &mut policy,
                admitted[0].uuid,
                id as f64,
            )
            .unwrap();
            <Policy as PlacementPolicy<DirectRequest>>::request_terminal(
                &mut policy,
                admitted[0].uuid,
                id as f64,
            )
            .unwrap();
        }
    }

    struct FixedPrefillLoadEstimator {
        duration: Duration,
    }

    impl PrefillLoadEstimator for FixedPrefillLoadEstimator {
        fn predict_prefill_duration(
            &self,
            _batch_size: usize,
            _effective_isl: usize,
            _prefix: usize,
        ) -> anyhow::Result<Duration> {
            Ok(self.duration)
        }
    }

    fn replay_args() -> MockerConfig {
        MockerConfig::from_value(serde_json::json!({
            "engine": {
                "block_size": 64,
                "max_num_batched_tokens": 256
            }
        }))
        .unwrap()
    }

    fn queueing_args() -> MockerConfig {
        MockerConfig::from_value(serde_json::json!({
            "engine": {
                "block_size": 64,
                "max_num_batched_tokens": 64
            }
        }))
        .unwrap()
    }

    fn router_config() -> KvRouterConfig {
        KvRouterConfig {
            router_track_prefill_tokens: true,
            router_prefill_load_model: RouterPrefillLoadModel::Ais,
            ..KvRouterConfig::default()
        }
    }

    fn queueing_router_config() -> KvRouterConfig {
        queueing_router_config_with_policy(RouterQueuePolicy::Fcfs)
    }

    fn queueing_router_config_with_policy(policy: RouterQueuePolicy) -> KvRouterConfig {
        KvRouterConfig {
            router_queue_threshold: Some(0.5),
            router_queue_policy: policy,
            ..KvRouterConfig::default()
        }
    }

    fn estimator(duration: Duration) -> ReplayPrefillLoadEstimator {
        Arc::new(FixedPrefillLoadEstimator { duration })
    }

    fn request(uuid: u128, token: u32) -> DirectRequest {
        request_with_priorities(uuid, token, 64, 0, 0)
    }

    fn request_with_priorities(
        uuid: u128,
        token: u32,
        input_tokens: usize,
        priority: i32,
        strict_priority: u32,
    ) -> DirectRequest {
        DirectRequest {
            tokens: vec![token; input_tokens],
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(uuid)),
            dp_rank: 0,
            preferred_dp_rank: None,
            preferred_prefill_dp_rank: None,
            kv_hints: None,
            arrival_timestamp_ms: Some(0.0),
            priority,
            strict_priority,
            policy_class: None,
            replay_context: None,
            images: Vec::new(),
        }
    }

    fn kv_actions_router(yaml: &str, workers: usize) -> OfflineReplayRouter {
        let path = std::env::temp_dir().join(format!(
            "kv-actions-{}-{}.yaml",
            std::process::id(),
            Uuid::new_v4()
        ));
        std::fs::write(&path, yaml).unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(path.display().to_string()),
            ..KvRouterConfig::default()
        };
        let router = OfflineReplayRouter::new(&replay_args(), Some(config), None, workers).unwrap();
        std::fs::remove_file(path).unwrap();
        router
    }

    fn tokens_request(uuid: u128, tokens: Vec<u32>) -> DirectRequest {
        let mut request = request(uuid, 0);
        request.tokens = tokens;
        request
    }

    fn queued_actions(router: &mut OfflineReplayRouter) -> Vec<(usize, KvAction)> {
        router
            .take_kv_actions()
            .into_iter()
            .flat_map(|message| {
                let scheduler_id = message.scheduler_id;
                message
                    .actions
                    .into_iter()
                    .map(move |request| (scheduler_id, request.action))
            })
            .collect()
    }

    #[test]
    fn invalidate_oracle_drops_a_streams_dead_blocks_on_every_rank() {
        let mut router = kv_actions_router("kv_actions:\n  invalidate: oracle\n", 2);
        let first: Vec<u32> = (0..192).collect();
        let first_hashes = ReplayRequestHashes::from_tokens(&first, router.block_size);
        router
            .on_request_arrival_for_session(
                &tokens_request(1, first),
                Some(first_hashes.clone()),
                Some("s".to_string()),
                0.0,
            )
            .unwrap();
        assert!(
            queued_actions(&mut router).is_empty(),
            "a stream's first request kills nothing"
        );

        // The next request keeps the first block and replaces the rest (a compaction).
        let mut second: Vec<u32> = (0..64).collect();
        second.extend(1000..1128);
        let second_hashes = ReplayRequestHashes::from_tokens(&second, router.block_size);
        assert_eq!(
            first_hashes.sequence_hashes[0],
            second_hashes.sequence_hashes[0]
        );
        router
            .on_request_arrival_for_session(
                &tokens_request(2, second),
                Some(second_hashes),
                Some("s".to_string()),
                1.0,
            )
            .unwrap();
        let dead = first_hashes.sequence_hashes[1..].to_vec();
        assert_eq!(
            queued_actions(&mut router),
            vec![
                (
                    0,
                    KvAction::Invalidate {
                        block_hashes: dead.clone()
                    }
                ),
                (1, KvAction::Invalidate { block_hashes: dead }),
            ]
        );

        // Another stream's request is compared with its own history only.
        let other: Vec<u32> = (5000..5064).collect();
        let other_hashes = ReplayRequestHashes::from_tokens(&other, router.block_size);
        router
            .on_request_arrival_for_session(
                &tokens_request(3, other),
                Some(other_hashes),
                Some("t".to_string()),
                2.0,
            )
            .unwrap();
        assert!(queued_actions(&mut router).is_empty());
    }

    #[test]
    fn invalidate_marker_waits_for_the_compaction_signal() {
        let mut router = kv_actions_router("kv_actions:\n  invalidate: marker\n", 1);
        for (uuid, tokens) in [
            (1, (0..192).collect::<Vec<u32>>()),
            (2, (900..964).collect()),
        ] {
            let hashes = ReplayRequestHashes::from_tokens(&tokens, router.block_size);
            router
                .on_request_arrival_for_session(
                    &tokens_request(uuid, tokens),
                    Some(hashes),
                    Some("s".to_string()),
                    0.0,
                )
                .unwrap();
        }
        // No request carried `compaction`, so nothing is invalidated.
        assert!(queued_actions(&mut router).is_empty());
    }

    #[test]
    fn session_end_frees_only_the_blocks_no_live_stream_holds() {
        let mut router = kv_actions_router("kv_actions:\n  free_at_session_end: true\n", 2);
        let worker = WorkerWithDpRank::new(0, 0);
        let plan = |stream: &str, sequence: Vec<u64>, session_final| KvBlockPlan {
            stream_id: stream.to_string(),
            local: sequence.clone(),
            sequence,
            compaction: false,
            think_ms: None,
            session_final,
        };
        // A parent's context [1, 2, 3] and its subagent reading [1, 2] plus its own [9].
        router.free_track("parent", &[1, 2]);
        router.free_track("parent", &[1, 2, 3]);
        router.free_track("child", &[1, 2, 9]);
        // An unfinished stream frees nothing.
        router.kv_actions.admitted.insert(
            Uuid::from_u128(1),
            (0, worker, plan("child", vec![1, 2, 9], false)),
        );
        router.demote_finished(Uuid::from_u128(1), 0.0).unwrap();
        assert!(queued_actions(&mut router).is_empty());
        // The parent ends first: only its own block 3 goes, on every rank; 1 and 2 stay for the
        // child.
        router.kv_actions.admitted.insert(
            Uuid::from_u128(2),
            (0, worker, plan("parent", vec![1, 2, 3], true)),
        );
        router.demote_finished(Uuid::from_u128(2), 0.0).unwrap();
        let invalidate = |hashes: Vec<u64>| KvAction::Invalidate {
            block_hashes: hashes,
        };
        assert_eq!(
            queued_actions(&mut router),
            vec![(0, invalidate(vec![3])), (1, invalidate(vec![3]))]
        );
        // Then the child ends: the shared blocks are now unheld.
        router.kv_actions.admitted.insert(
            Uuid::from_u128(3),
            (0, worker, plan("child", vec![1, 2, 9], true)),
        );
        router.demote_finished(Uuid::from_u128(3), 0.0).unwrap();
        assert_eq!(
            queued_actions(&mut router),
            vec![
                (0, invalidate(vec![1, 2, 9])),
                (1, invalidate(vec![1, 2, 9]))
            ]
        );
        let stats = &router.kv_actions.stats;
        assert_eq!(
            (
                stats.session_end_frees,
                stats.session_end_freed_blocks,
                stats.session_end_shared_blocks
            ),
            (2, 4, 2)
        );
        assert!(router.kv_actions.free_refs.is_empty());
        assert!(router.kv_actions.free_streams.is_empty());
    }

    #[test]
    fn demote_needs_a_long_think_and_a_full_g1_then_prefetches_before_the_return() {
        let mut router = kv_actions_router(
            "kv_actions:\n  demote:\n    prefetch: {lead_s: 1.0, load_gbs: 1.0, block_bytes: 1.0e9}\n",
            1,
        );
        let worker = WorkerWithDpRank::new(0, 0);
        let capacity = router.workers_with_configs[&0].total_kv_blocks;
        let plan = |think_ms| KvBlockPlan {
            stream_id: "s".to_string(),
            sequence: vec![7, 8],
            local: vec![70, 80],
            compaction: false,
            think_ms: Some(think_ms),
            session_final: false,
        };
        let admit = |router: &mut OfflineReplayRouter, uuid: u128, think_ms| {
            router
                .kv_actions
                .admitted
                .insert(Uuid::from_u128(uuid), (0, worker, plan(think_ms)));
        };

        // G1 at 80% is not above the threshold.
        router
            .kv_actions
            .g1_resident
            .insert(worker, capacity * 8 / 10);
        admit(&mut router, 1, 60_000);
        router.demote_finished(Uuid::from_u128(1), 0.0).unwrap();
        assert!(queued_actions(&mut router).is_empty());

        // Full G1 but a short think: kept.
        router.kv_actions.g1_resident.insert(worker, capacity);
        admit(&mut router, 2, 19_000);
        router.demote_finished(Uuid::from_u128(2), 0.0).unwrap();
        assert!(queued_actions(&mut router).is_empty());

        // Full G1 and a long think: demoted, with a prefetch due the lead plus the transfer
        // (two blocks of 1 GB at 1 GB/s) before the predicted return.
        admit(&mut router, 3, 60_000);
        router.demote_finished(Uuid::from_u128(3), 1_000.0).unwrap();
        assert_eq!(
            queued_actions(&mut router),
            vec![(
                0,
                KvAction::Demote {
                    block_hashes: vec![7, 8]
                }
            )]
        );
        assert_eq!(router.kv_actions.next_prefetch_ms(), Some(58_000.0));
        router.kv_actions.fire_prefetches(57_999.0);
        assert!(queued_actions(&mut router).is_empty());
        router.kv_actions.fire_prefetches(58_000.0);
        assert_eq!(
            queued_actions(&mut router),
            vec![(
                0,
                KvAction::Prefetch {
                    block_hashes: vec![7, 8],
                    local_hashes: vec![70, 80]
                }
            )]
        );
        assert_eq!(router.kv_actions.next_prefetch_ms(), None);
    }

    #[test]
    fn retain_value_rebuild_cost_carries_the_class_s_cost_model_to_the_engine() {
        let config = |json| -> dynamo_kv_router::scheduling::KvRetainValueConfig {
            serde_json::from_value(json).unwrap()
        };
        // The default (session) value carries no cost model: the engine values as before.
        assert_eq!(
            retain_cost(&config(serde_json::json!({})), Some("priority")),
            None
        );
        let cost = retain_cost(
            &config(serde_json::json!({"value": "rebuild_cost", "peers": true})),
            Some("standard"),
        )
        .unwrap();
        assert_eq!(
            cost,
            aisimulate_core::replay::KvRetainCost {
                rebuild_us_per_block: 3_000,
                ttft_target_ms: 8_000,
                stall_milli: 500,
                peers: true,
                fetch_latency_ms: 100,
                fetch_ns_per_block: 5_700,
            }
        );
        // A class without a TTFT target: 30 s.
        let other = retain_cost(
            &config(serde_json::json!({"value": "rebuild_cost"})),
            Some("x"),
        )
        .unwrap();
        assert_eq!((other.ttft_target_ms, other.peers), (30_000, false));
    }

    #[test]
    fn retain_until_return_keeps_blocks_until_the_expected_return_plus_the_margin() {
        let fields = |think_s: &str, last: bool| -> ReplayRequestFields {
            serde_json::from_value(serde_json::json!({
                "request_id": "r", "session_final": last,
                "agent_headers": {"x-dynamo-hint-next-think-s": [think_s]}}))
            .unwrap()
        };
        let fixed = kv_actions_router(
            "kv_actions:\n  retain:\n    expected: true\n    ttl_seconds: {priority: 3600}\n",
            1,
        );
        let hinted = kv_actions_router(
            "kv_actions:\n  retain:\n    expected: true\n    ttl_seconds: {priority: 3600}\n    until_return_margin_s: 30\n",
            1,
        );
        // A fixed TTL protects every expected session for the whole TTL.
        assert_eq!(
            fixed
                .retain_ttl_ms(Some("priority"), Some(&fields("12.5", false)))
                .unwrap(),
            Some(3_600_000)
        );
        // Until the return: the hint plus the margin, at most the class TTL.
        assert_eq!(
            hinted
                .retain_ttl_ms(Some("priority"), Some(&fields("12.5", false)))
                .unwrap(),
            Some(42_500)
        );
        assert_eq!(
            hinted
                .retain_ttl_ms(Some("priority"), Some(&fields("3590", false)))
                .unwrap(),
            Some(3_600_000)
        );
        // Not expected back within the TTL, a session's last request, or a class without a TTL: kept not.
        assert_eq!(
            hinted
                .retain_ttl_ms(Some("priority"), Some(&fields("7200", false)))
                .unwrap(),
            None
        );
        assert_eq!(
            hinted
                .retain_ttl_ms(Some("priority"), Some(&fields("12.5", true)))
                .unwrap(),
            None
        );
        assert_eq!(
            hinted
                .retain_ttl_ms(Some("batch"), Some(&fields("12.5", false)))
                .unwrap(),
            None
        );
    }

    #[test]
    fn a_delayed_demote_waits_delay_s_and_the_stream_s_next_request_cancels_it() {
        let mut router = kv_actions_router("kv_actions:\n  demote: {delay_s: 2.0}\n", 1);
        let worker = WorkerWithDpRank::new(0, 0);
        let capacity = router.workers_with_configs[&0].total_kv_blocks;
        router.kv_actions.g1_resident.insert(worker, capacity);
        let plan = KvBlockPlan {
            stream_id: "s".to_string(),
            sequence: vec![7, 8],
            local: vec![70, 80],
            compaction: false,
            think_ms: Some(60_000),
            session_final: false,
        };
        router
            .kv_actions
            .admitted
            .insert(Uuid::from_u128(1), (0, worker, plan.clone()));
        router.demote_finished(Uuid::from_u128(1), 1_000.0).unwrap();
        // Nothing at the request's end; the demote is due delay_s later.
        assert!(queued_actions(&mut router).is_empty());
        assert_eq!(router.kv_actions.next_prefetch_ms(), Some(3_000.0));
        router.kv_actions.fire_prefetches(2_999.0);
        assert!(queued_actions(&mut router).is_empty());
        router.kv_actions.fire_prefetches(3_000.0);
        assert_eq!(
            queued_actions(&mut router),
            vec![(
                0,
                KvAction::Demote {
                    block_hashes: vec![7, 8]
                }
            )]
        );
        assert_eq!(router.kv_actions.stats.demotes_sent, 1);

        // The stream's next request arrives before the delay ends: no demote.
        router
            .kv_actions
            .admitted
            .insert(Uuid::from_u128(2), (0, worker, plan.clone()));
        router
            .demote_finished(Uuid::from_u128(2), 10_000.0)
            .unwrap();
        let tokens: Vec<u32> = (0..128).collect();
        let hashes = ReplayRequestHashes::from_tokens(&tokens, router.block_size);
        router
            .on_request_arrival_for_session(
                &tokens_request(3, tokens),
                Some(hashes),
                Some("s".to_string()),
                11_000.0,
            )
            .unwrap();
        assert!(router.kv_actions.demotes.is_empty());
        router.kv_actions.fire_prefetches(20_000.0);
        assert!(queued_actions(&mut router).is_empty());
        assert_eq!(router.kv_actions.stats.demotes_sent, 1);
    }

    #[test]
    fn g1_residency_counts_device_events_only() {
        let mut router = kv_actions_router("kv_actions:\n  demote: {}\n", 1);
        router
            .on_kv_events(vec![
                store_event(0, 1, 11, StorageTier::Device),
                store_event(0, 2, 12, StorageTier::Device),
                store_event(0, 3, 13, StorageTier::HostPinned),
            ])
            .unwrap();
        assert_eq!(
            router.kv_actions.g1_resident[&WorkerWithDpRank::new(0, 0)],
            2
        );
    }

    #[test]
    fn unhandled_request_fields_are_rejected_instead_of_dropped() {
        let mut router = OfflineReplayRouter::new(&replay_args(), None, None, 1).unwrap();
        let mut request = request(1, 7);
        request.kv_hints = Some(aisimulate_core::replay::AuthoredKvHints {
            source_worker: 0,
            source_dp_rank: 0,
        });
        let error = router.on_request_arrival(&request, None, 0.0).unwrap_err();
        assert!(error.to_string().contains("authored KV fetch hints"));
        assert_eq!(router.pending_count(), 0);
        request.kv_hints = None;
        request.images.push(aisimulate_core::engine::ImageSpec {
            identity: 1,
            token_start: 0,
            token_end: 1,
            encoder: aisimulate_core::engine::EncoderShape {
                sequences: 1,
                patch_tokens: 1,
                transformer_tokens: 1,
                output_tokens: 1,
            },
            embedding_bytes: 1,
        });
        let error = router.on_request_arrival(&request, None, 0.0).unwrap_err();
        assert!(error.to_string().contains("image requests"));
        assert_eq!(router.pending_count(), 0);
        request.images.clear();
        assert_eq!(
            router
                .on_request_arrival(&request, None, 0.0)
                .unwrap()
                .admissions
                .len(),
            1
        );
    }

    #[test]
    fn length_only_execution_tokens_are_rejected_without_replay_hashes() {
        let router = OfflineReplayRouter::new(&replay_args(), None, None, 1).unwrap();
        let mut request = request(1, 7);
        request.replay_context = Some(ReplayRequestContext {
            agentic: None,
            authored_id: "length-only".into(),
            session_id: None,
            turn_index: None,
            metadata: Value::Null,
            prompt_token_source: ReplayPromptTokenSource::LengthOnlySynthetic,
        });

        let error =
            match router.build_pending_request(&request, request.max_output_tokens, None, None) {
                Ok(_) => panic!("length-only request unexpectedly reached KV placement"),
                Err(error) => error,
            };
        assert!(
            error
                .to_string()
                .contains("requires authored prompt token IDs or replay hashes"),
            "{error}"
        );
    }

    fn store_event(
        worker_id: WorkerId,
        event_id: u64,
        tokens_hash: u64,
        storage_tier: StorageTier,
    ) -> RouterEvent {
        store_event_for_rank(worker_id, 0, event_id, tokens_hash, storage_tier)
    }

    fn store_event_for_rank(
        worker_id: WorkerId,
        dp_rank: u32,
        event_id: u64,
        tokens_hash: u64,
        storage_tier: StorageTier,
    ) -> RouterEvent {
        RouterEvent::with_storage_tier(
            worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(event_id),
                        tokens_hash: LocalBlockHash(tokens_hash),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank,
            },
            storage_tier,
        )
    }

    #[test]
    fn session_identity_reaches_scheduling_request() {
        let router = OfflineReplayRouter::new(&replay_args(), None, None, 1).unwrap();
        let request = request(1, 7);
        let mut pending = router
            .build_pending_request(
                &request,
                request.max_output_tokens,
                None,
                Some("session-a".to_string()),
            )
            .unwrap();
        let scheduling_request = pending.scheduling_request(FxHashMap::default());

        assert_eq!(
            scheduling_request
                .session_context
                .as_ref()
                .map(|context| context.session_id()),
            Some("session-a")
        );
    }

    #[test]
    fn keyed_replay_hashes_derive_tracking_identities_from_tokens() {
        let mut key_file = NamedTempFile::new().unwrap();
        key_file.write_all(&[0x39; 32]).unwrap();
        let config = KvRouterConfig {
            router_tracking_hash: TrackingHashAlgorithm::KeyedXxh3V1,
            router_tracking_key_file: Some(key_file.path().to_path_buf()),
            router_tracking_key_id: Some("2026-01".to_string()),
            ..Default::default()
        };
        let router = OfflineReplayRouter::new(&replay_args(), Some(config), None, 1).unwrap();
        let request = request(1, 7);
        let replay_hashes = ReplayRequestHashes::from_tokens(&request.tokens, router.block_size);
        let expected = router.config.compute_seq_hashes_for_tracking_with_context(
            &router.tracking_hash,
            router.tracking_hash_scope(),
            &request.tokens,
            None,
            BlockHashOptions::default(),
            None,
        );

        let pending = router
            .build_pending_request(
                &request,
                request.max_output_tokens,
                Some(replay_hashes.clone()),
                None,
            )
            .unwrap();

        assert_eq!(pending.token_seq, expected);
        assert_ne!(pending.token_seq, Some(replay_hashes.sequence_hashes));
    }

    #[test]
    fn lower_tier_events_are_indexed_outside_the_primary_tree() {
        let mut indexer = SyncReplayIndexer::new(64);
        let worker = WorkerWithDpRank::new(7, 0);

        indexer
            .apply_event(store_event(7, 1, 101, StorageTier::HostPinned))
            .unwrap();
        assert_eq!(indexer.debug_snapshot().total_cached_blocks, 0);
        let tiered = indexer.find_matches_for_hashes(vec![LocalBlockHash(101)]);
        assert_eq!(tiered.device.overlap_scores.scores.get(&worker), None);
        assert_eq!(
            tiered.lower_tier[&StorageTier::HostPinned]
                .hits
                .get(&worker),
            Some(&1)
        );

        indexer
            .apply_event(store_event(7, 2, 101, StorageTier::Device))
            .unwrap();
        assert_eq!(indexer.debug_snapshot().total_cached_blocks, 1);

        indexer.remove_worker(7);
        let tiered = indexer.find_matches_for_hashes(vec![LocalBlockHash(101)]);
        assert!(tiered.device.overlap_scores.scores.is_empty());
        assert!(
            tiered
                .lower_tier
                .values()
                .all(|matches| matches.hits.is_empty())
        );
    }

    /// One store event covering a whole prompt, as AISimulate relays a
    /// cluster-shared G2 pool's residency (`HostPinned`) to each subscribed rank.
    fn prompt_store_event(
        worker_id: WorkerId,
        event_id: u64,
        local_hashes: &[LocalBlockHash],
        storage_tier: StorageTier,
    ) -> RouterEvent {
        chain_store_event(worker_id, event_id, 0, local_hashes, storage_tier)
    }

    /// Stores `local_hashes` as prompt blocks `first..`, chained under block
    /// `first - 1` so a lower tier can continue a device match.
    fn chain_store_event(
        worker_id: WorkerId,
        event_id: u64,
        first: usize,
        local_hashes: &[LocalBlockHash],
        storage_tier: StorageTier,
    ) -> RouterEvent {
        let block_hash = |index: usize| ExternalSequenceBlockHash(1_000 + index as u64);
        RouterEvent::with_storage_tier(
            worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: first.checked_sub(1).map(block_hash),
                    start_position: None,
                    blocks: local_hashes
                        .iter()
                        .enumerate()
                        .map(|(offset, hash)| KvCacheStoredBlockData {
                            block_hash: block_hash(first + offset),
                            tokens_hash: *hash,
                            mm_extra_info: None,
                        })
                        .collect(),
                }),
                dp_rank: 0,
            },
            storage_tier,
        )
    }

    #[test]
    fn host_pinned_hits_are_weighted_into_offline_placement() {
        let mut router = OfflineReplayRouter::new(&replay_args(), Some(router_config()), None, 2)
            .expect("router construction");
        let target = request_with_priorities(1, 7, 256, 0, 0);
        let local_hashes = compute_block_hash_for_seq(
            &target.tokens,
            router.block_size,
            BlockHashOptions::default(),
        );
        assert_eq!(local_hashes.len(), 4);
        router
            .on_kv_events(vec![prompt_store_event(
                1,
                1,
                &local_hashes,
                StorageTier::HostPinned,
            )])
            .unwrap();

        let effects = router.on_request_arrival(&target, None, 0.0).unwrap();
        // Four G2 blocks at the default host_cache_hit_weight of 0.75 score
        // three effective blocks, counted the same way for best and selected.
        assert_eq!(
            effects.admissions,
            vec![WorkerAdmission {
                uuid: Uuid::from_u128(1),
                worker_idx: 1,
                overlap_blocks: 3,
                best_available_overlap_blocks: 3,
                isl_blocks: 4,
                kv_fetch_source: None,
                retain_ttl_ms: None,
                retain_value: None,
                context_tokens: 0,
            }]
        );
    }

    #[test]
    fn host_pinned_blocks_extend_a_partial_device_match() {
        let mut router = OfflineReplayRouter::new(&replay_args(), Some(router_config()), None, 2)
            .expect("router construction");
        let target = request_with_priorities(1, 7, 256, 0, 0);
        let local_hashes = compute_block_hash_for_seq(
            &target.tokens,
            router.block_size,
            BlockHashOptions::default(),
        );
        assert_eq!(local_hashes.len(), 4);
        // Worker 1 holds block 0 in G1 and blocks 1..4 in G2.
        router
            .on_kv_events(vec![
                chain_store_event(1, 1, 0, &local_hashes[..1], StorageTier::Device),
                chain_store_event(1, 2, 1, &local_hashes[1..], StorageTier::HostPinned),
            ])
            .unwrap();

        let effects = router.on_request_arrival(&target, None, 0.0).unwrap();
        // One G1 block plus three G2 blocks at 0.75 is 3.25, rounded to 3.
        // Without the device-to-host continuation the G2 chain would not match.
        assert_eq!(
            effects.admissions,
            vec![WorkerAdmission {
                uuid: Uuid::from_u128(1),
                worker_idx: 1,
                overlap_blocks: 3,
                best_available_overlap_blocks: 3,
                isl_blocks: 4,
                kv_fetch_source: None,
                retain_ttl_ms: None,
                retain_value: None,
                context_tokens: 0,
            }]
        );
    }

    #[test]
    fn test_prefill_load_estimator_decays_offline_router_active_tokens() {
        let mut router = OfflineReplayRouter::new(
            &replay_args(),
            Some(router_config()),
            Some(estimator(Duration::from_secs(10))),
            1,
        )
        .unwrap();

        let effects = router
            .on_request_arrival(&request(1, 7), None, 0.0)
            .unwrap();
        assert_eq!(effects.admissions.len(), 1);
        assert_eq!(
            router.debug_snapshot(0.0).active_tokens_by_worker,
            vec![(0, 64)]
        );
        assert_eq!(
            router.debug_snapshot(5_000.0).active_tokens_by_worker,
            vec![(0, 32)]
        );
        assert_eq!(
            router.debug_snapshot(10_000.0).active_tokens_by_worker,
            vec![(0, 0)]
        );
    }

    #[test]
    fn test_single_worker_router_honors_queue_threshold() {
        let mut router =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 1)
                .unwrap();

        let first = router
            .on_request_arrival(&request(1, 7), None, 0.0)
            .unwrap();
        assert_eq!(first.admissions.len(), 1);

        let second = router
            .on_request_arrival(&request(2, 8), None, 0.0)
            .unwrap();
        assert!(second.admissions.is_empty());
        assert_eq!(router.pending_count(), 1);
        assert_eq!(
            router
                .debug_snapshot(0.0)
                .pending
                .into_iter()
                .map(|request| request.uuid)
                .collect::<Vec<_>>(),
            vec![Uuid::from_u128(2)]
        );
    }

    #[test]
    fn attention_dp_routes_one_mocker_worker_across_rank_targets() {
        let mut args = queueing_args();
        args.dp_size = 2;
        let mut router =
            OfflineReplayRouter::new(&args, Some(queueing_router_config()), None, 1).unwrap();

        let first = router
            .on_request_arrival(&request(1, 7), None, 0.0)
            .unwrap();
        let second = router
            .on_request_arrival(&request(2, 8), None, 0.0)
            .unwrap();
        let mut targets = first
            .admissions
            .into_iter()
            .chain(second.admissions)
            .map(|admission| admission.worker_idx)
            .collect::<Vec<_>>();
        targets.sort_unstable();

        assert_eq!(targets, vec![0, 1]);
        assert_eq!(router.pending_count(), 0);
    }

    #[test]
    fn attention_dp_kv_router_routes_cached_prefix_to_matching_rank() {
        let mut args = replay_args();
        args.dp_size = 2;
        let mut router = OfflineReplayRouter::new(&args, Some(router_config()), None, 1).unwrap();
        let target = request(1, 7);
        let hashes = ReplayRequestHashes::from_tokens(&target.tokens, router.block_size);

        router
            .on_kv_events(vec![store_event_for_rank(
                0,
                1,
                1,
                hashes.local_block_hashes[0],
                StorageTier::Device,
            )])
            .unwrap();

        let effects = router
            .on_request_arrival(&target, Some(hashes), 0.0)
            .unwrap();
        assert_eq!(
            effects.admissions,
            vec![WorkerAdmission {
                uuid: Uuid::from_u128(1),
                worker_idx: 1,
                overlap_blocks: 1,
                best_available_overlap_blocks: 1,
                isl_blocks: 1,
                kv_fetch_source: None,
                retain_ttl_ms: None,
                retain_value: None,
                context_tokens: 0,
            }]
        );
    }

    #[test]
    fn prompt_token_lookup_scores_indexed_prefix() {
        let mut router = OfflineReplayRouter::new(&replay_args(), Some(router_config()), None, 2)
            .expect("router construction");
        let target = request(1, 7);
        let local_hashes = compute_block_hash_for_seq(
            &target.tokens,
            router.block_size,
            BlockHashOptions::default(),
        );
        router
            .on_kv_events(vec![store_event(
                1,
                1,
                local_hashes[0].0,
                StorageTier::Device,
            )])
            .unwrap();

        let effects = router.on_request_arrival(&target, None, 0.0).unwrap();
        assert_eq!(
            effects.admissions,
            vec![WorkerAdmission {
                uuid: Uuid::from_u128(1),
                worker_idx: 1,
                overlap_blocks: 1,
                best_available_overlap_blocks: 1,
                isl_blocks: 1,
                kv_fetch_source: None,
                retain_ttl_ms: None,
                retain_value: None,
                context_tokens: 0,
            }]
        );
    }

    #[test]
    fn canceled_pending_request_is_not_admitted_after_capacity_frees() {
        let mut router =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 1)
                .unwrap();

        assert_eq!(
            router
                .on_request_arrival(&request(1, 7), None, 0.0)
                .unwrap()
                .admissions
                .len(),
            1
        );
        assert!(
            router
                .on_request_arrival(&request(2, 8), None, 0.0)
                .unwrap()
                .admissions
                .is_empty()
        );
        assert!(router.cancel_pending(Uuid::from_u128(2)));
        assert!(!router.cancel_pending(Uuid::from_u128(2)));
        assert_eq!(router.pending_count(), 0);

        let effects = router
            .on_request_completed(Uuid::from_u128(1), 1.0)
            .unwrap();
        assert!(effects.admissions.is_empty());
    }

    #[test]
    fn a_queue_rejection_goes_back_to_the_client_when_asked() {
        let path =
            std::env::temp_dir().join(format!("dynamo-replay-reject-{}.yaml", Uuid::new_v4()));
        std::fs::write(
            &path,
            "policy_classes:\n  - name: interactive\n    quantum: 1\n    prefill_busy_threshold: 0\n    request_queue_limit_per_worker: 0\n",
        )
        .unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(path.display().to_string()),
            ..KvRouterConfig::default()
        };
        let mut router = OfflineReplayRouter::new(&queueing_args(), Some(config), None, 1).unwrap();
        std::fs::remove_file(path).unwrap();
        router.queue_reject_retry = true;
        let mut first = request(1, 1);
        first.policy_class = Some("interactive".to_string());
        assert_eq!(
            router
                .on_request_arrival(&first, None, 0.0)
                .unwrap()
                .admissions
                .len(),
            1
        );
        let mut refused = request(2, 2);
        refused.policy_class = Some("interactive".to_string());
        let effects = router.on_request_arrival(&refused, None, 0.0).unwrap();
        assert!(effects.admissions.is_empty());
        assert_eq!(router.queue_rejections, 1);
        assert_eq!(
            router.retry_waiting.len(),
            1,
            "the client retries after its backoff"
        );
    }

    #[test]
    fn the_refusal_log_records_each_first_offer_and_refusal() {
        let path =
            std::env::temp_dir().join(format!("dynamo-replay-refusal-{}.yaml", Uuid::new_v4()));
        std::fs::write(
            &path,
            "policy_classes:\n  - name: interactive\n    quantum: 1\n    prefill_busy_threshold: 0\n    request_queue_limit_per_worker: 0\n",
        )
        .unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(path.display().to_string()),
            ..KvRouterConfig::default()
        };
        let mut router = OfflineReplayRouter::new(&queueing_args(), Some(config), None, 1).unwrap();
        std::fs::remove_file(path).unwrap();
        router.queue_reject_retry = true;
        let log =
            std::env::temp_dir().join(format!("dynamo-replay-refusals-{}.jsonl", Uuid::new_v4()));
        router.refusal_log = Some(std::io::BufWriter::new(
            std::fs::File::create(&log).unwrap(),
        ));
        for id in [1, 2] {
            let mut r = request(id, id as u32);
            r.policy_class = Some("interactive".to_string());
            router.on_request_arrival(&r, None, 0.0).unwrap();
        }
        // The refused request's retry is due at 2 s and is refused again: one offer, two refusals.
        router.release_due_retries(2000.0).unwrap();
        drop(router);
        let lines: Vec<serde_json::Value> = std::fs::read_to_string(&log)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        std::fs::remove_file(log).unwrap();
        let kinds = |id: u128| -> Vec<String> {
            lines
                .iter()
                .filter(|l| l["uuid"] == Uuid::from_u128(id).to_string())
                .map(|l| l["kind"].as_str().unwrap().to_string())
                .collect()
        };
        assert_eq!(kinds(1), ["offer"]);
        assert_eq!(kinds(2), ["offer", "refused", "refused"]);
    }

    #[test]
    fn policy_mapping_and_model_selection_use_shared_replay_queue_logic() {
        let path =
            std::env::temp_dir().join(format!("dynamo-replay-policy-{}.yaml", Uuid::new_v4()));
        std::fs::write(
            &path,
            r#"
policy_classes:
  - name: root
    quantum: 1
models:
  replay-model:
    default_policy_class: latency
    policy_classes:
      - name: latency
        quantum: 1
        prefill_busy_threshold: 0
        request_queue_limit_per_worker: 1
      - name: interactive
        quantum: 1
        prefill_busy_threshold: 0
        request_queue_limit_per_worker: 0
      - name: batch
        quantum: 4
        prefill_busy_threshold: 1024
"#,
        )
        .unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(path.display().to_string()),
            ..KvRouterConfig::default()
        }
        .with_policy_model_name(Some("replay-model".to_string()));
        let mut router = OfflineReplayRouter::new(&queueing_args(), Some(config), None, 1).unwrap();
        std::fs::remove_file(path).unwrap();

        let mut active = request(1, 1);
        active.policy_class = Some("interactive".to_string());
        assert_eq!(
            router
                .on_request_arrival(&active, None, 0.0)
                .unwrap()
                .admissions
                .len(),
            1
        );

        let mut mapped_batch = request(2, 2);
        mapped_batch.policy_class = Some("batch".to_string());
        assert_eq!(
            router
                .on_request_arrival(&mapped_batch, None, 0.0)
                .unwrap()
                .admissions
                .len(),
            1,
            "batch should select its own class and use its higher busy threshold"
        );

        let mut cached_latency = request(3, 3);
        cached_latency.policy_class = Some("latency".to_string());
        let cached_hashes =
            ReplayRequestHashes::from_tokens(&cached_latency.tokens, router.block_size);
        router
            .on_kv_events(vec![store_event(
                0,
                1,
                cached_hashes.local_block_hashes[0],
                StorageTier::Device,
            )])
            .unwrap();
        assert!(
            router
                .on_request_arrival(&cached_latency, Some(cached_hashes), 0.0)
                .unwrap()
                .admissions
                .is_empty(),
            "latency should queue while the worker is busy"
        );

        let mut fail_fast = request(4, 4);
        fail_fast.policy_class = Some("interactive".to_string());
        let error = router
            .on_request_arrival(&fail_fast, None, 0.0)
            .unwrap_err();
        let rejection = error
            .downcast_ref::<dynamo_admission::host::QueueRejection>()
            .expect("replay should preserve the typed queue rejection");
        assert_eq!(rejection.policy_class, "interactive");
        assert_eq!(rejection.current, 0);
        assert_eq!(rejection.limit, 0);

        router.add_worker(1, false).unwrap();
        let mut uncached_latency = request(5, 5);
        uncached_latency.policy_class = Some("latency".to_string());
        assert!(
            router
                .on_request_arrival(&uncached_latency, None, 0.0)
                .unwrap()
                .admissions
                .is_empty(),
            "an uncached request should join the same latency backlog under the raised limit"
        );
        assert_eq!(router.pending_count(), 2);

        router.remove_worker(1).unwrap();
        let mut root_class = request(6, 6);
        root_class.policy_class = Some("root".to_string());
        let error = router
            .on_request_arrival(&root_class, None, 0.0)
            .unwrap_err();
        assert!(
            matches!(
                error.downcast_ref::<dynamo_scheduler::host::KvSchedulerError>(),
                Some(dynamo_scheduler::host::KvSchedulerError::UnknownPolicyClass(name))
                    if name == "root"
            ),
            "a root class name is unknown to the model profile: {error:?}"
        );
        assert_eq!(router.pending_count(), 2);

        let error = router
            .on_request_arrival(&request(7, 7), None, 0.0)
            .unwrap_err();
        let rejection = error
            .downcast_ref::<dynamo_admission::host::QueueRejection>()
            .expect("worker removal should retain the typed queue rejection");
        assert_eq!(
            rejection.policy_class, "latency",
            "an omitted name uses the model's default class"
        );
        assert_eq!(rejection.current, 2);
        assert_eq!(rejection.limit, 1);
        assert_eq!(router.pending_count(), 2);
    }

    #[test]
    fn dequeue_asks_a_waiting_request_back_and_places_it_on_an_idle_worker() {
        let path =
            std::env::temp_dir().join(format!("dynamo-replay-policy-{}.yaml", Uuid::new_v4()));
        std::fs::write(
            &path,
            r#"
policy_classes:
  - name: standard
    quantum: 1
kv_actions:
  dequeue:
    after_s: 2.0
"#,
        )
        .unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(path.display().to_string()),
            ..router_config()
        };
        let mut router = OfflineReplayRouter::new(&queueing_args(), Some(config), None, 1).unwrap();
        std::fs::remove_file(path).unwrap();
        let effects = router
            .on_request_arrival(&request(1, 1), None, 0.0)
            .unwrap();
        assert_eq!(effects.admissions[0].worker_idx, 0);
        let uuid = Uuid::from_u128(1);
        assert!(router.dequeue.kept.contains_key(&uuid));
        router.add_worker(1, false).unwrap();
        // Not yet due at 1 s; its wake-up is at 2 s.
        router.dequeue_check(1000.0).unwrap();
        assert!(router.take_kv_actions().is_empty());
        assert_eq!(router.next_wakeup_ms(), Some(2000.0));
        router.dequeue_check(2500.0).unwrap();
        let sent = router.take_kv_actions();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].scheduler_id, 0);
        assert!(matches!(
            sent[0].actions[0].action,
            KvAction::Dequeue { request_id } if request_id == uuid
        ));
        // Asked once only.
        router.dequeue_check(9000.0).unwrap();
        assert!(router.take_kv_actions().is_empty());
        // Handed back: placed on the idle worker.
        let admissions = router.requeue(uuid, 2500.0).unwrap();
        assert_eq!(admissions[0].worker_idx, 1);
        assert_eq!(router.dequeue.stats.requeued, 1);
        assert!(!router.dequeue.kept.contains_key(&uuid));
        // A completed prefill is never asked back.
        router
            .on_request_arrival(&request(2, 2), None, 3000.0)
            .unwrap();
        router
            .on_prefill_completed(Uuid::from_u128(2), 3100.0)
            .unwrap();
        router.dequeue_check(9000.0).unwrap();
        assert!(router.take_kv_actions().is_empty());
    }

    #[test]
    fn peer_hbm_merges_device_holders_into_the_transfer_candidates() {
        use dynamo_kv_hashing::ExternalSequenceBlockHash as H;
        let w = |id| KvTransferCandidateSource::Worker(WorkerWithDpRank::new(id, 0));
        let candidates = |hashes: &[u64], owners: Vec<(KvTransferCandidateSource, usize)>| {
            KvTransferCandidates {
                block_hashes: hashes.iter().map(|&h| H(h)).collect(),
                owner_prefix_blocks: owners,
                ..Default::default()
            }
        };
        let host = candidates(&[1, 2], vec![(w(0), 2)]);
        let device = candidates(&[1, 2, 3, 4], vec![(w(0), 1), (w(1), 4)]);
        let merged = merge_candidates(Some(host.clone()), Some(device.clone())).unwrap();
        assert_eq!(merged.block_hashes.len(), 4);
        assert_eq!(merged.owner_prefix_blocks, vec![(w(0), 2), (w(1), 4)]);
        assert_eq!(merge_candidates(None, Some(device.clone())), Some(device));
        assert_eq!(merge_candidates(Some(host.clone()), None), Some(host));
        assert_eq!(merge_candidates(None, None), None);
    }

    #[test]
    fn a_fan_out_is_replicated_to_the_least_loaded_workers_that_lack_it() {
        let path =
            std::env::temp_dir().join(format!("dynamo-replay-policy-{}.yaml", Uuid::new_v4()));
        std::fs::write(
            &path,
            r#"
policy_classes:
  - name: standard
    quantum: 1
kv_actions:
  replicate:
    replicas: 2
"#,
        )
        .unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(path.display().to_string()),
            ..router_config()
        };
        let mut router = OfflineReplayRouter::new(&queueing_args(), Some(config), None, 1).unwrap();
        std::fs::remove_file(path).unwrap();
        router.add_worker(1, false).unwrap();
        router.add_worker(2, false).unwrap();
        let mut overlap = OverlapSignals::default();
        overlap
            .effective_overlap_blocks
            .insert(WorkerWithDpRank::new(0, 0), 2.0);
        let now = router.decay_now(0.0);
        router
            .plan_replicas("p".into(), vec![10, 11, 12], vec![1, 2, 3], &overlap, now)
            .unwrap();
        let sent = router.take_kv_actions();
        assert_eq!(sent.len(), 1, "two holders wanted, one exists");
        assert_eq!(sent[0].scheduler_id, 1);
        assert!(matches!(
            &sent[0].actions[0].action,
            KvAction::Replicate { source_scheduler_id: 0, block_hashes, local_hashes }
                if block_hashes == &vec![10, 11] && local_hashes == &vec![1, 2]
        ));
        // Once per parent within the window.
        router
            .plan_replicas(
                "p".into(),
                vec![10, 11, 12],
                vec![1, 2, 3],
                &overlap,
                router.decay_now(1000.0),
            )
            .unwrap();
        assert!(router.take_kv_actions().is_empty());
        // Nobody holds it: nothing to copy.
        router
            .plan_replicas(
                "q".into(),
                vec![10],
                vec![1],
                &OverlapSignals::default(),
                now,
            )
            .unwrap();
        assert!(router.take_kv_actions().is_empty());
    }

    #[test]
    fn without_dequeue_nothing_is_kept() {
        let mut router =
            OfflineReplayRouter::new(&queueing_args(), Some(router_config()), None, 1).unwrap();
        router
            .on_request_arrival(&request(1, 1), None, 0.0)
            .unwrap();
        assert!(router.dequeue.kept.is_empty());
        assert_eq!(router.next_wakeup_ms(), None);
    }

    #[test]
    fn replay_snapshot_ignores_removed_worker_cache() {
        let path =
            std::env::temp_dir().join(format!("dynamo-replay-policy-{}.yaml", Uuid::new_v4()));
        std::fs::write(
            &path,
            r#"
policy_classes:
  - name: standard
    quantum: 1
    prefill_busy_threshold: 0
"#,
        )
        .unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(path.display().to_string()),
            ..KvRouterConfig::default()
        };
        let mut router = OfflineReplayRouter::new(&queueing_args(), Some(config), None, 2).unwrap();
        std::fs::remove_file(path).unwrap();

        let target = request(2, 2);
        let target_hashes = ReplayRequestHashes::from_tokens(&target.tokens, router.block_size);
        router
            .on_kv_events(vec![store_event(
                1,
                1,
                target_hashes.local_block_hashes[0],
                StorageTier::Device,
            )])
            .unwrap();
        router.remove_worker(1).unwrap();

        router
            .on_request_arrival(&request(1, 1), None, 0.0)
            .unwrap();
        assert!(
            router
                .on_request_arrival(&target, Some(target_hashes), 0.0)
                .unwrap()
                .admissions
                .is_empty()
        );
        let stats = router.pending.class_stats(0);
        assert_eq!(
            (stats.requests, stats.raw_isl_tokens, stats.cached_tokens),
            (1, 64, 0),
            "cache state from a removed worker must not reach the queued snapshot"
        );

        let cached = request(3, 3);
        let cached_hashes = ReplayRequestHashes::from_tokens(&cached.tokens, router.block_size);
        router
            .on_kv_events(vec![store_event(
                0,
                2,
                cached_hashes.local_block_hashes[0],
                StorageTier::Device,
            )])
            .unwrap();
        assert!(
            router
                .on_request_arrival(&cached, Some(cached_hashes), 0.0)
                .unwrap()
                .admissions
                .is_empty()
        );
        let stats = router.pending.class_stats(0);
        assert_eq!(
            (stats.requests, stats.raw_isl_tokens, stats.cached_tokens),
            (2, 128, 64)
        );
    }

    #[test]
    fn strict_priority_precedes_each_offline_queue_policy() {
        for policy in [
            RouterQueuePolicy::Fcfs,
            RouterQueuePolicy::Lcfs,
            RouterQueuePolicy::Wspt,
        ] {
            let mut router = OfflineReplayRouter::new(
                &queueing_args(),
                Some(queueing_router_config_with_policy(policy)),
                None,
                1,
            )
            .unwrap();
            router
                .on_request_arrival(&request(1, 1), None, 0.0)
                .unwrap();
            router
                .on_request_arrival(&request_with_priorities(2, 2, 1, 1_000_000, 0), None, 0.0)
                .unwrap();
            router
                .on_request_arrival(&request_with_priorities(3, 3, 64, 0, 1), None, 1_000.0)
                .unwrap();

            let pending = router
                .debug_snapshot(1_000.0)
                .pending
                .into_iter()
                .map(|request| request.uuid)
                .collect::<Vec<_>>();
            assert_eq!(pending, vec![Uuid::from_u128(3), Uuid::from_u128(2)]);
        }
    }

    #[test]
    fn equal_strict_tiers_retain_offline_queue_policy_ordering() {
        let cases = [
            (RouterQueuePolicy::Fcfs, vec![2, 3]),
            (RouterQueuePolicy::Lcfs, vec![3, 2]),
            (RouterQueuePolicy::Wspt, vec![3, 2]),
        ];

        for (policy, expected) in cases {
            let mut router = OfflineReplayRouter::new(
                &queueing_args(),
                Some(queueing_router_config_with_policy(policy)),
                None,
                1,
            )
            .unwrap();
            router
                .on_request_arrival(&request(1, 1), None, 0.0)
                .unwrap();
            router
                .on_request_arrival(&request_with_priorities(2, 2, 64, 0, 4), None, 0.0)
                .unwrap();
            router
                .on_request_arrival(&request_with_priorities(3, 3, 1, 0, 4), None, 100.0)
                .unwrap();

            let pending = router
                .debug_snapshot(100.0)
                .pending
                .into_iter()
                .map(|request| request.uuid.as_u128())
                .collect::<Vec<_>>();
            assert_eq!(pending, expected);
        }
    }

    #[test]
    fn soft_priority_orders_within_an_offline_strict_tier() {
        let mut router =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 1)
                .unwrap();
        router
            .on_request_arrival(&request(1, 1), None, 0.0)
            .unwrap();
        router
            .on_request_arrival(&request_with_priorities(2, 2, 64, 0, 2), None, 0.0)
            .unwrap();
        router
            .on_request_arrival(&request_with_priorities(3, 3, 64, 1_000, 2), None, 100.0)
            .unwrap();

        let pending = router
            .debug_snapshot(100.0)
            .pending
            .into_iter()
            .map(|request| request.uuid)
            .collect::<Vec<_>>();
        assert_eq!(pending, vec![Uuid::from_u128(3), Uuid::from_u128(2)]);
    }

    #[test]
    fn edf_offline_queue_orders_by_virtual_due_time_without_priority_modifiers() {
        let path = std::env::temp_dir().join(format!("dynamo-replay-edf-{}.yaml", Uuid::new_v4()));
        std::fs::write(
            &path,
            r#"
policy_classes:
  - name: standard
    quantum: 64
    slo_ttft_ms: 250
    prefill_busy_threshold_frac: 0.5
"#,
        )
        .unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(path.display().to_string()),
            ..KvRouterConfig::default()
        };
        let mut router = OfflineReplayRouter::new(&queueing_args(), Some(config), None, 1).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(router.profile.class(0).queue_policy, RouterQueuePolicy::Edf);

        router
            .on_request_arrival(&request(1, 1), None, 0.0)
            .unwrap();
        // Each class due time counts from the virtual arrival, so later arrivals keep their
        // order; soft and strict priority do not move them ahead.
        for (uuid, priority, strict_priority, now_ms) in
            [(2, 0, 0, 10.0), (3, 1_000_000, 9, 20.0), (4, 0, 0, 30.0)]
        {
            router
                .on_request_arrival(
                    &request_with_priorities(uuid, uuid as u32, 64, priority, strict_priority),
                    None,
                    now_ms,
                )
                .unwrap();
        }
        let pending = router
            .debug_snapshot(30.0)
            .pending
            .into_iter()
            .map(|request| request.uuid.as_u128())
            .collect::<Vec<_>>();
        assert_eq!(pending, [2, 3, 4]);

        // A passed class budget neither rejects nor delays a queued request.
        let admitted = router
            .on_request_completed(Uuid::from_u128(1), 10_000.0)
            .unwrap();
        assert_eq!(admitted_uuids(admitted), [Uuid::from_u128(2)]);
        assert_eq!(router.pending_count(), 2);
    }

    #[test]
    fn test_scaled_down_router_matches_fresh_single_worker_queueing() {
        let mut fresh =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 1)
                .unwrap();
        fresh.on_request_arrival(&request(1, 7), None, 0.0).unwrap();
        fresh.on_request_arrival(&request(2, 8), None, 0.0).unwrap();

        let mut scaled =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 2)
                .unwrap();
        scaled.remove_worker(1).unwrap();
        scaled
            .on_request_arrival(&request(1, 7), None, 0.0)
            .unwrap();
        scaled
            .on_request_arrival(&request(2, 8), None, 0.0)
            .unwrap();

        assert_eq!(scaled.pending_count(), fresh.pending_count());
        assert_eq!(
            scaled
                .debug_snapshot(0.0)
                .pending
                .into_iter()
                .map(|request| request.uuid)
                .collect::<Vec<_>>(),
            fresh
                .debug_snapshot(0.0)
                .pending
                .into_iter()
                .map(|request| request.uuid)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_router_can_scale_from_zero_workers() {
        let mut router =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 1)
                .unwrap();

        router.remove_worker(0).unwrap();
        router.add_worker(3, false).unwrap();

        let effects = router
            .on_request_arrival(&request(1, 7), None, 0.0)
            .unwrap();
        assert_eq!(
            effects.admissions,
            vec![WorkerAdmission {
                uuid: Uuid::from_u128(1),
                worker_idx: 3,
                overlap_blocks: 0,
                best_available_overlap_blocks: 0,
                isl_blocks: 1,
                kv_fetch_source: None,
                retain_ttl_ms: None,
                retain_value: None,
                context_tokens: 0,
            }]
        );
    }

    #[test]
    fn cache_telemetry_excludes_removed_workers() {
        let mut router =
            OfflineReplayRouter::new(&replay_args(), Some(router_config()), None, 2).unwrap();
        let target = request(1, 7);
        let hashes = ReplayRequestHashes::from_tokens(&target.tokens, router.block_size);
        router
            .on_kv_events(vec![store_event(
                1,
                1,
                hashes.local_block_hashes[0],
                StorageTier::Device,
            )])
            .unwrap();
        router.remove_worker(1).unwrap();

        let effects = router
            .on_request_arrival(&target, Some(hashes), 0.0)
            .unwrap();
        assert_eq!(effects.admissions.len(), 1);
        assert_eq!(effects.admissions[0].worker_idx, 0);
        assert_eq!(effects.admissions[0].overlap_blocks, 0);
        assert_eq!(effects.admissions[0].best_available_overlap_blocks, 0);
    }

    #[test]
    fn finalized_worker_removal_drops_retained_topology_and_cache_state() {
        let mut router =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 1)
                .unwrap();
        let target = request(1, 7);
        let hashes = ReplayRequestHashes::from_tokens(&target.tokens, router.block_size);
        router
            .on_kv_events(vec![store_event(
                0,
                1,
                hashes.local_block_hashes[0],
                StorageTier::Device,
            )])
            .unwrap();
        assert_eq!(router.debug_snapshot(0.0).indexer.total_cached_blocks, 1);

        router.remove_worker(0).unwrap();
        router.finalize_worker_removal(0).unwrap();
        let snapshot = router.debug_snapshot(0.0);
        assert!(snapshot.active_blocks_by_worker.is_empty());
        assert!(snapshot.indexer.cached_blocks_by_worker.is_empty());
        router.add_worker(0, false).unwrap();
    }

    #[test]
    fn test_add_worker_preserves_draining_worker_state() {
        let mut router =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 2)
                .unwrap();

        router
            .on_request_arrival(&request(1, 7), None, 0.0)
            .unwrap();
        router
            .on_request_arrival(&request(2, 8), None, 0.0)
            .unwrap();
        assert_eq!(
            router.debug_snapshot(0.0).active_tokens_by_worker,
            vec![(0, 64), (1, 64)]
        );

        router.remove_worker(1).unwrap();
        router.add_worker(2, false).unwrap();

        assert_eq!(
            router.debug_snapshot(0.0).active_tokens_by_worker,
            vec![(0, 64), (1, 64), (2, 0)]
        );
    }

    #[test]
    fn test_readding_removed_worker_reuses_slot_state() {
        let mut router =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 1)
                .unwrap();

        router
            .on_request_arrival(&request(1, 7), None, 0.0)
            .unwrap();
        router.remove_worker(0).unwrap();
        router.add_worker(0, false).unwrap();

        assert_eq!(
            router.debug_snapshot(0.0).active_tokens_by_worker,
            vec![(0, 64)]
        );
    }

    #[test]
    fn test_topology_change_drains_pending_after_scale_up() {
        let mut router =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 1)
                .unwrap();

        router
            .on_request_arrival(&request(1, 7), None, 0.0)
            .unwrap();
        router
            .on_request_arrival(&request(2, 8), None, 0.0)
            .unwrap();
        assert_eq!(router.pending_count(), 1);

        router.add_worker(1, false).unwrap();
        let effects = router.on_topology_changed(0.0).unwrap();

        assert_eq!(
            effects.admissions,
            vec![WorkerAdmission {
                uuid: Uuid::from_u128(2),
                worker_idx: 1,
                overlap_blocks: 0,
                best_available_overlap_blocks: 0,
                isl_blocks: 1,
                kv_fetch_source: None,
                retain_ttl_ms: None,
                retain_value: None,
                context_tokens: 0,
            }]
        );
        assert_eq!(router.pending_count(), 0);
        assert_eq!(
            router.debug_snapshot(0.0).active_tokens_by_worker,
            vec![(0, 64), (1, 64)]
        );
    }

    #[test]
    fn no_workers_preserves_pending_requests_during_completion_drain() {
        let mut router =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 1)
                .unwrap();

        router
            .on_request_arrival(&request(1, 7), None, 0.0)
            .unwrap();
        router
            .on_request_arrival(&request(2, 8), None, 0.0)
            .unwrap();
        assert_eq!(router.pending_count(), 1);

        router.remove_worker(0).unwrap();
        let effects = router
            .on_request_completed(Uuid::from_u128(1), 0.0)
            .unwrap();

        assert!(effects.admissions.is_empty());
        assert_eq!(router.pending_count(), 1);
        assert_eq!(
            router.debug_snapshot(0.0).pending[0].uuid,
            Uuid::from_u128(2)
        );
    }

    fn admitted_uuids(effects: super::RouterEffects) -> Vec<Uuid> {
        effects
            .admissions
            .iter()
            .map(|admission| admission.uuid)
            .collect()
    }

    #[test]
    fn lagged_kv_events_stay_invisible_to_routing_until_due() {
        let mut router = OfflineReplayRouter::new(&replay_args(), None, None, 1).unwrap();
        router.set_kv_event_lag_ms(50.0).unwrap();
        let first = request(1, 7);
        let hashes = ReplayRequestHashes::from_tokens(&first.tokens, router.block_size);
        router
            .on_kv_events_at(
                vec![store_event(
                    0,
                    1,
                    hashes.local_block_hashes[0],
                    StorageTier::Device,
                )],
                0.0,
            )
            .unwrap();

        let hidden = router
            .on_request_arrival(&first, Some(hashes.clone()), 49.0)
            .unwrap();
        assert_eq!(hidden.admissions[0].best_available_overlap_blocks, 0);
        let visible = router
            .on_request_arrival(&request(2, 7), Some(hashes), 50.0)
            .unwrap();
        assert_eq!(visible.admissions[0].best_available_overlap_blocks, 1);
    }

    #[test]
    fn completions_stay_immediate_under_kv_event_lag() {
        let mut router = OfflineReplayRouter::new(&replay_args(), None, None, 1).unwrap();
        router.set_kv_event_lag_ms(50.0).unwrap();
        router
            .on_request_arrival(&request(1, 7), None, 0.0)
            .unwrap();
        let admitted = router.debug_snapshot(0.0);
        assert_eq!(admitted.active_tokens_by_worker, vec![(0, 64)]);
        assert_ne!(admitted.active_blocks_by_worker, vec![(0, 0)]);

        router
            .on_prefill_completed(Uuid::from_u128(1), 10.0)
            .unwrap();
        assert_eq!(
            router.debug_snapshot(10.0).active_tokens_by_worker,
            vec![(0, 0)]
        );
        router
            .on_request_completed(Uuid::from_u128(1), 20.0)
            .unwrap();
        assert_eq!(
            router.debug_snapshot(20.0).active_blocks_by_worker,
            vec![(0, 0)]
        );

        let mut router =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 1)
                .unwrap();
        router.set_kv_event_lag_ms(50.0).unwrap();
        router
            .on_request_arrival(&request(1, 7), None, 0.0)
            .unwrap();
        router
            .on_request_arrival(&request(2, 8), None, 0.0)
            .unwrap();
        assert_eq!(router.pending_count(), 1);

        let released = router
            .on_request_completed(Uuid::from_u128(1), 10.0)
            .unwrap();
        assert_eq!(admitted_uuids(released), vec![Uuid::from_u128(2)]);
        assert_eq!(router.pending_count(), 0);
    }

    #[test]
    fn completion_drain_applies_due_kv_events_first() {
        let mut router =
            OfflineReplayRouter::new(&queueing_args(), Some(queueing_router_config()), None, 1)
                .unwrap();
        router.set_kv_event_lag_ms(50.0).unwrap();
        router
            .on_request_arrival(&request(1, 7), None, 0.0)
            .unwrap();
        router
            .on_request_arrival(&request(2, 8), None, 0.0)
            .unwrap();
        router
            .on_kv_events_at(vec![store_event(0, 1, 11, StorageTier::Device)], 0.0)
            .unwrap();
        router
            .on_kv_events_at(vec![store_event(0, 2, 12, StorageTier::Device)], 20.0)
            .unwrap();
        assert_eq!(router.debug_snapshot(20.0).indexer.total_cached_blocks, 0);

        // The batch due at 50 ms lands before the drain at 60 ms; the batch due at 70 ms does not.
        let released = router
            .on_request_completed(Uuid::from_u128(1), 60.0)
            .unwrap();
        assert_eq!(admitted_uuids(released), vec![Uuid::from_u128(2)]);
        assert_eq!(router.debug_snapshot(60.0).indexer.total_cached_blocks, 1);
    }

    #[test]
    fn finalized_worker_removal_drops_its_lagged_kv_events() {
        let mut router = OfflineReplayRouter::new(&replay_args(), None, None, 2).unwrap();
        router.set_kv_event_lag_ms(50.0).unwrap();
        router
            .on_kv_events_at(
                vec![
                    store_event(0, 1, 11, StorageTier::Device),
                    store_event(1, 2, 12, StorageTier::Device),
                ],
                0.0,
            )
            .unwrap();
        router.remove_worker(1).unwrap();
        router.finalize_worker_removal(1).unwrap();

        router.on_topology_changed(50.0).unwrap();
        assert_eq!(
            router.debug_snapshot(50.0).indexer.cached_blocks_by_worker,
            vec![(0, 1)]
        );
    }

    #[test]
    fn kv_event_lag_is_validated() {
        let mut router = OfflineReplayRouter::new(&replay_args(), None, None, 1).unwrap();
        assert!(router.set_kv_event_lag_ms(-1.0).is_err());
        assert!(router.set_kv_event_lag_ms(f64::NAN).is_err());
    }

    fn session_router_config() -> KvRouterConfig {
        KvRouterConfig {
            enable_session_prefix_index: true,
            ..Default::default()
        }
    }

    fn session_store(worker: u64, rank: u32, hash: u64, local: u64, session: &str) -> RouterEvent {
        let mut event = store_event_for_rank(worker, rank, hash, local, StorageTier::Device);
        event.session_id = Some(session.to_owned());
        event
    }

    fn session_lineage(
        router: &OfflineReplayRouter,
        session: &str,
        worker: u64,
        rank: u32,
    ) -> Vec<Vec<u64>> {
        let mut paths = router
            .indexer
            .session_prefix_index
            .as_ref()
            .unwrap()
            .get_session_block_lineage(session, WorkerWithDpRank::new(worker, rank), None)
            .unwrap()
            .into_iter()
            .map(|path| path.into_iter().map(|hash| hash.0).collect())
            .collect::<Vec<Vec<u64>>>();
        paths.sort();
        paths
    }

    fn session_remove(worker: u64, rank: u32, hashes: Vec<u64>) -> RouterEvent {
        RouterEvent::new(
            worker,
            KvCacheEvent {
                event_id: 100,
                dp_rank: rank,
                data: KvCacheEventData::Removed(dynamo_kv_events::event::KvCacheRemoveData {
                    block_hashes: hashes.into_iter().map(ExternalSequenceBlockHash).collect(),
                }),
            },
        )
    }

    #[test]
    fn session_prefix_is_opt_in_and_requires_local_kv_events() {
        let router = OfflineReplayRouter::new(&replay_args(), None, None, 1).unwrap();
        assert!(router.indexer.session_prefix_index.is_none());
        for config in [
            KvRouterConfig {
                use_kv_events: false,
                ..session_router_config()
            },
            KvRouterConfig {
                use_remote_indexer: true,
                ..session_router_config()
            },
        ] {
            assert!(OfflineReplayRouter::new(&replay_args(), Some(config), None, 1).is_err());
        }
    }

    #[test]
    fn session_prefix_stores_removes_and_clears_are_worker_rank_scoped() {
        let mut args = replay_args();
        args.dp_size = 2;
        let mut router =
            OfflineReplayRouter::new(&args, Some(session_router_config()), None, 2).unwrap();
        let root = session_store(0, 0, 11, 1, "a");
        let mut child = session_store(0, 0, 12, 2, "a");
        let KvCacheEventData::Stored(stored) = &mut child.event.data else {
            unreachable!()
        };
        stored.parent_hash = Some(ExternalSequenceBlockHash(11));
        router
            .on_kv_events(vec![
                root,
                child,
                session_store(0, 1, 11, 1, "a"),
                session_store(1, 0, 11, 1, "b"),
            ])
            .unwrap();
        assert_eq!(session_lineage(&router, "a", 0, 0), vec![vec![11, 12]]);
        router
            .on_kv_events(vec![session_remove(0, 0, vec![12])])
            .unwrap();
        assert_eq!(session_lineage(&router, "a", 0, 0), vec![vec![11]]);
        router
            .on_kv_events(vec![RouterEvent::new(
                0,
                KvCacheEvent {
                    event_id: 101,
                    dp_rank: 0,
                    data: KvCacheEventData::Cleared,
                },
            )])
            .unwrap();
        assert!(session_lineage(&router, "a", 0, 0).is_empty());
        assert_eq!(session_lineage(&router, "a", 0, 1), vec![vec![11]]);
        assert_eq!(session_lineage(&router, "b", 1, 0), vec![vec![11]]);
        router.remove_worker(0).unwrap();
        router.finalize_worker_removal(0).unwrap();
        assert!(session_lineage(&router, "a", 0, 1).is_empty());
        assert_eq!(session_lineage(&router, "b", 1, 0), vec![vec![11]]);
    }

    #[test]
    fn session_prefix_ignores_unattributed_and_lower_tier_stores() {
        let mut router =
            OfflineReplayRouter::new(&replay_args(), Some(session_router_config()), None, 1)
                .unwrap();
        let mut host = session_store(0, 0, 12, 2, "a");
        host.storage_tier = StorageTier::HostPinned;
        router
            .on_kv_events(vec![store_event(0, 11, 1, StorageTier::Device), host])
            .unwrap();
        assert_eq!(
            router
                .indexer
                .session_prefix_index
                .as_ref()
                .unwrap()
                .session_count(),
            0
        );
        // A host-only match must not create a G1 session frontier.
        let hashes = ReplayRequestHashes {
            local_block_hashes: vec![2],
            sequence_hashes: vec![12],
        };
        router
            .on_request_arrival_for_session(&request(1, 7), Some(hashes), Some("a".into()), 0.0)
            .unwrap();
        assert!(session_lineage(&router, "a", 0, 0).is_empty());
    }

    #[test]
    fn session_prefix_does_not_apply_a_physically_rejected_store() {
        let mut router =
            OfflineReplayRouter::new(&replay_args(), Some(session_router_config()), None, 1)
                .unwrap();
        let mut child = session_store(0, 0, 12, 2, "a");
        let KvCacheEventData::Stored(stored) = &mut child.event.data else {
            unreachable!()
        };
        stored.parent_hash = Some(ExternalSequenceBlockHash(11));
        assert!(router.on_kv_events(vec![child]).is_err());
        assert_eq!(
            router
                .indexer
                .session_prefix_index
                .as_ref()
                .unwrap()
                .node_count(),
            0
        );
    }

    #[test]
    fn session_prefix_route_match_uses_device_hashes_not_active_tracking_hashes() {
        for track_active in [true, false] {
            let config = KvRouterConfig {
                router_track_active_blocks: track_active,
                router_assume_kv_reuse: false,
                ..session_router_config()
            };
            let mut router =
                OfflineReplayRouter::new(&replay_args(), Some(config), None, 2).unwrap();
            router
                .on_kv_events(vec![session_store(1, 0, 11, 1, "producer")])
                .unwrap();
            let hashes = ReplayRequestHashes {
                local_block_hashes: vec![1],
                sequence_hashes: vec![999],
            };
            let effects = router
                .on_request_arrival_for_session(
                    &request(1, 7),
                    Some(hashes),
                    Some("consumer".into()),
                    0.0,
                )
                .unwrap();
            assert_eq!(effects.admissions[0].worker_idx, 1);
            assert_eq!(session_lineage(&router, "consumer", 1, 0), vec![vec![11]]);
            assert!(session_lineage(&router, "consumer", 0, 0).is_empty());
        }
    }

    #[test]
    fn session_prefix_queued_match_rechecks_residency_at_admission() {
        let config = KvRouterConfig {
            enable_session_prefix_index: true,
            ..queueing_router_config()
        };
        let mut router = OfflineReplayRouter::new(&queueing_args(), Some(config), None, 1).unwrap();
        router
            .on_request_arrival(&request(1, 7), None, 0.0)
            .unwrap();
        router
            .on_kv_events(vec![session_store(0, 0, 11, 1, "producer")])
            .unwrap();
        let hashes = ReplayRequestHashes {
            local_block_hashes: vec![1],
            sequence_hashes: vec![11],
        };
        let effects = router
            .on_request_arrival_for_session(
                &request(2, 8),
                Some(hashes),
                Some("consumer".into()),
                0.0,
            )
            .unwrap();
        assert!(effects.admissions.is_empty());
        assert!(session_lineage(&router, "consumer", 0, 0).is_empty());
        router
            .on_kv_events(vec![session_remove(0, 0, vec![11])])
            .unwrap();
        let effects = router
            .on_request_completed(Uuid::from_u128(1), 10.0)
            .unwrap();
        assert_eq!(effects.admissions.len(), 1);
        assert!(session_lineage(&router, "consumer", 0, 0).is_empty());
    }

    #[test]
    fn session_prefix_respects_virtual_event_lag_and_worker_removal() {
        let mut router =
            OfflineReplayRouter::new(&replay_args(), Some(session_router_config()), None, 2)
                .unwrap();
        router.set_kv_event_lag_ms(50.0).unwrap();
        router
            .on_kv_events_at(
                vec![
                    session_store(0, 0, 11, 1, "a"),
                    session_store(1, 0, 11, 1, "b"),
                ],
                0.0,
            )
            .unwrap();
        assert!(session_lineage(&router, "a", 0, 0).is_empty());
        router.remove_worker(1).unwrap();
        router.finalize_worker_removal(1).unwrap();
        router.on_topology_changed(50.0).unwrap();
        assert_eq!(session_lineage(&router, "a", 0, 0), vec![vec![11]]);
        assert!(session_lineage(&router, "b", 1, 0).is_empty());
    }

    #[test]
    fn session_prefix_materialized_requests_preserve_placement() {
        let target = request(1, 7);
        let local =
            compute_block_hash_for_seq(&target.tokens, 64, BlockHashOptions::default())[0].0;
        let mut baseline = OfflineReplayRouter::new(&replay_args(), None, None, 1).unwrap();
        let mut tracked =
            OfflineReplayRouter::new(&replay_args(), Some(session_router_config()), None, 1)
                .unwrap();
        for router in [&mut baseline, &mut tracked] {
            router
                .on_kv_events(vec![session_store(0, 0, 11, local, "producer")])
                .unwrap();
        }
        let before = baseline
            .on_request_arrival_for_session(&target, None, Some("consumer".into()), 0.0)
            .unwrap();
        let after = tracked
            .on_request_arrival_for_session(&target, None, Some("consumer".into()), 0.0)
            .unwrap();
        assert_eq!(before.admissions, after.admissions);
        assert_eq!(session_lineage(&tracked, "consumer", 0, 0), vec![vec![11]]);
    }

    #[test]
    fn session_prefix_incomplete_attribution_does_not_fail_physical_routing() {
        let mut router =
            OfflineReplayRouter::new(&replay_args(), Some(session_router_config()), None, 1)
                .unwrap();
        let mut child = session_store(0, 0, 12, 2, "a");
        let KvCacheEventData::Stored(stored) = &mut child.event.data else {
            unreachable!()
        };
        stored.parent_hash = Some(ExternalSequenceBlockHash(11));
        // The physical parent is known, but its store had no session attribution.
        router
            .on_kv_events(vec![store_event(0, 11, 1, StorageTier::Device), child])
            .unwrap();
        let details = router
            .indexer
            .find_matches_for_hashes(vec![LocalBlockHash(1), LocalBlockHash(2)]);
        assert_eq!(
            details.device.last_matched_hashes[&WorkerWithDpRank::new(0, 0)],
            ExternalSequenceBlockHash(12)
        );
    }

    fn worker_topology(worker_id: usize, dp_size: u32, router_hint: bool) -> WorkerTopology {
        let dp_size = dp_size as usize;
        WorkerTopology {
            worker_id,
            scheduler_ids: (worker_id * dp_size..(worker_id + 1) * dp_size).collect(),
            router_hint,
        }
    }

    pub(super) fn topology(workers: usize, dp_size: u32, router_hint: bool) -> Vec<WorkerTopology> {
        (0..workers)
            .map(|worker_id| worker_topology(worker_id, dp_size, router_hint))
            .collect()
    }

    fn kv_fetch_placement(
        args: &MockerConfig,
        config: KvRouterConfig,
        workers: usize,
        router_hint: bool,
    ) -> KvRouterPlacement {
        let topology = topology(workers, args.dp_size.max(1), router_hint);
        KvRouterPlacement::new_with_selector_seed(args, Some(config), None, &topology, Some(0))
            .unwrap()
    }

    fn assert_fetches_from_a_peer(placement: &Placement) {
        assert_ne!(
            placement
                .kv_hints
                .and_then(|hints| hints.fetch)
                .map(|fetch| fetch.source_scheduler_id),
            Some(placement.scheduler_id)
        );
    }

    pub(super) fn place_now(
        placement: &mut KvRouterPlacement,
        request: &DirectRequest,
        now_ms: f64,
    ) -> Placement {
        match placement
            .place(request, KvReplayMetadata::default(), None, now_ms)
            .unwrap()
            .decision
        {
            PlacementDecision::Immediate(placement) => {
                assert_fetches_from_a_peer(&placement);
                placement
            }
            PlacementDecision::Queued => panic!("the request was queued"),
        }
    }

    pub(super) fn fetch_hint(source_scheduler_id: usize) -> Option<KvHints> {
        Some(KvHints::fetch_from(source_scheduler_id))
    }

    /// The 8-block prompt whose prefix the KV fetch tests look for.
    pub(super) fn kv_fetch_target(uuid: u128) -> DirectRequest {
        request_with_priorities(uuid, 7, 512, 0, 0)
    }

    fn kv_fetch_target_hashes() -> Vec<LocalBlockHash> {
        compute_block_hash_for_seq(&kv_fetch_target(0).tokens, 64, BlockHashOptions::default())
    }

    /// Worker `worker_id`, rank `dp_rank`, holds the whole target prompt in `tier`.
    fn target_prefix(worker_id: WorkerId, dp_rank: u32, tier: StorageTier) -> RouterEvent {
        let mut event = prompt_store_event(worker_id, 2, &kv_fetch_target_hashes(), tier);
        event.event.dp_rank = dp_rank;
        event
    }

    pub(super) fn target_prefix_in_g2(worker_id: WorkerId, dp_rank: u32) -> RouterEvent {
        target_prefix(worker_id, dp_rank, StorageTier::HostPinned)
    }

    /// Places a request of `input_tokens` on worker `worker_id`, rank `dp_rank`,
    /// where its first block is cached in G1.
    fn load_worker(
        placement: &mut KvRouterPlacement,
        worker_id: WorkerId,
        dp_rank: u32,
        input_tokens: usize,
    ) {
        let request = request_with_priorities(100, 9, input_tokens, 0, 0);
        let first_block =
            compute_block_hash_for_seq(&request.tokens, 64, BlockHashOptions::default())[0];
        placement
            .router
            .on_kv_events(vec![store_event_for_rank(
                worker_id,
                dp_rank,
                1,
                first_block.0,
                StorageTier::Device,
            )])
            .unwrap();
        let scheduler_id = (worker_id * placement.router.dp_size as u64 + dp_rank as u64) as usize;
        assert_eq!(
            place_now(placement, &request, 0.0).scheduler_id,
            scheduler_id
        );
    }

    /// Loads worker 1, rank `dp_rank`, with a 16-block request. Its prefill
    /// load outweighs that rank's G2 credit for the target, so the selector
    /// sends the target elsewhere.
    pub(super) fn occupy_worker_1(placement: &mut KvRouterPlacement, dp_rank: u32) {
        load_worker(placement, 1, dp_rank, 1024);
    }

    #[test]
    fn kv_fetch_hints_need_a_hint_capable_target_and_holder() {
        // `router_hint` of (target worker 0, holder worker 1), whether lookups
        // retain the chain, and the target's hint.
        for (router_hint, retained, hint) in [
            ((true, true), true, fetch_hint(1)),
            ((false, false), false, None),
            ((false, true), true, None),
            ((true, false), true, None),
        ] {
            let topology = [
                worker_topology(0, 1, router_hint.0),
                worker_topology(1, 1, router_hint.1),
            ];
            let mut placement = KvRouterPlacement::new_with_selector_seed(
                &replay_args(),
                Some(router_config()),
                None,
                &topology,
                Some(0),
            )
            .unwrap();
            occupy_worker_1(&mut placement, 0);
            placement
                .router
                .on_kv_events(vec![target_prefix_in_g2(1, 0)])
                .unwrap();
            let target = kv_fetch_target(1);

            // Every lookup sees worker 1's G2 prefix.
            let pending = placement
                .router
                .build_pending_request(&target, target.max_output_tokens, None, None)
                .unwrap();
            assert_eq!(
                pending
                    .overlap
                    .tier_overlap_blocks
                    .host_pinned
                    .get(&WorkerWithDpRank::new(1, 0)),
                Some(&8)
            );
            assert_eq!(
                placement.router.indexer.retain_kv_transfer_chain, retained,
                "{router_hint:?}"
            );
            assert_eq!(
                pending.kv_transfer_candidates.is_some(),
                retained,
                "{router_hint:?}"
            );
            let placed = place_now(&mut placement, &target, 0.0);
            assert_eq!(
                (placed.scheduler_id, placed.kv_hints),
                (0, hint),
                "{router_hint:?}"
            );
        }
    }

    #[test]
    fn kv_fetch_hints_need_an_index_that_retains_the_chain() {
        // Production builds no KV index for a policy that reads neither cache nor
        // fetch inputs (overlap credit 0 zeroes every inherited cache weight), and
        // a remote, approximate or predict-on-route index retains no chain.
        for (case, (config, hinted)) in [
            (router_config(), true),
            (
                KvRouterConfig {
                    overlap_score_credit: 0.0,
                    ..router_config()
                },
                false,
            ),
            (
                KvRouterConfig {
                    use_kv_events: false,
                    ..router_config()
                },
                false,
            ),
            (
                KvRouterConfig {
                    use_remote_indexer: true,
                    ..router_config()
                },
                false,
            ),
            (
                KvRouterConfig {
                    router_predicted_ttl_secs: Some(1.0),
                    ..router_config()
                },
                false,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let reads_index = config.overlap_score_credit > 0.0;
            let mut placement = kv_fetch_placement(&replay_args(), config, 3, true);
            assert_eq!(
                placement.router.kv_transfer_gate_warned, !hinted,
                "case {case}"
            );
            // A drain recomputes the gate from the remaining hint workers.
            placement.router.remove_worker(2).unwrap();
            assert_eq!(
                placement.router.indexer.retain_kv_transfer_chain, hinted,
                "case {case}"
            );
            // Without overlap credit the load setup ties, so that case skips it.
            if reads_index {
                occupy_worker_1(&mut placement, 0);
            }
            placement
                .router
                .on_kv_events(vec![target_prefix_in_g2(1, 0)])
                .unwrap();
            let target = kv_fetch_target(1);
            let pending = placement
                .router
                .build_pending_request(&target, target.max_output_tokens, None, None)
                .unwrap();
            assert_eq!(
                pending.kv_transfer_candidates.is_some(),
                hinted,
                "case {case}"
            );
            let placed = place_now(&mut placement, &target, 0.0);
            assert_eq!(
                placed.kv_hints,
                fetch_hint(1).filter(|_| hinted),
                "case {case}"
            );
        }
    }

    /// Queues the target while both workers are busy and worker 1 holds its
    /// prefix in G2.
    fn queue_kv_fetch_target() -> KvRouterPlacement {
        let mut placement = kv_fetch_placement(&queueing_args(), queueing_router_config(), 2, true);
        occupy_worker_1(&mut placement, 0);
        let worker_0_request = request_with_priorities(101, 8, 512, 0, 0);
        assert_eq!(
            place_now(&mut placement, &worker_0_request, 0.0).scheduler_id,
            0
        );
        placement
            .router
            .on_kv_events(vec![target_prefix_in_g2(1, 0)])
            .unwrap();
        let effects = placement
            .place(&kv_fetch_target(1), KvReplayMetadata::default(), None, 0.0)
            .unwrap();
        assert!(matches!(effects.decision, PlacementDecision::Queued));
        placement
    }

    /// Completes worker 0's request and returns the admitted target.
    fn admit_queued_kv_fetch_target(placement: &mut KvRouterPlacement) -> Placement {
        let released = PlacementPolicy::<DirectRequest>::request_terminal(
            placement,
            Uuid::from_u128(101),
            10.0,
        )
        .unwrap();
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].request_id, Uuid::from_u128(1));
        assert_fetches_from_a_peer(&released[0]);
        released[0]
    }

    #[test]
    fn queued_request_gets_kv_fetch_hint_at_admission() {
        let mut placement = queue_kv_fetch_target();
        let admitted = admit_queued_kv_fetch_target(&mut placement);
        assert_eq!(
            (admitted.scheduler_id, admitted.kv_hints),
            (0, fetch_hint(1))
        );
    }

    #[test]
    fn queued_request_gets_no_kv_fetch_hint_after_holder_drains() {
        let mut placement = queue_kv_fetch_target();
        // Candidates date from arrival and still name worker 1; eligibility is
        // checked at admission.
        let queued_holders = placement
            .router
            .pending
            .entries()
            .flat_map(|entry| entry.payload().kv_transfer_candidates.iter())
            .flat_map(|candidates| candidates.owner_prefix_blocks.iter())
            .map(|&(source, _)| source)
            .collect::<Vec<_>>();
        assert_eq!(
            queued_holders,
            vec![KvTransferCandidateSource::Worker(WorkerWithDpRank::new(
                1, 0
            ))]
        );
        placement.router.remove_worker(1).unwrap();
        let admitted = admit_queued_kv_fetch_target(&mut placement);
        assert_eq!((admitted.scheduler_id, admitted.kv_hints), (0, None));
    }

    #[test]
    fn kv_fetch_source_maps_dp_rank_to_scheduler_id() {
        let mut args = replay_args();
        args.dp_size = 2;
        let mut placement = kv_fetch_placement(&args, router_config(), 2, true);
        occupy_worker_1(&mut placement, 1);
        // Worker 0 rank 0 caches the first block in G1, so the target lands there.
        placement
            .router
            .on_kv_events(vec![
                target_prefix_in_g2(1, 1),
                chain_store_event(0, 3, 0, &kv_fetch_target_hashes()[..1], StorageTier::Device),
            ])
            .unwrap();

        let placed = place_now(&mut placement, &kv_fetch_target(1), 0.0);
        assert_eq!((placed.scheduler_id, placed.kv_hints), (0, fetch_hint(3)));
    }

    #[test]
    fn removed_worker_is_never_a_kv_fetch_source() {
        let mut placement = kv_fetch_placement(&replay_args(), router_config(), 2, true);
        occupy_worker_1(&mut placement, 0);
        placement
            .router
            .on_kv_events(vec![target_prefix_in_g2(1, 0)])
            .unwrap();
        let placed = place_now(&mut placement, &kv_fetch_target(1), 0.0);
        assert_eq!((placed.scheduler_id, placed.kv_hints), (0, fetch_hint(1)));

        // A draining worker keeps its indexed blocks but is no longer a source.
        placement.router.remove_worker(1).unwrap();
        let placed = place_now(&mut placement, &kv_fetch_target(2), 0.0);
        assert_eq!((placed.scheduler_id, placed.kv_hints), (0, None));

        // Without a hint-capable worker, lookups stop retaining candidates.
        placement.router.remove_worker(0).unwrap();
        let target = kv_fetch_target(3);
        let pending = placement
            .router
            .build_pending_request(&target, target.max_output_tokens, None, None)
            .unwrap();
        assert!(!placement.router.indexer.retain_kv_transfer_chain);
        assert!(pending.kv_transfer_candidates.is_none());
    }

    #[test]
    fn worker_added_mid_run_participates_in_kv_fetch_hints() {
        // Lookups retain no chain until a worker that takes hints arrives.
        let mut placement = kv_fetch_placement(&replay_args(), router_config(), 0, true);
        assert!(!placement.router.indexer.retain_kv_transfer_chain);
        for worker_id in [0, 1] {
            PlacementPolicy::<DirectRequest>::worker_ready(
                &mut placement,
                worker_topology(worker_id, 1, true),
                0.0,
            )
            .unwrap();
        }
        occupy_worker_1(&mut placement, 0);
        placement
            .router
            .on_kv_events(vec![target_prefix_in_g2(1, 0)])
            .unwrap();

        let placed = place_now(&mut placement, &kv_fetch_target(1), 0.0);
        assert_eq!((placed.scheduler_id, placed.kv_hints), (0, fetch_hint(1)));
    }

    #[test]
    fn affinity_waiter_refreshes_kv_fetch_candidates_with_its_overlap() {
        let mut policy = kv_fetch_placement(&replay_args(), router_config(), 2, true)
            .with_conversation_affinity(Some(Default::default()))
            .unwrap();
        // The session's first request binds it to worker 0, where its block is cached.
        let first = request(1, 3);
        let first_block =
            compute_block_hash_for_seq(&first.tokens, 64, BlockHashOptions::default())[0];
        policy
            .router
            .on_kv_events(vec![store_event(0, 1, first_block.0, StorageTier::Device)])
            .unwrap();
        let admitted = policy
            .router
            .on_request_arrival_for_session(&first, None, Some("a".into()), 0.0)
            .unwrap()
            .admissions;
        assert_eq!(admitted[0].worker_idx, 0);
        // The target waits for that binding; worker 1 stores its prefix meanwhile.
        assert!(
            policy
                .router
                .on_request_arrival_for_session(&kv_fetch_target(2), None, Some("a".into()), 0.0)
                .unwrap()
                .admissions
                .is_empty()
        );
        policy
            .router
            .on_kv_events(vec![target_prefix_in_g2(1, 0)])
            .unwrap();
        PlacementPolicy::<DirectRequest>::dispatch_committed(&mut policy, Uuid::from_u128(1), 0.0)
            .unwrap();

        let released = PlacementPolicy::<DirectRequest>::advance_clock(&mut policy, 0.0).unwrap();
        assert_eq!(released.len(), 1);
        assert_fetches_from_a_peer(&released[0]);
        assert_eq!(
            (released[0].scheduler_id, released[0].kv_hints),
            (0, fetch_hint(1))
        );
    }

    #[test]
    fn kv_fetch_hint_names_a_busy_g1_holder() {
        // The chain then comes from the device tier: alone, with no lower tier, or
        // with the holder's G2 copy, which adds nothing past its G1 prefix.
        for g2_copy in [false, true] {
            let mut placement = kv_fetch_placement(&replay_args(), router_config(), 2, true);
            occupy_worker_1(&mut placement, 0);
            let mut events = vec![target_prefix(1, 0, StorageTier::Device)];
            if g2_copy {
                events.push(target_prefix_in_g2(1, 0));
            }
            placement.router.on_kv_events(events).unwrap();
            assert_eq!(placement.router.indexer.lower_tier.is_some(), g2_copy);

            let placed = place_now(&mut placement, &kv_fetch_target(1), 0.0);
            assert_eq!(
                (placed.scheduler_id, placed.kv_hints),
                (0, fetch_hint(1)),
                "g2_copy={g2_copy}"
            );
        }
    }
}

/// Per-request KVCR probe at routing (diagnostics only): with `DYNAMO_REPLAY_KVCR_PROBE=<path>` set,
/// one JSON line per admitted request: the selected worker's device and host overlap, the fetch
/// source and prefix the selection committed, and the longest prefix the router's lower-tier index
/// holds for any source. Unset, it does nothing.
mod kvcr_probe {
    use std::io::Write;
    use std::sync::{Mutex, OnceLock};

    static OUT: OnceLock<Option<Mutex<std::io::LineWriter<std::fs::File>>>> = OnceLock::new();

    fn out() -> &'static Option<Mutex<std::io::LineWriter<std::fs::File>>> {
        OUT.get_or_init(|| {
            std::env::var_os("DYNAMO_REPLAY_KVCR_PROBE")
                .and_then(|path| std::fs::File::create(path).ok())
                .map(|file| Mutex::new(std::io::LineWriter::new(file)))
        })
    }

    pub(super) fn enabled() -> bool {
        out().is_some()
    }

    pub(super) fn write(line: &str) {
        if let Some(out) = out() {
            let _ = writeln!(out.lock().unwrap(), "{line}");
        }
    }
}

/// f47k: redo a request's overlap at selection (`DYNAMO_REPLAY_REFRESH_OVERLAP=1`); read once.
fn refresh_overlap() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DYNAMO_REPLAY_REFRESH_OVERLAP").is_ok_and(|v| v == "1"))
}

/// f47k: the engine's `first_admission_kvcr_reused_input_tokens` request-log field
/// (`DYNAMO_REPLAY_KVCR_FIELD=1`); read once.
fn kvcr_field() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DYNAMO_REPLAY_KVCR_FIELD").is_ok_and(|v| v == "1"))
}

/// f47k: tell the classifier the selected worker's cached prompt tokens
/// (`DYNAMO_REPLAY_PLACED_CACHE=1`); read once.
fn placed_cache() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DYNAMO_REPLAY_PLACED_CACHE").is_ok_and(|v| v == "1"))
}

/// ks2b: send each engine rank's gauges to the classifier when they change
/// (`DYNAMO_REPLAY_ENGINE_GAUGES=1`); read once.
fn engine_gauges_to_classifier() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DYNAMO_REPLAY_ENGINE_GAUGES").is_ok_and(|v| v == "1"))
}

/// f47i: whether the classifier's worker view reports per-tier overlap
/// (`DYNAMO_REPLAY_WORKER_VIEW_TIERS=1`); read once.
fn worker_view_tiers() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("DYNAMO_REPLAY_WORKER_VIEW_TIERS").is_ok_and(|v| v == "1"))
}

/// f49b: stderr counters for the engine-gauge path (`DYNAMO_REPLAY_GAUGE_DEBUG=1`); off by default.
mod gauge_debug {
    use std::collections::BTreeMap;
    use std::sync::{Mutex, OnceLock};

    fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("DYNAMO_REPLAY_GAUGE_DEBUG").is_ok_and(|v| v == "1"))
    }

    pub(super) fn first_snapshot(ids: impl Iterator<Item = (u64, u32)>) {
        static DONE: OnceLock<()> = OnceLock::new();
        if on() && DONE.set(()).is_ok() {
            let ids: Vec<_> = ids.collect();
            eprintln!("gauge-debug replay: first snapshot workers {ids:?}");
        }
    }

    pub(super) fn forwarded(worker: u64, dp_rank: u32) {
        if !on() {
            return;
        }
        type Counts = Mutex<(u64, BTreeMap<(u64, u32), u64>)>;
        static COUNTS: OnceLock<Counts> = OnceLock::new();
        let mut c = COUNTS.get_or_init(Default::default).lock().unwrap();
        c.0 += 1;
        *c.1.entry((worker, dp_rank)).or_insert(0) += 1;
        if c.0.is_multiple_of(1000) {
            eprintln!(
                "gauge-debug replay: forwarded {} gauge events, per worker {:?}",
                c.0, c.1
            );
        }
    }
}
