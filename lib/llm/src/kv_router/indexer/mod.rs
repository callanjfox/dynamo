// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, sync::Arc, time::Duration};

use anyhow::Result;
use dynamo_kv_router::{
    ConcurrentRadixTreeCompressed,
    approx::PruneConfig,
    config::{ApproximateCachePolicyKind, KvRouterConfig},
    indexer::{
        ApproximateLruIncarnation, ApproximateLruStats, ApproximateRetentionConfig, KvIndexer,
        KvIndexerInterface, KvIndexerMetrics, KvRouterError, LowerTierIndexers, ThreadPoolIndexer,
        record_unsupported_residency_event,
    },
    protocols::{
        DpRank, KvCacheEventData, ResidencyProjection, ResidencyRoutingSnapshot, RouterEvent,
        StorageTier, WorkerId,
    },
};

// Re-export tiered-match types so internal callers (`indexer::TieredMatchDetails`)
// keep working after these types moved to `dynamo-kv-router`.
pub(crate) use dynamo_kv_router::indexer::TieredMatchDetails;
#[allow(unused_imports)]
pub(crate) use dynamo_kv_router::indexer::WireTieredMatchDetails;
use dynamo_runtime::component::Component;
use dynamo_runtime::metrics::MetricsHierarchy;
use tokio_util::sync::CancellationToken;

mod embedding_cache;
mod lookup;
mod recording;
mod recovery;
pub mod remote;
mod side;

pub use self::embedding_cache::{
    EmbeddingCacheIndexer, preprocessed_multimodal_cache_keys, try_build_cache_indexer,
};
pub(crate) use self::recording::ApproximateRequestLease;
use self::remote::RemoteIndexer;
pub use self::remote::{ServedIndexerHandle, ServedIndexerMode, ensure_served_indexer_service};
pub use self::side::SideIndexer;
#[cfg(feature = "ckf-diagnostics")]
pub(crate) use recovery::WorkerQueryHealthSnapshot;
pub(crate) use recovery::{
    DEFAULT_RECOVERY_ATTEMPT_TIMEOUT, KvEventSubscriptionHandle, RecoveryResetReason,
    RecoverySupervisor, RecoveryTarget, TargetFaultDisposition, start_target_subscriber,
};
#[cfg(test)]
pub(crate) use recovery::{WorkerQueryClient, WorkerQueryTransport};
pub(crate) use recovery::{
    start_subscriber, start_worker_kv_query_endpoint, start_worker_kv_query_endpoint_with_status,
};

/// `approx` is the optional predict-on-route side indexer. It is always local
/// to this router, even when the primary indexer is served or consumed
/// remotely. Routing decisions populate it with a short TTL; engine KV events
/// go to the primary only. `find_match_details` queries both and returns the
/// per-worker max overlap. Keeping this separate from the primary avoids the
/// sequence-hash mismatch problem: vLLM/SGLang salt their hashes with
/// cryptographic digests the router can't reproduce, so writing
/// router-computed hashes into the primary would key the same block under two
/// hashes and pollute the tree.
#[derive(Clone)]
pub enum Indexer {
    KvIndexer {
        primary: KvIndexer,
        lower_tier: LowerTierIndexers,
        approx: Option<SideIndexer>,
        primary_records_routing_decisions: bool,
    },
    Concurrent {
        primary: Arc<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>,
        lower_tier: LowerTierIndexers,
        approx: Option<SideIndexer>,
        primary_records_routing_decisions: bool,
    },
    Remote {
        primary: Arc<RemoteIndexer>,
        approx: Option<SideIndexer>,
        primary_records_routing_decisions: bool,
    },
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolvedApproximatePrimaryPolicy {
    Disabled,
    Ttl,
    Lru,
    TtlRemoteFallback,
}

fn resolve_approximate_primary_policy(
    config: &KvRouterConfig,
) -> Result<ResolvedApproximatePrimaryPolicy> {
    if config.use_kv_events
        && config.router_approximate_cache_policy == ApproximateCachePolicyKind::Lru
    {
        anyhow::bail!(
            "router_approximate_cache_policy=lru requires use_kv_events=false; the local side indexer is TTL-only"
        );
    }
    if config.overlap_score_credit <= 0.0 {
        return Ok(ResolvedApproximatePrimaryPolicy::Disabled);
    }
    if config.use_kv_events
        || config.router_approximate_cache_policy == ApproximateCachePolicyKind::Ttl
    {
        return Ok(ResolvedApproximatePrimaryPolicy::Ttl);
    }
    if config.use_remote_indexer || config.serve_indexer {
        return Ok(ResolvedApproximatePrimaryPolicy::TtlRemoteFallback);
    }
    Ok(ResolvedApproximatePrimaryPolicy::Lru)
}

async fn dump_local_events(
    mut events: Vec<RouterEvent>,
    lower_tiers: &LowerTierIndexers,
) -> Result<Vec<RouterEvent>, KvRouterError> {
    for (tier, indexer) in lower_tiers.entries() {
        events.extend(indexer.dump_events().await?.into_iter().map(|mut event| {
            event.storage_tier = tier;
            event
        }));
    }
    Ok(events)
}

/// Samples the router's REAL live routing index - not the bounded cache-loss
/// history ledger (`CacheHistory`, dynamo_component_router_cache_loss_history_*,
/// a fixed byte budget unrelated to GPU state), and not the retired
/// `kv_cache_events_applied` event counter (drifted under duplicate-store
/// events) - and publishes it as a Prometheus gauge, in tokens, per worker.
/// `worker_lookup_stats()` round-trips through every worker's event-processing
/// task, so this samples on a multi-second interval rather than per-event or
/// per-request, keeping the overhead off the request-handling hot path.
fn spawn_live_index_gauge_sampler(
    component: &Component,
    primary: Arc<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>,
    block_size: u32,
    cancellation_token: CancellationToken,
) {
    let gauge = match component.metrics().create_intgaugevec(
        "router_live_index_tokens",
        "Live KV blocks tracked by the router's real routing index, per worker, in tokens \
         (block_count x block_size). Fed directly by KV add/remove events - the router's \
         actual current view of fleet residency, distinct from the bounded cache-loss \
         history ledger (dynamo_component_router_cache_loss_history_*), which answers a \
         different question and is not tied to real GPU state.",
        // NOT "worker_id" - that name is auto-injected as a const label
        // identifying this component (the router/frontend) itself, and
        // collides with any variable label of the same name (confirmed live:
        // "Variable label name 'worker_id' conflicts with auto-injected const
        // label"). "kv_worker_id" names the downstream vLLM worker this
        // gauge is actually about, which is a different thing entirely.
        &["kv_worker_id", "dp_rank"],
        &[],
    ) {
        Ok(gauge) => gauge,
        Err(e) => {
            tracing::warn!(
                "Failed to create router_live_index_tokens gauge: {e}. Live index size will not be exported."
            );
            return;
        }
    };

    tokio::spawn(async move {
        let mut known_workers: HashSet<(u64, u32)> = HashSet::new();
        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => break,
                _ = tokio::time::sleep(Duration::from_secs(3)) => {}
            }
            let stats = primary.worker_lookup_stats().await;
            let mut seen_this_round: HashSet<(u64, u32)> = HashSet::new();
            for (worker, block_count) in &stats.worker_blocks {
                let key = (worker.worker_id, worker.dp_rank);
                seen_this_round.insert(key);
                known_workers.insert(key);
                let tokens = (*block_count as u64).saturating_mul(u64::from(block_size));
                gauge
                    .with_label_values(&[&worker.worker_id.to_string(), &worker.dp_rank.to_string()])
                    .set(tokens as i64);
            }
            // A worker with a genuinely empty index (fully evicted, or gone) is
            // filtered out of `worker_blocks` entirely (see
            // WorkerLookupStats::from_worker_block_counts) - without this, its
            // gauge would silently freeze at its last nonzero value forever
            // instead of correctly reading 0.
            for (worker_id, dp_rank) in known_workers.difference(&seen_this_round) {
                gauge
                    .with_label_values(&[&worker_id.to_string(), &dp_rank.to_string()])
                    .set(0);
            }
            known_workers = seen_this_round;
        }
    });
}

/// Samples fleet-wide KV-cache redundancy from the router's live index - how
/// much of what's tracked is duplicated across workers versus genuinely
/// unique, which a raw per-worker capacity/usage number can't distinguish
/// (a fleet that "looks" 75% full could be 75% unique content, or a much
/// smaller unique set duplicated several times over - e.g. a session
/// getting hopped between workers a lot, leaving its prefix re-materialized
/// on each one). Diagnostic-only: `ConcurrentRadixTreeCompressed::
/// redundancy_stats` walks the whole tree (see its own doc comment), so
/// this still samples less often than `spawn_live_index_gauge_sampler`'s 3s -
/// 10s (dropped from an original 30s on 2026-09-14, to compare against
/// AIPerf's own finer-grained timeslices) keeps the walk's cost off
/// anything latency-sensitive while giving 3x the resolution of the
/// original interval. Has not been through this crate's AGENTS.md-mandated
/// benchmark pass or review; not cleared for a production-scale deployment -
/// re-check this cost class specifically if lowering the interval further.
fn spawn_kv_redundancy_gauge_sampler(
    component: &Component,
    primary: Arc<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>,
    block_size: u32,
    cancellation_token: CancellationToken,
) {
    let metrics = component.metrics();
    let distinct_tokens_gauge = match metrics.create_intgauge(
        "router_kv_index_distinct_tokens",
        "Distinct KV content tracked anywhere in the fleet's live routing index, in tokens \
         (distinct_blocks x block_size) - the unique-content denominator for \
         router_kv_index_total_token_copies. Diagnostic-only, sampled on a slow background \
         interval; not yet benchmarked at production scale.",
        &[],
    ) {
        Ok(g) => g,
        Err(e) => {
            tracing::warn!(
                "Failed to create router_kv_index_distinct_tokens gauge: {e}. KV redundancy will not be exported."
            );
            return;
        }
    };
    let total_copies_gauge = match metrics.create_intgauge(
        "router_kv_index_total_token_copies",
        "Sum, across every distinct tracked block's token content, of how many workers hold a \
         copy of it (total_block_copies x block_size) - equal to \
         router_kv_index_distinct_tokens when nothing in the fleet is duplicated, higher means \
         real duplication (e.g. a session's prefix re-materializing on a second worker after a \
         router hop). Diagnostic-only, sampled on a slow background interval.",
        &[],
    ) {
        Ok(g) => g,
        Err(e) => {
            tracing::warn!(
                "Failed to create router_kv_index_total_token_copies gauge: {e}. KV redundancy will not be exported."
            );
            return;
        }
    };
    let redundancy_ratio_gauge = match metrics.create_gauge(
        "router_kv_index_redundancy_ratio",
        "router_kv_index_total_token_copies / router_kv_index_distinct_tokens - 1.0 means no \
         duplication anywhere in the fleet, 2.0 means the average tracked content exists on two \
         workers, etc. block_size cancels out of this ratio, so it reads identically whether \
         computed from tokens or blocks. Diagnostic-only, sampled on a slow background interval.",
        &[],
    ) {
        Ok(g) => g,
        Err(e) => {
            tracing::warn!(
                "Failed to create router_kv_index_redundancy_ratio gauge: {e}. KV redundancy will not be exported."
            );
            return;
        }
    };

    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => break,
                _ = tokio::time::sleep(Duration::from_secs(10)) => {}
            }
            let stats = primary.redundancy_stats();
            let distinct_tokens = stats.distinct_blocks.saturating_mul(u64::from(block_size));
            let total_token_copies = stats.total_block_copies.saturating_mul(u64::from(block_size));
            distinct_tokens_gauge.set(distinct_tokens as i64);
            total_copies_gauge.set(total_token_copies as i64);
            if let Some(ratio) = stats.redundancy_ratio() {
                redundancy_ratio_gauge.set(ratio);
            }
        }
    });
}

/// Lower-tier (secondary-cache, e.g. KVCR host-pinned/G2) counterpart to
/// `spawn_kv_redundancy_gauge_sampler` above - same purpose and shape, but
/// samples `LowerTierIndexer::redundancy_stats()` (a `DashMap` edge-map walk,
/// `indexer/lower_tier.rs`) instead of the primary radix tree's BFS. The two
/// backends are structurally unrelated, so this reads a genuinely different
/// index, not a relabeled view of the same one.
///
/// Scoped to `StorageTier::HostPinned` only (the one lower tier this
/// deployment actually configures) rather than iterating every possible
/// tier - `LowerTierIndexers` allocates each tier lazily on first event, so
/// `get()` returns `None` (silently skipped, not an error) until this
/// worker's fleet has offloaded at least one block to that tier.
///
/// Metric names are tier-suffixed rather than sharing the primary gauges'
/// names with a `tier` label: the primary gauges above were registered as
/// plain (unlabeled) gauges via `create_intgauge(..., &[])`, and retrofitting
/// a label dimension onto an already-registered bare gauge name risks a
/// Prometheus registration conflict - out of scope for this basic-
/// functionality pass. Diagnostic-only, same caveats as the primary sampler
/// (not benchmarked, background-only, never a per-request path).
fn spawn_lower_tier_redundancy_gauge_sampler(
    component: &Component,
    lower_tier: LowerTierIndexers,
    block_size: u32,
    cancellation_token: CancellationToken,
) {
    let metrics = component.metrics();
    let distinct_tokens_gauge = match metrics.create_intgauge(
        "router_kv_index_distinct_tokens_host_pinned",
        "Distinct KV content tracked in the fleet's host-pinned (secondary-cache, e.g. KVCR G2) \
         lower-tier index, in tokens (distinct_blocks x block_size) - the unique-content \
         denominator for router_kv_index_total_token_copies_host_pinned. Diagnostic-only, \
         sampled on a slow background interval; not yet benchmarked at production scale.",
        &[],
    ) {
        Ok(g) => g,
        Err(e) => {
            tracing::warn!(
                "Failed to create router_kv_index_distinct_tokens_host_pinned gauge: {e}. \
                 Lower-tier KV redundancy will not be exported."
            );
            return;
        }
    };
    let total_copies_gauge = match metrics.create_intgauge(
        "router_kv_index_total_token_copies_host_pinned",
        "Sum, across every distinct tracked host-pinned-tier block's token content, of how many \
         workers hold a copy of it (total_block_copies x block_size) - equal to \
         router_kv_index_distinct_tokens_host_pinned when nothing in the fleet's host-pinned \
         tier is duplicated, higher means real duplication. Diagnostic-only, sampled on a slow \
         background interval.",
        &[],
    ) {
        Ok(g) => g,
        Err(e) => {
            tracing::warn!(
                "Failed to create router_kv_index_total_token_copies_host_pinned gauge: {e}. \
                 Lower-tier KV redundancy will not be exported."
            );
            return;
        }
    };
    let redundancy_ratio_gauge = match metrics.create_gauge(
        "router_kv_index_redundancy_ratio_host_pinned",
        "router_kv_index_total_token_copies_host_pinned / \
         router_kv_index_distinct_tokens_host_pinned - 1.0 means no duplication anywhere in the \
         fleet's host-pinned tier, 2.0 means the average tracked content exists on two workers, \
         etc. block_size cancels out of this ratio, so it reads identically whether computed \
         from tokens or blocks. Diagnostic-only, sampled on a slow background interval.",
        &[],
    ) {
        Ok(g) => g,
        Err(e) => {
            tracing::warn!(
                "Failed to create router_kv_index_redundancy_ratio_host_pinned gauge: {e}. \
                 Lower-tier KV redundancy will not be exported."
            );
            return;
        }
    };

    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => break,
                _ = tokio::time::sleep(Duration::from_secs(10)) => {}
            }
            let Some(host_pinned) = lower_tier.get(StorageTier::HostPinned) else {
                // Lazily allocated on first event - nothing to sample yet.
                continue;
            };
            let stats = host_pinned.redundancy_stats();
            let distinct_tokens = stats.distinct_blocks.saturating_mul(u64::from(block_size));
            let total_token_copies = stats.total_block_copies.saturating_mul(u64::from(block_size));
            distinct_tokens_gauge.set(distinct_tokens as i64);
            total_copies_gauge.set(total_token_copies as i64);
            if let Some(ratio) = stats.redundancy_ratio() {
                redundancy_ratio_gauge.set(ratio);
            }
        }
    });
}

/// Live/resident block age for the primary (device/G1) index
/// (`KVCR_INDEX_HEALTH_DESIGN.md` 1b): how old is the data currently sitting in the fleet's
/// live routing index, as min/p50/p99/max seconds since each block was first observed
/// stored. Reads `ThreadPoolIndexer::resident_age_percentiles()`, which walks only the
/// `age_tracking` side-table (see that module) - never the primary tree itself. Same 10s
/// interval (see the redundancy sampler's own doc comment for why 10s, not the original 30s)
/// and diagnostic-only status as the redundancy samplers above; not yet benchmarked at
/// production scale. A restart resets every tracked insertion time (no recovery path), so
/// expect misleadingly low ages for a while after any router restart.
fn spawn_kv_resident_age_gauge_sampler(
    component: &Component,
    primary: Arc<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>,
    cancellation_token: CancellationToken,
) {
    let metrics = component.metrics();
    let resident_age_gauge = match metrics.create_gaugevec(
        "router_kv_index_resident_age_seconds",
        "Age (seconds since first observed stored) of blocks currently resident in the fleet's \
         live primary (device/G1) routing index, by stat (min/p50/p99/max). Diagnostic-only, \
         sampled on a slow background interval; not yet benchmarked at production scale. \
         Expect misleadingly low values for a while after any router restart - tracked \
         insertion times are not recoverable across a restart.",
        &["stat"],
        &[],
    ) {
        Ok(g) => g,
        Err(e) => {
            tracing::warn!(
                "Failed to create router_kv_index_resident_age_seconds gauge: {e}. \
                 Resident-age reporting will not be exported."
            );
            return;
        }
    };

    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => break,
                _ = tokio::time::sleep(Duration::from_secs(10)) => {}
            }
            let Some(stats) = primary.resident_age_percentiles() else {
                continue;
            };
            resident_age_gauge.with_label_values(&["min"]).set(stats.min);
            resident_age_gauge.with_label_values(&["p50"]).set(stats.p50);
            resident_age_gauge.with_label_values(&["p99"]).set(stats.p99);
            resident_age_gauge.with_label_values(&["max"]).set(stats.max);
        }
    });
}

/// Lower-tier (host-pinned/G2) counterpart to `spawn_kv_resident_age_gauge_sampler` above -
/// same purpose and mechanism (reads the same `age_tracking` side-table type, this time the
/// one owned by the host-pinned `ThreadPoolIndexer<LowerTierIndexer>`), tier-suffixed metric
/// name for the same registration-conflict reason as the redundancy samplers. Scoped to
/// `StorageTier::HostPinned` only, same lazy-allocation caveat as
/// `spawn_lower_tier_redundancy_gauge_sampler`.
fn spawn_lower_tier_resident_age_gauge_sampler(
    component: &Component,
    lower_tier: LowerTierIndexers,
    cancellation_token: CancellationToken,
) {
    let metrics = component.metrics();
    let resident_age_gauge = match metrics.create_gaugevec(
        "router_kv_index_resident_age_seconds_host_pinned",
        "Age (seconds since first observed stored) of blocks currently resident in the fleet's \
         host-pinned (secondary-cache, e.g. KVCR G2) lower-tier index, by stat \
         (min/p50/p99/max). Diagnostic-only, sampled on a slow background interval; not yet \
         benchmarked at production scale. Expect misleadingly low values for a while after any \
         router restart - tracked insertion times are not recoverable across a restart.",
        &["stat"],
        &[],
    ) {
        Ok(g) => g,
        Err(e) => {
            tracing::warn!(
                "Failed to create router_kv_index_resident_age_seconds_host_pinned gauge: {e}. \
                 Lower-tier resident-age reporting will not be exported."
            );
            return;
        }
    };

    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => break,
                _ = tokio::time::sleep(Duration::from_secs(10)) => {}
            }
            let Some(host_pinned) = lower_tier.get(StorageTier::HostPinned) else {
                continue;
            };
            let Some(stats) = host_pinned.resident_age_percentiles() else {
                continue;
            };
            resident_age_gauge.with_label_values(&["min"]).set(stats.min);
            resident_age_gauge.with_label_values(&["p50"]).set(stats.p50);
            resident_age_gauge.with_label_values(&["p99"]).set(stats.p99);
            resident_age_gauge.with_label_values(&["max"]).set(stats.max);
        }
    });
}

/// Cumulative distinct-block ("unconstrained demand", `DASHBOARD_METRICS_ENGINEERING_PLAN.md`
/// section 2b) gauges for the primary (device/G1) index, reported as three simultaneous
/// load-average-style windows (1m/5m/15m) rather than one fixed window - see
/// `demand_tracking`'s module docs for why three windows and not one, and why this answers "is
/// my active working set trending up right now," not a capacity-sizing number. Reads
/// `ThreadPoolIndexer::demand_distinct_blocks()`, which walks only the `demand_tracking`
/// side-table (see that module) - never the primary tree itself. Same 10s interval (see the
/// redundancy sampler's own doc comment for why 10s, not the original 30s) and diagnostic-only
/// status as the redundancy/age samplers above; not yet benchmarked at
/// production scale (`DASHBOARD_METRICS_ENGINEERING_PLAN.md` section 7). Three separate metric
/// names rather than one name with a `window` label, matching the `_host_pinned` samplers'
/// reasoning elsewhere in this file (avoids retrofitting a label dimension onto an
/// already-registered bare gauge) and node_exporter's own convention for load averages
/// (`node_load1`/`node_load5`/`node_load15`).
fn spawn_kv_demand_gauge_sampler(
    component: &Component,
    primary: Arc<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>,
    block_size: u32,
    cancellation_token: CancellationToken,
) {
    let metrics = component.metrics();
    let make_gauge = |suffix: &str, window_desc: &str| {
        let name = format!("router_kv_index_demand_distinct_tokens_{suffix}");
        metrics
            .create_intgauge(
                &name,
                &format!(
                    "Distinct KV content (in tokens: distinct blocks x block_size) this fleet's \
                     primary (device/G1) index has observed stored at all over the trailing \
                     {window_desc}, regardless of current residency - a load-average-style \
                     \"is my active working set trending up\" signal, not a capacity-sizing \
                     number (use the offline trace-based approach in \
                     DASHBOARD_METRICS_ENGINEERING_PLAN.md section 2a for that). Diagnostic-only, \
                     sampled on a slow background interval; not yet benchmarked at production \
                     scale. Resets to 0 on every router restart with no recovery path.",
                ),
                &[],
            )
            .inspect_err(|e| {
                tracing::warn!("Failed to create {name} gauge: {e}. Demand tracking will not be exported.");
            })
            .ok()
    };
    let Some(gauge_1m) = make_gauge("1m", "60 seconds") else {
        return;
    };
    let Some(gauge_5m) = make_gauge("5m", "5 minutes") else {
        return;
    };
    let Some(gauge_15m) = make_gauge("15m", "15 minutes") else {
        return;
    };

    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => break,
                _ = tokio::time::sleep(Duration::from_secs(10)) => {}
            }
            let counts = primary.demand_distinct_blocks();
            let block_size = u64::from(block_size);
            gauge_1m.set((counts.last_1m as u64).saturating_mul(block_size) as i64);
            gauge_5m.set((counts.last_5m as u64).saturating_mul(block_size) as i64);
            gauge_15m.set((counts.last_15m as u64).saturating_mul(block_size) as i64);
        }
    });
}

/// Lower-tier (host-pinned/G2) counterpart to `spawn_kv_demand_gauge_sampler` above - same
/// purpose and mechanism, tier-suffixed metric names for the same registration-conflict reason
/// as the redundancy/age samplers. Scoped to `StorageTier::HostPinned` only, same
/// lazy-allocation caveat as `spawn_lower_tier_redundancy_gauge_sampler`.
fn spawn_lower_tier_demand_gauge_sampler(
    component: &Component,
    lower_tier: LowerTierIndexers,
    block_size: u32,
    cancellation_token: CancellationToken,
) {
    let metrics = component.metrics();
    let make_gauge = |suffix: &str, window_desc: &str| {
        let name = format!("router_kv_index_demand_distinct_tokens_{suffix}_host_pinned");
        metrics
            .create_intgauge(
                &name,
                &format!(
                    "Distinct KV content (in tokens: distinct blocks x block_size) this fleet's \
                     host-pinned (secondary-cache, e.g. KVCR G2) lower-tier index has observed \
                     stored at all over the trailing {window_desc}, regardless of current \
                     residency. Diagnostic-only, sampled on a slow background interval; not yet \
                     benchmarked at production scale. Resets to 0 on every router restart with \
                     no recovery path.",
                ),
                &[],
            )
            .inspect_err(|e| {
                tracing::warn!(
                    "Failed to create {name} gauge: {e}. Lower-tier demand tracking will not be exported."
                );
            })
            .ok()
    };
    let Some(gauge_1m) = make_gauge("1m", "60 seconds") else {
        return;
    };
    let Some(gauge_5m) = make_gauge("5m", "5 minutes") else {
        return;
    };
    let Some(gauge_15m) = make_gauge("15m", "15 minutes") else {
        return;
    };

    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => break,
                _ = tokio::time::sleep(Duration::from_secs(10)) => {}
            }
            let Some(host_pinned) = lower_tier.get(StorageTier::HostPinned) else {
                continue;
            };
            let counts = host_pinned.demand_distinct_blocks();
            let block_size = u64::from(block_size);
            gauge_1m.set((counts.last_1m as u64).saturating_mul(block_size) as i64);
            gauge_5m.set((counts.last_5m as u64).saturating_mul(block_size) as i64);
            gauge_15m.set((counts.last_15m as u64).saturating_mul(block_size) as i64);
        }
    });
}

impl Indexer {
    /// Publish a control-plane projection snapshot for subsequent lookups.
    ///
    /// Discovery and attachment reconciliation stay in lib/llm; router-core
    /// only consumes this immutable resolved view.
    pub fn set_residency_projection(&self, projection: ResidencyProjection) {
        match self {
            Self::KvIndexer { lower_tier, .. } | Self::Concurrent { lower_tier, .. } => {
                lower_tier.set_residency_projection(projection)
            }
            Self::Remote { .. } | Self::None => {}
        }
    }

    pub fn set_residency_routing_snapshot(&self, snapshot: ResidencyRoutingSnapshot) {
        match self {
            Self::KvIndexer { lower_tier, .. } | Self::Concurrent { lower_tier, .. } => {
                lower_tier.set_residency_routing_snapshot(snapshot)
            }
            Self::Remote { .. } | Self::None => {}
        }
    }

    pub(crate) fn supports_overlap_refresh(&self) -> bool {
        matches!(self, Self::KvIndexer { .. } | Self::Concurrent { .. })
    }

    pub(crate) fn supports_kv_transfer_chain_retention(&self) -> bool {
        matches!(
            self,
            Self::KvIndexer {
                approx: None,
                primary_records_routing_decisions: false,
                ..
            } | Self::Concurrent {
                approx: None,
                primary_records_routing_decisions: false,
                ..
            }
        )
    }

    pub async fn new(
        component: &Component,
        kv_router_config: &KvRouterConfig,
        block_size: u32,
        model_name: Option<&str>,
        cancellation_token: CancellationToken,
    ) -> Result<Self> {
        let approximate_policy = resolve_approximate_primary_policy(kv_router_config)?;
        if approximate_policy == ResolvedApproximatePrimaryPolicy::Disabled {
            return Ok(Self::None);
        }

        if approximate_policy == ResolvedApproximatePrimaryPolicy::TtlRemoteFallback {
            tracing::warn!(
                use_remote_indexer = kv_router_config.use_remote_indexer,
                serve_indexer = kv_router_config.serve_indexer,
                "Approximate LRU requires a router-local primary indexer; falling back to TTL"
            );
        }

        if kv_router_config.router_predicted_ttl_secs.is_some() && !kv_router_config.use_kv_events {
            anyhow::bail!(
                "router_predicted_ttl_secs requires use_kv_events=true; \
                 do not combine a primary approximate indexer with a side approximate indexer"
            );
        }
        if kv_router_config.use_remote_indexer {
            let model_name = model_name
                .ok_or_else(|| {
                    anyhow::anyhow!("model_name is required when use_remote_indexer is configured")
                })?
                .to_string();
            let indexer_component_name = component.name();
            tracing::info!(
                indexer_component = %indexer_component_name,
                model_name,
                "Using remote KV indexer"
            );
            let remote =
                RemoteIndexer::new(component, model_name, kv_router_config.use_kv_events).await?;
            let approx = SideIndexer::new_predict_on_route(
                component,
                kv_router_config,
                block_size,
                cancellation_token.child_token(),
            );
            return Ok(Self::Remote {
                primary: Arc::new(remote),
                approx,
                primary_records_routing_decisions: !kv_router_config.use_kv_events,
            });
        }

        if !kv_router_config.use_kv_events {
            let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
            let prune_config = PruneConfig {
                ttl: Duration::from_secs_f64(kv_router_config.router_ttl_secs),
            };
            let retention = if approximate_policy == ResolvedApproximatePrimaryPolicy::Lru {
                tracing::info!(
                    "Starting local primary approximate indexer with capacity-bounded LRU retention"
                );
                ApproximateRetentionConfig::Lru {
                    fallback_ttl: prune_config,
                }
            } else {
                ApproximateRetentionConfig::Ttl(prune_config)
            };
            if kv_router_config.router_event_threads > 1 {
                return Ok(Self::Concurrent {
                    primary: Arc::new(
                        ThreadPoolIndexer::new_with_metrics_and_approximate_retention(
                            ConcurrentRadixTreeCompressed::new(),
                            kv_router_config.router_event_threads as usize,
                            block_size,
                            Some(kv_indexer_metrics.clone()),
                            Some(retention),
                        ),
                    ),
                    lower_tier: LowerTierIndexers::new_with_metrics(
                        kv_router_config.router_event_threads as usize,
                        block_size,
                        Some(kv_indexer_metrics),
                    ),
                    approx: None,
                    primary_records_routing_decisions: true,
                });
            }

            return Ok(Self::KvIndexer {
                primary: KvIndexer::new_with_approximate_retention(
                    cancellation_token.child_token(),
                    block_size,
                    kv_indexer_metrics.clone(),
                    Some(retention),
                ),
                lower_tier: LowerTierIndexers::new_with_metrics(
                    1,
                    block_size,
                    Some(kv_indexer_metrics),
                ),
                approx: None,
                primary_records_routing_decisions: true,
            });
        }

        let approx = SideIndexer::new_predict_on_route(
            component,
            kv_router_config,
            block_size,
            cancellation_token.child_token(),
        );

        if kv_router_config.router_event_threads > 1 {
            let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
            let primary = Arc::new(ThreadPoolIndexer::new_with_metrics(
                ConcurrentRadixTreeCompressed::new(),
                kv_router_config.router_event_threads as usize,
                block_size,
                Some(kv_indexer_metrics.clone()),
            ));
            spawn_live_index_gauge_sampler(
                component,
                primary.clone(),
                block_size,
                cancellation_token.child_token(),
            );
            spawn_kv_redundancy_gauge_sampler(
                component,
                primary.clone(),
                block_size,
                cancellation_token.child_token(),
            );
            spawn_kv_resident_age_gauge_sampler(
                component,
                primary.clone(),
                cancellation_token.child_token(),
            );
            spawn_kv_demand_gauge_sampler(
                component,
                primary.clone(),
                block_size,
                cancellation_token.child_token(),
            );
            let lower_tier = LowerTierIndexers::new_with_metrics(
                kv_router_config.router_event_threads as usize,
                block_size,
                Some(kv_indexer_metrics),
            );
            spawn_lower_tier_redundancy_gauge_sampler(
                component,
                lower_tier.clone(),
                block_size,
                cancellation_token.child_token(),
            );
            spawn_lower_tier_resident_age_gauge_sampler(
                component,
                lower_tier.clone(),
                cancellation_token.child_token(),
            );
            spawn_lower_tier_demand_gauge_sampler(
                component,
                lower_tier.clone(),
                block_size,
                cancellation_token.child_token(),
            );
            return Ok(Self::Concurrent {
                primary,
                lower_tier,
                approx,
                primary_records_routing_decisions: false,
            });
        }

        let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
        Ok(Self::KvIndexer {
            primary: KvIndexer::new_with_pruning(
                cancellation_token.child_token(),
                block_size,
                kv_indexer_metrics.clone(),
                None,
            ),
            lower_tier: LowerTierIndexers::new_with_metrics(
                1,
                block_size,
                Some(kv_indexer_metrics),
            ),
            approx,
            primary_records_routing_decisions: false,
        })
    }

    pub(crate) async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        match self {
            Self::KvIndexer {
                primary,
                lower_tier,
                ..
            } => dump_local_events(primary.dump_events().await?, lower_tier).await,
            Self::Concurrent {
                primary,
                lower_tier,
                ..
            } => dump_local_events(primary.dump_events().await?, lower_tier).await,
            Self::Remote { .. } => Ok(Vec::new()),
            Self::None => Err(KvRouterError::Unsupported(
                "event dumping requires a KV indexer".to_string(),
            )),
        }
    }

    pub(crate) async fn try_apply_event(&self, event: RouterEvent) -> Result<(), KvRouterError> {
        let targets_primary = match event.targets_primary() {
            Ok(targets_primary) => targets_primary,
            Err(_) => {
                match self {
                    Self::KvIndexer { lower_tier, .. } | Self::Concurrent { lower_tier, .. } => {
                        lower_tier.record_unsupported_residency_event(&event);
                    }
                    Self::Remote { .. } | Self::None => {
                        record_unsupported_residency_event(None, &event);
                    }
                }
                return Ok(());
            }
        };
        let is_clear = matches!(&event.event.data, KvCacheEventData::Cleared);
        match self {
            Self::KvIndexer {
                primary,
                lower_tier,
                ..
            } => {
                if is_clear {
                    if targets_primary {
                        primary
                            .reset_worker_dp_rank_and_wait(event.worker_id, event.event.dp_rank)
                            .await?;
                    }

                    for indexer in lower_tier.all() {
                        indexer.apply_event_and_wait(event.clone()).await?;
                    }
                } else if targets_primary {
                    primary
                        .event_sender()
                        .send(event)
                        .await
                        .map_err(|_| KvRouterError::IndexerOffline)?;
                } else {
                    lower_tier
                        .get_or_create(event.storage_tier)
                        .enqueue_event(event)?;
                }
            }
            Self::Concurrent {
                primary,
                lower_tier,
                ..
            } => {
                if is_clear {
                    if targets_primary {
                        primary.apply_event_and_wait(event.clone()).await?;
                    }

                    for indexer in lower_tier.all() {
                        indexer.apply_event_and_wait(event.clone()).await?;
                    }
                } else if targets_primary {
                    primary.enqueue_event(event)?;
                } else {
                    lower_tier
                        .get_or_create(event.storage_tier)
                        .enqueue_event(event)?;
                }
            }
            Self::Remote { .. } | Self::None => {}
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) async fn apply_event(&self, event: RouterEvent) {
        if let Err(error) = self.try_apply_event(event).await {
            tracing::error!(%error, "Failed to enqueue KV event");
        }
    }

    /// Cold-reset one logical rank and wait until all local index tiers have completed the removal.
    ///
    /// NOTE: Unlike ordinary event application, rank removal is an infallible lane operation.
    /// Its FIFO completion must be visible before source activation or clearing a pending reset.
    pub(crate) async fn reset_worker_dp_rank_and_wait(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
    ) -> Result<(), KvRouterError> {
        match self {
            Self::KvIndexer {
                primary,
                lower_tier,
                approx,
                ..
            } => {
                primary
                    .reset_worker_dp_rank_and_wait(worker_id, dp_rank)
                    .await?;
                for indexer in lower_tier.all() {
                    indexer
                        .reset_worker_dp_rank_and_wait(worker_id, dp_rank)
                        .await?;
                }
                if let Some(approx) = approx {
                    approx
                        .reset_worker_dp_rank_and_wait(worker_id, dp_rank)
                        .await?;
                }
            }
            Self::Concurrent {
                primary,
                lower_tier,
                approx,
                ..
            } => {
                primary
                    .reset_worker_dp_rank_and_wait(worker_id, dp_rank)
                    .await?;
                for indexer in lower_tier.all() {
                    indexer
                        .reset_worker_dp_rank_and_wait(worker_id, dp_rank)
                        .await?;
                }
                if let Some(approx) = approx {
                    approx
                        .reset_worker_dp_rank_and_wait(worker_id, dp_rank)
                        .await?;
                }
            }
            Self::Remote { approx, .. } => {
                if let Some(approx) = approx {
                    approx
                        .reset_worker_dp_rank_and_wait(worker_id, dp_rank)
                        .await?;
                }
            }
            Self::None => {}
        }
        Ok(())
    }

    pub(crate) fn uses_approximate_lru(&self) -> bool {
        match self {
            Self::KvIndexer { primary, .. } => primary.approximate_lru_enabled(),
            Self::Concurrent { primary, .. } => primary.approximate_lru_enabled(),
            Self::Remote { .. } | Self::None => false,
        }
    }

    pub(crate) fn set_approximate_lru_capacity_now(
        &self,
        worker: dynamo_kv_router::protocols::WorkerWithDpRank,
        incarnation: ApproximateLruIncarnation,
        capacity: Option<usize>,
    ) -> Result<(), KvRouterError> {
        match self {
            Self::KvIndexer { primary, .. } => {
                primary.set_approximate_lru_capacity_now(worker, incarnation, capacity)
            }
            Self::Concurrent { primary, .. } => {
                primary.set_approximate_lru_capacity_now(worker, incarnation, capacity)
            }
            Self::Remote { .. } | Self::None => Ok(()),
        }
    }

    pub(crate) async fn approximate_lru_stats(&self) -> Result<ApproximateLruStats, KvRouterError> {
        match self {
            Self::KvIndexer { primary, .. } => primary.approximate_lru_stats().await,
            Self::Concurrent { primary, .. } => primary.approximate_lru_stats().await,
            Self::Remote { .. } | Self::None => Ok(ApproximateLruStats::default()),
        }
    }
}

#[cfg(test)]
pub(super) mod test_util {
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, RouterEvent, StorageTier,
        compute_seq_hash_for_block,
    };

    pub(crate) fn store_event(
        worker_id: u64,
        dp_rank: u32,
        event_id: u64,
        prefix_hashes: &[u64],
        local_hashes: &[u64],
        storage_tier: StorageTier,
    ) -> RouterEvent {
        let prefix_block_hashes: Vec<LocalBlockHash> =
            prefix_hashes.iter().copied().map(LocalBlockHash).collect();
        let parent_hash = compute_seq_hash_for_block(&prefix_block_hashes)
            .last()
            .copied()
            .map(ExternalSequenceBlockHash);

        let full_hashes: Vec<LocalBlockHash> = prefix_hashes
            .iter()
            .chain(local_hashes.iter())
            .copied()
            .map(LocalBlockHash)
            .collect();
        let full_sequence_hashes = compute_seq_hash_for_block(&full_hashes);
        let new_sequence_hashes = &full_sequence_hashes[prefix_hashes.len()..];
        let blocks = local_hashes
            .iter()
            .zip(new_sequence_hashes.iter())
            .map(|(&local_hash, &sequence_hash)| KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(sequence_hash),
                tokens_hash: LocalBlockHash(local_hash),
                mm_extra_info: None,
            })
            .collect();

        RouterEvent::with_storage_tier(
            worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash,
                    start_position: None,
                    blocks,
                }),
                dp_rank,
            },
            storage_tier,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use tokio_util::sync::CancellationToken;

    use super::test_util::store_event;
    use super::{Indexer, LowerTierIndexers};
    use dynamo_kv_router::{
        ConcurrentRadixTreeCompressed, ThreadPoolIndexer,
        approx::PruneConfig,
        indexer::{KvIndexer, KvIndexerInterface, KvIndexerMetrics, RoutingDecisionHashes},
        protocols::{
            BlockHashOptions, LocalBlockHash, StorageTier, TokensWithHashes, WorkerWithDpRank,
            compute_block_hash_for_seq, compute_seq_hash_for_block,
        },
    };

    fn make_test_indexer() -> Indexer {
        Indexer::KvIndexer {
            primary: KvIndexer::new(
                CancellationToken::new(),
                4,
                Arc::new(KvIndexerMetrics::new_unregistered()),
            ),
            lower_tier: LowerTierIndexers::new(1, 4),
            approx: None,
            primary_records_routing_decisions: false,
        }
    }

    fn make_test_concurrent_indexer() -> Indexer {
        Indexer::Concurrent {
            primary: Arc::new(ThreadPoolIndexer::new(
                ConcurrentRadixTreeCompressed::new(),
                2,
                4,
            )),
            lower_tier: LowerTierIndexers::new(2, 4),
            approx: None,
            primary_records_routing_decisions: false,
        }
    }

    fn make_test_concurrent_approx_indexer() -> Indexer {
        Indexer::Concurrent {
            primary: Arc::new(ThreadPoolIndexer::new_with_pruning(
                ConcurrentRadixTreeCompressed::new(),
                2,
                4,
                PruneConfig {
                    ttl: Duration::from_secs(60),
                },
            )),
            lower_tier: LowerTierIndexers::new(2, 4),
            approx: None,
            primary_records_routing_decisions: true,
        }
    }

    #[test]
    fn overlap_refresh_is_limited_to_local_indexers() {
        assert!(make_test_indexer().supports_overlap_refresh());
        assert!(make_test_concurrent_indexer().supports_overlap_refresh());
        assert!(!Indexer::None.supports_overlap_refresh());
    }

    #[test]
    fn kv_transfer_chain_retention_requires_event_driven_primary() {
        assert!(make_test_indexer().supports_kv_transfer_chain_retention());
        assert!(make_test_concurrent_indexer().supports_kv_transfer_chain_retention());
        assert!(!make_test_concurrent_approx_indexer().supports_kv_transfer_chain_retention());
        assert!(!Indexer::None.supports_kv_transfer_chain_retention());
    }

    async fn flush_indexer(indexer: &Indexer) {
        match indexer {
            Indexer::KvIndexer {
                primary,
                lower_tier,
                ..
            } => {
                let _ = primary.flush().await;
                for indexer in lower_tier.all() {
                    let _ = indexer.dump_events().await.unwrap();
                }
            }
            Indexer::Concurrent {
                primary,
                lower_tier,
                ..
            } => {
                primary.flush().await;
                for indexer in lower_tier.all() {
                    let _ = indexer.dump_events().await.unwrap();
                }
            }
            Indexer::Remote { .. } | Indexer::None => {}
        }
    }

    async fn assert_rank_reset_is_acknowledged(indexer: Indexer) {
        let reset_rank = WorkerWithDpRank::new(7, 0);
        let retained_rank = WorkerWithDpRank::new(7, 1);

        for dp_rank in [reset_rank.dp_rank, retained_rank.dp_rank] {
            indexer
                .apply_event(store_event(7, dp_rank, 1, &[], &[41], StorageTier::Device))
                .await;
            indexer
                .apply_event(store_event(
                    7,
                    dp_rank,
                    2,
                    &[41],
                    &[42],
                    StorageTier::HostPinned,
                ))
                .await;
        }
        flush_indexer(&indexer).await;

        indexer
            .reset_worker_dp_rank_and_wait(reset_rank.worker_id, reset_rank.dp_rank)
            .await
            .unwrap();

        let matches = indexer
            .find_matches_by_tier(vec![LocalBlockHash(41), LocalBlockHash(42)])
            .await
            .unwrap();
        assert!(
            !matches
                .device
                .overlap_scores
                .scores
                .contains_key(&reset_rank)
        );
        assert_eq!(
            matches.device.overlap_scores.scores.get(&retained_rank),
            Some(&1)
        );
        let host_hits = &matches
            .lower_tier
            .get(&StorageTier::HostPinned)
            .unwrap()
            .hits;
        assert!(!host_hits.contains_key(&reset_rank));
        assert_eq!(host_hits.get(&retained_rank), Some(&1));
    }

    #[tokio::test]
    async fn single_thread_rank_reset_waits_for_all_local_tiers() {
        assert_rank_reset_is_acknowledged(make_test_indexer()).await;
    }

    #[tokio::test]
    async fn concurrent_rank_reset_waits_for_all_local_tiers() {
        assert_rank_reset_is_acknowledged(make_test_concurrent_indexer()).await;
    }

    #[tokio::test]
    async fn tiered_query_chains_device_host_and_disk() {
        let indexer = make_test_indexer();
        let worker = WorkerWithDpRank::new(7, 0);

        indexer
            .apply_event(store_event(7, 0, 1, &[], &[11, 12], StorageTier::Device))
            .await;
        indexer
            .apply_event(store_event(
                7,
                0,
                2,
                &[11, 12],
                &[13],
                StorageTier::HostPinned,
            ))
            .await;
        indexer
            .apply_event(store_event(
                7,
                0,
                3,
                &[11, 12, 13],
                &[14],
                StorageTier::Disk,
            ))
            .await;
        flush_indexer(&indexer).await;

        let matches = indexer
            .find_matches_by_tier(vec![
                LocalBlockHash(11),
                LocalBlockHash(12),
                LocalBlockHash(13),
                LocalBlockHash(14),
            ])
            .await
            .unwrap();

        assert_eq!(matches.device.overlap_scores.scores.get(&worker), Some(&2));
        assert_eq!(
            matches
                .lower_tier
                .get(&StorageTier::HostPinned)
                .and_then(|tier| tier.hits.get(&worker)),
            Some(&1)
        );
        assert_eq!(
            matches
                .lower_tier
                .get(&StorageTier::Disk)
                .and_then(|tier| tier.hits.get(&worker)),
            Some(&1)
        );
    }

    #[tokio::test]
    async fn router_dump_includes_all_allocated_physical_tiers() {
        let indexer = make_test_indexer();
        for (event_id, tier, block) in [
            (1, StorageTier::Device, 11),
            (2, StorageTier::HostPinned, 12),
            (3, StorageTier::Disk, 13),
        ] {
            indexer
                .apply_event(store_event(7, 0, event_id, &[], &[block], tier))
                .await;
        }
        flush_indexer(&indexer).await;

        let events = indexer.dump_events().await.unwrap();
        for tier in [
            StorageTier::Device,
            StorageTier::HostPinned,
            StorageTier::Disk,
        ] {
            assert!(events.iter().any(|event| event.storage_tier == tier));
        }
    }

    #[tokio::test]
    async fn tiered_query_seeds_lower_tier_only_workers_without_affecting_device_scores() {
        let indexer = make_test_indexer();
        let device_worker = WorkerWithDpRank::new(10, 0);
        let host_only_worker = WorkerWithDpRank::new(20, 0);
        let disk_only_worker = WorkerWithDpRank::new(30, 0);

        indexer
            .apply_event(store_event(10, 0, 1, &[], &[21], StorageTier::Device))
            .await;
        indexer
            .apply_event(store_event(20, 0, 2, &[], &[21], StorageTier::HostPinned))
            .await;
        indexer
            .apply_event(store_event(30, 0, 3, &[], &[21], StorageTier::Disk))
            .await;
        flush_indexer(&indexer).await;

        let matches = indexer
            .find_matches_by_tier(vec![LocalBlockHash(21)])
            .await
            .unwrap();

        assert_eq!(
            matches.device.overlap_scores.scores.get(&device_worker),
            Some(&1)
        );
        assert!(
            !matches
                .device
                .overlap_scores
                .scores
                .contains_key(&host_only_worker)
        );
        assert!(
            !matches
                .device
                .overlap_scores
                .scores
                .contains_key(&disk_only_worker)
        );

        assert_eq!(
            matches
                .lower_tier
                .get(&StorageTier::HostPinned)
                .and_then(|tier| tier.hits.get(&host_only_worker)),
            Some(&1)
        );
        assert_eq!(
            matches
                .lower_tier
                .get(&StorageTier::Disk)
                .and_then(|tier| tier.hits.get(&disk_only_worker)),
            Some(&1)
        );
    }

    #[tokio::test]
    async fn tiered_query_only_seeds_matching_root_workers() {
        let indexer = make_test_indexer();
        let matching_host_worker = WorkerWithDpRank::new(20, 0);
        let nonmatching_host_worker = WorkerWithDpRank::new(21, 0);

        indexer
            .apply_event(store_event(20, 0, 1, &[], &[31], StorageTier::HostPinned))
            .await;
        indexer
            .apply_event(store_event(21, 0, 2, &[], &[32], StorageTier::HostPinned))
            .await;
        flush_indexer(&indexer).await;

        let matches = indexer
            .find_matches_by_tier(vec![LocalBlockHash(31)])
            .await
            .unwrap();

        assert_eq!(
            matches
                .lower_tier
                .get(&StorageTier::HostPinned)
                .and_then(|tier| tier.hits.get(&matching_host_worker)),
            Some(&1)
        );
        assert!(
            !matches
                .lower_tier
                .get(&StorageTier::HostPinned)
                .is_some_and(|tier| tier.hits.contains_key(&nonmatching_host_worker))
        );
    }

    #[tokio::test]
    async fn concurrent_tiered_query_chains_device_and_lower_tier_matches() {
        let indexer = make_test_concurrent_indexer();
        let worker = WorkerWithDpRank::new(7, 0);

        indexer
            .apply_event(store_event(7, 0, 1, &[], &[11, 12], StorageTier::Device))
            .await;
        indexer
            .apply_event(store_event(
                7,
                0,
                2,
                &[11, 12],
                &[13],
                StorageTier::HostPinned,
            ))
            .await;
        flush_indexer(&indexer).await;

        let matches = indexer
            .find_matches_by_tier(vec![
                LocalBlockHash(11),
                LocalBlockHash(12),
                LocalBlockHash(13),
            ])
            .await
            .unwrap();

        assert_eq!(matches.device.overlap_scores.scores.get(&worker), Some(&2));
        assert_eq!(
            matches
                .lower_tier
                .get(&StorageTier::HostPinned)
                .and_then(|tier| tier.hits.get(&worker)),
            Some(&1)
        );
    }

    #[tokio::test]
    async fn concurrent_records_hashed_routing_decision() {
        let indexer = make_test_concurrent_approx_indexer();
        assert!(indexer.records_routing_decisions());

        let worker = WorkerWithDpRank::new(7, 0);
        let tokens = vec![1, 2, 3, 4];
        let block_hashes = compute_block_hash_for_seq(&tokens, 4, BlockHashOptions::default());
        let sequence_hashes = compute_seq_hash_for_block(&block_hashes);

        indexer
            .record_hashed_routing_decision(worker, block_hashes.clone(), sequence_hashes)
            .await
            .unwrap();
        flush_indexer(&indexer).await;

        let matches = indexer.find_matches_by_tier(block_hashes).await.unwrap();
        assert_eq!(matches.device.overlap_scores.scores.get(&worker), Some(&1));
    }

    #[tokio::test]
    async fn concurrent_records_precomputed_routing_hashes() {
        let indexer = make_test_concurrent_approx_indexer();
        assert!(indexer.records_routing_decisions());

        let worker = WorkerWithDpRank::new(7, 0);
        let local_hashes = vec![LocalBlockHash(91), LocalBlockHash(92)];
        let sequence_hashes = compute_seq_hash_for_block(&local_hashes);
        indexer
            .record_routing_decision_hashes(
                worker,
                RoutingDecisionHashes {
                    local_hashes: local_hashes.clone(),
                    sequence_hashes,
                },
            )
            .await
            .unwrap();
        flush_indexer(&indexer).await;

        let matches = indexer.find_matches_by_tier(local_hashes).await.unwrap();
        assert_eq!(matches.device.overlap_scores.scores.get(&worker), Some(&2));
    }

    #[tokio::test]
    async fn event_driven_primary_without_side_skips_route_recording() {
        let indexer = make_test_indexer();
        assert!(!indexer.records_routing_decisions());

        let worker = WorkerWithDpRank::new(7, 0);
        let tokens = vec![1, 2, 3, 4];
        let block_hashes = compute_block_hash_for_seq(&tokens, 4, BlockHashOptions::default());
        let mut tokens_with_hashes = TokensWithHashes::new(tokens, 4);

        indexer
            .process_routing_decision_for_request(&mut tokens_with_hashes, worker)
            .await
            .unwrap();
        flush_indexer(&indexer).await;

        let matches = indexer.find_matches_by_tier(block_hashes).await.unwrap();
        assert!(
            !matches.device.overlap_scores.scores.contains_key(&worker),
            "event-driven primary without side overlay should rely on KV events, not route-time writes"
        );
    }

    #[tokio::test]
    async fn side_only_worker_scored_but_not_used_as_lower_tier_anchor() {
        // Build an Indexer::Concurrent with a real side indexer so
        // `record_hashed_routing_decision` populates only the side path.
        let primary = Arc::new(ThreadPoolIndexer::new(
            ConcurrentRadixTreeCompressed::new(),
            2,
            4,
        ));
        // PruneConfig is required to enable routing-decision recording on the
        // side indexer; without it the routing-decision path is a no-op.
        let side = Arc::new(ThreadPoolIndexer::new_with_pruning(
            ConcurrentRadixTreeCompressed::new(),
            1,
            4,
            PruneConfig {
                ttl: Duration::from_secs(60),
            },
        ));
        let side_for_flush = side.clone();
        let indexer = Indexer::Concurrent {
            primary,
            lower_tier: LowerTierIndexers::new(2, 4),
            approx: Some(super::SideIndexer::Concurrent(side)),
            primary_records_routing_decisions: false,
        };
        assert!(indexer.records_routing_decisions());

        let primary_worker = WorkerWithDpRank::new(10, 0);
        let side_only_worker = WorkerWithDpRank::new(20, 0);

        // Primary sees blocks [11, 12, 13] on Device for primary_worker;
        // extension block [14] on HostPinned for primary_worker.
        indexer
            .apply_event(store_event(
                10,
                0,
                1,
                &[],
                &[11, 12, 13],
                StorageTier::Device,
            ))
            .await;
        indexer
            .apply_event(store_event(
                10,
                0,
                2,
                &[11, 12, 13],
                &[14],
                StorageTier::HostPinned,
            ))
            .await;
        // Crucially, also give side_only_worker a HostPinned extension at
        // block 14 anchored on the same prefix [11, 12, 13]. If the lower
        // tier were seeded from the side-merged device score, the host walk
        // would find this and credit a hit; with the reorder it should not.
        indexer
            .apply_event(store_event(
                20,
                0,
                3,
                &[11, 12, 13],
                &[14],
                StorageTier::HostPinned,
            ))
            .await;

        // Side-only: route a decision so the side indexer learns
        // side_only_worker for the same device prefix. Primary never sees it.
        let block_hashes: Vec<LocalBlockHash> =
            [11, 12, 13].iter().copied().map(LocalBlockHash).collect();
        let sequence_hashes = compute_seq_hash_for_block(&block_hashes);
        indexer
            .record_hashed_routing_decision(side_only_worker, block_hashes.clone(), sequence_hashes)
            .await
            .unwrap();

        flush_indexer(&indexer).await;
        side_for_flush.flush().await;

        let matches = indexer
            .find_matches_by_tier(vec![
                LocalBlockHash(11),
                LocalBlockHash(12),
                LocalBlockHash(13),
                LocalBlockHash(14),
            ])
            .await
            .unwrap();

        // Merge worked: both workers carry device scores.
        assert_eq!(
            matches
                .device
                .overlap_scores
                .scores
                .get(&primary_worker)
                .copied(),
            Some(3)
        );
        assert_eq!(
            matches
                .device
                .overlap_scores
                .scores
                .get(&side_only_worker)
                .copied(),
            Some(3),
            "side-only worker should appear in merged device scores"
        );

        // Reorder enforced: lower-tier was seeded from primary only.
        // primary_worker still extends into HostPinned via its own device
        // anchor. side_only_worker's HostPinned extension exists in the
        // host tier, but because the side score wasn't used as a device
        // anchor, the host walk does not start for it and its host hit is
        // not credited.
        let host = matches
            .lower_tier
            .get(&StorageTier::HostPinned)
            .expect("host-pinned tier should have been allocated");
        assert_eq!(host.hits.get(&primary_worker).copied(), Some(1));
        assert_eq!(
            host.hits.get(&side_only_worker).copied().unwrap_or(0),
            0,
            "side-only worker's host extension must not be credited \
             when lower-tier seeding is primary-only"
        );
        assert!(
            !host.next_continuations.contains_key(&side_only_worker),
            "side-only worker must not appear in lower-tier continuations"
        );
    }

    #[tokio::test]
    async fn borrowed_tiered_lookup_matches_owned_with_lower_tier_and_side_overlay() {
        let primary = Arc::new(ThreadPoolIndexer::new(
            ConcurrentRadixTreeCompressed::new(),
            2,
            4,
        ));
        let side = Arc::new(ThreadPoolIndexer::new_with_pruning(
            ConcurrentRadixTreeCompressed::new(),
            1,
            4,
            PruneConfig {
                ttl: Duration::from_secs(60),
            },
        ));
        let side_for_flush = side.clone();
        let indexer = Indexer::Concurrent {
            primary,
            lower_tier: LowerTierIndexers::new(2, 4),
            approx: Some(super::SideIndexer::Concurrent(side)),
            primary_records_routing_decisions: false,
        };

        let primary_worker = WorkerWithDpRank::new(10, 0);
        let side_worker = WorkerWithDpRank::new(20, 0);
        indexer
            .apply_event(store_event(10, 0, 1, &[], &[11, 12], StorageTier::Device))
            .await;
        indexer
            .apply_event(store_event(
                10,
                0,
                2,
                &[11, 12],
                &[13],
                StorageTier::HostPinned,
            ))
            .await;

        let side_hashes = vec![LocalBlockHash(11), LocalBlockHash(12), LocalBlockHash(13)];
        indexer
            .record_routing_decision_hashes(
                side_worker,
                RoutingDecisionHashes {
                    local_hashes: side_hashes.clone(),
                    sequence_hashes: compute_seq_hash_for_block(&side_hashes),
                },
            )
            .await
            .unwrap();
        flush_indexer(&indexer).await;
        side_for_flush.flush().await;

        let query = vec![LocalBlockHash(11), LocalBlockHash(12), LocalBlockHash(13)];
        let borrowed = indexer.find_matches_by_tier_ref(&query).await.unwrap();
        let owned = indexer.find_matches_by_tier(query).await.unwrap();

        assert_eq!(
            borrowed.device.overlap_scores.scores,
            owned.device.overlap_scores.scores
        );
        assert_eq!(
            borrowed
                .lower_tier
                .get(&StorageTier::HostPinned)
                .map(|tier| &tier.hits),
            owned
                .lower_tier
                .get(&StorageTier::HostPinned)
                .map(|tier| &tier.hits)
        );
        assert_eq!(
            borrowed
                .device
                .overlap_scores
                .scores
                .get(&primary_worker)
                .copied(),
            Some(2)
        );
        assert_eq!(
            borrowed
                .device
                .overlap_scores
                .scores
                .get(&side_worker)
                .copied(),
            Some(3)
        );
    }

    #[tokio::test]
    async fn concurrent_tiered_query_seeds_lower_tier_only_workers_without_affecting_device_scores()
    {
        let indexer = make_test_concurrent_indexer();
        let device_worker = WorkerWithDpRank::new(10, 0);
        let host_only_worker = WorkerWithDpRank::new(20, 0);
        let disk_only_worker = WorkerWithDpRank::new(30, 0);

        indexer
            .apply_event(store_event(10, 0, 1, &[], &[21], StorageTier::Device))
            .await;
        indexer
            .apply_event(store_event(20, 0, 2, &[], &[21], StorageTier::HostPinned))
            .await;
        indexer
            .apply_event(store_event(30, 0, 3, &[], &[21], StorageTier::Disk))
            .await;
        flush_indexer(&indexer).await;

        let matches = indexer
            .find_matches_by_tier(vec![LocalBlockHash(21)])
            .await
            .unwrap();

        assert_eq!(
            matches.device.overlap_scores.scores.get(&device_worker),
            Some(&1)
        );
        assert!(
            !matches
                .device
                .overlap_scores
                .scores
                .contains_key(&host_only_worker)
        );
        assert!(
            !matches
                .device
                .overlap_scores
                .scores
                .contains_key(&disk_only_worker)
        );

        assert_eq!(
            matches
                .lower_tier
                .get(&StorageTier::HostPinned)
                .and_then(|tier| tier.hits.get(&host_only_worker)),
            Some(&1)
        );
        assert_eq!(
            matches
                .lower_tier
                .get(&StorageTier::Disk)
                .and_then(|tier| tier.hits.get(&disk_only_worker)),
            Some(&1)
        );
    }

    /// Regression test: when a worker has blocks in both device and lower-tier
    /// storage (e.g. same prefix stored on GPU and offloaded to host), the
    /// Concurrent indexer doesn't return last_matched_hashes. Without the fix,
    /// query_lower_tiers would re-query that worker from root in the lower tier,
    /// double-counting overlap blocks and producing cached_tokens > ISL.
    #[tokio::test]
    async fn concurrent_tiered_query_does_not_double_count_device_and_lower_tier_overlap() {
        let indexer = make_test_concurrent_indexer();
        let worker = WorkerWithDpRank::new(7, 0);

        // Device owns the prefix block; host-pinned extends it by one block.
        indexer
            .apply_event(store_event(7, 0, 1, &[], &[41], StorageTier::Device))
            .await;
        indexer
            .apply_event(store_event(7, 0, 2, &[41], &[42], StorageTier::HostPinned))
            .await;
        flush_indexer(&indexer).await;

        let matches = indexer
            .find_matches_by_tier(vec![LocalBlockHash(41), LocalBlockHash(42)])
            .await
            .unwrap();

        assert_eq!(matches.device.overlap_scores.scores.get(&worker), Some(&1));

        let host_hits = matches
            .lower_tier
            .get(&StorageTier::HostPinned)
            .and_then(|tier| tier.hits.get(&worker).copied())
            .unwrap_or(0);
        assert_eq!(
            host_hits, 1,
            "lower-tier should extend the device prefix without double-counting it"
        );
    }
}
