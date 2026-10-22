// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-session workload and cache-pressure telemetry for the KV router.
//!
//! Active only with `enable_session_prefix_index`. It is passive: it reads the session
//! index and the router's request lifecycle and writes metrics and logs, and nothing
//! reads it back to place or admit a request.
//!
//! - Counters move as events happen: thrash on arrival, session start and end.
//! - Gauges are set by a sweep task every `DYN_SESSION_STATS_SWEEP_SECS` (default 10),
//!   which also ends sessions idle past `DYN_SESSION_STATS_IDLE_TTL_SECS` (default 1800).
//!   `DYN_SESSION_STATS_ACTIVE_WINDOW_SECS` (default 300) sets how long a session counts
//!   as active.
//! - Per-session detail is one `session_summary` log event, at target
//!   `dynamo_session_summary`, when a session ends. Session IDs never reach a metric.

use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use dynamo_kv_router::protocols::{LocalBlockHash, WorkerWithDpRank};
use dynamo_kv_router::session_prefix_index::ResidencySnapshot;
use dynamo_kv_router::session_stats::{GAP_BUCKET_LABELS, SessionSummary, gap_bucket};
use dynamo_kv_router::{SessionPrefixIndexer, SessionStatsConfig, SessionStatsTracker};
use tokio_util::sync::CancellationToken;

use super::metrics::SessionMetrics;
use crate::discovery::RuntimeConfigWatch;

const SWEEP_SECS_ENV: &str = "DYN_SESSION_STATS_SWEEP_SECS";
const ACTIVE_WINDOW_SECS_ENV: &str = "DYN_SESSION_STATS_ACTIVE_WINDOW_SECS";
const IDLE_TTL_SECS_ENV: &str = "DYN_SESSION_STATS_IDLE_TTL_SECS";

const DEFAULT_SWEEP_SECS: u64 = 10;
const DEFAULT_ACTIVE_WINDOW_SECS: u64 = 300;
const DEFAULT_IDLE_TTL_SECS: u64 = 1_800;
/// How often the per-worker load rows are logged.
const WORKER_LOG_PERIOD_MS: u64 = 60_000;

fn env_secs(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

pub(crate) struct SessionStatsRuntime {
    tracker: SessionStatsTracker,
    index: Arc<SessionPrefixIndexer>,
    metrics: Arc<SessionMetrics>,
    /// Per-worker GPU KV capacity, when the router knows it.
    workers: Option<RuntimeConfigWatch>,
    block_size: u32,
    epoch: Instant,
    /// When the per-worker load log last ran, in ms since `epoch`; 0 until the first run.
    last_worker_log_ms: AtomicU64,
    /// IDs of recent requests without a session ID, so a retry of one is counted once.
    recent_uncovered: Mutex<(HashSet<String>, VecDeque<String>)>,
}

/// How many recent request IDs without a session the runtime remembers.
const RECENT_UNCOVERED: usize = 8192;

impl SessionStatsRuntime {
    pub(crate) fn new(
        index: Arc<SessionPrefixIndexer>,
        metrics: Arc<SessionMetrics>,
        block_size: u32,
        workers: Option<RuntimeConfigWatch>,
    ) -> Arc<Self> {
        let config = SessionStatsConfig {
            block_size,
            active_window_ms: env_secs(ACTIVE_WINDOW_SECS_ENV, DEFAULT_ACTIVE_WINDOW_SECS)
                .saturating_mul(1_000),
            idle_ttl_ms: env_secs(IDLE_TTL_SECS_ENV, DEFAULT_IDLE_TTL_SECS).saturating_mul(1_000),
            ..SessionStatsConfig::default()
        };
        Arc::new(Self {
            tracker: SessionStatsTracker::new(config),
            index,
            metrics,
            workers,
            block_size,
            epoch: Instant::now(),
            last_worker_log_ms: AtomicU64::new(0),
            recent_uncovered: Mutex::new((HashSet::new(), VecDeque::new())),
        })
    }

    #[cfg(test)]
    pub(crate) fn tracked_sessions(&self) -> usize {
        self.tracker.session_count()
    }

    fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }

    /// A routed request carried no session ID, so nothing tracks it. A retry or migration of a
    /// request already counted (the same request ID) is not counted again.
    pub(crate) fn on_uncovered(&self, isl_tokens: u64, request_id: Option<&str>) {
        if let Some(id) = request_id {
            let mut recent = self
                .recent_uncovered
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if recent.0.contains(id) {
                return;
            }
            recent.0.insert(id.to_string());
            recent.1.push_back(id.to_string());
            if recent.1.len() > RECENT_UNCOVERED
                && let Some(oldest) = recent.1.pop_front()
            {
                recent.0.remove(&oldest);
            }
        }
        self.record_coverage("uncovered", isl_tokens);
    }

    fn record_coverage(&self, coverage: &str, isl_tokens: u64) {
        self.metrics
            .coverage_requests_total
            .with_label_values(&[coverage])
            .inc();
        self.metrics
            .coverage_input_tokens_total
            .with_label_values(&[coverage])
            .inc_by(isl_tokens);
    }

    /// A request for `session_id` was routed to `worker`. `isl_tokens` is its prompt length.
    /// Thrash is what `worker` no longer holds of the session's prior prefix, not what the
    /// best worker holds, because the turn can only reuse its own worker's cache.
    #[cfg(test)]
    pub(crate) fn on_arrival(
        &self,
        session_id: &str,
        parent: Option<&str>,
        isl_tokens: u64,
        worker: WorkerWithDpRank,
    ) {
        self.on_arrival_with(session_id, parent, isl_tokens, worker, None);
    }

    /// As `on_arrival`, knowing the request ID so a retry of the same request replaces its
    /// failed attempt instead of counting as another turn.
    pub(crate) fn on_arrival_with(
        &self,
        session_id: &str,
        parent: Option<&str>,
        isl_tokens: u64,
        worker: WorkerWithDpRank,
        request_id: Option<&str>,
    ) {
        let resident = self
            .index
            .session_resident_blocks_on(session_id, Some(worker));
        let outcome = self.tracker.on_arrival_with(
            session_id,
            parent,
            isl_tokens,
            resident,
            self.now_ms(),
            request_id,
        );
        // A retry or migration of a request already counted is the same request.
        if !outcome.replay {
            self.record_coverage("covered", isl_tokens);
        }
        if outcome.returning {
            self.metrics
                .prior_prefix_tokens_total
                .inc_by(outcome.prior_tokens);
        }
        let thrash = &outcome.thrash;
        for (label, tokens) in [
            ("recomputed", thrash.recomputed),
            ("reloaded_host", thrash.reloaded_host),
            ("reloaded_disk", thrash.reloaded_disk),
        ] {
            if tokens > 0 {
                self.metrics
                    .thrash_tokens_total
                    .with_label_values(&[label])
                    .inc_by(tokens);
            }
        }
        if let (Some(gap_ms), Some(outcome)) = (outcome.gap_ms, outcome.outcome) {
            self.metrics
                .return_turns_total
                .with_label_values(&[GAP_BUCKET_LABELS[gap_bucket(gap_ms)], outcome.as_str()])
                .inc();
        }
    }

    /// Where the request was routed and the block hashes of its prompt.
    pub(crate) fn on_routed(
        &self,
        session_id: &str,
        worker: WorkerWithDpRank,
        hashes: &[LocalBlockHash],
    ) {
        let hashes: Vec<u64> = hashes.iter().map(|hash| hash.0).collect();
        self.tracker.on_routed(session_id, worker, &hashes);
    }

    /// Per-worker load. Metrics carry only fleet-level figures: how uneven the workers are
    /// (busiest over mean) and the cache left behind by sessions that moved. The per-worker
    /// rows go to a log, at most once a minute, because a worker identity is not a label.
    fn publish_workers(
        &self,
        home_demand: &std::collections::HashMap<WorkerWithDpRank, u64>,
        residency: &ResidencySnapshot,
    ) {
        let metrics = &self.metrics;
        let block = u64::from(self.block_size);
        let gauge = |value: u64| i64::try_from(value).unwrap_or(i64::MAX);
        metrics
            .tier_overlap_tokens
            .set(gauge(residency.device_host_overlap * block));

        // Every worker the router knows, so an idle worker pulls the mean down.
        let mut workers: Vec<(WorkerWithDpRank, Option<u64>)> = Vec::new();
        if let Some(watch) = &self.workers {
            for (&worker_id, config) in watch.borrow().iter() {
                let start = config.data_parallel_start_rank;
                let end = start.saturating_add(config.data_parallel_size);
                let capacity = config.total_kv_blocks.map(|blocks| blocks * block);
                workers.extend(
                    (start..end).map(|rank| (WorkerWithDpRank::new(worker_id, rank), capacity)),
                );
            }
        }
        if workers.is_empty() {
            let mut known: std::collections::HashSet<WorkerWithDpRank> =
                home_demand.keys().copied().collect();
            known.extend(residency.workers.keys().copied());
            workers = known.into_iter().map(|worker| (worker, None)).collect();
        }
        if workers.is_empty() {
            // Nothing to compare: leave no stale reading behind from the last time workers existed.
            metrics.worker_demand_imbalance_ratio.set(1.0);
            for tier in ["device", "host"] {
                metrics
                    .worker_resident_imbalance_ratio
                    .with_label_values(&[tier])
                    .set(1.0);
                metrics.left_behind_tokens.with_label_values(&[tier]).set(0);
            }
            return;
        }

        // Busiest over mean; 1.0 when nothing is held, so an empty fleet reads as even.
        let imbalance = |values: Vec<u64>| {
            let max = values.iter().copied().max().unwrap_or(0) as f64;
            let mean = values.iter().sum::<u64>() as f64 / values.len().max(1) as f64;
            if mean > 0.0 { max / mean } else { 1.0 }
        };
        metrics.worker_demand_imbalance_ratio.set(imbalance(
            workers
                .iter()
                .map(|(worker, _)| home_demand.get(worker).copied().unwrap_or(0))
                .collect(),
        ));
        type Pick = fn(&dynamo_kv_router::session_prefix_index::TierBlocks) -> u64;
        let tiers: [(&str, Pick); 2] = [("device", |b| b.device), ("host", |b| b.host)];
        for (tier, pick) in tiers {
            let held = workers
                .iter()
                .map(|(worker, _)| residency.workers.get(worker).map_or(0, pick))
                .collect();
            metrics
                .worker_resident_imbalance_ratio
                .with_label_values(&[tier])
                .set(imbalance(held));
            let left: u64 = residency.left_behind.values().map(pick).sum();
            metrics
                .left_behind_tokens
                .with_label_values(&[tier])
                .set(gauge(left * block));
        }

        let now_ms = self.now_ms();
        let last = self.last_worker_log_ms.load(Ordering::Relaxed);
        if last != 0 && now_ms.saturating_sub(last) < WORKER_LOG_PERIOD_MS {
            return;
        }
        self.last_worker_log_ms
            .store(now_ms.max(1), Ordering::Relaxed);
        for (worker, capacity) in &workers {
            let held = residency.workers.get(worker).copied().unwrap_or_default();
            let left = residency
                .left_behind
                .get(worker)
                .copied()
                .unwrap_or_default();
            tracing::info!(
                target: "dynamo_session_worker",
                event = "session_worker_load",
                worker_id = worker.worker_id,
                dp_rank = worker.dp_rank,
                demand_tokens = home_demand.get(worker).copied().unwrap_or(0),
                resident_device_tokens = held.device * block,
                resident_host_tokens = held.host * block,
                left_behind_device_tokens = left.device * block,
                left_behind_host_tokens = left.host * block,
                capacity_tokens = capacity.unwrap_or(0),
                "worker load"
            );
        }
    }

    /// A routed request never ran, or a failed attempt was handed to a retry. The retry
    /// arrives as its own request, so this attempt must not stay booked as a turn.
    pub(crate) fn on_abandoned(
        &self,
        session_id: &str,
        request: Option<(&str, u64)>,
        isl_tokens: u64,
    ) {
        self.tracker
            .on_abandoned_with(session_id, request, isl_tokens);
    }

    /// The attempt number the tracker gave the latest attempt of `request_id`, for the guard
    /// that reports its completion.
    pub(crate) fn attempt_of(&self, session_id: &str, request_id: &str) -> Option<u64> {
        self.tracker.attempt_of(session_id, request_id)
    }

    /// The request finished. `cached_tokens` is the prefix the router expected the worker
    /// to reuse; `session_final` ends the session.
    #[cfg(test)]
    pub(crate) fn on_complete(
        &self,
        session_id: &str,
        osl_tokens: u64,
        cached_tokens: u64,
        session_final: bool,
    ) {
        self.on_complete_with(session_id, None, osl_tokens, cached_tokens, session_final);
    }

    /// As `on_complete`, knowing the request ID. The output was generated either way, but a
    /// request replaced by a retry neither updates the session nor ends it.
    pub(crate) fn on_complete_with(
        &self,
        session_id: &str,
        request: Option<(&str, u64)>,
        osl_tokens: u64,
        cached_tokens: u64,
        session_final: bool,
    ) {
        let now_ms = self.now_ms();
        self.metrics.covered_output_tokens_total.inc_by(osl_tokens);
        let current =
            self.tracker
                .on_complete_with(session_id, request, osl_tokens, cached_tokens, now_ms);
        if current
            && session_final
            && let Some(summary) = self.tracker.end_session(session_id, now_ms)
        {
            self.record_summary(&summary);
        }
    }

    fn record_summary(&self, summary: &SessionSummary) {
        let metrics = &self.metrics;
        metrics.session_turns.observe(f64::from(summary.turns));
        metrics
            .session_peak_context_tokens
            .observe(summary.peak_demand_tokens as f64);
        metrics.session_cache_hit_ratio.observe(summary.hit_rate);
        metrics
            .session_duration_seconds
            .observe(summary.duration_ms as f64 / 1_000.0);
        if summary.turns > 1 {
            metrics
                .session_turn_gap_seconds
                .observe(summary.mean_turn_gap_ms / 1_000.0);
        }
        metrics
            .session_subagents
            .observe(f64::from(summary.children));
        tracing::info!(
            target: "dynamo_session_summary",
            event = "session_summary",
            session_id = %summary.session_id,
            parent_session_id = summary.parent_session_id.as_deref().unwrap_or(""),
            turns = summary.turns,
            input_tokens = summary.isl_tokens,
            output_tokens = summary.osl_tokens,
            cached_tokens = summary.cached_tokens,
            cache_hit_ratio = summary.hit_rate,
            duration_ms = summary.duration_ms,
            mean_turn_gap_ms = summary.mean_turn_gap_ms,
            peak_context_tokens = summary.peak_demand_tokens,
            subagents = summary.children,
            thrash_recomputed_tokens = summary.thrash.recomputed,
            thrash_reloaded_host_tokens = summary.thrash.reloaded_host,
            thrash_reloaded_disk_tokens = summary.thrash.reloaded_disk,
            "session ended"
        );
    }

    /// End idle sessions and refresh the gauges from the index.
    pub(crate) fn publish(&self) {
        let now_ms = self.now_ms();
        for summary in self.tracker.sweep(now_ms) {
            self.record_summary(&summary);
        }
        let active = self.tracker.active_sessions(now_ms);
        let homes = self.tracker.active_homes(now_ms);
        let residency = self.index.residency_snapshot_with_homes(
            |session| !active.contains(session),
            |session| homes.get(session).copied(),
        );
        let snapshot = self.tracker.snapshot(&residency, now_ms);
        let metrics = &self.metrics;
        let gauge = |value: u64| i64::try_from(value).unwrap_or(i64::MAX);
        metrics.shared_tokens.set(gauge(snapshot.shared_tokens));
        for (role, sessions, demand) in [
            (
                "primary",
                snapshot.active_primary_sessions,
                snapshot.demand_distinct_primary_tokens,
            ),
            (
                "subagent",
                snapshot.active_subagent_sessions,
                snapshot.demand_distinct_subagent_tokens,
            ),
        ] {
            metrics
                .active_by_role
                .with_label_values(&[role])
                .set(gauge(sessions));
            metrics
                .demand_by_role_tokens
                .with_label_values(&[role])
                .set(gauge(demand));
        }
        for (idle, tokens) in GAP_BUCKET_LABELS.iter().zip(snapshot.demand_idle_tokens) {
            metrics
                .demand_by_idle_tokens
                .with_label_values(&[idle])
                .set(gauge(tokens));
        }
        self.publish_workers(&snapshot.home_demand, &residency);
        for (tier, tokens) in [
            ("device", snapshot.distinct_device_tokens),
            ("host", snapshot.distinct_host_tokens),
            ("disk", snapshot.distinct_disk_tokens),
        ] {
            metrics
                .resident_tokens
                .with_label_values(&[tier])
                .set(gauge(tokens));
        }
        for (tier, tokens) in [
            ("device", snapshot.stranded_device_tokens),
            ("host", snapshot.stranded_host_tokens),
        ] {
            metrics
                .stranded_tokens
                .with_label_values(&[tier])
                .set(gauge(tokens));
        }
        for (tier, tokens) in [
            ("device", snapshot.replicated_device_tokens),
            ("host", snapshot.replicated_host_tokens),
        ] {
            metrics
                .replicated_tokens
                .with_label_values(&[tier])
                .set(gauge(tokens));
        }
    }

    pub(crate) fn spawn_sweeper(self: &Arc<Self>, cancellation: CancellationToken) {
        let runtime = Arc::clone(self);
        let period = Duration::from_secs(env_secs(SWEEP_SECS_ENV, DEFAULT_SWEEP_SECS));
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(period);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    _ = cancellation.cancelled() => break,
                    _ = interval.tick() => runtime.publish(),
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::protocols::{ExternalSequenceBlockHash, WorkerWithDpRank};

    fn runtime() -> (Arc<SessionStatsRuntime>, Arc<SessionPrefixIndexer>) {
        let registry = dynamo_runtime::MetricsRegistry::new();
        let index = Arc::new(SessionPrefixIndexer::new());
        let runtime =
            SessionStatsRuntime::new(index.clone(), SessionMetrics::for_test(&registry), 16, None);
        (runtime, index)
    }

    fn hashes(ids: std::ops::RangeInclusive<u64>) -> Vec<ExternalSequenceBlockHash> {
        ids.map(ExternalSequenceBlockHash).collect()
    }

    #[test]
    fn coverage_counts_requests_and_tokens_with_and_without_a_session() {
        let (runtime, _index) = runtime();
        runtime.on_arrival("s", None, 1_000, WorkerWithDpRank::new(1, 0));
        runtime.on_arrival("s", None, 1_200, WorkerWithDpRank::new(1, 0));
        runtime.on_uncovered(500, Some("r"));
        // The same request again (a retry) is not counted twice.
        runtime.on_uncovered(500, Some("r"));

        let metrics = &runtime.metrics;
        let get =
            |vec: &prometheus::IntCounterVec, label: &str| vec.with_label_values(&[label]).get();
        assert_eq!(get(&metrics.coverage_requests_total, "covered"), 2);
        assert_eq!(get(&metrics.coverage_requests_total, "uncovered"), 1);
        assert_eq!(get(&metrics.coverage_input_tokens_total, "covered"), 2_200);
        assert_eq!(get(&metrics.coverage_input_tokens_total, "uncovered"), 500);
    }

    #[test]
    fn a_returning_session_that_lost_its_prefix_counts_as_thrash_and_a_miss() {
        let (runtime, _index) = runtime();
        runtime.on_arrival("s", None, 1_600, WorkerWithDpRank::new(1, 0));
        runtime.on_complete("s", 0, 0, false);
        // Nothing of the 1_600 token prefix is resident any more.
        runtime.on_arrival("s", None, 1_700, WorkerWithDpRank::new(1, 0));

        let metrics = &runtime.metrics;
        assert_eq!(
            metrics
                .thrash_tokens_total
                .with_label_values(&["recomputed"])
                .get(),
            1_600
        );
        assert_eq!(
            metrics
                .return_turns_total
                .with_label_values(&["le_1m", "miss"])
                .get(),
            1
        );
    }

    #[test]
    fn a_returning_session_with_its_prefix_resident_is_a_hit_without_thrash() {
        let (runtime, index) = runtime();
        runtime.on_arrival("s", None, 160, WorkerWithDpRank::new(1, 0));
        runtime.on_complete("s", 0, 0, false);
        let worker = WorkerWithDpRank::new(1, 0);
        index
            .update_session_from_stored_blocks("s", worker, None, &hashes(1..=10))
            .unwrap();
        runtime.on_arrival("s", None, 200, WorkerWithDpRank::new(1, 0));

        let metrics = &runtime.metrics;
        assert_eq!(
            metrics
                .thrash_tokens_total
                .with_label_values(&["recomputed"])
                .get(),
            0
        );
        assert_eq!(
            metrics
                .return_turns_total
                .with_label_values(&["le_1m", "gpu"])
                .get(),
            1
        );
    }

    #[test]
    fn the_final_turn_ends_the_session_and_observes_its_summary_once() {
        let (runtime, _index) = runtime();
        runtime.on_arrival("root", None, 100, WorkerWithDpRank::new(1, 0));
        runtime.on_arrival("root/child", Some("root"), 100, WorkerWithDpRank::new(1, 0));
        runtime.on_complete("root/child", 10, 0, true);
        runtime.on_complete("root", 20, 0, true);
        runtime.on_complete("root", 20, 0, true);

        let metrics = &runtime.metrics;
        assert_eq!(metrics.session_turns.get_sample_count(), 2);
        assert_eq!(metrics.session_subagents.get_sample_sum(), 1.0);
    }

    #[test]
    fn publish_sets_the_gauges_from_the_index() {
        let (runtime, index) = runtime();
        runtime.on_arrival("s", None, 160, WorkerWithDpRank::new(1, 0));
        runtime.on_complete("s", 0, 0, false);
        let worker = WorkerWithDpRank::new(1, 0);
        index
            .update_session_from_stored_blocks("s", worker, None, &hashes(1..=10))
            .unwrap();
        runtime.publish();

        let metrics = &runtime.metrics;
        assert_eq!(
            metrics.active_by_role.with_label_values(&["primary"]).get(),
            1
        );
        // The session was last active just now, so its demand is in the first idle bucket.
        assert_eq!(
            metrics
                .demand_by_idle_tokens
                .with_label_values(&["le_1m"])
                .get(),
            160
        );
        assert_eq!(
            metrics.resident_tokens.with_label_values(&["device"]).get(),
            160
        );
        // Nothing was routed with hashes, so no block is referenced yet.
        assert_eq!(
            metrics
                .demand_by_role_tokens
                .with_label_values(&["primary"])
                .get(),
            0
        );
    }

    fn local(ids: impl IntoIterator<Item = u64>) -> Vec<LocalBlockHash> {
        ids.into_iter().map(LocalBlockHash).collect()
    }

    #[test]
    fn publish_reports_distinct_demand_by_role_and_the_reusable_prefix() {
        let (runtime, _index) = runtime();
        runtime.on_arrival("main", None, 8 * 16, WorkerWithDpRank::new(1, 0));
        runtime.on_routed("main", WorkerWithDpRank::new(1, 0), &local(1..=8));
        // The subagent starts from the first five blocks of its primary and adds two.
        runtime.on_arrival(
            "main/sub",
            Some("main"),
            7 * 16,
            WorkerWithDpRank::new(2, 0),
        );
        runtime.on_routed(
            "main/sub",
            WorkerWithDpRank::new(2, 0),
            &local((1..=5).chain([100, 101])),
        );
        runtime.publish();

        let metrics = &runtime.metrics;
        // The roles add up to the distinct demand: ten blocks, not the fifteen the sessions
        // present between them.
        assert_eq!(metrics.shared_tokens.get(), 5 * 16);
        assert_eq!(
            metrics.active_by_role.with_label_values(&["primary"]).get(),
            1
        );
        assert_eq!(
            metrics
                .active_by_role
                .with_label_values(&["subagent"])
                .get(),
            1
        );
        assert_eq!(
            metrics
                .demand_by_role_tokens
                .with_label_values(&["primary"])
                .get(),
            8 * 16
        );
        assert_eq!(
            metrics
                .demand_by_role_tokens
                .with_label_values(&["subagent"])
                .get(),
            2 * 16
        );
        // Without a capacity source the workers are the two that hold sessions: 128 and 112
        // tokens of distinct demand, so the busiest is 128 over a mean of 120.
        let ratio = metrics.worker_demand_imbalance_ratio.get();
        assert!((ratio - 128.0 / 120.0).abs() < 1e-9, "{ratio}");

        // A returning turn adds the prior turn's prefix to the reusable total.
        runtime.on_complete("main", 20, 0, false);
        runtime.on_arrival("main", None, 10 * 16, WorkerWithDpRank::new(1, 0));
        // The 20 tokens the turn generated are not part of the prefix.
        assert_eq!(metrics.prior_prefix_tokens_total.get(), 8 * 16);
    }
}
