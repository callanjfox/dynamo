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

use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_kv_router::session_stats::{GAP_BUCKET_LABELS, SessionSummary, gap_bucket};
use dynamo_kv_router::{SessionPrefixIndexer, SessionStatsConfig, SessionStatsTracker};
use tokio_util::sync::CancellationToken;

use super::metrics::SessionMetrics;

const SWEEP_SECS_ENV: &str = "DYN_SESSION_STATS_SWEEP_SECS";
const ACTIVE_WINDOW_SECS_ENV: &str = "DYN_SESSION_STATS_ACTIVE_WINDOW_SECS";
const IDLE_TTL_SECS_ENV: &str = "DYN_SESSION_STATS_IDLE_TTL_SECS";

const DEFAULT_SWEEP_SECS: u64 = 10;
const DEFAULT_ACTIVE_WINDOW_SECS: u64 = 300;
const DEFAULT_IDLE_TTL_SECS: u64 = 1_800;

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
    epoch: Instant,
}

impl SessionStatsRuntime {
    pub(crate) fn new(
        index: Arc<SessionPrefixIndexer>,
        metrics: Arc<SessionMetrics>,
        block_size: u32,
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
            epoch: Instant::now(),
        })
    }

    #[cfg(test)]
    pub(crate) fn tracked_sessions(&self) -> usize {
        self.tracker.session_count()
    }

    fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }

    /// A request for `session_id` was routed. `isl_tokens` is its prompt length.
    pub(crate) fn on_arrival(&self, session_id: &str, parent: Option<&str>, isl_tokens: u64) {
        let resident = self.index.session_resident_blocks(session_id);
        let outcome =
            self.tracker
                .on_arrival(session_id, parent, isl_tokens, resident, self.now_ms());
        if !outcome.returning {
            self.metrics.sessions_started_total.inc();
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

    /// The request finished. `cached_tokens` is the prefix the router expected the worker
    /// to reuse; `session_final` ends the session.
    pub(crate) fn on_complete(
        &self,
        session_id: &str,
        osl_tokens: u64,
        cached_tokens: u64,
        session_final: bool,
    ) {
        let now_ms = self.now_ms();
        self.tracker
            .on_complete(session_id, osl_tokens, cached_tokens, now_ms);
        if session_final && let Some(summary) = self.tracker.end_session(session_id, now_ms) {
            self.record_summary(&summary);
        }
    }

    fn record_summary(&self, summary: &SessionSummary) {
        let metrics = &self.metrics;
        metrics.sessions_ended_total.inc();
        metrics.session_turns.observe(f64::from(summary.turns));
        metrics
            .session_input_tokens
            .observe(summary.isl_tokens as f64);
        metrics
            .session_output_tokens
            .observe(summary.osl_tokens as f64);
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
        let residency = self
            .index
            .residency_snapshot_with(|session| !active.contains(session));
        let snapshot = self.tracker.snapshot(&residency, now_ms);
        let metrics = &self.metrics;
        let gauge = |value: u64| i64::try_from(value).unwrap_or(i64::MAX);
        metrics.sessions_active.set(gauge(snapshot.active_sessions));
        metrics.demand_tokens.set(gauge(snapshot.demand_tokens));
        metrics.sessions_split.set(gauge(snapshot.split_sessions));
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
            SessionStatsRuntime::new(index.clone(), SessionMetrics::for_test(&registry), 16);
        (runtime, index)
    }

    fn hashes(ids: std::ops::RangeInclusive<u64>) -> Vec<ExternalSequenceBlockHash> {
        ids.map(ExternalSequenceBlockHash).collect()
    }

    #[test]
    fn a_returning_session_that_lost_its_prefix_counts_as_thrash_and_a_miss() {
        let (runtime, _index) = runtime();
        runtime.on_arrival("s", None, 1_600);
        runtime.on_complete("s", 0, 0, false);
        // Nothing of the 1_600 token prefix is resident any more.
        runtime.on_arrival("s", None, 1_700);

        let metrics = &runtime.metrics;
        assert_eq!(metrics.sessions_started_total.get(), 1);
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
                .with_label_values(&["le_1s", "miss"])
                .get(),
            1
        );
    }

    #[test]
    fn a_returning_session_with_its_prefix_resident_is_a_hit_without_thrash() {
        let (runtime, index) = runtime();
        runtime.on_arrival("s", None, 160);
        runtime.on_complete("s", 0, 0, false);
        let worker = WorkerWithDpRank::new(1, 0);
        index
            .update_session_from_stored_blocks("s", worker, None, &hashes(1..=10))
            .unwrap();
        runtime.on_arrival("s", None, 200);

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
                .with_label_values(&["le_1s", "gpu"])
                .get(),
            1
        );
    }

    #[test]
    fn the_final_turn_ends_the_session_and_observes_its_summary_once() {
        let (runtime, _index) = runtime();
        runtime.on_arrival("root", None, 100);
        runtime.on_arrival("root/child", Some("root"), 100);
        runtime.on_complete("root/child", 10, 0, true);
        runtime.on_complete("root", 20, 0, true);
        runtime.on_complete("root", 20, 0, true);

        let metrics = &runtime.metrics;
        assert_eq!(metrics.sessions_ended_total.get(), 2);
        assert_eq!(metrics.session_turns.get_sample_count(), 2);
        assert_eq!(metrics.session_subagents.get_sample_sum(), 1.0);
    }

    #[test]
    fn publish_sets_the_gauges_from_the_index() {
        let (runtime, index) = runtime();
        runtime.on_arrival("s", None, 160);
        runtime.on_complete("s", 0, 0, false);
        let worker = WorkerWithDpRank::new(1, 0);
        index
            .update_session_from_stored_blocks("s", worker, None, &hashes(1..=10))
            .unwrap();
        runtime.publish();

        let metrics = &runtime.metrics;
        assert_eq!(metrics.sessions_active.get(), 1);
        assert_eq!(metrics.demand_tokens.get(), 160);
        assert_eq!(
            metrics.resident_tokens.with_label_values(&["device"]).get(),
            160
        );
        assert_eq!(metrics.sessions_split.get(), 0);
    }
}
