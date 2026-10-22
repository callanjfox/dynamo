// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session telemetry: the working set against capacity, thrash, and the shape of
//! the sessions the router sees. Per-session detail is a log, not a metric; see
//! `crate::kv_router::session_stats`.

use std::sync::{Arc, OnceLock};

use dynamo_kv_router::session_stats::GAP_BUCKET_LABELS;
use dynamo_runtime::component::Component;
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use prometheus::{GaugeVec, Histogram, IntCounter, IntCounterVec, IntGauge, IntGaugeVec};

use super::{labels, router_metric};

static SESSION_METRICS: OnceLock<Arc<SessionMetrics>> = OnceLock::new();

const TIER_LABEL: &str = "tier";
const OUTCOME_LABEL: &str = "outcome";
const GAP_LABEL: &str = "gap";
const COVERAGE_LABEL: &str = "coverage";
const ROLE_LABEL: &str = "role";
const WINDOW_LABEL: &str = "window";
const STAT_LABEL: &str = "stat";

pub(crate) const TIERS: [&str; 3] = ["device", "host", "disk"];
pub(crate) const STRANDED_TIERS: [&str; 2] = ["device", "host"];
pub(crate) const THRASH_OUTCOMES: [&str; 3] = ["recomputed", "reloaded_host", "reloaded_disk"];
pub(crate) const COVERAGE: [&str; 2] = ["covered", "uncovered"];
pub(crate) const ROLES: [&str; 2] = ["primary", "subagent"];
/// Follows `dynamo_kv_router::session_stats::DEMAND_WINDOWS_MS`.
pub(crate) const DEMAND_WINDOWS: [&str; 3] = ["1m", "5m", "30m"];
pub(crate) const WORKER_STATS: [&str; 3] = ["max", "min", "mean"];
const RETURN_OUTCOMES: [&str; 4] = ["gpu", "host", "disk", "miss"];

/// Experimental. Names, labels and units may change in any release.
///
/// Series per router: 38 gauges, counters 2 + 3 + 40 + 4 + 2, and eight histograms of up
/// to 14 series each, about 215 in all. No label carries a session, tenant or worker
/// identity.
#[cfg_attr(test, derive(Clone))]
pub struct SessionMetrics {
    /// Sessions with activity inside the active window that have not ended.
    pub sessions_active: IntGauge,
    /// Sum over active sessions of the prefix each will present next. A shared prefix
    /// is counted once per session, so this overstates distinct demand.
    pub demand_tokens: IntGauge,
    /// Demand counting each distinct prompt block once across active sessions.
    pub demand_distinct_tokens: IntGauge,
    /// Distinct tokens the index holds for sessions, per storage tier.
    pub resident_tokens: IntGaugeVec,
    /// Distinct tokens held only for inactive sessions, per storage tier.
    pub stranded_tokens: IntGaugeVec,
    /// Tokens of prefix held by more than one worker, copies beyond the first, per tier.
    pub replicated_tokens: IntGaugeVec,
    /// Prompt tokens referenced by more than one active session.
    pub shared_tokens: IntGauge,
    /// Active sessions, and their distinct demand, by role (primary session or subagent).
    /// A subagent counts only the blocks its primary sessions do not already cover.
    pub active_by_role: IntGaugeVec,
    pub demand_by_role_tokens: IntGaugeVec,
    /// Demand counting only sessions active within a short window.
    pub demand_window_tokens: IntGaugeVec,
    /// Distinct tokens the busiest worker holds, and the mean over workers holding any,
    /// per tier. Compare with one worker's capacity to see load that is uneven.
    pub worker_resident_tokens: IntGaugeVec,
    /// Demand homed on one worker (the worker that served the session's last turn): the
    /// busiest (max) and the mean over all workers.
    pub worker_demand_tokens: IntGaugeVec,
    /// The same demand as a share of that worker's GPU KV capacity. Above 1.0 means the
    /// worker is asked to hold more than its GPU can.
    pub worker_demand_ratio: GaugeVec,
    /// Distinct tokens each worker holds for active sessions whose last turn ran elsewhere:
    /// the cache left behind when sessions move.
    pub worker_left_behind_tokens: IntGaugeVec,
    /// Tokens held in both the GPU and host tier of a worker, summed over workers: how
    /// much host memory is a copy of GPU memory.
    pub tier_overlap_tokens: IntGauge,
    /// Output tokens of requests that carried a session ID.
    pub covered_output_tokens_total: IntCounter,
    /// Prefix tokens returning sessions could reuse if nothing had been evicted.
    pub prior_prefix_tokens_total: IntCounter,
    pub sessions_started_total: IntCounter,
    pub sessions_ended_total: IntCounter,
    /// Prefix tokens a returning session had to recompute or reload.
    pub thrash_tokens_total: IntCounterVec,
    /// Returning turns by the time since the session's last turn and where its prefix was.
    pub return_turns_total: IntCounterVec,
    /// Routed requests by whether they carried a session ID (covered) or not (uncovered).
    pub coverage_requests_total: IntCounterVec,
    /// Input tokens of routed requests, by the same split.
    pub coverage_input_tokens_total: IntCounterVec,
    pub session_turns: Histogram,
    pub session_input_tokens: Histogram,
    pub session_output_tokens: Histogram,
    pub session_peak_context_tokens: Histogram,
    pub session_cache_hit_ratio: Histogram,
    pub session_duration_seconds: Histogram,
    pub session_turn_gap_seconds: Histogram,
    pub session_subagents: Histogram,
}

fn exponential(start: f64, factor: f64, count: usize) -> Vec<f64> {
    prometheus::exponential_buckets(start, factor, count).expect("valid exponential buckets")
}

impl SessionMetrics {
    /// Create from a Component, memoized per process like the other router metrics.
    /// Gauges are last-router-wins when one process hosts several routers, for
    /// example a prefill and a decode pool; counters and histograms add up.
    pub fn from_component(component: &Component) -> Arc<Self> {
        SESSION_METRICS
            .get_or_init(|| {
                let router_id = component.drt().discovery().instance_id().to_string();
                let extra_labels: &[(&str, &str)] = &[(labels::ROUTER_ID, &router_id)];
                Arc::new(Self::build(component, extra_labels))
            })
            .clone()
    }

    pub(super) fn build<H: MetricsHierarchy>(hierarchy: &H, extra_labels: &[(&str, &str)]) -> Self {
        let metrics = hierarchy.metrics();
        let gauge = |name: &str, help: &str| {
            metrics
                .create_intgauge(&router_metric(name), help, extra_labels)
                .unwrap_or_else(|error| panic!("failed to create router_{name}: {error}"))
        };
        let counter = |name: &str, help: &str| {
            metrics
                .create_intcounter(&router_metric(name), help, extra_labels)
                .unwrap_or_else(|error| panic!("failed to create router_{name}: {error}"))
        };
        let histogram = |name: &str, help: &str, buckets: Vec<f64>| {
            metrics
                .create_histogram(&router_metric(name), help, extra_labels, Some(buckets))
                .unwrap_or_else(|error| panic!("failed to create router_{name}: {error}"))
        };

        let resident_tokens = metrics
            .create_intgaugevec(
                &router_metric("session_resident_tokens"),
                "Distinct tokens the session index holds, by storage tier (device, host, disk)",
                &[TIER_LABEL],
                extra_labels,
            )
            .expect("failed to create router_session_resident_tokens");
        let stranded_tokens = metrics
            .create_intgaugevec(
                &router_metric("session_stranded_tokens"),
                "Distinct tokens held only for sessions that are inactive, by storage tier (device, host)",
                &[TIER_LABEL],
                extra_labels,
            )
            .expect("failed to create router_session_stranded_tokens");
        let replicated_tokens = metrics
            .create_intgaugevec(
                &router_metric("session_replicated_tokens"),
                "Tokens of session prefix held by more than one worker, copies beyond the first, by storage tier (device, host)",
                &[TIER_LABEL],
                extra_labels,
            )
            .expect("failed to create router_session_replicated_tokens");
        let thrash_tokens_total = metrics
            .create_intcountervec(
                &router_metric("session_thrash_tokens_total"),
                "Estimated prefix tokens a returning session had to recompute or reload (recomputed, reloaded_host, reloaded_disk)",
                &[OUTCOME_LABEL],
                extra_labels,
            )
            .expect("failed to create router_session_thrash_tokens_total");
        let return_turns_total = metrics
            .create_intcountervec(
                &router_metric("session_return_turns_total"),
                "Returning session turns by time since the previous turn and where the prior prefix was found (gpu, host, disk, miss)",
                &[GAP_LABEL, OUTCOME_LABEL],
                extra_labels,
            )
            .expect("failed to create router_session_return_turns_total");

        let coverage_requests_total = metrics
            .create_intcountervec(
                &router_metric("session_coverage_requests_total"),
                "Routed requests by whether they carried a session ID (covered, uncovered)",
                &[COVERAGE_LABEL],
                extra_labels,
            )
            .expect("failed to create router_session_coverage_requests_total");
        let coverage_input_tokens_total = metrics
            .create_intcountervec(
                &router_metric("session_coverage_input_tokens_total"),
                "Input tokens of routed requests by whether they carried a session ID (covered, uncovered)",
                &[COVERAGE_LABEL],
                extra_labels,
            )
            .expect("failed to create router_session_coverage_input_tokens_total");

        let active_by_role = metrics
            .create_intgaugevec(
                &router_metric("sessions_active_by_role"),
                "Active sessions by role (primary, subagent)",
                &[ROLE_LABEL],
                extra_labels,
            )
            .expect("failed to create router_sessions_active_by_role");
        let demand_by_role_tokens = metrics
            .create_intgaugevec(
                &router_metric("session_demand_by_role_tokens"),
                "Demand of active sessions by role (primary, subagent)",
                &[ROLE_LABEL],
                extra_labels,
            )
            .expect("failed to create router_session_demand_by_role_tokens");
        let demand_window_tokens = metrics
            .create_intgaugevec(
                &router_metric("session_demand_window_tokens"),
                "Demand counting only sessions active within the window (30s, 1m, 5m)",
                &[WINDOW_LABEL],
                extra_labels,
            )
            .expect("failed to create router_session_demand_window_tokens");
        let worker_resident_tokens = metrics
            .create_intgaugevec(
                &router_metric("session_worker_resident_tokens"),
                "Distinct session tokens held on one worker, the busiest (max) and the mean over workers holding any, by tier (device, host)",
                &[STAT_LABEL, TIER_LABEL],
                extra_labels,
            )
            .expect("failed to create router_session_worker_resident_tokens");

        let worker_demand_tokens = metrics
            .create_intgaugevec(
                &router_metric("session_worker_demand_tokens"),
                "Distinct demand homed on one worker: the busiest (max), the least busy (min) and the mean over all workers",
                &[STAT_LABEL],
                extra_labels,
            )
            .expect("failed to create router_session_worker_demand_tokens");
        let worker_demand_ratio = metrics
            .create_gaugevec(
                &router_metric("session_worker_demand_ratio"),
                "Distinct homed demand over the worker's GPU KV capacity: the most loaded worker (max), the least (min) and the mean",
                &[STAT_LABEL],
                extra_labels,
            )
            .expect("failed to create router_session_worker_demand_ratio");

        let worker_left_behind_tokens = metrics
            .create_intgaugevec(
                &router_metric("session_worker_left_behind_tokens"),
                "Distinct tokens one worker holds for active sessions homed elsewhere: busiest (max), least (min) and mean, by tier (device, host)",
                &[STAT_LABEL, TIER_LABEL],
                extra_labels,
            )
            .expect("failed to create router_session_worker_left_behind_tokens");

        // Bounded label values exist from the first scrape so rate() and absent() work.
        for stat in WORKER_STATS {
            worker_demand_tokens.with_label_values(&[stat]);
            worker_demand_ratio.with_label_values(&[stat]);
        }
        for role in ROLES {
            active_by_role.with_label_values(&[role]);
            demand_by_role_tokens.with_label_values(&[role]);
        }
        for window in DEMAND_WINDOWS {
            demand_window_tokens.with_label_values(&[window]);
        }
        for tier in STRANDED_TIERS {
            for stat in WORKER_STATS {
                worker_left_behind_tokens.with_label_values(&[stat, tier]);
                worker_resident_tokens.with_label_values(&[stat, tier]);
            }
        }
        for tier in TIERS {
            resident_tokens.with_label_values(&[tier]);
        }
        for tier in STRANDED_TIERS {
            stranded_tokens.with_label_values(&[tier]);
        }
        for tier in STRANDED_TIERS {
            replicated_tokens.with_label_values(&[tier]);
        }
        for coverage in COVERAGE {
            coverage_requests_total.with_label_values(&[coverage]);
            coverage_input_tokens_total.with_label_values(&[coverage]);
        }
        for outcome in THRASH_OUTCOMES {
            thrash_tokens_total.with_label_values(&[outcome]);
        }
        for gap in GAP_BUCKET_LABELS {
            for outcome in RETURN_OUTCOMES {
                return_turns_total.with_label_values(&[gap, outcome]);
            }
        }

        Self {
            sessions_active: gauge(
                "sessions_active",
                "Sessions with activity inside the active window that have not ended",
            ),
            demand_tokens: gauge(
                "session_demand_tokens",
                "Sum over active sessions of the prefix tokens each will present next, counting a shared prefix once per session",
            ),
            resident_tokens,
            stranded_tokens,
            replicated_tokens,
            shared_tokens: gauge(
                "session_shared_tokens",
                "Prompt tokens referenced by more than one active session",
            ),
            demand_distinct_tokens: gauge(
                "session_demand_distinct_tokens",
                "Demand counting each distinct prompt block once across active sessions, so a shared prefix counts once",
            ),
            worker_demand_tokens,
            worker_demand_ratio,
            worker_left_behind_tokens,
            tier_overlap_tokens: gauge(
                "session_tier_overlap_tokens",
                "Tokens held in both the GPU and host tier of a worker, summed over workers",
            ),
            covered_output_tokens_total: counter(
                "session_covered_output_tokens_total",
                "Output tokens of requests that carried a session ID",
            ),
            active_by_role,
            demand_by_role_tokens,
            demand_window_tokens,
            worker_resident_tokens,
            prior_prefix_tokens_total: counter(
                "session_prior_prefix_tokens_total",
                "Prefix tokens returning sessions could reuse if nothing had been evicted: the prior turn's input and output, up to the new prompt length",
            ),
            coverage_requests_total,
            coverage_input_tokens_total,
            sessions_started_total: counter("sessions_started_total", "Sessions first seen"),
            sessions_ended_total: counter(
                "sessions_ended_total",
                "Sessions ended by a final-turn marker or by idling past the TTL",
            ),
            thrash_tokens_total,
            return_turns_total,
            session_turns: histogram(
                "session_turns",
                "Turns per session, observed when the session ends",
                exponential(1.0, 2.0, 9),
            ),
            session_input_tokens: histogram(
                "session_input_tokens",
                "Input tokens summed over a session's turns, observed when it ends",
                exponential(1_000.0, 4.0, 12),
            ),
            session_output_tokens: histogram(
                "session_output_tokens",
                "Output tokens summed over a session's turns, observed when it ends",
                exponential(100.0, 4.0, 12),
            ),
            session_peak_context_tokens: histogram(
                "session_peak_context_tokens",
                "Largest context a session presented, in tokens, observed when it ends",
                exponential(1_000.0, 2.0, 11),
            ),
            session_cache_hit_ratio: histogram(
                "session_cache_hit_ratio",
                "Share of a session's input tokens served from cache (0.0-1.0), observed when it ends",
                prometheus::linear_buckets(0.0, 0.1, 11).expect("valid linear buckets"),
            ),
            session_duration_seconds: histogram(
                "session_duration_seconds",
                "Time from a session's first request to its end",
                exponential(10.0, 2.0, 11),
            ),
            session_turn_gap_seconds: histogram(
                "session_turn_gap_seconds",
                "Mean time between a session's turns, observed when it ends",
                exponential(0.5, 2.0, 12),
            ),
            session_subagents: histogram(
                "session_subagents",
                "Subagent sessions a session spawned, observed when it ends",
                vec![0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0],
            ),
        }
    }
}
