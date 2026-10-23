// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session telemetry: the working set against capacity, thrash, and the shape of
//! the sessions the router sees. Per-session and per-worker detail are logs, not
//! metrics; see `crate::kv_router::session_stats`.

use std::sync::{Arc, OnceLock};

use dynamo_kv_router::session_stats::GAP_BUCKET_LABELS;
use dynamo_runtime::component::Component;
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::metrics::prometheus_names::router_session as names;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use prometheus::{Gauge, GaugeVec, Histogram, IntCounter, IntCounterVec, IntGauge, IntGaugeVec};

use super::labels;

static SESSION_METRICS: OnceLock<Arc<SessionMetrics>> = OnceLock::new();

const TIER_LABEL: &str = "tier";
const OUTCOME_LABEL: &str = "outcome";
const GAP_LABEL: &str = "gap";
const IDLE_LABEL: &str = "idle";
const COVERAGE_LABEL: &str = "coverage";
const ROLE_LABEL: &str = "role";

pub(crate) const TIERS: [&str; 3] = ["device", "host", "disk"];
pub(crate) const STRANDED_TIERS: [&str; 2] = ["device", "host"];
pub(crate) const THRASH_OUTCOMES: [&str; 3] = ["recomputed", "reloaded_host", "reloaded_disk"];
pub(crate) const COVERAGE: [&str; 2] = ["covered", "uncovered"];
pub(crate) const ROLES: [&str; 2] = ["primary", "subagent"];
const RETURN_OUTCOMES: [&str; 4] = ["gpu", "host", "disk", "miss"];

/// Experimental. Names, labels and units may change in any release.
///
/// Series per router: 20 gauge series, 25 counter series (3 + 16 + 4 + 2), and six
/// histograms of up to 14 series each, about 130 in all. The `gap` and `idle` labels
/// share the 1, 5 and 15 minute buckets. No label carries a session, tenant or worker
/// identity.
#[cfg_attr(test, derive(Clone))]
pub struct SessionMetrics {
    /// Active sessions by role (primary session or subagent).
    pub active_by_role: IntGaugeVec,
    /// Distinct prompt tokens the active sessions will present next, by role. A subagent
    /// counts only the blocks its primary sessions do not already cover.
    pub demand_by_role_tokens: IntGaugeVec,
    /// Per-session demand of sessions that have not ended, by time since their last
    /// activity. Not de-duplicated; the buckets are disjoint, so the sum is meaningful.
    pub demand_by_idle_tokens: IntGaugeVec,
    /// Prompt tokens referenced by more than one active session.
    pub shared_tokens: IntGauge,
    /// Distinct tokens the index holds for sessions, per storage tier.
    pub resident_tokens: IntGaugeVec,
    /// Distinct tokens held only for inactive sessions, per storage tier.
    pub stranded_tokens: IntGaugeVec,
    /// Tokens of prefix held by more than one worker, copies beyond the first, per tier.
    pub replicated_tokens: IntGaugeVec,
    /// Tokens held in both the GPU and host tier of a worker, summed over workers: how
    /// much host memory is a copy of GPU memory.
    pub tier_overlap_tokens: IntGauge,
    /// Distinct demand homed on the busiest worker over the mean worker's (1.0 is even).
    pub worker_demand_imbalance_ratio: Gauge,
    /// Distinct session tokens on the busiest worker over the mean worker's, per tier.
    pub worker_resident_imbalance_ratio: GaugeVec,
    /// Tokens workers hold for active sessions whose last turn ran on another worker: the
    /// cache left behind when sessions move, summed over workers, per tier.
    pub left_behind_tokens: IntGaugeVec,
    /// Output tokens of requests that carried a session ID.
    pub covered_output_tokens_total: IntCounter,
    /// Prefix tokens returning sessions could reuse if nothing had been evicted.
    pub prior_prefix_tokens_total: IntCounter,
    /// Prefix tokens a returning session had to recompute or reload.
    pub thrash_tokens_total: IntCounterVec,
    /// Returning turns by the time since the session's last turn and where its prefix was.
    pub return_turns_total: IntCounterVec,
    /// Routed requests by whether they carried a session ID (covered) or not (uncovered).
    pub coverage_requests_total: IntCounterVec,
    /// Input tokens of routed requests, by the same split.
    pub coverage_input_tokens_total: IntCounterVec,
    pub session_turns: Histogram,
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
        let intgauge = |name: &str, help: &str| {
            metrics
                .create_intgauge(name, help, extra_labels)
                .unwrap_or_else(|error| panic!("failed to create {name}: {error}"))
        };
        let intgaugevec = |name: &str, help: &str, label: &str| {
            metrics
                .create_intgaugevec(name, help, &[label], extra_labels)
                .unwrap_or_else(|error| panic!("failed to create {name}: {error}"))
        };
        let counter = |name: &str, help: &str| {
            metrics
                .create_intcounter(name, help, extra_labels)
                .unwrap_or_else(|error| panic!("failed to create {name}: {error}"))
        };
        let countervec = |name: &str, help: &str, labels: &[&str]| {
            metrics
                .create_intcountervec(name, help, labels, extra_labels)
                .unwrap_or_else(|error| panic!("failed to create {name}: {error}"))
        };
        let histogram = |name: &str, help: &str, buckets: Vec<f64>| {
            metrics
                .create_histogram(name, help, extra_labels, Some(buckets))
                .unwrap_or_else(|error| panic!("failed to create {name}: {error}"))
        };

        let active_by_role = intgaugevec(
            names::SESSIONS_ACTIVE_BY_ROLE,
            "Active sessions by role (primary, subagent)",
            ROLE_LABEL,
        );
        let demand_by_role_tokens = intgaugevec(
            names::DEMAND_BY_ROLE_TOKENS,
            "Distinct prompt tokens active sessions will present next, by role (primary, subagent); a subagent counts only blocks its primary sessions do not cover",
            ROLE_LABEL,
        );
        let demand_by_idle_tokens = intgaugevec(
            names::DEMAND_BY_IDLE_TOKENS,
            "Per-session demand of sessions that have not ended, by time since their last activity (le_1m, le_5m, le_15m, gt_15m)",
            IDLE_LABEL,
        );
        let resident_tokens = intgaugevec(
            names::RESIDENT_TOKENS,
            "Distinct tokens the session index holds, by storage tier (device, host, disk)",
            TIER_LABEL,
        );
        let stranded_tokens = intgaugevec(
            names::STRANDED_TOKENS,
            "Distinct tokens held only for sessions that are inactive, by storage tier (device, host)",
            TIER_LABEL,
        );
        let replicated_tokens = intgaugevec(
            names::REPLICATED_TOKENS,
            "Tokens of session prefix held by more than one worker, copies beyond the first, by storage tier (device, host)",
            TIER_LABEL,
        );
        let left_behind_tokens = intgaugevec(
            names::LEFT_BEHIND_TOKENS,
            "Tokens workers hold for active sessions whose last turn ran on another worker, by storage tier (device, host)",
            TIER_LABEL,
        );
        let worker_demand_imbalance_ratio = metrics
            .create_gauge(
                names::WORKER_DEMAND_IMBALANCE_RATIO,
                "Distinct session demand homed on the busiest worker over the mean worker's; 1.0 is even",
                extra_labels,
            )
            .expect("failed to create router_session_worker_demand_imbalance_ratio");
        let worker_resident_imbalance_ratio = metrics
            .create_gaugevec(
                names::WORKER_RESIDENT_IMBALANCE_RATIO,
                "Distinct session tokens on the busiest worker over the mean worker's, by storage tier (device, host); 1.0 is even",
                &[TIER_LABEL],
                extra_labels,
            )
            .expect("failed to create router_session_worker_resident_imbalance_ratio");
        let thrash_tokens_total = countervec(
            names::THRASH_TOKENS_TOTAL,
            "Estimated prefix tokens a returning session had to recompute or reload (recomputed, reloaded_host, reloaded_disk)",
            &[OUTCOME_LABEL],
        );
        let return_turns_total = countervec(
            names::RETURN_TURNS_TOTAL,
            "Returning session turns by time since the previous turn (le_1m, le_5m, le_15m, gt_15m) and where the prior prefix was found (gpu, host, disk, miss)",
            &[GAP_LABEL, OUTCOME_LABEL],
        );
        let coverage_requests_total = countervec(
            names::COVERAGE_REQUESTS_TOTAL,
            "Routed requests by whether they carried a session ID (covered, uncovered)",
            &[COVERAGE_LABEL],
        );
        let coverage_input_tokens_total = countervec(
            names::COVERAGE_INPUT_TOKENS_TOTAL,
            "Input tokens of routed requests by whether they carried a session ID (covered, uncovered)",
            &[COVERAGE_LABEL],
        );

        // Bounded label values exist from the first scrape so rate() and absent() work.
        for role in ROLES {
            active_by_role.with_label_values(&[role]);
            demand_by_role_tokens.with_label_values(&[role]);
        }
        for idle in GAP_BUCKET_LABELS {
            demand_by_idle_tokens.with_label_values(&[idle]);
        }
        for tier in TIERS {
            resident_tokens.with_label_values(&[tier]);
        }
        for tier in STRANDED_TIERS {
            stranded_tokens.with_label_values(&[tier]);
            replicated_tokens.with_label_values(&[tier]);
            left_behind_tokens.with_label_values(&[tier]);
            worker_resident_imbalance_ratio.with_label_values(&[tier]);
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
            active_by_role,
            demand_by_role_tokens,
            demand_by_idle_tokens,
            shared_tokens: intgauge(
                names::SHARED_TOKENS,
                "Prompt tokens referenced by more than one active session",
            ),
            resident_tokens,
            stranded_tokens,
            replicated_tokens,
            tier_overlap_tokens: intgauge(
                names::TIER_OVERLAP_TOKENS,
                "Tokens held in both the GPU and host tier of a worker, summed over workers",
            ),
            worker_demand_imbalance_ratio,
            worker_resident_imbalance_ratio,
            left_behind_tokens,
            covered_output_tokens_total: counter(
                names::COVERED_OUTPUT_TOKENS_TOTAL,
                "Output tokens of requests that carried a session ID",
            ),
            prior_prefix_tokens_total: counter(
                names::PRIOR_PREFIX_TOKENS_TOTAL,
                "Prefix tokens returning sessions could reuse if nothing had been evicted: the prior turn's input, up to the new prompt length",
            ),
            thrash_tokens_total,
            return_turns_total,
            coverage_requests_total,
            coverage_input_tokens_total,
            session_turns: histogram(
                names::TURNS,
                "Turns per session, observed when the session ends",
                exponential(1.0, 2.0, 9),
            ),
            session_peak_context_tokens: histogram(
                names::PEAK_CONTEXT_TOKENS,
                "Largest context a session presented, in tokens, observed when it ends",
                exponential(1_000.0, 2.0, 11),
            ),
            session_cache_hit_ratio: histogram(
                names::CACHE_HIT_RATIO,
                "Share of a session's input tokens served from cache (0.0-1.0), as the worker reports it, or as the router predicted when the worker reports nothing, observed when it ends",
                prometheus::linear_buckets(0.0, 0.1, 11).expect("valid linear buckets"),
            ),
            session_duration_seconds: histogram(
                names::DURATION_SECONDS,
                "Time from a session's first request to its end",
                exponential(10.0, 2.0, 11),
            ),
            session_turn_gap_seconds: histogram(
                names::TURN_GAP_SECONDS,
                "Mean time between a session's turns, observed when it ends",
                exponential(0.5, 2.0, 12),
            ),
            session_subagents: histogram(
                names::SUBAGENTS,
                "Subagent sessions a session spawned, observed when it ends",
                vec![0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0],
            ),
        }
    }
}
