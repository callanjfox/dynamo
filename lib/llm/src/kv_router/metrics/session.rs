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
use prometheus::{Histogram, IntCounter, IntCounterVec, IntGauge, IntGaugeVec};

use super::{labels, router_metric};

static SESSION_METRICS: OnceLock<Arc<SessionMetrics>> = OnceLock::new();

const TIER_LABEL: &str = "tier";
const OUTCOME_LABEL: &str = "outcome";
const GAP_LABEL: &str = "gap";

pub(crate) const TIERS: [&str; 3] = ["device", "host", "disk"];
pub(crate) const STRANDED_TIERS: [&str; 2] = ["device", "host"];
pub(crate) const THRASH_OUTCOMES: [&str; 3] = ["recomputed", "reloaded_host", "reloaded_disk"];
const RETURN_OUTCOMES: [&str; 4] = ["gpu", "host", "disk", "miss"];

/// Experimental. Names, labels and units may change in any release.
///
/// Series per router: gauges 1 + 1 + 3 + 2 + 1, counters 2 + 3 + 40, and eight
/// histograms of up to 14 series each, about 170 in all. No label carries a
/// session, tenant or worker identity.
#[cfg_attr(test, derive(Clone))]
pub struct SessionMetrics {
    /// Sessions with activity inside the active window that have not ended.
    pub sessions_active: IntGauge,
    /// Sum over active sessions of the prefix each will present next. A shared prefix
    /// is counted once per session, so this overstates distinct demand.
    pub demand_tokens: IntGauge,
    /// Distinct tokens the index holds for sessions, per storage tier.
    pub resident_tokens: IntGaugeVec,
    /// Distinct tokens held only for inactive sessions, per storage tier.
    pub stranded_tokens: IntGaugeVec,
    pub replicated_tokens: IntGaugeVec,
    /// Active sessions whose GPU prefix sits on more than one worker.
    pub sessions_split: IntGauge,
    pub sessions_started_total: IntCounter,
    pub sessions_ended_total: IntCounter,
    /// Prefix tokens a returning session had to recompute or reload.
    pub thrash_tokens_total: IntCounterVec,
    /// Returning turns by the time since the session's last turn and where its prefix was.
    pub return_turns_total: IntCounterVec,
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

        // Bounded label values exist from the first scrape so rate() and absent() work.
        for tier in TIERS {
            resident_tokens.with_label_values(&[tier]);
        }
        for tier in STRANDED_TIERS {
            stranded_tokens.with_label_values(&[tier]);
        }
        for tier in STRANDED_TIERS {
            replicated_tokens.with_label_values(&[tier]);
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
                "Sum over active sessions of the prefix tokens each will present next",
            ),
            resident_tokens,
            stranded_tokens,
            replicated_tokens,
            sessions_split: gauge(
                "sessions_split",
                "Active sessions whose GPU-resident prefix is on more than one worker",
            ),
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
