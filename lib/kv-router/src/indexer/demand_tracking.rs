// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic-only cumulative distinct-block ("unconstrained demand",
//! `DASHBOARD_METRICS_ENGINEERING_PLAN.md` section 2b) tracking, reported as three
//! simultaneous load-average-style windows (1m/5m/15m) rather than one fixed window.
//!
//! Unlike [`super::age_tracking::AgeTracker`] (keyed by `(worker, dp_rank, block)` - one entry
//! per resident copy, entries removed on a matching `Remove`/`Cleared`), this tracker is keyed
//! by block hash alone and entries are **never removed** on `Remove`/`Cleared`. The question
//! this answers is "how much distinct content has this fleet been asked to serve," not "what is
//! resident right now" - that question is already answered by the primary tree's own
//! `distinct_blocks` stat (`redundancy_stats()`). A block that gets evicted and never
//! re-requested still counts toward demand until it ages out of the longest window below.
//!
//! **Three windows, one design decision worth stating plainly**: 1m/5m/15m answers a
//! different question than a single long (e.g. 24h) window would - "is my active working set
//! trending up right now" (an operational, load-average-style signal) rather than "what's my
//! true unconstrained footprint for capacity sizing" (which needs a much longer horizon, or the
//! offline trace-based approach in `DASHBOARD_METRICS_ENGINEERING_PLAN.md` section 2a). Do not
//! use these three gauges as a capacity-sizing number - they intentionally forget anything
//! older than 15 minutes.
//!
//! **Bounding, stated plainly** (the metrics plan's own section 2b/7 flagged a truly
//! never-pruned membership set as a real, unbounded memory-growth risk, not a free addition):
//! this tracker is bounded by *time*, not by an arbitrary entry cap - and a 15-minute longest
//! window is a materially smaller bound than the single 24h window this module originally
//! shipped with. [`DemandTracker::prune_and_count_windows`] is called from the same slow (30s)
//! background sampler tick that already reads `resident_age_percentiles`/`redundancy_stats` -
//! the same cost class, not a new one - and evicts any entry whose first-seen time is older
//! than the longest (15m) window while counting all three thresholds in that same single pass.
//! Demand older than 15 minutes is simply not counted: this is a rolling window, not a lifetime
//! total. That is a deliberate, documented tradeoff, not an oversight.
//!
//! Per-event cost is a single DashMap insert-if-absent on `Stored` only (`Removed`/`Cleared`
//! are no-ops here) - the same O(1)-per-event class already accepted for `AgeTracker`, and
//! unaffected by having three windows instead of one, since all three read the same underlying
//! `first_seen` timestamp - only the once-per-30s counting pass does three comparisons per
//! entry instead of one.
//!
//! Like `AgeTracker`, this never reads or writes either backend's own structures, so it is not
//! subject to this crate's `AGENTS.md` hot-path lock/traversal review - it is a side-table
//! observer of the same event stream, not a change to the primary tree or lower-tier DashMap.
//! **Not yet load-tested at a high Store-event rate**; treat the resulting gauges as
//! diagnostic-only until that measurement exists
//! (`DASHBOARD_METRICS_ENGINEERING_PLAN.md` section 7).

use std::time::{Duration, Instant};

use dashmap::DashMap;
use rustc_hash::FxBuildHasher;

use crate::protocols::*;

/// The three simultaneous windows this tracker reports, load-average style. The longest
/// (`LAST_15M`) also bounds the tracker's own memory - see module docs.
pub const LAST_1M: Duration = Duration::from_secs(60);
pub const LAST_5M: Duration = Duration::from_secs(5 * 60);
pub const LAST_15M: Duration = Duration::from_secs(15 * 60);

/// Cumulative distinct-block ("unconstrained demand") counts over each of the three
/// [`DemandTracker`] windows, as of one [`DemandTracker::prune_and_count_windows`] call.
#[derive(Debug, Clone, Copy, Default)]
pub struct DemandWindowCounts {
    pub last_1m: usize,
    pub last_5m: usize,
    pub last_15m: usize,
}

/// Side-table of "first observed stored" times, keyed by block content hash alone (not
/// per-worker-copy) - see module docs for why this differs from `AgeTracker`'s keying.
#[derive(Default)]
pub(super) struct DemandTracker {
    first_seen: DashMap<ExternalSequenceBlockHash, Instant, FxBuildHasher>,
}

impl DemandTracker {
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// `Stored` events only: insert-if-absent, one entry per distinct block hash regardless of
    /// which worker/dp_rank stored it. `Removed`/`Cleared` are intentionally ignored - demand
    /// for content already observed does not un-happen when a resident copy is evicted.
    pub(super) fn observe_event(&self, event: &RouterEvent) {
        if let KvCacheEventData::Stored(store) = &event.event.data {
            for block in &store.blocks {
                self.first_seen
                    .entry(block.block_hash)
                    .or_insert_with(Instant::now);
            }
        }
    }

    /// Evicts entries older than the longest window (`LAST_15M`) and counts, in that same
    /// single pass, how many remaining entries also fall within `LAST_1M`/`LAST_5M`. O(tracker
    /// size) - call only from a slow background sampler, same cost class as
    /// `AgeTracker::resident_age_percentiles`.
    pub(super) fn prune_and_count_windows(&self) -> DemandWindowCounts {
        let now = Instant::now();
        let mut counts = DemandWindowCounts::default();
        self.first_seen.retain(|_, first_seen| {
            let age = now.saturating_duration_since(*first_seen);
            let keep = age < LAST_15M;
            if keep {
                counts.last_15m += 1;
                if age < LAST_5M {
                    counts.last_5m += 1;
                    if age < LAST_1M {
                        counts.last_1m += 1;
                    }
                }
            }
            keep
        });
        counts
    }
}

#[cfg(all(test, feature = "metrics"))]
mod tests {
    use super::*;

    fn store_event(worker_id: WorkerId, dp_rank: DpRank, hash: u64) -> RouterEvent {
        RouterEvent::with_storage_tier(
            worker_id,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(hash),
                        tokens_hash: LocalBlockHash(hash),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank,
            },
            StorageTier::Device,
        )
    }

    fn remove_event(worker_id: WorkerId, dp_rank: DpRank, hash: u64) -> RouterEvent {
        RouterEvent::with_storage_tier(
            worker_id,
            KvCacheEvent {
                event_id: 2,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(hash)],
                }),
                dp_rank,
            },
            StorageTier::Device,
        )
    }

    #[test]
    fn distinct_store_across_workers_counts_once() {
        let tracker = DemandTracker::new();
        tracker.observe_event(&store_event(1, 0, 42));
        tracker.observe_event(&store_event(2, 0, 42));
        let counts = tracker.prune_and_count_windows();
        assert_eq!(counts.last_1m, 1);
        assert_eq!(counts.last_5m, 1);
        assert_eq!(counts.last_15m, 1);
    }

    #[test]
    fn distinct_hashes_count_separately() {
        let tracker = DemandTracker::new();
        tracker.observe_event(&store_event(1, 0, 42));
        tracker.observe_event(&store_event(1, 0, 43));
        let counts = tracker.prune_and_count_windows();
        assert_eq!(counts.last_1m, 2);
        assert_eq!(counts.last_5m, 2);
        assert_eq!(counts.last_15m, 2);
    }

    #[test]
    fn remove_does_not_uncount_demand() {
        let tracker = DemandTracker::new();
        tracker.observe_event(&store_event(1, 0, 42));
        tracker.observe_event(&remove_event(1, 0, 42));
        let counts = tracker.prune_and_count_windows();
        assert_eq!(counts.last_1m, 1);
        assert_eq!(counts.last_5m, 1);
        assert_eq!(counts.last_15m, 1);
    }

    #[test]
    fn entries_older_than_the_longest_window_are_pruned() {
        let tracker = DemandTracker::new();
        tracker.observe_event(&store_event(1, 0, 42));
        // Back-date the entry past the 15m window without a real sleep in the test.
        tracker
            .first_seen
            .alter(&ExternalSequenceBlockHash(42), |_, _| {
                Instant::now() - (LAST_15M + Duration::from_secs(1))
            });
        let counts = tracker.prune_and_count_windows();
        assert_eq!(counts.last_1m, 0);
        assert_eq!(counts.last_5m, 0);
        assert_eq!(counts.last_15m, 0);
        assert_eq!(tracker.first_seen.len(), 0);
    }

    #[test]
    fn an_entry_between_5m_and_15m_only_counts_toward_the_wider_windows() {
        let tracker = DemandTracker::new();
        tracker.observe_event(&store_event(1, 0, 42));
        tracker
            .first_seen
            .alter(&ExternalSequenceBlockHash(42), |_, _| {
                Instant::now() - (LAST_5M + Duration::from_secs(1))
            });
        let counts = tracker.prune_and_count_windows();
        assert_eq!(counts.last_1m, 0);
        assert_eq!(counts.last_5m, 0);
        assert_eq!(counts.last_15m, 1);
    }
}
