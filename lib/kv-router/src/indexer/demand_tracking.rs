// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic-only cumulative distinct-block ("unconstrained demand",
//! `DASHBOARD_METRICS_ENGINEERING_PLAN.md` section 2b) tracking.
//!
//! Unlike [`super::age_tracking::AgeTracker`] (keyed by `(worker, dp_rank, block)` - one entry
//! per resident copy, entries removed on a matching `Remove`/`Cleared`), this tracker is keyed
//! by block hash alone and entries are **never removed** on `Remove`/`Cleared`. The question
//! this answers is "how much distinct content has this fleet been asked to serve," not "what is
//! resident right now" - that question is already answered by the primary tree's own
//! `distinct_blocks` stat (`redundancy_stats()`). A block that gets evicted and never
//! re-requested still counts toward demand until it ages out of `window` below.
//!
//! **Bounding, stated plainly** (the metrics plan's own section 2b/7 flagged a truly
//! never-pruned membership set as a real, unbounded memory-growth risk, not a free addition):
//! this tracker is bounded by *time*, not by an arbitrary entry cap. [`DemandTracker::prune_and_count`]
//! is called from the same slow (30s) background sampler tick that already reads
//! `resident_age_percentiles`/`redundancy_stats` - the same cost class, not a new one - and
//! evicts any entry whose first-seen time is older than `window`. Demand older than the window
//! is simply not counted: this is a rolling "distinct content observed over the last N seconds"
//! number, not a lifetime total. That is a deliberate, documented tradeoff, not an oversight.
//!
//! Per-event cost is a single DashMap insert-if-absent on `Stored` only (`Removed`/`Cleared`
//! are no-ops here) - the same O(1)-per-event class already accepted for `AgeTracker`.
//!
//! Like `AgeTracker`, this never reads or writes either backend's own structures, so it is not
//! subject to this crate's `AGENTS.md` hot-path lock/traversal review - it is a side-table
//! observer of the same event stream, not a change to the primary tree or lower-tier DashMap.
//! **Not yet load-tested at a high Store-event rate**; treat the resulting gauge as
//! diagnostic-only until that measurement exists
//! (`DASHBOARD_METRICS_ENGINEERING_PLAN.md` section 7).

use std::time::{Duration, Instant};

use dashmap::DashMap;
use rustc_hash::FxBuildHasher;

use crate::protocols::*;

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

    /// Evict entries whose first-seen time is older than `window`, then return the remaining
    /// distinct-block count. O(tracker size) - call only from a slow background sampler, same
    /// cost class as `AgeTracker::resident_age_percentiles`.
    pub(super) fn prune_and_count(&self, window: Duration) -> usize {
        let now = Instant::now();
        self.first_seen
            .retain(|_, first_seen| now.saturating_duration_since(*first_seen) < window);
        self.first_seen.len()
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
        assert_eq!(tracker.prune_and_count(Duration::from_secs(3600)), 1);
    }

    #[test]
    fn distinct_hashes_count_separately() {
        let tracker = DemandTracker::new();
        tracker.observe_event(&store_event(1, 0, 42));
        tracker.observe_event(&store_event(1, 0, 43));
        assert_eq!(tracker.prune_and_count(Duration::from_secs(3600)), 2);
    }

    #[test]
    fn remove_does_not_uncount_demand() {
        let tracker = DemandTracker::new();
        tracker.observe_event(&store_event(1, 0, 42));
        tracker.observe_event(&remove_event(1, 0, 42));
        assert_eq!(tracker.prune_and_count(Duration::from_secs(3600)), 1);
    }

    #[test]
    fn entries_older_than_window_are_pruned() {
        let tracker = DemandTracker::new();
        tracker.observe_event(&store_event(1, 0, 42));
        // A zero-width window prunes everything already observed - proves the eviction path
        // actually runs, without needing a real sleep in the test.
        assert_eq!(tracker.prune_and_count(Duration::from_secs(0)), 0);
    }
}
