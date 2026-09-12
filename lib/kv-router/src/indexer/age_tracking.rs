// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic-only block-lifetime ("age-at-eviction", 1a/2a) and live/resident-age (1b)
//! tracking (`KVCR_INDEX_HEALTH_DESIGN.md`).
//!
//! This is a side-table that observes the same `RouterEvent` stream `ThreadPoolIndexer`
//! already dispatches to its backend, entirely independent of either backend's own
//! internal shape - the primary `ConcurrentRadixTreeCompressed`'s `Node`, or
//! `LowerTierIndexer`'s `EdgeOwnersEntry`/`DashMap`. Both of those carry real
//! correctness/perf-review requirements (see this crate's `AGENTS.md`: hot-path lock/
//! traversal changes need a named reviewer and before/after benchmarks) that a
//! diagnostic-only addition should not have to clear, and should not risk by touching that
//! code at all. This module never reads or writes either backend's state.
//!
//! **Tradeoff, stated plainly**: this tracks "how long has this (worker, dp_rank, block)
//! triple been observed as stored, from the event stream's point of view" - not the
//! backend's own authoritative internal state. If a backend silently rejects a Store or
//! Remove event (a real possibility - see `KvIndexerMetrics::increment_event_applied`'s
//! `status` outcomes), this side-table can drift from what the backend actually holds.
//! Given the metric this feeds is already documented as approximate/diagnostic-only
//! ("how fast is the cache turning over," not an exact per-block guarantee), this is an
//! intentional, reasoned trade of perfect fidelity for zero risk to the protected hot-path
//! structures. Same restart caveat as the rest of this design: a process restart loses all
//! tracked insertion times with no recovery, so expect a burst of misleadingly short
//! lifetimes right after any router restart (correction 7 in the design doc).
//!
//! Cost: O(1) DashMap op per stored/removed block (insert-if-absent on Store, remove +
//! histogram observe on Remove) for 1a/2a, same event volume the backend already processes -
//! no new per-event scanning. `Cleared` does a bounded retain-scan over this tracker's own
//! (typically small, capacity-bounded) entry set for the one worker being cleared, not the
//! whole fleet.
//!
//! 1b (`resident_age_percentiles`) is a different cost class: it walks every entry this
//! tracker currently holds (which is, at any moment, exactly the set of blocks this tier
//! considers resident - entries are removed on the matching Remove/Cleared event above), so
//! it is O(this tracker's live size), same cost class the design doc already accepted for
//! the pre-existing `redundancy_stats()` walk. Call it rarely (the sampler in
//! `lib/llm/src/kv_router/indexer/mod.rs` uses the same slow interval as the redundancy
//! sampler), never on a per-request path. Reusing this side-table for 1b - rather than a
//! separate walk over the primary tree or the lower-tier `DashMap`, as the original design
//! doc draft assumed - avoids a second new data structure: the set of "things currently
//! tracked here" already is the live/resident set 1b needs.

use std::time::Instant;

use dashmap::DashMap;
use rustc_hash::FxBuildHasher;

use super::metrics::KvIndexerMetrics;
use crate::protocols::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct AgeKey {
    worker_id: WorkerId,
    dp_rank: DpRank,
    block_hash: ExternalSequenceBlockHash,
}

/// Per-`ThreadPoolIndexer` side-table of block insertion times, used only to compute
/// age-at-eviction on a matching `Removed` event. See module docs for the observation-vs-
/// backend-state caveat. One instance covers whichever tier its owning `ThreadPoolIndexer`
/// serves - the tier label recorded into the histogram comes from each event's own
/// `storage_tier` field, not from any assumption about which `ThreadPoolIndexer` this is.
#[derive(Default)]
pub(super) struct AgeTracker {
    inserted_at: DashMap<AgeKey, Instant, FxBuildHasher>,
}

impl AgeTracker {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn observe_event(&self, event: &RouterEvent, metrics: Option<&KvIndexerMetrics>) {
        let tier_label = tier_label(event.storage_tier);
        match &event.event.data {
            KvCacheEventData::Stored(store) => {
                for block in &store.blocks {
                    let key = AgeKey {
                        worker_id: event.worker_id,
                        dp_rank: event.event.dp_rank,
                        block_hash: block.block_hash,
                    };
                    // Insert-if-absent: a duplicate/redundant store for an already-tracked
                    // block reflects continued residency, not a fresh insert - resetting the
                    // timer here would bias every lifetime sample short.
                    self.inserted_at.entry(key).or_insert_with(Instant::now);
                }
            }
            KvCacheEventData::Removed(remove) => {
                for block_hash in &remove.block_hashes {
                    let key = AgeKey {
                        worker_id: event.worker_id,
                        dp_rank: event.event.dp_rank,
                        block_hash: *block_hash,
                    };
                    // Blocks this tracker never saw stored (pre-dates this process, or a
                    // block this tracker missed for any other reason) are silently skipped -
                    // there is no recoverable insertion time for them.
                    if let Some((_, inserted_at)) = self.inserted_at.remove(&key)
                        && let Some(metrics) = metrics
                    {
                        metrics.observe_block_lifetime(tier_label, inserted_at.elapsed().as_secs_f64());
                    }
                }
            }
            KvCacheEventData::Cleared => {
                let worker_id = event.worker_id;
                let dp_rank = event.event.dp_rank;
                self.inserted_at
                    .retain(|key, _| key.worker_id != worker_id || key.dp_rank != dp_rank);
            }
        }
    }

    /// Live/resident age (design doc 1b): min/p50/p99/max of `now - inserted_at` across every
    /// block this tracker currently considers resident. `None` when nothing is tracked yet
    /// (e.g. immediately after a router restart, before any Store event has landed).
    pub(super) fn resident_age_percentiles(&self) -> Option<AgeStats> {
        let now = Instant::now();
        let mut ages: Vec<f64> = self
            .inserted_at
            .iter()
            .map(|entry| now.saturating_duration_since(*entry.value()).as_secs_f64())
            .collect();
        if ages.is_empty() {
            return None;
        }
        ages.sort_by(|a, b| a.partial_cmp(b).expect("ages are finite, non-NaN durations"));
        let n = ages.len();
        let pct = |p: f64| ages[((p * n as f64) as usize).min(n - 1)];
        Some(AgeStats {
            min: ages[0],
            p50: pct(0.50),
            p99: pct(0.99),
            max: ages[n - 1],
        })
    }
}

/// Result of [`AgeTracker::resident_age_percentiles`], exposed via
/// `ThreadPoolIndexer::resident_age_percentiles()` - crate-public (like `RedundancyStats`)
/// since `lib/llm`'s periodic samplers read it across the crate boundary.
pub struct AgeStats {
    pub min: f64,
    pub p50: f64,
    pub p99: f64,
    pub max: f64,
}

fn tier_label(tier: StorageTier) -> &'static str {
    match tier {
        StorageTier::Device => "device",
        StorageTier::HostPinned => "host_pinned",
        StorageTier::Disk => "disk",
        StorageTier::External => "external",
    }
}

#[cfg(all(test, feature = "metrics"))]
mod tests {
    use super::*;
    use crate::indexer::metrics::KvIndexerMetrics;

    fn store_event(worker_id: WorkerId, dp_rank: DpRank, hash: u64, tier: StorageTier) -> RouterEvent {
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
            tier,
        )
    }

    fn remove_event(worker_id: WorkerId, dp_rank: DpRank, hash: u64, tier: StorageTier) -> RouterEvent {
        RouterEvent::with_storage_tier(
            worker_id,
            KvCacheEvent {
                event_id: 2,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(hash)],
                }),
                dp_rank,
            },
            tier,
        )
    }

    fn cleared_event(worker_id: WorkerId, dp_rank: DpRank) -> RouterEvent {
        RouterEvent::with_storage_tier(
            worker_id,
            KvCacheEvent {
                event_id: 3,
                data: KvCacheEventData::Cleared,
                dp_rank,
            },
            StorageTier::Device,
        )
    }

    #[test]
    fn store_then_remove_records_one_lifetime_sample() {
        let tracker = AgeTracker::new();
        let metrics = KvIndexerMetrics::new_unregistered();

        tracker.observe_event(&store_event(1, 0, 42, StorageTier::Device), Some(&metrics));
        tracker.observe_event(&remove_event(1, 0, 42, StorageTier::Device), Some(&metrics));

        let hist = metrics.block_lifetime_seconds.with_label_values(&["device"]);
        assert_eq!(hist.get_sample_count(), 1);
        assert_eq!(tracker.inserted_at.len(), 0);
    }

    #[test]
    fn remove_without_prior_store_records_nothing() {
        let tracker = AgeTracker::new();
        let metrics = KvIndexerMetrics::new_unregistered();

        tracker.observe_event(&remove_event(1, 0, 42, StorageTier::Device), Some(&metrics));

        let hist = metrics.block_lifetime_seconds.with_label_values(&["device"]);
        assert_eq!(hist.get_sample_count(), 0);
    }

    #[test]
    fn duplicate_store_does_not_reset_insertion_time() {
        let tracker = AgeTracker::new();
        tracker.observe_event(&store_event(1, 0, 42, StorageTier::Device), None);
        tracker.observe_event(&store_event(1, 0, 42, StorageTier::Device), None);
        assert_eq!(tracker.inserted_at.len(), 1);
    }

    #[test]
    fn host_pinned_tier_labels_its_own_histogram_series() {
        let tracker = AgeTracker::new();
        let metrics = KvIndexerMetrics::new_unregistered();

        tracker.observe_event(&store_event(1, 0, 42, StorageTier::HostPinned), Some(&metrics));
        tracker.observe_event(&remove_event(1, 0, 42, StorageTier::HostPinned), Some(&metrics));

        assert_eq!(
            metrics
                .block_lifetime_seconds
                .with_label_values(&["host_pinned"])
                .get_sample_count(),
            1
        );
        assert_eq!(
            metrics
                .block_lifetime_seconds
                .with_label_values(&["device"])
                .get_sample_count(),
            0
        );
    }

    #[test]
    fn cleared_purges_only_that_worker_dp_rank() {
        let tracker = AgeTracker::new();
        tracker.observe_event(&store_event(1, 0, 42, StorageTier::Device), None);
        tracker.observe_event(&store_event(2, 0, 43, StorageTier::Device), None);

        tracker.observe_event(&cleared_event(1, 0), None);

        assert_eq!(tracker.inserted_at.len(), 1);
        assert!(
            tracker
                .inserted_at
                .contains_key(&AgeKey {
                    worker_id: 2,
                    dp_rank: 0,
                    block_hash: ExternalSequenceBlockHash(43),
                })
        );
    }
}
