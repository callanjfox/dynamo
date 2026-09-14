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
//! **Tradeoff, stated plainly**: this tracks "how long has this (residency owner, block)
//! pair been observed as stored, from the event stream's point of view" - not the backend's
//! own authoritative internal state. If a backend silently rejects a Store or Remove event
//! (a real possibility - see `KvIndexerMetrics::increment_event_applied`'s `status`
//! outcomes), this side-table can drift from what the backend actually holds. Given the
//! metric this feeds is already documented as approximate/diagnostic-only ("how fast is the
//! cache turning over," not an exact per-block guarantee), this is an intentional, reasoned
//! trade of perfect fidelity for zero risk to the protected hot-path structures. Same restart
//! caveat as the rest of this design: a process restart loses all tracked insertion times
//! with no recovery, so expect a burst of misleadingly short lifetimes right after any
//! router restart (correction 7 in the design doc).
//!
//! **Keyed by `ResidencyOwner`, not raw `(worker_id, dp_rank)`** (fixed after an adversarial
//! review caught the original version keying every event by its literal `worker_id`,
//! including `CacheOwner`-domain events - e.g. KVCR host-pinned/G2 residency - whose whole
//! design point is a stable identity that survives worker failover. A `CacheOwner` event's
//! wire `worker_id` is `attachment_worker.worker_id`: whichever worker currently holds the
//! attachment (`lib/llm/src/kv_router/publisher/state_agent.rs`), which changes across a
//! worker replacement even though the logical owner does not. Keying by raw `worker_id`
//! meant a `Removed`/`Cleared` event after any such replacement could never match the entry
//! inserted under the old `worker_id`, permanently orphaning it - unbounded slow growth in
//! `inserted_at`, and `resident_age_percentiles()`'s p99/max skewed upward forever by phantom
//! entries. `RouterEvent::residency_owner()` already resolves the correct stable identity for
//! both domains (`Worker(WorkerWithDpRank)` or `CacheOwner(CacheOwnerId)`) - the same
//! resolution `LowerTierIndexer::apply_event` already relies on for its own real backend
//! state, so this fix makes the diagnostic side-table agree with the actual backend's
//! notion of ownership instead of silently disagreeing with it.
//!
//! Cost: O(1) DashMap op per stored/removed block (insert-if-absent on Store, remove +
//! histogram observe on Remove) for 1a/2a, same event volume the backend already processes -
//! no new per-event scanning. `Cleared` does a full scan of this tracker's own entry set
//! (a `retain` over every tracked entry, filtering by the resolved `ResidencyOwner`) - bounded
//! by this side-table's own size, not the whole fleet, but O(this tracker's size) rather than
//! O(1), same cost class as `resident_age_percentiles` below.
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

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use dashmap::DashMap;
use rustc_hash::FxBuildHasher;

use super::metrics::KvIndexerMetrics;
use crate::protocols::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct AgeKey {
    owner: ResidencyOwner,
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
                // An event whose residency domain can't be resolved (e.g. a CacheOwner-domain
                // store on the device tier, which `resolved_residency_domain()` rejects) is
                // silently skipped here, same as an event this tracker never saw - there is no
                // owner identity to key it by.
                let Ok(owner) = event.residency_owner() else {
                    return;
                };
                for block in &store.blocks {
                    let key = AgeKey {
                        owner,
                        block_hash: block.block_hash,
                    };
                    // Insert-if-absent: a duplicate/redundant store for an already-tracked
                    // block reflects continued residency, not a fresh insert - resetting the
                    // timer here would bias every lifetime sample short.
                    self.inserted_at.entry(key).or_insert_with(Instant::now);
                }
            }
            KvCacheEventData::Removed(remove) => {
                let Ok(owner) = event.residency_owner() else {
                    return;
                };
                for block_hash in &remove.block_hashes {
                    let key = AgeKey {
                        owner,
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
            // Mirrors `LowerTierIndexer::apply_event`'s Cleared handling exactly (same
            // `reset_scope()`/`ResidencyOwner` resolution), so this side-table purges the same
            // entries the real backend would - not the "current worker_id" this event happens
            // to carry, which for a CacheOwner-domain clear may not be the worker that owns
            // any of the entries being cleared.
            KvCacheEventData::Cleared => {
                let Ok(Some(scope)) = event.reset_scope() else {
                    return;
                };
                let worker_owner = ResidencyOwner::worker(WorkerWithDpRank::new(
                    event.worker_id,
                    event.event.dp_rank,
                ));
                match scope {
                    ResetScope::All => {
                        let cache_owner = event.state_source.map(ResidencyOwner::cache_owner);
                        self.inserted_at.retain(|key, _| {
                            key.owner != worker_owner && Some(key.owner) != cache_owner
                        });
                    }
                    ResetScope::Domain(ResidencyDomain::Worker) => {
                        self.inserted_at.retain(|key, _| key.owner != worker_owner);
                    }
                    ResetScope::Domain(ResidencyDomain::CacheOwner) => {
                        let Some(cache_owner_id) = event.state_source else {
                            return;
                        };
                        let cache_owner = ResidencyOwner::cache_owner(cache_owner_id);
                        self.inserted_at.retain(|key, _| key.owner != cache_owner);
                    }
                }
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

    /// Live/resident block count, grouped by worker (2026-09-14 follow-up to
    /// `KVCR_INDEX_HEALTH_DESIGN.md`: "how hard would it be to have metrics on G1/G2 usage per
    /// worker" - answered here for the tier this tracker serves). Reuses the exact same live
    /// entry set `resident_age_percentiles()` already walks, just grouped by owner instead of
    /// collapsed to one set of percentiles - no new bookkeeping, same O(this tracker's live
    /// size) cost class, same "background sampler only" caveat.
    ///
    /// `CacheOwner`-domain entries (a KVCR/cross-worker-shared G2 tier's stable identity, not
    /// produced by a local-only-G2 deployment - see the module docs' "Keyed by
    /// `ResidencyOwner`" section) are skipped: attributing one to "a worker" would require
    /// resolving the current cache-owner-to-worker attachment, which this diagnostic
    /// side-table deliberately does not do. They simply don't appear in the result, same as a
    /// worker with a fully-evicted (empty) index - the caller must treat "absent" as 0, not
    /// skip updating that worker's gauge (see `spawn_live_index_gauge_sampler`'s
    /// `known_workers`/`seen_this_round` handling for the established pattern).
    pub(super) fn resident_block_counts_by_worker(&self) -> HashMap<WorkerWithDpRank, usize> {
        let mut counts = HashMap::new();
        for entry in self.inserted_at.iter() {
            if let ResidencyOwner::Worker(worker) = entry.key().owner {
                *counts.entry(worker).or_insert(0usize) += 1;
            }
        }
        counts
    }

    /// Distinct block hashes currently resident anywhere in this tier, fleet-wide, deduplicated
    /// across owners (2026-09-14, "how much of G1 is in G2" - answered by intersecting this
    /// tracker's set against the other tier's own `AgeTracker`, one instance per tier, at the
    /// call site in `lib/llm`). Block hashes are content-addressed - the same hash resident in
    /// both tiers' sets means that exact block's content is duplicated across tiers, not merely
    /// a coincidence - so a plain set intersection is a real cross-tier overlap measurement,
    /// not an approximation. Same reuse-the-existing-side-table reasoning, cost class, and
    /// background-sampler-only caveat as `resident_age_percentiles`/
    /// `resident_block_counts_by_worker` above.
    pub(super) fn resident_block_hashes(&self) -> HashSet<ExternalSequenceBlockHash, FxBuildHasher> {
        self.inserted_at
            .iter()
            .map(|entry| entry.key().block_hash)
            .collect()
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
    use crate::identity::{
        CacheOwnerId, CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId,
        RoutingScopeId, StableDpSlotId,
    };
    use crate::indexer::metrics::KvIndexerMetrics;

    fn cache_owner_id() -> CacheOwnerId {
        CacheOwnerId::new(
            PoolId::new(
                IndexerDomainId::new(
                    CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                    RoutingScopeId::new([2; 16], IdentitySource::Explicit),
                ),
                DcId::new(3),
            ),
            StableDpSlotId::new([4; 16], IdentitySource::Explicit),
        )
    }

    fn cache_owner_store_event(worker_id: WorkerId, hash: u64, tier: StorageTier) -> RouterEvent {
        RouterEvent::with_cache_owner(
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
                dp_rank: 0,
            },
            tier,
            cache_owner_id(),
        )
    }

    fn cache_owner_remove_event(worker_id: WorkerId, hash: u64, tier: StorageTier) -> RouterEvent {
        RouterEvent::with_cache_owner(
            worker_id,
            KvCacheEvent {
                event_id: 2,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(hash)],
                }),
                dp_rank: 0,
            },
            tier,
            cache_owner_id(),
        )
    }

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
            tracker.inserted_at.contains_key(&AgeKey {
                owner: ResidencyOwner::worker(WorkerWithDpRank::new(2, 0)),
                block_hash: ExternalSequenceBlockHash(43),
            })
        );
    }

    #[test]
    fn resident_block_counts_by_worker_groups_by_owner() {
        let tracker = AgeTracker::new();
        tracker.observe_event(&store_event(1, 0, 42, StorageTier::Device), None);
        tracker.observe_event(&store_event(1, 0, 43, StorageTier::Device), None);
        tracker.observe_event(&store_event(2, 0, 44, StorageTier::Device), None);

        let counts = tracker.resident_block_counts_by_worker();
        assert_eq!(
            counts.get(&WorkerWithDpRank::new(1, 0)),
            Some(&2),
            "worker 1 has two resident blocks"
        );
        assert_eq!(
            counts.get(&WorkerWithDpRank::new(2, 0)),
            Some(&1),
            "worker 2 has one resident block"
        );
        assert_eq!(counts.len(), 2, "no phantom or missing workers");
    }

    #[test]
    fn resident_block_counts_by_worker_skips_cache_owner_entries() {
        let tracker = AgeTracker::new();
        tracker.observe_event(
            &cache_owner_store_event(1, 99, StorageTier::HostPinned),
            None,
        );
        tracker.observe_event(&store_event(2, 0, 100, StorageTier::HostPinned), None);

        let counts = tracker.resident_block_counts_by_worker();
        assert_eq!(
            counts.len(),
            1,
            "the CacheOwner-domain entry must not appear under any worker key"
        );
        assert_eq!(counts.get(&WorkerWithDpRank::new(2, 0)), Some(&1));
    }

    #[test]
    fn resident_block_hashes_dedupes_across_owners() {
        let tracker = AgeTracker::new();
        // Same block hash tracked under two different workers - still one distinct hash.
        tracker.observe_event(&store_event(1, 0, 42, StorageTier::Device), None);
        tracker.observe_event(&store_event(2, 0, 42, StorageTier::Device), None);
        tracker.observe_event(&store_event(1, 0, 43, StorageTier::Device), None);

        let hashes = tracker.resident_block_hashes();
        assert_eq!(hashes.len(), 2);
        assert!(hashes.contains(&ExternalSequenceBlockHash(42)));
        assert!(hashes.contains(&ExternalSequenceBlockHash(43)));
    }

    /// Regression test for the bug an adversarial review caught: a `CacheOwner`-domain block
    /// (e.g. KVCR host-pinned/G2 residency) must remain trackable across a worker failover,
    /// since its stable identity is the cache owner, not whichever worker currently holds the
    /// attachment. Before the fix (keying by raw `worker_id`), the `Removed` event below -
    /// carrying the *new* worker's id, exactly as `state_agent.rs` really emits it after a
    /// reattachment - would never match the entry inserted under the *old* worker's id, so it
    /// would leak forever and never record a lifetime sample.
    #[test]
    fn cache_owner_identity_survives_worker_failover() {
        let tracker = AgeTracker::new();
        let metrics = KvIndexerMetrics::new_unregistered();

        // Worker 1 holds the attachment when the block is first stored.
        tracker.observe_event(
            &cache_owner_store_event(1, 99, StorageTier::HostPinned),
            Some(&metrics),
        );
        assert_eq!(tracker.inserted_at.len(), 1);

        // The attachment fails over to worker 2 - same cache owner, different worker_id on
        // the wire - and worker 2 is the one that reports the eventual removal.
        tracker.observe_event(
            &cache_owner_remove_event(2, 99, StorageTier::HostPinned),
            Some(&metrics),
        );

        assert_eq!(
            tracker.inserted_at.len(),
            0,
            "the entry must be found and removed via the stable cache-owner identity, \
             not orphaned under worker 1's id"
        );
        assert_eq!(
            metrics
                .block_lifetime_seconds
                .with_label_values(&["host_pinned"])
                .get_sample_count(),
            1,
            "the lifetime sample must still be recorded despite the worker_id change"
        );
    }
}
