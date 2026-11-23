// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request classifier plugin contract: classify a request into a policy class before it
//! reaches the scheduler's queue.
//!
//! The stage lives in `dynamo-admission`; this module re-exports it at the paths plugins and
//! hosts already name and adds what only the composed router knows: the host's instance
//! config and the registry.

mod config;
mod registry;

pub use dynamo_admission::classifier::{
    AbortCause, ClassificationError, ClassificationOverrides, ClassifierError, ClassifyEvent,
    ClassifyRequest, DispatchDecision, DispatchGate, DispatchHead, DispatchRecheck, HoldWake,
    NoWorkers, RequestClassifier, RequestClassifierContext, RequestClassifierRuntime,
    RequestLifecycle, RequestProgress, RequestProgressUpdater, SessionHolder, SessionIndexReader,
    SessionIndexTotals, SessionResidency, SessionTier, WakeScope, WorkerCacheView, WorkerLoadView,
    WorkerView,
};

use std::sync::Arc;

use dynamo_kv_events::residency::StorageTier;
use dynamo_kv_indexer::{SessionPrefixIndexer, SessionResidencyOwner};

/// The router's [`SessionPrefixIndexer`] as the read-only [`SessionIndexReader`] a classifier
/// gets from [`RequestClassifierContext::session_index`]. It only reads; the router keeps
/// writing the index from KV events and route matches as before.
#[derive(Clone)]
pub struct SessionPrefixIndexReader {
    index: Arc<SessionPrefixIndexer>,
}

impl SessionPrefixIndexReader {
    pub fn new(index: Arc<SessionPrefixIndexer>) -> Self {
        Self { index }
    }
}

fn session_tier(tier: StorageTier) -> SessionTier {
    match tier {
        StorageTier::Device => SessionTier::Device,
        StorageTier::HostPinned => SessionTier::Host,
        StorageTier::Disk => SessionTier::Disk,
        StorageTier::External => SessionTier::External,
    }
}

impl SessionIndexReader for SessionPrefixIndexReader {
    fn session_residency(&self, session_id: &str) -> Vec<SessionResidency> {
        self.index
            .get_session_resident_depths(session_id)
            .into_iter()
            .map(|(owner, tier, blocks)| SessionResidency {
                holder: match owner {
                    SessionResidencyOwner::Worker(worker) => SessionHolder::Worker(worker),
                    SessionResidencyOwner::SharedCache(_) => SessionHolder::SharedCache,
                },
                tier: session_tier(tier),
                blocks,
            })
            .collect()
    }

    fn totals(&self) -> SessionIndexTotals {
        let (sessions, per_tier) = self.index.residency_totals();
        let mut totals = SessionIndexTotals {
            sessions: sessions as u64,
            ..SessionIndexTotals::default()
        };
        for (tier, blocks) in per_tier {
            match session_tier(tier) {
                SessionTier::Device => totals.device_blocks += blocks,
                SessionTier::Host => totals.host_blocks += blocks,
                SessionTier::Disk => totals.disk_blocks += blocks,
                SessionTier::External => totals.external_blocks += blocks,
            }
        }
        totals
    }

    fn unique_residency(&self) -> Vec<SessionResidency> {
        self.index
            .unique_residency()
            .into_iter()
            .map(|(owner, tier, blocks)| SessionResidency {
                holder: match owner {
                    SessionResidencyOwner::Worker(worker) => SessionHolder::Worker(worker),
                    SessionResidencyOwner::SharedCache(_) => SessionHolder::SharedCache,
                },
                tier: session_tier(tier),
                blocks,
            })
            .collect()
    }
}

pub(crate) use config::RawRequestClassifierConfig;
pub use config::RequestClassifierConfig;
pub(crate) use registry::RequestClassifierRegistry;
pub use registry::{
    RequestClassifierFactory, RequestClassifierParameters, RequestClassifierProvider,
    RequestClassifierProviderError, RequestClassifierRegistryError,
};

#[cfg(test)]
mod tests {
    use dynamo_kv_indexer::testing::make_blocks;
    use dynamo_router_protocol::worker::WorkerWithDpRank;

    use super::*;

    #[test]
    fn the_reader_maps_owners_tiers_and_depths_from_the_index() {
        let index = Arc::new(SessionPrefixIndexer::new());
        let chain: Vec<_> = make_blocks(vec![1, 2, 3])
            .into_iter()
            .map(|b| b.block_hash)
            .collect();
        let worker = WorkerWithDpRank::new(4, 1);
        index
            .update_session_from_stored_blocks("s", worker, None, &chain)
            .unwrap();
        index
            .update_session_from_stored_blocks_in_tier(
                "s",
                SessionResidencyOwner::Worker(worker),
                StorageTier::HostPinned,
                None,
                &chain[..1],
            )
            .unwrap();
        let reader = SessionPrefixIndexReader::new(Arc::clone(&index));
        let mut rows = reader.session_residency("s");
        rows.sort_by_key(|r| r.blocks);
        assert_eq!(
            rows,
            vec![
                SessionResidency {
                    holder: SessionHolder::Worker(worker),
                    tier: SessionTier::Host,
                    blocks: 1
                },
                SessionResidency {
                    holder: SessionHolder::Worker(worker),
                    tier: SessionTier::Device,
                    blocks: 3
                },
            ]
        );
        assert!(reader.session_residency("other").is_empty());
        assert_eq!(
            reader.totals(),
            SessionIndexTotals {
                sessions: 1,
                device_blocks: 3,
                host_blocks: 1,
                ..SessionIndexTotals::default()
            }
        );
        let mut unique = reader.unique_residency();
        unique.sort_by_key(|r| r.blocks);
        assert_eq!(unique, rows);
    }
}
