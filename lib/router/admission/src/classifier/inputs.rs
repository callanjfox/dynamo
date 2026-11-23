// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_router_protocol::columns::WorkerInfo;
use dynamo_router_protocol::worker::WorkerWithDpRank;
use tokio::time::Instant;

/// Storage tier of a session's resident blocks, as the router's session index reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionTier {
    Device,
    Host,
    Disk,
    External,
}

/// Holder of part of a session's KV: a worker rank, or a shared cache outside the workers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionHolder {
    Worker(WorkerWithDpRank),
    SharedCache,
}

/// One holder and tier of a session, and the depth in blocks of the deepest prefix of the
/// session that holder reported in that tier. The index assumes, as it does on the device, that
/// a reported block's ancestors are resident in the same tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionResidency {
    pub holder: SessionHolder,
    pub tier: SessionTier,
    pub blocks: u64,
}

/// Totals over every session the index tracks: sessions, and resident blocks per tier summed
/// over sessions and holders (a block shared by two sessions counts once per session).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SessionIndexTotals {
    pub sessions: u64,
    pub device_blocks: u64,
    pub host_blocks: u64,
    pub disk_blocks: u64,
    pub external_blocks: u64,
}

/// Read-only view of the router's session prefix index for plugins. Implementations read
/// cached index state and must not block on I/O.
pub trait SessionIndexReader: Send + Sync {
    /// Where `session_id` is resident, one row per holder and tier; empty when unknown.
    fn session_residency(&self, session_id: &str) -> Vec<SessionResidency>;
    /// Totals over every tracked session (O(sessions); for metrics, not the request path).
    fn totals(&self) -> SessionIndexTotals;
    /// Unique resident blocks per holder and tier (F34): a block several sessions reach counts
    /// once per holder and tier, as the cache holds it. O(unique blocks); for a periodic step,
    /// not the request path. Empty when the reader cannot tell (the default).
    fn unique_residency(&self) -> Vec<SessionResidency> {
        Vec::new()
    }
}

/// Cached discovery inputs for one router. Workers appear and disappear as the
/// host processes registration updates; this does not probe worker health.
#[derive(Clone)]
pub struct RequestClassifierContext {
    block_size: u32,
    workers: Arc<dyn Fn() -> Vec<WorkerInfo> + Send + Sync>,
    now: Arc<dyn Fn() -> Instant + Send + Sync>,
    session_index: Option<Arc<dyn SessionIndexReader>>,
}

impl RequestClassifierContext {
    /// The callback must return cached state without blocking I/O.
    pub fn new(
        block_size: u32,
        workers: impl Fn() -> Vec<WorkerInfo> + Send + Sync + 'static,
    ) -> Self {
        Self {
            block_size,
            workers: Arc::new(workers),
            now: Arc::new(Instant::now),
            session_index: None,
        }
    }

    /// Give classifiers a read-only handle on the router's session prefix index. Hosts set it
    /// when `enable_session_prefix_index` is on; without it [`Self::session_index`] is `None`.
    pub fn with_session_index(mut self, index: Arc<dyn SessionIndexReader>) -> Self {
        self.session_index = Some(index);
        self
    }

    /// The router's session prefix index, when the host tracks one.
    pub fn session_index(&self) -> Option<Arc<dyn SessionIndexReader>> {
        self.session_index.clone()
    }

    /// Replace the clock a classifier reads time through. Hosts keep the default
    /// (`tokio::time::Instant::now`); offline replay passes its virtual clock so a
    /// classifier's holds and deadlines run in simulated time.
    pub fn with_clock(mut self, now: impl Fn() -> Instant + Send + Sync + 'static) -> Self {
        self.now = Arc::new(now);
        self
    }

    /// The current time on this context's clock.
    pub fn now(&self) -> Instant {
        (self.now)()
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    pub fn workers(&self) -> Vec<WorkerInfo> {
        (self.workers)()
    }
}

impl std::fmt::Debug for RequestClassifierContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RequestClassifierContext")
            .field("block_size", &self.block_size)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Empty;

    impl SessionIndexReader for Empty {
        fn session_residency(&self, _session_id: &str) -> Vec<SessionResidency> {
            Vec::new()
        }
        fn totals(&self) -> SessionIndexTotals {
            SessionIndexTotals {
                sessions: 3,
                ..SessionIndexTotals::default()
            }
        }
    }

    #[test]
    fn the_session_index_is_absent_unless_the_host_sets_one() {
        let context = RequestClassifierContext::new(16, Vec::new);
        assert!(context.session_index().is_none());
        let context = context.with_session_index(Arc::new(Empty));
        let cloned = context.clone();
        assert_eq!(cloned.session_index().unwrap().totals().sessions, 3);
    }
}
