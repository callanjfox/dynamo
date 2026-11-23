// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session lineage tracked independently of physical cache eviction.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use rustc_hash::{FxHashMap, FxHashSet};
use slotmap::{SlotMap, new_key_type};

use crate::prelude::{ExternalSequenceBlockHash, StorageTier, WorkerId, WorkerWithDpRank};

/// Logical session identity as owned by this index.
pub type SessionId = String;

const DEFAULT_MAX_SESSIONS: usize = 16_384;
const MAX_SESSION_PREFIX_ID_BYTES: usize = 256;
/// Same cadence as kv-router's `cleanup::CLEANUP_INTERVAL_MS`; each stage owns its constant.
const CLEANUP_INTERVAL_MS: u64 = 5 * 60 * 1000;
const CLEANUP_INTERVAL: Duration = Duration::from_millis(CLEANUP_INTERVAL_MS);

new_key_type! {
    /// Generational handle to a [`LogicalNode`] in the arena.
    pub struct NodeId;
}

/// One block and its liveness links in the logical session forest.
#[derive(Clone, Copy, Debug)]
pub struct LogicalNode {
    block_hash: ExternalSequenceBlockHash,
    parent: Option<NodeId>,
    frontier_refs: u32,
    child_count: u32,
}

impl LogicalNode {
    pub fn block_hash(&self) -> ExternalSequenceBlockHash {
        self.block_hash
    }

    pub fn parent(&self) -> Option<NodeId> {
        self.parent
    }

    pub fn frontier_refs(&self) -> u32 {
        self.frontier_refs
    }

    pub fn child_count(&self) -> u32 {
        self.child_count
    }
}

/// Errors surfaced by [`SessionPrefixIndexer`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SessionPrefixIndexError {
    /// A lineage query named an unknown anchor hash.
    #[error("unknown anchor block hash {0:?}")]
    UnknownAnchor(ExternalSequenceBlockHash),

    /// A stored chain named an unknown parent without reconnecting known lineage.
    #[error("unknown parent block hash {parent:?}")]
    UnknownParent { parent: ExternalSequenceBlockHash },

    /// A block was attached beneath a conflicting parent.
    #[error("block {block:?} is already parented elsewhere")]
    ConflictingParent { block: ExternalSequenceBlockHash },

    /// A graft would create a parent cycle.
    #[error("block {block:?} would become its own ancestor")]
    CyclicParent { block: ExternalSequenceBlockHash },
}

/// Thread-safe session-aware logical prefix index.
#[derive(Debug, Default)]
pub struct SessionPrefixIndexer {
    state: RwLock<IndexState>,
}

/// Shared KV store that reports session residency without being a worker rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedCacheBackend {
    /// Mooncake Store, written by SGLang HiCache or its direct external linker.
    Mooncake,
}

/// Holder of one session frontier: a worker rank, or a shared store outside the workers.
///
/// The storage tier stays the physical medium. Which backend holds a shared frontier
/// is this owner's job, so a new backend adds one variant here rather than one tier
/// per medium.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionResidencyOwner {
    Worker(WorkerWithDpRank),
    SharedCache(SharedCacheBackend),
}

impl SessionResidencyOwner {
    pub const fn as_worker(self) -> Option<WorkerWithDpRank> {
        match self {
            Self::Worker(worker) => Some(worker),
            Self::SharedCache(_) => None,
        }
    }
}

/// Owner and storage tier that hold one session frontier.
type FrontierKey = (SessionResidencyOwner, StorageTier);

#[derive(Debug, Default)]
struct SessionEntry {
    worker_frontiers: FxHashMap<FrontierKey, FxHashSet<NodeId>>,
    last_touch: Option<u64>,
}

#[derive(Debug)]
struct IndexState {
    nodes: SlotMap<NodeId, LogicalNode>,
    /// Each node's children, kept with `child_count`, so a removal can find the frontiers below
    /// a removed block without walking every frontier to its root.
    children: FxHashMap<NodeId, FxHashSet<NodeId>>,
    hash_to_node: FxHashMap<ExternalSequenceBlockHash, NodeId>,
    session_to_worker_frontiers: HashMap<SessionId, SessionEntry>,
    worker_frontier_to_sessions: FxHashMap<FrontierKey, FxHashMap<NodeId, HashSet<SessionId>>>,
    lru: BTreeMap<u64, SessionId>,
    next_touch: u64,
    max_sessions: usize,
    prune_candidates: FxHashSet<NodeId>,
    last_cleanup: Instant,
}

impl Default for IndexState {
    fn default() -> Self {
        Self {
            nodes: SlotMap::default(),
            children: FxHashMap::default(),
            hash_to_node: FxHashMap::default(),
            session_to_worker_frontiers: HashMap::default(),
            worker_frontier_to_sessions: FxHashMap::default(),
            lru: BTreeMap::default(),
            next_touch: 0,
            max_sessions: DEFAULT_MAX_SESSIONS,
            prune_candidates: FxHashSet::default(),
            last_cleanup: Instant::now(),
        }
    }
}

impl SessionPrefixIndexer {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    fn with_max_sessions(max_sessions: usize) -> Self {
        Self {
            state: RwLock::new(IndexState {
                max_sessions: max_sessions.max(1),
                ..IndexState::default()
            }),
        }
    }

    pub fn get_node_from_hash(&self, block_hash: ExternalSequenceBlockHash) -> Option<NodeId> {
        self.state.read().hash_to_node.get(&block_hash).copied()
    }

    pub fn get_node(&self, node_id: NodeId) -> Option<LogicalNode> {
        self.state.read().nodes.get(node_id).copied()
    }

    /// Returns unordered worker-qualified device-tier frontier nodes.
    pub fn get_session_frontiers(&self, session_id: &str) -> Vec<(WorkerWithDpRank, NodeId)> {
        self.get_session_residency(session_id)
            .into_iter()
            .filter(|(_, tier, _)| tier.is_gpu())
            .filter_map(|(owner, _, frontier)| Some((owner.as_worker()?, frontier)))
            .collect()
    }

    /// Returns unordered frontier nodes for every owner and storage tier that
    /// holds part of the session. A tier's frontier is the deepest session block that
    /// tier reported; the index assumes, as it does on device, that its ancestors are
    /// resident in the same tier.
    pub fn get_session_residency(
        &self,
        session_id: &str,
    ) -> Vec<(SessionResidencyOwner, StorageTier, NodeId)> {
        self.state
            .read()
            .session_to_worker_frontiers
            .get(session_id)
            .map(|entry| {
                entry
                    .worker_frontiers
                    .iter()
                    .flat_map(|(&(owner, tier), frontiers)| {
                        frontiers
                            .iter()
                            .map(move |&frontier| (owner, tier, frontier))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// For each owner and storage tier holding part of the session, the depth in blocks of its
    /// deepest frontier (the longest resident prefix of the session there). Read-only; one
    /// parent walk per frontier.
    pub fn get_session_resident_depths(
        &self,
        session_id: &str,
    ) -> Vec<(SessionResidencyOwner, StorageTier, u64)> {
        let state = self.state.read();
        let Some(entry) = state.session_to_worker_frontiers.get(session_id) else {
            return Vec::new();
        };
        entry
            .worker_frontiers
            .iter()
            .filter_map(|(&(owner, tier), frontiers)| {
                let depth = frontiers
                    .iter()
                    .map(|&frontier| state.depth(frontier))
                    .max()?;
                Some((owner, tier, depth))
            })
            .collect()
    }

    /// Sessions tracked, and resident blocks per tier summed over sessions and owners (the
    /// deepest frontier per owner and tier, as in [`Self::get_session_resident_depths`]).
    /// O(sessions x frontiers x depth): for end-of-run metrics, not the request path.
    pub fn residency_totals(&self) -> (usize, Vec<(StorageTier, u64)>) {
        let state = self.state.read();
        let mut per_tier: FxHashMap<StorageTier, u64> = FxHashMap::default();
        for entry in state.session_to_worker_frontiers.values() {
            for (&(_, tier), frontiers) in &entry.worker_frontiers {
                if let Some(depth) = frontiers.iter().map(|&f| state.depth(f)).max() {
                    *per_tier.entry(tier).or_default() += depth;
                }
            }
        }
        (
            state.session_to_worker_frontiers.len(),
            per_tier.into_iter().collect(),
        )
    }

    /// Unique resident blocks per owner and storage tier (F34): every block on a path from one of
    /// that owner's frontiers in that tier to its root, counted once however many sessions'
    /// frontiers reach it (a shared system prompt or a parent's context counts once per owner).
    /// [`Self::residency_totals`] sums per session instead. O(unique nodes) per owner and tier
    /// (each walk stops at the first node already counted): for a periodic controller step, not
    /// the request path.
    pub fn unique_residency(&self) -> Vec<(SessionResidencyOwner, StorageTier, u64)> {
        let state = self.state.read();
        let limit = state.nodes.len();
        let mut out = Vec::with_capacity(state.worker_frontier_to_sessions.len());
        for (&(owner, tier), frontiers) in &state.worker_frontier_to_sessions {
            let mut seen: FxHashSet<NodeId> = FxHashSet::default();
            for &frontier in frontiers.keys() {
                let mut current = Some(frontier);
                while let Some(node) = current {
                    if seen.len() > limit || !seen.insert(node) {
                        break;
                    }
                    current = state.nodes.get(node).and_then(|n| n.parent);
                }
            }
            if !seen.is_empty() {
                out.push((owner, tier, seen.len() as u64));
            }
        }
        out
    }

    /// Returns root-first device-tier chains for one worker, optionally truncated at an anchor.
    pub fn get_session_block_lineage(
        &self,
        session_id: &str,
        worker: WorkerWithDpRank,
        anchor_hash: Option<ExternalSequenceBlockHash>,
    ) -> Result<Vec<Vec<ExternalSequenceBlockHash>>, SessionPrefixIndexError> {
        self.get_session_block_lineage_in_tier(
            session_id,
            SessionResidencyOwner::Worker(worker),
            StorageTier::Device,
            anchor_hash,
        )
    }

    /// Returns root-first chains held by one owner tier, optionally truncated at an anchor.
    pub fn get_session_block_lineage_in_tier(
        &self,
        session_id: &str,
        owner: SessionResidencyOwner,
        tier: StorageTier,
        anchor_hash: Option<ExternalSequenceBlockHash>,
    ) -> Result<Vec<Vec<ExternalSequenceBlockHash>>, SessionPrefixIndexError> {
        let state = self.state.read();

        let anchor_node = match anchor_hash {
            Some(hash) => Some(
                state
                    .hash_to_node
                    .get(&hash)
                    .copied()
                    .ok_or(SessionPrefixIndexError::UnknownAnchor(hash))?,
            ),
            None => None,
        };

        let Some(frontiers) = state
            .session_to_worker_frontiers
            .get(session_id)
            .and_then(|entry| entry.worker_frontiers.get(&(owner, tier)))
        else {
            return Ok(Vec::new());
        };

        let mut lineages = Vec::with_capacity(frontiers.len());
        for &frontier in frontiers {
            let path = state.path_to_root(frontier);
            let start = match anchor_node {
                Some(anchor) => match path.iter().position(|&node| node == anchor) {
                    Some(position) => position,
                    None => continue,
                },
                None => 0,
            };
            lineages.push(
                path[start..]
                    .iter()
                    .map(|&node| state.nodes[node].block_hash)
                    .collect(),
            );
        }
        Ok(lineages)
    }

    /// Records a route-time match and reports whether the frontier advanced.
    pub fn update_session_from_match(
        &self,
        session_id: &str,
        worker: WorkerWithDpRank,
        matched_hash: ExternalSequenceBlockHash,
    ) -> Result<bool, SessionPrefixIndexError> {
        if !valid_session_id(session_id) {
            return Ok(false);
        }
        let mut state = self.state.write();
        let node = state.resolve_or_insert_root(matched_hash);
        let updated = state.advance_frontier(
            session_id,
            (SessionResidencyOwner::Worker(worker), StorageTier::Device),
            node,
        );
        state.maybe_cleanup();
        Ok(updated)
    }

    /// Records a device-tier stored block chain and reports whether the frontier advanced.
    pub fn update_session_from_stored_blocks(
        &self,
        session_id: &str,
        worker: WorkerWithDpRank,
        parent_hash: Option<ExternalSequenceBlockHash>,
        block_hashes: &[ExternalSequenceBlockHash],
    ) -> Result<bool, SessionPrefixIndexError> {
        self.update_session_from_stored_blocks_in_tier(
            session_id,
            SessionResidencyOwner::Worker(worker),
            StorageTier::Device,
            parent_hash,
            block_hashes,
        )
    }

    /// Records a block chain stored in one owner tier and reports whether that
    /// tier's frontier advanced.
    pub fn update_session_from_stored_blocks_in_tier(
        &self,
        session_id: &str,
        owner: SessionResidencyOwner,
        tier: StorageTier,
        parent_hash: Option<ExternalSequenceBlockHash>,
        block_hashes: &[ExternalSequenceBlockHash],
    ) -> Result<bool, SessionPrefixIndexError> {
        if block_hashes.is_empty() {
            return Ok(false);
        }
        if !valid_session_id(session_id) {
            return Ok(false);
        }

        let mut state = self.state.write();
        let updated = state.store_chain(session_id, (owner, tier), parent_hash, block_hashes)?;
        state.maybe_cleanup();
        Ok(updated)
    }

    /// Records single blocks that one owner tier stored for sessions, each beneath the
    /// parent the index already holds for it. Event sources without lineage, such as
    /// Mooncake object events, use this so a stored block never conflicts with the
    /// parent its device events recorded; a block not known yet becomes a root that a
    /// later device event grafts. Parents are resolved under the write lock, so a device
    /// store landing between lookup and update cannot turn into a conflict. Returns the
    /// number of frontiers advanced.
    pub fn update_sessions_from_stored_blocks_with_recorded_parents(
        &self,
        owner: SessionResidencyOwner,
        stored: &[(&str, StorageTier, ExternalSequenceBlockHash)],
    ) -> usize {
        if stored.is_empty() {
            return 0;
        }

        let mut state = self.state.write();
        let mut advanced = 0;
        for &(session_id, tier, block_hash) in stored {
            if !valid_session_id(session_id) {
                continue;
            }
            let parent_hash = state.recorded_parent_hash(block_hash);
            match state.store_chain(session_id, (owner, tier), parent_hash, &[block_hash]) {
                Ok(true) => advanced += 1,
                Ok(false) => {}
                Err(error) => {
                    tracing::warn!(%error, session_id, "failed to record session residency")
                }
            }
        }
        state.maybe_cleanup();
        advanced
    }

    /// Attributes blocks that an owner tier stored without a session to every session
    /// whose frontier in another tier of the same owner already covers them, such as
    /// device blocks offloaded to host memory or host blocks loaded back to the device.
    /// A session advances only to a stored block whose stored ancestors run back to the
    /// root or to a block the target tier already holds, so a tier frontier keeps
    /// implying that its ancestors are in the same tier. Returns the number of
    /// frontiers advanced.
    ///
    /// Inheritance reads the source tier's frontier when the destination tier's store
    /// arrives, so it assumes a worker publishes the destination `Stored` before the
    /// source `Removed` for the same blocks. SGLang's unified cache does. An engine
    /// that removes first leaves the session's destination-tier residency unrecorded
    /// until its next tagged store.
    pub fn inherit_stored_blocks(
        &self,
        owner: SessionResidencyOwner,
        tier: StorageTier,
        block_hashes: &[ExternalSequenceBlockHash],
    ) -> usize {
        if block_hashes.is_empty() {
            return 0;
        }

        let mut state = self.state.write();
        let updated = state.inherit_stored_blocks((owner, tier), block_hashes);
        state.maybe_cleanup();
        updated
    }

    /// Recedes worker-local device-tier session frontiers affected by removed blocks.
    pub fn update_session_from_removed_blocks(
        &self,
        worker: WorkerWithDpRank,
        block_hashes: &[ExternalSequenceBlockHash],
    ) -> usize {
        self.update_session_from_removed_blocks_in_tier(
            SessionResidencyOwner::Worker(worker),
            StorageTier::Device,
            block_hashes,
        )
    }

    /// Recedes session frontiers in one owner tier affected by removed blocks.
    pub fn update_session_from_removed_blocks_in_tier(
        &self,
        owner: SessionResidencyOwner,
        tier: StorageTier,
        block_hashes: &[ExternalSequenceBlockHash],
    ) -> usize {
        if block_hashes.is_empty() {
            return 0;
        }

        let mut state = self.state.write();
        let updated = state.recede_removed_frontiers((owner, tier), block_hashes);
        state.maybe_cleanup();
        updated
    }

    /// Removes every session frontier held by one worker rank, in every tier.
    pub fn clear_worker_frontiers(&self, worker: WorkerWithDpRank) -> usize {
        self.clear_owner_frontiers(SessionResidencyOwner::Worker(worker))
    }

    /// Removes every session frontier held by one owner, in every tier.
    pub fn clear_owner_frontiers(&self, owner: SessionResidencyOwner) -> usize {
        let mut state = self.state.write();
        let updated = state.clear_owner_frontiers(owner);
        state.maybe_cleanup();
        updated
    }

    /// Removes every session frontier associated with all ranks of one worker.
    pub fn clear_worker_id_frontiers(&self, worker_id: WorkerId) -> usize {
        let mut state = self.state.write();
        let updated = state.clear_worker_id_frontiers(worker_id);
        state.maybe_cleanup();
        updated
    }

    pub fn node_count(&self) -> usize {
        self.state.read().nodes.len()
    }

    pub fn session_count(&self) -> usize {
        self.state.read().session_to_worker_frontiers.len()
    }
}

fn valid_session_id(session_id: &str) -> bool {
    session_id.len() <= MAX_SESSION_PREFIX_ID_BYTES
}

impl IndexState {
    fn insert_node(
        &mut self,
        block_hash: ExternalSequenceBlockHash,
        parent: Option<NodeId>,
    ) -> NodeId {
        let node = self.nodes.insert(LogicalNode {
            block_hash,
            parent,
            frontier_refs: 0,
            child_count: 0,
        });
        self.hash_to_node.insert(block_hash, node);
        if let Some(parent) = parent {
            self.nodes[parent].child_count += 1;
            self.children.entry(parent).or_default().insert(node);
        }
        node
    }

    fn recorded_parent_hash(
        &self,
        block_hash: ExternalSequenceBlockHash,
    ) -> Option<ExternalSequenceBlockHash> {
        let node = *self.hash_to_node.get(&block_hash)?;
        let parent = self.nodes[node].parent?;
        Some(self.nodes[parent].block_hash)
    }

    // Validate, then attach the chain and advance the frontier of one owner tier.
    fn store_chain(
        &mut self,
        session_id: &str,
        key: FrontierKey,
        parent_hash: Option<ExternalSequenceBlockHash>,
        block_hashes: &[ExternalSequenceBlockHash],
    ) -> Result<bool, SessionPrefixIndexError> {
        // Reject the whole update before mutating the arena.
        self.validate_chain(parent_hash, block_hashes)?;

        let mut parent = parent_hash.map(|hash| self.resolve_or_insert_root(hash));
        for &block_hash in block_hashes {
            let node = match self.hash_to_node.get(&block_hash).copied() {
                Some(existing) => {
                    // Graft nodes previously known only as roots.
                    if let (None, Some(expected)) = (self.nodes[existing].parent, parent) {
                        self.nodes[existing].parent = Some(expected);
                        self.nodes[expected].child_count += 1;
                        self.children.entry(expected).or_default().insert(existing);
                    }
                    existing
                }
                None => self.insert_node(block_hash, parent),
            };
            parent = Some(node);
        }

        let leaf = parent.expect("non-empty block chain always yields a node");
        Ok(self.advance_frontier(session_id, key, leaf))
    }

    // Reject conflicting parents and graft cycles before mutation.
    fn validate_chain(
        &self,
        parent_hash: Option<ExternalSequenceBlockHash>,
        block_hashes: &[ExternalSequenceBlockHash],
    ) -> Result<(), SessionPrefixIndexError> {
        let mut chain_hashes =
            FxHashSet::with_capacity_and_hasher(block_hashes.len(), Default::default());
        let mut reconnects_unknown_parent = false;
        let mut needs_ancestry_check = false;
        let mut conflicting_block = None;
        let mut expected_parent = parent_hash;
        for &block_hash in block_hashes {
            if Some(block_hash) == parent_hash || !chain_hashes.insert(block_hash) {
                return Err(SessionPrefixIndexError::CyclicParent { block: block_hash });
            }
            if let Some(&node) = self.hash_to_node.get(&block_hash) {
                let entry = &self.nodes[node];
                match entry.parent {
                    Some(recorded_parent)
                        if Some(self.nodes[recorded_parent].block_hash) != expected_parent =>
                    {
                        conflicting_block.get_or_insert(block_hash);
                    }
                    None => {
                        reconnects_unknown_parent = true;
                        needs_ancestry_check |= expected_parent.is_some() && entry.child_count > 0;
                    }
                    Some(_) => {}
                }
            }
            expected_parent = Some(block_hash);
        }

        let parent_node = parent_hash.and_then(|hash| self.hash_to_node.get(&hash).copied());
        let unknown_parent =
            parent_hash.filter(|_| parent_node.is_none() && !reconnects_unknown_parent);

        // Only grafting an existing subtree can introduce a cycle. New blocks,
        // existing edges, and parentless leaves do not require an ancestry walk.
        if needs_ancestry_check && let Some(parent_node) = parent_node {
            let mut current = Some(parent_node);
            let mut steps = 0usize;
            while let Some(node) = current {
                let entry = &self.nodes[node];
                if chain_hashes.contains(&entry.block_hash) {
                    return Err(SessionPrefixIndexError::CyclicParent {
                        block: entry.block_hash,
                    });
                }
                steps += 1;
                if steps > self.nodes.len() {
                    debug_assert!(false, "parent cycle in session prefix index forest");
                    tracing::error!(
                        "session prefix index ancestry walk exceeded the arena; aborting"
                    );
                    break;
                }
                current = entry.parent;
            }
        }

        if let Some(block) = conflicting_block {
            return Err(SessionPrefixIndexError::ConflictingParent { block });
        }
        match unknown_parent {
            Some(parent) => Err(SessionPrefixIndexError::UnknownParent { parent }),
            None => Ok(()),
        }
    }

    fn touch_session(&mut self, session_id: &str) {
        let touch = self.next_touch;
        self.next_touch += 1;
        let Some(entry) = self.session_to_worker_frontiers.get_mut(session_id) else {
            return;
        };
        if let Some(previous) = entry.last_touch.replace(touch) {
            self.lru.remove(&previous);
        }
        self.lru.insert(touch, session_id.to_string());
    }

    fn enforce_session_cap(&mut self) {
        while self.session_to_worker_frontiers.len() > self.max_sessions {
            let Some((_, victim)) = self.lru.pop_first() else {
                debug_assert!(false, "session LRU is empty while over capacity");
                break;
            };
            self.drop_session_bindings(&victim);
            tracing::debug!(
                session_id = %victim,
                max_sessions = self.max_sessions,
                "session prefix index evicted its least recently used session"
            );
        }
    }

    fn drop_session_bindings(&mut self, session_id: &str) -> bool {
        let Some(entry) = self.session_to_worker_frontiers.remove(session_id) else {
            return false;
        };
        if let Some(last_touch) = entry.last_touch {
            self.lru.remove(&last_touch);
        }
        for (key, frontiers) in entry.worker_frontiers {
            for node in frontiers {
                self.nodes[node].frontier_refs -= 1;
                self.remove_reverse_frontier(key, node, session_id);
                self.queue_prune_candidate(node);
            }
        }
        true
    }

    fn queue_prune_candidate(&mut self, node: NodeId) {
        if self
            .nodes
            .get(node)
            .is_some_and(|entry| entry.frontier_refs == 0 && entry.child_count == 0)
        {
            self.prune_candidates.insert(node);
        }
    }

    fn maybe_cleanup(&mut self) {
        if self.last_cleanup.elapsed() < CLEANUP_INTERVAL {
            return;
        }
        self.prune_stale_nodes();
        self.last_cleanup = Instant::now();
    }

    fn prune_stale_nodes(&mut self) -> usize {
        let mut removed = 0;
        while let Some(node) = self.prune_candidates.iter().next().copied() {
            self.prune_candidates.remove(&node);
            let Some(entry) = self.nodes.get(node).copied() else {
                continue;
            };
            if entry.frontier_refs > 0 || entry.child_count > 0 {
                continue;
            }

            self.nodes.remove(node);
            self.hash_to_node.remove(&entry.block_hash);
            self.children.remove(&node);
            removed += 1;

            if let Some(parent) = entry.parent {
                if let Some(siblings) = self.children.get_mut(&parent) {
                    siblings.remove(&node);
                    if siblings.is_empty() {
                        self.children.remove(&parent);
                    }
                }
                let parent_entry = &mut self.nodes[parent];
                debug_assert!(parent_entry.child_count > 0);
                parent_entry.child_count -= 1;
                self.queue_prune_candidate(parent);
            }
        }
        removed
    }

    fn resolve_or_insert_root(&mut self, block_hash: ExternalSequenceBlockHash) -> NodeId {
        match self.hash_to_node.get(&block_hash).copied() {
            Some(node) => node,
            None => self.insert_node(block_hash, None),
        }
    }

    // Bound parent walks so corrupted cycles cannot hang while holding the lock.
    /// Number of nodes from `tail` to its root, inclusive (the same walk as `path_to_root`).
    fn depth(&self, tail: NodeId) -> u64 {
        let limit = self.nodes.len();
        let mut depth = 0usize;
        let mut current = Some(tail);
        while let Some(node) = current {
            if depth >= limit {
                debug_assert!(false, "parent cycle in session prefix index forest");
                break;
            }
            depth += 1;
            current = self.nodes.get(node).and_then(|n| n.parent);
        }
        depth as u64
    }

    fn path_to_root(&self, tail: NodeId) -> Vec<NodeId> {
        let limit = self.nodes.len();
        let mut path = Vec::new();
        let mut current = Some(tail);
        while let Some(node) = current {
            if path.len() >= limit {
                debug_assert!(false, "parent cycle in session prefix index forest");
                tracing::error!("session prefix index parent walk exceeded the arena; truncating");
                break;
            }
            path.push(node);
            current = self.nodes[node].parent;
        }
        path.reverse();
        path
    }

    fn is_ancestor_or_self(&self, candidate: NodeId, node: NodeId) -> bool {
        let limit = self.nodes.len();
        let mut current = Some(node);
        let mut steps = 0usize;
        while let Some(walk) = current {
            if walk == candidate {
                return true;
            }
            steps += 1;
            if steps > limit {
                debug_assert!(false, "parent cycle in session prefix index forest");
                tracing::error!("session prefix index ancestry walk exceeded the arena; aborting");
                return false;
            }
            current = self.nodes[walk].parent;
        }
        false
    }

    /// The members of `frontiers` that are ancestors of `node` or `node` itself.
    fn frontiers_on_root_path(&self, node: NodeId, frontiers: &[NodeId]) -> Vec<NodeId> {
        if frontiers.is_empty() {
            return Vec::new();
        }
        let wanted: FxHashSet<NodeId> = frontiers.iter().copied().collect();
        let limit = self.nodes.len();
        let mut found = Vec::new();
        let mut current = Some(node);
        let mut steps = 0usize;
        while let Some(walk) = current {
            if wanted.contains(&walk) {
                found.push(walk);
                if found.len() == wanted.len() {
                    break;
                }
            }
            steps += 1;
            if steps > limit {
                debug_assert!(false, "parent cycle in session prefix index forest");
                tracing::error!("session prefix index ancestry walk exceeded the arena; aborting");
                break;
            }
            current = self.nodes[walk].parent;
        }
        found
    }

    // Keep only the deepest frontier on each worker-tier chain.
    fn advance_frontier(&mut self, session_id: &str, key: FrontierKey, node: NodeId) -> bool {
        let frontiers: Vec<NodeId> = self
            .session_to_worker_frontiers
            .get(session_id)
            .and_then(|entry| entry.worker_frontiers.get(&key))
            .map(|frontiers| frontiers.iter().copied().collect())
            .unwrap_or_default();
        // A node with no children is no frontier's strict ancestor, so the ancestry walks run
        // only for a node that already has descendants (a store that re-reports a shorter
        // prefix). The common case, a store that extends the session, does no walk here.
        let already_reached = frontiers.contains(&node)
            || (self.nodes[node].child_count > 0
                && frontiers
                    .iter()
                    .any(|&frontier| self.is_ancestor_or_self(node, frontier)));
        if already_reached {
            self.touch_session(session_id);
            return false;
        }

        // Frontiers on the node's own root path: one walk up from the node, stopping once
        // every frontier is found (a store that appends k blocks finds its frontier in k steps),
        // instead of one walk to the root per frontier.
        let subsumed = self.frontiers_on_root_path(node, &frontiers);
        for &frontier in &subsumed {
            self.remove_frontier_binding(session_id, key, frontier);
        }
        self.add_frontier_binding(session_id, key, node);
        self.touch_session(session_id);
        self.enforce_session_cap();
        true
    }

    fn add_frontier_binding(&mut self, session_id: &str, key: FrontierKey, node: NodeId) {
        let inserted = self
            .session_to_worker_frontiers
            .entry(session_id.to_string())
            .or_default()
            .worker_frontiers
            .entry(key)
            .or_default()
            .insert(node);
        if !inserted {
            return;
        }

        self.nodes[node].frontier_refs += 1;
        self.worker_frontier_to_sessions
            .entry(key)
            .or_default()
            .entry(node)
            .or_default()
            .insert(session_id.to_string());
    }

    fn remove_frontier_binding(
        &mut self,
        session_id: &str,
        key: FrontierKey,
        node: NodeId,
    ) -> bool {
        let removed = self
            .session_to_worker_frontiers
            .get_mut(session_id)
            .and_then(|entry| entry.worker_frontiers.get_mut(&key))
            .is_some_and(|frontiers| frontiers.remove(&node));
        if !removed {
            return false;
        }

        self.nodes[node].frontier_refs -= 1;
        self.remove_reverse_frontier(key, node, session_id);
        self.queue_prune_candidate(node);

        let remove_key = self
            .session_to_worker_frontiers
            .get(session_id)
            .and_then(|entry| entry.worker_frontiers.get(&key))
            .is_some_and(FxHashSet::is_empty);
        if remove_key && let Some(entry) = self.session_to_worker_frontiers.get_mut(session_id) {
            entry.worker_frontiers.remove(&key);
        }
        let remove_session_entry = self
            .session_to_worker_frontiers
            .get(session_id)
            .is_some_and(|entry| entry.worker_frontiers.is_empty());
        if remove_session_entry
            && let Some(entry) = self.session_to_worker_frontiers.remove(session_id)
            && let Some(last_touch) = entry.last_touch
        {
            self.lru.remove(&last_touch);
        }
        true
    }

    fn remove_reverse_frontier(&mut self, key: FrontierKey, node: NodeId, session_id: &str) {
        let remove_node = self
            .worker_frontier_to_sessions
            .get_mut(&key)
            .and_then(|frontiers| frontiers.get_mut(&node))
            .is_some_and(|sessions| {
                sessions.remove(session_id);
                sessions.is_empty()
            });
        if remove_node && let Some(frontiers) = self.worker_frontier_to_sessions.get_mut(&key) {
            frontiers.remove(&node);
        }
        let remove_key = self
            .worker_frontier_to_sessions
            .get(&key)
            .is_some_and(FxHashMap::is_empty);
        if remove_key {
            self.worker_frontier_to_sessions.remove(&key);
        }
    }

    fn recede_removed_frontiers(
        &mut self,
        key: FrontierKey,
        block_hashes: &[ExternalSequenceBlockHash],
    ) -> usize {
        let removed_nodes: FxHashSet<NodeId> = block_hashes
            .iter()
            .filter_map(|block_hash| self.hash_to_node.get(block_hash).copied())
            .collect();
        if removed_nodes.is_empty() {
            return 0;
        }

        // Frontiers of this owner tier at or below a removed block: search down from the removed
        // blocks through `children` (a tail eviction touches only a few nodes), instead of walking
        // every frontier of the owner to its root. Each affected frontier then recedes past the
        // shallowest removed block on its own root path, as before.
        let Some(frontier_sessions) = self.worker_frontier_to_sessions.get(&key) else {
            return 0;
        };
        let mut seen: FxHashSet<NodeId> = FxHashSet::default();
        let mut stack: Vec<NodeId> = removed_nodes.iter().copied().collect();
        let mut hit: Vec<NodeId> = Vec::new();
        while let Some(node) = stack.pop() {
            if !seen.insert(node) {
                continue;
            }
            if frontier_sessions.contains_key(&node) {
                hit.push(node);
            }
            if let Some(children) = self.children.get(&node) {
                stack.extend(children.iter().copied());
            }
        }
        let affected: Vec<(SessionId, NodeId, Option<NodeId>)> = hit
            .into_iter()
            .filter_map(|frontier| {
                let path = self.path_to_root(frontier);
                let shallowest_removed =
                    path.into_iter().find(|node| removed_nodes.contains(node))?;
                let replacement = self.nodes[shallowest_removed].parent;
                let sessions = frontier_sessions.get(&frontier)?;
                Some(
                    sessions
                        .iter()
                        .cloned()
                        .map(move |session_id| (session_id, frontier, replacement))
                        .collect::<Vec<_>>(),
                )
            })
            .flatten()
            .collect();

        for (session_id, old_frontier, replacement) in &affected {
            self.remove_frontier_binding(session_id, key, *old_frontier);
            if let Some(replacement) = *replacement {
                self.advance_frontier(session_id, key, replacement);
            }
        }
        affected.len()
    }

    fn inherit_stored_blocks(
        &mut self,
        key: FrontierKey,
        block_hashes: &[ExternalSequenceBlockHash],
    ) -> usize {
        let stored_nodes: FxHashSet<NodeId> = block_hashes
            .iter()
            .filter_map(|block_hash| self.hash_to_node.get(block_hash).copied())
            .collect();
        if stored_nodes.is_empty() {
            return 0;
        }

        let (owner, tier) = key;
        let source_frontiers = self
            .worker_frontier_to_sessions
            .iter()
            .filter(|((frontier_owner, frontier_tier), _)| {
                *frontier_owner == owner && *frontier_tier != tier
            })
            .map(|(_, frontiers)| frontiers);

        // Each frontier in another tier of this owner passes on the deepest stored
        // block on its path.
        let candidates: Vec<(SessionId, NodeId)> = if self.stored_set_is_closed(&stored_nodes) {
            // No frontier lies strictly below the stored blocks, so only frontiers that
            // sit on a stored block can inherit: look them up instead of walking paths.
            source_frontiers
                .flat_map(|frontiers| {
                    stored_nodes.iter().filter_map(move |&node| {
                        frontiers.get(&node).map(|sessions| {
                            sessions
                                .iter()
                                .cloned()
                                .map(move |session_id| (session_id, node))
                        })
                    })
                })
                .flatten()
                .collect()
        } else {
            source_frontiers
                .flat_map(|frontiers| frontiers.iter())
                .filter_map(|(&frontier, sessions)| {
                    let deepest = self.deepest_stored_ancestor(frontier, &stored_nodes)?;
                    Some(
                        sessions
                            .iter()
                            .cloned()
                            .map(move |session_id| (session_id, deepest)),
                    )
                })
                .flatten()
                .collect()
        };

        let inherited: Vec<(SessionId, NodeId)> = candidates
            .into_iter()
            .filter_map(|(session_id, deepest)| {
                self.contiguous_inherited_frontier(&session_id, key, deepest, &stored_nodes)
                    .map(|node| (session_id, node))
            })
            .collect();

        inherited
            .into_iter()
            .filter(|(session_id, node)| self.advance_frontier(session_id, key, *node))
            .count()
    }

    // True when every child of a stored node is itself stored, so no frontier can lie
    // strictly below the stored set.
    fn stored_set_is_closed(&self, stored_nodes: &FxHashSet<NodeId>) -> bool {
        let mut stored_children: FxHashMap<NodeId, u32> = FxHashMap::default();
        for &node in stored_nodes {
            if let Some(parent) = self.nodes[node].parent
                && stored_nodes.contains(&parent)
            {
                *stored_children.entry(parent).or_default() += 1;
            }
        }
        stored_nodes.iter().all(|node| {
            self.nodes[*node].child_count == stored_children.get(node).copied().unwrap_or(0)
        })
    }

    // Deepest node on `tail`'s root path that is in `stored_nodes`, without
    // materializing the path.
    fn deepest_stored_ancestor(
        &self,
        tail: NodeId,
        stored_nodes: &FxHashSet<NodeId>,
    ) -> Option<NodeId> {
        let limit = self.nodes.len();
        let mut current = Some(tail);
        let mut steps = 0usize;
        while let Some(node) = current {
            if stored_nodes.contains(&node) {
                return Some(node);
            }
            steps += 1;
            if steps > limit {
                debug_assert!(false, "parent cycle in session prefix index forest");
                tracing::error!("session prefix index parent walk exceeded the arena; truncating");
                return None;
            }
            current = self.nodes[node].parent;
        }
        None
    }

    // From the deepest stored node on a session's source path, find the deepest stored
    // node whose run of stored ancestors reaches the root or a node the target tier
    // already holds.
    fn contiguous_inherited_frontier(
        &self,
        session_id: &str,
        key: FrontierKey,
        deepest: NodeId,
        stored_nodes: &FxHashSet<NodeId>,
    ) -> Option<NodeId> {
        let limit = self.nodes.len();
        let mut steps = 0usize;
        let mut run_bottom = deepest;
        loop {
            let mut run_top = run_bottom;
            while let Some(parent) = self.nodes[run_top].parent {
                steps += 1;
                if steps > limit {
                    debug_assert!(false, "parent cycle in session prefix index forest");
                    tracing::error!(
                        "session prefix index parent walk exceeded the arena; truncating"
                    );
                    return None;
                }
                if !stored_nodes.contains(&parent) {
                    break;
                }
                run_top = parent;
            }
            let Some(above) = self.nodes[run_top].parent else {
                return Some(run_bottom);
            };
            if self.tier_holds(session_id, key, above) {
                return Some(run_bottom);
            }
            run_bottom = self.deepest_stored_ancestor(above, stored_nodes)?;
        }
    }

    // True when the session's frontier in `key` is at or below `node`.
    fn tier_holds(&self, session_id: &str, key: FrontierKey, node: NodeId) -> bool {
        self.session_to_worker_frontiers
            .get(session_id)
            .and_then(|entry| entry.worker_frontiers.get(&key))
            .is_some_and(|frontiers| {
                frontiers
                    .iter()
                    .any(|&frontier| self.is_ancestor_or_self(node, frontier))
            })
    }

    fn clear_frontiers(&mut self, key: FrontierKey) -> usize {
        let affected: Vec<(SessionId, NodeId)> = self
            .worker_frontier_to_sessions
            .get(&key)
            .map(|frontiers| {
                frontiers
                    .iter()
                    .flat_map(|(&node, sessions)| {
                        sessions
                            .iter()
                            .cloned()
                            .map(move |session_id| (session_id, node))
                    })
                    .collect()
            })
            .unwrap_or_default();
        for (session_id, node) in &affected {
            self.remove_frontier_binding(session_id, key, *node);
        }
        affected.len()
    }

    fn clear_matching_frontiers(
        &mut self,
        matches: impl Fn(SessionResidencyOwner) -> bool,
    ) -> usize {
        let keys: Vec<_> = self
            .worker_frontier_to_sessions
            .keys()
            .copied()
            .filter(|&(owner, _)| matches(owner))
            .collect();
        keys.into_iter().map(|key| self.clear_frontiers(key)).sum()
    }

    fn clear_owner_frontiers(&mut self, owner: SessionResidencyOwner) -> usize {
        self.clear_matching_frontiers(|frontier_owner| frontier_owner == owner)
    }

    fn clear_worker_id_frontiers(&mut self, worker_id: WorkerId) -> usize {
        self.clear_matching_frontiers(|frontier_owner| {
            frontier_owner
                .as_worker()
                .is_some_and(|worker| worker.worker_id == worker_id)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::make_blocks;

    fn hashes(ids: Vec<u64>) -> Vec<ExternalSequenceBlockHash> {
        make_blocks(ids)
            .into_iter()
            .map(|block| block.block_hash)
            .collect()
    }

    fn worker(worker_id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(worker_id, 0)
    }

    fn owner(worker_id: u64) -> SessionResidencyOwner {
        SessionResidencyOwner::Worker(worker(worker_id))
    }

    fn lineage_of(
        indexer: &SessionPrefixIndexer,
        session: &str,
    ) -> Vec<Vec<ExternalSequenceBlockHash>> {
        lineage_on_worker(indexer, session, worker(1))
    }

    fn lineage_on_worker(
        indexer: &SessionPrefixIndexer,
        session: &str,
        worker: WorkerWithDpRank,
    ) -> Vec<Vec<ExternalSequenceBlockHash>> {
        let mut lineages = indexer
            .get_session_block_lineage(session, worker, None)
            .expect("lineage query without an anchor cannot fail");
        lineages.sort();
        lineages
    }

    #[test]
    fn match_creates_lineage_and_repeat_is_idempotent() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();

        assert!(
            indexer
                .update_session_from_match("s1", worker(1), chain[0])
                .unwrap(),
            "first match must advance the frontier"
        );
        assert_eq!(indexer.node_count(), 1);
        assert_eq!(lineage_of(&indexer, "s1"), vec![vec![chain[0]]]);

        assert!(
            !indexer
                .update_session_from_match("s1", worker(1), chain[0])
                .unwrap(),
            "re-matching the same block is not an advance"
        );
        assert_eq!(
            indexer.node_count(),
            1,
            "a repeated match must not grow the arena"
        );
        assert_eq!(indexer.get_session_frontiers("s1").len(), 1);
    }

    #[test]
    fn deeper_match_replaces_the_shallower_frontier() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &chain)
            .unwrap();
        assert_eq!(indexer.get_session_frontiers("s1").len(), 1);

        assert!(
            !indexer
                .update_session_from_match("s1", worker(1), chain[1])
                .unwrap(),
            "a match above the current frontier must not move it"
        );
        assert_eq!(lineage_of(&indexer, "s1"), vec![chain.clone()]);

        assert!(
            indexer
                .update_session_from_match("s2", worker(1), chain[1])
                .unwrap()
        );
        assert_eq!(lineage_of(&indexer, "s2"), vec![chain[..2].to_vec()]);
    }

    #[test]
    fn stored_blocks_reuse_shared_prefix_nodes() {
        let trunk = hashes(vec![1, 2]);
        let branch = hashes(vec![3]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &trunk)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s2", worker(1), None, &trunk)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s2", worker(1), Some(trunk[1]), &branch)
            .unwrap();

        assert_eq!(
            indexer.node_count(),
            3,
            "the shared trunk must be stored once, not per session"
        );
        assert_eq!(lineage_of(&indexer, "s1"), vec![trunk.clone()]);
        assert_eq!(
            lineage_of(&indexer, "s2"),
            vec![vec![trunk[0], trunk[1], branch[0]]]
        );
    }

    #[test]
    fn stored_blocks_graft_onto_a_node_first_seen_as_a_match() {
        let chain = hashes(vec![1, 2]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_match("s1", worker(1), chain[1])
            .unwrap();
        let child = indexer.get_node_from_hash(chain[1]).unwrap();
        assert_eq!(indexer.get_node(child).unwrap().parent(), None);

        indexer
            .update_session_from_stored_blocks("s1", worker(1), Some(chain[0]), &chain[1..])
            .unwrap();

        assert_eq!(
            indexer.get_node_from_hash(chain[1]),
            Some(child),
            "grafting must keep the original node handle valid"
        );
        assert_eq!(
            indexer.node_count(),
            2,
            "grafting must not leave a duplicate root behind"
        );
        assert_eq!(lineage_of(&indexer, "s1"), vec![chain]);
    }

    #[test]
    fn wholly_unknown_parent_is_rejected() {
        let chain = hashes(vec![1, 2]);
        let indexer = SessionPrefixIndexer::new();

        let err = indexer
            .update_session_from_stored_blocks("s1", worker(1), Some(chain[0]), &chain[1..])
            .expect_err("an unknown parent cannot introduce wholly new lineage");

        assert_eq!(
            err,
            SessionPrefixIndexError::UnknownParent { parent: chain[0] }
        );
        assert_eq!(indexer.node_count(), 0);
        assert_eq!(indexer.session_count(), 0);
    }

    #[test]
    fn conflicting_parent_is_rejected() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &chain[..2])
            .unwrap();

        let err = indexer
            .update_session_from_stored_blocks("s1", worker(1), Some(chain[2]), &chain[1..2])
            .expect_err("re-parenting a known block violates the hash invariant");
        assert_eq!(
            err,
            SessionPrefixIndexError::ConflictingParent { block: chain[1] }
        );
    }

    #[test]
    fn branching_session_keeps_one_frontier_per_chain() {
        let trunk = hashes(vec![1]);
        let left = hashes(vec![2]);
        let right = hashes(vec![3]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &trunk)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s1", worker(1), Some(trunk[0]), &left)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s1", worker(1), Some(trunk[0]), &right)
            .unwrap();

        assert_eq!(
            indexer.get_session_frontiers("s1").len(),
            2,
            "each branch keeps its own frontier"
        );
        assert_eq!(
            lineage_of(&indexer, "s1"),
            vec![vec![trunk[0], left[0]], vec![trunk[0], right[0]]]
        );
    }

    #[test]
    fn lineage_anchor_truncates_and_filters() {
        let trunk = hashes(vec![1, 2]);
        let left = hashes(vec![3]);
        let right = hashes(vec![4]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &trunk)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s1", worker(1), Some(trunk[1]), &left)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &right)
            .unwrap();

        let anchored = indexer
            .get_session_block_lineage("s1", worker(1), Some(trunk[1]))
            .unwrap();
        assert_eq!(
            anchored,
            vec![vec![trunk[1], left[0]]],
            "anchoring drops chains that miss the anchor and trims the rest"
        );
    }

    #[test]
    fn unknown_anchor_is_an_error_and_unknown_session_is_empty() {
        let chain = hashes(vec![1]);
        let missing = hashes(vec![99]);
        let indexer = SessionPrefixIndexer::new();
        indexer
            .update_session_from_match("s1", worker(1), chain[0])
            .unwrap();

        assert_eq!(
            indexer.get_session_block_lineage("s1", worker(1), Some(missing[0])),
            Err(SessionPrefixIndexError::UnknownAnchor(missing[0]))
        );
        assert_eq!(
            indexer.get_session_block_lineage("unrouted", worker(1), None),
            Ok(Vec::new()),
            "a session with no routed requests is empty, not an error"
        );
    }

    #[test]
    fn eviction_shaped_replay_does_not_lose_lineage() {
        let chain = hashes(vec![1, 2]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &chain)
            .unwrap();
        let before = lineage_of(&indexer, "s1");

        indexer
            .update_session_from_match("s1", worker(1), chain[0])
            .unwrap();

        assert_eq!(
            lineage_of(&indexer, "s1"),
            before,
            "re-matching a shallow block must not truncate the recorded lineage"
        );
        assert_eq!(indexer.node_count(), 2);
    }

    #[test]
    fn session_cap_evicts_the_least_recently_used_bindings() {
        let chain = hashes(vec![1]);
        let indexer = SessionPrefixIndexer::with_max_sessions(2);

        indexer
            .update_session_from_match("s1", worker(1), chain[0])
            .unwrap();
        indexer
            .update_session_from_match("s2", worker(1), chain[0])
            .unwrap();
        indexer
            .update_session_from_match("s1", worker(1), chain[0])
            .unwrap();
        indexer
            .update_session_from_match("s3", worker(1), chain[0])
            .unwrap();

        assert_eq!(indexer.session_count(), 2);
        assert!(!indexer.get_session_frontiers("s1").is_empty());
        assert!(indexer.get_session_frontiers("s2").is_empty());
        assert!(!indexer.get_session_frontiers("s3").is_empty());
        assert_eq!(
            indexer.node_count(),
            1,
            "evicting one session must preserve topology shared by survivors"
        );
        let state = indexer.state.read();
        let node = state.hash_to_node[&chain[0]];
        let sessions = &state.worker_frontier_to_sessions[&(owner(1), StorageTier::Device)][&node];
        assert_eq!(sessions.len(), 2);
        assert!(!sessions.contains("s2"));
    }

    #[test]
    fn oversized_session_ids_do_not_mutate_the_index() {
        let chain = hashes(vec![1, 2]);
        let indexer = SessionPrefixIndexer::new();
        let valid_session = "s".repeat(MAX_SESSION_PREFIX_ID_BYTES);
        let oversized_session = "s".repeat(MAX_SESSION_PREFIX_ID_BYTES + 1);

        assert!(
            indexer
                .update_session_from_match(&valid_session, worker(1), chain[0])
                .unwrap()
        );

        assert_eq!(
            indexer.update_session_from_match(&oversized_session, worker(1), chain[1]),
            Ok(false)
        );
        assert_eq!(
            indexer.update_session_from_stored_blocks(
                &oversized_session,
                worker(1),
                Some(chain[0]),
                &chain[1..],
            ),
            Ok(false)
        );
        assert_eq!(indexer.node_count(), 1);
        assert_eq!(indexer.session_count(), 1);
    }

    #[test]
    fn removal_recedes_only_the_affected_worker_frontier() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();

        for target in [worker(1), worker(2)] {
            indexer
                .update_session_from_stored_blocks("s1", target, None, &chain)
                .unwrap();
        }

        assert_eq!(
            indexer.update_session_from_removed_blocks(worker(2), &chain[1..]),
            1
        );
        assert_eq!(
            lineage_on_worker(&indexer, "s1", worker(1)),
            vec![chain.clone()]
        );
        assert_eq!(
            lineage_on_worker(&indexer, "s1", worker(2)),
            vec![vec![chain[0]]]
        );
    }

    #[test]
    fn batched_removal_recedes_all_sessions_and_is_idempotent() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();

        for session in ["s1", "s2"] {
            indexer
                .update_session_from_stored_blocks(session, worker(1), None, &chain)
                .unwrap();
        }

        assert_eq!(
            indexer.update_session_from_removed_blocks(worker(1), &[chain[2], chain[1]]),
            2
        );
        for session in ["s1", "s2"] {
            assert_eq!(lineage_of(&indexer, session), vec![vec![chain[0]]]);
        }
        assert_eq!(
            indexer.update_session_from_removed_blocks(worker(1), &[chain[1], chain[2]]),
            0,
            "replaying the same removal must not recede the frontier again"
        );
    }

    #[test]
    fn removal_ignores_unknown_hashes_and_recedes_known_frontiers() {
        let chain = hashes(vec![1, 2, 3]);
        let unknown = hashes(vec![99])[0];
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &chain)
            .unwrap();

        assert_eq!(
            indexer.update_session_from_removed_blocks(worker(1), &[unknown, chain[1]]),
            1
        );
        assert_eq!(lineage_of(&indexer, "s1"), vec![vec![chain[0]]]);
    }

    #[test]
    fn interior_removal_recedes_before_later_frontier_removal() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &chain)
            .unwrap();

        assert_eq!(
            indexer.update_session_from_removed_blocks(worker(1), &[chain[1]]),
            1
        );
        assert_eq!(lineage_of(&indexer, "s1"), vec![vec![chain[0]]]);

        assert_eq!(
            indexer.update_session_from_removed_blocks(worker(1), &[chain[2]]),
            0,
            "removing the old tail must not restore its already-removed parent"
        );
        assert_eq!(lineage_of(&indexer, "s1"), vec![vec![chain[0]]]);
    }

    #[test]
    fn clear_removes_only_one_workers_frontiers() {
        let chain = hashes(vec![1, 2]);
        let indexer = SessionPrefixIndexer::new();

        for target in [worker(1), worker(2)] {
            indexer
                .update_session_from_stored_blocks("s1", target, None, &chain)
                .unwrap();
        }

        assert_eq!(indexer.clear_owner_frontiers(owner(2)), 1);
        assert!(lineage_on_worker(&indexer, "s1", worker(2)).is_empty());
        assert_eq!(lineage_on_worker(&indexer, "s1", worker(1)), vec![chain]);
        assert_eq!(indexer.node_count(), 2, "clear preserves logical topology");
    }

    #[test]
    fn worker_id_clear_removes_all_ranks_only_for_that_worker() {
        let chain = hashes(vec![1, 2]);
        let indexer = SessionPrefixIndexer::new();
        let first_rank = WorkerWithDpRank::new(1, 0);
        let second_rank = WorkerWithDpRank::new(1, 1);
        let retained_worker = WorkerWithDpRank::new(2, 0);

        for target in [first_rank, second_rank, retained_worker] {
            indexer
                .update_session_from_stored_blocks("s1", target, None, &chain)
                .unwrap();
        }

        assert_eq!(indexer.clear_worker_id_frontiers(1), 2);
        assert!(lineage_on_worker(&indexer, "s1", first_rank).is_empty());
        assert!(lineage_on_worker(&indexer, "s1", second_rank).is_empty());
        assert_eq!(
            lineage_on_worker(&indexer, "s1", retained_worker),
            vec![chain]
        );
    }

    fn residency_of(
        indexer: &SessionPrefixIndexer,
        session: &str,
    ) -> Vec<(
        SessionResidencyOwner,
        StorageTier,
        ExternalSequenceBlockHash,
    )> {
        indexer
            .get_session_residency(session)
            .into_iter()
            .map(|(owner, tier, node)| (owner, tier, indexer.get_node(node).unwrap().block_hash()))
            .collect()
    }

    #[test]
    fn offloaded_blocks_keep_session_residency_after_device_eviction() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();
        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &chain)
            .unwrap();

        // The device offloads its first two blocks to host memory, then evicts the last two.
        assert_eq!(
            indexer.inherit_stored_blocks(owner(1), StorageTier::HostPinned, &chain[..2]),
            1
        );
        indexer.update_session_from_removed_blocks(worker(1), &chain[1..]);

        assert_eq!(lineage_of(&indexer, "s1"), vec![vec![chain[0]]]);
        assert_eq!(
            indexer
                .get_session_block_lineage_in_tier("s1", owner(1), StorageTier::HostPinned, None)
                .unwrap(),
            vec![chain[..2].to_vec()]
        );
        let residency = residency_of(&indexer, "s1");
        assert_eq!(residency.len(), 2);
        assert!(residency.contains(&(owner(1), StorageTier::Device, chain[0])));
        assert!(residency.contains(&(owner(1), StorageTier::HostPinned, chain[1])));
        assert_eq!(
            indexer.get_session_frontiers("s1").len(),
            1,
            "device frontier queries ignore lower tiers"
        );
    }

    #[test]
    fn lower_tiers_inherit_only_within_a_worker_and_follow_tier_hops() {
        let chain = hashes(vec![1, 2]);
        let indexer = SessionPrefixIndexer::new();
        indexer
            .update_session_from_stored_blocks("s1", worker(2), None, &chain)
            .unwrap();
        assert_eq!(
            indexer.inherit_stored_blocks(owner(1), StorageTier::HostPinned, &chain),
            0,
            "another worker's frontier does not attribute blocks"
        );

        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &chain)
            .unwrap();
        assert_eq!(
            indexer.inherit_stored_blocks(owner(1), StorageTier::HostPinned, &chain),
            1
        );
        indexer.update_session_from_removed_blocks(worker(1), &chain);
        assert_eq!(
            indexer.inherit_stored_blocks(owner(1), StorageTier::Disk, &chain[..1]),
            1,
            "disk inherits from host after the device copy is gone"
        );
        assert_eq!(
            indexer.update_session_from_removed_blocks_in_tier(
                owner(1),
                StorageTier::HostPinned,
                &chain[1..]
            ),
            1
        );

        let residency = residency_of(&indexer, "s1");
        assert_eq!(residency.len(), 3);
        assert!(residency.contains(&(owner(1), StorageTier::HostPinned, chain[0])));
        assert!(residency.contains(&(owner(1), StorageTier::Disk, chain[0])));
        assert!(residency.contains(&(owner(2), StorageTier::Device, chain[1])));

        assert_eq!(
            indexer.clear_owner_frontiers(owner(1)),
            2,
            "a rank clear covers every tier"
        );
        assert_eq!(
            residency_of(&indexer, "s1"),
            vec![(owner(2), StorageTier::Device, chain[1])]
        );
    }

    #[test]
    fn shared_cache_frontiers_are_owned_apart_from_workers() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();
        let mooncake = SessionResidencyOwner::SharedCache(SharedCacheBackend::Mooncake);

        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &chain)
            .unwrap();
        assert_eq!(
            indexer.inherit_stored_blocks(mooncake, StorageTier::HostPinned, &chain),
            0,
            "a shared store does not inherit a worker's frontiers"
        );

        indexer
            .update_session_from_stored_blocks_in_tier(
                "s1",
                mooncake,
                StorageTier::HostPinned,
                None,
                &chain[..2],
            )
            .unwrap();
        assert_eq!(
            indexer.inherit_stored_blocks(owner(2), StorageTier::HostPinned, &chain),
            0,
            "a worker does not inherit the shared store's frontiers"
        );
        assert_eq!(
            indexer.get_session_frontiers("s1").len(),
            1,
            "device frontier queries report workers only"
        );

        assert_eq!(
            indexer.clear_worker_id_frontiers(1),
            1,
            "a worker clear leaves the shared store alone"
        );
        assert_eq!(
            residency_of(&indexer, "s1"),
            vec![(mooncake, StorageTier::HostPinned, chain[1])]
        );
        assert_eq!(indexer.clear_owner_frontiers(mooncake), 1);
        assert!(residency_of(&indexer, "s1").is_empty());
    }

    #[test]
    fn opportunistic_cleanup_reclaims_unreferenced_tail() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &chain)
            .unwrap();
        indexer.state.write().last_cleanup = Instant::now() - CLEANUP_INTERVAL;

        assert_eq!(
            indexer.update_session_from_removed_blocks(worker(1), &chain[1..]),
            1
        );
        assert_eq!(lineage_of(&indexer, "s1"), vec![vec![chain[0]]]);
        assert_eq!(indexer.node_count(), 1);
        assert!(indexer.get_node_from_hash(chain[0]).is_some());
        assert!(indexer.get_node_from_hash(chain[1]).is_none());
        assert!(indexer.get_node_from_hash(chain[2]).is_none());
    }

    #[test]
    fn a_chain_that_would_close_a_cycle_is_rejected() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &chain)
            .unwrap();

        let err = indexer
            .update_session_from_stored_blocks("s1", worker(1), Some(chain[2]), &chain[..1])
            .expect_err("grafting an ancestor under its own descendant must fail");
        assert!(
            matches!(
                err,
                SessionPrefixIndexError::CyclicParent { block } if block == chain[0]
            ),
            "expected CyclicParent for the offending block, got {err:?}"
        );

        assert_eq!(
            lineage_of(&indexer, "s1"),
            vec![chain.clone()],
            "a rejected chain must not half-apply"
        );
    }

    #[test]
    fn a_block_repeated_within_one_chain_is_rejected() {
        let chain = hashes(vec![1, 2]);
        let indexer = SessionPrefixIndexer::new();

        let repeating = vec![chain[0], chain[1], chain[0]];
        let err = indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &repeating)
            .expect_err("a chain that revisits its own block must fail");
        assert!(
            matches!(
                err,
                SessionPrefixIndexError::CyclicParent { block } if block == chain[0]
            ),
            "expected CyclicParent for the repeated block, got {err:?}"
        );
        assert!(
            lineage_of(&indexer, "s1").is_empty(),
            "a rejected chain must not create any nodes"
        );
    }

    #[test]
    fn inheritance_requires_contiguous_blocks_from_root_or_tier_frontier() {
        let chain = hashes(vec![1, 2, 3, 4]);
        let indexer = SessionPrefixIndexer::new();
        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &chain)
            .unwrap();
        let host = || {
            indexer
                .get_session_block_lineage_in_tier("s1", owner(1), StorageTier::HostPinned, None)
                .unwrap()
        };
        let inherit = |blocks: &[ExternalSequenceBlockHash]| {
            indexer.inherit_stored_blocks(owner(1), StorageTier::HostPinned, blocks)
        };

        // Blocks 1 and 3 without 2: only the run from the root counts.
        assert_eq!(inherit(&[chain[0], chain[2]]), 1);
        assert_eq!(host(), vec![chain[..1].to_vec()]);
        // Block 4 without 3 does not extend a frontier that stops at 1.
        assert_eq!(inherit(&chain[3..]), 0);
        assert_eq!(host(), vec![chain[..1].to_vec()]);
        // A later chunk continues from the host frontier.
        assert_eq!(inherit(&chain[1..3]), 1);
        assert_eq!(host(), vec![chain[..3].to_vec()]);
        assert_eq!(inherit(&chain[3..]), 1);
        assert_eq!(host(), vec![chain.clone()]);
    }

    #[test]
    fn shared_prefix_sessions_inherit_from_deeper_and_exact_frontiers() {
        let short = hashes(vec![1, 2, 3]);
        let long = hashes(vec![1, 2, 3, 4]);
        let indexer = SessionPrefixIndexer::new();
        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &short)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s2", worker(1), None, &long)
            .unwrap();

        // The stored set has a child outside it, so frontiers below it are found by
        // walking: s2's frontier at 4 inherits 3.
        assert_eq!(
            indexer.inherit_stored_blocks(owner(1), StorageTier::HostPinned, &short),
            2
        );
        // The full chain is closed: both frontiers sit on stored blocks.
        assert_eq!(
            indexer.inherit_stored_blocks(owner(1), StorageTier::Disk, &long),
            2
        );
        for (session, chain) in [("s1", &short), ("s2", &long)] {
            assert_eq!(
                indexer
                    .get_session_block_lineage_in_tier(
                        session,
                        owner(1),
                        StorageTier::HostPinned,
                        None
                    )
                    .unwrap(),
                vec![short.clone()]
            );
            assert_eq!(
                indexer
                    .get_session_block_lineage_in_tier(session, owner(1), StorageTier::Disk, None)
                    .unwrap(),
                vec![chain.clone()]
            );
        }
    }

    #[test]
    fn recorded_parent_store_reuses_device_lineage_and_roots_unknown_blocks() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();
        let shared = SessionResidencyOwner::SharedCache(SharedCacheBackend::Mooncake);
        indexer
            .update_session_from_stored_blocks("s", worker(1), None, &chain)
            .unwrap();

        // Known blocks attach beneath their recorded parent, so the store cannot
        // conflict with the device chain; the deepest one becomes the frontier.
        let advanced = indexer.update_sessions_from_stored_blocks_with_recorded_parents(
            shared,
            &[
                ("s", StorageTier::HostPinned, chain[1]),
                ("s", StorageTier::HostPinned, chain[2]),
            ],
        );
        assert_eq!(advanced, 2);
        let shared_frontiers: Vec<_> = indexer
            .get_session_residency("s")
            .into_iter()
            .filter(|(owner, _, _)| *owner == shared)
            .map(|(_, tier, node)| (tier, indexer.get_node(node).unwrap().block_hash()))
            .collect();
        assert_eq!(shared_frontiers, vec![(StorageTier::HostPinned, chain[2])]);
        assert_eq!(lineage_of(&indexer, "s"), vec![chain.clone()]);

        // An unknown block becomes a root that a later device chain grafts.
        let unknown = hashes(vec![9])[0];
        assert_eq!(
            indexer.update_sessions_from_stored_blocks_with_recorded_parents(
                shared,
                &[("t", StorageTier::Disk, unknown)],
            ),
            1
        );
        indexer
            .update_session_from_stored_blocks("t", worker(1), Some(chain[2]), &[unknown])
            .unwrap();
        let mut extended = chain.clone();
        extended.push(unknown);
        assert_eq!(lineage_on_worker(&indexer, "t", worker(1)), vec![extended]);
    }

    #[test]
    fn resident_depths_report_the_deepest_prefix_per_owner_and_tier() {
        let indexer = SessionPrefixIndexer::new();
        let chain = hashes(vec![1, 2, 3, 4]);
        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &chain)
            .unwrap();
        // A shorter copy of the same prefix on a second worker, and two blocks in host memory.
        indexer
            .update_session_from_stored_blocks("s1", worker(2), None, &chain[..2])
            .unwrap();
        indexer
            .update_session_from_stored_blocks_in_tier(
                "s1",
                owner(1),
                StorageTier::HostPinned,
                None,
                &chain[..2],
            )
            .unwrap();

        let mut depths = indexer.get_session_resident_depths("s1");
        depths
            .sort_by_key(|&(owner, tier, _)| (owner.as_worker().map(|w| w.worker_id), tier as u8));
        assert_eq!(
            depths,
            vec![
                (owner(1), StorageTier::Device, 4),
                (owner(1), StorageTier::HostPinned, 2),
                (owner(2), StorageTier::Device, 2),
            ]
        );
        assert!(indexer.get_session_resident_depths("unknown").is_empty());
    }

    #[test]
    fn resident_depths_take_the_deeper_of_two_branches() {
        let indexer = SessionPrefixIndexer::new();
        let trunk = hashes(vec![1, 2]);
        let left = hashes(vec![3]);
        let right = hashes(vec![4, 5, 6]);
        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &trunk)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s1", worker(1), Some(trunk[1]), &left)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s1", worker(1), Some(trunk[1]), &right)
            .unwrap();
        assert_eq!(
            indexer.get_session_resident_depths("s1"),
            vec![(owner(1), StorageTier::Device, 5)]
        );
    }

    #[test]
    fn residency_totals_sum_sessions_per_tier() {
        let indexer = SessionPrefixIndexer::new();
        indexer
            .update_session_from_stored_blocks("a", worker(1), None, &hashes(vec![1, 2, 3]))
            .unwrap();
        indexer
            .update_session_from_stored_blocks("b", worker(2), None, &hashes(vec![7, 8]))
            .unwrap();
        indexer
            .update_session_from_stored_blocks_in_tier(
                "b",
                owner(2),
                StorageTier::Disk,
                None,
                &hashes(vec![7]),
            )
            .unwrap();
        let (sessions, mut per_tier) = indexer.residency_totals();
        per_tier.sort_by_key(|&(tier, _)| tier as u8);
        assert_eq!(sessions, 2);
        assert_eq!(
            per_tier,
            vec![(StorageTier::Device, 5), (StorageTier::Disk, 1)]
        );
    }

    #[test]
    fn unique_residency_counts_a_shared_prefix_once_per_owner_and_tier() {
        let indexer = SessionPrefixIndexer::new();
        // Sessions a and b share blocks 1 and 2 on worker 1; b continues with 4.
        indexer
            .update_session_from_stored_blocks("a", worker(1), None, &hashes(vec![1, 2, 3]))
            .unwrap();
        indexer
            .update_session_from_stored_blocks("b", worker(1), None, &hashes(vec![1, 2, 4]))
            .unwrap();
        // Session a's prefix is also on worker 2, and one block of it in worker 1's host tier.
        indexer
            .update_session_from_stored_blocks("a", worker(2), None, &hashes(vec![1, 2]))
            .unwrap();
        indexer
            .update_session_from_stored_blocks_in_tier(
                "a",
                owner(1),
                StorageTier::HostPinned,
                None,
                &hashes(vec![1]),
            )
            .unwrap();
        let unique = indexer.unique_residency();
        let get = |o: SessionResidencyOwner, t: StorageTier| {
            unique
                .iter()
                .find(|&&(oo, tt, _)| oo == o && tt == t)
                .map(|&(_, _, n)| n)
        };
        // Worker 1 device: blocks 1, 2, 3, 4 once each, where the per-session sum counts 6.
        assert_eq!(get(owner(1), StorageTier::Device), Some(4));
        assert_eq!(get(owner(2), StorageTier::Device), Some(2));
        assert_eq!(get(owner(1), StorageTier::HostPinned), Some(1));
        let (_, per_tier) = indexer.residency_totals();
        let summed: u64 = per_tier
            .iter()
            .filter(|&&(t, _)| t == StorageTier::Device)
            .map(|&(_, n)| n)
            .sum();
        assert_eq!(summed, 3 + 3 + 2);
    }

    fn assert_children_match_child_count(indexer: &SessionPrefixIndexer) {
        let state = indexer.state.read();
        for (node, entry) in state.nodes.iter() {
            let listed = state.children.get(&node).map_or(0, |c| c.len());
            assert_eq!(listed as u32, entry.child_count, "children of {node:?}");
            for &child in state.children.get(&node).into_iter().flatten() {
                assert_eq!(state.nodes[child].parent, Some(node));
            }
        }
        assert!(
            state
                .children
                .keys()
                .all(|node| state.nodes.contains_key(*node))
        );
    }

    #[test]
    fn removals_recede_only_frontiers_below_a_removed_block_and_keep_child_links() {
        let indexer = SessionPrefixIndexer::new();
        let trunk = hashes(vec![1, 2, 3]);
        let left = hashes(vec![4, 5]);
        let right = hashes(vec![6]);
        indexer
            .update_session_from_stored_blocks("a", worker(1), None, &trunk)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("a", worker(1), Some(trunk[2]), &left)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("b", worker(1), Some(trunk[2]), &right)
            .unwrap();
        // A former root grafted under a parent keeps the child links too.
        let late = hashes(vec![9]);
        indexer
            .update_session_from_stored_blocks("c", worker(1), None, &late)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("c", worker(1), Some(right[0]), &late)
            .unwrap();
        assert_children_match_child_count(&indexer);

        // Removing the left tail recedes only session a, to the left branch's first block.
        assert_eq!(
            indexer.update_session_from_removed_blocks(worker(1), &[left[1]]),
            1
        );
        assert_eq!(
            lineage_of(&indexer, "a"),
            vec![vec![trunk[0], trunk[1], trunk[2], left[0]]]
        );
        assert_eq!(
            lineage_of(&indexer, "b"),
            vec![vec![trunk[0], trunk[1], trunk[2], right[0]]]
        );

        // Removing a trunk block recedes every frontier below it, past the shallowest removal.
        assert_eq!(
            indexer.update_session_from_removed_blocks(worker(1), &[trunk[2], right[0]]),
            3
        );
        for session in ["a", "b", "c"] {
            assert_eq!(
                lineage_of(&indexer, session),
                vec![vec![trunk[0], trunk[1]]]
            );
        }
        let pruned = indexer.state.write().prune_stale_nodes();
        assert!(pruned > 0);
        assert_children_match_child_count(&indexer);
    }
}
