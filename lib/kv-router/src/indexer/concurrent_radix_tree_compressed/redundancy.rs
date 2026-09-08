// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fleet-wide KV-cache redundancy: how much of the router's live index is
//! genuinely unique content versus copies of the same block sitting on more
//! than one worker (e.g. a session's prefix re-materializing on a second
//! worker after a router hop, or cache-aware routing simply missing).
//!
//! Diagnostic-only. This module has NOT been through the benchmark pass or
//! PeaBrane's review this crate's AGENTS.md requires for hot-path lock/
//! traversal changes - it is deliberately kept off any per-request path
//! (only ever called from a slow background sampler, see
//! `lib/llm/src/kv_router/indexer/mod.rs`) and does not touch locking,
//! versioning, or traversal semantics anywhere else in this module. Treat
//! it as bench/diagnostic-scoped, not cleared for a production-scale
//! deployment, until that review happens.

use std::collections::VecDeque;

use super::*;

impl ConcurrentRadixTreeCompressed {
    /// Walks the whole tree once, same BFS shape as the test-only
    /// `raw_child_edge_count`/`edge_lengths_for_test` helpers in `mod.rs`
    /// (walk from `root` via each node's own `live_children()`-filtered
    /// child set) - the difference is this is a real (non-test) diagnostic,
    /// and it only ever reads counts (`Node::redundancy_counts`), never
    /// clones edge/worker data out the way `dump_snapshot` does for the
    /// event-dump path.
    ///
    /// Locking: each visited node takes its own brief read lock (shape gate
    /// then state), released immediately - readers never block other
    /// readers, only a write landing on that exact node at that exact
    /// instant. Cost scales with the number of distinct compressed-edge
    /// nodes in the whole fleet's index, so this is O(fleet index size),
    /// not O(1) like the existing per-worker gauges - call it rarely (the
    /// sampler in lib/llm uses 30s, not the 3s the live-index gauge uses).
    pub fn redundancy_stats(&self) -> RedundancyStats {
        let mut stats = RedundancyStats::default();
        let mut queue = VecDeque::from([self.root.clone()]);

        while let Some(node) = queue.pop_front() {
            let counts = node.redundancy_counts();
            if counts.edge_len > 0 {
                stats.distinct_blocks += counts.edge_len as u64;
                stats.total_block_copies += (counts.edge_len * counts.full_edge_workers) as u64
                    + counts.partial_cutoff_sum as u64;
            }
            queue.extend(counts.live_children);
        }

        stats
    }
}
