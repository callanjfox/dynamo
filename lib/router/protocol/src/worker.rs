// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker and rank identities, and the capacity a worker advertises about itself.

use serde::{Deserialize, Serialize};

/// A worker identifier.
pub type WorkerId = u64;

/// A data parallel rank identifier.
pub type DpRank = u32;

/// A worker identifier combined with its data parallel rank.
/// Used for routing decisions in data parallel setups.
/// dp_rank = 0 indicates either DP not enabled or the first rank.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkerWithDpRank {
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
}

impl WorkerWithDpRank {
    pub fn new(worker_id: WorkerId, dp_rank: DpRank) -> Self {
        Self { worker_id, dp_rank }
    }

    pub fn from_worker_id(worker_id: WorkerId) -> Self {
        Self {
            worker_id,
            dp_rank: 0,
        }
    }
}

/// Prefill budget assumed for a rank that does not report `max_num_batched_tokens`. Large
/// enough that a fractional prefill threshold never fires on an unreported budget.
pub const DEFAULT_MAX_BATCHED_TOKENS: u64 = 10_000_000;

/// Advertised capacity of one worker rank, from its runtime config. It changes only when
/// the worker registers again, so hosts serve it from cached discovery state, not per request.
/// The values cross the process boundary in the worker's runtime config, and the slot
/// tracker, worker selection, and admission all name the type, so it lives in the wire crate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorkerCapacity {
    max_num_batched_tokens: Option<u64>,
    total_kv_blocks: Option<u64>,
    /// The rank's running-request limit (`max_num_seqs`), set only when the busy check counts
    /// it ([`busy_at_max_num_seqs`]).
    max_num_seqs: Option<u64>,
}

/// Whether a worker counts as busy at its running-request limit (`max_num_seqs`) as well as by
/// tokens: `DYN_ROUTER_BUSY_AT_MAX_NUM_SEQS=1`. Off by default. Read once.
pub fn busy_at_max_num_seqs() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("DYN_ROUTER_BUSY_AT_MAX_NUM_SEQS")
            .is_ok_and(|v| matches!(v.as_str(), "1" | "true" | "yes"))
    })
}

impl WorkerCapacity {
    /// A reported zero means unreported. vLLM registers `total_kv_blocks = 0` as a placeholder
    /// under the Ray data-parallel backend, and a zero budget would mark every rank full.
    pub fn new(max_num_batched_tokens: Option<u64>, total_kv_blocks: Option<u64>) -> Self {
        Self {
            max_num_batched_tokens: max_num_batched_tokens.filter(|&value| value > 0),
            total_kv_blocks: total_kv_blocks.filter(|&value| value > 0),
            max_num_seqs: None,
        }
    }

    /// With the rank's running-request limit; a reported zero means unreported.
    pub fn with_max_num_seqs(mut self, max_num_seqs: Option<u64>) -> Self {
        self.max_num_seqs = max_num_seqs.filter(|&value| value > 0);
        self
    }

    /// The running-request limit the busy check counts, when set.
    pub fn max_num_seqs(self) -> Option<u64> {
        self.max_num_seqs
    }

    /// Prefill tokens the rank batches per engine step, when reported.
    pub fn max_num_batched_tokens(self) -> Option<u64> {
        self.max_num_batched_tokens
    }

    /// Total KV blocks the worker advertises for this rank, when reported: capacity, not free
    /// blocks. Some adapters derive it from aggregate data (vLLM data parallelism publishes
    /// the process total divided by the rank count), so treat it as approximate.
    pub fn total_kv_blocks(self) -> Option<u64> {
        self.total_kv_blocks
    }

    /// The budget fractional prefill thresholds are taken of: `max_num_batched_tokens`, or
    /// [`DEFAULT_MAX_BATCHED_TOKENS`] when the rank does not report it.
    pub fn prefill_budget(self) -> u64 {
        self.max_num_batched_tokens
            .unwrap_or(DEFAULT_MAX_BATCHED_TOKENS)
    }
}
