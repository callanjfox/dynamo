// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! f62 `kv_actions.prefill_isolation`: keep big standard/batch prefills off the workers that
//! carry priority decodes.
//!
//! With HBM to spare (4x) and 8 workers, priority ITL is set by the chunked prefill of long
//! (~200k-token) standard contexts that share a worker with priority decodes: every step of such
//! a prefill carries a full `max_num_batched_tokens` chunk next to the decode batch (f54osl: ITL
//! about 45-50 ms against about 31 ms decode-only). Holding standard by running count broke cache
//! affinity and made it worse, so this neither holds nor moves anything already placed. It only
//! changes where a protected class's turn goes when the turn would compute a big prefill:
//!
//! - a worker is **protected** when it is in the priority pool (`workers: pool`, the pool size the
//!   session scheduler's pool scorer publishes through `adaptive_switches`), or when it runs at
//!   least `min_priority` priority requests admitted and not finished (`workers: priority_active`);
//! - a turn of a listed class leaves the selection of a protected worker on which it would compute
//!   more than `max_prefill_tokens` (its input less what that worker holds on device or host, or
//!   less the longest prefix a peer could hand it over KVCR, whichever is more);
//! - a turn that is cached on a protected worker (its usual next turn) keeps that worker: its
//!   prefill there is small;
//! - when that would leave no worker, the turn keeps only the excluded workers with the fewest
//!   priority requests (never an empty selection, never a hold).
//!
//! Priority is never excluded, held or moved. Off (absent) changes nothing.

use std::collections::HashSet;

use dynamo_kv_router::scheduling::{KvPrefillIsolationConfig, PrefillIsolationWorkers};
use dynamo_router_protocol::worker::{WorkerId, WorkerWithDpRank};
use rustc_hash::FxHashMap;
use uuid::Uuid;

/// Outcomes, logged when replay ends (`replay_prefill_isolation_stats`).
#[derive(Debug, Default)]
pub(crate) struct PrefillIsolationStats {
    /// Turns of a listed class scored.
    pub scored: u64,
    /// Turns that saw at least one protected worker.
    pub protected_seen: u64,
    /// Turns that left at least one protected worker.
    pub steered: u64,
    /// Protected workers left out over those turns.
    pub excluded: u64,
    /// Turns where every worker was left out, kept on the fewest-priority ones.
    pub fallback: u64,
    /// Turns with no pool size published yet (`workers: pool`).
    pub no_pool: u64,
}

/// One worker as the isolation sees it for one turn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Row {
    pub worker_id: WorkerId,
    pub protected: bool,
    /// Tokens the turn would compute there.
    pub new_tokens: usize,
    /// Priority requests running there.
    pub priority: usize,
    /// Left out for another reason before this.
    pub already: bool,
}

/// The decision, apart from the router: the workers to leave out (a superset of `already`), and
/// whether it fell back. `None` when it leaves nobody out beyond `already`.
pub(crate) fn exclude(
    max_prefill_tokens: usize,
    rows: &[Row],
) -> (Option<HashSet<WorkerId>>, bool) {
    let mut out: HashSet<WorkerId> = rows
        .iter()
        .filter(|r| r.already)
        .map(|r| r.worker_id)
        .collect();
    let before = out.len();
    let mine: Vec<&Row> = rows
        .iter()
        .filter(|r| !r.already && r.protected && r.new_tokens > max_prefill_tokens)
        .collect();
    if mine.is_empty() {
        return (None, false);
    }
    for r in &mine {
        out.insert(r.worker_id);
    }
    let open = rows.iter().any(|r| !out.contains(&r.worker_id));
    if open {
        return ((out.len() > before).then_some(out), false);
    }
    // Every worker left out: keep the protected ones this turn left with the fewest priority.
    let least = mine.iter().map(|r| r.priority).min().unwrap_or(0);
    for r in &mine {
        if r.priority == least {
            out.remove(&r.worker_id);
        }
    }
    ((out.len() > before).then_some(out), true)
}

pub(crate) struct PrefillIsolation {
    pub(crate) config: KvPrefillIsolationConfig,
    /// Priority requests admitted and not finished, per rank.
    priority: FxHashMap<WorkerWithDpRank, usize>,
    in_flight: FxHashMap<Uuid, WorkerWithDpRank>,
    pub(crate) stats: PrefillIsolationStats,
}

impl PrefillIsolation {
    pub(crate) fn new(config: KvPrefillIsolationConfig) -> Self {
        eprintln!("f62 kv_actions.prefill_isolation: {config:?}");
        Self {
            config,
            priority: FxHashMap::default(),
            in_flight: FxHashMap::default(),
            stats: PrefillIsolationStats::default(),
        }
    }

    pub(crate) fn applies_to(&self, class: &str) -> bool {
        self.config.classes.iter().any(|c| c == class)
    }

    pub(crate) fn is_priority(&self, class: &str) -> bool {
        class == self.config.priority_class
    }

    /// A request went to `worker`; counted when it is of the priority class.
    pub(crate) fn dispatched(&mut self, uuid: Uuid, worker: WorkerWithDpRank, priority: bool) {
        self.gone(uuid);
        if priority {
            self.in_flight.insert(uuid, worker);
            *self.priority.entry(worker).or_default() += 1;
        }
    }

    /// The request finished, was aborted or was handed back.
    pub(crate) fn gone(&mut self, uuid: Uuid) {
        if let Some(worker) = self.in_flight.remove(&uuid)
            && let Some(n) = self.priority.get_mut(&worker)
        {
            *n = n.saturating_sub(1);
        }
    }

    /// Priority requests on `worker_id`, all its ranks.
    pub(crate) fn priority_on(&self, worker_id: WorkerId, dp_size: u32) -> usize {
        (0..dp_size.max(1))
            .map(|rank| {
                self.priority
                    .get(&WorkerWithDpRank::new(worker_id, rank))
                    .copied()
                    .unwrap_or(0)
            })
            .sum()
    }

    /// Which of `ids` are protected now; `None` when nothing is (no pool published yet).
    pub(crate) fn protected(&mut self, ids: &[WorkerId], dp_size: u32) -> Option<Vec<bool>> {
        match self.config.workers {
            PrefillIsolationWorkers::Pool => {
                let Some(size) = dynamo_kv_router::plugins::adaptive_switches::priority_pool_size()
                else {
                    self.stats.no_pool += 1;
                    return None;
                };
                pool_side(ids, size)
            }
            PrefillIsolationWorkers::PriorityActive => Some(self.active(ids, dp_size)),
            // f68pool: the pool while it acts, else the workers running priority.
            PrefillIsolationWorkers::PoolElseActive => {
                match dynamo_kv_router::plugins::adaptive_switches::priority_pool_size() {
                    Some(size) => Some(pool_side(ids, size)?),
                    None => Some(self.active(ids, dp_size)),
                }
            }
        }
    }

    /// The workers running at least `min_priority` priority requests.
    fn active(&self, ids: &[WorkerId], dp_size: u32) -> Vec<bool> {
        ids.iter()
            .map(|id| self.priority_on(*id, dp_size) >= self.config.min_priority.max(1))
            .collect()
    }
}

/// The pool's workers among `ids`: the `size` lowest ids (`None` for an empty pool or list).
fn pool_side(ids: &[WorkerId], size: usize) -> Option<Vec<bool>> {
    let mut sorted = ids.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    if size == 0 || sorted.is_empty() {
        return None;
    }
    let edge = sorted[size.min(sorted.len()) - 1];
    Some(ids.iter().map(|id| *id <= edge).collect())
}

impl Drop for PrefillIsolation {
    fn drop(&mut self) {
        let s = &self.stats;
        eprintln!(
            "replay_prefill_isolation_stats scored={} protected_seen={} steered={} excluded={} fallback={} no_pool={}",
            s.scored, s.protected_seen, s.steered, s.excluded, s.fallback, s.no_pool
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(worker_id: WorkerId, protected: bool, new_tokens: usize, priority: usize) -> Row {
        Row {
            worker_id,
            protected,
            new_tokens,
            priority,
            already: false,
        }
    }

    #[test]
    fn small_prefills_and_unprotected_workers_stay() {
        let rows = [row(0, true, 2_000, 9), row(1, false, 200_000, 0)];
        assert_eq!(exclude(8192, &rows), (None, false));
    }

    #[test]
    fn a_big_prefill_leaves_the_protected_workers() {
        // Cached on w0 (small there): w0 stays. Uncached on w1 (protected): it leaves.
        let rows = [
            row(0, true, 1_000, 5),
            row(1, true, 200_000, 5),
            row(2, false, 200_000, 0),
        ];
        let (out, fell) = exclude(8192, &rows);
        assert_eq!(out, Some(HashSet::from([1])));
        assert!(!fell);
    }

    #[test]
    fn never_empty_keeps_the_fewest_priority() {
        let rows = [
            row(0, true, 200_000, 7),
            row(1, true, 200_000, 3),
            row(2, true, 200_000, 3),
        ];
        let (out, fell) = exclude(8192, &rows);
        assert_eq!(out, Some(HashSet::from([0])));
        assert!(fell);
    }

    #[test]
    fn keeps_what_was_already_left_out() {
        let mut rows = [
            row(0, true, 200_000, 1),
            row(1, false, 10, 0),
            row(2, false, 10, 0),
        ];
        rows[2].already = true;
        let (out, _) = exclude(8192, &rows);
        assert_eq!(out, Some(HashSet::from([0, 2])));
    }

    #[test]
    fn priority_counts_follow_dispatch_and_end() {
        let config = KvPrefillIsolationConfig {
            workers: PrefillIsolationWorkers::PriorityActive,
            ..Default::default()
        };
        let mut iso = PrefillIsolation::new(config);
        let a = Uuid::from_u128(1);
        let b = Uuid::from_u128(2);
        let w = WorkerWithDpRank::new(3, 0);
        iso.dispatched(a, w, true);
        iso.dispatched(b, w, false);
        assert_eq!(iso.priority_on(3, 1), 1);
        assert_eq!(iso.protected(&[3, 4], 1), Some(vec![true, false]));
        iso.gone(a);
        iso.gone(a);
        assert_eq!(iso.priority_on(3, 1), 0);
    }
}
