// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! f48h `kv_actions.engine_fit`: an HBM ledger per worker from the engines' own block counts,
//! so the router stops placing a turn where its whole prefix cannot be reserved now.
//!
//! A vLLM engine admits a waiting request only once it can reserve G1 blocks for the request's
//! whole prefix (a host-tier load included), and its waiting queue stops behind the first one that
//! cannot. The router's own bookings miss this: they count prompt blocks only, not generated
//! tokens, not what the engine's queue already waits to reserve, and (bug #13) the replay's tier
//! view lags. The ledger reads the engine instead, per worker:
//!
//! ```text
//! used  = active_blocks                      engine: blocks running and prefilling requests hold
//!       + ceil(waiting_tokens / block)       engine: what its waiting queue must reserve
//!       + ceil(growth_tokens / block)        router: hinted output of its in-flight requests
//! fits  = used + ceil(isl / block) <= total_blocks x margin
//! ```
//!
//! Live mapping: `active_blocks` and `total_blocks` are vLLM `kv_cache_usage_perc` x
//! `num_gpu_blocks` (or Dynamo KvStats `kv_active_blocks` / `kv_total_blocks`); `waiting_tokens` is
//! the forward-pass metrics' `queued_requests.sum_prefill_tokens` (preempted requests' generated
//! tokens are in `sum_decode_kv_tokens` there); growth is router-side from the
//! `expected_output_tokens` hint. Growth counts each in-flight request's whole hinted output, an
//! upper bound (what it already generated is in `active_blocks` too).
//!
//! `place` excludes the workers the turn does not fit on from selection while at least one fits
//! (the priority pool and the forecast then choose among those). `hold` keeps a queued turn at the
//! router while it fits nowhere. `dequeue` makes `kv_actions.dequeue` hand a stuck turn only to a
//! worker it fits on by this ledger instead of by the router's bookings.

use std::collections::HashSet;

use aisimulate_core::replay::ReplaySchedulerMetricsSnapshot;
use dynamo_kv_router::scheduling::KvEngineFitConfig;
use dynamo_router_protocol::worker::{WorkerId, WorkerWithDpRank};
use rustc_hash::FxHashMap;
use uuid::Uuid;

#[derive(Clone, Copy, Debug, Default)]
struct Gauge {
    active_blocks: u64,
    total_blocks: u64,
    waiting_tokens: u64,
    /// Running plus waiting requests.
    requests: u64,
}

/// Outcomes, logged when replay ends (`replay_engine_fit_stats`).
#[derive(Debug, Default)]
pub(crate) struct EngineFitStats {
    /// Placements where some workers were excluded and at least one fit.
    pub filtered: u64,
    /// Workers excluded over those placements.
    pub excluded: u64,
    /// Placements where the turn fit nowhere (placed as without the ledger).
    pub fit_nowhere: u64,
    /// Placements where it fit on every worker.
    pub fit_everywhere: u64,
    /// Queue checks that held a turn because it fit nowhere.
    pub held: u64,
    /// Gauge updates that freed room on some worker.
    pub freed: u64,
    /// Placements `spread` narrowed to the least committed workers.
    pub spread_narrowed: u64,
}

pub(crate) struct EngineFit {
    pub(crate) config: KvEngineFitConfig,
    block_size: u64,
    gauges: FxHashMap<WorkerWithDpRank, Gauge>,
    /// Hinted output tokens of in-flight requests, per worker.
    growth: FxHashMap<WorkerWithDpRank, u64>,
    in_flight: FxHashMap<Uuid, (WorkerWithDpRank, u64)>,
    pub(crate) stats: EngineFitStats,
}

impl EngineFit {
    pub(crate) fn new(config: KvEngineFitConfig, block_size: u32) -> Self {
        Self {
            config,
            block_size: u64::from(block_size.max(1)),
            gauges: FxHashMap::default(),
            growth: FxHashMap::default(),
            in_flight: FxHashMap::default(),
            stats: EngineFitStats::default(),
        }
    }

    /// Take the engines' gauges; whether any worker's committed blocks went down (room freed).
    pub(crate) fn set_gauges(&mut self, gauges: &[ReplaySchedulerMetricsSnapshot]) -> bool {
        let mut freed = false;
        for g in gauges {
            let worker = WorkerWithDpRank::new(g.worker_id as WorkerId, g.dp_rank);
            let next = Gauge {
                active_blocks: g.active_blocks,
                total_blocks: g.total_blocks,
                waiting_tokens: g.waiting_tokens,
                requests: g.running_requests + g.waiting_requests,
            };
            if let Some(prev) = self.gauges.insert(worker, next)
                && (self.committed(&next) < self.committed(&prev) || next.requests < prev.requests)
            {
                freed = true;
            }
        }
        self.stats.freed += u64::from(freed);
        freed
    }

    fn committed(&self, g: &Gauge) -> u64 {
        g.active_blocks + g.waiting_tokens.div_ceil(self.block_size)
    }

    /// A request went to `worker` with `output_tokens` hinted.
    pub(crate) fn dispatched(&mut self, uuid: Uuid, worker: WorkerWithDpRank, output_tokens: u64) {
        self.gone(uuid);
        self.in_flight.insert(uuid, (worker, output_tokens));
        *self.growth.entry(worker).or_default() += output_tokens;
    }

    /// The request finished, was aborted or was handed back.
    pub(crate) fn gone(&mut self, uuid: Uuid) {
        if let Some((worker, tokens)) = self.in_flight.remove(&uuid)
            && let Some(g) = self.growth.get_mut(&worker)
        {
            *g = g.saturating_sub(tokens);
        }
    }

    /// Whether a turn of `isl_tokens` fits on `worker` now; `None` before its first gauge.
    pub(crate) fn fits(&self, worker: WorkerWithDpRank, isl_tokens: usize) -> Option<bool> {
        let g = self.gauges.get(&worker)?;
        if g.total_blocks == 0 {
            return None;
        }
        let growth = self.growth.get(&worker).copied().unwrap_or(0);
        let used = self.committed(g) + growth.div_ceil(self.block_size);
        let need = (isl_tokens as u64).div_ceil(self.block_size);
        let hbm = (used + need) as f64 <= g.total_blocks as f64 * self.config.margin;
        let step = self.config.max_requests.is_none_or(|max| g.requests < max);
        Some(hbm && step)
    }

    /// Whether a turn fits on any rank of `workers`; `None` while any lacks a gauge.
    fn fits_anywhere<'a>(
        &self,
        workers: impl Iterator<Item = &'a WorkerWithDpRank>,
        isl_tokens: usize,
    ) -> Option<bool> {
        let mut any = false;
        for w in workers {
            any |= self.fits(*w, isl_tokens)?;
        }
        Some(any)
    }

    /// `hold`: whether a queued turn that has waited `waited_s` at the router stays there: it fits
    /// on no worker (once every worker has a gauge) and, with `hold_max_s`, its slack remains.
    pub(crate) fn holds(&self, isl_tokens: usize, waited_s: f64) -> bool {
        self.config.hold
            && self.config.hold_max_s.is_none_or(|max| waited_s < max)
            && !self.gauges.is_empty()
            && self.fits_anywhere(self.gauges.keys(), isl_tokens) == Some(false)
    }

    /// Committed blocks on `worker` (engine active, waiting queue, in-flight hinted output) and
    /// its G1 blocks; `None` before its first gauge.
    fn load(&self, worker: WorkerWithDpRank) -> Option<(f64, f64)> {
        let g = self.gauges.get(&worker)?;
        let growth = self.growth.get(&worker).copied().unwrap_or(0);
        Some((
            (self.committed(g) + growth.div_ceil(self.block_size)) as f64,
            g.total_blocks as f64,
        ))
    }

    /// The workers to exclude from this turn's selection; `None` when none is.
    ///
    /// `place`: those the turn does not fit on, while at least one of `workers` fits. `spread`:
    /// then, of the rest, those more than `spread` x G1 blocks above the least committed one
    /// (only once every remaining worker has a gauge).
    pub(crate) fn excluded(
        &mut self,
        workers: impl Iterator<Item = WorkerId>,
        dp_size: u32,
        isl_tokens: usize,
    ) -> Option<HashSet<WorkerId>> {
        if !self.config.place && self.config.spread.is_none() {
            return None;
        }
        let ids: Vec<WorkerId> = workers.collect();
        let ranks = dp_size.max(1);
        let mut out = HashSet::new();
        if self.config.place {
            let mut fit = 0usize;
            for &worker_id in &ids {
                // A worker is excluded when the turn fits on none of its ranks.
                let any = (0..ranks).any(|rank| {
                    self.fits(WorkerWithDpRank::new(worker_id, rank), isl_tokens) != Some(false)
                });
                if any {
                    fit += 1;
                } else {
                    out.insert(worker_id);
                }
            }
            if out.is_empty() {
                self.stats.fit_everywhere += 1;
            } else if fit == 0 {
                self.stats.fit_nowhere += 1;
                out.clear();
            } else {
                self.stats.filtered += 1;
                self.stats.excluded += out.len() as u64;
            }
        }
        if let Some(tolerance) = self.config.spread {
            // Each remaining worker at its least committed rank.
            let loads: Option<Vec<(WorkerId, f64, f64)>> = ids
                .iter()
                .filter(|w| !out.contains(w))
                .map(|&w| {
                    (0..ranks)
                        .filter_map(|rank| self.load(WorkerWithDpRank::new(w, rank)))
                        .min_by(|a, b| a.0.total_cmp(&b.0))
                        .map(|(used, total)| (w, used, total))
                })
                .collect();
            if let Some(loads) = loads
                && let Some(least) = loads.iter().map(|l| l.1).min_by(f64::total_cmp)
            {
                let before = out.len();
                for (w, used, total) in loads {
                    if used > least + tolerance * total {
                        out.insert(w);
                    }
                }
                self.stats.spread_narrowed += u64::from(out.len() > before);
            }
        }
        (!out.is_empty()).then_some(out)
    }
}

impl Drop for EngineFit {
    fn drop(&mut self) {
        tracing::info!(
            filtered = self.stats.filtered,
            excluded = self.stats.excluded,
            fit_nowhere = self.stats.fit_nowhere,
            fit_everywhere = self.stats.fit_everywhere,
            held = self.stats.held,
            freed = self.stats.freed,
            spread_narrowed = self.stats.spread_narrowed,
            workers = self.gauges.len(),
            "replay_engine_fit_stats"
        );
        eprintln!(
            "replay_engine_fit_stats filtered={} excluded={} fit_nowhere={} fit_everywhere={} held={} freed={} spread_narrowed={} workers={}",
            self.stats.filtered,
            self.stats.excluded,
            self.stats.fit_nowhere,
            self.stats.fit_everywhere,
            self.stats.held,
            self.stats.freed,
            self.stats.spread_narrowed,
            self.gauges.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gauge(
        worker: usize,
        active: u64,
        total: u64,
        waiting_tokens: u64,
    ) -> ReplaySchedulerMetricsSnapshot {
        ReplaySchedulerMetricsSnapshot {
            worker_id: worker,
            dp_rank: 0,
            active_blocks: active,
            inactive_blocks: 0,
            total_blocks: total,
            kv_cache_used_bytes: None,
            kv_cache_capacity_bytes: None,
            active_cache_usage: 0.0,
            physical_cache_usage: 0.0,
            running_requests: 2,
            waiting_requests: 1,
            host_used_blocks: 0,
            host_capacity_blocks: 0,
            waiting_tokens,
        }
    }

    fn fit(place: bool, hold: bool) -> EngineFit {
        EngineFit::new(
            KvEngineFitConfig {
                margin: 1.0,
                place,
                hold,
                hold_max_s: None,
                dequeue: false,
                max_requests: None,
                spread: None,
            },
            16,
        )
    }

    #[test]
    fn the_ledger_counts_active_waiting_and_growth() {
        let mut f = fit(true, false);
        f.set_gauges(&[gauge(0, 80, 100, 160)]); // 80 + 10 committed
        assert_eq!(f.fits(WorkerWithDpRank::new(0, 0), 160), Some(true)); // + 10 = 100
        assert_eq!(f.fits(WorkerWithDpRank::new(0, 0), 161), Some(false));
        f.dispatched(Uuid::from_u128(1), WorkerWithDpRank::new(0, 0), 16);
        assert_eq!(f.fits(WorkerWithDpRank::new(0, 0), 160), Some(false));
        f.gone(Uuid::from_u128(1));
        assert_eq!(f.fits(WorkerWithDpRank::new(0, 0), 160), Some(true));
        assert_eq!(f.fits(WorkerWithDpRank::new(1, 0), 1), None);
    }

    #[test]
    fn the_concurrency_limit_is_part_of_the_fit() {
        let mut f = fit(true, false);
        f.set_gauges(&[gauge(0, 0, 100, 0)]); // 3 requests
        assert_eq!(f.fits(WorkerWithDpRank::new(0, 0), 16), Some(true));
        f.config.max_requests = Some(4);
        assert_eq!(f.fits(WorkerWithDpRank::new(0, 0), 16), Some(true));
        f.config.max_requests = Some(3);
        assert_eq!(f.fits(WorkerWithDpRank::new(0, 0), 16), Some(false));
    }

    #[test]
    fn place_excludes_only_while_one_fits() {
        let mut f = fit(true, false);
        f.set_gauges(&[gauge(0, 100, 100, 0), gauge(1, 10, 100, 0)]);
        let ex = f.excluded([0u64, 1].into_iter(), 1, 160).unwrap();
        assert!(ex.contains(&0) && !ex.contains(&1));
        f.set_gauges(&[gauge(1, 100, 100, 0)]);
        assert!(f.excluded([0u64, 1].into_iter(), 1, 160).is_none());
        assert_eq!(f.stats.fit_nowhere, 1);
        assert!(
            fit(false, false)
                .excluded([0u64].into_iter(), 1, 1)
                .is_none()
        );
    }

    #[test]
    fn spread_keeps_the_least_committed_workers() {
        let mut f = fit(false, false);
        f.config.spread = Some(0.05);
        f.set_gauges(&[
            gauge(0, 50, 100, 0),
            gauge(1, 54, 100, 0),
            gauge(2, 60, 100, 0),
        ]);
        let ex = f.excluded([0u64, 1, 2].into_iter(), 1, 16).unwrap();
        assert_eq!(ex, HashSet::from([2]));
        // In-flight hinted output counts: 96 tokens = 6 blocks puts worker 1 over 55.
        f.dispatched(Uuid::from_u128(1), WorkerWithDpRank::new(1, 0), 96);
        let ex = f.excluded([0u64, 1, 2].into_iter(), 1, 16).unwrap();
        assert_eq!(ex, HashSet::from([1, 2]));
        // With the fit guard, a full worker leaves first; the spread is over the rest.
        f.config.place = true;
        f.set_gauges(&[gauge(0, 99, 100, 0)]);
        // Worker 1 (54 + 6 growth) and worker 2 (60) tie, so the spread narrows nothing more.
        let ex = f.excluded([0u64, 1, 2].into_iter(), 1, 32).unwrap();
        assert_eq!(ex, HashSet::from([0]));
        assert_eq!(f.stats.spread_narrowed, 2);
    }

    #[test]
    fn hold_only_when_no_worker_fits() {
        let mut f = fit(false, true);
        assert!(!f.holds(16, 0.0));
        f.set_gauges(&[gauge(0, 100, 100, 0), gauge(1, 99, 100, 0)]);
        assert!(!f.holds(16, 0.0));
        assert!(f.holds(32, 0.0));
        f.config.hold_max_s = Some(2.0);
        assert!(f.holds(32, 1.9));
        assert!(!f.holds(32, 2.0));
        assert!(f.set_gauges(&[gauge(1, 50, 100, 0)]));
        assert!(!f.holds(32, 0.0));
    }
}
