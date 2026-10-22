// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-session workload and cache-pressure accounting.
//!
//! [`SessionPrefixIndexer`](crate::SessionPrefixIndexer) records where each session's
//! prefix is resident; this tracker records what each session has asked for. Together
//! they give the working set a fleet needs against the capacity it has, and the
//! tokens a returning session had to recompute or reload (thrash).
//!
//! The tracker is passive. It reads no state to make a routing decision, takes every
//! timestamp from the caller so replay can drive it in simulated time, and keeps one
//! entry per session, bounded by `max_sessions`.
//!
//! Demand is tracked apart from residency on purpose. A session's frontier in the
//! index recedes when blocks are evicted, so it says what is held, not what the
//! session needs. Demand is the prior turn's input plus output, which is the prefix
//! the next turn will present if the harness resends its history.

use std::collections::{HashMap, HashSet};

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use serde::Serialize;

use crate::protocols::WorkerWithDpRank;
use crate::session_prefix_index::{ResidencySnapshot, SessionId, TierBlocks};

/// Upper edges of the turn-gap buckets, in milliseconds: the same 1, 5 and 15 minute
/// windows the demand gauges use. A gap above the last edge falls in the overflow bucket.
pub const GAP_BUCKET_EDGES_MS: [u64; 3] = [60_000, 300_000, 900_000];
const GAP_BUCKETS: usize = GAP_BUCKET_EDGES_MS.len() + 1;

/// Metric-label names for the turn-gap buckets, in bucket order.
pub const GAP_BUCKET_LABELS: [&str; GAP_BUCKETS] = ["le_1m", "le_5m", "le_15m", "gt_15m"];

/// Index of the turn-gap bucket a gap falls in.
pub fn gap_bucket(gap_ms: u64) -> usize {
    GAP_BUCKET_EDGES_MS
        .iter()
        .position(|&edge| gap_ms <= edge)
        .unwrap_or(GAP_BUCKETS - 1)
}

/// Where a returning session's prior prefix was found, to within a tenth of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReturnOutcome {
    HitDevice,
    HitHost,
    HitDisk,
    Miss,
}

impl ReturnOutcome {
    /// Metric-label name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HitDevice => "gpu",
            Self::HitHost => "host",
            Self::HitDisk => "disk",
            Self::Miss => "miss",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::HitDevice => 0,
            Self::HitHost => 1,
            Self::HitDisk => 2,
            Self::Miss => 3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SessionStatsConfig {
    pub block_size: u32,
    /// A session counts toward the working set while its last activity is within
    /// this window and it has not ended.
    pub active_window_ms: u64,
    /// A session with no activity for this long is ended by [`SessionStatsTracker::sweep`].
    pub idle_ttl_ms: u64,
    pub max_sessions: usize,
}

impl Default for SessionStatsConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            active_window_ms: 300_000,
            idle_ttl_ms: 1_800_000,
            max_sessions: 16_384,
        }
    }
}

/// Tokens a returning session had to rebuild rather than read from the GPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ThrashTokens {
    /// Prior prefix found in no tier, so recomputed from the prompt.
    pub recomputed: u64,
    /// Prior prefix found only in host memory, so reloadable.
    pub reloaded_host: u64,
    /// Prior prefix found only on disk, so reloadable.
    pub reloaded_disk: u64,
}

impl ThrashTokens {
    pub fn total(&self) -> u64 {
        self.recomputed + self.reloaded_host + self.reloaded_disk
    }

    fn add(&mut self, other: ThrashTokens) {
        self.recomputed += other.recomputed;
        self.reloaded_host += other.reloaded_host;
        self.reloaded_disk += other.reloaded_disk;
    }
}

/// What the tracker learned from one request arrival.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArrivalOutcome {
    /// The request opened a session: either the first one, or one that resumed
    /// after ending and now counts as a new episode.
    pub started: bool,
    /// The session had an earlier turn with a prefix to find.
    pub returning: bool,
    /// Prefix the turn was expected to reuse: the smaller of the prior turn's
    /// input and this turn's input.
    pub prior_tokens: u64,
    /// Time since the session's previous turn finished.
    pub gap_ms: Option<u64>,
    pub thrash: ThrashTokens,
    pub outcome: Option<ReturnOutcome>,
}

/// One session's totals, emitted when it ends. This is the record that carries
/// per-session detail; metrics only ever see aggregates of it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SessionSummary {
    pub session_id: SessionId,
    pub parent_session_id: Option<SessionId>,
    pub turns: u32,
    pub isl_tokens: u64,
    pub osl_tokens: u64,
    pub cached_tokens: u64,
    pub hit_rate: f64,
    pub duration_ms: u64,
    pub mean_turn_gap_ms: f64,
    pub peak_demand_tokens: u64,
    pub children: u32,
    pub thrash: ThrashTokens,
}

/// Fleet-level view at one instant. Serialized as one time-series row.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SessionStatsSnapshot {
    pub now_ms: u64,
    pub active_sessions: u64,
    pub sessions_started: u64,
    pub sessions_ended: u64,
    pub turns: u64,
    /// Sum over active sessions of the prefix they will present next.
    pub demand_tokens: u64,
    /// Demand counting each distinct prompt block once across active sessions, so a
    /// prefix two sessions share is not counted twice, plus each session's latest output.
    /// Split by role: a subagent counts only the blocks its primary sessions do not cover.
    pub demand_distinct_tokens: u64,
    pub demand_distinct_primary_tokens: u64,
    pub demand_distinct_subagent_tokens: u64,
    pub active_primary_sessions: u64,
    pub active_subagent_sessions: u64,
    /// Prompt tokens referenced by more than one active session: the sum of each
    /// session's prompt blocks minus the distinct blocks.
    pub shared_tokens: u64,
    /// Distinct demand of the active sessions homed on each worker, meaning the worker
    /// that served their last turn: a block two of them share counts once.
    #[serde(skip)]
    pub home_demand: HashMap<WorkerWithDpRank, u64>,
    /// The demand sum counting only sessions active within each of
    /// [`DEMAND_WINDOWS_MS`], so a short window shows demand with idle sessions decayed away.
    pub demand_window_tokens: [u64; 3],

    /// Tokens held per tier if no prefix were shared between sessions.
    pub resident_device_tokens: u64,
    pub resident_host_tokens: u64,
    pub resident_disk_tokens: u64,
    /// Distinct tokens held per tier, counting a shared prefix once.
    pub distinct_device_tokens: u64,
    pub distinct_host_tokens: u64,
    pub distinct_disk_tokens: u64,
    /// Tokens held for sessions that have ended or gone idle.
    pub stranded_device_tokens: u64,
    pub stranded_host_tokens: u64,
    /// Tokens held by more than one worker: copies of a prefix beyond the first.
    pub replicated_device_tokens: u64,
    pub replicated_host_tokens: u64,
    /// Sessions whose device-tier prefix sits on more than one worker.
    pub split_sessions: u64,
    /// Cumulative thrash since the tracker started.
    pub thrash: ThrashTokens,
    /// Cumulative returning turns and where their prefix was found, by turn gap.
    /// Rows follow [`GAP_BUCKET_EDGES_MS`] plus an overflow row; columns are
    /// device, host, disk, miss.
    pub gap_outcomes: Vec<[u64; 4]>,
}

/// Windows over which demand is also reported, so a session idle longer than the
/// window stops counting. The configured active window is separate.
pub const DEMAND_WINDOWS_MS: [u64; 3] = [60_000, 300_000, 900_000];

#[derive(Debug)]
struct Stat {
    parent: Option<SessionId>,
    started_ms: u64,
    last_arrival_ms: u64,
    last_done_ms: Option<u64>,
    ended_ms: Option<u64>,
    turns: u32,
    isl_sum: u64,
    osl_sum: u64,
    cached_sum: u64,
    inflight_isl: u64,
    demand: u64,
    peak_demand: u64,
    children: u32,
    gap_sum_ms: u64,
    gap_count: u32,
    thrash: ThrashTokens,
    /// An earlier episode left a prefix to find, though this one has no turns yet.
    carried: bool,
    /// Worker that served the latest turn.
    home: Option<WorkerWithDpRank>,
    /// Chained hashes of the latest prompt's blocks, counted in [`DistinctBlocks`]
    /// while the session is active.
    blocks: Vec<u64>,
}

/// Reference counts of the prompt blocks active sessions present, so demand can count
/// a shared block once.
#[derive(Debug, Default)]
struct DistinctBlocks {
    all: FxHashMap<u64, u32>,
    primary: FxHashMap<u64, u32>,
    /// The same counts per home worker, so a worker's own distinct need is known.
    by_worker: FxHashMap<WorkerWithDpRank, FxHashMap<u64, u32>>,
    /// Sum over sessions of their block count, shared blocks counted per session.
    references: u64,
}

impl DistinctBlocks {
    fn add(&mut self, hash: u64, primary: bool) {
        *self.all.entry(hash).or_default() += 1;
        if primary {
            *self.primary.entry(hash).or_default() += 1;
        }
        self.references += 1;
    }

    fn remove(&mut self, hash: u64, primary: bool) {
        fn drop_one(map: &mut FxHashMap<u64, u32>, hash: u64) {
            if let Some(count) = map.get_mut(&hash) {
                *count -= 1;
                if *count == 0 {
                    map.remove(&hash);
                }
            }
        }
        drop_one(&mut self.all, hash);
        if primary {
            drop_one(&mut self.primary, hash);
        }
        self.references = self.references.saturating_sub(1);
    }

    fn add_to_worker(&mut self, worker: WorkerWithDpRank, hash: u64) {
        *self
            .by_worker
            .entry(worker)
            .or_default()
            .entry(hash)
            .or_default() += 1;
    }

    fn remove_from_worker(&mut self, worker: WorkerWithDpRank, hash: u64) {
        if let Some(map) = self.by_worker.get_mut(&worker) {
            if let Some(count) = map.get_mut(&hash) {
                *count -= 1;
                if *count == 0 {
                    map.remove(&hash);
                }
            }
            if map.is_empty() {
                self.by_worker.remove(&worker);
            }
        }
    }

    /// Swap a session's blocks for `new`, touching only the part after the shared
    /// prefix: a prompt usually extends the previous one. When the session moved to
    /// another worker its need moves with it (blocks it left behind stay cached on
    /// the old worker, which the residency index reports).
    fn replace(
        &mut self,
        held: &mut Vec<u64>,
        new: Vec<u64>,
        primary: bool,
        old_home: Option<WorkerWithDpRank>,
        home: WorkerWithDpRank,
    ) {
        let common = held.iter().zip(&new).take_while(|(a, b)| a == b).count();
        for &hash in &held[common..] {
            self.remove(hash, primary);
        }
        for &hash in &new[common..] {
            self.add(hash, primary);
        }
        if old_home == Some(home) {
            for &hash in &held[common..] {
                self.remove_from_worker(home, hash);
            }
            for &hash in &new[common..] {
                self.add_to_worker(home, hash);
            }
        } else {
            if let Some(old) = old_home {
                for &hash in held.iter() {
                    self.remove_from_worker(old, hash);
                }
            }
            for &hash in &new {
                self.add_to_worker(home, hash);
            }
        }
        *held = new;
    }

    fn release(&mut self, held: &mut Vec<u64>, primary: bool, home: Option<WorkerWithDpRank>) {
        for &hash in held.iter() {
            self.remove(hash, primary);
            if let Some(home) = home {
                self.remove_from_worker(home, hash);
            }
        }
        held.clear();
    }
}

/// Hash each block together with the blocks before it, so equal hashes mean equal prefixes.
fn chain_hashes(local_hashes: &[u64]) -> Vec<u64> {
    let mut previous = 0x9E37_79B9_7F4A_7C15u64;
    local_hashes
        .iter()
        .map(|&hash| {
            previous = (previous.rotate_left(5) ^ hash).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            previous
        })
        .collect()
}

impl Stat {
    /// Begin a new episode after the session ended: the per-session totals start
    /// over, while the prefix demand and last activity carry across so the
    /// returning turn is still measured against what the session left behind.
    fn restart(&mut self, now_ms: u64) {
        self.started_ms = now_ms;
        self.turns = 0;
        self.isl_sum = 0;
        self.osl_sum = 0;
        self.cached_sum = 0;
        self.peak_demand = self.demand;
        self.children = 0;
        self.gap_sum_ms = 0;
        self.gap_count = 0;
        self.thrash = ThrashTokens::default();
        self.carried = true;
    }

    fn last_activity_ms(&self) -> u64 {
        self.last_done_ms
            .map_or(self.last_arrival_ms, |done| done.max(self.last_arrival_ms))
    }

    fn is_active(&self, now_ms: u64, window_ms: u64) -> bool {
        self.ended_ms.is_none() && now_ms.saturating_sub(self.last_activity_ms()) <= window_ms
    }

    fn summary(&self, session_id: &str, end_ms: u64) -> SessionSummary {
        SessionSummary {
            session_id: session_id.to_string(),
            parent_session_id: self.parent.clone(),
            turns: self.turns,
            isl_tokens: self.isl_sum,
            osl_tokens: self.osl_sum,
            cached_tokens: self.cached_sum,
            hit_rate: if self.isl_sum == 0 {
                0.0
            } else {
                self.cached_sum as f64 / self.isl_sum as f64
            },
            duration_ms: end_ms.saturating_sub(self.started_ms),
            mean_turn_gap_ms: if self.gap_count == 0 {
                0.0
            } else {
                self.gap_sum_ms as f64 / f64::from(self.gap_count)
            },
            peak_demand_tokens: self.peak_demand,
            children: self.children,
            thrash: self.thrash,
        }
    }
}

#[derive(Debug, Default)]
struct Inner {
    sessions: HashMap<SessionId, Stat>,
    started: u64,
    ended: u64,
    turns: u64,
    thrash: ThrashTokens,
    gap_outcomes: [[u64; 4]; GAP_BUCKETS],
    distinct: DistinctBlocks,
}

/// Thread-safe per-session stats table.
#[derive(Debug)]
pub struct SessionStatsTracker {
    config: SessionStatsConfig,
    inner: Mutex<Inner>,
}

impl SessionStatsTracker {
    pub fn new(config: SessionStatsConfig) -> Self {
        Self {
            config: SessionStatsConfig {
                block_size: config.block_size.max(1),
                max_sessions: config.max_sessions.max(1),
                ..config
            },
            inner: Mutex::new(Inner::default()),
        }
    }

    pub fn config(&self) -> &SessionStatsConfig {
        &self.config
    }

    /// Record a request arriving for `session_id`. `resident` is what the session
    /// index holds for the session at this moment, per tier.
    pub fn on_arrival(
        &self,
        session_id: &str,
        parent_session_id: Option<&str>,
        isl_tokens: u64,
        resident: TierBlocks,
        now_ms: u64,
    ) -> ArrivalOutcome {
        let block = u64::from(self.config.block_size);
        let mut inner = self.inner.lock();

        if !inner.sessions.contains_key(session_id) {
            if inner.sessions.len() >= self.config.max_sessions {
                inner.evict_least_recent();
            }
            inner.started += 1;
            if let Some(parent) = parent_session_id
                && let Some(parent_stat) = inner.sessions.get_mut(parent)
            {
                parent_stat.children += 1;
            }
            inner.sessions.insert(
                session_id.to_string(),
                Stat {
                    parent: parent_session_id.map(str::to_string),
                    started_ms: now_ms,
                    last_arrival_ms: now_ms,
                    last_done_ms: None,
                    ended_ms: None,
                    turns: 0,
                    isl_sum: 0,
                    osl_sum: 0,
                    cached_sum: 0,
                    inflight_isl: 0,
                    demand: 0,
                    peak_demand: 0,
                    children: 0,
                    gap_sum_ms: 0,
                    gap_count: 0,
                    thrash: ThrashTokens::default(),
                    carried: false,
                    home: None,
                    blocks: Vec::new(),
                },
            );
        }

        let stat = inner
            .sessions
            .get_mut(session_id)
            .expect("session inserted above");
        // A session that sends another turn after ending has resumed.
        let resumed = stat.ended_ms.take().is_some();
        if resumed {
            stat.restart(now_ms);
        }

        let returning = stat.turns > 0 || stat.carried;
        // The prefix a worker cached for the previous turn is that turn's input. Its
        // output is not counted: it is generated, not part of the prompt a worker
        // prefilled, so no reuse figure (the funnel included) counts it either.
        let prior = stat.inflight_isl.min(isl_tokens);
        let gap_ms = returning.then(|| now_ms.saturating_sub(stat.last_activity_ms()));

        let device = resident.device * block;
        let host = (resident.host * block).max(device);
        let disk = (resident.disk * block).max(host);
        let covered_device = prior.min(device);
        let covered_host = prior.min(host);
        let covered_disk = prior.min(disk);
        let thrash = if returning {
            ThrashTokens {
                recomputed: prior - covered_disk,
                reloaded_host: covered_host - covered_device,
                reloaded_disk: covered_disk - covered_host,
            }
        } else {
            ThrashTokens::default()
        };
        // The last turn's output blocks are often not yet resident anywhere, so a
        // hit allows a tenth of the prefix, or one block, to be missing.
        let tolerance = block.max(prior / 10);
        let outcome = (returning && prior > 0).then(|| {
            if covered_device + tolerance >= prior {
                ReturnOutcome::HitDevice
            } else if covered_host + tolerance >= prior {
                ReturnOutcome::HitHost
            } else if covered_disk + tolerance >= prior {
                ReturnOutcome::HitDisk
            } else {
                ReturnOutcome::Miss
            }
        });

        stat.turns += 1;
        stat.isl_sum += isl_tokens;
        stat.inflight_isl = isl_tokens;
        stat.last_arrival_ms = now_ms;
        stat.thrash.add(thrash);
        if let Some(gap) = gap_ms {
            stat.gap_sum_ms += gap;
            stat.gap_count += 1;
        }

        if resumed {
            inner.started += 1;
        }
        inner.turns += 1;
        inner.thrash.add(thrash);
        if let (Some(gap), Some(outcome)) = (gap_ms, outcome) {
            inner.gap_outcomes[gap_bucket(gap)][outcome.index()] += 1;
        }

        ArrivalOutcome {
            started: !returning || resumed,
            returning,
            prior_tokens: prior,
            gap_ms,
            thrash,
            outcome,
        }
    }

    /// Record where a request was routed and the block hashes of its prompt, so demand
    /// counts a prefix shared between sessions once.
    pub fn on_routed(&self, session_id: &str, worker: WorkerWithDpRank, local_hashes: &[u64]) {
        let mut guard = self.inner.lock();
        let inner = &mut *guard;
        let Some(stat) = inner.sessions.get_mut(session_id) else {
            return;
        };
        let old_home = stat.home.replace(worker);
        let primary = stat.parent.is_none();
        inner.distinct.replace(
            &mut stat.blocks,
            chain_hashes(local_hashes),
            primary,
            old_home,
            worker,
        );
    }

    /// Record a request finishing. `cached_tokens` is the prefix the worker reused.
    pub fn on_complete(&self, session_id: &str, osl_tokens: u64, cached_tokens: u64, now_ms: u64) {
        let mut inner = self.inner.lock();
        let Some(stat) = inner.sessions.get_mut(session_id) else {
            return;
        };
        stat.osl_sum += osl_tokens;
        stat.cached_sum += cached_tokens;
        stat.demand = stat.inflight_isl + osl_tokens;
        stat.peak_demand = stat.peak_demand.max(stat.demand);
        stat.last_done_ms = Some(now_ms);
    }

    /// End a session, for example on a final-turn marker. Returns its summary once.
    pub fn end_session(&self, session_id: &str, now_ms: u64) -> Option<SessionSummary> {
        let mut guard = self.inner.lock();
        let inner = &mut *guard;
        let stat = inner.sessions.get_mut(session_id)?;
        if stat.ended_ms.is_some() {
            return None;
        }
        stat.ended_ms = Some(now_ms);
        let summary = stat.summary(session_id, now_ms);
        let primary = stat.parent.is_none();
        inner.distinct.release(&mut stat.blocks, primary, stat.home);
        inner.ended += 1;
        Some(summary)
    }

    /// End every session still open, for example when a replay finishes. Returns the
    /// summaries of the sessions ended by this call.
    pub fn end_all(&self, now_ms: u64) -> Vec<SessionSummary> {
        let mut inner = self.inner.lock();
        let mut summaries = Vec::new();
        let Inner {
            sessions, distinct, ..
        } = &mut *inner;
        for (session_id, stat) in sessions.iter_mut() {
            if stat.ended_ms.is_none() {
                stat.ended_ms = Some(now_ms);
                summaries.push(stat.summary(session_id, now_ms));
            }
            let primary = stat.parent.is_none();
            distinct.release(&mut stat.blocks, primary, stat.home);
        }
        inner.ended += summaries.len() as u64;
        summaries
    }

    /// End sessions idle past the TTL and forget ones that ended a TTL ago. Returns
    /// the summaries of sessions ended by this call.
    pub fn sweep(&self, now_ms: u64) -> Vec<SessionSummary> {
        let ttl = self.config.idle_ttl_ms;
        let mut inner = self.inner.lock();
        let mut summaries = Vec::new();
        let window = self.config.active_window_ms;
        let Inner {
            sessions, distinct, ..
        } = &mut *inner;
        for (session_id, stat) in sessions.iter_mut() {
            if stat.ended_ms.is_none() && now_ms.saturating_sub(stat.last_activity_ms()) > ttl {
                stat.ended_ms = Some(now_ms);
                summaries.push(stat.summary(session_id, now_ms));
            }
            // A session idle past the active window no longer counts toward demand.
            if !stat.is_active(now_ms, window) {
                let primary = stat.parent.is_none();
                distinct.release(&mut stat.blocks, primary, stat.home);
            }
        }
        inner.ended += summaries.len() as u64;
        inner.sessions.retain(|_, stat| {
            stat.ended_ms
                .is_none_or(|ended| now_ms.saturating_sub(ended) <= ttl)
        });
        summaries
    }

    /// Working set, thrash and turn-gap outcomes at `now_ms`, joined with where the
    /// index says each session's prefix is resident.
    pub fn snapshot(&self, residency: &ResidencySnapshot, now_ms: u64) -> SessionStatsSnapshot {
        let block = u64::from(self.config.block_size);
        let inner = self.inner.lock();
        let mut snapshot = SessionStatsSnapshot {
            now_ms,
            sessions_started: inner.started,
            sessions_ended: inner.ended,
            turns: inner.turns,
            resident_device_tokens: residency.summed.device * block,
            resident_host_tokens: residency.summed.host * block,
            resident_disk_tokens: residency.summed.disk * block,
            distinct_device_tokens: residency.distinct.device * block,
            distinct_host_tokens: residency.distinct.host * block,
            distinct_disk_tokens: residency.distinct.disk * block,
            stranded_device_tokens: residency.stranded.device * block,
            stranded_host_tokens: residency.stranded.host * block,
            replicated_device_tokens: residency.replicated.device * block,
            replicated_host_tokens: residency.replicated.host * block,
            split_sessions: residency.split_sessions,
            thrash: inner.thrash,
            gap_outcomes: inner.gap_outcomes.to_vec(),
            ..SessionStatsSnapshot::default()
        };
        let (mut primary_output, mut subagent_output) = (0u64, 0u64);
        let mut worker_output: HashMap<WorkerWithDpRank, u64> = HashMap::new();
        for stat in inner.sessions.values() {
            let demand = stat.demand.max(stat.inflight_isl);
            if stat.is_active(now_ms, self.config.active_window_ms) {
                snapshot.active_sessions += 1;
                snapshot.demand_tokens += demand;
                // The prompt blocks are counted once in `distinct`; the latest output
                // is not part of the next prompt's hashes, so it is added per session.
                let output = stat.demand.saturating_sub(stat.inflight_isl);
                if stat.parent.is_some() {
                    snapshot.active_subagent_sessions += 1;
                    subagent_output += output;
                } else {
                    snapshot.active_primary_sessions += 1;
                    primary_output += output;
                }
                if let Some(home) = stat.home {
                    *worker_output.entry(home).or_default() += output;
                }
            }
            for (slot, window) in snapshot
                .demand_window_tokens
                .iter_mut()
                .zip(DEMAND_WINDOWS_MS)
            {
                if stat.is_active(now_ms, window) {
                    *slot += demand;
                }
            }
        }
        let distinct_blocks = inner.distinct.all.len() as u64;
        let primary_blocks = inner.distinct.primary.len() as u64;
        snapshot.demand_distinct_primary_tokens = primary_blocks * block + primary_output;
        snapshot.demand_distinct_subagent_tokens =
            (distinct_blocks - primary_blocks) * block + subagent_output;
        snapshot.demand_distinct_tokens =
            snapshot.demand_distinct_primary_tokens + snapshot.demand_distinct_subagent_tokens;
        snapshot.shared_tokens = inner.distinct.references.saturating_sub(distinct_blocks) * block;
        for (worker, blocks) in &inner.distinct.by_worker {
            let output = worker_output.get(worker).copied().unwrap_or(0);
            snapshot
                .home_demand
                .insert(*worker, blocks.len() as u64 * block + output);
        }
        snapshot
    }

    /// Sessions currently counted in the working set. A session absent from this set,
    /// including one the tracker has forgotten, is inactive; pass the complement to
    /// [`SessionPrefixIndexer::residency_snapshot_with`] so the snapshot can separate
    /// capacity held for sessions nobody is waiting on.
    ///
    /// [`SessionPrefixIndexer::residency_snapshot_with`]: crate::SessionPrefixIndexer::residency_snapshot_with
    pub fn active_sessions(&self, now_ms: u64) -> HashSet<SessionId> {
        self.inner
            .lock()
            .sessions
            .iter()
            .filter(|(_, stat)| stat.is_active(now_ms, self.config.active_window_ms))
            .map(|(session_id, _)| session_id.clone())
            .collect()
    }

    /// The worker that served each active session's last turn.
    pub fn active_homes(&self, now_ms: u64) -> HashMap<SessionId, WorkerWithDpRank> {
        self.inner
            .lock()
            .sessions
            .iter()
            .filter(|(_, stat)| stat.is_active(now_ms, self.config.active_window_ms))
            .filter_map(|(session_id, stat)| stat.home.map(|home| (session_id.clone(), home)))
            .collect()
    }

    pub fn session_count(&self) -> usize {
        self.inner.lock().sessions.len()
    }
}

impl Inner {
    fn evict_least_recent(&mut self) {
        let victim = self
            .sessions
            .iter()
            .min_by_key(|(_, stat)| stat.last_activity_ms())
            .map(|(session_id, _)| session_id.clone());
        if let Some(victim) = victim
            && let Some(mut stat) = self.sessions.remove(&victim)
        {
            let primary = stat.parent.is_none();
            self.distinct.release(&mut stat.blocks, primary, stat.home);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracker() -> SessionStatsTracker {
        SessionStatsTracker::new(SessionStatsConfig {
            block_size: 16,
            active_window_ms: 60_000,
            idle_ttl_ms: 120_000,
            max_sessions: 64,
        })
    }

    fn blocks(device: u64, host: u64, disk: u64) -> TierBlocks {
        TierBlocks { device, host, disk }
    }

    #[test]
    fn first_turn_has_no_prior_prefix_to_lose() {
        let tracker = tracker();
        let outcome = tracker.on_arrival("s", None, 1_000, TierBlocks::default(), 0);
        assert!(!outcome.returning);
        assert_eq!(outcome.thrash, ThrashTokens::default());
        assert_eq!(outcome.outcome, None);
    }

    #[test]
    fn full_device_hit_is_not_thrash() {
        let tracker = tracker();
        tracker.on_arrival("s", None, 1_000, TierBlocks::default(), 0);
        tracker.on_complete("s", 200, 0, 5_000);
        // The prior prefix is the previous turn's 1_000 input tokens (the 200 it generated
        // are not counted), 62 blocks of 16 plus a partial one, all on the device.
        let outcome = tracker.on_arrival("s", None, 1_500, blocks(63, 0, 0), 9_000);
        assert!(outcome.returning);
        assert_eq!(outcome.prior_tokens, 1_000);
        assert_eq!(outcome.gap_ms, Some(4_000));
        assert_eq!(outcome.thrash.total(), 0);
        assert_eq!(outcome.outcome, Some(ReturnOutcome::HitDevice));
    }

    #[test]
    fn thrash_splits_by_the_tier_that_still_holds_the_prefix() {
        let tracker = tracker();
        tracker.on_arrival("s", None, 1_600, TierBlocks::default(), 0);
        tracker.on_complete("s", 0, 0, 1_000);
        // Prior 1_600 tokens: 20 blocks on device, 60 in host, 80 on disk.
        let outcome = tracker.on_arrival("s", None, 2_000, blocks(20, 60, 80), 2_000);
        assert_eq!(
            outcome.thrash,
            ThrashTokens {
                recomputed: 1_600 - 1_280,
                reloaded_host: 960 - 320,
                reloaded_disk: 1_280 - 960,
            }
        );
        assert_eq!(outcome.outcome, Some(ReturnOutcome::Miss));
    }

    #[test]
    fn a_missing_tail_within_a_tenth_still_counts_as_a_hit() {
        let tracker = tracker();
        tracker.on_arrival("s", None, 1_600, TierBlocks::default(), 0);
        tracker.on_complete("s", 0, 0, 1);
        // 1_600 prior tokens; the host holds 1_440 of them, nothing on the GPU.
        let outcome = tracker.on_arrival("s", None, 1_700, blocks(0, 90, 0), 2);
        assert_eq!(outcome.outcome, Some(ReturnOutcome::HitHost));
        assert_eq!(outcome.thrash.recomputed, 160);
    }

    #[test]
    fn evicted_everywhere_is_all_recompute() {
        let tracker = tracker();
        tracker.on_arrival("s", None, 800, TierBlocks::default(), 0);
        tracker.on_complete("s", 0, 0, 1);
        let outcome = tracker.on_arrival("s", None, 900, TierBlocks::default(), 2);
        assert_eq!(outcome.thrash.recomputed, 800);
        assert_eq!(outcome.outcome, Some(ReturnOutcome::Miss));
    }

    #[test]
    fn gap_outcomes_bucket_by_gap_and_outcome() {
        let tracker = tracker();
        tracker.on_arrival("s", None, 800, TierBlocks::default(), 0);
        tracker.on_complete("s", 0, 0, 0);
        tracker.on_arrival("s", None, 800, blocks(50, 0, 0), 3_000);
        let snapshot = tracker.snapshot(&ResidencySnapshot::default(), 3_000);
        // 3s falls in the first bucket (up to 1 minute), outcome device.
        assert_eq!(snapshot.gap_outcomes[0][0], 1);
        assert_eq!(snapshot.gap_outcomes.len(), GAP_BUCKET_EDGES_MS.len() + 1);
    }

    #[test]
    fn gap_buckets_include_their_upper_edge() {
        assert_eq!(gap_bucket(0), 0);
        assert_eq!(gap_bucket(60_000), 0);
        assert_eq!(gap_bucket(60_001), 1);
        assert_eq!(gap_bucket(300_000), 1);
        assert_eq!(gap_bucket(900_000), 2);
        assert_eq!(gap_bucket(900_001), 3);
        assert_eq!(GAP_BUCKET_LABELS.len(), GAP_BUCKET_EDGES_MS.len() + 1);
    }

    #[test]
    fn subagent_fan_out_counts_on_the_parent() {
        let tracker = tracker();
        tracker.on_arrival("root", None, 100, TierBlocks::default(), 0);
        tracker.on_arrival("child-a", Some("root"), 100, TierBlocks::default(), 1);
        tracker.on_arrival("child-b", Some("root"), 100, TierBlocks::default(), 2);
        let summary = tracker.end_session("root", 10).unwrap();
        assert_eq!(summary.children, 2);
        let child = tracker.end_session("child-a", 10).unwrap();
        assert_eq!(child.parent_session_id.as_deref(), Some("root"));
    }

    #[test]
    fn end_session_reports_once_and_resumes_on_next_turn() {
        let tracker = tracker();
        tracker.on_arrival("s", None, 100, TierBlocks::default(), 0);
        tracker.on_complete("s", 50, 40, 5);
        let summary = tracker.end_session("s", 6).unwrap();
        assert_eq!(summary.turns, 1);
        assert_eq!(summary.osl_tokens, 50);
        assert!((summary.hit_rate - 0.4).abs() < 1e-9);
        assert!(tracker.end_session("s", 7).is_none());
        let outcome = tracker.on_arrival("s", None, 200, TierBlocks::default(), 8);
        assert!(outcome.started && outcome.returning);
        let second = tracker.end_session("s", 9).unwrap();
        assert_eq!(second.turns, 1, "a resumed session reports its own episode");
        assert_eq!(second.isl_tokens, 200);
    }

    #[test]
    fn idle_sessions_end_on_sweep_and_are_forgotten_a_ttl_later() {
        let tracker = tracker();
        tracker.on_arrival("s", None, 100, TierBlocks::default(), 0);
        assert!(tracker.sweep(100_000).is_empty());
        let ended = tracker.sweep(130_000);
        assert_eq!(ended.len(), 1);
        assert_eq!(tracker.session_count(), 1);
        assert!(tracker.sweep(300_000).is_empty());
        assert_eq!(tracker.session_count(), 0);
    }

    #[test]
    fn snapshot_separates_active_demand_from_stranded_residency() {
        let tracker = tracker();
        tracker.on_arrival("live", None, 1_000, TierBlocks::default(), 0);
        tracker.on_complete("live", 100, 0, 1_000);
        tracker.on_arrival("gone", None, 500, TierBlocks::default(), 0);
        tracker.on_complete("gone", 0, 0, 10);
        tracker.end_session("gone", 20);

        let active = tracker.active_sessions(2_000);
        assert!(active.contains("live") && !active.contains("gone"));
        assert_eq!(active.len(), 1);

        let residency = ResidencySnapshot {
            summed: blocks(100, 10, 0),
            distinct: blocks(90, 10, 0),
            stranded: blocks(30, 10, 0),
            replicated: blocks(12, 4, 0),
            ..ResidencySnapshot::default()
        };

        let snapshot = tracker.snapshot(&residency, 2_000);
        assert_eq!(snapshot.active_sessions, 1);
        assert_eq!(snapshot.demand_tokens, 1_100);
        assert_eq!(snapshot.resident_device_tokens, 1_600);
        assert_eq!(snapshot.distinct_device_tokens, 1_440);
        assert_eq!(snapshot.stranded_device_tokens, 480);
        assert_eq!(snapshot.stranded_host_tokens, 160);
        assert_eq!(snapshot.replicated_device_tokens, 192);
        assert_eq!(snapshot.replicated_host_tokens, 64);
        assert_eq!(snapshot.sessions_started, 2);
        assert_eq!(snapshot.sessions_ended, 1);
    }

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(id, 0)
    }

    #[test]
    fn distinct_demand_counts_a_shared_prefix_once_and_splits_by_role() {
        let tracker = tracker();
        // The primary presents 8 blocks. Its subagent starts from the first 5 and adds 2.
        let primary: Vec<u64> = (1..=8).collect();
        let sub: Vec<u64> = (1..=5).chain([100, 101]).collect();
        tracker.on_arrival("main", None, 8 * 16, TierBlocks::default(), 0);
        tracker.on_routed("main", worker(1), &primary);
        tracker.on_arrival(
            "main/sub",
            Some("main"),
            7 * 16,
            TierBlocks::default(),
            50_000,
        );
        tracker.on_routed("main/sub", worker(2), &sub);

        let snapshot = tracker.snapshot(&ResidencySnapshot::default(), 55_000);
        assert_eq!(snapshot.active_primary_sessions, 1);
        assert_eq!(snapshot.active_subagent_sessions, 1);
        // The sum counts the shared 5 blocks twice; the distinct count does not.
        assert_eq!(snapshot.demand_tokens, 8 * 16 + 7 * 16);
        assert_eq!(snapshot.demand_distinct_tokens, 10 * 16);
        assert_eq!(snapshot.demand_distinct_primary_tokens, 8 * 16);
        assert_eq!(snapshot.demand_distinct_subagent_tokens, 2 * 16);
        assert_eq!(snapshot.shared_tokens, 5 * 16);
        assert_eq!(snapshot.home_demand[&worker(1)], 8 * 16);
        assert_eq!(snapshot.home_demand[&worker(2)], 7 * 16);
        // The primary was last active 55 s ago, the subagent 5 s ago.
        assert_eq!(snapshot.demand_window_tokens, [15 * 16, 15 * 16, 15 * 16]);

        // The primary idles past the 60 s window: its blocks stop counting, the
        // subagent's shared ones now count in full.
        tracker.sweep(100_000);
        let later = tracker.snapshot(&ResidencySnapshot::default(), 100_000);
        assert_eq!(later.demand_distinct_tokens, 7 * 16);

        // A prompt that extends its previous one only adds the new blocks.
        tracker.on_arrival(
            "main/sub",
            Some("main"),
            9 * 16,
            TierBlocks::default(),
            121_000,
        );
        tracker.on_routed(
            "main/sub",
            worker(2),
            &sub.iter().copied().chain([102, 103]).collect::<Vec<_>>(),
        );
        assert_eq!(
            tracker
                .snapshot(&ResidencySnapshot::default(), 121_000)
                .demand_distinct_tokens,
            9 * 16
        );
        // Ending the session releases everything.
        tracker.end_session("main/sub", 122_000);
        assert_eq!(
            tracker
                .snapshot(&ResidencySnapshot::default(), 122_000)
                .demand_distinct_tokens,
            0
        );
    }

    #[test]
    fn worker_demand_is_distinct_and_follows_a_session_that_moves() {
        let tracker = tracker();
        // Two sessions on worker 1 share their first three blocks.
        tracker.on_arrival("a", None, 4 * 16, TierBlocks::default(), 0);
        tracker.on_routed("a", worker(1), &[1, 2, 3, 4]);
        tracker.on_arrival("b", None, 4 * 16, TierBlocks::default(), 0);
        tracker.on_routed("b", worker(1), &[1, 2, 3, 9]);
        let snapshot = tracker.snapshot(&ResidencySnapshot::default(), 1_000);
        // Five distinct blocks, not eight.
        assert_eq!(snapshot.home_demand[&worker(1)], 5 * 16);

        // Session b moves to worker 2: its need leaves worker 1 and lands on worker 2.
        tracker.on_arrival("b", None, 5 * 16, TierBlocks::default(), 2_000);
        tracker.on_routed("b", worker(2), &[1, 2, 3, 9, 10]);
        let moved = tracker.snapshot(&ResidencySnapshot::default(), 2_000);
        assert_eq!(moved.home_demand[&worker(1)], 4 * 16);
        assert_eq!(moved.home_demand[&worker(2)], 5 * 16);
        assert_eq!(tracker.active_homes(2_000)["b"], worker(2));

        // Ending both sessions leaves no worker holding need.
        tracker.end_session("a", 3_000);
        tracker.end_session("b", 3_000);
        let ended = tracker.snapshot(&ResidencySnapshot::default(), 3_000);
        assert!(ended.home_demand.is_empty());
    }

    #[test]
    fn session_cap_evicts_the_least_recently_active() {
        let tracker = SessionStatsTracker::new(SessionStatsConfig {
            max_sessions: 2,
            ..SessionStatsConfig::default()
        });
        tracker.on_arrival("a", None, 10, TierBlocks::default(), 0);
        tracker.on_arrival("b", None, 10, TierBlocks::default(), 5);
        tracker.on_arrival("c", None, 10, TierBlocks::default(), 9);
        assert_eq!(tracker.session_count(), 2);
        assert!(tracker.end_session("a", 10).is_none());
        assert!(tracker.end_session("b", 10).is_some());
    }
}
