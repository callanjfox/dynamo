// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dedicated priority workers: a worker-selection policy that keeps priority requests on a
//! reserved set of workers and the other classes off it (the simulator's `priorityWorkers`).
//!
//! It is the router's default policy plus one scorer ([`PriorityPoolScorer`]): a worker on the
//! wrong side of the pool for the request's class costs `penalty` more. With a large penalty the
//! pool is hard (a request leaves its side only when no worker on its side is a candidate, so the
//! candidate set is never emptied, unlike a `WorkerFilter`); with a small one it is a preference;
//! with zero it changes nothing, which is how the replay checks the policy against the plain
//! router.
//!
//! The request's class and the pool size come from [`PoolState`], which the session scheduler's
//! classifier feeds: at classify it records the request's class and input tokens, and at
//! completion it drops them. The selection context carries the request id, so the scorer reads the
//! class by id; no router change is needed to pass the class (ledger L57). The pool is the `size`
//! lowest worker ids. Its size is either fixed, or observed: priority's share of in-flight input
//! tokens, smoothed over `window_s`, times `multiplier`, resized at most every `resize_s`
//! (`docs/decision-rules.md`: the share of sessions undersizes the pool).
//!
//! f58pool, both default off:
//! - `demand: admitted` sizes the pool from admitted demand: a request's tokens count from the
//!   moment it is sent to a worker until it completes, not from classify. With `offered` (the
//!   default) a request held in the gate's lobby counts too, so classes the gate does not admit
//!   under overload (standard and batch waiting in the lobby) shrink priority's share and the pool,
//!   and the workers left out of the pool idle while priority queues for the pool workers.
//! - `spill_below: x` lets a priority request use a worker outside the pool when that worker's
//!   active KV (the router's decode blocks over its KV blocks) is under `x` and under every pool
//!   candidate's, so a worker kept for classes the gate is not admitting does not idle.
//! - `DYNAMO_POOL_LOG=<path>`: one JSON line a minute (pool size, share, offered and admitted
//!   tokens per class side, resizes, spills). Logging only; it changes no decision.
//!
//! f68pool, all default off:
//! - `exclude_others: true` makes the pool a constraint for the other classes: a standard or
//!   batch request never goes to a pool worker while a worker outside the pool is a candidate.
//!   The wrong side costs [`HARD_PENALTY`], above anything the forecast route adds (its excluded
//!   rows cost 1e15 and each predicted miss 1e7, which outbid the 1e9 penalty before), and the
//!   forecast route prices only workers outside the pool for such a request (`route.rs`): its
//!   cheapest set and the session's home are taken from them, so a home the pool grew over is
//!   left at the session's next turn (lazy re-home; the prefix is then fetched where the route
//!   prices it so). Priority keeps the soft rule (its home outside the pool stays, so a pool
//!   shrink does not force re-prefills), and the route adds its cheapest pool worker when its
//!   cheapest set holds none. With no worker outside the pool, the old rule applies.
//! - `share_seed: window_mean` seeds the smoothed share at the first resize with the time-weighted
//!   mean of the measured share over the first window, not with the first request's share (1.0
//!   when it is priority, so the pool was 7 of 8 workers at 5-15 min at m150 HBM4x).
//!   `prior_share: x` instead starts the share at `x` and sizes the pool from it before the
//!   first resize (in place of `initial_fraction`).
//! - `warmup_s: s` counts admitted demand for the first `s` seconds whatever `demand` says
//!   (queued primers in the lobby depress the offered share at start-up).
//! - The log line also counts sends per class and side of the pool (`sent_in`, `sent_out`; the
//!   leak) and the forecast route's placement overrides under `exclude_others` (`ovr`).
//!
//! f71pool, both default off:
//! - `size_rule: balanced` sizes an exclusive pool from both sides of the measured demand (the
//!   same token sums `demand` selects): the pool is `ceil(n x priority share)`, at least 1, and
//!   the workers outside it are at least `ceil(n x other share)`. When the two do not fit in `n`
//!   (the share times `n` is not whole) the workers are split in proportion to demand
//!   (`round(n x share)`, at least 1 and, while the other classes have demand, at most `n - 1`).
//!   `multiplier` does not apply. With `exclude_others` the multiplier rule held two of four
//!   workers for a 0.2-0.3 share, so standard had half the workers for about 70% of the demand.
//! - `headroom: {..}` moves the pool by whole workers around the rule's size, at each resize:
//!   one more when the attainment of priority turns completed since the last resize (TTFT from
//!   arrival and mean ITL within `ttft_s` / `itl_s`) is under `grow_below`, one fewer when it is
//!   at least `shrink_above` and the other classes have demand waiting (offered over admitted).
//!   Turns are counted only after `min_turns` of them; the size stays in the rule's bounds.

use std::collections::HashMap;
use std::io::Write as _;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use dynamo_kv_router::plugins::prelude::*;
use parking_lot::Mutex;
use serde::Deserialize;

use crate::clock::Time;

/// The `worker_selection` type this crate registers.
pub const PRIORITY_POOL_TYPE: &str = "priority-pool";

/// In-flight entries kept at most; beyond this new requests are not tracked (they score zero).
const MAX_IN_FLIGHT: usize = 1 << 20;

/// Log cadence of `DYNAMO_POOL_LOG`, seconds.
const LOG_EVERY_S: f64 = 60.0;

/// f68pool `exclude_others`: the cost of a pool worker for a standard or batch request. Above
/// the forecast route's excluded rows (1e15) plus anything it adds to them, so no route cost can
/// outbid it, and small enough that the default cost still orders rows that all carry it.
pub const HARD_PENALTY: f64 = 1e16;

/// f68pool: how the smoothed share is seeded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolSeed {
    /// The first sample's share (the original rule).
    #[default]
    First,
    /// The time-weighted mean of the measured share over the first window, at the first resize.
    WindowMean,
}

/// f68pool `share_seed: window_mean`: within the first window the pool follows the running mean
/// at most this often, and only after this much measurement, seconds.
const WARM_RESIZE_S: f64 = 60.0;

/// f68pool: class kinds the send counters split by.
const KINDS: [&str; 3] = ["priority", "standard", "batch"];

/// f58pool: which demand sizes an observed pool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolDemand {
    /// Every request from classify to completion, lobby waits included (the original rule).
    #[default]
    Offered,
    /// Only requests sent to a worker, until they complete.
    Admitted,
}

/// f71pool: how an observed share becomes a pool size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SizeRule {
    /// `ceil(n x share x multiplier)` (the original rule).
    #[default]
    Multiplier,
    /// Both sides of demand: see the module notes.
    Balanced,
}

/// f71pool `headroom`: the pool follows priority's measured attainment.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct Headroom {
    /// A priority turn meets its target when its TTFT (from arrival) is within this, seconds.
    pub ttft_s: f64,
    /// ... and its mean inter-token latency within this, seconds.
    pub itl_s: f64,
    /// Grow the pool by one worker when priority's attainment is under this.
    pub grow_below: f64,
    /// Shrink it by one when priority's attainment is at least this and other demand waits.
    pub shrink_above: f64,
    /// Least completed priority turns in a resize interval for a move.
    pub min_turns: u64,
}

impl Default for Headroom {
    fn default() -> Self {
        Self {
            ttft_s: 3.0,
            itl_s: 0.05,
            grow_below: 0.92,
            shrink_above: 0.97,
            min_turns: 20,
        }
    }
}

/// How the pool is sized and which class it serves.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PoolConfig {
    /// The shared-state name the classifier feeds (one per policy instance).
    pub name: String,
    /// A fixed pool size in workers; `None` sizes it from observed load.
    pub size: Option<usize>,
    /// Pool size over priority's share of workers (observed sizing).
    pub multiplier: f64,
    /// Least time between two pool resizes, seconds.
    pub resize_s: f64,
    /// Smoothing window of the observed share, seconds.
    pub window_s: f64,
    /// Share used before any load is observed.
    pub initial_fraction: f64,
    /// The policy class the pool serves.
    pub priority_class: String,
    /// Cost added for a worker on the wrong side of the pool.
    pub penalty: f64,
    /// Seed of the picker's random stream (tie-breaks); `None` draws from entropy. Offline replay
    /// sets AISimulate's canonical selector seed so a zero penalty reproduces the plain router.
    pub seed: Option<u64>,
    /// f48h: workers always left outside the pool. The default 1 keeps one worker for the
    /// other classes even when there are none (on priority-only traffic priority then gets
    /// n - 1 workers); 0 lets the pool cover every worker when priority carries the load.
    pub min_unreserved: usize,
    /// f58pool: the demand that sizes an observed pool (default `offered`).
    pub demand: PoolDemand,
    /// f58pool: a priority request may use a worker outside the pool whose active KV fraction is
    /// below this and below every pool candidate's (`None`: never).
    pub spill_below: Option<f64>,
    /// f68pool: standard and batch never use a pool worker while a worker outside it is a
    /// candidate (see the module notes).
    pub exclude_others: bool,
    /// f68pool: how the smoothed share is seeded (default `first`).
    pub share_seed: PoolSeed,
    /// f68pool: a prior for priority's share: the smoothed share starts here and sizes the pool
    /// until the first resize (`None`: `initial_fraction` and the measured seed).
    pub prior_share: Option<f64>,
    /// f68pool: count admitted demand for this many seconds after the first request, whatever
    /// `demand` says (`None`: never).
    pub warmup_s: Option<f64>,
    /// f71pool: how the share becomes a size (default `multiplier`).
    pub size_rule: SizeRule,
    /// f71pool: move the pool with priority's measured attainment (`None`: never).
    pub headroom: Option<Headroom>,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            name: "default".into(),
            size: None,
            multiplier: 1.5,
            resize_s: 600.0,
            window_s: 300.0,
            initial_fraction: 0.2,
            priority_class: "priority".into(),
            penalty: 1e9,
            seed: None,
            min_unreserved: 1,
            demand: PoolDemand::Offered,
            spill_below: None,
            exclude_others: false,
            share_seed: PoolSeed::First,
            prior_share: None,
            warmup_s: None,
            size_rule: SizeRule::Multiplier,
            headroom: None,
        }
    }
}

/// f71pool `size_rule: balanced`: the pool size for priority's `share` of demand over `workers`
/// (before the `min_unreserved` bound).
fn balanced_size(workers: usize, share: f64) -> usize {
    let n = workers as f64;
    let share = share.clamp(0.0, 1.0);
    let pool = ((n * share - 1e-9).ceil() as usize).max(1);
    let others = (n * (1.0 - share) - 1e-9).ceil() as usize;
    if pool + others <= workers {
        return pool.min(workers);
    }
    // The two sides do not fit: split in proportion to demand, one worker each at least.
    let split = ((n * share).round() as usize).max(1);
    if others > 0 {
        split.min(workers.saturating_sub(1).max(1))
    } else {
        split.min(workers)
    }
}

/// One tracked request: its class side, input tokens and whether it was sent to a worker.
#[derive(Clone, Copy)]
struct Entry {
    priority: bool,
    tokens: u64,
    sent: bool,
    /// f68pool: index into [`KINDS`] (logging only).
    kind: u8,
    /// f71pool `headroom`: when the request first arrived and its first token.
    arrived_at: Time,
    first_token: Option<Time>,
}

/// Token sums of one demand view.
#[derive(Clone, Copy, Default)]
struct Sums {
    priority: u64,
    total: u64,
}

impl Sums {
    fn add(&mut self, priority: bool, tokens: u64) {
        self.total += tokens;
        if priority {
            self.priority += tokens;
        }
    }

    fn sub(&mut self, priority: bool, tokens: u64) {
        self.total = self.total.saturating_sub(tokens);
        if priority {
            self.priority = self.priority.saturating_sub(tokens);
        }
    }
}

#[derive(Default)]
struct Load {
    in_flight: HashMap<String, Entry>,
    /// Every tracked request (classify to completion).
    offered: Sums,
    /// Tracked requests sent to a worker.
    admitted: Sums,
    share: Option<f64>,
    first_at: Option<Time>,
    sampled_at: Option<Time>,
    resized_at: Option<Time>,
    /// The fleet size at the last arrival (for samples taken at a send).
    workers: u64,
    logged_at: Option<Time>,
    /// f68pool `share_seed: window_mean`: the time-weighted sum of the measured share over the
    /// first window, its seconds, and the last sample (time, share).
    warm_sum: f64,
    warm_s: f64,
    warm_last: Option<(Time, f64)>,
    /// When the pool last followed the running mean within the first window.
    warm_sized_at: Option<Time>,
    /// f71pool `headroom`: workers added to (or, negative, taken from) the rule's size, and the
    /// priority turns completed since the last resize (all, and those on target).
    extra: i64,
    turns: u64,
    turns_ok: u64,
}

/// f68pool: why the forecast route changed its candidates for the pool (`exclude_others`).
#[derive(Clone, Copy, Debug)]
pub enum Override {
    /// A standard or batch request's home was a pool worker; it was dropped (re-home).
    HomeInPool,
    /// The route's cheapest set held pool workers; workers outside the pool replaced them.
    CutHadPool,
    /// The route's cheapest set was all pool workers.
    CutAllPool,
    /// An ITL or step-budget rule would have let a pool worker back in.
    RuleMasked,
    /// A priority request's cheapest set held no pool worker; its cheapest one was added.
    PriorityPoolAdded,
    /// No worker outside the pool was a candidate: the old rule applied.
    NoOutside,
}

const OVERRIDES: usize = 6;
const OVERRIDE_NAMES: [&str; OVERRIDES] = [
    "home_in_pool",
    "cut_had_pool",
    "cut_all_pool",
    "rule_masked",
    "priority_pool_added",
    "no_outside",
];

/// f68pool: the side of the pool for one request, as the forecast route reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct PoolSide {
    /// Per candidate (in the order given): whether it is on the request's own side.
    pub own: Vec<bool>,
    /// Whether the wrong side is excluded (standard and batch under `exclude_others`); otherwise
    /// the route only makes sure its own side is priced (priority).
    pub hard: bool,
}

/// What the classifier tells the pool, and what the scorer reads.
pub struct PoolState {
    config: PoolConfig,
    load: Mutex<Load>,
    /// Observed pool size; 0 until the first resize.
    observed: AtomicUsize,
    resizes: AtomicU64,
    unknown: AtomicU64,
    scored: AtomicU64,
    /// f58pool: selections where a priority request could use a worker outside the pool.
    spilled: AtomicU64,
    log: Mutex<Option<std::fs::File>>,
    /// f68pool: the highest pool worker id at the last selection the pool acted in (`u64::MAX`:
    /// none yet), for the send counters.
    edge: AtomicU64,
    /// f68pool: sends per class kind to a pool worker and to one outside it (logging only).
    sent_in: [AtomicU64; 3],
    sent_out: [AtomicU64; 3],
    /// f68pool: the forecast route's overrides under `exclude_others`.
    overrides: [AtomicU64; OVERRIDES],
    /// f68pool: selections where the pool scorer charged [`HARD_PENALTY`].
    hard: AtomicU64,
    /// f50: the mix selector's modes the pool acts in (`None`: every mode), set by the
    /// placement policy so the forecast route reads the same rule as the scorer.
    modes: Mutex<Option<Vec<crate::mix_select::MixMode>>>,
}

impl PoolState {
    pub fn new(config: PoolConfig) -> Self {
        Self {
            config,
            load: Mutex::new(Load::default()),
            observed: AtomicUsize::new(0),
            resizes: AtomicU64::new(0),
            unknown: AtomicU64::new(0),
            scored: AtomicU64::new(0),
            spilled: AtomicU64::new(0),
            log: Mutex::new(None),
            edge: AtomicU64::new(u64::MAX),
            sent_in: Default::default(),
            sent_out: Default::default(),
            overrides: Default::default(),
            hard: AtomicU64::new(0),
            modes: Mutex::new(None),
        }
    }

    /// f68pool: the mix selector's modes the pool acts in (from the placement policy).
    pub fn set_modes(&self, modes: Option<Vec<crate::mix_select::MixMode>>) {
        *self.modes.lock() = modes;
    }

    /// Whether the pool acts now: the adaptive place group is on and the mix selector's mode is
    /// one the pool acts in.
    fn acting(&self) -> bool {
        if !crate::adaptive::place_enabled() {
            return false;
        }
        match (&*self.modes.lock(), crate::mix_select::current()) {
            (Some(modes), Some(mode)) => modes.contains(&mode),
            _ => true,
        }
    }

    /// f68pool: count one forecast-route override.
    pub fn record_override(&self, why: Override) {
        self.overrides[why as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// f68pool: the request's side of the pool over `worker_ids` (the candidates, in order), for
    /// the forecast route. `None` when `exclude_others` is off, when the pool does not act (off
    /// in this mode, unknown class, a fleet under 2), or when no worker on the request's side is
    /// a candidate (for standard and batch counted: the old rule applies).
    pub fn side(&self, request_id: Option<&str>, worker_ids: &[u64]) -> Option<PoolSide> {
        if !self.config.exclude_others || !self.acting() {
            return None;
        }
        self.side_of(request_id, worker_ids)
    }

    /// [`Self::side`] without the acting check (the adaptive switch and the mix mode are process
    /// globals other code moves).
    fn side_of(&self, request_id: Option<&str>, worker_ids: &[u64]) -> Option<PoolSide> {
        let priority = self.is_priority(request_id?)?;
        let mut ids = worker_ids.to_vec();
        ids.sort_unstable();
        ids.dedup();
        let size = self.pool_size(ids.len());
        if size == 0 {
            return None;
        }
        let edge = ids[size - 1];
        let own: Vec<bool> = worker_ids
            .iter()
            .map(|&w| (w <= edge) == priority)
            .collect();
        if !own.iter().any(|&o| o) {
            if !priority {
                self.record_override(Override::NoOutside);
            }
            return None;
        }
        Some(PoolSide {
            own,
            hard: !priority,
        })
    }

    /// Install with the `DYNAMO_POOL_LOG` file, when that is set.
    fn with_env_log(self) -> Self {
        if let Some(file) =
            std::env::var_os("DYNAMO_POOL_LOG").and_then(|p| std::fs::File::create(p).ok())
        {
            *self.log.lock() = Some(file);
        }
        self
    }

    /// The token sums the configured demand sizes the pool from (f68pool `warmup_s`: admitted
    /// demand until that long after the first request).
    fn counted(&self, load: &Load, now: Time) -> Sums {
        let warm = self.config.warmup_s.is_some_and(|w| {
            load.first_at
                .is_none_or(|first| now.since(first) < Duration::from_secs_f64(w.max(0.0)))
        });
        match self.config.demand {
            PoolDemand::Offered if !warm => load.offered,
            _ => load.admitted,
        }
    }

    /// Whether a send changes the demand the pool is sized from (admitted demand counts).
    fn counts_sends(&self) -> bool {
        self.config.demand == PoolDemand::Admitted || self.config.warmup_s.is_some()
    }

    pub fn config(&self) -> &PoolConfig {
        &self.config
    }

    /// A request entered the router: its class and input tokens. `workers` is the fleet size.
    pub fn arrived(&self, request_id: &str, class: &str, tokens: u64, now: Time, workers: u64) {
        let priority = class == self.config.priority_class;
        let kind = if priority {
            0
        } else if class == "batch" {
            2
        } else {
            1
        };
        let mut load = self.load.lock();
        if load.in_flight.len() >= MAX_IN_FLIGHT && !load.in_flight.contains_key(request_id) {
            return;
        }
        let mut entry = Entry {
            priority,
            tokens,
            sent: false,
            kind,
            arrived_at: now,
            first_token: None,
        };
        if let Some(old) = load.in_flight.insert(request_id.to_owned(), entry) {
            Self::remove(&mut load, old);
            // A request seen again after it was sent stays admitted; it keeps its first arrival
            // and first token (f71pool `headroom`).
            entry.sent = old.sent;
            entry.arrived_at = old.arrived_at;
            entry.first_token = old.first_token;
            load.in_flight.insert(request_id.to_owned(), entry);
        }
        load.offered.add(priority, tokens);
        if entry.sent {
            load.admitted.add(priority, tokens);
        }
        load.workers = workers;
        self.sample(&mut load, now, workers);
    }

    /// f58pool: the request was sent to a worker; from now its tokens count as admitted demand.
    /// f68pool: `worker` is counted against the pool's edge at the last selection (logging).
    pub fn sent(&self, request_id: &str, now: Time, worker: u64) {
        let mut load = self.load.lock();
        let Some(entry) = load.in_flight.get_mut(request_id) else {
            return;
        };
        let edge = self.edge.load(Ordering::Relaxed);
        if edge != u64::MAX {
            let side = if worker <= edge {
                &self.sent_in
            } else {
                &self.sent_out
            };
            side[usize::from(entry.kind)].fetch_add(1, Ordering::Relaxed);
        }
        if entry.sent {
            return;
        }
        entry.sent = true;
        let (priority, tokens) = (entry.priority, entry.tokens);
        load.admitted.add(priority, tokens);
        // With `offered` sizing a send changes nothing the pool decides; no sample, as before.
        if self.counts_sends() {
            let workers = load.workers;
            self.sample(&mut load, now, workers);
        }
    }

    /// f71pool `headroom`: the request's first token (the first one counts). Nothing when off.
    pub fn first_token(&self, request_id: &str, now: Time) {
        if self.config.headroom.is_none() {
            return;
        }
        if let Some(entry) = self.load.lock().in_flight.get_mut(request_id) {
            entry.first_token.get_or_insert(now);
        }
    }

    /// f71pool `headroom`: a priority request completed with `output_tokens`; it counts as on
    /// target when its TTFT from arrival and its mean ITL are within the targets. Call before
    /// [`Self::finished`]. Nothing when off.
    pub fn completed(&self, request_id: &str, output_tokens: Option<usize>, now: Time) {
        let Some(target) = &self.config.headroom else {
            return;
        };
        let mut load = self.load.lock();
        let Some(entry) = load.in_flight.get(request_id).copied() else {
            return;
        };
        if !entry.priority {
            return;
        }
        let ok = entry.first_token.is_some_and(|first| {
            let ttft = first.since(entry.arrived_at).as_secs_f64();
            let itl = match output_tokens {
                Some(out) if out > 1 => now.since(first).as_secs_f64() / (out - 1) as f64,
                _ => 0.0,
            };
            ttft <= target.ttft_s && itl <= target.itl_s
        });
        load.turns += 1;
        load.turns_ok += u64::from(ok);
    }

    /// A request left the router (completed, aborted or refused).
    pub fn finished(&self, request_id: &str) {
        let mut load = self.load.lock();
        if let Some(old) = load.in_flight.remove(request_id) {
            Self::remove(&mut load, old);
        }
    }

    fn remove(load: &mut Load, old: Entry) {
        load.offered.sub(old.priority, old.tokens);
        if old.sent {
            load.admitted.sub(old.priority, old.tokens);
        }
    }

    /// Update the smoothed share and, when due, the observed pool size.
    fn sample(&self, load: &mut Load, now: Time, workers: u64) {
        self.sample_inner(load, now, workers);
        self.maybe_log(load, now, workers);
    }

    fn sample_inner(&self, load: &mut Load, now: Time, workers: u64) {
        let counted = self.counted(load, now);
        if self.config.size.is_some() || counted.total == 0 {
            return;
        }
        let instant = counted.priority as f64 / counted.total as f64;
        load.share = Some(match (load.share, load.sampled_at) {
            (Some(share), Some(at)) => {
                let dt = now.since(at).as_secs_f64();
                let alpha = 1.0 - (-dt / self.config.window_s.max(1e-9)).exp();
                share + alpha * (instant - share)
            }
            // f68pool `prior_share`: the first sample moves the prior, not replaces it (it moves
            // it by nothing: no time has passed).
            _ => self.config.prior_share.unwrap_or(instant),
        });
        load.sampled_at = Some(now);
        let first = *load.first_at.get_or_insert(now);
        // f68pool `share_seed: window_mean`: the measured share's time-weighted mean over the
        // first window (each sample's share holds until the next sample).
        let window_mean =
            self.config.share_seed == PoolSeed::WindowMean && self.config.prior_share.is_none();
        if window_mean && load.resized_at.is_none() {
            if let Some((at, share)) = load.warm_last {
                let dt = now.since(at).as_secs_f64();
                load.warm_sum += share * dt;
                load.warm_s += dt;
            }
            load.warm_last = Some((now, instant));
        }
        // The first size waits one smoothing window, so a handful of early requests do not fix
        // the pool for a whole resize interval; until then the initial fraction applies.
        let due = match load.resized_at {
            None => now.since(first) >= Duration::from_secs_f64(self.config.window_s.max(0.0)),
            Some(at) => now.since(at) >= Duration::from_secs_f64(self.config.resize_s.max(0.0)),
        };
        if workers < 2 {
            return;
        }
        if window_mean && load.resized_at.is_none() && load.warm_s > 0.0 {
            let mean = load.warm_sum / load.warm_s;
            if due {
                // The smoothed share restarts from the first window's mean.
                load.share = Some(mean);
            } else if load.warm_s >= WARM_RESIZE_S
                && load
                    .warm_sized_at
                    .is_none_or(|at| now.since(at).as_secs_f64() >= WARM_RESIZE_S)
            {
                // Within the first window the pool follows the running mean (at most once a
                // minute, after a minute of measurement), not the initial fraction.
                self.resize_to(mean, workers, load.extra);
                load.warm_sized_at = Some(now);
            }
        }
        if due {
            let share = load.share.unwrap_or(0.0);
            self.follow_headroom(load, share, workers as usize);
            self.resize_to(share, workers, load.extra);
            load.resized_at = Some(now);
        }
    }

    /// Set the observed pool size for priority's `share` of `workers` (counted when it changes).
    fn resize_to(&self, share: f64, workers: u64, extra: i64) {
        let want = self.size_for(share, workers as usize, extra);
        if want != self.observed.load(Ordering::Relaxed) {
            self.observed.store(want, Ordering::Relaxed);
            self.resizes.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The pool size for priority's `share` of `workers`: the size rule's, plus `extra` workers
    /// (f71pool `headroom`), within 1 and `workers - min_unreserved`.
    fn size_for(&self, share: f64, workers: usize, extra: i64) -> usize {
        let most = workers.saturating_sub(self.config.min_unreserved).max(1);
        let base = match self.config.size_rule {
            SizeRule::Multiplier => (workers as f64 * share * self.config.multiplier - 1e-9)
                .ceil()
                .max(1.0) as usize,
            SizeRule::Balanced => balanced_size(workers, share),
        };
        let most = i64::try_from(most).unwrap_or(i64::MAX);
        let base = i64::try_from(base).unwrap_or(i64::MAX);
        base.saturating_add(extra).clamp(1, most) as usize
    }

    /// f71pool `headroom`: at a resize, move the extra workers by one with priority's attainment
    /// since the last resize, then start a new count. The extra is kept to what the bounds let
    /// the size take, so it does not wind up.
    fn follow_headroom(&self, load: &mut Load, share: f64, workers: usize) {
        let Some(target) = &self.config.headroom else {
            return;
        };
        let (turns, ok) = (load.turns, load.turns_ok);
        load.turns = 0;
        load.turns_ok = 0;
        if turns < target.min_turns.max(1) {
            return;
        }
        let attainment = ok as f64 / turns as f64;
        let others_offered = load.offered.total.saturating_sub(load.offered.priority);
        let others_admitted = load.admitted.total.saturating_sub(load.admitted.priority);
        if attainment < target.grow_below {
            load.extra += 1;
        } else if attainment >= target.shrink_above && others_offered > others_admitted {
            load.extra -= 1;
        }
        let base = self.size_for(share, workers, 0) as i64;
        load.extra = self.size_for(share, workers, load.extra) as i64 - base;
    }

    /// One `DYNAMO_POOL_LOG` line when a minute has passed since the last. Writes only.
    fn maybe_log(&self, load: &mut Load, now: Time, workers: u64) {
        let mut log = self.log.lock();
        let Some(file) = log.as_mut() else {
            return;
        };
        if load
            .logged_at
            .is_some_and(|at| now.since(at) < Duration::from_secs_f64(LOG_EVERY_S))
        {
            return;
        }
        load.logged_at = Some(now);
        let share = load.share.map_or("null".to_owned(), |s| format!("{s:.4}"));
        let mode = crate::mix_select::current().map_or("null".to_owned(), |m| format!("\"{m:?}\""));
        // f68pool: sends per class kind on each side of the pool, and the route's overrides.
        let per_kind = |c: &[AtomicU64; 3]| {
            let body: Vec<String> = KINDS
                .iter()
                .zip(c)
                .map(|(k, n)| format!("\"{k}\":{}", n.load(Ordering::Relaxed)))
                .collect();
            format!("{{{}}}", body.join(","))
        };
        let ovr: Vec<String> = OVERRIDE_NAMES
            .iter()
            .zip(&self.overrides)
            .map(|(k, n)| format!("\"{k}\":{}", n.load(Ordering::Relaxed)))
            .collect();
        let edge = self.edge.load(Ordering::Relaxed);
        // f71pool: the size rule and headroom's state, only when either is on.
        let f71 = if self.config.size_rule != SizeRule::Multiplier || self.config.headroom.is_some()
        {
            format!(
                ",\"rule\":\"{:?}\",\"extra\":{},\"turns\":{},\"turns_ok\":{}",
                self.config.size_rule, load.extra, load.turns, load.turns_ok
            )
        } else {
            String::new()
        };
        let _ = writeln!(
            file,
            "{{\"t_s\":{:.1},\"workers\":{},\"size\":{},\"observed\":{},\"share\":{},\"demand\":\"{:?}\",\"offered_p\":{},\"offered_t\":{},\"admitted_p\":{},\"admitted_t\":{},\"in_flight\":{},\"resizes\":{},\"scored\":{},\"unknown\":{},\"spilled\":{},\"mix_mode\":{},\"edge\":{},\"sent_in\":{},\"sent_out\":{},\"hard\":{},\"ovr\":{{{}}}{}}}",
            now.as_secs_f64(),
            workers,
            self.pool_size(workers as usize),
            self.observed.load(Ordering::Relaxed),
            share,
            self.config.demand,
            load.offered.priority,
            load.offered.total,
            load.admitted.priority,
            load.admitted.total,
            load.in_flight.len(),
            self.resizes.load(Ordering::Relaxed),
            self.scored.load(Ordering::Relaxed),
            self.unknown.load(Ordering::Relaxed),
            self.spilled.load(Ordering::Relaxed),
            mode,
            if edge == u64::MAX {
                "null".to_owned()
            } else {
                edge.to_string()
            },
            per_kind(&self.sent_in),
            per_kind(&self.sent_out),
            self.hard.load(Ordering::Relaxed),
            ovr.join(","),
            f71,
        );
    }

    /// Whether the request is of the pool's class; `None` when the classifier never saw it.
    pub fn is_priority(&self, request_id: &str) -> Option<bool> {
        self.load
            .lock()
            .in_flight
            .get(request_id)
            .map(|e| e.priority)
    }

    /// The pool size for `workers` candidates: fixed, observed, or the initial fraction.
    pub fn pool_size(&self, workers: usize) -> usize {
        if workers < 2 {
            return 0;
        }
        let want = match self.config.size {
            Some(size) => size,
            None => match self.observed.load(Ordering::Relaxed) {
                // f68pool `prior_share`: the prior sizes the pool until the first resize.
                0 => match self.config.prior_share {
                    Some(prior) => self.size_for(prior, workers, 0),
                    None => (workers as f64 * self.config.initial_fraction - 1e-9).ceil() as usize,
                },
                observed => observed,
            },
        };
        want.clamp(1, workers.saturating_sub(self.config.min_unreserved).max(1))
    }

    /// Counters for logs and tests: (selections scored, selections with an unknown class,
    /// pool resizes, current observed size).
    pub fn stats(&self) -> (u64, u64, u64, usize) {
        (
            self.scored.load(Ordering::Relaxed),
            self.unknown.load(Ordering::Relaxed),
            self.resizes.load(Ordering::Relaxed),
            self.observed.load(Ordering::Relaxed),
        )
    }
}

fn pools() -> &'static Mutex<HashMap<String, Arc<PoolState>>> {
    static POOLS: OnceLock<Mutex<HashMap<String, Arc<PoolState>>>> = OnceLock::new();
    POOLS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Install a fresh state under `config.name`, replacing an earlier one (a new router startup).
pub fn install(config: PoolConfig) -> Arc<PoolState> {
    // f62: no size published until the scorer first sizes the pool.
    dynamo_kv_router::plugins::adaptive_switches::set_priority_pool_size(None);
    let state = Arc::new(PoolState::new(config).with_env_log());
    pools()
        .lock()
        .insert(state.config.name.clone(), Arc::clone(&state));
    state
}

/// The state a priority pool policy installed under `name`, if any.
pub fn installed(name: &str) -> Option<Arc<PoolState>> {
    pools().lock().get(name).cloned()
}

/// Adds `penalty` to every worker on the wrong side of the pool for the request's class.
pub struct PriorityPoolScorer {
    state: Arc<PoolState>,
    /// f50: the mix selector's modes the pool acts in (`None`: every mode). It is fed in all.
    modes: Option<Vec<crate::mix_select::MixMode>>,
}

impl PriorityPoolScorer {
    pub fn new(state: Arc<PoolState>) -> Self {
        Self { state, modes: None }
    }

    /// f50: act only in `modes` of the mix selector.
    pub fn with_modes(mut self, modes: Vec<crate::mix_select::MixMode>) -> Self {
        self.modes = Some(modes);
        self
    }
}

impl WorkerScorer for PriorityPoolScorer {
    /// f58pool: the load rows only when `spill_below` is set (off: no inputs, as before).
    fn required_worker_inputs(&self) -> WorkerInputs {
        if self.state.config.spill_below.is_some() {
            WorkerInputs::LOAD
        } else {
            WorkerInputs::NONE
        }
    }

    fn score(
        &self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        costs.fill(0.0);
        // The adaptive rule's place group (`adaptive.rs`) switched off: today's router decides.
        // f68pool: while the pool does not act, no size is published (the router's
        // `prefill_isolation` with `workers: pool_else_active` then protects priority-running
        // workers; it reads nothing else of this).
        if !crate::adaptive::place_enabled() {
            dynamo_kv_router::plugins::adaptive_switches::set_priority_pool_size(None);
            return Ok(());
        }
        // f50: the mix selector's mode leaves the pool out.
        if let (Some(modes), Some(mode)) = (&self.modes, crate::mix_select::current())
            && !modes.contains(&mode)
        {
            dynamo_kv_router::plugins::adaptive_switches::set_priority_pool_size(None);
            return Ok(());
        }
        self.state.scored.fetch_add(1, Ordering::Relaxed);
        let Some(priority) = context
            .request_id()
            .and_then(|id| self.state.is_priority(id))
        else {
            self.state.unknown.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        };
        let mut ids: Vec<u64> = candidates.iter().map(|c| c.worker().worker_id).collect();
        ids.sort_unstable();
        ids.dedup();
        let size = self.state.pool_size(ids.len());
        // f62: the router's `kv_actions.prefill_isolation` (`workers: pool`) reads it.
        dynamo_kv_router::plugins::adaptive_switches::set_priority_pool_size(Some(size));
        if size == 0 {
            return Ok(());
        }
        let edge = ids[size - 1];
        self.state.edge.store(edge, Ordering::Relaxed);
        // f68pool `exclude_others`: a standard or batch request pays HARD_PENALTY for a pool
        // worker while a worker outside the pool is a candidate (with none, all pay the same).
        let hard = self.state.config.exclude_others && !priority && ids.len() > size;
        if hard {
            self.state.hard.fetch_add(1, Ordering::Relaxed);
        }
        let penalty = if hard {
            HARD_PENALTY
        } else {
            self.state.config.penalty
        };
        // f58pool `spill_below`: the active KV fraction of each candidate, and the least of the
        // pool candidates'. A priority request may use a worker outside the pool below both.
        let spill = self.state.config.spill_below.filter(|_| priority);
        let frac = |c: WorkerCandidate<'_>| -> Option<f64> {
            let load = c.load().filter(|l| l.available)?;
            let blocks = c.info().total_kv_blocks().filter(|&b| b > 0)?;
            Some(load.active_decode_blocks as f64 / blocks as f64)
        };
        let pool_least = spill.and_then(|_| {
            candidates
                .iter()
                .filter(|c| c.worker().worker_id <= edge)
                .map(|c| frac(c).unwrap_or(f64::INFINITY))
                .reduce(f64::min)
        });
        let mut spilled = false;
        for (candidate, cost) in candidates.iter().zip(costs.iter_mut()) {
            let in_pool = candidate.worker().worker_id <= edge;
            if in_pool != priority {
                if let (Some(below), Some(least)) = (spill, pool_least)
                    && !in_pool
                    && frac(candidate).is_some_and(|f| f < below && f < least)
                {
                    spilled = true;
                    continue;
                }
                *cost = penalty;
            }
        }
        if spilled {
            self.state.spilled.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let config: PoolConfig = parameters.deserialize()?;
    if !(config.penalty.is_finite() && config.penalty >= 0.0) {
        return Err(WorkerSelectionPolicyProviderError::new(
            "priority-pool penalty must be finite and not negative",
        ));
    }
    let rng = config
        .seed
        .map(|seed| Arc::new(Mutex::new(fastrand::Rng::with_seed(seed))));
    let state = install(config);
    let compose = crate::host_default(parameters)?;
    Ok(Arc::new(
        move |router, worker_type: WorkerType, _partition| {
            compose(
                router.clone(),
                worker_type.as_str(),
                rng.clone(),
                vec![Box::new(PriorityPoolScorer::new(Arc::clone(&state)))],
            )
        },
    ))
}

/// Register the priority pool worker-selection type.
pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(PRIORITY_POOL_TYPE, Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: f64) -> Time {
        Time::from_secs_f64(s)
    }

    #[test]
    fn fixed_size_and_initial_fraction() {
        let fixed = PoolState::new(PoolConfig {
            size: Some(3),
            ..Default::default()
        });
        assert_eq!(fixed.pool_size(8), 3);
        assert_eq!(fixed.pool_size(2), 1);
        let observed = PoolState::new(PoolConfig::default());
        assert_eq!(observed.pool_size(8), 2); // ceil(0.2 x 8) before any load
        assert_eq!(observed.pool_size(32), 7);
    }

    #[test]
    fn min_unreserved_zero_lets_the_pool_cover_every_worker() {
        let all = PoolState::new(PoolConfig {
            size: Some(4),
            min_unreserved: 0,
            ..Default::default()
        });
        assert_eq!(all.pool_size(4), 4);
        let one = PoolState::new(PoolConfig {
            size: Some(4),
            ..Default::default()
        });
        assert_eq!(one.pool_size(4), 3); // the default keeps one worker out
    }

    #[test]
    fn observed_share_sizes_the_pool_and_waits_to_resize() {
        let state = PoolState::new(PoolConfig {
            window_s: 1.0,
            ..Default::default()
        });
        // Before one window has passed, the initial fraction: ceil(0.2 x 8) = 2.
        state.arrived("s", "standard", 750, t(0.0), 8);
        state.arrived("p", "priority", 250, t(0.0), 8);
        assert_eq!(state.pool_size(8), 2);
        // Priority holds 25% of in-flight tokens (the share is settled after many windows):
        // ceil(8 x 0.25 x 1.5) = 3.
        state.arrived("s0", "standard", 0, t(100.0), 8);
        assert_eq!(state.pool_size(8), 3);
        // The share rises to about 75%, but the pool waits for the resize interval.
        state.finished("s");
        state.arrived("p2", "priority", 500, t(110.0), 8);
        state.arrived("s2", "standard", 250, t(120.0), 8);
        assert_eq!(state.pool_size(8), 3);
        state.arrived("p3", "priority", 1, t(800.0), 8);
        assert!(state.pool_size(8) > 3);
        assert_eq!(state.is_priority("p2"), Some(true));
        assert_eq!(state.is_priority("s2"), Some(false));
        assert_eq!(state.is_priority("gone"), None);
    }

    #[test]
    fn finished_releases_tokens_once() {
        let state = PoolState::new(PoolConfig::default());
        state.arrived("a", "priority", 100, t(0.0), 4);
        state.finished("a");
        state.finished("a");
        let load = state.load.lock();
        assert_eq!((load.offered.priority, load.offered.total), (0, 0));
        assert!(load.in_flight.is_empty());
    }

    #[test]
    fn admitted_demand_ignores_the_lobby() {
        // Offered: standard waiting in the lobby (never sent) holds 75% of the tokens, so the
        // pool is ceil(4 x 0.25 x 1.5) = 2. Admitted: only priority was sent, so it is all 4 - 1.
        for (demand, want) in [(PoolDemand::Offered, 2), (PoolDemand::Admitted, 3)] {
            let state = PoolState::new(PoolConfig {
                window_s: 1.0,
                demand,
                ..Default::default()
            });
            state.arrived("s", "standard", 750, t(0.0), 4);
            state.arrived("p", "priority", 250, t(0.0), 4);
            state.sent("p", t(0.0), 0);
            state.arrived("p1", "priority", 0, t(100.0), 4);
            state.sent("p1", t(100.0), 0);
            assert_eq!(state.pool_size(4), want, "{demand:?}");
            let load = state.load.lock();
            assert_eq!((load.admitted.priority, load.admitted.total), (250, 250));
            assert_eq!((load.offered.priority, load.offered.total), (250, 1000));
        }
    }

    #[test]
    fn sent_then_finished_releases_admitted_tokens() {
        let state = PoolState::new(PoolConfig {
            demand: PoolDemand::Admitted,
            ..Default::default()
        });
        state.arrived("a", "standard", 100, t(0.0), 4);
        state.sent("a", t(1.0), 0);
        state.sent("a", t(2.0), 0);
        // Seen again after the send (a recheck): still admitted, counted once.
        state.arrived("a", "standard", 100, t(3.0), 4);
        {
            let load = state.load.lock();
            assert_eq!(load.admitted.total, 100);
            assert_eq!(load.offered.total, 100);
        }
        state.finished("a");
        let load = state.load.lock();
        assert_eq!((load.admitted.total, load.offered.total), (0, 0));
    }

    #[test]
    fn config_defaults_are_off() {
        let c: PoolConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(c.demand, PoolDemand::Offered);
        assert!(c.spill_below.is_none());
        let c: PoolConfig =
            serde_json::from_str(r#"{"demand": "admitted", "spill_below": 0.5}"#).unwrap();
        assert_eq!(c.demand, PoolDemand::Admitted);
        assert_eq!(c.spill_below, Some(0.5));
    }

    #[test]
    fn f68_config_defaults_are_off() {
        let c: PoolConfig = serde_json::from_str("{}").unwrap();
        assert!(!c.exclude_others);
        assert_eq!(c.share_seed, PoolSeed::First);
        assert!(c.prior_share.is_none() && c.warmup_s.is_none());
        let c: PoolConfig = serde_json::from_str(
            r#"{"exclude_others": true, "share_seed": "window_mean", "prior_share": 0.3, "warmup_s": 3600}"#,
        )
        .unwrap();
        assert!(c.exclude_others);
        assert_eq!(c.share_seed, PoolSeed::WindowMean);
        assert_eq!((c.prior_share, c.warmup_s), (Some(0.3), Some(3600.0)));
    }

    /// A priority primer first (admitted share 1.0 for a second), then standard at 0.1.
    fn startup(seed: PoolSeed) -> (usize, usize, usize) {
        let state = PoolState::new(PoolConfig {
            window_s: 100.0,
            initial_fraction: 0.5,
            demand: PoolDemand::Admitted,
            share_seed: seed,
            ..Default::default()
        });
        state.arrived("p", "priority", 100, t(0.0), 8);
        state.sent("p", t(0.0), 0);
        state.arrived("s", "standard", 900, t(1.0), 8);
        state.sent("s", t(1.0), 7);
        let early = state.pool_size(8);
        state.arrived("y", "standard", 0, t(70.0), 8);
        let warm = state.pool_size(8);
        state.arrived("x", "standard", 0, t(100.0), 8);
        (early, warm, state.pool_size(8))
    }

    #[test]
    fn window_mean_seed_ignores_the_first_request() {
        // First: the share starts at 1.0 and has decayed to about 0.43 by the first resize:
        // ceil(8 x 0.43 x 1.5) = 6; until then the initial fraction, ceil(8 x 0.5) = 4.
        assert_eq!(startup(PoolSeed::First), (4, 4, 6));
        // Window mean: (1.0 x 1 s + 0.1 x 99 s) / 100 s = 0.109: ceil(8 x 0.109 x 1.5) = 2, and
        // from a minute of measurement the pool follows the running mean (0.113 at 70 s: 2).
        assert_eq!(startup(PoolSeed::WindowMean), (4, 2, 2));
    }

    #[test]
    fn prior_share_sizes_the_pool_before_the_first_resize() {
        let state = PoolState::new(PoolConfig {
            prior_share: Some(0.25),
            ..Default::default()
        });
        assert_eq!(state.pool_size(8), 3); // ceil(8 x 0.25 x 1.5)
        // The first sample does not replace the prior.
        state.arrived("p", "priority", 100, t(0.0), 8);
        assert_eq!(state.load.lock().share, Some(0.25));
    }

    #[test]
    fn warmup_counts_admitted_demand_under_offered() {
        let state = PoolState::new(PoolConfig {
            warmup_s: Some(50.0),
            ..Default::default()
        });
        state.arrived("s", "standard", 900, t(0.0), 4);
        state.arrived("p", "priority", 100, t(0.0), 4);
        state.sent("p", t(0.0), 0);
        // Warm-up: only the sent priority request counts.
        assert_eq!(state.load.lock().share, Some(1.0));
        // After it: offered demand again (standard waits in the lobby).
        state.arrived("q", "priority", 0, t(60.0), 4);
        let share = state.load.lock().share.unwrap();
        assert!(share < 1.0, "{share}");
    }

    #[test]
    fn side_excludes_the_pool_for_other_classes_only_when_asked() {
        let make = |exclude_others: bool, min_unreserved: usize| {
            let state = PoolState::new(PoolConfig {
                size: Some(3),
                exclude_others,
                min_unreserved,
                ..Default::default()
            });
            state.arrived("s", "standard", 1, t(0.0), 5);
            state.arrived("b", "batch", 1, t(0.0), 5);
            state.arrived("p", "priority", 1, t(0.0), 5);
            state
        };
        let ids = [5, 1, 2, 3, 4];
        assert_eq!(make(false, 1).side(Some("s"), &ids), None);
        let state = make(true, 1);
        let s = state.side_of(Some("s"), &ids).expect("standard side");
        assert!(s.hard);
        assert_eq!(s.own, vec![true, false, false, false, true]);
        assert_eq!(state.side_of(Some("b"), &ids).map(|s| s.own), Some(s.own));
        let p = state.side_of(Some("p"), &ids).expect("priority side");
        assert!(!p.hard);
        assert_eq!(p.own, vec![false, true, true, true, false]);
        assert_eq!(state.side_of(Some("unknown"), &ids), None);
        // The pool covers every candidate: no worker outside it, the old rule (counted).
        let state = make(true, 0);
        assert_eq!(state.side_of(Some("s"), &[1, 2, 3]), None);
        assert_eq!(
            state.overrides[Override::NoOutside as usize].load(Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn f71_config_defaults_are_off() {
        let c: PoolConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(c.size_rule, SizeRule::Multiplier);
        assert!(c.headroom.is_none());
        let c: PoolConfig =
            serde_json::from_str(r#"{"size_rule": "balanced", "headroom": {"grow_below": 0.9}}"#)
                .unwrap();
        assert_eq!(c.size_rule, SizeRule::Balanced);
        let h = c.headroom.unwrap();
        assert_eq!(h.grow_below, 0.9);
        assert_eq!(
            (h.ttft_s, h.itl_s, h.shrink_above, h.min_turns),
            (3.0, 0.05, 0.97, 20)
        );
        assert!(serde_json::from_str::<PoolConfig>(r#"{"headroom": {"bogus": 1}}"#).is_err());
    }

    #[test]
    fn balanced_size_covers_both_sides_of_demand() {
        for (workers, share, want) in [
            // Whole shares: each side gets its share exactly.
            (4, 0.25, 1),
            (8, 0.25, 2),
            (16, 0.25, 4),
            (8, 0.5, 4),
            // Fractional: ceil(n x share) + ceil(n x rest) > n, so round(n x share).
            (4, 0.3, 1),   // 1.2
            (4, 0.6, 2),   // 2.4
            (8, 0.24, 2),  // 1.92
            (8, 0.32, 3),  // 2.56
            (16, 0.27, 4), // 4.32
            // At least one worker each while both sides have demand.
            (4, 0.01, 1),
            (4, 0.0, 1),
            (2, 0.9, 1),
            (4, 0.95, 3),
            // Only priority: every worker.
            (4, 1.0, 4),
        ] {
            assert_eq!(balanced_size(workers, share), want, "{workers} {share}");
        }
    }

    /// Priority holds `p` of `p + s` admitted tokens over 4 workers, one window on.
    fn sized(rule: SizeRule, p: u64, s: u64) -> usize {
        let state = PoolState::new(PoolConfig {
            window_s: 1.0,
            demand: PoolDemand::Admitted,
            min_unreserved: 0,
            size_rule: rule,
            ..Default::default()
        });
        state.arrived("s", "standard", s, t(0.0), 4);
        state.sent("s", t(0.0), 3);
        state.arrived("p", "priority", p, t(0.0), 4);
        state.sent("p", t(0.0), 0);
        state.arrived("x", "batch", 0, t(100.0), 4);
        state.pool_size(4)
    }

    #[test]
    fn balanced_rule_leaves_the_other_classes_their_share() {
        // A 0.25-0.3 share: the multiplier rule holds 2 of 4 workers, the balanced rule 1.
        assert_eq!(sized(SizeRule::Multiplier, 250, 750), 2);
        assert_eq!(sized(SizeRule::Balanced, 250, 750), 1);
        assert_eq!(sized(SizeRule::Balanced, 300, 700), 1);
        // A 0.6 share: 3 under the multiplier rule (capped by 4), 2 balanced.
        assert_eq!(sized(SizeRule::Multiplier, 600, 400), 4);
        assert_eq!(sized(SizeRule::Balanced, 600, 400), 2);
        // The prior sizes a balanced pool too.
        let state = PoolState::new(PoolConfig {
            prior_share: Some(0.25),
            size_rule: SizeRule::Balanced,
            ..Default::default()
        });
        assert_eq!(state.pool_size(8), 2);
    }

    /// One priority turn arriving at `at`, first token `ttft` later, `out` tokens at `itl` each.
    fn turn(state: &PoolState, id: &str, at: f64, ttft: f64, itl: f64) {
        let out = 11;
        state.arrived(id, "priority", 0, t(at), 8);
        state.sent(id, t(at), 0);
        state.first_token(id, t(at + ttft));
        state.completed(id, Some(out), t(at + ttft + itl * (out - 1) as f64));
        state.finished(id);
    }

    #[test]
    fn headroom_grows_on_misses_and_shrinks_with_room_and_waiting_demand() {
        let state = PoolState::new(PoolConfig {
            window_s: 1.0,
            resize_s: 100.0,
            demand: PoolDemand::Admitted,
            min_unreserved: 0,
            size_rule: SizeRule::Balanced,
            headroom: Some(Headroom {
                min_turns: 4,
                ..Default::default()
            }),
            ..Default::default()
        });
        // Admitted: priority 250 of 1000 tokens over 8 workers (balanced: 2); standard 2000
        // more waits in the lobby.
        state.arrived("s", "standard", 750, t(0.0), 8);
        state.sent("s", t(0.0), 7);
        state.arrived("w", "standard", 2000, t(0.0), 8);
        state.arrived("p", "priority", 250, t(0.0), 8);
        state.sent("p", t(0.0), 0);
        state.arrived("x", "batch", 0, t(10.0), 8);
        assert_eq!(state.pool_size(8), 2);
        // Half the priority turns miss (TTFT 5 s, or ITL 60 ms): one worker more.
        turn(&state, "a", 20.0, 0.5, 0.02);
        turn(&state, "b", 21.0, 5.0, 0.02);
        turn(&state, "c", 22.0, 0.5, 0.06);
        turn(&state, "d", 23.0, 0.5, 0.02);
        state.arrived("x", "batch", 0, t(120.0), 8);
        assert_eq!(state.pool_size(8), 3);
        // Too few turns: no move.
        turn(&state, "e", 130.0, 5.0, 0.02);
        state.arrived("x", "batch", 0, t(230.0), 8);
        assert_eq!(state.pool_size(8), 3);
        // All on target, standard waiting: one fewer, back to the rule's size.
        for (i, id) in ["f", "g", "h", "i"].iter().enumerate() {
            turn(&state, id, 240.0 + i as f64, 0.5, 0.02);
        }
        state.arrived("x", "batch", 0, t(340.0), 8);
        assert_eq!(state.pool_size(8), 2);
        // All on target but nothing waiting: no move.
        state.finished("w");
        for (i, id) in ["j", "k", "l", "m"].iter().enumerate() {
            turn(&state, id, 350.0 + i as f64, 0.5, 0.02);
        }
        state.arrived("x", "batch", 0, t(450.0), 8);
        assert_eq!(state.pool_size(8), 2);
    }

    #[test]
    fn headroom_does_not_wind_up_past_the_bounds() {
        let state = PoolState::new(PoolConfig {
            window_s: 1.0,
            resize_s: 100.0,
            demand: PoolDemand::Admitted,
            size_rule: SizeRule::Balanced,
            headroom: Some(Headroom {
                min_turns: 1,
                ..Default::default()
            }),
            ..Default::default()
        });
        state.arrived("s", "standard", 500, t(0.0), 8);
        state.sent("s", t(0.0), 7);
        state.arrived("w", "standard", 500, t(0.0), 8);
        state.arrived("p", "priority", 500, t(0.0), 8);
        state.sent("p", t(0.0), 0);
        // Every interval misses: 4, 5, 6, 7, then held at 8 - min_unreserved (1) = 7.
        for k in 0..6 {
            let at = 10.0 + 100.0 * k as f64;
            turn(&state, &format!("m{k}"), at, 5.0, 0.02);
            state.arrived("x", "batch", 0, t(at + 95.0), 8);
        }
        assert_eq!(state.pool_size(8), 7);
        assert_eq!(state.load.lock().extra, 3);
        // One interval on target: one fewer at once (no wound-up extra to work off).
        turn(&state, "ok", 700.0, 0.5, 0.02);
        state.arrived("x", "batch", 0, t(800.0), 8);
        assert_eq!(state.pool_size(8), 6);
    }

    #[test]
    fn headroom_off_tracks_no_turns() {
        let state = PoolState::new(PoolConfig::default());
        turn(&state, "a", 0.0, 5.0, 0.02);
        let load = state.load.lock();
        assert_eq!((load.turns, load.turns_ok, load.extra), (0, 0, 0));
    }

    #[test]
    fn sends_are_counted_per_side_once_the_edge_is_known() {
        let state = PoolState::new(PoolConfig::default());
        state.arrived("s", "standard", 1, t(0.0), 4);
        state.sent("s", t(0.0), 0); // no selection yet: not counted
        state.edge.store(1, Ordering::Relaxed);
        state.arrived("b", "batch", 1, t(0.0), 4);
        state.sent("b", t(0.0), 1);
        state.arrived("p", "priority", 1, t(0.0), 4);
        state.sent("p", t(0.0), 3);
        let n = |c: &[AtomicU64; 3]| {
            c.iter()
                .map(|x| x.load(Ordering::Relaxed))
                .collect::<Vec<_>>()
        };
        assert_eq!(n(&state.sent_in), vec![0, 0, 1]);
        assert_eq!(n(&state.sent_out), vec![1, 0, 0]);
    }
}
