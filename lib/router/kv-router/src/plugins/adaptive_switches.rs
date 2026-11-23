// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Switches a request classifier plugin may turn to steer router-owned capabilities it does not
//! run itself (the session scheduler's adaptive rule: its `kv` and `preempt` groups).
//!
//! - **KV actions** (`kv_actions` in the router policy file): on by default; off, the routing host
//!   sends no KV action, as if the section were absent.
//! - **Class preemption in the engine** (`preemptLower`): on by default; off, a host that forwards
//!   it tells its engines not to preempt lower classes.
//!
//! Process-wide: one router per process (a live router, or one offline replay at a time).

use std::sync::atomic::{AtomicBool, Ordering};

/// f50: a class's router busy threshold (`prefill_busy_threshold_frac`) set at run time; see
/// `dynamo_slot_tracker::busy::set_class_prefill_busy_frac`.
pub use dynamo_slot_tracker::busy::{class_prefill_busy_frac, set_class_prefill_busy_frac};

static KV_ACTIONS: AtomicBool = AtomicBool::new(true);
static CLASS_PREEMPTION: AtomicBool = AtomicBool::new(true);

/// Whether the router's KV actions are on.
pub fn kv_actions_enabled() -> bool {
    KV_ACTIONS.load(Ordering::Relaxed)
}

pub fn set_kv_actions_enabled(on: bool) {
    KV_ACTIONS.store(on, Ordering::Relaxed);
}

/// Whether engines may preempt lower classes for higher ones.
pub fn class_preemption_enabled() -> bool {
    CLASS_PREEMPTION.load(Ordering::Relaxed)
}

pub fn set_class_preemption_enabled(on: bool) {
    CLASS_PREEMPTION.store(on, Ordering::Relaxed);
}

// f49o batch yield (the session scheduler's `admission.batch_yield`): the level (f64 bits; 0 is off),
// the engine class rank at or above which classes are protected (usize::MAX: none), the remaining
// output tokens within which a lower-class request is let finish, and whether lower-class work steps
// aside for the protected ITL.
static BATCH_YIELD_LEVEL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BATCH_YIELD_RANK: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);
static BATCH_YIELD_FINISH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static BATCH_YIELD_STEP_ASIDE: AtomicBool = AtomicBool::new(false);

/// Set the batch-yield signal a host forwards to its engines.
pub fn set_batch_yield(
    level: f64,
    protect_rank: Option<usize>,
    finish_tokens: usize,
    step_aside: bool,
) {
    BATCH_YIELD_LEVEL.store(level.max(0.0).to_bits(), Ordering::Relaxed);
    BATCH_YIELD_RANK.store(protect_rank.unwrap_or(usize::MAX), Ordering::Relaxed);
    BATCH_YIELD_FINISH.store(finish_tokens, Ordering::Relaxed);
    BATCH_YIELD_STEP_ASIDE.store(step_aside, Ordering::Relaxed);
}

/// The batch-yield signal: (level, protected rank, finish tokens, step aside). Level 0 by default.
pub fn batch_yield() -> (f64, Option<usize>, usize, bool) {
    let rank = BATCH_YIELD_RANK.load(Ordering::Relaxed);
    (
        f64::from_bits(BATCH_YIELD_LEVEL.load(Ordering::Relaxed)),
        (rank != usize::MAX).then_some(rank),
        BATCH_YIELD_FINISH.load(Ordering::Relaxed),
        BATCH_YIELD_STEP_ASIDE.load(Ordering::Relaxed),
    )
}

// f62: the priority pool's current size, published by the session scheduler's pool scorer so the
// router can tell pool workers apart (`kv_actions.prefill_isolation`, `workers: pool`). The pool is
// the `size` lowest worker ids. usize::MAX: none published.
static PRIORITY_POOL_SIZE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

/// Publish the priority pool's size (`None`: no pool).
pub fn set_priority_pool_size(size: Option<usize>) {
    PRIORITY_POOL_SIZE.store(size.unwrap_or(usize::MAX), Ordering::Relaxed);
}

/// The priority pool's last published size, if any.
pub fn priority_pool_size() -> Option<usize> {
    let size = PRIORITY_POOL_SIZE.load(Ordering::Relaxed);
    (size != usize::MAX).then_some(size)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Only the defaults: other tests in this binary read the KV switch, so none turns it off.
    #[test]
    fn switches_default_on() {
        assert!(kv_actions_enabled() && class_preemption_enabled());
    }
}
