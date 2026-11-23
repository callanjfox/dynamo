// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Busy-aware worker selection: the router's default policy plus a scorer that adds a large cost
//! to every worker past the request's class busy threshold, so selection does not send a request
//! to a full worker for KV reasons while another has room (upstream feedback item 3).
//!
//! A worker is past the threshold when its active prefill tokens exceed the class's fraction of
//! its batched-token budget (as the router's `prefill_busy_threshold_frac` judges it), or when it
//! is at its running-request limit and its capacity carries one (`busy_at_max_num_seqs`). The
//! request's class comes from state the session scheduler classifier feeds, read by request id,
//! as in `pool.rs`. When every candidate is past the threshold all get the same penalty and the
//! default cost decides; at penalty 0 nothing changes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use dynamo_kv_router::plugins::prelude::*;
use parking_lot::Mutex;
use serde::Deserialize;

/// The `worker_selection` type this crate registers.
pub const CLASS_BUSY_TYPE: &str = "class-busy";

/// Requests tracked at most; beyond this new ones are not tracked (they score zero).
const MAX_TRACKED: usize = 1 << 20;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct BusyConfig {
    /// The shared-state name the classifier feeds.
    pub name: String,
    /// Prefill busy threshold per class, as a fraction of a worker's batched-token budget (match
    /// the router's `policy_classes`). A class not listed is never judged busy.
    pub thresholds: HashMap<String, f64>,
    /// Cost added to a worker past the request's class threshold.
    pub penalty: f64,
    /// Seed of the picker's random stream; see `pool.rs`.
    pub seed: Option<u64>,
}

impl Default for BusyConfig {
    fn default() -> Self {
        Self {
            name: "default".into(),
            thresholds: [("priority", 1.0), ("standard", 0.95), ("batch", 0.85)]
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v))
                .collect(),
            penalty: 1e9,
            seed: None,
        }
    }
}

/// Whether a worker with `load` and `capacity` is past `frac` of its prefill budget or at its
/// running-request limit.
pub fn past_threshold<L: RankLoad>(load: &L, capacity: WorkerCapacity, frac: f64) -> bool {
    let prefill = load.active_prefill_tokens() as f64 > frac * capacity.prefill_budget() as f64;
    let seqs = capacity
        .max_num_seqs()
        .zip(load.active_requests())
        .is_some_and(|(limit, running)| running as u64 >= limit);
    prefill || seqs
}

pub struct BusyState {
    config: BusyConfig,
    classes: Mutex<HashMap<String, String>>,
    scored: AtomicU64,
    penalised: AtomicU64,
}

impl BusyState {
    pub fn new(config: BusyConfig) -> Self {
        Self {
            config,
            classes: Mutex::new(HashMap::new()),
            scored: AtomicU64::new(0),
            penalised: AtomicU64::new(0),
        }
    }

    pub fn arrived(&self, request_id: &str, class: &str) {
        let mut classes = self.classes.lock();
        if classes.len() < MAX_TRACKED || classes.contains_key(request_id) {
            classes.insert(request_id.to_owned(), class.to_owned());
        }
    }

    pub fn finished(&self, request_id: &str) {
        self.classes.lock().remove(request_id);
    }

    fn threshold_of(&self, request_id: &str) -> Option<f64> {
        let class = self.classes.lock().get(request_id).cloned()?;
        self.config.thresholds.get(&class).copied()
    }

    /// (selections scored, worker rows penalised).
    pub fn stats(&self) -> (u64, u64) {
        (
            self.scored.load(Ordering::Relaxed),
            self.penalised.load(Ordering::Relaxed),
        )
    }
}

fn states() -> &'static Mutex<HashMap<String, Arc<BusyState>>> {
    static STATES: OnceLock<Mutex<HashMap<String, Arc<BusyState>>>> = OnceLock::new();
    STATES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Install a fresh state under `config.name`, replacing an earlier one.
pub fn install(config: BusyConfig) -> Arc<BusyState> {
    let state = Arc::new(BusyState::new(config));
    states()
        .lock()
        .insert(state.config.name.clone(), Arc::clone(&state));
    state
}

/// The state a class-busy policy installed under `name`, if any.
pub fn installed(name: &str) -> Option<Arc<BusyState>> {
    states().lock().get(name).cloned()
}

/// Adds `penalty` to every worker past the request's class busy threshold.
pub struct ClassBusyScorer {
    state: Arc<BusyState>,
}

impl WorkerScorer for ClassBusyScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn score(
        &self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        costs.fill(0.0);
        self.state.scored.fetch_add(1, Ordering::Relaxed);
        let Some(frac) = context
            .request_id()
            .and_then(|id| self.state.threshold_of(id))
        else {
            return Ok(());
        };
        for (candidate, cost) in candidates.iter().zip(costs.iter_mut()) {
            if let Some(load) = candidate.load()
                && past_threshold(load, candidate.info().capacity(), frac)
            {
                *cost = self.state.config.penalty;
                self.state.penalised.fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(())
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let config: BusyConfig = parameters.deserialize()?;
    if !(config.penalty.is_finite() && config.penalty >= 0.0) {
        return Err(WorkerSelectionPolicyProviderError::new(
            "class-busy penalty must be finite and not negative",
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
                vec![Box::new(ClassBusyScorer {
                    state: Arc::clone(&state),
                })],
            )
        },
    ))
}

/// Register the class-busy worker-selection type.
pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(CLASS_BUSY_TYPE, Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Load {
        prefill: usize,
        running: usize,
    }

    impl RankLoad for Load {
        fn active_prefill_tokens(&self) -> usize {
            self.prefill
        }
        fn active_decode_blocks(&self) -> usize {
            0
        }
        fn active_requests(&self) -> Option<usize> {
            Some(self.running)
        }
    }

    #[test]
    fn past_threshold_by_prefill_or_running_requests() {
        let capacity = WorkerCapacity::new(Some(1000), Some(100));
        let load = |prefill, running| Load { prefill, running };
        assert!(!past_threshold(&load(900, 10), capacity, 0.95));
        assert!(past_threshold(&load(960, 10), capacity, 0.95));
        // The running-request limit counts only when the capacity carries it.
        assert!(!past_threshold(&load(0, 8), capacity, 0.95));
        let capped = capacity.with_max_num_seqs(Some(8));
        assert!(past_threshold(&load(0, 8), capped, 0.95));
        assert!(!past_threshold(&load(0, 7), capped, 0.95));
    }

    #[test]
    fn classes_are_tracked_until_they_finish() {
        let state = BusyState::new(BusyConfig::default());
        state.arrived("r", "standard");
        assert_eq!(state.threshold_of("r"), Some(0.95));
        state.arrived("u", "unknown");
        assert_eq!(state.threshold_of("u"), None);
        state.finished("r");
        assert_eq!(state.threshold_of("r"), None);
    }
}
