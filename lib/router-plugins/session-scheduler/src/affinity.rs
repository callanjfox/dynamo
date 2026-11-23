// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session affinity as a worker-selection policy: a session goes back to its home worker, the
//! worker its previous turn was sent to, and a subagent's first turn follows its parent's home
//! (the simulator's `affinity`, `web/sim.js` `pickWorker`). With `spread_fanout` k, the first
//! turn of a subagent from a fan-out of k or more has no home, so the fan-out spreads by cost
//! (the simulator's `fanout: 'spread'`); the fan-out size comes from the client hint
//! `x-dynamo-hint-fanout`.
//!
//! It is the router's default policy plus one scorer ([`AffinityScorer`]) that adds `weight` to
//! every candidate other than the home. With the default (1e9) the home is taken whenever it is a
//! candidate, as the simulator's hard affinity does; when the home is not a candidate (not
//! eligible, or filtered) the default cost decides. At weight 0 it changes nothing.
//!
//! Homes come from shared state the session scheduler's classifier feeds: at classify it records
//! the request's session and parent, and at `Sent` the worker. The selection context carries the
//! request id, so no session index is needed: the home is where the router itself last sent the
//! session, not where its KV is now.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use dynamo_kv_router::plugins::prelude::*;
use parking_lot::Mutex;
use serde::Deserialize;

/// The `worker_selection` type this crate registers.
pub const SESSION_AFFINITY_TYPE: &str = "session-affinity";

/// Agent header with the size of the fan-out a subagent belongs to (subagents its parent's turn
/// spawned). Client signal; ignored when malformed.
pub const HEADER_HINT_FANOUT: &str = "x-dynamo-hint-fanout";

/// Requests and sessions tracked at most; beyond this new ones are not tracked (they score zero).
const MAX_TRACKED: usize = 1 << 20;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AffinityConfig {
    /// The shared-state name the classifier feeds.
    pub name: String,
    /// Cost added to every candidate other than the home.
    pub weight: f64,
    /// Subagent first turns of a fan-out of at least this many have no home; `None` never spreads.
    pub spread_fanout: Option<u32>,
    /// Seed of the picker's random stream; see `pool.rs`.
    pub seed: Option<u64>,
}

impl Default for AffinityConfig {
    fn default() -> Self {
        Self {
            name: "default".into(),
            weight: 1e9,
            spread_fanout: None,
            seed: None,
        }
    }
}

struct Request {
    session: String,
    parent: Option<String>,
    fanout: Option<u32>,
    session_final: bool,
}

#[derive(Default)]
struct Tables {
    requests: HashMap<String, Request>,
    homes: HashMap<String, WorkerWithDpRank>,
}

pub struct AffinityState {
    config: AffinityConfig,
    tables: Mutex<Tables>,
    scored: AtomicU64,
    home_found: AtomicU64,
    spread: AtomicU64,
}

impl AffinityState {
    pub fn new(config: AffinityConfig) -> Self {
        Self {
            config,
            tables: Mutex::new(Tables::default()),
            scored: AtomicU64::new(0),
            home_found: AtomicU64::new(0),
            spread: AtomicU64::new(0),
        }
    }

    pub fn config(&self) -> &AffinityConfig {
        &self.config
    }

    /// A request entered the router.
    pub fn arrived(
        &self,
        request_id: &str,
        session: &str,
        parent: Option<&str>,
        fanout: Option<u32>,
        session_final: bool,
    ) {
        let mut tables = self.tables.lock();
        if tables.requests.len() >= MAX_TRACKED && !tables.requests.contains_key(request_id) {
            return;
        }
        tables.requests.insert(
            request_id.to_owned(),
            Request {
                session: session.to_owned(),
                parent: parent.map(str::to_owned),
                fanout,
                session_final,
            },
        );
    }

    /// The router sent the request to `worker`: that worker is its session's home now.
    pub fn sent(&self, request_id: &str, worker: WorkerWithDpRank) {
        let mut tables = self.tables.lock();
        let Some(session) = tables.requests.get(request_id).map(|r| r.session.clone()) else {
            return;
        };
        if tables.homes.len() >= MAX_TRACKED && !tables.homes.contains_key(&session) {
            return;
        }
        tables.homes.insert(session, worker);
    }

    /// The request left the router. A session's final request also forgets its home.
    pub fn finished(&self, request_id: &str) {
        let mut tables = self.tables.lock();
        if let Some(request) = tables.requests.remove(request_id)
            && request.session_final
        {
            tables.homes.remove(&request.session);
        }
    }

    /// The home the request should go to: its session's, else (a first turn) its parent's,
    /// unless it belongs to a fan-out the config spreads.
    pub fn home_of(&self, request_id: &str) -> Option<WorkerWithDpRank> {
        let tables = self.tables.lock();
        let request = tables.requests.get(request_id)?;
        if let Some(home) = tables.homes.get(&request.session) {
            return Some(*home);
        }
        let parent = request.parent.as_ref()?;
        if let (Some(k), Some(n)) = (self.config.spread_fanout, request.fanout)
            && n >= k
        {
            self.spread.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        tables.homes.get(parent).copied()
    }

    /// (selections scored, selections with a home among the candidates, spread first turns).
    pub fn stats(&self) -> (u64, u64, u64) {
        (
            self.scored.load(Ordering::Relaxed),
            self.home_found.load(Ordering::Relaxed),
            self.spread.load(Ordering::Relaxed),
        )
    }
}

fn states() -> &'static Mutex<HashMap<String, Arc<AffinityState>>> {
    static STATES: OnceLock<Mutex<HashMap<String, Arc<AffinityState>>>> = OnceLock::new();
    STATES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Install a fresh state under `config.name`, replacing an earlier one.
pub fn install(config: AffinityConfig) -> Arc<AffinityState> {
    let state = Arc::new(AffinityState::new(config));
    states()
        .lock()
        .insert(state.config.name.clone(), Arc::clone(&state));
    state
}

/// The state a session affinity policy installed under `name`, if any.
pub fn installed(name: &str) -> Option<Arc<AffinityState>> {
    states().lock().get(name).cloned()
}

/// Adds `weight` to every candidate other than the request's home.
pub struct AffinityScorer {
    state: Arc<AffinityState>,
}

impl AffinityScorer {
    pub fn new(state: Arc<AffinityState>) -> Self {
        Self { state }
    }
}

impl WorkerScorer for AffinityScorer {
    fn score(
        &self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        costs.fill(0.0);
        // The adaptive rule's place group (`adaptive.rs`) switched off: today's router decides.
        if !crate::adaptive::place_enabled() {
            return Ok(());
        }
        self.state.scored.fetch_add(1, Ordering::Relaxed);
        let Some(home) = context.request_id().and_then(|id| self.state.home_of(id)) else {
            return Ok(());
        };
        if !candidates.iter().any(|c| c.worker() == home) {
            return Ok(());
        }
        self.state.home_found.fetch_add(1, Ordering::Relaxed);
        let weight = self.state.config.weight;
        for (candidate, cost) in candidates.iter().zip(costs.iter_mut()) {
            if candidate.worker() != home {
                *cost = weight;
            }
        }
        Ok(())
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let config: AffinityConfig = parameters.deserialize()?;
    if !(config.weight.is_finite() && config.weight >= 0.0) {
        return Err(WorkerSelectionPolicyProviderError::new(
            "session-affinity weight must be finite and not negative",
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
                vec![Box::new(AffinityScorer::new(Arc::clone(&state)))],
            )
        },
    ))
}

/// Register the session affinity worker-selection type.
pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(SESSION_AFFINITY_TYPE, Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(id, 0)
    }

    #[test]
    fn a_session_returns_home_and_a_subagent_follows_its_parent() {
        let state = AffinityState::new(AffinityConfig::default());
        state.arrived("r1", "s", None, None, false);
        assert_eq!(state.home_of("r1"), None);
        state.sent("r1", w(3));
        state.arrived("r2", "s", None, None, false);
        assert_eq!(state.home_of("r2"), Some(w(3)));
        state.arrived("c1", "s:sub", Some("s"), Some(2), false);
        assert_eq!(state.home_of("c1"), Some(w(3)));
        state.sent("c1", w(5));
        state.arrived("c2", "s:sub", Some("s"), Some(2), true);
        assert_eq!(state.home_of("c2"), Some(w(5)));
        state.finished("c2");
        state.arrived("c3", "s:sub", Some("s"), Some(2), false);
        assert_eq!(state.home_of("c3"), Some(w(3)));
    }

    #[test]
    fn a_large_fan_out_spreads() {
        let state = AffinityState::new(AffinityConfig {
            spread_fanout: Some(3),
            ..Default::default()
        });
        state.arrived("p", "s", None, None, false);
        state.sent("p", w(1));
        state.arrived("a", "s:a", Some("s"), Some(3), false);
        state.arrived("b", "s:b", Some("s"), Some(2), false);
        assert_eq!(state.home_of("a"), None);
        assert_eq!(state.home_of("b"), Some(w(1)));
        state.sent("a", w(4));
        state.arrived("a2", "s:a", Some("s"), Some(3), false);
        assert_eq!(state.home_of("a2"), Some(w(4)));
    }
}
