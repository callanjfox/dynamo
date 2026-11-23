// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `session-placement`: the priority pool and session affinity in one worker-selection policy.
//!
//! Worker selection takes one policy, so the two placement capabilities that the simulator stacks
//! (`priorityWorkers` and `affinity`, `web/sim.js` `eligible` then `pickWorker`) get a combined
//! type: the router's default cost plus both scorers. The pool's penalty should be larger than
//! affinity's weight (defaults 1e9 and 1e6), so the pool decides first and affinity picks the home
//! inside it, as the simulator does (it restricts a class to its pool, then prefers the home among
//! the eligible workers). Either part can be left out (`pool: null` or `affinity: null`).

use std::sync::Arc;

use dynamo_kv_router::plugins::prelude::*;
use parking_lot::Mutex;
use serde::Deserialize;

use crate::affinity::{self, AffinityConfig, AffinityScorer};
use crate::pool::{self, PoolConfig, PriorityPoolScorer};
use crate::route::{self, ForecastRouteConfig, ForecastRouteScorer};

pub const SESSION_PLACEMENT_TYPE: &str = "session-placement";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PlacementConfig {
    pub pool: Option<PoolConfig>,
    pub affinity: Option<AffinityConfig>,
    /// Forecast routing (`route.rs`, the simulator's `route: 'forecast'`); off unless given. It
    /// keeps a session at home itself, so leave `affinity` out when it is on.
    pub forecast: Option<ForecastRouteConfig>,
    /// Seed of the picker's random stream (tie-breaks); see `pool.rs`.
    pub seed: Option<u64>,
    /// f50: the parts that follow the session scheduler's mix selector (`mix_select.rs`);
    /// absent: none. With it, `forecast` holds the b0 mode's settings and is required.
    pub mix_select: Option<PlacementMix>,
}

/// f50: worker selection under the mix selector.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementMix {
    /// The modes the pool acts in (it is fed in every mode).
    pub pool_modes: Vec<crate::mix_select::MixMode>,
    /// The forecast route's settings in the pg mode (`forecast` holds the b0 mode's; one state,
    /// under `forecast.name`).
    pub pg_forecast: ForecastRouteConfig,
}

impl Default for PlacementConfig {
    fn default() -> Self {
        Self {
            pool: Some(PoolConfig::default()),
            affinity: Some(AffinityConfig {
                weight: 1e6,
                ..AffinityConfig::default()
            }),
            forecast: None,
            seed: None,
            mix_select: None,
        }
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let config: PlacementConfig = parameters.deserialize()?;
    if let Some(p) = &config.pool
        && !(p.penalty.is_finite() && p.penalty >= 0.0)
    {
        return Err(WorkerSelectionPolicyProviderError::new(
            "session-placement pool penalty must be finite and not negative",
        ));
    }
    if let Some(a) = &config.affinity
        && !(a.weight.is_finite() && a.weight >= 0.0)
    {
        return Err(WorkerSelectionPolicyProviderError::new(
            "session-placement affinity weight must be finite and not negative",
        ));
    }
    if let Some(f) = &config.forecast
        && !(f.weight.is_finite() && f.weight >= 0.0 && f.candidates > 0)
    {
        return Err(WorkerSelectionPolicyProviderError::new(
            "session-placement forecast weight must be finite and not negative, candidates positive",
        ));
    }
    let rng = config
        .seed
        .map(|seed| Arc::new(Mutex::new(fastrand::Rng::with_seed(seed))));
    let pool_state = config.pool.clone().map(pool::install);
    let affinity_state = config.affinity.clone().map(affinity::install);
    if let Some(m) = &config.mix_select {
        let f = &m.pg_forecast;
        if config.forecast.is_none() {
            return Err(WorkerSelectionPolicyProviderError::new(
                "session-placement: mix_select needs forecast (the b0 mode's settings)",
            ));
        }
        if !(f.weight.is_finite() && f.weight >= 0.0 && f.candidates > 0) {
            return Err(WorkerSelectionPolicyProviderError::new(
                "session-placement mix_select.pg_forecast weight must be finite and not negative, candidates positive",
            ));
        }
    }
    let route_state = match (&config.mix_select, &config.forecast) {
        (Some(m), Some(b0)) => Some(route::install_modes(b0.clone(), m.pg_forecast.clone())),
        _ => config.forecast.clone().map(route::install),
    };
    let pool_modes = config.mix_select.as_ref().map(|m| m.pool_modes.clone());
    // f68pool: the forecast route reads the pool's side (`exclude_others`; nothing when off),
    // under the same mix selector modes as the pool's scorer.
    if let Some(pool) = &pool_state {
        pool.set_modes(pool_modes.clone());
        if let Some(route) = &route_state {
            route.attach_pool(Arc::clone(pool));
        }
    }
    let compose = crate::host_default(parameters)?;
    Ok(Arc::new(
        move |router, worker_type: WorkerType, _partition| {
            let mut scorers: Vec<Box<dyn WorkerScorer>> = Vec::new();
            if let Some(s) = &pool_state {
                let scorer = PriorityPoolScorer::new(Arc::clone(s));
                scorers.push(Box::new(match &pool_modes {
                    Some(modes) => scorer.with_modes(modes.clone()),
                    None => scorer,
                }));
            }
            if let Some(s) = &affinity_state {
                scorers.push(Box::new(AffinityScorer::new(Arc::clone(s))));
            }
            if let Some(s) = &route_state {
                scorers.push(Box::new(ForecastRouteScorer::new(Arc::clone(s))));
            }
            compose(router.clone(), worker_type.as_str(), rng.clone(), scorers)
        },
    ))
}

/// Register the combined placement worker-selection type.
pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(SESSION_PLACEMENT_TYPE, Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_put_the_pool_ahead_of_affinity() {
        let c = PlacementConfig::default();
        let pool = c.pool.expect("pool on by default");
        let aff = c.affinity.expect("affinity on by default");
        assert!(
            pool.penalty > aff.weight,
            "the pool must decide before affinity"
        );
    }

    #[test]
    fn mix_select_takes_the_pg_forecast_beside_the_b0_one() {
        let c: PlacementConfig = serde_json::from_str(
            r#"{"pool": {"min_unreserved": 0}, "affinity": null,
                "forecast": {"itl_limit": true, "home": "last_sent"},
                "mix_select": {"pool_modes": ["pg"], "pg_forecast": {"lower_tier_hits": true}}}"#,
        )
        .unwrap();
        let m = c.mix_select.expect("mix_select");
        assert_eq!(m.pool_modes, vec![crate::mix_select::MixMode::Pg]);
        assert!(m.pg_forecast.lower_tier_hits && !m.pg_forecast.itl_limit);
        assert!(c.forecast.unwrap().itl_limit);
    }

    #[test]
    fn either_part_can_be_left_out() {
        let c: PlacementConfig =
            serde_json::from_str(r#"{"pool": null, "affinity": {"weight": 5.0}}"#).unwrap();
        assert!(c.pool.is_none());
        assert_eq!(c.affinity.unwrap().weight, 5.0);
        assert!(c.forecast.is_none(), "forecast routing is off unless given");
        let c: PlacementConfig = serde_json::from_str(
            r#"{"pool": null, "affinity": null, "forecast": {"weight": 0.0}}"#,
        )
        .unwrap();
        assert_eq!(c.forecast.unwrap().candidates, 3);
    }
}
