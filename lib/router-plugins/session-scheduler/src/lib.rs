// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Dynamo session scheduler: a router request classifier and dispatch gate that admits
//! agentic sessions against the KV budget and scores per-class service-level attainment.
//!
//! Rules every module keeps:
//!
//! - time comes from an injected [`clock::Clock`]; only [`clock::HostClock`] reads the host
//!   clock (`tests/no_system_clock.rs` enforces it);
//! - nothing on the request path waits or does more than O(1) amortized work per call, apart
//!   from the gate's check, which reads the session table;
//! - every table is bounded;
//! - the classifier never waits on the [`forecast::Forecast`]: it feeds it through a bounded
//!   channel and the forecast does its work when queried.
//!
//! See `README.md` for what is built, what is stubbed, and the upstream needs.

pub mod accounting;
pub mod adaptive;
pub mod admit;
pub mod affinity;
pub mod aimd;
pub mod batch_yield;
pub mod busy;
pub mod clock;
pub mod config;
pub mod coordinator;
pub mod experience;
pub mod forecast;
pub mod harm_debug;
pub mod honest_retry;
pub mod hooks;
pub mod hot_measure;
pub mod metrics;
pub mod mix_select;
pub mod placement;
pub mod pool;
pub mod route;
pub mod scheduler;
pub mod session;
pub mod session_signal;
pub mod sla_budget;
pub mod step_budget;
pub mod step_target;
pub mod tickets;
pub mod transfer;
pub mod uncontended;

pub use affinity::SESSION_AFFINITY_TYPE;
pub use hooks::{SESSION_SCHEDULER_TYPE, SessionSchedulerClassifier, SessionSchedulerGate};
pub use placement::SESSION_PLACEMENT_TYPE;
pub use pool::PRIORITY_POOL_TYPE;

use dynamo_kv_router::plugins::prelude::{
    DefaultPolicyWithScorers, RouterPluginRegistry, RouterPluginRegistryError,
    WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
};

/// The routing host's default policy with extra scorers, which every worker-selection policy here
/// builds on (so this crate needs only `dynamo-kv-router`; `dynamo-router-policies` supplies it
/// through `with_defaults`).
pub(crate) fn host_default(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<DefaultPolicyWithScorers, WorkerSelectionPolicyProviderError> {
    parameters.default_with_scorers().cloned().ok_or_else(|| {
        WorkerSelectionPolicyProviderError::new(
            "the routing host supplies no default policy to build on \
             (RouterPluginRegistry::with_default_policy_with_scorers)",
        )
    })
}

/// Register this crate's plugin types: the session scheduler (request classifier and dispatch
/// gate), and the priority pool and session affinity (worker selection).
pub fn register(registry: &mut RouterPluginRegistry) -> Result<(), RouterPluginRegistryError> {
    hooks::register(registry)?;
    pool::register(registry)?;
    affinity::register(registry)?;
    placement::register(registry)?;
    busy::register(registry)?;
    Ok(())
}
