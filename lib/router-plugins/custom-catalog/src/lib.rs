// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Custom worker-selection policy catalog: makes `sgl-router-cache-aware` and
//! `sticky-until-saturated` selectable via `DYN_ROUTER_WORKER_SELECTION_POLICY`
//! or `--router-policy-config`, alongside Dynamo's built-in `default` policy.

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyRegistry, WorkerSelectionPolicyRegistryError,
};

/// Register every policy type supplied by this catalog.
pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    sgl_router_cache_aware_dynamo_policy::register(registry)?;
    sticky_until_saturated_dynamo_policy::register(registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_both_policies() {
        let mut registry = WorkerSelectionPolicyRegistry::default();
        register(&mut registry).unwrap();

        assert!(matches!(
            sgl_router_cache_aware_dynamo_policy::register(&mut registry),
            Err(WorkerSelectionPolicyRegistryError::Duplicate { name }) if name == "sgl-router-cache-aware"
        ));
        assert!(matches!(
            sticky_until_saturated_dynamo_policy::register(&mut registry),
            Err(WorkerSelectionPolicyRegistryError::Duplicate { name }) if name == "sticky-until-saturated"
        ));
    }
}
