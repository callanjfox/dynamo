// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The scheduler's configuration loads from a realistic router policy file, under
//! `request_classifier: { type, parameters }`, through the same registry path the router uses.

use dynamo_kv_router::plugins::prelude::*;
use dynamo_session_scheduler::{SESSION_SCHEDULER_TYPE, register};

fn resolve(yaml: &str) -> Result<Option<RequestClassifierFactory>, RequestClassifierRegistryError> {
    let policy_file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(policy_file.path(), yaml).unwrap();
    let config = KvRouterConfig {
        router_policy_config: Some(policy_file.path().display().to_string()),
        ..Default::default()
    };
    let mut registry = RouterPluginRegistry::default();
    register(&mut registry).unwrap();
    registry.resolve_request_classifier(&config)
}

#[test]
fn loads_parameters_from_a_router_policy_file() {
    let yaml = format!(
        r#"
default_policy_class: standard
policy_classes:
  - name: priority
    quantum: 8
    slo_ttft_ms: 3000
    prefill_busy_threshold_frac: 0.9
  - name: standard
    quantum: 4
    slo_ttft_ms: 8000
  - name: batch
    quantum: 1
request_classifier:
  type: {SESSION_SCHEDULER_TYPE}
  parameters:
    mode: enforce
    default_class: standard
    admission:
      lobby_order: fair
      dram_tokens_per_worker: 2000000
    session_table_capacity: 4096
"#
    );
    let factory = resolve(&yaml)
        .unwrap()
        .expect("the scheduler is configured");
    let classifier = factory(RequestClassifierContext::new(16, Vec::new));
    assert!(
        classifier.dispatch_gate().is_some(),
        "enforce installs the gate"
    );
}

#[test]
fn defaults_apply_without_parameters() {
    let yaml = format!("request_classifier:\n  type: {SESSION_SCHEDULER_TYPE}\n");
    let factory = resolve(&yaml).unwrap().unwrap();
    // The default mode is shadow: the gate is installed and never holds.
    assert!(
        factory(RequestClassifierContext::new(16, Vec::new))
            .dispatch_gate()
            .is_some()
    );
}

#[test]
fn bad_parameters_say_what_is_wrong() {
    let yaml = format!(
        r#"
request_classifier:
  type: {SESSION_SCHEDULER_TYPE}
  parameters:
    default_class: gold
"#
    );
    let error = resolve(&yaml)
        .err()
        .expect("an unknown default class is rejected")
        .to_string();
    assert!(
        error.contains("default_class \"gold\" is not one of the classes"),
        "{error}"
    );

    let yaml = format!(
        "request_classifier:\n  type: {SESSION_SCHEDULER_TYPE}\n  parameters:\n    mod: enforce\n"
    );
    let error = resolve(&yaml)
        .err()
        .expect("an unknown key is rejected")
        .to_string();
    assert!(error.contains("mod"), "{error}");
}
