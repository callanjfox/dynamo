// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The session scheduler inside offline KV-router replay (the replay gate): configured through
//! the router policy file as in production, run in virtual time, fed the per-request fields a
//! frontend attaches (session end, tenant, client signals).
//!
//! One worker holds 1,024 tokens of KV. Session A takes 600 and then a second turn of 648 that
//! ends the session; session B, 600 tokens, arrives 10 ms after A. In enforce mode B cannot fit
//! beside A, so it waits in the lobby until A's last turn completes and A ends. In shadow mode
//! only the router's busy threshold delays it. Two enforce runs give the same answer.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::plugins::prelude::*;
use dynamo_mocker::common::protocols::MockerConfig;
use dynamo_mocker::loadgen::TraceFileFormat;
use dynamo_mocker::replay::{
    ReplayClassifierPlugin, ReplayDeterminism, ReplayGateRecheck, ReplayRequestFields,
    ReplayRouterMode, ReplayScope, SlaThresholds, TraceSimulationReport,
    simulate_trace_file_with_router_mode_and_format, with_replay_scope,
};
use dynamo_session_scheduler::{SESSION_SCHEDULER_TYPE, register};

fn policy(mode: &str) -> KvRouterConfig {
    policy_with(mode, "")
}

/// The policy with `admission` lines added to the classifier's parameters (indented 6).
fn policy_with(mode: &str, admission: &str) -> KvRouterConfig {
    let admission = if admission.is_empty() {
        String::new()
    } else {
        format!("    admission:\n{admission}")
    };
    let yaml = format!(
        r#"
default_policy_class: standard
policy_classes:
  - name: priority
    quantum: 8
    prefill_busy_threshold_frac: 0.5
  - name: standard
    quantum: 4
    prefill_busy_threshold_frac: 0.5
  - name: batch
    quantum: 1
    prefill_busy_threshold_frac: 0.5
request_classifier:
  type: {SESSION_SCHEDULER_TYPE}
  parameters:
    mode: {mode}
    default_class: standard
{admission}"#
    );
    let file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
    std::fs::write(file.path(), yaml).unwrap();
    let (_, path) = file.keep().unwrap();
    KvRouterConfig {
        router_policy_config: Some(path.display().to_string()),
        // The gate acts on queued requests only: each class needs a busy threshold (above).
        ..Default::default()
    }
}

fn factory(config: &KvRouterConfig) -> ReplayClassifierPlugin {
    let mut registry = RouterPluginRegistry::default();
    register(&mut registry).unwrap();
    ReplayClassifierPlugin(
        registry
            .resolve_request_classifier(config)
            .unwrap()
            .expect("configured"),
    )
}

fn trace() -> tempfile::NamedTempFile {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    let header = serde_json::json!({
        "schema": "dynamo.agentic_mooncake", "version": 2, "block_size": 4,
        "hash_id_scope": "local", "source": {"format": "test", "digest": "session-scheduler"}
    });
    writeln!(file, "{header}").unwrap();
    let blocks = |base: u64, n: u64| (base..base + n).collect::<Vec<_>>();
    let rows = [
        serde_json::json!({"request_id": "a-0", "play_id": "A", "session_id": "A", "model": "m",
            "not_before_ms": 0.0, "input_length": 600, "output_length": 4,
            "hash_ids": blocks(1, 150), "dependencies": []}),
        serde_json::json!({"request_id": "a-1", "play_id": "A", "session_id": "A", "model": "m",
            "not_before_ms": 0.0, "input_length": 648, "output_length": 4,
            "hash_ids": blocks(1, 162),
            "dependencies": [{"request_id": "a-0", "trigger": "completion", "delay_ms": 1000.0,
                              "relation": "sequence"}]}),
        serde_json::json!({"request_id": "b-0", "play_id": "B", "session_id": "B", "model": "m",
            "not_before_ms": 10.0, "input_length": 600, "output_length": 4,
            "hash_ids": blocks(10_000, 150), "dependencies": []}),
    ];
    for row in rows {
        writeln!(file, "{row}").unwrap();
    }
    file
}

fn fields() -> Arc<HashMap<String, ReplayRequestFields>> {
    let tenant = |t: &str| BTreeMap::from([("x-dynamo-tenant".to_owned(), vec![t.to_owned()])]);
    let rows = [
        ReplayRequestFields {
            request_id: "a-0".into(),
            agent_headers: tenant("acme"),
            ..Default::default()
        },
        ReplayRequestFields {
            request_id: "a-1".into(),
            session_final: Some(true),
            agent_headers: tenant("acme"),
            ..Default::default()
        },
        ReplayRequestFields {
            request_id: "b-0".into(),
            session_final: Some(true),
            agent_headers: tenant("globex"),
            ..Default::default()
        },
    ];
    Arc::new(
        rows.into_iter()
            .map(|f| (f.request_id.clone(), f))
            .collect(),
    )
}

fn run(mode: &str) -> TraceSimulationReport {
    let config = policy(mode);
    let scope = ReplayScope {
        request_classifier: Some(factory(&config)),
        request_fields: Some(fields()),
        determinism: ReplayDeterminism::CanonicalV1,
        ..Default::default()
    };
    let args = MockerConfig::from_value(serde_json::json!({
        "engine": {"block_size": 4, "num_gpu_blocks": 256, "max_num_batched_tokens": 256,
                   "max_num_seqs": 8, "speedup_ratio": 1.0}
    }))
    .unwrap();
    let file = trace();
    with_replay_scope(scope, || {
        simulate_trace_file_with_router_mode_and_format(
            args,
            Some(config),
            None,
            file.path(),
            4,
            1,
            1.0,
            ReplayRouterMode::KvRouter,
            TraceFileFormat::AgenticMooncake,
            0.0,
            0,
            true,
            None,
            SlaThresholds::default(),
        )
    })
    .unwrap()
    .unwrap()
}

/// (arrival, first admission to a worker, end) of one request, in replay ms.
fn times(report: &TraceSimulationReport, id: &str) -> (f64, f64, f64) {
    let r = report
        .per_request
        .iter()
        .find(|r| r.agentic.as_ref().is_some_and(|a| a.request_id == id))
        .unwrap();
    let end = r.arrival_time_ms + r.e2e_latency_ms.unwrap();
    (r.arrival_time_ms, r.first_admit_ms.unwrap(), end)
}

#[test]
fn enforce_holds_the_second_session_until_the_first_ends() {
    let wall = Instant::now();
    let report = run("enforce");
    let wall_ms = wall.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(report.request_counts.completed_requests, 3);
    let (_, _, a1_end) = times(&report, "a-1");
    let (b_arrival, b_admit, b_end) = times(&report, "b-0");
    assert!(
        b_admit >= a1_end - 1e-6,
        "B admitted at {b_admit} ms, before A's last turn ended at {a1_end} ms"
    );
    assert!(
        b_admit > b_arrival + 1000.0,
        "B waited for A's think gap too"
    );
    eprintln!(
        "simulated {:.0} ms in {:.1} ms of wall clock ({:.0}x)",
        b_end,
        wall_ms,
        b_end / wall_ms
    );
}

#[test]
fn shadow_mode_never_holds() {
    let report = run("shadow");
    assert_eq!(report.request_counts.completed_requests, 3);
    // Only the router's own busy threshold queues B (until A's first turn finishes); the
    // scheduler logs a hold but never enforces it, so B does not wait for A to end.
    let (_, _, a1_end) = times(&report, "a-1");
    let (_, b_admit, _) = times(&report, "b-0");
    assert!(
        b_admit < a1_end,
        "B admitted at {b_admit}, A ended at {a1_end}"
    );
}

#[test]
fn enforce_runs_are_identical() {
    let key = |r: &TraceSimulationReport| {
        let mut v: Vec<(String, u64, u64)> = r
            .per_request
            .iter()
            .map(|r| {
                (
                    r.agentic.as_ref().unwrap().request_id.clone(),
                    r.first_admit_ms.unwrap().to_bits(),
                    r.e2e_latency_ms.unwrap().to_bits(),
                )
            })
            .collect();
        v.sort();
        v
    };
    assert_eq!(key(&run("enforce")), key(&run("enforce")));
}

/// Speed: 64 sessions of 20 turns with 60 s think gaps through the enforcing scheduler on four
/// workers. Prints simulated against wall-clock time. `cargo test --release -- --ignored speed`.
#[test]
#[ignore]
fn speed() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    let header = serde_json::json!({
        "schema": "dynamo.agentic_mooncake", "version": 2, "block_size": 4,
        "hash_id_scope": "local", "source": {"format": "test", "digest": "speed"}
    });
    writeln!(file, "{header}").unwrap();
    let mut fields = HashMap::new();
    for s in 0..64u64 {
        for t in 0..20u64 {
            let id = format!("s{s}-{t}");
            let input = 256 + 64 * t as usize;
            let dependencies = if t == 0 {
                serde_json::json!([])
            } else {
                serde_json::json!([{"request_id": format!("s{s}-{}", t - 1),
                    "trigger": "completion", "delay_ms": 60_000.0, "relation": "sequence"}])
            };
            let row = serde_json::json!({"request_id": id, "play_id": format!("p{s}"),
                "session_id": format!("s{s}"), "model": "m",
                "not_before_ms": (s * 5_000) as f64, "input_length": input, "output_length": 64,
                "hash_ids": (0..input as u64 / 4).map(|b| s * 100_000 + b).collect::<Vec<_>>(),
                "dependencies": dependencies});
            writeln!(file, "{row}").unwrap();
            fields.insert(
                id.clone(),
                ReplayRequestFields {
                    request_id: id,
                    session_final: (t == 19).then_some(true),
                    ..Default::default()
                },
            );
        }
    }
    let config = policy("enforce");
    let scope = ReplayScope {
        request_classifier: Some(factory(&config)),
        request_fields: Some(Arc::new(fields)),
        determinism: ReplayDeterminism::CanonicalV1,
        ..Default::default()
    };
    let args = MockerConfig::from_value(serde_json::json!({
        "engine": {"block_size": 4, "num_gpu_blocks": 4096, "max_num_batched_tokens": 2048,
                   "max_num_seqs": 64, "speedup_ratio": 1.0}
    }))
    .unwrap();
    let wall = Instant::now();
    let report = with_replay_scope(scope, || {
        simulate_trace_file_with_router_mode_and_format(
            args,
            Some(config),
            None,
            file.path(),
            4,
            4,
            1.0,
            ReplayRouterMode::KvRouter,
            TraceFileFormat::AgenticMooncake,
            0.0,
            0,
            true,
            None,
            SlaThresholds::default(),
        )
    })
    .unwrap()
    .unwrap();
    let wall_s = wall.elapsed().as_secs_f64();
    let simulated_s = report
        .per_request
        .iter()
        .map(|r| r.arrival_time_ms + r.e2e_latency_ms.unwrap_or(0.0))
        .fold(0.0, f64::max)
        / 1000.0;
    assert_eq!(report.request_counts.completed_requests, 64 * 20);
    eprintln!(
        "speed: {} requests, {simulated_s:.0} s simulated in {wall_s:.2} s wall clock ({:.0}x)",
        64 * 20,
        simulated_s / wall_s
    );
}

/// Keyed rechecks (`DYNAMO_REPLAY_GATE_RECHECK=keyed`) replay exactly as the full recheck: 48
/// sessions of four turns in three classes and three tenants arrive within two seconds on two
/// small workers, so sessions wait in the lobby behind each other, some past their wait limit
/// (expired and retried). First-come, fair and tenant order, and with pacing.
#[test]
fn keyed_rechecks_replay_as_the_full_recheck() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    let header = serde_json::json!({
        "schema": "dynamo.agentic_mooncake", "version": 2, "block_size": 4,
        "hash_id_scope": "local", "source": {"format": "test", "digest": "keyed"}
    });
    writeln!(file, "{header}").unwrap();
    let mut fields = HashMap::new();
    let classes = ["priority", "standard", "batch"];
    for s in 0..48u64 {
        for t in 0..4u64 {
            let id = format!("s{s}-{t}");
            let input = 4 * (50 + 9 * ((s * 7 + t * 3) % 11) as usize);
            let dependencies = if t == 0 {
                serde_json::json!([])
            } else {
                serde_json::json!([{"request_id": format!("s{s}-{}", t - 1),
                    "trigger": "completion", "delay_ms": 500.0 + 300.0 * (s % 5) as f64,
                    "relation": "sequence"}])
            };
            let row = serde_json::json!({"request_id": id, "play_id": format!("p{s}"),
                "session_id": format!("s{s}"), "model": "m",
                "not_before_ms": (s * 40) as f64, "input_length": input, "output_length": 32,
                "hash_ids": (0..input as u64 / 4).map(|b| s * 100_000 + b).collect::<Vec<_>>(),
                "dependencies": dependencies});
            writeln!(file, "{row}").unwrap();
            let class = classes[(s % 3) as usize].to_owned();
            fields.insert(
                id.clone(),
                ReplayRequestFields {
                    request_id: id,
                    session_final: (t == 3).then_some(true),
                    policy_class: Some(class.clone()),
                    agent_headers: BTreeMap::from([
                        ("x-dynamo-meta-policy-class".to_owned(), vec![class]),
                        ("x-dynamo-tenant".to_owned(), vec![format!("t{}", s % 3)]),
                    ]),
                    ..Default::default()
                },
            );
        }
    }
    let fields = Arc::new(fields);
    let run = |admission: &str, recheck: ReplayGateRecheck| {
        let config = policy_with("enforce", admission);
        let scope = ReplayScope {
            request_classifier: Some(factory(&config)),
            request_fields: Some(Arc::clone(&fields)),
            determinism: ReplayDeterminism::CanonicalV1,
            gate_recheck: Some(recheck),
            ..Default::default()
        };
        let args = MockerConfig::from_value(serde_json::json!({
            "engine": {"block_size": 4, "num_gpu_blocks": 600, "max_num_batched_tokens": 512,
                       "max_num_seqs": 16, "speedup_ratio": 1.0}
        }))
        .unwrap();
        let report = with_replay_scope(scope, || {
            simulate_trace_file_with_router_mode_and_format(
                args,
                Some(config),
                None,
                file.path(),
                4,
                2,
                1.0,
                ReplayRouterMode::KvRouter,
                TraceFileFormat::AgenticMooncake,
                0.0,
                0,
                true,
                None,
                SlaThresholds::default(),
            )
        })
        .unwrap()
        .unwrap();
        let mut v: Vec<(String, u64, u64, u64)> = report
            .per_request
            .iter()
            .map(|r| {
                (
                    r.agentic.as_ref().unwrap().request_id.clone(),
                    r.arrival_time_ms.to_bits(),
                    r.first_admit_ms.unwrap_or(-1.0).to_bits(),
                    r.e2e_latency_ms.unwrap_or(-1.0).to_bits(),
                )
            })
            .collect();
        v.sort();
        let held = report
            .per_request
            .iter()
            .filter(|r| {
                r.first_admit_ms
                    .is_some_and(|a| a > r.arrival_time_ms + 1000.0)
            })
            .count();
        (v, held)
    };
    for admission in [
        "",
        "      lobby_order: fair\n",
        "      lobby_order: tenant\n",
        "      pace: {}\n",
    ] {
        let (full, held) = run(admission, ReplayGateRecheck::Full);
        let (keyed, _) = run(admission, ReplayGateRecheck::Keyed);
        assert!(held >= 20, "{admission:?}: only {held} requests waited");
        assert_eq!(full, keyed, "{admission:?}");
    }
}
