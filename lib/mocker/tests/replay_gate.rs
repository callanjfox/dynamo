// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The replay gate: a request classifier and its dispatch gate run inside offline KV-router
//! replay, in virtual time. A test classifier exercises each path: a hold released by a
//! completion (`notify`), a hold that ends at its deadline in an otherwise idle cluster
//! (`next_wakeup_ms`), a refusal the client retries (429), a `due_at` expiry the client retries
//! (504), and the per-request fields (session end, agent headers) reaching `classify`. With
//! keyed rechecks, holds the gate keys are answered without asking it until their key wakes, and
//! the run is the same as with the full recheck.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dynamo_admission::{
    ClassifierError, ClassifyEvent, ClassifyRequest, DispatchDecision, DispatchGate, DispatchHead,
    DispatchRecheck, HoldWake, RequestClassifier, RequestClassifierContext, WorkerView,
};
use dynamo_kv_router::plugins::request_classifier::RequestClassifierFactory;
use dynamo_mocker::common::protocols::MockerConfig;
use dynamo_mocker::loadgen::TraceFileFormat;
use dynamo_mocker::replay::{
    ReplayClassifierPlugin, ReplayGateRecheck, ReplayRequestFields, ReplayRouterMode, ReplayScope,
    SlaThresholds, TraceSimulationReport, simulate_trace_file_with_router_mode_and_format,
    with_replay_scope,
};
use parking_lot::Mutex;

#[derive(Default)]
struct State {
    /// Session of each request id the classifier saw.
    sessions: HashMap<String, String>,
    /// Sessions whose requests completed.
    completed: Vec<String>,
    /// Classify calls per session.
    classified: HashMap<String, u32>,
    /// What each classified request carried: (session_final, tenant header).
    attributes: HashMap<String, (Option<bool>, Option<String>)>,
    recheck: Option<DispatchRecheck>,
    /// Gate checks per session.
    checks: HashMap<String, u32>,
}

/// The key "second" and "third" wait on: "first" completing.
const FIRST_DONE: u64 = 1;

struct TestClassifier {
    state: Arc<Mutex<State>>,
    context: RequestClassifierContext,
}

#[async_trait]
impl RequestClassifier for TestClassifier {
    fn classify(
        &mut self,
        mut request: ClassifyRequest,
        _workers: &dyn WorkerView,
    ) -> Result<ClassifyRequest, Box<ClassifierError>> {
        let session = request
            .session_context()
            .map(|c| c.session_id().to_owned())
            .unwrap_or_default();
        let attributes = request.session_context().map(|c| {
            (
                c.session_final(),
                c.agent_headers()
                    .get("x-dynamo-tenant")
                    .and_then(|v| v.first().cloned()),
            )
        });
        let mut state = self.state.lock();
        let calls = {
            let calls = state.classified.entry(session.clone()).or_default();
            *calls += 1;
            *calls
        };
        if let Some(attributes) = attributes {
            state.attributes.insert(session.clone(), attributes);
        }
        state
            .sessions
            .insert(request.request_id().unwrap().to_owned(), session.clone());
        // The first request of "refused" is refused once; its client retries.
        if session == "refused" && calls == 1 {
            return Err("over quota".into());
        }
        // "expiring" is held until its 50 ms deadline passes the first time (504, retried).
        if session == "expiring" && calls == 1 {
            request.set_due_at(self.context.now() + Duration::from_millis(50));
        }
        Ok(request)
    }

    async fn on_event(&mut self, event: ClassifyEvent) {
        if let ClassifyEvent::Completed { request_id, .. } = event {
            let mut state = self.state.lock();
            if let Some(session) = state.sessions.get(&request_id).cloned() {
                if session == "first"
                    && let Some(recheck) = &state.recheck
                {
                    recheck.wake_keys(&[FIRST_DONE]);
                }
                state.completed.push(session);
            }
            if let Some(recheck) = &state.recheck {
                recheck.notify();
            }
        }
    }

    fn dispatch_gate(&self) -> Option<Arc<dyn DispatchGate>> {
        Some(Arc::new(TestGate {
            state: Arc::clone(&self.state),
            context: self.context.clone(),
        }))
    }
}

struct TestGate {
    state: Arc<Mutex<State>>,
    context: RequestClassifierContext,
}

impl DispatchGate for TestGate {
    fn check(&self, head: &DispatchHead<'_>) -> DispatchDecision {
        let session = head.session_context().map(|c| c.session_id()).unwrap_or("");
        let mut state = self.state.lock();
        *state.checks.entry(session.to_owned()).or_default() += 1;
        match session {
            // Wait for any request of "first" to complete; woken by notify.
            "second" | "third" if !state.completed.iter().any(|s| s == "first") => {
                DispatchDecision::Hold { until: None }
            }
            // Waits 400 ms after it queued, with nothing else to wake the replay.
            "timed" if self.context.now() < head.queued_at() + Duration::from_millis(400) => {
                DispatchDecision::Hold {
                    until: Some(head.queued_at() + Duration::from_millis(400)),
                }
            }
            // Held (with no deadline of its own) until its due_at expires it, the first time.
            "expiring" if state.classified.get("expiring").copied() == Some(1) => {
                DispatchDecision::Hold { until: None }
            }
            _ => DispatchDecision::Go { target: None },
        }
    }

    /// The holds of "second" and "third" end only when "first" completes.
    fn check_wake(&self, head: &DispatchHead<'_>) -> (DispatchDecision, HoldWake) {
        let decision = self.check(head);
        let session = head.session_context().map(|c| c.session_id()).unwrap_or("");
        let wake = match (decision, session) {
            (DispatchDecision::Hold { .. }, "second" | "third") => HoldWake::Keys(vec![FIRST_DONE]),
            _ => HoldWake::Any,
        };
        (decision, wake)
    }

    fn attach(&self, recheck: DispatchRecheck) {
        self.state.lock().recheck = Some(recheck);
    }
}

fn factory(state: Arc<Mutex<State>>) -> RequestClassifierFactory {
    Arc::new(move |context| {
        Box::new(TestClassifier {
            state: Arc::clone(&state),
            context,
        })
    })
}

fn row(id: &str, session: &str, at_ms: f64, input: usize) -> serde_json::Value {
    serde_json::json!({
        "request_id": id,
        "play_id": format!("play-{session}"),
        "session_id": session,
        "model": "model",
        "not_before_ms": at_ms,
        "input_length": input,
        "output_length": 4,
        "hash_ids": (0..input / 4).map(|i| i as u64 + 1000 * id.len() as u64).collect::<Vec<_>>(),
        "dependencies": []
    })
}

fn args() -> MockerConfig {
    MockerConfig::from_value(serde_json::json!({
        "engine": {
            "block_size": 4,
            "num_gpu_blocks": 256,
            "max_num_batched_tokens": 64,
            "max_num_seqs": 8,
            "speedup_ratio": 1000.0
        }
    }))
    .unwrap()
}

fn run(state: Arc<Mutex<State>>, trace: &std::path::Path) -> TraceSimulationReport {
    run_with(state, trace, None)
}

fn run_with(
    state: Arc<Mutex<State>>,
    trace: &std::path::Path,
    gate_recheck: Option<ReplayGateRecheck>,
) -> TraceSimulationReport {
    let fields: HashMap<String, ReplayRequestFields> = [ReplayRequestFields {
        request_id: "first-0".into(),
        session_final: Some(true),
        agent_headers: BTreeMap::from([("x-dynamo-tenant".into(), vec!["acme".into()])]),
        ..Default::default()
    }]
    .into_iter()
    .map(|f| (f.request_id.clone(), f))
    .collect();
    let scope = ReplayScope {
        request_classifier: Some(ReplayClassifierPlugin(factory(state))),
        request_fields: Some(Arc::new(fields)),
        determinism: dynamo_mocker::replay::ReplayDeterminism::CanonicalV1,
        gate_recheck,
        ..Default::default()
    };
    with_replay_scope(scope, || {
        simulate_trace_file_with_router_mode_and_format(
            args(),
            // A gate acts on queued requests: queueing needs a busy threshold, live as in replay.
            Some(dynamo_kv_router::config::KvRouterConfig {
                router_queue_threshold: Some(0.5),
                ..Default::default()
            }),
            None,
            trace,
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

fn trace_file() -> tempfile::NamedTempFile {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    let header = serde_json::json!({
        "schema": "dynamo.agentic_mooncake",
        "version": 2,
        "block_size": 4,
        "hash_id_scope": "local",
        "source": {"format": "test", "digest": "replay-gate"}
    });
    writeln!(file, "{header}").unwrap();
    for value in [
        row("first-0", "first", 0.0, 32),
        row("second-0", "second", 0.0, 32),
        row("third-0", "third", 0.0, 32),
        row("timed-0", "timed", 5_000.0, 32),
        row("refused-0", "refused", 10_000.0, 32),
        row("expiring-0", "expiring", 20_000.0, 32),
    ] {
        writeln!(file, "{value}").unwrap();
    }
    file
}

fn dispatch_ms(report: &TraceSimulationReport, id: &str) -> (f64, f64) {
    let record = report
        .per_request
        .iter()
        .find(|r| {
            r.agentic.as_ref().map(|a| a.request_id.as_str()) == Some(id)
                || r.request_id.as_deref() == Some(id)
        })
        .unwrap_or_else(|| panic!("no record for {id}"));
    // First admission to a worker: after any router hold and client retry.
    (record.arrival_time_ms, record.first_admit_ms.unwrap())
}

#[test]
fn the_replay_gate_holds_wakes_retries_and_expires_in_virtual_time() {
    let trace = trace_file();
    let state = Arc::new(Mutex::new(State::default()));
    let report = run(Arc::clone(&state), trace.path());
    assert_eq!(report.request_counts.completed_requests, 6);

    // "second" waits for "first" to complete: it goes after first's dispatch, not at arrival.
    let (_, first_dispatch) = dispatch_ms(&report, "first-0");
    let (second_arrival, second_dispatch) = dispatch_ms(&report, "second-0");
    let first_end = report
        .per_request
        .iter()
        .find(|r| {
            r.agentic
                .as_ref()
                .is_some_and(|a| a.request_id == "first-0")
        })
        .and_then(|r| r.e2e_latency_ms.map(|e2e| r.arrival_time_ms + e2e))
        .unwrap();
    assert!(
        second_dispatch > second_arrival && second_dispatch >= first_end - 1e-6,
        "second dispatched at {second_dispatch} (arrived {second_arrival}); first dispatched at {first_dispatch}, ended {first_end}"
    );

    // "timed" is released by its own deadline in an idle cluster, 400 ms after it queued.
    let (timed_arrival, timed_dispatch) = dispatch_ms(&report, "timed-0");
    assert!(
        (timed_dispatch - timed_arrival - 400.0).abs() < 1.0,
        "timed held {} ms",
        timed_dispatch - timed_arrival
    );

    // "refused" comes back after the client's 2 s backoff.
    let (refused_arrival, refused_dispatch) = dispatch_ms(&report, "refused-0");
    assert!(
        (refused_dispatch - refused_arrival - 2_000.0).abs() < 1.0,
        "refused waited {} ms",
        refused_dispatch - refused_arrival
    );

    // "expiring" expires at 50 ms, then retries 2 s later and goes.
    let (expiring_arrival, expiring_dispatch) = dispatch_ms(&report, "expiring-0");
    assert!(
        (expiring_dispatch - expiring_arrival - 2_050.0).abs() < 1.0,
        "expiring waited {} ms",
        expiring_dispatch - expiring_arrival
    );

    let state = state.lock();
    assert_eq!(state.classified["refused"], 2);
    assert_eq!(state.classified["expiring"], 2);
    // The request fields reached classify: session end and the tenant header.
    assert_eq!(
        state.attributes["first"],
        (Some(true), Some("acme".to_owned()))
    );
    assert_eq!(state.attributes["second"], (None, None));
}

#[test]
fn the_replay_gate_is_deterministic() {
    let trace = trace_file();
    let a = run(Arc::new(Mutex::new(State::default())), trace.path());
    let b = run(Arc::new(Mutex::new(State::default())), trace.path());
    assert_eq!(times(&a), times(&b));
}

/// (request id, first admission, arrival, end) of every request, by id.
fn times(report: &TraceSimulationReport) -> Vec<(String, f64, f64, f64)> {
    let mut v: Vec<_> = report
        .per_request
        .iter()
        .map(|r| {
            (
                r.agentic
                    .as_ref()
                    .map(|a| a.request_id.clone())
                    .unwrap_or_default(),
                r.first_admit_ms.unwrap_or(-1.0),
                r.arrival_time_ms,
                r.arrival_time_ms + r.e2e_latency_ms.unwrap_or(-1.0),
            )
        })
        .collect();
    v.sort_by(|x, y| x.0.cmp(&y.0));
    v
}

#[test]
fn keyed_holds_are_answered_without_asking_and_change_nothing() {
    let trace = trace_file();
    let full_state = Arc::new(Mutex::new(State::default()));
    let full = run_with(
        Arc::clone(&full_state),
        trace.path(),
        Some(ReplayGateRecheck::Full),
    );
    let keyed_state = Arc::new(Mutex::new(State::default()));
    let keyed = run_with(
        Arc::clone(&keyed_state),
        trace.path(),
        Some(ReplayGateRecheck::Keyed),
    );
    assert_eq!(times(&full), times(&keyed));
    let asked = |state: &Arc<Mutex<State>>| {
        let state = state.lock();
        state.checks["second"] + state.checks["third"]
    };
    // Each drain asks the first keyed hold it meets (a gate's upkeep runs as with the full
    // recheck) and answers the other from its hold.
    let (full_asks, keyed_asks) = (asked(&full_state), asked(&keyed_state));
    assert!(
        keyed_asks < full_asks,
        "keyed asked {keyed_asks} times, full {full_asks}"
    );
    // Other gates' holds are asked as before.
    assert_eq!(
        full_state.lock().checks["timed"],
        keyed_state.lock().checks["timed"]
    );
}
