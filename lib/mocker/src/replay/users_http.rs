// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Closed-loop users against a live Dynamo frontend over HTTP.
//!
//! Offline replay runs a users schedule (`schedule.json` from the users exporter) through
//! AISimulate's workload driver with the router in process. This module runs the same driver
//! against a real frontend: each ready turn is sent as an OpenAI chat completion with its prompt
//! as token ids (`nvext.token_data`) and the headers a production client sends (session id,
//! parent session, session end, policy class, `x-dynamo-hint-*`), from the schedule's request
//! fields file. A user's next turn starts only when the previous one finishes, as in offline
//! replay. A refusal (HTTP 429) is retried after its `Retry-After`, or, without one, after the
//! same backoff offline replay uses.
//!
//! Time: the driver runs in simulated time, `speedup` times faster than the wall clock. Run the
//! mock engines with the same `speedup_ratio` and the session scheduler with
//! `DYNAMO_SESSION_SCHEDULER_CLOCK_SCALE` set to it, so think times, engine steps and the
//! scheduler's waits and Retry-After advice share one clock.
//!
//! Output, in the offline replay's formats so the same scorers read both: one JSON line per
//! request (`<out>.jsonl`, the per-request record fields the scorers read), the refusal log
//! (`<out>.refusals.jsonl`: each request's first offer and each refusal with its Retry-After),
//! `<out>.options.json` and `<out>.summary.json`.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use futures::StreamExt as _;
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;
use tokio::task::JoinSet;
use tokio::time::Instant;
use uuid::Uuid;

use crate::loadgen::{AgenticUsers, ReadyTurn, WorkloadDriver};
use crate::replay::ReplayTerminalStatus;
use crate::replay::offline::extensions::kv_router::{
    ReplayRequestAnnotations, ReplayRequestFields, client_retry_delay_ms, load_request_annotations,
    resolve_request_fields,
};

/// What a users-over-HTTP run needs. Paths are on the driver's host.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UsersHttpConfig {
    /// The users export directory: `schedule.json`, its templates and `request_fields.users.jsonl`.
    pub users_dir: PathBuf,
    /// The frontend's base URL, for example `http://127.0.0.1:8000`.
    pub url: String,
    /// The served model name.
    pub model: String,
    /// The per-request record file (`<stem>.jsonl`); the other outputs sit beside it.
    pub out: PathBuf,
    /// Simulated seconds per wall second (the engines' `speedup_ratio`).
    #[serde(default = "one")]
    pub speedup: f64,
    /// Stop at this simulated time (s): no new sends; requests still open are recorded unfinished.
    #[serde(default)]
    pub stop_s: Option<f64>,
    /// The engine's KV block size, in tokens (the export's templates are in its blocks).
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    /// Token ids sent are below this (the served tokenizer's vocabulary). The driver's token ids
    /// are mapped into `[1000, vocab_size)` by a hash of the id and its position in its block,
    /// so equal blocks stay equal and distinct blocks stay distinct.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: u32,
    /// Seconds to wait for the model to appear on `/v1/models` before the first send.
    #[serde(default = "default_ready_timeout_s")]
    pub ready_timeout_s: f64,
    /// Settings recorded in `<stem>.options.json` (arm, policy file, engine settings, ...).
    #[serde(default)]
    pub record: serde_json::Map<String, serde_json::Value>,
}

fn one() -> f64 {
    1.0
}
fn default_block_size() -> usize {
    64
}
fn default_vocab_size() -> u32 {
    32_000
}
fn default_ready_timeout_s() -> f64 {
    600.0
}

/// Counts from a finished run.
#[derive(Clone, Debug, Default, Serialize)]
pub struct UsersHttpSummary {
    pub offered: u64,
    pub completed: u64,
    pub refused: u64,
    pub refused_with_retry_after: u64,
    pub failed: u64,
    pub unfinished: u64,
    pub output_tokens: u64,
    pub sim_s: f64,
    pub wall_s: f64,
    pub speedup: f64,
}

#[derive(Default)]
struct Counters {
    offered: AtomicU64,
    completed: AtomicU64,
    refused: AtomicU64,
    refused_ra: AtomicU64,
    failed: AtomicU64,
    output_tokens: AtomicU64,
    in_flight: AtomicU64,
}

/// The run's clock: simulated ms since the start, `speedup` times the wall clock.
#[derive(Clone, Copy)]
struct SimClock {
    start: Instant,
    speedup: f64,
}

impl SimClock {
    fn now_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0 * self.speedup
    }

    fn instant_at(&self, sim_ms: f64) -> Instant {
        self.start + Duration::from_secs_f64((sim_ms / self.speedup / 1000.0).max(0.0))
    }

    fn wall_for(&self, sim_ms: f64) -> Duration {
        Duration::from_secs_f64((sim_ms / self.speedup / 1000.0).max(0.0))
    }
}

/// Line-oriented outputs shared by the request tasks.
struct Outputs {
    requests: std::sync::Mutex<std::io::BufWriter<std::fs::File>>,
    refusals: std::sync::Mutex<std::io::BufWriter<std::fs::File>>,
}

impl Outputs {
    fn write_request(&self, row: &serde_json::Value) {
        let mut out = self.requests.lock().unwrap();
        let _ = writeln!(out, "{row}");
    }

    fn write_refusal(&self, row: &serde_json::Value) {
        let mut out = self.refusals.lock().unwrap();
        let _ = writeln!(out, "{row}");
    }

    fn flush(&self) {
        let _ = self.requests.lock().unwrap().flush();
        let _ = self.refusals.lock().unwrap().flush();
    }
}

struct Shared {
    config: UsersHttpConfig,
    client: reqwest::Client,
    fields: Option<ReplayRequestAnnotations>,
    driver: std::sync::Mutex<WorkloadDriver>,
    wake: Notify,
    clock: SimClock,
    outputs: Outputs,
    counters: Counters,
}

fn stem_of(out: &Path) -> PathBuf {
    let text = out.to_string_lossy();
    PathBuf::from(text.strip_suffix(".jsonl").unwrap_or(&text).to_owned())
}

fn with_suffix(stem: &Path, suffix: &str) -> PathBuf {
    PathBuf::from(format!("{}{suffix}", stem.display()))
}

/// Runs the users schedule against the frontend until the schedule drains or `stop_s`.
pub async fn run_users_http(config: UsersHttpConfig) -> Result<UsersHttpSummary> {
    ensure!(
        config.speedup.is_finite() && config.speedup > 0.0,
        "speedup must be finite and positive"
    );
    ensure!(config.block_size > 0, "block_size must be positive");
    ensure!(config.vocab_size > 2_000, "vocab_size must be above 2000");
    let schedule = config.users_dir.join("schedule.json");
    let users = AgenticUsers::load(&schedule)?;
    let driver = WorkloadDriver::new_agentic_users(users, config.block_size, false)?
        .with_deterministic_request_ids(1);
    let fields_path = config.users_dir.join("request_fields.users.jsonl");
    let fields = if fields_path.exists() {
        Some(load_request_annotations(&fields_path)?)
    } else {
        tracing::warn!(
            path = %fields_path.display(),
            "no request fields: requests carry no class, session or hint headers"
        );
        None
    };
    let stem = stem_of(&config.out);
    if let Some(parent) = stem.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let open = |path: PathBuf| -> Result<std::sync::Mutex<std::io::BufWriter<std::fs::File>>> {
        let file =
            std::fs::File::create(&path).with_context(|| format!("creating {}", path.display()))?;
        Ok(std::sync::Mutex::new(std::io::BufWriter::new(file)))
    };
    let outputs = Outputs {
        requests: open(with_suffix(&stem, ".jsonl"))?,
        refusals: open(with_suffix(&stem, ".refusals.jsonl"))?,
    };
    write_options(&config, &stem, &schedule)?;
    let client = reqwest::Client::builder()
        .pool_idle_timeout(Duration::from_secs(90))
        .tcp_nodelay(true)
        .build()?;
    wait_until_ready(&client, &config).await?;

    let shared = Arc::new(Shared {
        clock: SimClock {
            start: Instant::now(),
            speedup: config.speedup,
        },
        config,
        client,
        fields,
        driver: std::sync::Mutex::new(driver),
        wake: Notify::new(),
        outputs,
        counters: Counters::default(),
    });
    let stop_ms = shared.config.stop_s.map(|s| s * 1000.0);
    let mut tasks: JoinSet<()> = JoinSet::new();
    let mut next_progress_ms = 0.0;
    let mut open: BTreeMap<Uuid, OpenRequest> = BTreeMap::new();
    let (done_tx, mut done_rx) = tokio::sync::mpsc::unbounded_channel::<Uuid>();

    loop {
        let now = shared.clock.now_ms();
        if stop_ms.is_some_and(|stop| now >= stop) {
            break;
        }
        if now >= next_progress_ms {
            log_progress(&shared, now);
            next_progress_ms = now + 60_000.0;
        }
        while let Ok(uuid) = done_rx.try_recv() {
            open.remove(&uuid);
        }
        let ready = tokio::task::block_in_place(|| {
            shared.driver.lock().unwrap().pop_ready(now, usize::MAX)
        });
        for turn in ready {
            let request = OpenRequest::new(&shared, turn, now);
            open.insert(request.uuid, request.clone());
            let shared = Arc::clone(&shared);
            let done_tx = done_tx.clone();
            tasks.spawn(async move {
                let uuid = request.uuid;
                run_request(&shared, request).await;
                let _ = done_tx.send(uuid);
            });
        }
        while tasks.try_join_next().is_some() {}
        let wake = shared.wake.notified();
        tokio::pin!(wake);
        let (drained, next_ms) = {
            let mut driver = shared.driver.lock().unwrap();
            (driver.is_drained(), driver.next_ready_time_ms())
        };
        if drained && tasks.is_empty() {
            break;
        }
        let mut deadline_ms = next_ms.unwrap_or(f64::INFINITY).min(next_progress_ms);
        if let Some(stop) = stop_ms {
            deadline_ms = deadline_ms.min(stop);
        }
        let deadline = shared.clock.instant_at(deadline_ms.max(now));
        tokio::select! {
            _ = tokio::time::sleep_until(deadline) => {}
            _ = wake.as_mut() => {}
        }
    }

    // Stop: drop what is still running and record it unfinished, as offline replay reports the
    // requests still open at `max_sim_time_ms`.
    tasks.abort_all();
    while tasks.join_next().await.is_some() {}
    while let Ok(uuid) = done_rx.try_recv() {
        open.remove(&uuid);
    }
    let end_ms = shared.clock.now_ms();
    for request in open.values() {
        if request.finished.load(Ordering::Acquire) {
            continue;
        }
        shared
            .outputs
            .write_request(&request.row(&Progress::default(), None, "unfinished"));
    }
    shared.outputs.flush();
    let counters = &shared.counters;
    let summary = UsersHttpSummary {
        offered: counters.offered.load(Ordering::Relaxed),
        completed: counters.completed.load(Ordering::Relaxed),
        refused: counters.refused.load(Ordering::Relaxed),
        refused_with_retry_after: counters.refused_ra.load(Ordering::Relaxed),
        failed: counters.failed.load(Ordering::Relaxed),
        unfinished: open
            .values()
            .filter(|r| !r.finished.load(Ordering::Acquire))
            .count() as u64,
        output_tokens: counters.output_tokens.load(Ordering::Relaxed),
        sim_s: end_ms / 1000.0,
        wall_s: shared.clock.start.elapsed().as_secs_f64(),
        speedup: shared.config.speedup,
    };
    let stem = stem_of(&shared.config.out);
    std::fs::write(
        with_suffix(&stem, ".summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(summary)
}

fn log_progress(shared: &Shared, now_ms: f64) {
    let c = &shared.counters;
    tracing::info!(
        sim_s = format!("{:.0}", now_ms / 1000.0),
        offered = c.offered.load(Ordering::Relaxed),
        in_flight = c.in_flight.load(Ordering::Relaxed),
        completed = c.completed.load(Ordering::Relaxed),
        refused = c.refused.load(Ordering::Relaxed),
        failed = c.failed.load(Ordering::Relaxed),
        "users over HTTP"
    );
    shared.outputs.flush();
}

fn write_options(config: &UsersHttpConfig, stem: &Path, schedule: &Path) -> Result<()> {
    let sched: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(schedule)?)?;
    let pick = |key: &str| sched.get(key).cloned().unwrap_or(serde_json::Value::Null);
    let users_dir = std::fs::canonicalize(&config.users_dir).unwrap_or(config.users_dir.clone());
    let mut options = serde_json::json!({
        "schema": "dynosim.run-options/1",
        "script": "dynamo.replay.users_http (live frontend, mocker engines)",
        "argv": [],
        "users": {
            "users": pick("users"), "hours": pick("hours"), "horizon_s": pick("horizon_s"),
            "warm": pick("warm"), "cooldown_s": pick("cooldown_s"),
            "loss_cooldown_s": pick("loss_cooldown_s"), "seed": pick("seed"),
            "dir": users_dir,
        },
        "flags": {
            "live": true, "url": config.url, "model": config.model, "speedup": config.speedup,
            "stop_s": config.stop_s, "block_size": config.block_size,
            "vocab_size": config.vocab_size, "refusal_log": true,
        },
    });
    for (key, value) in &config.record {
        options[key] = value.clone();
    }
    std::fs::write(
        with_suffix(stem, ".options.json"),
        serde_json::to_vec_pretty(&options)?,
    )?;
    Ok(())
}

async fn wait_until_ready(client: &reqwest::Client, config: &UsersHttpConfig) -> Result<()> {
    let url = format!("{}/v1/models", config.url.trim_end_matches('/'));
    let deadline = std::time::Instant::now() + Duration::from_secs_f64(config.ready_timeout_s);
    loop {
        if let Ok(response) = client.get(&url).send().await
            && response.status().is_success()
            && let Ok(body) = response.json::<serde_json::Value>().await
            && body["data"]
                .as_array()
                .is_some_and(|models| models.iter().any(|m| m["id"] == config.model.as_str()))
        {
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            bail!(
                "model {} did not appear on {url} within {} s",
                config.model,
                config.ready_timeout_s
            );
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

/// One request of the schedule, from its first offer to its terminal outcome.
#[derive(Clone)]
struct OpenRequest {
    uuid: Uuid,
    request_id: String,
    play_id: Option<String>,
    session_id: String,
    class: Option<String>,
    headers: Arc<Vec<(String, String)>>,
    body: Arc<bytes::Bytes>,
    input_length: usize,
    output_length: usize,
    offered_ms: f64,
    finished: Arc<std::sync::atomic::AtomicBool>,
}

/// What a request's attempts observed.
#[derive(Default)]
struct Progress {
    attempt_ms: Option<f64>,
    first_token_ms: Option<f64>,
    last_token_ms: Option<f64>,
    terminal_ms: Option<f64>,
    output_tokens: usize,
    attempts: u32,
}

impl OpenRequest {
    fn new(shared: &Shared, turn: ReadyTurn, now_ms: f64) -> Self {
        let ReadyTurn {
            request_uuid,
            authored_request_id,
            play_id,
            session_id,
            request,
            ..
        } = turn;
        let request_id = request
            .replay_context
            .as_ref()
            .and_then(|context| context.agentic.as_ref())
            .map(|agentic| agentic.request_id.clone())
            .or(authored_request_id)
            .unwrap_or_else(|| request_uuid.to_string());
        let fields = shared
            .fields
            .as_ref()
            .and_then(|fields| resolve_request_fields(fields, &request_id));
        let headers = request_headers(fields.as_deref(), &session_id);
        let class = fields.as_deref().and_then(|f| f.policy_class.clone());
        let session_id = fields
            .as_deref()
            .and_then(|f| f.session_id.clone())
            .unwrap_or(session_id);
        let tokens = map_tokens(
            &request.tokens,
            shared.config.block_size,
            shared.config.vocab_size,
        );
        let body = request_body(&shared.config.model, &tokens, request.max_output_tokens);
        Self {
            uuid: request_uuid,
            request_id,
            play_id,
            session_id,
            class,
            headers: Arc::new(headers),
            body: Arc::new(bytes::Bytes::from(body)),
            input_length: request.tokens.len(),
            output_length: request.max_output_tokens,
            offered_ms: now_ms,
            finished: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// The per-request record, in offline replay's per-request field names. `arrival_time_ms`
    /// is the last attempt's send, as offline replay stamps a retried request, so TTFT leaves
    /// out the time refused.
    fn row(&self, progress: &Progress, worker: Option<u64>, status: &str) -> serde_json::Value {
        let arrival = progress.attempt_ms.unwrap_or(self.offered_ms);
        let ttft = progress.first_token_ms.map(|t| t - arrival);
        let itl = match (
            progress.first_token_ms,
            progress.last_token_ms,
            progress.output_tokens,
        ) {
            (Some(first), Some(last), n) if n > 1 => Some((last - first) / (n - 1) as f64),
            _ => None,
        };
        serde_json::json!({
            "request_id": self.request_id,
            "play_id": self.play_id,
            "session_id": self.session_id,
            "policy_class": self.class,
            "uuid": self.uuid.to_string(),
            "arrival_time_ms": arrival,
            "dispatched_at_ms": self.offered_ms,
            "first_admit_ms": serde_json::Value::Null,
            "terminal_time_ms": progress.terminal_ms,
            "first_token_ms": progress.first_token_ms,
            "last_token_ms": progress.last_token_ms,
            "ttft_ms": ttft,
            "e2e_latency_ms": progress.terminal_ms.map(|t| t - arrival),
            "itl_ms": itl,
            "input_length": self.input_length,
            "requested_output_length": self.output_length,
            "output_length": progress.output_tokens,
            "decode_worker_idx": worker,
            "attempts": progress.attempts,
            "terminal_status": status,
        })
    }
}

/// The headers a production client sends for this request: session id, parent session,
/// session end, the policy class and the export's agent headers (`x-dynamo-hint-*`, ...).
fn request_headers(
    fields: Option<&ReplayRequestFields>,
    session_id: &str,
) -> Vec<(String, String)> {
    let mut headers = Vec::new();
    let session = fields
        .and_then(|f| f.session_id.as_deref())
        .unwrap_or(session_id);
    headers.push(("x-dynamo-session-id".to_owned(), session.to_owned()));
    let Some(fields) = fields else {
        return headers;
    };
    if let Some(parent) = &fields.parent_session_id {
        headers.push(("x-dynamo-parent-session-id".to_owned(), parent.clone()));
    }
    if fields.session_final == Some(true) {
        headers.push(("x-dynamo-session-final".to_owned(), "true".to_owned()));
    }
    let class_header = "x-dynamo-meta-policy-class";
    if let Some(class) = &fields.policy_class
        && !fields.agent_headers.contains_key(class_header)
    {
        headers.push((class_header.to_owned(), class.clone()));
    }
    for (name, values) in &fields.agent_headers {
        for value in values {
            headers.push((name.clone(), value.clone()));
        }
    }
    headers
}

/// The driver's token ids mapped into `[1000, vocab_size)`: a hash of the id and its position in
/// its block, so a block's content decides its ids (equal blocks stay equal) and distinct blocks
/// do not collide in practice.
fn map_tokens(tokens: &[u32], block_size: usize, vocab_size: u32) -> Vec<u32> {
    const LOW: u64 = 1_000;
    let span = u64::from(vocab_size) - LOW;
    tokens
        .iter()
        .enumerate()
        .map(|(index, &token)| {
            let key = (u64::from(token) << 16) ^ (index % block_size) as u64;
            (LOW + splitmix64(key) % span) as u32
        })
        .collect()
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// A streamed chat completion whose prompt is `tokens` (no tokenizer on the way in) and whose
/// output is exactly `max_tokens` long. `nvext.agent_hints.osl` gives the router the expected
/// output length, as offline replay does.
fn request_body(model: &str, tokens: &[u32], max_tokens: usize) -> Vec<u8> {
    let mut ids = String::with_capacity(tokens.len() * 6 + 2);
    ids.push('[');
    for (i, token) in tokens.iter().enumerate() {
        if i > 0 {
            ids.push(',');
        }
        ids.push_str(&token.to_string());
    }
    ids.push(']');
    let max_tokens = max_tokens.max(1);
    format!(
        concat!(
            "{{\"model\":{model},\"messages\":[{{\"role\":\"user\",\"content\":\"-\"}}],",
            "\"max_tokens\":{max},\"min_tokens\":{max},\"ignore_eos\":true,\"stream\":true,",
            "\"stream_options\":{{\"include_usage\":true}},",
            "\"nvext\":{{\"token_data\":{ids},\"agent_hints\":{{\"osl\":{max}}}}}}}"
        ),
        model = serde_json::Value::from(model),
        max = max_tokens,
        ids = ids,
    )
    .into_bytes()
}

enum Attempt {
    Done(Progress),
    Refused(Option<f64>),
    Failed(String),
}

async fn run_request(shared: &Shared, request: OpenRequest) {
    let counters = &shared.counters;
    counters.offered.fetch_add(1, Ordering::Relaxed);
    counters.in_flight.fetch_add(1, Ordering::Relaxed);
    shared.outputs.write_refusal(&serde_json::json!({
        "t": request.offered_ms / 1000.0, "kind": "offer", "uuid": request.uuid.to_string(),
        "class": request.class, "session": request.session_id, "retry_after_s": null,
    }));
    let mut attempt = 0u32;
    let status = loop {
        let sent_ms = shared.clock.now_ms();
        match send_once(shared, &request, sent_ms).await {
            Attempt::Done(mut progress) => {
                progress.attempts = attempt + 1;
                counters.completed.fetch_add(1, Ordering::Relaxed);
                counters
                    .output_tokens
                    .fetch_add(progress.output_tokens as u64, Ordering::Relaxed);
                request.finished.store(true, Ordering::Release);
                shared
                    .outputs
                    .write_request(&request.row(&progress, None, "completed"));
                break ReplayTerminalStatus::Completed;
            }
            Attempt::Refused(retry_after_s) => {
                let now = shared.clock.now_ms();
                counters.refused.fetch_add(1, Ordering::Relaxed);
                if retry_after_s.is_some() {
                    counters.refused_ra.fetch_add(1, Ordering::Relaxed);
                }
                shared.outputs.write_refusal(&serde_json::json!({
                    "t": now / 1000.0, "kind": "refused", "uuid": request.uuid.to_string(),
                    "class": request.class, "session": request.session_id,
                    "retry_after_s": retry_after_s,
                }));
                // The server's Retry-After (simulated seconds), else offline replay's backoff.
                let wait_ms = retry_after_s
                    .map(|s| s * 1000.0)
                    .unwrap_or_else(|| client_retry_delay_ms(attempt));
                attempt += 1;
                tokio::time::sleep(shared.clock.wall_for(wait_ms)).await;
            }
            Attempt::Failed(error) => {
                counters.failed.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(request_id = %request.request_id, %error, "request failed");
                request.finished.store(true, Ordering::Release);
                let progress = Progress {
                    attempt_ms: Some(sent_ms),
                    terminal_ms: Some(shared.clock.now_ms()),
                    attempts: attempt + 1,
                    ..Default::default()
                };
                shared
                    .outputs
                    .write_request(&request.row(&progress, None, "failed"));
                break ReplayTerminalStatus::Failed;
            }
        }
    };
    counters.in_flight.fetch_sub(1, Ordering::Relaxed);
    let now = shared.clock.now_ms();
    let result = tokio::task::block_in_place(|| {
        shared
            .driver
            .lock()
            .unwrap()
            .on_terminal(request.uuid, now, status)
    });
    if let Err(error) = result {
        tracing::error!(%error, request_id = %request.request_id, "users driver rejected a terminal");
    }
    shared.wake.notify_one();
}

/// One HTTP attempt: a refusal (429) with its Retry-After, a failure, or the streamed result.
async fn send_once(shared: &Shared, request: &OpenRequest, sent_ms: f64) -> Attempt {
    let url = format!(
        "{}/v1/chat/completions",
        shared.config.url.trim_end_matches('/')
    );
    let mut builder = shared
        .client
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(request.body.as_ref().clone());
    for (name, value) in request.headers.iter() {
        builder = builder.header(name.as_str(), value.as_str());
    }
    let response = match builder.send().await {
        Ok(response) => response,
        Err(error) => return Attempt::Failed(format!("send: {error}")),
    };
    let status = response.status();
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        let retry_after = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.trim().parse::<f64>().ok());
        return Attempt::Refused(retry_after);
    }
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Attempt::Failed(format!("HTTP {status}: {}", truncate(&body, 400)));
    }
    let mut progress = Progress {
        attempt_ms: Some(sent_ms),
        ..Default::default()
    };
    let mut usage_tokens: Option<usize> = None;
    let mut chunks = 0usize;
    let mut buffer: Vec<u8> = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(item) = stream.next().await {
        let bytes = match item {
            Ok(bytes) => bytes,
            Err(error) => return Attempt::Failed(format!("stream: {error}")),
        };
        let now = shared.clock.now_ms();
        buffer.extend_from_slice(&bytes);
        while let Some(end) = buffer.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buffer.drain(..=end).collect();
            let line = String::from_utf8_lossy(&line);
            let Some(data) = line.trim().strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data == "[DONE]" {
                continue;
            }
            let Ok(event) = serde_json::from_str::<serde_json::Value>(data) else {
                continue;
            };
            if let Some(error) = event.get("error") {
                return Attempt::Failed(format!("stream error: {error}"));
            }
            if event["choices"].as_array().is_some_and(|c| !c.is_empty()) {
                chunks += 1;
                progress.first_token_ms.get_or_insert(now);
                progress.last_token_ms = Some(now);
            }
            if let Some(tokens) = event["usage"]["completion_tokens"].as_u64() {
                usage_tokens = Some(tokens as usize);
            }
        }
    }
    progress.terminal_ms = Some(shared.clock.now_ms());
    progress.output_tokens = usage_tokens.unwrap_or(chunks);
    if progress.first_token_ms.is_none() {
        return Attempt::Failed("stream ended without a token".to_owned());
    }
    Attempt::Done(progress)
}

fn truncate(text: &str, max: usize) -> &str {
    match text.char_indices().nth(max) {
        Some((index, _)) => &text[..index],
        None => text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapped_tokens_keep_block_equality_and_stay_in_the_vocabulary() {
        let a: Vec<u32> = (0..128).map(|i| 7 + i / 64).collect();
        let mut b = a.clone();
        b[100] = 9;
        let ma = map_tokens(&a, 64, 32_000);
        let mb = map_tokens(&b, 64, 32_000);
        assert_eq!(ma[..64], mb[..64]);
        assert_ne!(ma[64..], mb[64..]);
        assert!(ma.iter().all(|&t| (1_000..32_000).contains(&t)));
        // A block of one repeated id is not one repeated token, and ids a vocabulary apart differ.
        assert!(ma[..64].windows(2).any(|w| w[0] != w[1]));
        let far = map_tokens(&[7 + 31_000; 64], 64, 32_000);
        assert_ne!(far[..], map_tokens(&[7; 64], 64, 32_000)[..]);
    }

    #[test]
    fn the_body_carries_token_ids_output_length_and_hint() {
        let body = request_body("m", &[1001, 1002], 5);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["model"], "m");
        assert_eq!(value["max_tokens"], 5);
        assert_eq!(value["stream"], true);
        assert_eq!(
            value["nvext"]["token_data"],
            serde_json::json!([1001, 1002])
        );
        assert_eq!(value["nvext"]["agent_hints"]["osl"], 5);
    }

    #[test]
    fn headers_follow_the_request_fields() {
        let fields: ReplayRequestFields = serde_json::from_value(serde_json::json!({
            "request_id": "cp000001000002:r",
            "session_id": "cp000001000002:s",
            "parent_session_id": "cp000001000002:p",
            "session_final": true,
            "policy_class": "priority",
            "agent_headers": {"x-dynamo-hint-osl": ["20"], "x-dynamo-meta-policy-class": ["priority"]}
        }))
        .unwrap();
        let headers = request_headers(Some(&fields), "replay-session");
        let get = |name: &str| {
            headers
                .iter()
                .filter(|(n, _)| n == name)
                .map(|(_, v)| v.as_str())
                .collect::<Vec<_>>()
        };
        assert_eq!(get("x-dynamo-session-id"), ["cp000001000002:s"]);
        assert_eq!(get("x-dynamo-parent-session-id"), ["cp000001000002:p"]);
        assert_eq!(get("x-dynamo-session-final"), ["true"]);
        assert_eq!(get("x-dynamo-meta-policy-class"), ["priority"]);
        assert_eq!(get("x-dynamo-hint-osl"), ["20"]);
        let bare = request_headers(None, "replay-session");
        assert_eq!(
            bare,
            vec![(
                "x-dynamo-session-id".to_owned(),
                "replay-session".to_owned()
            )]
        );
    }
}
