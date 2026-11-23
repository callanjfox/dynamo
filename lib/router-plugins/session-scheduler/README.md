<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-session-scheduler

This crate is a router plugin. It admits agentic sessions against the KV budget, and it scores
per-class service-level attainment the way the admission simulator (`dynamo-admission-sim`)
does. It plugs into the router as a `RequestClassifier` with a `DispatchGate`, using the API in
PR #18443 (head `9be15177`). Its only Dynamo dependency is `dynamo-kv-router`, which is the
plugin rule in `scripts/router_layering_check.sh`.

This is build step 1: the foundation and the first admission policy. It has no forecast yet.

## Modules

| Module | What it does |
|---|---|
| `clock` | Provides an injected `Clock`. `HostClock` follows tokio time, including paused time. `ManualClock` is for tests. |
| `config` | Defines `ServicePolicy`, the `parameters` payload. It has defaults and validation errors that name the field. |
| `session` | Holds the session table, bounded by least-recently-used eviction. Sessions with a turn in flight are evicted last. |
| `metrics` | Computes per-request attainment and windowed-percentile attainment exactly as in `docs/slo-attainment.md`. It returns a plain `ClassSnapshot`, not Prometheus. |
| `admit` | The KV-budget session gate. It keeps a lobby ordered by class rank, then first come (or `fair`), and is first in, first out within a class. Decisions are admit, hold (with a reason and an until), expire, or throttle. It also holds the bounded, deduplicated decision log. |
| `scheduler` | The state machine that the hooks drive. It contains no router types. |
| `forecast` | The forecast, off the classifier lock: per session the next turn (time, new tokens, output tokens), per worker the prefill backlog and decode step times over a horizon, `what_if` misses for admitting a turn on a worker, and per-class prediction errors. Fed through a bounded, non-blocking channel. |
| `hooks` | Adapters to `RequestClassifier` and `DispatchGate`, `register`, and the `SessionIndexView` stub. |

## Configuration

Configure the plugin in the router policy file, under `request_classifier`:

```yaml
request_classifier:
  type: session-scheduler
  parameters:
    mode: shadow            # off | shadow (default) | enforce
    default_class: standard
    admission: { lobby_order: fifo, total_headroom: 0.95, dram_tokens_per_worker: 0, max_lobby_sessions: 0 }
    # classes: [...]        # defaults: priority 3 s / 50 ms / 90%, standard 8 s / 100 ms / 70%, batch none
```

The modes behave as follows:

- **Off** installs no gate and passes every request through.
- **Shadow** decides and logs every decision, but lets every head go.
- **Enforce** holds and expires requests.

A request is scheduled only when it carries both a session context and a request ID. Every
other request passes through untouched.

## Testing

```bash
cargo test -p dynamo-session-scheduler
cargo test -p dynamo-session-scheduler --features aisimulate   # AISIM_FORECAST_CONFIG=<json> to price real steps
cargo clippy --locked --no-deps -p dynamo-session-scheduler --all-targets -- -D warnings
```

The tests use `ManualClock` or paused tokio time. `tests/no_system_clock.rs` fails if any module
other than `clock.rs` reads the host clock. `tests/policy_file.rs` loads a realistic router
policy file through the registry.

## Rules

- **Time.** Read time only through the injected clock.
- **Classifier path.** `classify` and `on_event` share the router's classifier lock, and the
  event channel is unbounded. Both must do O(1) amortized work, with no waiting and no
  forecasting.
- **Gate checks.** The gate's `check` is a read. It may be asked many times about one head
  ("a Go is a permission, not a dispatch"), so it writes only the decision log. Admission is
  committed in `dispatched`.
- **Late completions.** `Completed` can land after the next turn of the same session has been
  classified, or even dispatched. Each request keeps its own arrival record and a dispatched
  flag, and each session counts its turns in flight, so a late completion settles the turn it
  belongs to.
- **Memory.** Every table is bounded: sessions, requests, the decision log, and the metric
  windows.

## Limits and upstream needs (PR #18443)

1. **One classifier per process.** `RequestClassifierConfig` has a single `type`, and
   `install_request_classifier` returns false once a classifier is set. This plugin and
   ThunderAgent cannot run together.
2. **No session index.** `RequestClassifierContext` carries only `block_size()` and
   `workers()`. `on_event` sees only lifecycle events, never KV events. `get_session_frontiers`
   covers the GPU tier only, and `with_max_sessions` is private.
   - What the plugin uses instead: the `WorkerCacheView` overlaps from `classify` (device-tier
     hits are not charged twice), plus its own context counts.
   - `SessionIndexView` and `NoSessionIndex` mark where an upstream read-only handle would plug
     in.
   - **Needed upstream:** a read-only handle to resident blocks per session and per tier.
3. **Lobby expiry returns 504.** A queued request is never classified again, so in enforce mode
   each waiting session of an `expire` class gets a deadline at classification:
   `due_at = ingress + max_lobby_wait_s`. When it passes, the router returns
   `DeadlineExceeded`, which maps to **504** today, not 429.
   - `due_at` also orders ahead of the policy score in the queue.
   - **Needed upstream:** remap lobby expiry to 429 with `Retry-After`.
4. **Throttle cannot return 429.** Throttling (`admission.max_lobby_sessions`) rejects in
   `classify`. `KvSchedulerError::RequestClassifierFailed` falls through to an unclassified error
   in `map_scheduler_error`, so the client gets no 429/529 and no `Retry-After`. The rejection
   carries its reason in the error message.
   - Counting the lobby is O(sessions), so throttling is off by default.
   - **Needed upstream:** let a classifier error carry a status and a retry hint.
5. **No engine priority.** `ClassificationOverrides` has only `policy_class`, `due_at`,
   `scheduling_cost_tokens`, and `worker_selection_target`.
   - The plugin does not set `policy_class` either, because the router's `policy_classes` may
     not use the same names.
   - A class-aware `WorkerFilter` would have to look up the class in state shared with the
     classifier, keyed by request (ThunderAgent registers both), because
     `WorkerSelectionContext` has no `policy_class`.
6. **No DRAM capacity.** The worker view has no host-tier capacity, so DRAM comes from
   `admission.dram_tokens_per_worker`. HBM is the sum of `total_kv_blocks × block_size` over the
   context's workers, cached for 100 ms.
7. **No ITL.** Lifecycle events do not report output tokens, so per-request ITL is unknown.
   Windows are judged on TTFT alone, as the simulator rule allows when no completion has an ITL.
8. **Gate cost.** `check` scans the session table: the lobby and the KV load are O(sessions).
   Incremental counters are a follow-up.

## The forecast

Enable it with `forecast: { enabled: true }` in the parameters (off by default). The classifier
then also declares the `LOAD` worker input, samples every worker's load at most once per
`snapshot_interval_s`, and sends `Arrived`, `Sent`, `FirstToken`, `Completed` and `Aborted`
observations with `try_send`; a full channel drops and counts them. The forecast drains them
under its own lock when queried (`SessionSchedulerClassifier::forecast()`), so neither
`classify` nor the gate ever waits on it.

The model is simple on purpose (module docs in `src/forecast.rs` give it in full): exponential
means of each session's think gap, new tokens and output tokens, falling back to its class's;
a worker's queued prefill drained in chunks, then one decode batch whose context grows a token
per step; `what_if` adds the candidate's prefill to the backlog and the candidate to the batch,
and counts its own TTFT and ITL misses plus running requests whose class ITL limit the larger
batch crosses. Arrivals and completions inside the horizon are not modelled, the load sample
is used as it stands (its age is reported), and output length is learned from
`context_tokens − prompt` at completion because events carry no output length.

Step times come from a `StepModel`. `TableStepModel` (interpolated grids; placeholder defaults,
not measurements) is the default and the test model. With the `aisimulate` feature,
`AisimulateStepModel` wraps AISimulate's `ForwardPassPerfModel`, built and queried exactly as
the Python bindings' `ais_callback.rs` does for the Mocker; set `forecast.aisimulate_config` to
a `ForwardPassPerfModelConfig` JSON file. The feature builds on macOS; whether a model is ready
depends on the AISimulate systems data installed, so its test runs only when
`AISIM_FORECAST_CONFIG` names a config (do that on Linux with the B300 data).
`aisimulate-core` is not a Dynamo crate, so the plugin rule (only `dynamo-kv-router`) holds, and
its tree reaches none of `axum hyper reqwest zmq`.
