# Request classifier and dispatch gate in dynosim offline replay (ledger G25)

Status: **built** on branch `replay-gate` (Dynamo, on PR #18443 head `9be15177`) and branch
`replay-gate-wakeup` of AISimulate (on `03eadda3`, the source of the pinned
`aisimulate-core 0.13.0-dev.202610290000000087`). Aggregated replay only. The plan as first
written follows the "As built" section and is kept for the reasoning; where they differ, "As
built" is what the code does.

## As built

| Piece | Where | What it does |
|---|---|---|
| Clock on the classifier context | `lib/router/admission/src/classifier/inputs.rs` | `RequestClassifierContext::with_clock` and `now()`; hosts keep tokio's clock, replay passes its virtual clock |
| Plugin clock | `src/clock.rs` `ContextClock`, used by `hooks::classifier_provider` | the scheduler reads the context's clock, so the same plugin runs live and in replay; `tests/no_system_clock.rs` still holds |
| Classifier supplied by the host | `lib/mocker/src/replay/mod.rs` `ReplayScope::request_classifier` (`ReplayClassifierPlugin`) | replay cannot link plugins; the host resolves the factory (Python bindings: `replay_gate_inputs`, feature `session-scheduler`) and installs it in the scope; `replay_selector_with_classifier` no longer bails when one is supplied |
| Classify on arrival | `kv_router/mod.rs` `classify_and_enqueue`, `worker_view` (`ReplayWorkerView`) | builds `ClassifyRequest` (uuid, session context, ISL, virtual ingress time, class), rows with `CACHE` (indexer overlap) and `LOAD` (slot projection) as declared; copies the class and `due_at` overrides |
| Gate in the drain | `kv_router/mod.rs` `drain_pending` | every queued head asks `check`; `Hold` blocks the lane with `BlockReason::Gate`; `Go{target}` becomes the soft affinity target; `dispatched` once per admitted head; every drain rechecks all held lanes |
| Always queue with a gate | `enqueue_or_admit` | as in the live router; requests of a class with no busy threshold are not queued and so never reach the gate (live behaviour too): **give each policy class a `prefill_busy_threshold_frac`** |
| Lifecycle events | `KvRouterPlacement::{dispatch_committed, prefill_completed, request_terminal, dispatch_aborted, cancel_pending}` | `Sent`, `Responding`, `Completed{context_tokens: isl + expected output}`, `Aborted`, delivered in replay order with `block_on` |
| Refusals | `classify_and_enqueue`, `expire_due`, `schedule_retry` | a classify error (429) or a passed `due_at` (504, `Aborted` delivered) sends the request back to its client, which retries after 2 s doubling to 30 s; counted in `ReplayGateStats` |
| Wake-up | AISimulate `PlacementPolicy::next_wakeup_ms`; Dynamo `OfflineReplayRouter::next_wakeup_ms` | the earliest of: the gate's `Hold{until}` still in the future, a client retry, a queued `due_at`; agg and disagg replay visit it and call `advance_clock` |
| Request fields | `kv_router/gate.rs` `ReplayRequestFields`, `ReplayScope::request_fields`; bindings env `DYNAMO_REPLAY_REQUEST_FIELDS` | per authored request id: session end, parent session, policy class, agent headers |
| Plugin reads them | `hooks.rs` `request_attributes`, `scheduler.rs` `on_classify_with` | tenant and client signals from agent headers, session end from `session_final`; a session ends when its final request completes |

### Request fields (one spec for the plugin, the KV action channel and the exporter)

| Field | Production carrier | Replay | Plugin use |
|---|---|---|---|
| Session end | `nvext.agent_context.session_final` (header `x-dynamo-session-final`), already in Dynamo (`SessionContext::session_final`) | `session_final: true` on a session's last request; a subagent is its own session (with `parent_session_id`), so its last request marks the subagent's end | ends the session when that request completes; the KV action channel's `kv.free_on_finish` rides the same request (`docs/kv-action-channel.md`) |
| Parent session | `nvext.agent_context.parent_session_id` | `parent_session_id` | subagent class and accounting |
| Policy class | `x-dynamo-meta-policy-class` | `policy_class` (exporter `classes.json`, seeded) | class |
| Tenant | agent header `x-dynamo-tenant` | `agent_headers`, seeded | session tenant (untrusted until L54) |
| Output length | `nvext.agent_hints.osl` (upstream `AgentHints`); also agent header `x-dynamo-hint-osl` | `agent_headers`, with the exporter's optional seeded noise | `Signals::osl_tokens` |
| Think time to the next turn | agent header `x-dynamo-hint-next-think-s` (not in upstream `AgentHints`) | as above | `Signals::next_think_s` (replaces the observed gap) |
| Next turn's new tokens | agent header `x-dynamo-hint-next-tokens` | as above | `Signals::next_turn_tokens` |

The file is JSON Lines, one `ReplayRequestFields` per line, keyed by `request_id` (the agentic
row's id, as `tools/weka-gaps` writes it). The exporter writes it from the same seed as
`classes.json`.

### Liveness (AISimulate `crates/core/src/replay/CLAUDE.md`)

- **No zero-duration spin.** `next_wakeup_ms` returns only deadlines; replay filters any that are
  not after the current time (`agg.rs`/`disagg.rs` `next_timestamps`), and `drain_pending` arms
  only `Hold{until}` values still in the future. A drain that releases nothing returns no
  placements, so it is not reported as progress.
- **Every unfinished request has a concrete future wakeup.** A held head waits on one of: its
  gate's `until` (a modeled deadline, visited through `next_wakeup_ms`), its `due_at` (likewise),
  a completion of another request (an engine event, after which replay drains and the plugin's
  `notify` sets the recheck flag), or its client's retry time (likewise visited). A gate that
  holds with `until: None` and never `notify`s, with nothing in flight, leaves a request with no
  wakeup; replay then fails with its existing "dead end" error instead of hanging, which is the
  contract's intent. The session scheduler always sets `until` (its wait limit) or notifies on
  completion and dispatch.
- **No timing tricks.** Events are delivered synchronously at their replay time; nothing is
  delayed to approximate async settlement. Eager forward-pass execution is untouched.

### Tests

- Mocker, `lib/mocker/tests/replay_gate.rs` (test classifier): a hold released by a completion's
  `notify`; a `Hold{until}` released at its deadline in an idle cluster (400 ms); a refusal
  retried after 2 s; a `due_at` expiry at 50 ms retried 2 s later; request fields reaching
  `classify`; identical admissions across two runs.
- Plugin, `tests/replay_gate.rs` (the real session scheduler, configured from a policy file):
  enforce holds session B until session A's final turn ends; shadow never holds; two enforce
  runs identical; `speed` (ignored) measures simulated against wall-clock time.
- Existing mocker replay tests (179) unchanged.

### Not done

- Disaggregated replay with a classifier (bails with a clear error).
- The router's worker set is a snapshot at construction; scaling changes are not seen by the
  classifier's context (static pools only).
- Per-request ITL is not reported to the plugin (as live).
- The exporter (`dynamo-admission-sim` `scripts/export_trace.py`) does not yet write the
  request fields file; the capability fork owns that.

## The plan as first written

## What replay does today

- `lib/mocker/src/replay/router_shared.rs::replay_selector_with_seed` bails with "offline replay
  does not support request_classifier plugins" whenever `request_classifier` is configured
  (test `replay_selector_rejects_request_classifier`).
- `OfflineReplayRouter` (`lib/mocker/src/replay/offline/extensions/kv_router/mod.rs`) already
  owns a `dynamo_admission::host::PolicyQueue` with session lanes. `drain_pending` calls
  `pending.pop_next(predicate)` with a capacity-only predicate and
  `recheck_blocked_lanes(RecheckScope::Capacity)`; its comment says "Replay has no dispatch
  gate". The queue already supports `LaneCheck::Blocked(BlockReason::Gate)` and
  `RecheckScope::All`, as the live router uses them (`lib/router/scheduler/src/queue.rs::gate_allows`).
- Replay time is `now_ms: f64`, mapped to an `Instant` by `OfflineReplayRouter::decay_now`
  (`decay_time_epoch + now_ms`). The epoch is `Instant::now()` at construction; nothing in
  replay advances tokio time.
- The replay engine (`aisimulate_core::replay`, `agg.rs::drain_current_timestamp`,
  `disagg.rs`) calls `PlacementPolicy::advance_clock(now_ms)` at every timestamp it visits, but
  a placement policy cannot ask to be woken at a future time.

## Changes, by crate

### 1. `dynamo-admission`: a clock on the classifier context (small, upstream)

`RequestClassifierContext` (`lib/router/admission/src/classifier/inputs.rs`) carries only
`block_size` and `workers`. Add `now: Arc<dyn Fn() -> Instant + Send + Sync>` with
`RequestClassifierContext::with_clock(self, now)` and `pub fn now(&self) -> Instant`, defaulting
to `tokio::time::Instant::now`. Hosts keep the default; replay passes `decay_now`.
Then in this crate, `hooks::classifier_provider` builds a `ContextClock` (a `Clock` over
`context.now()` with the same origin mapping as `HostClock`) instead of `HostClock::new()`.
`tests/no_system_clock.rs` keeps holding: only `clock.rs` reads time.

Alternative without this change: run replay inside a paused current-thread tokio runtime and
`tokio::time::advance` to each `now_ms` before calling the plugin. It works with `HostClock` as
is but costs a runtime turn per event and couples replay to tokio; not recommended.

### 2. Python bindings: hand replay the classifier factory

Replay cannot link plugins: `dynamo-mocker` is a host (`HOSTS` in
`scripts/router_layering_check.sh`) and plugins arrive through the catalog slot
(`lib/bindings/python/rust/lib.rs::register_core_with_router_plugins`, feature `custom-policy`).
In `lib/bindings/python/rust/llm/replay.rs`, resolve
`crate::router_plugins(&router_config)?.request_classifier().cloned()` and pass the
`Option<RequestClassifierFactory>` into the replay entrypoints
(`dynamo_mocker::replay::...` offline entrypoints → `KvRouterComposition` →
`KvRouterPlacement::new_with_selector_seed`).

### 3. `dynamo-mocker` offline replay: classify, gate, events

All in `lib/mocker/src/replay/offline/extensions/kv_router/mod.rs` unless noted.

| Function | Change |
|---|---|
| `router_shared.rs::replay_selector_with_seed` | Stop bailing when a classifier factory is supplied; still bail when one is configured but none was supplied (keeps today's error for the plain mocker CLI). Update `replay_selector_rejects_request_classifier`. |
| `OfflineReplayRouter::new_with_selector_seed` | Take `Option<RequestClassifierFactory>`. Build the classifier with `RequestClassifierContext::new(block_size, workers_fn).with_clock(decay_now_fn)`, where `workers_fn` returns `WorkerInfo` from `workers_with_configs`. Keep `classifier: Option<Box<dyn RequestClassifier>>`, `gate: Option<Arc<dyn DispatchGate>>` (from `dispatch_gate()`), `gate_recheck: Arc<AtomicBool>` set by the `DispatchRecheck` passed to `gate.attach`, and `worker_inputs` from `required_worker_inputs()`. |
| `on_compact_request_arrival_for_session` | Before `enqueue_or_admit`: build a `ClassifyRequest` (request id = uuid, `SessionContext` from `session_id`, `isl_tokens`, `ingress_at = decay_now(now_ms)`, policy class), call `classify(request, &ReplayWorkerView)`. `Err` = rejected: return a terminal rejection effect. `Ok`: copy the policy class and `due_at` overrides onto `PendingRequest` (`QueueMetadata::due_at`). |
| new `ReplayWorkerView` (`impl WorkerView`) | Rows from `workers_with_configs` × dp ranks: `info` from the config; `load` from `slots.project_worker_loads(None, decay_now)` when `LOAD` declared; `cache` from `indexer.find_matches_for_request` overlaps when `CACHE` declared. Built only when `worker_inputs != NONE`. |
| `enqueue_or_admit` | With a gate, always queue (as the live router does once a gate is installed) so the gate sees every head; without one, unchanged. |
| `drain_pending` | Predicate: capacity first, then `gate.check(&DispatchHead::new(uuid, session, isl, pinned, class, queued_at))`; `Hold { until }` → `LaneCheck::Blocked(BlockReason::Gate)` and record the earliest `until` as `gate_wake_ms`. Use `RecheckScope::All` when `gate_recheck` was set, else `Capacity`. After `pop_next`, call `gate.dispatched(&head, target)` and apply a `Go { target }` as the affinity target (as `queue.rs` does at its `gate.dispatched` call). |
| `KvRouterPlacement::dispatch_committed` | Deliver `ClassifyEvent::Sent { request_id, worker }`. |
| `KvRouterPlacement::prefill_completed` | Deliver `ClassifyEvent::Responding` (first token; replay has no separate first-token callback, and prefill completion is when it is emitted). |
| `KvRouterPlacement::request_terminal` | Deliver `ClassifyEvent::Completed { context_tokens: isl + output }`, then drain (a completion is when held sessions fit). |
| `KvRouterPlacement::dispatch_aborted`, `cancel_pending` | Deliver `ClassifyEvent::Aborted`. |
| `KvRouterPlacement::advance_clock` | Drain when `gate_recheck` is set or `now_ms >= gate_wake_ms` (today it returns early unless affinity or `needs_drain`). |

`on_event` is `async`; replay delivers events in order with `futures::executor::block_on`
(the session scheduler's `on_event` never awaits). Classification and events stay on the replay
thread, so replay remains deterministic.

### 4. `aisimulate-core` replay engine: a wake-up request (upstream, separate repo)

Add `fn next_wakeup_ms(&self) -> Option<f64> { None }` to `PlacementPolicy`
(`src/replay/replayer.rs` boundary and `src/replay/core/mod.rs`), and take it into account where
`agg.rs` and `disagg.rs` choose the next timestamp. `KvRouterPlacement` returns `gate_wake_ms`.
Without it a `Hold { until }` is honoured at the first replay event after `until`; with busy
workers that is within one engine pass, but a cluster that is idle apart from held sessions would
jump to the next arrival. The session scheduler's holds mostly end on completions (which are
events), so the first version can ship without this; wait-limit admits and lobby expiry would be
late by up to one event gap.

## Tests

1. Default behaviour identical: an existing replay fixture (e.g. the composition tests in
   `kv_router/composition.rs`) run with no classifier gives byte-identical reports before and
   after; the `dynamo-kv-replay-parity` skill's byte-parity campaign on one AgentX slice.
2. Gate honoured: a test classifier whose gate holds request B until request A completes; assert
   B's dispatch time equals A's completion time and that `check` ran before every dispatch and
   `dispatched` once per request.
3. Session scheduler shadow vs enforce on the same trace (needs the plugin linked, so this one
   lives in the bindings or in a `dynamo-session-scheduler` dev-dependency test that builds an
   `OfflineReplayRouter` through a test-only constructor): shadow has zero holds and the same
   dispatch times as no classifier; enforce has holds (`decisions()` non-empty with `Hold`) and
   no request exceeds the KV budget.

## Estimate

About 3 to 4 days: admission clock (0.5), bindings plumbing (0.5), mocker classify, view, gate
and events (1.5 to 2), tests and parity (1), plus the `aisimulate-core` change on its own release
cycle.
