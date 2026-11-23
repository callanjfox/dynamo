# Session scheduler on the mocker: quickstart

This branch runs the session scheduler plugin in a real Dynamo frontend, in front of mocker workers.
The mocker workers use AISimulate's engine core, the same step model, KV tiers and class scheduler the offline
simulator (dynosim) uses. So a policy can be tried end to end, over HTTP, without GPUs, and compared with dynosim.

What runs:

- 4 mocker workers. Each is a DeepSeek-V4-Pro TP4 worker on B300 (AISimulate op-level timing, vLLM 0.24.0), with
  136,050 HBM KV blocks of 64 tokens and 323 GB of host KV (G2, write-through, 128 GB/s, 5 ms to first byte).
- The Dynamo frontend with the KV router and a router policy file. The session scheduler is the policy file's
  `request_classifier` and `worker_selection`.
- A closed-loop users driver, `python -m dynamo.replay.users_http`. It plays a users export: each user runs its
  agent sessions turn by turn, waits for each answer, thinks, and goes on. It sends the session, class and hint
  headers a production client sends. It honours HTTP 429 and its `Retry-After`.

Three arms are set up in `examples/session_scheduler_mocker/`:

| Arm | Policy file | What it is |
|---|---|---|
| `kv` | `policies/kv.yaml` | The plain KV router. Classes exist only as labels. |
| `today` | `policies/today.yaml` | The KV router with per-class quanta and busy thresholds. No admission control. |
| `b0lt` | `policies/b0lt.yaml` | The session scheduler, first example policy B0+lt. Engine class scheduling on. |

## What this branch adds

Everything new is off unless you turn it on.

- Live output tokens. The router counts each request's output tokens from the response stream and hands them to
  the plugin on completion. The plugin needs them to judge ITL attainment.
- Hint and session headers reach the plugin. `x-dynamo-hint-*`, `x-dynamo-tenant` and `x-dynamo-user-id` are kept
  in the session context. `nvext.agent_hints.osl` and `.priority` fill `x-dynamo-hint-osl` and
  `x-dynamo-hint-priority` when the client did not send them.
- Retry-After on every refusal. A request the plugin holds in its lobby and then refuses at its wait limit now gets
  HTTP 429 with `Retry-After`, like a refusal at admission. Before, it got a deadline error with no advice.
- Class to engine. The router policy file's `engine_priority` map writes each class's priority into
  `routing.priority`. The mocker maps it back to the class (`--engine-priority`, the same map) and the engine
  schedules by class (`--class-scheduling`).
- Mocker flags: `--class-scheduling`, `--engine-priority`, `--host-offload-latency-ms` and
  `--host-offload-vllm-recency`. The first two also apply on top of `--extra-engine-args`.
- Speedup. `DYNAMO_SESSION_SCHEDULER_CLOCK_SCALE` makes the plugin's clock run as fast as the engines (the mocker's
  `speedup_ratio`). Waits, windows and Retry-After are then in simulated seconds, as in dynosim.
- The users driver, `python -m dynamo.replay.users_http`. It writes dynosim's per-request record format, so the
  same scorer reads both.

## Build

The Rust crates need AISimulate's engine core with this work's patches. Put an AISimulate checkout next to the
Dynamo checkout (the workspace `Cargo.toml` points at `../aisimulate`) and apply the patches:

```bash
git clone https://github.com/callanjfox/dynamo.git -b session-scheduler-mocker dynamo
git clone https://github.com/ai-dynamo/aisim.git aisimulate
cd aisimulate
git checkout -b session-scheduler 03eadda352afa385b573eff2b4fdb9c9fb371895
git am ../dynamo/aisimulate-patches/*.patch
cd ..
```

Build the Python bindings with the session scheduler into a virtual environment:

```bash
cd dynamo
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install pip 'maturin[patchelf]'
(cd lib/bindings/python && maturin develop --uv --release \
  --features kv-indexer,slot-tracker,select-service,mm-routing,ais-forward-pass,session-scheduler)
uv pip install -e .
python -c 'import dynamo._core, dynamo.llm; print("import ok")'
```

Linux with libclang is needed (`LIBCLANG_PATH`, for the NIXL bindings). A release build takes about 30 to 60
minutes on 32 cores and about 60 GB of target directory.

Checks:

```bash
cargo fmt -p dynamo-session-scheduler -p dynamo-mocker -p dynamo-scheduler -p dynamo-admission -p dynamo-llm --check
cargo clippy -p dynamo-session-scheduler -p dynamo-mocker -p dynamo-scheduler -p dynamo-admission --all-targets -- -D warnings
cargo test -p dynamo-session-scheduler -p dynamo-admission
cargo test -p dynamo-scheduler -p dynamo-mocker --lib
cargo test -p dynamo-llm --lib kv_router::
```

## Inputs

The driver plays a users export: `schedule.json`, its `templates.jsonl`, `request_fields.users.jsonl` and
`classes.json`. Make a synthetic one with the same shape, from a fixed seed:

```bash
python3 examples/session_scheduler_mocker/make_users.py runs/users-s31 --seed 31
```

It writes 300 agent plays and 832 users (30% priority, 50% standard, 20% batch) who join over the first 45
minutes and play one plan after another for 6 hours. Contexts grow from about 40k to up to 900k tokens, with
subagent fan-outs. At 4 workers this overloads the fleet in the first hour, as the admission simulator's m150
export does. Exports made by the admission simulator (`scripts/users_export.py`) from real traces work the same
way; they are not in this repository.

The served tokenizer is the TinyLlama test tokenizer in `lib/llm/tests/data/sample-models/TinyLlama_v1.1`. The
driver sends token ids (`nvext.token_data`), so the tokenizer only bounds the ids (below 32,000) and decodes the
output. Timing comes from the DeepSeek-V4-Pro step model whatever the tokenizer is.

## Run the 3-arm smoke test

One arm at a time, on one host with about 16 free cores:

```bash
export USERS_DIR=$PWD/runs/users-s31 SPEEDUP=1 STOP_S=2700 PYTHON=$PWD/.venv/bin/python
for arm in kv today b0lt; do
  examples/session_scheduler_mocker/run_arm.sh $arm runs/smoke-s31
done
```

`run_arm.sh` starts the 4 mocker workers, the frontend with the arm's policy file and the driver, waits for the
driver to finish (`STOP_S` simulated seconds), then stops what it started. Each arm writes into the output
directory:

| File | What |
|---|---|
| `<arm>.jsonl` | One record per request, in dynosim's per-request format. |
| `<arm>.refusals.jsonl` | Each request's first offer and each refusal with its Retry-After. |
| `<arm>.options.json`, `<arm>.summary.json` | Settings and counts. |
| `<arm>.sessions.jsonl`, `<arm>.f42.jsonl`, `<arm>.lobby.jsonl` | The plugin's session, attainment-loop and lobby logs. |
| `<arm>.frontend.log`, `<arm>.mocker<N>.log`, `<arm>.driver.log` | Process logs. |

Speedup: `SPEEDUP` runs the engines, the plugin clock and the driver that many times faster than real time. See
"Measured step-time fidelity" below before going above 1.

## Score

`score.py` reads the per-request records and the refusal log. It prints one row per arm and class:

```bash
python3 examples/session_scheduler_mocker/score.py --window-s 2700 --classes runs/users-s31/classes.json \
  kv=runs/smoke-s31/kv today=runs/smoke-s31/today b0lt=runs/smoke-s31/b0lt
```

Columns: requests offered and completed in the window, attainment over completed and over offered requests,
TTFT p50 and p90, median mean ITL, refusals, refusals with a Retry-After and its median, and output tokens. A
completed request is on target when its TTFT and mean ITL are within its class's limits (priority 3 s and 50 ms,
standard 8 s and 100 ms; batch has none). The records are in dynosim's per-request format, so the admission
simulator's scorer (`scripts/validate/bench_score.py`) reads them too.

## Results of the smoke test

Run on 2026-11-23: 4 mocker workers, speedup 1, 45 simulated minutes (`STOP_S=2700`), each arm on its own host.
The reference is dynosim (offline replay, the same engine core and plugin) on the same export, window and policy
files, without KVCR, since the live engines have none. Attainment is over completed requests (`att_done`).

Synthetic export (`make_users.py --seed 31`):

| Arm | Class | Live att | dynosim att | Live TTFT p50 / p90 (s) | dynosim TTFT p50 / p90 (s) | Live ITL p50 (ms) | dynosim ITL p50 (ms) | Live refusals (with Retry-After) | dynosim refusals |
|---|---|---|---|---|---|---|---|---|---|
| kv | priority | 0.06 | 0.07 | 2.9 / 63 | 2.5 / 65 | 149 | 139 | 0 | 0 |
| kv | standard | 0.24 | 0.24 | 3.1 / 66 | 3.0 / 64 | 149 | 142 | 0 | 0 |
| today | priority | 0.07 | 0.06 | 1.9 / 9.0 | 1.9 / 8.5 | 167 | 165 | 0 | 0 |
| today | standard | 0.25 | 0.26 | 2.2 / 67 | 2.3 / 65 | 158 | 156 | 0 | 0 |
| b0lt | priority | 0.45 | 0.46 | 0.22 / 1.2 | 0.18 / 1.2 | 51 | 51 | 160 (160) | 151 |
| b0lt | standard | 0.98 | 0.97 | 0.21 / 1.2 | 0.18 / 1.2 | 49 | 49 | 2,549 (2,549) | 2,755 |
| b0lt | batch | 1.00 | 1.00 | 0.20 / 1.2 | 0.17 / 1.2 | 44 | 45 | 636 (636) | 568 |

The live runs track dynosim closely on every arm. The live TTFT is about 30 ms higher, which is HTTP, frontend
and stream overhead. With KVCR on, dynosim's kv and today arms do better (priority 0.12 and 0.11, standard 0.48
and 0.52), so KVCR matters for those arms.

Checked on the live b0lt arm:

- Classes reach the engine. With trace logs on, every request reached the mocker with the class the driver sent:
  151 priority, 270 standard and 143 batch out of 564.
- Every 429 carried a `Retry-After` and the driver waited for it before it sent again (3,345 refusals, all with
  advice: priority p50 105 s, standard 120 s, batch 30 s).
- Session ids, parent sessions, `session_final`, the user id and the think-time hint reach the plugin. Its
  session log names the user (for example `"user":"u000186"`), and its trace shows the parent, end and
  `next_think_s` of each request.
- Live output tokens reach the plugin: 479 of 479 completions had `output_tokens=Some(..)`.
- The records score with `score.py` and with the admission simulator's scorer.

The admission simulator's m150 export (832 users, the same 4 workers): the kv and today arms match dynosim, with
completed requests, TTFT (minutes, at the start-up surge) and ITL (about 500 ms) all within about 10%. The b0lt
arm does not match yet. Live priority attainment is 0.16, against 0.42 in dynosim without KVCR. Live admits much
more standard work (961 standard requests completed against 182), and its fleet prefill runs at about twice the
prefill budget in the first 10 minutes. That export's contexts are larger (up to 950k tokens) and arrive as one
surge. The cause is not found yet: see Known gaps.

Measured step-time fidelity: one worker, 64 concurrent requests of 16k prompt tokens and 256 output tokens.

| Speedup | TTFT p50 (s) | TTFT p90 (s) | ITL p50 (ms) | ITL p90 (ms) |
|---|---|---|---|---|
| 1 | 26.9 | 47.9 | 123.1 | 199.3 |
| 2 | 27.5 | 48.5 | 123.1 | 199.3 |
| 4 | 28.8 | 49.7 | 123.1 | 199.3 |
| 8 | 31.8 | 52.8 | 123.1 | 199.3 |

Engine steps stay exact at every speedup (ITL is the same). TTFT gains about 0.6 s of wall time per burst
(request upload and frontend processing), which a speedup multiplies into simulated time: +0.6 s at 2x, +1.9 s
at 4x and +4.9 s at 8x. Against a 3 s priority TTFT target, use speedup 1 for attainment numbers. Speedup 2 is
usable for shape, with a small TTFT bias.

## Known gaps

- No router-to-engine switch channel. Batch yield and other adaptive engine switches (protect rank, preemption
  on/off) are set per dispatch inside offline replay only. Live engines run a fixed class scheduling config.
  Policies with `batch_yield` do not run live yet.
- No KVCR. Peer fetch between workers' host tiers is not wired live, so the engine config has no `kvcr` section.
  dynosim's reference cell has KVCR at 128 GB/s.
- `kv_actions` (retain value, release source and the other KV actions) are offline replay only. The live router
  reads and ignores them.
- The users driver measures TTFT at the client. It includes HTTP, frontend and stream overhead that dynosim does
  not have.
- The driver maps the export's token ids into the tokenizer's vocabulary by a hash of each id and its position in
  its block. Equal blocks stay equal; distinct blocks collide only by chance.
- DRAM capacity is not reported by workers. `dram_tokens_per_worker` in the policy file must match the mocker's
  host KV (`num_host_blocks x 64` tokens).
- dynosim's `DYNAMO_REPLAY_GATE_RECHECK=keyed` gate recheck mode has no live switch; the live router queue
  rechecks gates its own way.
- B0+lt on the large-context m150 export is not faithful yet: live priority attainment 0.16 against 0.42 in
  dynosim. Leads: fleet prefill runs at twice the plugin's prefill budget early in the live run, and live admits
  more standard sessions. Suspects are the replay-only worker view of host tier hits
  (`DYNAMO_REPLAY_WORKER_VIEW_TIERS`), engine gauges that offline replay hands the plugin, and the gate recheck
  mode above. The synthetic export, with smaller contexts, matches.
- The exporter writes `x-dynamo-hint-osl` and `x-dynamo-hint-next-tokens` as decimals ("450.0"). The plugin
  parses them as integers, so it drops them, in dynosim and live alike. The body hint (`nvext.agent_hints.osl`)
  does not fill them, because a non-blank header wins.
- dynosim's per-request records leave out requests still queued at the router when the run stops; the live
  records list them as `unfinished`. Compare completed requests, not offered ones.
- Speedup above 1 biases TTFT upward (see the fidelity table).
