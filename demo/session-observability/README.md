# Session observability demo

A synthetic agent workload replayed through the Dynamo mocker, with the router's session
prefix index on. The session stats recorder writes a working-set time series and one summary
per session. `index.html` charts them. Open it in a browser; it needs no server.

Switch scenarios with the buttons, or `index.html?run=0|1|2`.

| Run | Load | Tiers |
|---|---|---|
| 0 | synthetic, 15 tasks, replay | GPU only |
| 1 | synthetic, 40 tasks, replay | GPU only |
| 2 | synthetic, 40 tasks, replay | GPU + host |
| 3 | AgentX, 30 sessions, 1 hour, live | GPU only |
| 4 | AgentX, 30 sessions, 1 hour, live | GPU + host |
| 5 | AgentX, 14 sessions, 20 min, live, 30% of sessions without headers | GPU + host |
| 6 | synthetic, 40 tasks, live | GPU only |

Replay runs: 4 workers with 6000 GPU blocks (384k tokens) each; run 2 adds 12000 host blocks per worker.
Live AgentX runs: 4 mock workers with 10000 blocks of 64 tokens (2.56M tokens of GPU cache in all); runs 4 and 5
add 20000 host blocks per worker (5.12M tokens).

| Live AgentX run | Sessions | Peak demand (sum) | Prefix reuse | Thrash |
|---|---|---|---|---|
| 3, GPU only | 105 | 5.8M (2.3x GPU) | 70% | 147M recomputed |
| 4, GPU + host | 105 | 5.8M (2.3x GPU) | 85% | 53M recomputed, 89M reloaded from host |
| 5, 30% anonymous | 39 | 2.7M | 94% | 1M recomputed, 15M reloaded |

Runs 3 and 4 each took about 5,000 requests and 880M input tokens. Run 5 reports 87% of requests and 90% of
input tokens as session-covered.

## Reproduce

```bash
cargo build --release -p dynamo-mocker --example session_observability_demo
B=target/release/examples/session_observability_demo
DYN_SESSION_STATS_JSONL=out/heavy_gpu.jsonl  $B --sessions 40 --host-blocks 0
DYN_SESSION_STATS_JSONL=out/heavy_host.jsonl $B --sessions 40 --host-blocks 12000
DYN_SESSION_STATS_JSONL=out/light.jsonl      $B --sessions 15 --host-blocks 0
# copy the six files into data/final/, then
python3 build_demo.py
```

The router breaks ties between idle workers at random, so numbers move a little between runs.
`data/final/` holds the runs behind the committed `index.html`.

## AgentX live runs

The workload is the AgentX corpus (`semianalysisai/cc-traces-weka-062126`): recorded Claude Code sessions with subagent
fan-out. `live/agentx_subset.py` cuts sessions out of the corpus. `live/agentx_drive.py` replays them as token-id
prompts built from the recorded `hash_ids`, so requests that shared KV blocks in the recording share them here. Think
times are capped at 300 s as aiperf does and divided by `--time-scale` (3 here). Sessions get the Dynamo session headers,
subagent groups become child sessions with `x-dynamo-parent-session-id`, and `--anonymous-fraction` leaves some sessions
without headers. The frontend is the production router with `--enable-session-prefix-index`, the workers are mockers.

```bash
python3 live/agentx_subset.py traces.jsonl subset.json --count 40
OUT=/root/live/ax_host SUBSET=/root/subset.json SESSIONS=30 BLOCKS=10000 HOST_BLOCKS=20000 TIME_SCALE=3 \
  RAMP_S=1200 DURATION_S=3600 WINDOW=100 TTL=240 live/run_agentx.sh        # HOST_BLOCKS=0 for GPU only
grep dynamo_session_summary $OUT/frontend.log > $OUT/summaries.raw
python3 live/extract_summaries.py summaries.raw > summaries.jsonl; python3 live/trim_metrics.py metrics.jsonl
# copy meta.json, metrics.jsonl, requests.jsonl, summaries.jsonl into data/live/<name>/, then build_demo.py
```

Caveats: mock workers do not publish cache reuse, so funnel step F4 is not reported. Requests above one worker's 640k-token
capacity are refused by the mocker (about 33 per run). Demand is the router's per-session sum, which counts a prefix once per
session and overstates distinct demand, more so with subagents. The corpus has no prefix shared across sessions, so all sharing
here is within a session and its subagents.

## Synthetic live run

The fourth scenario is the production path: a real frontend and KV router with
`--enable-session-prefix-index`, mock workers, and a client that sends the Dynamo session headers
(`x-dynamo-session-id`, `x-dynamo-parent-session-id`, `x-dynamo-session-final`). Nothing in it is
replayed. The page reads the router's real `/metrics` and its `session_summary` log.

```bash
# in a venv with the repo's python on PYTHONPATH and a built ai-dynamo-runtime wheel
OUT=/root/live/run1 WORKERS=4 BLOCKS=2500 SESSIONS=40 RAMP_S=240 GROWTH_WORDS=700 WINDOW=60 TTL=120 \
  live/run_live.sh
# then copy meta.json, metrics.jsonl, requests.jsonl and the extracted summaries.jsonl
# into data/live/heavy/ and run build_demo.py
```

`live/run_live.sh` starts the frontend and mocker with file discovery (no etcd or NATS), `live/scrape.py`
polls `/metrics`, and `live/drive.py` generates the sessions. `summaries.jsonl` is the frontend log's
`dynamo_session_summary` lines. The committed run: 4 workers with 2500 GPU blocks each (160k tokens),
137 sessions, 1172 requests, all 200.

Caveats: the client does not resend model output in later prompts, so each turn's output tokens count as
thrash (about 38k of the 1.73M recomputed here, so 2%). Demand is the router's per-session sum, which overstates distinct demand,
especially with subagents that start from their parent's prompt. There is no host tier in the live run.

## Output

`$DYN_SESSION_STATS_JSONL` gets one row per sample interval (`DYN_SESSION_STATS_SAMPLE_MS`,
default 5000 simulated ms). `$DYN_SESSION_STATS_JSONL.sessions` gets one summary per session.
`DYN_SESSION_STATS_ACTIVE_WINDOW_MS` (default 300000) sets how long a session counts as active.

## Caveats

- Replay traces carry no parent link, so a session id `parent/child` is read as a subagent of
  `parent`. A live router would use the real parent from the request.
- Demand is the distinct tokens in active sessions' latest prompts, computed in the replay from
  block hashes. The tracker's own `demand_tokens` is a per-session sum and double counts shared
  prefixes.
- Thrash is estimated from the prior turn's input plus output against what the index holds.
- The workload is synthetic, not a customer trace.
