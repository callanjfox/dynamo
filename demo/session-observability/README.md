# Session observability demo

A synthetic agent workload replayed through the Dynamo mocker, with the router's session
prefix index on. The session stats recorder writes a working-set time series and one summary
per session. `index.html` charts them. Open it in a browser; it needs no server.

Switch scenarios with the buttons, or `index.html?run=0|1|2`.

| Run | Load | Tiers |
|---|---|---|
| 0 | 15 tasks | GPU only |
| 1 | 40 tasks | GPU only |
| 2 | 40 tasks | GPU + host |

All runs use 4 workers with 6000 GPU blocks each (384k tokens). Run 2 adds 12000 host blocks per
worker.

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
