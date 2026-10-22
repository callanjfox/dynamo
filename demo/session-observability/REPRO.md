# Reproducing the session observability findings

This reproduces the diagnosis dashboard's scenarios and the main finding: when the KV cache is oversubscribed, the router's view of the cache on the chosen worker (F3) is far above what the worker actually reuses when it admits the request (F4), and the gap grows with time to first token. It runs on CPU with mock workers. A real-GPU run uses the same driver, the same router flags and the engines' own reports.

## What you need

- A Dynamo build with the session observability change (this branch) and the mocker change that sets `engine_data.kv_cache_hit`, the same report vLLM, SGLang and TensorRT-LLM workers send (ai-dynamo/dynamo#17127; until it merges, cherry-pick it). Without it F4 is empty on mock workers.
- A Linux x86_64 machine or pod with 12 CPUs and 32 GB of memory or more. Each scenario takes about 3.5 CPUs and 6 GB. Four at once fit in 12 CPUs.
- The AgentX corpus, `semianalysisai/cc-traces-weka-062126`, as `traces.jsonl`.

## 1. Build

```bash
# Debian/Ubuntu: python3-venv python3-pip python3-dev protobuf-compiler cmake clang libclang-dev pkg-config libssl-dev git
git clone --branch session-observability-telemetry https://github.com/callanjfox/dynamo.git
python3 -m venv venv
venv/bin/pip install maturin aiohttp prometheus_client packaging pyyaml pydantic uvloop numpy msgspec zstandard \
    aisimulate==0.13.0.dev202610160000000073
cd dynamo/lib/bindings/python
../../../venv/bin/maturin build --release --features ais-forward-pass --out /root/wheels
../../../venv/bin/pip install --force-reinstall --no-deps /root/wheels/ai_dynamo_runtime-*.whl
```

The `ais-forward-pass` feature gives the mocker AISimulate's timing model (DeepSeek V4 Flash on B200 here). Without it, the mocker's built-in timing is used and latency numbers will differ.

## 2. Cut the workload

```bash
python3 demo/session-observability/live/agentx_subset.py traces.jsonl /root/subset.json --count 40 --min-turns 60 --max-turns 220
```

## 3. Run a scenario

```bash
export PATH=/root/venv/bin:$PATH
demo/session-observability/live/scenarios.sh <ample|short|mirror|route> <port> <name>
```

Presets (GPU cache is blocks of 64 tokens per worker):

| Preset | Workers | GPU blocks | Host blocks | Routing | Demand vs GPU |
|---|---|---|---|---|---|
| ample | 2 | 220000 | 0 | default | about 0.4x |
| short | 4 | 14000 | 0 | default | about 2.1x (the "far too little cache" case) |
| mirror | 4 | 14000 | 14000 | default | about 2.1x GPU, 1.1x GPU plus host |
| route | 4 | 40000 | 0 | cache-blind (overlap credit 0.001) | about 0.9x |

The dashboard's five tabs use: ample as is; "not enough cache" = short with `BLOCKS_OVERRIDE=28000` (1.2x); "cache not used well" = mirror with `BLOCKS_OVERRIDE=28000 HOST_BLOCKS_OVERRIDE=28000`; "routing ignores the cache" = route with `BLOCKS_OVERRIDE=80000`; "far too little cache" = short as is.

`BLOCKS_OVERRIDE` and `HOST_BLOCKS_OVERRIDE` resize any preset, for example `BLOCKS_OVERRIDE=28000 scenarios.sh short 9011 d_short` doubles short. Use a different port per concurrent run. Each run takes about 29 minutes: a 400 second ramp, 1500 seconds of load, then a 4 minute idle wait so sessions expire and the sweep publishes.

What the run does: `agentx_drive.py` replays recorded agent sessions as token-id prompts with `x-dynamo-session-id` headers (think times capped at 300 s and divided by 3), `chat_load.py` adds anonymous single-turn chat (1 to 2 requests a second), the frontend runs `--router-mode kv` with `--enable-session-prefix-index`, and `scrape.py` records the router and frontend metrics. The `router_selection` request trace is written to `rs.jsonl` under `$OUT_ROOT/trace_<name>/`.

Cache-blind routing sets `--router-kv-overlap-score-credit 0.001`; exactly 0 is rejected because the session index needs the selection policy to declare a cache input.

## 4. Post-process and build the dashboard data

```bash
cd demo/session-observability/live
for d in fz_<name>; do
  python3 extract_summaries.py $d/frontend.log > $d/summaries.jsonl     # session_summary logs
  python3 extract_workers.py   $d/frontend.log > $d/workers.jsonl       # session_worker_load logs
  python3 extract_selection.py trace_<name>/rs.jsonl > $d/selection.jsonl
  python3 trim_metrics.py $d/metrics.jsonl
done
cd .. && python3 build_diag.py --scenarios busy=<ample run> short=<short, 28000 blocks> mirror=<mirror, 28000+28000> route=<route, 80000> starved=<short run>
```

`build_diag.py` writes `diag/data/*.json`, which the dashboard page reads. The scenario keys (`busy`, `short`, `mirror`, `route`) are fixed names; the directory after the `=` is the run.

## 5. Check the result

The finding is in the funnel. For each run take the median over minutes 8 to 24 of `f2_pct` (best worker holds), `f3_pct` (chosen worker holds, as the router expected) and `f4_pct` (worker reused), from `diag/data/timeline.json`. Expected with the 2026-10-22 runs on the fixed router:

| Scenario | Demand vs GPU | F2 | F3 | F4 | Median TTFT | Rebuilt (share of agent input) |
|---|---|---|---|---|---|---|
| ample (2 x 220000) | 0.4x | 97 | 94.5 | 94.5 | 0.2 s | 2% |
| not enough cache (4 x 28000) | 1.2x | 86 | 69 | 68 | 1.2 s | 27% |
| cache not used well (28000 + 28000 host) | 1.2x | 86 | 67 | 66 | 1.5 s | 29% |
| routing ignores cache (4 x 80000) | 0.4x | 95 | 73 | 74 | 0.5 s | 23% |
| far too little cache (4 x 14000) | 2.2x | 71 | 57 | 9 | 16 s | 37% |

These are the final-wheel runs (`fin_*`); an earlier set of runs on the previous wheel gave the same picture within a couple of points, except cache-blind routing, which varies a little run to run: over five runs F3 ranged 73.2 to 77.7, F4 73.6 to 77.1 and rebuilt 18.9 to 23.2.

F3 and F4 agree in every scenario except the starved one. There the router counted cache on the chosen worker that was evicted before the request was admitted, and the worker reused about a tenth. In the 14000-block run F3 minus F4 is 0 to 6 points where TTFT is under 5 s and 37 to 48 points where it is over 5 s. Doubling the cache (1.2x) removes the gap and cuts TTFT from 16 s to 1.2 s. The router still leaves the best-cache worker about 17 points behind in that case, which is a real routing cost, not staleness. Three separate runs of the starved case gave F4 of 8.5% to 9.9%.

Other cross-checks that should hold: input tokens in `meta.json` equal the driver's total over `requests.jsonl` plus `chat.jsonl`; returning turns equal turns minus sessions; the return-gap buckets match gaps computed from `requests.jsonl`; and "prefix rebuilt" (`thrash_recomputed_pm` over agent input per minute) is within about 3 points of `pot_pct - f3_pct`.

To see per request which decisions left the best-cache worker, and by how much, run `live/extract_gap.py rs.jsonl requests.jsonl`. It joins the trace to the driver log by prompt length and start time, because the frontend assigns its own trace request IDs.

## 6. On real engines

Use the same driver and router flags with `dynamo.vllm`, `dynamo.sglang` or `dynamo.trtllm` workers and kv events enabled. Real workers send `engine_data.kv_cache_hit` themselves, so F4 fills with no change. Read each engine's own queue-time metric next to F3 and F4 to test whether the shortfall grows with the wait between the router's decision and admission (vLLM and TensorRT-LLM export `request_queue_time_seconds`, SGLang `queue_time_seconds`). That wait is the open question the mock runs point at; the mocker does not export it.

## 7. Fault injection

`live/fault_run.sh` runs two mock workers, kills one mid-run and (with `DYN_MIGRATION_LIMIT`, default 3 in the script) lets the frontend retry. `live/fault_check.py <out>` compares the router's session summaries with the driver log: turns should equal requests sent, with no session reporting twice. Use `KILL_AT=0` for a control run. Expected on the final wheel: turns equal requests sent (696 for 696 with 4 retries, 189 for 189 with 6 retries) and the covered-request counter equals requests sent; the old accounting counted every retry as another turn and another covered request. `MAX_TURNS=5 KILL_AT=46` cuts sessions short so several final-turn requests are in flight at the kill, and `VENV=<venv>` selects which wheel runs, for before and after comparisons (the script puts that environment first on `PATH`). A copied venv's `activate` and `bin/pip` still point at the original, so install into it with `<venv>/bin/python -m pip` and confirm with a checksum of `dynamo/_core.abi3.so`; both mistakes were made once and gave runs labelled "old" that used the new build. `OVERLAP_EVERY=3` sends every third turn without waiting so requests of one session overlap, and `ANON=0.3` sends a share of sessions without session headers. `fault_check.py` prints the coverage totals and the expected prior prefix.

## 8. Long run

`live/long_run.sh` runs 6 hours at real think times on 4 workers x 75000 blocks, with `live/monitor_resources.sh` sampling process memory and CPU each minute. It post-processes itself and writes `$OUT_ROOT/$NAME.status` as `POSTPROCESSED`. `NAME`, `SESSIONS` and `VENV` select the run, the number of sessions arriving evenly over the six hours, and the environment.

Pick `SESSIONS` from Little's law: sessions running at once is the arrival rate times how long a session lasts, and a session lasts about an hour at real think times when the fleet keeps up. `SESSIONS=260` gives about 40 running at once and demand near 0.4x the GPU cache. `SESSIONS=1440`, the first attempt, gave about 1,100 at once and demand of 7.8x the cache: requests queued for minutes, sessions made progress slowly and so lasted longer, and the load fed itself. That run is kept on the dashboard as "Six hours, overloaded".

Check that input tokens and covered requests equal the driver log, that process memory stays flat once the number of running sessions does, that F3, F4 and TTFT do not drift, and that the 5 and 15 minute demand windows separate. In the overloaded run the totals matched exactly (51,692 requests, 4,542,001,984 input tokens, no session summarized twice) and the frontend grew from 0.1 GB to 4.5 GB while running sessions grew from 120 to 1,180, so that run cannot say whether memory is a leak or scales with sessions.

## Known limits of this setup

- The host tier in the mocker is write-through, so 97 to 99% of host tokens are also on the GPU. A real offload tier may not be.
- Mock workers share one machine's CPU; keep the load average under the core count or latency numbers are distorted.
- The F3 minus F4 relationship with TTFT is a correlation across minutes of mock runs, not a controlled test.
- Overlapping requests of one session do not occur in the scenario workload (primaries are sequential, subagents have their own IDs); `fault_run.sh` with `OVERLAP_EVERY=3` covers that case.
- The 5 and 15 minute demand windows coincide here: the replay uses a 100 second active window and short think times. Use a 900 second `DYN_SESSION_STATS_ACTIVE_WINDOW_SECS` and real think times to separate them.
