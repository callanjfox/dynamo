#!/bin/bash
# The four example scenarios of the diagnosis dashboard, on mock workers (DeepSeek V4 Flash, B200 timing from AISimulate).
#
#   scenarios.sh <ample|short|mirror|route> <port> [name]
#
# Each run ramps agent sessions in over 400 s, runs 1500 s, and sends single-turn chat alongside. The router_selection
# trace is kept (rs.jsonl) so decisions can be joined to requests. Needs run_agentx.sh and the AgentX subset next to it
# (SUBSET, default /root/subset.json).
#
#   ample   2 workers x 220000 blocks (twice the room the working set needs), 60 sessions x2, chat 2 rps
#   short   4 workers x 14000 blocks, 40 sessions, chat 1 rps (demand about 2x GPU; BLOCKS_OVERRIDE=28000 gives the 1.2x case)
#   mirror  as short plus 14000 host blocks per worker
#   route   4 workers x 40000 blocks, cache-blind routing (overlap credit 0.001; exactly 0 is rejected)
set -e
here=$(cd "$(dirname "$0")" && pwd)
kind=$1; port=$2; name=${3:-$1}
case $kind in
  ample)  W=2; B=220000; H=0;     N=60; R=2; C=2; FE="" ;;
  short)  W=4; B=14000;  H=0;     N=40; R=1; C=1; FE="" ;;
  mirror) W=4; B=14000;  H=14000; N=40; R=1; C=1; FE="" ;;
  route)  W=4; B=40000;  H=0;     N=40; R=1; C=1; FE="--router-kv-overlap-score-credit 0.001" ;;
  *) echo "unknown scenario $kind" >&2; exit 1 ;;
esac
# BLOCKS_OVERRIDE and HOST_BLOCKS_OVERRIDE resize the cache of any preset, for example to double it.
B=${BLOCKS_OVERRIDE:-$B}; H=${HOST_BLOCKS_OVERRIDE:-$H}
out=${OUT_ROOT:-/root/live_out}
mkdir -p "$out/trace_$name"
env DYN_REQUEST_TRACE_RECORDS=router_selection DYN_REQUEST_TRACE_SINKS=file DYN_REQUEST_TRACE_FILE_FORMAT=jsonl \
  DYN_REQUEST_TRACE_FILE_PATH="$out/trace_$name/rs.jsonl" OUT="$out/fz_$name" SUBSET=${SUBSET:-/root/subset.json} \
  SESSIONS=$N REPEAT=$R WORKERS=$W MODEL=deepseek-ai/DeepSeek-V4-Flash BLOCKS=$B HOST_BLOCKS=$H SPEEDUP=1 TIME_SCALE=3 \
  RAMP_S=400 DURATION_S=1500 WINDOW=100 TTL=240 CHAT_RPS=$C CHAT_PROMPT=2000 SHARED_PREFIX=8192 PORT=$port FRONTEND_EXTRA="$FE" \
  MOCKER_EXTRA="--engine-type vllm --ais-perf-model --ais-system b200_sxm --ais-backend-version 0.24.0 --ais-tp-size 8 --ais-moe-tp-size 1 --ais-moe-ep-size 8" \
  setsid nohup "$here/run_agentx.sh" > "$out/$name.log" 2>&1 < /dev/null &
