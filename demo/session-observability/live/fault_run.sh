#!/bin/bash
# Kill one of two mock workers while sessions are running, so requests fail at dispatch or mid-stream and are
# retried. Checks that the session metrics do not count the failed attempts as extra turns.
#
#   OUT=/root/live_out/f_before PORT=9050 ./fault_run.sh
#
# Afterwards: python3 fault_check.py $OUT
OUT=${OUT:?}; PORT=${PORT:-9050}; KILL_AT=${KILL_AT:-70}   # 0 = control run, nothing killed
SUBSET=${SUBSET:-/root/subset.json}; MODEL=${MODEL:-deepseek-ai/DeepSeek-V4-Flash}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir "$OUT" || { echo "$OUT exists"; exit 1; }
export DYN_DISCOVERY_BACKEND=file DYN_FILE_KV="$OUT/kv" DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq
export DYN_SESSION_STATS_SWEEP_SECS=5 DYN_SESSION_STATS_ACTIVE_WINDOW_SECS=60 DYN_SESSION_STATS_IDLE_TTL_SECS=${TTL:-150}
export DYN_LOGGING_JSONL=1 DYN_LOG=info
# Migration is off by default; turn it on so failed attempts are retried on the surviving worker.
export DYN_MIGRATION_LIMIT=${MIGRATION_LIMIT:-3}
# Put the chosen environment first. Sourcing a copied venv's activate would still point at the original.
export VIRTUAL_ENV=${VENV:-/root/venv}
export PATH=$VIRTUAL_ENV/bin:$PATH
export PYTHONPATH=/root/dynamo/components/src
python -m dynamo.frontend --http-port $PORT --discovery-backend file --router-mode kv --router-kv-events \
  --kv-cache-block-size 64 --enable-session-prefix-index > "$OUT/frontend.log" 2>&1 &
FRONTEND=$!
MOCK="--model-path $MODEL --discovery-backend file --num-workers 1 --num-gpu-blocks-override 60000 --block-size 64 --max-num-seqs 64 --max-model-len 1100000 --speedup-ratio 1 --enable-prefix-caching --engine-type vllm --ais-perf-model --ais-system b200_sxm --ais-backend-version 0.24.0 --ais-tp-size 8 --ais-moe-tp-size 1 --ais-moe-ep-size 8"
DYN_SYSTEM_PORT=$((PORT + 100)) python -m dynamo.mocker $MOCK > "$OUT/mocker1.log" 2>&1 &
M1=$!
DYN_SYSTEM_PORT=$((PORT + 200)) python -m dynamo.mocker $MOCK > "$OUT/mocker2.log" 2>&1 &
M2=$!
for i in $(seq 1 90); do curl -sf localhost:$PORT/v1/models | grep -q "$MODEL" && break; sleep 2; done
echo ready
python "$HERE/agentx_drive.py" --subset "$SUBSET" --url http://localhost:$PORT --model "$MODEL" \
  --sessions 12 --repeat 1 --duration-s 240 --ramp-s 30 --time-scale 3 --max-turns ${MAX_TURNS:-0} --overlap-every ${OVERLAP_EVERY:-0} --anonymous-fraction ${ANON:-0} --out "$OUT/requests.jsonl" &
DRIVER=$!
if [ "$KILL_AT" != 0 ]; then
  sleep $KILL_AT
  echo "killing mocker 2 at $(date +%s)" > "$OUT/kill.txt"
  kill -9 $M2
fi
wait $DRIVER
sleep $(( ${TTL:-150} + 15 ))
curl -s localhost:$PORT/metrics > "$OUT/metrics_end.txt"
kill $FRONTEND $M1 2>/dev/null
echo "done: $OUT"
