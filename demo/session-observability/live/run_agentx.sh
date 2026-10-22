#!/bin/bash
# Replay AgentX sessions through a live frontend and KV router with mock workers.
# Needs the repo's python on PYTHONPATH and a built ai-dynamo-runtime in /root/venv.
#
#   OUT=/root/live/ax_gpu SUBSET=/root/subset.json HOST_BLOCKS=0 ./run_agentx.sh
#   OUT=/root/live/ax_host SUBSET=/root/subset.json HOST_BLOCKS=12000 ./run_agentx.sh
#
# Writes only under $OUT, which must not exist yet.
set -u
OUT=${OUT:?set OUT to a new directory}
SUBSET=${SUBSET:?set SUBSET to the agentx_subset.py output}
WORKERS=${WORKERS:-4}
BLOCKS=${BLOCKS:-5000}
HOST_BLOCKS=${HOST_BLOCKS:-0}
BLOCK=${BLOCK:-64}
SPEEDUP=${SPEEDUP:-4}
SESSIONS=${SESSIONS:-40}
DURATION_S=${DURATION_S:-3600}
RAMP_S=${RAMP_S:-900}
TIME_SCALE=${TIME_SCALE:-6}
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
PORT=${PORT:-8000}
WPORT=${WPORT:-8081}   # the mocker's system status server: engine-style worker metrics
ANON=${ANON:-0}
MAXSEQS=${MAXSEQS:-64}
FRONTEND_EXTRA=${FRONTEND_EXTRA:-}      # extra frontend flags, for example a router weight
MOCKER_EXTRA=${MOCKER_EXTRA:-}          # extra mocker flags, for example the AISimulate perf model
SHARED_PREFIX=${SHARED_PREFIX:-0}       # tokens every session shares at the start of its prompt
REPEAT=${REPEAT:-1}                     # replay the subset this many times as distinct sessions
CHAT_RPS=${CHAT_RPS:-0}                 # background chat requests per second, no session headers
CHAT_PROMPT=${CHAT_PROMPT:-1500}        # median chat prompt, tokens
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir "$OUT" || { echo "$OUT exists"; exit 1; }
export DYN_DISCOVERY_BACKEND=file DYN_FILE_KV="$OUT/kv" DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq
export DYN_SESSION_STATS_SWEEP_SECS=5
export DYN_SESSION_STATS_ACTIVE_WINDOW_SECS=${WINDOW:-60}
export DYN_SESSION_STATS_IDLE_TTL_SECS=${TTL:-180}
export DYN_LOGGING_JSONL=1 DYN_LOG=${DYN_LOG:-info}
source /root/venv/bin/activate
export PYTHONPATH=/root/dynamo/components/src

cat > "$OUT/meta.json" <<JSON
{"workers": $WORKERS, "num_gpu_blocks": $BLOCKS, "num_host_blocks": $HOST_BLOCKS, "block_size": $BLOCK,
 "speedup": $SPEEDUP, "sessions": $SESSIONS, "duration_s": $DURATION_S, "time_scale": $TIME_SCALE,
 "active_window_s": ${WINDOW:-60}, "idle_ttl_s": ${TTL:-180}, "workload": "agentx", "anonymous_fraction": $ANON, "model": "$MODEL"}
JSON

HOST_ARGS=""
[ "$HOST_BLOCKS" -gt 0 ] && HOST_ARGS="--num-host-blocks $HOST_BLOCKS"

python -m dynamo.frontend --http-port $PORT --discovery-backend file --router-mode kv \
  --router-kv-events --kv-cache-block-size "$BLOCK" --enable-session-prefix-index $FRONTEND_EXTRA \
  > "$OUT/frontend.log" 2>&1 &
FRONTEND=$!
python -m dynamo.mocker --model-path "$MODEL" --discovery-backend file \
  --num-workers "$WORKERS" --num-gpu-blocks-override "$BLOCKS" --block-size "$BLOCK" \
  --max-num-seqs "$MAXSEQS" --max-model-len 1100000 --speedup-ratio "$SPEEDUP" --enable-prefix-caching $HOST_ARGS $MOCKER_EXTRA \
  > "$OUT/mocker.log" 2>&1 &
MOCKER=$!

for _ in $(seq 1 120); do
  if curl -s localhost:$PORT/v1/models | grep -q "$MODEL"; then READY=1; break; fi
  sleep 2
done
[ "${READY:-0}" = 1 ] || { echo "frontend never saw the model"; tail -20 "$OUT/frontend.log" "$OUT/mocker.log"; kill $FRONTEND $MOCKER; exit 2; }
echo "ready"

python "$HERE/scrape.py" --url http://localhost:$PORT/metrics --out "$OUT/metrics.jsonl" \
  --stop-file "$OUT/stop" --interval 5 &
SCRAPER=$!
if [ "$CHAT_RPS" != "0" ]; then
  python "$HERE/chat_load.py" --url http://localhost:$PORT --model "$MODEL" --rps "$CHAT_RPS" \
    --prompt-median "$CHAT_PROMPT" --duration-s "$DURATION_S" --out "$OUT/chat.jsonl" &
  CHAT=$!
fi
python "$HERE/agentx_drive.py" --subset "$SUBSET" --url http://localhost:$PORT --model "$MODEL" \
  --sessions "$SESSIONS" --repeat "$REPEAT" --shared-prefix-tokens "$SHARED_PREFIX" --duration-s "$DURATION_S" --ramp-s "$RAMP_S" --time-scale "$TIME_SCALE" --anonymous-fraction "$ANON" \
  --out "$OUT/requests.jsonl"
[ -n "${CHAT:-}" ] && wait $CHAT
# Let idle sessions expire and the sweep publish before stopping.
sleep $(( ${TTL:-180} + 15 ))
touch "$OUT/stop"
sleep 6
kill $SCRAPER $FRONTEND $MOCKER 2>/dev/null
echo "done: $OUT"
