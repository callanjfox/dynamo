#!/bin/bash
# Run a live session workload: real frontend and KV router with the session index on,
# mock workers, a session-aware client, and a scraper on the frontend's /metrics.
# Needs the repo's python on PYTHONPATH and a built ai-dynamo-runtime in the venv.
#
#   OUT=/root/live/run1 WORKERS=4 BLOCKS=1500 SESSIONS=24 ./run_live.sh
#
# Writes only under $OUT, which must not exist yet.
set -u
OUT=${OUT:?set OUT to a new directory}
WORKERS=${WORKERS:-4}
BLOCKS=${BLOCKS:-1500}
SPEEDUP=${SPEEDUP:-4}
SESSIONS=${SESSIONS:-24}
RAMP_S=${RAMP_S:-120}
GAP_SCALE=${GAP_SCALE:-1.0}
GROWTH_WORDS=${GROWTH_WORDS:-250}
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir "$OUT" || { echo "$OUT exists"; exit 1; }
export DYN_DISCOVERY_BACKEND=file DYN_FILE_KV="$OUT/kv" DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq
export DYN_SESSION_STATS_SWEEP_SECS=2
export DYN_SESSION_STATS_ACTIVE_WINDOW_SECS=${WINDOW:-30}
export DYN_SESSION_STATS_IDLE_TTL_SECS=${TTL:-90}
export DYN_LOGGING_JSONL=1 DYN_LOG=${DYN_LOG:-info}
source /root/venv/bin/activate
export PYTHONPATH=/root/dynamo/components/src

cat > "$OUT/meta.json" <<JSON
{"workers": $WORKERS, "num_gpu_blocks": $BLOCKS, "block_size": 16, "speedup": $SPEEDUP,
 "sessions": $SESSIONS, "active_window_s": ${WINDOW:-30}, "idle_ttl_s": ${TTL:-90}}
JSON

python -m dynamo.frontend --http-port 8000 --discovery-backend file --router-mode kv \
  --router-kv-events --kv-cache-block-size 16 --enable-session-prefix-index \
  > "$OUT/frontend.log" 2>&1 &
FRONTEND=$!
python -m dynamo.mocker --model-path "$MODEL" --discovery-backend file \
  --num-workers "$WORKERS" --num-gpu-blocks-override "$BLOCKS" --block-size 16 \
  --max-num-seqs 64 --speedup-ratio "$SPEEDUP" --enable-prefix-caching \
  > "$OUT/mocker.log" 2>&1 &
MOCKER=$!

for _ in $(seq 1 120); do
  if curl -s localhost:8000/v1/models | grep -q "$MODEL"; then READY=1; break; fi
  sleep 2
done
[ "${READY:-0}" = 1 ] || { echo "frontend never saw the model"; tail -20 "$OUT/frontend.log" "$OUT/mocker.log"; kill $FRONTEND $MOCKER; exit 2; }
echo "ready"

python "$HERE/scrape.py" --url http://localhost:8000/metrics --out "$OUT/metrics.jsonl" \
  --stop-file "$OUT/stop" --interval 2 &
SCRAPER=$!
python "$HERE/drive.py" --url http://localhost:8000 --model "$MODEL" --sessions "$SESSIONS" \
  --ramp-s "$RAMP_S" --gap-scale "$GAP_SCALE" --growth-words "$GROWTH_WORDS" --out "$OUT/requests.jsonl"
# Let idle sessions expire and the sweep publish before stopping.
sleep $(( ${TTL:-90} + 10 ))
touch "$OUT/stop"
sleep 3
kill $SCRAPER $FRONTEND $MOCKER 2>/dev/null
echo "done: $OUT"
