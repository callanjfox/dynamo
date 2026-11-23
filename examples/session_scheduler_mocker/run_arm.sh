#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One arm of the session scheduler mocker test: mocker workers with the DeepSeek-V4-Pro TP4 B300 step model and
# host KV offload, the Dynamo frontend with the KV router and a router policy file, and the closed-loop users
# driver (python -m dynamo.replay.users_http) playing a users export. See docs/mocker-quickstart.md.
#
#   USERS_DIR=<users export> examples/session_scheduler_mocker/run_arm.sh <arm> <out dir>
#
# Arms: kv (plain KV router), today (KV router with per-class busy thresholds), b0lt (session scheduler B0+lt,
# with engine class scheduling). Settings (environment):
#   USERS_DIR  users export directory (schedule.json, templates, request_fields.users.jsonl); required
#   SPEEDUP    simulated seconds per wall second, for the mocker, the scheduler clock and the driver (default 1)
#   STOP_S     simulated seconds to run (default 2700)
#   WORKERS    mocker workers (default 4)
#   PORT       frontend HTTP port (default 8000)
#   PYTHON     python of the Dynamo environment (default python3)
set -euo pipefail
ARM=${1:?arm: kv, today or b0lt}
OUT=${2:?output directory}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
USERS_DIR=${USERS_DIR:?set USERS_DIR to a users export directory}
SPEEDUP=${SPEEDUP:-1}
STOP_S=${STOP_S:-2700}
WORKERS=${WORKERS:-4}
PORT=${PORT:-8000}
PYTHON=${PYTHON:-python3}
MODEL=deepseek-ai/DeepSeek-V4-Pro
TOKENIZER=${TOKENIZER:-$REPO/lib/llm/tests/data/sample-models/TinyLlama_v1.1}
VOCAB=${VOCAB:-32000}
POLICY=$HERE/policies/$ARM.yaml
[ -f "$POLICY" ] || { echo "no policy $POLICY"; exit 2; }
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

# The engine config at this speedup.
ENGINE=$OUT/$ARM.engine.json
"$PYTHON" - "$HERE/engine_v4pro_tp4_b300.json" "$ENGINE" "$SPEEDUP" <<'EOF'
import json, sys
config = json.load(open(sys.argv[1]))
config["engine"]["speedup_ratio"] = float(sys.argv[3])
json.dump(config, open(sys.argv[2], "w"), indent=1)
EOF
CLASS_ARGS=()
if [ "$ARM" = b0lt ]; then
  # Engine class scheduling: the router writes the class's engine_priority, the mocker maps it back.
  CLASS_ARGS=(--class-scheduling "@$HERE/class_scheduling.json"
              --engine-priority '{"priority": 2, "standard": 1, "batch": 0}')
fi

# File discovery private to this run; TCP request plane; ZMQ event plane.
export DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=$OUT/$ARM.discovery DYN_REQUEST_PLANE=tcp
mkdir -p "$DYN_FILE_KV"
PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for i in $(seq 1 "$WORKERS"); do
  "$PYTHON" -m dynamo.mocker --model-path "$TOKENIZER" --model-name "$MODEL" \
    --extra-engine-args "$ENGINE" --kv-bytes-per-token 4444 "${CLASS_ARGS[@]}" \
    > "$OUT/$ARM.mocker$i.log" 2>&1 &
  PIDS+=($!)
done
# The session scheduler's clock runs at the engines' speed.
DYNAMO_SESSION_SCHEDULER_CLOCK_SCALE=$SPEEDUP \
DYNAMO_SESSION_LOG=$OUT/$ARM.sessions.jsonl DYNAMO_F42_LOG=$OUT/$ARM.f42.jsonl DYNAMO_LOBBY_LOG=$OUT/$ARM.lobby.jsonl \
  "$PYTHON" -m dynamo.frontend --http-port "$PORT" --router-mode kv --router-policy-config "$POLICY" \
    --router-kv-overlap-score-credit 1.0 --router-prefill-load-scale 1.0 --router-temperature 0 \
    --router-prefill-load-model none \
    > "$OUT/$ARM.frontend.log" 2>&1 &
PIDS+=($!)

# Wait until every worker has registered with the frontend.
for _ in $(seq 1 900); do
  n=$( (grep -o "Adding worker WorkerWithDpRank" "$OUT/$ARM.frontend.log" 2>/dev/null || true) | wc -l)
  if [ "$n" -ge "$WORKERS" ] && curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "$MODEL"; then break; fi
  sleep 1
done

"$PYTHON" -m dynamo.replay.users_http --users-dir "$USERS_DIR" --url "http://127.0.0.1:$PORT" --model "$MODEL" \
  --out "$OUT/$ARM.jsonl" --speedup "$SPEEDUP" --stop-s "$STOP_S" --vocab-size "$VOCAB" --arm "$ARM" \
  --policy-file "$POLICY" --record "workers=$WORKERS" --record "engine=\"$ENGINE\"" \
  2>&1 | tee "$OUT/$ARM.driver.log"
