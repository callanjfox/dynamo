#!/bin/bash
# 6 hour run, 4 workers x 75000 blocks. Sessions arrive evenly over the run at real think times, so the number
# running at once settles at arrival rate times session length (about an hour here), and stays there if the
# fleet keeps up. SESSIONS sets the arrival rate: 260 gives about 40 running at once (demand near 0.4x the
# GPU cache); 1440 gave 1,100 running at once and demand of 7.8x the cache, so requests queued for minutes and
# every session was cut short.
#
#   NAME=steady6h SESSIONS=260 VENV=/root/venv_new ./long_run.sh
cd /root
NAME=${NAME:-long6h}; SESSIONS=${SESSIONS:-260}
export VIRTUAL_ENV=${VENV:-/root/venv}
export PATH=$VIRTUAL_ENV/bin:$PATH
OUTROOT=/root/live_out
mkdir -p $OUTROOT/trace_$NAME
env DYN_REQUEST_TRACE_RECORDS=router_selection DYN_REQUEST_TRACE_SINKS=file DYN_REQUEST_TRACE_FILE_FORMAT=jsonl \
  DYN_REQUEST_TRACE_FILE_PATH=$OUTROOT/trace_$NAME/rs.jsonl OUT=$OUTROOT/fz_$NAME SUBSET=/root/subset.json \
  SESSIONS=$SESSIONS REPEAT=$(( (SESSIONS + 39) / 40 )) WORKERS=4 MODEL=deepseek-ai/DeepSeek-V4-Flash BLOCKS=75000 HOST_BLOCKS=0 SPEEDUP=1 TIME_SCALE=1 \
  RAMP_S=21600 DURATION_S=21600 WINDOW=900 TTL=1800 CHAT_RPS=1 CHAT_PROMPT=2000 SHARED_PREFIX=8192 PORT=${PORT:-9200} \
  MOCKER_EXTRA="--engine-type vllm --ais-perf-model --ais-system b200_sxm --ais-backend-version 0.24.0 --ais-tp-size 8 --ais-moe-tp-size 1 --ais-moe-ep-size 8" \
  /root/live12/run_agentx.sh > $OUTROOT/$NAME.log 2>&1 &
RUN=$!
sleep 60
/root/live12/monitor_resources.sh $OUTROOT/fz_$NAME &
wait $RUN
# Post-process so the results are ready.
cd /root/live12
d=$OUTROOT/fz_$NAME
python3 extract_summaries.py $d/frontend.log > $d/summaries.jsonl
python3 extract_workers.py $d/frontend.log > $d/workers.jsonl
python3 extract_selection.py $OUTROOT/trace_$NAME/rs.jsonl > $d/selection.jsonl
python3 trim_metrics.py $d/metrics.jsonl
echo POSTPROCESSED > $OUTROOT/$NAME.status
