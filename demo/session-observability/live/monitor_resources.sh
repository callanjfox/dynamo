#!/bin/bash
# Sample the memory and CPU of the run's processes once a minute, to see growth over a long run.
#   monitor_resources.sh <out dir>
out=$1
while [ ! -e "$out/stop" ]; do
  now=$(date +%s)
  ps -eo rss,pcpu,args | grep -E "dynamo\.frontend|dynamo\.mocker|agentx_drive|chat_load|scrape\.py" | grep -v grep | \
    awk -v t=$now '{n=$3" "$4" "$5; printf "{\"t\":%d,\"rss_kb\":%d,\"pcpu\":%s,\"proc\":\"%s\"}\n", t, $1, $2, n}' >> "$out/resources.jsonl"
  echo "{\"t\":$now,\"cgroup_mem_mb\":$(( $(cat /sys/fs/cgroup/memory.current) / 1048576 )),\"load\":\"$(cut -d' ' -f1 /proc/loadavg)\"}" >> "$out/resources.jsonl"
  sleep 60
done
