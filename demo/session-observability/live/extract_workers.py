#!/usr/bin/env python3
"""Pull the router's per-worker session_worker_load log events out of a frontend log.

    python3 extract_workers.py frontend.log > workers.jsonl
"""

import json
import sys

for line in open(sys.argv[1]):
    if "session_worker_load" not in line:
        continue
    try:
        r = json.loads(line)
    except ValueError:
        continue
    fields = r.get("fields") or r
    keys = ("worker_id", "dp_rank", "demand_tokens", "resident_device_tokens", "resident_host_tokens",
            "left_behind_device_tokens", "left_behind_host_tokens", "capacity_tokens")
    if all(k in fields for k in keys):
        print(json.dumps({"time": r["time"], **{k: fields[k] for k in keys}}, separators=(",", ":")))
