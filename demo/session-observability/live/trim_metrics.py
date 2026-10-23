#!/usr/bin/env python3
"""Drop histogram bucket series from a scraped metrics.jsonl, except the latency histograms the page
turns into percentiles.

    python3 trim_metrics.py data/live/agentx_host/metrics.jsonl
"""

import json
import sys

path = sys.argv[1]
KEEP_BUCKETS = ("fe_time_to_first_token", "fe_inter_token_latency", "fe_request_duration")
rows = [json.loads(line) for line in open(path)]
# Keep every second scrape: the page windows are 30 s or longer.
rows = [row for i, row in enumerate(rows) if i % 2 == 0 or i == len(rows) - 1]
with open(path, "w") as out:
    for row in rows:
        if "m" in row:
            row["m"] = {k: v for k, v in row["m"].items() if "_bucket{" not in k or k.startswith(KEEP_BUCKETS)}
        out.write(json.dumps(row, separators=(",", ":")) + "\n")
