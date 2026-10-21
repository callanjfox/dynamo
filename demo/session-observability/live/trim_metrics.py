#!/usr/bin/env python3
"""Drop histogram bucket series from a scraped metrics.jsonl; the page does not use them.

    python3 trim_metrics.py data/live/agentx_host/metrics.jsonl
"""

import json
import sys

path = sys.argv[1]
rows = [json.loads(line) for line in open(path)]
with open(path, "w") as out:
    for row in rows:
        if "m" in row:
            row["m"] = {k: v for k, v in row["m"].items() if "_bucket{" not in k}
        out.write(json.dumps(row, separators=(",", ":")) + "\n")
