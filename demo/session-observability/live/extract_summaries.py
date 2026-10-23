#!/usr/bin/env python3
"""Pull the router's session_summary log lines out of a frontend log.

    python3 extract_summaries.py frontend.log > summaries.jsonl
"""

import json
import sys

for line in open(sys.argv[1]):
    if "dynamo_session_summary" not in line:
        continue
    try:
        record = json.loads(line)
    except ValueError:
        continue
    if record.get("target") == "dynamo_session_summary":
        record.pop("time", None)
        print(json.dumps(record))
