#!/usr/bin/env python3
"""Poll a frontend's /metrics and keep the router session series as JSON lines.

    python3 scrape.py --url http://localhost:8000/metrics --out metrics.jsonl --stop-file stop
"""

import argparse
import json
import os
import time
import urllib.request

from prometheus_client.parser import text_string_to_metric_families

KEEP = ("router_session", "router_sessions")
DROP_LABELS = {"dynamo_namespace", "dynamo_component", "dynamo_endpoint", "router_id", "worker_id"}


def sample(url):
    text = urllib.request.urlopen(url, timeout=5).read().decode()
    out = {}
    for family in text_string_to_metric_families(text):
        if not any(key in family.name for key in KEEP):
            continue
        for s in family.samples:
            labels = {k: v for k, v in s.labels.items() if k not in DROP_LABELS}
            key = s.name.split("router_", 1)[-1]
            if labels:
                key += "{" + ",".join(f"{k}={v}" for k, v in sorted(labels.items())) + "}"
            out[key] = out.get(key, 0.0) + s.value
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/metrics")
    parser.add_argument("--out", default="metrics.jsonl")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--stop-file", default="stop")
    args = parser.parse_args()
    with open(args.out, "w") as out:
        while not os.path.exists(args.stop_file):
            try:
                out.write(json.dumps({"t": time.time(), "m": sample(args.url)}) + "\n")
                out.flush()
            except Exception as error:
                out.write(json.dumps({"t": time.time(), "error": str(error)}) + "\n")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
