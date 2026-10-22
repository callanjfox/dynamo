#!/usr/bin/env python3
"""Poll the frontend's /metrics (and any other endpoint given) and keep the router session, cache-reuse funnel and
compute-health series as JSON lines. Frontend series keep their name; frontend-only names that would
collide with a router series get an "fe_" prefix.

    python3 scrape.py --url http://localhost:8000/metrics --url http://localhost:8081/metrics \
        --out metrics.jsonl --stop-file stop
"""

import argparse
import json
import os
import time
import urllib.request

from prometheus_client.parser import text_string_to_metric_families

KEEP = (
    "router_session",
    "router_sessions",
    "router_kv_best_eligible",
    "router_kv_selected",
    "router_kv_worker_reused",
    "router_input_sequence_tokens",
    "router_output_sequence_tokens",
    "router_requests",
    "router_queue_pending",
    "router_overhead",
)
# Compute health: frontend latency and queue series and the router's per-worker load (booked
# decode blocks, active requests by phase, queued prefill tokens).
COMPUTE = (
    "dynamo_frontend_time_to_first_token_seconds",
    "dynamo_frontend_inter_token_latency_seconds",
    "dynamo_frontend_request_duration_seconds",
    "dynamo_frontend_queued_requests",
    "dynamo_frontend_inflight_requests",
    "dynamo_frontend_worker_",
    "dynamo_frontend_requests",
    "queue_pending_requests",
)
DROP_LABELS = {"dynamo_namespace", "dynamo_component", "dynamo_endpoint", "router_id", "worker_id"}
# Per-worker series keep their worker or engine label (a handful of values, the worker count).
PER_WORKER = ("dynamo_frontend_worker_",)


def sample(url):
    text = urllib.request.urlopen(url, timeout=5).read().decode()
    out = {}
    for family in text_string_to_metric_families(text):
        compute = any(family.name.startswith(key) for key in COMPUTE)
        if not compute and not any(key in family.name for key in KEEP):
            continue
        per_worker = any(family.name.startswith(key) for key in PER_WORKER)
        for s in family.samples:
            drop = DROP_LABELS - ({"worker_id"} if per_worker else set())
            labels = {k: v for k, v in s.labels.items() if k not in drop}
            if compute:
                key = s.name.replace("dynamo_frontend_", "fe_", 1)
            else:
                key = s.name.split("router_", 1)[-1]
            if labels:
                key += "{" + ",".join(f"{k}={v}" for k, v in sorted(labels.items())) + "}"
            out[key] = out.get(key, 0.0) + s.value
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", action="append", help="a /metrics endpoint; repeat for the workers")
    parser.add_argument("--out", default="metrics.jsonl")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--stop-file", default="stop")
    args = parser.parse_args()
    with open(args.out, "w") as out:
        while not os.path.exists(args.stop_file):
            try:
                merged = {}
                for url in args.url or ["http://localhost:8000/metrics"]:
                    try:
                        merged.update(sample(url))
                    except Exception as error:
                        # A worker endpoint that is down must not lose the frontend sample.
                        merged.setdefault("scrape_errors", 0)
                        merged["scrape_errors"] += 1
                out.write(json.dumps({"t": time.time(), "m": merged}) + "\n")
                out.flush()
            except Exception as error:
                out.write(json.dumps({"t": time.time(), "error": str(error)}) + "\n")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
