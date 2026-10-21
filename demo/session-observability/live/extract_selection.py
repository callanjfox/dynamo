#!/usr/bin/env python3
"""Summarize router_selection trace records (Dynamo's per-candidate cost trace) into 60 second windows.

    python3 extract_selection.py rs.jsonl > selection.jsonl

Each output row: window start (unix ms), decisions, mean candidate count, mean gap between the
best and second-best cost, the share of decisions where the best cost was tied, and the busiest
worker's share of the decisions that were not ties.
"""

import collections
import json
import sys

WINDOW_MS = 60_000


def main():
    windows = collections.defaultdict(lambda: {"n": 0, "cands": 0, "margin": 0.0, "ties": 0, "won": collections.Counter(), "decided": 0})
    for line in open(sys.argv[1]):
        if not line.strip():
            continue
        event = json.loads(line)["event"]
        sel = event.get("selection")
        if not sel or not sel.get("candidates"):
            continue
        costs = sorted(sel["candidates"], key=lambda c: c[2])
        w = windows[event["event_time_unix_ms"] // WINDOW_MS * WINDOW_MS]
        w["n"] += 1
        w["cands"] += sel["candidate_count"]
        margin = costs[1][2] - costs[0][2] if len(costs) > 1 else 0.0
        w["margin"] += margin
        if margin <= 1e-9:
            w["ties"] += 1
        else:
            w["decided"] += 1
            w["won"][(costs[0][0], costs[0][1])] += 1
    for start in sorted(windows):
        w = windows[start]
        print(json.dumps({
            "t_ms": start,
            "decisions": w["n"],
            "candidates": w["cands"] / w["n"],
            "margin": w["margin"] / w["n"],
            "tie_share": w["ties"] / w["n"],
            "top_worker_share": (max(w["won"].values()) / w["decided"]) if w["decided"] else None,
        }, separators=(",", ":")))


if __name__ == "__main__":
    main()
