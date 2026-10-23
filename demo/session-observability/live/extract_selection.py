#!/usr/bin/env python3
"""Summarize router_selection trace records (Dynamo's per-candidate cost trace) into 60 second windows.

    python3 extract_selection.py rs.jsonl > selection.jsonl

Records carry every candidate worker with its total cost and, when the router exposes them, the
inputs the cost combined: [cached_tokens, active_prefill_tokens, decode_blocks, active_requests].
Per window the output has:

    decisions        routed requests
    best_cache       share of decisions that picked the candidate with the most cached tokens
                     (decisions where every candidate had the same cache are not counted either way)
    left_cache       decisions that did not, with the mean cached tokens given up, and the mean
                     prefill backlog (tokens) and decode blocks the best-cache worker had
                     over the chosen one
    reasons          of those decisions, how many the chosen worker beat the best-cache worker on
                     prefill backlog, on decode blocks, on active requests (a decision can count
                     toward several) and how many on none of them
    tie_share        decisions where the best cost was shared by several workers
    margin           mean gap between the best and second-best cost
"""

import collections
import json
import sys

WINDOW_MS = 60_000


def window():
    return {"n": 0, "ties": 0, "margin": 0.0, "with_inputs": 0, "differ": 0, "best": 0, "left": 0,
            "gave_up": 0.0, "d_prefill": 0.0, "d_decode": 0.0, "lighter_prefill": 0, "lighter_decode": 0,
            "lighter_requests": 0, "none": 0, "prompt": 0}


def main():
    windows = collections.defaultdict(window)
    for line in open(sys.argv[1]):
        if not line.strip():
            continue
        event = json.loads(line)["event"]
        sel = event.get("selection")
        if not sel or not sel.get("candidates"):
            continue
        order = sorted(range(len(sel["candidates"])), key=lambda i: sel["candidates"][i][2])
        costs = [sel["candidates"][i][2] for i in order]
        w = windows[event["event_time_unix_ms"] // WINDOW_MS * WINDOW_MS]
        w["n"] += 1
        margin = costs[1] - costs[0] if len(costs) > 1 else 0.0
        w["margin"] += margin
        w["prompt"] += sel.get("prompt_tokens") or 0
        if margin <= 1e-9:
            w["ties"] += 1
        inputs = sel.get("inputs")
        if not inputs or len(inputs) != len(sel["candidates"]) or margin <= 1e-9:
            continue
        # The chosen worker is the lowest cost; candidates are already sorted by cost in the record.
        rows = [inputs[i] for i in order]
        chosen = rows[0]
        caches = [r[0] for r in rows]
        if max(caches) == min(caches):
            continue
        w["differ"] += 1
        best = max(rows, key=lambda r: r[0])
        if chosen[0] >= best[0]:
            w["best"] += 1
            continue
        w["left"] += 1
        w["gave_up"] += best[0] - chosen[0]
        w["d_prefill"] += best[1] - chosen[1]
        w["d_decode"] += best[2] - chosen[2]
        lp, ld, lr = best[1] > chosen[1], best[2] > chosen[2], best[3] > chosen[3]
        w["lighter_prefill"] += lp
        w["lighter_decode"] += ld
        w["lighter_requests"] += lr
        w["none"] += not (lp or ld or lr)
    for start in sorted(windows):
        w = windows[start]
        left = max(w["left"], 1)
        print(json.dumps({
            "t_ms": start,
            "decisions": w["n"],
            "tie_share": w["ties"] / w["n"],
            "margin": w["margin"] / w["n"],
            "differ": w["differ"],
            "best_cache": w["best"],
            "left_cache": w["left"],
            "gave_up_tokens": w["gave_up"] / left,
            "prefill_saved_tokens": w["d_prefill"] / left,
            "decode_saved_blocks": w["d_decode"] / left,
            "lighter_prefill": w["lighter_prefill"],
            "lighter_decode": w["lighter_decode"],
            "lighter_requests": w["lighter_requests"],
            "no_load_advantage": w["none"],
        }, separators=(",", ":")))


if __name__ == "__main__":
    main()
