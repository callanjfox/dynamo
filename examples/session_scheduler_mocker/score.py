#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score runs (live mocker or dynosim per-request records) over one window: per class, requests offered,
completed, attainment (TTFT and mean ITL within the class limits, over completed requests and over offered),
TTFT p50/p90, mean ITL p50, refusals and Retry-After share, output tokens.

    python3 -I score.py --window-s 2700 --classes <users_dir>/classes.json NAME=STEM [NAME=STEM ...]

STEM is a run's path without .jsonl (for example runs/smoke-s31/b0lt). Class limits: priority 3 s TTFT and
50 ms mean ITL, standard 8 s and 100 ms; batch has none. A request counts in the window by its first offer;
primers (request ids ending :primer) are left out.
"""
import argparse
import json
import statistics

LIMITS = {"priority": (3.0, 0.050), "standard": (8.0, 0.100)}


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def load(stem, classes, window_s):
    by = {}
    for line in open(stem + ".jsonl"):
        r = json.loads(line)
        rid = r.get("request_id") or ""
        cls = classes.get(rid[2:14]) if rid.startswith("cp") else r.get("policy_class")
        cls = cls or r.get("policy_class") or "?"
        offer = (r.get("dispatched_at_ms") or r.get("arrival_time_ms") or 0) / 1000
        if offer >= window_s:
            continue
        c = by.setdefault(cls, dict(offered=0, done=0, met=0, ttft=[], itl=[], out=0, primer=0))
        if rid.endswith(":primer"):
            c["primer"] += 1
            continue
        c["offered"] += 1
        term = r.get("terminal_time_ms")
        done = r.get("terminal_status") == "completed" and (r.get("output_length") or 0) > 0 \
            and term is not None and term / 1000 <= window_s
        if not done:
            continue
        c["done"] += 1
        c["out"] += r.get("output_length") or 0
        ttft = r["ttft_ms"] / 1000 if r.get("ttft_ms") is not None else None
        ft, lt, n = r.get("first_token_ms"), r.get("last_token_ms"), r.get("output_length") or 0
        itl = (lt - ft) / (n - 1) / 1000 if n > 1 and ft is not None and lt is not None else None
        if ttft is not None:
            c["ttft"].append(ttft)
        if itl is not None:
            c["itl"].append(itl)
        tl, il = LIMITS.get(cls, (None, None))
        if tl is None or (ttft is not None and ttft <= tl and (itl is None or itl <= il)):
            c["met"] += 1
    refused, with_ra, ra = {}, {}, {}
    try:
        for line in open(stem + ".refusals.jsonl"):
            e = json.loads(line)
            if e.get("kind") != "refused" or (e.get("t") or 0) >= window_s:
                continue
            cls = e.get("class") or "?"
            refused[cls] = refused.get(cls, 0) + 1
            if e.get("retry_after_s") is not None:
                with_ra[cls] = with_ra.get(cls, 0) + 1
                ra.setdefault(cls, []).append(e["retry_after_s"])
    except FileNotFoundError:
        pass
    return by, refused, with_ra, ra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-s", type=float, required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    classes = {k[:12]: v for k, v in json.load(open(a.classes))["classes"].items()}
    rows = []
    for spec in a.runs:
        name, stem = spec.split("=", 1)
        by, refused, with_ra, ra = load(stem, classes, a.window_s)
        for cls in ("priority", "standard", "batch"):
            c = by.get(cls)
            if not c:
                continue
            f = lambda x: "-" if x is None else f"{x:.2f}"
            rows.append([name, cls, c["offered"], c["done"],
                         f(c["met"] / c["done"] if c["done"] else None),
                         f(c["met"] / c["offered"] if c["offered"] else None),
                         f(pct(c["ttft"], 0.5)), f(pct(c["ttft"], 0.9)),
                         "-" if not c["itl"] else f"{1000 * statistics.median(c['itl']):.1f}",
                         refused.get(cls, 0), with_ra.get(cls, 0),
                         "-" if not ra.get(cls) else f"{statistics.median(ra[cls]):.0f}",
                         f"{c['out'] / 1e6:.2f}"])
    head = ["run", "class", "offered", "done", "att_done", "att_offered", "ttft_p50_s", "ttft_p90_s",
            "itl_p50_ms", "refusals", "with_RA", "RA_p50_s", "out_Mtok"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")


if __name__ == "__main__":
    main()
