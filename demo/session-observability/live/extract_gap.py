#!/usr/bin/env python3
"""Why did the router pass over the highest-cache worker? Join the router_selection trace with the driver's request log.

    python3 extract_gap.py rs.jsonl requests.jsonl

Prints, per request: how much of its prompt the best-cache candidate and the chosen candidate held, then groups the decisions by

    gap since the same session's previous request finished   (does a quick follow-up miss cache the router has not indexed yet?)
    chosen-minus-best cost margin                             (was the pass-over a near tie, or a clear load win?)
    role                                                      (primary or subagent)
"""

import collections
import json
import sys


def bucket(x, edges, labels):
    for e, l in zip(edges, labels):
        if x <= e:
            return l
    return labels[-1]


def main():
    # The frontend gives each request its own trace ID, so join by prompt length and start time: the router
    # decides within a few milliseconds of the request leaving the driver.
    rows = []
    for line in open(sys.argv[2]):
        r = json.loads(line)
        if r.get("status") == 200 and r.get("end"):
            rows.append(r)
    by_session = collections.defaultdict(list)
    for r in rows:
        by_session[r["session"]].append(r)
    prev_end = {}
    for srows in by_session.values():
        srows.sort(key=lambda r: r["turn"])
        for i, r in enumerate(srows):
            prev_end[id(r)] = srows[i - 1]["end"] if i else None
    by_prompt = collections.defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_tokens"]].append(r)
    claimed = set()

    def match(sel, decided):
        best = None
        for r in by_prompt.get(sel.get("prompt_tokens"), ()):
            d = decided - r["t"]
            if id(r) in claimed or not -0.05 <= d <= 2.0:
                continue
            if best is None or d < best[0]:
                best = (d, r)
        if best:
            claimed.add(id(best[1]))
            return best[1]
        return None

    out = collections.defaultdict(lambda: collections.Counter())
    n = hits = 0
    for line in open(sys.argv[1]):
        if not line.strip():
            continue
        ev = json.loads(line)["event"]
        sel = ev.get("selection")
        if not sel or not sel.get("inputs") or len(sel["inputs"]) != len(sel["candidates"]):
            continue
        decided_ms = ev["event_time_unix_ms"] / 1000.0
        r = match(sel, decided_ms)
        if not r:
            continue
        hits += 1
        order = sorted(range(len(sel["candidates"])), key=lambda i: sel["candidates"][i][2])
        rows = [sel["inputs"][i] for i in order]
        costs = [sel["candidates"][i][2] for i in order]
        chosen = rows[0]
        best_i = max(range(len(rows)), key=lambda i: rows[i][0])
        best = rows[best_i]
        prompt = sel.get("prompt_tokens") or r["prompt_tokens"] or 1
        gap = (decided_ms - prev_end[id(r)]) if prev_end[id(r)] else None
        role = "subagent" if r["parent"] else "primary"
        left = chosen[0] < best[0]
        margin = (costs[best_i] - costs[0]) / max(costs[0], 1e-9)
        keys = [
            ("gap", "first turn" if gap is None else bucket(gap, [0.5, 2, 10, 60], ["<=0.5s", "<=2s", "<=10s", "<=60s", ">60s"])),
            ("role", role),
        ]
        if left:
            keys.append(("margin", bucket(margin, [0.02, 0.1, 0.5], ["<2% (near tie)", "<10%", "<50%", ">=50% (clear load win)"])))
        for k, v in keys:
            c = out[(k, v)]
            c["n"] += 1
            c["left"] += left
            c["chosen_frac_x1000"] += int(1000 * chosen[0] / prompt)
            c["best_frac_x1000"] += int(1000 * best[0] / prompt)
        n += 1
    print(f"joined {hits} decisions with a request record")
    for (k, v), c in sorted(out.items()):
        print(f"{k:7} {v:26} n={c['n']:6}  left_cache={100 * c['left'] / c['n']:5.1f}%  best-cache holds {c['best_frac_x1000'] / c['n'] / 10:5.1f}% of prompt  chosen holds {c['chosen_frac_x1000'] / c['n'] / 10:5.1f}%")


main()
