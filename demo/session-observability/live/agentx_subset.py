#!/usr/bin/env python3
"""Cut a few sessions out of the AgentX corpus (semianalysisai/cc-traces-weka-062126).

The corpus is one JSON object per line, one agent session each, with `hash_ids` per
request (the KV block ids, block_size tokens each) and recorded think times. This keeps
the fields the replay needs and flattens subagent groups into their own request lists.

    python3 agentx_subset.py traces.jsonl subset.json --count 40 --min-turns 60 --max-turns 220
"""

import argparse
import json


def request(q):
    return {
        "t": q["t"],
        "in": q["in"],
        "out": q["out"],
        "think": q.get("think_time", 0.0),
        "hash_ids": q["hash_ids"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("dest")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--min-turns", type=int, default=60)
    parser.add_argument("--max-turns", type=int, default=220)
    parser.add_argument("--max-subagents", type=int, default=12)
    args = parser.parse_args()

    picked = []
    with open(args.source) as handle:
        for line in handle:
            trace = json.loads(line)
            main_turns = [q for q in trace["requests"] if q.get("type") in ("n", "s")]
            groups = [q for q in trace["requests"] if q.get("type") == "subagent"]
            if not args.min_turns <= len(main_turns) <= args.max_turns:
                continue
            picked.append({
                "id": trace["id"],
                "block_size": trace["block_size"],
                "main": [request(q) for q in main_turns],
                "subagents": [
                    {
                        "id": group["agent_id"],
                        "t": group["t"],
                        "requests": [request(q) for q in group["requests"]],
                    }
                    for group in groups[: args.max_subagents]
                ],
            })
            if len(picked) >= args.count:
                break
    with open(args.dest, "w") as out:
        json.dump(picked, out, separators=(",", ":"))
    print(f"{len(picked)} sessions, {sum(len(p['main']) for p in picked)} main turns, "
          f"{sum(len(g['requests']) for p in picked for g in p['subagents'])} subagent requests")


if __name__ == "__main__":
    main()
