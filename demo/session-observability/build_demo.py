#!/usr/bin/env python3
"""Build index.html from the replay's session stats output.

Reads data/final/<scenario>.jsonl (one working-set row per sample interval) and
<scenario>.jsonl.sessions (one summary per session), and writes a single
self-contained index.html with the data embedded.

    python3 build_demo.py
"""

import json
import pathlib

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data" / "final"

SCENARIOS = [
    {
        "key": "light",
        "label": "Light load",
        "detail": "GPU memory only",
        "blurb": "15 agent tasks on 4 workers. The working set stays under GPU capacity.",
    },
    {
        "key": "heavy_gpu",
        "label": "Heavy load",
        "detail": "GPU memory only",
        "blurb": "40 agent tasks on the same 4 workers. Demand passes GPU capacity and returning sessions recompute their prefix.",
    },
    {
        "key": "heavy_host",
        "label": "Heavy load",
        "detail": "GPU + host memory",
        "blurb": "Same load with a host-memory tier. Evicted prefixes are reloaded instead of recomputed.",
    },
]


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    payload = []
    for scenario in SCENARIOS:
        rows = read_jsonl(DATA / f"{scenario['key']}.jsonl")
        sessions = read_jsonl(DATA / f"{scenario['key']}.jsonl.sessions")
        payload.append({**scenario, "rows": rows, "sessions": sessions})
    template = (HERE / "template.html").read_text()
    out = template.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    (HERE / "index.html").write_text(out)
    print(f"wrote {HERE / 'index.html'} ({len(out) // 1024} KiB)")


if __name__ == "__main__":
    main()
