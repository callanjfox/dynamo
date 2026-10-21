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


GAP_LABELS = ["le_1s", "le_5s", "le_10s", "le_30s", "le_60s", "le_120s", "le_300s", "le_600s", "le_1800s", "gt_1800s"]
OUTCOMES = ["gpu", "host", "disk", "miss"]


def live_run(directory):
    """Turn a live run (scraped /metrics plus the router's session_summary logs) into
    the same row and session shapes the replay produces."""
    meta = json.loads((directory / "meta.json").read_text())
    scrapes = [r for r in read_jsonl(directory / "metrics.jsonl") if "m" in r]
    t0 = scrapes[0]["t"]
    capacity = meta["workers"] * meta["num_gpu_blocks"] * meta["block_size"]
    rows = []
    for scrape in scrapes:
        m = scrape["m"]
        g = lambda key: int(m.get(key, 0))
        rows.append({
            "now_ms": int((scrape["t"] - t0) * 1000),
            "active_sessions": g("sessions_active"),
            "demand_tokens": g("session_demand_tokens"),
            "demand_distinct_tokens": g("session_demand_tokens"),
            "distinct_device_tokens": g("session_resident_tokens{tier=device}"),
            "distinct_host_tokens": g("session_resident_tokens{tier=host}"),
            "stranded_device_tokens": g("session_stranded_tokens{tier=device}"),
            "split_sessions": g("sessions_split"),
            "thrash": {
                "recomputed": g("session_thrash_tokens_total{outcome=recomputed}"),
                "reloaded_host": g("session_thrash_tokens_total{outcome=reloaded_host}"),
                "reloaded_disk": g("session_thrash_tokens_total{outcome=reloaded_disk}"),
            },
            "gap_outcomes": [
                [g(f"session_return_turns_total{{gap={gap},outcome={outcome}}}") for outcome in OUTCOMES]
                for gap in GAP_LABELS
            ],
            "capacity_device_tokens": capacity,
            "capacity_host_tokens": 0,
        })
    sessions = []
    for record in read_jsonl(directory / "summaries.jsonl"):
        sessions.append({
            "session_id": record["session_id"],
            "parent_session_id": record.get("parent_session_id") or None,
            "turns": record["turns"],
            "isl_tokens": record["input_tokens"],
            "osl_tokens": record["output_tokens"],
            "cached_tokens": record["cached_tokens"],
            "hit_rate": record["cache_hit_ratio"],
            "duration_ms": record["duration_ms"],
            "mean_turn_gap_ms": record["mean_turn_gap_ms"],
            "peak_demand_tokens": record["peak_context_tokens"],
            "children": record["subagents"],
            "thrash": {
                "recomputed": record["thrash_recomputed_tokens"],
                "reloaded_host": record["thrash_reloaded_host_tokens"],
                "reloaded_disk": record["thrash_reloaded_disk_tokens"],
            },
        })
    return rows, sessions, meta


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    payload = []
    for scenario in SCENARIOS:
        rows = read_jsonl(DATA / f"{scenario['key']}.jsonl")
        sessions = read_jsonl(DATA / f"{scenario['key']}.jsonl.sessions")
        payload.append({**scenario, "rows": rows, "sessions": sessions})
    live_dir = HERE / "data" / "live" / "heavy"
    if live_dir.exists():
        rows, sessions, meta = live_run(live_dir)
        payload.append({
            "key": "live",
            "label": "Live run",
            "detail": "Real router, mock workers",
            "blurb": (
                f"The production router with the session index on, {meta['workers']} mock workers "
                f"({meta['num_gpu_blocks'] * meta['block_size'] * meta['workers'] // 1000}k tokens of GPU cache), "
                "and a client sending Dynamo session headers. Every number is read from the router's real "
                "/metrics and its session_summary log."
            ),
            "demandIsSum": True,
            "intro": (
                "A synthetic agent workload sent through a live Dynamo frontend and KV router with mock workers. "
                "These are the router's own session metrics, scraped from /metrics while the workload ran."
            ),
            "note1": (
                "Demand is the router's sum over active sessions of the prefix each will present next. "
                "Resident is the distinct tokens the session index holds on the GPU."
            ),
            "foot": (
                "Wall-clock time. Real frontend and KV router with the session index on; workers are mockers, "
                "so there are no GPUs. Demand here is the router's per-session sum, which counts a shared prefix "
                "once per session and overstates distinct demand. Subagent prompts start from their parent's, "
                "so that overstatement is larger than in the replay runs."
            ),
            "rows": rows,
            "sessions": sessions,
        })
    template = (HERE / "template.html").read_text()
    out = template.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    (HERE / "index.html").write_text(out)
    print(f"wrote {HERE / 'index.html'} ({len(out) // 1024} KiB)")


if __name__ == "__main__":
    main()
