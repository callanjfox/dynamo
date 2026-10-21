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


LIVE_RUNS = [
    ("agentx_gpu", "AgentX, live", "GPU memory only",
     "Recorded AgentX coding-agent sessions ({sessions} including subagents) through the production router with the session index on, "
     "{workers} mock workers ({gpu}k tokens of GPU cache){host_note}."),
    ("agentx_host", "AgentX, live", "GPU + host memory",
     "The same AgentX sessions with a host-memory tier on each worker: {workers} mock workers, {gpu}k tokens of GPU cache{host_note}."),
    ("heavy", "Synthetic, live", "Real router, mock workers",
     "The production router with the session index on, {workers} mock workers ({gpu}k tokens of GPU cache), "
     "and a synthetic client sending Dynamo session headers."),
]


def total(m, prefix):
    """Sum the scraped series whose key starts with prefix (the scraper drops the model label)."""
    return int(sum(v for k, v in m.items() if k.startswith(prefix)))


def live_run(directory):
    """Turn a live run (scraped /metrics plus the router's session_summary logs) into
    the same row and session shapes the replay produces."""
    meta = json.loads((directory / "meta.json").read_text())
    scrapes = [r for r in read_jsonl(directory / "metrics.jsonl") if "m" in r]
    t0 = scrapes[0]["t"]
    capacity = meta["workers"] * meta["num_gpu_blocks"] * meta["block_size"]
    capacity_host = meta["workers"] * meta.get("num_host_blocks", 0) * meta["block_size"]
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
            "replicated_device_tokens": g("session_replicated_tokens{tier=device}"),
            "replicated_host_tokens": g("session_replicated_tokens{tier=host}"),
            "coverage": {
                "requests_covered": g("session_coverage_requests_total{coverage=covered}"),
                "requests_uncovered": g("session_coverage_requests_total{coverage=uncovered}"),
                "tokens_covered": g("session_coverage_input_tokens_total{coverage=covered}"),
                "tokens_uncovered": g("session_coverage_input_tokens_total{coverage=uncovered}"),
            },
            "funnel": {
                "f0": total(m, "input_sequence_tokens_sum"),
                "f2": total(m, "kv_best_eligible_cached_prefix_tokens_total"),
                "f3": total(m, "kv_selected_cached_prefix_tokens_total"),
                "f4": total(m, "kv_worker_reused_tokens_total"),
            },
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
            "capacity_host_tokens": capacity_host,
        })
    requests = [r for r in read_jsonl(directory / "requests.jsonl")] if (directory / "requests.jsonl").exists() else []
    first_last = {}
    for r in requests:
        if r.get("prompt_tokens"):
            first_last.setdefault(r["session"], []).append(r["prompt_tokens"])
    sessions = []
    for record in read_jsonl(directory / "summaries.jsonl"):
        prompts = first_last.get(record["session_id"], [])
        growth = (prompts[-1] - prompts[0]) / (len(prompts) - 1) if len(prompts) > 1 else 0
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
            "growth_per_turn": growth,
            "thrash": {
                "recomputed": record["thrash_recomputed_tokens"],
                "reloaded_host": record["thrash_reloaded_host_tokens"],
                "reloaded_disk": record["thrash_reloaded_disk_tokens"],
            },
        })
    ok = [r for r in requests if r.get("status") == 200]
    latencies = sorted(r["latency_s"] for r in ok)
    request_stats = {
        "sent": len(requests),
        "ok": len(ok),
        "failed": len(requests) - len(ok),
        "p50_s": latencies[len(latencies) // 2] if latencies else 0,
        "p90_s": latencies[int(len(latencies) * 0.9)] if latencies else 0,
    }
    return rows, sessions, meta, request_stats


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    payload = []
    for scenario in SCENARIOS:
        rows = read_jsonl(DATA / f"{scenario['key']}.jsonl")
        sessions = read_jsonl(DATA / f"{scenario['key']}.jsonl.sessions")
        payload.append({**scenario, "rows": rows, "sessions": sessions})
    for key, label, detail, blurb in LIVE_RUNS:
        directory = HERE / "data" / "live" / key
        if not (directory / "meta.json").exists():
            continue
        rows, sessions, meta, request_stats = live_run(directory)
        host_note = (
            f" plus {meta['num_host_blocks'] * meta['block_size'] * meta['workers'] // 1000}k tokens of host memory"
            if meta.get("num_host_blocks") else ""
        )
        payload.append({
            "key": key,
            "label": label,
            "detail": detail,
            "blurb": blurb.format(
                workers=meta["workers"],
                gpu=meta["num_gpu_blocks"] * meta["block_size"] * meta["workers"] // 1000,
                host_note=host_note,
                sessions=len(sessions),
            ),
            "demandIsSum": True,
            "requests": request_stats,
            "intro": (
                "AgentX coding-agent sessions replayed through a live Dynamo frontend and KV router with mock workers. "
                "These are the router's own session metrics, scraped from /metrics while the workload ran."
                if meta.get("workload") == "agentx" else
                "A synthetic agent workload sent through a live Dynamo frontend and KV router with mock workers. "
                "These are the router's own session metrics, scraped from /metrics while the workload ran."
            ),
            "note1": (
                "Demand is the router's sum over active sessions of the prefix each will present next. "
                "Resident is the distinct tokens the session index holds in each tier."
            ),
            "foot": (
                "Wall-clock time. Real frontend and KV router with the session index on; workers are mockers, "
                "so there are no GPUs and the cache model is the mocker's. Demand is the router's per-session sum, "
                "which counts a prefix once per session and overstates distinct demand when sessions share one "
                "(a subagent starts from its parent's prompt)."
                + (f" Sessions replay the recorded AgentX think times, capped at 300 s and divided by {meta['time_scale']}; "
                   f"the active window is {meta['active_window_s']} s." if meta.get("workload") == "agentx" else "")
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
