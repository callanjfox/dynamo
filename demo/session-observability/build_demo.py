#!/usr/bin/env python3
"""Build index.html from the replay's session stats output.

Reads data/final/<scenario>.jsonl (one working-set row per sample interval) and
<scenario>.jsonl.sessions (one summary per session), and writes a single
self-contained index.html with the data embedded.

    python3 build_demo.py
"""

import json
import pathlib
import re

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


# Return-gap buckets: up to 1 minute, 1 to 5, 5 to 15, over 15. Earlier runs used ten finer buckets.
GAP_LABELS = ["le_1m", "le_5m", "le_15m", "gt_15m"]
OLD_GAP_LABELS = ["le_1s", "le_5s", "le_10s", "le_30s", "le_60s", "le_120s", "le_300s", "le_600s", "le_1800s", "gt_1800s"]
# Which old bucket feeds which new one. The 10 to 30 minute bucket straddles 15 minutes and goes to "over".
OLD_GAP_TO_NEW = [0, 0, 0, 0, 0, 1, 1, 2, 3, 3]
OUTCOMES = ["gpu", "host", "disk", "miss"]


LIVE_RUNS = [
    ("ax_gpu", "AgentX, live", "GPU memory only",
     "Recorded AgentX coding-agent sessions ({sessions} including subagents) through the production router with the session index on, "
     "{workers} mock workers ({gpu}k tokens of GPU cache){host_note}."),
    ("ax_host", "AgentX, live", "GPU + host memory",
     "The same AgentX sessions with a host-memory tier on each worker: {workers} mock workers, {gpu}k tokens of GPU cache{host_note}."),
    ("ax_mixed", "AgentX, partial coverage", "30% of sessions without headers",
     "A shorter AgentX run through the same router, where about 30% of the sessions are sent without Dynamo session headers. "
     "{workers} mock workers, {gpu}k tokens of GPU cache{host_note}. The coverage panel shows how much of the traffic the session metrics describe."),
    ("ax_trace", "AgentX, routing decisions", "GPU + host, router_selection trace on",
     "A shorter AgentX run with the router's per-candidate cost trace (router_selection records) turned on, "
     "{workers} mock workers, {gpu}k tokens of GPU cache{host_note}. The routing decisions panel shows how the router chose between workers."),
    ("heavy", "Synthetic, live", "Real router, mock workers",
     "The production router with the session index on, {workers} mock workers ({gpu}k tokens of GPU cache), "
     "and a synthetic client sending Dynamo session headers."),
]


def total(m, prefix):
    """Sum the scraped series whose key starts with prefix (the scraper drops the model label)."""
    return int(sum(v for k, v in m.items() if k.startswith(prefix)))


def percentiles(xs):
    xs = sorted(xs)
    pick = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0
    return {"p50": pick(0.5), "p90": pick(0.9), "max": xs[-1] if xs else 0}


def opt(m, key, cast=int):
    """The series' value, or None when the router that produced the run did not export it."""
    return cast(m[key]) if key in m else None


def gap_outcomes(m):
    """Returning turns by gap bucket (rows) and outcome (columns), in the current four buckets."""
    if any(k.startswith("session_return_turns_total{gap=le_1m") for k in m):
        return [[int(m.get(f"session_return_turns_total{{gap={gap},outcome={outcome}}}", 0)) for outcome in OUTCOMES] for gap in GAP_LABELS]
    out = [[0] * len(OUTCOMES) for _ in GAP_LABELS]
    for old, new in zip(OLD_GAP_LABELS, OLD_GAP_TO_NEW):
        for k, outcome in enumerate(OUTCOMES):
            out[new][k] += int(m.get(f"session_return_turns_total{{gap={old},outcome={outcome}}}", 0))
    return out


def sum_prefix(m, prefix):
    return sum(v for k, v in m.items() if k.startswith(prefix))


def buckets(m, prefix):
    """Cumulative histogram buckets {le: count}, summed over the other labels."""
    out = {}
    for k, v in m.items():
        if k.startswith(prefix + "_bucket{"):
            le = re.search(r"le=([^,}]+)", k).group(1)
            out[float("inf") if le == "+Inf" else float(le)] = out.get(float("inf") if le == "+Inf" else float(le), 0) + v
    return out


def window_quantile(now, before, q):
    """Quantile of the observations between two cumulative bucket snapshots, interpolated inside the bucket."""
    edges = sorted(now)
    delta = [(e, now[e] - before.get(e, 0)) for e in edges]
    total = delta[-1][1] if delta else 0
    if total < 1:
        return None
    prev_edge, prev_cum = 0.0, 0.0
    for edge, cum in delta:
        if cum >= q * total:
            if edge == float("inf"):
                return prev_edge
            span = cum - prev_cum
            return prev_edge + (edge - prev_edge) * ((q * total - prev_cum) / span if span else 1)
        prev_edge, prev_cum = edge, cum
    return prev_edge


def compute_row(m):
    """Instantaneous compute-health gauges from the frontend and the router's per-worker load series, or None when the run did not scrape them."""
    if not any(k.startswith("fe_") for k in m):
        return None
    decode_blocks = [v for _, v in sorted((k, v) for k, v in m.items() if k.startswith("fe_worker_active_decode_blocks"))]
    return {
        "queued": int(m.get("fe_queued_requests", 0)) + int(sum_prefix(m, "queue_pending_requests")),
        "inflight": int(sum_prefix(m, "fe_inflight_requests")),
        "decoding": int(sum(v for k, v in m.items() if k.startswith("fe_worker_active_requests") and "request_phase=decode" in k)),
        "prefilling": int(sum(v for k, v in m.items() if k.startswith("fe_worker_active_requests") and "request_phase=prefill" in k)),
        "prefill_tokens": int(sum_prefix(m, "fe_worker_active_prefill_tokens")),
        "decode_blocks": decode_blocks,
        # Last TTFT and inter-token latency each worker reported, in seconds: the spread across workers.
        "worker_ttft": sorted(v for k, v in m.items() if k.startswith("fe_worker_last_time_to_first_token_seconds")),
        "worker_itl": sorted(v for k, v in m.items() if k.startswith("fe_worker_last_inter_token_latency_seconds")),
    }


def add_latency_percentiles(rows, hists, window_ms=30000):
    """TTFT and ITL p50 and p90 over the trailing window, from the frontend histograms."""
    for i, row in enumerate(rows):
        if not row["compute"]:
            continue
        j = i
        while j > 0 and row["now_ms"] - rows[j]["now_ms"] < window_ms:
            j -= 1
        for name, short in (("time_to_first_token_seconds", "ttft"), ("inter_token_latency_seconds", "itl"), ("request_duration_seconds", "dur")):
            now, before = hists[i][name], hists[j][name]
            row["compute"][short] = {"p50": window_quantile(now, before, 0.5), "p90": window_quantile(now, before, 0.9)} if now else None


def live_run(directory):
    """Turn a live run (scraped /metrics plus the router's session_summary logs) into
    the same row and session shapes the replay produces."""
    meta = json.loads((directory / "meta.json").read_text())
    scrapes = [r for r in read_jsonl(directory / "metrics.jsonl") if "m" in r]
    t0 = scrapes[0]["t"]
    capacity = meta["workers"] * meta["num_gpu_blocks"] * meta["block_size"]
    capacity_host = meta["workers"] * meta.get("num_host_blocks", 0) * meta["block_size"]
    # Runs before the window change report 30s, 1m and 5m; later ones 1m, 5m and 30m.
    last_keys = set().union(*(sc["m"].keys() for sc in scrapes[-5:]))
    if "session_demand_window_tokens{window=15m}" in last_keys:
        window_labels = ("1m", "5m", "15m")
    elif "session_demand_window_tokens{window=30m}" in last_keys:
        window_labels = ("1m", "5m", "30m")
    else:
        window_labels = ("30s", "1m", "5m")
    meta["window_labels"] = list(window_labels)
    rows, hists = [], []
    for scrape in scrapes:
        m = scrape["m"]
        g = lambda key: int(m.get(key, 0))
        rows.append({
            "now_ms": int((scrape["t"] - t0) * 1000),
            "active_sessions": g("sessions_active"),
            "demand_tokens": g("session_demand_tokens"),
            # Newer runs report demand with a shared prefix counted once; older ones only the per-session sum.
            "demand_distinct_tokens": g("session_demand_distinct_tokens") if "session_demand_distinct_tokens" in m else g("session_demand_tokens"),
            "demand_primary_tokens": g("session_demand_by_role_tokens{role=primary}"),
            "demand_subagent_tokens": g("session_demand_by_role_tokens{role=subagent}"),
            "active_primary": g("sessions_active_by_role{role=primary}"),
            "active_subagent": g("sessions_active_by_role{role=subagent}"),
            "shared_tokens": g("session_shared_tokens"),
            "demand_windows": [g(f"session_demand_window_tokens{{window={w}}}") for w in window_labels],
            "worker": {
                **{f"demand_{stat}": opt(m, f"session_worker_demand_tokens{{stat={stat}}}") for stat in ("max", "min", "mean")},
                **{f"ratio_{stat}": opt(m, f"session_worker_demand_ratio{{stat={stat}}}", float) for stat in ("max", "min", "mean")},
                **{f"resident_{stat}_{tier}": opt(m, f"session_worker_resident_tokens{{stat={stat},tier={tier}}}") for stat in ("max", "min", "mean") for tier in ("device", "host")},
                **{f"left_{stat}_{tier}": opt(m, f"session_worker_left_behind_tokens{{stat={stat},tier={tier}}}") for stat in ("max", "min", "mean") for tier in ("device", "host")},
            } if "session_worker_demand_tokens{stat=max}" in m else None,
            "prior_prefix": g("session_prior_prefix_tokens_total"),
            "tier_overlap_tokens": g("session_tier_overlap_tokens"),
            "covered_output_tokens": g("session_covered_output_tokens_total"),
            "distinct_demand": "session_demand_distinct_tokens" in m,
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
            "gap_outcomes": gap_outcomes(m),
            "capacity_device_tokens": capacity,
            "capacity_host_tokens": capacity_host,
            "requests_started": g("requests_started_total"),
            "input_seq": {"sum": total(m, "input_sequence_tokens_sum"), "count": total(m, "input_sequence_tokens_count")},
            "output_seq": {"sum": total(m, "output_sequence_tokens_sum"), "count": total(m, "output_sequence_tokens_count")},
            "compute": compute_row(m),
        })
        hists.append({name: buckets(m, "fe_" + name) for name in ("time_to_first_token_seconds", "inter_token_latency_seconds", "request_duration_seconds")})
    for row in rows:
        if row["compute"]:
            row["compute"]["block_capacity"] = meta["num_gpu_blocks"]
    add_latency_percentiles(rows, hists)
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
        "isl": percentiles([r["prompt_tokens"] for r in ok if r.get("prompt_tokens")]),
        "osl": percentiles([r["completion_tokens"] for r in ok if r.get("completion_tokens")]),
    }
    selection = []
    if (directory / "selection.jsonl").exists():
        t0_ms = t0 * 1000
        selection = [{**r, "now_ms": int(r["t_ms"] - t0_ms)} for r in read_jsonl(directory / "selection.jsonl")]
    meta["selection"] = selection
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
            "selection": meta.get("selection", []),
            "window_labels": meta["window_labels"],
            "block_size": meta["block_size"],
            "workers": meta["workers"],
            "time_scale": meta.get("time_scale", 1),
            "meta_window": meta["active_window_s"],
            "demandIsSum": not any(r.get("distinct_demand") for r in rows),
            "requests": request_stats,
            "intro": (
                "AgentX coding-agent sessions replayed through a live Dynamo frontend and KV router with mock workers. "
                "These are the router's own session metrics, scraped from /metrics while the workload ran."
                if meta.get("workload") == "agentx" else
                "A synthetic agent workload sent through a live Dynamo frontend and KV router with mock workers. "
                "These are the router's own session metrics, scraped from /metrics while the workload ran."
            ),
            "note1": (
                ("Demand counts each distinct prompt block once across the active sessions, so a shared prefix is not counted twice. "
                 if any(r.get("distinct_demand") for r in rows) else
                 "Demand is the router's sum over active sessions of the prefix each will present next, so a shared prefix counts once per session. ")
                + "Held memory is stacked: GPU resident, then host resident"
                + (" that is not also on the GPU." if any(r.get("tier_overlap_tokens") for r in rows) else ", which may include copies of GPU blocks (this run does not report the overlap).")
            ),
            "foot": (
                "Wall-clock time. Real frontend and KV router with the session index on; workers are mockers, "
                "so there are no GPUs and the cache model is the mocker's. Demand is the router's per-session sum, "
                "which counts a prefix once per session and overstates distinct demand when sessions share one "
                "(a subagent starts from its parent's prompt)."
                + (" Requests above one worker's 640k-token KV capacity are refused by the mocker; they show as failed requests."
                   if meta.get("workload") == "agentx" and meta["num_gpu_blocks"] * meta["block_size"] <= 640000 else "")
                + (f" Sessions replay the recorded AgentX think times, capped at 300 s and divided by {meta['time_scale']}; "
                   f"the active window is {meta['active_window_s']} s." if meta.get("workload") == "agentx" else "")
            ),
            "rows": rows,
            "sessions": sessions,
        })
    default = "ax_host"
    template = (HERE / "template.html").read_text()
    out = template.replace("__DATA__", json.dumps(payload, separators=(",", ":"))).replace("__DEFAULT__", default)
    (HERE / "index.html").write_text(out)
    print(f"wrote {HERE / 'index.html'} ({len(out) // 1024} KiB)")


if __name__ == "__main__":
    main()
