#!/usr/bin/env python3
"""Build the datasets for the customer performance diagnosis dashboard.

Reads a live run from data/live/<scenario> (the same files the main demo page uses) and writes JSON
row files under diag/data/: a timeline, the session list, where returning sessions found their
prefix, a comparison of runs, and routing decisions. Everything the page shows is read from these
files; the page does no arithmetic on raw counters.

    python3 build_diag.py --scenario ax_scen --compare ax_gpu ax_host ax_scen
"""

import argparse
import json
import pathlib

import build_demo as bd

HERE = pathlib.Path(__file__).parent
OUT = HERE / "diag" / "data"
ROUND = 3


def r(x, n=ROUND):
    return None if x is None else round(x, n)


def window_start(rows, i, ms):
    j = i
    while j > 0 and rows[i]["now_ms"] - rows[j]["now_ms"] >= 0 and rows[i]["now_ms"] - rows[j]["now_ms"] < ms:
        j -= 1
    return j


def rate(rows, i, ms, get):
    j = window_start(rows, i, ms)
    dt = (rows[i]["now_ms"] - rows[j]["now_ms"]) / 1000
    return max(0.0, (get(rows[i]) - get(rows[j])) / dt) if dt > 0 else 0.0


def timeline(rows, meta, adjust=False):
    out = []
    has_compute = any(x["compute"] for x in rows)
    cap_dev, cap_host = rows[0]["capacity_device_tokens"], rows[0]["capacity_host_tokens"]
    for i, x in enumerate(rows):
        c = x["compute"] or {}
        j = window_start(rows, i, 60000)
        f0d = x["funnel"]["f0"] - rows[j]["funnel"]["f0"]
        busy = f0d > 2e6
        share = lambda key: r(100 * (x["funnel"][key] - rows[j]["funnel"][key]) / f0d, 2) if busy else None
        # A router from before the fix counted the previous turn's output in the prior prefix; take it back out.
        out_back = (x["covered_output_tokens"] - rows[j]["covered_output_tokens"]) if adjust else 0
        pot = r(max(0, min(100, 100 * (x["prior_prefix"] - rows[j]["prior_prefix"] - out_back) / f0d)), 2) if busy and x.get("prior_prefix") else None
        out_total = rate(rows, i, 60000, lambda y: y["output_seq"]["sum"])
        out_agent = rate(rows, i, 60000, lambda y: y["covered_output_tokens"])
        in_total = rate(rows, i, 60000, lambda y: y["input_seq"]["sum"])
        in_agent = rate(rows, i, 60000, lambda y: y["coverage"]["tokens_covered"])
        in_chat = rate(rows, i, 60000, lambda y: y["coverage"]["tokens_uncovered"])
        req_agent = rate(rows, i, 60000, lambda y: y["coverage"]["requests_covered"])
        req_chat = rate(rows, i, 60000, lambda y: y["coverage"]["requests_uncovered"])
        thr = lambda y: y["thrash"]["recomputed"]
        thr_l = lambda y: y["thrash"]["reloaded_host"] + y["thrash"]["reloaded_disk"]
        w = x.get("worker") or {}
        host_only = max(0, x["distinct_host_tokens"] - (x.get("tier_overlap_tokens") or 0))
        workers = meta["workers"]
        left_total = x.get("left_total_device") if x.get("left_total_device") is not None else (w.get("left_mean_device") * workers if w.get("left_mean_device") is not None else None)
        windows = x.get("demand_windows") or [None, None, None]
        out.append({
            "t": r(x["now_ms"] / 60000, 3),
            "in_tps": r(in_total, 0), "out_tps": r(out_total, 1),
            "agent_in_tps": r(in_agent, 0), "chat_in_tps": r(in_chat, 0),
            "agent_out_tps": r(min(out_agent, out_total), 1) if x.get("covered_output_tokens") else None, "chat_out_tps": r(max(0.0, out_total - out_agent), 1) if x.get("covered_output_tokens") else None,
            "agent_req_ps": r(req_agent, 2), "chat_req_ps": r(req_chat, 2),
            "ttft_p50": r((c.get("ttft") or {}).get("p50"), 4), "ttft_p90": r((c.get("ttft") or {}).get("p90"), 4),
            "itl_p50_ms": r(((c.get("itl") or {}).get("p50") or 0) * 1000 if (c.get("itl") or {}).get("p50") is not None else None, 2),
            "itl_p90_ms": r(((c.get("itl") or {}).get("p90") or 0) * 1000 if (c.get("itl") or {}).get("p90") is not None else None, 2),
            "queued": c.get("queued"), "prefilling": c.get("prefilling"), "decoding": c.get("decoding"),
            "backlog_tokens": c.get("prefill_tokens"),
            "pot_pct": pot, "f2_pct": share("f2"), "f3_pct": share("f3"), "f4_pct": share("f4") if x["funnel"]["f4"] else None,
            "demand": x["demand_distinct_tokens"], "demand_sum": x["demand_tokens"],
            "demand_primary": x.get("demand_primary_tokens"), "demand_subagent": x.get("demand_subagent_tokens"),
            "w1": windows[0], "w2": windows[1], "w3": windows[2],
            "active_primary": x.get("active_primary"), "active_subagent": x.get("active_subagent"),
            "gpu_resident": x["distinct_device_tokens"], "host_resident": x["distinct_host_tokens"],
            "host_only": host_only, "overlap": x.get("tier_overlap_tokens"),
            "stranded_gpu": x["stranded_device_tokens"], "stranded_host": x.get("stranded_host_tokens"),
            "dup_gpu": r(1 + x["replicated_device_tokens"] / x["distinct_device_tokens"], 3) if x["distinct_device_tokens"] else 1,
            "dup_host": r(1 + x["replicated_host_tokens"] / x["distinct_host_tokens"], 3) if x["distinct_host_tokens"] else 1,
            "shared": x.get("shared_tokens"),
            "thrash_recomputed_pm": r(rate(rows, i, 60000, thr) * 60, 0), "thrash_reloaded_pm": r(rate(rows, i, 60000, thr_l) * 60, 0),
            "wd_max": w.get("demand_max"), "wd_min": w.get("demand_min"), "wd_mean": w.get("demand_mean"),
            "wr_gpu_max": w.get("resident_max_device"), "wr_gpu_min": w.get("resident_min_device"), "wr_gpu_mean": r(w.get("resident_mean_device"), 0) if w.get("resident_mean_device") is not None else None,
            "wr_host_max": w.get("resident_max_host"), "wr_host_min": w.get("resident_min_host"), "wr_host_mean": r(w.get("resident_mean_host"), 0) if w.get("resident_mean_host") is not None else None,
            "wl_gpu_max": w.get("left_max_device"), "wl_gpu_mean": w.get("left_mean_device"), "wl_host_max": w.get("left_max_host"),
            "imb_demand": x.get("demand_imbalance"), "imb_res_gpu": x.get("resident_imbalance_device"), "imb_res_host": x.get("resident_imbalance_host"),
            "left_gpu_total": left_total, "needed_gpu": max(0, x["distinct_device_tokens"] - (left_total or 0) - x["stranded_device_tokens"]) if left_total is not None else None,
        })
    return out


def run_summary(key, label, directory, adjust=False):
    rows, sessions, meta, stats = bd.live_run(directory)
    last = rows[-1]
    f0 = last["funnel"]["f0"] or 1
    ttft90 = sorted(x["compute"]["ttft"]["p90"] for x in rows if x["compute"] and x["compute"].get("ttft") and x["compute"]["ttft"]["p90"] is not None)
    cap_dev, cap_host = rows[0]["capacity_device_tokens"], rows[0]["capacity_host_tokens"]
    peak = max(x["demand_distinct_tokens"] for x in rows)
    th = last["thrash"]
    return {
        "key": key, "label": label, "workers": meta["workers"],
        "gpu_tokens": cap_dev, "host_tokens": cap_host,
        "sessions": len(sessions), "requests": stats["sent"], "failed": stats["failed"],
        "input_tokens": f0, "f2_pct": r(100 * last["funnel"]["f2"] / f0, 1), "f3_pct": r(100 * last["funnel"]["f3"] / f0, 1),
        "pot_pct": r(max(0, min(100, 100 * ((last.get("prior_prefix") or 0) - (last["covered_output_tokens"] if adjust else 0)) / f0)), 1) if last.get("prior_prefix") else None,
        "recomputed": th["recomputed"], "reloaded": th["reloaded_host"] + th["reloaded_disk"],
        "peak_demand_ratio": r(peak / cap_dev, 2),
        "ttft_p90_median": r(ttft90[len(ttft90) // 2], 3) if ttft90 else None,
        "ttft_p90_worst": r(ttft90[-1], 3) if ttft90 else None,
    }


SCENARIOS = {
    "busy": ("Busy, plenty of cache", "Two workers, each with room for the whole working set in GPU memory. Compute is the limit, so the answer is more compute."),
    "short": ("Not enough cache", "Four workers whose GPU memory holds about half the working set and no host tier. Compute looks contended because returning sessions recompute their prefix."),
    "mirror": ("Cache not used well", "GPU and host memory of equal size, with host memory mostly mirroring the GPU, so the extra memory adds little room."),
    "route": ("Routing ignores the cache", "Ample cache, but the router is told to ignore cache overlap. Sessions scatter across workers and copies multiply."),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=["busy=ax_host"],
                        help="scenario=run directory under data/live, one per scenario")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    metas, tl_all, ses_all, ret_all, routing_all = [], [], [], [], []
    labels = ["up to 1 min", "1 to 5 min", "5 to 15 min", "over 15 min"]
    for spec in args.scenarios:
        key, run = spec.split("=")
        adjust = run.endswith(":adjust")
        run = run.split(":")[0]
        label, blurb = SCENARIOS.get(key, (key, ""))
        directory = HERE / "data" / "live" / run
        rows, sessions, meta, stats = bd.live_run(directory)
        cap_dev, cap_host = rows[0]["capacity_device_tokens"], rows[0]["capacity_host_tokens"]
        workers = meta["workers"]
        summary = run_summary(key, label, directory, adjust)
        metas.append({
            "scenario": key, "label": label, "blurb": blurb, "model": meta.get("model", "Qwen/Qwen3-0.6B"),
            "workers": workers, "block_size": meta["block_size"],
            "cap_gpu": cap_dev, "cap_host": cap_host, "cap_all": cap_dev + cap_host,
            "per_worker_gpu": cap_dev // workers, "per_worker_all": (cap_dev + cap_host) // workers,
            "window_labels": meta.get("window_labels", ["30s", "1m", "5m"]),
            "time_scale": meta.get("time_scale", 1), "speedup": meta.get("speedup", 1),
            "requests": stats["sent"], "failed": stats["failed"],
            "chat_requests": (stats.get("chat") or {}).get("sent", 0), "chat_prompt_p50": (stats.get("chat") or {}).get("prompt_p50", 0),
            "has_overlap": any((x.get("tier_overlap_tokens") or 0) > 0 for x in rows),
            "has_chat": bool(stats.get("chat")),
            "f2_pct": summary["f2_pct"], "f3_pct": summary["f3_pct"], "pot_pct": summary["pot_pct"],
            "recomputed": summary["recomputed"], "reloaded": summary["reloaded"], "input_tokens": summary["input_tokens"],
            "peak_demand_ratio": summary["peak_demand_ratio"], "ttft_p90_median": summary["ttft_p90_median"], "ttft_p90_worst": summary["ttft_p90_worst"],
        })
        for row in timeline(rows, meta, adjust):
            tl_all.append({"scenario": key, **row})
        for s_ in sessions:
            ses_all.append({
                "scenario": key, "role": "subagent" if s_.get("parent_session_id") else "primary",
                "turns": s_["turns"], "isl_tokens": s_["isl_tokens"], "osl_tokens": s_["osl_tokens"],
                "isl_per_turn": r(s_["isl_tokens"] / max(1, s_["turns"]), 0), "osl_per_turn": r(s_["osl_tokens"] / max(1, s_["turns"]), 1),
                "peak_context": s_["peak_demand_tokens"], "hit_rate": r(s_["hit_rate"], 3),
                "duration_min": r(s_["duration_ms"] / 60000, 2), "mean_gap_s": r(s_["mean_turn_gap_ms"] / 1000, 2),
                "subagents": s_["children"], "growth_per_turn": r(s_.get("growth_per_turn") or 0, 0),
            })
        for k, g in enumerate(rows[-1]["gap_outcomes"]):
            ret_all.append({"scenario": key, "gap": labels[k], "gpu": g[0], "host": g[1] + g[2], "miss": g[3], "turns": sum(g)})
        sel_path = directory / "selection.jsonl"
        if sel_path.exists():
            t0_ms = bd.read_jsonl(directory / "metrics.jsonl")[0]["t"] * 1000
            for x in bd.read_jsonl(sel_path):
                if "differ" in x:
                    routing_all.append({"scenario": key, **x, "t": r((x["t_ms"] - t0_ms) / 60000, 2)})

    (OUT / "run_meta.json").write_text(json.dumps(metas))
    (OUT / "timeline.json").write_text(json.dumps(tl_all, separators=(",", ":")))
    (OUT / "sessions.json").write_text(json.dumps(ses_all, separators=(",", ":")))
    (OUT / "returns.json").write_text(json.dumps(ret_all))
    (OUT / "routing.json").write_text(json.dumps(routing_all, separators=(",", ":")))
    for p in sorted(OUT.glob("*.json")):
        print(p.name, p.stat().st_size)


if __name__ == "__main__":
    main()
