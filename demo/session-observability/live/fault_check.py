#!/usr/bin/env python3
"""Compare the router's session summaries with the driver's request log after fault_run.sh.

    python3 fault_check.py /root/live_out/f_before

Without retry accounting the router counts a retried request as an extra turn, and a retried final turn
ends its session early and resumes it, so a session reports twice.
"""
import collections
import json
import subprocess
import sys

out = sys.argv[1]
here = __file__.rsplit("/", 1)[0]
reqs = [json.loads(l) for l in open(f"{out}/requests.jsonl")]
ok = [r for r in reqs if r.get("status") == 200]
failed = [r for r in reqs if r.get("status") != 200]
summaries = [json.loads(l) for l in subprocess.run(
    ["python3", f"{here}/extract_summaries.py", f"{out}/frontend.log"], capture_output=True, text=True).stdout.splitlines() if l.strip()]
per = collections.Counter(s.get("session_id") for s in summaries)
turns = sum(s["turns"] for s in summaries)
isl = sum(s["input_tokens"] for s in summaries)
by_sid = collections.defaultdict(list)
for r in reqs:
    by_sid[r["session"]].append(r)
print(f"driver: {len(reqs)} requests sent, {len(ok)} succeeded, {len(failed)} failed; input tokens of successes {sum(r['prompt_tokens'] for r in ok)}")
print(f"router: {len(summaries)} session summaries over {len(per)} sessions, {turns} turns, {isl} input tokens")
print(f"sessions with more than one summary: {sum(1 for v in per.values() if v > 1)}")
extra = {s: sum(x['turns'] for x in summaries if x.get('session_id') == s) - len(by_sid[s]) for s in per if s in by_sid}
print("turns minus driver requests per session: nonzero for", {s: d for s, d in extra.items() if d}.__len__(), "sessions; total", sum(extra.values()))

# Coverage counters, when the run saved the frontend metrics at its end.
import os
end = f"{out}/metrics_end.txt"
if os.path.exists(end):
    def total(name, label):
        t = 0.0
        for line in open(end):
            if line.startswith("dynamo_component_router_" + name + "{") and label in line:
                t += float(line.rsplit(" ", 1)[1])
        return t
    cov_r, unc_r = total("session_coverage_requests_total", 'coverage="covered"'), total("session_coverage_requests_total", 'coverage="uncovered"')
    cov_t, unc_t = total("session_coverage_input_tokens_total", 'coverage="covered"'), total("session_coverage_input_tokens_total", 'coverage="uncovered"')
    prior = total("session_prior_prefix_tokens_total", "")
    print(f"coverage counters: {cov_r:.0f} covered + {unc_r:.0f} uncovered = {cov_r + unc_r:.0f} requests (driver sent {len(reqs)}); input {cov_t + unc_t:.0f} (driver successes {sum(r.get('prompt_tokens') or 0 for r in ok)}); prior prefix {prior:.0f}")
    # What the prior prefix should be: the smaller of each request's and its predecessor's input, over turns
    # that did not start while the session's previous request was still running.
    exp = exp_all = 0
    for rows in by_sid.values():
        rows = sorted((r for r in rows if r.get("end")), key=lambda r: r["t"])
        for a_, b_ in zip(rows, rows[1:]):
            p = min(a_["prompt_tokens"], b_["prompt_tokens"])
            exp_all += p
            if b_["t"] >= a_["end"]:
                exp += p
    print(f"expected prior prefix: {exp} excluding overlapping pairs ({exp_all} if every turn counted)")
