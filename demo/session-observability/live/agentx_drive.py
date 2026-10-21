#!/usr/bin/env python3
"""Replay AgentX sessions against a live Dynamo frontend.

Each corpus session becomes a Dynamo session. Its requests are sent as token ids built
from the recorded `hash_ids`, so two requests that share block ids share a KV prefix
exactly as they did in the recording. Subagent groups become child sessions with
`x-dynamo-parent-session-id` set. The next request of a session is sent after the
previous reply plus its recorded think time, capped like aiperf's trace-idle-gap-cap and
divided by --time-scale. Sessions start spread over --ramp-s and no new request is sent
after --duration-s; a session's last request carries `x-dynamo-session-final`.

    python3 agentx_drive.py --subset subset.json --model Qwen/Qwen3-0.6B --duration-s 3600
"""

import argparse
import asyncio
import json
import math
import random
import time

import aiohttp

VOCAB = 150000


def token_ids(trace_salt, hash_ids, block_size):
    """block_size token ids per hash id, a pure function of (trace, hash id)."""
    out = []
    for h in hash_ids:
        base = (h * 2654435761 + trace_salt * 40503) & 0xFFFFFFFF
        out.extend((base + j * 7919 + (base >> 7) * (j + 1)) % VOCAB + 1000 for j in range(block_size))
    return out


class Replay:
    def __init__(self, args, log):
        self.args = args
        self.log = log
        self.deadline = time.time() + args.duration_s
        self.sent = 0

    def anonymous(self, salt):
        """Sessions the client sends without Dynamo session headers, to leave traffic uncovered."""
        return random.Random(salt * 7919).random() < self.args.anonymous_fraction

    def gap(self, think):
        return min(think, self.args.idle_gap_cap_s) / self.args.time_scale

    async def run_requests(self, http, sid, parent, salt, block_size, requests, subagents=()):
        pending = sorted(subagents, key=lambda g: g["t"])
        tasks = []
        for index, q in enumerate(requests):
            while pending and pending[0]["t"] <= q["t"]:
                group = pending.pop(0)
                tasks.append(asyncio.ensure_future(self.run_requests(
                    http, f"{sid}/{group['id']}", sid, salt, block_size, group["requests"])))
            last = index == len(requests) - 1 or time.time() > self.deadline
            await self.send(http, sid, parent, salt, block_size, q, last, index)
            if last:
                break
            await asyncio.sleep(self.gap(requests[index + 1]["think"]))
        for group in pending:
            if time.time() <= self.deadline:
                tasks.append(asyncio.ensure_future(self.run_requests(
                    http, f"{sid}/{group['id']}", sid, salt, block_size, group["requests"])))
        if tasks:
            await asyncio.gather(*tasks)

    async def send(self, http, sid, parent, salt, block_size, q, final, index):
        headers = {}
        if not self.anonymous(salt):
            headers["x-dynamo-session-id"] = sid
            if parent:
                headers["x-dynamo-parent-session-id"] = parent
            if final:
                headers["x-dynamo-session-final"] = "true"
        body = {
            "model": self.args.model,
            "prompt": token_ids(salt, q["hash_ids"], block_size),
            "max_tokens": max(1, min(q["out"], self.args.max_output)),
            "stream": False,
        }
        started = time.time()
        row = {"t": started, "session": sid, "parent": parent, "turn": index,
               "recorded_in": q["in"], "recorded_out": q["out"]}
        try:
            async with http.post(f"{self.args.url}/v1/completions", json=body, headers=headers) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200:
                    row["error_body"] = str(data)[:300]
                usage = data.get("usage") or {}
                row.update(status=resp.status, latency_s=time.time() - started,
                           prompt_tokens=usage.get("prompt_tokens"),
                           completion_tokens=usage.get("completion_tokens"))
        except Exception as error:
            row["error"] = str(error)
        self.log.write(json.dumps(row) + "\n")
        self.log.flush()
        self.sent += 1


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--sessions", type=int, default=40)
    parser.add_argument("--duration-s", type=float, default=3600.0)
    parser.add_argument("--ramp-s", type=float, default=900.0)
    parser.add_argument("--time-scale", type=float, default=6.0)
    parser.add_argument("--idle-gap-cap-s", type=float, default=300.0)
    parser.add_argument("--anonymous-fraction", type=float, default=0.0,
                        help="share of sessions sent without session headers")
    parser.add_argument("--max-output", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="requests.jsonl")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    traces = json.load(open(args.subset))[: args.sessions]

    timeout = aiohttp.ClientTimeout(total=1800)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as http:
        with open(args.out, "w") as log:
            replay = Replay(args, log)
            tasks = []
            for index, trace in enumerate(traces):
                delay = args.ramp_s * rng.random()

                async def launch(trace=trace, index=index, delay=delay):
                    await asyncio.sleep(delay)
                    await replay.run_requests(
                        http, f"ax-{index:03d}", None, index + 1, trace["block_size"],
                        trace["main"], trace["subagents"])

                tasks.append(asyncio.ensure_future(launch()))
            await asyncio.gather(*tasks)
    print(f"sent {replay.sent} requests")


if __name__ == "__main__":
    asyncio.run(main())
