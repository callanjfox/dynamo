#!/usr/bin/env python3
"""Background chat traffic: single-turn requests with no session headers.

Stands in for the part of a fleet's work that is not agentic. Requests arrive as a Poisson
process at --rps. Each prompt is a few thousand tokens at most (lognormal around --prompt-median),
and a share of them start with one of a few common system prompts, so some prefix is shared across
requests the way a chat product's system prompt is. Prompts are sent as token ids, like the AgentX
driver, and carry no x-dynamo-session-* headers, so the router sees them as uncovered traffic.

    python3 chat_load.py --url http://localhost:8000 --rps 0.5 --duration-s 3600 --out chat.jsonl
"""

import argparse
import asyncio
import json
import math
import random
import time

import aiohttp

VOCAB = 150000


def tokens(rng, count):
    return [rng.randrange(1000, VOCAB) for _ in range(count)]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--rps", type=float, default=0.5, help="mean requests per second")
    parser.add_argument("--duration-s", type=float, default=3600.0)
    parser.add_argument("--prompt-median", type=int, default=1500)
    parser.add_argument("--prompt-sigma", type=float, default=0.8, help="lognormal sigma of the prompt length")
    parser.add_argument("--max-prompt", type=int, default=16000)
    parser.add_argument("--system-prompts", type=int, default=3, help="distinct common system prompts")
    parser.add_argument("--system-tokens", type=int, default=600)
    parser.add_argument("--system-share", type=float, default=0.6, help="share of requests that start with one")
    parser.add_argument("--max-output", type=int, default=256)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", default="chat.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    systems = [tokens(random.Random(1000 + i), args.system_tokens) for i in range(args.system_prompts)]
    deadline = time.time() + args.duration_s
    sent = 0

    async def one(http, log, request_id):
        length = int(min(args.max_prompt, max(50, rng.lognormvariate(math.log(args.prompt_median), args.prompt_sigma))))
        prompt = []
        if rng.random() < args.system_share:
            prompt = list(rng.choice(systems))
        prompt += tokens(rng, max(1, length - len(prompt)))
        body = {"model": args.model, "prompt": prompt, "max_tokens": rng.randint(32, args.max_output), "stream": False}
        started = time.time()
        row = {"t": started, "chat": request_id, "prompt_tokens_sent": len(prompt)}
        try:
            async with http.post(f"{args.url}/v1/completions", json=body) as resp:
                data = await resp.json(content_type=None)
                usage = data.get("usage") or {}
                row.update(status=resp.status, latency_s=time.time() - started,
                           prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"))
                if resp.status != 200:
                    row["error_body"] = str(data)[:300]
        except Exception as error:
            row["error"] = str(error)
        log.write(json.dumps(row) + "\n")
        log.flush()

    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as http:
        with open(args.out, "w") as log:
            tasks = []
            while time.time() < deadline:
                tasks.append(asyncio.ensure_future(one(http, log, sent)))
                sent += 1
                await asyncio.sleep(rng.expovariate(args.rps))
            await asyncio.gather(*tasks)
    print(f"sent {sent} chat requests")


if __name__ == "__main__":
    asyncio.run(main())
