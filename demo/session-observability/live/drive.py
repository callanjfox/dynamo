#!/usr/bin/env python3
"""Drive a synthetic agent workload through a live Dynamo frontend.

Each session is a chain of chat completions that grows its prompt by a tool result
every turn, pauses for tool time between turns, and sometimes spawns subagents that
start from the parent's prompt. Requests carry the Dynamo session headers.

    python3 drive.py --url http://localhost:8000 --model Qwen/Qwen3-0.6B --sessions 24
"""

import argparse
import asyncio
import json
import math
import random
import time

import aiohttp

WORDS = (
    "agent task file function module test build error fix patch review commit branch "
    "merge deploy cache token router worker queue request response stream batch kernel "
    "memory buffer index search match record update delete create read write parse "
    "format render compile link load save open close start stop run wait retry cancel "
    "config option value key field name type class method object array string number "
    "list map set graph tree node edge path root leaf block chain hash seed state "
    "event timer lock thread process signal pipe socket port host client server proxy "
    "session user group role policy limit quota budget metric log trace span alert"
).split()


def words(rng, count):
    return " ".join(rng.choice(WORDS) for _ in range(count))


def tool_gap_s(rng, scale):
    # Mostly quick tool calls, a tail of slow ones where a person or a long job is waited on.
    if rng.random() < 0.08:
        return scale * rng.uniform(15.0, 40.0)
    return scale * min(12.0, max(0.3, 2.0 * math.exp(0.9 * rng.gauss(0, 1))))


async def session(http, args, rng, log, sid, parent, prompt, turns):
    text = prompt
    spawned = 0
    for turn in range(turns):
        text += " " + words(rng, max(4, int(rng.lognormvariate(math.log(args.growth_words), 0.7))))
        headers = {"x-dynamo-session-id": sid}
        if parent:
            headers["x-dynamo-parent-session-id"] = parent
        if turn == turns - 1:
            headers["x-dynamo-session-final"] = "true"
        body = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": args.system},
                {"role": "user", "content": text},
            ],
            "max_tokens": rng.randint(16, 48),
            "stream": False,
        }
        started = time.time()
        try:
            async with http.post(f"{args.url}/v1/chat/completions", json=body, headers=headers) as resp:
                data = await resp.json()
                usage = data.get("usage") or {}
                log.write(json.dumps({
                    "t": started, "session": sid, "parent": parent, "turn": turn,
                    "status": resp.status, "latency_s": time.time() - started,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                }) + "\n")
        except Exception as error:  # keep the run going; the log records the failure
            log.write(json.dumps({"t": started, "session": sid, "turn": turn, "error": str(error)}) + "\n")
        if not parent and turn >= 2 and turn + 1 < turns and spawned < 3 and rng.random() < args.spawn_prob:
            child = f"{sid}/sub{spawned}"
            spawned += 1
            asyncio.ensure_future(
                session(http, args, random.Random(rng.random()), log, child, sid, text, rng.randint(3, 8))
            )
        await asyncio.sleep(tool_gap_s(rng, args.gap_scale))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--sessions", type=int, default=24)
    parser.add_argument("--ramp-s", type=float, default=120.0)
    parser.add_argument("--min-turns", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=24)
    parser.add_argument("--growth-words", type=int, default=250)
    parser.add_argument("--system-words", type=int, default=400)
    parser.add_argument("--gap-scale", type=float, default=1.0)
    parser.add_argument("--spawn-prob", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="requests.jsonl")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    args.system = "You are a coding agent. " + words(random.Random(0), args.system_words)

    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as http:
        with open(args.out, "w") as log:
            tasks = []
            for index in range(args.sessions):
                delay = args.ramp_s * math.sqrt(rng.random())
                task_rng = random.Random(rng.random())
                prompt = f"Task {index}: " + words(task_rng, 150)

                async def launch(delay=delay, index=index, task_rng=task_rng, prompt=prompt):
                    await asyncio.sleep(delay)
                    await session(http, args, task_rng, log, f"task-{index:03d}", None, prompt,
                                  task_rng.randint(args.min_turns, args.max_turns))

                tasks.append(asyncio.ensure_future(launch()))
            await asyncio.gather(*tasks)
            # Let spawned subagents finish.
            pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if pending:
                await asyncio.wait(pending)


if __name__ == "__main__":
    asyncio.run(main())
