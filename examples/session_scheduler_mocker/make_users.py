#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write a synthetic closed-loop users export for the session scheduler mocker test.

The export has the users exporter's format (the one AISimulate's users driver and
`python -m dynamo.replay.users_http` read): `templates.jsonl` (agent plays), `schedule.json` (which plays each
user runs, in order, and its class), `request_fields.users.jsonl` (the session, class and hint headers of each
request) and `classes.json` (each user copy's class, for scoring).

Every number is drawn from a seeded generator, so the same arguments give the same export. The shapes follow
long-context coding agents: a main session of many turns whose context grows turn by turn (median about 40k
tokens at the start and about 1.5k new tokens a turn, up to 900k), outputs of a few hundred tokens with a long
tail, short think times between turns, and now and then a fan-out of subagents that share the main session's
system prompt and join back. Users join over the first 45 minutes and play one plan after another until the
horizon. Classes: 30% priority, 50% standard, 20% batch.

    python3 make_users.py OUT_DIR [--users 832] [--templates 300] [--hours 6] [--seed 31]
"""

import argparse
import json
import math
import os
import random

BLOCK = 64
MAX_INPUT_TOKENS = 900_000
SYSTEM_BLOCKS = 20  # the system prompt a subagent shares with its main session


def lognormal(rng, median, sigma, low, high):
    return min(high, max(low, rng.lognormvariate(math.log(median), sigma)))


class Play:
    """One play template: nodes in the users templates' row format."""

    def __init__(self, index, rng):
        self.index = index
        self.rng = rng
        self.nodes = []  # [request_id, session_id, input, output, not_before_ms, deps, base, prefix, new_ranks]
        self.fields = []  # request fields rows
        self.ranks = []  # total prompt ranks per node
        self.next_rank = 0

    def new_ranks(self, count):
        ranks = list(range(self.next_rank, self.next_rank + count))
        self.next_rank += count
        return ranks

    def add(self, session, parent_session, deps, base, prefix, new_blocks, output, fanout=None):
        node = len(self.nodes)
        request_id = f"syn:t{self.index:05d}:r{node:05d}"
        total = prefix + new_blocks
        self.nodes.append([request_id, session, total * BLOCK, int(output), 0.0, deps,
                           base, prefix, self.new_ranks(new_blocks)])
        self.ranks.append(total)
        headers = {"x-dynamo-hint-osl": [f"{float(int(output))}"]}
        if fanout is not None:
            headers["x-dynamo-hint-fanout"] = [str(fanout)]
        self.fields.append({"request_id": request_id, "session_id": session,
                            "parent_session_id": parent_session, "session_final": False,
                            "agent_headers": headers})
        return node

    def session(self, name, parent, first_deps, base, prefix, root_blocks, turns, spawn):
        """A chain of turns; returns the session's node ids. `spawn(node)` may add subagents after a turn
        and returns the nodes the next turn joins on."""
        rng = self.rng
        nodes = []
        deps, cur_base, cur_prefix, new_blocks = first_deps, base, prefix, root_blocks
        for turn in range(turns):
            if (cur_prefix + new_blocks) * BLOCK > MAX_INPUT_TOKENS:
                break
            output = lognormal(rng, 450, 1.1, 16, 16_000)
            node = self.add(name, parent, deps, cur_base, cur_prefix, new_blocks, output)
            nodes.append(node)
            joins = spawn(node) if spawn else []
            think_ms = 0.0 if rng.random() < 0.7 else 1000.0 * lognormal(rng, 3.0, 1.5, 0.1, 600.0)
            deps = [[node, 1, round(think_ms, 1), 0]] + [[j, 1, 0.0, 2] for j in joins]
            cur_base, cur_prefix = node, self.ranks[node]
            new_blocks = max(1, round(lognormal(rng, 1500, 1.3, 64, 60_000) / BLOCK))
            # The client's hints about its next turn.
            hdr = self.fields[node]["agent_headers"]
            hdr["x-dynamo-hint-next-think-s"] = [f"{think_ms / 1000.0:.2f}"]
            hdr["x-dynamo-hint-next-tokens"] = [f"{float(new_blocks * BLOCK)}"]
        if nodes:
            self.fields[nodes[-1]]["session_final"] = True
        return nodes

    def build(self):
        rng = self.rng
        main = f"syn:t{self.index:05d}:s:m"
        subagents = [0]

        def spawn(node):
            if rng.random() >= 0.08:
                return []
            fanout = min(6, 1 + int(rng.expovariate(0.8)))
            last = []
            for _ in range(fanout):
                subagents[0] += 1
                name = f"syn:t{self.index:05d}:s:a{subagents[0]}"
                prefix = min(SYSTEM_BLOCKS, self.ranks[node])
                turns = int(lognormal(rng, 8, 0.8, 1, 60))
                root = max(1, round(lognormal(rng, 8000, 0.8, 512, 60_000) / BLOCK))
                nodes = self.session(name, main, [[node, 1, 0.0, 1]], node, prefix, root, turns, None)
                for n in nodes:
                    self.fields[n]["agent_headers"]["x-dynamo-hint-fanout"] = [str(fanout)]
                if nodes:
                    last.append(nodes[-1])
            return last

        root = max(1, round(lognormal(rng, 40_000, 1.0, 2_000, 300_000) / BLOCK))
        turns = int(lognormal(rng, 30, 1.0, 2, 400))
        self.session(main, None, [], -1, 0, root, turns, spawn)
        return {"i": self.index, "blocks": self.next_rank, "nodes": self.nodes}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--users", type=int, default=832)
    ap.add_argument("--templates", type=int, default=300)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--ramp-min", type=float, default=45.0, help="users join over this many minutes")
    ap.add_argument("--entries", type=int, default=40, help="plays scheduled per user (more than a horizon needs)")
    ap.add_argument("--seed", type=int, default=31)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)

    fields = []
    with open(os.path.join(a.out, "templates.jsonl"), "w") as f:
        f.write(json.dumps({"schema": "dynosim.users.templates", "version": 1, "block_size": BLOCK,
                            "templates": a.templates, "source": f"synthetic seed {a.seed}"}) + "\n")
        for i in range(a.templates):
            play = Play(i, random.Random(rng.getrandbits(64)))
            f.write(json.dumps(play.build(), separators=(",", ":")) + "\n")
            fields.extend(play.fields)

    classes = ["priority"] * 30 + ["standard"] * 50 + ["batch"] * 20
    plan, copy_classes = [], {}
    for user in range(a.users):
        cls = rng.choice(classes)
        entries = [{"tmpl": rng.randrange(a.templates)} for _ in range(a.entries)]
        entries[0]["at_s"] = round(rng.uniform(0.0, a.ramp_min * 60.0), 3)
        plan.append({"cls": cls, "entries": entries})
        fields.append({"request_id": f"cp{user:06d}", "policy_class": cls,
                       "agent_headers": {"x-dynamo-meta-policy-class": [cls], "x-dynamo-user-id": [f"u{user:06d}"]}})
        for position in range(a.entries):
            copy_classes[f"{user:06d}{position:06d}"] = cls
    schedule = {"schema": "dynosim.users.schedule.timed", "version": 2, "timed": True, "seed": a.seed,
                "users": a.users, "hours": a.hours, "horizon_s": a.hours * 3600.0, "warm": False,
                "cooldown_s": 0.0, "loss_cooldown_s": 120.0, "templates": "templates.jsonl", "plan": plan}
    with open(os.path.join(a.out, "schedule.json"), "w") as f:
        json.dump(schedule, f)
    with open(os.path.join(a.out, "request_fields.users.jsonl"), "w") as f:
        for row in fields:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    with open(os.path.join(a.out, "classes.json"), "w") as f:
        json.dump({"mix": "synthetic 30/50/20", "seed": a.seed, "key": "user then position, six digits each",
                   "classes": copy_classes}, f)
    requests = sum(1 for r in fields if not r["request_id"].startswith("cp"))
    print(f"{a.templates} templates, {requests} template requests, {a.users} users -> {a.out}")


if __name__ == "__main__":
    main()
