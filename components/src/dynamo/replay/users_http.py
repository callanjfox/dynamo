# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a closed-loop users export against a live Dynamo frontend.

The users schedule (``schedule.json``, its templates and ``request_fields.users.jsonl`` from the
users exporter) is played by AISimulate's workload driver, the same driver offline replay uses.
Each turn is sent to the frontend as a streamed chat completion with its prompt as token ids
(``nvext.token_data``) and the session, class and hint headers a production client sends. A
user's next turn waits for the previous one. HTTP 429 is retried after its ``Retry-After``.

Outputs (beside ``--out``): ``<arm>.jsonl`` (one record per request), ``<arm>.refusals.jsonl``,
``<arm>.options.json`` and ``<arm>.summary.json``, in offline replay's formats.

Example::

    python -m dynamo.replay.users_http --users-dir exports/m150-w4-s31 \\
        --url http://127.0.0.1:8000 --model deepseek-ai/DeepSeek-V4-Pro \\
        --speedup 4 --stop-s 2700 --out runs/today.jsonl --arm today

Run the mocker workers with the same ``speedup_ratio`` and the frontend with
``DYNAMO_SESSION_SCHEDULER_CLOCK_SCALE`` set to the same value.
"""

import argparse
import json
import sys

from dynamo._core import run_users_http_replay
from dynamo.runtime.logging import configure_dynamo_logging


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m dynamo.replay.users_http", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--users-dir", required=True, help="users export directory")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="frontend URL")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--out", required=True, help="per-request record file (.jsonl)")
    parser.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="simulated seconds per wall second (the mocker's speedup_ratio)",
    )
    parser.add_argument(
        "--stop-s", type=float, default=None, help="stop at this simulated time (s)"
    )
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="token ids sent stay below this (the served tokenizer's vocabulary)",
    )
    parser.add_argument("--ready-timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--arm", default=None, help="arm name, recorded in <arm>.options.json"
    )
    parser.add_argument(
        "--policy-file",
        default=None,
        help="router policy file the frontend runs, recorded in <arm>.options.json",
    )
    parser.add_argument(
        "--record",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="extra settings to record in <arm>.options.json (repeatable)",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> dict:
    record: dict = {}
    if args.arm:
        record["arm"] = args.arm
    if args.policy_file:
        record["policy_file"] = args.policy_file
        try:
            import yaml

            with open(args.policy_file, encoding="utf-8") as f:
                record["policy"] = yaml.safe_load(f)
        except (ImportError, OSError) as error:
            record["policy_error"] = str(error)
    for item in args.record:
        key, _, value = item.partition("=")
        try:
            record[key] = json.loads(value)
        except json.JSONDecodeError:
            record[key] = value
    return {
        "users_dir": args.users_dir,
        "url": args.url,
        "model": args.model,
        "out": args.out,
        "speedup": args.speedup,
        "stop_s": args.stop_s,
        "block_size": args.block_size,
        "vocab_size": args.vocab_size,
        "ready_timeout_s": args.ready_timeout_s,
        "record": record,
    }


def main(argv=None) -> int:
    configure_dynamo_logging()
    args = parse_args(argv)
    summary = json.loads(run_users_http_replay(json.dumps(build_config(args))))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
