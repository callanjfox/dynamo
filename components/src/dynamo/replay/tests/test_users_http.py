# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The users-over-HTTP CLI builds the driver's config from its flags."""

import json

import pytest

from dynamo.replay.users_http import build_config, parse_args

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
]


def test_config_carries_flags_and_records(tmp_path):
    policy = tmp_path / "policy.yaml"
    policy.write_text("default_policy_class: standard\n")
    args = parse_args(
        [
            "--users-dir",
            "/exports/m150",
            "--model",
            "m",
            "--out",
            "/runs/today.jsonl",
            "--speedup",
            "4",
            "--stop-s",
            "2700",
            "--arm",
            "today",
            "--policy-file",
            str(policy),
            "--record",
            'engine={"speedup_ratio": 4}',
        ]
    )

    config = build_config(args)

    assert config["users_dir"] == "/exports/m150"
    assert config["speedup"] == 4.0
    assert config["stop_s"] == 2700.0
    assert config["vocab_size"] == 32000
    assert config["record"]["arm"] == "today"
    assert config["record"]["policy"] == {"default_policy_class": "standard"}
    assert config["record"]["engine"] == {"speedup_ratio": 4}
    json.dumps(config)


def test_the_binding_rejects_a_bad_config():
    from dynamo._core import run_users_http_replay

    with pytest.raises(Exception):
        run_users_http_replay(json.dumps({"url": "http://127.0.0.1:1"}))
