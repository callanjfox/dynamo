// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replays a synthetic agent workload through the mocker with the session prefix
//! index on, so the session stats recorder can write a working-set time series.
//!
//! ```text
//! DYN_SESSION_STATS_JSONL=out/gpu_only.jsonl \
//!   cargo run --release -p dynamo-mocker --example session_observability_demo -- --host-blocks 0
//! ```
//!
//! The workload is shaped like coding-agent traffic: every session starts from one
//! shared system and tool prefix, grows its context each turn, pauses for tool time
//! between turns, and some sessions fan out subagents that start from the parent's
//! prefix. Subagents are named `parent/subN`, which the recorder reads as the parent
//! link. Nothing is drawn from real traces.

use dynamo_kv_router::config::KvRouterConfig;
use dynamo_mocker::common::protocols::MockerConfig;
use dynamo_mocker::loadgen::{SessionTrace, Trace, TurnTrace};
use dynamo_mocker::replay::{
    ReplayRouterMode, SlaThresholds, simulate_loaded_trace_with_router_mode_and_options,
};

const BLOCK_SIZE: usize = 16;
const SHARED_PREFIX_BLOCKS: usize = 200;
const UNIQUE_ID_BASE: u32 = 1_000_000;
/// Sizes a host block; a mid-size GQA model in bf16.
const KV_BYTES_PER_TOKEN: u64 = 131_072;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, low: usize, high: usize) -> usize {
        low + (self.next() as usize) % (high - low + 1)
    }

    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit().max(1e-12), self.unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn lognormal(&mut self, median: f64, sigma: f64) -> f64 {
        median * (sigma * self.normal()).exp()
    }
}

struct Args {
    sessions: usize,
    workers: usize,
    gpu_blocks: usize,
    host_blocks: usize,
    window_s: f64,
    seed: u64,
}

fn parse_args() -> Args {
    let mut args = Args {
        sessions: 120,
        workers: 4,
        gpu_blocks: 6_000,
        host_blocks: 0,
        window_s: 1_800.0,
        seed: 7,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(flag) = iter.next() {
        let mut value = || iter.next().expect("flag needs a value");
        match flag.as_str() {
            "--sessions" => args.sessions = value().parse().unwrap(),
            "--workers" => args.workers = value().parse().unwrap(),
            "--gpu-blocks" => args.gpu_blocks = value().parse().unwrap(),
            "--host-blocks" => args.host_blocks = value().parse().unwrap(),
            "--window-s" => args.window_s = value().parse().unwrap(),
            "--seed" => args.seed = value().parse().unwrap(),
            other => panic!("unknown flag {other}"),
        }
    }
    args
}

/// One turn: grow the prefix by the tool result and the prior output, then present it.
fn grow(
    rng: &mut Rng,
    prefix: &mut Vec<u32>,
    next_id: &mut u32,
    previous_output_blocks: usize,
) -> usize {
    let tool_blocks = rng.lognormal(12.0, 0.9).clamp(1.0, 400.0) as usize;
    for _ in 0..(tool_blocks + previous_output_blocks) {
        prefix.push(*next_id);
        *next_id += 1;
    }
    rng.range(100, 800)
}

fn tool_gap_ms(rng: &mut Rng) -> f64 {
    // Mostly quick tool calls, a tail of slow ones where a person or a long job is waited on.
    if rng.unit() < 0.08 {
        rng.range(60_000, 240_000) as f64
    } else {
        rng.lognormal(8_000.0, 0.9).clamp(1_000.0, 90_000.0)
    }
}

fn build_trace(args: &Args) -> Trace {
    let mut rng = Rng(args.seed);
    let mut next_id = UNIQUE_ID_BASE;
    let mut sessions = Vec::new();

    for index in 0..args.sessions {
        // Sessions start later more often than earlier, so load ramps up.
        let start_ms = args.window_s * 1_000.0 * rng.unit().sqrt();
        let session_id = format!("task-{index:03}");

        let mut prefix: Vec<u32> = (0..SHARED_PREFIX_BLOCKS as u32).collect();
        let task_blocks = rng.range(100, 600);
        for _ in 0..task_blocks {
            prefix.push(next_id);
            next_id += 1;
        }

        let turn_count = rng.range(6, 40);
        let mut turns = Vec::with_capacity(turn_count);
        let mut previous_output_blocks = 0;
        let mut elapsed_ms = 0.0;
        let mut children: Vec<(usize, Vec<u32>, f64)> = Vec::new();
        let spawns_children = rng.unit() < 0.35;

        for turn in 0..turn_count {
            let output_tokens = grow(&mut rng, &mut prefix, &mut next_id, previous_output_blocks);
            previous_output_blocks = output_tokens.div_ceil(BLOCK_SIZE);
            let gap = if turn == 0 {
                0.0
            } else {
                tool_gap_ms(&mut rng)
            };
            elapsed_ms += gap + 4_000.0;
            turns.push(TurnTrace {
                input_length: prefix.len() * BLOCK_SIZE,
                max_output_tokens: output_tokens,
                hash_ids: prefix.clone(),
                delay_after_previous_ms: gap,
                ..Default::default()
            });
            if spawns_children && turn >= 2 && turn + 1 < turn_count && rng.unit() < 0.25 {
                children.push((turn, prefix.clone(), start_ms + elapsed_ms));
            }
        }
        sessions.push(SessionTrace {
            session_id: session_id.clone(),
            first_arrival_timestamp_ms: Some(start_ms),
            turns,
        });

        for (child_index, (_, parent_prefix, child_start)) in children.into_iter().enumerate() {
            let mut prefix = parent_prefix;
            let child_turns = rng.range(3, 10);
            let mut turns = Vec::with_capacity(child_turns);
            let mut previous_output_blocks = 0;
            for turn in 0..child_turns {
                let output_tokens =
                    grow(&mut rng, &mut prefix, &mut next_id, previous_output_blocks);
                previous_output_blocks = output_tokens.div_ceil(BLOCK_SIZE);
                turns.push(TurnTrace {
                    input_length: prefix.len() * BLOCK_SIZE,
                    max_output_tokens: output_tokens,
                    hash_ids: prefix.clone(),
                    delay_after_previous_ms: if turn == 0 {
                        0.0
                    } else {
                        tool_gap_ms(&mut rng)
                    },
                    ..Default::default()
                });
            }
            sessions.push(SessionTrace {
                session_id: format!("{session_id}/sub{child_index}"),
                first_arrival_timestamp_ms: Some(child_start),
                turns,
            });
        }
    }

    sessions.sort_by(|a, b| {
        a.first_arrival_timestamp_ms
            .partial_cmp(&b.first_arrival_timestamp_ms)
            .unwrap()
    });
    Trace {
        block_size: BLOCK_SIZE,
        sessions,
    }
}

fn main() -> anyhow::Result<()> {
    let args = parse_args();
    let trace = build_trace(&args);
    let turns: usize = trace.sessions.iter().map(|s| s.turns.len()).sum();
    eprintln!(
        "workload: {} sessions, {} turns, gpu {} blocks x {} workers, host {} blocks/worker",
        trace.sessions.len(),
        turns,
        args.gpu_blocks,
        args.workers,
        args.host_blocks
    );

    let mut engine = serde_json::json!({
        "block_size": BLOCK_SIZE,
        "num_gpu_blocks": args.gpu_blocks,
        "max_num_seqs": 64,
        "speedup_ratio": 1.0,
        "kv_cache_bytes_per_token": KV_BYTES_PER_TOKEN
    });
    if args.host_blocks > 0 {
        engine["native_host_offload"] = serde_json::json!({
            "num_host_blocks": args.host_blocks
        });
    }
    let mocker = MockerConfig::from_value(serde_json::json!({ "engine": engine }))?;
    let router_config = KvRouterConfig {
        enable_session_prefix_index: true,
        router_temperature: 0.0,
        ..KvRouterConfig::default()
    };

    let report = simulate_loaded_trace_with_router_mode_and_options(
        mocker,
        Some(router_config),
        None,
        trace,
        args.workers,
        1.0,
        ReplayRouterMode::KvRouter,
        true,
        None,
        SlaThresholds::default(),
    )?;
    eprintln!(
        "prefix cache reused ratio: {:.3}",
        report.prefix_cache_reused_ratio
    );
    eprintln!("latency: {:?}", report.latency);
    Ok(())
}
