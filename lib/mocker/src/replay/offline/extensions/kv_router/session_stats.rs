// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Drives the session stats tracker from offline replay, in simulated time.
//!
//! The recorder is passive: it reads the session index and the router's request
//! lifecycle and writes what it sees, so it cannot change placement. Output is opt-in
//! through `DYN_SESSION_STATS_JSONL`:
//!
//! - `<path>` gets one working-set row per sample interval.
//! - `<path>.sessions` gets one summary per session, written when it ends.
//!
//! Replay traces carry a session id but no parent link, so a session id of the form
//! `parent/child` is treated as a subagent of `parent`. A live router reads the real
//! parent from the request instead.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

use dynamo_kv_router::{
    LocalBlockHash, SessionPrefixIndexer, SessionStatsConfig, SessionStatsSnapshot,
    SessionStatsTracker, session_stats::SessionSummary,
};
use rustc_hash::FxHashSet;
use serde::Serialize;
use uuid::Uuid;

const ENV_PATH: &str = "DYN_SESSION_STATS_JSONL";
const ENV_SAMPLE_MS: &str = "DYN_SESSION_STATS_SAMPLE_MS";
const ENV_ACTIVE_WINDOW_MS: &str = "DYN_SESSION_STATS_ACTIVE_WINDOW_MS";
const ENV_IDLE_TTL_MS: &str = "DYN_SESSION_STATS_IDLE_TTL_MS";

const DEFAULT_SAMPLE_MS: u64 = 5_000;
const DEFAULT_ACTIVE_WINDOW_MS: u64 = 300_000;
const DEFAULT_IDLE_TTL_MS: u64 = 600_000;

/// A request reaching the router, as the recorder needs to see it.
pub(super) struct ArrivingRequest<'a> {
    pub(super) uuid: Uuid,
    pub(super) session_id: &'a str,
    pub(super) isl_tokens: usize,
    pub(super) osl_tokens: usize,
    pub(super) prompt_hashes: Option<&'a [LocalBlockHash]>,
}

struct Inflight {
    session_id: String,
    osl_tokens: u64,
    cached_tokens: u64,
}

#[derive(Serialize)]
struct SeriesRow<'a> {
    #[serde(flatten)]
    snapshot: &'a SessionStatsSnapshot,
    /// Distinct tokens in the prompts of active sessions' latest requests, counting a
    /// shared prefix once. The tracker's `demand_tokens` is a per-session sum.
    demand_distinct_tokens: u64,
    capacity_device_tokens: u64,
    capacity_host_tokens: u64,
}

pub(super) struct SessionStatsRecorder {
    tracker: SessionStatsTracker,
    block_size: u64,
    capacity_device_tokens: u64,
    capacity_host_tokens: u64,
    sample_interval_ms: u64,
    next_sample_ms: u64,
    last_now_ms: u64,
    inflight: HashMap<Uuid, Inflight>,
    /// Each session's latest prompt as chained block hashes, so two prompts that
    /// share a prefix share entries.
    prompt_chains: HashMap<String, Vec<u64>>,
    series: Option<BufWriter<File>>,
    summaries: Option<BufWriter<File>>,
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

/// `parent/child` names a subagent of `parent`.
fn parent_session_id(session_id: &str) -> Option<&str> {
    session_id
        .rsplit_once('/')
        .map(|(parent, _)| parent)
        .filter(|parent| !parent.is_empty())
}

/// Chain each block hash with its predecessors, so equal values mean equal prefixes.
fn chain(hashes: &[LocalBlockHash]) -> Vec<u64> {
    let mut state = 0xcbf2_9ce4_8422_2325_u64;
    hashes
        .iter()
        .map(|hash| {
            state = (state ^ hash.0)
                .wrapping_mul(0x0000_0100_0000_01b3)
                .rotate_left(29)
                ^ state.wrapping_shr(7);
            state
        })
        .collect()
}

fn open(path: &str) -> Option<BufWriter<File>> {
    match File::create(path) {
        Ok(file) => Some(BufWriter::new(file)),
        Err(error) => {
            tracing::warn!(%error, path, "session stats output disabled");
            None
        }
    }
}

impl SessionStatsRecorder {
    pub(super) fn new(
        block_size: u32,
        capacity_device_tokens: u64,
        capacity_host_tokens: u64,
    ) -> Self {
        let path = std::env::var(ENV_PATH).ok().filter(|path| !path.is_empty());
        let sample_interval_ms = env_u64(ENV_SAMPLE_MS, DEFAULT_SAMPLE_MS);
        Self {
            tracker: SessionStatsTracker::new(SessionStatsConfig {
                block_size,
                active_window_ms: env_u64(ENV_ACTIVE_WINDOW_MS, DEFAULT_ACTIVE_WINDOW_MS),
                idle_ttl_ms: env_u64(ENV_IDLE_TTL_MS, DEFAULT_IDLE_TTL_MS),
                ..SessionStatsConfig::default()
            }),
            block_size: u64::from(block_size.max(1)),
            capacity_device_tokens,
            capacity_host_tokens,
            sample_interval_ms,
            next_sample_ms: 0,
            last_now_ms: 0,
            inflight: HashMap::new(),
            prompt_chains: HashMap::new(),
            summaries: path
                .as_deref()
                .and_then(|path| open(&format!("{path}.sessions"))),
            series: path.as_deref().and_then(open),
        }
    }

    pub(super) fn on_arrival(
        &mut self,
        index: &SessionPrefixIndexer,
        arrival: ArrivingRequest<'_>,
        now_ms: f64,
    ) {
        let now_ms = self.advance(index, now_ms);
        if let Some(hashes) = arrival.prompt_hashes {
            self.prompt_chains
                .insert(arrival.session_id.to_string(), chain(hashes));
        }
        let resident = index.session_resident_blocks(arrival.session_id);
        self.tracker.on_arrival(
            arrival.session_id,
            parent_session_id(arrival.session_id),
            arrival.isl_tokens as u64,
            resident,
            now_ms,
        );
        self.inflight.insert(
            arrival.uuid,
            Inflight {
                session_id: arrival.session_id.to_string(),
                osl_tokens: arrival.osl_tokens as u64,
                cached_tokens: 0,
            },
        );
    }

    /// The tokens the selected worker already held when the request was placed.
    pub(super) fn on_admitted(&mut self, uuid: Uuid, isl_tokens: usize, overlap_blocks: u32) {
        if let Some(inflight) = self.inflight.get_mut(&uuid) {
            inflight.cached_tokens =
                (u64::from(overlap_blocks) * self.block_size).min(isl_tokens as u64);
        }
    }

    pub(super) fn on_completed(&mut self, index: &SessionPrefixIndexer, uuid: Uuid, now_ms: f64) {
        let now_ms = self.advance(index, now_ms);
        if let Some(inflight) = self.inflight.remove(&uuid) {
            self.tracker.on_complete(
                &inflight.session_id,
                inflight.osl_tokens,
                inflight.cached_tokens,
                now_ms,
            );
        }
    }

    /// Move the clock to `now_ms`, writing a sample row for each interval crossed.
    fn advance(&mut self, index: &SessionPrefixIndexer, now_ms: f64) -> u64 {
        let now_ms = now_ms.max(0.0) as u64;
        self.last_now_ms = self.last_now_ms.max(now_ms);
        if now_ms >= self.next_sample_ms {
            self.sample(index, now_ms);
            self.next_sample_ms = (now_ms / self.sample_interval_ms + 1) * self.sample_interval_ms;
        }
        now_ms
    }

    fn sample(&mut self, index: &SessionPrefixIndexer, now_ms: u64) {
        let ended = self.tracker.sweep(now_ms);
        self.write_summaries(&ended);
        let Some(series) = self.series.as_mut() else {
            return;
        };
        let active = self.tracker.active_sessions(now_ms);
        let residency = index.residency_snapshot_with(|session| !active.contains(session));
        let snapshot = self.tracker.snapshot(&residency, now_ms);
        let mut demand: FxHashSet<u64> = FxHashSet::default();
        for session in &active {
            if let Some(chain) = self.prompt_chains.get(session) {
                demand.extend(chain.iter().copied());
            }
        }
        let row = SeriesRow {
            snapshot: &snapshot,
            demand_distinct_tokens: demand.len() as u64 * self.block_size,
            capacity_device_tokens: self.capacity_device_tokens,
            capacity_host_tokens: self.capacity_host_tokens,
        };
        if serde_json::to_writer(&mut *series, &row).is_ok() {
            let _ = series.write_all(b"\n");
        }
    }

    fn write_summaries(&mut self, summaries: &[SessionSummary]) {
        let Some(out) = self.summaries.as_mut() else {
            return;
        };
        for summary in summaries {
            if serde_json::to_writer(&mut *out, summary).is_ok() {
                let _ = out.write_all(b"\n");
            }
        }
    }
}

impl Drop for SessionStatsRecorder {
    fn drop(&mut self) {
        let ended = self.tracker.end_all(self.last_now_ms);
        self.write_summaries(&ended);
        if let Some(series) = self.series.as_mut() {
            let _ = series.flush();
        }
        if let Some(summaries) = self.summaries.as_mut() {
            let _ = summaries.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subagent_ids_name_their_parent() {
        assert_eq!(parent_session_id("task"), None);
        assert_eq!(parent_session_id("task/sub-1"), Some("task"));
        assert_eq!(parent_session_id("task/sub-1/leaf"), Some("task/sub-1"));
        assert_eq!(parent_session_id("/orphan"), None);
    }

    #[test]
    fn chained_hashes_match_only_for_equal_prefixes() {
        let a = chain(&[LocalBlockHash(1), LocalBlockHash(2), LocalBlockHash(3)]);
        let b = chain(&[LocalBlockHash(1), LocalBlockHash(2), LocalBlockHash(9)]);
        let shifted = chain(&[LocalBlockHash(2), LocalBlockHash(3)]);
        assert_eq!(a[..2], b[..2]);
        assert_ne!(a[2], b[2]);
        assert!(shifted.iter().all(|hash| !a.contains(hash)));
    }
}
