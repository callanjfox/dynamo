// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The working-set log (`DYNAMO_REPLAY_WORKSET_LOG=<path>`, off by default): one JSON line per
//! simulated minute with the KV the live sessions reference, against what the workers hold.
//!
//! A session is live from its first admitted request until a request marked `session_final`
//! completes. Its working set is the prompt blocks of its latest admitted request plus that
//! request's expected output. Prompt blocks are keyed by a chained hash (the router's own
//! sequence hashes, or local block hashes folded into their prefix's with [`chained_keys`]), so a
//! prefix shared by sessions (a system prompt, a parent's context in a subagent) counts once, as
//! in the router's radix tree. The log only reads: it
//! changes no routing or admission decision.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;

use dynamo_kv_events::event::{KvCacheEventData, RouterEvent};
use dynamo_kv_events::residency::StorageTier;
use dynamo_router_protocol::worker::WorkerWithDpRank;
use uuid::Uuid;

/// `DYNAMO_REPLAY_WORKSET_LOG=<path>`: write the working-set log there (off by default).
pub(crate) const REPLAY_WORKSET_LOG_ENV: &str = "DYNAMO_REPLAY_WORKSET_LOG";

/// Simulated time between two lines.
const INTERVAL_MS: f64 = 60_000.0;

pub(crate) fn workset_log() -> Option<WorksetLog> {
    let path = std::env::var_os(REPLAY_WORKSET_LOG_ENV)?;
    match std::fs::File::create(&path) {
        Ok(file) => Some(WorksetLog::new(Box::new(std::io::BufWriter::new(file)))),
        Err(error) => {
            tracing::warn!(path = ?path, %error, "cannot open the replay working-set log");
            None
        }
    }
}

/// Chained block keys from local block hashes: block `i`'s key folds its local hash into block
/// `i - 1`'s key, so equal keys mean equal prefixes.
pub(crate) fn chained_keys(local: &[u64]) -> Vec<u64> {
    let mut prev: u64 = 0x9e37_79b9_7f4a_7c15;
    local
        .iter()
        .map(|&h| {
            // splitmix64 of (prev, h): cheap and well mixed, fixed across runs.
            let mut z = prev ^ h.rotate_left(17) ^ 0xbf58_476d_1ce4_e5b9;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            prev = z ^ (z >> 31);
            prev
        })
        .collect()
}

struct LiveSession {
    class: String,
    keys: Vec<u64>,
    isl_tokens: u64,
    output_tokens: u64,
    in_flight: u32,
}

/// What the router holds outside the live set at a log line: requests in its queue (held by the
/// gate or waiting for capacity) and refused requests waiting for their client's retry.
#[derive(Default)]
pub(crate) struct Waiting {
    /// Per class: requests, sessions not yet live, and those sessions' input tokens.
    pub(crate) per_class: BTreeMap<String, [u64; 3]>,
    pub(crate) retry_requests: u64,
}

impl Waiting {
    pub(crate) fn add(&mut self, class: &str, new_session: bool, isl_tokens: u64) {
        let row = self.per_class.entry(class.to_string()).or_default();
        row[0] += 1;
        if new_session {
            row[1] += 1;
            row[2] += isl_tokens;
        }
    }
}

pub(crate) struct WorksetLog {
    out: Box<dyn Write + Send>,
    next_ms: f64,
    sessions: HashMap<String, LiveSession>,
    /// Requests admitted and not finished: their session and whether they end it.
    requests: HashMap<Uuid, (String, bool)>,
    /// Live sessions referencing each block key, overall and per class.
    refs: HashMap<u64, u32>,
    class_refs: HashMap<String, HashMap<u64, u32>>,
    /// Blocks each rank holds per tier, from its KV events.
    tiers: HashMap<(WorkerWithDpRank, StorageTier), u64>,
}

impl WorksetLog {
    pub(crate) fn new(out: Box<dyn Write + Send>) -> Self {
        Self {
            out,
            next_ms: INTERVAL_MS,
            sessions: HashMap::new(),
            requests: HashMap::new(),
            refs: HashMap::new(),
            class_refs: HashMap::new(),
            tiers: HashMap::new(),
        }
    }

    fn unref(&mut self, class: &str, keys: &[u64]) {
        for key in keys {
            if let Some(n) = self.refs.get_mut(key) {
                *n -= 1;
                if *n == 0 {
                    self.refs.remove(key);
                }
            }
        }
        if let Some(map) = self.class_refs.get_mut(class) {
            for key in keys {
                if let Some(n) = map.get_mut(key) {
                    *n -= 1;
                    if *n == 0 {
                        map.remove(key);
                    }
                }
            }
        }
    }

    fn add_refs(&mut self, class: &str, keys: &[u64]) {
        for &key in keys {
            *self.refs.entry(key).or_default() += 1;
        }
        let map = self.class_refs.entry(class.to_string()).or_default();
        for &key in keys {
            *map.entry(key).or_default() += 1;
        }
    }

    /// A request was admitted: its session becomes live (or stays live) with this request's
    /// blocks (`keys`, chained block hashes) as its working set.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn on_admit(
        &mut self,
        uuid: Uuid,
        session: &str,
        class: &str,
        keys: Vec<u64>,
        isl_tokens: u64,
        output_tokens: u64,
        ends_session: bool,
    ) {
        let (old_class, old_keys) = match self.sessions.get_mut(session) {
            Some(live) => {
                live.in_flight += 1;
                live.isl_tokens = isl_tokens;
                live.output_tokens = output_tokens;
                (
                    std::mem::replace(&mut live.class, class.to_string()),
                    std::mem::replace(&mut live.keys, keys.clone()),
                )
            }
            None => {
                self.sessions.insert(
                    session.to_string(),
                    LiveSession {
                        class: class.to_string(),
                        keys: keys.clone(),
                        isl_tokens,
                        output_tokens,
                        in_flight: 1,
                    },
                );
                (String::new(), Vec::new())
            }
        };
        self.add_refs(class, &keys);
        if !old_keys.is_empty() {
            self.unref(&old_class, &old_keys);
        }
        self.requests
            .insert(uuid, (session.to_string(), ends_session));
    }

    /// A request finished; a session's last request ends the session.
    pub(crate) fn on_terminal(&mut self, uuid: Uuid) {
        let Some((session, ends)) = self.requests.remove(&uuid) else {
            return;
        };
        let Some(live) = self.sessions.get_mut(&session) else {
            return;
        };
        live.in_flight = live.in_flight.saturating_sub(1);
        if ends && let Some(live) = self.sessions.remove(&session) {
            self.unref(&live.class, &live.keys);
        }
    }

    pub(crate) fn is_live(&self, session: &str) -> bool {
        self.sessions.contains_key(session)
    }

    /// Track each rank's blocks per tier from its KV events.
    pub(crate) fn on_events(&mut self, events: &[RouterEvent]) {
        for event in events {
            let rank = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
            let held = self.tiers.entry((rank, event.storage_tier)).or_default();
            match &event.event.data {
                KvCacheEventData::Stored(store) => *held += store.blocks.len() as u64,
                KvCacheEventData::Removed(remove) => {
                    *held = held.saturating_sub(remove.block_hashes.len() as u64)
                }
                KvCacheEventData::Cleared => *held = 0,
            }
        }
    }

    /// Whether a line is due at `now_ms`.
    pub(crate) fn due(&self, now_ms: f64) -> bool {
        now_ms >= self.next_ms
    }

    /// Write the lines due by `now_ms`, each stamped at its minute, from the current state.
    pub(crate) fn write_due(
        &mut self,
        now_ms: f64,
        block_size: u32,
        g1_capacity_blocks: &BTreeMap<WorkerWithDpRank, u64>,
        waiting: &Waiting,
        gate_report: Option<&str>,
    ) {
        while self.due(now_ms) {
            let line = self.line(
                self.next_ms,
                block_size,
                g1_capacity_blocks,
                waiting,
                gate_report,
            );
            let _ = writeln!(self.out, "{line}");
            self.next_ms += INTERVAL_MS;
        }
    }

    fn line(
        &self,
        t_ms: f64,
        block_size: u32,
        g1_capacity_blocks: &BTreeMap<WorkerWithDpRank, u64>,
        waiting: &Waiting,
        gate_report: Option<&str>,
    ) -> String {
        let bs = u64::from(block_size);
        // Per class: live sessions, in flight, unique blocks, session-summed prompt and output.
        let mut per: BTreeMap<&str, [u64; 5]> = BTreeMap::new();
        let (mut sum_prompt, mut sum_output) = (0u64, 0u64);
        for live in self.sessions.values() {
            let row = per.entry(live.class.as_str()).or_default();
            row[0] += 1;
            row[1] += u64::from(live.in_flight > 0);
            row[3] += live.isl_tokens;
            row[4] += live.output_tokens;
            sum_prompt += live.isl_tokens;
            sum_output += live.output_tokens;
        }
        for (class, map) in &self.class_refs {
            if let Some(row) = per.get_mut(class.as_str()) {
                row[2] = map.len() as u64;
            }
        }
        let classes: Vec<String> = per
            .iter()
            .map(|(class, r)| {
                let w = waiting.per_class.get(*class).copied().unwrap_or_default();
                format!(
                    "{class:?}:{{\"live\":{},\"in_flight\":{},\"idle\":{},\"unique_tokens\":{},\"session_sum_tokens\":{},\"output_tokens\":{},\"queued_requests\":{},\"waiting_sessions\":{},\"waiting_isl_tokens\":{}}}",
                    r[0],
                    r[1],
                    r[0] - r[1],
                    r[2] * bs + r[4],
                    r[3] + r[4],
                    r[4],
                    w[0],
                    w[1],
                    w[2]
                )
            })
            .collect();
        // Waiting rows of classes with no live session.
        let extra: Vec<String> = waiting
            .per_class
            .iter()
            .filter(|(class, _)| !per.contains_key(class.as_str()))
            .map(|(class, w)| {
                format!(
                    "{class:?}:{{\"live\":0,\"in_flight\":0,\"idle\":0,\"unique_tokens\":0,\"session_sum_tokens\":0,\"output_tokens\":0,\"queued_requests\":{},\"waiting_sessions\":{},\"waiting_isl_tokens\":{}}}",
                    w[0], w[1], w[2]
                )
            })
            .collect();
        let mut ranks: BTreeMap<WorkerWithDpRank, [u64; 3]> = g1_capacity_blocks
            .iter()
            .map(|(&rank, &cap)| (rank, [cap, 0, 0]))
            .collect();
        for (&(rank, tier), &blocks) in &self.tiers {
            let row = ranks.entry(rank).or_default();
            match tier {
                StorageTier::Device => row[1] += blocks,
                _ => row[2] += blocks,
            }
        }
        let workers: Vec<String> = ranks
            .iter()
            .map(|(rank, r)| {
                format!(
                    "{{\"worker\":{},\"dp_rank\":{},\"g1_capacity_tokens\":{},\"g1_tokens\":{},\"g2_tokens\":{}}}",
                    rank.worker_id,
                    rank.dp_rank,
                    r[0] * bs,
                    r[1] * bs,
                    r[2] * bs
                )
            })
            .collect();
        let g1_cap: u64 = ranks.values().map(|r| r[0]).sum::<u64>() * bs;
        let g1: u64 = ranks.values().map(|r| r[1]).sum::<u64>() * bs;
        let g2: u64 = ranks.values().map(|r| r[2]).sum::<u64>() * bs;
        let all_classes = classes
            .into_iter()
            .chain(extra)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"t_s\":{:.0},\"live_sessions\":{},\"unique_tokens\":{},\"unique_prompt_tokens\":{},\"session_sum_tokens\":{},\"g1_capacity_tokens\":{},\"g1_tokens\":{},\"g2_tokens\":{},\"retry_waiting_requests\":{},\"classes\":{{{}}},\"workers\":[{}],\"gate\":{}}}",
            t_ms / 1000.0,
            self.sessions.len(),
            self.refs.len() as u64 * bs + sum_output,
            self.refs.len() as u64 * bs,
            sum_prompt + sum_output,
            g1_cap,
            g1,
            g2,
            waiting.retry_requests,
            all_classes,
            workers.join(","),
            gate_report.unwrap_or("null")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default)]
    struct Sink(Arc<Mutex<Vec<u8>>>);
    impl Write for Sink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn lines(sink: &Sink) -> Vec<serde_json::Value> {
        String::from_utf8(sink.0.lock().unwrap().clone())
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    #[test]
    fn chained_keys_share_a_common_prefix_only() {
        let a = chained_keys(&[1, 2, 3]);
        let b = chained_keys(&[1, 2, 4]);
        let c = chained_keys(&[9, 2, 3]);
        assert_eq!(a[..2], b[..2]);
        assert_ne!(a[2], b[2]);
        // The same local hash after a different prefix is a different block.
        assert_ne!(a[1], c[1]);
    }

    #[test]
    fn shared_prefixes_count_once_and_ended_sessions_leave() {
        let sink = Sink::default();
        let mut log = WorksetLog::new(Box::new(sink.clone()));
        let (r1, r2, r3) = (Uuid::from_u128(1), Uuid::from_u128(2), Uuid::from_u128(3));
        // Two sessions share a two-block system prompt; each adds one block of its own.
        log.on_admit(
            r1,
            "s1",
            "priority",
            chained_keys(&[10, 11, 12]),
            192,
            8,
            false,
        );
        log.on_admit(
            r2,
            "s2",
            "standard",
            chained_keys(&[10, 11, 13]),
            192,
            4,
            true,
        );
        let caps = BTreeMap::from([(WorkerWithDpRank::new(0, 0), 100u64)]);
        let mut waiting = Waiting::default();
        waiting.add("batch", true, 500);
        log.write_due(60_000.0, 64, &caps, &waiting, Some("{\"hot_tokens\":7}"));
        // s1's next turn grows its context by a block; s2's only request ends it.
        log.on_terminal(r1);
        log.on_admit(
            r3,
            "s1",
            "priority",
            chained_keys(&[10, 11, 12, 14]),
            256,
            8,
            false,
        );
        log.on_terminal(r2);
        log.write_due(120_000.0, 64, &caps, &Waiting::default(), None);
        let out = lines(&sink);
        assert_eq!(out.len(), 2);
        // 4 unique blocks (10, 11 shared) of 64 tokens plus outputs 8 + 4.
        assert_eq!(out[0]["unique_prompt_tokens"], 4 * 64);
        assert_eq!(out[0]["unique_tokens"], 4 * 64 + 12);
        assert_eq!(out[0]["session_sum_tokens"], 192 + 8 + 192 + 4);
        assert_eq!(out[0]["live_sessions"], 2);
        assert_eq!(out[0]["classes"]["standard"]["unique_tokens"], 3 * 64 + 4);
        assert_eq!(out[0]["classes"]["batch"]["waiting_isl_tokens"], 500);
        assert_eq!(out[0]["g1_capacity_tokens"], 6400);
        assert_eq!(out[0]["gate"]["hot_tokens"], 7);
        assert_eq!(out[1]["live_sessions"], 1);
        assert_eq!(out[1]["unique_prompt_tokens"], 4 * 64);
        assert_eq!(out[1]["classes"]["priority"]["in_flight"], 1);
        assert!(out[1]["gate"].is_null());
    }

    #[test]
    fn lines_fill_every_minute_passed() {
        let sink = Sink::default();
        let mut log = WorksetLog::new(Box::new(sink.clone()));
        log.write_due(30_000.0, 64, &BTreeMap::new(), &Waiting::default(), None);
        assert!(lines(&sink).is_empty());
        log.write_due(185_000.0, 64, &BTreeMap::new(), &Waiting::default(), None);
        let out = lines(&sink);
        assert_eq!(
            out.iter()
                .map(|l| l["t_s"].as_f64().unwrap())
                .collect::<Vec<_>>(),
            vec![60.0, 120.0, 180.0]
        );
    }
}
