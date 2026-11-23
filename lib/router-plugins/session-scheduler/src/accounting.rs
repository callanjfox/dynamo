// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The gate's total budget from what the router knows is resident (F34,
//! `admission.accounting`).
//!
//! The session table sums each live session's context: a shared prefix counts once per session,
//! a session end is credited at once whether or not the engine frees anything, and nothing says
//! where the KV is. With `mode: index` the total budget's usage is instead the router's session
//! prefix index measured every `refresh_s`: unique resident blocks per tier over the fleet,
//! combined by the same inclusive and exclusive rules as the capacity
//! ([`crate::config::AdmissionPolicy::dram_beyond_hbm`]), plus what the table grew since the
//! measurement (admissions between two measurements are not seen by the index yet).
//!
//! `verify` checks the KV actions a session end implies: some time after a session ends, its
//! blocks at each holder should have dropped. Per worker, the honoured rate is the share of checks
//! that did. Under the threshold the table stops crediting session ends on that worker.
//!
//! Everything here is pure and bounded; [`crate::hooks`] calls it at most every `refresh_s`.

use std::collections::{HashMap, VecDeque};

use dynamo_kv_router::plugins::prelude::WorkerWithDpRank;

use crate::clock::Time;
use crate::config::{AdmissionPolicy, IndexTier, VerifyActions};

/// Most session ends waiting for their check, and most uncredited ends kept.
pub const MAX_PENDING: usize = 1 << 14;

/// A storage tier as the session index reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tier {
    Device,
    Host,
    Disk,
    External,
}

impl Tier {
    fn index_tier(self) -> Option<IndexTier> {
        match self {
            Tier::Device => None,
            Tier::Host => Some(IndexTier::Host),
            Tier::Disk => Some(IndexTier::Disk),
            Tier::External => Some(IndexTier::External),
        }
    }
}

/// One holder and tier of the index, in blocks: unique blocks over the fleet, or one session's
/// resident depth. `holder` is `None` for a shared store outside the workers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Row {
    pub holder: Option<WorkerWithDpRank>,
    pub tier: Tier,
    pub blocks: u64,
}

/// One measurement, in tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Measured {
    pub device: u64,
    pub host: u64,
    pub disk: u64,
    pub external: u64,
    /// Unique KV the configured tiers hold, by their inclusive and exclusive rules.
    pub unique: u64,
}

/// The tiers below HBM the capacity counts, as (index tier, inclusive). `None` when one cannot be
/// measured (a configured tier with no `index_tier`).
fn tier_rules(admission: &AdmissionPolicy) -> Option<Vec<(IndexTier, bool)>> {
    if admission.tiers.is_empty() {
        return Some(if admission.dram_tokens_per_worker > 0 {
            vec![(IndexTier::Host, admission.dram_write_through)]
        } else {
            Vec::new()
        });
    }
    admission
        .tiers
        .iter()
        .map(|t| t.index_tier.map(|i| (i, t.inclusive)))
        .collect()
}

/// Unique resident KV from the index's unique rows, combined as the capacity is: the largest of
/// HBM and the inclusive tiers, plus every exclusive tier, over the fleet. A tier the index reports
/// but the capacity does not count is reported but left out of `unique`. `None` when the rows are
/// empty (the host has no index, or it is not filled) or a configured tier cannot be measured.
pub fn measure(rows: &[Row], block_size: u64, admission: &AdmissionPolicy) -> Option<Measured> {
    if rows.is_empty() {
        return None;
    }
    let rules = tier_rules(admission)?;
    let mut m = Measured::default();
    let mut per: HashMap<IndexTier, u64> = HashMap::new();
    for row in rows {
        let tokens = row.blocks.saturating_mul(block_size);
        match row.tier {
            Tier::Device => m.device += tokens,
            Tier::Host => m.host += tokens,
            Tier::Disk => m.disk += tokens,
            Tier::External => m.external += tokens,
        }
        if let Some(i) = row.tier.index_tier() {
            *per.entry(i).or_default() += tokens;
        }
    }
    let mut inclusive = m.device;
    let mut exclusive = 0u64;
    for (tier, incl) in rules {
        let tokens = per.get(&tier).copied().unwrap_or(0);
        if incl {
            inclusive = inclusive.max(tokens);
        } else {
            exclusive = exclusive.saturating_add(tokens);
        }
    }
    m.unique = inclusive.saturating_add(exclusive);
    Some(m)
}

/// The last measurement and the table's total at that time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Snapshot {
    pub at: Time,
    pub measured: Measured,
    pub table_total: u64,
}

impl Snapshot {
    /// The total budget's usage now: the measured unique KV plus what the table grew since
    /// (sessions admitted, contexts grown). `None` when older than `stale_s`.
    pub fn usage(&self, now: Time, table_total: u64, stale_s: f64) -> Option<u64> {
        (now.since(self.at).as_secs_f64() <= stale_s).then(|| {
            self.measured
                .unique
                .saturating_add(table_total.saturating_sub(self.table_total))
        })
    }
}

/// A session end waiting for its check.
#[derive(Clone, Debug, PartialEq)]
struct Pending {
    session: String,
    at: Time,
    /// Per worker holder: device plus host blocks at the end.
    before: Vec<(WorkerWithDpRank, u64)>,
}

/// A session end the table does not credit (its worker's honoured rate is low).
#[derive(Clone, Debug, PartialEq)]
struct Uncredited {
    session: String,
    at: Time,
    context_tokens: u64,
}

/// The KV-action check (`verify`).
#[derive(Debug, Default)]
pub struct ActionCheck {
    pending: VecDeque<Pending>,
    uncredited: VecDeque<Uncredited>,
    /// Per worker: (checked, honoured).
    counts: HashMap<WorkerWithDpRank, (u64, u64)>,
}

/// Device plus host blocks per worker holder, from one session's residency rows.
pub fn held_by_worker(rows: &[Row]) -> Vec<(WorkerWithDpRank, u64)> {
    let mut per: HashMap<WorkerWithDpRank, (u64, u64)> = HashMap::new();
    for row in rows {
        let Some(worker) = row.holder else { continue };
        let e = per.entry(worker).or_default();
        match row.tier {
            Tier::Device => e.0 = e.0.max(row.blocks),
            Tier::Host => e.1 = e.1.max(row.blocks),
            Tier::Disk | Tier::External => {}
        }
    }
    let mut out: Vec<_> = per.into_iter().map(|(w, (d, h))| (w, d.max(h))).collect();
    out.sort();
    out
}

impl ActionCheck {
    /// Record a session end with its residency rows at the end. A session the index does not
    /// hold is not checked.
    pub fn ended(
        &mut self,
        verify: &VerifyActions,
        session: &str,
        at: Time,
        context_tokens: u64,
        rows: &[Row],
    ) {
        let before = held_by_worker(rows);
        if before.iter().all(|&(_, b)| b == 0) {
            return;
        }
        if before.iter().any(|&(w, _)| {
            self.rate(verify, w)
                .is_some_and(|r| r < verify.honoured_below)
        }) {
            if self.uncredited.len() >= MAX_PENDING {
                self.uncredited.pop_front();
            }
            self.uncredited.push_back(Uncredited {
                session: session.to_owned(),
                at,
                context_tokens,
            });
        }
        if self.pending.len() >= MAX_PENDING {
            self.pending.pop_front();
        }
        self.pending.push_back(Pending {
            session: session.to_owned(),
            at,
            before,
        });
    }

    /// Check every end older than `window_s` against `residency_now(session)`, and drop the
    /// uncredited ends the index shows gone or older than `max_hold_s`.
    pub fn step(
        &mut self,
        verify: &VerifyActions,
        now: Time,
        mut residency_now: impl FnMut(&str) -> Vec<Row>,
    ) {
        let mut gone: Vec<String> = Vec::new();
        while self
            .pending
            .front()
            .is_some_and(|p| now.since(p.at).as_secs_f64() >= verify.window_s)
        {
            let p = self.pending.pop_front().expect("front exists");
            let after: HashMap<_, _> = held_by_worker(&residency_now(&p.session))
                .into_iter()
                .collect();
            let mut all = true;
            for (worker, before) in p.before {
                let left = after.get(&worker).copied().unwrap_or(0);
                let honoured = (left as f64) <= (1.0 - verify.min_drop) * before as f64;
                all &= honoured;
                let e = self.counts.entry(worker).or_default();
                e.0 += 1;
                e.1 += u64::from(honoured);
            }
            if all {
                gone.push(p.session);
            }
        }
        self.uncredited.retain(|u| {
            now.since(u.at).as_secs_f64() < verify.max_hold_s && !gone.contains(&u.session)
        });
    }

    /// The share of `worker`'s checks that were honoured, once it has `min_checks`.
    pub fn rate(&self, verify: &VerifyActions, worker: WorkerWithDpRank) -> Option<f64> {
        let &(checked, honoured) = self.counts.get(&worker)?;
        (checked >= verify.min_checks.max(1)).then(|| honoured as f64 / checked as f64)
    }

    /// Context tokens of the session ends the table does not credit.
    pub fn uncredited_tokens(&self) -> u64 {
        self.uncredited.iter().map(|u| u.context_tokens).sum()
    }

    /// Per worker (sorted): checked, honoured.
    pub fn counts(&self) -> Vec<(WorkerWithDpRank, u64, u64)> {
        let mut out: Vec<_> = self.counts.iter().map(|(&w, &(c, h))| (w, c, h)).collect();
        out.sort();
        out
    }

    pub fn pending(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::config::KvTier;

    fn w(id: u64) -> Option<WorkerWithDpRank> {
        Some(WorkerWithDpRank::new(id, 0))
    }

    fn row(holder: Option<WorkerWithDpRank>, tier: Tier, blocks: u64) -> Row {
        Row {
            holder,
            tier,
            blocks,
        }
    }

    fn at(s: u64) -> Time {
        Time::ZERO.after(Duration::from_secs(s))
    }

    #[test]
    fn write_through_counts_the_larger_of_device_and_host_and_exclusive_tiers_add() {
        let mut a = AdmissionPolicy {
            dram_tokens_per_worker: 100,
            dram_write_through: true,
            ..Default::default()
        };
        let rows = [
            row(w(1), Tier::Device, 10),
            row(w(2), Tier::Device, 6),
            row(w(1), Tier::Host, 20),
            row(w(2), Tier::Disk, 7),
        ];
        // Block size 2: device 32, host 40; write-through: max(32, 40) = 40. Disk is reported but
        // no configured tier counts it.
        let m = measure(&rows, 2, &a).unwrap();
        assert_eq!((m.device, m.host, m.disk, m.unique), (32, 40, 14, 40));
        a.dram_write_through = false;
        assert_eq!(measure(&rows, 2, &a).unwrap().unique, 72);
        // A tier list: inclusive host, exclusive disk.
        a.tiers = vec![
            KvTier {
                name: "dram".into(),
                tokens_per_worker: 100,
                inclusive: true,
                index_tier: Some(IndexTier::Host),
            },
            KvTier {
                name: "ssd".into(),
                tokens_per_worker: 100,
                inclusive: false,
                index_tier: Some(IndexTier::Disk),
            },
        ];
        assert_eq!(measure(&rows, 2, &a).unwrap().unique, 40 + 14);
        // A configured tier the index cannot report: no measurement.
        a.tiers[1].index_tier = None;
        assert_eq!(measure(&rows, 2, &a), None);
        // No rows: no measurement.
        a.tiers.clear();
        assert_eq!(measure(&[], 2, &a), None);
    }

    #[test]
    fn usage_is_the_measurement_plus_the_table_growth_until_stale() {
        let s = Snapshot {
            at: at(100),
            measured: Measured {
                unique: 1_000,
                ..Default::default()
            },
            table_total: 5_000,
        };
        // The table grew by 300 since: admissions the index has not seen yet.
        assert_eq!(s.usage(at(110), 5_300, 90.0), Some(1_300));
        // It shrank: the measurement stands.
        assert_eq!(s.usage(at(110), 4_000, 90.0), Some(1_000));
        assert_eq!(s.usage(at(191), 5_300, 90.0), None);
    }

    #[test]
    fn verify_rates_each_worker_and_stops_crediting_where_ends_are_not_honoured() {
        let v = VerifyActions {
            window_s: 60.0,
            min_drop: 0.5,
            min_checks: 2,
            honoured_below: 0.5,
            max_hold_s: 600.0,
        };
        let mut c = ActionCheck::default();
        // Worker 1 frees what ends there; worker 2 does not.
        for (i, s) in ["a", "b"].iter().enumerate() {
            c.ended(&v, s, at(i as u64), 100, &[row(w(1), Tier::Device, 40)]);
        }
        for (i, s) in ["c", "d"].iter().enumerate() {
            c.ended(
                &v,
                s,
                at(i as u64),
                100,
                &[row(w(2), Tier::Device, 40), row(w(2), Tier::Host, 40)],
            );
        }
        assert_eq!(c.pending(), 4);
        // Before the window nothing is checked.
        c.step(&v, at(30), |_| Vec::new());
        assert_eq!(c.pending(), 4);
        c.step(&v, at(70), |s| match s {
            // A shared prefix of 10 blocks stays: still honoured (75% gone).
            "a" | "b" => vec![row(w(1), Tier::Device, 10)],
            _ => vec![row(w(2), Tier::Device, 40)],
        });
        assert_eq!(c.pending(), 0);
        assert_eq!(c.rate(&v, WorkerWithDpRank::new(1, 0)), Some(1.0));
        assert_eq!(c.rate(&v, WorkerWithDpRank::new(2, 0)), Some(0.0));
        assert_eq!(c.uncredited_tokens(), 0);
        // Now an end on worker 2 is not credited; one on worker 1 is.
        c.ended(&v, "e", at(100), 300, &[row(w(2), Tier::Device, 40)]);
        c.ended(&v, "f", at(100), 500, &[row(w(1), Tier::Device, 40)]);
        assert_eq!(c.uncredited_tokens(), 300);
        // The index shows e gone after the window: it is credited again.
        c.step(&v, at(170), |_| Vec::new());
        assert_eq!(c.uncredited_tokens(), 0);
        // An end held past max_hold_s is dropped even if the index never shows it gone.
        c.ended(&v, "g", at(200), 70, &[row(w(2), Tier::Device, 40)]);
        c.step(&v, at(270), |_| vec![row(w(2), Tier::Device, 40)]);
        assert_eq!(c.uncredited_tokens(), 70);
        c.step(&v, at(801), |_| vec![row(w(2), Tier::Device, 40)]);
        assert_eq!(c.uncredited_tokens(), 0);
        // Sessions the index never held are not checked.
        c.ended(&v, "h", at(900), 10, &[]);
        assert_eq!(c.pending(), 0);
    }
}
