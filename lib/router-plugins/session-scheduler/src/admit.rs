// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The first admission policy: a session gate on the KV budget (build step 2 of the scheduler
//! design, the admission simulator's `sessionGate` with `gateHarm: false`).
//!
//! A new session is admitted when its first turn fits two budgets:
//!
//! - **hot**: the sessions mid-turn, plus this one, fit within the class's `gate_share` of HBM;
//! - **total**: every live session's context, plus this one, fits within `total_headroom` of HBM
//!   plus DRAM.
//!
//! The rest wait in the lobby, ordered by class rank and then first come (or least served, with
//! `lobby_order: fair`). Within a class the lobby is first in, first out: while an earlier
//! session of the same class waits, a later one waits too. Room is reserved for every session
//! ahead in the order, so a lower class never jumps a higher one that does not fit yet. After the
//! class's `max_lobby_wait_s` the session is admitted anyway (`on_wait_limit: admit`) or expires
//! (`expire`).
//!
//! Everything here is a pure function of its inputs, so the gate's `check` can call it as a read.

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use crate::clock::Time;
use crate::config::{
    ClassOvercommitMode, ClassPolicy, LobbyOrder, Mode, ServicePolicy, WaitLimitAction,
};
use crate::session::{Session, SessionTable};

/// The KV picture the gate decides on, in tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KvBudget {
    pub hbm_tokens: u64,
    pub dram_tokens: u64,
    /// Context of sessions with a turn in flight.
    pub hot_tokens: u64,
    /// Context of every live (admitted) session.
    pub total_tokens: u64,
    /// Of `hot_tokens` and `total_tokens`, the yield class's (`admission.yield_class`).
    pub yield_hot_tokens: u64,
    pub yield_total_tokens: u64,
}

/// A session waiting for admission.
#[derive(Clone, Debug, PartialEq)]
pub struct LobbyEntry {
    pub session: String,
    pub class: String,
    pub arrived: Time,
    /// The arrival the lobby orders it by: `arrived`, or with `lobby_order: aging` the start of
    /// its wait counting retries and earlier waits ([`crate::session::SessionTable::order_at`]).
    pub order_at: Time,
    /// Input tokens of the first turn: what admitting it adds.
    pub isl: u64,
    /// Turns this user has been served, for `lobby_order: fair`.
    pub served: u32,
    /// The ordering key in millionths (`turn_order: run_las` or `server_wait_deficit`; 0
    /// otherwise), ahead of `order_at`.
    pub order: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmitReason {
    Fits,
    WaitLimit,
    /// f49q `admission.starvation_override`: starved by its session signal.
    Starved,
}

#[derive(Clone, Debug, PartialEq)]
pub enum HoldReason {
    /// An earlier session of the same class is still waiting.
    BehindEarlier { ahead: String },
    /// Not enough room once the sessions ahead are reserved.
    KvBudget {
        needed: u64,
        hot_room: u64,
        total_room: u64,
    },
    /// Shed by slack: a higher class (`short`) has no slack and no worker has KV room for this
    /// turn of an admitted session.
    NoSlack { short: String },
    /// Every worker is at its running-request limit (`admission.hold_at_max_num_seqs`).
    NoSeqRoom,
    /// Surge admission (`admission.surge`): the fleet's queued prefill takes `backlog_s` to clear,
    /// over `limit_s`, the TTFT limit of a class ranked above.
    PrefillBacklog { backlog_s: f64, limit_s: f64 },
    /// Admission on compute headroom (`admission.compute_headroom`, f36b): `class`, ranked at or
    /// above the session's, has run headroom `headroom`, under the hold threshold.
    ComputeHeadroom { class: String, headroom: f64 },
    /// Onboarding pace (`admission.onboard_pace`, f39): the onboarding prefill bucket is in debt
    /// (`level` tokens).
    OnboardBudget { level: f64 },
    /// The two-resource gate (`admission.prefill_budget`, f42): the fleet's prefill rate `used`
    /// plus the session's onboarding `own` (tokens per second) is over the budget `budget` (the
    /// workers' prefill budgets times the class's scale).
    PrefillBudget { used: f64, own: f64, budget: f64 },
    /// The target loop (`admission.target_loop`, f42) cannot meet `class`'s target even with the
    /// lower classes held back: new sessions of the lower classes wait (and are refused with
    /// Retry-After at their wait limit).
    TargetInfeasible { class: String },
    /// The forecast-harm check on new sessions (`admission.forecast_harm_new`, ks2b): the session's
    /// best placement adds `added` weighted misses to its class and the classes above it, over
    /// `limit`.
    ForecastHarm { added: f64, limit: f64 },
}

/// What the policy wants for one waiting session.
#[derive(Clone, Debug, PartialEq)]
pub enum Decision {
    Admit(AdmitReason),
    /// Keep waiting; decide again on a recheck or at `until` (the class's wait limit).
    Hold {
        reason: HoldReason,
        until: Time,
    },
    /// The wait limit passed and the class turns sessions away. The router expires the request
    /// through its `due_at`, which the classifier set at classification.
    Expire {
        waited: Duration,
    },
    /// A new session arrived while the lobby was full. Decided at classification, where the
    /// only way to act is to reject the request; the gate never sees one.
    Throttle {
        waiting: usize,
        limit: usize,
    },
}

/// Sort the lobby into service order.
pub fn order_lobby(policy: &ServicePolicy, lobby: &mut [LobbyEntry]) {
    let rank = |class: &str| policy.class(Some(class)).rank;
    lobby.sort_by(|a, b| {
        rank(&a.class)
            .cmp(&rank(&b.class))
            .then_with(|| match policy.admission.lobby_order {
                // Tenant order needs live tenant counts the entries do not carry; the indexed
                // decision (`decide_indexed`) applies it.
                // f49q `signal`: the ordering key carries it.
                LobbyOrder::Fifo | LobbyOrder::Aging | LobbyOrder::Tenant | LobbyOrder::Signal => {
                    std::cmp::Ordering::Equal
                }
                LobbyOrder::Fair => a.served.cmp(&b.served),
            })
            .then_with(|| a.order.cmp(&b.order))
            .then_with(|| a.order_at.cmp(&b.order_at))
            .then_with(|| a.session.cmp(&b.session))
    });
}

/// Decide for `candidate`, one of the sessions in `lobby`.
pub fn decide(
    policy: &ServicePolicy,
    now: Time,
    candidate: &str,
    lobby: &[LobbyEntry],
    kv: &KvBudget,
) -> Decision {
    let mut ordered = lobby.to_vec();
    order_lobby(policy, &mut ordered);
    let Some(index) = ordered.iter().position(|e| e.session == candidate) else {
        // Not waiting: nothing to hold.
        return Decision::Admit(AdmitReason::Fits);
    };
    let entry = &ordered[index];
    let class = policy.class(Some(&entry.class));
    let limit = Duration::from_secs_f64(class.admission.max_lobby_wait_s);
    let waited = now.since(entry.arrived);
    if waited >= limit {
        return match class.admission.on_wait_limit {
            WaitLimitAction::Admit => Decision::Admit(AdmitReason::WaitLimit),
            WaitLimitAction::Expire => Decision::Expire { waited },
        };
    }
    let until = entry.arrived.after(limit);
    let ahead = &ordered[..index];
    if let Some(earlier) = ahead.iter().rev().find(|e| e.class == entry.class) {
        return Decision::Hold {
            reason: HoldReason::BehindEarlier {
                ahead: earlier.session.clone(),
            },
            until,
        };
    }
    let reserved: u64 = ahead.iter().map(|e| e.isl).sum();
    fits(
        policy, class, entry.isl, entry.isl, reserved, kv, until, 1.0,
    )
}

/// f49k2: the hot budget (in tokens) of `class` in the gate's fit, with `scale` the share scale of
/// each class (pacing, release, the target loop and f49j's step target). A class's raw budget is
/// `gate_share x scale x hot_overcommit x class_overcommit x` HBM. Without
/// `admission.class_overcommit` (or with `class_overcommit_mode: legacy`, f49k) it is the class's
/// raw budget. Otherwise, as `admission.class_overcommit_mode` says ([`ClassOvercommitMode`]):
/// `max` (default) the largest raw budget of the classes ranked at or below it, so it never has
/// less hot room than a class ranked below it; `separate` its raw budget plus the overcommitted
/// part (the raw budget past its budget at `class_overcommit` 1) of each class ranked below it.
/// Classes the hot budget leaves out (`admission.hot_exempt_classes`) lend nothing.
pub fn hot_cap(
    policy: &ServicePolicy,
    class: &ClassPolicy,
    hbm_tokens: u64,
    scale: &dyn Fn(&ClassPolicy) -> f64,
) -> u64 {
    let raw = |c: &ClassPolicy, overcommit: f64| {
        (c.admission.gate_share
            * scale(c)
            * policy.admission.hot_overcommit
            * overcommit
            * hbm_tokens as f64) as u64
    };
    let own = raw(class, policy.class_overcommit(class));
    if policy.admission.class_overcommit.is_empty() {
        return own;
    }
    let others = policy
        .classes
        .iter()
        .filter(|c| c.name != class.name && !policy.admission.hot_exempt_classes.contains(&c.name));
    match policy.admission.class_overcommit_mode.unwrap_or_default() {
        ClassOvercommitMode::Legacy => own,
        ClassOvercommitMode::Max => others
            .filter(|c| c.rank >= class.rank)
            .map(|c| raw(c, policy.class_overcommit(c)))
            .fold(own, u64::max),
        ClassOvercommitMode::Separate => others
            .filter(|c| c.rank > class.rank)
            .map(|c| raw(c, policy.class_overcommit(c)).saturating_sub(raw(c, 1.0)))
            .fold(own, u64::saturating_add),
    }
}

/// The per-class scale of the callers that know only the decided class's: `scale` for `class`,
/// 1 for the others.
fn own_scale(class: &ClassPolicy, scale: f64) -> impl Fn(&ClassPolicy) -> f64 + '_ {
    move |c| if c.name == class.name { scale } else { 1.0 }
}

/// The budget test shared by [`decide`] and [`decide_indexed`]: does `isl` fit once `reserved`
/// tokens are kept for the sessions ahead? [`fits_with`] with the other classes at scale 1.
#[allow(clippy::too_many_arguments)]
fn fits(
    policy: &ServicePolicy,
    class: &ClassPolicy,
    isl: u64,
    total_isl: u64,
    reserved: u64,
    kv: &KvBudget,
    until: Time,
    share_scale: f64,
) -> Decision {
    fits_with(
        policy,
        class,
        isl,
        total_isl,
        reserved,
        kv,
        until,
        &own_scale(class, share_scale),
    )
}

/// [`fits`] with `scale` the share scale of each class ([`hot_cap`], f49k2).
#[allow(clippy::too_many_arguments)]
fn fits_with(
    policy: &ServicePolicy,
    class: &ClassPolicy,
    isl: u64,
    total_isl: u64,
    reserved: u64,
    kv: &KvBudget,
    until: Time,
    scale: &dyn Fn(&ClassPolicy) -> f64,
) -> Decision {
    let hot_cap = hot_cap(policy, class, kv.hbm_tokens, scale);
    // A higher class than the yield class does not count its KV: the engines make it give way.
    let yields_to = policy
        .admission
        .yield_class
        .as_deref()
        .is_some_and(|y| y != class.name && policy.class(Some(y)).rank > class.rank)
        // f49o `admission.batch_yield`: the same discount, its KV totals already times the level
        // (`Scheduler::kv_budget`; 0 while the level is 0).
        || (policy.admission.yield_class.is_none()
            && policy.admission.batch_yield.as_ref().is_some_and(|b| {
                b.admit && b.class != class.name && policy.class(Some(&b.class)).rank > class.rank
            }));
    let (hot_tokens, total_tokens) = if yields_to {
        (
            kv.hot_tokens.saturating_sub(kv.yield_hot_tokens),
            kv.total_tokens.saturating_sub(kv.yield_total_tokens),
        )
    } else {
        (kv.hot_tokens, kv.total_tokens)
    };
    let total_cap =
        (policy.admission.total_headroom * (kv.hbm_tokens + kv.dram_tokens) as f64) as u64;
    // Without the hot budget (`admission.hot_budget: false`), or for a class it leaves out
    // (`admission.hot_exempt_classes`), only the total budget binds here.
    let exempt = policy.admission.hot_exempt_classes.contains(&class.name);
    let hot_room = if policy.admission.hot_budget.on() && !exempt {
        hot_cap.saturating_sub(hot_tokens).saturating_sub(reserved)
    } else {
        u64::MAX
    };
    let total_room = total_cap
        .saturating_sub(total_tokens)
        .saturating_sub(reserved);
    if isl <= hot_room && total_isl <= total_room {
        Decision::Admit(AdmitReason::Fits)
    } else {
        Decision::Hold {
            reason: HoldReason::KvBudget {
                needed: isl,
                hot_room,
                total_room,
            },
            until,
        }
    }
}

/// f49k: what a waiting session's turn charges the hot budget: its input, or with
/// `admission.hot_budget: measured` its context plus hinted output (the larger of its context and
/// the turn's input, plus `osl_tokens`), as the session table counts a session mid-turn under
/// `hot_demand: next_turn`.
pub fn hot_charge(policy: &ServicePolicy, session: &Session) -> u64 {
    if policy.admission.hot_budget.is_measured() {
        session
            .context_tokens
            .max(session.pending_isl)
            .saturating_add(u64::from(session.signals.osl_tokens.unwrap_or(0)))
    } else {
        session.pending_isl
    }
}

/// The KV room a new session of `class` has now with nothing reserved (f49c, the tickets'
/// Retry-After forecast): the smaller of its hot room (when the hot budget binds it) and the total
/// room, as [`fits`] computes them.
pub fn room(policy: &ServicePolicy, class: &ClassPolicy, kv: &KvBudget) -> u64 {
    room_scaled(policy, class, kv, 1.0)
}

/// [`room`] with the class's hot budget multiplied by `scale`, as [`fits`] takes its
/// `share_scale` (f49j: `admission.step_target`'s multiple). At 1 the same as [`room`]; the total
/// room never scales.
pub fn room_scaled(policy: &ServicePolicy, class: &ClassPolicy, kv: &KvBudget, scale: f64) -> u64 {
    room_classes(policy, class, kv, &own_scale(class, scale))
}

/// [`room`] with `scale` the share scale of each class ([`hot_cap`], f49k2).
pub fn room_classes(
    policy: &ServicePolicy,
    class: &ClassPolicy,
    kv: &KvBudget,
    scale: &dyn Fn(&ClassPolicy) -> f64,
) -> u64 {
    let hot_cap = hot_cap(policy, class, kv.hbm_tokens, scale);
    let total_cap =
        (policy.admission.total_headroom * (kv.hbm_tokens + kv.dram_tokens) as f64) as u64;
    let exempt = policy.admission.hot_exempt_classes.contains(&class.name);
    let hot_room = if policy.admission.hot_budget.on() && !exempt {
        hot_cap.saturating_sub(kv.hot_tokens)
    } else {
        u64::MAX
    };
    hot_room.min(total_cap.saturating_sub(kv.total_tokens))
}

/// [`decide`] for a session of `table`, reading the table's incrementally kept lobby instead of
/// sorting a copy of it: O(classes · log n) per call (ledger L65). Same answer as [`decide`] over
/// the table's lobby; `tests::indexed_matches_full_scan` checks that on random event sequences.
pub fn decide_indexed(
    policy: &ServicePolicy,
    now: Time,
    session: &Session,
    table: &SessionTable,
    kv: &KvBudget,
) -> Decision {
    decide_indexed_scaled(policy, now, session, table, kv, 1.0)
}

/// [`decide_indexed`] with the class's gate share multiplied by `share_scale` (pacing).
pub fn decide_indexed_scaled(
    policy: &ServicePolicy,
    now: Time,
    session: &Session,
    table: &SessionTable,
    kv: &KvBudget,
    share_scale: f64,
) -> Decision {
    decide_indexed_held(policy, now, session, table, kv, share_scale, 0, false)
}

/// [`decide_indexed_scaled`] with admission tickets (`admission.tickets`, f49c): `held` KV tokens
/// are kept for live tickets on top of the room reserved for the other classes' sessions ahead,
/// and with `front` (the session holds a live ticket) earlier sessions of its class do not hold it
/// back. With `held` 0 and `front` false, the same answer as without tickets.
#[allow(clippy::too_many_arguments)]
pub fn decide_indexed_held(
    policy: &ServicePolicy,
    now: Time,
    session: &Session,
    table: &SessionTable,
    kv: &KvBudget,
    share_scale: f64,
    held: u64,
    front: bool,
) -> Decision {
    let class = policy.class(Some(&session.class));
    decide_indexed_classes(
        policy,
        now,
        session,
        table,
        kv,
        &own_scale(class, share_scale),
        held,
        front,
    )
}

/// [`decide_indexed_held`] with `scale` the share scale of each class ([`hot_cap`], f49k2): the
/// session's class's at its own, the other classes' for their hot budgets under
/// `admission.class_overcommit_mode`.
#[allow(clippy::too_many_arguments)]
pub fn decide_indexed_classes(
    policy: &ServicePolicy,
    now: Time,
    session: &Session,
    table: &SessionTable,
    kv: &KvBudget,
    scale: &dyn Fn(&ClassPolicy) -> f64,
    held: u64,
    front: bool,
) -> Decision {
    let Some(key) = table.lobby_key(session) else {
        return Decision::Admit(AdmitReason::Fits);
    };
    let class = policy.class(Some(&session.class));
    let limit = Duration::from_secs_f64(class.admission.max_lobby_wait_s);
    let waited = now.since(session.arrived);
    if waited >= limit && !session.resumed {
        // f47: a listed class that fits the budgets at its wait limit is admitted rather than
        // refused for the order of earlier sessions of its class.
        if matches!(class.admission.on_wait_limit, WaitLimitAction::Expire)
            && policy
                .admission
                .fit_at_limit_classes
                .contains(&session.class)
        {
            let reserved = table
                .lobby()
                .isl_ahead_in_other_classes(&session.class, &key)
                + held;
            let d = fits_with(
                policy,
                class,
                hot_charge(policy, session),
                table.total_need(session),
                reserved,
                kv,
                now,
                scale,
            );
            if matches!(d, Decision::Admit(_)) {
                return d;
            }
        }
        return match class.admission.on_wait_limit {
            WaitLimitAction::Admit => Decision::Admit(AdmitReason::WaitLimit),
            WaitLimitAction::Expire => Decision::Expire { waited },
        };
    }
    // A rotated session waiting to resume is never refused: it waits its turn and its room.
    let until = if session.resumed && waited >= limit {
        now.after(limit)
    } else {
        session.arrived.after(limit)
    };
    let earlier = if front {
        None
    } else if policy.admission.lobby_order == LobbyOrder::Tenant {
        let shares = &policy.admission.tenant_shares;
        table.lobby().earlier_in_class_by_tenant(
            &session.class,
            session.tenant.as_deref().unwrap_or(""),
            &key,
            |t| {
                table.tenant_live(&session.class, t) as f64
                    / shares.get(t).copied().unwrap_or(1.0).max(1e-6)
            },
        )
    } else if policy.admission.lobby_order == LobbyOrder::Aging {
        // A session away after its wait limit expired it holds its place until its retry is
        // overdue: its Retry-After, if it was given one, plus one more wait limit after the
        // deadline that expired it.
        let overdue = |k: &crate::session::LobbyKey| {
            table.get(&k.session).is_some_and(|s| {
                s.away() && {
                    let l = policy.class(Some(&s.class)).admission.max_lobby_wait_s;
                    now.since(s.arrived).as_secs_f64() > 2.0 * l + s.retry_after_s.unwrap_or(0.0)
                }
            })
        };
        table
            .lobby()
            .earlier_in_class_unless(&session.class, &key, overdue)
    } else {
        table.lobby().earlier_in_class(&session.class, &key)
    };
    if let Some(earlier) = earlier {
        return Decision::Hold {
            reason: HoldReason::BehindEarlier {
                ahead: earlier.session.clone(),
            },
            until,
        };
    }
    let reserved = table
        .lobby()
        .isl_ahead_in_other_classes(&session.class, &key)
        + held;
    fits_with(
        policy,
        class,
        hot_charge(policy, session),
        table.total_need(session),
        reserved,
        kv,
        until,
        scale,
    )
}

/// `session`'s budget test now, ignoring earlier sessions of its own class (the room the other
/// classes' sessions ahead need is still kept): f47's returnee guarantee.
pub fn fits_ignoring_order(
    policy: &ServicePolicy,
    now: Time,
    session: &Session,
    table: &SessionTable,
    kv: &KvBudget,
    share_scale: f64,
) -> Decision {
    fits_ignoring_order_held(policy, now, session, table, kv, share_scale, 0)
}

/// [`fits_ignoring_order`] with `held` KV tokens kept for live admission tickets (f49c).
pub fn fits_ignoring_order_held(
    policy: &ServicePolicy,
    now: Time,
    session: &Session,
    table: &SessionTable,
    kv: &KvBudget,
    share_scale: f64,
    held: u64,
) -> Decision {
    let class = policy.class(Some(&session.class));
    fits_ignoring_order_classes(
        policy,
        now,
        session,
        table,
        kv,
        &own_scale(class, share_scale),
        held,
    )
}

/// [`fits_ignoring_order_held`] with `scale` the share scale of each class ([`hot_cap`], f49k2).
pub fn fits_ignoring_order_classes(
    policy: &ServicePolicy,
    now: Time,
    session: &Session,
    table: &SessionTable,
    kv: &KvBudget,
    scale: &dyn Fn(&ClassPolicy) -> f64,
    held: u64,
) -> Decision {
    let Some(key) = table.lobby_key(session) else {
        return Decision::Admit(AdmitReason::Fits);
    };
    let class = policy.class(Some(&session.class));
    let reserved = table
        .lobby()
        .isl_ahead_in_other_classes(&session.class, &key)
        + held;
    fits_with(
        policy,
        class,
        hot_charge(policy, session),
        table.total_need(session),
        reserved,
        kv,
        now,
        scale,
    )
}

/// What the dispatch gate does with a decision, by mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GateAnswer {
    Go,
    /// Keep the head queued; recheck at `until` if set.
    Hold {
        until: Option<Time>,
    },
}

pub fn gate_answer(mode: Mode, decision: &Decision) -> GateAnswer {
    match (mode, decision) {
        (Mode::Off | Mode::Shadow, _)
        | (Mode::Enforce, Decision::Admit(_) | Decision::Throttle { .. }) => GateAnswer::Go,
        (Mode::Enforce, Decision::Hold { until, .. }) => GateAnswer::Hold {
            until: Some(*until),
        },
        // The request's `due_at` expires it; until then it stays queued.
        (Mode::Enforce, Decision::Expire { .. }) => GateAnswer::Hold { until: None },
    }
}

/// One logged decision.
#[derive(Clone, Debug, PartialEq)]
pub struct LoggedDecision {
    pub at: Time,
    pub session: String,
    pub class: String,
    pub mode: Mode,
    pub decision: Decision,
}

/// A bounded log of decisions. The gate may be asked about the same head many times; a decision
/// is logged only when it differs from the last one logged for that session.
#[derive(Debug)]
pub struct DecisionLog {
    capacity: usize,
    entries: VecDeque<LoggedDecision>,
    last: HashMap<String, Decision>,
}

impl DecisionLog {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: VecDeque::new(),
            last: HashMap::new(),
        }
    }

    /// Returns whether the decision was new for the session.
    pub fn record(&mut self, entry: LoggedDecision) -> bool {
        if self.last.get(&entry.session) == Some(&entry.decision) {
            return false;
        }
        if self.last.len() >= self.capacity * 4 {
            self.last.clear();
        }
        self.last
            .insert(entry.session.clone(), entry.decision.clone());
        self.entries.push_back(entry);
        while self.entries.len() > self.capacity {
            self.entries.pop_front();
        }
        true
    }

    /// Forget a session that left the lobby.
    pub fn forget(&mut self, session: &str) {
        self.last.remove(session);
    }

    pub fn entries(&self) -> impl Iterator<Item = &LoggedDecision> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: f64) -> Time {
        Time::from_secs_f64(s)
    }

    fn e(session: &str, class: &str, arrived: f64, isl: u64) -> LobbyEntry {
        LobbyEntry {
            session: session.into(),
            class: class.into(),
            arrived: t(arrived),
            order_at: t(arrived),
            isl,
            served: 0,
            order: 0,
        }
    }

    fn kv(hbm: u64, hot: u64, total: u64) -> KvBudget {
        KvBudget {
            hbm_tokens: hbm,
            dram_tokens: 0,
            hot_tokens: hot,
            total_tokens: total,
            ..Default::default()
        }
    }

    #[test]
    fn a_higher_class_does_not_count_the_yield_class_kv() {
        let mut policy = ServicePolicy::default();
        policy.admission.yield_class = Some("batch".into());
        let priority = policy.class(Some("priority")).clone();
        let batch = policy.class(Some("batch")).clone();
        // HBM 1000, all of it hot, 800 of it batch's.
        let budget = KvBudget {
            yield_hot_tokens: 800,
            yield_total_tokens: 800,
            ..kv(1000, 1000, 1000)
        };
        let until = Time::from_secs_f64(60.0);
        // Priority sees 200 hot: a 300-token session fits.
        assert_eq!(
            fits(&policy, &priority, 300, 300, 0, &budget, until, 1.0),
            Decision::Admit(AdmitReason::Fits)
        );
        // Batch itself counts everything: it waits.
        assert!(matches!(
            fits(&policy, &batch, 300, 300, 0, &budget, until, 1.0),
            Decision::Hold { .. }
        ));
        // Without a yield class, priority waits too.
        policy.admission.yield_class = None;
        assert!(matches!(
            fits(&policy, &priority, 300, 300, 0, &budget, until, 1.0),
            Decision::Hold { .. }
        ));
    }

    #[test]
    fn hot_overcommit_lifts_the_hot_budget_but_not_the_total() {
        let mut policy = ServicePolicy::default();
        let lobby = [e("a", "standard", 0.0, 100)];
        // Standard's hot budget is 0.9 of HBM 1000 = 900: 850 hot leaves 50, under 100.
        let budget = KvBudget {
            dram_tokens: 4000,
            ..kv(1000, 850, 800)
        };
        assert!(matches!(
            decide(&policy, t(1.0), "a", &lobby, &budget),
            Decision::Hold {
                reason: HoldReason::KvBudget { hot_room: 50, .. },
                ..
            }
        ));
        // At 1.5 the hot budget is 1,350: room 500.
        policy.admission.hot_overcommit = 1.5;
        assert_eq!(
            decide(&policy, t(1.0), "a", &lobby, &budget),
            Decision::Admit(AdmitReason::Fits)
        );
        // The total budget (0.95 of 5,000 = 4,750) is still the wall.
        let full = KvBudget {
            dram_tokens: 4000,
            ..kv(1000, 850, 4700)
        };
        assert!(matches!(
            decide(&policy, t(1.0), "a", &lobby, &full),
            Decision::Hold {
                reason: HoldReason::KvBudget { total_room: 50, .. },
                ..
            }
        ));
        // Under 1 is refused.
        policy.admission.hot_overcommit = 0.5;
        assert!(policy.validate().is_err());
    }

    /// f49k2: `class_overcommit` {standard 1.3, batch 1.6} under `mode`. Raw hot budgets at HBM
    /// 1000: priority 1000, standard 0.9 x 1.3 = 1170, batch 0.75 x 1.6 = 1200.
    fn overcommitted(mode: Option<ClassOvercommitMode>) -> ServicePolicy {
        let mut policy = ServicePolicy::default();
        policy
            .admission
            .class_overcommit
            .extend([("standard".to_owned(), 1.3), ("batch".to_owned(), 1.6)]);
        policy.admission.class_overcommit_mode = mode;
        policy.validate().unwrap();
        policy
    }

    #[test]
    fn f49k2_priority_keeps_hot_room_over_the_overcommitted_classes() {
        // 1150 hot (1.15 x HBM), the total budget far off.
        let budget = KvBudget {
            dram_tokens: 100_000,
            ..kv(1000, 1150, 1150)
        };
        let until = t(60.0);
        let room_of = |p: &ServicePolicy, c: &str| room(p, p.class(Some(c)), &budget);
        let fit =
            |p: &ServicePolicy, c: &str| fits(p, p.class(Some(c)), 40, 40, 0, &budget, until, 1.0);
        // f49k (legacy): the lower classes are past priority's 1000; priority has no room.
        let legacy = overcommitted(Some(ClassOvercommitMode::Legacy));
        assert_eq!(room_of(&legacy, "priority"), 0);
        assert_eq!(room_of(&legacy, "standard"), 20);
        assert_eq!(room_of(&legacy, "batch"), 50);
        assert!(matches!(
            fit(&legacy, "priority"),
            Decision::Hold {
                reason: HoldReason::KvBudget { hot_room: 0, .. },
                ..
            }
        ));
        // max (the default, absent): every class has at least batch's 1200.
        for max in [
            overcommitted(None),
            overcommitted(Some(ClassOvercommitMode::Max)),
        ] {
            assert_eq!(room_of(&max, "priority"), 50);
            assert_eq!(room_of(&max, "standard"), 50);
            assert_eq!(room_of(&max, "batch"), 50);
            assert_eq!(fit(&max, "priority"), Decision::Admit(AdmitReason::Fits));
        }
        // separate: priority 1000 + 270 (standard's overcommit) + 450 (batch's); standard
        // 1170 + 450; batch, the lowest, its own.
        let separate = overcommitted(Some(ClassOvercommitMode::Separate));
        assert_eq!(room_of(&separate, "priority"), 570);
        assert_eq!(room_of(&separate, "standard"), 470);
        assert_eq!(room_of(&separate, "batch"), 50);
        assert_eq!(
            fit(&separate, "priority"),
            Decision::Admit(AdmitReason::Fits)
        );
        // Batch, the lowest class, is the same in every mode.
        for p in [&legacy, &separate] {
            assert_eq!(room_of(p, "batch"), 50);
        }
    }

    #[test]
    fn f49k2_max_keeps_a_class_room_that_tops_the_classes_below_it() {
        // Batch at 1.5 (1125) under standard's 1170: max leaves standard's room as it was.
        let mut max = overcommitted(None);
        max.admission.class_overcommit.insert("batch".into(), 1.5);
        let mut legacy = max.clone();
        legacy.admission.class_overcommit_mode = Some(ClassOvercommitMode::Legacy);
        for hot in [0, 500, 1000, 1150, 1169, 1170, 2000] {
            let budget = KvBudget {
                dram_tokens: 100_000,
                ..kv(1000, hot, hot)
            };
            let standard = max.class(Some("standard"));
            assert_eq!(
                room(&max, standard, &budget),
                room(&legacy, standard, &budget),
                "{hot}"
            );
            let batch = max.class(Some("batch"));
            assert_eq!(room(&max, batch, &budget), room(&legacy, batch, &budget));
        }
        // Priority's is lifted to standard's 1170.
        let priority = max.class(Some("priority"));
        assert_eq!(hot_cap(&max, priority, 1000, &|_| 1.0), 1170);
        assert_eq!(hot_cap(&legacy, priority, 1000, &|_| 1.0), 1000);
    }

    #[test]
    fn f49k2_hot_cap_takes_each_class_at_its_own_scale_and_skips_exempt_classes() {
        let mut p = overcommitted(None);
        let priority = p.class(Some("priority")).clone();
        // A step target of 1.5 on standard (1755) lifts priority with it; the scale of the class
        // decided applies to its own budget only.
        let step = |c: &ClassPolicy| if c.name == "standard" { 1.5 } else { 1.0 };
        assert_eq!(hot_cap(&p, &priority, 1000, &step), 1755);
        assert_eq!(hot_cap(&p, &priority, 1000, &|_| 1.0), 1200);
        let paced = |c: &ClassPolicy| if c.name == "priority" { 2.0 } else { 1.0 };
        assert_eq!(hot_cap(&p, &priority, 1000, &paced), 2000);
        // The scalar wrapper scales only the class decided.
        assert_eq!(
            room_scaled(&p, &priority, &kv(1000, 0, 0), 0.5),
            room_classes(&p, &priority, &kv(1000, 0, 0), &own_scale(&priority, 0.5))
        );
        // A class the hot budget leaves out lends nothing: priority gets standard's 1170.
        p.admission.hot_exempt_classes.push("batch".into());
        assert_eq!(hot_cap(&p, &priority, 1000, &|_| 1.0), 1170);
    }

    #[test]
    fn f49k2_without_class_overcommit_every_mode_is_the_raw_budget() {
        // Hot caps and decisions without `class_overcommit` are f49k's whatever the mode (the
        // mode is refused by validation then, but the cap ignores it anyway).
        let base = ServicePolicy::default();
        for mode in [
            None,
            Some(ClassOvercommitMode::Max),
            Some(ClassOvercommitMode::Separate),
            Some(ClassOvercommitMode::Legacy),
        ] {
            let mut p = base.clone();
            p.admission.class_overcommit_mode = mode;
            for c in &p.classes {
                for scale in [0.3, 1.0, 1.7] {
                    let want =
                        (c.admission.gate_share * scale * p.admission.hot_overcommit * 1000_f64)
                            as u64;
                    assert_eq!(hot_cap(&p, c, 1000, &|_| scale), want);
                }
                for hot in [0, 400, 900, 1000, 1500] {
                    let budget = kv(1000, hot, hot);
                    assert_eq!(room(&p, c, &budget), room(&base, c, &budget));
                    assert_eq!(
                        fits(&p, c, 100, 100, 50, &budget, t(1.0), 1.0),
                        fits(&base, c, 100, 100, 50, &budget, t(1.0), 1.0)
                    );
                }
            }
        }
    }

    #[test]
    fn f49k2_mode_parsing_and_validation() {
        for (text, want) in [
            ("max", ClassOvercommitMode::Max),
            ("separate", ClassOvercommitMode::Separate),
            ("legacy", ClassOvercommitMode::Legacy),
        ] {
            let p: ServicePolicy = serde_json::from_str(&format!(
                r#"{{"admission": {{"class_overcommit": {{"standard": 1.3}},
                                    "class_overcommit_mode": "{text}"}}}}"#
            ))
            .unwrap();
            p.validate().unwrap();
            assert_eq!(p.admission.class_overcommit_mode, Some(want), "{text}");
            let back: ServicePolicy =
                serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
            assert_eq!(back, p);
        }
        // Absent is `max`, and nothing new is written.
        let p: ServicePolicy =
            serde_json::from_str(r#"{"admission": {"class_overcommit": {"standard": 1.3}}}"#)
                .unwrap();
        assert_eq!(p.admission.class_overcommit_mode, None);
        assert_eq!(
            p.admission.class_overcommit_mode.unwrap_or_default(),
            ClassOvercommitMode::Max
        );
        assert!(
            !serde_json::to_string(&p)
                .unwrap()
                .contains("class_overcommit_mode")
        );
        assert!(
            !serde_json::to_string(&ServicePolicy::default())
                .unwrap()
                .contains("class_overcommit")
        );
        // An unknown mode does not parse; a mode without `class_overcommit` is refused.
        assert!(
            serde_json::from_str::<ServicePolicy>(
                r#"{"admission": {"class_overcommit": {"standard": 1.3},
                                  "class_overcommit_mode": "min"}}"#
            )
            .is_err()
        );
        let mut lone = ServicePolicy::default();
        lone.admission.class_overcommit_mode = Some(ClassOvercommitMode::Max);
        assert!(matches!(
            lone.validate(),
            Err(crate::config::ConfigError::ClassOvercommit(_))
        ));
    }

    #[test]
    fn admits_when_both_budgets_have_room() {
        let policy = ServicePolicy::default();
        let lobby = [e("a", "standard", 0.0, 100)];
        // standard: 0.9 of HBM 1000 = 900 hot; 0.95 of 1000 = 950 total.
        assert_eq!(
            decide(&policy, t(1.0), "a", &lobby, &kv(1000, 700, 800)),
            Decision::Admit(AdmitReason::Fits)
        );
        let held = decide(&policy, t(1.0), "a", &lobby, &kv(1000, 850, 800));
        assert_eq!(
            held,
            Decision::Hold {
                reason: HoldReason::KvBudget {
                    needed: 100,
                    hot_room: 50,
                    total_room: 150
                },
                until: t(60.0),
            }
        );
    }

    #[test]
    fn higher_class_is_reserved_first_and_classes_are_fifo() {
        let policy = ServicePolicy::default();
        let lobby = [
            e("s1", "standard", 0.0, 100),
            e("p1", "priority", 5.0, 300),
            e("s2", "standard", 1.0, 10),
        ];
        let room = kv(1000, 500, 500);
        // p1 ranks first: 1.0 * 1000 - 500 = 500 hot room, it fits.
        assert_eq!(
            decide(&policy, t(6.0), "p1", &lobby, &room),
            Decision::Admit(AdmitReason::Fits)
        );
        // s1 must leave room for p1 (300): 900 - 500 - 300 = 100 hot, 950 - 500 - 300 = 150 total.
        assert_eq!(
            decide(&policy, t(6.0), "s1", &lobby, &room),
            Decision::Admit(AdmitReason::Fits)
        );
        // s2 is behind s1 in its class even though it is small.
        assert_eq!(
            decide(&policy, t(6.0), "s2", &lobby, &room),
            Decision::Hold {
                reason: HoldReason::BehindEarlier { ahead: "s1".into() },
                until: t(61.0)
            }
        );
    }

    #[test]
    fn fair_order_serves_the_least_served_first() {
        let mut policy = ServicePolicy::default();
        policy.admission.lobby_order = LobbyOrder::Fair;
        let mut a = e("a", "standard", 0.0, 10);
        a.served = 5;
        let b = e("b", "standard", 3.0, 10);
        let mut lobby = vec![a, b];
        order_lobby(&policy, &mut lobby);
        assert_eq!(lobby[0].session, "b");
    }

    #[test]
    fn tenant_order_serves_the_tenant_furthest_under_its_share_first() {
        use crate::session::{LobbyOrderKeys, SessionState};
        let mut policy = ServicePolicy::default();
        policy.admission.lobby_order = LobbyOrder::Tenant;
        let mut table = SessionTable::new(64);
        table.set_lobby_order(LobbyOrderKeys {
            ranks: policy
                .classes
                .iter()
                .map(|c| (c.name.clone(), c.rank))
                .collect(),
            default_rank: policy.class(None).rank,
            fair: false,
            aging: false,
            session_age: false,
        });
        // Tenant A already has two admitted sessions, tenant B none.
        for id in ["a-live1", "a-live2"] {
            table.on_turn(id, None, "standard", Some("A"), t(0.0), 10);
            table.update(id, |s| s.state = SessionState::Idle);
        }
        assert_eq!(table.tenant_live("standard", "A"), 2);
        table.on_turn("a-wait", None, "standard", Some("A"), t(1.0), 10);
        table.on_turn("b-wait", None, "standard", Some("B"), t(2.0), 10);
        let room = kv(1000, 0, 0);
        let a = table.get("a-wait").unwrap().clone();
        let b = table.get("b-wait").unwrap().clone();
        // First come would serve a-wait; tenant order serves B (0 live) first.
        assert_eq!(
            decide_indexed(&policy, t(3.0), &b, &table, &room),
            Decision::Admit(AdmitReason::Fits)
        );
        assert!(matches!(
            decide_indexed(&policy, t(3.0), &a, &table, &room),
            Decision::Hold { reason: HoldReason::BehindEarlier { ref ahead }, .. } if ahead == "b-wait"
        ));
        // A share of 4 puts A at 0.5; with B at two admitted sessions (2.0) A goes first.
        policy.admission.tenant_shares.insert("A".into(), 4.0);
        for id in ["b-live1", "b-live2"] {
            table.on_turn(id, None, "standard", Some("B"), t(0.0), 10);
            table.update(id, |s| s.state = SessionState::Idle);
        }
        assert_eq!(
            decide_indexed(&policy, t(3.0), &a, &table, &room),
            Decision::Admit(AdmitReason::Fits)
        );
        // First come again, the same table serves a-wait first.
        policy.admission.lobby_order = LobbyOrder::Fifo;
        assert!(matches!(
            decide_indexed(&policy, t(3.0), &b, &table, &room),
            Decision::Hold { .. }
        ));
    }

    #[test]
    fn wait_limits_admit_priority_and_expire_the_others() {
        let policy = ServicePolicy::default();
        let full = kv(1000, 1000, 1000);
        let lobby = [e("p", "priority", 0.0, 10), e("s", "standard", 0.0, 10)];
        assert!(matches!(
            decide(&policy, t(29.0), "p", &lobby, &full),
            Decision::Hold { .. }
        ));
        assert_eq!(
            decide(&policy, t(30.0), "p", &lobby, &full),
            Decision::Admit(AdmitReason::WaitLimit)
        );
        assert_eq!(
            decide(&policy, t(61.0), "s", &[e("s", "standard", 0.0, 10)], &full),
            Decision::Expire {
                waited: Duration::from_secs(61)
            }
        );
    }

    #[test]
    fn shadow_lets_everything_go_and_enforce_holds() {
        let hold = Decision::Hold {
            reason: HoldReason::BehindEarlier { ahead: "x".into() },
            until: t(9.0),
        };
        assert_eq!(gate_answer(Mode::Shadow, &hold), GateAnswer::Go);
        assert_eq!(gate_answer(Mode::Off, &hold), GateAnswer::Go);
        assert_eq!(
            gate_answer(Mode::Enforce, &hold),
            GateAnswer::Hold {
                until: Some(t(9.0))
            }
        );
        assert_eq!(
            gate_answer(Mode::Enforce, &Decision::Admit(AdmitReason::Fits)),
            GateAnswer::Go
        );
        assert_eq!(
            gate_answer(
                Mode::Enforce,
                &Decision::Expire {
                    waited: Duration::ZERO
                }
            ),
            GateAnswer::Hold { until: None }
        );
    }

    #[test]
    fn decision_log_dedupes_and_stays_bounded() {
        let mut log = DecisionLog::new(2);
        let d = |s: &str, decision: Decision| LoggedDecision {
            at: t(0.0),
            session: s.into(),
            class: "standard".into(),
            mode: Mode::Shadow,
            decision,
        };
        assert!(log.record(d("a", Decision::Admit(AdmitReason::Fits))));
        assert!(!log.record(d("a", Decision::Admit(AdmitReason::Fits))));
        assert!(log.record(d("b", Decision::Admit(AdmitReason::Fits))));
        assert!(log.record(d("c", Decision::Admit(AdmitReason::Fits))));
        assert_eq!(log.len(), 2);
        log.forget("a");
        assert!(log.record(d("a", Decision::Admit(AdmitReason::Fits))));
    }
}
