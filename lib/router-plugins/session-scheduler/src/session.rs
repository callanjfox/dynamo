// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The session table: what the scheduler knows about each session. Bounded; the least recently
//! touched idle session is dropped first. Every update is O(log n) or better, so it is safe on
//! the classifier's lock.
//!
//! The table also keeps, incrementally, what the gate reads (ledger L65): the KV load (context of
//! live sessions, and of those with a turn in flight) and the lobby in service order per class
//! with each class's total input tokens. Every change goes through [`SessionTable::on_turn`],
//! [`SessionTable::update`] or eviction, which take a session's contribution out before the
//! change and put it back after, so the gate's check and the throttle never scan the table.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::clock::Time;
use crate::config::{HotDemand, LifetimeBudget};

/// Where a session is in its life.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionState {
    /// Waiting for admission.
    Lobby,
    /// Admitted, with a turn in flight.
    Active,
    /// Admitted, between turns.
    Idle,
    /// Finished (the client said so) or expired.
    Ended,
    /// Rotated out at a stop point (`admission.rotate`): not live; its next turn waits in the
    /// lobby to resume and is never refused.
    Paused,
}

/// What the client or a history estimate says about the session's next turn.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Signals {
    pub next_think_s: Option<f64>,
    pub osl_tokens: Option<u32>,
    /// New prompt tokens the client expects its next turn to add.
    pub next_turn_tokens: Option<u64>,
    /// The largest context the client expects the session to reach from here
    /// (`x-dynamo-hint-peak-tokens`, F22; read only with `admission.lifetime`).
    pub peak_tokens: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Session {
    pub id: String,
    pub parent: Option<String>,
    pub class: String,
    pub tenant: Option<String>,
    pub state: SessionState,
    pub arrived: Time,
    pub admitted: Option<Time>,
    /// The latest turn's arrival, used to reconcile a completion that lands after the next
    /// turn has already been classified.
    pub last_turn_arrival: Time,
    pub last_turn_end: Option<Time>,
    pub in_flight: u32,
    /// Context tokens of the latest finished turn (the KV it leaves behind), or the latest
    /// turn's input when no turn has finished.
    pub context_tokens: u64,
    /// Input tokens of the turn waiting or in flight.
    pub pending_isl: u64,
    pub turns_served: u32,
    pub signals: Signals,
    /// Waiting in the lobby to resume after a rotation.
    pub resumed: bool,
    /// When it was last paused, while paused or waiting to resume.
    pub paused_at: Option<Time>,
    /// Seconds spent paused so far (not counted as run time).
    pub paused_s: f64,
    /// Waiting in the lobby without the deadline that would expire it at its wait limit: it was
    /// classified while the adaptive rule had the session gate off.
    pub no_lobby_deadline: bool,
    /// Seconds from dispatch to completion of the session's last finished turn
    /// (`admission.hot_demand: next_turn`).
    pub last_service_s: Option<f64>,
    /// When the current wait for admission began (F32): set when the session enters the lobby
    /// and kept when its wait limit expires it and its client retries (`lobby_expired`), cleared
    /// at admission.
    pub queued_since: Option<Time>,
    /// Lobby seconds of earlier waits that ended in admission (before a rotation paused it).
    pub waited_s: f64,
    /// Left the lobby because its wait limit expired it: its client's retry continues the wait.
    pub lobby_expired: bool,
    /// The latest admission out of the lobby (first admission or resuming after a rotation).
    pub last_admitted: Option<Time>,
    /// The Retry-After the latest attempt was given with its deadline, seconds
    /// (`admission.retry_after_estimate`): an away session's place is held that much longer.
    pub retry_after_s: Option<f64>,
    /// The session's first offer to the router: never reset by retries, rotations or waits
    /// (`admission.turn_order: session_age`, F35).
    pub started: Time,
    /// Interactive or agent, from `x-dynamo-hint-session-kind` (F35); not read by any policy yet.
    pub kind: Option<SessionKind>,
    /// The user the session belongs to, from `x-dynamo-user-id` (F35); a user can have several
    /// sessions. Not read by any policy yet.
    pub user: Option<String>,
    /// The lobby ordering key of its waiting request (`turn_order: run_las` or
    /// `server_wait_deficit`, f36b; lower first). `None` with the other orders.
    pub order_key: Option<f64>,
}

/// An ordering key in millionths, as the lobby orders by it (0 without one).
pub fn order_units(key: Option<f64>) -> i64 {
    key.map_or(0, |k| (k * 1e6).round() as i64)
}

/// Whether a person is waiting on the session's turns or an agent runs it (F35).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SessionKind {
    Interactive,
    Agent,
}

impl SessionKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "interactive" => Some(Self::Interactive),
            "agent" => Some(Self::Agent),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Interactive => "interactive",
            Self::Agent => "agent",
        }
    }
}

impl Session {
    pub fn is_live(&self) -> bool {
        matches!(self.state, SessionState::Active | SessionState::Idle)
    }

    /// Seconds the current wait for admission has lasted at `now` (across expiries and
    /// retries), or 0 when not waiting.
    pub fn lobby_wait_s(&self, now: Time) -> f64 {
        self.queued_since
            .map_or(0.0, |q| now.since(q).as_secs_f64())
    }

    /// Expired out of the lobby and not yet back: its client is waiting to retry.
    pub fn away(&self) -> bool {
        self.state == SessionState::Ended && self.lobby_expired && self.queued_since.is_some()
    }

    /// The time `lobby_order: aging` orders the session by: the current wait's start, moved back
    /// by earlier waits.
    pub fn aged_since(&self) -> Time {
        let since = self.queued_since.unwrap_or(self.arrived).as_secs_f64();
        Time::from_secs_f64((since - self.waited_s).max(0.0))
    }
}

/// Where a waiting session sits in the lobby's service order: class rank, then (with fair order)
/// turns served, then arrival, then id. This is the order of [`crate::admit::order_lobby`].
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LobbyKey {
    pub rank: u32,
    /// Turns served with fair order; 0 with first-come order.
    pub served: u32,
    /// The ordering key in millionths ([`Session::order_key`]); 0 without one.
    pub order: i64,
    pub arrived: Time,
    pub session: String,
}

/// The lobby in service order, per class, with each class's total input tokens.
#[derive(Debug, Default)]
pub struct LobbyIndex {
    by_class: HashMap<String, BTreeMap<LobbyKey, u64>>,
    class_isl: HashMap<String, u64>,
    /// Per class, per tenant, the waiting sessions in order (for `lobby_order: tenant`).
    by_tenant: HashMap<String, HashMap<String, BTreeSet<LobbyKey>>>,
    len: usize,
}

impl LobbyIndex {
    fn insert(&mut self, class: &str, tenant: &str, key: LobbyKey, isl: u64) {
        self.by_tenant
            .entry(class.to_owned())
            .or_default()
            .entry(tenant.to_owned())
            .or_default()
            .insert(key.clone());
        if self
            .by_class
            .entry(class.to_owned())
            .or_default()
            .insert(key, isl)
            .is_none()
        {
            *self.class_isl.entry(class.to_owned()).or_default() += isl;
            self.len += 1;
        }
    }

    fn remove(&mut self, class: &str, tenant: &str, key: &LobbyKey) {
        if let Some(tenants) = self.by_tenant.get_mut(class) {
            if let Some(set) = tenants.get_mut(tenant) {
                set.remove(key);
                if set.is_empty() {
                    tenants.remove(tenant);
                }
            }
            if tenants.is_empty() {
                self.by_tenant.remove(class);
            }
        }
        let Some(map) = self.by_class.get_mut(class) else {
            return;
        };
        if let Some(isl) = map.remove(key) {
            if let Some(sum) = self.class_isl.get_mut(class) {
                *sum -= isl;
            }
            self.len -= 1;
        }
        if map.is_empty() {
            self.by_class.remove(class);
            self.class_isl.remove(class);
        }
    }

    /// Sessions waiting.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether any session of `class` is waiting.
    pub fn has_class(&self, class: &str) -> bool {
        self.by_class.get(class).is_some_and(|m| !m.is_empty())
    }

    /// f49m: sessions of `class` waiting.
    pub fn class_len(&self, class: &str) -> usize {
        self.by_class.get(class).map_or(0, |m| m.len())
    }

    /// f49m: input tokens of the waiting sessions of the classes ranked above `rank`.
    pub fn isl_ranked_above(&self, rank: u32) -> u64 {
        self.by_class
            .iter()
            .filter(|(_, m)| m.first_key_value().is_some_and(|(k, _)| k.rank < rank))
            .map(|(c, _)| self.class_isl.get(c).copied().unwrap_or(0))
            .sum()
    }

    /// Sessions of `class` ahead of `key` in service order. O(ahead).
    pub fn place_in_class(&self, class: &str, key: &LobbyKey) -> usize {
        self.by_class
            .get(class)
            .map_or(0, |m| m.range(..key).count())
    }

    /// The nearest session of the same class ahead of `key` that `skip` does not pass over.
    /// O(skipped).
    pub fn earlier_in_class_unless(
        &self,
        class: &str,
        key: &LobbyKey,
        skip: impl Fn(&LobbyKey) -> bool,
    ) -> Option<&LobbyKey> {
        self.by_class
            .get(class)?
            .range(..key)
            .rev()
            .map(|(k, _)| k)
            .find(|k| !skip(k))
    }

    /// The nearest session of the same class ahead of `key`.
    pub fn earlier_in_class(&self, class: &str, key: &LobbyKey) -> Option<&LobbyKey> {
        self.by_class
            .get(class)?
            .range(..key)
            .next_back()
            .map(|(k, _)| k)
    }

    /// The nearest session of the same class and tenant ahead of `key`. With `lobby_order:
    /// tenant` a session waits behind it whatever the tenants' loads.
    pub fn earlier_in_tenant(
        &self,
        class: &str,
        tenant: &str,
        key: &LobbyKey,
    ) -> Option<&LobbyKey> {
        self.by_tenant
            .get(class)?
            .get(tenant)?
            .range(..key)
            .next_back()
    }

    /// With `lobby_order: tenant`: the first session of the same class ahead of `key` (tenant
    /// `tenant` at load `load(tenant)`), ordered by tenant load, then the key. O(tenants).
    pub fn earlier_in_class_by_tenant(
        &self,
        class: &str,
        tenant: &str,
        key: &LobbyKey,
        load: impl Fn(&str) -> f64,
    ) -> Option<&LobbyKey> {
        let mine = load(tenant);
        let mut best: Option<(f64, &LobbyKey)> = None;
        for (t, set) in self.by_tenant.get(class)? {
            let (l, first) = if t == tenant {
                // Within a tenant, first come.
                (mine, set.range(..key).next())
            } else {
                (load(t), set.first())
            };
            let Some(first) = first else { continue };
            let ahead = l < mine || (l == mine && first < key);
            if ahead && best.is_none_or(|(bl, bk)| l < bl || (l == bl && first < bk)) {
                best = Some((l, first));
            }
        }
        best.map(|(_, k)| k)
    }

    /// Input tokens of the sessions of other classes ahead of `key`. O(classes) when ranks
    /// differ; a class that shares the key's rank is walked up to the key.
    pub fn isl_ahead_in_other_classes(&self, class: &str, key: &LobbyKey) -> u64 {
        let mut ahead = 0;
        for (other, map) in &self.by_class {
            if other == class {
                continue;
            }
            let Some((first, _)) = map.first_key_value() else {
                continue;
            };
            // Every key in a class map has the class's rank.
            match first.rank.cmp(&key.rank) {
                std::cmp::Ordering::Less => {
                    ahead += self.class_isl.get(other).copied().unwrap_or(0)
                }
                std::cmp::Ordering::Greater => {}
                std::cmp::Ordering::Equal => {
                    ahead += map.range(..key).map(|(_, isl)| isl).sum::<u64>()
                }
            }
        }
        ahead
    }

    /// Every waiting session in service order (for inspection and tests; O(n log n)).
    pub fn ordered(&self) -> Vec<(&str, &LobbyKey, u64)> {
        let mut all: Vec<_> = self
            .by_class
            .iter()
            .flat_map(|(class, map)| map.iter().map(move |(k, isl)| (class.as_str(), k, *isl)))
            .collect();
        all.sort_by(|a, b| a.1.cmp(b.1));
        all
    }
}

/// What one session adds to the table's running totals.
#[derive(Clone, Debug, PartialEq)]
struct Contribution {
    total: u64,
    hot: u64,
    /// Of the yield class.
    yields: bool,
    lobby: Option<(String, String, LobbyKey, u64)>,
    /// (class, tenant) of a live top-level session, for tenant fair share.
    live_tenant: Option<(String, String)>,
}

/// How the lobby is ordered: class ranks and whether fair order counts turns served.
#[derive(Clone, Debug, Default)]
pub struct LobbyOrderKeys {
    pub ranks: HashMap<String, u32>,
    /// Rank of a class not in `ranks` (the policy's default class).
    pub default_rank: u32,
    pub fair: bool,
    /// Order by [`Session::aged_since`] instead of the latest arrival (`lobby_order: aging`).
    pub aging: bool,
    /// Order by [`Session::started`], the session's first offer (`admission.turn_order:
    /// session_age`); takes precedence over `aging`.
    pub session_age: bool,
}

/// Bounded session table with least-recently-used eviction.
#[derive(Debug)]
pub struct SessionTable {
    capacity: usize,
    entries: HashMap<String, (Session, u64)>,
    order: BTreeMap<u64, String>,
    seq: u64,
    evicted: u64,
    keys: LobbyOrderKeys,
    total_tokens: u64,
    hot_tokens: u64,
    /// Totals of the yield class's sessions (`set_yield_class`).
    yield_class: Option<String>,
    yield_total_tokens: u64,
    yield_hot_tokens: u64,
    /// What a live session counts toward `hot_tokens` (`set_hot_demand`).
    hot_demand: HotDemand,
    /// Classes whose sessions count nothing toward `hot_tokens` (`set_hot_exempt`).
    hot_exempt: Vec<String>,
    /// Whether sessions count at their hinted peak (`set_lifetime`).
    lifetime: LifetimeBudget,
    lobby: LobbyIndex,
    /// Live top-level sessions per (class, tenant).
    tenant_live: HashMap<(String, String), u64>,
    /// Lobby changes since the last [`Self::take_lobby_changes`], kept only once
    /// [`Self::track_lobby_changes`] is called (the dispatch gate's keyed holds).
    lobby_changes: Option<LobbyChanges>,
}

/// Sessions whose place in the lobby changed: each left the lobby or moved in its order.
/// `reset` when the whole lobby was rebuilt.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct LobbyChanges {
    pub sessions: Vec<String>,
    pub reset: bool,
}

impl LobbyChanges {
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty() && !self.reset
    }
}

impl SessionTable {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: HashMap::new(),
            order: BTreeMap::new(),
            seq: 0,
            evicted: 0,
            keys: LobbyOrderKeys::default(),
            total_tokens: 0,
            hot_tokens: 0,
            yield_class: None,
            yield_total_tokens: 0,
            yield_hot_tokens: 0,
            hot_demand: HotDemand::InFlight,
            hot_exempt: Vec::new(),
            lifetime: LifetimeBudget::Off,
            lobby: LobbyIndex::default(),
            tenant_live: HashMap::new(),
            lobby_changes: None,
        }
    }

    /// Start recording which sessions leave the lobby or move in it.
    pub fn track_lobby_changes(&mut self) {
        self.lobby_changes.get_or_insert_with(LobbyChanges::default);
    }

    /// The lobby changes since the last call (empty when not tracking).
    pub fn take_lobby_changes(&mut self) -> LobbyChanges {
        self.lobby_changes
            .as_mut()
            .map(std::mem::take)
            .unwrap_or_default()
    }

    /// Record that `id`'s lobby place changed from `before` to `after` (class, tenant, key).
    fn note_lobby_change(
        &mut self,
        id: &str,
        before: Option<&(String, String, LobbyKey, u64)>,
        after: Option<&(String, String, LobbyKey, u64)>,
    ) {
        let Some(changes) = self.lobby_changes.as_mut() else {
            return;
        };
        fn place(
            l: Option<&(String, String, LobbyKey, u64)>,
        ) -> Option<(&String, &String, &LobbyKey)> {
            l.map(|(c, t, k, _)| (c, t, k))
        }
        if before.is_some() && place(before) != place(after) {
            changes.sessions.push(id.to_owned());
        }
    }

    /// Share of live top-level sessions that are of `class` (0 with none live). O(tenants).
    pub fn live_share(&self, class: &str) -> f64 {
        let (mut mine, mut all) = (0u64, 0u64);
        for ((c, _), n) in &self.tenant_live {
            all += n;
            if c == class {
                mine += n;
            }
        }
        if all == 0 {
            0.0
        } else {
            mine as f64 / all as f64
        }
    }

    /// Live top-level sessions of `class` and `tenant` ("" for none).
    pub fn tenant_live(&self, class: &str, tenant: &str) -> u64 {
        self.tenant_live
            .get(&(class.to_owned(), tenant.to_owned()))
            .copied()
            .unwrap_or(0)
    }

    /// Set the lobby's order (class ranks, fair or first come) and rebuild the running totals.
    pub fn set_lobby_order(&mut self, keys: LobbyOrderKeys) {
        self.keys = keys;
        self.total_tokens = 0;
        self.hot_tokens = 0;
        self.yield_total_tokens = 0;
        self.yield_hot_tokens = 0;
        self.lobby = LobbyIndex::default();
        if let Some(changes) = self.lobby_changes.as_mut() {
            changes.reset = true;
        }
        self.tenant_live.clear();
        let all: Vec<Contribution> = self.iter().map(|s| self.contribution(s)).collect();
        for c in &all {
            self.add(c);
        }
    }

    pub fn rank_of(&self, class: &str) -> u32 {
        self.keys
            .ranks
            .get(class)
            .copied()
            .unwrap_or(self.keys.default_rank)
    }

    /// The lobby key of a waiting session, or `None` when it is not in the lobby. With `lobby_order:
    /// aging` a session whose wait limit expired it keeps its place while its client is away
    /// ([`Session::away`]): the gate passes over it once its retry is overdue.
    pub fn lobby_key(&self, session: &Session) -> Option<LobbyKey> {
        (session.state == SessionState::Lobby || (self.keys.aging && session.away())).then(|| {
            LobbyKey {
                rank: self.rank_of(&session.class),
                served: if self.keys.fair {
                    session.turns_served
                } else {
                    0
                },
                order: order_units(session.order_key),
                arrived: self.order_at(session),
                session: session.id.clone(),
            }
        })
    }

    /// The arrival the lobby orders `session` by: its latest arrival, or with `lobby_order:
    /// aging` the start of its wait counting retries and earlier waits.
    pub fn order_at(&self, session: &Session) -> Time {
        if self.keys.session_age {
            session.started
        } else if self.keys.aging {
            session.aged_since()
        } else {
            session.arrived
        }
    }

    fn contribution(&self, s: &Session) -> Contribution {
        let live = s.is_live();
        Contribution {
            yields: self.yield_class.as_deref() == Some(s.class.as_str()),
            total: if live { self.total_of(s) } else { 0 },
            hot: if live { self.hot_of(s) } else { 0 },
            lobby: self.lobby_key(s).map(|key| {
                (
                    s.class.clone(),
                    s.tenant.clone().unwrap_or_default(),
                    key,
                    s.pending_isl,
                )
            }),
            live_tenant: (live && s.parent.is_none())
                .then(|| (s.class.clone(), s.tenant.clone().unwrap_or_default())),
        }
    }

    /// The context live session `s` counts as (`admission.lifetime`): its context now, or the
    /// larger of that and its hinted peak.
    fn sized(&self, s: &Session, lifetime_on: bool) -> u64 {
        match s.signals.peak_tokens {
            Some(peak) if lifetime_on => s.context_tokens.max(peak),
            _ => s.context_tokens,
        }
    }

    /// What live session `s` counts toward the total budget.
    fn total_of(&self, s: &Session) -> u64 {
        self.sized(s, self.lifetime != LifetimeBudget::Off)
    }

    /// What a session waiting in the lobby needs from the total budget: its first turn's input,
    /// or with `admission.lifetime` the larger of that and its hinted peak.
    pub fn total_need(&self, s: &Session) -> u64 {
        match s.signals.peak_tokens {
            Some(peak) if self.lifetime != LifetimeBudget::Off => s.pending_isl.max(peak),
            _ => s.pending_isl,
        }
    }

    /// Count sessions at their hinted peak (`admission.lifetime`). Set before any session.
    pub fn set_lifetime(&mut self, lifetime: LifetimeBudget) {
        self.lifetime = lifetime;
    }

    /// Classes whose sessions count nothing toward the hot budget (`admission.hot_exempt_classes`).
    /// Set before any session is added.
    pub fn set_hot_exempt(&mut self, classes: Vec<String>) {
        debug_assert!(self.total_tokens == 0, "set before sessions are added");
        self.hot_exempt = classes;
    }

    /// What live session `s` counts toward the hot budget (`admission.hot_demand`).
    fn hot_of(&self, s: &Session) -> u64 {
        if self.hot_exempt.contains(&s.class) {
            return 0;
        }
        let hot_peak = self.lifetime == LifetimeBudget::Both;
        let context = self.sized(s, hot_peak);
        match self.hot_demand {
            HotDemand::InFlight => {
                if s.in_flight > 0 {
                    context.max(s.pending_isl)
                } else {
                    0
                }
            }
            HotDemand::NextTurn => {
                let osl = u64::from(s.signals.osl_tokens.unwrap_or(0));
                if s.in_flight > 0 {
                    return context.max(s.pending_isl) + osl;
                }
                let (Some(service), Some(think)) = (s.last_service_s, s.signals.next_think_s)
                else {
                    return 0;
                };
                let next =
                    (s.context_tokens + s.signals.next_turn_tokens.unwrap_or(0)).max(context) + osl;
                let service = service.max(0.0);
                let duty = service / (service + think.max(0.1));
                (next as f64 * duty) as u64
            }
        }
    }

    fn add(&mut self, c: &Contribution) {
        self.total_tokens += c.total;
        self.hot_tokens += c.hot;
        if c.yields {
            self.yield_total_tokens += c.total;
            self.yield_hot_tokens += c.hot;
        }
        if let Some((class, tenant, key, isl)) = &c.lobby {
            self.lobby.insert(class, tenant, key.clone(), *isl);
        }
        if let Some(k) = &c.live_tenant {
            *self.tenant_live.entry(k.clone()).or_default() += 1;
        }
    }

    fn take(&mut self, c: &Contribution) {
        self.total_tokens -= c.total;
        self.hot_tokens -= c.hot;
        if c.yields {
            self.yield_total_tokens -= c.total;
            self.yield_hot_tokens -= c.hot;
        }
        if let Some((class, tenant, key, _)) = &c.lobby {
            self.lobby.remove(class, tenant, key);
        }
        if let Some(k) = &c.live_tenant
            && let Some(n) = self.tenant_live.get_mut(k)
        {
            *n -= 1;
            if *n == 0 {
                self.tenant_live.remove(k);
            }
        }
    }

    /// The lobby in service order, kept incrementally.
    pub fn lobby(&self) -> &LobbyIndex {
        &self.lobby
    }

    /// Sessions waiting in the lobby. O(1).
    pub fn lobby_len(&self) -> usize {
        self.lobby.len()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Sessions dropped to stay within capacity.
    pub fn evicted(&self) -> u64 {
        self.evicted
    }

    pub fn get(&self, id: &str) -> Option<&Session> {
        self.entries.get(id).map(|(s, _)| s)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Session> {
        self.entries.values().map(|(s, _)| s)
    }

    fn touch(&mut self, id: &str) {
        if let Some((_, seq)) = self.entries.get_mut(id) {
            self.order.remove(seq);
            self.seq += 1;
            *seq = self.seq;
            self.order.insert(self.seq, id.to_owned());
        }
    }

    /// Update a session, marking it recently used. Its share of the running totals is taken out
    /// before `f` and put back after.
    pub fn update<R>(&mut self, id: &str, f: impl FnOnce(&mut Session) -> R) -> Option<R> {
        let before = self.entries.get(id).map(|(s, _)| self.contribution(s))?;
        self.take(&before);
        let result = self.entries.get_mut(id).map(|(s, _)| f(s));
        let after = self.entries.get(id).map(|(s, _)| self.contribution(s));
        if let Some(after) = &after {
            self.add(after);
        }
        self.note_lobby_change(
            id,
            before.lobby.as_ref(),
            after.as_ref().and_then(|a| a.lobby.as_ref()),
        );
        if result.is_some() {
            self.touch(id);
        }
        result
    }

    /// Whether a session's `parent` is admitted (active or between turns). A subagent takes its
    /// parent's admission: it never waits in the lobby or expires as a new session while its
    /// parent is admitted, as in the admission simulator, where subagents are streams of the
    /// parent's session.
    fn parent_admitted(&self, parent: Option<&str>) -> bool {
        parent
            .and_then(|p| self.entries.get(p))
            .is_some_and(|(s, _)| matches!(s.state, SessionState::Active | SessionState::Idle))
    }

    /// A turn of `id` arrived: create the session in the lobby if it is new, otherwise record
    /// the turn. A subagent (a session with an admitted `parent`) starts admitted instead.
    /// Returns the session's state after the update.
    pub fn on_turn(
        &mut self,
        id: &str,
        parent: Option<&str>,
        class: &str,
        tenant: Option<&str>,
        now: Time,
        isl: u64,
    ) -> SessionState {
        let with_parent = self.parent_admitted(parent);
        if let Some(state) = self.update(id, |s| {
            s.last_turn_arrival = now;
            s.pending_isl = isl;
            if s.state != SessionState::Lobby {
                s.no_lobby_deadline = false;
            }
            if s.state == SessionState::Paused {
                // Back from a rotation: wait in the lobby to resume.
                s.state = SessionState::Lobby;
                s.resumed = true;
                s.arrived = now;
                s.queued_since = Some(now);
            }
            if s.state == SessionState::Ended {
                if with_parent {
                    s.state = SessionState::Idle;
                    s.admitted = Some(now);
                    s.queued_since = None;
                    s.last_admitted = Some(now);
                } else {
                    s.state = SessionState::Lobby;
                    s.admitted = None;
                    // A retry after a wait-limit expiry continues the same wait.
                    if !s.lobby_expired || s.queued_since.is_none() {
                        s.queued_since = Some(now);
                        s.waited_s = 0.0;
                    }
                }
                s.lobby_expired = false;
                s.arrived = now;
            }
            s.state
        }) {
            return state;
        }
        self.make_room();
        self.seq += 1;
        let state = if with_parent {
            SessionState::Idle
        } else {
            SessionState::Lobby
        };
        let session = Session {
            id: id.to_owned(),
            parent: parent.map(str::to_owned),
            class: class.to_owned(),
            tenant: tenant.map(str::to_owned),
            state,
            arrived: now,
            admitted: with_parent.then_some(now),
            last_turn_arrival: now,
            last_turn_end: None,
            in_flight: 0,
            context_tokens: isl,
            pending_isl: isl,
            turns_served: 0,
            signals: Signals::default(),
            resumed: false,
            paused_at: None,
            paused_s: 0.0,
            no_lobby_deadline: false,
            last_service_s: None,
            queued_since: (state == SessionState::Lobby).then_some(now),
            waited_s: 0.0,
            lobby_expired: false,
            last_admitted: with_parent.then_some(now),
            retry_after_s: None,
            started: now,
            kind: None,
            user: None,
            order_key: None,
        };
        let contribution = self.contribution(&session);
        self.add(&contribution);
        self.entries.insert(id.to_owned(), (session, self.seq));
        self.order.insert(self.seq, id.to_owned());
        state
    }

    /// Drop the least recently used session that has nothing in flight; if every session has a
    /// turn in flight, drop the least recently used one anyway (memory stays bounded).
    fn make_room(&mut self) {
        while self.entries.len() >= self.capacity {
            let victim = self
                .order
                .iter()
                .find(|(_, id)| self.entries.get(*id).is_some_and(|(s, _)| s.in_flight == 0))
                .or_else(|| self.order.iter().next())
                .map(|(seq, id)| (*seq, id.clone()));
            let Some((seq, id)) = victim else { return };
            self.order.remove(&seq);
            if let Some((session, _)) = self.entries.remove(&id) {
                let contribution = self.contribution(&session);
                self.take(&contribution);
                self.note_lobby_change(&id, contribution.lobby.as_ref(), None);
            }
            self.evicted += 1;
        }
    }

    /// Sum of context tokens over live sessions, and over those with a turn in flight. O(1): kept
    /// incrementally.
    pub fn kv_load(&self) -> (u64, u64) {
        (self.total_tokens, self.hot_tokens)
    }

    /// The yield class's share of [`Self::kv_load`]: (total, hot).
    pub fn yield_load(&self) -> (u64, u64) {
        (self.yield_total_tokens, self.yield_hot_tokens)
    }

    /// Track the totals of `class`'s sessions apart (`admission.yield_class`). Set before any
    /// session is added.
    pub fn set_yield_class(&mut self, class: Option<String>) {
        debug_assert!(self.total_tokens == 0, "set before sessions are added");
        self.yield_class = class;
    }

    /// What a live session counts toward the hot budget (`admission.hot_demand`). Set before any
    /// session is added.
    pub fn set_hot_demand(&mut self, mode: HotDemand) {
        debug_assert!(self.total_tokens == 0, "set before sessions are added");
        self.hot_demand = mode;
    }

    /// The same totals by a full scan, for tests that check the incremental ones.
    pub fn kv_load_scan(&self) -> (u64, u64) {
        let mut total = 0;
        let mut hot = 0;
        for session in self.iter().filter(|s| s.is_live()) {
            total += session.context_tokens;
            hot += self.hot_of(session);
        }
        (total, hot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: f64) -> Time {
        Time::from_secs_f64(s)
    }

    #[test]
    fn lobby_changes_name_sessions_that_leave_or_move_only_while_tracked() {
        let mut table = SessionTable::new(3);
        table.on_turn("a", None, "standard", None, t(0.0), 100);
        table.update("a", |s| s.state = SessionState::Active);
        assert!(table.take_lobby_changes().is_empty(), "not tracked");
        table.track_lobby_changes();
        table.on_turn("b", None, "standard", None, t(1.0), 100);
        table.on_turn("c", None, "standard", None, t(2.0), 100);
        // Joining the lobby, or a new turn that keeps the session's place, moves nothing.
        table.on_turn("b", None, "standard", None, t(3.0), 500);
        assert!(table.take_lobby_changes().is_empty());
        // Leaving the lobby is a change.
        table.update("b", |s| s.state = SessionState::Active);
        assert_eq!(table.take_lobby_changes().sessions, vec!["b".to_string()]);
        // An eviction from the lobby is a change (the table holds 3: a, b, c; d evicts the least
        // recently used with nothing in flight, c, once a is used again).
        table.update("a", |_| ());
        assert!(table.take_lobby_changes().is_empty());
        table.on_turn("d", None, "standard", None, t(4.0), 100);
        assert_eq!(table.take_lobby_changes().sessions, vec!["c".to_string()]);
        // A rebuilt lobby resets every place.
        table.set_lobby_order(LobbyOrderKeys::default());
        assert!(table.take_lobby_changes().reset);
    }

    #[test]
    fn new_sessions_start_in_the_lobby_and_return_turns_keep_state() {
        let mut table = SessionTable::new(4);
        assert_eq!(
            table.on_turn("a", None, "standard", None, t(0.0), 100),
            SessionState::Lobby
        );
        table.update("a", |s| s.state = SessionState::Idle);
        assert_eq!(
            table.on_turn("a", None, "standard", None, t(5.0), 120),
            SessionState::Idle
        );
        let s = table.get("a").unwrap();
        assert_eq!((s.pending_isl, s.last_turn_arrival), (120, t(5.0)));
    }

    #[test]
    fn a_subagent_takes_its_parents_admission() {
        let mut table = SessionTable::new(8);
        // Parent in the lobby: the subagent waits with it.
        table.on_turn("p", None, "standard", None, t(0.0), 100);
        assert_eq!(
            table.on_turn("p:s1", Some("p"), "standard", None, t(1.0), 50),
            SessionState::Lobby
        );
        // Parent admitted: a new subagent starts admitted, never in the lobby.
        table.update("p", |s| {
            s.state = SessionState::Active;
            s.admitted = Some(t(2.0));
        });
        assert_eq!(
            table.on_turn("p:s2", Some("p"), "standard", None, t(3.0), 60),
            SessionState::Idle
        );
        let s = table.get("p:s2").unwrap();
        assert_eq!((s.admitted, s.parent.as_deref()), (Some(t(3.0)), Some("p")));
        // An ended subagent that returns while its parent is admitted is admitted again.
        table.update("p:s2", |s| s.state = SessionState::Ended);
        assert_eq!(
            table.on_turn("p:s2", Some("p"), "standard", None, t(4.0), 70),
            SessionState::Idle
        );
        // Unknown parent: a new session as before.
        assert_eq!(
            table.on_turn("q:s1", Some("q"), "standard", None, t(5.0), 10),
            SessionState::Lobby
        );
        // Its context counts in the live KV load.
        let (total, _) = table.kv_load();
        assert_eq!(total, table.kv_load_scan().0);
    }

    #[test]
    fn eviction_is_bounded_and_spares_sessions_in_flight() {
        let mut table = SessionTable::new(2);
        table.on_turn("a", None, "batch", None, t(0.0), 1);
        table.on_turn("b", None, "batch", None, t(1.0), 1);
        table.update("a", |s| s.in_flight = 1);
        table.on_turn("c", None, "batch", None, t(2.0), 1);
        assert_eq!(table.len(), 2);
        assert!(table.get("a").is_some(), "the session in flight is kept");
        assert!(table.get("b").is_none());
        assert_eq!(table.evicted(), 1);
    }

    #[test]
    fn next_turn_hot_demand_counts_in_flight_at_next_size_and_idle_by_duty_cycle() {
        let signals = Signals {
            next_think_s: Some(3.0),
            osl_tokens: Some(100),
            next_turn_tokens: Some(200),
            peak_tokens: None,
        };
        // a: mid-turn. b: between turns, its last turn took 1 s against 3 s of think (duty 1/4).
        // c: between turns, never served (no service time): counts nothing. d: in the lobby.
        let fill = |table: &mut SessionTable| {
            for (id, state, flight, service) in [
                ("a", SessionState::Active, 1, None),
                ("b", SessionState::Idle, 0, Some(1.0)),
                ("c", SessionState::Idle, 0, None),
                ("d", SessionState::Lobby, 0, None),
            ] {
                table.on_turn(id, None, "standard", None, t(0.0), 1000);
                table.update(id, |s| {
                    s.state = state;
                    s.in_flight = flight;
                    s.signals = signals;
                    s.last_service_s = service;
                });
            }
        };
        let mut table = SessionTable::new(8);
        table.set_hot_demand(HotDemand::NextTurn);
        fill(&mut table);
        // a: 1000 + 100 output; b: (1000 + 200 + 100) / 4. a, b and c are live.
        assert_eq!(table.kv_load(), (3000, 1100 + 325));
        assert_eq!(table.kv_load(), table.kv_load_scan());
        // b's turn ends, then its next one is in flight: it counts at its next size.
        table.update("b", |s| s.in_flight = 1);
        assert_eq!(table.kv_load(), (3000, 1100 + 1100));
        assert_eq!(table.kv_load(), table.kv_load_scan());
        // The default counts only the session mid-turn, at its context.
        let mut plain = SessionTable::new(8);
        fill(&mut plain);
        assert_eq!(plain.kv_load(), (3000, 1000));
    }

    #[test]
    fn kv_load_counts_live_sessions_and_hot_ones() {
        let mut table = SessionTable::new(8);
        for (id, ctx, state, flight) in [
            ("a", 100, SessionState::Active, 1),
            ("b", 50, SessionState::Idle, 0),
            ("c", 70, SessionState::Lobby, 0),
        ] {
            table.on_turn(id, None, "standard", None, t(0.0), ctx);
            table.update(id, |s| {
                s.state = state;
                s.in_flight = flight;
                s.context_tokens = ctx;
            });
        }
        assert_eq!(table.kv_load(), (150, 100));
    }
}
