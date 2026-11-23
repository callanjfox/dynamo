// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Admission tickets (`admission.tickets`, f49c; SLA "Must 1b": a refused session of a listed
//! class is admitted by its second advised return).
//!
//! Every refusal of a session of a listed class that carries a Retry-After issues (or replaces)
//! the session's ticket: its KV need, its onboarding prefill need, when it is advised back
//! (`due`) and when the ticket lapses (`expires`, `grace_s` after `due`). While a ticket is live:
//!
//! - **hold**: its KV need is kept out of the room new non-ticket sessions of its class and the
//!   classes ranked below it fit into (and its prefill need out of their onboarding term). The
//!   holds fill a cap of `max_hold_frac` of the HBM budget in precedence order: class rank, then
//!   oldest ticket first, so a lower class's tickets never reduce a higher class's hold. A ticket
//!   past the cap holds what is left (nothing once the cap is full) but keeps its other rights;
//! - **front of line**: back in the lobby, the session goes ahead of the non-ticket sessions of
//!   its class and fits around the tickets ahead of it only (its own hold is excluded).
//!
//! Admission consumes the ticket; expiry or the session's end drops it.
//!
//! f49e options (all off by default): `hold_while_waiting` keeps a ticket past its expiry while a
//! session that came back on it waits in the lobby; `per_root` keys one ticket per root session,
//! used by its sub-sessions; `keep_on_rerefusal` keeps a live ticket on a re-refusal and only
//! extends its expiry; `standard_cap_when_priority_below` caps the lower ranks' holds while the top
//! class is under its target. With `DYNAMO_TICKETS_LOG`, the scheduler also logs `ticket_hold`,
//! `ticket_rerefused` and `ticket_kept`.
//!
//! f49f `admission.returnee_first` (off by default) reads the live tickets of a class in issue
//! order ([`Tickets::iter`]): it refuses new sessions of the class while one is live and admits
//! the holders oldest first. It logs `ticket_hold` (reason `returnee_fifo`) and `returnee_block`.
//!
//! f49m `admission.honest_retry` (off by default) turns a ticket into a reservation
//! ([`Tickets::reserve`]): it holds from its own pool (`reserve_cap`), ahead of the unreserved
//! tickets of its rank, oldest reservation first.

use std::collections::HashMap;
use std::io::Write as _;
use std::time::Duration;

use crate::clock::Time;

/// One session's ticket.
#[derive(Clone, Debug, PartialEq)]
pub struct Ticket {
    pub session: String,
    pub class: String,
    /// The class's rank (lower first).
    pub rank: u32,
    /// KV tokens the session needs (its total need as the KV gate counts it).
    pub kv_need: u64,
    /// Prefill tokens its onboarding computes.
    pub prefill_need: u64,
    pub issued: Time,
    pub due: Time,
    pub expires: Time,
    /// Returns since the session's first ticket of this run of refusals.
    pub returns: u32,
    /// The latest return came before `due`.
    pub early: bool,
    /// Issue order, the tie-break after `issued`.
    pub seq: u64,
    /// f49e: the sessions that came back to the lobby on it (its holder, or with `per_root` the
    /// root's sub-sessions); `hold_while_waiting` reads it.
    pub waiting: std::collections::BTreeSet<String>,
    /// f49e `hold_while_waiting`: past `expires`, kept live while a session in `waiting` waits.
    pub waiting_hold: bool,
    /// f49m `admission.honest_retry`: when the ticket became a reservation (`None`: it is not one).
    pub reserved: Option<Time>,
}

/// The live tickets, the Retry-After advised to waiting sessions until their refusal, and the
/// optional log (`DYNAMO_TICKETS_LOG=<path>`, JSON lines).
#[derive(Debug, Default)]
pub struct Tickets {
    live: HashMap<String, Ticket>,
    /// Per waiting session, the Retry-After its deadline carries: issued as a ticket when the
    /// deadline refuses it.
    pending: HashMap<String, f64>,
    seq: u64,
    /// HBM and DRAM tokens of the gate's latest check (the cap and the forecast's room).
    pub(crate) last_kv: Option<crate::admit::KvBudget>,
    log: Option<std::io::BufWriter<std::fs::File>>,
    minute: Option<u64>,
    /// f49e `standard_cap_when_priority_below`: while set, `(rank, cap)`: the tickets ranked below
    /// `rank` hold at most `cap` together (within the overall cap).
    pub(crate) lower_cap: Option<(u32, u64)>,
    /// f49e `DYNAMO_TICKETS_LOG`: per waiting session, the hold reasons logged this wait, first
    /// first.
    waits: HashMap<String, Vec<&'static str>>,
    /// f49f2 `DYNAMO_TICKETS_LOG`: a `returnee_block` / `returnee_pass` line waiting for its
    /// session's KV need: (session, the line without its `,"need":..}` end).
    returnee_line: Option<(String, String)>,
    /// f49f2 `DYNAMO_TICKETS_LOG`: per class, the simulated second of its latest `returnee_pass`
    /// line and the passes since.
    passes: HashMap<String, (u64, u64)>,
    /// f49m `admission.honest_retry`: the most KV the reservations hold together (0 off).
    pub(crate) reserve_cap: u64,
}

/// What a session's gate check reads from the tickets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TicketTerms {
    /// KV tokens to keep out of the session's room.
    pub held_kv: u64,
    /// Prefill tokens to add to its onboarding term.
    pub held_prefill: u64,
    /// It holds a live ticket: front of its class's line.
    pub front: bool,
}

impl Tickets {
    /// With `log`, open the JSON-lines log `DYNAMO_TICKETS_LOG` names (off when unset).
    pub fn new(log: bool) -> Self {
        let log = log
            .then(|| std::env::var_os("DYNAMO_TICKETS_LOG"))
            .flatten()
            .and_then(|path| match std::fs::File::create(&path) {
                Ok(file) => Some(std::io::BufWriter::new(file)),
                Err(error) => {
                    tracing::warn!(?path, %error, "cannot open the tickets log");
                    None
                }
            });
        Self {
            log,
            ..Default::default()
        }
    }

    pub fn len(&self) -> usize {
        self.live.len()
    }

    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }

    pub fn get(&self, session: &str) -> Option<&Ticket> {
        self.live.get(session)
    }

    /// f49f: the tickets (live or not yet swept), in no order.
    pub fn iter(&self) -> impl Iterator<Item = &Ticket> {
        self.live.values()
    }

    /// Whether `session` holds a ticket live at `now`.
    pub fn is_live(&self, session: &str, now: Time) -> bool {
        // f49e: `waiting_hold` is only ever set with `hold_while_waiting`.
        self.live
            .get(session)
            .is_some_and(|t| now <= t.expires || t.waiting_hold)
    }

    /// The cap on held KV: `frac` of `hbm_tokens`.
    pub fn cap(frac: f64, hbm_tokens: u64) -> u64 {
        (frac.clamp(0.0, 1.0) * hbm_tokens as f64) as u64
    }

    /// The live tickets in precedence order (class rank, then oldest first) with the KV each
    /// holds within `cap`. f49m: within a rank the reservations come first (oldest reservation
    /// first) and hold within `reserve_cap` instead of `cap`.
    pub fn holds(&self, cap: u64) -> Vec<(&Ticket, u64)> {
        let mut all: Vec<&Ticket> = self.live.values().collect();
        all.sort_by(|a, b| {
            a.rank
                .cmp(&b.rank)
                .then_with(|| a.reserved.is_none().cmp(&b.reserved.is_none()))
                .then_with(|| a.reserved.cmp(&b.reserved))
                .then_with(|| a.issued.cmp(&b.issued))
                .then_with(|| a.seq.cmp(&b.seq))
        });
        let mut left = cap;
        let mut reserve_left = self.reserve_cap;
        // f49e `standard_cap_when_priority_below`: the lower ranks' own cap while it is set.
        let mut lower_left = self.lower_cap.map_or(u64::MAX, |(_, c)| c);
        all.into_iter()
            .map(|t| {
                let pool = if t.reserved.is_some() {
                    &mut reserve_left
                } else {
                    &mut left
                };
                let mut held = t.kv_need.min(*pool);
                if self.lower_cap.is_some_and(|(rank, _)| t.rank > rank) {
                    held = held.min(lower_left);
                    lower_left -= held;
                }
                *pool -= held;
                (t, held)
            })
            .collect()
    }

    /// f49m: the KV the ticket keyed `key` holds within `cap` (0 without one).
    pub fn held(&self, key: &str, cap: u64) -> u64 {
        self.holds(cap)
            .into_iter()
            .find(|(t, _)| t.session == key)
            .map_or(0, |(_, h)| h)
    }

    /// f49m: the KV the reservations hold together, and how many there are.
    pub fn reserved_held(&self, cap: u64) -> (u64, usize) {
        self.holds(cap)
            .into_iter()
            .filter(|(t, _)| t.reserved.is_some())
            .fold((0, 0), |(kv, n), (_, h)| (kv + h, n + 1))
    }

    /// f49m `admission.honest_retry`: the ticket keyed `key` becomes (or stays) a reservation that
    /// lapses `grace_s` after `retry_after_s` from `now`. Returns whether it became one now.
    pub fn reserve(&mut self, key: &str, retry_after_s: f64, grace_s: f64, now: Time) -> bool {
        let Some(t) = self.live.get_mut(key) else {
            return false;
        };
        let due = now.after(Duration::from_secs_f64(retry_after_s.max(0.0)));
        t.expires = due.after(Duration::from_secs_f64(grace_s.max(0.0)));
        let opened = t.reserved.is_none();
        if opened {
            t.reserved = Some(now);
        }
        opened
    }

    /// What a session of rank `rank` fits around: a ticket holder, the holds of the tickets ahead
    /// of its own; any other session, the holds of the tickets of its rank or above. Prefill
    /// counts for the tickets that hold KV.
    pub fn terms(&self, session: &str, rank: u32, cap: u64, now: Time) -> TicketTerms {
        if self.live.is_empty() {
            return TicketTerms::default();
        }
        let holds = self.holds(cap);
        let mine = self.is_live(session, now);
        let mut terms = TicketTerms {
            front: mine,
            ..Default::default()
        };
        for (t, held) in holds {
            if mine && t.session == session {
                break;
            }
            if t.rank > rank || t.session == session {
                continue;
            }
            terms.held_kv += held;
            if held > 0 {
                terms.held_prefill += t.prefill_need;
            }
        }
        terms
    }

    /// KV held by the tickets of rank `rank` or above other than `session`'s, within `cap`.
    pub fn held_by_others(&self, session: &str, rank: u32, cap: u64) -> u64 {
        self.holds(cap)
            .into_iter()
            .filter(|(t, _)| t.rank <= rank && t.session != session)
            .map(|(_, h)| h)
            .sum()
    }

    /// Advise `seconds` to waiting `session`: its ticket when its deadline refuses it.
    pub fn set_pending(&mut self, session: &str, seconds: f64) {
        self.pending.insert(session.to_owned(), seconds);
    }

    pub fn take_pending(&mut self, session: &str) -> Option<f64> {
        self.pending.remove(session)
    }

    /// Issue (or replace) `session`'s ticket, due `retry_after_s` from `now`, lapsing `grace_s`
    /// after. Returns whether it holds KV within `cap`.
    #[allow(clippy::too_many_arguments)]
    pub fn issue(
        &mut self,
        session: &str,
        class: &str,
        rank: u32,
        kv_need: u64,
        prefill_need: u64,
        retry_after_s: f64,
        grace_s: f64,
        cap: u64,
        now: Time,
    ) -> bool {
        self.issue_by(
            session,
            session,
            class,
            rank,
            kv_need,
            prefill_need,
            retry_after_s,
            grace_s,
            cap,
            now,
        )
    }

    /// f49e: [`Self::issue`] of the ticket keyed `session` on a refusal of `by` (itself, or with
    /// `per_root` a sub-session of the root `session`). The sessions waiting on a replaced ticket,
    /// other than `by`, stay on the new one.
    #[allow(clippy::too_many_arguments)]
    pub fn issue_by(
        &mut self,
        session: &str,
        by: &str,
        class: &str,
        rank: u32,
        kv_need: u64,
        prefill_need: u64,
        retry_after_s: f64,
        grace_s: f64,
        cap: u64,
        now: Time,
    ) -> bool {
        let due = now.after(Duration::from_secs_f64(retry_after_s.max(0.0)));
        let expires = due.after(Duration::from_secs_f64(grace_s.max(0.0)));
        // f49m: a replaced reservation stays one, its age kept.
        let (returns, mut waiting, reserved) = self
            .live
            .remove(session)
            .map_or((0, Default::default(), None), |t| {
                (t.returns, t.waiting, t.reserved)
            });
        waiting.remove(by);
        self.seq += 1;
        self.live.insert(
            session.to_owned(),
            Ticket {
                session: session.to_owned(),
                class: class.to_owned(),
                rank,
                kv_need,
                prefill_need,
                issued: now,
                due,
                expires,
                returns,
                early: false,
                seq: self.seq,
                waiting,
                waiting_hold: false,
                reserved,
            },
        );
        let held = self
            .holds(cap)
            .into_iter()
            .any(|(t, h)| t.session == session && h > 0);
        self.write(format_args!(
            "{{\"event\":\"ticket_issued\",\"t\":{:.3},\"session\":{:?},\"class\":{:?},\"kv_need\":{},\"prefill_need\":{},\"due\":{:.3},\"expires\":{:.3},\"held\":{}}}",
            now.as_secs_f64(),
            session,
            class,
            kv_need,
            prefill_need,
            due.as_secs_f64(),
            expires.as_secs_f64(),
            held
        ));
        held
    }

    /// `session` came back to the lobby: one more return on its live ticket.
    pub fn returned(&mut self, session: &str, now: Time) {
        self.returned_by(session, session, now);
    }

    /// f49e: `by` came back to the lobby on the ticket keyed `key` (its own, or with `per_root` its
    /// root's): one more return, and `by` waits on it.
    pub fn returned_by(&mut self, key: &str, by: &str, now: Time) {
        if let Some(t) = self.live.get_mut(key)
            && (now <= t.expires || t.waiting_hold)
        {
            t.returns += 1;
            t.early = now < t.due;
            t.waiting.insert(by.to_owned());
        }
    }

    /// `session` was admitted: its ticket is used up.
    pub fn admitted(&mut self, session: &str, now: Time) -> Option<Ticket> {
        self.admitted_by(session, session, now)
    }

    /// f49e: `by` was admitted on the ticket keyed `key` (its own, or with `per_root` its root's):
    /// the ticket is used up.
    pub fn admitted_by(&mut self, key: &str, by: &str, now: Time) -> Option<Ticket> {
        self.pending.remove(by);
        self.waits.remove(by);
        let session = key;
        let t = self.live.remove(session)?;
        self.write(format_args!(
            "{{\"event\":\"ticket_admitted\",\"t\":{:.3},\"session\":{:?},\"class\":{:?},\"early\":{},\"returns\":{}}}",
            now.as_secs_f64(),
            session,
            t.class,
            t.early,
            t.returns
        ));
        Some(t)
    }

    /// The session ended: drop its ticket.
    pub fn ended(&mut self, session: &str) {
        self.pending.remove(session);
        self.waits.remove(session);
        self.live.remove(session);
    }

    /// f49e `per_root`: `by` was admitted while other sessions still wait on the ticket keyed
    /// `key`: it stays, without `by`.
    pub fn left_wait(&mut self, key: &str, by: &str) {
        self.pending.remove(by);
        self.waits.remove(by);
        if let Some(t) = self.live.get_mut(key) {
            t.waiting.remove(by);
        }
    }

    /// Drop the tickets past their expiry at `now`.
    pub fn expire(&mut self, now: Time) {
        self.expire_keeping(now, |_| false);
    }

    /// f49e: [`Self::expire`], keeping (`hold_while_waiting`) each ticket past its expiry for which
    /// `keep` holds; it stays live (`waiting_hold`) until `keep` no longer holds at an expiry check.
    pub fn expire_keeping(&mut self, now: Time, keep: impl Fn(&Ticket) -> bool) {
        if self.live.is_empty() {
            return;
        }
        let mut gone: Vec<String> = Vec::new();
        for t in self.live.values_mut() {
            if now <= t.expires {
                t.waiting_hold = false;
            } else if keep(t) {
                t.waiting_hold = true;
            } else {
                gone.push(t.session.clone());
            }
        }
        gone.sort();
        for session in gone {
            self.live.remove(&session);
            self.write(format_args!(
                "{{\"event\":\"ticket_expired\",\"t\":{:.3},\"session\":{:?}}}",
                now.as_secs_f64(),
                session
            ));
        }
    }

    /// Once per simulated minute: the held KV and the live tickets per class (log only).
    pub fn minute(&mut self, now: Time, cap: u64) {
        if self.log.is_none() {
            return;
        }
        let minute = (now.as_secs_f64() / 60.0) as u64;
        if self.minute == Some(minute) {
            return;
        }
        self.minute = Some(minute);
        let mut by: std::collections::BTreeMap<&str, (u64, u64, u64)> = Default::default();
        let mut total = 0;
        for (t, held) in self.holds(cap) {
            let e = by.entry(t.class.as_str()).or_default();
            e.0 += 1;
            e.1 += held;
            e.2 += u64::from(held > 0);
            total += held;
        }
        let classes = by
            .iter()
            .map(|(c, (n, h, holding))| {
                format!("{c:?}:{{\"live\":{n},\"holding\":{holding},\"held_kv\":{h}}}")
            })
            .collect::<Vec<_>>()
            .join(",");
        let line = format!(
            "{{\"event\":\"minute\",\"t_s\":{},\"held_kv\":{},\"cap\":{},\"live\":{},\"classes\":{{{}}}}}",
            minute * 60,
            total,
            cap,
            self.live.len(),
            classes
        );
        self.write(format_args!("{line}"));
    }

    /// f49e: a refusal of `by` while the ticket keyed `key` is live keeps it (its issue time, due
    /// and returns): with `need` (`per_root`) its KV need becomes the larger of the two; with
    /// `extend` (`keep_on_rerefusal`: the new Retry-After and `grace_s`) it lapses no sooner than
    /// the new due plus the grace. Returns whether it holds KV within `cap`.
    pub fn keep(
        &mut self,
        key: &str,
        by: &str,
        need: Option<u64>,
        extend: Option<(f64, f64)>,
        cap: u64,
        now: Time,
    ) -> bool {
        let Some(t) = self.live.get_mut(key) else {
            return false;
        };
        if let Some(n) = need {
            t.kv_need = t.kv_need.max(n);
        }
        if let Some((retry_after_s, grace_s)) = extend {
            let due = now.after(Duration::from_secs_f64(retry_after_s.max(0.0)));
            let expires = due.after(Duration::from_secs_f64(grace_s.max(0.0)));
            t.expires = t.expires.max(expires);
        }
        t.waiting.remove(by);
        let (class, kv_need, expires) = (t.class.clone(), t.kv_need, t.expires);
        let held = self
            .holds(cap)
            .into_iter()
            .any(|(t, h)| t.session == key && h > 0);
        self.write(format_args!(
            "{{\"event\":\"ticket_kept\",\"t\":{:.3},\"session\":{:?},\"by\":{:?},\"class\":{:?},\"kv_need\":{},\"expires\":{:.3},\"held\":{}}}",
            now.as_secs_f64(),
            key,
            by,
            class,
            kv_need,
            expires.as_secs_f64(),
            held
        ));
        held
    }

    /// f49e: whether the tickets log is open (`DYNAMO_TICKETS_LOG`).
    pub fn logging(&self) -> bool {
        self.log.is_some()
    }

    /// f49e diagnostic: whether `reason` is new for `session` in its current wait (it is recorded).
    pub fn hold_seen(&mut self, session: &str, reason: &'static str) -> bool {
        let seen = self.waits.entry(session.to_owned()).or_default();
        if seen.contains(&reason) {
            return false;
        }
        seen.push(reason);
        true
    }

    /// f49e diagnostic: `session`'s wait is over; the first hold reason logged in it.
    pub fn wait_over(&mut self, session: &str) -> Option<&'static str> {
        self.waits.remove(session).and_then(|v| v.first().copied())
    }

    /// f49f2 diagnostic: keep `line` (a JSON object without its end) until `session`'s KV need is
    /// known ([`Self::returnee_need`]). A line still kept is written first with `"need":null`;
    /// returns it.
    pub fn returnee_line(&mut self, session: &str, line: String) -> Option<String> {
        let stale = self
            .returnee_line
            .replace((session.to_owned(), line))
            .map(|(_, old)| format!("{old},\"need\":null}}"));
        if let Some(stale) = &stale {
            self.write(format_args!("{stale}"));
        }
        stale
    }

    /// f49f2 diagnostic: `session`'s KV need is `need`: its kept line is written with it (nothing
    /// without one, or for another session's). Returns the line.
    pub fn returnee_need(&mut self, session: &str, need: u64) -> Option<String> {
        let (kept, _) = self.returnee_line.as_ref()?;
        if kept != session {
            return None;
        }
        let (_, line) = self.returnee_line.take()?;
        let line = format!("{line},\"need\":{need}}}");
        self.write(format_args!("{line}"));
        Some(line)
    }

    /// f49f2 diagnostic: one more `returnee_pass` of `class` at `now`. Returns the passes to
    /// report on a line now (since the class's previous line, this one included), or `None` within
    /// the simulated second of its previous line.
    pub fn returnee_pass(&mut self, class: &str, now: Time) -> Option<u64> {
        let second = now.as_secs_f64() as u64;
        let e = self.passes.entry(class.to_owned()).or_insert((u64::MAX, 0));
        e.1 += 1;
        if e.0 == second {
            return None;
        }
        e.0 = second;
        Some(std::mem::take(&mut e.1))
    }

    /// f49e diagnostic: one JSON line to the tickets log (nothing without it).
    pub fn log_line(&mut self, line: &str) {
        self.write(format_args!("{line}"));
    }

    fn write(&mut self, line: std::fmt::Arguments<'_>) {
        if let Some(out) = self.log.as_mut() {
            let _ = writeln!(out, "{line}");
            let _ = out.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: f64) -> Time {
        Time::from_secs_f64(s)
    }

    #[test]
    fn holds_fill_the_cap_by_rank_then_age() {
        let mut k = Tickets::default();
        // Standard first in time, then two priority tickets.
        k.issue("s1", "standard", 2, 300, 0, 10.0, 30.0, 1000, t(0.0));
        k.issue("p1", "priority", 1, 400, 0, 10.0, 30.0, 1000, t(1.0));
        k.issue("p2", "priority", 1, 400, 0, 10.0, 30.0, 1000, t(2.0));
        let holds: Vec<(&str, u64)> = k
            .holds(1000)
            .into_iter()
            .map(|(t, h)| (t.session.as_str(), h))
            .collect();
        // Priority first (oldest first), standard gets what is left.
        assert_eq!(holds, vec![("p1", 400), ("p2", 400), ("s1", 200)]);
        // A smaller cap: the second priority ticket holds part, standard nothing.
        let holds: Vec<u64> = k.holds(500).into_iter().map(|(_, h)| h).collect();
        assert_eq!(holds, vec![400, 100, 0]);
    }

    #[test]
    fn terms_follow_class_precedence() {
        let mut k = Tickets::default();
        k.issue("s1", "standard", 2, 300, 30, 10.0, 30.0, 10_000, t(0.0));
        k.issue("p1", "priority", 1, 400, 40, 10.0, 30.0, 10_000, t(1.0));
        // A new priority session fits around the priority ticket only.
        let p = k.terms("new", 1, 10_000, t(2.0));
        assert_eq!((p.held_kv, p.held_prefill, p.front), (400, 40, false));
        // Standard and batch fit around both.
        assert_eq!(k.terms("new", 2, 10_000, t(2.0)).held_kv, 700);
        assert_eq!(k.terms("new", 3, 10_000, t(2.0)).held_kv, 700);
        // A returning ticket holder: front, around the tickets ahead of its own only.
        let s = k.terms("s1", 2, 10_000, t(2.0));
        assert_eq!((s.held_kv, s.front), (400, true));
        let p = k.terms("p1", 1, 10_000, t(2.0));
        assert_eq!((p.held_kv, p.front), (0, true));
    }

    #[test]
    fn expiry_admission_and_end_drop_tickets() {
        let mut k = Tickets::default();
        k.issue("a", "priority", 1, 100, 0, 10.0, 30.0, 1000, t(0.0));
        k.issue("b", "priority", 1, 100, 0, 10.0, 30.0, 1000, t(0.0));
        k.issue("c", "priority", 1, 100, 0, 10.0, 30.0, 1000, t(0.0));
        // Due at 10, expires at 40.
        assert!(k.is_live("a", t(40.0)));
        k.expire(t(40.0));
        assert_eq!(k.len(), 3);
        k.returned("a", t(5.0));
        let a = k.admitted("a", t(5.0)).unwrap();
        assert_eq!((a.returns, a.early), (1, true));
        assert!(k.get("a").is_none(), "admission consumes the ticket");
        k.ended("b");
        assert!(k.get("b").is_none());
        k.expire(t(40.1));
        assert!(k.is_empty());
    }

    /// f49e: a ticket past its expiry is kept while `keep` holds, and live meanwhile.
    #[test]
    fn expire_keeping_holds_a_waiting_ticket_past_its_expiry() {
        let mut k = Tickets::default();
        k.issue("a", "standard", 2, 100, 0, 10.0, 30.0, 1000, t(0.0));
        k.returned("a", t(10.0));
        assert!(k.get("a").unwrap().waiting.contains("a"));
        k.expire_keeping(t(50.0), |t| t.waiting.contains("a"));
        assert!(k.is_live("a", t(50.0)));
        assert!(k.get("a").unwrap().waiting_hold);
        // A return while kept still counts.
        k.returned("a", t(51.0));
        assert_eq!(k.get("a").unwrap().returns, 2);
        k.expire_keeping(t(60.0), |_| false);
        assert!(k.is_empty());
    }

    /// f49e: `keep` keeps the issue time, due and returns, and only extends the expiry.
    #[test]
    fn keep_extends_only_the_expiry() {
        let mut k = Tickets::default();
        k.issue("a", "standard", 2, 100, 0, 10.0, 30.0, 1000, t(0.0));
        k.returned("a", t(10.0));
        k.keep("a", "a", None, Some((20.0, 30.0)), 1000, t(35.0));
        let a = k.get("a").unwrap();
        assert_eq!(
            (a.issued, a.due, a.returns, a.expires, a.kv_need, a.seq),
            (t(0.0), t(10.0), 1, t(85.0), 100, 1)
        );
        // `per_root`: the larger KV need, no earlier expiry.
        k.keep("a", "a:1", Some(300), Some((0.0, 0.0)), 1000, t(36.0));
        let a = k.get("a").unwrap();
        assert_eq!((a.kv_need, a.expires), (300, t(85.0)));
    }

    /// f49e: the lower ranks' cap binds them only, within the overall cap.
    #[test]
    fn the_lower_cap_binds_the_lower_ranks_only() {
        let mut k = Tickets::default();
        k.issue("s1", "standard", 2, 300, 0, 10.0, 30.0, 1000, t(0.0));
        k.issue("s2", "standard", 2, 300, 0, 10.0, 30.0, 1000, t(0.5));
        k.issue("p1", "priority", 1, 400, 0, 10.0, 30.0, 1000, t(1.0));
        k.lower_cap = Some((1, 350));
        let holds: Vec<u64> = k.holds(1000).into_iter().map(|(_, h)| h).collect();
        assert_eq!(holds, vec![400, 300, 50]);
        k.lower_cap = None;
        let holds: Vec<u64> = k.holds(1000).into_iter().map(|(_, h)| h).collect();
        assert_eq!(holds, vec![400, 300, 300]);
    }

    /// f49f2: `returnee_pass` reports one line per class per simulated second, with the passes
    /// since the previous one; a kept line ends with its session's need, or `null` when replaced.
    #[test]
    fn returnee_lines_carry_the_need_and_passes_are_one_per_second() {
        let mut k = Tickets::default();
        assert_eq!(k.returnee_pass("standard", t(1.2)), Some(1));
        assert_eq!(k.returnee_pass("standard", t(1.7)), None);
        assert_eq!(k.returnee_pass("priority", t(1.8)), Some(1));
        assert_eq!(k.returnee_pass("standard", t(1.9)), None);
        assert_eq!(k.returnee_pass("standard", t(2.1)), Some(3));
        assert_eq!(k.returnee_line("a", r#"{"x":1"#.to_owned()), None);
        assert_eq!(k.returnee_need("b", 5), None);
        assert_eq!(
            k.returnee_line("b", r#"{"x":2"#.to_owned()).as_deref(),
            Some(r#"{"x":1,"need":null}"#)
        );
        assert_eq!(
            k.returnee_need("b", 5).as_deref(),
            Some(r#"{"x":2,"need":5}"#)
        );
        assert_eq!(k.returnee_need("b", 5), None);
    }

    /// f49m: reservations hold from their own pool, by rank then oldest reservation first, ahead
    /// of the unreserved tickets of their rank; a replaced reservation stays one.
    #[test]
    fn reservations_hold_from_their_own_pool_ahead_of_their_rank() {
        let mut k = Tickets::default();
        k.issue("s1", "standard", 2, 300, 0, 10.0, 30.0, 1000, t(0.0));
        k.issue("s2", "standard", 2, 300, 0, 10.0, 30.0, 1000, t(1.0));
        k.issue("p1", "priority", 1, 300, 0, 10.0, 30.0, 1000, t(2.0));
        k.reserve_cap = 400;
        assert!(k.reserve("s2", 20.0, 30.0, t(3.0)));
        assert!(!k.reserve("s2", 20.0, 30.0, t(4.0)), "opened once");
        assert_eq!(k.get("s2").unwrap().expires, t(54.0));
        assert!(k.reserve("p1", 20.0, 30.0, t(5.0)));
        let holds: Vec<(&str, u64)> = k
            .holds(250)
            .into_iter()
            .map(|(t, h)| (t.session.as_str(), h))
            .collect();
        // p1 and s2 from the pool of 400 (p1 first: its rank), s1 from the ticket cap.
        assert_eq!(holds, vec![("p1", 300), ("s2", 100), ("s1", 250)]);
        assert_eq!(k.reserved_held(250), (400, 2));
        assert_eq!(k.held("s2", 250), 100);
        // s2 holder fits around p1 only; a new standard session around all three.
        assert_eq!(k.terms("s2", 2, 250, t(6.0)).held_kv, 300);
        assert_eq!(k.terms("new", 2, 250, t(6.0)).held_kv, 650);
        assert_eq!(k.terms("new", 1, 250, t(6.0)).held_kv, 300);
        k.issue("s2", "standard", 2, 300, 0, 10.0, 30.0, 1000, t(6.0));
        assert_eq!(k.get("s2").unwrap().reserved, Some(t(3.0)));
        assert!(!k.reserve("nobody", 1.0, 1.0, t(7.0)));
    }

    #[test]
    fn a_replaced_ticket_keeps_its_returns() {
        let mut k = Tickets::default();
        k.issue("a", "standard", 2, 100, 0, 10.0, 30.0, 1000, t(0.0));
        k.returned("a", t(12.0));
        k.issue("a", "standard", 2, 120, 0, 20.0, 30.0, 1000, t(13.0));
        let a = k.get("a").unwrap();
        assert_eq!((a.returns, a.kv_need, a.due), (1, 120, t(33.0)));
        assert_eq!(k.len(), 1, "one live ticket per session");
    }
}
