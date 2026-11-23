// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What each session gets (F35; runs and their uncontended time, f36b): its wall time split into
//! server time (lobby or queue wait, TTFT and decode: anything of the session in flight at the
//! router or an engine) and client time (think and tool time, a person reading), its stop points
//! and the runs between them, its requests and their on-target share, and its lobby holds and
//! refusals.
//!
//! Terms (goal 22): a *request* is one LLM call; a *run* is a session's requests from a user
//! message to the agent's hand-back (the stop point); a session is many runs. The experience
//! metric is *hand-back time* (a run's wall time) and *run slowdown* (hand-back time over the
//! run's uncontended time).
//!
//! A session here is a top-level session with its subagents (a subagent's requests count for the
//! root it descends from), as its user experiences it. Server time is the union of the intervals
//! with at least one of its requests in flight, from each attempt's classification to its
//! completion or abort, so a lobby wait (with the retries after a wait-limit expiry) counts.
//!
//! Stop points: a request that starts with a new user message ends the run before it. The
//! frontend says so for every request (`input_trigger`: a user message, a tool result or other),
//! or the client (`x-dynamo-hint-last-role`); where neither says (replay), a gap with nothing in
//! flight longer than `session_log.stop_gap_s` (default 180 s) counts as one. Only the root
//! session's requests mark stop points (a subagent's first request starts with its prompt as a
//! user message).
//!
//! A run's uncontended time (F1) is computed as `scripts/validate/family_b.py` does: per busy
//! interval (a request of the session in flight), the smaller of the summed uncontended times of
//! the requests completed in it ([`crate::uncontended`]) and the interval itself, plus the run's
//! client gaps as observed. Unknown (null) when a completed request has none (no step model).
//!
//! Added delay (`docs/test-plan-full.md`, "The run metric: added delay"), phase 1: D_1 =
//! waiting plus prefill excess on the run's critical path, the main agent's own requests (a
//! subagent's delay is attributed in a later phase): per request, the router wait from its
//! classification to dispatch, plus its time from dispatch to first token over the reference
//! prefill (the engine queue falls here: the router cannot tell it from prefill); every refused
//! or expired attempt's time and the client's retry backoff after it (not client time, and never
//! a stop point). A run starts at its first request's first classification, so a session held
//! or refused at its start is a run whose D_1 runs until it is served; one never served is
//! closed as `gave_up` once its client stops retrying for the stop gap. S_ref, the run's
//! reference service, is the main agent's requests' uncontended times; the allowance is
//! `run_level.delay_base_s + delay_per_ref x S_ref`.
//!
//! The log (`DYNAMO_SESSION_LOG`): one `run` line per finished run and one `session_summary` line
//! per ended session (the shape of the session observability telemetry's log line, with these
//! fields added). The scheduler reads the finished runs for the class's run attainment and the
//! current run's attained service for `turn_order: run_las`.

use std::collections::{HashMap, VecDeque};
use std::io::Write;

use crate::clock::Time;

/// Progress samples kept per run for the autonomous floor.
const MAX_PROGRESS_SAMPLES: usize = 1024;
/// Finished runs kept until the scheduler takes them.
const MAX_FINISHED: usize = 4096;

/// The run in progress.
#[derive(Clone, Debug, PartialEq)]
pub struct RunState {
    pub start: Time,
    pub server_s: f64,
    /// Uncontended seconds of the closed busy intervals and the client gaps so far.
    unc_s: f64,
    /// Uncontended seconds of the requests completed in the open busy interval.
    busy_unc_s: f64,
    /// Uncontended seconds of the run's completed requests: its attained service.
    pub attained_s: f64,
    pub requests: u32,
    /// Every completed request had an uncontended time.
    unc_known: bool,
    /// (time, progress) after each completion, for the autonomous floor.
    progress: VecDeque<(Time, f64)>,
    /// Attempts classified in the run (any outcome).
    attempts: u32,
    /// D_1's parts: waiting (router wait, refused attempts, retry backoff) and prefill excess.
    pub waiting_s: f64,
    pub prefill_excess_s: f64,
    /// Reference service of the main agent's completed requests.
    pub s_ref_s: f64,
}

impl RunState {
    fn new(start: Time) -> Self {
        Self {
            start,
            server_s: 0.0,
            unc_s: 0.0,
            busy_unc_s: 0.0,
            attained_s: 0.0,
            requests: 0,
            unc_known: true,
            progress: VecDeque::new(),
            attempts: 0,
            waiting_s: 0.0,
            prefill_excess_s: 0.0,
            s_ref_s: 0.0,
        }
    }

    /// Added delay so far, phase 1: waiting plus prefill excess.
    pub fn d1_s(&self) -> f64 {
        self.waiting_s + self.prefill_excess_s
    }

    /// Uncontended seconds the run has made so far: its client gaps plus the service of its
    /// completed requests (capped per closed busy interval at the interval).
    pub fn progress_s(&self) -> f64 {
        self.unc_s + self.busy_unc_s
    }

    /// The latest progress sample at or before `at`, or the run's start with none.
    pub fn progress_at(&self, at: Time) -> (Time, f64) {
        self.progress
            .iter()
            .rev()
            .find(|(t, _)| *t <= at)
            .copied()
            .unwrap_or((self.start, 0.0))
    }

    fn sample(&mut self, now: Time, keep_s: f64) {
        let p = self.progress_s();
        self.progress.push_back((now, p));
        // Keep the newest sample before the window and everything after it.
        while self.progress.len() > 1
            && self
                .progress
                .get(1)
                .is_some_and(|(t, _)| now.since(*t).as_secs_f64() >= keep_s)
        {
            self.progress.pop_front();
        }
        if self.progress.len() > MAX_PROGRESS_SAMPLES {
            self.progress.pop_front();
        }
    }
}

/// A finished run.
#[derive(Clone, Debug, PartialEq)]
pub struct FinishedRun {
    /// The root session.
    pub session: String,
    pub class: String,
    pub kind: Option<&'static str>,
    pub start: Time,
    pub end: Time,
    pub server_s: f64,
    /// Uncontended seconds; `None` when unknown.
    pub uncontended_s: Option<f64>,
    pub requests: u32,
    /// What ended it: `gap` (a gap over the stop gap), `user` (the next request starts with a user
    /// message), `end` (the session ended) or `gave_up` (never served; its client stopped retrying).
    pub ended_by: &'static str,
    /// Added delay D_1 and its parts, the reference service and the allowance (`None` for a class
    /// without `run_level`).
    pub waiting_s: f64,
    pub prefill_excess_s: f64,
    pub s_ref_s: f64,
    pub allowance_s: Option<f64>,
}

impl FinishedRun {
    pub fn d1_s(&self) -> f64 {
        self.waiting_s + self.prefill_excess_s
    }

    pub fn wall_s(&self) -> f64 {
        self.end.since(self.start).as_secs_f64()
    }

    /// Hand-back time over uncontended time, at least 1; `None` when either is unknown.
    pub fn slowdown(&self) -> Option<f64> {
        self.uncontended_s
            .filter(|u| *u > 0.0)
            .map(|u| (self.wall_s() / u).max(1.0))
    }
}

/// One session's running account.
#[derive(Clone, Debug, PartialEq)]
pub struct Account {
    pub class: String,
    pub kind: Option<&'static str>,
    pub user: Option<String>,
    pub tenant: Option<String>,
    pub started: Time,
    in_flight: u32,
    busy_since: Option<Time>,
    idle_since: Option<Time>,
    pub server_s: f64,
    pub client_s: f64,
    /// Requests completed.
    pub turns: u32,
    pub on_target: u32,
    pub judged: u32,
    pub holds: u32,
    pub refusals: u32,
    pub stops: u32,
    pub run: RunState,
    /// (wall s, server s, uncontended s) per finished run, up to `max_runs`.
    pub runs: Vec<(f64, f64, Option<f64>)>,
    pub runs_dropped: u32,
    /// Whether the next request starts with a user message, when the frontend or the client
    /// says (`None`: unknown, the stop gap decides).
    next_is_user: Option<bool>,
    /// The frontend or the client has said for one of its requests: its stop points come from
    /// them, so an idle gap never closes its run before its next request.
    told: bool,
    /// The idle gap already passed the stop gap and closed the run ([`Experience::sweep`]): the
    /// next request starts a new one.
    run_closed: bool,
    /// The last attempt was refused or expired: the gap to the next is retry backoff.
    retrying: bool,
    /// D_1 and allowance of the session's finished runs (`server_wait_deficit` by delay).
    pub d1_total_s: f64,
    pub allowance_total_s: f64,
    /// f49q `signal: slowdown`: standalone seconds of the session's completed requests.
    pub service_s: f64,
}

impl Account {
    fn new(class: &str, started: Time) -> Self {
        Self {
            class: class.to_owned(),
            kind: None,
            user: None,
            tenant: None,
            started,
            in_flight: 0,
            busy_since: None,
            idle_since: None,
            server_s: 0.0,
            client_s: 0.0,
            turns: 0,
            on_target: 0,
            judged: 0,
            holds: 0,
            refusals: 0,
            stops: 0,
            run: RunState::new(started),
            runs: Vec::new(),
            runs_dropped: 0,
            next_is_user: None,
            told: false,
            run_closed: false,
            retrying: false,
            d1_total_s: 0.0,
            allowance_total_s: 0.0,
            service_s: 0.0,
        }
    }

    /// The last moment the session had something in flight, or now if it still has.
    fn last_active(&self) -> Time {
        self.idle_since.unwrap_or(self.started)
    }

    /// Whether a request of the session is in flight.
    pub fn busy(&self) -> bool {
        self.in_flight > 0
    }

    /// The session's added delay less its allowance so far, its open run included (`allowance`
    /// per run as `(base, per_ref)`; `None`: none).
    pub fn delay_deficit_s(&self, allowance: Option<(f64, f64)>) -> f64 {
        let open = allowance.map_or(0.0, |(a, b)| a + b * self.run.s_ref_s);
        self.d1_total_s + self.run.d1_s() - self.allowance_total_s - open
    }

    /// f49q: wall time since the session's first request less its client gaps (time with nothing
    /// of it in flight, the open gap included; a retry backoff counts as active).
    pub fn active_s(&self, now: Time) -> f64 {
        let open_gap = match self.idle_since {
            Some(idle) if self.in_flight == 0 && (!self.retrying || self.run_closed) => {
                now.since(idle).as_secs_f64()
            }
            _ => 0.0,
        };
        (now.since(self.started).as_secs_f64() - self.client_s - open_gap).max(0.0)
    }

    /// Server time so far, the open busy interval included up to `now`.
    pub fn server_s_at(&self, now: Time) -> f64 {
        self.server_s
            + self
                .busy_since
                .map_or(0.0, |since| now.since(since).as_secs_f64())
    }
}

/// The accounts of every tracked root session, bounded like the session table.
#[derive(Debug)]
pub struct Experience {
    capacity: usize,
    stop_gap_s: f64,
    max_runs: usize,
    /// Progress kept per run for the autonomous floor, seconds.
    floor_window_s: f64,
    accounts: HashMap<String, Account>,
    /// Per class, the allowance `(delay_base_s, delay_per_ref)` of a run.
    allowances: HashMap<String, (f64, f64)>,
    finished: Vec<FinishedRun>,
    /// The last [`Self::sweep`] over the accounts.
    swept: Option<Time>,
    log: Option<std::fs::File>,
}

impl Experience {
    pub fn new(capacity: usize, stop_gap_s: f64, max_runs: usize) -> Self {
        let log = std::env::var_os("DYNAMO_SESSION_LOG").and_then(|path| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .ok()
        });
        Self {
            capacity: capacity.max(1),
            stop_gap_s,
            max_runs,
            floor_window_s: 600.0,
            accounts: HashMap::new(),
            allowances: HashMap::new(),
            finished: Vec::new(),
            swept: None,
            log,
        }
    }

    /// Each class's run allowance, `(delay_base_s, delay_per_ref)`.
    pub fn with_allowances(mut self, allowances: HashMap<String, (f64, f64)>) -> Self {
        self.allowances = allowances;
        self
    }

    /// A class's run allowance, `(delay_base_s, delay_per_ref)`.
    pub fn allowance(&self, class: &str) -> Option<(f64, f64)> {
        self.allowances.get(class).copied()
    }

    /// Keep enough progress per run to read the floor over `window_s`.
    pub fn with_floor_window(mut self, window_s: f64) -> Self {
        self.floor_window_s = window_s;
        self
    }

    pub fn get(&self, root: &str) -> Option<&Account> {
        self.accounts.get(root)
    }

    /// The runs finished since the last call.
    pub fn take_finished(&mut self) -> Vec<FinishedRun> {
        std::mem::take(&mut self.finished)
    }

    fn account(&mut self, root: &str, class: &str, now: Time) -> &mut Account {
        if !self.accounts.contains_key(root) && self.accounts.len() >= self.capacity {
            // Bounded: drop the account idle longest (its summary is lost, as an evicted session's is).
            if let Some(victim) = self
                .accounts
                .iter()
                .filter(|(_, a)| a.in_flight == 0)
                .min_by_key(|(_, a)| a.last_active())
                .map(|(k, _)| k.clone())
            {
                self.accounts.remove(&victim);
            }
        }
        self.accounts
            .entry(root.to_owned())
            .or_insert_with(|| Account::new(class, now))
    }

    /// Record the session's identity hints (the root's own; subagents inherit).
    pub fn identify(
        &mut self,
        root: &str,
        class: &str,
        kind: Option<&'static str>,
        user: Option<&str>,
        tenant: Option<&str>,
        now: Time,
    ) {
        let a = self.account(root, class, now);
        if kind.is_some() {
            a.kind = kind;
        }
        if let Some(u) = user {
            a.user = Some(u.to_owned());
        }
        if let Some(t) = tenant {
            a.tenant = Some(t.to_owned());
        }
    }

    /// The frontend or client said whether the root's next request starts with a new user
    /// message: if it does, the run before it ended at the hand-back (a stop point whatever the
    /// gap); if not (a tool result), no gap ends the run.
    pub fn next_starts_with_user(&mut self, root: &str, class: &str, now: Time, user: bool) {
        let a = self.account(root, class, now);
        a.next_is_user = Some(user);
        a.told = true;
    }

    /// Close the runs of sessions idle longer than the stop gap (that the frontend or client
    /// does not mark), so a run is known to be over when its gap passes rather than at the
    /// session's next request or end, as the scorer counts it. At most every tenth of the stop
    /// gap; O(accounts).
    pub fn sweep(&mut self, now: Time) {
        let every = self.stop_gap_s / 10.0;
        if self
            .swept
            .is_some_and(|at| now.since(at).as_secs_f64() < every)
        {
            return;
        }
        self.swept = Some(now);
        let (stop_gap, max_runs) = (self.stop_gap_s, self.max_runs);
        let mut closed = Vec::new();
        for (root, a) in &mut self.accounts {
            let Some(idle) = a.idle_since else { continue };
            // A client retrying a refusal is waiting on us whatever the frontend says; one that
            // has not retried for the stop gap gave up.
            if a.in_flight > 0 || (a.told && !a.retrying) || a.run_closed {
                continue;
            }
            if now.since(idle).as_secs_f64() > stop_gap {
                a.stops += 1;
                a.run_closed = true;
                let by = if a.retrying { "gave_up" } else { "gap" };
                a.retrying = false;
                let allow = self.allowances.get(&a.class).copied();
                closed.push(close_run(root, a, idle, by, max_runs, allow));
            }
        }
        // In a fixed order, whatever the map's.
        closed.sort_by(|x, y| x.end.cmp(&y.end).then_with(|| x.session.cmp(&y.session)));
        for run in closed {
            self.finish_run(run);
        }
    }

    /// The client said this request's last message is a new user message (F35's name).
    pub fn next_is_user(&mut self, root: &str, class: &str, now: Time) {
        self.next_starts_with_user(root, class, now, true);
    }

    /// An attempt of the session was classified (entered the router). Returns whether it opened
    /// a run and the retry backoff before it (added delay), seconds.
    pub fn start(&mut self, root: &str, class: &str, now: Time) -> (bool, f64) {
        let (stop_gap, max_runs) = (self.stop_gap_s, self.max_runs);
        let allow = self.allowances.get(class).copied();
        let a = self.account(root, class, now);
        let mut closed = None;
        let mut backoff = 0.0;
        if a.in_flight == 0 {
            if let Some(idle) = a.idle_since {
                let gap = now.since(idle).as_secs_f64();
                let stop = a.next_is_user.unwrap_or(gap > stop_gap);
                if a.retrying && !a.run_closed {
                    // Retry backoff after a refusal or expiry: added delay, not client time,
                    // never a stop point.
                    backoff = gap;
                    a.run.waiting_s += gap;
                } else if a.run_closed {
                    // The sweep closed the run when its gap passed the stop gap.
                    a.client_s += gap;
                    a.run_closed = false;
                    a.run = RunState::new(now);
                } else if stop {
                    a.client_s += gap;
                    a.stops += 1;
                    let by = if a.next_is_user == Some(true) {
                        "user"
                    } else {
                        "gap"
                    };
                    closed = Some(close_run(root, a, idle, by, max_runs, allow));
                    a.run = RunState::new(now);
                } else {
                    // Client time inside the run counts as uncontended time.
                    a.client_s += gap;
                    a.run.unc_s += gap;
                }
            }
            a.retrying = false;
            a.next_is_user = None;
            a.busy_since = Some(now);
        }
        a.in_flight += 1;
        let opened = a.run.attempts == 0;
        a.run.attempts += 1;
        if let Some(run) = closed {
            self.finish_run(run);
        }
        (opened, backoff)
    }

    /// A request of the main agent (the root's own) added `waiting_s` (router wait, or a refused
    /// attempt's whole time) and `prefill_excess_s` to its run's delay, with `s_ref_s` of
    /// reference service.
    pub fn note_delay(&mut self, root: &str, waiting_s: f64, prefill_excess_s: f64, s_ref_s: f64) {
        if let Some(a) = self.accounts.get_mut(root) {
            a.run.waiting_s += waiting_s.max(0.0);
            a.run.prefill_excess_s += prefill_excess_s.max(0.0);
            a.run.s_ref_s += s_ref_s.max(0.0);
        }
    }

    /// An attempt left the router or engine: completed (`on_target` when its class has limits;
    /// `uncontended_s` when the step model prices it), or aborted (`refused` when it never
    /// reached an engine).
    #[allow(clippy::too_many_arguments)]
    pub fn end(
        &mut self,
        root: &str,
        now: Time,
        on_target: Option<bool>,
        completed: bool,
        refused: bool,
        uncontended_s: Option<f64>,
    ) {
        let keep_s = self.floor_window_s;
        let Some(a) = self.accounts.get_mut(root) else {
            return;
        };
        if a.in_flight == 0 {
            return;
        }
        a.in_flight -= 1;
        if completed {
            a.turns += 1;
            a.run.requests += 1;
            if let Some(ok) = on_target {
                a.judged += 1;
                a.on_target += u32::from(ok);
            }
            match uncontended_s {
                Some(u) => {
                    a.run.busy_unc_s += u.max(0.0);
                    a.run.attained_s += u.max(0.0);
                }
                None => a.run.unc_known = false,
            }
        }
        if refused {
            a.refusals += 1;
        }
        if a.in_flight == 0 {
            a.retrying = refused && !completed;
            let since = a.busy_since.take().unwrap_or(now);
            let busy = now.since(since).as_secs_f64();
            a.server_s += busy;
            a.run.server_s += busy;
            a.run.unc_s += a.run.busy_unc_s.min(busy);
            a.run.busy_unc_s = 0.0;
            a.idle_since = Some(now);
        }
        if completed {
            a.run.sample(now, keep_s);
        }
    }

    /// f49q: a completed request of `root` took `seconds` standalone.
    pub fn add_service(&mut self, root: &str, seconds: f64) {
        if let Some(a) = self.accounts.get_mut(root) {
            a.service_s += seconds.max(0.0);
        }
    }

    /// The session waited in the lobby (held by the gate).
    pub fn hold(&mut self, root: &str) {
        if let Some(a) = self.accounts.get_mut(root) {
            a.holds += 1;
        }
    }

    /// The session ended: close its run, write its summary and forget it. `session` is the
    /// root's id.
    pub fn finish(&mut self, session: &str) -> Option<Account> {
        let mut a = self.accounts.remove(session)?;
        let end = a.last_active();
        if !a.run_closed {
            let allow = self.allowances.get(&a.class).copied();
            let run = close_run(session, &mut a, end, "end", self.max_runs, allow);
            self.finish_run(run);
        }
        if let Some(log) = self.log.as_mut() {
            let _ = writeln!(log, "{}", summary_line(session, &a, end));
        }
        Some(a)
    }

    fn finish_run(&mut self, run: FinishedRun) {
        if let Some(log) = self.log.as_mut() {
            let _ = writeln!(log, "{}", run_line(&run));
        }
        if self.finished.len() >= MAX_FINISHED {
            self.finished.remove(0);
        }
        self.finished.push(run);
    }
}

/// Close `a`'s current run at `end` (a run with no completed request has no uncontended time).
fn close_run(
    session: &str,
    a: &mut Account,
    end: Time,
    ended_by: &'static str,
    max_runs: usize,
    allowance: Option<(f64, f64)>,
) -> FinishedRun {
    let run = &a.run;
    let allowance_s = allowance.map(|(base, per_ref)| base + per_ref * run.s_ref_s);
    a.d1_total_s += run.d1_s();
    a.allowance_total_s += allowance_s.unwrap_or(0.0);
    let uncontended_s = (run.unc_known && run.requests > 0 && run.unc_s > 0.0).then_some(run.unc_s);
    let finished = FinishedRun {
        session: session.to_owned(),
        class: a.class.clone(),
        kind: a.kind,
        start: run.start,
        end,
        server_s: run.server_s,
        uncontended_s,
        requests: run.requests,
        ended_by,
        waiting_s: run.waiting_s,
        prefill_excess_s: run.prefill_excess_s,
        s_ref_s: run.s_ref_s,
        allowance_s,
    };
    if a.runs.len() < max_runs {
        a.runs
            .push((finished.wall_s(), finished.server_s, finished.uncontended_s));
    } else {
        a.runs_dropped += 1;
    }
    finished
}

fn opt_str(v: Option<&str>) -> String {
    v.map_or_else(|| "null".to_owned(), |s| format!("{s:?}"))
}

fn opt_num(v: Option<f64>, digits: usize) -> String {
    v.map_or_else(|| "null".to_owned(), |x| format!("{x:.digits$}"))
}

/// The `run` line for a finished run.
pub fn run_line(r: &FinishedRun) -> String {
    format!(
        "{{\"event\":\"run\",\"session\":{:?},\"class\":{:?},\"kind\":{},\"start_s\":{:.1},\"end_s\":{:.1},\
\"wall_s\":{:.1},\"server_s\":{:.1},\"uncontended_s\":{},\"slowdown\":{},\"requests\":{},\"ended_by\":{:?},\
\"d1_s\":{:.2},\"waiting_s\":{:.2},\"prefill_excess_s\":{:.2},\"s_ref_s\":{:.2},\"allowance_s\":{}}}",
        r.session,
        r.class,
        opt_str(r.kind),
        r.start.as_secs_f64(),
        r.end.as_secs_f64(),
        r.wall_s(),
        r.server_s,
        opt_num(r.uncontended_s, 2),
        opt_num(r.slowdown(), 3),
        r.requests,
        r.ended_by,
        r.d1_s(),
        r.waiting_s,
        r.prefill_excess_s,
        r.s_ref_s,
        opt_num(r.allowance_s, 2),
    )
}

/// The `session_summary` line for an ended session.
pub fn summary_line(session: &str, a: &Account, end: Time) -> String {
    let wall = end.since(a.started).as_secs_f64();
    let share = if wall > 0.0 { a.server_s / wall } else { 0.0 };
    let runs = a
        .runs
        .iter()
        .map(|(w, s, u)| format!("[{w:.1},{s:.1},{}]", opt_num(*u, 1)))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"event\":\"session_summary\",\"session\":{session:?},\"user\":{},\"tenant\":{},\"class\":{:?},\"kind\":{},\
\"started_s\":{:.1},\"ended_s\":{:.1},\"wall_s\":{wall:.1},\"server_s\":{:.1},\"client_s\":{:.1},\"server_share\":{share:.4},\
\"turns\":{},\"on_target\":{},\"judged\":{},\"holds\":{},\"refusals\":{},\"stops\":{},\"runs\":[{runs}],\"runs_dropped\":{}}}",
        opt_str(a.user.as_deref()),
        opt_str(a.tenant.as_deref()),
        a.class,
        opt_str(a.kind),
        a.started.as_secs_f64(),
        end.as_secs_f64(),
        a.server_s,
        a.client_s,
        a.turns,
        a.on_target,
        a.judged,
        a.holds,
        a.refusals,
        a.stops,
        a.runs_dropped,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: f64) -> Time {
        Time::from_secs_f64(s)
    }

    #[test]
    fn server_and_client_time_split_the_wall_and_long_gaps_are_stop_points() {
        let mut e = Experience::new(16, 30.0, 8);
        e.identify(
            "s",
            "standard",
            Some("interactive"),
            Some("u1"),
            None,
            t(0.0),
        );
        // Request 1: 0 to 10 s (lobby wait and service).
        e.start("s", "standard", t(0.0));
        e.end("s", t(10.0), Some(true), true, false, None);
        // Tool gap of 5 s, request 2: 15 to 20 s.
        e.start("s", "standard", t(15.0));
        e.end("s", t(20.0), Some(false), true, false, None);
        // A person reads for 100 s: stop point. Request 3: 120 to 125 s.
        e.start("s", "standard", t(120.0));
        e.end("s", t(125.0), Some(true), true, false, None);
        let a = e.finish("s").unwrap();
        assert!((a.server_s - 20.0).abs() < 1e-9);
        assert!((a.client_s - 105.0).abs() < 1e-9);
        assert_eq!((a.turns, a.on_target, a.judged, a.stops), (3, 2, 3, 1));
        assert_eq!(a.runs, vec![(20.0, 15.0, None), (5.0, 5.0, None)]);
        let line = summary_line("s", &a, t(125.0));
        assert!(line.contains("\"server_share\":0.1600"), "{line}");
        assert!(line.contains("\"kind\":\"interactive\""), "{line}");
        assert!(
            line.contains("\"runs\":[[20.0,15.0,null],[5.0,5.0,null]]"),
            "{line}"
        );
    }

    #[test]
    fn overlapping_subagent_requests_count_once_and_a_user_turn_is_a_stop_point() {
        let mut e = Experience::new(16, 30.0, 8);
        e.start("root", "priority", t(0.0));
        e.start("root", "priority", t(2.0)); // a subagent's request, counted for its root
        e.end("root", t(5.0), Some(true), true, false, None);
        e.end("root", t(8.0), Some(true), true, false, None);
        // A short gap, but the client says the next request starts with a user message.
        e.next_is_user("root", "priority", t(9.0));
        e.start("root", "priority", t(10.0));
        e.end("root", t(12.0), None, false, true, None);
        let a = e.finish("root").unwrap();
        assert!((a.server_s - 10.0).abs() < 1e-9);
        assert_eq!((a.stops, a.refusals, a.turns), (1, 1, 2));
        assert_eq!(a.runs, vec![(8.0, 8.0, None), (2.0, 2.0, None)]);
        let runs = e.take_finished();
        assert_eq!(
            runs.iter().map(|r| r.ended_by).collect::<Vec<_>>(),
            vec!["user", "end"]
        );
    }

    #[test]
    fn a_known_tool_result_keeps_the_run_whatever_the_gap() {
        let mut e = Experience::new(16, 30.0, 8);
        e.start("s", "standard", t(0.0));
        e.end("s", t(1.0), None, true, false, Some(1.0));
        // 100 s later, but the frontend says the request continues with a tool result.
        e.next_starts_with_user("s", "standard", t(101.0), false);
        e.start("s", "standard", t(101.0));
        e.end("s", t(102.0), None, true, false, Some(1.0));
        let a = e.get("s").unwrap();
        assert_eq!(a.stops, 0);
        // Uncontended: 1 + 100 s of client time + 1.
        assert!((a.run.progress_s() - 102.0).abs() < 1e-9);
    }

    #[test]
    fn a_runs_uncontended_time_caps_each_busy_interval_and_adds_client_gaps() {
        let mut e = Experience::new(16, 180.0, 8);
        // Two overlapping requests, 3 s and 2 s alone, in flight together 0 to 4 s: the interval
        // caps them at 4 s. Then a 6 s tool gap, then a request of 1 s alone that takes 5 s.
        e.start("s", "standard", t(0.0));
        e.start("s", "standard", t(0.5));
        e.end("s", t(3.0), None, true, false, Some(3.0));
        e.end("s", t(4.0), None, true, false, Some(2.0));
        e.start("s", "standard", t(10.0));
        e.end("s", t(15.0), None, true, false, Some(1.0));
        let a = e.get("s").unwrap();
        assert!((a.run.attained_s - 6.0).abs() < 1e-9);
        assert!((a.run.progress_s() - (4.0 + 6.0 + 1.0)).abs() < 1e-9);
        // A stop point: the run ends at 15 s with wall 15 s over 11 s uncontended.
        e.start("s", "standard", t(400.0));
        let runs = e.take_finished();
        assert_eq!(runs.len(), 1);
        assert_eq!((runs[0].ended_by, runs[0].requests), ("gap", 3));
        assert!((runs[0].uncontended_s.unwrap() - 11.0).abs() < 1e-9);
        assert!((runs[0].slowdown().unwrap() - 15.0 / 11.0).abs() < 1e-9);
        let line = run_line(&runs[0]);
        assert!(line.contains("\"uncontended_s\":11.00"), "{line}");
        assert!(line.contains("\"slowdown\":1.364"), "{line}");
        // The new run starts with nothing attained.
        assert_eq!(e.get("s").unwrap().run.attained_s, 0.0);
        assert_eq!(e.get("s").unwrap().run.start, t(400.0));
    }

    #[test]
    fn the_sweep_closes_a_run_once_its_gap_passes_the_stop_gap_unless_the_client_marks_stops() {
        let mut e = Experience::new(16, 180.0, 8);
        e.start("s", "standard", t(0.0));
        e.end("s", t(10.0), None, true, false, Some(5.0));
        e.start("told", "standard", t(0.0));
        e.end("told", t(10.0), None, true, false, Some(5.0));
        e.next_starts_with_user("told", "standard", t(5.0), false);
        e.sweep(t(100.0));
        assert!(e.take_finished().is_empty());
        e.sweep(t(200.0));
        let runs = e.take_finished();
        assert_eq!(runs.len(), 1);
        assert_eq!((runs[0].session.as_str(), runs[0].end), ("s", t(10.0)));
        // Swept once; the next request starts a new run without closing another, and the
        // session's end closes only that one.
        e.sweep(t(400.0));
        e.start("s", "standard", t(500.0));
        assert!(e.take_finished().is_empty());
        e.end("s", t(501.0), None, true, false, Some(1.0));
        let a = e.finish("s").unwrap();
        assert_eq!(a.stops, 1);
        assert_eq!(a.runs.len(), 2);
        assert_eq!(e.take_finished().len(), 1);
    }

    #[test]
    fn progress_samples_answer_the_floor_window() {
        let mut e = Experience::new(16, 180.0, 8).with_floor_window(100.0);
        for i in 0..10 {
            let at = f64::from(i) * 50.0;
            e.start("s", "batch", t(at));
            e.end("s", t(at + 10.0), None, true, false, Some(1.0));
        }
        let run = &e.get("s").unwrap().run;
        // Samples older than the window are dropped but the newest one before it stays.
        let (at, p) = run.progress_at(t(460.0 - 100.0));
        assert_eq!(at, t(360.0));
        assert!(p > 0.0);
        assert!(run.progress.len() <= 4, "{}", run.progress.len());
        // Before any sample: the run's start.
        let fresh = RunState::new(t(5.0));
        assert_eq!(fresh.progress_at(t(100.0)), (t(5.0), 0.0));
    }

    #[test]
    fn accounts_stay_bounded() {
        let mut e = Experience::new(2, 30.0, 8);
        for (i, s) in ["a", "b", "c"].iter().enumerate() {
            e.start(s, "batch", t(i as f64));
            e.end(s, t(i as f64 + 1.0), None, true, false, None);
        }
        assert!(e.get("a").is_none());
        assert!(e.get("c").is_some());
    }

    #[test]
    fn retry_backoff_is_delay_not_client_time_and_a_client_that_stops_retrying_gave_up() {
        let mut e = Experience::new(16, 180.0, 8)
            .with_allowances(HashMap::from([("standard".to_owned(), (10.0, 0.25))]));
        assert_eq!(e.start("s", "standard", t(0.0)), (true, 0.0));
        e.note_delay("s", 60.0, 0.0, 0.0);
        e.end("s", t(60.0), None, false, true, None);
        // Retried 300 s later: past the stop gap, but backoff, not a stop point.
        assert_eq!(e.start("s", "standard", t(360.0)), (false, 300.0));
        e.note_delay("s", 0.0, 2.0, 4.0);
        e.end("s", t(370.0), None, true, false, Some(4.0));
        let a = e.get("s").unwrap();
        assert_eq!((a.stops, a.client_s), (0, 0.0));
        assert!((a.run.d1_s() - 362.0).abs() < 1e-9);
        // Over its allowance (10 + 0.25 x 4) by 351 s.
        assert!((a.delay_deficit_s(Some((10.0, 0.25))) - 351.0).abs() < 1e-9);
        // A session refused at its start whose client stops retrying gave up.
        e.start("g", "standard", t(0.0));
        e.note_delay("g", 30.0, 0.0, 0.0);
        e.end("g", t(30.0), None, false, true, None);
        e.sweep(t(300.0));
        let runs = e.take_finished();
        assert_eq!(runs.len(), 1);
        assert_eq!((runs[0].ended_by, runs[0].requests), ("gave_up", 0));
        assert!((runs[0].d1_s() - 30.0).abs() < 1e-9);
        assert_eq!(runs[0].allowance_s, Some(10.0));
    }
}
