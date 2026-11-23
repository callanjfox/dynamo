// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! f49l3 `admission.sla_budget.mode: rebalance`: the step of [`SlaBudget`] in that mode (see the
//! module docs of [`crate::sla_budget`], items 9 to 15). Off (absent) by default.

use super::*;
// f49n: the transfer step shared with `target_loop.rebalance`.
use crate::transfer::{down, factor, give_way, share, up};

/// What a rebalance step found among the warm heads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Case {
    /// No class short.
    Clear,
    /// Some class short and some class over target.
    Feasible,
    /// Every class with a head short.
    Infeasible,
}

impl Case {
    pub fn as_str(self) -> &'static str {
        match self {
            Case::Clear => "clear",
            Case::Feasible => "feasible",
            Case::Infeasible => "infeasible",
        }
    }
}

/// The rebalance mode's state.
#[derive(Debug, Default)]
pub(super) struct RbState {
    /// Each class's attainment on admitted work alone, smoothed.
    meas_adm: HashMap<String, f64>,
    /// Per top class, the refusals made while the fallback held `K_p` under 1 (step, count).
    refused_own: HashMap<String, VecDeque<(Time, u64)>>,
    /// Each class's own step-target setpoint (classes with a service level, with `step_target`).
    pub(super) setpoints: BTreeMap<String, f64>,
    /// In the fallback: since when admitted priority has been over target.
    ok_since: Option<Time>,
    /// After the fallback: `K_p` recovers to 1 at the capped grow rate.
    recovering: bool,
    /// The latest step's case.
    case: Option<Case>,
}

impl RbState {
    pub(super) fn new(policy: &ServicePolicy) -> Self {
        let setpoints = policy
            .admission
            .step_target
            .as_ref()
            .map(|st| {
                policy
                    .classes
                    .iter()
                    .filter(|c| c.service_level.is_some())
                    .map(|c| (c.name.clone(), st.target_frac))
                    .collect()
            })
            .unwrap_or_default();
        Self {
            setpoints,
            ..Self::default()
        }
    }
}

/// Note `fresh` refusals at `now` in `q` and drop those older than `window_s`: the count left.
fn tally(q: &mut VecDeque<(Time, u64)>, now: Time, fresh: u64, window_s: f64) -> u64 {
    if fresh > 0 {
        q.push_back((now, fresh));
    }
    while q
        .front()
        .is_some_and(|(t, _)| now.since(*t).as_secs_f64() >= window_s)
    {
        q.pop_front();
    }
    q.iter().map(|(_, n)| n).sum()
}

// f49n: `down` and `up` are the shared ones ([`crate::transfer`]).

/// One class's figures at a step, for the log.
struct Row {
    class: String,
    measured: Option<f64>,
    admitted: Option<f64>,
    offered: Option<f64>,
    forecast: Option<f64>,
    signal: Option<f64>,
    head: Option<f64>,
    head_adm: Option<f64>,
}

impl SlaBudget {
    /// f49l3: whether `mode: rebalance` is on.
    pub fn rebalancing(&self) -> bool {
        self.rb.is_some()
    }

    /// f49l3: each class's own step-target setpoint (`None` without `mode: rebalance` or without
    /// `admission.step_target`).
    pub fn class_setpoints(&self) -> Option<&BTreeMap<String, f64>> {
        self.rb
            .as_ref()
            .map(|r| &r.setpoints)
            .filter(|m| !m.is_empty())
    }

    /// f49l3: the latest rebalance step's case (`None` before one or without the mode).
    pub fn case(&self) -> Option<Case> {
        self.rb.as_ref().and_then(|r| r.case)
    }

    /// f49l3: whether `K_p` is recovering after the fallback.
    pub fn recovering(&self) -> bool {
        self.rb.as_ref().is_some_and(|r| r.recovering)
    }

    /// f49l3: one rebalance step (its interval has passed; `dt` seconds since the last).
    pub(super) fn step_rebalance(
        &mut self,
        policy: &ServicePolicy,
        now: Time,
        dt: f64,
        t_now: f64,
        inputs: &HashMap<String, ClassInput>,
    ) -> Vec<String> {
        let mut st = self.rb.take().expect("rebalance state");
        let rb_cfg = self.cfg.rebalance_config();
        let alpha = 1.0 - (-dt / self.cfg.smooth_s).exp();
        let blend = self.cfg.blend;
        // (i) The top class's refusals over the interval just ended, made while the fallback held
        // K_p under 1, are the controller's own: not misses for its control.
        let own_refusals = self.mode == SlaMode::Priority && self.k_top < 1.0;
        let mut heads: HashMap<String, f64> = HashMap::new();
        let mut heads_adm: HashMap<String, f64> = HashMap::new();
        let mut demand: HashMap<String, bool> = HashMap::new();
        let mut rows = Vec::with_capacity(policy.classes.len());
        for class in &policy.classes {
            let name = class.name.as_str();
            let input = inputs.get(name).copied().unwrap_or_default();
            let seen = self
                .lost_seen
                .insert(name.to_owned(), input.lost)
                .unwrap_or(input.lost);
            let fresh = input.lost.saturating_sub(seen);
            let window_s = class
                .service_level
                .as_ref()
                .map_or(self.cfg.interval_s, |l| l.window_s);
            let top = self.is_top(name);
            let refused = tally(
                self.refused.entry(name.to_owned()).or_default(),
                now,
                fresh,
                window_s,
            );
            let own = tally(
                st.refused_own.entry(name.to_owned()).or_default(),
                now,
                if top && own_refusals { fresh } else { 0 },
                window_s,
            );
            demand.insert(name.to_owned(), input.waiting || fresh > 0);
            let share = |extra: u64| {
                input.counts.and_then(|(n, met)| {
                    let total = n + extra as usize;
                    (total >= MIN_WINDOW_COMPLETIONS).then(|| met as f64 / total as f64)
                })
            };
            let admitted = share(0);
            let offered = share(refused);
            // The top classes on offered work less the fallback's own refusals, the others on
            // admitted work (f49l).
            let measured = if top {
                share(refused.saturating_sub(own))
            } else {
                admitted
            };
            ema(&mut self.meas, name, measured, alpha);
            ema(&mut st.meas_adm, name, admitted, alpha);
            ema(&mut self.fcst, name, input.forecast, alpha);
            let signal = match (self.fcst.get(name), self.meas.get(name)) {
                (Some(f), Some(m)) => Some(blend * f + (1.0 - blend) * m),
                (f, m) => f.or(m).copied(),
            };
            let target = class
                .service_level
                .as_ref()
                .map(|l| l.attainment + self.cfg.margin(name));
            let head = target.zip(signal).map(|(t, s)| s - t);
            let head_adm = target
                .zip(st.meas_adm.get(name).copied())
                .map(|(t, s)| s - t);
            // f49l2: warm-up, latched.
            let warm = self.warm.contains(name)
                || self.cfg.is_warm(
                    t_now,
                    class.service_level.is_some(),
                    input.counts.map(|(n, _)| n),
                );
            if warm && !self.warm.contains(name) {
                self.warm.insert(name.to_owned());
            }
            if warm {
                if let Some(h) = head {
                    heads.insert(name.to_owned(), h);
                }
                if let Some(h) = head_adm {
                    heads_adm.insert(name.to_owned(), h);
                }
            }
            rows.push(Row {
                class: name.to_owned(),
                measured,
                admitted,
                offered,
                forecast: input.forecast,
                signal,
                head,
                head_adm,
            });
        }
        // f49l2: a class's lever moves only with it and the top classes warm.
        let top_warm = self.top.iter().all(|c| self.warm.contains(c));
        let gate: HashMap<String, bool> = policy
            .classes
            .iter()
            .map(|c| (c.name.clone(), top_warm && self.warm.contains(&c.name)))
            .collect();
        let may = |c: &str| gate.get(c).copied().unwrap_or(false);
        // f49l2: every multiplier within the rate caps.
        let caps = self.cfg.rate_caps();
        let bound = |f: f64| caps.cap(f, dt);
        // f49n: the shared transfer step's capped multiplicative moves.
        let shrink = |h: f64| factor(caps, -rb_cfg.shrink_per_min, share(h, rb_cfg.full_at), dt);
        let grow = |h: f64| factor(caps, rb_cfg.grow_per_min, share(h, rb_cfg.full_at), dt);
        let top0 = self.top.first().cloned().unwrap_or_default();
        let floor: HashMap<String, f64> = policy
            .classes
            .iter()
            .map(|c| {
                (
                    c.name.clone(),
                    self.cfg.rb_floor(&c.name, self.is_top(&c.name)),
                )
            })
            .collect();
        let ceiling: HashMap<String, f64> = policy
            .classes
            .iter()
            .map(|c| {
                (
                    c.name.clone(),
                    self.cfg.rb_ceiling(&c.name, self.is_top(&c.name)),
                )
            })
            .collect();
        let lo = |c: &str| floor.get(c).copied().unwrap_or(0.5);
        let hi = |c: &str| ceiling.get(c).copied().unwrap_or(2.0);
        let head_top_adm = self
            .top
            .iter()
            .filter_map(|c| heads_adm.get(c).copied())
            .reduce(f64::min);
        let worst = heads.values().fold(0.0_f64, |a, &h| a.min(h));
        let case = if worst >= 0.0 {
            Case::Clear
        } else if heads.values().any(|&h| h > 0.0) {
            Case::Feasible
        } else {
            Case::Infeasible
        };
        st.case = Some(case);
        let mut ranked: Vec<&crate::config::ClassPolicy> = policy.classes.iter().collect();
        ranked.sort_by_key(|c| c.rank);
        match self.mode {
            SlaMode::Normal => {
                match case {
                    Case::Feasible => {
                        // Surplus to short, whatever the rank; a short class never gives way.
                        for c in &ranked {
                            let name = c.name.as_str();
                            if !may(name) || (st.recovering && self.is_top(name)) {
                                continue;
                            }
                            let k = self.k(name);
                            let next = match heads.get(name) {
                                Some(&h) if h < 0.0 => up(k, grow(h), hi(name)),
                                Some(&h) if h > 0.0 => down(k, shrink(h), lo(name)),
                                None if c.service_level.is_none() => {
                                    down(k, shrink(worst), lo(name))
                                }
                                _ => k,
                            };
                            self.set_k(name, next);
                        }
                    }
                    Case::Infeasible => {
                        // Rank order: the lowest-ranked class above its floor gives way, while a
                        // class ranked above it (the top classes on admitted work) is short.
                        // f49n: the shared rank-ordered give-way.
                        let order: Vec<&str> = ranked
                            .iter()
                            .rev()
                            .map(|c| c.name.as_str())
                            .take_while(|c| !self.is_top(c))
                            .collect();
                        if let Some(name) = give_way(&order, |c| self.k(c), lo) {
                            let rank = policy.class(Some(name)).rank;
                            let drive = ranked
                                .iter()
                                .filter(|a| a.rank < rank)
                                .filter_map(|a| {
                                    if self.is_top(&a.name) {
                                        heads_adm.get(&a.name).copied()
                                    } else {
                                        heads.get(&a.name).copied()
                                    }
                                })
                                .fold(0.0_f64, f64::min);
                            // f49l2: held in warm-up, it holds the step's shrink.
                            if may(name) && drive < 0.0 {
                                let k = self.k(name);
                                self.set_k(name, down(k, shrink(drive), lo(name)));
                            }
                        }
                        // (v) K_p on admitted priority: up while over target (to its ceiling),
                        // down while short but not under 1 (under 1 only in the fallback).
                        if may(top0.as_str()) && !st.recovering {
                            let k = self.k_top;
                            self.k_top = match head_top_adm {
                                Some(h) if h > 0.0 => up(k, grow(h), hi(top0.as_str())),
                                Some(h) if h < 0.0 => down(k, shrink(h), 1.0),
                                _ => k,
                            };
                        }
                    }
                    Case::Clear => {
                        // f49l's growth of the lower classes with demand.
                        let rank = |c: &str| policy.class(Some(c)).rank;
                        for c in &ranked {
                            let name = c.name.as_str();
                            if self.is_top(name) {
                                continue;
                            }
                            let h = heads
                                .iter()
                                .filter(|(j, _)| rank(j.as_str()) <= c.rank)
                                .map(|(_, h)| *h)
                                .reduce(f64::min)
                                .unwrap_or(NO_HEAD);
                            let own_ok = heads
                                .get(name)
                                .is_none_or(|own| *own <= self.cfg.max_headroom);
                            if demand.get(name).copied().unwrap_or(false) && own_ok && may(name) {
                                let f =
                                    bound((self.cfg.grow_per_min / 60.0 * h / GROW_REF * dt).exp());
                                let k = self.k(name);
                                self.set_k(name, up(k, f, hi(name)));
                            }
                        }
                    }
                }
                // Each class's own setpoint: a short class tightens it, one over target loosens it.
                let sp = self.cfg.setpoint.clone();
                for (c, v) in st.setpoints.iter_mut() {
                    if !may(c.as_str()) {
                        continue;
                    }
                    match heads.get(c) {
                        Some(&h) if h < 0.0 => {
                            *v = (*v * bound((-sp.shrink_per_min / 60.0 * dt).exp())).max(sp.min);
                        }
                        Some(&h) if h > 0.0 => {
                            *v = (*v * bound((sp.grow_per_min / 60.0 * dt).exp())).min(sp.max);
                        }
                        _ => {}
                    }
                }
                // (ii) After the fallback K_p recovers to 1 at the capped grow rate while admitted
                // priority is not short.
                if st.recovering {
                    if may(top0.as_str()) && head_top_adm.is_none_or(|h| h >= 0.0) {
                        let rate = self.cfg.max_grow_per_min.unwrap_or(rb_cfg.grow_per_min);
                        self.k_top = up(self.k_top, bound((rate / 60.0 * dt).exp()), 1.0);
                    }
                    if self.k_top >= 1.0 {
                        st.recovering = false;
                    }
                }
                // (iv) Fallback: infeasible with every lower K at its floor and the top setpoint at
                // its min, and admitted priority short, for `fallback.after_s`.
                let pinned = self.lower.iter().all(|c| self.k(c) <= lo(c.as_str()))
                    && st
                        .setpoints
                        .get(&top0)
                        .is_none_or(|v| *v <= self.cfg.setpoint.min);
                if case == Case::Infeasible
                    && pinned
                    && may(top0.as_str())
                    && head_top_adm.is_some_and(|h| h < 0.0)
                {
                    let since = *self.short_since.get_or_insert(now);
                    if now.since(since).as_secs_f64() >= self.cfg.fallback.after_s {
                        self.mode = SlaMode::Priority;
                        st.ok_since = None;
                        st.recovering = false;
                    }
                } else {
                    self.short_since = None;
                }
            }
            SlaMode::Priority => {
                // (v) K_p from admitted priority's capacity: down while short, up while over.
                if may(top0.as_str()) {
                    let k = self.k_top;
                    self.k_top = match head_top_adm {
                        Some(h) if h < 0.0 => down(k, shrink(h), lo(top0.as_str())),
                        Some(h) if h > 0.0 => up(k, grow(h), hi(top0.as_str())),
                        _ => k,
                    };
                }
                // (ii) Exit once admitted priority has been over target for `exit_after_s`.
                if head_top_adm.is_some_and(|h| h > 0.0) {
                    let since = *st.ok_since.get_or_insert(now);
                    if now.since(since).as_secs_f64() >= rb_cfg.exit_after_s {
                        self.mode = SlaMode::Normal;
                        self.short_since = None;
                        st.ok_since = None;
                        st.recovering = self.k_top < 1.0;
                    }
                } else {
                    st.ok_since = None;
                }
            }
        }
        if self.setpoint.is_some()
            && let Some(v) = st.setpoints.get(&top0)
        {
            self.setpoint = Some(*v);
        }
        let t = now.since_origin().as_secs_f64();
        let lines: Vec<String> = rows
            .into_iter()
            .map(|r| {
                format!(
                    "{{\"t\":{t:.3},\"class\":{:?},\"measured\":{},\"forecast\":{},\"signal\":{},\"head\":{},\"K\":{:.6},\"setpoint\":{},\"mode\":\"{}\",\"warm\":{},\"measured_admitted\":{},\"measured_offered\":{},\"head_admitted\":{},\"case\":\"{}\",\"recovering\":{}}}",
                    r.class,
                    num(r.measured),
                    num(r.forecast),
                    num(r.signal),
                    num(r.head),
                    self.k(&r.class),
                    num(st.setpoints.get(&r.class).copied()),
                    self.mode.as_str(),
                    may(r.class.as_str()),
                    num(r.admitted),
                    num(r.offered),
                    num(r.head_adm),
                    case.as_str(),
                    st.recovering
                )
            })
            .collect();
        self.rb = Some(st);
        if let Some(LogSink(w)) = self.log.as_mut() {
            for line in &lines {
                let _ = writeln!(w, "{line}");
            }
            let _ = w.flush();
        }
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Mode;
    use crate::scheduler::Scheduler;
    use crate::step_target::StepTargetConfig;

    fn t(s: f64) -> Time {
        Time::from_secs_f64(s)
    }

    /// `mode: rebalance`, unsmoothed (the signal is the step's measure), edited by `f`.
    fn rb(f: impl FnOnce(&mut SlaBudgetConfig)) -> SlaBudgetConfig {
        let mut c = SlaBudgetConfig {
            mode: Some(BudgetMode::Rebalance),
            smooth_s: 1e-3,
            ..SlaBudgetConfig::default()
        };
        f(&mut c);
        c
    }

    fn policy(cfg: SlaBudgetConfig, step_target: bool) -> ServicePolicy {
        let mut p = ServicePolicy {
            mode: Mode::Enforce,
            ..Default::default()
        };
        p.admission.sla_budget = Some(cfg);
        if step_target {
            p.admission.step_target = Some(StepTargetConfig::default());
        }
        p
    }

    fn ctl(cfg: SlaBudgetConfig, step_target: bool) -> (SlaBudget, ServicePolicy) {
        let p = policy(cfg, step_target);
        p.validate().unwrap();
        (SlaBudget::from_policy(&p).unwrap(), p)
    }

    /// Priority's and standard's attainment over 1000 completions (`None`: none), and the
    /// refusals so far of (priority, standard, batch). Nobody waits.
    fn inputs(p: Option<f64>, s: Option<f64>, lost: [u64; 3]) -> HashMap<String, ClassInput> {
        let counts = |a: Option<f64>| a.map(|a| (1000_usize, (a * 1000.0).round() as usize));
        [
            ("priority", counts(p), lost[0]),
            ("standard", counts(s), lost[1]),
            ("batch", None, lost[2]),
        ]
        .into_iter()
        .map(|(c, counts, lost)| {
            (
                c.to_owned(),
                ClassInput {
                    counts,
                    waiting: false,
                    lost,
                    forecast: None,
                },
            )
        })
        .collect()
    }

    fn field(line: &str, key: &str) -> serde_json::Value {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        v[key].clone()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn f49l3_surplus_moves_to_the_short_class_both_ways() {
        // Priority 7 points over 0.92, standard 13 short of 0.73: one minute, full rates.
        let (mut sb, p) = ctl(rb(|_| {}), false);
        sb.step(&p, t(0.0), &inputs(None, None, [0; 3]));
        sb.step(&p, t(60.0), &inputs(Some(0.99), Some(0.60), [0; 3]));
        assert_eq!(sb.case(), Some(Case::Feasible));
        assert!(
            close(sb.k("standard"), 0.01_f64.exp()),
            "{}",
            sb.k("standard")
        );
        assert!(
            close(sb.k("priority"), (-0.02_f64).exp()),
            "{}",
            sb.k("priority")
        );
        // Batch has no target: it gives way at the worst shortfall's scale.
        assert!(close(sb.k("batch"), (-0.02_f64).exp()), "{}", sb.k("batch"));
        // Standard 7 over 0.73, priority 2 short of 0.92: priority grows (40 % of the rate),
        // standard shrinks.
        let (mut sb, p) = ctl(rb(|_| {}), false);
        sb.step(&p, t(0.0), &inputs(None, None, [0; 3]));
        sb.step(&p, t(60.0), &inputs(Some(0.90), Some(0.80), [0; 3]));
        assert!(
            close(sb.k("priority"), 0.004_f64.exp()),
            "{}",
            sb.k("priority")
        );
        assert!(
            close(sb.k("standard"), (-0.02_f64).exp()),
            "{}",
            sb.k("standard")
        );
        assert!(
            close(sb.k("batch"), (-0.008_f64).exp()),
            "{}",
            sb.k("batch")
        );
    }

    #[test]
    fn f49l3_a_short_class_never_shrinks() {
        let (mut sb, p) = ctl(rb(|_| {}), false);
        sb.step(&p, t(0.0), &inputs(None, None, [0; 3]));
        let mut last = sb.k("standard");
        for i in 1..=600 {
            sb.step(
                &p,
                t(60.0 * f64::from(i)),
                &inputs(Some(0.99), Some(0.60), [0; 3]),
            );
            assert!(sb.k("standard") >= last, "{i}");
            last = sb.k("standard");
        }
        // To the ceilings and floors.
        assert_eq!(sb.k("standard"), 2.5);
        assert_eq!(sb.k("priority"), 0.3);
        assert_eq!(sb.k("batch"), 0.333);
        // The scheduler keeps priority's cap at or over standard's (no inversion).
        let mut s = Scheduler::new(policy(rb(|_| {}), false));
        let sbm = s.sla_budget_mut().unwrap();
        sbm.set_k("standard", 2.5);
        sbm.set_k("priority", 0.3);
        let cap = |s: &Scheduler, c: &str| {
            let cp = s.policy().class(Some(c));
            cp.admission.gate_share
                * s.pace_scale(c)
                * s.step_target_scale(c)
                * s.policy().class_overcommit(cp)
                * s.sla_scale(c)
        };
        assert!(cap(&s, "priority") + 1e-12 >= cap(&s, "standard"));
        assert!(cap(&s, "standard") + 1e-12 >= cap(&s, "batch"));
    }

    #[test]
    fn f49l3_infeasible_protects_rank_order() {
        let (mut sb, p) = ctl(rb(|_| {}), false);
        sb.step(&p, t(0.0), &inputs(None, None, [0; 3]));
        let mut seen = Vec::new();
        for i in 1..=400 {
            sb.step(
                &p,
                t(60.0 * f64::from(i)),
                &inputs(Some(0.85), Some(0.60), [0; 3]),
            );
            assert_eq!(sb.case(), Some(Case::Infeasible));
            seen.push((sb.k("batch"), sb.k("standard"), sb.k_top(), sb.mode()));
        }
        for &(b, s, kp, mode) in &seen {
            if b > 0.333 {
                assert_eq!((s, kp), (1.0, 1.0), "{seen:?}");
            }
            if mode == SlaMode::Normal {
                assert_eq!(kp, 1.0, "{seen:?}");
            } else {
                assert_eq!((b, s), (0.333, 0.5), "{seen:?}");
            }
        }
        // Pinned, then the fallback, then K_p down to its floor.
        assert_eq!(sb.mode(), SlaMode::Priority);
        assert_eq!(sb.k_top(), 0.3);
        // Priority short on offered work only (refusals), admitted on target: the lower classes
        // stop giving way for it.
        let (mut sb, p) = ctl(rb(|_| {}), false);
        sb.set_k("batch", 0.333);
        sb.step(&p, t(0.0), &inputs(None, None, [0; 3]));
        for i in 1..=60_u32 {
            let lost = [200 * u64::from(i), 0, 0];
            sb.step(
                &p,
                t(60.0 * f64::from(i)),
                &inputs(Some(0.95), Some(0.60), lost),
            );
            assert_eq!(sb.case(), Some(Case::Infeasible), "{i}");
        }
        assert_eq!(sb.k("standard"), 1.0);
        assert_eq!(sb.mode(), SlaMode::Normal);
    }

    #[test]
    fn f49l3_k_p_follows_admitted_priority_capacity() {
        // Outside the fallback: priority short on offered work (200 refusals a minute), admitted
        // on target up to K_p 1.3 and short over it, standard short: K_p settles under 1.3.
        let (mut sb, p) = ctl(rb(|_| {}), false);
        sb.step(&p, t(0.0), &inputs(None, None, [0; 3]));
        for i in 1..=400_u32 {
            let adm = if sb.k_top() <= 1.3 { 0.97 } else { 0.85 };
            let lost = [200 * u64::from(i), 0, 0];
            sb.step(
                &p,
                t(60.0 * f64::from(i)),
                &inputs(Some(adm), Some(0.60), lost),
            );
            assert_eq!(sb.case(), Some(Case::Infeasible), "{i}");
            assert_eq!(sb.mode(), SlaMode::Normal, "{i}");
        }
        let kp = sb.k_top();
        assert!(
            kp > 1.3 * (-0.02_f64).exp() && kp <= 1.3 * 0.01_f64.exp(),
            "{kp}"
        );
        // In the fallback: admitted on target only at K_p 0.6 or under.
        let (mut sb, p) = ctl(rb(|_| {}), false);
        sb.step(&p, t(0.0), &inputs(None, None, [0; 3]));
        sb.mode = SlaMode::Priority;
        for i in 1..=400 {
            let adm = if sb.k_top() <= 0.6 { 0.95 } else { 0.80 };
            sb.step(
                &p,
                t(60.0 * f64::from(i)),
                &inputs(Some(adm), Some(0.60), [0; 3]),
            );
        }
        let kp = sb.k_top();
        assert!(
            kp > 0.6 * (-0.02_f64).exp() && kp <= 0.6 * 0.01_f64.exp(),
            "{kp}"
        );
        assert_eq!(sb.mode(), SlaMode::Priority, "never 5 minutes over target");
    }

    #[test]
    fn f49l3_the_fallback_excludes_its_own_refusals() {
        let run = |fallback: bool| {
            let (mut sb, p) = ctl(rb(|_| {}), false);
            sb.step(&p, t(0.0), &inputs(None, None, [0; 3]));
            if fallback {
                sb.mode = SlaMode::Priority;
                sb.k_top = 0.5;
            }
            // 950 met of 1000 admitted, 100 refused.
            let lines = sb.step(&p, t(60.0), &inputs(Some(0.95), Some(0.80), [100, 0, 0]));
            lines[0].clone()
        };
        let line = run(true);
        assert_eq!(field(&line, "measured"), serde_json::json!(0.95), "{line}");
        assert_eq!(field(&line, "measured_admitted"), serde_json::json!(0.95));
        let offered = field(&line, "measured_offered").as_f64().unwrap();
        assert!((offered - 950.0 / 1100.0).abs() < 1e-6, "{line}");
        assert_eq!(field(&line, "mode"), serde_json::json!("priority"));
        // Outside the fallback the refusals are misses.
        let line = run(false);
        let measured = field(&line, "measured").as_f64().unwrap();
        assert!((measured - 950.0 / 1100.0).abs() < 1e-6, "{line}");
        // The log has the documented fields.
        for key in [
            "t",
            "class",
            "measured",
            "forecast",
            "signal",
            "head",
            "K",
            "setpoint",
            "mode",
            "warm",
            "measured_admitted",
            "measured_offered",
            "head_admitted",
            "case",
            "recovering",
        ] {
            let v: serde_json::Value = serde_json::from_str(&line).unwrap();
            assert!(v.get(key).is_some(), "{key} in {line}");
        }
    }

    #[test]
    fn f49l3_the_fallback_exits_after_five_minutes_of_admitted_headroom() {
        let cfg = rb(|c| c.max_grow_per_min = Some(0.05));
        let (mut sb, p) = ctl(cfg, false);
        sb.step(&p, t(0.0), &inputs(None, None, [0; 3]));
        sb.mode = SlaMode::Priority;
        sb.k_top = 0.4;
        // Admitted priority 3 points over, offered far short (refusals): headroom from 60 s;
        // a short step at 180 s restarts the clock.
        let mut exited = None;
        for i in 1..=20_u32 {
            let now = 60.0 * f64::from(i);
            let adm = if i == 3 { 0.80 } else { 0.95 };
            sb.step(
                &p,
                t(now),
                &inputs(Some(adm), Some(0.60), [300 * u64::from(i), 0, 0]),
            );
            if exited.is_none() && sb.mode() == SlaMode::Normal {
                exited = Some(now);
                assert!(sb.recovering());
            }
        }
        // Over target from 240 s: out at 540 s.
        assert_eq!(exited, Some(540.0));
        // Recovering at the capped rate: 5 %/min while admitted priority is not short.
        let (mut sb, p) = ctl(rb(|c| c.max_grow_per_min = Some(0.05)), false);
        sb.step(&p, t(0.0), &inputs(None, None, [0; 3]));
        sb.mode = SlaMode::Priority;
        sb.k_top = 0.4;
        let mut i = 0_u32;
        while sb.mode() == SlaMode::Priority {
            i += 1;
            sb.step(
                &p,
                t(60.0 * f64::from(i)),
                &inputs(Some(0.95), Some(0.95), [0; 3]),
            );
        }
        let k = sb.k_top();
        i += 1;
        sb.step(
            &p,
            t(60.0 * f64::from(i)),
            &inputs(Some(0.95), Some(0.95), [0; 3]),
        );
        assert!(
            close(sb.k_top(), (k * 0.05_f64.exp()).min(1.0)),
            "{k} {}",
            sb.k_top()
        );
        for j in 1..=100 {
            sb.step(
                &p,
                t(60.0 * f64::from(i + j)),
                &inputs(Some(0.95), Some(0.95), [0; 3]),
            );
        }
        assert_eq!((sb.k_top(), sb.recovering()), (1.0, false));
    }

    #[test]
    fn f49l3_entry_needs_admitted_priority_short() {
        // Every lever pinned and priority short on offered work only: no fallback (f49l's would
        // enter after 600 s).
        let (mut sb, p) = ctl(rb(|_| {}), false);
        sb.set_k("batch", 0.333);
        sb.set_k("standard", 0.5);
        sb.step(&p, t(0.0), &inputs(None, None, [0; 3]));
        for i in 1..=60_u32 {
            let lost = [300 * u64::from(i), 0, 0];
            sb.step(
                &p,
                t(60.0 * f64::from(i)),
                &inputs(Some(0.95), Some(0.60), lost),
            );
            assert_eq!(sb.mode(), SlaMode::Normal, "{i}");
        }
        // Admitted short too: in after 600 s.
        let (mut sb, p) = ctl(rb(|_| {}), false);
        sb.set_k("batch", 0.333);
        sb.set_k("standard", 0.5);
        sb.step(&p, t(0.0), &inputs(None, None, [0; 3]));
        let mut entered = None;
        for i in 1..=20 {
            sb.step(
                &p,
                t(60.0 * f64::from(i)),
                &inputs(Some(0.85), Some(0.60), [0; 3]),
            );
            if entered.is_none() && sb.mode() == SlaMode::Priority {
                entered = Some(60.0 * f64::from(i));
            }
        }
        assert_eq!(entered, Some(660.0));
    }

    #[test]
    fn f49l3_per_class_setpoints_reach_the_workers() {
        let mut s = Scheduler::new(policy(rb(|_| {}), true));
        s.sla_budget_apply(t(0.0), &inputs(None, None, [0; 3]));
        s.sla_budget_apply(t(60.0), &inputs(Some(0.99), Some(0.60), [0; 3]));
        let sp = s.sla_budget().unwrap().class_setpoints().unwrap().clone();
        // Priority over target loosens its own (1 %/min), standard short tightens its own (5 %/min).
        assert!(close(sp["priority"], 0.85 * 0.01_f64.exp()), "{sp:?}");
        assert!(close(sp["standard"], 0.85 * (-0.05_f64).exp()), "{sp:?}");
        assert!(!sp.contains_key("batch"));
        assert_eq!(s.sla_budget().unwrap().setpoint(), Some(sp["priority"]));
        let st = s.step_target().unwrap();
        assert!(close(st.class_frac("standard"), sp["standard"]));
        assert!(close(st.target_s(), sp["priority"] * 0.05));
        // The configured policy is untouched.
        assert_eq!(
            s.policy()
                .admission
                .step_target
                .as_ref()
                .unwrap()
                .target_frac,
            0.85
        );
    }

    #[test]
    fn f49l3_rate_caps_and_warmup_hold_every_move() {
        // Caps: priority's shrink at 0.5 %/min, standard's growth at 0.4 %/min.
        let cfg = rb(|c| {
            c.max_shrink_per_min = Some(0.005);
            c.max_grow_per_min = Some(0.004);
        });
        let (mut sb, p) = ctl(cfg, true);
        sb.step(&p, t(0.0), &inputs(None, None, [0; 3]));
        sb.step(&p, t(60.0), &inputs(Some(0.99), Some(0.60), [0; 3]));
        assert!(close(sb.k("priority"), (-0.005_f64).exp()));
        assert!(close(sb.k("standard"), 0.004_f64.exp()));
        let sp = sb.class_setpoints().unwrap();
        assert!(close(sp["priority"], 0.85 * 0.004_f64.exp()), "{sp:?}");
        assert!(close(sp["standard"], 0.85 * (-0.005_f64).exp()), "{sp:?}");
        // Warm-up: nothing moves before 1200 s (enough scored turns throughout), then it does.
        let cfg = rb(|c| {
            c.warmup = Some(WarmupConfig {
                min_scored: 50,
                after_s: 1200.0,
            });
        });
        let (mut sb, p) = ctl(cfg, true);
        sb.step(&p, t(0.0), &inputs(None, None, [0; 3]));
        for i in 1..20 {
            let lines = sb.step(
                &p,
                t(60.0 * f64::from(i)),
                &inputs(Some(0.99), Some(0.60), [0; 3]),
            );
            for c in ["priority", "standard", "batch"] {
                assert_eq!(sb.k(c), 1.0, "{i} {c}");
            }
            assert_eq!(sb.class_setpoints().unwrap()["standard"], 0.85);
            assert_eq!(field(&lines[0], "warm"), serde_json::json!(false));
        }
        sb.step(&p, t(1200.0), &inputs(Some(0.99), Some(0.60), [0; 3]));
        assert!(sb.k("standard") > 1.0 && sb.k("priority") < 1.0);
    }

    #[test]
    fn f49l3_options_off_are_f49l() {
        // Absent by default and not serialized; nothing is built.
        let d = SlaBudgetConfig::default();
        assert_eq!((d.mode, d.rebalance.clone()), (None, None));
        let json = serde_json::to_string(&d).unwrap();
        assert!(
            !json.contains("mode") && !json.contains("rebalance"),
            "{json}"
        );
        let p = policy(d, true);
        let sb = SlaBudget::from_policy(&p).unwrap();
        assert!(!sb.rebalancing() && sb.class_setpoints().is_none() && sb.case().is_none());
        // f49l's step (lines without the new fields) and the scheduler's handoff are unchanged.
        let (mut sb, p) = ctl(SlaBudgetConfig::default(), true);
        sb.step(&p, t(0.0), &inputs(None, None, [0; 3]));
        let lines = sb.step(&p, t(60.0), &inputs(Some(0.99), Some(0.60), [0; 3]));
        assert!(!lines[0].contains("measured_admitted"), "{}", lines[0]);
        assert_eq!(sb.k_top(), 1.0);
        let mut s = Scheduler::new(policy(SlaBudgetConfig::default(), true));
        s.sla_budget_apply(t(0.0), &inputs(None, None, [0; 3]));
        s.sla_budget_apply(t(60.0), &inputs(Some(0.99), Some(0.60), [0; 3]));
        let st = s.step_target().unwrap();
        assert_eq!(st.class_frac("standard"), st.target_frac());
        // Validation: priority's floor and ceiling and the rebalance block need the mode.
        let check = |cfg: SlaBudgetConfig| policy(cfg, false).validate().map_err(|e| e.to_string());
        let with_top = |mode: Option<BudgetMode>| SlaBudgetConfig {
            mode,
            floors: [("priority".to_owned(), 0.3)].into_iter().collect(),
            ceilings: [("priority".to_owned(), 1.5)].into_iter().collect(),
            ..SlaBudgetConfig::default()
        };
        assert!(check(with_top(None)).unwrap_err().contains("top-ranked"));
        check(with_top(Some(BudgetMode::Rebalance))).unwrap();
        let err = check(SlaBudgetConfig {
            rebalance: Some(RebalanceConfig::default()),
            ..SlaBudgetConfig::default()
        })
        .unwrap_err();
        assert!(err.contains("needs mode: rebalance"), "{err}");
        for bad in [
            RebalanceConfig {
                shrink_per_min: -1.0,
                ..RebalanceConfig::default()
            },
            RebalanceConfig {
                full_at: 0.0,
                ..RebalanceConfig::default()
            },
            RebalanceConfig {
                exit_after_s: f64::NAN,
                ..RebalanceConfig::default()
            },
        ] {
            let err = check(rb(|c| c.rebalance = Some(bad.clone()))).unwrap_err();
            assert!(err.starts_with("admission.sla_budget: rebalance."), "{err}");
        }
        // The YAML parses.
        let p: ServicePolicy = serde_json::from_str(
            r#"{"admission": {"sla_budget": {"mode": "rebalance",
                "floors": {"priority": 0.3, "standard": 0.5, "batch": 0.333},
                "ceilings": {"priority": 1.5, "standard": 2.5, "batch": 2.0},
                "rebalance": {"shrink_per_min": 0.02, "grow_per_min": 0.01, "full_at": 0.05,
                    "exit_after_s": 300}}}}"#,
        )
        .unwrap();
        p.validate().unwrap();
        let c = p.admission.sla_budget.as_ref().unwrap();
        assert_eq!(c.rebalance_config(), RebalanceConfig::default());
        assert_eq!(
            (c.rb_floor("priority", true), c.rb_ceiling("priority", true)),
            (0.3, 1.5)
        );
        assert!(
            serde_json::from_str::<ServicePolicy>(
                r#"{"admission": {"sla_budget": {"mode": "spread"}}}"#
            )
            .is_err()
        );
    }
}
