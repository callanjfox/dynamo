// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-class attainment, the two ways the admission simulator scores it
//! (`docs/slo-attainment.md` in dynamo-admission-sim):
//!
//! - per request: a request meets its service level when its own TTFT and its own mean ITL are
//!   within the limits;
//! - windowed percentile: at each judgement, the trailing window is judged on its own; it passes
//!   when the p-th percentile TTFT and ITL of the completions in it are within the limits. A
//!   window with fewer than 5 completions is not judged. ITL is judged only over completions that
//!   have one; with none, the window is judged on TTFT alone.
//!
//! A plain snapshot struct, not Prometheus: plugin crates depend on `dynamo-kv-router` alone.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::Duration;

use crate::clock::Time;
use crate::config::{ServiceLevel, ServicePolicy};

/// Fewest completions a window needs to be judged.
pub const MIN_WINDOW_COMPLETIONS: usize = 5;

/// One finished request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Completion {
    pub at: Time,
    pub ttft_s: f64,
    /// Mean inter-token latency; `None` when the request produced one token or the host did not
    /// report it.
    pub itl_s: Option<f64>,
}

/// The p-th percentile as the simulator computes it: sorted, index `min(n - 1, floor(p * n))`.
pub fn percentile(values: &mut [f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let index = ((p * values.len() as f64).floor() as usize).min(values.len() - 1);
    Some(values[index])
}

/// The outcome of judging one window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowJudgement {
    pub passed: bool,
    pub ttft_p: f64,
    pub itl_p: Option<f64>,
    pub completions: usize,
}

#[derive(Debug, Default)]
struct ClassState {
    window: VecDeque<Completion>,
    requests: u64,
    requests_met: u64,
    judged: u64,
    passed: u64,
    hourly: BTreeMap<u64, (u64, u64)>,
    last: Option<WindowJudgement>,
}

/// Attainment for one class at a point in time.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassSnapshot {
    pub class: String,
    pub per_request: Option<f64>,
    pub requests: u64,
    pub windowed: Option<f64>,
    pub windows_judged: u64,
    pub target: Option<f64>,
    pub last_window: Option<WindowJudgement>,
    /// Per hour of scheduler time: (windows judged, windows passed).
    pub hourly: Vec<(u64, u64, u64)>,
}

#[derive(Debug)]
pub struct Metrics {
    levels: HashMap<String, ServiceLevel>,
    classes: HashMap<String, ClassState>,
}

impl Metrics {
    pub fn new(policy: &ServicePolicy) -> Self {
        Self {
            levels: policy
                .classes
                .iter()
                .filter_map(|c| c.service_level.clone().map(|l| (c.name.clone(), l)))
                .collect(),
            classes: policy
                .classes
                .iter()
                .map(|c| (c.name.clone(), ClassState::default()))
                .collect(),
        }
    }

    /// Record a finished request of `class`.
    pub fn record(&mut self, class: &str, completion: Completion) {
        let level = self.levels.get(class);
        let Some(state) = self.classes.get_mut(class) else {
            return;
        };
        state.requests += 1;
        if let Some(level) = level {
            let met = completion.ttft_s <= level.ttft_s
                && completion.itl_s.is_none_or(|itl| itl <= level.itl_s);
            if met {
                state.requests_met += 1;
            }
            state.window.push_back(completion);
            // Keep the window bounded even if `judge` is not called for a while.
            let horizon = Duration::from_secs_f64(level.window_s);
            while state
                .window
                .front()
                .is_some_and(|c| completion.at.since(c.at) > horizon)
            {
                state.window.pop_front();
            }
        }
    }

    /// Judge every class's trailing window at `now` (the simulator does this at each sample tick).
    pub fn judge(&mut self, now: Time) {
        for (class, state) in &mut self.classes {
            let Some(level) = self.levels.get(class) else {
                continue;
            };
            let horizon = Duration::from_secs_f64(level.window_s);
            // The window is (now - window_s, now].
            while state
                .window
                .front()
                .is_some_and(|c| now.since(c.at) >= horizon)
            {
                state.window.pop_front();
            }
            if state.window.len() < MIN_WINDOW_COMPLETIONS {
                state.last = None;
                continue;
            }
            let mut ttft: Vec<f64> = state.window.iter().map(|c| c.ttft_s).collect();
            let mut itl: Vec<f64> = state.window.iter().filter_map(|c| c.itl_s).collect();
            let ttft_p = percentile(&mut ttft, level.percentile).unwrap_or(f64::INFINITY);
            let itl_p = percentile(&mut itl, level.percentile);
            let passed = ttft_p <= level.ttft_s && itl_p.is_none_or(|v| v <= level.itl_s);
            state.judged += 1;
            if passed {
                state.passed += 1;
            }
            let hour = now.since_origin().as_secs() / 3600;
            let entry = state.hourly.entry(hour).or_insert((0, 0));
            entry.0 += 1;
            if passed {
                entry.1 += 1;
            }
            state.last = Some(WindowJudgement {
                passed,
                ttft_p,
                itl_p,
                completions: state.window.len(),
            });
        }
    }

    /// Share of the class's completions in the trailing window (`window_s` before `now`) that met
    /// its service level, the simulator's `windowAtt`; `None` under [`MIN_WINDOW_COMPLETIONS`] or
    /// for a class without one. What shedding and pacing read.
    pub fn window_share(&self, class: &str, now: Time) -> Option<f64> {
        let (n, met) = self.window_counts(class, now)?;
        (n >= MIN_WINDOW_COMPLETIONS).then(|| met as f64 / n as f64)
    }

    /// f49l: (completions, met) in the class's trailing window, as [`Self::window_share`] counts
    /// them, with no minimum; `None` for a class without a service level.
    pub fn window_counts(&self, class: &str, now: Time) -> Option<(usize, usize)> {
        let level = self.levels.get(class)?;
        let state = self.classes.get(class)?;
        let horizon = Duration::from_secs_f64(level.window_s);
        let (mut n, mut met) = (0usize, 0usize);
        for c in state.window.iter().filter(|c| now.since(c.at) < horizon) {
            n += 1;
            if c.ttft_s <= level.ttft_s && c.itl_s.is_none_or(|itl| itl <= level.itl_s) {
                met += 1;
            }
        }
        Some((n, met))
    }

    /// The `q` quantile of the class's TTFT over its completions in the last `window_s` seconds
    /// (f47's TTFT cap); `None` under [`MIN_WINDOW_COMPLETIONS`] completions.
    pub fn ttft_quantile(&self, class: &str, now: Time, window_s: f64, q: f64) -> Option<f64> {
        let state = self.classes.get(class)?;
        let horizon = Duration::from_secs_f64(window_s);
        let mut v: Vec<f64> = state
            .window
            .iter()
            .filter(|c| now.since(c.at) < horizon)
            .map(|c| c.ttft_s)
            .collect();
        if v.len() < MIN_WINDOW_COMPLETIONS {
            return None;
        }
        percentile(&mut v, q)
    }

    /// Share of the class's completions in the trailing window whose mean ITL was over its limit
    /// (of those that report one); `None` under [`MIN_WINDOW_COMPLETIONS`] such completions.
    pub fn itl_miss_share(&self, class: &str, now: Time) -> Option<f64> {
        let level = self.levels.get(class)?;
        let state = self.classes.get(class)?;
        let horizon = Duration::from_secs_f64(level.window_s);
        let (mut n, mut miss) = (0usize, 0usize);
        for c in state.window.iter().filter(|c| now.since(c.at) < horizon) {
            if let Some(itl) = c.itl_s {
                n += 1;
                if itl > level.itl_s {
                    miss += 1;
                }
            }
        }
        (n >= MIN_WINDOW_COMPLETIONS).then(|| miss as f64 / n as f64)
    }

    pub fn snapshot(&self, class: &str) -> Option<ClassSnapshot> {
        let state = self.classes.get(class)?;
        let level = self.levels.get(class);
        Some(ClassSnapshot {
            class: class.to_owned(),
            per_request: (level.is_some() && state.requests > 0)
                .then(|| state.requests_met as f64 / state.requests as f64),
            requests: state.requests,
            windowed: (state.judged > 0).then(|| state.passed as f64 / state.judged as f64),
            windows_judged: state.judged,
            target: level.map(|l| l.attainment),
            last_window: state.last,
            hourly: state
                .hourly
                .iter()
                .map(|(h, (n, p))| (*h, *n, *p))
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(at: f64, ttft_s: f64, itl_s: Option<f64>) -> Completion {
        Completion {
            at: Time::from_secs_f64(at),
            ttft_s,
            itl_s,
        }
    }

    #[test]
    fn percentile_matches_the_simulator() {
        // n = 10, p = 0.9: index floor(9.0) = 9, the largest.
        let mut v: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_eq!(percentile(&mut v, 0.9), Some(10.0));
        // n = 5: index floor(4.5) = 4.
        let mut v = vec![5.0, 1.0, 4.0, 2.0, 3.0];
        assert_eq!(percentile(&mut v, 0.9), Some(5.0));
        // n = 20: index 18, the second largest.
        let mut v: Vec<f64> = (1..=20).map(f64::from).collect();
        assert_eq!(percentile(&mut v, 0.9), Some(19.0));
        assert_eq!(percentile(&mut [], 0.9), None);
    }

    #[test]
    fn per_request_and_windowed_differ_as_documented() {
        let mut m = Metrics::new(&ServicePolicy::default());
        // 19 fast priority requests and one slow one in the same minute.
        for i in 0..19 {
            m.record("priority", c(10.0 + f64::from(i), 1.0, Some(0.02)));
        }
        m.record("priority", c(40.0, 9.0, Some(0.02)));
        m.judge(Time::from_secs_f64(41.0));
        let s = m.snapshot("priority").unwrap();
        assert_eq!(s.per_request, Some(19.0 / 20.0));
        // p90 over 20 values is the second largest, 1.0 s: the window passes.
        assert_eq!(s.windowed, Some(1.0));
        let w = s.last_window.unwrap();
        assert_eq!((w.ttft_p, w.completions), (1.0, 20));
        // Three slow ones push the p90 over the limit.
        for i in 0..3 {
            m.record("priority", c(45.0 + f64::from(i), 9.0, Some(0.02)));
        }
        m.judge(Time::from_secs_f64(48.0));
        let s = m.snapshot("priority").unwrap();
        assert_eq!((s.windows_judged, s.windowed), (2, Some(0.5)));
    }

    #[test]
    fn small_windows_are_not_judged_and_old_completions_leave() {
        let mut m = Metrics::new(&ServicePolicy::default());
        for i in 0..4 {
            m.record("standard", c(f64::from(i), 1.0, None));
        }
        m.judge(Time::from_secs_f64(5.0));
        assert_eq!(m.snapshot("standard").unwrap().windows_judged, 0);
        m.record("standard", c(5.0, 1.0, None));
        m.judge(Time::from_secs_f64(5.0));
        assert_eq!(m.snapshot("standard").unwrap().windows_judged, 1);
        // 70 s later the window (now - 60 s, now] is empty again.
        m.judge(Time::from_secs_f64(75.0));
        let s = m.snapshot("standard").unwrap();
        assert_eq!((s.windows_judged, s.last_window), (1, None));
    }

    #[test]
    fn itl_over_the_limit_fails_a_window_and_batch_has_no_level() {
        let mut m = Metrics::new(&ServicePolicy::default());
        for i in 0..10 {
            m.record("standard", c(f64::from(i), 1.0, Some(0.2)));
            m.record("batch", c(f64::from(i), 100.0, None));
        }
        m.judge(Time::from_secs_f64(10.0));
        assert_eq!(m.snapshot("standard").unwrap().windowed, Some(0.0));
        let b = m.snapshot("batch").unwrap();
        assert_eq!((b.per_request, b.windowed, b.requests), (None, None, 10));
    }

    #[test]
    fn ttft_quantile_over_the_window() {
        let policy = crate::config::ServicePolicy::default();
        let mut m = Metrics::new(&policy);
        let t = |s: f64| Time::ZERO.after(Duration::from_secs_f64(s));
        for (i, ttft) in [1.0, 2.0, 3.0, 4.0, 10.0].iter().enumerate() {
            m.record(
                "priority",
                Completion {
                    at: t(100.0 + i as f64),
                    ttft_s: *ttft,
                    itl_s: None,
                },
            );
        }
        assert_eq!(m.ttft_quantile("priority", t(110.0), 60.0, 0.5), Some(3.0));
        // Too few in a short window.
        assert_eq!(m.ttft_quantile("priority", t(110.0), 8.0, 0.5), None);
    }
}
