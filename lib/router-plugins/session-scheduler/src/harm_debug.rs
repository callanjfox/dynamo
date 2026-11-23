// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! f49d diagnostics, no decision changes: with `DYNAMO_HARM_DEBUG=1` the scheduler counts, per
//! class, what the session gate's forecast-harm check on new sessions (`admission.forecast_harm_new`,
//! ks2b) saw at each gate check with the check configured, and prints a summary line to stderr every
//! 1,000 checks, at most once a simulated minute. Off by default.
//!
//! Summary line: `harm-debug t=<sim s> checks=<n> | <class> checks=.. found=.. lobby=.. exempt=..
//! no_forecast=.. rows_empty=.. harm_none=.. harm_some=.. reached=.. held=.. expired=..
//! hist=0:..,(0,0.001]:..,(0.001,0.01]:..,(0.01,0.1]:..,(0.1,1]:..,>1:.. max=<x or ->` per class.
//! `?` collects checks whose session was not in the table.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::time::Duration;

use crate::admit::Decision;
use crate::clock::Time;
use crate::session::SessionState;

/// Checks between two summary lines (at least).
const EVERY_CHECKS: u64 = 1_000;
/// Simulated seconds between two summary lines (at least).
const EVERY_S: u64 = 60;
/// The histogram's bins of added weighted misses.
pub const BINS: [&str; 6] = [
    "0",
    "(0,0.001]",
    "(0.001,0.01]",
    "(0.01,0.1]",
    "(0.1,1]",
    ">1",
];

/// Whether `DYNAMO_HARM_DEBUG=1` is set.
pub fn enabled_by_env() -> bool {
    std::env::var("DYNAMO_HARM_DEBUG").is_ok_and(|v| v == "1")
}

/// What one gate check fed the harm check (`hooks.rs`, `new_harm`), in the order it is decided.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HarmCheck {
    /// The head's session is not in the table.
    NoSession,
    /// Found, but not waiting in the lobby (a turn of an admitted session): no harm is computed.
    NotLobby,
    /// In the lobby, of an exempt class: no harm is computed.
    Exempt,
    /// In the lobby, but the classifier has no forecast.
    NoForecast,
    /// In the lobby, but the host's worker rows are empty (no load sample taken yet).
    NoRows,
    /// Rows, but the forecast priced none of them (`None`: no worker sample there).
    NoHarm,
    /// The best placement's weighted added misses.
    Harm(f64),
}

/// Classify one check from what the host saw: the head's session (class, state) if found, the
/// exempt classes, whether there is a forecast, whether the worker rows are empty, and the harm.
pub fn classify(
    session: Option<(&str, SessionState)>,
    exempt: &[String],
    has_forecast: bool,
    rows_empty: bool,
    harm: Option<f64>,
) -> HarmCheck {
    let Some((class, state)) = session else {
        return HarmCheck::NoSession;
    };
    if state != SessionState::Lobby {
        return HarmCheck::NotLobby;
    }
    if exempt.iter().any(|c| c == class) {
        return HarmCheck::Exempt;
    }
    if !has_forecast {
        return HarmCheck::NoForecast;
    }
    if rows_empty {
        return HarmCheck::NoRows;
    }
    match harm {
        Some(h) => HarmCheck::Harm(h),
        None => HarmCheck::NoHarm,
    }
}

/// The histogram bin of `x` added weighted misses.
pub fn bin(x: f64) -> usize {
    if x <= 0.0 {
        0
    } else if x <= 0.001 {
        1
    } else if x <= 0.01 {
        2
    } else if x <= 0.1 {
        3
    } else if x <= 1.0 {
        4
    } else {
        5
    }
}

/// One class's counters.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClassCounts {
    pub checks: u64,
    pub found: u64,
    pub lobby: u64,
    pub exempt: u64,
    pub no_forecast: u64,
    pub rows_empty: u64,
    pub harm_none: u64,
    pub harm_some: u64,
    /// `harm_hold` called with the decision still `Admit(Fits)`.
    pub reached: u64,
    pub held: u64,
    pub expired: u64,
    pub hist: [u64; 6],
    pub max: Option<f64>,
}

/// The counters and the summary's pacing.
#[derive(Clone, Debug, Default)]
pub struct HarmDebug {
    per: BTreeMap<String, ClassCounts>,
    checks: u64,
    since_print: u64,
    last_print: Option<Time>,
}

impl HarmDebug {
    /// Count one gate check of a session of `class` (`?` when not found).
    pub fn check(&mut self, class: &str, check: HarmCheck) {
        self.checks += 1;
        self.since_print += 1;
        let c = self.per.entry(class.to_owned()).or_default();
        c.checks += 1;
        if check != HarmCheck::NoSession {
            c.found += 1;
        }
        if !matches!(check, HarmCheck::NoSession | HarmCheck::NotLobby) {
            c.lobby += 1;
        }
        match check {
            HarmCheck::Exempt => c.exempt += 1,
            HarmCheck::NoForecast => c.no_forecast += 1,
            HarmCheck::NoRows => c.rows_empty += 1,
            HarmCheck::NoHarm => c.harm_none += 1,
            HarmCheck::Harm(h) => {
                c.harm_some += 1;
                c.hist[bin(h)] += 1;
                c.max = Some(c.max.map_or(h, |m| m.max(h)));
            }
            HarmCheck::NoSession | HarmCheck::NotLobby => {}
        }
    }

    /// `harm_hold` was reached for `class` with the decision still `Fits`; `outcome` is its answer.
    pub fn reached(&mut self, class: &str, outcome: Option<&Decision>) {
        let c = self.per.entry(class.to_owned()).or_default();
        c.reached += 1;
        match outcome {
            Some(Decision::Hold { .. }) => c.held += 1,
            Some(Decision::Expire { .. }) => c.expired += 1,
            _ => {}
        }
    }

    pub fn counts(&self, class: &str) -> Option<&ClassCounts> {
        self.per.get(class)
    }

    pub fn checks(&self) -> u64 {
        self.checks
    }

    /// The summary line when one is due at `now` (1,000 checks since the last, and a simulated
    /// minute), which restarts the pacing.
    pub fn due(&mut self, now: Time) -> Option<String> {
        if self.since_print < EVERY_CHECKS {
            return None;
        }
        if let Some(last) = self.last_print
            && now.since(last) < Duration::from_secs(EVERY_S)
        {
            return None;
        }
        self.since_print = 0;
        self.last_print = Some(now);
        Some(self.summary(now))
    }

    /// One line: the totals, then each class's counters.
    pub fn summary(&self, now: Time) -> String {
        let mut out = format!(
            "harm-debug t={:.1} checks={}",
            now.as_secs_f64(),
            self.checks
        );
        for (class, c) in &self.per {
            let hist = BINS
                .iter()
                .zip(c.hist)
                .map(|(b, n)| format!("{b}:{n}"))
                .collect::<Vec<_>>()
                .join(",");
            let max = c.max.map_or_else(|| "-".to_owned(), |m| format!("{m:.6}"));
            let _ = write!(
                out,
                " | {class} checks={} found={} lobby={} exempt={} no_forecast={} rows_empty={} harm_none={} harm_some={} reached={} held={} expired={} hist={hist} max={max}",
                c.checks,
                c.found,
                c.lobby,
                c.exempt,
                c.no_forecast,
                c.rows_empty,
                c.harm_none,
                c.harm_some,
                c.reached,
                c.held,
                c.expired,
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: f64) -> Time {
        Time::from_secs_f64(s)
    }

    #[test]
    fn classify_follows_the_hosts_order() {
        let exempt = vec!["batch".to_owned()];
        let lobby = Some(("standard", SessionState::Lobby));
        assert_eq!(
            classify(None, &exempt, true, false, Some(1.0)),
            HarmCheck::NoSession
        );
        for state in [
            SessionState::Active,
            SessionState::Idle,
            SessionState::Paused,
            SessionState::Ended,
        ] {
            assert_eq!(
                classify(Some(("standard", state)), &exempt, true, false, None),
                HarmCheck::NotLobby
            );
        }
        assert_eq!(
            classify(
                Some(("batch", SessionState::Lobby)),
                &exempt,
                true,
                false,
                None
            ),
            HarmCheck::Exempt
        );
        assert_eq!(
            classify(lobby, &exempt, false, false, None),
            HarmCheck::NoForecast
        );
        assert_eq!(
            classify(lobby, &exempt, true, true, None),
            HarmCheck::NoRows
        );
        assert_eq!(
            classify(lobby, &exempt, true, false, None),
            HarmCheck::NoHarm
        );
        assert_eq!(
            classify(lobby, &exempt, true, false, Some(0.0)),
            HarmCheck::Harm(0.0)
        );
    }

    #[test]
    fn counters_bins_and_pacing() {
        assert_eq!(
            [0.0, 0.0005, 0.001, 0.005, 0.05, 0.5, 1.0, 3.0].map(bin),
            [0, 1, 1, 2, 3, 4, 4, 5]
        );
        let mut d = HarmDebug::default();
        d.check("?", HarmCheck::NoSession);
        d.check("standard", HarmCheck::NotLobby);
        d.check("standard", HarmCheck::NoRows);
        d.check("standard", HarmCheck::NoHarm);
        d.check("standard", HarmCheck::Harm(0.0));
        d.check("standard", HarmCheck::Harm(0.5));
        d.reached("standard", None);
        d.reached(
            "standard",
            Some(&Decision::Hold {
                reason: crate::admit::HoldReason::ForecastHarm {
                    added: 0.5,
                    limit: 0.01,
                },
                until: t(9.0),
            }),
        );
        let c = d.counts("standard").unwrap();
        assert_eq!(
            (
                c.checks,
                c.found,
                c.lobby,
                c.rows_empty,
                c.harm_none,
                c.harm_some
            ),
            (5, 5, 4, 1, 1, 2)
        );
        assert_eq!((c.reached, c.held, c.expired), (2, 1, 0));
        assert_eq!(c.hist, [1, 0, 0, 0, 1, 0]);
        assert_eq!(c.max, Some(0.5));
        assert_eq!(d.counts("?").unwrap().found, 0);
        // Not due before 1,000 checks; then once, and not again within a simulated minute.
        assert_eq!(d.due(t(0.0)), None);
        for _ in 0..1_000 {
            d.check("priority", HarmCheck::NotLobby);
        }
        let line = d.due(t(10.0)).unwrap();
        assert!(line.starts_with("harm-debug t=10.0 checks=1006 | ?"));
        assert!(line.contains(
            " | standard checks=5 found=5 lobby=4 exempt=0 no_forecast=0 rows_empty=1 harm_none=1 harm_some=2 reached=2 held=1 expired=0 hist=0:1,(0,0.001]:0,(0.001,0.01]:0,(0.01,0.1]:0,(0.1,1]:1,>1:0 max=0.500000"
        ));
        for _ in 0..1_000 {
            d.check("priority", HarmCheck::NotLobby);
        }
        assert_eq!(d.due(t(69.0)), None);
        assert!(d.due(t(70.0)).is_some());
    }
}
