// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Time for the scheduler.
//!
//! Every time read in this crate goes through a [`Clock`]. This is a hard rule: the plugin must
//! run inside an offline replay (dynosim) at simulated speed, and deterministically in tests. The
//! host clock reads `tokio::time::Instant`, which follows the runtime's clock, so it is virtual
//! when the runtime's time is paused (tokio's test utilities, and a replay that drives tokio
//! time). `tests/no_system_clock.rs` fails if any other module reads the system clock.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::time::Instant;

/// A point in scheduler time: how long after the clock's origin.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Time(Duration);

impl Time {
    pub const ZERO: Time = Time(Duration::ZERO);

    pub fn from_secs_f64(seconds: f64) -> Self {
        Time(Duration::from_secs_f64(seconds.max(0.0)))
    }

    pub fn as_secs_f64(self) -> f64 {
        self.0.as_secs_f64()
    }

    pub fn since_origin(self) -> Duration {
        self.0
    }

    /// `self - earlier`, or zero when `earlier` is later.
    pub fn since(self, earlier: Time) -> Duration {
        self.0.saturating_sub(earlier.0)
    }

    pub fn after(self, delta: Duration) -> Time {
        Time(self.0.saturating_add(delta))
    }
}

/// The scheduler's source of time.
pub trait Clock: Send + Sync + 'static {
    fn now(&self) -> Time;

    /// The host instant for a scheduler time, used for deadlines handed to the router queue
    /// (`Hold { until }`, `due_at`). `None` when the clock has no host mapping (the manual clock).
    fn host_instant(&self, at: Time) -> Option<Instant>;

    /// The scheduler time of a host instant (a queue head's `queued_at`, a request's
    /// `ingress_at`). `None` when the clock has no host mapping.
    fn time_of_host(&self, instant: Instant) -> Option<Time>;
}

/// The clock used inside the router: tokio's instant from a fixed origin.
#[derive(Debug)]
pub struct HostClock {
    origin: Instant,
}

impl HostClock {
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Default for HostClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for HostClock {
    fn now(&self) -> Time {
        Time(Instant::now().saturating_duration_since(self.origin))
    }

    fn host_instant(&self, at: Time) -> Option<Instant> {
        Some(self.origin + at.0)
    }

    fn time_of_host(&self, instant: Instant) -> Option<Time> {
        Some(Time(instant.saturating_duration_since(self.origin)))
    }
}

/// The clock of the classifier context the router hands the plugin
/// (`RequestClassifierContext::now`). Live routers give tokio's instant; offline replay gives its
/// virtual clock, so the same scheduler runs at simulated speed there.
pub struct ContextClock {
    now: Box<dyn Fn() -> Instant + Send + Sync>,
    origin: Instant,
}

impl ContextClock {
    pub fn new(now: impl Fn() -> Instant + Send + Sync + 'static) -> Self {
        let origin = now();
        Self {
            now: Box::new(now),
            origin,
        }
    }
}

impl std::fmt::Debug for ContextClock {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ContextClock")
            .field("origin", &self.origin)
            .finish_non_exhaustive()
    }
}

impl Clock for ContextClock {
    fn now(&self) -> Time {
        Time((self.now)().saturating_duration_since(self.origin))
    }

    fn host_instant(&self, at: Time) -> Option<Instant> {
        Some(self.origin + at.0)
    }

    fn time_of_host(&self, instant: Instant) -> Option<Time> {
        Some(Time(instant.saturating_duration_since(self.origin)))
    }
}

/// A clock that runs `scale` times as fast as another: scheduler time is the inner clock's time
/// times `scale`, and a scheduler deadline maps back to the inner clock's instant. A live router
/// in front of engines that run `scale` times faster than real time (the mocker's
/// `speedup_ratio`) uses it, so the scheduler's waits, windows and Retry-After advice are in the
/// engines' time, as in offline replay. See [`clock_scale_from_env`].
pub struct ScaledClock {
    inner: Box<dyn Clock>,
    scale: f64,
}

impl ScaledClock {
    /// `scale` must be finite and positive.
    pub fn new(inner: Box<dyn Clock>, scale: f64) -> Self {
        assert!(
            scale.is_finite() && scale > 0.0,
            "clock scale must be finite and positive"
        );
        Self { inner, scale }
    }

    fn up(&self, time: Time) -> Time {
        Time(time.0.mul_f64(self.scale))
    }

    fn down(&self, time: Time) -> Time {
        Time(time.0.div_f64(self.scale))
    }
}

impl Clock for ScaledClock {
    fn now(&self) -> Time {
        self.up(self.inner.now())
    }

    fn host_instant(&self, at: Time) -> Option<Instant> {
        self.inner.host_instant(self.down(at))
    }

    fn time_of_host(&self, instant: Instant) -> Option<Time> {
        self.inner.time_of_host(instant).map(|time| self.up(time))
    }
}

/// `DYNAMO_SESSION_SCHEDULER_CLOCK_SCALE`: how many times faster than the router's clock the
/// scheduler's time runs. Unset, empty or 1 (the default): the router's clock as it is. Only for
/// a live router in front of sped-up mock engines; offline replay already runs on virtual time.
pub const CLOCK_SCALE_ENV: &str = "DYNAMO_SESSION_SCHEDULER_CLOCK_SCALE";

/// The clock scale [`CLOCK_SCALE_ENV`] sets, or `None` for the router's clock as it is. A value
/// that is not a finite positive number is an error.
pub fn clock_scale_from_env() -> Result<Option<f64>, String> {
    match std::env::var(CLOCK_SCALE_ENV) {
        Ok(text) if !text.trim().is_empty() => parse_clock_scale(&text),
        _ => Ok(None),
    }
}

fn parse_clock_scale(text: &str) -> Result<Option<f64>, String> {
    let scale: f64 = text
        .trim()
        .parse()
        .map_err(|_| format!("{CLOCK_SCALE_ENV}={text:?} is not a number"))?;
    if !(scale.is_finite() && scale > 0.0) {
        return Err(format!(
            "{CLOCK_SCALE_ENV}={text:?} must be finite and positive"
        ));
    }
    Ok((scale != 1.0).then_some(scale))
}

/// A clock moved by hand, for tests and for hosts that drive time themselves.
#[derive(Debug, Default)]
pub struct ManualClock {
    nanos: AtomicU64,
}

impl ManualClock {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&self, at: Time) {
        let nanos = u64::try_from(at.0.as_nanos()).unwrap_or(u64::MAX);
        self.nanos.store(nanos, Ordering::Relaxed);
    }

    pub fn advance(&self, by: Duration) {
        let nanos = u64::try_from(by.as_nanos()).unwrap_or(u64::MAX);
        self.nanos.fetch_add(nanos, Ordering::Relaxed);
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Time {
        Time(Duration::from_nanos(self.nanos.load(Ordering::Relaxed)))
    }

    fn host_instant(&self, _at: Time) -> Option<Instant> {
        None
    }

    fn time_of_host(&self, _instant: Instant) -> Option<Time> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_clock_moves_only_when_told() {
        let clock = ManualClock::new();
        assert_eq!(clock.now(), Time::ZERO);
        clock.advance(Duration::from_secs(5));
        assert_eq!(clock.now().as_secs_f64(), 5.0);
        clock.set(Time::from_secs_f64(2.5));
        assert_eq!(clock.now().as_secs_f64(), 2.5);
    }

    #[test]
    fn scaled_clock_runs_faster_and_maps_deadlines_back() {
        let base = Instant::now();
        let offset = std::sync::Arc::new(AtomicU64::new(0));
        let reader = std::sync::Arc::clone(&offset);
        let inner =
            ContextClock::new(move || base + Duration::from_millis(reader.load(Ordering::Relaxed)));
        let clock = ScaledClock::new(Box::new(inner), 10.0);
        assert_eq!(clock.now(), Time::ZERO);
        offset.store(3_000, Ordering::Relaxed);
        assert_eq!(clock.now().as_secs_f64(), 30.0);
        // A deadline 25 s of scheduler time ahead is 2.5 s of host time ahead.
        let due = clock.now().after(Duration::from_secs(25));
        assert_eq!(
            clock.host_instant(due).unwrap(),
            base + Duration::from_millis(5_500)
        );
        assert_eq!(
            clock.time_of_host(base + Duration::from_millis(5_500)),
            Some(due)
        );
    }

    #[test]
    fn clock_scale_parses_only_finite_positive_numbers() {
        assert_eq!(parse_clock_scale("1"), Ok(None));
        assert_eq!(parse_clock_scale(" 4 "), Ok(Some(4.0)));
        assert_eq!(parse_clock_scale("0.5"), Ok(Some(0.5)));
        for bad in ["0", "-2", "nan", "inf", "fast"] {
            assert!(parse_clock_scale(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn context_clock_follows_the_hosts_clock() {
        let base = Instant::now();
        let offset = std::sync::Arc::new(AtomicU64::new(0));
        let reader = std::sync::Arc::clone(&offset);
        let clock =
            ContextClock::new(move || base + Duration::from_millis(reader.load(Ordering::Relaxed)));
        assert_eq!(clock.now(), Time::ZERO);
        offset.store(90_000, Ordering::Relaxed);
        assert_eq!(clock.now().as_secs_f64(), 90.0);
        let at = clock.now();
        assert_eq!(
            clock.time_of_host(clock.host_instant(at).unwrap()),
            Some(at)
        );
    }

    #[tokio::test(start_paused = true)]
    async fn host_clock_follows_paused_tokio_time() {
        let clock = HostClock::new();
        tokio::time::advance(Duration::from_secs(3600)).await;
        assert_eq!(clock.now().since_origin(), Duration::from_secs(3600));
        let at = clock.now();
        let instant = clock.host_instant(at).unwrap();
        assert_eq!(clock.time_of_host(instant), Some(at));
    }
}
