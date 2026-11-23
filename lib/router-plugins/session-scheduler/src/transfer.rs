// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! f49n: the share transfer step shared by f49l3's `admission.sla_budget.mode: rebalance`
//! ([`crate::sla_budget`]) and the target loop's `admission.target_loop.rebalance`
//! ([`crate::config::LoopRebalanceConfig`]). A class's share moves multiplicatively at a rate per
//! minute scaled by a head (a surplus or a shortfall, saturating at `full_at`), within f49l2's rate
//! caps, and stops at its floor or ceiling; when a share must give way, an order (rank) picks the
//! donor: the first one above its floor.

use crate::sla_budget::RateCaps;

/// `k x f` down to `lo` (unchanged when already at or under it).
pub fn down(k: f64, f: f64, lo: f64) -> f64 {
    if k > lo { (k * f).max(lo) } else { k }
}

/// `k x f` up to `hi` (unchanged when already at or over it).
pub fn up(k: f64, f: f64, hi: f64) -> f64 {
    if k < hi { (k * f).min(hi) } else { k }
}

/// The share of the full rate a head `h` drives: `|h| / full_at`, at most 1.
pub fn share(h: f64, full_at: f64) -> f64 {
    (h.abs() / full_at).min(1.0)
}

/// A step's multiplier over `dt` seconds at `per_min` a minute (negative: a shrink) times `share`,
/// `exp(per_min / 60 x share x dt)`, within the rate caps.
pub fn factor(caps: RateCaps, per_min: f64, share: f64, dt: f64) -> f64 {
    caps.cap((per_min / 60.0 * share * dt).exp(), dt)
}

/// Rank-ordered give-way: the first class of `order` (first to give first) whose share `k` is
/// above its floor `lo`.
pub fn give_way<'a>(
    order: &[&'a str],
    k: impl Fn(&str) -> f64,
    lo: impl Fn(&str) -> f64,
) -> Option<&'a str> {
    order.iter().copied().find(|&c| k(c) > lo(c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f49n_moves_clamp_scale_cap_and_give_way_in_order() {
        assert_eq!(down(1.0, 0.5, 0.6), 0.6);
        assert_eq!(down(0.5, 0.5, 0.6), 0.5);
        assert_eq!(up(1.0, 2.0, 1.5), 1.5);
        assert_eq!(up(2.0, 2.0, 1.5), 2.0);
        assert!((share(0.01, 0.05) - 0.2).abs() < 1e-12);
        assert_eq!(share(-0.2, 0.05), 1.0);
        let none = RateCaps::default();
        assert!((factor(none, 0.06, 1.0, 60.0) - 0.06_f64.exp()).abs() < 1e-12);
        assert!((factor(none, -0.06, 0.5, 60.0) - (-0.03_f64).exp()).abs() < 1e-12);
        let caps = RateCaps {
            max_shrink_per_min: Some(0.005),
            max_grow_per_min: Some(0.004),
        };
        assert!((factor(caps, 1.0, 1.0, 60.0) - 0.004_f64.exp()).abs() < 1e-12);
        assert!((factor(caps, -1.0, 1.0, 60.0) - (-0.005_f64).exp()).abs() < 1e-12);
        let k = |c: &str| if c == "batch" { 0.333 } else { 1.0 };
        let lo = |c: &str| if c == "batch" { 0.333 } else { 0.5 };
        assert_eq!(give_way(&["batch", "standard"], k, lo), Some("standard"));
        assert_eq!(give_way(&["batch"], k, lo), None);
        assert_eq!(give_way(&["standard", "batch"], k, lo), Some("standard"));
    }
}
