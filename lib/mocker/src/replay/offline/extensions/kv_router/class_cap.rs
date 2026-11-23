// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! f49h `policy_classes[].active_requests_per_worker`: a static cap on each class's active requests
//! per worker rank (admitted to the rank and not finished). A rank holding the class's cap or more
//! of its requests is busy for that class: the class's busy test counts it full beside the busy
//! thresholds, so a request of the class queues at the router under the class's queue limits while
//! every rank is busy, and selection leaves out a worker whose ranks are all at the cap while
//! another has room. Other classes see no change. With no class setting a cap (the default),
//! nothing is tracked and the busy test is the one before.

use std::collections::{HashMap, HashSet};

use rustc_hash::FxHashMap;
use uuid::Uuid;

use dynamo_admission::host::{PolicyClassConfig, PolicyProfile};
use dynamo_kv_router::plugins::worker_selection::WorkerCapacity;
use dynamo_kv_router::scheduling::BusyThresholdsExt;
use dynamo_router_protocol::worker::{WorkerId, WorkerWithDpRank};
use dynamo_scheduler::host::{WorkerCapacityExt, WorkerConfigLike};
use dynamo_slot_tracker::host::{BusyRanksExt, WorkerLoadProjection};

/// Requests admitted per rank and class (class index), while any class sets a cap.
#[derive(Debug, Default)]
pub(crate) struct ClassActive {
    by_request: FxHashMap<Uuid, (WorkerWithDpRank, usize)>,
    counts: FxHashMap<(WorkerWithDpRank, usize), usize>,
}

impl ClassActive {
    /// Tracking for `profile`, or `None` when no class sets a cap.
    pub(crate) fn for_profile(profile: &PolicyProfile) -> Option<Self> {
        profile
            .classes()
            .iter()
            .any(|class| class.active_requests_per_worker.is_some())
            .then(Self::default)
    }

    /// `uuid` was admitted to `worker` as a request of class `class_index`.
    pub(crate) fn book(&mut self, uuid: Uuid, worker: WorkerWithDpRank, class_index: usize) {
        self.release(uuid);
        self.by_request.insert(uuid, (worker, class_index));
        *self.counts.entry((worker, class_index)).or_default() += 1;
    }

    /// `uuid` finished or left its worker; a second release does nothing.
    pub(crate) fn release(&mut self, uuid: Uuid) {
        if let Some(key) = self.by_request.remove(&uuid)
            && let Some(count) = self.counts.get_mut(&key)
        {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.counts.remove(&key);
            }
        }
    }

    /// Active requests of class `class_index` on `worker`.
    pub(crate) fn count(&self, worker: WorkerWithDpRank, class_index: usize) -> usize {
        self.counts
            .get(&(worker, class_index))
            .copied()
            .unwrap_or(0)
    }

    /// The cap of `class` (index `class_index`), when it sets one.
    pub(crate) fn cap<'a>(
        &'a self,
        class_index: usize,
        class: &PolicyClassConfig,
    ) -> Option<ClassCap<'a>> {
        class.active_requests_per_worker.map(|cap| ClassCap {
            active: self,
            class_index,
            cap,
        })
    }
}

/// One class's cap over the active counts.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ClassCap<'a> {
    active: &'a ClassActive,
    class_index: usize,
    cap: usize,
}

impl ClassCap<'_> {
    /// Whether `worker` holds the cap or more of the class's requests.
    pub(crate) fn rank_at_cap(&self, worker: WorkerWithDpRank) -> bool {
        self.active.count(worker, self.class_index) >= self.cap
    }
}

fn ranks<C: WorkerConfigLike>(config: &C) -> std::ops::Range<u32> {
    let start = config.data_parallel_start_rank();
    start..start.saturating_add(config.data_parallel_size())
}

/// The class's busy test with its cap: the `target` rank (or worker, with no rank) when the
/// request has one, else every worker, is busy past the busy thresholds or at the cap. Without a
/// cap, the busy thresholds alone, as before.
pub(crate) fn workers_busy<C: WorkerConfigLike>(
    loads: &FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
    workers: &HashMap<WorkerId, C>,
    class: &PolicyClassConfig,
    target: Option<(WorkerId, Option<u32>)>,
    cap: Option<ClassCap<'_>>,
) -> bool {
    let limits = class.busy_thresholds.for_class(&class.name);
    let rank_full = |worker: WorkerWithDpRank, config: &C| {
        limits.rank_is_full(worker, WorkerCapacity::of(config), loads)
            || cap.is_some_and(|cap| cap.rank_at_cap(worker))
    };
    let worker_full = |worker_id: WorkerId, config: &C| match cap {
        None => limits.worker_is_full(worker_id, config, loads),
        Some(_) => {
            ranks(config).all(|rank| rank_full(WorkerWithDpRank::new(worker_id, rank), config))
        }
    };
    match target {
        Some((worker_id, rank)) => workers.get(&worker_id).is_none_or(|config| match rank {
            Some(rank) => rank_full(WorkerWithDpRank::new(worker_id, rank), config),
            None => worker_full(worker_id, config),
        }),
        None => workers
            .iter()
            .all(|(&worker_id, config)| worker_full(worker_id, config)),
    }
}

/// The workers selection leaves out for a request of the capped class: `already` (those left out
/// for another reason) plus every worker whose ranks are all at the cap, while at least one other
/// worker has a rank under it. `None` when the cap leaves nobody out.
pub(crate) fn excluded<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    cap: ClassCap<'_>,
    already: Option<&HashSet<WorkerId>>,
) -> Option<HashSet<WorkerId>> {
    let mut out = HashSet::new();
    let mut open = 0usize;
    for (&worker_id, config) in workers {
        if already.is_some_and(|set| set.contains(&worker_id)) {
            continue;
        }
        if ranks(config).all(|rank| cap.rank_at_cap(WorkerWithDpRank::new(worker_id, rank))) {
            out.insert(worker_id);
        } else {
            open += 1;
        }
    }
    if open == 0 || out.is_empty() {
        return None;
    }
    if let Some(set) = already {
        out.extend(set.iter().copied());
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::super::{OfflineReplayRouter, RouterEffects};
    use super::*;
    use crate::common::protocols::{DirectRequest, MockerConfig};
    use dynamo_kv_router::config::KvRouterConfig;
    use dynamo_slot_tracker::busy::BusyThresholds;

    #[derive(Clone)]
    struct OneRank;

    impl WorkerConfigLike for OneRank {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }

        fn data_parallel_size(&self) -> u32 {
            1
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            Some(1000)
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            None
        }
    }

    fn class(name: &str, cap: Option<usize>, frac: Option<f64>) -> PolicyClassConfig {
        PolicyClassConfig {
            name: name.to_string(),
            queue_policy: dynamo_admission::host::RouterQueuePolicy::Edf,
            quantum: 1,
            slo_ttft_ms: None,
            controlled_delay: false,
            busy_thresholds: BusyThresholds {
                prefill_busy_threshold_frac: frac,
                ..Default::default()
            },
            request_queue_limit_per_worker: None,
            raw_isl_token_queue_limit_per_worker: None,
            cached_token_queue_limit_per_worker: None,
            active_requests_per_worker: cap,
        }
    }

    #[test]
    fn the_cap_makes_a_worker_busy_for_its_class_only() {
        let (w0, w1) = (WorkerWithDpRank::new(0, 0), WorkerWithDpRank::new(1, 0));
        let workers: HashMap<WorkerId, OneRank> = [(0, OneRank), (1, OneRank)].into();
        let loads = FxHashMap::default();
        let priority = class("priority", Some(2), Some(1.0));
        let standard = class("standard", Some(4), Some(1.0));
        let mut active = ClassActive::default();
        // Priority (class 0): two on w0, one on w1. Standard (class 1): three on w0.
        for (i, (worker, class_index)) in [(w0, 0), (w0, 0), (w1, 0), (w0, 1), (w0, 1), (w0, 1)]
            .into_iter()
            .enumerate()
        {
            active.book(Uuid::from_u128(i as u128), worker, class_index);
        }
        let prio_cap = active.cap(0, &priority);
        let std_cap = active.cap(1, &standard);
        // w0 is busy for priority (2 >= 2) and not for standard (3 < 4); w1 has room for both.
        assert!(workers_busy(
            &loads,
            &workers,
            &priority,
            Some((0, Some(0))),
            prio_cap
        ));
        assert!(!workers_busy(
            &loads,
            &workers,
            &standard,
            Some((0, Some(0))),
            std_cap
        ));
        assert!(!workers_busy(&loads, &workers, &priority, None, prio_cap));
        assert_eq!(
            excluded(&workers, prio_cap.unwrap(), None),
            Some(HashSet::from([0]))
        );
        assert_eq!(excluded(&workers, std_cap.unwrap(), None), None);
        // The second priority request on w1 makes every worker busy for priority, and the cap
        // then leaves nobody out of selection (no worker has room).
        active.book(Uuid::from_u128(10), w1, 0);
        let prio_cap = active.cap(0, &priority);
        assert!(workers_busy(&loads, &workers, &priority, None, prio_cap));
        assert_eq!(excluded(&workers, prio_cap.unwrap(), None), None);
        assert!(!workers_busy(
            &loads,
            &workers,
            &standard,
            None,
            active.cap(1, &standard)
        ));
        // A finished request frees its slot, once.
        active.release(Uuid::from_u128(10));
        active.release(Uuid::from_u128(10));
        assert_eq!(active.count(w1, 0), 1);
        assert!(!workers_busy(
            &loads,
            &workers,
            &priority,
            None,
            active.cap(0, &priority)
        ));
    }

    #[test]
    fn without_a_cap_the_busy_test_is_unchanged() {
        let workers: HashMap<WorkerId, OneRank> = [(0, OneRank), (1, OneRank)].into();
        let busy = WorkerLoadProjection {
            active_prefill_tokens: 2000,
            ..Default::default()
        };
        let mut loads = FxHashMap::default();
        loads.insert(WorkerWithDpRank::new(0, 0), busy);
        let mut active = ClassActive::default();
        active.book(Uuid::from_u128(1), WorkerWithDpRank::new(1, 0), 0);
        let uncapped = class("priority", None, Some(1.0));
        assert!(active.cap(0, &uncapped).is_none());
        for target in [None, Some((0, Some(0))), Some((0, None)), Some((1, None))] {
            let before = match target {
                Some((id, Some(rank))) => uncapped.busy_thresholds.rank_is_full(
                    WorkerWithDpRank::new(id, rank),
                    WorkerCapacity::of(&workers[&id]),
                    &loads,
                ),
                Some((id, None)) => {
                    uncapped
                        .busy_thresholds
                        .worker_is_full(id, &workers[&id], &loads)
                }
                None => workers
                    .iter()
                    .all(|(&id, c)| uncapped.busy_thresholds.worker_is_full(id, c, &loads)),
            };
            assert_eq!(
                workers_busy(&loads, &workers, &uncapped, target, None),
                before,
                "{target:?}"
            );
        }
        // No class with a cap: nothing is tracked.
        let profile = PolicyProfile::from_classes(vec![uncapped.clone()], None).unwrap();
        assert!(ClassActive::for_profile(&profile).is_none());
        let capped = PolicyProfile::from_classes(
            vec![uncapped, class("batch", Some(2), None)],
            Some("priority"),
        )
        .unwrap();
        assert!(ClassActive::for_profile(&capped).is_some());
    }

    fn turn(uuid: u128, class: &str) -> DirectRequest {
        DirectRequest {
            tokens: vec![uuid as u32; 64],
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(uuid)),
            dp_rank: 0,
            preferred_dp_rank: None,
            preferred_prefill_dp_rank: None,
            kv_hints: None,
            arrival_timestamp_ms: Some(0.0),
            priority: 0,
            strict_priority: 0,
            policy_class: Some(class.to_string()),
            replay_context: None,
            images: Vec::new(),
        }
    }

    fn router(yaml: &str) -> OfflineReplayRouter {
        let path = std::env::temp_dir().join(format!("f49h-cap-{}.yaml", Uuid::new_v4()));
        std::fs::write(&path, yaml).unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(path.display().to_string()),
            ..KvRouterConfig::default()
        };
        let args = MockerConfig::from_value(serde_json::json!({
            "engine": {"block_size": 64, "max_num_batched_tokens": 4096}
        }))
        .unwrap();
        let router = OfflineReplayRouter::new(&args, Some(config), None, 1).unwrap();
        std::fs::remove_file(path).unwrap();
        router
    }

    const CLASSES: &str = "default_policy_class: standard
policy_classes:
  - name: priority
    quantum: 8
    prefill_busy_threshold_frac: 4.0
CAP
  - name: standard
    quantum: 4
    prefill_busy_threshold_frac: 4.0
";

    #[test]
    fn a_capped_class_queues_at_its_cap_and_other_classes_do_not() {
        let mut router = router(&CLASSES.replace("CAP", "    active_requests_per_worker: 1"));
        let admitted = |effects: RouterEffects| effects.admissions.len();
        assert_eq!(
            admitted(
                router
                    .on_request_arrival(&turn(1, "priority"), None, 0.0)
                    .unwrap()
            ),
            1
        );
        // The only worker holds one priority request: the next priority one queues.
        assert_eq!(
            admitted(
                router
                    .on_request_arrival(&turn(2, "priority"), None, 1.0)
                    .unwrap()
            ),
            0
        );
        // Standard is not capped and goes.
        assert_eq!(
            admitted(
                router
                    .on_request_arrival(&turn(3, "standard"), None, 2.0)
                    .unwrap()
            ),
            1
        );
        // The first priority request finishing lets the queued one go.
        let effects = router
            .on_request_completed(Uuid::from_u128(1), 3.0)
            .unwrap();
        assert_eq!(effects.admissions.len(), 1);
        assert_eq!(effects.admissions[0].uuid, Uuid::from_u128(2));
    }

    #[test]
    fn without_the_cap_the_same_turns_all_go() {
        let mut router = router(&CLASSES.replace("CAP\n", ""));
        for (i, class) in [(1, "priority"), (2, "priority"), (3, "standard")] {
            assert_eq!(
                router
                    .on_request_arrival(&turn(i, class), None, i as f64)
                    .unwrap()
                    .admissions
                    .len(),
                1,
                "request {i}"
            );
        }
    }
}
