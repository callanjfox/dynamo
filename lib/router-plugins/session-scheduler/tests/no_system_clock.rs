// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Only `clock.rs` may read the host clock; everything else takes time from the injected clock,
//! so the scheduler runs the same under a manual clock, paused tokio time, and the simulator.

use std::path::Path;

const FORBIDDEN: &[&str] = &[
    "Instant::now(",
    "SystemTime::now(",
    "Utc::now(",
    "Local::now(",
];

#[test]
fn no_module_reads_the_system_clock_directly() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut offenders = Vec::new();
    for entry in std::fs::read_dir(&src).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "rs") || path.file_name().unwrap() == "clock.rs" {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        // Test modules may build router fixtures (a `DispatchHead` needs an `Instant`); the rule
        // covers the code that runs in the router.
        let code = text.split("#[cfg(test)]").next().unwrap();
        for (number, line) in code.lines().enumerate() {
            if FORBIDDEN.iter().any(|f| line.contains(f)) {
                offenders.push(format!(
                    "{}:{}: {}",
                    path.display(),
                    number + 1,
                    line.trim()
                ));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "read the injected clock instead:\n{}",
        offenders.join("\n")
    );
}
