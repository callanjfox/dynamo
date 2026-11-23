// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Uncontended time (F1, service class family (b)): how long a request would take alone on a
//! worker, from the step model. The denominator of run slowdown (hand-back time over the run's
//! uncontended time) and the unit of attained service for `turn_order: run_las`.
//!
//! Per request: prefill of its uncached tokens after its cached prefix, in chunks of the worker's
//! batch token budget (`max_num_batched_tokens`), plus decode of its output at batch 1 at the mean
//! context. The cached prefix at no contention is the larger of what a worker held when it was
//! classified and the session's previous context (agentic contexts are append-only, so an idle
//! worker keeps the session's history). This mirrors `scripts/validate/family_b.py`, so the
//! plugin's run log and the offline scorer agree.

use crate::forecast::StepModel;

/// Seconds `prompt` input tokens (of which `cached` are a cached prefix) and `output` output
/// tokens take alone, prefill chunked at `chunk` tokens. `None` when the model has no answer.
pub fn request_s(
    model: &dyn StepModel,
    chunk: u64,
    prompt: u64,
    cached: u64,
    output: u64,
) -> Option<f64> {
    let cached = cached.min(prompt);
    let mut t = prefill_s(model, chunk, prompt - cached, cached)?;
    if output > 1 {
        let context = prompt.saturating_add(output / 2);
        let step = model.decode_s(1, u32::try_from(context).unwrap_or(u32::MAX))?;
        t += (output - 1) as f64 * step;
    }
    Some(t)
}

/// Prefill of `new` tokens after `prefix` cached ones, alone, in chunks of `chunk` tokens.
pub fn prefill_s(model: &dyn StepModel, chunk: u64, new: u64, prefix: u64) -> Option<f64> {
    let tokens = |t: u64| u32::try_from(t).unwrap_or(u32::MAX);
    let chunk = chunk.max(1);
    if new == 0 {
        return model.prefill_s(1, 0, tokens(prefix));
    }
    let mut t = 0.0;
    let mut done = 0u64;
    while done < new {
        let n = chunk.min(new - done);
        t += model.prefill_s(1, tokens(n), tokens(prefix + done))?;
        done += n;
    }
    Some(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecast::TableStepModel;

    fn table() -> TableStepModel {
        TableStepModel {
            prefill_tokens: vec![0.0, 1000.0, 2000.0],
            prefill_s: vec![0.01, 0.11, 0.21],
            prefix_weight: 0.0,
            decode_batch: vec![1.0, 2.0],
            decode_context: vec![1000.0, 3000.0],
            decode_s: vec![0.01, 0.02, 0.02, 0.04],
        }
    }

    #[test]
    fn prefill_is_chunked_and_decode_is_at_batch_one_and_mean_context() {
        let m = table();
        // 2,500 new tokens in chunks of 1,000: 0.11 + 0.11 + 0.06.
        let p = prefill_s(&m, 1000, 2500, 0).unwrap();
        assert!((p - 0.28).abs() < 1e-9, "{p}");
        // A cached prefix costs nothing at prefix weight 0; nothing new is one empty step.
        assert!((prefill_s(&m, 1000, 0, 5000).unwrap() - 0.01).abs() < 1e-9);
        // 1,000 prompt tokens, none cached: 0.11 prefill; 11 output tokens at context 1,005
        // (batch 1): 10 decode steps.
        let r = request_s(&m, 4096, 1000, 0, 11).unwrap();
        let decode = 0.01 + 0.01 * 5.0 / 2000.0;
        assert!((r - (0.11 + 10.0 * decode)).abs() < 1e-9, "{r}");
        // A cached prefix larger than the prompt is the whole prompt.
        let all = request_s(&m, 4096, 1000, 5000, 1).unwrap();
        assert!((all - 0.01).abs() < 1e-9);
    }
}
