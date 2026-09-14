#!/usr/bin/env python3
"""Adds a new metric-name constant for the G2/CPU-offload total block capacity
(2026-09-14) to vLLM's generic CPU offloading manager (vllm/v1/kv_offload/cpu/common.py -
part of the stock vLLM base image, NOT the KVCR PR overlay, since our deployment runs
vLLM's own OffloadingConnector directly, no KVCR secondary tier).

Companion patches: g2_capacity_spec_patch.py (registers the Prometheus gauge definition)
and g2_capacity_manager_patch.py (sets its value from CPUOffloadingManager's own already-
tracked `_num_blocks`, exact - the same value vLLM's own cpu_cache_usage_perc denominator
uses, computed from cpu_bytes_to_use and vLLM's real per-block byte size - not the
externally-derived Qwen3.5-2B-architecture estimate the dashboard used before this).
"""

import sys

path = sys.argv[1]

with open(path) as f:
    src = f.read()

anchor = (
    '    CPU_CACHE_READ_USAGE_PERC = "vllm:kv_offload_cpu_cache_read_usage_perc"\n'
)
assert anchor in src, f"anchor not found in {path}"
src = src.replace(
    anchor,
    anchor
    + '    CPU_NUM_BLOCKS = "vllm:kv_offload_cpu_num_blocks"\n',
    1,
)

with open(path, "w") as f:
    f.write(src)

print(f"G2 capacity metric-name patch applied to {path}")
