#!/usr/bin/env python3
"""Sets the new G2/CPU-offload total block capacity gauge (2026-09-14) from
CPUOffloadingManager's own already-tracked `_num_blocks` (vllm/v1/kv_offload/cpu/manager.py) -
the exact same value `cpu_cache_usage_perc` already divides by, just never exported as its
own metric before. See g2_capacity_common_patch.py / g2_capacity_spec_patch.py for the
metric-name constant and its Prometheus gauge definition.
"""

import sys

path = sys.argv[1]

with open(path) as f:
    src = f.read()

anchor = (
    "        usage = num_used / self._num_blocks if self._num_blocks > 0 else 0.0\n"
    "        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC, usage)\n"
)
assert anchor in src, f"anchor not found in {path}"
src = src.replace(
    anchor,
    anchor
    + "        stats.set_gauge(\n"
    "            CPUOffloadingMetrics.CPU_NUM_BLOCKS, self._num_blocks\n"
    "        )\n",
    1,
)

with open(path, "w") as f:
    f.write(src)

print(f"G2 capacity metric-value patch applied to {path}")
