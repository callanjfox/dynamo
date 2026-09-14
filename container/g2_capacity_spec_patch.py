#!/usr/bin/env python3
"""Registers the Prometheus gauge definition for the new G2/CPU-offload total block
capacity metric (2026-09-14) in vLLM's CPUOffloadingSpec.build_metric_definitions()
(vllm/v1/kv_offload/cpu/spec.py). See g2_capacity_common_patch.py for the metric-name
constant and g2_capacity_manager_patch.py for where its value is actually set.
"""

import sys

path = sys.argv[1]

with open(path) as f:
    src = f.read()

anchor = (
    "            CPUOffloadingMetrics.CPU_ALLOCATION_SIZE: OffloadingHistogramMetadata(\n"
    "                documentation=(\n"
    '                    "Histogram of the number of CPU blocks requested by each "\n'
    '                    "KV offload prepare_store call."\n'
    "                ),\n"
    "                buckets=(1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144),\n"
    "            ),\n"
    "        }\n"
)
assert anchor in src, f"anchor not found in {path}"
src = src.replace(
    anchor,
    anchor.replace(
        "        }\n",
        (
            "            CPUOffloadingMetrics.CPU_NUM_BLOCKS: OffloadingGaugeMetadata(\n"
            "                documentation=(\n"
            '                    "Total CPU/G2 offload blocks configured for this '
            'worker "\n'
            '                    "(cpu_bytes_to_use divided by vLLM\'s own real "\n'
            '                    "per-block byte size for the running model - exact, "\n'
            '                    "not estimated). Constant after construction, same "\n'
            '                    "denominator cpu_cache_usage_perc already divides by."\n'
            "                ),\n"
            "            ),\n"
            "        }\n"
        ),
    ),
    1,
)

with open(path, "w") as f:
    f.write(src)

print(f"G2 capacity metric-definition patch applied to {path}")
