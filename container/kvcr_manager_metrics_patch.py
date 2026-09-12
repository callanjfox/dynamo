#!/usr/bin/env python3
"""Registers the new KVCR/router-adjacent index-health metrics (2026-09-11,
KVCR_INDEX_HEALTH_DESIGN.md) with vLLM's Prometheus wrapper by adding entries to
_kvcr_metric_definitions(). Applied to the vLLM secondary-tier adapter overlay
(vllm/v1/kv_offload/tiering/kvcr/manager.py) after it's copied from the pinned vLLM-fork
checkout, same pattern as kvcr_debug_patch.py and kvcr_g2_age_patch.py.

Only the G2 resident-age gauge (section 2b) needs an entry here - it's the one new metric
this pass adds under the vllm:kvcr_* namespace. Router-side metrics (device/host_pinned
tier age and redundancy) are separate Rust-side gauges registered directly against the
frontend component's own metrics registry, not through this vLLM-side path.
"""

import sys

path = sys.argv[1]

with open(path) as f:
    src = f.read()

import_anchor = (
    "from kvcr import (\n"
    "    DURATION_METRIC,\n"
    "    KVCR,\n"
    "    STATE_METRIC,\n"
    "    TRANSFER_BLOCKS_METRIC,\n"
    "    TRANSFER_BYTES_METRIC,\n"
    "    KVCRBindings,\n"
    ")\n"
)
assert import_anchor in src, f"import anchor not found in {path}"
src = src.replace(
    import_anchor,
    import_anchor + "from kvcr.core import G2_RESIDENT_AGE_METRIC\n",
    1,
)

definitions_anchor = (
    '        _vllm_metric_name(STATE_METRIC): OffloadingGaugeMetadata(\n'
    '            documentation="Current KVCR metadata and operation counts.",\n'
    '            labelnames=("resource",),\n'
    '        ),\n'
    "    }\n"
)
assert definitions_anchor in src, f"metric definitions anchor not found in {path}"
src = src.replace(
    definitions_anchor,
    (
        '        _vllm_metric_name(STATE_METRIC): OffloadingGaugeMetadata(\n'
        '            documentation="Current KVCR metadata and operation counts.",\n'
        '            labelnames=("resource",),\n'
        '        ),\n'
        '        _vllm_metric_name(G2_RESIDENT_AGE_METRIC): OffloadingGaugeMetadata(\n'
        "            documentation=(\n"
        '                "Resident age (seconds) of blocks currently held in this "\n'
        '                "worker\'s local G2/DRAM tier, by stat (min/p50/p99/max). "\n'
        '                "Diagnostic-only, computed from LRUPolicy\'s existing "\n'
        '                "last_access timestamps - no new tracking."\n'
        "            ),\n"
        '            labelnames=("stat",),\n'
        '        ),\n'
        "    }\n"
    ),
    1,
)

with open(path, "w") as f:
    f.write(src)

print(f"Manager metrics-registration patch applied to {path}")
