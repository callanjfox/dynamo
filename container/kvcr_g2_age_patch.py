#!/usr/bin/env python3
"""Adds a G2 (local DRAM) resident-age gauge to KVCR's own get_stats() (2026-09-11,
KVCR_INDEX_HEALTH_DESIGN.md section 2b). KVCR's LRUPolicy already stamps
_BlockRecord.last_access (via self._clock(), i.e. time.monotonic()) on every access, purely
for its own eviction ordering - this patch is the first thing to also read it back out as a
diagnostic. Scan cost is O(G2 slot count) (4096 in our config), bounded regardless of fleet
size, riding the same ~10s get_stats() tick every other vllm:kvcr_state gauge already uses -
no new sampling loop.

Deliberately NOT touching FW_G2 (fw_mem) or G3 residency here: this is scoped to LOCAL_G2
specifically (record.local_dram is not None), matching what our secondary_g2_slots config
actually allocates and what section 2b of the design doc asked for.
"""

import sys

path = sys.argv[1]

with open(path) as f:
    src = f.read()

metric_anchor = 'STATE_METRIC = "kvcr_state"'
assert metric_anchor in src, f"metric constant anchor not found in {path}"
src = src.replace(
    metric_anchor,
    metric_anchor + '\nG2_RESIDENT_AGE_METRIC = "kvcr_g2_resident_age_seconds"',
    1,
)

stats_anchor = (
    "        for resource, value in resources.items():\n"
    "            stats.set_gauge(STATE_METRIC, value, (resource,))\n"
    "        self._stats = self._stats_factory() if self._stats_factory else None\n"
)
assert stats_anchor in src, f"get_stats() anchor not found in {path}"

age_block = (
    "        for resource, value in resources.items():\n"
    "            stats.set_gauge(STATE_METRIC, value, (resource,))\n"
    "        g2_ages = [\n"
    "            now - record.last_access\n"
    "            for record in self._block_record_map.values()\n"
    "            if record.local_dram is not None and record.last_access is not None\n"
    "            for now in (self._clock(),)\n"
    "        ]\n"
    "        if g2_ages:\n"
    "            g2_ages.sort()\n"
    "            n = len(g2_ages)\n"
    "            g2_age_stats = {\n"
    "                'min': g2_ages[0],\n"
    "                'p50': g2_ages[min(n - 1, int(0.50 * n))],\n"
    "                'p99': g2_ages[min(n - 1, int(0.99 * n))],\n"
    "                'max': g2_ages[-1],\n"
    "            }\n"
    "            for stat, value in g2_age_stats.items():\n"
    "                stats.set_gauge(G2_RESIDENT_AGE_METRIC, value, (stat,))\n"
    "        self._stats = self._stats_factory() if self._stats_factory else None\n"
)
src = src.replace(stats_anchor, age_block, 1)

with open(path, "w") as f:
    f.write(src)

print(f"G2 resident-age patch applied to {path}")
