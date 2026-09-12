#!/usr/bin/env python3
"""TEMPORARY debug patch (2026-09-10, RUNBOOK_KVCR.md). Inserts one logging call into
_take_stored_event so we can empirically see every OffloadingEvent's medium that reaches
self-describing event translation - specifically, whether a secondary-tier/KVCR completion
(non-CPU medium) ever arrives at all. Remove once the cross-worker KVCR gap is resolved.
"""

import sys

path = sys.argv[1]

with open(path) as f:
    src = f.read()

anchor = (
    "    def _take_stored_event(self, event: OffloadingEvent) -> Iterable[KVCacheEvent]:"
)
assert anchor in src, f"anchor not found in {path}"

debug_line = (
    anchor
    + "\n        logger.warning("
    + "\n            'KVCR_DEBUG _take_stored_event medium=%s ownership=%s removal_expected=%s',"
    + "\n            event.medium, event.ownership, event.removal_expected,"
    + "\n        )"
)

src = src.replace(anchor, debug_line, 1)

with open(path, "w") as f:
    f.write(src)

print(f"DEBUG PATCH APPLIED to {path}")
