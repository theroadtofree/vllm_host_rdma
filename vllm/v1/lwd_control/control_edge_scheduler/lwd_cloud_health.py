# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Health-aware round-robin selection for direct multi-cloud LWD links."""

from __future__ import annotations

import time


class LwdCloudHealthTracker:
    """Track probe health without changing existing request bindings."""

    def __init__(self, cloud_ids: list[int] | tuple[int, ...]) -> None:
        self._cloud_ids = tuple(sorted(cloud_ids))
        self._failures = dict.fromkeys(self._cloud_ids, 0)
        self._cooldown_until: dict[int, float] = {}
        self._cursor = 0

    def candidates(self, now: float | None = None) -> tuple[int, ...]:
        now = time.monotonic() if now is None else now
        healthy = tuple(
            cloud_id
            for cloud_id in self._cloud_ids
            if self._cooldown_until.get(cloud_id, 0.0) <= now
        )
        # Fail open when every direct candidate is cooling down. Existing
        # request routes are owned by the scheduler and are never rewritten.
        return healthy or self._cloud_ids

    def next(self) -> int | None:
        candidates = self.candidates()
        if not candidates:
            return None
        cloud_id = candidates[self._cursor % len(candidates)]
        self._cursor += 1
        return cloud_id

    def record_success(self, cloud_id: int) -> None:
        self._failures[cloud_id] = 0
        self._cooldown_until.pop(cloud_id, None)

    def record_failure(
        self,
        cloud_id: int,
        *,
        now: float | None = None,
        threshold: int = 3,
        cooldown_s: float = 60.0,
    ) -> None:
        now = time.monotonic() if now is None else now
        failures = self._failures.get(cloud_id, 0) + 1
        self._failures[cloud_id] = failures
        if failures >= threshold:
            self._cooldown_until[cloud_id] = now + cooldown_s
