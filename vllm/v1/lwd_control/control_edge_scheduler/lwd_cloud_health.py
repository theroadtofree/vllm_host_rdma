# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Health-aware round-robin selection for direct cloud DP links."""

from __future__ import annotations

import time

LwdCloudDpId = tuple[int, int]


class LwdCloudHealthTracker:
    """Track DP health without changing existing request bindings."""

    def __init__(self, targets: tuple[LwdCloudDpId, ...]) -> None:
        self._targets = tuple(sorted(set(targets)))
        self._failures = dict.fromkeys(self._targets, 0)
        self._cooldown_until: dict[LwdCloudDpId, float] = {}
        self._cursor = 0

    def add_target(self, target: LwdCloudDpId) -> None:
        """Elastic scale-out: extend the rotation with a runtime-adopted
        cloud DP. Idempotent. The rotation is replaced by one atomic tuple
        swap, so a concurrent candidates() snapshot sees either the old or
        the new rotation, never a torn one."""
        if target in self._failures:
            return
        self._failures[target] = 0
        self._targets = tuple(sorted(self._failures))

    @property
    def targets(self) -> tuple[LwdCloudDpId, ...]:
        """The full rotation, cooling-down cloud DPs included (diagnostics)."""
        return self._targets

    def candidates(self, now: float | None = None) -> tuple[LwdCloudDpId, ...]:
        now = time.monotonic() if now is None else now
        healthy = tuple(
            target
            for target in self._targets
            if self._cooldown_until.get(target, 0.0) <= now
        )
        return healthy or self._targets

    def next(self, allowed: set[LwdCloudDpId] | None = None) -> LwdCloudDpId | None:
        """Pick the next rotation member. ``allowed`` scopes the choice
        (probe targeting excludes what liveness declared silent). An empty
        filter fails open: the admission gate owns the fully-silent steady
        state, this covers only the sweep-gap race."""
        candidates = self.candidates()
        if allowed is not None:
            live = tuple(target for target in candidates if target in allowed)
            candidates = live or candidates
        if not candidates:
            return None
        target = candidates[self._cursor % len(candidates)]
        self._cursor += 1
        return target

    def record_success(self, target: LwdCloudDpId) -> None:
        self._failures[target] = 0
        self._cooldown_until.pop(target, None)

    def record_failure(
        self,
        target: LwdCloudDpId,
        *,
        now: float | None = None,
        threshold: int = 3,
        cooldown_s: float = 60.0,
    ) -> bool:
        """Count one probe failure. Returns True when this failure put
        the cloud DP into cooldown (the caller's diagnostics cue)."""
        now = time.monotonic() if now is None else now
        failures = self._failures.get(target, 0) + 1
        self._failures[target] = failures
        if failures >= threshold:
            self._cooldown_until[target] = now + cooldown_s
            return True
        return False
