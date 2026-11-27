# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Health-aware round-robin selection for direct cloud DP links.

The tracker is the single owner of probe-target health: cooldown from
repeated probe failures, the control-plane liveness exclusion, and the
rotation itself. The hard availability verdict (admission gate, routing,
teardown) stays in the engine at connection granularity — this class only
decides *where to probe*, not *whether a request may be admitted*.
"""

from __future__ import annotations

import time

LwdCloudDpId = tuple[int, int]


def lwd_rotate_targets(
    targets: tuple[LwdCloudDpId, ...], edge_id: int
) -> tuple[LwdCloudDpId, ...]:
    """Rotate DP targets so the edge's staggered first target leads.

    Edges are staggered over the cloud x DP cross product: the cloud rank
    advances first (``edge_id % cloud_count``), then the DP rank inside that
    cloud (``edge_id // cloud_count`` modulo the cloud's DP count).
    """
    ordered = sorted(set(targets))
    if not ordered:
        return ()
    clouds = sorted({cloud_id for cloud_id, _dp_idx in ordered})
    first_cloud = clouds[edge_id % len(clouds)]
    cloud_dps = [dp_idx for cloud_id, dp_idx in ordered if cloud_id == first_cloud]
    first_dp = cloud_dps[(edge_id // len(clouds)) % len(cloud_dps)]
    start = ordered.index((first_cloud, first_dp))
    return tuple(ordered[start:] + ordered[:start])


class LwdCloudHealthTracker:
    """Track DP health without changing existing request bindings."""

    def __init__(self, targets: tuple[LwdCloudDpId, ...], edge_id: int = 0) -> None:
        self._edge_id = edge_id
        self._targets = lwd_rotate_targets(targets, edge_id)
        self._failures = dict.fromkeys(self._targets, 0)
        self._cooldown_until: dict[LwdCloudDpId, float] = {}
        self._silent: frozenset[LwdCloudDpId] = frozenset()
        self._cursor = 0

    def set_silent(self, targets: frozenset[LwdCloudDpId]) -> None:
        """Replace the liveness-excluded cloud DPs in one atomic swap.

        The engine pushes its verdict from the control-plane IO thread, so
        this is a single frozenset assignment: a concurrent candidates()
        reader sees either the old or the new exclusion, never a torn one.
        Unlike probe cooldown, this exclusion is a *hard* availability
        signal, but it still fails open when it covers everything — the
        admission gate owns the fully-silent steady state, and probing a
        cloud that just resumed is how the verdict gets cleared.
        """
        self._silent = frozenset(targets)

    @property
    def silent(self) -> frozenset[LwdCloudDpId]:
        """The current liveness exclusion (diagnostics)."""
        return self._silent

    def add_target(self, target: LwdCloudDpId) -> None:
        """Elastic scale-out: extend the rotation with a runtime-adopted
        cloud DP. Idempotent. The rotation is replaced by one atomic tuple
        swap, so a concurrent candidates() snapshot sees either the old or
        the new rotation, never a torn one."""
        if target in self._failures:
            return
        self._failures[target] = 0
        self._targets = lwd_rotate_targets(tuple(self._failures), self._edge_id)

    @property
    def targets(self) -> tuple[LwdCloudDpId, ...]:
        """The full rotation, cooling-down cloud DPs included (diagnostics)."""
        return self._targets

    def candidates(
        self,
        now: float | None = None,
        *,
        scope: frozenset[LwdCloudDpId] | None = None,
        exclude_silent: bool = False,
    ) -> tuple[LwdCloudDpId, ...]:
        """Return the currently selectable cloud DPs.

        ``scope`` narrows the answer to one edge's visible connections;
        ``exclude_silent`` applies the control-plane liveness verdict. Both
        are opt-in so ``next()`` keeps its original rotation semantics and
        the two filters can be ordered exactly where the caller needs them:

        1. drop cooling-down DPs, failing open when that empties the set;
        2. narrow to ``scope`` (visible connections of this edge);
        3. drop silent DPs, failing open when that empties the set.

        The admission gate owns the fully-silent steady state; step 3's
        fail-open only covers the admit/sweep race.
        """
        now = time.monotonic() if now is None else now
        healthy = tuple(
            target
            for target in self._targets
            if self._cooldown_until.get(target, 0.0) <= now
        )
        if not healthy:
            healthy = self._targets
        if scope is not None:
            healthy = tuple(target for target in healthy if target in scope)
        if exclude_silent and self._silent:
            live = tuple(target for target in healthy if target not in self._silent)
            healthy = live or healthy
        return healthy

    def next(self, allowed: set[LwdCloudDpId] | None = None) -> LwdCloudDpId | None:
        """Pick the next rotation member. ``allowed`` scopes the choice to
        the caller's already-filtered probe candidates; an empty filter
        fails open. Callers that want the liveness exclusion applied should
        use ``candidates(exclude_silent=True)`` when building ``allowed`` —
        ``next()`` itself keeps the original cooldown-only rotation so the
        two filters are never applied twice."""
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
