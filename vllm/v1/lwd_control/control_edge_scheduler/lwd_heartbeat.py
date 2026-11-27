# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LWD cloud heartbeat monitor.

An independent liveness component: it owns the per-connection activity
ledgers, decides when a cloud DP has gone silent (or has restarted), fires
the linked teardown, and publishes an immutable verdict to subscribers.

Consumers depend only on the read-only query surface and on subscription.
The admission gate, the scheduler and the health tracker never own or write
liveness state — the health tracker in particular only *caches* the verdict,
it does not run detection. This keeps the hard availability verdict (one
event, with side effects) separate from probe-target cooldown (a counter).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from vllm.config.lwd.topology import LwdConnectionId
from vllm.lwd_logger import lwd_logger
from vllm.v1.lwd_control.lwd_comm.lwd_restart_hooks import (
    notify_cloud_restarted,
)


@dataclass(frozen=True)
class LwdHeartbeatVerdict:
    """Immutable liveness snapshot published on each state transition.

    ``silent_routes`` is connection-grained (admission gate / scheduler).
    ``silent_clouds`` is cloud-grained (probe targeting): a cloud leaves the
    probe rotation only when EVERY adopted DP route of it is silent, since
    the coordination HTTP listener lives with the instance process.
    """

    silent_routes: frozenset[LwdConnectionId]
    silent_clouds: frozenset[int]


class LwdHeartbeatMonitor:
    """Own heartbeat/business activity, the silent verdict and its effects.

    Written on the control-plane IO thread (``note_activity``) and ticked on
    the engine thread (``sweep``); all state is guarded by one lock. Read-only
    queries may be called from either thread. Verdict callbacks run outside
    the lock, on whichever thread detected the transition.
    """

    def __init__(self, silent_s: float, sweep_interval_s: float = 1.0) -> None:
        self._silent_s = silent_s
        self._sweep_interval_s = sweep_interval_s
        self._lock = threading.Lock()
        self._hello_last: dict[LwdConnectionId, float] = {}
        self._business_last: dict[LwdConnectionId, float] = {}
        self._boot: dict[LwdConnectionId, str] = {}
        self._adopted: set[LwdConnectionId] = set()
        self._silent: set[LwdConnectionId] = set()
        self._subscribers: list[Callable[[LwdHeartbeatVerdict], None]] = []
        self._last_sweep = 0.0

    # ------------------------------------------------------------------
    # Subscriptions: the single event exit point
    # ------------------------------------------------------------------
    def subscribe(self, callback: Callable[[LwdHeartbeatVerdict], None]) -> None:
        """Register a verdict consumer. Callbacks must be non-blocking and
        thread-safe: they run on the IO thread or the engine thread."""
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def _emit(self) -> None:
        """Publish one immutable verdict, outside the lock.

        A subscriber that fails is logged, never swallowed silently and
        never allowed to kill the thread that detected the transition.
        """
        with self._lock:
            verdict = LwdHeartbeatVerdict(
                silent_routes=frozenset(self._silent),
                silent_clouds=self._silent_clouds_locked(),
            )
            subscribers = tuple(self._subscribers)
        for callback in subscribers:
            try:
                callback(verdict)
            except Exception:
                lwd_logger.exception("[LWD] heartbeat verdict subscriber failed")

    # ------------------------------------------------------------------
    # Write path: control-plane IO thread
    # ------------------------------------------------------------------
    def note_adopted(self, connection: LwdConnectionId) -> None:
        """Seed an activity account for a newly adopted cloud DP. Idempotent:
        an activity record that already arrived wins over the seeding."""
        with self._lock:
            self._adopted.add(connection)
            self._hello_last.setdefault(connection, time.monotonic())

    def note_activity(
        self,
        connection: LwdConnectionId,
        *,
        heartbeat_boot_id: str | None = None,
    ) -> None:
        """Record one valid frame from a cloud DP.

        Heartbeats carry a ``boot_id`` epoch (a change is a restart verdict);
        business payloads only refresh the activity timestamp. Either kind
        proves the process is alive and releases a silent verdict, so a cloud
        still answering requests is not torn down merely because its
        heartbeat was queued behind traffic or the cloud engine thread was
        busy with a long step.
        """
        now = time.monotonic()
        previous_boot = None
        boot_changed = False
        with self._lock:
            if heartbeat_boot_id is not None:
                self._hello_last[connection] = now
                previous_boot = self._boot.get(connection)
                self._boot[connection] = heartbeat_boot_id
                boot_changed = (
                    previous_boot is not None and previous_boot != heartbeat_boot_id
                )
            else:
                self._business_last[connection] = now
            was_silent = connection in self._silent
            if was_silent:
                self._silent.discard(connection)
        if was_silent:
            self._emit()
            lwd_logger.error(
                "[edge] %s back after silence (%s resumed)",
                connection,
                "heartbeat" if heartbeat_boot_id is not None else "cloud traffic",
            )
        if boot_changed:
            lwd_logger.error(
                "[edge] %s RESTARTED (boot_id %s -> %s)",
                connection,
                previous_boot,
                heartbeat_boot_id,
            )
            notify_cloud_restarted(
                connection.cloud_id, connection.dp_idx, "boot id changed"
            )

    # ------------------------------------------------------------------
    # Tick: engine thread
    # ------------------------------------------------------------------
    def sweep(self) -> None:
        """Rate-limited liveness sweep. Emits one verdict per sweep and fires
        the linked teardown for every connection that just went silent."""
        now = time.monotonic()
        with self._lock:
            if now - self._last_sweep < self._sweep_interval_s:
                return
            self._last_sweep = now
            gone = sorted(
                (connection, now - last)
                for connection, last in self._last_activity_locked().items()
                if connection not in self._silent and now - last > self._silent_s
            )
            for connection, _ in gone:
                self._silent.add(connection)
        self._emit()
        for connection, silent_for in gone:
            lwd_logger.error(
                "[edge] %s silent %.0fs (no heartbeat or cloud traffic)",
                connection,
                silent_for,
            )
            notify_cloud_restarted(
                connection.cloud_id,
                connection.dp_idx,
                f"no cloud activity {silent_for:.0f}s",
            )

    # ------------------------------------------------------------------
    # Read-only query surface (the Protocol consumers depend on)
    # ------------------------------------------------------------------
    def frozen_silent_routes(self) -> frozenset[LwdConnectionId]:
        """The last swept verdict, without the live staleness view."""
        with self._lock:
            return frozenset(self._silent)

    def silent_routes(self, now: float | None = None) -> frozenset[LwdConnectionId]:
        """Frozen verdict unioned with a live staleness view, so a cloud that
        went silent while the engine idled cannot admit the request that
        wakes it."""
        now = time.monotonic() if now is None else now
        with self._lock:
            stale = {
                connection
                for connection, last in self._last_activity_locked().items()
                if now - last > self._silent_s
            }
            return frozenset(self._silent) | stale

    def is_alive(self, connection: LwdConnectionId, now: float | None = None) -> bool:
        """Whether one connection currently counts as alive."""
        return connection not in self.silent_routes(now)

    def silent_clouds(self) -> frozenset[int]:
        """Cloud-grained verdict for probe targeting."""
        with self._lock:
            return self._silent_clouds_locked()

    def boot_id(self, connection: LwdConnectionId) -> str | None:
        """The last boot epoch seen for this connection, if any."""
        with self._lock:
            return self._boot.get(connection)

    def last_activity(self, connection: LwdConnectionId) -> float:
        """Newest heartbeat or business evidence timestamp for diagnostics."""
        with self._lock:
            return self._last_activity_locked().get(connection, 0.0)

    # ------------------------------------------------------------------
    # Locked helpers
    # ------------------------------------------------------------------
    def _last_activity_locked(self) -> dict[LwdConnectionId, float]:
        activity = dict(self._hello_last)
        for connection, last in self._business_last.items():
            if last > activity.get(connection, 0.0):
                activity[connection] = last
        return activity

    def _silent_clouds_locked(self) -> frozenset[int]:
        per_cloud: dict[int, list[bool]] = {}
        for connection in self._adopted:
            per_cloud.setdefault(connection.cloud_id, []).append(
                connection in self._silent
            )
        return frozenset(
            cloud_id
            for cloud_id, silent_flags in per_cloud.items()
            if all(silent_flags)
        )
