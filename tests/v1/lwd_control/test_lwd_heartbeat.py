# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LwdHeartbeatMonitor: independent detection, verdict subscription and the
read-only query surface.

These tests exercise the monitor directly, with the engine completely out of
the picture — the point of the component boundary is that consumers depend
on an interface, not on engine-owned state.
"""

from __future__ import annotations

import time
from dataclasses import FrozenInstanceError

import pytest

from vllm.config.lwd.topology import LwdConnectionId
from vllm.v1.lwd_control.control_edge_scheduler import lwd_heartbeat
from vllm.v1.lwd_control.control_edge_scheduler.lwd_heartbeat import (
    LwdHeartbeatMonitor,
)

CONNECTION = LwdConnectionId(0, 0, 0)
DP1 = LwdConnectionId(0, 0, 1)
OTHER_CLOUD = LwdConnectionId(0, 1, 0)


def _monitor(*connections: LwdConnectionId) -> LwdHeartbeatMonitor:
    monitor = LwdHeartbeatMonitor(10.0, 1.0)
    for connection in connections:
        monitor.note_adopted(connection)
    return monitor


def test_activity_and_query_surface() -> None:
    monitor = _monitor(CONNECTION)
    assert monitor.is_alive(CONNECTION)
    assert monitor.boot_id(CONNECTION) is None
    assert monitor.last_activity(CONNECTION) > 0

    monitor.note_activity(CONNECTION, heartbeat_boot_id="boot-a")
    assert monitor.boot_id(CONNECTION) == "boot-a"

    monitor._hello_last[CONNECTION] = time.monotonic() - 30.0
    assert not monitor.is_alive(CONNECTION)
    assert monitor.silent_routes() == frozenset({CONNECTION})
    # frozen_silent_routes() has no live staleness view: no sweep has run.
    assert monitor.frozen_silent_routes() == frozenset()


def test_business_activity_is_independent_evidence() -> None:
    monitor = _monitor(CONNECTION)
    monitor._hello_last[CONNECTION] = time.monotonic() - 30.0
    monitor.note_activity(CONNECTION)
    assert monitor.is_alive(CONNECTION)
    monitor.sweep()
    assert monitor.frozen_silent_routes() == frozenset()


def test_sweep_publishes_verdict_and_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    notified: list = []
    monkeypatch.setattr(
        lwd_heartbeat,
        "notify_cloud_restarted",
        lambda cloud_id, dp_idx, reason: notified.append((cloud_id, dp_idx, reason)),
    )
    monitor = _monitor(CONNECTION)
    verdicts: list = []
    monitor.subscribe(verdicts.append)

    monitor._hello_last[CONNECTION] = time.monotonic() - 30.0
    monitor.sweep()
    assert monitor.frozen_silent_routes() == frozenset({CONNECTION})
    assert verdicts[-1].silent_routes == frozenset({CONNECTION})
    assert verdicts[-1].silent_clouds == frozenset({0})
    assert len(notified) == 1
    assert "no cloud activity" in notified[0][2]

    # One frame is enough to release the verdict and publish the recovery.
    monitor.note_activity(CONNECTION, heartbeat_boot_id="boot-a")
    assert monitor.frozen_silent_routes() == frozenset()
    assert verdicts[-1].silent_routes == frozenset()
    assert verdicts[-1].silent_clouds == frozenset()
    assert len(notified) == 1  # recovery is not a teardown


def test_boot_id_change_fires_teardown_but_not_a_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notified: list = []
    monkeypatch.setattr(
        lwd_heartbeat,
        "notify_cloud_restarted",
        lambda cloud_id, dp_idx, reason: notified.append((cloud_id, dp_idx, reason)),
    )
    monitor = _monitor(CONNECTION)
    verdicts: list = []
    monitor.subscribe(verdicts.append)

    monitor.note_activity(CONNECTION, heartbeat_boot_id="boot-a")
    assert notified == []  # first epoch is not a restart
    monitor.note_activity(CONNECTION, heartbeat_boot_id="boot-b")
    assert notified == [(0, 0, "boot id changed")]
    # A boot change is a different kind of event: no silent verdict changed.
    assert verdicts == []


def test_silent_clouds_require_every_adopted_dp_to_be_silent() -> None:
    monitor = _monitor(CONNECTION, DP1, OTHER_CLOUD)
    monitor._hello_last[CONNECTION] = time.monotonic() - 30.0
    monitor.sweep()
    # cloud 0 still has a live DP: it keeps answering probes.
    assert monitor.frozen_silent_routes() == frozenset({CONNECTION})
    assert monitor.silent_clouds() == frozenset()

    monitor._hello_last[DP1] = time.monotonic() - 30.0
    monitor._last_sweep = 0.0
    monitor.sweep()
    assert monitor.silent_clouds() == frozenset({0})
    # cloud 1 was seeded fresh and is untouched.
    assert monitor.is_alive(OTHER_CLOUD)


def test_subscriber_failure_is_logged_not_swallowed_or_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _monitor(CONNECTION)
    monkeypatch.setattr(lwd_heartbeat, "notify_cloud_restarted", lambda *args: None)
    seen: list = []

    def bad_subscriber(verdict):  # noqa: ANN001
        raise RuntimeError("subscriber boom")

    monitor.subscribe(bad_subscriber)
    monitor.subscribe(seen.append)
    monitor._hello_last[CONNECTION] = time.monotonic() - 30.0
    monitor.sweep()  # must not raise
    assert len(seen) == 1


def test_verdict_is_immutable() -> None:
    verdict = lwd_heartbeat.LwdHeartbeatVerdict(
        silent_routes=frozenset({CONNECTION}), silent_clouds=frozenset({0})
    )
    with pytest.raises(FrozenInstanceError):
        verdict.silent_routes = frozenset()  # type: ignore[misc]
