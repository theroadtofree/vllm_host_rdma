# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Health rotation, runtime expansion and liveness exclusion tests."""

from __future__ import annotations

import threading
import time

from vllm.v1.lwd_control.control_edge_scheduler.lwd_cloud_health import (
    LwdCloudHealthTracker,
)


def test_failure_threshold_cools_only_one_dp_until_expiry() -> None:
    dp0, dp1 = (0, 0), (0, 1)
    tracker = LwdCloudHealthTracker((dp0, dp1))
    for _ in range(2):
        tracker.record_failure(dp0, now=10.0, threshold=3, cooldown_s=5.0)
        assert tracker.candidates(now=10.0) == (dp0, dp1)

    tracker.record_failure(dp0, now=10.0, threshold=3, cooldown_s=5.0)
    assert tracker.candidates(now=10.0) == (dp1,)
    assert tracker.candidates(now=14.9) == (dp1,)
    assert tracker.candidates(now=15.0) == (dp0, dp1)


def test_all_cooling_dps_fail_open() -> None:
    dp0, dp1 = (0, 0), (1, 0)
    tracker = LwdCloudHealthTracker((dp1, dp0))
    for target in (dp0, dp1):
        tracker.record_failure(target, now=10.0, threshold=1, cooldown_s=5.0)
    assert tracker.candidates(now=10.0) == (dp0, dp1)
    assert tracker.candidates(now=15.0) == (dp0, dp1)


def test_round_robin_staggers_cloud_and_dp_by_edge_id() -> None:
    targets = ((0, 0), (0, 1), (1, 0), (1, 1))
    starts = [
        LwdCloudHealthTracker(targets, edge_id=edge_id).next() for edge_id in range(4)
    ]
    assert starts == [(0, 0), (1, 0), (0, 1), (1, 1)]


def test_round_robin_staggers_dps_within_one_cloud() -> None:
    targets = ((0, 0), (0, 1))
    starts = [
        LwdCloudHealthTracker(targets, edge_id=edge_id).next() for edge_id in range(4)
    ]
    assert starts == [(0, 0), (0, 1), (0, 0), (0, 1)]


def test_round_robin_ranks_cloud_ids_by_order_not_value() -> None:
    tracker = LwdCloudHealthTracker(((5, 0), (7, 0), (9, 0)), edge_id=1)
    assert tracker.next() == (7, 0)


def test_round_robin_starts_after_a_cooling_first_cloud() -> None:
    tracker = LwdCloudHealthTracker(((0, 0), (1, 0), (2, 0)), edge_id=1)
    tracker.record_failure((1, 0), now=time.monotonic(), threshold=1, cooldown_s=60.0)
    assert tracker.next() == (2, 0)


def test_success_clears_failures_and_next_respects_logical_edge() -> None:
    dp0, dp1, other_edge = (0, 0), (0, 1), (1, 0)
    tracker = LwdCloudHealthTracker((dp0, dp1, other_edge))
    for _ in range(2):
        tracker.record_failure(dp0, now=10.0, threshold=2, cooldown_s=60.0)
    assert tracker.candidates(now=10.0) == (dp1, other_edge)

    tracker.record_success(dp0)
    assert tracker.candidates(now=10.0) == (dp0, dp1, other_edge)
    assert tracker.next(allowed={dp0, dp1}) in (dp0, dp1)
    tracker.record_failure(dp0, now=11.0, threshold=2, cooldown_s=60.0)
    assert tracker.candidates(now=11.0) == (dp0, dp1, other_edge)


def test_add_target_extends_the_round_robin_sorted() -> None:
    tracker = LwdCloudHealthTracker(((0, 0),))
    assert [tracker.next() for _ in range(2)] == [(0, 0), (0, 0)]
    tracker.add_target((1, 0))
    assert [tracker.next() for _ in range(4)] == [(0, 0), (1, 0), (0, 0), (1, 0)]
    tracker.add_target((2, 0))
    assert [tracker.next() for _ in range(3)] == [(0, 0), (1, 0), (2, 0)]


def test_add_target_is_idempotent() -> None:
    tracker = LwdCloudHealthTracker(((0, 0), (2, 0)))
    tracker.add_target((1, 0))
    tracker.add_target((1, 0))
    tracker.add_target((0, 0))
    assert [tracker.next() for _ in range(3)] == [(0, 0), (1, 0), (2, 0)]


def test_added_target_can_cool_down_like_a_startup_target() -> None:
    tracker = LwdCloudHealthTracker(((0, 0),))
    tracker.add_target((1, 0))
    for _ in range(3):
        tracker.record_failure((1, 0))
    assert tracker.candidates(now=1000.0) == ((0, 0),)
    tracker.record_success((1, 0))
    assert tracker.candidates(now=1000.0) == ((0, 0), (1, 0))


def test_add_target_preserves_edge_staggering() -> None:
    tracker = LwdCloudHealthTracker(((0, 0),), edge_id=1)
    tracker.add_target((1, 0))
    assert tracker.targets == ((1, 0), (0, 0))
    assert tracker.next() == (1, 0)


def test_next_scopes_candidates_and_fails_open_when_allowed_is_empty() -> None:
    """Allowed targets are included; an empty allowed set fails open."""
    tracker = LwdCloudHealthTracker(((0, 0), (1, 0)))
    assert [tracker.next(frozenset({(1, 0)})) for _ in range(3)] == [
        (1, 0),
        (1, 0),
        (1, 0),
    ]
    assert [tracker.next(frozenset({(0, 0)})) for _ in range(2)] == [
        (0, 0),
        (0, 0),
    ]
    assert [tracker.next(frozenset()) for _ in range(2)] == [
        (1, 0),
        (0, 0),
    ]


def test_silent_dps_leave_the_rotation_and_recover() -> None:
    """The control-plane liveness verdict lives in the tracker now: a
    silent cloud DP leaves candidates() and returns on the empty verdict,
    without a second liveness set being kept in the probe manager."""
    dp0, dp1 = (0, 0), (1, 0)
    tracker = LwdCloudHealthTracker((dp0, dp1))
    tracker.set_silent(frozenset({dp1}))
    assert tracker.silent == frozenset({dp1})
    assert tracker.candidates(now=10.0, exclude_silent=True) == (dp0,)
    # next() keeps its original semantics: the caller's allowed set wins.
    assert tracker.next(allowed=frozenset({dp1})) == dp1

    tracker.set_silent(frozenset())
    assert tracker.candidates(now=10.0, exclude_silent=True) == (dp0, dp1)


def test_silent_fail_open_when_every_target_is_silent() -> None:
    """The admission gate owns the fully-silent steady state; probe
    targeting still fails open so a resumed cloud can be probed at all."""
    dp0, dp1 = (0, 0), (1, 0)
    tracker = LwdCloudHealthTracker((dp0, dp1))
    tracker.set_silent(frozenset({dp0, dp1}))
    assert tracker.candidates(now=10.0, exclude_silent=True) == (dp0, dp1)


def test_cooldown_fail_open_does_not_resurrect_a_silent_dp() -> None:
    """Ordering: cooldown fallback first, liveness exclusion last. A
    cooling-down DP that is also declared silent stays excluded."""
    dp0, dp1 = (0, 0), (1, 0)
    tracker = LwdCloudHealthTracker((dp0, dp1))
    for target in (dp0, dp1):
        tracker.record_failure(target, now=10.0, threshold=1, cooldown_s=60.0)
    tracker.set_silent(frozenset({dp1}))
    assert tracker.candidates(now=10.0) == (dp0, dp1)  # cooldown fail-open
    assert tracker.candidates(now=10.0, exclude_silent=True) == (dp0,)


def test_silent_exclusion_happens_after_scope_narrowing() -> None:
    """Scope (this edge's visible DPs) is applied before the silent
    fail-open, so the fallback stays within the visible set — the same
    ordering the probe manager used to do by hand."""
    dp0, dp1, dp2 = (0, 0), (1, 0), (2, 0)
    tracker = LwdCloudHealthTracker((dp0, dp1, dp2))
    tracker.set_silent(frozenset({dp0, dp1}))
    scope = frozenset({dp0, dp1})
    assert tracker.candidates(now=10.0, scope=scope, exclude_silent=True) == (
        dp0,
        dp1,
    )
    assert tracker.candidates(now=10.0, scope=scope) == (dp0, dp1)
    assert tracker.candidates(now=10.0, exclude_silent=True) == (dp2,)


def test_set_silent_is_an_atomic_swap() -> None:
    tracker = LwdCloudHealthTracker(((0, 0), (1, 0)))
    for cloud_id in range(2, 100):
        tracker.add_target((cloud_id, 0))
    stop = threading.Event()
    failures: list[Exception] = []

    def reader() -> None:
        try:
            while not stop.is_set():
                for target in tracker.candidates(exclude_silent=True):
                    assert len(target) == 2
        except Exception as exc:  # pragma: no cover - only on regression
            failures.append(exc)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    for cloud_id in range(100):
        tracker.set_silent(frozenset({(cloud_id, 0)}))
    tracker.set_silent(frozenset())
    stop.set()
    thread.join(timeout=5.0)
    assert not failures
    assert len(tracker.candidates(exclude_silent=True)) == 100


def test_concurrent_candidates_reads_never_see_a_torn_rotation() -> None:
    tracker = LwdCloudHealthTracker(((0, 0),))
    stop = threading.Event()
    failures: list[Exception] = []

    def reader() -> None:
        try:
            while not stop.is_set():
                for target in tracker.candidates():
                    assert len(target) == 2
        except Exception as exc:  # pragma: no cover - only on regression
            failures.append(exc)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    for cloud_id in range(1, 200):
        tracker.add_target((cloud_id, 0))
    stop.set()
    thread.join(timeout=5.0)
    assert not failures
    assert len(tracker.candidates()) == 200
