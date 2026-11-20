# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading

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
    """Elastic scale-out: a runtime-adopted cloud DP joins the probe
    rotation without disturbing it. The rotation is one atomic tuple
    swap, so a concurrent candidates() reader never sees a torn tuple."""
    tracker = LwdCloudHealthTracker([(0, 0)])
    assert [tracker.next() for _ in range(2)] == [(0, 0), (0, 0)]
    tracker.add_target((1, 0))
    assert [tracker.next() for _ in range(4)] == [(0, 0), (1, 0), (0, 0), (1, 0)]
    tracker.add_target((2, 0))
    # The cursor carries over the addition (no reset): the rotation just
    # continues over the widened tuple.
    assert [tracker.next() for _ in range(3)] == [(0, 0), (1, 0), (2, 0)]


def test_add_target_is_idempotent() -> None:
    tracker = LwdCloudHealthTracker([(0, 0), (2, 0)])
    tracker.add_target((1, 0))
    tracker.add_target((1, 0))
    tracker.add_target((0, 0))
    assert [tracker.next() for _ in range(3)] == [(0, 0), (1, 0), (2, 0)]


def test_added_target_can_cool_down_like_a_startup_target() -> None:
    tracker = LwdCloudHealthTracker([(0, 0)])
    tracker.add_target((1, 0))
    entered_cooldown = False
    for _ in range(3):
        # Only the threshold-crossing failure reports the cooldown entry.
        entered_cooldown = tracker.record_failure(
            (1, 0), now=10.0, threshold=3, cooldown_s=60.0
        )
    assert entered_cooldown is True
    assert tracker.candidates(now=10.0) == ((0, 0),)
    tracker.record_success((1, 0))
    assert tracker.candidates(now=10.0) == ((0, 0), (1, 0))


def test_next_fails_open_when_the_scope_covers_everything() -> None:
    """An empty allowed scope falls back to the full rotation: the
    admission gate owns the fully-silent steady state, this covers only
    the sweep-gap race. The cursor carries through the scoped period."""
    tracker = LwdCloudHealthTracker([(0, 0), (1, 0)])
    assert [tracker.next({(0, 0)}) for _ in range(3)] == [(0, 0), (0, 0), (0, 0)]
    assert [tracker.next({(1, 0)}) for _ in range(2)] == [(1, 0), (1, 0)]
    assert [tracker.next(set()) for _ in range(2)] == [(1, 0), (0, 0)]
    assert tracker.next() in ((0, 0), (1, 0))


def test_concurrent_candidates_reads_never_see_a_torn_rotation() -> None:
    tracker = LwdCloudHealthTracker([(0, 0)])
    stop = threading.Event()
    failures: list[Exception] = []

    def reader() -> None:
        try:
            while not stop.is_set():
                for target in tracker.candidates():
                    assert isinstance(target, tuple)
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
