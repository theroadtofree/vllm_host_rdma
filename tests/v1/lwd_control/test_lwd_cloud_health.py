# SPDX-License-Identifier: Apache-2.0
"""Runtime extension of the cloud health rotation (elastic scale-out).

A cloud adopted mid-run must join the probe round-robin without
disturbing the startup rotation: sorted order, idempotent replays, and
the cooldown exclusion apply to it exactly like to a startup cloud. The
rotation swap is atomic, so a concurrent candidates() reader never sees
a torn tuple.
"""

from __future__ import annotations

import threading

from vllm.v1.lwd_control.control_edge_scheduler.lwd_cloud_health import (
    LwdCloudHealthTracker,
)


def test_add_cloud_extends_the_round_robin_sorted() -> None:
    tracker = LwdCloudHealthTracker([0])
    assert [tracker.next() for _ in range(2)] == [0, 0]
    tracker.add_cloud(1)
    assert [tracker.next() for _ in range(4)] == [0, 1, 0, 1]
    tracker.add_cloud(2)
    # The cursor carries over the addition (no reset): the rotation just
    # continues over the widened tuple.
    assert [tracker.next() for _ in range(3)] == [0, 1, 2]


def test_add_cloud_is_idempotent() -> None:
    tracker = LwdCloudHealthTracker([0, 2])
    tracker.add_cloud(1)
    tracker.add_cloud(1)
    tracker.add_cloud(0)
    assert [tracker.next() for _ in range(3)] == [0, 1, 2]


def test_added_cloud_can_cool_down_like_a_startup_cloud() -> None:
    tracker = LwdCloudHealthTracker([0])
    tracker.add_cloud(1)
    for _ in range(3):
        tracker.record_failure(1)
    now = 1000.0
    # Cloud 1 is past the failure threshold: candidates drop it while
    # its cooldown holds, and the fail-open clause only kicks in when
    # EVERY cloud cools down.
    assert tracker.candidates(now=now) == (0,)
    tracker.record_success(1)
    assert tracker.candidates(now=now) == (0, 1)


def test_next_skips_excluded_clouds_with_fail_open() -> None:
    """The liveness exclusion: a fully silent cloud leaves probe targeting.
    Fails open when the exclusion covers everything (the admission gate
    owns the fully-silent steady state)."""
    tracker = LwdCloudHealthTracker([0, 1])
    assert [tracker.next(frozenset({1})) for _ in range(3)] == [0, 0, 0]
    assert [tracker.next(frozenset({0})) for _ in range(2)] == [1, 1]
    # Fail open: the exclusion covers everything (the admission gate owns
    # the fully-silent steady state). The cursor carries through the
    # excluded period — the rotation just continues over the full tuple.
    assert [tracker.next(frozenset({0, 1})) for _ in range(2)] == [1, 0]
    assert tracker.next(frozenset()) in (0, 1)


def test_concurrent_candidates_reads_never_see_a_torn_rotation() -> None:
    tracker = LwdCloudHealthTracker([0])
    stop = threading.Event()
    failures: list[Exception] = []

    def reader() -> None:
        try:
            while not stop.is_set():
                for cloud_id in tracker.candidates():
                    assert isinstance(cloud_id, int)
        except Exception as exc:  # pragma: no cover - only on regression
            failures.append(exc)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    for cloud_id in range(1, 200):
        tracker.add_cloud(cloud_id)
    stop.set()
    thread.join(timeout=5.0)
    assert not failures
    assert len(tracker.candidates()) == 200
