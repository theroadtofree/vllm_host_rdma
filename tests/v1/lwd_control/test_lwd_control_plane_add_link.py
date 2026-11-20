# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime link addition on the edge control plane (elastic scale-out,
phase B): `add_link` mid-run opens a DEALER on the IO thread at a poll
gap, the handshake (same session id) registers with the new cloud, and a
directed send reaches it. Real ZMQ sockets on loopback, no engine."""

from __future__ import annotations

import threading
import time

import pytest

from vllm.config.lwd.topology import LwdConnectionId
from vllm.v1.lwd_control.lwd_comm.lwd_control_plane import (
    LwdCloudControlPlane,
    LwdCloudControlPlaneSpec,
    LwdEdgeControlPlane,
    LwdEdgeControlPlaneSpec,
)
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdPeer

_LWD_TEST_HOST = "127.0.0.1"
_PORT_PLANNED = 15670
_PORT_ADDED = 15671
_LWD_WAIT_S = 15.0


class _Recorder:
    def __init__(self) -> None:
        self.peers: list = []
        self.payloads: list = []
        self.fatal: list[str] = []

    def on_peer_up(self, peer) -> None:
        self.peers.append(peer)

    def on_message(self, peer, payload: bytes) -> None:
        self.payloads.append((peer, payload))

    def on_fatal(self, reason: str) -> None:
        self.fatal.append(reason)


def _wait_for(predicate, timeout_s: float = _LWD_WAIT_S) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    pytest.fail("condition not reached within the timeout")


def test_added_link_registers_and_carries_payloads() -> None:
    """The planned cloud (0) is up at start; cloud 1 exists but joins the
    edge only through add_link: the link opens on the IO thread, the
    periodic HELLO registers with cloud 1, and a directed send reaches
    it — the same guarantees a startup-planned link has."""
    recorder_planned = _Recorder()
    recorder_added = _Recorder()
    cloud_planned = LwdCloudControlPlane(
        LwdCloudControlPlaneSpec(
            instance_id=0,
            dp_rank=0,
            dp_size=1,
            bind_endpoint=f"tcp://{_LWD_TEST_HOST}:{_PORT_PLANNED}",
        )
    )
    cloud_added = LwdCloudControlPlane(
        LwdCloudControlPlaneSpec(
            instance_id=1,
            dp_rank=0,
            dp_size=1,
            bind_endpoint=f"tcp://{_LWD_TEST_HOST}:{_PORT_ADDED}",
        )
    )
    edge = LwdEdgeControlPlane(
        LwdEdgeControlPlaneSpec(
            instance_id=0,
            dp_rank=0,
            dp_size=1,
            cloud_endpoints=(f"tcp://{_LWD_TEST_HOST}:{_PORT_PLANNED}",),
        )
    )
    try:
        sender = edge.start(recorder_planned)
        cloud_planned.start(recorder_planned)
        cloud_added.start(recorder_added)
        _wait_for(lambda: len(edge.registered_peers()) == 1)

        done = threading.Event()
        edge.add_link(
            f"tcp://{_LWD_TEST_HOST}:{_PORT_ADDED}",
            LwdConnectionId(0, 1, 0),
            1,
            done=done,
        )
        assert done.wait(_LWD_WAIT_S), "the IO thread never applied add_link"

        # The handshake registers with the added cloud within one hello
        # retry, and the directed send reaches it over the new link.
        _wait_for(lambda: len(edge.registered_peers()) == 2)
        assert sender.send(LwdPeer(1, 0), b"runtime-added-payload")
        _wait_for(lambda: recorder_added.payloads)
    finally:
        edge.shutdown()
        cloud_planned.shutdown()
        cloud_added.shutdown()


def test_add_link_is_idempotent_for_a_replayed_increment() -> None:
    """A re-detected topology increment replays the same connection: the
    second add is dropped (no second session, the live one undisturbed)."""
    recorder = _Recorder()
    cloud = LwdCloudControlPlane(
        LwdCloudControlPlaneSpec(
            instance_id=1,
            dp_rank=0,
            dp_size=1,
            bind_endpoint=f"tcp://{_LWD_TEST_HOST}:{_PORT_ADDED}",
        )
    )
    edge = LwdEdgeControlPlane(
        LwdEdgeControlPlaneSpec(
            instance_id=0,
            dp_rank=0,
            dp_size=1,
            cloud_endpoints=(f"tcp://{_LWD_TEST_HOST}:{_PORT_PLANNED}",),
        )
    )
    try:
        edge.start(recorder)
        cloud.start(recorder)
        connection = LwdConnectionId(0, 1, 0)
        first_done = threading.Event()
        second_done = threading.Event()
        edge.add_link(
            f"tcp://{_LWD_TEST_HOST}:{_PORT_ADDED}",
            connection,
            1,
            done=first_done,
        )
        assert first_done.wait(_LWD_WAIT_S)
        edge.add_link(
            f"tcp://{_LWD_TEST_HOST}:{_PORT_ADDED}",
            connection,
            1,
            done=second_done,
        )
        assert second_done.wait(_LWD_WAIT_S)
        # One handshake per connection: the replay did not grow the table.
        assert connection in edge._handshakes
        assert len(edge._handshakes) == 2  # the planned one + the added one
        _wait_for(lambda: len(edge.registered_peers()) == 1)
    finally:
        edge.shutdown()
        cloud.shutdown()
