# SPDX-License-Identifier: Apache-2.0
"""Edge control-plane periodic HELLO (re-registration).

The HELLO doubles as registration and its re-send heartbeat: a restarted
cloud's ROUTER holds no session until the next HELLO arrives, so a
completed handshake must never stop sending it — otherwise the edge stays
registered only in its own view while the cloud never learns the edge
again (no heartbeat, no business routing, a permanent admission gate).

Also covers the startup barrier's elastic partial start: the edge waits
for the FIRST accept within the long window, then gives the remaining
planned clouds a short straggler grace and proceeds with the accepted
subset (the periodic HELLO keeps the missing handshakes alive for runtime
adoption).
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from vllm.config.lwd.topology import LwdConnectionId
from vllm.v1.lwd_control.lwd_comm.lwd_control_plane import (
    LwdEdgeControlPlane,
    LwdHandshakeError,
)
from vllm.v1.lwd_control.lwd_comm.lwd_handshake import LwdEdgeHandshake
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdLinkHello

CONNECTION = LwdConnectionId(0, 0, 0)
CONNECTION_C1 = LwdConnectionId(0, 1, 0)


def _hello() -> LwdLinkHello:
    return LwdLinkHello(
        link_version=1,
        message_version=1,
        instance_id=0,
        dp_rank=0,
        dp_size=1,
        session_id="edge-session",
    )


class _FakeTransport:
    def __init__(self) -> None:
        self.sent: list = []

    def send(self, kind, payload, link_index):  # noqa: ANN001
        self.sent.append((kind, payload, link_index))
        return True


def _plane(monkeypatch, completed: bool) -> LwdEdgeControlPlane:
    plane = object.__new__(LwdEdgeControlPlane)
    handshake = LwdEdgeHandshake(_hello(), hello_retry_interval_s=5.0)
    if completed:
        handshake.cloud = SimpleNamespace(instance_id=0, dp_rank=0)
    handshake._last_hello_at = None
    plane._handshakes = {CONNECTION: handshake}
    plane._link_indices = {CONNECTION: 0}
    plane._num_hellos_sent = 0
    plane._endpoints = ["cloud-endpoint"]
    plane._label = "test-edge"
    return plane


def test_completed_handshake_keeps_resending_the_hello(monkeypatch) -> None:  # noqa: ANN001, ARG001
    """Regression (silent cloud restart): a completed handshake must keep
    sending the periodic HELLO — it is the cloud's only re-registration
    path after a restart."""
    plane = _plane(monkeypatch, completed=True)
    transport = _FakeTransport()
    plane._on_tick(transport, time.monotonic())
    assert len(transport.sent) == 1


def test_hello_resend_is_interval_limited(monkeypatch) -> None:  # noqa: ANN001, ARG001
    """Within the retry interval the tick does not re-send."""
    plane = _plane(monkeypatch, completed=True)
    transport = _FakeTransport()
    now = time.monotonic()
    plane._on_tick(transport, now)
    assert len(transport.sent) == 1
    plane._on_tick(transport, now + 1.0)  # within 5 s
    assert len(transport.sent) == 1
    plane._on_tick(transport, now + 6.0)
    assert len(transport.sent) == 2


# --------------------------------------------------------------------- #
# Startup barrier: elastic partial start (cloud side)                    #
# --------------------------------------------------------------------- #


def _barrier_plane(accepted: tuple[int, ...]) -> LwdEdgeControlPlane:
    """A not-started plane with two planned cloud connections; the listed
    cloud ids have accepted their handshake."""
    plane = object.__new__(LwdEdgeControlPlane)
    handshakes = {}
    for connection in (CONNECTION, CONNECTION_C1):
        handshake = LwdEdgeHandshake(_hello(), hello_retry_interval_s=5.0)
        if connection.cloud_id in accepted:
            handshake.cloud = SimpleNamespace(
                instance_id=connection.cloud_id, dp_rank=0
            )
        handshakes[connection] = handshake
    plane._handshakes = handshakes
    plane._settled = threading.Event()
    plane._handshake_error = None
    plane._num_hellos_sent = 0
    plane._label = "test-edge"
    plane.shutdown = lambda: None  # stubbed: the plane was never started
    return plane


def test_wait_until_ready_returns_every_accepted_cloud() -> None:
    """A full deployment still collects the complete set without waiting
    out any grace window."""
    plane = _barrier_plane(accepted=(0, 1))
    peers = plane.wait_until_ready(10.0)
    assert set(peers) == {CONNECTION, CONNECTION_C1}


def test_wait_until_ready_proceeds_with_the_accepted_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Elastic partial start: the yaml plans two clouds, only one is
    launched -- the barrier proceeds with the accepted subset after the
    straggler grace instead of failing, and the missing handshake keeps
    its retry state (the periodic HELLO owns the late adoption)."""
    import vllm.v1.lwd_control.lwd_comm.lwd_control_plane as plane_module

    monkeypatch.setattr(plane_module, "_LWD_STRAGGLER_GRACE_S", 0.1)
    plane = _barrier_plane(accepted=(0,))
    peers = plane.wait_until_ready(10.0)
    assert set(peers) == {CONNECTION}
    assert plane._handshake_error is None
    assert plane._handshakes[CONNECTION_C1].cloud is None


def test_wait_until_ready_fails_fast_when_nothing_accepted() -> None:
    """Zero accepts within the window is a deployment error, not a partial
    start: the barrier still fails fast."""
    plane = _barrier_plane(accepted=())
    with pytest.raises(LwdHandshakeError, match="did not accept"):
        plane.wait_until_ready(0.2)
