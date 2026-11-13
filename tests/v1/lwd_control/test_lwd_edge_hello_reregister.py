# SPDX-License-Identifier: Apache-2.0
"""Edge control-plane periodic HELLO (re-registration).

The HELLO doubles as registration and its re-send heartbeat: a restarted
cloud's ROUTER holds no session until the next HELLO arrives, so a
completed handshake must never stop sending it — otherwise the edge stays
registered only in its own view while the cloud never learns the edge
again (no heartbeat, no business routing, a permanent admission gate).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from vllm.config.lwd.topology import LwdConnectionId
from vllm.v1.lwd_control.lwd_comm.lwd_control_plane import (
    LwdEdgeControlPlane,
)
from vllm.v1.lwd_control.lwd_comm.lwd_handshake import LwdEdgeHandshake
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdLinkHello

CONNECTION = LwdConnectionId(0, 0, 0)


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
