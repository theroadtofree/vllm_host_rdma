# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Control-plane DSCP marking (``qos.ctrl_plane``) reaches the sockets.

Repro of the silent-unmark bug: the TOS used to be read from the
transient vLLM config global at socket-open time, where it is always
None -- control sockets open outside every ``set_current_vllm_config``
context -- so a configured dscp 48 never reached any socket while the
startup log still claimed the policy. The chain is now
``lwd_ctrl_tos(lwd_config) -> spec.ctrl_tos -> transport.open ->
ZMQ_TOS``; these tests deliberately never set the config global.
"""

from __future__ import annotations

import platform
from types import SimpleNamespace as NS

import pytest
import zmq

from vllm.config.lwd.topology import LwdQoS, LwdQosCtrlPlane
from vllm.v1.lwd_control.control_edge_scheduler.lwd_coord_client import LwdCoordClient
from vllm.v1.lwd_control.lwd_comm.lwd_control_plane import (
    LwdCloudControlPlane,
    LwdCloudControlPlaneSpec,
)
from vllm.v1.lwd_control.lwd_comm.lwd_control_plane_factory import lwd_ctrl_tos
from vllm.v1.lwd_control.lwd_comm.lwd_transport import (
    LwdCloudTransport,
    LwdEdgeTransport,
)

_HOST = "127.0.0.1"

# IP_TOS needs privileges on Windows (WSAEACCES), and the libzmq bundled
# with the pyzmq wheels asserts on the failure, aborting the process.
# Tests that reach a real setsockopt(TOS) run where the deployment runs.
_marks_ip_tos = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="IP_TOS setsockopt needs privileges on Windows; this pyzmq "
    "build asserts and aborts the process",
)


def _config_of(qos: LwdQoS | None) -> NS:
    return NS(topology=NS(qos=qos))


class _Recorder:
    """Minimal listener for plane.start()."""

    def on_peer_up(self, peer) -> None:
        return None

    def on_message(self, peer, payload: bytes) -> None:
        return None

    def on_fatal(self, reason: str) -> None:
        return None


def test_lwd_ctrl_tos_derives_from_the_topology_qos_block() -> None:
    marked = LwdQoS(ctrl_plane=LwdQosCtrlPlane(dscp=48))
    assert lwd_ctrl_tos(_config_of(marked)) == 192
    unset_field = LwdQoS(ctrl_plane=LwdQosCtrlPlane(dscp=None))
    assert lwd_ctrl_tos(_config_of(unset_field)) is None
    assert lwd_ctrl_tos(_config_of(LwdQoS(ctrl_plane=None))) is None
    assert lwd_ctrl_tos(_config_of(None)) is None


@_marks_ip_tos
def test_cloud_transport_open_marks_the_bound_router() -> None:
    ctx = zmq.Context()
    transport = LwdCloudTransport(f"tcp://{_HOST}:0")
    try:
        transport.open(ctx, 192)
        assert transport.socket.getsockopt(zmq.TOS) == 192
    finally:
        transport.close()
        ctx.term()


@_marks_ip_tos
def test_edge_transport_open_marks_every_link() -> None:
    ctx = zmq.Context()
    transport = LwdEdgeTransport(
        (f"tcp://{_HOST}:15671", f"tcp://{_HOST}:15672"),
        (b"edge0-link0", b"edge0-link1"),
    )
    try:
        transport.open(ctx, 192)
        assert [sock.getsockopt(zmq.TOS) for sock in transport.sockets] == [192, 192]
    finally:
        transport.close()
        ctx.term()


def test_transport_open_without_tos_leaves_sockets_unmarked() -> None:
    ctx = zmq.Context()
    transport = LwdCloudTransport(f"tcp://{_HOST}:0")
    try:
        transport.open(ctx, None)
        assert transport.socket.getsockopt(zmq.TOS) == 0
    finally:
        transport.close()
        ctx.term()


def test_configured_tos_fails_fast_without_zmq_tos_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = zmq.Context()
    unmarked = LwdCloudTransport(f"tcp://{_HOST}:0")
    marked = LwdCloudTransport(f"tcp://{_HOST}:0")
    monkeypatch.delattr(zmq, "TOS")
    try:
        unmarked.open(ctx, None)  # unmarked by design, needs no ZMQ_TOS
        with pytest.raises(RuntimeError, match="ZMQ_TOS"):
            marked.open(ctx, 192)
    finally:
        unmarked.close()
        marked.close()
        ctx.term()


def test_spec_rejects_out_of_range_ctrl_tos() -> None:
    with pytest.raises(ValueError, match="ctrl_tos"):
        LwdCloudControlPlaneSpec(
            instance_id=0,
            dp_rank=0,
            dp_size=1,
            bind_endpoint=f"tcp://{_HOST}:0",
            ctrl_tos=256,
        )


@_marks_ip_tos
def test_plane_passes_the_spec_ctrl_tos_to_the_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int | None] = []
    original = LwdCloudTransport.open

    def spy(self: LwdCloudTransport, ctx: zmq.Context, ctrl_tos: int | None = None):
        seen.append(ctrl_tos)
        return original(self, ctx, ctrl_tos)

    monkeypatch.setattr(LwdCloudTransport, "open", spy)
    plane = LwdCloudControlPlane(
        LwdCloudControlPlaneSpec(
            instance_id=0,
            dp_rank=0,
            dp_size=1,
            bind_endpoint=f"tcp://{_HOST}:0",
            ctrl_tos=192,
        )
    )
    try:
        plane.start(_Recorder())
    finally:
        plane.shutdown()
    assert seen == [192]


@_marks_ip_tos
def test_coord_client_marks_its_socket_with_the_given_tos() -> None:
    client = LwdCoordClient(0, f"tcp://{_HOST}:15673", 192)
    try:
        # This line keeps one socket per thread: the engine thread's wake
        # socket is the deterministic one to inspect; the stats thread
        # marks its own socket from the same stored tos.
        assert client._ctrl_tos == 192
        assert client._wake_socket.getsockopt(zmq.TOS) == 192
    finally:
        client.shutdown()
