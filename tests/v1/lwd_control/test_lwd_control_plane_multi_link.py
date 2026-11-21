# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-link edge control plane: one edge process serving every cloud DP
rank of its instance.

Real ZMQ sockets on loopback, no engine: the checks cover the handshake to
every cloud, directed sends, per-link inbound attribution, the per-peer
hold of full links, the elastic partial start (a missing planned cloud
proceeds with the accepted subset) and the startup failure when no cloud
accepts at all.
"""

from __future__ import annotations

import time

import pytest

from vllm.v1.lwd_control.lwd_comm.lwd_control_plane import (
    LwdCloudControlPlane,
    LwdCloudControlPlaneSpec,
    LwdEdgeControlPlane,
    LwdEdgeControlPlaneSpec,
    LwdHandshakeError,
)

_LWD_TEST_HOST = "127.0.0.1"
_LWD_TEST_PORTS = (15660, 15661, 15662)
_LWD_WAIT_S = 15.0


class _Recorder:
    """Listener that records events for assertions."""

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


def _cloud(dp_rank: int, port: int, dp_size: int) -> LwdCloudControlPlane:
    return LwdCloudControlPlane(
        LwdCloudControlPlaneSpec(
            instance_id=0,
            dp_rank=dp_rank,
            dp_size=dp_size,
            bind_endpoint=f"tcp://{_LWD_TEST_HOST}:{port}",
        )
    )


def _edge(ports: tuple[int, ...]) -> LwdEdgeControlPlane:
    return LwdEdgeControlPlane(
        LwdEdgeControlPlaneSpec(
            instance_id=0,
            dp_rank=0,
            dp_size=1,
            cloud_endpoints=tuple(f"tcp://{_LWD_TEST_HOST}:{p}" for p in ports),
        )
    )


def _drain(recorders: list[_Recorder], count: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if sum(len(r.payloads) for r in recorders) >= count:
            return
        time.sleep(0.05)


def test_edge_serves_every_cloud_dp_rank() -> None:
    ports = _LWD_TEST_PORTS[:2]
    clouds = [_cloud(0, ports[0], 2), _cloud(1, ports[1], 2)]
    recorders = [_Recorder(), _Recorder()]
    senders = [cloud.start(r) for cloud, r in zip(clouds, recorders)]
    edge = _edge(ports)
    edge_recorder = _Recorder()
    edge_sender = edge.start(edge_recorder)
    try:
        peers = edge.wait_until_ready(_LWD_WAIT_S)
        assert sorted(peers) == [0, 1]
        assert [r.peers for r in recorders] == [[p] for p in peers.values()]

        # Directed sends reach only the addressed cloud DP rank.
        assert edge_sender.send(peers[0], b"to-dp0")
        assert edge_sender.send(peers[1], b"to-dp1")
        _drain(recorders, 2, _LWD_WAIT_S)
        assert [p for _peer, p in recorders[0].payloads] == [b"to-dp0"]
        assert [p for _peer, p in recorders[1].payloads] == [b"to-dp1"]

        # Inbound payloads attribute to the link's peer.
        assert senders[0].send(recorders[0].peers[0], b"from-dp0")
        assert senders[1].send(recorders[1].peers[0], b"from-dp1")
        _drain([edge_recorder], 2, _LWD_WAIT_S)
        assert {peer.dp_rank: p for peer, p in edge_recorder.payloads} == {
            0: b"from-dp0",
            1: b"from-dp1",
        }
        assert not edge_recorder.fatal
        assert all(not r.fatal for r in recorders)
    finally:
        edge.shutdown()
        for cloud in clouds:
            cloud.shutdown()


def test_edge_fails_fast_when_no_cloud_accepts() -> None:
    """Zero accepts within the window is a deployment error: the edge
    still fails fast (elastic partial start only tolerates a MISSING
    cloud, not a cloud-less edge)."""
    edge = _edge(_LWD_TEST_PORTS[:1] + (_LWD_TEST_PORTS[2],))
    try:
        with pytest.raises(LwdHandshakeError, match="did not accept"):
            edge.wait_until_ready(2.0)
    finally:
        edge.shutdown()


def test_edge_proceeds_with_accepted_subset_when_a_cloud_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Elastic partial start: the yaml plans two clouds but only one is
    launched -- the edge proceeds with the accepted subset after the
    straggler grace, and the accepted cloud is fully served."""
    import vllm.v1.lwd_control.lwd_comm.lwd_control_plane as plane_module

    monkeypatch.setattr(plane_module, "_LWD_STRAGGLER_GRACE_S", 0.2)
    ports = _LWD_TEST_PORTS[:2]
    clouds = [_cloud(0, ports[0], 2)]
    recorders = [_Recorder()]
    clouds[0].start(recorders[0])
    edge = _edge(ports)
    edge_recorder = _Recorder()
    edge_sender = edge.start(edge_recorder)
    try:
        peers = edge.wait_until_ready(_LWD_WAIT_S)
        assert len(peers) == 1
        assert not edge_recorder.fatal
        # The accepted subset is fully served.
        (peer,) = peers.values()
        assert edge_sender.send(peer, b"to-dp0")
        _drain(recorders, 1, _LWD_WAIT_S)
        assert [p for _peer, p in recorders[0].payloads] == [b"to-dp0"]
        assert not recorders[0].fatal
    finally:
        edge.shutdown()
        for cloud in clouds:
            cloud.shutdown()
