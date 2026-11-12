# SPDX-License-Identifier: Apache-2.0
"""Cloud engine retire of one connection's requests (phase 2).

A link-class error retires the connection's in-flight requests but does
NOT permanently fail the connection (the peer is coming back); the
permanent marking stays with the application-failure path.
"""

from __future__ import annotations

from vllm.config.lwd.topology import LwdConnectionId
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine import (
    LwdCloudEngineCore,
)

CONNECTION_A = LwdConnectionId(0, 0, 0)
CONNECTION_B = LwdConnectionId(1, 0, 0)


class _Queue:
    def __init__(self) -> None:
        self.items: list = []

    def put_nowait(self, item) -> None:  # noqa: ANN001
        self.items.append(item)


def _engine(monkeypatch, retired: list) -> LwdCloudEngineCore:  # noqa: ANN001
    engine = object.__new__(LwdCloudEngineCore)
    engine._lwd_request_routes = {"e0-r1": "peerA", "e0-r2": "peerB"}
    engine._lwd_failed_connections = set()
    engine.aborts_queue = _Queue()
    engine.input_queue = _Queue()
    monkeypatch.setattr(
        engine,
        "_lwd_connection_of",
        lambda peer: {
            "peerA": CONNECTION_A,
            "peerB": CONNECTION_B,
        }[peer],
    )
    monkeypatch.setattr(
        engine,
        "_lwd_retire_request",
        lambda request_id, preserve_prompt=False: retired.append(
            (request_id, preserve_prompt)
        ),
    )
    return engine


def test_retire_aborts_only_the_connections_requests(monkeypatch) -> None:  # noqa: ANN001
    retired: list = []
    engine = _engine(monkeypatch, retired)
    engine._lwd_retire_connection_requests(CONNECTION_A)
    # Only the connection's own requests retire (prompt preserved), each
    # with an abort for the engine loop.
    assert retired == [("e0-r1", True)]
    assert engine.aborts_queue.items == [["e0-r1"]]
    assert engine.input_queue.items == [(EngineCoreRequestType.ABORT, ["e0-r1"])]
    # Retiring does NOT decide the connection is gone.
    assert engine._lwd_failed_connections == set()


def test_retire_of_a_connection_without_requests_is_a_noop(monkeypatch) -> None:  # noqa: ANN001
    retired: list = []
    engine = _engine(monkeypatch, retired)
    engine._lwd_retire_connection_requests(CONNECTION_B)
    assert retired == [("e0-r2", True)]
    empty = LwdConnectionId(9, 9, 0)
    engine._lwd_retire_connection_requests(empty)
    assert len(retired) == 1
