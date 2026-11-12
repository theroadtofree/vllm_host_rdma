# SPDX-License-Identifier: Apache-2.0
"""Liveness layer of the edge engine (peer-restart line, phase 2).

Cloud heartbeats keep per-connection accounts; a silent verdict fires the
linked teardown through the restart-hook registry and freezes the
admission gate's snapshot; the next beat releases it. A boot-id change is
a restart verdict on its own.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import pytest

from vllm.config.lwd.topology import LwdConnectionId
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine import (
    LwdCloudEngineCore,
)
from vllm.v1.lwd_control.control_edge_scheduler import lwd_edge_engine
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
    LwdEdgeEngineCore,
)
from vllm.v1.lwd_control.lwd_comm import lwd_restart_hooks as hooks_module
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LwdCloudHeartbeat,
    lwd_decode_cloud_to_edge,
    lwd_encode_message,
)

CONNECTION = LwdConnectionId(0, 0, 0)
CONNECTION_E1 = LwdConnectionId(1, 1, 0)


@pytest.fixture(autouse=True)
def _restore_hook_registry() -> Iterator[None]:
    saved = list(hooks_module._hooks)
    yield
    hooks_module._hooks[:] = saved


def _raising_hook(cloud_id: int, dp_idx: int, reason: str) -> None:
    raise RuntimeError("hook boom")


def _edge_engine(
    monkeypatch: pytest.MonkeyPatch, connections, notified: list
) -> LwdEdgeEngineCore:
    engine = object.__new__(LwdEdgeEngineCore)
    engine._lwd_hello_lock = threading.Lock()
    engine._lwd_hello_last = {
        connection: time.monotonic() for connection in connections
    }
    engine._lwd_hello_boot = {}
    engine._lwd_silent = set()
    engine._lwd_silent_routes = frozenset()
    engine._lwd_hello_silent_s = 10.0
    engine._lwd_last_sweep = 0.0
    engine._lwd_clouds = {connection: None for connection in connections}
    monkeypatch.setattr(
        lwd_edge_engine,
        "notify_cloud_restarted",
        lambda cloud_id, dp_idx, reason: notified.append((cloud_id, dp_idx, reason)),
    )
    return engine


def test_heartbeat_codec_round_trip() -> None:
    decoded = lwd_decode_cloud_to_edge(lwd_encode_message(LwdCloudHeartbeat("boot-1")))
    assert isinstance(decoded, LwdCloudHeartbeat)
    assert decoded.boot_id == "boot-1"


def test_beat_detects_a_boot_id_change(monkeypatch: pytest.MonkeyPatch) -> None:
    notified: list = []
    engine = _edge_engine(monkeypatch, [CONNECTION], notified)
    engine._lwd_beat(CONNECTION, "boot-a")
    assert notified == []  # the first boot id is not a restart
    engine._lwd_beat(CONNECTION, "boot-b")
    assert notified == [(0, 0, "boot id changed")]


def test_beat_releases_the_gate_after_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    notified: list = []
    engine = _edge_engine(monkeypatch, [CONNECTION], notified)
    engine._lwd_silent = {CONNECTION}
    engine._lwd_silent_routes = frozenset({CONNECTION})
    engine._lwd_beat(CONNECTION, "boot-a")
    assert engine._lwd_silent_routes == frozenset()
    assert notified == []


def test_sweep_marks_silent_and_fires_the_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notified: list = []
    engine = _edge_engine(monkeypatch, [CONNECTION], notified)
    engine._lwd_hello_last[CONNECTION] = time.monotonic() - 30.0
    engine._lwd_sweep_liveness()
    assert engine._lwd_silent_routes == frozenset({CONNECTION})
    assert len(notified) == 1
    assert notified[0][0] == 0 and notified[0][1] == 0
    assert "heartbeat silent" in notified[0][2]
    # The silent transition fires once: a second sweep does not re-notify.
    engine._lwd_last_sweep = 0.0
    engine._lwd_sweep_liveness()
    assert len(notified) == 1


def test_sweep_is_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    notified: list = []
    engine = _edge_engine(monkeypatch, [CONNECTION], notified)
    engine._lwd_hello_last[CONNECTION] = time.monotonic() - 30.0
    engine._lwd_sweep_liveness()
    engine._lwd_sweep_liveness()  # within the sweep interval: no-op
    assert len(notified) == 1


def test_gate_rejects_only_when_the_routes_are_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notified: list = []
    engine = _edge_engine(monkeypatch, [CONNECTION, CONNECTION_E1], notified)
    assert engine._lwd_gate_reason("plain-req") is None
    engine._lwd_silent_routes = frozenset({CONNECTION_E1})
    # Whole-route silence rejects everything.
    engine._lwd_silent_routes = frozenset({CONNECTION, CONNECTION_E1})
    assert engine._lwd_gate_reason("plain-req") is not None
    # Partial silence: an unwrapped id (edge 0) still has a live route.
    engine._lwd_silent_routes = frozenset({CONNECTION_E1})
    assert engine._lwd_gate_reason("plain-req") is None
    # A wrapped id whose source edge is fully silent is rejected.
    assert engine._lwd_gate_reason("e1-wrapped-req") is not None


def test_registry_fires_every_hook_and_survives_exceptions() -> None:
    fired: list = []
    hooks_module.register_cloud_restart_hook(
        lambda cloud_id, dp_idx, reason: fired.append(("ok", cloud_id))
    )
    hooks_module.register_cloud_restart_hook(_raising_hook)
    hooks_module.notify_cloud_restarted(1, 0, "test reason")
    assert fired == [("ok", 1)]


def test_cloud_engine_sends_one_heartbeat_per_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = object.__new__(LwdCloudEngineCore)
    sent: list = []
    engine._lwd_edges = ["peer-a"]
    engine._lwd_boot_id = "boot-42"
    engine._lwd_hello_interval_s = 1000.0
    engine._lwd_last_heartbeat_at = 0.0
    monkeypatch.setattr(
        engine,
        "_lwd_send",
        lambda peer, message: sent.append((peer, message)) or True,
    )
    engine._lwd_send_heartbeat_if_due()
    assert len(sent) == 1
    assert isinstance(sent[0][1], LwdCloudHeartbeat)
    assert sent[0][1].boot_id == "boot-42"
    # Within the interval: no second beat.
    engine._lwd_send_heartbeat_if_due()
    assert len(sent) == 1
