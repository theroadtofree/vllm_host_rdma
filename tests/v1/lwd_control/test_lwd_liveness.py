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


class _RecordingQueue:
    def __init__(self) -> None:
        self.items: list = []

    def put_nowait(self, item) -> None:  # noqa: ANN001
        self.items.append(item)


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


def test_gate_uses_live_staleness_before_the_first_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep runs at step head: a cloud that went silent while the
    engine idled must not admit the very request that wakes it."""
    notified: list = []
    engine = _edge_engine(monkeypatch, [CONNECTION], notified)
    engine._lwd_hello_last[CONNECTION] = time.monotonic() - 30.0
    assert engine._lwd_silent_routes == frozenset()  # no sweep has run
    assert engine._lwd_gate_reason("plain-req") is not None


def test_gate_rejection_uses_the_abort_channel_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (duplicate request id): the gate must NOT enqueue the
    request itself — the caller enqueues unconditionally after admission,
    so the rejection goes through aborts_queue alone (drained at the next
    step head, after the single enqueue entered the scheduler)."""
    from types import SimpleNamespace

    engine = _edge_engine(monkeypatch, [CONNECTION], [])
    engine._lwd_silent_routes = frozenset({CONNECTION})
    engine.aborts_queue = _RecordingQueue()
    engine.input_queue = _RecordingQueue()
    request = SimpleNamespace(request_id="plain-req", client_index=0)
    engine._lwd_admit_and_notify(request, None)
    assert engine.aborts_queue.items == [["plain-req"]]
    assert engine.input_queue.items == []  # the caller owns the single enqueue


def test_abort_skips_the_cloud_notify_for_silent_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An abort notify to a silent cloud cannot arrive and would stall the
    abort behind send retries: it is skipped, the live cloud still gets
    its notify."""
    from types import SimpleNamespace

    from vllm.v1.engine.core import EngineCoreProc

    monkeypatch.setattr(EngineCoreProc, "abort_requests", lambda self, ids: None)
    engine = object.__new__(LwdEdgeEngineCore)
    engine._lwd_probe_manager = SimpleNamespace(abort=lambda request_id: None)
    engine._lwd_probe_chains = {}
    engine._lwd_silent_routes = frozenset({CONNECTION})
    sent: list = []
    engine.scheduler = SimpleNamespace(
        lwd_edge_abort=lambda ids: [
            (CONNECTION, "e0-r1"),
            (CONNECTION_E1, "e0-r2"),
        ],
        requests={"e0-r2": SimpleNamespace(lwd_prefix_nonce=0)},
    )
    monkeypatch.setattr(
        engine, "_lwd_send_to_cloud", lambda dp_idx, message: sent.append(dp_idx)
    )
    engine.abort_requests(["e0-r1", "e0-r2"])
    assert sent == [CONNECTION_E1]


def test_registry_fires_every_hook_and_survives_exceptions() -> None:
    fired: list = []
    hooks_module.register_cloud_restart_hook(
        lambda cloud_id, dp_idx, reason: fired.append(("ok", cloud_id))
    )
    hooks_module.register_cloud_restart_hook(_raising_hook)
    hooks_module.notify_cloud_restarted(1, 0, "test reason")
    assert fired == [("ok", 1)]


def test_heartbeat_tick_dispatches_the_beat_to_the_engine_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The heartbeat thread only touches the input queue (the control
    sender is bound to the engine thread); the tick makes the engine
    thread send the beat."""
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.lwd_control.control_cloud_scheduler import (
        lwd_cloud_engine as cloud_module,
    )

    engine = object.__new__(LwdCloudEngineCore)
    beats: list = []
    monkeypatch.setattr(engine, "_lwd_send_heartbeat_if_due", lambda: beats.append(1))
    engine._handle_client_request(
        EngineCoreRequestType.WAKEUP, cloud_module._LWD_HEARTBEAT_TICK
    )
    assert beats == [1]


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


def test_heartbeat_prunes_an_unreachable_edge_instead_of_dying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (cloud died with the edge): the plane unregisters a gone
    edge; the next beat must skip it — a liveness probe that crashes on a
    dead probed peer kills the prober."""
    from vllm.v1.lwd_control.lwd_comm.lwd_control_plane import (
        LwdPeerNotRegisteredError,
    )

    gone, live = "peer-gone", "peer-live"
    engine = object.__new__(LwdCloudEngineCore)
    engine._lwd_edges = {gone, live}
    engine._lwd_boot_id = "boot-42"
    engine._lwd_hello_interval_s = 1000.0
    engine._lwd_last_heartbeat_at = 0.0

    def send(peer, message):  # noqa: ANN001
        if peer == gone:
            raise LwdPeerNotRegisteredError(f"{peer} has not completed the handshake")
        return True

    monkeypatch.setattr(engine, "_lwd_send", send)
    engine._lwd_send_heartbeat_if_due()
    # The unreachable edge is pruned; the live one keeps its beat.
    assert engine._lwd_edges == {live}


def test_cloud_send_until_queued_drops_for_an_unreachable_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Business messages to a gone edge drop (the plane already drops its
    payloads) instead of failing the engine."""
    from vllm.v1.engine.core import EngineShutdownState
    from vllm.v1.lwd_control.lwd_comm.lwd_control_plane import (
        LwdPeerNotRegisteredError,
    )
    from vllm.v1.lwd_control.lwd_comm.lwd_messages import LwdAbortNotify

    engine = object.__new__(LwdCloudEngineCore)
    engine.shutdown_state = EngineShutdownState.RUNNING

    def send(peer, message):  # noqa: ANN001
        raise LwdPeerNotRegisteredError(f"{peer} has not completed the handshake")

    monkeypatch.setattr(engine, "_lwd_send", send)
    # Must not raise; the message is dropped for the unreachable peer.
    engine._lwd_send_until_queued("peer-gone", LwdAbortNotify("req-1", 0))


def test_edge_send_to_cloud_drops_for_an_unreachable_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The edge's business send follows the same drop-until-reachable
    semantic: the request fails through the data-plane containment path,
    not through an engine death."""
    from vllm.v1.lwd_control.lwd_comm.lwd_control_plane import (
        LwdPeerNotRegisteredError,
    )
    from vllm.v1.lwd_control.lwd_comm.lwd_messages import LwdAbortNotify

    engine = object.__new__(LwdEdgeEngineCore)
    engine._lwd_clouds = {CONNECTION: "peer-gone"}

    def send(peer, message):  # noqa: ANN001
        raise LwdPeerNotRegisteredError(f"{peer} has not completed the handshake")

    monkeypatch.setattr(engine, "_lwd_send", send)
    engine._lwd_send_to_cloud(CONNECTION, LwdAbortNotify("req-1", 0))  # no raise


def test_edge_reregistration_resets_the_connection_seqno_books(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (fifth incident): a restarted edge's seqno stream restarts
    at 0 while _lwd_latest_seqnos kept the old incarnation's high-water
    mark — the new stream's first RangeNotify was dropped as a duplicate
    and the request hung with its UP frame already sent."""
    from types import SimpleNamespace

    from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdPeer

    engine = object.__new__(LwdCloudEngineCore)
    engine._lwd_edges = set()
    engine._lwd_cloud_params = SimpleNamespace()
    engine._lwd_latest_seqnos = {CONNECTION: 3}
    engine._lwd_pending_seqnos = {CONNECTION: {3: {"old-req"}}}
    engine.lwd_config = SimpleNamespace(instance_id=0)
    sent: list = []
    monkeypatch.setattr(
        engine, "_lwd_send_until_queued", lambda peer, message: sent.append(peer)
    )
    engine._lwd_handle_peer_registered(LwdPeer(instance_id=0, dp_rank=0))
    # The old incarnation's seqno books are gone; the new stream's
    # seqno=0 passes the duplicate gate (0 > -1).
    assert engine._lwd_latest_seqnos == {}
    assert engine._lwd_pending_seqnos == {}
    assert len(sent) == 1  # scheduling limits still go out first
