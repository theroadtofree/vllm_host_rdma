# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Liveness layer of the edge engine (peer-restart line, phase 2).

Cloud heartbeats AND cloud business traffic keep per-connection activity
accounts; a silent verdict fires the linked teardown through the
restart-hook registry and freezes the admission gate's snapshot; the next
heartbeat or business frame releases it. A boot-id change is a restart
verdict on its own.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator
from types import MethodType, SimpleNamespace
from typing import Any

import pytest

from vllm.config.lwd.topology import LwdConnectionId
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine import (
    LwdCloudEngineCore,
)
from vllm.v1.lwd_control.control_edge_scheduler import lwd_heartbeat
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
    LwdEdgeEngineCore,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdCloudDpAccount,
    LwdEdgeScheduler,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_heartbeat import (
    LwdHeartbeatMonitor,
)
from vllm.v1.lwd_control.lwd_comm import lwd_restart_hooks as hooks_module
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LwdCloudHeartbeat,
    LwdPrefixReadyNotify,
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


def _scheduler_stub(
    *, edge_share: bool = False, instance_id: int = 0
) -> SimpleNamespace:
    """Scheduler carrying the routing edge-identity fields the gate reads;
    ``_lwd_edge_id_of`` is the REAL scheduler method bound to the stub, so
    the gate and its tests judge edge identity exactly like routing does.
    ``lwd_update_silent_routes`` records the liveness pushes the sweep
    and the beat-recovery mirror into the real scheduler."""
    stub = SimpleNamespace(
        _lwd_edge_share=edge_share,
        _lwd_instance_id=instance_id,
        silent_pushes=[],
    )
    stub._lwd_edge_id_of = MethodType(LwdEdgeScheduler._lwd_edge_id_of, stub)
    stub.lwd_update_silent_routes = lambda silent: stub.silent_pushes.append(silent)
    return stub


def _edge_engine(
    monkeypatch: pytest.MonkeyPatch,
    connections,
    notified: list,
    *,
    edge_share: bool = False,
    instance_id: int = 0,
) -> Any:
    # Dynamic unit-test stub: the real engine needs a full torch/NPU
    # assembly that this CPU-only suite cannot construct.
    engine: Any = object.__new__(LwdEdgeEngineCore)
    monitor = LwdHeartbeatMonitor(10.0, 1.0)
    engine._lwd_heartbeat = monitor
    engine._lwd_clouds = {connection: None for connection in connections}
    for connection in connections:
        monitor.note_adopted(connection)
    # The gate's elastic scale-out branch reads the runtime/serviceable sets;
    # empty = every planned connection is serviceable (the pass-through shape).
    engine._lwd_runtime_connections = set()
    engine._lwd_serviceable_connections = set()
    engine.scheduler = _scheduler_stub(edge_share=edge_share, instance_id=instance_id)
    # Probe-manager stand-in: records the cloud-grained liveness pushes
    # (a fully silent cloud leaves probe targeting until recovery).
    probe = SimpleNamespace(silent_pushes=[])
    probe.set_silent_clouds = lambda ids: probe.silent_pushes.append(ids)
    engine._lwd_probe_manager = probe
    # The monitor is the single event exit point: both consumers subscribe,
    # exactly as the engine wires them at assembly time.
    monitor.subscribe(
        lambda verdict: engine.scheduler.lwd_update_silent_routes(verdict.silent_routes)
    )
    monitor.subscribe(
        lambda verdict: engine._lwd_probe_manager.set_silent_clouds(
            verdict.silent_clouds
        )
    )
    monkeypatch.setattr(
        lwd_heartbeat,
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
    engine._lwd_heartbeat.note_activity(CONNECTION, heartbeat_boot_id="boot-a")
    assert notified == []  # the first boot id is not a restart
    engine._lwd_heartbeat.note_activity(CONNECTION, heartbeat_boot_id="boot-b")
    assert notified == [(0, 0, "boot id changed")]


def test_beat_releases_the_gate_after_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    notified: list = []
    engine = _edge_engine(monkeypatch, [CONNECTION], notified)
    engine._lwd_heartbeat._silent = {CONNECTION}
    engine._lwd_heartbeat.note_activity(CONNECTION, heartbeat_boot_id="boot-a")
    assert engine._lwd_heartbeat.frozen_silent_routes() == frozenset()
    assert notified == []
    # Recovery releases the ROUTING exclusion together with the gate.
    assert engine.scheduler.silent_pushes[-1] == frozenset()
    # ...and the cloud re-enters the probe rotation.
    assert engine._lwd_probe_manager.silent_pushes[-1] == frozenset()


def test_sweep_marks_silent_and_fires_the_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notified: list = []
    engine = _edge_engine(monkeypatch, [CONNECTION], notified)
    engine._lwd_heartbeat._hello_last[CONNECTION] = time.monotonic() - 30.0
    engine._lwd_heartbeat.sweep()
    assert engine._lwd_heartbeat.frozen_silent_routes() == frozenset({CONNECTION})
    assert len(notified) == 1
    assert notified[0][0] == 0 and notified[0][1] == 0
    assert "no cloud activity" in notified[0][2]
    # The frozen verdict is mirrored into the scheduler: routing keeps
    # the dead route out of its candidates until recovery.
    assert engine.scheduler.silent_pushes[-1] == frozenset({CONNECTION})
    # ...and into the probe manager: the fully silent cloud (every
    # adopted route of it is silent) leaves probe targeting.
    assert engine._lwd_probe_manager.silent_pushes[-1] == frozenset({0})
    # The silent transition fires once: a second sweep does not re-notify.
    engine._lwd_heartbeat._last_sweep = 0.0
    engine._lwd_heartbeat.sweep()
    assert len(notified) == 1


def test_probe_targeting_leaves_only_fully_silent_clouds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cloud-grained probe exclusion: a cloud stays probeable while ANY of
    its adopted DP routes is alive (the coordination HTTP listener lives
    with the instance process); only a cloud whose every adopted route is
    silent leaves probe targeting."""
    notified: list = []
    dp0 = LwdConnectionId(0, 0, 0)
    dp1 = LwdConnectionId(0, 0, 1)
    engine = _edge_engine(monkeypatch, [dp0, dp1], notified)
    engine._lwd_heartbeat._hello_last[dp0] = time.monotonic() - 30.0
    engine._lwd_heartbeat.sweep()
    # dp0 silent, dp1 alive: cloud 0 still answers probes.
    assert engine._lwd_probe_manager.silent_pushes[-1] == frozenset()
    engine._lwd_heartbeat._hello_last[dp1] = time.monotonic() - 30.0
    engine._lwd_heartbeat._last_sweep = 0.0
    engine._lwd_heartbeat.sweep()
    # Every adopted route of cloud 0 is silent: probe targeting skips it
    # until activity resumes.
    assert engine._lwd_probe_manager.silent_pushes[-1] == frozenset({0})


def test_business_traffic_keeps_a_cloud_alive_without_heartbeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A heartbeat frame queued behind a busy outbound FIFO must not make a
    healthy cloud look dead: a recent business frame is independent evidence
    that the cloud process is still answering requests."""
    notified: list = []
    engine = _edge_engine(monkeypatch, [CONNECTION], notified)
    engine._lwd_heartbeat._hello_last[CONNECTION] = time.monotonic() - 30.0
    engine._lwd_heartbeat.note_activity(CONNECTION)
    engine._lwd_heartbeat.sweep()
    assert engine._lwd_heartbeat.frozen_silent_routes() == frozenset()
    assert notified == []
    assert (
        engine._lwd_gate_reason(SimpleNamespace(request_id="r", client_index=0)) is None
    )


def test_business_traffic_releases_a_silent_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Traffic after a silent verdict clears the gate and the routing
    exclusion immediately, without waiting for the next heartbeat. It is
    evidence of life, NOT a restart, so no teardown hook fires."""
    notified: list = []
    engine = _edge_engine(monkeypatch, [CONNECTION], notified)
    engine._lwd_heartbeat._silent = {CONNECTION}
    engine._lwd_heartbeat.note_activity(CONNECTION)
    assert engine._lwd_heartbeat.frozen_silent_routes() == frozenset()
    assert engine.scheduler.silent_pushes[-1] == frozenset()
    assert engine._lwd_probe_manager.silent_pushes[-1] == frozenset()
    assert notified == []


def test_stale_heartbeat_and_stale_traffic_still_mark_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The new evidence has the same 10 s horizon as the heartbeat: a cloud
    that went quiet on BOTH channels is still torn down. The activity
    signal widens the evidence, it does not disable liveness."""
    notified: list = []
    engine = _edge_engine(monkeypatch, [CONNECTION], notified)
    old = time.monotonic() - 30.0
    engine._lwd_heartbeat._hello_last[CONNECTION] = old
    engine._lwd_heartbeat._business_last[CONNECTION] = old
    engine._lwd_heartbeat.sweep()
    assert engine._lwd_heartbeat.frozen_silent_routes() == frozenset({CONNECTION})
    assert len(notified) == 1


def test_dispatch_of_a_business_frame_refreshes_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real IO-thread dispatch path must feed the activity ledger —
    otherwise the second evidence channel would exist but never be used."""
    notified: list = []
    engine = _edge_engine(monkeypatch, [CONNECTION], notified)
    engine._lwd_heartbeat._hello_last[CONNECTION] = time.monotonic() - 30.0
    engine.input_queue = _RecordingQueue()
    monkeypatch.setattr(engine, "_lwd_connection_of", lambda peer: CONNECTION)
    peer = SimpleNamespace(instance_id=0, dp_rank=0)
    msg = LwdPrefixReadyNotify(request_id="r", hit_tokens=3, nonce=1)
    assert engine._lwd_dispatch_message(peer, msg) is True
    assert CONNECTION in engine._lwd_heartbeat._business_last
    assert (
        engine._lwd_heartbeat._business_last[CONNECTION]
        > engine._lwd_heartbeat._hello_last[CONNECTION]
    )
    assert len(engine.input_queue.items) == 1
    engine._lwd_heartbeat.sweep()
    assert engine._lwd_heartbeat.frozen_silent_routes() == frozenset()


def test_sweep_is_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    notified: list = []
    engine = _edge_engine(monkeypatch, [CONNECTION], notified)
    engine._lwd_heartbeat._hello_last[CONNECTION] = time.monotonic() - 30.0
    engine._lwd_heartbeat.sweep()
    engine._lwd_heartbeat.sweep()  # within the sweep interval: no-op
    assert len(notified) == 1


def test_routing_excludes_silent_routes_but_fails_open() -> None:
    """A dead cloud must not eat round-robin traffic while its sibling
    serves: the sweep's frozen verdict keeps the silent route out of the
    routing candidates (a preferred pointing at it falls back too). When
    EVERY candidate is silent the filter fails open — the gate owns the
    fully-silent rejection at admission, the transfer containment owns
    the residual admit/route race."""
    alive = LwdConnectionId(1, 0, 0)
    dead = LwdConnectionId(1, 1, 0)
    stub = SimpleNamespace(
        _lwd_connections=(alive, dead),
        _lwd_accounts={cid: LwdCloudDpAccount(cid) for cid in (alive, dead)},
        _lwd_route={},
        _lwd_last_engine=-1,
        _lwd_silent_routes=frozenset({dead}),
    )
    stub._lwd_route_request = MethodType(LwdEdgeScheduler._lwd_route_request, stub)
    # A preferred that points at the silent cloud falls back; plain
    # round-robin never picks it either.
    assert stub._lwd_route_request("r1", 1, dead) == alive
    assert stub._lwd_route_request("r2", 1) == alive
    assert stub._lwd_route_request("r3", 1, None) == alive
    # Fail open: every candidate silent -> still routable (the race arm;
    # the gate has already rejected the steady-state fully-silent case).
    stub._lwd_silent_routes = frozenset({alive, dead})
    assert stub._lwd_route_request("r4", 1) in (alive, dead)
    # And with no silence at all, both routes are candidates again.
    stub._lwd_silent_routes = frozenset()
    routed = {stub._lwd_route_request(f"r{5 + i}", 1) for i in range(8)}
    assert routed == {alive, dead}


def test_gate_rejects_only_when_the_routes_are_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notified: list = []
    engine = _edge_engine(
        monkeypatch, [CONNECTION, CONNECTION_E1], notified, edge_share=True
    )
    # Edge-side ids are BARE: identity comes from client_index (routing's
    # source), never the id string.
    edge0_req = SimpleNamespace(request_id="chatcmpl-x", client_index=0)
    edge1_req = SimpleNamespace(request_id="chatcmpl-y", client_index=1)
    assert engine._lwd_gate_reason(edge0_req) is None
    engine._lwd_heartbeat._silent = {CONNECTION_E1}
    # Whole-route silence rejects everything.
    engine._lwd_heartbeat._silent = {CONNECTION, CONNECTION_E1}
    assert engine._lwd_gate_reason(edge0_req) is not None
    # Partial silence: edge 0 still has a live route.
    engine._lwd_heartbeat._silent = {CONNECTION_E1}
    assert engine._lwd_gate_reason(edge0_req) is None
    # A request whose source edge is fully silent is rejected.
    assert engine._lwd_gate_reason(edge1_req) is not None


def test_gate_uses_live_staleness_before_the_first_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep runs at step head: a cloud that went silent while the
    engine idled must not admit the very request that wakes it."""
    notified: list = []
    engine = _edge_engine(monkeypatch, [CONNECTION], notified)
    engine._lwd_heartbeat._hello_last[CONNECTION] = time.monotonic() - 30.0
    assert (
        engine._lwd_heartbeat.frozen_silent_routes() == frozenset()
    )  # no sweep has run
    request = SimpleNamespace(request_id="chatcmpl-x", client_index=0)
    assert engine._lwd_gate_reason(request) is not None


def test_gate_rejects_when_the_requests_edge_has_no_adopted_cloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Elastic partial start: instance 1's cloud starts later — its
    frontend's request must fail fast at the gate, never reach a routing
    layer that would address the un-adopted connection (the KeyError on
    the shared engine's first drill)."""
    notified: list = []
    # edge_share: edge 0 adopted, edge 1's cloud still loading.
    engine = _edge_engine(monkeypatch, [CONNECTION], notified, edge_share=True)
    reason = engine._lwd_gate_reason(
        SimpleNamespace(request_id="chatcmpl-x", client_index=1)
    )
    assert reason is not None
    assert "not started" in reason
    # The adopted edge still admits.
    assert (
        engine._lwd_gate_reason(
            SimpleNamespace(request_id="chatcmpl-y", client_index=0)
        )
        is None
    )


def test_gate_admits_a_bare_id_on_a_standalone_nonzero_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (cloud_share drill): a standalone edge engine whose
    instance id is NOT 0 serves bare request ids. The gate used to derive
    the edge from the id string, and ``parse_edge_id`` defaults an
    unparsable id to edge 0 — on edge 1 no connection matched and every
    request 500'd with "the planned cloud has not started/registered
    yet" although the cloud was fully adopted."""
    notified: list = []
    engine = _edge_engine(
        monkeypatch, [LwdConnectionId(1, 0, 0)], notified, instance_id=1
    )
    request = SimpleNamespace(request_id="chatcmpl-814f54ce", client_index=0)
    assert engine._lwd_gate_reason(request) is None


def test_routing_to_an_unadopted_cloud_fails_the_request_not_the_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-cloud residual: when routing still scores an un-adopted
    sibling, the admission is cancelled and the request aborted (fast,
    retryable) — the notify must never address a cloud the engine cannot
    send to."""
    from types import SimpleNamespace

    notified: list = []
    engine = _edge_engine(monkeypatch, [CONNECTION_E1], notified)
    engine.aborts_queue = _RecordingQueue()
    cancelled: list = []
    unadopted = LwdConnectionId(1, 2, 0)
    engine.scheduler = SimpleNamespace(
        _lwd_edge_id_of=lambda request: 1,
        lwd_admit_request=lambda request, preferred: unadopted,
        lwd_edge_cancel_admission=cancelled.append,
    )
    engine._lwd_admit_and_notify(
        SimpleNamespace(request_id="chatcmpl-x", client_index=0)
    )
    assert cancelled == ["chatcmpl-x"]
    assert engine.aborts_queue.items == [["chatcmpl-x"]]


def test_send_to_an_unadopted_cloud_drops_instead_of_raising() -> None:
    """Last line of defense: a message still landing on an un-adopted
    connection drops with a log — the KeyError killed the shared engine
    (instance 0's requests went down with it)."""
    from types import SimpleNamespace

    engine: Any = object.__new__(LwdEdgeEngineCore)
    engine._lwd_clouds = {}
    engine._lwd_send_to_cloud(CONNECTION_E1, SimpleNamespace())


def test_gate_rejection_uses_the_abort_channel_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (duplicate request id): the gate must NOT enqueue the
    request itself — the caller enqueues unconditionally after admission,
    so the rejection goes through aborts_queue alone (drained at the next
    step head, after the single enqueue entered the scheduler)."""
    from types import SimpleNamespace

    engine = _edge_engine(monkeypatch, [CONNECTION], [])
    engine._lwd_heartbeat._silent = {CONNECTION}
    engine.aborts_queue = _RecordingQueue()
    engine.input_queue = _RecordingQueue()
    request = SimpleNamespace(request_id="chatcmpl-x", client_index=0)
    engine._lwd_admit_and_notify(request, None)
    assert engine.aborts_queue.items == [["chatcmpl-x"]]
    assert engine.input_queue.items == []  # the caller owns the single enqueue


def test_entry_gate_rejects_before_the_probe_machinery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate is the FIRST thing a request meets (old-branch invariant):
    with the route silent, add_request finishes the request directly
    (client-visible ERROR finish — a request failure, not a cancellation
    — nothing enqueues afterwards) and never touches the probe machinery
    — a probe sent to the dead cloud would hang for its full timeout
    instead of rejecting."""
    from types import SimpleNamespace

    notified: list = []
    engine = _edge_engine(monkeypatch, [CONNECTION], notified)
    engine._lwd_heartbeat._silent = {CONNECTION}
    errored: list = []
    monkeypatch.setattr(
        engine,
        "_send_error_outputs_to_client",
        lambda ids, client_index: errored.append((ids, client_index)),
    )
    engine.scheduler = SimpleNamespace(
        _lwd_edge_id_of=lambda request: 0,
        lwd_edge_request_active=lambda request_id: False,
    )
    engine._lwd_probe_manager = SimpleNamespace(
        submit=lambda request: pytest.fail("probe machinery must not be touched")
    )
    request = SimpleNamespace(
        request_id="chatcmpl-x",
        client_index=0,
        num_prompt_tokens=4,
        abort_immediately=False,
    )
    engine.add_request(request)
    assert errored == [(["chatcmpl-x"], 0)]


def test_abort_skips_the_cloud_notify_for_silent_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An abort notify to a silent cloud cannot arrive and would stall the
    abort behind send retries: it is skipped, the live cloud still gets
    its notify."""
    from types import SimpleNamespace

    from vllm.v1.engine.core import EngineCoreProc

    monkeypatch.setattr(EngineCoreProc, "abort_requests", lambda self, ids: None)
    engine: Any = object.__new__(LwdEdgeEngineCore)
    engine._lwd_probe_manager = SimpleNamespace(abort=lambda request_id: None)
    engine._lwd_probe_chains = {}
    engine._lwd_heartbeat = LwdHeartbeatMonitor(10.0, 1.0)
    engine._lwd_heartbeat._silent = {CONNECTION}
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

    engine: Any = object.__new__(LwdCloudEngineCore)
    beats: list = []
    monkeypatch.setattr(engine, "_lwd_send_heartbeat_if_due", lambda: beats.append(1))
    engine._handle_client_request(
        EngineCoreRequestType.WAKEUP, cloud_module._LWD_HEARTBEAT_TICK
    )
    assert beats == [1]


def test_cloud_engine_sends_one_heartbeat_per_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine: Any = object.__new__(LwdCloudEngineCore)
    sent: list = []
    engine._lwd_edges = ["peer-a"]
    engine._lwd_boot_id = "boot-42"
    engine._lwd_hello_interval_s = 1000.0
    engine._lwd_last_heartbeat_at = 0.0

    def send(peer: str, message: object) -> bool:
        sent.append((peer, message))
        return True

    monkeypatch.setattr(engine, "_lwd_send", send)
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
    engine: Any = object.__new__(LwdCloudEngineCore)
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

    engine: Any = object.__new__(LwdCloudEngineCore)
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

    engine: Any = object.__new__(LwdEdgeEngineCore)
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

    engine: Any = object.__new__(LwdCloudEngineCore)
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


def test_beat_survives_the_mid_assembly_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression (sixth incident): the control plane's IO thread dispatches
    into the engine from the moment the plane starts, but assembly may still
    be waiting on the handshake or the cloud params — a beat arriving in
    that window must land in the (pre-seeded, empty) accounts instead of
    raising AttributeError, whose _lwd_fail side effect queued an engine
    failure."""
    notified: list = []
    engine = _edge_engine(monkeypatch, [], notified)  # no connections seeded
    # A beat for a connection the handshake has not even discovered yet.
    engine._lwd_heartbeat.note_activity(CONNECTION, heartbeat_boot_id="boot-a")
    assert engine._lwd_heartbeat._hello_last[CONNECTION] > 0
    assert engine._lwd_heartbeat._boot[CONNECTION] == "boot-a"
    # The gate does not blow up mid-assembly: with nothing adopted yet it
    # returns the elastic no-adoption reason (a clean rejection), never an
    # exception off the half-assembled state.
    request = SimpleNamespace(request_id="chatcmpl-x", client_index=0)
    assert "not started" in engine._lwd_gate_reason(request)


def test_gate_iteration_survives_concurrent_adoption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control-plane IO thread adopts (inserts into _lwd_clouds)
    while the engine thread gates: every engine-thread read of the dict
    snapshots first — tuple(dict) completes in C, a Python-level
    iteration raises 'dictionary changed size during iteration'
    (the pre-fix shape that would kill the shared engine per request).
    Stress-shaped but one-sided: it can only fail while the race
    exists, never spuriously."""
    notified: list = []
    engine = _edge_engine(monkeypatch, [CONNECTION], notified)
    request = SimpleNamespace(request_id="chatcmpl-x", client_index=0)
    stop = threading.Event()

    def adopter() -> None:
        while not stop.is_set():
            engine._lwd_clouds[CONNECTION_E1] = None
            del engine._lwd_clouds[CONNECTION_E1]

    saved_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # force the interleave the race needs
    try:
        thread = threading.Thread(target=adopter, daemon=True)
        thread.start()
        for _ in range(2000):
            engine._lwd_gate_reason(request)
        stop.set()
        thread.join(timeout=5.0)
        assert not thread.is_alive()
    finally:
        sys.setswitchinterval(saved_interval)
