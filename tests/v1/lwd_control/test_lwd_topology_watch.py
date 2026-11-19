# SPDX-License-Identifier: Apache-2.0
"""Phase A of elastic scale-out: the shared edge engine's topology-file
watch (detect only).

The poll thread owns nothing but the stat and the input-queue tick; the
engine thread (the single writer) re-reads, validates and diffs. A
pure-addition update is detected; an illegal update is rejected with a
reason that names the field; unchanged content (mtime touches included)
is a no-op — and none of the three ever stops serving.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from vllm.config.lwd import LwdTopology, LwdTopologyUpdateRejected
from vllm.config.lwd.topology import LwdConnectionId
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.lwd_control.control_cloud_scheduler import (
    lwd_cloud_engine as lwd_cloud_engine_module,
)
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine import (
    LwdCloudEngineCore,
)
from vllm.v1.lwd_control.control_edge_scheduler import lwd_edge_engine
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
    LwdEdgeEngineCore,
)
from vllm.v1.lwd_control.control_scheduler import lwd_base_engine
from vllm.v1.lwd_control.control_scheduler.lwd_base_engine import LwdBaseEngineCore


class _RecordingPlane:
    """Stands in for the edge control plane: records add_link calls."""

    def __init__(self) -> None:
        self.links: list[tuple[str, LwdConnectionId, int]] = []

    def add_link(
        self,
        endpoint: str,
        connection_id: LwdConnectionId,
        cloud_dp_size: int,
        *,
        done: threading.Event | None = None,
    ) -> None:
        self.links.append((endpoint, connection_id, cloud_dp_size))


class _RecordingScheduler:
    """Stands in for the scheduler: records runtime-connection seeds and
    arming."""

    def __init__(self) -> None:
        self.connections: list[LwdConnectionId] = []
        self.armed: list[LwdConnectionId] = []

    def _lwd_edge_id_of(self, request: Any) -> int:
        """The tests' identity convention: the ``e<id>-`` prefix names the
        source edge, a bare id means edge 0 (the REAL scheduler derives
        this from the deployment shape; here it just mirrors the ids the
        tests craft)."""
        prefix = request.request_id.partition("-")[0]
        if prefix.startswith("e") and prefix[1:].isdigit():
            return int(prefix[1:])
        return 0

    def lwd_add_runtime_connection(self, connection_id: LwdConnectionId) -> None:
        self.connections.append(connection_id)

    def lwd_arm_runtime_connection(self, connection_id: LwdConnectionId) -> None:
        self.armed.append(connection_id)


def _payload(instances: int) -> dict[str, Any]:
    """An N-pair edge_share host-comm topology."""
    return {
        "deployment": {
            "mode": 0,
            "scene": "edge_share",
            "edges_num": instances,
            "clouds_num": instances,
            "data_plane_comm": "host",
        },
        "feature_ctrl": {
            "enable_early_recv": False,
            "scramble": {"enable": False, "device_id": None, "custom_factor": None},
        },
        "edges": [
            {"id": i, "dp": [{"dp_idx": 0, "addr": "135.150.2.2"}]}
            for i in range(instances)
        ],
        "clouds": [
            {
                "id": i,
                "dp": [
                    {
                        "dp_idx": 0,
                        "addr": "16.32.2.2",
                        "ctrl_port": 5550 + i,
                        "data_port": 5560 + i,
                    }
                ],
            }
            for i in range(instances)
        ],
        "instance_links": [{"edge": i, "cloud": i} for i in range(instances)],
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    """Atomic replace: the deployment discipline the watcher expects."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=False))
    os.replace(tmp, path)


def _engine(path: Path) -> LwdEdgeEngineCore:
    engine = object.__new__(LwdEdgeEngineCore)
    engine.lwd_config = SimpleNamespace(
        path=str(path), topology=LwdTopology.from_dict(_payload(1))
    )
    engine._lwd_topology_file_digest = engine.lwd_config.topology.digest
    engine._lwd_control_plane = _RecordingPlane()
    engine.scheduler = _RecordingScheduler()
    engine._lwd_runtime_connections = {}
    engine._lwd_armed_connections = set()
    return engine


def test_an_added_pair_is_detected(tmp_path: Path) -> None:
    topology_file = tmp_path / "topology.yaml"
    _write(topology_file, _payload(1))
    engine = _engine(topology_file)
    _write(topology_file, _payload(2))
    verdict = engine._lwd_process_topology_reload()
    assert verdict is not None
    assert not isinstance(verdict, LwdTopologyUpdateRejected)
    assert verdict.new_edge_ids == (1,)
    assert verdict.new_cloud_ids == (1,)
    assert len(verdict.new_connections) == 1
    # Phase B applied the control-plane half: one queued link with the
    # cloud's endpoint and DP size, the runtime registry seeded, the
    # scheduler bookkeeping created.
    assert engine._lwd_control_plane.links == [
        ("tcp://16.32.2.2:5551", LwdConnectionId(1, 1, 0), 1)
    ]
    assert (1, 0) in engine._lwd_runtime_connections
    assert engine.scheduler.connections == [LwdConnectionId(1, 1, 0)]


def test_applying_the_same_increment_twice_adds_each_link_once(
    tmp_path: Path,
) -> None:
    """A re-detected increment (the realized snapshot never advances in
    phase B) replays silently: the registry skips the connection."""
    topology_file = tmp_path / "topology.yaml"
    _write(topology_file, _payload(1))
    engine = _engine(topology_file)
    _write(topology_file, _payload(2))
    first = engine._lwd_process_topology_reload()
    assert first is not None and not isinstance(first, LwdTopologyUpdateRejected)
    engine._lwd_apply_topology_increment(first)
    assert len(engine._lwd_control_plane.links) == 1


def test_a_refused_data_plane_arms_on_the_next_increment(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The phase C retry semantics: a refused data-plane build keeps the
    connection control-plane only; the next detected increment (a replay
    of the same one) retries ONLY the data-plane half — no duplicate
    control-plane add — and, on success, arms the connection for
    service."""
    build_results: list[bool] = []

    def fake_notify(connection: Any) -> bool:
        return build_results.pop() if build_results else False

    monkeypatch.setattr(lwd_edge_engine, "notify_data_plane_added", fake_notify)
    topology_file = tmp_path / "topology.yaml"
    _write(topology_file, _payload(1))
    engine = _engine(topology_file)
    _write(topology_file, _payload(2))
    first = engine._lwd_process_topology_reload()
    assert first is not None and not isinstance(first, LwdTopologyUpdateRejected)
    # The refusal kept the connection control-plane only: unarmed, never a
    # routing candidate, and the gate rejects with the real reason.
    assert (1, 0) in engine._lwd_runtime_connections
    assert (1, 0) not in engine._lwd_armed_connections
    assert engine.scheduler.armed == []
    assert (
        engine._lwd_gate_reason(SimpleNamespace(request_id="e1-wrapped-req"))
        is not None
    )
    # The next detected increment replays the same verdict: the retry
    # touches only the data-plane half.
    build_results.append(True)
    engine._lwd_apply_topology_increment(first)
    assert engine._lwd_control_plane.links == [
        ("tcp://16.32.2.2:5551", LwdConnectionId(1, 1, 0), 1)
    ]  # no duplicate control-plane add
    assert engine.scheduler.connections == [LwdConnectionId(1, 1, 0)]
    assert (1, 0) in engine._lwd_armed_connections
    assert engine.scheduler.armed == [LwdConnectionId(1, 1, 0)]
    # An armed runtime connection is servable: the gate admits it.
    engine._lwd_hello_lock = threading.Lock()
    engine._lwd_hello_last = {LwdConnectionId(1, 1, 0): time.monotonic()}
    engine._lwd_hello_boot = {}
    engine._lwd_silent = set()
    engine._lwd_silent_routes = frozenset()
    engine._lwd_hello_silent_s = 10.0
    engine._lwd_clouds = {LwdConnectionId(1, 1, 0): None}
    assert engine._lwd_gate_reason(SimpleNamespace(request_id="e1-wrapped-req")) is None


def test_an_illegal_update_is_rejected_without_repeats(tmp_path: Path) -> None:
    topology_file = tmp_path / "topology.yaml"
    _write(topology_file, _payload(1))
    engine = _engine(topology_file)
    illegal = _payload(2)
    illegal["clouds"][0]["dp"][0]["ctrl_port"] = 5599
    _write(topology_file, illegal)
    verdict = engine._lwd_process_topology_reload()
    assert isinstance(verdict, LwdTopologyUpdateRejected)
    assert "existing instance modified" in verdict.reason
    # Same content again (even with a fresh mtime) stays silent: the
    # last-seen file digest suppresses repeat verdicts.
    _write(topology_file, illegal)
    assert engine._lwd_process_topology_reload() is None


def test_unchanged_content_is_a_no_op(tmp_path: Path) -> None:
    topology_file = tmp_path / "topology.yaml"
    _write(topology_file, _payload(1))
    engine = _engine(topology_file)
    _write(topology_file, _payload(1))  # mtime moves, semantics do not
    assert engine._lwd_process_topology_reload() is None


def test_a_corrupt_file_is_rejected_and_recovers(tmp_path: Path) -> None:
    topology_file = tmp_path / "topology.yaml"
    _write(topology_file, _payload(1))
    engine = _engine(topology_file)
    topology_file.write_text("deployment: [not a mapping")
    verdict = engine._lwd_process_topology_reload()
    assert isinstance(verdict, LwdTopologyUpdateRejected)
    assert "invalid file" in verdict.reason
    _write(topology_file, _payload(2))
    recovered = engine._lwd_process_topology_reload()
    assert recovered is not None
    assert not isinstance(recovered, LwdTopologyUpdateRejected)
    assert recovered.new_edge_ids == (1,)


def test_the_watch_thread_queues_one_tick_per_change(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The watcher only stats and ticks: one input-queue WAKEUP with the
    reload sentinel per observed mtime change, nothing else."""

    class _Queue:
        def __init__(self) -> None:
            self.items: list[Any] = []

        def put_nowait(self, item: Any) -> None:
            self.items.append(item)

    monkeypatch.setattr(lwd_edge_engine, "_LWD_TOPOLOGY_POLL_INTERVAL_S", 0.02)
    topology_file = tmp_path / "topology.yaml"
    _write(topology_file, _payload(1))
    engine = _engine(topology_file)
    engine.input_queue = _Queue()
    engine._lwd_topology_watch_stop = threading.Event()
    engine._lwd_start_topology_watch()
    try:
        _write(topology_file, _payload(2))
        deadline = time.monotonic() + 2.0
        while not engine.input_queue.items:
            assert time.monotonic() < deadline, "watch thread never ticked"
            time.sleep(0.01)
    finally:
        engine._lwd_topology_watch_stop.set()
    (request_type, tick) = engine.input_queue.items[0]
    assert request_type == EngineCoreRequestType.WAKEUP
    assert tick is lwd_base_engine._LWD_TOPOLOGY_RELOAD_TICK


def test_the_gate_rejects_a_control_plane_only_connection(
    tmp_path: Path,
) -> None:
    """Phase B semantics: a runtime-added connection is adopted on the
    control plane but carries no data-plane link yet — the admission
    gate rejects it with the real reason instead of letting routing
    score a link that cannot carry traffic. A planned, adopted, healthy
    connection still admits."""
    topology_file = tmp_path / "topology.yaml"
    _write(topology_file, _payload(1))
    engine = _engine(topology_file)
    engine._lwd_hello_lock = threading.Lock()
    engine._lwd_hello_last = {LwdConnectionId(1, 1, 0): time.monotonic()}
    engine._lwd_hello_boot = {}
    engine._lwd_silent = set()
    engine._lwd_silent_routes = frozenset()
    engine._lwd_hello_silent_s = 10.0
    engine._lwd_clouds = {LwdConnectionId(1, 1, 0): None}
    engine._lwd_runtime_connections = {(1, 0): None}
    reason = engine._lwd_gate_reason(SimpleNamespace(request_id="e1-wrapped-req"))
    assert reason is not None
    assert "control plane only" in reason
    # The planned connection is not runtime: a fresh heartbeat admits it.
    engine._lwd_clouds = {LwdConnectionId(0, 0, 0): None}
    assert engine._lwd_gate_reason(SimpleNamespace(request_id="plain-req")) is None


def test_the_data_plane_hook_registry_reports_and_survives_raising_hooks(
    tmp_path: Path,
) -> None:
    """The registry contract: an empty registry refuses (a connection
    with no assembler stays control-plane only), a raising hook is
    logged and never breaks the caller, and a working hook reports the
    build."""
    from vllm.v1.lwd_control.lwd_comm import (
        lwd_data_plane_hooks as hooks,
    )

    topology_file = tmp_path / "topology.yaml"
    _write(topology_file, _payload(1))
    engine = _engine(topology_file)
    _write(topology_file, _payload(2))
    verdict = engine._lwd_process_topology_reload()
    assert verdict is not None and not isinstance(verdict, LwdTopologyUpdateRejected)
    # The test process registered no assembler: the notify refuses and the
    # connection stays control-plane only (the engine logs the refusal).
    assert (1, 0) not in engine._lwd_armed_connections

    saved = list(hooks._hooks)
    try:

        def boom(connection: Any) -> bool:
            raise RuntimeError("assembler boom")

        built: list = []

        def works(connection: Any) -> bool:
            built.append(connection.connection_id)
            return True

        hooks.register_data_plane_add_hook(boom)
        hooks.register_data_plane_add_hook(works)
        (connection,) = verdict.new_connections
        assert hooks.notify_data_plane_added(connection) is True
        assert built == [LwdConnectionId(1, 1, 0)]
    finally:
        hooks._hooks[:] = saved


def test_the_watch_gate_watches_only_the_shared_side_of_the_scene(
    tmp_path: Path,
) -> None:
    """The shared side watches its plan: the edge engine in edge_share
    (clouds join the running engine), the cloud engine in cloud_share
    (edges join the running cloud). The other side always starts fresh
    with the grown yaml and never watches; device comm never watches."""
    topology_file = tmp_path / "topology.yaml"
    _write(topology_file, _payload(1))

    def engine(role: str, scene: str, *, host_comm: bool) -> LwdBaseEngineCore:
        engine = object.__new__(LwdBaseEngineCore)
        engine.lwd_config = SimpleNamespace(
            path=str(topology_file),
            role=role,
            topology=SimpleNamespace(
                is_host_comm=host_comm,
                deployment=SimpleNamespace(scene=scene),
                digest="d",
            ),
        )
        return engine

    def gated(role: str, scene: str, *, host_comm: bool) -> bool:
        return engine(role, scene, host_comm=host_comm)._lwd_topology_watch_enabled()

    assert gated("edge", "edge_share", host_comm=True)
    assert gated("cloud", "cloud_share", host_comm=True)
    # The fresh side of each scene, the flipped scene, and device comm.
    assert not gated("cloud", "edge_share", host_comm=True)
    assert not gated("edge", "cloud_share", host_comm=True)
    assert not gated("edge", "single_instance", host_comm=True)
    assert not gated("edge", "edge_share", host_comm=False)
    assert not gated("cloud", "cloud_share", host_comm=False)


def test_the_cloud_engine_admits_edges_and_builds_the_data_plane(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """C-2/C-3: the cloud's apply hook grows the handshake allowlist per
    new edge AND calls the data-plane build for each new connection.
    A recording plane and a monkeypatched data-plane hook stand in — the
    real handshake accept/replay semantics live in
    test_lwd_handshake_allow_edge, the real link build in the ascend
    suite."""

    class _RecordingCloudPlane:
        def __init__(self) -> None:
            self.allowed: list[int] = []

        def allow_edge(self, edge_id: int) -> None:
            self.allowed.append(edge_id)

    built: list = []

    def fake_notify(connection: Any) -> bool:
        built.append(connection.connection_id)
        return True

    monkeypatch.setattr(lwd_cloud_engine_module, "notify_data_plane_added", fake_notify)
    topology_file = tmp_path / "cloud_topology.yaml"
    _cloud_share_write(topology_file, 1)
    engine = object.__new__(LwdCloudEngineCore)
    engine._lwd_control_plane = _RecordingCloudPlane()
    _cloud_share_write(topology_file, 2)
    verdict = engine._lwd_process_topology_reload()
    assert verdict is not None and not isinstance(verdict, LwdTopologyUpdateRejected)
    assert verdict.new_edge_ids == (1,)
    assert engine._lwd_control_plane.allowed == [1]
    # The data-plane hook fired for the new connection.
    assert built == [LwdConnectionId(1, 0, 0)]
    # A replay of the same increment: allow_edge fires again (idempotent),
    # and the data-plane hook fires again (the ascend builder is
    # idempotent on the connection's key).
    engine._lwd_apply_topology_increment(verdict)
    assert engine._lwd_control_plane.allowed == [1, 1]
    assert built == [LwdConnectionId(1, 0, 0), LwdConnectionId(1, 0, 0)]


def _cloud_share_write(path: Path, edges: int) -> None:
    """A cloud_share topology on the cloud host's copy of the file: N
    edges, one cloud, full mesh."""
    payload = {
        "deployment": {
            "mode": 0,
            "scene": "cloud_share",
            "edges_num": edges,
            "clouds_num": 1,
            "data_plane_comm": "host",
        },
        "feature_ctrl": {
            "enable_early_recv": False,
            "scramble": {"enable": False, "device_id": None, "custom_factor": None},
        },
        "edges": [
            {
                "id": i,
                "coordination": {"enabled": True},
                "dp": [{"dp_idx": 0, "addr": "135.150.2.2"}],
            }
            for i in range(edges)
        ],
        "clouds": [
            {
                "id": 0,
                "coordination": {"enabled": True},
                "dp": [
                    {
                        "dp_idx": 0,
                        "addr": "16.32.2.2",
                        "ctrl_port": 5550,
                        "data_port": 5560,
                    }
                ],
            }
        ],
        "instance_links": [{"edge": i, "cloud": 0} for i in range(edges)],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
