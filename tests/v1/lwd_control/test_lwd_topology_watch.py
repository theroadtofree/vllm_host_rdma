# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from vllm.config.lwd import LwdTopology, LwdTopologyUpdateRejected
from vllm.config.lwd.topology import LwdConnectionId
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine import (
    LwdCloudEngineCore,
)
from vllm.v1.lwd_control.control_edge_scheduler import lwd_edge_engine
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
    LwdEdgeEngineCore,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_heartbeat import (
    LwdHeartbeatMonitor,
)
from vllm.v1.lwd_control.control_scheduler import lwd_base_engine
from vllm.v1.lwd_control.control_scheduler.lwd_base_engine import LwdBaseEngineCore
from vllm.v1.lwd_control.lwd_comm.lwd_control_plane import (
    LwdCloudControlPlane,
    LwdEdgeControlPlane,
)


class _RecordingPlane(LwdEdgeControlPlane):
    """Stands in for the edge control plane: records add_link calls. A
    subclass (with no plane construction) because the engine's apply hook
    asserts the control-plane type before touching it."""

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


class _RecordingProbeManager:
    """Stands in for the probe manager: records runtime connection
    registrations (the probe-targeting half of elastic scale-out). The
    scale API is connection-grained: register_connection takes the
    LwdConnectionId (cloud_id, dp_idx included) and the coordination
    URL, and the health rotation widens with (cloud_id, dp_idx) targets."""

    def __init__(self) -> None:
        self.registered: list[tuple[LwdConnectionId, str]] = []

    def register_connection(self, connection_id: LwdConnectionId, url: str) -> None:
        self.registered.append((connection_id, url))

    def rotation(self) -> tuple[tuple[int, int], ...]:
        # Mirrors LwdCloudHealthTracker.targets: (cloud_id, dp_idx) pairs.
        return tuple(
            (connection_id.cloud_id, connection_id.dp_idx)
            for connection_id, _ in self.registered
        )


class _RecordingScheduler:
    """Stands in for the scheduler: records runtime-connection seeds and
    opening for service."""

    def __init__(self) -> None:
        self.connections: list[LwdConnectionId] = []
        self.serviceable: list[LwdConnectionId] = []

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

    def lwd_add_runtime_route(self, connection_id: LwdConnectionId) -> None:
        self.serviceable.append(connection_id)


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


def _cluster_payload(edges: int, clouds: int) -> dict[str, Any]:
    """An E×C lwd_cluster host-comm topology. The scene requires an
    enabled coordination block on every instance, and every cloud DP
    carries its coordination endpoint itself (coord_port, with coord_url
    derived from it at parse time) — so the increment contract keeps
    existing instances byte-identical and a new cloud's endpoint arrives
    attached to the new cloud's own DP block, never as an edit to an
    edge."""
    return {
        "deployment": {
            "mode": 0,
            "scene": "lwd_cluster",
            "edges_num": edges,
            "clouds_num": clouds,
            "data_plane_comm": "host",
        },
        "feature_ctrl": {
            "enable_early_recv": False,
            "scramble": {"enable": False, "device_id": None, "custom_factor": None},
        },
        "edges": [
            {
                "id": e,
                "dp": [{"dp_idx": 0, "addr": "135.150.2.2"}],
                "coordination": {
                    "enabled": True,
                    "tenant_key_file": "/run/secrets/edge-cloud-tenant-key",
                    "connect_timeout": 5.0,
                },
            }
            for e in range(edges)
        ],
        "clouds": [
            {
                "id": c,
                "dp": [
                    {
                        "dp_idx": 0,
                        "addr": f"16.32.10.1{c}",
                        "ctrl_port": 5550 + c,
                        "data_port": 5560 + c,
                        "coord_port": 8100 + c,
                    }
                ],
                "coordination": {"enabled": True, "listen_host": "0.0.0.0"},
            }
            for c in range(clouds)
        ],
        "instance_links": [
            {"edge": e, "cloud": c} for e in range(edges) for c in range(clouds)
        ],
    }


def _engine(
    path: Path,
    payload: dict[str, Any] | None = None,
    *,
    instance_id: int = 0,
) -> Any:
    payload = _payload(1) if payload is None else payload
    topology = LwdTopology.from_dict(payload)
    # Dynamic unit-test stub: the real engine needs a full torch/NPU
    # assembly that this CPU-only suite cannot construct.
    engine: Any = object.__new__(LwdEdgeEngineCore)
    engine.lwd_config = SimpleNamespace(
        path=str(path),
        topology=topology,
        instance_id=instance_id,
        # The watch gate reads the role against the scene before the
        # watcher thread ever starts; the coordination block mirrors the
        # config layer's per-instance LwdCoordination (scale fields:
        # enabled/placement_only/tenant_key_file/... — the probe/prefix
        # paths read it; there is no edge-side URL template anymore).
        role="edge",
        coordination=topology.instance("edge", instance_id).coordination,
    )
    engine._lwd_topology_file_digest = engine.lwd_config.topology.digest
    engine._lwd_control_plane = _RecordingPlane()
    engine.scheduler = _RecordingScheduler()
    engine._lwd_runtime_connections = {}
    engine._lwd_serviceable_connections = set()
    engine._lwd_probe_manager = _RecordingProbeManager()
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


def test_the_runtime_probe_url_comes_from_the_cloud_dp_block() -> None:
    """The runtime-registered probe URL comes from the adopted cloud's
    OWN DP block: an explicit coord_url (DNS/NAT/gateway deployments —
    the complete URL the probe POSTs to) wins, otherwise it is derived
    from the DP's routable address and coord_port. There is no edge-side
    URL template to borrow a path from on this schema; a DP with no
    coordination endpoint registers nothing (an edge_share scale cloud,
    by design)."""
    engine: Any = object.__new__(LwdEdgeEngineCore)
    engine._lwd_probe_manager = _RecordingProbeManager()
    # Explicit coord_url wins over the addr:coord_port derivation (a
    # gateway whose external port differs from anything derivable).
    engine._lwd_register_probe_cloud(
        SimpleNamespace(
            connection_id=LwdConnectionId(0, 1, 0),
            cloud=SimpleNamespace(
                addr="16.32.10.11",
                coord_port=8101,
                coord_url="https://gateway.example.com:443",
            ),
        )
    )
    # No explicit URL: derived from this DP's own address + coord_port.
    engine._lwd_register_probe_cloud(
        SimpleNamespace(
            connection_id=LwdConnectionId(0, 2, 0),
            cloud=SimpleNamespace(addr="16.32.10.12", coord_port=8102, coord_url=None),
        )
    )
    # No coordination endpoint at all: nothing is registered.
    engine._lwd_register_probe_cloud(
        SimpleNamespace(
            connection_id=LwdConnectionId(0, 3, 0),
            cloud=SimpleNamespace(addr="16.32.10.13", coord_port=None, coord_url=None),
        )
    )
    assert engine._lwd_probe_manager.registered == [
        (LwdConnectionId(0, 1, 0), "https://gateway.example.com:443"),
        (LwdConnectionId(0, 2, 0), "http://16.32.10.12:8102"),
    ]
    # The registrations widen the health rotation to the adopted cloud DPs.
    assert engine._lwd_probe_manager.rotation() == ((1, 0), (2, 0))


def test_a_lwd_cluster_scale_out_teaches_the_probe_manager_the_new_cloud(
    tmp_path: Path,
) -> None:
    """lwd_cluster routing fix: the probe manager's target set — the
    health rotation and the URL map — was frozen at startup, so after a
    2e1c -> 2e2c scale-out every probe (and the routing preference it
    produces) stayed pinned to cloud 0. The apply hook registers the
    adopted connection with the scale API — register_connection takes
    the LwdConnectionId and the URL straight from the new cloud's own DP
    block (coord_url, derived at parse time from its addr:coord_port);
    no edge-side template is involved."""
    topology_file = tmp_path / "topology.yaml"
    _write(topology_file, _cluster_payload(2, 1))
    engine = _engine(topology_file, _cluster_payload(2, 1))
    # The grown deployment: both edges byte-identical, cloud 1 appended
    # with its own coordination endpoint in its own DP block.
    _write(topology_file, _cluster_payload(2, 2))
    verdict = engine._lwd_process_topology_reload()
    assert verdict is not None and not isinstance(verdict, LwdTopologyUpdateRejected)
    assert verdict.new_cloud_ids == (1,)
    # Edge 0's engine applied only its own connection (edge 1's engine
    # watches the same file and applies its own): the control-plane link
    # to the adopted cloud's DP endpoint.
    assert engine._lwd_control_plane.links == [
        ("tcp://16.32.10.11:5551", LwdConnectionId(0, 1, 0), 1)
    ]
    # The probe manager learned the new cloud through its CONNECTION
    # (cloud_id and dp_idx travel in the LwdConnectionId): the registered
    # URL is the cloud DP's own coord_url — derived from its routable
    # address and coordination port, the complete URL the probe POSTs
    # to (no path suffix, nothing borrowed from the edge).
    assert engine._lwd_probe_manager.registered == [
        (LwdConnectionId(0, 1, 0), "http://16.32.10.11:8101")
    ]
    assert engine._lwd_probe_manager.rotation() == ((1, 0),)
    # The data-plane hook registry is empty in this process, so the
    # connection stays control-plane only — but the probe registration
    # has already happened (it precedes the open-for-service attempt and never rolls
    # back: the next increment retries only the data-plane half).
    assert (1, 0) in engine._lwd_runtime_connections
    assert (1, 0) not in engine._lwd_serviceable_connections


def test_a_lwd_cluster_scale_out_needs_no_existing_edge_edit(
    tmp_path: Path,
) -> None:
    """R2 on this schema: adding a cloud requires NO edit to any existing
    instance — the new cloud's coordination URL lives in its own DP block
    (coord_port/coord_url), not in an edge-side map, so the natural hand
    edit (append the cloud instance, touch nothing else) passes the
    strict byte-identical comparison. The retired old-branch exemption
    (growing an edge's control_url_map) targeted a field this schema
    rejects at parse time, so nothing is relaxed — and a hand edit that
    DOES touch a realized instance (here an edge's coordination tunable)
    is rejected outright."""
    topology_file = tmp_path / "topology.yaml"
    _write(topology_file, _cluster_payload(2, 1))
    engine = _engine(topology_file, _cluster_payload(2, 1))
    # The natural hand edit: cloud 1 appended, every existing edge and
    # cloud byte-identical — accepted, no exemption needed.
    _write(topology_file, _cluster_payload(2, 2))
    verdict = engine._lwd_process_topology_reload()
    assert verdict is not None and not isinstance(verdict, LwdTopologyUpdateRejected)
    assert verdict.new_cloud_ids == (1,)
    # The same addition with one extra tweak to an EXISTING instance is
    # rejected by the strict comparison: no field of a realized instance
    # may move, coordination tunables included.
    touched = _cluster_payload(2, 2)
    touched["edges"][0]["coordination"]["connect_timeout"] = 6.0
    _write(topology_file, touched)
    rejected = engine._lwd_process_topology_reload()
    assert isinstance(rejected, LwdTopologyUpdateRejected)
    assert "existing instance modified" in rejected.reason
    assert "edge 0.coordination.connect_timeout" in rejected.reason


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


def test_a_refused_data_plane_opens_for_service_on_the_next_increment(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The phase C retry semantics: a refused data-plane build keeps the
    connection control-plane only; the next detected increment (a replay
    of the same one) retries ONLY the data-plane half — no duplicate
    control-plane add — and, on success, opens the connection for
    service."""
    build_results: list[bool] = []

    def fake_notify(connection: Any) -> bool:
        return build_results.pop() if build_results else False

    monkeypatch.setattr(lwd_edge_engine, "notify_data_plane_added", fake_notify)
    topology_file = tmp_path / "topology.yaml"
    _write(topology_file, _payload(1))
    engine = _engine(topology_file)
    # Liveness state the gate reads (a runtime adoption is recorded in
    # _lwd_clouds by the control-plane IO thread in production): seed the
    # monitor up front, both gate checks below run against it.
    engine._lwd_heartbeat = LwdHeartbeatMonitor(10.0, 1.0)
    engine._lwd_heartbeat.note_adopted(LwdConnectionId(1, 1, 0))
    engine._lwd_clouds = {LwdConnectionId(1, 1, 0): None}
    _write(topology_file, _payload(2))
    first = engine._lwd_process_topology_reload()
    assert first is not None and not isinstance(first, LwdTopologyUpdateRejected)
    # The refusal kept the connection control-plane only: not serviceable, never a
    # routing candidate, and the gate rejects with the real reason.
    assert (1, 0) in engine._lwd_runtime_connections
    assert (1, 0) not in engine._lwd_serviceable_connections
    assert engine.scheduler.serviceable == []
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
    assert (1, 0) in engine._lwd_serviceable_connections
    assert engine.scheduler.serviceable == [LwdConnectionId(1, 1, 0)]
    # A serviceable runtime connection is routable: the gate admits it.
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

    # The poll pace lives in the base engine since the watch moved there
    # (36230ecdfa): patch it there — the watch thread re-reads the module
    # global every iteration.
    monkeypatch.setattr(lwd_base_engine, "_LWD_TOPOLOGY_POLL_INTERVAL_S", 0.02)
    topology_file = tmp_path / "topology.yaml"
    _write(topology_file, _payload(1))
    engine = _engine(topology_file)
    engine.input_queue = _Queue()
    engine._lwd_topology_watch_stop = threading.Event()
    engine._lwd_start_topology_watch()
    try:
        _write(topology_file, _payload(2))
        # Windows file timestamps share the ~15.6ms system-clock tick, so
        # a replace milliseconds after the startup write can keep the SAME
        # mtime and the watcher (correctly) sees no change. Linux (the
        # deployment target) has nanosecond granularity; nudge the mtime
        # deterministically so the test never depends on the clock.
        stat = topology_file.stat()
        os.utime(
            topology_file,
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
        )
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
    engine._lwd_heartbeat = LwdHeartbeatMonitor(10.0, 1.0)
    engine._lwd_heartbeat.note_adopted(LwdConnectionId(1, 1, 0))
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
    assert (1, 0) not in engine._lwd_serviceable_connections

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
    (edges join the running cloud) — and BOTH sides in lwd_cluster,
    where every engine watches the file and applies only its own
    instance's connections. The other side always starts fresh
    with the grown yaml and never watches; device comm never watches."""
    topology_file = tmp_path / "topology.yaml"
    _write(topology_file, _payload(1))

    def engine(role: str, scene: str, *, host_comm: bool) -> LwdBaseEngineCore:
        engine: Any = object.__new__(LwdBaseEngineCore)
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
    # lwd_cluster: every engine watches (each applies its own instance).
    assert gated("edge", "lwd_cluster", host_comm=True)
    assert gated("cloud", "lwd_cluster", host_comm=True)
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
    new edge AND drives the data-plane build for each new connection —
    through collective_rpc since the UCX listener moved into the Worker
    processes (the endpoint Worker answers True, the others None). The
    real handshake accept/replay semantics live in
    test_lwd_handshake_allow_edge, the real link build in the ascend
    suite."""

    class _RecordingCloudPlane(LwdCloudControlPlane):
        """Stands in for the cloud control plane (see _RecordingPlane):
        the cloud's apply hook asserts the control-plane type before
        touching it."""

        def __init__(self) -> None:
            self.allowed: list[int] = []

        def allow_edge(self, edge_id: int) -> None:
            self.allowed.append(edge_id)

    built: list = []

    def fake_collective_rpc(name: str, args: Any = None) -> list[Any]:
        # The data-plane build moved behind collective_rpc (the shared
        # listener lives in the Worker processes): the endpoint Worker
        # answers True, the others None (not their job).
        if name == "lwd_build_runtime_connection":
            built.append(args[0].connection_id)
            return [True, None]
        raise AssertionError(f"unexpected rpc under test: {name}")

    topology_file = tmp_path / "cloud_topology.yaml"
    _cloud_share_write(topology_file, 1)
    engine: Any = object.__new__(LwdCloudEngineCore)
    engine.lwd_config = SimpleNamespace(
        path=str(topology_file),
        topology=LwdTopology.from_dict(yaml.safe_load(topology_file.read_text())),
        instance_id=0,
    )
    engine.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_index=0)
    )
    engine._lwd_topology_file_digest = engine.lwd_config.topology.digest
    engine._lwd_control_plane = _RecordingCloudPlane()
    engine._lwd_hint_mqs = {}
    engine._lwd_done_mqs = {}
    engine.model_executor = SimpleNamespace(collective_rpc=fake_collective_rpc)
    wired: list = []

    def fake_wire(connection: LwdConnectionId) -> None:
        # Mirrors the real method's first-line guard: a replayed wire call
        # returns early on an already-wired connection.
        if connection in engine._lwd_hint_mqs:
            return
        wired.append(connection)
        engine._lwd_hint_mqs[connection] = "hint-mq"
        engine._lwd_done_mqs[connection] = "done-mq"

    monkeypatch.setattr(engine, "_lwd_wire_runtime_hint_mq", fake_wire)
    _cloud_share_write(topology_file, 2)
    verdict = engine._lwd_process_topology_reload()
    assert verdict is not None and not isinstance(verdict, LwdTopologyUpdateRejected)
    assert verdict.new_edge_ids == (1,)
    assert engine._lwd_control_plane.allowed == [1]
    # The data-plane build was driven through collective_rpc and the MQ
    # wiring ran.
    assert built == [LwdConnectionId(1, 0, 0)]
    assert wired == [LwdConnectionId(1, 0, 0)]
    assert engine._lwd_hint_mqs == {LwdConnectionId(1, 0, 0): "hint-mq"}
    assert engine._lwd_done_mqs == {LwdConnectionId(1, 0, 0): "done-mq"}
    # A replay of the same increment: allow_edge fires again (idempotent),
    # the data-plane build fires again, but the MQ wiring skips (already
    # wired).
    engine._lwd_apply_topology_increment(verdict)
    assert engine._lwd_control_plane.allowed == [1, 1]
    assert built == [LwdConnectionId(1, 0, 0), LwdConnectionId(1, 0, 0)]
    assert wired == [LwdConnectionId(1, 0, 0)]  # not re-wired


def test_cloud_apply_adopts_only_its_own_dp_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multi-DP cloud runs one engine per DP. Every engine sees the same
    topology increment, but only the DP whose data_parallel_index matches
    the connection's dp_idx may adopt it — otherwise each Worker extends
    state with the other DPs' connection ids and the runner loses
    per-edge connection uniqueness."""
    built: list = []

    class _Plane(LwdCloudControlPlane):
        def __init__(self) -> None:
            self.allowed: list[int] = []

        def allow_edge(self, edge_id: int) -> None:
            self.allowed.append(edge_id)

    def fake_collective_rpc(name: str, args: Any = None) -> list[Any]:
        assert name == "lwd_build_runtime_connection"
        built.append(args[0].connection_id)
        return [True, None]

    engine: Any = object.__new__(LwdCloudEngineCore)
    engine.lwd_config = SimpleNamespace(instance_id=0)
    engine.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_index=0)
    )
    engine._lwd_control_plane = _Plane()
    engine._lwd_hint_mqs = {}
    engine._lwd_done_mqs = {}
    engine.model_executor = SimpleNamespace(collective_rpc=fake_collective_rpc)
    monkeypatch.setattr(engine, "_lwd_wire_runtime_hint_mq", lambda connection: None)

    own = SimpleNamespace(connection_id=LwdConnectionId(1, 0, 0))
    other_dp = SimpleNamespace(connection_id=LwdConnectionId(1, 0, 1))
    increment = SimpleNamespace(new_edge_ids=(1,), new_connections=(own, other_dp))
    engine._lwd_apply_topology_increment(increment)

    assert built == [LwdConnectionId(1, 0, 0)]
    assert engine._lwd_control_plane.allowed == [1]


def _cloud_share_write(path: Path, edges: int) -> None:
    """A cloud_share topology on the cloud host's copy of the file: N
    edges, one cloud, full mesh. The scene's enabled coordination blocks
    make the cloud DP's coordination endpoint (coord_port, coord_url
    derived from it) mandatory."""
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
                        "coord_port": 8100,
                    }
                ],
            }
        ],
        "instance_links": [{"edge": i, "cloud": 0} for i in range(edges)],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


class _RecordingRuntimeMq:
    def __init__(self, handle: str) -> None:
        self.handle = handle
        self.enqueued: list[Any] = []
        self.shutdown_calls = 0

    def export_handle(self) -> str:
        return self.handle

    def enqueue(self, hint: Any) -> None:
        self.enqueued.append(hint)

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _runtime_mq_harness(monkeypatch: Any, collective_rpc: Any) -> SimpleNamespace:
    """Stub only the optional MQ boundaries around the real wiring method."""
    engine: Any = object.__new__(LwdCloudEngineCore)
    engine._lwd_hint_mqs = {}
    engine._lwd_done_mqs = {}
    engine.model_executor = SimpleNamespace(collective_rpc=collective_rpc)
    hint_mqs: list[_RecordingRuntimeMq] = []
    done_mqs: list[_RecordingRuntimeMq] = []
    senders: dict[LwdConnectionId, Any] = {}

    def create_hint_mq(connection: LwdConnectionId) -> _RecordingRuntimeMq:
        hint_mq = _RecordingRuntimeMq(f"hint-handle-{len(hint_mqs)}")
        hint_mqs.append(hint_mq)
        return hint_mq

    def register_hint_sender(sender: Any, connection: LwdConnectionId) -> None:
        senders[connection] = sender

    def create_from_handle(handle: str, rank: int) -> _RecordingRuntimeMq:
        assert rank == 0
        done_mq = _RecordingRuntimeMq(handle)
        done_mqs.append(done_mq)
        return done_mq

    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv",
        SimpleNamespace(
            _encode_handle=lambda handle: f"encoded:{handle}",
            _hint_senders=senders,
            create_hint_mq=create_hint_mq,
            register_hint_sender=register_hint_sender,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.device_communicators.shm_broadcast",
        SimpleNamespace(
            MessageQueue=SimpleNamespace(create_from_handle=create_from_handle)
        ),
    )
    return SimpleNamespace(
        engine=engine,
        hint_mqs=hint_mqs,
        done_mqs=done_mqs,
        senders=senders,
    )


def test_runtime_mq_rpc_error_is_contained_and_rolled_back(
    monkeypatch: Any,
) -> None:
    connection = LwdConnectionId(1, 0, 0)
    existing = LwdConnectionId(0, 0, 0)
    existing_hint = object()
    existing_done = object()
    existing_sender = object()

    def fail_rpc(name: str, args: Any = None) -> list[Any]:
        raise TypeError("legacy worker attach signature")

    harness = _runtime_mq_harness(monkeypatch, fail_rpc)
    harness.engine._lwd_hint_mqs[existing] = existing_hint
    harness.engine._lwd_done_mqs[existing] = existing_done
    harness.senders[existing] = existing_sender

    harness.engine._lwd_wire_runtime_hint_mq(connection)

    assert harness.engine._lwd_hint_mqs == {existing: existing_hint}
    assert harness.engine._lwd_done_mqs == {existing: existing_done}
    assert harness.senders == {existing: existing_sender}
    assert len(harness.hint_mqs) == 1
    assert harness.hint_mqs[0].shutdown_calls == 1
    assert harness.done_mqs == []


@pytest.mark.parametrize(
    "handles",
    [[None, None], ["done-handle-0", "done-handle-1"]],
    ids=["zero-handles", "two-handles"],
)
def test_runtime_mq_bad_handle_count_rolls_back(
    monkeypatch: Any, handles: list[Any]
) -> None:
    connection = LwdConnectionId(1, 0, 0)
    calls: list[tuple[str, Any]] = []

    def rpc(name: str, args: Any = None) -> list[Any]:
        calls.append((name, args))
        return handles

    harness = _runtime_mq_harness(monkeypatch, rpc)
    harness.engine._lwd_wire_runtime_hint_mq(connection)

    assert calls == [
        (
            "lwd_attach_runtime_connection_mqs",
            (connection, "encoded:hint-handle-0"),
        )
    ]
    assert harness.engine._lwd_hint_mqs == {}
    assert harness.engine._lwd_done_mqs == {}
    assert harness.senders == {}
    assert harness.hint_mqs[0].shutdown_calls == 1
    assert harness.done_mqs == []


def test_runtime_mq_failure_can_be_retried_with_the_same_rpc_contract(
    monkeypatch: Any,
) -> None:
    connection = LwdConnectionId(1, 0, 0)
    outcomes: list[Any] = [
        RuntimeError("temporary attach failure"),
        [None, "done-handle"],
    ]
    calls: list[tuple[str, Any]] = []

    def rpc(name: str, args: Any = None) -> list[Any]:
        calls.append((name, args))
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    harness = _runtime_mq_harness(monkeypatch, rpc)
    harness.engine._lwd_wire_runtime_hint_mq(connection)
    assert connection not in harness.engine._lwd_hint_mqs
    assert connection not in harness.engine._lwd_done_mqs
    assert connection not in harness.senders

    harness.engine._lwd_wire_runtime_hint_mq(connection)

    new_hint = harness.hint_mqs[1]
    assert harness.engine._lwd_hint_mqs == {connection: new_hint}
    assert harness.engine._lwd_done_mqs == {connection: harness.done_mqs[0]}
    assert harness.hint_mqs[0].shutdown_calls == 1
    assert new_hint.shutdown_calls == 0
    assert harness.done_mqs[0].handle == "done-handle"
    hint = {"seqno": 1}
    harness.senders[connection](hint)
    assert new_hint.enqueued == [hint]
    assert calls == [
        (
            "lwd_attach_runtime_connection_mqs",
            (connection, "encoded:hint-handle-0"),
        ),
        (
            "lwd_attach_runtime_connection_mqs",
            (connection, "encoded:hint-handle-1"),
        ),
    ]


def test_runtime_mq_existing_connection_is_not_rewired_or_cleaned(
    monkeypatch: Any,
) -> None:
    connection = LwdConnectionId(0, 0, 0)
    hint_mq = _RecordingRuntimeMq("existing-hint")
    done_mq = _RecordingRuntimeMq("existing-done")
    sender = object()

    def unexpected_rpc(name: str, args: Any = None) -> list[Any]:
        raise AssertionError("an existing connection must not be attached again")

    harness = _runtime_mq_harness(monkeypatch, unexpected_rpc)
    harness.engine._lwd_hint_mqs[connection] = hint_mq
    harness.engine._lwd_done_mqs[connection] = done_mq
    harness.senders[connection] = sender

    harness.engine._lwd_wire_runtime_hint_mq(connection)

    assert harness.engine._lwd_hint_mqs[connection] is hint_mq
    assert harness.engine._lwd_done_mqs[connection] is done_mq
    assert harness.senders[connection] is sender
    assert hint_mq.shutdown_calls == 0
    assert done_mq.shutdown_calls == 0
    assert harness.hint_mqs == []
    assert harness.done_mqs == []
