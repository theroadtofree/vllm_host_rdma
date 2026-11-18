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
from vllm.v1.lwd_control.control_edge_scheduler import lwd_edge_engine
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
    LwdEdgeEngineCore,
)


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
    """Stands in for the scheduler: records runtime-connection seeds."""

    def __init__(self) -> None:
        self.connections: list[LwdConnectionId] = []

    def lwd_add_runtime_connection(self, connection_id: LwdConnectionId) -> None:
        self.connections.append(connection_id)


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
    engine._lwd_apply_increment(first)
    assert len(engine._lwd_control_plane.links) == 1


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
    assert tick is lwd_edge_engine._LWD_TOPOLOGY_RELOAD_TICK


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
    reason = engine._lwd_gate_reason("e1-wrapped-req")
    assert reason is not None
    assert "control plane only" in reason
    # The planned connection is not runtime: a fresh heartbeat admits it.
    engine._lwd_clouds = {LwdConnectionId(0, 0, 0): None}
    assert engine._lwd_gate_reason("plain-req") is None
