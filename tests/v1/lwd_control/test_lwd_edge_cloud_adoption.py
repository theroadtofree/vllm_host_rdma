# SPDX-License-Identifier: Apache-2.0
"""Elastic partial start: a cloud that starts after the edge is adopted
when its scheduling limits arrive on the control-plane IO thread.

The edge's startup barrier returns the accepted subset (see
``test_lwd_edge_hello_reregister.py``); the missing cloud's handshake
keeps being retried by the periodic HELLO, and its ``LwdCloudParamsNotify``
is the first runtime traffic that adopts it: sendable peer, liveness
account seed, probe hash sizes — the same routine the init path uses.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

from vllm.config.lwd.topology import LwdConnectionId
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
    LwdEdgeEngineCore,
)
from vllm.v1.lwd_control.lwd_comm.lwd_messages import LwdCloudParamsNotify

CLOUD_0 = LwdConnectionId(0, 0, 0)
CLOUD_1 = LwdConnectionId(0, 1, 0)


def _engine() -> LwdEdgeEngineCore:
    engine = object.__new__(LwdEdgeEngineCore)
    engine.lwd_config = SimpleNamespace(
        connections=[
            SimpleNamespace(cloud_id=0, dp_idx=0, connection_id=CLOUD_0),
            SimpleNamespace(cloud_id=1, dp_idx=0, connection_id=CLOUD_1),
        ]
    )
    engine.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=64)
    )
    engine._lwd_hello_lock = threading.Lock()
    engine._lwd_hello_last = {CLOUD_0: 1.0}
    # Cloud 0 is the accepted startup subset; cloud 1 starts later.
    engine._lwd_clouds = {CLOUD_0: SimpleNamespace(instance_id=0, dp_rank=0)}
    engine._lwd_cloud_params = {}
    engine._lwd_cloud_params_received = threading.Event()
    engine._lwd_coord_clients = {}
    engine._lwd_probe_manager = SimpleNamespace(_hash_sizes={})
    return engine


def _params() -> LwdCloudParamsNotify:
    return LwdCloudParamsNotify(
        max_num_batched_tokens=8192,
        max_num_seqs=64,
        num_speculative_tokens=0,
        prefix_hash_block_size=128,
    )


def test_late_cloud_params_adopt_the_connection() -> None:
    engine = _engine()
    updates: list = []
    engine.scheduler = SimpleNamespace(
        lwd_update_cloud_params=lambda connection, params: updates.append(
            (connection, params)
        )
    )

    peer = SimpleNamespace(instance_id=1, dp_rank=0)
    msg = _params()
    assert engine._lwd_dispatch_message(peer, msg)

    assert CLOUD_1 in engine._lwd_clouds  # sendable peer adopted
    assert engine._lwd_cloud_params[CLOUD_1] is msg
    assert engine._lwd_coord_clients == {}  # no coordinator addr -> no client
    assert engine._lwd_probe_manager._hash_sizes[1] == msg.prefix_hash_block_size
    assert CLOUD_1 in engine._lwd_hello_last  # liveness account seeded
    assert engine._lwd_cloud_params_received.is_set()
    assert updates == [(CLOUD_1, msg)]

    # Idempotent: a re-arrival (e.g. the watchdog's periodic limits
    # refresh) just refreshes the same state.
    assert engine._lwd_dispatch_message(peer, msg)
    assert len(engine._lwd_clouds) == 2


def test_config_mismatch_on_adoption_fails_loudly() -> None:
    """The runtime adoption runs the same limits check as init: a late
    cloud with mismatched scheduling limits is a deployment error."""
    engine = _engine()
    engine.scheduler = SimpleNamespace(
        lwd_update_cloud_params=lambda connection, params: None
    )
    peer = SimpleNamespace(instance_id=1, dp_rank=0)
    bad = _params()
    bad.max_num_batched_tokens = 4096
    try:
        engine._lwd_dispatch_message(peer, bad)
    except RuntimeError as exc:
        assert "mismatch" in str(exc)
    else:  # pragma: no cover - the check must raise
        raise AssertionError("mismatched limits were accepted on adoption")
    # The connection was not half-adopted.
    assert CLOUD_1 not in engine._lwd_clouds
