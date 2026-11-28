# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression for the gateway coordination mode's liveness verdict path.

The new main installs either LwdEdgeProbeManager (direct) or
LwdGatewayActivationManager (gateway) as ``_lwd_probe_manager``. The latter
used to lack ``set_silent_clouds`` / ``register_connection`` / ``rotation``,
so every heartbeat monitor emit logged a subscriber failure (one per sweep)
and elastic scale-out missed the runtime-added connection entirely.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from vllm.config.lwd.topology import LwdConnectionId
from vllm.v1.lwd_control.control_edge_scheduler import lwd_heartbeat
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
    LwdEdgeEngineCore,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_probe import (
    LwdEdgeProbeManager,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_gateway_activation import (
    LwdGatewayActivationManager,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_heartbeat import (
    LwdHeartbeatMonitor,
    LwdHeartbeatVerdict,
)

CONNECTION = LwdConnectionId(0, 0, 0)
ADDED = LwdConnectionId(0, 1, 0)


def _gateway_manager() -> LwdGatewayActivationManager:
    manager = object.__new__(LwdGatewayActivationManager)
    manager._silent_clouds = frozenset()
    manager._targets = (CONNECTION,)
    return manager


def _config() -> SimpleNamespace:
    """Minimal real input for the probe manager: one adopted cloud DP with
    a coordination URL. Configuration data is an external input, not a
    mocked service."""
    return SimpleNamespace(
        lwd_config=SimpleNamespace(
            instance_id=0,
            connections=[
                SimpleNamespace(
                    connection_id=CONNECTION,
                    cloud=SimpleNamespace(
                        coord_url="http://cloud-0:9000",
                        coord_port=None,
                        addr="cloud-0",
                    ),
                )
            ],
        )
    )


def _engine(manager: object) -> LwdEdgeEngineCore:
    engine = object.__new__(LwdEdgeEngineCore)
    engine._lwd_probe_manager = manager
    return engine


def _verdict(*cloud_ids: int) -> LwdHeartbeatVerdict:
    return LwdHeartbeatVerdict(
        silent_routes=frozenset(), silent_clouds=frozenset(cloud_ids)
    )


def test_gateway_manager_caches_verdict_and_clears_on_recovery() -> None:
    manager = _gateway_manager()
    manager.set_silent_clouds(frozenset({0}))
    assert manager.silent_clouds == frozenset({0})
    manager.set_silent_clouds(frozenset())
    assert manager.silent_clouds == frozenset()


def test_gateway_registration_widens_visible_targets_idempotently() -> None:
    manager = _gateway_manager()
    manager.register_connection(ADDED, "http://unused-by-gateway")
    assert manager._targets == (CONNECTION, ADDED)
    assert manager._visible_targets(0) == (CONNECTION, ADDED)
    assert manager.rotation() == ((0, 0), (1, 0))

    manager.register_connection(ADDED, "http://unused-by-gateway")
    assert manager._targets == (CONNECTION, ADDED)


def test_gateway_branch_updates_the_cache_without_raising() -> None:
    manager = _gateway_manager()
    _engine(manager)._lwd_apply_silent_clouds(_verdict(0))
    assert manager.silent_clouds == frozenset({0})


def test_direct_branch_excludes_the_cloud_in_the_health_tracker() -> None:
    manager = LwdEdgeProbeManager(_config(), lambda: None)
    _engine(manager)._lwd_apply_silent_clouds(_verdict(0))
    assert manager._health.silent == frozenset({(0, 0)})
    # Every target silent: probe targeting fails open; the admission gate
    # owns the steady-state rejection.
    assert manager._health.candidates(exclude_silent=True) == ((0, 0),)


def test_unknown_manager_fails_fast() -> None:
    engine = _engine(object())
    with pytest.raises(TypeError, match="unsupported probe manager"):
        engine._lwd_apply_silent_clouds(_verdict(0))


def test_monitor_sweep_with_gateway_sink_logs_no_subscriber_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The original defect: a gateway deployment logged one exception per
    sweep because the verdict subscriber called a direct-probe-only method."""
    failures: list[str] = []
    monkeypatch.setattr(
        lwd_heartbeat.lwd_logger,
        "exception",
        lambda message, *args: failures.append(message),
    )
    manager = _gateway_manager()
    engine = _engine(manager)

    monitor = LwdHeartbeatMonitor(10.0, 0.0)
    monitor.subscribe(engine._lwd_apply_silent_clouds)
    monitor.note_adopted(CONNECTION)
    monitor._hello_last[CONNECTION] = time.monotonic() - 30.0
    monitor.sweep()

    assert manager.silent_clouds == frozenset({0})
    assert failures == []
