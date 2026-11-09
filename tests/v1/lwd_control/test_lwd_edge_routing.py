# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config.lwd import LwdConnectionId
from vllm.v1.lwd_control.control_edge_scheduler.lwd_cloud_dp_account import (
    LwdCloudDpAccount,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_probe import (
    LwdEdgeProbeManager,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdEdgeScheduler,
)


@pytest.mark.parametrize(
    ("edge_id", "expected"),
    [(0, [0, 1, 2]), (1, [1, 2, 0]), (2, [2, 0, 1]), (3, [0, 1, 2])],
)
def test_control_url_map_round_robin_starts_at_edge_offset(
    edge_id: int, expected: list[int]
) -> None:
    coordination = SimpleNamespace(
        control_url_map={cloud_id: f"http://cloud-{cloud_id}" for cloud_id in range(3)}
    )
    vllm_config = SimpleNamespace(
        lwd_config=SimpleNamespace(instance_id=edge_id, coordination=coordination)
    )
    manager = LwdEdgeProbeManager(vllm_config, lambda: None)

    assert [manager._health.next() for _ in range(3)] == expected


def _routing_scheduler(edge_id: int, dp_count: int) -> LwdEdgeScheduler:
    connections = tuple(
        LwdConnectionId(edge_id, cloud_id, dp_idx)
        for cloud_id in range(3)
        for dp_idx in range(dp_count)
    )
    scheduler = object.__new__(LwdEdgeScheduler)
    scheduler._lwd_connections = connections
    scheduler._lwd_accounts = {
        connection_id: LwdCloudDpAccount(connection_id) for connection_id in connections
    }
    scheduler._lwd_route = {}
    scheduler._lwd_last_engine = -1
    return scheduler


@pytest.mark.parametrize("dp_count", [1, 2])
@pytest.mark.parametrize(
    ("edge_id", "expected_cloud"), [(0, 0), (1, 1), (2, 2), (3, 0)]
)
def test_fallback_routing_starts_at_edge_offset(
    edge_id: int, expected_cloud: int, dp_count: int
) -> None:
    scheduler = _routing_scheduler(edge_id, dp_count)

    selected = scheduler._lwd_route_request("request", edge_id)

    assert selected == LwdConnectionId(edge_id, expected_cloud, 0)
