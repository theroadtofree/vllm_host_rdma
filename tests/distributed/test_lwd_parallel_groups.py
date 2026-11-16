# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.config.lwd import LwdConfig, LwdConnectionId, LwdTopology
from vllm.distributed.lwd import (
    LwdChannelEndpoints,
    LwdParallelGroupsSpec,
    lwd_check_parallel_config,
)
from vllm.v1.lwd_control.lwd_comm.lwd_control_plane_factory import _lwd_validate_dp


def _instance(
    identity: int,
    ranks: list[list[int]],
    cloud: bool,
    addr: str | None = None,
    coordination: bool = False,
) -> dict[str, Any]:
    address = addr if addr is not None else f"10.{int(cloud)}.0.{identity + 1}"
    return {
        "id": identity,
        "dp": [
            {
                "dp_idx": index,
                "addr": address,
                "ranks": group,
                **({"ctrl_port": 5550 + index} if cloud else {}),
            }
            for index, group in enumerate(ranks)
        ],
        **({"coordination": {"enabled": True}} if coordination else {}),
    }


def _topology(
    scene: str,
    edges: list[dict[str, Any]],
    clouds: list[dict[str, Any]],
    links: list[tuple[int, int]],
) -> LwdTopology:
    world_size = len(
        {rank for dps in (*edges, *clouds) for dp in dps["dp"] for rank in dp["ranks"]}
    )
    return LwdTopology.from_dict(
        {
            "deployment": {
                "mode": 0,
                "scene": scene,
                "hccl_world_size": world_size,
                "edges_num": len(edges),
                "clouds_num": len(clouds),
            },
            "feature_ctrl": {
                "enable_early_recv": False,
                "scramble": {
                    "enable": False,
                    "device_id": None,
                    "custom_factor": None,
                },
            },
            "edges": edges,
            "clouds": clouds,
            "instance_links": [{"edge": e, "cloud": c} for e, c in links],
        }
    )


def test_single_instance_groups_each_side_and_pairs_the_first_ranks() -> None:
    spec = LwdParallelGroupsSpec.from_topology(
        _topology(
            "single_instance",
            [_instance(0, [[0, 1]], cloud=False)],
            [_instance(0, [[2, 3, 4, 5]], cloud=True)],
            [(0, 0)],
        )
    )

    assert spec.world_size == 6
    assert spec.tp_groups == ((0, 1), (2, 3, 4, 5))
    assert spec.singleton_groups == ((0,), (1,), (2,), (3,), (4,), (5,))
    # One DP per instance: DP groups degenerate to singletons and the EP
    # groups equal the TP groups, as before multi-DP.
    assert spec.dp_groups == spec.singleton_groups
    assert spec.ep_groups == spec.tp_groups
    assert spec.channel_endpoints == (
        LwdChannelEndpoints(LwdConnectionId(0, 0, 0), 0, 2),
    )
    assert spec.tp_group_of(4) == (2, 3, 4, 5)
    assert spec.channel_endpoints_of(2) == (
        LwdChannelEndpoints(LwdConnectionId(0, 0, 0), 0, 2),
    )
    assert spec.channel_endpoints_of(1) == ()


def test_single_instance_multi_dp_serves_every_cloud_dp_from_one_edge() -> None:
    spec = LwdParallelGroupsSpec.from_topology(
        _topology(
            "single_instance",
            [_instance(0, [[0], [0]], cloud=False)],
            [_instance(0, [[1, 2], [3, 4]], cloud=True)],
            [(0, 0)],
        )
    )

    assert spec.world_size == 5
    # The edge DPs share their ranks (one process), so the edge collapses to
    # one allocation; each cloud DP keeps its own.
    assert spec.tp_groups == ((0,), (1, 2), (3, 4))
    # Co-located cloud ranks across the DPs; edge ranks stay singletons.
    assert spec.dp_groups == ((0,), (1, 3), (2, 4))
    # Experts span every cloud rank of the instance (dp x tp, as natively).
    assert spec.ep_groups == ((0,), (1, 2, 3, 4))
    # One channel pair per DP connection, all held by the edge endpoint.
    assert spec.channel_endpoints == (
        LwdChannelEndpoints(LwdConnectionId(0, 0, 0), 0, 1),
        LwdChannelEndpoints(LwdConnectionId(0, 0, 1), 0, 3),
    )
    assert spec.channel_endpoints_of(0) == (
        LwdChannelEndpoints(LwdConnectionId(0, 0, 0), 0, 1),
        LwdChannelEndpoints(LwdConnectionId(0, 0, 1), 0, 3),
    )
    assert spec.channel_endpoints_of(3) == (
        LwdChannelEndpoints(LwdConnectionId(0, 0, 1), 0, 3),
    )
    assert spec.channel_endpoints_of(2) == ()


def test_edge_share_serves_each_cloud_from_the_shared_edge_rank() -> None:
    spec = LwdParallelGroupsSpec.from_topology(
        _topology(
            "edge_share",
            [
                _instance(0, [[0]], cloud=False, addr="10.0.0.1"),
                _instance(1, [[0]], cloud=False, addr="10.0.0.1"),
            ],
            [
                _instance(0, [[1]], cloud=True, addr="10.1.0.1"),
                _instance(1, [[2]], cloud=True, addr="10.1.0.2"),
            ],
            [(0, 0), (1, 1)],
        )
    )

    # The shared edge rank holds both one-to-one channels; each cloud keeps
    # its own leader-rank endpoint.
    assert spec.world_size == 3
    assert spec.channel_endpoints == (
        LwdChannelEndpoints(LwdConnectionId(0, 0, 0), 0, 1),
        LwdChannelEndpoints(LwdConnectionId(1, 1, 0), 0, 2),
    )
    assert spec.channel_endpoints_of(0) == (
        LwdChannelEndpoints(LwdConnectionId(0, 0, 0), 0, 1),
        LwdChannelEndpoints(LwdConnectionId(1, 1, 0), 0, 2),
    )
    assert spec.channel_endpoints_of(1) == (
        LwdChannelEndpoints(LwdConnectionId(0, 0, 0), 0, 1),
    )
    assert spec.channel_endpoints_of(2) == (
        LwdChannelEndpoints(LwdConnectionId(1, 1, 0), 0, 2),
    )


def test_full_mesh_cluster_connects_every_edge_to_every_cloud() -> None:
    spec = LwdParallelGroupsSpec.from_topology(
        _topology(
            "lwd_cluster",
            [
                _instance(0, [[0]], cloud=False, coordination=True),
                _instance(1, [[1]], cloud=False, coordination=True),
            ],
            [
                _instance(0, [[2]], cloud=True, coordination=True),
                _instance(1, [[3]], cloud=True, coordination=True),
            ],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        )
    )

    # Every edge rank and cloud rank now holds the channels of several DP
    # connections, one pair per edge-cloud link.
    assert spec.world_size == 4
    assert spec.channel_endpoints == (
        LwdChannelEndpoints(LwdConnectionId(0, 0, 0), 0, 2),
        LwdChannelEndpoints(LwdConnectionId(0, 1, 0), 0, 3),
        LwdChannelEndpoints(LwdConnectionId(1, 0, 0), 1, 2),
        LwdChannelEndpoints(LwdConnectionId(1, 1, 0), 1, 3),
    )
    assert spec.channel_endpoints_of(0) == (
        LwdChannelEndpoints(LwdConnectionId(0, 0, 0), 0, 2),
        LwdChannelEndpoints(LwdConnectionId(0, 1, 0), 0, 3),
    )
    assert spec.channel_endpoints_of(2) == (
        LwdChannelEndpoints(LwdConnectionId(0, 0, 0), 0, 2),
        LwdChannelEndpoints(LwdConnectionId(1, 0, 0), 1, 2),
    )


def test_rank_outside_the_world_has_no_tp_group() -> None:
    spec = LwdParallelGroupsSpec.from_topology(
        _topology(
            "single_instance",
            [_instance(0, [[0]], cloud=False)],
            [_instance(0, [[1]], cloud=True)],
            [(0, 0)],
        )
    )

    with pytest.raises(ValueError, match="not in"):
        spec.tp_group_of(2)


def _host_config(
    role: str,
    tensor_parallel_size: int,
    *,
    dp_count: int = 1,
    dp_size: int = 1,
) -> Any:
    """Host layout of one edge and one cloud, no ranks, the cloud listens."""
    topology = LwdTopology.from_dict(
        {
            "deployment": {
                "mode": 0,
                "scene": "single_instance",
                "edges_num": 1,
                "clouds_num": 1,
                "data_plane_comm": "host",
            },
            "feature_ctrl": {"enable_early_recv": False},
            "edges": [
                {
                    "id": 0,
                    "dp": [
                        {"dp_idx": index, "addr": "10.0.0.1"}
                        for index in range(dp_count)
                    ],
                }
            ],
            "clouds": [
                {
                    "id": 0,
                    "dp": [
                        {
                            "dp_idx": index,
                            "addr": "10.1.0.1",
                            "ctrl_port": 5550 + index,
                            "data_port": 7000 + index,
                        }
                        for index in range(dp_count)
                    ],
                }
            ],
            "instance_links": [{"edge": 0, "cloud": 0}],
        }
    )
    return SimpleNamespace(
        lwd_config=LwdConfig.from_topology("host_topology.yaml", role, 0, topology),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_elastic_ep=False,
            data_parallel_size=dp_size,
            tensor_parallel_size=tensor_parallel_size,
        ),
        engram_config=None,
    )


def _device_config(role: str, *, dp_count: int, dp_size: int) -> Any:
    """Device layout of one edge and one cloud: shared world, per-DP ranks."""
    topology = _topology(
        "single_instance",
        [_instance(0, [[0]] * dp_count, cloud=False)],
        [_instance(0, [[index + 1] for index in range(dp_count)], cloud=True)],
        [(0, 0)],
    )
    return SimpleNamespace(
        lwd_config=LwdConfig.from_topology("device_topology.yaml", role, 0, topology),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_elastic_ep=False,
            data_parallel_size=dp_size,
            tensor_parallel_size=1,
        ),
        engram_config=None,
    )


def test_host_edge_rejects_tensor_parallelism() -> None:
    """Only edge TP0 holds the host links; TP>1 has no aggregation path."""
    with pytest.raises(
        ValueError, match="host edge supports only tensor_parallel_size=1"
    ):
        lwd_check_parallel_config(_host_config("edge", 2))


@pytest.mark.parametrize(("role", "tensor_parallel_size"), [("edge", 1), ("cloud", 4)])
def test_host_mode_accepts_edge_tp1_and_cloud_tensor_parallelism(
    role: str, tensor_parallel_size: int
) -> None:
    """The rejection is edge-only: a single-rank edge stays valid, and the
    cloud keeps its TP group for the prompt broadcast."""
    lwd_check_parallel_config(_host_config(role, tensor_parallel_size))


def test_multi_dp_config_check_leaves_the_dp_layout_to_the_engine() -> None:
    """The config object must not enforce the DP layout.

    vLLM rebuilds derived configs after an independent non-MoE DP engine reset
    `data_parallel_size` to 1 (e.g. the MTP draft config), so a check on this
    object would reject a valid 2-DP launch. The role rule is enforced where
    the launch value still exists: the engine dispatch and the control plane.
    """
    lwd_check_parallel_config(_host_config("edge", 1, dp_count=2, dp_size=1))
    lwd_check_parallel_config(_host_config("cloud", 1, dp_count=2, dp_size=1))
    lwd_check_parallel_config(_device_config("edge", dp_count=2, dp_size=1))
    lwd_check_parallel_config(_device_config("cloud", dp_count=2, dp_size=1))


@pytest.mark.parametrize("host_comm", [True, False])
def test_control_plane_dp_rule_follows_the_role(host_comm: bool) -> None:
    """One rule for both data planes: edge=1, cloud=len(dp).

    The control plane receives the launch DP size, so it is the place that
    rejects an edge or cloud started with the wrong DP size regardless of the
    transport.
    """
    if host_comm:
        edge = _host_config("edge", 1, dp_count=2, dp_size=1).lwd_config
        cloud = _host_config("cloud", 1, dp_count=2, dp_size=1).lwd_config
    else:
        edge = _device_config("edge", dp_count=2, dp_size=1).lwd_config
        cloud = _device_config("cloud", dp_count=2, dp_size=1).lwd_config

    _lwd_validate_dp(edge, dp_rank=0, dp_size=1)
    _lwd_validate_dp(cloud, dp_rank=1, dp_size=2)
    with pytest.raises(ValueError, match="data parallel size 1"):
        _lwd_validate_dp(edge, dp_rank=0, dp_size=2)
    with pytest.raises(ValueError, match="data parallel size 1"):
        _lwd_validate_dp(cloud, dp_rank=0, dp_size=1)
