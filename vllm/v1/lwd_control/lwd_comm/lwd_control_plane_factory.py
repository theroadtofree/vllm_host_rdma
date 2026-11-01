# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the control plane of one DP rank from `VllmConfig.lwd_config`.

Every cloud DP rank binds its topology `ctrl_port` on all interfaces. One
edge process serves every DP connection of its instance: the edge plane of
DP rank i connects to the cloud DP rank of the same index.
"""

from vllm.config.lwd import LwdConfig
from vllm.utils.network_utils import is_valid_ipv6_address, make_zmq_path
from vllm.v1.lwd_control.lwd_comm.lwd_control_plane import (
    LwdCloudControlPlane,
    LwdCloudControlPlaneSpec,
    LwdControlPlane,
    LwdEdgeControlPlane,
    LwdEdgeControlPlaneSpec,
)

# The topology address may be unreachable from inside the host (NAT,
# containers), so the cloud binds every interface of the address family.
_LWD_IPV4_ANY = "0.0.0.0"
_LWD_IPV6_ANY = "::"


def lwd_create_control_plane(
    lwd_config: LwdConfig, dp_rank: int, dp_size: int
) -> LwdControlPlane:
    """Create, not start, the control plane of this engine's DP rank.

    Args:
        lwd_config: `VllmConfig.lwd_config` of this instance.
        dp_rank: `ParallelConfig.data_parallel_index` of this engine.
        dp_size: Global DP size. Non-MoE engine processes reset
            `ParallelConfig.data_parallel_size` to 1 before the engine is
            built, so read it before that.

    Raises:
        ValueError: `dp_size` or `dp_rank` does not match the topology or
            the role.
    """
    _lwd_validate_dp(lwd_config, dp_rank, dp_size)
    if lwd_config.role == "cloud":
        return LwdCloudControlPlane(_lwd_cloud_spec(lwd_config, dp_rank, dp_size))
    return LwdEdgeControlPlane(_lwd_edge_spec(lwd_config, dp_rank, dp_size))


def _lwd_validate_dp(lwd_config: LwdConfig, dp_rank: int, dp_size: int) -> None:
    """The edge runs one process per instance; the cloud one engine per DP."""
    topology_dp_size = len(lwd_config.instance.dp)
    if lwd_config.role == "edge":
        # One edge process serves every DP connection, so its vLLM DP size
        # is 1 and its dp_rank is 0.
        if dp_size != 1 or dp_rank != 0:
            raise ValueError(
                f"[LWD] edge instance {lwd_config.instance_id} runs one engine "
                f"for its {topology_dp_size} DP connections; start it with "
                f"data parallel size 1, got dp_size={dp_size} dp_rank={dp_rank}"
            )
        if topology_dp_size < 1:
            raise ValueError("[LWD] instance must have at least one DP")
        return
    if dp_size != topology_dp_size:
        raise ValueError(
            f"[LWD] vLLM data parallel size {dp_size} != topology DP count "
            f"{topology_dp_size} of cloud instance {lwd_config.instance_id}"
        )
    if not 0 <= dp_rank < dp_size:
        raise ValueError(f"[LWD] dp_rank {dp_rank} not in [0, {dp_size})")


def _lwd_cloud_spec(
    lwd_config: LwdConfig, dp_rank: int, dp_size: int
) -> LwdCloudControlPlaneSpec:
    local = lwd_config.instance.dp[dp_rank]
    assert local.ctrl_port is not None, "topology requires ctrl_port on clouds"
    bind_host = _LWD_IPV6_ANY if is_valid_ipv6_address(local.addr) else _LWD_IPV4_ANY
    return LwdCloudControlPlaneSpec(
        instance_id=lwd_config.instance_id,
        dp_rank=dp_rank,
        dp_size=dp_size,
        bind_endpoint=make_zmq_path("tcp", bind_host, local.ctrl_port),
        config=lwd_config.topology.control_plane,
        expected_edge_ids=tuple(
            sorted(
                c.edge_id for c in lwd_config.connections if c.dp_idx == dp_rank
            )
        ),
    )


def _lwd_edge_spec(
    lwd_config: LwdConfig, dp_rank: int, dp_size: int
) -> LwdEdgeControlPlaneSpec:
    """Endpoints of the cloud DP ranks this edge engine connects to.

    One edge process serves every DP connection of its instance: link i
    connects to the cloud DP rank of DP index i.
    """
    connections = sorted(lwd_config.connections, key=lambda c: c.connection_id)
    cloud_dp_sizes = {
        cloud.id: len(cloud.dp) for cloud in lwd_config.topology.clouds
    }
    return LwdEdgeControlPlaneSpec(
        instance_id=lwd_config.instance_id,
        dp_rank=dp_rank,
        dp_size=dp_size,
        cloud_endpoints=tuple(
            make_zmq_path("tcp", c.cloud.addr, c.cloud.ctrl_port)
            for c in connections
            if c.cloud.ctrl_port is not None
        ),
        connection_ids=tuple(c.connection_id for c in connections),
        cloud_dp_sizes=tuple(cloud_dp_sizes[c.cloud_id] for c in connections),
        config=lwd_config.topology.control_plane,
    )
