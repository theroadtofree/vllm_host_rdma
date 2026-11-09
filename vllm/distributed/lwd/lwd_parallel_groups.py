# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank groups of the HCCL world shared by the LWD edge and cloud instances.

Design: docs/design/lwd_data_plane.md.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.lwd import LwdTopology

LwdRankGroup = tuple[int, ...]


@dataclass(frozen=True)
class LwdChannelEndpoints:
    """Global ranks that hold the data-plane channels of one DP connection."""

    dp_idx: int
    edge_rank: int
    cloud_rank: int


@dataclass(frozen=True)
class LwdParallelGroupsSpec:
    """Rank groups that every process of the shared world builds.

    Each edge instance and each cloud DP runs like a centralized pp=1
    deployment on its own ranks: its TP group is its rank allocation and the
    PP, DCP and PCP groups are singletons. A cloud instance with several
    DPs is the native multi-DP layout: its DP groups hold the co-located
    ranks of every DP (MoE engines run their DP collectives there) and its
    EP groups span every cloud rank (experts sharded dp x tp, as natively).
    Each DP connection adds one edge-cloud rank pair for the data-plane
    channels; in a single_instance topology the edge process serves every
    connection, so its endpoint rank holds several channel pairs.

    Built by `from_topology`; every process derives the same spec from the
    same topology, so group creation (a world collective) runs in the same
    order everywhere.
    """

    world_size: int
    tp_groups: tuple[LwdRankGroup, ...]
    """One group per edge instance and cloud instance DP, by first rank."""
    dp_groups: tuple[LwdRankGroup, ...]
    """One group per co-located cloud rank across the DPs of its instance;
    singletons when every instance has one DP. Used by MoE engines only."""
    ep_groups: tuple[LwdRankGroup, ...]
    """Expert-parallel groups: one per edge instance, one per cloud instance
    spanning all its DPs' ranks. Equals tp_groups at one DP per instance."""
    channel_endpoints: tuple[LwdChannelEndpoints, ...]
    """One pair per DP connection, by (edge rank, cloud rank)."""

    @classmethod
    def from_topology(cls, topology: LwdTopology) -> LwdParallelGroupsSpec:
        """Derive the groups, or raise before any process creates one.

        Raises:
            NotImplementedError: Instances share a rank allocation, or a
                rank holds the channels of several DP connections, outside
                single_instance.
            ValueError: Allocations of distinct instances or DPs overlap.
        """
        scene = topology.deployment.scene
        # 1. One allocation per edge instance (its DPs share its ranks: one
        # process serves them all) and one per cloud DP (one engine each).
        edge_allocations = [edge.dp[0].ranks for edge in topology.edges]
        cloud_allocations = [dp.ranks for cloud in topology.clouds for dp in cloud.dp]
        if scene != "single_instance" and len(set(edge_allocations)) != len(
            edge_allocations
        ):
            # TODO(lwd): host shared edge ranks once each DP connection gets
            # its own communication domain instead of one shared world.
            raise NotImplementedError(
                "[LWD] edge instances sharing ranks cannot run in one shared "
                f"HCCL world (scene={scene}): one rank is one process"
            )
        seen: set[int] = set()
        for allocation in (*edge_allocations, *cloud_allocations):
            if seen.intersection(allocation):
                raise ValueError(
                    "[LWD] separate allocations require distinct ranks; "
                    f"overlapping ranks {sorted(seen.intersection(allocation))}"
                )
            seen.update(allocation)
        # 2. Endpoints are the first ranks; the cloud one broadcasts to its
        # TP group. In single_instance the edge process serves every DP
        # connection, so its endpoint rank holds several channels by design.
        connections = [
            connection
            for edge in topology.edges
            for connection in topology.connections("edge", edge.id)
        ]
        channel_endpoints = sorted(
            (
                LwdChannelEndpoints(
                    connection.dp_idx,
                    connection.edge.ranks[0],
                    connection.cloud_leader_rank,
                )
                for connection in connections
            ),
            key=lambda endpoints: (endpoints.edge_rank, endpoints.cloud_rank),
        )
        if scene != "single_instance":
            endpoint_uses = Counter(
                rank
                for endpoints in channel_endpoints
                for rank in (endpoints.edge_rank, endpoints.cloud_rank)
            )
            overloaded = sorted(
                rank for rank, uses in endpoint_uses.items() if uses > 1
            )
            if overloaded:
                # TODO(lwd): key the data-plane channels by DP connection.
                raise NotImplementedError(
                    f"[LWD] ranks {overloaded} hold the channels of several DP "
                    "connections; the data plane supports one per rank "
                    f"(scene={scene})"
                )
        # 3. Co-located DP groups: one per cloud-instance position, holding
        # that position's rank of every DP of the instance (all DPs split
        # the instance ranks equally); edge ranks are singletons.
        dp_groups: list[LwdRankGroup] = [
            (rank,) for edge in topology.edges for rank in edge.dp[0].ranks
        ]
        dp_groups += [
            tuple(dp.ranks[position] for dp in cloud.dp)
            for cloud in topology.clouds
            for position in range(len(cloud.dp[0].ranks))
        ]
        # 4. EP groups: experts sharded dp x tp over each cloud instance, as
        # natively; the edge holds no experts but joins a group so every
        # rank is covered.
        ep_groups: list[LwdRankGroup] = [edge.dp[0].ranks for edge in topology.edges]
        ep_groups += [
            tuple(rank for dp in cloud.dp for rank in dp.ranks)
            for cloud in topology.clouds
        ]
        # 5. Sorted TP groups make every process create them in the same
        # order.
        return cls(
            world_size=topology.deployment.hccl_world_size,
            tp_groups=tuple(sorted((*edge_allocations, *cloud_allocations))),
            dp_groups=tuple(dp_groups),
            ep_groups=tuple(ep_groups),
            channel_endpoints=tuple(channel_endpoints),
        )

    @property
    def singleton_groups(self) -> tuple[LwdRankGroup, ...]:
        return tuple((rank,) for rank in range(self.world_size))

    def tp_group_of(self, rank: int) -> LwdRankGroup:
        for group in self.tp_groups:
            if rank in group:
                return group
        raise ValueError(f"[LWD] rank {rank} is not in [0, {self.world_size})")

    def channel_endpoints_of(self, rank: int) -> tuple[LwdChannelEndpoints, ...]:
        """The channel pairs this rank holds; empty for non-endpoint ranks."""
        return tuple(
            endpoints
            for endpoints in self.channel_endpoints
            if rank in (endpoints.edge_rank, endpoints.cloud_rank)
        )


def lwd_check_parallel_config(vllm_config: VllmConfig) -> None:
    """Reject at config time what the shared HCCL world cannot host.

    Every process reads the same topology, so both sides fail at startup,
    before any worker joins the world.

    Raises:
        NotImplementedError: A parallel feature or the topology is unsupported.
        ValueError: The tensor parallel size differs from the ranks of each DP
            of the local instance.
    """
    lwd_config = vllm_config.lwd_config
    assert lwd_config is not None
    parallel_config = vllm_config.parallel_config
    unsupported = [
        name
        for name, enabled in (
            ("pipeline parallelism", parallel_config.pipeline_parallel_size > 1),
            (
                "prefill context parallelism",
                parallel_config.prefill_context_parallel_size > 1,
            ),
            (
                "decode context parallelism",
                parallel_config.decode_context_parallel_size > 1,
            ),
            ("elastic EP", parallel_config.enable_elastic_ep),
            ("Engram", vllm_config.engram_config is not None),
        )
        if enabled
    ]
    if unsupported:
        raise NotImplementedError(
            f"[LWD] not supported in edge-cloud mode: {', '.join(unsupported)}"
        )
    if lwd_config.topology.is_host_comm:
        # Host comm has no shared HCCL world and no topology ranks; the
        # rank-allocation checks below are device-only. Each DP is an
        # independent engine launch, so the parallel sizes must match the
        # local instance's DP count instead.
        if parallel_config.data_parallel_size != len(lwd_config.instance.dp):
            raise ValueError(
                f"[LWD] data parallel size {parallel_config.data_parallel_size} "
                f"!= {len(lwd_config.instance.dp)} DPs of {lwd_config.role} "
                f"instance {lwd_config.instance_id} (host comm: one engine "
                "launch per DP)"
            )
        return
    # Raises for topologies the shared HCCL world cannot host.
    LwdParallelGroupsSpec.from_topology(lwd_config.topology)
    # The topology gives every DP of an instance the same rank count.
    dp_ranks = lwd_config.instance.dp[0].ranks
    if parallel_config.tensor_parallel_size != len(dp_ranks):
        raise ValueError(
            f"[LWD] tensor parallel size {parallel_config.tensor_parallel_size} "
            f"!= {len(dp_ranks)} ranks per DP of {lwd_config.role} instance "
            f"{lwd_config.instance_id}"
        )
