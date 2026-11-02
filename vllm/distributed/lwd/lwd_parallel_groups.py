# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank groups of the HCCL world shared by the LWD edge and cloud instances.

Design: docs/design/lwd_data_plane.md.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.config.engram import model_has_engram_layers
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.lwd.topology import LwdDP, LwdTopology

logger = init_logger(__name__)

LwdRankGroup = tuple[int, ...]

_LWD_ENGRAM_ARCHITECTURES = ("DeepseekV41ForCausalLM",)
"""Architectures whose Engram the edge hashes and the cloud looks up."""


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
    engram_dp_groups: tuple[LwdRankGroup, ...]
    """The host-local part of each DP group: one group per co-located rank
    and host, joining the DP replicas of that position placed on that host.

    An Engram table is held by the replicas of one host, which map one copy
    of it instead of one each. Replicas on different hosts cannot share
    memory, so they fall in different groups; with one replica per host every
    group is a singleton and each host holds a table of its own."""
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
        # 3b. Engram groups are the host-local part of the DP groups: only
        # co-located replicas can map one table instead of one each.
        engram_dp_groups: list[LwdRankGroup] = [
            (rank,) for edge in topology.edges for rank in edge.dp[0].ranks
        ]
        for cloud in topology.clouds:
            engram_dp_groups += _lwd_host_local_groups(cloud.dp)
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
            engram_dp_groups=tuple(engram_dp_groups),
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

    def dp_group_of(self, rank: int) -> LwdRankGroup:
        return self._group_of(self.dp_groups, rank, "DP")

    def engram_dp_group_of(self, rank: int) -> LwdRankGroup:
        return self._group_of(self.engram_dp_groups, rank, "Engram DP")

    def ep_group_of(self, rank: int) -> LwdRankGroup:
        return self._group_of(self.ep_groups, rank, "EP")

    def _group_of(
        self, groups: tuple[LwdRankGroup, ...], rank: int, kind: str
    ) -> LwdRankGroup:
        for group in groups:
            if rank in group:
                return group
        raise ValueError(
            f"[LWD] rank {rank} is in no {kind} group of [0, {self.world_size})"
        )

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
        NotImplementedError: A parallel feature, the topology or the Engram
            placement is unsupported.
        ValueError: The tensor parallel size differs from the ranks of each DP
            of the local instance, or the edge model kept n-gram layers.
    """
    lwd_config = vllm_config.lwd_config
    assert lwd_config is not None
    parallel_config = vllm_config.parallel_config
    engram_config = vllm_config.engram_config
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
            (
                "Engram embedding_across_dp",
                engram_config is not None and engram_config.embedding_across_dp,
            ),
        )
        if enabled
    ]
    if unsupported:
        raise NotImplementedError(
            f"[LWD] not supported in edge-cloud mode: {', '.join(unsupported)}"
        )
    # Raises for topologies the shared HCCL world cannot host.
    spec = LwdParallelGroupsSpec.from_topology(lwd_config.topology)
    # The topology gives every DP of an instance the same rank count.
    dp_ranks = lwd_config.instance.dp[0].ranks
    if parallel_config.tensor_parallel_size != len(dp_ranks):
        raise ValueError(
            f"[LWD] tensor parallel size {parallel_config.tensor_parallel_size} "
            f"!= {len(dp_ranks)} ranks per DP of {lwd_config.role} instance "
            f"{lwd_config.instance_id}"
        )
    _lwd_check_engram_layers(vllm_config)
    _lwd_check_engram_sharing(vllm_config, spec)


def _lwd_host_local_groups(dps: tuple[LwdDP, ...]) -> list[LwdRankGroup]:
    """Group the DP replicas of one instance by co-located rank and host.

    Native derives the same grouping from rank numbering, because a
    centralized deployment has only ranks to go by: it takes the greatest
    divisor of the DP count that keeps replicas whole within a node, then
    verifies the node boundaries with `in_the_same_node_as`
    (`_engram_dp_shard_size`). A topology names the host of every DP, so the
    address is the node identity and the grouping is direct.

    Groups of one host may differ in size from those of another, which the
    native shard size, a single number for the whole deployment, cannot
    express.
    """
    by_host: dict[str, list[LwdRankGroup]] = {}
    for dp in dps:
        by_host.setdefault(dp.addr, []).append(dp.ranks)
    return [
        tuple(sorted(replicas))
        for replicas_of_host in by_host.values()
        for replicas in zip(*replicas_of_host)
    ]


def _lwd_check_engram_sharing(
    vllm_config: VllmConfig, spec: LwdParallelGroupsSpec
) -> None:
    """DP replicas sharing a host hold their Engram table by sharing it.

    Without `dp_shared_memory` each replica of the group splits the table
    instead, so the shards become `TP x Engram DP` while the hash columns
    stay fixed: four replicas of four ranks make 16 shards of 24 columns,
    which the table rejects for uneven heads. Sharing keeps the split at TP
    and maps one host copy, which also keeps the driver's pinned pool from
    holding a copy per replica -- the pool, not host memory, is what runs out
    first. A host with a replica of its own has a group of one and nothing to
    share, which is why the check reads the group rather than the DP size.

    Reads the model, like `_lwd_check_engram_layers`, and demands the flag be
    set explicitly: a platform may resolve the Engram config after this check
    (Ascend does, in `check_and_update_config`), so an absent config here is
    one about to be defaulted with sharing off, not one that will stay
    absent.

    Raises:
        ValueError: Co-located DP replicas serve Engram without
            `dp_shared_memory`.
    """
    model_config = vllm_config.model_config
    if model_config is None or not model_has_engram_layers(model_config):
        return
    lwd_config = vllm_config.lwd_config
    assert lwd_config is not None
    shared = max(
        len(spec.engram_dp_group_of(rank))
        for dp in lwd_config.instance.dp
        for rank in dp.ranks
    )
    if shared == 1:
        return
    engram_config = vllm_config.engram_config
    if engram_config is not None and engram_config.dp_shared_memory:
        return
    raise ValueError(
        f"[LWD] {shared} data parallel replicas of {lwd_config.role} instance "
        f"{lwd_config.instance_id} share a host, which needs --engram-config "
        '\'{"cpu_offload":true,"dp_shared_memory":true}\': co-located replicas '
        "map one Engram table instead of splitting it over tensor parallel x "
        "Engram data parallel shards"
    )


def _lwd_check_engram_layers(vllm_config: VllmConfig) -> None:
    """Engram tables live on the cloud, for the architectures LWD hashes.

    Reads the model, not `engram_config`: a platform may resolve the Engram
    config after this check (Ascend does, in `check_and_update_config`).

    Raises:
        ValueError: The edge model kept its n-gram layers through the edge
            projection.
        NotImplementedError: The cloud model has n-gram layers of an
            architecture LWD does not hash.
    """
    model_config = vllm_config.model_config
    if model_config is None or not model_has_engram_layers(model_config):
        return
    lwd_config = vllm_config.lwd_config
    assert lwd_config is not None
    architecture = model_config.architecture
    if lwd_config.role == "edge":
        raise ValueError(
            f"[LWD] the edge {architecture} model keeps its n-gram embedding "
            "layers; the edge projection clears them unless hf_overrides is a "
            "dict that updates a sub-config: pass flat keys or a callable"
        )
    if architecture not in _LWD_ENGRAM_ARCHITECTURES:
        raise NotImplementedError(
            f"[LWD] Engram of {architecture} is not supported in edge-cloud "
            f"mode; supported: {_LWD_ENGRAM_ARCHITECTURES}"
        )
    logger.info("[LWD] cloud hosts the Engram tables of %s", architecture)
