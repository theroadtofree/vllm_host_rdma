# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""File schema for LWD, independent of devices and distributed initialization."""

from collections.abc import Hashable
from hashlib import sha256
from ipaddress import ip_address
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    TypeAdapter,
    field_validator,
)
from pydantic.dataclasses import dataclass

LwdId = Annotated[int, Field(strict=True, ge=0)]
LwdCount = Annotated[int, Field(strict=True, ge=1)]
LwdPort = Annotated[int, Field(strict=True, ge=1, le=65535)]
LwdString = Annotated[StrictStr, Field(pattern=r"\S")]


def _address(value: str) -> str:
    address = ip_address(value)
    if address.is_unspecified or address.is_multicast:
        raise ValueError("[LWD] addr must be a unicast endpoint IP")
    return str(address)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate mapping keys instead of silently replacing values."""

    def construct_mapping(
        self, node: yaml.MappingNode, deep: bool = False
    ) -> dict[Hashable, Any]:
        result: dict[Hashable, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("[LWD] YAML mapping keys must be strings")
            if key in result:
                raise ValueError(f"[LWD] duplicate YAML key: {key!r}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdDeployment:
    mode: Annotated[int, Field(strict=True, ge=0, le=0)]
    scene: Literal["single_instance", "edge_share", "cloud_share", "lwd_cluster"]
    hccl_world_size: LwdCount
    edges_num: LwdCount
    clouds_num: LwdCount


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdScramble:
    enable: StrictBool = False
    device_id: LwdString | None = None
    custom_factor: LwdString | None = None


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdFeatures:
    enable_early_recv: StrictBool = False
    scramble: LwdScramble = Field(default_factory=LwdScramble)


LwdPositiveSeconds = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdControlPlaneConfig:
    """Edge-cloud control link tunables, shared by every instance."""

    send_queue_max: Annotated[int, Field(strict=True, ge=1)] = 1000
    """Outbound messages buffered per DP rank before sending reports
    backpressure."""
    hello_retry_interval_s: LwdPositiveSeconds = 5.0
    """Seconds between edge registration attempts until the cloud answers."""
    handshake_timeout_s: LwdPositiveSeconds = 1800.0
    """Seconds an edge waits for its cloud; must cover a full cloud engine
    startup (weights, KV cache, graph capture)."""


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdCoordination:
    """Prefix-cache coordination block, embedded per instance for cloud_share."""

    enabled: StrictBool = False
    # Edge-side fields
    control_url: LwdString | None = None
    tenant_key_file: LwdString | None = None
    consumer_id: LwdString | None = None
    # Cloud-side fields
    listen_host: LwdString = "0.0.0.0"
    listen_port: LwdPort = 8100
    # Shared
    connect_timeout: Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)] = (
        10.0
    )

    @field_validator("control_url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        if value is not None:
            url = urlsplit(value)
            if url.scheme not in ("http", "https") or not url.hostname:
                raise ValueError("[LWD] control_url must be an HTTP(S) endpoint")
        return value


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdDP:
    dp_idx: LwdId
    addr: Annotated[StrictStr, AfterValidator(_address)]
    ranks: Annotated[tuple[LwdId, ...], Field(min_length=1)]
    ctrl_port: LwdPort | None = None
    """Control-plane (ZMQ) port of this DP; not used by the data plane."""
    data_port: LwdPort | None = None
    """UCX data-plane listener port of this DP (family=ucx only), used
    exactly as configured. Only CLOUD DPs listen: each cloud DP binds
    one shared listener (both its UP/DOWN links attach to it, routed by
    handshake token); edges connect out and never listen. Required on
    every cloud DP when ``data_plane_family: ucx``, omitted on edges;
    unused by hccl."""


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdInstance:
    id: LwdId
    dp: Annotated[tuple[LwdDP, ...], Field(min_length=1)]
    coordination: LwdCoordination | None = None

    @field_validator("dp")
    @classmethod
    def order_dps(cls, dps: tuple[LwdDP, ...]) -> tuple[LwdDP, ...]:
        dps = tuple(sorted(dps, key=lambda dp: dp.dp_idx))
        if [dp.dp_idx for dp in dps] != list(range(len(dps))):
            raise ValueError("[LWD] dp_idx must be unique and contiguous from 0")
        return dps


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdLink:
    edge: LwdId
    cloud: LwdId


@dataclass(frozen=True)
class LwdDPConnection:
    """One explicit instance link expanded at matching DP indices."""

    edge_id: int
    cloud_id: int
    dp_idx: int
    edge: LwdDP
    cloud: LwdDP

    @property
    def cloud_leader_rank(self) -> int:
        return self.cloud.ranks[0]


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdTopology:
    deployment: LwdDeployment
    feature_ctrl: LwdFeatures
    edges: Annotated[tuple[LwdInstance, ...], Field(min_length=1)]
    clouds: Annotated[tuple[LwdInstance, ...], Field(min_length=1)]
    instance_links: Annotated[tuple[LwdLink, ...], Field(min_length=1)]
    data_plane_family: Literal["hccl", "ucx"] = "hccl"
    """Cross-node transport of the LWD data plane: HCCL P2P over the
    shared torch world (default) or UCX host-RoCE links (vllm's
    ``vllm/v1/lwd_comm`` + vllm-ascend's ``lwd_data_plane/ucx``). Both
    peers of a link must select the same family."""
    control_plane: LwdControlPlaneConfig = Field(default_factory=LwdControlPlaneConfig)
    digest: str = Field(default="", init=False)
    """First 16 hex characters of the source file digest; empty for from_dict."""

    @classmethod
    def from_file(cls, path: str) -> "LwdTopology":
        try:
            file_bytes = Path(path).read_bytes()
            raw = yaml.load(file_bytes.decode("utf-8"), Loader=_UniqueKeyLoader)
            topology = cls.from_dict(raw)
            # The digest belongs to the loaded snapshot, not to YAML input.
            object.__setattr__(topology, "digest", sha256(file_bytes).hexdigest()[:16])
            return topology
        except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
            raise ValueError(f"[LWD] Invalid topology file {path!r}: {exc}") from exc

    @classmethod
    def from_dict(cls, raw: Any) -> "LwdTopology":
        topology = TypeAdapter(cls).validate_python(raw)
        topology.validate_layout()
        return topology

    @field_validator("edges", "clouds")
    @classmethod
    def order_instances(
        cls, instances: tuple[LwdInstance, ...]
    ) -> tuple[LwdInstance, ...]:
        instances = tuple(sorted(instances, key=lambda instance: instance.id))
        if [instance.id for instance in instances] != list(range(len(instances))):
            raise ValueError("[LWD] Instance IDs must be unique and contiguous from 0")
        return instances

    def validate_layout(self) -> None:
        """Validate instance links and placement without initializing communication."""
        if (self.deployment.edges_num, self.deployment.clouds_num) != (
            len(self.edges),
            len(self.clouds),
        ):
            raise ValueError(
                "[LWD] deployment instance counts do not match edges/clouds"
            )
        if len(set(self.instance_links)) != len(self.instance_links):
            raise ValueError("[LWD] duplicate instance link")
        if any(
            link.edge >= len(self.edges) or link.cloud >= len(self.clouds)
            for link in self.instance_links
        ):
            raise ValueError("[LWD] instance_links: unknown edge/cloud instance")
        for instances, cloud in ((self.edges, False), (self.clouds, True)):
            if any(
                (dp.ctrl_port is not None) != cloud for i in instances for dp in i.dp
            ):
                raise ValueError(
                    "[LWD] ctrl_port is required on clouds and omitted on edges"
                )
        self._validate_scene()
        if len({len(instance.dp) for instance in (*self.edges, *self.clouds)}) != 1:
            raise ValueError(
                "[LWD] Instances must have matching DP indices and the same DP count"
            )
        if self.data_plane_family == "ucx":
            # Only cloud DPs listen (one shared listener per DP; edges
            # connect out), mirroring the ctrl_port placement rule.
            for instance in self.edges:
                if any(dp.data_port is not None for dp in instance.dp):
                    raise ValueError(
                        "[LWD] data_port is cloud-only (edges connect out "
                        "and never listen)"
                    )
            for instance in self.clouds:
                for dp in instance.dp:
                    if dp.data_port is None:
                        raise ValueError(
                            "[LWD] data_port is required on every cloud DP "
                            "when data_plane_family=ucx (ctrl_port is "
                            "control-plane only)"
                        )
            by_addr: dict[str, list[int]] = {}
            for instance in self.clouds:
                for dp in instance.dp:
                    by_addr.setdefault(dp.addr, []).append(dp.data_port)
            for addr, ports in by_addr.items():
                if len(ports) != len(set(ports)):
                    raise ValueError(
                        f"[LWD] duplicate data_port on {addr}"
                    )
        self._validate_ranks()
        self._validate_control_ports()

    def _validate_ranks(self) -> None:
        for edge in self.edges:
            if len({(dp.addr, dp.ranks) for dp in edge.dp}) != 1:
                raise ValueError(
                    "[LWD] Edge DPs must share their instance's ranks/address"
                )
        for cloud in self.clouds:
            if len({len(dp.ranks) for dp in cloud.dp}) != 1:
                raise ValueError(
                    "[LWD] Cloud instance must split its ranks equally among DPs"
                )
        # Edge DPs reuse an allocation; edge_share also reuses it across instances.
        edges = self.edges
        if self.deployment.scene == "edge_share":
            if len({(edge.dp[0].addr, edge.dp[0].ranks) for edge in edges}) != 1:
                raise ValueError(
                    "[LWD] edge_share instances must share edge ranks/address"
                )
            edges = edges[:1]
        allocations = [edge.dp[0] for edge in edges]
        allocations.extend(dp for cloud in self.clouds for dp in cloud.dp)
        ranks: set[int] = set()
        for dp in allocations:
            if len(set(dp.ranks)) != len(dp.ranks) or ranks.intersection(dp.ranks):
                raise ValueError(
                    "[LWD] Separate allocations require distinct ranks; "
                    "duplicate or overlapping ranks"
                )
            ranks.update(dp.ranks)
        if ranks != set(range(self.deployment.hccl_world_size)):
            raise ValueError(
                "[LWD] Unique ranks must cover [0, hccl_world_size) without gaps"
            )

    def _validate_control_ports(self) -> None:
        dps = [dp for cloud in self.clouds for dp in cloud.dp]
        if self.deployment.scene == "edge_share":
            # These are shared per-DP channel references, not cloud bind addresses.
            ports: dict[int, int | None] = {}
            for dp in dps:
                if ports.setdefault(dp.dp_idx, dp.ctrl_port) != dp.ctrl_port:
                    raise ValueError(
                        "[LWD] edge_share requires the same ctrl_port for "
                        "the same DP across all cloud instances"
                    )
            if len(set(ports.values())) != len(ports):
                raise ValueError(
                    "[LWD] edge_share requires different ctrl_port values per DP"
                )
            return
        if len({(dp.addr, dp.ctrl_port) for dp in dps}) != len(dps):
            raise ValueError(
                "[LWD] Cloud DPs on the same address must use "
                "different ctrl_port values"
            )

    def _validate_scene(self) -> None:
        """Validate logical instance shape, independently of physical host count."""
        scene = self.deployment.scene
        edge_count, cloud_count = len(self.edges), len(self.clouds)
        if scene == "single_instance" and (edge_count, cloud_count) != (1, 1):
            raise ValueError("[LWD] single_instance requires one edge and one cloud")
        if scene == "cloud_share" and (edge_count < 2 or cloud_count != 1):
            raise ValueError("[LWD] cloud_share requires multiple edges and one cloud")
        if scene == "lwd_cluster" and (edge_count < 2 or cloud_count < 2):
            raise ValueError("[LWD] lwd_cluster requires multiple edges and clouds")
        # coordination only exists for cloud_share, embedded in each instance
        if scene == "cloud_share":
            for edge in self.edges:
                if edge.coordination is None or not edge.coordination.enabled:
                    raise ValueError(
                        "[LWD] cloud_share requires each edge instance to have "
                        "an enabled coordination block"
                    )
            for cloud in self.clouds:
                if cloud.coordination is None or not cloud.coordination.enabled:
                    raise ValueError(
                        "[LWD] cloud_share requires each cloud instance to have "
                        "an enabled coordination block"
                    )
        else:
            for instance in (*self.edges, *self.clouds):
                if instance.coordination is not None:
                    raise ValueError(
                        f"[LWD] coordination is only allowed in cloud_share scene, "
                        f"got scene={scene}"
                    )
        if scene == "edge_share":
            if edge_count != cloud_count or cloud_count < 2:
                raise ValueError(
                    "[LWD] edge_share requires equal multiple edge/cloud IDs"
                )
            if (
                len(self.instance_links) != edge_count
                or len({link.edge for link in self.instance_links}) != edge_count
                or len({link.cloud for link in self.instance_links}) != cloud_count
            ):
                raise ValueError("[LWD] edge_share requires one-to-one instance links")
        elif set(self.instance_links) != {
            LwdLink(edge.id, cloud.id) for edge in self.edges for cloud in self.clouds
        }:
            raise ValueError(
                "[LWD] Scene requires full-mesh instance_links; "
                "missing pair or isolated instance"
            )

    def connections(self, role: str, instance_id: int) -> tuple[LwdDPConnection, ...]:
        """Describe every DP connection for an instance without opening sockets."""
        self.instance(role, instance_id)
        result = []
        for link in self.instance_links:
            local_id = link.edge if role == "edge" else link.cloud
            if local_id != instance_id:
                continue
            for edge_dp, cloud_dp in zip(
                self.edges[link.edge].dp, self.clouds[link.cloud].dp
            ):
                result.append(
                    LwdDPConnection(
                        link.edge,
                        link.cloud,
                        edge_dp.dp_idx,
                        edge_dp,
                        cloud_dp,
                    )
                )
        return tuple(result)

    def instance(self, role: str, instance_id: int) -> LwdInstance:
        if role not in ("edge", "cloud"):
            raise ValueError("[LWD] role must be 'edge' or 'cloud'")
        instances = self.edges if role == "edge" else self.clouds
        for instance in instances:
            if instance.id == instance_id:
                return instance
        raise ValueError(
            f"[LWD] Instance not found: role={role}, instance_id={instance_id}"
        )
