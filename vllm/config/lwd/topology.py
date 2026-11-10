# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""File schema for LWD, independent of devices and distributed initialization."""

import json
from collections.abc import Hashable
from dataclasses import asdict
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
    model_validator,
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


def _validate_http_url(value: str | None) -> None:
    if value is not None:
        url = urlsplit(value)
        if url.scheme not in ("http", "https") or not url.hostname:
            raise ValueError("[LWD] control_url must be an HTTP(S) endpoint")


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
class LwdHostDataPlane:
    """Settings of the host (RoCE) data plane; only meaningful when
    ``deployment.data_plane_comm`` is ``host``."""

    up_buffer_size: LwdCount | None = None
    """UP (edge->cloud) block size in MB. Omitted = derived from the model
    (hidden size, hash layers, ``max_num_batched_tokens``) and rounded up
    to 32 MiB. Set it only to override the derivation: a value below the
    derived requirement is accepted with a startup warning, and a frame
    larger than the block is rejected at submit. Identical on both peers."""

    down_buffer_size: LwdCount = 32
    """DOWN (cloud->edge) block size in MB; identical on both peers."""

    deployment: Literal["in_process", "sidecar"] = "in_process"
    """"in_process" = the transport runs inside each engine process;
    "sidecar" = one dedicated transport daemon per host owns the
    listeners and engines attach through the ctl socket + shm. Both peers
    of a link must match."""


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdDeployment:
    mode: Annotated[int, Field(strict=True, ge=0, le=0)]
    scene: Literal["single_instance", "edge_share", "cloud_share", "lwd_cluster"]
    edges_num: LwdCount
    clouds_num: LwdCount
    hccl_world_size: LwdCount | None = None
    data_plane_comm: Literal["device", "host"] = "device"
    data_plane_host: LwdHostDataPlane | None = None
    """Host-comm settings block, required when ``data_plane_comm: host``
    and omitted otherwise — the extension point for future host-plane
    properties."""


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdQosDataPlane:
    """QoS of the LWD data plane (host comm: injected into UCX at link
    init, code-level, no env; device comm ignores these)."""

    traffic_class: Annotated[int, Field(strict=True, ge=0, le=255)] | None = None
    """RoCEv2 DSCP / IB Traffic Class; None = UCX default (auto -> 106
    for RoCEv2). Effective QoS needs matching switch PFC/ETS policy,
    usually paired with ``sl``; identical on both peers."""

    sl: Annotated[int, Field(strict=True, ge=0, le=15)] | None = None
    """RoCEv2 Ethernet priority (PCP) / IB Service Level; None = UCX
    default (0). Pair with ``traffic_class``; identical on both peers."""


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdQosCtrlPlane:
    """QoS of the LWD control plane (ZMQ links, DSCP-marked sockets)."""

    dscp: Annotated[int, Field(strict=True, ge=0, le=63)] | None = None
    """DSCP mark for control-plane traffic; None = leave sockets
    unmarked (OS default). Identical on both peers."""


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdQoS:
    """Cross-plane QoS policy of the deployment (top-level ``qos`` block)."""

    data_plane: LwdQosDataPlane | None = None
    ctrl_plane: LwdQosCtrlPlane | None = None


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
    control_url_map: dict[int, LwdString] = Field(default_factory=dict)
    placement_only: StrictBool = False
    tenant_key_file: LwdString | None = None
    consumer_id: LwdString | None = None
    # Cloud-side fields
    listen_host: LwdString = "0.0.0.0"
    listen_port: LwdPort = 8100
    # Shared
    connect_timeout: Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)] = (
        10.0
    )
    probe_timeout: Annotated[float, Field(strict=True, gt=5.0, allow_inf_nan=False)] = (
        25.0
    )
    enforce_mm_abi_match: StrictBool = True

    @model_validator(mode="before")
    @classmethod
    def normalize_control_url(cls, raw: Any) -> Any:
        if not isinstance(raw, dict) or not isinstance(raw.get("control_url"), dict):
            return raw
        if raw.get("control_url_map"):
            raise ValueError(
                "[LWD] control_url map and control_url_map are mutually exclusive"
            )
        return {**raw, "control_url": None, "control_url_map": raw["control_url"]}

    @field_validator("control_url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        _validate_http_url(value)
        return value

    @field_validator("control_url_map")
    @classmethod
    def validate_url_map(cls, value: dict[int, str]) -> dict[int, str]:
        for cloud_id, url in value.items():
            if cloud_id < 0:
                raise ValueError("[LWD] control_url_map keys must be non-negative")
            _validate_http_url(url)
        return value


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdDP:
    dp_idx: LwdId
    addr: Annotated[StrictStr, AfterValidator(_address)]
    ranks: tuple[LwdId, ...] = ()
    """Global ranks of this DP. Required when ``data_plane_comm`` is
    ``device``; omitted in host comm, where no global comm domain exists."""
    ctrl_port: LwdPort | None = None
    data_port: LwdPort | None = None


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


@dataclass(frozen=True, order=True)
class LwdConnectionId:
    """Stable identity of one edge -> cloud DP/engine connection."""

    edge_id: int
    cloud_id: int
    dp_idx: int


@dataclass(frozen=True)
class LwdDPConnection:
    """One explicit instance link expanded at matching DP indices."""

    edge_id: int
    cloud_id: int
    dp_idx: int
    edge: LwdDP
    cloud: LwdDP

    @property
    def connection_id(self) -> LwdConnectionId:
        return LwdConnectionId(self.edge_id, self.cloud_id, self.dp_idx)

    @property
    def cloud_leader_rank(self) -> int:
        # Device-comm only: host comm has no ranks (see LwdDP.ranks).
        if not self.cloud.ranks:
            raise ValueError(
                "cloud_leader_rank requires data_plane_comm=device "
                "(no ranks in host comm)"
            )
        return self.cloud.ranks[0]


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdTopology:
    deployment: LwdDeployment
    feature_ctrl: LwdFeatures
    edges: Annotated[tuple[LwdInstance, ...], Field(min_length=1)]
    clouds: Annotated[tuple[LwdInstance, ...], Field(min_length=1)]
    instance_links: Annotated[tuple[LwdLink, ...], Field(min_length=1)]
    qos: LwdQoS | None = None
    control_plane: LwdControlPlaneConfig = Field(default_factory=LwdControlPlaneConfig)
    digest: str = Field(default="", init=False)
    """First 16 hex characters of the canonical topology SHA-256 digest."""

    @classmethod
    def from_file(cls, path: str) -> "LwdTopology":
        try:
            file_bytes = Path(path).read_bytes()
            raw = yaml.load(file_bytes.decode("utf-8"), Loader=_UniqueKeyLoader)
            return cls.from_dict(raw)
        except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
            raise ValueError(f"[LWD] Invalid topology file {path!r}: {exc}") from exc

    @classmethod
    def from_dict(cls, raw: Any) -> "LwdTopology":
        topology = TypeAdapter(cls).validate_python(raw)
        topology.validate_layout()
        # Hash the validated semantic snapshot, never YAML spelling/layout.
        # Exclude the derived digest field itself from the canonical payload.
        canonical = asdict(topology)
        canonical.pop("digest", None)
        canonical["instance_links"] = sorted(
            canonical["instance_links"],
            key=lambda link: (link["edge"], link["cloud"]),
        )
        encoded = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        object.__setattr__(topology, "digest", sha256(encoded).hexdigest()[:16])
        return topology

    @property
    def is_host_comm(self) -> bool:
        """True when ``data_plane_comm`` is host: edge and cloud are separate worlds."""
        return self.deployment.data_plane_comm == "host"

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
        if self.is_host_comm:
            # No shared HCCL world: the host-specific layout rules apply
            # instead of the rank-allocation ones.
            self._validate_host_layout()
        else:
            self._validate_ranks()
        self._validate_control_ports()

    def _validate_host_layout(self) -> None:
        """Layout rules of data_plane_comm=host (no global comm domain).

        Ranks are meaningless and must be omitted everywhere; the cloud
        is the only listener (one shared listener per cloud DP, edges
        connect out), so data_port is cloud-only.
        """
        for instance in (*self.edges, *self.clouds):
            if any(dp.ranks for dp in instance.dp):
                raise ValueError(
                    "[LWD] ranks must be omitted when "
                    "data_plane_comm=host (no global comm domain; "
                    "links are established from addr/data_port)"
                )
        if self.deployment.hccl_world_size is not None:
            raise ValueError(
                "[LWD] hccl_world_size must be omitted when "
                "data_plane_comm=host (the edge and cloud are "
                "separate worlds; their sizes come from their own "
                "launches)"
            )
        for instance in self.edges:
            if any(dp.data_port is not None for dp in instance.dp):
                raise ValueError(
                    "[LWD] data_port is cloud-only (edges connect out and never listen)"
                )
        by_addr: dict[str, list[int]] = {}
        for instance in self.clouds:
            for dp in instance.dp:
                if dp.data_port is None:
                    raise ValueError(
                        "[LWD] data_port is required on every cloud DP "
                        "when data_plane_comm=host (ctrl_port is "
                        "control-plane only)"
                    )
                by_addr.setdefault(dp.addr, []).append(dp.data_port)
        for addr, ports in by_addr.items():
            if len(ports) != len(set(ports)):
                raise ValueError(f"[LWD] duplicate data_port on {addr}")

    def _validate_ranks(self) -> None:
        if self.deployment.data_plane_host is not None:
            raise ValueError(
                "[LWD] data_plane_host is host-comm-only and must be "
                "omitted when data_plane_comm=device"
            )
        if self.deployment.hccl_world_size is None:
            raise ValueError(
                "[LWD] hccl_world_size is required when "
                "data_plane_comm=device (the edge and cloud share one world)"
            )
        for instance in (*self.edges, *self.clouds):
            for dp in instance.dp:
                if not dp.ranks:
                    raise ValueError(
                        "[LWD] ranks are required on every DP when "
                        "data_plane_comm=device"
                    )
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
        world_size = self.deployment.hccl_world_size
        assert world_size is not None  # checked above
        if ranks != set(range(world_size)):
            raise ValueError(
                "[LWD] Unique ranks must cover [0, hccl_world_size) without gaps"
            )

    def _validate_control_ports(self) -> None:
        # Every cloud DP binds its ctrl_port on all interfaces, so two DPs on
        # the same address must not share a port. Ports may repeat across
        # addresses (their namespaces are independent), which also covers
        # edge_share clouds hosted on distinct machines.
        dps = [dp for cloud in self.clouds for dp in cloud.dp]
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
        if scene == "lwd_cluster" and (edge_count < 1 or cloud_count < 2):
            raise ValueError("[LWD] lwd_cluster requires at least two clouds")
        # Coordination may be enabled explicitly in every scene. Shared and
        # multi-cloud scenes require it for every instance.
        if scene in ("cloud_share", "lwd_cluster"):
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
