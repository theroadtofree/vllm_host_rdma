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
    StrictBool,
    StrictStr,
    TypeAdapter,
    field_validator,
    model_validator,
)
from pydantic.dataclasses import dataclass

LwdId = Annotated[int, Field(strict=True, ge=0)]
LwdCount = Annotated[int, Field(strict=True, ge=1)]
LwdPort = Annotated[int, Field(strict=True, ge=1, le=65535)]
LwdString = Annotated[StrictStr, Field(pattern=r"\S")]
LwdPositiveSeconds = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]


def _canonical_json(payload: dict[str, Any]) -> bytes:
    """Deterministic encoding of a validated semantic snapshot: hashing
    this (never the YAML spelling) is what makes the digests stable under
    formatting and key order."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _address(value: str) -> str:
    address = ip_address(value)
    if address.is_unspecified or address.is_multicast:
        raise ValueError("[LWD] addr must be a unicast endpoint IP")
    return str(address)


def _reject_legacy_coordination_fields(raw: Any) -> Any:
    if isinstance(raw, dict):
        retired = {"control_url", "control_url_map", "listen_port"}.intersection(raw)
        if retired:
            names = ", ".join(sorted(retired))
            raise ValueError(
                f"[LWD] {names} are retired; configure cloud DP "
                "coord_port/coord_url instead"
            )
    return raw


def _validate_coord_url(value: str, coord_port: int | None) -> None:
    if any(char.isspace() for char in value):
        raise ValueError("[LWD] coord_url must not contain whitespace")
    try:
        url = urlsplit(value)
        port = url.port
    except ValueError as exc:
        raise ValueError(f"[LWD] invalid coord_url: {exc}") from exc
    if url.scheme not in ("http", "https") or not url.hostname:
        raise ValueError("[LWD] coord_url must be an HTTP(S) endpoint with a host")
    try:
        address = ip_address(url.hostname)
    except ValueError:
        # DNS names are supported for NAT, proxies and external gateways.
        pass
    else:
        if address.is_unspecified or address.is_multicast:
            raise ValueError("[LWD] coord_url host must be a routable endpoint")
    if coord_port is None:
        raise ValueError("[LWD] coord_url requires coord_port on the same DP")
    if port is None or port < 1:
        raise ValueError("[LWD] coord_url must include a valid explicit port")


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


LWD_API_BIND_BASE_PORT_DEFAULT = 61000
"""Default ``deployment.api_bind_base_port``: the reserved edge_share
engine IO band then occupies [61000, 61015]."""

LWD_API_MAX_FRONTENDS = 8
"""Frontend capacity of the edge_share engine IO band: two reserved
ports per frontend (``api_bind_base_port + 2*idx / +1``), 16 in total.
The cap is enforced where ports are actually occupied (the
``engine_io_addresses`` derivation: bind side, pre-seed and
registration authority all derive through it), NOT at config time —
the yaml may describe larger plans."""


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
    api_bind_base_port: Annotated[int, Field(strict=True, ge=1024, le=65520)] = (
        LWD_API_BIND_BASE_PORT_DEFAULT
    )
    """Base of the reserved edge_share engine IO port band. Attach
    frontend ``idx`` binds ``base + 2*idx`` (input ROUTER) and ``+1``
    (output PULL); the whole band ``[base, base + 2*8 - 1]`` is reserved
    for up to ``LWD_API_MAX_FRONTENDS`` frontends, so deployment ports
    (``--port``, master/registration ports) must stay out of it. The
    fixed derivation is what makes a frontend restart rebind the SAME
    ports, letting the running engine's DEALER/PUSH reconnect
    transparently. Only meaningful — and only configurable — for
    ``scene: edge_share``."""

    @model_validator(mode="after")
    def _validate_api_bind_base_port(self) -> "LwdDeployment":
        if (
            self.scene != "edge_share"
            and self.api_bind_base_port != LWD_API_BIND_BASE_PORT_DEFAULT
        ):
            raise ValueError(
                "[LWD] deployment.api_bind_base_port is an edge_share "
                f"setting and must not be configured for scene="
                f"{self.scene!r}"
            )
        return self


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdQosDataPlane:
    """QoS of the LWD data plane (host comm: injected into UCX at link
    init, code-level, no env; device comm ignores these)."""

    traffic_class: Annotated[int, Field(strict=True, ge=0, le=255)] | None = None
    """IB traffic-class byte, i.e. RoCEv2 ``DSCP << 2 | ECN`` (138 is DSCP
    34 + ECN 2). None = UCX default (auto -> 106 for RoCEv2). Host comm
    defaults it to 138. Effective QoS needs matching switch PFC/ETS policy,
    usually paired with ``sl``; identical on both peers."""

    sl: Annotated[int, Field(strict=True, ge=0, le=15)] | None = None
    """RoCEv2 Ethernet priority (PCP) / IB Service Level; None = UCX
    default (0). Host comm defaults it to 4. Pair with ``traffic_class``;
    identical on both peers."""


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdQosCtrlPlane:
    """QoS of the LWD control plane (ZMQ links, DSCP-marked sockets)."""

    dscp: Annotated[int, Field(strict=True, ge=0, le=63)] | None = None
    """DSCP mark for control-plane traffic (48 = CS6); None = leave
    sockets unmarked (OS default). Host comm defaults it to 48. Identical
    on both peers."""


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdQoS:
    """Cross-plane QoS policy of the deployment (top-level ``qos`` block).

    With ``data_plane_comm: host`` the planes this block omits fall back to
    the host policy below; device (HCCL) comm keeps them unset.
    """

    data_plane: LwdQosDataPlane | None = None
    ctrl_plane: LwdQosCtrlPlane | None = None


# QoS defaults of the host (RoCE) data plane, applied when the topology omits
# ``qos`` or one of its planes: RoCE QoS is a contract with the switch
# (PFC/ETS steer on DSCP + PCP), so the host plane ships one policy instead of
# leaving each deployment to invent its own. Device (HCCL) comm keeps the
# historical behaviour, where nothing is marked.
_HOST_QOS_TRAFFIC_CLASS = 138  # IB TC byte: DSCP 34 (AF41) << 2 | ECN 2
_HOST_QOS_SL = 4  # RoCEv2 PCP / IB SL
_HOST_QOS_CTRL_DSCP = 48  # CS6


def _host_qos_defaults(qos: LwdQoS | None) -> LwdQoS:
    """Fill every qos plane the host topology left out.

    A plane that is present is kept as written: the operator owns it then,
    so a None field inside it still means "leave it to UCX/the OS".
    """
    data_plane = qos.data_plane if qos is not None else None
    ctrl_plane = qos.ctrl_plane if qos is not None else None
    if data_plane is None:
        data_plane = LwdQosDataPlane(
            traffic_class=_HOST_QOS_TRAFFIC_CLASS, sl=_HOST_QOS_SL
        )
    if ctrl_plane is None:
        ctrl_plane = LwdQosCtrlPlane(dscp=_HOST_QOS_CTRL_DSCP)
    return LwdQoS(data_plane=data_plane, ctrl_plane=ctrl_plane)


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdScramble:
    enable: StrictBool = False
    device_id: LwdString | None = None
    custom_factor: LwdString | None = None


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdFeatures:
    enable_early_recv: StrictBool = False
    scramble: LwdScramble = Field(default_factory=LwdScramble)


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
    placement_only: StrictBool = False
    tenant_key_file: LwdString | None = None
    consumer_id: LwdString | None = None
    # Cloud-side fields
    listen_host: LwdString = "0.0.0.0"
    probe_poll_budget: Annotated[int, Field(strict=True, gt=0)] = 32
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
    def reject_legacy_fields(cls, raw: Any) -> Any:
        return _reject_legacy_coordination_fields(raw)


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdDP:
    dp_idx: LwdId
    addr: Annotated[StrictStr, AfterValidator(_address)]
    ranks: tuple[LwdId, ...] = ()
    """Global ranks of this DP. Required when ``data_plane_comm`` is
    ``device``; omitted in host comm, where no global comm domain exists."""
    ctrl_port: LwdPort | None = None
    data_port: LwdPort | None = None
    coord_port: LwdPort | None = None
    coord_url: LwdString | None = None
    """Advertised HTTP coordination endpoint; derived from addr/coord_port
    when omitted. Explicit URLs support DNS, NAT and gateways whose external
    port differs from coord_port; multiple DPs may share a gateway URL."""

    @model_validator(mode="after")
    def normalize_coord_url(self) -> "LwdDP":
        if self.coord_url is None:
            if self.coord_port is not None:
                host = f"[{self.addr}]" if ":" in self.addr else self.addr
                object.__setattr__(
                    self, "coord_url", f"http://{host}:{self.coord_port}"
                )
        else:
            _validate_coord_url(self.coord_url, self.coord_port)
        return self


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

    connection_id: LwdConnectionId
    edge: LwdDP
    cloud: LwdDP

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
class LwdGateway:
    """Explicit Edge-to-gateway entrypoint; Cloud selection stays at the gateway."""

    enabled: StrictBool = False
    url: LwdString | None = None
    connect_timeout: LwdPositiveSeconds = 10.0

    @model_validator(mode="after")
    def validate_url(self) -> "LwdGateway":
        if self.enabled and self.url is None:
            raise ValueError("[LWD] enabled gateway requires a URL")
        if self.url is None:
            return self
        value = self.url
        if any(char.isspace() for char in value):
            raise ValueError("[LWD] gateway URL must not contain whitespace")
        try:
            url = urlsplit(value)
            _ = url.port
        except ValueError as exc:
            raise ValueError(f"[LWD] invalid gateway URL: {exc}") from exc
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or not url.path
            or url.path == "/"
            or url.fragment
            or url.username is not None
        ):
            raise ValueError(
                "[LWD] gateway.url must be a complete HTTP(S) URL "
                "with a non-root path and without credentials or fragments"
            )
        return self


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdTopology:
    deployment: LwdDeployment
    feature_ctrl: LwdFeatures
    edges: Annotated[tuple[LwdInstance, ...], Field(min_length=1)]
    clouds: Annotated[tuple[LwdInstance, ...], Field(min_length=1)]
    gateway: LwdGateway = Field(default_factory=LwdGateway)
    instance_links: Annotated[tuple[LwdLink, ...], Field(min_length=1)] | None = None
    qos: LwdQoS | None = None
    control_plane: LwdControlPlaneConfig = Field(default_factory=LwdControlPlaneConfig)
    digest: str = Field(default="", init=False)
    """First 16 hex characters of the canonical topology SHA-256 digest."""
    frontend_digest: str = Field(default="", init=False)
    """First 16 hex of the canonical SHA-256 over everything EXCEPT the
    instance inventory (edges, clouds, links and the instance-count
    declarations): the subset every attached frontend must match. An
    elastic addition changes only the inventory, so this digest stays
    stable across increments while ``digest`` moves with them."""

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
        if topology.instance_links is None:
            object.__setattr__(
                topology,
                "instance_links",
                tuple(
                    LwdLink(edge.id, cloud.id)
                    for edge in topology.edges
                    for cloud in topology.clouds
                ),
            )
        topology.validate_layout()
        # Hash the validated semantic snapshot, never YAML spelling/layout.
        # Exclude the derived digest fields themselves from the payload.
        canonical = asdict(topology)
        canonical.pop("digest", None)
        canonical.pop("frontend_digest", None)
        canonical["instance_links"] = sorted(
            canonical["instance_links"],
            key=lambda link: (link["edge"], link["cloud"]),
        )
        object.__setattr__(
            topology, "digest", sha256(_canonical_json(canonical)).hexdigest()[:16]
        )
        frontend = {
            key: value
            for key, value in canonical.items()
            if key not in ("edges", "clouds", "instance_links")
        }
        frontend["deployment"] = {
            key: value
            for key, value in canonical["deployment"].items()
            if key not in ("edges_num", "clouds_num")
        }
        object.__setattr__(
            topology,
            "frontend_digest",
            sha256(_canonical_json(frontend)).hexdigest()[:16],
        )
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

    @model_validator(mode="after")
    def fill_host_qos(self) -> "LwdTopology":
        """Host comm ships one RoCE QoS policy: fill the planes it left out.

        Runs during validation, so every construction path gets the defaults
        and the digest snapshots the effective policy.
        """
        if self.is_host_comm:
            object.__setattr__(self, "qos", _host_qos_defaults(self.qos))
        return self

    def validate_layout(self) -> None:
        """Validate instance links and placement without initializing communication."""
        assert self.instance_links is not None
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
        self._validate_coordination()
        self._validate_gateway()
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

    def _validate_coordination(self) -> None:
        """Require a complete advertised endpoint on every coordinated cloud DP."""
        assert self.instance_links is not None
        for cloud in self.clouds:
            if cloud.coordination is not None and cloud.coordination.enabled:
                for dp in cloud.dp:
                    if dp.coord_port is None or dp.coord_url is None:
                        raise ValueError(
                            "[LWD] enabled cloud coordination requires "
                            "coord_port on every DP"
                        )
        for edge in self.edges:
            if edge.coordination is None or not edge.coordination.enabled:
                continue
            linked_clouds = (
                self.clouds[link.cloud]
                for link in self.instance_links
                if link.edge == edge.id
            )
            if any(
                dp.coord_port is None or dp.coord_url is None
                for cloud in linked_clouds
                for dp in cloud.dp
            ):
                raise ValueError(
                    "[LWD] enabled edge coordination requires coord_port "
                    "on every linked cloud DP"
                )

    def _validate_gateway(self) -> None:
        if not self.gateway.enabled:
            return
        for instance in (*self.edges, *self.clouds):
            coordination = instance.coordination
            if coordination is None or not coordination.enabled:
                raise ValueError(
                    "[LWD] enabled gateway requires enabled coordination "
                    "on every Edge and Cloud instance"
                )

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
        if self.deployment.scene == "edge_share":
            for cloud_dps in zip(*(cloud.dp for cloud in self.clouds)):
                if (
                    len({dp.addr for dp in cloud_dps}) == len(cloud_dps)
                    and len({dp.ctrl_port for dp in cloud_dps}) != 1
                ):
                    raise ValueError(
                        "[LWD] edge_share clouds on distinct addresses must use "
                        "the same ctrl_port per DP"
                    )
        coord_ports = [
            (dp.addr, dp.coord_port) for dp in dps if dp.coord_port is not None
        ]
        if len(set(coord_ports)) != len(coord_ports):
            raise ValueError(
                "[LWD] Cloud DPs on the same address must use "
                "different coord_port values"
            )
        listeners = {
            (dp.addr, port): f"{kind}_port"
            for dp in dps
            for kind, port in (("ctrl", dp.ctrl_port), ("data", dp.data_port))
            if port is not None
        }
        for dp in dps:
            if dp.coord_port is None:
                continue
            conflict = listeners.get((dp.addr, dp.coord_port))
            if conflict is not None:
                raise ValueError(
                    f"[LWD] coord_port conflicts with {conflict} on {dp.addr}"
                )

    def _validate_scene(self) -> None:
        """Validate logical instance shape, independently of physical host count."""
        assert self.instance_links is not None
        scene = self.deployment.scene
        edge_count, cloud_count = len(self.edges), len(self.clouds)
        if scene == "single_instance" and (edge_count, cloud_count) != (1, 1):
            raise ValueError("[LWD] single_instance requires one edge and one cloud")
        # A single-edge cloud_share is the degenerate form of the shared
        # cloud pipeline: it runs the same coordination machinery (the
        # scene below still requires enabled coordination blocks) and the
        # same multi-edge request routing, so a deployment can verify the
        # whole cloud_share pipeline with one instance and grow by adding
        # edges — mirroring the degenerate single-pair edge_share form.
        if scene == "cloud_share" and (edge_count < 1 or cloud_count != 1):
            raise ValueError(
                "[LWD] cloud_share requires at least one edge and one cloud"
            )
        # lwd_cluster has no minimum-count check beyond the type-level
        # edges_num/clouds_num >= 1: a 1-edge + 1-cloud deployment is the
        # degenerate form of the full-mesh pipeline, usable for verification
        # and gradual scale-out.
        # Coordination may be enabled explicitly in every scene. Shared and
        # multi-cloud scenes require it for every instance.
        if scene in ("cloud_share", "lwd_cluster"):
            for edge in self.edges:
                if edge.coordination is None or not edge.coordination.enabled:
                    raise ValueError(
                        f"[LWD] {scene} requires each edge instance to have "
                        "an enabled coordination block"
                    )
            for cloud in self.clouds:
                if cloud.coordination is None or not cloud.coordination.enabled:
                    raise ValueError(
                        f"[LWD] {scene} requires each cloud instance to have "
                        "an enabled coordination block"
                    )
        if scene == "edge_share":
            # A single pair is the degenerate form of the shared-edge
            # pipeline: it runs the attach frontend machinery and the
            # client_index-based request routing exactly like the N-pair
            # form, so a deployment can verify the whole pipeline with one
            # instance and grow by extending the plan. The equal-count
            # clause below is the actual edge_share invariant; the count
            # itself is not, and every consumer is count-generic.
            if edge_count != cloud_count:
                raise ValueError(
                    "[LWD] edge_share requires equal edge/cloud instance IDs"
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
        assert self.instance_links is not None
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
                        LwdConnectionId(link.edge, link.cloud, edge_dp.dp_idx),
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


# --------------------------------------------------------------------- #
# Runtime topology update (host-comm edge_share elastic scale-out)       #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class LwdTopologyIncrement:
    """A validated pure-addition update for a running Host engine.

    ``new_connections`` are everything Phase B/C must build. The startup
    topology remains the diff baseline; ``new_topology`` is only candidate
    data carried with this increment and never advances that baseline. An
    empty ``new_connections`` is a no-op (the file was touched but its
    semantics did not change)."""

    new_edge_ids: tuple[int, ...]
    new_cloud_ids: tuple[int, ...]
    new_connections: tuple[LwdDPConnection, ...]
    new_topology: LwdTopology


@dataclass(frozen=True)
class LwdTopologyUpdateRejected:
    """A runtime topology update that violates the increment contract.

    ``reason`` names the broken rule and the exact field. The engine keeps
    serving on its realized snapshot; the verdict is deterministic, so
    re-reading the same file yields the same result."""

    reason: str


def _first_change(path: str, old: Any, new: Any) -> str | None:
    """Deepest differing field of two ``asdict`` trees, as ``path: a -> b``."""
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            if key not in old:
                return f"{path}.{key}: <absent> -> {new[key]!r}"
            if key not in new:
                return f"{path}.{key}: {old[key]!r} -> <absent>"
            if old[key] != new[key]:
                return _first_change(f"{path}.{key}", old[key], new[key])
        return None
    if isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)):
        if len(old) != len(new):
            return f"{path}: {len(old)} -> {len(new)} entries"
        for index, (old_item, new_item) in enumerate(zip(old, new)):
            if old_item != new_item:
                return _first_change(f"{path}[{index}]", old_item, new_item)
        return None
    return f"{path}: {old!r} -> {new!r}"


def _frozen_blocks(topology: LwdTopology) -> dict[str, Any]:
    """Global blocks an increment must not touch: deployment (minus the
    instance-count declarations, which follow the added instances),
    feature_ctrl and control_plane. ``qos`` is exempt by deployment
    guarantee, not by code (2026-11-18 review decision)."""
    deployment = asdict(topology.deployment)
    deployment.pop("edges_num")
    deployment.pop("clouds_num")
    return {
        "deployment": deployment,
        "feature_ctrl": asdict(topology.feature_ctrl),
        "control_plane": asdict(topology.control_plane),
    }


def _relaxed_edge_snapshot(
    snapshot: dict[str, Any], added_clouds: set[int]
) -> dict[str, Any]:
    """R2 no-op on this schema: an existing EDGE never needs an edit when
    clouds are added, because a new cloud's coordination URL lives in the
    cloud's own instance block (``LwdDP.coord_url``), not in the edge's
    coordination. The old-branch exemption (growing an edge's
    ``coordination.control_url_map``) targeted a field this schema retired
    — ``LwdCoordination.reject_legacy_fields`` rejects it outright — so R2
    stays a strict byte-identical comparison and this helper documents why
    no exemption is needed here."""
    return snapshot


def diff_topology(
    old: LwdTopology, new: LwdTopology
) -> LwdTopologyIncrement | LwdTopologyUpdateRejected:
    """Validate a runtime topology update as a pure addition (v1 contract).

    Preconditions: both topologies come from ``from_dict``, so the full
    layout validation (scene shape, port uniqueness, count consistency,
    contiguous instance ids, link completeness) already ran. The rules, in
    order:

    * R0 the scene stays the same — and only Host deployment scenes have
      runtime-update semantics: ``edge_share`` adds whole edge/cloud pairs,
      ``cloud_share`` adds edges, and ``lwd_cluster`` adds either role with
      full-mesh links. A same-shape ``single_instance`` file parses clean,
      so the scene must be compared explicitly;
    * R1 global blocks are frozen (see ``_frozen_blocks``): they size the
      realized pools (``up_buffer_size``), seed channel scrambling and
      configure the control plane — none can be applied retroactively;
    * R2 every realized instance survives byte-identical, dp list and
      coordination block included — which also pins every realized
      connection and rules out shrink (scale-in keeps its existing
      kill + liveness semantics). No exemption is needed on this
      schema: a new cloud's coordination URL lives in the cloud's own
      ``LwdDP.coord_url`` block, so adding a cloud requires no edit to
      any existing instance (the old-branch ``control_url_map``
      exemption targeted a field this schema retired);
    * R3 links are addition-only: a swap of realized links is a change;
    * R4 the addition shape is the scene's: edge_share adds whole
      instance pairs and a new link may not cross old and new instances
      (one-to-one); cloud_share adds edges only and every new link MUST
      cross (new edge -> the existing cloud), per the full-mesh scene.

    Instance ids are append-only by construction: layout validation
    requires ids unique and contiguous from 0, so a valid addition can
    only append the next id."""
    if old.digest == new.digest:
        return LwdTopologyIncrement((), (), (), new)
    if old.deployment.scene != new.deployment.scene:
        return LwdTopologyUpdateRejected(
            f"scene changed: {old.deployment.scene} -> {new.deployment.scene}"
        )
    if old.deployment.scene not in (
        "edge_share",
        "cloud_share",
        "lwd_cluster",
    ):
        return LwdTopologyUpdateRejected(
            f"scene {old.deployment.scene} has no runtime-update "
            "semantics; restart the engine to change its plan"
        )
    frozen_old = _frozen_blocks(old)
    frozen_new = _frozen_blocks(new)
    for name, old_block in frozen_old.items():
        change = _first_change(name, old_block, frozen_new[name])
        if change is not None:
            return LwdTopologyUpdateRejected(f"global block changed: {change}")
    old_edges = {instance.id for instance in old.edges}
    old_clouds = {instance.id for instance in old.clouds}
    added_clouds = {instance.id for instance in new.clouds} - old_clouds
    for role, old_instances, new_instances in (
        ("edge", old.edges, new.edges),
        ("cloud", old.clouds, new.clouds),
    ):
        new_by_id = {instance.id: instance for instance in new_instances}
        for instance in old_instances:
            fresh = new_by_id.get(instance.id)
            if fresh is None:
                return LwdTopologyUpdateRejected(
                    f"instance removed: {role} {instance.id} (shrink is not "
                    "supported; scale-in keeps the kill + liveness semantics)"
                )
            fresh_snapshot = (
                _relaxed_edge_snapshot(asdict(fresh), added_clouds)
                if role == "edge"
                else asdict(fresh)
            )
            change = _first_change(
                f"{role} {instance.id}", asdict(instance), fresh_snapshot
            )
            if change is not None:
                return LwdTopologyUpdateRejected(
                    f"existing instance modified: {change}"
                )
    assert old.instance_links is not None
    assert new.instance_links is not None
    old_links = set(old.instance_links)
    new_links = set(new.instance_links)

    def link_key(link: LwdLink) -> tuple[int, int]:
        return (link.edge, link.cloud)

    if old_links - new_links:
        removed = sorted(old_links - new_links, key=link_key)[0]
        return LwdTopologyUpdateRejected(
            f"link removed: edge {removed.edge} - cloud {removed.cloud}"
        )
    added_links = sorted(new_links - old_links, key=link_key)
    new_edge_ids = tuple(sorted({instance.id for instance in new.edges} - old_edges))
    new_cloud_ids = tuple(sorted({instance.id for instance in new.clouds} - old_clouds))
    if old.deployment.scene == "edge_share":
        for link in added_links:
            if (link.edge in old_edges) != (link.cloud in old_clouds):
                return LwdTopologyUpdateRejected(
                    f"new link crosses old and new instances: "
                    f"edge {link.edge} - cloud {link.cloud}"
                )
        if len(new_edge_ids) != len(new_cloud_ids):
            return LwdTopologyUpdateRejected(
                f"additions must be instance pairs: {len(new_edge_ids)} "
                f"edge(s) vs {len(new_cloud_ids)} cloud(s)"
            )
    elif old.deployment.scene == "cloud_share":
        if new_cloud_ids:
            return LwdTopologyUpdateRejected(
                f"cloud_share additions are edges only; new cloud(s) "
                f"{list(new_cloud_ids)} (restart the cloud to change its "
                "own plan)"
            )
        for link in added_links:
            if link.edge not in new_edge_ids or link.cloud not in old_clouds:
                return LwdTopologyUpdateRejected(
                    f"a cloud_share addition links every new edge to the "
                    f"existing cloud; got edge {link.edge} - cloud "
                    f"{link.cloud}"
                )
    # lwd_cluster needs no shape-specific check: the full-mesh scene
    # validation already enforces that every new edge links every cloud
    # and every new cloud is linked from every edge.
    connections: list[LwdDPConnection] = []
    for link in added_links:
        # Filter by the link's cloud: in lwd_cluster a new cloud means
        # existing edges gain a connection to it, and `connections("edge",
        # edge)` returns ALL of that edge's pairs — only the one to this
        # link's cloud is new. In edge_share/cloud_share the filter is a
        # no-op (a new edge's every connection is new).
        edge_conns = new.connections("edge", link.edge)
        connections.extend(
            conn for conn in edge_conns if conn.connection_id.cloud_id == link.cloud
        )
    return LwdTopologyIncrement(new_edge_ids, new_cloud_ids, tuple(connections), new)
