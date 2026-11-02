# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-release compatibility for the numeric and role-registry LWD CLI."""

from pathlib import Path
from typing import Any

import yaml

from vllm.logger import init_logger

from .api import LwdAPIConfig
from .entry import LwdConfig
from .topology import LwdTopology

logger = init_logger(__name__)

_CONTROL_PORT_BASE = 5550
_COORDINATION_PORT = 8100


def has_legacy_lwd_args(args: Any) -> bool:
    additional = getattr(args, "additional_config", None)
    entry = additional.get("lwd_config") if isinstance(additional, dict) else None
    if isinstance(entry, dict) and entry.get("path"):
        return False
    if entry is not None:
        return True
    return bool(
        getattr(args, "role_registry", "")
        or getattr(args, "edge_npu_count", 0)
        or getattr(args, "cloud_npu_count", 0)
        or getattr(args, "cloud_dp_size", None) is not None
        or getattr(args, "instance_parallel_start_rank", None) is not None
        or getattr(args, "instance_parallel_size", 1) != 1
        or getattr(args, "edge_id", 0)
        or getattr(args, "cloud_id", 0)
    )


def _identity(args: Any, role: str) -> int:
    if role == "edge":
        return args.edge_id
    if args.cloud_id == 0 and args.instance_parallel_start_rank is not None:
        return args.instance_parallel_start_rank
    return args.cloud_id


def _coordination(
    role: str, instances: list[dict[str, Any]], clouds: list[dict[str, Any]]
) -> None:
    for instance in instances:
        if role == "edge":
            instance["coordination"] = {
                "enabled": True,
                "control_url_map": {
                    cloud["id"]: (
                        f"http://{cloud['dp'][0]['addr']}:{_COORDINATION_PORT}"
                        "/v1/chat/completions"
                    )
                    for cloud in clouds
                },
                "tenant_key_file": "/run/secrets/edge-cloud-tenant-key",
                "consumer_id": "enterprise-a",
            }
        else:
            instance["coordination"] = {
                "enabled": True,
                "listen_host": "0.0.0.0",
                "listen_port": _COORDINATION_PORT,
            }


def _from_registry(args: Any, address: str) -> LwdTopology:
    try:
        raw = yaml.safe_load(Path(args.role_registry).read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(
            f"[LWD] Invalid legacy role registry {args.role_registry!r}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError("[LWD] Legacy role registry must contain a mapping")
    edges_in, clouds_in = raw.get("edges"), raw.get("clouds")
    if not isinstance(edges_in, list) or not isinstance(clouds_in, list):
        raise ValueError("[LWD] Legacy role registry requires edges and clouds lists")
    if not edges_in or not clouds_in:
        raise ValueError("[LWD] Legacy role registry requires at least one edge/cloud")

    edges = []
    for edge in edges_in:
        ranks = list(edge.get("ranks", ()))
        dp_count = clouds_in[0].get("dp_size", 1)
        edges.append(
            {
                "id": edge["id"],
                "dp": [
                    {"dp_idx": index, "addr": edge.get("addr") or address, "ranks": ranks}
                    for index in range(dp_count)
                ],
            }
        )
    dp_counts = {int(cloud.get("dp_size", 1)) for cloud in clouds_in}
    if len(dp_counts) != 1:
        raise ValueError("[LWD] Legacy conversion requires the same DP count per cloud")
    clouds = []
    for cloud in clouds_in:
        ranks = list(cloud.get("ranks", ()))
        dp_count = int(cloud.get("dp_size", 1))
        if dp_count < 1 or len(ranks) % dp_count:
            raise ValueError(
                f"[LWD] Legacy cloud {cloud.get('id')} ranks cannot be split "
                f"evenly into dp_size={dp_count}"
            )
        per_dp = len(ranks) // dp_count
        engine_ports = list(cloud.get("engine_ports", ()))
        base_port = int(cloud.get("zmq_port", cloud.get("zmq_base_port", 0)))
        if not engine_ports:
            if dp_count != 1:
                raise ValueError(
                    f"[LWD] Legacy cloud {cloud.get('id')} with DP requires "
                    "engine_ports for per-DP control endpoints"
                )
            engine_ports = [base_port]
        if len(engine_ports) != dp_count:
            raise ValueError(
                f"[LWD] Legacy cloud {cloud.get('id')} engine_ports must "
                "match dp_size"
            )
        clouds.append(
            {
                "id": cloud["id"],
                "dp": [
                    {
                        "dp_idx": index,
                        "addr": cloud.get("addr") or address,
                        "ranks": ranks[index * per_dp : (index + 1) * per_dp],
                        "ctrl_port": engine_ports[index],
                    }
                    for index in range(dp_count)
                ],
            }
        )

    scene = "single_instance"
    if len(edges) > 1 or len(clouds) > 1:
        scene = "cloud_share" if len(clouds) == 1 else "lwd_cluster"
    if scene != "single_instance":
        _coordination("edge", edges, clouds)
        _coordination("cloud", clouds, clouds)
    world_size = sum(len(edge["dp"][0]["ranks"]) for edge in edges) + sum(
        len(dp["ranks"]) for cloud in clouds for dp in cloud["dp"]
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
            "feature_ctrl": {"enable_early_recv": False},
            "edges": edges,
            "clouds": clouds,
            "instance_links": [
                {"edge": edge["id"], "cloud": cloud["id"]}
                for edge in edges
                for cloud in clouds
            ],
        }
    )


def _from_counts(args: Any, role: str, address: str) -> LwdTopology:
    edge_count = args.edge_npu_count
    cloud_total = args.cloud_npu_count
    instances = args.instance_parallel_size
    cloud_dp = args.cloud_dp_size
    if cloud_dp is not None and role == "cloud":
        # Legacy dual meaning: explicit DP means per-engine on cloud commands;
        # edge commands already carry the cross-instance grand total.
        cloud_total *= cloud_dp * instances
    elif cloud_dp is None:
        cloud_dp = args.data_parallel_size
    if edge_count < 1 or cloud_total < 1:
        raise ValueError(
            "[LWD] Legacy numeric CLI requires positive edge/cloud NPU counts"
        )
    if not 1 <= instances <= 16 or cloud_dp < 1:
        raise ValueError("[LWD] Invalid legacy instance_parallel_size/cloud_dp_size")
    if cloud_dp > 1 and edge_count != 1:
        raise ValueError(
            "[LWD] Legacy numeric multi-DP CLI requires a shared one-NPU edge; "
            "use a role registry/topology for multi-NPU edges"
        )
    if instances > 1 and edge_count != 1:
        raise ValueError("[LWD] Legacy multi-instance CLI requires a shared one-NPU edge")
    divisor = instances * cloud_dp
    if cloud_total % divisor:
        raise ValueError(
            f"[LWD] Normalized cloud_npu_count={cloud_total} must be divisible "
            f"by instance_parallel_size * cloud_dp_size={divisor}"
        )
    identity = _identity(args, role)
    if role == "cloud" and not 0 <= identity < instances:
        raise ValueError(
            f"[LWD] cloud instance_id={identity} outside [0, {instances})"
        )

    # Multi-instance numeric mode has one physical shared edge, represented by
    # logical edge IDs paired one-to-one with clouds.
    edge_ids = range(instances) if instances > 1 else range(1)
    edges = [
        {
            "id": edge_id,
            "dp": [
                {
                    "dp_idx": dp_idx,
                    "addr": address,
                    "ranks": list(range(edge_count)),
                }
                for dp_idx in range(cloud_dp)
            ],
        }
        for edge_id in edge_ids
    ]
    per_dp = cloud_total // divisor
    clouds = []
    next_rank = edge_count if instances == 1 else 1
    for cloud_id in range(instances):
        clouds.append(
            {
                "id": cloud_id,
                "dp": [
                    {
                        "dp_idx": dp_idx,
                        "addr": address,
                        "ranks": list(
                            range(
                                next_rank + dp_idx * per_dp,
                                next_rank + (dp_idx + 1) * per_dp,
                            )
                        ),
                        "ctrl_port": _CONTROL_PORT_BASE + dp_idx,
                    }
                    for dp_idx in range(cloud_dp)
                ],
            }
        )
        next_rank += cloud_dp * per_dp
    return LwdTopology.from_dict(
        {
            "deployment": {
                "mode": 0,
                "scene": "edge_share" if instances > 1 else "single_instance",
                "hccl_world_size": next_rank,
                "edges_num": len(edges),
                "clouds_num": len(clouds),
            },
            "feature_ctrl": {"enable_early_recv": False},
            "edges": edges,
            "clouds": clouds,
            "instance_links": [
                {"edge": cloud_id, "cloud": cloud_id}
                for cloud_id in range(instances)
            ]
            if instances > 1
            else [{"edge": 0, "cloud": 0}],
        }
    )


def resolve_legacy_lwd_config(args: Any, *, api: LwdAPIConfig) -> LwdConfig:
    additional = args.additional_config if isinstance(args.additional_config, dict) else {}
    entry = additional.get("lwd_config") or {}
    if not isinstance(entry, dict) or entry.get("role") not in ("edge", "cloud"):
        raise ValueError(
            "[LWD] Legacy numeric CLI requires additional_config.lwd_config.role "
            "to be 'edge' or 'cloud'"
        )
    role = entry["role"]
    identity = _identity(args, role)
    address = args.master_addr or "127.0.0.1"
    topology = (
        _from_registry(args, address)
        if args.role_registry
        else _from_counts(args, role, address)
    )
    logger.warning(
        "DEPRECATION: legacy LWD numeric/--role-registry arguments are "
        "deprecated; use additional_config.lwd_config.path instead"
    )
    return LwdConfig.from_topology(
        args.role_registry or "<legacy-lwd-cli>", role, identity, topology, api=api
    )
