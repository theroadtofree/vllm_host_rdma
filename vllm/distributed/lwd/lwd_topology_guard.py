# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed LWD topology checks before HCCL initialization."""

import json
from datetime import timedelta
from typing import TYPE_CHECKING

from vllm import envs
from vllm.config.lwd.topology import LwdTopology
from vllm.distributed.utils import create_tcp_store

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def _expected_owners(topology: LwdTopology) -> list[set[tuple[str, int]]]:
    owners = [set() for _ in range(topology.deployment.hccl_world_size)]
    for role, instances in (("edge", topology.edges), ("cloud", topology.clouds)):
        for instance in instances:
            for dp in instance.dp:
                for rank in dp.ranks:
                    owners[rank].add((role, instance.id))
    return owners


def _check_owner(
    topology: LwdTopology,
    owners: list[set[tuple[str, int]]],
    rank: int,
    role: str,
    instance_id: int,
) -> None:
    expected = sorted(owners[rank])
    actual = (role, instance_id)
    if actual not in owners[rank]:
        raise ValueError(
            f"[LWD] rank ownership mismatch at rank={rank}: "
            f"expected one of role/instance_id={expected}, "
            f"actual role={role!r}, instance_id={instance_id}"
        )
    # Also make a malformed local identity fail at its source, rather than
    # relying only on peers to reject this rank's registration.
    topology.instance(role, instance_id)


def _check_port(master_port: int) -> int:
    port = (
        envs.VLLM_LWD_TOPOLOGY_CHECK_PORT
        if envs.VLLM_LWD_TOPOLOGY_CHECK_PORT is not None
        else master_port + 2
    )
    if not 0 < port < 65536:
        raise ValueError(
            f"[LWD] topology check port {port} is out of range; set "
            "VLLM_LWD_TOPOLOGY_CHECK_PORT to a port in (0, 65536)"
        )
    return port


def validate_lwd_topology_pre_init(vllm_config: "VllmConfig", rank: int) -> None:
    """Exchange canonical topology and identity over a one-shot TCPStore."""
    lwd = vllm_config.lwd_config
    assert lwd is not None
    topology = lwd.topology
    parallel = vllm_config.parallel_config
    world_size = topology.deployment.hccl_world_size
    if not 0 <= rank < world_size:
        raise ValueError(
            f"[LWD] rank ownership mismatch: rank={rank} is outside "
            f"topology world_size={world_size}"
        )
    owners = _expected_owners(topology)
    _check_owner(topology, owners, rank, lwd.role, lwd.instance_id)

    timeout_s = envs.VLLM_LWD_TOPOLOGY_CHECK_TIMEOUT_S
    if not 0 < timeout_s <= 3600:
        raise ValueError(
            "[LWD] VLLM_LWD_TOPOLOGY_CHECK_TIMEOUT_S must be in (0, 3600], "
            f"got {timeout_s}"
        )
    timeout = timedelta(seconds=timeout_s)
    port = _check_port(parallel.master_port)
    try:
        store = create_tcp_store(
            parallel.master_addr,
            port,
            world_size=world_size,
            is_master=rank == 0,
            wait_for_workers=False,
            timeout=timeout,
            use_libuv=False,
        )
        store.set_timeout(timeout)
    except Exception as exc:
        raise RuntimeError(
            f"[LWD] topology precheck cannot create TCPStore at "
            f"{parallel.master_addr}:{port} for rank={rank}; free the port or "
            "set VLLM_LWD_TOPOLOGY_CHECK_PORT on every launch command"
        ) from exc

    payload = {
        "digest": topology.digest,
        "role": lwd.role,
        "instance_id": lwd.instance_id,
    }
    try:
        store.set(
            f"lwd/topology/{rank}",
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
        )
        for remote_rank in range(world_size):
            raw = store.get(f"lwd/topology/{remote_rank}")
            try:
                remote = json.loads(raw.decode("utf-8"))
                remote_digest = remote["digest"]
                remote_role = remote["role"]
                remote_instance = remote["instance_id"]
                if (
                    not isinstance(remote_digest, str)
                    or not isinstance(remote_role, str)
                    or type(remote_instance) is not int
                ):
                    raise TypeError("digest/role/instance_id have invalid types")
            except (UnicodeError, ValueError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"[LWD] invalid topology precheck payload from "
                    f"rank={remote_rank}: {raw!r}"
                ) from exc
            if remote_digest != topology.digest:
                raise ValueError(
                    f"[LWD] topology digest mismatch at rank={rank} with "
                    f"rank={remote_rank}: expected={topology.digest}, "
                    f"actual={remote_digest}"
                )
            _check_owner(
                topology,
                owners,
                remote_rank,
                remote_role,
                remote_instance,
            )
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise RuntimeError(
            f"[LWD] topology precheck timed out or failed at rank={rank} "
            f"after at most {timeout_s}s"
        ) from exc
    finally:
        del store
