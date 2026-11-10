# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Log the resolved LWD metadata without reading files or starting execution."""

import json
import logging
from dataclasses import asdict

from .entry import LwdConfig


def log_lwd_config(logger: logging.Logger, config: LwdConfig) -> None:
    """Emit one startup record with all fields, defaults and selected connections."""
    # Rank-derived fields exist only in device comm (shared HCCL world);
    # host comm declares no ranks, so they are logged as None instead of
    # being derived.
    is_device = not config.topology.is_host_comm
    snapshot = {
        "metadata_only": True,
        "lwd_config": asdict(config),
        "selected_instance": asdict(config.instance),
        "connections": [
            {
                **asdict(link),
                "cloud_leader_rank": (
                    link.cloud_leader_rank if is_device and link.cloud.ranks else None
                ),
            }
            for link in config.connections
        ],
        "cloud_cards": {
            str(cloud.id): (
                sum(len(dp.ranks) for dp in cloud.dp) if is_device else None
            )
            for cloud in config.topology.clouds
        },
    }
    # Serialize the parsed snapshot only; tenant_key_file remains a path, not content.
    logger.info("[LWD][config] resolved=%s", json.dumps(snapshot, ensure_ascii=False))
