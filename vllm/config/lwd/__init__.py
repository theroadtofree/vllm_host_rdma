# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-independent LWD configuration, without execution hooks."""

from typing import Any

from .api import LwdAPIConfig
from .coordination import LwdCoordinationConfig
from .entry import LwdConfig
from .logging import log_lwd_config
from .topology import LwdConnectionId, LwdTopology

__all__ = [
    "LwdAPIConfig",
    "LwdConfig",
    "LwdConnectionId",
    "LwdCoordinationConfig",
    "LwdTopology",
    "log_lwd_config",
    "resolve_lwd_config",
]


def resolve_lwd_config(
    additional_config: Any,
    *,
    api: LwdAPIConfig | None = None,
) -> LwdConfig | None:
    """Parse once into the VllmConfig snapshot forwarded to engines/workers."""
    if not isinstance(additional_config, dict):
        additional_config = {}
    if "edge_cloud_config" in additional_config:
        raise ValueError("[LWD] edge_cloud_config is retired; use lwd_config.path")
    if "lwd_coordination" in additional_config:
        raise ValueError(
            "[LWD] Top-level lwd_coordination is retired; "
            "move it into lwd_config.coordination"
        )
    lwd = (
        LwdConfig.from_dict(additional_config["lwd_config"], api=api)
        if "lwd_config" in additional_config
        else None
    )
    if lwd is None and api is not None:
        api.validate_topology(None)
    return lwd
