# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data-plane link-add hook registry (elastic scale-out, phase C).

The vllm repo owns the DECISION and the ORDER: the edge engine's topology
reload applies the control-plane half first, then asks the data plane to
build the connection's UCX links. The ASSEMBLY belongs to the NPU side
(vllm-ascend), which registers a hook at host-channel init; the layering
rule is the pair's: the vllm repo never imports the plugin. A hook that
raises is logged and counts as a refusal — the connection stays
"control-plane only" and the admission gate keeps rejecting it until a
later topology change retries it."""

from __future__ import annotations

import threading
from collections.abc import Callable

from vllm.config.lwd.topology import LwdDPConnection
from vllm.lwd_logger import lwd_logger

LwdDataPlaneAddHook = Callable[[LwdDPConnection], bool]
"""(connection) -> True when the connection's UCX links were built."""


_hooks: list[LwdDataPlaneAddHook] = []
_lock = threading.Lock()


def register_data_plane_add_hook(hook: LwdDataPlaneAddHook) -> None:
    """Register the data-plane link assembler (idempotent per process)."""
    with _lock:
        if hook not in _hooks:
            _hooks.append(hook)


def notify_data_plane_added(connection: LwdDPConnection) -> bool:
    """Fire every registered hook; True when one built the links."""
    with _lock:
        hooks = list(_hooks)
    built = False
    for hook in hooks:
        try:
            built = hook(connection) or built
        except Exception:
            lwd_logger.exception(
                "[LWD] data-plane add hook failed for %s",
                connection.connection_id,
            )
    return built
