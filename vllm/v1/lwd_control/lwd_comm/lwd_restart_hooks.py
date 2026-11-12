# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cloud-restart hook registry (peer-restart line, phase 2).

The vllm repo owns the VERDICTS: the edge engine's liveness sweep fires
``notify_cloud_restarted`` when a cloud DP goes silent or its boot id
changes. The TEARDOWN belongs to the NPU assembly (vllm-ascend), which
registers a hook at channel-init time; the layering rule is the old
branch's: the vllm repo never imports the plugin. A hook that raises is
logged, never swallowed — a swallowed teardown is a silent zombie link.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from vllm.lwd_logger import lwd_logger

LwdCloudRestartHook = Callable[[int, int, str], None]
"""(cloud_id, dp_idx, reason) -> teardown of that cloud DP's local links."""


_hooks: list[LwdCloudRestartHook] = []
_lock = threading.Lock()


def register_cloud_restart_hook(hook: LwdCloudRestartHook) -> None:
    """Register the data-plane teardown hook (idempotent per process)."""
    with _lock:
        if hook not in _hooks:
            _hooks.append(hook)


def notify_cloud_restarted(cloud_id: int, dp_idx: int, reason: str) -> None:
    """Fire every registered hook for one cloud DP restart verdict."""
    with _lock:
        hooks = list(_hooks)
    for hook in hooks:
        try:
            hook(cloud_id, dp_idx, reason)
        except Exception:
            lwd_logger.exception(
                "[LWD] cloud restart hook failed for cloud=%d dp=%d (%s)",
                cloud_id,
                dp_idx,
                reason,
            )
