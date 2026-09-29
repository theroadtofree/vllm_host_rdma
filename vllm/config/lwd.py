# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LWD (layerwise disaggregated) configuration.

LWD splits a model's transformer layers across an edge / cloud pair:
the edge process holds the head ``[0, head_k)`` and tail
``[N - tail_k, N)`` ranges (or none in ``embedding_only`` mode), while
the cloud process holds the middle range ``[head_k, N - tail_k)``.
``make_layers`` consumes :meth:`LwdConfig.local_layer_indices` so
weights are created directly on the owning device with
``PPMissingLayer`` placeholders elsewhere.

The config is owned by the vLLM repository: ``additional_config``
key ``lwd_config`` is parsed once in ``VllmConfig.__post_init__``;
``lwd_config.enabled`` is the single master switch, and the no-CLI
narrow fields ``ParallelConfig.enable_lwd`` / ``is_edge_node`` are
back-filled from it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

from vllm.logger import init_logger

logger = init_logger(__name__)

_VALID_LWD_ROLES = ("edge", "cloud")
_VALID_LWD_MODES = ("head_tail", "embedding_only", "prefill_only")

# --- LWD prefill-only edge batch queue depth ------------------------- #
# Single-instance baseline: total in-flight batches (embed+unembed) the
# edge engine may hold un-harvested with one cloud instance; also the
# embed dispatch reserve cap (the historical fixed value of 4).
LWD_EDGE_BASE_DEPTH = 4
# Extra unembed in-flight quota per additional cloud instance: keeps the
# per-instance ">= 2 in flight" pipeline alignment as N grows.
LWD_EDGE_UNEMBED_PER_INSTANCE = 2
# Total-depth ceiling: bounds pooled-buffer/memory growth at the config
# upper bound of instance_parallel_size (deployments cap N at 8 -> 18).
LWD_EDGE_BATCH_QUEUE_DEPTH_MAX = 32
# Deployment override for the formula depth (>= 1); bad values warn and
# fall back to the formula, never raise.
_LWD_EDGE_QUEUE_DEPTH_ENV = "VLLM_ASCEND_LWD_EDGE_QUEUE_DEPTH"


def _parse_lwd_layer_split(value: Any) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"[LWD] edge_head_tail_layers must be a 2-element array [head_k, tail_k], got {value!r}")
    if len(value) != 2:
        raise ValueError(f"[LWD] edge_head_tail_layers must have exactly 2 elements, got {value!r}")
    split = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"[LWD] edge_head_tail_layers elements must be non-negative integers, got {value!r}")
        split.append(item)
    return split[0], split[1]  # type: ignore[return-value]


@dataclass(frozen=True)
class LwdConfig:
    enabled: bool = False
    role: str = "edge"
    """Process role: "edge" or "cloud"."""
    mode: str = "head_tail"
    """Layer distribution mode: "head_tail", "embedding_only" or "prefill_only"."""
    edge_head_tail_layers: tuple[int, int] = (1, 1)
    """Fixed 2-element (head_k, tail_k) asymmetric splits allowed."""

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "LwdConfig":
        cfg = cls(
            enabled=bool(raw.get("enabled", False)),
            role=str(raw.get("role", "edge")),
            mode=str(raw.get("mode", "head_tail")),
            edge_head_tail_layers=_parse_lwd_layer_split(raw.get("edge_head_tail_layers", [1, 1])),
        )
        cfg.validate()
        return cfg

    @property
    def is_edge(self) -> bool:
        return self.role == "edge"

    def validate(self) -> None:
        if self.role not in _VALID_LWD_ROLES:
            raise ValueError(f"[LWD] role must be one of {_VALID_LWD_ROLES}, got {self.role!r}")
        if self.mode not in _VALID_LWD_MODES:
            raise ValueError(f"[LWD] mode must be one of {_VALID_LWD_MODES}, got {self.mode!r}")
        head_k, tail_k = self.edge_head_tail_layers
        if self.mode in ("embedding_only", "prefill_only"):
            if (head_k, tail_k) != (0, 0):
                raise ValueError(f"[LWD] {self.mode} requires edge_head_tail_layers [0, 0], got [{head_k},{tail_k}]")
        elif head_k + tail_k < 1:
            raise ValueError(f"[LWD] 'head_tail' mode requires at least one edge layer, got [{head_k}, {tail_k}]")

    def validate_num_hidden_layers(self, num_hidden_layers: int) -> None:
        head_k, tail_k = self.edge_head_tail_layers
        if head_k + tail_k >= num_hidden_layers:
            raise ValueError(
                "[LWD] layer split must leave a non-empty middle range for "
                f"the cloud: head_k + tail_k ({head_k + tail_k}) must be "
                f"< num_hidden_layers ({num_hidden_layers})")

    def local_layer_indices(self, num_hidden_layers: int) -> set[int]:
        head_k, tail_k = self.edge_head_tail_layers
        if self.mode == "head_tail":
            self.validate_num_hidden_layers(num_hidden_layers)
        if self.is_edge:
            if (head_k, tail_k) == (0, 0):
                return set()
            return (set(range(head_k)) | set(range(num_hidden_layers - tail_k, num_hidden_layers)))
        return set(range(head_k, num_hidden_layers - tail_k))

    def __repr__(self) -> str:
        return (f"[LWD] config(enabled={self.enabled}, role={self.role!r}, "
                f"mode={self.mode!r}, edge_head_tail_layers="
                f"{list(self.edge_head_tail_layers)})")


def lwd_edge_batch_queue_depth(vllm_config) -> int | None:
    """In-flight batch queue depth for the LWD prefill-only edge engine.

    Returns ``None`` unless this process is the edge of an enabled
    ``prefill_only`` LWD deployment. The depth scales with the number of
    cloud instances ``N`` so per-instance unembed dispatches are not
    squeezed by a single-instance cap::

        depth = LWD_EDGE_BASE_DEPTH
                + LWD_EDGE_UNEMBED_PER_INSTANCE * (N - 1)

    ``N = 1`` keeps the historical fixed depth of 4 (bit-for-bit legacy
    behavior). The value is shared by the edge engine (dispatch
    throttling) and the worker buffer pools
    (``set_default_max_concurrency``); it lives here rather than in the
    control-plane assembly layer because workers are constructed before
    the edge engine assembles and multi-process workers only see the
    pickled ``vllm_config`` plus inherited env — so the
    ``VLLM_ASCEND_LWD_EDGE_QUEUE_DEPTH`` override is read here as well to
    keep both sides on the same value.
    """
    lwd = getattr(vllm_config, "lwd_config", None)
    if (
        lwd is None
        or not lwd.enabled
        or lwd.mode != "prefill_only"
        or not lwd.is_edge
    ):
        return None
    parallel = getattr(vllm_config, "parallel_config", None)
    parallel_lwd = getattr(parallel, "lwd_config", None)
    try:
        n = int(getattr(parallel_lwd, "instance_parallel_size", 1))
    except (TypeError, ValueError):
        n = 1
    n = max(1, n)
    depth = min(
        LWD_EDGE_BASE_DEPTH + LWD_EDGE_UNEMBED_PER_INSTANCE * (n - 1),
        LWD_EDGE_BATCH_QUEUE_DEPTH_MAX,
    )
    raw = os.environ.get(_LWD_EDGE_QUEUE_DEPTH_ENV)
    if raw is not None:
        try:
            override = int(raw)
        except ValueError:
            logger.warning(
                "[Lwd] ignore invalid env %s=%r, keep formula depth %d",
                _LWD_EDGE_QUEUE_DEPTH_ENV,
                raw,
                depth,
            )
            return depth
        if override >= 1:
            return override
        logger.warning(
            "[Lwd] ignore non-positive env %s=%r, keep formula depth %d",
            _LWD_EDGE_QUEUE_DEPTH_ENV,
            raw,
            depth,
        )
    return depth
