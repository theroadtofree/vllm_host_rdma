# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`lwd_edge` load format: read only the checkpoint tensors the edge holds.

The edge projection selects it for architectures in `LWD_EDGE_WEIGHT_PREFIXES`.
The safetensors index names the shard of every tensor, so the loader opens
only the shards holding whitelisted tensors and reads only those tensors: the
edge checkpoint directory needs the index, the config files and those shards.
The worker process imports this module to register the format.
Design: docs/design/lwd_deepseek_v41.md.
"""

import json
import os
import time
from collections.abc import Generator

import torch
from safetensors import safe_open
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.v1.lwd_control.lwd_model.lwd_edge_projection import (
    LWD_EDGE_LOAD_FORMAT,
    LWD_EDGE_WEIGHT_PREFIXES,
)

logger = init_logger(__name__)

LWD_EDGE_INDEX_FILES = (
    "quant_model_weights.safetensors.index.json",
    SAFE_WEIGHTS_INDEX_NAME,
)
"""Index files looked up in order: the Ascend quantized export, then the HF
one; the same order the Ascend Engram table reader uses."""

LwdShardTensors = dict[str, list[str]]
"""Shard file path -> names of the whitelisted tensors it holds."""


def _lwd_read_weight_map(checkpoint_dir: str) -> tuple[str, dict[str, str]]:
    """The first index of `LWD_EDGE_INDEX_FILES` found, and its weight map.

    Raises:
        ValueError: The directory holds no index.
    """
    for index_file in LWD_EDGE_INDEX_FILES:
        index_path = os.path.join(checkpoint_dir, index_file)
        if os.path.isfile(index_path):
            with open(index_path) as f:
                return index_file, json.load(f)["weight_map"]
    raise ValueError(
        f"[LWD] {LWD_EDGE_LOAD_FORMAT} needs a safetensors index in "
        f"{checkpoint_dir}: one of {LWD_EDGE_INDEX_FILES}"
    )


def lwd_edge_shard_tensors(model_config: ModelConfig) -> LwdShardTensors:
    """The whitelisted tensors of the edge, grouped by the shard holding them.

    Raises:
        ValueError: The architecture has no whitelist, the checkpoint is not a
            local directory, or its index holds no whitelisted tensor.
        FileNotFoundError: A shard holding whitelisted tensors is missing.
    """
    architecture = model_config.architecture
    prefixes = LWD_EDGE_WEIGHT_PREFIXES.get(architecture)
    if prefixes is None:
        raise ValueError(
            f"[LWD] {LWD_EDGE_LOAD_FORMAT} has no edge weight whitelist for "
            f"{architecture}"
        )
    checkpoint_dir = model_config.model
    if not os.path.isdir(checkpoint_dir):
        raise ValueError(
            f"[LWD] {LWD_EDGE_LOAD_FORMAT} reads a local checkpoint directory, "
            f"got {checkpoint_dir!r}"
        )
    index_file, weight_map = _lwd_read_weight_map(checkpoint_dir)
    shard_tensors: LwdShardTensors = {}
    for name, shard in weight_map.items():
        if name.startswith(prefixes):
            shard_path = os.path.join(checkpoint_dir, shard)
            shard_tensors.setdefault(shard_path, []).append(name)
    if not shard_tensors:
        raise ValueError(
            f"[LWD] {index_file} in {checkpoint_dir} holds no tensor with the "
            f"{architecture} edge prefixes {prefixes}"
        )
    missing = sorted(path for path in shard_tensors if not os.path.isfile(path))
    if missing:
        raise FileNotFoundError(f"[LWD] edge checkpoint shards missing: {missing}")
    logger.info(
        "[LWD] edge reads %d of %d tensors from %d of %d shards (%s)",
        sum(len(names) for names in shard_tensors.values()),
        len(weight_map),
        len(shard_tensors),
        len(set(weight_map.values())),
        index_file,
    )
    return dict(sorted(shard_tensors.items()))


@register_model_loader(LWD_EDGE_LOAD_FORMAT)
class LwdEdgeModelLoader(DefaultModelLoader):
    """Default loader reading only the edge's whitelisted tensors.

    The tensors are read lazily, one shard after another, so the safetensors
    load strategy and multithreaded loading do not apply.
    """

    def download_model(self, model_config: ModelConfig) -> None:
        lwd_edge_shard_tensors(model_config)

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        if getattr(model, "secondary_weights", ()):
            raise ValueError(
                f"[LWD] {LWD_EDGE_LOAD_FORMAT} reads one checkpoint; "
                f"{type(model).__name__} declares secondary weights"
            )
        shard_tensors = lwd_edge_shard_tensors(model_config)
        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        for shard_path, names in shard_tensors.items():
            with safe_open(shard_path, framework="pt") as f:
                for name in names:
                    yield name, f.get_tensor(name)
