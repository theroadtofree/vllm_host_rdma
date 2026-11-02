# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Edge projection of the engine arguments.

The LWD edge computes the prompt embeddings, the vision tower and the LM head:
it runs no decoder layer (`make_layers` pins it at zero layers), drafts
nothing and holds no Engram table. The projection keeps the edge from building
or reading what only the cloud uses. Design: docs/design/lwd_deepseek_v41.md.
"""

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from transformers import PretrainedConfig

from vllm.config.engram import _NGRAM_LAYER_FIELDS
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.config.model import HfOverrides
    from vllm.engine.arg_utils import EngineArgs

logger = init_logger(__name__)

LWD_EDGE_LOAD_FORMAT = "lwd_edge"
"""Load format of `LwdEdgeModelLoader`."""

LWD_EDGE_WEIGHT_PREFIXES: dict[str, tuple[str, ...]] = {
    # TODO(lwd): add the Qwen-VL and DeepSeek V4 edge whitelists.
    "DeepseekV41ForCausalLM": (
        "embed.",
        "head.",
        "norm.",
        "vision.",
        "aligner.",
        "image_",
    ),
}
"""Per architecture, checkpoint name prefixes of the tensors the edge uses."""

_LWD_EDGE_REPLACED_LOAD_FORMATS = ("auto", "safetensors")
"""Formats `lwd_edge` reads the same way, restricted to the whitelist."""
_LWD_EDGE_KEPT_LOAD_FORMATS = ("dummy", LWD_EDGE_LOAD_FORMAT)
"""Formats that already read nothing beyond the whitelist."""


def lwd_create_edge_model_config(engine_args: "EngineArgs") -> "ModelConfig":
    """Project `engine_args` in place for the edge and create its model config.

    `hf_overrides` is projected before the config is parsed; `load_format`
    after, since it depends on the resolved architecture.

    Raises:
        ValueError: A cloud-only argument is set, or the load format would read
            the whole checkpoint.
    """
    lwd_check_edge_engine_args(engine_args)
    engine_args.hf_overrides = lwd_edge_hf_overrides(engine_args.hf_overrides)
    model_config = engine_args.create_model_config()
    engine_args.load_format = lwd_edge_load_format(
        engine_args.load_format, model_config
    )
    return model_config


def lwd_check_edge_engine_args(engine_args: "EngineArgs") -> None:
    """Reject engine arguments for what only the cloud runs.

    Raises:
        ValueError: An Engram or speculative config is set on the edge.
    """
    if engine_args.engram_config is not None:
        raise ValueError(
            "[LWD] --engram-config is cloud-only: the edge holds no Engram table"
        )
    spec_flags = (
        engine_args.spec_method,
        engine_args.spec_model,
        engine_args.spec_tokens,
    )
    if engine_args.speculative_config is not None or any(
        flag is not None for flag in spec_flags
    ):
        raise ValueError(
            "[LWD] the edge takes no speculative config: it drafts nothing and "
            "adopts num_speculative_tokens from the cloud"
        )


def _lwd_clear_ngram_layers(hf_config: PretrainedConfig) -> PretrainedConfig:
    """Drop the n-gram embedding layers, so the edge builds no table for them.

    The edge Engram hasher re-reads the checkpoint config for the hash
    parameters, so nothing on the edge needs the cleared field.
    """
    for architecture in hf_config.architectures or ():
        field = _NGRAM_LAYER_FIELDS.get(architecture)
        if field is None:
            continue
        text_config = hf_config.get_text_config()
        layer_ids = getattr(text_config, field, None)
        if layer_ids:
            setattr(text_config, field, [])
            logger.info(
                "[LWD] edge builds no n-gram embedding layer: %s %s=%s cleared",
                architecture,
                field,
                layer_ids,
            )
    return hf_config


def _lwd_project_after_callable(
    user_hf_overrides: Callable[[PretrainedConfig], PretrainedConfig],
    hf_config: PretrainedConfig,
) -> PretrainedConfig:
    return _lwd_clear_ngram_layers(user_hf_overrides(hf_config))


def _lwd_project_after_update(
    user_hf_overrides: dict[str, Any],
    hf_config: PretrainedConfig,
) -> PretrainedConfig:
    hf_config.update(user_hf_overrides)
    return _lwd_clear_ngram_layers(hf_config)


def lwd_edge_hf_overrides(hf_overrides: "HfOverrides") -> "HfOverrides":
    """`hf_overrides` of the edge: the user's, then the n-gram layers cleared.

    The projection is a `functools.partial` over module functions, so the
    `ModelConfig` stays picklable for engine-core processes. A dict of flat
    keys is applied the way `ModelConfig` applies it, `config.update` at the
    point a callable runs. A dict that updates a sub-config is kept as is:
    `ModelConfig` applies such updates after the config is parsed, which a
    callable cannot reproduce, so for a model with n-gram layers
    `lwd_check_parallel_config` then rejects the edge.
    """
    if callable(hf_overrides):
        return functools.partial(_lwd_project_after_callable, hf_overrides)
    nested_keys = sorted(
        key for key, value in hf_overrides.items() if isinstance(value, dict)
    )
    if nested_keys:
        logger.info(
            "[LWD] edge keeps the dict hf_overrides as given: sub-config "
            "updates %s leave the n-gram layers in place",
            nested_keys,
        )
        return hf_overrides
    return functools.partial(_lwd_project_after_update, dict(hf_overrides))


def lwd_edge_load_format(load_format: str, model_config: "ModelConfig") -> str:
    """The edge load format: `lwd_edge` for architectures with a whitelist.

    Raises:
        ValueError: The load format would read the whole checkpoint of an
            architecture with a whitelist.
    """
    architecture = model_config.architecture
    prefixes = LWD_EDGE_WEIGHT_PREFIXES.get(architecture)
    normalized = load_format.lower()
    if prefixes is None or normalized in _LWD_EDGE_KEPT_LOAD_FORMATS:
        return load_format
    if normalized not in _LWD_EDGE_REPLACED_LOAD_FORMATS:
        raise ValueError(
            f"[LWD] edge load_format {load_format!r} would read the whole "
            f"{architecture} checkpoint; use one of "
            f"{_LWD_EDGE_REPLACED_LOAD_FORMATS + _LWD_EDGE_KEPT_LOAD_FORMATS}"
        )
    logger.info(
        "[LWD] edge loads only %s tensors of %s: load_format %s -> %s",
        prefixes,
        architecture,
        load_format,
        LWD_EDGE_LOAD_FORMAT,
    )
    return LWD_EDGE_LOAD_FORMAT
