# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skip checkpoint tensors of modules the LWD role does not own.

Build-time pruning (the ``LWD_MODULE_OWNERSHIP`` guards) already avoids
constructing those modules, but the checkpoint on disk still contains
their tensors: loading them would fail with unmatched weights and waste
disk reads. This filter drops such tensors at iteration time, deriving
its skip keys from the same ownership table so load-time and build-time
stay consistent.
"""

from functools import lru_cache

from vllm.model_executor.models.utils import LWD_MODULE_OWNERSHIP


@lru_cache(maxsize=2)
def parse_skip_keys(role: str) -> frozenset[str]:
    """Module-path suffixes the role does not own, from the ownership table."""
    return frozenset(
        suffix for suffix, owners in LWD_MODULE_OWNERSHIP.items() if role not in owners
    )


def _matches_module_suffix(name: str, key: str) -> bool:
    """Whether a dot-boundary prefix of ``name`` is a module path ending in
    ``key``, mirroring the suffix semantics of ``lwd_skips_module``.

    Checkpoint names may nest the vLLM module path under extra components
    (``model.visual.merger`` for the ``visual`` tower), so every module
    prefix of the tensor name is checked, not just the whole name.
    """
    parts = name.split(".")
    # The last component is the leaf parameter (e.g. "weight"), not a module.
    for i in range(1, len(parts)):
        path = ".".join(parts[:i])
        if path == key or path.endswith("." + key):
            return True
    return False


_VL_TEXT_PREFIX = "model.language_model."


def _unwrap_vl_text_prefix(name: str) -> str:
    """Rewrite the VL checkpoint nesting ``model.language_model.*`` to
    ``model.*``.

    VL checkpoints (Qwen3-VL, GLM-4.5V) store the language model under
    ``model.language_model.`` while the ownership table and flat text
    checkpoints use ``model.`` — the same community-checkpoint rewrite the
    Qwen3.5 weights mapper applies.
    """
    if name.startswith(_VL_TEXT_PREFIX):
        return "model." + name[len(_VL_TEXT_PREFIX) :]
    return name


def lwd_should_skip_weight(name: str, role: str | None) -> bool:
    """Whether checkpoint tensor ``name`` belongs to a module ``role`` skips.

    All matches are on dot boundaries, so ``visual.`` never matches
    ``visualizer.``. Checkpoint name layouts covered:

    - flat: the key prefixes the name (``model.layers.0...`` for the
      ``model.layers`` container, ``visual.merger...`` for the tower);
    - VL nesting: ``model.language_model.*`` is unwrapped to ``model.*``
      before matching (Qwen3-VL / GLM-4.5V checkpoints);
    - native DeepSeek, which drops the leading ``model`` component
      (``layers.3.mlp.w1``, ``norm.weight``): matched by the bare last key
      component at the head of the name only, so tower-internal names such
      as ``visual.merger.norm.weight`` are never touched.

    A module prefix of the unwrapped name may also end with the key, so
    vLLM module-tree paths (``language_model.model.norm.weight``) keep
    matching defensively.
    """
    if role is None:
        return False
    name = _unwrap_vl_text_prefix(name)
    for key in parse_skip_keys(role):
        bare = key.rsplit(".", 1)[-1]
        if (
            name == key
            or name.startswith(key + ".")
            or _matches_module_suffix(name, key)
            or name.startswith(bare + ".")
        ):
            return True
    return False
