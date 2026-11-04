# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drop checkpoint tensors that belong to modules this LWD role does not build.

LWD (layerwise disaggregated) edge/cloud deployments construct only the
modules their role actually runs, so the model contains ``PPMissingLayer`` /
``StageMissingLayer`` placeholders (see
:mod:`vllm.model_executor.models.utils`).  When such a process is still pointed
at a full checkpoint -- the usual state while bringing a deployment up -- the
loader can skip the placeholders' weights *before* they are read from disk.
This mirrors :mod:`vllm.model_executor.model_loader.ep_weight_filter`, which
does the same for experts a rank does not own.

Matching is component-wise rather than prefix-based on purpose:

* the checkpoint name and the vLLM module name do not always agree on their
  leading components -- a multimodal checkpoint stores the tower as
  ``model.visual.*`` while the vLLM module is ``visual``, and DeepSeek's native
  checkpoints drop the ``model.`` prefix entirely (``layers.3.mlp`` for the
  module ``model.layers.3``) -- so a prefix test would silently miss them;
  a leading ``model`` component is therefore ignored on both sides;
* component matching keeps ``layers.3`` from matching ``layers.30``.

Checkpoint layout note: :meth:`DefaultModelLoader._prepare_weights` enumerates
the shards that are actually present on disk and only uses
``model.safetensors.index.json`` to drop duplicates, so a model directory that
holds a subset of the shards (or newly written ones) loads fine.  When an index
file is kept, every shard that must be read has to be listed in its
``weight_map``: an unlisted file is filtered out and, when nothing is left, the
loader raises ``Cannot find any model weights``.  ``config.json`` -- and the
tokenizer/processor configs -- must stay in place, since the model class is
built from them.
"""

from __future__ import annotations

from collections.abc import Iterable


def _strip_model_prefix(parts: list[str]) -> list[str]:
    """Drop a leading ``model`` component (present in HF, absent in native)."""
    return parts[1:] if len(parts) > 1 and parts[0] == "model" else parts


def parse_skip_keys(placeholder_names: Iterable[str]) -> list[list[str]]:
    """Convert dotted placeholder module names into component keys.

    Args:
        placeholder_names: Module names as returned by
            :func:`vllm.model_executor.models.utils.get_pp_missing_layer_names`,
            i.e. dotted paths, usually with a trailing dot.

    Returns:
        One list of path components per name; empty names are dropped.
    """
    return [
        _strip_model_prefix(name.rstrip(".").split("."))
        for name in placeholder_names
        if name
    ]


def name_is_skipped(weight_name: str, skip_keys: list[list[str]]) -> bool:
    """Return whether *weight_name* lives inside one of the skipped modules.

    Args:
        weight_name: Checkpoint tensor name, e.g. ``model.layers.3.mlp.w1`` or
            ``layers.3.mlp.w1`` (native checkpoint layout).
        skip_keys: Component keys produced by :func:`parse_skip_keys`.

    Returns:
        ``True`` when some key matches a contiguous run of components of
        *weight_name*.
    """
    if not skip_keys:
        return False
    parts = _strip_model_prefix(weight_name.split("."))
    num_parts = len(parts)
    for key in skip_keys:
        key_len = len(key)
        if key_len == 0 or key_len > num_parts:
            continue
        for start in range(num_parts - key_len + 1):
            if parts[start : start + key_len] == key:
                return True
    return False
