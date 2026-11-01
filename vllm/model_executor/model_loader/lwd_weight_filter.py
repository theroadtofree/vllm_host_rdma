# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skip checkpoint tensors belonging to modules an LWD role does not build."""

from __future__ import annotations

from collections.abc import Iterable


def parse_skip_keys(placeholder_names: Iterable[str]) -> list[list[str]]:
    """Convert dotted placeholder module names into component keys."""
    return [name.rstrip(".").split(".") for name in placeholder_names if name]


def name_is_skipped(weight_name: str, skip_keys: list[list[str]]) -> bool:
    """Return whether a weight belongs to one of the skipped modules."""
    if not skip_keys:
        return False
    parts = weight_name.split(".")
    for key in skip_keys:
        key_len = len(key)
        if key_len == 0 or key_len > len(parts):
            continue
        for start in range(len(parts) - key_len + 1):
            if parts[start : start + key_len] == key:
                return True
    return False
