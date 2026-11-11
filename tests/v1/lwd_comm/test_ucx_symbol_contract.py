# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard: libucp is reached only through real ``ucp_*`` entry points.

ctypes resolves *any* attribute on a ``CDLL`` as a symbol lookup, so a
Python-style attribute left on the handle (the old ``_ucp._ucp_init_major``
version cache) only fails on a machine that has libucp: ``connect failed
(AttributeError(... undefined symbol: _ucp_init_major))``, and the edge
waits for a handshake that never starts.

Neither mypy (``CDLL.__getattr__`` is typed as returning ``Any``) nor ruff
sees this, so it is pinned here: every attribute read on the handle must be
a libucp symbol.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

DATA_COMM = (
    Path(__file__).resolve().parents[3] / "vllm" / "v1" / "lwd_comm" / "data_comm"
)

# The handle appears either as the module global or through the accessor.
HANDLE_NAMES = frozenset({"_ucp", "ucp"})
HANDLE_CALLS = frozenset({"_require_ucp"})


def _is_handle(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in HANDLE_NAMES
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in HANDLE_CALLS
    )


@pytest.mark.parametrize(
    "source", sorted(DATA_COMM.glob("ucx_*.py")), ids=lambda path: path.name
)
def test_libucp_attributes_are_entry_points(source: Path) -> None:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    offenders = [
        f"{node.attr!r} on line {node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and _is_handle(node.value)
        and not node.attr.startswith("ucp_")
    ]
    assert not offenders, (
        f"{source.name} reads non-libucp attributes off the handle: {offenders}. "
        "ctypes treats those as symbol lookups and they fail only where libucp "
        "is installed."
    )
