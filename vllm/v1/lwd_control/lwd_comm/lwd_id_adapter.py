# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request ID namespacing for requests received from multiple edges."""

from __future__ import annotations

_LWD_ID_PREFIX = "e"


def wrap_req_id(edge_id: int, req_id: str) -> str:
    """Build the cloud-local canonical ID for an edge request."""
    return f"{_LWD_ID_PREFIX}{edge_id}-{req_id}"


def parse_edge_id(wrapped: str) -> tuple[int, str]:
    """Parse an ID produced by :func:`wrap_req_id`."""
    prefix, sep, rest = wrapped.partition("-")
    if (
        sep == "-"
        and len(prefix) > 1
        and prefix[0] == _LWD_ID_PREFIX
        and prefix[1:].isdigit()
    ):
        return int(prefix[1:]), rest
    return 0, wrapped


def unwrap_req_id(wrapped: str) -> str:
    return parse_edge_id(wrapped)[1]
