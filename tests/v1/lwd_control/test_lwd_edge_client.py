# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_client import (
    LwdEdgePrefixClient,
)


class _Content:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.limit: int | None = None

    async def read(self, limit: int) -> bytes:
        self.limit = limit
        return self._payload


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.content = _Content(payload)

    async def read(self, *args: object) -> bytes:
        # aiohttp's ClientResponse.read() accepts no size argument; mirror
        # that so a regression to response.read(limit) fails here.
        raise TypeError(
            "ClientResponse.read() takes 1 positional argument but 2 were given"
        )


def test_error_body_reads_the_content_stream():
    response = _Response(b'{"error":"boom"}\n')

    body = asyncio.run(LwdEdgePrefixClient._read_error_body(response))

    assert body == '{"error":"boom"}'
    assert response.content.limit == 512
