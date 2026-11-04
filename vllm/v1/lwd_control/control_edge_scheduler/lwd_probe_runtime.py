# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Background runtime for asynchronous LWD gateway probes."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import Any

import aiohttp


class LwdProbeRuntime:
    """Own a bounded probe event loop outside the engine thread."""

    def __init__(self, *, max_connections: int = 0, max_inflight: int = 64) -> None:
        """max_connections: 0 = unlimited (aiohttp TCPConnector semantics).

        The probe pool is shared with usage-drain streams, each of which
        holds one connection until the request settles on the cloud. Any
        finite cap therefore lets long-lived drains starve new probes into
        the connect timeout under load; the old baseline deliberately ran
        unlimited (_LWD_PROBE_MAX_CONNS = 0)."""
        if max_connections < 0 or max_inflight <= 0:
            raise ValueError(
                "probe connection limit must be non-negative and the "
                "inflight limit must be positive"
            )
        self._max_connections = max_connections
        self._loop = asyncio.new_event_loop()
        self._session: aiohttp.ClientSession | None = None
        self._inflight: asyncio.Semaphore | None = None
        self._max_inflight = max_inflight
        self._thread = threading.Thread(
            target=self._run, name="lwd-prefix-probe", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._inflight = asyncio.Semaphore(self._max_inflight)
        self._loop.run_forever()

    async def ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=self._max_connections)
            )
        return self._session

    def submit(self, coroutine: Coroutine[Any, Any, Any]) -> Future[Any]:
        """Submit one probe, bounding queued and executing HTTP work."""

        async def bounded() -> Any:
            assert self._inflight is not None
            async with self._inflight:
                return await coroutine

        return asyncio.run_coroutine_threadsafe(bounded(), self._loop)

    def stop(self) -> None:
        """Close the HTTP session and stop the loop without blocking the engine."""

        def cleanup() -> None:
            if self._session is not None:
                asyncio.ensure_future(self._session.close())
            self._loop.call_later(1.0, self._loop.stop)

        try:
            self._loop.call_soon_threadsafe(cleanup)
        except RuntimeError:
            pass
