# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""UniProcExecutor variant scoped to LWD host comm (``data_plane_comm=host``).

The LWD edge's forwards pay host-blocking comm waits (the DOWN frame waits
of the UCX data plane), so its non_block RPCs must actually overlap the
engine thread's scheduling. Native UniProcExecutor answers non_block RPCs
inline with an already-done future, which silently degrades
step_with_batch_queue into synchronous stepping; this subclass runs them
on a single background worker instead. Wired in by VllmConfig when the LWD
topology is host comm and the engine would run native uniproc
(world_size == 1, e.g. the tp=1 edge); device comm and non-LWD engines
keep the native executor untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.serial_utils import run_method


class LwdHostUniProcExecutor(UniProcExecutor):
    """Pool-backed UniProcExecutor for world_size==1 LWD host-comm engines."""

    def _init_executor(self) -> None:
        super()._init_executor()
        # Lazy single-worker pool for non_block RPCs (see collective_rpc).
        self._rpc_thread: ThreadPoolExecutor | None = None

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        single_value: bool = False,
    ) -> Any:
        if kwargs is None:
            kwargs = {}
        if not non_block:
            # Sync RPCs stay native: inline on the calling thread.
            return super().collective_rpc(
                method,
                timeout=timeout,
                args=args,
                kwargs=kwargs,
                non_block=False,
                single_value=single_value,
            )

        # Async path: run on the single background worker, never inline.
        # The native inline + already-done future silently degraded
        # step_with_batch_queue into synchronous stepping: every batch's
        # full cost -- including host-blocking comm waits (the LWD edge's
        # DOWN frame waits) -- serialized on the engine thread, and
        # multi-instance throughput halved. One worker keeps strict FIFO
        # submission, which is the overlapped scheduler's harvest-order
        # contract (futures complete in dispatch order).
        if self._rpc_thread is None:
            self._rpc_thread = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="LwdHostUniProcRpc"
            )

        def _rpc() -> Any:
            # Same resolution semantics as the sync branch, just on the
            # pool thread: an AsyncModelRunnerOutput blocks the pool
            # worker until its output is ready (ordering preserved), the
            # engine's harvest sees a plain ModelRunnerOutput as before.
            result = run_method(self.driver_worker, method, args, kwargs)
            if isinstance(result, AsyncModelRunnerOutput):
                result = result.get_output()
            return result if single_value else [result]

        return self._rpc_thread.submit(_rpc)

    def shutdown(self) -> None:
        rpc_thread = getattr(self, "_rpc_thread", None)
        if rpc_thread is not None:
            self._rpc_thread = None
            # Let queued RPCs finish before the worker they target goes away.
            rpc_thread.shutdown(wait=True)
        super().shutdown()
