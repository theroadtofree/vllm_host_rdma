# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Nonce-scoped edge prefix negotiation and READY timeout handling."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from vllm.lwd_logger import lwd_logger
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.lwd_control.lwd_comm.lwd_messages import LwdPrefixReadyNotify

if TYPE_CHECKING:
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
        LwdEdgeEngineCore,
    )
    from vllm.v1.lwd_control.lwd_comm.lwd_messages import LwdCloudParamsNotify
    from vllm.v1.request import Request


class LwdEdgePrefixControl:
    """Own prefix waiters on the engine thread; timers only post wakeups."""

    def __init__(self, engine: LwdEdgeEngineCore, timeout_s: float) -> None:
        self._engine = engine
        self._timeout_s = timeout_s
        self._nonce = 0
        self._hash_block_size = 0
        self._pending: dict[str, tuple[Request, threading.Timer]] = {}

    @property
    def pending(self) -> dict[str, tuple[Request, threading.Timer]]:
        return self._pending

    @property
    def hash_block_size(self) -> int:
        return self._hash_block_size

    def configure(self, params: LwdCloudParamsNotify) -> None:
        if params.hash_block_size <= 0:
            raise ValueError(
                "cloud advertised prefix caching without a valid "
                f"hash_block_size ({params.hash_block_size})"
            )
        engine = self._engine
        hash_fn = get_hash_fn_by_name(
            engine.vllm_config.cache_config.prefix_caching_hash_algo
        )
        if engine.request_block_hasher is None:
            init_none_hash(hash_fn)
        engine.request_block_hasher = get_request_block_hasher(
            params.hash_block_size, hash_fn
        )
        self._hash_block_size = params.hash_block_size
        lwd_logger.info(
            "[prefix] edge hasher rebuilt: hash_block_size=%d "
            "alignment=%d(required=%s) use_eagle=%s",
            params.hash_block_size,
            params.prefix_alignment_tokens,
            params.prefix_alignment_required,
            params.prefix_use_eagle,
        )

    def add_request(self, request: Request) -> None:
        if request.request_id in self._pending:
            raise ValueError(f"duplicate prefix waiter: {request.request_id}")
        engine = self._engine
        self._nonce += 1
        request.lwd_prefix_nonce = self._nonce
        request.block_hashes = []
        request._block_hasher = engine.request_block_hasher
        request.update_block_hashes()
        expired = LwdPrefixReadyNotify(
            request.request_id,
            0,
            request.lwd_prefix_nonce,
            "prefix reply timed out",
        )
        timer = threading.Timer(
            self._timeout_s,
            engine.input_queue.put_nowait,
            args=((EngineCoreRequestType.WAKEUP, expired),),
        )
        timer.daemon = True
        self._pending[request.request_id] = (request, timer)
        try:
            if not engine._lwd_probe_manager.submit(request):
                engine._lwd_add_request_after_probe(request)
        except Exception:
            self._pending.pop(request.request_id, None)
            timer.cancel()
            raise
        timer.start()

    def handle_client_request(
        self, request_type: EngineCoreRequestType, reply: Any
    ) -> bool:
        if request_type != EngineCoreRequestType.WAKEUP or not isinstance(
            reply, LwdPrefixReadyNotify
        ):
            return False
        pending = self._pending.get(reply.request_id)
        if pending is None or pending[0].lwd_prefix_nonce != reply.nonce:
            return True
        request, timer = self._pending.pop(reply.request_id)
        timer.cancel()
        if reply.error:
            lwd_logger.error(
                "[prefix] READY error req=%s: %s", reply.request_id, reply.error
            )
            self._rollback_and_abort(request)
            return True
        if self._engine.scheduler.lwd_edge_prefix_ready(request, reply.hit_tokens):
            lwd_logger.info(
                "[prefix] READY req=%s hit_tokens=%d",
                reply.request_id,
                reply.hit_tokens,
            )
            self._engine._lwd_enqueue_request(request)
        else:
            self._rollback_and_abort(request)
        return True

    def _rollback_and_abort(self, request: Request) -> None:
        assert self._engine.scheduler.lwd_edge_prefix_ready(request, 0)
        self._engine._lwd_enqueue_request(request)
        self._engine._lwd_abort_internally(request)

    def prepare_abort(self, request_ids: list[str]) -> None:
        for request_id in request_ids:
            pending = self._pending.pop(request_id, None)
            if pending is None:
                continue
            request, timer = pending
            timer.cancel()
            assert self._engine.scheduler.lwd_edge_prefix_ready(request, 0)
            self._engine._lwd_enqueue_request(request)

    def shutdown(self) -> None:
        for _request, timer in self._pending.values():
            timer.cancel()
