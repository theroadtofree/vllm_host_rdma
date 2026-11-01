# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cloud engine cores serving multiple edge connections per DP rank.

Range notifies post extended early-receive hints, request notifies become
native ADDs with prompt-length and M-RoPE metadata, and abort notifies use the
native ABORT path. Step metadata is flushed to the edge in post_step.

The LWD seams live in LwdCloudEngineCoreMixin and are orthogonal to the
engine base: LwdCloudEngineCore (EngineCoreProc) serves one DP rank of a
cloud instance and non-MoE multi-DP deployments; LwdCloudDPEngineCore
(DPEngineCoreProc) serves MoE multi-DP deployments, inheriting the native
DP semantics (wave state machine, dummy batches, START_DP_WAVE, stats
publishing) unchanged.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterable
from typing import Any

import msgspec
import torch

from vllm.config.lwd import LwdConnectionId
from vllm.lwd_logger import lwd_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc, EngineShutdownState
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_scheduler import (
    LwdCloudScheduler,
)
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_prefix_cache import (
    init_lwd_prefix_cache,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_engine import (
    LwdBaseEngineCore,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    lwd_spec_factor,
)
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdPeer
from vllm.v1.lwd_control.lwd_comm.lwd_id_adapter import unwrap_req_id, wrap_req_id
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_SAMPLING_DEFAULTS,
    LwdAbortNotify,
    LwdC2eNotify,
    LwdCloudParamsNotify,
    LwdCloudToEdgeMessage,
    LwdEdgeToCloudMessage,
    LwdFinishNotify,
    LwdGenChainNotify,
    LwdMessage,
    LwdPrefixReadyNotify,
    LwdRangeNotify,
    LwdRefillNotify,
    LwdRequestNotify,
    lwd_decode_edge_to_cloud,
)
from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
from vllm.v1.request import Request

# Sleep between retries of a full outbound queue: step metadata must not be
# lost, the edge posts its receive of the exact size from it.
_LWD_SEND_RETRY_SLEEP_S = 0.05
_LWD_INGEST_JOIN_TIMEOUT_S = 5.0


_LwdIngestItem = tuple[LwdPeer, LwdEdgeToCloudMessage]


class LwdCloudEngineCoreMixin(LwdBaseEngineCore):
    """The LWD seams of a cloud engine, shared by both engine bases."""

    lwd_scheduler_cls = LwdCloudScheduler
    scheduler: LwdCloudScheduler

    def __init__(self, *args: Any, lwd_dp_size: int, **kwargs: Any) -> None:
        vllm_config = kwargs["vllm_config"]
        if (
            vllm_config.cache_config.enable_prefix_caching
            and getattr(vllm_config, "kv_transfer_config", None) is not None
        ):
            raise ValueError(
                "[LWD] prefix caching with KV connector is unsupported: "
                "external hits bypass the negotiated READY boundary"
            )
        # Pre-recv hint MQ must exist before super().__init__ spawns the
        # executor workers (same process): create_hint_mq() publishes the
        # handle through the environment, and only processes spawned after
        # that see it. The sideband MQ exists because rpc_broadcast_mq is
        # drained by the worker's single-threaded busy loop, which sits
        # inside a several-hundred-ms prefill forward - hints queued behind
        # it would lose the whole point of pre-posting the receive.
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import create_hint_mq

        dp_idx = vllm_config.parallel_config.data_parallel_index
        connections = tuple(
            connection.connection_id
            for connection in vllm_config.lwd_config.connections
            if connection.dp_idx == dp_idx
        ) or (LwdConnectionId(0, vllm_config.lwd_config.instance_id, dp_idx),)
        self._lwd_hint_mqs = {
            connection: create_hint_mq(connection) for connection in connections
        }
        self._lwd_hint_mq = self._lwd_hint_mqs[connections[0]]
        super().__init__(*args, lwd_dp_size=lwd_dp_size, **kwargs)
        # The cloud endpoint worker created its done-report MQ during
        # init_device; now that the executor is up, fetch its handle over
        # the executor's collective RPC (the executor itself stays free of
        # LWD plumbing). Exactly one worker - the endpoint rank - must
        # report a handle; anything else is a wiring error.
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        self._lwd_done_mqs = {}
        for connection in self._lwd_hint_mqs:
            handles = [
                handle
                for handle in self.model_executor.collective_rpc(
                    "lwd_early_recv_done_handle", args=(connection,)
                )
                if handle is not None
            ]
            if len(handles) != 1:
                raise RuntimeError(
                    "[LWD] early receive expects one endpoint worker reporting "
                    f"completions for {connection}, got {len(handles)}"
                )
            self._lwd_done_mqs[connection] = MessageQueue.create_from_handle(
                handles[0], 0
            )
        self._lwd_done_mq = next(iter(self._lwd_done_mqs.values()))
        lwd_logger.info(
            "[pre-recv] %d hint MQ(s) created, done MQs attached via RPC",
            len(self._lwd_hint_mqs),
        )

    def _lwd_init_prefix_cache(self) -> None:
        state = init_lwd_prefix_cache(
            self.vllm_config,
            self.scheduler.kv_cache_config,
            self.scheduler.kv_cache_manager,
            use_eagle=bool(getattr(self.scheduler, "use_eagle", False)),
        )
        self._lwd_prefix_state = state
        self._lwd_prefix_enabled = state.enabled
        self._lwd_hash_block_size = state.hash_block_size
        if state.enabled:
            self.scheduler.lwd_prefix_reservations = state.reservations
            self.scheduler.lwd_refill_sender = self._lwd_send_refill

    def _lwd_assemble(self) -> None:
        self._lwd_init_prefix_cache()
        self._lwd_ingest_thread: threading.Thread | None = None
        self._lwd_edges: set[LwdPeer] = set()
        self._lwd_prompt_lens: dict[str, int] = {}
        self._lwd_retired_prompt_lens: dict[str, int] = {}
        self._lwd_request_nonces: dict[str, int] = {}
        self._lwd_retired_nonces: dict[str, int] = {}
        self._lwd_retired_req_ids: set[str] = set()
        self._lwd_rejected_nonces: dict[str, int] = {}
        self._lwd_request_routes: dict[str, LwdPeer] = {}
        self._lwd_pending_seqnos: dict[
            LwdConnectionId, dict[int, set[str]]
        ] = {}
        self._lwd_request_seqnos: dict[str, set[int]] = {}
        self._lwd_latest_seqnos: dict[LwdConnectionId, int] = {}
        self._lwd_failed_connections: set[LwdConnectionId] = set()
        self._lwd_edge_chains: dict[str, tuple[int, list[bytes]]] = {}
        self.scheduler.lwd_edge_chains = self._lwd_edge_chains
        self.scheduler.lwd_hash_block_size = 0
        self._lwd_prefix_hits: dict[str, int] = {}
        self._lwd_prefix_coordinator = None
        prefix_hash_block_size = 0
        coordination = self.lwd_config.coordination
        if coordination is not None and coordination.enabled:
            from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
            from vllm.v1.lwd_control.control_cloud_scheduler.lwd_prefix_coordinator import (
                LwdPrefixCoordinator,
            )

            hash_block_size = resolve_kv_cache_block_sizes(
                self.scheduler.kv_cache_config, self.vllm_config
            )[1]
            prefix_hash_block_size = hash_block_size
            self.scheduler.lwd_hash_block_size = hash_block_size
            self._lwd_prefix_coordinator = LwdPrefixCoordinator(
                hash_block_size=hash_block_size,
                instance_id=f"cloud-{self.lwd_config.instance_id}",
                kv_cache_manager=self.scheduler.kv_cache_manager,
                cloud_id=self.lwd_config.instance_id,
            )
        self._lwd_coord_pending: queue.Queue[dict[str, Any]] = queue.Queue()
        self._lwd_control_processor = None
        self.cloud_control_event_queue = getattr(
            self, "cloud_control_event_queue", None
        )
        command_queue = getattr(self, "cloud_control_command_queue", None)
        if (
            self._lwd_prefix_coordinator is not None
            and command_queue is not None
            and self.cloud_control_event_queue is not None
        ):
            from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_control import (
                LwdCloudControlProcessor,
            )

            self._lwd_control_processor = LwdCloudControlProcessor(
                self._lwd_coord_pending, self.cloud_control_event_queue
            )
            threading.Thread(
                target=self._lwd_coord_wake_loop,
                name="lwd-ctrl-wake",
                daemon=True,
            ).start()
        self._lwd_engine_index = int(
            self.vllm_config.parallel_config.data_parallel_index
        )
        sched_config = self.vllm_config.scheduler_config
        # Scheduling limits are checked by the edge. The coordinator address
        # lets the edge
        # trigger the first DP wave; its published host may not be routable
        # from the edge, so it is rewritten to this DP's topology address.
        self._lwd_cloud_params = LwdCloudParamsNotify(
            max_num_batched_tokens=sched_config.max_num_batched_tokens,
            max_num_seqs=sched_config.max_num_seqs,
            num_speculative_tokens=lwd_spec_factor(self.vllm_config) - 1,
            dp_coordinator_addr=self._lwd_dp_coordinator_addr(),
            prefix_caching=self._lwd_prefix_enabled,
            hash_block_size=self._lwd_hash_block_size,
            prefix_alignment_tokens=self._lwd_prefix_state.alignment_tokens,
            prefix_alignment_required=self._lwd_prefix_state.alignment_required,
            prefix_use_eagle=self._lwd_prefix_state.use_eagle,
            prefix_hash_block_size=prefix_hash_block_size,
        )
        self._lwd_register_recv_hint()
        # Edge messages are handled off the IO thread: building a request
        # allocates its placeholder embeds and the pre-recv hint may block,
        # which must not delay step metadata going out.
        self._lwd_ingest_queue: queue.SimpleQueue[_LwdIngestItem | None] = (
            queue.SimpleQueue()
        )
        self._lwd_ingest_thread = threading.Thread(
            target=self._lwd_ingest_loop, name="lwd-cloud-ingest", daemon=True
        )
        self._lwd_ingest_thread.start()
        plane = self._lwd_start_control_plane()
        lwd_logger.info("[cloud] engine assembled: control plane on %s", plane.endpoint)

    def _lwd_dp_coordinator_addr(self) -> str:
        """Routable coordinator stats endpoint, or empty without one."""
        address = getattr(self, "frontend_stats_publish_address", None)
        if not self.has_coordinator or not address:
            return ""
        if not str(address).startswith("tcp://"):
            return ""
        _, _, port = str(address).rpartition(":")
        dp_rank = self.vllm_config.parallel_config.data_parallel_index
        local = self.lwd_config.instance.dp[dp_rank]
        return f"tcp://{local.addr}:{port}"

    def _lwd_decode(self, payload: bytes) -> LwdMessage:
        return lwd_decode_edge_to_cloud(payload)

    def _lwd_handle_peer_registered(self, peer: LwdPeer) -> None:
        """Engine thread: remember the edge and send it the scheduling
        limits before anything else."""
        self._lwd_edges.add(peer)
        self._lwd_send_until_queued(peer, self._lwd_cloud_params)
        lwd_logger.info(
            "[cloud] edge %s registered; scheduling limits sent: %s",
            peer,
            self._lwd_cloud_params,
        )

    def _lwd_send_until_queued(
        self, peer: LwdPeer, message: LwdCloudToEdgeMessage
    ) -> None:
        """Retry a full outbound queue until the message is queued; gives up
        with an error once shutdown is requested, so a stalled edge cannot
        block it."""
        while not self._lwd_send(peer, message):
            if self.shutdown_state != EngineShutdownState.RUNNING:
                lwd_logger.error(
                    "[cloud] %s to edge %s dropped: shutting down with the "
                    "outbound queue full",
                    type(message).__name__,
                    peer,
                )
                return
            time.sleep(_LWD_SEND_RETRY_SLEEP_S)

    def _lwd_send_prefix_ready(
        self, request_id: str, hit_tokens: int, nonce: int, error: str = ""
    ) -> None:
        peer = self._lwd_request_routes.get(request_id)
        if peer is None:
            raise RuntimeError(f"[LWD] prefix reply route missing for {request_id}")
        self._lwd_send_until_queued(
            peer,
            LwdPrefixReadyNotify(
                unwrap_req_id(request_id), hit_tokens, nonce, error
            ),
        )
        lwd_logger.info(
            "[prefix] READY req=%s hit=%d nonce=%d error=%r",
            request_id,
            hit_tokens,
            nonce,
            error,
        )

    def _lwd_send_refill(
        self, request_id: str, start_row: int, end_row: int, prefix_nonce: int
    ) -> None:
        peer = self._lwd_request_routes.get(request_id)
        if peer is None:
            raise RuntimeError(f"[LWD] refill route missing for {request_id}")
        self._lwd_send_until_queued(
            peer,
            LwdRefillNotify(
                unwrap_req_id(request_id), start_row, end_row, prefix_nonce
            ),
        )
        lwd_logger.info(
            "[prefix] REFILL req=%s rows=[%d, %d) nonce=%d",
            request_id,
            start_row,
            end_row,
            prefix_nonce,
        )

    def add_request(self, request: Request, request_wave: int = 0) -> None:
        if self._lwd_prefix_coordinator is not None:
            super().add_request(request, request_wave)
            return
        nonce = getattr(request, "lwd_prefix_nonce", 0)
        if not nonce:
            if self._lwd_prefix_enabled:
                lwd_logger.error(
                    "[prefix] reject req=%s: missing prefix nonce",
                    request.request_id,
                )
                return
            super().add_request(request, request_wave)
            return

        request_id = request.request_id
        if request_id in getattr(self.scheduler, "requests", {}):
            self._lwd_send_prefix_ready(
                request_id, 0, nonce, error="duplicate active request id"
            )
            return
        reservations = getattr(self.scheduler, "lwd_prefix_reservations", None)
        if reservations is None:
            self._lwd_send_prefix_ready(
                request_id, 0, nonce, error="cloud prefix caching is disabled"
            )
            return
        if not getattr(request, "lwd_wire_hashes_adopted", False):
            self._lwd_send_prefix_ready(
                request_id,
                0,
                nonce,
                error="prefix hash chain does not match cloud hash block size",
            )
            return
        try:
            hit_tokens = reservations.reserve(request)
        except Exception as exc:
            lwd_logger.exception("[prefix] reserve failed req=%s", request_id)
            self._lwd_send_prefix_ready(request_id, 0, nonce, error=str(exc))
            return
        try:
            super().add_request(request, request_wave)
        except Exception:
            reservations.release(request_id)
            self._lwd_send_prefix_ready(
                request_id, 0, nonce, error="cloud admission failed"
            )
            raise
        self._lwd_send_prefix_ready(request_id, hit_tokens, nonce)

    def _lwd_register_recv_hint(self) -> None:
        """[LWD][pre-recv] Early receive: the hint delivery channel =
        engine -> cloud endpoint worker comm thread. It goes through
        the sideband MQ this engine creates in __init__ (not
        rpc_broadcast_mq: that one is consumed single-threaded by the
        busy_loop and gets blocked by a several-hundred-ms prefill
        forward, so the hint would no longer be "early")."""
        if self._lwd_hint_mqs:
            from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
                register_hint_sender,
            )

            for connection, hint_mq in self._lwd_hint_mqs.items():
                if hint_mq is None:
                    continue
                # Blocking enqueue: once the gate is on, hints are a
                # correctness dependency; dropping one stalls its connection.
                register_hint_sender(
                    lambda hint, mq=hint_mq: mq.enqueue(hint), connection
                )
            lwd_logger.info(
                "[pre-recv] hint senders registered for %d connection(s)",
                len(self._lwd_hint_mqs),
            )
        else:
            # PD_mix is the only scheduling mode: a missing hint MQ =
            # hints never go out and the gate never opens; this must be
            # explicitly visible
            lwd_logger.warning(
                "[pre-recv] hint MQ missing: pre-recv gate will never open"
            )

    def _lwd_dispatch_message(self, peer: LwdPeer, msg: LwdMessage) -> bool:
        """IO thread: hand all edge business notifies to the ingest thread."""
        if not isinstance(
            msg,
            (
                LwdRangeNotify,
                LwdRequestNotify,
                LwdAbortNotify,
                LwdGenChainNotify,
                LwdFinishNotify,
            ),
        ):
            return False
        self._lwd_ingest_queue.put((peer, msg))
        return True

    def _lwd_ingest_loop(self) -> None:
        while (item := self._lwd_ingest_queue.get()) is not None:
            try:
                self._lwd_handle_edge_message(*item)
            except Exception as e:
                lwd_logger.exception(
                    "[cloud] ingest of %s failed", type(item[1]).__name__
                )
                self._lwd_fail(f"ingest of {type(item[1]).__name__}: {e!r}")
                return

    def _lwd_handle_edge_message(
        self, peer: LwdPeer, msg: LwdEdgeToCloudMessage
    ) -> None:
        lwd_logger.info(
            "[cloud] ingest msg=%s seqno=%s",
            type(msg).__name__,
            getattr(msg, "seqno", "-"),
        )
        connection_id = self._lwd_connection_of(peer)
        if connection_id in self._lwd_failed_connections:
            if isinstance(msg, LwdRequestNotify):
                from vllm.v1.engine import FinishReason

                self._lwd_send_until_queued(
                    peer,
                    LwdC2eNotify(
                        hidden_num_elements=0,
                        top_id_ths=[[]],
                        num_accepted_tokens=[0],
                        req_ids=[msg.request_id],
                        finish_reasons=[int(FinishReason.ERROR)],
                    ),
                )
            return
        if isinstance(msg, LwdRangeNotify):
            # Every request id in the batch enters the cloud namespace once.
            msg = msgspec.structs.replace(
                msg,
                items=[
                    msgspec.structs.replace(
                        item, request_id=wrap_req_id(peer.instance_id, item.request_id)
                    )
                    for item in msg.items
                ],
            )
            self._lwd_handle_range_notify(connection_id, msg)
            return
        if not isinstance(
            msg,
            (
                LwdRequestNotify,
                LwdAbortNotify,
                LwdGenChainNotify,
                LwdFinishNotify,
            ),
        ):
            return
        canonical_id = wrap_req_id(peer.instance_id, msg.request_id)
        msg = msgspec.structs.replace(msg, request_id=canonical_id)
        if isinstance(msg, LwdRequestNotify):
            self._lwd_request_routes[canonical_id] = peer
            self._lwd_handle_request_notify(msg)
        elif isinstance(msg, LwdAbortNotify):
            self._lwd_handle_abort_notify(msg)
        elif isinstance(msg, LwdGenChainNotify):
            # Generation notifications carry the full chain, not a delta.
            self._lwd_edge_chains[canonical_id] = (
                msg.block_size,
                list(msg.block_hashes),
            )
        elif isinstance(msg, LwdFinishNotify):
            coordinator = getattr(self, "_lwd_prefix_coordinator", None)
            if coordinator is not None:
                coordinator.apply_finish_chain(
                    canonical_id,
                    msg.prompt_tokens,
                    msg.completion_tokens,
                    list(msg.full_block_hashes),
                    msg.publish_cache,
                )
            self._lwd_edge_chains.pop(canonical_id, None)

    def _lwd_connection_of(self, peer: LwdPeer) -> LwdConnectionId:
        return LwdConnectionId(
            peer.instance_id, self.lwd_config.instance_id, peer.dp_rank
        )

    def _lwd_handle_range_notify(
        self, connection_id: LwdConnectionId, notify: LwdRangeNotify
    ) -> None:
        """Post each new extended hint exactly once, per source connection."""
        rejected = getattr(self, "_lwd_rejected_nonces", {})
        if any(
            rejected.get(item.request_id) == item.prefix_nonce for item in notify.items
        ):
            return
        current_nonces = getattr(self, "_lwd_request_nonces", {})
        if any(
            item.request_id in current_nonces
            and current_nonces[item.request_id] != item.prefix_nonce
            for item in notify.items
        ):
            lwd_logger.warning("[cloud] ignore stale RangeNotify seqno=%d", notify.seqno)
            return
        latest = self._lwd_latest_seqnos.get(connection_id, -1)
        if notify.seqno <= latest:
            lwd_logger.info(
                "[cloud] ignore duplicate RangeNotify seqno=%d", notify.seqno
            )
            return
        items = notify.items
        request_ids = [item.request_id for item in items]
        prompt_lens = [
            self._lwd_prompt_lens.get(
                request_id, self._lwd_retired_prompt_lens.get(request_id, 0)
            )
            for request_id in request_ids
        ]
        if any(prompt_len <= 0 for prompt_len in prompt_lens):
            raise RuntimeError(
                f"[LWD] RangeNotify seqno={notify.seqno} arrived before "
                "RequestNotify prompt lengths were recorded"
            )
        self._lwd_track_range(connection_id, notify.seqno, request_ids)
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
            make_recv_hint,
            post_irecv_hint,
        )

        num_tokens = sum(item.num_tokens for item in items)
        lwd_logger.info(
            "[Lwd][pre-recv] HINT seqno=%d num_tokens=%d ts=%.3f",
            notify.seqno,
            num_tokens,
            time.monotonic(),
        )
        hint_fields: dict[str, Any] = {
            "req_ids": request_ids,
            "rows": [item.num_tokens for item in items],
            "prompt_offsets": [item.offset for item in items],
            "prompt_lens": prompt_lens,
            "prefix_nonces": [item.prefix_nonce for item in items],
        }
        if any(item.is_refill for item in items):
            hint_fields["refills"] = [item.is_refill for item in items]
        post_irecv_hint(
            make_recv_hint(notify.seqno, num_tokens, connection_id, **hint_fields)
        )

    def _lwd_track_range(
        self,
        connection_id: LwdConnectionId,
        seqno: int,
        request_ids: list[str],
    ) -> None:
        self._lwd_latest_seqnos[connection_id] = seqno
        self._lwd_pending_seqnos.setdefault(connection_id, {})[seqno] = set(
            request_ids
        )
        for request_id in request_ids:
            self._lwd_request_seqnos.setdefault(request_id, set()).add(seqno)

    def _lwd_complete_seqno(
        self, connection_id: LwdConnectionId, seqno: int
    ) -> None:
        pending_by_seqno = self._lwd_pending_seqnos.get(connection_id)
        if pending_by_seqno is None:
            return
        request_ids = pending_by_seqno.pop(seqno, None)
        if not pending_by_seqno:
            self._lwd_pending_seqnos.pop(connection_id, None)
        if request_ids is None:
            return
        for request_id in request_ids:
            pending = self._lwd_request_seqnos.get(request_id)
            if pending is not None:
                pending.discard(seqno)
            if pending:
                continue
            self._lwd_request_seqnos.pop(request_id, None)
            if request_id in self._lwd_retired_req_ids:
                self._lwd_clear_retired(request_id)

    def _lwd_handle_request_notify(self, notify: LwdRequestNotify) -> None:
        request_id = notify.request_id
        current = getattr(self, "_lwd_request_nonces", {}).get(request_id)
        if (self._lwd_request_seqnos.get(request_id) and current is None) or (
            current is not None and current != notify.prefix_nonce
        ):
            self._lwd_rejected_nonces[request_id] = notify.prefix_nonce
            if notify.prefix_nonce:
                self._lwd_send_prefix_ready(
                    request_id,
                    0,
                    notify.prefix_nonce,
                    error="request id reused before previous generation drained",
                )
                return
            raise RuntimeError(
                f"[LWD] request_id {request_id} reused before previous generation drained"
            )
        coordinator = self._lwd_prefix_coordinator
        if coordinator is not None:
            hit = coordinator.claim(notify.request_id)
            if hit is not None:
                self._lwd_prefix_hits[notify.request_id] = hit
        self._lwd_prompt_lens[request_id] = notify.num_prompt_tokens
        if notify.prefix_nonce:
            self._lwd_request_nonces[request_id] = notify.prefix_nonce
        request = self._lwd_build_request(notify)
        self.input_queue.put_nowait((EngineCoreRequestType.ADD, (request, 0)))
        lwd_logger.info(
            "[cloud] request added req=%s tokens=%d nonce=%d",
            request_id,
            notify.num_prompt_tokens,
            notify.prefix_nonce,
        )
        if not self.engines_running:
            self.engines_running = True
            lwd_logger.info(
                "[cloud] first request wakes a paused wave (local fallback)"
            )

    def _lwd_handle_abort_notify(self, notify: LwdAbortNotify) -> None:
        request_id = notify.request_id
        current = getattr(self, "_lwd_request_nonces", {}).get(request_id)
        retired = getattr(self, "_lwd_retired_nonces", {}).get(request_id)
        if request_id in self._lwd_retired_req_ids and retired == notify.prefix_nonce:
            lwd_logger.info(
                "[cloud] ignore duplicate abort req=%s nonce=%d",
                request_id,
                notify.prefix_nonce,
            )
            return
        if (
            getattr(self, "_lwd_rejected_nonces", {}).get(request_id)
            == notify.prefix_nonce
        ):
            lwd_logger.info(
                "[cloud] ignore abort for rejected generation req=%s nonce=%d",
                request_id,
                notify.prefix_nonce,
            )
            return
        if current is not None and current != notify.prefix_nonce:
            lwd_logger.warning(
                "[cloud] ignore stale abort req=%s nonce=%d current=%d",
                request_id,
                notify.prefix_nonce,
                current,
            )
            return
        if current is None and retired is not None and retired != notify.prefix_nonce:
            lwd_logger.warning(
                "[cloud] ignore stale abort req=%s nonce=%d retired=%d",
                request_id,
                notify.prefix_nonce,
                retired,
            )
            return
        if current is None and retired is None and notify.prefix_nonce != 0:
            lwd_logger.warning(
                "[cloud] ignore abort for unknown generation req=%s nonce=%d",
                request_id,
                notify.prefix_nonce,
            )
            return
        self._lwd_retire_request(request_id, preserve_prompt=True)
        lwd_logger.info(
            "[cloud] abort req=%s nonce=%d", request_id, notify.prefix_nonce
        )
        self.aborts_queue.put_nowait([request_id])
        self.input_queue.put_nowait((EngineCoreRequestType.ABORT, [request_id]))

    def _lwd_retire_request(
        self, request_id: str, preserve_prompt: bool = False
    ) -> None:
        self.scheduler.lwd_recv_rows.pop(request_id, None)
        retired_nonces = getattr(self, "_lwd_retired_nonces", None)
        if retired_nonces is not None:
            retired_nonces[request_id] = getattr(self, "_lwd_request_nonces", {}).get(
                request_id, 0
            )
        if not self._lwd_request_seqnos.get(request_id):
            self._lwd_clear_retired(request_id)
            return
        self._lwd_retired_req_ids.add(request_id)
        if preserve_prompt:
            prompt_len = self._lwd_prompt_lens.get(request_id)
            if prompt_len is not None:
                self._lwd_retired_prompt_lens[request_id] = prompt_len

    def _lwd_clear_retired(self, request_id: str) -> None:
        self._lwd_retired_req_ids.discard(request_id)
        self._lwd_retired_prompt_lens.pop(request_id, None)
        getattr(self, "_lwd_retired_nonces", {}).pop(request_id, None)
        getattr(self, "_lwd_request_nonces", {}).pop(request_id, None)
        self._lwd_request_seqnos.pop(request_id, None)
        self._lwd_prompt_lens.pop(request_id, None)
        getattr(self, "_lwd_rejected_nonces", {}).pop(request_id, None)

    def shutdown(self) -> None:
        # Before the executor goes: a pre-recv hint blocks on its queue.
        if self._lwd_ingest_thread is not None:
            self._lwd_ingest_queue.put(None)
            self._lwd_ingest_thread.join(_LWD_INGEST_JOIN_TIMEOUT_S)
        super().shutdown()

    def _lwd_build_request(self, req_notify: LwdRequestNotify) -> Request:
        """Build a placeholder request and adopt the edge's native wire hashes
        for single-instance prefix negotiation."""
        sampling_params = self._lwd_build_sampling_params(req_notify)
        prompt_embeds = torch.empty(
            req_notify.num_prompt_tokens,
            self.vllm_config.model_config.get_hidden_size(),
            dtype=self.vllm_config.model_config.dtype,
            device="meta",
        )
        local_hasher = getattr(self, "request_block_hasher", None)
        hash_block_size = getattr(self, "_lwd_hash_block_size", 0)
        expected = (
            req_notify.num_prompt_tokens // hash_block_size
            if local_hasher is not None and hash_block_size > 0
            else 0
        )

        def block_hasher(request: Request) -> list[bytes]:
            if request.num_output_tokens > 0:
                return []
            if (
                local_hasher is not None
                and not request.block_hashes
                and len(req_notify.block_hashes) == expected
            ):
                return list(req_notify.block_hashes)
            assert local_hasher is not None
            return local_hasher(request)

        request = Request(
            request_id=req_notify.request_id,
            prompt_token_ids=None,
            prompt_embeds=prompt_embeds,
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=block_hasher if local_hasher is not None else None,
        )
        request.lwd_mrope_positions = req_notify.mrope_positions
        request.lwd_prefix_nonce = req_notify.prefix_nonce
        request.lwd_wire_hashes_adopted = (
            local_hasher is not None and len(req_notify.block_hashes) == expected
        )
        request.block_hashes = list(req_notify.block_hashes)
        return request

    @staticmethod
    def _lwd_build_sampling_params(req_notify: LwdRequestNotify) -> SamplingParams:
        """Rebuild sampling state carried by the request notify."""
        values = {f: getattr(req_notify, f) for f in LWD_SAMPLING_DEFAULTS}
        values["stop_token_ids"] = list(values["stop_token_ids"])
        sampling_params = SamplingParams(
            **values,
            skip_reading_prefix_cache=req_notify.skip_reading_prefix_cache,
        )
        sampling_params.update_from_generation_config({}, req_notify.eos_token_id)
        LwdControlDiag.cloud_request_admitted(req_notify, sampling_params)
        return sampling_params

    def _lwd_coord_wake_loop(self) -> None:
        """Move MP commands to a local queue and wake the engine thread."""
        command_queue = self.cloud_control_command_queue
        while True:
            try:
                command = command_queue.get()
            except (EOFError, OSError):
                return
            self._lwd_coord_pending.put(command)
            self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

    def _lwd_poll_coordination(self) -> None:
        coordinator = self._lwd_prefix_coordinator
        if coordinator is None:
            return
        processor = getattr(
            self, "cloud_control_processor", self._lwd_control_processor
        )
        if processor is not None:
            processor.poll(coordinator)
        coordinator.sweep_expired(time.monotonic())

    def has_work(self) -> bool:
        # Native WAKEUP reevaluates this predicate even when the scheduler is
        # idle, giving pending HTTP probe commands a non-blocking poll point.
        self._lwd_poll_coordination()
        return super().has_work()

    def step_with_batch_queue(self):
        """Drain the early-recv watermark at the head of the step (done
        MQ -> gate readiness predicate); the rest is native."""
        self._lwd_poll_coordination()
        self._lwd_drain_irecv_completions()
        engine_core_outputs, model_executed = super().step_with_batch_queue()
        self._lwd_complete_prefix_requests(engine_core_outputs)
        self._lwd_publish_finished_usage(engine_core_outputs)
        self._lwd_retire_finished_outputs(engine_core_outputs)
        return engine_core_outputs, model_executed

    def step(self):
        """Same for the synchronous step path: drain the watermark at
        the head of the step, the rest is native."""
        self._lwd_poll_coordination()
        self._lwd_drain_irecv_completions()
        engine_core_outputs, model_executed = super().step()
        self._lwd_complete_prefix_requests(engine_core_outputs)
        self._lwd_publish_finished_usage(engine_core_outputs)
        self._lwd_retire_finished_outputs(engine_core_outputs)
        return engine_core_outputs, model_executed

    def _lwd_complete_prefix_requests(self, engine_core_outputs: Any) -> None:
        coordinator = self._lwd_prefix_coordinator
        if coordinator is None or not engine_core_outputs:
            return
        for outputs in engine_core_outputs.values():
            for output in outputs.outputs:
                request = self.scheduler.requests.get(output.request_id)
                if (
                    request is not None
                    and request.num_computed_tokens >= request.num_prompt_tokens
                ):
                    coordinator.complete_request(output.request_id)

    def _lwd_publish_finished_usage(self, engine_core_outputs: Any) -> None:
        records = self.scheduler.lwd_finished_records
        event_queue = getattr(
            self, "cloud_control_event_queue", self.cloud_control_event_queue
        )
        from vllm.v1.lwd_control.lwd_comm.lwd_prefix import LwdUsage

        for request_id, (prompt_tokens, completion_tokens) in list(records.items()):
            cached_tokens = self._lwd_prefix_hits.get(request_id, 0)
            usage = LwdUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
            )
            if self._lwd_prefix_coordinator is not None:
                # Release the reservation and remember cloud-side counters. The
                # tenant-keyed generated hashes arrive in LwdFinishNotify.
                usage = self._lwd_prefix_coordinator.finish(
                    request_id, [], prompt_tokens, completion_tokens
                )
            if event_queue is not None:
                event_queue.put(
                    {
                        "type": "usage",
                        "request_id": request_id,
                        "usage": usage,
                        "engine_index": self._lwd_engine_index,
                    }
                )
            records.pop(request_id, None)
            self._lwd_prefix_hits.pop(request_id, None)

    def _lwd_retire_finished_outputs(self, engine_core_outputs: Any) -> None:
        if not engine_core_outputs:
            return
        for outputs in engine_core_outputs.values():
            for request_id in outputs.finished_requests or ():
                self._lwd_retire_request(request_id)
            for output in outputs.outputs:
                if output.finish_reason is not None:
                    self._lwd_retire_request(output.request_id)

    def _lwd_drain_irecv_completions(self) -> None:
        """Merge every connection's done MQ into independent watermarks."""
        done_mqs = getattr(self, "_lwd_done_mqs", None)
        if not done_mqs:
            return
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
            record_irecv_completions,
        )

        watermark_items: list[Any] = []
        for connection, done_mq in done_mqs.items():
            if done_mq is None:
                continue
            while True:
                try:
                    item = done_mq.dequeue(timeout=0)
                except TimeoutError:
                    break
                except Exception:
                    lwd_logger.exception(
                        "[pre-recv] done MQ drain failed for %s", connection
                    )
                    break
                if (
                    isinstance(item, tuple)
                    and item
                    and item[0] == "refill_done"
                ):
                    if len(item) == 2:
                        self.scheduler.lwd_refill_done(str(item[1]))
                    elif len(item) == 3:
                        self.scheduler.lwd_refill_done(str(item[1]), int(item[2]))
                    continue
                if (
                    isinstance(item, tuple)
                    and len(item) == 4
                    and item[1] == "error"
                ):
                    self._lwd_fail_connection(connection, item[2], item[3])
                    continue
                if (
                    isinstance(item, tuple)
                    and len(item) == 3
                    and item[0] == "error"
                ):
                    self._lwd_fail_connection(connection, item[1], item[2])
                    continue
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], tuple)
                ):
                    key, arrived = item
                    watermark_items.append(key)
                    if len(key) == 3:
                        self._lwd_note_arrived(key[0], key[2], arrived)
                else:
                    watermark_items.append(item)
        if not watermark_items:
            return
        advanced = record_irecv_completions(watermark_items)
        for item in advanced:
            if len(item) == 3:
                self._lwd_complete_seqno(item[0], item[2])
        lwd_logger.info(
            "[Lwd][pre-recv] DRAIN n=%d advanced=%s ts=%.3f",
            len(watermark_items),
            advanced,
            time.monotonic(),
        )

    def _lwd_fail_connection(
        self, connection: LwdConnectionId, seqno: int, error: Any
    ) -> None:
        """Abort only the failed connection; other edges keep making progress."""
        if connection in self._lwd_failed_connections:
            return
        self._lwd_failed_connections.add(connection)
        lwd_logger.error(
            "[pre-recv] connection %s failed at seqno=%s: %s",
            connection,
            seqno,
            error,
        )
        request_ids = [
            request_id
            for request_id, peer in self._lwd_request_routes.items()
            if self._lwd_connection_of(peer) == connection
        ]
        for request_id in request_ids:
            self._lwd_retire_request(request_id, preserve_prompt=True)
            self.aborts_queue.put_nowait([request_id])
            self.input_queue.put_nowait((EngineCoreRequestType.ABORT, [request_id]))

    def _lwd_note_arrived(
        self,
        connection_id: LwdConnectionId,
        seqno: int,
        arrived: Iterable[tuple[str, int]] | None,
    ) -> None:
        """Merge arrivals only for requests in this connection's pending seqno."""
        pending = self._lwd_pending_seqnos.get(connection_id, {})
        request_ids = pending.get(seqno)
        if not arrived or request_ids is None:
            return
        touched: list[tuple[str, int, int]] = []
        for request_id, rows in arrived:
            try:
                request_id, rows = str(request_id), int(rows)
            except (TypeError, ValueError):
                continue
            if (
                rows <= 0
                or request_id not in request_ids
                or request_id in self._lwd_retired_req_ids
            ):
                continue
            previous = self.scheduler.lwd_recv_rows.get(request_id, 0)
            if rows > previous:
                self.scheduler.lwd_recv_rows[request_id] = rows
                touched.append((request_id, previous, rows))
        if touched:
            lwd_logger.info("[cloud] arrival high watermarks updated: %s", touched)

    def post_step(self, model_executed: bool) -> None:
        """Flush pending c2e to the edge at the end of the step (the
        scheduler only queues them; both native step paths reach this hook)."""
        if self.scheduler.pending_c2e:
            for notify in self.scheduler.pending_c2e:
                if not notify.req_ids:
                    continue
                peers = {
                    self._lwd_request_routes.get(request_id)
                    for request_id in notify.req_ids
                }
                if len(peers) != 1 or None in peers:
                    raise RuntimeError(
                        "[LWD] one C2E batch must have exactly one known source edge"
                    )
                peer = next(iter(peers))
                assert peer is not None
                restored = msgspec.structs.replace(
                    notify,
                    req_ids=[unwrap_req_id(request_id) for request_id in notify.req_ids],
                )
                self._lwd_send_until_queued(peer, restored)
                # Finished routes have no later result. Other routes stay pinned
                # for every chunk and result of the request lifetime.
                for index, request_id in enumerate(notify.req_ids):
                    if (
                        notify.finish_reasons
                        and notify.finish_reasons[index] != -1
                    ):
                        self._lwd_request_routes.pop(request_id, None)
            self.scheduler.pending_c2e.clear()
        super().post_step(model_executed)


class LwdCloudEngineCore(LwdCloudEngineCoreMixin, EngineCoreProc):
    """Cloud PO engine of one DP rank (dp=1 and non-MoE multi-DP): the
    engine processes of a multi-DP non-MoE cloud are independent, so the
    native EngineCoreProc base is the whole story."""


class LwdCloudDPEngineCore(LwdCloudEngineCoreMixin, DPEngineCoreProc):
    """Cloud PO engine of a MoE multi-DP cloud: the native DP engine base
    carries the wave state machine, dummy batches, START_DP_WAVE handling
    and stats publishing, so the MoE engines stay in lockstep exactly as a
    centralized DP deployment; the LWD seams come from the mixin."""
