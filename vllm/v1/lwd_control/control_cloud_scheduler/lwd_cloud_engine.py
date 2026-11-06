# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cloud engine cores for one edge connection per DP rank.

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

import torch

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
    lwd_draft_slots,
    lwd_spec_factor,
)
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdPeer
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_SAMPLING_DEFAULTS,
    LwdAbortNotify,
    LwdCloudParamsNotify,
    LwdCloudToEdgeMessage,
    LwdEdgeToCloudMessage,
    LwdMessage,
    LwdPrefixReadyNotify,
    LwdRangeNotify,
    LwdRefillNotify,
    LwdRequestNotify,
    lwd_decode_edge_to_cloud,
)
from vllm.v1.lwd_control.lwd_control_diag import (
    LwdCloudArrivalMeter,
    LwdControlDiag,
)
from vllm.v1.lwd_control.lwd_model.lwd_model_family import (
    LwdRequestMetadataLayout,
    lwd_cloud_engram_fingerprint,
)
from vllm.v1.request import Request

# Sleep between retries of a full outbound queue: step metadata must not be
# lost, the edge posts its receive of the exact size from it.
_LWD_SEND_RETRY_SLEEP_S = 0.05
_LWD_INGEST_JOIN_TIMEOUT_S = 5.0

# Admitted requests with no UP rows for this long, while nothing runs, is
# an edge/cloud stall rather than a slow edge: a healthy edge keeps the
# data plane moving well inside this window.
LWD_ARRIVAL_STARVATION_S = 30.0


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

        self._lwd_hint_mq = create_hint_mq()
        super().__init__(*args, lwd_dp_size=lwd_dp_size, **kwargs)
        # The cloud endpoint worker created its done-report MQ during
        # init_device; now that the executor is up, fetch its handle over
        # the executor's collective RPC (the executor itself stays free of
        # LWD plumbing). Exactly one worker - the endpoint rank - must
        # report a handle; anything else is a wiring error.
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        handles = [
            handle
            for handle in self.model_executor.collective_rpc(
                "lwd_early_recv_done_handle"
            )
            if handle is not None
        ]
        if len(handles) != 1:
            raise RuntimeError(
                "[LWD] early receive expects one endpoint worker reporting "
                f"completions, got {len(handles)}"
            )
        self._lwd_done_mq = MessageQueue.create_from_handle(handles[0], 0)
        lwd_logger.info("[pre-recv] hint MQ created, done MQ attached via RPC")

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
        self._lwd_edge: LwdPeer | None = None
        self._lwd_prompt_lens: dict[str, int] = {}
        self._lwd_retired_prompt_lens: dict[str, int] = {}
        self._lwd_request_nonces: dict[str, int] = {}
        self._lwd_retired_nonces: dict[str, int] = {}
        self._lwd_retired_req_ids: set[str] = set()
        self._lwd_rejected_nonces: dict[str, int] = {}
        self._lwd_pending_seqnos: dict[int, set[str]] = {}
        self._lwd_request_seqnos: dict[str, set[int]] = {}
        self._lwd_latest_seqno = -1
        self._lwd_last_arrival_ts = time.monotonic()
        self._lwd_arrival_meter = LwdCloudArrivalMeter()
        self._lwd_starved_since: float | None = None
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
            num_draft_slots=lwd_draft_slots(self.vllm_config),
            engram_fingerprint=lwd_cloud_engram_fingerprint(
                self.vllm_config.model_config
            ),
        )
        # Layout of the whole-request metadata the edge announces; checked
        # at ingest so values that do not fit this cloud's model never reach
        # the runner as wrong rows.
        self._lwd_metadata_layout = LwdRequestMetadataLayout.from_model_config(
            self.vllm_config.model_config
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
        self._lwd_edge = peer
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
        assert self._lwd_edge is not None, "prefix reply before edge registration"
        self._lwd_send_until_queued(
            self._lwd_edge,
            LwdPrefixReadyNotify(request_id, hit_tokens, nonce, error),
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
        assert self._lwd_edge is not None, "refill before edge registration"
        self._lwd_send_until_queued(
            self._lwd_edge,
            LwdRefillNotify(request_id, start_row, end_row, prefix_nonce),
        )
        lwd_logger.info(
            "[prefix] REFILL req=%s rows=[%d, %d) nonce=%d",
            request_id,
            start_row,
            end_row,
            prefix_nonce,
        )

    def add_request(self, request: Request, request_wave: int = 0) -> None:
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
        hint_mq = self._lwd_hint_mq
        if hint_mq is not None:
            from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
                register_hint_sender,
            )

            # Blocking enqueue: once the gate is on, hints are a
            # correctness dependency — drop one and nobody pre-posts
            # that recv, the watermark stops advancing, and batches
            # behind the gate are never dispatched.
            register_hint_sender(lambda hint, mq=hint_mq: mq.enqueue(hint))
            lwd_logger.info("[pre-recv] hint sender registered (PD_mix -> hint MQ)")
        else:
            # PD_mix is the only scheduling mode: a missing hint MQ =
            # hints never go out and the gate never opens; this must be
            # explicitly visible
            lwd_logger.warning(
                "[pre-recv] hint MQ missing: pre-recv gate will never open"
            )

    def _lwd_dispatch_message(self, peer: LwdPeer, msg: LwdMessage) -> bool:
        """IO thread: hand all edge business notifies to the ingest thread."""
        if not isinstance(msg, (LwdRangeNotify, LwdRequestNotify, LwdAbortNotify)):
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
        # TODO(lwd): several edges per cloud DP rank need requests tracked by
        # the peer that owns them, issue #8.
        if isinstance(msg, LwdRangeNotify):
            self._lwd_handle_range_notify(msg)
        elif isinstance(msg, LwdRequestNotify):
            self._lwd_handle_request_notify(msg)
        elif isinstance(msg, LwdAbortNotify):
            self._lwd_handle_abort_notify(msg)

    def _lwd_handle_range_notify(self, notify: LwdRangeNotify) -> None:
        """Post each new extended hint exactly once, including mixed batches."""
        rejected = getattr(self, "_lwd_rejected_nonces", {})
        if any(
            rejected.get(item.request_id) == item.prefix_nonce for item in notify.items
        ):
            lwd_logger.info(
                "[cloud] ignore RangeNotify for rejected generation seqno=%d",
                notify.seqno,
            )
            return
        current_nonces = getattr(self, "_lwd_request_nonces", {})
        if any(
            item.request_id in current_nonces
            and current_nonces[item.request_id] != item.prefix_nonce
            for item in notify.items
        ):
            lwd_logger.warning(
                "[cloud] ignore stale RangeNotify seqno=%d", notify.seqno
            )
            return
        if notify.seqno <= self._lwd_latest_seqno:
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
        self._lwd_track_range(notify.seqno, request_ids)
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
            make_recv_hint,
            post_irecv_hint,
        )

        num_tokens = sum(item.num_tokens for item in items)
        self._lwd_arrival_meter.note_hint(notify.seqno, num_tokens)
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
        post_irecv_hint(make_recv_hint(notify.seqno, num_tokens, 0, **hint_fields))

    def _lwd_track_range(self, seqno: int, request_ids: list[str]) -> None:
        self._lwd_latest_seqno = seqno
        self._lwd_pending_seqnos[seqno] = set(request_ids)
        for request_id in request_ids:
            self._lwd_request_seqnos.setdefault(request_id, set()).add(seqno)

    def _lwd_complete_seqno(self, seqno: int) -> None:
        # A seqno that completes without ever merging an arrival (every
        # row aborted or retired) would otherwise keep its start time.
        self._lwd_arrival_meter.forget(seqno)
        request_ids = self._lwd_pending_seqnos.pop(seqno, None)
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
            self._lwd_send_prefix_ready(
                request_id,
                0,
                notify.prefix_nonce,
                error="request id reused before previous generation drained",
            )
            lwd_logger.warning(
                "[cloud] reject early reuse req=%s nonce=%d",
                request_id,
                notify.prefix_nonce,
            )
            return
        # Before the request is built: metadata that does not fit this
        # cloud's model would otherwise reach the runner as wrong rows.
        self._lwd_metadata_layout.validate(notify)
        if notify.image_placeholder_spans:
            # Paired with the edge's "[engram] edge" line: the same values on
            # the far side of the hop, so one diff tells a transport fault
            # from a hashing one.
            lwd_logger.info(
                "[engram] cloud metadata req=%s prompt=%d spans=%s",
                request_id,
                notify.num_prompt_tokens,
                notify.image_placeholder_spans,
            )
        self._lwd_prompt_lens[request_id] = notify.num_prompt_tokens
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
        """Build a placeholder request whose initial hashes come from the edge."""
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
        request.lwd_image_placeholder_spans = req_notify.image_placeholder_spans
        request.lwd_prefix_nonce = req_notify.prefix_nonce
        request.lwd_wire_hashes_adopted = (
            local_hasher is not None and len(req_notify.block_hashes) == expected
        )
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

    def step_with_batch_queue(self):
        """Drain the early-recv watermark at the head of the step (done
        MQ -> gate readiness predicate); the rest is native."""
        self._lwd_drain_irecv_completions()
        self._lwd_check_starvation()
        engine_core_outputs, model_executed = super().step_with_batch_queue()
        self._lwd_retire_finished_outputs(engine_core_outputs)
        return engine_core_outputs, model_executed

    def step(self):
        """Same for the synchronous step path: drain the watermark at
        the head of the step, the rest is native."""
        self._lwd_drain_irecv_completions()
        self._lwd_check_starvation()
        engine_core_outputs, model_executed = super().step()
        self._lwd_retire_finished_outputs(engine_core_outputs)
        return engine_core_outputs, model_executed

    def _lwd_retire_finished_outputs(self, engine_core_outputs: Any) -> None:
        if not engine_core_outputs:
            return
        for outputs in engine_core_outputs.values():
            for request_id in outputs.finished_requests or ():
                self._lwd_retire_request(request_id)
            for output in outputs.outputs:
                if output.finish_reason is not None:
                    self._lwd_retire_request(output.request_id)

    def _lwd_check_starvation(self) -> None:
        """Report a cloud that holds admitted requests but receives no UP
        rows for them.

        The edge and the cloud can disagree silently: the control plane
        keeps admitting requests while the data plane stopped delivering,
        and both sides then look healthy (no error, health 200) while
        nothing runs. Log the transition so the stall is visible from the
        cloud alone, with the edge's view of it unavailable here."""
        waiting = len(self.scheduler.waiting)
        if not waiting or self.scheduler.running:
            self._lwd_starved_since = None
            return
        idle_s = time.monotonic() - self._lwd_last_arrival_ts
        if idle_s < LWD_ARRIVAL_STARVATION_S:
            return
        if self._lwd_starved_since is not None:
            return
        self._lwd_starved_since = time.monotonic()
        lwd_logger.warning(
            "[cloud] UP starvation: waiting=%d running=0 for %.0fs, "
            "pending_seqnos=%s latest_seqno=%d; the edge admitted these "
            "requests but sent no rows for them",
            waiting,
            idle_s,
            sorted(self._lwd_pending_seqnos),
            self._lwd_latest_seqno,
        )

    def _lwd_drain_irecv_completions(self) -> None:
        """Merge done-MQ arrivals into per-request rows and UP watermarks."""
        done_mq = getattr(self, "_lwd_done_mq", None)
        if done_mq is None:
            return
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
            record_irecv_completions,
        )

        items: list[Any] = []
        while True:
            try:
                items.append(done_mq.dequeue(timeout=0))
            except TimeoutError:
                break
            except Exception:
                lwd_logger.exception("[pre-recv] done MQ drain failed")
                break
        watermark_items: list[Any] = []
        for item in items:
            if isinstance(item, tuple) and len(item) == 3 and item[0] == "error":
                self._lwd_fail(f"[LWD] worker receive error seqno={item[1]}: {item[2]}")
                return
            if (
                isinstance(item, tuple)
                and len(item) in (2, 3)
                and item[0] == "refill_done"
            ):
                if len(item) == 2:
                    self.scheduler.lwd_refill_done(str(item[1]))
                else:
                    self.scheduler.lwd_refill_done(str(item[1]), int(item[2]))
                continue
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], tuple)
            ):
                key, arrived = item
                watermark_items.append(key)
                self._lwd_note_arrived(key[1], arrived)
            else:
                watermark_items.append(item)
        if not watermark_items:
            return
        advanced = record_irecv_completions(watermark_items)
        if advanced:
            self._lwd_last_arrival_ts = time.monotonic()
        for _channel, seqno in advanced:
            self._lwd_complete_seqno(seqno)
        lwd_logger.info(
            "[Lwd][pre-recv] DRAIN n=%d seqnos=%s advanced=%s ts=%.3f",
            len(watermark_items),
            sorted({seqno for _channel, seqno in watermark_items}),
            sorted({seqno for _channel, seqno in advanced}),
            time.monotonic(),
        )

    def _lwd_note_arrived(
        self, seqno: int, arrived: Iterable[tuple[str, int]] | None
    ) -> None:
        """Merge arrivals only for requests in this still-pending seqno."""
        request_ids = self._lwd_pending_seqnos.get(seqno)
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
            # Rows a window can open over exist only from here: time the
            # hop against the hint that started it.
            self._lwd_arrival_meter.note_arrival(seqno)
            lwd_logger.info("[cloud] arrival high watermarks updated: %s", touched)

    def post_step(self, model_executed: bool) -> None:
        """Flush pending c2e to the edge at the end of the step (the
        scheduler only queues them; both native step paths reach this hook)."""
        if self.scheduler.pending_c2e:
            assert self._lwd_edge is not None, "c2e produced before any edge"
            for notify in self.scheduler.pending_c2e:
                self._lwd_send_until_queued(self._lwd_edge, notify)
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
