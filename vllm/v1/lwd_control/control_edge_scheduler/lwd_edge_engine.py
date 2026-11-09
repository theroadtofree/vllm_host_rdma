# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Edge engine core: the control plane to every cloud DP rank, the check of
each cloud's scheduling limits and the request entry (edge admission +
announcement to the routed cloud DP rank; communication stays in the engine).

Stepping is the native step_with_batch_queue, adapted at construction: depth
2, is_pooling_model=True (no native sample_tokens) and an executor proxy (a
normally completed batch returning None becomes a placeholder).

Startup blocks until every cloud DP rank accepted the handshake and sent its
scheduling limits, within one handshake timeout; any failure stops the
engine. There is no self-healing: restart the whole group.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from vllm.lwd_logger import lwd_logger
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.lwd_control.control_edge_scheduler.lwd_coord_client import (
    LwdCoordClient,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_executor_proxy import (
    LwdEdgeExecutorProxy,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_prefix_control import (
    LwdEdgePrefixControl,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdEdgeScheduler,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_engine import (
    LwdBaseEngineCore,
)
from vllm.v1.lwd_control.lwd_comm.lwd_control_plane import LwdEdgeControlPlane
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdPeer
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_SAMPLING_DEFAULTS,
    LwdAbortNotify,
    LwdC2eNotify,
    LwdCloudParamsNotify,
    LwdEdgeToCloudMessage,
    LwdMessage,
    LwdPrefixReadyNotify,
    LwdRefillNotify,
    LwdRequestNotify,
    lwd_decode_cloud_to_edge,
)
from vllm.v1.request import Request

# Native batch queue depth: dispatched (embed + unembed) batches not yet
# reaped. 2 keeps the pipeline aligned; the reference branch's 4 is
# conservative, validate performance with 2 first.
LWD_EDGE_BATCH_QUEUE_DEPTH = 2
LWD_PREFIX_READY_TIMEOUT_S = 60.0

# A full outbound queue is retried with growing intervals (about 3 s in
# total) before failing: by then the cloud stopped reading or is gone.
_LWD_SEND_RETRY_STEPS = 5
_LWD_SEND_RETRY_INTERVAL_S = 0.2


class LwdEdgeEngineCore(LwdBaseEngineCore):
    """Edge PO engine: control plane to every cloud DP rank, cloud limits
    check and native stepping."""

    lwd_scheduler_cls = LwdEdgeScheduler
    scheduler: LwdEdgeScheduler

    def _lwd_assemble(self) -> None:
        self._lwd_prefix_enabled = bool(
            self.vllm_config.cache_config.enable_prefix_caching
        )
        self._lwd_prefix_control = (
            LwdEdgePrefixControl(self, LWD_PREFIX_READY_TIMEOUT_S)
            if self._lwd_prefix_enabled
            else None
        )
        self._lwd_cloud_params: dict[int, LwdCloudParamsNotify] = {}
        self._lwd_cloud_params_received = threading.Event()
        plane = self._lwd_start_control_plane()
        assert isinstance(plane, LwdEdgeControlPlane)
        handshake_timeout_s = plane.spec.config.handshake_timeout_s
        deadline = time.monotonic() + handshake_timeout_s
        self._lwd_clouds: dict[int, LwdPeer] = plane.wait_until_ready(
            handshake_timeout_s
        )
        expected = len(self._lwd_clouds)
        while len(self._lwd_cloud_params) < expected:
            if not self._lwd_cloud_params_received.wait(
                max(0.0, deadline - time.monotonic())
            ):
                missing = expected - len(self._lwd_cloud_params)
                raise RuntimeError(
                    f"[LWD] edge engine init failed: {missing} of "
                    f"{expected} cloud DP ranks accepted the handshake "
                    f"but sent no scheduling limits within "
                    f"{handshake_timeout_s}s"
                )
            self._lwd_cloud_params_received.clear()
        for dp_idx, params in self._lwd_cloud_params.items():
            self._lwd_check_cloud_params(dp_idx, params)
        self._lwd_adopt_cloud_draft_slots()
        self._lwd_configure_prefix_cache()
        self._lwd_coord_clients: dict[int, LwdCoordClient] = {
            dp_idx: LwdCoordClient(dp_idx, params.dp_coordinator_addr)
            for dp_idx, params in self._lwd_cloud_params.items()
            if params.dp_coordinator_addr
        }
        self.scheduler.lwd_send_to_cloud = self._lwd_send_to_cloud
        self.scheduler.lwd_refill_abort = self._lwd_abort_refill
        self.is_pooling_model = True
        self.batch_queue_size = LWD_EDGE_BATCH_QUEUE_DEPTH
        self.batch_queue = deque(maxlen=self.batch_queue_size)
        self.step_fn = self.step_with_batch_queue
        self.model_executor = LwdEdgeExecutorProxy(self.model_executor)
        lwd_logger.info(
            "[edge] engine assembled: %d cloud DP ranks, native batch-queue "
            "stepping depth=%d",
            len(self._lwd_clouds),
            self.batch_queue_size,
        )

    def _lwd_configure_prefix_cache(self) -> None:
        capabilities: set[tuple[int, int, bool, bool]] = set()
        for dp_idx, params in self._lwd_cloud_params.items():
            if self._lwd_prefix_enabled != params.prefix_caching:
                raise RuntimeError(
                    f"[LWD] prefix caching mismatch on dp{dp_idx}: "
                    f"edge={self._lwd_prefix_enabled} "
                    f"cloud={params.prefix_caching}"
                )
            capabilities.add(
                (
                    params.hash_block_size,
                    params.prefix_alignment_tokens,
                    params.prefix_alignment_required,
                    params.prefix_use_eagle,
                )
            )
        control = getattr(self, "_lwd_prefix_control", None)
        if control is None:
            return
        if len(capabilities) != 1:
            raise RuntimeError(
                f"[LWD] inconsistent prefix capabilities across cloud DP ranks: "
                f"{sorted(capabilities)}"
            )
        params = next(iter(self._lwd_cloud_params.values()))
        control.configure(params)
        self.scheduler._lwd_prefix_hash_block_size = control.hash_block_size

    def shutdown(self) -> None:
        control = getattr(self, "_lwd_prefix_control", None)
        if control is not None:
            control.shutdown()
        for client in getattr(self, "_lwd_coord_clients", {}).values():
            client.shutdown()
        super().shutdown()

    def _lwd_decode(self, payload: bytes) -> LwdMessage:
        return lwd_decode_cloud_to_edge(payload)

    def _lwd_dispatch_message(self, peer: LwdPeer, msg: LwdMessage) -> bool:
        """Cloud limits -> record and wake the startup wait; c2e -> scheduler
        notify queue + WAKEUP; anything else is unknown."""
        if isinstance(msg, LwdCloudParamsNotify):
            self._lwd_cloud_params[peer.dp_rank] = msg
            self._lwd_cloud_params_received.set()
            return True
        if isinstance(msg, (LwdPrefixReadyNotify, LwdRefillNotify)):
            self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, msg))
            return True
        if isinstance(msg, LwdC2eNotify):
            # One per decode step per engine on the control-plane IO thread,
            # which shares the GIL with the only scheduling thread: debug,
            # not info.
            lwd_logger.debug(
                "[edge] c2e received from dp%d reqs=%d hidden_numel=%d",
                peer.dp_rank,
                len(msg.req_ids),
                msg.hidden_num_elements,
            )
            self.scheduler.unembed_notify_queue.append((peer.dp_rank, msg))
            self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
            return True
        return False

    def _lwd_check_cloud_params(
        self, dp_idx: int, params: LwdCloudParamsNotify
    ) -> None:
        """(max_num_batched_tokens, max_num_seqs) must match every cloud DP
        rank: a mismatch makes the cloud hold prefills behind decodes forever
        without any error. The Engram hash parameters must match too."""
        sched = self.vllm_config.scheduler_config
        local = (sched.max_num_batched_tokens, sched.max_num_seqs)
        remote = (params.max_num_batched_tokens, params.max_num_seqs)
        if local != remote:
            raise RuntimeError(
                f"[LWD] edge/cloud scheduling config mismatch on dp{dp_idx}: "
                f"(max_num_batched_tokens, max_num_seqs) edge={local} != "
                f"cloud={remote}; a mismatch makes the cloud hold prefills "
                "behind decodes forever, so start both sides with the same values"
            )
        self._lwd_check_engram_fingerprint(dp_idx, params.engram_fingerprint)
        lwd_logger.info("[cloud-params] dp%d limits verified: %s", dp_idx, local)

    def _lwd_check_engram_fingerprint(self, dp_idx: int, remote: str) -> None:
        """The edge's Engram bucket indices must address the cloud's tables:
        both sides derive the hash parameters from their own tokenizer and
        config, and a mismatch would hash every prompt into wrong rows
        without any error."""
        local = self.scheduler.lwd_model_family.engram_fingerprint()
        if local != remote:
            raise RuntimeError(
                f"[LWD] Engram fingerprint mismatch on dp{dp_idx}: "
                f"edge={local!r} != cloud={remote!r}; the edge's bucket "
                "indices would address wrong table rows, so serve both sides "
                "the same model and tokenizer"
            )

    def _lwd_adopt_cloud_draft_slots(self) -> None:
        """Adopt the cloud's per-request drafting slots: they reserve input
        budget of cloud steps, and the edge has no legitimate local source
        (a spec config on the edge would start the draft machinery). The
        edge picker budgets them so both sides cut the same chunks."""
        remote = {params.num_draft_slots for params in self._lwd_cloud_params.values()}
        if len(remote) != 1:
            raise RuntimeError(
                f"[LWD] inconsistent drafting slots across cloud DP ranks: "
                f"{sorted(remote)}"
            )
        draft_slots = remote.pop()
        if draft_slots == self.scheduler._lwd_draft_slots:
            return
        lwd_logger.warning(
            "[cloud-params] adopt cloud draft_slots=%d over edge local %d",
            draft_slots,
            self.scheduler._lwd_draft_slots,
        )
        self.scheduler._lwd_draft_slots = draft_slots

    def _lwd_send_to_cloud(self, dp_idx: int, message: LwdEdgeToCloudMessage) -> None:
        """Send to one cloud DP rank, retrying a full outbound queue; raises
        once the retries run out instead of losing the message."""
        cloud = self._lwd_clouds[dp_idx]
        for attempt in range(_LWD_SEND_RETRY_STEPS):
            if self._lwd_send(cloud, message):
                return
            time.sleep(_LWD_SEND_RETRY_INTERVAL_S * (attempt + 1))
        raise RuntimeError(
            f"[LWD] {type(message).__name__} to cloud dp{dp_idx} {cloud} not "
            f"sent: outbound queue still full after {_LWD_SEND_RETRY_STEPS} "
            "retries (cloud stalled or gone?)"
        )

    # ------------------------------------------------------------------ #
    # Engine interface overrides                                         #
    # ------------------------------------------------------------------ #
    def add_request(self, request: Request, _request_wave: int = 0) -> None:
        # Start of the edge's share of TTFT, and the close of the frontend
        # segment: everything before this ran off the engine thread.
        self.scheduler.lwd_ttft_meter.note_add_request(
            request.request_id, request.arrival_time
        )
        lwd_logger.info(
            "[edge] request received req=%s tokens=%d (admit + announce)",
            request.request_id,
            request.num_prompt_tokens,
        )
        if self.scheduler.lwd_edge_request_active(request.request_id):
            lwd_logger.error(
                "[edge] reject duplicate active request req=%s", request.request_id
            )
            raise AssertionError(f"duplicate active request id: {request.request_id}")
        control = getattr(self, "_lwd_prefix_control", None)
        if control is not None and not request.abort_immediately:
            control.add_request(request)
            return
        self._lwd_admit_and_notify(request)
        self._lwd_enqueue_request(request)

    def _lwd_admit_and_notify(self, request: Request) -> None:
        dp_idx = self.scheduler.lwd_admit_request(request)
        client = self._lwd_coord_clients.get(dp_idx)
        try:
            if client is not None:
                client.send_first_req()
            self._lwd_notify_request(dp_idx, request)
        except Exception:
            self.scheduler.lwd_edge_cancel_admission(request.request_id)
            raise

    def _lwd_enqueue_request(self, request: Request) -> None:
        super().add_request(request)

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        control = getattr(self, "_lwd_prefix_control", None)
        if control is not None and control.handle_client_request(request_type, request):
            return
        if request_type == EngineCoreRequestType.WAKEUP and isinstance(
            request, LwdRefillNotify
        ):
            self._lwd_handle_refill(request)
            return
        super()._handle_client_request(request_type, request)

    def _lwd_handle_refill(self, notify: LwdRefillNotify) -> None:
        outputs = self.scheduler.lwd_edge_build_refill(notify)
        if outputs is None:
            lwd_logger.error("[edge-refill] req=%s cannot be served", notify.request_id)
            self._lwd_abort_refill(notify.request_id)
            return
        self.scheduler.lwd_edge_queue_refill(outputs)

    def _lwd_abort_refill(self, request_id: str) -> None:
        request = self.scheduler.requests.get(request_id)
        if request is None:
            lwd_logger.warning("[edge-refill] stale abort req=%s ignored", request_id)
            return
        self._lwd_abort_internally(request)

    def _lwd_abort_internally(self, request: Request) -> None:
        self.abort_requests([request.request_id])
        self._send_abort_outputs_to_client([request.request_id], request.client_index)

    def abort_requests(self, request_ids: list[str]) -> None:
        control = getattr(self, "_lwd_prefix_control", None)
        if control is not None:
            control.prepare_abort(request_ids)
        routed_aborts = self.scheduler.lwd_edge_abort(request_ids)
        for dp_idx, request_id in routed_aborts:
            request = getattr(self.scheduler, "requests", {}).get(request_id)
            self._lwd_send_to_cloud(
                dp_idx,
                LwdAbortNotify(request_id, getattr(request, "lwd_prefix_nonce", 0)),
            )
        super().abort_requests(request_ids)

    def _lwd_notify_request(self, dp_idx: int, request: Request) -> None:
        """Announce the request to one cloud DP rank, packing the sampling
        fields from the shared table."""
        sampling_params = request.sampling_params
        assert sampling_params is not None, "admission rejects pooling requests"
        values: dict[str, Any] = {
            f: getattr(sampling_params, f) for f in LWD_SAMPLING_DEFAULTS
        }
        if values["max_tokens"] is None:
            values["max_tokens"] = LWD_SAMPLING_DEFAULTS["max_tokens"]
        values["stop_token_ids"] = list(values["stop_token_ids"] or [])
        metadata = self.scheduler.lwd_model_family.request_metadata(request)
        self._lwd_send_to_cloud(
            dp_idx,
            LwdRequestNotify(
                request_id=request.request_id,
                num_prompt_tokens=request.num_prompt_tokens,
                block_hashes=list(request.block_hashes),
                prefix_nonce=getattr(request, "lwd_prefix_nonce", 0),
                skip_reading_prefix_cache=bool(
                    getattr(sampling_params, "skip_reading_prefix_cache", False)
                ),
                eos_token_id=sampling_params.eos_token_id,
                mrope_positions=self.scheduler.lwd_mrope_positions(request.request_id),
                image_placeholder_spans=metadata.image_placeholder_spans,
                **values,
            ),
        )
        # Stamped after the send: a full outbound queue retries for about
        # three seconds, and that is admission time, not dispatch time.
        self.scheduler.lwd_ttft_meter.note_notify(request.request_id)
