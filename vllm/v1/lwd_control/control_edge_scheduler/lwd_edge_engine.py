# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Edge engine core: the control plane to every cloud DP rank, the check of
each cloud's scheduling limits and the request entry (edge admission +
announcement to the routed cloud DP rank; communication stays in the engine).

Stepping is the reference-branch damped step (_lwd_step_damped): per step,
EMBED batches dispatch first under a reserved budget (4), then the queued
UNEMBED batches under the remaining quota (step depth 4 + 2 * (N - 1),
N = cloud DP ranks) popped round-robin per connection; uniproc executes
synchronously, so each batch runs and harvests inline within the step.
is_pooling_model=True (no native sample_tokens) and an executor proxy (a
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

from vllm.config.lwd import LwdConnectionId
from vllm.lwd_logger import lwd_logger
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
from vllm.v1.lwd_control.control_edge_scheduler.lwd_coord_client import (
    LwdCoordClient,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_executor_proxy import (
    LwdEdgeExecutorProxy,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_prefix_control import (
    LwdEdgePrefixControl,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_probe import (
    LwdEdgeProbeManager,
    LwdProbeCompletion,
    LwdProbeSubmission,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdEdgeScheduler,
    lwd_build_unembed_batch,
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
    lwd_encode_structured_outputs,
)
from vllm.v1.request import Request

# Damped-step budgets, ported from the performance-proven reference branch
# (its own formula: depth = 4 + 2 * (N - 1), N = cloud instance count; N=1
# keeps the historical 4). Per step: EMBEDs claim up to the base 4 FIRST,
# whatever the unembed backlog -- an unembed flood can never cut the UP
# prefill supply; UNEMBEDs spend the remaining quota round-robin per
# instance -- one instance's flood can never starve the other's token
# delivery. With inline (synchronous) execution the notify backlog then
# drains inside ONE step call with ~0 frame waits (the frames landed while
# the embeds ran), the damping the pre-fix one-batch-per-iteration
# stepping lacked.
_LWD_STEP_EMBED_BUDGET = 4
_LWD_STEP_UNEMBED_PER_INSTANCE = 2
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
        self._lwd_cloud_params: dict[LwdConnectionId, LwdCloudParamsNotify] = {}
        self._lwd_probe_manager = LwdEdgeProbeManager(
            self.vllm_config,
            lambda: self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None)),
        )
        self._lwd_probe_chains: dict[str, list[bytes]] = {}
        self.scheduler.lwd_on_finish = self._lwd_finish_request
        self.scheduler.lwd_has_pending = lambda: (
            self._lwd_probe_manager.pending
            or bool(self._lwd_prefix_control and self._lwd_prefix_control.pending)
        )
        self._lwd_cloud_params_received = threading.Event()
        plane = self._lwd_start_control_plane()
        assert isinstance(plane, LwdEdgeControlPlane)
        handshake_timeout_s = plane.spec.config.handshake_timeout_s
        deadline = time.monotonic() + handshake_timeout_s
        self._lwd_clouds: dict[LwdConnectionId, LwdPeer] = plane.wait_until_ready(
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
        self._lwd_configure_prefix_cache()
        self._lwd_coord_clients: dict[LwdConnectionId, LwdCoordClient] = {
            dp_idx: LwdCoordClient(dp_idx.dp_idx, params.dp_coordinator_addr)
            for dp_idx, params in self._lwd_cloud_params.items()
            if params.dp_coordinator_addr
        }
        for dp_idx, params in self._lwd_cloud_params.items():
            if params.prefix_hash_block_size > 0:
                self._lwd_probe_manager._hash_sizes[dp_idx.cloud_id] = (
                    params.prefix_hash_block_size
                )
        self.scheduler.lwd_send_to_cloud = self._lwd_send_to_cloud
        self.scheduler.lwd_refill_abort = self._lwd_abort_refill
        self.is_pooling_model = True
        # Reference-branch damped stepping: embed-reserved budget +
        # per-instance round-robin unembed quota + multi-batch drain per
        # step (see _lwd_step_damped). The native batch_queue stays EMPTY
        # (every batch executes and harvests inline inside one step call --
        # uniproc executes synchronously); it only remains as a has_work()
        # source for the base engine.
        self._lwd_step_depth = _LWD_STEP_EMBED_BUDGET + (
            _LWD_STEP_UNEMBED_PER_INSTANCE * max(0, len(self._lwd_clouds) - 1)
        )
        self.batch_queue = deque()
        self.step_fn = self._lwd_step_damped
        self.model_executor = LwdEdgeExecutorProxy(self.model_executor)
        lwd_logger.info(
            "[edge] engine assembled: %d cloud DP ranks, damped stepping "
            "embed_budget=%d step_depth=%d",
            len(self._lwd_clouds),
            _LWD_STEP_EMBED_BUDGET,
            self._lwd_step_depth,
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
        # Reached after a failed assembly too, before the clients exist.
        self._lwd_probe_manager.close()
        for client in getattr(self, "_lwd_coord_clients", {}).values():
            client.shutdown()
        super().shutdown()

    def _lwd_decode(self, payload: bytes) -> LwdMessage:
        return lwd_decode_cloud_to_edge(payload)

    def _lwd_connection_of(self, peer: LwdPeer) -> LwdConnectionId:
        return next(
            connection
            for connection in self.lwd_config.connections
            if (connection.connection_id.cloud_id, connection.connection_id.dp_idx)
            == (peer.instance_id, peer.dp_rank)
        ).connection_id

    def _lwd_dispatch_message(self, peer: LwdPeer, msg: LwdMessage) -> bool:
        """Cloud limits -> record and wake the startup wait; c2e -> scheduler
        notify queue + WAKEUP; anything else is unknown."""
        if isinstance(msg, LwdCloudParamsNotify):
            connection = self._lwd_connection_of(peer)
            self._lwd_cloud_params[connection] = msg
            self.scheduler.lwd_update_cloud_params(connection, msg)
            self._lwd_cloud_params_received.set()
            return True
        if isinstance(msg, (LwdPrefixReadyNotify, LwdRefillNotify)):
            self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, msg))
            return True
        if isinstance(msg, LwdC2eNotify):
            connection_id = self._lwd_connection_of(peer)
            lwd_logger.info(
                "[edge] c2e received from %s reqs=%d hidden_numel=%d",
                connection_id,
                len(msg.req_ids),
                msg.hidden_num_elements,
            )
            self.scheduler.unembed_notify_queue.append((connection_id, msg))
            self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
            return True
        return False

    def _lwd_check_cloud_params(
        self, dp_idx: LwdConnectionId, params: LwdCloudParamsNotify
    ) -> None:
        """(max_num_batched_tokens, max_num_seqs) must match every cloud DP
        rank: a mismatch makes the cloud hold prefills behind decodes forever
        without any error."""
        sched = self.vllm_config.scheduler_config
        local = (sched.max_num_batched_tokens, sched.max_num_seqs)
        remote = (params.max_num_batched_tokens, params.max_num_seqs)
        if local != remote:
            raise RuntimeError(
                f"[LWD] edge/cloud scheduling config mismatch on {dp_idx}: "
                f"(max_num_batched_tokens, max_num_seqs) edge={local} != "
                f"cloud={remote}; a mismatch makes the cloud hold prefills "
                "behind decodes forever, so start both sides with the same values"
            )
        lwd_logger.info("[cloud-params] %s limits verified: %s", dp_idx, local)

    def _lwd_send_to_cloud(
        self, dp_idx: LwdConnectionId, message: LwdEdgeToCloudMessage
    ) -> None:
        """Send to one cloud DP rank, retrying a full outbound queue; raises
        once the retries run out instead of losing the message."""
        cloud = self._lwd_clouds[dp_idx]
        for attempt in range(_LWD_SEND_RETRY_STEPS):
            if self._lwd_send(cloud, message):
                return
            time.sleep(_LWD_SEND_RETRY_INTERVAL_S * (attempt + 1))
        raise RuntimeError(
            f"[LWD] {type(message).__name__} to cloud {dp_idx} {cloud} not "
            f"sent: outbound queue still full after {_LWD_SEND_RETRY_STEPS} "
            "retries (cloud stalled or gone?)"
        )

    # ------------------------------------------------------------------ #
    # Engine interface overrides                                         #
    # ------------------------------------------------------------------ #
    def add_request(self, request: Request, _request_wave: int = 0) -> None:
        lwd_logger.info(
            "[edge] request received req=%s tokens=%d (admit + announce)",
            request.request_id,
            request.num_prompt_tokens,
        )
        if self.scheduler.lwd_edge_request_active(request.request_id):
            raise AssertionError(f"duplicate active request id: {request.request_id}")
        if request.abort_immediately:
            self._lwd_add_request_after_probe(request)
            return
        control = getattr(self, "_lwd_prefix_control", None)
        if control is not None:
            control.add_request(request)
            return
        submission = self._lwd_probe_manager.submit(request)
        if submission is not LwdProbeSubmission.UNAVAILABLE:
            return
        self._lwd_add_request_after_probe(request)

    def _lwd_admit_and_notify(
        self,
        request: Request,
        preferred: LwdConnectionId | None = None,
    ) -> None:
        dp_idx = self.scheduler.lwd_admit_request(request, preferred)
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

    def _lwd_add_request_after_probe(
        self,
        request: Request,
        completion: LwdProbeCompletion | None = None,
    ) -> None:
        preferred = self._lwd_preferred_connection(completion)
        request.lwd_prefix_probe_succeeded = False
        if completion is not None and completion.chain is not None:
            self._lwd_probe_chains[request.request_id] = completion.chain
        if completion is not None and completion.result is not None:
            coordination = self.lwd_config.coordination
            placement_only = coordination is not None and coordination.placement_only
            request.lwd_prefix_probe_succeeded = not placement_only
            hit = 0 if placement_only else completion.result.hit_tokens
            hit = min(max(hit, 0), max(request.num_prompt_tokens - 1, 0))
            alignment = completion.result.alignment_tokens
            if alignment > 0:
                hit -= hit % alignment
            request.num_computed_tokens = hit
        self._lwd_admit_and_notify(request, preferred)
        control = getattr(self, "_lwd_prefix_control", None)
        if control is None or request.request_id not in control.pending:
            self._lwd_enqueue_request(request)

    def _lwd_preferred_connection(
        self, completion: LwdProbeCompletion | None
    ) -> LwdConnectionId | None:
        if completion is None or completion.result is None:
            return None
        cloud_id = completion.cloud_id
        engine = completion.engine_index
        if engine is not None:
            candidate = LwdConnectionId(self.lwd_config.instance_id, cloud_id, engine)
            if candidate in self._lwd_clouds:
                return candidate
        return next(
            (
                connection_id
                for connection_id in sorted(self._lwd_clouds)
                if connection_id.cloud_id == cloud_id
            ),
            None,
        )

    def _lwd_drain_probe_completions(self) -> None:
        for completion in self._lwd_probe_manager.drain():
            self._lwd_add_request_after_probe(completion.request, completion)

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        self._lwd_drain_probe_completions()
        return super().step()

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        self._lwd_drain_probe_completions()
        return super().step_with_batch_queue()

    def _lwd_step_damped(self) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """Reference-branch damped edge step, adapted to inline execution.

        Three properties ported from the performance-proven old branch,
        where they kept the same synchronous single-threaded loop healthy
        under dual-instance load while the native one-batch-per-iteration
        stepping collapsed to half throughput:
        1. multi-batch drain: several batches run in ONE step call, so a
           notify backlog drains inside a single step with ~0 frame waits
           (their frames landed long before) instead of stretching over
           many engine iterations;
        2. embed-reserved budget: up to _LWD_STEP_EMBED_BUDGET EMBED
           batches dispatch FIRST, whatever the unembed backlog -- the UP
           prefill supply never stops, so the clouds keep stepping and
           DOWN frames keep flowing (the pre-fix collapse entered exactly
           where unembed waits squeezed embed dispatch out);
        3. per-instance round-robin unembed quota: the remaining step
           budget is spent round-robin across connections.
        uniproc executes synchronously, so each batch executes inline and
        is harvested immediately; outputs merge across the whole step.
        """
        self._lwd_drain_probe_completions()
        outputs_by_client: dict[int, EngineCoreOutputs] = {}
        executed = False

        def _run_batch(scheduler_output: Any) -> None:
            nonlocal executed
            # non_block=True is a CONTRACT with the executor proxy, not an
            # optimization: uniproc's sync path returns a bare
            # ModelRunnerOutput (not a Future) and the proxy's
            # NoneSafeFuture would crash calling .result() on it. With
            # non_block=True uniproc executes inline and returns a done
            # Future -- the result() below then never blocks (same for
            # multiproc, where it waits on the real future).
            future = self.model_executor.execute_model(
                scheduler_output, non_block=True
            )
            model_output = future.result()
            merged = self.scheduler.update_from_output(
                scheduler_output, model_output
            )
            for client_index, step_outputs in merged.items():
                existing = outputs_by_client.get(client_index)
                if existing is None:
                    outputs_by_client[client_index] = step_outputs
                    continue
                existing.outputs.extend(step_outputs.outputs)
                if step_outputs.finished_requests:
                    if existing.finished_requests is None:
                        existing.finished_requests = list(
                            step_outputs.finished_requests
                        )
                    else:
                        existing.finished_requests.extend(
                            step_outputs.finished_requests
                        )
            executed = True

        embeds = 0
        for _ in range(_LWD_STEP_EMBED_BUDGET):
            scheduler_output = self.scheduler.schedule_embed_only()
            if scheduler_output is None:
                break
            _run_batch(scheduler_output)
            embeds += 1
        for _ in range(max(0, self._lwd_step_depth - embeds)):
            item = self.scheduler.lwd_pop_unembed_rr()
            if item is None:
                break
            _run_batch(lwd_build_unembed_batch(item[0], item[1]))
        return (outputs_by_client or None), executed

    def abort_requests(self, request_ids: list[str]) -> None:
        """Notify each request's original cloud DP, then finish natively."""
        control = getattr(self, "_lwd_prefix_control", None)
        if control is not None:
            control.prepare_abort(request_ids)
        for request_id in request_ids:
            self._lwd_probe_manager.abort(request_id)
            self._lwd_probe_chains.pop(request_id, None)
        routed_aborts = self.scheduler.lwd_edge_abort(request_ids)
        for dp_idx, request_id in routed_aborts:
            request = self.scheduler.requests.get(request_id)
            self._lwd_send_to_cloud(
                dp_idx,
                LwdAbortNotify(request_id, getattr(request, "lwd_prefix_nonce", 0)),
            )
        super().abort_requests(request_ids)

    def _lwd_notify_request(self, dp_idx: LwdConnectionId, request: Request) -> None:
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
        self._lwd_send_to_cloud(
            dp_idx,
            LwdRequestNotify(
                request_id=request.request_id,
                num_prompt_tokens=request.num_prompt_tokens,
                block_hashes=list(
                    self._lwd_probe_chains.get(request.request_id, request.block_hashes)
                ),
                eos_token_id=sampling_params.eos_token_id,
                prefix_nonce=getattr(request, "lwd_prefix_nonce", 0),
                skip_reading_prefix_cache=(
                    request.skip_reading_prefix_cache
                    or (
                        self.lwd_config.coordination is not None
                        and self.lwd_config.coordination.placement_only
                    )
                ),
                prefix_hit_tokens=request.num_computed_tokens,
                prefix_probe_succeeded=getattr(
                    request, "lwd_prefix_probe_succeeded", False
                ),
                mrope_positions=self.scheduler.lwd_mrope_positions(request.request_id),
                structured_outputs=(
                    ""
                    if sampling_params.structured_outputs is None
                    else lwd_encode_structured_outputs(
                        sampling_params.structured_outputs
                    )
                ),
                **values,
            ),
        )

    def _lwd_finish_request(self, request: Request) -> None:
        self._lwd_probe_chains.pop(request.request_id, None)
