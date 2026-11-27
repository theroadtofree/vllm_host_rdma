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
import uuid
from collections.abc import Iterable
from typing import Any

import msgspec
import torch

from vllm.config.lwd import LwdConnectionId, LwdTopologyIncrement
from vllm.lwd_logger import lwd_logger
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc, EngineShutdownState
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_ledger import (
    LwdCloudLedger,
)
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_scheduler import (
    LwdCloudScheduler,
)
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_prefix_cache import (
    LwdProbeQueryAdapter,
    init_lwd_prefix_cache,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_engine import (
    LwdBaseEngineCore,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    lwd_spec_factor,
)
from vllm.v1.lwd_control.lwd_comm.lwd_control_plane import (
    LwdCloudControlPlane,
    LwdPeerNotRegisteredError,
)
from vllm.v1.lwd_control.lwd_comm.lwd_id_adapter import unwrap_req_id, wrap_req_id
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdPeer
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_SAMPLING_DEFAULTS,
    LwdAbortNotify,
    LwdC2eNotify,
    LwdCloudHeartbeat,
    LwdCloudParamsNotify,
    LwdCloudToEdgeMessage,
    LwdEdgeToCloudMessage,
    LwdMessage,
    LwdPrefixReadyNotify,
    LwdRangeNotify,
    LwdRefillNotify,
    LwdRequestNotify,
    lwd_decode_edge_to_cloud,
    lwd_decode_reasoning_parser_kwargs,
    lwd_decode_structured_outputs,
)
from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
from vllm.v1.request import Request

# Sleep between retries of a full outbound queue: step metadata must not be
# lost, the edge posts its receive of the exact size from it.
_LWD_SEND_RETRY_SLEEP_S = 0.05
# Debug-level cloud concurrency snapshot interval.
_LWD_CLOUD_CONC_LOG_INTERVAL_S = 10.0
# Seconds between one cloud DP's liveness heartbeats to its edges. This
# cadence and the edge's silent threshold (LWD_HELLO_SILENT_S in
# lwd_edge_engine) are the two halves of ONE liveness contract — the
# sender's pace and the judge's threshold — so any change must consider
# both files together.
LWD_HELLO_INTERVAL_S = 1.0


class _LwdHeartbeatTick:
    """Input-queue sentinel of the heartbeat thread: the engine thread (the
    control sender's owner) sends one liveness beat on receipt."""


_LWD_HEARTBEAT_TICK = _LwdHeartbeatTick()


def _lwd_log_sent_to_edge(peer: LwdPeer, message: LwdCloudToEdgeMessage) -> None:
    """Full-payload log of one cloud->edge control-plane message once queued
    for the peer; pairs with the edge-side `[edge] c2e received` log. The
    per-step c2e notify gets per-row detail (top tree / token payloads are
    reduced to lengths to bound log volume)."""
    if isinstance(message, LwdC2eNotify):
        lwd_logger.debug(
            "[LWD][cloud] c2e sent to edge %s seqno=%d reqs=%d hidden_numel=%d "
            "req_ids=%s accepted=%s finish=%s nonces=%s top_tree_lens=%s "
            "token_lens=%s",
            peer,
            message.down_seqno,
            len(message.req_ids),
            message.hidden_num_elements,
            message.req_ids,
            message.num_accepted_tokens,
            message.finish_reasons,
            message.prefix_nonces,
            [len(top) for top in message.top_id_ths],
            [len(tokens) for tokens in message.token_ids],
        )
    else:
        lwd_logger.debug(
            "[LWD][cloud] %s sent to edge %s: %s",
            type(message).__name__,
            peer,
            message,
        )


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
            if connection.connection_id.dp_idx == dp_idx
        )
        if not connections:
            # A topology connection set is the single source of edge routing:
            # fabricating a default edge here would misroute every hint of
            # this DP rank, so an empty set is a wiring error, not a default.
            raise ValueError(
                f"[LWD] cloud DP rank {dp_idx} of instance "
                f"{vllm_config.lwd_config.instance_id} has no edge connection"
            )
        self._lwd_hint_mqs = {
            connection: create_hint_mq(connection) for connection in connections
        }
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
        lwd_logger.info(
            "[LWD][pre-recv] %d hint MQ(s) created, done MQs attached via RPC",
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
        if self.vllm_config.model_config.skip_tokenizer_init:
            # StructuredOutputManager builds its tokenizer and compile
            # executor only with a tokenizer; without one the first
            # constrained request would crash at grammar_init. Fail fast at
            # startup instead (the cloud slice ships the tokenizer per F2
            # metadata rules).
            raise ValueError(
                "[LWD] the cloud engine requires its tokenizer "
                "(skip_tokenizer_init=True is not supported): grammar "
                "compilation and the sampling bitmask happen on the cloud"
            )
        self._lwd_init_prefix_cache()
        self._lwd_edges: set[LwdPeer] = set()
        self._lwd_ledger = LwdCloudLedger(self.scheduler)
        self.scheduler.lwd_hash_block_size = self._lwd_hash_block_size
        prefix_hash_block_size = self._lwd_hash_block_size
        coordination = self.lwd_config.coordination
        self._lwd_probe_query = None
        # The probe query serves probes whenever coordination is enabled,
        # NOT only with prefix caching: probe replies also carry placement
        # and routing data (engine_index / cloud_id) plus health
        # bookkeeping, which must keep flowing with prefix caching off.
        # With caching disabled the NoPrefixCache KV coordinator returns an
        # empty hit, so probes answer with zero hits instead of hanging in
        # a consumer-less HTTP command queue until the 20s timeout.
        if coordination is not None and coordination.enabled:
            self._lwd_probe_query = LwdProbeQueryAdapter(
                manager=self.scheduler.kv_cache_manager,
                hash_block_size=self._lwd_hash_block_size,
                alignment_tokens=self._lwd_prefix_state.alignment_tokens,
                instance_id=f"cloud-{self.lwd_config.instance_id}",
                cloud_id=self.lwd_config.instance_id,
                # With caching off the query only carries placement/health;
                # the manifest chain is built at the edge's own granularity,
                # so the cloud hash-block-size contract does not apply.
                prefix_caching=self._lwd_prefix_enabled,
            )
        self.scheduler.lwd_prefix_hits = self._lwd_ledger.prefix_hits
        self._lwd_coord_pending: queue.Queue[dict[str, Any]] = queue.Queue()
        self._lwd_control_processor = None
        self.cloud_control_event_queue = getattr(
            self, "cloud_control_event_queue", None
        )
        command_queue = getattr(self, "cloud_control_command_queue", None)
        if self._lwd_probe_query is not None:
            if command_queue is None or self.cloud_control_event_queue is None:
                raise RuntimeError(
                    "[LWD] prefix caching requires HTTP coordination queues"
                )
            from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_control import (
                LwdCloudControlProcessor,
            )

            self._lwd_control_processor = LwdCloudControlProcessor(
                self._lwd_coord_pending,
                self.cloud_control_event_queue,
                poll_budget=coordination.probe_poll_budget,
            )
            threading.Thread(
                target=self._lwd_coord_wake_loop,
                name="lwd-ctrl-wake",
                daemon=True,
            ).start()
        self._lwd_cloud_conc_last_log = 0.0
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
        # Business messages are handed to the engine input queue. Request,
        # reservation, routing and coordinator state therefore have one owner:
        # the engine thread, same as the native scheduler and KV manager.
        plane = self._lwd_start_control_plane()
        lwd_logger.info(
            "[LWD][cloud] engine assembled: control plane on %s", plane.endpoint
        )
        # Liveness epoch (peer-restart line): random per process launch; the
        # edge treats a change of it as "this cloud DP restarted".
        self._lwd_boot_id = uuid.uuid4().hex[:16]
        self._lwd_hello_interval_s = LWD_HELLO_INTERVAL_S
        self._lwd_last_heartbeat_at = 0.0
        self._lwd_heartbeat_stop = threading.Event()
        threading.Thread(
            target=self._lwd_heartbeat_thread,
            name="lwd-cloud-heartbeat",
            daemon=True,
        ).start()

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
        # A (re)registered edge is a NEW process incarnation: its seqno
        # streams restart from zero, so the connection's old-incarnation
        # bookkeeping must not gate the new one. _lwd_latest_seqnos kept
        # the old stream's high-water mark, dropping the new stream's
        # first RangeNotify as a "duplicate" — the request hung with its
        # UP frame already sent and nothing posted to receive it. (This
        # callback fires only for a fresh registration or a session
        # replacement, never for the same process's retried hellos, so
        # the reset cannot fire on a mere link blip.)
        connection = self._lwd_connection_of(peer)
        self._lwd_ledger.latest_seqnos.pop(connection, None)
        self._lwd_ledger.pending_seqnos.pop(connection, None)
        self._lwd_send_until_queued(peer, self._lwd_cloud_params)
        lwd_logger.info(
            "[LWD][cloud] edge %s registered; scheduling limits sent: %s",
            peer,
            self._lwd_cloud_params,
        )

    def _lwd_send_until_queued(
        self, peer: LwdPeer, message: LwdCloudToEdgeMessage
    ) -> None:
        """Retry a full outbound queue until the message is queued; gives up
        with an error once shutdown is requested, so a stalled edge cannot
        block it. An unregistered peer (its ROUTER is gone and the plane
        already drops its payloads) drops the message instead of failing
        the engine: the restarted edge's state is fresh and the stale
        message has no receiver."""
        try:
            while not self._lwd_send(peer, message):
                if self.shutdown_state != EngineShutdownState.RUNNING:
                    lwd_logger.error(
                        "[LWD][cloud] %s to edge %s dropped: shutting down "
                        "with the outbound queue full",
                        type(message).__name__,
                        peer,
                    )
                    return
                time.sleep(_LWD_SEND_RETRY_SLEEP_S)
        except LwdPeerNotRegisteredError:
            lwd_logger.error(
                "[LWD][cloud] %s to edge %s dropped: peer unreachable",
                type(message).__name__,
                peer,
            )
            return
        _lwd_log_sent_to_edge(peer, message)

    def _lwd_prefix_replay_start(self, request: Request, hit_tokens: int) -> int:
        connector = self.scheduler.connector
        if hit_tokens <= 0 or (
            connector is not None
            and self.scheduler._load_restores_replay_window(request)
        ):
            return hit_tokens
        replay_tokens = getattr(self.scheduler, "prefix_replay_tokens", 0)
        return hit_tokens - min(hit_tokens, replay_tokens)

    def _lwd_send_prefix_ready(
        self, request_id: str, hit_tokens: int, nonce: int, error: str = ""
    ) -> None:
        peer = self._lwd_ledger.request_routes.get(request_id)
        if peer is None:
            raise RuntimeError(f"[LWD] prefix reply route missing for {request_id}")
        request = self.scheduler.requests.get(request_id)
        replay_start = (
            self._lwd_prefix_replay_start(request, hit_tokens)
            if request is not None
            else hit_tokens
        )
        self._lwd_send_until_queued(
            peer,
            LwdPrefixReadyNotify(
                unwrap_req_id(request_id), hit_tokens, nonce, error, replay_start
            ),
        )
        lwd_logger.info(
            "[LWD][prefix] READY req=%s hit=%d replay_start=%d nonce=%d error=%r",
            request_id,
            hit_tokens,
            replay_start,
            nonce,
            error,
        )

    def _lwd_send_refill(
        self, request_id: str, start_row: int, end_row: int, prefix_nonce: int
    ) -> None:
        peer = self._lwd_ledger.request_routes.get(request_id)
        if peer is None:
            raise RuntimeError(f"[LWD] refill route missing for {request_id}")
        self._lwd_send_until_queued(
            peer,
            LwdRefillNotify(
                unwrap_req_id(request_id), start_row, end_row, prefix_nonce
            ),
        )
        lwd_logger.info(
            "[LWD][prefix] REFILL req=%s rows=[%d, %d) nonce=%d",
            request_id,
            start_row,
            end_row,
            prefix_nonce,
        )

    def _lwd_publish_usage_start(self, request: Request) -> None:
        event_queue = getattr(self, "cloud_control_event_queue", None)
        if event_queue is None:
            return
        event_queue.put(
            {
                "type": "usage_start",
                "request_id": request.request_id,
                "engine_index": self._lwd_engine_index,
                "prefix_nonce": getattr(request, "lwd_prefix_nonce", 0),
            }
        )

    def add_request(self, request: Request, request_wave: int = 0) -> None:
        nonce = getattr(request, "lwd_prefix_nonce", 0)
        if not self._lwd_prefix_enabled:
            super().add_request(request, request_wave)
            self._lwd_publish_usage_start(request)
            return
        if not nonce:
            lwd_logger.error(
                "[LWD][prefix] reject req=%s: missing prefix nonce",
                request.request_id,
            )
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
            lwd_logger.exception("[LWD][prefix] reserve failed req=%s", request_id)
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
        self._lwd_publish_usage_start(request)
        self._lwd_ledger.prefix_hits[request_id] = hit_tokens
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
                "[LWD][pre-recv] hint senders registered for %d connection(s)",
                len(self._lwd_hint_mqs),
            )
        else:
            # PD_mix is the only scheduling mode: a missing hint MQ =
            # hints never go out and the gate never opens; this must be
            # explicitly visible
            lwd_logger.warning(
                "[LWD][pre-recv] hint MQ missing: pre-recv gate will never open"
            )

    def _lwd_dispatch_message(self, peer: LwdPeer, msg: LwdMessage) -> bool:
        """IO thread: preserve order and hand business messages to the engine."""
        if not isinstance(
            msg,
            (
                LwdRangeNotify,
                LwdRequestNotify,
                LwdAbortNotify,
            ),
        ):
            return False
        self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, (peer, msg)))
        return True

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        if request_type == EngineCoreRequestType.WAKEUP and isinstance(
            request, _LwdHeartbeatTick
        ):
            # Engine thread (the sender's owner): one liveness beat per
            # interval. The tick comes from the heartbeat thread, which only
            # touches the input queue — the control sender is bound to this
            # thread and an idle engine must still step for the beat.
            self._lwd_send_heartbeat_if_due()
            return
        if (
            request_type == EngineCoreRequestType.WAKEUP
            and isinstance(request, tuple)
            and len(request) == 2
            and isinstance(request[0], LwdPeer)
        ):
            peer, msg = request
            try:
                self._lwd_handle_edge_message(peer, msg)
            except Exception as exc:
                lwd_logger.exception(
                    "[LWD][cloud] ingest of %s failed", type(msg).__name__
                )
                self._lwd_fail(f"ingest of {type(msg).__name__}: {exc!r}")
            return
        super()._handle_client_request(request_type, request)

    def _lwd_handle_edge_message(
        self, peer: LwdPeer, msg: LwdEdgeToCloudMessage
    ) -> None:
        lwd_logger.info(
            "[LWD][cloud] ingest msg=%s seqno=%s",
            type(msg).__name__,
            getattr(msg, "seqno", "-"),
        )
        connection_id = self._lwd_connection_of(peer)
        if connection_id in self._lwd_ledger.failed_connections:
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
        if not isinstance(msg, (LwdRequestNotify, LwdAbortNotify)):
            return
        canonical_id = wrap_req_id(peer.instance_id, msg.request_id)
        msg = msgspec.structs.replace(msg, request_id=canonical_id)
        if isinstance(msg, LwdRequestNotify):
            self._lwd_ledger.retired_routes.pop(canonical_id, None)
            self._lwd_ledger.request_routes[canonical_id] = peer
            self._lwd_handle_request_notify(msg)
        else:
            self._lwd_handle_abort_notify(msg)

    def _lwd_connection_of(self, peer: LwdPeer) -> LwdConnectionId:
        return LwdConnectionId(
            peer.instance_id, self.lwd_config.instance_id, peer.dp_rank
        )

    def _lwd_apply_topology_increment(self, increment: LwdTopologyIncrement) -> None:
        """C-2/C-3/C-4: admit each new edge to the handshake allowlist,
        build its data-plane pair on the shared listener, and wire its
        hint/done MQ pair. The rest of the join is the existing runtime
        machinery — registration resets the seqno ledgers and resends
        the scheduling limits, the heartbeat fan-out picks the peer up
        from `_lwd_edges`. A data-plane or MQ refusal keeps the edge
        control-plane only; the next detected increment retries.
        Idempotent per edge and per connection: replays do not disturb
        the live allowlist, the listener's route table or the MQs."""
        plane = self._lwd_control_plane
        assert isinstance(plane, LwdCloudControlPlane)
        # In lwd_cluster each cloud engine serves only its own instance's
        # connections (other cloud engines watch the same file and apply
        # their own). In cloud_share every new connection goes to the
        # single shared cloud, but a shared cloud with multiple DP ranks
        # runs one engine per DP: each engine must adopt only its own
        # DP's connections (the same per-DP filter as __init__), or every
        # engine extends worker state with the other DPs' connection ids
        # and the runner's per-edge connection resolution loses
        # uniqueness.
        my_connections = [
            connection
            for connection in increment.new_connections
            if connection.connection_id.cloud_id == self.lwd_config.instance_id
            and connection.connection_id.dp_idx
            == self.vllm_config.parallel_config.data_parallel_index
        ]
        for edge_id in increment.new_edge_ids:
            # Only admit edges that connect to THIS cloud.
            if not any(
                c.connection_id.edge_id == edge_id
                and c.connection_id.cloud_id == self.lwd_config.instance_id
                for c in my_connections
            ):
                continue
            plane.allow_edge(edge_id)
            lwd_logger.error(
                "[LWD][cloud] edge %d admitted to the handshake allowlist "
                "(topology increment)",
                edge_id,
            )
        for connection in my_connections:
            # The UCX listener, token route table and IO registry live in
            # the WORKER processes (MultiprocExecutor); the data-plane
            # hook registry is process-local and empty in the EngineCore.
            # Build via collective_rpc: the endpoint Worker (the one
            # that owns the shared listener) does the work, others return
            # None (not their job, distinct from False = refused).
            results = self.model_executor.collective_rpc(
                "lwd_build_runtime_connection", args=(connection,)
            )
            built = any(result is True for result in results)
            if not built:
                lwd_logger.error(
                    "[LWD] data-plane build refused for %s; the edge's UP "
                    "frames will keep retrying the listener (retried on "
                    "the next topology change)",
                    connection.connection_id,
                )
                continue
            lwd_logger.error(
                "[LWD] runtime connection %s built (serve pair on the shared listener)",
                connection.connection_id,
            )
            self._lwd_wire_runtime_hint_mq(connection.connection_id)

    def _lwd_wire_runtime_hint_mq(self, connection: LwdConnectionId) -> None:
        """C-4: wire one runtime connection's hint/done MQ pair — the
        engine creates the hint MQ, registers the hint sender, RPCs the
        worker to attach/create the per-connection MQs (passing the hint
        MQ's handle explicitly: env vars do NOT cross the process
        boundary in MultiprocExecutor at runtime — they only work at
        startup where workers inherit the engine's env), and stores the
        returned done MQ handle for the drain loop. Idempotent: an
        already-wired connection skips. A failed attempt rolls back its
        local resources so a later topology increment can retry."""
        if connection in self._lwd_hint_mqs:
            return
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
            _encode_handle,
            _hint_senders,
            create_hint_mq,
            register_hint_sender,
        )

        hint_mq = None
        done_mq = None
        sender = None
        previous_sender = _hint_senders.get(connection)
        try:
            hint_mq = create_hint_mq(connection)

            def send_hint(hint: dict[str, Any], mq: Any = hint_mq) -> None:
                mq.enqueue(hint)

            sender = send_hint
            register_hint_sender(sender, connection)
            hint_handle_str = _encode_handle(hint_mq.export_handle())
            handles = [
                handle
                for handle in self.model_executor.collective_rpc(
                    "lwd_attach_runtime_connection_mqs",
                    args=(connection, hint_handle_str),
                )
                if handle is not None
            ]
            if len(handles) != 1:
                lwd_logger.error(
                    "[LWD] runtime MQ wiring failed for %s: expected one "
                    "endpoint worker to report a done MQ handle, got %d; "
                    "rolling back for a later topology increment to retry",
                    connection,
                    len(handles),
                )
            else:
                from vllm.distributed.device_communicators.shm_broadcast import (
                    MessageQueue,
                )

                done_mq = MessageQueue.create_from_handle(handles[0], 0)
        except Exception:
            lwd_logger.exception(
                "[LWD] runtime MQ wiring failed for %s; rolling back for "
                "a later topology increment to retry",
                connection,
            )
        else:
            if hint_mq is not None and done_mq is not None:
                self._lwd_hint_mqs[connection] = hint_mq
                self._lwd_done_mqs[connection] = done_mq
                lwd_logger.error("[LWD] runtime hint/done MQs wired for %s", connection)
                return

        if sender is not None and _hint_senders.get(connection) is sender:
            if previous_sender is None:
                _hint_senders.pop(connection, None)
            else:
                _hint_senders[connection] = previous_sender
        for mq in (done_mq, hint_mq):
            if mq is None:
                continue
            try:
                mq.shutdown()
            except Exception:
                lwd_logger.exception(
                    "[LWD] runtime MQ rollback cleanup failed for %s", connection
                )

    def _lwd_handle_range_notify(
        self, connection_id: LwdConnectionId, notify: LwdRangeNotify
    ) -> None:
        """Post each new extended hint exactly once, per source connection."""
        latest = self._lwd_ledger.latest_seqnos.get(connection_id, -1)
        if notify.seqno <= latest:
            lwd_logger.info(
                "[LWD][cloud] ignore duplicate RangeNotify seqno=%d", notify.seqno
            )
            return
        items = notify.items
        request_ids = [item.request_id for item in items]
        prompt_lens = [
            self._lwd_ledger.prompt_lens.get(
                request_id, self._lwd_ledger.retired_prompt_lens.get(request_id, 0)
            )
            for request_id in request_ids
        ]
        if any(prompt_len <= 0 for prompt_len in prompt_lens):
            raise RuntimeError(
                f"[LWD] RangeNotify seqno={notify.seqno} arrived before "
                "RequestNotify prompt lengths were recorded"
            )
        self._lwd_ledger.track_range(connection_id, notify.seqno, request_ids)
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
            make_recv_hint,
            post_irecv_hint,
        )

        num_tokens = sum(item.num_tokens for item in items)
        lwd_logger.info(
            "[LWD][pre-recv] HINT seqno=%d num_tokens=%d ts=%.3f",
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
            make_recv_hint(
                notify.seqno,
                num_tokens,
                connection_id=connection_id,
                **hint_fields,
            )
        )

    def _lwd_handle_request_notify(self, notify: LwdRequestNotify) -> None:
        request_id = notify.request_id
        self._lwd_ledger.prompt_lens[request_id] = notify.num_prompt_tokens
        if notify.prefix_nonce:
            self._lwd_ledger.request_nonces[request_id] = notify.prefix_nonce
        request = self._lwd_build_request(notify)
        if request.use_structured_output:
            # This path constructs Request directly and therefore bypasses
            # EngineCore.preprocess_add_request, the native grammar-init hook;
            # without this explicit init the request would sit in
            # WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR forever with no error.
            self.structured_output_manager.grammar_init(request)
        self.input_queue.put_nowait((EngineCoreRequestType.ADD, (request, 0)))
        lwd_logger.info(
            "[LWD][cloud] request added req=%s tokens=%d nonce=%d",
            request_id,
            notify.num_prompt_tokens,
            notify.prefix_nonce,
        )
        # A coordinated MoE DP engine must let the native ADD path observe the
        # paused state and emit start_wave. Waking only this rank here suppresses
        # that notification and can leave its peers asleep for the next collective.
        native_dp_wake = isinstance(self, DPEngineCoreProc) and self.has_coordinator
        if not self.engines_running and not native_dp_wake:
            self.engines_running = True
            lwd_logger.info(
                "[LWD][cloud] first request wakes a paused wave (local fallback)"
            )

    def _lwd_handle_abort_notify(self, notify: LwdAbortNotify) -> None:
        request_id = notify.request_id
        ledger = self._lwd_ledger
        retired = ledger.retired_nonces.get(request_id)
        if request_id in ledger.retired_req_ids and retired == notify.prefix_nonce:
            return
        if (
            ledger.request_nonces.get(request_id) is None
            and retired is None
            and notify.prefix_nonce != 0
        ):
            lwd_logger.warning(
                "[LWD][cloud] ignore abort for unknown generation req=%s nonce=%d",
                request_id,
                notify.prefix_nonce,
            )
            return
        self.scheduler._lwd_refill_gate.discard(request_id)
        self.scheduler._lwd_refill_pending.pop(request_id, None)
        reservations = getattr(self.scheduler, "lwd_prefix_reservations", None)
        if reservations is not None:
            reservations.release(request_id)
        ledger.retire_request(request_id, preserve_prompt=True)
        lwd_logger.info(
            "[LWD][cloud] abort req=%s nonce=%d", request_id, notify.prefix_nonce
        )
        self.aborts_queue.put_nowait([request_id])
        self.input_queue.put_nowait((EngineCoreRequestType.ABORT, [request_id]))

    def shutdown(self) -> None:
        self._lwd_heartbeat_stop.set()
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

        def block_hasher(request: Request) -> list[BlockHash]:
            if request.num_output_tokens > 0:
                return []
            # The cloud holds only a meta placeholder for the prompt
            # embeddings (the real rows arrive later over the UP channel), so
            # a local token/embedding hash raises "Cannot copy out of meta
            # tensor".  Adopt the edge's wire chain only when it matches the
            # cloud hash granularity; otherwise leave the request without
            # hashes and let add_request fail closed through
            # lwd_wire_hashes_adopted.
            if (
                local_hasher is not None
                and not request.block_hashes
                and len(req_notify.block_hashes) == expected
            ):
                return [BlockHash(h) for h in req_notify.block_hashes]
            return []

        request = Request(
            request_id=req_notify.request_id,
            prompt_token_ids=None,
            prompt_embeds=prompt_embeds,
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=block_hasher if local_hasher is not None else None,
            reasoning_ended=req_notify.reasoning_ended,
            reasoning_parser_kwargs=lwd_decode_reasoning_parser_kwargs(
                req_notify.reasoning_parser_kwargs
            ),
        )
        request.lwd_mrope_positions = req_notify.mrope_positions
        request.lwd_image_placeholder_spans = list(req_notify.image_placeholder_spans)
        request.lwd_prefix_nonce = req_notify.prefix_nonce
        request.lwd_wire_hashes_adopted = (
            local_hasher is not None and len(req_notify.block_hashes) == expected
        )
        request.block_hashes = [BlockHash(h) for h in req_notify.block_hashes]
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
        if req_notify.structured_outputs:
            # Decoded after update_from_generation_config so no later native
            # hook re-validates and overwrites the edge-chosen `_backend`.
            # Fail fast: a half-parsed request must never enter the queue.
            try:
                sampling_params.structured_outputs = lwd_decode_structured_outputs(
                    req_notify.structured_outputs, StructuredOutputsParams
                )
            except ValueError as exc:
                raise ValueError(
                    f"[LWD][cloud] request {req_notify.request_id}: {exc}"
                ) from exc
        LwdControlDiag.cloud_request_admitted(req_notify, sampling_params)
        return sampling_params

    def _lwd_coord_wake_loop(self) -> None:
        """Move MP commands to a local queue and wake the engine thread."""
        command_queue = self.cloud_control_command_queue
        assert command_queue is not None, (
            "[LWD] coordination wake loop requires the control command queue"
        )
        while True:
            try:
                command = command_queue.get()
            except (EOFError, OSError):
                return
            self._lwd_coord_pending.put(command)
            self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

    def _lwd_poll_coordination(self) -> None:
        probe_query = getattr(self, "_lwd_probe_query", None)
        if probe_query is None:
            return
        processor = getattr(
            self, "cloud_control_processor", self._lwd_control_processor
        )
        if processor is not None:
            consumed = processor.poll(probe_query)
            if consumed and not self._lwd_coord_pending.empty():
                self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

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
        self._lwd_publish_finished_usage(engine_core_outputs)
        self._lwd_retire_finished_outputs(engine_core_outputs)
        return engine_core_outputs, model_executed

    def _lwd_heartbeat_thread(self) -> None:
        """Idle-safe liveness beats: an engine with no traffic never steps,
        so the step loop cannot carry the heartbeat — a silent-but-alive
        idle cloud would then read as dead on the edge's first step and
        get its healthy links torn down. This thread only touches the
        input queue (the control sender is bound to the engine thread):
        one tick per hello interval wakes the loop and the engine thread
        sends the beat."""
        while not self._lwd_heartbeat_stop.wait(self._lwd_hello_interval_s):
            self.input_queue.put_nowait(
                (EngineCoreRequestType.WAKEUP, _LWD_HEARTBEAT_TICK)
            )

    def _lwd_send_heartbeat_if_due(self) -> None:
        """Engine thread: one liveness beat per hello interval to every
        registered edge. Dropped (not retried) on a full outbound queue:
        the next beat covers the loss, and retrying would block the engine
        step behind a stalled edge."""
        now = time.monotonic()
        if now - self._lwd_last_heartbeat_at < self._lwd_hello_interval_s:
            return
        self._lwd_last_heartbeat_at = now
        live: list[LwdPeer] = []
        for peer in self._lwd_edges:
            try:
                self._lwd_send(peer, LwdCloudHeartbeat(self._lwd_boot_id))
                live.append(peer)
            except LwdPeerNotRegisteredError:
                # A liveness probe must never kill the prober: the edge's
                # ROUTER is gone (plane already unregistered it and drops
                # its payloads). Stop probing it; re-registration re-adds
                # it to _lwd_edges and the beats resume.
                lwd_logger.error(
                    "[LWD][cloud] edge %s unreachable; heartbeat paused until "
                    "it re-registers",
                    peer,
                )
        self._lwd_edges = set(live)
        lwd_logger.debug(
            "[LWD][cloud] heartbeat sent boot_id=%s edges=%d",
            self._lwd_boot_id,
            len(self._lwd_edges),
        )

    def step(self):
        """Same for the synchronous step path: drain the watermark at
        the head of the step, the rest is native."""
        self._lwd_poll_coordination()
        self._lwd_drain_irecv_completions()
        engine_core_outputs, model_executed = super().step()
        self._lwd_publish_finished_usage(engine_core_outputs)
        self._lwd_retire_finished_outputs(engine_core_outputs)
        return engine_core_outputs, model_executed

    def _lwd_publish_finished_usage(self, engine_core_outputs: Any) -> None:
        records = self.scheduler.lwd_finished_records
        event_queue = getattr(
            self, "cloud_control_event_queue", self.cloud_control_event_queue
        )
        from vllm.v1.lwd_control.lwd_comm.lwd_prefix import LwdUsage

        for request_id, (prompt_tokens, completion_tokens, prefix_nonce) in list(
            records.items()
        ):
            cached_tokens = self._lwd_ledger.prefix_hits.get(request_id, 0)
            usage = None
            if completion_tokens is not None:
                usage = LwdUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                )
            if event_queue is not None:
                event_queue.put(
                    {
                        "type": "usage",
                        "request_id": request_id,
                        "usage": usage,
                        "engine_index": self._lwd_engine_index,
                        "prefix_nonce": prefix_nonce,
                    }
                )
            records.pop(request_id, None)
            self._lwd_ledger.prefix_hits.pop(request_id, None)

    def _lwd_retire_finished_outputs(self, engine_core_outputs: Any) -> None:
        if not engine_core_outputs:
            return
        for outputs in engine_core_outputs.values():
            for request_id in outputs.finished_requests or ():
                self._lwd_ledger.retire_request(request_id)
            for output in outputs.outputs:
                if output.finish_reason is not None:
                    self._lwd_ledger.retire_request(output.request_id)

    def _lwd_drain_irecv_completions(self) -> None:
        """Merge every connection's done MQ into independent watermarks."""
        from vllm_ascend.worker.lwd_cloud_model.lwd_pre_recv import (
            record_irecv_completions,
        )

        watermark_items: list[Any] = []
        for connection, done_mq in self._lwd_done_mqs.items():
            while True:
                try:
                    item = done_mq.dequeue(timeout=0)
                except TimeoutError:
                    break
                except Exception:
                    lwd_logger.exception(
                        "[LWD][pre-recv] done MQ drain failed for %s", connection
                    )
                    break
                if (
                    isinstance(item, tuple)
                    and len(item) == 3
                    and item[0] == "refill_done"
                ):
                    self.scheduler.lwd_refill_done(str(item[1]), int(item[2]))
                    continue
                if (
                    isinstance(item, tuple)
                    and len(item) == 4
                    and item[1] == "link_error"
                ):
                    if item[0] != connection:
                        raise ValueError(
                            f"[LWD] cross-connection early-recv error on "
                            f"{connection}: {item!r}"
                        )
                    lwd_logger.error(
                        "[LWD][pre-recv] link error on %s at seqno=%s: %s "
                        "(transient: retiring its requests, connection NOT "
                        "failed)",
                        connection,
                        item[2],
                        item[3],
                    )
                    self._lwd_retire_connection_requests(connection)
                    continue
                if isinstance(item, tuple) and len(item) == 4 and item[1] == "error":
                    if item[0] != connection:
                        raise ValueError(
                            f"[LWD] cross-connection early-recv error on "
                            f"{connection}: {item!r}"
                        )
                    self._lwd_fail_connection(connection, item[2], item[3])
                    continue
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], tuple)
                ):
                    key, arrived = item
                    if len(key) != 3:
                        raise ValueError(
                            "[LWD] early-recv completion key must be "
                            f"(connection, channel, seqno), got {key!r}"
                        )
                    if key[0] != connection:
                        raise ValueError(
                            f"[LWD] cross-connection early-recv completion on "
                            f"{connection}: {item!r}"
                        )
                    watermark_items.append(key)
                    self._lwd_note_arrived(key[0], key[2], arrived)
                    continue
                raise ValueError(
                    f"[LWD] unsupported early-recv completion item: {item!r}"
                )
        if not watermark_items:
            return
        advanced = record_irecv_completions(watermark_items)
        for connection, _, seqno in advanced:
            self._lwd_ledger.complete_seqno(connection, seqno)
        lwd_logger.info(
            "[LWD][pre-recv] DRAIN n=%d advanced=%s ts=%.3f",
            len(watermark_items),
            advanced,
            time.monotonic(),
        )

    def _lwd_fail_connection(
        self, connection: LwdConnectionId, seqno: int, error: Any
    ) -> None:
        """Abort only the failed connection; other edges keep making progress."""
        ledger = self._lwd_ledger
        if connection in ledger.failed_connections:
            return
        ledger.failed_connections.add(connection)
        lwd_logger.error(
            "[LWD][pre-recv] connection %s failed at seqno=%s: %s",
            connection,
            seqno,
            error,
        )
        self._lwd_retire_connection_requests(connection)

    def _lwd_retire_connection_requests(self, connection: LwdConnectionId) -> None:
        """Abort every request routed to one connection without deciding
        whether the connection may come back: the permanent failed marking
        stays with the caller (a link error retires its in-flight requests
        but leaves the connection recoverable)."""
        ledger = self._lwd_ledger
        request_ids = [
            request_id
            for request_id, peer in ledger.request_routes.items()
            if self._lwd_connection_of(peer) == connection
        ]
        for request_id in request_ids:
            ledger.retire_request(request_id, preserve_prompt=True)
            self.aborts_queue.put_nowait([request_id])
            self.input_queue.put_nowait((EngineCoreRequestType.ABORT, [request_id]))

    def _lwd_note_arrived(
        self,
        connection_id: LwdConnectionId,
        seqno: int,
        arrived: Iterable[tuple[str, int]] | None,
    ) -> None:
        """Merge arrivals only for requests in this connection's pending seqno."""
        ledger = self._lwd_ledger
        pending = ledger.pending_seqnos.get(connection_id, {})
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
                or request_id in ledger.retired_req_ids
            ):
                continue
            previous = self.scheduler.lwd_recv_rows.get(request_id, 0)
            if rows > previous:
                self.scheduler.lwd_recv_rows[request_id] = rows
                touched.append((request_id, previous, rows))
        if touched:
            lwd_logger.info("[LWD][cloud] arrival high watermarks updated: %s", touched)

    def _lwd_log_cloud_concurrency(self) -> None:
        """Debug-level snapshot of this engine's concurrency: how many
        requests it is running right now, plus the backlog and the KV
        cache occupancy they hold."""
        now = time.monotonic()
        if now - self._lwd_cloud_conc_last_log < _LWD_CLOUD_CONC_LOG_INTERVAL_S:
            return
        self._lwd_cloud_conc_last_log = now
        scheduler = self.scheduler
        block_pool = scheduler.kv_cache_manager.block_pool
        lwd_logger.debug(
            "[LWD][dbg] cloud concurrency cloud_id=%d engine=%d running=%d "
            "waiting=%d skipped=%d total=%d step=%d | kv_cache usage=%.1f%% "
            "free=%d total=%d",
            self.lwd_config.instance_id,
            self._lwd_engine_index,
            len(scheduler.running),
            len(scheduler.waiting),
            len(scheduler.deferred_waiting),
            len(scheduler.requests),
            scheduler.current_step,
            block_pool.get_usage() * 100.0,
            block_pool.get_num_free_blocks(),
            max(0, block_pool.num_gpu_blocks - 1),
        )

    def post_step(self, model_executed: bool) -> None:
        """Flush pending c2e to the edge at the end of the step (the
        scheduler only queues them; both native step paths reach this hook)."""
        self._lwd_log_cloud_concurrency()
        if self.scheduler.pending_c2e:
            finished_ids: list[str] = []
            for notify in self.scheduler.pending_c2e:
                if not notify.req_ids:
                    continue
                peers = {
                    self._lwd_ledger.request_routes.get(request_id)
                    or self._lwd_ledger.retired_routes.get(request_id)
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
                    req_ids=[
                        unwrap_req_id(request_id) for request_id in notify.req_ids
                    ],
                )
                self._lwd_send_until_queued(peer, restored)
                # Finished routes stay pinned until every pending C2E has
                # flushed; later zero-accepted/retirement notices are legal.
                for index, request_id in enumerate(notify.req_ids):
                    if notify.finish_reasons and notify.finish_reasons[index] != -1:
                        finished_ids.append(request_id)
            self.scheduler.pending_c2e.clear()
            for request_id in finished_ids:
                self._lwd_ledger.clear_retired(request_id)
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
