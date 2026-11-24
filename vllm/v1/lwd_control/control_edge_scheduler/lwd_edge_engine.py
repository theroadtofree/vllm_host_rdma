# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Edge engine core: the control plane to every cloud DP rank, the check of
each cloud's scheduling limits and the request entry (edge admission +
announcement to the routed cloud DP rank; communication stays in the engine).

Stepping is the native step_with_batch_queue, adapted at construction: host
comm sizes the queue for dynamic scale-out (max instances x 2 slots), device
comm keeps the depth-2 native baseline; is_pooling_model=True (no native
sample_tokens) and an executor proxy (a normally completed batch returning
None becomes a placeholder).

Startup blocks until every cloud DP rank accepted the handshake and sent its
scheduling limits, within one handshake timeout; any failure stops the
engine. There is no self-healing: restart the whole group.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from vllm.config.lwd import LwdConnectionId, LwdTopologyIncrement
from vllm.config.lwd.topology import LwdDPConnection
from vllm.lwd_logger import lwd_logger
from vllm.utils.network_utils import make_zmq_path
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
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_gateway_activation import (
    LwdGatewayActivationManager,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_heartbeat import (
    LwdHeartbeatMonitor,
    LwdHeartbeatVerdict,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_engine import (
    LwdBaseEngineCore,
)
from vllm.v1.lwd_control.lwd_comm.lwd_control_plane import (
    LwdEdgeControlPlane,
    LwdPeerNotRegisteredError,
)
from vllm.v1.lwd_control.lwd_comm.lwd_control_plane_factory import lwd_ctrl_tos
from vllm.v1.lwd_control.lwd_comm.lwd_data_plane_hooks import (
    notify_data_plane_added,
)
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdPeer
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_SAMPLING_DEFAULTS,
    LwdAbortNotify,
    LwdC2eNotify,
    LwdCloudHeartbeat,
    LwdCloudParamsNotify,
    LwdEdgeToCloudMessage,
    LwdMessage,
    LwdPrefixReadyNotify,
    LwdRefillNotify,
    LwdRequestNotify,
    lwd_decode_cloud_to_edge,
    lwd_encode_reasoning_parser_kwargs,
    lwd_encode_structured_outputs,
)
from vllm.v1.lwd_control.lwd_exceptions import LwdTransferFailedError
from vllm.v1.request import Request

# Native batch queue depth: dispatched (embed + unembed) batches not yet
# reaped. 2 keeps the pipeline aligned; the reference branch's 4 is
# conservative, validate performance with 2 first.
LWD_EDGE_BATCH_QUEUE_DEPTH = 2
# Host comm (UCX) scales the instance count dynamically, so the queue is
# sized once for the maximum deployment instead of the assembly-time
# connection count: LWD_EDGE_HOST_MAX_INSTANCES x slots. 2 slots is the
# per-connection
# end-to-end window (the cloud's early-recv pump caps in-flight UP chunks
# per connection at 2); deeper per-connection queuing only accumulates
# behind its backpressure.
LWD_EDGE_HOST_MAX_INSTANCES = 4
LWD_EDGE_HOST_SLOTS_PER_INSTANCE = 2
LWD_PREFIX_READY_TIMEOUT_S = 60.0
# Liveness sweep rate limit: the verdict's latency granularity.
_LWD_SWEEP_INTERVAL_S = 1.0
# Seconds without ANY cloud activity (heartbeat or a valid business frame)
# before the edge tears that cloud DP's data-plane links down and gates new
# requests to it; the next heartbeat or business frame releases the gate.
# Also bounds the in-flight failure latency of the teardown. This threshold
# and the cloud's heartbeat cadence (LWD_HELLO_INTERVAL_S in
# lwd_cloud_engine) are the two halves of ONE liveness contract — the
# judge's threshold and the sender's pace — so any change must consider
# both files together. Business traffic is a second, independent evidence
# of life: a heartbeat frame queued behind a busy outbound FIFO must not
# make a healthy cloud look dead.
LWD_HELLO_SILENT_S = 10.0

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
        # Runtime-added connections (elastic scale-out), keyed by
        # (cloud_id, dp_idx): the engine thread puts, the control-plane IO
        # thread only gets — a keyed read cannot observe a half-applied add.
        self._lwd_runtime_connections: dict[tuple[int, int], LwdDPConnection] = {}
        # Only connections whose data-plane links are ready open for service.
        # A refused build keeps the connection control-plane only; the next
        # detected increment retries that half.
        self._lwd_serviceable_connections: set[tuple[int, int]] = set()
        self._lwd_prefix_enabled = bool(
            self.vllm_config.cache_config.enable_prefix_caching
        )
        self._lwd_prefix_control = (
            LwdEdgePrefixControl(self, LWD_PREFIX_READY_TIMEOUT_S)
            if self._lwd_prefix_enabled
            else None
        )
        self._lwd_cloud_params: dict[LwdConnectionId, LwdCloudParamsNotify] = {}
        gateway = self.lwd_config.topology.gateway
        lwd_logger.info(
            "[LWD] coordination mode=%s gateway_url=%s",
            "gateway" if gateway.enabled else "direct",
            gateway.url if gateway.enabled else "-",
        )
        manager_cls = (
            LwdGatewayActivationManager if gateway.enabled else LwdEdgeProbeManager
        )
        self._lwd_probe_manager = manager_cls(
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
        # Liveness is owned by LwdHeartbeatMonitor, created BEFORE the
        # control plane starts: its IO thread dispatches payloads into this
        # engine immediately, and a frame arriving mid-assembly (long
        # handshake or params waits) must land in an existing monitor, not
        # a half-constructed object. The engine only wires the monitor to
        # its consumers; it holds no liveness state of its own.
        self._lwd_heartbeat = LwdHeartbeatMonitor(
            LWD_HELLO_SILENT_S, _LWD_SWEEP_INTERVAL_S
        )
        self._lwd_heartbeat.subscribe(
            lambda verdict: self.scheduler.lwd_update_silent_routes(
                verdict.silent_routes
            )
        )
        self._lwd_heartbeat.subscribe(self._lwd_apply_silent_clouds)
        # Adopted-cloud state must exist before the control plane starts:
        # a cloud's scheduling limits can arrive mid-assembly (the startup
        # wait below is exactly that arrival), and the params handler
        # adopts into these — same reason the liveness accounts live here.
        self._lwd_clouds: dict[LwdConnectionId, LwdPeer] = {}
        self._lwd_coord_clients: dict[LwdConnectionId, LwdCoordClient] = {}
        plane = self._lwd_start_control_plane()
        assert isinstance(plane, LwdEdgeControlPlane)
        handshake_timeout_s = plane.spec.config.handshake_timeout_s
        # Assembly-time anchor: while this wait runs the engine serves
        # NOTHING (no io loop yet), so a request sent during it produces
        # zero engine logs by design — this line is how you tell that
        # state apart from a stuck engine thread.
        lwd_logger.info(
            "[edge] waiting for cloud handshakes: %d planned connections "
            "(timeout %.0fs)",
            len(self.lwd_config.connections),
            handshake_timeout_s,
        )
        deadline = time.monotonic() + handshake_timeout_s
        self._lwd_clouds.update(plane.wait_until_ready(handshake_timeout_s))
        # Seed the accounts for the discovered connections (a beat that
        # already arrived mid-assembly wins over the seeding). tuple()
        # snapshot: _lwd_clouds also gains entries on the control-plane
        # IO thread (a cloud's limits adopt mid-assembly), and a
        # Python-level dict iteration raises on that insert while
        # tuple(dict) completes in C.
        for connection in tuple(self._lwd_clouds):
            self._lwd_heartbeat.note_adopted(connection)
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
        # Snapshot: a straggler cloud completes its handshake after the
        # wait and its limits land on the IO thread while this loop runs.
        for connection, params in tuple(self._lwd_cloud_params.items()):
            self._lwd_adopt_cloud(connection, self._lwd_clouds[connection], params)
        self._lwd_configure_prefix_cache()
        self.scheduler.lwd_send_to_cloud = self._lwd_send_to_cloud
        self.scheduler.lwd_refill_abort = self._lwd_abort_refill
        self.is_pooling_model = True
        if self.lwd_config.topology.is_host_comm:
            # Host comm scales instances dynamically, so size for the
            # maximum once (see LWD_EDGE_HOST_MAX_INSTANCES); the queue
            # survives scale-out without re-assembly.
            self.batch_queue_size = (
                LWD_EDGE_HOST_MAX_INSTANCES * LWD_EDGE_HOST_SLOTS_PER_INSTANCE
            )
        else:
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

    def _lwd_apply_silent_clouds(self, verdict: LwdHeartbeatVerdict) -> None:
        """Route the cloud-grained verdict to the active probe sink.

        Direct and gateway coordination modes install different managers.
        Dispatch by type so a gateway deployment never calls a
        direct-probe-only method, which would raise on every monitor emit
        (one per sweep) and be logged as a subscriber failure. An unknown
        manager is a wiring error, not a state to silently ignore.
        """
        manager = self._lwd_probe_manager
        if isinstance(manager, (LwdEdgeProbeManager, LwdGatewayActivationManager)):
            manager.set_silent_clouds(verdict.silent_clouds)
        else:
            raise TypeError(
                "[LWD] unsupported probe manager for the liveness verdict: "
                f"{type(manager).__name__}"
            )

    def _lwd_apply_topology_increment(self, increment: LwdTopologyIncrement) -> None:
        """Elastic scale-out: build both halves of every new connection —
        the control plane (DEALER + handshake + bookkeeping), then the
        data plane (UCX links), which on success opens the connection for
        service. Idempotent per connection: serviceable connections replay
        silently; a connection whose data-plane build was refused retries
        only that half on the next detected increment."""
        plane = self._lwd_control_plane
        assert isinstance(plane, LwdEdgeControlPlane)
        scene = self.lwd_config.topology.deployment.scene
        for connection in increment.new_connections:
            # In lwd_cluster each edge engine serves only its own
            # instance's connections (other engines watch the same file
            # and apply their own). In edge_share the shared engine owns
            # EVERY logical edge — including ones the increment adds
            # beyond its own startup snapshot — so only lwd_cluster
            # filters, and by instance identity, never by the startup
            # connection set.
            if (
                scene == "lwd_cluster"
                and connection.connection_id.edge_id != self.lwd_config.instance_id
            ):
                lwd_logger.error(
                    "[LWD] topology increment %s skipped: owned by edge %d "
                    "(this engine serves edge %d)",
                    connection.connection_id,
                    connection.connection_id.edge_id,
                    self.lwd_config.instance_id,
                )
                continue
            key = (
                connection.connection_id.cloud_id,
                connection.connection_id.dp_idx,
            )
            if key in self._lwd_runtime_connections:
                if key not in self._lwd_serviceable_connections:
                    self._lwd_open_runtime_connection_for_service(connection)
                continue
            if connection.cloud.ctrl_port is None:
                lwd_logger.error(
                    "[LWD] topology increment %s dropped: the cloud DP has "
                    "no ctrl_port",
                    connection.connection_id,
                )
                continue
            self._lwd_register_probe_cloud(connection)
            self._lwd_runtime_connections[key] = connection
            self.scheduler.lwd_add_runtime_connection(connection.connection_id)
            plane.add_link(
                make_zmq_path("tcp", connection.cloud.addr, connection.cloud.ctrl_port),
                connection.connection_id,
                len(
                    increment.new_topology.instance(
                        "cloud", connection.connection_id.cloud_id
                    ).dp
                ),
            )
            lwd_logger.error(
                "[LWD] topology increment applied (control plane): "
                "connection %s -> %s:%s",
                connection.connection_id,
                connection.cloud.addr,
                connection.cloud.ctrl_port,
            )
            self._lwd_open_runtime_connection_for_service(connection)

    def _lwd_register_probe_cloud(self, connection: LwdDPConnection) -> None:
        """Elastic scale-out: teach the probe manager the adopted cloud
        DP's coordination URL, so probe targeting — and the routing
        preference probes produce — spreads over every adopted cloud
        instead of staying pinned to the startup set. On this schema the
        URL lives in the cloud's own instance block: an explicit
        ``coord_url`` (DNS/NAT/gateway deployments) wins, else it is
        derived from this DP's routable address and coordination port —
        the same host the control-plane DEALER dials."""
        cloud_dp = connection.cloud
        if cloud_dp.coord_url is None and cloud_dp.coord_port is None:
            # An edge_share-scale cloud runs without a coordination block:
            # no probe targeting for it, by design.
            lwd_logger.error(
                "[LWD] probe target %s not registered: the cloud DP has no "
                "coordination endpoint",
                connection.connection_id,
            )
            return
        url = cloud_dp.coord_url or f"http://{cloud_dp.addr}:{cloud_dp.coord_port}"
        # First registration wins: a later replay of the same increment
        # confirms the rotation membership without rewriting the URL.
        self._lwd_probe_manager.register_connection(connection.connection_id, url)
        lwd_logger.error(
            "[LWD] probe target registered: connection=%s url=%s rotation=%s "
            "(probes — and the routing preference they produce — now "
            "spread over every adopted cloud)",
            connection.connection_id,
            url,
            self._lwd_probe_manager.rotation(),
        )

    def _lwd_open_runtime_connection_for_service(
        self, connection: LwdDPConnection
    ) -> None:
        """Build the connection's data-plane links; on success open it for
        service (admission gate + routing candidates). A refusal — budget,
        NPU assembly — keeps the connection control-plane only: the gate
        keeps rejecting it, and the next detected increment retries this
        half."""
        started = time.monotonic()
        lwd_logger.error(
            "[LWD] data-plane build start for %s (the engine thread waits "
            "for the build to return; requests are not served until it "
            "does)",
            connection.connection_id,
        )
        if not notify_data_plane_added(connection):
            lwd_logger.error(
                "[LWD] data-plane build refused for %s (%.1fs); the connection "
                "stays control-plane only (retried on the next topology "
                "change)",
                connection.connection_id,
                time.monotonic() - started,
            )
            return
        key = (
            connection.connection_id.cloud_id,
            connection.connection_id.dp_idx,
        )
        self._lwd_serviceable_connections.add(key)
        self.scheduler.lwd_add_runtime_route(connection.connection_id)
        lwd_logger.error(
            "[LWD] runtime connection %s open for service (data plane up, %.1fs)",
            connection.connection_id,
            time.monotonic() - started,
        )

    def _lwd_configure_prefix_cache(self) -> None:
        capabilities: set[tuple[int, int, bool, bool]] = set()
        # Snapshot: a straggler cloud's limits can land on the IO thread
        # while this engine-thread loop runs.
        for dp_idx, params in tuple(self._lwd_cloud_params.items()):
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
        # Snapshot: a late adoption on the IO thread can add a client
        # while shutdown iterates.
        for client in tuple(getattr(self, "_lwd_coord_clients", {}).values()):
            client.shutdown()
        super().shutdown()

    def _lwd_decode(self, payload: bytes) -> LwdMessage:
        return lwd_decode_cloud_to_edge(payload)

    def _lwd_connection_of(self, peer: LwdPeer) -> LwdConnectionId:
        # Runtime-added connections first (keyed read on the IO thread),
        # then the immutable planned set.
        runtime = self._lwd_runtime_connections.get((peer.instance_id, peer.dp_rank))
        if runtime is not None:
            return runtime.connection_id
        return next(
            connection
            for connection in self.lwd_config.connections
            if (connection.connection_id.cloud_id, connection.connection_id.dp_idx)
            == (peer.instance_id, peer.dp_rank)
        ).connection_id

    def _lwd_dispatch_message(self, peer: LwdPeer, msg: LwdMessage) -> bool:
        """Cloud limits -> record and wake the startup wait; heartbeats ->
        liveness accounts; c2e -> scheduler notify queue + WAKEUP; anything
        else is unknown. Every accepted frame also refreshes the cloud DP's
        activity evidence in the heartbeat monitor, so a cloud that is
        answering requests is never judged dead because its heartbeat was
        delayed."""
        if isinstance(msg, LwdCloudHeartbeat):
            self._lwd_heartbeat.note_activity(
                self._lwd_connection_of(peer), heartbeat_boot_id=msg.boot_id
            )
            return True
        if isinstance(msg, LwdCloudParamsNotify):
            connection = self._lwd_connection_of(peer)
            self._lwd_cloud_params[connection] = msg
            # Elastic partial start: a cloud that started after the edge
            # adopts here — the periodic HELLO kept its handshake alive
            # and these limits are the cloud's first runtime traffic.
            self._lwd_adopt_cloud(connection, peer, msg)
            self.scheduler.lwd_update_cloud_params(connection, msg)
            self._lwd_cloud_params_received.set()
            self._lwd_heartbeat.note_activity(connection)
            return True
        if isinstance(msg, (LwdPrefixReadyNotify, LwdRefillNotify)):
            self._lwd_heartbeat.note_activity(self._lwd_connection_of(peer))
            self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, msg))
            return True
        if isinstance(msg, LwdC2eNotify):
            connection_id = self._lwd_connection_of(peer)
            # Any valid cloud frame is liveness evidence: a heartbeat queued
            # behind traffic (or a long cloud step) must not make a cloud
            # that is still answering requests look dead.
            self._lwd_heartbeat.note_activity(connection_id)
            # One per decode step per engine on the control-plane IO thread,
            # which shares the GIL with the only scheduling thread: debug,
            # not info.
            lwd_logger.debug(
                "[edge] c2e received from %s reqs=%d hidden_numel=%d",
                connection_id,
                len(msg.req_ids),
                msg.hidden_num_elements,
            )
            self.scheduler.unembed_notify_queue.append((connection_id, msg))
            self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
            return True
        return False

    def _lwd_adopt_cloud(
        self, connection: LwdConnectionId, peer: LwdPeer, params: LwdCloudParamsNotify
    ) -> None:
        """Register one accepted cloud with the runtime engine state:
        validated scheduling limits, the probe coordinator client when the
        cloud runs one, prefix hash sizes, the sendable peer, and a
        liveness account seed.

        Init adopts the accepted subset this way; a cloud that starts
        later is adopted on the control-plane IO thread when its
        scheduling limits arrive — its handshake keeps being retried by
        the periodic HELLO (elastic partial start). Idempotent: a
        re-arrival just refreshes the state."""
        self._lwd_check_cloud_params(connection, params)
        self._lwd_clouds[connection] = peer
        self._lwd_heartbeat.note_adopted(connection)
        if params.dp_coordinator_addr:
            tos = lwd_ctrl_tos(self.lwd_config)
            self._lwd_coord_clients[connection] = LwdCoordClient(
                connection.dp_idx, params.dp_coordinator_addr, tos
            )
        if params.prefix_hash_block_size > 0:
            self._lwd_probe_manager._hash_sizes[
                (connection.cloud_id, connection.dp_idx)
            ] = params.prefix_hash_block_size

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
        once the retries run out instead of losing the message. An
        unregistered cloud (its ROUTER is gone; the plane already drops
        the payloads) drops the message instead of failing the engine:
        the restarted cloud's state is fresh, the request fails through
        the data-plane containment path."""
        cloud = self._lwd_clouds.get(dp_idx)
        if cloud is None:
            # Elastic partial start residual: the connection's cloud is
            # planned but not adopted yet (adoption happens when its
            # scheduling limits arrive). The admission gate rejects such
            # requests up front; anything that still lands here drops
            # with a log like the unregistered path — a KeyError would
            # kill the shared engine.
            lwd_logger.error(
                "[edge] %s to cloud %s dropped: connection not adopted yet",
                type(message).__name__,
                dp_idx,
            )
            return
        try:
            for attempt in range(_LWD_SEND_RETRY_STEPS):
                if self._lwd_send(cloud, message):
                    return
                time.sleep(_LWD_SEND_RETRY_INTERVAL_S * (attempt + 1))
        except LwdPeerNotRegisteredError:
            lwd_logger.error(
                "[edge] %s to cloud %s %s dropped: peer unreachable",
                type(message).__name__,
                dp_idx,
                cloud,
            )
            return
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
        # Availability gate, at the ADMISSION ENTRY (old-branch invariant):
        # the gate is the first thing a request meets, before the
        # probe/prefix machinery — a probe sent to a dead cloud would hang
        # for its full timeout instead of rejecting. The
        # shutdown-rejection paradigm finishes the request DIRECTLY:
        # nothing enqueues after this returns, so the abort-channel path
        # used post-probe would never find it.
        gate_reason = self._lwd_gate_reason(request)
        if gate_reason is not None:
            lwd_logger.error(
                "[edge] request rejected req=%s: %s",
                request.request_id,
                gate_reason,
            )
            # ERROR finish code (NOT abort), old-branch parity: the client
            # sees a request FAILURE, not a cancellation — the "service not
            # ready" semantics of an unstarted edge-cloud pair. An abort
            # finish would surface as a cancelled/empty completion instead
            # of an explicit reject.
            self._send_error_outputs_to_client(
                [request.request_id], request.client_index
            )
            return
        if request.abort_immediately:
            self._lwd_add_request_after_probe(request)
            return
        control = getattr(self, "_lwd_prefix_control", None)
        if control is not None:
            control.add_request(request)
            return
        submission = self._lwd_probe_manager.submit(request)
        if submission is LwdProbeSubmission.SUBMITTED:
            return
        if submission is LwdProbeSubmission.FAILED:
            self._lwd_abort_internally(request)
            return
        self._lwd_add_request_after_probe(request)

    def _lwd_admit_and_notify(
        self,
        request: Request,
        preferred: LwdConnectionId | None = None,
    ) -> None:
        gate_reason = self._lwd_gate_reason(request)
        if gate_reason is not None:
            # Availability gate, defense in depth (the primary check is the
            # add_request ENTRY): this one covers the probe race — the
            # probe went out while the cloud was alive, and it completed
            # after the cloud went silent. Reject through the abort
            # channel — aborts_queue is drained at the NEXT step head, after
            # the caller's single native enqueue has entered the scheduler,
            # so the abort finds the request exactly once and the client
            # gets a fast, retryable finish instead of a zombie-link hang.
            # (Enqueueing HERE as well double-adds the request: the caller
            # enqueues unconditionally after this returns.)
            lwd_logger.error(
                "[edge] request rejected req=%s: %s",
                request.request_id,
                gate_reason,
            )
            self.aborts_queue.put_nowait([request.request_id])
            return
        try:
            dp_idx = self.scheduler.lwd_admit_request(request, preferred)
            # The one line that answers "which cloud did this request go to":
            # routed-connection vs preferred-connection contrast exposes a
            # dropped probe preference (None = round-robin/scoring decided).
            lwd_logger.info(
                "[edge] request routed req=%s -> %s (probe preferred=%s)",
                request.request_id,
                dp_idx,
                preferred,
            )
            if dp_idx not in self._lwd_clouds:
                # Elastic partial start: routing scored a planned connection
                # whose cloud is not adopted yet (a multi-cloud-per-edge pick
                # of the un-adopted sibling, or a race with adoption). Same
                # fast, retryable rejection as the gate — cancel the
                # admission, never notify an unaddressable cloud.
                lwd_logger.error(
                    "[edge] request rejected req=%s: %s not adopted yet",
                    request.request_id,
                    dp_idx,
                )
                self.scheduler.lwd_edge_cancel_admission(request.request_id)
                self.aborts_queue.put_nowait([request.request_id])
                return
            client = self._lwd_coord_clients.get(dp_idx)
            if client is not None:
                client.send_first_req()
            self._lwd_notify_request(dp_idx, request)
            self._lwd_probe_manager.mark_admitted(
                request.request_id, getattr(request, "lwd_prefix_nonce", 0)
            )
        except Exception:
            self._lwd_probe_manager.abort(request.request_id)
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
        if completion is not None:
            request.lwd_prefix_nonce = completion.probe_nonce
        if completion is not None and completion.chain is not None:
            self._lwd_probe_chains[request.request_id] = completion.chain
        if completion is not None and completion.result is not None:
            coordination = self.lwd_config.coordination
            placement_only = (
                (coordination is not None and coordination.placement_only)
                or not self._lwd_prefix_enabled
                or request.skip_reading_prefix_cache
            )
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
        if completion is None or not completion.activated:
            return None
        preferred = completion.connection_id
        if preferred is None or preferred in self._lwd_clouds:
            return preferred
        sibling = next(
            (
                connection_id
                for connection_id in sorted(self._lwd_clouds)
                if connection_id.cloud_id == preferred.cloud_id
            ),
            None,
        )
        if sibling is not None:
            return sibling
        # The probe SUCCEEDED against this connection but the engine has
        # no adopted connection for it: the routing preference the probe
        # produced is dropped and the request falls back to
        # score/round-robin. On a scaled-out deployment this means the
        # control-plane join (handshake -> params -> adopt) of the new
        # cloud never completed on this edge.
        lwd_logger.error(
            "[edge] probe preferred %s but it is not adopted "
            "(adopted=%s); routing falls back to round-robin",
            preferred,
            sorted(self._lwd_clouds),
        )
        return None

    def _lwd_drain_probe_completions(self) -> None:
        for completion in self._lwd_probe_manager.drain():
            if not completion.activated:
                if not completion.candidates:
                    lwd_logger.warning(
                        "[edge-probe] all candidates failed req=%s targets=%s; "
                        "using full prefill",
                        completion.request.request_id,
                        completion.failed_targets,
                    )
                hits = {
                    candidate.connection_id: candidate.normalized_hit_tokens(
                        completion.request.num_prompt_tokens
                    )
                    for candidate in completion.candidates
                }
                target = self.scheduler.lwd_select_probe_target(
                    completion.request, hits
                )
                candidate = next(
                    (
                        item
                        for item in completion.candidates
                        if item.connection_id == target
                    ),
                    None,
                )
                manager = self._lwd_probe_manager
                if isinstance(manager, LwdEdgeProbeManager) and not manager.activate(
                    completion, target, candidate
                ):
                    lwd_logger.warning(
                        "[edge-probe] stale activation req=%s ignored",
                        completion.request.request_id,
                    )
                continue
            if completion.activation_error is not None:
                lwd_logger.error(
                    "[edge-probe] activate failed req=%s target=%s: %s",
                    completion.request.request_id,
                    completion.connection_id,
                    completion.activation_error,
                )
                self._lwd_abort_internally(completion.request)
                continue
            self._lwd_add_request_after_probe(completion.request, completion)

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        self._lwd_drain_probe_completions()
        return super().step()

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        self._lwd_drain_probe_completions()
        self._lwd_heartbeat.sweep()
        try:
            return super().step_with_batch_queue()
        except LwdTransferFailedError as failure:
            return self._lwd_contain_transfer_failure(failure)

    def _lwd_contain_transfer_failure(
        self, failure: LwdTransferFailedError
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """Containment: a mid-batch data-plane failure aborts its requests
        (each cloud notified through the normal abort path, bookkeeping
        settled) and the engine keeps stepping instead of dying with the
        frontend. See docs/design/lwd_peer_restart_recovery.md."""
        request_ids = failure.req_ids
        if request_ids is None:
            request_ids = self.scheduler.lwd_edge_up_batch_request_ids(
                failure.connection, failure.seqnos
            )
        lwd_logger.error(
            "[edge] transfer failed on %s seqnos=%s: "
            "aborting %d request(s), engine continues",
            failure.connection,
            sorted(failure.seqnos),
            len(request_ids),
        )
        if request_ids:
            self.abort_requests(request_ids)
        # Empty outputs: the batch produced nothing, but the engine keeps
        # stepping. Matches the step_with_batch_queue return contract.
        return {}, False

    def _lwd_gate_reason(self, request: Request) -> str | None:
        """Admission gate: None admits, a reason rejects. The request's
        source edge comes from the SCHEDULER's routing identity
        (``_lwd_edge_id_of``: edge_share -> the attach client_index,
        standalone -> the engine's instance_id) — the same source the
        router uses, never the request id string (the ``e<id>-`` prefix
        exists only on the CLOUD side, after ``wrap_req_id``; edge-side
        ids are bare and an unparsable one defaulted to edge 0, which
        rejected every request on any edge whose instance id is not 0).
        The heartbeat monitor's ``silent_routes()`` already unions the
        frozen sweep snapshot with a LIVE staleness view: the sweep runs at
        step head, so a cloud that went silent while the engine idled must
        not admit the very request that wakes it.
        All routes silent rejects everything; a request whose source
        edge is fully silent is rejected. Elastic partial start adds
        the not-yet-adopted case: a request whose source edge has NO
        adopted cloud (the planned cloud starts later, or is still
        loading) is rejected the same way — fail fast and retryably.
        Letting it through would route it to an unaddressable
        connection and kill the shared engine."""
        edge_id = self.scheduler._lwd_edge_id_of(request)
        # Snapshot before iterating: _lwd_clouds is written by the
        # control-plane IO thread (runtime adoption) while this runs on
        # the engine thread per request; a Python-level iteration raises
        # on the concurrent insert, tuple(dict) completes in C.
        scoped = {c for c in tuple(self._lwd_clouds) if c.edge_id == edge_id}
        if not scoped:
            return (
                f"cloud unavailable (no adopted connection for edge "
                f"{edge_id}; the planned cloud has not started/registered "
                f"yet)"
            )
        # Elastic scale-out: a runtime-added connection whose data-plane
        # build was refused (or has not run) carries no transport — reject
        # fast with the real reason instead of letting routing score a
        # link that cannot carry traffic. Must sit before the silence
        # check: the silence block returns None (admit) whenever nothing
        # is silent, and a connection that is not serviceable is never a routing
        # candidate. A serviceable runtime connection falls through: its
        # data plane exists and its heartbeats flow.
        if all(
            (c.cloud_id, c.dp_idx) in self._lwd_runtime_connections
            and (c.cloud_id, c.dp_idx) not in self._lwd_serviceable_connections
            for c in scoped
        ):
            return (
                "cloud unavailable (adopted on the control plane only; the "
                "data-plane build was refused or has not run)"
            )
        silent = self._lwd_heartbeat.silent_routes()
        if not silent:
            return None
        if set(self._lwd_clouds) <= silent or (scoped and scoped <= silent):
            return f"cloud unavailable (silent routes {sorted(silent)})"
        return None

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
            if dp_idx in self._lwd_heartbeat.frozen_silent_routes():
                # A silent cloud cannot receive the notify anyway, and its
                # own bookkeeping dies with the restart/reset; skipping
                # keeps the abort (and the containment driving it) from
                # stalling behind the send retries of a dead peer.
                lwd_logger.error(
                    "[edge] abort notify skipped req=%s: %s is silent",
                    request_id,
                    dp_idx,
                )
                continue
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
        structured_req = request.structured_output_request
        reasoning_ended = (
            None if structured_req is None else structured_req.reasoning_ended
        )
        reasoning_parser_kwargs = (
            ""
            if structured_req is None
            else lwd_encode_reasoning_parser_kwargs(
                structured_req.reasoning_parser_kwargs
            )
        )
        metadata = self.scheduler.lwd_model_family.request_metadata(request)
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
                mrope_positions=self.scheduler.lwd_mrope_positions(request.request_id),
                image_placeholder_spans=metadata.image_placeholder_spans,
                structured_outputs=(
                    ""
                    if sampling_params.structured_outputs is None
                    else lwd_encode_structured_outputs(
                        sampling_params.structured_outputs
                    )
                ),
                reasoning_ended=reasoning_ended,
                reasoning_parser_kwargs=reasoning_parser_kwargs,
                **values,
            ),
        )

    def _lwd_finish_request(self, request: Request) -> None:
        self._lwd_probe_chains.pop(request.request_id, None)
