# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Common base of the LWD edge and cloud engine cores: scheduler injection,
the control plane of the DP rank and the hand-over of its IO thread events to
the engine thread."""

from __future__ import annotations

import os
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import msgspec
import zmq

from vllm.config.lwd import (
    LwdTopology,
    LwdTopologyIncrement,
    LwdTopologyUpdateRejected,
    diff_topology,
)
from vllm.lwd_logger import lwd_logger
from vllm.multimodal.cache import MultiModalCacheMissError
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.fault_tolerance.engine_core_sentinel import FT_UTILITY_METHOD
from vllm.v1.lwd_control.lwd_comm.lwd_control_plane import (
    LwdControlPlane,
    LwdControlSender,
)
from vllm.v1.lwd_control.lwd_comm.lwd_control_plane_factory import (
    lwd_create_control_plane,
)
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdPeer, LwdRunnerKind
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LwdMessage,
    lwd_encode_message,
)
from vllm.v1.serial_utils import MsgpackDecoder

# Topology-file poll cadence (host-comm elastic scale-out): the shared
# side of a scene re-reads the topology when the file's mtime moves, so
# instances added to the plan are picked up without an engine restart.
# The stat is one syscall per interval and rides no other loop (the
# engine thread blocks in input_queue between steps), so it gets its
# own thread.
_LWD_TOPOLOGY_POLL_INTERVAL_S = 1.0


class _LwdTopologyReloadTick:
    """Input-queue sentinel of the topology-file watcher thread: the
    engine thread (the single writer) re-reads, validates and diffs the
    file, then applies its half of the increment."""


_LWD_TOPOLOGY_RELOAD_TICK = _LwdTopologyReloadTick()


@dataclass(frozen=True)
class _LwdPeerUp:
    """`WAKEUP` payload: `peer` completed the control plane handshake."""

    peer: LwdPeer


@dataclass(frozen=True)
class _LwdControlPlaneFailed:
    """`EXECUTOR_FAILED` payload naming the control plane as the cause."""

    reason: str


class _LwdEngineListener:
    """Control plane callbacks (IO thread), forwarded to the engine."""

    def __init__(self, engine: LwdBaseEngineCore) -> None:
        self._engine = engine

    def on_peer_up(self, peer: LwdPeer) -> None:
        # The sender belongs to the engine thread, so registration is handled
        # there; input_queue FIFO puts it before any request from `peer`.
        self._engine.input_queue.put_nowait(
            (EngineCoreRequestType.WAKEUP, _LwdPeerUp(peer))
        )

    def on_message(self, peer: LwdPeer, payload: bytes) -> None:
        try:
            self._engine._lwd_on_payload(peer, payload)
        except Exception as e:
            # Keep the IO thread alive; the engine fails on its own thread.
            lwd_logger.exception("[comm] handling a payload from %s failed", peer)
            self._engine._lwd_fail(f"handling a payload from {peer}: {e!r}")

    def on_fatal(self, reason: str) -> None:
        # The plane has logged the crash.
        self._engine._lwd_fail(reason)


class LwdBaseEngineCore(EngineCoreProc):
    """EngineCoreProc of an LWD edge or cloud.

    Subclasses declare `lwd_scheduler_cls`, injected before `super()`, which
    builds the scheduler from `scheduler_cls` once. Upstream disables chunked
    prefill for deployments without KV cache groups, while LWD edge chunks are
    pure scheduling bookkeeping, so it is restored for LWD engines afterwards.
    `_lwd_assemble` then sets up the role and starts the control plane; READY
    has already reached the frontend by then.
    """

    lwd_scheduler_cls: type

    def __init__(self, *args: Any, lwd_dp_size: int, **kwargs: Any) -> None:
        self._lwd_dp_size = lwd_dp_size
        self._lwd_control_plane: LwdControlPlane | None = None
        self._lwd_sender: LwdControlSender | None = None
        # has_work state at the last transition log; None = not logged yet.
        self._lwd_last_has_work_state: tuple[bool, bool, bool] | None = None
        # The topology-file watch stop event predates everything fallible:
        # shutdown reaches it after a failed assembly too.
        self._lwd_topology_watch_stop = threading.Event()
        vllm_config = kwargs["vllm_config"]
        scheduler_config = vllm_config.scheduler_config
        scheduler_config.scheduler_cls = self.lwd_scheduler_cls
        chunked_prefill_wanted = scheduler_config.enable_chunked_prefill
        super().__init__(*args, **kwargs)
        if not scheduler_config.enable_chunked_prefill and chunked_prefill_wanted:
            scheduler_config.enable_chunked_prefill = True
        self.lwd_config = vllm_config.lwd_config
        try:
            self._lwd_assemble()
        except BaseException:
            # run_engine_core never gets this engine to shut down, so release
            # the executor and the control plane before failing startup.
            self.shutdown()
            raise
        # Elastic scale-out, after a successful assembly: the shared side of
        # the scene watches its plan (see _lwd_topology_watch_enabled).
        if self._lwd_topology_watch_enabled():
            self._lwd_start_topology_watch()
        else:
            # Diagnosis anchor: a host-comm lwd_cluster deployment must land
            # in the OTHER branch on both roles. Anything unexpected here
            # means the gate read a different scene/comm than the
            # deployment intends, and every later "why didn't the edge see
            # the new cloud" question starts at this line.
            lwd_logger.info(
                "[LWD] topology watch NOT started: role=%s scene=%s host_comm=%s",
                self.lwd_config.role,
                self.lwd_config.topology.deployment.scene,
                self.lwd_config.topology.is_host_comm,
            )

    def _lwd_assemble(self) -> None:
        """Role setup once the engine is built, including the control plane."""
        raise NotImplementedError

    def _lwd_topology_watch_enabled(self) -> bool:
        """The shared side of a scene watches its plan: the edge engine in
        edge_share (clouds are added to the running shared engine), the
        cloud engine in cloud_share (edges are added to the running shared
        cloud). In lwd_cluster BOTH sides watch — each edge engine detects
        new clouds to connect to, each cloud engine detects new edges to
        accept; each applies only the connections of its own instance
        (see the apply hooks). Device comm never watches — the HCCL path
        stays untouched."""
        scene = self.lwd_config.topology.deployment.scene
        if scene == "lwd_cluster":
            return self.lwd_config.topology.is_host_comm
        watched = "edge_share" if self.lwd_config.role == "edge" else "cloud_share"
        return self.lwd_config.topology.is_host_comm and scene == watched

    def _lwd_start_topology_watch(self) -> None:
        """Start the mtime poll thread: it owns nothing but the stat and
        the input-queue tick; the engine thread (the single writer)
        re-reads, validates and diffs on receipt."""
        self._lwd_topology_file_digest = self.lwd_config.topology.digest
        try:
            self._lwd_topology_mtime_ns: int | None = os.stat(
                self.lwd_config.path
            ).st_mtime_ns
        except OSError:
            self._lwd_topology_mtime_ns = None
        threading.Thread(
            target=self._lwd_watch_topology_file,
            name="lwd-topology-watch",
            daemon=True,
        ).start()
        lwd_logger.info(
            "[LWD] topology file watch up: %s (every %.1fs)",
            self.lwd_config.path,
            _LWD_TOPOLOGY_POLL_INTERVAL_S,
        )

    def _lwd_watch_topology_file(self) -> None:
        """Poll the topology file's mtime; a change queues one reload tick
        for the engine thread. Transient stat failures (the atomic-rename
        window) retry on the next tick."""
        path = self.lwd_config.path
        while not self._lwd_topology_watch_stop.wait(_LWD_TOPOLOGY_POLL_INTERVAL_S):
            try:
                mtime_ns = os.stat(path).st_mtime_ns
            except OSError:
                continue
            if mtime_ns != self._lwd_topology_mtime_ns:
                self._lwd_topology_mtime_ns = mtime_ns
                self.input_queue.put_nowait(
                    (EngineCoreRequestType.WAKEUP, _LWD_TOPOLOGY_RELOAD_TICK)
                )

    def _lwd_process_topology_reload(
        self,
    ) -> LwdTopologyIncrement | LwdTopologyUpdateRejected | None:
        """Detect-only core, engine thread: re-read + validate + diff the
        topology file, then hand an accepted increment to the role hook.
        A rejected or unparsable file never touches the realized snapshot
        and serving continues on it. The last-seen file digest suppresses
        repeat verdicts for unchanged content (mtime touches included)."""
        try:
            candidate = LwdTopology.from_file(self.lwd_config.path)
        except ValueError as exc:
            lwd_logger.error("[LWD] topology update rejected: invalid file: %s", exc)
            return LwdTopologyUpdateRejected(reason=f"invalid file: {exc}")
        except OSError as exc:
            # The atomic-replace window (or a transient FS error): the file
            # vanished between the watcher's stat and this read. Serving
            # continues on the realized snapshot; the next tick retries.
            lwd_logger.error("[LWD] topology update rejected: unreadable file: %s", exc)
            return LwdTopologyUpdateRejected(reason=f"unreadable file: {exc}")
        if candidate.digest == self._lwd_topology_file_digest:
            # The file moved (mtime ticked) but its canonical content is
            # identical — comments/formatting only. The watcher IS alive;
            # there is simply nothing to apply.
            lwd_logger.error(
                "[LWD] topology file changed but the canonical digest is "
                "identical; nothing to apply"
            )
            return None
        self._lwd_topology_file_digest = candidate.digest
        verdict = diff_topology(self.lwd_config.topology, candidate)
        if isinstance(verdict, LwdTopologyUpdateRejected):
            lwd_logger.error("[LWD] topology update rejected: %s", verdict.reason)
            return verdict
        if not verdict.new_connections:
            # The file's semantics moved (e.g. a tunable-only change) but
            # nothing this engine can apply arrived.
            lwd_logger.error(
                "[LWD] topology update accepted with no new connections: "
                "new_edge_ids=%s new_cloud_ids=%s",
                list(verdict.new_edge_ids),
                list(verdict.new_cloud_ids),
            )
            return None
        lwd_logger.error(
            "[LWD] topology increment detected: +%d edge(s) %s, +%d cloud(s) "
            "%s, %d connection(s)",
            len(verdict.new_edge_ids),
            list(verdict.new_edge_ids),
            len(verdict.new_cloud_ids),
            list(verdict.new_cloud_ids),
            len(verdict.new_connections),
        )
        self._lwd_apply_topology_increment(verdict)
        return verdict

    def _lwd_apply_topology_increment(self, increment: LwdTopologyIncrement) -> None:
        """Role hook: apply the half of the increment this engine owns. The
        default only records the detection — the shared edge builds the
        control-plane and data-plane links, and the shared cloud grows its
        allowlist and listener, each in its override."""
        lwd_logger.info(
            "[LWD] topology increment recorded (role=%s has no runtime apply yet)",
            self.lwd_config.role,
        )

    def _lwd_start_control_plane(self) -> LwdControlPlane:
        """Start this DP rank's control plane; the calling (engine) thread
        owns its sender."""
        runner_kind: LwdRunnerKind = (
            "v2" if self.vllm_config.use_v2_model_runner else "v1"
        )
        worker_kinds = set(self.model_executor.collective_rpc("lwd_model_runner_kind"))
        if worker_kinds != {runner_kind}:
            raise RuntimeError(
                f"[LWD] model runner mismatch: engine={runner_kind}, "
                f"workers={sorted(worker_kinds)}"
            )
        dp_rank = self.vllm_config.parallel_config.data_parallel_index
        plane = lwd_create_control_plane(
            self.lwd_config, dp_rank, self._lwd_dp_size, runner_kind=runner_kind
        )
        self._lwd_control_plane = plane
        self._lwd_sender = plane.start(_LwdEngineListener(self))
        lwd_logger.info(
            "[comm] control plane dp%d/%d started on %s",
            dp_rank,
            self._lwd_dp_size,
            plane.endpoint,
        )
        return plane

    def _lwd_send(self, peer: LwdPeer, message: LwdMessage) -> bool:
        """Queue `message` for `peer`; False when the outbound queue is full."""
        assert self._lwd_sender is not None, "control plane not started"
        return self._lwd_sender.send(peer, lwd_encode_message(message))

    def _lwd_fail(self, reason: str) -> None:
        """Any thread: fail the engine on the engine thread, where
        run_engine_core logs the failure. Callers log the cause."""
        self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, _LwdControlPlaneFailed(reason))
        )

    # ------------------------------------------------------------------ #
    # Frontend input thread (full override of the native one)            #
    # ------------------------------------------------------------------ #
    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ):
        """Input socket IO thread, LWD override.

        Everything the native thread does, plus the frontend-restart
        self-heal (the engine re-sends READY on every DEALER
        (re)connection — READY is the running engine's only visibility
        into a frontend that restarted or started late), the drill-log
        chain (address self-certification, three-state connection
        events, rate-limited frame probes) and a death containment: this
        is a daemon thread, so an uncaught exception would otherwise
        kill it silently and every frontend would hang with no error
        anywhere. Kept as a full method override so vllm's native
        core.py stays pristine.

        Scenario gate: the override exists for the host-comm deployment
        (data_plane_comm=host, UCX host-RoCE). With data_plane_comm=device
        (HCCL) the native input thread stays untouched — defer straight
        to the parent so the device plane runs the stock vLLM frontend
        path.
        """
        # Read the config off vllm_config, NOT self.lwd_config: this thread
        # starts INSIDE super().__init__(), before this subclass assigns
        # self.lwd_config a few statements after the super call — the
        # thread can reach this gate first (startup race, AttributeError
        # on the drill machine). self.vllm_config is set before the thread
        # starts, and lwd_config is always present on an LWD engine.
        lwd_config = self.vllm_config.lwd_config
        assert lwd_config is not None
        if not lwd_config.topology.is_host_comm:
            return super().process_input_sockets(
                input_addresses, coord_input_address, identity, ready_event
            )

        # Msgpack serialization decoding with optional tensor IPC receiver.
        add_request_decoder = MsgpackDecoder(
            EngineCoreRequest, oob_tensor_provider=self.tensor_ipc_receiver
        )
        generic_decoder = MsgpackDecoder(oob_tensor_provider=self.tensor_ipc_receiver)

        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False
                    )
                )
                for input_address in input_addresses
            ]
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(
                        ctx,
                        coord_input_address,
                        zmq.XSUB,
                        identity=identity,
                        bind=False,
                    )
                )
                # Send subscription message to coordinator.
                coord_socket.send(b"\x01")

            # Register sockets with poller.
            poller = zmq.Poller()
            ready_response = self._make_ready_response()
            ready_payload = msgspec.msgpack.encode(ready_response)
            # First-hand evidence for frontend-restart drills: the exact
            # addresses this input thread actually holds. A wrong, stale or
            # placeholder address surfaces here instead of as silently
            # evaporating requests.
            lwd_logger.error(
                "[Core] input thread up: %d frontend(s), identity=%s, addresses=%s",
                len(input_sockets),
                identity.hex(),
                list(input_addresses),
            )
            # A frontend restarted at runtime rebinds its input ROUTER (the
            # deterministic engine IO port) and its fresh MPClient waits for
            # this engine's READY — but READY is sent ONCE, below. The DEALER
            # reconnect is the engine's only visibility into the frontend's
            # return, so every (re)connection re-sends it. An already-settled
            # frontend leaves the extra frame in its ROUTER queue, unread and
            # harmless; the LWD client dedupes it during init. The
            # disconnect/retry events complete the connection-layer evidence
            # for restart drills (a message that never arrives while the
            # address is correct means the connection itself never came up).
            monitor_events = (
                zmq.EVENT_CONNECTED | zmq.EVENT_CONNECT_RETRIED | zmq.EVENT_DISCONNECTED
            )
            ready_monitors: dict[zmq.Socket, tuple[zmq.Socket, str]] = {}
            for input_socket, input_address in zip(input_sockets, input_addresses):
                # Send initial message to each input socket - this is required
                # before the front-end ROUTER socket can send input messages
                # back to us.
                input_socket.send(ready_payload)
                monitor = stack.enter_context(
                    input_socket.get_monitor_socket(monitor_events)
                )
                ready_monitors[monitor] = (input_socket, input_address)
                poller.register(monitor, zmq.POLLIN)
                poller.register(input_socket, zmq.POLLIN)

            if coord_socket is not None:
                # Wait for ready message from coordinator.
                assert coord_socket.recv() == b"READY"
                poller.register(coord_socket, zmq.POLLIN)

            ready_event.set()
            del ready_event
            input_address_of = dict(zip(input_sockets, input_addresses))
            frame_trace_interval_s = 5.0
            frame_trace_last = 0.0
            retry_log_interval_s = 60.0
            retry_log_last: dict[zmq.Socket, float] = {}
            # Logging discipline: a socket that NEVER connected is an
            # unclaimed capacity slot of the reserved frontend band — its
            # steady-state retrying is not news (the startup
            # "input thread up" line already self-certifies the address
            # list), so it stays at debug. A socket that connected once
            # and now retries means a REAL frontend vanished — a rare
            # event, error-worthy and part of the restart drill chain.
            ever_connected: set[zmq.Socket] = set()
            while True:
                # The input thread is a daemon: an uncaught exception would
                # kill it silently and every frontend request would hang
                # with no error anywhere. Contain it as a visible engine
                # failure instead.
                try:
                    for polled, _ in poller.poll():
                        now = time.monotonic()
                        if now - frame_trace_last >= frame_trace_interval_s:
                            frame_trace_last = now
                            received_address = input_address_of.get(polled)
                            if received_address is not None:
                                lwd_logger.debug(
                                    "[Core] input frame received on %s",
                                    received_address,
                                )
                        target = ready_monitors.get(polled)
                        if target is not None:
                            # Monitor frames are (event|value, endpoint).
                            frames = polled.recv_multipart()
                            event = int.from_bytes(bytes(frames[0])[:2], "little")
                            input_socket, input_address = target
                            if event == zmq.EVENT_CONNECTED:
                                ever_connected.add(input_socket)
                                input_socket.send(ready_payload)
                                lwd_logger.error(
                                    "[Core] input DEALER connected to %s; "
                                    "READY sent (frontend restart self-heal)",
                                    input_address,
                                )
                            elif event == zmq.EVENT_DISCONNECTED:
                                lwd_logger.error(
                                    "[Core] input DEALER disconnected from "
                                    "%s (frontend gone; awaiting its return)",
                                    input_address,
                                )
                            elif (
                                now - retry_log_last.get(polled, 0.0)
                                >= retry_log_interval_s
                            ):
                                retry_log_last[polled] = now
                                if input_socket in ever_connected:
                                    lwd_logger.error(
                                        "[Core] input DEALER connection to "
                                        "%s still retrying; frontend not "
                                        "bound or unreachable, its ROUTER "
                                        "drops the messages",
                                        input_address,
                                    )
                                else:
                                    lwd_logger.debug(
                                        "[Core] input DEALER to %s still "
                                        "waiting (unclaimed capacity slot)",
                                        input_address,
                                    )
                            continue
                        # (RequestType, RequestData)
                        type_frame, *data_frames = polled.recv_multipart(copy=False)
                        # NOTE(yongji): ignore READY message sent by DP coordinator
                        # that is used to notify newly started engines
                        if type_frame.buffer == b"READY":
                            assert polled == coord_socket
                            continue
                        request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                        # Deserialize the request data.
                        request: Any
                        if request_type == EngineCoreRequestType.ADD:
                            req: EngineCoreRequest = add_request_decoder.decode(
                                data_frames
                            )
                            lwd_logger.debug(
                                "[Core] input ADD req=%s client=%d",
                                req.request_id,
                                req.client_index,
                            )
                            try:
                                request = self.preprocess_add_request(req)
                            except MultiModalCacheMissError as e:
                                # P0/P1 shadow drift -- return a retryable signal (P0
                                # drops the stale entry, client resends with data).
                                self._handle_mm_cache_miss(req, e)
                                continue
                            except Exception:
                                self._handle_request_preproc_error(req)
                                continue
                        elif request_type == EngineCoreRequestType.UTILITY:
                            request = generic_decoder.decode(data_frames)
                            client_idx, call_id, method, args = request
                            if method == FT_UTILITY_METHOD:
                                self.ft_sentinel.handle_command(
                                    client_idx, call_id, args[0]
                                )
                                continue
                        else:
                            request = generic_decoder.decode(data_frames)

                            if request_type == EngineCoreRequestType.ABORT:
                                # Aborts are added to *both* queues, allowing
                                # eager processing while keeping input-queue
                                # ordering (no leaked requests); aborting in
                                # the scheduler is idempotent.
                                self.aborts_queue.put_nowait(request)

                        # Push to input queue for core busy loop.
                        self.input_queue.put_nowait((request_type, request))
                except Exception:
                    lwd_logger.exception(
                        "[Core] input socket IO thread died; escalating "
                        "engine failure (frontends would otherwise hang "
                        "silently)"
                    )
                    self.input_queue.put_nowait(
                        (EngineCoreRequestType.EXECUTOR_FAILED, b"")
                    )
                    return

    # ------------------------------------------------------------------ #
    # IO thread message entry                                            #
    # ------------------------------------------------------------------ #
    def _lwd_on_payload(self, peer: LwdPeer, payload: bytes) -> None:
        """Common entry of received payloads: `_lwd_dispatch_message` handles
        them; malformed payloads and unknown types are dropped here with a
        warning."""
        try:
            msg = self._lwd_decode(payload)
        except (msgspec.DecodeError, msgspec.ValidationError) as e:
            lwd_logger.warning("[comm] drop malformed payload from %s: %s", peer, e)
            return
        if not self._lwd_dispatch_message(peer, msg):
            lwd_logger.warning(
                "[comm] drop unexpected frame %r from %s", type(msg), peer
            )

    def _lwd_decode(self, payload: bytes) -> LwdMessage:
        """Decode a payload sent by the peer role."""
        raise NotImplementedError

    def _lwd_dispatch_message(self, peer: LwdPeer, msg: LwdMessage) -> bool:
        """IO thread; must not block. True = handled, False = unknown."""
        raise NotImplementedError

    def _lwd_handle_peer_registered(self, peer: LwdPeer) -> None:
        """Engine thread: `peer` completed the handshake."""

    # ------------------------------------------------------------------ #
    # Engine interface overrides                                         #
    # ------------------------------------------------------------------ #
    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        if request_type == EngineCoreRequestType.WAKEUP and isinstance(
            request, _LwdTopologyReloadTick
        ):
            # Engine thread (the single writer): detect-only reload of the
            # topology file (elastic scale-out), then the role's apply hook.
            self._lwd_process_topology_reload()
            return
        if request_type == EngineCoreRequestType.WAKEUP and isinstance(
            request, _LwdPeerUp
        ):
            self._lwd_handle_peer_registered(request.peer)
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED and isinstance(
            request, _LwdControlPlaneFailed
        ):
            raise RuntimeError(f"[LWD] control plane failed: {request.reason}")
        else:
            super()._handle_client_request(request_type, request)

    def has_work(self) -> bool:
        """Native three-source check, logging its transitions (lwd_logger,
        level VLLM_LWD_LOG_LEVEL, silent by default)."""
        state = (
            self.engines_running,
            self.scheduler.has_requests(),
            bool(self.batch_queue),
        )
        if state != self._lwd_last_has_work_state:
            lwd_logger.info(
                "[has_work] engines_running=%s has_requests=%s batch_queue=%s",
                *state,
            )
            self._lwd_last_has_work_state = state
        return any(state)

    def shutdown(self) -> None:
        # Also reached after a failed assembly, before the plane exists.
        self._lwd_topology_watch_stop.set()
        if self._lwd_control_plane is not None:
            self._lwd_control_plane.shutdown()
        super().shutdown()
