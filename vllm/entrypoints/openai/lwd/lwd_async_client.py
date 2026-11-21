# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The LWD frontend's AsyncMPClient: engine-restart self-heal layered on
the native client by INHERITANCE ONLY (vllm's native core_client.py stays
pristine).

Carried over from the old branch's primary-restart line:

* ``ROUTER_HANDOVER`` unconditionally on — a restarted engine reconnects
  with the SAME identity and must replace the dead connection's
  identity-table entry, or its requests evaporate silently behind the
  stale entry;
* init-time READY dedupe — the LWD engine re-sends READY on every DEALER
  (re)connection, so the initial send and the first connect event can
  both land during init, and the native config sync is not idempotent
  (``num_gpu_blocks`` adds);
* the output reader survives the old engine's ``ENGINE_CORE_DEAD``
  broadcast and the dead flag clears on either of two complementary
  criteria — a normal output frame (the replacement engine already
  produced output) or the READY re-send buffered on the input ROUTER,
  probed from the send path when the dead marker intercepts a request;
* a frontend-side ROUTER connection monitor (drill-log chain).

The factory wrapper (``apply_lwd_async_client_patch``) routes frontend
client construction through this class when the LWD host-comm config is
active (data_plane_comm=host only); non-LWD deployments, device-comm (HCCL)
and DP>1 shapes keep the native factory untouched.
"""

from __future__ import annotations

import asyncio
import threading
import weakref
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import zmq

from vllm.logger import init_logger
from vllm.tracing import instrument
from vllm.v1.engine import (
    EEP_NOTIFICATION_CALL_ID,
    FT_STATUS_CALL_ID,
    EngineCoreOutputs,
)
from vllm.v1.engine.core_client import (
    AsyncMPClient,
    EngineCoreClient,
    _process_utility_output,
)
from vllm.v1.engine.exceptions import EngineDeadError

logger = init_logger(__name__)

Thread = threading.Thread


class LwdAsyncMPClient(AsyncMPClient):
    """AsyncMPClient with the LWD frontend self-heal (see the module doc);
    a full native-preserved subclass: every new member is LWD's, every
    changed method is overridden here."""

    def __init__(
        self,
        vllm_config: Any,
        executor_class: Any,
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
        renderer: Any | None = None,
    ):
        # Ready-response dedupe state must exist BEFORE super().__init__:
        # the engine's READY re-send can land inside the init collection.
        self._lwd_applied_ready: bytes | None = None
        super().__init__(
            vllm_config,
            executor_class,
            log_stats,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
            renderer=renderer,
        )
        # A restarted engine reconnects with the SAME identity: the input
        # ROUTER needs handover to let the new engine replace the dead
        # connection's identity-table entry — without it the new DEALER
        # connects but stays unaddressable behind the stale entry and
        # requests evaporate silently. Set post-bind: handover only takes
        # effect when a duplicate identity actually arrives.
        self.input_socket.setsockopt(zmq.ROUTER_HANDOVER, 1)
        if self.resources.engine_manager is None:
            # Attach frontend (engines managed externally): watch the
            # engine DEALER's connect/disconnect for the drill-log chain.
            self._lwd_start_router_monitor()

    # ------------------------------------------------------------------ #
    # READY dedupe (the engine re-sends READY on every (re)connection)    #
    # ------------------------------------------------------------------ #

    def _apply_ready_response(self, payload: bytes) -> None:
        if payload and payload == self._lwd_applied_ready:
            # The engine re-sends READY on every DEALER (re)connection, so
            # the initial send and the first connect event can both land
            # here during init; the native config sync is not idempotent
            # (num_gpu_blocks adds), so the duplicate is dropped instead of
            # applied twice.
            return
        self._lwd_applied_ready = payload
        super()._apply_ready_response(payload)

    # ------------------------------------------------------------------ #
    # Engine revival: send-path READY probe                               #
    # ------------------------------------------------------------------ #

    def ensure_alive(self):
        if self.resources.engine_dead and self._maybe_revived():
            # Revival criterion #2 cleared the dead flag; let this send
            # through (a replacement engine has taken over).
            pass
        if self.resources.engine_dead:
            raise EngineDeadError()

    def _maybe_revived(self) -> bool:
        """Probe the input ROUTER for a replacement engine's READY.

        After the old engine's DEAD broadcast nothing reaches the input
        ROUTER but the READY that every engine (re)connection re-sends
        (engine replies travel on the output socket), so a buffered frame
        here means a new engine has connected and its READY went unread —
        the client never receives on the input socket after init. The
        reader loop uses a normal output frame as the complementary
        criterion (see ``_ensure_output_queue_task``); whichever fires
        first revives the client.
        """
        sock = self.resources.input_socket
        if sock is None:
            return False
        try:
            # Shadow so the probe works for the asyncio socket as well
            # (asyncio recv returns futures); all calls are non-blocking.
            probe = zmq.Socket.shadow(sock)
            if not probe.poll(0):
                return False
            while probe.poll(0):
                probe.recv_multipart(flags=zmq.NOBLOCK)
        except Exception:
            return False
        self.resources.engine_dead = False
        logger.error(
            "[Core-client] engine revived (READY probe on input socket): "
            "request flow resumed (main command restarted)"
        )
        return True

    # ------------------------------------------------------------------ #
    # Output reader: DEAD survival + revival criterion #1                #
    # ------------------------------------------------------------------ #

    def _ensure_output_queue_task(self):
        resources = self.resources
        if resources.output_queue_task is not None:
            return

        # Perform IO in separate task to parallelize as much as possible.
        # Avoid task having direct reference back to the client.
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue
        output_handler: (
            Callable[[AsyncMPClient, EngineCoreOutputs], Awaitable[None]] | None
        ) = getattr(self.__class__, "process_engine_outputs", None)
        _self_ref = weakref.ref(self)
        output_socket = resources.output_socket
        assert output_socket is not None

        notification_callback_handler: (
            Callable[[AsyncMPClient, Sequence[Any]], Any] | None
        ) = getattr(self.__class__, "eep_process_engine_core_notification", None)

        async def process_outputs_socket():
            try:
                while True:
                    frames = await output_socket.recv_multipart(copy=False)
                    try:
                        resources.validate_alive(frames)
                    except EngineDeadError:
                        # Engine death (the old engine's DEAD broadcast on a
                        # primary restart) must not end the reader: the
                        # frontend stays up and a later normal output frame
                        # from the replacement engine clears the flag. Exiting
                        # here would leave every subsequent output unread and
                        # hang requests with no error anywhere.
                        logger.error(
                            "[Core-client] engine DEAD received; reader kept "
                            "alive awaiting revival (main command restarted?)"
                        )
                        continue
                    if resources.engine_dead:
                        # Revival criterion #1: a normal output frame arriving
                        # while the dead flag is set means a replacement
                        # engine has taken over; error level so one restart
                        # is grep-able end to end.
                        resources.engine_dead = False
                        logger.error(
                            "[Core-client] engine revived (normal output "
                            "frame after DEAD): request flow resumed "
                            "(main command restarted)"
                        )
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    if outputs.utility_output:
                        if (
                            outputs.utility_output.call_id == EEP_NOTIFICATION_CALL_ID
                            and notification_callback_handler is not None
                        ):
                            assert _self_ref is not None
                            _self = _self_ref()
                            if not _self:
                                return
                            if outputs.utility_output.result is None:
                                continue
                            notification_data = outputs.utility_output.result.result
                            assert isinstance(notification_data, Sequence)
                            assert len(notification_data) == 2
                            asyncio.create_task(
                                notification_callback_handler(_self, notification_data)
                            )
                        elif outputs.utility_output.call_id == FT_STATUS_CALL_ID:
                            _self = _self_ref()
                            if not _self:
                                return
                            if outputs.utility_output.result is not None:
                                _self._engine_status[outputs.engine_index] = (
                                    outputs.utility_output.result.result
                                )
                        else:
                            _process_utility_output(
                                outputs.utility_output, utility_results
                            )
                        continue

                    if output_handler is not None:
                        assert _self_ref is not None
                        _self = _self_ref()
                        if not _self:
                            # Client has been garbage collected, abort.
                            return
                        await output_handler(_self, outputs)

                    if outputs.outputs or outputs.scheduler_stats:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)
            except asyncio.CancelledError:
                outputs_queue.put_nowait(EngineDeadError())

        resources.output_queue_task = asyncio.create_task(
            process_outputs_socket(), name="EngineCoreOutputQueueTask"
        )

    # ------------------------------------------------------------------ #
    # Router connection monitor (drill-log chain)                          #
    # ------------------------------------------------------------------ #

    def _lwd_start_router_monitor(self) -> None:
        # The monitor socket of an asyncio client is asyncio-flavoured, so
        # shadow it for the plain probe thread; probe setup failure must
        # not break the client.
        try:
            input_address = self.input_socket.getsockopt(zmq.LAST_ENDPOINT).decode()
        except (AttributeError, zmq.ZMQError):
            input_address = "input ROUTER"
        try:
            router_monitor = zmq.Socket.shadow(
                self.input_socket.get_monitor_socket(
                    zmq.EVENT_ACCEPTED | zmq.EVENT_DISCONNECTED
                )
            )

            def log_router_events(monitor=router_monitor) -> None:
                try:
                    while True:
                        if not monitor.poll(60000):
                            continue
                        frames = monitor.recv_multipart()
                        event = int.from_bytes(bytes(frames[0])[:2], "little")
                        if event == zmq.EVENT_ACCEPTED:
                            logger.error(
                                "[Core-client] input ROUTER accepted "
                                "engine connection at %s (engine can "
                                "receive requests)",
                                input_address,
                            )
                        elif event == zmq.EVENT_DISCONNECTED:
                            logger.error(
                                "[Core-client] input ROUTER lost "
                                "engine connection at %s (engine "
                                "died/restarting?)",
                                input_address,
                            )
                except zmq.ZMQError:
                    return  # socket closed (client shutdown)

            Thread(
                target=log_router_events,
                daemon=True,
                name="core-client-router-monitor",
            ).start()
        except Exception:  # noqa: BLE001 - probe must not break the client
            logger.exception("[Core-client] input ROUTER monitor setup failed")


# ---------------------------------------------------------------------- #
# Factory routing: native for non-LWD, LwdAsyncMPClient for LWD DP=1       #
# ---------------------------------------------------------------------- #

_applied = False


def apply_lwd_async_client_patch() -> None:
    """Route frontend engine-client construction through
    ``LwdAsyncMPClient`` when the LWD config is active.

    Idempotent; the wrapper delegates to the NATIVE factory for non-LWD
    deployments and DP>1 shapes, so applying it is always safe. Called by
    the ascend platform's config pass — the one LWD-reachable code point
    that every frontend process traverses before constructing the client
    (frontend workers import no LWD module of their own).
    """
    global _applied
    if _applied:
        return
    _applied = True
    native = EngineCoreClient.make_async_mp_client

    @instrument(span_name="Overall Loading")
    def _lwd_make_async_mp_client(*args: Any, **kwargs: Any):
        vllm_config = kwargs.get("vllm_config") or (args[0] if args else None)
        if (
            getattr(vllm_config, "lwd_config", None) is None
            or vllm_config.parallel_config.data_parallel_size > 1
            # Scenario gate: the LWD frontend self-heal (READY dedupe,
            # ROUTER handover, engine revival) exists for the host-comm
            # plane (data_plane_comm=host, UCX host-RoCE). Device-comm
            # (HCCL) deployments keep the native AsyncMPClient untouched.
            or not vllm_config.lwd_config.topology.is_host_comm
        ):
            return native(*args, **kwargs)
        return LwdAsyncMPClient(*args, **kwargs)

    EngineCoreClient.make_async_mp_client = staticmethod(_lwd_make_async_mp_client)
