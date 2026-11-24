# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Edge-side client of a cloud DP coordinator: FIRST_REQ wake-up.

A MoE multi-DP cloud runs its engines on the native DP engine core, whose
wave coordination pauses idle engines; the cloud has no front-end traffic
of its own, so nothing would ever send the new-request message that wakes
a wave. The edge -- the cloud's front-end stand-in -- sends FIRST_REQ
(engine dp_idx, wave 0) when a request is admitted while the engines may
be paused; wave 0 is a safe stale value, the coordinator turns a stale
wave into a broadcast to every engine. Stats coming back are logged for
reconciliation only: routing decisions use the local per-DP accounts.

Two directions, two threads, so two sockets: the engine thread writes
wake-ups, the stats thread reads stats, and neither ever touches the
other's socket. A zmq socket is not thread-safe, so one shared socket
would need a lock -- and a recv loop can only wait by holding that lock
across a poll, which turns every wake-up into a wait for the poll to
expire. Under a request stream that waiting compounds: the engine thread
releases and re-acquires nothing while the stats thread takes the lock
straight back, so admission stalled for seconds per request rather than
the one poll period the arrangement promised.

The wake socket sends no subscription byte, because it never reads. The
coordinator serves both from the one XPUB endpoint and tells them apart
by payload -- `\\x01`/`\\x00` are subscriptions to ignore, anything else
is a wake-up (`vllm/v1/engine/coordinator.py`). Native reaches the same
arrangement from the other side: its front-end hands wake-ups to the
stats task over an inproc pair, so there too the stats socket has a
single thread. Here the two are kept apart instead, which also keeps the
wake-up off the stats thread's critical path.

The wire payload is ``(engine_index, wave)``, not the full
``LwdConnectionId``: the native coordinator only excludes the engine that
received the request. Several edges may wake the same cloud DP through their
own clients; all of them name the same engine index, which is the intended
identity for this protocol.
"""

from __future__ import annotations

import threading

import msgspec
import zmq

from vllm.lwd_logger import lwd_logger
from vllm.v1.lwd_control.lwd_comm.lwd_transport import lwd_apply_ctrl_qos

_LWD_STATS_POLL_MS = 50
_LWD_SHUTDOWN_JOIN_S = 1.0


class LwdCoordClient:
    """One cloud DP rank's coordinator endpoint, one socket per thread."""

    def __init__(
        self, dp_idx: int, stats_addr: str, ctrl_tos: int | None = None
    ) -> None:
        self._dp_idx = dp_idx
        self._stats_addr = stats_addr
        self._ctrl_tos = ctrl_tos
        self._closed = False
        # Engine-thread socket. The lock covers it against shutdown closing
        # it underneath a send, and nothing else: both sides are
        # non-blocking, so it is never held for longer than a send takes.
        self._wake_lock = threading.Lock()
        self._wake_socket = zmq.Context.instance().socket(zmq.XSUB)
        # Inter-node control link: carry the topology's ctrl-plane DSCP.
        lwd_apply_ctrl_qos(self._wake_socket, ctrl_tos)
        self._wake_socket.connect(stats_addr)
        self._thread = threading.Thread(
            target=self._stats_loop, name=f"lwd-coord-dp{dp_idx}", daemon=True
        )
        self._thread.start()
        lwd_logger.info("[coord] client of dp%d connected: %s", dp_idx, stats_addr)

    @property
    def dp_idx(self) -> int:
        return self._dp_idx

    def send_first_req(self) -> None:
        """Wake the engines of this coordinator's wave if they are paused;
        a running wave swallows the message guardedly."""
        if self._closed:
            return
        try:
            with self._wake_lock:
                if self._closed:
                    return
                self._wake_socket.send(
                    msgspec.msgpack.encode((self._dp_idx, 0)), flags=zmq.NOBLOCK
                )
        except zmq.ZMQError:
            lwd_logger.warning(
                "[coord] FIRST_REQ of dp%d dropped", self._dp_idx, exc_info=True
            )

    def _stats_loop(self) -> None:
        """Downstream stats (reconciliation log only): counts are the
        engines' [waiting, running], the anchor for manual comparison with
        the local per-DP accounts.

        The socket is opened and closed here, so no other thread ever
        touches it; the poll slices now only bound how long shutdown waits.
        """
        socket = zmq.Context.instance().socket(zmq.XSUB)
        # Inter-node control link: carry the topology's ctrl-plane DSCP.
        lwd_apply_ctrl_qos(socket, self._ctrl_tos)
        socket.connect(self._stats_addr)
        socket.send(b"\x01")  # subscribe to everything
        try:
            while not self._closed:
                try:
                    if not socket.poll(_LWD_STATS_POLL_MS):
                        continue
                    buf = socket.recv(zmq.NOBLOCK)
                except zmq.ZMQError:
                    if self._closed:
                        return
                    continue
                try:
                    counts, wave, running = msgspec.msgpack.decode(buf)
                except (msgspec.DecodeError, TypeError, ValueError):
                    continue
                lwd_logger.debug(
                    "[coord] dp%d stats counts=%s wave=%s engines_running=%s",
                    self._dp_idx,
                    counts,
                    wave,
                    running,
                )
        finally:
            socket.close(0)

    def shutdown(self) -> None:
        """Idempotent."""
        if self._closed:
            return
        self._closed = True
        # The stats thread owns its socket and closes it on the way out; a
        # poll slice bounds the wait.
        self._thread.join(timeout=_LWD_SHUTDOWN_JOIN_S)
        with self._wake_lock:
            self._wake_socket.close(0)
