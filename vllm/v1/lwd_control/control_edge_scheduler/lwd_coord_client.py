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
"""

from __future__ import annotations

import threading

import msgspec
import zmq

from vllm.lwd_logger import lwd_logger

_LWD_STATS_POLL_MS = 50
_LWD_SHUTDOWN_JOIN_S = 1.0


class LwdCoordClient:
    """One XSUB link to one cloud DP rank's coordinator stats endpoint."""

    def __init__(self, dp_idx: int, stats_addr: str) -> None:
        self._dp_idx = dp_idx
        self._closed = False
        # A zmq socket is not thread-safe: the stats thread recv and the
        # engine thread send serialize on this lock; the recv loop polls in
        # short slices so a FIRST_REQ waits at most one poll period.
        self._lock = threading.Lock()
        self._socket = zmq.Context.instance().socket(zmq.XSUB)
        self._socket.connect(stats_addr)
        with self._lock:
            self._socket.send(b"\x01")  # subscribe to everything
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
            with self._lock:
                self._socket.send(
                    msgspec.msgpack.encode((self._dp_idx, 0)), flags=zmq.NOBLOCK
                )
        except zmq.ZMQError:
            lwd_logger.warning(
                "[coord] FIRST_REQ of dp%d dropped", self._dp_idx, exc_info=True
            )

    def _stats_loop(self) -> None:
        """Downstream stats (reconciliation log only): counts are the
        engines' [waiting, running], the anchor for manual comparison with
        the local per-DP accounts."""
        while not self._closed:
            try:
                with self._lock:
                    if not self._socket.poll(_LWD_STATS_POLL_MS):
                        continue
                    buf = self._socket.recv(zmq.NOBLOCK)
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

    def shutdown(self) -> None:
        """Idempotent."""
        if self._closed:
            return
        self._closed = True
        with self._lock:
            self._socket.close(0)
        self._thread.join(timeout=_LWD_SHUTDOWN_JOIN_S)
