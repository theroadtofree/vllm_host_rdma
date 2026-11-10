# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LWD UCX registered buffer pool (a standalone piece of the wire data plane
block pool).

Pure refactor with no behaviour change:
  * LwdUcxBlock     -- one recv/send buffer: a plain bytearray that UCP
                 rcache registers implicitly and caches on the first
                 send/recv touching that address (the pool is long-lived, so
                 rcache holds exactly one entry per block; same after
                 tiering);
  * LwdUcxBlockPool -- single-size equal-sized block pool: free-queue
                 turnover plus a wholesale reset on link change.

Wire-layer semantics are unchanged:
  * blocks are allocated once, at wire construction time; the free pool
    starts empty and is filled wholesale on the first adopt/reset;
  * the wait/progress-drive policy on acquire still belongs to wire
    (_acquire_block); the pool only offers a non-blocking get with a
    timeout, returning None on timeout;
  * reset means refilling every block on a link change (old-link in-flight
    state is settled by wire's _fail_all_pending; this only handles buffer
    turnover);
  * memory cost is n_blocks x block_bytes per wire.

Threading keeps wire's current contract: acquire/release/reset run only on
the owning wire's IO thread (except at construction), so the lock-free free
queue is safe.

Future tiering extension points (comments only, not implemented in this
file; see the design analysis):
  * send side: the sender knows the frame size up front, so acquire can
    pick the smallest sufficient tier -- tier sizes recommended as
    multiples of 2MB (page-aligned, keeps rcache regions cleanly merged);
  * recv side: wildcard pre-posted recv requires every posted block to be
    >= the largest possible frame (otherwise truncation breaks the link),
    so tiering must change the delivery model: high tag bits route tiers
    (the receiver pre-posts per tier) or put slots are shipped per tier
    (the GPU version's recv_sizes[] is the ready-made template);
  * tier sizing evidence: the nbytes distribution of the LARGE/SMALL
    buckets in the sendlat log, not a guess.
"""

from __future__ import annotations

import ctypes
import queue
from typing import Any

# --------------------------------------------------------------------- #
# Block pool defaults (wire constructor defaults; the assembly layer can #
# override them via its constructor args and does the env parsing).      #
# --------------------------------------------------------------------- #
_DEFAULT_BLOCK_MB = 128
_DEFAULT_N_BLOCKS = 4


class LwdUcxBlock:
    """One recv/send buffer (a plain bytearray; UCP rcache registers and
    caches it on first use)."""

    __slots__ = (
        "buf",
        "addr",
        "index",
        "future",
        "event",
        "recv_req",
        "send_req",
        "t_post",
        "t_enq",
        "t_disp",
        "nbytes",
    )

    def __init__(self, index: int, block_bytes: int) -> None:
        self.buf = bytearray(block_bytes)
        self.addr = ctypes.addressof(
            (ctypes.c_char * block_bytes).from_buffer(self.buf)
        )
        self.index = index
        self.future: Any = None
        self.event: Any = None
        self.recv_req = 0
        self.send_req = 0
        self.t_post = 0.0
        self.t_enq = 0.0  # enqueue time (set by the submit_send caller thread)
        self.t_disp = 0.0  # dispatch start (IO thread)
        self.nbytes = 0  # frame bytes (for latency sampling)


class LwdUcxBlockPool:
    """Single-size equal-sized block pool: private to the IO thread
    (lock-free), turning buffers over through a free queue.

    n_blocks/block_bytes match wire's constructor parameters; blocks are
    allocated at construction, and the free pool starts empty, filled by
    reset().
    """

    def __init__(self, n_blocks: int, block_bytes: int) -> None:
        self.block_bytes = block_bytes
        self.blocks: list[LwdUcxBlock] = [
            LwdUcxBlock(i, block_bytes) for i in range(n_blocks)
        ]
        self._free: queue.Queue[LwdUcxBlock] = queue.Queue()

    def __len__(self) -> int:
        return len(self.blocks)

    def __iter__(self):
        return iter(self.blocks)

    def qsize(self) -> int:
        """Free block count (stall diagnostic: qsize / in flight / total)."""
        return self._free.qsize()

    def get(self, timeout: float) -> LwdUcxBlock | None:
        """Take one free buffer; on timeout return None. The waiting policy
        belongs to the caller."""
        try:
            return self._free.get(timeout=timeout)
        except queue.Empty:
            return None

    def release(self, blk: LwdUcxBlock) -> None:
        """Return a block: the shared entry point for send-completion
        callbacks, failure paths and link-change fallback."""
        self._free.put(blk)

    def reset(self) -> None:
        """Link-change reset: refill the free pool with all blocks (the blocks
        themselves are reused across links, never reallocated)."""
        self._free = queue.Queue()
        for blk in self.blocks:
            self._free.put(blk)
