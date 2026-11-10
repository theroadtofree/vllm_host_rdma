# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Daemon-mode shared memory segment and SPSC rings, no /dev/shm needed.

Docker caps /dev/shm at 64MB by default while a single block is 128MB, so
this creates an anonymous shared memory segment with memfd_create and
passes its fd over the ctl unix socket via SCM_RIGHTS: nothing lands on a
mount point, so the tmpfs quota does not apply (engine and daemon share
the same container and IPC namespace, so the fd is directly usable).

Segment layout (byte offsets, block areas page-aligned):
  0      header (fixed 128B: magic, sizes, slot counts, ring head+tail;
         offsets in _H_*)
  4096   cmd ring  (engine -> daemon, _CMD_ENTRY bytes per slot)
  ...    cpl ring  (daemon -> engine, _CPL_ENTRY bytes per slot)
  ...    staging block area (n_staging x block_bytes; engine writes D2H
         directly, daemon sends from its own mapping)
  ...    recv block area (n_recv x block_bytes; daemon recv_into lands
         directly, engine reads H2D from its own mapping and returns the
         block via RECV_RELEASE when done)

Rings are SPSC: the cmd ring is written only by engine and read only by
daemon, the cpl ring the other way round; wake-up uses pipe bytes (two
os.pipe pairs, fds passed over ctl along with the memfd). Indices are u32
and wrap around, the full test is tail-head and the empty test is
head==tail; the producer writes only tail and the consumer only head,
which is safe on x86/arm64 given the store ordering (the usual shm queue
convention).

Linux only: memfd/mmap MAP_SHARED is *nix-specific; instantiating this on
Windows raises (daemon mode is only deployed in Linux containers).
"""

from __future__ import annotations

import contextlib
import ctypes
import mmap
import os
import platform
import struct
import sys

# cmd slot: kind,rsv,tag,block_idx,rsv,size,op_id
_CMD_ENTRY = struct.Struct("!IIQIIQQ")
# cpl slot: kind,rsv,op_id,size,err,block_idx (RECV_DONE only, else 0)
_CPL_ENTRY = struct.Struct("!IIQQII")

_MAGIC = 0x4C5744554358444D  # "LWDUCXDM"
_VERSION = 1
_PAGE = 4096
_H_SIZE = 128
# header field offsets (magic u64 at 0; the rest u32, packed explicitly by
# the creator against this same table)
_H_N_STG = 12  # u32
_H_N_RECV = 16  # u32
_H_CMD_SLOTS = 20  # u32
_H_CPL_SLOTS = 24  # u32
_H_CMD_HEAD = 28  # u32: written by consumer (daemon)
_H_CMD_TAIL = 32  # u32: written by producer (engine)
_H_CPL_HEAD = 36  # u32: written by consumer (engine)
_H_CPL_TAIL = 40  # u32: written by producer (daemon)
_H_READY = 44  # u32: written by daemon (engine polls as fallback)
_H_BROKEN = 48  # u32: written by daemon
_H_VERSION = 52  # u32
_H_BLOCK_BYTES = 8  # u32 (<=4GB; a block is far smaller)

# cmd kinds
CMD_SEND = 1  # tag/block_idx/size/op_id: send a staging block
CMD_RECV = 2  # tag/op_id: pre-posted recv (delivered via recv block)
CMD_RECV_DISCARD = 3  # tag: pre-posted discard recv (skip; takes no recv block)
CMD_RECV_RELEASE = 4  # block_idx: engine returns the recv block when done
CMD_FAIL = 5  # unrecoverable application error (broken-link cleanup)
CMD_CLOSE = 6  # orderly channel shutdown
# cpl kinds
CPL_SEND_DONE = 1  # op_id/size/err: send complete (implies staging block return)
CPL_RECV_DONE = 2  # op_id/size/err/block_idx (engine H2D uses block_idx if err=0)
CPL_READY = 3  # link is ready
CPL_BROKEN = 4  # broken link
CPL_CLOSED = 5  # channel shut down

_DEFAULT_CMD_SLOTS = 1024  # need headroom when drain lags (256 slots once overflowed)
_DEFAULT_CPL_SLOTS = 4096

# memfd_create syscall numbers (Linux only)
_MEMFD_NR = {"x86_64": 319, "aarch64": 279}


def _memfd_create() -> int:
    nr = _MEMFD_NR.get(platform.machine())
    if nr is None or not sys.platform.startswith("linux"):
        raise RuntimeError("ucx daemon mode requires Linux (memfd_create)")
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(ctypes.c_long(nr), b"lwd-ucx-chan", 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "memfd_create failed")
    return fd


def segment_size(
    block_bytes: int,
    n_staging: int,
    n_recv: int,
    cmd_slots: int = _DEFAULT_CMD_SLOTS,
    cpl_slots: int = _DEFAULT_CPL_SLOTS,
) -> int:
    cmd_off = _PAGE
    cpl_off = cmd_off + cmd_slots * _CMD_ENTRY.size
    stg_off = (cpl_off + cpl_slots * _CPL_ENTRY.size + _PAGE - 1) // _PAGE * _PAGE
    return stg_off + n_staging * block_bytes + n_recv * block_bytes


def _addr_of(mm: mmap.mmap, offset: int, length: int) -> int:
    """Absolute address of offset in the mapping, passed as send/recv_into addr."""
    return ctypes.addressof((ctypes.c_char * length).from_buffer(mm, offset))


class LwdUcxShmSegment:
    """One daemon-engine shared memory segment: memfd plus one mmap per side.

    The side creating it (creator=True, the daemon) owns the memfd, the
    ftruncate and the header init; the other side (engine) mmaps the fd it
    received. Block access: `staging_view(i)` (D2H write) and
    `recv_view(i, n)` (H2D read); the daemon side passes `*_addr(i)` to
    LwdUcxChannel.
    """

    def __init__(
        self,
        *,
        block_bytes: int,
        n_staging: int,
        n_recv: int,
        fd: int | None = None,
        creator: bool = False,
        cmd_slots: int = _DEFAULT_CMD_SLOTS,
        cpl_slots: int = _DEFAULT_CPL_SLOTS,
    ) -> None:
        self.block_bytes = block_bytes
        self.n_staging = n_staging
        self.n_recv = n_recv
        self.cmd_slots = cmd_slots
        self.cpl_slots = cpl_slots
        self.size = segment_size(block_bytes, n_staging, n_recv, cmd_slots, cpl_slots)
        if creator:
            fd = _memfd_create()
            os.ftruncate(fd, self.size)
        if fd is None:
            raise RuntimeError("LwdUcxShmSegment needs fd (or creator=True)")
        self.fd = fd
        # Linux: MAP_SHARED shared mapping; elsewhere (dev-machine smoke
        # test) fall back to ACCESS_WRITE -- production daemons are Linux only
        if hasattr(mmap, "MAP_SHARED"):
            self._mm = mmap.mmap(fd, self.size, mmap.MAP_SHARED)
        else:  # pragma: no cover - dev-machine smoke only
            self._mm = mmap.mmap(fd, self.size, access=mmap.ACCESS_WRITE)
        self.cmd_off = _PAGE
        self.cpl_off = self.cmd_off + cmd_slots * _CMD_ENTRY.size
        self.stg_off = (
            (self.cpl_off + cpl_slots * _CPL_ENTRY.size + _PAGE - 1) // _PAGE * _PAGE
        )
        self.recv_off = self.stg_off + n_staging * block_bytes
        if creator:
            # explicit pack against the _H_* offsets (magic + sizes/slot
            # counts + zeroed ring head/tail)
            struct.pack_into("<Q", self._mm, 0, _MAGIC)
            struct.pack_into(
                "<IIIII",
                self._mm,
                _H_BLOCK_BYTES,
                block_bytes,
                n_staging,
                n_recv,
                cmd_slots,
                cpl_slots,
            )
            struct.pack_into("<IIIII", self._mm, _H_CMD_HEAD, 0, 0, 0, 0, 0)
            struct.pack_into("<III", self._mm, _H_VERSION, _VERSION, 0, 0)
        else:
            (magic,) = struct.unpack_from("<Q", self._mm, 0)
            if magic != _MAGIC:
                raise RuntimeError(f"shm header magic mismatch {magic:#x}")
            (ver,) = struct.unpack_from("<I", self._mm, _H_VERSION)
            if ver != _VERSION:
                raise RuntimeError(f"shm layout version mismatch {ver}")

    # -- block access (engine view) ----------------------------------------
    def staging_addr(self, idx: int) -> int:
        return _addr_of(
            self._mm, self.stg_off + idx * self.block_bytes, self.block_bytes
        )

    def staging_view(self, idx: int) -> memoryview:
        return memoryview(self._mm)[
            self.stg_off + idx * self.block_bytes : self.stg_off
            + (idx + 1) * self.block_bytes
        ]

    def recv_addr(self, idx: int) -> int:
        return _addr_of(
            self._mm, self.recv_off + idx * self.block_bytes, self.block_bytes
        )

    def recv_view(self, idx: int, length: int | None = None) -> memoryview:
        n = self.block_bytes if length is None else length
        return memoryview(self._mm)[
            self.recv_off + idx * self.block_bytes : self.recv_off
            + idx * self.block_bytes
            + n
        ]

    # -- SPSC rings --------------------------------------------------------
    def cmd_put(
        self, kind: int, tag: int = 0, block_idx: int = 0, size: int = 0, op_id: int = 0
    ) -> None:
        self._put(
            self.cmd_off,
            self.cmd_slots,
            _CMD_ENTRY,
            _H_CMD_TAIL,
            _H_CMD_HEAD,
            (kind, 0, tag, block_idx, 0, size, op_id),
        )

    def cmd_get(self) -> tuple | None:
        return self._get(
            self.cmd_off, self.cmd_slots, _CMD_ENTRY, _H_CMD_TAIL, _H_CMD_HEAD
        )

    def cpl_put(
        self, kind: int, op_id: int = 0, size: int = 0, err: int = 0, block_idx: int = 0
    ) -> None:
        self._put(
            self.cpl_off,
            self.cpl_slots,
            _CPL_ENTRY,
            _H_CPL_TAIL,
            _H_CPL_HEAD,
            (kind, 0, op_id, size, err, block_idx),
        )

    def cpl_get(self) -> tuple | None:
        return self._get(
            self.cpl_off, self.cpl_slots, _CPL_ENTRY, _H_CPL_TAIL, _H_CPL_HEAD
        )

    # -- status words (daemon writes, engine polls as fallback) ------------
    def set_status(self, ready: bool, broken: bool) -> None:
        struct.pack_into(
            "<II", self._mm, _H_READY, 1 if ready else 0, 1 if broken else 0
        )

    def get_status(self) -> tuple[bool, bool]:
        r, b = struct.unpack_from("<II", self._mm, _H_READY)
        return bool(r), bool(b)

    def _put(
        self,
        ring_off: int,
        slots: int,
        entry: struct.Struct,
        tail_field: int,
        head_field: int,
        values: tuple,
    ) -> None:
        (tail,) = struct.unpack_from("<I", self._mm, tail_field)
        (head,) = struct.unpack_from("<I", self._mm, head_field)
        if tail - head >= slots:
            raise RuntimeError(
                "shm ring full — should not happen: depth bounded by block pools"
            )
        entry.pack_into(self._mm, ring_off + (tail % slots) * entry.size, *values)
        struct.pack_into("<I", self._mm, tail_field, tail + 1)

    def _get(
        self,
        ring_off: int,
        slots: int,
        entry: struct.Struct,
        tail_field: int,
        head_field: int,
    ) -> tuple | None:
        (tail,) = struct.unpack_from("<I", self._mm, tail_field)
        (head,) = struct.unpack_from("<I", self._mm, head_field)
        if head == tail:
            return None
        vals = entry.unpack_from(self._mm, ring_off + (head % slots) * entry.size)
        struct.pack_into("<I", self._mm, head_field, head + 1)
        return vals

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._mm.close()
        with contextlib.suppress(Exception):
            os.close(self.fd)
