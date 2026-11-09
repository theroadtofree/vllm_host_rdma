# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""daemon 模式的共享内存段与 SPSC 环(memfd,无 /dev/shm 依赖)。

Docker 默认 /dev/shm 仅 64MB,而单块即 128MB——故用 memfd_create 匿名
共享内存 + fd 经 ctl unix socket SCM_RIGHTS 传递,不落任何挂载点,
不受 tmpfs 配额约束(engine 与 daemon 同容器同 IPC namespace,fd 直接
可用)。

段布局(字节偏移,块区页对齐):
  0      header(定长 128B:magic/尺寸/槽数/环头尾,偏移见 _H_*)
  4096   cmd ring  (engine→daemon,每槽 _CMD_ENTRY 字节)
  ...    cpl ring  (daemon→engine,每槽 _CPL_ENTRY 字节)
  ...    staging 块区(n_staging × block_bytes;engine D2H 直写,
         daemon 从自身映射发送)
  ...    recv 块区  (n_recv × block_bytes;daemon recv_into 直落,
         engine 从自身映射 H2D,用毕经 RECV_RELEASE 归还)

环为 SPSC:cmd 环 engine 单写/daemon 单读,cpl 环 daemon 单写/engine
单读;唤醒走管道字节(两对 os.pipe,fd 随 memfd 一并经 ctl 传递)。
索引 u32 回绕,判满 tail-head,判空 head==tail;producer 只写 tail、
consumer 只写 head,x86/arm64 依赖存储顺序下安全(常见 shm 队列口径)。

Linux only:memfd/mmap MAP_SHARED 依赖 *nix;Windows 上实例化时报错
(daemon 模式本就只部署在 Linux 容器)。
"""

from __future__ import annotations

import contextlib
import ctypes
import mmap
import os
import struct
import sys

# cmd 槽:kind,rsv,tag,block_idx,rsv,size,op_id
_CMD_ENTRY = struct.Struct("!IIQIIQQ")
# cpl 槽:kind,rsv,op_id,size,err,block_idx(RECV_DONE 用,其余 0)
_CPL_ENTRY = struct.Struct("!IIQQII")

_MAGIC = 0x4C5744554358444D  # "LWDUCXDM"
_VERSION = 1
_PAGE = 4096
_H_SIZE = 128
# header 内字段偏移(magic u64 @0;其余 u32,creator 按同表显式 pack)
_H_N_STG = 12         # u32
_H_N_RECV = 16        # u32
_H_CMD_SLOTS = 20     # u32
_H_CPL_SLOTS = 24     # u32
_H_CMD_HEAD = 28      # u32:consumer(daemon)写
_H_CMD_TAIL = 32      # u32:producer(engine)写
_H_CPL_HEAD = 36      # u32:consumer(engine)写
_H_CPL_TAIL = 40      # u32:producer(daemon)写
_H_READY = 44         # u32:daemon 写(engine 轮询兜底)
_H_BROKEN = 48        # u32:daemon 写
_H_VERSION = 52       # u32
_H_BLOCK_BYTES = 8    # u32(≤4GB,块容量远小于此)

# cmd kinds
CMD_SEND = 1          # tag/block_idx/size/op_id:发 staging 块
CMD_RECV = 2          # tag/op_id:预挂接收(经 recv 块交付)
CMD_RECV_DISCARD = 3  # tag:预挂丢弃接收(skip;不占 recv 块)
CMD_RECV_RELEASE = 4  # block_idx:engine 用毕归还 recv 块
CMD_FAIL = 5          # 应用侧不可恢复错(断链善后)
CMD_CLOSE = 6         # 有序关停通道
# cpl kinds
CPL_SEND_DONE = 1     # op_id/size/err:发送完成(隐含 staging 块归还)
CPL_RECV_DONE = 2     # op_id/size/err/block_idx(err=0 时 engine H2D 用)
CPL_READY = 3         # 链路就绪
CPL_BROKEN = 4        # 断链
CPL_CLOSED = 5        # 通道已关停

_DEFAULT_CMD_SLOTS = 1024   # 瞬态排水落后需余量(曾 256 在负载尖峰打满)
_DEFAULT_CPL_SLOTS = 4096

# memfd_create syscall 号(仅 Linux)
_MEMFD_NR = {"x86_64": 319, "aarch64": 279}


def _memfd_create() -> int:
    nr = _MEMFD_NR.get(getattr(os, "uname", lambda: None)().machine
                       if hasattr(os, "uname") else "")
    if nr is None or not sys.platform.startswith("linux"):
        raise RuntimeError("ucx daemon mode requires Linux (memfd_create)")
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(ctypes.c_long(nr), b"lwd-ucx-chan", 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "memfd_create failed")
    return fd


def segment_size(block_bytes: int, n_staging: int, n_recv: int,
                 cmd_slots: int = _DEFAULT_CMD_SLOTS,
                 cpl_slots: int = _DEFAULT_CPL_SLOTS) -> int:
    cmd_off = _PAGE
    cpl_off = cmd_off + cmd_slots * _CMD_ENTRY.size
    stg_off = (cpl_off + cpl_slots * _CPL_ENTRY.size
               + _PAGE - 1) // _PAGE * _PAGE
    return stg_off + n_staging * block_bytes + n_recv * block_bytes


def _addr_of(mm: mmap.mmap, offset: int, length: int) -> int:
    """映射内 offset 的绝对地址(供 UcxChannel.send/recv_into 传 addr)。"""
    return ctypes.addressof((ctypes.c_char * length).from_buffer(mm, offset))


class ShmSegment:
    """一段 daemon-engine 共享内存:memfd + 双方各自 mmap。

    creator=True 的一方(daemon)负责 memfd、ftruncate、header 初始化;
    另一方(engine)用收到的 fd 直接 mmap。块访问:`staging_view(i)`
    (D2H 写)/`recv_view(i, n)`(H2D 读);daemon 侧用 `*_addr(i)` 传给
    UcxChannel。
    """

    def __init__(self, *, block_bytes: int, n_staging: int, n_recv: int,
                 fd: int | None = None, creator: bool = False,
                 cmd_slots: int = _DEFAULT_CMD_SLOTS,
                 cpl_slots: int = _DEFAULT_CPL_SLOTS) -> None:
        self.block_bytes = block_bytes
        self.n_staging = n_staging
        self.n_recv = n_recv
        self.cmd_slots = cmd_slots
        self.cpl_slots = cpl_slots
        self.size = segment_size(block_bytes, n_staging, n_recv,
                                 cmd_slots, cpl_slots)
        if creator:
            fd = _memfd_create()
            os.ftruncate(fd, self.size)
        if fd is None:
            raise RuntimeError("ShmSegment needs fd (or creator=True)")
        self.fd = fd
        # Linux:MAP_SHARED 共享映射;非 Linux(开发机冒烟测试)回退
        # ACCESS_WRITE(生产 daemon 仅部署于 Linux)
        if hasattr(mmap, "MAP_SHARED"):
            self._mm = mmap.mmap(fd, self.size, mmap.MAP_SHARED)
        else:  # pragma: no cover - dev-machine smoke only
            self._mm = mmap.mmap(fd, self.size, access=mmap.ACCESS_WRITE)
        self.cmd_off = _PAGE
        self.cpl_off = self.cmd_off + cmd_slots * _CMD_ENTRY.size
        self.stg_off = (self.cpl_off + cpl_slots * _CPL_ENTRY.size
                        + _PAGE - 1) // _PAGE * _PAGE
        self.recv_off = self.stg_off + n_staging * block_bytes
        if creator:
            # 按 _H_* 偏移显式 pack(magic + 尺寸/槽数 + 环头尾清零)
            struct.pack_into("<Q", self._mm, 0, _MAGIC)
            struct.pack_into("<IIIII", self._mm, _H_BLOCK_BYTES,
                             block_bytes, n_staging, n_recv,
                             cmd_slots, cpl_slots)
            struct.pack_into("<IIIII", self._mm, _H_CMD_HEAD, 0, 0, 0, 0, 0)
            struct.pack_into("<III", self._mm, _H_VERSION, _VERSION, 0, 0)
        else:
            (magic,) = struct.unpack_from("<Q", self._mm, 0)
            if magic != _MAGIC:
                raise RuntimeError(f"shm header magic mismatch {magic:#x}")
            (ver,) = struct.unpack_from("<I", self._mm, _H_VERSION)
            if ver != _VERSION:
                raise RuntimeError(f"shm layout version mismatch {ver}")

    # -- 块访问(engine 视角) ------------------------------------------
    def staging_addr(self, idx: int) -> int:
        return _addr_of(self._mm, self.stg_off + idx * self.block_bytes,
                        self.block_bytes)

    def staging_view(self, idx: int) -> memoryview:
        return memoryview(self._mm)[
            self.stg_off + idx * self.block_bytes:
            self.stg_off + (idx + 1) * self.block_bytes]

    def recv_addr(self, idx: int) -> int:
        return _addr_of(self._mm, self.recv_off + idx * self.block_bytes,
                        self.block_bytes)

    def recv_view(self, idx: int, length: int | None = None) -> memoryview:
        n = self.block_bytes if length is None else length
        return memoryview(self._mm)[
            self.recv_off + idx * self.block_bytes:
            self.recv_off + idx * self.block_bytes + n]

    # -- 环 SPSC --------------------------------------------------------
    def cmd_put(self, kind: int, tag: int = 0, block_idx: int = 0,
                size: int = 0, op_id: int = 0) -> None:
        self._put(self.cmd_off, self.cmd_slots, _CMD_ENTRY,
                  _H_CMD_TAIL, _H_CMD_HEAD,
                  (kind, 0, tag, block_idx, 0, size, op_id))

    def cmd_get(self) -> tuple | None:
        return self._get(self.cmd_off, self.cmd_slots, _CMD_ENTRY,
                         _H_CMD_TAIL, _H_CMD_HEAD)

    def cpl_put(self, kind: int, op_id: int = 0, size: int = 0,
                err: int = 0, block_idx: int = 0) -> None:
        self._put(self.cpl_off, self.cpl_slots, _CPL_ENTRY,
                  _H_CPL_TAIL, _H_CPL_HEAD,
                  (kind, 0, op_id, size, err, block_idx))

    def cpl_get(self) -> tuple | None:
        return self._get(self.cpl_off, self.cpl_slots, _CPL_ENTRY,
                         _H_CPL_TAIL, _H_CPL_HEAD)

    # -- 状态字(daemon 写,engine 轮询兜底) ---------------------------
    def set_status(self, ready: bool, broken: bool) -> None:
        struct.pack_into("<II", self._mm, _H_READY,
                         1 if ready else 0, 1 if broken else 0)

    def get_status(self) -> tuple[bool, bool]:
        r, b = struct.unpack_from("<II", self._mm, _H_READY)
        return bool(r), bool(b)

    def _put(self, ring_off: int, slots: int, entry: struct.Struct,
             tail_field: int, head_field: int, values: tuple) -> None:
        (tail,) = struct.unpack_from("<I", self._mm, tail_field)
        (head,) = struct.unpack_from("<I", self._mm, head_field)
        if tail - head >= slots:
            raise RuntimeError("shm ring full — should not happen: "
                               "depth bounded by block pools")
        entry.pack_into(self._mm, ring_off + (tail % slots) * entry.size,
                        *values)
        struct.pack_into("<I", self._mm, tail_field, tail + 1)

    def _get(self, ring_off: int, slots: int, entry: struct.Struct,
             tail_field: int, head_field: int) -> tuple | None:
        (tail,) = struct.unpack_from("<I", self._mm, tail_field)
        (head,) = struct.unpack_from("<I", self._mm, head_field)
        if head == tail:
            return None
        vals = entry.unpack_from(
            self._mm, ring_off + (head % slots) * entry.size)
        struct.pack_into("<I", self._mm, head_field, head + 1)
        return vals

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._mm.close()
        with contextlib.suppress(Exception):
            os.close(self.fd)
