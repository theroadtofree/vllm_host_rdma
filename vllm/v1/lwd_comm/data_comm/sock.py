# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD 数据面 wire 握手用的 TCP socket ctypes 助手(device 无关)。

从 vllm_ascend/distributed/lwd_comm/smc_wire.py 移植的 TCP 子集
(原文件不动):vllm 侧的 UCX wire 核心依赖这些助手完成 TCP 握手,
但不能反向依赖 vllm_ascend——故此处持有独立拷贝。AF_SMC 数据面
socket 留在 vllm_ascend,这里只有 tcp;语义与原件逐一一致
(阻塞模型、MSG_NOSIGNAL、EINTR 重试、可中断 connect)。
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno as _errno
import os
import socket as _socket
import struct

from vllm.logger import init_logger

logger = init_logger(__name__)

_SOCK_STREAM = 1
_SOL_SOCKET = 1
_SO_REUSEADDR = 2
_SO_ERROR = 4
_SO_RCVTIMEO = 20
_SO_SNDTIMEO = 21
_MSG_NOSIGNAL = 0x4000  # 对端关闭时返回 EPIPE 而非杀进程 SIGPIPE
_SOCKADDR_IN = struct.Struct("=HH4s8s")
_SOCKADDR_STORAGE_BYTES = 128  # 容得下 in/in6/un,accept 出参缓冲

try:
    _libc = ctypes.CDLL(
        ctypes.util.find_library("c") or "libc.so.6", use_errno=True
    )
    _libc.socket.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    _libc.socket.restype = ctypes.c_int
    _libc.setsockopt.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_uint,
    ]
    _libc.setsockopt.restype = ctypes.c_int
    _libc.getsockopt.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint),
    ]
    _libc.getsockopt.restype = ctypes.c_int
    _libc.bind.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    _libc.bind.restype = ctypes.c_int
    _libc.listen.argtypes = [ctypes.c_int, ctypes.c_int]
    _libc.listen.restype = ctypes.c_int
    _libc.connect.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_uint
    ]
    _libc.connect.restype = ctypes.c_int
    _libc.accept.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint)
    ]
    _libc.accept.restype = ctypes.c_int
    _libc.send.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    ]
    _libc.send.restype = ctypes.c_ssize_t
    _libc.recv.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    ]
    _libc.recv.restype = ctypes.c_ssize_t
    _libc.close.argtypes = [ctypes.c_int]
    _libc.close.restype = ctypes.c_int
except OSError:
    _libc = None


def _require_libc() -> None:
    if _libc is None:
        raise RuntimeError("lwd wire handshake requires Linux libc")


def _raise_errno(prefix: str, addr: tuple | None = None) -> None:
    err = ctypes.get_errno()
    raise OSError(err, f"{prefix}: {os.strerror(err)}", addr)


def _pack_sockaddr_in(host: str, port: int) -> bytes:
    return _SOCKADDR_IN.pack(
        _socket.AF_INET,
        _socket.htons(port),
        _socket.inet_aton(host),
        b"\x00" * 8,
    )


def _parse_sockaddr_in(raw: bytes) -> tuple[str, int]:
    """从 accept 出参的 sockaddr_in 解 (host, port);异常时给占位值。"""
    if len(raw) < 8:
        return ("?", 0)
    sa_family = int.from_bytes(raw[0:2], "little")
    if sa_family != _socket.AF_INET:
        return (f"<family {sa_family}>", 0)
    return (
        _socket.inet_ntoa(raw[4:8]),
        struct.unpack("!H", raw[2:4])[0],
    )


def c_open() -> int:
    """建 TCP 流式 socket 返回裸 fd;默认阻塞(libc 语义)。"""
    _require_libc()
    fd = _libc.socket(_socket.AF_INET, _SOCK_STREAM, 0)
    if fd < 0:
        _raise_errno("socket")
    return fd


def c_setsockopt_int(
    fd: int, level: int, optname: int, value: int, *, what: str
) -> None:
    val = struct.pack("i", value)
    if _libc.setsockopt(fd, level, optname, val, len(val)) != 0:
        _raise_errno(f"setsockopt {what}")


def c_getsockopt_int(fd: int, level: int, optname: int, what: str) -> int:
    val = ctypes.c_int(0)
    vlen = ctypes.c_uint(ctypes.sizeof(val))
    if _libc.getsockopt(
        fd, level, optname, ctypes.byref(val), ctypes.byref(vlen)
    ) != 0:
        _raise_errno(f"getsockopt {what}")
    return val.value


def c_setsockopt_int_raw(
    fd: int, level: int, optname: int, raw: bytes, *, what: str
) -> None:
    """setsockopt 原始字节版(int 版见 c_setsockopt_int;timeval 等
    非整型选项用)。"""
    if _libc.setsockopt(fd, level, optname, raw, len(raw)) != 0:
        _raise_errno(f"setsockopt {what}")


def c_bind(fd: int, host: str, port: int) -> None:
    addr = _pack_sockaddr_in(host, port)
    if _libc.bind(fd, addr, len(addr)) < 0:
        _raise_errno("lwd bind", (host, port))


def c_listen(fd: int, backlog: int = 1) -> None:
    if _libc.listen(fd, backlog) != 0:
        _raise_errno("lwd listen")


def c_connect_timeout(
    fd: int, host: str, port: int, timeout_s: float
) -> None:
    """带超时的 TCP connect(握手/重连专用):非阻塞发起 + select 等
    可写 + SO_ERROR 判定,超时抛 OSError(ETIMEDOUT),成功后恢复阻塞。

    可中断性是重连监督线程的硬要求:不可达目标的内核级阻塞 connect
    要等 ~2min SYN 重试,关停 join 无法收敛——每轮必须有界。"""
    import select as _select
    addr = _pack_sockaddr_in(host, port)
    os.set_blocking(fd, False)
    try:
        if _libc.connect(fd, addr, len(addr)) < 0:
            err = ctypes.get_errno()
            if err not in (_errno.EINPROGRESS, _errno.EINTR):
                raise OSError(err, os.strerror(err), f"{host}:{port}")
            _, writable, _ = _select.select([], [fd], [fd], timeout_s)
            if not writable:
                raise OSError(
                    _errno.ETIMEDOUT,
                    f"connect timeout >{timeout_s}s", f"{host}:{port}")
            so_err = c_getsockopt_int(
                fd, _SOL_SOCKET, _SO_ERROR, what="SO_ERROR")
            if so_err != 0:
                raise OSError(so_err, os.strerror(so_err), f"{host}:{port}")
    finally:
        os.set_blocking(fd, True)


def c_set_sock_timeouts(fd: int, timeout_s: float) -> None:
    """握手收发超时(SO_RCVTIMEO/SO_SNDTIMEO):对端僵死/半开时
    recv/send 以 EAGAIN 抛出而非无限阻塞。timeval{秒,微秒}。"""
    tv = struct.pack(
        "qq", int(timeout_s), int((timeout_s % 1) * 1_000_000))
    for opt, name in ((_SO_RCVTIMEO, "SO_RCVTIMEO"),
                      (_SO_SNDTIMEO, "SO_SNDTIMEO")):
        c_setsockopt_int_raw(fd, _SOL_SOCKET, opt, tv, what=name)


def c_accept(srv_fd: int) -> tuple[int, str]:
    """accept 一条连接,返回 (conn_fd, "peer_host:peer_port")。"""
    addrbuf = ctypes.create_string_buffer(_SOCKADDR_STORAGE_BYTES)
    addrlen = ctypes.c_uint(_SOCKADDR_STORAGE_BYTES)
    conn_fd = _libc.accept(srv_fd, addrbuf, ctypes.byref(addrlen))
    if conn_fd < 0:
        _raise_errno("lwd accept")
    host, port = _parse_sockaddr_in(addrbuf.raw[: addrlen.value])
    return conn_fd, f"{host}:{port}"


def c_close(fd: int) -> None:
    import contextlib
    with contextlib.suppress(OSError):
        _libc.close(fd)


def c_send_bytes(fd: int, data: bytes) -> None:
    """小段发送(帧头);复用 c_send_ptr 的 MSG_NOSIGNAL/EINTR 语义。"""
    buf = ctypes.create_string_buffer(data, len(data))
    c_send_ptr(fd, ctypes.addressof(buf), len(data))


def c_send_ptr(fd: int, ptr: int, size: int) -> None:
    """大块发送;语义同 sendall。"""
    sent = 0
    while sent < size:
        n = _libc.send(
            fd, ctypes.c_void_p(ptr + sent), size - sent, _MSG_NOSIGNAL
        )
        if n < 0:
            err = ctypes.get_errno()
            if err == _errno.EINTR:
                continue
            _raise_errno("lwd send")
        sent += n


def c_recv_ptr(fd: int, ptr: int, size: int) -> int:
    """读至多 size 字节到 ptr;返回 0 = 对端关闭。EINTR 重试。"""
    while True:
        n = _libc.recv(fd, ctypes.c_void_p(ptr), size, 0)
        if n >= 0:
            return n
        err = ctypes.get_errno()
        if err == _errno.EINTR:
            continue
        _raise_errno("lwd recv")
