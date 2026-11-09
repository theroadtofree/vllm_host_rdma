# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD 数据面 wire 握手用的 TCP 助手(device 无关)。

对外仍是 fd 中心接口(裸 fd 出入,所有权归调用方):内部用 Python
socket 实现每个 c_* 助手——每次操作以 socket.socket(fileno=fd)
临时包一层,用完必须 detach(),否则对象 GC 会关闭底层 fd。握手
语义与 ctypes 版逐一等价:阻塞模型、MSG_NOSIGNAL、有界 connect、
内核级收发超时;EINTR 由 PEP 475 自动重试。IPv4-only。
"""

from __future__ import annotations

import contextlib
import ctypes
import errno as _errno
import os
import socket as _socket
import struct

from vllm.logger import init_logger

logger = init_logger(__name__)

_SOL_SOCKET = _socket.SOL_SOCKET
_SO_REUSEADDR = _socket.SO_REUSEADDR  # ucx_channel's listener sets it directly
_SO_RCVTIMEO = getattr(_socket, "SO_RCVTIMEO", 20)
_SO_SNDTIMEO = getattr(_socket, "SO_SNDTIMEO", 21)
_MSG_NOSIGNAL = getattr(_socket, "MSG_NOSIGNAL", 0)


def c_open() -> int:
    """建 TCP 流式 socket 返回裸 fd;默认阻塞。"""
    return _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM).detach()


def c_setsockopt_int(
    fd: int, level: int, optname: int, value: int, *, what: str
) -> None:
    s = _socket.socket(fileno=fd)
    try:
        s.setsockopt(level, optname, value)
    finally:
        s.detach()


def c_getsockopt_int(fd: int, level: int, optname: int, what: str) -> int:
    s = _socket.socket(fileno=fd)
    try:
        return s.getsockopt(level, optname)
    finally:
        s.detach()


def c_setsockopt_int_raw(
    fd: int, level: int, optname: int, raw: bytes, *, what: str
) -> None:
    """setsockopt 原始字节版(timeval 等非整型选项用)。"""
    s = _socket.socket(fileno=fd)
    try:
        s.setsockopt(level, optname, raw)
    finally:
        s.detach()


def c_bind(fd: int, host: str, port: int) -> None:
    s = _socket.socket(fileno=fd)
    try:
        s.bind((host, port))
    except OSError as exc:
        raise OSError(
            exc.errno, f"lwd bind: {exc.strerror}", (host, port)) from exc
    finally:
        s.detach()


def c_listen(fd: int, backlog: int = 1) -> None:
    s = _socket.socket(fileno=fd)
    try:
        s.listen(backlog)
    finally:
        s.detach()


def c_connect_timeout(
    fd: int, host: str, port: int, timeout_s: float
) -> None:
    """带超时的 TCP connect(握手/重连专用):超时抛 OSError(ETIMEDOUT),
    成功后恢复阻塞。

    可中断性是重连监督线程的硬要求:不可达目标的内核级阻塞 connect
    要等 ~2min SYN 重试,关停 join 无法收敛——每轮必须有界。"""
    s = _socket.socket(fileno=fd)
    try:
        s.settimeout(timeout_s)
        s.connect((host, port))
    except _socket.timeout:
        raise OSError(
            _errno.ETIMEDOUT,
            f"connect timeout >{timeout_s}s", f"{host}:{port}") from None
    except OSError as exc:
        raise OSError(
            exc.errno, exc.strerror, f"{host}:{port}") from exc
    finally:
        os.set_blocking(fd, True)
        s.detach()


def c_set_sock_timeouts(fd: int, timeout_s: float) -> None:
    """握手收发超时(SO_RCVTIMEO/SO_SNDTIMEO):对端僵死/半开时
    recv/send 以 EAGAIN 抛出而非无限阻塞。timeval{秒,微秒}。"""
    tv = struct.pack(
        "qq", int(timeout_s), int((timeout_s % 1) * 1_000_000))
    for opt in (_SO_RCVTIMEO, _SO_SNDTIMEO):
        c_setsockopt_int_raw(fd, _SOL_SOCKET, opt, tv, what="sock timeout")


def c_accept(srv_fd: int) -> tuple[int, str]:
    """accept 一条连接,返回 (conn_fd, "peer_host:peer_port")。"""
    s = _socket.socket(fileno=srv_fd)
    try:
        conn, addr = s.accept()
    except OSError as exc:
        raise OSError(exc.errno, f"lwd accept: {exc.strerror}") from exc
    finally:
        s.detach()
    return conn.detach(), f"{addr[0]}:{addr[1]}"


def c_close(fd: int) -> None:
    with contextlib.suppress(OSError):
        os.close(fd)


def c_send_bytes(fd: int, data: bytes) -> None:
    s = _socket.socket(fileno=fd)
    try:
        s.sendall(data, _MSG_NOSIGNAL)
    except OSError as exc:
        raise OSError(exc.errno, f"lwd send: {exc.strerror}") from exc
    finally:
        s.detach()


def c_send_ptr(fd: int, ptr: int, size: int) -> None:
    """大块发送(任意内存地址);语义同 sendall。"""
    s = _socket.socket(fileno=fd)
    try:
        s.sendall(
            (ctypes.c_char * size).from_address(ptr), _MSG_NOSIGNAL)
    except OSError as exc:
        raise OSError(exc.errno, f"lwd send: {exc.strerror}") from exc
    finally:
        s.detach()


def c_recv_ptr(fd: int, ptr: int, size: int) -> int:
    """读至多 size 字节到 ptr;返回 0 = 对端关闭。"""
    s = _socket.socket(fileno=fd)
    try:
        return s.recv_into((ctypes.c_char * size).from_address(ptr))
    except OSError as exc:
        raise OSError(exc.errno, f"lwd recv: {exc.strerror}") from exc
    finally:
        s.detach()
