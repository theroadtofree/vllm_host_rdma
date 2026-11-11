# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TCP helpers for the LWD data plane wire handshake (device-agnostic).

The external interface remains fd-centric (bare fds in and out, ownership
stays with the caller): each c_* helper wraps the fd in a temporary
socket.socket(fileno=fd) and must detach() before returning, otherwise the
object's GC closes the underlying fd. Handshake semantics: blocking model,
MSG_NOSIGNAL, bounded connect, kernel-level send/recv timeouts; EINTR is
retried automatically by PEP 475. IPv4 only.
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
    """Create a TCP stream socket and return its bare fd; blocking by default."""
    return _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM).detach()


def c_setsockopt_int(
    fd: int, level: int, optname: int, value: int, *, what: str
) -> None:
    """Set one integer socket option, naming it when the kernel rejects it."""
    s = _socket.socket(fileno=fd)
    try:
        s.setsockopt(level, optname, value)
    except OSError as exc:
        raise OSError(exc.errno, f"lwd setsockopt {what}: {exc.strerror}") from exc
    finally:
        s.detach()


def c_setsockopt_int_raw(
    fd: int, level: int, optname: int, raw: bytes, *, what: str
) -> None:
    """Raw-bytes setsockopt, used for non-int options such as timeval."""
    s = _socket.socket(fileno=fd)
    try:
        s.setsockopt(level, optname, raw)
    except OSError as exc:
        raise OSError(exc.errno, f"lwd setsockopt {what}: {exc.strerror}") from exc
    finally:
        s.detach()


def c_bind(fd: int, host: str, port: int) -> None:
    s = _socket.socket(fileno=fd)
    try:
        s.bind((host, port))
    except OSError as exc:
        raise OSError(exc.errno, f"lwd bind: {exc.strerror}", (host, port)) from exc
    finally:
        s.detach()


def c_listen(fd: int, backlog: int = 1) -> None:
    s = _socket.socket(fileno=fd)
    try:
        s.listen(backlog)
    finally:
        s.detach()


def c_connect_timeout(fd: int, host: str, port: int, timeout_s: float) -> None:
    """TCP connect with a timeout, for handshake and reconnect: on timeout
    raise OSError(ETIMEDOUT); the fd is blocking again on success.

    Interruptibility is a hard requirement of the reconnect supervisor
    thread: a kernel-level blocking connect to an unreachable peer waits for
    the ~2min SYN retry, so a shutdown join never converges. Every attempt
    must be bounded.
    """
    s = _socket.socket(fileno=fd)
    try:
        s.settimeout(timeout_s)
        s.connect((host, port))
    except TimeoutError:
        raise OSError(
            _errno.ETIMEDOUT, f"connect timeout >{timeout_s}s", f"{host}:{port}"
        ) from None
    except OSError as exc:
        raise OSError(exc.errno, exc.strerror, f"{host}:{port}") from exc
    finally:
        os.set_blocking(fd, True)
        s.detach()


def c_set_sock_timeouts(fd: int, timeout_s: float) -> None:
    """Handshake send/recv timeouts (SO_RCVTIMEO/SO_SNDTIMEO): if the peer is
    stuck or half-open, recv/send raises EAGAIN instead of blocking forever.
    Packs a timeval{sec, usec}."""
    tv = struct.pack("qq", int(timeout_s), int((timeout_s % 1) * 1_000_000))
    for opt in (_SO_RCVTIMEO, _SO_SNDTIMEO):
        c_setsockopt_int_raw(fd, _SOL_SOCKET, opt, tv, what="sock timeout")


def c_accept(srv_fd: int) -> tuple[int, str]:
    """Accept one connection; returns (conn_fd, "peer_host:peer_port")."""
    s = _socket.socket(fileno=srv_fd)
    try:
        conn, addr = s.accept()
    except OSError as exc:
        raise OSError(exc.errno, f"lwd accept: {exc.strerror}") from exc
    finally:
        s.detach()
    return conn.detach(), f"{addr[0]}:{addr[1]}"


def c_set_tos(fd: int, tos: int) -> None:
    """IP_TOS (same DSCP as RDMA data frames: tos = traffic_class << 2).
    The server sets it before bind so accepted connections inherit it (Linux
    inherits the listening socket's TOS); the client sets it before connect.
    Failure is not fatal (warn only when the kernel/platform does not
    support it)."""
    s = _socket.socket(fileno=fd)
    try:
        s.setsockopt(_socket.IPPROTO_IP, _socket.IP_TOS, tos)
    except OSError as exc:
        logger.warning("[lwd-sock] setsockopt IP_TOS=%d failed: %s", tos, exc)
    finally:
        s.detach()


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
    """Send a large buffer from an arbitrary memory address; same semantics
    as sendall."""
    s = _socket.socket(fileno=fd)
    try:
        s.sendall((ctypes.c_char * size).from_address(ptr), _MSG_NOSIGNAL)
    except OSError as exc:
        raise OSError(exc.errno, f"lwd send: {exc.strerror}") from exc
    finally:
        s.detach()


def c_recv_ptr(fd: int, ptr: int, size: int) -> int:
    """Read up to size bytes into ptr; returns 0 when the peer closed."""
    s = _socket.socket(fileno=fd)
    try:
        return s.recv_into((ctypes.c_char * size).from_address(ptr))
    except OSError as exc:
        raise OSError(exc.errno, f"lwd recv: {exc.strerror}") from exc
    finally:
        s.detach()
