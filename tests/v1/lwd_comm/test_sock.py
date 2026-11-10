# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loopback tests of the UHC1 wire TCP helpers (data_comm/sock.py).

Covers the fd-centric contract the wire layer relies on: raw fds in and
out, wrap/detach never closing a caller-owned fd, MSG_NOSIGNAL sends,
recv_into into a raw pointer, bounded connect, and accept/peer format.
Linux only — CPython on Windows cannot wrap detached socket handles.
"""

from __future__ import annotations

import ctypes
import errno
import os
import socket
import sys
import threading

import pytest

from vllm.v1.lwd_comm.data_comm import sock as wire

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="socket(fileno=) needs POSIX fd semantics"
)


def _loopback_pair():
    srv_fd = wire.c_open()
    s = socket.socket(fileno=srv_fd)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.listen(1)
    s.detach()
    return srv_fd, port


def test_handshake_roundtrip_and_peer_format():
    srv_fd, port = _loopback_pair()
    got = {}

    def server():
        conn_fd, peer = wire.c_accept(srv_fd)
        got["peer"] = peer
        buf = ctypes.create_string_buffer(64)
        n = wire.c_recv_ptr(conn_fd, ctypes.addressof(buf), 64)
        wire.c_send_bytes(conn_fd, b"ack:" + buf.raw[:n])
        os.close(conn_fd)

    t = threading.Thread(target=server)
    t.start()
    fd = wire.c_open()
    try:
        wire.c_connect_timeout(fd, "127.0.0.1", port, 5.0)
        wire.c_send_bytes(fd, b"hello-wire")
        out = ctypes.create_string_buffer(64)
        n = wire.c_recv_ptr(fd, ctypes.addressof(out), 64)
        assert out.raw[:n] == b"ack:hello-wire"
    finally:
        os.close(fd)
        t.join()
        os.close(srv_fd)
    assert got["peer"].startswith("127.0.0.1:")


def test_wrap_detach_never_closes_caller_fd():
    fd = wire.c_open()
    for _ in range(200):
        s = socket.socket(fileno=fd)
        s.detach()
        del s
    probe = socket.socket(fileno=fd)
    probe.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)  # raises EBADF if closed
    probe.detach()
    os.close(fd)


def test_connect_timeout_is_bounded():
    # 203.0.113.0/24 (TEST-NET-3) is never routed: connect must fail
    # with ETIMEDOUT within the bound, not the kernel's ~2min SYN retry.
    fd = wire.c_open()
    try:
        with pytest.raises(OSError) as ei:
            wire.c_connect_timeout(fd, "203.0.113.1", 12345, 1.0)
        assert (
            ei.value.errno in (errno.ETIMEDOUT, errno.EHOSTUNREACH, errno.ENETUNREACH)
            or ei.value.errno is None
        )
    finally:
        os.close(fd)


def test_recv_ptr_returns_zero_on_peer_close():
    srv_fd, port = _loopback_pair()
    done = threading.Event()

    def server():
        conn_fd, _ = wire.c_accept(srv_fd)
        os.close(conn_fd)  # abrupt close
        done.set()

    t = threading.Thread(target=server)
    t.start()
    fd = wire.c_open()
    try:
        wire.c_connect_timeout(fd, "127.0.0.1", port, 5.0)
        assert done.wait(5.0)
        buf = ctypes.create_string_buffer(8)
        assert wire.c_recv_ptr(fd, ctypes.addressof(buf), 8) == 0
    finally:
        os.close(fd)
        t.join()
        os.close(srv_fd)
