# SPDX-License-Identifier: Apache-2.0
"""Attach-runtime self-healing of the frontend registration server
(peer-restart line, phase 3).

A restarted attach frontend re-registers with the same client index: the
server must reset the dead process's stale state instead of rejecting
it, and a resident service must keep answering after the startup barrier
(the barrier covers launch; runtime re-registrations are the watchdog's
silent rounds or a restarted frontend's fresh attach). Endpoints
re-report replaces the stale entry.
"""

from __future__ import annotations

from typing import Any

import pytest

from vllm.entrypoints.openai.lwd.frontend_attach import FrontendAttachServer


def _server() -> FrontendAttachServer:
    server = object.__new__(FrontendAttachServer)
    server.client_count = 2
    server.config_payload = {"k": "v"}
    server.frontend_args_payload = {}
    server.stats_update_address = None
    server._listen_addresses = {0: ("127.0.0.1", 8000)}
    server._accepted = set()
    server._accepted_pids = {}
    server._reported = {}
    server._closed = False
    return server


def _attach_message(pid: int, listen_port: int = 8100) -> dict[str, Any]:
    return {
        "type": "LWD_FRONTEND_ATTACH",
        "client_index": 1,
        "config_payload": {"k": "v"},
        "listen_host": "127.0.0.1",
        "listen_port": listen_port,
        "pid": pid,
    }


def _endpoints_message(input_address: str, output_address: str) -> dict[str, Any]:
    return {
        "type": "LWD_FRONTEND_ENDPOINTS",
        "client_index": 1,
        "input_address": input_address,
        "output_address": output_address,
    }


def _handle(server: FrontendAttachServer, message: dict[str, Any]) -> dict[str, Any]:
    replies: list[dict[str, Any]] = []

    def reply(response: dict[str, Any]) -> None:
        replies.append(response)

    server._handle_message(b"identity", _payload(message), reply=reply)
    return replies[-1]


def _payload(message: dict[str, Any]) -> bytes:
    import msgspec

    return msgspec.msgpack.encode(message)


def test_restarted_frontend_reregister_resets_its_stale_state() -> None:
    """A new pid on the same index is a restarted frontend: the old state
    (listen address, endpoints) belonged to the dead process and must be
    reset before the fresh attach is accepted."""
    server = _server()
    # First incarnation attaches and reports.
    _handle(server, _attach_message(pid=111, listen_port=8100))
    _handle(
        server,
        _endpoints_message("tcp://127.0.0.1:9001", "tcp://127.0.0.1:9002"),
    )
    assert server._reported[1] == ("tcp://127.0.0.1:9001", "tcp://127.0.0.1:9002")
    # The restarted frontend (pid 222) re-registers: silent reset + accept.
    response = _handle(server, _attach_message(pid=222, listen_port=8100))
    assert response["type"] == "LWD_FRONTEND_ACCEPT"
    assert 1 in server._accepted
    assert server._accepted_pids[1] == 222
    assert server._reported == {}  # stale endpoints are gone
    assert 1 not in server._listen_addresses or server._listen_addresses[1] == (
        "127.0.0.1",
        8100,
    )


def test_same_pid_reregistration_is_a_silent_no_op() -> None:
    """Watchdog rounds from the live frontend must not disturb the
    registration state the barrier already collected."""
    server = _server()
    _handle(server, _attach_message(pid=111))
    _handle(
        server,
        _endpoints_message("tcp://127.0.0.1:9001", "tcp://127.0.0.1:9002"),
    )
    response = _handle(server, _attach_message(pid=111))
    assert response["type"] == "LWD_FRONTEND_ACCEPT"
    # The barrier's collected endpoints survive the round.
    assert server._reported[1] == ("tcp://127.0.0.1:9001", "tcp://127.0.0.1:9002")
    assert server._accepted_pids[1] == 111


def test_endpoints_re_report_replaces_the_entry() -> None:
    """A live frontend re-reporting (a restarted primary's fresh barrier is
    repopulated by exactly this) replaces instead of being rejected."""
    server = _server()
    _handle(server, _attach_message(pid=111))
    _handle(
        server,
        _endpoints_message("tcp://127.0.0.1:9001", "tcp://127.0.0.1:9002"),
    )
    response = _handle(
        server,
        _endpoints_message("tcp://127.0.0.1:9101", "tcp://127.0.0.1:9102"),
    )
    assert response["type"] == "LWD_FRONTEND_BARRIER"
    assert server._reported[1] == ("tcp://127.0.0.1:9101", "tcp://127.0.0.1:9102")


def test_config_mismatch_is_still_rejected_on_reregistration() -> None:
    """The reset path must not weaken the config-consistency check."""
    server = _server()
    _handle(server, _attach_message(pid=111))
    restart = _attach_message(pid=222)
    restart["config_payload"] = {"k": "different"}
    response = _handle(server, restart)
    assert response["type"] == "LWD_FRONTEND_ERROR"
    assert "mismatch" in response["reason"]


def test_serve_background_thread_is_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    """The resident service thread must be a daemon (process exit kills it)."""
    import threading

    server = _server()
    polls: list = []

    class _Socket:
        def poll(self, timeout_ms: int) -> int:
            polls.append(timeout_ms)
            return 0  # idle: no events

    server._socket = _Socket()
    server.serve_background()
    thread = next(
        t for t in threading.enumerate() if t.name == "lwd-frontend-attach-server"
    )
    assert thread.daemon
    assert polls  # the resident loop is polling
    server._closed = True
