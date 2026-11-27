# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attach-runtime self-healing of the frontend registration server
(peer-restart line, phase 3).

A restarted attach frontend re-registers with the same client index: the
server must reset the dead process's stale state instead of rejecting
it, and a resident service must keep answering after the startup barrier
(the barrier covers launch; runtime re-registrations are the watchdog's
silent rounds or a restarted frontend's fresh attach). Endpoints
re-report replaces the stale entry.

Elastic partial start (yaml plans N frontends, fewer are launched): the
grace-window barrier proceeds with the registered subset instead of
failing, and the engine's frontend list is pre-seeded with the
deterministic attach addresses so a late-starting frontend joins
through the transparent reconnection.
"""

from __future__ import annotations

import argparse
from typing import Any

import pytest
import yaml

import vllm.entrypoints.openai.lwd.frontend_attach as frontend_attach
from vllm.config.lwd.topology import LWD_API_MAX_FRONTENDS
from vllm.entrypoints.openai.lwd.frontend_attach import (
    FrontendAttachServer,
    lwd_dispatch_attach_frontend,
    resolve_attach_plan,
    wait_for_attach_registrations,
)


def _server() -> FrontendAttachServer:
    server = object.__new__(FrontendAttachServer)
    server.client_count = 2
    server.expected_attaches = 1
    server.config_payload = {"k": "v"}
    server.frontend_args_payload = {}
    server.stats_update_address = None
    server._listen_addresses = {0: ("127.0.0.1", 8000)}
    server._accepted = set()
    server._accepted_pids = {}
    server._reported = {}
    server._closed = False
    server.endpoint = "tcp://127.0.0.1:23657"
    server.raw_endpoint = "127.0.0.1:23657"
    server.api_bind_base_port = 61000
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


def test_attach_scope_does_not_close_the_registration_server() -> None:
    """Regression (seventh incident): the serve with-block exits as soon as
    the startup barrier collects — the scope must not close the server on
    exit, or nothing listens on the registration endpoint and a restarted
    attach frontend hangs forever on its ATTACH reply."""
    from vllm.entrypoints.openai.lwd.frontend_attach import attach_scope

    server = _server()
    with attach_scope(server):
        pass
    assert not server._closed


class _IdleSocket:
    """Socket stub: an idle poller that never delivers events."""

    def poll(self, timeout_ms: int) -> int:
        return 0


class _Addresses:
    def __init__(self, count: int) -> None:
        self.inputs = [f"tcp://127.0.0.1:{10000 + i}" for i in range(count)]
        self.outputs = [f"tcp://127.0.0.1:{11000 + i}" for i in range(count)]


def test_grace_window_proceeds_with_registered_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Elastic partial start (yaml plans 2, only instance 0 launched): the
    grace window elapses without frontend 1 and the barrier returns the
    registered subset instead of failing startup."""
    monkeypatch.setattr(frontend_attach, "_ATTACH_GRACE_S", 0.05)
    server = _server()
    server._socket = _IdleSocket()
    reported = server.wait_for_registrations()
    assert reported == {}
    assert server._closed is False


def test_partial_start_seeds_deterministic_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine's frontend list is pre-seeded with the deterministic
    attach addresses, so the missing frontend's slot still carries a
    bindable address and a late-starting instance joins through it."""
    monkeypatch.setattr(frontend_attach, "_ATTACH_GRACE_S", 0.05)
    monkeypatch.setattr(frontend_attach, "get_route_ip", lambda host: "127.0.0.1")
    server = _server()
    server._socket = _IdleSocket()
    server._closed = True  # resident service is not exercised here
    addresses = _Addresses(count=2)
    wait_for_attach_registrations(server, addresses, engine_manager=None)
    assert addresses.inputs[1] == "tcp://127.0.0.1:61002"
    assert addresses.outputs[1] == "tcp://127.0.0.1:61003"


def test_registration_within_grace_upgrades_the_seeded_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frontend that registers inside the grace window contributes its
    actual endpoints; the deterministic seed is only the fallback for
    whoever is missing."""
    server = _server()
    server._socket = _IdleSocket()
    server._closed = True
    # As if frontend 1 registered before the barrier was entered.
    server._reported[1] = ("tcp://10.1.1.5:9001", "tcp://10.1.1.5:9002")
    addresses = _Addresses(count=2)
    wait_for_attach_registrations(server, addresses, engine_manager=None)
    assert addresses.inputs[1] == "tcp://10.1.1.5:9001"
    assert addresses.outputs[1] == "tcp://10.1.1.5:9002"


def test_engine_io_addresses_derivation() -> None:
    """The bind-side and pre-seed-side single source: port arithmetic is
    api_bind_base_port + 2*idx / +1 on the route IP toward the endpoint."""
    input_address, output_address = frontend_attach.engine_io_addresses(
        "127.0.0.1:23657", base_port=61000, client_index=1
    )
    # The host is the local route IP toward the endpoint; only the port
    # arithmetic is asserted here.
    assert input_address.rpartition(":")[2] == "61002"
    assert output_address.rpartition(":")[2] == "61003"


def test_engine_io_addresses_caps_at_the_reserved_band() -> None:
    """The band holds two ports per frontend for at most 8 frontends. The
    cap is enforced where ports are OCCUPIED (the derivation itself — the
    bind side, the pre-seed and the registration authority all pass
    through it), not at config time: the yaml may describe larger plans,
    and an oversized one fails loudly at startup instead of silently
    running past the band."""
    input_address, output_address = frontend_attach.engine_io_addresses(
        "127.0.0.1:23657", base_port=61000, client_index=7
    )
    assert input_address.rpartition(":")[2] == "61014"
    assert output_address.rpartition(":")[2] == "61015"
    with pytest.raises(ValueError, match="reserved engine"):
        frontend_attach.engine_io_addresses(
            "127.0.0.1:23657", base_port=61000, client_index=8
        )


def test_engine_io_band_conflict_fails_the_command_loudly() -> None:
    """Bind-side check: a configured port inside the reserved band is a
    startup error naming the band, not a runtime EADDRINUSE."""
    with pytest.raises(ValueError, match="reserved engine IO band"):
        frontend_attach._assert_ports_outside_engine_io_band(
            61000, {"api listen": 61003}, "attach frontend 1"
        )
    # Ports outside the band pass.
    frontend_attach._assert_ports_outside_engine_io_band(
        61000, {"api listen": 8000, "registration": 23657}, "attach frontend 1"
    )


def test_registration_rejects_a_listen_port_inside_the_band() -> None:
    """Authority-side check: a frontend registering an HTTP listen port
    inside the reserved engine IO band is rejected with a pointed
    error; the default 8100 stays fine."""
    server = _server()
    inside = _attach_message(pid=111, listen_port=61002)
    response = _handle(server, inside)
    assert response["type"] == "LWD_FRONTEND_ERROR"
    assert "reserved engine IO band" in response["reason"]
    assert _handle(server, _attach_message(pid=111))["type"] == "LWD_FRONTEND_ACCEPT"


def _one_pair_edge_share_yaml(path: Any, *, host_comm: bool) -> None:
    """A single-pair edge_share topology; device comm adds the rank
    allocations host comm does not use."""
    deployment: dict[str, Any] = {
        "mode": 0,
        "scene": "edge_share",
        "edges_num": 1,
        "clouds_num": 1,
    }
    edge_dp: dict[str, Any] = {"dp_idx": 0, "addr": "135.150.2.2"}
    cloud_dp: dict[str, Any] = {"dp_idx": 0, "addr": "16.32.2.2", "ctrl_port": 5550}
    if host_comm:
        deployment["data_plane_comm"] = "host"
        cloud_dp["data_port"] = 5560
    else:
        deployment["hccl_world_size"] = 2
        edge_dp["ranks"] = [0]
        cloud_dp["ranks"] = [1]
    path.write_text(
        yaml.safe_dump(
            {
                "deployment": deployment,
                "feature_ctrl": {
                    "enable_early_recv": False,
                    "scramble": {
                        "enable": False,
                        "device_id": None,
                        "custom_factor": None,
                    },
                },
                "edges": [{"id": 0, "dp": [edge_dp]}],
                "clouds": [{"id": 0, "dp": [cloud_dp]}],
                "instance_links": [{"edge": 0, "cloud": 0}],
            },
            sort_keys=False,
        )
    )


@pytest.mark.parametrize("host_comm,expected", [(True, True), (False, False)])
def test_one_frontend_primary_flags_the_multi_server_path_by_comm(
    tmp_path: Any, host_comm: bool, expected: bool
) -> None:
    """A single-pair attach primary provisions the full reserved frontend
    capacity on host comm -- api_server_count carries the band capacity,
    so frontends beyond the yaml plan attach without a primary restart
    and the serve dispatch reaches run_multi_api_server (whose
    registration server is the only thing that ever binds
    --api-server-rpc-port). Device comm keeps the plan count and the
    native single-server path."""
    yaml_path = tmp_path / "edge_share.yaml"
    _one_pair_edge_share_yaml(yaml_path, host_comm=host_comm)
    args = argparse.Namespace(
        additional_config={
            "lwd_config": {
                "path": str(yaml_path),
                "role": "edge",
                "instance_id": 0,
            }
        },
        api_server_rpc_port="127.0.0.1:23657",
        api_server_attach=None,
        api_server_count=None,
    )
    assert lwd_dispatch_attach_frontend(args) is False
    assert args.api_server_count == (LWD_API_MAX_FRONTENDS if host_comm else 1)
    assert args.lwd_attach_primary is expected
    if host_comm:
        # The registration authority re-resolves the plan AFTER the
        # dispatch set the capacity value; that re-resolve must accept
        # the programmatic count (the regression: "--api-server-count
        # (8) must equal the edge_share frontend count (1)").
        replan = resolve_attach_plan(args)
        assert replan is not None and replan.client_count == 1
    # A genuinely user-passed wrong count still fails fast.
    args.api_server_count = 3
    with pytest.raises(ValueError, match="must equal"):
        resolve_attach_plan(args)


def test_authority_accepts_attach_beyond_the_plan_within_capacity() -> None:
    """Capacity, not the plan, bounds attach indices: a primary whose yaml
    plans one frontend still accepts client_index 1 (the previous
    rejection 'client_index must be in [1, 1)'); the band capacity is the
    bound, and index 0 or >= capacity stays rejected."""
    server = _server()
    server.client_count = LWD_API_MAX_FRONTENDS
    server.expected_attaches = 0
    assert _handle(server, _attach_message(pid=111))["type"] == "LWD_FRONTEND_ACCEPT"
    beyond = _attach_message(pid=112)
    beyond["client_index"] = LWD_API_MAX_FRONTENDS
    response = _handle(server, beyond)
    assert response["type"] == "LWD_FRONTEND_ERROR"
    assert f"client_index must be in [1, {LWD_API_MAX_FRONTENDS})" in response["reason"]


def test_the_attach_matches_on_the_frontend_digest_not_the_inventory() -> None:
    """Item 11: the config diff compares `lwd_frontend_digest` — stable
    across elastic additions, so a frontend launched with the grown yaml
    attaches to the primary that started with the smaller one; a
    genuinely different deployment (different digest) still refuses with
    the key named in the reason."""
    server = _server()
    server.config_payload = {"lwd_frontend_digest": "fd-stable"}
    same = _attach_message(pid=111)
    same["config_payload"] = {"lwd_frontend_digest": "fd-stable"}
    assert _handle(server, same)["type"] == "LWD_FRONTEND_ACCEPT"
    drifted = _attach_message(pid=112)
    drifted["config_payload"] = {"lwd_frontend_digest": "fd-different"}
    response = _handle(server, drifted)
    assert response["type"] == "LWD_FRONTEND_ERROR"
    assert "lwd_frontend_digest" in response["reason"]
    assert "fd-stable" in response["reason"] and "fd-different" in response["reason"]
