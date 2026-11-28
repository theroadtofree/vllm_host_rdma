# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Attach one frontend per LWD edge instance to a shared native engine group.

The primary edge owns the engines and frontend client index zero. Every other
edge starts one ordinary API-server child, binds its native ROUTER/PULL engine
endpoints, and supplies those endpoints to the primary. The primary holds the
``launch_core_engines`` startup barrier for a bounded grace window, then
proceeds with whichever frontends registered (elastic partial start: the
engine's frontend list is pre-seeded with the deterministic attach addresses,
so a frontend that starts later binds the same address and joins through the
transparent reconnection), so the existing engine handshake and
``request.client_index`` output grouping need no LWD-specific engine changes.

Wire messages are msgpack dictionaries over ZMQ ROUTER/DEALER:

* ``LWD_FRONTEND_ATTACH`` -> ``LWD_FRONTEND_ACCEPT``
* ``LWD_FRONTEND_ENDPOINTS`` -> ``LWD_FRONTEND_BARRIER``
* failures use ``LWD_FRONTEND_ERROR``
"""

import contextlib
import copy
import json
import os
import signal
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeGuard

import msgspec
import zmq

import vllm.envs as envs
from vllm.config.lwd import LwdAPIConfig, resolve_lwd_config
from vllm.config.lwd.api import parse_endpoint
from vllm.config.lwd.topology import (
    LWD_API_BIND_BASE_PORT_DEFAULT,
    LWD_API_MAX_FRONTENDS,
)
from vllm.logger import init_logger
from vllm.lwd_logger import lwd_logger
from vllm.utils.network_utils import (
    get_ip,
    get_tcp_uri,
    make_zmq_listener,
    make_zmq_socket,
)

logger = init_logger(__name__)

_MSG_ATTACH = "LWD_FRONTEND_ATTACH"
_MSG_ACCEPT = "LWD_FRONTEND_ACCEPT"
_MSG_ENDPOINTS = "LWD_FRONTEND_ENDPOINTS"
_MSG_BARRIER = "LWD_FRONTEND_BARRIER"
_MSG_ERROR = "LWD_FRONTEND_ERROR"
_POLL_INTERVAL_S = 1.0
# Re-register watchdog cadence: while the primary lives each round is a
# silent no-op; after a primary restart the next round releases its startup
# barrier (old-branch invariant: barrier for startup, resident rounds for
# runtime).
_REREGISTER_INTERVAL_S = 10.0
_REREGISTER_TIMEOUT_S = 5.0
# Startup grace window for attach frontends (elastic partial start). The
# engine processes are already loading while the barrier waits, and a
# model load takes minutes, so this window normally costs no wall-clock
# time; after it the primary proceeds with the registered subset and a
# late-starting frontend joins via its deterministic engine IO address.
_ATTACH_GRACE_S = 60.0
# Captured by run_attach_api_server for the re-register watchdog: the args
# namespace and listen address are stable for the process lifetime.
_WATCHDOG_ARGS: Any = None
_WATCHDOG_LISTEN: tuple[str, int] = ("0.0.0.0", 0)

# Engine-affecting options explicitly set by an attach command must agree with
# the primary. Listen, HTTP transport, attachment plumbing, and unused engine
# rendezvous placement are intentionally local to each command.
_ATTACH_CONFIG_EXCLUDE_KEYS = frozenset(
    {
        "host",
        "port",
        "uds",
        "api_server_count",
        "api_server_rpc_port",
        "api_server_attach",
        "additional_config",
        "master_addr",
        "master_port",
        "data_parallel_address",
        "data_parallel_rpc_port",
        "uvicorn_log_level",
        "disable_uvicorn_access_log",
        "disable_access_log_for_endpoints",
        "allow_credentials",
        "allowed_origins",
        "allowed_methods",
        "allowed_headers",
        "ssl_keyfile",
        "ssl_certfile",
        "ssl_ca_certs",
        "enable_ssl_refresh",
        "ssl_cert_reqs",
        "ssl_ciphers",
        "root_path",
        "middleware",
        "enable_request_id_headers",
        "disable_fastapi_docs",
        "h11_max_incomplete_event_size",
        "h11_max_header_count",
        "enable_offline_docs",
    }
)

_FRONTEND_INHERIT_ARGS = frozenset(
    {
        "served_model_name",
        "chat_template",
        "chat_template_content_format",
        "trust_request_chat_template",
        "default_chat_template_kwargs",
        "response_role",
        "return_tokens_as_token_ids",
        "enable_auto_tool_choice",
        "tool_call_parser",
        "tool_parser_plugin",
        "exclude_tools_when_tool_choice_none",
        "reasoning_parser",
        "enable_log_requests",
        "enable_log_outputs",
        "enable_log_deltas",
        "max_log_len",
        "enable_prompt_tokens_details",
        "enable_force_include_usage",
        "fingerprint_mode",
        "fingerprint_value",
        "api_key",
        "enable_tokenizer_info_endpoint",
        "tokens_only",
    }
)

_WILDCARD_LISTEN_HOSTS = frozenset({"0.0.0.0", "::"})


@dataclass(frozen=True)
class FrontendAttachPlan:
    mode: str
    client_index: int
    client_count: int
    endpoint: str
    api_bind_base_port: int
    topology_digest: str
    """Full topology digest, observability only (the authority logs it)."""
    frontend_digest: str
    """The compared digest: stable across elastic additions (it covers
    everything except the instance inventory), so a frontend launched
    with the grown yaml still attaches to the running primary."""
    additional_config: dict[str, Any]
    host_comm: bool
    """Whether the primary must run the attach pipeline: the serve
    dispatch routes an attach primary through ``run_multi_api_server``
    for host comm even at ``client_count == 1``; device-comm deployments
    keep the native single-server path."""


def _additional_config_dict(args: Any) -> dict[str, Any]:
    additional = getattr(args, "additional_config", None)
    if isinstance(additional, str):
        additional = json.loads(additional)
    return additional if isinstance(additional, dict) else {}


def resolve_attach_plan(args: Any) -> FrontendAttachPlan | None:
    """Resolve frontend index/count from the same topology used by engines."""
    api = LwdAPIConfig.from_namespace(args)
    if not api.requested:
        return None
    if envs.VLLM_USE_RUST_FRONTEND:
        raise ValueError("[LWD] API attachment supports Python HTTP frontends only")
    lwd = resolve_lwd_config(_additional_config_dict(args), api=api)
    assert lwd is not None
    count = len(lwd.topology.edges)
    requested_count = getattr(args, "api_server_count", None)
    # The host-comm primary sets the reserved band capacity
    # programmatically (lwd_dispatch_attach_frontend): that value is
    # not a user override, and the later re-resolves (the registration
    # authority, the attach watchdog) must accept it.
    capacity_value = (
        lwd.topology.is_host_comm and requested_count == LWD_API_MAX_FRONTENDS
    )
    if requested_count is not None and requested_count != count and not capacity_value:
        raise ValueError(
            f"[LWD] --api-server-count ({requested_count}) must equal "
            f"the edge_share frontend count ({count})"
        )
    mode = "primary" if api.api_server_rpc_port is not None else "attach"
    endpoint = api.api_server_rpc_port or api.api_server_attach
    assert endpoint is not None
    return FrontendAttachPlan(
        mode=mode,
        client_index=lwd.instance_id,
        client_count=count,
        endpoint=endpoint,
        api_bind_base_port=lwd.topology.deployment.api_bind_base_port,
        topology_digest=lwd.topology.digest,
        frontend_digest=lwd.topology.frontend_digest,
        additional_config=_additional_config_dict(args),
        host_comm=lwd.topology.is_host_comm,
    )


def _canonical_additional_config(plan: FrontendAttachPlan) -> dict[str, Any]:
    """Ignore per-command identity/path but retain all other LWD settings."""
    additional = copy.deepcopy(plan.additional_config)
    lwd = additional.get("lwd_config")
    if isinstance(lwd, dict):
        lwd.pop("path", None)
        lwd["instance_id"] = 0
    return additional


def compute_attach_config_payload(
    args: Any, plan: FrontendAttachPlan | None = None
) -> dict[str, Any]:
    from vllm.entrypoints.common.startup import jsonify_non_default_args

    plan = plan or resolve_attach_plan(args)
    assert plan is not None
    payload = jsonify_non_default_args(args, exclude=set(_ATTACH_CONFIG_EXCLUDE_KEYS))
    payload["lwd_frontend_digest"] = plan.frontend_digest
    payload["lwd_additional_config"] = _canonical_additional_config(plan)
    return payload


def compute_frontend_args_payload(args: Any) -> dict[str, Any]:
    from vllm.entrypoints.common.startup import jsonify_non_default_args

    payload = jsonify_non_default_args(args)
    return {
        key: value for key, value in payload.items() if key in _FRONTEND_INHERIT_ARGS
    }


def apply_inherited_frontend_args(
    args: Any, frontend_args_payload: dict[str, Any] | None
) -> list[str]:
    if not frontend_args_payload:
        return []
    from vllm.entrypoints.common.startup import get_non_default_args

    locally_set = set(get_non_default_args(args))
    inherited = []
    for key, value in frontend_args_payload.items():
        if key in _FRONTEND_INHERIT_ARGS and key not in locally_set:
            setattr(args, key, value)
            inherited.append(key)
    return inherited


def _short_value(key: str, value: object) -> str:
    if any(word in key.lower() for word in ("key", "token", "secret", "password")):
        return "'***'"
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def diff_config_payloads(
    primary_payload: dict[str, Any], attach_payload: dict[str, Any]
) -> list[str]:
    """Report explicit attach options incompatible with the primary command."""
    differences = []
    for key in sorted(attach_payload):
        if key not in primary_payload:
            differences.append(
                f"only on attach command: "
                f"{key}={_short_value(key, attach_payload[key])}"
            )
        elif attach_payload[key] != primary_payload[key]:
            differences.append(
                f"value mismatch: {key}: "
                f"primary={_short_value(key, primary_payload[key])} "
                f"attach={_short_value(key, attach_payload[key])}"
            )
    return differences


def get_route_ip(target_ip: str) -> str:
    """Return the local source IP used to reach the primary endpoint."""
    family = socket.AF_INET6 if ":" in target_ip else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_DGRAM) as sock:
            sock.connect((target_ip, 1))
            local_ip = sock.getsockname()[0]
            if local_ip:
                return local_ip
    except OSError:
        pass
    return get_ip()


def engine_io_addresses(
    endpoint: str, base_port: int, client_index: int
) -> tuple[str, str]:
    """Deterministic native engine IO endpoints of one attach frontend:
    ``api_bind_base_port + 2*idx`` (input ROUTER) and ``+1`` (output
    PULL) from the topology yaml's reserved band, on the route IP toward
    the registration endpoint (the same-host edge_share model).

    Single source of truth for both sides: the attach command binds
    these, and the primary pre-seeds the engine's frontend list with
    them for an elastic partial start, so the two derivations can never
    drift. A frontend that starts after the engine handshakes binds the
    same address and the engine's DEALER/PUSH reconnect transparently.

    The band capacity check lives HERE — where ports are actually
    occupied — not at config time: the bind side, the pre-seed and the
    registration authority all derive through this function, so an
    oversized plan fails loudly at startup instead of silently running
    past the reserved band."""
    if client_index >= LWD_API_MAX_FRONTENDS:
        band = _engine_io_port_band(base_port)
        raise ValueError(
            f"[LWD] attach frontend {client_index} has no reserved engine "
            f"IO port in the band {band.start}-{band.stop - 1} "
            f"(deployment.api_bind_base_port={base_port} holds at most "
            f"{LWD_API_MAX_FRONTENDS} frontends)"
        )
    bind_ip = get_route_ip(parse_endpoint(endpoint, "--api-server-attach")[0])
    engine_io_base = base_port + 2 * client_index
    return (
        get_tcp_uri(bind_ip, engine_io_base),
        get_tcp_uri(bind_ip, engine_io_base + 1),
    )


def _engine_io_port_band(base_port: int) -> range:
    """The reserved engine IO band: two ports per frontend, for up to
    LWD_API_MAX_FRONTENDS frontends — the spec reserves the whole band,
    not just the configured slice."""
    return range(base_port, base_port + 2 * LWD_API_MAX_FRONTENDS)


def _assert_ports_outside_engine_io_band(
    base_port: int, ports: dict[str, int], owner: str
) -> None:
    """Startup side of the bidirectional conflict check: this command's
    own configured ports must stay out of the reserved engine IO band the
    frontends bind into (a collision is at best an EADDRINUSE crash, at
    worst a wrong-service connection). The authority side re-checks every
    registration's listen port."""
    band = _engine_io_port_band(base_port)
    hits = {name: port for name, port in ports.items() if port in band}
    if hits:
        raise ValueError(
            f"[LWD] {owner} port(s) {hits} fall inside the reserved engine "
            f"IO band {band.start}-{band.stop - 1} "
            f"(deployment.api_bind_base_port={base_port}); move the "
            "conflicting port or the base"
        )


def _resolve_listen(args: Any) -> tuple[str, int]:
    return getattr(args, "host", None) or "0.0.0.0", args.port


def _listen_addresses_conflict(
    host_a: str, port_a: int, host_b: str, port_b: int
) -> bool:
    if port_a != port_b:
        return False
    if host_a in _WILDCARD_LISTEN_HOSTS or host_b in _WILDCARD_LISTEN_HOSTS:
        return True
    return host_a == host_b


def _valid_zmq_address(value: object) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and value.startswith(("tcp://", "ipc://"))
        and not value.endswith(":0")
    )


class FrontendAttachServer:
    """Primary-side attachment authority and pre-handshake barrier."""

    def __init__(
        self,
        endpoint: str,
        client_count: int,
        config_payload: dict[str, Any],
        main_listen: tuple[str, int],
        frontend_args: dict[str, Any] | None = None,
        api_bind_base_port: int = LWD_API_BIND_BASE_PORT_DEFAULT,
        expected_attaches: int | None = None,
    ) -> None:
        host, port = parse_endpoint(endpoint, "--api-server-rpc-port")
        self.endpoint = get_tcp_uri(host, port)
        # Raw "host:port" form: the elastic pre-seed derives every attach
        # frontend's deterministic engine IO addresses from it.
        self.raw_endpoint = endpoint
        # Reserved engine IO band base (deployment.api_bind_base_port): the
        # pre-seed and the bidirectional registration-side conflict check
        # derive from the same value the attach frontends bind.
        self.api_bind_base_port = api_bind_base_port
        # Capacity, not the yaml plan: the number of frontend slots the
        # primary provisions (registration index bounds, pre-seeded engine
        # IO addresses). The yaml plan governs which instances the engine
        # routes; the barrier waits only for the planned remote frontends.
        self.client_count = client_count
        self.expected_attaches = (
            client_count - 1 if expected_attaches is None else expected_attaches
        )
        self.config_payload = config_payload
        self.frontend_args_payload = frontend_args or {}
        self.stats_update_address: str | None = None
        self._listen_addresses = {0: main_listen}
        self._accepted: set[int] = set()
        # client_index -> pid of the live attached process: distinguishes a
        # restarted frontend (new pid -> reset stale state) from the
        # watchdog's routine re-registration (same pid -> silent no-op).
        self._accepted_pids: dict[int, Any] = {}
        self._reported: dict[int, tuple[str, str]] = {}
        # The engine's resolved frontend address table. The resident service
        # publishes every ENDPOINTS report into it, including reports that
        # arrive after the startup barrier (a late frontend or a watchdog
        # round after a primary restart). None until wait_for_attach_
        # registrations binds it; the startup path also consumes _reported.
        self._engine_addresses: Any = None
        self._addresses_lock = threading.Lock()
        self._ctx = zmq.Context()
        self._socket = make_zmq_socket(
            self._ctx, self.endpoint, zmq.ROUTER, bind=True, linger=0
        )
        self._closed = False

    def bind_engine_addresses(self, addresses: Any) -> None:
        """Give the resident service the engine's resolved address table.

        Must be called before the startup barrier: ENDPOINTS messages handled
        inside ``wait_for_registrations`` then reach the engine too, and the
        same path serves every registration that arrives afterwards.
        """
        self._engine_addresses = addresses

    def _publish_endpoints(
        self, index: int, input_address: str, output_address: str
    ) -> None:
        """Record and publish one frontend's actual ZMQ endpoints.

        ``EngineZmqAddresses.inputs`` / ``outputs`` are lists whose item
        assignment is atomic under the GIL, so the engine reading the same
        table cannot observe a torn reference. A late or restarted frontend
        therefore updates the engine in place instead of requiring a restart.
        """
        with self._addresses_lock:
            self._reported[index] = (input_address, output_address)
            addresses = self._engine_addresses
            if addresses is not None:
                addresses.inputs[index] = input_address
                addresses.outputs[index] = output_address

    def wait_for_registrations(
        self, engine_manager: Any = None, timeout: float | None = None
    ) -> dict[int, tuple[str, str]]:
        """Soft startup barrier: collect endpoint reports for up to
        ``timeout`` seconds (the grace window), then proceed with whoever
        reported.

        Elastic partial start: the primary must serve with the frontends
        that exist — the engine's frontend list is pre-seeded with the
        deterministic attach addresses (see
        ``wait_for_attach_registrations``), so an index that never starts
        costs nothing, and one that starts later binds the same address
        and joins through the transparent reconnection. A registration
        that lands inside the window still upgrades the seed to the
        frontend's actual endpoints. Engine process death during the
        window remains a startup failure."""
        timeout = _ATTACH_GRACE_S if timeout is None else timeout
        deadline = time.monotonic() + timeout
        logger.info(
            "[LWD][frontend-attach] waiting up to %.0fs for %d frontend(s) at %s",
            timeout,
            self.expected_attaches,
            self.endpoint,
        )
        while len(self._reported) < self.expected_attaches:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing = sorted(set(range(1, self.client_count)) - set(self._reported))
                logger.warning(
                    "[LWD][frontend-attach] grace window elapsed; proceeding "
                    "with the registered frontends (elastic partial start). "
                    "Missing client indices %s will serve once they start and "
                    "bind their deterministic engine IO addresses: %s",
                    missing,
                    {
                        index: engine_io_addresses(
                            self.raw_endpoint, self.api_bind_base_port, index
                        )
                        for index in missing
                    },
                )
                return dict(self._reported)
            events = self._socket.poll(min(_POLL_INTERVAL_S, remaining) * 1000)
            while events:
                identity, payload = self._socket.recv_multipart()
                self._handle_message(identity, payload)
                events = self._socket.poll(0)
            if engine_manager is not None and (dead := engine_manager.finished_procs()):
                raise RuntimeError(
                    f"Engine process(es) {dead} exited while waiting for "
                    "LWD attach frontends"
                )
        logger.info(
            "[LWD][frontend-attach] all frontends supplied endpoints: %s",
            {index: self._reported[index] for index in sorted(self._reported)},
        )
        return dict(self._reported)

    def _handle_message(
        self,
        identity: bytes,
        payload: bytes,
        reply: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Handle one decoded attach-protocol message. ``reply`` is
        injectable so tests can drive the handler against a server built
        with ``object.__new__`` (no real socket)."""
        if reply is None:

            def reply(message: dict[str, Any]) -> None:
                self._socket.send_multipart([identity, msgspec.msgpack.encode(message)])

        try:
            message = msgspec.msgpack.decode(payload)
            if not isinstance(message, dict):
                raise ValueError("message body must be a dictionary")
            message_type = message.get("type")
            if message_type == _MSG_ATTACH:
                self._handle_attach(message, reply)
            elif message_type == _MSG_ENDPOINTS:
                self._handle_endpoints(message, reply)
            else:
                reply(
                    {
                        "type": _MSG_ERROR,
                        "reason": f"unknown attach message type: {message_type!r}",
                    }
                )
        except Exception as exc:  # Keep the authority alive for other frontends.
            logger.error("[LWD][frontend-attach] invalid message: %s", exc)
            with contextlib.suppress(zmq.ZMQError):
                reply({"type": _MSG_ERROR, "reason": str(exc)})

    def _handle_attach(
        self,
        message: dict[str, Any],
        reply: Callable[[dict[str, Any]], None],
    ) -> None:
        index = message.get("client_index")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not (1 <= index < self.client_count)
        ):
            reply(
                {
                    "type": _MSG_ERROR,
                    "reason": f"client_index must be in [1, {self.client_count})",
                }
            )
            return
        if index in self._accepted:
            if message.get("pid") == self._accepted_pids.get(index):
                # Routine watchdog round from the same live process: the
                # primary is healthy, nothing changed — silent no-op.
                reply(
                    {
                        "type": _MSG_ACCEPT,
                        "client_count": self.client_count,
                        "stats_update_address": self.stats_update_address,
                        "frontend_args": self.frontend_args_payload,
                    }
                )
                return
            # A restarted attach frontend re-registers with the same index:
            # the old process is gone, so reset its stale state instead of
            # starving the re-registration (its listen address and reported
            # endpoints belonged to the dead process).
            lwd_logger.info(
                "LWD attach frontend %d restarted (pid %s -> %s); "
                "resetting prior state",
                index,
                self._accepted_pids.get(index),
                message.get("pid"),
            )
            self._accepted.discard(index)
            self._reported.pop(index, None)
            self._listen_addresses.pop(index, None)
        remote_payload = message.get("config_payload")
        if not isinstance(remote_payload, dict):
            reply({"type": _MSG_ERROR, "reason": "missing config_payload"})
            return
        differences = diff_config_payloads(self.config_payload, remote_payload)
        if differences:
            reason = "common configuration mismatch: " + "; ".join(differences)
            logger.error(
                "[LWD][frontend-attach] frontend %d rejected: %s", index, reason
            )
            reply({"type": _MSG_ERROR, "reason": reason})
            return
        listen_host = message.get("listen_host")
        listen_port = message.get("listen_port")
        if (
            not isinstance(listen_host, str)
            or not listen_host
            or not isinstance(listen_port, int)
        ):
            reply({"type": _MSG_ERROR, "reason": "invalid API listen address"})
            return
        if listen_port in _engine_io_port_band(self.api_bind_base_port):
            # Bidirectional conflict check, authority side: a registered
            # frontend's HTTP listen must stay out of the band its engine
            # IO pair binds into.
            band = _engine_io_port_band(self.api_bind_base_port)
            reply(
                {
                    "type": _MSG_ERROR,
                    "reason": (
                        f"API listen port {listen_port} for client {index} "
                        f"falls inside the reserved engine IO band "
                        f"{band.start}-{band.stop - 1}; move --port or "
                        "deployment.api_bind_base_port"
                    ),
                }
            )
            return
        for other_index, (other_host, other_port) in self._listen_addresses.items():
            if _listen_addresses_conflict(
                listen_host, listen_port, other_host, other_port
            ):
                reason = (
                    f"API listen address {listen_host}:{listen_port} for client "
                    f"{index} conflicts with client {other_index} "
                    f"({other_host}:{other_port})"
                )
                reply({"type": _MSG_ERROR, "reason": reason})
                return
        self._listen_addresses[index] = (listen_host, listen_port)
        self._accepted.add(index)
        self._accepted_pids[index] = message.get("pid")
        logger.info(
            "[LWD][frontend-attach] accepted frontend %d (pid %s)",
            index,
            message.get("pid"),
        )
        reply(
            {
                "type": _MSG_ACCEPT,
                "client_count": self.client_count,
                "stats_update_address": self.stats_update_address,
                "frontend_args": self.frontend_args_payload,
            }
        )

    def _handle_endpoints(
        self,
        message: dict[str, Any],
        reply: Callable[[dict[str, Any]], None],
    ) -> None:
        index = message.get("client_index")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index not in self._accepted
        ):
            reply({"type": _MSG_ERROR, "reason": "frontend has not attached"})
            return
        if index in self._reported:
            # Re-report from a live frontend (a restarted primary's fresh
            # barrier is repopulated by this): replace the stale entry
            # instead of rejecting — the endpoints are stable while the
            # frontend lives.
            logger.debug(
                "LWD attach frontend %d re-reported endpoints; replacing",
                index,
            )
        input_address = message.get("input_address")
        output_address = message.get("output_address")
        if not _valid_zmq_address(input_address) or not _valid_zmq_address(
            output_address
        ):
            reply({"type": _MSG_ERROR, "reason": "invalid native ZMQ endpoints"})
            return
        self._publish_endpoints(index, input_address, output_address)
        reply({"type": _MSG_BARRIER, "client_index": index})

    def serve_background(self) -> None:
        """Answer (re)registrations for the rest of the process lifetime.

        The startup barrier only covers launch: an attach frontend
        restarted at runtime must be able to re-register, which a closed
        server would starve (barrier for startup, resident service for
        runtime)."""
        if self._closed:
            return
        threading.Thread(
            target=self._serve_forever,
            name="lwd-frontend-attach-server",
            daemon=True,
        ).start()

    def _serve_forever(self) -> None:
        while not self._closed:
            try:
                events = self._socket.poll(_POLL_INTERVAL_S * 1000)
            except zmq.ZMQError:
                return  # closed at shutdown
            while events:
                identity, payload = self._socket.recv_multipart()
                self._handle_message(identity, payload)
                events = self._socket.poll(0)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._socket.close(linger=0)
        finally:
            self._ctx.term()

    def __enter__(self) -> "FrontendAttachServer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class FrontendAttachClient:
    """Non-primary frontend's half of the native endpoint barrier."""

    def __init__(self, endpoint: str, recv_timeout_s: float | None = None) -> None:
        host, port = parse_endpoint(endpoint, "--api-server-attach")
        self.main_address = get_tcp_uri(host, port)
        self._ctx = zmq.Context()
        self._socket = make_zmq_socket(
            self._ctx, self.main_address, zmq.DEALER, bind=False, linger=0
        )
        timeout: float
        if recv_timeout_s is None:
            timeout = envs.VLLM_ENGINE_READY_TIMEOUT_S
        else:
            timeout = recv_timeout_s
        self._socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
        self._closed = False

    def _roundtrip(self, message: dict[str, Any]) -> dict[str, Any]:
        self._socket.send_multipart([msgspec.msgpack.encode(message)])
        try:
            response = msgspec.msgpack.decode(self._socket.recv_multipart()[-1])
        except zmq.Again:
            raise RuntimeError(
                f"No response from LWD primary frontend at {self.main_address} "
                f"within {envs.VLLM_ENGINE_READY_TIMEOUT_S:.0f}s"
            ) from None
        if not isinstance(response, dict):
            raise RuntimeError("Malformed response from LWD primary frontend")
        if response.get("type") == _MSG_ERROR:
            raise RuntimeError(
                f"LWD primary frontend rejected attachment: {response.get('reason')}"
            )
        return response

    def attach(
        self,
        client_index: int,
        config_payload: dict[str, Any],
        listen: tuple[str, int],
    ) -> dict[str, Any]:
        response = self._roundtrip(
            {
                "type": _MSG_ATTACH,
                "client_index": client_index,
                "config_payload": config_payload,
                "listen_host": listen[0],
                "listen_port": listen[1],
                "pid": os.getpid(),
            }
        )
        if response.get("type") != _MSG_ACCEPT:
            raise RuntimeError(f"Unexpected LWD attach response: {response!r}")
        return response

    def report_endpoints(
        self, client_index: int, input_address: str, output_address: str
    ) -> None:
        response = self._roundtrip(
            {
                "type": _MSG_ENDPOINTS,
                "client_index": client_index,
                "input_address": input_address,
                "output_address": output_address,
            }
        )
        if response.get("type") != _MSG_BARRIER:
            raise RuntimeError(f"Unexpected LWD endpoint response: {response!r}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._socket.close(linger=0)
        finally:
            self._ctx.term()

    def __enter__(self) -> "FrontendAttachClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def create_frontend_registration_server(
    args: Any, client_count: int
) -> FrontendAttachServer:
    plan = resolve_attach_plan(args)
    assert plan is not None and plan.mode == "primary"
    # client_count is the reserved frontend capacity: it must cover the
    # yaml plan and stay within the engine IO band. Frontends beyond the
    # plan attach into the spare capacity; only the planned remote
    # frontends gate the startup barrier.
    if not plan.client_count <= client_count <= LWD_API_MAX_FRONTENDS:
        raise ValueError(
            f"[LWD] API server count ({client_count}) must cover the "
            f"edge_share frontend plan ({plan.client_count}) and stay "
            f"within the reserved engine IO band ({LWD_API_MAX_FRONTENDS})"
        )
    # Bidirectional conflict check, authority side: the primary's own
    # configured ports must stay out of the band its frontends bind into.
    ports = {
        "api listen": _resolve_listen(args)[1],
        "registration": parse_endpoint(plan.endpoint, "--api-server-rpc-port")[1],
    }
    for name in ("master_port", "data_parallel_rpc_port"):
        if (value := getattr(args, name, None)) is not None:
            ports[name] = int(value)
    _assert_ports_outside_engine_io_band(plan.api_bind_base_port, ports, "primary")
    lwd_logger.info(
        "LWD registration authority: frontend digest %s (topology %s)",
        plan.frontend_digest,
        plan.topology_digest,
    )
    return FrontendAttachServer(
        endpoint=plan.endpoint,
        client_count=client_count,
        config_payload=compute_attach_config_payload(args, plan),
        main_listen=_resolve_listen(args),
        frontend_args=compute_frontend_args_payload(args),
        api_bind_base_port=plan.api_bind_base_port,
        expected_attaches=plan.client_count - 1,
    )


def run_attach_api_server(args: Any) -> None:
    """Run one non-primary API frontend without launching local engines."""
    plan = resolve_attach_plan(args)
    assert plan is not None and plan.mode == "attach"
    # Watchdog rounds run after this function's scope: capture the config
    # payload and listen address now (both are stable for the process
    # lifetime — the payload derives from CLI args, the listen from args).
    global _WATCHDOG_ARGS, _WATCHDOG_LISTEN
    _WATCHDOG_ARGS = args
    _WATCHDOG_LISTEN = _resolve_listen(args)
    with FrontendAttachClient(plan.endpoint) as client:
        accepted = client.attach(
            plan.client_index,
            compute_attach_config_payload(args, plan),
            _resolve_listen(args),
        )
        client_count = accepted["client_count"]
        args.api_server_count = client_count
        inherited = apply_inherited_frontend_args(args, accepted.get("frontend_args"))
        if inherited:
            logger.info(
                "[LWD][frontend-attach] inherited serving args: %s",
                ", ".join(sorted(inherited)),
            )

        shutdown_requested = False

        def signal_handler(signum: int, frame: Any) -> None:
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        from vllm.entrypoints.launchers.api_server.entry import setup_server
        from vllm.v1.utils import (
            APIServerProcessManager,
            wait_for_completion_or_failure,
        )

        listen_address, sock = setup_server(args, reuse_port=False)
        # Deterministic engine IO ports (old-branch invariant): the ports
        # come from the topology yaml's reserved band (single source:
        # ``engine_io_addresses``, shared with the primary's elastic
        # pre-seed). A late or restarted attach frontend rebinds the SAME
        # port, so the running engine's retrying DEALER/PUSH reconnect
        # transparently and the READY re-send completes this frontend's
        # client handshake — an ephemeral port would leave the engine
        # connected to the pre-seeded address that nobody ever binds.
        # Bind-side conflict check: this command's own configured ports
        # must stay out of the band it is about to bind.
        _assert_ports_outside_engine_io_band(
            plan.api_bind_base_port,
            {
                "api listen": _resolve_listen(args)[1],
                "registration": parse_endpoint(plan.endpoint, "--api-server-attach")[1],
            },
            f"attach frontend {plan.client_index}",
        )
        input_address, output_address = engine_io_addresses(
            plan.endpoint, plan.api_bind_base_port, plan.client_index
        )
        input_listener = make_zmq_listener(input_address, zmq.ROUTER)
        try:
            output_listener = make_zmq_listener(output_address, zmq.PULL)
        except BaseException:
            input_listener.close()
            raise
        manager = APIServerProcessManager(
            listen_address=listen_address,
            sock=sock,
            args=args,
            num_servers=1,
            input_listeners=[input_listener],
            output_listeners=[output_listener],
            stats_update_address=accepted.get("stats_update_address"),
            client_count=client_count,
            client_indices=[plan.client_index],
        )
        try:
            client.report_endpoints(
                plan.client_index,
                input_listener.address,
                output_listener.address,
            )
            # The listeners bound the deterministic band ports, so the
            # reported addresses are the actual endpoints AND the addresses
            # the engine is already retrying toward; the watchdog re-reports
            # the same pair, releasing a restarted primary's fresh barrier.
            _start_reregister_watchdog(
                plan, input_listener.address, output_listener.address
            )
            wait_for_completion_or_failure(api_server_manager=manager)
        finally:
            manager.shutdown()


def _start_reregister_watchdog(
    plan: FrontendAttachPlan,
    input_address: str,
    output_address: str,
) -> None:
    """Primary-restart self-healing (old-branch invariant): re-register on
    a fixed cadence with an independent short connection per round. While
    the primary lives each round is a silent no-op; after a primary restart
    the next round re-attaches and re-reports, releasing the new primary's
    startup barrier. Runs until the process exits (daemon)."""

    def loop() -> None:
        while True:
            time.sleep(_REREGISTER_INTERVAL_S)
            try:
                with FrontendAttachClient(
                    plan.endpoint, recv_timeout_s=_REREGISTER_TIMEOUT_S
                ) as client:
                    client.attach(
                        plan.client_index,
                        compute_attach_config_payload(_resolve_watchdog_args(), plan),
                        _resolve_watchdog_listen(),
                    )
                    client.report_endpoints(
                        plan.client_index, input_address, output_address
                    )
            except Exception:
                logger.debug(
                    "LWD attach re-registration round failed (primary down?); "
                    "retrying next tick",
                    exc_info=True,
                )

    threading.Thread(target=loop, name="lwd-attach-reregister", daemon=True).start()


def _resolve_watchdog_args() -> Any:
    return _WATCHDOG_ARGS


def _resolve_watchdog_listen() -> tuple[str, int]:
    return _WATCHDOG_LISTEN


def lwd_dispatch_attach_frontend(args: Any) -> bool:
    """Dispatch an attach command, or apply primary frontend-count defaults."""
    plan = resolve_attach_plan(args)
    if plan is None:
        return False
    if plan.mode == "attach":
        run_attach_api_server(args)
        return True
    # Frontend capacity is the reserved engine IO band, not the yaml
    # plan: a host-comm primary provisions all frontend slots up front,
    # so a frontend beyond the current yaml plan can attach without a
    # primary restart (its requests stay gate-rejected until the plan
    # that launched the engine includes its instance). The capacity
    # count also routes the primary through run_multi_api_server, home
    # of the registration server the single in-process path never
    # builds; device-comm primaries keep the plan count and may take
    # the native single-server path.
    args.api_server_count = (
        LWD_API_MAX_FRONTENDS if plan.host_comm else plan.client_count
    )
    args.lwd_attach_primary = plan.host_comm
    return False


def prepare_attach_mode(
    args: Any, num_api_servers: int, is_ray_dp: bool
) -> tuple[bool, int, FrontendAttachServer | None]:
    if getattr(args, "api_server_rpc_port", None) is None:
        return False, num_api_servers, None
    if is_ray_dp:
        raise ValueError("[LWD] API attachment does not support the Ray DP backend")
    return True, 1, create_frontend_registration_server(args, num_api_servers)


def bind_attach_stats(
    registration_server: FrontendAttachServer | None,
    stats_update_address: str | None,
    tensor_queue: Any,
) -> None:
    if registration_server is None:
        return
    if tensor_queue is not None:
        raise NotImplementedError(
            "[LWD] API attachment does not support multiprocess multimodal "
            "tensor queues"
        )
    registration_server.stats_update_address = stats_update_address


def wait_for_attach_registrations(
    registration_server: FrontendAttachServer,
    addresses: Any,
    engine_manager: Any,
) -> None:
    """Elastic partial start: seed every capacity slot of the reserved
    engine IO band with its deterministic address, then let the
    grace-window barrier upgrade the planned slots that actually
    registered to their reported endpoints.

    The engine therefore starts with the FULL capacity frontend list and
    serves with whichever frontends exist; a frontend that starts later
    binds the same deterministic address and the engine's DEALER/PUSH
    reconnect transparently (the READY re-send completes its client
    handshake). The seed derives the address from THIS host's route
    toward the registration endpoint — exact for the same-host
    edge_share deployment (one engine host, N API frontends); a
    cross-host attach frontend must register inside the grace window to
    contribute its actual address."""
    registration_server.bind_engine_addresses(addresses)
    for client_index in range(1, registration_server.client_count):
        seeded_input, seeded_output = engine_io_addresses(
            registration_server.raw_endpoint,
            registration_server.api_bind_base_port,
            client_index,
        )
        addresses.inputs[client_index] = seeded_input
        addresses.outputs[client_index] = seeded_output
    reported = registration_server.wait_for_registrations(engine_manager)
    # Idempotent: ENDPOINTS handled inside the barrier already published
    # through _publish_endpoints. This loop also serves deployments that call
    # wait_for_registrations() without a bound address table.
    for client_index, (input_address, output_address) in reported.items():
        addresses.inputs[client_index] = input_address
        addresses.outputs[client_index] = output_address
    # The barrier covered startup; runtime re-registrations (a restarted
    # attach frontend, or a restarted primary's watchdog rounds) are served
    # by the resident thread.
    registration_server.serve_background()


def attach_scope(registration_server: FrontendAttachServer | None) -> Any:
    """The registration server outlives the STARTUP with-block: the scope's
    exit runs as soon as the startup barrier collects (the serve loop runs
    outside the with), and closing there left nothing listening on the
    registration endpoint — an attach frontend restarted at runtime hung
    forever on its ATTACH reply. The resident `serve_background` thread
    owns the socket for the rest of the process lifetime (old-branch
    invariant: barrier for startup, resident service for runtime; the
    daemon thread and process exit release it)."""
    del registration_server
    return contextlib.nullcontext()
