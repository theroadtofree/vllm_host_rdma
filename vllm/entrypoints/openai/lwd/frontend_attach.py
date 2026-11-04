# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Attach one frontend per LWD edge instance to a shared native engine group.

The primary edge owns the engines and frontend client index zero. Every other
edge starts one ordinary API-server child, binds its native ROUTER/PULL engine
endpoints, and supplies those endpoints to the primary. The primary holds the
``launch_core_engines`` startup barrier until all indices are present, so the
existing engine handshake and ``request.client_index`` output grouping need no
LWD-specific engine changes.

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
import time
from dataclasses import dataclass
from typing import Any

import msgspec
import zmq

import vllm.envs as envs
from vllm.config.lwd import LwdAPIConfig, resolve_lwd_config
from vllm.config.lwd.api import parse_endpoint
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, get_tcp_uri, make_zmq_socket

logger = init_logger(__name__)

_MSG_ATTACH = "LWD_FRONTEND_ATTACH"
_MSG_ACCEPT = "LWD_FRONTEND_ACCEPT"
_MSG_ENDPOINTS = "LWD_FRONTEND_ENDPOINTS"
_MSG_BARRIER = "LWD_FRONTEND_BARRIER"
_MSG_ERROR = "LWD_FRONTEND_ERROR"
_POLL_INTERVAL_S = 1.0

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
    topology_digest: str
    additional_config: dict[str, Any]


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
    if requested_count is not None and requested_count != count:
        raise ValueError(
            f"[LWD] --api-server-count ({requested_count}) must equal the "
            f"edge_share frontend count ({count})"
        )
    mode = "primary" if api.api_server_rpc_port is not None else "attach"
    endpoint = api.api_server_rpc_port or api.api_server_attach
    assert endpoint is not None
    return FrontendAttachPlan(
        mode=mode,
        client_index=lwd.instance_id,
        client_count=count,
        endpoint=endpoint,
        topology_digest=lwd.topology.digest,
        additional_config=_additional_config_dict(args),
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
    from vllm.entrypoints.serve.utils.api_utils import jsonify_non_default_args

    plan = plan or resolve_attach_plan(args)
    assert plan is not None
    payload = jsonify_non_default_args(
        args, exclude=set(_ATTACH_CONFIG_EXCLUDE_KEYS)
    )
    payload["lwd_topology_digest"] = plan.topology_digest
    payload["lwd_additional_config"] = _canonical_additional_config(plan)
    return payload


def compute_frontend_args_payload(args: Any) -> dict[str, Any]:
    from vllm.entrypoints.serve.utils.api_utils import jsonify_non_default_args

    payload = jsonify_non_default_args(args)
    return {
        key: value for key, value in payload.items() if key in _FRONTEND_INHERIT_ARGS
    }


def apply_inherited_frontend_args(
    args: Any, frontend_args_payload: dict[str, Any] | None
) -> list[str]:
    if not frontend_args_payload:
        return []
    from vllm.entrypoints.serve.utils.api_utils import get_non_default_args

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


def _valid_zmq_address(value: object) -> bool:
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
    ):
        host, port = parse_endpoint(endpoint, "--api-server-rpc-port")
        self.endpoint = get_tcp_uri(host, port)
        self.client_count = client_count
        self.config_payload = config_payload
        self.frontend_args_payload = frontend_args or {}
        self.stats_update_address: str | None = None
        self._listen_addresses = {0: main_listen}
        self._accepted: set[int] = set()
        self._reported: dict[int, tuple[str, str]] = {}
        self._ctx = zmq.Context()
        self._socket = make_zmq_socket(
            self._ctx, self.endpoint, zmq.ROUTER, bind=True, linger=0
        )
        self._closed = False

    @property
    def expected_attaches(self) -> int:
        return self.client_count - 1

    def wait_for_registrations(
        self, engine_manager: Any = None, timeout: float | None = None
    ) -> dict[int, tuple[str, str]]:
        timeout = envs.VLLM_ENGINE_READY_TIMEOUT_S if timeout is None else timeout
        deadline = time.monotonic() + timeout
        logger.info(
            "LWD frontend attach: waiting for %d frontend(s) at %s",
            self.expected_attaches,
            self.endpoint,
        )
        while len(self._reported) < self.expected_attaches:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing = sorted(
                    set(range(1, self.client_count)) - set(self._reported)
                )
                raise RuntimeError(
                    f"Timed out waiting for LWD attach frontends; missing "
                    f"client indices: {missing}"
                )
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
            "All LWD attach frontends supplied endpoints: %s",
            {index: self._reported[index] for index in sorted(self._reported)},
        )
        return dict(self._reported)

    def _handle_message(self, identity: bytes, payload: bytes) -> None:
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
            logger.error("Invalid LWD frontend attach message: %s", exc)
            with contextlib.suppress(zmq.ZMQError):
                reply({"type": _MSG_ERROR, "reason": str(exc)})

    def _handle_attach(self, message: dict[str, Any], reply: Any) -> None:
        index = message.get("client_index")
        if not isinstance(index, int) or isinstance(index, bool) or not (
            1 <= index < self.client_count
        ):
            reply(
                {
                    "type": _MSG_ERROR,
                    "reason": f"client_index must be in [1, {self.client_count})",
                }
            )
            return
        if index in self._accepted:
            reply(
                {
                    "type": _MSG_ERROR,
                    "reason": f"client_index {index} is already attached",
                }
            )
            return
        remote_payload = message.get("config_payload")
        if not isinstance(remote_payload, dict):
            reply({"type": _MSG_ERROR, "reason": "missing config_payload"})
            return
        differences = diff_config_payloads(self.config_payload, remote_payload)
        if differences:
            reason = "common configuration mismatch: " + "; ".join(differences)
            logger.error("LWD attach frontend %d rejected: %s", index, reason)
            reply({"type": _MSG_ERROR, "reason": reason})
            return
        listen_host = message.get("listen_host")
        listen_port = message.get("listen_port")
        if not isinstance(listen_host, str) or not listen_host or not isinstance(
            listen_port, int
        ):
            reply({"type": _MSG_ERROR, "reason": "invalid API listen address"})
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
        logger.info(
            "Accepted LWD attach frontend %d (pid %s)", index, message.get("pid")
        )
        reply(
            {
                "type": _MSG_ACCEPT,
                "client_count": self.client_count,
                "stats_update_address": self.stats_update_address,
                "frontend_args": self.frontend_args_payload,
            }
        )

    def _handle_endpoints(self, message: dict[str, Any], reply: Any) -> None:
        index = message.get("client_index")
        if index not in self._accepted:
            reply({"type": _MSG_ERROR, "reason": "frontend has not attached"})
            return
        if index in self._reported:
            reply({"type": _MSG_ERROR, "reason": "endpoints already supplied"})
            return
        input_address = message.get("input_address")
        output_address = message.get("output_address")
        if not _valid_zmq_address(input_address) or not _valid_zmq_address(
            output_address
        ):
            reply({"type": _MSG_ERROR, "reason": "invalid native ZMQ endpoints"})
            return
        self._reported[index] = (input_address, output_address)
        reply({"type": _MSG_BARRIER, "client_index": index})

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

    def __init__(self, endpoint: str):
        host, port = parse_endpoint(endpoint, "--api-server-attach")
        self.main_address = get_tcp_uri(host, port)
        self._ctx = zmq.Context()
        self._socket = make_zmq_socket(
            self._ctx, self.main_address, zmq.DEALER, bind=False, linger=0
        )
        self._socket.setsockopt(
            zmq.RCVTIMEO, int(envs.VLLM_ENGINE_READY_TIMEOUT_S * 1000)
        )
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
    if client_count != plan.client_count:
        raise ValueError("[LWD] API server count does not match edge_share topology")
    return FrontendAttachServer(
        endpoint=plan.endpoint,
        client_count=client_count,
        config_payload=compute_attach_config_payload(args, plan),
        main_listen=_resolve_listen(args),
        frontend_args=compute_frontend_args_payload(args),
    )


def run_attach_api_server(args: Any) -> None:
    """Run one non-primary API frontend without launching local engines."""
    plan = resolve_attach_plan(args)
    assert plan is not None and plan.mode == "attach"
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
                "Inherited serving args from LWD primary: %s",
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
        bind_ip = get_route_ip(parse_endpoint(plan.endpoint, "--api-server-attach")[0])
        manager = APIServerProcessManager(
            listen_address=listen_address,
            sock=sock,
            args=args,
            num_servers=1,
            input_addresses=[get_tcp_uri(bind_ip, 0)],
            output_addresses=[get_tcp_uri(bind_ip, 0)],
            stats_update_address=accepted.get("stats_update_address"),
            client_count=client_count,
            client_indices=[plan.client_index],
        )
        try:
            inputs, outputs = manager.gather_actual_addresses()
            client.report_endpoints(plan.client_index, inputs[0], outputs[0])
            wait_for_completion_or_failure(api_server_manager=manager)
        finally:
            manager.shutdown()


def lwd_dispatch_attach_frontend(args: Any) -> bool:
    """Dispatch an attach command, or apply primary frontend-count defaults."""
    plan = resolve_attach_plan(args)
    if plan is None:
        return False
    if plan.mode == "attach":
        run_attach_api_server(args)
        return True
    args.api_server_count = plan.client_count
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
    reported = registration_server.wait_for_registrations(engine_manager)
    for client_index, (input_address, output_address) in reported.items():
        addresses.inputs[client_index] = input_address
        addresses.outputs[client_index] = output_address


def attach_scope(registration_server: FrontendAttachServer | None) -> Any:
    if registration_server is None:
        return contextlib.nullcontext()
    return registration_server
