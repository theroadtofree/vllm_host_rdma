# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LWD multi-instance frontend attach support (one frontend per cloud
instance).

The registration protocol lives here in vLLM (it is platform-agnostic:
plain ZMQ + msgpack over ``make_zmq_socket``), so the attach feature does
not depend on the Ascend platform runtime. The two entrypoints are wired
into ``vllm.entrypoints.cli.serve``:

* ``run_attach_api_server`` - runs one attach command
  (``--api-server-attach``): exactly one ApiServer with an independent
  ``--host/--port`` listen, no EngineCore processes, no rendezvous.
* ``create_frontend_registration_server`` - creates the main command's
  registration server (``--api-server-rpc-port``) used inside
  ``run_multi_api_server``'s attach-mode orchestration.

Topology:

* One **main** command (``--api-server-rpc-port <port>``) starts the
  EngineCore process group and instance 0's ApiServer frontend, then acts
  as the registration authority: it binds a ZMQ ROUTER on
  ``tcp://*:<rpc_port>`` and waits until the frontends of instances
  ``1..N-1`` - each started by its own attach command
  (``--api-server-attach <ip:port>``) - have registered.
* Each **attach** command starts exactly one ApiServer (independent
  ``--host/--port`` listen), binds its ZMQ ROUTER (engine input) and PULL
  (engine output) on a locally allocated endpoint, and registers
  ``(client_index, input_address, output_address)`` with the main command.

Only after all N frontends have registered does the main command release
the engine handshake (``launch_core_engines`` with-exit), so every engine
learns the complete frontend address list in its init message.

Wire protocol (msgpack over ZMQ ROUTER/DEALER)::

    attach -> main   REGISTER  {client_index, config_payload,
                                listen_host, listen_port, pid}
    main  -> attach  GRANT     {client_count, stats_update_address,
                                frontend_args}                    (or ERROR)
    attach -> main   ADDRESSES {client_index, input_address, output_address}
    main  -> attach  ACK       {client_index}                        (or ERROR)
"""

import contextlib
import ipaddress
import json
import os
import signal
import socket
import time
from typing import Any

import msgspec
import vllm.envs as envs
import zmq
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, get_tcp_uri, make_zmq_socket

logger = init_logger(__name__)

# CLI args that legitimately differ between the main command and an attach
# command and are therefore excluded from the common-config comparison:
# - 前端监听/实例区分:host/port/uds/instance index;
# - attach 自身管道:api-server-rpc-port / api-server-attach;
# - 引擎侧 rendezvous/DP 网络地址:纯前端 attach 命令不参与,要求逐字
#   一致纯属部署摩擦(master_addr/master_port/data_parallel_*);
# - HTTP 传输层(ssl/CORS/middleware/uvicorn/h11 等)每个前端独立
#   监听、独立配置,stay per-frontend。
ATTACH_CONFIG_EXCLUDE_KEYS = frozenset(
    {
        # frontend listen / instance identity
        "host",
        "port",
        "uds",
        "instance_parallel_start_rank",
        # attach plumbing
        "api_server_rpc_port",
        "api_server_attach",
        # engine rendezvous / DP network placement (inert on frontend)
        "master_addr",
        "master_port",
        "data_parallel_address",
        "data_parallel_rpc_port",
        # HTTP transport / per-frontend
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

# Serving-layer (frontend-local) args that each ApiServer reads from its
# own CLI namespace; the main command forwards its values in the GRANT so
# an attach command that omits them inherits the same API surface.
# ``additional_config`` 也走继承:attach 命令可不写,写了则须与主命令一致
# (见 ATTACH_CONFIG_EXCLUDE_KEYS 中未将其排除,故仍参与 diff)。
FRONTEND_INHERIT_ARGS = frozenset(
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
        "additional_config",
    }
)

_REGISTRATION_POLL_INTERVAL_S = 1.0
_DIFF_VALUE_MAX_LEN = 60

_MSG_REGISTER = "REGISTER"
_MSG_GRANT = "GRANT"
_MSG_ADDRESSES = "ADDRESSES"
_MSG_ACK = "ACK"
_MSG_ERROR = "ERROR"


def _is_lwd_enabled(args) -> bool:
    """Best-effort detection of LWD (edge-cloud) enablement from raw CLI args."""
    additional = getattr(args, "additional_config", None)
    if isinstance(additional, str):
        try:
            additional = json.loads(additional)
        except (TypeError, ValueError):
            additional = None
    return bool((additional or {}).get("lwd_config", {}).get("enabled", False))


def _parse_ip_port(value: str, arg_name: str) -> tuple[str, int]:
    """Validate a ``<ip>:<port>`` endpoint string and return (ip, port).

    IPv6 addresses must be bracketed (``[::1]:29550``).
    """
    host, sep, port_str = value.rpartition(":")
    if not sep or not host:
        raise ValueError(
            f"Error: {arg_name} must be of the form <ip>:<port>, got {value!r}"
        )
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    try:
        port = int(port_str)
    except ValueError:
        raise ValueError(
            f"Error: {arg_name} port must be an integer, got {port_str!r}"
        ) from None
    if not 1 <= port <= 65535:
        raise ValueError(f"Error: {arg_name} port must be in [1, 65535], got {port}")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(
            f"Error: {arg_name} host must be a valid IP address, got {host!r}"
        ) from None
    if ip.is_unspecified:
        raise ValueError(
            f"Error: {arg_name} host must be a specific routable IP address, "
            f"got {host!r} (0.0.0.0/:: only make sense as a --host bind "
            f"address, not as a connect endpoint)"
        )
    return host, port


def validate_frontend_attach_args(args) -> None:
    """Fail-fast checks for LWD multi-instance frontend args.

    Covers --instance-parallel-size / --instance-parallel-start-rank and
    the one-instance-one-endpoint attach form (--api-server-rpc-port on
    the main command, --api-server-attach on the N-1 attach commands).
    """
    size = getattr(args, "instance_parallel_size", None)
    size = 1 if size is None else size
    start_rank = getattr(args, "instance_parallel_start_rank", None)
    rpc_port = getattr(args, "api_server_rpc_port", None)
    attach = getattr(args, "api_server_attach", None)

    if size < 1:
        raise ValueError(f"Error: --instance-parallel-size must be >= 1, got {size}")

    if start_rank is not None and not 0 <= start_rank < size:
        raise ValueError(
            f"Error: --instance-parallel-start-rank ({start_rank}) must be "
            f"in [0, --instance-parallel-size={size})"
        )

    # --instance-parallel-* 只在 LWD 模式有效;attach 命令(纯前端、无
    # worker/engine)本身即 LWD 多实例语义,不要求额外写 --additional-config。
    if (
        attach is None
        and (size > 1 or start_rank is not None)
        and not _is_lwd_enabled(args)
    ):
        raise ValueError(
            "Error: --instance-parallel-size/--instance-parallel-start-rank "
            "require LWD (edge-cloud) mode"
        )

    if rpc_port is not None and attach is not None:
        raise ValueError(
            "Error: --api-server-rpc-port (main command) and "
            "--api-server-attach (attach command) are mutually exclusive"
        )

    if rpc_port is not None or attach is not None:
        if getattr(args, "headless", False):
            raise ValueError(
                "Error: --api-server-rpc-port/--api-server-attach cannot be "
                "used with --headless (attach mode requires frontends)"
            )
        if getattr(args, "grpc", False):
            raise ValueError(
                "Error: --api-server-rpc-port/--api-server-attach only "
                "support the HTTP OpenAI API server, not --grpc"
            )
        if getattr(args, "data_parallel_multi_port_external_lb", False):
            raise ValueError(
                "Error: --api-server-rpc-port/--api-server-attach cannot be "
                "combined with --data-parallel-multi-port-external-lb "
                "(attach frontends listen independently; no supervisor "
                "port sharing)"
            )

    if attach is not None:
        if size <= 1:
            raise ValueError(
                "Error: --api-server-attach requires --instance-parallel-size > 1"
            )
        if start_rank is None or not 1 <= start_rank < size:
            raise ValueError(
                f"Error: --api-server-attach requires "
                f"--instance-parallel-start-rank in [1, {size}) (instance 0's "
                f"frontend is started by the main command)"
            )
        _parse_ip_port(attach, "--api-server-attach")
    elif rpc_port is not None:
        if size <= 1:
            raise ValueError(
                "Error: --api-server-rpc-port requires --instance-parallel-size > 1"
            )
        api_server_count = getattr(args, "api_server_count", None)
        if api_server_count is not None and api_server_count != size:
            raise ValueError(
                f"Error: --api-server-count ({api_server_count}) must equal "
                f"--instance-parallel-size ({size}) when --api-server-rpc-port "
                "is set: the count is the total number of frontends across "
                "the main command and the attach commands"
            )
        if not 1 <= rpc_port <= 65535:
            raise ValueError(
                f"Error: --api-server-rpc-port must be in [1, 65535], got {rpc_port}"
            )


def compute_attach_config_payload(args) -> dict[str, Any]:
    """Non-default CLI args an attach command must agree on with the main
    command (fail-fast on mismatch)."""
    from vllm.entrypoints.serve.utils.api_utils import jsonify_non_default_args

    return jsonify_non_default_args(args, exclude=set(ATTACH_CONFIG_EXCLUDE_KEYS))


def compute_frontend_args_payload(args) -> dict[str, Any]:
    """Non-default serving-layer args the main command forwards to attach
    commands in the GRANT."""
    from vllm.entrypoints.serve.utils.api_utils import jsonify_non_default_args

    payload = jsonify_non_default_args(args)
    return {key: value for key, value in payload.items() if key in FRONTEND_INHERIT_ARGS}


def apply_inherited_frontend_args(args, frontend_args_payload) -> list[str]:
    """Set on ``args`` each whitelisted serving-layer arg the attach command
    left at default, using the main command's value from the GRANT."""
    if not frontend_args_payload:
        return []
    from vllm.entrypoints.serve.utils.api_utils import get_non_default_args

    locally_set = set(get_non_default_args(args))
    inherited: list[str] = []
    for key, value in frontend_args_payload.items():
        if key in FRONTEND_INHERIT_ARGS and key not in locally_set:
            setattr(args, key, value)
            inherited.append(key)
    return inherited


def _short_repr(value: object) -> str:
    text = repr(value)
    if len(text) > _DIFF_VALUE_MAX_LEN:
        text = text[: _DIFF_VALUE_MAX_LEN - 3] + "..."
    return text


def diff_config_payloads(main_payload: dict[str, Any], attach_payload: dict[str, Any]) -> list[str]:
    """One-directional diff of an attach command's config payload against
    the main command's. Empty list means the attach command is compatible."""
    lines: list[str] = []
    for key in sorted(attach_payload):
        if key not in main_payload:
            lines.append(f"only on attach command: {key}={_short_repr(attach_payload[key])}")
        elif attach_payload[key] != main_payload[key]:
            lines.append(
                f"value mismatch: {key}: main={_short_repr(main_payload[key])} "
                f"attach={_short_repr(attach_payload[key])}"
            )
    return lines


def get_route_ip(target_ip: str) -> str:
    """Local source IP used to reach ``target_ip`` (no packet is sent)."""
    for family in (socket.AF_INET, socket.AF_INET6):
        is_v6 = ":" in target_ip
        if (family == socket.AF_INET) == is_v6:
            continue
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as s:
                s.connect((target_ip, 1))
                local_ip = s.getsockname()[0]
                if local_ip:
                    return local_ip
        except OSError:
            continue
    return get_ip()


def _recv_dict(socket_: zmq.Socket) -> dict:
    frames = socket_.recv_multipart()
    msg = msgspec.msgpack.decode(frames[-1])
    if not isinstance(msg, dict):
        raise RuntimeError(f"Malformed registration message: {msg!r}")
    return msg


def _send_dict(socket_: zmq.Socket, msg: dict) -> None:
    socket_.send_multipart([msgspec.msgpack.encode(msg)])


def _is_valid_zmq_addr(addr: object) -> bool:
    return isinstance(addr, str) and addr.startswith(("tcp://", "ipc://")) and not addr.endswith(":0")


_WILDCARD_LISTEN_HOSTS = frozenset({"0.0.0.0", "::"})


def _resolve_listen_host(host: str | None) -> str:
    return "0.0.0.0" if not host else host


def _listen_addresses_conflict(
    host_a: str, port_a: int, host_b: str, port_b: int
) -> bool:
    if port_a != port_b:
        return False
    if host_a in _WILDCARD_LISTEN_HOSTS or host_b in _WILDCARD_LISTEN_HOSTS:
        return True
    return host_a == host_b


class FrontendRegistrationServer:
    """Main-command side: collect attach-frontend registrations."""

    def __init__(
        self,
        rpc_port: int,
        client_count: int,
        config_payload: dict[str, Any],
        main_listen_host: str,
        main_listen_port: int,
        frontend_args: dict[str, Any] | None = None,
    ):
        self.rpc_port = rpc_port
        self.client_count = client_count
        self.config_payload = config_payload
        self._listen_addresses: dict[int, tuple[str, int]] = {
            0: (main_listen_host, main_listen_port)
        }
        self.frontend_args_payload = frontend_args or {}
        self.stats_update_address: str | None = None

        self._ctx = zmq.Context()
        self._socket = make_zmq_socket(self._ctx, f"tcp://*:{rpc_port}", zmq.ROUTER, bind=True)
        self._granted: set[int] = set()
        self._reported: dict[int, tuple[str, str]] = {}
        self._closed = False

    @property
    def expected_attaches(self) -> int:
        return self.client_count - 1

    def wait_for_registrations(
        self,
        engine_manager=None,
        timeout: float | None = None,
    ) -> dict[int, tuple[str, str]]:
        if timeout is None:
            timeout = envs.VLLM_ENGINE_READY_TIMEOUT_S
        deadline = time.monotonic() + timeout
        logger.info(
            "LWD multi-instance attach mode: waiting for %d attach "
            "frontend(s) to register on port %d (timeout %.0fs)",
            self.expected_attaches,
            self.rpc_port,
            timeout,
        )

        while len(self._reported) < self.expected_attaches:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing = sorted(set(range(1, self.client_count)) - set(self._reported))
                raise RuntimeError(
                    f"Timed out after {timeout:.1f}s waiting for attach "
                    f"frontends to register on port {self.rpc_port}; missing "
                    f"instance indices: {missing}. The main command requires "
                    f"--api-server-attach commands for every instance in "
                    f"[1, {self.client_count}) before the service can start."
                )
            events = self._socket.poll(timeout=min(_REGISTRATION_POLL_INTERVAL_S, remaining) * 1000)
            while events:
                identity, payload = self._socket.recv_multipart()
                self._handle_message(identity, payload)
                events = self._socket.poll(0)

            if engine_manager is not None:
                dead = engine_manager.finished_procs()
                if dead:
                    raise RuntimeError(
                        f"Engine process(es) {dead} exited while waiting "
                        f"for attach frontends to register on port "
                        f"{self.rpc_port}."
                    )

        logger.info(
            "All %d attach frontend(s) registered on port %d: %s",
            self.expected_attaches,
            self.rpc_port,
            {idx: self._reported[idx] for idx in sorted(self._reported)},
        )
        return dict(self._reported)

    def _handle_message(self, identity: bytes, payload: bytes) -> None:
        def reply(msg: dict) -> None:
            self._socket.send_multipart([identity, msgspec.msgpack.encode(msg)])

        try:
            msg = msgspec.msgpack.decode(payload)
            if not isinstance(msg, dict):
                raise RuntimeError(f"malformed message: {msg!r}")
            mtype = msg.get("type")

            if mtype == _MSG_REGISTER:
                self._handle_register(msg, reply)
            elif mtype == _MSG_ADDRESSES:
                self._handle_addresses(msg, reply)
            else:
                reply(
                    {
                        "type": _MSG_ERROR,
                        "reason": f"unknown message type: {mtype!r}",
                    }
                )
        except Exception as e:  # noqa: BLE001 - reply and survive
            logger.error("Invalid registration message: %s", e)
            with contextlib.suppress(zmq.ZMQError):
                reply({"type": _MSG_ERROR, "reason": str(e)})

    def _handle_register(self, msg: dict, reply) -> None:
        idx = msg.get("client_index")
        if not isinstance(idx, int) or not 0 <= idx < self.client_count:
            reply(
                {
                    "type": _MSG_ERROR,
                    "reason": (f"client_index {idx!r} out of range [0, {self.client_count})"),
                }
            )
            return
        if idx == 0:
            reply(
                {
                    "type": _MSG_ERROR,
                    "reason": (
                        "client_index 0 (instance 0's frontend) is started "
                        "by the main command; attach commands must use "
                        "--instance-parallel-start-rank in "
                        f"[1, {self.client_count})"
                    ),
                }
            )
            return
        remote_payload = msg.get("config_payload")
        if not isinstance(remote_payload, dict):
            reply(
                {
                    "type": _MSG_ERROR,
                    "reason": "REGISTER is missing its config_payload",
                }
            )
            return
        diff = diff_config_payloads(self.config_payload, remote_payload)
        if diff:
            reply(
                {
                    "type": _MSG_ERROR,
                    "reason": (
                        "common configuration mismatch: args the attach "
                        "command set explicitly must match the main "
                        "command's (args set only on the main command "
                        "describe the engine and may be omitted on attach "
                        "commands). Differences: " + "; ".join(diff)
                    ),
                }
            )
            logger.error(
                "Attach frontend %d rejected: config mismatch:\n  %s",
                idx,
                "\n  ".join(diff),
            )
            return
        listen_host = msg.get("listen_host")
        listen_port = msg.get("listen_port")
        if not isinstance(listen_host, str) or not listen_host:
            reply(
                {
                    "type": _MSG_ERROR,
                    "reason": "REGISTER is missing its listen_host",
                }
            )
            return
        if not isinstance(listen_port, int):
            reply(
                {
                    "type": _MSG_ERROR,
                    "reason": "REGISTER is missing its listen_port",
                }
            )
            return
        for other_idx, (other_host, other_port) in self._listen_addresses.items():
            if other_idx == idx:
                continue
            if _listen_addresses_conflict(
                listen_host, listen_port, other_host, other_port
            ):
                reply(
                    {
                        "type": _MSG_ERROR,
                        "reason": (
                            f"duplicate API listen address "
                            f"{listen_host}:{listen_port}: instance {idx} "
                            f"collides with instance {other_idx} "
                            f"({other_host}:{other_port})"
                        ),
                    }
                )
                logger.error(
                    "Attach frontend %d rejected: API listen address "
                    "%s:%s collides with instance %d (%s:%s)",
                    idx,
                    listen_host,
                    listen_port,
                    other_idx,
                    other_host,
                    other_port,
                )
                return
        self._listen_addresses[idx] = (listen_host, listen_port)
        self._granted.add(idx)
        logger.info("Attach frontend %d registered (pid %s).", idx, msg.get("pid"))
        reply(
            {
                "type": _MSG_GRANT,
                "client_count": self.client_count,
                "stats_update_address": self.stats_update_address,
                "frontend_args": self.frontend_args_payload,
            }
        )

    def _handle_addresses(self, msg: dict, reply) -> None:
        idx = msg.get("client_index")
        if idx not in self._granted:
            reply(
                {
                    "type": _MSG_ERROR,
                    "reason": f"client_index {idx!r} has not registered",
                }
            )
            return
        if idx in self._reported:
            reply(
                {
                    "type": _MSG_ERROR,
                    "reason": (f"client_index {idx!r} already reported addresses"),
                }
            )
            return
        input_address = msg.get("input_address")
        output_address = msg.get("output_address")
        if not _is_valid_zmq_addr(input_address) or not _is_valid_zmq_addr(output_address):
            reply(
                {
                    "type": _MSG_ERROR,
                    "reason": (f"invalid ZMQ addresses: {input_address!r}, {output_address!r}"),
                }
            )
            return
        self._reported[idx] = (input_address, output_address)
        reply({"type": _MSG_ACK, "client_index": idx})
        logger.info(
            "Attach frontend %d reported addresses: input=%s output=%s",
            idx,
            input_address,
            output_address,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._socket.close(linger=0)
        finally:
            self._ctx.term()

    def __enter__(self) -> "FrontendRegistrationServer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class AttachRegistrationClient:
    """Attach-command side: register one frontend with the main command."""

    def __init__(self, main_ip: str, main_port: int):
        self.main_address = get_tcp_uri(main_ip, main_port)
        timeout_ms = int(envs.VLLM_ENGINE_READY_TIMEOUT_S * 1000)
        self._ctx = zmq.Context()
        self._socket = make_zmq_socket(self._ctx, self.main_address, zmq.DEALER, bind=False)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._closed = False

    def _roundtrip(self, msg: dict) -> dict:
        _send_dict(self._socket, msg)
        try:
            reply = _recv_dict(self._socket)
        except zmq.Again:
            raise RuntimeError(
                f"No response from the main command at {self.main_address} "
                f"within {envs.VLLM_ENGINE_READY_TIMEOUT_S:.0f}s. Verify the "
                "main command (--api-server-rpc-port) is running and "
                "reachable."
            ) from None
        if reply.get("type") == _MSG_ERROR:
            raise RuntimeError(f"Main command at {self.main_address} rejected registration: {reply.get('reason')}")
        return reply

    def register(
        self,
        client_index: int,
        config_payload: dict[str, Any],
        listen_host: str,
        listen_port: int,
    ) -> dict:
        grant = self._roundtrip(
            {
                "type": _MSG_REGISTER,
                "client_index": client_index,
                "config_payload": config_payload,
                "listen_host": listen_host,
                "listen_port": listen_port,
                "pid": os.getpid(),
            }
        )
        if grant.get("type") != _MSG_GRANT:
            raise RuntimeError(f"Unexpected reply to REGISTER: {grant!r}")
        return grant

    def report_addresses(self, client_index: int, input_address: str, output_address: str) -> None:
        ack = self._roundtrip(
            {
                "type": _MSG_ADDRESSES,
                "client_index": client_index,
                "input_address": input_address,
                "output_address": output_address,
            }
        )
        if ack.get("type") != _MSG_ACK:
            raise RuntimeError(f"Unexpected reply to ADDRESSES: {ack!r}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._socket.close(linger=0)
        finally:
            self._ctx.term()


def create_frontend_registration_server(args, client_count: int):
    """Build the main command's registration server from its CLI args."""
    return FrontendRegistrationServer(
        rpc_port=args.api_server_rpc_port,
        client_count=client_count,
        config_payload=compute_attach_config_payload(args),
        main_listen_host=_resolve_listen_host(args.host),
        main_listen_port=args.port,
        frontend_args=compute_frontend_args_payload(args),
    )


def run_attach_api_server(args):
    """Run one LWD multi-instance frontend via ``--api-server-attach``."""
    start_rank = args.instance_parallel_start_rank
    main_ip, main_port = _parse_ip_port(args.api_server_attach, "--api-server-attach")

    # The attach frontend is an edge-side frontend: normalize the topology
    # args so this command's frontend builds the same ParallelConfig as the
    # main command's frontend (attach commands don't carry --node-rank).
    if args.node_rank != 0:
        raise ValueError(
            f"--api-server-attach commands run on the edge node and must use --node-rank 0 (got {args.node_rank})."
        )
    if args.nnodes != 2:
        logger.info(
            "Normalizing --nnodes from %d to 2 for the edge-side attach frontend (LWD edge-cloud roles are fixed).",
            args.nnodes,
        )
        args.nnodes = 2

    registration_client = AttachRegistrationClient(main_ip, main_port)
    grant = registration_client.register(
        start_rank,
        compute_attach_config_payload(args),
        listen_host=_resolve_listen_host(args.host),
        listen_port=args.port,
    )
    client_count = grant["client_count"]
    stats_update_address = grant.get("stats_update_address")
    if args.api_server_count is not None and args.api_server_count != client_count:
        raise ValueError(
            f"--api-server-count ({args.api_server_count}) does not match "
            f"the frontend count granted by the main command "
            f"({client_count}); drop the arg or set it to {client_count}"
        )
    inherited = apply_inherited_frontend_args(args, grant.get("frontend_args"))
    if inherited:
        logger.info(
            "Inherited serving args from the main command: %s",
            ", ".join(sorted(inherited)),
        )
    logger.info(
        "Attach frontend %d granted by main command at %s:%d (frontend count %d).",
        start_rank,
        main_ip,
        main_port,
        client_count,
    )

    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        logger.debug("Received %d signal.", signum)
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    from vllm.entrypoints.openai.api_server import setup_server

    listen_address, sock = setup_server(args)
    bind_ip = get_route_ip(main_ip)
    input_address = get_tcp_uri(bind_ip, 0)
    output_address = get_tcp_uri(bind_ip, 0)

    from vllm.v1.utils import APIServerProcessManager

    api_server_manager = APIServerProcessManager(
        listen_address=listen_address,
        sock=sock,
        args=args,
        num_servers=1,
        input_addresses=[input_address],
        output_addresses=[output_address],
        stats_update_address=stats_update_address,
        client_count=client_count,
        client_indices=[start_rank],
    )

    try:
        from vllm.v1.utils import wait_for_completion_or_failure

        actual_inputs, actual_outputs = api_server_manager.gather_actual_addresses()
        registration_client.report_addresses(start_rank, actual_inputs[0], actual_outputs[0])
        wait_for_completion_or_failure(api_server_manager=api_server_manager)
    finally:
        registration_client.close()
        api_server_manager.shutdown()


def dispatch_attach_mode(args) -> bool:
    """Handle attach command dispatch / main-command attach defaults.

    Returns ``True`` when the command was fully handled (attach command),
    in which case the caller should return immediately.
    """
    if getattr(args, "api_server_attach", None) is not None:
        run_attach_api_server(args)
        return True
    if getattr(args, "api_server_rpc_port", None) is not None:
        if getattr(args, "headless", False):
            raise ValueError(
                "--api-server-rpc-port cannot be used with --headless "
                "(the main command must start instance 0's frontend)."
            )
        if args.api_server_count is None:
            args.api_server_count = args.instance_parallel_size
    return False


def prepare_attach_mode(
    args, num_api_servers: int, is_ray_dp: bool
) -> tuple[bool, int, FrontendRegistrationServer | None]:
    """Resolve main-command attach-mode state.

    Returns ``(attach_mode, num_local_servers, registration_server)``.
    """
    attach_mode = getattr(args, "api_server_rpc_port", None) is not None
    if not attach_mode:
        return False, num_api_servers, None
    if is_ray_dp:
        raise ValueError(
            "--api-server-rpc-port (attach mode) is not supported with the "
            "ray data parallel backend."
        )
    return (
        True,
        1,
        create_frontend_registration_server(args, num_api_servers),
    )


def bind_attach_stats(
    registration_server: FrontendRegistrationServer | None,
    stats_update_address: str | None,
    tensor_queue: Any,
) -> None:
    """Bind the coordinator stats address onto the registration server."""
    if registration_server is None:
        return
    if tensor_queue is not None:
        raise NotImplementedError(
            "--api-server-rpc-port (attach mode) does not "
            "support multimodal tensor queue yet."
        )
    registration_server.stats_update_address = stats_update_address


def wait_for_attach_registrations(
    registration_server: FrontendRegistrationServer,
    addresses: Any,
    engine_manager: Any,
) -> None:
    """Block until all attach frontends registered, then fill ``addresses``."""
    registrations = registration_server.wait_for_registrations(
        engine_manager=engine_manager,
    )
    for client_index, (in_addr, out_addr) in registrations.items():
        addresses.inputs[client_index] = in_addr
        addresses.outputs[client_index] = out_addr


def attach_scope(registration_server: FrontendRegistrationServer | None):
    """Return a context manager that closes ``registration_server`` on exit."""
    if registration_server is None:
        return contextlib.nullcontext()
    return registration_server
