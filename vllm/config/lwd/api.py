# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CLI and topology validation for multi-frontend API attachment."""

import argparse
from dataclasses import dataclass
from ipaddress import ip_address
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .entry import LwdConfig


def _endpoint(value: str | None, flag: str) -> None:
    if value is not None:
        parse_endpoint(value, flag)


def parse_endpoint(value: str, flag: str) -> tuple[str, int]:
    """Return the concrete IP and port from an LWD API endpoint."""
    if not isinstance(value, str):
        raise ValueError(f"[LWD] {flag} requires IP:port")
    host, sep, port = value.rpartition(":")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    elif ":" in host:
        raise ValueError(f"[LWD] {flag}: IPv6 must use [IP]:port")
    if not sep or not port.isascii() or not port.isdecimal():
        raise ValueError(f"[LWD] {flag} requires IP:port")
    if not 1 <= int(port) <= 65535:
        raise ValueError(f"[LWD] {flag}: port must be in [1, 65535]")
    try:
        addr = ip_address(host)
    except ValueError as exc:
        raise ValueError(f"[LWD] {flag}: invalid IP {host!r}") from exc
    if addr.is_unspecified or addr.is_multicast:
        raise ValueError(f"[LWD] {flag}: use a concrete unicast IP")
    return host, int(port)


@dataclass(frozen=True)
class LwdAPIConfig:
    """LWD primary/attach frontend endpoint configuration."""

    api_server_rpc_port: str | None = None
    """Primary endpoint as IP:port, despite the historical option name."""
    api_server_attach: str | None = None
    """Primary endpoint referenced by a non-primary edge frontend."""

    @property
    def requested(self) -> bool:
        return (
            self.api_server_rpc_port is not None or self.api_server_attach is not None
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("LWD API configuration")
        for flag, action in (
            ("rpc-port", "accept non-primary frontends on the primary edge"),
            ("attach", "attach this non-primary frontend to the primary edge"),
        ):
            group.add_argument(
                f"--api-server-{flag}",
                metavar="IP:port",
                help=f"LWD multi-frontend mode: {action}.",
            )

    @staticmethod
    def validate_frontend(args: Any, *, headless: bool = False) -> None:
        requested = any(
            getattr(args, name, None) is not None
            for name in ("api_server_rpc_port", "api_server_attach")
        )
        if requested and (
            headless or getattr(args, "headless", False) or getattr(args, "grpc", False)
        ):
            raise ValueError("[LWD] API options require the HTTP frontend")
        if requested and getattr(args, "data_parallel_multi_port_external_lb", False):
            raise ValueError(
                "[LWD] API attachment cannot be combined with "
                "--data-parallel-multi-port-external-lb"
            )
        if requested and (count := getattr(args, "api_server_count", None)) is not None:
            if count < 1:
                raise ValueError("[LWD] --api-server-count must be positive")

    @classmethod
    def from_namespace(cls, args: Any, *, headless: bool = False) -> "LwdAPIConfig":
        cls.validate_frontend(args, headless=headless)
        return cls(
            api_server_rpc_port=getattr(args, "api_server_rpc_port", None),
            api_server_attach=getattr(args, "api_server_attach", None),
        )

    def validate_topology(self, lwd: "LwdConfig | None") -> None:
        if not self.requested:
            return
        if self.api_server_rpc_port is not None and self.api_server_attach is not None:
            raise ValueError(
                "[LWD] --api-server-rpc-port and --api-server-attach "
                "are mutually exclusive"
            )
        _endpoint(self.api_server_rpc_port, "--api-server-rpc-port")
        _endpoint(self.api_server_attach, "--api-server-attach")
        if lwd is None:
            raise ValueError("[LWD] API options require lwd_config.path")
        if lwd.role != "edge":
            raise ValueError("[LWD] API options apply to edge instances only")
        if self.api_server_rpc_port is not None and lwd.instance_id != 0:
            raise ValueError("[LWD] --api-server-rpc-port requires edge instance_id=0")
        if self.api_server_attach is not None and lwd.instance_id == 0:
            raise ValueError("[LWD] --api-server-attach requires a non-primary edge")
        if lwd.topology.deployment.scene != "edge_share":
            raise ValueError("[LWD] API attachment requires scene=edge_share")
