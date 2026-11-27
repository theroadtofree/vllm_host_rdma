# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression for the attach handshake's argument helpers.

On the new main ``jsonify_non_default_args`` / ``get_non_default_args`` live
in ``vllm.entrypoints.common.startup``; ``serve.utils.api_utils`` no longer
exports them. The old import path raised ImportError inside
``compute_attach_config_payload``, which ``client.attach()`` calls on the
main attach path.
"""

from __future__ import annotations

import ast
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from vllm.entrypoints.openai.lwd import frontend_attach
from vllm.entrypoints.openai.lwd.frontend_attach import FrontendAttachServer

_HELPERS = frozenset({"jsonify_non_default_args", "get_non_default_args"})


@pytest.fixture
def startup_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Stand in for the external startup module (arg parsing / CLI defaults)."""
    module = cast(Any, types.ModuleType("vllm.entrypoints.common.startup"))
    module.jsonify_non_default_args = lambda args, *, exclude=None: {
        key: value
        for key, value in vars(args).items()
        if exclude is None or key not in exclude
    }
    module.get_non_default_args = lambda args: dict(vars(args))
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.common.startup", module)
    return module


def test_attach_helpers_import_from_common_startup() -> None:
    """The three lazy imports must resolve to the module that owns them."""
    tree = ast.parse(Path(frontend_attach.__file__).read_text(encoding="utf-8"))
    modules = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and any(alias.name in _HELPERS for alias in node.names)
    ]
    assert modules == ["vllm.entrypoints.common.startup"] * 3


def test_attach_payload_and_inherited_args_resolve_the_helpers(
    startup_module: types.ModuleType,
) -> None:
    args = SimpleNamespace(model="m", port=8000)
    plan = SimpleNamespace(
        frontend_digest="digest-1",
        additional_config={"lwd_config": {"path": "/tmp/x", "foo": 1}},
    )

    payload = frontend_attach.compute_attach_config_payload(args, plan)
    assert payload["model"] == "m"
    assert payload["lwd_frontend_digest"] == "digest-1"
    lwd = payload["lwd_additional_config"]["lwd_config"]
    assert lwd["instance_id"] == 0
    assert "path" not in lwd

    frontend_payload = frontend_attach.compute_frontend_args_payload(args)
    assert "port" not in frontend_payload

    inherited = frontend_attach.apply_inherited_frontend_args(
        args, {"served_model_name": "alias"}
    )
    assert inherited == ["served_model_name"]
    assert args.served_model_name == "alias"


def test_late_endpoint_report_publishes_into_engine_addresses() -> None:
    """P1-1: an ENDPOINTS report arriving after the startup barrier must
    publish the frontend's reported endpoints into the bound engine
    address table, keeping the authority's view current (a late same-host
    frontend joins through its deterministic band rebind; the engine's
    DEALER reconnect plus the READY re-send complete its handshake)."""
    server: Any = object.__new__(FrontendAttachServer)
    server._accepted = {1}
    server._reported = {}
    server._addresses_lock = threading.Lock()
    server._engine_addresses = None
    addresses = SimpleNamespace(inputs=["", ""], outputs=["", ""])
    server.bind_engine_addresses(addresses)

    server._handle_endpoints(
        {
            "client_index": 1,
            "input_address": "tcp://127.0.0.1:5001",
            "output_address": "tcp://127.0.0.1:5002",
        },
        lambda _reply: None,
    )

    assert addresses.inputs[1] == "tcp://127.0.0.1:5001"
    assert addresses.outputs[1] == "tcp://127.0.0.1:5002"
    assert server._reported[1] == (
        "tcp://127.0.0.1:5001",
        "tcp://127.0.0.1:5002",
    )


def test_watchdog_reports_the_actual_listener_addresses() -> None:
    """P1-2: the re-registration watchdog must be wired to the bound
    ROUTER/PULL listeners (the deterministic engine IO band ports), so a
    restarted primary's fresh barrier learns exactly the addresses the
    running engine is retrying toward."""
    tree = ast.parse(Path(frontend_attach.__file__).read_text(encoding="utf-8"))
    run_fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_attach_api_server"
    )
    calls = [
        node
        for node in ast.walk(run_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_start_reregister_watchdog"
    ]
    assert len(calls) == 1
    assert [ast.unparse(arg) for arg in calls[0].args[1:]] == [
        "input_listener.address",
        "output_listener.address",
    ]
