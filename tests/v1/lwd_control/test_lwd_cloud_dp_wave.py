# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A late LWD request must wake its DP peers through the native wave path."""

from __future__ import annotations

import queue
from types import SimpleNamespace
from unittest.mock import Mock

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine import (
    LwdCloudDPEngineCore,
    LwdCloudEngineCore,
)
from vllm.v1.lwd_control.lwd_comm.lwd_messages import LwdRequestNotify


def _engine(engine_cls, *, has_coordinator: bool):
    engine = object.__new__(engine_cls)
    engine._lwd_prompt_lens = {}
    engine._lwd_request_nonces = {}
    engine._lwd_request_seqnos = {}
    engine._lwd_rejected_nonces = {}
    engine._lwd_prefix_enabled = False
    engine.input_queue = queue.Queue()
    engine.output_queue = queue.Queue()
    engine.engines_running = False
    engine.has_coordinator = has_coordinator
    engine.current_wave = 2
    engine.scheduler = SimpleNamespace(
        add_request=Mock(), pause_state=PauseState.UNPAUSED
    )
    request = SimpleNamespace(
        request_id="r",
        use_structured_output=False,
        lwd_prefix_nonce=0,
        pooling_params=None,
        kv_transfer_params=None,
        ec_transfer_params=None,
        abort_immediately=False,
    )
    engine._lwd_build_request = Mock(return_value=request)
    return engine, request


def test_late_dp_request_wakes_peers_through_native_start_wave() -> None:
    engine, request = _engine(LwdCloudDPEngineCore, has_coordinator=True)

    engine._lwd_handle_request_notify(LwdRequestNotify("r", 16))

    # The request notify must not strand the peer by waking only this rank.
    # Processing the native ADD below owns the DP state transition instead.
    assert engine.engines_running is False
    request_type, (added_request, request_wave) = engine.input_queue.get_nowait()
    assert request_type == EngineCoreRequestType.ADD
    assert added_request is request
    assert request_wave == 0

    engine.add_request(added_request, request_wave)

    assert engine.engines_running is True
    engine.scheduler.add_request.assert_called_once_with(request)
    client_index, outputs = engine.output_queue.get_nowait()
    assert client_index == -1
    assert outputs.start_wave == 2


def test_request_notify_keeps_local_wake_without_coordinator() -> None:
    engine, _request = _engine(LwdCloudDPEngineCore, has_coordinator=False)

    engine._lwd_handle_request_notify(LwdRequestNotify("r", 16))

    assert engine.engines_running is True


def test_request_notify_keeps_local_wake_for_independent_engine() -> None:
    engine, _request = _engine(LwdCloudEngineCore, has_coordinator=False)

    engine._lwd_handle_request_notify(LwdRequestNotify("r", 16))

    assert engine.engines_running is True
