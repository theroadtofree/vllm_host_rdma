# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import queue
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest

from vllm.v1.engine import EngineCoreRequestType, FinishReason
from vllm.v1.lwd_control.control_edge_scheduler.lwd_cloud_dp_account import (
    LwdCloudDpAccount,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
    LwdEdgeEngineCore,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_prefix_control import (
    LwdEdgePrefixControl,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdEdgeScheduler,
)
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_NOT_FINISHED,
    LwdC2eNotify,
    LwdCloudParamsNotify,
    LwdPrefixReadyNotify,
    LwdRefillNotify,
)


class _Request(SimpleNamespace):
    def update_block_hashes(self) -> None:
        self.block_hashes = [b"h0", b"h1"]


def _request(request_id: str = "r") -> _Request:
    return _Request(
        request_id=request_id,
        num_prompt_tokens=12,
        num_computed_tokens=0,
        block_hashes=[b"old"],
        _block_hasher=None,
    )


def _control_engine() -> SimpleNamespace:
    return SimpleNamespace(
        scheduler=SimpleNamespace(lwd_edge_prefix_ready=Mock(return_value=True)),
        input_queue=queue.Queue(),
        request_block_hasher=Mock(),
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(prefix_caching_hash_algo="builtin")
        ),
        _lwd_admit_and_notify=Mock(),
        _lwd_enqueue_request=Mock(),
        _lwd_abort_internally=Mock(),
        abort_requests=Mock(),
    )


def test_ready_nonce_stale_timeout_and_error_rollback() -> None:
    engine = _control_engine()
    control = LwdEdgePrefixControl(engine, timeout_s=60.0)
    request = _request()
    control.add_request(request)

    assert request.lwd_prefix_nonce == 1
    assert request.block_hashes == [b"h0", b"h1"]
    engine._lwd_admit_and_notify.assert_called_once_with(request)
    assert not engine._lwd_enqueue_request.called

    assert control.handle_client_request(
        EngineCoreRequestType.WAKEUP, LwdPrefixReadyNotify("r", 8, 99)
    )
    assert "r" in control.pending
    _request_pending, timer = control.pending["r"]
    timer.cancel()
    timer.function(*timer.args)
    request_type, timeout = engine.input_queue.get_nowait()
    assert request_type == EngineCoreRequestType.WAKEUP
    assert timeout.error

    assert control.handle_client_request(EngineCoreRequestType.WAKEUP, timeout)
    engine.scheduler.lwd_edge_prefix_ready.assert_called_with(request, 0)
    engine._lwd_enqueue_request.assert_called_once_with(request)
    engine._lwd_abort_internally.assert_called_once_with(request)
    assert "r" not in control.pending

    assert control.handle_client_request(EngineCoreRequestType.WAKEUP, timeout)
    engine._lwd_enqueue_request.assert_called_once_with(request)


def test_matching_ready_enqueues_hit_and_abort_cancels_waiter() -> None:
    engine = _control_engine()
    control = LwdEdgePrefixControl(engine, timeout_s=60.0)
    request = _request()
    control.add_request(request)
    control.handle_client_request(
        EngineCoreRequestType.WAKEUP, LwdPrefixReadyNotify("r", 8, 1)
    )
    engine.scheduler.lwd_edge_prefix_ready.assert_called_once_with(request, 8)
    engine._lwd_enqueue_request.assert_called_once_with(request)

    aborted = _request("a")
    control.add_request(aborted)
    control.prepare_abort(["a"])
    assert "a" not in control.pending
    engine.scheduler.lwd_edge_prefix_ready.assert_called_with(aborted, 0)
    assert engine._lwd_enqueue_request.call_count == 2
    control.shutdown()


def test_announce_failure_removes_waiter() -> None:
    engine = _control_engine()
    engine._lwd_admit_and_notify.side_effect = RuntimeError("full")
    control = LwdEdgePrefixControl(engine, timeout_s=60.0)

    with pytest.raises(RuntimeError, match="full"):
        control.add_request(_request())

    assert not control.pending


def _params(dp_block_size: int = 16) -> LwdCloudParamsNotify:
    return LwdCloudParamsNotify(
        max_num_batched_tokens=128,
        max_num_seqs=4,
        num_speculative_tokens=0,
        prefix_caching=True,
        hash_block_size=dp_block_size,
        prefix_alignment_tokens=16,
    )


def test_dp_prefix_capabilities_must_match() -> None:
    engine = object.__new__(LwdEdgeEngineCore)
    engine._lwd_prefix_enabled = True
    engine._lwd_cloud_params = {0: _params(), 1: _params(32)}
    engine._lwd_prefix_control = Mock(hash_block_size=16)
    engine.scheduler = SimpleNamespace()

    with pytest.raises(RuntimeError, match="inconsistent prefix capabilities"):
        engine._lwd_configure_prefix_cache()

    engine._lwd_cloud_params[1] = _params()
    engine._lwd_configure_prefix_cache()
    engine._lwd_prefix_control.configure.assert_called_once()
    assert engine.scheduler._lwd_prefix_hash_block_size == 16


def _refill_scheduler(send: Mock) -> LwdEdgeScheduler:
    scheduler = object.__new__(LwdEdgeScheduler)
    scheduler._lwd_accounts = (LwdCloudDpAccount(0), LwdCloudDpAccount(1))
    scheduler._lwd_route = {"r": 1}
    scheduler._lwd_pending_refills = deque()
    scheduler._lwd_announced_refills = set()
    scheduler._lwd_mrope_positions_dict = {}
    scheduler._lwd_aborted_req_ids = {0: [], 1: []}
    scheduler._lwd_chunk_unit = 4
    scheduler.scheduler_config = SimpleNamespace(max_num_batched_tokens=3)
    scheduler.requests = {
        "r": SimpleNamespace(
            request_id="r",
            client_index=9,
            num_prompt_tokens=10,
            prompt_token_ids=list(range(10)),
            mm_features=["image"],
            lwd_prefix_nonce=7,
            is_finished=lambda: False,
        )
    }
    scheduler.lwd_send_to_cloud = send
    scheduler.lwd_refill_abort = Mock()
    return scheduler


def test_refill_announces_at_dispatch_and_drains_after_abort() -> None:
    send = Mock()
    scheduler = _refill_scheduler(send)

    outputs = scheduler.lwd_edge_build_refill(
        LwdRefillNotify("r", 4, 7, prefix_nonce=7)
    )

    assert outputs is not None and len(outputs) == 1
    send.assert_not_called()
    assert scheduler._lwd_accounts[1].seqno == 0
    output = outputs[0]
    meta = output.lwd_batch.batch_meta
    assert meta.token_ids == [[4, 5, 6]] and meta.is_refill
    transported = msgspec.msgpack.decode(
        msgspec.msgpack.encode(meta),
        type=type(meta),
    )
    assert transported.is_refill
    assert output.scheduled_new_reqs[0].mm_features == ["image"]
    assert output.num_scheduled_tokens == {}

    scheduler.lwd_edge_queue_refill(outputs)
    assert scheduler._lwd_schedule_step() is output
    notify = send.call_args.args[1]
    assert send.call_args.args[0] == 1
    assert notify.seqno == 0 and notify.items[0].is_refill
    assert notify.items[0].prefix_nonce == 7
    assert scheduler._lwd_accounts[1].pending_batches == {0: {"r"}}

    assert scheduler.lwd_edge_abort(["r"]) == [(1, "r")]
    assert scheduler._lwd_accounts[1].pending_batches == {0: {"r"}}
    assert scheduler.lwd_edge_request_active("r")
    scheduler.requests.clear()
    scheduler.update_from_output(
        SimpleNamespace(lwd_batch=None),
        SimpleNamespace(lwd_up_send_watermarks={1: 0}),
    )
    assert scheduler._lwd_accounts[1].pending_batches == {}
    assert not scheduler.lwd_edge_request_active("r")


def test_refill_splits_over_budget_and_abort_only_filters_unannounced() -> None:
    send = Mock()
    scheduler = _refill_scheduler(send)

    outputs = scheduler.lwd_edge_build_refill(
        LwdRefillNotify("r", 0, 10, prefix_nonce=7)
    )

    assert outputs is not None
    chunk_rows = [len(out.lwd_batch.batch_meta.token_ids[0]) for out in outputs]
    assert chunk_rows == [3, 3, 3, 1]
    send.assert_not_called()
    scheduler.lwd_edge_queue_refill(outputs)
    scheduler._lwd_schedule_step()
    scheduler._lwd_schedule_step()
    assert send.call_count == 2
    assert scheduler._lwd_accounts[1].seqno == 2

    scheduler.lwd_edge_abort(["r"])

    assert not scheduler._lwd_pending_refills
    assert send.call_count == 2
    assert scheduler._lwd_accounts[1].pending_batches == {0: {"r"}, 1: {"r"}}


def test_stale_queued_refill_is_dropped_before_announce() -> None:
    scheduler = _refill_scheduler(Mock())
    outputs = scheduler.lwd_edge_build_refill(
        LwdRefillNotify("r", 4, 7, prefix_nonce=7)
    )
    assert outputs is not None
    scheduler.requests["r"].lwd_prefix_nonce = 8
    scheduler.lwd_edge_queue_refill(outputs)

    output = scheduler._lwd_schedule_step()

    assert output.lwd_batch is None
    scheduler.lwd_send_to_cloud.assert_not_called()
    assert scheduler._lwd_accounts[1].seqno == 0


def test_refill_send_failure_leaves_no_seqno_or_pending_batch() -> None:
    scheduler = _refill_scheduler(Mock(side_effect=RuntimeError("full")))
    outputs = scheduler.lwd_edge_build_refill(
        LwdRefillNotify("r", 4, 7, prefix_nonce=7)
    )
    assert outputs is not None
    scheduler.lwd_edge_queue_refill(outputs)

    output = scheduler._lwd_schedule_step()

    assert output.lwd_batch is None
    scheduler.lwd_refill_abort.assert_called_once_with("r")
    assert scheduler._lwd_accounts[1].seqno == 0
    assert scheduler._lwd_accounts[1].pending_batches == {}


def test_stale_refill_notify_is_ignored() -> None:
    scheduler = _refill_scheduler(Mock())

    assert (
        scheduler.lwd_edge_build_refill(LwdRefillNotify("r", 4, 7, prefix_nonce=6))
        == []
    )
    scheduler.lwd_send_to_cloud.assert_not_called()
    scheduler.lwd_refill_abort.assert_not_called()


def test_invalid_ready_hit_is_rejected_without_raising() -> None:
    scheduler = object.__new__(LwdEdgeScheduler)
    scheduler._lwd_prefix_hash_block_size = 4
    request = _request()

    assert not scheduler.lwd_edge_prefix_ready(request, 5)
    assert request.num_computed_tokens == 0
    assert scheduler.lwd_edge_prefix_ready(request, 8)
    assert request.num_computed_tokens == 8


def _engine_request(request_id: str = "r") -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        client_index=5,
        num_prompt_tokens=12,
        abort_immediately=False,
    )


def test_internal_abort_sends_output_to_request_client() -> None:
    engine = object.__new__(LwdEdgeEngineCore)
    engine.abort_requests = Mock()
    engine._send_abort_outputs_to_client = Mock()
    request = _engine_request()

    engine._lwd_abort_internally(request)

    engine.abort_requests.assert_called_once_with(["r"])
    engine._send_abort_outputs_to_client.assert_called_once_with(["r"], 5)


def test_unservable_refill_uses_internal_abort_output() -> None:
    engine = object.__new__(LwdEdgeEngineCore)
    request = _engine_request()
    engine.scheduler = SimpleNamespace(
        requests={"r": request},
        lwd_edge_build_refill=Mock(return_value=None),
    )
    engine.abort_requests = Mock()
    engine._send_abort_outputs_to_client = Mock()

    engine._lwd_handle_refill(LwdRefillNotify("r", 4, 7, prefix_nonce=7))

    engine.abort_requests.assert_called_once_with(["r"])
    engine._send_abort_outputs_to_client.assert_called_once_with(["r"], 5)


def test_duplicate_active_request_is_rejected_without_touching_old_generation() -> None:
    engine = object.__new__(LwdEdgeEngineCore)
    engine.scheduler = SimpleNamespace(lwd_edge_request_active=Mock(return_value=True))
    engine._lwd_prefix_control = None
    engine._lwd_admit_and_notify = Mock()
    engine._lwd_enqueue_request = Mock()
    engine._send_abort_outputs_to_client = Mock()
    request = _engine_request()

    with pytest.raises(AssertionError, match="duplicate active request id"):
        engine.add_request(request)

    engine._send_abort_outputs_to_client.assert_not_called()
    engine._lwd_admit_and_notify.assert_not_called()
    engine._lwd_enqueue_request.assert_not_called()

    engine.scheduler.lwd_edge_request_active.return_value = False
    engine.add_request(request)
    engine._lwd_admit_and_notify.assert_called_once_with(request)
    engine._lwd_enqueue_request.assert_called_once_with(request)


def test_stale_c2e_is_ignored_and_current_generation_delivers() -> None:
    scheduler = _refill_scheduler(Mock())
    scheduler.finish_requests = Mock()
    scheduler.vllm_config = None
    stale = LwdC2eNotify(
        hidden_num_elements=0,
        top_id_ths=[[]],
        num_accepted_tokens=[0],
        req_ids=["r"],
        finish_reasons=[int(FinishReason.ABORT)],
        prefix_nonces=[6],
    )

    outputs, finished = scheduler._lwd_deliver_unembed(stale, {"r": []})

    assert outputs == {} and finished == {}
    scheduler.finish_requests.assert_not_called()
    assert scheduler._lwd_route["r"] == 1

    current = LwdC2eNotify(
        hidden_num_elements=1,
        top_id_ths=[[]],
        num_accepted_tokens=[1],
        req_ids=["r"],
        finish_reasons=[LWD_NOT_FINISHED],
        prefix_nonces=[7],
    )
    outputs, finished = scheduler._lwd_deliver_unembed(current, {"r": [42]})

    assert list(outputs) == [0]
    assert outputs[0][0].new_token_ids == [42]
    assert finished == {}
