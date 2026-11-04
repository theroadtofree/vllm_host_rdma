# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine import (
    LwdCloudEngineCoreMixin,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_engine import LwdBaseEngineCore
from vllm.v1.lwd_control.lwd_comm.lwd_messages import LwdRequestNotify
from vllm.v1.request import RequestStatus


def _cloud_engine() -> LwdCloudEngineCoreMixin:
    engine = object.__new__(LwdCloudEngineCoreMixin)
    engine._lwd_prefix_enabled = True
    engine.scheduler = SimpleNamespace()
    engine._lwd_send_prefix_ready = Mock()
    return engine


def test_add_request_reserves_before_native_add_then_replies(monkeypatch) -> None:
    engine = _cloud_engine()
    reservations = Mock()
    reservations.reserve.return_value = 32
    engine.scheduler.lwd_prefix_reservations = reservations
    request = SimpleNamespace(
        request_id="r", lwd_prefix_nonce=7, lwd_wire_hashes_adopted=True
    )
    added = []
    monkeypatch.setattr(
        LwdBaseEngineCore, "add_request", lambda _self, req, wave=0: added.append(req)
    )

    engine.add_request(request)

    reservations.reserve.assert_called_once_with(request)
    assert added == [request]
    engine._lwd_send_prefix_ready.assert_called_once_with("r", 32, 7)


def test_reserve_failure_replies_error_without_native_add(monkeypatch) -> None:
    engine = _cloud_engine()
    reservations = Mock()
    reservations.reserve.side_effect = RuntimeError("no blocks")
    engine.scheduler.lwd_prefix_reservations = reservations
    request = SimpleNamespace(
        request_id="r", lwd_prefix_nonce=7, lwd_wire_hashes_adopted=True
    )
    native_add = Mock()
    monkeypatch.setattr(LwdBaseEngineCore, "add_request", native_add)

    engine.add_request(request)

    native_add.assert_not_called()
    engine._lwd_send_prefix_ready.assert_called_once_with("r", 0, 7, error="no blocks")


def test_active_duplicate_replies_error_without_native_add(monkeypatch) -> None:
    engine = _cloud_engine()
    engine.scheduler.requests = {"r": object()}
    engine.scheduler.lwd_prefix_reservations = Mock()
    request = SimpleNamespace(
        request_id="r", lwd_prefix_nonce=7, lwd_wire_hashes_adopted=True
    )
    native_add = Mock()
    monkeypatch.setattr(LwdBaseEngineCore, "add_request", native_add)

    engine.add_request(request)

    native_add.assert_not_called()
    engine.scheduler.lwd_prefix_reservations.reserve.assert_not_called()
    engine._lwd_send_prefix_ready.assert_called_once_with(
        "r", 0, 7, error="duplicate active request id"
    )


def test_request_build_adopts_wire_hashes_and_skip_reading() -> None:
    engine = object.__new__(LwdCloudEngineCoreMixin)
    engine.request_block_hasher = Mock(return_value=[])
    engine._lwd_hash_block_size = 2
    engine.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_hidden_size=lambda: 8,
            dtype=torch.float32,
        )
    )
    notify = LwdRequestNotify(
        "r",
        4,
        block_hashes=[b"h0", b"h1"],
        prefix_nonce=7,
        skip_reading_prefix_cache=True,
    )

    request = engine._lwd_build_request(notify)

    assert request.block_hashes == [b"h0", b"h1"]
    assert request.lwd_wire_hashes_adopted
    assert request.lwd_prefix_nonce == 7
    assert request.skip_reading_prefix_cache


@pytest.mark.parametrize("wire_hashes", [[], [b"h0"]])
def test_request_build_never_hashes_meta_embeds_on_wire_chain_mismatch(
    wire_hashes: list[bytes],
) -> None:
    """A missing/partial edge chain must not fall back to local hashing.

    The cloud request owns a meta prompt-embeds placeholder, so the local
    hasher raises ``Cannot copy out of meta tensor`` (seen under concurrent
    gateway/probe requests whose chain length differs from the cloud block
    count).  Admission must fail closed through ``lwd_wire_hashes_adopted``
    instead.
    """
    engine = object.__new__(LwdCloudEngineCoreMixin)
    engine.request_block_hasher = Mock(
        side_effect=AssertionError("local hasher read placeholder input")
    )
    engine._lwd_hash_block_size = 2
    engine.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_hidden_size=lambda: 8,
            dtype=torch.float32,
        )
    )
    notify = LwdRequestNotify(
        "r",
        4,
        block_hashes=wire_hashes,
        prefix_nonce=7,
    )

    request = engine._lwd_build_request(notify)

    assert request.block_hashes == wire_hashes
    assert not request.lwd_wire_hashes_adopted
    engine.request_block_hasher.assert_not_called()


@pytest.mark.parametrize("prompt_tokens", [3, 4])
def test_decode_never_hashes_or_caches_output_blocks(prompt_tokens: int) -> None:
    engine = object.__new__(LwdCloudEngineCoreMixin)
    engine.request_block_hasher = Mock(
        side_effect=AssertionError("local hasher read placeholder input")
    )
    engine._lwd_hash_block_size = 2
    engine.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_hidden_size=lambda: 8,
            dtype=torch.float32,
        )
    )
    wire_hashes = [f"h{i}".encode() for i in range(prompt_tokens // 2)]
    request = engine._lwd_build_request(
        LwdRequestNotify(
            "r",
            prompt_tokens,
            block_hashes=wire_hashes,
            prefix_nonce=7,
        )
    )
    assert request.block_hashes == wire_hashes
    assert request.shared_prefix_boundary == 0

    request.append_output_token_ids([11, 12, 13])

    assert request.num_output_tokens == 3
    assert request.block_hashes == wire_hashes
    engine.request_block_hasher.assert_not_called()

    coordinator = Mock()
    coordinator.get_num_blocks_to_allocate.return_value = 0
    coordinator.allocate_new_blocks.return_value = ([],)
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.max_model_len = 100
    manager.lwd_prefix_caching = True
    manager.retained_hit_group_ids = ()
    manager.empty_kv_cache_blocks = KVCacheBlocks(((),))
    manager.coordinator = coordinator
    manager.block_pool = SimpleNamespace(get_num_free_blocks=lambda: 100)
    manager.watermark_blocks = 0

    new_blocks = manager.allocate_slots(
        request,
        request.num_tokens,
        has_scheduled_reqs=False,
    )

    assert new_blocks is not None
    coordinator.cache_blocks.assert_called_once_with(request, prompt_tokens)


@pytest.mark.parametrize("prompt_tokens", [3, 4])
def test_async_output_update_never_hashes_or_caches_output_blocks(
    prompt_tokens: int,
) -> None:
    engine = object.__new__(LwdCloudEngineCoreMixin)
    engine.request_block_hasher = Mock(
        side_effect=AssertionError("local hasher read placeholder input")
    )
    engine._lwd_hash_block_size = 2
    engine.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_hidden_size=lambda: 8,
            dtype=torch.float32,
        )
    )
    wire_hashes = [f"h{i}".encode() for i in range(prompt_tokens // 2)]
    request = engine._lwd_build_request(
        LwdRequestNotify("r", prompt_tokens, block_hashes=wire_hashes)
    )
    request.status = RequestStatus.RUNNING
    output_token_ids = [11, 12, 13]
    request.num_computed_tokens = request.num_tokens + len(output_token_ids)
    request.num_output_placeholders = len(output_token_ids)

    manager = KVCacheManager.__new__(KVCacheManager)
    manager.enable_caching = True
    manager.lwd_prefix_caching = True
    manager.coordinator = Mock()
    scheduler = object.__new__(AsyncScheduler)
    scheduler.max_model_len = 100
    scheduler.kv_cache_manager = manager

    new_token_ids, stopped = scheduler._update_request_with_output(
        request, output_token_ids
    )

    assert new_token_ids == output_token_ids
    assert not stopped
    assert request.block_hashes == wire_hashes
    engine.request_block_hasher.assert_not_called()
    manager.coordinator.cache_blocks.assert_called_once_with(request, prompt_tokens)


@pytest.mark.parametrize(
    ("done", "expected_args"),
    [
        (("refill_done", "r"), ("r",)),
        (("refill_done", "r", 7), ("r", 7)),
    ],
)
def test_refill_done_bypasses_watermark_recording(
    monkeypatch, done, expected_args
) -> None:
    class DoneMq:
        def __init__(self) -> None:
            self.items = [done]

        def dequeue(self, timeout: int):
            if self.items:
                return self.items.pop(0)
            raise TimeoutError

    engine = object.__new__(LwdCloudEngineCoreMixin)
    engine._lwd_done_mq = DoneMq()
    engine.scheduler = SimpleNamespace(lwd_refill_done=Mock())
    from vllm_ascend.worker.lwd_cloud_model import lwd_pre_recv

    record = Mock()
    monkeypatch.setattr(lwd_pre_recv, "record_irecv_completions", record)

    engine._lwd_drain_irecv_completions()

    engine.scheduler.lwd_refill_done.assert_called_once_with(*expected_args)
    record.assert_not_called()


def test_send_refill_carries_generation() -> None:
    engine = object.__new__(LwdCloudEngineCoreMixin)
    engine._lwd_edge = object()
    engine._lwd_send_until_queued = Mock()

    engine._lwd_send_refill("r", 16, 32, 7)

    peer, notify = engine._lwd_send_until_queued.call_args.args
    assert peer is engine._lwd_edge
    assert (notify.request_id, notify.prefix_nonce) == ("r", 7)


def test_prefix_with_kv_connector_fails_before_startup() -> None:
    config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        kv_transfer_config=SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="external hits bypass.*READY"):
        LwdCloudEngineCoreMixin(vllm_config=config, lwd_dp_size=1)
