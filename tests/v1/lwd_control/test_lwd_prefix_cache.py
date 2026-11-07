# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_prefix_cache import (
    LwdPrefixReservations,
    LwdProbeQueryAdapter,
    init_lwd_prefix_cache,
    prefix_start,
)
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import (
    DIGEST_SIZE,
    LwdPrefixManifest,
)


def _request(*, skip=False, computed=0, floor=None):
    request = SimpleNamespace(
        request_id="r",
        num_prompt_tokens=64,
        num_tokens=64,
        num_computed_tokens=computed,
        block_hashes=["a", "b"],
        skip_reading_prefix_cache=skip,
    )
    if floor is not None:
        request.lwd_resume_cap_tokens = floor
    return request


def _manager(groups=(), hit=0, uncached=0):
    manager = SimpleNamespace(
        coordinator=Mock(),
        block_pool=Mock(),
        lwd_prefix_caching=False,
    )
    manager.coordinator.find_longest_cache_hit.return_value = (
        groups,
        hit,
        uncached,
    )
    manager.coordinator.single_type_managers = (SimpleNamespace(block_size=16),)
    return manager


def test_reserve_pins_real_blocks_and_freezes_snapshot():
    block = SimpleNamespace(is_null=False)
    null_block = SimpleNamespace(is_null=True)
    manager = _manager(([block, null_block],), hit=16)
    reservations = LwdPrefixReservations(manager)
    request = _request()

    assert reservations.reserve(request) == 16
    manager.block_pool.touch.assert_called_once_with([block])
    assert request.lwd_resume_cap_tokens == 16
    assert reservations.snapshot(request) == (((block, null_block),), 16)

    reservations.release("r")
    reservations.release("r")
    manager.block_pool.free_blocks.assert_called_once_with([block])


def test_reserve_miss_snapshot_does_not_change_with_cache_growth():
    block = SimpleNamespace(is_null=False)
    manager = _manager()
    reservations = LwdPrefixReservations(manager)
    request = _request()
    assert reservations.reserve(request) == 0

    manager.coordinator.find_longest_cache_hit.return_value = (([block],), 16, 0)
    assert reservations.snapshot(request) == ((), 0)
    assert manager.coordinator.find_longest_cache_hit.call_count == 1


def test_skip_reading_reserves_zero_without_lookup():
    manager = _manager(([SimpleNamespace(is_null=False)],), hit=16)
    reservations = LwdPrefixReservations(manager)
    request = _request(skip=True)

    assert reservations.reserve(request) == 0
    assert reservations.snapshot(request) == ((), 0)
    manager.coordinator.find_longest_cache_hit.assert_not_called()
    manager.block_pool.touch.assert_not_called()


def test_missing_negotiated_snapshot_fails_closed():
    reservations = LwdPrefixReservations(_manager())
    with pytest.raises(RuntimeError, match="reserved prefix"):
        reservations.snapshot(_request(floor=16))


def test_reserve_hit_preserves_floor_and_truncate_preserves_layout():
    blocks = [SimpleNamespace(is_null=False) for _ in range(3)]
    reservations = LwdPrefixReservations(_manager())
    request = _request(floor=48)

    truncated = reservations.truncate((blocks,), 32)
    assert truncated == (blocks[:2],)
    assert len(blocks) == 3
    assert reservations.reserve_hit(request, truncated, 32) == 32
    assert request.lwd_resume_cap_tokens == 48
    assert reservations.snapshot(request) == ((tuple(blocks[:2]),), 32)
    assert prefix_start(request, reservations) == 32


def test_prefix_start_uses_native_watermark_after_allocation():
    request = _request(computed=24, floor=16)
    assert prefix_start(request) == 24
    assert prefix_start(_request(floor=16)) == 16
    assert prefix_start(_request()) == 0


def _lwd_kv_manager(reservations):
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.enable_caching = True
    manager.lwd_prefix_caching = True
    manager.lwd_prefix_reservations = reservations
    manager.enable_kv_cache_events = False
    manager.log_stats = False
    manager.prefix_cache_stats = None
    manager.empty_kv_cache_blocks = KVCacheBlocks(((),))
    return manager


def test_lwd_kv_and_connector_apis_return_frozen_triple():
    block = SimpleNamespace(is_null=False)
    reservations = LwdPrefixReservations(_manager(([block],), hit=16, uncached=16))
    request = _request()
    reservations.reserve(request)
    manager = _lwd_kv_manager(reservations)

    blocks, hit, boundary = manager.get_computed_blocks(request)
    assert blocks.blocks == ((block,),)
    assert (hit, boundary) == (16, 0)

    connector_result = manager.get_computed_blocks_for_connector(request)
    assert connector_result == (blocks, 16, 0, False)


def test_lwd_connector_missing_snapshot_fails_closed():
    reservations = LwdPrefixReservations(_manager())
    manager = _lwd_kv_manager(reservations)
    request = _request(floor=16)

    with pytest.raises(RuntimeError, match="reserved prefix"):
        manager.get_computed_blocks_for_connector(request)


def test_init_builds_reservations_and_alignment(monkeypatch):
    manager = _manager()
    manager.enable_caching = True
    manager.coordinator.scheduler_block_size = 16
    manager.coordinator.lcm_block_size = 32
    spec = object.__new__(FullAttentionSpec)
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)]
    )
    monkeypatch.setattr(
        "vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes",
        lambda config, vllm_config: (16, 16),
    )

    state = init_lwd_prefix_cache(vllm_config, kv_cache_config, manager, use_eagle=True)

    assert state.enabled
    assert state.hash_block_size == 16
    assert state.alignment_tokens == 32
    assert state.use_eagle
    assert manager.lwd_prefix_reservations is state.reservations


def _manifest(block_size: int = 16, prompt_tokens: int = 64) -> LwdPrefixManifest:
    return LwdPrefixManifest(
        request_id="r",
        prompt_tokens=prompt_tokens,
        block_size=block_size,
        full_block_hashes=(b"h" * DIGEST_SIZE,) * (prompt_tokens // block_size),
    )


def test_probe_query_without_prefix_caching_skips_block_size_contract():
    """Placement-only clouds answer zero hits at the manifest granularity.

    The edge cannot know this cloud's hash block size when prefix caching is
    disabled on both sides; gating on it turned every probe into a 503.
    """
    manager = _manager()
    adapter = LwdProbeQueryAdapter(
        manager,
        hash_block_size=128,
        alignment_tokens=0,
        instance_id="cloud-0",
        cloud_id=0,
        prefix_caching=False,
    )

    result = adapter.probe(_manifest(block_size=16))

    assert (result.block_size, result.hit_blocks, result.hit_tokens) == (16, 0, 0)
    assert result.cloud_id == 0
    manager.coordinator.find_longest_cache_hit.assert_not_called()


def test_probe_query_with_prefix_caching_enforces_block_size():
    adapter = LwdProbeQueryAdapter(
        _manager(),
        hash_block_size=128,
        alignment_tokens=0,
        instance_id="cloud-0",
        cloud_id=0,
    )

    with pytest.raises(ValueError, match="must match the cloud hash block size"):
        adapter.probe(_manifest(block_size=16))


def test_probe_query_with_prefix_caching_reports_cloud_hit():
    manager = _manager(hit=32)
    adapter = LwdProbeQueryAdapter(
        manager,
        hash_block_size=16,
        alignment_tokens=0,
        instance_id="cloud-0",
        cloud_id=1,
    )

    result = adapter.probe(_manifest(block_size=16))

    assert (result.block_size, result.hit_tokens, result.cloud_id) == (16, 32, 1)
