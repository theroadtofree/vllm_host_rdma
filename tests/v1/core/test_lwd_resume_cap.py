# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LWD resume cap / eagle-peek margin regression tests.

DeepSeek-V4 + MTP: probe can hit a grid-aligned prefix (e.g. 16384) while
resume collapses to 0 when the declared-hit cap leaves no room for the
eagle "peek one block then drop" lookahead. Margin must cover one logical
block (``block_size * compress_ratio``), not one hash block.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

pytestmark = pytest.mark.cpu_test


def _mla_spec(block_size: int, compress_ratio: int) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        compress_ratio=compress_ratio,
        model_version="deepseek_v4",
    )


def test_spec_token_block_size_matches_compress_manager() -> None:
    compressed = _mla_spec(block_size=8, compress_ratio=4)
    assert KVCacheManager._lwd_spec_token_block_size(compressed) == 32

    plain = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    assert KVCacheManager._lwd_spec_token_block_size(plain) == 128


def test_eagle_peek_covers_uniform_type_inner_compress() -> None:
    inner = _mla_spec(block_size=8, compress_ratio=128)
    uniform = UniformTypeKVCacheSpecs.from_specs({"c128": inner})
    assert uniform is not None
    # UniformType has no compress_ratio attr; peek must still see the
    # inner logical block (8 * 128 = 1024), not just the container's 8.
    assert KVCacheManager._lwd_eagle_peek_tokens(uniform) == 1024


def test_resume_margin_at_least_one_logical_block() -> None:
    compressed = _mla_spec(block_size=8, compress_ratio=128)
    plain = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    group_compressed = KVCacheGroupSpec(["c128"], compressed)
    group_plain = KVCacheGroupSpec(["full"], plain)
    config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[group_compressed, group_plain],
    )
    # hash_bs=8 is what the old fix used; it is far too small here.
    manager = KVCacheManager(
        kv_cache_config=config,
        max_model_len=4096,
        scheduler_block_size=128,
        hash_block_size=8,
        use_eagle=True,
    )
    # Force hybrid-style attention_groups (spec, ids, cls) tuples.
    manager.coordinator.attention_groups = [
        (compressed, [0], object),
        (plain, [1], object),
    ]
    manager.coordinator.lcm_block_size = 128
    margin = manager._lwd_resume_lookahead_margin()
    # Must cover the c128 logical block (8*128=1024) used as eagle peek.
    assert margin >= 1024
    assert margin >= 8  # still at least hash_bs
