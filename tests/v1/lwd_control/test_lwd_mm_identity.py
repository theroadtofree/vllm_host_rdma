# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""处理器指纹与 MM-ABI 头契约(lwd_mm_identity)。

核心判据是**路径中性**:同一份 artifact 在边云常挂不同本地路径,指纹不能因此
分叉——否则 ``enforce_mm_abi_match`` 一开,正常图片流量全被云侧 400。
同时它必须对真正影响媒体预处理口径的字段敏感(那些字段一变,已入库的媒体块
摘要就再也查不到)。
"""

import hashlib

import pytest

from vllm.v1.lwd_control.control_communication.lwd_mm_identity import (
    MM_ABI_VERSION,
    compute_processor_fingerprint,
    mm_abi_header_value,
    validate_mm_abi_header,
)
from vllm.v1.lwd_control.control_communication.lwd_prefix import DIGEST_SIZE

pytestmark = pytest.mark.cpu_test

FINGERPRINT = hashlib.sha256(b"processor-config").digest()


class _FakeConfig:
    """够用的 ModelConfig/MultiModalConfig 替身。

    ``compute_hash`` 按自身字段求摘要(与 vLLM 同语义),``__repr__`` 稳定,
    这样嵌套配置参与摘要时不会因对象地址而抖动。"""

    def __init__(self, **fields) -> None:
        self.__dict__.update(fields)

    def __repr__(self) -> str:
        return repr(sorted(
            (k, repr(v)) for k, v in self.__dict__.items()
            if not k.startswith("_")
        ))

    def compute_hash(self) -> str:
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()


def _model_config(model: str = "/data/edge/qwen3.5", **overrides) -> _FakeConfig:
    mm_kwargs = overrides.pop("mm_processor_kwargs", None)
    mm = _FakeConfig(
        mm_processor_kwargs=mm_kwargs,
        media_io_kwargs=None,
        limit_per_prompt=None,
        interleave_mm_strings=None,
        video_pruning_rate=None,
        mm_encoder_fp8_scale_path=None,
    )
    fields = {
        "model": model,
        "model_weights": None,
        "tokenizer": model,
        "revision": None,
        "code_revision": None,
        "tokenizer_revision": None,
        "hf_config": {"_name_or_path": model, "hidden_size": 4096},
        "hf_text_config": {"model_type": "qwen3_5_vl"},
        "_architecture": None,
        "multimodal_config": mm,
        "hf_image_processor_config": None,
    }
    fields.update(overrides)
    return _FakeConfig(**fields)


# --------------------------------------------------------------------- #
# MM-ABI 头:取值形态                                  #
# --------------------------------------------------------------------- #
def test_mm_abi_header_round_trips() -> None:
    value = mm_abi_header_value(FINGERPRINT)
    assert value.startswith(f"{MM_ABI_VERSION}:")
    assert "=" not in value  # base64url 去填充:网关不会把 = 吞成参数分隔
    assert validate_mm_abi_header(value) == value


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        f"{MM_ABI_VERSION}:",
        f"mm2:{mm_abi_header_value(FINGERPRINT).partition(':')[2]}",
        f"{MM_ABI_VERSION}:zz!!",
        f"{MM_ABI_VERSION}:{mm_abi_header_value(FINGERPRINT)[4:-4]}",
        "plain-fingerprint",
    ],
)
def test_mm_abi_header_rejects_malformed_values(value) -> None:
    with pytest.raises(ValueError):
        validate_mm_abi_header(value)


# --------------------------------------------------------------------- #
# 指纹:路径中性 + 预处理口径敏感                        #
# --------------------------------------------------------------------- #
def test_fingerprint_ignores_local_artifact_paths() -> None:
    """边云挂载路径不同不等于 ABI 不同:指纹必须相同。"""
    edge = _model_config("/data/edge/qwen3.5")
    cloud = _model_config("/mnt/npu/models/qwen3.5-served")
    assert compute_processor_fingerprint(edge) == compute_processor_fingerprint(cloud)


def test_fingerprint_is_deterministic() -> None:
    config = _model_config(mm_processor_kwargs={"size": {"width": 1024}})
    digest = compute_processor_fingerprint(config)
    assert len(digest) == DIGEST_SIZE
    assert digest == compute_processor_fingerprint(_model_config(
        "/another/path",
        mm_processor_kwargs={"size": {"width": 1024}},
    ))


@pytest.mark.parametrize(
    "field,value",
    [
        ("mm_processor_kwargs", {"size": {"width": 512}}),
        ("revision", "abcdef1"),
        ("code_revision", "deadbee"),
        ("tokenizer_revision", "cafe123"),
        ("hf_text_config", {"model_type": "qwen3_5_moe"}),
        ("hf_image_processor_config", {"crop_size": 448}),
    ],
)
def test_fingerprint_is_sensitive_to_preprocessing_identity(
    field: str, value
) -> None:
    """真正会改媒体占位/摘要口径的字段一变即分叉(旧块摘要从此查不到)。"""
    base = compute_processor_fingerprint(_model_config())
    drifted = compute_processor_fingerprint(_model_config(**{field: value}))
    assert base != drifted


def test_fingerprint_handles_a_missing_multimodal_config() -> None:
    """纯文本部署没有 MultiModalConfig:不能抛,且仍路径中性。"""
    first = _model_config("/data/edge/qwen3.5")
    first.multimodal_config = None
    second = _model_config("/mnt/npu/models/qwen3.5-served")
    second.multimodal_config = None
    digest = compute_processor_fingerprint(first)
    assert len(digest) == DIGEST_SIZE
    assert digest == compute_processor_fingerprint(second)
    assert digest != compute_processor_fingerprint(_model_config())
