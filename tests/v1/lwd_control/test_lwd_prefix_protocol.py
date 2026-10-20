# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""边云前缀协商线协议契约(《边-网关-云的交互逻辑》的可执行版)。

lwd_prefix 是纯 stdlib 层,不 import torch/vllm 运行件,故这几条在任何
机器上都应跑得过——它们就是边云两侧共用的字面契约,网关只做透明转发。
"""

import hashlib
from dataclasses import dataclass

import pytest

from vllm.v1.lwd_control.control_communication.lwd_prefix import (
    BLOCK_HASH_PREFIX,
    DIGEST_SIZE,
    HEADER_ALIGNMENT_TOKENS,
    HEADER_BLOCK_SIZE,
    HEADER_CLOUD_ID,
    HEADER_EDGE_ID,
    HEADER_HIT_BLOCKS,
    HEADER_HIT_TOKENS,
    HEADER_INSTANCE_ID,
    HEADER_MM_ABI,
    HEADER_PROMPT_TOKENS,
    HEADER_PROTOCOL,
    HEADER_REQUEST_ID,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_MM,
    TAIL_HASH_PREFIX,
    LwdPrefixHasher,
    LwdPrefixManifest,
    LwdProbeResult,
    LwdUsage,
    encode_digest,
)

pytestmark = pytest.mark.cpu_test

TENANT_KEY = b"tenant-a-secret-key-material"
BLOCK_SIZE = 4
FINGERPRINT = hashlib.sha256(b"processor-config").digest()


@dataclass(frozen=True)
class MediaItem:
    """媒体身份(协议层按属性鸭子类型读取,不依赖 vllm 的 MMFeature)。"""

    modality: str
    digest: bytes
    offset: int
    length: int


def _hasher(fingerprint: bytes | None = FINGERPRINT) -> LwdPrefixHasher:
    return LwdPrefixHasher(TENANT_KEY, BLOCK_SIZE, fingerprint)


def _manifest(tokens: int, request_id: str = "req-001") -> LwdPrefixManifest:
    return _hasher().build_manifest(request_id, list(range(tokens)))


def _wire_headers(
    manifest: LwdPrefixManifest, **overrides: str
) -> dict[str, str]:
    """文档「Edge 向平台发送请求」一节的请求头集合(文本协议 v1)。"""
    headers = {
        **manifest.to_headers(),
        HEADER_EDGE_ID: "0",
    }
    headers.update(overrides)
    return headers


# --------------------------------------------------------------------- #
# 请求侧:请求头 + messages 摘要体                        #
# --------------------------------------------------------------------- #
def test_manifest_headers_carry_the_documented_controls() -> None:
    """manifest 自带全部必需请求头,不靠调用方外挂(漏一项云侧即 400)。"""
    manifest = _manifest(10)  # 2 满块 + 尾块 2
    headers = manifest.to_headers()
    assert headers == {
        HEADER_PROTOCOL: PROTOCOL_VERSION,
        HEADER_REQUEST_ID: "req-001",
        HEADER_BLOCK_SIZE: str(BLOCK_SIZE),
        HEADER_PROMPT_TOKENS: "10",
    }
    messages = manifest.to_messages()
    assert [m["role"] for m in messages] == ["user"] * 3
    assert all(
        m["content"].startswith((BLOCK_HASH_PREFIX, TAIL_HASH_PREFIX))
        for m in messages
    )
    assert messages[-1]["content"].startswith(TAIL_HASH_PREFIX)


def test_manifest_round_trips_through_the_openai_request() -> None:
    manifest = _manifest(10)
    parsed = LwdPrefixManifest.from_openai_request(
        _wire_headers(manifest), {"messages": manifest.to_messages()}
    )
    assert parsed == manifest


@pytest.mark.parametrize("tokens", [0, 4, 8, 10, 13])
def test_tail_hash_present_exactly_for_partial_block(tokens: int) -> None:
    manifest = _manifest(tokens)
    assert manifest.full_block_count == tokens // BLOCK_SIZE
    assert (manifest.tail_hash is not None) == (tokens % BLOCK_SIZE != 0)
    parsed = LwdPrefixManifest.from_openai_request(
        _wire_headers(manifest), {"messages": manifest.to_messages()}
    )
    assert parsed == manifest


def test_prompt_tokens_travel_in_the_header_only() -> None:
    """网关 DTO 转换会剥 body 非标准字段:prompt 长度只认请求头。"""
    manifest = _manifest(10)
    headers = _wire_headers(manifest)
    del headers[HEADER_PROMPT_TOKENS]
    with pytest.raises(ValueError, match=HEADER_PROMPT_TOKENS):
        LwdPrefixManifest.from_openai_request(
            headers,
            {
                "messages": manifest.to_messages(),
                HEADER_PROMPT_TOKENS: "10",  # body 里补上也不认
            },
        )


@pytest.mark.parametrize(
    "prompt_tokens",
    [
        "", " ", "-10", "10.0", "abc",
        # int() 会收下但长度不再自证一致的形式(严格化的理由):
        "+10", "1_0", " 10", "10 ", "１０",
    ],
)
def test_prompt_tokens_header_must_be_a_non_negative_integer(
    prompt_tokens: str,
) -> None:
    manifest = _manifest(10)
    headers = _wire_headers(manifest, **{HEADER_PROMPT_TOKENS: prompt_tokens})
    with pytest.raises(ValueError):
        LwdPrefixManifest.from_openai_request(
            headers, {"messages": manifest.to_messages()}
        )


def test_v2_marker_is_not_parsed_by_the_v1_reader() -> None:
    """v2 的线格式与 v1 同构,但必须先过云侧协议门禁改写标记再进解析器。"""
    manifest = _manifest(10)
    headers = _wire_headers(manifest, **{HEADER_PROTOCOL: PROTOCOL_VERSION_MM})
    with pytest.raises(ValueError, match="unsupported edge-cloud protocol"):
        LwdPrefixManifest.from_openai_request(
            headers, {"messages": manifest.to_messages()}
        )


def test_message_body_must_be_edge_cloud_hashes() -> None:
    """真实 prompt / 采样参数不许混进摘要体。"""
    manifest = _manifest(10)
    body = {
        "messages": [
            {"role": "user", "content": "把这段 summarize 一下"},
            *manifest.to_messages(),
        ],
        "max_tokens": 32,
    }
    with pytest.raises(ValueError):
        LwdPrefixManifest.from_openai_request(_wire_headers(manifest), body)
    tail_first = {
        "messages": [
            *manifest.to_messages()[:-1],
            manifest.to_messages()[-1],
            manifest.to_messages()[0],  # 满块摘要排在尾块之后
        ]
    }
    with pytest.raises(ValueError):
        LwdPrefixManifest.from_openai_request(_wire_headers(manifest), tail_first)


# --------------------------------------------------------------------- #
# 媒体覆盖块:0x04/0x05 域与处理器指纹                     #
# (MM-ABI 头与指纹材料本身见 test_lwd_mm_identity.py)    #
# --------------------------------------------------------------------- #
def test_media_blocks_diverge_from_the_text_chain() -> None:
    ids = list(range(2 * BLOCK_SIZE))
    text = _hasher().build_manifest("req-001", ids)
    media = _hasher().build_manifest(
        "req-001",
        ids,
        [MediaItem("image", hashlib.sha256(b"image-a").digest(), 0, BLOCK_SIZE)],
    )
    assert text.full_block_hashes != media.full_block_hashes
    # 同一媒体身份可复现,换图即分叉
    same = _hasher().build_manifest(
        "req-002",
        ids,
        [MediaItem("image", hashlib.sha256(b"image-a").digest(), 0, BLOCK_SIZE)],
    )
    assert media.full_block_hashes == same.full_block_hashes
    other = _hasher().build_manifest(
        "req-003",
        ids,
        [MediaItem("image", hashlib.sha256(b"image-b").digest(), 0, BLOCK_SIZE)],
    )
    assert media.full_block_hashes != other.full_block_hashes


def test_media_chain_requires_a_processor_fingerprint() -> None:
    with pytest.raises(ValueError):
        _hasher(fingerprint=None).build_manifest(
            "req-001",
            list(range(2 * BLOCK_SIZE)),
            [MediaItem("image", b"0" * DIGEST_SIZE, 0, BLOCK_SIZE)],
        )


def test_unsupported_modality_is_rejected() -> None:
    with pytest.raises(ValueError):
        _hasher().build_manifest(
            "req-001",
            list(range(2 * BLOCK_SIZE)),
            [MediaItem("audio", b"0" * DIGEST_SIZE, 0, BLOCK_SIZE)],
        )


# --------------------------------------------------------------------- #
# 回显侧:启动消息 + 响应头 + usage                        #
# --------------------------------------------------------------------- #
def _probe_result() -> LwdProbeResult:
    return LwdProbeResult(
        request_id="req-001",
        instance_id="cloud-0",
        block_size=BLOCK_SIZE,
        hit_blocks=2,
        hit_tokens=2 * BLOCK_SIZE,
        alignment_tokens=512,
        cloud_id=3,
    )


def test_probe_control_message_round_trips_with_extensions() -> None:
    result = _probe_result()
    message = result.to_control_message(PROTOCOL_VERSION_MM, "mm1:abc")
    assert message["type"] == "edge_cloud_probe"
    assert message["protocol"] == PROTOCOL_VERSION_MM
    assert message["mm_abi"] == "mm1:abc"
    assert LwdProbeResult.from_control_message(message) == result
    # 文本请求不带这两个字段
    plain = result.to_control_message()
    assert plain["protocol"] == PROTOCOL_VERSION
    assert "mm_abi" not in plain
    with pytest.raises(ValueError):
        LwdProbeResult.from_control_message({**message, "type": "other"})


def test_probe_headers_are_case_insensitive_and_backwards_compatible() -> None:
    headers = {k.lower(): v for k, v in _probe_result().to_headers().items()}
    assert LwdProbeResult.from_headers(headers) == _probe_result()
    # 旧版云不公告 alignment/cloud_id:按 0/None 降级,不报错
    for name in (HEADER_ALIGNMENT_TOKENS, HEADER_CLOUD_ID):
        headers.pop(name.lower(), None)
    downgraded = LwdProbeResult.from_headers(headers)
    assert downgraded.alignment_tokens == 0
    assert downgraded.cloud_id is None
    headers[HEADER_PROTOCOL] = "edge-cloud-prefix-v9"
    with pytest.raises(ValueError):
        LwdProbeResult.from_headers(headers)


def test_probe_result_rejects_inconsistent_hit_math() -> None:
    with pytest.raises(ValueError):
        LwdProbeResult(
            request_id="req-001",
            instance_id="cloud-0",
            block_size=BLOCK_SIZE,
            hit_blocks=2,
            hit_tokens=BLOCK_SIZE,  # != hit_blocks * block_size
        )


def test_usage_keeps_the_standard_openai_shape() -> None:
    """文档事件四:真实输入/输出/缓存,不新增私有字段。"""
    usage = LwdUsage(prompt_tokens=1024, completion_tokens=128, cached_tokens=768)
    assert usage.to_openai_dict() == {
        "prompt_tokens": 1024,
        "completion_tokens": 128,
        "total_tokens": 1152,
        "prompt_tokens_details": {"cached_tokens": 768},
    }
    with pytest.raises(ValueError):
        LwdUsage(prompt_tokens=10, completion_tokens=1, cached_tokens=11)


def test_documented_request_header_names_are_the_wire_contract() -> None:
    """头名字面量与文档表格逐条对齐(改名即破契约,须由这几条钉住)。"""
    assert HEADER_PROTOCOL == "X-Edge-Cloud-Protocol"
    assert HEADER_REQUEST_ID == "X-Edge-Cloud-Request-ID"
    assert HEADER_EDGE_ID == "X-Edge-Cloud-Edge-Id"
    assert HEADER_BLOCK_SIZE == "X-Edge-Cloud-Block-Size"
    assert HEADER_PROMPT_TOKENS == "X-Edge-Cloud-Prompt-Tokens"
    assert HEADER_MM_ABI == "X-Edge-Cloud-MM-ABI"
    assert HEADER_INSTANCE_ID == "X-Edge-Cloud-Instance-ID"
    assert HEADER_HIT_BLOCKS == "X-Edge-Cloud-Prefix-Hit-Blocks"
    assert HEADER_HIT_TOKENS == "X-Edge-Cloud-Prefix-Hit-Tokens"
    assert (PROTOCOL_VERSION, PROTOCOL_VERSION_MM) == (
        "edge-cloud-prefix-v1",
        "edge-cloud-prefix-v2",
    )


# --------------------------------------------------------------------- #
# 黄金向量:钉死链的字节级输出                            #
# --------------------------------------------------------------------- #
# 固定输入 → 固定摘要。domain 字节、parent 链式顺序、uint32 大端编码、seed
# (HMAC(tenant_key, PROTOCOL_VERSION))或 base64url 口径任何一处改动,这几条
# 立刻转红。为什么必须有:这些改动不会让任何一侧报错——两边都"自证一致",只是
# 边发的哈希不再等于云存的那份,表现为命中突然恒 0 且要等真机压测才发现。
# 动这些常量必须是一次显式的协议版本决策,不能是顺手重构。
_GOLDEN_TENANT_KEY = b"0123456789abcdef"  # 16B:协议允许的最小密钥
_GOLDEN_BLOCK_SIZE = 4
_GOLDEN_TOKENS = list(range(10))  # 2 满块 + 2 token 尾块
_GOLDEN_FINGERPRINT = bytes(range(32))
_GOLDEN_IMAGE_A = hashlib.sha256(b"image-a").digest()
_GOLDEN_IMAGE_B = hashlib.sha256(b"image-b").digest()
_GOLDEN_TEXT_MESSAGES = [
    {"role": "user",
     "content": "ecb1:-o_M5mvivgvdu3dzIfEFyyvxoZ5ym-bDNBG0pzer3gY"},
    {"role": "user",
     "content": "ecb1:sHDhLsLlqWK1oIpqWuwG_tWVmElRQtG1Wv9VxGy-Fo0"},
    {"role": "user",
     "content": "ect1:d8eGsBxzXx0CtrZ6aVb8vGd4t8NsqVRZIpvscpn0UHk"},
]
# 媒体向量把四个哈希域一次钉全:块 0 走 0x04(含媒体)、块 1 走 0x01(纯文本)、
# 尾块走 0x05(含媒体,图 B 覆盖 token 8)、未掺媒体的 parent 链仍按 0x01 递推。
_GOLDEN_MEDIA_MESSAGES = [
    {"role": "user",
     "content": "ecb1:_FdYyCpW6HyiLsYOV7froRxlKpq59NUIPQSIFFhJy6U"},
    {"role": "user",
     "content": "ecb1:1VmUqPPegnXkX-ykIf-wK0RHduynCCDI1R9veFwPqs8"},
    {"role": "user",
     "content": "ect1:Edtq_7VFZyEuP3WNFLpQh3_D-8OHPVyqsJXqU_2UBL0"},
]


def test_text_hash_chain_matches_the_golden_vector() -> None:
    manifest = LwdPrefixHasher(_GOLDEN_TENANT_KEY, _GOLDEN_BLOCK_SIZE).build_manifest(
        "req-000", _GOLDEN_TOKENS
    )
    assert [encode_digest(d) for d in manifest.full_block_hashes] == [
        message["content"][len(BLOCK_HASH_PREFIX):]
        for message in _GOLDEN_TEXT_MESSAGES
        if message["content"].startswith(BLOCK_HASH_PREFIX)
    ]
    assert encode_digest(manifest.tail_hash) == (
        _GOLDEN_TEXT_MESSAGES[-1]["content"][len(TAIL_HASH_PREFIX):]
    )
    # 线上形态整条钉住(前缀 + 编码 + 顺序)
    assert manifest.to_messages() == _GOLDEN_TEXT_MESSAGES


def test_media_aware_hash_chain_matches_the_golden_vector() -> None:
    media = [
        MediaItem("image", _GOLDEN_IMAGE_A, 0, _GOLDEN_BLOCK_SIZE),
        MediaItem("image", _GOLDEN_IMAGE_B, 8, 1),
    ]
    manifest = LwdPrefixHasher(
        _GOLDEN_TENANT_KEY, _GOLDEN_BLOCK_SIZE, _GOLDEN_FINGERPRINT
    ).build_manifest("req-000", _GOLDEN_TOKENS, media)
    assert manifest.to_messages() == _GOLDEN_MEDIA_MESSAGES
    # 掺媒体即分叉:同一条 prompt 在纯文本口径下是另一组摘要
    text_only = LwdPrefixHasher(
        _GOLDEN_TENANT_KEY, _GOLDEN_BLOCK_SIZE, _GOLDEN_FINGERPRINT
    ).build_manifest("req-000", _GOLDEN_TOKENS)
    assert text_only.to_messages() == _GOLDEN_TEXT_MESSAGES


def test_seed_matches_the_golden_vector() -> None:
    """seed = HMAC(tenant_key, 协议名):换协议名即整链分叉,单独钉一次。"""
    import hmac as _hmac

    seed = _hmac.digest(
        _GOLDEN_TENANT_KEY, PROTOCOL_VERSION.encode("utf-8"), "sha256"
    )
    assert encode_digest(seed) == (
        "lJuUWkbgy4fG-CBiS_C9e7q7yfDnXHXzSk58fsRDiFI"
    )

