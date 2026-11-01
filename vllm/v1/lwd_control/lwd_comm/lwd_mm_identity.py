# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""处理器指纹与 MM-ABI 标识:边云两侧多模态建链口径的同源性判定。

指纹既是媒体覆盖块(0x04/0x05 哈希域)的 HMAC 材料,也以
``X-Edge-Cloud-MM-ABI: mm1:<base64url(指纹)>`` 随 v2 请求上线,供云侧仲裁。
边侧建链与云侧比对必须逐位同源,故两处实现合在本模块一处(摘要的线上编码
口径仍由协议层 ``lwd_prefix`` 单点提供)。

**指纹材料刻意不含模型/tokenizer 的本地路径字符串**:同一份 artifact 在边云
常挂在不同路径下,路径掺进指纹会让"同一次部署"算出两个指纹,
``enforce_mm_abi_match`` 一开就把正常图片流量全打成 400。稳定身份由
revision / code_revision / config hash 承载(参考分支
``edge_cloud/mm_identity.py`` 同款判断,那边已过真机验收)。

纯文本请求不受影响:指纹只进媒体覆盖块,文本链逐位不变。
"""

from __future__ import annotations

import copy
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from vllm import envs
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import (
    HEADER_MM_ABI,
    decode_digest,
    encode_digest,
)

MM_ABI_VERSION = "mm1"

_FILE_HASH_CHUNK_SIZE = 1024 * 1024
# 决定"媒体怎样变成占位 token 与内容摘要"的预处理包版本。
_PROCESSOR_SOFTWARE_PACKAGES = (
    "vllm",
    "vllm-ascend",
    "transformers",
    "pillow",
)
# MultiModalConfig 里会影响媒体预处理/占位排布/内容摘要的字段。
_MM_CONFIG_FIELDS = (
    "mm_processor_kwargs",
    "media_io_kwargs",
    "limit_per_prompt",
    "interleave_mm_strings",
    "video_pruning_rate",
)
# 这些字段是节点本地资源位置,不参与部署身份。
_ARTIFACT_LOCATION_PLACEHOLDER = "<model-artifact>"


def mm_abi_header_value(fingerprint: bytes) -> str:
    """Render a processor fingerprint as the ``X-Edge-Cloud-MM-ABI`` value."""
    return f"{MM_ABI_VERSION}:{encode_digest(fingerprint)}"


def validate_mm_abi_header(value: str | None) -> str:
    """Validate an ``X-Edge-Cloud-MM-ABI`` header value, returned unchanged."""
    if value is None:
        raise ValueError(f"missing required header {HEADER_MM_ABI}")
    declared, separator, encoded = value.partition(":")
    if not separator or declared != MM_ABI_VERSION or not encoded:
        raise ValueError(
            f"{HEADER_MM_ABI} must be "
            f"'{MM_ABI_VERSION}:<base64url fingerprint>'"
        )
    # decode_digest 顺带校验 32 字节长度
    decode_digest(encoded)
    return value


def compute_processor_fingerprint(model_config: Any) -> bytes:
    """Return a 32-byte digest identifying the multimodal deployment ABI.

    覆盖:已解析的模型/处理器配置 + 预处理包版本 + 媒体摘要算法。它能发现
    常规部署漂移(换模型版本、改 processor/media io 配置、升级 transformers),
    但不证明两份媒体输入执行等价(沿用上游 mm_hash 的既有风险)。
    全部字段经 ``getattr`` 读取:缺项按 None 参与摘要,两侧口径一致即可。"""
    mm_config = getattr(model_config, "multimodal_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    payload: dict[str, Any] = {
        "revision": getattr(model_config, "revision", None),
        "code_revision": getattr(model_config, "code_revision", None),
        "tokenizer_revision": getattr(model_config, "tokenizer_revision", None),
        "hf_config_commit": getattr(hf_config, "_commit_hash", None),
        "model_type": getattr(hf_text_config, "model_type", None),
        "architecture": getattr(model_config, "_architecture", None),
        # vLLM 自身的 config hash 覆盖那些手工列举字段容易漏掉的图选择
        # (dtype/量化、MM encoder attention backend/TP/FP8 等)。
        "model_config_hash": _config_hash(
            model_config,
            neutralized_fields={
                "model": _ARTIFACT_LOCATION_PLACEHOLDER,
                "model_weights": _ARTIFACT_LOCATION_PLACEHOLDER,
                "tokenizer": _ARTIFACT_LOCATION_PLACEHOLDER,
            },
            scrub_hf_locations=True,
        ),
        "multimodal_config_hash": _config_hash(
            mm_config,
            neutralized_fields={
                # 权重标定文件按内容而非本地路径参与身份
                "mm_encoder_fp8_scale_path": _file_content_identity(
                    getattr(mm_config, "mm_encoder_fp8_scale_path", None)
                )
            },
        ),
        "hf_image_processor_config": getattr(
            model_config, "hf_image_processor_config", None
        ),
        "mm_hasher_algorithm": getattr(
            envs, "VLLM_MM_HASHER_ALGORITHM", None
        ),
        "software_versions": {
            package: _package_version(package)
            for package in _PROCESSOR_SOFTWARE_PACKAGES
        },
    }
    for field in _MM_CONFIG_FIELDS:
        value = getattr(mm_config, field, None)
        if value is None:
            # mm_processor_kwargs / media_io_kwargs 是 InitVar,可能只落在
            # ModelConfig 上;缺 MultiModalConfig 时回退,保证两侧口径一致。
            value = getattr(model_config, field, None)
        payload[field] = value
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           default=repr)
    return hashlib.sha256(canonical.encode("utf-8")).digest()


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _config_hash(
    config: Any,
    *,
    neutralized_fields: dict[str, Any] | None = None,
    scrub_hf_locations: bool = False,
) -> str | None:
    """按"抹掉节点本地资源位置"后的配置求 hash(无 compute_hash 则返回 None)。"""
    if config is None:
        return None
    stable_config = copy.copy(config)
    for field, value in (neutralized_fields or {}).items():
        if hasattr(stable_config, field):
            setattr(stable_config, field, value)
    if scrub_hf_locations:
        for field in ("hf_config", "hf_text_config"):
            nested = getattr(stable_config, field, None)
            if nested is None:
                continue
            stable_nested = copy.deepcopy(nested)
            _scrub_name_or_path(stable_nested, set())
            setattr(stable_config, field, stable_nested)
    compute_hash = getattr(stable_config, "compute_hash", None)
    if not callable(compute_hash):
        return None
    return str(compute_hash())


def _scrub_name_or_path(value: Any, seen: set[int]) -> None:
    """递归把 HF 配置里的 ``_name_or_path`` 换成占位常量(去掉本地路径)。"""
    if value is None or isinstance(
        value, (bool, int, float, str, bytes)
    ):
        return
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(value, dict):
        if "_name_or_path" in value:
            value["_name_or_path"] = _ARTIFACT_LOCATION_PLACEHOLDER
        for nested in value.values():
            _scrub_name_or_path(nested, seen)
        return
    if isinstance(value, (list, tuple, set)):
        for nested in value:
            _scrub_name_or_path(nested, seen)
        return
    attributes = getattr(value, "__dict__", None)
    if not isinstance(attributes, dict):
        return
    if "_name_or_path" in attributes:
        attributes["_name_or_path"] = _ARTIFACT_LOCATION_PLACEHOLDER
    for nested in attributes.values():
        _scrub_name_or_path(nested, seen)


def _file_content_identity(path: Any) -> str | None:
    """按字节而非本地路径标识一个图输入(空路径返回 None)。"""
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(_FILE_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
