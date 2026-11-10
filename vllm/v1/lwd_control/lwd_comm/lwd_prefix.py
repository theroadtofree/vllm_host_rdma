# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Wire protocol for privacy-preserving edge-cloud prefix negotiation.

Derived from the reference branch's ``edge_cloud/prefix_protocol.py``; class
names are renamed to the local ``Lwd*`` convention, logic unchanged.  The
protocol intentionally uses ordinary OpenAI chat messages and response
headers so an unmodified OpenAI-compatible gateway can forward and account
for the internal request while the prompt stays on the edge.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

PROTOCOL_VERSION = "edge-cloud-prefix-v1"
PROTOCOL_VERSION_MM = "edge-cloud-prefix-v2"
BLOCK_HASH_PREFIX = "ecb1:"
TAIL_HASH_PREFIX = "ect1:"
DIGEST_SIZE = hashlib.sha256().digest_size

HEADER_PROTOCOL = "X-Edge-Cloud-Protocol"
HEADER_REQUEST_ID = "X-Edge-Cloud-Request-ID"
HEADER_EDGE_ID = "X-Edge-Cloud-Edge-Id"
HEADER_INSTANCE_ID = "X-Edge-Cloud-Instance-ID"
HEADER_BLOCK_SIZE = "X-Edge-Cloud-Block-Size"
HEADER_HIT_BLOCKS = "X-Edge-Cloud-Prefix-Hit-Blocks"
HEADER_HIT_TOKENS = "X-Edge-Cloud-Prefix-Hit-Tokens"
HEADER_ALIGNMENT_TOKENS = "X-Edge-Cloud-Alignment-Tokens"
HEADER_MM_ABI = "X-Edge-Cloud-MM-ABI"
HEADER_CLOUD_ID = "X-Edge-Cloud-Cloud-Id"
HEADER_PROMPT_TOKENS = "X-Edge-Cloud-Prompt-Tokens"
# 云内 DP:probe 放置结果(选中引擎的云内 dp_rank);缺省 0 向后兼容
HEADER_ENGINE_INDEX = "X-Edge-Cloud-Engine-Index"

_FULL_BLOCK_DOMAIN = b"\x01"
_TAIL_BLOCK_DOMAIN = b"\x02"
_MEDIA_BLOCK_DOMAIN = b"\x04"
_MEDIA_TAIL_BLOCK_DOMAIN = b"\x05"
_UINT32 = struct.Struct(">I")

# 多模态 ABI 预留:image=1,audio/video 后续。不可在运行时改写。
_MODALITY_IDS = MappingProxyType({"image": 1})


class _MediaItem(Protocol):
    """Duck-typed media identity description (协议层保持纯 stdlib)。"""

    modality: str
    digest: bytes
    offset: int
    length: int


def encode_digest(digest: bytes) -> str:
    """base64url(去填充)——摘要上线的唯一编码口径。"""
    if len(digest) != DIGEST_SIZE:
        raise ValueError(f"digest must contain {DIGEST_SIZE} bytes")
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def decode_digest(value: str) -> bytes:
    try:
        raw = value.encode("ascii")
        digest = base64.b64decode(
            raw + b"=" * (-len(raw) % 4), altchars=b"-_", validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError("invalid base64url digest") from exc
    if len(digest) != DIGEST_SIZE:
        raise ValueError(f"digest must contain {DIGEST_SIZE} bytes")
    return digest


def _encode_tokens(token_ids: Sequence[int]) -> bytes:
    encoded = bytearray()
    for token_id in token_ids:
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise TypeError("token IDs must be integers")
        if not 0 <= token_id <= 0xFFFFFFFF:
            raise ValueError("token IDs must fit in an unsigned 32-bit integer")
        encoded.extend(_UINT32.pack(token_id))
    return bytes(encoded)


@dataclass(frozen=True)
class LwdPrefixManifest:
    """Opaque hash chain sent from an edge to a cloud endpoint."""

    request_id: str
    prompt_tokens: int
    block_size: int
    full_block_hashes: tuple[bytes, ...]
    tail_hash: bytes | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must not be negative")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        expected_blocks = self.prompt_tokens // self.block_size
        if len(self.full_block_hashes) != expected_blocks:
            raise ValueError("full block hash count does not match prompt_tokens and block_size")
        for digest in self.full_block_hashes:
            if len(digest) != DIGEST_SIZE:
                raise ValueError(f"digest must contain {DIGEST_SIZE} bytes")
        has_partial_tail = self.prompt_tokens % self.block_size != 0
        if has_partial_tail != (self.tail_hash is not None):
            raise ValueError("tail_hash must be present exactly for a partial block")
        if self.tail_hash is not None and len(self.tail_hash) != DIGEST_SIZE:
            raise ValueError(f"tail digest must contain {DIGEST_SIZE} bytes")

    @property
    def full_block_count(self) -> int:
        return len(self.full_block_hashes)

    def to_messages(self) -> list[dict[str, str]]:
        messages = [
            {"role": "user", "content": BLOCK_HASH_PREFIX + encode_digest(d)}
            for d in self.full_block_hashes
        ]
        if self.tail_hash is not None:
            messages.append(
                {"role": "user",
                 "content": TAIL_HASH_PREFIX + encode_digest(self.tail_hash)})
        return messages

    def to_headers(self) -> dict[str, str]:
        """请求侧必需头(文档「Edge 向平台发送请求」表格口径)。

        ``prompt_tokens`` 只走头:真实输入长度不在 body 里(网关 DTO 转换会
        剥非标准字段),缺它云侧直接 400——所以它是 manifest 自带的一部分,
        不靠调用方外挂。"""
        return {
            HEADER_PROTOCOL: PROTOCOL_VERSION,
            HEADER_REQUEST_ID: self.request_id,
            HEADER_BLOCK_SIZE: str(self.block_size),
            HEADER_PROMPT_TOKENS: str(self.prompt_tokens),
        }

    @classmethod
    def from_openai_request(
        cls, headers: Mapping[str, str], body: Mapping[str, Any]
    ) -> LwdPrefixManifest:
        normalized_headers = {k.lower(): v for k, v in headers.items()}

        def required_header(name: str) -> str:
            try:
                return normalized_headers[name.lower()]
            except KeyError as exc:
                raise ValueError(f"missing required header {name}") from exc

        protocol = required_header(HEADER_PROTOCOL)
        if protocol != PROTOCOL_VERSION:
            raise ValueError(f"unsupported edge-cloud protocol {protocol!r}")
        request_id = required_header(HEADER_REQUEST_ID)
        try:
            block_size = int(required_header(HEADER_BLOCK_SIZE))
        except ValueError as exc:
            raise ValueError("edge-cloud block size must be an integer") from exc

        messages = body.get("messages")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        full_hashes: list[bytes] = []
        tail_hash: bytes | None = None
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping) or message.get("role") != "user":
                raise ValueError("every manifest message must have role=user")
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("every manifest message must have string content")
            if content.startswith(BLOCK_HASH_PREFIX):
                if tail_hash is not None:
                    raise ValueError("a full block hash cannot follow the tail hash")
                full_hashes.append(decode_digest(content[len(BLOCK_HASH_PREFIX):]))
            elif content.startswith(TAIL_HASH_PREFIX):
                if tail_hash is not None or index != len(messages) - 1:
                    raise ValueError("the tail hash must be the final unique message")
                tail_hash = decode_digest(content[len(TAIL_HASH_PREFIX):])
            else:
                raise ValueError("message content is not an edge-cloud hash")

        # prompt_tokens 只从 header 取(与参考分支一致):NewAPI 等网关
        # DTO 转换会剥掉 body 非标准字段。
        # 严格十进制:不用 int() —— 它会收下 "+1024"/" 1024"/"1_0"/全角数字,
        # 而这些"看着能解析"的值一旦落到 manifest 上,长度与摘要条数就不再
        # 自证一致,错配只会表现为命中异常而不是 400。
        prompt_tokens_raw = normalized_headers.get(HEADER_PROMPT_TOKENS.lower())
        if prompt_tokens_raw is None:
            raise ValueError(f"missing required header {HEADER_PROMPT_TOKENS}")
        if not (prompt_tokens_raw.isascii() and prompt_tokens_raw.isdecimal()):
            raise ValueError(
                f"{HEADER_PROMPT_TOKENS} must be a non-negative integer"
            )
        prompt_tokens_value = int(prompt_tokens_raw)
        return cls(
            request_id=request_id,
            prompt_tokens=prompt_tokens_value,
            block_size=block_size,
            full_block_hashes=tuple(full_hashes),
            tail_hash=tail_hash,
        )


class LwdPrefixHasher:
    """Build a tenant-scoped HMAC-SHA256 chain at KV block boundaries."""

    def __init__(
        self,
        tenant_key: bytes,
        block_size: int,
        processor_fingerprint: bytes | None = None,
    ) -> None:
        if len(tenant_key) < 16:
            raise ValueError("tenant_key must contain at least 16 bytes")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if processor_fingerprint is not None and len(processor_fingerprint) != DIGEST_SIZE:
            raise ValueError(f"processor_fingerprint must contain {DIGEST_SIZE} bytes")
        self._tenant_key = tenant_key
        self.block_size = block_size
        self._processor_fingerprint = processor_fingerprint
        self._seed = hmac.digest(tenant_key, PROTOCOL_VERSION.encode(), "sha256")

    @staticmethod
    def _validate_media_items(
        media_items: Sequence[_MediaItem], prompt_tokens: int
    ) -> tuple[tuple[str, bytes, int, int], ...]:
        validated: list[tuple[str, bytes, int, int]] = []
        for item in media_items:
            if item.modality not in _MODALITY_IDS:
                raise ValueError(f"unsupported media modality {item.modality!r}")
            if len(item.digest) != DIGEST_SIZE:
                raise ValueError(f"media digest must contain {DIGEST_SIZE} bytes")
            if item.offset < 0:
                raise ValueError("media offset must not be negative")
            if item.length <= 0:
                raise ValueError("media length must be positive")
            if item.offset + item.length > prompt_tokens:
                raise ValueError("media item range exceeds the prompt length")
            validated.append((item.modality, item.digest, item.offset, item.length))
        validated.sort(key=lambda entry: entry[2])
        previous_end = 0
        for _, _, offset, length in validated:
            if offset < previous_end:
                raise ValueError("media item ranges must not overlap")
            previous_end = offset + length
        return tuple(validated)

    def _encode_media_fields(
        self, covered_items: Sequence[tuple[str, bytes, int, int]]
    ) -> bytes:
        fingerprint = self._processor_fingerprint
        assert fingerprint is not None
        encoded = bytearray(fingerprint)
        encoded.extend(_UINT32.pack(len(covered_items)))
        for modality, digest, offset, length in covered_items:
            encoded.append(_MODALITY_IDS[modality])
            encoded.extend(digest)
            encoded.extend(_UINT32.pack(offset))
            encoded.extend(_UINT32.pack(length))
        return bytes(encoded)

    @staticmethod
    def _covered_media_items(
        validated_items: tuple[tuple[str, bytes, int, int], ...],
        block_start: int,
        block_end: int,
    ) -> list[tuple[str, bytes, int, int]]:
        return [item for item in validated_items
                if item[2] < block_end and block_start < item[2] + item[3]]

    def chain_extend(
        self,
        prev_chain: Sequence[bytes],
        token_ids: Sequence[int],
    ) -> list[bytes]:
        """增量续链(append-only):只对新增满块算 HMAC,消除逐步全量重建。

        prev_chain 为已建满块链(末元素即链式 parent,空链用种子);
        token_ids 为全量序列(prompt+生成)。只对
        [len(prev_chain)*block_size, N) 内的新满块哈希,返回新增摘要
        (不含 prev_chain)。与 build_manifest 的全量块循环逐字节同源
        (同域分隔符/同 parent 链式),产出链与全量重建逐位相等。

        仅支持纯文本路径(无 media_items):生成段链场景不含多模态
        覆盖项;带 media 的调用方须走 build_manifest 全量重建。"""
        full_total = len(token_ids) // self.block_size
        done = len(prev_chain)
        if full_total <= done:
            return []
        parent = prev_chain[-1] if prev_chain else self._seed
        region = token_ids[done * self.block_size : full_total * self.block_size]
        token_bytes = _encode_tokens(region)
        encoded_block_size = _UINT32.pack(self.block_size)
        out: list[bytes] = []
        for index in range(full_total - done):
            start = index * self.block_size * _UINT32.size
            block_bytes = token_bytes[start : start + self.block_size * _UINT32.size]
            parent = hmac.digest(
                self._tenant_key,
                _FULL_BLOCK_DOMAIN + parent + encoded_block_size + block_bytes,
                "sha256",
            )
            out.append(parent)
        return out

    def build_manifest(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int],
        media_items: Sequence[_MediaItem] = (),
    ) -> LwdPrefixManifest:
        prompt_tokens = len(prompt_token_ids)
        validated_items = self._validate_media_items(media_items, prompt_tokens)
        if validated_items and self._processor_fingerprint is None:
            raise ValueError("processor_fingerprint is required when media items are present")

        token_bytes = _encode_tokens(prompt_token_ids)
        parent = self._seed
        full_hashes: list[bytes] = []
        full_count = prompt_tokens // self.block_size
        encoded_block_size = _UINT32.pack(self.block_size)

        for index in range(full_count):
            block_start = index * self.block_size
            block_end = block_start + self.block_size
            start = block_start * _UINT32.size
            end = start + self.block_size * _UINT32.size
            block_bytes = token_bytes[start:end]
            covered = self._covered_media_items(validated_items, block_start, block_end)
            if covered:
                message = (
                    _MEDIA_BLOCK_DOMAIN + parent + encoded_block_size
                    + block_bytes + self._encode_media_fields(covered))
            else:
                message = _FULL_BLOCK_DOMAIN + parent + encoded_block_size + block_bytes
            parent = hmac.digest(self._tenant_key, message, "sha256")
            full_hashes.append(parent)

        remainder = prompt_tokens % self.block_size
        tail_hash = None
        if remainder:
            tail_start = full_count * self.block_size
            tail_bytes = token_bytes[tail_start * _UINT32.size:]
            covered = self._covered_media_items(validated_items, tail_start, prompt_tokens)
            if covered:
                message = (
                    _MEDIA_TAIL_BLOCK_DOMAIN + parent + _UINT32.pack(remainder)
                    + tail_bytes + self._encode_media_fields(covered))
            else:
                message = _TAIL_BLOCK_DOMAIN + parent + _UINT32.pack(remainder) + tail_bytes
            tail_hash = hmac.digest(self._tenant_key, message, "sha256")

        return LwdPrefixManifest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            block_size=self.block_size,
            full_block_hashes=tuple(full_hashes),
            tail_hash=tail_hash,
        )


@dataclass(frozen=True)
class LwdProbeResult:
    """Cloud instance selection and reserved prefix returned to the edge."""

    request_id: str
    instance_id: str
    block_size: int
    hit_blocks: int
    hit_tokens: int
    # 云侧命中对齐(coordinator.cache_hit_alignment_tokens:细粒度部署为压缩
    # state 边界,如 128;否则 scheduler LCM):边侧按它对 probe 命中取整、
    # 对 chunk 做 mamba 对齐重切。0 = 未公告(旧版云/降级),边侧不重切
    alignment_tokens: int = 0
    # 路由回执的云 id:多云选路时由中央调度器/网关填;None=单云或未启用选路
    cloud_id: int | None = None
    # 云内 DP 放置结果:命中/预留落在的云内引擎序号(dp_rank);0=单引擎或
    # 旧版云(未公告),边侧路由登记按 0 处理,向后兼容
    engine_index: int = 0

    def __post_init__(self) -> None:
        if not self.request_id or not self.instance_id:
            raise ValueError("request_id and instance_id must not be empty")
        if self.block_size <= 0 or self.hit_blocks < 0 or self.hit_tokens < 0:
            raise ValueError("invalid prefix hit values")
        if self.hit_tokens != self.hit_blocks * self.block_size:
            raise ValueError("hit_tokens must equal hit_blocks * block_size")
        if self.alignment_tokens < 0:
            raise ValueError("alignment_tokens must not be negative")
        if self.engine_index < 0:
            raise ValueError("engine_index must not be negative")

    def to_headers(self) -> dict[str, str]:
        headers = {
            HEADER_PROTOCOL: PROTOCOL_VERSION,
            HEADER_REQUEST_ID: self.request_id,
            HEADER_INSTANCE_ID: self.instance_id,
            HEADER_BLOCK_SIZE: str(self.block_size),
            HEADER_HIT_BLOCKS: str(self.hit_blocks),
            HEADER_HIT_TOKENS: str(self.hit_tokens),
            HEADER_ALIGNMENT_TOKENS: str(self.alignment_tokens),
        }
        if self.cloud_id is not None:
            headers[HEADER_CLOUD_ID] = str(self.cloud_id)
        if self.engine_index:
            headers[HEADER_ENGINE_INDEX] = str(self.engine_index)
        return headers

    def to_control_message(
        self, protocol: str = PROTOCOL_VERSION, mm_abi: str | None = None
    ) -> dict[str, Any]:
        """Build the JSON object carried inside an OpenAI delta.content.

        SSE 形态下 Probe 结果走 delta.content 而非响应头——网关 DTO
        转换不会丢失消息体,响应头可能被吃掉。"""
        message: dict[str, Any] = {
            "type": "edge_cloud_probe",
            "protocol": protocol,
            "request_id": self.request_id,
            "instance_id": self.instance_id,
            "block_size": self.block_size,
            "hit_blocks": self.hit_blocks,
            "hit_tokens": self.hit_tokens,
            "alignment_tokens": self.alignment_tokens,
        }
        if self.cloud_id is not None:
            message["cloud_id"] = self.cloud_id
        if self.engine_index:
            message["engine_index"] = self.engine_index
        if mm_abi is not None:
            message["mm_abi"] = mm_abi
        return message

    @classmethod
    def from_control_message(cls, message: Mapping[str, Any]) -> LwdProbeResult:
        """Validate the probe payload from an SSE delta.content JSON object."""
        if message.get("type") != "edge_cloud_probe":
            raise ValueError("expected an edge_cloud_probe control message")
        for name in ("request_id", "instance_id"):
            if not isinstance(message.get(name), str):
                raise ValueError(f"probe {name} must be a string")
        for name in ("block_size", "hit_blocks", "hit_tokens"):
            if type(message.get(name)) is not int:
                raise ValueError(f"probe {name} must be an integer")
        cloud_id = message.get("cloud_id")
        if cloud_id is not None and type(cloud_id) is not int:
            raise ValueError("probe cloud_id must be an integer or null")
        # 旧版云不公告 engine_index:缺省 0(单引擎语义),向后兼容
        engine_index = message.get("engine_index", 0)
        if type(engine_index) is not int or engine_index < 0:
            raise ValueError("probe engine_index must be a non-negative integer")
        return cls(
            request_id=message["request_id"],
            instance_id=message["instance_id"],
            block_size=message["block_size"],
            hit_blocks=message["hit_blocks"],
            hit_tokens=message["hit_tokens"],
            alignment_tokens=int(message.get("alignment_tokens", 0)),
            cloud_id=cloud_id,
            engine_index=engine_index,
        )

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> LwdProbeResult:
        values = {k.lower(): v for k, v in headers.items()}

        def required(name: str) -> str:
            try:
                return values[name.lower()]
            except KeyError as exc:
                raise ValueError(f"missing required header {name}") from exc

        if required(HEADER_PROTOCOL) != PROTOCOL_VERSION:
            raise ValueError("unsupported edge-cloud protocol")
        try:
            cloud_id_raw = values.get(HEADER_CLOUD_ID.lower())
            return cls(
                request_id=required(HEADER_REQUEST_ID),
                instance_id=required(HEADER_INSTANCE_ID),
                block_size=int(required(HEADER_BLOCK_SIZE)),
                hit_blocks=int(required(HEADER_HIT_BLOCKS)),
                hit_tokens=int(required(HEADER_HIT_TOKENS)),
                # 旧版云无此头:缺省 0(不公告、不重切),向后兼容
                alignment_tokens=int(values.get(
                    HEADER_ALIGNMENT_TOKENS.lower(), "0"
                )),
                cloud_id=int(cloud_id_raw) if cloud_id_raw else None,
                # 旧版云无此头:缺省 0(单引擎语义),向后兼容
                engine_index=int(values.get(
                    HEADER_ENGINE_INDEX.lower(), "0"
                )),
            )
        except ValueError as exc:
            raise ValueError("cloud prefix result contains invalid integers") from exc


@dataclass(frozen=True)
class LwdUsage:
    """Usage counters returned in the final OpenAI streaming chunk."""

    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int

    def __post_init__(self) -> None:
        if min(self.prompt_tokens, self.completion_tokens, self.cached_tokens) < 0:
            raise ValueError("usage counters must not be negative")
        if self.cached_tokens > self.prompt_tokens:
            raise ValueError("cached_tokens must not exceed prompt_tokens")

    def to_openai_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": self.cached_tokens},
        }