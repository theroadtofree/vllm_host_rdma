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


def _encode_digest(digest: bytes) -> str:
    if len(digest) != DIGEST_SIZE:
        raise ValueError(f"digest must contain {DIGEST_SIZE} bytes")
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _decode_digest(value: str) -> bytes:
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
            {"role": "user", "content": BLOCK_HASH_PREFIX + _encode_digest(d)}
            for d in self.full_block_hashes
        ]
        if self.tail_hash is not None:
            messages.append(
                {"role": "user",
                 "content": TAIL_HASH_PREFIX + _encode_digest(self.tail_hash)})
        return messages

    def to_headers(self) -> dict[str, str]:
        return {
            HEADER_PROTOCOL: PROTOCOL_VERSION,
            HEADER_REQUEST_ID: self.request_id,
            HEADER_BLOCK_SIZE: str(self.block_size),
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
                full_hashes.append(_decode_digest(content[len(BLOCK_HASH_PREFIX):]))
            elif content.startswith(TAIL_HASH_PREFIX):
                if tail_hash is not None or index != len(messages) - 1:
                    raise ValueError("the tail hash must be the final unique message")
                tail_hash = _decode_digest(content[len(TAIL_HASH_PREFIX):])
            else:
                raise ValueError("message content is not an edge-cloud hash")

        prompt_tokens_value = body.get("edge_cloud_prompt_tokens")
        if not isinstance(prompt_tokens_value, int) or isinstance(prompt_tokens_value, bool):
            raise ValueError("edge_cloud_prompt_tokens must be an integer")
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
    # 云侧对齐网格(scheduler_block_size,各 KV 组块大小 LCM):边侧按它
    # 对 chunk 做 mamba 对齐重切。0 = 未公告(旧版云/降级),边侧不重切
    alignment_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.request_id or not self.instance_id:
            raise ValueError("request_id and instance_id must not be empty")
        if self.block_size <= 0 or self.hit_blocks < 0 or self.hit_tokens < 0:
            raise ValueError("invalid prefix hit values")
        if self.hit_tokens != self.hit_blocks * self.block_size:
            raise ValueError("hit_tokens must equal hit_blocks * block_size")
        if self.alignment_tokens < 0:
            raise ValueError("alignment_tokens must not be negative")

    def to_headers(self) -> dict[str, str]:
        return {
            HEADER_PROTOCOL: PROTOCOL_VERSION,
            HEADER_REQUEST_ID: self.request_id,
            HEADER_INSTANCE_ID: self.instance_id,
            HEADER_BLOCK_SIZE: str(self.block_size),
            HEADER_HIT_BLOCKS: str(self.hit_blocks),
            HEADER_HIT_TOKENS: str(self.hit_tokens),
            HEADER_ALIGNMENT_TOKENS: str(self.alignment_tokens),
        }

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