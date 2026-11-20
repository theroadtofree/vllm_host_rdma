# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Business messages of the LWD edge-cloud control plane.

Ported from the `prefill_only_chunked_prefill_mm` branch. Messages carry
scheduling metadata only; tensors travel on the data plane. The control link
treats them as opaque payloads.

Fields may only be appended with defaults. Any other change bumps
`LWD_MESSAGE_VERSION`, which the link handshake checks on both sides.
"""

import dataclasses
import json
from typing import Any, TypeAlias, TypeVar, cast

import msgspec

T = TypeVar("T")

# Version 5: LwdRequestNotify carries `structured_outputs`. An old cloud would
# drop the constraint silently (msgspec skips unknown fields on decode), so
# mixed versions fail the handshake instead of serving unstructured output.
LWD_MESSAGE_VERSION = 5

LWD_NOT_FINISHED = -1
"""`LwdC2eNotify.finish_reasons` value of a request still running."""


class LwdRangeItem(msgspec.Struct, gc=False):  # type: ignore[call-arg]
    """One request's prompt chunk inside a prefill batch."""

    request_id: str
    offset: int
    """Chunk start in the prompt; the cloud requires it to equal the number
    of tokens it has already computed for the request."""
    num_tokens: int
    has_mrope: bool = False
    mrope_positions: bytes = b""
    """Legacy per-chunk positions; empty on the current protocol path. Full
    prompt positions travel in LwdRequestNotify."""
    is_refill: bool = False
    """Out-of-band embedding refill after cloud preemption."""
    prefix_nonce: int = 0
    """Generation of the request that owns this chunk."""


class LwdRangeNotify(msgspec.Struct, gc=False, tag=True):  # type: ignore[call-arg]
    """Edge -> cloud. Announces the chunks of one prefill batch.

    Duplicates are idempotent per (request_id, offset).
    """

    seqno: int
    """Batch number pairing the batch with its data-plane transfer."""
    items: list[LwdRangeItem] = []


class LwdRequestNotify(msgspec.Struct, gc=False, tag=True):  # type: ignore[call-arg]
    """Edge -> cloud. Registers a request before its prefill chunks.

    Sampling fields are the ones that change cloud-side token selection;
    detokenizer-level parameters such as stop strings stay on the edge.
    """

    request_id: str
    num_prompt_tokens: int
    max_tokens: int = 16
    block_hashes: list[bytes] = []
    """Full-block hash chain of the prompt for cloud prefix cache hits; empty
    makes the cloud hash locally."""
    prefix_nonce: int = 0
    """Nonzero while the edge waits for a matching LwdPrefixReadyNotify."""
    skip_reading_prefix_cache: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: int | None = None
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    ignore_eos: bool = False
    stop_token_ids: list[int] = []
    min_tokens: int = 0
    eos_token_id: int | None = None
    """Resolved by the edge tokenizer; the cloud cannot derive it. None leaves
    `max_tokens` as the only stop condition on the cloud."""
    mrope_positions: bytes = b""
    """Contiguous int64 [3, num_prompt_tokens] M-RoPE positions of the full
    prompt; empty when unused."""
    prefix_hit_tokens: int = 0
    """Exact prefix boundary trimmed by the edge after its gateway probe.

    Native nonce negotiation ignores this field and uses its reservation. The
    gateway path must consume the same snapshot at this boundary (or discard a
    stale reservation for a fail-open zero hit), never renegotiate the hit.
    """
    prefix_probe_succeeded: bool = False
    """Whether a gateway probe completed and created the matching reservation.

    False also covers placement-only and fail-open. Keeping this separate from
    a zero hit preserves the HMAC manifest/finish chain of successful cache
    misses while allowing fail-open requests to discard stale reservations.
    """
    structured_outputs: str = ""
    """`lwd_encode_structured_outputs` wire JSON of the validated
    `StructuredOutputsParams`; empty for a plain request. The cloud rebuilds
    the params, compiles the grammar and applies the sampling bitmask; the
    edge compiles its own copy only for the picker's grammar-ready filter."""
    reasoning_ended: bool | None = None
    """Initial reasoning-ended state computed by the edge frontend; the cloud
    seeds its grammar gate from this instead of defaulting to thinking=True."""
    reasoning_parser_kwargs: str = ""
    """`json.dumps` of the request's `reasoning_parser_kwargs` (chat-template
    kwargs incl. enable_thinking/thinking/reasoning_effort); empty when absent.
    The cloud feeds this to `StructuredOutputManager._get_reasoner` so its
    grammar gate matches the edge's prompt rendering."""


def lwd_encode_structured_outputs(params: Any) -> str:
    """JSON wire form of a validated `StructuredOutputsParams`.

    The private fields (`_backend*`, `init=False`) never ride the field
    dict: `params_cls(**fields)` rejects them as constructor keywords, so the
    edge-chosen backend travels next to the fields in the envelope and is
    re-attached after decode. This is the single implementation shared by the
    edge packer and the cloud rebuild.
    """
    data = dataclasses.asdict(params)
    backend = data.pop("_backend", None)
    data.pop("_backend_was_auto", None)
    return json.dumps({"fields": data, "backend": backend})


def lwd_decode_structured_outputs(payload: str, params_cls: type[T]) -> T | None:
    """Rebuild the constraint params from the wire JSON; None if plain.

    Args:
        payload: The `LwdRequestNotify.structured_outputs` string.
        params_cls: The dataclass to rebuild (its public, `init=True` fields
            must match the envelope's `fields` dict).

    Returns:
        The rebuilt params with `_backend` re-attached, or None when the
        payload is the plain-request empty string.

    Raises:
        ValueError: The payload is not a well-formed envelope.
    """
    if not payload:
        return None
    try:
        envelope = json.loads(payload)
        fields = envelope["fields"]
        backend = envelope["backend"]
        if not isinstance(fields, dict):
            raise TypeError(f"fields must be a dict, got {type(fields).__name__}")
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"malformed structured_outputs payload {payload!r}") from exc
    try:
        params = params_cls(**fields)
    except TypeError as exc:
        raise ValueError(
            f"structured_outputs fields do not fit {params_cls}: {exc}"
        ) from exc
    # init=False private field: cannot ride the constructor, reattach here.
    cast(Any, params)._backend = backend
    return params


def lwd_encode_reasoning_parser_kwargs(kwargs: dict[str, Any] | None) -> str:
    """Wire JSON of a request's `reasoning_parser_kwargs`; empty when absent."""
    return "" if kwargs is None else json.dumps(kwargs)


def lwd_decode_reasoning_parser_kwargs(payload: str) -> dict[str, Any] | None:
    """Rebuild `reasoning_parser_kwargs` from the wire JSON; None when absent."""
    return None if not payload else json.loads(payload)


LWD_SAMPLING_FIELDS: tuple[tuple[str, object], ...] = (
    ("max_tokens", 16),
    ("temperature", 1.0),
    ("top_p", 1.0),
    ("top_k", 0),
    ("min_p", 0.0),
    ("seed", None),
    ("repetition_penalty", 1.0),
    ("presence_penalty", 0.0),
    ("frequency_penalty", 0.0),
    ("ignore_eos", False),
    ("stop_token_ids", []),
    ("min_tokens", 0),
)
"""(field, default) of the `SamplingParams` arguments `LwdRequestNotify`
carries; the edge packs and the cloud rebuilds them from this one table.
`eos_token_id` is not a `SamplingParams` argument and travels on its own."""
LWD_SAMPLING_DEFAULTS = dict(LWD_SAMPLING_FIELDS)


class LwdAbortNotify(msgspec.Struct, gc=False, tag=True):  # type: ignore[call-arg]
    """Edge -> cloud. Aborts a request."""

    request_id: str
    prefix_nonce: int = 0
    """Generation to abort; zero is the legacy form."""


class LwdGenChainNotify(msgspec.Struct, gc=False, tag=True):  # type: ignore[call-arg]
    """Edge -> cloud. Full prompt+generated hash chain; replaces prior chains."""

    request_id: str
    block_hashes: list[bytes] = []
    block_size: int = 0


class LwdFinishNotify(msgspec.Struct, gc=False, tag=True):  # type: ignore[call-arg]
    """Edge -> cloud. Final full chain, usage counters and publish decision."""

    request_id: str
    prompt_tokens: int
    completion_tokens: int
    full_block_hashes: list[bytes] = []
    publish_cache: bool = True


class LwdCloudParamsNotify(msgspec.Struct, gc=False, tag=True):  # type: ignore[call-arg]
    """Cloud -> edge. Scheduling limits both sides must agree on.

    A mismatch does not break the link: it makes the cloud hold prefills and
    serialize them behind decodes without any error, so the edge must check
    these against its own configuration before serving.
    """

    max_num_batched_tokens: int
    max_num_seqs: int
    num_speculative_tokens: int
    dp_coordinator_addr: str = ""
    """Frontend stats endpoint of this DP rank's coordinator, for the edge to
    trigger the first DP wave; empty when the cloud runs no coordinator
    (single DP or non-MoE engines)."""
    prefix_caching: bool = False
    hash_block_size: int = 0
    prefix_alignment_tokens: int = 0
    prefix_alignment_required: bool = False
    prefix_use_eagle: bool = False
    prefix_hash_block_size: int = 0


class LwdPrefixReadyNotify(msgspec.Struct, gc=False, tag=True):  # type: ignore[call-arg]
    """Cloud -> edge. Result of one nonce-scoped prefix reservation."""

    request_id: str
    hit_tokens: int
    nonce: int
    error: str = ""


class LwdRefillNotify(msgspec.Struct, gc=False, tag=True):  # type: ignore[call-arg]
    """Cloud -> edge. Re-embed an absolute prompt range after preemption."""

    request_id: str
    start_row: int
    end_row: int
    prefix_nonce: int = 0
    """Request generation that requested this refill; zero is the legacy form."""


class LwdC2eNotify(msgspec.Struct, gc=False, tag=True):  # type: ignore[call-arg]
    """Cloud -> edge. Metadata of one cloud step.

    Arrives before the step's hidden states so the edge can post a receive of
    the exact size. `hidden_num_elements == 0` only finishes requests.
    """

    hidden_num_elements: int
    top_id_ths: list[list[int]]
    num_accepted_tokens: list[int]
    req_ids: list[str]
    finish_reasons: list[int] = []
    """Per request, aligned with `req_ids`: `LWD_NOT_FINISHED` or a
    `FinishReason` value."""
    down_seqno: int = -1
    """Channel sequence number of this step's DOWN hidden-state transfer; the
    edge pairs its receive with it."""
    token_ids: list[list[int]] = []
    """Diagnostics only: accepted token ids per request; empty unless the
    edge skips sampling."""
    prefix_nonces: list[int] = []
    """Generation of each request row, aligned with req_ids."""


class LwdCloudHeartbeat(msgspec.Struct, gc=False, tag=True):  # type: ignore[call-arg]
    """Cloud -> edge. Liveness keepalive of one cloud DP rank.

    The edge keeps per-connection liveness accounts: no beat within the
    silent threshold (the edge engine's ``LWD_HELLO_SILENT_S``) is a
    silent verdict (the edge
    tears the data-plane
    links down and gates new requests until the next beat), and a
    ``boot_id`` change is a cloud restart verdict. The boot id is random
    per cloud process launch, i.e. the epoch of this DP rank."""

    boot_id: str


LwdEdgeToCloudMessage: TypeAlias = (
    LwdRequestNotify
    | LwdRangeNotify
    | LwdAbortNotify
    | LwdGenChainNotify
    | LwdFinishNotify
)
LwdCloudToEdgeMessage: TypeAlias = (
    LwdCloudParamsNotify
    | LwdCloudHeartbeat
    | LwdC2eNotify
    | LwdPrefixReadyNotify
    | LwdRefillNotify
)
LwdMessage: TypeAlias = LwdEdgeToCloudMessage | LwdCloudToEdgeMessage

_LWD_EDGE_TO_CLOUD_DECODER = msgspec.msgpack.Decoder(LwdEdgeToCloudMessage)
_LWD_CLOUD_TO_EDGE_DECODER = msgspec.msgpack.Decoder(LwdCloudToEdgeMessage)


def lwd_encode_message(message: LwdMessage) -> bytes:
    return msgspec.msgpack.encode(message)


def lwd_decode_edge_to_cloud(data: bytes | memoryview) -> LwdEdgeToCloudMessage:
    """Decode a payload received by the cloud.

    Raises:
        msgspec.DecodeError: Not valid msgpack.
        msgspec.ValidationError: Not an edge -> cloud message.
    """
    return _LWD_EDGE_TO_CLOUD_DECODER.decode(data)


def lwd_decode_cloud_to_edge(data: bytes | memoryview) -> LwdCloudToEdgeMessage:
    """Decode a payload received by the edge.

    Raises:
        msgspec.DecodeError: Not valid msgpack.
        msgspec.ValidationError: Not a cloud -> edge message.
    """
    return _LWD_CLOUD_TO_EDGE_DECODER.decode(data)
