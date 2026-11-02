# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-model-family request metadata the edge extracts at admission.

Only the edge holds the real prompt. Model families whose cloud-side forward
reads more than the prompt embeddings get that extra input derived here, once
per request, in a form that carries no token id: M-RoPE positions for the
Qwen-VL family; image placeholder runs and Engram tail compressed ids for
DeepSeek V4.1. The edge scheduler and engine use the family selected for the
model; the cloud checks what arrives with `LwdRequestMetadataLayout`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.lwd_logger import lwd_logger
from vllm.v1.lwd_control.lwd_model.lwd_engram_hasher import LwdEngramHasher

if TYPE_CHECKING:
    from collections.abc import Callable

    from vllm.config import ModelConfig, VllmConfig
    from vllm.v1.lwd_control.lwd_comm.lwd_messages import LwdRequestNotify
    from vllm.v1.request import Request

    LwdMropePositionsFn = Callable[..., tuple[torch.Tensor, Any]]

_LWD_DEEPSEEK_V41_ARCHITECTURE = "DeepseekV41ForCausalLM"

LWD_IMAGE_PLACEHOLDER_SPAN_WIDTH = 3
"""`(start, length, token_id)` of one image placeholder run."""

# Width of the id range, starting at `image_token_id`, that the DeepSeek MoE
# router sends through the vision routing bias: the `+ 5` of the vLLM
# `_dsv4_topk_kernel` sentinel test and vllm-ascend's
# DEEPSEEK_V4_IMAGE_SENTINEL_COUNT; neither is importable from here.
_LWD_DEEPSEEK_IMAGE_ROUTING_IDS = 5


@dataclass(frozen=True)
class LwdRequestMetadata:
    """Whole-request values `LwdRequestNotify` carries for the model family;
    an empty field is not used by the family."""

    image_placeholder_spans: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class LwdRequestMetadataLayout:
    """What the cloud model expects in the `LwdRequestNotify` of a request."""

    image_token_id: int | None
    """First image placeholder id; None when spans must be empty."""

    @classmethod
    def from_model_config(cls, model_config: ModelConfig) -> LwdRequestMetadataLayout:
        """Layout of the cloud's own model config."""
        image_token_id = None
        if model_config.architecture == _LWD_DEEPSEEK_V41_ARCHITECTURE:
            image_token_id = model_config.hf_config.image_token_id
        return cls(image_token_id)

    def validate(self, notify: LwdRequestNotify) -> None:
        """Check the metadata of a request notify against the cloud model.

        Raises:
            ValueError: A field disagrees with the cloud model or the prompt.
        """
        if notify.image_placeholder_spans:
            self._validate_spans(notify)

    def _validate_spans(self, notify: LwdRequestNotify) -> None:
        spans = notify.image_placeholder_spans
        request_id = notify.request_id
        if self.image_token_id is None:
            raise ValueError(
                f"[LWD] request {request_id}: image placeholder spans sent to "
                f"a cloud model that does not use them"
            )
        if len(spans) % LWD_IMAGE_PLACEHOLDER_SPAN_WIDTH:
            raise ValueError(
                f"[LWD] request {request_id}: image placeholder spans length "
                f"{len(spans)} is not a multiple of "
                f"{LWD_IMAGE_PLACEHOLDER_SPAN_WIDTH}"
            )
        image_ids = range(
            self.image_token_id,
            self.image_token_id + _LWD_DEEPSEEK_IMAGE_ROUTING_IDS,
        )
        end = 0
        for i in range(0, len(spans), LWD_IMAGE_PLACEHOLDER_SPAN_WIDTH):
            start, length, token_id = spans[i : i + LWD_IMAGE_PLACEHOLDER_SPAN_WIDTH]
            if start < end or length <= 0 or token_id not in image_ids:
                raise ValueError(
                    f"[LWD] request {request_id}: malformed image placeholder "
                    f"span {(start, length, token_id)} (previous end {end}, "
                    f"image ids [{image_ids.start}, {image_ids.stop}))"
                )
            end = start + length
        if end > notify.num_prompt_tokens:
            raise ValueError(
                f"[LWD] request {request_id}: image placeholder span ends at "
                f"{end}, past the prompt length {notify.num_prompt_tokens}"
            )


class LwdModelFamily:
    """Model family whose cloud reads the prompt embeddings only: nothing to
    extract. Subclasses override what their cloud forward needs."""

    def admit(self, request: Request) -> None:
        """Precompute at admission what the range notifies of `request`
        slice."""

    def request_metadata(self, request: Request) -> LwdRequestMetadata:
        """Whole-request values announced with the request."""
        return LwdRequestMetadata()

    def has_mrope_positions(self, request_id: str) -> bool:
        """Whether the request carries M-RoPE positions. Read per range
        item, so it must not serialize them."""
        return False

    def prompt_mrope_positions(self, request_id: str) -> bytes | None:
        """Raw int64 `[3, num_prompt_tokens]` M-RoPE positions of the whole
        prompt; None when the request carries none."""
        return None

    def release(self, request_id: str) -> None:
        """Drop what `admit` kept for the request."""

    def held_request_ids(self) -> list[str]:
        """Requests that still hold admission state."""
        return []

    def engram_fingerprint(self) -> str:
        """Digest of the Engram hash parameters; empty without Engram."""
        return ""


class LwdMropeModelFamily(LwdModelFamily):
    """Qwen-VL family: the cloud's M-RoPE positions of multimodal prompts.

    Positions need the real prompt and the image grids, so the edge computes
    them for the whole prompt at admission and the request notify carries
    them whole.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        self._vllm_config = vllm_config
        # Resolved on the first multimodal request; model classes lacking it
        # fail fast there.
        self._positions_fn: LwdMropePositionsFn | None = None
        # Whole-prompt [3, N] CPU positions per multimodal request.
        self._positions: dict[str, torch.Tensor] = {}

    def admit(self, request: Request) -> None:
        if not _lwd_request_has_mm(request):
            return
        assert request.prompt_token_ids is not None
        mm_features = [f for f in request.mm_features if f.modality != "prompt_embeds"]
        positions, _delta = self._resolve_positions_fn(request)(
            input_tokens=list(request.prompt_token_ids),
            mm_features=mm_features,
            config=self._vllm_config.model_config.hf_config,
        )
        assert positions.shape[1] == request.num_prompt_tokens, (
            f"[LWD] mrope positions width {positions.shape[1]} != prompt "
            f"len {request.num_prompt_tokens} (request {request.request_id})"
        )
        self._positions[request.request_id] = positions

    def _resolve_positions_fn(self, request: Request) -> LwdMropePositionsFn:
        if self._positions_fn is None:
            from vllm.model_executor.models import ModelRegistry

            model_config = self._vllm_config.model_config
            model_cls, _ = ModelRegistry.resolve_model_cls(
                model_config.architectures, model_config
            )
            fn = getattr(model_cls, "_get_mrope_input_positions", None)
            if fn is None:
                raise ValueError(
                    f"[LWD] multimodal requests in prefill-only mode "
                    f"require a model class exposing "
                    f"_get_mrope_input_positions (Qwen3VL family, incl. "
                    f"Qwen3.5); got {model_cls.__name__} "
                    f"(request {request.request_id})"
                )
            self._positions_fn = fn
        return self._positions_fn

    def has_mrope_positions(self, request_id: str) -> bool:
        return request_id in self._positions

    def prompt_mrope_positions(self, request_id: str) -> bytes | None:
        positions = self._positions.get(request_id)
        if positions is None:
            return None
        return positions.contiguous().numpy().tobytes()

    def release(self, request_id: str) -> None:
        self._positions.pop(request_id, None)

    def held_request_ids(self) -> list[str]:
        return list(self._positions)


class LwdDeepseekV41ModelFamily(LwdModelFamily):
    """DeepSeek V4.1: image placeholder runs.

    The cloud's MoE vision routing and Engram image mask read image
    placeholder ids, so the edge sends their runs (public ids and positions,
    no text), once with the request notify.

    The first decode positions hash the prompt tail, whose ids the cloud also
    lacks, but the tail needs nothing on the wire: its positions are prompt
    positions, so their embeddings ride the UP payload like every other prompt
    row, and the cloud looks the ids back up from its own embedding table
    (`LwdEmbedLookup`).
    """

    def __init__(
        self, model_config: ModelConfig, engram_hasher: LwdEngramHasher | None
    ) -> None:
        self._image_token_id: int = model_config.hf_config.image_token_id
        # Held for the handshake fingerprint alone: the edge hashes prompt
        # positions with these parameters, so edge and cloud must agree on
        # them. The tail needs none of it.
        self._engram_hasher = engram_hasher

    def request_metadata(self, request: Request) -> LwdRequestMetadata:
        assert request.prompt_token_ids is not None
        prompt = torch.tensor(request.prompt_token_ids, dtype=torch.int64)
        spans = self._image_placeholder_spans(prompt)
        if self._engram_hasher is not None:
            # Paired with the cloud's "[engram] cloud tail" line: the ids the
            # cloud recovers from the tail embeddings have to be these, or its
            # look-back window lands on wrong table rows, which nothing but the
            # accuracy would show. Listed newest first, as the cloud lists them;
            # by index rather than a negative step, which a tensor cannot take.
            depth = self._engram_hasher.lookback_depth
            tail = prompt[max(prompt.numel() - depth, 0) :].tolist()
            lwd_logger.info(
                "[engram] edge tail req=%s prompt=%d ids=%s",
                request.request_id,
                prompt.numel(),
                tail[::-1],
            )
        return LwdRequestMetadata(image_placeholder_spans=spans)

    def _image_placeholder_spans(self, prompt: torch.Tensor) -> list[int]:
        """Flattened `(start, length, token_id)` of the maximal runs of equal
        ids in the image routing range.

        Scans the ids rather than the multimodal features: placeholder text
        written out in the prompt routes as image too.
        """
        is_image = (prompt >= self._image_token_id) & (
            prompt < self._image_token_id + _LWD_DEEPSEEK_IMAGE_ROUTING_IDS
        )
        if not bool(is_image.any()):
            return []
        values = torch.where(is_image, prompt, -1)
        run_begins = torch.ones_like(is_image)
        run_begins[1:] = values[1:] != values[:-1]
        run_starts = torch.nonzero(run_begins).flatten()
        run_ends = torch.cat((run_starts[1:], torch.tensor([prompt.numel()])))
        image_runs = is_image[run_starts]
        starts = run_starts[image_runs]
        spans = torch.stack(
            (starts, run_ends[image_runs] - starts, prompt[starts]), dim=1
        )
        return spans.flatten().tolist()

    def engram_fingerprint(self) -> str:
        if self._engram_hasher is None:
            return ""
        return self._engram_hasher.fingerprint()


def _lwd_request_has_mm(request: Request) -> bool:
    """Whether the request carries real multimodal data (prompt_embeds
    passthrough does not count -- it has no grid metadata, mrope is
    handled as text positions, consistent with the native definition)."""
    return any(f.modality != "prompt_embeds" for f in request.mm_features)


def lwd_select_model_family(vllm_config: VllmConfig) -> LwdModelFamily:
    """The request metadata family of the served model."""
    model_config = vllm_config.model_config
    family: LwdModelFamily
    if model_config.architecture == _LWD_DEEPSEEK_V41_ARCHITECTURE:
        hasher = LwdEngramHasher.from_model_config(model_config)
        family = LwdDeepseekV41ModelFamily(model_config, hasher)
    elif model_config.uses_mrope:
        family = LwdMropeModelFamily(vllm_config)
    else:
        family = LwdModelFamily()
    lwd_logger.info("[LWD] request metadata family: %s", type(family).__name__)
    return family


def lwd_cloud_engram_fingerprint(model_config: ModelConfig) -> str:
    """Digest of the cloud's Engram hash parameters, derived the way the edge
    family derives its own; empty without Engram."""
    if model_config.architecture != _LWD_DEEPSEEK_V41_ARCHITECTURE:
        return ""
    hasher = LwdEngramHasher.from_model_config(model_config)
    return hasher.fingerprint() if hasher is not None else ""
