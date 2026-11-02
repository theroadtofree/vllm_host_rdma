# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Edge-side Engram n-gram hashing for DeepSeek V4.1.

The cloud holds the Engram tables but no prompt token ids. The edge therefore
hashes every prompt position into the final bucket indices the cloud looks
up, carrying no token id.

The first decode steps look back at the last prompt tokens, which the cloud
recovers from the embeddings of those positions instead (the cloud's
`LwdEmbedLookup`); nothing about the tail is hashed or sent here.
"""

import hashlib
from typing import TYPE_CHECKING

import torch
from transformers import AutoTokenizer

from vllm.lwd_logger import lwd_logger
from vllm.transformers_utils.config import get_config, get_hf_text_config

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.models.deepseek_v41.common.engram import EngramLayout


class LwdEngramHasher:
    """Engram hash of prompt positions, bit-identical to the cloud's kernel.

    Mirrors `_hash_ids_kernel` of vllm-ascend `AscendNgramHashState` run over
    the full prompt: compressed ids, look-back blocked at the sequence start
    and at image tokens (blocking persists for longer n-grams), blocked slots
    hashed as the pad token, a rolling XOR of `value * multiplier` per shift,
    and one prime-sized bucket range per (n-gram size, head).
    """

    def __init__(
        self,
        layout: "EngramLayout",
        token_map: list[int],
        multipliers: torch.Tensor,
        image_token_ids: tuple[int, int],
    ) -> None:
        self.layer_ids: tuple[int, ...] = layout.layer_ids
        self.num_layers: int = len(layout.layer_ids)
        self.num_hash_cols: int = layout.n_hash_cols
        self.max_ngram_size: int = layout.max_ngram_size
        self.lookback_depth: int = layout.max_ngram_size - 1
        self._n_heads = layout.n_heads
        self._compressed_vocab_size = layout.compressed_vocab_size
        self._token_map = torch.tensor(token_map, dtype=torch.int64)
        self._pad_id = token_map[layout.pad_token_id]
        self._multipliers = multipliers.to(torch.int64)  # [L, M]
        self._primes = torch.tensor(layout.primes, dtype=torch.int64)  # [L, M-1, H]
        self._offsets = layout.offsets.to(torch.int64)  # [L, C]
        self._image_token_ids = torch.tensor(image_token_ids, dtype=torch.int64)
        self._fingerprint = self._compute_fingerprint()

    @classmethod
    def from_model_config(cls, model_config: "ModelConfig") -> "LwdEngramHasher | None":
        """Build the hasher from the original checkpoint config.

        The edge projection clears the Engram fields through `hf_overrides`,
        so the config is re-read from disk without overrides. The tokenizer
        is loaded the way upstream `NgramHashState` loads it.

        Returns:
            None when the checkpoint has no Engram layers.

        Raises:
            ValueError: The tokenizer's compressed vocab size differs from the
                config, or the config has no `image_token_id`.
        """
        hf_config = get_config(
            model_config.hf_config_path or model_config.model,
            model_config.trust_remote_code,
            model_config.revision,
            model_config.code_revision,
            model_config.config_format,
            token=model_config.hf_token,
        )
        text_config = get_hf_text_config(hf_config)
        if not getattr(text_config, "engram_layer_ids", None):
            return None
        # Imported here: the module pulls in the whole DeepSeek V4.1 model
        # package, which models without Engram never need.
        from vllm.models.deepseek_v41.common.engram import (
            EngramLayout,
            build_compressed_token_map,
            compute_hash_multipliers,
        )

        layout = EngramLayout(text_config)
        image_token_id = getattr(hf_config, "image_token_id", None)
        if image_token_id is None:
            raise ValueError(
                "[LWD] Engram checkpoint config has no image_token_id; the "
                "Engram hash cannot tell image positions apart"
            )
        image_pad_token_id = getattr(
            hf_config, "image_pad_token_id", image_token_id + 1
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer,
            trust_remote_code=model_config.trust_remote_code,
            revision=model_config.revision,
        )
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != layout.compressed_vocab_size:
            raise ValueError(
                f"[LWD] Engram compressed vocab size {vocab_size} built from the "
                f"tokenizer != config {layout.compressed_vocab_size}; every hash "
                "multiplier derives from it"
            )
        multipliers = compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, vocab_size
        )
        lwd_logger.info(
            "[engram] edge hasher built for layers %s: %d -> %d compressed ids",
            layout.layer_ids,
            len(token_map),
            vocab_size,
        )
        return cls(layout, token_map, multipliers, (image_token_id, image_pad_token_id))

    def prompt_hashes(
        self, prompt_token_ids: torch.Tensor, start: int, end: int
    ) -> torch.Tensor:
        """Bucket indices of prompt positions `[start, end)`.

        Args:
            prompt_token_ids: CPU 1-D ids of the full prompt; the look-back of
                `start` reads the positions before it.
            start: First position to hash.
            end: One past the last position to hash.

        Returns:
            CPU int32 `[end - start, num_layers, num_hash_cols]`, columns
            ordered `(n-gram size - 2) * n_heads + head`.

        Raises:
            ValueError: The prompt is malformed, the range is outside it, or
                a token id is outside the vocab.
        """
        self._check_prompt(prompt_token_ids)
        if not 0 <= start <= end <= prompt_token_ids.numel():
            raise ValueError(
                f"[LWD] Engram hash range [{start}, {end}) outside the prompt of "
                f"{prompt_token_ids.numel()} tokens"
            )
        window_start = max(start - self.lookback_depth, 0)
        compressed, dead = self._compress(prompt_token_ids[window_start:end])
        positions = torch.arange(start, end, dtype=torch.int64)
        blocked = torch.zeros(end - start, dtype=torch.bool)
        rolling = torch.zeros(end - start, self.num_layers, dtype=torch.int64)
        hashes = []
        for shift in range(self.max_ngram_size):
            lookback = positions - shift
            index = (lookback - window_start).clamp_min(0)
            blocked |= (lookback < 0) | dead[index]
            value = torch.where(blocked, self._pad_id, compressed[index])
            rolling ^= value[:, None] * self._multipliers[:, shift]
            if shift == 0:
                continue
            cols = slice((shift - 1) * self._n_heads, shift * self._n_heads)
            hashes.append(
                rolling[..., None] % self._primes[:, shift - 1] + self._offsets[:, cols]
            )
        return torch.cat(hashes, dim=-1).to(torch.int32)

    def fingerprint(self) -> str:
        """Hex digest of every hash parameter, compared at the handshake."""
        return self._fingerprint

    def _compute_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for values in (
            self._token_map,
            self._multipliers,
            self._primes,
            self._offsets,
            self._image_token_ids,
            torch.tensor(self.layer_ids, dtype=torch.int64),
        ):
            digest.update(values.contiguous().numpy().tobytes())
        digest.update(
            repr(
                (
                    self._pad_id,
                    self.max_ngram_size,
                    self._n_heads,
                    self._compressed_vocab_size,
                )
            ).encode()
        )
        return digest.hexdigest()

    def _compress(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compressed ids (int64) and image flags (bool) of `token_ids`."""
        token_ids = token_ids.to(torch.int64)
        if token_ids.numel() and (
            token_ids.min() < 0 or token_ids.max() >= self._token_map.numel()
        ):
            raise ValueError(
                f"[LWD] Engram hash got token ids outside "
                f"[0, {self._token_map.numel()})"
            )
        return self._token_map[token_ids], torch.isin(token_ids, self._image_token_ids)

    @staticmethod
    def _check_prompt(prompt_token_ids: torch.Tensor) -> None:
        if prompt_token_ids.ndim != 1 or prompt_token_ids.numel() == 0:
            raise ValueError(
                f"[LWD] Engram hash needs a non-empty 1-D prompt, got shape "
                f"{tuple(prompt_token_ids.shape)}"
            )
        if prompt_token_ids.device.type != "cpu":
            raise ValueError(
                f"[LWD] Engram hash runs on CPU, got prompt ids on "
                f"{prompt_token_ids.device}"
            )
