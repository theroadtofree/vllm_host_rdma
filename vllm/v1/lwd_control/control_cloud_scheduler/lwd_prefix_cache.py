# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Short-lived prefix reservations for LWD cloud scheduling."""

from __future__ import annotations

from math import lcm
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


class _LwdPrefixReservation(NamedTuple):
    groups: tuple[Sequence[KVCacheBlock], ...]
    hit_tokens: int
    pinned: list[KVCacheBlock]


class LwdPrefixReservations:
    """Pin negotiated prefix blocks until native allocation takes ownership."""

    def __init__(self, manager: KVCacheManager) -> None:
        self.manager = manager
        self._reservations: dict[str, _LwdPrefixReservation] = {}
        manager.lwd_prefix_reservations = self
        manager.lwd_prefix_caching = True

    def reserve(self, request: Request) -> int:
        """Reserve the negotiated hit and freeze its snapshot."""
        if request.request_id in self._reservations:
            raise ValueError(f"duplicate LWD prefix reservation: {request.request_id}")
        if request.skip_reading_prefix_cache:
            groups: tuple[Sequence[KVCacheBlock], ...] = ()
            hit = 0
        else:
            groups, hit, _ = self.manager.coordinator.find_longest_cache_hit(
                request.block_hashes,
                max_cache_hit_length=max(0, request.num_prompt_tokens - 1),
            )
        return self._touch_and_snapshot(request, groups, hit, set_floor=True)

    def reserve_hit(
        self,
        request: Request,
        groups: tuple[Sequence[KVCacheBlock], ...],
        hit: int,
    ) -> int:
        """Reserve a readmission hit without changing the staging floor."""
        if request.request_id in self._reservations:
            raise ValueError(f"duplicate LWD prefix reservation: {request.request_id}")
        return self._touch_and_snapshot(request, groups, hit, set_floor=False)

    def touch(self, groups: tuple[Sequence[KVCacheBlock], ...]) -> list[KVCacheBlock]:
        """Pin real blocks while preserving null slots only in the snapshot."""
        pinned = [
            block
            for group in groups
            for block in group
            if not getattr(block, "is_null", False)
        ]
        if pinned:
            self.manager.block_pool.touch(pinned)
        return pinned

    def snapshot(
        self, request: Request
    ) -> tuple[tuple[Sequence[KVCacheBlock], ...], int]:
        """Return the frozen hit, refusing to renegotiate a trimmed request."""
        entry = self._reservations.get(request.request_id)
        if entry is not None:
            return entry.groups, entry.hit_tokens
        if getattr(request, "lwd_resume_cap_tokens", None) is not None:
            raise RuntimeError(
                f"LWD reserved prefix for {request.request_id} is unavailable; "
                "cannot consume trimmed embeddings"
            )
        groups, hit, _ = self.manager.coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_tokens - 1
        )
        return groups, hit

    def truncate(
        self,
        groups: tuple[Sequence[KVCacheBlock], ...],
        hit: int,
    ) -> tuple[Sequence[KVCacheBlock], ...]:
        """Truncate each cache group to the blocks covered by hit."""
        managers = self.manager.coordinator.single_type_managers
        return tuple(
            # Keep the block covering the hit endpoint: with fine-grained
            # partial hits the endpoint can land inside a block, and that
            # block holds the hit tokens.
            list(group[: (hit + manager.block_size - 1) // manager.block_size])
            for group, manager in zip(groups, managers, strict=True)
        )

    def release(self, request_id: str) -> None:
        """Release a reservation and its pins; repeated calls are no-ops."""
        entry = self._reservations.pop(request_id, None)
        if entry is not None and entry.pinned:
            self.manager.block_pool.free_blocks(entry.pinned)

    def has(self, request_id: str) -> bool:
        return request_id in self._reservations

    def hit_of(self, request_id: str) -> int:
        entry = self._reservations.get(request_id)
        return entry.hit_tokens if entry is not None else 0

    def _touch_and_snapshot(
        self,
        request: Request,
        groups: tuple[Sequence[KVCacheBlock], ...],
        hit: int,
        *,
        set_floor: bool,
    ) -> int:
        pinned = self.touch(groups)
        self._reservations[request.request_id] = _LwdPrefixReservation(
            groups=tuple(tuple(group) for group in groups),
            hit_tokens=hit,
            pinned=pinned,
        )
        if set_floor:
            request.lwd_resume_cap_tokens = hit
        return hit


def prefix_start(
    request: Request, reservations: LwdPrefixReservations | None = None
) -> int:
    """Return the prefill window start before and after native allocation."""
    if request.num_computed_tokens > 0:
        return request.num_computed_tokens
    if reservations is not None and reservations.has(request.request_id):
        return reservations.hit_of(request.request_id)
    return getattr(request, "lwd_resume_cap_tokens", 0)


class LwdPrefixCacheState(NamedTuple):
    enabled: bool
    hash_block_size: int
    alignment_tokens: int = 0
    alignment_required: bool = False
    use_eagle: bool = False
    reservations: LwdPrefixReservations | None = None


def init_lwd_prefix_cache(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    manager: KVCacheManager,
    *,
    use_eagle: bool = False,
) -> LwdPrefixCacheState:
    """Validate the cloud KV layout and create its reservation table."""
    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MambaSpec,
        SlidingWindowSpec,
        UniformTypeKVCacheSpecs,
    )

    enabled = bool(vllm_config.cache_config.enable_prefix_caching)
    hash_block_size = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)[1]
    if not enabled:
        return LwdPrefixCacheState(enabled=False, hash_block_size=hash_block_size)
    if not manager.enable_caching:
        raise ValueError("LWD prefix caching requires --enable-prefix-caching on cloud")
    if vllm_config.parallel_config.prefill_context_parallel_size > 1:
        raise ValueError(
            "LWD prefix caching does not support PCP "
            "(prefill_context_parallel_size > 1)"
        )

    specs = []
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        specs.extend(
            spec.kv_cache_specs.values()
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else [spec]
        )
    if not specs or any(
        not isinstance(spec, (FullAttentionSpec, SlidingWindowSpec, MambaSpec))
        or (
            isinstance(spec, MambaSpec)
            and spec.mamba_cache_mode not in ("align", "all")
        )
        for spec in specs
    ):
        raise ValueError(
            "LWD prefix caching currently supports Full Attention/MLA, "
            "sliding-window and Mamba align/all groups only"
        )

    coordinator = manager.coordinator
    # The coordinator's hit alignment is authoritative: fine-grained partial
    # hits align to the compressed-state unit (e.g. 128 tokens per C128
    # state), not the scheduler LCM block, and the edge rounds the probe hit
    # down to this value. Fall back to the LCM formula for coordinators that
    # do not expose it.
    hit_alignment = getattr(coordinator, "cache_hit_alignment_tokens", None)
    alignment_tokens = (
        hit_alignment
        if isinstance(hit_alignment, int) and hit_alignment > 0
        else lcm(
            coordinator.scheduler_block_size,
            getattr(coordinator, "lcm_block_size", coordinator.scheduler_block_size),
        )
    )
    alignment_required = any(
        isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align"
        for spec in specs
    )
    return LwdPrefixCacheState(
        enabled=True,
        hash_block_size=hash_block_size,
        alignment_tokens=alignment_tokens,
        alignment_required=alignment_required,
        use_eagle=use_eagle,
        reservations=LwdPrefixReservations(manager),
    )


class LwdProbeQueryAdapter:
    """Read-only HTTP prefix lookup; RequestNotify owns reservations."""

    def __init__(
        self,
        manager: KVCacheManager,
        hash_block_size: int,
        alignment_tokens: int,
        instance_id: str,
        cloud_id: int,
        *,
        prefix_caching: bool = True,
    ) -> None:
        self._manager = manager
        self._hash_block_size = hash_block_size
        self._alignment_tokens = alignment_tokens
        self._instance_id = instance_id
        self._cloud_id = cloud_id
        self._prefix_caching = prefix_caching

    def _query(self, manifest) -> Any:
        from vllm.v1.lwd_control.lwd_comm.lwd_prefix import LwdProbeResult

        if not self._prefix_caching:
            # Placement-only probe: the cloud runs no prefix cache, so the
            # coordinator can only answer "zero hit". The edge has no way to
            # know this cloud's hash granularity (prefix hashing is disabled
            # on both sides), so the block-size contract below must not gate
            # the reply: enforcing it turns every probe into a 503 and drops
            # the placement/health signal. Echo the manifest granularity so
            # the edge neither learns a size it cannot hash with nor rebuilds
            # its chain for a hit that is always zero.
            return LwdProbeResult(
                request_id=manifest.request_id,
                instance_id=self._instance_id,
                block_size=manifest.block_size,
                hit_blocks=0,
                hit_tokens=0,
                alignment_tokens=self._alignment_tokens,
                cloud_id=self._cloud_id,
            )
        if manifest.block_size != self._hash_block_size:
            raise ValueError(
                "LWD manifest block size must match the cloud hash block size "
                f"({manifest.block_size} != {self._hash_block_size})"
            )
        _, hit, _ = self._manager.coordinator.find_longest_cache_hit(
            list(manifest.full_block_hashes),
            max_cache_hit_length=max(0, manifest.prompt_tokens - 1),
        )
        hit = hit // self._hash_block_size * self._hash_block_size
        return LwdProbeResult(
            request_id=manifest.request_id,
            instance_id=self._instance_id,
            block_size=self._hash_block_size,
            hit_blocks=hit // self._hash_block_size,
            hit_tokens=hit,
            alignment_tokens=self._alignment_tokens,
            cloud_id=self._cloud_id,
        )

    def peek(self, manifest) -> Any:
        return self._query(manifest)

    def probe(self, manifest) -> Any:
        return self._query(manifest)
