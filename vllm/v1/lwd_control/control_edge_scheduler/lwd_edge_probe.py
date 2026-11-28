# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Asynchronous direct-DP placement and prefix probe batches."""

from __future__ import annotations

import hashlib
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from vllm.config.lwd.topology import LwdConnectionId
from vllm.logger import init_logger
from vllm.v1.lwd_control.control_edge_scheduler.lwd_cloud_health import (
    LwdCloudHealthTracker,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_client import (
    LwdEdgePrefixClient,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_probe_runtime import (
    LwdProbeRuntime,
)
from vllm.v1.lwd_control.lwd_comm.lwd_mm_identity import compute_processor_fingerprint
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import PROBE_HEADROOM_S, LwdProbeResult

logger = init_logger(__name__)


class LwdProbeSubmission(Enum):
    SUBMITTED = "submitted"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True)
class _MediaIdentity:
    modality: str
    digest: bytes
    offset: int
    length: int


@dataclass(frozen=True)
class LwdProbeCandidate:
    connection_id: LwdConnectionId
    result: LwdProbeResult
    chain: list[bytes] | None

    def normalized_hit_tokens(self, num_prompt_tokens: int) -> int:
        hit = min(max(self.result.hit_tokens, 0), max(num_prompt_tokens - 1, 0))
        alignment = self.result.alignment_tokens
        return hit - hit % alignment if alignment > 0 else hit


@dataclass(frozen=True)
class LwdProbeCompletion:
    request: Any
    probe_nonce: int = 0
    candidates: tuple[LwdProbeCandidate, ...] = ()
    failed_targets: tuple[LwdConnectionId, ...] = ()
    connection_id: LwdConnectionId | None = None
    result: LwdProbeResult | None = None
    chain: list[bytes] | None = None
    activated: bool = False
    activation_error: str | None = None


@dataclass
class _InflightProbe:
    request: Any
    nonce: int
    edge_id: int
    media_items: tuple[_MediaIdentity, ...]
    future: Any = None
    connection_id: LwdConnectionId | None = None
    candidate: LwdProbeCandidate | None = None
    drain_task: Any = None
    cancelled: bool = False


class LwdEdgeProbeManager:
    """Run fanout probes off-thread and apply selections on the engine thread."""

    def __init__(self, vllm_config: Any, wakeup: Callable[[], None]) -> None:
        self._vllm_config = vllm_config
        self._wakeup = wakeup
        self._runtime: LwdProbeRuntime | None = None
        self._done: queue.Queue[tuple[str, int, str, Any]] = queue.Queue()
        self._inflight: dict[str, _InflightProbe] = {}
        self._nonce = 0
        self._clients: dict[tuple[int, int, int], LwdEdgePrefixClient] = {}
        self._hash_sizes: dict[tuple[int, int], int] = {}
        self._processor_fingerprint: bytes | None = None
        self._tenant_key: bytes | None = None
        self._targets: dict[LwdConnectionId, str] = {}
        for connection in vllm_config.lwd_config.connections:
            connection_id = connection.connection_id
            cloud = connection.cloud
            url = cloud.coord_url or f"http://{cloud.addr}:{cloud.coord_port}"
            self._targets[connection_id] = url.rstrip("/")
        self._health = LwdCloudHealthTracker(
            tuple((target.cloud_id, target.dp_idx) for target in self._targets),
            edge_id=vllm_config.lwd_config.instance_id,
        )

    def set_silent_clouds(self, cloud_ids: frozenset[int]) -> None:
        """Verdict subscriber: mirror the cloud-grained liveness verdict
        into the health rotation. Called from the heartbeat monitor's single
        event exit point, on whichever thread detected the transition.
        Cloud ids expand to ``(cloud_id, dp_idx)`` here so the tracker stays
        keyed by cloud DP; ``candidates()`` then owns the exclusion and its
        fail-open, and no second liveness set has to be kept in sync. The
        monitor re-pushes the empty set on recovery."""
        self._health.set_silent(
            frozenset(
                target for target in self._health.targets if target[0] in cloud_ids
            )
        )

    def rotation(self) -> tuple[tuple[int, int], ...]:
        """Cloud-DP ids the health rotation may target (diagnostics)."""
        return self._health.targets

    def register_connection(self, connection_id: LwdConnectionId, url: str) -> None:
        """Teach the manager a runtime-adopted connection's coordination
        URL and open it for probe targeting (elastic scale-out: the
        rotation widens without restarting, so probes — and the routing
        preference they produce — spread over all adopted clouds). First
        registration wins; replays only confirm the health-rotation
        membership (idempotent)."""
        self._targets.setdefault(connection_id, url.rstrip("/"))
        self._health.add_target((connection_id.cloud_id, connection_id.dp_idx))

    @property
    def enabled(self) -> bool:
        coordination = self._vllm_config.lwd_config.coordination
        # Runtime-adopted connections count too: register_connection adds
        # to _targets, so elastic scale-out widens probing without a
        # restart.
        return bool(coordination and coordination.enabled and self._targets)

    @property
    def pending(self) -> bool:
        return bool(self._inflight) or not self._done.empty()

    def _edge_id(self, request: Any) -> int:
        if self._vllm_config.lwd_config.topology.deployment.scene == "edge_share":
            return request.client_index
        return self._vllm_config.lwd_config.instance_id

    def _fingerprint(self) -> bytes:
        if self._processor_fingerprint is None:
            self._processor_fingerprint = compute_processor_fingerprint(
                self._vllm_config.model_config
            )
        return self._processor_fingerprint

    def _key(self) -> bytes:
        if self._tenant_key is None:
            path = self._vllm_config.lwd_config.coordination.tenant_key_file
            if path:
                with open(path, "rb") as key_file:
                    self._tenant_key = key_file.read()
            else:
                self._tenant_key = b""
        return self._tenant_key

    def _client(self, target: LwdConnectionId, block_size: int) -> LwdEdgePrefixClient:
        key = (target.cloud_id, target.dp_idx, block_size)
        client = self._clients.get(key)
        if client is not None:
            return client
        coordination = self._vllm_config.lwd_config.coordination
        client = LwdEdgePrefixClient(
            self._targets[target],
            block_size,
            self._key(),
            consumer_id=coordination.consumer_id,
            connect_timeout=coordination.connect_timeout,
            processor_fingerprint=self._fingerprint(),
            probe_timeout=coordination.probe_timeout,
        )
        for stale in tuple(self._clients):
            if stale[:2] == key[:2] and stale != key:
                del self._clients[stale]
        self._clients[key] = client
        return client

    @staticmethod
    def _media_items(request: Any) -> tuple[_MediaIdentity, ...]:
        return tuple(
            _MediaIdentity(
                modality=feature.modality,
                digest=hashlib.sha256(feature.identifier.encode("utf-8")).digest(),
                offset=feature.mm_position.offset,
                length=feature.mm_position.length,
            )
            for feature in getattr(request, "mm_features", ())
            if feature.modality == "image"
        )

    def _visible_targets(self, edge_id: int) -> tuple[LwdConnectionId, ...]:
        return tuple(target for target in self._targets if target.edge_id == edge_id)

    def _uses_prefix_hit(self, request: Any) -> bool:
        coordination = self._vllm_config.lwd_config.coordination
        return bool(
            self._vllm_config.cache_config.enable_prefix_caching
            and not coordination.placement_only
            and not request.skip_reading_prefix_cache
        )

    def _selected_targets(
        self, request: Any, edge_id: int
    ) -> tuple[LwdConnectionId, ...]:
        visible = self._visible_targets(edge_id)
        allowed = {(target.cloud_id, target.dp_idx): target for target in visible}
        # Cooling-down DPs, this edge's visibility and the liveness verdict
        # are all resolved inside the health tracker now (in that order),
        # including the probe-side fail-open. No second liveness set is
        # kept here, and next() keeps its original rotation semantics.
        healthy = self._health.candidates(scope=frozenset(allowed), exclude_silent=True)
        targets = tuple(allowed[key] for key in healthy if key in allowed)
        if self._uses_prefix_hit(request):
            return targets
        if not targets:
            return ()
        healthy_ids = {(target.cloud_id, target.dp_idx) for target in targets}
        selected = self._health.next(allowed=healthy_ids)
        if selected is None:
            return ()
        return (allowed[selected],)

    def submit(self, request: Any) -> LwdProbeSubmission:
        if request.request_id in self._inflight:
            self.abort(request.request_id)
        if not self.enabled:
            return LwdProbeSubmission.UNAVAILABLE
        edge_id = self._edge_id(request)
        targets = self._selected_targets(request, edge_id)
        if not targets:
            return LwdProbeSubmission.UNAVAILABLE
        if self._runtime is None:
            self._runtime = LwdProbeRuntime()
        runtime = self._runtime
        coordination = self._vllm_config.lwd_config.coordination
        self._nonce += 1
        nonce = self._nonce
        media_items = self._media_items(request)
        inflight = _InflightProbe(request, nonce, edge_id, media_items)
        self._inflight[request.request_id] = inflight
        placement_only = not self._uses_prefix_hit(request)

        async def one(target: LwdConnectionId) -> Any:
            session = await runtime.ensure_session()
            block_size = self._hash_sizes.get(
                (target.cloud_id, target.dp_idx),
                self._vllm_config.cache_config.block_size,
            )
            client = self._client(target, block_size)
            manifest = client.build_manifest(
                request.request_id,
                list(request.prompt_token_ids),
                media_items=media_items,
            )
            try:
                result = await client.probe(
                    session,
                    request.request_id,
                    list(request.prompt_token_ids),
                    edge_id=edge_id,
                    cloud_id=target.cloud_id,
                    dp_idx=target.dp_idx,
                    probe_nonce=nonce,
                    manifest=manifest,
                    media_items=media_items,
                    model=self._model_name(),
                )
                chain = None if placement_only else list(manifest.full_block_hashes)
                return target, result, chain, block_size, None
            except Exception as exc:  # noqa: BLE001
                return target, None, None, block_size, exc

        batch_coroutines = [one(target) for target in targets]
        try:
            future = runtime.submit_group(batch_coroutines)
        except Exception:
            for coroutine in batch_coroutines:
                coroutine.close()
            self._inflight.pop(request.request_id, None)
            logger.exception("[LWD] failed to submit DP probe batch")
            return LwdProbeSubmission.UNAVAILABLE
        inflight.future = future

        def enqueue(kind: str, outcome: Any) -> None:
            self._done.put((request.request_id, nonce, kind, outcome))
            self._wakeup()

        def watchdog() -> None:
            if not future.done():
                future.cancel()

        timer = threading.Timer(coordination.probe_timeout + PROBE_HEADROOM_S, watchdog)
        timer.daemon = True

        def done(future: Any) -> None:
            timer.cancel()
            try:
                outcome = future.result()
            except BaseException as exc:
                logger.warning(
                    "[LWD] probe batch failed; using full prefill", exc_info=True
                )
                outcome = tuple((target, None, None, 0, exc) for target in targets)
            enqueue("probe", outcome)

        future.add_done_callback(done)
        timer.start()
        return LwdProbeSubmission.SUBMITTED

    def _model_name(self) -> str:
        model_config = self._vllm_config.model_config
        return model_config.served_model_name or model_config.model

    def activate(
        self,
        completion: LwdProbeCompletion,
        connection_id: LwdConnectionId,
        candidate: LwdProbeCandidate | None,
    ) -> bool:
        request = completion.request
        inflight = self._inflight.get(request.request_id)
        if inflight is None or self._runtime is None:
            return False
        runtime = self._runtime
        coordination = self._vllm_config.lwd_config.coordination
        block_size = (
            candidate.result.block_size
            if candidate is not None
            else self._hash_sizes.get(
                (connection_id.cloud_id, connection_id.dp_idx),
                self._vllm_config.cache_config.block_size,
            )
        )
        client = self._client(connection_id, block_size)
        inflight.connection_id = connection_id
        inflight.candidate = candidate

        async def job() -> str | None:
            try:
                session = await runtime.ensure_session()
                manifest = client.build_manifest(
                    request.request_id,
                    list(request.prompt_token_ids),
                    media_items=inflight.media_items,
                )
                drain_task = await client.activate(
                    session,
                    request.request_id,
                    list(request.prompt_token_ids),
                    edge_id=inflight.edge_id,
                    cloud_id=connection_id.cloud_id,
                    dp_idx=connection_id.dp_idx,
                    probe_nonce=inflight.nonce,
                    manifest=manifest,
                    media_items=inflight.media_items,
                    model=self._model_name(),
                )
                inflight.drain_task = drain_task
                if (
                    inflight.cancelled
                    or self._inflight.get(request.request_id) is not inflight
                ):
                    runtime.cancel_task(drain_task)
                return None
            except Exception as exc:  # noqa: BLE001
                return str(exc)

        future = runtime.submit(job(), timeout=coordination.probe_timeout)
        inflight.future = future

        def done(future: Any) -> None:
            try:
                error = future.result()
            except BaseException as exc:
                error = str(exc)
            self._done.put((request.request_id, inflight.nonce, "activate", error))
            self._wakeup()

        future.add_done_callback(done)
        return True

    def mark_admitted(self, request_id: str, nonce: int) -> None:
        inflight = self._inflight.get(request_id)
        if inflight is not None and inflight.nonce == nonce:
            self._inflight.pop(request_id, None)

    def abort(self, request_id: str) -> None:
        inflight = self._inflight.pop(request_id, None)
        if inflight is None:
            return
        inflight.cancelled = True
        if inflight.future is not None:
            inflight.future.cancel()
        if inflight.drain_task is not None and self._runtime is not None:
            self._runtime.cancel_task(inflight.drain_task)

    def drain(self) -> list[LwdProbeCompletion]:
        completions: list[LwdProbeCompletion] = []
        while True:
            try:
                request_id, nonce, kind, outcome = self._done.get_nowait()
            except queue.Empty:
                break
            inflight = self._inflight.get(request_id)
            if inflight is None or inflight.nonce != nonce:
                continue
            if kind == "activate":
                connection_id = inflight.connection_id
                candidate = inflight.candidate
                completions.append(
                    LwdProbeCompletion(
                        inflight.request,
                        probe_nonce=nonce,
                        connection_id=connection_id,
                        result=candidate.result if candidate is not None else None,
                        chain=candidate.chain if candidate is not None else None,
                        activated=True,
                        activation_error=outcome,
                    )
                )
                continue
            candidates: list[LwdProbeCandidate] = []
            failed: list[LwdConnectionId] = []
            for target, result, chain, build_size, error in outcome:
                dp_id = (target.cloud_id, target.dp_idx)
                if error is not None or result is None:
                    if self._health.record_failure(dp_id):
                        # The rotation skips this cloud DP until its cooldown
                        # expires (or a probe succeeds after it) — the
                        # "all requests stayed on the other cloud"
                        # signature when the target is a runtime-adopted
                        # cloud whose HTTP endpoint is not answering.
                        logger.error(
                            "[LWD] probe target cloud=%d dp=%d entered cooldown "
                            "after repeated failures (url=%s); it re-enters "
                            "the rotation once a probe succeeds",
                            target.cloud_id,
                            target.dp_idx,
                            self._targets.get(target),
                        )
                    failed.append(target)
                    continue
                self._health.record_success(dp_id)
                learned = int(result.block_size)
                self._hash_sizes[dp_id] = learned
                if chain is not None and learned != build_size:
                    rebuilt = self._client(target, learned).build_manifest(
                        request_id,
                        list(inflight.request.prompt_token_ids),
                        media_items=inflight.media_items,
                    )
                    chain = list(rebuilt.full_block_hashes)
                candidates.append(LwdProbeCandidate(target, result, chain))
            completions.append(
                LwdProbeCompletion(
                    inflight.request,
                    probe_nonce=nonce,
                    candidates=tuple(candidates),
                    failed_targets=tuple(failed),
                )
            )
        return completions

    def close(self) -> None:
        inflights = list(self._inflight.values())
        self._inflight.clear()
        runtime = self._runtime
        for inflight in inflights:
            inflight.cancelled = True
            if inflight.future is not None:
                inflight.future.cancel()
            if inflight.drain_task is not None and runtime is not None:
                runtime.cancel_task(inflight.drain_task)
        if runtime is not None:
            runtime.stop()
            self._runtime = None
