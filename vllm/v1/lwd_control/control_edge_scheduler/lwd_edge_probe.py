# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Asynchronous edge-side placement and cross-edge prefix probes."""

from __future__ import annotations

import hashlib
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable

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
from vllm.v1.lwd_control.lwd_comm.lwd_mm_identity import (
    compute_processor_fingerprint,
)
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import LwdProbeResult

logger = init_logger(__name__)

_WATCHDOG_HEADROOM_S = 5.0


@dataclass(frozen=True)
class _MediaIdentity:
    modality: str
    digest: bytes
    offset: int
    length: int


@dataclass(frozen=True)
class LwdProbeCompletion:
    request: Any
    result: LwdProbeResult | None
    chain: list[bytes] | None
    target_cloud: int | None
    build_block_size: int

    @property
    def cloud_id(self) -> int:
        if self.target_cloud is not None:
            return self.target_cloud
        if self.result is not None and self.result.cloud_id is not None:
            return self.result.cloud_id
        return 0

    @property
    def engine_index(self) -> int | None:
        return getattr(self.result, "engine_index", None)


@dataclass
class _InflightProbe:
    request: Any
    nonce: int
    media_items: tuple[_MediaIdentity, ...]


class LwdEdgeProbeManager:
    """Run HTTP probes off-thread and apply results on the engine thread."""

    def __init__(self, vllm_config: Any, wakeup: Callable[[], None]) -> None:
        self._vllm_config = vllm_config
        self._wakeup = wakeup
        self._runtime: LwdProbeRuntime | None = None
        self._done: queue.Queue[tuple[str, int, int, Any, int | None]] = (
            queue.Queue()
        )
        self._inflight: dict[str, _InflightProbe] = {}
        self._nonce = 0
        self._clients: dict[tuple[int, int], LwdEdgePrefixClient] = {}
        self._hash_sizes: dict[int, int] = {}
        self._processor_fingerprint: bytes | None = None
        self._tenant_key: bytes | None = None
        lwd_config = vllm_config.lwd_config
        coordination = lwd_config.coordination
        cloud_ids = tuple(coordination.control_url_map) if coordination else ()
        if not cloud_ids and coordination is not None and coordination.control_url:
            # Legacy single-URL deployment: the fixed target IS cloud 0
            # (the old submit() hardcoded cloud_id = 0 for it).
            cloud_ids = (0,)
        self._health = LwdCloudHealthTracker(cloud_ids)
        # Runtime-adopted clouds (elastic scale-out): the increment contract
        # keeps this edge's own coordination block byte-identical, so a cloud
        # added mid-run brings its URL here instead of the startup map.
        self._runtime_urls: dict[int, str] = {}

    def rotation(self) -> tuple[int, ...]:
        """Cloud ids the health rotation may target (diagnostics)."""
        return self._health.cloud_ids

    def register_cloud(self, cloud_id: int, url: str) -> None:
        """Teach the manager a runtime-adopted cloud's coordination URL and
        open it for probe targeting (elastic scale-out). First
        registration wins; replays only confirm the health-rotation
        membership (idempotent)."""
        self._runtime_urls.setdefault(cloud_id, url)
        self._health.add_cloud(cloud_id)

    def _url_for(self, cloud_id: int) -> str | None:
        """One cloud's coordination URL: runtime-adopted first, then the
        startup map; the legacy single URL belongs to cloud 0 by
        convention."""
        url = self._runtime_urls.get(cloud_id)
        if url is not None:
            return url
        coordination = self._vllm_config.lwd_config.coordination
        url = coordination.control_url_map.get(cloud_id)
        if url is not None:
            return url
        if cloud_id == 0:
            return coordination.control_url
        return None

    @property
    def enabled(self) -> bool:
        coordination = self._vllm_config.lwd_config.coordination
        return bool(
            coordination
            and coordination.enabled
            and (
                coordination.control_url
                or coordination.control_url_map
                or self._runtime_urls
            )
        )

    @property
    def pending(self) -> bool:
        return bool(self._inflight) or not self._done.empty()

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

    def _client(self, cloud_id: int, block_size: int, url: str) -> LwdEdgePrefixClient:
        key = (cloud_id, block_size)
        client = self._clients.get(key)
        if client is not None:
            return client
        coordination = self._vllm_config.lwd_config.coordination
        client = LwdEdgePrefixClient(
            url,
            block_size,
            self._key(),
            edge_id=self._vllm_config.lwd_config.instance_id,
            consumer_id=coordination.consumer_id,
            connect_timeout=coordination.connect_timeout,
            processor_fingerprint=self._fingerprint(),
            probe_timeout=coordination.probe_timeout,
        )
        for stale in tuple(self._clients):
            if stale[0] == cloud_id and stale != key:
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

    def submit(self, request: Any) -> bool:
        if not self.enabled or request.request_id in self._inflight:
            return False
        coordination = self._vllm_config.lwd_config.coordination
        # Health rotation = startup map (+ the legacy single URL as cloud 0)
        # plus every runtime-adopted cloud: elastic scale-out widens the
        # rotation without restarting, so probes — and the routing
        # preference they produce — spread over all adopted clouds.
        target_cloud = self._health.next()
        if target_cloud is None:
            logger.error(
                "[LWD] probe enabled but the rotation is empty "
                "(runtime=%s map_keys=%s)",
                sorted(self._runtime_urls),
                sorted(coordination.control_url_map),
            )
            return False
        url = self._url_for(target_cloud)
        if url is None:
            logger.error(
                "[LWD] no coordination URL for probe target cloud=%d "
                "(rotation=%s); request falls back to full prefill",
                target_cloud,
                self.rotation(),
            )
            return False
        cloud_id = target_cloud
        block_size = self._hash_sizes.get(
            cloud_id, self._vllm_config.cache_config.block_size
        )
        client = self._client(cloud_id, block_size, url)
        request_id = request.request_id
        prompt_ids = list(request.prompt_token_ids)
        media_items = self._media_items(request)
        model_config = self._vllm_config.model_config
        model_name = model_config.served_model_name or model_config.model
        placement_only = coordination.placement_only
        manifest_chain: list[bytes] | None = None
        if self._runtime is None:
            self._runtime = LwdProbeRuntime()
        runtime = self._runtime

        async def job() -> Any:
            nonlocal manifest_chain
            session = await runtime.ensure_session()
            manifest = client.build_manifest(
                request_id, prompt_ids, media_items=media_items
            )
            manifest_chain = None if placement_only else list(
                manifest.full_block_hashes
            )
            result = await client.negotiate(
                session,
                request_id,
                prompt_ids,
                manifest=manifest,
                media_items=media_items,
                model=model_name,
            )
            return result, manifest_chain

        try:
            future = runtime.submit(job())
        except Exception:
            logger.warning("[LWD] probe submission failed; using full prefill", exc_info=True)
            runtime.stop()
            self._runtime = None
            return False
        self._nonce += 1
        nonce = self._nonce
        self._inflight[request_id] = _InflightProbe(request, nonce, media_items)

        def enqueue(outcome: Any) -> None:
            self._done.put((request_id, nonce, block_size, outcome, target_cloud))
            self._wakeup()

        def watchdog() -> None:
            if not future.done():
                future.cancel()
                enqueue((None, manifest_chain))

        timer = threading.Timer(
            coordination.probe_timeout + _WATCHDOG_HEADROOM_S, watchdog
        )
        timer.daemon = True

        def done(future: Any) -> None:
            timer.cancel()
            try:
                outcome = future.result()
            except Exception:
                logger.error(
                    "[LWD] probe to cloud=%d failed (url=%s); using full "
                    "prefill",
                    target_cloud,
                    url,
                    exc_info=True,
                )
                outcome = (None, manifest_chain)
            enqueue(outcome)

        future.add_done_callback(done)
        timer.start()
        return True

    def abort(self, request_id: str) -> None:
        self._inflight.pop(request_id, None)

    def drain(self) -> list[LwdProbeCompletion]:
        """Return each current completion once; stale retries are discarded."""
        completions: list[LwdProbeCompletion] = []
        while True:
            try:
                request_id, nonce, build_size, outcome, target = self._done.get_nowait()
            except queue.Empty:
                break
            inflight = self._inflight.get(request_id)
            if inflight is None or inflight.nonce != nonce:
                continue
            del self._inflight[request_id]
            if target is not None:
                if outcome is None or outcome[0] is None:
                    if self._health.record_failure(target):
                        # The rotation skips this cloud until its cooldown
                        # expires (or a probe succeeds after it) — the
                        # "all requests stayed on the other cloud"
                        # signature when the target is a runtime-adopted
                        # cloud whose HTTP endpoint is not answering.
                        logger.error(
                            "[LWD] probe target cloud=%d entered cooldown "
                            "after repeated failures (url=%s); it re-enters "
                            "the rotation once a probe succeeds",
                            target,
                            self._url_for(target),
                        )
                else:
                    self._health.record_success(target)
            if outcome is None:
                completions.append(
                    LwdProbeCompletion(inflight.request, None, None, target, build_size)
                )
                continue
            result, chain = outcome
            if result is None:
                completions.append(
                    LwdProbeCompletion(inflight.request, None, chain, target, build_size)
                )
                continue
            learned = int(result.block_size)
            self._hash_sizes[self._cloud_for_result(target, result)] = learned
            if chain is not None and learned != build_size:
                url = (
                    self._url_for(target)
                    if target is not None
                    else self._vllm_config.lwd_config.coordination.control_url
                )
                rebuilt = self._client(
                    self._cloud_for_result(target, result), learned, url
                ).build_manifest(
                    request_id,
                    list(inflight.request.prompt_token_ids),
                    media_items=inflight.media_items,
                )
                chain = list(rebuilt.full_block_hashes)
            completions.append(
                LwdProbeCompletion(
                    inflight.request, result, chain, target, build_size
                )
            )
        return completions

    @staticmethod
    def _cloud_for_result(target: int | None, result: LwdProbeResult) -> int:
        if target is not None:
            return target
        return result.cloud_id if result.cloud_id is not None else 0

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.stop()
            self._runtime = None
        self._inflight.clear()
