# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Asynchronous activation through the explicitly configured gateway URL."""

from __future__ import annotations

import queue
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from typing import Any, cast

from vllm.config.lwd.topology import LwdConnectionId
from vllm.logger import init_logger
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_probe import (
    LwdEdgeProbeManager,
    LwdProbeCompletion,
    LwdProbeSubmission,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_gateway_client import (
    LwdGatewayClient,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_probe_runtime import (
    LwdProbeRuntime,
)
from vllm.v1.lwd_control.lwd_comm.lwd_messages import LWD_SAMPLING_DEFAULTS
from vllm.v1.lwd_control.lwd_comm.lwd_mm_identity import compute_processor_fingerprint
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import PROBE_HEADROOM_S, LwdProbeResult

logger = init_logger(__name__)


@dataclass
class _LwdInflightGateway:
    request: Any
    generation: int
    edge_id: int
    media_items: tuple[Any, ...]
    future: Any = None
    connection_id: LwdConnectionId | None = None
    drain_task: Any = None
    prefix_nonce: int = 0


class LwdGatewayActivationManager:
    """Run one gateway stream per request and bind the returned Cloud/DP."""

    def __init__(self, vllm_config: Any, wakeup: Callable[[], None]) -> None:
        self._vllm_config = vllm_config
        self._wakeup = wakeup
        self._runtime: LwdProbeRuntime | None = None
        self._done: queue.Queue[tuple[str, int, Any]] = queue.Queue()
        self._inflight: dict[str, _LwdInflightGateway] = {}
        self._generation = 0
        self._clients: dict[tuple[str, int], LwdGatewayClient] = {}
        self._hash_sizes: dict[tuple[int, int], int] = {}
        self._processor_fingerprint: bytes | None = None
        self._tenant_key: bytes | None = None
        # Cloud-grained liveness verdict cached from the heartbeat monitor.
        # One atomic frozenset swap; the monitor re-pushes the empty set on
        # recovery. See set_silent_clouds for why this does not filter the
        # gateway targets.
        self._silent_clouds: frozenset[int] = frozenset()
        self._targets = tuple(
            connection.connection_id
            for connection in vllm_config.lwd_config.connections
        )

    @property
    def enabled(self) -> bool:
        lwd_config = self._vllm_config.lwd_config
        coordination = lwd_config.coordination
        gateway = lwd_config.topology.gateway
        return bool(
            gateway.enabled
            and gateway.url is not None
            and coordination is not None
            and coordination.enabled
        )

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

    def _client(self, url: str, block_size: int) -> LwdGatewayClient:
        key = (url, block_size)
        client = self._clients.get(key)
        if client is not None:
            return client
        lwd_config = self._vllm_config.lwd_config
        coordination = lwd_config.coordination
        client = LwdGatewayClient(
            url,
            block_size,
            self._key(),
            consumer_id=coordination.consumer_id,
            connect_timeout=lwd_config.topology.gateway.connect_timeout,
            processor_fingerprint=self._fingerprint(),
            probe_timeout=coordination.probe_timeout,
        )
        self._clients[key] = client
        return client

    @property
    def silent_clouds(self) -> frozenset[int]:
        """The cached cloud-grained liveness verdict (diagnostic surface)."""
        return self._silent_clouds

    def set_silent_clouds(self, cloud_ids: frozenset[int]) -> None:
        """Verdict subscriber for the gateway coordination mode.

        The gateway service, not the edge, chooses the Cloud/DP, so the edge
        cannot drop a silent cloud from the probe targets the way the direct
        probe manager does: filtering ``_visible_targets`` here would turn a
        legal gateway response into "unconfigured Cloud" and fail the request
        instead of falling back. The admission gate already rejects requests
        while every route is silent, so the verdict is cached here as the
        single source of truth for diagnostics and for any future gateway
        protocol field carrying excluded clouds.
        """
        self._silent_clouds = cloud_ids

    def register_connection(self, connection_id: LwdConnectionId, url: str) -> None:
        """Elastic scale-out: add a runtime-adopted connection to the visible
        target set. The gateway service owns Cloud selection, so the
        per-cloud coordination URL is not used here — only the connection id
        matters for ``_visible_targets``. Idempotent, like the direct probe
        manager's registration.
        """
        del url
        if connection_id not in self._targets:
            self._targets = (*self._targets, connection_id)

    def rotation(self) -> tuple[tuple[int, int], ...]:
        """Visible Cloud-DP ids (diagnostics, mirrors the direct manager)."""
        return tuple((target.cloud_id, target.dp_idx) for target in self._targets)

    def _visible_targets(self, edge_id: int) -> tuple[LwdConnectionId, ...]:
        # Deliberately unfiltered: see set_silent_clouds. The gateway owns
        # Cloud selection; the admission gate owns the fully-silent verdict.
        return tuple(target for target in self._targets if target.edge_id == edge_id)

    def _uses_prefix_hit(self, request: Any) -> bool:
        coordination = self._vllm_config.lwd_config.coordination
        return bool(
            self._vllm_config.cache_config.enable_prefix_caching
            and not coordination.placement_only
            and not request.skip_reading_prefix_cache
        )

    def _resolve_connection(
        self, result: LwdProbeResult, edge_id: int
    ) -> tuple[LwdConnectionId, LwdProbeResult]:
        visible = self._visible_targets(edge_id)
        cloud_ids = {target.cloud_id for target in visible}
        cloud_id = result.cloud_id
        if cloud_id is None:
            if len(cloud_ids) != 1:
                raise RuntimeError(
                    "gateway must return cloud_id when multiple Clouds are visible"
                )
            cloud_id = next(iter(cloud_ids))
        dp_indices = {
            target.dp_idx for target in visible if target.cloud_id == cloud_id
        }
        if not dp_indices:
            raise RuntimeError(f"gateway returned an unconfigured Cloud: {cloud_id}")
        engine_index = result.engine_index
        if engine_index is None:
            if len(dp_indices) != 1:
                raise RuntimeError(
                    "gateway must return engine_index when multiple DPs are visible"
                )
            engine_index = next(iter(dp_indices))
        connection = next(
            (
                candidate
                for candidate in visible
                if (candidate.cloud_id, candidate.dp_idx) == (cloud_id, engine_index)
            ),
            None,
        )
        if connection is None:
            raise RuntimeError(
                "gateway returned an unconfigured Cloud/DP: "
                f"({cloud_id}, {engine_index})"
            )
        expected_instance_id = f"cloud-{cloud_id}"
        if result.instance_id != expected_instance_id:
            raise RuntimeError(
                "gateway returned a different instance: "
                f"expected={expected_instance_id} actual={result.instance_id}"
            )
        normalized = dataclass_replace(
            result, cloud_id=cloud_id, engine_index=engine_index
        )
        return connection, normalized

    def _block_size(self) -> int:
        learned = set(self._hash_sizes.values())
        if len(learned) == 1:
            return learned.pop()
        return self._vllm_config.cache_config.block_size

    def submit(self, request: Any) -> LwdProbeSubmission:
        if request.request_id in self._inflight:
            self.abort(request.request_id)
        if not self.enabled:
            return LwdProbeSubmission.UNAVAILABLE
        edge_id = self._edge_id(request)
        url = self._vllm_config.lwd_config.topology.gateway.url
        assert url is not None
        if self._runtime is None:
            self._runtime = LwdProbeRuntime()
        runtime = self._runtime
        coordination = self._vllm_config.lwd_config.coordination
        self._generation += 1
        generation = self._generation
        media_items = LwdEdgeProbeManager._media_items(request)
        inflight = _LwdInflightGateway(request, generation, edge_id, media_items)
        self._inflight[request.request_id] = inflight
        placement_only = not self._uses_prefix_hit(request)
        max_tokens = getattr(request.sampling_params, "max_tokens", None)
        if max_tokens is None:
            max_tokens = cast(int, LWD_SAMPLING_DEFAULTS["max_tokens"])

        async def job() -> tuple[Any, ...]:
            session = await runtime.ensure_session()
            block_size = self._block_size()
            client = self._client(url, block_size)
            manifest = client.build_manifest(
                request.request_id,
                list(request.prompt_token_ids),
                media_items=media_items,
            )
            activation = await client.activate_gateway(
                session,
                request.request_id,
                list(request.prompt_token_ids),
                edge_id=edge_id,
                manifest=manifest,
                media_items=media_items,
                model=self._model_name(),
                max_tokens=max_tokens,
            )
            if self._inflight.get(request.request_id) is inflight:
                inflight.drain_task = activation.drain_task
                inflight.prefix_nonce = activation.prefix_nonce
            connection, result = self._resolve_connection(activation.result, edge_id)
            if self._inflight.get(request.request_id) is inflight:
                inflight.connection_id = connection
            chain = None if placement_only else list(manifest.full_block_hashes)
            return connection, result, chain, block_size, activation.prefix_nonce, url

        coroutine = job()
        try:
            future = runtime.submit(
                coroutine, timeout=coordination.probe_timeout + PROBE_HEADROOM_S
            )
        except Exception:
            coroutine.close()
            logger.exception("[LWD] failed to submit gateway activation")
            self._inflight.pop(request.request_id, None)
            return LwdProbeSubmission.FAILED
        inflight.future = future

        def done(future: Any) -> None:
            try:
                outcome = future.result()
            except BaseException as exc:
                outcome = str(exc)
            self._done.put((request.request_id, generation, outcome))
            self._wakeup()

        future.add_done_callback(done)
        return LwdProbeSubmission.SUBMITTED

    def _model_name(self) -> str:
        model_config = self._vllm_config.model_config
        served = model_config.served_model_name
        if isinstance(served, (list, tuple)):
            return str(served[0] if served else model_config.model)
        return str(served or model_config.model)

    def mark_admitted(self, request_id: str, nonce: int) -> None:
        inflight = self._inflight.get(request_id)
        if inflight is not None and inflight.prefix_nonce == nonce:
            self._inflight.pop(request_id, None)

    def abort(self, request_id: str) -> None:
        inflight = self._inflight.pop(request_id, None)
        if inflight is None:
            return
        if inflight.drain_task is None and inflight.future is not None:
            inflight.future.cancel()

    def drain(self) -> list[LwdProbeCompletion]:
        completions: list[LwdProbeCompletion] = []
        while True:
            try:
                request_id, generation, outcome = self._done.get_nowait()
            except queue.Empty:
                break
            inflight = self._inflight.get(request_id)
            if inflight is None or inflight.generation != generation:
                continue
            if isinstance(outcome, str):
                completions.append(
                    LwdProbeCompletion(
                        inflight.request,
                        probe_nonce=inflight.prefix_nonce,
                        connection_id=inflight.connection_id,
                        activated=True,
                        activation_error=outcome,
                    )
                )
                continue
            connection, result, chain, build_size, prefix_nonce, url = outcome
            learned = int(result.block_size)
            dp_id = (connection.cloud_id, connection.dp_idx)
            self._hash_sizes[dp_id] = learned
            if chain is not None and learned != build_size:
                rebuilt = self._client(url, learned).build_manifest(
                    request_id,
                    list(inflight.request.prompt_token_ids),
                    media_items=inflight.media_items,
                )
                chain = list(rebuilt.full_block_hashes)
            completions.append(
                LwdProbeCompletion(
                    inflight.request,
                    probe_nonce=prefix_nonce,
                    connection_id=connection,
                    result=result,
                    chain=chain,
                    activated=True,
                )
            )
        return completions

    def close(self) -> None:
        inflights = list(self._inflight.values())
        self._inflight.clear()
        runtime = self._runtime
        for inflight in inflights:
            if inflight.future is not None:
                inflight.future.cancel()
            if inflight.drain_task is not None and runtime is not None:
                runtime.cancel_task(inflight.drain_task)
        if runtime is not None:
            runtime.stop()
            self._runtime = None
