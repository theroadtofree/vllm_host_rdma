# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.config.lwd.topology import LwdConnectionId
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
    LwdEdgeEngineCore,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_probe import (
    LwdEdgeProbeManager,
    LwdProbeSubmission,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_gateway_activation import (
    LwdGatewayActivationManager,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_gateway_client import (
    LwdGatewayActivation,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_probe_runtime import (
    LwdProbeRuntime,
)
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import LwdPrefixManifest, LwdProbeResult


def _connection(edge_id: int, cloud_id: int, dp_idx: int) -> SimpleNamespace:
    return SimpleNamespace(
        connection_id=LwdConnectionId(edge_id, cloud_id, dp_idx),
        cloud=SimpleNamespace(
            coord_url=f"http://cloud{cloud_id}:{8100 + dp_idx}",
            coord_port=8100 + dp_idx,
            addr=f"cloud{cloud_id}",
        ),
    )


def _config(prefix_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        lwd_config=SimpleNamespace(
            connections=[_connection(0, 0, 0), _connection(0, 0, 1)],
            coordination=SimpleNamespace(
                enabled=True,
                placement_only=False,
                consumer_id=None,
                connect_timeout=1.0,
                probe_timeout=1.0,
                tenant_key_file=None,
            ),
            topology=SimpleNamespace(
                deployment=SimpleNamespace(scene="single_instance")
            ),
            instance_id=0,
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=prefix_enabled,
            block_size=4,
        ),
        model_config=SimpleNamespace(served_model_name="test", model="test"),
    )


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        request_id="req",
        client_index=0,
        skip_reading_prefix_cache=False,
        prompt_token_ids=list(range(16)),
        num_prompt_tokens=16,
        sampling_params=SimpleNamespace(max_tokens=32),
        mm_features=[],
    )


class _Client:
    def __init__(
        self,
        target: LwdConnectionId,
        calls: list[tuple[int, int]],
        activity: dict[str, int],
    ) -> None:
        self.target = target
        self.calls = calls
        self.activity = activity

    def build_manifest(
        self, request_id: str, prompt_token_ids: list[int], media_items: tuple
    ) -> LwdPrefixManifest:
        return LwdPrefixManifest(
            request_id=request_id,
            prompt_tokens=len(prompt_token_ids),
            block_size=4,
            full_block_hashes=(bytes(32), bytes(32), bytes(32), bytes(32)),
            tail_hash=None,
        )

    async def probe(
        self, _session: object, request_id: str, _prompt: list[int], **kwargs
    ):
        self.calls.append((self.target.cloud_id, self.target.dp_idx))
        self.activity["active"] += 1
        self.activity["max_active"] = max(
            self.activity["max_active"], self.activity["active"]
        )
        await asyncio.sleep(0.02)
        self.activity["active"] -= 1
        hit_blocks = self.target.dp_idx + 1
        return LwdProbeResult(
            request_id=request_id,
            instance_id="cloud-0",
            block_size=4,
            hit_blocks=hit_blocks,
            hit_tokens=hit_blocks * 4,
            cloud_id=self.target.cloud_id,
            engine_index=self.target.dp_idx,
        )

    async def activate(
        self, _session: object, _request_id: str, _prompt: list[int], **kwargs
    ):
        assert kwargs["edge_id"] == 0
        assert kwargs["dp_idx"] == self.target.dp_idx


def _drain(manager: LwdEdgeProbeManager, activated: bool = False):
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        completions = manager.drain()
        if completions:
            assert completions[0].activated is activated
            return completions[0]
        time.sleep(0.005)
    raise AssertionError("probe completion did not arrive")


@pytest.mark.parametrize(
    ("prefix_enabled", "expected_probes", "expected_concurrency"),
    [(True, 2, 2), (False, 1, 1)],
)
def test_submit_fans_out_only_for_prefix_and_activates_winner(
    prefix_enabled: bool, expected_probes: int, expected_concurrency: int
) -> None:
    manager = LwdEdgeProbeManager(_config(prefix_enabled), lambda: None)
    calls: list[tuple[int, int]] = []
    activity = {"active": 0, "max_active": 0}
    clients = {target: _Client(target, calls, activity) for target in manager._targets}
    manager._client = lambda target, _block_size: clients[target]
    try:
        assert manager.submit(_request()) is LwdProbeSubmission.SUBMITTED
        completion = _drain(manager)
        assert len(calls) == expected_probes
        assert len(completion.candidates) == expected_probes
        assert activity["max_active"] == expected_concurrency

        winner = max(
            completion.candidates,
            key=lambda candidate: candidate.normalized_hit_tokens(16),
        )
        assert manager.activate(completion, winner.connection_id, winner)
        activated = _drain(manager, activated=True)
        assert activated.connection_id == winner.connection_id
        assert activated.result == winner.result
        assert activated.probe_nonce == completion.probe_nonce
        manager.mark_admitted("req", activated.probe_nonce)
        assert not manager._inflight
    finally:
        manager.close()


def _multi_cloud_config(edge_id: int) -> SimpleNamespace:
    config = _config(False)
    config.lwd_config.connections = [
        _connection(edge_id, cloud_id, dp_idx)
        for cloud_id in (0, 1)
        for dp_idx in (0, 1)
    ]
    config.lwd_config.instance_id = edge_id
    return config


def test_placement_staggers_cloud_and_dp_by_edge_id() -> None:
    for edge_id in range(4):
        manager = LwdEdgeProbeManager(_multi_cloud_config(edge_id), lambda: None)
        try:
            selected = manager._selected_targets(_request(), edge_id)
            assert selected == (LwdConnectionId(edge_id, edge_id % 2, edge_id // 2),)
        finally:
            manager.close()


def test_activate_style_submit_timeout_covers_semaphore_queue() -> None:
    runtime = LwdProbeRuntime(max_inflight=1)
    started = threading.Event()
    release = threading.Event()

    async def hold_permit() -> None:
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.001)

    first = runtime.submit(hold_permit())
    assert started.wait(timeout=1)
    try:
        queued = runtime.submit(asyncio.sleep(0), timeout=0.01)
        with pytest.raises(TimeoutError):
            queued.result(timeout=1)
    finally:
        release.set()
        first.result(timeout=1)
        runtime.stop()


def test_bounded_cancellation_closes_unstarted_coroutine() -> None:
    async def run() -> None:
        runtime = object.__new__(LwdProbeRuntime)
        runtime._inflight = asyncio.Semaphore(0)
        coroutine = Mock()
        task = asyncio.create_task(runtime._bounded(coroutine))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        coroutine.close.assert_called_once_with()

    asyncio.run(run())


def test_submit_failure_cleans_inflight_and_falls_back() -> None:
    class _StoppedRuntime:
        def submit_group(self, _coroutines):
            raise RuntimeError("event loop stopped")

        def stop(self) -> None:
            pass

    manager = LwdEdgeProbeManager(_config(True), lambda: None)
    manager._runtime = _StoppedRuntime()

    assert manager.submit(_request()) is LwdProbeSubmission.UNAVAILABLE
    assert not manager._inflight
    manager.close()


def test_resubmission_supersedes_old_batch_and_advances_nonce() -> None:
    manager = LwdEdgeProbeManager(_config(True), lambda: None)
    calls: list[tuple[int, int]] = []
    activity = {"active": 0, "max_active": 0}
    clients = {target: _Client(target, calls, activity) for target in manager._targets}
    manager._client = lambda target, _block_size: clients[target]
    try:
        request = _request()
        assert manager.submit(request) is LwdProbeSubmission.SUBMITTED
        assert manager.submit(request) is LwdProbeSubmission.SUBMITTED
        completion = _drain(manager)
        assert completion.probe_nonce == 2
        assert len(manager._inflight) == 1
    finally:
        manager.close()


class _DrainClient(_Client):
    async def activate(self, *args, **kwargs):
        return asyncio.create_task(asyncio.sleep(60))


def test_abort_cancels_only_unadmitted_usage_drain() -> None:
    manager = LwdEdgeProbeManager(_config(True), lambda: None)
    calls: list[tuple[int, int]] = []
    activity = {"active": 0, "max_active": 0}
    clients = {
        target: _DrainClient(target, calls, activity) for target in manager._targets
    }
    manager._client = lambda target, _block_size: clients[target]

    def activate_winner():
        completion = _drain(manager)
        winner = max(
            completion.candidates,
            key=lambda candidate: candidate.normalized_hit_tokens(16),
        )
        assert manager.activate(completion, winner.connection_id, winner)
        return _drain(manager, activated=True)

    try:
        request = _request()
        assert manager.submit(request) is LwdProbeSubmission.SUBMITTED
        activate_winner()
        first_task = manager._inflight["req"].drain_task
        manager.abort("req")
        deadline = time.monotonic() + 1
        while not first_task.done() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert first_task.cancelled()

        assert manager.submit(request) is LwdProbeSubmission.SUBMITTED
        activated = activate_winner()
        second_task = manager._inflight["req"].drain_task
        manager.mark_admitted("req", activated.probe_nonce)
        manager.abort("req")
        time.sleep(0.01)
        assert not second_task.done()
        manager._runtime.cancel_task(second_task)
    finally:
        manager.close()


def test_admit_marks_usage_drain_only_after_request_notify() -> None:
    calls = []
    connection = LwdConnectionId(0, 0, 0)
    request = SimpleNamespace(request_id="r", lwd_prefix_nonce=7)
    engine = object.__new__(LwdEdgeEngineCore)
    engine.scheduler = SimpleNamespace(lwd_admit_request=Mock(return_value=connection))
    engine._lwd_coord_clients = {}
    engine._lwd_notify_request = lambda _dp, _request: calls.append("notify")
    engine._lwd_probe_manager = Mock()
    engine._lwd_probe_manager.mark_admitted.side_effect = lambda _rid, _nonce: (
        calls.append("mark")
    )

    engine._lwd_admit_and_notify(request, connection)

    assert calls == ["notify", "mark"]
    engine._lwd_probe_manager.mark_admitted.assert_called_once_with("r", 7)


def test_admit_validation_failure_cancels_unadmitted_usage_drain() -> None:
    request = SimpleNamespace(request_id="r", lwd_prefix_nonce=7)
    engine = object.__new__(LwdEdgeEngineCore)
    engine.scheduler = SimpleNamespace(
        lwd_admit_request=Mock(side_effect=ValueError("invalid request")),
        lwd_edge_cancel_admission=Mock(),
    )
    engine._lwd_coord_clients = {}
    engine._lwd_notify_request = Mock()
    engine._lwd_probe_manager = Mock()

    with pytest.raises(ValueError, match="invalid request"):
        engine._lwd_admit_and_notify(request)

    engine._lwd_probe_manager.abort.assert_called_once_with("r")
    engine._lwd_notify_request.assert_not_called()
    engine._lwd_probe_manager.mark_admitted.assert_not_called()
    engine.scheduler.lwd_edge_cancel_admission.assert_called_once_with("r")


def test_admit_failure_cancels_unadmitted_usage_drain() -> None:
    connection = LwdConnectionId(0, 0, 0)
    request = SimpleNamespace(request_id="r", lwd_prefix_nonce=7)
    engine = object.__new__(LwdEdgeEngineCore)
    engine.scheduler = SimpleNamespace(
        lwd_admit_request=Mock(return_value=connection),
        lwd_edge_cancel_admission=Mock(),
    )
    engine._lwd_coord_clients = {}
    engine._lwd_notify_request = Mock(side_effect=RuntimeError("notify failed"))
    engine._lwd_probe_manager = Mock()

    with pytest.raises(RuntimeError, match="notify failed"):
        engine._lwd_admit_and_notify(request, connection)

    engine._lwd_probe_manager.abort.assert_called_once_with("r")
    engine._lwd_probe_manager.mark_admitted.assert_not_called()
    engine.scheduler.lwd_edge_cancel_admission.assert_called_once_with("r")


class _GatewayClient(_Client):
    async def activate_gateway(
        self, _session: object, request_id: str, _prompt: list[int], **kwargs: object
    ) -> LwdGatewayActivation:
        self.activity["max_tokens"] = kwargs["max_tokens"]
        self.calls.append((99, 99))
        result = LwdProbeResult(
            request_id=request_id,
            instance_id="cloud-0",
            block_size=4,
            hit_blocks=1,
            hit_tokens=4,
            cloud_id=0,
            engine_index=1,
        )
        return LwdGatewayActivation(
            result=result,
            prefix_nonce=9,
            drain_task=asyncio.create_task(asyncio.sleep(60)),
        )


def _gateway_config() -> SimpleNamespace:
    config = _config(True)
    config.lwd_config.topology.gateway = SimpleNamespace(
        enabled=True,
        url="https://gateway.example.com/custom/v1/chat/completions",
        connect_timeout=1.0,
    )
    return config


def test_gateway_binds_returned_dp_without_edge_preselection() -> None:
    manager = LwdGatewayActivationManager(_gateway_config(), lambda: None)
    calls: list[tuple[int, int]] = []
    activity = {"active": 0, "max_active": 0}
    target = LwdConnectionId(0, 0, 1)
    client = _GatewayClient(target, calls, activity)
    manager._client = lambda _url, _block_size: client
    try:
        assert manager.submit(_request()) is LwdProbeSubmission.SUBMITTED
        completion = _drain(manager, activated=True)
        assert completion.connection_id == target
        assert completion.result is not None
        assert completion.result.engine_index == 1
        assert completion.probe_nonce == 9
        assert calls == [(99, 99)]
        assert activity["max_tokens"] == 32

        drain_task = manager._inflight["req"].drain_task
        manager.abort("req")
        assert not drain_task.done()
        manager._runtime.cancel_task(drain_task)
    finally:
        manager.close()


@pytest.mark.parametrize("sampling_params", [None, SimpleNamespace(max_tokens=None)])
def test_gateway_missing_max_tokens_uses_lwd_default_for_usage_timeout(
    sampling_params: SimpleNamespace | None,
) -> None:
    manager = LwdGatewayActivationManager(_gateway_config(), lambda: None)
    activity: dict[str, int] = {}
    client = _GatewayClient(LwdConnectionId(0, 0, 1), [], activity)
    manager._client = lambda _url, _block_size: client
    request = _request()
    request.sampling_params = sampling_params
    try:
        assert manager.submit(request) is LwdProbeSubmission.SUBMITTED
        _drain(manager, activated=True)
        assert activity["max_tokens"] == 16
    finally:
        manager.close()


def test_gateway_rejects_ambiguous_or_mismatched_identity() -> None:
    manager = LwdGatewayActivationManager(_gateway_config(), lambda: None)
    base = LwdProbeResult(
        request_id="req",
        instance_id="cloud-0",
        block_size=4,
        hit_blocks=0,
        hit_tokens=0,
        cloud_id=None,
        engine_index=None,
    )
    with pytest.raises(RuntimeError, match="engine_index"):
        manager._resolve_connection(base, edge_id=0)

    config = _gateway_config()
    config.lwd_config.connections = [
        _connection(0, 0, 0),
        _connection(0, 1, 0),
    ]
    multi_cloud = LwdGatewayActivationManager(config, lambda: None)
    with pytest.raises(RuntimeError, match="cloud_id"):
        multi_cloud._resolve_connection(base, edge_id=0)

    wrong_instance = LwdProbeResult(
        request_id="req",
        instance_id="cloud-1",
        block_size=4,
        hit_blocks=0,
        hit_tokens=0,
        cloud_id=0,
        engine_index=0,
    )
    with pytest.raises(RuntimeError, match="different instance"):
        manager._resolve_connection(wrong_instance, edge_id=0)


def test_gateway_submit_failure_returns_failed_and_cleans_inflight() -> None:
    class _StoppedRuntime:
        def submit(self, _coroutine, *, timeout=None):
            raise RuntimeError("event loop stopped")

        def stop(self) -> None:
            pass

    manager = LwdGatewayActivationManager(_gateway_config(), lambda: None)
    manager._runtime = _StoppedRuntime()

    assert manager.submit(_request()) is LwdProbeSubmission.FAILED
    assert not manager._inflight
    manager.close()


def test_gateway_registration_widens_probe_targeting() -> None:
    """Elastic scale-out: a runtime-adopted connection widens the gateway's
    visible target set and rotation, idempotently.

    Rewritten for the gateway interface: there is no per-cloud coordination
    URL to register. The gateway service owns Cloud selection, so only the
    connection id matters for _visible_targets.
    """
    manager = LwdGatewayActivationManager(_gateway_config(), lambda: None)
    added = LwdConnectionId(0, 1, 0)
    startup = manager._visible_targets(0)
    assert startup == (
        LwdConnectionId(0, 0, 0),
        LwdConnectionId(0, 0, 1),
    )
    manager.register_connection(added, "http://unused-by-gateway")
    assert manager._visible_targets(0) == (*startup, added)
    assert (1, 0) in manager.rotation()
    manager.register_connection(added, "http://unused-by-gateway")
    assert manager._visible_targets(0) == (*startup, added)
    manager.close()


def test_gateway_registration_alone_enables_probing() -> None:
    """A gateway manager with no startup targets has nothing to resolve;
    one runtime registration makes the Cloud/DP visible.

    Unlike the direct probe manager, ``enabled`` is a configuration property
    (gateway URL + coordination), not a function of registered targets —
    that is the semantic difference this rewrite captures.
    """
    config = _gateway_config()
    config.lwd_config.connections = []
    manager = LwdGatewayActivationManager(config, lambda: None)
    assert manager._visible_targets(0) == ()
    assert manager.rotation() == ()
    target = LwdConnectionId(0, 5, 0)
    manager.register_connection(target, "http://runtime:8100")
    assert manager._visible_targets(0) == (target,)
    assert manager.rotation() == ((5, 0),)
    manager.close()


def test_gateway_silent_verdict_cached_until_recovered() -> None:
    """Cloud-grained liveness verdict is cached and cleared on recovery.

    Rewritten for the gateway interface: the gateway service chooses the
    Cloud/DP, so the edge does NOT drop silent clouds from
    _visible_targets — that would turn a legal gateway response into
    "unconfigured Cloud" and fail the request instead of falling back.
    The admission gate owns the fully-silent steady state.
    """
    manager = LwdGatewayActivationManager(_gateway_config(), lambda: None)
    visible = manager._visible_targets(0)
    manager.set_silent_clouds(frozenset({0}))
    assert manager.silent_clouds == frozenset({0})
    assert manager._visible_targets(0) == visible
    manager.set_silent_clouds(frozenset())
    assert manager.silent_clouds == frozenset()
    manager.set_silent_clouds(frozenset({0, 1}))
    assert manager.silent_clouds == frozenset({0, 1})
    manager.close()
