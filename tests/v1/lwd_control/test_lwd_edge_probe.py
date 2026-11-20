# SPDX-License-Identifier: Apache-2.0
"""Runtime cloud registration in the edge probe manager (elastic
scale-out).

The manager's cloud set was frozen at startup: the health rotation came
from the edge's own coordination block, and a legacy single-URL
deployment hardcoded cloud 0 — so after a mid-run scale-out every probe
(and the routing preference it produces) stayed pinned to the startup
clouds. register_cloud() widens the rotation with a runtime URL, and
submit() targets clouds through the unified _url_for resolution.

The fake runtime resolves every probe immediately with a success, so
each submit + drain roundtrip yields the completion's target cloud —
the same value the engine's routing preference is derived from.
"""

from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any

import pytest

try:
    from vllm.config.lwd.topology import LwdCoordination
    from vllm.v1.lwd_control.control_edge_scheduler import lwd_edge_probe
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_probe import (
        LwdEdgeProbeManager,
    )
except Exception:  # pragma: no cover - 轻量环境(见 conftest)
    pytest.skip(
        "probe manager 完整链不可用(需 torch/aiohttp)",
        allow_module_level=True,
    )


def _request(request_id: str) -> Any:
    return SimpleNamespace(
        request_id=request_id, prompt_token_ids=[1, 2, 3], mm_features=[]
    )


def _manager(coordination: LwdCoordination) -> LwdEdgeProbeManager:
    return LwdEdgeProbeManager(
        SimpleNamespace(
            lwd_config=SimpleNamespace(coordination=coordination, instance_id=0),
            cache_config=SimpleNamespace(block_size=128),
            model_config=SimpleNamespace(served_model_name="m", model="m"),
        ),
        lambda: None,
    )


@pytest.fixture
def probe_targets(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Resolve every probe immediately with a success (no HTTP at all)
    and hand back the per-submit target sequence via submit+drain."""

    class _ImmediateRuntime:
        def __init__(self) -> None:
            self.submitted = 0

        def submit(self, coroutine: Any) -> Future[Any]:
            self.submitted += 1
            coroutine.close()
            future: Future[Any] = Future()
            future.set_result((SimpleNamespace(block_size=128), None))
            return future

    runtimes: list[_ImmediateRuntime] = []

    def _runtime() -> _ImmediateRuntime:
        runtime = _ImmediateRuntime()
        runtimes.append(runtime)
        return runtime

    class _NoHTTPClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

    monkeypatch.setattr(lwd_edge_probe, "LwdEdgePrefixClient", _NoHTTPClient)
    monkeypatch.setattr(lwd_edge_probe, "LwdProbeRuntime", _runtime)
    monkeypatch.setattr(
        lwd_edge_probe, "compute_processor_fingerprint", lambda _cfg: b"f"
    )

    state = {"submitted": 0}

    def targets(manager: LwdEdgeProbeManager, count: int) -> list[int]:
        """One submit + drain per round: the target the probe LEFT for
        the routing preference. Callable repeatedly per manager."""
        out: list[int] = []
        for i in range(count):
            assert manager.submit(_request(f"r{state['submitted'] + i}"))
            (completion,) = manager.drain()
            assert completion.result is not None
            out.append(completion.target_cloud)
        state["submitted"] += count
        assert runtimes and runtimes[0].submitted == state["submitted"]
        return out

    return targets


def test_submit_rotates_over_the_startup_map(probe_targets: Any) -> None:
    manager = _manager(
        LwdCoordination(
            enabled=True,
            control_url_map={0: "http://c0:8100", 2: "http://c2:8100"},
            probe_timeout=6.0,
        )
    )
    assert probe_targets(manager, 4) == [0, 2, 0, 2]


def test_register_cloud_widens_the_rotation(probe_targets: Any) -> None:
    manager = _manager(
        LwdCoordination(
            enabled=True,
            control_url_map={0: "http://c0:8100"},
            probe_timeout=6.0,
        )
    )
    manager.register_cloud(1, "http://c1:8100")
    assert probe_targets(manager, 4) == [0, 1, 0, 1]


def test_legacy_single_url_seeds_cloud_0_and_joins_the_rotation(
    probe_targets: Any,
) -> None:
    manager = _manager(
        LwdCoordination(enabled=True, control_url="http://c0:8100", probe_timeout=6.0)
    )
    assert manager._url_for(0) == "http://c0:8100"
    manager.register_cloud(1, "http://c1:8100")
    assert probe_targets(manager, 2) == [0, 1]


def test_url_resolution_prefers_the_runtime_registration() -> None:
    manager = _manager(
        LwdCoordination(
            enabled=True,
            control_url_map={0: "http://c0:8100"},
            probe_timeout=6.0,
        )
    )
    manager.register_cloud(1, "http://runtime:8100")
    assert manager._url_for(1) == "http://runtime:8100"
    assert manager._url_for(0) == "http://c0:8100"
    assert manager._url_for(3) is None
    # First registration wins; a replay never rewrites the URL.
    manager.register_cloud(1, "http://later:8100")
    assert manager._url_for(1) == "http://runtime:8100"


def test_a_runtime_registration_alone_enables_probing() -> None:
    manager = _manager(LwdCoordination(enabled=True, probe_timeout=6.0))
    assert not manager.enabled
    manager.register_cloud(5, "http://c5:8100")
    assert manager.enabled
    assert manager._url_for(5) == "http://c5:8100"
    assert manager._url_for(0) is None


def test_silent_clouds_leave_probe_targeting_until_recovered(
    probe_targets: Any,
) -> None:
    """The liveness verdict (cloud-grained) stops probe targeting a fully
    silent cloud — without it, a killed member keeps eating 3 fast-fail
    probes per cooldown cycle and the affected requests lose their probe
    preference. Fails open when the exclusion covers everything: the
    admission gate owns the fully-silent steady state."""
    manager = _manager(
        LwdCoordination(
            enabled=True,
            control_url_map={0: "http://c0:8100", 1: "http://c1:8100"},
            probe_timeout=6.0,
        )
    )
    assert probe_targets(manager, 2) == [0, 1]
    manager.set_silent_clouds(frozenset({1}))
    assert probe_targets(manager, 4) == [0, 0, 0, 0]
    # Recovery (the engine pushes the empty set on the resumed beat):
    # the cursor carried through the excluded period, the rotation just
    # continues over the full tuple.
    manager.set_silent_clouds(frozenset())
    assert probe_targets(manager, 2) == [0, 1]
    # Fail open: every rotation member silent — probing still beats not
    # probing in the gate's race window.
    manager.set_silent_clouds(frozenset({0, 1}))
    assert probe_targets(manager, 2) == [0, 1]


def test_drain_rebuilds_a_runtime_cloud_manifest_without_a_keyerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The drain-time block-size rebuild resolves its URL through
    _url_for: the old control_url_map[target] lookup raised KeyError the
    moment a runtime-adopted cloud answered with a different hash block
    size."""
    rebuilt_urls: list[str] = []

    class _RebuildClient:
        def __init__(self, url: str, *_args: Any, **_kwargs: Any) -> None:
            rebuilt_urls.append(url)

        def build_manifest(self, *_args: Any, **_kwargs: Any) -> Any:
            return SimpleNamespace(full_block_hashes=[b"x", b"y"])

    class _RebuildRuntime:
        def submit(self, coroutine: Any) -> Future[Any]:
            coroutine.close()
            future: Future[Any] = Future()
            future.set_result((SimpleNamespace(block_size=64), [b"chain"]))
            return future

    monkeypatch.setattr(lwd_edge_probe, "LwdEdgePrefixClient", _RebuildClient)
    monkeypatch.setattr(lwd_edge_probe, "LwdProbeRuntime", _RebuildRuntime)
    monkeypatch.setattr(
        lwd_edge_probe, "compute_processor_fingerprint", lambda _cfg: b"f"
    )
    manager = _manager(
        LwdCoordination(
            enabled=True,
            control_url_map={0: "http://c0:8100"},
            probe_timeout=6.0,
        )
    )
    manager.register_cloud(1, "http://c1:8100")
    for i in range(2):
        assert manager.submit(_request(f"r{i}"))
    completions = manager.drain()
    assert [completion.target_cloud for completion in completions] == [0, 1]
    # Submit-time clients for both clouds, then the 64-token rebuilds —
    # the runtime cloud's rebuild resolved ITS URL instead of raising.
    assert rebuilt_urls == [
        "http://c0:8100",
        "http://c1:8100",
        "http://c0:8100",
        "http://c1:8100",
    ]
    assert completions[1].chain == [b"x", b"y"]
