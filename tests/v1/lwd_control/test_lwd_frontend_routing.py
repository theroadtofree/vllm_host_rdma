# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MR5 regressions: multi-frontend frontend attach (registration and
output grouping).

Covers the behavioral seams of the frontend-attach plan:

* A - scene validation (``LwdAPIConfig.validate_topology``):
  ``edge_share`` keeps the one-to-one primary/non-primary pairing and is
  the only scene that accepts attach metadata; ``client_index`` stays a
  frontend grouping key (the attach frontend's non-primary edge instance
  id), never a topology identity;
* B - the edge scheduler's ``_lwd_req_id_to_client`` pin table and the
  per-client output grouping in ``update_from_output`` (added with the
  grouping commit).

Everything is CPU-only: scene validation runs against real typed
topologies (no sockets), the scheduler seams are exercised via AST
extraction onto a stub base class, and the serve wiring is checked with
source-scan guards (no process is started).
"""

from __future__ import annotations

import ast
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest

from vllm.config.lwd import LwdAPIConfig, LwdConfig, LwdTopology

REPO_ROOT = Path(__file__).resolve().parents[3]
VLLM_DIR = REPO_ROOT / "vllm"
SERVE_PATH = VLLM_DIR / "entrypoints" / "cli" / "serve.py"
ATTACH_PATH = VLLM_DIR / "entrypoints" / "openai" / "lwd" / "frontend_attach.py"
SCHED_PATH = (
    VLLM_DIR / "v1" / "lwd_control" / "control_edge_scheduler" / "lwd_edge_scheduler.py"
)

# 1E1C edge_share: two one-to-one pairs (edge i -> cloud i).
EDGE_SHARE = ("edge_share", 2, 2)
# 2E1C cloud_share: two edges sharing one cloud, full mesh.
CLOUD_SHARE = ("cloud_share", 2, 1)
# 2E2C lwd_cluster: full mesh.
LWD_CLUSTER = ("lwd_cluster", 2, 2)
# 1E1C single instance.
SINGLE_INSTANCE = ("single_instance", 1, 1)


def _topology(shape: tuple[str, int, int]) -> LwdTopology:
    """Valid typed topology for a scene.

    edge_share instances share one edge allocation (one physical engine
    group); every other scene gives each instance its own rank. Cloud DPs
    always carry a ctrl_port; distinct cloud DPs on one address use
    different ports.
    """
    scene, edges, clouds = shape
    shared_edge = scene == "edge_share"
    # Shared/multi-cloud scenes require an enabled coordination block on
    # every edge (topology validation); edge_share/single_instance don't.
    coordinated = scene in ("cloud_share", "lwd_cluster")
    edge_entries = [
        {
            "id": edge_id,
            **(
                {
                    "coordination": {
                        "enabled": True,
                        "consumer_id": f"enterprise-a-edge{edge_id}",
                    }
                }
                if coordinated
                else {}
            ),
            "dp": [
                {
                    "dp_idx": 0,
                    "addr": "127.0.0.1",
                    "ranks": [0] if shared_edge else [edge_id],
                }
            ],
        }
        for edge_id in range(edges)
    ]
    cloud_base = 1 if shared_edge else edges
    cloud_entries = [
        {
            "id": cloud_id,
            **({"coordination": {"enabled": True}} if coordinated else {}),
            "dp": [
                {
                    "dp_idx": 0,
                    "addr": "127.0.0.1",
                    "ranks": [cloud_base + cloud_id],
                    "ctrl_port": 5550 + cloud_id,
                    # Enabled cloud coordination requires an advertised
                    # coordination endpoint on every DP (coord_url derives
                    # from addr + coord_port).
                    **({"coord_port": 9130 + cloud_id} if coordinated else {}),
                }
            ],
        }
        for cloud_id in range(clouds)
    ]
    if scene == "edge_share":
        links = [{"edge": i, "cloud": i} for i in range(edges)]
    else:
        links = [{"edge": i, "cloud": j} for i in range(edges) for j in range(clouds)]
    return LwdTopology.from_dict(
        {
            "deployment": {
                "mode": 0,
                "scene": scene,
                "hccl_world_size": cloud_base + clouds,
                "edges_num": edges,
                "clouds_num": clouds,
            },
            "feature_ctrl": {"enable_early_recv": False},
            "edges": edge_entries,
            "clouds": cloud_entries,
            "instance_links": links,
        }
    )


def _lwd(
    shape: tuple[str, int, int], role: str, instance_id: int, **api: str
) -> LwdConfig:
    """LWD identity with attach API metadata; construction itself runs
    ``LwdAPIConfig.validate_topology`` (raises on any violated rule)."""
    return LwdConfig(
        path="",
        role=role,
        instance_id=instance_id,
        topology=_topology(shape),
        api=LwdAPIConfig(**api),
    )


class TestAttachSceneValidation:
    """A: scene rules of the attach metadata validation."""

    ATTACH = "10.0.0.1:9700"
    RPC = "10.0.0.1:9701"

    def test_edge_share_attach_requires_non_primary_edge(self) -> None:
        # Legacy one-to-one pairing: frontend i is edge instance i's
        # frontend; instance 0's frontend belongs to the main command.
        _lwd(EDGE_SHARE, "edge", 1, api_server_attach=self.ATTACH)
        with pytest.raises(ValueError, match="non-primary"):
            _lwd(EDGE_SHARE, "edge", 0, api_server_attach=self.ATTACH)

    def test_edge_share_rpc_port_requires_primary_edge(self) -> None:
        _lwd(EDGE_SHARE, "edge", 0, api_server_rpc_port=self.RPC)
        with pytest.raises(ValueError, match="instance_id=0"):
            _lwd(EDGE_SHARE, "edge", 1, api_server_rpc_port=self.RPC)

    def test_decoupled_scenes_reject_attach(self) -> None:
        # Attachment remains an edge_share feature; cloud_share and
        # lwd_cluster do not acquire frontend pairing semantics.
        for shape in (CLOUD_SHARE, LWD_CLUSTER):
            with pytest.raises(ValueError, match="scene=edge_share"):
                _lwd(shape, "edge", 1, api_server_attach=self.ATTACH)

    def test_decoupled_scenes_keep_primary_rules(self) -> None:
        with pytest.raises(ValueError, match="non-primary"):
            _lwd(CLOUD_SHARE, "edge", 0, api_server_attach=self.ATTACH)
        with pytest.raises(ValueError, match="scene=edge_share"):
            _lwd(CLOUD_SHARE, "edge", 0, api_server_rpc_port=self.RPC)
        with pytest.raises(ValueError, match="instance_id=0"):
            _lwd(LWD_CLUSTER, "edge", 1, api_server_rpc_port=self.RPC)

    def test_single_instance_scene_rejects_attach(self) -> None:
        # single_instance has exactly one edge (instance 0), so attach
        # trips the non-primary rule first; either rule rejects it.
        with pytest.raises(ValueError, match="non-primary edge|scene=edge_share"):
            _lwd(SINGLE_INSTANCE, "edge", 0, api_server_attach=self.ATTACH)

    def test_cloud_role_rejects_attach(self) -> None:
        with pytest.raises(ValueError, match="edge instances only"):
            _lwd(EDGE_SHARE, "cloud", 0, api_server_attach=self.ATTACH)

    def test_rpc_port_and_attach_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            _lwd(
                EDGE_SHARE,
                "edge",
                0,
                api_server_rpc_port=self.RPC,
                api_server_attach=self.ATTACH,
            )

    def test_unrequested_api_config_accepts_any_identity(self) -> None:
        # No API options: validation is inert (single-instance identity
        # without attach flags must construct cleanly).
        _lwd(SINGLE_INSTANCE, "edge", 0)


def _load_attach_fn(fn_name: str, **dependencies: Any) -> Any:
    """Extract one top-level function from the attach module and compile
    it with stubbed heavy dependencies (the module import chain pulls in
    zmq/vllm; the functions under test only need lightweight stand-ins)."""
    tree = ast.parse(ATTACH_PATH.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name
    )
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(ATTACH_PATH), "exec"), dependencies)
    return dependencies[fn_name]


_LWD_IDENTITIES: dict[int, NS] = {}


def _stub_lwd_identity(additional_config: dict[str, Any], *, api: LwdAPIConfig) -> NS:
    """Identity stub keyed by the additional-config mapping."""
    return _LWD_IDENTITIES[id(additional_config)]


class TestFrontendCountDecoupling:
    """Frontend count resolution through the current attach entrypoints."""

    RPC = "10.0.0.1:9701"

    def _args(self, edges: int, count: int | None) -> NS:
        additional_config = {"marker": object()}
        args = NS(
            api_server_count=count,
            api_server_rpc_port=self.RPC,
            api_server_attach=None,
            additional_config=additional_config,
        )
        _LWD_IDENTITIES[id(additional_config)] = NS(
            instance_id=0,
            topology=NS(
                digest="topology-digest",
                edges=[None] * edges,
            ),
        )
        return args

    def _resolve(self, edges: int, count: int | None) -> int:
        args = self._args(edges, count)
        resolve = _load_attach_fn(
            "resolve_attach_plan",
            LwdAPIConfig=LwdAPIConfig,
            envs=NS(VLLM_USE_RUST_FRONTEND=False),
            resolve_lwd_config=_stub_lwd_identity,
            _additional_config_dict=lambda value: value.additional_config,
            FrontendAttachPlan=NS,
        )
        try:
            return resolve(args).client_count
        finally:
            _LWD_IDENTITIES.pop(id(args.additional_config), None)

    def test_count_defaults_to_edge_instances(self) -> None:
        assert self._resolve(3, None) == 3
        assert self._resolve(2, 2) == 2

    def test_explicit_count_must_pair_one_to_one(self) -> None:
        with pytest.raises(ValueError, match="edge_share"):
            self._resolve(3, 2)

    def test_prepare_attach_mode_uses_resolved_count(self) -> None:
        args = self._args(3, 3)
        registration_server = object()
        prepare = _load_attach_fn(
            "prepare_attach_mode",
            create_frontend_registration_server=lambda actual_args, count: (
                registration_server
            ),
        )
        try:
            assert prepare(args, 3, False) == (True, 1, registration_server)
        finally:
            _LWD_IDENTITIES.pop(id(args.additional_config), None)

    def test_explicit_zero_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="api-server-count"):
            self._resolve(2, 0)


def test_serve_wires_frontend_attach_dispatch() -> None:
    """Source-scan guard: the serve entrypoint dispatches attach commands
    before the generic frontend-count defaulting and runs the count-to-K
    barrier inside the engine-launch scope."""
    serve = SERVE_PATH.read_text(encoding="utf-8")
    assert "lwd_dispatch_attach_frontend" in serve
    assert "prepare_attach_mode" in serve
    assert "wait_for_attach_registrations" in serve
    assert "attach_scope" in serve
    assert "lwd_control" not in serve


def test_attach_module_homes_the_registration_protocol() -> None:
    """Source-scan guard: the protocol frame types and both entrypoints
    live in the entrypoints module, platform-agnostic (no engine-side
    rendezvous concepts)."""
    attach = ATTACH_PATH.read_text(encoding="utf-8")
    for frame in (
        '_MSG_ATTACH = "LWD_FRONTEND_ATTACH"',
        '_MSG_ACCEPT = "LWD_FRONTEND_ACCEPT"',
        '_MSG_ENDPOINTS = "LWD_FRONTEND_ENDPOINTS"',
        '_MSG_BARRIER = "LWD_FRONTEND_BARRIER"',
        '_MSG_ERROR = "LWD_FRONTEND_ERROR"',
    ):
        assert frame in attach
    assert "def run_attach_api_server(" in attach
    assert "def create_frontend_registration_server(" in attach
    for retired in ("instance_parallel", "role_registry"):
        assert retired not in attach


# ------------------------------------------------------------------ #
# B: the edge scheduler's frontend pin table and per-client grouping  #
# ------------------------------------------------------------------ #

_LWD_NOT_FINISHED = -1


class _StubLogger:
    """vllm.lwd_logger stand-in: the module import chain pulls in torch
    (unavailable on light hosts); the methods under test only use
    warning/info."""

    def warning(self, *a: Any, **k: Any) -> None:
        pass

    def info(self, *a: Any, **k: Any) -> None:
        pass


class _StubDiag:
    """LwdControlDiag stand-in: token traces only."""

    @staticmethod
    def edge_tokens_delivered(*a: Any, **k: Any) -> None:
        pass

    @staticmethod
    def trace_edge_kill_row(*a: Any, **k: Any) -> None:
        pass


class _FinishReason:
    """FinishReason stand-in: sentinel codes plus the callable used by
    _lwd_finish_code to re-materialize a code."""

    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"
    ERROR = "error"

    def __init__(self, code: str) -> None:
        self.code = code


class _RequestStatus:
    FINISHED_STOPPED = "finished_stopped"
    FINISHED_LENGTH_CAPPED = "finished_length_capped"
    FINISHED_ABORTED = "finished_aborted"


class _EngineCoreOutput:
    def __init__(
        self,
        request_id: str,
        new_token_ids: list[int],
        finish_reason: object = None,
    ) -> None:
        self.request_id = request_id
        self.new_token_ids = new_token_ids
        self.finish_reason = finish_reason


class _EngineCoreOutputs:
    def __init__(
        self,
        outputs: list[_EngineCoreOutput] | None = None,
        finished_requests: set[str] | None = None,
    ) -> None:
        self.outputs = outputs
        self.finished_requests = finished_requests


class _LwdBatchType:
    LWD_UNEMBED = "unembed"


_FINISH_STATUS = {
    _FinishReason.STOP: _RequestStatus.FINISHED_STOPPED,
    _FinishReason.LENGTH: _RequestStatus.FINISHED_LENGTH_CAPPED,
    _FinishReason.ABORT: _RequestStatus.FINISHED_ABORTED,
    _FinishReason.ERROR: _RequestStatus.FINISHED_ABORTED,
}


class _Account:
    """Cloud-DP account bookkeeping stand-in: only the fields the settle
    path touches."""

    def __init__(self, connection_id: tuple[str, int, int]) -> None:
        self.connection_id = connection_id
        self.pending_batches = {}
        self.dropped = []

    def drop_request(self, request_id: str) -> None:
        self.dropped.append(request_id)


_CONNECTION = ("cloud", 0, 0)


class _SchedBase:
    """Attribute/method stubs for everything the extracted scheduler
    methods reference but do not need real logic for."""

    def __init__(self) -> None:
        self.requests = {}
        self.vllm_config = None
        self._lwd_route = {}
        self._lwd_accounts = {_CONNECTION: _Account(_CONNECTION)}
        self._lwd_req_id_to_client = {}
        self._lwd_mrope_positions_dict = {}
        self._lwd_pending_refills = deque()
        self._lwd_aborted_req_ids = {_CONNECTION: []}
        self._lwd_announced_refills = set()
        # 引擎注入的回调钩子(真实调度器在 __init__ 尾部置 None,由引擎
        # 接线;这里给可空实现,投递路径的调用即 no-op)。
        self.lwd_on_tokens = lambda *a, **k: None
        self.lwd_on_finish = lambda *a, **k: None

    def _lwd_validate_request(self, request: NS) -> None:
        pass

    def _lwd_compute_mrope_positions(self, request: NS) -> None:
        pass

    def _lwd_edge_id_of(self, request: NS) -> int:
        # 镜像真实调度器的 edge_share 语义:client_index 缺省的旧式调用
        # 方默认 0(与 _request() 不设置该属性的构造一致)。
        return getattr(request, "client_index", 0)

    def _lwd_route_request(
        self, request_id: str, edge_id: int = 0, preferred: object = None
    ) -> tuple[str, int, int]:
        self._lwd_route[request_id] = _CONNECTION
        return _CONNECTION

    def _lwd_dp_of_settling(self, request_id: str) -> tuple[str, int, int] | None:
        return self._lwd_route.get(request_id)

    def _lwd_account_of(self, request_id: str) -> _Account | None:
        connection_id = self._lwd_dp_of_settling(request_id)
        return None if connection_id is None else self._lwd_accounts[connection_id]

    def _lwd_watermark_connection(self, key: Any) -> Any:
        return key

    def _lwd_purge_mrope_positions(self) -> None:
        pass

    def finish_requests(self, request_ids: list[str], status: object = None) -> None:
        for rid in request_ids:
            request = self.requests.get(rid)
            if request is not None:
                request.status = status
                request.finished = True


_PIN_METHODS = (
    "lwd_admit_request",
    "lwd_edge_cancel_admission",
    "lwd_edge_abort",
    "lwd_edge_deliver_tokens",
    "_lwd_settle_request",
    "update_from_output",
    "_lwd_deliver_unembed",
    "_lwd_row_disposition",
    "_lwd_finish_code",
)


def _make_sched() -> Any:
    """Compile the pin/grouping methods out of the scheduler source into a
    class block inheriting the stub base (super() needs the class body's
    __class__ cell), with the module-level dependencies stubbed."""
    tree = ast.parse(SCHED_PATH.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LwdEdgeScheduler"
    )
    fns = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in _PIN_METHODS
    ]
    assert {fn.name for fn in fns} == set(_PIN_METHODS), "pin method drifted"
    class_def = ast.ClassDef(
        name="LwdEdgeSchedulerUnderTest",
        bases=[ast.Name(id="_SchedBase", ctx=ast.Load())],
        keywords=[],
        body=fns,
        decorator_list=[],
    )
    module = ast.Module(body=[class_def], type_ignores=[])
    ast.fix_missing_locations(module)
    scope: dict = {
        "_SchedBase": _SchedBase,
        "lwd_logger": _StubLogger(),
        "LwdControlDiag": _StubDiag(),
        "FinishReason": _FinishReason,
        "RequestStatus": _RequestStatus,
        "_LWD_FINISH_STATUS": _FINISH_STATUS,
        "EngineCoreOutput": _EngineCoreOutput,
        "EngineCoreOutputs": _EngineCoreOutputs,
        "LwdBatchType": _LwdBatchType,
        "LWD_NOT_FINISHED": _LWD_NOT_FINISHED,
        "defaultdict": defaultdict,
        "deque": deque,
    }
    exec(compile(module, str(SCHED_PATH), "exec"), scope)
    return scope["LwdEdgeSchedulerUnderTest"]()


def _request(rid: str, client_index: int | None = None, finished: bool = False) -> NS:
    """Request stand-in; ``client_index`` left unset mimics legacy
    single-frontend callers (pin defaults to 0)."""
    request = NS(
        request_id=rid,
        lwd_prefix_nonce=0,
        status="finished" if finished else "waiting",
        finished=finished,
    )
    if client_index is not None:
        request.client_index = client_index
    request.is_finished = lambda: request.finished
    # 投递路径对每个 token 调 request.append_output_token_ids(...)
    request.append_output_token_ids = lambda *a, **k: None
    return request


def _notify(req_ids: list[str], finish_reasons: list[object] | None = None) -> NS:
    return NS(
        req_ids=list(req_ids),
        finish_reasons=list(finish_reasons or [_LWD_NOT_FINISHED] * len(req_ids)),
        num_accepted_tokens=None,
        prefix_nonces=None,
        down_seqno=7,
    )


def _unembed_step(
    sched: Any, notify: NS, sampled: dict[str, list[int]]
) -> dict[int, Any]:
    """Drive update_from_output with one UNEMBED batch."""
    scheduler_output = NS(
        lwd_batch=NS(batch_type=_LwdBatchType.LWD_UNEMBED),
        lwd_c2e_notify=[notify],
    )
    model_runner_output = NS(
        req_ids=list(sampled),
        sampled_token_ids=[sampled[rid] for rid in sampled],
    )
    return sched.update_from_output(scheduler_output, model_runner_output)


class TestPinTableLifecycle:
    """B: req -> client_index pin write and cleanup (output grouping key)."""

    def test_admit_registers_pin(self) -> None:
        # Admission pins the route (the surviving pinned table); the
        # frontend identity lives on request.client_index and delivery
        # groups by it.
        sched = _make_sched()
        request = _request("r1", client_index=2)
        sched.lwd_admit_request(request)
        assert sched._lwd_route["r1"] == _CONNECTION
        assert request.client_index == 2
        legacy = _request("r2")
        sched.lwd_admit_request(legacy)
        assert sched._lwd_route["r2"] == _CONNECTION
        assert getattr(legacy, "client_index", 0) == 0

    def test_abort_pops_pin(self) -> None:
        sched = _make_sched()
        sched.lwd_admit_request(_request("r1", client_index=1))
        sched.lwd_edge_abort(["r1"])
        assert "r1" not in sched._lwd_req_id_to_client
        assert "r1" not in sched._lwd_route

    def test_abort_immediate_lifecycle_pops_pin(self) -> None:
        # The engine path for abort_immediately: admission (pin written,
        # cloud announced) then the native add_request aborts right away,
        # which routes through the same lwd_edge_abort cleanup.
        sched = _make_sched()
        sched.lwd_admit_request(_request("r1", client_index=1, finished=False))
        sched.lwd_edge_abort(["r1"])
        assert "r1" not in sched._lwd_req_id_to_client

    def test_cancel_admission_pops_pin(self) -> None:
        # The engine's announcement-failure path: the request leaves the
        # books before any output can exist.
        sched = _make_sched()
        sched.lwd_admit_request(_request("r1", client_index=3))
        sched.lwd_edge_cancel_admission("r1")
        assert "r1" not in sched._lwd_req_id_to_client

    def test_finish_pops_pin(self) -> None:
        sched = _make_sched()
        request = _request("r1", client_index=2)
        sched.requests["r1"] = request
        sched.lwd_admit_request(request)
        results = _unembed_step(
            sched,
            _notify(["r1"], [_FinishReason.STOP]),
            {"r1": [11]},
        )
        assert "r1" not in sched._lwd_req_id_to_client
        assert set(results) == {2}


class TestOutputGrouping:
    """B: outputs group per client_index into per-client
    EngineCoreOutputs; late outputs neither drop nor cross-group."""

    def test_multi_client_grouping(self) -> None:
        sched = _make_sched()
        for rid, client in (("a", 0), ("b", 1), ("c", 2)):
            request = _request(rid, client_index=client)
            sched.requests[rid] = request
            sched.lwd_admit_request(request)
        results = _unembed_step(
            sched,
            _notify(["a", "b", "c"]),
            {"a": [1], "b": [2], "c": [3]},
        )
        assert set(results) == {0, 1, 2}
        assert [out.request_id for out in results[1].outputs] == ["b"]
        assert [out.request_id for out in results[2].outputs] == ["c"]
        assert results[0].finished_requests is None

    def test_finished_requests_group_per_client(self) -> None:
        sched = _make_sched()
        for rid, client in (("a", 1), ("b", 2)):
            request = _request(rid, client_index=client)
            sched.requests[rid] = request
            sched.lwd_admit_request(request)
        results = _unembed_step(
            sched,
            _notify(["a", "b"], [_FinishReason.STOP, _FinishReason.LENGTH]),
            {"a": [1], "b": [2]},
        )
        assert set(results[1].finished_requests) == {"a"}
        assert set(results[2].finished_requests) == {"b"}

    def test_single_frontend_path_unchanged(self) -> None:
        # K=1 regression: unpinned requests group under 0 exactly like the
        # pre-attach hardcoded {0: EngineCoreOutputs(...)}.
        sched = _make_sched()
        for rid in ("a", "b"):
            request = _request(rid)
            sched.requests[rid] = request
            sched.lwd_admit_request(request)
        results = _unembed_step(
            sched,
            _notify(["a", "b"], [_FinishReason.STOP, _LWD_NOT_FINISHED]),
            {"a": [1], "b": [2]},
        )
        assert set(results) == {0}
        assert [out.request_id for out in results[0].outputs] == ["a", "b"]
        assert set(results[0].finished_requests) == {"a"}

    def test_late_row_for_finished_request_is_dropped(self) -> None:
        # Late cloud payload after the terminal settle: the request is
        # gone, the row is idempotently dropped -- nothing lands in any
        # group (in particular not the 0 group).
        sched = _make_sched()
        request = _request("r1", client_index=2)
        sched.requests["r1"] = request
        sched.lwd_admit_request(request)
        first = _unembed_step(sched, _notify(["r1"], [_FinishReason.STOP]), {"r1": [1]})
        assert set(first) == {2}
        late = _unembed_step(sched, _notify(["r1"], [_FinishReason.STOP]), {"r1": [9]})
        assert late == {}

    def test_late_row_for_inflight_request_keeps_pin(self) -> None:
        # The attach window: a request already streaming (first token
        # delivered, not finished) keeps its frontend identity
        # (request.client_index), so a late row still reaches the
        # originating frontend's group.
        sched = _make_sched()
        request = _request("r1", client_index=2)
        sched.requests["r1"] = request
        sched.lwd_admit_request(request)
        first = _unembed_step(sched, _notify(["r1"]), {"r1": [1]})
        assert set(first) == {2}
        assert sched.requests["r1"].client_index == 2
        late = _unembed_step(sched, _notify(["r1"]), {"r1": [2]})
        assert set(late) == {2}
        assert [out.new_token_ids for out in late[2].outputs] == [[2]]

    def test_terminal_output_reaches_pinned_client(self) -> None:
        # The grouping key is fixed at row entry, before the finished
        # branch settles the request and pops the pin: the terminal token
        # and finish signal of an attached frontend (client_index > 0)
        # must not fall back to the 0 group.
        sched = _make_sched()
        request = _request("r1", client_index=3)
        sched.requests["r1"] = request
        sched.lwd_admit_request(request)
        results = _unembed_step(
            sched, _notify(["r1"], [_FinishReason.STOP]), {"r1": [42]}
        )
        (out,) = results[3].outputs
        assert out.new_token_ids == [42]
        assert results[3].finished_requests == {"r1"}
        assert "r1" not in sched._lwd_req_id_to_client


if __name__ == "__main__":
    raise SystemExit("run with pytest")
