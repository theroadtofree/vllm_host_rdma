# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""edge_share 槽位计费回归(4 前端共享 1 个 EngineCore 的槽位塌缩修复)。

修复前的行为:共享 running 队列的占用整队对照单云配额(max_num_seqs,
与云侧逐云 force-equal)结算 -- 4 个前端挤一份配额,4x74 并发下
running 触顶后所有连接的准入闸一起关死,decode 依旧流畅、首 token
前的 prefill 全部停摆(TTFT x30,TPOT 反而变好),且槽位判据不写
HOLD 日志,现象在日志里不可见。

修复后的契约(本文件钉住):
- 计费单位是云 DP((cloud_id, dp_idx)):链接共享一个云 DP 时分同一
  份配额(cloud_share),指向不同云时各带一份(edge_share);
- 原生 running 上限 = 单云配额 x 服务的云 DP 数(原生调度器自身的
  等待准入与步末断言以此执法);
- picker 的 free_seq 同样按云 DP 结算;
- HOLD(slots) 与 HOLD(chunks) 都落日志,放开时记 OPEN。

方法提取自真实调度器源(AST,与 test_lwd_frontend_routing 同法);
被提取方法引用的模块级名字以桩注入,数据结构用真实类型
(LwdConnectionId / LwdCloudDpAccount / lwd_edge_picker 纯函数)。
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

try:
    from vllm.config.lwd import LwdConnectionId
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_cloud_dp_account import (
        LWD_MAX_PENDING_CHUNKS,
        LwdCloudDpAccount,
    )
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_picker import (
        LwdPrefillCandidate,
        lwd_grammar_ready,
        lwd_pick_prefill_batch,
    )
except Exception:  # pragma: no cover - 轻量环境垫片失效时显式跳过
    pytest.skip("lwd 纯逻辑模块不可导入", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHED_PATH = (
    REPO_ROOT
    / "vllm"
    / "v1"
    / "lwd_control"
    / "control_edge_scheduler"
    / "lwd_edge_scheduler.py"
)

_EXTRACTED_METHODS = (
    "_lwd_slot_target",
    "_lwd_resize_slot_pool",
    "_lwd_num_running_reqs",
    "_lwd_pick_serving_engine",
    "_lwd_log_gate_transition",
    "_lwd_has_prefill_chunk_inflight",
    "_lwd_pick_prefill_batch",
)
"""同步自 main_host_rdma_scale_perf 的两轮槽位修复;本仓尚无弹性扩连
端口(lwd_arm_runtime_connection),故不提取它——扩容后池重算的属性
由 test_resize_pool_recomputes_from_connections 直接驱动验证。"""


class _Logger:
    """lwd_logger 桩:记录格式化后的行,供 HOLD/OPEN 断言。"""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def _record(self, level: str, fmt: str, *args: object) -> None:
        self.lines.append(f"[{level}] {fmt % args if args else fmt}")

    def info(self, fmt: str, *args: object) -> None:
        self._record("info", fmt, *args)

    def warning(self, fmt: str, *args: object) -> None:
        self._record("warning", fmt, *args)

    def error(self, fmt: str, *args: object) -> None:
        self._record("error", fmt, *args)

    def exception(self, fmt: str, *args: object) -> None:
        self._record("error", fmt, *args)


class _SchedBase:
    """提取方法引用的宿主桩:只填被提取方法真正触碰的名字。"""

    def __init__(self) -> None:
        self.running: list[NS] = []
        self.waiting: list[NS] = []
        self.skipped_waiting: list[NS] = []
        self._lwd_route: dict[str, LwdConnectionId] = {}
        self._lwd_accounts: dict[LwdConnectionId, LwdCloudDpAccount] = {}
        self._lwd_connections: tuple[LwdConnectionId, ...] = ()
        self._lwd_cloud_slot_cap = 0
        self._lwd_last_engine = -1
        self.max_num_running_reqs = 0
        self.max_num_scheduled_tokens = 8192
        self._lwd_chunk_unit = 8192

    def _lwd_reqs_in_phase(self, phase: object, *queues: object):
        return iter(())

    def _lwd_dp_of(self, request_id: str) -> LwdConnectionId:
        return self._lwd_route[request_id]


def _make_sched(
    connections: tuple[LwdConnectionId, ...],
    slot_cap: int,
    *,
    running: list[NS] | None = None,
    route: dict[str, LwdConnectionId] | None = None,
    logger: _Logger | None = None,
) -> NS:
    tree = ast.parse(SCHED_PATH.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LwdEdgeScheduler"
    )
    fns = [n for n in cls.body if getattr(n, "name", None) in _EXTRACTED_METHODS]
    assert {fn.name for fn in fns} == set(_EXTRACTED_METHODS), "提取方法集漂移"
    class_def = ast.ClassDef(
        name="LwdEdgeSchedulerUnderTest",
        bases=[ast.Name(id="_SchedBase", ctx=ast.Load())],
        keywords=[],
        body=fns,
        decorator_list=[],
    )
    module = ast.Module(body=[class_def], type_ignores=[])
    ast.fix_missing_locations(module)
    logger = logger or _Logger()
    scope: dict = {
        "_SchedBase": _SchedBase,
        "lwd_logger": logger,
        "LwdPrefillCandidate": LwdPrefillCandidate,
        "LwdReqPhase": NS(PREFILL="prefill", DECODE="decode"),
        "lwd_grammar_ready": lwd_grammar_ready,
        "lwd_pick_prefill_batch": lwd_pick_prefill_batch,
        "RequestStatus": NS(WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR=object()),
    }
    exec(compile(module, str(SCHED_PATH), "exec"), scope)
    sched = scope["LwdEdgeSchedulerUnderTest"]()
    sched._lwd_connections = connections
    sched._lwd_cloud_slot_cap = slot_cap
    sched._lwd_accounts = {
        connection_id: LwdCloudDpAccount(connection_id) for connection_id in connections
    }
    sched._lwd_resize_slot_pool()
    sched.running = list(running or [])
    sched._lwd_route = dict(route or {})
    sched.logger = logger
    return sched


def _request(
    request_id: str,
    *,
    prompt: int = 10,
    computed: int | None = None,
) -> NS:
    """decode 期驻留(computed=prompt)是默认;waiting 新开候选需显式
    computed=0,否则 remaining=0 会被 picker 跳过。"""
    return NS(
        request_id=request_id,
        num_prompt_tokens=prompt,
        num_computed_tokens=computed if computed is not None else prompt,
        is_finished=lambda: False,
        status=None,
    )


def _edge_share_connections(n: int) -> tuple[LwdConnectionId, ...]:
    return tuple(LwdConnectionId(i, i, 0) for i in range(n))


class TestPerCloudDpBilling:
    """计费单位是云 DP,不是整条共享队列。"""

    def test_occupancy_bills_only_the_connections_cloud(self) -> None:
        conns = _edge_share_connections(4)
        route = {
            "r1": conns[0],
            "r2": conns[0],
            "r3": conns[1],
        }
        sched = _make_sched(conns, 2, running=[_request(r) for r in route], route=route)
        assert sched._lwd_num_running_reqs(conns[0]) == 2
        assert sched._lwd_num_running_reqs(conns[1]) == 1
        assert sched._lwd_num_running_reqs(conns[2]) == 0
        assert sched._lwd_num_running_reqs(conns[3]) == 0

    def test_billed_counts_sum_to_the_shared_queue(self) -> None:
        conns = _edge_share_connections(4)
        route = {f"r{i}": conns[i % 4] for i in range(9)}
        sched = _make_sched(conns, 8, running=[_request(r) for r in route], route=route)
        billed = sum(sched._lwd_num_running_reqs(c) for c in conns)
        assert billed == len(sched.running) == 9

    def test_gate_is_per_cloud_dp_not_global(self) -> None:
        # 修复前的塌缩形态:conn0 按自己的配额已满,但其他连接仍有
        # 空闲 -- 整队对照单份配额会把它们一起关死。
        conns = _edge_share_connections(4)
        route = {"r1": conns[0], "r2": conns[0], "r3": conns[1]}
        sched = _make_sched(
            conns,
            2,
            running=[_request(r) for r in ("r1", "r2")],
            route=route,
        )
        assert sched._lwd_pick_serving_engine() == conns[1]

    def test_all_engines_gate_only_when_their_own_quota_fills(self) -> None:
        conns = _edge_share_connections(2)
        route = {
            "a1": conns[0],
            "a2": conns[0],
            "b1": conns[1],
            "b2": conns[1],
        }
        sched = _make_sched(conns, 2, running=[_request(r) for r in route], route=route)
        assert sched._lwd_pick_serving_engine() is None
        # 两个云各自满配,而不是"整队 4 >= 单云配额 2"就全关。
        assert sched._lwd_num_running_reqs(conns[0]) == 2
        assert sched._lwd_num_running_reqs(conns[1]) == 2

    def test_eight_frontend_spec(self) -> None:
        # 规格形态 8 边 8 云:每前端独占一个云 DP,池=8x单云配额,
        # 每条连接在自己的配额内独立开合。
        conns = _edge_share_connections(8)
        route = {f"r{i}": conns[i] for i in range(8)}
        sched = _make_sched(
            conns, 2, running=[_request(f"r{i}") for i in range(8)], route=route
        )
        assert sched.max_num_running_reqs == 16
        # 每连接补到满配:全部 HOLD,但池恰好容纳 8x2。
        for i in range(8):
            sched.running.append(_request(f"s{i}"))
            sched._lwd_route[f"s{i}"] = conns[i]
        assert sched._lwd_pick_serving_engine() is None
        # 任一连接让出 1 个槽,只有它重开。
        sched.running.pop()
        assert sched._lwd_pick_serving_engine() == conns[7]
        assert sched._lwd_num_running_reqs(conns[3]) == 2


class TestSlotPoolSizing:
    """原生 running 上限 = 单云配额 x 服务的云 DP 数。"""

    def test_edge_share_pool_scales_with_cloud_dps(self) -> None:
        sched = _make_sched(_edge_share_connections(4), 74)
        assert sched.max_num_running_reqs == 4 * 74

    def test_single_cloud_pool_stays_one_grant(self) -> None:
        sched = _make_sched((LwdConnectionId(0, 0, 0),), 74)
        assert sched.max_num_running_reqs == 74

    def test_cloud_share_links_split_one_grant(self) -> None:
        # 3 个边缘链接同一个云 DP:配额只算一份,且占用在三者间
        # 互相可见(1118-1 形态:2 边 1 云)。
        conns = (
            LwdConnectionId(0, 0, 0),
            LwdConnectionId(1, 0, 0),
            LwdConnectionId(2, 0, 0),
        )
        sched = _make_sched(conns, 8)
        assert sched.max_num_running_reqs == 8
        route = {"r1": conns[0], "r2": conns[1]}
        sched.running = [_request(r) for r in route]
        sched._lwd_route = route
        assert sched._lwd_num_running_reqs(conns[2]) == 2
        assert (
            sched._lwd_accounts[conns[2]].is_gated(
                2, has_resume_inflight=False, num_running_reqs=2
            )
            == "slots"
        )

    def test_resize_pool_recomputes_from_connections(self) -> None:
        # 扩容属性:池总是从当前连接集合重算(本仓无弹性扩连端口,
        # 直接驱动 _lwd_resize_slot_pool 验证同一性质)。
        sched = _make_sched(_edge_share_connections(4), 8)
        assert sched.max_num_running_reqs == 32
        sched._lwd_connections = (
            *sched._lwd_connections,
            LwdConnectionId(4, 4, 0),
        )
        sched._lwd_resize_slot_pool()
        assert sched.max_num_running_reqs == 40
        # 重算幂等:连接集合不变时值不变。
        sched._lwd_resize_slot_pool()
        assert sched.max_num_running_reqs == 40


class TestPickerFreeSeq:
    """picker 的 free_seq 按云 DP 结算(新开名额数)。"""

    @staticmethod
    def _waiting_for(sched: NS, conn: LwdConnectionId, ids: list[str]) -> None:
        sched.waiting = [_request(rid, computed=0) for rid in ids]
        sched._lwd_route.update({rid: conn for rid in ids})

    def test_free_seq_ignores_other_clouds_occupancy(self) -> None:
        conns = _edge_share_connections(2)
        route = {"r1": conns[1], "r2": conns[1], "r3": conns[1]}
        sched = _make_sched(conns, 2, running=[_request(r) for r in route], route=route)
        self._waiting_for(sched, conns[0], ["w1", "w2"])
        req_ids, _, _ = sched._lwd_pick_prefill_batch(conns[0])
        # conn1 占 2 个槽不影响 conn0 的配额:两个新开都放行。
        assert req_ids == ["w1", "w2"]

    def test_free_seq_zero_blocks_new_starts_of_the_full_cloud(self) -> None:
        conns = _edge_share_connections(2)
        route = {"r1": conns[0], "r2": conns[0]}
        sched = _make_sched(conns, 2, running=[_request(r) for r in route], route=route)
        self._waiting_for(sched, conns[0], ["w1", "w2"])
        req_ids, _, _ = sched._lwd_pick_prefill_batch(conns[0])
        # 自己的云满了:一个新开都放不出去(全部留给在飞 chunk 的续跑)。
        assert req_ids == []

    def test_partial_occupancy_admits_the_remaining_quota(self) -> None:
        conns = _edge_share_connections(1)
        route = {"r1": conns[0]}
        sched = _make_sched(conns, 3, running=[_request(r) for r in route], route=route)
        self._waiting_for(sched, conns[0], ["w1", "w2", "w3", "w4"])
        req_ids, _, _ = sched._lwd_pick_prefill_batch(conns[0])
        assert req_ids == ["w1", "w2"]


class TestGateLogging:
    """HOLD(slots) 必须可见 -- 修复前槽位判据静默,4 前端全闸死时
    日志里一条 HOLD 都没有。"""

    def test_slots_hold_is_logged(self) -> None:
        conns = _edge_share_connections(2)
        route = {"r1": conns[0], "r2": conns[0]}
        sched = _make_sched(conns, 2, running=[_request(r) for r in route], route=route)
        sched._lwd_pick_serving_engine()
        holds = [ln for ln in sched.logger.lines if "HOLD(slots)" in ln]
        assert any(str(conns[0]) in ln for ln in holds)

    def test_chunks_hold_is_logged(self) -> None:
        conns = _edge_share_connections(1)
        # picker 只检查有已路由工作的连接:路由一个 waiting 请求。
        route = {"w1": conns[0]}
        sched = _make_sched(conns, 10, route=route)
        sched.waiting = [_request("w1", computed=0)]
        account = sched._lwd_accounts[conns[0]]
        for seqno in range(LWD_MAX_PENDING_CHUNKS):
            account.register_batch(seqno, {f"c{seqno}"})
        assert sched._lwd_pick_serving_engine() is None
        assert any("HOLD(chunks)" in ln for ln in sched.logger.lines)

    def test_release_logs_open_once(self) -> None:
        conns = _edge_share_connections(1)
        route = {"r1": conns[0], "r2": conns[0]}
        sched = _make_sched(conns, 2, running=[_request(r) for r in route], route=route)
        assert sched._lwd_pick_serving_engine() is None
        sched.running.clear()
        # 槽位释放后,一次放行恰好一条 OPEN,重复步进不再刷屏。
        assert sched._lwd_pick_serving_engine() is not None
        sched._lwd_pick_serving_engine()
        opens = [ln for ln in sched.logger.lines if "prefill gate OPEN" in ln]
        assert len(opens) == 1
