# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""边侧 cap 的静态最坏口径契约(lwd_edge_scheduler._lwd_pick_prefill_batch)。

钉住 `f2406aa73e`「cap 改静态最坏口径」这条不变式——它此前没有测试。

根因(该提交原文):chunk 是边侧切好的原子单元,云侧不可再切。原 cap 用
「切块时点的在途数」预留 decode 空间,首 chunk 在途=0 时切出**满预算**的块;
到云侧执行恰逢 decode 活动,`chunk + decode` 顶爆 runner 按
`max_num_batched_tokens` 开死的输入缓冲(`np.add` 崩溃);而云侧 preflight
若以 hold 处理则 chunk 永进不来(prefill 饿死)。

静态预留后 `chunk + decode ≤ 预算` 恒成立。真机验证(2E1C、预算 20480、
max_num_seqs 256、spec_factor 2):cap = 20480 − 256×2 = 19968,chunk 序列
19968/12672/132,`hold(tokens)` 归零。

关键性质是 **cap 与在途数无关**:在途数是切块时点的量,预测不了云侧执行
时点的 decode 人口,多边下更看不到对边——所以必须按满员静态预留。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdEdgeScheduler,
)

pytestmark = pytest.mark.cpu_test

# 真机部署值(2E1C、DSv4 + MTP num_speculative_tokens=1)
BUDGET = 20480
MAX_NUM_SEQS = 256
SPEC_FACTOR = 2
CAP = BUDGET - MAX_NUM_SEQS * SPEC_FACTOR  # 19968
PROMPT = 32772
CHUNK_UNIT = 20480
REQ = "edge0-req0"


class _Request:
    def __init__(self, request_id: str, num_prompt_tokens: int) -> None:
        self.request_id = request_id
        self.num_prompt_tokens = num_prompt_tokens
        self.num_computed_tokens = 0


class _FakeEdgeScheduler:
    """只装配 `_lwd_pick_prefill_batch` 真正读到的字段。

    `_lwd_seq_quota` 借用真类方法;`_lwd_n_edges` 显式给定,故不查 registry。
    `_mamba_block_aligned_split` 返回原值 = 网格未学到(block_size<=0)时的
    真实行为,使断言落在 cap 本身而非网格收边上。
    """

    _lwd_pick_prefill_batch = LwdEdgeScheduler._lwd_pick_prefill_batch
    _lwd_seq_quota = LwdEdgeScheduler._lwd_seq_quota

    def __init__(
        self,
        *,
        inflight: int = 0,
        n_edges: int = 1,
        budget: int = BUDGET,
        max_num_seqs: int = MAX_NUM_SEQS,
        spec_factor: int = SPEC_FACTOR,
        chunk_unit: int = CHUNK_UNIT,
        waiting: tuple[int, ...] = (),
        model_type: str = "deepseek_v4",
    ) -> None:
        self._lwd_mixed = True
        self._lwd_n_edges = n_edges
        self.max_num_scheduled_tokens = budget
        self.max_num_running_reqs = max_num_seqs
        self._lwd_spec_factor = spec_factor
        self._lwd_chunk_unit = chunk_unit
        self.running: list = []
        self._lwd_awaiting = {f"a{i}": 0.0 for i in range(inflight)}
        self.waiting = [_Request(REQ, n) for n in waiting]
        self.skipped_waiting: list = []
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(model_type=model_type)
            )
        )

    def _mamba_block_aligned_split(self, request, num_new_tokens, *args):
        return num_new_tokens

    def _lwd_mamba_dep_conflict(self, request, chunk, picked, expected):
        return False


def _pick(sched):
    return sched._lwd_pick_prefill_batch()


# --------------------------------------------------------------------- #
# 静态最坏口径                                                           #
# --------------------------------------------------------------------- #
def test_cap_reserves_worst_case_cloud_decode() -> None:
    """cap = 预算 − 满员 × spec 系数,与构造期活性校验同源。"""
    sched = _FakeEdgeScheduler(inflight=0)

    assert _pick(sched)[2] == CAP  # 19968


def test_cap_is_independent_of_inflight() -> None:
    """**核心性质**:cap 不随在途数变化。

    旧公式 `预算 − 在途 × 系数` 在 inflight=0 时给出满预算,正是它切出
    顶爆云侧输入缓冲的满预算 chunk。静态口径下 inflight 只影响 free_seq
    (还能开几个),不影响 cap(每块多大)。
    """
    caps = {
        _pick(_FakeEdgeScheduler(inflight=n))[2] for n in (0, 1, 7, 64, 200)
    }

    assert caps == {CAP}


def test_chunk_never_fills_whole_budget() -> None:
    """云侧 runner 输入缓冲按 max_num_batched_tokens 开死,故 chunk 必须
    严格小于预算,给云侧 decode 留座位——否则 np.add 崩溃。

    用真机参数复现:32772 的 prompt、单元 20480 ⇒ 首块被 cap 截到 19968。
    """
    sched = _FakeEdgeScheduler(inflight=0, waiting=(PROMPT,))

    picked, expected, cap = _pick(sched)

    assert picked == [REQ]
    assert expected[REQ] == CAP  # 19968,不是 20480
    assert expected[REQ] < BUDGET
    assert cap < BUDGET


def test_token_leg_structurally_impossible() -> None:
    """不变式:cap + 满员 × 系数 == 预算,故云侧 token 腿恒不触发。

    云侧判 `items_only + num_decode × factor > 预算`,其中
    `items_only ≤ cap`、`num_decode ≤ max_num_running_reqs`。
    """
    for n_edges in (1, 2, 4):
        for inflight in (0, 1, 64):
            sched = _FakeEdgeScheduler(n_edges=n_edges, inflight=inflight)
            cap = _pick(sched)[2]
            assert cap + MAX_NUM_SEQS * SPEC_FACTOR <= BUDGET


def test_cap_stays_positive_under_the_liveness_bound() -> None:
    """启动期活性检查保证 `预算 − 满员 × 系数 ≥ 1`,故 cap 恒 ≥ 1。

    否则静态预留会引入新的死锁配置(cap 恒 < 1 ⇒ 空批 ⇒ 续传永不推进),
    正是 __init__ 与 HELLO 采纳后三处 fail-fast 要防的。
    """
    sched = _FakeEdgeScheduler(inflight=0)

    assert BUDGET - MAX_NUM_SEQS * SPEC_FACTOR >= 1  # 前置条件
    assert _pick(sched)[2] >= 1


# --------------------------------------------------------------------- #
# free_seq 仍按边数均分(与 cap 正交:cap 管 token,free_seq 管名额)      #
# --------------------------------------------------------------------- #
def test_free_seq_still_divides_by_edge_count() -> None:
    """多边共享一台云 ⇒ 每边名额 = 云名额 ÷ 边数,避免结构性超订。"""
    one = _FakeEdgeScheduler(n_edges=1, waiting=(PROMPT,))
    two = _FakeEdgeScheduler(n_edges=2, waiting=(PROMPT,))

    assert one._lwd_seq_quota() == MAX_NUM_SEQS
    assert two._lwd_seq_quota() == MAX_NUM_SEQS // 2
    # 名额减半 ⇒ 本边可开的新请求数减半,但 cap(chunk 尺寸)不变
    assert _pick(one)[2] == _pick(two)[2] == CAP


def test_free_seq_shrinks_with_inflight() -> None:
    """cap 与在途无关,但 free_seq 必须随在途递减。"""
    idle = _FakeEdgeScheduler(n_edges=1, inflight=0, waiting=(PROMPT,))
    busy = _FakeEdgeScheduler(n_edges=1, inflight=100, waiting=(PROMPT,))

    assert idle._lwd_seq_quota() - 0 == MAX_NUM_SEQS
    assert busy._lwd_seq_quota() - 100 == MAX_NUM_SEQS - 100
    assert _pick(idle)[2] == _pick(busy)[2]


def test_seq_quota_falls_back_when_registry_unknown() -> None:
    """registry 缺失时按单边处理(名额不均分),不炸也不误判。"""
    sched = _FakeEdgeScheduler(n_edges=2)
    sched._lwd_n_edges = None  # 模拟 registry 未解析出边数

    assert sched._lwd_seq_quota() == MAX_NUM_SEQS
