# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LWD 云侧 preflight KV 预估回归测试。

背景(实测:DSv4 32K/128K 压测 + ``max_num_batched_tokens=20480`` 挂死,改
8192 恢复,8K 短 prompt 正常):``_lwd_preflight_ok`` 的 KV 腿与
``allocate_slots`` 口径发散,把「本请求即将自己释放的块」算成缺口。

``allocate_slots`` 的次序是「先回收窗外块 → 再估增量 → 用回收后的空闲数
判定」(kv_cache_manager.py:407/411/421),而 preflight 跳过了回收那一步。
DSv4 的 compressor state 组 block_size 只有 8/32,每 chunk 毛需求
``chunk/8 + chunk/32``,续传时其中绝大部分会被 ``remove_skipped_blocks``
立刻回收(state 只需保留最近 ``coff × compress_ratio`` 个 token 的位置)。

叠加云侧禁抢占、``prefill_notify_queue`` 只看队首、hold 只打 debug,虚缺口
就变成不可自愈的循环等待:半途 prefill 的请求既跑不完也腾不出块,边侧
``_lwd_awaiting`` 涨满后闸门关闭并睡死,云侧却因 ``has_requests()`` 恒真而
空转打日志——「云侧忙、边侧不动、请求不报错也不返回」。

另两处同向偏差一并修掉:逐组估算漏除 ``compress_ratio``(Ascend
``CompressAttentionManager`` 会除);decode 项按「每请求每组 +1 块」的粗上界
计(真机实录:4 个 decode 粗算 24 块、精确约 0 块,正是这 24 块把 need 从
1424 顶到 1448、越过 free=1440 造成 9 次 hold,最长 28.4s)。现全部改为委托
``coordinator.get_num_blocks_to_allocate``(``allocate_slots`` 用的同一入口),
decode 项走两段式:粗上界先判(热路径零开销),不过再精算复核。
"""

from __future__ import annotations

import pytest

from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_mixed_scheduler import (
    LwdCloudMixedScheduler,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import LwdRangeItem

pytestmark = pytest.mark.cpu_test

REQ = "edge0-req0"
CHUNK = 20480
NUM_GROUPS = 6

# DSv4 逐组真实增量(chunk=20480):主 MLA C4 / indexer 各 cdiv(20480//4,128)
# =40,C128 cdiv(20480//128,128)=2,SWA 160,c4 state(bs8)2560,
# c128 state(bs32)640。
DSV4_PER_GROUP = [40, 40, 2, 160, 2560, 640]
DSV4_TOTAL = sum(DSV4_PER_GROUP)  # 3442
# 续传 chunk 会被 remove_skipped_blocks 立刻回收的部分:两个 state 组的
# 窗外块(2560-1 与 640-4),即虚缺口的来源。
DSV4_RECYCLED = (2560 - 1) + (640 - 4)  # 3195

# 真机现场(threshold=8192):逐组 16/0/64/64/1024/256 = 1424,4 个 decode
# 的粗上界 4×6 = 24 ⇒ need 1448,而 free = 1440 ⇒ 差 8 块被 hold。
PROD_PER_GROUP = [16, 0, 64, 64, 1024, 256]
PROD_PREFILL = sum(PROD_PER_GROUP)  # 1424
PROD_FREE = 1440
PROD_DECODE = 4


def _decode_ids(n: int) -> list[str]:
    return [f"d{i}" for i in range(n)]


class _Pool:
    """块池替身:回收会真实增加空闲数,并记录读取时点(验次序)。"""

    def __init__(self, free_blocks: int) -> None:
        self._free = free_blocks
        self.reads: list[int] = []
        self.recycle_log: list[str] = []

    def get_num_free_blocks(self) -> int:
        self.reads.append(len(self.recycle_log))
        return self._free


class _StubManager:
    """替身单组 manager:签名与 SingleTypeKVCacheManager 一致。"""

    def __init__(self, blocks: int) -> None:
        self._blocks = blocks

    def get_num_blocks_to_allocate(
        self, request_id, num_tokens, new_computed_blocks,
        total_computed_tokens, num_tokens_main_model,
        apply_admission_cap=False,
    ) -> int:
        return self._blocks


class _StubCoordinator:
    """替身 coordinator。

    ``overrides`` 按 request_id 给出该请求的块需求(缺省用 per_group 之和);
    ``recycle_overrides`` 同理给出每次回收释放的块数。
    """

    def __init__(
        self,
        pool: _Pool,
        per_group: list[int],
        recycle: int = 0,
        overrides: dict[str, int] | None = None,
        recycle_overrides: dict[str, int] | None = None,
    ) -> None:
        self.pool = pool
        self.recycle = recycle
        self.overrides = overrides or {}
        self.recycle_overrides = recycle_overrides or {}
        self.single_type_managers = tuple(
            _StubManager(blocks) for blocks in per_group
        )
        self._total = sum(per_group)
        self.calls: list[dict] = []
        self.recycle_calls: list[tuple[str, int]] = []

    def get_num_blocks_to_allocate(self, **kwargs) -> int:
        self.calls.append(kwargs)
        return self.overrides.get(kwargs["request_id"], self._total)

    def remove_skipped_blocks(self, request_id, total_computed_tokens) -> None:
        self.recycle_calls.append((request_id, total_computed_tokens))
        self.pool.recycle_log.append(request_id)
        self.pool._free += self.recycle_overrides.get(request_id, self.recycle)


class _Spec:
    def __init__(self, block_size: int) -> None:
        self.block_size = block_size


class _Req:
    """队列成员替身:真实 Request 同时带 request_id 与 num_computed_tokens
    (第三段回收要读后者)。"""

    def __init__(self, request_id: str, num_computed_tokens: int = 0) -> None:
        self.request_id = request_id
        self.num_computed_tokens = num_computed_tokens


class _ReqState:
    def __init__(self, num_computed_tokens: int) -> None:
        self.num_computed_tokens = num_computed_tokens


class _KvMgr:
    def __init__(self, coordinator: _StubCoordinator) -> None:
        self.coordinator = coordinator
        self.block_pool = coordinator.pool


class _FakeScheduler:
    """只装配 ``_lwd_preflight_ok`` 真正读到的字段。

    ``residents`` 按队列名给出额外的在飞请求 ``(request_id, num_computed)``,
    用来复现「常驻请求扛着窗外垃圾、而它本步不是候选」的现场。
    """

    _lwd_preflight_ok = LwdCloudMixedScheduler._lwd_preflight_ok
    _lwd_decode_blocks_exact = LwdCloudMixedScheduler._lwd_decode_blocks_exact
    _lwd_recycle_resident_blocks = (
        LwdCloudMixedScheduler._lwd_recycle_resident_blocks
    )

    def __init__(
        self,
        coordinator: _StubCoordinator,
        *,
        max_model_len: int = 131072,
        budget: int = CHUNK,
        num_lookahead_tokens: int = 0,
        spec_factor: int = 1,
        is_continuation: bool = False,
        computed: int = 0,
        decode_ids: list[str] | None = None,
        decode_computed: int = 32768,
        residents: dict[str, list[tuple[str, int]]] | None = None,
    ) -> None:
        self.max_num_scheduled_tokens = budget
        self.max_num_running_reqs = 128
        self.max_model_len = max_model_len
        self.num_lookahead_tokens = num_lookahead_tokens
        self._lwd_spec_factor = spec_factor
        # 组 spec 带 block_size 但不带 compress_ratio:与旧代码面对的
        # AscendSlidingWindowMLASpec 同形。
        self._lwd_group_specs = [_Spec(128) for _ in range(NUM_GROUPS)]
        res = residents or {}
        queues = {
            name: list(res.get(name, ()))
            for name in ("running", "waiting", "skipped_waiting")
        }
        if is_continuation:
            queues["running"].insert(0, (REQ, computed))
        self.requests = {REQ: _ReqState(computed)}
        for members in queues.values():
            for rid, comp in members:
                self.requests[rid] = _ReqState(comp)
        self.running = [_Req(rid, comp) for rid, comp in queues["running"]]
        self.waiting = [_Req(rid, comp) for rid, comp in queues["waiting"]]
        self.skipped_waiting = [
            _Req(rid, comp) for rid, comp in queues["skipped_waiting"]
        ]
        self._decode_ids = list(decode_ids or [])
        for rid in self._decode_ids:
            self.requests[rid] = _ReqState(decode_computed)
        self.kv_cache_manager = _KvMgr(coordinator)


_COORD_KWARGS = ("overrides", "recycle_overrides")


def _make(
    per_group: list[int] | None = None,
    free_blocks: int = 0,
    recycle: int = 0,
    decode: int = 0,
    **kwargs,
) -> _FakeScheduler:
    pool = _Pool(free_blocks)
    coord = _StubCoordinator(
        pool,
        per_group if per_group is not None else [0] * NUM_GROUPS,
        recycle,
        **{k: v for k, v in kwargs.items() if k in _COORD_KWARGS},
    )
    rest = {k: v for k, v in kwargs.items() if k not in _COORD_KWARGS}
    return _FakeScheduler(coord, decode_ids=_decode_ids(decode), **rest)


def _item(num_tokens: int = CHUNK) -> LwdRangeItem:
    return LwdRangeItem(request_id=REQ, offset=0, num_tokens=num_tokens)


# --------------------------------------------------------------------- #
# 回收信用(首块 vs 续传)                                                 #
# --------------------------------------------------------------------- #
def test_continuation_chunk_passes_once_recycled_blocks_are_credited() -> None:
    """回归主案:续传 chunk 的真实净增极小,回收后必须放行。

    DSv4 chunk=20480:逐组毛需求合计 3442,其中 3195 是 state 组的窗外块,
    会被 remove_skipped_blocks 立刻回收。空闲只有 300 时:
      - 不信用回收 ⇒ 3442 > 300 ⇒ 永久 hold(旧行为,挂死)
      - 信用回收   ⇒ 3442 ≤ 300 + 3195 = 3495 ⇒ 放行
    """
    sched = _make(
        per_group=DSV4_PER_GROUP,
        free_blocks=300,
        recycle=DSV4_RECYCLED,
        is_continuation=True,
        computed=20480,
    )

    assert DSV4_TOTAL > 300  # 不信用回收必然 hold
    assert sched._lwd_preflight_ok([_item()], []) is True


def test_recycle_runs_for_every_item_before_free_blocks_is_read() -> None:
    """次序契约:回收必须发生在读空闲块数之前(对齐 allocate_slots:407/421)。"""
    sched = _make(
        per_group=DSV4_PER_GROUP,
        free_blocks=300,
        recycle=DSV4_RECYCLED,
        is_continuation=True,
        computed=20480,
    )
    coord = sched.kv_cache_manager.coordinator
    pool = coord.pool

    assert sched._lwd_preflight_ok([_item()], []) is True

    assert coord.recycle_calls == [(REQ, 20480)]
    # 读空闲块数时,回收已经全部发生
    assert pool.reads and min(pool.reads) >= len(coord.recycle_calls)


def test_recycle_uses_pre_chunk_computed_position() -> None:
    """回收量按 chunk *之前*的位置算——与 allocate_slots 传的
    ``total_computed_tokens`` 同口径,且过 max_model_len 截断。"""
    sched = _make(
        per_group=[0] * NUM_GROUPS,
        free_blocks=10,
        is_continuation=True,
        computed=40960,
        max_model_len=32768,
    )
    coord = sched.kv_cache_manager.coordinator

    sched._lwd_preflight_ok([_item(num_tokens=1000)], [])

    assert coord.recycle_calls == [(REQ, 32768)]


def test_fresh_request_without_recycle_still_gated() -> None:
    """首块无可回收:闸门必须照常拦住真实缺口,不能被这次修复拆掉。"""
    sched = _make(per_group=DSV4_PER_GROUP, free_blocks=300, recycle=0)

    assert sched._lwd_preflight_ok([_item()], []) is False


# --------------------------------------------------------------------- #
# decode 项:粗上界 + 精算复核                                             #
# --------------------------------------------------------------------- #
def test_production_hold_cleared_by_exact_decode_recheck() -> None:
    """真机现场复现:粗上界差 8 块被 hold,精算后放行。

    逐组 16/0/64/64/1024/256 = 1424,4 个 decode 粗算 4×6 = 24 ⇒ 1448,
    free = 1440 ⇒ 差 8。decode 每步只加 1+k token,精确需求 0 ⇒
    1424 ≤ 1440 ⇒ 放行。这正是线上那 9 次 hold(最长 28.4s)的成因。
    """
    ids = _decode_ids(PROD_DECODE)
    sched = _make(
        per_group=PROD_PER_GROUP,
        free_blocks=PROD_FREE,
        decode=PROD_DECODE,
        budget=CHUNK,
        overrides={rid: 0 for rid in ids},
    )
    coord = sched.kv_cache_manager.coordinator

    assert PROD_PREFILL + PROD_DECODE * NUM_GROUPS > PROD_FREE  # 粗判必不过
    assert sched._lwd_preflight_ok([_item(num_tokens=8192)], ids) is True
    # 精算确实被触发:prefill 项之后逐 decode 请求问过分配器
    assert [c["request_id"] for c in coord.calls] == [REQ] + ids


def test_coarse_bound_short_circuits_when_it_already_passes() -> None:
    """粗上界够用就不精算——热路径零额外开销(perf 分支)。"""
    sched = _make(
        per_group=[40] + [0] * (NUM_GROUPS - 1),
        free_blocks=60,
        budget=CHUNK + 64,
        decode=3,
    )
    coord = sched.kv_cache_manager.coordinator

    # 40 + 3×6 = 58 ≤ 60 ⇒ 粗判直接放行,不进精算
    assert sched._lwd_preflight_ok([_item()], _decode_ids(3)) is True
    # 只有 prefill 项被问过/被回收过,decode 请求一个都没碰
    assert [c["request_id"] for c in coord.calls] == [REQ]
    assert [rid for rid, _ in coord.recycle_calls] == [REQ]


def test_exact_recheck_still_holds_when_really_short() -> None:
    """精算只去掉高估,不能把闸门拆掉:真实缺口仍须 hold。"""
    ids = _decode_ids(PROD_DECODE)
    sched = _make(
        per_group=PROD_PER_GROUP,
        free_blocks=PROD_FREE,
        decode=PROD_DECODE,
        # 每个 decode 请求精确需 200 块 ⇒ 1424 + 800 = 2224 > 1440
        overrides={rid: 200 for rid in ids},
    )

    assert sched._lwd_preflight_ok([_item(num_tokens=8192)], ids) is False


def test_exact_recheck_credits_decode_recycle_and_rereads_free() -> None:
    """精算路径也要「先回收再问」,且回收后必须重读空闲数。

    item 用 8192 而非 CHUNK:chunk == budget 时 token 腿会先行 hold
    (20480 + num_decode > 20480),测不到 KV 腿——这本身就是那条
    「预算零余量」残留风险的现场演示。
    """
    ids = _decode_ids(2)
    sched = _make(
        per_group=[1400] + [0] * (NUM_GROUPS - 1),
        free_blocks=100,  # 粗判:1400 + 2×6 = 1412 > 100 ⇒ 进精算
        decode=2,
        overrides={rid: 0 for rid in ids},
        # 两个 decode 请求各回收 700 ⇒ free 变 100 + 1400 = 1500 ≥ 1400
        recycle_overrides={rid: 700 for rid in ids},
    )
    coord = sched.kv_cache_manager.coordinator

    assert sched._lwd_preflight_ok([_item(num_tokens=8192)], ids) is True
    assert [rid for rid, _ in coord.recycle_calls] == [REQ] + ids
    # 最后一次读空闲发生在全部回收之后(含 decode 请求自己那次)
    assert max(coord.pool.reads) == len(coord.recycle_calls)


def test_decode_exact_uses_spec_factor_as_num_new() -> None:
    """精算的 num_new 取 1+k(spec_factor),并同样叠加 lookahead。"""
    ids = _decode_ids(1)
    sched = _make(
        per_group=[10**9] + [0] * (NUM_GROUPS - 1),  # 强制粗判不过
        free_blocks=0,
        decode=1,
        spec_factor=2,
        num_lookahead_tokens=3,
        decode_computed=1000,
        overrides={ids[0]: 0},
    )
    coord = sched.kv_cache_manager.coordinator

    # 同上:item 取 8192,避免 token 腿(20480 + 1×2 > 20480)先行 hold
    sched._lwd_preflight_ok([_item(num_tokens=8192)], ids)

    call = next(c for c in coord.calls if c["request_id"] == "d0")
    assert call["num_tokens_main_model"] == 1000 + 2
    assert call["num_tokens"] == 1000 + 2 + 3
    assert call["apply_admission_cap"] is False


# --------------------------------------------------------------------- #
# 常驻请求的窗外块回收(第三段)                                            #
# --------------------------------------------------------------------- #
# 真机死锁现场(chunk=20352):候选首块毛需求 3544,free 只有 2179;而在飞的
# A 正扛着上一个 chunk 留在 state 组的 ~3176 块窗外垃圾——惰性回收下没人替
# 它收,因为唯一的触发点是它自己的下一次 allocate_slots,而那需要它的续传
# notify 被消费,而它压在队首后面。
DEADLOCK_PER_GROUP = [40, 2, 160, 160, 2545, 637]
DEADLOCK_NEED = sum(DEADLOCK_PER_GROUP)  # 3544
DEADLOCK_FREE = 2179
DEADLOCK_RESIDENT_GARBAGE = (2545 - 2) + (637 - 4)  # 3176
RESIDENT = "edge0-A"


def test_resident_garbage_recycled_breaks_the_deadlock() -> None:
    """回收常驻请求的窗外块后,原本进不来的首块必须放行。

    对应真机日志:need_blocks=3544 > free=2179、running=1、decode_ready=0、
    pending_notify=25、stuck=275s。第三段回收 A 的 ~3176 块 ⇒ free≈5355 ⇒
    3544 ≤ 5355 ⇒ 放行,循环等待被打断。
    """
    sched = _make(
        per_group=DEADLOCK_PER_GROUP,
        free_blocks=DEADLOCK_FREE,
        residents={"running": [(RESIDENT, 20352)]},
        recycle_overrides={RESIDENT: DEADLOCK_RESIDENT_GARBAGE},
    )
    coord = sched.kv_cache_manager.coordinator

    assert DEADLOCK_NEED > DEADLOCK_FREE  # 不回收常驻则必然 hold
    assert sched._lwd_preflight_ok([_item(num_tokens=20352)], []) is True
    assert (RESIDENT, 20352) in coord.recycle_calls


def test_resident_recycle_not_triggered_on_hot_path() -> None:
    """粗判够用就不碰常驻请求——热路径零额外开销(perf 分支)。"""
    sched = _make(
        per_group=[40] + [0] * (NUM_GROUPS - 1),
        free_blocks=1000,
        residents={"running": [(RESIDENT, 20352)]},
        recycle_overrides={RESIDENT: 500},
    )
    coord = sched.kv_cache_manager.coordinator

    assert sched._lwd_preflight_ok([_item(num_tokens=8192)], []) is True
    # 只有候选项被回收过,常驻请求一次都没碰
    assert [rid for rid, _ in coord.recycle_calls] == [REQ]


def test_resident_recycle_covers_all_three_queues() -> None:
    """三个队列都要扫:被跳过/回插的请求同样可能扛着块。"""
    sched = _make(
        per_group=[1000] + [0] * (NUM_GROUPS - 1),
        free_blocks=100,
        residents={
            "running": [("rA", 8192)],
            "waiting": [("wB", 4096)],
            "skipped_waiting": [("sC", 2048)],
        },
    )
    coord = sched.kv_cache_manager.coordinator

    sched._lwd_preflight_ok([_item(num_tokens=8192)], [])

    assert {rid for rid, _ in coord.recycle_calls} == {REQ, "rA", "wB", "sC"}
    # 回收位置按各请求自己的 num_computed_tokens,不是候选项的
    assert dict(coord.recycle_calls)["wB"] == 4096


def test_resident_recycle_does_not_replace_a_real_shortfall() -> None:
    """常驻回收只归还垃圾,不能凭空造出块:真实缺口仍须 hold。"""
    sched = _make(
        per_group=DEADLOCK_PER_GROUP,
        free_blocks=DEADLOCK_FREE,
        residents={"running": [(RESIDENT, 20352)]},
        recycle_overrides={RESIDENT: 100},  # 只回收 100,远不够
    )

    assert sched._lwd_preflight_ok([_item(num_tokens=20352)], []) is False


# --------------------------------------------------------------------- #
# 委托契约                                                               #
# --------------------------------------------------------------------- #
def test_preflight_delegates_with_allocate_slots_arguments() -> None:
    """逐条 item 问一次真实分配器,参数与 allocate_slots 对齐。"""
    sched = _make(
        per_group=[40] + [0] * (NUM_GROUPS - 1),
        free_blocks=1000,
        is_continuation=True,
        computed=20480,
        num_lookahead_tokens=3,
    )
    coord = sched.kv_cache_manager.coordinator

    assert sched._lwd_preflight_ok([_item(num_tokens=12288)], []) is True

    assert len(coord.calls) == 1
    call = coord.calls[0]
    assert call["request_id"] == REQ
    # num_tokens = num_tokens_need_slot = computed + chunk + lookahead
    assert call["num_tokens_main_model"] == 20480 + 12288
    assert call["num_tokens"] == 20480 + 12288 + 3
    assert call["total_computed_tokens"] == 20480
    # mixed 步 scheduler_reserve_full_isl 恒 False ⇒ 走 per-step 口径;
    # 套上整序列准入门会按 max_model_len 预留,造成假性 hold
    assert call["apply_admission_cap"] is False
    # 前缀命中量此刻未知,按每组空命中传入(高估是安全方向)
    assert len(call["new_computed_blocks"]) == NUM_GROUPS
    assert all(len(hits) == 0 for hits in call["new_computed_blocks"])


def test_num_tokens_need_slot_clamped_to_max_model_len() -> None:
    """逼近 max_model_len 的尾块:need_slot 截断,main_model 不截断。"""
    sched = _make(
        per_group=[40] + [0] * (NUM_GROUPS - 1),
        free_blocks=1000,
        max_model_len=32768,
        is_continuation=True,
        computed=32000,
        num_lookahead_tokens=3,
    )
    coord = sched.kv_cache_manager.coordinator

    assert sched._lwd_preflight_ok([_item(num_tokens=1000)], []) is True

    call = coord.calls[0]
    assert call["num_tokens_main_model"] == 33000
    assert call["num_tokens"] == 32768
