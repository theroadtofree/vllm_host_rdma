"""云侧混合调度器:prefill/decode 同批下发(设计:lwd_mixed_batch_design.md
+ lwd_chunked_prefill_design.md)。

chunk 混排语义:边侧按 c = min(剩余 prompt, chunk 单元) 切块组批
(续传优先 + FCFS 前缀),云侧每步至多消费 prefill_notify_queue 队首
一条批量 RangeNotify,与全部 decode 态请求混排一个批下发:

  - preflight 三检(预算/名额/KV 保守预估)任一不过 -> 本步 decode-only,
    预告留队首原位,decode 泄压后下步重试;
  - 放行先做 offset 对账(item.offset == 云侧 num_computed,chunk
    断流/重复/乱序即 fail-fast),再按 T = max(items) 临时覆写
    long_prefill_token_threshold——原生调度的 threshold 截断逐请求
    复现边侧 chunk 边界(自限 chunk,设计 §2.3 证明与处理顺序无关);
  - 步后核验批内每请求 num_scheduled == 预告量(chunk 精确准入),
    不等即 fail-fast(一次 UP recv 对应整批张量,错排会留残余行错配)。

前置约束:spec decode(MTP)预算已按 1+k 系数对齐(边 cap 与本类
preflight),但端到端未经真机验证(eagle 纯度判据失真风险,台账
P3-3/设计 M-8);
与前缀复用的关系(D1 定稿):边侧 probe 命中后按命中长度 trim,
首 chunk 的 offset 即命中长度(绝对量),云侧 num_computed 以同一
命中起点初始化(resume 封顶 prefix_hit_tokens),故
item.offset == num_computed 的绝对量对账在命中路径下天然成立,
不放宽、不互斥;不兼容云侧抢占重算(占位 embeds 无法本地重
prefill)——部署须保证 KV 充足,本类的核验负责把违背变成当场
报错而非静默错算。
"""

from __future__ import annotations

import time
from collections import deque

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import (
    LwdBatch,
    LwdBatchType,
    LwdEmbedBatch,
    SchedulerOutput,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LwdRangeItem,
    LwdRangeNotify,
)
from vllm.v1.lwd_control.control_communication.lwd_id_adapter import (
    parse_edge_id,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
    aligned_chunk_length,
)

logger = init_logger(__name__)


class LwdCloudMixedScheduler(LwdBaseScheduler):
    """notify 门控的 prefill/decode 混排调度器(相位机的继任者)。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # prefill 通知队列:边侧批量范围预告(RangeNotify)逐条入队,每步
        # 至多消费队首一条;预告自带 seqno 即本批 UP 链配对号
        self.prefill_notify_queue: deque[LwdRangeNotify] = deque()
        # [Lwd][sched] 调度批日志步计数(饿死分析:RangeNotify 到达 ->
        # 混排步消费的间隔与中间插入的 decode-only 步数)
        self._lwd_sched_step = 0
        # MTP 预算系数:decode 每请求每步消耗 1+k 个 token(草稿 token
        # 占预算与 KV);k = num_speculative_tokens(配置解析期已求值,
        # 缺省 None 按 0),无 spec 配置即 1
        spec_config = getattr(self.vllm_config, "speculative_config", None)
        num_spec = (
            getattr(spec_config, "num_speculative_tokens", None) or 0
        )
        self._lwd_spec_factor: int = 1 + num_spec
        logger.info(
            "[Lwd] cloud mixed scheduler: chunked-prefill batches mixed "
            "with decode (one notify consumed per step at most, "
            "spec_factor=%d)",
            self._lwd_spec_factor,
        )
        # 生效预算探针(preflight hold 排查用):max_num_scheduled_tokens
        # 是调度器真正读的值(config 显式设置则覆盖 batched_tokens);
        # 两者不一致即配置/派生错位,启动期一行可见
        logger.info(
            "[Lwd] cloud mixed scheduler budgets: "
            "max_num_scheduled_tokens=%d max_num_batched_tokens=%d "
            "max_num_seqs=%d",
            self.max_num_scheduled_tokens,
            self.scheduler_config.max_num_batched_tokens,
            self.max_num_running_reqs,
        )
        # spec 接受率累计器(引擎进程内直出):LWD 云前端没有注册请求,
        # 前端 stats 链(output_processor 丢弃未知请求输出 + lifespan
        # 周期任务)在云侧不可靠;接受率数据的源头本来就在调度器里,
        # 此处按原生口径(scheduler.py:1414-1424)累计并周期直出,
        # 与集中式的 Prometheus 计数器/前端日志同公式可直接对比
        self._lwd_spec_drafts = 0
        self._lwd_spec_draft_tokens = 0
        self._lwd_spec_accepted = 0
        self._lwd_spec_last_log = time.monotonic()
        # 已完结请求 LRU 台账(classify 用):区分"ADD 尚未落地"
        # (unknown,整批 defer)与"已完结释放"(迟到/重复预告,剔除
        # 防堵队首)。有界 LRU,超容量逐出最旧(逐出后迟到的预告退化为
        # unknown,只会 defer 一拍,待 abort 清扫兜底,不会永久堵)。
        from collections import OrderedDict

        self._lwd_recent_finished: OrderedDict[str, None] = OrderedDict()
        # 当次消费的幽灵条目(classify 写入,建批/释放读取)
        self._lwd_cur_ghost_items: list[LwdRangeItem] = []
        # KV preflight 的分组块粒度:混合模型块池跨组共享,占用须逐组
        # 对齐 allocate_slots 的真实口径——attention 组按 cdiv(chunk +
        # lookahead, bs)(lookahead 进 num_tokens_need_slot 可顶破块
        # 边界);mamba align 组首准入恒 1+num_speculative_blocks(最新
        # 状态块+草稿快照,与 chunk 长度无关),续传至多 +1,lookahead
        # 豁免。用 self.block_size(=scheduler_block_size,混合下为组块
        # LCM 网格)按单粒度估会少算 ~组数倍,preflight 失守放进批 →
        # allocate_slots 失败 → 步后核验以 scheduled=0 炸(实测
        # seqno=195)
        self._lwd_group_specs = [
            group.kv_cache_spec
            for group in self.kv_cache_manager.kv_cache_config.kv_cache_groups
        ]

    def update_from_output(self, scheduler_output, model_runner_output):
        """先累计 spec 接受率(镜像原生 update_from_output 的统计分支),
        再走原生输出处理。"""
        self._lwd_observe_spec_acceptance(scheduler_output, model_runner_output)
        return super().update_from_output(scheduler_output, model_runner_output)

    def finish_requests(self, request_ids, finished_status):
        """原生终结钩子:登记完结台账(classify 区分迟到预告)后透传。

        ``request_ids=None`` 是原生的「终结全部」语义(关停路径
        ``EngineCore._handle_shutdown`` 就传 None,见 scheduler.py:1850),
        必须先展开再遍历:直接 for 在 None 上会抛 TypeError,使关停半途
        fatal、EngineCore 永不退出,父进程卡在等 worker 收尾。"""
        if request_ids is None:
            ids = list(self.requests)
        elif isinstance(request_ids, str):
            ids = [request_ids]
        else:
            ids = request_ids
        for rid in ids:
            self._lwd_note_finished(rid)
        return super().finish_requests(request_ids, finished_status)

    def _free_request(self, request, delay_free_blocks: bool = False):
        """完结台账登记补挂**释放链唯一收口点**。

        finish_requests 覆写只覆盖 abort/error 路径;正常终结
        (update_from_output 里 check_stop 命中)直接调本方法、不经
        finish_requests——漏登会让该请求事后到达的重复预告被 classify
        误判 unknown(ADD 未落地)而整批 defer,请求永不落地即队首
        永久堵塞。与 phase 臂同型(那边只挂本方法);abort 路径两处
        登记幂等(set 语义,重复写无害)。登记失败不阻断释放。"""
        try:
            self._lwd_note_finished(request.request_id)
        except Exception:  # noqa: BLE001
            logger.warning(
                "[Lwd][cloud-sched] note-finished failed (non-fatal)",
                exc_info=True,
            )
        return super()._free_request(request, delay_free_blocks)

    _LWD_SPEC_LOG_INTERVAL_S = 10.0

    def _lwd_observe_spec_acceptance(
        self, scheduler_output, model_runner_output
    ) -> None:
        """逐请求累计草稿/接受数并周期打运行总计。

        口径与原生完全一致:num_draft_tokens = len(scheduled_spec_token_ids),
        num_accepted = len(generated_token_ids) - 1(含 bonus,减一剔除);
        跳过已终结/未知请求(与原生 :1403-1412 同款过滤)。"""
        spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        if not spec_tokens:
            return
        sampled = model_runner_output.sampled_token_ids
        req_id_to_index = model_runner_output.req_id_to_index
        for req_id, spec_ids in spec_tokens.items():
            if not spec_ids:
                continue
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue
            req_index = req_id_to_index.get(req_id)
            if req_index is None:
                continue
            generated = sampled[req_index] if sampled else []
            if not generated:
                continue
            self._lwd_spec_draft_tokens += len(spec_ids)
            self._lwd_spec_accepted += len(generated) - 1
            self._lwd_spec_drafts += 1
        now = time.monotonic()
        if now - self._lwd_spec_last_log < self._LWD_SPEC_LOG_INTERVAL_S:
            return
        self._lwd_spec_last_log = now
        if not self._lwd_spec_draft_tokens:
            return
        rate = self._lwd_spec_accepted / self._lwd_spec_draft_tokens * 100
        mean_accept_len = 1 + self._lwd_spec_accepted / max(
            self._lwd_spec_drafts, 1
        )
        logger.info(
            "[Lwd][spec] cumulative: rate=%.1f%% mean_accept_len=%.2f "
            "accepted=%d draft_tokens=%d drafts=%d",
            rate, mean_accept_len, self._lwd_spec_accepted,
            self._lwd_spec_draft_tokens, self._lwd_spec_drafts,
        )

    @staticmethod
    def _lwd_is_decode(request) -> bool:
        """prompt 已算完 = decode 态(可采样)。chunk 化后 running 可含
        prefill 半途请求(num_computed 爬坡中)——它们不进 decode 集,
        留在隐藏集等自己的下一条 chunk 预告驱动续排。"""
        return request.num_computed_tokens >= request.num_prompt_tokens

    def _lwd_collect_decode_requests(self) -> list[str]:
        """收集所有 decode 态(prompt 已算完)请求的 req_id。

        按 running/waiting/skipped 顺序遍历三队列,输出可直接作为
        _lwd_schedule_for_visible_reqs 的入参。"""
        return [
            req.request_id
            for queue in (self.running, self.waiting, self.skipped_waiting)
            for req in queue
            if self._lwd_is_decode(req)
        ]

    # ------------------------------------------------------------------ #
    # notify 条目分类与 preflight                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _lwd_notify_items(notify: LwdRangeNotify) -> list[LwdRangeItem]:
        """取批条目:优先 items(mixed 组批);空则回退旧版单请求语义。"""
        if notify.items:
            return list(notify.items)
        return [
            LwdRangeItem(
                request_id=notify.request_id,
                offset=notify.offset,
                num_tokens=notify.num_tokens,
                # 旧版边(无 items):mrope 标记在顶层标量上,回退透传,
                # 与相位机逃生通道口径一致
                has_mrope=notify.has_mrope,
            )
        ]

    _LWD_RECENT_FINISHED_CAP = 4096

    def _lwd_classify_items(
        self, items: list[LwdRangeItem]
    ) -> tuple[list[LwdRangeItem], bool]:
        """条目四分:ready(待 prefill)/ ghost(已 abort,照收不注入,
        保 UP numel 配对)/ stale(已完结或重复预告,剔除)/ unknown
        (ADD 尚未落地)。返回 (ready, 是否有 unknown);幽灵记入
        self._lwd_cur_ghosts(仅本次消费有效,随批建完即清)。

        unknown 存在即整批 defer(all-or-nothing:一次 UP recv 对应整批
        张量,不能只放行子集);abort 经引擎幽灵台账(照收不注入),
        队首不会被永久堵住。"""
        ready: list[LwdRangeItem] = []
        unknown = False
        self._lwd_cur_ghost_items: list[LwdRangeItem] = []
        for item in items:
            if item.request_id in self.lwd_ghost_req_ids:
                # 已 abort 的在飞 chunk:行照收(保 numel 配对)不调度
                self._lwd_cur_ghost_items.append(item)
                continue
            request = self.requests.get(item.request_id)
            if request is None:
                if item.request_id in self._lwd_recent_finished:
                    # 已完结并释放:边侧队满重试同 seqno 的迟到预告,
                    # 剔除而非整批 defer(否则队首永久堵塞)
                    logger.warning(
                        "[Lwd][cloud-sched] drop stale notify item req=%s "
                        "num=%d (request already finished and released)",
                        item.request_id, item.num_tokens,
                    )
                    continue
                unknown = True
                continue
            if self._lwd_is_decode(request):
                logger.warning(
                    "[Lwd][cloud-sched] drop stale notify item req=%s "
                    "num=%d (already prefilled; duplicate notify?)",
                    item.request_id, item.num_tokens,
                )
                continue
            ready.append(item)
        return ready, unknown

    def _lwd_note_finished(self, request_id: str) -> None:
        """登记请求完结(LRU 有界):供 classify 区分迟到预告与未落地。"""
        self._lwd_recent_finished[request_id] = None
        if len(self._lwd_recent_finished) > self._LWD_RECENT_FINISHED_CAP:
            self._lwd_recent_finished.popitem(last=False)

    def _lwd_preflight_ok(
        self, items: list[LwdRangeItem], num_decode: int
    ) -> bool:
        """放行三检:token 预算 / 并发名额 / KV 保守预估。

        预算与名额本由边侧 cap 保证(设计 §2),此处是双保险;KV 预估
        按各 KV 组块粒度分别求和(混合模型块池跨组共享,单粒度估计会
        按 LCM 倍率少算),且忽略前缀缓存命中(命中只降占用,高估是
        安全方向)。decode 占用
        按 1+k 计(MTP 草稿 token 同样吃预算;KV 块预留同系数上取整)。"""
        need_tokens = (
            sum(item.num_tokens for item in items)
            + num_decode * self._lwd_spec_factor
        )
        items_only = sum(item.num_tokens for item in items)
        if items_only > self.max_num_scheduled_tokens:
            # 永久超限(单批条目合计 > 云预算):等多久 decode 泄压都
            # 不会过——边侧 cap 读的本地预算比云侧大(边云
            # max_num_batched_tokens 未对齐时只能靠这里拦),hold 会
            # 永久堵队首空转。当场报错。
            raise RuntimeError(
                f"[Lwd][cloud-sched] prefill batch can never fit: "
                f"items_total={items_only} > cloud budget "
                f"{self.max_num_scheduled_tokens} — edge/cloud "
                f"max_num_batched_tokens misaligned (edge caps by its "
                f"local budget); align the two sides' "
                f"max_num_batched_tokens"
            )
        if need_tokens > self.max_num_scheduled_tokens:
            logger.debug(
                "[Lwd][cloud-sched] preflight hold: tokens %d > budget %d",
                need_tokens, self.max_num_scheduled_tokens,
            )
            return False
        # 名额只数新增准入:续传 chunk 的请求已在 running,再按条目数
        # 计会双重占用名额造成假性 hold(chunk 化前批内恒为新增,
        # len(items) 与新增数相等)
        running_ids = {req.request_id for req in self.running}
        new_items = sum(
            1 for item in items if item.request_id not in running_ids
        )
        if len(self.running) + new_items > self.max_num_running_reqs:
            logger.debug(
                "[Lwd][cloud-sched] preflight hold: seqs %d+%d > max %d",
                len(self.running), new_items, self.max_num_running_reqs,
            )
            return False
        # 逐组对齐 allocate_slots 真实口径(口径依据见 __init__ 注释);
        # resume 命中块只 touch 不占空闲池,忽略之是保守方向
        spec_tokens = self._lwd_spec_factor - 1  # lookahead(MTP 草稿)占块
        need_blocks = 0
        for item in items:
            fresh = item.request_id not in running_ids
            for spec in self._lwd_group_specs:
                num_spec_blocks = getattr(spec, "num_speculative_blocks", 0)
                if getattr(spec, "mamba_cache_mode", None) == "align":
                    need_blocks += (1 + num_spec_blocks) if fresh else 1
                else:
                    need_blocks += (
                        cdiv(item.num_tokens + spec_tokens, spec.block_size)
                        + num_spec_blocks
                    )
        need_blocks += num_decode * len(self._lwd_group_specs)
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        if need_blocks > free_blocks:
            logger.debug(
                "[Lwd][cloud-sched] preflight hold: kv blocks %d > free %d",
                need_blocks, free_blocks,
            )
            return False
        return True

    @staticmethod
    def _lwd_verify_full_admission(
        out: SchedulerOutput, notify: LwdRangeNotify, items: list[LwdRangeItem]
    ) -> None:
        """步后核验:批内每请求排程量 == 预告量(chunk 精确准入)。
        不等即配置发散/KV 失守(错排会让整批 UP 张量留残余行),
        fail-fast 不当场炸就会静默错算。"""
        for item in items:
            scheduled = out.num_scheduled_tokens.get(item.request_id, 0)
            if scheduled != item.num_tokens:
                raise RuntimeError(
                    f"[Lwd][cloud-sched] batch admission mismatch "
                    f"req={item.request_id} seqno={notify.seqno}: "
                    f"scheduled={scheduled} != notify={item.num_tokens} "
                    f"(mixed mode forbids partial admission)"
                )

    # ------------------------------------------------------------------ #
    # 调度主体                                                            #
    # ------------------------------------------------------------------ #
    def _schedule_impl(self) -> SchedulerOutput:
        decode_ids = self._lwd_collect_decode_requests()
        notify = (
            self.prefill_notify_queue[0] if self.prefill_notify_queue else None
        )
        if notify is not None:
            items, unknown = self._lwd_classify_items(
                self._lwd_notify_items(notify)
            )
            if not items and not unknown:
                # 全部条目失效(完结/重复):弹队,本步纯 decode。
                # 按对象身份弹(abort 清扫可能整体替换过 deque,见
                # _schedule_mixed 同款处理)
                queue = self.prefill_notify_queue
                if queue and queue[0] is notify:
                    queue.popleft()
                if self._lwd_cur_ghost_items:
                    # 全幽灵批:张量仍在路上,必须照收(保 HCCL numel
                    # 配对)但不注入;decode 照常
                    out = self._lwd_schedule_for_visible_reqs(decode_ids)
                    out.lwd_batch = self._lwd_build_embed_batch(
                        notify, self._lwd_notify_items(notify)
                    )
                    self._lwd_release_ghosts(notify)
                    return out
                notify = None
            elif unknown:
                # 请求 ADD 未落地(input_queue 次序滞后):预告留队首,
                # 本步先 decode;abort 清扫兜底防永久堵队首
                logger.debug(
                    "[Lwd][cloud-sched] notify seqno=%d deferred: "
                    "request(s) not added yet",
                    notify.seqno,
                )
                notify = None
            elif not self._lwd_preflight_ok(items, len(decode_ids)):
                # 预算/名额/KV 不足:预告留队首,decode 泄压后重试
                notify = None
            else:
                return self._schedule_mixed(notify, items, decode_ids)
        return self._lwd_schedule_for_visible_reqs(decode_ids)

    def _schedule_mixed(
        self,
        notify: LwdRangeNotify,
        items: list[LwdRangeItem],
        decode_ids: list[str],
    ) -> SchedulerOutput:
        """混排步:decode 集 + 批内 chunk 同批下发。

        offset 对账:item.offset 必须 == 云侧当前 num_computed(async
        关闭前提下调度时刻即真实水位)——不等即 chunk 断流(>)/
        重复(<)/乱序,协议级事故 fail-fast,不当场炸就会把
        embeddings 注入错位窗口静默错算。
        双钳制复现(设计 §2.3/§2.4 修订,前提:批内非末位 chunk 同尺寸,
        由边侧 picker 同质性门控保证——网格收边会产出 < 单元的非末位
        块,异尺寸混批时 T 压不住小块所属请求的余量,会被预算超排):① 阈值复现——临时覆写
        long_prefill_token_threshold = max(items.num_tokens),非末位
        自限块由 R/C/T 定界逐请求复现;② 预算钳制——临时把
        max_num_scheduled_tokens 压到 decode 精确消耗 + Σ chunk,
        末位 cap 截断块由"预算耗尽"复现(与集中式 chunked prefill
        的截断语义逐字对齐)。两个 finally 均恢复。
        注意 decode 请求 num_new = 1+k 远小于正常量级的 T,不受覆写
        影响(病态小 chunk 配置见设计 §8-C9)。"""
        logger.debug(
            "[Lwd][cloud-sched] mixed notify seqno=%d reqs=%s tokens=%d",
            notify.seqno, [item.request_id for item in items],
            sum(item.num_tokens for item in items),
        )
        for item in items:
            request = self.requests[item.request_id]
            computed = request.num_computed_tokens
            if computed != item.offset and computed == 0 and item.offset > 0:
                # 首 chunk 且边侧前缀命中 trim(offset>0):云侧 resume
                # 要等原生准入(get_computed_blocks)才落 num_computed,
                # 对账时点尚未生效——按块池实查的 resume 量对账(与准入
                # 同源、含边侧声明封顶,纯读 lookup 幂等;代价是命中
                # 统计双计一次)。
                computed = self.kv_cache_manager.get_computed_blocks(request)[1]
            if computed != item.offset:
                # 诊断:抢占复位(num_preemptions>0 且 computed 归零)
                # 与非抢占的断流/乱序分开报——LWD 占位 embeds 不可
                # 重算,抢占即协议不可恢复,必须一眼可辨
                preempted = getattr(request, "num_preemptions", 0) > 0
                in_queues = (
                    "running" if request in self.running
                    else "waiting" if request in self.waiting
                    else "skipped" if request in self.skipped_waiting
                    else "none"
                )
                raise RuntimeError(
                    f"[Lwd][cloud-sched] chunk offset mismatch "
                    f"req={item.request_id} seqno={notify.seqno}: "
                    f"notify offset={item.offset} != cloud computed="
                    f"{computed} (resume-checked)"
                    f"(num_preemptions={getattr(request, 'num_preemptions', '?')}"
                    f"{', PREEMPTED—占位 embeds 不可重算,抢占不可恢复' if preempted else ''}"
                    f", status={request.status}, queue={in_queues}, "
                    f"prompt={request.num_prompt_tokens}; "
                    f"chunk 断流/重复/乱序或 resume 与边侧 trim 不一致)"
                )
        req_ids = decode_ids + [item.request_id for item in items]
        # decode 精确消耗:预算钳制的精确性依赖逐字复刻原生 running 段
        # (scheduler.py:381-416)的 num_new 推导——1+len(spec) 不是准确式:
        # ① placeholder(spec 拒绝遗留的非零值)计入排程窗口;
        # ② max_tokens 守卫(占位>0 且距 max_tokens 不足 2)与
        #   next_decode_eligible_step 会让请求整步跳过、消耗 0。
        # 不逐字复刻就会留下余量,使末位截断块被超排(真机实录:
        # decode 请求被守卫跳过消耗 0,末位块 604 被排成 608)。
        # threshold 不约束 decode(正常量级 T ≫ num_new,病态小 chunk
        # 配置见设计 §8-C9);预算由构造保证充足,不影响本推导。
        decode_exact = 0
        for rid in decode_ids:
            req = self.requests[rid]
            placeholders = req.num_output_placeholders
            if (
                placeholders > 0
                and req.num_computed_tokens + 2 - placeholders
                >= req.num_prompt_tokens + req.max_tokens
            ):
                continue  # 原生 max_tokens 守卫:本步跳过,消耗 0
            if self.current_step < req.next_decode_eligible_step:
                continue  # V2+PP 节拍守卫:本步跳过,消耗 0
            decode_exact += max(
                0,
                min(
                    req.num_tokens_with_spec
                    + placeholders
                    - req.num_computed_tokens,
                    self.max_model_len - 1 - req.num_computed_tokens,
                ),
            )
        saved_threshold = self.scheduler_config.long_prefill_token_threshold
        saved_budget = self.max_num_scheduled_tokens
        # 跟随公告 chunk 豁免(与基类 token_budget_cap 分支同义,但
        # threshold 不能置 0——它是本类复现边侧 chunk 边界的手段,
        # 故自管 save/restore 而不走基类 cap 参数):
        # mamba 块对齐复切/full_isl 整序列就位会把调度窗口切离预告
        # 边界,步后核验必炸;云侧不复切边侧已定的 chunk。
        saved_mamba_align = self.need_mamba_block_aligned_split
        saved_full_isl = self.scheduler_reserve_full_isl
        self.need_mamba_block_aligned_split = False
        self.scheduler_reserve_full_isl = False
        self.scheduler_config.long_prefill_token_threshold = max(
            item.num_tokens for item in items
        )
        self.max_num_scheduled_tokens = decode_exact + sum(
            item.num_tokens for item in items
        )
        # mamba/GDN 网格守卫:云模型有混合注意力时(原生
        # need_mamba_block_aligned_split 为真),chunk 边界必须落在
        # 云侧对齐网格上(边侧经 probe 学到网格后重切)。mixed 步豁免
        # 原生复切的前提就是这个;边侧没学到网格(协调未启用/probe
        # 失败)时 chunk 任意边界直上 GDN = 静默乱码,必须当场暴露。
        if saved_mamba_align:
            grid = int(
                getattr(
                    self.kv_cache_manager.coordinator,
                    "scheduler_block_size",
                    0,
                )
            )
            if grid > 0:
                use_eagle = getattr(self, "_lwd_prefix_use_eagle", False)
                for item in items:
                    req = self.requests[item.request_id]
                    end = item.offset + item.num_tokens
                    # 三方同源公式:与边侧 picker 预测、原生实切一致;
                    # use_eagle 取云侧 EAGLE 尾块回退口径。
                    aligned = aligned_chunk_length(
                        item.offset,
                        req.num_prompt_tokens,
                        item.num_tokens,
                        grid,
                        True,
                        use_eagle,
                    )
                    if aligned != item.num_tokens:
                        raise RuntimeError(
                            "[Lwd][cloud-sched] chunk 违背云侧前缀检查点对齐 "
                            f"req={item.request_id} seqno={notify.seqno}: "
                            f"[{item.offset}, {end}) grid={grid} "
                            f"tokens={item.num_tokens} aligned={aligned} — "
                            "边侧未学到云 mamba 网格(协调未启用/probe 失败)"
                            "或 EAGLE 尾块回退口径与云侧不一致;未对齐的 GDN "
                            "状态切片会静默乱码"
                        )
        # 末位截断块的"预算耗尽复现"要求它最后消费:running 内
        # 续传请求与 decode 按队列序交错,若它不在末尾,轮到自己时
        # 剩余预算可能 > c_trunc(前序有人少消耗),拿到更多 → 步后
        # 核验误炸(2E1C 在途翻倍后窗口显著放大)。移到 running 队尾
        # 直接消掉该顺序假设(waiting 来源的截断块本就按 FCFS 末位,
        # 天然正确)。截断块恒为 items 末位(边侧 picker 语义)。
        last_id = items[-1].request_id
        for i, req in enumerate(self.running):
            if req.request_id == last_id and i != len(self.running) - 1:
                self.running.append(self.running.pop(i))
                break
        try:
            out = self._lwd_schedule_for_visible_reqs(req_ids)
        finally:
            self.scheduler_config.long_prefill_token_threshold = (
                saved_threshold
            )
            self.max_num_scheduled_tokens = saved_budget
            self.need_mamba_block_aligned_split = saved_mamba_align
            self.scheduler_reserve_full_isl = saved_full_isl
        self._lwd_verify_full_admission(out, notify, items)
        # 消费队首:abort 清扫(IO 线程)可能整体替换过 deque,按对象
        # 身份弹队首,避免错弹替换后的新队首
        queue = self.prefill_notify_queue
        if queue and queue[0] is notify:
            queue.popleft()
        # UP 链 seqno 随批下发云 worker:批配对号直接取预告自带 seqno(与
        # 边侧 EMBED 批派发号同源同值)。batch_meta 承载 worker 的 recv
        # 尺寸与注入切行信息:req_ids 取批内逐请求,token_ids 为占位
        # 列表——长度必须等于边侧实际发送的逐请求 token 数,recv numel
        # 才能与边侧 isend 严格相等(HCCL P2P 要求两端 numel 匹配)。
        # prompt_offsets/has_mrope 逐 item 透传:worker 据此求 aux 帧
        # 尺寸(Σ 有 mrope 条目的 n_i×3),runner 据此对 MM 请求注入
        # mrope 行(文本请求零 aux 流量、零路径变化)。
        out.lwd_batch = self._lwd_build_embed_batch(
            notify, self._lwd_notify_items(notify)
        )
        self._lwd_release_ghosts(notify)
        return out

    def _lwd_build_embed_batch(
        self, notify: LwdRangeNotify, all_items: list[LwdRangeItem]
    ) -> LwdBatch:
        """组 EMBED 批(含幽灵条目)。

        必须按原始 items(含幽灵)建 meta:边侧 UP 组帧含幽灵请求的
        embeds 行(组帧先行不可撤),recv numel 按 meta.token_ids 逐
        请求长度求和——少一行即 HCCL 两端 numel 不等(通道楔死)。
        幽灵行由 runner 照收不注入(ghost_req_ids)。

        UP 链 seqno 随批下发云 worker:批配对号直接取预告自带 seqno(与
        边侧 EMBED 批派发号同源同值)。token_ids 为占位列表——长度
        必须等于边侧实际发送的逐请求 token 数(HCCL P2P 要求两端
        numel 严格相等)。prompt_offsets/has_mrope 逐 item 透传。"""
        ghost_ids = [i.request_id for i in self._lwd_cur_ghost_items]
        return LwdBatch(
            batch_type=LwdBatchType.LWD_EMBED,
            seqno=notify.seqno,
            batch_meta=LwdEmbedBatch(
                req_ids=[item.request_id for item in all_items],
                token_ids=[[0] * item.num_tokens for item in all_items],
                # 一条 RangeNotify 来自一条边(单 publish 原子),批内
                # items 同属该边(入口已按同一 edge_id 包装);UP recv
                # 选路维度取首条目解析值,与相位机同口径
                edge_id=parse_edge_id(all_items[0].request_id)[0],
                prompt_offsets=[item.offset for item in all_items],
                has_mrope=[item.has_mrope for item in all_items],
                ghost_req_ids=ghost_ids,
            ),
        )

    def _lwd_release_ghosts(self, notify: LwdRangeNotify) -> None:
        """消费后摘除幽灵台账条目:仅当队列中再无该请求的其他待消费
        预告时摘除(同一请求的后续 chunk 预告仍需按幽灵照收)。"""
        ghost_ids = {i.request_id for i in self._lwd_cur_ghost_items}
        self._lwd_cur_ghost_items = []
        if not ghost_ids:
            return
        pending_ids = set()
        for pending in self.prefill_notify_queue:
            if pending.items:
                pending_ids.update(i.request_id for i in pending.items)
            else:
                pending_ids.add(pending.request_id)
        self.lwd_ghost_req_ids.difference_update(ghost_ids - pending_ids)

    def schedule(self) -> SchedulerOutput:
        # [Lwd][perf] 云侧每步 LWD 税分段之一:调度(容器交换)时长
        _t = time.monotonic()
        out = self._schedule_impl()
        self._lwd_sched_step += 1
        self._lwd_log_sched_batch(out)
        logger.debug(
            "[Lwd][perf] cloud-sched dur=%.2fms", (time.monotonic() - _t) * 1000
        )
        return out

    def _lwd_log_sched_batch(self, out: SchedulerOutput) -> None:
        """[Lwd][sched] 每步调度批结构化日志(lwd_backlog_probe ⑤ 段锚点)。

        phase 判定:lwd_batch=EMBED 且批内还有 decode 请求即 MIXED;EMBED
        独占为 PREFILL;有排程 token 为 DECODE;否则 EMPTY。pending_notify
        给出待吃量——EMPTY/DECODE 连跑且 pending_notify>0 = 通知已到但
        preflight 不过或 ADD 未落地。reqs 超 12 个截断,防大 decode 批
        刷屏。"""
        batch = getattr(out, "lwd_batch", None)
        if batch is not None and batch.batch_type == LwdBatchType.LWD_EMBED:
            n_prefill = len(batch.batch_meta.req_ids)
            n_reqs = len(out.num_scheduled_tokens)
            phase = "MIXED" if n_reqs > n_prefill else "PREFILL"
            seqno: object = batch.seqno
        else:
            phase = "DECODE" if out.total_num_scheduled_tokens else "EMPTY"
            seqno = ""
        reqs = list(out.num_scheduled_tokens)
        reqs_str = ",".join(reqs[:12]) + (
            f",+{len(reqs) - 12}more" if len(reqs) > 12 else ""
        )
        logger.debug(
            "[Lwd][sched] cloud step=%d phase=%s seqno=%s reqs=[%s] tokens=%d "
            "pending_notify=%d decode_ready=%d",
            self._lwd_sched_step, phase, seqno, reqs_str,
            out.total_num_scheduled_tokens, len(self.prefill_notify_queue),
            len(self._lwd_collect_decode_requests()),
        )
