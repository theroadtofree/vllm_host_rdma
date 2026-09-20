"""云侧混合调度器:prefill/decode 同批下发(设计:lwd_mixed_batch_design.md)。

整体 prefill 组批语义:边侧只发整 prompt 批(不切 chunk,预算不足即
边侧排队),云侧每步至多消费 prefill_notify_queue 队首一条批量
RangeNotify,与全部 decode 态请求混排一个批下发:

  - preflight 三检(预算/名额/KV 保守预估)任一不过 -> 本步 decode-only,
    预告留队首原位,decode 泄压后下步重试;
  - 放行则可见集 = decode 集 + 批内请求,容器交换调原生 schedule();
  - 步后全量核验批内每请求 num_scheduled == 预告量,不等即 fail-fast
    (mixed 不允许部分准入:一次 UP recv 对应整批张量,部分注入会留
    残余行错配)。

前置约束:spec decode(MTP)预算已按 1+k 系数对齐(边 cap 与本类
preflight),但端到端未经真机验证(eagle 纯度判据失真风险,台账
P3-3/设计 M-8);
不兼容云侧抢占重算(占位 embeds 无法本地重 prefill)与 prefix caching
(命中使调度量 < 整 prompt,触发注入窗口看门狗)——部署须保证 KV
充足并关闭缓存,本类的核验只负责把违背变成当场报错而非静默错算。
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
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
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
        # 占预算与 KV);k = num_spec_tokens,无 spec 配置即 1
        spec_config = getattr(self.vllm_config, "speculative_config", None)
        self._lwd_spec_factor: int = (
            1 + spec_config.num_spec_tokens if spec_config is not None else 1
        )
        logger.info(
            "[Lwd] cloud mixed scheduler: whole-prefill batches mixed with "
            "decode (one notify consumed per step at most, spec_factor=%d)",
            self._lwd_spec_factor,
        )

    @staticmethod
    def _lwd_is_decode(request) -> bool:
        """prompt 已算完 = decode 态(可采样)。整体组批下 running 请求
        恒为 decode 态(num_computed 一跳到底),本判据是兜底过滤。"""
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
            )
        ]

    def _lwd_classify_items(
        self, items: list[LwdRangeItem]
    ) -> tuple[list[LwdRangeItem], bool]:
        """条目三分:ready(待 prefill)/ stale(已完结或重复预告,剔除)
        / unknown(ADD 尚未落地)。返回 (ready, 是否有 unknown)。

        unknown 存在即整批 defer(all-or-nothing:一次 UP recv 对应整批
        张量,不能只放行子集);abort 清扫(引擎侧)负责把永不落地的
        条目摘除,队首不会被永久堵住。"""
        ready: list[LwdRangeItem] = []
        unknown = False
        for item in items:
            request = self.requests.get(item.request_id)
            if request is None:
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

    def _lwd_preflight_ok(
        self, items: list[LwdRangeItem], num_decode: int
    ) -> bool:
        """放行三检:token 预算 / 并发名额 / KV 保守预估。

        预算与名额本由边侧 cap 保证(设计 §2),此处是双保险;KV 预估
        忽略前缀缓存命中(命中只降占用,高估是安全方向)。decode 占用
        按 1+k 计(MTP 草稿 token 同样吃预算;KV 块预留同系数上取整)。"""
        need_tokens = (
            sum(item.num_tokens for item in items)
            + num_decode * self._lwd_spec_factor
        )
        if need_tokens > self.max_num_scheduled_tokens:
            logger.info(
                "[Lwd][cloud-sched] preflight hold: tokens %d > budget %d",
                need_tokens, self.max_num_scheduled_tokens,
            )
            return False
        if len(self.running) + len(items) > self.max_num_running_reqs:
            logger.info(
                "[Lwd][cloud-sched] preflight hold: seqs %d+%d > max %d",
                len(self.running), len(items), self.max_num_running_reqs,
            )
            return False
        need_blocks = (
            sum(cdiv(item.num_tokens, self.block_size) for item in items)
            + num_decode * cdiv(self._lwd_spec_factor, self.block_size)
        )
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        if need_blocks > free_blocks:
            logger.info(
                "[Lwd][cloud-sched] preflight hold: kv blocks %d > free %d",
                need_blocks, free_blocks,
            )
            return False
        return True

    @staticmethod
    def _lwd_verify_full_admission(
        out: SchedulerOutput, notify: LwdRangeNotify, items: list[LwdRangeItem]
    ) -> None:
        """步后全量核验:批内每请求排程量 == 预告量。不等即配置发散/
        KV 失守(部分准入会让整批 UP 张量留残余行),fail-fast 不当场
        炸就会静默错算。"""
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
                notify = None
            elif unknown:
                # 请求 ADD 未落地(input_queue 次序滞后):预告留队首,
                # 本步先 decode;abort 清扫兜底防永久堵队首
                logger.info(
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
        """混排步:decode 集 + 批内整 prompt 请求同批下发。"""
        logger.info(
            "[Lwd][cloud-sched] mixed notify seqno=%d reqs=%s tokens=%d",
            notify.seqno, [item.request_id for item in items],
            sum(item.num_tokens for item in items),
        )
        req_ids = decode_ids + [item.request_id for item in items]
        out = self._lwd_schedule_for_visible_reqs(req_ids)
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
        out.lwd_batch = LwdBatch(
            batch_type=LwdBatchType.LWD_EMBED,
            seqno=notify.seqno,
            batch_meta=LwdEmbedBatch(
                req_ids=[item.request_id for item in items],
                token_ids=[[0] * item.num_tokens for item in items],
            ),
        )
        return out

    def schedule(self) -> SchedulerOutput:
        # [Lwd][perf] 云侧每步 LWD 税分段之一:调度(容器交换)时长
        _t = time.monotonic()
        out = self._schedule_impl()
        self._lwd_sched_step += 1
        self._lwd_log_sched_batch(out)
        logger.info(
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
        logger.info(
            "[Lwd][sched] cloud step=%d phase=%s seqno=%s reqs=[%s] tokens=%d "
            "pending_notify=%d decode_ready=%d",
            self._lwd_sched_step, phase, seqno, reqs_str,
            out.total_num_scheduled_tokens, len(self.prefill_notify_queue),
            len(self._lwd_collect_decode_requests()),
        )
