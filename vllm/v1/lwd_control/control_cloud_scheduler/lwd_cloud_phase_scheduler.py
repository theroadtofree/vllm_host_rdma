"""云侧相位调度器:工作纯相位批次。prefill 步只算 prompt 工作(prefill 首块 +
尾巴),decode 步只算已完结请求的 1-token 采样;prefill 批最多一个请求。"""

from __future__ import annotations

import time
from collections import deque

from vllm.logger import init_logger
from vllm.v1.core.sched.output import (
    LwdBatch,
    LwdBatchType,
    LwdEmbedBatch,
    SchedulerOutput,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import LwdRangeNotify
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
)

logger = init_logger(__name__)

_LWD_PHASE_PREFILL_FIRST = "prefill_first"
_LWD_PHASE_DECODE_FIRST = "decode_first"


class LwdCloudPhaseScheduler(LwdBaseScheduler):
    """工作纯相位批次策略;相位(prefill_first/decode_first)构造期自解析。
    前置约束:不兼容 spec decode(eagle 会 shift num_computed_tokens,纯度判据失真)。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lwd_prefill_first = self._lwd_resolve_phase()
        # One-shot 翻转:某相位空步而另一相位有活时,强制下一步走后者;
        # 双标志显式定向,按偏好取反会错翻,造成空步死循环。
        self._force_prefill_once: bool = False
        self._force_decode_once: bool = False
        # 上一个非空步是否为 prefill,驱动 schedule() 的禁连续 prefill 不变量
        self._last_step_was_prefill: bool = False
        # prefill 通知队列:边侧范围预告(RangeNotify)逐条入队,每步取
        # 队首点名其 request_id;预告自带 seqno 即本步 UP 链配对号
        self.prefill_notify_queue: deque[LwdRangeNotify] = deque()
        # [Lwd][sched] 调度批日志步计数(饿死分析:RangeNotify 到达 →
        # PREFILL 步消费的间隔与中间插入的 decode 步数)
        self._lwd_sched_step = 0
        logger.info(
            "[Lwd] cloud phase scheduler: single-request prefill batches "
            "enforced (edge/cloud chunk stream stays per-request contiguous)"
        )

    def _lwd_resolve_phase(self) -> bool:
        """返回 True=prefill_first;缺省/未知相位告警回退 prefill_first。"""
        from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
            LwdConfig,
        )

        phase = LwdConfig.from_env_and_config(self.vllm_config).scheduler_name
        if phase == _LWD_PHASE_DECODE_FIRST:
            return False
        if phase != _LWD_PHASE_PREFILL_FIRST:
            logger.warning(
                "[Lwd] unknown cloud scheduler phase %r "
                "(known: %s); falling back to %s",
                phase,
                [_LWD_PHASE_DECODE_FIRST, _LWD_PHASE_PREFILL_FIRST],
                _LWD_PHASE_PREFILL_FIRST,
            )
        return True

    @staticmethod
    def _lwd_is_decode(request) -> bool:
        """prompt 已算完 = decode 态(可采样);未算完的是 prefill 尾巴。"""
        return request.num_computed_tokens >= request.num_prompt_tokens

    def _lwd_has_prefill_tails(self) -> bool:
        return any(not self._lwd_is_decode(r) for r in self.running)

    def _lwd_has_decode_ready(self) -> bool:
        return any(self._lwd_is_decode(r) for r in self.running)

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
    # Phase primitives(容器交换;原生 schedule() 零改动)                  #
    # ------------------------------------------------------------------ #
    def _schedule_pure_prefill(self) -> SchedulerOutput:
        """纯 prefill 步:取队首预告,按公告量钳制本步预算,照单执行。"""
        notify = q.popleft() if (q := self.prefill_notify_queue) else None
        if notify is None:
            return self._lwd_schedule_for_visible_reqs([])
        logger.info(
            "[Lwd][cloud-sched] prefill notify req=%s seqno=%s num=%s",
            notify.request_id, notify.seqno, notify.num_tokens,
        )
        out = self._lwd_schedule_for_visible_reqs(
            [notify.request_id], token_budget_cap=notify.num_tokens
        )
        out.lwd_batch = LwdBatch(
            batch_type=LwdBatchType.LWD_EMBED,
            seqno=notify.seqno,
            batch_meta=LwdEmbedBatch(
                req_ids=[notify.request_id],
                token_ids=[[0] * notify.num_tokens],
                token_offsets=[notify.offset],
            ),
        )
        return out

    def _schedule_pure_decode(self) -> SchedulerOutput:
        """纯 decode 步:收集全部 decode 态请求,剔除单独调度后按落点拼回。"""
        req_ids = self._lwd_collect_decode_requests()
        if req_ids:
            logger.info("[Lwd][cloud-sched] decode reqs=%s", req_ids)
        return self._lwd_schedule_for_visible_reqs(req_ids)

    @staticmethod
    def _is_empty(out: SchedulerOutput) -> bool:
        return out.total_num_scheduled_tokens == 0

    def _prefer_prefill(self) -> bool:
        """相位选择:prefill_first 有等待/尾巴即 prefill;
        decode_first 只要存在纯 decode 活就优先 decode。"""
        if self._lwd_prefill_first:
            return bool(self.waiting) or self._lwd_has_prefill_tails()
        return not self._lwd_has_decode_ready()

    def schedule(self) -> SchedulerOutput:
        # [Lwd][perf] 云侧每步 LWD 税分段之一:相位调度(容器交换)时长
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

        phase 判定:lwd_batch=EMBED 即纯 prefill 步(携带 UP 配对 seqno);
        有排程 token 为 DECODE;否则 EMPTY。pending_notify/decode_ready
        给出两相位各自的待吃量——EMPTY 且 pending_notify>0 = 通知已到
        但引擎没吃(引擎线程被 publish 小睡/收割阻塞的直接信号);
        DECODE 连跑且 pending_notify>0 = decode 插队饿 prefill。reqs 超
        12 个截断,防大 decode 批刷屏。"""
        batch = getattr(out, "lwd_batch", None)
        if batch is not None and batch.batch_type == LwdBatchType.LWD_EMBED:
            phase, seqno = "PREFILL", batch.seqno
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

    def _schedule_impl(self) -> SchedulerOutput:
        prefer_prefill = self._prefer_prefill()
        if self._force_prefill_once:
            self._force_prefill_once = False
            prefer_prefill = True
        elif self._force_decode_once:
            self._force_decode_once = False
            prefer_prefill = False
        # 不变量:prefill 不连续执行两步;仅当中间的 decode 步为空时才允许
        # 连续,空 decode 步经 _force_prefill_once 翻回。
        if prefer_prefill and self._last_step_was_prefill and self.running:
            prefer_prefill = False
        self._last_step_was_prefill = False

        if prefer_prefill:
            out = self._schedule_pure_prefill()
            if self._is_empty(out) and self.running:
                # prefill 受 KV 压力阻塞:放行空步,下一步转 decode 泄压
                self._force_decode_once = True
            else:
                self._last_step_was_prefill = True
            return out
        out = self._schedule_pure_decode()
        if self._is_empty(out) and (self.waiting or self._lwd_has_prefill_tails()):
            # decode 无活但有 waiting/尾巴:翻回 prefill(不变量的空步出口)
            self._force_prefill_once = True
        return out
