"""Lwd 边/云调度器公共基类:收敛两侧共享的调度语义。

仍继承 AsyncScheduler,原生调度行为与 isinstance 判定不变;共享逻辑
随重构逐步上提至此类。
"""

from __future__ import annotations

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.request import Request

from vllm.logger import init_logger

logger = init_logger(__name__)


def aligned_chunk_length(
    start, prompt_tokens, limit, alignment, required, use_eagle=False
):
    """在 prompt 状态检查点坐标上收 chunk:mamba/GDN 状态只有完整的
    检查点区间才可复用,`use_eagle` 时再留一个区间给 EAGLE 尾块重算。
    边侧 picker 预测、边侧实切、云侧守卫三方共用本公式。"""
    count = max(0, min(prompt_tokens - start, limit))
    if alignment <= 0 or count == 0:
        return count
    last_checkpoint = prompt_tokens // alignment * alignment
    if required and use_eagle:
        # 对齐原生 Mamba 语义:留一个完整检查点区间给 EAGLE 尾块重算,
        # 不缓存部分尾块状态。
        last_checkpoint = max(0, last_checkpoint - alignment)
    if start >= last_checkpoint:
        return count
    end = start + count
    checkpoint = min(end // alignment * alignment, last_checkpoint)
    if checkpoint <= start:
        return 0 if required else count
    return checkpoint - start


class LwdBaseScheduler(AsyncScheduler):
    """边/云调度器公共基类(AsyncScheduler 子类)。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 收尾快照:request_id -> (prompt_tokens, completion_tokens)。
        # 释放链唯一收口点是 ``_free_request``(常规终结 update_from_output 直接
        # 调它,**不经 finish_requests**);而 LWD 的 finish/usage 结算发生在其后
        # 的 lwd_handle_model_output —— 那时请求已从 self.requests 摘除。故在
        # _free_request 覆写里抓最小快照,由引擎侧消费(消费即摘除);超上限
        # 按插入序兜底淘汰残留。phase/mixed 两个云调度臂共用。
        self.lwd_finished_records: dict[str, tuple[int, int]] = {}
        # abort 幽灵台账:已发布预告(UP 张量必随行)但请求已 abort 的
        # request_id 集合。消费时这些条目照收不注入(保 HCCL numel
        # 配对),由调度臂建批时挂 ghost_req_ids 并摘除台账条目。
        self.lwd_ghost_req_ids: set[str] = set()

    def _free_request(self, request, delay_free_blocks: bool = False):
        """收尾快照(见 ``lwd_finished_records``):挂在**释放链唯一收口点**。

        常规终结(update_from_output 里 check_stop 命中)直接
        ``self._free_request(request)``(**不经 finish_requests**),abort/error
        走 ``finish_requests`` → ``_free_request``;两者都收口于本方法,故快照
        必须挂这里(挂 finish_requests 会漏掉绝大多数正常结束)。此刻请求尚在、
        长度已是最终值(update_from_output 先落 token 再终结);由引擎侧消费
        (消费即摘除),超上限按插入序兜底淘汰残留。"""
        try:
            self.lwd_finished_records[request.request_id] = (
                request.num_prompt_tokens,
                request.num_tokens - request.num_prompt_tokens,
            )
            if len(self.lwd_finished_records) > 1024:
                for stale in list(self.lwd_finished_records)[:256]:
                    self.lwd_finished_records.pop(stale, None)
                logger.warning(
                    "[Lwd][coord] finished-record table full; evicted oldest"
                )
        except Exception:  # noqa: BLE001  快照失败不阻断释放
            logger.warning(
                "[Lwd][coord] finished snapshot failed (non-fatal)", exc_info=True
            )
        return super()._free_request(request, delay_free_blocks)

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """LWD 云侧禁抢占:请求的 prompt embeds 是占位零值(真实内容在
        边侧,UP 通道一次性流转),抢占复位后云侧无法重 prefill,边侧
        也不会重发已消费 chunk——抢占即协议级不可恢复。在原生日志点
        之前 fail-fast,把"KV 不足"暴露在真正的故障点,而不是等到
         chunk offset 对账时以错位报错收场(实测:抢占复位 computed=0,
               下一 chunk offset=768 对账炸,根因不可辨)。

        边侧(0-KV 组)分配恒成功,本路径不可达,不受影响。"""
        raise RuntimeError(
            f"[Lwd] cloud preemption is unrecoverable for LWD requests "
            f"(placeholder embeds cannot be re-prefilled, edge will not "
            f"resend consumed chunks): req={request.request_id} "
            f"prompt={request.num_prompt_tokens} "
            f"computed={request.num_computed_tokens} "
            f"preemptions={request.num_preemptions} — cloud KV/budget "
            f"insufficient for the admitted load; reduce concurrency or "
            f"raise KV capacity"
        )

    def _lwd_new_queue(self, reqs: list[Request]) -> RequestQueue:
        """新建调度策略队列并装入 reqs。"""
        queue = create_request_queue(self.policy)
        for req in reqs:
            queue.add_request(req)
        return queue

    def _lwd_schedule_for_visible_reqs(
        self, req_ids: list[str], token_budget_cap: int | None = None
    ) -> SchedulerOutput:
        """把 req_ids 指定的请求从三队列剔除、单独调度,步后按原队列拼回。

        waiting/skipped 来源的请求走原生准入窗口(allocate/状态迁移/
        记账一样不少),running 来源的走续跑。拼回:仍被调度的接在隐藏
        running 之后,被抢占的排 waiting 尾部,被跳过的排 skipped 队首。

        参数 ``token_budget_cap``:本步调度预算上限,决定权移交边侧公告量
        (云侧跟随边侧 chunk)。仅收紧不放宽;跟随模式下同时豁免原生
        复切/就位约束,保证调度窗口 == 预告行数。"""
        picked = {self.requests[req_id] for req_id in req_ids}
        from_running = [req for req in self.running if req in picked]
        from_waiting = [req for req in self.waiting if req in picked]
        from_skipped = [req for req in self.skipped_waiting if req in picked]
        # 剔除后保存队列状态(隐藏集)
        self.running = [req for req in self.running if req not in picked]
        self.waiting.remove_requests(from_waiting)
        self.skipped_waiting.remove_requests(from_skipped)
        saved = (self.running, self.waiting, self.skipped_waiting)
        # 被剔除的请求按原队列归位成可见集,单独调度
        self.running = from_running
        self.waiting = self._lwd_new_queue(from_waiting)
        self.skipped_waiting = self._lwd_new_queue(from_skipped)
        saved_token_budget = self.max_num_scheduled_tokens
        # 边侧是 prefill 切块权威:跟随模式下豁免原生调度三项约束
        # (long_prefill 阈值复切/mamba 对齐复切/整序列就位),否则调度
        # 窗口 != 预告行数,UP 注入窗口失配;预算钳制(min)仍生效。
        saved_long_prefill_threshold = (
            self.scheduler_config.long_prefill_token_threshold
        )
        saved_mamba_align = self.need_mamba_block_aligned_split
        saved_full_isl = self.scheduler_reserve_full_isl
        if token_budget_cap is not None:
            self.max_num_scheduled_tokens = min(
                saved_token_budget, max(0, token_budget_cap)
            )
            self.scheduler_config.long_prefill_token_threshold = 0
            self.need_mamba_block_aligned_split = False
            self.scheduler_reserve_full_isl = False
        try:
            out = super().schedule()
        finally:
            self.max_num_scheduled_tokens = saved_token_budget
            self.scheduler_config.long_prefill_token_threshold = (
                saved_long_prefill_threshold
            )
            self.need_mamba_block_aligned_split = saved_mamba_align
            self.scheduler_reserve_full_isl = saved_full_isl
            # 按原队列拼回:running 存活者接尾,waiting 被抢占者排队尾,
            # skipped 被跳过者排队首
            post = (self.running, self.waiting, self.skipped_waiting)
            self.running, self.waiting, self.skipped_waiting = saved
            self.running += post[0]
            self.waiting.extend(post[1])
            self.skipped_waiting.prepend_requests(post[2])
        return out
