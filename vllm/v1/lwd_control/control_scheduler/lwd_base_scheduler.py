"""Lwd 边/云调度器公共基类:收敛两侧共享的调度语义。

仍继承 AsyncScheduler,原生调度行为与 isinstance 判定不变;共享逻辑
随重构逐步上提至此类。
"""

from __future__ import annotations

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.request import Request


class LwdBaseScheduler(AsyncScheduler):
    """边/云调度器公共基类(AsyncScheduler 子类)。"""

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

        参数 ``token_budget_cap``:本步调度预算上限,把"本步执行多少
        token"的决定权从本侧原生预算移交给边侧公告量(云侧跟随边侧
        chunk)。仅用于收紧,不放宽原生上限;跟随模式下同时豁免
        long_prefill_token_threshold 复切(云侧不复切边侧 chunk,保证
        调度窗口 == 预告行数);原生预算内的块对齐复切(mamba align)
        等机制照常生效。"""
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
        # 边侧是 prefill 切块权威:跟随模式(token_budget_cap 给定 = 云侧
        # 照单执行边侧 chunk)下豁免原生调度的所有"再切"机制,否则调度
        # 窗口 != 预告 chunk 行数,UP 注入窗口失配直接挂死:
        # 1) long_prefill_token_threshold 复切(max_num_partial_prefills>1
        #    且未显式配置时自动取 max_model_len*0.04);
        # 2) mamba/混合注意力状态块对齐复切(_mamba_block_aligned_split,
        #    mamba_cache_mode=align 时把 chunk 收到 block_size 整数倍;
        #    边侧多模态整块发出,天然非对齐,云侧复切即失配)。
        # 云侧不复切边侧已定的 chunk;预算钳制(min)仍生效,内存上界不变。
        saved_long_prefill_threshold = (
            self.scheduler_config.long_prefill_token_threshold
        )
        saved_mamba_align = self.need_mamba_block_aligned_split
        if token_budget_cap is not None:
            self.max_num_scheduled_tokens = min(
                saved_token_budget, max(0, token_budget_cap)
            )
            self.scheduler_config.long_prefill_token_threshold = 0
            self.need_mamba_block_aligned_split = False
        try:
            out = super().schedule()
        finally:
            self.max_num_scheduled_tokens = saved_token_budget
            self.scheduler_config.long_prefill_token_threshold = (
                saved_long_prefill_threshold
            )
            self.need_mamba_block_aligned_split = saved_mamba_align
            # 按原队列拼回:running 存活者接尾,waiting 被抢占者排队尾,
            # skipped 被跳过者排队首
            post = (self.running, self.waiting, self.skipped_waiting)
            self.running, self.waiting, self.skipped_waiting = saved
            self.running += post[0]
            self.waiting.extend(post[1])
            self.skipped_waiting.prepend_requests(post[2])
        return out
