# SPDX-License-Identifier: Apache-2.0
"""集中式纯相位调度器(prefill_only 对照实验臂)。

仿 LWD prefill_only 云侧相位调度的调度语义,但不依赖任何 LWD 数据面:
每步只排程一种相位——prefill 步单请求(不组批),decode 步收全部
decode 态请求;相位优先级与 prefill_only 相同(有 waiting 或 prefill
尾巴即 prefill);无禁连续 prefill 不变量(与 0920 实验分支对齐)。

与 LwdCloudPhaseScheduler 的唯一差异是 prefill 的请求来源:集中式没有
边侧 RangeNotify 点名,prefill 步按「running 中的 prefill 尾巴优先、
否则 waiting 队首」自选(对齐边云链路 chunk 按请求连续的形态)。

经 SchedulerConfig.scheduler_cls 注入使用,仅用于对照实验。
"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


class CentralizedPhaseScheduler(LwdBaseScheduler):
    """集中式纯相位调度(prefill_only 对照实验臂,非 LWD 部署使用)。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # One-shot 翻转:某相位空步而另一相位有活时,强制下一步走后者;
        # 双标志显式定向,按偏好取反会错翻,造成空步死循环。
        self._force_prefill_once: bool = False
        self._force_decode_once: bool = False
        logger.info(
            "[phase-sched] centralized phase scheduler: single-request "
            "prefill batches, prefill_first priority (experiment arm)"
        )

    # ------------------------------------------------------------------ #
    # Phase predicates(与 LWD 云侧同语义)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_decode(request: Request) -> bool:
        """prompt 已算完 = decode 态(可采样);未算完的是 prefill 尾巴。"""
        return request.num_computed_tokens >= request.num_prompt_tokens

    def _has_prefill_tails(self) -> bool:
        return any(not self._is_decode(r) for r in self.running)

    def _has_decode_ready(self) -> bool:
        return any(self._is_decode(r) for r in self.running)

    # ------------------------------------------------------------------ #
    # Phase primitives(复用 LwdBaseScheduler 的可见集交换)
    # ------------------------------------------------------------------ #
    def _pick_prefill_request(self) -> list[str]:
        """prefill 步点名:running 中的 prefill 尾巴优先(chunked prefill
        续跑,对齐边云链路 chunk 按请求连续),否则 waiting 队首。"""
        for req in self.running:
            if not self._is_decode(req):
                return [req.request_id]
        if self.waiting:
            return [self.waiting.peek_request().request_id]
        return []

    def _collect_decode_requests(self) -> list[str]:
        """收集所有 decode 态(prompt 已算完)请求的 req_id。"""
        return [
            req.request_id
            for queue in (self.running, self.waiting, self.skipped_waiting)
            for req in queue
            if self._is_decode(req)
        ]

    def _schedule_pure_prefill(self) -> SchedulerOutput:
        return self._lwd_schedule_for_visible_reqs(self._pick_prefill_request())

    def _schedule_pure_decode(self) -> SchedulerOutput:
        return self._lwd_schedule_for_visible_reqs(
            self._collect_decode_requests()
        )

    @staticmethod
    def _is_empty(out: SchedulerOutput) -> bool:
        return out.total_num_scheduled_tokens == 0

    # ------------------------------------------------------------------ #
    # schedule
    # ------------------------------------------------------------------ #
    def schedule(self) -> SchedulerOutput:
        # 相位偏好与 prefill_only(prefill_first)一致;无禁连续 prefill
        # 不变量(0920 实验分支语义)
        prefer_prefill = bool(self.waiting) or self._has_prefill_tails()
        if self._force_prefill_once:
            self._force_prefill_once = False
            prefer_prefill = True
        elif self._force_decode_once:
            self._force_decode_once = False
            prefer_prefill = False

        if prefer_prefill:
            out = self._schedule_pure_prefill()
            if self._is_empty(out) and self.running:
                # prefill 受 KV 压力阻塞:放行空步,下一步转 decode 泄压
                self._force_decode_once = True
            return out
        out = self._schedule_pure_decode()
        if self._is_empty(out) and (self.waiting or self._has_prefill_tails()):
            # decode 无活但有 waiting/尾巴:翻回 prefill(空步出口)
            self._force_prefill_once = True
        return out
