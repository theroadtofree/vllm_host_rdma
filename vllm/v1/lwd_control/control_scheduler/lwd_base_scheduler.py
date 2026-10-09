"""Lwd 边/云调度器公共基类:收敛两侧共享的调度语义。

仍继承 AsyncScheduler,原生调度行为与 isinstance 判定不变;共享逻辑
随重构逐步上提至此类。
"""

from __future__ import annotations

from collections.abc import Iterable

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class LwdBaseScheduler(AsyncScheduler):
    """边/云调度器公共基类(AsyncScheduler 子类)。

    额外承载云侧 abort 语义:边侧 abort 后不再把该请求未处理完的 chunk 发给
    云侧,但**已经预告、数据已在线上**的 chunk 云侧必须先算完,再终结整个
    请求(``finish_requests`` + ``_lwd_settle_deferred_aborts``)。边侧调度器
    没有预告队列,本语义对其完全透明。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 云侧延迟终结台账:request_id -> 终结状态。请求在预告队列里还有
        # 未消费的 chunk 时先进本表,等这些 chunk 算完再真正 finish。
        self._lwd_abort_wait: dict[str, RequestStatus] = {}

    def _lwd_new_queue(self, reqs: list[Request]) -> RequestQueue:
        """新建调度策略队列并装入 reqs。"""
        queue = create_request_queue(self.policy)
        for req in reqs:
            queue.add_request(req)
        return queue

    def _lwd_schedule_for_visible_reqs(self, req_ids: list[str]) -> SchedulerOutput:
        """把 req_ids 指定的请求从三队列剔除、单独调度,步后按原队列拼回。

        waiting/skipped 来源的请求走原生准入窗口(allocate/状态迁移/
        记账一样不少),running 来源的走续跑。拼回:仍被调度的接在隐藏
        running 之后,被抢占的排 waiting 尾部,被跳过的排 skipped 队首。"""
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
        try:
            out = super().schedule()
        finally:
            # 按原队列拼回:running 存活者接尾,waiting 被抢占者排队尾,
            # skipped 被跳过者排队首
            post = (self.running, self.waiting, self.skipped_waiting)
            self.running, self.waiting, self.skipped_waiting = saved
            self.running += post[0]
            self.waiting.extend(post[1])
            self.skipped_waiting.prepend_requests(post[2])
        return out

    # ------------------------------------------------------------------ #
    # 云侧 abort:先把未处理完的 chunk 算完,再终结请求                       #
    # ------------------------------------------------------------------ #
    def _lwd_notify_queue(self):
        """云侧 prefill 预告队列;边侧调度器没有该队列(返回 None)。"""
        return getattr(self, "prefill_notify_queue", None)

    def _lwd_has_queued_chunk(self, request_id: str) -> bool:
        """该请求在预告队列里是否还有未消费的 chunk(批量预告读 items)。"""
        for notify in self._lwd_notify_queue() or ():
            for item in notify.items or (notify,):
                if item.request_id == request_id:
                    return True
        return False

    def _lwd_drop_queued_chunks(self, request_id: str) -> int:
        """摘除队列中该请求的预告条目,返回摘除条数。

        只用于请求对象已不存在、无从消费的场景(关停期 ADD 未落地/重复
        abort/已被其它路径终结):条目留着会让混排调度把队首永久 defer。
        在主线程调用,不与 IO 线程并发改预告对象。"""
        queue = self._lwd_notify_queue()
        if not queue:
            return 0
        kept = queue.__class__()
        removed = 0
        for notify in queue:
            if notify.items:
                items = [
                    item
                    for item in notify.items
                    if item.request_id != request_id
                ]
                removed += len(notify.items) - len(items)
                if items:
                    notify.items = items
                    kept.append(notify)
            elif notify.request_id != request_id:
                kept.append(notify)
            else:
                removed += 1
        if removed:
            self.prefill_notify_queue = kept
        return removed

    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: RequestStatus,
    ) -> list[tuple[str, int]]:
        """云侧 finish 入口:abort 时先把未消费的 chunk 算完,再终结请求。

        只有 abort(``FINISHED_ABORTED``)且存在预告队列时才延迟;其余原生
        路径(自然停止/长度截断,以及 ``request_ids is None`` 的关停收尾)与
        边侧调度器完全不变。"""
        if (
            finished_status is not RequestStatus.FINISHED_ABORTED
            or request_ids is None
            or self._lwd_notify_queue() is None
        ):
            return super().finish_requests(request_ids, finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        immediate: list[str] = []
        for request_id in request_ids:
            if request_id not in self.requests:
                # 请求已不存在:摘除残留预告,避免队首被永久 defer
                removed = self._lwd_drop_queued_chunks(request_id)
                if removed:
                    logger.info(
                        "[Lwd][sched] abort req=%s gone; dropped %d stale "
                        "notify item(s)",
                        request_id,
                        removed,
                    )
                continue
            if self._lwd_has_queued_chunk(request_id):
                self._lwd_abort_wait.setdefault(request_id, finished_status)
                logger.info(
                    "[Lwd][sched] abort deferred req=%s: waiting for queued "
                    "chunk(s) to be computed",
                    request_id,
                )
                continue
            immediate.append(request_id)
        if not immediate:
            return []
        return super().finish_requests(immediate, finished_status)

    def _lwd_settle_deferred_aborts(self) -> None:
        """把"预告已算完"的延迟 abort 真正落地(每步 schedule() 开头调用)。

        必须在取可见集之前调用:此刻上一步的批已回收,释放该请求的 KV 是
        安全的;若晚于 `_lwd_schedule_for_visible_reqs`,已终结的请求会被
        当成可调度对象。"""
        if not self._lwd_abort_wait:
            return
        for request_id in list(self._lwd_abort_wait):
            if request_id not in self.requests:
                # 已被别的路径终结:撤下台账即可
                del self._lwd_abort_wait[request_id]
                continue
            if self._lwd_has_queued_chunk(request_id):
                continue
            status = self._lwd_abort_wait.pop(request_id)
            logger.info(
                "[Lwd][sched] deferred abort settled req=%s (queued chunk(s) "
                "consumed)",
                request_id,
            )
            super().finish_requests(request_id, status)
