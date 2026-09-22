"""云侧相位调度器:工作纯相位批次。prefill 步只算 prompt 工作(prefill 首块 +
尾巴),decode 步只算已完结请求的 1-token 采样;prefill 批最多一个请求。"""

from __future__ import annotations

from collections import deque

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.core.sched.output import (
    LwdBatch,
    LwdBatchType,
    LwdEmbedBatch,
    SchedulerOutput,
)
from vllm.v1.core.sched.scheduler import Scheduler as _NativeScheduler
from vllm.v1.lwd_control.control_communication.lwd_id_adapter import parse_edge_id
from vllm.v1.lwd_control.control_communication.lwd_notify import LwdRangeNotify
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
)
from vllm.v1.request import RequestStatus

logger = init_logger(__name__)

_LWD_PHASE_PREFILL_FIRST = "prefill_first"
_LWD_PHASE_DECODE_FIRST = "decode_first"


class LwdCloudPhaseScheduler(LwdBaseScheduler):
    """工作纯相位批次策略;相位(prefill_first/decode_first)构造期自解析。
    前置约束:不兼容 spec decode(eagle 会 shift num_computed_tokens,纯度判据失真)。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 收尾快照:request_id -> (prompt_tokens, completion_tokens)。
        # 释放链唯一收口点是 ``_free_request``(常规终结 update_from_output 直接
        # 调它,**不经 finish_requests**);而 LWD 的 finish/usage 结算发生在其后
        # 的 lwd_handle_model_output —— 那时请求已从 self.requests 摘除(旧实现
        # 因此从不结算、预留泄漏)。故在 _free_request 覆写里抓最小快照,由引擎
        # 侧消费(消费即摘除);超上限按插入序兜底淘汰残留。
        self.lwd_finished_records: dict[str, tuple[int, int]] = {}
        # 块哈希粒度(与边侧链同域):用于生成段块登记窗口的 token 换算
        self._lwd_hash_block_size = resolve_kv_cache_block_sizes(
            self.kv_cache_config, self.vllm_config
        )[1]
        # 生成段块登记读数:request_id -> 已按边侧链登记的满块数(验证多轮命中边界是否
        # 含生成段;每登记一个新满块打一条,量级 = 每 block_size 生成 token 一次)
        self.lwd_gen_registered_blocks: dict[str, int] = {}
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
        logger.info(
            "[Lwd] cloud phase scheduler: single-request prefill batches "
            "enforced (edge/cloud chunk stream stays per-request contiguous)"
        )

    def _free_request(self, request, delay_free_blocks: bool = False):
        """收尾快照(见 ``lwd_finished_records``):挂在**释放链唯一收口点**。

        常规终结(update_from_output 里 check_stop 命中)是直接
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

    def _lwd_capped_cache_tokens(self, request) -> int:
        """生成段块登记窗口:封顶到「边侧 HMAC 链已覆盖」的 token 数。

        生成段 token 的 HMAC 只有持密钥的边能算(``LwdGenChainNotify`` 全量
        替换上报);链未到达前不得登记这些块——否则会以 local_hasher 键入池,
        而块一旦登记无法按 HMAC 重登记。链缺失 → 只登记 prompt 段(fail-safe
        降级:生成段块不入池);链到达后由后续步自动追平。"""
        num_computed = (
            request.num_computed_tokens - request.num_output_placeholders
        )
        prompt_end = request.num_prompt_tokens
        if num_computed <= prompt_end:
            return num_computed
        chains = getattr(self, "lwd_edge_chains", None)
        if chains is None:
            # 协调未启用:不封顶,保持原生登记行为(生成段块照常按本地哈希入池)
            return num_computed
        entry = chains.get(request.request_id)
        if not entry:
            return prompt_end
        block_size, chain = entry
        if int(block_size) != self._lwd_hash_block_size:
            # 链粒度 ≠ 本地 hash 粒度(ratio≠1):链索引与云内 block_hashes
            # 错位,不采用(生成段块不入池,fail-safe)
            return prompt_end
        covered = len(chain) * self._lwd_hash_block_size
        return min(num_computed, max(prompt_end, covered))

    def _lwd_log_gen_registration(self, request, capped_tokens: int) -> None:
        """生成段块按边侧链的登记进度读数(每新登记一个满块打一条)。

        prompt 段之后才是生成段;水位按 request_id 记。验证多轮复用时看这条:
        下一条请求的 probe 命中边界应能覆盖到这里登记的块数。"""
        prompt_end = request.num_prompt_tokens
        if capped_tokens <= prompt_end:
            return
        covered = capped_tokens // self._lwd_hash_block_size
        if covered <= self.lwd_gen_registered_blocks.get(request.request_id, 0):
            return
        if len(self.lwd_gen_registered_blocks) >= 4096:
            for stale in list(self.lwd_gen_registered_blocks)[:512]:
                self.lwd_gen_registered_blocks.pop(stale, None)
        self.lwd_gen_registered_blocks[request.request_id] = covered
        logger.debug(
            "[Lwd][coord] gen-blocks registered req=%s covered_blocks=%d "
            "(prompt_blocks=%d)",
            request.request_id, covered,
            prompt_end // self._lwd_hash_block_size,
        )

    def _update_request_with_output(self, request, new_token_ids):
        """与 AsyncScheduler 同款,仅把生成段块的 cache 窗口封顶。

        直接调 ``Scheduler`` 的实现(而非 super():super 是 AsyncScheduler,
        会按未封顶窗口再登记一次),再按 async 语义补 async_tokens_to_discard
        与 num_output_placeholders 两项记账。

        **升级检查点**:本覆写入参/语义与
        ``vllm/v1/core/sched/async_scheduler.py:AsyncScheduler._update_request_with_output``
        及 ``vllm/v1/core/sched/scheduler.py:Scheduler._update_request_with_output``
        对偶——上游若改这两处(async 记账项/停止判定),此处必须同步。"""
        if request.async_tokens_to_discard > 0:
            request.async_tokens_to_discard -= 1
            return [], False
        status_before_update = request.status
        new_token_ids, stopped = _NativeScheduler._update_request_with_output(
            self, request, new_token_ids
        )
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0
        if status_before_update == RequestStatus.RUNNING:
            capped_tokens = self._lwd_capped_cache_tokens(request)
            self.kv_cache_manager.cache_blocks(request, capped_tokens)
            self._lwd_log_gen_registration(request, capped_tokens)
        return new_token_ids, stopped

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
        """纯 prefill 步:prefill_notify_queue 有预告则取队首 msg,单独
        调度其请求(按公告量钳制本步预算,照单执行;按原队列归位,
        waiting/skipped 来源走原生准入);没有则空集进窗口,等价空步,
        三队列原样保留。"""
        q = self.prefill_notify_queue
        if not q:
            return self._lwd_schedule_for_visible_reqs([])
        # peek 而非 popleft:队首请求可能尚未准入(RequestNotify 走
        # input_queue ADD,RangeNotify 直接进本队列,两者有时序 race)。
        # 直接 pop+丢弃会丢掉该 chunk,造成 UP seqno 空洞 → 数据面 recv
        # 端按序 hold → 云侧 prefill 卡死、decode 停摆(边侧空闲、云侧
        # 空转的根因)。请求未准入时本步空排,等下一轮 ADD 处理后再消费。
        notify = q[0]
        if notify.request_id not in self.requests:
            return self._lwd_schedule_for_visible_reqs([])
        q.popleft()
        logger.info(
            "[Lwd][cloud-sched] prefill notify req=%s seqno=%s num=%s",
            notify.request_id, notify.seqno, notify.num_tokens,
        )
        # 边侧是 prefill 切块权威,云侧按预告 chunk 钳制本步预算执行
        # (只收紧不放宽原生上限):num_scheduled_tokens == 公告量,边云
        # 锁步;mamba align 的块对齐复切在原生路径内照常生效(边侧
        # chunk 块对齐时为 no-op)。
        out = self._lwd_schedule_for_visible_reqs(
            [notify.request_id],
            token_budget_cap=notify.num_tokens,
        )
        if not out.num_scheduled_tokens:
            # 未实际准入(典型 KV 压力空步):预告塞回队首原位,decode
            # 泄压后重新点名;本步不挂 lwd_batch,不向 worker 预告配对号
            self.prefill_notify_queue.appendleft(notify)
            return out
        # UP 链 seqno 随批下发云 worker(§9.12 数据面接缝):批配对号直接
        # 取点名预告自带的 seqno(与边侧 EMBED 批派发号同源同值),worker
        # 的 UP recv 以此配对边侧发来的 embeds 张量。batch_meta 承载
        # worker 的 recv 尺寸与注入切行信息:req_ids 取预告请求(单请求
        # 批),token_ids 为占位列表——长度必须等于边侧实际发送的 chunk
        # token 数(= RangeNotify.num_tokens),recv numel 才能与边侧
        # isend 严格相等(HCCL P2P 要求两端 numel 匹配)。
        edge_id, _ = parse_edge_id(notify.request_id)
        out.lwd_batch = LwdBatch(
            batch_type=LwdBatchType.LWD_EMBED,
            seqno=notify.seqno,
            batch_meta=LwdEmbedBatch(
                req_ids=[notify.request_id],
                token_ids=[[0] * notify.num_tokens],
                edge_id=edge_id,
                token_offsets=[notify.offset],
                has_mrope=notify.has_mrope,
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
