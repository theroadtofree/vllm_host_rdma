"""云侧 EngineCore 子类:覆写 socket IO 线程入口,PRE_OUT 循环独立成线程,
边侧预告与步内元数据经 input_queue 走原生分发;仅 prefill_only 云角色启用。"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.engine import EngineCoreRequestType, FinishReason
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
    LwdControlPublisher,
)
from vllm.v1.lwd_control.control_communication.lwd_control_subscriber import (
    LwdControlSubscriber,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LWD_NOT_FINISHED,
    LwdAbortNotify,
    LwdC2eNotify,
    LwdHelloNotify,
    LwdRangeNotify,
    LwdRequestNotify,
    lwd_decode_notify,
    lwd_encode_cloud_notify,
)
from vllm.v1.lwd_control.control_communication.lwd_zmq_factory import (
    create_cloud_zmq_channel,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import LwdConfig
from vllm.v1.lwd_debug import LwdDebug
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_phase_scheduler import (
    LwdCloudPhaseScheduler,
)
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.outputs import LwdC2eMeta, ModelRunnerOutput

logger = init_logger(__name__)

# PRE_OUT recv 超时拍:仅作关停响应上限(HELLO 首拍一次,无重发)
LWD_PRE_OUT_RECV_TIMEOUT_MS = 5000

# 步元数据队满重试小睡:元数据不可丢(边侧据此预挂精确尺寸 recv)
_LWD_C2E_SEND_RETRY_SLEEP_S = 0.05


class LwdCloudEngineCore(EngineCoreProc):
    """云 PO 引擎:覆写 socket IO 线程入口,其余全走原生。"""

    def __init__(self, *args, **kwargs) -> None:
        # 调度器自注入须赶在 super() 之前(与边侧 LwdEdgeEngineCore 同款):
        # super 构建 self.scheduler 时一次性消费 scheduler_cls,后设无效。
        # 依赖 lwd_serve_guard 注入不可靠——guard 只在 headless serve 入口
        # 执行,完整 serve 路径的 EngineCore 子进程不经 guard,缺注入会让
        # IO 线程把 RangeNotify 写进裸 AsyncScheduler 而崩溃。
        vllm_config = kwargs["vllm_config"]
        vllm_config.scheduler_config.scheduler_cls = LwdCloudPhaseScheduler
        super().__init__(*args, **kwargs)
        # 通信面构造期装配(与边侧 _lwd_setup_zmq 同构):subscriber 自带
        # 搬运线程后,socket 不再受泵线程的 zmq 单线程亲和约束;建站失败
        # = 构造抛异常,与边侧同一错误路径。传输模式经 zmq_mode 选择,
        # 边云两侧必须一致。
        self._lwd_setup_zmq_router_dealer()
        # self._lwd_setup_zmq()

    def _lwd_setup_zmq(self) -> None:
        """构造期装配通信面:PRE_OUT bind 收边(内部搬运线程)+ POST_OUT
        connect 边,承载首拍 HELLO 通告与步内元数据。"""
        config = LwdConfig.from_env_and_config(self.vllm_config)
        self._lwd_subscriber = LwdControlSubscriber(
            config.lwd_pre_out_endpoint(), bind=True,
            decoder=lwd_decode_notify,
        )
        master_addr = self.vllm_config.parallel_config.master_addr
        self._lwd_post_out = LwdControlPublisher(
            f"tcp://{master_addr}:{config.post_out_port}",
            bind=False,
            encoder=lwd_encode_cloud_notify,
        )
        self._lwd_hello = LwdHelloNotify(
            pre_out_host=config.pre_out_host, pre_out_port=config.pre_out_port
        )
        # 首拍即通告(边侧可能已 bind 等待)
        self._lwd_announce()
        # 门池:元数据查重与暂存,到达即构建放行;仅本泵线程独占
        self._lwd_gate_pending: dict[str, LwdRequestNotify] = {}
        # UP 链 seqno 登记(边→云→云 worker 的最后一跳,§9.12 接缝):
        # rid -> [chunk 序 seqno 列表],RangeNotify 到达即登记;调度器
        # 出 prefill 批时取快照挂 SO.lwd_up_seqnos 随批下发云 worker
        # (数据面 UP recv 配对键,与边侧 SO.lwd_batch.seqno 同源同值)。
        # registry 引用交付调度器(泵线程登记 / 主循环读,dict 赋值原子)。
        self._lwd_seqno_registry: dict[str, list[int]] = {}
        self.scheduler.lwd_seqno_registry = self._lwd_seqno_registry
        threading.Thread(
            target=self._lwd_pre_out_loop, daemon=True, name="lwd-pre-out"
        ).start()
        logger.info(
            "[Lwd] cloud engine assembled: PRE_OUT bind %s, POST_OUT announce -> "
            "%s:%s via master %s",
            config.lwd_pre_out_endpoint(),
            config.pre_out_host,
            config.pre_out_port,
            master_addr,
        )

    def _lwd_pre_out_loop(self) -> None:
        """PRE_OUT 泵线程:泵取搬运线程攒下的通知逐条分发(_lwd_dispatch);
        阻塞拍仅作关停响应上限,关停(closed)退出。两种传输模式下
        _lwd_subscriber 均提供 recv_batch,循环逐行同构。"""
        while not self._lwd_subscriber.closed:
            for msg in self._lwd_subscriber.recv_batch(
                timeout_ms=LWD_PRE_OUT_RECV_TIMEOUT_MS
            ):
                self._lwd_dispatch(msg)

    def _lwd_setup_zmq_router_dealer(self) -> None:
        """full_duplex 模式装配:单条 ROUTER/DEALER 连接承载双向控制面,
        云 DEALER 以显式 identity 连边绑定的端口;不再有 POST_OUT 发布面
        与云端点发现(业务层 HELLO 通告保留,作为边侧"云全量就绪"屏障)。

        实例号三处同源(config.instance_id):DEALER identity、业务 HELLO
        戳、c2e 戳——任何一处单独取值即未来串线隐患。"""
        config = LwdConfig.from_env_and_config(self.vllm_config)
        master_addr = self.vllm_config.parallel_config.master_addr
        self._lwd_instance_id = config.instance_id
        channel = create_cloud_zmq_channel(
            master_addr=master_addr,
            dp_rank=0,
            pre_out_port=config.pre_out_port,
            name=f"lwd-cloud-inst{config.instance_id}",
            instance_id=config.instance_id,
        )
        # 收发同通道:泵线程/发布方共用同一对象
        self._lwd_subscriber = channel
        self._lwd_post_out = channel
        self._lwd_hello = LwdHelloNotify(
            pre_out_host=config.pre_out_host,
            pre_out_port=config.pre_out_port,
            instance_id=config.instance_id,
        )
        # 业务层就绪通告:通道 HELLO/WELCOME 只证明 TCP 连通,本通告
        # 证明云全量初始化(权重/KV/图编译)完成,边侧以此放行
        self._lwd_announce()
        # 门池与 registry:与 push_pull 装配同款,泵线程独占
        self._lwd_gate_pending: dict[str, LwdRequestNotify] = {}
        self._lwd_seqno_registry: dict[str, list[int]] = {}
        self.scheduler.lwd_seqno_registry = self._lwd_seqno_registry
        threading.Thread(
            target=self._lwd_pre_out_loop, daemon=True, name="lwd-pre-out"
        ).start()
        logger.info(
            "[Lwd] cloud engine assembled (full_duplex): DEALER connect -> "
            "%s:%s (identity=%d)",
            master_addr,
            config.pre_out_port,
            config.instance_id,
        )

    def _lwd_announce(self) -> None:
        """首拍 HELLO 通告一次;队满不重试,由边侧等待超时 fail-fast 兜底。"""
        self._lwd_post_out.publish(self._lwd_hello)
        logger.info(
            "[Lwd][cloud] HELLO announced: pre_out=%s:%s",
            self._lwd_hello.pre_out_host, self._lwd_hello.pre_out_port,
        )

    def shutdown(self) -> None:
        """两面关停后走原生(幂等;装配失败路径两面可能未建,容忍缺省)。"""
        subscriber = getattr(self, "_lwd_subscriber", None)
        if subscriber is not None:
            subscriber.shutdown()
        publisher = getattr(self, "_lwd_post_out", None)
        if publisher is not None:
            publisher.shutdown()
        super().shutdown()

    def _lwd_dispatch(self, msg) -> None:
        """PRE_OUT 三类分派(本 IO 线程):元数据转 Request / abort 终结 /
        范围预告登记 seqno。"""
        if isinstance(msg, LwdRangeNotify):
            # 范围预告:登记 UP 链 seqno(数据面配对键,§9.12)。幂等去重
            # 按"单调性"(seqno 不大于该请求已登记尾号即重复,边侧队满
            # 重试天然产生重复预告,重试复用同一号不产生新登记)。
            seqnos = self._lwd_seqno_registry.setdefault(msg.request_id, [])
            if not seqnos or msg.seqno > seqnos[-1]:
                seqnos.append(msg.seqno)
            logger.info(
                "[Lwd][cloud-ctrl] RangeNotify req=%s num=%s seqno=%s",
                msg.request_id, msg.num_tokens, msg.seqno,
            )
            # 每条预告都整条入队(重复预告即重复点名,剔除-调度-拼回幂等,
            # 无副作用;PRE_OUT 只 append,调度主线程单独 popleft,deque
            # 单操作原子;预告自带 seqno,出批时作 UP 链配对号)
            self.scheduler.prefill_notify_queue.append(msg)
            return
        if isinstance(msg, LwdAbortNotify):
            logger.info("[Lwd][cloud-ctrl] AbortNotify req=%s", msg.request_id)
            self._lwd_gate_pending.pop(msg.request_id, None)
            # 双队列与原生 ABORT 同款:eager 处理 + 保持 input_queue 次序
            self.aborts_queue.put_nowait([msg.request_id])
            self.input_queue.put_nowait((EngineCoreRequestType.ABORT, [msg.request_id]))
            return
        rid = msg.request_id
        if rid in self._lwd_gate_pending:
            logger.warning("[Lwd] duplicate request metadata %s ignored", rid)
            return
        logger.info("[Lwd][cloud-ctrl] RequestNotify req=%s", rid)
        self._lwd_gate_pending[rid] = msg
        self._lwd_promote(rid)

    def _lwd_promote(self, request_id: str) -> None:
        """过门:门池取 wire,转 Request 投 input_queue 走原生 ADD 分发。"""
        wire = self._lwd_gate_pending.pop(request_id, None)
        if wire is not None:
            request = self._lwd_build_request(wire)
            self.input_queue.put_nowait((EngineCoreRequestType.ADD, (request, 0)))
            logger.info("[Lwd] cloud request %s admitted via gate", request_id)

    def _lwd_build_request(self, wire: LwdRequestNotify) -> Request:
        """请求构建(唯一建请求点,Request/SamplingParams 留 L3)。

        采样参数自边侧 LwdRequestNotify 透传(采样核/惩罚/EOS 策略/
        min_tokens),云侧按客户端真实参数采样,不再落默认值。"""
        sampling_params = SamplingParams(
            max_tokens=wire.max_tokens,
            temperature=wire.temperature,
            top_p=wire.top_p,
            top_k=wire.top_k,
            min_p=wire.min_p,
            seed=wire.seed,
            repetition_penalty=wire.repetition_penalty,
            presence_penalty=wire.presence_penalty,
            frequency_penalty=wire.frequency_penalty,
            ignore_eos=wire.ignore_eos,
            stop_token_ids=list(wire.stop_token_ids),
            min_tokens=wire.min_tokens,
        )
        # eos_token_id 非构造入参,走原生回填入口(同边侧前端
        # input_processor):设 _eos_token_id 并计入 _all_stop_token_ids
        # 供 min_tokens 判定;云侧无客户端 generation_config,传空。
        sampling_params.update_from_generation_config({}, wire.eos_token_id)
        LwdDebug.cloud_request_admitted(wire, sampling_params)  # [lwd-debug]
        # 不传真实 ids 也不造占位:按原生 prompt-embeds 语义挂零缓冲,
        # 行数即 prompt 长度(UP chunk 注入直接写该缓冲的对应窗口);
        # ids=None 时 input_batch 自动把 prompt 段 is_token_ids 置 False,
        # M-RoPE 走纯文本直通构造(与扫描结果逐值一致)。
        prompt_ids: list[int] | None = None
        prompt_embeds = torch.zeros(
            wire.num_prompt_tokens,
            self.vllm_config.model_config.get_hidden_size(),
            dtype=self.vllm_config.model_config.dtype,
        )
        logger.info(
            "[Lwd][cloud-ctrl] build request req=%s prompt=%d ids=none+embeds_buf",
            wire.request_id, wire.num_prompt_tokens,
        )
        local_hasher = self.request_block_hasher
        if local_hasher is None:
            # prefix caching 未启用:请求不挂 hasher,整链机制不激活
            request = Request(
                request_id=wire.request_id,
                prompt_token_ids=prompt_ids,
                prompt_embeds=prompt_embeds,
                sampling_params=sampling_params,
                pooling_params=None,
            )
            # 占位 embeds 不随 SO 上传运输层(NewRequestData 只带形状,
            # worker 本地分配):35MB 零 buffer 走 MQ overflow 通道实测
            # 单程 240ms+,是 prefill SO 开工延迟的主因。
            request.lwd_embeds_placeholder = True
            return request
        hash_block_size = resolve_kv_cache_block_sizes(
            self.scheduler.kv_cache_config, self.vllm_config
        )[1]

        def block_hasher(request: Request) -> list[bytes]:
            # prompt 首建用边侧预告链,续算/缺链/长度不符回退本地(fail-open)
            if len(request.block_hashes) == 0 and request.num_output_tokens == 0:
                expected = request.num_prompt_tokens // hash_block_size
                if len(wire.block_hashes) == expected:
                    return wire.block_hashes
            return local_hasher(request)

        request = Request(
            request_id=wire.request_id,
            prompt_token_ids=prompt_ids,
            prompt_embeds=prompt_embeds,
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=block_hasher,
        )
        request.lwd_embeds_placeholder = True  # 同上:占位 embeds 不上 MQ
        return request
    
    def step_with_batch_queue(self):
        """步骤执行时长打点(开始执行→执行结束;不含引擎空等)。"""
        _t0 = time.monotonic()
        out = super().step_with_batch_queue()
        logger.info(
            "[Lwd][perf] cloud-step exec=%.2fms",
            (time.monotonic() - _t0) * 1000,
        )
        return out

    def step(self):
        """同步步路径同款打点。"""
        _t0 = time.monotonic()
        out = super().step()
        logger.info(
            "[Lwd][perf] cloud-step exec=%.2fms",
            (time.monotonic() - _t0) * 1000,
        )
        return out

    def lwd_handle_model_output(
        self,
        model_output: ModelRunnerOutput,
        engine_core_outputs: dict[int, EngineCoreOutputs],
    ) -> ModelRunnerOutput:
        """rank-replay:解码 worker 主流末尾 pinned 物化的步 meta(就绪由
        响应入队处的 event synchronize 保证),组 c2e 通告经 POST_OUT
        先于 hidden 发边;finish 码取自 engine_core_outputs。

        pinned 布局:[ranks(各调度段行)..., counts(accepted/请求)...,
        seg_lens(段长/请求)...];top_id_ths 按段长切,被拒行一并携带,
        边侧按 num_accepted 取有效前缀。"""
        # [Lwd][perf] cloud-step dt 已由 exec 时长替代(见 step_with_batch_queue
        # 覆写):开始执行→执行结束,不含无请求的空等。
        carrier = getattr(model_output, "lwd_down_carrier", None)
        if carrier is not None:
            pinned, req_ids, hidden_numel, seqno = carrier
            n_req = len(req_ids)
            vals = pinned.tolist()
            counts = vals[-2 * n_req : -n_req]
            seg_lens = vals[-n_req:]
            ranks_flat = vals[: -2 * n_req]
            top_id_ths: list[list[int]] = []
            off = 0
            for seg_len in seg_lens:
                top_id_ths.append(ranks_flat[off : off + seg_len])
                off += seg_len
            from vllm.v1.outputs import LwdC2eMeta

            meta = LwdC2eMeta(
                hidden_num_elements=hidden_numel,
                top_id_ths=top_id_ths,
                num_accepted_tokens=list(counts),
                req_ids=list(req_ids),
                down_seqno=seqno,
            )
            logger.info(
                "[Lwd][cloud-ctrl] publish c2e(rank-replay): reqs=%s "
                "rows=%d seqno=%d",
                meta.req_ids, off, seqno,
            )
            LwdDebug.cloud_step(self.scheduler, meta, engine_core_outputs)  # [lwd-debug]
            _t = time.monotonic()
            self._lwd_publish_c2e(
                meta, self._lwd_c2e_finish_reasons(meta, engine_core_outputs)
            )
            # [Lwd][perf] 云侧 LWD 税:finish 码推导 + ZMQ publish
            logger.info(
                "[Lwd][perf] publish reqs=%d dur=%.2fms",
                len(meta.req_ids), (time.monotonic() - _t) * 1000,
            )
        else:
            logger.info(
                "[Lwd][cloud-ctrl] handle_model_output: no down carrier this step"
            )
        return model_output

    @staticmethod
    def _lwd_c2e_finish_reasons(
        meta: LwdC2eMeta,
        engine_core_outputs: dict[int, EngineCoreOutputs],
    ) -> list[int]:
        """req_ids 对齐的逐请求完成码:本步任一 EngineCoreOutputs 里带
        finish_reason 的输出取其码;仅进 finished_requests 的缺口按 ABORT
        兜底;其余 LWD_NOT_FINISHED(边侧保持 awaiting,由后续步通告收口)。"""
        reasons: dict[str, int] = {}
        for outputs in engine_core_outputs.values():
            for out in outputs.outputs:
                if out.finish_reason is not None:
                    reasons.setdefault(out.request_id, int(out.finish_reason))
            for request_id in outputs.finished_requests or ():
                reasons.setdefault(request_id, int(FinishReason.ABORT))
        return [reasons.get(request_id, LWD_NOT_FINISHED)
                for request_id in meta.req_ids]

    def _lwd_publish_c2e(self, meta: LwdC2eMeta, finish_reasons: list[int]) -> None:
        """步元数据(含逐请求 finish_reasons 完成码)发边:队满小睡重试
        (元数据不可丢),关停(closed)退出。instance_id 与 DEALER identity
        同源盖章,边侧据此分发到对应实例的 UNEMBED/数据面路径。"""
        notify = LwdC2eNotify(
            hidden_num_elements=meta.hidden_num_elements,
            top_id_ths=meta.top_id_ths,
            num_accepted_tokens=meta.num_accepted_tokens,
            req_ids=meta.req_ids,
            finish_reasons=finish_reasons,
            down_seqno=meta.down_seqno,
            instance_id=getattr(self, "_lwd_instance_id", 0),
        )
        while not self._lwd_post_out.closed:
            if self._lwd_post_out.publish(notify):
                logger.info(
                    "[Lwd][cloud-ctrl] publish C2eNotify reqs=%d down_seqno=%s "
                    "finish=%s hidden_elems=%s",
                    len(notify.req_ids),
                    notify.down_seqno,
                    finish_reasons,
                    notify.hidden_num_elements,
                )
                return
            time.sleep(_LWD_C2E_SEND_RETRY_SLEEP_S)
