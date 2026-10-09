"""云侧 EngineCore 子类:覆写 socket IO 线程入口,PRE_OUT 循环独立成线程,
边侧预告与步内元数据经 input_queue 走原生分发;仅 prefill_only 云角色启用。"""

from __future__ import annotations

import os
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
    LWD_WIRE_VERSION,
    LwdAbortNotify,
    LwdC2eNotify,
    LwdHelloNotify,
    LwdRangeItem,
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
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_mixed_scheduler import (
    LwdCloudMixedScheduler,
)
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_phase_scheduler import (
    LwdCloudPhaseScheduler,
)
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.outputs import LwdC2eMeta, ModelRunnerOutput

logger = init_logger(__name__)


def _lwd_diag_token_passthrough_enabled() -> bool:
    """诊断开关:SKIP_SAMPLE(边免采样)/ DISABLE_DOWN(全直通)任一开启。"""
    return (os.environ.get("VLLM_ASCEND_LWD_EDGE_SKIP_SAMPLE") == "1"
            or os.environ.get("VLLM_ASCEND_LWD_DISABLE_DOWN") == "1")

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
        # mixed(默认)= 整体 prefill 组批混排;prefill_first/decode_first
        # = 旧相位调度器(A/B 对照与逃生通道,见 lwd_mixed_batch_design.md)。
        vllm_config = kwargs["vllm_config"]
        scheduler_name = LwdConfig.from_env_and_config(
            vllm_config
        ).scheduler_name
        vllm_config.scheduler_config.scheduler_cls = (
            LwdCloudPhaseScheduler
            if scheduler_name in ("prefill_first", "decode_first")
            else LwdCloudMixedScheduler
        )
        # [Lwd][pre-recv] 提前收只挂在 PD_mix(默认分支)上:该分支按
        # "UP embeds 收完才下发 prefill 批"的闸门调度,提示与水位都由它消费。
        # 相位调度是逃生通道,不设闸门,也就不发提示(避免无人排空的旁路 MQ)。
        self._lwd_early_recv = scheduler_name not in (
            "prefill_first", "decode_first",
        )
        logger.info(
            "[Lwd][pre-recv] armed=%s scheduler=%s",
            self._lwd_early_recv, scheduler_name,
        )
        super().__init__(*args, **kwargs)
        # 通信面构造期装配(与边侧 _lwd_setup_zmq 同构):subscriber 自带
        # 搬运线程后,socket 不再受泵线程的 zmq 单线程亲和约束;建站失败
        # = 构造抛异常,与边侧同一错误路径。传输模式经 zmq_mode 选择,
        # 边云两侧必须一致。
        self._lwd_setup_zmq_router_dealer()
        # self._lwd_setup_zmq_publisher_subscriber()

    def _lwd_setup_zmq_publisher_subscriber(self) -> None:
        """构造期装配通信面:PRE_OUT bind 收边(内部搬运线程)+ POST_OUT
        connect 边,承载首拍 HELLO 通告与步内元数据。"""
        config = LwdConfig.from_env_and_config(self.vllm_config)
        # 多 DP:每 engine 的 PRE_OUT 端口按 dp_rank 派生,避免同机 bind
        # 冲突;HELLO 通告派生后的真值,边侧据此建 per-engine 端点表。
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        pre_out_port = config.pre_out_port + dp_rank
        self._lwd_subscriber = LwdControlSubscriber(
            f"tcp://{config.pre_out_host}:{pre_out_port}", bind=True,
            decoder=lwd_decode_notify,
        )
        master_addr = self.vllm_config.parallel_config.master_addr
        self._lwd_post_out = LwdControlPublisher(
            f"tcp://{master_addr}:{config.post_out_port}",
            bind=False,
            encoder=lwd_encode_cloud_notify,
        )
        sched_config = self.vllm_config.scheduler_config
        spec_config = getattr(self.vllm_config, "speculative_config", None)
        self._lwd_hello = LwdHelloNotify(
            pre_out_host=config.pre_out_host,
            pre_out_port=pre_out_port,
            # 调度三元组供边侧互校(cap/preflight 公式的同源输入)
            max_num_batched_tokens=sched_config.max_num_batched_tokens,
            max_num_seqs=sched_config.max_num_seqs,
            num_speculative_tokens=(
                getattr(spec_config, "num_speculative_tokens", None) or 0
            ),
            # 控制面协议版本互校(mrope 直传;边侧构造期 fail-fast)
            wire_version=LWD_WIRE_VERSION,
            # 多 DP 身份与 coordinator stats 地址(边侧建端点表/路由账/
            # 唤醒通道用);dp=1 时 engine_index=0、无 coordinator 为空
            engine_index=dp_rank,
            coord_stats_addr=self._lwd_coord_stats_addr(
                config, self.frontend_stats_publish_address
            ),
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
        # [Lwd][pre-recv] 提前收:提示下发通道 = 引擎 -> 云端点 worker 通信线程。
        # 走 executor 上的旁路 MQ(不是 rpc_broadcast_mq:那条由 busy_loop 单线程
        # 消费,会被几百毫秒的 prefill 前向堵住,提示就"早"不起来了)。引擎在
        # super().__init__ 里就把 executor 建好了,故此处直接取。
        hint_mq = getattr(self.model_executor, "lwd_recv_hint_mq", None)
        if self._lwd_early_recv and hint_mq is not None:
            from vllm_ascend.distributed.lwd_comm.pre_recv import (
                register_hint_sender,
            )

            # 阻塞投递:闸门开启时提示是正确性依赖,丢一条就没人提前挂该 recv,
            # 水位不前进,闸门后的批永不下发。
            register_hint_sender(lambda hint, mq=hint_mq: mq.enqueue(hint))
            logger.info(
                "[Lwd][pre-recv] hint sender registered (PD_mix -> hint MQ)"
            )
        else:
            # armed=False 是相位调度分支的正常形态;armed=True 却没有 hint MQ
            # 就是接线坏了(提示发不出去 ⇒ 闸门永不放行),必须显式可见。
            logger.warning(
                "[Lwd][pre-recv] hint sender NOT registered armed=%s "
                "hint_mq=%s",
                self._lwd_early_recv, hint_mq is not None,
            )
        threading.Thread(
            target=self._lwd_pre_out_loop, daemon=True, name="lwd-pre-out"
        ).start()
        logger.info(
            "[Lwd] cloud engine assembled: PRE_OUT bind %s, POST_OUT announce -> "
            "%s:%s via master %s",
            f"tcp://{config.pre_out_host}:{pre_out_port}",
            config.pre_out_host,
            pre_out_port,
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
        # 多 DP:每 engine 的 PRE_OUT 端口按 dp_rank 派生,避免同机 bind
        # 冲突;DEALER identity / HELLO 通告都用派生后的真值。
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        pre_out_port = config.pre_out_port + dp_rank
        channel = create_cloud_zmq_channel(
            master_addr=master_addr,
            dp_rank=dp_rank,
            pre_out_port=pre_out_port,
            name=f"lwd-cloud-inst{config.instance_id}",
            instance_id=config.instance_id,
        )
        # 收发同通道:泵线程/发布方共用同一对象
        self._lwd_subscriber = channel
        self._lwd_post_out = channel
        sched_config = self.vllm_config.scheduler_config
        spec_config = getattr(self.vllm_config, "speculative_config", None)
        self._lwd_hello = LwdHelloNotify(
            pre_out_host=config.pre_out_host,
            pre_out_port=pre_out_port,
            instance_id=config.instance_id,
            # 调度三元组供边侧互校(cap/preflight 公式的同源输入)
            max_num_batched_tokens=sched_config.max_num_batched_tokens,
            max_num_seqs=sched_config.max_num_seqs,
            num_speculative_tokens=(
                getattr(spec_config, "num_speculative_tokens", None) or 0
            ),
            # 控制面协议版本互校(mrope 直传;边侧构造期 fail-fast)
            wire_version=LWD_WIRE_VERSION,
        )
        # 业务层就绪通告:通道 HELLO/WELCOME 只证明 TCP 连通,本通告
        # 证明云全量初始化(权重/KV/图编译)完成,边侧以此放行
        self._lwd_announce()
        # 门池与 registry:与 push_pull 装配同款,泵线程独占
        self._lwd_gate_pending: dict[str, LwdRequestNotify] = {}
        self._lwd_seqno_registry: dict[str, list[int]] = {}
        self.scheduler.lwd_seqno_registry = self._lwd_seqno_registry
        # [Lwd][pre-recv] 提前收:提示下发通道 = 引擎 -> 云端点 worker 通信线程。
        # 走 executor 上的旁路 MQ(不是 rpc_broadcast_mq:那条由 busy_loop 单线程
        # 消费,会被几百毫秒的 prefill 前向堵住,提示就"早"不起来了)。引擎在
        # super().__init__ 里就把 executor 建好了,故此处直接取。
        # 顺序:必须排在泵线程启动之前——泵线程一起来就会分发 notify 并调
        # post_irecv_hint,注册晚一步首条提示就会被丢(闸门永不放行)。
        hint_mq = getattr(self.model_executor, "lwd_recv_hint_mq", None)
        if self._lwd_early_recv and hint_mq is not None:
            from vllm_ascend.distributed.lwd_comm.pre_recv import (
                register_hint_sender,
            )

            # 阻塞投递:闸门开启时提示是正确性依赖,丢一条就没人提前挂该 recv,
            # 水位不前进,闸门后的批永不下发。
            register_hint_sender(lambda hint, mq=hint_mq: mq.enqueue(hint))
            logger.info(
                "[Lwd][pre-recv] hint sender registered (PD_mix -> hint MQ)"
            )
        else:
            # armed=False 是相位调度分支的正常形态;armed=True 却没有 hint MQ
            # 就是接线坏了(提示发不出去 ⇒ 闸门永不放行),必须显式可见。
            logger.warning(
                "[Lwd][pre-recv] hint sender NOT registered armed=%s "
                "hint_mq=%s",
                self._lwd_early_recv, hint_mq is not None,
            )
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

    @staticmethod
    def _lwd_coord_stats_addr(config, frontend_stats_publish_address) -> str:
        """HELLO 通告的 coordinator stats 地址:host 改写为本机可路由 IP
        (pre_out_host)——coordinator 与云 engine 同机,其 front XPUB 在
        LWD 下绑 0.0.0.0,通告值必须是边可路由地址;无 coordinator
        (dp=1)时为空。"""
        addr = frontend_stats_publish_address
        if not addr or not str(addr).startswith("tcp://"):
            return ""
        _, _, port = str(addr).rpartition(":")
        return f"tcp://{config.pre_out_host}:{port}"

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
            # mixed 组批:items 逐请求登记同一批号;空 items = 旧版单请求
            items = msg.items or [
                LwdRangeItem(msg.request_id, msg.offset, msg.num_tokens)
            ]
            for item in items:
                seqnos = self._lwd_seqno_registry.setdefault(
                    item.request_id, []
                )
                if not seqnos or msg.seqno > seqnos[-1]:
                    seqnos.append(msg.seqno)
            logger.info(
                "[Lwd][cloud-ctrl] RangeNotify reqs=%d num=%s seqno=%s",
                len(items), sum(item.num_tokens for item in items),
                msg.seqno,
            )
            # 每条预告都整条入队(重复预告即重复点名,剔除-调度-拼回幂等,
            # 无副作用;PRE_OUT 只 append,调度主线程单独 popleft,deque
            # 单操作原子;预告自带 seqno,出批时作 UP 链配对号)
            self.scheduler.prefill_notify_queue.append(msg)
            # [Lwd][pre-recv] 提前收①:预告到达即给云 worker 下发提示(seqno +
            # flat token 数),worker 的通信线程收到就挂精确尺寸的
            # UP irecv(主帧 hash 载荷) —— 不再等这个批的
            # SchedulerOutput 下发。num_tokens 取 items 逐条目之和
            # (mixed 恒填 items;顶层字段只描述首条目)。mrope positions
            # 已随 RangeNotify 走控制面(wire_version>=2),UP 数据面只有
            # 主帧,aux_rows 恒 0——若残留非零,worker 会预挂一条永远收
            # 不到数据的 aux irecv,水位停摆、闸门后的批永不下发。
            if self._lwd_early_recv:
                from vllm_ascend.distributed.lwd_comm.pre_recv import (
                    make_recv_hint,
                    post_irecv_hint,
                )

                num_tokens = sum(item.num_tokens for item in items)
                aux_rows = 0
                # 先打点再入队:worker 通信线程 0.5ms 就能取走,若打点在入队之后,
                # 跨进程写日志的次序会让 hint2sub 出现负值,看着像时序错乱。
                logger.info(
                    "[Lwd][pre-recv] HINT seqno=%d num_tokens=%d aux_rows=%d "
                    "ts=%.3f",
                    msg.seqno, num_tokens, aux_rows, time.monotonic(),
                )
                post_irecv_hint(
                    make_recv_hint(msg.seqno, num_tokens, aux_rows)
                )
            return
        if isinstance(msg, LwdAbortNotify):
            logger.info("[Lwd][cloud-ctrl] AbortNotify req=%s", msg.request_id)
            self._lwd_gate_pending.pop(msg.request_id, None)
            self._lwd_seqno_registry.pop(msg.request_id, None)
            # 预告队列**不改**:该请求已预告的 chunk 数据已在线上,必须由
            # 调度器先算完再终结(见 LwdBaseScheduler.finish_requests);
            # "请求已不存在"时的条目清理也在那里做。
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
        # 行数即 prompt 长度(UP 注入直接写该缓冲的对应窗口);
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
        self._lwd_drain_irecv_completions()
        out = super().step_with_batch_queue()
        logger.info(
            "[Lwd][perf] cloud-step exec=%.2fms",
            (time.monotonic() - _t0) * 1000,
        )
        return out

    def step(self):
        """同步步路径同款打点。"""
        _t0 = time.monotonic()
        self._lwd_drain_irecv_completions()
        out = super().step()
        logger.info(
            "[Lwd][perf] cloud-step exec=%.2fms",
            (time.monotonic() - _t0) * 1000,
        )
        return out

    def _lwd_drain_irecv_completions(self) -> None:
        """[Lwd][pre-recv] 提前收③:把 worker 上报的收完项并进水位(每步一次)。

        云 worker 的通信线程每收完一条 UP recv 就上报 (channel, seqno);本处
        在每步开头排空 done MQ,水位随即成为 PD_mix 闸门
        (``_lwd_pre_recv_ready``) 的就绪判据 —— 收完的批本步就能下发,
        没收完的本步只跑 decode。
        """
        done_mq = getattr(self.model_executor, "lwd_irecv_done_mq", None)
        if done_mq is None:
            return
        from vllm_ascend.distributed.lwd_comm.pre_recv import (
            record_irecv_completions,
        )

        items = []
        while True:
            try:
                items.append(done_mq.dequeue(timeout=0))
            except TimeoutError:
                break
            except Exception:
                # 关停时 ring 被取消:记一条即可,水位停在原处不影响收尾
                logger.exception("[Lwd][pre-recv] done MQ drain failed")
                break
        if not items:
            return
        for channel, seqno in record_irecv_completions(items):
            # 引擎侧收到上报的落点(与 worker 的 REPORT 行对减 = 上报过 MQ +
            # 等到本步排空的延迟)
            logger.info(
                "[Lwd][pre-recv] watermark advanced channel=%s seqno=%d ts=%.3f",
                channel.value, seqno, time.monotonic(),
            )

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
            # 诊断旁路(SKIP_SAMPLE 边免采样 / DISABLE_DOWN 全直通):
            # 把本步逐请求 accepted token ids 捎带给边侧直接交付。
            token_ids: list[list[int]] = []
            if _lwd_diag_token_passthrough_enabled():
                ids_by_req = self._lwd_step_token_ids(engine_core_outputs)
                token_ids = [ids_by_req.get(rid, []) for rid in meta.req_ids]
            self._lwd_publish_c2e(
                meta,
                self._lwd_c2e_finish_reasons(meta, engine_core_outputs),
                token_ids,
            )
            # [Lwd][perf] 云侧 LWD 税:finish 码推导 + ZMQ publish
            logger.info(
                "[Lwd][perf] publish reqs=%d dur=%.2fms",
                len(meta.req_ids), (time.monotonic() - _t) * 1000,
            )
        elif os.environ.get("VLLM_ASCEND_LWD_DISABLE_DOWN") == "1":
            # VLLM_ASCEND_LWD_DISABLE_DOWN=1 诊断:cloud runner 不做
            # collect、不发 DOWN,但 c2e 通告必须照常驱动边侧交付——
            # 逐请求 token ids 直传,hidden_num_elements=0 标记无张量;
            # 仅进 finished_requests 的请求补空行以携带 ABORT 码。
            ids_by_req = self._lwd_step_token_ids(engine_core_outputs)
            for outputs in engine_core_outputs.values():
                for rid in outputs.finished_requests or ():
                    ids_by_req.setdefault(rid, [])
            if not ids_by_req:
                # 空步(无输出也无完结):空通告没有任何可交付内容,
                # 发了只会让边侧白醒来一次并空派一个 UNEMBED 批占配额,
                # 直接跳过。
                return model_output
            req_ids = list(ids_by_req)
            from vllm.v1.outputs import LwdC2eMeta

            meta = LwdC2eMeta(
                hidden_num_elements=0,
                top_id_ths=[],
                num_accepted_tokens=[],
                req_ids=req_ids,
                down_seqno=-1,
            )
            self._lwd_publish_c2e(
                meta,
                self._lwd_c2e_finish_reasons(meta, engine_core_outputs),
                [ids_by_req[rid] for rid in req_ids],
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

    @staticmethod
    def _lwd_step_token_ids(
        engine_core_outputs: dict[int, EngineCoreOutputs],
    ) -> dict[str, list[int]]:
        """本步逐请求 new_token_ids(request_id 索引),诊断旁路共用。"""
        ids_by_req: dict[str, list[int]] = {}
        for outputs in engine_core_outputs.values():
            for out in outputs.outputs:
                ids_by_req[out.request_id] = list(out.new_token_ids)
        return ids_by_req

    def _lwd_publish_c2e(
        self,
        meta: LwdC2eMeta,
        finish_reasons: list[int],
        token_ids: list[list[int]] | None = None,
    ) -> None:
        """步元数据(含逐请求 finish_reasons 完成码)发边:队满小睡重试
        (元数据不可丢),关停(closed)退出。instance_id 与 DEALER identity
        同源盖章,边侧据此分发到对应实例的 UNEMBED/数据面路径。token_ids 仅诊断旁路
        (VLLM_ASCEND_LWD_EDGE_SKIP_SAMPLE)携带。"""
        notify = LwdC2eNotify(
            hidden_num_elements=meta.hidden_num_elements,
            top_id_ths=meta.top_id_ths,
            num_accepted_tokens=meta.num_accepted_tokens,
            req_ids=meta.req_ids,
            finish_reasons=finish_reasons,
            down_seqno=meta.down_seqno,
            instance_id=getattr(self, "_lwd_instance_id", 0),
            token_ids=token_ids or [],
            # POST_OUT fan-in 下供边侧归账到本 engine 的 seqno 流/路由账
            engine_index=self.vllm_config.parallel_config.data_parallel_rank,
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
