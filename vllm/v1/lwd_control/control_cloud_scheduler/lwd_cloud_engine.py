"""云侧 EngineCore 子类:覆写 socket IO 线程入口,PRE_OUT 循环独立成线程,
边侧预告与步内元数据经 input_queue 走原生分发;仅 prefill_only 云角色启用。"""

from __future__ import annotations

import queue
import threading
import time
from logging import DEBUG
from typing import TYPE_CHECKING

import msgspec
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
from vllm.v1.lwd_control.control_communication.lwd_id_adapter import (
    unwrap_req_id,
    wrap_req_id,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LWD_NOT_FINISHED,
    LwdAbortNotify,
    LwdC2eNotify,
    LwdFinishNotify,
    LwdGenChainNotify,
    LwdHelloNotify,
    LwdRangeNotify,
    LwdRequestNotify,
    lwd_decode_wire_notify,
    lwd_encode_cloud_notify,
)
from vllm.v1.lwd_control.control_communication.lwd_prefix import LwdUsage
from vllm.v1.lwd_control.control_communication.lwd_role_registry import (
    get_role_registry,
    init_role_registry,
)
from vllm.v1.lwd_control.control_communication.lwd_router_channel import (
    LwdControlRouterChannel,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import LwdConfig
from vllm.v1.lwd_debug import LwdDebug
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_phase_scheduler import (
    LwdCloudPhaseScheduler,
)
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_prefix_coordinator import (
    LwdPrefixCoordinator,
)
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_control import (
    LwdCloudControlProcessor,
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

# 云侧复用 consume 阻塞拍:空批时短暂等待,避免忙转
_LWD_ROUTER_CONSUME_TIMEOUT_S = 1.0


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
        # 前缀复用协调:协调器 + 处理器(队列由父进程传入,None = 未启用)。
        # **必须在 super() 之前占位**:super 内即启动 PRE_OUT IO 线程,该线程经
        # _lwd_setup_zmq → _lwd_init_dispatch_state 读这些属性(以及按配置判断
        # 协调是否启用);后设会与之竞态(实测 AttributeError → EXECUTOR_FAILED)。
        self._lwd_prefix_coordinator: LwdPrefixCoordinator | None = None
        self._lwd_cloud_control_processor: LwdCloudControlProcessor | None = None
        # 生成段链台账:rid -> (链粒度, 链)。**同样在 super() 前建**——
        # PRE_OUT IO 线程的分派分支要写它;交付调度器(读侧)在
        # _lwd_setup_coordination 里做(主线程,步循环之前,无竞态)。
        self._lwd_edge_chains: dict[str, tuple[int, list[bytes]]] = {}
        super().__init__(*args, **kwargs)
        self._lwd_setup_coordination()

    def _lwd_setup_coordination(self) -> None:
        """装配前缀复用协调器 + 控制面处理器(队列由父进程传入,None = 未启用)。"""
        coord_cfg = getattr(self.vllm_config, "lwd_coordination", None)
        if coord_cfg is None or not coord_cfg.enabled:
            return
        if (
            self.cloud_control_command_queue is None
            or self.cloud_control_event_queue is None
        ):
            logger.warning(
                "[Lwd][ctrl] coordination enabled but queues absent; skip")
            return
        hash_block_size = resolve_kv_cache_block_sizes(
            self.scheduler.kv_cache_config, self.vllm_config
        )[1]
        # 诊断:KV 组规格。混合注意力下各组 block_size/滑窗不同,命中长度取
        # 各组最小值(SWA/Mamba 还会把窗口外/非对齐块排除出哈希表),直接
        # 决定长 prompt 能被声明多少前缀——排查命中异常先看这行。
        # mamba_cache_mode:align 只存「每步最后一个 token 且落在 i*block」的
        # state(单步整段 prefill 时几乎无快照),all 每个块边界都存。
        logger.info(
            "[Lwd][ctrl] kv groups (type, block_size, window, mamba_mode)=%s",
            [
                (
                    type(g.kv_cache_spec).__name__,
                    g.kv_cache_spec.block_size,
                    getattr(g.kv_cache_spec, "sliding_window", None),
                    getattr(g.kv_cache_spec, "mamba_cache_mode", None),
                )
                for g in self.scheduler.kv_cache_config.kv_cache_groups
            ],
        )
        self._lwd_prefix_coordinator = LwdPrefixCoordinator(
            hash_block_size,
            instance_id=coord_cfg.instance_id,
            kv_cache_manager=getattr(self.scheduler, "kv_cache_manager", None),
        )
        self._lwd_cloud_control_processor = LwdCloudControlProcessor(
            self.cloud_control_command_queue, self.cloud_control_event_queue
        )
        # 生成段链台账交付调度器(引用交付:IO 线程写 / 调度线程读,dict 赋值
        # 原子,与 seqno registry 同款)。**只在此暴露**(而非 IO 线程的
        # _lwd_init_dispatch_state):此刻协调确实建成,步循环尚未开始,无竞态;
        # 未启用协调时调度器看不到它 → 生成段块照原生 local 哈希登记。
        self.scheduler.lwd_edge_chains = self._lwd_edge_chains
        # 生成段哈希:云侧**不持租户密钥**——生成段(completion)token 的 HMAC
        # 摘要由边侧在 LwdFinishNotify 里上报,云只做校验与记账发布
        # (apply_finish_chain)。这样"密钥仅存边侧、永不过网"的隐私性质
        # 成立(参考分支即此语义)。
        logger.info(
            "[Lwd][ctrl] coordination ready (hash_block_size=%d, "
            "tenant key held edge-side only)",
            hash_block_size,
        )

    def _lwd_poll_coordination(self) -> None:
        """消费控制面 probe 命令(引擎步循环头部与空闲拍,非阻塞)。"""
        processor = self._lwd_cloud_control_processor
        coordinator = self._lwd_prefix_coordinator
        if processor is not None and coordinator is not None:
            processor.poll(coordinator)

    # 空闲拍控制面服务间隔:原生 _process_input_queue 在无请求时于
    # input_queue 上无限阻塞,引擎不进 step,probe 消费点(步循环头部)
    # 永不执行——云空闲时边侧同步 probe 必然干等到超时(表现为边侧
    # hit_tokens=0/fail-open 或手动探针 TimeoutError)。改为有界等待,
    # 每拍服务一次控制面命令队列,使 probe 与引擎步循环解耦。
    _LWD_IDLE_WAIT_S = 0.05

    def _process_input_queue(self) -> None:
        """空闲等待覆写:仅协调启用时生效(未启用走原生,零行为变化)。"""
        if getattr(self, "_lwd_cloud_control_processor", None) is None:
            return super()._process_input_queue()
        waited = False
        while not self.has_work() and self.is_running():
            # 空闲拍先服务控制面:probe 命令不与「引擎有待办请求」耦合
            self._lwd_poll_coordination()
            self._notify_idle_state_callbacks()
            if self.input_queue.empty():
                # Drain aborts queue; all aborts are also processed via input_queue.
                with self.aborts_queue.mutex:
                    self.aborts_queue.queue.clear()
                if logger.isEnabledFor(DEBUG):
                    logger.debug("EngineCore waiting for work.")
                    waited = True
            try:
                req = self.input_queue.get(timeout=self._LWD_IDLE_WAIT_S)
                self._handle_client_request(*req)
            except queue.Empty:
                continue
        if waited:
            logger.debug("EngineCore loop active.")
        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _lwd_claim_reservation(self, request: Request) -> None:
        """认领前缀复用预留(admit 期,观测用;续算由 block_hasher 驱动)。"""
        coordinator = self._lwd_prefix_coordinator
        if coordinator is None:
            return
        hit_tokens = coordinator.claim(request.request_id)
        if hit_tokens is not None:
            logger.info(
                "[Lwd][coord] admit req=%s claim hit_tokens=%d",
                request.request_id, hit_tokens,
            )

    def _lwd_publish_completed_prefixes(self, req_ids: list[str]) -> None:
        """prompt 已算完的请求发布其 manifest 摘要进 completed 集。"""
        coordinator = self._lwd_prefix_coordinator
        if coordinator is None:
            return
        requests = getattr(self.scheduler, "requests", {})
        for request_id in req_ids:
            request = requests.get(request_id)
            if request is None:
                continue
            if request.num_computed_tokens >= request.num_prompt_tokens:
                coordinator.complete_request(request_id)

    def _lwd_publish_finished_usage(
        self, engine_core_outputs: dict[int, "EngineCoreOutputs"]
    ) -> dict[str, LwdUsage]:
        """请求完结结算(finish→记账),返回 {req_id: LwdUsage} 供 C2e 回边。

        请求在 ``update_from_output`` 内已被 ``_free_blocks`` 从
        ``scheduler.requests`` 摘除(见 ``Scheduler._free_blocks``),故本
        方法不再回查请求,改用调度器在 ``finish_requests`` 时抓取的快照
        (``scheduler.lwd_finished_records``)——否则结算永不触发、预留泄漏。
        生成段哈希由边侧 ``LwdFinishNotify`` 上报(见 ``apply_finish_chain``),
        云侧不再持租户密钥,故此处 ``generated_hashes=[]``。

        usage 的消费口径与参考分支一致(仅边侧日志留痕,不注入客户端响应);
        参考经探针 SSE 流末 chunk 承载,我们改挂本步 C2e 回边(见
        ``_lwd_consume_usage``),故此处直接返回而不经控制面桥。"""
        coordinator = self._lwd_prefix_coordinator
        if coordinator is None:
            return {}
        finished: set[str] = set()
        for outputs in engine_core_outputs.values():
            for out in outputs.outputs:
                if out.finish_reason is not None:
                    finished.add(out.request_id)
            for request_id in outputs.finished_requests or ():
                finished.add(request_id)
        records = getattr(self.scheduler, "lwd_finished_records", None) or {}
        usages: dict[str, LwdUsage] = {}
        for request_id in finished:
            snapshot = records.pop(request_id, None)
            if snapshot is None:
                continue
            prompt_tokens, completion_tokens = snapshot
            usage = coordinator.finish(
                request_id,
                generated_hashes=[],
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            usages[request_id] = usage
        return usages

    def _lwd_setup_zmq(self) -> None:
        """介入 ZMQ 双面:PRE_OUT bind 收边;POST_OUT connect 边,承载首拍
        HELLO 通告与步内元数据。建站失败走 EXECUTOR_FAILED 升级。

        装配期模式分叉:registry_path 非空 = 云侧复用,建单 ROUTER 通道
        (bind 单端口,identity=cloud{self_cloud_id}),服务全部边,无
        mux——单 socket fair-queue + 单 IO 线程天然串行,来源 edge_id 由
        信封 identity 携带;为空 = 现状 1E1C 单套 subscriber + publisher
        + HELLO 首拍。"""
        config = LwdConfig.from_env_and_config(self.vllm_config)
        self._lwd_channel: LwdControlRouterChannel | None = None
        self._lwd_subscriber: LwdControlSubscriber | None = None
        self._lwd_post_out: LwdControlPublisher | None = None
        self._lwd_cloud_id = getattr(
            self.vllm_config.parallel_config.lwd_config, "cloud_id", 0
        )
        if config.is_cloud_reuse:
            registry = get_role_registry()
            if registry is None:
                registry = init_role_registry(config.registry_path)
            self._lwd_channel = LwdControlRouterChannel(
                registry.bind_endpoint(config.self_cloud_id),
                bind=True,
                identity=f"cloud{config.self_cloud_id}",
                decoder=lwd_decode_wire_notify,
                expected_instances=len(registry.edge_ids),
            )
            self._lwd_channel.start()
            self._lwd_init_dispatch_state()
            logger.info(
                "[Lwd] cloud engine assembled: cloud-reuse router channel "
                "bind %s (identity=cloud%d, expects %d edges)",
                registry.bind_endpoint(config.self_cloud_id),
                config.self_cloud_id,
                len(registry.edge_ids),
            )
            return
        self._lwd_subscriber = LwdControlSubscriber(
            config.lwd_pre_out_endpoint(), bind=True
        )
        master_addr = self.vllm_config.parallel_config.master_addr
        self._lwd_post_out = LwdControlPublisher(
            f"tcp://{master_addr}:{config.post_out_port}",
            bind=False,
            encoder=lwd_encode_cloud_notify,
        )
        self._lwd_hello = LwdHelloNotify(
            pre_out_host=config.pre_out_host,
            pre_out_port=config.pre_out_port,
            cloud_id=self._lwd_cloud_id,
        )
        self._lwd_init_dispatch_state()
        # 首拍即通告(边侧可能已 bind 等待)
        self._lwd_announce()
        logger.info(
            "[Lwd] cloud engine assembled: PRE_OUT bind %s, POST_OUT announce -> "
            "%s:%s via master %s",
            config.lwd_pre_out_endpoint(),
            config.pre_out_host,
            config.pre_out_port,
            master_addr,
        )

    def _lwd_init_dispatch_state(self) -> None:
        """两种模式共用的分派状态(门池/seqno 登记表)。"""
        # 门池:元数据查重与暂存,到达即构建放行;仅本 IO 线程独占
        self._lwd_gate_pending: dict[str, LwdRequestNotify] = {}
        # UP 链 seqno 登记(边→云→云 worker 的最后一跳,§9.12 接缝):
        # rid -> [chunk 序 seqno 列表],RangeNotify 到达即登记;调度器
        # 出 prefill 批时取快照挂 SO.lwd_up_seqnos 随批下发云 worker
        # (数据面 UP recv 配对键,与边侧 SO.lwd_batch.seqno 同源同值)。
        # registry 引用交付调度器(IO 线程登记 / 主循环读,dict 赋值原子)。
        self._lwd_seqno_registry: dict[str, list[int]] = {}
        self.scheduler.lwd_seqno_registry = self._lwd_seqno_registry

    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ) -> None:
        """父线程照跑父类原版,PRE_OUT 循环独立成线程,两生产者共用 input_queue。"""
        threading.Thread(
            target=self._lwd_pre_out_loop, daemon=True, name="lwd-pre-out"
        ).start()
        super().process_input_sockets(
            input_addresses, coord_input_address, identity, ready_event
        )

    def _lwd_pre_out_loop(self) -> None:
        """PRE_OUT 接收循环:socket 与门状态在本线程内先建后用(zmq 单线程
        亲和);recv 挂超时拍仅作关停响应上限,关停(closed)退出。

        云侧复用:循环 consume_new_outputs() 单通道,逐条 (edge_id, notify)
        交分派(单消费者串行,替代 mux 轮询);1E1C 阻塞收单 subscriber。"""
        try:
            self._lwd_setup_zmq()
        except Exception:
            logger.exception("[Lwd] cloud PRE_OUT setup failed")
            self.input_queue.put_nowait((EngineCoreRequestType.EXECUTOR_FAILED, b""))
            return
        if self._lwd_channel is not None:
            channel = self._lwd_channel
            while not channel.closed:
                for edge_id, notify in channel.consume_new_outputs(
                    timeout_s=_LWD_ROUTER_CONSUME_TIMEOUT_S
                ):
                    self._lwd_dispatch(edge_id, notify)
            return
        while True:
            msg = self._lwd_subscriber.recv(timeout_ms=LWD_PRE_OUT_RECV_TIMEOUT_MS)
            if msg is None:
                if self._lwd_subscriber.closed:
                    break
                continue
            self._lwd_dispatch(getattr(msg, "edge_id", 0), msg)

    def _lwd_announce(self) -> None:
        """首拍 HELLO 通告一次;队满不重试,由边侧等待超时 fail-fast 兜底。"""
        self._lwd_post_out.publish(self._lwd_hello)
        logger.info(
            "[Lwd][cloud] HELLO announced: pre_out=%s:%s",
            self._lwd_hello.pre_out_host, self._lwd_hello.pre_out_port,
        )

    def shutdown(self) -> None:
        """通信面关停后走原生(幂等;装配失败路径可能未建,容忍缺省)。"""
        channel = getattr(self, "_lwd_channel", None)
        if channel is not None:
            channel.shutdown()
        subscriber = getattr(self, "_lwd_subscriber", None)
        if subscriber is not None:
            subscriber.shutdown()
        publisher = getattr(self, "_lwd_post_out", None)
        if publisher is not None:
            publisher.shutdown()
        super().shutdown()

    def _lwd_dispatch(self, edge_id: int, msg) -> None:
        """PRE_OUT 三类分派(本 IO 线程):元数据转 Request / abort 终结 /
        范围预告登记 seqno。

        edge_id:云侧复用取自信封 identity(来源标注免费携带,不解析
        载荷);1E1C 取消息自带字段(缺省 0,单边兼容)。入口即做
        req_id 命名空间包装(wrap_req_id),云内只认包装 id。"""
        if isinstance(msg, LwdRangeNotify):
            # 入口命名空间包装:云内统一用 wrapped_req_id,调度器据此
            # 解析 edge_id 做数据面分桶。
            msg = msgspec.structs.replace(
                msg, request_id=wrap_req_id(edge_id, msg.request_id)
            )
            # 范围预告:登记 UP 链 seqno(数据面配对键,§9.12)。幂等去重
            # 按"单调性"(seqno 不大于该请求已登记尾号即重复,边侧队满
            # 重试天然产生重复预告,重试复用同一号不产生新登记)。
            seqnos = self._lwd_seqno_registry.setdefault(msg.request_id, [])
            if not seqnos or msg.seqno > seqnos[-1]:
                seqnos.append(msg.seqno)
            logger.info(
                "[Lwd][cloud-ctrl] RangeNotify req=%s edge=%d num=%s seqno=%s",
                msg.request_id, edge_id, msg.num_tokens, msg.seqno,
            )
            # 每条预告都整条入队(重复预告即重复点名,剔除-调度-拼回幂等,
            # 无副作用;PRE_OUT 只 append,调度主线程单独 popleft,deque
            # 单操作原子;预告自带 seqno,出批时作 UP 链配对号)
            self.scheduler.prefill_notify_queue.append(msg)
            return
        if isinstance(msg, LwdAbortNotify):
            msg = msgspec.structs.replace(
                msg, request_id=wrap_req_id(edge_id, msg.request_id)
            )
            logger.info("[Lwd][cloud-ctrl] AbortNotify req=%s", msg.request_id)
            self._lwd_gate_pending.pop(msg.request_id, None)
            self._lwd_edge_chains.pop(msg.request_id, None)
            # 双队列与原生 ABORT 同款:eager 处理 + 保持 input_queue 次序
            self.aborts_queue.put_nowait([msg.request_id])
            self.input_queue.put_nowait((EngineCoreRequestType.ABORT, [msg.request_id]))
            return
        if isinstance(msg, LwdFinishNotify):
            # 收尾记账(§11.4):边上报全量块哈希链(prompt+生成段)。云侧只
            # 校验 prompt 前缀链未变 + 发布生成段摘要,**不接触明文、不持密钥**。
            msg = msgspec.structs.replace(
                msg, request_id=wrap_req_id(edge_id, msg.request_id)
            )
            coordinator = getattr(self, "_lwd_prefix_coordinator", None)
            if coordinator is None:
                return
            coordinator.apply_finish_chain(
                msg.request_id,
                msg.prompt_tokens,
                msg.completion_tokens,
                list(msg.full_block_hashes),
                msg.publish_cache,
            )
            # 收尾后生成段链台账即可释放(后续块不再需要登记)
            self._lwd_edge_chains.pop(msg.request_id, None)
            return
        if isinstance(msg, LwdGenChainNotify):
            # 生成段哈希链增量上报(全量替换)。存台账供两处同源消费:
            # ① build_request 的 block_hasher 填生成段哈希;② 调度器封顶
            # 生成段块的登记窗口(链未到前不按 local_hasher 误登记)。
            msg = msgspec.structs.replace(
                msg, request_id=wrap_req_id(edge_id, msg.request_id)
            )
            if len(self._lwd_edge_chains) >= 4096:
                for stale in list(self._lwd_edge_chains)[:512]:
                    self._lwd_edge_chains.pop(stale, None)
                logger.warning("[Lwd][ctrl] gen-chain table full; evicted oldest")
            # 存 (链粒度, 链):消费侧只在粒度与本地 hash 块粒度相等(ratio=1)
            # 时采用——非等宽时链索引与云内 block_hashes(hash 粒度)错位,
            # 采用会按错键登记块。
            self._lwd_edge_chains[msg.request_id] = (
                int(msg.block_size), list(msg.block_hashes)
            )
            logger.info(
                "[Lwd][cloud-ctrl] GenChainNotify req=%s blocks=%d bs=%d",
                msg.request_id, len(msg.block_hashes), msg.block_size,
            )
            return
        rid = wrap_req_id(edge_id, msg.request_id)
        msg = msgspec.structs.replace(msg, request_id=rid)
        if rid in self._lwd_gate_pending:
            logger.warning("[Lwd] duplicate request metadata %s ignored", rid)
            return
        logger.info("[Lwd][cloud-ctrl] RequestNotify req=%s edge=%d", rid, edge_id)
        self._lwd_gate_pending[rid] = msg
        self._lwd_promote(rid)

    def _lwd_promote(self, request_id: str) -> None:
        """过门:门池取 wire,转 Request 投 input_queue 走原生 ADD 分发。"""
        wire = self._lwd_gate_pending.pop(request_id, None)
        if wire is not None:
            request = self._lwd_build_request(wire)
            self._lwd_claim_reservation(request)
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
            # resume 封顶:边侧声明的续算边界(= 其 chunk 起点)是数据面
            # 单真源,云侧块池命中不得超过它——边侧 probe 超时/并发下池
            # 增长时,原生 find_longest_cache_hit 会命中更多,注入起点即错位。
            request.lwd_resume_cap_tokens = wire.prefix_hit_tokens
            return request
        hash_block_size = resolve_kv_cache_block_sizes(
            self.scheduler.kv_cache_config, self.vllm_config
        )[1]

        def block_hasher(request: Request) -> list[bytes]:
            # 有预留且仍在纯 prefill 时用 HMAC 协调摘要(跨请求 prompt 前缀
            # 命中);一旦开始 decode 即回退本地哈希,以覆盖全部(prompt+生成)
            # 满块——协调摘要只覆盖 prompt 段,decode 所需满块数会超长,
            # cache_full_blocks 有 len(block_hashes)>=num_full_blocks 断言。
            coordinator = self._lwd_prefix_coordinator
            if coordinator is not None and request.num_output_tokens == 0:
                hashes = coordinator.coordination_hashes(request.request_id)
                if hashes is not None and len(hashes) == (
                    request.num_prompt_tokens // hash_block_size
                ):
                    return hashes
            if len(request.block_hashes) == 0 and request.num_output_tokens == 0:
                expected = request.num_prompt_tokens // hash_block_size
                if len(wire.block_hashes) == expected:
                    return wire.block_hashes
            # 生成段满块取边侧回传链(与 prompt 段同域同源)。链只在
            # 「粒度与本地 hash 粒度相等且尚有未覆盖块」时采用;其余情况
            # **必须回退 local_hasher**:块哈希列表要随 token 增长持续补满
            # cache_full_blocks 的 len>=num_full_blocks 断言,返回空会把
            # 列表冻住(实测 DeepSeek:边链粒度 128 vs 云 hash 8,decode 每
            # 8 token 跨一次块边界,冻结即刻断言)。回退登记的本地键不会
            # 被误读——读侧由边侧声明封顶(probe 命中 0 → 云侧 resume 封 0)。
            gen_entry = self._lwd_edge_chains.get(request.request_id)
            if gen_entry is not None and int(gen_entry[0]) == hash_block_size:
                gen_chain = gen_entry[1]
                start = len(request.block_hashes)
                if start < len(gen_chain):
                    return list(gen_chain[start:])
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
        request.lwd_resume_cap_tokens = wire.prefix_hit_tokens
        return request
    
    def step_with_batch_queue(self):
        self._lwd_poll_coordination()
        return super().step_with_batch_queue()

    def step(self):
        self._lwd_poll_coordination()
        return super().step()

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
        # 收尾结算先行:usage 需随本步 C2e 回边,而 C2e 在此处发布,
        # 故先算好 finish/usage 再发(原实现放在方法末尾,回边拿不到)。
        usages = self._lwd_publish_finished_usage(engine_core_outputs)
        # 注意:meta.req_ids 出口已剥包装,而 usages/records 键是云内包装 id
        carrier = getattr(model_output, "lwd_down_carrier", None)
        if carrier:
            for edge_id, pinned, req_ids, hidden_numel, seqno in carrier:
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
                    "[Lwd][cloud-ctrl] publish c2e(rank-replay): edge=%d "
                    "reqs=%s rows=%d seqno=%d",
                    edge_id, meta.req_ids, off, seqno,
                )
                LwdDebug.cloud_step(self.scheduler, meta, engine_core_outputs)  # [lwd-debug]
                _t = time.monotonic()
                self._lwd_publish_c2e(
                    edge_id,
                    meta,
                    self._lwd_c2e_finish_reasons(meta, engine_core_outputs),
                    usages,
                )
                # prompt 算完的请求发布其 manifest 摘要 → 后续 probe 可命中
                self._lwd_publish_completed_prefixes(meta.req_ids)
                # [Lwd][perf] 云侧 LWD 税:finish 码推导 + ZMQ publish
                logger.info(
                    "[Lwd][perf] publish edge=%d reqs=%d dur=%.2fms",
                    edge_id, len(meta.req_ids),
                    (time.monotonic() - _t) * 1000,
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

    def _lwd_publish_c2e(
        self,
        edge_id: int,
        meta: LwdC2eMeta,
        finish_reasons: list[int],
        usages: dict[str, LwdUsage] | None = None,
    ) -> None:
        """步元数据(含逐请求 finish_reasons 完成码 + usage)发边:队满小睡
        重试(元数据不可丢),关停(closed)退出。req_ids 出口剥离命名空间,
        边侧只看原始 id。

        usages:按云内包装 id 命中,与出口 req_ids 按位对齐;空 dict 项
        表示该请求本步无 usage。参考分支经探针 SSE 末 chunk 回 usage,我们
        改挂本通道(边侧已按步消费,零新管道)。

        按边分组已在云 worker 的 down carrier 完成(每 edge 独立
        down_seqno,一份 C2eNotify 严格配对一次 DOWN 发送),本方法只
        按分组结果定向发布:云侧复用经单通道 publish(dest=edge_id,
        per-identity FIFO 独立,某边消费慢只反压自己);1E1C 走单
        publisher。"""
        usage_by_req = usages or {}
        notify = LwdC2eNotify(
            hidden_num_elements=meta.hidden_num_elements,
            top_id_ths=meta.top_id_ths,
            num_accepted_tokens=meta.num_accepted_tokens,
            req_ids=[unwrap_req_id(r) for r in meta.req_ids],
            finish_reasons=finish_reasons,
            down_seqno=meta.down_seqno,
            cloud_id=self._lwd_cloud_id,
            usages=[
                (u.to_openai_dict() if (u := usage_by_req.get(rid)) else {})
                for rid in meta.req_ids
            ],
        )
        if self._lwd_channel is not None:
            while not self._lwd_channel.closed:
                if self._lwd_channel.publish(notify, edge_id):
                    logger.info(
                        "[Lwd][cloud-ctrl] publish C2eNotify edge=%d reqs=%d "
                        "down_seqno=%s finish=%s hidden_elems=%s",
                        edge_id,
                        len(notify.req_ids),
                        notify.down_seqno,
                        finish_reasons,
                        notify.hidden_num_elements,
                    )
                    return
                time.sleep(_LWD_C2E_SEND_RETRY_SLEEP_S)
            return
        while not self._lwd_post_out.closed:
            if self._lwd_post_out.publish(notify):
                logger.info(
                    "[Lwd][cloud-ctrl] publish C2eNotify edge=%d reqs=%d "
                    "down_seqno=%s finish=%s hidden_elems=%s",
                    edge_id,
                    len(notify.req_ids),
                    notify.down_seqno,
                    finish_reasons,
                    notify.hidden_num_elements,
                )
                return
            time.sleep(_LWD_C2E_SEND_RETRY_SLEEP_S)
