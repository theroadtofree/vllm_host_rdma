"""边侧引擎子类:run_engine_core 类选择点注入的边 EngineCore。

边引擎职责:通信面装配(POST_OUT bind/接收线程/PRE_OUT 延迟连接)、
调度器注入(LwdEdgeScheduler)、请求入口/步进/关停的引擎接口覆写。
全部状态收进子类字段,不外挂引擎属性。

构造约束:
  ① 引擎主体先行(super() 最先调):接收线程 WAKEUP 敲门的
     self.input_queue 由 super 创建,后启动线程即无构造期竞态;代价是
     云缺失时引擎主体启动白费,超时路径只清理通信面,主体随进程退出
     兜底回收。通信面随后构建:bind POST_OUT、建云载荷队列与接收线程,
     阻塞等云首拍 HELLO——HELLO 在云引擎全量初始化(权重/KV/图编译)
     完成后才发出,等到了它才允许边侧对外就绪;超时(hello_timeout_s,
     默认 600s,须覆盖云全量启动时长)fail-fast。
     HELLO 首拍一次、无周期重发,不考虑任一侧重启自愈:重启即整组重拉,
     构造期等待是边侧唯一的发现窗口。
  ② 调度器经 scheduler_cls 注入裸类,须在 super() 之前设值——super 内
     构建调度器时一次性消费该配置,后设无效;引擎构造完成后回填
     publisher(早于任何请求,等价构造注入)。

步进编排(embed 优先):步首收割在飞批(unembed 交付 token/请求终结,
embed 登记进度)-> prefill 编排(单请求组批 -> 范围预告 -> executor
异步执行,优先占深度配额)-> 云载荷消费(c2e -> UNEMBED 批,剩余配额)。
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections import deque

from vllm.logger import init_logger
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequestType,
    FinishReason,
)
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
    LwdControlPublisher,
)
from vllm.v1.lwd_control.control_communication.lwd_control_subscriber import (
    LwdControlSubscriber,
)
from vllm.v1.lwd_control.control_communication.lwd_role_registry import (
    get_role_registry,
)
from vllm.v1.lwd_control.control_communication.lwd_router_channel import (
    LwdControlRouterChannel,
)
from vllm.v1.lwd_debug import LwdLogBase
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LWD_NOT_FINISHED,
    LwdC2eNotify,
    LwdHelloNotify,
    lwd_decode_cloud_notify,
    lwd_decode_wire_notify,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
    LwdConfig,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdEdgeScheduler,
    lwd_build_unembed_batch,
)

logger = init_logger(__name__)


def _lwd_disable_down_enabled() -> bool:
    """VLLM_ASCEND_LWD_DISABLE_DOWN=1:云不 collect/不发 DOWN,边直通交付。"""
    return os.environ.get("VLLM_ASCEND_LWD_DISABLE_DOWN") == "1"


# 云->边载荷队列容量:队满时接收线程阻塞在 put,背压沿 ZMQ 直达云侧
# 步发送循环(载荷不可丢)
LWD_C2E_META_QUEUE_MAX = 1000

# 批队列在飞深度上限:派发(embed+unembed 合计)不收割的批数上限,
# 突发积压时连续派发喂饱 worker(参照仓经验 >=4 才能维持流水对齐)
LWD_EDGE_BATCH_QUEUE_DEPTH = 4


class LwdEdgeEngineCore(EngineCoreProc):
    """边 PO 引擎:通信面装配 + 调度器注入 + step/add/abort/shutdown 覆写。"""

    def __init__(self, *args, **kwargs) -> None:
        vllm_config = kwargs["vllm_config"]
        # 调度器注入须赶在 super() 之前:super 内构建 self.scheduler 时
        # 一次性消费 scheduler_cls,后设无效(注入失效,首请求即崩)
        vllm_config.scheduler_config.scheduler_cls = LwdEdgeScheduler
        super().__init__(*args, **kwargs)
        # probe 完成唤醒钩:probe 线程完成回调经它敲 WAKEUP,使纯睡眠期
        # 的引擎进步回灌结果(钩在 super 之后、任何请求之前安装,等价
        # 构造注入;input_queue 由 super 创建,无构造期竞态)
        self.scheduler.lwd_probe_wakeup = self._lwd_wakeup
        config = LwdConfig.from_env_and_config(vllm_config)
        # 层日志总开关:env 已开则不动,config 段开则补开(仅本进程)
        LwdLogBase.set_debug(config.debug)
        # 云->边唯一载荷队列:生产端接收线程,消费端引擎步;数据面经
        # UNEMBED 批的 lwd_c2e_notifies 拿元数据,不直接读队列
        # (单消费者语义)。云侧复用元素为 (cloud_id, notify, t),
        # 1E1C 为 (notify, t)
        self.lwd_c2e_meta_queue = queue.Queue(maxsize=LWD_C2E_META_QUEUE_MAX)
        # 已派发待收割的批队列 (kind, payload, t_dispatch, future):
        # kind="unembed" payload=notify;kind="embed" payload=scheduler_output。
        # 派发不收割,队首 FIFO 收割——必须全局单队列:executor 的 FutureWrapper
        # 按底层队列序排水,交错收割会连带等错批。t_dispatch 供
        # [Lwd][sched] harvest wait(派发→收割)度量批在队列里的滞留时长
        self._lwd_batch_queue: deque = deque()
        # 云侧回边的 usage(含 prompt_tokens_details.cached_tokens)。
        # 消费口径**与参考分支一致:仅边侧留痕**(参考打 event
        # `edge_usage_received`,我们落此 dict + 同字段日志),**不注入客户端
        # 响应**;留存可供后续可选消费者取用(边界淘汰最旧,防长期运行增长)。
        self.lwd_edge_usages: dict[str, dict] = {}
        # 各云 HELLO 载荷留存(cloud_id -> LwdHelloNotify):调度三元组
        # 互校的输入(1E1C 在构造期校验;云侧复用在接收线程逐云校验)
        self._lwd_peer_hellos: dict[int, LwdHelloNotify] = {}
        # 装配期模式分叉:registry_path 非空 = 云侧复用,单 ROUTER 通道
        # connect 全部云;为空 = 现状 1E1C 单套 Publisher/Subscriber +
        # 阻塞等云首拍 HELLO
        self._edge_channel: LwdControlRouterChannel | None = None
        self._edge_receiver: LwdControlSubscriber | None = None
        self._edge_sender: LwdControlPublisher | None = None
        if config.is_cloud_reuse:
            self._edge_channel = self._lwd_build_control_channel(config)
            self.scheduler.lwd_edge_channel = self._edge_channel
            threading.Thread(
                target=self._receive_thread_channel,
                name="lwd-router-in",
                daemon=True,
            ).start()
            logger.info(
                "[Lwd] edge engine assembled: cloud-reuse router channel "
                "(identity=edge%d)",
                config.self_edge_id,
            )
            return
        # 通信面:bind POST_OUT 订阅面 + 延迟连接的 PRE_OUT 发布面;
        # 云端点由 HELLO 通告决定(边不预知云地址)
        self._edge_receiver = self._lwd_build_post_out(config)
        self._edge_sender = LwdControlPublisher(
            None, bind=False, queue_max=config.publish_queue_max
        )
        hello_event = threading.Event()
        discovery = threading.Thread(
            target=self._receive_thread,
            args=(hello_event,),
            name="lwd-post-in",
            daemon=True,
        )
        discovery.start()
        if not hello_event.wait(config.hello_timeout_s):
            self._lwd_shutdown_planes()
            raise RuntimeError(
                f"[Lwd] edge engine init failed: no cloud HELLO within "
                f"{config.hello_timeout_s}s on POST_OUT "
                f"(bind {config.lwd_post_out_bind_endpoint()}; check cloud "
                f"master_addr connectivity and POST_OUT port)"
            )
        self._lwd_cross_check_hello(0)

        self.scheduler.lwd_edge_publisher = self._edge_sender
        logger.info(
            "[Lwd] edge engine assembled: POST_OUT bind %s, PRE_OUT discovered",
            config.lwd_post_out_bind_endpoint(),
        )

    # ------------------------------------------------------------------ #
    # 通信面                                                              #
    # ------------------------------------------------------------------ #
    def _lwd_build_control_channel(self, config: LwdConfig) -> LwdControlRouterChannel:
        """云侧复用:建单 ROUTER 通道(identity=edge{self_edge_id}),按
        registry 端点公式 connect 全部云并逐云 HELLO;云侧就绪前消息
        滞留 per-identity FIFO,就绪后按序补投,不阻塞装配。

        connect 侧同为 ROUTER(``connect_as_router=True``,上层装配
        决定):一个 socket 连 N 台云,双向 identity 首帧寻址。"""
        registry = get_role_registry()
        if registry is None:
            from vllm.v1.lwd_control.control_communication.lwd_role_registry import (
                init_role_registry,
            )

            registry = init_role_registry(config.registry_path)
        channel = LwdControlRouterChannel(
            None,
            bind=False,
            identity=f"edge{config.self_edge_id}",
            decoder=lwd_decode_wire_notify,
            expected_instances=len(registry.cloud_ids),
            connect_as_router=True,
        )
        for cloud_id in registry.cloud_ids:
            channel.connect(registry.endpoint(cloud_id))
            channel.announce(cloud_id)
        channel.start()
        return channel

    def _lwd_build_post_out(self, config: LwdConfig) -> LwdControlSubscriber:
        """bind POST_OUT 订阅面;云经 master_addr 主动来连(仅 1E1C)。"""
        return LwdControlSubscriber(
            config.lwd_post_out_bind_endpoint(),
            bind=True,
            decoder=lwd_decode_cloud_notify,
        )

    def _receive_thread(self, hello_event: threading.Event) -> None:
        """POST_OUT 接收线程体,按消息类型分发。

        HELLO -> retarget PRE_OUT:云端点唯一事实源,首拍一次通告;
        retarget 队满时无下条 HELLO 可等,须本线程自旋重试到成功
        (构造期发布队列必空,该路径仅防御性保活)。
        LwdC2eNotify(云->边唯一载荷)-> 载荷队列(阻塞 put,不可丢)
        + WAKEUP 唤醒主循环:prefill 全部完成后请求转入 awaiting,
        引擎无排程工作、阻塞在 input_queue.get(),载荷只进队列不会
        唤醒任何线程,必须向 input_queue 敲门;WAKEUP 原生语义即丢弃
        消息体,数据与唤醒分离,多投无害(空 drain 一步即返回)。
        其余帧(坏帧已被订阅层丢弃后仍不认识的类型)告警丢弃。
        """
        receiver = self._edge_receiver
        publisher = self._edge_sender
        while not receiver.closed:
            msg = receiver.recv(timeout_ms=5000)
            if msg is None:
                continue
            if isinstance(msg, LwdHelloNotify):
                endpoint = f"tcp://{msg.pre_out_host}:{msg.pre_out_port}"
                if not hello_event.is_set():
                    logger.info(
                        "[Lwd] cloud discovered via HELLO: PRE_OUT -> %s", endpoint
                    )
                # 载荷留存供构造线程互校(调度三元组,见 _lwd_cross_check_hello)
                self._lwd_peer_hellos[msg.cloud_id] = msg
                while not publisher.retarget(endpoint):
                    if receiver.closed:
                        break
                    logger.warning(
                        "[Lwd] PRE_OUT retarget deferred (queue full), retrying"
                    )
                    threading.Event().wait(0.05)
                # 采纳云侧调度语义须在 hello_event.set() 之前,先于调度步
                self._lwd_adopt_cloud_semantics(msg)
                hello_event.set()
            elif isinstance(msg, LwdC2eNotify):
                logger.debug(
                    "[Lwd][edge-ctrl] C2eNotify reqs=%d down_seqno=%s",
                    len(getattr(msg, "req_ids", []) or []),
                    getattr(msg, "down_seqno", None),
                )
                # 队列元素带到达时间戳:消费滞后(c2e_wait=到达→派发)
                # 是"边消费不动 → 云 publisher 队满小睡"闭环的前置指标
                self.lwd_c2e_meta_queue.put((msg, time.monotonic()))
                self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
            else:
                logger.warning("[Lwd] drop unexpected POST_OUT frame %r", type(msg))

    def _receive_thread_channel(self) -> None:
        """云侧复用接收线程:轮询单通道 consume_new_outputs(),收全部云。

        单 IO 线程串行收(fair-queue),来源云 id 由信封 identity 携带;
        (cloud_id, notify) 投递 c2e 队列(阻塞 put,不可丢)+ WAKEUP
        唤醒主循环(多生产者单消费者是 queue.Queue 标配用法),
        _lwd_edge_step 逻辑不变。"""
        channel = self._edge_channel
        assert channel is not None
        while not channel.closed:
            batch = channel.consume_new_outputs(timeout_s=1.0)
            for cloud_id, notify in batch:
                if isinstance(notify, LwdHelloNotify):
                    # 云按边定向送达的调度三元组(互校):信封 cloud_id
                    # 为准(覆盖载荷字段,防错配),逐云校验
                    self._lwd_peer_hellos[cloud_id] = notify
                    try:
                        # 复用模式没有 1E1C 的 hello_event 同步点,
                        # 语义只能从这里采纳
                        self._lwd_adopt_cloud_semantics(notify)
                        self._lwd_cross_check_hello(cloud_id)
                    except Exception:
                        # 接收线程内异常不能静默死线程(配置错位必须
                        # 升级):报主循环,按引擎级失败处理
                        logger.exception(
                            "[Lwd] HELLO triple cross-check failed "
                            "(cloud%d)", cloud_id,
                        )
                        self.input_queue.put_nowait(
                            (EngineCoreRequestType.EXECUTOR_FAILED, b"")
                        )
                    continue
                if not isinstance(notify, LwdC2eNotify):
                    logger.warning(
                        "[Lwd] drop unexpected router frame %r from cloud%d",
                        type(notify), cloud_id,
                    )
                    continue
                # 信封携带的来源云为准(覆盖载荷字段,防错配)
                notify.cloud_id = cloud_id
                logger.debug(
                    "[Lwd][edge-ctrl] C2eNotify cloud=%d reqs=%d down_seqno=%s",
                    cloud_id, len(notify.req_ids), notify.down_seqno,
                )
                self.lwd_c2e_meta_queue.put((cloud_id, notify, time.monotonic()))
                self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

    def _lwd_cross_check_hello(self, cloud_id: int) -> None:
        """HELLO 调度三元组互校(逐云;1E1C 构造期调用,云侧复用在
        接收线程收到该云 HELLO 时调用)。

        max_num_batched_tokens/max_num_seqs 是边 cap 公式与云 preflight
        的同源输入(云通告的是**生效值**,派生/缺省差异也被覆盖),
        不一致即 fail-fast——不一致不会让链路报错,只会让云 preflight
        恒 hold、prefill 被 decode 静默串行(真机实录:边 factor=1/
        云 factor=4 时 chunk 8191 + decode 4 > 预算 8192,prefill
        积压 280 步)。num_speculative_tokens 不一致时以云为准采纳
        (边本地取不到合法来源)并复核活性下界。对端为旧版(三值全 0)
        仅告警不拦截。"""
        hello = self._lwd_peer_hellos.get(cloud_id)
        if hello is None:
            return
        if hello.max_num_batched_tokens == 0:
            logger.warning(
                "[Lwd] cloud%d HELLO carries no scheduling triple "
                "(legacy peer); edge/cloud config alignment UNCHECKED",
                cloud_id,
            )
            return
        sched = self.vllm_config.scheduler_config
        spec_config = getattr(self.vllm_config, "speculative_config", None)
        local = (
            sched.max_num_batched_tokens,
            sched.max_num_seqs,
            getattr(spec_config, "num_speculative_tokens", None) or 0,
        )
        remote = (
            hello.max_num_batched_tokens,
            hello.max_num_seqs,
            hello.num_speculative_tokens,
        )
        if local[:2] != remote[:2]:
            self._lwd_shutdown_planes()
            raise RuntimeError(
                f"[LWD] edge/cloud scheduling config mismatch: "
                f"(max_num_batched_tokens, max_num_seqs) "
                f"edge={local[:2]} != cloud{cloud_id}={remote[:2]}; "
                f"cap/preflight 公式两侧同源输入不一致会让 preflight 恒 "
                f"hold、prefill 被 decode 静默串行——两侧拉起配置须同值"
            )
        if local[2] != remote[2]:
            # spec 系数以云为准:它预留的是云 decode 步的草稿预算,边侧
            # 本地取不到合法来源(边实例带 spec 配置会拖起 draft 机制)。
            # 采纳对端值而非 fail-fast——部署上边侧无需再配 spec。
            logger.warning(
                "[Lwd] adopt cloud%d spec_factor=%d over edge local %d "
                "(edge has no legitimate local source for it)",
                cloud_id, remote[2] + 1, local[2] + 1,
            )
            self.scheduler._lwd_spec_factor = remote[2] + 1
            # 采纳后复核活性下界(构造期检查用的是本地 factor,见
            # LwdEdgeScheduler.__init__)
            if (
                self.scheduler.max_num_scheduled_tokens
                - self.scheduler.max_num_running_reqs * (remote[2] + 1) < 1
            ):
                self._lwd_shutdown_planes()
                raise RuntimeError(
                    f"[LWD] 死锁配置: 采纳云侧 spec_factor={remote[2] + 1} "
                    f"后 cap 恒 < 1(预算 {self.scheduler.max_num_scheduled_tokens}"
                    f" − 满员 {self.scheduler.max_num_running_reqs} × "
                    f"{remote[2] + 1});请增大 max_num_batched_tokens "
                    f"或减小 max_num_seqs"
                )
        logger.info(
            "[Lwd] HELLO scheduling triple verified (cloud%d): %s",
            cloud_id, local,
        )

    def _lwd_adopt_cloud_semantics(self, hello: LwdHelloNotify) -> None:
        """采纳云侧调度语义(HELLO 首拍,先于任何调度步):
        ① spec_factor/EAGLE 尾块回退口径——MTP 只在云侧配置,边侧本地
        取不到合法来源,一律以云侧公告为准;② 调度三元组互校,不一致
        fail-fast(否则 preflight 恒 hold、prefill 被 decode 静默串行);
        ③ 前缀对齐网格写入边侧并按网格重切 chunk。"""
        scheduler = self.scheduler
        count = int(getattr(hello, "num_speculative_tokens", 0) or 0)
        adopted_factor = count + 1
        old_factor = int(getattr(scheduler, "_lwd_spec_factor", 1))
        use_eagle = bool(getattr(hello, "prefix_use_eagle", False))
        logger.info(
            "[Lwd][edge] adopt cloud MTP semantics: spec_factor %d -> %d "
            "(num_speculative_tokens=%d), prefix_use_eagle=%s",
            old_factor, adopted_factor, count, use_eagle,
        )
        scheduler._lwd_spec_factor = adopted_factor
        scheduler._lwd_prefix_use_eagle = use_eagle
        # ② 调度三元组互校
        if hello.max_num_batched_tokens != 0:
            sched = self.vllm_config.scheduler_config
            local = (sched.max_num_batched_tokens, sched.max_num_seqs)
            remote = (hello.max_num_batched_tokens, hello.max_num_seqs)
            if local != remote:
                self._lwd_shutdown_planes()
                raise RuntimeError(
                    f"[LWD] 边云调度配置不一致: "
                    f"(max_num_batched_tokens, max_num_seqs) "
                    f"edge={local} != cloud={remote}; cap/preflight 公式"
                    f"两侧同源输入不一致会让云 preflight 恒 hold、prefill "
                    f"被 decode 静默串行——两侧拉起配置须同值"
                )
            logger.info("[Lwd] HELLO scheduling triple verified: %s", local)
        else:
            logger.warning(
                "[Lwd] peer HELLO carries no scheduling triple "
                "(legacy peer; UNCHECKED)"
            )
        # ③ 前缀对齐网格采纳(与 probe 响应头学习同源,幂等兜底)
        alignment = int(getattr(hello, "prefix_alignment_tokens", 0) or 0)
        required = bool(getattr(hello, "prefix_alignment_required", False))
        if alignment > 0:
            cloud_id = scheduler._lwd_default_cloud_id()
            scheduler._lwd_align_grid_by_cloud[cloud_id] = alignment
            if not scheduler.need_mamba_block_aligned_split:
                scheduler.need_mamba_block_aligned_split = True
            logger.info(
                "[Lwd][edge] adopt cloud prefix alignment: cloud=%d "
                "alignment=%d required=%s (chunk auto-alignment enabled)",
                cloud_id, alignment, required,
            )
            # required 时须留出一个对齐块 + 满员 decode 预算,否则续传
            # chunk 永无可调度预算。
            if required:
                usable = (
                    scheduler.max_num_scheduled_tokens
                    - scheduler.max_num_running_reqs * adopted_factor
                )
                if usable < alignment:
                    self._lwd_shutdown_planes()
                    raise RuntimeError(
                        f"[LWD] Mamba align 需 max_num_batched_tokens >= "
                        f"{alignment + scheduler.max_num_running_reqs * adopted_factor} "
                        f"(当前 {scheduler.max_num_scheduled_tokens},满员 "
                        f"{scheduler.max_num_running_reqs} × spec_factor "
                        f"{adopted_factor} 后只剩 {usable} < 对齐块 "
                        f"{alignment});请增大 max_num_batched_tokens"
                    )
        # 采纳后复核活性下界(边侧 __init__ 的死锁检查用的是本地 factor)
        if scheduler._lwd_mixed and (
            scheduler.max_num_scheduled_tokens
            - scheduler.max_num_running_reqs * adopted_factor
            < 1
        ):
            self._lwd_shutdown_planes()
            raise RuntimeError(
                f"[LWD] 采纳云侧 spec_factor={adopted_factor} 后 cap 恒 < 1"
                f"(预算 {scheduler.max_num_scheduled_tokens} − 满员 "
                f"{scheduler.max_num_running_reqs} × {adopted_factor});"
                "请增大 max_num_batched_tokens 或减小 max_num_seqs"
            )

    def _lwd_shutdown_planes(self) -> None:
        """通信面关停(幂等):云侧复用单通道;1E1C 两面。"""
        channel = getattr(self, "_edge_channel", None)
        if channel is not None:
            channel.shutdown()
        receiver = getattr(self, "_edge_receiver", None)
        if receiver is not None:
            receiver.shutdown()
        publisher = getattr(self, "_edge_sender", None)
        if publisher is not None:
            publisher.shutdown()

    # ------------------------------------------------------------------ #
    # 引擎接口覆写                                                        #
    # ------------------------------------------------------------------ #
    def add_request(self, request, _request_wave: int = 0) -> None:
        """边校验 + 云预告 + 本地入队。

        _request_wave:原生主循环按位置传入(core.py ADD 分发),本模式
        无 DP wave 语义,仅保形参契约,不消费。"""
        self.scheduler.lwd_edge_add_request(request)

    def abort_requests(self, request_ids: list[str]) -> None:
        """abort 信号先出云,再走原生本地清理。"""
        self.scheduler.lwd_edge_abort(request_ids)
        super().abort_requests(request_ids)

    def step(self):
        return self._lwd_edge_step()

    def step_with_batch_queue(self):
        return self._lwd_edge_step()

    def has_work(self) -> bool:
        """c2e 载荷队列与待收割批队列都计入工作:WAKEUP 只能打断阻塞的
        get,轮询循环是否退出由本判据决定(core.py while not has_work)。
        批队列非空必须计入——否则最后一批派发后引擎睡眠,最终 token
        永不收割交付。probe 完成队列/发布滞留队列同样计入:回灌与
        补投都只发生在步循环里,不计入会在纯等待期睡死。"""
        return (super().has_work() or not self.lwd_c2e_meta_queue.empty()
                or self._lwd_batch_queue
                or self.scheduler.lwd_probe_results_pending()
                or self.scheduler.lwd_has_pending_publishes())

    def _lwd_wakeup(self) -> None:
        """probe 完成唤醒(probe 线程经 scheduler.lwd_probe_wakeup 调用)。"""
        try:
            self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
        except Exception:  # noqa: BLE001  队列关闭等关停竞态,丢弃无害
            pass

    def _lwd_edge_step(self) -> tuple[dict[int, object] | None, bool]:
        """单步编排(批队列异步化+多批在飞,embed 优先):收割队首直至
        清空 -> prefill 连续派发(优先占深度配额) -> 消费云载荷(剩余
        配额派发 UNEMBED)-> 返回({0: 云侧结果输出} | None,prefill
        是否有派发)。

        embed/unembed 均派发入队不收割,队首全局 FIFO 收割(序与
        executor 排水序同构,不可交错);突发时一步连派至深度上限
        喂饱 worker。派发序 embed 优先:196 并发下每云步都产 c2e,
        若 unembed 先吃配额会把 embed 挤出本步(EMBED 被 decode 回程
        挤住的饿死形态),TTFT 劣化——故 embed 先占位,unembed 用剩余
        配额,代价是 decode token 交付最多延后 1-2 步(顶部全量收割
        + c2e 队列背压兜底,不会饿死)。

        [Lwd][sched] 三类日志(饿死分析锚点):每批 harvest(含队列滞留
        wait)、每批 dispatch(含 ahead 队列构成/c2e 水位)、每步汇总。"""
        # 步首先回灌异步 probe 结果(完成入队,可能即刻产生可调度请求),
        # 再补投背压滞留的控制消息(保序在一切新发布之前)
        self.scheduler.lwd_edge_drain_probe_results()
        self.scheduler.lwd_edge_flush_pending()
        outputs: list = []
        finished_reqs: set = set()
        prefill_work = False
        h_emb = h_unemb = 0
        while self._lwd_batch_queue:
            kind, payload, t_dispatch, future = self._lwd_batch_queue.popleft()
            if kind == "unembed":
                self._lwd_deliver_unembed(
                    payload, future, outputs, finished_reqs
                )
                h_unemb += 1
            else:
                self.scheduler.lwd_edge_update_progress(
                    dict(payload.num_scheduled_tokens)
                )
                prefill_work = True
                h_emb += 1
            logger.info(
                "[Lwd][sched] edge harvest kind=%s seqno=%s wait=%.2fms",
                kind, self._lwd_batch_seqno(kind, payload),
                (time.monotonic() - t_dispatch) * 1000,
            )
        d_emb = 0
        while (len(self._lwd_batch_queue) < LWD_EDGE_BATCH_QUEUE_DEPTH
               and self.scheduler.has_requests()):
            scheduler_output = self.scheduler.schedule()
            if not scheduler_output.num_scheduled_tokens:
                break
            future = self._lwd_dispatch_embed(scheduler_output)
            if future is None:
                break
            self._lwd_batch_queue.append(
                ("embed", scheduler_output, time.monotonic(), future)
            )
            prefill_work = True
            d_emb += 1
        d_unemb = self._lwd_edge_consume_c2e()
        n_emb, n_unemb = self._lwd_queue_mix()
        logger.info(
            "[Lwd][sched] edge step harvest_emb=%d harvest_unemb=%d "
            "disp_emb=%d disp_unemb=%d queue=%demb/%dunemb c2e_pending=%d",
            h_emb, h_unemb, d_emb, d_unemb, n_emb, n_unemb,
            self.lwd_c2e_meta_queue.qsize(),
        )
        if outputs:
            step_outputs = EngineCoreOutputs(
                outputs=outputs,
                finished_requests=finished_reqs or None,
            )
            return {0: step_outputs}, prefill_work
        return None, prefill_work

    def _lwd_queue_mix(self) -> tuple[int, int]:
        """批队列构成 (embed数, unembed数):dispatch 日志的 ahead_unemb
        = EMBED 批前面压着的 UNEMBED 批数,是"EMBED 被 decode 回程挤住"
        (饿死假说)的直接读数。"""
        n_emb = sum(1 for e in self._lwd_batch_queue if e[0] == "embed")
        return n_emb, len(self._lwd_batch_queue) - n_emb

    @staticmethod
    def _lwd_batch_seqno(kind: str, payload) -> str:
        """harvest 日志的配对号:unembed 用 down_seqno(与云 DOWN 同号),
        embed 用批 seqno(与 RangeNotify/UP 同号)。"""
        if kind == "unembed":
            return str(getattr(payload, "down_seqno", "?"))
        batch = getattr(payload, "lwd_batch", None)
        return str(getattr(batch, "seqno", "?")) if batch else "?"

    def _lwd_edge_consume_c2e(self) -> int:
        """消费 c2e 通告:派发 UNEMBED 批入批队列(异步收割)。
        云侧有活请求才产 meta(build_hidden_payload 空则返回 None),
        每条 entry 行数>=1——通告必有行,无需无行分流。
        云侧复用队列元素为 (cloud_id, notify, t),1E1C 为 (notify, t)。
        DISABLE_DOWN 诊断:通告只携带 token_ids(hidden=0)时跳过 worker
        派发,以 future=None 直通收割,不挂任何 recv。
        返回本步派发条数。"""
        quota = LWD_EDGE_BATCH_QUEUE_DEPTH - len(self._lwd_batch_queue)
        entries: list[tuple[LwdC2eNotify, float]] = []
        while len(entries) < quota:
            try:
                item = self.lwd_c2e_meta_queue.get_nowait()
            except queue.Empty:
                break
            if self._edge_channel is not None:
                _, notify, t_arrive = item
            else:
                notify, t_arrive = item
            entries.append((notify, t_arrive))
        for notify, t_arrive in entries:
            # 直通条件改为载荷驱动为主:hidden=0 + 携带 token_ids 即
            # 直通(云侧 DISABLE_DOWN 开、边侧未开时也不会去挂一个
            # numel=0/seqno=-1 的 recv 把引擎步炸掉);env 开关保留
            # 兼容(SKIP_SAMPLE 等 hidden>0 场景)。
            if notify.token_ids and (
                notify.hidden_num_elements == 0 or _lwd_disable_down_enabled()
            ):
                logger.info(
                    "[Lwd][sched] edge direct-deliver seqno=%s reqs=%d "
                    "(token_ids passthrough, no unembed dispatch)",
                    notify.down_seqno, len(notify.req_ids),
                )
                future = None
            else:
                future = self._lwd_dispatch_unembed(notify, t_arrive)
            self._lwd_batch_queue.append(
                ("unembed", notify, time.monotonic(), future)
            )
        return len(entries)

    def _lwd_dispatch_embed(self, scheduler_output):
        """EMBED 批提交侧(唯一提交点,同步/异步共用):范围预告(发布
        成功才组批挂 lwd_batch)后以 non_block 提交,返回 future;
        发布失败返回 None,本轮不再提交。

        正确性前提:发布通道不丢消息。失败分支无回退可施——进度
        已按排程乐观推进,该 chunk 随 SO 废弃即永久丢失(进度虚高),
        正确性由通道不丢保证;同步/异步失败语义镜像(均为弃批)。"""
        if not self.scheduler.lwd_edge_notify(scheduler_output):
            return None
        batch = scheduler_output.lwd_batch
        n_emb, n_unemb = self._lwd_queue_mix()
        logger.info(
            "[Lwd][sched] edge dispatch-embed seqno=%d reqs=%s tokens=%d "
            "ahead_unemb=%d ahead_emb=%d c2e_pending=%d",
            batch.seqno, batch.batch_meta.req_ids,
            scheduler_output.total_num_scheduled_tokens,
            n_unemb, n_emb, self.lwd_c2e_meta_queue.qsize(),
        )
        return self.model_executor.execute_model(
            scheduler_output, non_block=True
        )

    def _lwd_dispatch_unembed(self, notify: LwdC2eNotify, t_arrive: float):
        """UNEMBED 批提交侧:组批并以 non_block 提交 worker,立即返回
        future(不等执行)。同步路径提交后立即收割;异步路径
        (batch_queue)将 (notify, future) 入队,收割阶段再等 future——
        提交与等待的边界即本函数返回处。

        c2e_wait(到达→派发)是消费滞后读数:持续偏大说明边引擎步
        循环被收割/派发占住,c2e 在积压,云侧 publisher 队满小睡在即。"""
        unembed_batch = lwd_build_unembed_batch(notify)
        n_emb, n_unemb = self._lwd_queue_mix()
        logger.info(
            "[Lwd][sched] edge dispatch-unembed seqno=%s reqs=%d rows=%d "
            "ahead_unemb=%d ahead_emb=%d c2e_wait=%.2fms c2e_pending=%d",
            notify.down_seqno, len(notify.req_ids),
            sum(notify.num_accepted_tokens), n_unemb, n_emb,
            (time.monotonic() - t_arrive) * 1000,
            self.lwd_c2e_meta_queue.qsize(),
        )
        return self.model_executor.execute_model(
            unembed_batch, non_block=True
        )

    @staticmethod
    def _lwd_row_disposition(
        accepted: int, has_tokens: bool, finished: bool
    ) -> str:
        """逐行交付裁定(pad-and-skip):

        - "skip":零接受活行(accepted=0 且未完结)——mid-prefill 非参与
          行,无内容无终态,通告里保留只为 DOWN 张量对位,不交付、
          不 ERROR、不标记首 token(修复前此形态被 ERROR 杀请求);
        - "error":声明 accepted>0 却无 token = unembed 真失败,ERROR
          优先于云侧完成码(原语义保留);
        - "deliver":其余(含零接受但带完结码的收尾行——终结信号必须
          透传,吞掉会丢完结)。"""
        if accepted == 0 and not finished:
            return "skip"
        if not has_tokens and not finished:
            return "error"
        return "deliver"

    def _lwd_deliver_unembed(
        self, notify: LwdC2eNotify, future, outputs: list, finished_reqs: set,
    ) -> None:
        """UNEMBED 批收割侧:等 worker 应答取采样 token 表,逐请求交付。
        不感知提交时机,只消费 (notify, future)。应答契约:
        token ids 按 ModelRunnerOutput 的 req_ids x sampled_token_ids
        按位对齐还原(批的 req_ids 原样下发,worker 逐位回填);行裁定
        走 _lwd_row_disposition(零接受活行跳过,accepted>0 缺 token 才
        ERROR);迟到载荷幂等丢弃。"""
        _t = time.monotonic()
        if future is None:
            # DISABLE_DOWN 直通:无 worker 批,token 直接取自 c2e 通告。
            sampled_token_map: dict[str, list[int]] = {
                rid: list(toks)
                for rid, toks in zip(notify.req_ids, notify.token_ids)
            }
        else:
            result = future.result()
            # req_ids x sampled_token_ids 按位对齐:worker lm_head 恢复的采样
            # token,即该请求本步的生成内容;后续仅两处流向——
            # lwd_edge_deliver_tokens(调度器只对账 awaiting 生命周期,
            # 不消费内容)与 EngineCoreOutput 的 new_token_ids(outputs ->
            # EngineCoreOutputs -> 主循环按 frontend 消费 -> 前端解流交付
            # 客户端,即最终输出的生成 token)。
            sampled_token_map = (
                {} if result is None
                else dict(zip(result.req_ids, result.sampled_token_ids))
            )
        # [Lwd][perf] 临时探针:收割时长 = RPC 往返 + worker 执行全长
        # (与 worker 侧 [Lwd][perf] unembed 分段对账,差值即进程往返开销)
        logger.debug(
            "[Lwd][perf] harvest dur=%.2fms", (time.monotonic() - _t) * 1000
        )
        # accepted 与行按位对齐(lwd_build_unembed_batch 断言保证);
        # DISABLE_DOWN 旁路不带 accepted,以回填 token 有无代替
        accepts = notify.num_accepted_tokens
        for index, request_id in enumerate(notify.req_ids):
            finish_reason = self._lwd_finish_code(notify, index)
            finished = finish_reason is not None
            sampled_token_ids = (
                list(sampled_token_map.get(request_id, []))
                if sampled_token_map else []
            )
            accepted = (
                accepts[index] if accepts
                else (1 if sampled_token_ids else 0)
            )
            disposition = self._lwd_row_disposition(
                accepted, bool(sampled_token_ids), finished
            )
            if disposition == "skip":
                # 零接受活行:不交付不出空输出,也无谓打调试税
                continue
            # [lwd-debug] 性能关闭:每拍 tokenizer.decode() 是 CPU 重活,
            # 诊断乱码时恢复
            # LwdDebug.edge_tokens_delivered(
            #     request_id, sampled_token_ids, finish_reason,
            #     self.vllm_config,
            # )
            if disposition == "error":
                # 声明接受却无 token = unembed 失败,ERROR 优先于云侧码
                # (已完结的空行属正常,如 DISABLE_DOWN 下的 ABORT)
                finish_reason = FinishReason.ERROR
                finished = True
            if not self.scheduler.lwd_edge_deliver_tokens(
                request_id, sampled_token_ids, finished=finished
            ):
                logger.warning(
                    "[Lwd] drop stale cloud payload for %s (not awaiting)",
                    request_id,
                )
                continue
            if not sampled_token_ids and not finished:
                # 无内容且未完结:不出空输出
                continue
            outputs.append(
                EngineCoreOutput(
                    request_id, sampled_token_ids,
                    finish_reason=finish_reason,
                )
            )
            if finished:
                finished_reqs.add(request_id)
                self._lwd_consume_usage(notify, index, request_id)

    def _lwd_consume_usage(
        self, notify: LwdC2eNotify, index: int, request_id: str
    ) -> None:
        """消费云侧随 C2e 回边的 usage(含 cached_tokens)。

        参考分支经探针 SSE 末 chunk 回 usage;prefill_only 边侧探针同步、
        不消费 body,故改挂步元数据通道。落 ``lwd_edge_usages`` 供记账/前端
        取用,fail-open(缺字段即跳过)。"""
        usages = getattr(notify, "usages", None)
        if not usages or index >= len(usages) or not usages[index]:
            return
        usage = usages[index]
        if len(self.lwd_edge_usages) >= 4096:
            for stale in list(self.lwd_edge_usages)[:512]:
                self.lwd_edge_usages.pop(stale, None)
        self.lwd_edge_usages[request_id] = usage
        details = usage.get("prompt_tokens_details") or {}
        logger.info(
            "[Lwd][edge] usage req=%s prompt=%s completion=%s total=%s cached=%s",
            request_id, usage.get("prompt_tokens"), usage.get("completion_tokens"),
            usage.get("total_tokens"), details.get("cached_tokens"),
        )

    @staticmethod
    def _lwd_finish_code(
        notify: LwdC2eNotify, index: int
    ) -> FinishReason | None:
        """逐请求完成码:哨兵 = 未完结(None);否则透传 FinishReason。
        与 req_ids 按位严格对齐,错配 IndexError fail-fast。"""
        code = notify.finish_reasons[index]
        return None if code == LWD_NOT_FINISHED else FinishReason(code)

    def shutdown(self) -> None:
        """两面关停后走原生;初始化失败路径两面可能未建,容忍缺省。"""
        self._lwd_shutdown_planes()
        super().shutdown()
