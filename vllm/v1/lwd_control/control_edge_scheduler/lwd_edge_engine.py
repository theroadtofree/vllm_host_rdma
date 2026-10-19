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
异步执行,优先占 EMBED 保留预算)-> 云载荷消费(c2e -> UNEMBED 批,
剩余配额;批队列总深度随云实例数扩容,公式见 vllm.config.lwd)。
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections import defaultdict, deque
from typing import Any

from vllm.config.lwd import (
    LWD_EDGE_BASE_DEPTH,
    lwd_edge_batch_queue_depth,
)
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
from vllm.v1.lwd_control.control_communication.lwd_coord_client import (
    LwdCoordClient,
)
from vllm.v1.lwd_control.control_communication.lwd_zmq_factory import (
    create_edge_zmq_channel,
)
from vllm.v1.lwd_debug import LwdDebug, LwdLogBase
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LWD_NOT_FINISHED,
    LWD_WIRE_VERSION,
    LwdC2eNotify,
    LwdHelloNotify,
    lwd_decode_cloud_notify,
    lwd_encode_notify,
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

# POST_OUT 泵线程阻塞拍:仅作关停响应上限;消息到达由通知事件即时唤醒
LWD_POST_OUT_PUMP_TIMEOUT_MS = 5000


class LwdEdgeEngineCore(EngineCoreProc):
    """边 PO 引擎:通信面装配 + 调度器注入 + step/add/abort/shutdown 覆写。"""

    def __init__(self, *args, **kwargs) -> None:
        vllm_config = kwargs["vllm_config"]
        # 调度器注入须赶在 super() 之前:super 内构建 self.scheduler 时
        # 一次性消费 scheduler_cls,后设无效(注入失效,首请求即崩)
        vllm_config.scheduler_config.scheduler_cls = LwdEdgeScheduler
        super().__init__(*args, **kwargs)
        config = LwdConfig.from_env_and_config(vllm_config)
        # 层日志总开关:env 已开则不动,config 段开则补开(仅本进程)
        LwdLogBase.set_debug(config.debug)
        # 云->边唯一载荷的按实例暂存桶(生产端泵线程 append,消费端引擎
        # 步轮询 popleft;deque 单操作原子,单生产者单消费者无锁):
        # 泵不阻塞,按实例背压由通道层 per-identity FIFO(上限 1000)
        # 承担;桶水位是消费滞后的本地读数(正常远低于通道 FIFO)。
        # 数据面经 UNEMBED 批的 lwd_c2e_notifies 拿元数据,不直接读桶
        # (单消费者语义)
        # 桶集合构造期一次建全、永不增删、属性永不重绑:泵线程的
        # setdefault 恒命中既有键(取桶与 append 之间无 detach 窗口),
        # 引擎侧遍历时也不会因新键插入抛 dict-changed-size——若消费后
        # 整体重绑 self._lwd_c2e_pending,泵线程可能把 append 落在旧
        # 字典对象上,该 c2e 通告永久丢失(请求 finish 丢失即永不返回)。
        # 键数 = 实例数,有界,不存在膨胀问题
        self._lwd_c2e_pending: dict[int, deque] = {
            iid: deque() for iid in range(config.instance_count)
        }
        # 实例数标记:泵线程 c2e 越界防御用(与屏障/通道的 N 同源)
        self._lwd_instance_count = config.instance_count
        # 云 DP engine 数(D):控制面按 (instance, engine) 双维度定向
        self._lwd_cloud_dp = (
            vllm_config.parallel_config.lwd_config.cloud_dp_size
        )
        # per-engine 定向发布面:{engine_index: router channel / publisher}
        self._edge_senders: dict[int, Any] = {}
        # coordinator 客户端按 (instance, engine) 一一对应(FIRST_REQ/stats)
        self._lwd_coord_clients: dict[tuple[int, int], LwdCoordClient] = {}
        # 批队列总深度(embed+unembed 合计在飞上限)与 EMBED 保留预算:
        # 总深度随云实例数 N 扩容(公式唯一实现见
        # vllm.config.lwd.lwd_edge_batch_queue_depth,N=1 退化为单实例
        # 历史值 4);EMBED 派发以保留预算为门槛而非总深度——多实例下
        # 若仍以总深度为门槛,持续 prefill 会把 UNEMBED 挤出队列
        # (decode token 交付停摆)。worker 侧 buffer 池深经
        # model_runner 以同公式同源贯通,不得单独改此值。
        self._lwd_batch_queue_depth = lwd_edge_batch_queue_depth(vllm_config)
        if self._lwd_batch_queue_depth is None:
            raise RuntimeError(
                "[Lwd] edge engine selected but batch queue depth unresolved "
                "(lwd_config missing or not enabled prefill_only edge); "
                "check additional_config['lwd_config'] parsing"
            )
        self._lwd_embed_slots = min(
            LWD_EDGE_BASE_DEPTH, self._lwd_batch_queue_depth
        )
        logger.info(
            "[Lwd] edge batch queue depth=%d (embed_slots=%d, instances=%d)",
            self._lwd_batch_queue_depth,
            self._lwd_embed_slots,
            self._lwd_instance_count,
        )
        # HELLO 载荷按 (instance, engine) 留存(泵线程收首个 HELLO 时写入,
        # 构造线程 _lwd_cross_check_hello 在集合屏障后消费;dict 单键
        # 赋值原子,重复 HELLO 幂等跳过)
        self._lwd_peer_hellos: dict[tuple[int, int], LwdHelloNotify] = {}
        # 已派发待收割的批队列 (kind, payload, t_dispatch, future):
        # kind="unembed" payload=notify;kind="embed" payload=scheduler_output。
        # 派发不收割,队首 FIFO 收割——必须全局单队列:executor 的 FutureWrapper
        # 按底层队列序排水,交错收割会连带等错批。t_dispatch 供
        # [Lwd][sched] harvest wait(派发→收割)度量批在队列里的滞留时长
        self._lwd_batch_queue: deque = deque()
        # 传输模式标记:泵线程 HELLO 分支按此决定是否 retarget
        # (full_duplex 单通道无发布面换址概念)
        self._lwd_zmq_mode = "full_duplex"
        self._lwd_setup_zmq_router_dealer(config)
        # self._lwd_setup_zmq_publisher_subscriber(config)
        self.scheduler.lwd_edge_publishers = self._edge_senders
        # FIRST_REQ 唤醒钩子:仅有多云 engine coordinator 时接上
        self.scheduler.lwd_first_req_hook = (
            self._lwd_send_first_req if self._lwd_coord_clients else None
        )
        logger.info(
            "[Lwd] edge engine assembled: POST_OUT bind %s, "
            "engines=%d, instances=%d, coord_clients=%d",
            config.lwd_post_out_bind_endpoint(),
            self._lwd_cloud_dp,
            self._lwd_instance_count,
            len(self._lwd_coord_clients),
        )

    # ------------------------------------------------------------------ #
    # 通信面                                                              #
    # ------------------------------------------------------------------ #
    def _lwd_setup_zmq_publisher_subscriber(self, config: LwdConfig) -> None:
        """构造期装配通信面(与云侧 _lwd_setup_zmq 同构):
        bind POST_OUT 订阅面(内部搬运线程)+ 延迟连接的 PRE_OUT 发布面,
        云端点由 HELLO 通告决定(边不预知云地址);泵线程泵取搬运线程攒下
        的消息并分发;HELLO 屏障超时(hello_timeout_s)fail-fast。"""
        if config.instance_count > 1:
            # push_pull 双单向面无按实例寻址能力(需 ROUTER/DEALER 的
            # identity 信封),多实例在此 fail-fast 而非静默串线
            raise ValueError(
                "[LWD] edge-cloud multi-instance (instance_parallel_size "
                f"= {config.instance_count}) requires the router/dealer "
                "ZMQ mode; push_pull is single-instance only"
            )
        self._edge_receiver = LwdControlSubscriber(
            config.lwd_post_out_bind_endpoint(),
            bind=True,
            decoder=lwd_decode_cloud_notify,
        )
        self._edge_sender = LwdControlPublisher(
            None,
            bind=False,
            queue_max=config.publish_queue_max,
            encoder=lwd_encode_notify,
        )
        hello_event = threading.Event()
        pump = threading.Thread(
            target=self._lwd_pump_cloud_messages_push_pull,
            args=(hello_event,),
            name="lwd-post-in",
            daemon=True,
        )
        pump.start()
        if not hello_event.wait(config.hello_timeout_s):
            self._lwd_shutdown_planes()
            raise RuntimeError(
                f"[Lwd] edge engine init failed: no cloud HELLO within "
                f"{config.hello_timeout_s}s on POST_OUT "
                f"(bind {config.lwd_post_out_bind_endpoint()}; check cloud "
                f"master_addr connectivity and POST_OUT port)"
            )
        self._lwd_cross_check_hello()

    def _lwd_pump_cloud_messages_push_pull(
        self,
        hello_event: threading.Event,
    ) -> None:
        """push_pull 模式(_lwd_setup_zmq_publisher_subscriber)的
        POST_OUT 泵线程:泵取搬运线程攒下的消息,按消息类型分发。

        HELLO -> retarget PRE_OUT(云端点唯一事实源,首拍一次通告;
        队满自旋重试)。
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
            for msg in receiver.recv_batch(
                timeout_ms=LWD_POST_OUT_PUMP_TIMEOUT_MS
            ):
                if isinstance(msg, LwdHelloNotify):
                    endpoint = f"tcp://{msg.pre_out_host}:{msg.pre_out_port}"
                    if not hello_event.is_set():
                        logger.info(
                            "[Lwd] cloud discovered via HELLO: PRE_OUT -> %s",
                            endpoint,
                        )
                        # 载荷留存供构造线程互校(调度三元组,
                        # 见 _lwd_cross_check_hello)
                        self._lwd_peer_hello = msg
                    while not publisher.retarget(endpoint):
                        if receiver.closed:
                            break
                        logger.warning(
                            "[Lwd] PRE_OUT retarget deferred (queue full), "
                            "retrying",
                        )
                        threading.Event().wait(0.05)
                    hello_event.set()
                elif isinstance(msg, LwdC2eNotify):
                    self._lwd_dispatch_c2e_notify(msg)
                else:
                    self._lwd_drop_unexpected_frame(msg)

    def _lwd_setup_zmq_router_dealer(self, config: LwdConfig) -> None:
        """full_duplex:每云 engine 一条 ROUTER 端口(pre_out_port+engine),
        每条服务 N 个实例 DEALER。集合屏障收齐 N*D 个 (instance, engine)
        HELLO 才放行;通过后与各通道 identity 集合交叉对账,再做逐 engine
        调度三元组互校。"""
        receivers: dict[int, Any] = {}
        for eng in range(self._lwd_cloud_dp):
            ch = create_edge_zmq_channel(
                dp_rank=eng,
                pre_out_port=config.pre_out_port,
                name=f"lwd-edge-eng{eng}",
                instance_count=config.instance_count,
            )
            receivers[eng] = ch
            self._edge_senders[eng] = ch
        self._edge_receivers = receivers
        # 兼容旧的单通道引用(单 engine 时等价)
        self._edge_receiver = receivers[0]
        hello_event = threading.Event()
        ready: set[tuple[int, int]] = set()
        self._lwd_hello_ready = ready
        pump = threading.Thread(
            target=self._lwd_pump_cloud_messages_router_dealer,
            args=(hello_event, ready, config.instance_count),
            name="lwd-post-in",
            daemon=True,
        )
        pump.start()
        expected = config.instance_count * self._lwd_cloud_dp
        if not hello_event.wait(config.hello_timeout_s):
            missing = sorted(
                {
                    (i, e)
                    for i in range(config.instance_count)
                    for e in range(self._lwd_cloud_dp)
                }
                - ready
            )
            self._lwd_shutdown_planes()
            raise RuntimeError(
                f"[Lwd] edge engine init failed (full_duplex): only "
                f"{len(ready)}/{expected} (instance, engine) HELLOs within "
                f"{config.hello_timeout_s}s (missing={missing}); check every "
                f"cloud instance is up with a distinct "
                f"--instance-parallel-start-rank and all "
                f"--cloud-dp-size engines are running"
            )
        # 交叉对账:HELLO 实例集合 vs 各 engine 通道 identity 集合
        wire_ready: set[int] = set()
        for ch in receivers.values():
            wire_ready |= ch.ready_instance_ids()
        hello_instances = {iid for iid, _ in ready}
        if wire_ready != hello_instances:
            logger.error(
                "[Lwd] HELLO instance set %s != channel identity set %s: "
                "a cloud instance stamps HELLO/c2e with an instance_id "
                "different from its DEALER identity (routing will "
                "misdeliver); check --instance-parallel-start-rank "
                "consistency on that instance",
                sorted(hello_instances),
                sorted(wire_ready),
            )
        # 集合屏障收齐后做调度三元组互校(逐 (instance, engine) vs 本地)
        self._lwd_cross_check_hello()

    def _lwd_pump_cloud_messages_router_dealer(
        self,
        hello_event: threading.Event,
        ready: set[tuple[int, int]],
        instance_count: int,
    ) -> None:
        """轮询 D 条 ROUTER 通道,按 (instance, engine) 双维度收集 HELLO。

        engine 由端口分离(HELLO.engine_index 必须与接收端口一致),
        instance 由 DEALER identity/业务戳决定;重复 HELLO 幂等,越界即
        error 丢弃。收齐 N*D 才放行。
        """
        receivers = self._edge_receivers
        expected = instance_count * self._lwd_cloud_dp
        while not any(ch.closed for ch in receivers.values()):
            drained = False
            for eng, receiver in receivers.items():
                if receiver.closed:
                    continue
                for msg in receiver.recv_batch(timeout_ms=0):
                    drained = True
                    if isinstance(msg, LwdHelloNotify):
                        iid = msg.instance_id
                        if not 0 <= iid < instance_count:
                            logger.error(
                                "[Lwd] drop HELLO with out-of-range "
                                "instance_id=%d (expect [0, %d)); this cloud "
                                "has a misconfigured "
                                "--instance-parallel-start-rank",
                                iid,
                                instance_count,
                            )
                            continue
                        if msg.engine_index != eng:
                            logger.error(
                                "[Lwd] drop HELLO: engine_index=%d arrived "
                                "on the ROUTER port of engine %d (cloud "
                                "PRE_OUT port/engine config mismatch)",
                                msg.engine_index,
                                eng,
                            )
                            continue
                        key = (iid, eng)
                        if key in ready:
                            continue
                        ready.add(key)
                        self._lwd_peer_hellos[key] = msg
                        if msg.coord_stats_addr:
                            self._lwd_coord_clients[key] = LwdCoordClient(
                                msg.coord_stats_addr
                            )
                        logger.info(
                            "[Lwd] cloud (inst=%d, engine=%d) ready via "
                            "HELLO (%d/%d)",
                            iid,
                            eng,
                            len(ready),
                            expected,
                        )
                        if len(ready) >= expected:
                            hello_event.set()
                    elif isinstance(msg, LwdC2eNotify):
                        self._lwd_dispatch_c2e_notify(msg)
                    else:
                        self._lwd_drop_unexpected_frame(msg)
            if not drained:
                threading.Event().wait(0.01)

    def _lwd_dispatch_c2e_notify(self, msg: LwdC2eNotify) -> None:
        """两种 ZMQ 模式共用的 C2eNotify 分发(见泵线程 docstring)。"""
        iid = msg.instance_id
        if not 0 <= iid < self._lwd_instance_count:
            # 云侧 start-rank 配错(fail-loud 不 fail-fast:坏戳不污染
            # 批队列,实例屏障/看门狗另行定位)
            logger.error(
                "[Lwd] drop C2eNotify with out-of-range instance_id=%d "
                "(expect [0, %d)); reqs=%d down_seqno=%s — this cloud has "
                "a misconfigured --instance-parallel-start-rank",
                iid,
                self._lwd_instance_count,
                len(getattr(msg, "req_ids", []) or []),
                getattr(msg, "down_seqno", None),
            )
            return
        bucket = self._lwd_c2e_pending.setdefault(iid, deque())
        # 元素带到达时间戳:消费滞后(c2e_wait=到达→派发)是
        # "边消费不动 → 云 publisher 队满小睡"闭环的前置指标
        bucket.append((msg, time.monotonic()))
        if len(bucket) > LWD_C2E_META_QUEUE_MAX:
            # 本地桶水位越过告警线:引擎步排水停滞(正常背压应止步于
            # 通道层 per-identity FIFO)
            logger.warning(
                "[Lwd] c2e pending bucket for instance %d at %d "
                "(edge engine consumption stalled?)",
                iid,
                len(bucket),
            )
        logger.info(
            "[Lwd][edge-ctrl] C2eNotify inst=%d reqs=%d down_seqno=%s",
            iid,
            len(getattr(msg, "req_ids", []) or []),
            getattr(msg, "down_seqno", None),
        )
        self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

    def _lwd_drop_unexpected_frame(self, msg: Any) -> None:
        logger.warning("[Lwd] drop unexpected POST_OUT frame %r", type(msg))

    def _lwd_shutdown_planes(self) -> None:
        """关停全部通信面(幂等):各 engine ROUTER 通道 + coord 客户端。"""
        receivers = getattr(self, "_edge_receivers", None)
        if receivers:
            for ch in receivers.values():
                ch.shutdown()
        else:
            receiver = getattr(self, "_edge_receiver", None)
            if receiver is not None:
                receiver.shutdown()
            publisher = getattr(self, "_edge_sender", None)
            if publisher is not None and publisher is not receiver:
                publisher.shutdown()
        for client in getattr(self, "_lwd_coord_clients", {}).values():
            client.shutdown()

    def _lwd_send_first_req(self, instance_id: int, engine_index: int) -> None:
        """FIRST_REQ 唤醒:按 (instance, engine) 找对应 coordinator 客户端。"""
        client = self._lwd_coord_clients.get((instance_id, engine_index))
        if client is None:
            return
        client.send_first_req(engine_index)

    def _lwd_cross_check_hello(self) -> None:
        """HELLO 调度三元组互校(构造线程;集合屏障之后)。

        max_num_batched_tokens/max_num_seqs 是边 cap 公式与云 preflight
        的同源输入(设计 §2.2/§5.2 的部署对齐前提),不一致即构造期
        fail-fast——不一致不会让链路报错,只会让云 preflight 恒 hold、
        prefill 被 decode 静默串行(真机实录:边 factor=1/云 factor=4
        时 chunk 8191 + decode 4 > 预算 8192,prefill 积压 280 步)。

        多实例/多 engine 扩展:各 (instance, engine) 三元组先做全簇互检
        ——边 cap 公式读的是一份本地预算,_lwd_spec_factor 也是全局单值,
        异构会导致部分 engine 的 preflight 与边 cap 永久错位;云集群
        部署本就要求同构,互检不一致即 fail-fast。
        num_speculative_tokens 与本地不一致时以对端为准采纳(边本地
        取不到合法来源)并复核活性下界。对端为旧版(三值全 0)时
        仅告警不拦截。"""
        hellos: dict[tuple[int, int], LwdHelloNotify] = dict(
            getattr(self, "_lwd_peer_hellos", None) or {}
        )
        # push_pull 兼容路径仍用单条 _lwd_peer_hello
        peer = getattr(self, "_lwd_peer_hello", None)
        if peer is not None and not hellos:
            hellos[(peer.instance_id, peer.engine_index)] = peer
        if not hellos:
            return
        # 多 engine 要求 wire_version>=3(HELLO 携带 engine_index/
        # coord_stats_addr),否则无法按身份建立端点表
        if self._lwd_cloud_dp > 1:
            stale = sorted(
                key for key, h in hellos.items()
                if h.wire_version < LWD_WIRE_VERSION
            )
            if stale:
                self._lwd_shutdown_planes()
                raise RuntimeError(
                    f"[LWD] cloud (instance, engine) {stale} wire_version < "
                    f"{LWD_WIRE_VERSION}: 多 DP 部署要求 HELLO 携带 "
                    f"engine_index/coord_stats_addr,两侧必须同版本部署"
                )
        # mrope + 旧协议 fail-fast
        if self.vllm_config.model_config.uses_mrope:
            stale = sorted(
                key for key, h in hellos.items()
                if h.wire_version < LWD_WIRE_VERSION
            )
            if stale:
                self._lwd_shutdown_planes()
                raise RuntimeError(
                    f"[LWD] edge/cloud wire protocol mismatch: model uses "
                    f"mrope but cloud HELLO wire_version<{LWD_WIRE_VERSION} "
                    f"on (instance, engine) {stale}——云侧仍按数据面 aux 帧"
                    f"协议运行,mrope positions 将静默缺失(msgspec 容忍未知"
                    f"字段,旧云不会报错),两侧必须同版本部署"
                )
        legacy = sorted(
            key for key, h in hellos.items()
            if h.max_num_batched_tokens == 0
        )
        if legacy:
            logger.warning(
                "[Lwd] HELLO from (instance, engine) %s carries no "
                "scheduling triple (legacy peer); edge/cloud config "
                "alignment UNCHECKED for them",
                legacy,
            )
        live = {
            key: h for key, h in hellos.items()
            if h.max_num_batched_tokens != 0
        }
        if not live:
            return
        triples = {
            key: (
                h.max_num_batched_tokens,
                h.max_num_seqs,
                h.num_speculative_tokens,
            )
            for key, h in live.items()
        }
        # 全簇互检(先于逐项 vs 本地:异构即整簇拒绝)
        ref_key = min(triples)
        divergent = sorted(
            key for key, t in triples.items() if t != triples[ref_key]
        )
        if divergent:
            self._lwd_shutdown_planes()
            raise RuntimeError(
                f"[LWD] cloud (instance, engine) disagree on the scheduling "
                f"triple (max_num_batched_tokens, max_num_seqs, "
                f"num_speculative_tokens): {ref_key}="
                f"{triples[ref_key]} vs "
                f"{[(key, triples[key]) for key in divergent]}; "
                f"边 cap 公式读一份本地预算,异构 engine 的 preflight 将"
                f"永久错位——云集群须同构拉起"
            )
        sched = self.vllm_config.scheduler_config
        spec_config = getattr(self.vllm_config, "speculative_config", None)
        local = (
            sched.max_num_batched_tokens,
            sched.max_num_seqs,
            getattr(spec_config, "num_speculative_tokens", None) or 0,
        )
        remote = triples[ref_key]
        if local[:2] != remote[:2]:
            self._lwd_shutdown_planes()
            raise RuntimeError(
                f"[LWD] edge/cloud scheduling config mismatch: "
                f"(max_num_batched_tokens, max_num_seqs) "
                f"edge={local[:2]} != cloud={remote[:2]}; "
                f"cap/preflight 公式两侧同源输入不一致会让 preflight 恒 "
                f"hold、prefill 被 decode 静默串行——两侧拉起配置须同值"
            )
        if local[2] != remote[2]:
            # spec 系数以云为准:它预留的是云 decode 步的草稿预算,边侧
            # 本地取不到合法来源(边实例带 spec 配置会拖起 draft 机制)。
            # 采纳对端值而非 fail-fast——部署上边侧无需再配 spec。
            logger.warning(
                "[Lwd] adopt cloud spec_factor=%d over edge local %d "
                "(edge has no legitimate local source for it)",
                remote[2] + 1, local[2] + 1,
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
            "[Lwd] HELLO scheduling triple verified: %s (instances %s)",
            local, sorted(triples),
        )

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
        永不收割交付。"""
        return (super().has_work() or any(self._lwd_c2e_pending.values())
                or self._lwd_batch_queue)

    def _lwd_edge_step(self) -> tuple[dict[int, object] | None, bool]:
        """单步编排(批队列异步化+多批在飞,embed 优先):收割队首直至
        清空 -> prefill 连续派发(优先占深度配额) -> 消费云载荷(剩余
        配额派发 UNEMBED)-> 返回({client_index: 云侧结果输出} | None,
        prefill 是否有派发)。

        输出按请求的实例 pin 分组(client_index == instance,pin 校验
        保证):前端 attach 按字典 key 送达对应客户端,多实例下各组
        各回各家;单实例退化为 {0: ...}(与原行为等价)。

        embed/unembed 均派发入队不收割,队首全局 FIFO 收割(序与
        executor 排水序同构,不可交错);突发时一步连派至深度上限
        喂饱 worker。派发序 embed 优先:196 并发下每云步都产 c2e,
        若 unembed 先吃配额会把 embed 挤出本步(EMBED 被 decode 回程
        挤住的饿死形态),TTFT 劣化——故 embed 先占位,unembed 用剩余
        配额,代价是 decode token 交付最多延后 1-2 步(顶部全量收割
        + c2e 队列背压兜底,不会饿死)。

        [Lwd][sched] 三类日志(饿死分析锚点):每批 harvest(含队列滞留
        wait)、每批 dispatch(含 ahead 队列构成/c2e 水位)、每步汇总。"""
        outputs_by_client: dict[int, list] = defaultdict(list)
        finished_by_client: dict[int, set] = defaultdict(set)
        prefill_work = False
        h_emb = h_unemb = 0
        while self._lwd_batch_queue:
            kind, payload, t_dispatch, future = self._lwd_batch_queue.popleft()
            if kind == "unembed":
                self._lwd_deliver_unembed(
                    payload, future, outputs_by_client, finished_by_client
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
        # EMBED 派发门槛 = 保留预算(队列内 embed 计数),而非总深度:
        # 多实例下总深度随 N 扩容,若仍以总深度为门槛,持续 prefill 会
        # 把 UNEMBED 挤出本步(decode token 交付停摆);保留预算使
        # embed 优先有界,UNEMBED 消费剩余配额。N=1 时总深度 ==
        # 保留预算,与历史行为逐位等价。
        while (self._lwd_queue_mix()[0] < self._lwd_embed_slots
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
            self._lwd_c2e_pending_total(),
        )
        if outputs_by_client:
            step_outputs = {
                client_index: EngineCoreOutputs(
                    outputs=outs,
                    finished_requests=finished_by_client[client_index] or None,
                )
                for client_index, outs in outputs_by_client.items()
            }
            return step_outputs, prefill_work
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

    def _lwd_c2e_pending_total(self) -> int:
        """全部实例桶的 c2e 积压总数(水位日志用)。"""
        return sum(len(b) for b in self._lwd_c2e_pending.values())

    def _lwd_edge_consume_c2e(self) -> int:
        """消费 c2e 通告:按实例轮询取条,派发 UNEMBED 批入批队列
        (异步收割)。每轮每桶至多取 1 条直至总配额用尽——单实例退化为
        顺序取(与原单队列行为等价);多实例下繁忙实例不再按到达比例
        挤占其他实例的 token 交付配额。
        云侧有活请求才产 meta(build_hidden_payload 空则返回 None),
        每条 entry 行数>=1——通告必有行,无需无行分流。
        返回本步派发条数。
        DISABLE_DOWN 诊断:通告只携带 token_ids(hidden=0)时跳过 worker
        派发,以 future=None 直通收割,不挂任何 recv。"""
        quota = self._lwd_batch_queue_depth - len(self._lwd_batch_queue)
        notifies: list[tuple[LwdC2eNotify, float]] = []
        while len(notifies) < quota:
            progressed = False
            for iid in sorted(self._lwd_c2e_pending):
                bucket = self._lwd_c2e_pending[iid]
                if not bucket:
                    continue
                notifies.append(bucket.popleft())
                progressed = True
                if len(notifies) >= quota:
                    break
            if not progressed:
                break
        # 不删空桶、不重绑字典:键数 = 实例数恒定,无膨胀;重绑会让泵
        # 线程的 append 落在旧字典对象上,c2e 通告静默丢失(见 __init__)
        for notify, t_arrive in notifies:
            if _lwd_disable_down_enabled() and notify.token_ids:
                logger.info(
                    "[Lwd][sched] edge direct-deliver seqno=%s reqs=%d "
                    "(DISABLE_DOWN, no unembed dispatch)",
                    notify.down_seqno, len(notify.req_ids),
                )
                future = None
            else:
                future = self._lwd_dispatch_unembed(notify, t_arrive)
            self._lwd_batch_queue.append(
                ("unembed", notify, time.monotonic(), future)
            )
        return len(notifies)

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
            n_unemb, n_emb, self._lwd_c2e_pending_total(),
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
            "[Lwd][sched] edge dispatch-unembed seqno=%s inst=%d reqs=%d "
            "rows=%d ahead_unemb=%d ahead_emb=%d c2e_wait=%.2fms "
            "c2e_pending=%d",
            notify.down_seqno, notify.instance_id, len(notify.req_ids),
            sum(notify.num_accepted_tokens), n_unemb, n_emb,
            (time.monotonic() - t_arrive) * 1000,
            self._lwd_c2e_pending_total(),
        )
        return self.model_executor.execute_model(
            unembed_batch, non_block=True
        )

    def _lwd_deliver_unembed(
        self,
        notify: LwdC2eNotify,
        future,
        outputs_by_client: dict[int, list],
        finished_by_client: dict[int, set],
    ) -> None:
        """UNEMBED 批收割侧:等 worker 应答取采样 token 表,逐请求交付。
        不感知提交时机,只消费 (notify, future)。应答契约:
        token ids 按 ModelRunnerOutput 的 req_ids x sampled_token_ids
        按位对齐还原(批的 req_ids 原样下发,worker 逐位回填);请求
        缺席或行无 token = unembed 失败,ERROR 优先于云侧完成码;
        迟到载荷幂等丢弃。

        输出按请求的实例 pin 归组(client_index == instance,入口 pin
        校验保证):与原生 update_from_output 的 outputs[request.
        client_index] 分组同构,前端 attach 据此送达对应客户端。"""
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
        logger.info(
            "[Lwd][perf] harvest dur=%.2fms", (time.monotonic() - _t) * 1000
        )
        routing = self.scheduler._req_id_to_instance
        for index, request_id in enumerate(notify.req_ids):
            finish_reason = self._lwd_finish_code(notify, index)
            finished = finish_reason is not None
            sampled_token_ids = (
                list(sampled_token_map.get(request_id, []))
                if sampled_token_map else []
            )
            LwdDebug.edge_tokens_delivered(  # [lwd-debug]
                request_id, sampled_token_ids, finish_reason,
                self.vllm_config,
            )
            if not sampled_token_ids and not finished:
                # 行在批里但无 token 且未完结 = unembed 失败,ERROR 优先于
                # 云侧码(已完结的空行属正常,如 DISABLE_DOWN 下的 ABORT)
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
            client_index = routing.get(request_id, 0)
            outputs_by_client[client_index].append(
                EngineCoreOutput(
                    request_id, sampled_token_ids,
                    finish_reason=finish_reason,
                )
            )
            if finished:
                finished_by_client[client_index].add(request_id)

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
