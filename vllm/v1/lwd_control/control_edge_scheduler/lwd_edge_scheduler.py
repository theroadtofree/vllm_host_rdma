"""边侧调度器:chunk 化 prefill 组批 + 控制面发布(notify/abort/seqno)。

原生 AsyncScheduler 的两个前提在边侧不成立:prompt 算完会转 decode、
请求只能由模型输出终结。边侧只做 embedding(执行层无 decode),故需
专用调度器接管请求的边侧生命周期:

  入队 -> EMBEDDING(chunk 组批/范围预告)
       -> 嵌入完结:走原生 finish_requests 清出调度器(释放边侧 KV
          簿记、通知 worker 释放缓存),登记 awaiting
       -> AWAITING(等待云结果;前端未收到输出继续等待)
       -> 云结果终结(lwd_edge_deliver_tokens);迟到结果幂等丢弃。

纯 prefill 的实现依据:schedule() 全量复用原生——边侧请求从不产生
输出 token(num_tokens_with_spec 恒等于 num_prompt_tokens),且嵌入
完结当步即被清出调度器,原生 RUNNING 段每步只会调度剩余 prefill,
decode 分支不可达。

chunk 组批约束(lwd_chunked_prefill_design.md):prompt 按
c = min(剩余, chunk 单元 C) 切块,批 = 续传 chunk 优先 + waiting FCFS
前缀,Σc ≤ cap(云侧预算 − 在途数 × spec 系数,边云同值前提下读本地
配置);续传队首装不下时的 cap 截断块独占一批(云侧按批内最大条目
复现 chunk 边界,设计 §2.4)。一个批 = 一条批量 RangeNotify = 一个
seqno = 一次 UP send,数据面 chunk 流按批连续,跨请求交错/重组复杂度
仍为零。
"""

from __future__ import annotations

import asyncio
import hashlib
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.v1.core.sched.output import (
    LwdBatch,
    LwdBatchType,
    LwdEmbedBatch,
    LwdUnembedBatch,
    SchedulerOutput,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LwdAbortNotify,
    LwdC2eNotify,
    LwdFinishNotify,
    LwdGenChainNotify,
    LwdRangeItem,
    LwdRangeNotify,
    LwdRequestNotify,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_binding import (
    LwdEdgeBindingTable,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
)
from vllm.v1.lwd_debug import LwdControlLog
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
        LwdControlPublisher,
    )
    from vllm.v1.lwd_control.control_communication.lwd_router_channel import (
        LwdControlRouterChannel,
    )
    from vllm.v1.request import Request

logger = init_logger(__name__)

# 有序发布滞留的 fail-fast 时限:队首消息滞留超过该时长视为云侧通道
# 死亡(引擎步内抛出,与旧版 add 路径重试耗尽即错的语义同级,但不再
# 用 sleep 阻塞主循环)
_LWD_PENDING_FAIL_AFTER_S = 30.0

# probe 看门狗时限:远大于钳制后的 connect_timeout(3s)+云侧处理余量,
# 只兜「probe 线程静默死亡」的底,正常 RTT 不会触发
_LWD_PROBE_WATCHDOG_S = 15.0


@dataclass(frozen=True)
class _LwdMediaIdentity:
    """协议层 ``_MediaItem`` 的边侧实现:多模态内容身份进 HMAC 链。

    digest 取 ``MultiModalFeatureSpec.identifier``(encoder 缓存键,含
    LoRA 前缀)的 sha256——同图同配置同 digest,换图/换 LoRA 即分叉;
    ``mm_position`` 的 offset/length 即该 item 在 prompt 中的占位区间。
    图像占位展开 token 与内容无关,不掺身份则不同图共享块哈希,
    云侧会跨图误命中并复用旧图 KV。"""

    modality: str
    digest: bytes
    offset: int
    length: int


class _LwdProbeRuntime:
    """前缀 probe 专用后台线程:常驻事件循环 + 共享 ClientSession。

    根治「add_request 在引擎主循环 asyncio.run + 每请求新建 TCP」:
    probe(HMAC 建链 + HTTP RTT)全部移出主循环,多请求 probe 在单
    事件循环上并发交错;连接池常驻,消除每请求建连开销。结果经
    done-callback 投回调度器的线程安全队列,由主循环统一定入队
    (全部调度器状态变更只发生在主线程,无锁)。
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._session = None
        self._thread = threading.Thread(
            target=self._run, name="lwd-prefix-probe", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def stop(self) -> None:
        """停事件循环并关会话(submit 失败重建路径的旧实例回收,
        防线程/连接泄漏)。非阻塞:清理排在循环内异步执行。"""
        def _cleanup() -> None:
            if self._session is not None:
                asyncio.ensure_future(self._session.close())
            self._loop.call_later(1.0, self._loop.stop)

        try:
            self._loop.call_soon_threadsafe(_cleanup)
        except Exception:  # noqa: BLE001  循环已死,忽略
            pass

    async def ensure_session(self):
        if self._session is None:
            import aiohttp

            self._session = aiohttp.ClientSession()
        return self._session

    def submit(self, coro) -> "asyncio.Future":
        return asyncio.run_coroutine_threadsafe(coro, self._loop)


class LwdEdgeScheduler(LwdBaseScheduler):
    """纯 prefill 调度语义 + 控制面出口(notify/abort/seqno)。

    云侧复用(``control_channel`` 注入,registry 非空):请求经
    ``LwdEdgeBindingTable`` 绑定目标云,全部通告按 dest 定向发布;
    seqno 按 (cloud_id) 对分域计数。1E1C(publisher 注入)行为与原
    实现一致。
    """

    def __init__(
        self,
        *args,
        publisher: LwdControlPublisher | None = None,
        control_channel: LwdControlRouterChannel | None = None,
        **kwargs,
    ) -> None:
        """publisher(1E1C)/control_channel(云侧复用)经构造注入(与调度器
        同生命周期);两者互斥,云侧复用只认 control_channel。"""
        super().__init__(*args, **kwargs)
        self.lwd_edge_publisher = publisher
        self.lwd_edge_channel = control_channel
        # seqno 按 (cloud_id) 对分域:UP 数据通道按连续号序配对,号只能
        # 分配给真正上 wire 的块(peek-then-advance,发布成功才进位);
        # 1E1C 恒用 self._lwd_cloud_id 单键,行为与原单值计数器一致
        self._lwd_seqno: dict[int, int] = {}
        # 多边多云身份维度:edge_id 取自身;cloud_id 为默认绑定云。
        lwd_cfg = self.vllm_config.parallel_config.lwd_config
        self._lwd_edge_id = getattr(lwd_cfg, "edge_id", 0)
        self._lwd_cloud_id = getattr(lwd_cfg, "cloud_id", 0)
        # 请求-云绑定表:入口登记、finish/abort 解绑、通告/abort 按它定向
        self._lwd_binding = LwdEdgeBindingTable()
        # 收尾记账(§11.4):边侧请求在"嵌入完结"时即清出调度器(awaiting
        # 接管),终结时拿不到 Request 对象,故在入口留 prompt ids、在投递处
        # 累积生成 ids,终结即用租户密钥建全量块哈希链上报云侧并清理。
        self._lwd_finish_prompt_ids: dict[str, list[int]] = {}
        self._lwd_finish_gen_ids: dict[str, list[int]] = {}
        # 多模态内容身份留档(request_id -> media items):Request 终结后
        # mm_features 不可达,add_request 入口一次提取供 probe/finish/
        # 生成段建链续用;与 prompt_ids 同生命周期(finish/abort 摘除)。
        # 三处建链共用同一份快照,保证查链与存链口径一致。
        self._lwd_finish_media_items: dict[str, list[Any]] = {}
        # processor 指纹惰性缓存(sha256(模型 + mm_processor_kwargs)):
        # 只掺入媒体覆盖块的 HMAC——纯文本链逐位不变,新旧边混布兼容。
        # 协议要求带 media_items 的建链必须持指纹,probe client 与
        # _lwd_prefix_hasher 两个构造点都要传,缺一侧即建链必抛。
        self._lwd_processor_fp: bytes | None = None
        # 生成段链增量上报水位(request_id -> 已上报满块数):只在满块数
        # 增加时发一条(全量替换语义),避免逐步刷控制面。
        self._lwd_gen_chain_sent: dict[str, int] = {}
        # 云侧块哈希粒度学习缓存(cloud_id -> 云 hash 粒度):probe 响应头
        # 携带云侧 hash_block_size(压缩/混合粒度模型上与边块粒度不同,
        # 如 DSv4 云 hash=8 而边块 128)。链按该粒度构建后,云侧采纳条件
        # len(链)==prompt//hash 粒度 恒成立,不再要求 prompt 为边块整数倍。
        # 未学习(首请求前)回退边块粒度,行为与学习前一致。
        self._lwd_hash_bs_by_cloud: dict[int, int] = {}
        # 云侧 mamba 对齐网格学习缓存(cloud_id -> scheduler_block_size,
        # 即云各 KV 组块大小 LCM):probe 响应头公告。学到即置
        # need_mamba_block_aligned_split=True 并用网格覆写
        # _mamba_block_aligned_split——边侧恢复上游"chunk 自动对齐"
        # 语义,max_num_batched_tokens 回归纯性能旋钮;0/未学到不重切。
        self._lwd_align_grid_by_cloud: dict[int, int] = {}
        # 按粒度缓存的 hasher(同租户密钥不同 block_size;数量极小,
        # 实际部署每云一个粒度)
        self._lwd_hasher_by_granularity: dict[int, Any] = {}
        # 请求 probe 链粒度记账(request_id -> 建链粒度):学习换挡只发生
        # 在重启后首个请求(probe 用旧粒度、finish 须用同一粒度才能与云侧
        # 存储的 manifest 对账),稳态请求 probe/finish 同粒度自然一致。
        self._lwd_probe_bs_by_req: dict[str, int] = {}
        # 探前缀客户端惰性缓存(复用 hasher/tenant_key,避免每请求读文件重建)
        self._lwd_probe_client_cache = None
        # probe 异步化(E1 根治):请求先投 probe 线程,结果回灌主循环后
        # 才走绑定/预告/入队。inflight 持有 (Request, nonce, t_submit)
        # (主循环移交所有权,前端不再触碰);nonce 区分同 id 重发的
        # 多次提交;abort 即摘 inflight(迟到的 probe 结果在 drain 按
        # 缺席/旧 nonce 丢弃);watchdog 保证 probe 线程静默死亡时请求
        # 也能超时直通(不会永久滞留)。
        self._lwd_probe_inflight: dict[str, tuple[Request, int, float]] = {}
        self._lwd_probe_nonce = 0
        # probe 完成队列(线程安全,单生产者 probe 线程/单消费者主循环):
        # 元素 (request_id, nonce, build_bs, (result, chain) | None)
        self._lwd_probe_done: queue.Queue = queue.Queue()
        # 引擎注入的唤醒钩(probe 完成时敲 input_queue WAKEUP;未注入=单测)
        self.lwd_probe_wakeup = None
        self._lwd_probe_runtime: _LwdProbeRuntime | None = None
        # 有序发布滞留队列(E4 根治):cloud_id -> deque[(message, t_enqueue)]。
        # 发布失败/通道背压时消息入队保序,引擎步首 flush 重投;RangeNotify
        # 的 seqno 进位推迟到实际发布成功(peek-then-advance 契约不变)。
        self._lwd_pending_publishes: dict[int, deque] = {}
        # 生成段链增量建链状态(E2 根治):request_id -> 已建满块链(list,
        # 只追加)。逐步只对新增满块算 HMAC,替代每步全量 O(N) 重建。
        self._lwd_gen_chain_state: dict[str, list[bytes]] = {}
        # 前缀缓存:manager 级关命中,配置级保留使能。两级拆分的原因:
        # - 必须关命中:命中会跳过 token 排程,首条 RangeNotify 的
        #   offset != 0,云侧按 offset==0 识别首块的约定失效;且被
        #   "命中"的块在 EMBED 批下从未写入 KV(边侧不落 KV),
        #   缓存登记残留会持续占用块池。关闭后命中恒空、free 直接
        #   归还块池。
        # - 不能在配置级关:request_block_hasher 只在配置级使能时
        #   创建,关掉则 Request.block_hashes 恒空,LwdRequestNotify
        #   带不出哈希链——云侧 prompt 是占位零值 token,前缀缓存
        #   只能靠边侧按真实内容算出的哈希链命中。
        self.kv_cache_manager.enable_caching = False
        # 引擎初始化对 0 KV 组的模型强制关 chunked prefill
        # (core.py:136-141,"model without KVCache")——边侧恰是 0 KV 组
        # (worker get_kv_cache_spec={}),该开关一关,原生 waiting 段对
        # 超预算 prompt 直接 break 不截断(scheduler.py:696-702),chunk
        # 切片无从谈起(逃生通道的"允许原生截断"同样依赖它)。原生关它
        # 的理由(encoder 模型无 KV 不能分块前向)对边侧不成立:边侧
        # forward 被 LWD 劫持,chunk 是纯调度记账,执行层只 embed 切片。
        # 此处重新打开;flag 在 schedule() 时才被读取,构造期生效。
        self.scheduler_config.enable_chunked_prefill = True
        # awaiting:嵌入完待云结果的 request_id -> 登记时刻(单调钟)。
        # 请求本体已清出调度器,此表是结果路径的唯一生命周期台账。
        self._lwd_awaiting: dict[str, float] = {}
        # 多模态请求的整 prompt mrope positions([3,N] CPU 张量,准入时
        # 一次性算好,按 chunk 切片随 LwdEmbedBatch 下发 worker);嵌入
        # 完结/abort 时摘除。key 在 = 该请求各 chunk 的 has_mrope 恒真。
        self._lwd_mrope_positions_dict: dict[str, torch.Tensor] = {}
        # 惰性解析的模型类静态方法 _get_mrope_input_positions(直接缓存
        # 方法本身而非模型类,语义直达;首用即验,不支持的族 fail-fast)。
        self._lwd_mrope_positions_fn = None
        # MTP 预算系数:decode 每请求每步消耗 1+k 个 token(草稿 token
        # 同样占预算与 KV);k = num_speculative_tokens(配置解析期已
        # 求值,缺省 None 按 0),无 spec 配置即 1。
        # 边云 max_num_batched_tokens/max_num_seqs 部署对齐(同值),故
        # cap 直接读本地配置(设计 §2.3,配置对齐为部署约束)
        spec_config = getattr(self.vllm_config, "speculative_config", None)
        num_spec = (
            getattr(spec_config, "num_speculative_tokens", None) or 0
        )
        self._lwd_spec_factor: int = 1 + num_spec
        # mixed 开关:与云侧调度器选择同源(scheduler_name,部署双侧同值)。
        # 非 mixed(prefill_first/decode_first 逃生通道)= 旧单请求组批语义:
        # 只取队首、允许原生截断成 chunk、不做整 prompt 核验
        from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
            LwdConfig,
        )

        self._lwd_mixed: bool = (
            LwdConfig.from_env_and_config(self.vllm_config).scheduler_name
            == "mixed"
        )
        # chunk 单元 C:prompt 按 min(剩余, C) 切块下发
        # (lwd_chunked_prefill_design.md)。C 取本地
        # long_prefill_token_threshold(原生调度本就按它截断 prompt 排程,
        # scheduler.py:408/690,双侧天然复现同一边界);未配置(≤0,默认)
        # 时退化为 max_num_scheduled_tokens(= 边云对齐预算,整 prompt 不
        # 超预算不切块,与整体组批形态连续过渡)。云侧不读本值,按每条
        # RangeNotify 的 max(items) 临时覆写阈值复现(设计 §2.3)。
        threshold = self.scheduler_config.long_prefill_token_threshold
        self._lwd_chunk_unit: int = (
            threshold if threshold > 0 else self.max_num_scheduled_tokens
        )
        # 活性 fail-fast:最坏在途(满员)下 cap 恒 < 1 即死锁配置
        # (设计 §2.6),拉起时直接拒绝
        if self._lwd_mixed and (
            self.max_num_scheduled_tokens
            - self.max_num_running_reqs * self._lwd_spec_factor
            < 1
        ):
            raise RuntimeError(
                f"[LWD] 死锁配置: max_num_batched_tokens("
                f"{self.max_num_scheduled_tokens}) - max_num_seqs("
                f"{self.max_num_running_reqs}) * spec_factor("
                f"{self._lwd_spec_factor}) < 1, 续传 chunk 永无可调度预算; "
                "请增大 max_num_batched_tokens 或减小 max_num_seqs"
            )

    # ------------------------------------------------------------------ #
    # 云侧复用选路与控制面出口                                            #
    # ------------------------------------------------------------------ #
    def _lwd_default_cloud_id(self) -> int:
        """默认绑定云(选路接入前固定第一台云;1E1C 即单云 id)。"""
        if self.lwd_edge_channel is None:
            return self._lwd_cloud_id
        from vllm.v1.lwd_control.control_communication.lwd_role_registry import (
            get_role_registry,
        )

        registry = get_role_registry()
        if registry is not None and registry.cloud_ids:
            return registry.cloud_ids[0]
        return self._lwd_cloud_id

    def _lwd_publish(self, message, cloud_id: int) -> bool:
        """控制面出口统一:云侧复用按 dest 定向 publish;1E1C 走单
        publisher。False = 未入队(队/FIFO 满),调用方退避重试。"""
        if self.lwd_edge_channel is not None:
            return self.lwd_edge_channel.publish(message, cloud_id)
        if self.lwd_edge_publisher is not None:
            return self.lwd_edge_publisher.publish(message)
        return False

    # ------------------------------------------------------------------ #
    # 有序发布(E4 根治):背压时滞留保序,步首 flush,主循环零 sleep      #
    # ------------------------------------------------------------------ #
    def _lwd_publish_ordered(self, message, cloud_id: int) -> bool:
        """按云保序发布:该云已有滞留则直接入队(保序),否则尝试即发,
        失败入队。返回 True = 已实际入 wire 队列。

        全类型控制消息共用同一有序通道:RequestNotify 必须先于本请求
        RangeNotify 到云,Finish/GenChain 不得越过在途预告——背压期间
        的保序只能经单一滞留队列保证。"""
        pending = self._lwd_pending_publishes.setdefault(cloud_id, deque())
        if pending:
            pending.append((message, time.monotonic()))
            return False
        if self._lwd_publish(message, cloud_id):
            return True
        pending.append((message, time.monotonic()))
        logger.warning(
            "[Lwd][edge-notify] publish deferred: cloud=%d type=%s "
            "(channel backpressure)",
            cloud_id, type(message).__name__,
        )
        return False

    def lwd_edge_flush_pending(self) -> None:
        """步首重投滞留消息(FIFO;RangeNotify 发布成功才进位 seqno)。

        peek-then-advance 契约保持:滞留期间计数器不动,任何新
        RangeNotify 在滞留清空前不得入队(lwd_edge_notify 守卫),
        flush 按序补投后号链依旧无空洞。队首滞留超
        _LWD_PENDING_FAIL_AFTER_S 视为云侧通道死亡,fail-fast(与旧版
        add 重试耗尽即错的同级语义,但全程不阻塞主循环)。"""
        now = time.monotonic()
        for cloud_id, pending in self._lwd_pending_publishes.items():
            while pending:
                message, t_enqueue = pending[0]
                if now - t_enqueue > _LWD_PENDING_FAIL_AFTER_S:
                    raise RuntimeError(
                        f"[LWD] control publish to cloud {cloud_id} stalled "
                        f"for {_LWD_PENDING_FAIL_AFTER_S}s "
                        f"({type(message).__name__} still pending; "
                        "cloud channel dead?)"
                    )
                if not self._lwd_publish(message, cloud_id):
                    break
                pending.popleft()
                if isinstance(message, LwdRangeNotify):
                    # seqno 进位绑定实际发布成功(与 lwd_edge_notify 直发
                    # 分支同一进位点),滞留补投后号链无空洞
                    self._lwd_seqno[cloud_id] = message.seqno + 1
                logger.info(
                    "[Lwd][edge-notify] deferred publish flushed: cloud=%d "
                    "type=%s lag=%.1fms",
                    cloud_id, type(message).__name__,
                    (now - t_enqueue) * 1000,
                )

    def lwd_has_pending_publishes(self) -> bool:
        """任一云有滞留消息(引擎 has_work 计入:滞留须靠步循环排干)。"""
        return any(self._lwd_pending_publishes.values())

    def schedule(self) -> SchedulerOutput:
        """chunk 化 prefill 组批 + 原生准入。

        EMBED 批的 LwdBatch(seqno/token 片段)由 lwd_edge_notify 在
        发布成功后挂批——seqno 必须与发布成功绑定。

        开新准入:云侧在途满员(lwd_edge_max_num_seqs_check 为 False)
        时本步空排、新开请求留 waiting 等云侧排水;running 尚有未发完
        embed 的续传不受闸门约束(先收尾再开新,亦防上限=1 时自锁)。"""
        LwdControlLog.flight(len(self.running), len(self._lwd_awaiting))
        if (
            not self.lwd_edge_max_num_seqs_check()
            and not self._lwd_has_prefill_chunk_inflight()
        ):
            return SchedulerOutput.make_empty()
        return self._lwd_schedule_batch()

    def _lwd_pick_prefill_batch(
        self,
    ) -> tuple[list[str], dict[str, int], int]:
        """选下一步 embed 批:续传优先 + FCFS 前缀组批,按 chunk 单元切块。

        返回 (req_ids, expected, cap):expected 为逐请求的本步应排 token
        数(步后 chunk 核验用,不等即配置发散 fail-fast);cap 为本步
        token 预算上界(云侧预算 − 在途数 × spec 系数,边云同值前提下
        本地配置即云侧预算),供 _lwd_schedule_batch 做预算钳制。

        chunk 语义(lwd_chunked_prefill_design.md §3.1):
        - 每个 chunk 自限:c = min(剩余 prompt, chunk 单元 C),云侧原生
          调度在阈值复现(T=批内最大)下逐请求推出同一边界;
        - 批内非末位 chunk 同质(T 复现的前提):云侧复现非末位块只靠
          阈值从上压,尺寸 < T 的非末位块会被阈值放过、被预算超排
          (实测:网格收边产出 768 非末位块与 1536 块同批,T=1536
          压不住 1512 的 prompt,整段 1512 被排进,步后核验炸)。
          网格收边(_legal_chunk)会按 prompt 长度产出 < 单元的非末位
          块,故组批强制:非末位块同尺寸且不小于任何末位块,违例即
          收批留待下一条 notify(空批恒合法,无饿死);唯一例外是
          批尾的预算截断块(预算耗尽复现,见下);
        - 续传(先收尾再开新):running 中未发完的 prefill 按 FCFS 序取
          c = min(剩余, C),Σ ≤ cap;
        - 开新:waiting(skipped 优先,原生准入失败回插者)按 FCFS
          迭代序取前缀,c = min(prompt, C),满足 Σchunk ≤ cap 且
          新增请求数 ≤ free_seq;不跳过队首(保 FCFS 与 seqno 链次序);
        - DeepSeek V4 总 prompt 长度 >=8000 的请求独占 prefill 批
          (含续传);短请求允许合批,不限制云侧 decode 混排;
        - 末位截断(续传/新开共用):批内最后一个条目允许被剩余额度
          截尾成 cap−used,与集中式 chunked prefill 的"预算耗尽截断"
          逐语义对齐——截断块恒为末位,边侧(§2.5)与云侧(§2.3)的
          预算钳制都靠预算耗尽精确复现它,非末位块由 R/C/T 定界与
          截断块尺寸无关。无此逃生,在途 decode 把 cap 压到 C 以下时
          大 prompt 永不可开新,流水线全程串行;
        priority 调度策略下 waiting 迭代序非弹出序,本前缀语义未适配
        (LWD 部署恒 FCFS,设计 M-10)。"""
        # 逃生通道(旧相位语义):单请求、不 cap、允许原生截断成 chunk、
        # 不做核验——两侧同配 prefill_first/decode_first 时逐字节
        # 复现旧行为
        if not self._lwd_mixed:
            # 续传优先(对齐目标分支 _lwd_pick_prefill_req_id):running
            # 中第一个未发完的 prefill 先收尾,否则 waiting 队首;
            # 缺了这半截,chunked prefill 强开后超长 prompt 的尾巴
            # 留在 running 永不被选中,chunk 流断流挂死
            running_prefill = next(
                (r for r in self.running
                 if r.num_computed_tokens < r.num_prompt_tokens),
                None,
            )
            if running_prefill is not None:
                return [running_prefill.request_id], {}, self.max_num_scheduled_tokens
            candidates = list(self.skipped_waiting) + list(self.waiting)
            picked = [candidates[0].request_id] if candidates else []
            return picked, {}, self.max_num_scheduled_tokens
        # cap 语义:批 Σchunk ≤ 预算 − 在途数 × spec 系数(云混排步里
        # decode 每请求占 1+k token,MTP 下 k = num_spec_tokens;在途数
        # 是 decode 人口的恒成立上界,设计 §2.2)。边云预算部署对齐,
        # 本地 max_num_scheduled_tokens 即云侧预算
        inflight = len(self.running) + len(self._lwd_awaiting)
        cap = self.max_num_scheduled_tokens - inflight * self._lwd_spec_factor
        free_seq = self.max_num_running_reqs - inflight
        picked: list[str] = []
        expected: dict[str, int] = {}
        used = 0
        if cap < 1:
            return picked, expected, cap
        unit = self._lwd_chunk_unit
        # 批内同质性跟踪(见 docstring「批内非末位 chunk 同质」):
        # nonlast_size = 批内非末位块的统一尺寸(未定 None);
        # batch_max = 批内最大 chunk(末位块也计入,T 由它决定)
        nonlast_size: int | None = None
        batch_max = 0

        def _legal_chunk(request: Request, chunk: int) -> int:
            """预测原生调度对 chunk 的全部合法改写,作为步后核验基准:
            云公告 mamba 网格重切(边侧 chunk 权威的另一半,网格未学到
            时恒等)。不预测即与核验冲突:picker 按 min(剩余,C)/cap
            截断算的 expected,原生还会再按网格收边(实测 1514→768)。"""
            return self._mamba_block_aligned_split(request, chunk)

        # Apply the limit before scheduling/publishing so RangeNotify and the
        # embeddings payload retain identical request sets and token counts.
        is_deepseek_v4 = (
            self.vllm_config.model_config.hf_text_config.model_type == "deepseek_v4"
        )
        single_prefill = False
        # 1) 续传优先(先收尾再开新)
        for request in self.running:
            if single_prefill and picked:
                break
            remaining = (
                request.num_prompt_tokens - request.num_computed_tokens
            )
            if remaining <= 0:
                continue
            # Use the full prompt length, not this chunk's remaining length.
            # If a long request follows short ones, close the batch here and
            # leave it for the next batch; do not bypass it with waiting work.
            single_prefill = is_deepseek_v4 and request.num_prompt_tokens >= 8000
            if single_prefill and picked:
                break
            chunk = _legal_chunk(request, min(remaining, unit))
            # 同质性门控:违例即收批(本请求留下一条 notify)。
            # 末位截断分支不受此限:截断块恒为批尾,由预算耗尽复现
            if chunk < remaining:
                # 非末位块:须与批内已有非末位块同尺寸,且不小于任何
                # 已入批末位块(否则 T 被抬高,本块被阈值放过而超排)
                if (
                    nonlast_size is not None and chunk != nonlast_size
                ) or chunk < batch_max:
                    break
                nonlast_size = chunk
            elif nonlast_size is not None and chunk > nonlast_size:
                # 末位块(收尾):可调度量 == 剩余,天然精确;但不得把
                # T 抬过已有非末位块尺寸
                break
            batch_max = max(batch_max, chunk)
            if used + chunk > cap:
                # 末位截断:批内最后一个条目允许被剩余额度截尾,
                # 批收尾(预算耗尽,与集中式 chunked prefill 逐语义
                # 对齐)。复现依据:截断块恒为末位,边侧预算钳制
                # (§2.5)与云侧预算钳制(§2.3)都靠"预算耗尽"复现它;
                # 非末位块由 R/C/T 定界,与截断块尺寸无关
                c = _legal_chunk(request, cap - used)
                if c >= 1:
                    picked.append(request.request_id)
                    expected[request.request_id] = c
                    used += c  # 截断块同样占预算,漏记会被新开循环超批
                break
            picked.append(request.request_id)
            expected[request.request_id] = chunk
            used += chunk
        # 2) 开新:FCFS 前缀,自限 chunk;装不下时同上末位截断。
        # 无截断逃生时,任一在途请求(含 decode 中的 awaiting)把 cap
        # 压到 C 以下会让大 prompt 永不可开新、流水线全程串行(真机
        # 已现:decode 阶段云侧只剩 1-2 个请求在跑)
        new_count = 0
        candidates = list(self.skipped_waiting) + list(self.waiting)
        for request in candidates:
            if single_prefill and picked:
                break
            single_prefill = is_deepseek_v4 and request.num_prompt_tokens >= 8000
            if single_prefill and picked:
                break
            # 前缀命中 trim 的请求 num_computed > 0,可调度量按剩余算
            remaining = (
                request.num_prompt_tokens - request.num_computed_tokens
            )
            if remaining <= 0:
                continue
            chunk = _legal_chunk(request, min(remaining, unit))
            # 同质性门控:同续传循环(违例收批,保 FCFS 前缀语义)
            if chunk < remaining:
                if (
                    nonlast_size is not None and chunk != nonlast_size
                ) or chunk < batch_max:
                    break
                nonlast_size = chunk
            elif nonlast_size is not None and chunk > nonlast_size:
                break
            batch_max = max(batch_max, chunk)
            if new_count >= free_seq:
                break
            if used + chunk > cap:
                c = _legal_chunk(request, cap - used)
                if c >= 1:
                    picked.append(request.request_id)
                    expected[request.request_id] = c
                    used += c
                break
            picked.append(request.request_id)
            expected[request.request_id] = chunk
            used += chunk
            new_count += 1
        # 组批自洽:截断分支同样记账,任何路径下 Σexpected == used ≤ cap
        # (真机实录:截断分支漏记 used,新开循环按陈旧余量超批 2×cap)
        total = sum(expected.values())
        assert total == used and total <= cap, (
            f"[Lwd][edge-sched] picker budget overflow: "
            f"Σexpected={total} used={used} cap={cap}"
        )
        return picked, expected, cap

    def _lwd_schedule_batch(self) -> SchedulerOutput:
        """chunk 组批:picker 选批,预算钳制后经基类可见集机制调度 +
        逐 chunk 核验。

        选择规则见 _lwd_pick_prefill_batch;队列剔除/隔离/拼回复用基类
        _lwd_schedule_for_visible_reqs(waiting/skipped 来源走原生准入
        窗口,running 来源走续跑)。

        预算钳制(设计 §2.5):可见集调用期间把 max_num_scheduled_tokens
        临时压到 cap(原生 token 预算即读该字段,
        scheduler.py:360),原生 budget/threshold 双重截断逐请求复现
        picker 的 chunk 划分。语义注记:与旧手写容器交换不同,本步被
        抢占的请求回 waiting 尾部、被跳过的回 skipped 队首,均取基类
        统一语义,不再做队首回插。"""
        req_ids, expected, cap = self._lwd_pick_prefill_batch()
        if req_ids:
            logger.info(
                "[Lwd][edge-sched] pick batch reqs=%s tokens=%d cap=%d",
                req_ids, sum(expected.values()), cap,
            )
        if req_ids and cap < self.max_num_scheduled_tokens:
            saved_budget = self.max_num_scheduled_tokens
            self.max_num_scheduled_tokens = cap
            try:
                out = self._lwd_schedule_for_visible_reqs(req_ids)
            finally:
                self.max_num_scheduled_tokens = saved_budget
        else:
            out = self._lwd_schedule_for_visible_reqs(req_ids)
        # chunk 核验:picker 的 chunk 划分必须被原生调度逐请求复现
        # (不等即 cap/threshold 失守或原生行为发散;云侧注入窗口
        # 看门狗也会把它变成 RuntimeError——这里先炸,错在边侧调度,
        # 不等到数据面)
        for req_id, want in expected.items():
            got = out.num_scheduled_tokens.get(req_id, 0)
            if got != want:
                raise RuntimeError(
                    f"[Lwd][edge-sched] chunk schedule mismatch: "
                    f"req={req_id} scheduled={got} != expected={want} "
                    f"(cap/threshold 失守或原生行为发散)"
                )
        return out

    def lwd_edge_add_request(self, request: Request) -> None:
        """请求入口:边界校验 -> probe 投递 -> (异步)绑定 -> 云预告 -> 入队。

        协调启用时 probe 走异步两阶段(E1 根治):本方法只做校验与投递
        (HMAC 建链 + HTTP 都在专用 probe 线程),请求入 inflight 即刻
        返回,主循环零阻塞;结果回灌(lwd_edge_drain_probe_results)
        时才完成绑定/预告/入队。协调未启用时退化为同步直通(零开销)。

        预告先于本地登记:云侧视图领先本地工作,云只能准备不能提前
        计算(没有 chunk 预告就不会开算)。abort_immediately 请求走
        finish + abort 出口,与原生语义一致。
        """
        self._lwd_validate_request(request)
        # 收尾记账备料:prompt ids 留档(终结时 Request 已不在调度器)
        self._lwd_finish_prompt_ids[request.request_id] = list(
            request.prompt_token_ids
        )
        # 多模态内容身份同一入口留档(主循环线程提取,probe 协程只读
        # 快照不触碰 Request):probe/finish/生成段三处建链共用这一份
        # 快照,保证查链与存链口径一致。纯文本请求留档为空,不入字典。
        media_items = self._lwd_media_items(request)
        if media_items:
            self._lwd_finish_media_items[request.request_id] = media_items
        if request.abort_immediately:
            # 立即终结请求不付 probe RTT,也不给云侧制造必被 TTL 清扫
            # 的孤儿预留:直接本地终结(与 finalize 后再 abort 同终态)
            self._lwd_abort_immediate(request)
            return
        if self._lwd_probe_submit_async(request):
            return
        # 协调未启用:不续算,链取 request 自身原生块哈希(与旧行为一致)
        self._lwd_finalize_add_request(request, list(request.block_hashes), 0)

    def _lwd_abort_immediate(self, request: Request) -> None:
        """本地即时终结(abort_immediately / probe 在飞被 abort 共用):
        入队后按 ABORTED 终结,finished_req_ids 随下一步输出——恢复
        「abort 必有终态信号」契约。不上云:从未预告,云侧零残留
        (probe 已落的孤儿预留由云侧 TTL 清扫兜底)。"""
        # 收尾备料一并清理(该请求不再上报 finish)
        self._lwd_finish_prompt_ids.pop(request.request_id, None)
        self._lwd_finish_gen_ids.pop(request.request_id, None)
        self._lwd_finish_media_items.pop(request.request_id, None)
        super().add_request(request)
        self.finish_requests(
            [request.request_id], RequestStatus.FINISHED_ABORTED
        )

    def _lwd_finalize_add_request(
        self, request: Request, block_hashes: list[bytes], prefix_hit_tokens: int
    ) -> None:
        """add 第二阶段(主循环唯一执行点):续算推进 -> 绑定 -> 预告 -> 入队。

        同步直通与 probe 回灌共用。命中时推进 num_computed_tokens 使边侧
        只 embed/发尾巴:命中边界以云侧块池单真源为准,chunk offset ==
        云侧 computed,续算闭环;命中值随 notify 下发,云侧按它封顶
        resume——probe 超时/并发下池增长时,云侧不会越过边侧实际裁剪的
        边界。"""
        if prefix_hit_tokens > 0:
            request.num_computed_tokens = prefix_hit_tokens
        cloud_id = self._lwd_default_cloud_id()
        self._lwd_binding.bind(request.request_id, cloud_id)
        self._lwd_compute_mrope_positions(request)
        self.lwd_edge_notify_request(
            request_id=request.request_id,
            num_prompt_tokens=len(request.prompt_token_ids),
            sampling_params=request.sampling_params,
            block_hashes=block_hashes,
            prefix_hit_tokens=prefix_hit_tokens,
            cloud_id=cloud_id,
        )
        super().add_request(request)

    # ------------------------------------------------------------------ #
    # 前缀 probe(异步两阶段)                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _lwd_media_items(request: Request) -> list[_LwdMediaIdentity]:
        """提取多模态内容身份(mm_features -> 协议层 media item)。

        主循环线程调用(probe 协程只拿快照列表,不触碰 Request)。仅取
        协议 ABI 已支持的 modality(当前 ``image``;audio/video 协议
        预留未开,掺入会被 _validate_media_items 拒绝使 probe fail-open
        丢命中——不支持模态暂保持盲链,待协议扩展)。无匹配项返回空表,
        纯文本路径建链与旧版逐位一致。"""
        return [
            _LwdMediaIdentity(
                modality=feature.modality,
                digest=hashlib.sha256(
                    feature.identifier.encode("utf-8")
                ).digest(),
                offset=feature.mm_position.offset,
                length=feature.mm_position.length,
            )
            for feature in request.mm_features
            if feature.modality == "image"
        ]

    def _lwd_processor_fingerprint(self) -> bytes:
        """处理器指纹(32B):模型 + mm_processor_kwargs 的 sha256。

        掺入媒体覆盖块 HMAC,保证换模型/换 processor 配置即链分叉。
        云侧不复算指纹(链值即块池键,边侧单边稳定即可),无需与云
        对齐。"""
        if self._lwd_processor_fp is None:
            model_cfg = self.vllm_config.model_config
            material = "|".join((
                str(model_cfg.model),
                # InitVar 可能不在实例属性上,缺省 None 参与摘要
                repr(getattr(model_cfg, "mm_processor_kwargs", None)),
            ))
            self._lwd_processor_fp = hashlib.sha256(
                material.encode("utf-8")
            ).digest()
        return self._lwd_processor_fp

    def _lwd_probe_coord_cfg(self):
        """协调配置可用则返回 (coord_cfg, vllm_config),否则 None(直通)。"""
        vllm_config = getattr(self, "vllm_config", None)
        if vllm_config is None:
            return None
        coord_cfg = getattr(vllm_config, "lwd_coordination", None)
        if coord_cfg is None or not getattr(coord_cfg, "enabled", False):
            return None
        if not coord_cfg.control_url or not coord_cfg.tenant_key_file:
            return None
        return coord_cfg, vllm_config

    def _lwd_probe_client(self, coord_cfg, build_bs: int):
        """probe 客户端缓存(按当前建链粒度):hasher/密钥只建一次。"""
        client = self._lwd_probe_client_cache
        if client is None or client._hasher.block_size != build_bs:
            with open(coord_cfg.tenant_key_file, "rb") as f:
                tenant_key = f.read()
            from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_client import (
                LwdEdgePrefixClient,
            )

            client = LwdEdgePrefixClient(
                coord_cfg.control_url,
                build_bs,
                tenant_key,
                edge_id=self._lwd_edge_id,
                consumer_id=coord_cfg.consumer_id,
                # 总超时钳到 3s:死锁残余时快速 fail-open
                connect_timeout=min(coord_cfg.connect_timeout, 3.0),
                # 指纹与 _lwd_prefix_hasher 同源:probe 查链侧也要建
                # media 覆盖块,协议要求带 media_items 的建链必持指纹,
                # 缺此参数 mm 请求 probe 建链必抛(fail-open 丢命中)
                processor_fingerprint=self._lwd_processor_fingerprint(),
            )
            self._lwd_probe_client_cache = client
        return client

    def _lwd_probe_submit_async(self, request: Request) -> bool:
        """投递 probe 到专用线程(成功投递返回 True,请求进 inflight)。

        协程内完成 HMAC 建链 + HTTP 探测(都不占主循环);完成回调把
        (result, chain) 投线程安全队列并 WAKEUP 引擎。fail-open:
        协程异常由回调归一为 None 结果,回灌时按未命中直通。"""
        cfg_pair = self._lwd_probe_coord_cfg()
        if cfg_pair is None:
            return False
        coord_cfg, vllm_config = cfg_pair
        if self._lwd_probe_runtime is None:
            self._lwd_probe_runtime = _LwdProbeRuntime()
        runtime = self._lwd_probe_runtime
        cloud_id = self._lwd_default_cloud_id()
        # 探测链粒度:优先用已学习的云 hash 粒度(稳态下 probe 的
        # manifest 即云域,ratio=1 直通);未学习回退边块粒度。
        known_bs = self._lwd_hash_bs_by_cloud.get(cloud_id)
        build_bs = known_bs or vllm_config.cache_config.block_size
        client = self._lwd_probe_client(coord_cfg, build_bs)
        request_id = request.request_id
        prompt_ids = list(request.prompt_token_ids)
        # 媒体身份取入口留档快照(主循环线程):协程只读列表,不触碰
        # Request;纯文本/无图请求留档为空。
        media_items = self._lwd_finish_media_items.get(request_id) or ()

        async def _job():
            session = await runtime.ensure_session()
            manifest = client.build_manifest(
                request_id, prompt_ids, media_items=media_items
            )
            result = await client.negotiate(
                session, request_id, prompt_ids, manifest=manifest,
                media_items=media_items,
            )
            return result, list(manifest.full_block_hashes)

        try:
            future = runtime.submit(_job())
        except Exception:  # noqa: BLE001  线程/循环死亡:fail-open 同步直通
            logger.warning(
                "[Lwd][edge] probe runtime submit failed (fail-open)",
                exc_info=True,
            )
            runtime.stop()
            self._lwd_probe_runtime = None
            return False
        self._lwd_probe_nonce += 1
        nonce = self._lwd_probe_nonce
        self._lwd_probe_inflight[request_id] = (request, nonce, time.monotonic())

        def _watchdog() -> None:
            """兜底:probe 线程/事件循环静默死亡时 future 永不完成,
            超时按 fail-open 直通(与协程异常同价),防 inflight 永久
            滞留占水位、客户端干等。正常完成时由 _done 取消,不会
            触发;与 _done 竞争幂等(drain 对缺席 inflight 丢弃)。"""
            if future.done():
                return
            future.cancel()
            logger.warning(
                "[Lwd][edge] prefix probe watchdog timeout (fail-open): "
                "req=%s", request_id,
            )
            self._lwd_probe_done.put((request_id, nonce, build_bs, None))
            wakeup = self.lwd_probe_wakeup
            if wakeup is not None:
                wakeup()

        timer = threading.Timer(_LWD_PROBE_WATCHDOG_S, _watchdog)
        timer.daemon = True  # 不拖累进程退出(残留计时器最多 15s 自灭)

        def _done(fut) -> None:
            timer.cancel()  # 正常完成即取消看门狗:否则 Timer(线程子类)
            # 会活满 15s,稳态并发 = 请求速率 x 15s,高 QPS 下线程堆积
            try:
                outcome = fut.result()
            except Exception:  # noqa: BLE001  fail-open:回灌按未命中直通
                logger.warning(
                    "[Lwd][edge] prefix probe failed (fail-open)", exc_info=True
                )
                outcome = None
            self._lwd_probe_done.put((request_id, nonce, build_bs, outcome))
            wakeup = self.lwd_probe_wakeup
            if wakeup is not None:
                wakeup()

        future.add_done_callback(_done)
        timer.start()
        return True

    def lwd_edge_drain_probe_results(self) -> int:
        """回灌 probe 结果(主循环,引擎步首/WAKEUP 后调用)。

        对每条完成结果:nonce 比对(同 id 重发的旧结果直接丢弃,不动
        新 inflight)、学习云侧粒度(不一致按学值重建链,仅首请求)、
        记账建链粒度、fail-open 归一,然后走 finalize 完成入队。
        probe 在飞期间被 abort 的请求(墓碑 nonce 命中)直接丢弃。
        返回完成入队的请求数。"""
        completed = 0
        while True:
            try:
                request_id, nonce, build_bs, outcome = (
                    self._lwd_probe_done.get_nowait()
                )
            except queue.Empty:
                break
            entry = self._lwd_probe_inflight.get(request_id)
            if entry is None or entry[1] != nonce:
                # 已 abort/已终结(条目摘除)或同 id 重发的旧提交结果:
                # 丢弃,不影响在飞的新提交
                continue
            request, _, _ = entry
            self._lwd_probe_inflight.pop(request_id, None)
            if outcome is None:
                # fail-open:探测失败不阻断请求,不续算
                self._lwd_finalize_add_request(
                    request, list(request.block_hashes), 0
                )
                completed += 1
                continue
            result, chain = outcome
            # 学习云侧 hash 粒度;与本次建链粒度不一致时按其重建链。
            # notify 在 probe 之后才发,故首请求(首次学习)即生效:非
            # 整除 prompt 的链覆盖到云 hash 粒度,采纳条件恒成立。
            learned = int(result.block_size)
            cloud_id = self._lwd_default_cloud_id()
            self._lwd_hash_bs_by_cloud[cloud_id] = learned
            # 学习云侧 mamba 对齐网格并开启边侧自动对齐重切:此后原生
            # 调度的两个重切点(scheduler.py:436/726)用网格自动收 chunk,
            # 快照网格与 chunk 网格重合,max_num_batched_tokens 无需
            # 再手工配成对齐倍数(根治"静默不命中")。
            grid = int(getattr(result, "alignment_tokens", 0) or 0)
            if grid > 0:
                self._lwd_align_grid_by_cloud[cloud_id] = grid
                if not self.need_mamba_block_aligned_split:
                    self.need_mamba_block_aligned_split = True
                    logger.info(
                        "[Lwd][edge] mamba align grid learned: cloud=%d "
                        "grid=%d (auto chunk alignment enabled)",
                        cloud_id, grid,
                    )
                if grid > self.max_num_scheduled_tokens:
                    logger.warning(
                        "[Lwd][edge] align grid %d > max_num_batched_tokens "
                        "%d: chunks smaller than one grid cannot be "
                        "aligned (request would stall); raise "
                        "max_num_batched_tokens",
                        grid, self.max_num_scheduled_tokens,
                    )
            if learned != build_bs:
                cfg_pair = self._lwd_probe_coord_cfg()
                if cfg_pair is not None:
                    coord_cfg, vllm_config = cfg_pair
                    chain = list(
                        self._lwd_prefix_hasher(
                            coord_cfg, vllm_config, learned
                        ).build_manifest(
                            request_id, list(request.prompt_token_ids),
                            # 与 probe 建链同一份媒体快照
                            media_items=(
                                self._lwd_finish_media_items.get(request_id)
                                or ()
                            ),
                        ).full_block_hashes
                    )
            # 记账本请求 probe 的建链粒度(注意是 build_bs 而非学习值):
            # 云侧存储的 manifest 即 probe 粒度,finish 须按同一粒度建链
            # 对账;学习换挡只发生在重启后首个请求,故按请求记账。
            self._lwd_probe_bs_by_req[request_id] = build_bs
            logger.info(
                "[Lwd][edge] prefix probe req=%s hit_tokens=%d blocks=%d "
                "hash_bs=%d",
                request_id, result.hit_tokens, len(chain), learned,
            )
            # 多模态命中与文本同权:mm 链已掺图像内容身份,不同图在图像
            # 首个覆盖块即分叉必不命中;同图任意边界(含半图)复用的 KV
            # 与锚点请求逐位一致,无需位置截断。
            self._lwd_finalize_add_request(request, chain, result.hit_tokens)
            completed += 1
        return completed

    def lwd_probe_results_pending(self) -> bool:
        """probe 完成队列非空(引擎 has_work 计入:WAKEUP 后须进步回灌)。"""
        return not self._lwd_probe_done.empty()

    def lwd_edge_max_num_seqs_check(self) -> bool:
        """max_num_seqs 适配检查:云侧在途水位(running + awaiting)是否
        还有名额,True=可开新请求。

        awaiting 请求已清出调度器,原生准入只数 running(边侧恒在飞批
        深度以内)永远拦不住;以 running+awaiting 对账云侧在途数,达到
        max_num_running_reqs(边云部署对齐,即云侧名额)即满员。续传
        豁免不在本判断(schedule 闸门经 _lwd_has_prefill_chunk_inflight
        放行收尾);请求到达时的 announce 亦不受约束(云只登记不计算)。

        probe 在飞请求计入水位:它们即将成为 awaiting,突发并发下
        不计入会瞬时超订云侧在途配额。

        注:云侧复用下一台云收多边的请求,边侧按云聚合水位并不准确,
        维持本判据(本地在途对账),按云配额留待云侧反馈机制接入。"""
        return (
            len(self.running)
            + len(self._lwd_awaiting)
            + len(self._lwd_probe_inflight)
            < self.max_num_running_reqs
        )

    def _lwd_has_prefill_chunk_inflight(self) -> bool:
        """running 中是否存在未发完的 embed 请求(续传收尾中)。

        与 _lwd_pick_prefill_batch 的续传分支同判据,两处需保持一致:
        闸门放行收尾的前提是 picker 必然挑中该续传请求而非开新。"""
        return any(
            req.num_computed_tokens < req.num_prompt_tokens
            for req in self.running
        )

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        """用云侧公告的对齐网格重放原生 mamba 对齐重切(根治"1536 手工对齐")。

        背景:边侧无 mamba 层(has_mamba_layers=False),原生开关恒关;
        云侧 follow 模式又不能自行复切边侧 chunk——对齐责任只能落在
        边侧。网格(scheduler_block_size = 云各 KV 组块大小 LCM)经
        probe 响应头逐云学习;学到后 need_mamba_block_aligned_split
        置真,原生调度的两个重切点自动走这里。

        语义与原生 ``Scheduler._mamba_block_aligned_split`` 逐行一致
        (中间块收到网格整数倍、末个对齐块强制收在 last_cache_position、
        尾巴直通),仅块大小来源换成云侧公告网格。网格未学到时本方法
        不被调用(开关恒关),行为与学习前完全一致。

        多模态请求照常参与对齐重切:mm 链已掺图像内容身份(不同图链
        分叉,不会跨图误命中),同图任意边界(含半图)复用放行——与
        文本命中同权;对齐保证的 mamba 快照正是命中前缀可复用的前提,
        全量新算的图像段不受 chunk 边界影响。"""
        cloud_id = self._lwd_binding.cloud_of(
            request.request_id, default=self._lwd_default_cloud_id()
        )
        block_size = self._lwd_align_grid_by_cloud.get(cloud_id, 0)
        if block_size <= 0:
            return num_new_tokens
        num_computed_tokens = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        if num_computed_tokens < max(
            request.num_prompt_tokens, request.num_tokens - 1
        ):
            last_cache_position = (
                request.num_tokens - request.num_tokens % block_size
            )
            if self.use_eagle:
                last_cache_position = max(last_cache_position - block_size, 0)
            num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens
            if num_computed_tokens_after_sched < last_cache_position:
                # align to grid
                num_new_tokens = num_new_tokens // block_size * block_size
            elif (
                num_computed_tokens
                < last_cache_position
                < num_computed_tokens_after_sched
            ):
                # force to cache the last aligned chunk
                num_new_tokens = last_cache_position - num_computed_tokens
            else:
                # prefill the last few tokens
                pass
        return num_new_tokens

    @staticmethod
    def _lwd_request_has_mm(request: Request) -> bool:
        """请求是否带真实多模态数据(prompt_embeds passthrough 不算——
        它没有 grid 元数据,mrope 按文本位置处理,与原生口径一致)。"""
        return any(
            feature.modality != "prompt_embeds"
            for feature in request.mm_features
        )

    def _lwd_compute_mrope_positions(self, request: Request) -> None:
        """准入时为多模态请求一次性算全 prompt 的 [3,N] mrope positions。

        设计分工(对照 latest_lwd 边云):token_id 不上线路,云侧拿
        不到 grid 元数据,无法从第一性原理构造 3D 位置——由边侧(唯一
        持有真实 prompt + mm_features 的一侧)算好,逐 chunk 切片随
        EMBED 批经数据面发给云;delta 云侧收齐末 chunk 后自推
        (max+1-N),不上 wire。

        模型类静态方法直接以 hf_config 调用(Qwen3VL 族,含 Qwen3.5),
        无需模型实例/权重;模型族不提供该静态入口即拒绝(报错信息给出
        白名单口径),不静默按文本位置错算。"""
        if not self.vllm_config.model_config.uses_mrope:
            return
        if not self._lwd_request_has_mm(request):
            return
        if self._lwd_mrope_positions_fn is None:
            from vllm.model_executor.models import ModelRegistry

            model_cls, _ = ModelRegistry.resolve_model_cls(
                self.vllm_config.model_config.architectures,
                self.vllm_config.model_config,
            )
            fn = getattr(model_cls, "_get_mrope_input_positions", None)
            if fn is None:
                raise ValueError(
                    f"[LWD] multimodal requests in prefill-only mode "
                    f"require a model class exposing "
                    f"_get_mrope_input_positions (Qwen3VL family, incl. "
                    f"Qwen3.5); got {model_cls.__name__} "
                    f"(request {request.request_id})"
                )
            self._lwd_mrope_positions_fn = fn
        mrope_features = [
            f for f in request.mm_features if f.modality != "prompt_embeds"
        ]
        positions, _delta = self._lwd_mrope_positions_fn(
            input_tokens=list(request.prompt_token_ids),
            mm_features=mrope_features,
            config=self.vllm_config.model_config.hf_config,
        )
        assert positions.shape[1] == request.num_prompt_tokens, (
            f"[LWD] mrope positions width {positions.shape[1]} != prompt "
            f"len {request.num_prompt_tokens} (request {request.request_id})"
        )
        self._lwd_mrope_positions_dict[request.request_id] = positions
        logger.info(
            "[Lwd][edge-sched] mrope positions computed: req=%s prompt=%d",
            request.request_id, request.num_prompt_tokens,
        )
    def lwd_edge_notify(self, scheduler_output: SchedulerOutput) -> bool:
        """对新调度的 prefill 批发一条批量 LwdRangeNotify(seqno 先行)。

        seqno 无空洞契约:UP 数据通道按连续号序配对(通道层对超前号
        扣留等待,一个空洞即永久挂死整条链),因此号只能分配给真正
        上 wire 的批:
        - peek-then-advance:发布成功才进位该批所属云的计数器;
        - 单 notify 前提:本方法每步至多发一条(批量条目全部装在同
          一条里),发布原子性即整批原子性,不存在部分成功。
        云侧复用 seqno 按 (cloud_id) 对分域:每对通道内无空洞契约
        独立成立,互不影响。mixed 批按云分组发布(一批一云一号),
        跨云混批 fail-fast(当前部署单边绑单云,见函数内守卫)。

        前提:控制面发布通道不丢消息,publish 恒成功——步末回退
        对账已按此前提移除。若前提被破坏返回 False,调用方本步不
        派发但进度不回退,该批永久丢失;重复预告在云侧按
        (request_id, offset) 幂等登记。"""
        scheduled = scheduler_output.num_scheduled_tokens
        items: list[LwdRangeItem] = []
        req_ids: list[str] = []
        token_slices: list[list[int]] = []
        prompt_offsets: list[int] = []
        has_mrope: list[bool] = []
        mrope_slices: list[list[list[int]]] = []
        cloud_id: int | None = None
        for request_id, num_tokens in scheduled.items():
            request = self.requests.get(request_id)
            if request is None:
                continue
            req_cloud = self._lwd_binding.cloud_of(
                request_id, default=self._lwd_default_cloud_id()
            )
            # 滞留守卫(E4):该云滞留未清空时不得再发 RangeNotify——
            # 新预告会复用未进位的 seqno 造成重号,或越过滞留的
            # RequestNotify 乱序。先尝试 flush,仍未清空则本步弃批
            # 下步再来(与旧发布失败同语义,不阻塞)。
            if self._lwd_pending_publishes.get(req_cloud):
                self.lwd_edge_flush_pending()
                if self._lwd_pending_publishes.get(req_cloud):
                    # 弃批即该 chunk 永久丢失(进度已乐观推进,请求将
                    # 永远 awaiting、云侧空转——实测 2E1C 症状),
                    #  fail-fast 暴露背压失衡,不再静默丢批
                    raise RuntimeError(
                        f"[Lwd][edge-notify] pending publishes to cloud "
                        f"{req_cloud} not drained; refusing to dispatch "
                        f"batch (chunks would be silently dropped): "
                        f"req={request_id} offset={request.num_computed_tokens - num_tokens} "
                        f"num={num_tokens} pending={len(self._lwd_pending_publishes[req_cloud])}"
                    )
            if cloud_id is None:
                cloud_id = req_cloud
            elif req_cloud != cloud_id:
                # mixed 批一云一条 RangeNotify(单 seqno 配对该云 UP
                # 通道);当前部署单边绑单云,跨云混批属配置错误,
                # fail-fast 暴露而非静默丢批(进度已乐观推进)。
                raise RuntimeError(
                    f"[Lwd][edge-notify] mixed batch spans clouds: "
                    f"{request_id} -> cloud {req_cloud}, batch already "
                    f"bound to cloud {cloud_id} (edge multi-cloud binding "
                    "is not supported)"
                )
            # _update_after_schedule 已乐观推进 num_computed,起点需回退本步量
            offset = request.num_computed_tokens - num_tokens
            # 多模态:准入时预算了整 prompt positions 的请求,各 chunk
            # 恒携带 mrope 行(key 在即恒真);[3,N] 切本 chunk 列 ->
            # [n,3] 行序与 token 轴对齐,随批下发 worker 组 aux 帧
            positions = self._lwd_mrope_positions_dict.get(request_id)
            req_has_mrope = positions is not None
            items.append(
                LwdRangeItem(
                    request_id=request_id,
                    offset=offset,
                    num_tokens=num_tokens,
                    has_mrope=req_has_mrope,
                )
            )
            req_ids.append(request_id)
            token_slices.append(
                list(request.prompt_token_ids[offset : offset + num_tokens])
            )
            prompt_offsets.append(offset)
            has_mrope.append(req_has_mrope)
            if req_has_mrope:
                assert positions is not None
                mrope_slices.append(
                    positions[:, offset : offset + num_tokens].t().tolist()
                )
            else:
                mrope_slices.append([])
        if not items:
            return True
        assert cloud_id is not None
        seqno = self._lwd_seqno.get(cloud_id, 0)
        first = items[0]
        # RangeNotify 直发不入滞留队列:入队后补投会让云侧登记一个
        # 对应批已被丢弃的 seqno,UP 数据面等永不来的张量而挂死。
        # 发布失败 = 批被弃但进度已推进 = 请求永久 awaiting 云侧空转,
        # fail-fast 暴露背压失衡(实测 2E1C 静默症状的根源嫌疑)。
        # 顶层单请求字段 = 首条目:旧版云侧(不读 items)仍可消费单请求批
        if not self._lwd_publish(
            LwdRangeNotify(
                request_id=first.request_id,
                offset=first.offset,
                num_tokens=first.num_tokens,
                seqno=seqno,
                edge_id=self._lwd_edge_id,
                has_mrope=any(has_mrope),
                items=items,
            ),
            cloud_id,
        ):
            raise RuntimeError(
                f"[Lwd][edge-notify] RangeNotify publish failed "
                f"(channel backpressure): seqno={seqno} cloud={cloud_id} "
                f"reqs={[i.request_id for i in items]} "
                f"tokens={sum(i.num_tokens for i in items)}"
            )
        self._lwd_seqno[cloud_id] = seqno + 1
        logger.info(
            "[Lwd][edge-notify] batch reqs=%d tokens=%d seqno=%d cloud=%d "
            "mrope=%d",
            len(items), sum(item.num_tokens for item in items), seqno,
            cloud_id, sum(has_mrope),
        )
        # 发布成功即组 EMBED 批挂 SO:seqno 是数据面发云张量的
        # 配对键(与云侧 RangeNotify 登记同值),embed 载荷为本
        # 批逐请求的 token 片段(边 worker 按 req_ids 序扁平化);
        # mrope 行同序携带(仅 MM 请求非空),worker 据此组 aux 帧
        scheduler_output.lwd_batch = LwdBatch(
            batch_type=LwdBatchType.LWD_EMBED,
            seqno=seqno,
            batch_meta=LwdEmbedBatch(
                req_ids=req_ids,
                token_ids=token_slices,
                edge_id=self._lwd_edge_id,
                cloud_id=cloud_id,
                prompt_offsets=prompt_offsets,
                has_mrope=has_mrope,
                mrope_positions=mrope_slices,
            ),
        )
        return True

    def lwd_edge_notify_request(
        self,
        request_id: str,
        num_prompt_tokens: int,
        sampling_params: SamplingParams | None = None,
        block_hashes: list[bytes] | None = None,
        prefix_hit_tokens: int = 0,
        cloud_id: int | None = None,
    ) -> None:
        """发 LwdRequestNotify(请求元数据预告)。

        block_hashes = prompt 全量满块哈希链(自位置 0 起)。云侧
        prompt token 是占位零值,本地算不出真实内容哈希,前缀缓存
        命中只能靠这条链;缺省空链 = 不提供,云侧回退占位链。

        prefix_hit_tokens = 边侧 probe 命中并 trim 后的续算边界:云侧
        resume 封顶于此(0 = 不续算,与恒发整段一致)。

        sampling_params 只透传影响云侧 token 选择的字段(采样核/惩罚/
        EOS 策略/min_tokens);stop 字符串等 detokenizer 层参数留在
        边侧前端原生处理,不上 wire。

        cloud_id:云侧复用按它定向 publish(dest=cloud_id);None 回退
        默认绑定云(1E1C 即单云,行为不变)。

        失败语义(E4):背压时消息进该云有序滞留队列,由引擎步首
        flush 补投(保序在本请求 RangeNotify 之前),主循环零阻塞;
        滞留超 _LWD_PENDING_FAIL_AFTER_S 由 flush fail-fast(云通道
        死亡),与旧版"重试耗尽抛错"同级但不再用 sleep 卡主循环。"""
        if self.lwd_edge_publisher is None and self.lwd_edge_channel is None:
            return
        if cloud_id is None:
            cloud_id = self._lwd_default_cloud_id()
        sp = sampling_params
        message = LwdRequestNotify(
            request_id=request_id,
            num_prompt_tokens=num_prompt_tokens,
            max_tokens=(
                sp.max_tokens if sp is not None and sp.max_tokens is not None else 16
            ),
            cloud_id=cloud_id,
            edge_id=self._lwd_edge_id,
            block_hashes=block_hashes if block_hashes is not None else [],
            prefix_hit_tokens=prefix_hit_tokens,
            temperature=sp.temperature if sp is not None else 1.0,
            top_p=sp.top_p if sp is not None else 1.0,
            top_k=sp.top_k if sp is not None else 0,
            min_p=sp.min_p if sp is not None else 0.0,
            seed=sp.seed if sp is not None else None,
            repetition_penalty=sp.repetition_penalty if sp is not None else 1.0,
            presence_penalty=sp.presence_penalty if sp is not None else 0.0,
            frequency_penalty=sp.frequency_penalty if sp is not None else 0.0,
            ignore_eos=sp.ignore_eos if sp is not None else False,
            stop_token_ids=(
                list(sp.stop_token_ids) if sp is not None and sp.stop_token_ids else []
            ),
            min_tokens=sp.min_tokens if sp is not None else 0,
            eos_token_id=sp.eos_token_id if sp is not None else None,
        )
        if self._lwd_publish_ordered(message, cloud_id):
            logger.info(
                "[Lwd][edge-notify] request meta announced: req=%s "
                "prompt=%d cloud=%d",
                request_id, num_prompt_tokens, cloud_id,
            )

    def lwd_edge_abort(self, request_ids: list[str]) -> None:
        """发 LwdAbortNotify + 摘除 awaiting/绑定;调度器内清理走原生路径。

        awaiting 请求已不在调度器视野(嵌入完结时清出),原生
        finish_requests 触不到它,须在此显式摘除,防迟到云结果被误认领。
        云侧复用按绑定表的 cloud_id 定向 publish;未绑定云的请求
        (发布前 abort)仅本地清台账。"""
        for request_id in request_ids:
            self._lwd_awaiting.pop(request_id, None)
            # 收尾备料随 abort 一并清理(不再上报 finish)
            self._lwd_finish_prompt_ids.pop(request_id, None)
            self._lwd_finish_gen_ids.pop(request_id, None)
            self._lwd_finish_media_items.pop(request_id, None)
            self._lwd_gen_chain_sent.pop(request_id, None)
            self._lwd_gen_chain_state.pop(request_id, None)
            self._lwd_probe_bs_by_req.pop(request_id, None)
            # mrope 台账随 abort 清理(多模态准入期预计算的位置缓存)
            self._lwd_mrope_positions_dict.pop(request_id, None)
            # probe 在飞期间的 abort:摘除 inflight(迟到的 probe 结果在
            # drain 按缺席丢弃,水位名额即刻归还),随后本地即时终结给前端
            # 终态信号;不上云(从未预告,云侧零残留;云侧预留由 TTL 兜底)
            inflight = self._lwd_probe_inflight.pop(request_id, None)
            if inflight is not None:
                self._lwd_probe_bs_by_req.pop(request_id, None)
                logger.info(
                    "[Lwd][edge-notify] abort req=%s during probe inflight: "
                    "local abort with terminal signal", request_id,
                )
                self._lwd_abort_immediate(inflight[0])
            if self.lwd_edge_publisher is None and self.lwd_edge_channel is None:
                continue
            cloud_id = self._lwd_binding.unbind(request_id)
            if cloud_id is None and self.lwd_edge_channel is not None:
                logger.info(
                    "[Lwd][edge-notify] abort req=%s unbound: local cleanup only",
                    request_id,
                )
                continue
            if cloud_id is None:
                cloud_id = self._lwd_default_cloud_id()
            if not self._lwd_publish_ordered(
                LwdAbortNotify(request_id=request_id, edge_id=self._lwd_edge_id),
                cloud_id,
            ):
                logger.info(
                    "[Lwd][edge-notify] abort req=%s deferred (backpressure)",
                    request_id,
                )
            else:
                logger.info(
                    "[Lwd][edge-notify] AbortNotify req=%s cloud=%d",
                    request_id, cloud_id,
                )

    def lwd_edge_finish(self, request_id: str) -> None:
        """收尾记账(§11.4):向云上报**全量块哈希链**(prompt + 生成段,HMAC 域)。

        生成段(completion)哈希只有持租户密钥的边能算,云侧据此校验
        prompt 前缀链未变并发布生成段摘要——**云不持密钥、不接触明文**
        (参考分支同语义:原经 SO 附发 EdgeCloudFinishedRequest,我们走
        控制面 LwdFinishNotify)。

        fail-open:未启用协调/无密钥/建链或发布失败均仅告警,不阻断终结
        路径(记账缺失不影响数据面正确性)。"""
        prompt_ids = self._lwd_finish_prompt_ids.pop(request_id, None)
        gen_ids = self._lwd_finish_gen_ids.pop(request_id, None)
        media_items = self._lwd_finish_media_items.pop(request_id, None) or ()
        self._lwd_gen_chain_sent.pop(request_id, None)
        self._lwd_gen_chain_state.pop(request_id, None)
        if prompt_ids is None:
            return
        vllm_config = getattr(self, "vllm_config", None)
        if vllm_config is None:
            return
        coord_cfg = getattr(vllm_config, "lwd_coordination", None)
        if coord_cfg is None or not getattr(coord_cfg, "enabled", False):
            return
        if not coord_cfg.tenant_key_file:
            return
        try:
            # finish 链粒度取本请求 probe 时的建链粒度(而非全局学习值):
            # 云侧存储的 manifest 即 probe 粒度,两处一致校验才能对账。
            # 学习换挡只发生在重启后首个请求,故需按请求记账。
            probe_bs = self._lwd_probe_bs_by_req.pop(request_id, None)
            hasher = self._lwd_prefix_hasher(coord_cfg, vllm_config, probe_bs)
            completion_ids = gen_ids or []
            manifest = hasher.build_manifest(
                request_id, list(prompt_ids) + completion_ids,
                # 收尾存链与 probe 查链同一媒体口径
                media_items=media_items,
            )
            message = LwdFinishNotify(
                request_id=request_id,
                prompt_tokens=len(prompt_ids),
                completion_tokens=len(completion_ids),
                full_block_hashes=list(manifest.full_block_hashes),
                edge_id=self._lwd_edge_id,
            )
            cloud_id = self._lwd_binding.cloud_of(
                request_id, default=self._lwd_default_cloud_id()
            )
            if not self._lwd_publish_ordered(message, cloud_id):
                logger.info(
                    "[Lwd][edge] finish announce deferred req=%s", request_id
                )
                return
            logger.info(
                "[Lwd][edge] finish announced req=%s prompt=%d completion=%d "
                "blocks=%d cloud=%d",
                request_id, message.prompt_tokens, message.completion_tokens,
                len(message.full_block_hashes), cloud_id,
            )
        except Exception:  # noqa: BLE001  fail-open:记账缺失不阻断终结
            logger.warning(
                "[Lwd][edge] finish announce failed (fail-open)", exc_info=True
            )

    def _lwd_announce_gen_chain(self, request_id: str) -> None:
        """按步增量上报生成段链:满块数增加才发,全量替换语义。

        云侧据此把生成段块按边侧 HMAC 链键入块池(多轮续写可命中)。会丢包
        不要紧:下一条带全量链,云侧覆盖写自愈。fail-open:未启用协调/
        无密钥/发布失败仅告警——链缺失只让云侧生成段块晚登记(fail-safe
        降级),不影响数据面。

        增量建链(E2 根治):先按满块数廉价预判(没新增块零哈希零计算),
        有新增也只对新增满块续链(chain_extend),替代旧版每步对
        prompt+gen 全量 O(N) 重建(长输出下 O(N²) 的主循环 CPU)。"""
        prompt_ids = self._lwd_finish_prompt_ids.get(request_id)
        vllm_config = getattr(self, "vllm_config", None)
        if prompt_ids is None or vllm_config is None:
            return
        coord_cfg = getattr(vllm_config, "lwd_coordination", None)
        if coord_cfg is None or not getattr(coord_cfg, "enabled", False):
            return
        if not coord_cfg.tenant_key_file:
            return
        try:
            hasher = self._lwd_prefix_hasher(coord_cfg, vllm_config)
            gen_ids = self._lwd_finish_gen_ids.get(request_id) or []
            tokens = list(prompt_ids) + gen_ids
            # 廉价预判:满块数没增加直接返回(任何哈希之前)
            if len(tokens) // hasher.block_size <= self._lwd_gen_chain_sent.get(
                request_id, 0
            ):
                return
            state = self._lwd_gen_chain_state.setdefault(request_id, [])
            # prompt 段按媒体感知链播种(与 probe/finish 同源),此后
            # chain_extend 纯增量;纯文本请求播种与旧链逐位一致
            # (chain_extend 空链与 build_manifest 同源)。
            if not state:
                state.extend(hasher.build_manifest(
                    request_id, list(prompt_ids),
                    media_items=(
                        self._lwd_finish_media_items.get(request_id) or ()
                    ),
                ).full_block_hashes)
            state.extend(hasher.chain_extend(state, tokens))
            blocks = list(state)
            cloud_id = self._lwd_binding.cloud_of(
                request_id, default=self._lwd_default_cloud_id()
            )
            message = LwdGenChainNotify(
                request_id=request_id,
                block_hashes=blocks,
                block_size=hasher.block_size,
                edge_id=self._lwd_edge_id,
            )
            if not self._lwd_publish_ordered(message, cloud_id):
                # deferred 也推进水位:消息已按全量链入队保序,重复入队
                # 只浪费不纠错;后续新增块的新链照常覆盖
                self._lwd_gen_chain_sent[request_id] = len(blocks)
                logger.info(
                    "[Lwd][edge] gen-chain deferred req=%s blocks=%d",
                    request_id, len(blocks),
                )
                return
            self._lwd_gen_chain_sent[request_id] = len(blocks)
            logger.info(
                "[Lwd][edge] gen-chain announced req=%s blocks=%d cloud=%d",
                request_id, len(blocks), cloud_id,
            )
        except Exception:  # noqa: BLE001  fail-open:链缺失只降级不阻断
            logger.warning(
                "[Lwd][edge] gen-chain announce failed (fail-open)", exc_info=True
            )

    def _lwd_prefix_hasher(
        self,
        coord_cfg,
        vllm_config,
        hash_block_size: int | None = None,
    ):
        """惰性构造并缓存租户 HMAC hasher(probe / 生成段链 / 收尾共用)。

        ``hash_block_size`` 指定链的块粒度:缺省为边块粒度(cache_config
        .block_size,学习前的行为);传入从 probe 响应学到的云 hash 粒度后,
        链与云侧采纳条件 len(链)==prompt//粒度 恒对齐。同密钥不同粒度
        各缓存一个实例。"""
        if hash_block_size is None:
            hash_block_size = vllm_config.cache_config.block_size
        cached = self._lwd_hasher_by_granularity.get(hash_block_size)
        if cached is not None:
            return cached

        with open(coord_cfg.tenant_key_file, "rb") as f:
            tenant_key = f.read()
        from vllm.v1.lwd_control.control_communication.lwd_prefix import (
            LwdPrefixHasher,
        )

        hasher = LwdPrefixHasher(
            tenant_key,
            hash_block_size,
            # 指纹只影响媒体覆盖块(纯文本链逐位不变);与 probe client
            # 同源,保证查链/存链两侧媒体块哈希一致
            self._lwd_processor_fingerprint(),
        )
        self._lwd_hasher_by_granularity[hash_block_size] = hasher
        return hasher

    def lwd_edge_update_progress(self, executed: dict[str, int]) -> None:
        """步末登记执行量;嵌入完结即转入 awaiting。

        notify 恒成功前提:排程即派发即执行(同步执行),executed 与
        本步排程集恒等,原生 _update_after_schedule 的乐观推进即真实
        水位,无需回退对账。前提被破坏(发布队满未派发)时进度将
        静默虚高、该 chunk 永久丢失,由发布通道不丢消息保证不发生。

        嵌入完结 = 边侧工作结束而非请求结束:走原生 finish_requests
        做全套簿记(移出 running/requests、释放边侧 KV、进
        finished_req_ids 通知 worker 释放缓存——该通知随下一个排程步
        下发,引擎纯睡眠期滞后),同时登记 awaiting;前端未收到任何
        输出会继续等待,终结由云结果驱动(lwd_edge_deliver_tokens)。"""
        finished_ids: list[str] = []
        for request_id in executed:
            request = self.requests.get(request_id)
            if request is None:
                # 本步内已终结(abort/更早完成):迟到的登记无对象
                continue
            if request.num_computed_tokens >= request.num_prompt_tokens:
                finished_ids.append(request_id)
        if finished_ids:
            self.finish_requests(finished_ids, RequestStatus.FINISHED_STOPPED)
            now = time.monotonic()
            for request_id in finished_ids:
                self._lwd_awaiting[request_id] = now
                self._lwd_mrope_positions_dict.pop(request_id, None)
                logger.info(
                    "[Lwd][edge-progress] req=%s embed done -> awaiting",
                    request_id,
                )

    def lwd_edge_deliver_tokens(
        self, request_id: str, token_ids: list[int], finished: bool
    ) -> bool:
        """云结果投递(awaiting 消费点,引擎步内调用)。

        token_ids 由边侧 worker unembedding 产生,本层不消费内容,
        仅做生命周期对账:
        - 请求在 awaiting:认领;finished=True 时出 awaiting(请求本体
          已在嵌入完结时清出调度器,无需再 finish);
        - 请求不在(已 abort/更早完结/未知):迟到结果,返回 False,
          调用方丢弃告警(幂等,不复活)。

        token_ids/finished 的输出组包归引擎层(EngineCoreOutputs)。"""
        if request_id not in self._lwd_awaiting:
            logger.warning(
                "[Lwd][edge-deliver] stale result for req=%s (not awaiting)",
                request_id,
            )
            return False
        if token_ids:
            self._lwd_finish_gen_ids.setdefault(request_id, []).extend(token_ids)
            self._lwd_announce_gen_chain(request_id)
        if finished:
            # 收尾记账先于解绑(上报按绑定表定向选云)
            self.lwd_edge_finish(request_id)
            del self._lwd_awaiting[request_id]
            # finish 即解绑:请求与云实例的生命周期绑定结束
            # (迟到结果不再认领,绑定表出表)
            self._lwd_binding.unbind(request_id)
        logger.info(
            "[Lwd][edge-deliver] req=%s tokens=%d finished=%s",
            request_id, len(token_ids), finished,
        )
        return True

    def _lwd_validate_request(self, request: Request) -> None:
        """模式边界校验;违规抛 ValueError,在入队之前拒绝,错误经
        add_request 调用链回到客户端 error 路径。

        边界 = 边侧能力面:边是 embedding 属主(拒绝客户端自带
        prompt_embeds),只处理纯文本补全(拒 pooling/结构化
        输出),prompt 非空;不上 wire 的采样参数(logit_bias/
        allowed_token_ids/logprobs)缺省即拒,不静默丢约束。

        多模态白名单:仅 image;video/audio 等其余模态与 EVS
        (multimodal pruning)拒绝——merge/mrope 链路按 image 语义
        实现,其余模态未适配,静默放过即错算。"""
        if request.prompt_embeds is not None:
            raise ValueError(
                f"[LWD] prefill-only mode does not accept client-provided "
                f"prompt_embeds (request {request.request_id}); the edge "
                "is the embedding owner"
            )
        if request.mm_features:
            unsupported_mm = sorted(
                {
                    feature.modality
                    for feature in request.mm_features
                    if feature.modality not in ("image", "prompt_embeds")
                }
            )
            if unsupported_mm:
                raise ValueError(
                    f"[LWD] prefill-only mode supports image-only "
                    f"multimodal requests; got modality "
                    f"{', '.join(unsupported_mm)} "
                    f"(request {request.request_id})"
                )
            if (
                self.vllm_config.model_config.multimodal_config
                .is_multimodal_pruning_enabled()
            ):
                raise ValueError(
                    "[LWD] prefill-only mode does not support multimodal "
                    f"pruning (EVS) (request {request.request_id})"
                )
        if request.pooling_params is not None:
            raise ValueError(
                "[LWD] prefill-only mode does not support pooling requests "
                f"(request {request.request_id})"
            )
        if request.use_structured_output:
            raise ValueError(
                "[LWD] prefill-only mode does not support structured output "
                f"(request {request.request_id})"
            )
        sp = request.sampling_params
        if sp.n != 1:
            raise ValueError(
                "[LWD] prefill-only mode requires n=1; got n="
                f"{sp.n} (request {request.request_id}): n 不上 wire,云侧只采样"
                "一路,多路会静默缺路"
            )
        unsupported = [
            name
            for name, value in (
                ("logit_bias", sp.logit_bias),
                ("allowed_token_ids", sp.allowed_token_ids),
                ("logprobs", sp.logprobs),
                ("prompt_logprobs", sp.prompt_logprobs),
            )
            if value is not None
        ]
        if unsupported:
            raise ValueError(
                "[LWD] prefill-only mode does not support sampling "
                f"param(s) {', '.join(unsupported)} "
                f"(request {request.request_id}): not carried on the "
                "edge->cloud wire, cloud would silently sample without them"
            )

def lwd_build_unembed_batch(notify: LwdC2eNotify) -> SchedulerOutput:
    """组 UNEMBED 批(引擎步内调用,云载荷派发给边 worker 做 lm_head)。

    云结果不经过原生 schedule,无原生排程产物可用——以 make_empty
    为骨架:
    - lwd_batch 携带 LwdUnembedBatch:req_ids(隐藏行序)/
      num_accept_tokens/top_id_ths 逐请求透传自 c2e;批序号将随 c2e
      通告携带(规划),当前占位 0;recv_num_elements
      (DOWN 通道每请求接收元素数)与 out_token_idxs(生成序号)控制面
      不可知,留空由数据面按 DOWN 张量实收推导;
    - 请求集合同步镜像到 num_scheduled_tokens:值 = 该请求本步
      hidden 行数(num_accepted_tokens 对位,spec 步可 >1),保持
      原生管道字段语义一致,不作 token 预算解释。
    """
    scheduler_output = SchedulerOutput.make_empty()
    # 每请求本步还原的 token 数 = hidden 行数 = num_accepted_tokens
    # (spec 步一请求可多行,非 spec 恒 1);与 req_ids 按位对齐,错配
    # fail-fast
    assert len(notify.num_accepted_tokens) == len(notify.req_ids)
    scheduler_output.num_scheduled_tokens = dict(
        zip(notify.req_ids, notify.num_accepted_tokens)
    )
    scheduler_output.total_num_scheduled_tokens = sum(
        notify.num_accepted_tokens
    )
    scheduler_output.lwd_batch = LwdBatch(
        batch_type=LwdBatchType.LWD_UNEMBED,
        seqno=notify.down_seqno,
        batch_meta=LwdUnembedBatch(
            req_ids=list(notify.req_ids),
            num_accept_tokens=list(notify.num_accepted_tokens),
            recv_num_elements=notify.hidden_num_elements,
            out_token_idxs=[],
            top_id_ths=list(notify.top_id_ths),
            cloud_id=notify.cloud_id,
            token_ids=[list(t) for t in notify.token_ids],
        ),
    )
    scheduler_output.lwd_c2e_notify = [notify]
    return scheduler_output
