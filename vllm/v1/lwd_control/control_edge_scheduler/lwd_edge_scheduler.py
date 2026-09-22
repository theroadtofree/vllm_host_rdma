"""边侧调度器:纯 prefill 调度 + 控制面发布(notify/abort/seqno)。

原生 AsyncScheduler 的两个前提在边侧不成立:prompt 算完会转 decode、
请求只能由模型输出终结。边侧只做 embedding(执行层无 decode),故需
专用调度器接管请求的边侧生命周期:

  入队 -> EMBEDDING(单请求组批/chunked 决策/范围预告)
       -> 嵌入完结:走原生 finish_requests 清出调度器(释放边侧 KV
          簿记、通知 worker 释放缓存),登记 awaiting
       -> AWAITING(等待云结果;前端未收到输出继续等待)
       -> 云结果终结(lwd_edge_deliver_tokens);迟到结果幂等丢弃。

纯 prefill 的实现依据:schedule() 全量复用原生——边侧请求从不产生
输出 token(num_tokens_with_spec 恒等于 num_prompt_tokens),且嵌入
完结当步即被清出调度器,原生 RUNNING 段每步只会调度剩余 prefill,
decode 分支不可达。

单请求组批约束:prefill 批最多含一个请求。目的:数据面 chunk 流按
请求连续(全局 seqno 链上单请求的 chunk 相邻),消除跨请求交错带来
的张量配对/重组复杂度。
"""

from __future__ import annotations

import time
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

# add 预告发布重试:次数 x 递增间隔(共约 3s),耗尽即请求级报错
_LWD_ADD_RETRY_STEPS = 5
_LWD_ADD_RETRY_INTERVAL_S = 0.2


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
        # 生成段链增量上报水位(request_id -> 已上报满块数):只在满块数
        # 增加时发一条(全量替换语义),避免逐步刷控制面。
        self._lwd_gen_chain_sent: dict[str, int] = {}
        # 云侧块哈希粒度学习缓存(cloud_id -> 云 hash 粒度):probe 响应头
        # 携带云侧 hash_block_size(压缩/混合粒度模型上与边块粒度不同,
        # 如 DSv4 云 hash=8 而边块 128)。链按该粒度构建后,云侧采纳条件
        # len(链)==prompt//hash 粒度 恒成立,不再要求 prompt 为边块整数倍。
        # 未学习(首请求前)回退边块粒度,行为与学习前一致。
        self._lwd_hash_bs_by_cloud: dict[int, int] = {}
        # 按粒度缓存的 hasher(同租户密钥不同 block_size;数量极小,
        # 实际部署每云一个粒度)
        self._lwd_hasher_by_granularity: dict[int, Any] = {}
        # 请求 probe 链粒度记账(request_id -> 建链粒度):学习换挡只发生
        # 在重启后首个请求(probe 用旧粒度、finish 须用同一粒度才能与云侧
        # 存储的 manifest 对账),稳态请求 probe/finish 同粒度自然一致。
        self._lwd_probe_bs_by_req: dict[str, int] = {}
        # 探前缀客户端惰性缓存(复用 hasher/tenant_key,避免每请求读文件重建)
        self._lwd_probe_client_cache = None
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

    def schedule(self) -> SchedulerOutput:
        """单请求组批 + 原生分块决策。

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
        return self._lwd_schedule_single()

    def _lwd_pick_prefill_req_id(self) -> str | None:
        """选下一步 embed 工作单元:running 中第一个未发完的 prefill
        优先(断点续传,先收尾再开新),否则 waiting 队首(FCFS 到达序);
        两处皆无返回 None。"""
        running_prefill = next(
            (r for r in self.running if r.num_computed_tokens < r.num_prompt_tokens),
            None,
        )
        if running_prefill is not None:
            return running_prefill.request_id
        return self.waiting.peek_request().request_id if self.waiting else None

    def _lwd_schedule_single(self) -> SchedulerOutput:
        """单请求组批:picker 选一个工作单元,经基类可见集机制单独调度。

        选择规则见 _lwd_pick_prefill_req_id;队列剔除/隔离/拼回复用基类
        _lwd_schedule_for_visible_reqs(waiting/skipped 来源走原生准入
        窗口,running 来源走续跑)。语义注记:与旧手写容器交换不同,
        本步被抢占的请求回 waiting 尾部、被跳过的回 skipped 队首,均取
        基类统一语义,不再做队首回插。"""
        req_id = self._lwd_pick_prefill_req_id()
        if req_id is not None:
            logger.info("[Lwd][edge-sched] pick req=%s", req_id)
        return self._lwd_schedule_for_visible_reqs([req_id] if req_id else [])

    def lwd_edge_max_num_seqs_check(self) -> bool:
        """max_num_seqs 适配检查:云侧在途水位(running + awaiting)是否
        还有名额,True=可开新请求。

        awaiting 请求已清出调度器,原生准入只数 running(边侧恒≤1)
        永远拦不住;以 running+awaiting 对账云侧在途数,达到
        max_num_running_reqs 即满员。续传豁免不在本判断(schedule
        闸门经 _lwd_has_prefill_chunk_inflight 放行收尾);请求到达
        时的 announce 亦不受约束(云只登记不计算)。

        注:云侧复用下一台云收多边的请求,边侧按云聚合水位并不准确,
        维持本判据(本地在途对账),按云配额留待云侧反馈机制接入。"""
        return len(self.running) + len(self._lwd_awaiting) < self.max_num_running_reqs

    def _lwd_has_prefill_chunk_inflight(self) -> bool:
        """running 中是否存在未发完的 embed 请求(续传收尾中)。

        与 _lwd_pick_prefill_req_id 的续传分支同判据,两处需保持一致:
        闸门放行收尾的前提是 picker 必然挑中该续传请求而非开新。"""
        return any(
            req.num_computed_tokens < req.num_prompt_tokens
            for req in self.running
        )

    def lwd_edge_add_request(self, request: Request) -> None:
        """请求入口:边界校验 -> 登记绑定表 -> 云预告 -> 本地入队。

        预告先于本地登记:云侧视图领先本地工作,云只能准备不能提前
        计算(没有 chunk 预告就不会开算)。abort_immediately 请求走
        finish + abort 出口,与原生语义一致。
        """
        self._lwd_validate_request(request)
        # 收尾记账备料:prompt ids 留档(终结时 Request 已不在调度器)
        self._lwd_finish_prompt_ids[request.request_id] = list(
            request.prompt_token_ids
        )
        # add_request 前自动 HTTP 探前缀(manifest.request_id == 数据面
        # request_id,云侧按此键落预留;命中时推进 num_computed 只发尾巴)。
        # 返回的块哈希链随 notify 上云:协调启用时恒为 HMAC 协调链,与
        # 云侧块池/协调器同域(不依赖探测成败),续算边界两侧一致。
        block_hashes, prefix_hit_tokens = self._lwd_probe_prefix(request)
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
        if request.abort_immediately:
            self.finish_requests([request.request_id], RequestStatus.FINISHED_ABORTED)
            self.lwd_edge_abort([request.request_id])

    def _lwd_probe_prefix(self, request: Request) -> tuple[list[bytes], int]:
        """add_request 前自动 HTTP 探前缀,manifest.request_id 取数据面
        request_id,云侧按 (edge_id 包裹后的) 同键落预留(claim/complete 同键)。

        fail-open:未启用/缺配置/探测失败均放行;命中时推进
        request.num_computed_tokens 使边侧只 embed/发尾巴(续算)。

        返回 (随 notify 上云的 prompt 满块哈希链, 续算边界 token 数):
        协调启用时链恒为 HMAC 协调链(本地产 manifest,与云侧块池/
        协调器同域),命中值即 trim 后的边侧 chunk 起点,云侧据此封顶
        resume;未启用/回退时返回 request 自身原生块哈希 + 0(不续算)。"""
        vllm_config = getattr(self, "vllm_config", None)
        if vllm_config is None:
            return list(request.block_hashes), 0
        coord_cfg = getattr(vllm_config, "lwd_coordination", None)
        if coord_cfg is None or not getattr(coord_cfg, "enabled", False):
            return list(request.block_hashes), 0
        if not coord_cfg.control_url or not coord_cfg.tenant_key_file:
            return list(request.block_hashes), 0
        try:
            import asyncio

            import aiohttp

            from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_client import (
                LwdEdgePrefixClient,
            )

            client = self._lwd_probe_client_cache
            if client is None:
                with open(coord_cfg.tenant_key_file, "rb") as f:
                    tenant_key = f.read()
                # 探前缀客户端复用(hasher/密钥只建一次),并把总超时钳到
                # 3s:死锁残余时快速 fail-open,避免边侧主循环 10s 级卡顿。
                # 探测链粒度:优先用已学习的云 hash 粒度(稳态下 probe 的
            # manifest 即云域,ratio=1 直通);未学习回退边块粒度。
            cloud_id = self._lwd_default_cloud_id()
            known_bs = self._lwd_hash_bs_by_cloud.get(cloud_id)
            build_bs = known_bs or vllm_config.cache_config.block_size
            client = LwdEdgePrefixClient(
                    coord_cfg.control_url,
                    build_bs,
                    tenant_key,
                    edge_id=self._lwd_edge_id,
                    consumer_id=coord_cfg.consumer_id,
                    connect_timeout=min(coord_cfg.connect_timeout, 3.0),
                )
                self._lwd_probe_client_cache = client
            manifest = client.build_manifest(
                request.request_id, list(request.prompt_token_ids)
            )
            chain = list(manifest.full_block_hashes)

            async def _probe():
                async with aiohttp.ClientSession() as session:
                    return await client.negotiate(
                        session,
                        request.request_id,
                        list(request.prompt_token_ids),
                    )

            result = asyncio.run(_probe())
            # 学习云侧 hash 粒度;与本次建链粒度不一致时按其重建链。
            # notify 在 probe 之后才发,故首请求(首次学习)即生效:非
            # 整除 prompt 的链覆盖到云 hash 粒度,采纳条件恒成立。
            learned = int(result.block_size)
            self._lwd_hash_bs_by_cloud[cloud_id] = learned
            if learned != build_bs:
                chain = list(
                    self._lwd_prefix_hasher(
                        coord_cfg, vllm_config, learned
                    ).build_manifest(
                        request.request_id, list(request.prompt_token_ids)
                    ).full_block_hashes
                )
            # 记账本请求 probe 的建链粒度(注意是 build_bs 而非学习值):
            # 云侧存储的 manifest 即 probe 粒度,finish 须按同一粒度建链
            # 对账;学习换挡只发生在重启后首个请求,故按请求记账。
            self._lwd_probe_bs_by_req[request.request_id] = build_bs
            logger.info(
                "[Lwd][edge] prefix probe req=%s hit_tokens=%d blocks=%d "
                "hash_bs=%d",
                request.request_id, result.hit_tokens, len(chain), learned,
            )
            # 命中边界以云侧块池单真源为准(probe 现读块池,与云侧请求
            # resume 同池同链),推进 num_computed 后边侧只 embed/发尾巴,
            # chunk offset == 云侧 computed,续算闭环。命中值随 notify
            # 下发,云侧按它封顶 resume——probe 超时/并发下池增长时,
            # 云侧不会越过边侧实际裁剪的边界。
            if result.hit_tokens > 0:
                request.num_computed_tokens = result.hit_tokens
            return chain, result.hit_tokens
        except Exception:  # noqa: BLE001  fail-open:探测失败不阻断请求
            logger.warning(
                "[Lwd][edge] prefix probe failed (fail-open)", exc_info=True,
            )
            return list(request.block_hashes), 0

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
        """对新调度的 prefill 块发 LwdRangeNotify(seqno 先行)。

        seqno 无空洞契约:UP 数据通道按连续号序配对(通道层对超前号
        扣留等待,一个空洞即永久挂死整条链),因此号只能分配给真正
        上 wire 的块:
        - peek-then-advance:发布成功才进位该请求所属云的计数器;
        - 单 notify 前提:本方法每步至多发一条,依赖单请求组批
          约束。若放开多请求组批,部分成功的 notify 已上 wire 而
          整步不派发,会同时产生号空洞与张量失配,届时必须改为按
          已成功子集执行。
        云侧复用 seqno 按 (cloud_id) 对分域:每对通道内无空洞契约
        独立成立,互不影响。

        前提:控制面发布通道不丢消息,publish 恒成功——步末回退
        对账已按此前提移除。若前提被破坏返回 False,调用方本步不
        派发但进度不回退,该 chunk 永久丢失;重复预告在云侧按
        (request_id, offset) 幂等登记。"""
        scheduled = scheduler_output.num_scheduled_tokens
        for request_id, num_tokens in scheduled.items():
            request = self.requests.get(request_id)
            if request is None:
                continue
            cloud_id = self._lwd_binding.cloud_of(
                request_id, default=self._lwd_default_cloud_id()
            )
            # _update_after_schedule 已乐观推进 num_computed,起点需回退本步量
            offset = request.num_computed_tokens - num_tokens
            seqno = self._lwd_seqno.get(cloud_id, 0)
            positions = self._lwd_mrope_positions_dict.get(request_id)
            has_mrope = positions is not None
            if not self._lwd_publish(
                LwdRangeNotify(
                    request_id=request_id,
                    offset=offset,
                    num_tokens=num_tokens,
                    seqno=seqno,
                    edge_id=self._lwd_edge_id,
                    has_mrope=has_mrope,
                ),
                cloud_id,
            ):
                return False
            self._lwd_seqno[cloud_id] = seqno + 1
            logger.info(
                "[Lwd][edge-notify] req=%s offset=%d num=%d seqno=%d "
                "cloud=%d has_mrope=%s",
                request_id, offset, num_tokens, seqno, cloud_id, has_mrope,
            )
            # 发布成功即组 EMBED 批挂 SO:seqno 是数据面发云张量的
            # 配对键(与云侧 RangeNotify 登记同值),embed 载荷为本
            # chunk 的 token 片段
            mrope_slice: list[list[int]] = []
            if has_mrope:
                assert positions is not None
                # [3,N] 切本 chunk 列 -> [n,3] 行序与 token 轴对齐
                mrope_slice = (
                    positions[:, offset : offset + num_tokens].t().tolist()
                )
            scheduler_output.lwd_batch = LwdBatch(
                batch_type=LwdBatchType.LWD_EMBED,
                seqno=seqno,
                batch_meta=LwdEmbedBatch(
                    req_ids=[request_id],
                    token_ids=[
                        list(
                            request.prompt_token_ids[offset : offset + num_tokens]
                        )
                    ],
                    edge_id=self._lwd_edge_id,
                    cloud_id=cloud_id,
                    token_offsets=[offset],
                    has_mrope=has_mrope,
                    mrope_positions=mrope_slice,
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

        失败语义 fail-fast:发布队满时短退避重试(瞬态背压几乎必在
        秒级窗口内腾出),耗尽即抛 RuntimeError——异常沿 add_request
        调用链回前端 error 通道,用户立即得到失败;不做无限阻塞
        重试(发布点在引擎主线程,云宕机会把整个引擎卡死在 add)。
        此时请求未入队、云侧零残留,无需补发 abort。"""
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
        for attempt in range(_LWD_ADD_RETRY_STEPS):
            if self._lwd_publish(message, cloud_id):
                logger.info(
                    "[Lwd][edge-notify] request meta announced: req=%s "
                    "prompt=%d cloud=%d",
                    request_id, num_prompt_tokens, cloud_id,
                )
                return
            time.sleep(_LWD_ADD_RETRY_INTERVAL_S * (attempt + 1))
        raise RuntimeError(
            f"[LWD] add-request notify for {request_id} dropped: publish "
            f"queue full after {_LWD_ADD_RETRY_STEPS} retries "
            f"(cloud {cloud_id} PRE_OUT consumption stalled?)"
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
            self._lwd_gen_chain_sent.pop(request_id, None)
            # mrope 台账随 abort 清理(多模态准入期预计算的位置缓存)
            self._lwd_mrope_positions_dict.pop(request_id, None)
            cloud_id = self._lwd_binding.unbind(request_id)
            if cloud_id is None and self.lwd_edge_channel is not None:
                logger.info(
                    "[Lwd][edge-notify] abort req=%s unbound: local cleanup only",
                    request_id,
                )
                continue
            if cloud_id is None:
                cloud_id = self._lwd_default_cloud_id()
            if not self._lwd_publish(
                LwdAbortNotify(request_id=request_id, edge_id=self._lwd_edge_id),
                cloud_id,
            ):
                logger.warning(
                    "[Lwd] drop abort signal for %s: publish queue full", request_id
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
        self._lwd_gen_chain_sent.pop(request_id, None)
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
                request_id, list(prompt_ids) + completion_ids
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
            if not self._lwd_publish(message, cloud_id):
                logger.warning(
                    "[Lwd][edge] finish announce publish failed req=%s", request_id
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
        降级),不影响数据面。"""
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
            manifest = hasher.build_manifest(
                request_id, list(prompt_ids) + gen_ids
            )
            blocks = list(manifest.full_block_hashes)
            if len(blocks) <= self._lwd_gen_chain_sent.get(request_id, 0):
                return
            cloud_id = self._lwd_binding.cloud_of(
                request_id, default=self._lwd_default_cloud_id()
            )
            message = LwdGenChainNotify(
                request_id=request_id,
                block_hashes=blocks,
                block_size=hasher.block_size,
                edge_id=self._lwd_edge_id,
            )
            if not self._lwd_publish(message, cloud_id):
                logger.warning(
                    "[Lwd][edge] gen-chain publish failed req=%s blocks=%d",
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

        hasher = LwdPrefixHasher(tenant_key, hash_block_size)
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
        ),
    )
    scheduler_output.lwd_c2e_notify = [notify]
    return scheduler_output
