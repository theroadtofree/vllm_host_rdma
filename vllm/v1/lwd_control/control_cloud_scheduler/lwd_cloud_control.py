# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""云侧前缀协商 HTTP 控制面(SSE 网关形态)。

从参考分支 ``edge_cloud/cloud_control.py`` 演化:父进程 FastAPI 应用 + 桥
(LwdCloudControlBridge),子进程处理器(LwdCloudControlProcessor,poll 非阻塞),
经两条 multiprocessing 队列对接。宿主是 ``LwdCloudEngineCore``。

控制面承载 probe + usage 两个事件,走 **OpenAI 兼容 SSE**:
  1. Probe 结果放 ``delta.content`` 的 JSON 控制消息(网关 DTO 转换不丢);
  2. 紧跟空 delta 帧,冲刷网关末 chunk 缓冲(NewAPI 等);
  3. 期间每 10s 发 SSE comment 心跳,防网关上游空闲超时;
  4. 推理完结后发标准 OpenAI usage chunk + ``[DONE]``,网关据此计费;拿不到
     真实用量时只发 comment + ``[DONE]``,绝不伪造零用量帧。

协议门禁:文本走 ``edge-cloud-prefix-v1``;图片走 ``edge-cloud-prefix-v2``
且必带 ``X-Edge-Cloud-MM-ABI`` 指纹(媒体覆盖块的建链口径),两套栈不得互相
降级。响应头仍保留 ``X-Edge-Cloud-*`` 供直连调试/网关观测,不参与协商。
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import json
import queue
import threading
import time
from collections.abc import Mapping
from dataclasses import replace as dataclass_replace
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.logger import init_logger
from vllm.v1.lwd_control.lwd_comm.lwd_id_adapter import wrap_req_id
from vllm.v1.lwd_control.lwd_comm.lwd_mm_identity import (
    compute_processor_fingerprint,
    mm_abi_header_value,
    validate_mm_abi_header,
)
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import (
    HEADER_EDGE_ID,
    HEADER_MM_ABI,
    HEADER_PROTOCOL,
    HEADER_REQUEST_ID,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_MM,
    LwdPrefixManifest,
    LwdProbeResult,
    LwdUsage,
)

logger = init_logger(__name__)

# SSE 心跳间隔(秒):须小于网关 stream_idle_timeout,留足余量
_USAGE_HEARTBEAT_INTERVAL = 10.0
# usage 等待总超时(秒):覆盖完整推理时长,超时后结束流(不伪造用量)
_USAGE_WAIT_TIMEOUT = 600.0
# 云侧探针上限 = 边侧 ``lwd_coordination.probe_timeout`` 减这个余量:必须让云
# 先作答(200 或 503),否则边侧超时放弃时云侧可能仍在等子进程,那笔预留与
# 计费流都成了白挂。503 时边侧按 fail-open 走全量,行为与今天一致。
_CLOUD_PROBE_HEADROOM_S = 5.0


class LwdCloudControlBridge:
    """父进程侧:把 HTTP 协程与子进程的 probe/usage 事件用队列对接。

    等待走 **per-request asyncio.Future**,零线程占用(沿用参考分支
    ``edge_cloud/cloud_control.py`` 的形态,该形态已过 NewAPI/Higress 真机
    验收):dispatch 线程只把事件经 ``loop.call_soon_threadsafe`` 投回事件
    循环,handler 协程直接 await。``_loop`` 在 ``probe()``(handler 协程内)
    捕获,即 uvicorn 实际跑着的那条循环——不在启动线程取,避免"投给了没在跑
    的循环"。

    为什么不用 ``run_in_executor(None, waiter.get)``:usage 等待要常驻到请求
    推理完结(真机 hold≈8s),线程池形态下一笔在飞 = 一条 executor 线程,默认
    池上限 ``min(32, cpu+4)`` 一撞顶,新探针连"等结果"都排不上队(真机证据见
    8cfb9f499:busy_usage=32、exec_delay_ms=-1、probe total_ms=60000);更糟的是
    ``wait_for`` 超时后已开跑的线程**不可取消**,而 waiter 已从字典摘除、迟到
    事件唤不醒它 → 每笔超时永久漏一条线程,漏满一池控制面彻底不答。Future
    形态下超时只是取消一个协程,在飞数与线程数解耦,池尺寸这个容量参数整体
    作废。早期本类 docstring 写"跨线程 future 不 resolve 会诡异卡死"并因此
    改用队列,查提交历史无该事故的记录(该形态出自初次移植 2e0979767)。
    """

    def __init__(self, command_queue: Any, event_queue: Any) -> None:
        # 云内 DP 每引擎命令队列:接受单队列(dp=1,行为与旧版逐值一致)或
        # 队列列表(dp>1,probe 扇出按引擎寻址);event 队列保持共享。
        if isinstance(command_queue, (list, tuple)):
            self._command_queues: list[Any] = list(command_queue)
        else:
            self._command_queues = [command_queue]
        self._command_queue = self._command_queues[0]
        self._event_queue = event_queue
        self._probe_futures: dict[str, asyncio.Future] = {}
        self._usage_futures: dict[str, asyncio.Future] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        # 扇出/预留应答收集:键为 reply_key(扇出 "<rid>#probe"、赢家预留
        # "<rid>#reserve"),与单引擎 future 命名空间隔离,迟到的扇出
        # 应答找不到 waiter 即丢弃,不污染后续同 rid 流量
        self._fanout_waiters: dict[str, queue.Queue] = {}
        # 并列轮转计数器(命中并列时依次轮换赢家引擎,防热点钉死)
        self._rr_counter = 0
        self._lock = threading.Lock()
        self._closed = False
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_events, name="lwd-ctrl-dispatch", daemon=True
        )

    def start(self) -> None:
        self._dispatch_thread.start()

    def close(self) -> None:
        """关停:停 dispatch 线程,并把仍在等的 future 置异常。

        由 ``start_lwd_cloud_control_server`` 注册的 atexit 钩子调用——本支
        uvicorn 起在守护线程、没有 shutdown 通道,不接这个钩子的话退出路径
        上 dispatch 线程只能靠 daemon 被硬杀。"""
        self._closed = True
        with self._lock:
            pending = [
                *self._probe_futures.values(), *self._usage_futures.values()
            ]
            self._probe_futures.clear()
            self._usage_futures.clear()
            loop = self._loop
        alive = [future for future in pending if not future.done()]
        if loop is None or not alive:
            return
        error = RuntimeError("cloud control plane stopped")

        def _fail_pending() -> None:
            for future in alive:
                if future.done():
                    continue
                future.set_exception(error)
                # usage future 常常无人认领(handler 已随边侧断开退场):取一次
                # 清 log 标记,否则循环退出时刷 "exception was never retrieved"
                future.exception()

        try:
            loop.call_soon_threadsafe(_fail_pending)
        except RuntimeError:
            # 循环已停:无人再等这些 future,丢弃即可
            logger.info(
                "[Lwd][ctrl] close: loop already stopped, drop %d waiters",
                len(alive),
            )

    async def probe(
        self, manifest: LwdPrefixManifest, timeout: float = 20.0
    ) -> LwdProbeResult:
        """发探针命令并等结果;超时抛 TimeoutError(由 handler 转 503)。

        **同时**登记 probe 与 usage 两个 future:usage 由云侧结算线程随时
        推来,晚注册就会丢(结算可能早于 handler 走到 wait_usage)。等待者
        先注册再发命令,事件不会落在注册之前。

        超时口径与边侧对齐:边侧 probe 看门狗在 ``lwd_coordination.probe_timeout``
        +5 才放弃,本上限由调用方按同一配置减余量派生,保证**云先作答**(200 或
        503)而不是边先弃——负载期引擎主循环步长被数据面拉长时,probe 命令可能
        躺到超时才被消费,那种情况下结果只用于观测。

        同 req_id 重发是正常路径(同一 id 被重新 add_request:客户端重试、
        abort 后同 id 重投。引擎侧的 id 由 ``X-Request-Id`` 头决定,缺省才用
        random uuid——见 entrypoints/openai/engine/serving.py:387-396,所以
        重试时重放同一个头就会复用同一 id,这条路径不是理论上的。边侧 nonce 与
        协调器 ``probe()`` 的"同键重复 probe(边侧重试)释放旧 pin"都是为它
        准备的),故重复不拒绝而是**取代**:旧 usage future 就地给 None(它那笔
        的用量归新流,旧流以"无用量"收口),旧 probe future 置异常(旧 handler
        转 503;此时 200 尚未发出,无计费可丢)。参考分支的 ``edge_client`` 是
        "重复即拒",那是对着它自己 ``probe.block_size`` 严格相等校验的形态写的
        (它不存在同 id 二次探测);照搬到本支会把重试路径打断。

        云内 dp>1 走 ``_probe_fanout``(逐引擎只读求命中 → 放置 → 赢家落预留),
        应答经 ``_fanout_waiters`` 收集,不走 future 命名空间。"""
        request_id = manifest.request_id
        logger.debug("[Lwd][ctrl] bridge probe req=%s", request_id)
        if len(self._command_queues) > 1:
            # 云内 DP:扇出逐引擎只读求命中 → 放置 → 赢家引擎落预留。
            # usage future 仍须先行注册(A4):结算方(赢家引擎的 finish)
            # 按 request_id 对 _usage_futures 结账,wait_usage 依赖它在
            # 位——此前扇出路径先 return、future 从未注册 → dp>1 的
            # wait_usage 恒 None(每请求 SSE 无 usage 帧、网关计费为零)。
            # 重复 probe 取代语义同单引擎路径:旧流以「无用量」收口。
            loop = asyncio.get_running_loop()
            self._loop = loop
            with self._lock:
                stale_usage = self._usage_futures.pop(request_id, None)
                self._usage_futures[request_id] = loop.create_future()
            if stale_usage is not None and not stale_usage.done():
                stale_usage.set_result(None)
            try:
                return await self._probe_fanout(manifest, timeout)
            except BaseException:
                # 扇出失败:摘除已注册的 usage future(此时 wait_usage
                # 不会到达,留着即泄漏)并原样上抛
                with self._lock:
                    self._usage_futures.pop(request_id, None)
                raise
        loop = asyncio.get_running_loop()
        self._loop = loop
        with self._lock:
            stale_probe = self._probe_futures.pop(request_id, None)
            stale_usage = self._usage_futures.pop(request_id, None)
            probe_future: asyncio.Future = loop.create_future()
            usage_future: asyncio.Future = loop.create_future()
            self._probe_futures[request_id] = probe_future
            self._usage_futures[request_id] = usage_future
        if stale_probe is not None or stale_usage is not None:
            logger.warning(
                "[Lwd][ctrl] duplicate control request req=%s: superseding "
                "the previous registration", request_id,
            )
            if stale_usage is not None and not stale_usage.done():
                stale_usage.set_result(None)
            if stale_probe is not None and not stale_probe.done():
                stale_probe.set_exception(RuntimeError(
                    f"control request {request_id} superseded by a newer probe"))
        # [Lwd][perf] I8a:t_cmd = 命令投递时刻(父进程 CLOCK_MONOTONIC)。
        # 云父/子进程同机,单调钟同源可比;跨节点不可比,别拿边/云时钟差读。
        self._command_queue.put({
            "type": "probe",
            "manifest": manifest,
            "t_cmd": time.monotonic(),
        })
        try:
            return await asyncio.wait_for(probe_future, timeout)
        except BaseException:
            # 探针没成(超时/取消/云侧报错):这笔控制请求到此为止,usage
            # future 一并摘除——handler 不会再等 usage,留着就是无人认领的
            # 字典项(成功路径由 _deliver_event 摘)
            with self._lock:
                if self._probe_futures.get(request_id) is probe_future:
                    self._probe_futures.pop(request_id, None)
                if self._usage_futures.get(request_id) is usage_future:
                    self._usage_futures.pop(request_id, None)
            raise

    def release(self, request_id: str) -> None:
        """摘除该控制请求的登记(幂等)。

        SSE 生成器结束时必须调:客户端在启动消息之后就断开的话,
        ``wait_usage`` 从未被调用,字典项只能等进程 ``close()`` 收——那是
        兜底,不是正常路径。迟到的结算由此走孤儿分支丢弃。"""
        with self._lock:
            self._probe_futures.pop(request_id, None)
            self._usage_futures.pop(request_id, None)

    async def _probe_fanout(
        self, manifest: LwdPrefixManifest, timeout: float
    ) -> LwdProbeResult:
        """云内 DP probe 扇出与放置(仅 dp>1 路径)。

        两段式:① 只读求命中命令(reserve=False)广播到全部引擎的
        Processor,逐引擎聚合命中;② 按 (命中长度降序, 并列轮转) 选定
        赢家引擎,只向赢家发正式 probe(reserve=True)落预留——跨引擎
        预留是虚假命中,预留必须与后续数据面同引擎。响应命中值取赢家
        引擎的命中,engine_index=赢家。

        应答经共享 event_queue 上行,命令携带 engine_index/reply_key
        由引擎侧原样回显(见 Processor.poll),Bridge 按 reply_key 分发
        到本方法的收集队列;单引擎路径的队列协议(无新增字段)逐值不变。
        """
        n = len(self._command_queues)
        probe_key = f"{manifest.request_id}#probe"
        waiter: queue.Queue = queue.Queue(maxsize=n)
        with self._lock:
            self._fanout_waiters[probe_key] = waiter
        for idx, cmd_queue in enumerate(self._command_queues):
            cmd_queue.put({
                "type": "probe",
                "manifest": manifest,
                "reserve": False,
                "engine_index": idx,
                "reply_key": probe_key,
            })
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        results: dict[int, LwdProbeResult] = {}
        try:
            while len(results) < n:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    ok, eng_idx, payload = await asyncio.wait_for(
                        loop.run_in_executor(None, waiter.get), remaining
                    )
                except asyncio.TimeoutError:
                    break
                if ok:
                    results[eng_idx] = payload
                else:
                    logger.warning(
                        "[Lwd][ctrl] fanout probe failed req=%s engine=%s: %s",
                        manifest.request_id, eng_idx, payload,
                    )
        finally:
            with self._lock:
                self._fanout_waiters.pop(probe_key, None)
            # 毒丸:超时退出后 run_in_executor 的线程仍阻塞在
            # waiter.get;补一条让它返回释放(每超时一次否则漏一条
            # executor 线程,死引擎场景下单调增长)
            with contextlib.suppress(Exception):
                waiter.put_nowait((False, -1, "poison"))
        missing = [i for i in range(n) if i not in results]
        if missing:
            # 超时/失败引擎按 0 命中参与放置(不阻断其余引擎);
            # 全灭时才整体超时失败
            logger.warning(
                "[Lwd][ctrl] fanout probe partial req=%s missing=%s "
                "(treated as zero hit)",
                manifest.request_id, missing,
            )
            if not results:
                raise TimeoutError(
                    f"prefix probe fanout timed out after {timeout}s")
        # 放置:argmax(命中 token),并列按进程内单调计数器轮转。
        # 候选只含已应答引擎:missing(死/慢)引擎按 0 命中参与 hits
        # 展示,但绝不入候选——冷缓存全场 0 命中时若把它们算进并列,
        # 轮转会把请求钉到死引擎上(UP 通道楔死)。
        hits = [
            results[i].hit_tokens if i in results else 0 for i in range(n)
        ]
        top = max(hits[i] for i in results) if results else 0
        candidates = [i for i in results if hits[i] == top]
        if len(candidates) > 1:
            winner = candidates[self._rr_counter % len(candidates)]
            self._rr_counter += 1
        else:
            winner = candidates[0]
        logger.info(
            "[Lwd][ctrl] fanout placement req=%s hits=%s winner=%d",
            manifest.request_id, hits, winner,
        )
        # 预留只落赢家引擎:复用 probe 命令的预留语义(reserve=True)
        reserve_key = f"{manifest.request_id}#reserve"
        rsv_waiter: queue.Queue = queue.Queue(maxsize=1)
        with self._lock:
            self._fanout_waiters[reserve_key] = rsv_waiter
        self._command_queues[winner].put({
            "type": "probe",
            "manifest": manifest,
            "reserve": True,
            "engine_index": winner,
            "reply_key": reserve_key,
        })
        try:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"prefix reserve timed out after {timeout}s")
            ok, _, payload = await asyncio.wait_for(
                loop.run_in_executor(None, rsv_waiter.get), remaining
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"prefix reserve timed out after {timeout}s") from None
        finally:
            with self._lock:
                self._fanout_waiters.pop(reserve_key, None)
            with contextlib.suppress(Exception):
                rsv_waiter.put_nowait((False, -1, "poison"))
        if not ok:
            raise RuntimeError(payload or "reserve failed")
        return dataclass_replace(payload, engine_index=winner)

    async def wait_usage(
        self, request_id: str, timeout: float = _USAGE_WAIT_TIMEOUT
    ) -> LwdUsage | None:
        """等推理完结的 usage 事件;超时或无此控制请求返回 None(handler 收流
        但不发用量帧)。future 已在 ``probe()`` 里登记,故结算先到也不丢。"""
        with self._lock:
            future = self._usage_futures.get(request_id)
        if future is None:
            logger.warning(
                "[Lwd][ctrl] usage wait without control request req=%s",
                request_id,
            )
            return None
        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            logger.warning("[Lwd][ctrl] usage wait timeout req=%s", request_id)
            return None
        finally:
            with self._lock:
                # 只摘自己那条:同 req_id 若已被新探针取代,别动新登记
                if self._usage_futures.get(request_id) is future:
                    self._usage_futures.pop(request_id, None)

    def _dispatch_events(self) -> None:
        """单常驻线程:搬事件进事件循环,自己不碰 future(线程亲和性归循环)。"""
        while not self._closed:
            try:
                event = self._event_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            kind = event.get("type")
            request_id = event.get("request_id")
            with self._lock:
                table = (self._probe_futures if kind == "probe"
                         else self._usage_futures)
                registered = request_id in table
                loop = self._loop
            # 每事件一条:降噪到 debug(与本支热点路径日志口径一致)
            logger.debug(
                "[Lwd][ctrl] dispatch got event type=%s req=%s ok=%s engine=%s",
                kind, request_id, event.get("ok"), event.get("engine_index"),
            )
            # 扇出/预留应答(云内 DP):按 reply_key 找收集队列;迟到应答
            # (等待已结束)无 waiter,丢弃无害。这些事件携带 engine_index,
            # 不走 future 结算路径。
            if kind == "probe" and "engine_index" in event:
                with self._lock:
                    waiter = self._fanout_waiters.get(request_id)
                if waiter is not None:
                    try:
                        waiter.put_nowait((
                            bool(event.get("ok")),
                            int(event.get("engine_index", 0)),
                            event.get("result")
                            if event.get("ok") else event.get("error"),
                        ))
                    except queue.Full:
                        pass
                continue
            if not registered or loop is None:
                # 等待者已超时/断开摘除(迟到结算),或事件早于首个 probe
                # (数据面请求不经探针也能完结):丢弃,不报错
                logger.warning(
                    "[Lwd][ctrl] orphan %s event req=%s (no waiter/loop)",
                    kind, request_id,
                )
                continue
            try:
                loop.call_soon_threadsafe(self._deliver_event, event)
            except RuntimeError:
                # 循环已关(uvicorn 停机与 atexit 竞态):此时没有任何协程还
                # 等得到结算,继续搬只会每笔刷一条 traceback ⇒ 收线程。
                # 与 close() 里的同款保护保持对称。
                logger.info(
                    "[Lwd][ctrl] dispatch stopping: loop closed (%d control "
                    "request(s) abandoned)",
                    len(self._probe_futures) + len(self._usage_futures),
                )
                return

    def _deliver_event(self, event: dict[str, Any]) -> None:
        """在事件循环线程上结算 future(唯一 set_result/set_exception 点)。"""
        kind = event.get("type")
        request_id = event.get("request_id")
        if kind == "probe":
            with self._lock:
                future = self._probe_futures.pop(request_id, None)
            if future is None or future.done():
                return  # 已超时/取消:边侧早已 fail-open,结果作废
            if event.get("ok"):
                future.set_result(event.get("result"))
            else:
                future.set_exception(
                    RuntimeError(event.get("error") or "probe failed"))
            return
        if kind == "usage":
            # 结算只带 request_id,不带"哪一代探针"的信息,所以这里只能投给
            # 当前登记的那条 future。这么做的依据不是"云侧会丢弃重复请求"——
            # 云侧 duplicate RangeNotify 抑制(见 lwd_cloud_engine 的
            # "duplicate RangeNotify suppressed" 日志锚点)只覆盖**门池
            # 窗口**(还没过门);
            # 过门之后再投同 rid 会走到原生 Scheduler.add_request
            # (vllm/v1/core/sched/scheduler.py:1823-1836):流式输入会话被折叠
            # 成一条 update,否则撞上 assert "duplicate request id"。
            # 也就是说"一个 rid 只有一个在飞尝试"是整栈的**前提**(边侧
            # nonce/abort 墓碑 + 协调器单预留覆写 + 门池窗口共同维持,并由该断言
            # 兜底),而不是云侧会替我们温和去重。推并发安全性请从这三处推,别把
            # "忽略"当全局保证。
            with self._lock:
                future = self._usage_futures.get(request_id)
            if future is None or future.done():
                logger.warning(
                    "[Lwd][ctrl] usage event arrived too late req=%s", request_id
                )
                return
            future.set_result(event.get("usage"))


class LwdCloudControlProcessor:
    """子进程侧:step 循环头部/WAKEUP 后非阻塞 poll,消费 probe 命令。

    命令来源是本地线程安全队列(唤醒线程已从 mp 命令队列搬入并敲
    过 WAKEUP):不再有 mp.Queue.get_nowait 的 feeder 竞态。"""

    def __init__(self, pending_queue: Any, event_queue: Any) -> None:
        self._pending_queue = pending_queue
        self._event_queue = event_queue

    def poll(self, coordinator: Any) -> int:
        """消费待处理命令,返回本轮消费条数([Lwd][perf] I4)。

        条数 >1 即"探针在等云引擎步头被批量消费"——探针服务能力受云
        侧步频约束的直接证据。"""
        consumed = 0
        while True:
            try:
                command = self._pending_queue.get_nowait()
            except queue.Empty:
                return consumed
            consumed += 1
            logger.debug("[Lwd][ctrl] poll consumed command type=%s", command.get("type"))
            if command.get("type") == "probe":
                manifest: LwdPrefixManifest = command["manifest"]
                # 云内 DP 协议字段(缺省=旧版单队列语义,逐值不变):
                #   reserve=False 只读求命中(扇出采集,不种预留);
                #   reply_key     应答键改写(扇出/预留应答按它分发);
                #   engine_index  应答回显(Bridge 识别来源引擎)。
                reserve = command.get("reserve", True)
                reply_key = command.get("reply_key", manifest.request_id)
                # [Lwd][perf] I8a:cmd_wait_ms = 命令从父进程投递到被主循环
                # 取走的排队时长(含 mp 队列 + 唤醒线程 + 等步头)。它若接近
                # 边侧看到的 http_ms,说明卡在"主循环没空取命令";它若≈0,
                # 那 456ms 花在结果回传/结算那一段。
                t_take = time.monotonic()
                t_cmd = command.get("t_cmd")
                cmd_wait_ms = (
                    (t_take - t_cmd) * 1000.0 if t_cmd is not None else -1.0
                )
                try:
                    if reserve:
                        result = coordinator.probe(manifest)
                    else:
                        result = coordinator.peek(manifest)
                    logger.info(
                        "[Lwd][perf][cloud] probe_consumed req=%s "
                        "cmd_wait_ms=%.1f probe_ms=%.1f",
                        manifest.request_id,
                        cmd_wait_ms,
                        (time.monotonic() - t_take) * 1000.0,
                    )
                    event = {
                        "type": "probe",
                        "request_id": reply_key,
                        "ok": True,
                        "result": result,
                    }
                except Exception as exc:  # noqa: BLE001
                    logger.exception("[Lwd][ctrl] probe failed")
                    event = {
                        "type": "probe",
                        "request_id": reply_key,
                        "ok": False,
                        "error": str(exc),
                    }
                if "engine_index" in command:
                    event["engine_index"] = command["engine_index"]
                self._event_queue.put(event)


def _negotiate_lwd_protocol_headers(
    headers: Mapping[str, str],
    expected_mm_abi: str | None,
    *,
    enforce_mm_abi_match: bool,
) -> str | None:
    """返回 v2 请求校验通过的 MM-ABI 头值;v1 请求返回 None。

    v1 保持原语义;v1 却带 MM-ABI、v2 缺或畸形 MM-ABI 一律拒绝——两套协议
    栈不得互相降级(否则图片链以文本口径入账,命中会静默错位)。
    ``enforce_mm_abi_match`` 缺省打开:边云必须比较各自本地计算的处理器
    指纹,云侧响应也只发布云侧身份,绝不回显边侧值。"""
    normalized = {k.lower(): v for k, v in headers.items()}
    protocol = normalized.get(HEADER_PROTOCOL.lower())
    mm_abi = normalized.get(HEADER_MM_ABI.lower())
    if protocol == PROTOCOL_VERSION_MM:
        value = validate_mm_abi_header(mm_abi)
        if expected_mm_abi is None:
            raise ValueError(
                f"{PROTOCOL_VERSION_MM} requires a locally computed "
                f"{HEADER_MM_ABI} fingerprint on the cloud"
            )
        if enforce_mm_abi_match and value != expected_mm_abi:
            raise ValueError(
                f"{HEADER_MM_ABI} fingerprint does not match this cloud"
            )
        # Honest negotiation: advertise the cloud's own identity. Echoing the
        # edge value would make an unchecked negotiation look successful.
        return expected_mm_abi
    if mm_abi is not None:
        raise ValueError(f"{HEADER_MM_ABI} requires protocol {PROTOCOL_VERSION_MM}")
    return None


def _sse_data(obj: Any) -> str:
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"


def _build_chunk(
    request_id: str, model: str, choices: list, usage: dict | None = None
) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def create_lwd_cloud_control_app(
    bridge: LwdCloudControlBridge,
    processor_fingerprint: bytes | None = None,
    *,
    enforce_mm_abi_match: bool = True,
    probe_timeout: float = 25.0,
) -> Any:
    """FastAPI 控制面应用;成功响应走 OpenAI 兼容 SSE。

    错误路径仍用 JSONResponse(网关对非 2xx 的 JSON 错误体有标准处理)。
    ``processor_fingerprint`` 为云侧本地处理器指纹,仅在
    ``enforce_mm_abi_match`` 打开时参与 v2 仲裁;``probe_timeout`` 是本进程
    等探针结果的硬上限(由边侧 ``probe_timeout`` 减余量派生,见常量注释)。"""
    app = FastAPI()
    expected_mm_abi = (
        mm_abi_header_value(processor_fingerprint)
        if processor_fingerprint is not None
        else None
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """网关主动健康检查/探活口(不参与边云协商)。"""
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        try:
            headers = dict(request.headers)
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": f"bad request: {exc}"}, status_code=400)
        try:
            mm_abi_header = _negotiate_lwd_protocol_headers(
                headers, expected_mm_abi,
                enforce_mm_abi_match=enforce_mm_abi_match,
            )
            parse_headers: Mapping[str, str] = headers
            if mm_abi_header is not None:
                # v2 的 manifest 线格式与 v1 同构:协议标记与 ABI 校验通过后,
                # 按 v1 解析器读消息体(解析器只认 v1 标记)
                parse_headers = {
                    k: v
                    for k, v in headers.items()
                    if k.lower() != HEADER_PROTOCOL.lower()
                }
                parse_headers[HEADER_PROTOCOL] = PROTOCOL_VERSION
            manifest = LwdPrefixManifest.from_openai_request(
                parse_headers, body
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        # 多边入口命名空间包裹:按自报 X-Edge-Cloud-Edge-Id 把 manifest.request_id
        # 包成云内唯一 id("e{edge_id}-{req_id}"),与云引擎 _lwd_dispatch 的同款
        # wrap 对齐,使协调器预留键 == 数据面请求键。
        # 注意:ASGI/Starlette 给的 header key **全小写**,此处必须按小写取——
        # 否则恒回退 "0"(实测:探针恒包成 e0-,而数据面按信封 identity 包成
        # e{真边},预留键与请求键错位 → claim/前缀校验全部失效)。
        headers_lc = {k.lower(): v for k, v in headers.items()}
        try:
            edge_id = int(headers_lc.get(HEADER_EDGE_ID.lower(), "0"))
        except (TypeError, ValueError) as exc:
            return JSONResponse(
                {"error": f"invalid {HEADER_EDGE_ID}: {exc}"}, status_code=400
            )
        if edge_id < 0:
            return JSONResponse({"error": "negative edge_id"}, status_code=400)
        manifest = dataclass_replace(
            manifest, request_id=wrap_req_id(edge_id, manifest.request_id)
        )

        try:
            logger.debug(
                "[Lwd][ctrl] handler entering probe req=%s edge=%d",
                manifest.request_id, edge_id,
            )
            result = await bridge.probe(manifest, timeout=probe_timeout)
            logger.debug(
                "[Lwd][ctrl] handler probe resolved req=%s hit=%d",
                manifest.request_id, result.hit_tokens,
            )
        except TimeoutError as exc:
            # 本上限已由边侧 probe_timeout 减余量派生,正常情况下云是先作答
            # (200/503)而非边先弃;走到这里说明引擎主循环连派生窗口内都没消费
            # probe 命令(负载期步长被数据面拉长)。此时边侧多半已按 fail-open
            # 继续,本结果只作观测——一行 warning 足够,不打异常栈(频发会刷屏)。
            # 真正要关注的是出现频率:常态化即主循环响应控制面劣化,查步长。
            logger.warning(
                "[Lwd][ctrl] probe timeout req=%s edge=%d: %s "
                "(engine loop did not answer in time; edge already "
                "failed-open)",
                manifest.request_id, edge_id, exc,
            )
            return JSONResponse({"error": str(exc)}, status_code=503)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[Lwd][ctrl] handler probe raised")
            return JSONResponse({"error": str(exc)}, status_code=503)

        # Probe 结果走 SSE delta.content(网关 DTO 转换不丢),响应头保留
        # 供直连调试/网关观测。usage 在同一条 SSE 流末 chunk,网关据此计费。
        # 回显给边侧的是剥掉 wrap 前缀的原始 id(边侧校验一致性)
        display_request_id = result.request_id
        if display_request_id.startswith(f"e{edge_id}-"):
            display_request_id = display_request_id[len(f"e{edge_id}-"):]
        model = body.get("model", "edge-cloud-prefix")

        # 回显按协商结果:边侧据此校验云确实按本端协议应答(不得静默降 v1)
        protocol = (
            PROTOCOL_VERSION_MM if mm_abi_header is not None else PROTOCOL_VERSION
        )
        control_message = result.to_control_message(protocol, mm_abi_header)
        # 控制消息里的 request_id 用回显形式
        control_message["request_id"] = display_request_id

        async def _events_inner():
            # 1. Probe 控制消息(delta.content 携带,网关 DTO 转换不丢)
            yield _sse_data(_build_chunk(
                display_request_id,
                model,
                [{"index": 0, "delta": {
                    "role": "assistant",
                    "content": json.dumps(control_message, separators=(",", ":")),
                }, "finish_reason": None}],
            ))
            # 2. 空 delta 帧:冲刷网关末 chunk 缓冲(NewAPI 等)
            yield _sse_data(_build_chunk(
                display_request_id,
                model,
                [{"index": 0, "delta": {}, "finish_reason": None}],
            ))
            # 3. 等 usage(推理完结后由引擎推入)
            usage: LwdUsage | None = None
            usage_task = asyncio.create_task(
                bridge.wait_usage(manifest.request_id))
            try:
                while not usage_task.done():
                    done, _ = await asyncio.wait(
                        {usage_task}, timeout=_USAGE_HEARTBEAT_INTERVAL)
                    if not done:
                        yield ": edge-cloud-usage-pending\n\n"
                usage = usage_task.result()
            except asyncio.CancelledError:
                if not usage_task.done():
                    usage_task.cancel()
                await asyncio.gather(usage_task, return_exceptions=True)
                raise
            # 4. usage 末 chunk + [DONE]
            if usage is None:
                # 拿不到真实用量就**不发** usage 帧:伪造 0 会让网关按零 Token
                # 出账(quota=0 的白嫖账),而协议口径是 usage 恒为真实推理量。
                # 仍以 comment + [DONE] 正常收流——半途断流会触发网关侧
                # client_gone/错误统计,比"这笔没有上游用量"更糟。
                logger.error(
                    "[Lwd][ctrl] usage unavailable req=%s: ending stream "
                    "without usage frame (never bill fabricated zeros)",
                    manifest.request_id,
                )
                yield ": edge-cloud-usage-unavailable\n\n"
                yield "data: [DONE]\n\n"
                return
            yield _sse_data(_build_chunk(
                display_request_id,
                model,
                [],
                usage=usage.to_openai_dict(),
            ))
            yield "data: [DONE]\n\n"

        async def events():
            try:
                async for frame in _events_inner():
                    yield frame
            finally:
                # 正常收流、边侧提前断开、生成器被取消都要摘登记:否则
                # usage future 一直悬在字典里,只能等进程 close() 收口
                bridge.release(manifest.request_id)

        headers_out = {
            **result.to_headers(),
            # 协调器持的是 wrap 后的云内 id,响应头必须回剥壳形式:边侧发的
            # 就是原始 id,否则按 req id 跨边/云对账时两条口径不一致
            HEADER_REQUEST_ID: display_request_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        if mm_abi_header is not None:
            headers_out[HEADER_PROTOCOL] = PROTOCOL_VERSION_MM
            headers_out[HEADER_MM_ABI] = mm_abi_header
        return StreamingResponse(
            events(), media_type="text/event-stream", headers=headers_out
        )

    return app


def start_lwd_cloud_control_server(
    vllm_config: Any,
) -> tuple[list[Any] | None, Any | None]:
    """父进程侧启动 HTTP 控制服务 + 队列桥(仅云侧且启用协调时)。

    返回 ``(command_queues, event_queue)``,交由子进程 EngineCore 经
    ``CoreEngineProcManager`` 透传;未启用/边侧返回 ``(None, None)``。
    command_queues 为 per-engine 列表:云内 dp>1 时长度=dp_size,按全局
    dp_rank 逐引擎分发(probe 扇出按引擎寻址);dp=1 为单元素列表,语义
    与旧版单队列一致。event_queue 保持全引擎共享(事件自带 engine_index)。

    uvicorn 与桥均在后台线程运行,须在 spawn EngineCoreProc 之前调用。"""
    lwd_config = getattr(vllm_config, "lwd_config", None)
    coord_cfg = getattr(lwd_config, "coordination", None)
    if coord_cfg is None or not coord_cfg.enabled:
        return None, None
    if lwd_config.role != "cloud":
        return None, None

    from vllm.utils.system_utils import get_mp_context

    import uvicorn

    mp_ctx = get_mp_context()
    # 云内 DP 数:每引擎进程独立 KV 池/前缀缓存,probe 需逐引擎扇出
    dp_size = max(int(vllm_config.parallel_config.data_parallel_size), 1)
    command_queues = [mp_ctx.Queue() for _ in range(dp_size)]
    event_queue = mp_ctx.Queue()
    bridge = LwdCloudControlBridge(command_queues, event_queue)
    bridge.start()
    # uvicorn 跑在守护线程、没有 shutdown 通道:接 atexit,让 dispatch 线程
    # 正常收口并把仍在等的 future 置异常(否则退出路径只能靠 daemon 硬杀)
    atexit.register(bridge.close)
    # 云侧本地指纹与边侧 _lwd_processor_fingerprint 同源(mm_identity 单点
    # 实现);仅在 enforce_mm_abi_match 打开时参与 v2 仲裁。
    # 探针上限取边侧 probe_timeout 减余量:保证云先作答(200/503)。
    cloud_probe_timeout = max(
        coord_cfg.probe_timeout - _CLOUD_PROBE_HEADROOM_S, 1.0
    )
    app = create_lwd_cloud_control_app(
        bridge,
        compute_processor_fingerprint(vllm_config.model_config),
        enforce_mm_abi_match=coord_cfg.enforce_mm_abi_match,
        probe_timeout=cloud_probe_timeout,
    )
    threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={
            "host": coord_cfg.listen_host,
            "port": coord_cfg.listen_port,
            "log_level": "warning",
        },
        name="lwd-cloud-ctrl",
        daemon=True,
    ).start()
    logger.info(
        "[Lwd] cloud control HTTP server listening on %s:%s "
        "(cloud dp_size=%d, %d engine command queue(s), "
        "probe_timeout=%.1fs enforce_mm_abi_match=%s)",
        coord_cfg.listen_host, coord_cfg.listen_port,
        dp_size, len(command_queues),
        cloud_probe_timeout, coord_cfg.enforce_mm_abi_match,
    )
    return command_queues, event_queue
