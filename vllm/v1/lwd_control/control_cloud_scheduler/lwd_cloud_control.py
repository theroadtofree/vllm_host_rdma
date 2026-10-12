# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""云侧前缀协商 HTTP 控制面(SSE 网关形态)。

从参考分支 ``edge_cloud/cloud_control.py`` 演化:父进程 FastAPI 应用 + 桥
(LwdCloudControlBridge),子进程处理器(LwdCloudControlProcessor,poll 非阻塞),
经两条 multiprocessing 队列对接。宿主是 ``LwdCloudEngineCore``。

控制面承载 probe + usage 两个事件,走 **OpenAI 兼容 SSE**:
  1. Probe 结果放 ``delta.content`` 的 JSON 控制消息(网关 DTO 转换不丢);
  2. 紧跟空 delta 帧,冲刷网关末 chunk 缓冲(NewAPI 等);
  3. 期间每 10s 发 SSE comment 心跳,防网关上游空闲超时;
  4. 推理完结后发标准 OpenAI usage chunk + ``[DONE]``,网关据此计费。

响应头仍保留 ``X-Edge-Cloud-*`` 供直连调试/网关观测,不参与协商。
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from dataclasses import replace as dataclass_replace
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.logger import init_logger
from vllm.v1.lwd_control.control_communication.lwd_id_adapter import wrap_req_id
from vllm.v1.lwd_control.control_communication.lwd_prefix import (
    HEADER_EDGE_ID,
    LwdPrefixManifest,
    LwdProbeResult,
    LwdUsage,
)

logger = init_logger(__name__)

# SSE 心跳间隔(秒):须小于网关 stream_idle_timeout,留足余量
_USAGE_HEARTBEAT_INTERVAL = 10.0
# usage 等待总超时(秒):覆盖完整推理时长,超时后发空 usage 收流
_USAGE_WAIT_TIMEOUT = 600.0


class LwdCloudControlBridge:
    """父进程侧:把 HTTP 协程与子进程的 probe/usage 事件用队列对接。

    result 回传走 per-request 一次性 queue.Queue(dispatch 线程 put、
    handler 协程经 run_in_executor 阻塞 get):既消掉字典轮询的
    20ms 级延迟,又刻意避开「跨线程 asyncio.Future +
    call_soon_threadsafe」——后者在 uvicorn 挂线程 + uvloop 环境下
    会出现 future 不 resolve、响应头迟迟不发的诡异卡死。
    queue.Queue.get 在线程池里阻塞,与事件循环零交互,简单可观测。
    """

    def __init__(self, command_queue: Any, event_queue: Any) -> None:
        self._command_queue = command_queue
        self._event_queue = event_queue
        self._waiters: dict[str, queue.Queue] = {}
        self._usage_waiters: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_events, name="lwd-ctrl-dispatch", daemon=True
        )

    def start(self) -> None:
        self._dispatch_thread.start()

    def close(self) -> None:
        self._closed = True

    async def probe(
        self, manifest: LwdPrefixManifest, timeout: float = 60.0
    ) -> LwdProbeResult:
        """发探针命令并阻塞等结果;超时抛 TimeoutError(由 handler 转 503)。

        等待者先注册再发命令(无丢事件竞态);结果经一次性队列回传,
        run_in_executor 阻塞等待不碰事件循环(见类 docstring)。"""
        logger.info("[Lwd][ctrl] bridge probe req=%s", manifest.request_id)
        waiter: queue.Queue = queue.Queue(maxsize=1)
        with self._lock:
            self._waiters[manifest.request_id] = waiter
        self._command_queue.put({"type": "probe", "manifest": manifest})
        loop = asyncio.get_running_loop()
        try:
            ok, payload = await asyncio.wait_for(
                loop.run_in_executor(None, waiter.get), timeout
            )
        except asyncio.TimeoutError:
            with self._lock:
                self._waiters.pop(manifest.request_id, None)
            try:
                waiter.put_nowait((False, "probe timeout"))
            except queue.Full:
                pass
            raise TimeoutError(f"prefix probe timed out after {timeout}s")
        if ok:
            return payload
        raise RuntimeError(payload or "probe failed")

    async def wait_usage(
        self, request_id: str, timeout: float = _USAGE_WAIT_TIMEOUT
    ) -> LwdUsage | None:
        """等推理完结的 usage 事件;超时返回 None。

        与 probe 同款一次性队列;SSE handler 在 probe 发出后调用,
        流保持打开直到 usage 到位或超时。"""
        waiter: queue.Queue = queue.Queue(maxsize=1)
        with self._lock:
            self._usage_waiters[request_id] = waiter
        loop = asyncio.get_running_loop()
        try:
            payload = await asyncio.wait_for(
                loop.run_in_executor(None, waiter.get), timeout
            )
            return payload
        except asyncio.TimeoutError:
            logger.warning(
                "[Lwd][ctrl] usage wait timeout req=%s", request_id)
            return None
        finally:
            with self._lock:
                self._usage_waiters.pop(request_id, None)

    def publish_usage(self, request_id: str, usage: LwdUsage) -> None:
        """子进程/引擎侧推送 usage 事件(dispatch 线程消费后投给 waiter)。

        由 ``LwdCloudEngineCore._lwd_publish_finished_usage`` 在请求完结
        结算后调用;事件经 mp event_queue 到父进程,再由 dispatch 线程
        唤醒等待中的 SSE handler。"""
        try:
            self._event_queue.put({
                "type": "usage",
                "request_id": request_id,
                "usage": usage,
            })
        except Exception:  # noqa: BLE001
            logger.exception(
                "[Lwd][ctrl] publish_usage failed req=%s", request_id)

    def _dispatch_events(self) -> None:
        while not self._closed:
            try:
                event = self._event_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            logger.info(
                "[Lwd][ctrl] dispatch got event type=%s req=%s ok=%s",
                event.get("type"), event.get("request_id"), event.get("ok"),
            )
            kind = event.get("type")
            if kind == "probe":
                with self._lock:
                    waiter = self._waiters.pop(event["request_id"], None)
                if waiter is not None:
                    try:
                        waiter.put_nowait((
                            bool(event.get("ok")),
                            event.get("result")
                            if event.get("ok") else event.get("error"),
                        ))
                    except queue.Full:
                        pass
            elif kind == "usage":
                with self._lock:
                    waiter = self._usage_waiters.pop(event["request_id"], None)
                if waiter is not None:
                    try:
                        waiter.put_nowait(event.get("usage"))
                    except queue.Full:
                        pass


class LwdCloudControlProcessor:
    """子进程侧:step 循环头部/WAKEUP 后非阻塞 poll,消费 probe 命令。

    命令来源是本地线程安全队列(唤醒线程已从 mp 命令队列搬入并敲
    过 WAKEUP):不再有 mp.Queue.get_nowait 的 feeder 竞态。"""

    def __init__(self, pending_queue: Any, event_queue: Any) -> None:
        self._pending_queue = pending_queue
        self._event_queue = event_queue

    def poll(self, coordinator: Any) -> None:
        while True:
            try:
                command = self._pending_queue.get_nowait()
            except queue.Empty:
                return
            logger.info("[Lwd][ctrl] poll consumed command type=%s", command.get("type"))
            if command.get("type") == "probe":
                manifest: LwdPrefixManifest = command["manifest"]
                try:
                    result = coordinator.probe(manifest)
                    self._event_queue.put({
                        "type": "probe",
                        "request_id": manifest.request_id,
                        "ok": True,
                        "result": result,
                    })
                except Exception as exc:  # noqa: BLE001
                    logger.exception("[Lwd][ctrl] probe failed")
                    self._event_queue.put({
                        "type": "probe",
                        "request_id": manifest.request_id,
                        "ok": False,
                        "error": str(exc),
                    })


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
) -> Any:
    """FastAPI 控制面应用;成功响应走 OpenAI 兼容 SSE。

    错误路径仍用 JSONResponse(网关对非 2xx 的 JSON 错误体有标准处理)。
    """
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        try:
            headers = dict(request.headers)
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": f"bad request: {exc}"}, status_code=400)
        try:
            manifest = LwdPrefixManifest.from_openai_request(headers, body)
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
        # 成员校验(尽力而为):registry 若已在本进程装载则校验 edge_id 属于
        # 实边集合;未装载(单边/1E1C)跳过。
        try:
            from vllm.v1.lwd_control.control_communication.lwd_role_registry import (
                get_role_registry,
            )

            registry = get_role_registry()
            if registry is not None and edge_id not in registry.edge_ids:
                return JSONResponse(
                    {"error": f"unknown edge_id {edge_id}"}, status_code=400
                )
        except Exception:  # noqa: BLE001  registry 未就绪等,不阻断主链路
            pass
        manifest = dataclass_replace(
            manifest, request_id=wrap_req_id(edge_id, manifest.request_id)
        )

        try:
            logger.info(
                "[Lwd][ctrl] handler entering probe req=%s edge=%d",
                manifest.request_id, edge_id,
            )
            result = await bridge.probe(manifest)
            logger.info(
                "[Lwd][ctrl] handler probe resolved req=%s hit=%d",
                manifest.request_id, result.hit_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[Lwd][ctrl] handler probe raised")
            return JSONResponse({"error": str(exc)}, status_code=503)

        # Probe 结果走 SSE delta.content(网关 DTO 转换不丢),响应头保留
        # 供直连调试/网关观测。usage 在同一条 SSE 流末 chunk,网关据此计费。
        raw_request_id = manifest.request_id
        # 回显给边侧的是剥掉 wrap 前缀的原始 id(边侧校验一致性)
        display_request_id = result.request_id
        if display_request_id.startswith(f"e{edge_id}-"):
            display_request_id = display_request_id[len(f"e{edge_id}-"):]
        model = body.get("model", "edge-cloud-prefix")

        control_message = result.to_control_message()
        # 控制消息里的 request_id 用回显形式
        control_message["request_id"] = display_request_id

        async def events():
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
            usage_dict = (
                usage.to_openai_dict() if usage is not None else {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            )
            yield _sse_data(_build_chunk(
                display_request_id,
                model,
                [],
                usage=usage_dict,
            ))
            yield "data: [DONE]\n\n"

        headers_out = {
            **result.to_headers(),
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(
            events(), media_type="text/event-stream", headers=headers_out
        )

    return app


def start_lwd_cloud_control_server(vllm_config: Any) -> tuple[Any | None, Any | None]:
    """父进程侧启动 HTTP 控制服务 + 队列桥(仅云侧且启用协调时)。

    返回 ``(command_queue, event_queue)``,交由子进程 EngineCore 经
    ``CoreEngineProcManager`` 透传;未启用/边侧返回 ``(None, None)``。

    uvicorn 与桥均在后台线程运行,须在 spawn EngineCoreProc 之前调用。"""
    coord_cfg = getattr(vllm_config, "lwd_coordination", None)
    if coord_cfg is None or not coord_cfg.enabled:
        return None, None
    if vllm_config.lwd_config.is_edge:
        return None, None

    from vllm.utils.system_utils import get_mp_context

    import uvicorn

    mp_ctx = get_mp_context()
    command_queue = mp_ctx.Queue()
    event_queue = mp_ctx.Queue()
    bridge = LwdCloudControlBridge(command_queue, event_queue)
    bridge.start()
    app = create_lwd_cloud_control_app(bridge)
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
        "[Lwd] cloud control HTTP server listening on %s:%s",
        coord_cfg.listen_host, coord_cfg.listen_port,
    )
    return command_queue, event_queue
