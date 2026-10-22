# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""云侧控制面契约:SSE 五事件顺序、协议门禁、命名空间剥壳、用量缺席。

不调 HTTP 端口,直接取 FastAPI 注册的处理函数 + 鸭子类型的 Request 替身,
所以本机不需要 httpx/TestClient;桥用桩件替换,不牵子进程与 mp 队列。
判据逐条对应《边-网关-云的交互逻辑》:事件一启动消息(content 内)、事件二
空 delta、事件三 comment 心跳、事件四 choices:[] + 标准 usage、事件五 DONE,
以及"用量拿不到就不发用量帧"。
"""

import asyncio
import hashlib
import json
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from vllm.v1.lwd_control.control_cloud_scheduler import lwd_cloud_control as ctrl
from vllm.v1.lwd_control.control_communication.lwd_mm_identity import (
    compute_processor_fingerprint,
    mm_abi_header_value,
)
from vllm.v1.lwd_control.control_communication.lwd_prefix import (
    HEADER_EDGE_ID,
    HEADER_MM_ABI,
    HEADER_PROMPT_TOKENS,
    HEADER_PROTOCOL,
    HEADER_REQUEST_ID,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_MM,
    LwdPrefixHasher,
    LwdPrefixManifest,
    LwdProbeResult,
    LwdUsage,
)

pytestmark = pytest.mark.cpu_test

TENANT_KEY = b"tenant-a-secret-key-material"
BLOCK_SIZE = 4
PROMPT_TOKENS = 10
EDGE_ID = 3
REQUEST_ID = "req-001"
CLOUD_FINGERPRINT = compute_processor_fingerprint(
    SimpleNamespace(model="qwen3.5", mm_processor_kwargs=None)
)
CLOUD_MM_ABI = mm_abi_header_value(CLOUD_FINGERPRINT)
OTHER_MM_ABI = mm_abi_header_value(hashlib.sha256(b"other").digest())
USAGE = LwdUsage(prompt_tokens=1024, completion_tokens=128, cached_tokens=768)


class _FakeRequest:
    """够用的 Starlette Request 替身:处理函数只读 headers 与 json()。"""

    def __init__(self, headers: dict[str, str], body: bytes) -> None:
        self.headers = {k.lower(): v for k, v in headers.items()}
        self._body = body

    async def json(self):
        return json.loads(self._body)


class _StubBridge:
    """记录 probe 收到的 manifest,并按协调器口径回结果与 usage。"""

    def __init__(self, *, usage=USAGE, usage_delay: float = 0.0) -> None:
        self.usage = usage
        self.usage_delay = usage_delay
        self.probed: list = []
        self.probe_timeouts: list[float] = []
        self.waited: list[str] = []
        self.released: list[str] = []

    def release(self, request_id: str) -> None:
        self.released.append(request_id)

    async def probe(self, manifest, timeout: float = 60.0) -> LwdProbeResult:
        self.probed.append(manifest)
        self.probe_timeouts.append(timeout)
        hit_blocks = min(2, manifest.prompt_tokens // manifest.block_size)
        return LwdProbeResult(
            request_id=manifest.request_id,
            instance_id="cloud-0",
            block_size=manifest.block_size,
            hit_blocks=hit_blocks,
            hit_tokens=hit_blocks * manifest.block_size,
            alignment_tokens=512,
            cloud_id=1,
        )

    async def wait_usage(self, request_id: str, timeout: float = 600.0):
        self.waited.append(request_id)
        if self.usage_delay:
            await asyncio.sleep(self.usage_delay)
        return self.usage


def _wire(
    *,
    protocol: str = PROTOCOL_VERSION,
    mm_abi: str | None = None,
    edge_id: int = EDGE_ID,
    prompt_tokens: str | None = None,
) -> tuple[dict[str, str], bytes]:
    manifest = LwdPrefixHasher(TENANT_KEY, BLOCK_SIZE).build_manifest(
        REQUEST_ID, list(range(PROMPT_TOKENS))
    )
    headers = {
        **manifest.to_headers(),
        HEADER_EDGE_ID: str(edge_id),
        HEADER_PROMPT_TOKENS: str(
            manifest.prompt_tokens
            if prompt_tokens is None
            else prompt_tokens
        ),
    }
    if protocol != PROTOCOL_VERSION:
        headers[HEADER_PROTOCOL] = protocol
    if mm_abi is not None:
        headers[HEADER_MM_ABI] = mm_abi
    body = json.dumps(
        {
            "model": "qwen3.5",
            "messages": manifest.to_messages(),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    return headers, body


def _app(
    *,
    fingerprint: bytes | None = CLOUD_FINGERPRINT,
    enforce: bool = False,
    probe_timeout: float | None = None,
):
    bridge = _StubBridge()
    kwargs: dict = {"enforce_mm_abi_match": enforce}
    if probe_timeout is not None:
        kwargs["probe_timeout"] = probe_timeout
    app = ctrl.create_lwd_cloud_control_app(bridge, fingerprint, **kwargs)
    return app, bridge


def _handler(app, path: str, method: str):
    for route in app.routes:
        if route.path == path and method in getattr(route, "methods", ()):
            return route.endpoint
    raise AssertionError(f"route {method} {path} not registered")


def _classify(chunk) -> tuple[str, object]:
    text = (chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk).strip()
    if text.startswith(":"):
        return "comment", text.lstrip(":").strip()
    assert text.startswith("data:"), text
    payload = text[len("data:"):].strip()
    if payload == "[DONE]":
        return "done", None
    return "data", json.loads(payload)


def _stream(response) -> tuple[int, dict, list]:
    """跑完 SSE 生成器:返回 (状态码, 响应头, [(事件类型, 载荷), ...])。"""

    async def _collect():
        chunks = [c async for c in response.body_iterator]
        return [_classify(c) for c in chunks]

    return response.status_code, dict(response.headers), asyncio.run(_collect())


def _post(app, headers, body):
    request = _FakeRequest(headers, body)
    return asyncio.run(_handler(app, "/v1/chat/completions", "POST")(request))


def _control_payload(events) -> dict:
    """事件一的 content JSON(启动消息)。"""
    for kind, payload in events:
        if kind != "data":
            continue
        choices = payload.get("choices") or []
        if not choices:
            continue
        content = (choices[0].get("delta") or {}).get("content")
        if content:
            return json.loads(content)
    raise AssertionError("no probe content frame")


# --------------------------------------------------------------------- #
# SSE 事件顺序(文档「Cloud 按顺序发送 SSE 事件」)            #
# --------------------------------------------------------------------- #
def test_sse_event_order_matches_the_document() -> None:
    app, bridge = _app()
    headers, body = _wire()
    response = _post(app, headers, body)
    status, resp_headers, events = _stream(response)
    assert status == 200
    assert resp_headers["content-type"].startswith("text/event-stream")
    assert resp_headers["cache-control"] == "no-cache"
    assert resp_headers["x-accel-buffering"] == "no"

    kinds = [kind for kind, _ in events]
    assert kinds[0] == "data"
    assert kinds[1] == "data"
    assert kinds[-2] == "data"
    assert kinds[-1] == "done"

    probe = _control_payload(events)
    assert probe == {
        "type": "edge_cloud_probe",
        "protocol": PROTOCOL_VERSION,
        "request_id": REQUEST_ID,
        "instance_id": "cloud-0",
        "block_size": BLOCK_SIZE,
        "hit_blocks": 2,
        "hit_tokens": 2 * BLOCK_SIZE,
        "alignment_tokens": 512,
        "cloud_id": 1,
    }
    # 事件二:空 delta,不结束响应
    empty = events[1][1]
    assert empty["choices"] == [
        {"index": 0, "delta": {}, "finish_reason": None}
    ]
    # 事件四:choices 为空 + 标准 usage(含 cached_tokens 明细)
    usage_frame = events[-2][1]
    assert usage_frame["choices"] == []
    assert usage_frame["usage"] == {
        "prompt_tokens": 1024,
        "completion_tokens": 128,
        "total_tokens": 1152,
        "prompt_tokens_details": {"cached_tokens": 768},
    }
    assert events[-1] == ("done", None)
    # 云内键是包裹后的 id,边侧看到的 content 是原始 id
    assert bridge.probed[0].request_id == f"e{EDGE_ID}-{REQUEST_ID}"
    assert bridge.waited == [f"e{EDGE_ID}-{REQUEST_ID}"]
    # 流结束(正常或被掐)必须摘登记,否则字典只涨不落
    assert bridge.released == [f"e{EDGE_ID}-{REQUEST_ID}"]


def test_response_headers_echo_unwrapped_request_id() -> None:
    """文档响应头示例:Request-ID 与启动消息同形,不带云内 e{n}- 前缀。"""
    app, _ = _app()
    response = _post(app, *_wire())
    status, resp_headers, _ = _stream(response)
    assert status == 200
    assert resp_headers[HEADER_REQUEST_ID.lower()] == REQUEST_ID
    assert resp_headers["x-edge-cloud-instance-id"] == "cloud-0"
    assert resp_headers["x-edge-cloud-prefix-hit-blocks"] == "2"
    assert resp_headers["x-edge-cloud-prefix-hit-tokens"] == str(2 * BLOCK_SIZE)


def test_heartbeat_keeps_the_stream_alive_until_usage_lands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, bridge = _app()
    bridge.usage_delay = 0.05
    monkeypatch.setattr(ctrl, "_USAGE_HEARTBEAT_INTERVAL", 0.001)
    response = _post(app, *_wire())
    _, _, events = _stream(response)
    comments = [payload for kind, payload in events if kind == "comment"]
    assert comments and all(c == "edge-cloud-usage-pending" for c in comments)
    assert events[-1] == ("done", None)
    assert _control_payload(events)["request_id"] == REQUEST_ID


def test_handler_releases_registration_on_early_disconnect() -> None:
    """边侧收到启动消息就走:生成器被关也要摘登记,否则 usage future 悬到
    进程关停(handler 从未走到 wait_usage,没人会去摘它)。"""
    app, bridge = _app()
    response = _post(app, *_wire())

    async def scenario() -> str:
        stream = response.body_iterator
        first = await stream.__anext__()
        await stream.aclose()  # 等价于客户端断开:生成器收到 GeneratorExit
        return first

    asyncio.run(scenario())
    assert bridge.waited == []  # 没等到 usage 那一步
    assert bridge.released == [f"e{EDGE_ID}-{REQUEST_ID}"]


def test_health_endpoint() -> None:
    app, _ = _app()
    handler = _handler(app, "/health", "GET")
    assert asyncio.run(handler()) == {"status": "ok"}


# --------------------------------------------------------------------- #
# 用量缺席:不伪造零用量                                  #
# --------------------------------------------------------------------- #
def test_missing_usage_ends_the_stream_without_a_fake_usage_frame() -> None:
    """超时/abort 结算不触发时,宁可不给用量帧,也不让网关按 0 出账。"""
    app, bridge = _app()
    bridge.usage = None
    response = _post(app, *_wire())
    _, _, events = _stream(response)
    data_frames = [payload for kind, payload in events if kind == "data"]
    assert all("usage" not in frame for frame in data_frames)
    assert ("comment", "edge-cloud-usage-unavailable") in events
    assert events[-1] == ("done", None)


# --------------------------------------------------------------------- #
# 协议门禁:v1/v2 与 MM-ABI                              #
# --------------------------------------------------------------------- #
def test_v1_request_is_accepted_and_declares_no_mm_abi() -> None:
    app, _ = _app()
    response = _post(app, *_wire())
    _, resp_headers, events = _stream(response)
    assert _control_payload(events)["protocol"] == PROTOCOL_VERSION
    assert "mm_abi" not in _control_payload(events)
    assert HEADER_MM_ABI.lower() not in resp_headers


def test_v2_request_is_accepted_and_echoes_mm_abi() -> None:
    app, _ = _app()
    response = _post(app, *_wire(protocol=PROTOCOL_VERSION_MM, mm_abi=CLOUD_MM_ABI))
    _, resp_headers, events = _stream(response)
    probe = _control_payload(events)
    assert probe["protocol"] == PROTOCOL_VERSION_MM
    assert probe["mm_abi"] == CLOUD_MM_ABI
    assert resp_headers[HEADER_PROTOCOL.lower()] == PROTOCOL_VERSION_MM
    assert resp_headers[HEADER_MM_ABI.lower()] == CLOUD_MM_ABI


def _reject(headers, body, *, enforce=False, fingerprint=CLOUD_FINGERPRINT) -> str:
    app, bridge = _app(enforce=enforce, fingerprint=fingerprint)
    response = _post(app, headers, body)
    assert response.status_code == 400, response.status_code
    assert bridge.probed == []  # 门禁在探针之前,不给云侧留孤儿预留
    return json.loads(response.body)["error"]


@pytest.mark.parametrize("mm_abi", [None, "", "mm1:", "bogus", "mm1:****"])
def test_v2_request_without_a_well_formed_mm_abi_is_rejected(mm_abi) -> None:
    """v2 缺 ABI = 媒体建链口径无从判定:必须拒,不能按文本链命中。"""
    assert _reject(*_wire(protocol=PROTOCOL_VERSION_MM, mm_abi=mm_abi))


def test_v1_request_carrying_mm_abi_is_rejected() -> None:
    """两套协议栈不得互相夹带:v1 带 ABI 说明边侧发错了标记。"""
    error = _reject(*_wire(mm_abi=CLOUD_MM_ABI))
    assert PROTOCOL_VERSION_MM in error


def test_unknown_protocol_version_is_rejected() -> None:
    error = _reject(*_wire(protocol="edge-cloud-prefix-v9"))
    assert "unsupported edge-cloud protocol" in error


def test_missing_prompt_tokens_header_is_rejected() -> None:
    headers, body = _wire()
    del headers[HEADER_PROMPT_TOKENS]
    assert _reject(headers, body)


def test_bad_edge_id_is_rejected() -> None:
    """edge_id 决定云内预留键,非法值不能默默回退 0(会与数据面包裹错位)。"""
    for edge_id in ("-1", "not-a-number"):
        assert "edge" in _reject(*_wire(edge_id=edge_id)).lower()


def test_enforce_mm_abi_match_arbitrates_the_fingerprint() -> None:
    """开关打开:边云指纹不符即 400(云侧仲裁,缺省关)。"""
    error = _reject(
        *_wire(protocol=PROTOCOL_VERSION_MM, mm_abi=OTHER_MM_ABI),
        enforce=True,
    )
    assert "does not match this cloud" in error
    app, _ = _app(enforce=True)
    response = _post(app, *_wire(protocol=PROTOCOL_VERSION_MM, mm_abi=CLOUD_MM_ABI))
    _, _, events = _stream(response)
    assert _control_payload(events)["mm_abi"] == CLOUD_MM_ABI


def test_enforce_mm_abi_match_without_a_cloud_fingerprint_is_rejected() -> None:
    error = _reject(
        *_wire(protocol=PROTOCOL_VERSION_MM, mm_abi=CLOUD_MM_ABI),
        enforce=True,
        fingerprint=None,
    )
    assert "locally computed" in error


def test_probe_failure_returns_503() -> None:
    """探针排队/云引擎不响应:边侧据此 fail-open,不能回 200 半条流。"""

    class _FailingBridge(_StubBridge):
        async def probe(self, manifest, timeout: float = 60.0):
            raise TimeoutError(f"prefix probe timed out after {timeout}s")

    bridge = _FailingBridge()
    app = ctrl.create_lwd_cloud_control_app(bridge, CLOUD_FINGERPRINT)
    response = _post(app, *_wire())
    assert response.status_code == 503
    assert bridge.waited == []


def test_probe_timeout_flows_from_config_to_the_bridge() -> None:
    """云侧探针上限必须由配置派生,不能在三处各写一个数字。"""
    app, bridge = _app(probe_timeout=7.5)
    _post(app, *_wire())
    assert bridge.probe_timeouts == [7.5]
    default_app, default_bridge = _app()
    _post(default_app, *_wire())
    assert default_bridge.probe_timeouts == [25.0]


def test_cloud_probe_cap_is_below_the_edge_wait() -> None:
    """派生关系:云上限 = 边侧等待 − 余量 ⇒ 慢法上永远是云先作答(200/503)。"""
    edge_wait = 25.0  # lwd_coordination.probe_timeout 缺省(本支)
    cap = max(edge_wait - ctrl._CLOUD_PROBE_HEADROOM_S, 1.0)
    assert ctrl._CLOUD_PROBE_HEADROOM_S > 0
    assert cap < edge_wait


# --------------------------------------------------------------------- #
# 桥本体:事件驱动、零线程                                #
# --------------------------------------------------------------------- #
def _bridge_pair() -> tuple[ctrl.LwdCloudControlBridge, queue.Queue, queue.Queue]:
    """父进程侧桥 + 两条本地队列(真机是 mp.Queue,协议形态一致)。"""
    command_queue: queue.Queue = queue.Queue()
    event_queue: queue.Queue = queue.Queue()
    bridge = ctrl.LwdCloudControlBridge(command_queue, event_queue)
    bridge.start()
    return bridge, command_queue, event_queue


def _probe_manifest(request_id: str = "req-001") -> LwdPrefixManifest:
    return LwdPrefixHasher(TENANT_KEY, BLOCK_SIZE).build_manifest(
        request_id, list(range(PROMPT_TOKENS))
    )


def _probe_result(manifest: LwdPrefixManifest) -> LwdProbeResult:
    hit = 2
    return LwdProbeResult(
        request_id=manifest.request_id,
        instance_id="cloud-0",
        block_size=manifest.block_size,
        hit_blocks=hit,
        hit_tokens=hit * manifest.block_size,
    )


DISPATCH_THREAD_NAME = "lwd-ctrl-dispatch"


def _non_dispatch_thread_names() -> set[str]:
    """桥自带 dispatch 线程之外的线程名——用来判"在飞等待有没有占线程"。"""
    return {
        thread.name for thread in threading.enumerate()
        if thread.name != DISPATCH_THREAD_NAME
    }


async def _await_registration(bridge: ctrl.LwdCloudControlBridge,
                              request_id: str) -> None:
    while request_id not in bridge._probe_futures:
        await asyncio.sleep(0.001)


async def _await_command(command_queue: queue.Queue) -> str:
    while command_queue.empty():
        await asyncio.sleep(0.001)
    return str(command_queue.get_nowait()["type"])


def _probe_event(manifest: LwdPrefixManifest) -> dict:
    return {
        "type": "probe",
        "request_id": manifest.request_id,
        "ok": True,
        "result": _probe_result(manifest),
    }


def test_probe_in_flight_does_not_occupy_a_thread() -> None:
    """等待只占协程:探针在飞期间桥一条线程都不多(容量与并发解耦的判据)。

    老形态这里走 ``run_in_executor(None, waiter.get)``,一笔在飞 = 一条
    executor 线程,默认池 ``min(32, cpu+4)`` 撞顶后新探针连"等结果"都排不上
    队(真机证据见 8cfb9f499),且超时后线程不可取消、每笔永久漏一条。"""
    bridge, command_queue, event_queue = _bridge_pair()
    manifest = _probe_manifest()
    before = _non_dispatch_thread_names()

    async def scenario() -> LwdProbeResult:
        task = asyncio.create_task(bridge.probe(manifest, timeout=5.0))
        await _await_registration(bridge, manifest.request_id)
        assert _non_dispatch_thread_names() == before, "在飞等待占了线程池"
        assert await _await_command(command_queue) == "probe"
        event_queue.put(_probe_event(manifest))
        return await task

    result = asyncio.run(scenario())
    assert result.hit_tokens == 2 * BLOCK_SIZE
    # usage future 与 probe future 同时登记,故探针完结后它仍在册;
    # 落册生命周期由 wait_usage / release 收口(handler 的 finally 调 release)
    assert list(bridge._usage_futures) == [manifest.request_id]
    assert bridge._probe_futures == {}
    bridge.release(manifest.request_id)
    assert bridge._usage_futures == {}
    bridge.close()


def test_probe_timeout_leaves_no_waiter_and_survives_late_events() -> None:
    """超时只是取消协程:不留字典项,迟到的结算也不炸(原线程池形态会漏线程)。"""
    bridge, _command_queue, event_queue = _bridge_pair()
    manifest = _probe_manifest("req-ghost")

    async def scenario() -> None:
        with pytest.raises(TimeoutError):
            await bridge.probe(manifest, timeout=0.2)
        assert bridge._probe_futures == {}
        assert bridge._usage_futures == {}
        event_queue.put({
            "type": "usage", "request_id": "req-ghost", "usage": USAGE
        })
        await asyncio.sleep(0.3)  # 走 dispatch 的孤儿分支
        assert bridge._usage_futures == {}

    asyncio.run(scenario())
    bridge.close()


def test_usage_event_arriving_early_is_not_lost() -> None:
    """usage future 与 probe future 同时登记:结算早于 handler 走到
    ``wait_usage`` 也不丢(老形态此时字典里还没有等待者)。"""
    bridge, command_queue, event_queue = _bridge_pair()
    manifest = _probe_manifest("req-early")

    async def scenario() -> None:
        task = asyncio.create_task(bridge.probe(manifest, timeout=5.0))
        await _await_registration(bridge, manifest.request_id)
        await _await_command(command_queue)
        event_queue.put(_probe_event(manifest))
        event_queue.put({"type": "usage", "request_id": manifest.request_id,
                         "usage": USAGE})
        await task  # 探针先完结,usage 紧随其后落到 future
        assert await bridge.wait_usage(manifest.request_id, timeout=1.0) == USAGE
        assert bridge._usage_futures == {}

    asyncio.run(scenario())
    bridge.close()


def test_duplicate_control_request_supersedes_the_previous_one() -> None:
    """同 req_id 重发是正常路径:首探针学到云 hash 粒度后,边侧会按新粒度
    重建链再探一次。第二笔必须被受理(拒绝=打断学习路径),旧登记被取代。"""
    bridge, command_queue, event_queue = _bridge_pair()
    manifest = _probe_manifest("req-dup")

    async def scenario() -> None:
        first = asyncio.create_task(bridge.probe(manifest, timeout=5.0))
        await _await_registration(bridge, manifest.request_id)
        await _await_command(command_queue)
        event_queue.put(_probe_event(manifest))
        await first
        stale_usage = bridge._usage_futures[manifest.request_id]

        second = asyncio.create_task(bridge.probe(manifest, timeout=5.0))
        await asyncio.sleep(0.01)  # 让第二笔跑到登记处
        # 旧流就地收口:它那笔的用量归新流,旧 handler 收到 None(不发用量帧)
        assert stale_usage.done() and stale_usage.result() is None
        assert bridge._usage_futures[manifest.request_id] is not stale_usage
        await _await_command(command_queue)
        event_queue.put(_probe_event(manifest))
        assert (await second).request_id == manifest.request_id
        event_queue.put({"type": "usage", "request_id": manifest.request_id,
                         "usage": USAGE})
        await asyncio.sleep(0.2)
        assert await bridge.wait_usage(manifest.request_id, timeout=1.0) == USAGE
        # 取代之后才到的那条 usage 归最新一代。依据是"一个 rid 只有一个在飞
        # 尝试"这个整栈前提(边侧 nonce/abort 墓碑 + 协调器单预留覆写 + 门池窗口
        # 拦截共同维持,过门后重投同 rid 会撞 Scheduler.add_request 的
        # assert "duplicate request id"),不是"云侧会温和去重"——所以事件不带代际
        # 时投最新登记是唯一自洽的走法;加代际墓碑会把这条也是唯一一条用量判死。
        assert list(bridge._usage_futures) == []

    asyncio.run(scenario())
    bridge.close()


def test_close_fails_pending_waits() -> None:
    """关停把仍在等的 future 置异常:handler 不会永等,dispatch 线程收口。"""
    bridge, _command_queue, _event_queue = _bridge_pair()
    manifest = _probe_manifest("req-shutdown")

    async def scenario() -> None:
        pending = asyncio.create_task(bridge.probe(manifest, timeout=5.0))
        await _await_registration(bridge, manifest.request_id)
        bridge.close()
        await asyncio.sleep(0.05)  # 让 call_soon_threadsafe 的结算跑一轮
        with pytest.raises(RuntimeError, match="control plane stopped"):
            await pending

    asyncio.run(scenario())


def test_dispatch_stops_cleanly_when_the_loop_is_already_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """循环关掉后才到的事件:dispatch 线程**安静**退出,不许带异常死。

    关停路径上 uvicorn 的循环与 dispatch 线程没有先后保证(atexit 与
    asyncio.run 收尾竞态),这里手工摆出"登记在册但循环已关"的状态,避免依赖
    调度时序。判据不能只看线程是否退出——没有 try/except 时它同样会停,只是
    以未捕获 RuntimeError 的形式(每来一条事件刷一次 traceback),所以必须断言
    excepthook 没被触发。"""
    command_queue: queue.Queue = queue.Queue()
    event_queue: queue.Queue = queue.Queue()
    bridge = ctrl.LwdCloudControlBridge(command_queue, event_queue)
    closed_loop = asyncio.new_event_loop()
    closed_loop.close()
    bridge._loop = closed_loop
    # 只需让 request_id 看起来"有等待者";真正结算永不发生
    bridge._usage_futures["req-after-close"] = object()  # type: ignore[assignment]

    caught: list[tuple[str, type[BaseException]]] = []

    def _record(args) -> None:
        caught.append((
            args.thread.name if args.thread is not None else "?",
            args.exc_type,
        ))

    monkeypatch.setattr(threading, "excepthook", _record)

    bridge._dispatch_thread.start()
    event_queue.put({
        "type": "usage", "request_id": "req-after-close", "usage": USAGE
    })
    deadline = time.monotonic() + 3.0
    while bridge._dispatch_thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not bridge._dispatch_thread.is_alive(), "循环已关时 dispatch 未收线程"
    assert caught == [], f"dispatch 线程带异常退出: {caught}"
    # 哨兵不是真 future:摘干净再关,避免 close() 对它调 done()
    bridge._usage_futures.clear()
    bridge.close()
