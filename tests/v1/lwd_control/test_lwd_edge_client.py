# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""边侧探针客户端契约:影子体请求、SSE 启动消息解析、计费流归属。

覆盖《边-网关-云的交互逻辑》里 Edge 这一侧的全部可执行判据:
  - 过网的只有 model/messages/stream/stream_options(真实 prompt、采样参数
    与图片数据一律不出边);
  - 五个 X-Edge-Cloud-* 请求头 + Accept + 可选 Authorization;
  - 文本发 v1、图片发 v2 并带 MM-ABI 指纹;
  - 启动消息只从 delta.content 取(响应头可以被网关吃掉);
  - 云侧回显必须与本端发声一致(v1/v2、MM-ABI),否则 fail-closed;
  - 边侧放弃握手时不掐已建立的计费流。
"""

import asyncio
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass

import pytest

from vllm.v1.lwd_control.control_communication.lwd_mm_identity import (
    mm_abi_header_value,
)
from vllm.v1.lwd_control.control_communication.lwd_prefix import (
    BLOCK_HASH_PREFIX,
    HEADER_BLOCK_SIZE,
    HEADER_EDGE_ID,
    HEADER_MM_ABI,
    HEADER_PROMPT_TOKENS,
    HEADER_PROTOCOL,
    HEADER_REQUEST_ID,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_MM,
    TAIL_HASH_PREFIX,
    LwdPrefixManifest,
    LwdProbeResult,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_client import (
    LwdEdgePrefixClient,
)

pytestmark = pytest.mark.cpu_test

TENANT_KEY = b"tenant-a-secret-key-material"
BLOCK_SIZE = 4
CONTROL_URL = "https://gateway.example.com/v1/chat/completions"
FINGERPRINT = hashlib.sha256(b"processor-config").digest()
MM_ABI = mm_abi_header_value(FINGERPRINT)
IMAGE_DIGEST = hashlib.sha256(b"image-a").digest()


@dataclass(frozen=True)
class MediaItem:
    modality: str
    digest: bytes
    offset: int
    length: int


MEDIA_ITEMS = (MediaItem("image", IMAGE_DIGEST, 0, BLOCK_SIZE),)


# --------------------------------------------------------------------- #
# 替身:网关/云侧的 HTTP 报文与 aiohttp 会话                    #
# --------------------------------------------------------------------- #
def _control_message(
    *,
    protocol: str = PROTOCOL_VERSION,
    request_id: str = "req-001",
    hit_blocks: int = 1,
    mm_abi: str | None = None,
) -> str:
    message = {
        "type": "edge_cloud_probe",
        "protocol": protocol,
        "request_id": request_id,
        "instance_id": "cloud-0",
        "block_size": BLOCK_SIZE,
        "hit_blocks": hit_blocks,
        "hit_tokens": hit_blocks * BLOCK_SIZE,
        "alignment_tokens": 0,
    }
    if mm_abi is not None:
        message["mm_abi"] = mm_abi
    return json.dumps(message, separators=(",", ":"))


def _frame(content: str | None = None, *, usage: dict | None = None) -> str:
    chunk: dict = {
        "id": "req-001",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "qwen3.5",
        "choices": [],
    }
    if content is not None:
        chunk["choices"] = [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": None,
            }
        ]
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"


def _empty_delta() -> str:
    """文档事件二:delta 是空对象(finish_reason 仍为 null)。"""
    chunk = {
        "id": "req-001",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "qwen3.5",
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"


def _comment(text: str = "edge-cloud-usage-pending") -> str:
    return f": {text}\n\n"


def _error_frame() -> str:
    return 'data: {"error":{"message":"upstream unavailable"}}\n\n'


def _usage_frame(prompt: int = 10, completion: int = 2, cached: int = 4) -> str:
    return _frame(
        usage={
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": cached},
        }
    )


DONE = "data: [DONE]\n\n"


class _FakeStream:
    """aiohttp StreamReader 替身:按行异步迭代(与真实行为一致)。"""

    def __init__(self, text: str) -> None:
        self._lines = text.splitlines(keepends=True)

    def __aiter__(self):
        async def _emit():
            for line in self._lines:
                yield line.encode("utf-8")

        return _emit()


class _FakeResponse:
    def __init__(self, sse_text: str, status: int = 200) -> None:
        self.status = status
        self.content = _FakeStream(sse_text)
        self.close_calls = 0
        self.close_calls_on_exit = 0

    def close(self) -> None:
        self.close_calls += 1

    async def read(self, limit: int = -1) -> bytes:
        return b'{"error":"upstream rejected the manifest"}'


class _NeverStream:
    """有连接、有 200,但首帧永不落地的流(云/网关收下请求后挂住的形态)。"""

    def __init__(self) -> None:
        self._gate = asyncio.Event()

    def __aiter__(self):
        async def _emit():
            if not self._gate.is_set():
                # 让这里成为异步生成器,同时永不产出第一行
                await self._gate.wait()
            yield b""

        return _emit()


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.requests: list[dict] = []

    async def post(self, url, json=None, headers=None, timeout=None):
        self.requests.append(
            {"url": url, "body": json, "headers": headers, "timeout": timeout}
        )
        return self.response


def _client(
    *,
    fingerprint: bytes | None = FINGERPRINT,
    consumer_id: str | None = None,
    probe_timeout: float = 30.0,
) -> LwdEdgePrefixClient:
    return LwdEdgePrefixClient(
        CONTROL_URL,
        BLOCK_SIZE,
        TENANT_KEY,
        edge_id=7,
        consumer_id=consumer_id,
        processor_fingerprint=fingerprint,
        probe_timeout=probe_timeout,
    )


def _manifest(
    client: LwdEdgePrefixClient, tokens: int = 10, media: tuple = ()
) -> LwdPrefixManifest:
    return client.build_manifest(
        "req-001", list(range(tokens)), media_items=media
    )


@contextmanager
def _gateway_token(value: str | None):
    previous = os.environ.get("VLLM_LWD_CLOUD_API_KEY")
    if value is None:
        os.environ.pop("VLLM_LWD_CLOUD_API_KEY", None)
    else:
        os.environ["VLLM_LWD_CLOUD_API_KEY"] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("VLLM_LWD_CLOUD_API_KEY", None)
        else:
            os.environ["VLLM_LWD_CLOUD_API_KEY"] = previous


async def _negotiate(client, sse_text: str, *, media: tuple = (), status=200):
    """跑一次 negotiate,返回 (probe | None, error | None, 请求快照, 响应替身)。

    替身上另留 ``close_calls_on_exit``:negotiate 返回/抛出那一刻的 close
    次数——契约是"握手成败都不由 negotiate 掐流",掐流只能发生在后台排空
    task 读完之后。顺带收干净排空 task(它们挂在同一条 loop 上)。"""
    response = _FakeResponse(sse_text, status=status)
    session = _FakeSession(response)
    error: BaseException | None = None
    probe = None
    try:
        probe = await client.negotiate(
            session, "req-001", list(range(10)), media_items=media
        )
    except BaseException as exc:  # noqa: BLE001
        error = exc
    response.close_calls_on_exit = response.close_calls
    pending = list(client._drain_tasks)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    return probe, error, session.requests, response


# --------------------------------------------------------------------- #
# 请求侧:影子体与请求头                                #
# --------------------------------------------------------------------- #
def test_control_request_is_a_whitelisted_openai_shadow_body() -> None:
    with _gateway_token(None):
        client = _client(consumer_id="tenant-a")
        manifest = _manifest(client)
        headers, body = client.build_control_request(manifest, model="qwen3.5")
    assert set(body) == {"model", "messages", "stream", "stream_options"}
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert all(m["role"] == "user" for m in body["messages"])
    assert all(
        m["content"].startswith((BLOCK_HASH_PREFIX, TAIL_HASH_PREFIX))
        for m in body["messages"]
    )
    # 采样参数/输出上限/原始 messages 一律不过网
    assert not {"max_tokens", "max_completion_tokens", "temperature", "n"} & set(body)
    lowered = {k.lower() for k in headers}
    assert {
        HEADER_PROTOCOL.lower(),
        HEADER_REQUEST_ID.lower(),
        HEADER_EDGE_ID.lower(),
        HEADER_BLOCK_SIZE.lower(),
        HEADER_PROMPT_TOKENS.lower(),
        "accept",
        "x-mse-consumer",
    } <= lowered
    assert "authorization" not in lowered
    assert headers[HEADER_PROMPT_TOKENS] == "10"
    assert headers[HEADER_EDGE_ID] == "7"
    assert headers["Accept"] == "text/event-stream"


def test_text_probe_declares_v1_without_mm_abi() -> None:
    client = _client()
    headers, _ = client.build_control_request(_manifest(client))
    assert headers[HEADER_PROTOCOL] == PROTOCOL_VERSION
    assert HEADER_MM_ABI not in headers
    headers, _ = client.build_control_request(
        _manifest(client), multimodal=False
    )
    assert headers[HEADER_PROTOCOL] == PROTOCOL_VERSION


def test_image_probe_declares_v2_with_mm_abi() -> None:
    client = _client()
    manifest = _manifest(client, media=MEDIA_ITEMS)
    headers, _ = client.build_control_request(
        manifest, model="qwen3.5", multimodal=True
    )
    assert headers[HEADER_PROTOCOL] == PROTOCOL_VERSION_MM
    assert headers[HEADER_MM_ABI] == MM_ABI


def test_image_probe_without_fingerprint_is_rejected() -> None:
    """没指纹就发不出 v2:宁可抛错,也不发一条云侧无法辨识口径的请求。"""
    client = _client(fingerprint=None)
    manifest = _manifest(client)  # 纯文本链仍建得出
    with pytest.raises(ValueError, match=HEADER_MM_ABI):
        client.build_control_request(manifest, multimodal=True)


def test_gateway_bearer_token_shape() -> None:
    with _gateway_token("sk-gw-test"):
        headers, _ = _client().build_control_request(_manifest(_client()))
        assert headers["Authorization"] == "Bearer sk-gw-test"
    for bad in ("", " ", "sk with space", "sk-ünïcode"):
        with _gateway_token(bad), pytest.raises(ValueError):
            _client()


# --------------------------------------------------------------------- #
# 回显侧:启动消息解析                                    #
# --------------------------------------------------------------------- #
def test_negotiate_reads_probe_from_content_and_keeps_stream_open() -> None:
    async def scenario() -> None:
        client = _client()
        stream = (
            _frame(content=_control_message(hit_blocks=2))
            + _empty_delta()
            + _comment()
            + _usage_frame()
            + DONE
        )
        probe, error, requests, response = await _negotiate(client, stream)
        assert error is None
        assert isinstance(probe, LwdProbeResult)
        assert probe.request_id == "req-001"
        assert probe.hit_blocks == 2 and probe.hit_tokens == 2 * BLOCK_SIZE
        assert probe.instance_id == "cloud-0"
        # 握手不依赖响应头:替身不提供任何 X-Edge-Cloud-* 响应头
        assert requests[0]["url"] == CONTROL_URL
        assert requests[0]["timeout"].total is None  # 不钳总时长,等推理完结
        # 握手归来 negotiate 自己不关流;读完 usage 才由排空 task 收口
        assert response.close_calls_on_exit == 0
        assert response.close_calls == 1
        assert not client._drain_tasks  # 排空完成后登记已摘除

    asyncio.run(scenario())


def test_negotiate_assembles_probe_split_across_frames() -> None:
    """网关可能把 delta.content 切成多帧,中间还夹心跳 comment。"""
    async def scenario() -> None:
        payload = _control_message(hit_blocks=1)
        half = len(payload) // 2
        stream = (
            _frame(content=payload[:half])
            + _comment()
            + _frame(content=payload[half:])
            + _empty_delta()
            + _usage_frame()
            + DONE
        )
        probe, error, _, _ = await _negotiate(_client(), stream)
        assert error is None
        assert probe is not None and probe.hit_tokens == BLOCK_SIZE

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "cloud_protocol", [PROTOCOL_VERSION_MM, "edge-cloud-prefix-v9"]
)
def test_text_probe_rejects_unacknowledged_protocol(cloud_protocol: str) -> None:
    """边发 v1,云回 v2 或未知版本一律拒:两侧哈希域不同,命中必错位。"""

    async def scenario() -> None:
        stream = (
            _frame(content=_control_message(protocol=cloud_protocol))
            + _usage_frame()
            + DONE
        )
        probe, error, _, _ = await _negotiate(_client(), stream)
        assert probe is None
        assert isinstance(error, RuntimeError)
        assert cloud_protocol in str(error)

    asyncio.run(scenario())


def test_image_probe_requires_matching_mm_abi_echo() -> None:
    async def scenario() -> None:
        for echoed in (None, "mm1:someotherfingerprint"):
            stream = (
                _frame(
                    content=_control_message(
                        protocol=PROTOCOL_VERSION_MM, mm_abi=echoed
                    )
                )
                + _usage_frame()
                + DONE
            )
            probe, error, _, _ = await _negotiate(
                _client(), stream, media=MEDIA_ITEMS
            )
            assert probe is None, echoed
            assert isinstance(error, RuntimeError), echoed

    asyncio.run(scenario())


def test_image_probe_happy_path_accepts_mm_abi_echo() -> None:
    async def scenario() -> None:
        stream = (
            _frame(
                content=_control_message(
                    protocol=PROTOCOL_VERSION_MM, hit_blocks=2, mm_abi=MM_ABI
                )
            )
            + _empty_delta()
            + _usage_frame()
            + DONE
        )
        probe, error, requests, _ = await _negotiate(
            _client(), stream, media=MEDIA_ITEMS
        )
        assert error is None
        assert probe is not None and probe.hit_tokens == 2 * BLOCK_SIZE
        assert requests[0]["headers"][HEADER_PROTOCOL] == PROTOCOL_VERSION_MM
        assert requests[0]["headers"][HEADER_MM_ABI] == MM_ABI

    asyncio.run(scenario())


def test_usage_or_error_before_probe_is_rejected() -> None:
    """启动消息之前出现用量/错误帧 = 流序不对,不能当残缺握手用。"""

    async def scenario() -> None:
        for premature in (_usage_frame(), _error_frame()):
            stream = _empty_delta() + premature + DONE
            probe, error, _, _ = await _negotiate(_client(), stream)
            assert probe is None, premature
            assert isinstance(error, RuntimeError), premature

    asyncio.run(scenario())


def test_text_probe_rejects_mm_abi_echo() -> None:
    """纯文本请求云侧却回 MM-ABI:两侧协议栈不同,拒。"""

    async def scenario() -> None:
        stream = (
            _frame(content=_control_message(mm_abi=MM_ABI))
            + _usage_frame()
            + DONE
        )
        probe, error, _, _ = await _negotiate(_client(), stream)
        assert probe is None
        assert isinstance(error, RuntimeError)

    asyncio.run(scenario())


def test_stream_without_probe_is_rejected() -> None:
    async def scenario() -> None:
        probe, error, _, _ = await _negotiate(
            _client(), _empty_delta() + _comment() + _usage_frame() + DONE
        )
        assert probe is None
        assert isinstance(error, RuntimeError)

    asyncio.run(scenario())


def test_request_id_mismatch_is_rejected() -> None:
    async def scenario() -> None:
        stream = (
            _frame(content=_control_message(request_id="someone-elses-req"))
            + _usage_frame()
            + DONE
        )
        probe, error, _, _ = await _negotiate(_client(), stream)
        assert probe is None
        assert isinstance(error, RuntimeError)

    asyncio.run(scenario())


# --------------------------------------------------------------------- #
# 计费流归属:边侧放弃不等于掐流                        #
# --------------------------------------------------------------------- #
def test_abandoned_probe_still_drains_the_established_stream() -> None:
    """看门狗/校验失败放弃握手时,已建立的 200 流必须交给排空 task。

    云侧 handler 在 probe 之后一直挂着等 usage;边侧掐流会让网关记
    client_gone,这笔真实用量就再也落不到账单里。"""

    async def scenario() -> None:
        client = _client()
        stream = (
            _frame(content=_control_message(protocol="edge-cloud-prefix-v9"))
            + _usage_frame(prompt=4096, completion=64, cached=1024)
            + DONE
        )
        probe, error, _, response = await _negotiate(client, stream)
        assert probe is None and error is not None
        # 放弃握手不等于掐流:排空 task 接手后才能关
        assert response.close_calls_on_exit == 0
        assert response.close_calls == 1

    asyncio.run(scenario())


def test_same_request_id_keeps_both_drain_tasks_referenced() -> None:
    """同 req_id 重发(re-add/重试)时两条排空流都要在册。

    按 req_id 索引的登记会被第二笔覆盖,前一条 task 的唯一强引用随之消失;
    而挂在网络读上的协程没有待执行回调,被 GC 掉就是半途丢流(网关记
    client_gone、这笔用量永不落地)。登记改成 set 后覆盖不可能发生。"""

    async def _payloads(*items: str):
        for item in items:
            yield item

    usage_payload = json.dumps({
        "id": "req-1", "object": "chat.completion.chunk", "created": 0,
        "model": "qwen3.5", "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                  "total_tokens": 12},
    }, separators=(",", ":"))

    async def scenario() -> None:
        client = _client()
        first = client._start_drain(
            "req-1", _FakeResponse(""), _payloads(usage_payload, "[DONE]"))
        second = client._start_drain(
            "req-1", _FakeResponse(""), _payloads(usage_payload, "[DONE]"))
        assert len(client._drain_tasks) == 2
        assert first in client._drain_tasks
        assert second in client._drain_tasks
        await asyncio.gather(first, second)
        assert not client._drain_tasks  # 完成后各自摘除,不会互相顶掉

    asyncio.run(scenario())


def test_non_200_closes_the_response() -> None:
    async def scenario() -> None:
        client = _client()
        probe, error, _, response = await _negotiate(
            client, "not an sse stream", status=503
        )
        assert probe is None
        assert isinstance(error, RuntimeError)
        assert "503" in str(error)
        assert response.close_calls == 1
        assert not client._drain_tasks

    asyncio.run(scenario())


def test_probe_read_timeout_keeps_the_billing_stream() -> None:
    """200 到手但首帧永不落地:按 ``probe_timeout`` 放弃,但不掐那条流。

    云侧 handler 在 probe 之后一直挂着等 usage;边侧掐流会让网关记
    client_gone,这笔真实用量永不落地。所以超时也必须走"交后台排空"。"""

    async def scenario() -> None:
        client = _client(probe_timeout=0.05)
        response = _FakeResponse("")
        response.content = _NeverStream()
        session = _FakeSession(response)
        with pytest.raises(RuntimeError, match="timed out"):
            await client.negotiate(session, "req-1", list(range(10)))
        assert response.close_calls == 0
        assert len(client._drain_tasks) == 1
        for task in list(client._drain_tasks):
            task.cancel()
        await asyncio.gather(*client._drain_tasks, return_exceptions=True)
        await asyncio.sleep(0)
        assert not client._drain_tasks

    asyncio.run(scenario())


def test_probe_timeout_must_be_positive() -> None:
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="probe_timeout"):
            _client(probe_timeout=bad)
