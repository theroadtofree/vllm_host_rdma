# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""边侧前缀协商客户端(SSE 网关形态)。

从参考分支 ``edge_cloud/edge_client.py`` 演化:HMAC manifest 影子体 POST +
SSE 解析 Probe 控制消息 + 后台排空 usage 流。原始 messages/采样参数不过网。

协议形态(OpenAI 兼容,经网关透明转发):
  请求: POST control_url, body 为 manifest 影子体, headers 带
        X-Edge-Cloud-* + 可选 Authorization Bearer。图片请求协议声明
        ``edge-cloud-prefix-v2`` 并附 ``X-Edge-Cloud-MM-ABI`` 指纹。
  响应: SSE 流——首 chunk 的 delta.content 内嵌 edge_cloud_probe JSON
        控制消息;随后空 delta 帧;心跳 comment;末 chunk 为标准 OpenAI
        usage + [DONE]。网关(delta.content 不丢)可据此计费。

usage 消费口径:后台排空 SSE 排 usage 做边侧日志留痕(与数据面
``LwdC2eNotify.usages`` 双通道,后者供记账/前端)。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from typing import Any

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.lwd_control.lwd_comm.lwd_mm_identity import (
    mm_abi_header_value,
)
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import (
    HEADER_EDGE_ID,
    HEADER_MM_ABI,
    HEADER_PROTOCOL,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_MM,
    LwdPrefixHasher,
    LwdPrefixManifest,
    LwdProbeResult,
)

logger = init_logger(__name__)

# SSE 控制消息 JSON 尺寸上限(防恶意/异常响应撑爆内存)
_MAX_CONTROL_MESSAGE_CHARS = 16 * 1024


async def _iter_sse_data(
    content: AsyncIterable[bytes],
) -> AsyncIterator[str]:
    """Read SSE events, preserving multiline data and ignoring heartbeats."""
    data: list[str] = []
    size = 0
    async for raw_line in content:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                payload = "\n".join(data)
                data.clear()
                size = 0
                yield payload
        elif line == "data" or line.startswith("data:"):
            value = line.partition(":")[2].removeprefix(" ")
            size += len(value) + 1
            if size > _MAX_CONTROL_MESSAGE_CHARS:
                raise ValueError("edge-cloud SSE event exceeds the control message limit")
            data.append(value)
    if data:
        yield "\n".join(data)


class LwdEdgePrefixClient:
    """边侧准入客户端:HMAC manifest 影子体 + SSE Probe 解析。"""

    def __init__(
        self,
        control_url: str,
        block_size: int,
        tenant_key: bytes,
        edge_id: int = 0,
        consumer_id: str | None = None,
        connect_timeout: float = 10.0,
        processor_fingerprint: bytes | None = None,
        probe_timeout: float = 30.0,
    ) -> None:
        if not control_url.startswith(("http://", "https://")):
            raise ValueError("control_url must use http:// or https://")
        if probe_timeout <= 0:
            raise ValueError("probe_timeout must be positive")
        self._control_url = control_url
        self._hasher = LwdPrefixHasher(
            tenant_key, block_size, processor_fingerprint
        )
        # 图片请求的 MM-ABI 头值(由本地处理器指纹渲染);无指纹则 v2 无法发声
        self._mm_abi_header: str | None = (
            mm_abi_header_value(processor_fingerprint)
            if processor_fingerprint is not None
            else None
        )
        self._edge_id = edge_id
        self._consumer_id = consumer_id
        self._connect_timeout = connect_timeout
        # 整段探针(建连+等首帧)的端到端上限;云侧上限是它减余量,
        # 所以正常情况下云先作答。值来自 lwd_coordination.probe_timeout。
        self._probe_timeout = probe_timeout
        # 探针流后台排空 task(asyncio 只持弱引用,这里必须持强引用;
        # set 而非按 req_id 索引 —— 见 _start_drain)
        self._drain_tasks: set[Any] = set()
        # 网关 Bearer 令牌(敏感,不入日志):env 读取,缺省 None = 不发
        api_key = envs.VLLM_LWD_CLOUD_API_KEY
        if api_key is not None:
            if (
                not api_key
                or not api_key.isascii()
                or not api_key.isprintable()
                or any(c.isspace() for c in api_key)
            ):
                raise ValueError(
                    "VLLM_LWD_CLOUD_API_KEY must contain a non-empty ASCII "
                    "API token without whitespace"
                )
        self._api_key: str | None = api_key

    @property
    def block_size(self) -> int:
        return self._hasher.block_size

    def build_manifest(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        media_items: Sequence[Any] = (),
    ) -> LwdPrefixManifest:
        """租户密钥链式 HMAC 产 manifest(原始 prompt 不出边)。"""
        return self._hasher.build_manifest(
            request_id, prompt_token_ids, media_items=media_items
        )

    def build_control_request(
        self,
        manifest: LwdPrefixManifest,
        model: str | None = None,
        *,
        multimodal: bool = False,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """白名单影子体:仅 manifest/stream,采样参数不过网。

        必需控制头(含 ``X-Edge-Cloud-Prompt-Tokens``)由
        ``LwdPrefixManifest.to_headers()`` 一次给全,这里只补边侧身份与网关
        相关的头;model 可选透传(网关按它选渠道)。
        ``multimodal`` 由调用方按建链时的 media_items 决定:媒体覆盖块走
        0x04/0x05 域,协议必须同步声明 v2 并带 MM-ABI 指纹,否则云侧无从
        判定这套链的口径。"""
        headers = manifest.to_headers()
        if multimodal:
            if self._mm_abi_header is None:
                raise ValueError(
                    "multimodal probe needs a processor fingerprint for "
                    f"{HEADER_MM_ABI}"
                )
            headers[HEADER_PROTOCOL] = PROTOCOL_VERSION_MM
            headers[HEADER_MM_ABI] = self._mm_abi_header
        headers[HEADER_EDGE_ID] = str(self._edge_id)
        headers["Accept"] = "text/event-stream"
        if self._consumer_id:
            headers["X-Mse-Consumer"] = self._consumer_id
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        body: dict[str, Any] = {
            "messages": manifest.to_messages(),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if model is not None:
            body["model"] = model
        return headers, body

    async def negotiate(
        self,
        session: Any,
        request_id: str,
        prompt_token_ids: list[int],
        manifest: LwdPrefixManifest | None = None,
        media_items: Sequence[Any] = (),
        model: str | None = None,
    ) -> LwdProbeResult:
        """POST 探测并从 SSE delta.content 解析 Probe;失败即抛(fail-closed)。

        命中结果由调用方消费(命中时推进 request.num_computed_tokens,
        使边侧只 embed/发尾巴)。manifest 可传入调用方已建好的链复用
        (避免一次探测两次 O(N) 建链);缺省现建(media_items 须与
        调用方建链口径一致,防缺省路径漏掺多模态身份)。

        整段(建连 + 等首帧)由 ``probe_timeout`` 钳住。云侧 handler 的上限是
        ``probe_timeout - 5``,所以正常慢法是**云先作答**(200 或 503),这条
        超时只在云/网关真没答时兜底——不再是边侧抢先放弃、把仍在等 usage 的
        流掐掉。放弃时计费流的归属见 ``_post_and_read_probe``。"""
        if manifest is None:
            manifest = self.build_manifest(
                request_id, prompt_token_ids, media_items
            )
        # 媒体身份非空 ⇔ 建链掺了 0x04/0x05 域 ⇔ 必须以 v2 发声
        multimodal = bool(media_items)
        headers, body = self.build_control_request(
            manifest, model=model, multimodal=multimodal
        )
        try:
            response, sse_data, probe = await asyncio.wait_for(
                self._post_and_read_probe(
                    session, headers, body,
                    request_id=request_id,
                    prompt_token_ids=prompt_token_ids,
                    multimodal=multimodal,
                ),
                self._probe_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"prefix probe timed out after {self._probe_timeout}s "
                f"url={self._control_url}"
            ) from exc
        # probe 到手即返回,但**不能**在这里关流:云侧 handler 要等推理完结
        # 才发 usage 末 chunk(最长 600s),真实 token 用量在这一帧落地。
        self._start_drain(request_id, response, sse_data)

        logger.info(
            "[Lwd][edge-client] negotiate req=%s hit_tokens=%d cloud_id=%s",
            request_id, probe.hit_tokens, probe.cloud_id,
        )
        return probe

    async def _post_and_read_probe(
        self,
        session: Any,
        headers: dict[str, str],
        body: dict[str, Any],
        *,
        request_id: str,
        prompt_token_ids: list[int],
        multimodal: bool,
    ) -> tuple[Any, AsyncIterator[str], LwdProbeResult]:
        """发影子体、读 Probe;放弃时决定"交排空还是关闭"。

        200 已到手 = 云侧 handler 已经挂在这笔的 usage 上。此后无论放弃原因
        是超时、被取消还是回显校验不过,都**不能** close:边侧提前断开会让
        网关记成 client_gone/context canceled,这笔真实用量永久丢失。"""
        import aiohttp

        timeout = aiohttp.ClientTimeout(
            total=None, connect=self._connect_timeout
        )
        response = await session.post(
            self._control_url, json=body, headers=headers, timeout=timeout
        )
        # 非 None ⇔ 200 已到手且拿到了流迭代器(放弃时据此决定排空还是关闭)
        sse_data: AsyncIterator[str] | None = None
        try:
            if response.status != 200:
                detail = await self._read_error_body(response)
                raise RuntimeError(
                    f"prefix negotiation HTTP {response.status} "
                    f"url={self._control_url} body={detail}"
                )
            # 与排空 task 共用同一迭代器:probe 帧之后的剩余帧由 task 续读
            sse_data = _iter_sse_data(response.content)
            probe = await self._read_probe_from_sse(
                sse_data, multimodal=multimodal
            )
            if probe.request_id != request_id:
                raise RuntimeError("cloud returned a different request ID")
            # 块粒度不强求一致:混合模型(如 Mamba/SWA)下云 hash 粒度可
            # 小于边 KV 块粒度,由调用方按 probe.block_size 学习重建链
            # (见 _lwd_hash_bs_by_cloud)。
            if probe.hit_tokens > len(prompt_token_ids):
                raise RuntimeError("cloud prefix hit exceeds the prompt length")
        except BaseException:
            if sse_data is not None:
                self._start_drain(request_id, response, sse_data)
            else:
                response.close()
            raise
        assert sse_data is not None  # 成功路径必已建流
        return response, sse_data, probe

    def _start_drain(
        self,
        request_id: str,
        response: Any,
        sse_data: AsyncIterator[str],
    ) -> Any:
        """把 probe 帧之后的剩余帧交后台 task 排空(同条 loop,不占 _job 返回路径)。

        登记必须是 **set** 而不是按 req_id 索引的 dict:asyncio 只持弱引用,
        而挂在网络读上的协程没有待执行回调,这份登记就是它唯一的强引用。同
        req_id 重发(re-add/重试)时按 id 覆盖会丢掉**上一条**流的唯一强引用,
        那条流可能在读完 usage 之前被 GC 掉——网关记 client_gone、这笔用量丢。
        返回 task 只为测试可断言登记归属。"""
        task = asyncio.create_task(
            self._drain_stream(request_id, response, sse_data)
        )
        self._drain_tasks.add(task)
        task.add_done_callback(self._drain_tasks.discard)
        return task

    async def _drain_stream(
        self,
        request_id: str,
        response: Any,
        sse_data: AsyncIterator[str],
    ) -> None:
        """排空探针流至 usage 末 chunk:供网关计费的收流路径。

        本侧只留痕不消费(记账以云侧 C2e 回边的 ``[Lwd][edge] usage`` 为准),
        但必须读完:读完才让云侧 handler 正常走到末 chunk + [DONE]。
        云侧 usage 等待有 600s 上限,超时也照样以 [DONE] 收流(只是不带
        usage 帧),故 task 数有界(= 在飞请求数),无需额外超时钳制。
        usage 缺席按 warning 留痕——这一笔网关侧无上游用量可计。"""
        usage: dict | None = None
        try:
            async for payload in sse_data:
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
            if usage is None:
                logger.warning(
                    "[Lwd][edge-client] usage missing in probe stream req=%s",
                    request_id,
                )
                return
            details = usage.get("prompt_tokens_details") or {}
            logger.info(
                "[Lwd][edge-client] usage req=%s prompt=%s completion=%s "
                "total=%s cached=%s",
                request_id,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                usage.get("total_tokens"),
                details.get("cached_tokens"),
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001  排空失败只丢这笔网关账,不影响推理
            logger.warning(
                "[Lwd][edge-client] probe stream drain failed req=%s",
                request_id,
                exc_info=True,
            )
        finally:
            try:
                response.close()
            except Exception:  # noqa: BLE001  关闭失败无需上抛
                logger.debug(
                    "[Lwd][edge-client] probe stream close failed req=%s",
                    request_id,
                    exc_info=True,
                )

    @staticmethod
    async def _read_error_body(response: Any, limit: int = 512) -> str:
        """非 200 时读错误体(截断 + 折叠空白:网关常回整页 HTML)。"""
        try:
            raw = await response.read(limit)
            return " ".join(raw.decode("utf-8", "replace").split())
        except Exception as exc:  # noqa: BLE001  诊断路径不再抛
            return f"<unreadable: {type(exc).__name__}>"

    async def _read_probe_from_sse(
        self, sse_data: AsyncIterator[str], *, multimodal: bool
    ) -> LwdProbeResult:
        """从 SSE 流解析首个 edge_cloud_probe 控制消息(delta.content 累积)。

        回显必校验(fail-closed):云侧 ack 的 protocol 必须是本端发声的那一套
        ——v2 图片被静默降回 v1 意味着媒体覆盖块按文本口径入账,命中会错位;
        v2 还须回带与本端一致的 MM-ABI 指纹,纯文本则不得出现该字段。probe
        之前出现 usage/error 帧说明流序不对,同样抛错而不是当残缺握手用。
        content 被网关分片送达时逐帧拼接,故只在拼出完整对象后判定。"""
        expected_protocol = (
            PROTOCOL_VERSION_MM if multimodal else PROTOCOL_VERSION
        )
        content_buf = ""
        async for payload in sse_data:
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if not isinstance(chunk, dict):
                raise ValueError("cloud returned a non-object SSE chunk")
            if chunk.get("usage") is not None or chunk.get("error") is not None:
                raise RuntimeError(
                    "cloud returned usage or an error before the probe"
                )
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content_buf += delta.get("content") or ""
            if not content_buf:
                continue
            try:
                message = json.loads(content_buf)
            except json.JSONDecodeError:
                continue  # 分片未拼全,续读
            if not isinstance(message, dict):
                raise ValueError("cloud probe must contain a JSON object")
            if "protocol" not in message:
                continue  # 恰好切在首个键之后:再拼一帧
            if message["protocol"] != expected_protocol:
                raise RuntimeError(
                    f"cloud did not acknowledge {expected_protocol} "
                    f"(returned {message['protocol']!r})"
                )
            mm_abi = message.get("mm_abi")
            if multimodal:
                if mm_abi != self._mm_abi_header:
                    raise RuntimeError(
                        "cloud MM-ABI fingerprint does not match the local "
                        "processor fingerprint"
                    )
            elif mm_abi is not None:
                raise RuntimeError("cloud returned MM-ABI for a text-only probe")
            return LwdProbeResult.from_control_message(message)
        raise RuntimeError("edge-cloud probe control message not found in SSE stream")
