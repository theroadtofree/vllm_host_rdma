# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""边侧前缀协商客户端(SSE 网关形态)。

从参考分支 ``edge_cloud/edge_client.py`` 演化:HMAC manifest 影子体 POST +
SSE 解析 Probe 控制消息 + 后台排空 usage 流。原始 messages/采样参数不过网。

协议形态(OpenAI 兼容,经网关透明转发):
  请求: POST control_url, body 为 manifest 影子体, headers 带
        X-Edge-Cloud-* + 可选 Authorization Bearer。
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
from vllm.v1.lwd_control.control_communication.lwd_prefix import (
    HEADER_EDGE_ID,
    HEADER_PROMPT_TOKENS,
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
    ) -> None:
        if not control_url.startswith(("http://", "https://")):
            raise ValueError("control_url must use http:// or https://")
        self._control_url = control_url
        self._hasher = LwdPrefixHasher(
            tenant_key, block_size, processor_fingerprint
        )
        self._edge_id = edge_id
        self._consumer_id = consumer_id
        self._connect_timeout = connect_timeout
        # 探针流后台排空 task(按 req_id 持强引用:asyncio 只持弱引用,
        # 无引用可能被 GC 半途丢账)
        self._drain_tasks: dict[str, Any] = {}
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
        self, manifest: LwdPrefixManifest, model: str | None = None
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """白名单影子体:仅 manifest/stream,采样参数不过网。

        与参考分支(thujenkinbot)一致:prompt_tokens 只走 header
        (NewAPI 等网关 DTO 转换会剥 body 非标准字段);model 可选透传。"""
        headers = manifest.to_headers()
        headers[HEADER_EDGE_ID] = str(self._edge_id)
        headers[HEADER_PROMPT_TOKENS] = str(manifest.prompt_tokens)
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

        SSE 流在 Probe 解析后由后台 task 排空(等 usage 末 chunk),
        不阻塞主流程。"""
        import aiohttp

        if manifest is None:
            manifest = self.build_manifest(
                request_id, prompt_token_ids, media_items
            )
        headers, body = self.build_control_request(manifest, model=model)
        timeout = aiohttp.ClientTimeout(
            total=None, connect=self._connect_timeout
        )
        response = await session.post(
            self._control_url, json=body, headers=headers, timeout=timeout
        )
        try:
            if response.status != 200:
                detail = await self._read_error_body(response)
                raise RuntimeError(
                    f"prefix negotiation HTTP {response.status} "
                    f"url={self._control_url} body={detail}"
                )
            # 与排空 task 共用同一迭代器:probe 帧之后的剩余帧由 task 续读
            sse_data = _iter_sse_data(response.content)
            probe = await self._read_probe_from_sse(sse_data)
            if probe.request_id != request_id:
                raise RuntimeError("cloud returned a different request ID")
            # 块粒度不强求一致:混合模型(如 Mamba/SWA)下云 hash 粒度可
            # 小于边 KV 块粒度,由调用方按 probe.block_size 学习重建链
            # (见 _lwd_hash_bs_by_cloud)。
            if probe.hit_tokens > len(prompt_token_ids):
                raise RuntimeError("cloud prefix hit exceeds the prompt length")
        except BaseException:
            response.close()
            raise
        # probe 到手即返回,但**不能**在这里关流:云侧 handler 要等推理完结
        # 才发 usage 末 chunk(最长 600s),边侧提前断开会让网关记成
        # client_gone/context canceled,真实 token 用量丢失、改按文本估算。
        # 剩余帧交后台 task 排空(同条 loop,不占 _job 返回路径)。
        task = asyncio.create_task(
            self._drain_stream(request_id, response, sse_data)
        )
        self._drain_tasks[request_id] = task
        task.add_done_callback(
            lambda done, rid=request_id: self._finish_drain(rid, done)
        )

        logger.info(
            "[Lwd][edge-client] negotiate req=%s hit_tokens=%d cloud_id=%s",
            request_id, probe.hit_tokens, probe.cloud_id,
        )
        return probe

    def _finish_drain(self, request_id: str, task: Any) -> None:
        """同 id 重发时旧 task 不摘除新 task 的登记。"""
        if self._drain_tasks.get(request_id) is task:
            self._drain_tasks.pop(request_id, None)

    async def _drain_stream(
        self,
        request_id: str,
        response: Any,
        sse_data: AsyncIterator[str],
    ) -> None:
        """排空探针流至 usage 末 chunk:供网关计费的收流路径。

        本侧只留痕不消费(记账以云侧 C2e 回边的 ``[Lwd][edge] usage`` 为准),
        但必须读完:读完才让云侧 handler 正常走到末 chunk + [DONE]。
        云侧 usage 等待有 600s 上限,超时也必发空 usage 收流,故 task 数
        有界(= 在飞请求数),无需额外超时钳制。"""
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
        self, sse_data: AsyncIterator[str]
    ) -> LwdProbeResult:
        """从 SSE 流解析首个 edge_cloud_probe 控制消息(delta.content 累积)。"""
        content_buf = ""
        async for payload in sse_data:
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
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
                continue
            if isinstance(message, dict) and message.get("type") == "edge_cloud_probe":
                return LwdProbeResult.from_control_message(message)
        raise RuntimeError("edge-cloud probe control message not found in SSE stream")
