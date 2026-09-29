# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""边侧前缀协商客户端(prefill_only 裁剪版)。

从参考分支 ``edge_cloud/edge_client.py`` 裁剪:只保留纯文本 v1 协商
(manifest 影子体 POST + 响应头解析 hit_tokens)。原始 messages/采样参数
不过网;参考分支在探针 SSE 流末 chunk 回 usage 并由调用方后台排空,本裁剪
版探针只读响应头——usage 改挂数据面步元数据通道(见
``LwdEdgeEngineCore._lwd_consume_usage``),消费口径仍是"仅边侧留痕"。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vllm.logger import init_logger
from vllm.v1.lwd_control.control_communication.lwd_prefix import (
    HEADER_EDGE_ID,
    LwdPrefixHasher,
    LwdPrefixManifest,
    LwdProbeResult,
)

logger = init_logger(__name__)


class LwdEdgePrefixClient:
    """边侧准入客户端:HMAC manifest 影子体 + 命中结果解析。"""

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

    def build_manifest(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        media_items: Sequence[Any] = (),
    ) -> LwdPrefixManifest:
        """租户密钥链式 HMAC 产 manifest(原始 prompt 不出边)。

        media_items 把多模态内容身份(identifier digest + 占位区间)
        掺入被覆盖块的 HMAC:同文本同分辨率不同图不再共享盲链块哈希,
        云侧不会跨图误命中。"""
        return self._hasher.build_manifest(
            request_id, prompt_token_ids, media_items=media_items
        )

    def build_control_request(
        self, manifest: LwdPrefixManifest
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """白名单影子体:仅 manifest/stream/prompt 长度,采样参数不过网。"""
        headers = manifest.to_headers()
        headers[HEADER_EDGE_ID] = str(self._edge_id)
        if self._consumer_id:
            headers["X-Mse-Consumer"] = self._consumer_id
        body = {
            "messages": manifest.to_messages(),
            "stream": True,
            "edge_cloud_prompt_tokens": manifest.prompt_tokens,
            "model": "edge-cloud-prefix",
        }
        return headers, body

    async def negotiate(
        self,
        session: Any,
        request_id: str,
        prompt_token_ids: list[int],
        manifest: LwdPrefixManifest | None = None,
        media_items: Sequence[Any] = (),
    ) -> LwdProbeResult:
        """POST 探测并解析响应头命中;失败即抛(fail-closed,由边拒绝请求)。

        命中结果由调用方消费(命中时推进 request.num_computed_tokens,
        使边侧只 embed/发尾巴)。manifest 可传入调用方已建好的链复用
        (避免一次探测两次 O(N) 建链);缺省现建(media_items 须与
        调用方建链口径一致,防缺省路径漏掺多模态身份)。"""
        import aiohttp

        if manifest is None:
            manifest = self.build_manifest(
                request_id, prompt_token_ids, media_items
            )
        headers, body = self.build_control_request(manifest)
        timeout = aiohttp.ClientTimeout(total=self._connect_timeout)
        async with session.post(
            self._control_url, json=body, headers=headers, timeout=timeout
        ) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"prefix negotiation HTTP {response.status}"
                )
            result = LwdProbeResult.from_headers(response.headers)
        logger.info(
            "[Lwd][edge-client] negotiate req=%s hit_tokens=%d",
            request_id, result.hit_tokens,
        )
        return result