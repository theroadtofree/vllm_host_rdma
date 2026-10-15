# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Prefix-cache coordination config (prefill_only 云侧算力复用).

边侧持 ``control_url`` / ``tenant_key_file``;云侧持 ``listen_host`` /
``listen_port`` / ``instance_id``。校验对齐 prefill_only(要求
``lwd_config.mode == "prefill_only"`` 且 ``enable_prefix_caching``),不依赖
参考分支的 ``pd_separation``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class LwdCoordinationConfig:
    enabled: bool = False
    # 仅放置模式:probe 只承担云内 DP 的引擎放置(纯负载,无 HMAC 建链/
    # 不续算/不上报收尾链),不要求 prefix caching 与租户密钥——供不开
    # 前缀复用但需云内多引擎的部署(GLM dp>1)
    placement_only: bool = False
    # 边
    control_url: str | None = None
    tenant_key_file: str | None = None
    consumer_id: str | None = None
    # 云
    listen_host: str = "0.0.0.0"
    listen_port: int = 8100
    instance_id: str = "cloud-0"
    # 通用
    connect_timeout: float = 10.0
    enforce_mm_abi_match: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any) -> "LwdCoordinationConfig":
        if not isinstance(raw, dict):
            return cls()
        known = {
            "enabled", "placement_only", "control_url", "tenant_key_file",
            "consumer_id", "listen_host", "listen_port", "instance_id",
            "connect_timeout", "enforce_mm_abi_match",
        }
        kwargs = {k: v for k, v in raw.items() if k in known}
        return cls(**kwargs)

    def validate(
        self, lwd_mode: str, is_edge: bool, enable_prefix_caching: bool
    ) -> None:
        if not self.enabled:
            return
        if lwd_mode != "prefill_only":
            raise ValueError(
                "lwd prefix coordination requires lwd_config.mode "
                "'prefill_only'"
            )
        if not is_edge and not enable_prefix_caching and not self.placement_only:
            # 仅云侧依赖本地前缀缓存(命中/续算);边侧无 KV,不强求,
            # 且边侧开 enable_prefix_caching 会触发 AscendHybridKVCacheCoordinator
            # 对 prefill_only 空 spec 的 >1 组断言崩溃。
            raise ValueError(
                "lwd prefix coordination requires cache_config"
                ".enable_prefix_caching=True on the cloud"
            )
        if is_edge and not self.tenant_key_file and not self.placement_only:
            # 租户密钥只在边侧读取(probe 建链 + 收尾上报生成段链);
            # 云侧不持密钥——这是隐私前提,故仅边侧强制要求。
            raise ValueError(
                "lwd prefix coordination requires tenant_key_file on the edge "
                "(key is read edge-side only; the cloud must not hold it)"
            )
        if not is_edge and self.tenant_key_file:
            logger.warning(
                "[Lwd][coord] tenant_key_file set on the cloud side is "
                "ignored: the tenant key must be held edge-side only"
            )
        if self.control_url and not self.control_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError("control_url must use http:// or https://")
        if not 1 <= self.listen_port <= 65535:
            raise ValueError("listen_port must be between 1 and 65535")
        logger.info(
            "[Lwd][coord] enabled: control_url=%r listen=%s:%s instance=%s",
            self.control_url, self.listen_host, self.listen_port,
            self.instance_id,
        )