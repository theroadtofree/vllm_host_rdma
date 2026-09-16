"""边云共享的配置支撑:LwdConfig(env/config 唯一解析入口)与
is_lwd_prefill_only(模式判定唯一实现)。

两侧引擎装配各自收在自己的子类里(边 LwdEdgeEngineCore / 云
LwdCloudEngineCore),本文件只负责配置解析;内核模块只收 plain 值,
不解析配置、不读 env。

配置源优先级:
  1. 生效配置类 vllm_config.lwd_config(vllm/config/lwd.py,由
     additional_config["lwd_config"] 在 VllmConfig.__post_init__ 解析):
     提供 enabled/role/mode;
  2. additional_config 的 lwd_config 段:传输层字段(pre_out_host 等);
  3. env(VLLM_ASCEND_LWD_*):只覆盖地址/端口/超时/调试开关,坏值
     告警后保留默认,不抛。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
    LWD_PUBLISH_QUEUE_MAX,
)

logger = init_logger(__name__)

LWD_PRE_OUT_PORT_DEFAULT = 5558
LWD_POST_OUT_PORT_DEFAULT = LWD_PRE_OUT_PORT_DEFAULT + 1
# 等云首拍 HELLO 的预算:HELLO 在云引擎全量初始化(权重/KV/图编译
# capture)完成后才发出,预算须覆盖云的全量启动时长(边不开图、
# 云开图是本场景固定形态),默认 600s;部署可经 hello_timeout_s/env 覆盖
LWD_HELLO_TIMEOUT_S_DEFAULT = 600.0

_LWD_CONFIG_SECTION = "lwd_config"
# 传输层字段(pre_out_host 等)的历史段名;仅作兼容回退
_LWD_LEGACY_SECTION = "edge_cloud_config"
_LWD_ENV_PREFIX = "VLLM_ASCEND_LWD_"


@dataclass(frozen=True)
class LwdConfig:
    """plain 值配置对象;引擎装配期一次成型。"""

    is_edge_node: bool = True
    pre_out_host: str = "127.0.0.1"
    pre_out_port: int = LWD_PRE_OUT_PORT_DEFAULT
    post_out_port: int = LWD_POST_OUT_PORT_DEFAULT
    post_out_bind: str = "*"
    hello_timeout_s: float = LWD_HELLO_TIMEOUT_S_DEFAULT
    scheduler_name: str = "prefill_first"
    publish_queue_max: int = LWD_PUBLISH_QUEUE_MAX
    debug: bool = False

    def lwd_pre_out_endpoint(self) -> str:
        """PRE_OUT 端点:云侧 bind 地址,随 HELLO 通告给边。"""
        return f"tcp://{self.pre_out_host}:{self.pre_out_port}"

    def lwd_post_out_bind_endpoint(self) -> str:
        """POST_OUT 端点:边侧 bind,云经 master_addr 来连。"""
        return f"tcp://{self.post_out_bind}:{self.post_out_port}"

    @classmethod
    def from_env_and_config(cls, vllm_config) -> LwdConfig:
        """合并生效配置类(角色)、lwd_config 段(传输层字段)与 env
        覆盖(地址/端口/超时/调试)。"""
        section = _lwd_read_section(vllm_config)
        effective = getattr(vllm_config, "lwd_config", None)
        config = cls(
            is_edge_node=effective.is_edge
            if effective is not None
            else str(section.get("role", "edge")) == "edge",
            pre_out_host=str(section.get("pre_out_host", "127.0.0.1")),
            pre_out_port=int(section.get("pre_out_port", LWD_PRE_OUT_PORT_DEFAULT)),
            post_out_port=int(section.get("post_out_port", LWD_POST_OUT_PORT_DEFAULT)),
            post_out_bind=str(section.get("post_out_bind", "*")),
            hello_timeout_s=float(
                section.get("hello_timeout_s", LWD_HELLO_TIMEOUT_S_DEFAULT)
            ),
            scheduler_name=str(section.get("scheduler", "prefill_first")),
            publish_queue_max=int(
                section.get("publish_queue_max", LWD_PUBLISH_QUEUE_MAX)
            ),
            debug=bool(section.get("debug", False)),
        )
        return _lwd_apply_env_overrides(config)


def is_lwd_prefill_only(vllm_config) -> bool:
    """模式判定唯一实现:生效配置类 enabled 且 mode == "prefill_only"
    才生效;配置类缺位时回退读 lwd_config/legacy 段的 mode 字段
    (兼容不经 VllmConfig.__post_init__ 构造的调用方)。"""
    effective = getattr(vllm_config, "lwd_config", None)
    if effective is not None:
        return effective.enabled and effective.mode == "prefill_only"
    section = _lwd_read_section(vllm_config)
    return section.get("mode") == "prefill_only"


def _lwd_read_section(vllm_config) -> dict:
    """取 additional_config 下传输层字段所在段:lwd_config 优先,
    legacy 段回退;缺省/非 dict 均按空段处理。"""
    additional = vllm_config.additional_config or {}
    section = additional.get(_LWD_CONFIG_SECTION)
    if isinstance(section, dict):
        return section
    legacy = additional.get(_LWD_LEGACY_SECTION)
    return legacy if isinstance(legacy, dict) else {}


def _lwd_apply_env_overrides(config: LwdConfig) -> LwdConfig:
    """env 只覆盖部署相关项(地址/端口/超时/调试);坏值告警并保留
    默认,不抛(部署期输入错误不应炸掉引擎启动)。"""
    host = os.getenv(_LWD_ENV_PREFIX + "PRE_OUT_HOST")
    port = _lwd_read_env_int("PRE_OUT_PORT")
    post_port = _lwd_read_env_int("POST_OUT_PORT")
    post_bind = os.getenv(_LWD_ENV_PREFIX + "POST_OUT_BIND")
    hello_timeout = _lwd_read_env_float("HELLO_TIMEOUT_S")
    debug = os.getenv(_LWD_ENV_PREFIX + "DEBUG")
    return LwdConfig(
        is_edge_node=config.is_edge_node,
        pre_out_host=host if host else config.pre_out_host,
        pre_out_port=port if port is not None else config.pre_out_port,
        post_out_port=post_port if post_port is not None else config.post_out_port,
        post_out_bind=post_bind if post_bind else config.post_out_bind,
        hello_timeout_s=(
            hello_timeout if hello_timeout is not None else config.hello_timeout_s
        ),
        scheduler_name=config.scheduler_name,
        publish_queue_max=config.publish_queue_max,
        debug=config.debug or (debug is not None and debug.lower() == "1"),
    )


def _lwd_read_env_int(name: str) -> int | None:
    """读整型 env;缺省/坏值返回 None(坏值告警,调用方保留默认)。"""
    raw = os.getenv(_LWD_ENV_PREFIX + name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("[Lwd] ignore invalid env %s%s=%r", _LWD_ENV_PREFIX, name, raw)
        return None


def _lwd_read_env_float(name: str) -> float | None:
    """读浮点 env;语义与 _lwd_read_env_int 一致。"""
    raw = os.getenv(_LWD_ENV_PREFIX + name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("[Lwd] ignore invalid env %s%s=%r", _LWD_ENV_PREFIX, name, raw)
        return None
