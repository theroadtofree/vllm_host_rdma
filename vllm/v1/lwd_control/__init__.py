"""Lwd prefill-only 控制面内核(仅 vllm 仓):控制面调度与控制面通信,数据面
经既有接缝对接。设计文档与 import 白名单见 docs/refactor/prefill_only_migration.md。"""

from __future__ import annotations


def _lwd_init_registry_and_validate(vllm_config, config) -> None:
    """云侧复用装配前置:加载全场共享 registry(进程级单例)并校验本机
    身份在场(fail-fast);config_digest 供全场一致性核对。1E1C(无
    registry)为 no-op。"""
    if not config.is_cloud_reuse:
        return
    from vllm.logger import init_logger
    from vllm.v1.lwd_control.control_communication.lwd_role_registry import (
        init_role_registry,
    )

    registry = init_role_registry(config.registry_path)
    role = "edge" if config.is_edge_node else "cloud"
    instance_id = config.self_edge_id if config.is_edge_node else config.self_cloud_id
    registry.validate_self(role, instance_id)
    init_logger(__name__).info(
        "[Lwd] cloud-reuse mode: registry digest=%s self=%s id=%d",
        registry.config_digest,
        role,
        instance_id,
    )


def lwd_resolve_engine_cls(vllm_config):
    """core.py 类选择点的分流点:云角色返回 LwdCloudEngineCore,边角色返回
    LwdEdgeEngineCore,其余返回 None(调用方用原生 EngineCoreProc)。

    云侧复用(registry 非空)在此完成 registry 加载 + validate_self,
    引擎装配期直接取单例。"""
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
        LwdConfig,
        is_lwd_prefill_only,
    )

    if not is_lwd_prefill_only(vllm_config):
        return None
    config = LwdConfig.from_env_and_config(vllm_config)
    _lwd_init_registry_and_validate(vllm_config, config)
    from vllm.logger import init_logger

    if config.is_edge_node:
        from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
            LwdEdgeEngineCore,
        )

        init_logger(__name__).info(
            "[Lwd] prefill_only edge: engine class selected (LwdEdgeEngineCore)"
        )
        return LwdEdgeEngineCore
    from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine import (
        LwdCloudEngineCore,
    )

    init_logger(__name__).info(
        "[Lwd] prefill_only cloud: engine class selected (LwdCloudEngineCore)"
    )
    return LwdCloudEngineCore


def lwd_serve_guard(vllm_config) -> None:
    """serve 入口守卫:云角色经 _lwd_cloud_deploy_guard 校验后按
    scheduler_name 注入调度器(mixed 默认混排;prefill_first/decode_first
    旧相位机);云引擎类由子进程内 lwd_resolve_engine_cls 解析,边调度器
    由引擎自注入。

    云侧复用:registry 承担部署校验(身份在场/全场一致),云侧不再要求
    master_addr(ROUTER bind 自身端口,边主动来连);调度器按 scheduler_name
    照常注入。"""
    from vllm.logger import init_logger
    from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_mixed_scheduler import (
        LwdCloudMixedScheduler,
    )
    from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_phase_scheduler import (
        LwdCloudPhaseScheduler,
    )
    from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble import (
        LwdConfig,
        is_lwd_prefill_only,
    )

    if not is_lwd_prefill_only(vllm_config):
        return
    config = LwdConfig.from_env_and_config(vllm_config)
    _lwd_init_registry_and_validate(vllm_config, config)
    if config.is_edge_node:
        return
    if not config.is_cloud_reuse:
        _lwd_cloud_deploy_guard(vllm_config, config)
    vllm_config.scheduler_config.scheduler_cls = (
        LwdCloudPhaseScheduler
        if config.scheduler_name in ("prefill_first", "decode_first")
        else LwdCloudMixedScheduler
    )
    init_logger(__name__).info(
        "[Lwd] prefill_only cloud: %s scheduler injected (construction-time)",
        config.scheduler_name,
    )


def _lwd_cloud_deploy_guard(vllm_config, config) -> None:
    """云角色单机部署校验(fail-fast,serve 入口即拦;仅 1E1C 路径):
    master_addr 非空(POST_OUT 连边必需);pre_out_host 非 0.0.0.0
    (通告值须可路由)。"""
    from vllm.logger import init_logger

    master_addr = vllm_config.parallel_config.master_addr
    if not master_addr:
        raise ValueError(
            "[Lwd] prefill_only cloud requires --master-addr (POST_OUT connect)"
        )
    if config.pre_out_host == "0.0.0.0":
        raise ValueError(
            "[Lwd] prefill_only cloud pre_out_host=0.0.0.0 is not announceable; "
            "set a routable IP (VLLM_ASCEND_LWD_PRE_OUT_HOST)"
        )
    if config.pre_out_host == "127.0.0.1" and master_addr not in (
        "127.0.0.1",
        "localhost",
    ):
        init_logger(__name__).warning(
            "[Lwd] cloud announces pre_out_host=127.0.0.1 but master_addr=%s "
            "is remote; edge will fail to reach PRE_OUT unless same host",
            master_addr,
        )
