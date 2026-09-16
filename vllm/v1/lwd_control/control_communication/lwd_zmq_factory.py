"""控制面 ZMQ 通道装配工厂。

* ``full_duplex`` —— 单条 ROUTER/DEALER 连接
  (:class:`LwdRouterDealerChannel`);边 bind、云 connect,云以显式
  ``identity``(实例 id)寻址,支持一 ROUTER 服务 N 个云实例。

本模块只接受 plain 参数(端口/模式/编解码器),不解析配置、不读 env,也不绑定
具体配置类;端侧角色与 bind/connect 角色由调用方(引擎装配层)决定。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vllm.logger import init_logger
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    lwd_decode_cloud_notify,
    lwd_decode_notify,
    lwd_encode_cloud_notify,
    lwd_encode_notify,
)
from vllm.v1.lwd_control.control_communication.lwd_router_dealer_channel import (
    LwdRouterDealerChannel,
)

logger = init_logger(__name__)

def create_edge_zmq_channel(
    *,
    dp_rank: int,
    pre_out_port: int,
    name: str,
    instance_count: int = 1,
    encoder: Callable[[Any], bytes] = lwd_encode_notify,
    decoder: Callable[[bytes], Any] = lwd_decode_cloud_notify,
):
    """构造边侧(主控节点)PD 分离控制面 ZMQ 通道。

    Args:
        instance_count: full_duplex 下本 ROUTER 服务的云实例总数;
        encoder: 出站(边 -> 云)编码器。
        decoder: 入站(云 -> 边)解码器。
    """

    # full_duplex(默认):边 bind 单 ROUTER 端口,云按实例 id 连入。
    bind_addr = f"tcp://*:{pre_out_port + dp_rank}"
    return LwdRouterDealerChannel(
        endpoint=bind_addr,
        bind=True,
        encoder=encoder,
        decoder=decoder,
        name=name,
        expected_instances=instance_count,
    )


def create_cloud_zmq_channel(
    *,
    master_addr: str,
    dp_rank: int,
    pre_out_port: int,
    name: str,
    instance_id: int = 0,
    encoder: Callable[[Any], bytes] = lwd_encode_cloud_notify,
    decoder: Callable[[bytes], Any] = lwd_decode_notify,
):
    """构造云侧(从节点)PD 分离控制面 ZMQ 通道。

    Args:
        instance_id: full_duplex 下本云实例的 DEALER identity;
        encoder: 出站(云 -> 边)编码器。
        decoder: 入站(边 -> 云)解码器。
    """

    # 云 DEALER 连边地址,并以实例 id 作为显式 identity。
    connect_addr = f"tcp://{master_addr}:{pre_out_port + dp_rank}"
    return LwdRouterDealerChannel(
        endpoint=connect_addr,
        bind=False,
        encoder=encoder,
        decoder=decoder,
        name=name,
        identity=str(instance_id),
    )
