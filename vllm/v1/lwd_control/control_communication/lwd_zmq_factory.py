"""控制面 ZMQ 通道装配工厂。

按 ``zmq_mode`` 在两种传输模式间选型:

* ``push_pull``   —— 双向 PUSH/PULL 对(:class:`LwdPushPullChannel`);
  边/云经一次性 TCPStore 交换云侧 IP(单实例)。
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
from vllm.v1.lwd_control.control_communication.lwd_pushpull_channel import (
    LwdPushPullChannel,
)
from vllm.v1.lwd_control.control_communication.lwd_router_dealer_channel import (
    LwdRouterDealerChannel,
)

logger = init_logger(__name__)


def _tcpstore(host_name: str, port: int, is_master: bool, timeout: float):
    """构造一次性 TCPStore 用于边/云启动握手(仅 push_pull 模式)。"""
    import torch.distributed as dist
    from datetime import timedelta

    return dist.TCPStore(
        host_name=host_name,
        port=port,
        world_size=2,
        is_master=is_master,
        timeout=timedelta(seconds=timeout),
    )


def create_edge_zmq_channel(
    *,
    zmq_mode: str,
    master_addr: str,
    master_port: int,
    dp_rank: int,
    pre_out_port: int,
    post_out_port: int,
    name: str,
    instance_count: int = 1,
    encoder: Callable[[Any], bytes] = lwd_encode_notify,
    decoder: Callable[[bytes], Any] = lwd_decode_cloud_notify,
):
    """构造边侧(主控节点)PD 分离控制面 ZMQ 通道。

    Args:
        zmq_mode: ``push_pull`` 或 ``full_duplex``。
        instance_count: full_duplex 下本 ROUTER 服务的云实例总数;push_pull
            仅支持 1。
        encoder: 出站(边 -> 云)编码器。
        decoder: 入站(云 -> 边)解码器。
    """
    if zmq_mode == "push_pull":
        if instance_count > 1:
            raise ValueError(
                "Edge-cloud multi-instance (instance_parallel_size > 1) "
                "requires the full_duplex ZMQ mode: per-instance addressing "
                "needs the ROUTER/DEALER identity envelope."
            )
        _addr_store = _tcpstore(
            master_addr, master_port + 1 + dp_rank, is_master=True, timeout=300
        )
        cloud_addr = _addr_store.get("cloud_ip").decode()
        del _addr_store
        pre_out = f"tcp://*:{pre_out_port + dp_rank * 2}"
        post_out = f"tcp://{cloud_addr}:{post_out_port + dp_rank * 2}"
        return LwdPushPullChannel(
            send_endpoint=pre_out,
            recv_endpoint=post_out,
            send_bind=True,
            recv_bind=False,
            encoder=encoder,
            decoder=decoder,
            name=name,
        )

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
    zmq_mode: str,
    master_addr: str,
    master_port: int,
    dp_rank: int,
    pre_out_port: int,
    post_out_port: int,
    name: str,
    instance_id: int = 0,
    encoder: Callable[[Any], bytes] = lwd_encode_cloud_notify,
    decoder: Callable[[bytes], Any] = lwd_decode_notify,
):
    """构造云侧(从节点)PD 分离控制面 ZMQ 通道。

    Args:
        zmq_mode: ``push_pull`` 或 ``full_duplex``。
        instance_id: full_duplex 下本云实例的 DEALER identity;push_pull 仅
            支持 0。
        encoder: 出站(云 -> 边)编码器。
        decoder: 入站(边 -> 云)解码器。
    """
    if zmq_mode == "push_pull":
        if instance_id > 0:
            raise ValueError(
                "Edge-cloud multi-instance (instance_parallel_size > 1) "
                "requires the full_duplex ZMQ mode: per-instance addressing "
                "needs the ROUTER/DEALER identity envelope."
            )
        from vllm.utils.network_utils import get_ip

        _cloud_ip = get_ip()
        _addr_store = _tcpstore(
            master_addr, master_port + 1 + dp_rank, is_master=False, timeout=600
        )
        _addr_store.set("cloud_ip", _cloud_ip)
        del _addr_store
        post_out_bind = f"tcp://*:{post_out_port + dp_rank * 2}"
        pre_out_connect = f"tcp://{master_addr}:{pre_out_port + dp_rank * 2}"
        return LwdPushPullChannel(
            send_endpoint=post_out_bind,
            recv_endpoint=pre_out_connect,
            send_bind=True,
            recv_bind=False,
            encoder=encoder,
            decoder=decoder,
            name=name,
        )

    # full_duplex(默认):云 DEALER 连边地址,并以实例 id 作为显式 identity。
    connect_addr = f"tcp://{master_addr}:{pre_out_port + dp_rank}"
    return LwdRouterDealerChannel(
        endpoint=connect_addr,
        bind=False,
        encoder=encoder,
        decoder=decoder,
        name=name,
        identity=str(instance_id),
    )
