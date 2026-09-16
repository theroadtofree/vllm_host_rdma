"""控制面双向通道:一个 PUSH 发布端 + 一个 PULL 订阅端组合成的对称通道。

只认方向(send/recv)不认边/云;bind/connect 角色由装配层(工厂)显式给定,
编码器/解码器经构造注入,不绑定具体消息类型。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vllm.logger import init_logger
from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
    LWD_PUBLISH_QUEUE_MAX,
    LwdControlPublisher,
)
from vllm.v1.lwd_control.control_communication.lwd_control_subscriber import (
    LwdControlSubscriber,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    lwd_decode_notify,
    lwd_encode_notify,
)

logger = init_logger(__name__)


class LwdPushPullChannel:
    """双向 PUSH/PULL 控制面通道(side-agnostic)。

    组合一个 :class:`LwdControlPublisher`(send 方向)与一个
    :class:`LwdControlSubscriber`(recv 方向)。两侧的 bind/connect 角色由
    ``send_bind`` / ``recv_bind`` 显式给定,不靠 endpoint 里的 ``*`` 猜测
    角色,交给工厂(装配层)决定。

    载荷完全透明:``encoder`` / ``decoder`` 注入,不 import 调度/执行类,
    也不依赖具体消息类型。
    """

    def __init__(
        self,
        send_endpoint: str,
        recv_endpoint: str,
        *,
        send_bind: bool,
        recv_bind: bool,
        encoder: Callable[[Any], bytes] = lwd_encode_notify,
        decoder: Callable[[bytes], Any] = lwd_decode_notify,
        queue_max: int = LWD_PUBLISH_QUEUE_MAX,
        name: str = "lwd-pushpull",
    ) -> None:
        self._name = name
        self._publisher = LwdControlPublisher(
            send_endpoint,
            bind=send_bind,
            queue_max=queue_max,
            encoder=encoder,
        )
        self._subscriber = LwdControlSubscriber(
            recv_endpoint,
            bind=recv_bind,
            decoder=decoder,
        )
        logger.info(
            "[Lwd][zmq] pushpull channel[%s] up: send=%s(bind=%s) "
            "recv=%s(bind=%s)",
            name,
            send_endpoint,
            send_bind,
            recv_endpoint,
            recv_bind,
        )

    @property
    def closed(self) -> bool:
        """任一方向关停即视为通道关闭。"""
        return self._publisher.closed or self._subscriber.closed

    def publish(self, msg: Any) -> bool:
        """入队一条消息由 PUSH 端发送;False = 队满未发(背压,不丢已发)。"""
        return self._publisher.publish(msg)

    def consume_new_outputs(self) -> list[Any]:
        """取出并清空自上次调用以来收到的全部消息(FIFO 序)。"""
        outputs: list[Any] = []
        while True:
            msg = self._subscriber.recv(timeout_ms=0)
            if msg is None:
                break
            outputs.append(msg)
        return outputs

    def shutdown(self) -> None:
        """关停(幂等):先关 send 端再关 recv 端。"""
        self._publisher.shutdown()
        self._subscriber.shutdown()
