"""传输层方向原语:INBOUND 订阅端(side-agnostic);内部搬运线程收包解码
入列表,消费方经 recv_batch 泵取并自行分发——搬运线程不跑任何业务逻辑,
门池/registry 等引擎状态归调用方泵线程独占。

线程化对齐 LwdRouterDealerChannel 的消费模型(IO 线程只搬运 +
consume_new_outputs 泵取):poll 即时唤醒,投递延迟为线程唤醒量级;
空闲 poll 超时节拍仅作关停响应上限,与投递延迟无关。"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import msgspec
import zmq

from vllm.logger import init_logger
from vllm.v1.lwd_control.control_communication.lwd_control_communicator import (
    LwdControlCommunicator,
)

logger = init_logger(__name__)

# 搬运线程空闲唤醒节拍:仅作关停(closed)响应上限;消息到达由 poll
# 即时返回,不构成投递延迟
_LWD_RECV_POLL_TIMEOUT_MS = 5000


class LwdControlSubscriber:
    """控制面订阅端;内部搬运线程 poll->recv->解码->入列表,新消息到达
    即 set 通知事件。消费方 recv_batch 阻塞泵取一批并分发;关停后泵取
    返回空表(以 closed 属性区分)。"""

    SHUTDOWN_TIMEOUT: float = 2.0

    def __init__(
        self,
        endpoint: str,
        *,
        bind: bool,
        decoder: Callable[[bytes], Any],
    ) -> None:
        self._closed = False
        self._decoder = decoder
        self._communicator = LwdControlCommunicator(endpoint, zmq.PULL, bind=bind)
        self._lock = threading.Lock()
        self._received: list[Any] = []
        self._notify = threading.Event()
        self._thread = threading.Thread(
            target=self._receive_thread,
            daemon=True,
            name=f"lwd-subscriber-{endpoint}",
        )
        # 线程最后启动:全部自有状态就绪后才开始收包
        self._thread.start()

    @property
    def closed(self) -> bool:
        """已关停为 True;泵取返回空表时以此区分超时与关停。"""
        return self._closed

    def recv_batch(self, timeout_ms: int | None = None) -> list[Any]:
        """阻塞等一批新消息并取走清空(FIFO 序);超时/关停返回空表。

        wait -> clear -> 取走 的次序保证不丢唤醒:晚于取走的 append 其
        set 保留在事件上,下一次 wait 立即返回;clear 与取走之间的 append
        被本次取走覆盖。"""
        self._notify.wait(
            timeout_ms / 1000.0 if timeout_ms is not None else None
        )
        self._notify.clear()
        with self._lock:
            msgs = self._received
            self._received = []
            return msgs

    def _receive_thread(self) -> None:
        """搬运线程体:收包->解码(坏包丢弃)->入列表->set 通知。"""
        communicator = self._communicator
        while not self._closed:
            try:
                if not communicator.poll(_LWD_RECV_POLL_TIMEOUT_MS):
                    continue
                data = communicator.recv(block=False)
            except zmq.ZMQError:
                # ETERM(关停打断)与 NOBLOCK 竞态 Again 都走此出口
                if self._closed:
                    break
                logger.warning("[Lwd] subscriber recv interrupted, retrying")
                continue
            try:
                msg = self._decoder(data)
            except (msgspec.DecodeError, msgspec.ValidationError):
                logger.warning("[Lwd] drop malformed notify frame")
                continue
            with self._lock:
                self._received.append(msg)
            self._notify.set()

    def shutdown(self) -> None:
        """关停(幂等):close 释放 fd,term 使搬运线程以 ETERM 退出,
        随后 join 收尾。"""
        if self._closed:
            return
        self._closed = True
        self._communicator.close()
        self._communicator.terminate()
        if self._thread.is_alive():
            self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)
