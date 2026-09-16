"""控制面全双工通道(ROUTER/DEALER):单条 TCP 连接承载双向控制面消息。

ROUTER 侧(bind)可服务 N 个 DEALER 实例,按每个 DEALER 的显式 ``identity``
(实例 id)做逐实例寻址;载荷经 ``encoder`` / ``decoder`` 注入,不绑定具体
消息类型。

与 push_pull 通道不同:本通道两端共用同一条连接、同一个 socket,因此发送方向
与接收方向往往携带不同类型,``encoder`` / ``decoder`` 必须由装配层(工厂)
按端侧角色显式注入,无缺省值。
"""

from __future__ import annotations

import contextlib
import errno
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import zmq
from vllm.logger import init_logger

logger = init_logger(__name__)

_EMPTY_FRAME = b""
_HELLO_FRAME = b"HELLO"
# ROUTER -> DEALER 对 HELLO 的应答。一个从未收到 WELCOME 的 DEALER 是
# 被饿死的重复 identity:libzmq 对重复 identity 只保留第一条连接并静默丢弃
# 后续连接的所有帧(含 HELLO),所以该 ack 是应用层唯一可见的饥饿证据。
_WELCOME_FRAME = b"WELCOME"
# 旧版握手帧(兼容混合版本边/云对)。
_READY_FRAME = b"READY"
_SEQ_BYTES = 8
# ROUTER_MANDATORY 下不可路由的发送,libzmq 以 EHOSTUNREACH 上报;POSIX 值
# 110 在 Windows 上 errno 不同,两者都收,保证"identity 未就绪 -> 暂存重试"
# 分支跨平台可用。
_UNROUTABLE_ERRNOS = {110, errno.EHOSTUNREACH}


def _instance_identity(instance_id: int) -> bytes:
    """云实例 ``instance_id`` 的线级 identity。

    dp 已由每个 ROUTER 绑定的端口分离,identity 只需编码实例维度;
    (dp, instance) 组合在跨 socket 间全局唯一。
    """
    return str(int(instance_id)).encode()


class LwdRouterDealerChannel:
    """单条 TCP 连接上的全双工 ZMQ 通道(ROUTER/DEALER)。

    边(bind)起一个 ``zmq.ROUTER`` 服务至多 ``expected_instances`` 个云
    DEALER;云(connect)以一个显式 ``identity``(实例 id 字符串)连接
    ``zmq.DEALER``。每个 DEALER 连接时发一次性 ``HELLO`` 握手,ROUTER 据此
    学习可寻址的 peer identity;identity 未就绪前,该实例的出站消息暂存在其
    专属 FIFO 中而非丢弃。
    """

    SHUTDOWN_TIMEOUT: float = 2.0
    SEND_HWM: int = 1000
    # 单实例暂存(已编码)出站积压上限:未就绪或慢实例不得让边侧内存无限增长。
    PER_INSTANCE_PENDING_LIMIT: int = 1000
    POLL_TIMEOUT_MS: int = 1
    # DEALER 已发 HELLO 却在此时长内未收到 WELCOME,即被饿死的重复 identity
    # (两个云进程共享一个 --instance-parallel-start-rank)。
    WELCOME_ACK_TIMEOUT: float = 30.0

    def __init__(
        self,
        endpoint: str,
        *,
        bind: bool,
        encoder: Callable[[Any], bytes],
        decoder: Callable[[bytes], Any],
        identity: str | None = None,
        expected_instances: int = 1,
        name: str = "lwd-router-dealer",
    ) -> None:
        self._name = name
        self._endpoint = endpoint
        self._running = True
        self._seq = 0
        self._is_router = bind
        self._expected_instances = max(1, int(expected_instances))
        self._encoder = encoder
        self._decoder = decoder
        # 每 peer identity 的暂存(已编码)出站帧,按 FIFO。仅 ROUTER 侧使用:
        # 尚未 HELLO(或 peer HWM 已满)的 identity 把消息暂存于此。
        self._identity_pending: dict[bytes, deque[tuple[bytes, bytes]]] = {}
        # 已收到 HELLO 的 identity 集合(就绪表)。
        self._known_identities: set[bytes] = set()
        self._send_queue: queue.Queue[tuple[int, int, Any]] = queue.Queue(
            maxsize=self.SEND_HWM
        )
        # 已编码、等待 socket 可写的出站消息(HWM 背压重试);避免阻塞发送时
        # 重复序列化载荷。
        self._pending_send: tuple[bytes, bytes] | None = None
        self._received_outputs: list[Any] = []
        self._recv_lock = threading.Lock()

        self._ctx = zmq.Context()
        if self._is_router:
            self._socket = self._ctx.socket(zmq.ROUTER)
            # 对未连接 identity 的定向发送须报错(EHOSTUNREACH)而非被 ZMQ
            # 静默丢弃;actor 捕获后暂存消息等待重试。
            self._socket.set(zmq.ROUTER_MANDATORY, 1)
            self._socket.bind(endpoint)
            self._handshake_pending = False
        else:
            self._socket = self._ctx.socket(zmq.DEALER)
            self._identity_str = str(identity) if identity is not None else ""
            if identity is not None:
                self._socket.set(zmq.IDENTITY, str(identity).encode())
            self._socket.connect(endpoint)
            self._handshake_pending = True
            # identity 冲突看门狗:ROUTER 以 WELCOME 应答 HELLO;若始终未到,
            # 说明另一进程已先占用本 identity。
            self._welcome_received = False
            self._welcome_warned = False
            self._welcome_deadline: float | None = None
        self._socket.set_hwm(self.SEND_HWM)
        logger.info(
            "[Lwd][zmq] router-dealer channel[%s] %s on %s (%s%s)",
            name,
            "bound" if bind else "connecting to",
            endpoint,
            "ROUTER" if bind else "DEALER",
            (
                f", expects {self._expected_instances} instance identities"
                if bind
                else f", identity={identity}"
            ),
        )

        self._thread = threading.Thread(
            target=self._io_thread,
            daemon=True,
            name=f"lwd-router-dealer-{name}",
        )
        self._thread.start()

    @property
    def closed(self) -> bool:
        """是否已关停。"""
        return not self._running

    def publish(self, msg: Any, instance_id: int = 0) -> bool:
        """入队一条消息发往对端;False = 队满未发(背压,不丢已发)。

        ``instance_id`` 仅在 ROUTER 侧用于选择目标云实例(消息进入该实例的
        专属队列);DEALER 侧恒为单 peer,忽略该参数。
        """
        if not self._running:
            return False
        try:
            seq = self._seq
            self._seq += 1
            self._send_queue.put_nowait((instance_id, seq, msg))
        except queue.Full:
            logger.warning(
                "[Lwd][zmq] router-dealer channel[%s] send queue full, "
                "dropping message",
                self._name,
            )
            return False
        return True

    def ready_instance_ids(self) -> set[int]:
        """已收到 HELLO 握手的实例 id 集合(ROUTER 侧)。

        调度器可用其避免向尚未完成连接的云实例派发工作。
        """
        return {
            int(identity.decode())
            for identity in self._known_identities
            if identity.decode().isdigit()
        }

    def consume_new_outputs(self) -> list[Any]:
        """取出并清空自上次调用以来收到的全部消息(FIFO 序)。"""
        with self._recv_lock:
            outputs = self._received_outputs
            self._received_outputs = []
            return outputs

    def _io_thread(self) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while self._running:
            # DEALER 连接后自报身份,ROUTER 才能寻址;DEALER 会在连接建立前
            # 缓冲该帧,因此不会丢失。
            if self._handshake_pending:
                self._send_handshake()
            self._check_welcome_watchdog()
            self._flush_send_queue()
            # 有发送积压时非阻塞轮询,立即回到发送排空;否则短暂阻塞等入站。
            # 尚未学到任何 peer identity 的 ROUTER 不得忙等:阻塞在 poll 上
            # 直到握手到达。
            backlog = (
                not self._send_queue.empty()
                or self._pending_send is not None
                or any(self._identity_pending.values())
            )
            can_send = (not self._is_router) or bool(self._known_identities)
            timeout = 0 if (backlog and can_send) else self.POLL_TIMEOUT_MS
            try:
                events = dict(poller.poll(timeout=timeout))
            except Exception:
                if not self._running:
                    break
                logger.exception(
                    "[Lwd][zmq] router-dealer channel[%s] IO thread error",
                    self._name,
                )
                time.sleep(0.01)
                continue
            if events.get(self._socket, 0) & zmq.POLLIN:
                self._recv_available()

    def _check_welcome_watchdog(self) -> None:
        """DEALER 侧 identity 冲突检测器。

        libzmq 对重复 ROUTER identity 的处理是静默饿死后连接:其 HELLO 到
        不了 ROUTER,ROUTER 也看不到冲突。但被饿死的重复连接同样收不到
        WELCOME ack——这只能在 DEALER 侧观察到。"""
        if (
            self._is_router
            or self._welcome_received
            or self._welcome_warned
            or self._welcome_deadline is None
        ):
            return
        if time.monotonic() < self._welcome_deadline:
            return
        self._welcome_warned = True
        logger.error(
            "[Lwd][zmq] router-dealer channel[%s] HELLO 已发送但边侧 ROUTER "
            "未应答(%.0fs 内无 WELCOME)。几乎可断定另一进程已先用相同 "
            "identity=%s 连接:libzmq 会静默饿死后续的重复 identity 连接。"
            "本实例收不到任何消息——请检查每个云实例都传了互不相同的 "
            "--instance-parallel-start-rank。",
            self._name,
            self.WELCOME_ACK_TIMEOUT,
            self._identity_label(),
        )

    def _identity_label(self) -> str:
        return getattr(self, "_identity_str", None) or "<unset>"

    def _send_handshake(self) -> None:
        """发送一次性 DEALER -> ROUTER 的 HELLO 握手。"""
        try:
            self._socket.send_multipart(
                [_EMPTY_FRAME, _HELLO_FRAME], flags=zmq.NOBLOCK
            )
        except zmq.Again:
            return
        except Exception:
            if self._running:
                logger.exception(
                    "[Lwd][zmq] router-dealer channel[%s] handshake error",
                    self._name,
                )
            return
        self._handshake_pending = False
        if self._welcome_deadline is None:
            self._welcome_deadline = time.monotonic() + self.WELCOME_ACK_TIMEOUT
        logger.info(
            "[Lwd][zmq] router-dealer channel[%s] HELLO handshake sent",
            self._name,
        )

    def _flush_send_queue(self) -> None:
        # 阶段 1:把自上次 tick 以来入队的所有消息编码,ROUTER 侧放入每实例
        # FIFO,DEALER 侧直接上单槽 in-flight。
        while self._running:
            if not self._is_router and self._pending_send is not None:
                # DEALER 只有一个 in-flight 槽;槽内重试未排空前不再编码新消息。
                break
            try:
                instance_id, seq, msg = self._send_queue.get_nowait()
            except queue.Empty:
                break
            try:
                data = self._encoder(msg)
            except Exception:
                logger.exception(
                    "[Lwd][zmq] router-dealer channel[%s] encode failed",
                    self._name,
                )
                continue
            if self._is_router:
                identity = _instance_identity(instance_id)
                pending = self._identity_pending.setdefault(identity, deque())
                if len(pending) >= self.PER_INSTANCE_PENDING_LIMIT:
                    # 未就绪或慢实例高水位:丢弃会丢失消息,故大声记录——正常
                    # 部署不应到达此上限。
                    logger.error(
                        "[Lwd][zmq] router-dealer channel[%s] instance %s "
                        "pending queue at limit (%d); dropping message",
                        self._name,
                        identity.decode(),
                        self.PER_INSTANCE_PENDING_LIMIT,
                    )
                    continue
                pending.append((seq.to_bytes(_SEQ_BYTES, "big"), data))
            else:
                self._pending_send = (seq.to_bytes(_SEQ_BYTES, "big"), data)
        # 阶段 2:排空每实例 FIFO(ROUTER)/ 单 in-flight 槽(DEALER)。
        if self._is_router:
            for identity in list(self._identity_pending.keys()):
                self._flush_identity(identity)
        elif self._pending_send is not None:
            self._flush_dealer()

    def _flush_identity(self, identity: bytes) -> None:
        """发送某实例暂存消息,直到 EAGAIN / 未知 identity。"""
        pending = self._identity_pending.get(identity)
        if not pending:
            self._identity_pending.pop(identity, None)
            return
        while self._running and pending:
            seq_bytes, data = pending[0]
            try:
                self._socket.send_multipart(
                    [identity, _EMPTY_FRAME, seq_bytes, data],
                    flags=zmq.NOBLOCK,
                )
            except zmq.Again:
                # peer HWM 已满:仅背压该 identity。
                return
            except zmq.ZMQError as e:
                if e.errno in _UNROUTABLE_ERRNOS:
                    # identity 尚未连接(或已断):暂存消息直到其 HELLO 到达。
                    return
                if self._running:
                    logger.exception(
                        "[Lwd][zmq] router-dealer channel[%s] send to "
                        "instance %s failed",
                        self._name,
                        identity.decode(),
                    )
                pending.popleft()
                continue
            except Exception:
                if self._running:
                    logger.exception(
                        "[Lwd][zmq] router-dealer channel[%s] send failed",
                        self._name,
                    )
                pending.popleft()
                continue
            pending.popleft()
        if not pending:
            self._identity_pending.pop(identity, None)

    def _flush_dealer(self) -> None:
        assert self._pending_send is not None
        frames = [_EMPTY_FRAME, self._pending_send[0], self._pending_send[1]]
        try:
            self._socket.send_multipart(frames, flags=zmq.NOBLOCK)
        except zmq.Again:
            # peer HWM 已满(或尚未连接):下一 tick 重试。
            return
        except Exception:
            if self._running:
                logger.exception(
                    "[Lwd][zmq] router-dealer channel[%s] send failed",
                    self._name,
                )
        self._pending_send = None

    def _send_welcome(self, identity: bytes) -> None:
        """ROUTER -> DEALER 对 HELLO 的应答。尽力而为的 NOBLOCK 发送。"""
        try:
            self._socket.send_multipart(
                [identity, _EMPTY_FRAME, _WELCOME_FRAME], flags=zmq.NOBLOCK
            )
        except zmq.Again:
            logger.debug(
                "[Lwd][zmq] router-dealer channel[%s] WELCOME for identity "
                "%s deferred (connection back-pressured)",
                self._name,
                identity.decode(),
            )
        except Exception:
            if self._running:
                logger.exception(
                    "[Lwd][zmq] router-dealer channel[%s] failed to send "
                    "WELCOME to identity %s",
                    self._name,
                    identity.decode(),
                )

    def _recv_available(self) -> None:
        try:
            frames = self._socket.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            return
        except Exception:
            if self._running:
                logger.exception(
                    "[Lwd][zmq] router-dealer channel[%s] recv failed",
                    self._name,
                )
            return
        if not frames:
            return
        # ROUTER 收到 [identity, empty, ...];DEALER 收到 [empty, ...]。
        # 握手在 ROUTER 侧为 [identity, empty, HELLO/READY](3 帧)。
        if (
            self._is_router
            and len(frames) == 3
            and frames[2] in (_HELLO_FRAME, _READY_FRAME)
        ):
            identity = frames[0]
            is_new = identity not in self._known_identities
            self._known_identities.add(identity)
            if is_new:
                logger.info(
                    "[Lwd][zmq] router-dealer channel[%s] cloud instance "
                    "connected (identity=%s, %d/%d ready)",
                    self._name,
                    identity.decode(),
                    len(self._known_identities),
                    self._expected_instances,
                )
            else:
                logger.warning(
                    "[Lwd][zmq] router-dealer channel[%s] identity=%s "
                    "HELLO 重复:云实例已重连(重启或网络抖动),其暂存消息恢复投递",
                    self._name,
                    identity.decode(),
                )
            self._send_welcome(identity)
            return
        # DEALER 收到 ROUTER 的 WELCOME 应答:[empty, WELCOME]。
        if (
            not self._is_router
            and len(frames) == 2
            and frames[1] in (_WELCOME_FRAME, _READY_FRAME)
        ):
            self._welcome_received = True
            logger.info(
                "[Lwd][zmq] router-dealer channel[%s] edge ROUTER confirmed "
                "identity=%s (WELCOME received)",
                self._name,
                self._identity_label(),
            )
            return
        seq_bytes, data = frames[-2], frames[-1]
        if len(seq_bytes) != _SEQ_BYTES:
            logger.warning(
                "[Lwd][zmq] router-dealer channel[%s] dropping malformed "
                "message (frames=%d)",
                self._name,
                len(frames),
            )
            return
        try:
            msg = self._decoder(data)
        except Exception:
            logger.warning(
                "[Lwd][zmq] router-dealer channel[%s] drop malformed frame",
                self._name,
            )
            return
        with self._recv_lock:
            self._received_outputs.append(msg)

    def shutdown(self) -> None:
        """关停(幂等):停 IO 线程后关 socket(linger=0 立即丢弃未发帧)。"""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)
        with contextlib.suppress(Exception):
            self._socket.close(linger=0)
