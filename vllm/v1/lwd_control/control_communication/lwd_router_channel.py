"""ROUTER/ROUTER 全双工控制面单通道(云侧复用,多边多云)。

自 ``vllm-ascend/v1/engine/pp_scheduler_zmq_router.py`` 的
``PPSchedulerZmqRouterDealerChannel`` 移植,定位为**通用通信类**:IO 线程
复用、HELLO/WELCOME 握手、per-identity 滞留、ROUTER_MANDATORY、
fair-queue 接收等机制与原类同构,命名保持原类命名;载荷编解码经构造
注入(``decoder``),通道本身不感知消息类型。

两种使用形态(socket 类型由**上层装配**选择):

1. **通用形态(ROUTER-DEALER,与原 pp 类一致)**:``bind=True`` 一侧
   ROUTER bind 单端口服务 N 个 connect 侧;connect 侧 DEALER(单端点,
   构造即连,publish 无需 instance_id 寻址,消息由 libzmq 缓冲至建连,
   发送侧无信封)。原 pp 调度输出交换即此形态。
2. **云侧复用形态(ROUTER-ROUTER)**:connect 侧传 ``connect_as_router=
   True``,同样是 ROUTER(一个 ROUTER socket 允许 connect 多个
   endpoint),由上层逐对端 ``connect()`` + ``announce()``——双向选路
   均靠 identity 首帧(ZMQ 级),无 DEALER round-robin 混流问题,消息
   来源由信封 identity 免费携带,``consume_new_outputs()`` 输出
   ``list[(对端id, notify)]``。

线上帧格式::

    ROUTER->ROUTER  [peer_id, b"", seq8, payload]   对端收 [己方id, b"", seq8, payload]
    DEALER->ROUTER  [b"", seq8, payload]             ROUTER 收 [dealer_id, b"", seq8, payload]
    ROUTER->DEALER  [dealer_id, b"", seq8, payload]  DEALER 收 [b"", seq8, payload]
    握手(connect侧发) [peer_id, b"", b"HELLO"](ROUTER)/ [b"", b"HELLO"](DEALER)
    应答(bind侧回)   [peer_id, b"", b"WELCOME"]

identity 方案兼容两套:``edge{n}``/``cloud{n}`` 前缀(云侧复用,双向
寻址)与纯十进制 ``str(instance_id)``(原 pp 类 DEALER 形态)。

**publish 不丢消息**(相对原 pp 类的唯一语义适配,消息格式不变):pp 类
发送桥队列满时丢弃(调度输出可容忍);本通道承载请求通告(不可丢),
``publish(notify, instance_id)`` 在 per-identity 待发 FIFO 上限
(``PER_INSTANCE_PENDING_LIMIT``)内不丢,达上限返回 False,由调用方沿用
既有退避重试语义。

握手(HELLO/WELCOME)承担两个职责——
* **就绪表**:connect 侧发 ``HELLO``,bind 侧回 ``WELCOME`` 并把对端
  identity 记入就绪表。未就绪对端的消息滞留 per-identity FIFO,
  就绪后按序补投。
* **重复 identity 检测**:libzmq 对同一 ROUTER socket 上的重复 identity
  静默饿死后连者,被饿死者收不到 ``WELCOME``;``WELCOME_ACK_TIMEOUT``
  内未收到即打 error 告警(与 ``validate_self`` 构成配置错误双保险)。

seq8 为通道层全局递增序号(诊断/排序用);应用层 per-(边,云)对的
seqno 契约(预告 seqno / down_seqno)不因共享连接改变。

ZMQ socket 不允许跨线程访问:收发都在单 IO 线程内复用(pp 通道
``_io_thread`` 模式),调用方经 ``publish``/``consume_new_outputs`` 与
IO 线程交互;connect 侧的 ``connect()``/``announce()``(仅 ROUTER
形态)收敛在 ``start()`` 之前完成。
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

import msgspec
import zmq

from vllm.logger import init_logger

logger = init_logger(__name__)

_EMPTY_FRAME = b""
_HELLO_FRAME = b"HELLO"
_WELCOME_FRAME = b"WELCOME"
_SEQ_BYTES = 8
# 身份前缀:云侧复用 identity 公式 edge{id} / cloud{id}
_EDGE_PREFIX = "edge"
_CLOUD_PREFIX = "cloud"
# libzmq reports an unroutable mandatory ROUTER send as EHOSTUNREACH using
# the POSIX value (110) even on Windows; accept both so the "identity not
# connected yet -> hold and retry" branch works everywhere.
_UNROUTABLE_ERRNOS = {110, errno.EHOSTUNREACH}


def _instance_identity(prefix: str, instance_id: int, engine: int = 0) -> bytes:
    """对端线上 identity:带前缀 ``edge{e}``/``cloud{c}``;前缀为空时
    退化为原 pp 类的纯十进制 ``str(instance_id)``。

    云内 DP:云引擎 identity 为 ``cloud{c}e{i}``(engine=0 时不带后缀,
    与单引擎线上格式逐值一致)——同一边 socket 上同云两引擎必须用不同
    identity,否则 libzmq 重复 identity 静默饿死后连者。"""
    if prefix == _CLOUD_PREFIX and engine > 0:
        return f"{prefix}{instance_id}e{engine}".encode()
    return f"{prefix}{instance_id}".encode()


def _parse_identity_ex(identity: bytes) -> tuple[str, int, int] | None:
    """解析信封 identity -> (前缀, id, engine);不认识的格式返回 None。

    兼容两套方案:``edge{n}``/``cloud{c}`` 前缀与纯十进制(前缀 "");
    云内 DP 的云引擎后缀 ``cloud{c}e{i}``(engine 缺省 0)。"""
    try:
        text = identity.decode()
    except UnicodeDecodeError:
        return None
    for prefix in (_EDGE_PREFIX, _CLOUD_PREFIX):
        if text.startswith(prefix):
            rest = text[len(prefix):]
            engine = 0
            if prefix == _CLOUD_PREFIX and "e" in rest:
                head, _, tail = rest.partition("e")
                if head.isdigit() and tail.isdigit():
                    return prefix, int(head), int(tail)
            if rest.isdigit():
                return prefix, int(rest), engine
    if text.isdigit():
        return "", int(text), 0
    return None


def _parse_identity(identity: bytes) -> tuple[str, int] | None:
    """解析信封 identity -> (前缀, id);不认识的格式返回 None。

    兼容两套方案:``edge{n}``/``cloud{n}`` 前缀与纯十进制(前缀 "")。
    云内 DP 的 ``cloud{c}e{i}`` 后缀在此剥除(engine 维由载荷的
    engine_index 字段承载,信封归账到云 id)。"""
    parsed = _parse_identity_ex(identity)
    if parsed is None:
        return None
    return parsed[0], parsed[1]


def _peer_prefix_of(self_prefix: str) -> str:
    """对端身份前缀与己方相反:边发 cloud{c},云发 edge{e};未知前缀
    (含纯十进制/未设 identity)退化为空前缀(原 pp 类方案)。"""
    if self_prefix == _EDGE_PREFIX:
        return _CLOUD_PREFIX
    if self_prefix == _CLOUD_PREFIX:
        return _EDGE_PREFIX
    return ""


class LwdControlRouterChannel:
    """云侧复用控制面单通道:每进程 1 个 socket + 1 个 IO 线程。

    bind 侧:ROUTER,设 IDENTITY + ROUTER_MANDATORY 后 bind 单端口,
    构造后直接 ``start()``;connect 侧两种形态——DEALER(原 pp 类语义,
    构造即连单端点)或 ROUTER(``connect_as_router=True``,随后逐对端
    ``connect()`` + ``announce()``,一个 ROUTER socket 连 N 台云,最后
    ``start()``)。ZMQ socket 非线程安全,connect/announce 收敛在 IO
    线程启动之前完成。
    """

    SHUTDOWN_TIMEOUT: float = 2.0
    SEND_HWM: int = 1000
    # 单对端待发(编码后)滞留上限:未就绪/慢对端不得无界吃内存;
    # 达上限 publish 返回 False,由调用方退避重试(消息不可丢)
    PER_INSTANCE_PENDING_LIMIT: int = 1000
    POLL_TIMEOUT_MS: int = 1
    # HELLO 后 WELCOME 应答超时:超时即重复 identity(被饿死)或对端
    # 未起,打 error 告警(配置错误双保险之一)
    WELCOME_ACK_TIMEOUT: float = 30.0

    def __init__(
        self,
        bind_endpoint: str | None,
        *,
        bind: bool,
        decoder: Callable[[bytes], Any],
        identity: str | None = None,
        expected_instances: int = 1,
        connect_as_router: bool = False,
        name: str = "lwd-router-channel",
    ) -> None:
        """Args:
            bind_endpoint: bind=True 为 bind 侧端点(``tcp://*:port``);
                bind=False 且 DEALER 形态为 connect 端点(构造即连);
                bind=False 且 ROUTER 形态(connect_as_router=True)传
                None,由调用方随后 ``connect()`` 逐对端连接。
            decoder: 载荷解码器(通用通信类不感知消息类型,由上层注入)。
            identity: 本进程线上身份。ROUTER-ROUTER(云侧复用)必填
                ``edge{e}``/``cloud{c}``;DEALER 形态按原 pp 类语义可传
                任意 identity(如 ``str(instance_id)``)或 None。
            expected_instances: 期望对端数(仅日志用)。
            connect_as_router: connect 侧用 ROUTER 而非 DEALER(云侧
                复用:边侧连多台云、需 identity 寻址)。
        """
        self._name = name
        self._decoder = decoder
        self._running = True
        self._started = False
        self._seq = 0
        # socket 类型:bind 侧恒 ROUTER;connect 侧 DEALER(原类语义)
        # 或 ROUTER(connect_as_router,云侧复用)
        self._is_router = bind or connect_as_router
        self._identity = identity.encode() if identity is not None else None
        parsed = _parse_identity(self._identity) if self._identity else None
        if self._identity is not None and parsed is None:
            raise ValueError(
                f"[lwd] router channel cannot parse identity {identity!r}; "
                f"expected 'edge{{id}}'/'cloud{{id}}' or a decimal instance id"
            )
        self._self_prefix = parsed[0] if parsed is not None else ""
        self._peer_prefix = _peer_prefix_of(self._self_prefix)
        self._expected_instances = max(1, int(expected_instances))

        # 调用线程 -> IO 线程的发送桥(有界,publish 满即 False)
        self._send_queue: queue.Queue[tuple[int, int, Any]] = queue.Queue(
            maxsize=self.SEND_HWM
        )
        # ROUTER 形态:per-identity 待发 FIFO(IO 线程独占写,
        # publish 只读 len 做上限判定)
        self._identity_pending: dict[bytes, deque[tuple[bytes, bytes]]] = {}
        # 就绪表:已完成 HELLO/WELCOME 握手的对端 identity(仅 ROUTER 形态)
        self._known_identities: set[bytes] = set()
        # IO 线程 -> 消费方的接收列表 + 条件变量(可阻塞消费)
        self._received: list[tuple[int, Any]] = []
        self._recv_cond = threading.Condition()

        # DEALER 形态握手/发送状态(原 pp 类命名:单对端标量语义)
        self._handshake_pending = False
        self._welcome_received = False
        self._welcome_warned = False
        self._welcome_deadline: float | None = None
        self._pending_send: tuple[bytes, bytes] | None = None
        # ROUTER-connect 形态握手状态:待发 HELLO 的对端 / WELCOME 看门狗
        self._hello_pending: set[bytes] = set()
        self._welcomed_ids: set[bytes] = set()
        self._welcome_warned_ids: set[bytes] = set()
        self._welcome_deadlines: dict[bytes, float] = {}

        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(
            zmq.ROUTER if self._is_router else zmq.DEALER
        )
        if self._identity is not None:
            self._socket.set(zmq.IDENTITY, self._identity)
        if self._is_router:
            # Directed sends to an identity with no live connection must
            # fail loudly (EHOSTUNREACH) instead of being silently dropped
            # by ZMQ; the IO thread catches it and keeps the message queued
            # for retry.
            self._socket.set(zmq.ROUTER_MANDATORY, 1)
        self._socket.set_hwm(self.SEND_HWM)
        if bind:
            self._socket.bind(bind_endpoint)
        elif not self._is_router:
            # DEALER 形态:构造即连单端点(原 pp 类语义)
            self._socket.connect(bind_endpoint)
            self._handshake_pending = True
        self._socket.set(zmq.LINGER, 0)
        logger.info(
            "[Lwd][router-channel] %s identity=%s %s %s expects %d instances",
            name,
            identity,
            "ROUTER" if self._is_router else "DEALER",
            (f"bound {bind_endpoint}" if bind else
             f"connecting to {bind_endpoint}" if bind_endpoint else ""),
            self._expected_instances,
        )
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    # 对外接口(调用线程;connect/announce 须在 start() 之前)             #
    # ------------------------------------------------------------------ #
    def connect(self, endpoint: str) -> None:
        """connect 侧(ROUTER 形态)追加对端端点:一个 socket 连 N 台云。

        仅允许 start() 之前调用(ZMQ socket 非线程安全);DEALER 形态为
        单端点构造即连,无此接口。"""
        if self._started or not self._is_router:
            raise RuntimeError(
                "[lwd] router channel connect() is only allowed before "
                "start() on the ROUTER connect form (ZMQ sockets are not "
                "thread-safe; the DEALER form connects at construction)"
            )
        self._socket.connect(endpoint)
        logger.info(
            "[Lwd][router-channel] %s connected -> %s", self._name, endpoint
        )

    def announce(self, instance_id: int, engine: int = 0) -> None:
        """connect 侧(ROUTER 形态)向对端发起 HELLO(定向首帧,幂等;
        start 前调用)。

        TCP 未建连时 HELLO 发送失败(EHOSTUNREACH),由 IO 线程每 tick
        重试到成功;成功后挂 WELCOME 看门狗。engine: 云内 DP 的引擎
        下标(仅云对端有意义,缺省 0 与单引擎逐值一致)。"""
        if self._started or not self._is_router:
            raise RuntimeError(
                "[lwd] router channel announce() is only allowed before "
                "start() on the ROUTER connect form"
            )
        identity = _instance_identity(self._peer_prefix, instance_id, engine)
        self._hello_pending.add(identity)
        self._welcome_deadlines.pop(identity, None)

    def start(self) -> None:
        """拉起 IO 线程(幂等);bind 侧与 DEALER 形态构造后直接 start。"""
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._io_thread,
            daemon=True,
            name=f"lwd-router-{self._name}",
        )
        self._thread.start()

    def publish(self, notify: Any, instance_id: int = 0, engine: int = 0) -> bool:
        """把消息放入待发队列(DEALER 形态单对端,忽略 instance_id)。

        ROUTER 形态按 (instance_id, engine) 寻址 per-identity FIFO;FIFO
        未满返回 True,达 ``PER_INSTANCE_PENDING_LIMIT``(或发送桥满)
        返回 False,由调用方退避重试——控制面消息不可丢,通道内不丢弃。
        """
        if not self._running:
            return False
        if self._is_router:
            identity = _instance_identity(self._peer_prefix, instance_id, engine)
            pending = self._identity_pending.get(identity)
            if (
                pending is not None
                and len(pending) >= self.PER_INSTANCE_PENDING_LIMIT
            ):
                logger.warning(
                    "[Lwd][router-channel] %s pending FIFO for %s at limit "
                    "(%d); publish refused (caller should retry)",
                    self._name,
                    identity.decode(),
                    self.PER_INSTANCE_PENDING_LIMIT,
                )
                return False
        try:
            seq = self._seq
            self._seq += 1
            self._send_queue.put_nowait(((instance_id, engine), seq, notify))
        except queue.Full:
            logger.warning(
                "[Lwd][router-channel] %s send bridge queue full; publish "
                "refused (caller should retry)",
                self._name,
            )
            return False
        return True

    def consume_new_outputs(
        self, timeout_s: float | None = None
    ) -> list[tuple[int, Any]]:
        """返回并清空自上次调用以来收到的 ``[(对端id, notify)]``。

        ROUTER 形态对端 id 从信封 identity 解出(边侧为云 id、云侧为边
        id),不解析载荷;DEALER 形态恒单对端,首元素恒为 0。
        ``timeout_s`` 非 None 时列表为空则阻塞等待(或至超时),供接收
        循环做阻塞消费;``closed`` 后返回空列表。
        """
        with self._recv_cond:
            if not self._received and timeout_s is not None:
                self._recv_cond.wait_for(
                    lambda: self._received or not self._running, timeout_s
                )
            outputs = self._received
            self._received = []
            return outputs

    def ready_instance_ids(self) -> set[int]:
        """就绪表:已完成 HELLO/WELCOME 握手的对端实例 id 集合(仅 ROUTER
        形态;DEALER 形态恒为空,与原 pp 类一致——DEALER 建连由 libzmq
        缓冲保证,就绪性经 WELCOME 看门狗告警暴露)。

        边侧为云 id(``ready_cloud_ids`` 语义)、云侧为边 id
        (``ready_edge_ids`` 语义),供准入/调度跳过未连上的对端。
        迭代做快照(IO 线程可能并发新增 identity)。"""
        return {
            instance_id
            for identity in tuple(self._known_identities)
            if (parsed := _parse_identity(identity)) is not None
            and parsed[0] == self._peer_prefix
            and (instance_id := parsed[1]) is not None
        }

    @property
    def closed(self) -> bool:
        return not self._running

    def shutdown(self) -> None:
        """优雅关停(幂等):停 IO 线程、linger=0 关 socket。"""
        self._running = False
        with self._recv_cond:
            self._recv_cond.notify_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)
        with contextlib.suppress(Exception):
            self._socket.close(linger=0)

    # ------------------------------------------------------------------ #
    # IO 线程(zmq 单线程亲和:收发都在本线程)                             #
    # ------------------------------------------------------------------ #
    def _io_thread(self) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while self._running:
            self._flush_hello()
            self._check_welcome_watchdog()
            flushed = self._flush_send_queue()
            # 有发送积压且可发时零超时轮询(尽快回去排水);全部滞留
            # (对端未就绪/断连)则短暂阻塞,避免忙转。DEALER 消息由
            # libzmq 缓冲,恒视为可发(原 pp 类语义)。
            backlog = (
                not self._send_queue.empty()
                or self._pending_send is not None
                or any(self._identity_pending.values())
            )
            can_send = (
                (not self._is_router)
                or bool(self._known_identities)
                or bool(self._hello_pending)
            )
            timeout = 0 if (backlog and can_send and flushed) else self.POLL_TIMEOUT_MS
            try:
                events = dict(poller.poll(timeout=timeout))
            except Exception:
                if not self._running:
                    break
                logger.exception(
                    "[Lwd][router-channel] %s IO thread poll error", self._name
                )
                time.sleep(0.01)
                continue
            if events.get(self._socket, 0) & zmq.POLLIN:
                self._recv_available()

    def _flush_hello(self) -> None:
        """connect 侧发 HELLO:DEALER 单对端一次性(原 pp 类语义);
        ROUTER 形态逐对端重试到送达。"""
        if not self._is_router:
            if not self._handshake_pending:
                return
            try:
                self._socket.send_multipart(
                    [_EMPTY_FRAME, _HELLO_FRAME], flags=zmq.NOBLOCK
                )
            except zmq.Again:
                # Connection not writable yet: retry on the next tick.
                return
            except Exception:
                if self._running:
                    logger.exception(
                        "[Lwd][router-channel] %s handshake error", self._name
                    )
                return
            self._handshake_pending = False
            if self._welcome_deadline is None:
                # HELLO is on the wire; the WELCOME ack should follow within
                # one timeout.  (Set here rather than at connect time so slow
                # TCP connect establishment does not eat into the budget.)
                self._welcome_deadline = (
                    time.monotonic() + self.WELCOME_ACK_TIMEOUT
                )
            logger.info(
                "[Lwd][router-channel] %s HELLO handshake sent", self._name
            )
            return
        for identity in list(self._hello_pending):
            try:
                self._socket.send_multipart(
                    [identity, _EMPTY_FRAME, _HELLO_FRAME], flags=zmq.NOBLOCK
                )
            except (zmq.Again, zmq.ZMQError):
                # TCP 未建连/对端 HWM 满:下一 tick 重试
                continue
            except Exception:
                if self._running:
                    logger.exception(
                        "[Lwd][router-channel] %s HELLO send error",
                        self._name,
                    )
                continue
            self._hello_pending.discard(identity)
            if identity not in self._welcomed_ids:
                self._welcome_deadlines[identity] = (
                    time.monotonic() + self.WELCOME_ACK_TIMEOUT
                )
            logger.info(
                "[Lwd][router-channel] %s HELLO sent -> %s",
                self._name,
                identity.decode(),
            )

    def _check_welcome_watchdog(self) -> None:
        """connect 侧重复 identity 检测:HELLO 已发但 WELCOME 未回。

        libzmq 对重复 identity 静默饿死后连者:其 HELLO 到不了对端,
        WELCOME 也永远不回——此处是唯一可见痕迹(两个实例误配同一 id),
        与 ``validate_self`` 构成配置错误双保险。"""
        now = time.monotonic()
        if not self._is_router:
            if (
                self._welcome_received
                or self._welcome_warned
                or self._welcome_deadline is None
                or now < self._welcome_deadline
            ):
                return
            self._welcome_warned = True
            logger.error(
                "[Lwd][router-channel] %s HELLO was sent but the peer never "
                "acked it (no WELCOME in %ds). Another process almost "
                "certainly connected FIRST with the same identity=%s: libzmq "
                "silently starves later duplicate-identity connections.",
                self._name,
                int(self.WELCOME_ACK_TIMEOUT),
                self._identity.decode() if self._identity else "<unset>",
            )
            return
        for identity, deadline in list(self._welcome_deadlines.items()):
            if identity in self._welcomed_ids or now < deadline:
                continue
            self._welcome_deadlines.pop(identity, None)
            if identity in self._welcome_warned_ids:
                continue
            self._welcome_warned_ids.add(identity)
            logger.error(
                "[Lwd][router-channel] %s HELLO was sent but %s never acked "
                "(no WELCOME in %ds). Another process almost certainly owns "
                "the same identity, or the peer is not up; THIS side is "
                "receiving nothing from %s.",
                self._name,
                identity.decode(),
                int(self.WELCOME_ACK_TIMEOUT),
                identity.decode(),
            )

    def _flush_send_queue(self) -> bool:
        """排水发送桥 -> per-identity FIFO(ROUTER)/ 单飞行槽(DEALER);
        返回本 tick 是否有消息入队。"""
        staged = False
        while self._running:
            if not self._is_router and self._pending_send is not None:
                # DEALER has a single in-flight slot; the retry left in it
                # must drain before the next message is serialized.
                break
            try:
                (instance_id, engine), seq, notify = self._send_queue.get_nowait()
            except queue.Empty:
                break
            try:
                data = msgspec.msgpack.encode(notify)
            except Exception:
                logger.exception(
                    "[Lwd][router-channel] %s failed to serialize notify",
                    self._name,
                )
                continue
            if self._is_router:
                identity = _instance_identity(
                    self._peer_prefix, instance_id, engine
                )
                pending = self._identity_pending.setdefault(identity, deque())
                pending.append((seq.to_bytes(_SEQ_BYTES, "big"), data))
            else:
                self._pending_send = (seq.to_bytes(_SEQ_BYTES, "big"), data)
            staged = True
        # Stage 2: drain per-identity FIFOs (ROUTER) / the single in-flight
        # slot (DEALER).
        if self._is_router:
            for identity in list(self._identity_pending.keys()):
                self._flush_identity(identity)
        elif self._pending_send is not None:
            self._flush_dealer()
        return staged

    def _flush_identity(self, identity: bytes) -> None:
        """ROUTER:发单对端滞留消息,直至 EAGAIN(背压)/EHOSTUNREACH。"""
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
                # 对端 HWM 满:只背压该 identity
                return
            except zmq.ZMQError as e:
                if e.errno in _UNROUTABLE_ERRNOS:
                    # 未连/未就绪:滞留等 HELLO 或重连
                    return
                if self._running:
                    logger.exception(
                        "[Lwd][router-channel] %s send to %s error",
                        self._name,
                        identity.decode(),
                    )
                pending.popleft()
                continue
            except Exception:
                if self._running:
                    logger.exception(
                        "[Lwd][router-channel] %s send error", self._name
                    )
                pending.popleft()
                continue
            pending.popleft()
        if not pending:
            self._identity_pending.pop(identity, None)

    def _flush_dealer(self) -> None:
        """DEALER:发单飞行槽消息(无信封,原 pp 类语义),背压下 tick 重试。"""
        assert self._pending_send is not None
        frames = [_EMPTY_FRAME, self._pending_send[0], self._pending_send[1]]
        try:
            self._socket.send_multipart(frames, flags=zmq.NOBLOCK)
        except zmq.Again:
            # Peer HWM full (or not yet connected): retry on the next tick.
            return
        except Exception:
            if self._running:
                logger.exception(
                    "[Lwd][router-channel] %s send error", self._name
                )
        self._pending_send = None

    def _send_welcome(self, identity: bytes) -> None:
        """bind 侧对 HELLO 的应答:被饿死的重复 identity 收不到它,
        对端看门狗即报错——重复 identity 唯一的应用层可见痕迹。"""
        try:
            self._socket.send_multipart(
                [identity, _EMPTY_FRAME, _WELCOME_FRAME], flags=zmq.NOBLOCK
            )
        except Exception:
            if self._running:
                logger.exception(
                    "[Lwd][router-channel] %s failed to send WELCOME to %s",
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
                    "[Lwd][router-channel] %s recv error", self._name
                )
            return
        if not frames:
            return
        if not self._is_router:
            self._recv_dealer(frames)
            return
        # ROUTER 双向对称:首帧恒为发送方 identity
        identity = frames[0]
        # 握手帧 [identity, b"", HELLO/WELCOME]
        if len(frames) == 3 and frames[2] in (_HELLO_FRAME, _WELCOME_FRAME):
            self._handle_handshake(identity, frames[2])
            return
        # 数据帧 [identity, b"", seq8, payload]
        if len(frames) != 4:
            logger.warning(
                "[Lwd][router-channel] %s dropping malformed message "
                "(frames=%d)",
                self._name,
                len(frames),
            )
            return
        seq_bytes, data = frames[2], frames[3]
        if len(seq_bytes) != _SEQ_BYTES:
            logger.warning(
                "[Lwd][router-channel] %s dropping malformed message "
                "(seq len=%d)",
                self._name,
                len(seq_bytes),
            )
            return
        parsed = _parse_identity(identity)
        if parsed is None or parsed[0] != self._peer_prefix:
            logger.warning(
                "[Lwd][router-channel] %s dropping message from unknown "
                "identity %r",
                self._name,
                identity,
            )
            return
        self._deliver(parsed[1], data)

    def _recv_dealer(self, frames: list[bytes]) -> None:
        """DEALER 收帧(原 pp 类语义):[b"", WELCOME] 握手应答 /
        [b"", seq8, payload] 数据(ROUTER 的信封被 DEALER 剥掉)。"""
        # WELCOME ack: [empty, WELCOME]
        if len(frames) == 2 and frames[1] == _WELCOME_FRAME:
            self._welcome_received = True
            logger.info(
                "[Lwd][router-channel] %s peer confirmed identity=%s "
                "(WELCOME received)",
                self._name,
                self._identity.decode() if self._identity else "<unset>",
            )
            return
        seq_bytes, data = frames[-2], frames[-1]
        if len(seq_bytes) != _SEQ_BYTES:
            logger.warning(
                "[Lwd][router-channel] %s dropping malformed message "
                "(frames=%d)",
                self._name,
                len(frames),
            )
            return
        # DEALER 恒单对端:首元素恒为 0
        self._deliver(0, data)

    def _deliver(self, instance_id: int, data: bytes) -> None:
        """解码载荷并投递消费列表;坏包告警丢弃,不中断 IO 线程。"""
        try:
            notify = self._decoder(data)
        except (msgspec.DecodeError, msgspec.ValidationError):
            logger.warning(
                "[Lwd][router-channel] %s drop malformed notify frame",
                self._name,
            )
            return
        with self._recv_cond:
            self._received.append((instance_id, notify))
            self._recv_cond.notify_all()

    def _handle_handshake(self, identity: bytes, frame: bytes) -> None:
        parsed = _parse_identity(identity)
        if parsed is None or parsed[0] != self._peer_prefix:
            logger.warning(
                "[Lwd][router-channel] %s handshake from unknown identity %r",
                self._name,
                identity,
            )
            return
        if frame == _HELLO_FRAME:
            is_new = identity not in self._known_identities
            self._known_identities.add(identity)
            if is_new:
                logger.info(
                    "[Lwd][router-channel] %s peer connected (identity=%s, "
                    "%d/%d ready)",
                    self._name,
                    identity.decode(),
                    len(self._known_identities),
                    self._expected_instances,
                )
            else:
                # 同 identity 重新 HELLO:对端重启/断网重连,滞留消息按序补投
                logger.warning(
                    "[Lwd][router-channel] %s identity=%s HELLO'd again: peer "
                    "reconnected (restart or network drop); held messages "
                    "resume delivery",
                    self._name,
                    identity.decode(),
                )
            self._send_welcome(identity)
            return
        # WELCOME:connect 侧(边)就绪标记 + 看门狗解除
        if identity not in self._welcomed_ids:
            self._welcomed_ids.add(identity)
            self._welcome_deadlines.pop(identity, None)
            self._known_identities.add(identity)
            logger.info(
                "[Lwd][router-channel] %s peer confirmed identity=%s "
                "(WELCOME received, %d/%d ready)",
                self._name,
                identity.decode(),
                len(self._known_identities),
                self._expected_instances,
            )
