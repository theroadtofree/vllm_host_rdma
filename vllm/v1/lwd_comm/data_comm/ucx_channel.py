# SPDX-License-Identifier: Apache-2.0
"""通用 host-host UCX 传输库(UHC1 协议族)——用法见 ucx_channel_demo.py。

定位:带 tag 的字节管道。创建通道(任意条数,不绑定 up/down)、
send(host 地址进)、recv(独立 bytes 或调用方缓冲出);RDMA 注册
(libucp rcache 隐式)、IO 线程、断链自动重连、对端重启检测全部
内部化。设备拷贝(D2H/H2D)与 seqno/aux 编排属于调用方——本文件
零 torch import,可在 NPU/GPU/CPU 任意边侧组合下复用。

与 legacy ucx_wire.py(LWD 数据面,已删除)的关系(方案 A:只增不改):
    * 复用 ucx_ucp.py 绑定、sock.py TCP 工具、ucx_pool.py 块池与
    _UcxLink 链路对象(经 _ChanLink 子类扩展 masked tag 预挂;现已
    迁至 ucx_link.py,legacy wire 主体已删除);
    * 线上为 UHC1 新协议族:无 LWD2 帧头——长度由 UCP completion 自
    描述、tag 由 API 参数直传,软件 CRC 取消(RoCE FCS/ICRC + RC
    重传已端到端兜底,原 CRC 的动机是诊断自家暂存拷贝,本库发送侧
    零暂存)。灰度单位 = 通道整对切换(两端同为 UHC1)。

接口契约(硬约束):
  * send(tag, addr, size):调用返回即库可读 addr 上的字节(外部
    保证 D2H 已完成);op 完成(OK)前调用方不得复写 addr(rndv
    下库持有该缓冲直至对端落盘),完成后缓冲自由复用;
  * recv(tag):交付独立 bytes,调用方持引用即可发起 H2D(GC 保活
    到拷贝完成);同一 tag 允许多次预挂(队列式,每次到达消费一条,
    MPI 式流式接收);早到帧(先到后挂)按 tag FIFO 暂存(同 tag
    连发不丢帧),总条数上界 early_max;
  * recv_into(tag, addr, cap):完成即数据落在 addr;完成前 addr
    归库(收侧仍经库内块中转,一次 host 拷贝入 addr;线上级零拷贝
    的精确 tag 预挂为后续扩展);
  * 断链:在途 op 全部以 ERROR 终结(投递状态未知,不重放);
    connect 侧自动重连(可关),serve 侧循环 accept 换链;对端重启
    (session 指纹变化)经 on_peer_restart 回调通知,tag 基线复位
    是调用方的事;
  * tag 取值 [0, 2^61):高 3 位保留给 put 模式门铃/ACK 内部标记。

put 大帧路径(可选;cfg.put_thresh>0 且两端同配才生效):发送方
ucp_put_nbx 单边直写对端槽位(tag % n_slots),完成后发 16B 门铃
(tag,length),收方消费后回 ACK 门控槽位复用;小消息仍走 eager
tag 内联。启用后接收分池 masked 匹配(数据/门铃/ACK 各自预挂,
互不误配)。

部署前置(与 legacy wire 相同):两端 LD_LIBRARY_PATH 指向
UCX 1.22;TLS 按网卡固化(hns: rc_verbs,tcp,self / mlx5:
rc_mlx5 或 rc_verbs)。
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import queue
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from vllm.logger import init_logger
from vllm.v1.lwd_comm.data_comm import sock as _sock
from vllm.v1.lwd_comm.data_comm.ucx_pool import Block as _Block
from vllm.v1.lwd_comm.data_comm.ucx_pool import BlockPool
from vllm.v1.lwd_comm.data_comm.ucx_ucp import (
    _RecvCb,
    _SendCb,
    _UCS_OK,
    _ptr_is_err,
    _ptr_status,
    _req_param,
    _ucp,
)
from vllm.v1.lwd_comm.data_comm.ucx_link import (
    _UcxLink,
    _hs_pack_caps,
    _hs_unpack_caps,
)

logger = init_logger(__name__)


class HostDoneEvent:
    """发送 future 的完成事件(host 侧,device 无关):query() 兼容
    LwdCommFuture 的 done()/wait()/_finalize 轮询路径;wait/wait_event
    为 no-op(发送链帧序由 IO 线程串行保证)。从 legacy ucx_wire.py
    (已删除)原样迁入。"""

    # Marker consulted by LwdTransferFuture.wait_on_current_stream:
    # a host event has no device ordering semantics, so ordering a
    # stream after it must be skipped.
    is_host_event = True

    def __init__(self) -> None:
        self._flag = threading.Event()

    def set(self) -> None:
        self._flag.set()

    def query(self) -> bool:
        return self._flag.is_set()

    def wait_event(self, _event_or_stream: Any) -> None:  # no-op
        pass

    def wait(self, _stream: Any = None) -> None:  # no-op
        """torch Stream.wait_event 的反向调用路径(host 事件无 device
        排序语义)。"""


# ucp_request_cancel:close 路径撤回未完成的预挂接收(按需补挂绑定,
# 不动共享的 ucx_ucp.py)
if _ucp is not None:
    _ucp.ucp_request_cancel.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p]
    _ucp.ucp_request_cancel.restype = ctypes.c_void_p


# --------------------------------------------------------------------- #
# 配置(全部构造期参数;默认值偏轻量,重负载按需调大)                     #
# --------------------------------------------------------------------- #
@dataclass
class UcxCfg:
    """一条通道的传输配置(两端 block_mb 必须一致:大端发送会截断小端)。

    block_mb:   单块容量:收向=接收块(对端单条消息上界);pool_staging
                通道=D2H 暂存块(本端单条发送上界)
    n_blocks:   接收块数(早到帧暂存条数默认值;put 槽位数)
    put_thresh: >0 启用 RMA put 大帧路径(需两端同配;且 unset
                UCX_RNDV_THRESH,否则小内联帧被 UCX 转 rndv)
    hs_timeout_s: TCP 握手超时(建连/accept 两侧有界)
    token:      通道标识(握手互验,防误连到别人的端口;默认 0 不校验
                语义,仅要求两端同值)
    early_max:  早到帧暂存上界(条数;0 = n_blocks)
    traffic_class: IB Traffic Class / RoCEv2 DSCP(None=UCX 默认 auto:
                RoCEv2 取端口全局 TC 否则 106);两端同值
    sl:         IB Service Level / RoCEv2 以太网优先级 PCP(None=默认 0);
                交换机 PFC/ETS 按 DSCP+PCP 联合分流,通常与
                traffic_class 配套配置;两端同值
    """

    block_mb: int = 64
    n_blocks: int = 4
    put_thresh: int = 0
    hs_timeout_s: float = 5.0
    token: int = 0
    early_max: int = 0   # 早到暂存总条数上界(0 = n_blocks)
    traffic_class: int | None = None
    sl: int | None = None
    pool_staging: bool = False  # True:本通道池块仅作发送 D2H 暂存
                                # (acquire_staging/release_staging 周转),
                                # 不挂 wildcard——单向"只发"通道用


# tag 位域:低 61 位归应用;高 2 位为 put 模式控制消息标记
_TAG_ACK = 1 << 61          # 收方 → 发方:槽位已消费(门控复用)
_TAG_DOORBELL = 1 << 62     # 发方 → 收方:数据已落你的槽位,速来消费
_TAG_USER_MAX = (1 << 61) - 1
_CTL_BITS = _TAG_ACK | _TAG_DOORBELL

# 控制消息载荷:门铃 (tag, length);ACK (tag)
_DB = struct.Struct("!QQ")
_AKP = struct.Struct("!Q")

# 握手(UHC1):TCP 交换 worker 地址 + put 槽位能力 + 容量/会话/标识
_HS_MAGIC = b"UHC1\0\0\0\0"
_HS_FMT = "!8sIQQQIII"  # magic, addr_len, max_msg, session, token, n_slots, caps_len, put_thresh
_HS_HEADER = struct.Struct(_HS_FMT)
# index in _hs_parse_header()'s return tuple, NOT the raw struct field index
# (raw token is field 4; parsed tuple is alen,max,session,token,... => 3)
_HS_TOKEN_INDEX = 3
_HS_SESSION = int.from_bytes(os.urandom(8), "little")

_UCX_ENV_LOGGED = False
_DEFAULT_HS_TIMEOUT_S = 5.0


def _log_ucx_env_once() -> None:
    global _UCX_ENV_LOGGED
    if _UCX_ENV_LOGGED:
        return
    _UCX_ENV_LOGGED = True
    tls = os.getenv("UCX_TLS", "")
    if not tls:
        logger.warning(
            "[ucx-chan] UCX_TLS 未设置,使用 UCX 默认选择。推荐显式固化:"
            "hns 侧 rc_verbs,tcp,self;mlx5 侧 rc_mlx5 或 rc_verbs")
    else:
        logger.warning("[ucx-chan] env: UCX_TLS=%r UCX_NET_DEVICES=%r",
                       tls, os.getenv("UCX_NET_DEVICES", "") or "<默认>")


# --------------------------------------------------------------------- #
# 握手(TCP 交换 worker 地址 + 能力;session/token/max_msg 随行)         #
# --------------------------------------------------------------------- #
def _hs_send(fd: int, addr: bytes, max_msg: int, token: int,
             n_slots: int, caps: bytes, put_thresh: int) -> None:
    _sock.c_send_bytes(
        fd,
        struct.pack(_HS_FMT, _HS_MAGIC, len(addr), max_msg, _HS_SESSION,
                    token, n_slots, len(caps), put_thresh)
        + addr + caps)


def _hs_recv_exact(fd: int, n: int) -> bytes:
    buf = ctypes.create_string_buffer(n)
    got = 0
    base = ctypes.addressof(buf)
    while got < n:
        m = _sock.c_recv_ptr(fd, base + got, n - got)
        if m == 0:
            raise ConnectionError("peer closed during ucx-channel handshake")
        got += m
    return buf.raw


def _hs_parse_header(
    hdr: bytes,
    local_max: int | None = None,
    expect_token: int | None = None,
) -> tuple[int, int, int, int, int, int, int]:
    """校验 UHC1 握手头;返回解析后的定长字段。

    ``local_max`` / ``expect_token`` 为 None 时跳过对应校验——listener
    只读头拿 token 做路由,完整的 max/token 校验仍由目标 channel 在
    读 body 前完成。
    """
    magic, alen, max_msg, session, token, n_slots, caps_len, put_thresh = (
        _HS_HEADER.unpack(hdr)
    )
    if magic != _HS_MAGIC:
        raise ConnectionError(
            f"bad handshake magic {magic!r}: 对端不是 UHC1(ucx_channel)"
            f"——旧 LWD wire 与本库不可互通,两端须同库")
    if alen <= 0 or alen > (1 << 20):
        raise ConnectionError(f"bad handshake address length {alen}")
    if caps_len > (1 << 24) or n_slots > 64:
        raise ConnectionError(f"bad handshake caps {caps_len}/{n_slots}")
    if expect_token is not None and token != expect_token:
        raise ConnectionError(
            f"handshake token mismatch {token:#x} != {expect_token:#x}:"
            f"对端不是本通道(检查两端 token/端口配置)")
    if local_max is not None and max_msg > local_max:
        # 对端单条消息上界超过本端接收块容量:超长消息将截断——启动即报
        raise ConnectionError(
            f"peer max_msg {max_msg} > local {local_max}: 两端 block_mb "
            f"须一致(对齐 cfg.block_mb)")
    return (
        alen, max_msg, session, token, n_slots, caps_len, put_thresh,
    )


def _hs_recv(fd: int, local_max: int,
             expect_token: int) -> tuple[bytes, int, int, int, bytes, int]:
    """收对端握手;返回 (addr, session, n_slots, put_thresh, caps, max_msg)。"""
    hdr = _hs_recv_exact(fd, _HS_HEADER.size)
    alen, max_msg, session, _token, n_slots, caps_len, put_thresh = (
        _hs_parse_header(hdr, local_max, expect_token)
    )
    body = _hs_recv_exact(fd, alen + caps_len)
    return (body[:alen], session, n_slots, put_thresh,
            body[alen:] if caps_len else b"", max_msg)


# --------------------------------------------------------------------- #
# 完成句柄                                                              #
# --------------------------------------------------------------------- #
class UcxOp:
    """一次 send/recv 的完成句柄。

    完成在 IO 线程置位;done/wait 可在任意线程调用。成功:
    send → size 生效;recv → data(bytes)生效;recv_into → size 生效
    且数据已写至调用方缓冲。失败:wait 抛 RuntimeError,error 属性非空。
    用户回调(cb)在 IO 线程执行——只许提交异步操作,禁止阻塞。
    """

    __slots__ = ("tag", "size", "data", "error", "_ev", "_cb", "_ref",
                 "_into_addr", "_into_cap")

    def __init__(self, tag: int, cb: Callable[["UcxOp"], None] | None = None,
                 ref: Any = None, into_addr: int = 0, into_cap: int = 0):
        self.tag = tag
        self.size = 0
        self.data: bytes | None = None
        self.error: BaseException | None = None
        self._ev = threading.Event()
        self._cb = cb
        self._ref = ref          # 发送源对象保活(如 bytes/bytearray)
        self._into_addr = into_addr
        self._into_cap = into_cap

    def _finish(self, *, data: bytes | None = None, size: int | None = None,
                error: BaseException | None = None) -> None:
        if self._ev.is_set():
            return
        self.data = data
        self.size = size if size is not None else (
            len(data) if data is not None else 0)
        self.error = error
        self._ref = None         # 释放发送源保活
        self._ev.set()
        cb = self._cb
        if cb is not None:
            try:
                cb(self)
            except Exception:  # noqa: BLE001
                logger.error("[ucx-chan] op cb error tag=%s", self.tag,
                             exc_info=True)

    def done(self) -> bool:
        """非阻塞查询(纯 host,不碰设备)。"""
        return self._ev.is_set()

    def wait(self, timeout: float | None = None) -> "UcxOp":
        """阻塞至完成;失败抛 RuntimeError,超时抛 TimeoutError。"""
        if not self._ev.wait(timeout):
            raise TimeoutError(
                f"ucx channel op tag={self.tag} wait timed out "
                f"({timeout}s)")
        if self.error is not None:
            raise RuntimeError(
                f"ucx channel op tag={self.tag} failed: {self.error}")
        return self

    def __repr__(self) -> str:  # pragma: no cover - 诊断辅助
        state = "ERR" if self.error else ("OK" if self._ev.is_set() else "PEND")
        return f"<UcxOp tag={self.tag} {state} size={self.size}>"


# --------------------------------------------------------------------- #
# 链路:复用 _UcxLink,新增 masked tag 预挂与 err 路由                    #
# --------------------------------------------------------------------- #
class _ChanLink(_UcxLink):
    """UHC1 链路:生命周期/握手地址/put 能力全复用 _UcxLink(owner 需
    暴露 _pool/_block_bytes/_put_thresh 属性,由 UcxChannel 提供)。"""

    def __init__(self, owner: "UcxChannel") -> None:
        super().__init__(owner)
        self.peer_max_msg = 0    # 握手填充(对端单条消息上界)

    def recv_masked_nb(self, addr: int, length: int, tag: int, mask: int,
                       cb: Any) -> int:
        """masked tag 预挂((msg_tag & mask) == tag 即匹配):数据/门铃/
        ACK 分池路由的基础。返回 request 指针。"""
        return _ucp.ucp_tag_recv_nbx(
            ctypes.c_void_p(self._worker), ctypes.c_void_p(addr), length,
            tag, mask, ctypes.byref(_req_param(cb)))

    def _on_ep_err(self, arg, ep, status) -> None:  # noqa: ANN001
        # 覆盖基类路由(owner 是 UcxChannel,不走旧 _WIRES 注册表)
        self._owner._on_ep_error(int(status))


# --------------------------------------------------------------------- #
# 通道                                                                  #
# --------------------------------------------------------------------- #
class UcxChannel:
    """一条双向 host-host 通道:一个 ep + 一个 IO 线程 + 一组接收块。

    线程模型:send/recv 提交任意线程、永不阻塞(内部任务队列);完成
    回调与用户 cb 全部在 IO 线程触发(与任务分发同线程,免锁)。建连
    由监督线程(connect 侧:首连+断链重建)或 accept 线程(serve 侧:
    循环 accept)产出新链路,经 _linkq 移交 IO 线程 adopt;此后
    worker/ep 仅由 IO 线程触碰。

    首连窗口语义:链路未就绪时提交的任务在队列原地保序等待(不报
    错);就绪后断链,在途与后续任务立即以 ERROR 终结(fail-fast),
    重建 adopt 后恢复——与 legacy wire 真机口径一致。
    """

    def __init__(self, cfg: UcxCfg | None = None, *, name: str = "") -> None:
        c = cfg or UcxCfg()
        self._block_bytes = max(1, int(c.block_mb)) << 20
        self._n_blocks = max(2, int(c.n_blocks))
        self._put_thresh = max(0, int(c.put_thresh))
        # UCT 级配置注入(见 _ChanLink:read→modify→init):键为 UCX 配置
        # 名(不带 UCX_ 前缀),值由 ucp_config_modify 在 init 前写入。
        self._cfg_traffic_class = c.traffic_class
        self._uct_config: dict[str, str] = {}
        if c.traffic_class is not None:
            self._uct_config["IB_TRAFFIC_CLASS"] = str(int(c.traffic_class))
        if c.sl is not None:
            self._uct_config["IB_SL"] = str(int(c.sl))
        self._hs_timeout_s = max(1.0, float(c.hs_timeout_s))
        self._token = int(c.token)
        self._early_max = c.early_max or self._n_blocks
        self._pool_staging = bool(c.pool_staging)
        self._name = name or "ucx-chan"
        self._role = ""            # "connect" / "serve"(日志用)
        # _UcxLink 依赖的 owner 属性(_pool/_block_bytes/_put_thresh)
        self._pool = BlockPool(self._n_blocks, self._block_bytes)
        if self._pool_staging:
            # staging 通道:池块从构造起即可用(BlockPool 空闲队列
            # 初始为空,靠 adopt/reset 灌入——staging 通道跳过了
            # reset,首次 acquire_staging 将永远等不到块:真机实录
            # 边侧 sender 线程卡 acquire_staging、云侧 recv unbound)
            for blk in self._pool.blocks:
                self._pool.release(blk)
        # 接收配对:tag -> 等待队列(同 tag 多次预挂,到达逐条消费);
        # _expects_t0 = 队列从空转非空时刻(悬空检测用)
        self._expects: dict[int, deque[UcxOp]] = {}
        self._expects_t0 = 0.0
        # 早到帧(到达早于预挂):tag -> FIFO 字节队列(同 tag 连发
        # 不丢帧,逐条与预挂配对);总条数上界 _early_max
        self._early: dict[int, deque[bytes]] = {}
        self._early_count = 0
        # request 指针 -> 完成路由(send/put: (op, tag, size))
        self._send_reqs: dict[int, tuple[UcxOp, int, int]] = {}
        # request 指针 -> (块, 池类别)——重挂路由
        self._recv_reqs: dict[int, tuple[_Block, str]] = {}
        self._q: queue.Queue[Any | None] = queue.Queue()
        self._linkq: queue.Queue[_ChanLink] = queue.Queue()
        self._ready = threading.Event()
        self._broken_event = threading.Event()
        self._stopped = False
        self._broken = False
        self._ep_err: int | None = None
        self._link: _ChanLink | None = None
        self._gen = 0
        self._peer_session = 0
        self._peer_max_msg = 0
        # put 路径状态(IO 线程独占)
        self._put_armed = False       # 发送侧:可 put 直写对端槽位
        self._recv_ctl = False        # 接收侧:大帧走 put 槽位,内联走小池
        self._data_mask = 0           # 数据预挂掩码(put 启用时避开控制位)
        self._ctl_pool: list[_Block] = []
        self._db_pool: list[_Block] = []
        self._ack_pool: list[_Block] = []
        self._ack_bufs: list[bytearray] = []
        self._slot_last: dict[int, int] = {}
        self._slot_acked: dict[int, int] = {}
        self._db_pend: dict[int, bytearray] = {}
        # 用户回调
        self._broken_cb: Callable[[BaseException], None] | None = None
        self._restart_cb: Callable[[], None] | None = None
        # UCP 回调闭包长存(CFUNCTYPE 对象被 GC 后回调即悬空)
        self._send_cb = _SendCb(self._on_send_done)
        self._recv_cb = _RecvCb(self._on_recv_done)
        self._put_cb = _SendCb(self._on_put_done)
        self._db_cb = _SendCb(self._on_db_done)
        # 线程
        self._sup_thread: threading.Thread | None = None
        self._acc_thread: threading.Thread | None = None
        self._srv_fd: int | None = None
        self._listener: "UcxListener | None" = None
        self._thread = threading.Thread(
            target=self._run, name=f"ucx-chan-io-{self._name}", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ #
    # 通道创建                                                            #
    # ------------------------------------------------------------------ #
    @classmethod
    def connect(cls, host: str, port: int, *, cfg: UcxCfg | None = None,
                auto_reconnect: bool = True) -> "UcxChannel":
        """客户端侧:后台首连(退避重试),断链自动重建(可关)。

        返回即通道对象(不阻塞);wait_ready() 等首连就绪;就绪前的
        提交在队列保序等待。
        """
        ch = cls(cfg, name=f"cli:{host}:{port}")
        ch._role = "connect"
        logger.warning(
            "[ucx-chan] %s connect token=%#x hs=%ss",
            ch._name, ch._token, ch._hs_timeout_s,
        )
        ch._sup_thread = threading.Thread(
            target=ch._supervise, args=(host, port, auto_reconnect),
            name=f"ucx-chan-sup-{ch._name}", daemon=True)
        ch._sup_thread.start()
        return ch

    @classmethod
    def serve(cls, host: str, port: int, *,
              cfg: UcxCfg | None = None) -> "UcxChannel":
        """服务端侧:循环 accept——对端首连与其断链重连都从同端口进入,
        新链经 adopt 换链(通道对象/预挂状态跨重连保留)。

        一条通道一个端口;LWD 1e1c 的 UP/DOWN 共享端口走
        ``UcxListener`` + ``serve_via``（按握手 token 路由）。
        """
        ch = cls(cfg, name=f"srv:{port}")
        ch._role = "serve"
        ch._srv_fd = _sock.c_open()
        if ch._cfg_traffic_class is not None:
            # bind 前设置:accept 出的连接继承 listening socket 的 TOS
            _sock.c_set_tos(ch._srv_fd, ch._cfg_traffic_class << 2)
        _sock.c_setsockopt_int(
            ch._srv_fd, _sock._SOL_SOCKET, _sock._SO_REUSEADDR, 1,
            what="SO_REUSEADDR",
        )
        _sock.c_bind(ch._srv_fd, host, port)
        _sock.c_listen(ch._srv_fd, 1)
        ch._acc_thread = threading.Thread(
            target=ch._accept_loop, args=(port,),
            name=f"ucx-chan-acc-{port}", daemon=True)
        ch._acc_thread.start()
        logger.warning("[ucx-chan] %s handshake listening on %s:%d",
                       ch._name, host, port)
        return ch

    @classmethod
    def serve_via(cls, listener: "UcxListener", *,
                  cfg: UcxCfg | None = None) -> "UcxChannel":
        """共享 listener 形态:通道不再 bind,由 listener 按 token 投递。

        listener 必须先 register、后 start;channel 的 accept 线程由
        listener 的 accept 线程替代,IO/adopt 逻辑不变。
        """
        c = cfg or UcxCfg()
        ch = cls(c, name=f"srv:{listener.port}:{c.token:x}")
        ch._role = "serve"
        ch._listener = listener
        listener.register(ch._token, ch, name=ch._name)
        return ch

    # ------------------------------------------------------------------ #
    # 公共 API                                                            #
    # ------------------------------------------------------------------ #
    def send(self, tag: int, addr: int, size: int, *,
             ref: Any = None, cb: Callable[[UcxOp], None] | None = None
             ) -> UcxOp:
        """发送 addr 起始的 size 字节(tag 随消息直传对端)。

        契约:调用返回即字节已就绪(外部 D2H 已完成);完成前不得复写
        addr(rndv 下库持有该缓冲至对端落盘)。ref 为可选的 Python 对象
        保活(持有 addr 所属缓冲,如 bytes/bytearray)。
        """
        self._check_open()
        _check_tag(tag)
        if size < 0:
            raise ValueError(f"size {size} < 0")
        if not addr:
            raise ValueError(f"addr {addr!r} invalid(空指针直发会段错误)")
        op = UcxOp(tag, cb=cb, ref=ref)
        self._q.put(("send", tag, addr, size, op))
        return op

    def send_bytes(self, tag: int, data: bytes, *,
                   cb: Callable[[UcxOp], None] | None = None) -> UcxOp:
        """便捷发送:bytes 直发(rcache 隐式注册),ref 自动保活。"""
        self._check_open()
        _check_tag(tag)
        addr = ctypes.cast(ctypes.c_char_p(data), ctypes.c_void_p).value
        op = UcxOp(tag, cb=cb, ref=data)
        self._q.put(("send", tag, addr, len(data), op))
        return op

    def acquire_staging(self) -> "_Block":
        """发送向通道:取一块池内存做 D2H 暂存(仅 pool_staging 通道)。

        阻塞至有空块(池深=在途发送上限,承担背压;块由发送完成回调
        经 release_staging 归还后唤醒本等待——完成回调在库 IO 线程,
        本调用可在任意线程,典型为装配层发送线程)。shutdown 时抛错
        退出。块的 .buf/.addr 语义与接收块相同(rcache 首次 send 注册)。
        """
        if not self._pool_staging:
            raise RuntimeError(
                f"acquire_staging on non-staging channel {self._name}")
        while True:
            blk = self._pool.get(timeout=0.05)
            if blk is not None:
                return blk
            if self._stopped:
                raise RuntimeError(
                    f"staging wait aborted (shutdown) on {self._name}")

    def release_staging(self, blk: "_Block") -> None:
        """归还暂存块(发送完成回调调用;与 acquire 配对)。"""
        self._pool.release(blk)

    def recv(self, tag: int, *,
             cb: Callable[[UcxOp], None] | None = None) -> UcxOp:
        """预挂一条接收:完成时 op.data 为独立 bytes。

        同一 tag 可多次预挂(队列式);先到后挂走早到暂存(按 tag
        FIFO,同 tag 连发不丢帧;总条数上界 cfg.early_max,超出丢
        全局最早一条并告警)。
        """
        self._check_open()
        _check_tag(tag)
        op = UcxOp(tag, cb=cb)
        self._q.put(("recv", tag, op))
        return op

    def recv_into(self, tag: int, addr: int, cap: int, *,
                  cb: Callable[[UcxOp], None] | None = None) -> UcxOp:
        """预挂一条接收并直落调用方缓冲:完成即数据在 addr、op.size 为
        实际字节数。完成前 addr 归库(不得读写)。"""
        self._check_open()
        _check_tag(tag)
        if cap <= 0:
            raise ValueError(f"cap {cap} <= 0")
        op = UcxOp(tag, cb=cb, into_addr=addr, into_cap=cap)
        self._q.put(("recv", tag, op))
        return op

    def on_broken(self, cb: Callable[[BaseException], None]) -> None:
        """断链回调(IO 线程执行):在途 op 已全部 ERROR 终结之后触发。"""
        self._broken_cb = cb

    def on_peer_restart(self, cb: Callable[[], None]) -> None:
        """对端重启回调(IO 线程执行):握手 session 变化时触发。

        对端重启后其 tag 序列作废,本端预挂/早到已被清空;应用层
        的 tag 基线复位由本回调自行完成。
        """
        self._restart_cb = cb

    def fail(self, reason: BaseException | str) -> None:
        """应用侧检出不可恢复错(协议违例等):立即断链善后——在途
        op 全部 ERROR 终结、connect 侧自动重建/serve 侧等待重接。

        幂等;建议在 IO 线程回调内调用(协议错现场,善后逻辑按 IO
        线程独占设计);跨线程调用为尽力而为。
        """
        if self._stopped or self._link is None:
            return
        err = (reason if isinstance(reason, BaseException)
               else RuntimeError(str(reason)))
        self._fail_all_pending(err)

    def wait_ready(self, timeout: float | None = None) -> bool:
        """等首链就绪;False = 超时(首连仍在重试)。"""
        return self._ready.wait(timeout)

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and not self._broken

    @property
    def broken(self) -> bool:
        return self._broken

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_msg(self) -> int:
        """本端单条消息接收上界(= 接收块容量;告知对端发送方)。"""
        return self._block_bytes

    @property
    def peer_max_msg(self) -> int:
        """对端单条消息接收上界(发送侧超限即该条 op 报错)。"""
        return self._peer_max_msg

    @property
    def peer_session(self) -> int:
        return self._peer_session

    def close(self, timeout: float = 2.0) -> None:
        """关停通道:停线程、拆链路、未完成 op 以 ERROR 终结(不触发
        on_broken——有序关停不是断链)。"""
        self._stopped = True
        self._ready.set()
        self._broken_event.set()   # 唤醒监督/accept 线程退出
        with contextlib.suppress(queue.Full):
            self._q.put_nowait(None)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        for th in (self._sup_thread, self._acc_thread):
            if th is not None:
                th.join(timeout=timeout)
        if self._srv_fd is not None:
            with contextlib.suppress(Exception):
                _sock.c_close(self._srv_fd)
            self._srv_fd = None
        # 排空未 adopt 的链路
        while True:
            try:
                link = self._linkq.get_nowait()
            except queue.Empty:
                break
            link.close()
        self._cancel_posted_recvs()
        if self._link is not None:
            self._link.close()
            self._link = None
        # 未完成任务失败终结
        err = RuntimeError("channel closed")
        while True:
            try:
                task = self._q.get_nowait()
            except queue.Empty:
                break
            if task is not None:
                with contextlib.suppress(Exception):
                    task[-1]._finish(error=err)
        for q in self._expects.values():
            for op in q:
                with contextlib.suppress(Exception):
                    op._finish(error=err)
        self._expects.clear()
        self._early.clear()
        self._early_count = 0
        self._send_reqs.clear()

    def _cancel_posted_recvs(self) -> None:
        """close 前撤回未完成的预挂接收:ep FORCE close 不取消 recv,
        直接销毁 worker 会留 mpool 未归还告警(真机实录:4×8MiB 块)。
        cancel 为异步完成——排水至 _recv_reqs 清空或超时。"""
        link = self._link
        if link is None or not self._recv_reqs or _ucp is None:
            return
        cancel = getattr(_ucp, "ucp_request_cancel", None)
        if cancel is None:
            return
        for req in list(self._recv_reqs):
            with contextlib.suppress(Exception):
                cancel(ctypes.c_void_p(link.worker),
                       ctypes.c_void_p(req))
        deadline = time.monotonic() + 0.5
        while self._recv_reqs and time.monotonic() < deadline:
            with contextlib.suppress(Exception):
                link.progress()

    def __enter__(self) -> "UcxChannel":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._stopped:
            raise RuntimeError(f"ucx channel {self._name} closed")

    # ------------------------------------------------------------------ #
    # 建连(connect 监督线程 / serve accept 线程)                        #
    # ------------------------------------------------------------------ #
    def _supervise(self, host: str, port: int, auto_reconnect: bool) -> None:
        import errno as _errno
        backoff = 0.2
        tries = 0
        while not self._stopped:
            if self._ready.is_set() and not self._broken:
                self._broken_event.wait(timeout=1.0)
                continue
            if self._ready.is_set() and self._broken and not auto_reconnect:
                break   # 曾就绪后断链且不自动重连:监督退出
            try:
                link = self._build_link(host, port)
            except OSError as exc:
                tries += 1
                hint = (" [对端 listener 未就绪]"
                        if exc.errno == _errno.ECONNREFUSED else "")
                if tries % 10 == 1:
                    logger.warning(
                        "[ucx-chan] %s connect %s:%d deferred (%r), "
                        "retrying (backoff %.1fs)%s",
                        self._name, host, port, exc, backoff, hint)
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                continue
            except Exception as exc:  # noqa: BLE001
                tries += 1
                if tries % 10 == 1:
                    logger.warning(
                        "[ucx-chan] %s connect failed (%r), backoff %.1fs",
                        self._name, exc, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                continue
            tries = 0
            backoff = 0.2
            self._linkq.put(link)
            link.adopted.wait(timeout=10.0)

    def _accept_loop(self, port: int) -> None:
        while not self._stopped:
            try:
                fd, peer = _sock.c_accept(self._srv_fd)
            except OSError as exc:
                if self._stopped:
                    break
                logger.error("[ucx-chan] %s accept failed on port %d: %r",
                             self._name, port, exc)
                time.sleep(0.2)
                continue
            try:
                self._linkq.put(self._build_from_fd(fd))
                logger.warning("[ucx-chan] %s accepted from %s port %d",
                               self._name, peer, port)
            except Exception as exc:  # noqa: BLE001
                logger.error("[ucx-chan] %s handshake with %s failed: %r",
                             self._name, peer, exc)

    def _build_link(self, host: str, port: int) -> _ChanLink:
        """connect 侧:TCP 握手交换地址 + 建 ep;返回待 adopt 的链路。"""
        link = _ChanLink(self)
        try:
            mine = link.my_address()
            fd = _sock.c_open()
            try:
                if self._cfg_traffic_class is not None:
                    _sock.c_set_tos(fd, self._cfg_traffic_class << 2)
                _sock.c_connect_timeout(fd, host, port, self._hs_timeout_s)
                _sock.c_set_sock_timeouts(fd, self._hs_timeout_s)
                _hs_send(fd, mine, self._block_bytes, self._token,
                         link.caps_n_slots, link.caps_blob, self._put_thresh)
                addr, session, n_slots, pthresh, caps, pmax = _hs_recv(
                    fd, self._block_bytes, self._token)
            finally:
                _sock.c_close(fd)
            link.peer_session = session
            link.create_ep(addr)
            link.set_peer_caps(n_slots, pthresh, caps)
            link.peer_max_msg = pmax
        except Exception:
            link.close()
            raise
        return link

    def _build_from_fd(self, fd: int, hdr: bytes | None = None) -> _ChanLink:
        """serve 侧:accept 出的 fd 上完成握手(同样有界)。

        共享 listener 已先读固定头做 token 路由时,``hdr`` 直接传入,
        本方法只读 body;独立 serve 端口路径不传,自己读头。
        """
        link = _ChanLink(self)
        try:
            try:
                _sock.c_set_sock_timeouts(fd, self._hs_timeout_s)
                if hdr is None:
                    hdr = _hs_recv_exact(fd, _HS_HEADER.size)
                (alen, pmax, session, _token, n_slots, caps_len,
                 pthresh) = _hs_parse_header(
                    hdr, self._block_bytes, self._token)
                body = _hs_recv_exact(fd, alen + caps_len)
                addr = body[:alen]
                caps = body[alen:] if caps_len else b""
                _hs_send(fd, link.my_address(), self._block_bytes,
                         self._token, link.caps_n_slots, link.caps_blob,
                         self._put_thresh)
            finally:
                _sock.c_close(fd)
            link.peer_session = session
            link.create_ep(addr)
            link.set_peer_caps(n_slots, pthresh, caps)
            link.peer_max_msg = pmax
        except Exception:
            link.close()
            raise
        return link

    def _accept_fd(self, fd: int, hdr: bytes, peer: str) -> None:
        """共享 listener 路由到本通道后:完成握手并交给 IO 线程 adopt。"""
        link = self._build_from_fd(fd, hdr)
        self._linkq.put(link)
        logger.warning("[ucx-chan] %s accepted from %s port %d",
                       self._name, peer, self._listener.port
                       if self._listener is not None else -1)

    def _adopt_link(self, link: _ChanLink) -> None:
        """IO 线程内换链:拆旧、复位状态、分池预挂、恢复可用。"""
        self._broken_event.clear()
        try:
            if self._expects or self._send_reqs:
                # 旧链(或对端重启前)遗留的未完成 op:全部失败——对端
                # 重启后 tag 序列作废,悬空等待是死路
                self._fail_all_pending(RuntimeError(
                    "link replaced: in-flight ops dropped"))
            old = self._link
            self._link = None
            if old is not None:
                old.close()    # 拆毁由 IO 线程执行(唯一 progress 方)
            if not self._pool_staging:
                # staging 通道不 reset:在途暂存块经发送完成回调归还,
                # reset 会把已借出块重复灌回空闲队列(双借出=数据踩踏)
                self._pool.reset()
            # 早到暂存随旧链作废:新链消息流全新,旧链滞留字节属陈旧
            # 数据(对端不重放),残留会与新 tag 序列错配
            self._early.clear()
            self._early_count = 0
            self._link = link
            self._ep_err = None
            self._broken = False
            self._peer_max_msg = link.peer_max_msg
            self._reset_put_state()
            # put 模式装配(按方向独立;需两端同配 put_thresh>0):
            #   发送侧:对端发布了接收槽位 → 大帧可 put 直写
            #   接收侧:对端会向我 put → 数据预挂改小池(ctl),大块转为
            #           纯 put 落点,门铃独立分池
            self._put_armed = (self._put_thresh > 0
                               and link.peer_n_slots > 0)
            self._recv_ctl = (self._put_thresh > 0
                              and link.peer_put_thresh > 0)
            self._data_mask = (_CTL_BITS
                               if (self._put_armed or self._recv_ctl) else 0)
            self._ctl_pool = []
            self._db_pool = []
            self._ack_pool = []
            if self._recv_ctl:
                # 内联小消息上界 = 对端 put_thresh(低于该值走 eager)
                ctl_bytes = min(
                    self._block_bytes,
                    max(link.peer_put_thresh, 64 << 10) + 64)
                self._ctl_pool = [
                    _Block(i, ctl_bytes) for i in range(self._n_blocks)]
                self._db_pool = [_Block(i, 16) for i in range(self._n_blocks)]
                for blk in self._db_pool:
                    self._post_db(blk)
            if self._put_armed:
                self._ack_pool = [_Block(i, 16) for i in range(self._n_blocks)]
                for blk in self._ack_pool:
                    self._post_ack(blk)
                self._ack_bufs = [bytearray(16)
                                  for _ in range(self._n_blocks * 2)]
            if not self._pool_staging:
                # staging 通道不挂数据 wildcard(池块全部供发送暂存周转;
                # 单向"只发"通道本就无到达流量,put 模式的 ack 池照常挂)
                data_pool = (self._ctl_pool if self._recv_ctl
                             else self._pool.blocks)
                for blk in data_pool:
                    self._post_data(blk)
            restart = (self._peer_session != 0
                       and self._peer_session != link.peer_session)
            self._peer_session = link.peer_session
            self._ready.set()
            self._broken_event.clear()
            self._gen += 1
            if self._gen == 1:
                _log_ucx_env_once()
            logger.warning(
                "[ucx-chan] %s up gen=%d peer_session=%016x%s "
                "max_msg=%dMiB put=%s%s",
                self._name, self._gen, link.peer_session,
                " PEER-RESTARTED" if restart else "",
                self._block_bytes >> 20,
                "on" if (self._put_armed or self._recv_ctl) else "off",
                "" if not restart else " (on_peer_restart fired)")
            if restart and self._restart_cb is not None:
                with contextlib.suppress(Exception):
                    self._restart_cb()
            if self._gen > 1:
                logger.error(
                    "[ucx-chan] %s link REPLACED gen=%d peer_session=%016x"
                    "(对端重连/重启;若有第二个进程连入本端口同样表现"
                    "——边云各自应只有一个进程建 wire,serve 端 adopt "
                    "最后一个连接)", self._name, self._gen,
                    link.peer_session)
        except Exception as exc:  # noqa: BLE001
            logger.error("[ucx-chan] %s adopt link failed: %r",
                         self._name, exc)
            self._fail_all_pending(RuntimeError(f"adopt link failed: {exc}"))
        finally:
            link.adopted.set()

    # ------------------------------------------------------------------ #
    # IO 主循环                                                           #
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        last_log = 0.0
        while True:
            # 换链优先:监督/accept 线程产出的新链路先接管
            with contextlib.suppress(queue.Empty):
                link = self._linkq.get_nowait()
                if link is not None:
                    self._adopt_link(link)
            task = None
            # 首链未至时任务不出队(FIFO 原地保序):就绪后断链才 fail-fast
            if self._ready.is_set() or self._broken or self._stopped:
                with contextlib.suppress(queue.Empty):
                    task = self._q.get(timeout=0.005)
            else:
                time.sleep(0.005)
            if task is None and self._stopped:
                break
            if task is not None:
                if not self._ready.is_set() or self._broken:
                    task[-1]._finish(error=RuntimeError(
                        f"ucx channel {self._name} not available ("
                        + ("first connect pending)" if not self._ready.is_set()
                           else "reconnecting"
                           + (f", last ep status={self._ep_err})"
                              if self._ep_err else ")"))))
                    continue
                try:
                    self._dispatch(task)
                except Exception as exc:  # noqa: BLE001
                    self._broken = True
                    self._broken_event.set()
                    task[-1]._finish(error=exc)
                    logger.error("[ucx-chan] %s broken: %r",
                                 self._name, exc)
            try:
                self._progress_link()
            except Exception as exc:  # noqa: BLE001
                logger.error("[ucx-chan] %s progress error (continuing): %r",
                             self._name, exc)
            now = time.monotonic()
            if now - last_log >= 10.0:
                if (self._expects and self._expects_t0
                        and now - self._expects_t0 > 5.0):
                    # 预挂悬空>5s:帧未到(对端未发/线上丢失)或消费停滞
                    # ——正常节奏帧毫秒级到达,持续悬空即异常
                    logger.error(
                        "[ucx-chan] %s recv expects stuck %.1fs: tags=%d "
                        "min_tag=%d early=%d", self._name,
                        now - self._expects_t0, len(self._expects),
                        min(self._expects), self._early_count)
                logger.warning(
                    "[ucx-chan][stall] %s sends=%d expects=%d(tags=%d) "
                    "early=%d broken=%s",
                    self._name, len(self._send_reqs),
                    sum(len(q) for q in self._expects.values()),
                    len(self._expects), self._early_count, self._broken)
                last_log = now

    def _progress_link(self) -> None:
        if self._link is None or not self._ready.is_set():
            return
        self._link.progress()
        if self._ep_err is not None and not self._broken:
            self._fail_all_pending(
                RuntimeError(f"ucx ep error status={self._ep_err}"))

    def _on_ep_error(self, status: int) -> None:
        # 可能在 progress 内(IO 线程)触发;置位由 IO 线程消费
        self._ep_err = status
        logger.error("[ucx-chan] %s ep error status=%d",
                     self._name, status)

    def _dispatch(self, task: tuple) -> None:
        if task[0] == "send":
            _, tag, addr, size, op = task
            self._do_send(tag, addr, size, op)
        else:
            _, tag, op = task
            self._do_recv(tag, op)

    # ---------------- 发送:裸地址直发 / put 直写 ---------------------- #

    def _do_send(self, tag: int, addr: int, size: int, op: UcxOp) -> None:
        link = self._link
        if link is None:
            raise RuntimeError("no link")
        if self._peer_max_msg and size > self._peer_max_msg:
            # 调用方超限(对端块容量):只失败本条,不断链(区别于链路错误)
            op._finish(error=RuntimeError(
                f"size {size} > peer max_msg {self._peer_max_msg}:"
                f" raise peer cfg.block_mb"))
            return
        if self._put_armed and size >= self._put_thresh:
            self._put_send(link, tag, addr, size, op)
            return
        req = link.send_nb(addr, size, tag, self._send_cb)
        # restype=c_void_p:NULL 立即完成返回 None,一并按 0 处理
        if not req:
            op._finish(size=size)
            return
        if _ptr_is_err(req):
            op._finish(error=RuntimeError(
                f"ucp_tag_send_nbx failed status={_ptr_status(req)}"))
            self._fail_all_pending(RuntimeError("ucx send post failed"))
            return
        self._send_reqs[int(req)] = (op, tag, size)

    def _on_send_done(self, request, status, user_data=None) -> None:  # noqa: ANN001
        ent = self._send_reqs.pop(int(request), None)
        try:
            self._link.request_free(int(request))
        except Exception:  # noqa: BLE001
            pass
        if ent is None:
            return
        op, _tag, size = ent
        if status == _UCS_OK:
            op._finish(size=size)
        else:
            op._finish(error=RuntimeError(f"ucx send status={status}"))
            self._fail_all_pending(RuntimeError(f"ucx send status={status}"))

    # ---------------- RMA put:零握手直写 + 门铃/ACK ------------------- #

    def _put_send(self, link: _ChanLink, tag: int, addr: int, size: int,
                  op: UcxOp) -> None:
        """大消息单边 WRITE 直写对端槽位(tag % 对端槽数)。

        槽位复用由收方 ACK 门控——杜绝"发方覆写收方未消费槽"竞态。
        put 完成(RC CQE=数据已落对端)即调用方缓冲可释放,门铃携带
        (tag, size) 不依赖源缓冲内容。
        """
        slot = tag % link.peer_n_slots
        while (slot in self._slot_last
               and self._slot_acked.get(slot, -1) < self._slot_last[slot]):
            if self._broken or self._stopped or self._link is not link:
                raise RuntimeError(
                    f"ucx put slot {slot} wait aborted (link changed)")
            if not link.progress():
                time.sleep(0.0002)
        self._slot_last[slot] = tag
        req = link.put_nb(addr, size, link.peer_slot_addrs[slot],
                          link.peer_slot_rkeys[slot], self._put_cb)
        if not req:
            # 立即完成(罕见):直接门铃+收尾
            self._send_doorbell(tag, size)
            op._finish(size=size)
            return
        if _ptr_is_err(req):
            self._slot_last.pop(slot, None)
            op._finish(error=RuntimeError(
                f"ucp_put_nbx failed status={_ptr_status(req)}"))
            self._fail_all_pending(RuntimeError("ucx put post failed"))
            return
        self._send_reqs[int(req)] = (op, tag, size)

    def _on_put_done(self, request, status, user_data=None) -> None:  # noqa: ANN001
        ent = self._send_reqs.pop(int(request), None)
        try:
            self._link.request_free(int(request))
        except Exception:  # noqa: BLE001
            pass
        if ent is None:
            return
        op, tag, size = ent
        if status == _UCS_OK:
            # CQE=数据已落对端:门铃后必安全(不再依赖源缓冲)
            self._send_doorbell(tag, size)
            op._finish(size=size)
        else:
            op._finish(error=RuntimeError(f"ucx put status={status}"))
            self._fail_all_pending(RuntimeError(f"ucx put status={status}"))

    def _send_doorbell(self, tag: int, size: int) -> None:
        buf = bytearray(_DB.size)
        _DB.pack_into(buf, 0, tag, size)
        self._ctl_send(buf, tag | _TAG_DOORBELL)

    def _send_ack(self, tag: int) -> None:
        buf = self._ack_bufs.pop(0) if self._ack_bufs else bytearray(16)
        _AKP.pack_into(buf, 0, tag)
        self._ctl_send(buf, tag | _TAG_ACK)

    def _ctl_send(self, buf: bytearray, tag: int) -> None:
        """小控制消息(门铃/ACK):eager 单消息;请求与缓冲同挂
        _db_pend 防覆写/GC,完成后丢弃。"""
        addr = ctypes.addressof((ctypes.c_char * len(buf)).from_buffer(buf))
        req = self._link.send_nb(addr, len(buf), tag, self._db_cb)
        if req:
            if _ptr_is_err(req):
                logger.error("[ucx-chan] %s ctl send failed status=%d",
                             self._name, _ptr_status(req))
                self._broken = True   # 门铃丢失=对端槽位死等,须断链重建
            else:
                self._db_pend[int(req)] = buf

    def _on_db_done(self, request, status, user_data=None) -> None:  # noqa: ANN001
        try:
            self._link.request_free(int(request))
        except Exception:  # noqa: BLE001
            pass
        self._db_pend.pop(int(request), None)
        if status != _UCS_OK:
            self._broken = True
            logger.error("[ucx-chan] %s ctl msg status=%d",
                         self._name, status)

    def _reset_put_state(self) -> None:
        """换链/断链复位:槽位门控表/挂起控制消息作废(请求随旧链销毁)。"""
        self._slot_last.clear()
        self._slot_acked.clear()
        self._db_pend.clear()

    # ---------------- 接收:分池 masked 预挂 + tag 配对 ---------------- #

    def _post_data(self, blk: _Block) -> None:
        self._post_masked(blk, 0, self._data_mask, "data")

    def _post_db(self, blk: _Block) -> None:
        self._post_masked(blk, _TAG_DOORBELL, _TAG_DOORBELL, "db")

    def _post_ack(self, blk: _Block) -> None:
        self._post_masked(blk, _TAG_ACK, _TAG_ACK, "ack")

    def _post_masked(self, blk: _Block, tag: int, mask: int,
                     kind: str) -> None:
        req = self._link.recv_masked_nb(
            blk.addr, len(blk.buf), tag, mask, self._recv_cb)
        if not req or _ptr_is_err(req):
            raise RuntimeError(f"ucp_tag_recv_nbx failed req={req!r}")
        blk.recv_req = req
        self._recv_reqs[int(req)] = (blk, kind)

    def _on_recv_done(self, request, status, info, user_data=None) -> None:  # noqa: ANN001
        ent = self._recv_reqs.pop(int(request), None)
        try:
            self._link.request_free(int(request))
        except Exception:  # noqa: BLE001
            pass
        if ent is None:
            return
        blk, kind = ent
        if status != _UCS_OK:
            if self._stopped:
                return   # close 撤回的取消完成:顶部已 free/pop,静默
            self._broken = True
            self._fail_all_pending(RuntimeError(
                f"ucx recv status={status} on {self._name}"))
            return
        length = int(info.contents.length)
        sender_tag = int(info.contents.sender_tag)
        if kind == "ack":
            self._post_ack(blk)
            tag = sender_tag & _TAG_USER_MAX
            slot = tag % self._link.peer_n_slots
            if tag > self._slot_acked.get(slot, -1):
                self._slot_acked[slot] = tag
            return
        if kind == "db":
            self._post_db(blk)
            tag, size = _DB.unpack_from(blk.buf, 0)
            self._on_doorbell(tag, size)
            return
        # 数据(inline):数据就在本接收块内,拷出即重挂
        data = bytes(memoryview(blk.buf)[:length])
        self._post_data(blk)
        self._deliver(sender_tag & _TAG_USER_MAX, data)

    def _on_doorbell(self, tag: int, size: int) -> None:
        """门铃:数据已落本端槽位 blocks[tag % n]。消费(拷出即完成)后
        回 ACK——不 ACK 会让发方槽位门控永久卡死。"""
        try:
            if size > self._block_bytes:
                raise ConnectionError(
                    f"doorbell size {size} > block {self._block_bytes} "
                    f"on {self._name}")
            slot = self._pool.blocks[tag % self._n_blocks]
            data = bytes(memoryview(slot.buf)[:size])
            self._deliver(tag, data)
        except Exception as exc:  # noqa: BLE001
            self._broken = True
            self._fail_all_pending(exc)
            return
        finally:
            if not self._broken:
                self._send_ack(tag)

    def _do_recv(self, tag: int, op: UcxOp) -> None:
        q = self._early.get(tag)
        if q:
            data = q.popleft()
            self._early_count -= 1
            if not q:
                del self._early[tag]
            self._settle(op, data)
            return
        if not self._expects:
            self._expects_t0 = time.monotonic()
        self._expects.setdefault(tag, deque()).append(op)

    def _deliver(self, tag: int, data: bytes) -> None:
        q = self._expects.get(tag)
        if q:
            op = q.popleft()
            if not q:
                del self._expects[tag]
            self._settle(op, data)
            if not self._expects:
                self._expects_t0 = 0.0
            return
        # 早到:无预挂可配对,按 tag 入 FIFO(与预挂队列对称,同 tag
        # 连发不丢帧);总条数上界 early_max,满则丢全局最早一条并告警
        if self._early_count >= self._early_max:
            old_tag = next(t for t, dq in self._early.items() if dq)
            logger.warning(
                "[ucx-chan] %s early store full (%d): drop tag=%d"
                "——提高 cfg.early_max 或及时预挂", self._name,
                self._early_max, old_tag)
            dq = self._early[old_tag]
            dq.popleft()
            self._early_count -= 1
            if not dq:
                del self._early[old_tag]
        self._early.setdefault(tag, deque()).append(data)
        self._early_count += 1

    def _settle(self, op: UcxOp, data: bytes) -> None:
        if op._into_addr:
            if len(data) > op._into_cap:
                op._finish(error=RuntimeError(
                    f"recv_into cap {op._into_cap} < {len(data)}B"))
                return
            ctypes.memmove(op._into_addr, data, len(data))
            op._finish(size=len(data))
            return
        op._finish(data=data)

    # ---------------- 断链善后 ---------------------------------------- #

    def _fail_all_pending(self, error: BaseException) -> None:
        """断链善后(统一入口,幂等):在途 send 与等待配对(expects)的
        op 全部失败,清请求路由表,置断链事件,触发 on_broken。"""
        self._broken = True
        for op, _tag, _size in self._send_reqs.values():
            with contextlib.suppress(Exception):
                op._finish(error=error)
        self._send_reqs.clear()
        self._recv_reqs.clear()
        for q in self._expects.values():
            for op in q:
                with contextlib.suppress(Exception):
                    op._finish(error=error)
        self._expects.clear()
        self._expects_t0 = 0.0
        self._early.clear()
        self._early_count = 0
        self._reset_put_state()
        self._broken_event.set()
        logger.error("[ucx-chan] %s broken (%s): %r", self._name,
                     "自动重连中" if self._role == "connect"
                     else "等待对端重连", error)
        cb = self._broken_cb
        if cb is not None:
            try:
                cb(error)
            except Exception:  # noqa: BLE001
                logger.error("[ucx-chan] %s on_broken cb error",
                             self._name, exc_info=True)


def _check_tag(tag: int) -> None:
    if not 0 <= tag <= _TAG_USER_MAX:
        raise ValueError(
            f"tag {tag} out of range [0, 2^61):高 3 位保留(put 门铃/ACK)")


class UcxListener:
    """一个 TCP listener,按 UHC1 握手 token 把连接投递给多个 UcxChannel。

    只共享 accept/握手入口;每个 channel 仍有独立 UCP ep/IO 线程/块池。
    当前 1e1c 用两个 token(UP/DOWN);多边时 token 里含 edge_id。
    """

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self.port = int(port)
        self._fd = _sock.c_open()
        self._stopped = False
        self._thread: threading.Thread | None = None
        self._routes: dict[int, UcxChannel] = {}
        self._lock = threading.Lock()
        self._traffic_class: int | None = None
        _sock.c_setsockopt_int(
            self._fd, _sock._SOL_SOCKET, _sock._SO_REUSEADDR, 1,
            what="SO_REUSEADDR",
        )
        _sock.c_bind(self._fd, host, self.port)
        _sock.c_listen(self._fd, 16)

    def set_traffic_class(self, traffic_class: int | None) -> None:
        """Mark the handshake listener's TOS (accepted connections inherit
        it). Called at channel-register time, before the accept loop
        starts; channels sharing one listener must share one tc."""
        if traffic_class is None or self._traffic_class is not None:
            return
        self._traffic_class = traffic_class
        _sock.c_set_tos(self._fd, int(traffic_class) << 2)

    def register(self, token: int, channel: UcxChannel, *,
                 name: str = "") -> None:
        self.set_traffic_class(channel._cfg_traffic_class)
        token = int(token)
        with self._lock:
            if token in self._routes:
                raise ValueError(
                    f"duplicate UHC1 token {token:#x} on listener "
                    f"{self._host}:{self.port} ({name})"
                )
            self._routes[token] = channel

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._accept_loop,
            name=f"ucx-chan-acc-{self.port}", daemon=True,
        )
        self._thread.start()
        logger.warning(
            "[ucx-chan] unified listener %s:%d tokens=%d %s",
            self._host, self.port, len(self._routes),
            [hex(t) for t in sorted(self._routes)],
        )

    def _accept_loop(self) -> None:
        while not self._stopped:
            try:
                fd, peer = _sock.c_accept(self._fd)
            except OSError as exc:
                if self._stopped:
                    break
                logger.error(
                    "[ucx-chan] unified accept failed on %s:%d: %r",
                    self._host, self.port, exc,
                )
                time.sleep(0.2)
                continue
            threading.Thread(
                target=self._handshake, args=(fd, peer),
                name=f"ucx-chan-hs-{self.port}", daemon=True,
            ).start()

    def _handshake(self, fd: int, peer: str) -> None:
        owned = True
        try:
            _sock.c_set_sock_timeouts(fd, _DEFAULT_HS_TIMEOUT_S)
            hdr = _hs_recv_exact(fd, _HS_HEADER.size)
            fields = _hs_parse_header(hdr)
            token = fields[_HS_TOKEN_INDEX]
            with self._lock:
                ch = self._routes.get(token)
            if ch is None:
                logger.error(
                    "[ucx-chan] unknown token %#x from %s on %s:%d",
                    token, peer, self._host, self.port,
                )
                return
            # channel 的 _build_from_fd 负责读 body、回握手并 close fd;
            # 从这里开始 fd 所有权交给 channel,避免 double close。
            owned = False
            ch._accept_fd(fd, hdr, peer)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[ucx-chan] handshake with %s failed on %s:%d: %r",
                peer, self._host, self.port, exc,
            )
        finally:
            if owned:
                with contextlib.suppress(Exception):
                    _sock.c_close(fd)

    def close(self) -> None:
        self._stopped = True
        if self._fd is not None:
            with contextlib.suppress(Exception):
                _sock.c_close(self._fd)
            self._fd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
