# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic host-to-host UCX transport library (UHC1 protocol family).

Entry points: ``LwdUcxChannel.connect`` on the client side,
``LwdUcxChannel.serve`` for a dedicated port and
``LwdUcxChannel.serve_via`` to share an existing ``LwdUcxListener``;
data moves through ``send``/``send_bytes`` and ``recv``/``recv_into``,
while ``on_broken``/``on_peer_restart`` report asynchronous state.

Positioning: a tagged byte pipe. Channel creation (any number of
channels, not bound to up/down), send (host address in), recv
(standalone bytes or a caller buffer out), RDMA registration (implicit
through the libucp rcache), the IO thread, automatic reconnect on a
broken link and peer-restart detection are all internal. Device copies
(D2H/H2D) and seqno/aux orchestration belong to the caller -- this file
imports no torch, so it composes with any NPU/GPU/CPU side split.

Composition (additive; nothing existing changes):
    * reuses the ucx_ucp.py bindings, the sock.py TCP helpers, the
      ucx_pool.py block pool and the _UcxLink link object (extended by
      the _ChanLink subclass with masked-tag pre-posting, now in
      ucx_link.py);
    * the wire format is the UHC1 family: a byte stream with no frame
      header at all -- the length is self-describing from the UCP
      completion and the tag travels as a plain API argument. The
      software CRC is gone: RoCE FCS/ICRC plus RC retransmission already
      cover the wire end to end, and the CRC only ever diagnosed the
      library's own staging copy, which the send path does not keep.
      Rollout unit is one whole channel pair: both peers must speak
      UHC1.

Interface contract (hard requirements):
  * send(tag, addr, size): once the call returns, the library may read
    the bytes at addr (the caller guarantees D2H has completed); the
    caller must not rewrite addr before the op completes OK (under rndv
    the library holds that buffer until the peer has landed it), and the
    buffer is free to reuse after that;
  * recv(tag): delivers standalone bytes, and holding a reference is all
    the caller needs to start the H2D (GC keeps it alive until the copy
    finishes). One tag may be pre-posted many times (queue style: each
    arrival consumes one post, i.e. MPI-like streaming recv); a frame
    that arrives before its post is parked per tag in FIFO order (one tag
    sent repeatedly loses nothing), bounded to early_max entries in
    total;
  * recv_into(tag, addr, cap): on completion the data is in addr; until
    then addr belongs to the library (the receive side still relays
    through an internal block, i.e. one host copy into addr; exact-tag
    pre-posting with no on-wire copy is a later extension);
  * broken link: every in-flight op ends with ERROR (delivery state
    unknown, never replayed); the connect side reconnects automatically
    (can be disabled), the serve side loops on accept to replace the
    link; a peer restart (session fingerprint change) is reported through
    the on_peer_restart callback -- resetting tag baselines is the
    caller's job;
  * tag range [0, 2^61): the top 3 bits are reserved for the put-mode
    doorbell/ACK internal markers.

put large-frame path (optional; takes effect only with cfg.put_thresh >
0 on both peers): the sender writes straight into a peer slot
(tag % n_slots) with a one-sided ucp_put_nbx, then sends a 16 B
doorbell (tag, length); the receiver ACKs after consuming, and that ACK
gates slot reuse. Small messages still go eager-tag inline. Once
enabled, receives pre-post into separate masked pools (data, doorbell
and ACK each get their own post, so they never mismatch).

Deployment prerequisites: both peers point LD_LIBRARY_PATH at UCX 1.22
and pin TLS per NIC (hns: rc_verbs,tcp,self / mlx5: rc_mlx5 or
rc_verbs).
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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger
from vllm.v1.lwd_comm.data_comm import sock as _sock
from vllm.v1.lwd_comm.data_comm.ucx_link import (
    _UcxLink,
)
from vllm.v1.lwd_comm.data_comm.ucx_pool import LwdUcxBlock, LwdUcxBlockPool
from vllm.v1.lwd_comm.data_comm.ucx_ucp import (
    _UCS_OK,
    _ptr_is_err,
    _ptr_status,
    _RecvCb,
    _req_param,
    _require_ucp,
    _SendCb,
)

logger = init_logger(__name__)


class LwdHostDoneEvent:
    """Completion event for a send future (host side, device-agnostic).

    query() serves LwdCommFuture's done()/wait()/_finalize polling path;
    wait/wait_event are no-ops, because the IO thread already serialises
    frame order on the send link.
    """

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
        """Reverse path of torch Stream.wait_event: a host event carries
        no device ordering semantics."""


# ucp_request_cancel: the close path withdraws pre-posted receives that
# never completed. Bound here on demand so shared ucx_ucp.py stays as is;
# a missing libucp must not break importing this module.
def _bind_request_cancel() -> Any:
    try:
        ucp = _require_ucp()
    except RuntimeError:
        return None
    ucp.ucp_request_cancel.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    ucp.ucp_request_cancel.restype = ctypes.c_void_p
    return ucp.ucp_request_cancel


_ucp_request_cancel = _bind_request_cancel()


# --------------------------------------------------------------------- #
# Configuration (construction-time; defaults stay light)                #
# --------------------------------------------------------------------- #
@dataclass
class LwdUcxConfig:
    """Transport configuration for one channel.

    Both peers must use the same block_mb: the larger sender truncates
    on the smaller receiver.

    block_mb:    per-block capacity. Receive side: the receive block,
                 i.e. the peer's per-message ceiling. pool_staging
                 channels: the D2H staging block, i.e. this side's
                 per-send ceiling.
    n_blocks:    number of receive blocks (the default early-frame
                 store size; also the put slot count).
    put_thresh: >0 enables the RMA put large-frame path. Needs the same
                 setting on both peers and UCX_RNDV_THRESH unset, or
                 UCX turns small inline frames into rndv.
    hs_timeout_s: TCP handshake timeout (bounded on both the connect and
                 the accept side).
    token:       channel identity, cross-checked in the handshake so a
                 stray connect cannot land on someone else's port.
                 Default 0 validates nothing, it only asks both peers to
                 agree.
    early_max:   bound on stored early frames (count; 0 = n_blocks).
    traffic_class: IB Traffic Class / RoCEv2 DSCP (None = UCX auto:
                 RoCEv2 takes the port's global TC, else 106). Same on
                 both peers.
    sl:          IB Service Level / RoCEv2 Ethernet priority PCP
                 (None = 0). Switch PFC/ETS steering keys on DSCP plus
                 PCP, so this is usually configured together with
                 traffic_class. Same on both peers.
    """

    block_mb: int = 64
    n_blocks: int = 4
    put_thresh: int = 0
    hs_timeout_s: float = 5.0
    token: int = 0
    early_max: int = 0  # total early-frame store bound (0 = n_blocks)
    traffic_class: int | None = None
    sl: int | None = None
    pool_staging: bool = False  # True: this channel's pool blocks serve only as
    # send D2H staging (turned around by
    # acquire_staging/release_staging) and no
    # wildcard is posted -- one-way "send only"
    # channels.


# tag bit layout: the low 61 bits belong to the application, the top 2 bits
# are put-mode control-message markers
_TAG_ACK = 1 << 61  # receiver -> sender: slot consumed (gates reuse)
_TAG_DOORBELL = 1 << 62  # sender -> receiver: data in your slot, come and consume
_TAG_USER_MAX = (1 << 61) - 1
_CTL_BITS = _TAG_ACK | _TAG_DOORBELL

# control-message payloads: doorbell (tag, length); ACK (tag)
_DB = struct.Struct("!QQ")
_AKP = struct.Struct("!Q")

# Handshake (UHC1): TCP exchange of worker address, put slot caps,
# capacity, session and identity
_HS_MAGIC = b"UHC1\0\0\0\0"
# Handshake header: magic, addr_len, max_msg, session, token, n_slots,
# caps_len, put_thresh.
_HS_FMT = "!8sIQQQIII"
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
            "[ucx-chan] UCX_TLS is unset; using the UCX default. Pin it "
            "explicitly: rc_verbs,tcp,self for hns; rc_mlx5 or rc_verbs "
            "for mlx5"
        )
    else:
        logger.warning(
            "[ucx-chan] env: UCX_TLS=%r UCX_NET_DEVICES=%r",
            tls,
            os.getenv("UCX_NET_DEVICES", "") or "<default>",
        )


# --------------------------------------------------------------------- #
# Handshake (TCP: worker address + caps, plus session/token)            #
# --------------------------------------------------------------------- #
def _hs_send(
    fd: int,
    addr: bytes,
    max_msg: int,
    token: int,
    n_slots: int,
    caps: bytes,
    put_thresh: int,
) -> None:
    _sock.c_send_bytes(
        fd,
        struct.pack(
            _HS_FMT,
            _HS_MAGIC,
            len(addr),
            max_msg,
            _HS_SESSION,
            token,
            n_slots,
            len(caps),
            put_thresh,
        )
        + addr
        + caps,
    )


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
    """Validate a UHC1 handshake header; return its fixed-length fields.

    ``local_max`` / ``expect_token`` of None skip the matching check --
    the listener only reads the header to route on the token, while the
    target channel still runs the full max/token checks before it reads
    the body.
    """
    magic, alen, max_msg, session, token, n_slots, caps_len, put_thresh = (
        _HS_HEADER.unpack(hdr)
    )
    if magic != _HS_MAGIC:
        raise ConnectionError(
            f"bad handshake magic {magic!r}: the peer is not UHC1 "
            f"(ucx_channel); the legacy LWD wire cannot interoperate, "
            f"both ends must run the same library"
        )
    if alen <= 0 or alen > (1 << 20):
        raise ConnectionError(f"bad handshake address length {alen}")
    if caps_len > (1 << 24) or n_slots > 64:
        raise ConnectionError(f"bad handshake caps {caps_len}/{n_slots}")
    if expect_token is not None and token != expect_token:
        raise ConnectionError(
            f"handshake token mismatch {token:#x} != {expect_token:#x}:"
            f"the peer is not on this channel (check the token and port "
            f"configuration on both ends)"
        )
    if local_max is not None and max_msg > local_max:
        # The peer's per-message ceiling exceeds our receive block, so an oversize
        # message would be truncated -- fail at startup instead
        raise ConnectionError(
            f"peer max_msg {max_msg} > local {local_max}: block_mb must "
            f"match on both ends (see cfg.block_mb)"
        )
    return (
        alen,
        max_msg,
        session,
        token,
        n_slots,
        caps_len,
        put_thresh,
    )


def _hs_recv(
    fd: int, local_max: int, expect_token: int
) -> tuple[bytes, int, int, int, bytes, int]:
    """Receive the peer handshake.

    Returns (addr, session, n_slots, put_thresh, caps, max_msg).
    """
    hdr = _hs_recv_exact(fd, _HS_HEADER.size)
    alen, max_msg, session, _token, n_slots, caps_len, put_thresh = _hs_parse_header(
        hdr, local_max, expect_token
    )
    body = _hs_recv_exact(fd, alen + caps_len)
    return (
        body[:alen],
        session,
        n_slots,
        put_thresh,
        body[alen:] if caps_len else b"",
        max_msg,
    )


# --------------------------------------------------------------------- #
# Completion handle                                                     #
# --------------------------------------------------------------------- #
class LwdUcxOp:
    """Completion handle for one send/recv.

    Completion is signalled on the IO thread; done/wait may be called
    from any thread. On success: send -> size is valid; recv -> data
    (bytes) is valid; recv_into -> size is valid and the data is already
    in the caller buffer. On failure: wait raises RuntimeError and error
    is set. The user callback (cb) runs on the IO thread, so it may only
    submit asynchronous work -- it must never block.
    """

    __slots__ = (
        "tag",
        "size",
        "data",
        "error",
        "_ev",
        "_cb",
        "_ref",
        "_into_addr",
        "_into_cap",
    )

    def __init__(
        self,
        tag: int,
        cb: Callable[[LwdUcxOp], None] | None = None,
        ref: Any = None,
        into_addr: int = 0,
        into_cap: int = 0,
    ):
        self.tag = tag
        self.size = 0
        self.data: bytes | None = None
        self.error: BaseException | None = None
        self._ev = threading.Event()
        self._cb = cb
        self._ref = ref  # keeps the send source alive (e.g. bytes/bytearray)
        self._into_addr = into_addr
        self._into_cap = into_cap

    def _finish(
        self,
        *,
        data: bytes | None = None,
        size: int | None = None,
        error: BaseException | None = None,
    ) -> None:
        if self._ev.is_set():
            return
        self.data = data
        self.size = size if size is not None else (len(data) if data is not None else 0)
        self.error = error
        self._ref = None  # drop the send source keep-alive
        self._ev.set()
        cb = self._cb
        if cb is not None:
            try:
                cb(self)
            except Exception:  # noqa: BLE001
                logger.exception("[ucx-chan] op cb error tag=%s", self.tag)

    def done(self) -> bool:
        """Non-blocking query (pure host, never touches the device)."""
        return self._ev.is_set()

    def wait(self, timeout: float | None = None) -> LwdUcxOp:
        """Block until completion.

        Raises RuntimeError on failure, TimeoutError on timeout.
        """
        if not self._ev.wait(timeout):
            raise TimeoutError(
                f"ucx channel op tag={self.tag} wait timed out ({timeout}s)"
            )
        if self.error is not None:
            raise RuntimeError(f"ucx channel op tag={self.tag} failed: {self.error}")
        return self

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        state = "ERR" if self.error else ("OK" if self._ev.is_set() else "PEND")
        return f"<LwdUcxOp tag={self.tag} {state} size={self.size}>"


# --------------------------------------------------------------------- #
# Link: _UcxLink reuse + masked-tag pre-post + err routing              #
# --------------------------------------------------------------------- #
class _ChanLink(_UcxLink):
    """UHC1 link: lifecycle, handshake address and put capability all
    come from _UcxLink (its owner must expose _pool/_block_bytes/
    _put_thresh, supplied by LwdUcxChannel)."""

    def __init__(self, owner: LwdUcxChannel) -> None:
        super().__init__(owner)
        self.peer_max_msg = 0  # set by the handshake (peer per-message ceiling)

    def recv_masked_nb(
        self, addr: int, length: int, tag: int, mask: int, cb: Any
    ) -> int:
        """Post a masked tag receive (matches when (msg_tag & mask) ==
        tag): the basis for routing data, doorbells and ACKs into
        separate pools. Returns the request pointer."""
        return _require_ucp().ucp_tag_recv_nbx(
            ctypes.c_void_p(self._worker),
            ctypes.c_void_p(addr),
            length,
            tag,
            mask,
            ctypes.byref(_req_param(cb)),
        )

    def _on_ep_err(self, arg, ep, status) -> None:  # noqa: ANN001
        # Override the base-class route: the owner is a LwdUcxChannel and does not
        # go through the shared _WIRES registry
        self._owner._on_ep_error(int(status))


# --------------------------------------------------------------------- #
# Channel                                                               #
# --------------------------------------------------------------------- #
class LwdUcxChannel:
    """One bidirectional host-to-host channel: one ep, one IO thread and
    one set of receive blocks.

    Threading model: send/recv may be submitted from any thread and never
    block (an internal task queue absorbs them); completion callbacks and
    user cb all fire on the IO thread, the same thread that dispatches
    tasks, which keeps them lock-free. Link setup is done by the
    supervisor thread (connect side: first connect plus rebuild after a
    break) or by the accept thread (serve side: loop on accept); the
    resulting link reaches the IO thread through _linkq and is adopted
    there. From then on only the IO thread touches worker/ep.

    First-connect window semantics: tasks submitted before a link is
    ready wait in the queue and keep their order (they do not fail); once
    ready, a broken link terminates in-flight and later tasks with ERROR
    immediately (fail-fast), and service resumes after a new link is
    adopted.
    """

    def __init__(self, cfg: LwdUcxConfig | None = None, *, name: str = "") -> None:
        c = cfg or LwdUcxConfig()
        self._block_bytes = max(1, int(c.block_mb)) << 20
        self._n_blocks = max(2, int(c.n_blocks))
        self._put_thresh = max(0, int(c.put_thresh))
        # UCT-level config injection (see _ChanLink: read, modify, init): keys are
        # UCX config names without the UCX_ prefix, written by ucp_config_modify
        # before init.
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
        self._role = ""  # "connect" / "serve" (for logging)
        # Owner attributes _UcxLink relies on (_pool/_block_bytes/_put_thresh)
        self._pool = LwdUcxBlockPool(self._n_blocks, self._block_bytes)
        if self._pool_staging:
            # staging channel: pool blocks must be usable from construction
            # (LwdUcxBlockPool starts with an empty free queue, filled only by
            # adopt/reset -- a staging channel skips reset, so the first
            # acquire_staging would never get a block: observed on real hardware as
            # the edge sender thread stuck in acquire_staging while the cloud side sat
            # in recv unbound)
            for blk in self._pool.blocks:
                self._pool.release(blk)
        # recv pairing: tag -> queue of waiters (one tag may be pre-posted several
        # times, each arrival consumes one); _expects_t0 = the moment the queue
        # went from empty to non-empty (used for stuck detection)
        self._expects: dict[int, deque[LwdUcxOp]] = {}
        self._expects_t0 = 0.0
        # early frames (arrived before the post): tag -> FIFO of bytes (one tag
        # sent repeatedly loses nothing, arrivals pair up with posts one by one);
        # total count bounded by _early_max
        self._early: dict[int, deque[bytes]] = {}
        self._early_count = 0
        # request pointer -> completion routing (send/put: (op, tag, size))
        self._send_reqs: dict[int, tuple[LwdUcxOp, int, int]] = {}
        # request pointer -> (block, pool kind), used to re-post
        self._recv_reqs: dict[int, tuple[LwdUcxBlock, str]] = {}
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
        # put-path state, owned by the IO thread
        self._put_armed = False  # send side: put can hit peer slots directly
        self._recv_ctl = False  # recv side: large -> put slots, inline -> small
        self._data_mask = 0  # data pre-post mask (skips ctl bits under put)
        self._ctl_pool: list[LwdUcxBlock] = []
        self._db_pool: list[LwdUcxBlock] = []
        self._ack_pool: list[LwdUcxBlock] = []
        self._ack_bufs: list[bytearray] = []
        self._slot_last: dict[int, int] = {}
        self._slot_acked: dict[int, int] = {}
        self._db_pend: dict[int, bytearray] = {}
        # user callbacks
        self._broken_cb: Callable[[BaseException], None] | None = None
        self._restart_cb: Callable[[], None] | None = None
        # UCP callback closures must outlive the channel: a collected CFUNCTYPE
        # object leaves a dangling callback
        self._send_cb = _SendCb(self._on_send_done)
        self._recv_cb = _RecvCb(self._on_recv_done)
        self._put_cb = _SendCb(self._on_put_done)
        self._db_cb = _SendCb(self._on_db_done)
        # threads
        self._sup_thread: threading.Thread | None = None
        self._acc_thread: threading.Thread | None = None
        self._srv_fd: int | None = None
        self._listener: LwdUcxListener | None = None
        self._thread = threading.Thread(
            target=self._run, name=f"ucx-chan-io-{self._name}", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------ #
    # Channel creation                                                   #
    # ------------------------------------------------------------------ #
    @classmethod
    def connect(
        cls,
        host: str,
        port: int,
        *,
        cfg: LwdUcxConfig | None = None,
        auto_reconnect: bool = True,
    ) -> LwdUcxChannel:
        """Client side: connect in the background (retry with backoff);
        rebuild the link automatically after a break (can be disabled).

        Returns the channel object at once (non-blocking); wait_ready()
        waits for the first connect, and submissions made before that
        wait in the queue in order.
        """
        ch = cls(cfg, name=f"cli:{host}:{port}")
        ch._role = "connect"
        logger.warning(
            "[ucx-chan] %s connect token=%#x hs=%ss",
            ch._name,
            ch._token,
            ch._hs_timeout_s,
        )
        ch._sup_thread = threading.Thread(
            target=ch._supervise,
            args=(host, port, auto_reconnect),
            name=f"ucx-chan-sup-{ch._name}",
            daemon=True,
        )
        ch._sup_thread.start()
        return ch

    @classmethod
    def serve(
        cls, host: str, port: int, *, cfg: LwdUcxConfig | None = None
    ) -> LwdUcxChannel:
        """Server side: loop on accept -- both the first connect and the
        reconnect after a break come in on the same port, and the new link
        replaces the old one via adopt (the channel object and its
        pre-post state survive the reconnect).

        One port per channel; LWD 1e1c shares one port between UP and
        DOWN through ``LwdUcxListener`` + ``serve_via`` (routed by
        handshake token).
        """
        ch = cls(cfg, name=f"srv:{port}")
        ch._role = "serve"
        ch._srv_fd = _sock.c_open()
        if ch._cfg_traffic_class is not None:
            # set before bind: accepted connections inherit the listener TOS
            _sock.c_set_tos(ch._srv_fd, ch._cfg_traffic_class << 2)
        _sock.c_setsockopt_int(
            ch._srv_fd,
            _sock._SOL_SOCKET,
            _sock._SO_REUSEADDR,
            1,
            what="SO_REUSEADDR",
        )
        _sock.c_bind(ch._srv_fd, host, port)
        _sock.c_listen(ch._srv_fd, 1)
        ch._acc_thread = threading.Thread(
            target=ch._accept_loop,
            args=(port,),
            name=f"ucx-chan-acc-{port}",
            daemon=True,
        )
        ch._acc_thread.start()
        logger.warning(
            "[ucx-chan] %s handshake listening on %s:%d", ch._name, host, port
        )
        return ch

    @classmethod
    def serve_via(
        cls, listener: LwdUcxListener, *, cfg: LwdUcxConfig | None = None
    ) -> LwdUcxChannel:
        """Shared-listener form: the channel does not bind, the listener
        hands it connections by token.

        The listener must be registered before it is started; the
        listener's accept thread replaces the channel's own, and the
        IO/adopt logic is unchanged.
        """
        c = cfg or LwdUcxConfig()
        ch = cls(c, name=f"srv:{listener.port}:{c.token:x}")
        ch._role = "serve"
        ch._listener = listener
        listener.register(ch._token, ch, name=ch._name)
        return ch

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def send(
        self,
        tag: int,
        addr: int,
        size: int,
        *,
        ref: Any = None,
        cb: Callable[[LwdUcxOp], None] | None = None,
    ) -> LwdUcxOp:
        """Send size bytes starting at addr (the tag travels with the
        message to the peer).

        Contract: when the call returns the bytes are ready (the caller
        has completed D2H); addr must not be rewritten before completion
        (under rndv the library holds that buffer until the peer has
        landed it). ref is an optional Python object kept alive; it owns
        the buffer behind addr (e.g. bytes/bytearray).
        """
        self._check_open()
        _check_tag(tag)
        if size < 0:
            raise ValueError(f"size {size} < 0")
        if not addr:
            raise ValueError(
                f"addr {addr!r} invalid (posting a null pointer would segfault)"
            )
        op = LwdUcxOp(tag, cb=cb, ref=ref)
        self._q.put(("send", tag, addr, size, op))
        return op

    def send_bytes(
        self, tag: int, data: bytes, *, cb: Callable[[LwdUcxOp], None] | None = None
    ) -> LwdUcxOp:
        """Convenience send: hand over bytes directly (the rcache
        registers them implicitly); ref keeps them alive.
        """
        self._check_open()
        _check_tag(tag)
        addr = ctypes.cast(ctypes.c_char_p(data), ctypes.c_void_p).value
        op = LwdUcxOp(tag, cb=cb, ref=data)
        self._q.put(("send", tag, addr, len(data), op))
        return op

    def acquire_staging(self) -> LwdUcxBlock:
        """Send-direction channel: take a pool block for D2H staging
        (pool_staging channels only).

        Blocks until a block is free: the pool depth caps in-flight sends
        and so carries backpressure. The send completion callback returns
        the block through release_staging and wakes this wait; that
        callback runs on the library IO thread while this call may run on
        any thread, typically the assembly layer send thread. Raises once
        the channel shuts down. The block .buf/.addr mean the same as on
        a receive block (the rcache registers it on the first send).
        """
        if not self._pool_staging:
            raise RuntimeError(f"acquire_staging on non-staging channel {self._name}")
        while True:
            blk = self._pool.get(timeout=0.05)
            if blk is not None:
                return blk
            if self._stopped:
                raise RuntimeError(f"staging wait aborted (shutdown) on {self._name}")

    def release_staging(self, blk: LwdUcxBlock) -> None:
        """Return a staging block (called from the send completion
        callback; pairs with acquire_staging)."""
        self._pool.release(blk)

    def recv(
        self, tag: int, *, cb: Callable[[LwdUcxOp], None] | None = None
    ) -> LwdUcxOp:
        """Pre-post one receive: on completion op.data holds standalone
        bytes.

        One tag may be pre-posted many times (queue style). A frame that
        arrives first goes to the early store (per-tag FIFO, so one tag
        sent repeatedly loses nothing; total count bounded by
        cfg.early_max -- on overflow the globally oldest entry is dropped
        and a warning is logged).
        """
        self._check_open()
        _check_tag(tag)
        op = LwdUcxOp(tag, cb=cb)
        self._q.put(("recv", tag, op))
        return op

    def recv_into(
        self,
        tag: int,
        addr: int,
        cap: int,
        *,
        cb: Callable[[LwdUcxOp], None] | None = None,
    ) -> LwdUcxOp:
        """Pre-post one receive that lands in the caller buffer: on
        completion the data is at addr and op.size is the byte count.
        Until then addr belongs to the library (neither read nor write
        it)."""
        self._check_open()
        _check_tag(tag)
        if cap <= 0:
            raise ValueError(f"cap {cap} <= 0")
        op = LwdUcxOp(tag, cb=cb, into_addr=addr, into_cap=cap)
        self._q.put(("recv", tag, op))
        return op

    def on_broken(self, cb: Callable[[BaseException], None]) -> None:
        """Broken-link callback (runs on the IO thread): fires after every
        in-flight op has already been terminated with ERROR."""
        self._broken_cb = cb

    def on_peer_restart(self, cb: Callable[[], None]) -> None:
        """Peer-restart callback (runs on the IO thread): fires when the
        handshake session changes.

        A restarted peer invalidates its tag sequence, and this side's
        pre-posts and early frames are already dropped; resetting the
        application's tag baselines is up to this callback.
        """
        self._restart_cb = cb

    def fail(self, reason: BaseException | str) -> None:
        """Unrecoverable error found by the application (a protocol
        violation and the like): tear the link down at once -- every
        in-flight op ends with ERROR, the connect side rebuilds
        automatically and the serve side waits for a reconnect.

        Idempotent. Prefer calling it from an IO-thread callback (it runs
        at the violation site, and the cleanup is designed for exclusive
        IO-thread use); a cross-thread call is best effort.
        """
        if self._stopped or self._link is None:
            return
        err = reason if isinstance(reason, BaseException) else RuntimeError(str(reason))
        self._fail_all_pending(err)

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Wait for the first link; False means timeout (still retrying)."""
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
        """Local per-message ceiling (= receive block size; told to the
        peer)."""
        return self._block_bytes

    @property
    def peer_max_msg(self) -> int:
        """Peer per-message ceiling (a send above it fails that op)."""
        return self._peer_max_msg

    @property
    def peer_session(self) -> int:
        return self._peer_session

    def _require_link(self) -> _ChanLink:
        """The established link; callbacks never run without one."""
        link = self._link
        if link is None:
            raise RuntimeError(f"{self._name}: UCX link is not established")
        return link

    def close(self, timeout: float = 2.0) -> None:
        """Shut the channel down: stop threads, tear down the link, end
        unfinished ops with ERROR (without firing on_broken -- an orderly
        shutdown is not a broken link)."""
        self._stopped = True
        self._ready.set()
        self._broken_event.set()  # wake the supervisor/accept threads to exit
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
        # drain links that were never adopted
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
        # fail unfinished tasks
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
        """Withdraw unfinished pre-posted receives before close: an ep
        FORCE close does not cancel recv, and destroying the worker leaves
        an "mpool not returned" warning (seen on real hardware with 4x8MiB
        blocks). Cancellation completes asynchronously, so progress until
        _recv_reqs is empty or the deadline passes."""
        link = self._link
        cancel = _ucp_request_cancel
        if link is None or not self._recv_reqs or cancel is None:
            return
        for req in list(self._recv_reqs):
            with contextlib.suppress(Exception):
                cancel(ctypes.c_void_p(link.worker), ctypes.c_void_p(req))
        deadline = time.monotonic() + 0.5
        while self._recv_reqs and time.monotonic() < deadline:
            with contextlib.suppress(Exception):
                link.progress()

    def __enter__(self) -> LwdUcxChannel:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._stopped:
            raise RuntimeError(f"ucx channel {self._name} closed")

    # ------------------------------------------------------------------ #
    # Link setup (connect supervisor / serve accept thread)              #
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
                break  # ready once, broken, no auto-reconnect: supervisor exits
            try:
                link = self._build_link(host, port)
            except OSError as exc:
                tries += 1
                hint = (
                    " [peer listener not ready]"
                    if exc.errno == _errno.ECONNREFUSED
                    else ""
                )
                if tries % 10 == 1:
                    logger.warning(
                        "[ucx-chan] %s connect %s:%d deferred (%r), "
                        "retrying (backoff %.1fs)%s",
                        self._name,
                        host,
                        port,
                        exc,
                        backoff,
                        hint,
                    )
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                continue
            except Exception as exc:  # noqa: BLE001
                tries += 1
                if tries % 10 == 1:
                    logger.warning(
                        "[ucx-chan] %s connect failed (%r), backoff %.1fs",
                        self._name,
                        exc,
                        backoff,
                    )
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
                srv_fd = self._srv_fd
                if srv_fd is None:
                    return
                fd, peer = _sock.c_accept(srv_fd)
            except OSError as exc:
                if self._stopped:
                    break
                logger.error(
                    "[ucx-chan] %s accept failed on port %d: %r", self._name, port, exc
                )
                time.sleep(0.2)
                continue
            try:
                self._linkq.put(self._build_from_fd(fd))
                logger.warning(
                    "[ucx-chan] %s accepted from %s port %d", self._name, peer, port
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[ucx-chan] %s handshake with %s failed: %r", self._name, peer, exc
                )

    def _build_link(self, host: str, port: int) -> _ChanLink:
        """Connect side: exchange addresses over TCP, create the ep, and
        return a link ready for the IO thread to adopt."""
        link = _ChanLink(self)
        try:
            mine = link.my_address()
            fd = _sock.c_open()
            try:
                if self._cfg_traffic_class is not None:
                    _sock.c_set_tos(fd, self._cfg_traffic_class << 2)
                _sock.c_connect_timeout(fd, host, port, self._hs_timeout_s)
                _sock.c_set_sock_timeouts(fd, self._hs_timeout_s)
                _hs_send(
                    fd,
                    mine,
                    self._block_bytes,
                    self._token,
                    link.caps_n_slots,
                    link.caps_blob,
                    self._put_thresh,
                )
                addr, session, n_slots, pthresh, caps, pmax = _hs_recv(
                    fd, self._block_bytes, self._token
                )
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
        """Serve side: finish the handshake on an accepted fd (also
        bounded).

        When a shared listener has already read the fixed header to route
        on the token, ``hdr`` is passed in and this method reads only the
        body; the dedicated serve-port path passes nothing and reads the
        header itself.
        """
        link = _ChanLink(self)
        try:
            try:
                _sock.c_set_sock_timeouts(fd, self._hs_timeout_s)
                if hdr is None:
                    hdr = _hs_recv_exact(fd, _HS_HEADER.size)
                (alen, pmax, session, _token, n_slots, caps_len, pthresh) = (
                    _hs_parse_header(hdr, self._block_bytes, self._token)
                )
                body = _hs_recv_exact(fd, alen + caps_len)
                addr = body[:alen]
                caps = body[alen:] if caps_len else b""
                _hs_send(
                    fd,
                    link.my_address(),
                    self._block_bytes,
                    self._token,
                    link.caps_n_slots,
                    link.caps_blob,
                    self._put_thresh,
                )
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
        """After the shared listener routed a connection here: finish the
        handshake and hand the link to the IO thread to adopt."""
        link = self._build_from_fd(fd, hdr)
        self._linkq.put(link)
        logger.warning(
            "[ucx-chan] %s accepted from %s port %d",
            self._name,
            peer,
            self._listener.port if self._listener is not None else -1,
        )

    def _adopt_link(self, link: _ChanLink) -> None:
        """Swap links inside the IO thread: tear down the old one, reset
        state, pre-post the pools, and become available again."""
        self._broken_event.clear()
        try:
            if self._expects or self._send_reqs:
                # Ops the old link left unfinished (or the peer left before it
                # restarted): fail them all -- a restarted peer invalidates the tag
                # sequence, so waiting on them is a dead end
                self._fail_all_pending(
                    RuntimeError("link replaced: in-flight ops dropped")
                )
            old = self._link
            self._link = None
            if old is not None:
                old.close()  # torn down here: the IO thread is the only progressor
            if not self._pool_staging:
                # staging channels do not reset: in-flight staging blocks come back
                # through the send completion callback, and a reset would push borrowed
                # blocks back into the free queue (a double lend means data corruption)
                self._pool.reset()
            # the early store dies with the old link: the new link brings a fresh
            # message stream, bytes stuck on the old one are stale (the peer never
            # replays) and would misalign the new tag sequence
            self._early.clear()
            self._early_count = 0
            self._link = link
            self._ep_err = None
            self._broken = False
            self._peer_max_msg = link.peer_max_msg
            self._reset_put_state()
            # put-mode assembly (per direction; both peers need put_thresh > 0):
            #   send side: the peer published receive slots -> large frames may be
            #              written straight through put
            #   recv side: the peer will put into us -> data pre-posts move to the
            #              small ctl pool, the large blocks become pure put landing
            #              zones, and doorbells get a pool of their own
            self._put_armed = self._put_thresh > 0 and link.peer_n_slots > 0
            self._recv_ctl = self._put_thresh > 0 and link.peer_put_thresh > 0
            self._data_mask = _CTL_BITS if (self._put_armed or self._recv_ctl) else 0
            self._ctl_pool = []
            self._db_pool = []
            self._ack_pool = []
            if self._recv_ctl:
                # inline small-message ceiling = peer put_thresh (below it messages
                # go eager)
                ctl_bytes = min(
                    self._block_bytes, max(link.peer_put_thresh, 64 << 10) + 64
                )
                self._ctl_pool = [
                    LwdUcxBlock(i, ctl_bytes) for i in range(self._n_blocks)
                ]
                self._db_pool = [LwdUcxBlock(i, 16) for i in range(self._n_blocks)]
                for blk in self._db_pool:
                    self._post_db(blk)
            if self._put_armed:
                self._ack_pool = [LwdUcxBlock(i, 16) for i in range(self._n_blocks)]
                for blk in self._ack_pool:
                    self._post_ack(blk)
                self._ack_bufs = [bytearray(16) for _ in range(self._n_blocks * 2)]
            if not self._pool_staging:
                # staging channels post no data wildcard (every pool block is held for
                # send staging; a one-way send-only channel sees no arrivals anyway, and
                # the put-mode ack pool is still posted)
                data_pool = self._ctl_pool if self._recv_ctl else self._pool.blocks
                for blk in data_pool:
                    self._post_data(blk)
            restart = (
                self._peer_session != 0 and self._peer_session != link.peer_session
            )
            self._peer_session = link.peer_session
            self._ready.set()
            self._broken_event.clear()
            self._gen += 1
            if self._gen == 1:
                _log_ucx_env_once()
            logger.warning(
                "[ucx-chan] %s up gen=%d peer_session=%016x%s max_msg=%dMiB put=%s%s",
                self._name,
                self._gen,
                link.peer_session,
                " PEER-RESTARTED" if restart else "",
                self._block_bytes >> 20,
                "on" if (self._put_armed or self._recv_ctl) else "off",
                "" if not restart else " (on_peer_restart fired)",
            )
            if restart and self._restart_cb is not None:
                with contextlib.suppress(Exception):
                    self._restart_cb()
            if self._gen > 1:
                logger.error(
                    "[ucx-chan] %s link REPLACED gen=%d peer_session=%016x"
                    "(peer reconnect or restart; a second process "
                    "connecting to this port looks the same -- the edge "
                    "and the cloud each run one wire process, and the "
                    "serve side adopts the last connection)",
                    self._name,
                    self._gen,
                    link.peer_session,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("[ucx-chan] %s adopt link failed: %r", self._name, exc)
            self._fail_all_pending(RuntimeError(f"adopt link failed: {exc}"))
        finally:
            link.adopted.set()

    # ------------------------------------------------------------------ #
    # IO main loop                                                       #
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        last_log = 0.0
        while True:
            # link swap first: a link produced by the supervisor/accept thread takes
            # over before anything else
            with contextlib.suppress(queue.Empty):
                link = self._linkq.get_nowait()
                if link is not None:
                    self._adopt_link(link)
            task = None
            # tasks stay off the queue until the first link arrives (FIFO, order
            # kept); only once ready does a break fail fast
            if self._ready.is_set() or self._broken or self._stopped:
                with contextlib.suppress(queue.Empty):
                    task = self._q.get(timeout=0.005)
            else:
                time.sleep(0.005)
            if task is None and self._stopped:
                break
            if task is not None:
                if not self._ready.is_set() or self._broken:
                    task[-1]._finish(
                        error=RuntimeError(
                            f"ucx channel {self._name} not available ("
                            + (
                                "first connect pending)"
                                if not self._ready.is_set()
                                else "reconnecting"
                                + (
                                    f", last ep status={self._ep_err})"
                                    if self._ep_err
                                    else ")"
                                )
                            )
                        )
                    )
                    continue
                try:
                    self._dispatch(task)
                except Exception as exc:  # noqa: BLE001
                    self._broken = True
                    self._broken_event.set()
                    task[-1]._finish(error=exc)
                    logger.error("[ucx-chan] %s broken: %r", self._name, exc)
            try:
                self._progress_link()
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[ucx-chan] %s progress error (continuing): %r", self._name, exc
                )
            now = time.monotonic()
            if now - last_log >= 10.0:
                if self._expects and self._expects_t0 and now - self._expects_t0 > 5.0:
                    # pre-posts dangling >5s: the frame never came (the peer never sent
                    # it, or the wire dropped it) or consumption stalled -- frames
                    # normally arrive in milliseconds, so a persistent gap is odd
                    logger.error(
                        "[ucx-chan] %s recv expects stuck %.1fs: tags=%d "
                        "min_tag=%d early=%d",
                        self._name,
                        now - self._expects_t0,
                        len(self._expects),
                        min(self._expects),
                        self._early_count,
                    )
                logger.warning(
                    "[ucx-chan][stall] %s sends=%d expects=%d(tags=%d) "
                    "early=%d broken=%s",
                    self._name,
                    len(self._send_reqs),
                    sum(len(q) for q in self._expects.values()),
                    len(self._expects),
                    self._early_count,
                    self._broken,
                )
                last_log = now

    def _progress_link(self) -> None:
        if self._link is None or not self._ready.is_set():
            return
        self._link.progress()
        if self._ep_err is not None and not self._broken:
            self._fail_all_pending(RuntimeError(f"ucx ep error status={self._ep_err}"))

    def _on_ep_error(self, status: int) -> None:
        # may fire inside progress (IO thread); the status is consumed by the
        # IO thread
        self._ep_err = status
        logger.error("[ucx-chan] %s ep error status=%d", self._name, status)

    def _dispatch(self, task: tuple) -> None:
        if task[0] == "send":
            _, tag, addr, size, op = task
            self._do_send(tag, addr, size, op)
        else:
            _, tag, op = task
            self._do_recv(tag, op)

    # ---------------- Send: raw address / put write ------------------- #

    def _do_send(self, tag: int, addr: int, size: int, op: LwdUcxOp) -> None:
        link = self._link
        if link is None:
            raise RuntimeError("no link")
        if self._peer_max_msg and size > self._peer_max_msg:
            # caller exceeded the peer block capacity: fail only this op, do not
            # break the link (unlike a link error)
            op._finish(
                error=RuntimeError(
                    f"size {size} > peer max_msg {self._peer_max_msg}:"
                    f" raise peer cfg.block_mb"
                )
            )
            return
        if self._put_armed and size >= self._put_thresh:
            self._put_send(link, tag, addr, size, op)
            return
        req = link.send_nb(addr, size, tag, self._send_cb)
        # restype=c_void_p: an immediate completion returns None, treated as 0
        if not req:
            op._finish(size=size)
            return
        if _ptr_is_err(req):
            op._finish(
                error=RuntimeError(f"ucp_tag_send_nbx failed status={_ptr_status(req)}")
            )
            self._fail_all_pending(RuntimeError("ucx send post failed"))
            return
        self._send_reqs[int(req)] = (op, tag, size)

    def _on_send_done(self, request, status, user_data=None) -> None:  # noqa: ANN001
        ent = self._send_reqs.pop(int(request), None)
        with contextlib.suppress(Exception):
            self._require_link().request_free(int(request))
        if ent is None:
            return
        op, _tag, size = ent
        if status == _UCS_OK:
            op._finish(size=size)
        else:
            op._finish(error=RuntimeError(f"ucx send status={status}"))
            self._fail_all_pending(RuntimeError(f"ucx send status={status}"))

    # ---------------- RMA put: one-sided write + doorbell/ACK --------- #

    def _put_send(
        self, link: _ChanLink, tag: int, addr: int, size: int, op: LwdUcxOp
    ) -> None:
        """Write a large message straight into a peer slot (tag % peer
        slot count) with a one-sided WRITE.

        Slot reuse is gated by the receiver's ACK, which rules out the
        race where a sender overwrites a slot the receiver has not
        consumed. Once put completes (RC CQE means the data landed) the
        caller buffer may be released; the doorbell carries (tag, size)
        and does not depend on the source buffer contents.
        """
        slot = tag % link.peer_n_slots
        while (
            slot in self._slot_last
            and self._slot_acked.get(slot, -1) < self._slot_last[slot]
        ):
            if self._broken or self._stopped or self._link is not link:
                raise RuntimeError(f"ucx put slot {slot} wait aborted (link changed)")
            if not link.progress():
                time.sleep(0.0002)
        self._slot_last[slot] = tag
        req = link.put_nb(
            addr,
            size,
            link.peer_slot_addrs[slot],
            link.peer_slot_rkeys[slot],
            self._put_cb,
        )
        if not req:
            # completed immediately (rare): doorbell and finish right away
            self._send_doorbell(tag, size)
            op._finish(size=size)
            return
        if _ptr_is_err(req):
            self._slot_last.pop(slot, None)
            op._finish(
                error=RuntimeError(f"ucp_put_nbx failed status={_ptr_status(req)}")
            )
            self._fail_all_pending(RuntimeError("ucx put post failed"))
            return
        self._send_reqs[int(req)] = (op, tag, size)

    def _on_put_done(self, request, status, user_data=None) -> None:  # noqa: ANN001
        ent = self._send_reqs.pop(int(request), None)
        with contextlib.suppress(Exception):
            self._require_link().request_free(int(request))
        if ent is None:
            return
        op, tag, size = ent
        if status == _UCS_OK:
            # CQE means the data landed on the peer: the doorbell is safe now (no
            # further dependence on the source buffer)
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
        """Small control message (doorbell/ACK): a single eager message.
        The request and its buffer are parked in _db_pend together so the
        buffer is neither overwritten nor collected, and dropped on
        completion."""
        addr = ctypes.addressof((ctypes.c_char * len(buf)).from_buffer(buf))
        req = self._require_link().send_nb(addr, len(buf), tag, self._db_cb)
        if req:
            if _ptr_is_err(req):
                logger.error(
                    "[ucx-chan] %s ctl send failed status=%d",
                    self._name,
                    _ptr_status(req),
                )
                self._broken = True  # a lost doorbell hangs the peer, so rebuild
            else:
                self._db_pend[int(req)] = buf

    def _on_db_done(self, request, status, user_data=None) -> None:  # noqa: ANN001
        with contextlib.suppress(Exception):
            self._require_link().request_free(int(request))
        self._db_pend.pop(int(request), None)
        if status != _UCS_OK:
            self._broken = True
            logger.error("[ucx-chan] %s ctl msg status=%d", self._name, status)

    def _reset_put_state(self) -> None:
        """Reset on link swap/break: slot gating tables and pending
        control messages are void (their requests die with the old
        link)."""
        self._slot_last.clear()
        self._slot_acked.clear()
        self._db_pend.clear()

    # ---------------- Receive: masked pre-post + tag pairing ---------- #

    def _post_data(self, blk: LwdUcxBlock) -> None:
        self._post_masked(blk, 0, self._data_mask, "data")

    def _post_db(self, blk: LwdUcxBlock) -> None:
        self._post_masked(blk, _TAG_DOORBELL, _TAG_DOORBELL, "db")

    def _post_ack(self, blk: LwdUcxBlock) -> None:
        self._post_masked(blk, _TAG_ACK, _TAG_ACK, "ack")

    def _post_masked(self, blk: LwdUcxBlock, tag: int, mask: int, kind: str) -> None:
        req = self._require_link().recv_masked_nb(
            blk.addr, len(blk.buf), tag, mask, self._recv_cb
        )
        if not req or _ptr_is_err(req):
            raise RuntimeError(f"ucp_tag_recv_nbx failed req={req!r}")
        blk.recv_req = req
        self._recv_reqs[int(req)] = (blk, kind)

    def _on_recv_done(
        self,
        request,
        status,
        info,
        user_data=None,  # noqa: ANN001
    ) -> None:
        ent = self._recv_reqs.pop(int(request), None)
        with contextlib.suppress(Exception):
            self._require_link().request_free(int(request))
        if ent is None:
            return
        blk, kind = ent
        if status != _UCS_OK:
            if self._stopped:
                return  # cancel completion from close: freed/pop above, stay quiet
            self._broken = True
            self._fail_all_pending(
                RuntimeError(f"ucx recv status={status} on {self._name}")
            )
            return
        length = int(info.contents.length)
        sender_tag = int(info.contents.sender_tag)
        if kind == "ack":
            self._post_ack(blk)
            tag = sender_tag & _TAG_USER_MAX
            slot = tag % self._require_link().peer_n_slots
            if tag > self._slot_acked.get(slot, -1):
                self._slot_acked[slot] = tag
            return
        if kind == "db":
            self._post_db(blk)
            tag, size = _DB.unpack_from(blk.buf, 0)
            self._on_doorbell(tag, size)
            return
        # data (inline): the bytes sit in this receive block; copy out, then
        # re-post
        data = bytes(memoryview(blk.buf)[:length])
        self._post_data(blk)
        self._deliver(sender_tag & _TAG_USER_MAX, data)

    def _on_doorbell(self, tag: int, size: int) -> None:
        """Doorbell: the data is already in this side's slot
        blocks[tag % n]. Copying it out completes consumption, then ACK
        -- skipping the ACK would wedge the sender's slot gating for
        good."""
        try:
            if size > self._block_bytes:
                raise ConnectionError(
                    f"doorbell size {size} > block {self._block_bytes} on {self._name}"
                )
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

    def _do_recv(self, tag: int, op: LwdUcxOp) -> None:
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
        # arrived early: no pre-post to pair with, so park it per tag in FIFO
        # order (symmetric with the pre-post queue; one tag sent repeatedly loses
        # nothing); bounded by early_max, and on overflow the globally oldest
        # entry is dropped with a warning
        if self._early_count >= self._early_max:
            old_tag = next(t for t, dq in self._early.items() if dq)
            logger.warning(
                "[ucx-chan] %s early store full (%d): drop tag=%d"
                "-- raise cfg.early_max or pre-post sooner",
                self._name,
                self._early_max,
                old_tag,
            )
            dq = self._early[old_tag]
            dq.popleft()
            self._early_count -= 1
            if not dq:
                del self._early[old_tag]
        self._early.setdefault(tag, deque()).append(data)
        self._early_count += 1

    def _settle(self, op: LwdUcxOp, data: bytes) -> None:
        if op._into_addr:
            if len(data) > op._into_cap:
                op._finish(
                    error=RuntimeError(f"recv_into cap {op._into_cap} < {len(data)}B")
                )
                return
            ctypes.memmove(op._into_addr, data, len(data))
            op._finish(size=len(data))
            return
        op._finish(data=data)

    # ---------------- Broken-link cleanup ----------------------------- #

    def _fail_all_pending(self, error: BaseException) -> None:
        """Broken-link cleanup (single idempotent entry point): fail every
        in-flight send and every op waiting for a pair (expects), clear
        the request routing tables, set the broken event and fire
        on_broken."""
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
        logger.error(
            "[ucx-chan] %s broken (%s): %r",
            self._name,
            "auto-reconnecting"
            if self._role == "connect"
            else "waiting for the peer to reconnect",
            error,
        )
        cb = self._broken_cb
        if cb is not None:
            try:
                cb(error)
            except Exception:  # noqa: BLE001
                logger.exception("[ucx-chan] %s on_broken cb error", self._name)


def _check_tag(tag: int) -> None:
    if not 0 <= tag <= _TAG_USER_MAX:
        raise ValueError(
            f"tag {tag} out of range [0, 2^61): the top 3 bits are "
            f"reserved (put doorbell/ACK)"
        )


class LwdUcxListener:
    """One TCP listener that hands connections to several
    LwdUcxChannels by UHC1 handshake token.

    Only the accept/handshake entry is shared; every channel keeps its
    own UCP ep, IO thread and block pool. 1e1c currently uses two tokens
    (UP/DOWN); with more edges the token embeds the edge_id.
    """

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self.port = int(port)
        self._fd: int | None = _sock.c_open()
        self._stopped = False
        self._thread: threading.Thread | None = None
        self._routes: dict[int, LwdUcxChannel] = {}
        self._lock = threading.Lock()
        self._traffic_class: int | None = None
        _sock.c_setsockopt_int(
            self._fd,
            _sock._SOL_SOCKET,
            _sock._SO_REUSEADDR,
            1,
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
        fd = self._fd
        if fd is None:
            return
        _sock.c_set_tos(fd, int(traffic_class) << 2)

    def register(self, token: int, channel: LwdUcxChannel, *, name: str = "") -> None:
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
            name=f"ucx-chan-acc-{self.port}",
            daemon=True,
        )
        self._thread.start()
        logger.warning(
            "[ucx-chan] unified listener %s:%d tokens=%d %s",
            self._host,
            self.port,
            len(self._routes),
            [hex(t) for t in sorted(self._routes)],
        )

    def _accept_loop(self) -> None:
        srv_fd = self._fd
        if srv_fd is None:
            return
        while not self._stopped:
            try:
                fd, peer = _sock.c_accept(srv_fd)
            except OSError as exc:
                if self._stopped:
                    break
                logger.error(
                    "[ucx-chan] unified accept failed on %s:%d: %r",
                    self._host,
                    self.port,
                    exc,
                )
                time.sleep(0.2)
                continue
            threading.Thread(
                target=self._handshake,
                args=(fd, peer),
                name=f"ucx-chan-hs-{self.port}",
                daemon=True,
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
                    token,
                    peer,
                    self._host,
                    self.port,
                )
                return
            # the channel's _build_from_fd reads the body, replies with its handshake
            # and closes the fd; from here on the fd belongs to the channel, which
            # rules out a double close
            owned = False
            ch._accept_fd(fd, hdr, peer)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[ucx-chan] handshake with %s failed on %s:%d: %r",
                peer,
                self._host,
                self.port,
                exc,
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
