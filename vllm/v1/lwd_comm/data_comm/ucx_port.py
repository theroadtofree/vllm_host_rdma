# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UCX transport port abstraction: one entry point for the two modes,
embedded (in-process, direct LwdUcxChannel) and daemon (standalone
transport daemon ucx_daemon + shm ring).

The engine-side assembly layer (e.g. ucx_npu_channel in vllm-ascend)
programs only against the ``LwdUcxTransportPort`` protocol of this module
and no longer touches the ``LwdUcxChannel`` construction class methods.
embedded returns LwdUcxChannel itself (current path, zero behaviour
change); daemon returns ``LwdUcxDaemonPort`` -- data plane over the shm
ring (ucx_shm), control plane over the ctl unix socket, wake-up over
single-byte writes on two pipes. Both modes are UHC1 on the wire, so the
peer cannot tell, but both ends of one link must be deployed as a matched
pair in the same mode (a mixed start raises immediately, no silent
fallback -- same rollout policy as family).

The protocol methods are the subset of the LwdUcxChannel public call
surface used by the assembly layer:
  * acquire_staging / release_staging -- send-side D2H staging block
    reuse (pool depth = in-flight cap, carries backpressure);
  * send / recv / fail -- data and protocol error paths;
  * wait_ready / close -- lifecycle;
  * recv_discard (optional in daemon, absent in embedded) -- discard
    pre-post for skip.
"""

from __future__ import annotations

import contextlib
import json
import os
import select
import socket as _pysocket
import subprocess
import sys
import threading
import time
from typing import Any, Protocol

from vllm.logger import init_logger
from vllm.v1.lwd_comm.data_comm import ucx_shm as _shm
from vllm.v1.lwd_comm.data_comm.ucx_channel import LwdUcxChannel, LwdUcxConfig

logger = init_logger(__name__)

_SPAWN_CONNECT_TIMEOUT_S = 10.0


class LwdUcxTransportPort(Protocol):
    """Transport port of one channel (engine-side view; method contracts
    in LwdUcxChannel)."""

    def acquire_staging(self) -> Any:
        """Take a staging block from the pool for D2H (blocks until one is
        free; raises on shutdown)."""
        ...

    def release_staging(self, blk: Any) -> None:
        """Return a staging block (error path; the normal path returns it
        on send completion)."""
        ...

    def send(
        self, tag: int, addr: int, size: int, *, ref: Any = None, cb: Any = None
    ) -> Any:
        """Send size bytes starting at addr; returns a completion handle."""
        ...

    def recv(self, tag: int, *, cb: Any = None) -> Any:
        """Pre-post one receive (the completion callback gets op.data; in
        daemon mode data is the shm view of the recv block, so the caller
        must call op.mark_recycle(event) after H2D)."""
        ...

    def fail(self, reason: BaseException | str) -> None:
        """Unrecoverable error detected on the application side: every
        in-flight op ends in ERROR."""
        ...

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Wait for the first link to be ready; False = timeout."""
        ...

    def close(self, timeout: float = 2.0) -> None:
        """Orderly shutdown (does not fire on_broken)."""
        ...


def open_transport(
    cfg: LwdUcxConfig,
    *,
    mode: str = "embedded",
    serve: bool = False,
    host: str = "",
    port: int = 0,
    listener: Any = None,
    ctl_sock: str = "",
) -> LwdUcxTransportPort:
    """Open one transport port for the given mode.

    embedded: in-process, direct LwdUcxChannel (current path, zero change).
    daemon:   standalone daemon (ucx_daemon) + shm ring (LwdUcxDaemonPort).

    serve/listener semantics match LwdUcxChannel.serve/serve_via: the edge
    side connects (host, port); the cloud side serves with serve=True
    (host, port) or with a shared listener (routed by handshake token). In
    daemon mode listener must be None -- the listener itself moves into the
    daemon (shared per (host,port), token routing); ctl_sock is the daemon
    ctl unix socket path (empty = default /tmp/lwd-host-comm-{uid}.sock,
    and the daemon is spawned on demand by this process).
    """
    if mode == "embedded":
        if serve:
            if listener is not None:
                return LwdUcxChannel.serve_via(listener, cfg=cfg)
            return LwdUcxChannel.serve(host, port, cfg=cfg)
        if not (host and port):
            raise RuntimeError(
                "connect mode requires host and port (serve mode requires serve=True)"
            )
        return LwdUcxChannel.connect(host, port, cfg=cfg)
    if mode == "daemon":
        if listener is not None:
            raise RuntimeError(
                "daemon mode does not accept an engine-side listener "
                "(the listener belongs to the daemon process; pass "
                "listener=None for daemon mode)"
            )
        return LwdUcxDaemonPort(
            cfg, serve=serve, host=host, port=port, ctl_sock=ctl_sock
        )
    raise RuntimeError(
        f"unknown ucx transport mode {mode!r} (expect 'embedded' or 'daemon')"
    )


# --------------------------------------------------------------------- #
# daemon mode: engine-side port (shm ring + ctl)                        #
# --------------------------------------------------------------------- #
def _default_ctl_sock() -> str:
    # External name lwd_host_comm (matches ucx_daemon.PROC_NAME); code
    # identifiers stay ucx/daemon (implementation name).
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return f"/tmp/lwd-host-comm-{uid}.sock"


def _daemon_log_path(ctl_sock: str) -> str:
    return ctl_sock + ".log"


def _kill_stale_daemon(ctl_sock: str) -> None:
    """Kill an incompatible stale daemon via its pidfile (best effort)."""
    import signal

    pid_path = ctl_sock + ".pid"
    try:
        with open(pid_path) as f:
            pid = int(f.read().strip())
        os.kill(pid, signal.SIGKILL)
        logger.warning(
            "[ucx-port] killed stale lwd_host_comm pid=%d (ctl proto mismatch)", pid
        )
    except (OSError, ValueError):
        pass  # no pidfile (older build) or gone: user pkill is the fallback
    with contextlib.suppress(OSError):
        os.unlink(pid_path)
    with contextlib.suppress(OSError):
        os.unlink(ctl_sock)


def _ctl_connect_or_spawn(ctl_sock: str) -> _pysocket.socket:
    """Connect to an existing daemon; if there is none, spawn one
    (start_new_session detaches it from the engine process group) and
    retry the connect. Raises on failure -- no silent fallback to
    embedded (both ends must use the same mode)."""

    def _try() -> _pysocket.socket | None:
        s = _pysocket.socket(_pysocket.AF_UNIX, _pysocket.SOCK_STREAM)
        try:
            s.connect(ctl_sock)
            # Connected != usable: check the ctl protocol version (an old
            # daemon is incompatible and may swallow the ATTACH exception
            # and just drop the connection -- observed on real hardware)
            s.sendall(b'{"cmd": "PING"}\n')
            data = s.recv(65536)
            reply = json.loads(data.split(b"\n", 1)[0]) if data else {}
            if not reply.get("ok") or int(reply.get("proto", 0)) < 2:
                s.close()
                _kill_stale_daemon(ctl_sock)
                return None
            return s
        except OSError:
            s.close()
            return None

    conn = _try()
    if conn is not None:
        return conn
    lock_path = ctl_sock + ".spawn.lock"
    os.makedirs(os.path.dirname(ctl_sock) or ".", exist_ok=True)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        import fcntl

        fcntl.flock(lock_fd, fcntl.LOCK_EX)  # same-container first connects serialize
        conn = _try()
        if conn is None:
            # daemon stdout/stderr go to a file (the vllm logger writes to
            # stderr) -- inspect it when spawn/ATTACH fails
            log_path = _daemon_log_path(ctl_sock)
            with open(log_path, "ab", buffering=0) as log_f:
                subprocess.Popen(  # noqa: S603
                    [
                        sys.executable,
                        "-m",
                        "vllm.v1.lwd_comm.data_comm.ucx_daemon",
                        "--ctl",
                        ctl_sock,
                    ],
                    env=os.environ.copy(),
                    start_new_session=True,
                    stdin=subprocess.DEVNULL,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                )
            deadline = time.monotonic() + _SPAWN_CONNECT_TIMEOUT_S
            while time.monotonic() < deadline:
                conn = _try()
                if conn is not None:
                    break
                time.sleep(0.1)
    finally:
        os.close(lock_fd)
    if conn is None:
        raise RuntimeError(
            f"ucx daemon spawn/attach failed on {ctl_sock} (mode=daemon "
            "no fallback to embedded; see the daemon log "
            f"{_daemon_log_path(ctl_sock)}"
        )
    return conn


class _PortOp:
    """LwdUcxOp-compatible completion handle (daemon mode; finishes as
    soon as the cpl arrives).

    On the recv path ``data`` is the shm view of the recv block: after
    H2D the caller registers a device event via ``mark_recycle(event)``,
    the reaper polls event.query() and then returns the block
    (RECV_RELEASE). An op with no registered event is returned by the 1s
    fallback after its callback returns.
    """

    __slots__ = (
        "tag",
        "size",
        "data",
        "error",
        "_cb",
        "block_idx",
        "_stg_idx",
        "_recycle_evt",
        "_recycle_t0",
        "_fin",
    )

    def __init__(self, tag: int, cb: Any = None) -> None:
        self.tag = tag
        self.size = 0
        self.data: Any = None
        self.error: BaseException | None = None
        self._cb = cb
        self.block_idx = -1  # recv block index (set on cpl; release basis)
        self._stg_idx = -1  # send staging block index (release on done)
        self._recycle_evt: Any = None
        self._recycle_t0 = 0.0
        self._fin = False

    def _finish(
        self,
        *,
        size: int = 0,
        err: int = 0,
        data: Any = None,
        error: BaseException | None = None,
    ) -> None:
        if self._fin:
            return
        self._fin = True
        self.size = size
        self.error = error or (
            RuntimeError("daemon transport op failed") if err else None
        )
        self.data = data
        cb = self._cb
        if cb is not None:
            try:
                cb(self)
            except Exception:  # noqa: BLE001
                logger.exception("[ucx-port] op cb error tag=%s", self.tag)

    def done(self) -> bool:
        return self._fin

    def mark_recycle(self, event: Any) -> None:
        """Register the recv-side H2D completion event (the assembly layer
        calls this after future._bind)."""
        self._recycle_evt = event
        self._recycle_t0 = time.monotonic()


class _ShmStaging:
    """Staging block returned by acquire_staging: same shape as
    ucx_pool.LwdUcxBlock (buf/addr/index), so the assembly layer writes
    D2H directly via torch.frombuffer(stg.buf)."""

    __slots__ = (
        "buf",
        "addr",
        "index",
        "future",
        "event",
        "recv_req",
        "send_req",
        "t_post",
        "t_enq",
        "t_disp",
        "nbytes",
        "_port",
        "_released",
    )

    def __init__(
        self, seg: _shm.LwdUcxShmSegment, index: int, port: LwdUcxDaemonPort
    ) -> None:
        self.index = index
        self.buf = seg.staging_view(index)
        self.addr = seg.staging_addr(index)
        self.future = None
        self.event = None
        self.recv_req = 0
        self.send_req = 0
        self.t_post = 0.0
        self.t_enq = 0.0
        self.t_disp = 0.0
        self.nbytes = 0
        self._port = port
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._port.release_staging(self)


class LwdUcxDaemonPort:
    """daemon-mode engine-side port, satisfying LwdUcxTransportPort.

    The ctl reply carries three fds (SCM_RIGHTS): [shm, wake_daemon_w,
    wake_engine_r] -- the engine writes wake_daemon_w to wake the daemon
    cmd pump; the dispatcher blocks on wake_engine_r waiting for a daemon
    wake-up (EOF = session teardown).

    Data path: send = D2H direct write into a staging block -> cmd SEND
    (no extra copy, the daemon sends from the same physical page);
    recv = cpl RECV_DONE -> caller does H2D from the recv block ->
    op.mark_recycle(npu_event) -> reaper returns the block. Backpressure
    matches embedded: staging pool depth = in-flight send cap (local free
    list, replenished by SEND_DONE); recv block depth = in-flight recv cap
    (daemon-side free list).
    """

    def __init__(
        self, cfg: LwdUcxConfig, *, serve: bool, host: str, port: int, ctl_sock: str
    ) -> None:
        self._stopped = False
        self._ready = threading.Event()
        self._staging_cond = threading.Condition()
        self._free_staging: list[int] = []
        self._ops: dict[int, _PortOp] = {}  # op_id -> op
        self._pending_recycle: list[_PortOp] = []
        self._rel_retry: list[int] = []  # recv blocks to release again
        self._next_op_id = 1
        self._key = 0  # filled from the attach reply; used by DETACH
        # cmd ring producer mutex: the engine-side sender/dispatcher/compute
        # threads all send commands, and the ring's SPSC contract only
        # covers cross-process -- multiple producers in one process must be
        # serialized locally, or concurrent puts overwrite commands (send
        # completions lost / futures hang; a 1000-request concurrency stress
        # test reproduces this every time)
        self._cmd_lock = threading.Lock()
        self._wake_daemon_w = -1
        self._wake_engine_r = -1
        self._seg: _shm.LwdUcxShmSegment | None = None
        self._ctl = _ctl_connect_or_spawn(ctl_sock or _default_ctl_sock())
        req = {
            "cmd": "ATTACH",
            "role": "serve" if serve else "connect",
            "host": host,
            "port": port,
            "token": cfg.token,
            "block_mb": cfg.block_mb,
            "n_blocks": cfg.n_blocks,
            "put_thresh": cfg.put_thresh,
            "hs_timeout_s": cfg.hs_timeout_s,
            "early_max": cfg.early_max,
            "pool_staging": cfg.pool_staging,
        }
        reply, fds = self._ctl_roundtrip(req)
        if not reply.get("ok") or len(fds) != 3:
            self._ctl.close()
            raise RuntimeError(f"daemon attach failed: {reply} fds={len(fds)}")
        self._seg = _shm.LwdUcxShmSegment(
            block_bytes=reply["block_bytes"],
            n_staging=reply["n_staging"],
            n_recv=reply["n_recv"],
            fd=fds[0],
        )
        self._wake_daemon_w = fds[1]
        self._wake_engine_r = fds[2]
        self._key = int(reply["key"])  # needed by DETACH (a DETACH without
        # the key matches no session on the
        # daemon side, so the daemon never
        # exits when idle -- observed on real
        # hardware)
        os.set_blocking(self._wake_daemon_w, False)
        with self._staging_cond:
            self._free_staging = list(range(reply["n_staging"]))
        self._disp = threading.Thread(
            target=self._dispatch_loop, name="ucx-port-disp", daemon=True
        )
        self._disp.start()
        logger.warning(
            "[ucx-port] daemon attach ok key=%s staging=%d recv=%d block=%dMiB",
            reply.get("key"),
            reply["n_staging"],
            reply["n_recv"],
            reply["block_bytes"] >> 20,
        )

    # ------------------------------------------------------------------ #
    # ctl                                                                #
    # ------------------------------------------------------------------ #
    def _require_seg(self) -> _shm.LwdUcxShmSegment:
        """The shm segment attached at ATTACH; the request paths need it."""
        seg = self._seg
        if seg is None:
            raise RuntimeError("daemon shm segment is not attached (ATTACH failed?)")
        return seg

    def _ctl_roundtrip(self, req: dict) -> tuple[dict, list]:
        self._ctl.sendall(json.dumps(req).encode() + b"\n")
        # ATTACH is the only request with a reply; the reply's ancillary
        # data carries 3 fds. recvmsg's ancbufsize defaults to 0, which
        # silently drops SCM_RIGHTS -- the ancillary buffer must be passed
        # explicitly.
        data, ancdata, _flags, _addr = self._ctl.recvmsg(
            65536, _pysocket.CMSG_SPACE(4 * 8)
        )
        if not data:
            raise ConnectionError(
                "ucx daemon ctl closed without reply (see the "
                "daemon log "
                f"{_daemon_log_path(_default_ctl_sock())})"
            )
        fds = []
        for lvl, typ, cd in ancdata:
            if typ == _pysocket.SCM_RIGHTS and cd:
                import array

                ints = array.array("i")
                ints.frombytes(cd[: len(cd) // 4 * 4])
                fds.extend(ints.tolist())
        return json.loads(data.split(b"\n", 1)[0]), fds

    # ------------------------------------------------------------------ #
    # protocol methods (LwdUcxTransportPort)                             #
    # ------------------------------------------------------------------ #
    def acquire_staging(self) -> Any:
        while True:
            with self._staging_cond:
                if self._free_staging:
                    idx = self._free_staging.pop()
                    return _ShmStaging(self._require_seg(), idx, self)
                if self._stopped:
                    raise RuntimeError("staging wait aborted (shutdown)")
                self._staging_cond.wait(timeout=0.05)

    def release_staging(self, blk: Any) -> None:
        """Engine-side release entry (error path; normally done by
        SEND_DONE)."""
        with self._staging_cond:
            self._free_staging.append(blk.index)
            self._staging_cond.notify_all()

    def send(
        self, tag: int, addr: int, size: int, *, ref: Any = None, cb: Any = None
    ) -> _PortOp:
        if self._stopped:
            raise RuntimeError("daemon port closed")
        idx = self._staging_index_of(addr)
        op = _PortOp(tag, cb=cb)
        op._stg_idx = idx
        op_id = self._next_op_id
        self._next_op_id += 1
        self._ops[op_id] = op
        self._cmd(_shm.CMD_SEND, tag=tag, block_idx=idx, size=size, op_id=op_id)
        return op

    def recv(self, tag: int, *, cb: Any = None) -> _PortOp:
        if self._stopped:
            raise RuntimeError("daemon port closed")
        op = _PortOp(tag, cb=cb)
        op_id = self._next_op_id
        self._next_op_id += 1
        self._ops[op_id] = op
        # A full cmd ring makes a pre-post merely deferrable (retry once the
        # daemon drains) -- bounded backoff instead of pushing the transient
        # ring-full up to the caller's ordering layer
        deadline = time.monotonic() + 30.0
        while True:
            try:
                self._cmd(_shm.CMD_RECV, tag=tag, op_id=op_id)
                return op
            except RuntimeError:
                if self._stopped or time.monotonic() >= deadline:
                    self._ops.pop(op_id, None)
                    raise
                time.sleep(0.01)

    def recv_discard(self, tag: int) -> None:
        """Pre-post a discard receive (skip path; takes no recv block and
        yields no cpl)."""
        if not self._stopped:
            self._cmd(_shm.CMD_RECV_DISCARD, tag=tag)

    def fail(self, reason: BaseException | str) -> None:
        logger.error("[ucx-port] fail: %s", reason)
        if not self._stopped:
            self._cmd(_shm.CMD_FAIL)

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Wait for the first link to be ready; False = timeout or session
        already terminated (dispatcher EOF means the daemon-side pump died
        or was torn down -- fail fast rather than wait out the timeout)."""
        if timeout is None:
            timeout = 3600.0
        deadline = time.monotonic() + timeout
        while not self._ready.wait(min(0.5, max(0.01, deadline - time.monotonic()))):
            if self._stopped or time.monotonic() >= deadline:
                return False
        return True

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def close(self, timeout: float = 2.0) -> None:
        if self._stopped:
            return
        self._stopped = True
        with contextlib.suppress(Exception):
            self._cmd(_shm.CMD_CLOSE)
        if self._disp.is_alive():
            self._disp.join(timeout=timeout)
        for fd in (self._wake_daemon_w, self._wake_engine_r):
            with contextlib.suppress(OSError):
                os.close(fd)
        with contextlib.suppress(Exception):
            self._ctl.sendall(
                json.dumps({"cmd": "DETACH", "key": self._key}).encode() + b"\n"
            )
            self._ctl.close()
        with contextlib.suppress(Exception):
            self._require_seg().close()
        with self._staging_cond:
            self._staging_cond.notify_all()

    # ------------------------------------------------------------------ #
    # internals                                                          #
    # ------------------------------------------------------------------ #
    def _staging_index_of(self, addr: int) -> int:
        base = self._require_seg().staging_addr(0)
        off = addr - base
        idx, rem = divmod(off, self._require_seg().block_bytes)
        if idx < 0 or idx >= self._require_seg().n_staging or rem != 0:
            raise RuntimeError(
                f"send addr {addr:#x} not staging-block aligned "
                f"(base={base:#x} blk={self._require_seg().block_bytes})"
            )
        return idx

    def _wake_daemon(self) -> None:
        with contextlib.suppress(BlockingIOError, OSError):
            os.write(self._wake_daemon_w, b"\x01")

    def _cmd(self, kind: int, **kw: Any) -> None:
        """Single entry point for cmd ring producers (put + wake under the
        lock, so concurrent threads cannot overwrite)."""
        with self._cmd_lock:
            self._require_seg().cmd_put(kind, **kw)
            self._wake_daemon()

    def _dispatch_loop(self) -> None:
        """cpl dispatch + recv block recycling (single dispatcher thread,
        cpl FIFO order preserved)."""
        while not self._stopped:
            ready, _, _ = select.select([self._wake_engine_r], [], [], 0.05)
            if ready:
                try:
                    if not os.read(self._wake_engine_r, 64):
                        self._stopped = True  # daemon-side session teardown (EOF)
                        return
                except OSError:
                    self._stopped = True
                    return
            while True:
                cpl = self._require_seg().cpl_get()
                if cpl is None:
                    break
                try:
                    self._on_cpl(*cpl)
                except Exception:  # noqa: BLE001
                    # A failing cpl must not kill the dispatcher (the ring
                    # would stall and the whole channel would appear dead);
                    # log and continue
                    logger.exception("[ucx-port] cpl handle error")
                if self._stopped:
                    return
            try:
                self._reap_recycle()
            except Exception:  # noqa: BLE001
                logger.exception("[ucx-port] recycle error")

    def _on_cpl(
        self, kind: int, _r: int, op_id: int, size: int, err: int, block_idx: int
    ) -> None:
        if kind == _shm.CPL_SEND_DONE:
            op = self._ops.pop(op_id, None)
            if op is None:
                return
            if op._stg_idx >= 0:
                with self._staging_cond:
                    self._free_staging.append(op._stg_idx)
                    self._staging_cond.notify_all()
            op._finish(size=size, err=err)
        elif kind == _shm.CPL_RECV_DONE:
            op = self._ops.pop(op_id, None)
            if op is None:  # abandoned (timeout/teardown): return the block
                self._release_recv(block_idx)
                return
            op.block_idx = block_idx
            if err:
                op._finish(err=err)
                self._release_recv(block_idx)
                return
            op._finish(size=size, data=self._require_seg().recv_view(block_idx, size))
            op._recycle_t0 = time.monotonic()
            self._pending_recycle.append(op)
        elif kind == _shm.CPL_READY:
            self._ready.set()
        elif kind == _shm.CPL_BROKEN:
            self._ready.clear()
        elif kind == _shm.CPL_CLOSED:
            self._stopped = True

    def _reap_recycle(self) -> None:
        # Drain the release retry queue first (releases that hit a transient
        # failure such as a full ring -- one missed entry leaks a recv block
        # forever and the receive direction stalls once the daemon pool is
        # exhausted; observed on real hardware)
        if self._rel_retry:
            still_retry: list[int] = []
            for blk in self._rel_retry:
                if not self._try_release(blk):
                    still_retry.append(blk)
            self._rel_retry[:] = still_retry
        if not self._pending_recycle:
            return
        still: list[_PortOp] = []
        for op in self._pending_recycle:
            evt = op._recycle_evt
            done = False
            if evt is not None:
                try:
                    done = bool(evt.query())
                except Exception:  # noqa: BLE001
                    done = True
            elif time.monotonic() - op._recycle_t0 > 1.0:
                done = True  # no device event registered: 1s fallback
            if done:
                if not self._try_release(op.block_idx):
                    # transient (full ring): stay pending, retry next round
                    still.append(op)
                    logger.error(
                        "[ucx-port] recv release deferred "
                        "(cmd ring full?) blk=%d pending=%d",
                        op.block_idx,
                        len(self._pending_recycle),
                    )
            else:
                still.append(op)
        self._pending_recycle[:] = still

    def _try_release(self, block_idx: int) -> bool:
        """Try to return a recv block; False on failure (full ring etc.),
        retried by the caller."""
        if block_idx < 0 or self._seg is None or self._stopped:
            return True  # nothing to return (treated as done)
        try:
            self._cmd(_shm.CMD_RECV_RELEASE, block_idx=block_idx)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _release_recv(self, block_idx: int) -> None:
        """Return a recv block (never lost: a failure goes to _rel_retry and
        the dispatcher retries it)."""
        if not self._try_release(block_idx):
            self._rel_retry.append(block_idx)
            logger.error(
                "[ucx-port] recv release queued for retry blk=%d backlog=%d",
                block_idx,
                len(self._rel_retry),
            )
