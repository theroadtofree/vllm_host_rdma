# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UCX transport daemon (the daemon-mode body; the engine-side counterpart
is ucx_port.LwdUcxDaemonPort).

Deployment: same container as the engine, spawned on demand by the first
daemon-mode engine process
(``python -m vllm.v1.lwd_comm.data_comm.ucx_daemon --ctl <sock>``). The UCX
env (LD_LIBRARY_PATH/UCX_TLS/UCX_NET_DEVICES) is inherited as-is from the
engine process.

Owns every LwdUcxChannel/LwdUcxListener. The wire protocol (UHC1),
handshake, token routing, reconnect and put path all reuse ucx_channel
with zero changes. The engine ATTACHes/DETACHes over a ctl unix socket;
the data plane runs on shm rings (ucx_shm) plus two pipe pairs for wake-up.

Session (ChannelSession): one ATTACH = one channel. On the daemon side:
  * cmd pump thread: drains the cmd ring and executes SEND (send from a
    staging block) / RECV (take a free recv block, recv_into) /
    RECV_DISCARD / RECV_RELEASE / FAIL / CLOSE. It doubles as the link
    state observer (ready/broken change -> cpl + shm status word);
  * LwdUcxChannel completion callbacks (lib IO thread) -> cpl ring + pipe
    wake-up to the engine.

Lifecycle (v1 scope): engine exit/crash (ctl disconnect) means DETACH: the
channel closes and the existing "restart the whole deployment" policy
applies. Keeping the link alive across an engine restart requires rebuilding
upper-layer seqno state and is deferred to a later version. The daemon exits
on its own after 30s with no session.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import select
import socket
import sys
import threading
import time
import traceback
from collections import deque

from vllm.logger import init_logger
from vllm.v1.lwd_comm.data_comm import ucx_shm as shm
from vllm.v1.lwd_comm.data_comm.ucx_channel import (
    LwdUcxChannel,
    LwdUcxConfig,
    LwdUcxListener,
)

logger = init_logger("lwd_host_comm")

# --------------------------------------------------------------------- #
# External names (process/thread/log): lwd_host_comm; code identifiers stay
# ucx/daemon (implementation names). This is the single source of truth for
# external names; engine-side ucx_port matches them by the sock convention.
# Any change must be mirrored in the deployment doc's sock/log/pkill rules.
# --------------------------------------------------------------------- #
PROC_NAME = "lwd_host_comm"  # process title seen by ps/systemd/container
THREAD_PREFIX = "lwd-hc-"  # thread name prefix (ps -T/py-spy)
LOG_TAG = "[lwd-hc]"  # log prefix

_IDLE_EXIT_S = 30.0
_CTL_BACKLOG = 8
_CTL_PROTO = 2  # ctl protocol version; engine validates via PING, kills on mismatch


def _pack_fds(fds: list) -> bytes:
    """SCM_RIGHTS payload: int fd list -> native int array bytes."""
    import array

    return array.array("i", fds).tobytes()


class _Session:
    """One ATTACH channel: LwdUcxChannel + shm segment + cmd pump."""

    _next_key = 1
    _key_lock = threading.Lock()

    def __init__(self, req: dict, daemon: LwdUcxDaemon) -> None:
        with _Session._key_lock:
            self.key = _Session._next_key
            _Session._next_key += 1
        cfg = LwdUcxConfig(
            block_mb=int(req["block_mb"]),
            n_blocks=int(req["n_blocks"]),
            put_thresh=int(req.get("put_thresh", 0)),
            hs_timeout_s=float(req.get("hs_timeout_s", 5.0)),
            token=int(req["token"]),
            early_max=int(req.get("early_max", 0)),
            pool_staging=bool(req.get("pool_staging", False)),
        )
        self._block_bytes = max(1, int(cfg.block_mb)) << 20
        # Send direction: engine D2H lands in staging blocks. Recv direction:
        # daemon recv_into lands in recv blocks (engine H2D). A send-only
        # channel builds no recv pool, and a recv-only channel builds no
        # staging pool (same direction semantics as the embedded block pool).
        self._n_staging = cfg.n_blocks if cfg.pool_staging else 0
        self._n_recv = 0 if cfg.pool_staging else cfg.n_blocks
        self.seg = shm.LwdUcxShmSegment(
            block_bytes=self._block_bytes,
            n_staging=self._n_staging,
            n_recv=self._n_recv,
            creator=True,
        )
        # Wake-up pipes: engine writes daemon_wake_w; daemon writes engine_wake_w
        self._d_wake_r, self._d_wake_w = os.pipe()
        self._e_wake_r, self._e_wake_w = os.pipe()
        os.set_blocking(self._d_wake_w, False)
        os.set_blocking(self._e_wake_w, False)
        self._ch = daemon.open_channel(cfg, req, self)
        # Free recv block pool (daemon side). A RECV command takes one, a
        # RECV_RELEASE gives one back. With no free block, RECV goes into
        # _pending_recvs and is re-posted in FIFO order once a block returns,
        # so the pump never blocks. Concurrent pre-posts may far exceed the
        # pool depth: taking a block inline once blocked the pump thread,
        # filled the cmd ring and crashed the engine dispatcher on a real run
        # (27 concurrent pre-posts, pool depth 4).
        self._free_recv: queue.Queue[int] = queue.Queue()
        for i in range(self._n_recv):
            self._free_recv.put(i)
        self._pending_recvs: deque[tuple[int, int]] = deque()  # (tag, op_id)
        # cpl ring producer mutex: lib IO completion callbacks and the pump
        # thread (status changes) write concurrently. Multiple producers in
        # one process must serialize locally, or concurrent overwrites drop
        # cpl entries and hang the engine-side future (same root cause as the
        # engine cmd ring).
        self._cpl_lock = threading.Lock()
        self._stopped = False
        self._stop_requested = False  # set by ctl-side DETACH; pump polls it
        self._closed = threading.Event()
        self._pump = threading.Thread(
            target=self._pump_loop, name=f"{THREAD_PREFIX}sess-{self.key}", daemon=True
        )
        self._pump.start()

    # ------------------------------------------------------------------ #
    # engine-side fd/shm handles (when answering ctl)                    #
    # ------------------------------------------------------------------ #
    def fds_for_engine(self) -> list[int]:
        # [shm, wake_daemon_w (engine writes, pump reads the other end),
        #  wake_engine_r (daemon writes the other end, engine dispatcher
        #  reads)]
        return [self.seg.fd, self._d_wake_w, self._e_wake_r]

    def attach_reply(self) -> dict:
        return {
            "ok": True,
            "key": self.key,
            "block_bytes": self._block_bytes,
            "n_staging": self._n_staging,
            "n_recv": self._n_recv,
            "cmd_slots": self.seg.cmd_slots,
            "cpl_slots": self.seg.cpl_slots,
        }

    # ------------------------------------------------------------------ #
    # LwdUcxChannel callbacks (lib IO thread) -> cpl                     #
    # ------------------------------------------------------------------ #
    def on_send_done(self, op_id: int):
        def cb(op) -> None:  # noqa: ANN001
            self._cpl(
                shm.CPL_SEND_DONE,
                op_id=op_id,
                size=op.size,
                err=0 if op.error is None else 1,
            )

        return cb

    def on_recv_done(self, op_id: int, block_idx: int):
        def cb(op) -> None:  # noqa: ANN001
            self._cpl(
                shm.CPL_RECV_DONE,
                op_id=op_id,
                size=op.size,
                err=0 if op.error is None else 1,
                block_idx=block_idx,
            )
            if op.error is not None:
                self._free_recv.put(block_idx)
                self._serve_pending_recvs()

        return cb

    def _wake_engine(self) -> None:
        # A wake-up is already pending; a full pipe coalesces.
        with contextlib.suppress(BlockingIOError):
            os.write(self._e_wake_w, b"\x01")

    def _cpl(self, kind: int, **kw) -> None:
        """Sole entry point for producing into the cpl ring: put and wake
        under the lock, so concurrent threads cannot overwrite each other."""
        with self._cpl_lock:
            self.seg.cpl_put(kind, **kw)
            self._wake_engine()

    # ------------------------------------------------------------------ #
    # cmd pump                                                           #
    # ------------------------------------------------------------------ #
    def _pump_loop(self) -> None:
        last_ready = None
        last_broken = None
        try:
            while not self._stopped and not self._stop_requested:
                # 1) drain the cmd ring
                progressed = False
                while True:
                    cmd = self.seg.cmd_get()
                    if cmd is None:
                        break
                    progressed = True
                    self._exec(*cmd)  # kind,rsv,tag,block_idx,rsv,size,op_id
                    if self._stopped:
                        break
                # 2) link state watch (a change -> cpl + shm status word)
                ready = self._ch.ready
                broken = self._ch.broken
                if ready != last_ready or broken != last_broken:
                    last_ready, last_broken = ready, broken
                    self.seg.set_status(ready, broken)
                    self._cpl(shm.CPL_READY if ready else shm.CPL_BROKEN)
                # 3) wait for wake-up (100ms timeout as a fallback status
                #    poll). Read only when select reports readable: the fd is
                #    blocking, so an unconditional read would block forever,
                #    leaving the pump dead in its first iteration and READY
                #    never published (observed on real runs).
                if not progressed:
                    try:
                        r, _, _ = select.select([self._d_wake_r], [], [], 0.1)
                        if r:
                            os.read(self._d_wake_r, 64)
                    except (BlockingIOError, InterruptedError, OSError):
                        pass
        except Exception:  # noqa: BLE001
            # A pump crash with no cleanup hangs the engine-side wait_ready
            # forever and stops link state updates (observed on real runs:
            # the session thread vanished and the engine timed out). Teardown
            # closes e_wake_w, so the engine dispatcher sees EOF and exits at
            # once: a silent hang becomes an explicit failure, and the
            # traceback lands in the daemon log.
            logger.exception("%s sess=%d pump crashed", LOG_TAG, self.key)
        finally:
            self._teardown()

    def _exec(
        self,
        kind: int,
        _r: int,
        tag: int,
        block_idx: int,
        _r2: int,
        size: int,
        op_id: int,
    ) -> None:
        if kind == shm.CMD_SEND:
            if block_idx >= self._n_staging:
                logger.error(
                    "%s sess=%d bad staging idx %d", LOG_TAG, self.key, block_idx
                )
                return
            self._ch.send(
                tag, self.seg.staging_addr(block_idx), size, cb=self.on_send_done(op_id)
            )
        elif kind == shm.CMD_RECV:
            if self._n_recv == 0:
                self._cpl(shm.CPL_RECV_DONE, op_id=op_id, err=1)
                return
            self._pending_recvs.append((tag, op_id))
            self._serve_pending_recvs()
        elif kind == shm.CMD_RECV_DISCARD:
            self._ch.recv(tag, cb=None)
        elif kind == shm.CMD_RECV_RELEASE:
            if 0 <= block_idx < self._n_recv:
                self._free_recv.put(block_idx)
                self._serve_pending_recvs()
        elif kind == shm.CMD_FAIL:
            self._ch.fail(RuntimeError("engine-side fatal"))
        elif kind == shm.CMD_CLOSE:
            self._stopped = True

    def _serve_pending_recvs(self) -> None:
        """Re-post pending RECVs in FIFO order whenever a free recv block
        appears.

        The pump thread and lib IO callbacks may call this concurrently. The
        per-operation atomicity of queue.Queue and deque guarantees neither
        duplicates nor omissions: each loop iteration takes distinct blocks
        and requests.
        """
        while self._pending_recvs:
            try:
                idx = self._free_recv.get_nowait()
            except queue.Empty:
                return
            tag, op_id = self._pending_recvs.popleft()
            self._ch.recv_into(
                tag,
                self.seg.recv_addr(idx),
                self._block_bytes,
                cb=self.on_recv_done(op_id, idx),
            )

    def _teardown(self) -> None:
        with contextlib.suppress(Exception):
            self._ch.close(timeout=2.0)
        self._cpl(shm.CPL_CLOSED)
        for fd in (self._d_wake_r, self._e_wake_w):
            with contextlib.suppress(Exception):
                os.close(fd)
        self.seg.close()
        self._closed.set()

    def request_stop(self) -> None:
        """ctl-side DETACH: set the stop flag; the pump loop (<=100ms cycle)
        exits by itself.

        The cmd ring is not written here: its producer side belongs to the
        engine process, and writing from the ctl thread would add a third,
        cross-process producer and break the SPSC contract (the engine side
        already locks within its own process, and a lock cannot be shared
        across processes).
        """
        self._stop_requested = True

    def wait_closed(self, timeout: float = 5.0) -> None:
        self._closed.wait(timeout)


class LwdUcxDaemon:
    def __init__(self, sock_path: str) -> None:
        self._path = sock_path
        self._sessions: dict[int, _Session] = {}
        self._lock = threading.Lock()
        self._listeners: dict[tuple[str, int], LwdUcxListener] = {}
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._idle_since: float | None = time.monotonic()

    def open_channel(
        self, cfg: LwdUcxConfig, req: dict, sess: _Session
    ) -> LwdUcxChannel:
        role = req.get("role", "connect")
        host, port = req.get("host", ""), int(req.get("port", 0))
        if role == "serve":
            # One shared listener per (host,port); tokens route to each
            # channel. Register first (done inside serve_via) and start
            # afterwards, so an accept cannot hit an unknown token before
            # routing is ready.
            lk = (host or "0.0.0.0", port)
            listener = self._listeners.get(lk)
            fresh = listener is None
            if listener is None:
                listener = LwdUcxListener(lk[0], lk[1])
                self._listeners[lk] = listener
            ch = LwdUcxChannel.serve_via(listener, cfg=cfg)
            if fresh:
                listener.start()
            return ch
        if not (host and port):
            raise RuntimeError("connect role requires host and port")
        return LwdUcxChannel.connect(host, port, cfg=cfg)

    # ------------------------------------------------------------------ #
    # ctl service                                                        #
    # ------------------------------------------------------------------ #
    def serve_forever(self) -> None:
        if os.path.exists(self._path):
            os.unlink(self._path)
        self._srv.bind(self._path)
        self._srv.listen(_CTL_BACKLOG)
        # Timeout on accept: the idle-exit check must run periodically;
        # otherwise accept blocks forever and the daemon lingers after the
        # engine exits (observed on real runs).
        self._srv.settimeout(1.0)
        with open(self._path + ".pid", "w") as f:
            f.write(str(os.getpid()))
        logger.warning(
            "%s ctl listening on %s pid=%d proto=%d",
            LOG_TAG,
            self._path,
            os.getpid(),
            _CTL_PROTO,
        )
        while True:
            with self._lock:
                if not self._sessions:
                    if (
                        self._idle_since is not None
                        and time.monotonic() - self._idle_since > _IDLE_EXIT_S
                    ):
                        logger.warning("%s idle %.0fs, exiting", LOG_TAG, _IDLE_EXIT_S)
                        with contextlib.suppress(OSError):
                            os.unlink(self._path + ".pid")
                        return
                else:
                    self._idle_since = None
            try:
                conn, _ = self._srv.accept()
            except TimeoutError:
                continue  # loop back periodically to run the idle check
            except OSError:
                continue
            # One thread per ctl connection; engine disconnect DETACHes all
            threading.Thread(
                target=self._serve_conn,
                args=(conn,),
                name=f"{THREAD_PREFIX}ctl",
                daemon=True,
            ).start()

    def _serve_conn(self, conn: socket.socket) -> None:
        attached: list[int] = []
        buf = b""
        try:
            while True:
                data = conn.recv(65536)
                if not data:
                    break  # engine exit: DETACH all (whole-deployment restart policy)
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    req = json.loads(line)
                    try:
                        resp, fds = self._handle(req, attached)
                    except Exception:  # noqa: BLE001
                        # Errors while handling ATTACH must be returned to
                        # the engine (traceback included); otherwise the
                        # engine only sees a ctl disconnect and cannot
                        # diagnose it.
                        logger.exception("%s handle %r failed", LOG_TAG, req.get("cmd"))
                        resp = {"ok": False, "error": traceback.format_exc(limit=8)}
                        fds = []
                    conn.sendmsg(
                        [json.dumps(resp).encode() + b"\n"],
                        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, _pack_fds(fds))]
                        if fds
                        else [],
                    )
        except (ConnectionError, json.JSONDecodeError, OSError):
            pass
        except Exception:  # noqa: BLE001
            logger.exception("%s ctl conn crashed", LOG_TAG)
        finally:
            with contextlib.suppress(Exception):
                conn.close()
            for key in attached:
                self._detach(key)

    def _handle(self, req: dict, attached: list) -> tuple[dict, list]:
        cmd = req.get("cmd")
        if cmd == "PING":
            return {
                "ok": True,
                "proto": _CTL_PROTO,
                "sessions": len(self._sessions),
            }, []
        if cmd == "ATTACH":
            try:
                sess = _Session(req, self)
            except Exception as exc:  # noqa: BLE001
                logger.exception("%s attach failed: %r", LOG_TAG, exc)
                return {"ok": False, "error": repr(exc)}, []
            with self._lock:
                self._sessions[sess.key] = sess
            attached.append(sess.key)
            logger.warning(
                "%s attach key=%d role=%s host=%s port=%d token=%#x",
                LOG_TAG,
                sess.key,
                req.get("role"),
                req.get("host"),
                req.get("port"),
                req.get("token"),
            )
            return sess.attach_reply(), sess.fds_for_engine()
        if cmd == "DETACH":
            self._detach(int(req.get("key", 0)))
            return {"ok": True}, []
        return {"ok": False, "error": f"unknown cmd {cmd!r}"}, []

    def _detach(self, key: int) -> None:
        with self._lock:
            sess = self._sessions.pop(key, None)
            if self._sessions:
                self._idle_since = None
            elif self._idle_since is None:
                self._idle_since = time.monotonic()
        if sess is None:
            return
        logger.warning("%s detach key=%d", LOG_TAG, key)
        # The pump thread stops via a CLOSE cmd; here set the stop flag and
        # wait for teardown.
        sess.request_stop()
        sess.wait_closed()


def _main() -> None:
    ap = argparse.ArgumentParser(prog=PROC_NAME)
    ap.add_argument("--ctl", required=True, help="ctl unix socket path")
    args = ap.parse_args()
    _set_proc_title()
    LwdUcxDaemon(args.ctl).serve_forever()
    sys.exit(0)


def _set_proc_title() -> None:
    """External process title (visible to ps/systemd/pkill -f); skipped when
    setproctitle is unavailable."""
    try:
        import setproctitle

        setproctitle.setproctitle(PROC_NAME)
    except ImportError:
        pass


if __name__ == "__main__":
    _main()
