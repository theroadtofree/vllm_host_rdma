# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""UCX 传输守护进程(daemon 模式执行体;engine 侧对应 ucx_port.DaemonPort)。

部署形态:与 engine 同容器、由首个 daemon 模式引擎进程按需 spawn
(``python -m vllm.v1.lwd_comm.ucx_daemon --ctl <sock>``);UCX env
(LD_LIBRARY_PATH/UCX_TLS/UCX_NET_DEVICES)从引擎进程原样继承。

职责:持有全部 UcxChannel/UcxListener(线上协议 UHC1、握手、token
路由、重连、put 路径全部复用 ucx_channel,零改动);engine 经 ctl
unix socket ATTACH/DETACH,数据面走 shm 环(ucx_shm)+ 两对管道唤醒。

会话(ChannelSession):一条 ATTACH = 一条通道。daemon 侧:
  * cmd 泵线程——drain cmd 环,执行 SEND(从 staging 块发)/RECV
    (取空闲 recv 块 recv_into)/RECV_DISCARD/RECV_RELEASE/FAIL/CLOSE;
    同时兼链路状态观察者(ready/broken 变化 → cpl + shm 状态字);
  * UcxChannel 完成回调(lib IO 线程)→ cpl 环 + 管道唤醒 engine。

生命周期(v1 口径):engine 退出/崩溃(ctl 断开)即 DETACH——通道
关闭、按现有"整部署重启"口径处理;"engine 重启链路保活"需上层
seqno 状态重建,留待后续版本。无会话 30s 后自动退出。
"""

from __future__ import annotations

import argparse
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
from vllm.v1.lwd_comm import ucx_shm as shm
from vllm.v1.lwd_comm.ucx_channel import UcxChannel, UcxCfg, UcxListener

logger = init_logger("lwd_host_comm")

# --------------------------------------------------------------------- #
# 对外呈现名(进程/线程/日志):lwd_host_comm;代码标识符保持 ucx/daemon
# (实现名)。对外名的唯一事实源在此,engine 侧 ucx_port 按 sock 约定
# 对齐;修改须同步部署文档的 sock/日志/pkill 口径。
# --------------------------------------------------------------------- #
PROC_NAME = "lwd_host_comm"     # ps/systemd/容器可见的进程名
THREAD_PREFIX = "lwd-hc-"       # 线程名前缀(ps -T/py-spy)
LOG_TAG = "[lwd-hc]"            # 日志前缀

_IDLE_EXIT_S = 30.0
_CTL_BACKLOG = 8
_CTL_PROTO = 2   # ctl 协议版本(engine 侧 PING 校验;不匹配即杀重启)


def _pack_fds(fds: list) -> bytes:
    """SCM_RIGHTS 载荷:int fd 列表 → 平台原生 int 数组字节。"""
    import array
    return array.array("i", fds).tobytes()


class _Session:
    """一条 ATTACH 通道:UcxChannel + shm 段 + cmd 泵。"""

    _next_key = 1
    _key_lock = threading.Lock()

    def __init__(self, req: dict, daemon: "UcxDaemon") -> None:
        with _Session._key_lock:
            self.key = _Session._next_key
            _Session._next_key += 1
        cfg = UcxCfg(
            block_mb=int(req["block_mb"]),
            n_blocks=int(req["n_blocks"]),
            put_thresh=int(req.get("put_thresh", 0)),
            hs_timeout_s=float(req.get("hs_timeout_s", 5.0)),
            token=int(req["token"]),
            early_max=int(req.get("early_max", 0)),
            pool_staging=bool(req.get("pool_staging", False)),
        )
        self._block_bytes = max(1, int(cfg.block_mb)) << 20
        # 发送向:engine D2H 进 staging 块;接收向:daemon recv_into 落
        # recv 块(engine H2D)。只发通道不建 recv 池,只收通道不建
        # staging 池(与 embedded 块池方向语义一致)。
        self._n_staging = cfg.n_blocks if cfg.pool_staging else 0
        self._n_recv = 0 if cfg.pool_staging else cfg.n_blocks
        self.seg = shm.ShmSegment(block_bytes=self._block_bytes,
                                  n_staging=self._n_staging,
                                  n_recv=self._n_recv, creator=True)
        # 唤醒管道:engine 写 daemon_wake_w;daemon 写 engine_wake_w
        self._d_wake_r, self._d_wake_w = os.pipe()
        self._e_wake_r, self._e_wake_w = os.pipe()
        os.set_blocking(self._d_wake_w, False)
        os.set_blocking(self._e_wake_w, False)
        self._ch = daemon.open_channel(cfg, req, self)
        # recv 块空闲池(daemon 侧);RECV 命令到达时取,RECV_RELEASE 归还。
        # 无空闲块时 RECV 进 _pending_recvs 等块归还后 FIFO 补挂——泵
        # 永不阻塞(并发预挂数可远超池深;曾因取块阻塞泵线程导致 cmd 环
        # 打满、engine dispatcher 崩溃,真机实录并发27/池深4)。
        self._free_recv: queue.Queue[int] = queue.Queue()
        for i in range(self._n_recv):
            self._free_recv.put(i)
        self._pending_recvs: deque[tuple[int, int]] = deque()  # (tag, op_id)
        # cpl 环生产者互斥:lib IO 回调(完成)与泵线程(状态变化)并发写,
        # 同进程多生产者必须本地互斥,否则并发覆盖丢 cpl(engine 侧
        # future 卡死;与 engine cmd 环同因)
        self._cpl_lock = threading.Lock()
        self._stopped = False
        self._stop_requested = False   # ctl 侧 DETACH 置位,泵循环自查
        self._closed = threading.Event()
        self._pump = threading.Thread(target=self._pump_loop,
                                      name=f"{THREAD_PREFIX}sess-{self.key}",
                                      daemon=True)
        self._pump.start()

    # ------------------------------------------------------------------ #
    # engine 侧收 fd/shm 句柄(ctl 应答时调用)                          #
    # ------------------------------------------------------------------ #
    def fds_for_engine(self) -> list[int]:
        # [shm, wake_daemon_w(engine 写,泵读对侧), wake_engine_r(
        #  daemon 写对侧,engine dispatcher 读)]
        return [self.seg.fd, self._d_wake_w, self._e_wake_r]

    def attach_reply(self) -> dict:
        return {"ok": True, "key": self.key,
                "block_bytes": self._block_bytes,
                "n_staging": self._n_staging, "n_recv": self._n_recv,
                "cmd_slots": self.seg.cmd_slots,
                "cpl_slots": self.seg.cpl_slots}

    # ------------------------------------------------------------------ #
    # UcxChannel 回调(lib IO 线程)→ cpl                                #
    # ------------------------------------------------------------------ #
    def on_send_done(self, op_id: int):
        def cb(op) -> None:  # noqa: ANN001
            self._cpl(shm.CPL_SEND_DONE, op_id=op_id,
                      size=op.size, err=0 if op.error is None else 1)
        return cb

    def on_recv_done(self, op_id: int, block_idx: int):
        def cb(op) -> None:  # noqa: ANN001
            self._cpl(shm.CPL_RECV_DONE, op_id=op_id,
                      size=op.size,
                      err=0 if op.error is None else 1,
                      block_idx=block_idx)
            if op.error is not None:
                self._free_recv.put(block_idx)
                self._serve_pending_recvs()
        return cb

    def _wake_engine(self) -> None:
        try:
            os.write(self._e_wake_w, b"\x01")
        except BlockingIOError:
            pass  # 已有未消费的唤醒,管道满即合并

    def _cpl(self, kind: int, **kw) -> None:
        """cpl 环生产者唯一入口(锁内 put+唤醒,防多线程并发覆盖)。"""
        with self._cpl_lock:
            self.seg.cpl_put(kind, **kw)
            self._wake_engine()

    # ------------------------------------------------------------------ #
    # cmd 泵                                                             #
    # ------------------------------------------------------------------ #
    def _pump_loop(self) -> None:
        last_ready = None
        last_broken = None
        try:
            while not self._stopped and not self._stop_requested:
                # 1) drain cmd 环
                progressed = False
                while True:
                    cmd = self.seg.cmd_get()
                    if cmd is None:
                        break
                    progressed = True
                    self._exec(*cmd)  # kind,rsv,tag,block_idx,rsv,size,op_id
                    if self._stopped:
                        break
                # 2) 链路状态观察(变化即 cpl + shm 状态字)
                ready = self._ch.ready
                broken = self._ch.broken
                if ready != last_ready or broken != last_broken:
                    last_ready, last_broken = ready, broken
                    self.seg.set_status(ready, broken)
                    self._cpl(shm.CPL_READY if ready else shm.CPL_BROKEN)
                # 3) 等唤醒(100ms 超时兜底轮询状态)。只在 select 报告
                #    可读时才 read——fd 是阻塞的,无条件 read 会永久阻塞,
                #    泵死在首循环、READY 状态永远推不出去(真机实录)。
                if not progressed:
                    try:
                        r, _, _ = select.select([self._d_wake_r], [], [], 0.1)
                        if r:
                            os.read(self._d_wake_r, 64)
                    except (BlockingIOError, InterruptedError, OSError):
                        pass
        except Exception:  # noqa: BLE001
            # 泵异常死亡若无善后:engine 侧 wait_ready 永久挂死、链路
            # 状态不再更新(真机实录:sess 线程消失、engine 超时)。
            # teardown 关 e_wake_w → engine dispatcher EOF 立即退出,
            # 静默挂死转为显式失败;traceback 落 daemon 日志。
            logger.error(f"{LOG_TAG} sess=%d pump crashed", self.key,
                         exc_info=True)
        finally:
            self._teardown()

    def _exec(self, kind: int, _r: int, tag: int, block_idx: int,
              _r2: int, size: int, op_id: int) -> None:
        if kind == shm.CMD_SEND:
            if block_idx >= self._n_staging:
                logger.error(f"{LOG_TAG} sess=%d bad staging idx %d",
                             self.key, block_idx)
                return
            self._ch.send(tag, self.seg.staging_addr(block_idx), size,
                          cb=self.on_send_done(op_id))
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
        """有空闲 recv 块就按 FIFO 补挂待处理 RECV(泵线程/lib IO 回调
        可能并发调用;queue.Queue 与 deque 的单操作原子性保证不重不漏,
        每次循环至多取走互不相同的块/请求)。"""
        while self._pending_recvs:
            try:
                idx = self._free_recv.get_nowait()
            except queue.Empty:
                return
            tag, op_id = self._pending_recvs.popleft()
            self._ch.recv_into(tag, self.seg.recv_addr(idx),
                               self._block_bytes,
                               cb=self.on_recv_done(op_id, idx))

    def _teardown(self) -> None:
        with _suppress():
            self._ch.close(timeout=2.0)
        self._cpl(shm.CPL_CLOSED)
        for fd in (self._d_wake_r, self._e_wake_w):
            with _suppress():
                os.close(fd)
        self.seg.close()
        self._closed.set()

    def request_stop(self) -> None:
        """ctl 侧 DETACH:置停止标志,泵循环(≤100ms 周期)自行退出。

        不写 cmd 环——cmd 环生产端属 engine 进程,本方法在 ctl 线程
        写会引入跨进程第三生产者破坏 SPSC 契约(engine 侧已加同进程
        锁,跨进程无法共享锁)。
        """
        self._stop_requested = True

    def wait_closed(self, timeout: float = 5.0) -> None:
        self._closed.wait(timeout)


class UcxDaemon:
    def __init__(self, sock_path: str) -> None:
        self._path = sock_path
        self._sessions: dict[int, _Session] = {}
        self._lock = threading.Lock()
        self._listeners: dict[tuple[str, int], UcxListener] = {}
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._idle_since: float | None = time.monotonic()

    def open_channel(self, cfg: UcxCfg, req: dict,
                     sess: _Session) -> UcxChannel:
        role = req.get("role", "connect")
        host, port = req.get("host", ""), int(req.get("port", 0))
        if role == "serve":
            # 每 (host,port) 一个共享 listener:token 路由到各通道。
            # 先 register(serve_via 内完成)后 start——避免 accept 先于
            # 路由就绪撞上未知 token。
            lk = (host or "0.0.0.0", port)
            listener = self._listeners.get(lk)
            fresh = listener is None
            if listener is None:
                listener = UcxListener(lk[0], lk[1])
                self._listeners[lk] = listener
            ch = UcxChannel.serve_via(listener, cfg=cfg)
            if fresh:
                listener.start()
            return ch
        if not (host and port):
            raise RuntimeError("connect 形态需 host/port")
        return UcxChannel.connect(host, port, cfg=cfg)

    # ------------------------------------------------------------------ #
    # ctl 服务                                                            #
    # ------------------------------------------------------------------ #
    def serve_forever(self) -> None:
        if os.path.exists(self._path):
            os.unlink(self._path)
        self._srv.bind(self._path)
        self._srv.listen(_CTL_BACKLOG)
        # accept 带超时:空闲退出检查必须周期性执行(否则 accept 永久
        # 阻塞,daemon 在 engine 退出后残留——旧版真机实录)
        self._srv.settimeout(1.0)
        with open(self._path + ".pid", "w") as f:
            f.write(str(os.getpid()))
        logger.warning(f"{LOG_TAG} ctl listening on %s pid=%d proto=%d",
                       self._path, os.getpid(), _CTL_PROTO)
        while True:
            with self._lock:
                if not self._sessions:
                    if self._idle_since is not None and \
                            time.monotonic() - self._idle_since > _IDLE_EXIT_S:
                        logger.warning(f"{LOG_TAG} idle %.0fs, exiting",
                                       _IDLE_EXIT_S)
                        try:
                            os.unlink(self._path + ".pid")
                        except OSError:
                            pass
                        return
                else:
                    self._idle_since = None
            try:
                conn, _ = self._srv.accept()
            except TimeoutError:
                continue   # 周期性回到循环头做空闲检查
            except OSError:
                continue
            # 每 ctl 连接一个服务线程(engine 崩溃断连 = DETACH 全部)
            threading.Thread(target=self._serve_conn, args=(conn,),
                             name=f"{THREAD_PREFIX}ctl", daemon=True).start()

    def _serve_conn(self, conn: socket.socket) -> None:
        attached: list[int] = []
        buf = b""
        try:
            while True:
                data = conn.recv(65536)
                if not data:
                    break  # engine 退出:DETACH 全部(整部署重启口径)
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    req = json.loads(line)
                    try:
                        resp, fds = self._handle(req, attached)
                    except Exception:  # noqa: BLE001
                        # ATTACH 处理期异常必须回传 engine(含 traceback),
                        # 否则 engine 只见 ctl 断连无从诊断
                        logger.error(f"{LOG_TAG} handle %r failed", req.get("cmd"),
                                     exc_info=True)
                        resp = {"ok": False,
                                "error": traceback.format_exc(limit=8)}
                        fds = []
                    conn.sendmsg(
                        [json.dumps(resp).encode() + b"\n"],
                        [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                          _pack_fds(fds))] if fds else [])
        except (ConnectionError, json.JSONDecodeError, OSError):
            pass
        except Exception:  # noqa: BLE001
            logger.error(f"{LOG_TAG} ctl conn crashed", exc_info=True)
        finally:
            with _suppress():
                conn.close()
            for key in attached:
                self._detach(key)

    def _handle(self, req: dict, attached: list) -> tuple[dict, list]:
        cmd = req.get("cmd")
        if cmd == "PING":
            return {"ok": True, "proto": _CTL_PROTO,
                    "sessions": len(self._sessions)}, []
        if cmd == "ATTACH":
            try:
                sess = _Session(req, self)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"{LOG_TAG} attach failed: %r", exc, exc_info=True)
                return {"ok": False, "error": repr(exc)}, []
            with self._lock:
                self._sessions[sess.key] = sess
            attached.append(sess.key)
            logger.warning(f"{LOG_TAG} attach key=%d role=%s host=%s port=%d "
                           "token=%#x", sess.key, req.get("role"),
                           req.get("host"), req.get("port"),
                           req.get("token"))
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
        logger.warning(f"{LOG_TAG} detach key=%d", key)
        # 泵线程经 CLOSE cmd 停;直接置停并等善后
        sess.request_stop()
        sess.wait_closed()


class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True


def _main() -> None:
    ap = argparse.ArgumentParser(prog=PROC_NAME)
    ap.add_argument("--ctl", required=True, help="ctl unix socket path")
    args = ap.parse_args()
    _set_proc_title()
    UcxDaemon(args.ctl).serve_forever()
    sys.exit(0)


def _set_proc_title() -> None:
    """对外进程名(ps/systemd/pkill -f 可见);无 setproctitle 时跳过。"""
    try:
        import setproctitle
        setproctitle.setproctitle(PROC_NAME)
    except ImportError:
        pass


if __name__ == "__main__":
    _main()
