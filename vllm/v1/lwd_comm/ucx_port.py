# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""UCX 传输端口抽象:embedded(进程内直连 UcxChannel)/ daemon(独立
传输守护进程 ucx_daemon + shm 环)双模式的统一入口。

引擎侧装配层(如 vllm-ascend 的 ucx_npu_channel)只面向本模块的
``UcxTransportPort`` 协议编程,不再直接触碰 ``UcxChannel`` 的构造类
方法。embedded 返回 UcxChannel 本体(现状,零行为变化);daemon 返回
``DaemonPort``——数据面走 shm 环(ucx_shm),控制面走 ctl unix socket,
唤醒走两对管道字节。两模式线上均为 UHC1,对端无感知,但同一条链路
两端须同模式整对部署(混合态启动即报错,不静默回退——灰度口径与
family 一致)。

协议方法即 UcxChannel 公共调用面中被装配层使用的子集:
  * acquire_staging / release_staging —— 发送向 D2H 暂存块周转
    (池深=在途上限,承担背压);
  * send / recv / fail —— 数据与协议错路径;
  * wait_ready / close —— 生命周期;
  * recv_discard(daemon 可选,embedded 不存在)—— skip 的丢弃预挂。
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
from vllm.v1.lwd_comm import ucx_shm as _shm
from vllm.v1.lwd_comm.ucx_channel import UcxChannel, UcxCfg

logger = init_logger(__name__)

_SPAWN_CONNECT_TIMEOUT_S = 10.0


class UcxTransportPort(Protocol):
    """一条通道的传输端口(engine 侧视角;方法契约见 UcxChannel)。"""

    def acquire_staging(self) -> Any:
        """取一块池内存做 D2H 暂存(阻塞至有空块;shutdown 抛错)。"""
        ...

    def release_staging(self, blk: Any) -> None:
        """归还暂存块(异常路径;正常路径由发送完成归还)。"""
        ...

    def send(self, tag: int, addr: int, size: int, *,
             ref: Any = None, cb: Any = None) -> Any:
        """发送 addr 起始的 size 字节;返回完成句柄。"""
        ...

    def recv(self, tag: int, *, cb: Any = None) -> Any:
        """预挂一条接收(完成回调拿 op.data;daemon 模式 data 为 recv
        块的 shm 视图,调用方 H2D 后须调 op.mark_recycle(event))。"""
        ...

    def fail(self, reason: BaseException | str) -> None:
        """应用侧检出不可恢复错:在途 op 全部 ERROR 终结。"""
        ...

    def wait_ready(self, timeout: float | None = None) -> bool:
        """等首链就绪;False = 超时。"""
        ...

    def close(self, timeout: float = 2.0) -> None:
        """有序关停(不触发 on_broken)。"""
        ...


def open_transport(
    cfg: UcxCfg,
    *,
    mode: str = "embedded",
    serve: bool = False,
    host: str = "",
    port: int = 0,
    listener: Any = None,
    ctl_sock: str = "",
) -> UcxTransportPort:
    """按模式开一条传输端口。

    embedded:进程内直连 UcxChannel(现状路径,零行为变化)。
    daemon:   独立守护进程(ucx_daemon)+ shm 环(DaemonPort)。

    serve/listener 语义与 UcxChannel.serve/serve_via 对齐:边侧
    connect(host, port);云侧 serve=True(host, port)或传入共享
    listener(按握手 token 路由)。daemon 模式下 listener 必须为
    None——listener 本身移入守护进程(按 (host,port) 共享、token
    路由);ctl_sock 为 daemon ctl unix socket 路径(空 = 默认
    /tmp/lwd-host-comm-{uid}.sock,daemon 由本进程按需 spawn)。
    """
    if mode == "embedded":
        if serve:
            if listener is not None:
                return UcxChannel.serve_via(listener, cfg=cfg)
            return UcxChannel.serve(host, port, cfg=cfg)
        if not (host and port):
            raise RuntimeError(
                "connect 形态需 host/port(serve 形态需 serve=True)")
        return UcxChannel.connect(host, port, cfg=cfg)
    if mode == "daemon":
        if listener is not None:
            raise RuntimeError(
                "daemon 模式不支持 engine 侧 listener(listener 属守护"
                "进程;lwd_wire 装配层 daemon 分支应传 listener=None)")
        return DaemonPort(cfg, serve=serve, host=host, port=port,
                          ctl_sock=ctl_sock)
    raise RuntimeError(f"unknown ucx transport mode {mode!r} "
                       "(expect 'embedded' or 'daemon')")


# --------------------------------------------------------------------- #
# daemon 模式:engine 侧端口(shm 环 + ctl)                             #
# --------------------------------------------------------------------- #
def _default_ctl_sock() -> str:
    # 对外名 lwd_host_comm(与 ucx_daemon.PROC_NAME 对齐);代码标识符
    # 保持 ucx/daemon(实现名)。
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return f"/tmp/lwd-host-comm-{uid}.sock"


def _daemon_log_path(ctl_sock: str) -> str:
    return ctl_sock + ".log"


def _kill_stale_daemon(ctl_sock: str) -> None:
    """按 pidfile 击杀不兼容的旧 daemon(无 pidfile 时尽力而为)。"""
    import signal
    pid_path = ctl_sock + ".pid"
    try:
        with open(pid_path) as f:
            pid = int(f.read().strip())
        os.kill(pid, signal.SIGKILL)
        logger.warning("[ucx-port] killed stale lwd_host_comm pid=%d "
                       "(ctl proto mismatch)", pid)
    except (OSError, ValueError):
        pass  # 无 pidfile(更旧版本)或已退出:由用户 pkill 兜底
    with contextlib.suppress(OSError):
        os.unlink(pid_path)
    with contextlib.suppress(OSError):
        os.unlink(ctl_sock)


def _ctl_connect_or_spawn(ctl_sock: str) -> _pysocket.socket:
    """连已有 daemon;没有则 spawn(start_new_session 脱离引擎进程组)
    并重试连接。失败抛错——不静默回退 embedded(两端须同模式)。"""
    def _try() -> _pysocket.socket | None:
        s = _pysocket.socket(_pysocket.AF_UNIX, _pysocket.SOCK_STREAM)
        try:
            s.connect(ctl_sock)
            # 连上 ≠ 可用:校验 ctl 协议版本(旧版 daemon 行为不兼容,
            # 且可能吞 ATTACH 异常直接断连——真机实录)
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
        fcntl.flock(lock_fd, fcntl.LOCK_EX)  # 同容器并发首连串行化
        conn = _try()
        if conn is None:
            # daemon stdout/stderr 落盘(vllm logger 走 stderr)——
            # spawn/ATTACH 失败时可查
            log_path = _daemon_log_path(ctl_sock)
            log_f = open(log_path, "ab", buffering=0)
            try:
                subprocess.Popen(  # noqa: S603
                    [sys.executable, "-m", "vllm.v1.lwd_comm.ucx_daemon",
                     "--ctl", ctl_sock],
                    env=os.environ.copy(), start_new_session=True,
                    stdin=subprocess.DEVNULL,
                    stdout=log_f, stderr=subprocess.STDOUT)
            finally:
                with contextlib.suppress(Exception):
                    log_f.close()
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
            f"不回退 embedded;查 daemon 日志 {_daemon_log_path(ctl_sock)})")
    return conn


class _PortOp:
    """UcxOp 兼容完成句柄(daemon 模式;cpl 到达即 _finish)。

    recv 路径的 ``data`` 是 recv 块的 shm 视图:调用方 H2D 后经
    ``mark_recycle(event)`` 登记设备事件,reaper 轮询 event.query()
    后归还块(RECV_RELEASE)。未登记事件的 op 在回调返回后 1s 兜底
    归还。
    """

    __slots__ = ("tag", "size", "data", "error", "_cb", "block_idx",
                 "_stg_idx", "_recycle_evt", "_recycle_t0", "_fin")

    def __init__(self, tag: int, cb: Any = None) -> None:
        self.tag = tag
        self.size = 0
        self.data: Any = None
        self.error: BaseException | None = None
        self._cb = cb
        self.block_idx = -1    # recv 块号(cpl 到达时填;归还依据)
        self._stg_idx = -1     # send 的 staging 块号(完成归还依据)
        self._recycle_evt: Any = None
        self._recycle_t0 = 0.0
        self._fin = False

    def _finish(self, *, size: int = 0, err: int = 0, data: Any = None,
                error: BaseException | None = None) -> None:
        if self._fin:
            return
        self._fin = True
        self.size = size
        self.error = error or (RuntimeError("daemon transport op failed")
                               if err else None)
        self.data = data
        cb = self._cb
        if cb is not None:
            try:
                cb(self)
            except Exception:  # noqa: BLE001
                logger.error("[ucx-port] op cb error tag=%s", self.tag,
                             exc_info=True)

    def done(self) -> bool:
        return self._fin

    def mark_recycle(self, event: Any) -> None:
        """收侧 H2D 完成事件登记(装配层在 future._bind 后调用)。"""
        self._recycle_evt = event
        self._recycle_t0 = time.monotonic()


class _ShmStaging:
    """acquire_staging 返回的暂存块:与 ucx_pool.Block 同形(buf/addr/
    index),供装配层 torch.frombuffer(stg.buf) D2H 直写。"""

    __slots__ = ("buf", "addr", "index", "future", "event", "recv_req",
                 "send_req", "t_post", "t_enq", "t_disp", "nbytes",
                 "_port", "_released")

    def __init__(self, seg: _shm.ShmSegment, index: int,
                 port: "DaemonPort") -> None:
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


class DaemonPort:
    """daemon 模式 engine 侧端口:满足 UcxTransportPort 协议。

    ctl 应答携带三个 fd(SCM_RIGHTS):[shm, wake_daemon_w,
    wake_engine_r]——engine 写 wake_daemon_w 唤醒 daemon cmd 泵;
    dispatcher 阻塞读 wake_engine_r 等 daemon 唤醒(EOF = 会话关停)。

    数据通路:send = D2H 直写 staging 块 → cmd SEND(零额外拷贝,
    daemon 从同一物理页发送);recv = cpl RECV_DONE → 调用方从 recv
    块 H2D → op.mark_recycle(npu_event) → reaper 归还。背压与
    embedded 同口径:staging 池深 = 在途发送上限(本地空闲表,
    SEND_DONE 补充);recv 块深 = 在途接收上限(daemon 侧空闲表)。
    """

    def __init__(self, cfg: UcxCfg, *, serve: bool, host: str, port: int,
                 ctl_sock: str) -> None:
        self._stopped = False
        self._ready = threading.Event()
        self._staging_cond = threading.Condition()
        self._free_staging: list[int] = []
        self._ops: dict[int, _PortOp] = {}     # op_id -> op
        self._pending_recycle: list[_PortOp] = []
        self._rel_retry: list[int] = []        # 归还失败待重试的 recv 块
        self._next_op_id = 1
        self._key = 0            # attach 应答填充;DETACH 用
        # cmd 环生产者互斥:engine 侧 sender/dispatcher/计算线程都发命令,
        # 环的 SPSC 契约只约束跨进程——同进程多生产者必须本地互斥,
        # 否则并发 put 覆盖丢命令(send 完成丢失/future 卡死;真机实录
        # 1000 请求并发压测必现)
        self._cmd_lock = threading.Lock()
        self._wake_daemon_w = -1
        self._wake_engine_r = -1
        self._seg: _shm.ShmSegment | None = None
        self._ctl = _ctl_connect_or_spawn(ctl_sock or _default_ctl_sock())
        req = {"cmd": "ATTACH",
               "role": "serve" if serve else "connect",
               "host": host, "port": port, "token": cfg.token,
               "block_mb": cfg.block_mb, "n_blocks": cfg.n_blocks,
               "put_thresh": cfg.put_thresh,
               "hs_timeout_s": cfg.hs_timeout_s,
               "early_max": cfg.early_max,
               "pool_staging": cfg.pool_staging}
        reply, fds = self._ctl_roundtrip(req)
        if not reply.get("ok") or len(fds) != 3:
            self._ctl.close()
            raise RuntimeError(f"daemon attach failed: {reply} fds={len(fds)}")
        self._seg = _shm.ShmSegment(block_bytes=reply["block_bytes"],
                                    n_staging=reply["n_staging"],
                                    n_recv=reply["n_recv"], fd=fds[0])
        self._wake_daemon_w = fds[1]
        self._wake_engine_r = fds[2]
        self._key = int(reply["key"])   # DETACH 用(不带 key 的 DETACH
                                        # 在 daemon 侧 detach 不到 session,
                                        # daemon 将永不空闲退出——真机实录)
        os.set_blocking(self._wake_daemon_w, False)
        with self._staging_cond:
            self._free_staging = list(range(reply["n_staging"]))
        self._disp = threading.Thread(target=self._dispatch_loop,
                                      name="ucx-port-disp", daemon=True)
        self._disp.start()
        logger.warning("[ucx-port] daemon attach ok key=%s staging=%d "
                       "recv=%d block=%dMiB", reply.get("key"),
                       reply["n_staging"], reply["n_recv"],
                       reply["block_bytes"] >> 20)

    # ------------------------------------------------------------------ #
    # ctl                                                                 #
    # ------------------------------------------------------------------ #
    def _ctl_roundtrip(self, req: dict) -> tuple[dict, list]:
        self._ctl.sendall(json.dumps(req).encode() + b"\n")
        # ATTACH 是唯一带应答的请求;应答 ancillary 携带 3 个 fd。
        # recvmsg 的 ancbufsize 默认 0 会静默丢弃 SCM_RIGHTS——必须
        # 显式给 ancillary 缓冲。
        data, anc, _flags, _addr = self._ctl.recvmsg(
            65536, _pysocket.CMSG_SPACE(4 * 8))
        if not data:
            raise ConnectionError(
                "ucx daemon ctl closed without reply (查 daemon 日志 "
                f"{_daemon_log_path(_default_ctl_sock())})")
        fds = []
        for lvl, typ, cd in anc:
            if typ == _pysocket.SCM_RIGHTS and cd:
                import array
                ints = array.array("i")
                ints.frombytes(cd[:len(cd) // 4 * 4])
                fds.extend(ints.tolist())
        return json.loads(data.split(b"\n", 1)[0]), fds

    # ------------------------------------------------------------------ #
    # 协议方法(UcxTransportPort)                                        #
    # ------------------------------------------------------------------ #
    def acquire_staging(self) -> Any:
        while True:
            with self._staging_cond:
                if self._free_staging:
                    idx = self._free_staging.pop()
                    return _ShmStaging(self._seg, idx, self)
                if self._stopped:
                    raise RuntimeError("staging wait aborted (shutdown)")
                self._staging_cond.wait(timeout=0.05)

    def release_staging(self, blk: Any) -> None:
        """engine 侧归还入口(异常路径;正常由 SEND_DONE 归还)。"""
        with self._staging_cond:
            self._free_staging.append(blk.index)
            self._staging_cond.notify_all()

    def send(self, tag: int, addr: int, size: int, *,
             ref: Any = None, cb: Any = None) -> _PortOp:
        if self._stopped:
            raise RuntimeError("daemon port closed")
        idx = self._staging_index_of(addr)
        op = _PortOp(tag, cb=cb)
        op._stg_idx = idx
        op_id = self._next_op_id
        self._next_op_id += 1
        self._ops[op_id] = op
        self._cmd(_shm.CMD_SEND, tag=tag, block_idx=idx,
                  size=size, op_id=op_id)
        return op

    def recv(self, tag: int, *, cb: Any = None) -> _PortOp:
        if self._stopped:
            raise RuntimeError("daemon port closed")
        op = _PortOp(tag, cb=cb)
        op_id = self._next_op_id
        self._next_op_id += 1
        self._ops[op_id] = op
        # cmd 环满时预挂是可等的事(daemon 排水后重试)——有界退避,
        # 不把瞬态环满直接抛给上层排序层
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
        """预挂丢弃接收(skip 路径;不占 recv 块、不产生 cpl)。"""
        if not self._stopped:
            self._cmd(_shm.CMD_RECV_DISCARD, tag=tag)

    def fail(self, reason: BaseException | str) -> None:
        logger.error("[ucx-port] fail: %s", reason)
        if not self._stopped:
            self._cmd(_shm.CMD_FAIL)

    def wait_ready(self, timeout: float | None = None) -> bool:
        """等首链就绪;False = 超时或会话已终止(dispatcher EOF 即
        daemon 侧泵死亡/关停——不等满超时,快速失败)。"""
        if timeout is None:
            timeout = 3600.0
        deadline = time.monotonic() + timeout
        while not self._ready.wait(
                min(0.5, max(0.01, deadline - time.monotonic()))):
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
                json.dumps({"cmd": "DETACH", "key": self._key}).encode()
                + b"\n")
            self._ctl.close()
        with contextlib.suppress(Exception):
            self._seg.close()
        with self._staging_cond:
            self._staging_cond.notify_all()

    # ------------------------------------------------------------------ #
    # 内部                                                                #
    # ------------------------------------------------------------------ #
    def _staging_index_of(self, addr: int) -> int:
        base = self._seg.staging_addr(0)
        off = addr - base
        idx, rem = divmod(off, self._seg.block_bytes)
        if idx < 0 or idx >= self._seg.n_staging or rem != 0:
            raise RuntimeError(
                f"send addr {addr:#x} not staging-block aligned "
                f"(base={base:#x} blk={self._seg.block_bytes})")
        return idx

    def _wake_daemon(self) -> None:
        try:
            os.write(self._wake_daemon_w, b"\x01")
        except (BlockingIOError, OSError):
            pass

    def _cmd(self, kind: int, **kw: Any) -> None:
        """cmd 环生产者唯一入口(锁内 put+唤醒,防多线程并发覆盖)。"""
        with self._cmd_lock:
            self._seg.cmd_put(kind, **kw)
            self._wake_daemon()

    def _dispatch_loop(self) -> None:
        """cpl 分发 + recv 块回收(dispatcher 单线程,cpl FIFO 保序)。"""
        while not self._stopped:
            ready, _, _ = select.select([self._wake_engine_r], [], [], 0.05)
            if ready:
                try:
                    if not os.read(self._wake_engine_r, 64):
                        self._stopped = True   # daemon 侧会话关停(EOF)
                        return
                except OSError:
                    self._stopped = True
                    return
            while True:
                cpl = self._seg.cpl_get()
                if cpl is None:
                    break
                try:
                    self._on_cpl(*cpl)
                except Exception:  # noqa: BLE001
                    # 单条 cpl 处理异常不得杀死 dispatcher(否则环停摆,
                    # 整条通道假死);记录并继续
                    logger.error("[ucx-port] cpl handle error", exc_info=True)
                if self._stopped:
                    return
            try:
                self._reap_recycle()
            except Exception:  # noqa: BLE001
                logger.error("[ucx-port] recycle error", exc_info=True)

    def _on_cpl(self, kind: int, _r: int, op_id: int, size: int,
                err: int, block_idx: int) -> None:
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
            if op is None:   # 已废弃(超时/关停):直接归还块
                self._release_recv(block_idx)
                return
            op.block_idx = block_idx
            if err:
                op._finish(err=err)
                self._release_recv(block_idx)
                return
            op._finish(size=size, data=self._seg.recv_view(block_idx, size))
            op._recycle_t0 = time.monotonic()
            self._pending_recycle.append(op)
        elif kind == _shm.CPL_READY:
            self._ready.set()
        elif kind == _shm.CPL_BROKEN:
            self._ready.clear()
        elif kind == _shm.CPL_CLOSED:
            self._stopped = True

    def _reap_recycle(self) -> None:
        # 先排水归还重试队列(环满等瞬态失败过的 release——漏一条即
        # 永久泄漏 recv 块,daemon 池耗尽后收方向停摆;真机实录)
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
                done = True    # 未登记设备事件的兜底:1s 后归还
            if done:
                if not self._try_release(op.block_idx):
                    # 环满等瞬态:留在 pending 下一轮重试
                    still.append(op)
                    logger.error("[ucx-port] recv release deferred "
                                 "(cmd ring full?) blk=%d pending=%d",
                                 op.block_idx, len(self._pending_recycle))
            else:
                still.append(op)
        self._pending_recycle[:] = still

    def _try_release(self, block_idx: int) -> bool:
        """尝试归还 recv 块;失败(环满等)返回 False,由调用方重试。"""
        if block_idx < 0 or self._seg is None or self._stopped:
            return True   # 无块可还(视为完成)
        try:
            self._cmd(_shm.CMD_RECV_RELEASE, block_idx=block_idx)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _release_recv(self, block_idx: int) -> None:
        """归还 recv 块(永不丢失:失败进 _rel_retry 由 dispatcher 重试)。"""
        if not self._try_release(block_idx):
            self._rel_retry.append(block_idx)
            logger.error("[ucx-port] recv release queued for retry "
                         "blk=%d backlog=%d", block_idx, len(self._rel_retry))
