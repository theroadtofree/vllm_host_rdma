"""边侧到云 DPCoordinator 的客户端:FIRST_REQ 唤醒上行 + stats 订阅对账。

仅多云内 DP(云侧 HELLO 通告 coord_stats_addr 非空,或 registry 声明
coord_port)时由边引擎创建;单引擎无 coordinator,不建。

XSUB 单 socket 双向复用(与原生 DPAsyncMPClient 同机制):上行
FIRST_REQ 经订阅通道的反向消息到达 coordinator 的 XPUB;下行 stats
三元组 (counts, wave, engines_running) 仅作 bring-up 对账日志——路由
决策以云侧中央控制面(probe 回执)为权威,不消费 stats。
"""

from __future__ import annotations

import threading

import msgspec
import zmq

from vllm.logger import init_logger

logger = init_logger(__name__)


class LwdCoordClient:
    """coordinator 客户端;close 幂等,关停后 send 静默丢弃。"""

    def __init__(self, coord_stats_addr: str) -> None:
        self._closed = False
        # zmq socket 非线程安全:stats 线程 recv 与主线程 send_first_req
        # 经本锁串行;recv 循环用短 poll 让出锁,FIRST_REQ 最多等一个
        # poll 周期。
        self._lock = threading.Lock()
        self._socket = zmq.Context.instance().socket(zmq.XSUB)
        self._socket.connect(coord_stats_addr)
        with self._lock:
            self._socket.send(b"\x01")  # subscribe-all
        self._thread = threading.Thread(
            target=self._stats_loop, name="lwd-coord-stats", daemon=True
        )
        self._thread.start()
        logger.info("[Lwd] coord client connected: %s", coord_stats_addr)

    def send_first_req(self, engine_index: int) -> None:
        """上行 FIRST_REQ(engine_index, wave=0):coordinator 在 engines
        暂停时广播 START_DP_WAVE 唤醒全波,运行中被守卫吞掉。wave 填 0
        是安全退化——coordinator 发现陈旧即改向全体广播
        (coordinator.py 的 wave < current_wave 分支)。"""
        if self._closed:
            return
        # 阻塞拿锁(与 1E1C legacy 分支一致):FIRST_REQ 是 LWD 云内 DP
        # 的唯一唤醒路径——LWD 请求以 request_wave=0 注入,引擎
        # add_request 的「request_wave != current_wave」自醒分支不触发,
        # 没有「引擎自醒兜底」可丢。带超时拿锁一旦在 stats 线程持锁
        # poll(50ms) 期间竞争失败,即静默丢唤醒 → 对侧引擎不醒 →
        # DP 锁步死锁。阻塞 acquire 最多等一个 poll 周期(50ms),可接受。
        try:
            with self._lock:
                self._socket.send(
                    msgspec.msgpack.encode((engine_index, 0)), flags=zmq.NOBLOCK
                )
        except zmq.ZMQError:
            logger.warning(
                "[Lwd] FIRST_REQ to coordinator dropped", exc_info=True
            )
            return
        # 唤醒链排障锚点:引擎入睡后无此日志 = FIRST_REQ 未发出
        # (hook 未接线);有日志但云侧无 wave 动静 = coordinator
        # 地址/端口错(coord_port 不一致)
        logger.info(
            "[Lwd] FIRST_REQ sent to coordinator: engine=%d", engine_index
        )

    def _stats_loop(self) -> None:
        """stats 下行(仅对账日志):counts 是各 engine 的 [waiting, running],
        与边侧本地台账的手工比对锚点。change-only 打印:stats 每步发布,
        逐条 info 会刷屏淹没关键日志;wave/engines_running 翻转(入睡/
        唤醒)与负载变化是排障关心的状态迁移,才值得落日志。"""
        last = None
        while not self._closed:
            try:
                with self._lock:
                    if not self._socket.poll(50):
                        continue
                    buf = self._socket.recv(zmq.NOBLOCK)
            except zmq.ZMQError:
                if self._closed:
                    return
                continue
            try:
                counts, wave, running = msgspec.msgpack.decode(buf)
            except (msgspec.DecodeError, TypeError, ValueError):
                continue
            cur = (tuple(map(tuple, counts)) if isinstance(counts, list)
                   else counts, wave, running)
            if cur == last:
                continue
            last = cur
            logger.info(
                "[Lwd][coord] stats counts=%s wave=%s engines_running=%s",
                counts, wave, running,
            )

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            self._socket.close(0)
        self._thread.join(timeout=1.0)
