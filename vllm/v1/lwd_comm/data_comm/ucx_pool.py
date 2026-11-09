# SPDX-License-Identifier: Apache-2.0
"""LWD UCX 注册缓冲池(wire 数据面块池的独立组件)。

从 legacy ucx_wire.py(已删除)原样抽出(纯重构,零行为变化):
  * Block     —— 一块接收/发送缓冲:普通 bytearray,UCP rcache 首次
                 收发触碰该地址时隐式注册并缓存(块池常驻,rcache 条目
                 恒为每块一条;分档后同理);
  * BlockPool —— 单尺寸等大块池:空闲队列周转 + 换链整体复位。

wire 层语义保持不变:
  * 块的分配在 wire 构造期一次性完成;空闲池初始为空,首次 adopt/
    reset 时整体灌入(与原实现逐行为一致);
  * acquire 的等待/progress 驱动策略仍属 wire(_acquire_block);
    池只提供带超时的非阻塞获取,超时返回 None;
  * reset 语义 = 换链时全部块重灌(旧链在途状态由 wire 的
    _fail_all_pending 清算,此处只管缓冲周转);
  * 内存代价 = n_blocks × block_bytes / wire。

线程约束沿用 wire 现状:acquire/release/reset 仅由所属 wire 的
IO 线程执行(构造期除外),空闲队列免锁是安全的。

后续分档扩展点(仅注释,不在本文件实现;见方案分析):
  * 发送侧:发送方预知帧大小,acquire 可按 size 选最小够用档——
    档尺寸建议 2MB 倍数(页对齐,rcache 区域合并干净);
  * 接收侧:wildcard 预挂要求每个挂出的块 ≥ 最大可能帧(否则截断
    断链),分档必须换交付模型——tag 高位档位路由(收方按档预挂)
    或 put 槽位逐档发环(GPU 版 recv_sizes[] 为现成模板);
  * 定档依据:sendlat 日志 LARGE/SMALL 桶的 nbytes 分布,而非拍定。
"""

from __future__ import annotations

import ctypes
import queue
from typing import Any

# --------------------------------------------------------------------- #
# 块池默认配置(wire 构造参数缺省值;装配层可用构造参数覆盖,env 解析在  #
# 装配层——与 legacy ucx_wire 既有口径一致)                                    #
# --------------------------------------------------------------------- #
_DEFAULT_BLOCK_MB = 128
_DEFAULT_N_BLOCKS = 4


class Block:
    """一块接收/发送缓冲(普通 bytearray;UCP rcache 首用注册并缓存)。"""

    __slots__ = ("buf", "addr", "index", "future", "event", "recv_req",
                 "send_req", "t_post", "t_enq", "t_disp", "nbytes")

    def __init__(self, index: int, block_bytes: int) -> None:
        self.buf = bytearray(block_bytes)
        self.addr = ctypes.addressof(
            (ctypes.c_char * block_bytes).from_buffer(self.buf))
        self.index = index
        self.future: Any = None
        self.event: Any = None
        self.recv_req = 0
        self.send_req = 0
        self.t_post = 0.0
        self.t_enq = 0.0   # 入队时刻(submit_send 提交线程记)
        self.t_disp = 0.0  # 分发开始时刻(IO 线程)
        self.nbytes = 0    # 帧字节数(延迟采样用)


class BlockPool:
    """单尺寸等大块池:IO 线程私有(免锁),空闲队列周转。

    n_blocks/block_bytes 与原 wire 构造参数同义;块在构造期分配,
    空闲池初始为空、由 reset() 灌入(保持原实现的时序语义)。
    """

    def __init__(self, n_blocks: int, block_bytes: int) -> None:
        self.block_bytes = block_bytes
        self.blocks: list[Block] = [
            Block(i, block_bytes) for i in range(n_blocks)]
        self._free: queue.Queue[Block] = queue.Queue()

    def __len__(self) -> int:
        return len(self.blocks)

    def __iter__(self):
        return iter(self.blocks)

    def qsize(self) -> int:
        """空闲块数(stall 诊断口径:qsize/在途/总数)。"""
        return self._free.qsize()

    def get(self, timeout: float) -> Block | None:
        """取一块空闲缓冲;超时返回 None(等待策略由调用方负责)。"""
        try:
            return self._free.get(timeout=timeout)
        except queue.Empty:
            return None

    def release(self, blk: Block) -> None:
        """归还:发送完成回调/失败路径/换链兜底统一入口。"""
        self._free.put(blk)

    def reset(self) -> None:
        """换链复位:全部块重灌空闲池(块本身跨链复用,不重新分配)。"""
        self._free = queue.Queue()
        for blk in self.blocks:
            self._free.put(blk)
