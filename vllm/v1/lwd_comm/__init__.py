"""LWD 数据面(vllm 仓):host-host 传输库与数据面关停注册表。

现行 UCX 传输为 ucx_channel.py(UHC1 协议族,带 tag 的字节管道);
ucx_link.py 承载 UCP 链路生命周期,ucx_ucp.py/ucx_pool.py/sock.py 为
共享底层。设备拷贝(D2H/H2D)编排属调用方——NPU 实现在 vllm_ascend
(ucx_npu_channel.py),GPU/CPU 边侧按同接口在各自插件仓实现。
"""

from __future__ import annotations

import contextlib
from typing import Callable

# 数据面关停回调注册表:插件仓(如 vllm-ascend)在装配数据面时经
# register_data_plane_shutdown 注册自己的关停函数,引擎 shutdown 统一
# 经 shutdown_data_plane 调用——核心不反向 import 插件(此前引擎侧
# 懒加载 vllm_ascend.distributed.lwd_wire 属依赖方向违规)。
_DATA_PLANE_SHUTDOWN_HOOKS: list[Callable[[], None]] = []


def register_data_plane_shutdown(fn: Callable[[], None]) -> None:
    """登记数据面关停回调(幂等:同一函数只登记一次)。"""
    for hook in _DATA_PLANE_SHUTDOWN_HOOKS:
        if hook is fn:
            return
    _DATA_PLANE_SHUTDOWN_HOOKS.append(fn)


def shutdown_data_plane() -> None:
    """按登记序关停数据面(逐个 suppress:单点失败不阻断其余与引擎退出)。"""
    for fn in list(_DATA_PLANE_SHUTDOWN_HOOKS):
        with contextlib.suppress(Exception):
            fn()
