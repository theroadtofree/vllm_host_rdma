# SPDX-License-Identifier: Apache-2.0
"""libucp(UCX 1.22)ctypes 绑定——从 legacy ucx_wire.py(已删除)原样抽出(纯重构)。

对齐 UCX 1.22.0(C:/cann/ucx wtt_v1.22.0 分支,与官方 tag v1.22.0
一致)。ucp_mem_destroy/ucp_mem_pack/ucp_mem_unpack 与 _nb 系
(tag_send/recv/put、ep_close、worker_get_address)系官方上游在 1.22 前
废弃移除(1.12 尚存——旧代码据此误写,启动报 undefined symbol 的根因),
统一为 memh_pack 系 + _nbx 系 + ucp_worker_query;签名/结构体布局逐字
段核对该源码 src/ucp/api/{ucp.h,ucp_def.h})。

本文件只含绑定层:库加载、状态值/常量、参数与回调结构体、argtypes/
restype 绑定(_bind_ucp 在 import 时执行,与原在 legacy ucx_wire 导入期执行
等价)、指针-状态换算(_ptr_is_err/_ptr_status)与 _nbx 请求参数构造
(_req_param)。协议/链路/块池不在本文件。
"""

from __future__ import annotations

import ctypes
from typing import Any


# --------------------------------------------------------------------- #
# libucp ctypes 绑定(对齐 UCX 1.22.0:C:\cann\ucx wtt_v1.22.0 分支,    #
# 与官方 tag v1.22.0 一致。ucp_mem_destroy/ucp_mem_pack/ucp_mem_unpack  #
# 与 _nb 系(tag_send/recv/put、ep_close、worker_get_address)系官方    #
# 上游在 1.22 前废弃移除(1.12 尚存——旧代码据此误写,启动报          #
# undefined symbol 的根因),统一为 memh_pack 系 + _nbx 系 +            #
# ucp_worker_query;签名/结构体布局逐字段核对该源码                    #
# src/ucp/api/{ucp.h,ucp_def.h})                                      #
# --------------------------------------------------------------------- #
try:
    _ucp = ctypes.CDLL("libucp.so.0")
except OSError:
    try:
        _ucp = ctypes.CDLL("libucp.so")
    except OSError:
        _ucp = None

# 状态值(ucs/type/status.h):错误为负;UCS_ERR_LAST=-100
_UCS_OK = 0
_UCS_ERR_LAST = -100


def _ptr_is_err(p: int) -> bool:
    return p >= (1 << 64) + _UCS_ERR_LAST  # (uintptr_t)(-100)..(uintptr_t)(-1)


def _ptr_status(p: int) -> int:
    return p - (1 << 64)  # 还原为负数错误码


_UCP_FEATURE_TAG = 1 << 0
_UCP_FEATURE_RMA = 1 << 1
_UCP_PARAM_FIELD_FEATURES = 1 << 0
_UCP_EP_PARAM_FIELD_REMOTE_ADDRESS = 1 << 0
_UCP_EP_PARAM_FIELD_ERR_HANDLER = 1 << 2
_UCP_MEM_PARAM_FIELD_ADDRESS = 1 << 0
_UCP_MEM_PARAM_FIELD_LENGTH = 1 << 1
# _nbx 请求参数(ucp_request_param_t):callback 模式只置 op_attr_mask
# 的 CALLBACK 位,datatype 不置位(库默认 ucp_dt_make_contig(1),
# count 语义=字节数,与旧 _nb 调用一致)
_UCP_OP_ATTR_FIELD_CALLBACK = 1 << 1
_UCP_EP_CLOSE_FLAG_FORCE = 1 << 0   # ep_close_nbx:经 param.flags 传递
_UCP_WORKER_ATTR_FIELD_ADDRESS = 1 << 1  # worker_query 换取本端地址
_UCP_MEMH_PACK_PARAM_FIELD_FLAGS = 1 << 0


class _UcpParams(ctypes.Structure):
    # 只设置 field_mask/features(offset 0/8);其余字段经掩码屏蔽不被
    # 读,尾部多垫 128B 覆盖版本差异
    _fields_ = [
        ("field_mask", ctypes.c_uint64),
        ("features", ctypes.c_uint64),
        ("request_size", ctypes.c_size_t),
        ("request_init", ctypes.c_void_p),
        ("request_cleanup", ctypes.c_void_p),
        ("tag_sender_mask", ctypes.c_uint64),
        ("_tail", ctypes.c_uint8 * 128),
    ]


class _UcpWorkerParams(ctypes.Structure):
    # field_mask=0 全默认(SINGLE 线程模式);worker 由建连线程创建后
    # 移交 IO 线程独占使用(严格先后,无并发)
    _fields_ = [
        ("field_mask", ctypes.c_uint64),
        ("_tail", ctypes.c_uint8 * 256),
    ]


class _UcpEpParams(ctypes.Structure):
    # 部署版布局(ucp_def.h ucp_ep_params_t):address 后无 address_length
    # (地址自描述);err_handler 为内嵌 {cb,arg} 结构体,展开为两个指针字段
    _fields_ = [
        ("field_mask", ctypes.c_uint64),
        ("address", ctypes.c_void_p),
        ("err_mode", ctypes.c_int),
        ("_pad0", ctypes.c_int),
        ("err_handler_cb", ctypes.c_void_p),
        ("err_handler_arg", ctypes.c_void_p),
        ("user_data", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("_pad1", ctypes.c_uint32),
        ("sockaddr_addr", ctypes.c_void_p),
        ("sockaddr_len", ctypes.c_uint32),
        ("_pad2", ctypes.c_uint32),
        ("conn_request", ctypes.c_void_p),
        ("name", ctypes.c_void_p),
        ("local_sockaddr_addr", ctypes.c_void_p),
        ("local_sockaddr_len", ctypes.c_uint32),
        ("_pad3", ctypes.c_uint32),
    ]


class _RecvInfo(ctypes.Structure):
    _fields_ = [
        ("sender_tag", ctypes.c_uint64),
        ("length", ctypes.c_size_t),
    ]


class _UcpMemParams(ctypes.Structure):
    # 对齐部署版 ucp_mem_map_params_t:field_mask/address/length/flags/
    # prot/exported_memh_buffer(经掩码生效,尾部垫 64B 容忍版本差异)
    _fields_ = [
        ("field_mask", ctypes.c_uint64),
        ("address", ctypes.c_void_p),
        ("length", ctypes.c_size_t),
        ("flags", ctypes.c_uint),
        ("prot", ctypes.c_uint),
        ("exported_memh_buffer", ctypes.c_void_p),
        ("_tail", ctypes.c_uint8 * 64),
    ]


class _MemhPackParams(ctypes.Structure):
    # 部署版 ucp_memh_pack_params_t:{field_mask, flags}
    _fields_ = [
        ("field_mask", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
    ]


class _MemhBufReleaseParams(ctypes.Structure):
    # 部署版 ucp_memh_buffer_release_params_t:{field_mask}
    _fields_ = [("field_mask", ctypes.c_uint64)]


class _UcpRequestParam(ctypes.Structure):
    # 部署版 ucp_request_param_t:op_attr_mask/flags(u32+u32)、request、
    # cb 联合体(单指针槽,send/recv 回调同槽)、datatype、user_data…
    # 仅 callback 模式用到前两段+cb;尾部垫足覆盖联合体后半/长度
    _fields_ = [
        ("op_attr_mask", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("request", ctypes.c_void_p),
        ("cb", ctypes.c_void_p),
        ("datatype", ctypes.c_uint64),
        ("user_data", ctypes.c_void_p),
        ("_tail", ctypes.c_uint8 * 56),
    ]


class _UcpWorkerAttr(ctypes.Structure):
    # 部署版 ucp_worker_attr_t:FIELD_ADDRESS 时由库填
    # address/address_length;地址由库分配,拷贝后须
    # ucp_worker_release_address(worker, attr.address) 归还
    _fields_ = [
        ("field_mask", ctypes.c_uint64),
        ("address_flags", ctypes.c_uint32),
        ("_pad0", ctypes.c_uint32),
        ("address", ctypes.c_void_p),
        ("address_length", ctypes.c_size_t),
        ("max_am_header", ctypes.c_size_t),
        ("max_debug_string", ctypes.c_size_t),
    ]


def _req_param(cb: Any) -> _UcpRequestParam:
    """构造 callback 模式的 _nbx 请求参数(cb 为 CFUNCTYPE 实例)。"""
    p = _UcpRequestParam()
    p.op_attr_mask = _UCP_OP_ATTR_FIELD_CALLBACK
    p.cb = ctypes.cast(cb, ctypes.c_void_p)
    return p


# 回调原型(必须模块级持有引用,防 GC)。_nbx 回调比旧 _nb 多一个
# 尾参 user_data(取自 request_param.user_data,不置位时恒 NULL)
_SendCb = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int,
                           ctypes.c_void_p)
_RecvCb = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int,
                           ctypes.POINTER(_RecvInfo), ctypes.c_void_p)
_ErrCb = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_int)


def _bind_ucp() -> None:
    if _ucp is None:
        return
    # ucp_init 是头文件 static inline 包装(经 UCP_API_VERSION 调
    # ucp_init_version),libucp 不导出——直绑 ucp_init_version;版本
    # 不硬编码:绑库自带 ucp_get_version 取 major/minor 回传(同 major
    # 且库 minor>=请求值即兼容,不匹配也仅 WARN 不失败)
    _ucp.ucp_init_version.argtypes = [
        ctypes.c_uint, ctypes.c_uint,
        ctypes.POINTER(_UcpParams), ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _ucp.ucp_init_version.restype = ctypes.c_int
    _ucp.ucp_get_version.argtypes = [
        ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
    ]
    maj, mnr, rel = ctypes.c_uint(0), ctypes.c_uint(0), ctypes.c_uint(0)
    _ucp.ucp_get_version(ctypes.byref(maj), ctypes.byref(mnr),
                         ctypes.byref(rel))
    _ucp._ucp_init_major = maj.value or 1
    _ucp._ucp_init_minor = mnr.value
    _ucp.ucp_cleanup.argtypes = [ctypes.c_void_p]
    _ucp.ucp_worker_create.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_UcpWorkerParams),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _ucp.ucp_worker_create.restype = ctypes.c_int
    _ucp.ucp_worker_destroy.argtypes = [ctypes.c_void_p]
    # worker 地址:旧 ucp_worker_get_address 已移除,经 ucp_worker_query
    # + FIELD_ADDRESS 取;attr.address 由库分配,拷贝后必须以
    # ucp_worker_release_address 归还(重连循环下否则泄漏)
    _ucp.ucp_worker_query.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_UcpWorkerAttr),
    ]
    _ucp.ucp_worker_query.restype = ctypes.c_int
    _ucp.ucp_worker_release_address.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    _ucp.ucp_ep_create.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_UcpEpParams),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _ucp.ucp_ep_create.restype = ctypes.c_int
    _ucp.ucp_ep_close_nbx.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_UcpRequestParam),
    ]
    _ucp.ucp_ep_close_nbx.restype = ctypes.c_void_p
    _ucp.ucp_tag_send_nbx.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64,
        ctypes.POINTER(_UcpRequestParam),
    ]
    _ucp.ucp_tag_send_nbx.restype = ctypes.c_void_p
    _ucp.ucp_tag_recv_nbx.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64,
        ctypes.c_uint64, ctypes.POINTER(_UcpRequestParam),
    ]
    _ucp.ucp_tag_recv_nbx.restype = ctypes.c_void_p
    _ucp.ucp_request_free.argtypes = [ctypes.c_void_p]
    _ucp.ucp_worker_progress.argtypes = [ctypes.c_void_p]
    _ucp.ucp_worker_progress.restype = ctypes.c_uint
    # RMA put 路径:本端 mem_map+memh_pack(blob 须 ucp_memh_buffer_release
    # 释放,不能 libc free);对端 ep_rkey_unpack 出按本端 ep 可用的 rkey
    _ucp.ucp_mem_map.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_UcpMemParams),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _ucp.ucp_mem_map.restype = ctypes.c_int
    _ucp.ucp_mem_unmap.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _ucp.ucp_memh_pack.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_MemhPackParams),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t),
    ]
    _ucp.ucp_memh_pack.restype = ctypes.c_int
    _ucp.ucp_memh_buffer_release.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_MemhBufReleaseParams),
    ]
    _ucp.ucp_ep_rkey_unpack.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _ucp.ucp_ep_rkey_unpack.restype = ctypes.c_int
    _ucp.ucp_rkey_destroy.argtypes = [ctypes.c_void_p]
    _ucp.ucp_put_nbx.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_uint64, ctypes.c_void_p,
        ctypes.POINTER(_UcpRequestParam),
    ]
    _ucp.ucp_put_nbx.restype = ctypes.c_void_p


_bind_ucp()


def _require_ucp() -> None:
    if _ucp is None:
        raise RuntimeError(
            "libucp.so.0 not found: ucx data plane requires UCX runtime "
            "(LD_LIBRARY_PATH 指向 UCX 安装的 lib 目录)")
