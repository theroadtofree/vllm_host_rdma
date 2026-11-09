# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""UCP 链路层:一条链路 = context + worker + ep + put 槽位能力。

从 legacy ucx_wire.py(已删除)原样抽出——UHC1(ucx_channel)经
_ChanLink 子类复用:链路生命周期(ucp_init → worker_create → 握手
交换地址 → ep_create 挂 err handler → progress → close)、本端接收
块池 mem_map/rkey 打包(put 能力凭证)、对端槽位能力解开。

worker 自创建后仅在"建连线程完成移交、IO 线程独占"约束下使用。
"""

from __future__ import annotations

import contextlib
import ctypes
import struct
import threading
from typing import Any

from vllm.v1.data_comm.ucx_ucp import (
    _ErrCb,
    _MemhBufReleaseParams,
    _MemhPackParams,
    _UCP_EP_CLOSE_FLAG_FORCE,
    _UCP_EP_PARAM_FIELD_ERR_HANDLER,
    _UCP_EP_PARAM_FIELD_REMOTE_ADDRESS,
    _UCP_FEATURE_RMA,
    _UCP_FEATURE_TAG,
    _UCP_MEMH_PACK_PARAM_FIELD_FLAGS,
    _UCP_MEM_PARAM_FIELD_ADDRESS,
    _UCP_MEM_PARAM_FIELD_LENGTH,
    _UCP_PARAM_FIELD_FEATURES,
    _UCP_WORKER_ATTR_FIELD_ADDRESS,
    _UCS_OK,
    _UcpEpParams,
    _UcpMemParams,
    _UcpParams,
    _UcpRequestParam,
    _UcpWorkerAttr,
    _UcpWorkerParams,
    _req_param,
    _require_ucp,
    _ucp,
)


# --------------------------------------------------------------------- #
# 握手能力段打包(put 槽位地址 + rkey blob)                             #
# --------------------------------------------------------------------- #
def _hs_pack_caps(slot_addrs: list[int], rkey_blobs: list[bytes]) -> bytes:
    """打包本端 put 接收能力:[u64 addr × n][u32 len + blob × n]。"""
    out = struct.pack(f"!{len(slot_addrs)}Q", *slot_addrs)
    for blob in rkey_blobs:
        out += struct.pack("!I", len(blob)) + blob
    return out


def _hs_unpack_caps(n: int, data: bytes) -> tuple[list[int], list[bytes]]:
    addrs = list(struct.unpack_from(f"!{n}Q", data, 0))
    off = 8 * n
    blobs: list[bytes] = []
    for _ in range(n):
        (blen,) = struct.unpack_from("!I", data, off)
        off += 4
        blobs.append(data[off:off + blen])
        off += blen
    return addrs, blobs


# --------------------------------------------------------------------- #
# UCP 资源(context/worker/ep)                                         #
# --------------------------------------------------------------------- #
class _UcxLink:
    """一条 UCP 链路:context + worker + ep;建连双方各持一份。

    生命周期:ucp_init → worker_create → 握手交换地址 → ep_create
    (挂 err handler)→ release 自身地址 → progress 收发 → close。
    worker 自创建后仅在"建连线程完成移交、IO 线程独占"约束下使用。
    """

    def __init__(self, owner: Any) -> None:
        _require_ucp()
        self._owner = owner
        self._ctx: int | None = None
        self._worker: int | None = None
        self._ep: int | None = None
        self._closed = False
        # 移交完成事件 + 对端 session(握手时填,adopt 日志用)
        self.adopted = threading.Event()
        self.peer_session = 0
        # put 能力:本端接收块池的 mem 句柄/packed blob;对端槽位表
        self._mem_hs: list[int] = []
        self.caps_n_slots = 0
        self.caps_blob = b""
        self.peer_slot_addrs: list[int] = []
        self.peer_slot_rkeys: list[int] = []
        self.peer_n_slots = 0
        self.peer_put_thresh = 0
        # 回调闭包必须长存(CFUNCTYPE 对象被 GC 后回调即悬空)
        self._err_cb = _ErrCb(self._on_ep_err)
        params = _UcpParams()
        params.field_mask = _UCP_PARAM_FIELD_FEATURES
        params.features = (_UCP_FEATURE_TAG
                           | (getattr(owner, "_put_thresh", 0) > 0
                              and _UCP_FEATURE_RMA or 0))
        ctx = ctypes.c_void_p(0)
        rc = _ucp.ucp_init_version(
            _ucp._ucp_init_major, _ucp._ucp_init_minor,
            ctypes.byref(params), None, ctypes.byref(ctx))
        if rc != _UCS_OK:
            raise RuntimeError(f"ucp_init_version failed status={rc}"
                               "(检查 UCX_TLS/UCX_NET_DEVICES 配置)")
        self._ctx = ctx.value
        try:
            wparams = _UcpWorkerParams()
            wparams.field_mask = 0
            worker = ctypes.c_void_p(0)
            rc = _ucp.ucp_worker_create(ctx.value,
                                        ctypes.byref(wparams),
                                        ctypes.byref(worker))
            if rc != _UCS_OK:
                raise RuntimeError(f"ucp_worker_create failed status={rc}")
            self._worker = worker.value
            if getattr(owner, "_put_thresh", 0) > 0:
                self._map_local_slots()
        except Exception:
            self.close()
            raise

    def _map_local_slots(self) -> None:
        """本端接收块池逐块 mem_map 并 pack(接收 put 的能力凭证)。
        块池字节串由通道持有、跨链稳定;map 归属本链 context,换链
        重做(旧链 close 时销毁)。"""
        blobs: list[bytes] = []
        for blk in self._owner._pool.blocks:
            mp = _UcpMemParams()
            mp.field_mask = (_UCP_MEM_PARAM_FIELD_ADDRESS
                             | _UCP_MEM_PARAM_FIELD_LENGTH)
            mp.address = ctypes.c_void_p(blk.addr)
            mp.length = self._owner._block_bytes
            memh = ctypes.c_void_p(0)
            rc = _ucp.ucp_mem_map(self._ctx, ctypes.byref(mp),
                                  ctypes.byref(memh))
            if rc != _UCS_OK:
                raise RuntimeError(f"ucp_mem_map failed status={rc}")
            self._mem_hs.append(memh.value)
            pp = _MemhPackParams()
            pp.field_mask = _UCP_MEMH_PACK_PARAM_FIELD_FLAGS
            pp.flags = 0  # 不置 EXPORT:按 RKEY 打包,对端 ep_rkey_unpack
            buf = ctypes.c_void_p(0)
            blen = ctypes.c_size_t(0)
            rc = _ucp.ucp_memh_pack(memh, ctypes.byref(pp),
                                    ctypes.byref(buf), ctypes.byref(blen))
            if rc != _UCS_OK:
                raise RuntimeError(f"ucp_memh_pack failed status={rc}")
            blobs.append(ctypes.string_at(buf.value, blen.value))
            rp = _MemhBufReleaseParams()
            rp.field_mask = 0
            _ucp.ucp_memh_buffer_release(buf, ctypes.byref(rp))
        self.caps_n_slots = len(self._owner._pool.blocks)
        self.caps_blob = _hs_pack_caps(
            [b.addr for b in self._owner._pool.blocks], blobs)

    def set_peer_caps(self, n_slots: int, put_thresh: int,
                      caps: bytes) -> None:
        """解开对端槽位能力:ep_rkey_unpack 出按本端 ep 可用的 rkey 列表
        (须在 create_ep 之后调用——rkey 绑定于本端到对端的 ep)。"""
        if n_slots <= 0:
            return
        addrs, blobs = _hs_unpack_caps(n_slots, caps)
        for blob in blobs:
            rbuf = ctypes.create_string_buffer(blob, len(blob))
            rkey = ctypes.c_void_p(0)
            rc = _ucp.ucp_ep_rkey_unpack(ctypes.c_void_p(self._ep), rbuf,
                                         ctypes.byref(rkey))
            if rc != _UCS_OK:
                raise RuntimeError(f"ucp_ep_rkey_unpack failed status={rc}")
            self.peer_slot_rkeys.append(rkey.value)
        self.peer_slot_addrs = addrs
        self.peer_n_slots = n_slots
        self.peer_put_thresh = put_thresh

    @property
    def worker(self) -> int:
        return self._worker

    def my_address(self) -> bytes:
        attr = _UcpWorkerAttr()
        attr.field_mask = _UCP_WORKER_ATTR_FIELD_ADDRESS
        rc = _ucp.ucp_worker_query(ctypes.c_void_p(self._worker),
                                   ctypes.byref(attr))
        if rc != _UCS_OK:
            raise RuntimeError(f"ucp_worker_query failed status={rc}")
        if not attr.address or not attr.address_length:
            raise RuntimeError("ucp_worker_query returned empty address")
        try:
            return ctypes.string_at(attr.address, attr.address_length)
        finally:
            _ucp.ucp_worker_release_address(
                ctypes.c_void_p(self._worker),
                ctypes.c_void_p(attr.address))

    def create_ep(self, peer_addr: bytes) -> None:
        buf = ctypes.create_string_buffer(peer_addr, len(peer_addr))
        p = _UcpEpParams()
        p.field_mask = (_UCP_EP_PARAM_FIELD_REMOTE_ADDRESS
                        | _UCP_EP_PARAM_FIELD_ERR_HANDLER)
        p.address = ctypes.cast(buf, ctypes.c_void_p)
        # PEER 模式:UCX keepalive 仅在该模式启用(ucp_ep.c: err_mode
        # == PEER 才检查心跳)——死对端在 ~UCX_KEEPALIVE_INTERVAL 内
        # 暴露 ep error,通道侧立即 broken 重连。NONE 模式下死链只能
        # 靠下次操作超时暴露(真机实录:云重启后旧 ep 挂死在途发送约
        # 60s,云侧 unbound 60s 先到、引擎崩)。配套两端 env:
        # UCX_KEEPALIVE_INTERVAL=5。回退 = 本行改回 0 并去掉 env。
        p.err_mode = 1  # UCP_ERR_HANDLING_MODE_PEER
        p.err_handler_cb = ctypes.cast(self._err_cb, ctypes.c_void_p)
        p.err_handler_arg = ctypes.c_void_p(id(self._owner))
        ep = ctypes.c_void_p(0)
        rc = _ucp.ucp_ep_create(self._worker, ctypes.byref(p),
                                ctypes.byref(ep))
        if rc != _UCS_OK:
            raise RuntimeError(
                f"ucp_ep_create failed status={rc}(对端地址不合法或 "
                f"TLS 组合不可达)")
        self._ep = ep.value  # 漏存会让 send_nb 持 NULL ep 直接段错误

    def progress(self, max_ops: int = 64) -> int:
        """驱动完成(回调在本调用内触发);返回推进的操作数。"""
        total = 0
        for _ in range(max_ops):
            n = _ucp.ucp_worker_progress(self._worker)
            total += n
            if n == 0:
                break
        return total

    def send_nb(self, addr: int, length: int, tag: int, cb: Any) -> int:
        """返回 None/0=立即完成(无回调);>0=request;错误见 _ptr_is_err"""
        if not self._ep:
            raise RuntimeError("ucx ep not created before send_nb")
        return _ucp.ucp_tag_send_nbx(
            ctypes.c_void_p(self._ep), ctypes.c_void_p(addr), length,
            tag, ctypes.byref(_req_param(cb)))

    def recv_nb(self, addr: int, length: int, cb: Any) -> int:
        """wildcard(tag=0/mask=0)预挂;返回 request 指针"""
        return _ucp.ucp_tag_recv_nbx(
            ctypes.c_void_p(self._worker), ctypes.c_void_p(addr), length,
            0, 0, ctypes.byref(_req_param(cb)))

    def put_nb(self, local_addr: int, length: int, remote_addr: int,
               rkey: int, cb: Any) -> int:
        """单边 WRITE(rc 下 CQE=数据已落对端);返回值语义同 send_nb。"""
        if not self._ep:
            raise RuntimeError("ucx ep not created before put_nb")
        return _ucp.ucp_put_nbx(
            ctypes.c_void_p(self._ep), ctypes.c_void_p(local_addr), length,
            remote_addr, ctypes.c_void_p(rkey),
            ctypes.byref(_req_param(cb)))

    def request_free(self, req: int) -> None:
        _ucp.ucp_request_free(ctypes.c_void_p(req))

    def _on_ep_err(self, arg, ep, status) -> None:  # noqa: ANN001
        # 基类默认路由到 owner 的 _on_ep_error;子类可覆盖
        handler = getattr(self._owner, "_on_ep_error", None)
        if handler is not None:
            handler(int(status))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for rkey in self.peer_slot_rkeys:
            with contextlib.suppress(Exception):
                _ucp.ucp_rkey_destroy(ctypes.c_void_p(rkey))
        self.peer_slot_rkeys = []
        if self._ep:
            with contextlib.suppress(Exception):
                # FORCE:不等待在途排空(shutdown 路径,IO 线程已停);
                # ep_close_nbx 的 flags 经 ucp_request_param_t.flags 传递
                cp = _UcpRequestParam()
                cp.flags = _UCP_EP_CLOSE_FLAG_FORCE
                _ucp.ucp_ep_close_nbx(ctypes.c_void_p(self._ep),
                                      ctypes.byref(cp))
            self._ep = None
        if self._worker:
            with contextlib.suppress(Exception):
                _ucp.ucp_worker_destroy(ctypes.c_void_p(self._worker))
            self._worker = None
        for memh in self._mem_hs:
            with contextlib.suppress(Exception):
                _ucp.ucp_mem_unmap(ctypes.c_void_p(self._ctx),
                                   ctypes.c_void_p(memh))
        self._mem_hs = []
        if self._ctx:
            with contextlib.suppress(Exception):
                _ucp.ucp_cleanup(ctypes.c_void_p(self._ctx))
            self._ctx = None
