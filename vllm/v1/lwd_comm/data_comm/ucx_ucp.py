# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""libucp (UCX 1.22) ctypes bindings.

Targets UCX 1.22.0 (C:/cann/ucx wtt_v1.22.0 branch, consistent with the
official v1.22.0 tag). ucp_mem_destroy/ucp_mem_pack/ucp_mem_unpack and the
_nb family (tag_send/recv/put, ep_close, worker_get_address) were deprecated
and removed upstream before 1.22 (still present in 1.12; code relying on them
fails at startup with an undefined symbol). Everything therefore uses the
memh_pack family, the _nbx family and ucp_worker_query; signatures and struct
layouts were checked field by field against
src/ucp/api/{ucp.h,ucp_def.h} in that source tree.

This file is the binding layer only: library loading, status values and
constants, param and callback structs, argtypes/restype bindings (_bind_ucp
runs at import time), pointer-to-status conversion (_ptr_is_err/_ptr_status)
and _nbx request param construction (_req_param). Protocol, link and block
pool are not here.
"""

from __future__ import annotations

import ctypes
from typing import Any

# --------------------------------------------------------------------- #
# libucp ctypes bindings (UCX 1.22.0: C:/cann/ucx wtt_v1.22.0 branch,   #
# same as official tag v1.22.0). The memh_pack family + _nbx family +   #
# ucp_worker_query replace the pre-1.22 API that upstream removed       #
# (still present in 1.12; using it fails at startup with an undefined   #
# symbol). Signatures and struct layouts were checked field by field    #
# against src/ucp/api/{ucp.h,ucp_def.h}.                                #
# --------------------------------------------------------------------- #
_ucp: ctypes.CDLL | None
try:
    _ucp = ctypes.CDLL("libucp.so.0")
except OSError:
    try:
        _ucp = ctypes.CDLL("libucp.so")
    except OSError:
        _ucp = None

# ucp_init_version arguments, resolved once at bind time (1, 0 until then)
_UCP_INIT_MAJOR = 1
_UCP_INIT_MINOR = 0

# status values (ucs/type/status.h): errors are negative; UCS_ERR_LAST=-100
_UCS_OK = 0
_UCS_ERR_LAST = -100


def _ptr_is_err(p: int) -> bool:
    return p >= (1 << 64) + _UCS_ERR_LAST  # (uintptr_t)(-100)..(uintptr_t)(-1)


def _ptr_status(p: int) -> int:
    return p - (1 << 64)  # back to a negative error code


_UCP_FEATURE_TAG = 1 << 0
_UCP_FEATURE_RMA = 1 << 1
_UCP_PARAM_FIELD_FEATURES = 1 << 0
_UCP_EP_PARAM_FIELD_REMOTE_ADDRESS = 1 << 0
_UCP_EP_PARAM_FIELD_ERR_HANDLER = 1 << 2
_UCP_MEM_PARAM_FIELD_ADDRESS = 1 << 0
_UCP_MEM_PARAM_FIELD_LENGTH = 1 << 1
# _nbx request params (ucp_request_param_t): callback mode sets only the
# CALLBACK bit of op_attr_mask; datatype is left unset (the library then
# defaults to ucp_dt_make_contig(1), so count means bytes)
_UCP_OP_ATTR_FIELD_CALLBACK = 1 << 1
_UCP_EP_CLOSE_FLAG_FORCE = 1 << 0  # ep_close_nbx: passed via param.flags
_UCP_WORKER_ATTR_FIELD_ADDRESS = 1 << 1  # worker_query returns the local address
_UCP_MEMH_PACK_PARAM_FIELD_FLAGS = 1 << 0


class _UcpParams(ctypes.Structure):
    # only field_mask/features are set (offset 0/8); the other fields are
    # masked off and never read, and the extra 128B tail absorbs version
    # differences
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
    # field_mask=0 means all defaults (SINGLE thread mode); the worker is
    # created by the thread that establishes the link and then handed to the
    # IO thread for exclusive use (strict hand-off, no concurrency)
    _fields_ = [
        ("field_mask", ctypes.c_uint64),
        ("_tail", ctypes.c_uint8 * 256),
    ]


class _UcpEpParams(ctypes.Structure):
    # deployed layout (ucp_def.h ucp_ep_params_t): no address_length after
    # address (the address is self-describing); err_handler is an embedded
    # {cb,arg} struct, expanded here into two pointer fields
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
    # matches the deployed ucp_mem_map_params_t: field_mask/address/length/
    # flags/prot/exported_memh_buffer (active via the mask; a 64B tail
    # absorbs version differences)
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
    # deployed ucp_memh_pack_params_t: {field_mask, flags}
    _fields_ = [
        ("field_mask", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
    ]


class _MemhBufReleaseParams(ctypes.Structure):
    # deployed ucp_memh_buffer_release_params_t: {field_mask}
    _fields_ = [("field_mask", ctypes.c_uint64)]


class _UcpRequestParam(ctypes.Structure):
    # deployed ucp_request_param_t: op_attr_mask/flags (u32+u32), request,
    # the cb union (one pointer slot, shared by the send/recv callbacks),
    # datatype, user_data, ... Only callback mode uses the first two fields
    # plus cb; the tail covers the union's rear half and the total length
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
    # deployed ucp_worker_attr_t: with FIELD_ADDRESS the library fills in
    # address/address_length; the address is library-allocated, so after
    # copying it must be returned via
    # ucp_worker_release_address(worker, attr.address)
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
    """Build callback-mode _nbx request params (cb is a CFUNCTYPE instance)."""
    p = _UcpRequestParam()
    p.op_attr_mask = _UCP_OP_ATTR_FIELD_CALLBACK
    p.cb = ctypes.cast(cb, ctypes.c_void_p)
    return p


# callback prototypes (must be held at module level to prevent GC). _nbx
# callbacks take one extra trailing user_data argument (from
# request_param.user_data; always NULL when that field is not set)
_SendCb = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
_RecvCb = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(_RecvInfo), ctypes.c_void_p
)
_ErrCb = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)


def _bind_ucp() -> None:
    global _UCP_INIT_MAJOR, _UCP_INIT_MINOR

    ucp = _ucp
    if ucp is None:
        return
    # ucp_init is a header static inline wrapper (calls ucp_init_version
    # with UCP_API_VERSION) and libucp does not export it -- bind
    # ucp_init_version directly. The version is not hardcoded: the loaded
    # library reports major/minor via ucp_get_version (compatible when the
    # major matches and the library minor >= the requested value; a mismatch
    # only warns, it does not fail)
    ucp.ucp_init_version.argtypes = [
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.POINTER(_UcpParams),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    ucp.ucp_init_version.restype = ctypes.c_int
    ucp.ucp_get_version.argtypes = [
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
    ]
    maj, mnr, rel = ctypes.c_uint(0), ctypes.c_uint(0), ctypes.c_uint(0)
    ucp.ucp_get_version(ctypes.byref(maj), ctypes.byref(mnr), ctypes.byref(rel))
    _UCP_INIT_MAJOR = maj.value or 1
    _UCP_INIT_MINOR = mnr.value
    # ucp_config_*: rewrite the UCX config from code (keys carry no UCX_
    # prefix; unrecognized keys pass through the empty UCP prefix into the
    # subtable cache and are sent to UCT when resources open, e.g.
    # IB_TRAFFIC_CLASS)
    ucp.ucp_config_read.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    ucp.ucp_config_read.restype = ctypes.c_int
    ucp.ucp_config_modify.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]
    ucp.ucp_config_modify.restype = ctypes.c_int
    ucp.ucp_config_release.argtypes = [ctypes.c_void_p]
    ucp.ucp_config_release.restype = None
    ucp.ucp_cleanup.argtypes = [ctypes.c_void_p]
    ucp.ucp_worker_create.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_UcpWorkerParams),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    ucp.ucp_worker_create.restype = ctypes.c_int
    ucp.ucp_worker_destroy.argtypes = [ctypes.c_void_p]
    # worker address: ucp_worker_get_address was removed -- use
    # ucp_worker_query + FIELD_ADDRESS. attr.address is library-allocated
    # and must be released with ucp_worker_release_address after copying
    # (it would otherwise leak on every reconnect loop)
    ucp.ucp_worker_query.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_UcpWorkerAttr),
    ]
    ucp.ucp_worker_query.restype = ctypes.c_int
    ucp.ucp_worker_release_address.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    ucp.ucp_ep_create.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_UcpEpParams),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    ucp.ucp_ep_create.restype = ctypes.c_int
    ucp.ucp_ep_close_nbx.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_UcpRequestParam),
    ]
    ucp.ucp_ep_close_nbx.restype = ctypes.c_void_p
    ucp.ucp_tag_send_nbx.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.POINTER(_UcpRequestParam),
    ]
    ucp.ucp_tag_send_nbx.restype = ctypes.c_void_p
    ucp.ucp_tag_recv_nbx.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(_UcpRequestParam),
    ]
    ucp.ucp_tag_recv_nbx.restype = ctypes.c_void_p
    ucp.ucp_request_free.argtypes = [ctypes.c_void_p]
    ucp.ucp_worker_progress.argtypes = [ctypes.c_void_p]
    ucp.ucp_worker_progress.restype = ctypes.c_uint
    # RMA put path: local mem_map + memh_pack (the blob must be freed with
    # ucp_memh_buffer_release, never libc free); the peer runs
    # ep_rkey_unpack to get an rkey usable with its own ep
    ucp.ucp_mem_map.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_UcpMemParams),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    ucp.ucp_mem_map.restype = ctypes.c_int
    ucp.ucp_mem_unmap.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    ucp.ucp_memh_pack.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_MemhPackParams),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    ucp.ucp_memh_pack.restype = ctypes.c_int
    ucp.ucp_memh_buffer_release.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_MemhBufReleaseParams),
    ]
    ucp.ucp_ep_rkey_unpack.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    ucp.ucp_ep_rkey_unpack.restype = ctypes.c_int
    ucp.ucp_rkey_destroy.argtypes = [ctypes.c_void_p]
    ucp.ucp_put_nbx.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.POINTER(_UcpRequestParam),
    ]
    ucp.ucp_put_nbx.restype = ctypes.c_void_p


_bind_ucp()


def _require_ucp() -> ctypes.CDLL:
    """The libucp handle, or a startup error naming the runtime requirement."""
    if _ucp is None:
        raise RuntimeError(
            "libucp.so.0 not found: ucx data plane requires UCX runtime "
            "(set LD_LIBRARY_PATH to the UCX install lib directory)"
        )
    return _ucp
