# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UCP link layer: one link = context + worker + ep + put slot capabilities.

UHC1 (ucx_channel) reuses it through the _ChanLink subclass: link lifecycle
(ucp_init -> worker_create -> handshake address exchange -> ep_create with
err handler -> progress -> close), packing the local recv block pool as
mem_map/rkey (the put capability credential), and unpacking the peer's slot
capabilities.

Once created, the worker is only used under the rule that the connecting
thread hand-off completes first and the IO thread then owns it exclusively.
"""

from __future__ import annotations

import contextlib
import ctypes
import struct
import threading
from typing import Any

from vllm.v1.lwd_comm.data_comm.ucx_ucp import (
    _UCP_EP_CLOSE_FLAG_FORCE,
    _UCP_EP_PARAM_FIELD_ERR_HANDLER,
    _UCP_EP_PARAM_FIELD_REMOTE_ADDRESS,
    _UCP_FEATURE_RMA,
    _UCP_FEATURE_TAG,
    _UCP_INIT_MAJOR,
    _UCP_INIT_MINOR,
    _UCP_MEM_PARAM_FIELD_ADDRESS,
    _UCP_MEM_PARAM_FIELD_LENGTH,
    _UCP_MEMH_PACK_PARAM_FIELD_FLAGS,
    _UCP_PARAM_FIELD_FEATURES,
    _UCP_WORKER_ATTR_FIELD_ADDRESS,
    _UCS_OK,
    _ErrCb,
    _MemhBufReleaseParams,
    _MemhPackParams,
    _req_param,
    _require_ucp,
    _UcpEpParams,
    _UcpMemParams,
    _UcpParams,
    _UcpRequestParam,
    _UcpWorkerAttr,
    _UcpWorkerParams,
)


# --------------------------------------------------------------------- #
# Handshake capability packing (put slot addresses + rkey blobs)        #
# --------------------------------------------------------------------- #
def _hs_pack_caps(slot_addrs: list[int], rkey_blobs: list[bytes]) -> bytes:
    """Pack the local put receive capabilities: [u64 addr x n][u32 len + blob x n]."""
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
        blobs.append(data[off : off + blen])
        off += blen
    return addrs, blobs


# --------------------------------------------------------------------- #
# UCP resources (context/worker/ep)                                     #
# --------------------------------------------------------------------- #
def _ptr_value(ptr: ctypes.c_void_p) -> int:
    """Value of a ctypes out-pointer that libucp has just filled."""
    value = ptr.value
    if value is None:
        raise RuntimeError("libucp returned a null pointer")
    return value


class _UcxLink:
    """One UCP link: context + worker + ep; each peer holds one instance.

    Lifecycle: ucp_init -> worker_create -> handshake address exchange ->
    ep_create (with err handler) -> release own address -> progress both
    directions -> close. Once created, the worker is only used under the rule
    that the connecting thread hand-off completes first and the IO thread
    then owns it exclusively.
    """

    def __init__(self, owner: Any) -> None:
        _require_ucp()
        self._owner = owner
        self._ctx: int | None = None
        self._worker: int | None = None
        self._ep: int | None = None
        self._closed = False
        # Hand-off done event + peer session (set at handshake; used in the
        # adopt log)
        self.adopted = threading.Event()
        self.peer_session = 0
        # put capabilities: mem handles/packed blobs of the local recv block
        # pool; peer slot table
        self._mem_hs: list[int] = []
        self.caps_n_slots = 0
        self.caps_blob = b""
        self.peer_slot_addrs: list[int] = []
        self.peer_slot_rkeys: list[int] = []
        self.peer_n_slots = 0
        self.peer_put_thresh = 0
        # The callback closure must stay alive: a collected CFUNCTYPE object
        # leaves the callback dangling
        self._err_cb = _ErrCb(self._on_ep_err)
        params = _UcpParams()
        params.field_mask = _UCP_PARAM_FIELD_FEATURES
        params.features = _UCP_FEATURE_TAG | (
            getattr(owner, "_put_thresh", 0) > 0 and _UCP_FEATURE_RMA or 0
        )
        # UCT-level settings (e.g. IB_TRAFFIC_CLASS/IB_SL) are injected from
        # code: read (including env defaults) -> modify each item -> init ->
        # release. Keys carry no UCX_ prefix; keys absent from the UCP table
        # pass through the empty-prefix sub-table cache and are pushed down
        # to UCT when the resource opens.
        uct_cfg = getattr(owner, "_uct_config", None) or {}
        uconfig = ctypes.c_void_p(0)
        rc = _require_ucp().ucp_config_read(None, None, ctypes.byref(uconfig))
        if rc != _UCS_OK:
            raise RuntimeError(f"ucp_config_read failed status={rc}")
        try:
            for name, value in uct_cfg.items():
                rc = _require_ucp().ucp_config_modify(
                    uconfig, str(name).encode(), str(value).encode()
                )
                if rc != _UCS_OK:
                    raise RuntimeError(
                        f"ucp_config_modify({name}={value}) failed status={rc}"
                    )
            ctx = ctypes.c_void_p(0)
            rc = _require_ucp().ucp_init_version(
                _UCP_INIT_MAJOR,
                _UCP_INIT_MINOR,
                ctypes.byref(params),
                uconfig,
                ctypes.byref(ctx),
            )
            if rc != _UCS_OK:
                raise RuntimeError(
                    f"ucp_init_version failed status={rc}"
                    "(check the UCX_TLS/UCX_NET_DEVICES configuration)"
                )
        finally:
            _require_ucp().ucp_config_release(uconfig)
        self._ctx = ctx.value
        try:
            wparams = _UcpWorkerParams()
            wparams.field_mask = 0
            worker = ctypes.c_void_p(0)
            rc = _require_ucp().ucp_worker_create(
                ctx.value, ctypes.byref(wparams), ctypes.byref(worker)
            )
            if rc != _UCS_OK:
                raise RuntimeError(f"ucp_worker_create failed status={rc}")
            self._worker = worker.value
            if getattr(owner, "_put_thresh", 0) > 0:
                self._map_local_slots()
        except Exception:
            self.close()
            raise

    def _map_local_slots(self) -> None:
        """mem_map and pack each block of the local recv pool (the capability
        credential for receiving puts). The pool's byte buffers are owned by
        the channel and stay stable across links; the mapping belongs to this
        link's context and is redone on a link change (the old link destroys
        it on close)."""
        blobs: list[bytes] = []
        for blk in self._owner._pool.blocks:
            mp = _UcpMemParams()
            mp.field_mask = _UCP_MEM_PARAM_FIELD_ADDRESS | _UCP_MEM_PARAM_FIELD_LENGTH
            mp.address = ctypes.c_void_p(blk.addr)
            mp.length = self._owner._block_bytes
            memh = ctypes.c_void_p(0)
            rc = _require_ucp().ucp_mem_map(
                self._ctx, ctypes.byref(mp), ctypes.byref(memh)
            )
            if rc != _UCS_OK:
                raise RuntimeError(f"ucp_mem_map failed status={rc}")
            self._mem_hs.append(_ptr_value(memh))
            pp = _MemhPackParams()
            pp.field_mask = _UCP_MEMH_PACK_PARAM_FIELD_FLAGS
            pp.flags = 0  # no EXPORT: pack as RKEY for peer ep_rkey_unpack
            buf = ctypes.c_void_p(0)
            blen = ctypes.c_size_t(0)
            rc = _require_ucp().ucp_memh_pack(
                memh, ctypes.byref(pp), ctypes.byref(buf), ctypes.byref(blen)
            )
            if rc != _UCS_OK:
                raise RuntimeError(f"ucp_memh_pack failed status={rc}")
            blobs.append(ctypes.string_at(_ptr_value(buf), int(blen.value or 0)))
            rp = _MemhBufReleaseParams()
            rp.field_mask = 0
            _require_ucp().ucp_memh_buffer_release(buf, ctypes.byref(rp))
        self.caps_n_slots = len(self._owner._pool.blocks)
        self.caps_blob = _hs_pack_caps(
            [b.addr for b in self._owner._pool.blocks], blobs
        )

    def set_peer_caps(self, n_slots: int, put_thresh: int, caps: bytes) -> None:
        """Unpack the peer's slot capabilities: ep_rkey_unpack yields the
        rkeys usable by this ep (must be called after create_ep -- an rkey is
        bound to the local-to-peer ep)."""
        if n_slots <= 0:
            return
        addrs, blobs = _hs_unpack_caps(n_slots, caps)
        for blob in blobs:
            rbuf = ctypes.create_string_buffer(blob, len(blob))
            rkey = ctypes.c_void_p(0)
            rc = _require_ucp().ucp_ep_rkey_unpack(
                ctypes.c_void_p(self._ep), rbuf, ctypes.byref(rkey)
            )
            if rc != _UCS_OK:
                raise RuntimeError(f"ucp_ep_rkey_unpack failed status={rc}")
            self.peer_slot_rkeys.append(_ptr_value(rkey))
        self.peer_slot_addrs = addrs
        self.peer_n_slots = n_slots
        self.peer_put_thresh = put_thresh

    @property
    def worker(self) -> int:
        worker = self._worker
        if worker is None:
            raise RuntimeError("UCP worker is not created yet")
        return worker

    def my_address(self) -> bytes:
        attr = _UcpWorkerAttr()
        attr.field_mask = _UCP_WORKER_ATTR_FIELD_ADDRESS
        rc = _require_ucp().ucp_worker_query(
            ctypes.c_void_p(self._worker), ctypes.byref(attr)
        )
        if rc != _UCS_OK:
            raise RuntimeError(f"ucp_worker_query failed status={rc}")
        if not attr.address or not attr.address_length:
            raise RuntimeError("ucp_worker_query returned empty address")
        try:
            return ctypes.string_at(attr.address, attr.address_length)
        finally:
            _require_ucp().ucp_worker_release_address(
                ctypes.c_void_p(self._worker), ctypes.c_void_p(attr.address)
            )

    def create_ep(self, peer_addr: bytes) -> None:
        buf = ctypes.create_string_buffer(peer_addr, len(peer_addr))
        p = _UcpEpParams()
        p.field_mask = (
            _UCP_EP_PARAM_FIELD_REMOTE_ADDRESS | _UCP_EP_PARAM_FIELD_ERR_HANDLER
        )
        p.address = ctypes.cast(buf, ctypes.c_void_p)
        # PEER mode: UCX keepalive is enabled only in this mode (ucp_ep.c
        # checks heartbeats only when err_mode == PEER), so a dead peer
        # surfaces as an ep error within ~UCX_KEEPALIVE_INTERVAL and the
        # channel immediately reconnects as broken. Under NONE a dead link
        # only surfaces via the next operation's timeout (observed on real
        # hardware: after a cloud restart an old ep hung on in-flight sends
        # for ~60s while the cloud-side unbound fired at 60s first and the
        # engine crashed). Matching env on both ends:
        # UCX_KEEPALIVE_INTERVAL=5. To roll back, set this line back to 0 and
        # drop the env.
        p.err_mode = 1  # UCP_ERR_HANDLING_MODE_PEER
        p.err_handler_cb = ctypes.cast(self._err_cb, ctypes.c_void_p)
        p.err_handler_arg = ctypes.c_void_p(id(self._owner))
        ep = ctypes.c_void_p(0)
        rc = _require_ucp().ucp_ep_create(
            self._worker, ctypes.byref(p), ctypes.byref(ep)
        )
        if rc != _UCS_OK:
            raise RuntimeError(
                f"ucp_ep_create failed status={rc} (peer address invalid or "
                f"TLS combination unreachable)"
            )
        self._ep = ep.value  # must be stored: send_nb segfaults on a NULL ep

    def progress(self, max_ops: int = 64) -> int:
        """Drive completions (callbacks fire inside this call); returns the
        number of operations advanced."""
        total = 0
        for _ in range(max_ops):
            n = _require_ucp().ucp_worker_progress(self._worker)
            total += n
            if n == 0:
                break
        return total

    def send_nb(self, addr: int, length: int, tag: int, cb: Any) -> int:
        """Return None/0 if it completed immediately (no callback); >0 is a
        request; for errors see _ptr_is_err."""
        if not self._ep:
            raise RuntimeError("ucx ep not created before send_nb")
        return _require_ucp().ucp_tag_send_nbx(
            ctypes.c_void_p(self._ep),
            ctypes.c_void_p(addr),
            length,
            tag,
            ctypes.byref(_req_param(cb)),
        )

    def recv_nb(self, addr: int, length: int, cb: Any) -> int:
        """Wildcard (tag=0/mask=0) pre-post; returns the request pointer."""
        return _require_ucp().ucp_tag_recv_nbx(
            ctypes.c_void_p(self._worker),
            ctypes.c_void_p(addr),
            length,
            0,
            0,
            ctypes.byref(_req_param(cb)),
        )

    def put_nb(
        self, local_addr: int, length: int, remote_addr: int, rkey: int, cb: Any
    ) -> int:
        """One-sided WRITE (under rc the CQE means the data landed on the
        peer); return value semantics match send_nb."""
        if not self._ep:
            raise RuntimeError("ucx ep not created before put_nb")
        return _require_ucp().ucp_put_nbx(
            ctypes.c_void_p(self._ep),
            ctypes.c_void_p(local_addr),
            length,
            remote_addr,
            ctypes.c_void_p(rkey),
            ctypes.byref(_req_param(cb)),
        )

    def request_free(self, req: int) -> None:
        _require_ucp().ucp_request_free(ctypes.c_void_p(req))

    def _on_ep_err(self, arg, ep, status) -> None:  # noqa: ANN001
        # Base implementation routes to the owner's _on_ep_error; subclasses
        # may override
        handler = getattr(self._owner, "_on_ep_error", None)
        if handler is not None:
            handler(int(status))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for rkey in self.peer_slot_rkeys:
            with contextlib.suppress(Exception):
                _require_ucp().ucp_rkey_destroy(ctypes.c_void_p(rkey))
        self.peer_slot_rkeys = []
        if self._ep:
            with contextlib.suppress(Exception):
                # FORCE: do not wait for in-flight to drain (shutdown path,
                # the IO thread has already stopped); ep_close_nbx flags are
                # passed via ucp_request_param_t.flags
                cp = _UcpRequestParam()
                cp.flags = _UCP_EP_CLOSE_FLAG_FORCE
                _require_ucp().ucp_ep_close_nbx(
                    ctypes.c_void_p(self._ep), ctypes.byref(cp)
                )
            self._ep = None
        if self._worker:
            with contextlib.suppress(Exception):
                _require_ucp().ucp_worker_destroy(ctypes.c_void_p(self._worker))
            self._worker = None
        for memh in self._mem_hs:
            with contextlib.suppress(Exception):
                _require_ucp().ucp_mem_unmap(
                    ctypes.c_void_p(self._ctx), ctypes.c_void_p(memh)
                )
        self._mem_hs = []
        if self._ctx:
            with contextlib.suppress(Exception):
                _require_ucp().ucp_cleanup(ctypes.c_void_p(self._ctx))
            self._ctx = None
