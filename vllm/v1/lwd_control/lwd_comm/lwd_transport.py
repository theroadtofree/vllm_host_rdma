# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-level ZMQ transport of the LWD control links.

The cloud binds one ROUTER; the edge connects one DEALER per cloud DP rank
it serves, each identity carrying its session id. A transport must be
opened, used and closed on a single thread (ZMQ sockets are not
thread-safe).

Both directions are bounded by `_LWD_LINK_HWM`, so a peer that stops reading
eventually makes `send()` report a full link instead of buffering without
limit.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import zmq

from vllm.lwd_logger import init_lwd_logger
from vllm.utils.network_utils import is_valid_ipv6_address, split_zmq_path
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdFrameKind

logger = init_lwd_logger(__name__)

# TCP keepalive drops an idle link to a vanished host in about
# IDLE + INTVL * CNT seconds (~2 min) instead of the OS default (often 2 h).
# It does not cover a link with unacknowledged data, which the kernel
# retransmission timeout governs.
# TODO(lwd): replace with application heartbeats in the peer loss detection
# change, issue #8.
_LWD_TCP_KEEPALIVE_IDLE_S = 60
_LWD_TCP_KEEPALIVE_INTVL_S = 10
_LWD_TCP_KEEPALIVE_CNT = 6
# Messages queued per connection and direction (the ZMQ default). The receive
# side must be bounded too: otherwise libzmq keeps reading TCP into memory and
# a stalled peer never backs up to the sender.
_LWD_LINK_HWM = 1000


class LwdPeerUnreachableError(RuntimeError):
    """The peer has no live connection to deliver a frame to."""


@dataclass(frozen=True)
class LwdInboundFrame:
    identity: bytes | None
    """Routing identity of the sender; None on the cloud side. The edge
    transport carries the link index instead, so a payload names the link
    that delivered it."""
    kind: LwdFrameKind
    body: bytes
    link: int | None = None
    """Edge only: index of the link the frame arrived on."""


class LwdControlTransport(ABC):
    """The ZMQ sockets of a control plane: options, framing, non-blocking
    I/O. One socket for the cloud ROUTER, one DEALER per link for the edge.

    Subclasses provide the socket type, role options, attach and framing.
    """

    _SOCKET_TYPE: ClassVar[int]

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self._socket: zmq.Socket | None = None

    def open(self, ctx: zmq.Context) -> None:
        """Create, configure and attach the sockets. Cleans up on failure."""
        sock = self._create_socket(ctx)
        try:
            # Options only apply to connections made after they are set, so
            # `make_zmq_socket`, which attaches immediately, cannot be used.
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.SNDHWM, _LWD_LINK_HWM)
            sock.setsockopt(zmq.RCVHWM, _LWD_LINK_HWM)
            sock.setsockopt(zmq.TCP_KEEPALIVE, 1)
            sock.setsockopt(zmq.TCP_KEEPALIVE_IDLE, _LWD_TCP_KEEPALIVE_IDLE_S)
            sock.setsockopt(zmq.TCP_KEEPALIVE_INTVL, _LWD_TCP_KEEPALIVE_INTVL_S)
            sock.setsockopt(zmq.TCP_KEEPALIVE_CNT, _LWD_TCP_KEEPALIVE_CNT)
            _, host, _ = split_zmq_path(self.endpoint)
            if is_valid_ipv6_address(host):
                sock.setsockopt(zmq.IPV6, 1)
            self._configure(sock)
            self._attach(sock)
        except Exception:
            sock.close(linger=0)
            raise
        self._socket = sock

    def _create_socket(self, ctx: zmq.Context) -> zmq.Socket:
        return ctx.socket(self._SOCKET_TYPE)

    @property
    def socket(self) -> zmq.Socket:
        if self._socket is None:
            raise RuntimeError("control link transport is not open")
        return self._socket

    @property
    def sockets(self) -> list[zmq.Socket]:
        """Every socket the IO loop polls."""
        return [self.socket]

    def recv_all(self) -> list[LwdInboundFrame]:
        """Drain every pending message of every socket without blocking."""
        frames: list[LwdInboundFrame] = []
        for link, socket in enumerate(self.sockets):
            while True:
                try:
                    parts = socket.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
                frame = self._parse(parts, link)
                if frame is None:
                    logger.warning(
                        "control link %s: dropped malformed message of %d parts",
                        self.endpoint,
                        len(parts),
                    )
                    continue
                frames.append(frame)
        return frames

    def send(self, kind: LwdFrameKind, body: bytes, identity: bytes | None) -> bool:
        """Send one message without blocking.

        Args:
            identity: The ROUTER routing identity on the cloud; the link
                index, as decimal bytes, on the edge.

        Returns:
            False when the link cannot take the message now: the high-water
            mark is reached, or the link is not connected yet.

        Raises:
            LwdPeerUnreachableError: The cloud has no connection to
                `identity`.
        """
        try:
            self._send(kind.value, body, identity)
        except zmq.Again:
            return False
        except zmq.ZMQError as e:
            if e.errno != zmq.EHOSTUNREACH:
                raise
            raise LwdPeerUnreachableError(f"{self.endpoint}: peer gone") from e
        return True

    def close(self) -> None:
        """Idempotent; pending outbound frames are discarded."""
        if self._socket is None:
            return
        self._socket.close(linger=0)
        self._socket = None

    @abstractmethod
    def _configure(self, sock: zmq.Socket) -> None: ...

    @abstractmethod
    def _attach(self, sock: zmq.Socket) -> None: ...

    @abstractmethod
    def _parse(self, parts: list[bytes], link: int) -> LwdInboundFrame | None: ...

    @abstractmethod
    def _send(self, kind: bytes, body: bytes, identity: bytes | None) -> None: ...


def _lwd_frame_kind(raw: bytes) -> LwdFrameKind | None:
    try:
        return LwdFrameKind(raw)
    except ValueError:
        return None


class LwdCloudTransport(LwdControlTransport):
    """ROUTER bound by the cloud; messages are [identity, kind, body]."""

    _SOCKET_TYPE = zmq.ROUTER

    def _configure(self, sock: zmq.Socket) -> None:
        # Fail with EHOSTUNREACH instead of silently dropping frames to an
        # identity whose connection is gone.
        sock.setsockopt(zmq.ROUTER_MANDATORY, 1)
        # Edge identities include the session id, so handover only applies to
        # one edge process reconnecting, which then resumes at once.
        sock.setsockopt(zmq.ROUTER_HANDOVER, 1)

    def _attach(self, sock: zmq.Socket) -> None:
        sock.bind(self.endpoint)

    def _parse(self, parts: list[bytes], link: int) -> LwdInboundFrame | None:
        if len(parts) != 3:
            return None
        kind = _lwd_frame_kind(parts[1])
        if kind is None:
            return None
        return LwdInboundFrame(identity=parts[0], kind=kind, body=parts[2])

    def _send(self, kind: bytes, body: bytes, identity: bytes | None) -> None:
        if identity is None:
            raise ValueError("ROUTER send requires a peer identity")
        self.socket.send_multipart((identity, kind, body), zmq.NOBLOCK)


class LwdEdgeTransport(LwdControlTransport):
    """DEALERs connected by the edge, one per cloud DP rank; messages are
    [kind, body].

    One socket per link: a DEALER with several connects would round-robin
    sends across the clouds and replies could not be attributed to a link.
    Link i connects to cloud DP rank i and carries the identity of that
    link's handshake; a full or unestablished link reports EAGAIN for its
    own sends only.
    """

    _SOCKET_TYPE = zmq.DEALER

    def __init__(
        self, endpoints: tuple[str, ...], identities: tuple[bytes, ...]
    ) -> None:
        super().__init__(endpoints[0])
        if len(endpoints) != len(identities):
            raise ValueError(
                f"[LWD] one identity per link, got {len(endpoints)} endpoints "
                f"and {len(identities)} identities"
            )
        self._endpoints = endpoints
        self._identities = identities
        self._links: list[zmq.Socket] = []

    def open(self, ctx: zmq.Context) -> None:
        for endpoint, identity in zip(self._endpoints, self._identities):
            link = ctx.socket(self._SOCKET_TYPE)
            try:
                link.setsockopt(zmq.LINGER, 0)
                link.setsockopt(zmq.SNDHWM, _LWD_LINK_HWM)
                link.setsockopt(zmq.RCVHWM, _LWD_LINK_HWM)
                link.setsockopt(zmq.TCP_KEEPALIVE, 1)
                link.setsockopt(zmq.TCP_KEEPALIVE_IDLE, _LWD_TCP_KEEPALIVE_IDLE_S)
                link.setsockopt(zmq.TCP_KEEPALIVE_INTVL, _LWD_TCP_KEEPALIVE_INTVL_S)
                link.setsockopt(zmq.TCP_KEEPALIVE_CNT, _LWD_TCP_KEEPALIVE_CNT)
                _, host, _ = split_zmq_path(endpoint)
                if is_valid_ipv6_address(host):
                    link.setsockopt(zmq.IPV6, 1)
                link.setsockopt(zmq.IDENTITY, identity)
                # Never queue frames for a link that is not established:
                # sends report a full link (EAGAIN) instead of piling up
                # until the cloud appears.
                link.setsockopt(zmq.IMMEDIATE, 1)
                link.connect(endpoint)
            except Exception:
                link.close(linger=0)
                self.close()
                raise
            self._links.append(link)

    @property
    def socket(self) -> zmq.Socket:
        if not self._links:
            raise RuntimeError("control link transport is not open")
        return self._links[0]

    @property
    def sockets(self) -> list[zmq.Socket]:
        return list(self._links)

    def _configure(self, sock: zmq.Socket) -> None:
        return None  # options set per link in open()

    def _attach(self, sock: zmq.Socket) -> None:
        return None  # links attached in open()

    def _parse(self, parts: list[bytes], link: int) -> LwdInboundFrame | None:
        if len(parts) != 2:
            return None
        kind = _lwd_frame_kind(parts[0])
        if kind is None:
            return None
        return LwdInboundFrame(identity=None, kind=kind, body=parts[1], link=link)

    def _send(self, kind: bytes, body: bytes, identity: bytes | None) -> None:
        if identity is None:
            raise ValueError("DEALER send requires a link index")
        link = int(identity)
        self._links[link].send_multipart((kind, body), zmq.NOBLOCK)

    def close(self) -> None:
        """Idempotent; pending outbound frames are discarded."""
        for link in self._links:
            link.close(linger=0)
        self._links = []
