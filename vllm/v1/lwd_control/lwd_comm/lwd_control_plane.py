# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LWD control plane of one DP rank: a ROUTER/DEALER link plus its IO thread.

Each plane owns a ZMQ context and one IO thread, the only thread touching the
TCP socket. The engine thread hands outbound payloads over through a bounded
inproc PUSH/PULL pair (`LwdControlSender`); inbound events leave the IO thread
through `LwdControlListener` callbacks.

A full link holds outbound payloads on the IO thread until the peer reads
again, so the sender's queue fills and `send()` reports backpressure; the edge
treats a lost cloud the same way. Peer loss is not detected yet: the cloud
drops payloads to an edge whose connection is gone, with an error log, and the
plane never stops the engine on its own.
"""

import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import msgspec
import zmq

from vllm.config.lwd.topology import LwdConnectionId, LwdControlPlaneConfig
from vllm.lwd_logger import init_lwd_logger
from vllm.v1.lwd_control.lwd_comm.lwd_handshake import (
    LwdCloudHandshake,
    LwdCloudHandshakeDecision,
    LwdEdgeHandshake,
    LwdHandshakeError,
)
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import (
    LWD_LINK_PROTOCOL_VERSION,
    LwdFrameKind,
    LwdLinkAccept,
    LwdLinkHello,
    LwdLinkMessage,
    LwdLinkReject,
    LwdPeer,
    lwd_decode_link_message,
    lwd_decode_peer,
    lwd_edge_identity,
    lwd_encode_link_message,
    lwd_encode_peer,
)
from vllm.v1.lwd_control.lwd_comm.lwd_messages import LWD_MESSAGE_VERSION
from vllm.v1.lwd_control.lwd_comm.lwd_transport import (
    LwdCloudTransport,
    LwdControlTransport,
    LwdEdgeTransport,
    LwdPeerUnreachableError,
)

logger = init_lwd_logger(__name__)

# Upper bound of IO thread latency for hello retries and shutdown.
_LWD_IO_POLL_TICK_MS = 100
# Retry period of a payload held back by a full link.
_LWD_HOLD_RETRY_TICK_MS = 10
_LWD_STARTUP_TIMEOUT_S = 30.0
_LWD_SHUTDOWN_JOIN_TIMEOUT_S = 5.0
# The inproc pipe capacity is SNDHWM + RCVHWM, so the IO side stays minimal
# and the sender's queue limit alone bounds the outbound queue.
_LWD_OUTBOUND_RCVHWM = 1
# A fixed name is safe: inproc endpoints are scoped to the plane's own context.
_LWD_OUTBOUND_PATH = "inproc://lwd-ctrl-out"


class LwdPeerNotRegisteredError(RuntimeError):
    """The peer has not completed the handshake, or the plane has stopped."""


class LwdControlPlaneStartupError(RuntimeError):
    """The control link could not be opened (e.g. the port is in use)."""


class LwdControlListener(Protocol):
    """Callbacks run on the IO thread; they must neither block nor raise."""

    def on_peer_up(self, peer: LwdPeer) -> None:
        """`peer` is registered; sending to it is now allowed."""

    def on_message(self, peer: LwdPeer, payload: bytes) -> None:
        """A business payload from `peer`, in the order it was sent."""

    def on_fatal(self, reason: str) -> None:
        """The IO thread crashed; the plane has stopped."""


@dataclass(frozen=True, kw_only=True)
class LwdControlPlaneSpec:
    """Identity and config of one DP rank's control plane.

    Usually built from `LwdConfig` by `lwd_create_control_plane`.
    """

    instance_id: int
    """This engine's LWD instance id in the topology."""
    dp_rank: int
    """`ParallelConfig.data_parallel_index`. Not `data_parallel_rank`, which
    non-MoE engine processes reset to 0."""
    dp_size: int
    """Global DP size. Read it before non-MoE engine processes reset
    `ParallelConfig.data_parallel_size` to 1."""
    config: LwdControlPlaneConfig = field(default_factory=LwdControlPlaneConfig)

    def __post_init__(self) -> None:
        if self.instance_id < 0:
            raise ValueError(f"[LWD] instance_id {self.instance_id} must be >= 0")
        if not 0 <= self.dp_rank < self.dp_size:
            raise ValueError(f"[LWD] dp_rank {self.dp_rank} not in [0, {self.dp_size})")


@dataclass(frozen=True, kw_only=True)
class LwdCloudControlPlaneSpec(LwdControlPlaneSpec):
    bind_endpoint: str
    """Endpoint the cloud listens on, e.g. "tcp://0.0.0.0:5550"."""
    expected_edge_ids: tuple[int, ...] = ()
    """Topology-declared edges; empty disables the allowlist for unit tests."""


@dataclass(frozen=True, kw_only=True)
class LwdEdgeControlPlaneSpec(LwdControlPlaneSpec):
    cloud_endpoints: tuple[str, ...]
    """Cloud DP endpoints, ordered by their parallel `connection_ids`."""
    connection_ids: tuple[LwdConnectionId, ...] = ()
    cloud_dp_sizes: tuple[int, ...] = ()
    """DP size of each remote cloud, not the total number of connections."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.cloud_endpoints:
            raise ValueError("[LWD] edge control plane needs a cloud endpoint")
        if not self.connection_ids:
            object.__setattr__(
                self,
                "connection_ids",
                tuple(
                    LwdConnectionId(self.instance_id, 0, dp_idx)
                    for dp_idx in range(len(self.cloud_endpoints))
                ),
            )
        if not self.cloud_dp_sizes:
            object.__setattr__(
                self, "cloud_dp_sizes", (len(self.cloud_endpoints),) * len(self.cloud_endpoints)
            )
        if not (
            len(self.cloud_endpoints)
            == len(self.connection_ids)
            == len(self.cloud_dp_sizes)
        ):
            raise ValueError("[LWD] cloud endpoint identity/size arrays must align")
        if self.dp_rank != 0:
            raise ValueError(
                f"[LWD] one edge process serves every DP connection; its "
                f"dp_rank is 0, got {self.dp_rank}"
            )


class _LwdRegisteredPeers:
    """Peers that completed the handshake; shared by the IO and engine threads.

    Copy-on-write: written on handshakes and shutdown, read on every send.
    Writers swap in a new frozenset; readers take the reference without a
    lock, since reading a reference is atomic.
    """

    def __init__(self) -> None:
        # Serializes writers only: shutdown may race a still-running IO thread.
        self._write_lock = threading.Lock()
        self._peers: frozenset[LwdPeer] = frozenset()

    def add(self, peer: LwdPeer) -> None:
        with self._write_lock:
            self._peers = self._peers | {peer}

    def discard(self, peer: LwdPeer) -> None:
        with self._write_lock:
            self._peers = self._peers - {peer}

    def clear(self) -> None:
        with self._write_lock:
            self._peers = frozenset()

    def __contains__(self, peer: object) -> bool:
        return peer in self._peers

    def snapshot(self) -> frozenset[LwdPeer]:
        return self._peers


class LwdControlSender:
    """Engine-side handle for outbound payloads, bound to its creating thread."""

    def __init__(
        self,
        ctx: zmq.Context,
        outbound_path: str,
        send_queue_max: int,
        registered_peers: _LwdRegisteredPeers,
    ) -> None:
        self._registered_peers = registered_peers
        self._owner_thread = threading.get_ident()
        sock = ctx.socket(zmq.PUSH)
        sock.setsockopt(zmq.SNDHWM, send_queue_max)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(outbound_path)
        self._socket: zmq.Socket | None = sock

    def send(self, peer: LwdPeer, payload: bytes) -> bool:
        """Queue `payload` for `peer` without blocking.

        Returns:
            False when the outbound queue is full: a peer that stopped
            reading backed up the link, or the edge lost its cloud. Retrying
            or dropping is the caller's decision.

        Raises:
            LwdPeerNotRegisteredError: `peer` is not registered, or the
                plane has stopped.
        """
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("LwdControlSender used outside its owner thread")
        if self._socket is None:
            raise LwdPeerNotRegisteredError("[LWD] the control plane has stopped")
        if peer not in self._registered_peers:
            raise LwdPeerNotRegisteredError(
                f"[LWD] {peer} has not completed the handshake"
            )
        try:
            self._socket.send_multipart((lwd_encode_peer(peer), payload), zmq.NOBLOCK)
        except zmq.Again:
            return False
        return True

    def close(self) -> None:
        if self._socket is None:
            return
        self._socket.close(linger=0)
        self._socket = None


class LwdControlPlane(ABC):
    """Common IO thread, framing, outbound queue and shutdown.

    Subclasses provide the transport, the handshake and peer resolution of
    their role.
    """

    _ROLE: str

    def __init__(self, spec: LwdControlPlaneSpec, endpoint: str) -> None:
        self.spec = spec
        self.endpoint = endpoint
        self._label = f"{self._ROLE}{spec.instance_id} dp{spec.dp_rank}/{spec.dp_size}"
        self._ctx = zmq.Context()
        self._registered_peers = _LwdRegisteredPeers()
        self._opened = threading.Event()
        self._stop = threading.Event()
        self._startup_error: Exception | None = None
        self._listener: LwdControlListener | None = None
        self._thread: threading.Thread | None = None
        self._sender: LwdControlSender | None = None
        self._is_shut_down = False
        # IO thread only. Each peer has an independent bounded FIFO. A full
        # FIFO applies backpressure through the shared inproc socket rather
        # than dropping an already-accepted payload; inbound IO stays live.
        self._num_dropped: dict[LwdPeer, int] = {}
        self._held: dict[LwdPeer, deque[bytes]] = {}
        self._held_since: dict[LwdPeer, float] = {}

    @property
    def registered_peers(self) -> frozenset[LwdPeer]:
        return self._registered_peers.snapshot()

    def start(self, listener: LwdControlListener) -> LwdControlSender:
        """Open the link on a new IO thread and return the outbound handle.

        Raises:
            LwdControlPlaneStartupError: The link could not be opened.
        """
        if self._thread is not None:
            raise RuntimeError("LWD control plane already started")
        self._listener = listener
        self._thread = threading.Thread(
            target=self._run_io_thread,
            name=f"lwd-ctrl-io-{self._ROLE}-dp{self.spec.dp_rank}",
            daemon=True,
        )
        self._thread.start()
        if not self._opened.wait(_LWD_STARTUP_TIMEOUT_S):
            self.shutdown()
            raise LwdControlPlaneStartupError(
                f"IO thread did not start within {_LWD_STARTUP_TIMEOUT_S}s"
            )
        if self._startup_error is not None:
            error = self._startup_error
            self.shutdown()
            raise LwdControlPlaneStartupError(
                f"cannot open control link {self.endpoint}: {error}"
            ) from error
        # The PULL end is bound before `_opened`, so EAGAIN on the sender can
        # only mean backpressure, never "no receiver yet".
        self._sender = LwdControlSender(
            self._ctx,
            _LWD_OUTBOUND_PATH,
            self.spec.config.send_queue_max,
            self._registered_peers,
        )
        return self._sender

    def shutdown(self, join_timeout_s: float = _LWD_SHUTDOWN_JOIN_TIMEOUT_S) -> None:
        """Idempotent; queued outbound payloads are discarded.

        Must run on the thread that called `start()`, which owns the sender.
        """
        if self._is_shut_down:
            return
        self._is_shut_down = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(join_timeout_s)
        self._registered_peers.clear()
        if self._sender is not None:
            self._sender.close()
        if self._thread is not None and self._thread.is_alive():
            logger.error(
                "control plane %s: IO thread still alive after %.1fs; "
                "destroying its context",
                self._label,
                join_timeout_s,
            )
            self._ctx.destroy(linger=0)
            return
        self._ctx.term()
        logger.info("control plane %s shut down", self._label)

    # ---------------------------------------------------------------- IO thread

    def _run_io_thread(self) -> None:
        try:
            transport, outbound = self._open_sockets()
        except Exception as e:
            self._startup_error = e
            self._opened.set()
            return
        self._opened.set()
        logger.info("control plane %s opened %s", self._label, self.endpoint)
        try:
            self._io_loop(transport, outbound)
        except Exception as e:
            logger.exception("control plane %s: IO thread crashed", self._label)
            self._on_io_crash(f"IO thread crashed: {e!r}")
        finally:
            transport.close()
            outbound.close(linger=0)

    def _open_sockets(self) -> tuple[LwdControlTransport, zmq.Socket]:
        outbound = self._ctx.socket(zmq.PULL)
        try:
            outbound.setsockopt(zmq.RCVHWM, _LWD_OUTBOUND_RCVHWM)
            outbound.setsockopt(zmq.LINGER, 0)
            outbound.bind(_LWD_OUTBOUND_PATH)
            transport = self._create_transport()
            transport.open(self._ctx)
        except Exception:
            outbound.close(linger=0)
            raise
        return transport, outbound

    def _io_loop(self, transport: LwdControlTransport, outbound: zmq.Socket) -> None:
        poller = zmq.Poller()
        for socket in transport.sockets:
            poller.register(socket, zmq.POLLIN)
        poller.register(outbound, zmq.POLLIN)
        while not self._stop.is_set():
            # Drain the inproc queue while peer FIFOs have room. Blocked peers
            # retain their FIFOs; inbound traffic continues to be processed.
            poller.poll(_LWD_IO_POLL_TICK_MS)
            self._drain_inbound(transport)
            if self._stop.is_set():
                return
            self._drain_outbound(transport, outbound)
            self._on_tick(transport, time.monotonic())

    def _drain_inbound(self, transport: LwdControlTransport) -> None:
        for frame in transport.recv_all():
            if frame.kind is LwdFrameKind.LINK:
                self._handle_link_frame(transport, frame.identity, frame.body)
            else:
                self._handle_payload(frame.identity, frame.body, frame.link)
            if self._stop.is_set():
                return

    def _handle_link_frame(
        self, transport: LwdControlTransport, identity: bytes | None, body: bytes
    ) -> None:
        try:
            msg = lwd_decode_link_message(body)
        except (msgspec.DecodeError, msgspec.ValidationError) as e:
            logger.warning(
                "control plane %s: dropped malformed link frame from %r: %s",
                self._label,
                identity,
                e,
            )
            return
        self._on_link_message(transport, identity, msg)

    def _handle_payload(
        self, identity: bytes | None, payload: bytes, link: int | None = None
    ) -> None:
        # The edge names its links by index; the cloud keeps routing
        # identities. One convention per role keeps _inbound_peer simple.
        peer = self._inbound_peer(identity if link is None else str(link).encode())
        if peer is None:
            logger.warning(
                "control plane %s: dropped payload from unregistered %r",
                self._label,
                identity,
            )
            return
        assert self._listener is not None
        self._listener.on_message(peer, payload)

    def _drain_outbound(
        self, transport: LwdControlTransport, outbound: zmq.Socket
    ) -> None:
        while True:
            # Once any peer FIFO reaches its bound, leave further messages in
            # the inproc PULL socket. This applies backpressure to senders
            # instead of acknowledging a message here and silently dropping it
            # later. Inbound traffic and already-held peers are still drained.
            if any(
                len(queue) >= self.spec.config.send_queue_max
                for queue in self._held.values()
            ):
                break
            try:
                peer_frame, payload = outbound.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                break
            peer = lwd_decode_peer(peer_frame)
            self._held.setdefault(peer, deque()).append(payload)

        # Retry only each FIFO's head, preserving per-peer ordering while
        # allowing every other peer to make progress.
        for peer, queue in list(self._held.items()):
            if not queue:
                self._release_held(peer)
                continue
            if self._deliver(transport, peer, queue[0]):
                queue.popleft()
                if not queue:
                    self._release_held(peer)
            else:
                self._note_held(peer)

    def _note_held(self, peer: LwdPeer) -> None:
        if peer not in self._held_since:
            self._held_since[peer] = time.monotonic()
            logger.warning(
                "control plane %s: link to %s is full or not connected; "
                "holding outbound payloads",
                self._label,
                peer,
            )

    def _release_held(self, peer: LwdPeer) -> None:
        self._held.pop(peer, None)
        held_since = self._held_since.pop(peer, None)
        if held_since is not None:
            logger.info(
                "control plane %s: link to %s drained after %.1fs",
                self._label,
                peer,
                time.monotonic() - held_since,
            )

    def _deliver(
        self, transport: LwdControlTransport, peer: LwdPeer, payload: bytes
    ) -> bool:
        """Send or drop `payload`; False when the link is full and it must be
        retried."""
        try:
            identity = self._outbound_identity(peer)
            if not transport.send(LwdFrameKind.PAYLOAD, payload, identity):
                return False
        except LwdPeerUnreachableError as e:
            self._on_peer_unreachable(peer)
            num_dropped = self._num_dropped.get(peer, 0) + 1
            self._num_dropped[peer] = num_dropped
            if num_dropped == 1:
                logger.error(
                    "control plane %s: %s unreachable (%s); dropping payloads "
                    "to it until it is reachable again",
                    self._label,
                    peer,
                    e,
                )
            return True
        num_dropped = self._num_dropped.pop(peer, 0)
        if num_dropped:
            logger.warning(
                "control plane %s: %s reachable again; %d payloads were dropped",
                self._label,
                peer,
                num_dropped,
            )
        return True

    def _on_peer_unreachable(self, peer: LwdPeer) -> None:
        """Role hook for immediate ROUTER disconnect notification."""

    def _unregister_peer(self, peer: LwdPeer, reason: str) -> None:
        self._registered_peers.discard(peer)
        self._held.pop(peer, None)
        self._held_since.pop(peer, None)
        logger.warning("control plane %s: peer %s down: %s", self._label, peer, reason)
        callback = getattr(self._listener, "on_peer_down", None)
        if callback is not None:
            callback(peer)

    def _register_peer(self, peer: LwdPeer) -> None:
        self._registered_peers.add(peer)
        assert self._listener is not None
        self._listener.on_peer_up(peer)

    def _on_io_crash(self, reason: str) -> None:
        self._registered_peers.clear()
        if self._listener is not None:
            self._listener.on_fatal(reason)

    # ------------------------------------------------------------ role hooks

    @abstractmethod
    def _create_transport(self) -> LwdControlTransport: ...

    @abstractmethod
    def _on_link_message(
        self,
        transport: LwdControlTransport,
        identity: bytes | None,
        msg: LwdLinkMessage,
    ) -> None: ...

    @abstractmethod
    def _inbound_peer(self, identity: bytes | None) -> LwdPeer | None:
        """The registered peer behind `identity`, or None."""

    @abstractmethod
    def _outbound_identity(self, peer: LwdPeer) -> bytes | None:
        """Routing identity of `peer` for outbound frames."""

    def _on_tick(self, transport: LwdControlTransport, now: float) -> None:
        return None


class LwdEdgeControlPlane(LwdControlPlane):
    """Edge side: DEALERs that register with every cloud DP rank it serves.

    One DEALER connects to each cloud endpoint; all of them share the edge
    identity, so each cloud DP rank sees its own session. Inbound replies
    are attributed to the peer that completed the handshake for the reply's
    (dp_rank, session) pair; a payload held by one full link only holds
    that link's payloads, never the other links'.
    """

    _ROLE = "edge"

    def __init__(self, spec: LwdEdgeControlPlaneSpec) -> None:
        super().__init__(spec, spec.cloud_endpoints[0])
        self._endpoints = spec.cloud_endpoints
        self._connection_ids = spec.connection_ids
        self._link_indices = {
            connection_id: index
            for index, connection_id in enumerate(self._connection_ids)
        }
        session_id = uuid.uuid4().hex
        # Each DEALER registers with one explicit (edge, cloud, engine) link.
        self._handshakes: dict[LwdConnectionId, LwdEdgeHandshake] = {
            connection_id: LwdEdgeHandshake(
                LwdLinkHello(
                    link_version=LWD_LINK_PROTOCOL_VERSION,
                    message_version=LWD_MESSAGE_VERSION,
                    instance_id=connection_id.edge_id,
                    dp_rank=connection_id.dp_idx,
                    dp_size=cloud_dp_size,
                    session_id=session_id,
                ),
                spec.config.hello_retry_interval_s,
            )
            for connection_id, cloud_dp_size in zip(
                self._connection_ids, spec.cloud_dp_sizes, strict=True
            )
        }
        self._settled = threading.Event()  # Every handshake settled, or a failure.
        self._handshake_error: str | None = None
        self._num_hellos_sent = 0

    def _create_transport(self) -> LwdControlTransport:
        identities = tuple(
            lwd_edge_identity(
                connection_id.edge_id,
                connection_id.dp_idx,
                self._handshakes[connection_id].hello.session_id,
            )
            for connection_id in self._connection_ids
        )
        return LwdEdgeTransport(self._endpoints, identities)

    # ------------------------------------------------------------ ready check

    def wait_until_ready(
        self, timeout_s: float | None = None
    ) -> dict[LwdConnectionId, LwdPeer]:
        """Block until every cloud DP rank accepted; shuts down on failure.

        Must run on the thread that called `start()`.

        Args:
            timeout_s: Defaults to the configured `handshake_timeout_s`.

        Returns:
            The cloud peers by dp_rank to send to.

        Raises:
            LwdHandshakeError: Timeout, rejection, or IO thread failure.
        """
        if timeout_s is None:
            timeout_s = self.spec.config.handshake_timeout_s
        if not self._settled.wait(timeout_s):
            missing = self._missing_handshakes()
            self.shutdown()
            raise LwdHandshakeError(
                f"{len(missing)} of {len(self._handshakes)} cloud DP ranks "
                f"did not accept within {timeout_s}s "
                f"({self._num_hellos_sent} hellos delivered); missing connections "
                f"{sorted(missing)}"
            )
        if self._handshake_error is not None:
            self.shutdown()
            raise LwdHandshakeError(self._handshake_error)
        clouds: dict[LwdConnectionId, LwdPeer] = {}
        for connection_id, handshake in self._handshakes.items():
            cloud = handshake.cloud
            assert cloud is not None, "settled without every cloud accepted"
            clouds[connection_id] = cloud
        return clouds

    def _missing_handshakes(self) -> set[LwdConnectionId]:
        return {
            connection_id
            for connection_id, handshake in self._handshakes.items()
            if handshake.cloud is None
        }

    # ---------------------------------------------------------- link handling

    def _on_link_message(
        self,
        transport: LwdControlTransport,
        identity: bytes | None,
        msg: LwdLinkMessage,
    ) -> None:
        # A reply names the dp_rank it answers; only that link's handshake
        # consumes it. Other shapes fall through to every handshake, which
        # rejects them and fails the plane.
        if isinstance(msg, LwdLinkAccept):
            consumers = [
                handshake
                for connection_id, handshake in self._handshakes.items()
                if (connection_id.cloud_id, connection_id.dp_idx)
                == (msg.instance_id, msg.dp_rank)
            ]
            if not consumers:
                consumers = list(self._handshakes.values())
        else:
            consumers = list(self._handshakes.values())
        for handshake in consumers:
            try:
                handshake.on_reply(msg)
            except LwdHandshakeError as e:
                if handshake.completed:
                    logger.error(
                        "control plane %s: ignored link message after the "
                        "handshake: %s",
                        self._label,
                        e,
                    )
                else:
                    self._fail_handshake(str(e))
                return
        self._register_settled_peers()
        if all(h.completed for h in self._handshakes.values()):
            self._settled.set()

    def _register_settled_peers(self) -> None:
        for handshake in self._handshakes.values():
            cloud = handshake.cloud
            if cloud is None or cloud in self._registered_peers:
                continue
            logger.info(
                "control plane %s: accepted by cloud %s, session %s",
                self._label,
                cloud,
                handshake.hello.session_id,
            )
            self._register_peer(cloud)

    def _fail_handshake(self, reason: str) -> None:
        logger.error("control plane %s: handshake failed: %s", self._label, reason)
        self._handshake_error = reason
        self._settled.set()
        self._stop.set()

    def _inbound_peer(self, identity: bytes | None) -> LwdPeer | None:
        """`identity` is the link index (decimal bytes) of the delivering
        DEALER; the link index is the dp_rank it registered as."""
        if identity is None:
            return None
        try:
            connection_id = self._connection_ids[int(identity)]
        except (ValueError, IndexError):
            return None
        handshake = self._handshakes.get(connection_id)
        if handshake is None:
            return None
        cloud = handshake.cloud
        if cloud is not None:
            # Any authenticated frame proves the peer is alive, not only Accept.
            handshake.touch()
        return cloud

    def _outbound_identity(self, peer: LwdPeer) -> bytes | None:
        """The link index of the DEALER that reaches `peer`."""
        connection_id = next(
            (
                candidate
                for candidate in self._connection_ids
                if (candidate.cloud_id, candidate.dp_idx)
                == (peer.instance_id, peer.dp_rank)
            ),
            None,
        )
        index = self._link_indices.get(connection_id)
        return None if index is None else str(index).encode()

    def _on_tick(self, transport: LwdControlTransport, now: float) -> None:
        for connection_id, handshake in self._handshakes.items():
            if handshake.completed or not handshake.should_send_hello(now):
                continue
            hello = lwd_encode_link_message(handshake.hello)
            link_index = self._link_indices[connection_id]
            if not transport.send(
                LwdFrameKind.LINK, hello, str(link_index).encode()
            ):
                continue  # Cloud not connected yet; retried on the next tick.
            handshake.on_hello_sent(now)
            self._num_hellos_sent += 1
            if self._num_hellos_sent == 1:
                logger.info(
                    "control plane %s: hello sent to %s",
                    self._label,
                    self._endpoints,
                )

    def _on_io_crash(self, reason: str) -> None:
        if not self._settled.is_set():
            self._handshake_error = reason
            self._settled.set()
        super()._on_io_crash(reason)


class LwdCloudControlPlane(LwdControlPlane):
    """Cloud side: a ROUTER that registers the edge of its DP rank."""

    _ROLE = "cloud"

    def __init__(self, spec: LwdCloudControlPlaneSpec) -> None:
        super().__init__(spec, spec.bind_endpoint)
        self._handshake = LwdCloudHandshake(
            spec.instance_id, spec.dp_rank, spec.dp_size, spec.expected_edge_ids
        )

    def _create_transport(self) -> LwdControlTransport:
        return LwdCloudTransport(self.endpoint)

    def _on_link_message(
        self,
        transport: LwdControlTransport,
        identity: bytes | None,
        msg: LwdLinkMessage,
    ) -> None:
        if identity is None or not isinstance(msg, LwdLinkHello):
            logger.warning(
                "control plane %s: dropped unexpected %s from %r",
                self._label,
                type(msg).__name__,
                identity,
            )
            return
        decision = self._handshake.on_hello(identity, msg)
        self._reply(transport, identity, decision)
        if decision.registered is not None:
            logger.info(
                "control plane %s: registered edge %s, session %s",
                self._label,
                decision.registered,
                msg.session_id,
            )
            self._register_peer(decision.registered)

    def _reply(
        self,
        transport: LwdControlTransport,
        identity: bytes,
        decision: LwdCloudHandshakeDecision,
    ) -> None:
        reply = decision.reply
        if isinstance(reply, LwdLinkReject):
            logger.error(
                "control plane %s: rejected hello from %r: %s",
                self._label,
                identity,
                reply.reason,
            )
        try:
            sent = transport.send(
                LwdFrameKind.LINK, lwd_encode_link_message(reply), identity
            )
        except LwdPeerUnreachableError:
            sent = False
        if not sent:
            logger.warning(
                "control plane %s: cannot answer %r; the edge resends its hello",
                self._label,
                identity,
            )

    def _inbound_peer(self, identity: bytes | None) -> LwdPeer | None:
        if identity is None:
            return None
        self._handshake.touch(identity)
        return self._handshake.peer_of(identity)

    def _on_peer_unreachable(self, peer: LwdPeer) -> None:
        if self._handshake.drop_peer(peer) is not None:
            self._unregister_peer(peer, "ROUTER reported the DEALER unreachable")

    def _on_tick(self, transport: LwdControlTransport, now: float) -> None:
        # Static deployments use startup handshake and explicit socket errors;
        # idle periods never invalidate an established peer.
        return None

    def _outbound_identity(self, peer: LwdPeer) -> bytes | None:
        identity = self._handshake.identity_of(peer)
        if identity is None:
            raise LwdPeerUnreachableError(f"{peer} is not registered")
        return identity
