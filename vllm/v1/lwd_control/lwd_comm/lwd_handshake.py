# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Socket-free handshake state machines of the LWD control link.

The edge registers with `LwdLinkHello` until the cloud answers with
`LwdLinkAccept` or `LwdLinkReject`. Each cloud DP rank keeps independent
sessions for its declared edges.
"""

from dataclasses import dataclass, field
import time

from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import (
    LWD_LINK_PROTOCOL_VERSION,
    LwdLinkAccept,
    LwdLinkHello,
    LwdLinkMessage,
    LwdLinkReject,
    LwdPeer,
    lwd_edge_identity,
)
from vllm.v1.lwd_control.lwd_comm.lwd_messages import LWD_MESSAGE_VERSION


class LwdHandshakeError(RuntimeError):
    """The edge-cloud handshake failed; the edge must not serve."""


class LwdEdgeHandshake:
    def __init__(self, hello: LwdLinkHello, hello_retry_interval_s: float) -> None:
        self.hello = hello
        self._hello_retry_interval_s = hello_retry_interval_s
        self._last_hello_at: float | None = None
        self._last_accept_at: float | None = None
        self.cloud: LwdPeer | None = None

    @property
    def completed(self) -> bool:
        return self.cloud is not None

    def should_send_hello(self, now: float) -> bool:
        """HELLO doubles as registration and the application heartbeat."""
        return (
            self._last_hello_at is None
            or now - self._last_hello_at >= self._hello_retry_interval_s
        )

    def heartbeat_expired(self, now: float, timeout_s: float) -> bool:
        return (
            self.completed
            and self._last_accept_at is not None
            and now - self._last_accept_at >= timeout_s
        )

    def disconnect(self) -> None:
        self.cloud = None
        self._last_accept_at = None
        self._last_hello_at = None

    def on_hello_sent(self, now: float) -> None:
        self._last_hello_at = now

    def on_reply(self, msg: LwdLinkMessage) -> None:
        """Consume a cloud reply; repeated accepts of retried hellos are no-ops.

        Raises:
            LwdHandshakeError: Rejected, or an accept that does not answer
                this hello.
        """
        if isinstance(msg, LwdLinkReject):
            raise LwdHandshakeError(f"cloud rejected the hello: {msg.reason}")
        if not isinstance(msg, LwdLinkAccept):
            raise LwdHandshakeError(f"unexpected {type(msg).__name__} from the cloud")
        expected = (
            LWD_LINK_PROTOCOL_VERSION,
            self.hello.dp_rank,
            self.hello.dp_size,
            self.hello.session_id,
        )
        actual = (msg.link_version, msg.dp_rank, msg.dp_size, msg.session_id)
        if actual != expected:
            raise LwdHandshakeError(
                f"accept (link_version, dp_rank, dp_size, session) = {actual} "
                f"does not answer hello {expected}"
            )
        if self.cloud is None:
            self.cloud = LwdPeer(msg.instance_id, msg.dp_rank)
        self._last_accept_at = time.monotonic()


@dataclass(frozen=True)
class LwdCloudHandshakeDecision:
    reply: LwdLinkAccept | LwdLinkReject
    registered: LwdPeer | None = None
    """Set when this hello registered a new edge."""


@dataclass
class _LwdEdgeSession:
    peer: LwdPeer
    identity: bytes
    session_id: str
    last_seen: float = field(default_factory=time.monotonic)


class LwdCloudHandshake:
    """Registers edges on one cloud DP rank and maps them to ZMQ identities."""

    def __init__(
        self,
        instance_id: int,
        dp_rank: int,
        dp_size: int,
        expected_edge_ids: tuple[int, ...] = (),
    ) -> None:
        self._instance_id = instance_id
        self._dp_rank = dp_rank
        self._dp_size = dp_size
        self._expected_edge_ids = frozenset(expected_edge_ids)
        self._sessions_by_identity: dict[bytes, _LwdEdgeSession] = {}
        self._sessions_by_peer: dict[LwdPeer, _LwdEdgeSession] = {}

    def peer_of(self, identity: bytes) -> LwdPeer | None:
        session = self._sessions_by_identity.get(identity)
        return session.peer if session is not None else None

    def identity_of(self, peer: LwdPeer) -> bytes | None:
        session = self._sessions_by_peer.get(peer)
        return session.identity if session is not None else None

    def on_hello(
        self, identity: bytes, hello: LwdLinkHello
    ) -> LwdCloudHandshakeDecision:
        reason = self._mismatch_reason(identity, hello)
        if reason is not None:
            return LwdCloudHandshakeDecision(LwdLinkReject(reason))
        peer = LwdPeer(hello.instance_id, hello.dp_rank)
        previous = self._sessions_by_peer.get(peer)
        if previous is not None and (
            previous.identity != identity or previous.session_id != hello.session_id
        ):
            # A restarted edge gets a fresh DEALER identity. Replace the old
            # session atomically; payloads from its former identity no longer
            # authenticate as this peer.
            self._sessions_by_identity.pop(previous.identity, None)
        elif previous is not None:
            previous.last_seen = time.monotonic()
            return LwdCloudHandshakeDecision(self._accept(hello))
        session = _LwdEdgeSession(peer, identity, hello.session_id)
        self._sessions_by_identity[identity] = session
        self._sessions_by_peer[peer] = session
        return LwdCloudHandshakeDecision(self._accept(hello), registered=peer)

    def _mismatch_reason(self, identity: bytes, hello: LwdLinkHello) -> str | None:
        if self._expected_edge_ids and hello.instance_id not in self._expected_edge_ids:
            return (
                f"edge instance {hello.instance_id} is not connected to cloud "
                f"{self._instance_id} dp{self._dp_rank} in the topology"
            )
        if hello.link_version != LWD_LINK_PROTOCOL_VERSION:
            return (
                f"link protocol version {hello.link_version} != cloud "
                f"{LWD_LINK_PROTOCOL_VERSION}"
            )
        if hello.message_version != LWD_MESSAGE_VERSION:
            return (
                f"message version {hello.message_version} != cloud "
                f"{LWD_MESSAGE_VERSION}"
            )
        if hello.dp_rank != self._dp_rank:
            return (
                f"edge dp_rank {hello.dp_rank} reached cloud dp_rank "
                f"{self._dp_rank}; check the control ports"
            )
        if hello.dp_size != self._dp_size:
            return f"edge dp_size {hello.dp_size} != cloud dp_size {self._dp_size}"
        expected = lwd_edge_identity(hello.instance_id, hello.dp_rank, hello.session_id)
        if identity != expected:
            return f"identity {identity!r} != expected {expected!r}"
        return None

    def touch(self, identity: bytes) -> None:
        session = self._sessions_by_identity.get(identity)
        if session is not None:
            session.last_seen = time.monotonic()

    def drop_peer(self, peer: LwdPeer) -> bytes | None:
        session = self._sessions_by_peer.pop(peer, None)
        if session is None:
            return None
        self._sessions_by_identity.pop(session.identity, None)
        return session.identity

    def expired_peers(self, now: float, timeout_s: float) -> list[LwdPeer]:
        return [
            session.peer
            for session in self._sessions_by_peer.values()
            if now - session.last_seen >= timeout_s
        ]

    def _accept(self, hello: LwdLinkHello) -> LwdLinkAccept:
        return LwdLinkAccept(
            link_version=LWD_LINK_PROTOCOL_VERSION,
            instance_id=self._instance_id,
            dp_rank=self._dp_rank,
            dp_size=self._dp_size,
            session_id=hello.session_id,
        )
