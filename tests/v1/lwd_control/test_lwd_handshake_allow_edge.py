# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests of the cloud handshake's runtime allowlist (C-2, elastic
scale-out): `allow_edge` grows the frozenset atomically, so a late edge's
HELLO stops bouncing off the startup plan and registers through the
normal path."""

from __future__ import annotations

from vllm.v1.lwd_control.lwd_comm.lwd_handshake import LwdCloudHandshake
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import (
    LWD_LINK_PROTOCOL_VERSION,
    LwdLinkAccept,
    LwdLinkHello,
    LwdLinkReject,
)
from vllm.v1.lwd_control.lwd_comm.lwd_messages import LWD_MESSAGE_VERSION


def _hello(edge_id: int, *, session: str = "s1", dp_rank: int = 0) -> LwdLinkHello:
    return LwdLinkHello(
        link_version=LWD_LINK_PROTOCOL_VERSION,
        message_version=LWD_MESSAGE_VERSION,
        instance_id=edge_id,
        dp_rank=dp_rank,
        dp_size=1,
        session_id=session,
    )


def _handshake(*expected: int) -> LwdCloudHandshake:
    return LwdCloudHandshake(
        instance_id=0, dp_rank=0, dp_size=1, expected_edge_ids=tuple(expected)
    )


def test_a_late_edge_bounces_off_the_startup_plan() -> None:
    decision = _handshake(0).on_hello(b"id1", _hello(1))
    assert isinstance(decision.reply, LwdLinkReject)
    assert "edge instance 1 is not connected to cloud 0 dp0" in decision.reply.reason


def test_allow_edge_admits_a_late_edge() -> None:
    handshake = _handshake(0)
    handshake.allow_edge(1)
    decision = handshake.on_hello(b"id1", _hello(1))
    assert isinstance(decision.reply, LwdLinkAccept)
    assert decision.registered is not None
    assert decision.registered.instance_id == 1


def test_allow_edge_is_idempotent_and_does_not_disturb_live_sessions() -> None:
    handshake = _handshake(0)
    first = handshake.on_hello(b"id0", _hello(0, session="s-old"))
    assert isinstance(first.reply, LwdLinkAccept)
    # Idempotent double-allow keeps the same set (no KeyError, no reset).
    handshake.allow_edge(1)
    handshake.allow_edge(1)
    # The live session for edge 0 is untouched.
    assert handshake.peer_of(b"id0") is not None
    # A HELLO from the newly allowed edge still registers cleanly.
    decision = handshake.on_hello(b"id1", _hello(1))
    assert isinstance(decision.reply, LwdLinkAccept)


def test_allow_edge_on_an_empty_allowlist_stays_open() -> None:
    """An empty allowlist disables the check entirely (unit-test mode);
    allow_edge must not accidentally close it."""
    handshake = _handshake()
    decision = handshake.on_hello(b"id9", _hello(9))
    assert isinstance(decision.reply, LwdLinkAccept)
    handshake.allow_edge(0)
    decision = handshake.on_hello(b"id8", _hello(8))
    assert isinstance(decision.reply, LwdLinkAccept)


def test_a_replayed_hello_from_the_same_session_is_a_refresh_not_a_registration():
    """The atomic-swap allowlist must not confuse the session logic: a
    retried HELLO from the same process refreshes (registered=False),
    only a NEW session registers."""
    handshake = _handshake(0, 1)
    first = handshake.on_hello(b"id1", _hello(1))
    assert first.registered is not None
    # Same identity, same session: a refresh.
    retry = handshake.on_hello(b"id1", _hello(1))
    assert isinstance(retry.reply, LwdLinkAccept)
    assert retry.registered is None
