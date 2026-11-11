# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Link handshake: message-version mismatches must fail the handshake.

`LWD_MESSAGE_VERSION` bumps whenever the wire payload grows a field the old
peer would drop silently; an old edge talking to a new cloud (or the other
way around) has to be rejected here instead of serving degraded requests.
"""

from vllm.v1.lwd_control.lwd_comm.lwd_handshake import LwdCloudHandshake
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import (
    LWD_LINK_PROTOCOL_VERSION,
    LwdLinkAccept,
    LwdLinkHello,
    LwdLinkReject,
    lwd_edge_identity,
)
from vllm.v1.lwd_control.lwd_comm.lwd_messages import LWD_MESSAGE_VERSION


def _hello(message_version: int) -> LwdLinkHello:
    return LwdLinkHello(
        link_version=LWD_LINK_PROTOCOL_VERSION,
        message_version=message_version,
        instance_id=1,
        dp_rank=0,
        dp_size=1,
        session_id="s1",
    )


def test_handshake_accepts_current_message_version():
    cloud = LwdCloudHandshake(instance_id=2, dp_rank=0, dp_size=1)
    identity = lwd_edge_identity(1, 0, "s1")
    decision = cloud.on_hello(identity, _hello(LWD_MESSAGE_VERSION))

    assert isinstance(decision.reply, LwdLinkAccept)
    assert decision.registered is not None


def test_handshake_rejects_stale_message_version():
    """An old edge (version 3) against a new cloud (version 4) must fail
    fast: the old cloud would drop `structured_outputs` silently."""
    cloud = LwdCloudHandshake(instance_id=2, dp_rank=0, dp_size=1)
    stale = LWD_MESSAGE_VERSION - 1

    decision = cloud.on_hello(lwd_edge_identity(1, 0, "s1"), _hello(stale))

    assert isinstance(decision.reply, LwdLinkReject)
    assert "message version" in decision.reply.reason
    assert decision.registered is None


def test_handshake_rejects_newer_message_version():
    cloud = LwdCloudHandshake(instance_id=2, dp_rank=0, dp_size=1)

    decision = cloud.on_hello(
        lwd_edge_identity(1, 0, "s1"), _hello(LWD_MESSAGE_VERSION + 1)
    )

    assert isinstance(decision.reply, LwdLinkReject)
    assert "message version" in decision.reply.reason
