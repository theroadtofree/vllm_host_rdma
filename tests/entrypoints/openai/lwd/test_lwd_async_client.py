# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LwdAsyncMPClient: the LWD frontend's engine client, subclassed from the
native AsyncMPClient (vllm's core_client.py stays pristine).

A frontend that outlives its engine (the attach frontend when the main
command restarts) must survive the old engine's ENGINE_CORE_DEAD broadcast
and adopt the replacement engine; the engine re-sends READY on every
DEALER (re)connection, so init must also dedupe the doubled READY.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest
import zmq
import zmq.asyncio

from vllm.entrypoints.openai.lwd.lwd_async_client import (
    LwdAsyncMPClient,
    apply_lwd_async_client_patch,
)
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import BackgroundResources, EngineCoreClient
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


def test_native_validate_alive_marks_dead_but_never_clears() -> None:
    """The NATIVE marking stays: a DEAD frame marks and raises, and a
    later DEAD re-marks; clearing the flag is the LWD override's job
    (the reader loop below owns revival), so a normal frame through the
    native path leaves the flag set."""
    resources = BackgroundResources(ctx=zmq.Context())
    dead_frame = zmq.Frame(EngineCoreProc.ENGINE_CORE_DEAD)
    with pytest.raises(EngineDeadError):
        resources.validate_alive([dead_frame])
    assert resources.engine_dead
    with pytest.raises(EngineDeadError):
        resources.validate_alive([zmq.Frame(b"payload")])
    assert resources.engine_dead


def _client_with_input_socket(input_socket: zmq.Socket) -> LwdAsyncMPClient:
    client = object.__new__(LwdAsyncMPClient)
    client.resources = BackgroundResources(ctx=zmq.Context())
    client.resources.input_socket = input_socket
    return client


def test_ready_response_dedupe_skips_the_repeated_payload() -> None:
    """The engine re-sends READY on every (re)connection; the native
    config sync is not idempotent (num_gpu_blocks adds), so the second
    identical payload must be dropped."""
    client = object.__new__(LwdAsyncMPClient)
    client._lwd_applied_ready = b"ready"
    # A duplicate never reaches the native sync: with no vllm_config set
    # up, a delegation would raise AttributeError, so returning quietly is
    # the assertion.
    client._apply_ready_response(b"ready")
    # An empty payload delegates to the native early-return (no state
    # needed), and becomes the dedupe key only after being applied.
    client._apply_ready_response(b"")
    assert client._lwd_applied_ready == b""


def test_ensure_alive_raises_while_dead_without_buffered_ready() -> None:
    ctx = zmq.Context()
    router = ctx.socket(zmq.ROUTER)
    router.bind("inproc://lwd-client-no-ready")

    client = _client_with_input_socket(router)
    client.resources.engine_dead = True

    with pytest.raises(EngineDeadError):
        client.ensure_alive()
    assert client.resources.engine_dead

    router.close(linger=0)
    ctx.term()


def test_ensure_alive_revives_on_buffered_ready() -> None:
    ctx = zmq.Context()
    router = ctx.socket(zmq.ROUTER)
    router.bind("inproc://lwd-client-ready")
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.IDENTITY, (0).to_bytes(2, "little"))
    dealer.connect("inproc://lwd-client-ready")

    client = _client_with_input_socket(router)
    client.resources.engine_dead = True

    # The replacement engine's READY re-send lands on the input ROUTER.
    dealer.send(b"READY")
    assert router.poll(5000), "READY never arrived on the input ROUTER"

    client.ensure_alive()  # must not raise: the probe cleared the flag
    assert not client.resources.engine_dead

    # The buffered READY was drained: a second probe finds nothing.
    assert not client._maybe_revived()

    dealer.close(linger=0)
    router.close(linger=0)
    ctx.term()


@pytest.mark.asyncio
async def test_async_reader_survives_dead_and_revives() -> None:
    ctx = zmq.asyncio.Context()
    output_socket = ctx.socket(zmq.PULL)
    output_socket.bind("inproc://lwd-client-outputs")
    push = ctx.socket(zmq.PUSH)
    push.connect("inproc://lwd-client-outputs")

    client = object.__new__(LwdAsyncMPClient)
    client.resources = BackgroundResources(ctx=zmq.Context())
    client.resources.output_socket = output_socket
    client.outputs_queue = asyncio.Queue()
    client.utility_results = {}
    client.decoder = MsgpackDecoder(EngineCoreOutputs)
    client._ensure_output_queue_task()
    task = client.resources.output_queue_task
    assert task is not None and not task.done()

    # The old engine's DEAD broadcast must not end the reader.
    await push.send_multipart([EngineCoreProc.ENGINE_CORE_DEAD])
    for _ in range(500):
        if client.resources.engine_dead:
            break
        await asyncio.sleep(0.01)
    assert client.resources.engine_dead
    assert not task.done(), "reader exited on ENGINE_CORE_DEAD"

    # A normal output frame from the replacement engine revives the client
    # and is delivered to the queue as usual.
    frames = list(
        MsgpackEncoder().encode(EngineCoreOutputs(scheduler_stats=SchedulerStats()))
    )
    await push.send_multipart(frames)

    outputs = await asyncio.wait_for(client.get_output_async(), timeout=5)
    assert outputs.scheduler_stats is not None
    assert not task.done()
    assert not client.resources.engine_dead

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    push.close(linger=0)
    output_socket.close(linger=0)
    ctx.term()


def test_factory_patch_is_idempotent() -> None:
    """The factory wrapper applies once: a second call must not re-wrap
    (the first patched function would be captured as 'native' and the
    delegation chain would grow)."""
    apply_lwd_async_client_patch()
    after_first = EngineCoreClient.make_async_mp_client
    apply_lwd_async_client_patch()
    assert EngineCoreClient.make_async_mp_client is after_first
