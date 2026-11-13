# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-revival tests (LWD primary restart).

A frontend that outlives its engine (the attach frontend when the main
command restarts) must survive the old engine's ENGINE_CORE_DEAD
broadcast and adopt the replacement engine:

* the output reader loop stays alive across DEAD (an exited reader means
  every later output is unread and requests hang with no error);
* a normal output frame arriving while dead clears the flag (revival
  criterion #1, ``validate_alive``);
* a READY re-send buffered on the input ROUTER clears the flag when a
  send is intercepted by the dead marker (criterion #2,
  ``_maybe_revived`` via ``ensure_alive``).

The clients are built with ``object.__new__`` to skip the engine-spawning
``__init__``; only the attributes the probed code paths touch are set.
"""

import asyncio
import contextlib

import pytest
import zmq
import zmq.asyncio

from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import (
    AsyncMPClient,
    BackgroundResources,
    MPClient,
)
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


def test_validate_alive_marks_dead_and_revives_on_normal_frame():
    resources = BackgroundResources(ctx=zmq.Context())
    dead_frame = zmq.Frame(EngineCoreProc.ENGINE_CORE_DEAD)

    with pytest.raises(EngineDeadError):
        resources.validate_alive([dead_frame])
    assert resources.engine_dead

    # A normal output frame while dead: the replacement engine has taken
    # over; the flag clears instead of poisoning later sends.
    resources.validate_alive([zmq.Frame(b"payload")])
    assert not resources.engine_dead

    # A later DEAD (e.g. another restart) re-marks the flag.
    with pytest.raises(EngineDeadError):
        resources.validate_alive([dead_frame])
    assert resources.engine_dead


def _client_with_input_socket(input_socket: zmq.Socket) -> MPClient:
    client = object.__new__(MPClient)
    client.resources = BackgroundResources(ctx=zmq.Context())
    client.resources.input_socket = input_socket
    return client


def test_ensure_alive_raises_while_dead_without_buffered_ready():
    ctx = zmq.Context()
    router = ctx.socket(zmq.ROUTER)
    router.bind("inproc://lwd-revival-no-ready")

    client = _client_with_input_socket(router)
    client.resources.engine_dead = True

    with pytest.raises(EngineDeadError):
        client.ensure_alive()
    assert client.resources.engine_dead

    router.close(linger=0)
    ctx.term()


def test_ensure_alive_revives_on_buffered_ready():
    ctx = zmq.Context()
    router = ctx.socket(zmq.ROUTER)
    router.bind("inproc://lwd-revival-ready")
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.IDENTITY, (0).to_bytes(2, "little"))
    dealer.connect("inproc://lwd-revival-ready")

    client = _client_with_input_socket(router)
    client.resources.engine_dead = True

    # The replacement engine's READY re-send lands on the input ROUTER
    # (the engine replies on the output socket, so nothing else can be
    # buffered here after init).
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
async def test_async_reader_survives_dead_and_revives():
    ctx = zmq.asyncio.Context()
    output_socket = ctx.socket(zmq.PULL)
    output_socket.bind("inproc://lwd-revival-outputs")
    push = ctx.socket(zmq.PUSH)
    push.connect("inproc://lwd-revival-outputs")

    client = object.__new__(AsyncMPClient)
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

    # A normal output frame from the replacement engine revives the
    # client and is delivered to the queue as usual.
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
