# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.executor.multiproc_executor import WorkerProc


class _ExitWorkerLoop(RuntimeError):
    pass


class _RpcPayload:
    pass


class _PayloadLifetimeCheckingQueue:
    def __init__(self) -> None:
        self.payload_ref: weakref.ReferenceType[_RpcPayload] | None = None
        self.dequeue_count = 0

    def dequeue(self, *, indefinite: bool):
        assert indefinite
        self.dequeue_count += 1
        if self.dequeue_count == 1:
            payload = _RpcPayload()
            self.payload_ref = weakref.ref(payload)
            return "consume", (payload,), {}, None

        assert self.payload_ref is not None
        assert self.payload_ref() is None
        raise _ExitWorkerLoop


def test_worker_rpc_payload_released_before_next_dequeue():
    queue = _PayloadLifetimeCheckingQueue()
    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rpc_broadcast_mq = queue
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(consume=lambda payload: payload)
    worker_proc.handle_output = lambda output: None

    with pytest.raises(_ExitWorkerLoop):
        worker_proc.worker_busy_loop()

    assert queue.dequeue_count == 2


def test_execute_worker_rpc_returns_worker_exception():
    def fail():
        raise RuntimeError("test error")

    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(fail=fail)
    outputs: list[Any] = []
    worker_proc.handle_output = outputs.append

    worker_proc._execute_worker_rpc(("fail", (), {}, None))

    assert len(outputs) == 1
    assert isinstance(outputs[0], RuntimeError)
    assert str(outputs[0]) == "test error"


def test_lwd_down_meta_harvested_before_output_enqueue(monkeypatch):
    calls = []
    stream = MagicMock()
    stream.wait_event.side_effect = lambda event: calls.append(("wait", event))
    done = MagicMock()
    done.record.side_effect = lambda: calls.append("record")
    done.synchronize.side_effect = lambda: calls.append("synchronize")

    @contextmanager
    def stream_context(_stream):
        calls.append("enter")
        yield
        calls.append("exit")

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            Stream=MagicMock(return_value=stream),
            Event=MagicMock(return_value=done),
            stream=stream_context,
        ),
        raising=False,
    )
    ready_event = object()
    output = SimpleNamespace(
        lwd_down_carrier=(
            torch.tensor([10, 11], dtype=torch.int32),
            [2, 1],
            ["req-a", "req-b"],
            99,
            7,
        ),
        lwd_meta_ready_event=ready_event,
    )
    enqueued = []

    def enqueue(result):
        calls.append("enqueue")
        enqueued.append(result)

    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.worker_response_mq = SimpleNamespace(enqueue=enqueue)
    worker_proc.enqueue_output(output)

    assert calls == [
        "enter",
        ("wait", ready_event),
        "record",
        "exit",
        "synchronize",
        "enqueue",
    ]
    assert not hasattr(output, "lwd_meta_ready_event")
    packed, req_ids, hidden_numel, seqno = output.lwd_down_carrier
    assert packed.tolist() == [10, 11, 2, 1]
    assert req_ids == ["req-a", "req-b"]
    assert hidden_numel == 99
    assert seqno == 7
    assert enqueued == [(WorkerProc.ResponseStatus.SUCCESS, output)]


def test_lwd_down_meta_requires_ready_event():
    output = SimpleNamespace(
        lwd_down_carrier=(
            torch.tensor([10], dtype=torch.int32),
            None,
            ["req-a"],
            99,
            7,
        )
    )
    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.worker_response_mq = None

    with pytest.raises(RuntimeError, match="missing its ready event"):
        worker_proc.enqueue_output(output)


def test_enqueue_output_without_lwd_carrier_is_unchanged():
    output = SimpleNamespace(lwd_down_carrier=None)
    enqueued = []
    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.worker_response_mq = SimpleNamespace(enqueue=enqueued.append)

    worker_proc.enqueue_output(output)

    assert enqueued == [(WorkerProc.ResponseStatus.SUCCESS, output)]
    assert not hasattr(worker_proc, "_lwd_harvest_stream")
    assert not hasattr(worker_proc, "_lwd_harvest_pinned")
