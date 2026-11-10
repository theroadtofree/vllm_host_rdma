# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import LwdBatch, LwdBatchType, SchedulerOutput
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine import (
    LwdCloudEngineCoreMixin,
)
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_scheduler import (
    LwdCloudScheduler,
    _LwdPrefillWindow,
)
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_prefix_cache import (
    LwdPrefixReservations,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
)
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LwdAbortNotify,
    LwdRangeItem,
    LwdRangeNotify,
    LwdRequestNotify,
)
from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
from vllm.v1.request import RequestStatus


class _Request(SimpleNamespace):
    __hash__ = object.__hash__


def _request(
    request_id: str = "p",
    prompt: int = 10,
    computed: int = 2,
    *,
    preemptions: int = 0,
    skip: bool = False,
    floor: int = 0,
):
    return _Request(
        request_id=request_id,
        num_prompt_tokens=prompt,
        num_tokens=prompt,
        num_computed_tokens=computed,
        num_preemptions=preemptions,
        block_hashes=[],
        skip_reading_prefix_cache=skip,
        lwd_resume_cap_tokens=floor,
        is_finished=lambda: False,
        lwd_window_cap=None,
    )


def _visible_set_scheduler(running, waiting) -> LwdBaseScheduler:
    scheduler = LwdBaseScheduler.__new__(LwdBaseScheduler)
    scheduler.running = running
    scheduler.waiting = FCFSRequestQueue(waiting)
    scheduler.skipped_waiting = FCFSRequestQueue()
    scheduler.requests = {r.request_id: r for r in running + waiting}
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.max_num_running_reqs = len(running)
    scheduler.max_num_scheduled_tokens = 100
    scheduler.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
    return scheduler


def _scheduler(request) -> LwdCloudScheduler:
    scheduler = LwdCloudScheduler.__new__(LwdCloudScheduler)
    scheduler.running = [request]
    scheduler.waiting = []
    scheduler.skipped_waiting = []
    scheduler.requests = {request.request_id: request}
    scheduler.lwd_recv_rows = {}
    scheduler._lwd_dropped = set()
    scheduler._lwd_abort_pending = []
    scheduler._lwd_prefix_enabled = False
    scheduler._lwd_refill_gate = set()
    scheduler._lwd_refill_pending = {}
    scheduler._lwd_prefill_group_cursor = 0
    scheduler.lwd_prefix_hits = {}
    return scheduler


def _prefix_scheduler(request, groups, hit: int, recv_rows: int):
    scheduler = _scheduler(request)
    scheduler._lwd_prefix_enabled = True
    scheduler.lwd_recv_rows[request.request_id] = recv_rows
    scheduler.lwd_refill_sender = Mock()
    manager = SimpleNamespace(coordinator=Mock(), block_pool=Mock())
    manager.coordinator.find_longest_cache_hit.return_value = (groups, hit, 0)
    manager.coordinator.single_type_managers = (SimpleNamespace(block_size=16),)
    manager.coordinator.scheduler_block_size = 16
    manager.coordinator.cache_hit_alignment_tokens = 16
    scheduler.kv_cache_manager = manager
    scheduler.lwd_prefix_reservations = LwdPrefixReservations(manager)
    return scheduler


def _engine(prompt_lens: dict[str, int] | None = None) -> LwdCloudEngineCoreMixin:
    engine = LwdCloudEngineCoreMixin.__new__(LwdCloudEngineCoreMixin)
    engine._lwd_prompt_lens = dict(prompt_lens or {})
    engine._lwd_retired_prompt_lens = {}
    engine._lwd_request_nonces = {}
    engine._lwd_retired_nonces = {}
    engine._lwd_retired_req_ids = set()
    engine._lwd_rejected_nonces = {}
    engine._lwd_pending_seqnos = {}
    engine._lwd_request_seqnos = {}
    engine._lwd_latest_seqno = -1
    engine.scheduler = SimpleNamespace(lwd_recv_rows={})
    return engine


def _allow_request_add(engine: LwdCloudEngineCoreMixin) -> None:
    engine.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_hidden_size=lambda: 8,
            dtype=torch.float32,
        )
    )
    engine.input_queue = queue.Queue()
    engine.engines_running = True


def test_fragment_waits_but_final_and_budget_limited_windows_run():
    request = _request()
    scheduler = _scheduler(request)
    scheduler.lwd_recv_rows["p"] = 5

    assert scheduler._lwd_collect_prefill_windows(room=8) == []

    budget_window = scheduler._lwd_collect_prefill_windows(room=2)
    assert [(w.start, w.num_rows) for w in budget_window] == [(2, 2)]

    scheduler.lwd_recv_rows["p"] = 10
    final_window = scheduler._lwd_collect_prefill_windows(room=8)
    assert [(w.start, w.num_rows) for w in final_window] == [(2, 8)]


def test_decode_exact_is_accounted_before_prefill_windows():
    scheduler = _scheduler(_request())
    scheduler.max_num_scheduled_tokens = 10
    calls = []
    out = SchedulerOutput.make_empty()
    scheduler._lwd_collect_decode_requests = lambda: calls.append("decode") or ["d"]
    scheduler._lwd_compute_decode_consumption = lambda ids: calls.append(ids) or 3
    scheduler._lwd_collect_prefill_windows = lambda room: calls.append(room) or []
    scheduler._lwd_schedule = lambda ids: out

    assert scheduler._lwd_schedule_step() is out
    assert calls == ["decode", ["d"], 7]


def test_window_cap_is_cleared_when_native_scheduling_fails():
    request = _request()
    scheduler = _scheduler(request)
    window = _LwdPrefillWindow(request=request, num_rows=4, start=2)

    def fail_schedule(request_ids, *, token_budget_cap=None):
        assert request_ids == ["d", "p"]
        assert token_budget_cap == 7
        assert request.lwd_window_cap == 4
        raise RuntimeError("native failure")

    scheduler._lwd_schedule = fail_schedule
    with pytest.raises(RuntimeError, match="native failure"):
        scheduler._lwd_dispatch_windows([window], ["d"], decode_exact=3)
    assert request.lwd_window_cap is None


def test_w2_rejects_unarrived_rows_and_allows_native_under_scheduling():
    request = _request()
    scheduler = _scheduler(request)
    scheduler.lwd_recv_rows["p"] = 5
    window = _LwdPrefillWindow(request=request, num_rows=3, start=2)
    out = SimpleNamespace(num_scheduled_tokens={"p": 4})

    with pytest.raises(RuntimeError, match="W2 violation"):
        scheduler._lwd_verify_w2(out, [window], decode_exact=1)

    out.num_scheduled_tokens["p"] = 1
    scheduler._lwd_verify_w2(out, [window], decode_exact=1)
    scheduler._lwd_warn_under_scheduled(out, [window])


def test_visible_set_restores_native_preempted_at_head_order(monkeypatch):
    hidden_running = _request("hidden-running")
    hidden_waiting = _request("hidden-waiting")
    p1 = _request("p1")
    rest = _request("rest")
    p2 = _request("p2")
    scheduler = _visible_set_scheduler([hidden_running, p1, p2], [hidden_waiting, rest])

    def native_schedule(self, throttle_prefills=False):
        self.running = []
        self.waiting = FCFSRequestQueue([p1, rest, p2])
        out = SchedulerOutput.make_empty()
        out.preempted_req_ids = {"p1", "p2"}
        return out

    monkeypatch.setattr(AsyncScheduler, "schedule", native_schedule)
    scheduler._lwd_schedule(["p1", "rest", "p2"])

    assert [r.request_id for r in scheduler.running] == ["hidden-running"]
    assert [r.request_id for r in scheduler.waiting] == [
        "p1",
        "p2",
        "hidden-waiting",
        "rest",
    ]


def test_visible_set_preserves_non_preempted_order_when_output_is_none(monkeypatch):
    hidden_running = _request("hidden-running")
    visible_running = _request("visible-running")
    hidden_waiting = _request("hidden-waiting")
    rest1 = _request("rest1")
    rest2 = _request("rest2")
    scheduler = _visible_set_scheduler(
        [hidden_running, visible_running], [hidden_waiting, rest1, rest2]
    )

    monkeypatch.setattr(AsyncScheduler, "schedule", lambda self: None)
    assert scheduler._lwd_schedule(["visible-running", "rest1", "rest2"]) is None
    assert [r.request_id for r in scheduler.running] == [
        "hidden-running",
        "visible-running",
    ]
    assert [r.request_id for r in scheduler.waiting] == [
        "hidden-waiting",
        "rest1",
        "rest2",
    ]


def test_preemption_recomputes_window_from_zero_and_clears_cap():
    request = _request(prompt=10, computed=4)
    scheduler = _scheduler(request)
    scheduler.max_num_scheduled_tokens = 10
    scheduler.lwd_recv_rows["p"] = 10
    calls = []

    def schedule(request_ids, *, token_budget_cap=None):
        calls.append(
            (request.num_computed_tokens, request.lwd_window_cap, token_budget_cap)
        )
        out = SchedulerOutput.make_empty()
        if len(calls) == 1:
            request.num_computed_tokens = 0
            out.preempted_req_ids = {"p"}
        else:
            out.num_scheduled_tokens["p"] = 10
        return out

    scheduler._lwd_schedule = schedule
    scheduler._lwd_schedule_step()
    assert request.num_computed_tokens == 0
    assert request.lwd_window_cap is None
    assert scheduler.lwd_recv_rows == {"p": 10}

    scheduler._lwd_schedule_step()
    assert calls == [(4, 6, 6), (0, 10, 10)]
    assert request.lwd_window_cap is None


def test_abort_records_drop_and_preserves_native_iterable(monkeypatch):
    request = _request()
    scheduler = _scheduler(request)
    received = []

    def native_finish(self, request_ids, finished_status):
        received.append((request_ids, finished_status))
        return []

    monkeypatch.setattr(LwdBaseScheduler, "finish_requests", native_finish)
    scheduler.finish_requests((rid for rid in ["p"]), RequestStatus.FINISHED_ABORTED)

    assert scheduler._lwd_dropped == {"p"}
    assert scheduler._lwd_abort_pending == ["p"]
    assert received == [(("p",), RequestStatus.FINISHED_ABORTED)]


def test_abort_only_embed_and_non_embed_retry_do_not_lose_notice():
    scheduler = _scheduler(_request())
    scheduler._lwd_abort_pending = ["p"]
    out = SchedulerOutput.make_empty()

    scheduler._lwd_attach_abort_notice(out)
    assert out.lwd_batch.seqno == -1
    assert out.lwd_batch.batch_type is LwdBatchType.LWD_EMBED
    assert out.lwd_batch.batch_meta.aborted_req_ids == ["p"]
    assert scheduler._lwd_abort_pending == []

    scheduler._lwd_abort_pending = ["q"]
    out.lwd_batch = LwdBatch(LwdBatchType.LWD_UNEMBED, 4, None)
    scheduler._lwd_attach_abort_notice(out)
    assert scheduler._lwd_abort_pending == ["q"]
    assert out.lwd_batch.batch_type is LwdBatchType.LWD_UNEMBED


def test_pending_abort_keeps_native_engine_alive(monkeypatch):
    scheduler = _scheduler(_request())
    scheduler._lwd_abort_pending = ["p"]
    monkeypatch.setattr(LwdBaseScheduler, "has_requests", lambda self: False)
    assert scheduler.has_requests()


def test_full_prompt_mrope_reaches_scheduled_new_request():
    request = _request()
    request.lwd_mrope_positions = b"positions"
    scheduler = _scheduler(request)
    data = SimpleNamespace(req_id="p", lwd_mrope_positions=None)
    out = SimpleNamespace(scheduled_new_reqs=[data])

    scheduler._lwd_attach_mrope_positions(out)
    assert data.lwd_mrope_positions == b"positions"


def test_request_build_records_full_prompt_mrope():
    engine = LwdCloudEngineCoreMixin.__new__(LwdCloudEngineCoreMixin)
    engine.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_hidden_size=lambda: 8,
            dtype=torch.float32,
        )
    )
    notify = LwdRequestNotify("p", 3, mrope_positions=b"positions")

    request = engine._lwd_build_request(notify)
    assert request.num_prompt_tokens == 3
    assert request.prompt_embeds.is_meta
    assert request.lwd_mrope_positions == b"positions"


def test_range_notify_posts_only_extended_hint(monkeypatch):
    from vllm_ascend.worker.lwd_cloud_model import lwd_pre_recv

    engine = _engine({"a": 8, "b": 6})
    posted = []
    monkeypatch.setattr(
        lwd_pre_recv,
        "make_recv_hint",
        lambda *args, **kwargs: (args, kwargs),
    )
    monkeypatch.setattr(lwd_pre_recv, "post_irecv_hint", posted.append)
    notify = LwdRangeNotify(
        7,
        [
            LwdRangeItem("a", 2, 3, prefix_nonce=3),
            LwdRangeItem("b", 1, 2, prefix_nonce=5),
        ],
    )

    engine._lwd_handle_range_notify(notify)
    args, kwargs = posted[0]
    assert args == (7, 5, 0)
    assert kwargs == {
        "req_ids": ["a", "b"],
        "rows": [3, 2],
        "prompt_offsets": [2, 1],
        "prompt_lens": [8, 6],
        "prefix_nonces": [3, 5],
    }


def test_range_notify_without_prompt_length_fails_fast():
    engine = _engine()
    notify = LwdRangeNotify(7, [LwdRangeItem("a", 0, 3)])

    with pytest.raises(RuntimeError, match="before RequestNotify"):
        engine._lwd_handle_range_notify(notify)


def test_done_arrivals_merge_by_max_and_keep_watermark_items(monkeypatch):
    from vllm_ascend.worker.lwd_cloud_model import lwd_pre_recv

    class DoneMq:
        def __init__(self):
            self.items = [("UP", 1), (("UP", 2), [("a", 4), ("b", 3)])]

        def dequeue(self, timeout):
            if self.items:
                return self.items.pop(0)
            raise TimeoutError

    engine = _engine()
    engine._lwd_done_mq = DoneMq()
    engine._lwd_track_range(1, ["a"])
    engine._lwd_track_range(2, ["a", "b"])
    watermark = []
    monkeypatch.setattr(
        lwd_pre_recv,
        "record_irecv_completions",
        lambda items: watermark.extend(items) or list(items),
    )

    engine._lwd_drain_irecv_completions()
    engine._lwd_track_range(3, ["b"])
    engine._lwd_note_arrived(3, [("a", 2), ("b", 5)])
    assert watermark == [("UP", 1), ("UP", 2)]
    assert engine.scheduler.lwd_recv_rows == {"a": 4, "b": 5}


def test_cloud_abort_uses_eager_and_ordered_native_queues():
    engine = _engine({"p": 4})
    engine.aborts_queue = queue.Queue()
    engine.input_queue = queue.Queue()

    engine._lwd_handle_abort_notify(LwdAbortNotify("p"))
    assert engine.aborts_queue.get_nowait() == ["p"]
    request_type, request_ids = engine.input_queue.get_nowait()
    assert request_type.name == "ABORT"
    assert request_ids == ["p"]
    assert engine._lwd_prompt_lens == {}


def test_abort_ignores_late_arrival_publish():
    engine = _engine({"p": 4})
    engine.scheduler.lwd_recv_rows["p"] = 2
    engine.aborts_queue = queue.Queue()
    engine.input_queue = queue.Queue()
    engine._lwd_track_range(7, ["p"])

    engine._lwd_handle_abort_notify(LwdAbortNotify("p"))
    engine._lwd_note_arrived(7, [("p", 4)])
    assert engine.scheduler.lwd_recv_rows == {}


def test_abort_duplicate_range_is_idempotent_after_prompt_cleanup():
    engine = _engine({"p": 4})
    engine.aborts_queue = queue.Queue()
    engine.input_queue = queue.Queue()
    engine._lwd_track_range(7, ["p"])
    engine._lwd_handle_abort_notify(LwdAbortNotify("p"))
    engine._lwd_complete_seqno(7)
    assert engine._lwd_prompt_lens == {}

    notify = LwdRangeNotify(7, [LwdRangeItem("p", 0, 4)])
    engine._lwd_handle_range_notify(notify)
    assert engine._lwd_pending_seqnos == {}
    assert engine._lwd_latest_seqno == 7


def test_normal_finish_retires_from_outputs_and_cleans_after_drain():
    engine = _engine({"p": 4})
    engine.scheduler.requests = {}
    engine._lwd_track_range(8, ["p"])
    outputs = {0: SimpleNamespace(finished_requests=["p"], outputs=[])}

    engine._lwd_retire_finished_outputs(outputs)
    assert engine._lwd_retired_req_ids == {"p"}
    assert engine._lwd_prompt_lens == {"p": 4}

    engine._lwd_complete_seqno(8)
    assert engine._lwd_prompt_lens == {}
    assert engine._lwd_pending_seqnos == {}
    assert engine._lwd_request_seqnos == {}


def test_abort_before_mixed_range_still_posts_complete_hint(monkeypatch):
    from vllm_ascend.worker.lwd_cloud_model import lwd_pre_recv

    engine = _engine({"a": 8, "b": 6})
    engine.aborts_queue = queue.Queue()
    engine.input_queue = queue.Queue()
    engine._lwd_track_range(6, ["a"])
    engine._lwd_handle_abort_notify(LwdAbortNotify("a"))
    posted = []
    monkeypatch.setattr(
        lwd_pre_recv,
        "make_recv_hint",
        lambda *args, **kwargs: (args, kwargs),
    )
    monkeypatch.setattr(lwd_pre_recv, "post_irecv_hint", posted.append)
    notify = LwdRangeNotify(
        7,
        [LwdRangeItem("a", 2, 3), LwdRangeItem("b", 1, 2)],
    )

    engine._lwd_handle_range_notify(notify)
    args, kwargs = posted[0]
    assert args == (7, 5, 0)
    assert kwargs["req_ids"] == ["a", "b"]
    assert kwargs["prompt_lens"] == [8, 6]


@pytest.mark.parametrize("retire_kind", ["finish", "abort"])
def test_drained_retirement_allows_same_id_reuse(retire_kind):
    engine = _engine({"p": 4})
    engine._lwd_track_range(1, ["p"])
    if retire_kind == "abort":
        engine.aborts_queue = queue.Queue()
        engine.input_queue = queue.Queue()
        engine._lwd_handle_abort_notify(LwdAbortNotify("p"))
    else:
        outputs = {0: SimpleNamespace(finished_requests=["p"], outputs=[])}
        engine._lwd_retire_finished_outputs(outputs)
    assert engine._lwd_retired_req_ids == {"p"}

    engine._lwd_complete_seqno(1)
    assert engine._lwd_retired_req_ids == set()
    assert engine._lwd_retired_prompt_lens == {}
    _allow_request_add(engine)
    engine._lwd_handle_request_notify(LwdRequestNotify("p", 4))
    request_type, (request, _wave) = engine.input_queue.get_nowait()
    assert request_type.name == "ADD" and request.request_id == "p"

    engine._lwd_track_range(2, ["p"])
    engine._lwd_note_arrived(2, [("p", 4)])
    engine._lwd_note_arrived(1, [("p", 99)])
    assert engine.scheduler.lwd_recv_rows == {"p": 4}


def test_early_reuse_error_ready_late_messages_and_resubmit(monkeypatch) -> None:
    from vllm_ascend.worker.lwd_cloud_model import lwd_pre_recv

    engine = _engine({"p": 4})
    engine._lwd_request_nonces["p"] = 1
    engine._lwd_track_range(1, ["p"])
    engine._lwd_retire_request("p", preserve_prompt=True)
    _allow_request_add(engine)
    engine.aborts_queue = queue.Queue()
    engine._lwd_send_prefix_ready = Mock()
    posted = []
    monkeypatch.setattr(
        lwd_pre_recv, "make_recv_hint", lambda *args, **kwargs: (args, kwargs)
    )
    monkeypatch.setattr(lwd_pre_recv, "post_irecv_hint", posted.append)

    engine._lwd_handle_request_notify(LwdRequestNotify("p", 8, prefix_nonce=2))

    ready_args = engine._lwd_send_prefix_ready.call_args
    assert ready_args.args[:3] == ("p", 0, 2)
    assert "previous generation" in ready_args.kwargs["error"]
    assert engine._lwd_rejected_nonces == {"p": 2}
    assert engine.input_queue.empty()
    assert engine._lwd_prompt_lens["p"] == 4

    engine._lwd_handle_range_notify(
        LwdRangeNotify(2, [LwdRangeItem("p", 0, 8, prefix_nonce=2)])
    )
    engine._lwd_handle_abort_notify(LwdAbortNotify("p", prefix_nonce=2))

    assert not posted
    assert engine.aborts_queue.empty()
    assert engine._lwd_latest_seqno == 1
    assert engine._lwd_pending_seqnos == {1: {"p"}}

    engine._lwd_complete_seqno(1)

    assert engine._lwd_rejected_nonces == {}
    engine._lwd_handle_request_notify(LwdRequestNotify("p", 8, prefix_nonce=2))
    request_type, (request, _wave) = engine.input_queue.get_nowait()
    assert request_type.name == "ADD"
    assert request.lwd_prefix_nonce == 2
    assert engine._lwd_prompt_lens["p"] == 8

    engine._lwd_handle_range_notify(
        LwdRangeNotify(2, [LwdRangeItem("p", 0, 8, prefix_nonce=2)])
    )
    assert len(posted) == 1
    assert engine._lwd_pending_seqnos == {2: {"p"}}


def test_rejected_range_does_not_block_other_seqnos(monkeypatch) -> None:
    from vllm_ascend.worker.lwd_cloud_model import lwd_pre_recv

    engine = _engine({"p": 4, "q": 6})
    engine._lwd_request_nonces.update({"p": 1, "q": 1})
    engine._lwd_track_range(1, ["p"])
    engine._lwd_retire_request("p", preserve_prompt=True)
    _allow_request_add(engine)
    engine._lwd_send_prefix_ready = Mock()
    posted = []
    monkeypatch.setattr(
        lwd_pre_recv, "make_recv_hint", lambda *args, **kwargs: (args, kwargs)
    )
    monkeypatch.setattr(lwd_pre_recv, "post_irecv_hint", posted.append)
    engine._lwd_handle_request_notify(LwdRequestNotify("p", 8, prefix_nonce=2))
    engine._lwd_handle_range_notify(
        LwdRangeNotify(2, [LwdRangeItem("p", 0, 8, prefix_nonce=2)])
    )

    engine._lwd_handle_range_notify(
        LwdRangeNotify(3, [LwdRangeItem("q", 0, 6, prefix_nonce=1)])
    )

    assert [args[0] for args, _kwargs in posted] == [3]
    assert engine._lwd_latest_seqno == 3
    assert engine._lwd_pending_seqnos == {1: {"p"}, 3: {"q"}}


def test_stale_abort_is_ignored_for_current_generation() -> None:
    engine = _engine({"p": 8})
    engine._lwd_request_nonces["p"] = 2
    engine._lwd_retired_nonces["p"] = 1
    engine.aborts_queue = queue.Queue()
    engine.input_queue = queue.Queue()

    engine._lwd_handle_abort_notify(LwdAbortNotify("p", prefix_nonce=1))

    assert engine.input_queue.empty()
    assert engine.aborts_queue.empty()
    assert engine._lwd_request_nonces["p"] == 2
    assert engine._lwd_prompt_lens["p"] == 8


def test_retired_state_is_bounded_after_drain():
    engine = _engine()
    for seqno in range(5):
        engine._lwd_prompt_lens["p"] = 4
        engine._lwd_track_range(seqno, ["p"])
        engine._lwd_retire_request("p", preserve_prompt=True)
        assert engine._lwd_retired_req_ids == {"p"}
        engine._lwd_complete_seqno(seqno)
        assert engine._lwd_retired_req_ids == set()
        assert engine._lwd_retired_prompt_lens == {}
        assert engine._lwd_request_seqnos == {}
        assert engine._lwd_pending_seqnos == {}


def test_finish_without_pending_clears_retired_state_immediately():
    engine = _engine({"p": 4})
    outputs = {0: SimpleNamespace(finished_requests=["p"], outputs=[])}

    engine._lwd_retire_finished_outputs(outputs)
    assert engine._lwd_retired_req_ids == set()
    assert engine._lwd_retired_prompt_lens == {}
    assert engine._lwd_prompt_lens == {}


def test_worker_error_fails_engine_without_advancing_watermark(monkeypatch):
    from vllm_ascend.worker.lwd_cloud_model import lwd_pre_recv

    class DoneMq:
        def __init__(self):
            self.items = [("error", 9, "boom"), ("UP", 10)]

        def dequeue(self, timeout):
            if self.items:
                return self.items.pop(0)
            raise TimeoutError

    engine = _engine()
    engine._lwd_done_mq = DoneMq()
    failures = []
    engine._lwd_fail = failures.append
    watermark = []
    monkeypatch.setattr(
        lwd_pre_recv,
        "record_irecv_completions",
        lambda items: watermark.extend(items),
    )

    engine._lwd_drain_irecv_completions()
    assert len(failures) == 1
    assert "seqno=9" in failures[0] and "boom" in failures[0]
    assert watermark == []


@pytest.mark.parametrize("floor", [16, 32])
def test_prefix_readmission_self_hit_uses_frozen_start(floor):
    blocks = [SimpleNamespace(is_null=False) for _ in range(2)]
    request = _request(prompt=64, computed=0, preemptions=1, floor=floor)
    scheduler = _prefix_scheduler(request, (blocks,), hit=32, recv_rows=40)

    windows = scheduler._lwd_collect_prefill_windows(room=8)

    assert [(w.start, w.num_rows) for w in windows] == [(32, 8)]
    assert scheduler.lwd_prefix_reservations.hit_of("p") == 32
    scheduler.lwd_refill_sender.assert_not_called()
    scheduler.kv_cache_manager.block_pool.touch.assert_called_once_with(blocks)


def test_prefix_readmission_clamps_to_arrival_and_scheduler_block():
    blocks = [SimpleNamespace(is_null=False) for _ in range(2)]
    request = _request(prompt=64, computed=0, preemptions=1)
    scheduler = _prefix_scheduler(request, (blocks,), hit=32, recv_rows=24)

    windows = scheduler._lwd_collect_prefill_windows(room=8)

    assert [(w.start, w.num_rows) for w in windows] == [(16, 8)]
    scheduler.kv_cache_manager.block_pool.touch.assert_called_once_with(blocks[:1])


def test_prefix_readmission_refill_gates_then_done_advances_floor():
    blocks = [SimpleNamespace(is_null=False) for _ in range(2)]
    request = _request(prompt=64, computed=0, preemptions=1, floor=32)
    request.lwd_prefix_nonce = 11
    scheduler = _prefix_scheduler(request, (blocks,), hit=32, recv_rows=0)

    assert scheduler._lwd_collect_prefill_windows(room=32) == []
    scheduler.lwd_refill_sender.assert_called_once_with("p", 0, 32, 11)
    assert scheduler._lwd_refill_gate == {"p"}
    pending = scheduler._lwd_refill_pending["p"]
    assert (pending.hit_tokens, pending.prefix_nonce) == (0, 11)

    scheduler.lwd_refill_done("p", 10)
    assert scheduler._lwd_refill_gate == {"p"}
    assert scheduler._lwd_refill_pending == {"p": pending}
    scheduler.lwd_refill_done("p", 11)
    scheduler.lwd_recv_rows["p"] = 32
    windows = scheduler._lwd_collect_prefill_windows(room=32)
    assert request.lwd_resume_cap_tokens == 0
    assert scheduler._lwd_refill_gate == set()
    assert [(w.start, w.num_rows) for w in windows] == [(0, 32)]
    assert scheduler.kv_cache_manager.coordinator.find_longest_cache_hit.call_count == 1


def test_skip_reading_readmission_does_not_lookup():
    request = _request(prompt=32, computed=0, preemptions=1, skip=True)
    scheduler = _prefix_scheduler(request, (), hit=32, recv_rows=16)

    windows = scheduler._lwd_collect_prefill_windows(room=16)

    assert [(w.start, w.num_rows) for w in windows] == [(0, 16)]
    scheduler.kv_cache_manager.coordinator.find_longest_cache_hit.assert_not_called()
    scheduler.kv_cache_manager.block_pool.touch.assert_not_called()


def test_c2e_prefix_nonces_read_from_snapshot() -> None:
    request = _request()
    scheduler = _scheduler(request)

    assert scheduler._lwd_c2e_prefix_nonces(["p", "missing"], {"p": 7}) == [7, 0]


@pytest.mark.parametrize(("nonce", "finish_code"), [(7, 0), (9, 2)])
def test_update_c2e_preserves_nonce_after_finish_and_abort(
    monkeypatch, nonce, finish_code
) -> None:
    request = _request()
    request.lwd_prefix_nonce = nonce
    scheduler = _scheduler(request)
    scheduler.pending_c2e = []
    scheduler._lwd_diag = SimpleNamespace(observe_spec_acceptance=Mock())

    def finish_and_delete(self, scheduler_output, model_runner_output):
        self.requests.clear()
        output = SimpleNamespace(
            request_id="p", finish_reason=finish_code, new_token_ids=[]
        )
        return {0: SimpleNamespace(finished_requests={"p"}, outputs=[output])}

    monkeypatch.setattr(AsyncScheduler, "update_from_output", finish_and_delete)
    monkeypatch.setattr(LwdControlDiag, "disable_down_enabled", lambda: True)
    monkeypatch.setattr(LwdControlDiag, "disable_down_rows", lambda outputs: {"p": []})

    scheduler.update_from_output(
        SchedulerOutput.make_empty(), SimpleNamespace(lwd_down_carrier=None)
    )

    assert scheduler.requests == {}
    assert scheduler.pending_c2e[0].prefix_nonces == [nonce]
    assert scheduler.pending_c2e[0].finish_reasons == [finish_code]


def test_visible_set_releases_only_successful_prefix_allocation(monkeypatch):
    p1 = _request("p1")
    p2 = _request("p2")
    scheduler = _visible_set_scheduler([p1, p2], [])
    scheduler.lwd_prefix_reservations = Mock()

    def native_schedule(self, throttle_prefills=False):
        out = SchedulerOutput.make_empty()
        out.num_scheduled_tokens = {"p1": 4, "p2": 0}
        return out

    monkeypatch.setattr(AsyncScheduler, "schedule", native_schedule)
    scheduler._lwd_schedule(["p1", "p2"])

    scheduler.lwd_prefix_reservations.release.assert_called_once_with("p1")


def test_free_request_cleans_prefix_reservation_and_refill_gate(monkeypatch):
    request = _request()
    scheduler = _visible_set_scheduler([request], [])
    scheduler.lwd_prefix_reservations = Mock()
    scheduler._lwd_refill_gate = {"p"}
    scheduler._lwd_refill_pending = {"p": 16}
    monkeypatch.setattr(
        AsyncScheduler,
        "_free_request",
        lambda self, request, delay_free_blocks=False: (None, None),
    )

    assert scheduler._free_request(request) == (None, None)
    scheduler.lwd_prefix_reservations.release.assert_called_once_with("p")
    assert scheduler._lwd_refill_gate == set()
    assert scheduler._lwd_refill_pending == {}
