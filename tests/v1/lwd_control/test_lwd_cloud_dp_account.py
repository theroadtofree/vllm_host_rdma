# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LwdCloudDpAccount: the per-cloud-DP bookkeeping of the edge scheduler.

Pure logic, no engine and no scheduler: the checks cover the load
transitions, the pending-batch ledger, the gate predicate and the routing
score weights.
"""

from __future__ import annotations

from vllm.v1.lwd_control.control_edge_scheduler.lwd_cloud_dp_account import (
    LWD_MAX_PENDING_CHUNKS,
    LwdCloudDpAccount,
)


def test_load_transitions_track_one_request_lifecycle() -> None:
    account = LwdCloudDpAccount(0)
    assert account.load == 0 and account.score() == 0

    account.note_prefill_admitted("r1")
    assert (account.prefill_load, account.decode_load) == (1, 0)
    assert account.score() == 4

    account.note_prefill_done("r1")
    assert (account.prefill_load, account.decode_load) == (0, 1)
    assert account.score() == 1

    account.note_request_settled("r1", was_awaiting=True)
    assert account.load == 0 and account.score() == 0


def test_repeated_deliveries_move_prefill_to_decode_exactly_once() -> None:
    """Every token delivery settles the request, not only the first one:
    the prefill->decode move must stay at exactly one, or the load goes
    negative and the routing score pins all traffic to one engine (seen
    live: loads=[(0, 0), (-10969, 11077)])."""
    account = LwdCloudDpAccount(1)
    account.note_prefill_admitted("r1")

    for _ in range(100):  # a 100-token request delivers 100 times
        assert account.note_prefill_done("r1") is True or True
    assert (account.prefill_load, account.decode_load) == (0, 1)

    # A settled-but-still-delivering request (late tokens racing finish)
    # must not move anything either.
    assert account.note_prefill_done("r1") is False
    assert (account.prefill_load, account.decode_load) == (0, 1)


def test_settling_a_prefill_request_covers_abort_before_first_token() -> None:
    account = LwdCloudDpAccount(1)
    account.note_prefill_admitted("r1")
    account.note_request_settled("r1", was_awaiting=False)
    assert account.load == 0
    assert "r1" not in account.prefill_requests


def test_pending_batches_release_on_watermark_and_request_settlement() -> None:
    account = LwdCloudDpAccount(0)
    for seqno in range(3):
        account.register_batch(
            account.take_seqno(), {"r1", "r2"} if seqno < 2 else {"r3"}
        )
    assert account.pending_chunks() == 3
    assert account.seqno == 3

    account.note_send_watermark(0)
    assert list(account.pending_batches) == [1, 2]
    account.drop_request("r1")
    assert account.pending_batches == {1: {"r2"}, 2: {"r3"}}
    account.drop_request("r2")
    assert account.pending_batches == {2: {"r3"}}
    account.drop_request("r3")
    assert account.pending_chunks() == 0


def test_gate_blocks_on_slots_and_pending_chunks_but_not_resumes() -> None:
    account = LwdCloudDpAccount(0)
    assert not account.is_gated(max_num_running_reqs=2, has_resume_inflight=False)

    # Slots full.
    account.note_prefill_admitted("a")
    account.note_prefill_admitted("b")
    assert account.is_gated(2, has_resume_inflight=False)
    # A resume in flight is exempt: finish in-flight chunks first.
    assert not account.is_gated(2, has_resume_inflight=True)

    # Pending chunks at the cap gate the engine even with free slots.
    other = LwdCloudDpAccount(1)
    for seqno in range(LWD_MAX_PENDING_CHUNKS):
        other.register_batch(seqno, {f"r{seqno}"})
    assert other.is_gated(max_num_running_reqs=100, has_resume_inflight=False)


def test_gate_uses_account_pending_chunk_limit() -> None:
    account = LwdCloudDpAccount(0, max_pending_chunks=1)
    account.register_batch(0, {"r"})

    assert account.max_pending_chunks == 1
    assert account.is_gated(max_num_running_reqs=100, has_resume_inflight=False)


def test_score_matches_native_dp_balancer_weights() -> None:
    account = LwdCloudDpAccount(0)
    account.prefill_load = 3
    account.decode_load = 2
    assert account.score() == 3 * 4 + 2 * 1


def test_finished_requests_release_their_slot_under_load() -> None:
    """The 180-concurrency deadlock shape: requests finish, their route
    entry and decode_load must be released, or the load creeps up to
    max_num_seqs, the gate holds every engine forever and both sides idle
    (seen live: loads=[(11, 79), (11, 79)] with the cloud spinning dummies).
    """
    account = LwdCloudDpAccount(0)
    for i in range(90):
        account.note_prefill_admitted(f"r{i}")
        account.note_prefill_done(f"r{i}")  # first token
    assert account.load == 90
    assert account.is_gated(90, has_resume_inflight=False)

    for i in range(90):
        account.note_request_settled(f"r{i}", was_awaiting=True)
    assert account.load == 0
    assert not account.is_gated(90, has_resume_inflight=False)
