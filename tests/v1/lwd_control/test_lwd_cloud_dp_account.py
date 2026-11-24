# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LwdCloudDpAccount: the per-cloud-DP bookkeeping of the edge scheduler.

Pure logic, no engine and no scheduler: the checks cover the pending-batch
ledger and the gate predicate.
"""

from __future__ import annotations

from vllm.v1.lwd_control.control_edge_scheduler.lwd_cloud_dp_account import (
    LWD_MAX_PENDING_CHUNKS,
    LwdCloudDpAccount,
)


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
    assert account.is_gated(2, has_resume_inflight=False, num_running_reqs=0) is None

    # Slots full (per-cloud-DP billing is the caller's side of the
    # contract; the account only compares against slot_cap).
    assert account.is_gated(2, has_resume_inflight=False, num_running_reqs=2) == "slots"
    # A resume in flight is exempt: finish in-flight chunks first.
    assert account.is_gated(2, has_resume_inflight=True, num_running_reqs=2) is None

    # Pending chunks at the cap gate the engine even with free slots.
    other = LwdCloudDpAccount(1)
    for seqno in range(LWD_MAX_PENDING_CHUNKS):
        other.register_batch(seqno, {f"r{seqno}"})
    assert (
        other.is_gated(100, has_resume_inflight=False, num_running_reqs=0) == "chunks"
    )


def test_gate_uses_account_pending_chunk_limit() -> None:
    account = LwdCloudDpAccount(0, max_pending_chunks=1)
    account.register_batch(0, {"r"})

    assert account.max_pending_chunks == 1
    assert (
        account.is_gated(100, has_resume_inflight=False, num_running_reqs=0) == "chunks"
    )
