# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure functions for edge-side prefill batch assembly: resume-first +
FCFS prefix + tail truncation (PD_mix is the only supported form; the
single-request escape hatch for PD-disaggregated mode was removed along
with the phase machine).

chunk semantics follow lwd_chunked_prefill_design.md §3.1; called by the
LwdEdgeScheduler adapter layer after it assembles candidates. Inputs and
outputs are all plain values with no scheduler state, so the logic is
table-driven testable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple


class LwdPrefillCandidate(NamedTuple):
    """Batch assembly candidate: the scheduler distills a Request into
    these three fields at the call site."""

    request_id: str
    num_prompt_tokens: int
    num_computed_tokens: int


def lwd_pick_prefill_batch(
    running: Sequence[LwdPrefillCandidate],
    waiting: Sequence[LwdPrefillCandidate],
    *,
    cap: int,
    free_seq: int,
    chunk_unit: int,
) -> tuple[list[str], dict[str, int]]:
    """Pick the next embed batch, returning (req_ids, expected).

    running holds resume candidates (in running order: finish in-flight
    requests before starting new ones); waiting holds new-start candidates
    (the caller pre-joins skipped-first + the FCFS order of waiting).
    chunk_unit is the chunk unit C; each piece self-limits to
    c = min(remaining, C). cap < 1 returns an empty batch; the number of
    new requests is clamped by free_seq; tail truncation: the last entry
    in the batch may be cut short by the remaining budget (cap − used),
    and a truncation closes the batch.
    """
    picked: list[str] = []
    expected: dict[str, int] = {}
    if cap < 1:
        return picked, expected
    used = _pick_resumes(running, picked, expected, cap, chunk_unit)
    _pick_new_requests(
        waiting,
        picked,
        expected,
        used,
        cap,
        free_seq,
        chunk_unit,
    )
    return picked, expected


def _append_chunk(
    picked: list[str],
    expected: dict[str, int],
    used: int,
    cap: int,
    request_id: str,
    chunk: int,
) -> tuple[int, bool]:
    """Pack one chunk into the batch; when used + chunk exceeds the
    budget, fall back to tail truncation (remaining budget c = cap − used;
    the entry joins only if c ≥ 1, and the truncated piece still consumes
    budget). A truncation closes the batch. Returns (new used, whether
    the batch is closed)."""
    if used + chunk > cap:
        c = cap - used
        if c >= 1:
            picked.append(request_id)
            expected[request_id] = c
            used += c
        return used, True
    picked.append(request_id)
    expected[request_id] = chunk
    return used + chunk, False


def _pick_resumes(
    running: Sequence[LwdPrefillCandidate],
    picked: list[str],
    expected: dict[str, int],
    cap: int,
    chunk_unit: int,
) -> int:
    """Resume loop: walk requests in running order, each taking
    c = min(remaining, C). Returns used."""
    used = 0
    for cand in running:
        remaining = cand.num_prompt_tokens - cand.num_computed_tokens
        if remaining <= 0:
            continue
        used, batch_closed = _append_chunk(
            picked,
            expected,
            used,
            cap,
            cand.request_id,
            min(remaining, chunk_unit),
        )
        if batch_closed:
            break
    return used


def _pick_new_requests(
    waiting: Sequence[LwdPrefillCandidate],
    picked: list[str],
    expected: dict[str, int],
    used: int,
    cap: int,
    free_seq: int,
    chunk_unit: int,
) -> int:
    """New-start loop: take the FCFS prefix of the waiting order,
    c = min(prompt, C), new-start count ≤ free_seq; never skip the queue
    head (preserves FCFS and seqno chain order). Returns used."""
    for new_count, cand in enumerate(waiting):
        remaining = cand.num_prompt_tokens - cand.num_computed_tokens
        if remaining <= 0:
            continue
        chunk = min(remaining, chunk_unit)
        if new_count >= free_seq:
            break
        used, batch_closed = _append_chunk(
            picked, expected, used, cap, cand.request_id, chunk
        )
        if batch_closed:
            break
    return used
