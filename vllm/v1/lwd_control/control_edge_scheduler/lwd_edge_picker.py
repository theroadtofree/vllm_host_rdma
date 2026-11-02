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
    single_prefill_min_tokens: int = 0,
    slot_overhead: int = 0,
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

    single_prefill_min_tokens: when > 0, a request whose total prompt
    length reaches this value takes the prefill batch exclusively
    (resumes included; no other entries are packed before or after it in
    the batch); 0 = disabled (DeepSeek V4 passes 8000, other models pass 0).

    slot_overhead: budget each new start consumes on top of its chunk (the
    cloud's per-request parallel-drafting slots; resumes are in flight, so
    the caller's cap already reserves theirs); 0 = none.

    """
    picked: list[str] = []
    expected: dict[str, int] = {}
    if cap < 1:
        return picked, expected
    budget = _LwdPickBudget(cap, slot_overhead)
    single_prefill = _pick_resumes(
        running, picked, expected, budget, chunk_unit, single_prefill_min_tokens
    )
    _pick_new_requests(
        waiting,
        picked,
        expected,
        budget,
        free_seq,
        chunk_unit,
        single_prefill_min_tokens,
        single_prefill,
    )
    return picked, expected


class _LwdPickBudget:
    """Batch budget being filled: each new start consumes its chunk plus
    `slot_overhead`, each resume its chunk."""

    def __init__(self, cap: int, slot_overhead: int) -> None:
        self.cap = cap
        self.slot_overhead = slot_overhead
        self.used = 0


def _is_single_prefill(single_prefill_min_tokens: int, num_prompt_tokens: int) -> bool:
    """Whether this request triggers the exclusive rule (total prompt
    length reaches the threshold)."""
    return (
        single_prefill_min_tokens > 0 and num_prompt_tokens >= single_prefill_min_tokens
    )


def _append_chunk(
    picked: list[str],
    expected: dict[str, int],
    budget: _LwdPickBudget,
    request_id: str,
    chunk: int,
    overhead: int,
) -> bool:
    """Pack one chunk, consuming chunk + overhead of the budget; when that
    exceeds the budget, fall back to tail truncation (c = cap − used −
    overhead; the entry joins only if c ≥ 1, and the truncated piece still
    consumes budget). A truncation closes the batch. Returns whether the
    batch is closed."""
    if budget.used + chunk + overhead > budget.cap:
        c = budget.cap - budget.used - overhead
        if c >= 1:
            picked.append(request_id)
            expected[request_id] = c
            budget.used += c + overhead
        return True
    picked.append(request_id)
    expected[request_id] = chunk
    budget.used += chunk + overhead
    return False


def _pick_resumes(
    running: Sequence[LwdPrefillCandidate],
    picked: list[str],
    expected: dict[str, int],
    budget: _LwdPickBudget,
    chunk_unit: int,
    single_prefill_min_tokens: int,
) -> bool:
    """Resume loop: walk requests in running order, each taking
    c = min(remaining, C). A resume is in flight already, so the caller's
    cap has reserved its slot overhead.

    Exclusive rule: a qualifying request sets the single_prefill flag; if
    the batch already has entries, close it on the spot (long requests
    are never batched with short ones); the flag carries across loops
    (this batch closes, and the new-start loop packs nothing more).
    Returns the single_prefill flag."""
    single_prefill = False
    for cand in running:
        if single_prefill and picked:
            break
        remaining = cand.num_prompt_tokens - cand.num_computed_tokens
        if remaining <= 0:
            continue
        # The exclusive check uses total prompt length, not this chunk's
        # remaining tokens: when a long request arrives later, close the
        # current batch first and leave the next batch to it alone, so
        # waiting new-starts cannot bypass it
        single_prefill = _is_single_prefill(
            single_prefill_min_tokens, cand.num_prompt_tokens
        )
        if single_prefill and picked:
            break
        batch_closed = _append_chunk(
            picked,
            expected,
            budget,
            cand.request_id,
            min(remaining, chunk_unit),
            overhead=0,
        )
        if batch_closed:
            break
    return single_prefill


def _pick_new_requests(
    waiting: Sequence[LwdPrefillCandidate],
    picked: list[str],
    expected: dict[str, int],
    budget: _LwdPickBudget,
    free_seq: int,
    chunk_unit: int,
    single_prefill_min_tokens: int,
    single_prefill: bool,
) -> None:
    """New-start loop: take the FCFS prefix of the waiting order,
    c = min(prompt, C), new-start count ≤ free_seq; never skip the queue
    head (preserves FCFS and seqno chain order). Each new start also
    consumes the budget's slot overhead. single_prefill is passed in from
    the resume loop (True means this batch is already closed)."""
    for new_count, cand in enumerate(waiting):
        if single_prefill and picked:
            break
        single_prefill = _is_single_prefill(
            single_prefill_min_tokens, cand.num_prompt_tokens
        )
        if single_prefill and picked:
            break
        remaining = cand.num_prompt_tokens - cand.num_computed_tokens
        if remaining <= 0:
            continue
        chunk = min(remaining, chunk_unit)
        if new_count >= free_seq:
            break
        batch_closed = _append_chunk(
            picked,
            expected,
            budget,
            cand.request_id,
            chunk,
            overhead=budget.slot_overhead,
        )
        if batch_closed:
            break
