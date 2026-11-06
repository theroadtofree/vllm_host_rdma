# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Arrival-driven cloud prefill/decode scheduling.

RangeNotify only drives early data-plane receives. The scheduler opens a
per-request window from the cumulative arrived-row high watermark and the
native computed-token watermark, then lets the native scheduler allocate and
account for the mixed batch. Prefix hits use short-lived reservations and
frozen snapshots. Native preemption resets the computed-token watermark while
the arrived-row high watermark is retained for readmission.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from vllm.lwd_logger import lwd_logger
from vllm.v1.core.sched.output import (
    LwdBatch,
    LwdBatchType,
    LwdEmbedBatch,
    SchedulerOutput,
)
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_prefix_cache import (
    prefix_start,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
    LwdReqPhase,
    lwd_draft_slots,
)
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_NOT_FINISHED,
    LwdC2eNotify,
)
from vllm.v1.lwd_control.lwd_control_diag import (
    LwdCloudAdmissionMeter,
    LwdControlDiag,
)
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.outputs import ModelRunnerOutput

LwdDownCarrier = tuple[torch.Tensor, list[str], int, int]
"""Finished ModelRunnerOutput.lwd_down_carrier harvested by the executor
output thread."""


@dataclass
class LwdC2eMeta:
    """Step metadata unpacked from the cloud worker's DOWN carrier."""

    hidden_num_elements: int
    top_id_ths: list[list[int]]
    num_accepted_tokens: list[int]
    req_ids: list[str]
    down_seqno: int


@dataclass(frozen=True)
class _LwdPrefillWindow:
    request: Request
    num_rows: int
    start: int


class _LwdRefillPending(NamedTuple):
    hit_tokens: int
    prefix_nonce: int


class LwdCloudScheduler(LwdBaseScheduler):
    """Mix all decode requests with prefill windows whose rows have arrived."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lwd_recv_rows: dict[str, int] = {}
        self._lwd_dropped: set[str] = set()
        self._lwd_abort_pending: list[str] = []
        self._lwd_prefix_enabled = bool(
            self.vllm_config.cache_config.enable_prefix_caching
        )
        self._lwd_refill_gate: set[str] = set()
        self._lwd_refill_pending: dict[str, _LwdRefillPending] = {}
        self._lwd_diag = LwdControlDiag()
        self._lwd_admission_meter = LwdCloudAdmissionMeter()
        # Per-step terms the meter reads, set by the step that computes
        # them: the drafting reservation and the queue it was charged
        # against, and the two reasons a candidate yielded no window.
        self._lwd_reserved = 0
        self._lwd_queue_depth = 0
        self._lwd_blocked: tuple[int, int] = (0, 0)
        self.pending_c2e: list[LwdC2eNotify] = []
        # Input slots the native step charges per scheduled request for
        # parallel drafting, on top of its tokens; the sole home of the
        # formula is lwd_base_scheduler.lwd_draft_slots.
        self._lwd_draft_slots: int = lwd_draft_slots(self.vllm_config)
        lwd_logger.info(
            "[cloud-sched] arrival-driven prefill/decode scheduling (draft_slots=%d)",
            self._lwd_draft_slots,
        )

    def has_requests(self) -> bool:
        """Keep the engine alive until an abort-only worker notice is sent."""
        return bool(self._lwd_abort_pending) or super().has_requests()

    def _lwd_schedule_step(self) -> SchedulerOutput:
        decode_ids = self._lwd_collect_decode_requests()
        decode_exact = self._lwd_compute_decode_consumption(decode_ids)
        room = self._lwd_prefill_room(decode_ids, decode_exact)
        windows = self._lwd_collect_prefill_windows(room)
        if windows:
            out = self._lwd_dispatch_windows(windows, decode_ids, decode_exact)
        else:
            out = self._lwd_schedule(decode_ids)
        if windows:
            self._lwd_verify_w2(out, windows, decode_exact)
            self._lwd_warn_under_scheduled(out, windows)
        self._lwd_meter_admission(out, windows, room)
        self._lwd_attach_request_metadata(out)
        self._lwd_attach_abort_notice(out)
        return out

    def _lwd_meter_admission(
        self, out: SchedulerOutput, windows: list[_LwdPrefillWindow], room: int
    ) -> None:
        """Feed the admission meter what this step offered and took.

        The admitted count comes from the scheduler output, not from the
        windows: the native step can cut a window short, and a window
        asked for is not a row run.
        """
        admitted = sum(
            out.num_scheduled_tokens.get(window.request.request_id, 0)
            for window in windows
        )
        self._lwd_admission_meter.note_step(
            room=room,
            admitted=admitted,
            num_windows=len(windows),
            blocked=self._lwd_blocked,
            queue_depth=self._lwd_queue_depth,
            reserved=self._lwd_reserved,
        )

    def _lwd_prefill_room(self, decode_ids: list[str], decode_exact: int) -> int:
        """Prefill rows this step may admit, after the drafting slots.

        The native step charges a second budget besides its token budget:
        every scheduled request also spends `max_num_new_slots_for_drafting`
        input slots, and caps each request at what is left of them. Rows that
        ignore those slots are cut short by the native step, which costs a
        retry per step, so they are reserved up front.

        Windows are not collected yet, so prefills are charged at the number
        of requests that could still take one. That over-reserves when fewer
        of them have arrivals ready, which only defers rows to the next step;
        under-reserving is what causes the native truncation.

        The charge counts the queues, so it grows with backlog rather than
        with what the step can run: `_lwd_reserved` and `_lwd_queue_depth`
        record the two terms for the admission meter, which is what tells a
        budget that is merely small from one the backlog ate.
        """
        room = self.max_num_scheduled_tokens - decode_exact
        self._lwd_reserved = 0
        self._lwd_queue_depth = 0
        if not self._lwd_draft_slots:
            return room
        decoding = set(decode_ids)
        num_prefill = sum(
            1
            for request in itertools.chain(
                self.running, self.waiting, self.skipped_waiting
            )
            if request.request_id not in decoding
            and request.request_id not in self._lwd_dropped
        )
        self._lwd_queue_depth = num_prefill
        self._lwd_reserved = (len(decode_ids) + num_prefill) * self._lwd_draft_slots
        return room - self._lwd_reserved

    def _lwd_collect_decode_requests(self) -> list[str]:
        return [
            request.request_id
            for request in self._lwd_reqs_in_phase(
                LwdReqPhase.DECODE,
                self.running,
                self.waiting,
                self.skipped_waiting,
            )
        ]

    def _lwd_compute_decode_consumption(self, decode_ids: list[str]) -> int:
        """Mirror native decode guards and placeholder accounting exactly."""
        decode_exact = 0
        for request_id in decode_ids:
            request = self.requests[request_id]
            placeholders = request.num_output_placeholders
            if (
                placeholders > 0
                and request.num_computed_tokens + 2 - placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                continue
            if self.current_step < request.next_decode_eligible_step:
                continue
            decode_exact += max(
                0,
                min(
                    request.num_tokens_with_spec
                    + placeholders
                    - request.num_computed_tokens,
                    self.max_model_len
                    - self.num_sampled_tokens_per_step
                    - request.num_computed_tokens,
                ),
            )
        return decode_exact

    def _lwd_prep_prefix_readmission(self, request: Request) -> bool:
        """Freeze a preempted request's arrival-clamped hit before admission."""
        reservations = self.lwd_prefix_reservations
        floor = getattr(request, "lwd_resume_cap_tokens", 0)
        if request.skip_reading_prefix_cache:
            groups, h_prime = (), 0
        else:
            groups, h_prime, _ = (
                self.kv_cache_manager.coordinator.find_longest_cache_hit(
                    request.block_hashes,
                    max_cache_hit_length=max(0, request.num_prompt_tokens - 1),
                )
            )

        recv_rows = self.lwd_recv_rows.get(request.request_id, 0)
        alignment = self.kv_cache_manager.coordinator.scheduler_block_size
        clamped_hit = min(h_prime, recv_rows // alignment * alignment)
        if clamped_hit < h_prime:
            lwd_logger.info(
                "[cloud-sched] clamp readmission req=%s hit=%d -> %d "
                "recv_rows=%d alignment=%d",
                request.request_id,
                h_prime,
                clamped_hit,
                recv_rows,
                alignment,
            )
            groups = reservations.truncate(groups, clamped_hit)
            h_prime = clamped_hit

        reservations.reserve_hit(request, groups, h_prime)
        if h_prime >= floor:
            return True

        sender = getattr(self, "lwd_refill_sender", None)
        if sender is None:
            raise RuntimeError("[LWD] refill channel is not wired")
        prefix_nonce = getattr(request, "lwd_prefix_nonce", 0)
        self._lwd_refill_gate.add(request.request_id)
        self._lwd_refill_pending[request.request_id] = _LwdRefillPending(
            h_prime, prefix_nonce
        )
        lwd_logger.info(
            "[cloud-sched] refill req=%s hit=%d floor=%d nonce=%d",
            request.request_id,
            h_prime,
            floor,
            prefix_nonce,
        )
        sender(request.request_id, h_prime, floor, prefix_nonce)
        return False

    def lwd_refill_done(self, request_id: str, prefix_nonce: int = 0) -> None:
        """Lift only the gate belonging to this exact request generation."""
        pending = self._lwd_refill_pending.get(request_id)
        request = self.requests.get(request_id)
        if pending is None:
            lwd_logger.warning(
                "[cloud-sched] stale refill_done req=%s ignored", request_id
            )
            return
        current_nonce = getattr(request, "lwd_prefix_nonce", 0) if request else 0
        if prefix_nonce != pending.prefix_nonce or (
            request is not None and current_nonce != pending.prefix_nonce
        ):
            lwd_logger.warning(
                "[cloud-sched] stale refill_done req=%s nonce=%d ignored: "
                "pending=%d current=%d",
                request_id,
                prefix_nonce,
                pending.prefix_nonce,
                current_nonce,
            )
            return
        self._lwd_refill_pending.pop(request_id)
        self._lwd_refill_gate.discard(request_id)
        if request is not None:
            request.lwd_resume_cap_tokens = pending.hit_tokens

    def _lwd_collect_prefill_windows(self, room: int) -> list[_LwdPrefillWindow]:
        windows: list[_LwdPrefillWindow] = []
        # (partial window, no arrival): the two ways a candidate that
        # still owes prompt rows yields no window this step. Their cure
        # is opposite -- a partial window waits on more rows of a prompt
        # already arriving, no arrival waits on the transfer starting --
        # so the meter keeps them apart.
        self._lwd_blocked = (0, 0)
        reservations = getattr(self, "lwd_prefix_reservations", None)
        prefix_enabled = getattr(self, "_lwd_prefix_enabled", False)
        for request in itertools.chain(
            self.running, self.waiting, self.skipped_waiting
        ):
            if room <= 0:
                break
            request_id = request.request_id
            if request_id in self._lwd_dropped or request.is_finished():
                continue
            start = request.num_computed_tokens
            if prefix_enabled:
                if request_id in self._lwd_refill_gate:
                    continue
                if start == 0:
                    if (
                        request.num_preemptions > 0
                        and reservations is not None
                        and not reservations.has(request_id)
                        and not self._lwd_prep_prefix_readmission(request)
                    ):
                        continue
                    start = prefix_start(request, reservations)
            if start >= request.num_prompt_tokens:
                continue
            recv_rows = self.lwd_recv_rows.get(request_id, 0)
            arrived = recv_rows - start
            if arrived <= 0:
                partial, no_arrival = self._lwd_blocked
                self._lwd_blocked = (partial, no_arrival + 1)
                continue
            num_rows = min(arrived, room)
            if num_rows < room and recv_rows < request.num_prompt_tokens:
                partial, no_arrival = self._lwd_blocked
                self._lwd_blocked = (partial + 1, no_arrival)
                continue
            windows.append(_LwdPrefillWindow(request, num_rows, start))
            room -= num_rows
        return windows

    def _lwd_dispatch_windows(
        self,
        windows: list[_LwdPrefillWindow],
        decode_ids: list[str],
        decode_exact: int,
    ) -> SchedulerOutput:
        total_windows = sum(window.num_rows for window in windows)
        request_ids = decode_ids + [window.request.request_id for window in windows]
        for window in windows:
            window.request.lwd_window_cap = window.num_rows
        try:
            return self._lwd_schedule(
                request_ids,
                token_budget_cap=decode_exact + total_windows,
            )
        finally:
            for window in windows:
                window.request.lwd_window_cap = None

    def _lwd_verify_w2(
        self,
        out: SchedulerOutput,
        windows: list[_LwdPrefillWindow],
        decode_exact: int,
    ) -> None:
        for window in windows:
            request_id = window.request.request_id
            scheduled = out.num_scheduled_tokens.get(request_id, 0)
            if scheduled == 0:
                continue
            recv_rows = self.lwd_recv_rows.get(request_id, 0)
            if scheduled > window.num_rows or window.start + scheduled > recv_rows:
                raise RuntimeError(
                    f"[LWD] W2 violation req={request_id}: scheduled rows "
                    f"[{window.start}, {window.start + scheduled}) with "
                    f"window={window.num_rows}, recv_rows={recv_rows}, "
                    f"decode_exact={decode_exact}"
                )

    @staticmethod
    def _lwd_warn_under_scheduled(
        out: SchedulerOutput, windows: list[_LwdPrefillWindow]
    ) -> None:
        requested = sum(window.num_rows for window in windows)
        scheduled = sum(
            out.num_scheduled_tokens.get(window.request.request_id, 0)
            for window in windows
        )
        if scheduled < requested:
            lwd_logger.warning(
                "[cloud-sched] native under-scheduled prefill=%d < windows=%d; "
                "the remainder is retried next step",
                scheduled,
                requested,
            )

    def _lwd_attach_request_metadata(self, out: SchedulerOutput) -> None:
        """Hand the whole-request values the edge announced to the runner."""
        for data in out.scheduled_new_reqs:
            request = self.requests.get(data.req_id)
            if request is None:
                continue
            positions = getattr(request, "lwd_mrope_positions", b"")
            data.lwd_mrope_positions = b"" if positions is None else positions
            spans = getattr(request, "lwd_image_placeholder_spans", None)
            data.lwd_image_placeholder_spans = list(spans) if spans else []

    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: RequestStatus,
    ) -> list[Request]:
        if finished_status == RequestStatus.FINISHED_ABORTED:
            if request_ids is None:
                request_ids = list(self.requests)
            elif isinstance(request_ids, str):
                request_ids = (request_ids,)
            else:
                request_ids = tuple(request_ids)
            for request_id in request_ids:
                self._lwd_drop_request(request_id)
        return super().finish_requests(request_ids, finished_status)

    def _lwd_drop_request(self, request_id: str) -> None:
        if request_id in self._lwd_dropped:
            return
        self._lwd_dropped.add(request_id)
        self._lwd_abort_pending.append(request_id)
        request = self.requests.get(request_id)
        if request is not None:
            remaining = max(request.num_prompt_tokens - request.num_computed_tokens, 0)
            lwd_logger.info(
                "[cloud-sched] abort dropped req=%s remaining_rows=%d",
                request_id,
                remaining,
            )

    def _lwd_attach_abort_notice(self, out: SchedulerOutput) -> None:
        pending = self._lwd_abort_pending
        if not pending:
            return
        batch = out.lwd_batch
        if batch is None:
            out.lwd_batch = LwdBatch(
                batch_type=LwdBatchType.LWD_EMBED,
                seqno=-1,
                batch_meta=LwdEmbedBatch(aborted_req_ids=list(pending)),
            )
        elif (
            batch.batch_type is LwdBatchType.LWD_EMBED and batch.batch_meta is not None
        ):
            existing = list(batch.batch_meta.aborted_req_ids or ())
            batch.batch_meta.aborted_req_ids = existing + [
                request_id for request_id in pending if request_id not in existing
            ]
        else:
            return
        self._lwd_abort_pending = []

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        nonce_snapshot = {
            request_id: getattr(request, "lwd_prefix_nonce", 0)
            for request_id, request in self.requests.items()
        }
        self._lwd_diag.observe_spec_acceptance(
            scheduler_output, model_runner_output, self.requests
        )
        engine_core_outputs = super().update_from_output(
            scheduler_output, model_runner_output
        )
        self._lwd_cleanup_arrivals(engine_core_outputs)
        carrier = getattr(model_runner_output, "lwd_down_carrier", None)
        if carrier is not None:
            self._lwd_append_carrier_c2e(carrier, engine_core_outputs, nonce_snapshot)
        elif LwdControlDiag.disable_down_enabled():
            self._lwd_append_diag_c2e(engine_core_outputs, nonce_snapshot)
        return engine_core_outputs

    def _lwd_cleanup_arrivals(
        self, engine_core_outputs: dict[int, EngineCoreOutputs]
    ) -> None:
        for outputs in engine_core_outputs.values():
            for request_id in outputs.finished_requests or ():
                self.lwd_recv_rows.pop(request_id, None)
                self._lwd_dropped.discard(request_id)

    def _lwd_c2e_prefix_nonces(
        self, request_ids: list[str], nonce_snapshot: dict[str, int]
    ) -> list[int]:
        return [nonce_snapshot.get(request_id, 0) for request_id in request_ids]

    def _lwd_append_carrier_c2e(
        self,
        carrier: LwdDownCarrier,
        engine_core_outputs: dict[int, EngineCoreOutputs],
        nonce_snapshot: dict[str, int],
    ) -> None:
        meta = self._lwd_carrier_to_meta(carrier)
        committed = {
            out.request_id: len(out.new_token_ids)
            for outputs in engine_core_outputs.values()
            for out in outputs.outputs
        }
        meta.num_accepted_tokens = [committed.get(rid, 0) for rid in meta.req_ids]
        LwdControlDiag.cloud_step(self, meta, engine_core_outputs)
        finish_reasons = self._lwd_c2e_finish_reasons(meta.req_ids, engine_core_outputs)
        LwdControlDiag.trace_cloud_c2e_rows(
            meta.down_seqno,
            meta.req_ids,
            meta.num_accepted_tokens,
            finish_reasons,
        )
        token_ids: list[list[int]] = []
        if LwdControlDiag.token_passthrough_enabled():
            token_ids = LwdControlDiag.bypass_token_payload(
                meta.req_ids, engine_core_outputs
            )
        self.pending_c2e.append(
            LwdC2eNotify(
                hidden_num_elements=meta.hidden_num_elements,
                top_id_ths=meta.top_id_ths,
                num_accepted_tokens=meta.num_accepted_tokens,
                req_ids=meta.req_ids,
                finish_reasons=finish_reasons,
                down_seqno=meta.down_seqno,
                token_ids=token_ids,
                prefix_nonces=self._lwd_c2e_prefix_nonces(meta.req_ids, nonce_snapshot),
            )
        )

    def _lwd_append_diag_c2e(
        self,
        engine_core_outputs: dict[int, EngineCoreOutputs],
        nonce_snapshot: dict[str, int],
    ) -> None:
        rows = LwdControlDiag.disable_down_rows(engine_core_outputs)
        if not rows:
            return
        request_ids = list(rows)
        self.pending_c2e.append(
            LwdC2eNotify(
                hidden_num_elements=0,
                top_id_ths=[],
                num_accepted_tokens=[],
                req_ids=request_ids,
                finish_reasons=self._lwd_c2e_finish_reasons(
                    request_ids, engine_core_outputs
                ),
                down_seqno=-1,
                token_ids=[rows[request_id] for request_id in request_ids],
                prefix_nonces=self._lwd_c2e_prefix_nonces(request_ids, nonce_snapshot),
            )
        )

    @staticmethod
    def _lwd_carrier_to_meta(carrier: LwdDownCarrier) -> LwdC2eMeta:
        packed, request_ids, hidden_numel, seqno = carrier
        num_requests = len(request_ids)
        values = packed.tolist()
        counts = values[-2 * num_requests : -num_requests]
        segment_lengths = values[-num_requests:]
        ranks_flat = values[: -2 * num_requests]
        top_id_ths: list[list[int]] = []
        offset = 0
        for segment_length in segment_lengths:
            top_id_ths.append(ranks_flat[offset : offset + segment_length])
            offset += segment_length
        return LwdC2eMeta(
            hidden_num_elements=hidden_numel,
            top_id_ths=top_id_ths,
            num_accepted_tokens=list(counts),
            req_ids=list(request_ids),
            down_seqno=seqno,
        )

    @staticmethod
    def _lwd_c2e_finish_reasons(
        request_ids: list[str], engine_core_outputs: dict[int, EngineCoreOutputs]
    ) -> list[int]:
        from vllm.v1.engine import FinishReason

        reasons: dict[str, int] = {}
        for outputs in engine_core_outputs.values():
            for out in outputs.outputs:
                if out.finish_reason is not None:
                    reasons.setdefault(out.request_id, int(out.finish_reason))
            for request_id in outputs.finished_requests or ():
                reasons.setdefault(request_id, int(FinishReason.ABORT))
        return [reasons.get(request_id, LWD_NOT_FINISHED) for request_id in request_ids]
