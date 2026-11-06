# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prefill admission meters: why a loaded deployment admits prefill
slower than it could.

Throughput loss that lands entirely on TTFT -- unchanged TPOT, unchanged
concurrency, end-to-end latency growing by exactly what the first token
grew by -- means prompts are not reaching the cloud's batches, not that
the batches run slow. Three questions separate the causes, and each
meter answers exactly one:

  - cloud budget: of the room a step had for prefill, how much did it
    use? Rows the budget refused and rows no arrival could fill look the
    same in the scheduler output and have opposite fixes.
  - cloud arrival: how long from the hint that opens a window to the
    rows landing in host memory? This is the staging hop neither
    scheduler can see and no existing counter times.
  - edge dispatch: of the steps that produced no EMBED batch, which gate
    term stopped them? A gate that holds on seats is a cloud that is
    full; one that holds on chunks is a data plane that is behind.

Where the resulting latency lands, rather than which gate refused it, is
`lwd_ttft_meter`; the interval gate and reporting conventions both share
are in `lwd_meter_base`.
"""

from __future__ import annotations

import time

from vllm.logger import init_logger
from vllm.v1.lwd_control.lwd_control_diag.lwd_meter_base import LwdIntervalMeter

logger = init_logger(__name__)


class LwdCloudAdmissionMeter(LwdIntervalMeter):
    """Cloud prefill admission: budget offered, budget taken, and what
    stood between them.

    `room` is what the step could still spend on prefill after decode and
    the drafting-slot reservation; `admitted` is what the windows
    actually took. Their ratio is the whole question: a step that admits
    what it offers is limited by the budget (the reservation is charged
    against a queue, so it grows with backlog), while a step that offers
    far more than it admits is limited by arrivals and the budget is
    innocent.

    `blocked_partial` counts requests skipped by the partial-window guard
    -- rows arrived, but fewer than the room left, and the prompt is not
    yet fully received. Those are deferred, not lost, and separating them
    keeps a queue of half-arrived prompts from reading as an arrival
    drought.
    """

    def _reset(self) -> None:
        self._steps = 0
        self._steps_with_room = 0
        self._steps_admitting = 0
        self._room = 0
        self._admitted = 0
        self._windows = 0
        self._blocked_partial = 0
        self._blocked_no_arrival = 0
        self._queue_depth = 0
        self._reserved = 0

    def note_step(
        self,
        room: int,
        admitted: int,
        num_windows: int,
        blocked: tuple[int, int],
        queue_depth: int,
        reserved: int,
    ) -> None:
        """Record one scheduling step's prefill admission.

        `blocked` is (partial_window, no_arrival) -- the two reasons a
        candidate yielded no window. `reserved` is the drafting-slot
        reservation this step withheld from `room`; carrying it here is
        what tells a budget that is merely small from one the backlog
        ate.
        """
        self._steps += 1
        self._room += max(0, room)
        self._admitted += admitted
        self._windows += num_windows
        self._blocked_partial += blocked[0]
        self._blocked_no_arrival += blocked[1]
        self._queue_depth += queue_depth
        self._reserved += reserved
        if room > 0:
            self._steps_with_room += 1
        if admitted:
            self._steps_admitting += 1
        self._roll()

    def _report(self) -> None:
        if not self._steps:
            return
        steps = self._steps
        logger.info(
            "[lwd][cloud-sched] prefill admission over %d steps: "
            "room=%d admitted=%d (%.1f%% of room) windows=%d; "
            "steps with room=%d admitting=%d; "
            "blocked partial=%d no_arrival=%d; "
            "avg queue=%.1f reserved=%.0f tok/step",
            steps,
            self._room,
            self._admitted,
            self._admitted / self._room * 100 if self._room else 0.0,
            self._windows,
            self._steps_with_room,
            self._steps_admitting,
            self._blocked_partial,
            self._blocked_no_arrival,
            self._queue_depth / steps,
            self._reserved / steps,
        )


class LwdCloudArrivalMeter(LwdIntervalMeter):
    """Hint-to-arrival latency of the staging hop.

    The edge announces a range, the worker posts the receive, the rows
    land in host memory, and only then can a window open over them. That
    span is invisible from either scheduler: the edge sees its send
    complete, the cloud sees rows appear. Timing it decides whether an
    arrival-starved cloud is waiting on the network, on D2H, or on
    nothing at all.

    Measured per seqno (one range notify), from the hint being posted to
    the first arrival merged for it.
    """

    def __init__(self) -> None:
        self._hint_ts: dict[int, float] = {}
        super().__init__()

    def _reset(self) -> None:
        # In-flight transfers keep their start time across windows:
        # dropping it would silently exclude the slowest ones.
        self._count = 0
        self._total_s = 0.0
        self._max_s = 0.0
        self._rows = 0

    def note_hint(self, seqno: int, num_tokens: int) -> None:
        self._hint_ts[seqno] = time.monotonic()
        self._rows += num_tokens

    def note_arrival(self, seqno: int) -> None:
        """First arrival of a seqno stops its clock; later merges for the
        same seqno are the tail of one transfer, not a new one."""
        started = self._hint_ts.pop(seqno, None)
        if started is None:
            return
        elapsed = time.monotonic() - started
        self._count += 1
        self._total_s += elapsed
        self._max_s = max(self._max_s, elapsed)
        self._roll()

    def forget(self, seqno: int) -> None:
        """Drop a seqno that will never arrive (rejected or aborted), so
        a stale start time cannot leak."""
        self._hint_ts.pop(seqno, None)

    def _report(self) -> None:
        if not self._count:
            return
        logger.info(
            "[lwd][cloud-recv] hint->arrival over %d transfers: avg=%.1fms "
            "max=%.1fms rows=%d inflight=%d",
            self._count,
            self._total_s / self._count * 1000,
            self._max_s * 1000,
            self._rows,
            len(self._hint_ts),
        )


class LwdEdgeDispatchMeter(LwdIntervalMeter):
    """Edge steps that produced no EMBED batch, split by the gate term
    that stopped them.

    The stall log already says *that* every engine is gated; under load
    that is the normal state and says nothing about cost. What matters is
    the share of steps lost to each term: seats held means the cloud is
    full and the edge is right to wait, while chunks held means the
    pending-send window is the limit and raising it would help.
    """

    def _reset(self) -> None:
        self._steps = 0
        self._dispatched = 0
        self._idle = 0
        self._idle_unsent = 0
        self._gated = 0
        self._gated_seats = 0
        self._gated_chunks = 0
        self._empty_batch = 0

    def note_dispatch(self) -> None:
        self._steps += 1
        self._dispatched += 1
        self._roll()

    def note_empty_batch(self) -> None:
        self._steps += 1
        self._empty_batch += 1
        self._roll()

    def note_idle(self, unsent: bool = False) -> None:
        """No engine held prefill work: a decode steady state, not a
        stall. Counted so the gated share has a denominator.

        `unsent` splits off the case that is not a steady state at all:
        requests exist that still owe prompt rows, yet the work predicate
        reports none. Those steps are the edge parking on prompts it
        holds, and the two are indistinguishable without this flag.
        """
        self._steps += 1
        self._idle += 1
        self._idle_unsent += unsent
        self._roll()

    def note_gated(self, seats_held: int, chunks_held: int) -> None:
        """Prefill work existed and every engine refused it. The two
        terms are counted per engine, since engines can hold on
        different ones within one step."""
        self._steps += 1
        self._gated += 1
        self._gated_seats += seats_held
        self._gated_chunks += chunks_held
        self._roll()

    def _report(self) -> None:
        if not self._steps:
            return
        steps = self._steps
        logger.info(
            "[lwd][edge-sched] dispatch over %d steps: dispatched=%d (%.1f%%) "
            "idle=%d (unsent=%d) gated=%d (%.1f%%) empty=%d; "
            "gated engine-steps on seats=%d chunks=%d",
            steps,
            self._dispatched,
            self._dispatched / steps * 100,
            self._idle,
            self._idle_unsent,
            self._gated,
            self._gated / steps * 100,
            self._empty_batch,
            self._gated_seats,
            self._gated_chunks,
        )
