# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Where a first token's latency goes on the cloud side of the link.

TTFT is measured at the edge, but most of what it covers happens after
the edge has handed the request over: the request notify lands, the edge
streams prompt rows up, the rows reach host memory, the scheduler opens
windows over them, the last window produces the first token, and the
token goes back down. One TTFT number cannot say which of those grew,
and the candidate stories call for opposite fixes -- a cloud that queues
prompts it already holds is a scheduling problem, a cloud that holds
nothing because the rows are still in flight is a data-plane one.

`LwdCloudTtftMeter` stamps each hop and reports the segments between
them. Their sum is what the cloud is accountable for; the gap between
that sum and the benchmark's TTFT is what the edge spends before it
notifies and after the token comes back, which the cloud cannot see. Its
`tracking` count is the cloud's own backlog: requests it knows about that
have produced no token yet.

`LwdCloudStepMeter` does the same for the steady state. The interval
between one request's own committed tokens is the cloud's share of ITL,
and the count of steps that committed nothing at all is what a prefill
chunk costs every request that was decoding while it ran -- the two
numbers that say whether an ITL measured at the edge was spent here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.lwd_control.lwd_control_diag.lwd_meter_base import (
    LwdIntervalMeter,
    LwdSpan,
)

logger = init_logger(__name__)


@dataclass(slots=True)
class _LwdRequestTimeline:
    """Monotonic stamps of one request's hops, in the order they occur.

    Zero means not reached yet, which is also how a hop that never
    happens for this request (a full prefix hit arrives no rows) is kept
    out of the averages rather than entering them as a zero-length
    segment.
    """

    notify: float = 0.0
    hint: float = 0.0
    first_arrival: float = 0.0
    last_arrival: float = 0.0
    last_scheduled: float = 0.0
    first_token: float = 0.0
    chunks: int = 0
    arrivals: int = 0


class LwdCloudTtftMeter(LwdIntervalMeter):
    """Per-request breakdown of the cloud's share of TTFT.

    Six segments, each the wait for one thing:
      - hold: the cloud knows about the request, the edge has announced
        no rows for it yet.
      - flight: rows announced, not yet in host memory.
      - fill: first rows landed, the rest of the prompt still streaming.
      - queue: every row needed is here and the scheduler has not yet run
        the window that finishes the prompt. This is the only segment the
        cloud scheduler itself can shorten.
      - exec: that last window running.
      - send: first token committed, handed to the edge link.

    The stamps come from two threads -- the ingest thread notifies and
    hints, the step thread does the rest -- so each note is a single dict
    operation and only the step thread ever reports.
    """

    def __init__(self) -> None:
        self._timelines: dict[str, _LwdRequestTimeline] = {}
        super().__init__()

    def _reset(self) -> None:
        # Timelines deliberately survive the window: a request whose
        # first token lands after the report would otherwise lose the
        # stamps that explain it, and those are the slow ones.
        self._done = 0
        self._total = LwdSpan()
        self._hold = LwdSpan()
        self._flight = LwdSpan()
        self._fill = LwdSpan()
        self._queue = LwdSpan()
        self._exec = LwdSpan()
        self._send = LwdSpan()
        self._chunks = 0
        self._arrivals = 0

    def note_notify(self, request_id: str) -> None:
        """The cloud learned the request exists. Start of its timeline."""
        self._timelines[request_id] = _LwdRequestTimeline(notify=time.monotonic())

    def note_hint(self, request_id: str) -> None:
        """The edge announced rows for it; only the first one starts the
        clock, later chunks are the fill segment."""
        timeline = self._timelines.get(request_id)
        if timeline is not None and not timeline.hint:
            timeline.hint = time.monotonic()

    def note_arrival(self, request_id: str) -> None:
        """Rows of it reached host memory."""
        timeline = self._timelines.get(request_id)
        if timeline is None:
            return
        now = time.monotonic()
        if not timeline.first_arrival:
            timeline.first_arrival = now
        timeline.last_arrival = now
        timeline.arrivals += 1

    def note_scheduled(self, request_id: str) -> None:
        """A prefill window of it ran. The last one before the first token
        is the one that finished the prompt, so the stamp is overwritten."""
        timeline = self._timelines.get(request_id)
        if timeline is None:
            return
        timeline.last_scheduled = time.monotonic()
        timeline.chunks += 1

    def note_first_token(self, request_id: str) -> None:
        """Its first token was committed by this step."""
        timeline = self._timelines.get(request_id)
        if timeline is not None and not timeline.first_token:
            timeline.first_token = time.monotonic()

    def note_sent(self, request_id: str) -> None:
        """Its step metadata is being handed to the edge link. Closes the
        timeline; the same call for a decoding request finds nothing."""
        timeline = self._timelines.get(request_id)
        if timeline is None or not timeline.first_token:
            return
        del self._timelines[request_id]
        self._fold(timeline, time.monotonic())

    def forget(self, request_id: str) -> None:
        """A request that will send no more. One that already had its first
        token still counts, minus the send segment it never reached."""
        timeline = self._timelines.pop(request_id, None)
        if timeline is not None and timeline.first_token:
            self._fold(timeline, 0.0)

    def tick(self) -> None:
        """Close the window if it is due. Driven per step rather than per
        completion: a cloud that completes nothing for ten seconds is the
        case worth a line, and it would have no completion to ride on."""
        self._roll()

    def _fold(self, timeline: _LwdRequestTimeline, sent: float) -> None:
        self._done += 1
        self._chunks += timeline.chunks
        self._arrivals += timeline.arrivals
        self._add(self._total, timeline.notify, timeline.first_token)
        self._add(self._hold, timeline.notify, timeline.hint)
        self._add(self._flight, timeline.hint, timeline.first_arrival)
        self._add(self._fill, timeline.first_arrival, timeline.last_arrival)
        self._add(self._queue, timeline.last_arrival, timeline.last_scheduled)
        self._add(self._exec, timeline.last_scheduled, timeline.first_token)
        self._add(self._send, timeline.first_token, sent)

    @staticmethod
    def _add(span: LwdSpan, start: float, end: float) -> None:
        """Record a segment only when both of its ends were reached."""
        if start and end:
            span.add(end - start)

    def _report(self) -> None:
        # A window with no completions still reports when requests are
        # being held: zero first tokens against a standing backlog is the
        # clearest line this meter can print.
        if not self._done and not self._timelines:
            return
        done = self._done
        logger.info(
            "[lwd][cloud-ttft] %d first tokens: cloud total=%s; hold=%s "
            "flight=%s fill=%s queue=%s exec=%s send=%s (avg/max ms); "
            "chunks/req=%.1f arrivals/req=%.1f; tracking=%d",
            done,
            self._total,
            self._hold,
            self._flight,
            self._fill,
            self._queue,
            self._exec,
            self._send,
            self._chunks / done if done else 0.0,
            self._arrivals / done if done else 0.0,
            len(self._timelines),
        )


class LwdCloudStepMeter(LwdIntervalMeter):
    """Step cadence and the cloud's share of ITL.

    An ITL measured at the edge is a step interval plus however many steps
    passed without that request producing anything. Both terms are here:
    `step` is the interval between steps, `gap` is the interval between
    one request's own tokens, and `silent` counts the steps that committed
    no token at all -- a prefill chunk running is exactly such a step, and
    it lands on every request that was decoding.
    """

    def __init__(self) -> None:
        self._last_commit: dict[str, float] = {}
        self._last_step = 0.0
        super().__init__()

    def _reset(self) -> None:
        # `_last_step` and `_last_commit` are not cleared: an interval
        # straddling the report boundary belongs to one of the windows,
        # and dropping it would drop the long ones first.
        self._steps = 0
        self._silent = 0
        self._tokens = 0
        self._committing = 0
        self._committed_reqs = 0
        self._step = LwdSpan()
        self._gap = LwdSpan()

    def note_step(self, request_ids: list[str], num_tokens: int) -> None:
        """One step's committed tokens, by the requests that got them."""
        now = time.monotonic()
        if self._last_step:
            self._step.add(now - self._last_step)
        self._last_step = now
        self._steps += 1
        self._tokens += num_tokens
        if not request_ids:
            self._silent += 1
            self._roll()
            return
        self._committing += 1
        self._committed_reqs += len(request_ids)
        for request_id in request_ids:
            previous = self._last_commit.get(request_id)
            if previous is not None:
                self._gap.add(now - previous)
            self._last_commit[request_id] = now
        self._roll()

    def forget(self, request_id: str) -> None:
        self._last_commit.pop(request_id, None)

    def _report(self) -> None:
        if not self._steps:
            return
        logger.info(
            "[lwd][cloud-step] %d steps: step=%s silent=%d; tokens=%d over %d "
            "committing steps, reqs/commit=%.1f; per-req token gap=%s "
            "(avg/max ms)",
            self._steps,
            self._step,
            self._silent,
            self._tokens,
            self._committing,
            self._committed_reqs / self._committing if self._committing else 0.0,
            self._gap,
        )
