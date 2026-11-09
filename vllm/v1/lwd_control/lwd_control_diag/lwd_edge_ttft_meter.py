# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Where a first token's latency goes on the edge side of the link.

The cloud meters account for the span between the request notify landing
and the first token going back, and on a loaded run that span is a
minority of the TTFT the benchmark measures. The rest is spent before the
edge ever announces the request -- a stretch no counter times today, and
one that holds the majority of the run-to-run variance.

Three segments, each the wait for one thing, together covering everything
from the client's arrival stamp to the first prompt rows leaving:

  - frontend: the client's arrival stamp to `add_request` running on the
    engine thread. HTTP, tokenization, the ZMQ hop and the engine's input
    queue -- all native vLLM, no LWD code on the path.
  - admit: `add_request` entry to the request notify being sent. Edge
    admission and routing, plus the prefix READY round trip when prefix
    caching is on; close to zero when it is off.
  - dispatch: notify sent to the first prompt rows announced for it. The
    edge scheduler choosing this request -- the picker and the gate.

`dispatch` is the same span the cloud already times as `hold`, measured
from the other end. The control plane is about a millisecond one way (the
cloud's own `send` segment), so the two should agree within noise, and a
disagreement means a stamp or a clock is wrong rather than a deployment
being slow. Nothing else here is cross-checkable, so this one is worth
the duplicate.

`queued` counts requests whose notify went out and whose rows have not:
the edge's own backlog, to be read against the cloud's `tracking`. The
cloud knowing about requests the edge is not holding means they never
reached `add_request`, which is the shape that points at the frontend.

Timing the frontend segment means crossing a process boundary, so it is
the one span measured with `time.time()` rather than a monotonic clock;
frontend and engine share a host here, and the segment is seconds long
where the clocks differ by far less.
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
class _LwdEdgeTimeline:
    """Monotonic stamps of one request's edge hops. Zero means not reached."""

    admit: float = 0.0
    notify: float = 0.0
    dispatch: float = 0.0


class LwdEdgeTtftMeter(LwdIntervalMeter):
    """Per-request breakdown of the edge's share of TTFT."""

    def __init__(self) -> None:
        self._timelines: dict[str, _LwdEdgeTimeline] = {}
        super().__init__()

    def _reset(self) -> None:
        # Timelines survive the window: a request whose rows go out after
        # the report would otherwise lose the stamps that explain it, and
        # those are the slow ones.
        self._arrived = 0
        self._done = 0
        self._frontend = LwdSpan()
        self._admit = LwdSpan()
        self._dispatch = LwdSpan()

    def note_add_request(self, request_id: str, arrival_time: float) -> None:
        """`add_request` reached the engine thread. Closes the frontend
        segment against the client's arrival stamp and opens admit.

        `arrived` counts these, and is the one direct measurement of the
        input stream: every other view of arrivals so far has been
        reconstructed from how a backlog grew.
        """
        self._timelines[request_id] = _LwdEdgeTimeline(admit=time.monotonic())
        self._arrived += 1
        if arrival_time > 0:
            self._frontend.add(max(0.0, time.time() - arrival_time))

    def note_notify(self, request_id: str) -> None:
        """The request notify went out to the routed cloud DP rank."""
        timeline = self._timelines.get(request_id)
        if timeline is not None and not timeline.notify:
            timeline.notify = time.monotonic()

    def note_dispatch(self, request_id: str) -> None:
        """First prompt rows announced for it. Closes its timeline; later
        chunks of the same request find nothing."""
        timeline = self._timelines.get(request_id)
        if timeline is None or timeline.dispatch:
            return
        now = timeline.dispatch = time.monotonic()
        del self._timelines[request_id]
        self._done += 1
        if timeline.notify:
            self._admit.add(timeline.notify - timeline.admit)
            self._dispatch.add(now - timeline.notify)

    def forget(self, request_id: str) -> None:
        """A request that will announce no rows (aborted before dispatch)."""
        self._timelines.pop(request_id, None)

    def tick(self) -> None:
        """Close the window if it is due. Driven per step rather than per
        dispatch: an edge that dispatches nothing for ten seconds is the
        case worth a line, and it would have no dispatch to ride on."""
        self._roll()

    def _report(self) -> None:
        # A window with no arrivals and no dispatches still reports the
        # backlog it holds: zero rows announced against a standing queue
        # is the clearest line this meter can print.
        if not self._arrived and not self._done and not self._timelines:
            return
        logger.info(
            "[lwd][edge-ttft] %d arrived %d dispatched: frontend=%s "
            "admit=%s dispatch=%s (avg/max ms); queued=%d",
            self._arrived,
            self._done,
            self._frontend,
            self._admit,
            self._dispatch,
            len(self._timelines),
        )
