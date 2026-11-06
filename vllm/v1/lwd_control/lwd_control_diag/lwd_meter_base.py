# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Interval gate and span accumulator shared by the LWD meters.

A meter is a resident counter read from a scheduling thread, so it must
cost a few adds per step and report rarely enough that the logging cannot
become the thing being measured. Reporting is edge triggered by the
interval rather than scheduled, so an idle deployment stops logging
instead of printing zeros.

Meter lines go to the native logger, not `lwd_logger`: the rates they
measure are only meaningful on a run whose per-step LWD logging is off
(VLLM_LWD_LOG_LEVEL=disable), and a meter silenced by the very switch the
measurement needs would have nothing to report. The native logger adds no
prefix of its own, so the "[lwd]" tag is written into each format string
to keep these few lines greppable among the native ones.
"""

from __future__ import annotations

import time

# One line per meter per interval. Long enough that a loaded second
# cannot flood the log, short enough to see a phase change within a run.
LWD_METER_INTERVAL_S = 10.0


class LwdIntervalMeter:
    """Accumulate freely, report at most once per interval, then start the
    next window from zero."""

    def __init__(self) -> None:
        self._last_report = time.monotonic()
        self._reset()

    def _reset(self) -> None:
        """Clear this window's counters. Subclasses define the fields."""
        raise NotImplementedError

    def _report(self) -> None:
        """Emit one line for the window just closed."""
        raise NotImplementedError

    def _roll(self) -> None:
        """Close the window if the interval elapsed."""
        now = time.monotonic()
        if now - self._last_report < LWD_METER_INTERVAL_S:
            return
        self._last_report = now
        self._report()
        self._reset()


class LwdSpan:
    """One segment of a timeline: how many samples, their mean, the worst.

    Segments are accumulated apart rather than summed because each has its
    own cure, and one average over all of them hides the one that moved.
    The mean alone would also hide a segment that is fast except when it
    is not, which is the shape every stall in this system has had.
    """

    __slots__ = ("count", "_max_s", "_total_s")

    def __init__(self) -> None:
        self.count = 0
        self._total_s = 0.0
        self._max_s = 0.0

    def add(self, seconds: float) -> None:
        self.count += 1
        self._total_s += seconds
        self._max_s = max(self._max_s, seconds)

    def __str__(self) -> str:
        """`avg/max` in whole ms, compact enough to put six on one line."""
        if not self.count:
            return "-/-"
        return f"{self._total_s / self.count * 1000:.0f}/{self._max_s * 1000:.0f}"
