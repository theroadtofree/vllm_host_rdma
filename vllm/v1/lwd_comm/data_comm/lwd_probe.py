# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hot-path-safe latency probes for the LWD data plane (Phase-0 baseline).

The probes exist to quantify the host-RoCE cloud-side receive path before
it is redesigned (direct-to-pinned landing): IO-thread occupancy per
frame, its copy-out / re-post split, sender-thread D2H cost, and the
early-recv submit-to-published envelope. The rules are pinned by the
regression analysis -- a probe that logs or allocates on the hot path
measures itself:

* hot paths only accumulate: a few ``perf_counter`` reads plus one ring
  slot write per frame, no logging, no dynamic allocation, nothing that
  can raise on the library IO thread;
* percentiles are computed at flush time from a fixed sample ring, on the
  ~10s cadence the call sites already tick, never per frame;
* interval counters reset on every ``take_lines()``; the ring keeps the
  recent window for percentiles (if fewer than ring-size samples arrived,
  the window reaches into the previous interval -- acceptable and noted).

Call sites (all flush with logger.error during the baseline campaign so
they surface at every log level; demote once the campaign ends):

* ``ucx_channel._on_recv_done`` -- recv IO occupancy: ``recv_copyout``
  (block -> bytes), ``recv_repost`` (wildcard re-post), ``recv_total``
  (whole data branch incl. the user callback when the frame pairs at
  arrival), ``recv_bytes`` / ``pool_free`` gauges;
* ``ucx_io._do_send`` -- sender thread: ``send_acq`` (staging acquire
  wait), ``send_d2h`` (synchronous device -> staging copy), ``send_post``;
* ``lwd_cloud_worker`` -- ``landing``: submit -> published envelope
  (quantised by the completion-report poll cadence).

The single-frame delimitation traces (``_tr_*`` in ucx_io) are unchanged.

``LWD_PROBE=0`` disables sampling at the call sites (one module-global
read per branch). This module stays stdlib-only so it is testable and
usable without torch/vllm imports.
"""

from __future__ import annotations

import os
import threading
from time import perf_counter

__all__ = ["LwdProbeSet", "LwdProbeSeries", "probe_enabled"]

PROBE_ENABLED = os.environ.get("LWD_PROBE", "1").strip().lower() not in (
    "0",
    "false",
    "off",
)


def probe_enabled() -> bool:
    """Sampling gate shared by every call site (decided once at import)."""
    return PROBE_ENABLED


class LwdProbeSeries:
    """Duration samples: interval counters for avg/max, ring for percentiles.

    ``add`` is the hot-path method (arithmetic + one preallocated slot
    write; ``thread_safe=True`` adds one uncontended lock for the
    cross-thread envelope series). ``take`` is cold-path only.
    """

    __slots__ = (
        "_name",
        "_ring",
        "_ring_cap",
        "_ring_idx",
        "_ring_n",
        "_n",
        "_total_s",
        "_max_s",
        "_lock",
    )

    def __init__(self, name: str, ring: int = 1024, thread_safe: bool = False) -> None:
        self._name = name
        self._ring_cap = max(16, int(ring))
        self._ring = [0.0] * self._ring_cap
        self._ring_idx = 0
        self._ring_n = 0
        self._n = 0
        self._total_s = 0.0
        self._max_s = 0.0
        self._lock = threading.Lock() if thread_safe else None

    @property
    def name(self) -> str:
        return self._name

    def add(self, dur_s: float) -> None:
        if self._lock is not None:
            with self._lock:
                self._add(dur_s)
        else:
            self._add(dur_s)

    def add_since(self, t0_s: float) -> None:
        self.add(perf_counter() - t0_s)

    def _add(self, dur_s: float) -> None:
        self._ring[self._ring_idx] = dur_s
        self._ring_idx = (self._ring_idx + 1) % self._ring_cap
        if self._ring_n < self._ring_cap:
            self._ring_n += 1
        self._n += 1
        self._total_s += dur_s
        if dur_s > self._max_s:
            self._max_s = dur_s

    def take(self) -> dict | None:
        """Interval stats (n/avg/max over the interval, percentiles over the
        recent ring window); resets the interval counters. None when no
        samples arrived since the last take."""
        if self._lock is not None:
            with self._lock:
                return self._take()
        return self._take()

    def _take(self) -> dict | None:
        n, total_s, max_s = self._n, self._total_s, self._max_s
        self._n = 0
        self._total_s = 0.0
        self._max_s = 0.0
        if n == 0:
            return None
        window = sorted(self._ring[: self._ring_n])

        def pct(p: float) -> float:
            idx = min(len(window) - 1, int(p * (len(window) - 1)))
            return window[idx]

        return {
            "n": n,
            "avg_ms": total_s / n * 1e3,
            "p50_ms": pct(0.50) * 1e3,
            "p95_ms": pct(0.95) * 1e3,
            "max_ms": max_s * 1e3,
        }


def _fmt_num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.3f}"


class LwdProbeSet:
    """Named duration series + interval counters + min gauges under one
    label, flushed as ready-to-log lines by the caller's periodic tick.

    Single-thread probes need no lock (the channel IO thread and the sender
    thread each flush their own set). ``count``/``min_gauge`` are
    GIL-atomic-enough for probe bookkeeping; the cross-thread series opt in
    via ``thread_safe=True``.
    """

    def __init__(self, label: str) -> None:
        self._label = label
        self._series: dict[str, LwdProbeSeries] = {}
        self._counters: dict[str, float] = {}
        self._min_gauges: dict[str, float] = {}

    def series(
        self, name: str, *, ring: int = 1024, thread_safe: bool = False
    ) -> LwdProbeSeries:
        """Get (create-once) a duration series. Call sites pre-create every
        series at construction so the hot path never constructs anything."""
        got = self._series.get(name)
        if got is None:
            got = LwdProbeSeries(name, ring=ring, thread_safe=thread_safe)
            self._series[name] = got
        return got

    def count(self, name: str, n: float = 1.0) -> None:
        self._counters[name] = self._counters.get(name, 0.0) + n

    def min_gauge(self, name: str, value: float) -> None:
        cur = self._min_gauges.get(name)
        if cur is None or value < cur:
            self._min_gauges[name] = value

    def take_lines(self) -> list[str]:
        """Format and reset everything accumulated since the last call;
        empty list when nothing new arrived."""
        parts: list[str] = [self._label]
        for name in sorted(self._series):
            stats = self._series[name].take()
            if stats is not None:
                parts.append(
                    f"{name} n={stats['n']} avg={stats['avg_ms']:.2f}ms "
                    f"p50={stats['p50_ms']:.2f} p95={stats['p95_ms']:.2f} "
                    f"max={stats['max_ms']:.2f}"
                )
        for name in sorted(self._counters):
            value = self._counters.pop(name, 0.0)
            if value:
                parts.append(f"{name}={_fmt_num(value)}")
        for name in sorted(self._min_gauges):
            value = self._min_gauges.pop(name, None)
            if value is not None:
                parts.append(f"{name}_min={_fmt_num(value)}")
        return [] if len(parts) == 1 else [" | ".join(parts)]
