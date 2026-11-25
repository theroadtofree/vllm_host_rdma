# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Phase-0 probe utilities (lwd_probe).

The module under test is stdlib-only; these tests stay stdlib-only too so
they run anywhere (the Windows dev box has no torch/vllm).

Hot-path contract pinned here: ``add`` does arithmetic plus one preallocated
ring write (no logging, no allocation, nothing that can raise), ``take`` /
``take_lines`` are the only cold paths and reset the interval counters.
"""

from __future__ import annotations

import threading

from vllm.v1.lwd_comm.data_comm import lwd_probe


def test_series_take_reports_interval_and_resets() -> None:
    s = lwd_probe.LwdProbeSeries("t")
    for v in (0.001, 0.002, 0.003):
        s.add(v)
    stats = s.take()
    assert stats is not None
    assert stats["n"] == 3
    assert abs(stats["avg_ms"] - 2.0) < 1e-9
    assert abs(stats["max_ms"] - 3.0) < 1e-9
    # interval counters reset: a second take with no samples is None
    assert s.take() is None


def test_series_percentile_ordering() -> None:
    s = lwd_probe.LwdProbeSeries("t", ring=64)
    for i in range(64):
        s.add(0.001 * (i + 1))
    stats = s.take()
    assert stats is not None
    assert stats["n"] == 64
    assert stats["p50_ms"] <= stats["p95_ms"] <= stats["max_ms"]
    assert stats["p50_ms"] >= 1.0  # ~mid-range, not the minimum
    assert stats["p95_ms"] > stats["p50_ms"]


def test_series_ring_wraps_without_loss_of_percentile_bounds() -> None:
    s = lwd_probe.LwdProbeSeries("t", ring=16)
    for i in range(100):
        s.add(0.0001 * (i % 10 + 1))
    stats = s.take()
    assert stats is not None
    assert stats["n"] == 100  # interval count is exact regardless of ring
    # the ring keeps the most recent window only, but stays within bounds
    assert 0.1 <= stats["p50_ms"] <= 1.0
    assert stats["max_ms"] == 1.0


def test_probe_set_lines_format_and_reset() -> None:
    ps = lwd_probe.LwdProbeSet("chan")
    s = ps.series("recv_total")
    s.add(0.004)
    ps.count("recv_bytes", 4096.0)
    ps.count("recv_bytes", 2048.0)
    ps.min_gauge("pool_free", 3)
    ps.min_gauge("pool_free", 2)  # min over the interval wins
    lines = ps.take_lines()
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("chan")
    assert "recv_total n=1" in line
    assert "p50=" in line and "p95=" in line and "max=" in line
    assert "recv_bytes=6144" in line
    assert "pool_free_min=2" in line
    # everything reset: next flush is empty
    assert ps.take_lines() == []


def test_probe_set_empty_when_no_samples() -> None:
    ps = lwd_probe.LwdProbeSet("chan")
    assert ps.take_lines() == []
    ps.series("recv_copyout")  # created but never fed
    assert ps.take_lines() == []


def test_series_thread_safe_shared_across_threads() -> None:
    ps = lwd_probe.LwdProbeSet("envelope")
    s = ps.series("landing", thread_safe=True)
    n_threads, per_thread = 4, 500

    def spin() -> None:
        for i in range(per_thread):
            s.add(0.0001 * ((i % 5) + 1))

    threads = [threading.Thread(target=spin) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stats = s.take()
    assert stats is not None
    assert stats["n"] == n_threads * per_thread


def test_probe_enabled_env_parsing(monkeypatch) -> None:
    monkeypatch.setattr(lwd_probe, "PROBE_ENABLED", False)
    assert not lwd_probe.probe_enabled()
    monkeypatch.setattr(lwd_probe, "PROBE_ENABLED", True)
    assert lwd_probe.probe_enabled()


def test_add_since_records_positive_elapsed() -> None:
    s = lwd_probe.LwdProbeSeries("t")
    t0 = lwd_probe.perf_counter()
    s.add_since(t0)
    stats = s.take()
    assert stats is not None
    assert stats["avg_ms"] >= 0.0
    assert stats["n"] == 1
