# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""lwd_pick_prefill_batch 表驱动测试:chunk 组批算术的边界锁死。

用例覆盖原实现注释里记录的事故形态(末位截断漏记、预算超批)。
"""

from __future__ import annotations

import enum
from types import SimpleNamespace

import pytest

from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_picker import (
    LwdPrefillCandidate,
    lwd_grammar_ready,
    lwd_pick_prefill_batch,
)


def candidate(rid: str, prompt: int, computed: int = 0) -> LwdPrefillCandidate:
    return LwdPrefillCandidate(rid, prompt, computed)


@pytest.mark.parametrize(
    (
        "running",
        "waiting",
        "cap",
        "free_seq",
        "chunk_unit",
        "want",
    ),
    [
        # cap < 1:空批
        ([], [candidate("a", 100)], 0, 5, 8192, {}),
        # 简单开新:FCFS 前缀
        (
            [],
            [candidate("a", 100), candidate("b", 50)],
            200,
            5,
            8192,
            {"a": 100, "b": 50},
        ),
        # 续传优先:running 未发完的先收尾,再开新
        (
            [candidate("r", 300, computed=100)],
            [candidate("a", 100)],
            400,
            5,
            8192,
            {"r": 200, "a": 100},
        ),
        # 续传剩余耗尽(computed == prompt):跳过
        (
            [candidate("r", 100, computed=100)],
            [candidate("a", 10)],
            400,
            5,
            8192,
            {"a": 10},
        ),
        # 末位截断:最后条目被剩余额度截尾,Σ == cap
        (
            [],
            [candidate("a", 100), candidate("b", 100)],
            150,
            5,
            8192,
            {"a": 100, "b": 50},
        ),
        # 截断剩余额度为 0:该条目不入批
        (
            [],
            [candidate("a", 100), candidate("b", 10)],
            100,
            5,
            8192,
            {"a": 100},
        ),
        # 截断块占预算:后续开新不超批(真机事故:漏记 used 超批 2x)
        (
            [],
            [candidate("a", 4000), candidate("b", 3000)],
            6000,
            5,
            8192,
            {"a": 4000, "b": 2000},
        ),
        # 续传段截断
        (
            [candidate("r", 500)],
            [],
            300,
            5,
            8192,
            {"r": 300},
        ),
        # free_seq 钳制:开新名额耗尽即停
        (
            [],
            [candidate("a", 10), candidate("b", 10)],
            100,
            1,
            8192,
            {"a": 10},
        ),
        # free_seq = 0:一个新请求都不开
        (
            [],
            [candidate("a", 10)],
            100,
            0,
            8192,
            {},
        ),
        # chunk_unit 切块:开新 c = min(prompt, C)
        (
            [],
            [candidate("a", 100)],
            1000,
            5,
            30,
            {"a": 30},
        ),
        # chunk_unit 切块:续传 c = min(剩余, C)
        (
            [candidate("r", 100, computed=70)],
            [],
            1000,
            5,
            40,
            {"r": 30},
        ),
        # 长请求与短请求正常合批(模型无关,无独占规则)
        (
            [],
            [candidate("a", 100), candidate("b", 9000)],
            20000,
            5,
            8192,
            {"a": 100, "b": 8192},
        ),
    ],
)
def test_pick_table(running, waiting, cap, free_seq, chunk_unit, want):
    req_ids, got = lwd_pick_prefill_batch(
        running,
        waiting,
        cap=cap,
        free_seq=free_seq,
        chunk_unit=chunk_unit,
    )
    assert got == want
    assert req_ids == list(want)  # 次序 = 入批次序
    assert sum(got.values()) <= cap  # 预算不变量
    assert all(c >= 1 for c in got.values())  # 不产生零长条目


def test_pick_order_is_running_then_waiting():
    """批内次序:续传条目在前,开新条目在后(数据面 chunk 流按批连续)。"""
    req_ids, _ = lwd_pick_prefill_batch(
        [candidate("r", 10)],
        [candidate("a", 10)],
        cap=100,
        free_seq=5,
        chunk_unit=8192,
    )
    assert req_ids == ["r", "a"]


def test_waiting_candidates_use_negotiated_computed_tokens():
    req_ids, got = lwd_pick_prefill_batch(
        [],
        [candidate("hit", 100, computed=64), candidate("done", 32, computed=32)],
        cap=100,
        free_seq=5,
        chunk_unit=16,
    )

    assert req_ids == ["hit"]
    assert got == {"hit": 16}


# --- grammar-ready filter (edge/cloud execution-set consistency) ---

_MISSING = object()


class _Status(enum.Enum):
    WAITING = 0
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = 5


_BLOCKED = _Status.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR


def _grammar_request(status, grammar):
    return SimpleNamespace(
        status=status,
        structured_output_request=(
            None if grammar is _MISSING else SimpleNamespace(grammar=grammar)
        ),
    )


def test_grammar_ready_passes_plain_requests():
    """No constraint (or already promoted): selectable."""
    assert lwd_grammar_ready(_grammar_request(_Status.WAITING, None), _BLOCKED)
    assert lwd_grammar_ready(
        _grammar_request(_Status.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR, object()),
        _Status.WAITING,
    )


def test_grammar_ready_defers_blocked_uncompiled_requests():
    """Blocked and still compiling: the cloud native scheduler defers it,
    so the edge picker must defer it too."""
    assert not lwd_grammar_ready(
        _grammar_request(_Status.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR, None),
        _BLOCKED,
    )
    assert not lwd_grammar_ready(
        _grammar_request(_Status.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR, _MISSING),
        _BLOCKED,
    )


def test_grammar_ready_passes_compiled_or_failed_grammars():
    """Compiled grammar (or a compile failure, which the native error path
    finishes) is selectable: promotion stays with the native scheduler."""
    grammar = SimpleNamespace(done=True)
    assert lwd_grammar_ready(
        _grammar_request(_Status.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR, grammar),
        _BLOCKED,
    )
    assert lwd_grammar_ready(
        _grammar_request(_Status.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR, ValueError()),
        _BLOCKED,
    )
