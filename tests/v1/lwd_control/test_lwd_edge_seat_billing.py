# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Seat billing of the edge scheduler: the native running pool must cover
the fleet's seats and the picker's free seats must be per connection.

Before the fix the pool kept one cloud DP's seat count and free_seq
subtracted the whole running queue, so an edge serving several cloud DPs
admitted at one DP's concurrency -- the fleet idled at (N-1)/N of its
seats while waiting requests piled up."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace as NS

from vllm.config.lwd.topology import LwdConnectionId
from vllm.v1.lwd_control.control_edge_scheduler.lwd_cloud_dp_account import (
    LwdCloudDpAccount,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_picker import (
    LwdPrefillCandidate,
    lwd_grammar_ready,
    lwd_pick_prefill_batch,
)

SCHED_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm"
    / "v1"
    / "lwd_control"
    / "control_edge_scheduler"
    / "lwd_edge_scheduler.py"
)

_EXTRACTED_METHODS = (
    "_lwd_dp_of",
    "_lwd_running_on",
    "_lwd_resize_seat_pool",
    "_lwd_pick_prefill_batch",
    "_lwd_servable_connection",
)


class _SchedBase:
    """The picker's only base-class calls are phase scans; this stub
    yields every queue member and the picker itself skips the
    decode-phase residents (remaining=0), so no phase fidelity is
    needed."""

    def _lwd_reqs_in_phase(self, phase, *queues):
        for queue in queues:
            yield from queue

    def _lwd_has_prefill_work(self, connection_id) -> bool:
        # Routed work exists for the connection (running or waiting).
        return connection_id in self._lwd_route.values()

    def _lwd_has_prefill_chunk_inflight(self, connection_id) -> bool:
        return False


def _make_sched(connections: tuple[LwdConnectionId, ...], seat_cap: int) -> NS:
    tree = ast.parse(SCHED_PATH.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LwdEdgeScheduler"
    )
    fns = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in _EXTRACTED_METHODS
    ]
    assert {fn.name for fn in fns} == set(_EXTRACTED_METHODS), "method drift"
    class_def = ast.ClassDef(
        name="LwdEdgeSchedulerUnderTest",
        bases=[ast.Name(id="_SchedBase", ctx=ast.Load())],
        keywords=[],
        body=fns,
        decorator_list=[],
    )
    module = ast.Module(body=[class_def], type_ignores=[])
    ast.fix_missing_locations(module)
    scope: dict = {
        "_SchedBase": _SchedBase,
        "LwdReqPhase": NS(PREFILL="prefill", DECODE="decode"),
        "LwdPrefillCandidate": LwdPrefillCandidate,
        "lwd_pick_prefill_batch": lwd_pick_prefill_batch,
        "lwd_grammar_ready": lwd_grammar_ready,
        "RequestStatus": NS(WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR=object()),
    }
    exec(compile(module, str(SCHED_PATH), "exec"), scope)
    sched = scope["LwdEdgeSchedulerUnderTest"]()
    sched._lwd_connections = connections
    sched._lwd_seat_cap = seat_cap
    sched.max_num_running_reqs = seat_cap
    sched._lwd_accounts = {
        connection_id: LwdCloudDpAccount(connection_id) for connection_id in connections
    }
    sched._lwd_route = {}
    sched.running = []
    sched.waiting = []
    sched.skipped_waiting = []
    sched.max_num_scheduled_tokens = 2048
    sched._lwd_chunk_unit = 2048
    return sched


def _request(rid: str, computed: int | None = None) -> NS:
    """Decode-phase resident by default (computed=prompt); waiting new
    starts pass computed=0 so the picker sees remaining > 0."""
    prompt = 8
    return NS(
        request_id=rid,
        num_prompt_tokens=prompt,
        num_computed_tokens=computed if computed is not None else prompt,
        is_finished=lambda: False,
        status=None,
    )


def test_pool_covers_the_fleet_seats() -> None:
    conns = (LwdConnectionId(0, 0, 0), LwdConnectionId(0, 1, 0))
    sched = _make_sched(conns, 2)
    sched._lwd_resize_seat_pool()
    assert sched.max_num_running_reqs == 4
    # Recomputing from the same connection set is idempotent.
    sched._lwd_resize_seat_pool()
    assert sched.max_num_running_reqs == 4


def test_free_seq_bills_only_the_own_connection() -> None:
    conns = (LwdConnectionId(0, 0, 0), LwdConnectionId(0, 1, 0))
    conn_a, conn_b = conns
    sched = _make_sched(conns, 2)
    sched._lwd_resize_seat_pool()
    route = {"r1": conn_b, "r2": conn_b}
    sched._lwd_route = route
    # The sibling connection is fully seated ...
    sched.running = [_request(rid) for rid in route]
    assert sched._lwd_running_on(conn_b) == 2
    assert sched._lwd_running_on(conn_a) == 0
    # ... and does not block this connection's new starts.
    sched._lwd_route.update({"w1": conn_a, "w2": conn_a})
    sched.waiting = [_request("w1", computed=0), _request("w2", computed=0)]
    req_ids, _ = sched._lwd_pick_prefill_batch(conn_a)
    assert req_ids == ["w1", "w2"]
    # The full connection admits nothing for itself.
    req_ids_b, _ = sched._lwd_pick_prefill_batch(conn_b)
    assert req_ids_b == []


def test_gate_bills_the_own_connection_seats() -> None:
    """The gate holds a connection at its own seats, not the fleet pool."""
    conns = (LwdConnectionId(0, 0, 0), LwdConnectionId(0, 1, 0))
    conn_a, _conn_b = conns
    sched = _make_sched(conns, 2)
    sched._lwd_resize_seat_pool()
    sched._lwd_route = {"r1": conn_a, "r2": conn_a, "w1": conn_a}
    sched.running = [_request("r1"), _request("r2")]  # conn_a holds 2/2 seats
    sched._lwd_last_engine = -1
    # The pool now offers 4 seats fleet-wide; the gate must still skip
    # this connection at its own 2 (the sibling is empty, so with the
    # pool as the cap this connection would wrongly pass the gate).
    assert sched._lwd_servable_connection() is None


def test_own_seats_exhausted_blocks_own_new_starts() -> None:
    conns = (LwdConnectionId(0, 0, 0),)
    conn = conns[0]
    sched = _make_sched(conns, 2)
    sched._lwd_resize_seat_pool()
    sched._lwd_route = {"r1": conn, "r2": conn, "w1": conn}
    sched.running = [_request("r1"), _request("r2")]
    sched.waiting = [_request("w1", computed=0)]
    req_ids, _ = sched._lwd_pick_prefill_batch(conn)
    assert req_ids == []
