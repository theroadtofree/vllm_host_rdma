# SPDX-License-Identifier: Apache-2.0
"""Transfer-failure containment of the edge engine (phase 2).

A mid-batch data-plane failure aborts its requests through the normal
abort path (cloud notified, bookkeeping settled) and the engine keeps
stepping instead of dying with the frontend.
"""

from __future__ import annotations

from types import SimpleNamespace

from vllm.config.lwd.topology import LwdConnectionId
from vllm.v1.lwd_control.control_edge_scheduler.lwd_cloud_dp_account import (
    LwdCloudDpAccount,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
    LwdEdgeEngineCore,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdEdgeScheduler,
)
from vllm.v1.lwd_control.lwd_exceptions import LwdTransferFailedError

CONNECTION = LwdConnectionId(0, 0, 0)


class _Resolver:
    """Stands in for the scheduler's batch-membership lookup."""

    def __init__(self, request_ids: list[str]) -> None:
        self.request_ids = request_ids
        self.calls: list = []

    def __call__(self, connection, seqnos):  # noqa: ANN001
        self.calls.append((connection, seqnos))
        return self.request_ids


def _engine(monkeypatch, resolver: _Resolver, aborted: list) -> LwdEdgeEngineCore:  # noqa: ANN001
    engine = object.__new__(LwdEdgeEngineCore)
    engine.scheduler = SimpleNamespace(lwd_edge_up_batch_request_ids=resolver)
    monkeypatch.setattr(engine, "abort_requests", aborted.extend)
    return engine


def test_up_failure_resolves_requests_through_the_scheduler(monkeypatch) -> None:  # noqa: ANN001
    """The UP send path does not know its batch membership: the engine
    resolves the requests from the scheduler's accounts."""
    aborted: list = []
    resolver = _Resolver(["e0-r1", "e0-r2"])
    engine = _engine(monkeypatch, resolver, aborted)
    failure = LwdTransferFailedError(CONNECTION, [5])
    outputs, executed = engine._lwd_contain_transfer_failure(failure)
    assert resolver.calls == [(CONNECTION, [5])]
    assert aborted == [["e0-r1", "e0-r2"]]
    assert outputs == [] and executed is False


def test_down_failure_takes_its_request_ids_directly(monkeypatch) -> None:  # noqa: ANN001
    """The DOWN unembed path carries its batch membership on the batch
    metadata; no scheduler lookup happens."""
    aborted: list = []
    resolver = _Resolver(["never"])
    engine = _engine(monkeypatch, resolver, aborted)
    failure = LwdTransferFailedError(CONNECTION, [9], req_ids=["e0-r7"])
    engine._lwd_contain_transfer_failure(failure)
    assert resolver.calls == []
    assert aborted == [["e0-r7"]]


def test_failure_without_requests_skips_the_abort(monkeypatch) -> None:  # noqa: ANN001
    aborted: list = []
    resolver = _Resolver([])
    engine = _engine(monkeypatch, resolver, aborted)
    engine._lwd_contain_transfer_failure(LwdTransferFailedError(CONNECTION, [4]))
    assert aborted == []


def test_up_batch_request_ids_reads_the_accounts(monkeypatch) -> None:  # noqa: ANN001
    scheduler = object.__new__(LwdEdgeScheduler)
    account = LwdCloudDpAccount(CONNECTION)
    account.register_batch(5, {"e0-r2", "e0-r1"})
    scheduler._lwd_accounts = {CONNECTION: account}
    assert scheduler.lwd_edge_up_batch_request_ids(CONNECTION, [5]) == [
        "e0-r1",
        "e0-r2",
    ]
    # Unknown connection and empty seqnos resolve to no requests.
    other = LwdConnectionId(3, 3, 0)
    assert scheduler.lwd_edge_up_batch_request_ids(other, [5]) == []
    assert scheduler.lwd_edge_up_batch_request_ids(CONNECTION, []) == []
