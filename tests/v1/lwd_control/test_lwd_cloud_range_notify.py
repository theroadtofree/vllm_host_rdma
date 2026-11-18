# SPDX-License-Identifier: Apache-2.0
"""Cloud-side tolerance of an orphaned RangeNotify (restart boundary).

An edge request that straddles a cloud restart sends its continuation
RangeNotify to the fresh cloud, while the RequestNotify that recorded its
prompt length was consumed by the OLD incarnation. The old branch treats
the missing-metadata hint as a SKIP with an error log (the hint is only
the early-recv optimization; the fatal variant killed the freshly
restarted cloud one second after startup).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.config.lwd.topology import LwdConnectionId
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine import (
    LwdCloudEngineCore,
)

CONNECTION = LwdConnectionId(0, 0, 0)


def _engine() -> LwdCloudEngineCore:
    engine = object.__new__(LwdCloudEngineCore)
    engine._lwd_latest_seqnos = {}
    engine._lwd_prompt_lens = {}
    engine._lwd_retired_prompt_lens = {}
    return engine


def _notify(seqno: int, request_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        seqno=seqno,
        items=[
            SimpleNamespace(
                request_id=request_id, prefix_nonce=1, num_tokens=4, offset=0
            )
        ],
    )


def test_orphaned_range_notify_is_skipped_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hint whose RequestNotify metadata was recorded by the previous
    cloud incarnation must be skipped with an error log — the fresh cloud
    survives (old-branch invariant: never fatal). The hint is not tracked
    either: nothing is pre-posted for it."""
    engine = _engine()
    monkeypatch.setattr(
        engine,
        "_lwd_track_range",
        lambda *args: pytest.fail("a skipped hint must not be tracked"),
    )
    # seqno=1 past the empty latest ledger; the request is unknown here.
    engine._lwd_handle_range_notify(CONNECTION, _notify(1, "e0-r1"))
    # The seqno stream was not advanced by the skipped hint.
    assert engine._lwd_latest_seqnos == {}
