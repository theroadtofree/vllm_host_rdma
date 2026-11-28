# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-listener route teardown for dynamically built UCX channels."""

from __future__ import annotations

import threading

import pytest

from vllm.v1.lwd_comm.data_comm.ucx_channel import (
    LwdUcxChannel,
    LwdUcxConfig,
    LwdUcxListener,
)


def _listener() -> LwdUcxListener:
    listener = object.__new__(LwdUcxListener)
    listener._host = "127.0.0.1"
    listener.port = 0
    listener._fd = None
    listener._stopped = True
    listener._thread = None
    listener._routes = {}
    listener._lock = threading.Lock()
    listener._traffic_class = None
    return listener


def test_unregister_is_idempotent_and_preserves_other_routes() -> None:
    listener = _listener()
    first = object.__new__(LwdUcxChannel)
    first._cfg_traffic_class = None
    second = object.__new__(LwdUcxChannel)
    second._cfg_traffic_class = None
    listener.register(1, first)
    listener.register(2, second)

    listener.unregister(1, second)
    assert listener._routes == {1: first, 2: second}
    listener.unregister(1, first)
    listener.unregister(1, first)
    assert listener._routes == {2: second}


def test_serve_via_duplicate_closes_new_channel_and_allows_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _listener()
    cfg = LwdUcxConfig(block_mb=1, n_blocks=2, token=7)
    original = LwdUcxChannel.serve_via(listener, cfg=cfg)
    closed: list[LwdUcxChannel] = []
    close = LwdUcxChannel.close

    def record_close(channel: LwdUcxChannel, timeout: float = 2.0) -> None:
        closed.append(channel)
        close(channel, timeout=timeout)

    monkeypatch.setattr(LwdUcxChannel, "close", record_close)
    try:
        with pytest.raises(ValueError, match="duplicate UHC1 token"):
            LwdUcxChannel.serve_via(listener, cfg=cfg)

        assert len(closed) == 1
        assert closed[0] is not original
        assert closed[0]._stopped
        assert not closed[0]._thread.is_alive()
        assert listener._routes[7] is original

        original.close()
        assert 7 not in listener._routes
        replacement = LwdUcxChannel.serve_via(listener, cfg=cfg)
        assert listener._routes[7] is replacement
        replacement.close()
        assert not listener._routes
    finally:
        original.close()
