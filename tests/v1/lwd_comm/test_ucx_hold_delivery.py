# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hold-recv delivery semantics (block-held receive, Phase-1 library change).

No libucp needed: the tests drive ``_deliver_hold`` / ``release_data``
directly on a channel whose link never exists. Re-posting a wildcard needs
a live link, so the parking-path test monkeypatches ``_post_data``; the
paired-path release is simulated by calling ``_repost_block`` the way the
IO loop's recycle drain does (without a link it must simply drop the held
marking and not re-lend).
"""

from __future__ import annotations

import pytest

from vllm.v1.lwd_comm.data_comm.ucx_channel import LwdUcxChannel, LwdUcxConfig
from vllm.v1.lwd_comm.data_comm.ucx_pool import LwdUcxBlockPool


@pytest.fixture()
def hold_channel():
    # n_blocks=4 -> early-held admission cap = 4-2 = 2
    ch = LwdUcxChannel(LwdUcxConfig(hold_recv=True, block_mb=1, n_blocks=4))
    yield ch
    ch.close()


def _prepost(ch: LwdUcxChannel, tag: int):
    """Register one receive the way the IO thread would once the link is
    ready (the queue only drains after a link exists)."""
    op = ch.recv(tag)
    ch._do_recv(tag, op)
    return op


def test_paired_frame_delivers_block_view(hold_channel):
    ch = hold_channel
    op = _prepost(ch, 7)
    blk = ch._pool.blocks[1]
    ch._deliver_hold(7, blk, 128)
    assert op.done()
    assert isinstance(op.data, memoryview)
    assert len(op.data) == 128
    assert op._blk is blk
    assert ch._held[blk.index][0] is blk
    # the block is NOT re-posted (zero-copy hold): the free pool stays empty
    assert ch._pool.qsize() == 0
    assert ch._recycle_q.qsize() == 0


def test_release_returns_block_and_repost_drops_without_link(hold_channel):
    ch = hold_channel
    op = _prepost(ch, 3)
    blk = ch._pool.blocks[0]
    ch._deliver_hold(3, blk, 64)
    op.release_data()
    assert op._blk is None
    assert op.data is None
    # release enqueues for the IO thread; simulate its recycle drain. With
    # no live link _repost_block must not re-lend, only clear the marking.
    assert ch._recycle_q.qsize() == 1
    ch._repost_block(ch._recycle_q.get_nowait())
    assert blk.index not in ch._held
    assert ch._recycle_q.qsize() == 0


def test_release_data_is_idempotent_and_noop_without_block(hold_channel):
    ch = hold_channel
    op = _prepost(ch, 1)
    blk = ch._pool.blocks[0]
    ch._deliver_hold(1, blk, 32)
    op.release_data()
    op.release_data()  # second call: no-op, no double recycle
    assert ch._recycle_q.qsize() == 1
    # an op without a held block (e.g. standalone bytes) releases silently
    plain = ch.recv(2)
    plain.release_data()
    assert ch._recycle_q.qsize() == 1


def test_early_frame_parks_holding_its_block(hold_channel):
    ch = hold_channel
    blk = ch._pool.blocks[0]
    ch._deliver_hold(99, blk, 64)  # no pre-post waits for tag 99
    assert ch._early_count == 1
    entry = ch._early[99][0]
    assert isinstance(entry, tuple) and entry[0] is blk and entry[1] == 64
    assert ch._held[blk.index][0] is blk  # parked held: block lent, not posted
    assert ch._pool.qsize() == 0 and ch._recycle_q.qsize() == 0


def test_parked_held_frame_pairs_zero_copy_on_prepost(hold_channel):
    ch = hold_channel
    blk = ch._pool.blocks[2]
    ch._deliver_hold(7, blk, 128)
    op = ch.recv(7)
    ch._do_recv(7, op)  # pairs with the parked held entry
    assert op.done()
    assert isinstance(op.data, memoryview) and len(op.data) == 128
    assert op._blk is blk
    assert ch._early_count == 0
    op.release_data()
    assert ch._recycle_q.qsize() == 1
    ch._repost_block(ch._recycle_q.get_nowait())
    assert blk.index not in ch._held


def test_early_park_over_capacity_falls_back_to_bytes(hold_channel, monkeypatch):
    ch = hold_channel
    posted: list = []
    monkeypatch.setattr(ch, "_post_data", lambda b: posted.append(b))
    # cap = n_blocks - 2 = 2: fill it with held parks
    ch._deliver_hold(1, ch._pool.blocks[0], 16)
    ch._deliver_hold(2, ch._pool.blocks[1], 16)
    assert ch._early_count == 2 and len(ch._held) == 2
    ch._deliver_hold(3, ch._pool.blocks[2], 16)  # over cap -> bytes fallback
    assert ch._early_count == 3
    assert isinstance(ch._early[3][0], bytes)
    assert posted == [ch._pool.blocks[2]]  # block went straight back
    assert ch._pool.blocks[2].index not in ch._held


def test_early_max_overflow_drop_reclaims_held_block(hold_channel, monkeypatch):
    ch = LwdUcxChannel(LwdUcxConfig(hold_recv=True, block_mb=1, n_blocks=4, early_max=1))
    try:
        blk_a = ch._pool.blocks[0]
        ch._deliver_hold(1, blk_a, 16)
        assert ch._early_count == 1
        ch._deliver_hold(2, ch._pool.blocks[1], 16)  # store full: drop oldest
        assert ch._early_count == 1
        assert blk_a.index not in ch._held  # the dropped held entry was reclaimed
        assert 2 in ch._early and 1 not in ch._early
    finally:
        ch.close()


def test_deadline_demotes_stale_held_park_to_bytes(hold_channel, monkeypatch):
    import time as _time

    ch = hold_channel
    posted: list = []
    monkeypatch.setattr(ch, "_post_data", lambda b: posted.append(b))
    blk = ch._pool.blocks[0]
    ch._deliver_hold(9, blk, 32)
    # back-date the park so the reaper sees it as stale (deque has no item
    # assignment: clear + append)
    entry = ch._early[9][0]
    dq = ch._early[9]
    dq.clear()
    dq.append((entry[0], entry[1], _time.monotonic() - 10.0))
    demoted = ch._demote_stale_early_holds(_time.monotonic())
    assert demoted == 1
    assert isinstance(ch._early[9][0], bytes)
    assert blk.index not in ch._held
    ch.close()


def test_fail_all_pending_reclaims_parked_held_blocks(hold_channel):
    ch = hold_channel
    blk = ch._pool.blocks[3]
    ch._deliver_hold(5, blk, 16)
    assert len(ch._held) == 1
    ch._fail_all_pending(RuntimeError("link broke"))
    assert ch._early_count == 0 and not ch._early
    assert blk.index not in ch._held  # reclaimed without repost (no link)


def test_discard_consumer_releases_parked_held_block(hold_channel):
    from vllm.v1.lwd_comm.data_comm.ucx_channel import LwdUcxOp

    ch = hold_channel
    blk = ch._pool.blocks[1]
    ch._deliver_hold(4, blk, 16)

    def discard(op):  # the embedded skip path's cb
        release = getattr(op, "release_data", None)
        if release is not None:
            release()

    op = LwdUcxOp(4, cb=discard)
    ch._do_recv(4, op)
    assert op.done()
    assert ch._recycle_q.qsize() == 1  # the discard gave the block back
    ch._repost_block(ch._recycle_q.get_nowait())
    assert blk.index not in ch._held


def test_recv_into_falls_back_to_copy_out(hold_channel, monkeypatch):
    import ctypes

    ch = hold_channel
    posted: list = []
    monkeypatch.setattr(ch, "_post_data", lambda blk: posted.append(blk))
    buf = ctypes.create_string_buffer(16)
    op = ch.recv_into(11, ctypes.addressof(buf), 16)
    ch._do_recv(11, op)
    blk = ch._pool.blocks[1]
    ch._deliver_hold(11, blk, 16)
    assert op.done()
    assert op.data is None  # recv_into settles into the caller buffer
    assert buf.raw == bytes(memoryview(blk.buf)[:16])  # the copy landed
    assert op._blk is None  # block was re-posted, not held
    assert posted == [blk]
    op.release_data()  # no-op, nothing held
    assert ch._recycle_q.qsize() == 0


def test_non_hold_channel_keeps_copy_out_path(hold_channel, monkeypatch):
    ch = LwdUcxChannel(LwdUcxConfig(hold_recv=False, block_mb=1, n_blocks=2))
    try:
        posted: list = []
        monkeypatch.setattr(ch, "_post_data", lambda blk: posted.append(blk))
        op = _prepost(ch, 5)
        blk = ch._pool.blocks[0]
        ch._deliver(5, b"payload")
        assert op.done()
        assert op.data == b"payload"
        op.release_data()  # no held block: no-op
        assert ch._recycle_q.qsize() == 0
    finally:
        ch.close()


def test_pool_reset_excludes_held_blocks():
    pool = LwdUcxBlockPool(3, 1024)
    pool.reset()
    a, b, c = pool.blocks
    pool.reset(exclude={b})
    assert pool.qsize() == 2
    got = {pool.get(0).index for _ in range(2)}
    assert got == {a.index, c.index}
    assert b.index not in got
    # wholesale reset (no exclude) keeps the old behaviour
    pool.release(b)
    pool.reset()
    assert pool.qsize() == 3
