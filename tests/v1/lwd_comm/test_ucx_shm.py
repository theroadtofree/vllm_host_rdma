# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SPSC ring tests of the daemon shared-memory segment (data_comm/ucx_shm.py).

The cmd/cpl rings are the whole daemon-mode control channel: the producer
writes only the tail, the consumer only the head, full is ``tail - head >=
slots`` and empty is ``head == tail``. Indices are u32 and must wrap.

memfd_create is Linux-only, so the segment here is created over a temp
file: same shared mapping, same rings, and the test runs wherever mmap
does.
"""

from __future__ import annotations

import os

import pytest

from vllm.v1.lwd_comm.data_comm import ucx_shm as shm


@pytest.fixture
def segment(tmp_path, monkeypatch):
    """A creator-side segment whose "memfd" is a plain temp file."""
    fd = os.open(tmp_path / "shm.bin", os.O_RDWR | os.O_CREAT, 0o600)
    monkeypatch.setattr(shm, "_memfd_create", lambda: fd)
    seg = shm.LwdUcxShmSegment(
        block_bytes=4096,
        n_staging=2,
        n_recv=2,
        creator=True,
        cmd_slots=4,
        cpl_slots=4,
    )
    yield seg
    seg.close()


@pytest.fixture
def peer(segment):
    """Second mapping of the same fd: the other side of every ring."""
    other = shm.LwdUcxShmSegment(
        block_bytes=4096,
        n_staging=2,
        n_recv=2,
        fd=segment.fd,
        creator=False,
        cmd_slots=4,
        cpl_slots=4,
    )
    yield other
    other._mm.close()


def test_cmd_ring_delivers_entries_in_order(segment, peer) -> None:
    segment.cmd_put(shm.CMD_SEND, tag=7, block_idx=1, size=4096, op_id=11)
    segment.cmd_put(shm.CMD_RECV, tag=9, op_id=12)

    first = peer.cmd_get()
    second = peer.cmd_get()
    assert first == (shm.CMD_SEND, 0, 7, 1, 0, 4096, 11)
    assert second == (shm.CMD_RECV, 0, 9, 0, 0, 0, 12)


def test_ring_reads_empty_once_drained(segment, peer) -> None:
    assert segment.cmd_get() is None
    segment.cmd_put(shm.CMD_CLOSE)
    assert peer.cmd_get() == (shm.CMD_CLOSE, 0, 0, 0, 0, 0, 0)
    assert peer.cmd_get() is None


def test_ring_rejects_a_fifth_entry_instead_of_overwriting(segment) -> None:
    for op_id in range(4):
        segment.cmd_put(shm.CMD_RECV, op_id=op_id)
    with pytest.raises(RuntimeError, match="ring full"):
        segment.cmd_put(shm.CMD_RECV, op_id=99)


def test_released_slot_is_reused_and_indices_wrap(segment) -> None:
    slots = segment.cmd_slots
    for op_id in range(slots):
        segment.cmd_put(shm.CMD_RECV, op_id=op_id)
    assert segment.cmd_get()[6] == 0

    # one slot free again: the tail keeps growing past the slot count
    segment.cmd_put(shm.CMD_RECV, op_id=1000)
    drained = [segment.cmd_get()[6] for _ in range(slots)]

    assert drained == [1, 2, 3, 1000]
    assert segment.cmd_get() is None


def test_cpl_ring_carries_block_index_error_and_size(segment, peer) -> None:
    segment.cpl_put(shm.CPL_RECV_DONE, op_id=5, size=2048, err=0, block_idx=3)
    segment.cpl_put(shm.CPL_SEND_DONE, op_id=6, size=4096, err=12)

    assert peer.cpl_get() == (shm.CPL_RECV_DONE, 0, 5, 2048, 0, 3)
    assert peer.cpl_get() == (shm.CPL_SEND_DONE, 0, 6, 4096, 12, 0)
    assert peer.cpl_get() is None


@pytest.mark.parametrize("ready,broken", [(False, False), (True, False), (True, True)])
def test_status_words_are_visible_to_the_peer(segment, peer, ready, broken) -> None:
    segment.set_status(ready=ready, broken=broken)
    assert peer.get_status() == (ready, broken)


def test_block_areas_are_shared_between_the_two_mappings(segment, peer) -> None:
    payload = b"uhc1-block-payload"
    segment.staging_view(1)[: len(payload)] = payload
    assert peer.staging_view(1)[: len(payload)] == payload

    segment.recv_view(0)[: len(payload)] = payload
    assert peer.recv_view(0, len(payload))[:] == payload

    assert segment.staging_addr(1) != segment.staging_addr(0)
    assert peer.staging_addr(1) - peer.staging_addr(0) == segment.block_bytes
    assert segment.recv_addr(1) - segment.recv_addr(0) == segment.block_bytes


def test_engine_side_rejects_a_foreign_segment(tmp_path, monkeypatch) -> None:
    fd = os.open(tmp_path / "foreign.bin", os.O_RDWR | os.O_CREAT, 0o600)
    os.ftruncate(fd, shm.segment_size(4096, 2, 2, cmd_slots=4, cpl_slots=4))
    with pytest.raises(RuntimeError, match="magic mismatch|version mismatch"):
        shm.LwdUcxShmSegment(
            block_bytes=4096, n_staging=2, n_recv=2, fd=fd, creator=False
        )
    os.close(fd)
