# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-cloud-DP account of the edge scheduler.

One account holds everything the edge knows about one cloud DP rank: its
seqno stream (the UP pairing contract is per channel, so every engine has
its own), its pending prefill batches and its gate state. The scheduler
keeps one account per DP connection plus one routing table (request id ->
dp_idx) that is pinned at admission and never changes: a request's KV and
scheduler state live only in its target engine, so every chunk, abort and
result travels through the account of the engine the request was routed
to.
"""

from __future__ import annotations

from vllm.config.lwd import LwdConnectionId

LWD_MAX_PENDING_CHUNKS = 10


class LwdCloudDpAccount:
    """The edge-side bookkeeping of one edge -> cloud DP connection."""

    def __init__(
        self,
        connection_id: LwdConnectionId | int,
        max_pending_chunks: int = LWD_MAX_PENDING_CHUNKS,
    ) -> None:
        if isinstance(connection_id, int):
            connection_id = LwdConnectionId(0, 0, connection_id)
        self.connection_id = connection_id
        self.max_pending_chunks = max_pending_chunks
        self.num_speculative_tokens = 0
        self.prefix_use_eagle = False
        self.prefix_alignment_tokens = 0
        self.prefix_alignment_required = False
        self.seqno = 0
        """Next UP pairing number of this engine; taken only by a range
        notify that was sent."""
        self.pending_batches: dict[int, set[str]] = {}
        """seqno -> in-batch request ids awaiting UP send completion, request
        settlement, or a covering send watermark."""
        self.gate_held: str | None = None
        """Hold reason of the last gate state (None = open), for HOLD/OPEN
        logging only."""

    @property
    def dp_idx(self) -> int:
        """Compatibility view for diagnostics; never use this as a map key."""
        return self.connection_id.dp_idx

    # ----------------------------------------------------------------- batch

    def take_seqno(self) -> int:
        """Consume the next pairing number for a range notify."""
        seqno = self.seqno
        self.seqno = seqno + 1
        return seqno

    def register_batch(self, seqno: int, request_ids: set[str]) -> None:
        self.pending_batches[seqno] = set(request_ids)

    def pending_chunks(self) -> int:
        return len(self.pending_batches)

    def note_send_watermark(self, watermark: int) -> None:
        """Release this DP's batches whose sends completed in order."""
        for seqno in list(self.pending_batches):
            if seqno <= watermark:
                del self.pending_batches[seqno]

    def drop_request(self, request_id: str) -> None:
        """Remove a request from every batch entry (abort or finish);
        idempotent for requests in no batch."""
        for seqno in list(self.pending_batches):
            batch = self.pending_batches[seqno]
            batch.discard(request_id)
            if not batch:
                del self.pending_batches[seqno]

    # ------------------------------------------------------------------ gate

    def is_gated(
        self,
        slot_cap: int,
        has_resume_inflight: bool,
        *,
        num_running_reqs: int,
    ) -> str | None:
        """Whether new prefill admission to this engine must wait: None
        admits, otherwise the hold reason ("slots"/"chunks").

        A resume in flight is exempt (finish in-flight chunks before new
        ones; also prevents a slot_cap=1 config from deadlocking on
        itself). Otherwise the engine is gated when its request slots are
        full or its pending chunks reached the cap. Slot occupancy counts
        only requests billed to this connection's cloud DP: the contract
        grant (max_num_seqs, force-equal across clouds) belongs to the
        cloud, so connections sharing one cloud split a single grant and
        edge_share connections each carry their own.
        """
        if has_resume_inflight:
            return None
        if num_running_reqs >= slot_cap:
            return "slots"
        if self.pending_chunks() >= self.max_pending_chunks:
            return "chunks"
        return None
