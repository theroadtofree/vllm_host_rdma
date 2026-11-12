# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LWD cross-layer exceptions shared by the engines and the NPU assembly.

``LwdTransferFailedError`` is the containment contract of the peer-restart
line: a data-plane transfer that died mid-batch reaches the owning engine
as this exception, which aborts the batch's requests and keeps the engine
alive, instead of the default escalation where any runner exception is
fatal to the EngineCore (and with it the frontend). See
docs/design/lwd_peer_restart_recovery.md.
"""

from __future__ import annotations

from vllm.config.lwd.topology import LwdConnectionId


class LwdTransferFailedError(RuntimeError):
    """One batch's transfer failed; contain it per request, engine lives on.

    ``req_ids`` is filled when the raiser knows the batch membership (the
    DOWN unembed path); the UP send path leaves it empty and the engine
    resolves the requests from the scheduler's per-connection accounts.
    """

    def __init__(
        self,
        connection: LwdConnectionId,
        seqnos: list[int],
        req_ids: list[str] | None = None,
    ) -> None:
        self.connection = connection
        self.seqnos = list(seqnos)
        self.req_ids = list(req_ids) if req_ids is not None else None
        super().__init__(
            f"data-plane transfer failed on {connection} seqnos={sorted(self.seqnos)}"
        )
