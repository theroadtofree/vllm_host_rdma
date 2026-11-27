# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Edge scheduler: per-cloud-DP routing and gating + chunked prefill batch
assembly + control-plane egress (RangeNotify/seqno).

The edge side only does embedding (no decode); the request lifecycle stays
with the native scheduler throughout: enqueue -> EMBED (chunk batch
assembly / range announcement) -> on embedding completion, transition to
decode phase and stay in running awaiting the cloud result -> the cloud
result terminates the request (finish code mapped to native finish), and
late results are idempotently dropped.

One LwdCloudDpAccount holds every per-engine bookkeeping (seqno stream,
pending batches, gate); the routing table is pinned at admission and
never changes, so a request's chunks, aborts and results all travel
through its engine's account. Each schedule step serves one engine: a
RangeNotify, a seqno and a UP channel belong to exactly one engine. With
one cloud DP rank every account method degenerates to the previous
single-cloud behaviour.
"""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.config.lwd import LwdConnectionId
from vllm.lwd_logger import lwd_logger
from vllm.v1.core.sched.output import (
    LwdBatch,
    LwdBatchType,
    LwdEmbedBatch,
    LwdUnembedBatch,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    FinishReason,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_cloud_dp_account import (
    LwdCloudDpAccount,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_cloud_health import (
    lwd_rotate_targets,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_picker import (
    LwdPrefillCandidate,
    lwd_grammar_ready,
    lwd_pick_prefill_batch,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    LwdBaseScheduler,
    LwdReqPhase,
)
from vllm.v1.lwd_control.lwd_comm.lwd_messages import (
    LWD_NOT_FINISHED,
    LwdC2eNotify,
    LwdCloudParamsNotify,
    LwdEdgeToCloudMessage,
    LwdRangeItem,
    LwdRangeNotify,
    LwdRefillNotify,
)
from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
from vllm.v1.lwd_control.lwd_model.lwd_model_family import (
    LwdModelFamily,
    lwd_select_model_family,
)
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request

LwdSendToCloud = Callable[[LwdConnectionId, LwdEdgeToCloudMessage], None]
"""Sends one message on a full edge -> cloud DP connection."""

# Cloud finish code -> edge terminal state (frontend reason goes through the
# output channel; ERROR maps to ABORTED)
_LWD_FINISH_STATUS = {
    FinishReason.STOP: RequestStatus.FINISHED_STOPPED,
    FinishReason.LENGTH: RequestStatus.FINISHED_LENGTH_CAPPED,
    FinishReason.ABORT: RequestStatus.FINISHED_ABORTED,
    FinishReason.ERROR: RequestStatus.FINISHED_ABORTED,
}


def lwd_edge_scheduler_config(vllm_config: VllmConfig) -> VllmConfig:
    scheduler_config = copy.copy(vllm_config)
    scheduler_config.cache_config = copy.copy(vllm_config.cache_config)
    scheduler_config.cache_config.enable_prefix_caching = False
    return scheduler_config


def _lwd_first_connection(
    connections: tuple[LwdConnectionId, ...], edge_id: int
) -> int:
    """Index of the connection of the edge's staggered first DP."""
    pairs = tuple(
        (connection_id.cloud_id, connection_id.dp_idx) for connection_id in connections
    )
    rotated = lwd_rotate_targets(pairs, edge_id)
    if not rotated:
        return 0
    first = rotated[0]
    return next(
        index
        for index, connection_id in enumerate(connections)
        if (connection_id.cloud_id, connection_id.dp_idx) == first
    )


class LwdEdgeScheduler(LwdBaseScheduler):
    """Per-cloud-DP routing/gating + chunked prefill scheduling +
    control-plane egress (RangeNotify/seqno)."""

    def __init__(self, *args, **kwargs) -> None:
        original_config = args[0] if args else kwargs["vllm_config"]
        self._lwd_prefix_enabled = bool(
            original_config.cache_config.enable_prefix_caching
        )
        if args:
            args = (lwd_edge_scheduler_config(args[0]), *args[1:])
        else:
            kwargs["vllm_config"] = lwd_edge_scheduler_config(kwargs["vllm_config"])
        super().__init__(*args, **kwargs)
        self._lwd_prefix_hash_block_size = 0
        self._lwd_pending_refills: deque[SchedulerOutput] = deque()
        self._lwd_announced_refills: set[tuple[LwdConnectionId, int]] = set()
        # Injected by the engine once every cloud accepted the handshake.
        self.lwd_send_to_cloud: LwdSendToCloud | None = None
        self.lwd_refill_abort: Callable[[str], None] | None = None
        self.lwd_on_finish: Callable[[Request], None] | None = None
        self.lwd_on_tokens: Callable[[Request], None] | None = None
        self.lwd_has_pending: Callable[[], bool] | None = None
        # One account per full connection: dp_idx alone is not unique when
        # this edge connects to several clouds.
        lwd_config = self.vllm_config.lwd_config
        assert lwd_config is not None
        self._lwd_instance_id = lwd_config.instance_id
        self._lwd_edge_share = lwd_config.topology.deployment.scene == "edge_share"
        connections = tuple(sorted(c.connection_id for c in lwd_config.connections))
        self._lwd_connections = connections
        # Liveness verdict of the engine's sweep (frozen snapshot,
        # atomically swapped in via lwd_update_silent_routes): a route the
        # sweep declared dead stays out of the routing candidates while
        # its siblings are alive.
        self._lwd_silent_routes: frozenset[LwdConnectionId] = frozenset()
        self._lwd_accounts: dict[LwdConnectionId, LwdCloudDpAccount] = {
            connection_id: LwdCloudDpAccount(connection_id)
            for connection_id in connections
        }
        self._lwd_aborted_req_ids: dict[LwdConnectionId, list[str]] = {
            connection_id: [] for connection_id in connections
        }
        self._lwd_route: dict[str, LwdConnectionId] = {}
        """Request id -> full connection, pinned at admission."""
        self._lwd_last_engine = (
            _lwd_first_connection(connections, lwd_config.instance_id) - 1
        )
        """Round-robin anchor shared by routing and engine picking; advanced
        only by the step that serves a connection, never by a liveness
        query."""
        # max_num_seqs is one cloud DP's seat count (the handshake rejects a
        # mismatch), so the edge serves that many per connection and this
        # product is the concurrency the deployment actually offers.
        lwd_logger.info(
            "[edge-sched] seats: %d per connection x %d connections = %d",
            self.max_num_running_reqs,
            len(connections),
            self.max_num_running_reqs * len(connections),
        )
        # Prefix caching: manager-level hits must stay off (the edge holds
        # no KV cache -- a hit would skip embedding blocks never written);
        # the config-level flag stays on so the engine still builds the
        # block hash chain for cloud-side hits.
        self.kv_cache_manager.enable_caching = False
        # c2e notify queue: appended by the control-plane IO thread with the
        # delivering link's dp_idx, consumed within a scheduling step to
        # produce an UNEMBED batch
        self.unembed_notify_queue: deque[tuple[LwdConnectionId, LwdC2eNotify]] = deque()
        # Per-model-family admission metadata: whole-prompt M-RoPE
        # positions for multimodal families. Computed once at admission from
        # the real prompt (only the edge holds it) and announced with the
        # request; reclaimed by the harvest reconciliation once the prompt
        # is fully sent.
        self.lwd_model_family: LwdModelFamily = lwd_select_model_family(
            self.vllm_config
        )
        # Chunk unit C = min(remaining, C): taken from the local
        # long_prefill_token_threshold (native truncation boundary,
        # naturally reproduced on both sides); when unset (<=0), falls back
        # to the aligned budget max_num_scheduled_tokens. The cloud does not
        # read this value; it overwrites per the in-batch max(items)
        threshold = self.scheduler_config.long_prefill_token_threshold
        self._lwd_chunk_unit: int = (
            threshold if threshold > 0 else self.max_num_scheduled_tokens
        )

    # ------------------------------------------------------------- routing

    def lwd_update_cloud_params(
        self, connection_id: LwdConnectionId, params: LwdCloudParamsNotify
    ) -> None:
        """Apply one cloud's HELLO parameters without mutating global config."""
        account = self._lwd_accounts[connection_id]
        account.num_speculative_tokens = max(0, params.num_speculative_tokens)
        account.prefix_use_eagle = params.prefix_use_eagle
        account.prefix_alignment_tokens = max(0, params.prefix_alignment_tokens)
        account.prefix_alignment_required = params.prefix_alignment_required

    def lwd_add_runtime_connection(self, connection_id: LwdConnectionId) -> None:
        """Seed a runtime-added connection's bookkeeping BEFORE its link
        opens: the cloud's limits can arrive on the IO thread within one
        hello-retry of registration and `lwd_update_cloud_params`
        subscripts these maps. Idempotent. The connection deliberately
        does NOT join the routing candidates — the admission gate keeps
        rejecting it until its data-plane link exists (a later phase),
        and routing an unservable connection is worse than refusing it."""
        if connection_id not in self._lwd_accounts:
            self._lwd_accounts[connection_id] = LwdCloudDpAccount(connection_id)
        self._lwd_aborted_req_ids.setdefault(connection_id, [])

    def lwd_update_silent_routes(self, silent: frozenset[LwdConnectionId]) -> None:
        """Engine thread: the liveness sweep's frozen verdict (a cloud DP
        went silent, or its heartbeat resumed). One atomic frozenset swap;
        routing reads it on the same thread."""
        self._lwd_silent_routes = silent

    def lwd_add_runtime_route(self, connection_id: LwdConnectionId) -> None:
        """Open a runtime connection for service: it joins the routing
        candidates (sorted, so the round-robin order stays stable).
        Idempotent. Legacy int-indexed send watermarks
        (`_lwd_connections[dp_idx]`) predate runtime additions and keep
        meaning link indices of the startup plan."""
        if connection_id in self._lwd_connections:
            return
        self._lwd_connections = tuple(sorted((*self._lwd_connections, connection_id)))

    def _lwd_dp_of(self, request_id: str) -> LwdConnectionId:
        """Routed dp_idx of an active request; unrouted ids are a bug."""
        return self._lwd_route[request_id]

    def _lwd_dp_of_settling(self, request_id: str) -> LwdConnectionId | None:
        """Routed dp_idx, None for a request that already left the books
        (a late result racing its own finish)."""
        return self._lwd_route.get(request_id)

    def _lwd_edge_id_of(self, request: Request) -> int:
        """Logical edge id owning the request. In edge_share the shared edge
        engine serves every logical frontend, whose identity is the attach
        client_index; a standalone edge process owns exactly its
        instance_id."""
        if self._lwd_edge_share:
            return request.client_index
        return self._lwd_instance_id

    def lwd_select_probe_target(
        self, request: Request, hits: dict[LwdConnectionId, int]
    ) -> LwdConnectionId:
        """Select a probe winner without pinning or charging an account."""
        edge_id = self._lwd_edge_id_of(request)
        candidates = {
            connection_id: self._lwd_accounts[connection_id]
            for connection_id in self._lwd_connections
            if connection_id.edge_id == edge_id
        }
        if hits:
            best_hit = max(hits.values())
            candidates = {
                connection_id: account
                for connection_id, account in candidates.items()
                if hits.get(connection_id) == best_hit
            }
        if not candidates:
            raise RuntimeError(f"[LWD] no probe target for edge_id={edge_id}")
        best = min(
            candidates.values(),
            key=lambda account: (
                (
                    self._lwd_connections.index(account.connection_id)
                    - self._lwd_last_engine
                    - 1
                )
                % len(self._lwd_connections),
            ),
        )
        return best.connection_id

    def _lwd_route_request(
        self, request_id: str, edge_id: int, preferred: LwdConnectionId | None = None
    ) -> LwdConnectionId:
        """Pick and pin the target connection for the requesting edge. Only
        connections originating at edge_id are candidates (the shared
        edge_share engine would otherwise balance every logical edge's
        request across every cloud); a healthy preferred route wins when
        supplied, otherwise ordered round-robin decides."""
        accounts = self._lwd_accounts
        candidates = {
            connection_id: accounts[connection_id]
            for connection_id in self._lwd_connections
            if connection_id.edge_id == edge_id
        }
        if not candidates:
            raise RuntimeError(
                f"[LWD] no cloud connection for edge_id={edge_id} "
                f"(request {request_id})"
            )
        # Liveness: a route the sweep declared silent is not a candidate —
        # round-robin must not keep sending requests to a dead cloud while
        # a sibling route is alive (partial silence: the gate admits, so
        # the dead cloud would otherwise eat every other request). Fail
        # open when every candidate is silent: the gate already rejects
        # the fully-silent case at admission, so this covers only the
        # admit/route race, and the transfer containment owns whatever
        # still slips through.
        healthy = {
            connection_id: account
            for connection_id, account in candidates.items()
            if connection_id not in self._lwd_silent_routes
        }
        if healthy:
            candidates = healthy
        if preferred is not None and preferred not in candidates:
            # The probe produced a preference the router cannot honor: the
            # connection is not a serviceable routing candidate (data-plane
            # build refused) or the liveness sweep declared it silent. The
            # request silently falls back — this log is the elastic
            # scale-out diagnosis cue.
            lwd_logger.error(
                "[LWD] preferred %s is not a routing candidate for "
                "edge_id=%d (request %s): candidates=%s silent=%s; "
                "falling back to score/round-robin",
                preferred,
                edge_id,
                request_id,
                [cid for cid in candidates],
                sorted(self._lwd_silent_routes),
            )
        if preferred is not None and preferred in candidates:
            best = candidates[preferred]
        elif len(candidates) == 1:
            best = next(iter(candidates.values()))
        else:
            best = min(
                candidates.values(),
                key=lambda account: (
                    (
                        self._lwd_connections.index(account.connection_id)
                        - self._lwd_last_engine
                        - 1
                    )
                    % len(self._lwd_connections),
                ),
            )
        connection_id = best.connection_id
        self._lwd_route[request_id] = connection_id
        lwd_logger.info(
            "[edge-route] req=%s edge=%d -> %s",
            request_id,
            edge_id,
            connection_id,
        )
        return connection_id

    def _lwd_account_of(self, request_id: str) -> LwdCloudDpAccount | None:
        dp_idx = self._lwd_dp_of_settling(request_id)
        if dp_idx is None:
            return None
        return self._lwd_accounts[dp_idx]

    # ------------------------------------------------------------- stepping

    def has_requests(self) -> bool:
        """Whether this step has anything to do; False parks the engine on
        its input queue.

        Keep stepping while any UP send still needs watermark settlement: a
        full pending gate stops new EMBED batches, but the empty UNEMBED
        step still executes the worker and harvests completed sends, and
        without that term the edge sleeps holding sends no later batch can
        clear.

        Every term is one an event can clear while parked: requests arrive
        on the input queue, and the control plane posts a WAKEUP for c2e,
        prefix-ready and refill notifies
        (`LwdEdgeEngineCore._lwd_dispatch_message`). A term no such event
        follows would park the edge for good -- which is why
        dispatchability asks for prefill work rather than for routed
        requests: a connection whose requests have all sent their prompts
        keeps the latter true in every decode steady state, and the edge
        then runs a worker round trip per step producing nothing, competing
        with the control-plane IO thread for the GIL.
        """
        return (
            bool(self.unembed_notify_queue)
            or bool(self._lwd_pending_refills)
            or (self.lwd_has_pending is not None and self.lwd_has_pending())
            or any(self._lwd_aborted_req_ids.values())
            or any(account.pending_chunks() for account in self._lwd_accounts.values())
            or self._lwd_embed_dispatchable()
        )

    def _lwd_schedule_step(self) -> SchedulerOutput:
        """Dispatch refills and EMBED batches before consuming cloud results."""
        if self._lwd_pending_refills:
            return self._lwd_dispatch_refill(self._lwd_pending_refills.popleft())
        LwdControlDiag.edge_step_activity(
            len(self.running),
            sum(1 for _ in self._lwd_reqs_in_phase(LwdReqPhase.DECODE, self.running)),
        )
        out = self._lwd_schedule_embed()
        if out is not None:
            return out
        return self._lwd_schedule_unembed()

    def _lwd_servable_connection(self) -> tuple[int, LwdConnectionId] | None:
        """The connection this step would serve, with its rotation index.

        Pure by contract: the liveness predicate calls this on every step,
        so advancing the rotation anchor or logging a gate transition here
        would make a query indistinguishable from a dispatch -- the
        rotation would skip connections nobody served and the log would
        flip on every step. Both belong to the dispatching caller.
        """
        connections = self._lwd_connections
        for offset in range(len(connections)):
            index = (self._lwd_last_engine + 1 + offset) % len(connections)
            connection_id = connections[index]
            if not self._lwd_has_prefill_work(connection_id):
                continue
            if self._lwd_accounts[connection_id].is_gated(
                self.max_num_running_reqs,
                self._lwd_has_prefill_chunk_inflight(connection_id),
                num_running_reqs=len(self.running),
            ):
                continue
            return index, connection_id
        return None

    def _lwd_take_servable_connection(self) -> LwdConnectionId | None:
        """`_lwd_servable_connection` for the step that actually serves it:
        advances the rotation anchor and reports the gate transitions."""
        servable = self._lwd_servable_connection()
        for connection_id, account in self._lwd_accounts.items():
            if self._lwd_has_prefill_work(connection_id):
                self._lwd_log_gate_transition(account)
        if servable is None:
            return None
        self._lwd_last_engine, connection_id = servable
        return connection_id

    def _lwd_log_gate_transition(self, account: LwdCloudDpAccount) -> None:
        held = account.pending_chunks() >= account.max_pending_chunks
        if held != account.gate_held:
            account.gate_held = held
            lwd_logger.info(
                "[edge-sched] prefill gate %s: %s pending_chunks=%d",
                "HOLD" if held else "OPEN",
                account.connection_id,
                account.pending_chunks(),
            )

    def _lwd_embed_dispatchable(self) -> bool:
        """EMBED dispatchability: some connection holds prefill work and
        passes its gate."""
        return self._lwd_servable_connection() is not None

    def _lwd_schedule_embed(self) -> SchedulerOutput | None:
        """Chunked batch + budget clamp + RangeNotify of one engine, then
        the EMBED batch (seqno bound to the notify). None when nothing is
        dispatchable.

        The batch is never dropped once scheduled: scheduling already
        advanced num_computed_tokens, so a dropped chunk would never be
        embedded and the next offset would not match the cloud. A notify
        that cannot be sent raises instead."""
        dp_idx = self._lwd_take_servable_connection()
        if dp_idx is None:
            return self._lwd_schedule_abort_notice()
        out = self._lwd_schedule_batch(dp_idx)
        if not out.num_scheduled_tokens:
            return self._lwd_schedule_abort_notice()
        self.lwd_edge_notify_range(dp_idx, out)
        return out

    def _lwd_schedule_abort_notice(self) -> SchedulerOutput | None:
        for dp_idx, request_ids in self._lwd_aborted_req_ids.items():
            if not request_ids:
                continue
            out = SchedulerOutput.make_empty()
            out.lwd_batch = LwdBatch(
                batch_type=LwdBatchType.LWD_EMBED,
                seqno=-1,
                batch_meta=LwdEmbedBatch(
                    aborted_req_ids=request_ids,
                    connection_id=dp_idx,
                ),
            )
            self._lwd_aborted_req_ids[dp_idx] = []
            return out
        return None

    def _lwd_schedule_unembed(self) -> SchedulerOutput:
        """Consume a c2e notify to produce an UNEMBED batch."""
        if self.unembed_notify_queue:
            dp_idx, notify = self.unembed_notify_queue.popleft()
            return lwd_build_unembed_batch(dp_idx, notify)
        return SchedulerOutput.make_empty()

    def update_from_output(self, scheduler_output, model_runner_output):
        """LWD harvest (the return value is the engine step output; none of
        it goes through native bookkeeping): EMBED batches register
        execution progress (finished ones stay in running and transition to
        decode phase); UNEMBED batches deliver per request via the
        worker-filled req_ids x sampled_token_ids; placeholder batches
        without lwd_batch (the underlying layer returns None on normal
        completion, replaced by the engine proxy with a placeholder) only
        consume send watermarks."""
        watermarks = getattr(model_runner_output, "lwd_up_send_watermarks", {})
        for connection_id, watermark in watermarks.items():
            self._lwd_accounts[connection_id].note_send_watermark(watermark)
            self._lwd_announced_refills = {
                key
                for key in getattr(self, "_lwd_announced_refills", set())
                if key[0] != connection_id or key[1] > watermark
            }
        batch = getattr(scheduler_output, "lwd_batch", None)
        outputs: dict[int, list[EngineCoreOutput]] = {}
        finished: dict[int, set[str]] = {}
        if batch is not None and batch.batch_type == LwdBatchType.LWD_UNEMBED:
            notify = scheduler_output.lwd_c2e_notify[0]
            req_ids = getattr(model_runner_output, "req_ids", None) or ()
            sampled = (
                dict(zip(req_ids, model_runner_output.sampled_token_ids))
                if req_ids
                else {}
            )
            outputs, finished = self._lwd_deliver_unembed(notify, sampled)
        # EMBED batches have no local accounting: progress was advanced
        # optimistically during scheduling.
        self._lwd_release_expired_admission_metadata()
        if not outputs:
            return {}
        return {
            client_index: EngineCoreOutputs(
                outputs=outs,
                finished_requests=finished.get(client_index) or None,
            )
            for client_index, outs in outputs.items()
        }

    @staticmethod
    def _lwd_row_disposition(accepted: int, has_tokens: bool, finished: bool) -> str:
        """Per-row delivery disposition (pad-and-skip, ported from the mm
        branch 7ff2ddc3e):

        - "skip": zero-accepted live row (accepted=0 and not finished) -- in
          a mixed step, prefill requests occupy only reserved slots in the
          sampling layout (Ascend MTP sampling only fills rows the draft
          participated in), so the collector computes accepted=0 via
          (sampled != -1).sum(). Such rows are protocol artifacts of the
          mixed step; they are kept in the notify only for DOWN tensor
          alignment: no delivery, no ERROR, no first-token marking (before
          the fix this form used to kill requests as ERROR);
        - "error": declares accepted>0 yet has no token = a real unembed
          failure; ERROR takes precedence over the cloud finish code
          (original semantics preserved);
        - "deliver": everything else (including zero-accepted closing rows
          that carry a finish code -- the termination signal must be passed
          through; swallowing it would lose the completion)."""
        if accepted == 0 and not finished:
            return "skip"
        if not has_tokens and not finished:
            return "error"
        return "deliver"

    def _lwd_deliver_unembed(
        self, notify: LwdC2eNotify, sampled_token_map: dict[str, list[int]]
    ) -> tuple[dict[int, list[EngineCoreOutput]], dict[int, set[str]]]:
        """Per-request delivery: row disposition goes through
        _lwd_row_disposition (zero-accepted live rows are skipped; ERROR
        only when accepted>0 yet token missing); late results not in
        awaiting are idempotently dropped.

        Results are grouped by ``request.client_index`` so each logical edge
        receives only its own outputs (edge0->client 0, edge2->client 2)."""
        outputs: dict[int, list[EngineCoreOutput]] = {}
        finished: dict[int, set[str]] = {}
        # accepted is position-aligned with rows (same contract as
        # lwd_build_unembed_batch); the DISABLE_DOWN bypass carries no
        # accepted, so token presence/absence is used instead
        accepts = notify.num_accepted_tokens
        prefix_nonces = notify.prefix_nonces
        if prefix_nonces and len(prefix_nonces) != len(notify.req_ids):
            raise RuntimeError("[LWD] C2E prefix_nonces is not aligned with req_ids")
        for index, request_id in enumerate(notify.req_ids):
            if prefix_nonces:
                request = self.requests.get(request_id)
                current_nonce = getattr(request, "lwd_prefix_nonce", 0)
                if request is None or prefix_nonces[index] != current_nonce:
                    lwd_logger.warning(
                        "[edge-deliver] drop stale generation req=%s nonce=%d "
                        "current=%d",
                        request_id,
                        prefix_nonces[index],
                        current_nonce,
                    )
                    continue
            finish_reason = self._lwd_finish_code(notify, index)
            finished_now = finish_reason is not None
            sampled_token_ids = list(sampled_token_map.get(request_id, []))
            accepted = accepts[index] if accepts else (1 if sampled_token_ids else 0)
            disposition = self._lwd_row_disposition(
                accepted, bool(sampled_token_ids), finished_now
            )
            if disposition == "skip":
                # Zero-accepted live row: no delivery, no empty output, and
                # no needless debug-tax logging
                continue
            LwdControlDiag.edge_tokens_delivered(
                request_id,
                sampled_token_ids,
                finish_reason,
                self.vllm_config,
            )
            if disposition == "error":
                # [Lwd][eos-trace] point B: leave a trace for kill rows;
                # pair by down_seqno with A (cloud zero-accepted row) / C
                # (worker missing row) to settle the root cause
                LwdControlDiag.trace_edge_kill_row(
                    seqno=notify.down_seqno,
                    rid=request_id,
                    accepted=accepted,
                    finish_code=(
                        notify.finish_reasons[index] if notify.finish_reasons else -1
                    ),
                    batch_rows=len(notify.req_ids),
                    sampled_rows=len(sampled_token_map),
                )
                finish_reason = FinishReason.ERROR
                finished_now = True
            if not self.lwd_edge_deliver_tokens(
                request_id, sampled_token_ids, finished=finished_now
            ):
                lwd_logger.warning(
                    "[edge-deliver] drop stale cloud payload for %s (not awaiting)",
                    request_id,
                )
                continue
            request = self.requests[request_id]
            client_index = getattr(request, "client_index", 0)
            if sampled_token_ids:
                request.append_output_token_ids(sampled_token_ids)
                if self.lwd_on_tokens is not None:
                    self.lwd_on_tokens(request)
            if finished_now and self.lwd_on_finish is not None:
                self.lwd_on_finish(request)
            if finished_now:
                # The ground truth of the cloud's stop decision lands here:
                # native finish settles the account (KV/cache release and
                # the finished_req_ids handover all go through the native
                # path). Settle the books BEFORE finish_requests: the
                # native path removes the request, after which the engine's
                # route entry and pending state would have no owner left
                # to clean them -- a leak would hold the gate forever
                # (both sides then idle: the edge gates every engine, the
                # cloud has no work).
                self._lwd_settle_request(request_id)
                self.finish_requests(
                    [request_id],
                    _LWD_FINISH_STATUS.get(
                        finish_reason, RequestStatus.FINISHED_STOPPED
                    ),
                )
            outputs.setdefault(client_index, []).append(
                EngineCoreOutput(
                    request_id,
                    sampled_token_ids,
                    finish_reason=finish_reason,
                )
            )
            if finished_now:
                finished.setdefault(client_index, set()).add(request_id)
        return outputs, finished

    @staticmethod
    def _lwd_finish_code(notify: LwdC2eNotify, index: int) -> FinishReason | None:
        """Per-request finish code: sentinel = not finished (None);
        position-aligned with req_ids, a mismatch raises IndexError
        fail-fast."""
        code = notify.finish_reasons[index]
        return None if code == LWD_NOT_FINISHED else FinishReason(code)

    @staticmethod
    def _lwd_grammar_ready(request: Request) -> bool:
        """Keep grammar-compiling requests out of the native visible set."""
        if request.status != RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            return True
        structured = request.structured_output_request
        return bool(structured and structured.grammar)

    def _lwd_pick_prefill_batch(
        self, dp_idx: LwdConnectionId
    ) -> tuple[list[str], dict[str, int], int]:
        """Pick routed candidates within decode debt and native edge slots."""
        account = self._lwd_accounts[dp_idx]
        # Decode debt on demand: native DECODE-phase requests routed to
        # this engine reserve spec-token budget once embedding completes
        # (a superset of the retired first-cloud-token count -- the request
        # needs the DOWN capacity from that moment on).
        decode_debt = account.num_speculative_tokens * sum(
            self._lwd_dp_of(r.request_id) == dp_idx
            for r in self._lwd_reqs_in_phase(LwdReqPhase.DECODE, self.running)
        )
        cap = max(0, self.max_num_scheduled_tokens - decode_debt)
        # Native edge slots are shared across DP routes. Waiting requests
        # must not reserve slots before admission.
        free_seq = self.max_num_running_reqs - len(self.running)
        # Resume candidates take only the prefill phase (decode-phase
        # residents have remaining=0, which the picker would skip anyway;
        # after the filter, a non-empty candidate list is itself the resume
        # criterion). Batch engine uniformity: only requests routed to
        # dp_idx enter -- one RangeNotify, one seqno, one UP channel.
        running = [
            LwdPrefillCandidate(
                r.request_id, r.num_prompt_tokens, r.num_computed_tokens
            )
            for r in self._lwd_reqs_in_phase(LwdReqPhase.PREFILL, self.running)
            if self._lwd_dp_of(r.request_id) == dp_idx
        ]
        waiting = [
            LwdPrefillCandidate(
                r.request_id, r.num_prompt_tokens, r.num_computed_tokens
            )
            for r in [*self.kv_holding_waiting, *self.waiting]
            if self._lwd_dp_of(r.request_id) == dp_idx
            and lwd_grammar_ready(
                r, RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
            )
        ]
        req_ids, expected = lwd_pick_prefill_batch(
            running,
            waiting,
            cap=cap,
            free_seq=free_seq,
            chunk_unit=(
                account.prefix_alignment_tokens
                if account.prefix_alignment_required
                and account.prefix_alignment_tokens > 0
                else self._lwd_chunk_unit
            ),
        )
        return req_ids, expected, cap

    def _lwd_schedule_batch(self, dp_idx: LwdConnectionId) -> SchedulerOutput:
        """Chunk batch assembly: the picker selects the batch of one engine,
        scheduled via the base-class visible-set mechanism (budget clamped
        to that engine's cap; the chunk split is guaranteed by the clamping
        parameters to be reproduced natively).

        Only this connection's hidden requests are charged against the seat
        budget; the siblings' seats belong to other cloud DPs."""
        req_ids, _expected, cap = self._lwd_pick_prefill_batch(dp_idx)
        picked = set(req_ids)
        hidden_seats = sum(
            1
            for request in self.running
            if request.request_id not in picked
            and self._lwd_route.get(request.request_id) == dp_idx
        )
        return self._lwd_schedule(
            req_ids, token_budget_cap=cap, hidden_seats=hidden_seats
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int], list[int]]:
        """Edge-side passthrough: encoder execution and caching are managed
        by the worker itself; native rollback would break the chunk
        contract, so nothing is scheduled, consumed, or rolled back."""
        return [], num_new_tokens, encoder_compute_budget, [], []

    def _lwd_has_prefill_work(self, dp_idx: LwdConnectionId) -> bool:
        """Whether this connection holds prefill the picker could batch.

        Same two sources as `_lwd_pick_prefill_batch`: a resume still in
        running, or a queued request routed here. A routed request that
        already sent its whole prompt is not edge work -- only the cloud
        advances it -- so a connection holding nothing else is the normal
        decode steady state, and stepping for it produces nothing.
        """
        if self._lwd_has_prefill_chunk_inflight(dp_idx):
            return True
        return any(
            self._lwd_route.get(request.request_id) == dp_idx
            for queue in (self.kv_holding_waiting, self.waiting)
            for request in queue
        )

    def _lwd_has_prefill_chunk_inflight(self, dp_idx: LwdConnectionId) -> bool:
        """Whether dp_idx has an embed request not fully sent; same
        criterion as the picker resume branch -- the two must stay
        consistent (the gate lets a resume finish before new work)."""
        return any(
            req.num_computed_tokens < req.num_prompt_tokens
            and self._lwd_route.get(req.request_id) == dp_idx
            for req in self.running
            if not req.is_finished()
        )

    def lwd_edge_request_active(self, request_id: str) -> bool:
        if request_id in self._lwd_route or request_id in self.requests:
            return True
        announced: set[tuple[LwdConnectionId, int]] = getattr(
            self, "_lwd_announced_refills", set()
        )
        return any(
            (account.connection_id, seqno) in announced and request_id in request_ids
            for account in self._lwd_accounts.values()
            for seqno, request_ids in account.pending_batches.items()
        )

    def lwd_admit_request(
        self, request: Request, preferred: LwdConnectionId | None = None
    ) -> LwdConnectionId:
        """Scheduler-local half of request admission: mode boundary
        validation, model-family metadata precomputation and routing.
        Returns the routed dp_idx; the cloud announcement and enqueueing
        belong to the engine's add_request; abort_immediately goes through
        the native engine path (the ledger is pruned in the abort_requests
        override)."""
        self._lwd_validate_request(request)
        self.lwd_model_family.admit(request)
        return self._lwd_route_request(
            request.request_id, self._lwd_edge_id_of(request), preferred
        )

    def lwd_edge_prefix_ready(
        self, request: Request, hit_tokens: int, replay_start: int = -1
    ) -> bool:
        hash_block_size = self._lwd_prefix_hash_block_size
        start = hit_tokens if replay_start < 0 else replay_start
        valid = 0 <= start <= hit_tokens < request.num_prompt_tokens and (
            hash_block_size <= 0 or hit_tokens % hash_block_size == 0
        )
        if not valid:
            lwd_logger.error(
                "[prefix] invalid READY hit req=%s hit=%d replay_start=%d",
                request.request_id,
                hit_tokens,
                replay_start,
            )
            request.num_computed_tokens = 0
            return False
        request.num_computed_tokens = start
        if 0 <= replay_start < hit_tokens:
            request.replay_start = replay_start
        return True

    def lwd_edge_cancel_admission(self, request_id: str) -> None:
        self._lwd_settle_request(request_id)
        self.lwd_model_family.release(request_id)

    def lwd_mrope_positions(self, request_id: str) -> bytes:
        """Return full-prompt positions for the engine's RequestNotify."""
        return self.lwd_model_family.prompt_mrope_positions(request_id) or b""

    def lwd_edge_notify_range(
        self, dp_idx: LwdConnectionId, scheduler_output: SchedulerOutput
    ) -> None:
        """Send one batch RangeNotify of dp_idx and attach the EMBED batch;
        a seqno of that engine is only taken by a notify that was sent."""
        built = self._lwd_build_range_items(scheduler_output.num_scheduled_tokens)
        if built is None:
            return
        items, req_ids, token_slices, prompt_offsets, has_mrope = built
        account = self._lwd_accounts[dp_idx]
        seqno = account.take_seqno()
        assert self.lwd_send_to_cloud is not None, "cloud not connected"
        lwd_logger.info(
            "[edge] range announced %s seqno=%d reqs=%d tokens=%d",
            dp_idx,
            seqno,
            len(items),
            sum(i.num_tokens for i in items),
        )
        self.lwd_send_to_cloud(dp_idx, LwdRangeNotify(seqno=seqno, items=items))
        account.register_batch(seqno, set(req_ids))
        # On successful send, assemble the EMBED batch and attach it to SO:
        # seqno is the pairing key of the data-plane tensor sent to the
        # cloud (same value as registered by the cloud-side RangeNotify);
        # the embed payload is this batch's per-request token slices (the
        # edge worker flattens them in req_ids order).
        aborted_req_ids = self._lwd_aborted_req_ids[dp_idx]
        scheduler_output.lwd_batch = LwdBatch(
            batch_type=LwdBatchType.LWD_EMBED,
            seqno=seqno,
            batch_meta=LwdEmbedBatch(
                req_ids=req_ids,
                token_ids=token_slices,
                prompt_offsets=prompt_offsets,
                has_mrope=has_mrope,
                connection_id=dp_idx,
                aborted_req_ids=aborted_req_ids,
            ),
        )
        self._lwd_aborted_req_ids[dp_idx] = []

    def lwd_edge_build_refill(
        self, notify: LwdRefillNotify
    ) -> list[SchedulerOutput] | None:
        request_id = notify.request_id
        request = self.requests.get(request_id)
        dp_idx = self._lwd_dp_of_settling(request_id)
        if request is None or dp_idx is None:
            return []
        prefix_nonce = getattr(request, "lwd_prefix_nonce", 0)
        if notify.prefix_nonce != prefix_nonce:
            lwd_logger.warning(
                "[edge-refill] stale req=%s nonce=%d ignored: current=%d",
                request_id,
                notify.prefix_nonce,
                prefix_nonce,
            )
            return []
        if request.prompt_token_ids is None:
            return None
        start, end = notify.start_row, notify.end_row
        if not 0 <= start < end <= request.num_prompt_tokens:
            return None
        chunk_size = min(
            self._lwd_chunk_unit, self.scheduler_config.max_num_batched_tokens
        )
        if chunk_size <= 0:
            return None

        outputs: list[SchedulerOutput] = []
        for chunk_start in range(start, end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, end)
            scheduler_output = SchedulerOutput.make_empty()
            if request.mm_features:
                scheduler_output.scheduled_new_reqs = [
                    NewRequestData(
                        req_id=request_id,
                        prompt_token_ids=None,
                        mm_features=list(request.mm_features),
                        sampling_params=None,
                        pooling_params=None,
                        block_ids=tuple(),
                        num_computed_tokens=chunk_start,
                        lora_request=None,
                        lwd_mrope_positions=self.lwd_mrope_positions(request_id),
                    )
                ]
            batch_meta = LwdEmbedBatch(
                req_ids=[request_id],
                token_ids=[list(request.prompt_token_ids[chunk_start:chunk_end])],
                prompt_offsets=[chunk_start],
                has_mrope=[False],
                connection_id=dp_idx,
            )
            batch_meta.is_refill = True
            batch_meta.lwd_prefix_nonce = prefix_nonce
            scheduler_output.lwd_batch = LwdBatch(
                batch_type=LwdBatchType.LWD_EMBED,
                seqno=-1,
                batch_meta=batch_meta,
            )
            outputs.append(scheduler_output)
        return outputs

    def lwd_edge_queue_refill(self, outputs: list[SchedulerOutput]) -> None:
        self._lwd_pending_refills.extend(outputs)

    def _lwd_dispatch_refill(
        self, scheduler_output: SchedulerOutput
    ) -> SchedulerOutput:
        batch = scheduler_output.lwd_batch
        assert batch is not None and batch.batch_meta is not None
        meta = batch.batch_meta
        request_id = meta.req_ids[0]
        connection = meta.connection_id
        request = self.requests.get(request_id)
        prefix_nonce = getattr(meta, "lwd_prefix_nonce", 0)
        if (
            request is None
            or getattr(request, "lwd_prefix_nonce", 0) != prefix_nonce
            or self._lwd_dp_of_settling(request_id) != connection
        ):
            lwd_logger.warning(
                "[edge-refill] drop stale queued chunk req=%s nonce=%d",
                request_id,
                prefix_nonce,
            )
            return SchedulerOutput.make_empty()

        start = meta.prompt_offsets[0]
        item = LwdRangeItem(
            request_id=request_id,
            offset=start,
            num_tokens=len(meta.token_ids[0]),
            is_refill=True,
            prefix_nonce=prefix_nonce,
        )
        account = self._lwd_accounts[connection]
        seqno = account.seqno
        assert self.lwd_send_to_cloud is not None, "cloud not connected"
        try:
            self.lwd_send_to_cloud(
                connection, LwdRangeNotify(seqno=seqno, items=[item])
            )
        except Exception:
            lwd_logger.exception("[edge-refill] send failed req=%s", request_id)
            if self.lwd_refill_abort is None:
                raise
            self.lwd_refill_abort(request_id)
            return SchedulerOutput.make_empty()
        account.take_seqno()
        account.register_batch(seqno, {request_id})
        self._lwd_announced_refills.add((connection, seqno))
        batch.seqno = seqno
        return scheduler_output

    def _lwd_build_range_items(
        self, scheduled: dict[str, int]
    ) -> (
        tuple[list[LwdRangeItem], list[str], list[list[int]], list[int], list[bool]]
        | None
    ):
        """Assemble per-request RangeItems and EMBED payload slices.
        Returns None when there are no entries."""
        items: list[LwdRangeItem] = []
        req_ids: list[str] = []
        token_slices: list[list[int]] = []
        prompt_offsets: list[int] = []
        has_mrope: list[bool] = []
        for request_id, num_tokens in scheduled.items():
            request = self.requests.get(request_id)
            if request is None:
                continue
            # _update_after_schedule already advanced num_computed
            # optimistically; the start offset must step back by this
            # step's amount
            offset = request.num_computed_tokens - num_tokens
            req_has_mrope = self.lwd_model_family.has_mrope_positions(request_id)
            items.append(
                LwdRangeItem(
                    request_id=request_id,
                    offset=offset,
                    num_tokens=num_tokens,
                    has_mrope=req_has_mrope,
                    prefix_nonce=getattr(request, "lwd_prefix_nonce", 0),
                )
            )
            req_ids.append(request_id)
            token_slices.append(
                list(request.prompt_token_ids[offset : offset + num_tokens])
            )
            prompt_offsets.append(offset)
            has_mrope.append(req_has_mrope)
        return (
            (items, req_ids, token_slices, prompt_offsets, has_mrope) if items else None
        )

    def lwd_edge_abort(
        self, request_ids: list[str]
    ) -> list[tuple[LwdConnectionId, str]]:
        """Settle aborted requests and return their original DP routes."""
        aborted = set(request_ids)
        self._lwd_pending_refills = deque(
            output
            for output in getattr(self, "_lwd_pending_refills", ())
            if not aborted.intersection(output.lwd_batch.batch_meta.req_ids)
        )
        routed_aborts = []
        for request_id in request_ids:
            dp_idx = self._lwd_dp_of_settling(request_id)
            if dp_idx is not None:
                routed_aborts.append((dp_idx, request_id))
                self._lwd_aborted_req_ids[dp_idx].append(request_id)
            self.lwd_model_family.release(request_id)
            self._lwd_settle_request(request_id)
        return routed_aborts

    def lwd_edge_up_batch_request_ids(
        self, connection: LwdConnectionId, seqnos: list[int]
    ) -> list[str]:
        """Requests whose UP send failed mid-flight (containment input): the
        batch entries still sit in the account's pending bookkeeping and
        release through the abort path that follows this lookup."""
        account = self._lwd_accounts.get(connection)
        if account is None:
            return []
        request_ids: set[str] = set()
        for seqno in seqnos:
            request_ids.update(account.pending_batches.get(seqno, ()))
        return sorted(request_ids)

    def lwd_edge_deliver_tokens(
        self, request_id: str, token_ids: list[int], finished: bool
    ) -> bool:
        """Claim a live cloud result; terminated/unknown requests are
        idempotently dropped."""
        request = self.requests.get(request_id)
        if request is None or request.is_finished():
            # A late cloud result racing the request's own finish: the
            # books may still hold it if the finish path did not pass
            # through here (stale payload drop idempotent). Clean the
            # account, or leaked pending state holds the gate forever.
            self._lwd_settle_request(request_id)
            lwd_logger.warning(
                "[edge-deliver] stale result for req=%s (not awaiting)",
                request_id,
            )
            return False
        return True

    def _lwd_settle_request(self, request_id: str) -> None:
        """Drop all request state on finish/abort. Send watermarks alone
        release pending batches after the first token."""
        account = self._lwd_account_of(request_id)
        if account is None:
            return
        refill_batches = {
            seqno: set(account.pending_batches[seqno])
            for connection_id, seqno in getattr(self, "_lwd_announced_refills", set())
            if connection_id == account.connection_id
            and seqno in account.pending_batches
            and request_id in account.pending_batches[seqno]
        }
        account.drop_request(request_id)
        account.pending_batches.update(refill_batches)
        self._lwd_route.pop(request_id, None)

    def _lwd_release_expired_admission_metadata(self) -> None:
        """Release metadata whose request is gone or whose prefill is complete.

        Resident requests remain in `self.requests` while waiting for cloud
        decode, so request removal alone cannot reclaim admission metadata.
        """
        for request_id in self.lwd_model_family.held_request_ids():
            request = self.requests.get(request_id)
            prefill_finished = (
                request is not None
                and self._lwd_req_phase(request) is LwdReqPhase.DECODE
            )
            if request is None or prefill_finished:
                self.lwd_model_family.release(request_id)

    def _lwd_validate_request(self, request: Request) -> None:
        """Mode boundary validation (edge capability surface): reject
        client-provided prompt_embeds/pooling/sampling params not carried
        on the wire; violations raise ValueError and are rejected before
        enqueueing."""
        if request.prompt_embeds is not None:
            raise ValueError(
                f"[LWD] prefill-only mode does not accept client-provided "
                f"prompt_embeds (request {request.request_id}); the edge "
                "is the embedding owner"
            )
        self._lwd_validate_mm_request(request)
        if request.pooling_params is not None:
            raise ValueError(
                "[LWD] prefill-only mode does not support pooling requests "
                f"(request {request.request_id})"
            )
        sp = request.sampling_params
        unsupported = [
            name
            for name, value in (
                ("logit_bias", sp.logit_bias),
                ("allowed_token_ids", sp.allowed_token_ids),
                ("logprobs", sp.logprobs),
            )
            if value is not None
        ]
        if unsupported:
            raise ValueError(
                "[LWD] prefill-only mode does not support sampling "
                f"param(s) {', '.join(unsupported)} "
                f"(request {request.request_id}): not carried on the "
                "edge->cloud wire, cloud would silently sample without them"
            )

    def _lwd_validate_mm_request(self, request: Request) -> None:
        """Multimodal allowlist: image only (the merge/mrope chain is
        implemented with image semantics; other modalities are unadapted,
        silently letting them through would compute wrongly); reject EVS
        (multimodal pruning)."""
        if not request.mm_features:
            return
        unsupported_mm = sorted(
            {
                feature.modality
                for feature in request.mm_features
                if feature.modality not in ("image", "prompt_embeds")
            }
        )
        if unsupported_mm:
            raise ValueError(
                f"[LWD] prefill-only mode supports image-only "
                f"multimodal requests; got modality "
                f"{', '.join(unsupported_mm)} "
                f"(request {request.request_id})"
            )
        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config.is_multimodal_pruning_enabled():
            raise ValueError(
                "[LWD] prefill-only mode does not support multimodal "
                f"pruning (EVS) (request {request.request_id})"
            )


def lwd_build_unembed_batch(
    dp_idx: LwdConnectionId, notify: LwdC2eNotify
) -> SchedulerOutput:
    """Assemble the UNEMBED batch: cloud results bypass native scheduling;
    use make_empty as the skeleton and attach LwdUnembedBatch of dp_idx."""
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = dict(
        zip(notify.req_ids, notify.num_accepted_tokens)
    )
    scheduler_output.total_num_scheduled_tokens = sum(notify.num_accepted_tokens)
    scheduler_output.lwd_batch = LwdBatch(
        batch_type=LwdBatchType.LWD_UNEMBED,
        seqno=notify.down_seqno,
        batch_meta=LwdUnembedBatch(
            req_ids=list(notify.req_ids),
            num_accept_tokens=list(notify.num_accepted_tokens),
            recv_num_elements=notify.hidden_num_elements,
            out_token_idxs=[],
            top_id_ths=list(notify.top_id_ths),
            token_ids=[list(t) for t in notify.token_ids],
            connection_id=dp_idx,
        ),
    )
    scheduler_output.lwd_c2e_notify = [notify]
    return scheduler_output
