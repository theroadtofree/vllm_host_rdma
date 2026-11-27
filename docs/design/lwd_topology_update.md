# LWD Runtime Topology Update — Increment Contract (item 1)

> Elastic scale-out of the host-RoCE `edge_share` scene, reviewed item by
> item. This document fixes the **increment comparison contract** — the
> foundation of Phase A (detect). Later items (trigger method, control-plane
> link addition, data-plane incremental assembly, digest split) extend this
> document as they are approved.
>
> Related: `lwd_peer_restart_recovery.md` (the restart-recovery design
> this feature builds on).

## 1. Scope and design principle

* **Scope (v1)**: running engines with `data_plane_comm: host` in
  `edge_share`, `cloud_share` and `lwd_cluster`. The HCCL (device-comm)
  path is untouched: the watch is gated on host comm plus the scene/role
  rules below, and the increment contract itself is pure config-layer
  code.
* **Knowledge source**: the yaml file on the edge host doubles as the
  shared store. The operator adds new instances (with their
  `addr/ctrl_port/data_port`) to the file; the engine re-reads it at
  runtime. No separate database, no registration protocol.
* **v1 does not cover shrink**: scale-in keeps its existing semantics
  (kill + heartbeat silence + linked teardown + gate freeze).

## 2. Trigger (decided 2026-11-18: mtime polling)

The engine polls the topology file itself; there is no operator signal
and no separate database:

* **Who polls (2026-11-19: lifted to the shared base, role-aware)**: the
  **shared side of the scene** — `LwdBaseEngineCore` starts the watch
  after a successful assembly when
  `_lwd_topology_watch_enabled()`: the edge engine in `edge_share`
  (clouds join the running shared engine), the cloud engine in
  `cloud_share` (edges join the running shared cloud), and **both roles**
  in `lwd_cluster` (each engine applies only its own instance's
  connections). The other side of a shared scene starts fresh with the
  grown yaml and needs no watch. Device comm never watches — the HCCL
  path is untouched.
* **How**: a daemon thread (`lwd-topology-watch`, cadence
  `_LWD_TOPOLOGY_POLL_INTERVAL_S = 1.0s`) stats the file's mtime. A
  change queues one `(WAKEUP, _LWD_TOPOLOGY_RELOAD_TICK)` into the
  engine's input queue — the same host pattern as the liveness
  heartbeat: the poll thread owns nothing but the stat and the tick, and
  the engine thread (the single writer) re-reads, validates, diffs and
  calls the role's `_lwd_apply_topology_increment` hook (the edge builds
  its halves; the cloud's steps arrive as C-2..C-4 below).
* **Deployment discipline**: edits must go through temp-file + atomic
  rename. A half-written file is caught by parse validation (rejected,
  serving continues); a `touch` with unchanged content no-ops on the
  digest fast path.
* **Reload behavior**: an accepted increment is logged
  (`[LWD] topology increment detected: +N pair(s) ...`) and passed to the
  role's apply hook; a rejected or unparsable file is error-logged with
  the contract reason and serving continues. The startup topology stays
  the diff baseline after apply as well as after rejection. The
  last-seen file digest only suppresses repeat verdicts for unchanged
  content; it is not the diff baseline.

## 3. Phase B: runtime link addition on the control plane (item 3)

Decided 2026-11-18: **SimpleQueue + IO-tick drain** for the command
channel, and runtime connections are **"control-plane ready, gate
rejecting"** until their data-plane link exists (Phase C).

### 3.1 Command channel

* `LwdEdgeControlPlane.add_link(endpoint, connection_id, cloud_dp_size,
  *, done=None)` — any thread may call (the engine's reload handler
  does, fire-and-forget); the command lands in a `SimpleQueue` and the
  IO thread applies it in the next poll gap (≤ one IO tick, 100 ms).
  `done` is set once applied (or refused) for callers that synchronize.
* The IO thread is the **single writer** of everything it grows: the
  transport's DEALER list, the poller, the endpoint/connection tuples,
  the link-index map and the handshake table. No new locks.
* Idempotent: a replayed connection id is dropped with a debug log (the
  live session is never disturbed).
* HCCL neutrality: the queue's only producer is `add_link`; the only
  caller is the reload handler, gated on
  `is_host_comm && scene == edge_share`. Device-comm planes never
  enqueue — the base-class drain hook is a no-op by construction.

### 3.2 Transport

`LwdEdgeTransport.open_link(ctx, endpoint, identity)` opens one more
DEALER mid-run with the same per-link options — `IMMEDIATE=1` keeps
sends to a cloud that has not started at EAGAIN, so the periodic HELLO
keeps retrying until it appears (the same contract a planned-but-absent
cloud has at startup).

### 3.3 Engine and scheduler glue

* `LwdEdgeEngineCore._lwd_apply_topology_increment(increment)`: for each new
  connection — seed the runtime registry
  (`dict[(cloud_id, dp_idx) -> LwdDPConnection]`, engine thread puts,
  IO thread only key-gets), seed the scheduler bookkeeping
  (`lwd_add_runtime_connection`: account + abort list — must exist
  BEFORE the link opens, because the cloud's limits can arrive on the
  IO thread one hello-retry after registration and
  `lwd_update_cloud_params` subscripts them), then queue the plane's
  `add_link`. Idempotent per connection.
* `_lwd_connection_of` resolves runtime connections first (keyed read),
  then the immutable planned set — heartbeats and limits of a runtime
  peer attribute correctly and `_lwd_adopt_cloud` adopts as usual.
* **Gate semantics (Phase B)**: a scoped set whose connections are all
  runtime is rejected with "adopted on the control plane only; the
  data-plane link arrives with the next phase" — before the silence
  check, because the silence block admits whenever nothing is silent,
  and a runtime connection is never a routing candidate until Phase C
  opens it for service.

### 3.4 Tests

`tests/v1/lwd_control/test_lwd_control_plane_add_link.py` (real ZMQ on
loopback): an added link registers with its cloud and carries directed
payloads; a replayed add is a no-op. `test_lwd_topology_watch.py`
additionally covers the engine glue (reload → add_link with the
cloud's endpoint and DP size, registry and scheduler seeding,
idempotent replay) and the gate rejection of a control-plane-only
connection.

## 4. Phase C: runtime data-plane assembly (items 6-7; item 8 needs nothing)

Decided 2026-11-18: a refused data-plane build keeps the connection
**control-plane only, retried by the next detected increment** — no
timers, no new channel.

### 4.1 Scope correction

The queue layer is already lazy: `LwdDataPlane._submit` materializes a
queue on first submit per `(connection, channel, op)` and the host queue
resolves its UCX IO by connection — so item 8 (transfer queues) needs
no work. In `edge_share`, the added cloud is a fresh process reading the
updated yaml, so its listener, tokens and hint MQs exist from its own
init. The running cloud in `cloud_share` and each existing cloud in
`lwd_cluster` instead need the worker RPC and MQ additions in C-2..C-4
below.

### 4.2 Layering

* `vllm/v1/lwd_control/lwd_comm/lwd_data_plane_hooks.py` (new, mirrors
  `lwd_restart_hooks`): `register_data_plane_add_hook(hook)` /
  `notify_data_plane_added(connection) -> bool`. A raising hook is
  logged and counts as a refusal. This process-local hook is used by the
  **edge** engine. The cloud engine cannot use it for its worker-owned
  listener and instead calls `collective_rpc` (C-3 below). Device comm
  never enters the host-gated reload path.
* `vllm-ascend` `lwd_host_channels.py`:
    * `LwdHostLinkPlan` — the link-building inputs (role, block sizes,
    qos, sidecar) snapshotted at init: `get_current_vllm_config()` is
    gone by request time, and the values are process-lifetime
    constants. `_ALLOCATED_MB` is the budget ledger: the startup plan's
    need plus every runtime-built pair, against the same 12 GiB
    per-rank limit.
    * `_build_connection_ios(conn, listener, plan)` — the shared body of
    the init loop and the runtime addition; rolls its own partial
    builds back on failure.
    * `lwd_add_runtime_connection(conn) -> bool` — refuses, never raises:
    uninitialized plane, missing `data_port`, a missing cloud listener,
    budget shortfall, or a build failure (rolled back). It is role-aware
    and idempotent on replay. Registration at the end of
    `lwd_init_host_channels` makes it available to the edge's in-process
    hook; a cloud worker reaches the same helper through its RPC method.
* Engine (vllm): `_lwd_open_runtime_connection_for_service` — notify; on success
  the connection joins `_lwd_serviceable_connections` and the scheduler's
  routing candidates (`lwd_add_runtime_route`, sorted insert so
  round-robin order stays stable). On refusal: error log only.
* **Gate**: the phase-B runtime-only rejection narrows to
  "runtime AND not serviceable" — a serviceable runtime connection falls through to
  the silence check like any planned link.
* `_teardown_cloud_dp` snapshots `_UCX_IOS.items()` (tuple(dict.items())
  completes in C): runtime adds land on the engine thread while a
  boot-change verdict can fire the teardown on the IO thread.

### 4.3 Legacy boundary

`LwdEdgeScheduler`'s int-indexed send-watermark mapping
(`_lwd_connections[dp_idx]`) predates runtime additions: it keeps
meaning link indices of the STARTUP plan. Runtime connections produce
`LwdConnectionId`-keyed watermarks.

### 4.4 Tests

`tests/ut/distributed/lwd_data_plane/test_lwd_host_channels.py`
(+6): builds both links with the right token and closures, ledger
accounting, idempotent replay, budget shortfall refusal, uninitialized
and missing-listener refusal, partial-build rollback, missing data_port.
`test_lwd_topology_watch.py` (+2): refused-then-serviceable retry semantics
(no duplicate control-plane add, gate flips to admit) and the hook
registry contract (empty refuses, a raising hook survives the caller,
a working hook reports the build).

## 5. Item 11: the frontend digest split (attach dimension)

Decided 2026-11-18 (option "narrow the compared digest"): the attach
config check compares a **`frontend_digest`** — the canonical digest
over everything EXCEPT the instance inventory (edges, clouds, links and
the `edges_num`/`clouds_num` declarations) — while the full
`LwdTopology.digest` stays observability-only.

* **Why**: the previous comparison used the full digest, so a frontend
  launched with the grown yaml could never attach to the primary that
  started with the smaller one ("value mismatch: lwd_topology_digest")
  — the elastic scale-out chain would dead-end at the frontend. An
  addition changes only the inventory, so the narrowed digest is stable
  across increments; a change outside it (deployment block, qos, scene)
  still refuses the attach with the key named in the reason.
* **What the subset is**: exactly the R1-frozen blocks plus the scene —
  the same invariants the increment contract enforces on the running
  engine, now enforced symmetrically on the frontend dimension. `qos`
  is included deliberately: the deployment guarantee says it will not
  change, and a violating change SHOULD refuse the attach.
* **Where**:
    * `LwdTopology.frontend_digest`, computed beside `digest` in
    `from_dict` (the shared encoding lives in `_canonical_json`);
    * `FrontendAttachPlan` carries both digests; the payload key switches
    `lwd_topology_digest` → `lwd_frontend_digest` (both sides run the
    same build, so the rename is synchronized by construction);
    * the authority logs both at startup ("LWD registration authority:
    frontend digest X (topology Y)").
* **Unchanged consumers**: the file watcher's dedup and
  `diff_topology`'s no-op check keep the FULL digest — they must notice
  every content change; only the frontend comparison narrows.

## 6. cloud_share scale-out plan (2026-11-19 re-analysis)

The elastic chain above grew the shared EDGE. cloud_share is its mirror:
the running shared CLOUD grows by edges. Re-analysis against this repo
(kill corrected item by item — the old session's three: listener token
registration is already supported by the locked `_routes` on
`LwdUcxListener` (`register` is callable mid-run; `serve_via` registers
synchronously); `expected_edge_ids` is real but shrinks to an atomic
superset swap; the topology digest does not exist in the edge→cloud
join path at all — the frontend-attach digest is the only one, and
cloud_share edges do not attach):

* **C-1 (this change)**: scene-parameterized contract + the watch lifted
  to `LwdBaseEngineCore` with the role-aware gate. The cloud engine
  detects `cloud_share` increments (edges only, links to the existing
  cloud) and `lwd_cluster` increments affecting its own instance; its
  apply hook runs C-2..C-4 below.
* **C-2**: `LwdCloudControlPlane.allow_edge(edge_id)` — an atomic
  frozenset superset swap on the handshake's allowlist — wired from the
  cloud's `_lwd_apply_topology_increment`. Late-edge HELLOs then
  register through the existing runtime path
  (`_lwd_handle_peer_registered` already handles re-registration).
* **C-3**: the cloud engine calls
  `collective_rpc("lwd_build_runtime_connection", args=(connection,))`.
  The endpoint worker invokes role-aware `lwd_add_runtime_connection` to
  build its serve-side pair on the EXISTING shared listener
  (`_UCX_LISTENERS[(cloud_id, dp_idx)]`), reusing `_build_connection_ios`
  — `serve_via` registers the token mid-run (locked), and the budget
  ledger is shared. Every worker extends its connection bookkeeping;
  only the endpoint worker builds and returns `True`, while non-endpoint
  workers return `None`.
* **C-4 (the genuinely new piece)**: the hint/done MQ pair for a late
  edge — the running CLOUD creates `create_hint_mq` + registers the
  hint sender + passes the hint handle explicitly through
  `collective_rpc("lwd_attach_runtime_connection_mqs")`; the worker
  attaches/creates per-connection MQs and the early-recv loop's
  one-time snapshot (`hint_mqs = self._irecv_hint_mqs`) becomes a
  per-iteration `tuple()` snapshot.

**C-2 detail (2026-11-19)**: `LwdCloudHandshake.allow_edge(edge_id)` —
an atomic frozenset superset swap (copy-on-write; the IO thread's
`_mismatch_reason` sees either the old or the new set, never a
half-built one). `LwdCloudControlPlane.allow_edge` forwards; the cloud
engine's `_lwd_apply_topology_increment` calls it per new edge and logs
the admission. Idempotent: a replayed increment re-places the same
element harmlessly. The rest of the join chain — registration seqno
reset, scheduling-limits resend, heartbeat fan-out from `_lwd_edges`,
unreachable-peer pruning — is the existing runtime machinery; the
cloud's `_lwd_connection_of` is a pure constructor (no registry
needed). Tests: `test_lwd_handshake_allow_edge.py` (5: reject before /
accept after / idempotent + live sessions / empty allowlist stays
open / same-session retry is a refresh) and `test_lwd_topology_watch.py`
(+1: the cloud's apply hook fires allow_edge per detected edge).

**C-3 detail (2026-11-19)**: `lwd_add_runtime_connection` is
role-aware. The EDGE path is unchanged (`listener=None`, connect out).
The CLOUD path looks up the shared listener from
`_UCX_LISTENERS[(cloud_id, dp_idx)]` (bound at init, still alive — no
re-bind), passes it to `_build_connection_ios` which sets `serve=True`
— `LwdUcxChannel.serve_via` registers the connection's two tokens on
the listener's **locked route table** mid-run (the accept loop reads it
under the same lock, so a retrying edge's next UHC1 handshake
completes). The idempotent key is role-aware (edge: UP-send, cloud:
UP-recv). The budget ledger is shared. A cloud without a listener
(sidecar daemon) refuses — the edge's UP frames keep bouncing and the
next increment retries. Because this helper and the listener live in
the worker process, the cloud engine calls
`collective_rpc("lwd_build_runtime_connection")`, not
`notify_data_plane_added`. The RPC extends `_lwd_connections` and
`_lwd_up_src_ranks` on every worker, builds only on the endpoint worker,
and treats `any(result is True)` as success; `None` means non-endpoint,
not refusal. A refusal leaves the connection control-plane only for a
later increment to retry.

**C-4 detail (2026-11-19)**: the hint/done MQ pair for a late edge —
the ONE genuinely new mechanism of the cloud_share chain (in
edge_share, a new cloud is a fresh process whose MQs exist from its
own init):

* **Engine** (`_lwd_wire_runtime_hint_mq`): `create_hint_mq`,
  `register_hint_sender`, then
  `collective_rpc("lwd_attach_runtime_connection_mqs")` with the encoded
  hint MQ handle as an explicit RPC argument. The returned done MQ
  handle is stored for the drain loop. Idempotent per connection;
  failures roll back local sender/MQ state so a later increment retries.
* **Worker** (`lwd_attach_runtime_connection_mqs`): attaches the hint MQ
  from the RPC argument (runtime environment variables do not cross the
  executor process boundary), creates the done MQ, adds both to the
  per-connection dicts, and returns the done MQ handle. Idempotent.
* **Early-recv loop**: the one-time snapshot
  (`hint_mqs = self._irecv_hint_mqs or ...`) became a per-iteration
  `tuple()` snapshot — a runtime-added connection joins the loop on
  the next tick (the engine thread's insert and the early-recv
  thread's iteration are concurrent; `tuple(dict.items())` completes
  in C, a Python-level iteration would raise). This also closes the
  latent P1-family risk for the STARTUP connections' dict.

Acceptance: 1-edge cloud_share running → grow the cloud host's yaml to
2 edges → launch edge 1 → the cloud's watch fires, the edge registers,
its UCX pair builds on the shared listener → requests through edge 1
serve, edge 0 never drops.

**lwd_cluster routing detail (2026-11-19)**: the above chains grow the
LINK; this piece grows the probe TARGETING, without which a grown
lwd_cluster serves on both clouds but routes every request to the
startup cloud. The edge probe manager's connection/URL map and health
rotation were frozen at startup. The current direct-DP coordination
schema stores each target URL on the cloud DP itself; the retired
edge-side `control_url` / `control_url_map` fields play no role. The fix
is connection-grained:

* **`LwdCloudHealthTracker.add_target((cloud_id, dp_idx))`** extends the
  rotation by one atomic tuple swap (`lwd_rotate_targets`), so a
  concurrent `candidates()` reader sees either the old or the new
  rotation, never a torn one. Idempotent.
* **`LwdEdgeProbeManager.register_connection(connection_id, url)`**
  first registration wins; it stores the URL by `LwdConnectionId` and
  calls `add_target`. Probe submission intersects these health targets
  with the requesting edge's visible connections. The drain-time block
  size rebuild uses the same connection-keyed URL map, and `enabled`
  counts runtime registrations.
* **The edge engine's `_lwd_apply_topology_increment`** calls
  `register_connection` before the open-for-service attempt. An explicit cloud DP
  `coord_url` wins; otherwise `LwdTopology.from_dict` derives it from
  the DP's `addr` and `coord_port`. A DP with no coordination endpoint
  registers nothing. A refused data-plane build never rolls the probe
  registration back, and a cloud that is still loading accrues probe
  failures → cooldown → recovery under the tracker's existing semantics.

The same increment-apply filter was made scene-aware: only
lwd_cluster filters per instance (`connection.edge_id !=
instance_id`); edge_share's shared engine must NOT filter — its
startup `lwd_config.connections` cannot contain the logical edges the
increment adds (the startup topology remains the diff baseline and never
advances), so a snapshot-based filter dropped exactly the connections
the scale-out was supposed to add. `cloud_share` edges never watch; the
cloud side's instance filter (`cloud_id == instance_id`) was already
correct, and `lwd_cluster` clouds use the same filter.

Acceptance: 2-edge 1-cloud lwd_cluster running → grow to 2 clouds →
launch cloud 1 → each edge's watch fires, the control plane and data
plane build, and the probe rotation widens to cloud 1 → requests
spread over both clouds (round-robin preference via probe completions;
an not serviceable or cooling cloud falls back to the serviceable candidates).

**Partial-death routing (2026-11-20)**: a stopped cloud must not keep
eating traffic while its sibling serves. The `LwdHeartbeatMonitor`'s frozen
verdict (`verdict.silent_routes`) is published at its single event exit
point and mirrored into the scheduler at BOTH transitions — death (sweep)
and recovery (activity) — via
`lwd_update_silent_routes`; `_lwd_route_request` filters the silent
routes out of the round-robin candidates, failing open when every
candidate is silent (the gate owns the fully-silent rejection at
admission; the transfer containment owns the residual admit/route
race). A probe preferred pointing at a silent cloud falls back the
same way. Probe cooldown already keeps the dead cloud out of probe
targeting; this closes the preferred=None path that round-robined
into it.

## 7. Processing order (caller flow, Phase A)

```text
trigger (host comm; the shared side of the scene — the edge engine in
         edge_share, the cloud engine in cloud_share)
→ LwdTopology.from_file(path)   # parse + the FULL standard validation
                                  # (scene shape, port uniqueness, count
                                  # consistency, contiguous ids, link
                                  # completeness). Failure: error log, keep
                                  # the old snapshot, serving continues.
→ diff_topology(startup, new)  # this contract: pure-addition verdict;
                               # the startup topology is always the baseline
→ LwdTopologyIncrement | LwdTopologyUpdateRejected
```

Standard validation runs **first**, so count mismatches, id gaps and port
collisions never reach the diff — they fail at parse with the existing
messages.

## 8. Rules (all must hold to accept)

| Rule | Content | Why |
| --- | --- | --- |
| R0 | the scene stays the same. `edge_share`, `cloud_share` and `lwd_cluster` have runtime-update semantics; only `single_instance` increments are refused with "no runtime-update semantics; restart the engine" | a same-shape `single_instance` file parses clean, so the scene must be compared explicitly |
| R1 | global blocks frozen: `deployment` (minus `edges_num`/`clouds_num`, which follow the added instances), `feature_ctrl`, `control_plane`. **`qos` is exempt by deployment guarantee, not by code** (2026-11-18 review decision) | they size the realized pools (`up_buffer_size`), seed channel scrambling (`feature_ctrl.scramble`) and configure the control plane — none can be applied retroactively |
| R2 | every realized instance survives byte-identical — dp list and coordination block included, with **no exemption**. The old `control_url_map` exemption is retired: that field is rejected at parse time, and a new cloud carries its own `coord_url` / `coord_port`, so adding it requires no edit to an existing edge | pins every realized connection; rules out shrink and mid-flight renumbering |
| R3 | links are addition-only | a swap of realized links is a change |
| R4 | the addition shape is the scene's (2026-11-19): **edge_share** adds whole instance pairs (2026-11-18 review decision) and a new link may not cross old and new instances; **cloud_share** adds edges only, and every new link MUST cross — new edge → the existing cloud, per the full-mesh scene; **lwd_cluster** permits edge and/or cloud additions and relies on standard full-mesh validation | each scene's "who joins the running process" differs; the standard scene validation enforces the link shape, and the shared-scene diff asserts it for a pointed message |

Ids are append-only by construction: layout validation requires instance
ids unique and contiguous from 0, so a valid addition can only append the
next id.

## 9. Interface

```python
# vllm/config/lwd/topology.py

@dataclass(frozen=True)
class LwdTopologyIncrement:
    new_edge_ids: tuple[int, ...]
    new_cloud_ids: tuple[int, ...]
    new_connections: tuple[LwdDPConnection, ...]  # Phase B/C input
    new_topology: LwdTopology                      # candidate; not installed as the diff baseline

@dataclass(frozen=True)
class LwdTopologyUpdateRejected:
    reason: str  # names the rule and the exact field

def diff_topology(old: LwdTopology, new: LwdTopology
                  ) -> LwdTopologyIncrement | LwdTopologyUpdateRejected: ...
```

Empty `new_connections` on an accepted verdict = no-op (the file was
touched but its semantics did not change; the digest fast path catches it).

## 10. Diff-baseline semantics

* The engine keeps its **startup topology** in `self.lwd_config.topology`;
  every reload calls `diff_topology(startup_topology, candidate)`.
* The baseline never advances, including after an increment is fully
  applied. `new_topology` only carries the candidate's new endpoints and
  DP sizes to the apply hooks; it is not assigned back to the engine.
* Re-reading the same grown file therefore produces the same increment
  again. Runtime registries, `add_link`, worker connection state, data-
  plane assembly and MQ wiring make that replay idempotent.
* `_lwd_topology_file_digest` advances solely to deduplicate file-change
  verdicts. It does not change the diff baseline, and a rejected update
  never changes the startup topology either.

## 11. Rejection messages

Pattern `[LWD] topology update rejected: <rule>: <detail>`:

```text
scene changed: edge_share -> single_instance
global block changed: deployment.api_bind_base_port: 61000 -> 62000
global block changed: feature_ctrl.enable_early_recv: False -> True
existing instance modified: cloud 0.dp[0].ctrl_port: 5550 -> 5599
existing instance modified: edge 0.coordination.connect_timeout: 5.0 -> 6.0
instance removed: edge 1 (shrink is not supported; ...)
link removed: edge 0 - cloud 0
new link crosses old and new instances: edge 2 - cloud 0
```

## 12. Tests

`tests/config/lwd/test_topology_update.py` — no-op (digest fast
path, deepcopy), edge_share pair additions, cloud_share edge additions,
scene support/change, frozen deployment / feature_ctrl, qos exemption,
strict R2 with no retired-field exemption, existing instance and
coordination modification, shrink, link swap, id gap at parse, frontend
digest stability, and parse-before-diff ordering.

`tests/v1/lwd_control/test_lwd_topology_watch.py` — the shared-edge watch
chain (detect/apply/replay/refusal/gate/watcher-thread/role-gate), the
cloud's allowlist+collective_rpc apply, and the lwd_cluster scale-out
probe registration (string-keyed yaml payloads; the recording planes
subclass the real control-plane classes because both apply hooks assert
the plane type). `test_lwd_cloud_health.py` — the rotation's runtime
extension: sorted spread, idempotence, cooldown interplay, the atomic
swap under a concurrent reader. `test_lwd_edge_probe.py` — the probe
manager's connection-grained runtime registration: rotation widening,
first-registration-wins, enabled-by-registration, and silent-cloud
exclusion/recovery. The probe/health files skip under the
torch-less lightweight conftest (the full chain needs torch/aiohttp);
the watch file runs everywhere.
