# LWD Runtime Topology Update — Increment Contract (item 1)

> Elastic scale-out of the host-RoCE `edge_share` scene, reviewed item by
> item. This document fixes the **increment comparison contract** — the
> foundation of Phase A (detect). Later items (trigger method, control-plane
> link addition, data-plane incremental assembly, digest split) extend this
> document as they are approved.
>
> Related: `lwd_peer_restart_recovery.md` (restart recovery, the
> already-verified half this feature builds on).

## 1. Scope and design principle

* **Scope (v1)**: the running shared edge engine of `scene: edge_share`
  with `data_plane_comm: host` only. The HCCL (device-comm) path is
  untouched: the whole mechanism is gated on
  `is_host_comm && scene == edge_share && role == edge`, and the contract
  itself is pure config-layer code.
* **Knowledge source**: the yaml file on the edge host doubles as the
  shared store. The operator adds new instances (with their
  `addr/ctrl_port/data_port`) to the file; the engine re-reads it at
  runtime. No separate database, no registration protocol.
* **v1 does not cover shrink**: scale-in keeps its existing semantics
  (kill + heartbeat silence + linked teardown + gate freeze).

## 2. Trigger (decided 2026-11-18: mtime polling)

The engine polls the topology file itself; there is no operator signal
and no separate database:

* **Who polls**: the running **shared edge engine** only
  (`is_host_comm && scene == edge_share && role == edge`; the HCCL path
  never enters the branch — v1 is edge_share-first, cloud_share will
  join as the second consumer with its own addition shape).
* **How**: a daemon thread (`lwd-topology-watch`, cadence
  `_LWD_TOPOLOGY_POLL_INTERVAL_S = 1.0s`) stats the file's mtime. A
  change queues one `(WAKEUP, _LWD_TOPOLOGY_RELOAD_TICK)` into the
  engine's input queue — the same host pattern as the liveness
  heartbeat: the poll thread owns nothing but the stat and the tick, and
  the engine thread (the single writer) re-reads, validates and diffs.
* **Deployment discipline**: edits must go through temp-file + atomic
  rename. A half-written file is caught by parse validation (rejected,
  serving continues); a `touch` with unchanged content no-ops on the
  digest fast path.
* **Phase A behavior (detect only)**: an accepted increment is logged
  (`[LWD] topology increment detected: +N pair(s) ...`) and returned
  for the later phases; a rejected or unparsable file is error-logged
  with the contract reason. Nothing is applied and the realized snapshot
  never advances until Phase B/C exist. The last-seen file digest
  suppresses repeat verdicts for unchanged content.

## 3. Processing order (caller flow, Phase A)

```text
trigger (host + edge_share + edge gate)
→ LwdTopology.from_file(path)   # parse + the FULL standard validation
                                  # (scene shape, port uniqueness, count
                                  # consistency, contiguous ids, one-to-one
                                  # links). Failure: error log, keep the old
                                  # snapshot, serving continues.
→ diff_topology(realized, new) # this contract: pure-addition verdict
→ LwdTopologyIncrement | LwdTopologyUpdateRejected
```

Standard validation runs **first**, so count mismatches, id gaps and port
collisions never reach the diff — they fail at parse with the existing
messages.

## 4. Rules (all must hold to accept)

| Rule | Content | Why |
| --- | --- | --- |
| R0 | scene stays `edge_share` on both sides | a same-shape `single_instance` file parses clean, so the scene must be compared explicitly |
| R1 | global blocks frozen: `deployment` (minus `edges_num`/`clouds_num`, which follow the added instances), `feature_ctrl`, `control_plane`. **`qos` is exempt by deployment guarantee, not by code** (2026-11-18 review decision) | they size the realized pools (`up_buffer_size`), seed channel scrambling (`feature_ctrl.scramble`) and configure the control plane — none can be applied retroactively |
| R2 | every realized instance survives byte-identical — dp list and coordination block included | pins every realized connection; rules out shrink and mid-flight renumbering |
| R3 | links are addition-only | a swap of realized links is a change |
| R4 | additions are **whole instance pairs** (2026-11-18 review decision) | edge_share grows by pairs; one-to-one scene validation already keeps new links off old instances, asserted for a pointed message |

Ids are append-only by construction: layout validation requires instance
ids unique and contiguous from 0, so a valid addition can only append the
next id.

## 5. Interface

```python
# vllm/config/lwd/topology.py

@dataclass(frozen=True)
class LwdTopologyIncrement:
    new_edge_ids: tuple[int, ...]
    new_cloud_ids: tuple[int, ...]
    new_connections: tuple[LwdDPConnection, ...]  # Phase B/C input
    new_topology: LwdTopology                      # snapshot advance target

@dataclass(frozen=True)
class LwdTopologyUpdateRejected:
    reason: str  # names the rule and the exact field

def diff_topology(old: LwdTopology, new: LwdTopology
                  ) -> LwdTopologyIncrement | LwdTopologyUpdateRejected: ...
```

Empty `new_connections` on an accepted verdict = no-op (the file was
touched but its semantics did not change; the digest fast path catches it).

## 6. Snapshot semantics

* The engine holds a **realized snapshot** — the topology its current
  links were built from (initial: the startup yaml).
* The snapshot advances to `new_topology` only **after the increment is
  fully applied** (Phase B/C complete). Phase A (detect) never advances it.
* A rejected update never advances it; the verdict is deterministic, so
  re-reading the same file yields the same result.

## 7. Rejection messages

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

## 8. Tests

`tests/config/lwd/test_topology_update.py` — 13 cases: no-op (digest fast
path, deepcopy), +1 pair, +2 pairs, scene change, frozen deployment /
feature_ctrl, qos exemption (locks the review decision), existing instance
and coordination modification, shrink, link swap, id gap at parse, and the
parse-before-diff ordering.
