# LWD Host Data Plane

The default LWD data plane (see [lwd_data_plane.md](lwd_data_plane.md)) runs inside one HCCL world shared by the edge and the cloud: channel groups are derived from topology ranks, and every process joins the same collectives.

`deployment.data_plane_comm: host` replaces that with a **host-host RoCE transport** (UCX, wire protocol UHC1). The edge and the cloud become two independent torch worlds: no shared HCCL world, no topology ranks, no cross-side collectives, no `LwdParallelGroupsSpec`. Tensors travel over RoCE as tagged byte streams between peer processes.

| Aspect | `device` (default) | `host` |
| --- | --- | --- |
| Data path | HCCL P2P inside one world | UCX/UHC1 over RoCE |
| Rank model | Global ranks from the topology | No ranks; peers derive endpoints from identity |
| Group creation | Topology-derived groups in `initialize_model_parallel` / `init_ascend_model_parallel` | Skipped; each instance builds its native TP/DP groups |
| Cloud listener | Implicit in the HCCL world | One `data_port` per cloud DP |
| Sizing | HCCL P2P needs matching buffers | Block sizes from the topology, verified per link at handshake |
| Deviceless machines | NPU only | Edge/cloud host processes can run without an NPU device |

## Topology

```yaml
deployment:
  mode: 0
  scene: single_instance
  edges_num: 1
  clouds_num: 1
  data_plane_comm: host          # device | host
  data_plane_host:               # host-only block
    up_buffer_size: 128          # MB; omit to derive it from the model
    down_buffer_size: 32         # MB
    deployment: in_process       # in_process | sidecar
edges:
  - id: 0
    dp:
      - {dp_idx: 0, addr: 10.0.0.1, ctrl_port: 5560}
clouds:
  - id: 0
    dp:
      - {dp_idx: 0, addr: 10.1.0.1, ctrl_port: 5560, data_port: 5700}
qos:
  data_plane: {traffic_class: 106, sl: 5}   # optional; defaults to 138 / 4
  ctrl_plane: {dscp: 26}                    # optional; defaults to 48
```

| Field | Meaning |
| --- | --- |
| `deployment.data_plane_comm` | Transport family; `host` selects this document |
| `deployment.data_plane_host.up_buffer_size` | UP block in MB. Omitted (`null`) = derived from `hidden_size`, hash layers and `max_num_batched_tokens`, rounded up to 32 MiB |
| `deployment.data_plane_host.down_buffer_size` | DOWN block in MB, used as configured (default 32) |
| `deployment.data_plane_host.deployment` | `in_process` = transport inside each engine; `sidecar` = one transport daemon per host |
| `dp[].data_port` | Cloud-only, required on every cloud DP in host mode; the edge never listens |
| `deployment.hccl_world_size`, `dp[].ranks` | Must be omitted in host mode |

`submit_wait_ready_s` is not a YAML field. The bounded submit-time reconnect wait is fixed by the code constant `LWD_SUBMIT_WAIT_READY_S = 5.0` in the vllm-ascend host-channel wiring and passed only to the internal `LwdNpuUcxChannelIoConfig`.

`LwdTopology.validate_layout` dispatches to `_validate_host_layout`, which fails startup when ranks or `hccl_world_size` are present, when an edge declares `data_port`, when a cloud DP lacks it, or when one address repeats a `data_port`. The DP layout follows the role on both data planes: the engine dispatch requires `data_parallel_size == 1` for the edge (one process serves every DP connection) and `data_parallel_size == len(instance.dp)` for the cloud, and the control plane re-checks it with the launched DP size. Host mode does not support tensor parallelism on the edge: only edge TP rank 0 holds the host links and there is no cross-TP aggregation path, so a TP>1 edge fails at its first UP submit with a missing-IO error. Cloud tensor parallelism is unaffected.

## Link model

Every DP connection owns **two** UHC1 links, one per direction, each an independent `LwdUcxChannel` with its own UCP endpoint, IO thread and block pool.

- The cloud DP binds one shared `LwdUcxListener` on its `data_port`; both links attach through `serve_via`. The listener is created once per cloud DP and reused by every edge connected to that DP; the handshake token routes each incoming connection to its own channel. See [lwd_host_multi_edge_cloud.md](lwd_host_multi_edge_cloud.md).
- The edge connects out for both links: UHC1 is full-duplex, so which side binds the handshake socket is independent of the data direction.
- Endpoint process of a connection: `dp_rank == dp_idx` and TP rank 0. Other ranks hold no link objects. An edge is a single-rank world in host mode (`tensor_parallel_size=1`); only the cloud may use tensor parallelism.

```mermaid
flowchart LR
    subgraph Edge["edge instance (own world)"]
        EU["endpoint rank<br/>dp_rank=0, tp_rank=0"]
    end
    subgraph Cloud["cloud instance (own world)"]
        CL["LwdUcxListener<br/>data_port 5700"]
        CU["endpoint rank<br/>dp_rank=0, tp_rank=0"]
    end
    EU -- "UP: connect + send" --> CL
    CL --> CU
    CU -- "DOWN: shared listener" --> CL
    CL -- "recv" --> EU
```

Handshake: the edge sends a fixed header (`magic, addr_len, max_msg, session, token, n_slots, caps_len, put_thresh`) followed by its worker address; the token is `blake2b-8` over `"edge_id:cloud_id:dp_idx:channel"` truncated to 64 bits, so a connection that carries the wrong identity is rejected by the listener's token router instead of being accepted by the wrong channel.

Warmup is asymmetric by design: the cloud starts first and only reports the listener bound, adopting each edge asynchronously. On the edge, the first local link gets the long handshake window (the 600 s transport handshake timeout plus 5 s); remaining links then share a 60 s straggler grace, after which the edge proceeds with the ready subset. No ready link at all fails startup. There is no cross-world barrier to join.

## Transport

`vllm/v1/lwd_comm/data_comm/` is device-agnostic: no torch, no NPU calls.

| Concern | Behaviour |
| --- | --- |
| Framing | seqno is the tag; zero frame header, zero software CRC |
| Receive side | pre-posted wildcard receives from a built-in block pool |
| Send side | `acquire_staging()` takes a pool block, one D2H copy writes into it, `send(tag, addr)` posts it, the completion callback returns it |
| Backpressure | pool depth = in-flight cap |
| Large frames | RMA-put path is disabled (`put_thresh = 0`); a frame larger than the block fails at submit, before queueing |
| Threading | one library IO thread drives completions; user callbacks run on it and must not block |

The NPU assembly (`vllm_ascend/.../lwd_data_plane/host/`) owns the device side: D2H/H2D copies, the per-request pinned staging, the scrambler and the `LwdTransferFuture` wiring. It faces the transport only through `LwdUcxTransportPort`, so an embedded channel and a daemon-attached channel are interchangeable.

## Sidecar mode

With `deployment: sidecar` the listener lives in a dedicated transport daemon, one per host:

- the first engine process spawns it with the parent environment and sets `LD_LIBRARY_PATH` for libucp;
- the engine attaches over a ctl unix socket (protocol version is checked on attach) and passes memfd-backed shared memory with `SCM_RIGHTS`;
- the data path is an SPSC command ring (engine → daemon) and completion ring (daemon → engine) over that shared memory, with two pipe pairs for wake-up instead of polling;
- the daemon owns every channel and listener; an engine exit detaches its sessions, and a daemon with no session for 30 s exits.

`in_process` keeps the same channel code inside the engine and is the default: fewer moving parts, one process to observe.

## Block sizing

Block size is per direction and must match on both peers; it is verified at handshake.

- `up_buffer_size` omitted → `ucx_block_policy.resolve` derives `max_num_batched_tokens × per-token elements × 2 bytes`, rounded up to 32 MiB, with a 128 MiB floor. `per-token elements` is `hidden_size`, plus `4 × num_hash_layers × num_experts_per_tok` for `deepseek_v4`. Nested `text_config` values are used when the top level lacks them.
- An explicit value wins; if it is below the derived requirement, startup logs a warning and the frame that does not fit fails later at submit.
- Unknown model type → fallback 128 MiB with a warning.

Initialization also enforces a per-rank host memory budget (`_HOST_MEM_BUDGET_MB = 12 GiB`, sized for the multi-instance scenes): `pools × 4 × (up + down)` must fit, otherwise startup fails with the computed numbers rather than exhausting host memory at runtime.

## QoS

QoS is configuration, not environment:

| Plane | Field | Applied as |
| --- | --- | --- |
| Data | `qos.data_plane.traffic_class` | `IB_TRAFFIC_CLASS` in the UCT config of the link, plus the same byte as `IP_TOS` on the TCP handshake sockets (the field is the 8-bit RoCE traffic class, `DSCP << 2 \| ECN`) |
| Data | `qos.data_plane.sl` | `IB_SL` in the same UCT config |
| Control | `qos.ctrl_plane.dscp` | ZMQ `TOS = dscp << 2` on every control socket |

Both peers must carry the same values; effective QoS needs matching switch PFC/ETS policy.

Host comm ships one QoS policy: `LwdTopology` fills every plane the topology omits when `data_plane_comm: host`, so `data_plane` becomes `traffic_class: 138` (DSCP 34 + ECN 2) / `sl: 4` and `ctrl_plane` becomes `dscp: 48` (CS6). A plane written in the topology is kept as is, `null` fields included: the operator owns it then. Device (HCCL) comm is unchanged — LWD marks nothing there, and the HCCL data plane keeps CANN's own `HCCL_RDMA_TC` / `HCCL_RDMA_SL` unless those are set explicitly. Startup logs the effective policy once per process as `[LWD][qos] comm=host data_plane traffic_class=138 (dscp=34 ecn=2) sl=4 | ctrl_plane dscp=48 (tos=192)`, and the full record stays in the `[LWD][config] resolved=...` snapshot.

## Failure model

- Startup fails fast: topology validation, missing `data_port`, a frame larger than the block, an exceeded memory budget, a handshake timeout.
- Peer-restart recovery (embedded transport) follows [lwd_peer_restart_recovery.md](lwd_peer_restart_recovery.md): Phase 1 re-baselines the affected FIFOs on a session-fingerprint change (first frame wins), drops the cloud's early-recv ledger for that connection, and waits out a bounded reconnect window at submit (the 5 s code constant `LWD_SUBMIT_WAIT_READY_S`). Phase 2 adds control-plane liveness, linked teardown and the availability gate; Phase 3 adds frontend/primary-edge restart self-healing. In sidecar (daemon) mode the restart event does not cross the ctl protocol, so a restarted peer still means restarting the whole deployment.
- A broken link surfaces as channel error; the request fails instead of hanging.

## Code map

| Part | Location |
| --- | --- |
| Transport (device-agnostic) | `vllm/v1/lwd_comm/data_comm/`: `sock.py`, `ucx_pool.py`, `ucx_ucp.py`, `ucx_link.py`, `ucx_channel.py`, `ucx_shm.py`, `ucx_port.py`, `ucx_daemon.py` |
| Configuration and validation | `vllm/config/lwd/topology.py` (`LwdHostDataPlane`, `LwdQos*`, `_validate_host_layout`) |
| World-args guard | `vllm/distributed/parallel_state.py`, `vllm/distributed/lwd/lwd_parallel_groups.py` |
| NPU assembly | `vllm_ascend/distributed/lwd_data_plane/host/`: `lwd_host_channels.py`, `ucx_io.py`, `ucx_block_policy.py`, `lwd_host_transfer_queue.py` |
| Queue selection | `vllm_ascend/distributed/lwd_data_plane/lwd_data_plane.py` (`_queue_class`) |
| Ascend groups | `vllm_ascend/distributed/parallel_state.py` (`_lwd_init_ascend_model_parallel`) |

## Tests

| Test | Covers |
| --- | --- |
| `tests/v1/lwd_comm/test_sock.py` | TCP helpers: handshake round-trip, fd ownership across wrap/detach, bounded connect timeout, peer-close return |
| `tests/config/lwd/test_topology.py` | Host layout validation errors and `data_plane_host` misuse |
| `tests/distributed/test_lwd_parallel_groups.py` | Group derivation, plus the host parallel checks: edge TP>1 rejected, edge TP=1 and cloud TP>1 accepted |
| `tests/ut/distributed/lwd_data_plane/test_lwd_host_channels.py` | One shared listener per cloud DP, reused by every edge of that DP; edges and sidecar engines build none |
| `tests/ut/distributed/lwd_data_plane/test_lwd_ucx_block_policy.py` | Block derivation, explicit override, fallback, invalid inputs |
| `tests/ut/distributed/lwd_data_plane/test_lwd_ucx_frame_guard.py` | Submit-time rejection of an oversized frame |
| `tests/ut/distributed/lwd_data_plane/test_lwd_host_channels.py` | Registry layout per scene: one shared listener per cloud DP across edges, sidecar attach, endpoint filtering and the data_port failure branch |
| `tests/ut/distributed/lwd_data_plane/test_lwd_peer_restart_queue.py` | Lazy seqno baseline, reset vs stale holds and skip holes, reset MISS, connection stays submittable |
| `tests/ut/worker/test_lwd_cloud_worker_restart.py` | Early-recv ledger drop scoped to the restarted connection, module-hook MISS |

Anything that needs libucp, RoCE hardware or an NPU requires verification on the real deployment; historical results from another branch do not establish acceptance on this branch. See the runbooks in the RFC issue.

## Multi-edge and multi-cloud

`cloud_share` (several edges share one cloud DP), `lwd_cluster` (one edge reaches every cloud) and `edge_share` are supported in host mode. Constraints:

- one listener per cloud DP, shared by every edge connected to it;
- edge `tensor_parallel_size=1`; cloud TP is unrestricted;
- the fixed 12 GiB host memory budget, which covers the current specifications;
- the host endpoint fixed at each cloud DP's TP0, with no rotation across TP ranks.

See [lwd_host_multi_edge_cloud.md](lwd_host_multi_edge_cloud.md) for the design, its acceptance criteria and the failure paths.

## Out of scope

- Sidecar/daemon peer-restart recovery: restart events do not cross the ctl protocol; this mode keeps the whole-deployment-restart semantics described above.
- GPU-side and CPU-side assemblies of the same transport.
