# Compare serve.py lwd_attach_primary Changes

> **Workspace:** c:\mainmain

---

**User**

相比C:\github_main_host_rdma_old\vllm_host_rdma\vllm\entrypoints\cli\serve.py， serve.py为什么增加getattr(args, "lwd_attach_primary", False)

**Assistant**

The user is asking about a difference between two versions of serve.py: one at C:\github_main_host_rdma_old\vllm_host_rdma\vllm\entrypoints\cli\serve.py and the current one at c:\mainmain\vllm_host_rdma\vllm\entrypoints\cli\serve.py. They ask why serve.py added `getattr(args, "lwd_attach_primary", False)`.

I should read both files and compare. Let me look.

Now let's inspect the relevant logic. The key: `lwd_dispatch_attach_frontend(args)` sets `args.lwd_attach_primary = plan.host_comm` in frontend_attach.py line 986. Let me read around there and understand `lwd_dispatch_attach_frontend`.

Also need to compare with the old repo at C:\github_main_host_rdma_old. Let me check whether that path exists.

Let me grep in old path.

Now the question is "相比旧版本，serve.py 为什么增加 getattr(args, "lwd_attach_primary", False)". We should explain why: because host-comm (host data plane) attach primary must run through run_multi_api_server even when api_server_count == 1, since registration server is only built there; otherwise `--api-server-rpc-port` never gets bound and attach frontends hang.

Let's verify the old version around line 156 to see whether it had the condition absent. Let me read the old file's relevant section, plus check whether `lwd_attach_primary` exists in old repo at all.

Also check git history perhaps. Let me read old file 50-170.

Also need to verify LWD_API_MAX_FRONTENDS and plan.host_comm semantics. Let's read the old serve.py relevant slice and frontend_attach definitions (LWD_API_MAX_FRONTENDS, resolve_attach_plan, host_comm).

Also check the test file test_frontend_attach_reregister.py line ~376 for intent.

Let me do these in parallel.

So the difference: new code adds `or getattr(args, "lwd_attach_primary", False)` in the dispatch to run_multi_api_server, plus `lwd_attach_primary` attr set in frontend_attach.

Why: host_comm primary (host data plane edge_share deployments) sets api_server_count = LWD_API_MAX_FRONTENDS (8?), which is > 1, so old condition would already go to run_multi_api_server... Hmm, then why need the flag? Wait, if host_comm, args.api_server_count = LWD_API_MAX_FRONTENDS which is likely > 1, so `args.api_server_count > 1` already routes multi. Unless LWD_API_MAX_FRONTENDS == 1? Let's check. Probably 8. Then the flag is redundant for host_comm? Hmm, but maybe later the `vllm serve` argument validation caps or the `--api-server-count` is overwritten? No.

Wait—maybe there is a path where api_server_count gets reset: In `run_multi_api_server`, `num_api_servers = args.api_server_count`. Hmm.

Hold on. Maybe in current main, LWD_API_MAX_FRONTENDS is 1? Let's check topology.py. Also, maybe host_comm capacity is capped by something in `api_server_count` defaults? Actually the dispatch sets before the default logic; and then the `if args.api_server_count is None` branch skipped. So count = 8 > 1 → multi already.

Hmm, but wait—what about device comm primary? `lwd_attach_primary = plan.host_comm` = False, so no flag. host_comm → count 8, multi. So flag seems redundant... unless LWD_API_MAX_FRONTENDS == 1. Let me check.

Actually maybe the point: host_comm capacity count may be capped somewhere else or the flag guards a future where count==1. Let me read topology.py.

Also consider `--headless` etc. Let's check git log/blame for the added lines to get commit message rationale. Use git log -S.

So LWD_API_MAX_FRONTENDS = 8, so api_server_count = 8 > 1 → run_multi. Then `getattr(args, "lwd_attach_primary", False)` would be redundant? Not exactly—there could be another path: maybe validate_parsed_serve_args or elastic EP caps api_server_count to 1 (line 144-150!). Yes! In serve.py, Elastic EP caps api_server_count to 1. So if host_comm primary also has enable_elastic_ep... Hmm, unlikely.

More importantly, maybe in the future LWD_API_MAX_FRONTENDS could be 1? No.

Hmm, let me think about the actual commit. The test docstring says: "A single-pair attach primary provisions the full reserved frontend capacity on host comm -- api_server_count carries the band capacity, so frontends beyond the yaml plan attach without a primary restart and the serve dispatch reaches run_multi_api_server". Test asserts args.api_server_count == LWD_API_MAX_FRONTENDS if host_comm else 1, and flag == host_comm.

So both count>1 and flag are set for host_comm. The flag is a belt-and-braces / explicit intent: routing decision shouldn't depend on the incidental count value. The comment says "An LWD attach primary must run the attach pipeline (its registration server) even with a single planned frontend; the single in-process path leaves --api-server-rpc-port unbound and attach frontends hang on it."

But "even with a single planned frontend" — the count is capacity 8, not plan count 1, so dispatch already >1. Unless... hmm, is the flag added because api_server_count could be reset elsewhere? Let's check the git commit content and maybe the full diff. Also check whether there's a code path where api_server_count could be 1 for host_comm primary: e.g. user passes --api-server-count=1? resolve_attach_plan validates count must equal plan count or capacity; user passing 1 would match plan count if plan has 1 → capacity_value False → args.api_server_count = 8 anyway. So no.

Maybe elastic EP: if `enable_elastic_ep`, count capped to 1 → then old condition skips multi → single in-process path → registration server not built. That would be a real case: host-comm primary with elastic EP. But elastic EP with edge_share+attach? Possibly.

Also perhaps Ray DP? No.

Let's check where else api_server_count gets modified: `validate_parsed_serve_args`? It runs in validate() before cmd? In cli main, validate then cmd. Check launchers/cli_args.py for API server count reset.

Let's grep "api_server_count =" across repo.

Let's look at commit d81ffbe7a4 diff for serve.py, and the commit message.

No output? Maybe serve.py diff in another commit. Let's search git log for changes to serve.py lines with lwd_attach_primary at line 163. Use git log -L.

Commit 2e676b2aae "fix(lwd): 单前端主节点在主机通信模式下必须使用多服务器路径". Let's see its diff.

Now see frontend_attach.py diff in that commit to understand why api_server_count might be 1.

So originally (before this fix), `lwd_dispatch_attach_frontend` set `args.api_server_count = plan.client_count`, which is 1 for a single planned frontend. So the flag is needed. Later, the capacity change (commit b2d2128cab) changed count to 8 for host comm. But the flag remains (and remains correct/intentional).

Let me verify timeline: 2e676b2aae (Nov 18) added count=client_count + flag. Later b2d2128cab "feat(lwd): 支持超出计划但在容量范围内的前端接入" changed count to LWD_API_MAX_FRONTENDS for host_comm. Let me check that commit to confirm, and whether flag became redundant then. Actually after that change, host_comm→count=8>1, so `or lwd_attach_primary` is redundant at the moment, but kept as explicit routing intent (in case elastic EP capping or config changes). Hmm — but need to answer honestly: the reason it was added, plus current relevance.

Let me check b2d2128cab diff for serve/frontend_attach.

看代码和提交历史后，这个改动的来龙去脉很明确。

## 为什么加这一行

旧版（`C:\github_main_host_rdma_old\...\serve.py`）的路由条件只有：

```python
elif args.api_server_count > 1 or rust_frontend_path:
    run_multi_api_server(args)
else:
    args.api_server_count = None
    uvloop.run(run_server(args))   # 原生单进程路径
```

而 attach 主节点的 `api_server_count` 由 `lwd_dispatch_attach_frontend` 按 YAML 计划里的前端数设置。当部署计划里只有 1 个前端时，`api_server_count == 1`，就落到了最后一个分支的原生单进程路径。

问题在于：注册服务器和 `--api-server-rpc-port` 的绑定只存在于 `run_multi_api_server` 里（通过 [prepare_attach_mode / create_frontend_registration_server](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L990-L997) 创建），单进程路径根本不会构建它。结果是：

- `--api-server-rpc-port` 静默未绑定；
- 所有 attach 前端在 ATTACH 阶段一直等到超时挂起。

host 通信（`deployment.data_plane_comm: host`）的 edge_share 部署正是"单引擎 + 多个前端 attach"的形态，所以这个 bug 在 host comm 下必现。

修复分两处，同属提交 `2e676b2aae`（*fix(lwd): 单前端主节点在主机通信模式下必须使用多服务器路径*）：

1. [frontend_attach.py#L983-L986](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L966-L987)：dispatch 时打上 `args.lwd_attach_primary = plan.host_comm`，告诉 serve 入口"我是需要 attach 流水线的主节点"。
2. [serve.py#L156-L165](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/cli/serve.py#L156-L165)：路由条件 OR 上这个标志，即使 `client_count == 1` 也走 `run_multi_api_server`。

几处设计细节：

- **用 `getattr(..., False)` 而不是 `args.lwd_attach_primary`**：这个属性只有 LWD attach 主节点才会被设置，普通 `vllm serve` 的 argparse Namespace 里没有它，直接访问会 AttributeError。
- **标志用 `plan.host_comm` 而非恒为 True**：device comm 主节点仍走原生单服务器路径，不额外引入多进程/注册服务器开销。
- **为什么放 `args` 上**：dispatch 发生在 `cmd()` 前半段，真正的路由决策在同一函数的后面，但计划解析逻辑在另一个模块，用 Namespace 传一个临时标志最省侵入。

补充一点：后来的提交 `b2d2128cab`（容量化前端接入）把 host comm 主节点的 `api_server_count` 改成了 `LWD_API_MAX_FRONTENDS`（8，见 [frontend_attach.py#L983-L985](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L983-L985)），此时 `> 1` 已能路由到多服务器路径，标志看起来变冗余了；但它仍是有意义的显式意图保护——例如 `enable_elastic_ep` 会在 [serve.py#L144-L150](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/cli/serve.py#L144-L150) 把 `api_server_count` 压回 1，没有这个标志就会重新退回单进程路径、RPC 端口再次不绑定。

---

**User**

对比C:\github_main_host_rdma_old\vllm_host_rdma\vllm\config\lwd\__init__.py，__init__.py新增的LwdTopologyIncrement、LwdTopologyUpdateRejected、diff_topology是干什么的

**Assistant**

Need to compare the two files. Let me read both __init__.py files, and search where these symbols are used.

Read topology.py around 700-960 to get the definitions, and the design doc lwd_topology_update.md for intended contract (R0-R4 rules). Also read lwd_base_engine.py 200-300 for usage.

Now I should also check the design doc for the R0–R4 contract and the role hooks (edge/cloud apply). Let me read docs/design/lwd_topology_update.md key sections briefly to accurately summarize intent. And check lwd_edge_engine/lwd_cloud_engine apply implementations briefly.

Also note lwd_cluster scene in the new code vs design doc mention R0 only edge_share/cloud_share. Let me read doc around 380-440 and top sections.

这三个名字都不是新实现，而是把 `vllm/config/lwd/topology.py` 里新增的"运行时拓扑更新契约"重新导出到 `vllm.config.lwd` 公共入口。对比旧版 [__init__.py](file:///C:/github_main_host_rdma_old/vllm_host_rdma/vllm/config/lwd/__init__.py#L11-L21)，新版只多了这三个符号的 import 和 `__all__` 条目（[__init__.py#L11-L30](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/__init__.py#L11-L30)），实现全部在 [topology.py#L725-L946](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L725-L946)。

## 解决什么问题

host-RoCE 的 `edge_share` 场景下拓扑 yaml 文件同时充当"共享存储"：运维往文件里追加新的 edge/cloud 实例对，运行中的共享 edge 引擎要热加载这份增长后的拓扑，而不是重启。`diff_topology` 就是这个热加载的**校验器**，两个 dataclass 是它的**结论类型**。

## 三者各自的职责

**`LwdTopologyIncrement`**（[topology.py#L730-L743](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L730-L743)）— 校验通过的"纯增量"描述，冻结 dataclass，四个字段：
- `new_edge_ids` / `new_cloud_ids`：本次新增的实例 id
- `new_connections`：Phase B/C 需要新建的全部 `LwdDPConnection`（控制面 DEALER + 数据面 UCX 链接都按它建）
- `new_topology`：增量全部生效后应成为的"已实现快照"

`new_connections` 为空表示文件 mtime 变了但语义没变（no-op），引擎直接忽略。

**`LwdTopologyUpdateRejected`**（[topology.py#L746-L754](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L746-L754)）— 违反增量契约时的结论，只有一个 `reason` 字符串，指出违反了哪条规则、哪个字段。它是确定性的：同一份文件重复读会得到同样的拒绝原因。关键语义是**拒绝不影响服务**——引擎继续用已实现的快照跑，不会因为一次坏编辑而崩溃。

**`diff_topology(old, new)`**（[topology.py#L807-L946](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L807-L946)）— 纯函数，比较两份已通过 `from_dict` 布局校验的拓扑，按顺序判定 R0~R4：

| 规则 | 内容 |
| --- | --- |
| 快速路径 | 两份 digest 相同 → 返回空增量 |
| R0 | scene 不能变，且只有有热更新语义的 scene 才允许：`edge_share`（共享 edge 按实例对增长）、`cloud_share`（共享 cloud 按 edge 增长）、`lwd_cluster`；`single_instance` 必须显式比较后拒绝，提示重启 |
| R1 | 全局块冻结：`deployment`（去掉随实例增长的 `edges_num`/`clouds_num`）、`feature_ctrl`、`control_plane` 都不能改——它们决定了已实现的缓冲池大小、channel scramble、控制面配置，无法追溯生效；`qos` 靠部署纪律豁免而非代码检查 |
| R2 | 已有实例必须逐字节不变（含 dp 列表和 coordination 块），因此天然禁止缩容。新 schema 下不再需要旧的 `control_url_map` 豁免：新 cloud 的协调 URL 放在它自己的 `LwdDP.coord_url` 里，加 cloud 不需要动任何已有实例（[`_relaxed_edge_snapshot`](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L793-L804) 就是记录这个决策的空操作） |
| R3 | link 只增不减；已有 link 被替换也算修改 |
| R4 | 增量形状必须符合 scene：`edge_share` 只能整对增加，且新 link 不能跨新旧实例；`cloud_share` 只加 edge，且每条新 link 必须连到已有 cloud；`lwd_cluster` 由全互联的 scene 校验兜底 |

通过后在末尾把新增 link 展开成具体 DP 连接，按 link 的 cloud 过滤（`lwd_cluster` 下老 edge 连新 cloud 时，一个 edge 的全部连接里只有连该 cloud 的那条是新的，见 [L936-L945](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L934-L945)），产出 `LwdTopologyIncrement`。

## 调用链

1. [lwd_base_engine.py#L200-L214](file:///c:/mainmain/vllm_host_rdma/vllm/v1/lwd_control/control_scheduler/lwd_base_engine.py#L200-L214)：守护线程按 1s 轮询 mtime，变化则往引擎 input queue 塞一个 reload tick（只在 host comm + 共享侧启用）。
2. [lwd_base_engine.py#L216-L271](file:///c:/mainmain/vllm_host_rdma/vllm/v1/lwd_control/control_scheduler/lwd_base_engine.py#L216-L271)：引擎线程重新解析文件 → digest 快路径去重 → `diff_topology` 得到 verdict；拒绝只打 error 日志并保留已实现快照。
3. 命中增量时调角色钩子：edge 引擎建控制面 + 数据面链接（[lwd_edge_engine.py#L230-L290](file:///c:/mainmain/vllm_host_rdma/vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_engine.py#L230-L290)），cloud 引擎扩充 handshake allowlist 和 UCX listener 路由（[lwd_cloud_engine.py#L616-L661](file:///c:/mainmain/vllm_host_rdma/vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_engine.py#L616-L661)）。基类默认实现只记录"该角色暂无运行时应用"。

## 为什么放在 `__init__.py`

`vllm.config.lwd` 是设备无关的配置层公共 API；引擎、测试只需 `from vllm.config.lwd import LwdTopologyIncrement, LwdTopologyUpdateRejected, diff_topology`，不必知道它们位于 `topology.py`。同时保证契约本身是纯配置代码——不引入任何执行钩子或设备依赖，符合设计文档"the contract itself is pure config-layer code"的定位。

完整设计见 [lwd_topology_update.md](file:///c:/mainmain/vllm_host_rdma/docs/design/lwd_topology_update.md)（Phase A 检测、Phase B 控制面加链、Phase C 数据面装配）。

---

**User**

对比C:\github_main_host_rdma_old\vllm_host_rdma\vllm\config\lwd\topology.py，topology.py新增_canonical_json干说明

**Assistant**

Let me search for `_canonical_json` in both files to understand what was added.

Let me read the relevant sections of both files to compare. The old file uses `json.dumps` directly inline; the new file introduced `_canonical_json` helper and adds a `frontend_digest`. Let me read both in detail.

Now I understand. Let me also check where `frontend_digest` is consumed to explain why two digests exist and why `_canonical_json` was extracted (it's now used twice — for digest and frontend_digest).

Let me grep frontend_digest usage.

Now I have the full picture. Let me write the answer.

Key points:
1. `_canonical_json` is a refactored helper that was previously inlined in `from_dict`.
2. It does deterministic JSON encoding: sort_keys=True, compact separators, ensure_ascii=False, allow_nan=False, then UTF-8 encode.
3. The reason it was extracted: new code computes TWO digests — `digest` (full topology, used by `diff_topology` fast path) and `frontend_digest` (excludes instance inventory edges/clouds/instance_links, used by attach frontend config matching). Both need the same canonical encoding, so the logic was pulled into a shared helper to avoid duplication and drift.
4. Why `frontend_digest` exists: attach frontends attach to a running primary; the inventory (edges/clouds) grows elastically, so the full `digest` changes whenever instances are added. The attach handshake must match on config that is stable across elastic additions — so a separate digest that excludes the instance inventory but covers the frozen blocks (deployment minus count declarations, feature_ctrl, control_plane, qos) is needed.
5. ensure_ascii=False + allow_nan=False: stability and safety.

Let me verify the old file's imports - it has json imported already. The refactor is clean.

Let me confirm the design doc explanation for frontend_digest split. Read lwd_topology_update.md around 185-215.

## `_canonical_json` 是干什么的

`_canonical_json`（[topology.py#L35-L45](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L35-L45)）是一个**确定性 JSON 编码**的小工具函数，把一份语义快照编码成字节串供 SHA-256 取摘要：

```python
def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,        # 忽略 key 书写顺序
        separators=(",", ":"), # 最紧凑，忽略空白
        ensure_ascii=False,    # 非 ASCII 字符保持稳定
        allow_nan=False,       # 禁止 NaN/Infinity（JSON 非标准、跨平台不稳定）
    ).encode("utf-8")
```

它保证：**只要拓扑语义相同，编码出的字节就相同**，与 YAML 的缩进、key 顺序、注释、空白无关——这是 digest 能稳定比较的前提。

## 相比旧版为什么要新增这个函数

旧版 [topology.py#L339-L346](file:///C:/github_main_host_rdma_old/vllm_host_rdma/vllm/config/lwd/topology.py#L339-L346) 只算一个 `digest`，编码逻辑是**内联**在 `from_dict` 里的：

```python
encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
```

新版 [topology.py#L385-L412](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L385-L412) 要算**两个** digest：

1. **`digest`** — 完整拓扑摘要（排除自身派生字段），给文件 watcher 去重和 `diff_topology` 的 no-op 快速路径用，必须感知**每一次**内容变化。
2. **`frontend_digest`** — 是新增的"收窄"摘要：排除实例清单（`edges`、`clouds`、`instance_links`）以及随实例增长的 `edges_num`/`clouds_num`，只覆盖 R1 冻结块（`deployment` 去掉计数声明、`feature_ctrl`、`control_plane`）、`qos` 和 scene。

两个摘要用的是**完全相同**的编码规则，所以把那段 `json.dumps(...)` 抽成 `_canonical_json` 共享，避免两处复制粘贴导致规则漂移。

## 为什么需要 `frontend_digest`

这是设计文档 [lwd_topology_update.md#L186-L216](file:///c:/mainmain/vllm_host_rdma/docs/design/lwd_topology_update.md#L186-L216) 的 Item 11（前端对账维度拆分）：

- attach 前端和运行中的 primary 做配置对账时，原来比对的是**完整** `digest`；
- 但弹性扩容时，primary 用小 yaml 启动，前端用扩容后的大 yaml 启动——实例清单变了，完整 `digest` 必然不同，前端 attach 会被 `value mismatch: lwd_topology_digest` 拒绝，整个弹性扩缩链条在前端这一环死锁；
- 扩缩只改实例清单，所以把对账摘要收窄为 `frontend_digest`：清单增长时它保持不变，前端能正常 attach；而清单之外的改动（deployment 块、qos、scene）仍会让 `frontend_digest` 变化，从而正确拒绝 attach。

对应的 payload key 也从 `lwd_topology_digest` 改成了 `lwd_frontend_digest`（[frontend_attach.py#L230](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L230)），primary 启动时同时打两个摘要的日志（[frontend_attach.py#L818-L822](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L818-L822)）。

一句话：`_canonical_json` 是为配合 `frontend_digest` 的引入而抽出的共享编码原语——**同一个确定性编码规则被两处 digest 复用**，本身不是新语义，只是把旧的内联编码显式化、可复用。

---

**User**

api_bind_base_port对 data_plane_comm= device（边云是hccl传输hidden）生效吗

**Assistant**

The user asks whether `api_bind_base_port` takes effect for `data_plane_comm=device` (edge-cloud uses HCCL to transmit hidden). Let me investigate.

From what I've seen, `api_bind_base_port` is in `deployment` config and relates to the engine IO port band reserved for edge_share attach frontends. Let me grep where it's read and whether it's gated on host comm / scene.

Let me search.

Key validation at topology.py #L169-175 says api_bind_base_port is an edge_share field — rejected outside edge_share. But the question is whether it "takes effect" for data_plane_comm=device. Let me read the validation and check how api_bind_base_port interacts with host comm. The frontend attach and engine IO band. Let me read topology deploy section around 125-200.

The user asks: does `api_bind_base_port` take effect for `data_plane_comm=device` (edge-cloud uses HCCL to transmit hidden)?

Key facts:
1. `api_bind_base_port` is an **edge_share** setting. The validation only checks `scene != edge_share` → reject if non-default. It does NOT check `data_plane_comm`. So config-level: for edge_share scene, whether device or host comm, you can configure it.

2. But does it "take effect" (bind the ports)? `api_bind_base_port` is used by the frontend attach / engine IO band mechanism — that mechanism requires attach mode (the `--api-server-rpc-port`, registration server). Is attach mode enabled for device comm?

Recall from earlier: in `frontend_attach.py`, `lwd_dispatch_attach_frontend` sets `args.api_server_count = LWD_API_MAX_FRONTENDS if plan.host_comm else plan.client_count`, and `args.lwd_attach_primary = plan.host_comm`. And `resolve_attach_plan` is gated on `LwdAPIConfig.from_namespace(args).requested`. Let me check `LwdAPIConfig` — does it gate attach on host_comm?

Also AGENTS.md said: "host 数据面 (deployment.data_plane_comm: host): ... 配置块 deployment.data_plane_host". And "数据面与两段 PP: 边云共享一个 HCCL world (仅 data_plane_comm: device), rank 即拓扑里的全局 rank".

The question is whether attach/engine IO port band mechanism (which uses api_bind_base_port) is host-comm only. Let me check LwdAPIConfig and resolve_attach_plan, and the engine wiring for device comm.

From earlier reads: `resolve_attach_plan` returns None if `not api.requested`. Let me check what makes api.requested true — likely `--api-server-rpc-port` or similar. And the comment in FrontendAttachPlan.host_comm: "Whether the primary must run the attach pipeline ... device-comm primaries keep the native single-server path." That strongly suggests the attach pipeline (and thus api_bind_base_port port band) is only used for host comm. For device comm, even if you set api_bind_base_port, the attach pipeline isn't run.

Let me verify by reading LwdAPIConfig and resolve_attach_plan and where engine binds the IO band. Also check the ascend side / data plane. Let me read frontend_attach around 171-210 and api.py.

The Glob returned nothing because maybe the api.py is in a different path. Let me find LwdAPIConfig. Actually I recall import `from .api import LwdAPIConfig` in __init__. Let me glob.

So api.py exists but Glob pattern with ** returned a different root. Let me read api.py around requested / validate_topology. Actually I have enough to answer. Let me confirm whether attach/engine IO band is gated to host comm in the engine side. The key question: does api_bind_base_port "take effect" for device comm?

From the code:
- `api_bind_base_port` config is allowed regardless of data_plane_comm (validation only checks scene==edge_share).
- It's consumed by `resolve_attach_plan` → `FrontendAttachPlan` → `FrontendAttachServer` / `engine_io_addresses` derivation → port band reservation and binding. This whole attach/engine-IO-band mechanism runs when `LwdAPIConfig.requested` is true (i.e., `--api-server-rpc-port` or `--api-server-attach` set).

But does device comm use attach? The comment in host_comm field and `lwd_attach_primary = plan.host_comm` strongly implies the attach pipeline is host-comm-only. But is it forbidden for device comm, or just not routed through multi-server?

Let me check `LwdAPIConfig.validate_topology` and whether it restricts attach to host comm. Also check the engine side: where engine IO addresses use api_bind_base_port — `engine_io_addresses`. Does that run for device comm? It's in `wait_for_attach_registrations` which only runs in `run_multi_api_server` when attach_mode is True (prepare_attach_mode returns True if api_server_rpc_port set). So if a device-comm primary sets --api-server-rpc-port, prepare_attach_mode would build a registration server and bind the band... but wait, there's a check `lwd_attach_primary` only routes host comm to multi_server. For device comm, count = plan.client_count (= len(edges)), could be 1 → single in-process path → no registration server → band not bound. If count > 1, goes to run_multi_api_server regardless? Let me re-check serve.py routing:

```
elif (args.api_server_count > 1 or rust_frontend_path or getattr(args, "lwd_attach_primary", False)):
    run_multi_api_server(args)
```

For device comm primary with multiple edges, api_server_count = count > 1 → run_multi_api_server. There, prepare_attach_mode checks `getattr(args, "api_server_rpc_port", None) is None` → if set, attach_mode True → builds registration server with api_bind_base_port. So technically device comm *could* use it if configured. But is that a supported/intended path? The design says attach is host-comm. Let me check LwdAPIConfig.validate_topology — maybe it rejects attach for device comm.

Let me read api.py.

`validate_topology` does NOT restrict attach to host comm. So technically attach could be configured for device comm edge_share. But the question is whether `api_bind_base_port` takes effect for device comm.

Let me check the engine side: where the engine actually binds the engine IO band derived from api_bind_base_port. That's `engine_io_addresses` and the pre-seeding in `wait_for_attach_registrations`, only in run_multi_api_server attach_mode path. For device comm primary, `lwd_attach_primary = False`, so routing depends on api_server_count > 1.

But more importantly: does device comm even support the attach frontend model? The AGENTS.md says attach is part of host data plane design (`edge_share` with `data_plane_comm: host`). Let me check the design docs and whether there's an explicit gate. Actually the user's question is specific: does api_bind_base_port take effect for data_plane_comm=device where edge-cloud transmits hidden via HCCL.

`api_bind_base_port` is the base of the reserved engine IO port band for **attach frontends** (API frontends connecting to the shared edge engine). This is about the **API frontend ↔ engine** data path (the engine's input/output ZMQ addresses bound deterministically so a restarted frontend rebinds the same port). This is independent of whether edge-cloud hidden transmission uses HCCL (device) or host UCX.

Wait — is it? The "engine IO band" `[base, base+15]` is for attach frontends to bind their engine-facing ZMQ sockets (input ROUTER / output PULL) so the engine's DEALER/PUSH can reconnect. The data_plane_comm (device vs host) is about edge↔cloud hidden/KV transmission. These are different planes!

So `api_bind_base_port` relates to the **API frontend attach plane**, not the edge-cloud data plane. The question conflates them. Let me verify: the reserved band ports are `base + 2*idx` (input) and `+1` (output) — these are the ZMQ sockets between attach frontend and engine (engine IO). So `api_bind_base_port` takes effect when attach frontends are used (edge_share + --api-server-rpc-port/--api-server-attach), regardless of data_plane_comm.

BUT: from the code, `lwd_attach_primary = plan.host_comm` and the comment says "device-comm primaries keep the native single-server path". This means for device comm, the primary does NOT run the attach pipeline (registration server) even at count 1. However, does device comm support attach frontends at all? Let me check if there's a gate preventing attach for device comm. `validate_topology` does not gate on host_comm. Hmm.

Let me look more carefully. Actually `resolve_attach_plan` is called for both. For device comm, `host_comm=False`, so `capacity_value` False, count = len(edges), and if user sets --api-server-rpc-port on edge 0 → mode primary. Then `lwd_dispatch_attach_frontend` sets api_server_count = plan.client_count (not LWD_API_MAX_FRONTENDS) and lwd_attach_primary=False. If count > 1, routing → run_multi_api_server → prepare_attach_mode → if api_server_rpc_port set → builds registration server using api_bind_base_port. So it WOULD take effect.

But is device comm + attach a supported combo? Let me search design docs / AGENTS for whether attach requires host comm. The AGENTS.md "host 数据面" section describes the attach/registration server within host data plane context. Let me grep in docs and code for any gate like "attach requires host comm" or is_host_comm checks near attach.

Let me grep "is_host_comm" in frontend_attach and engine and validation.

Let me check lwd_async_client.py:348 — that's likely where attach client is gated on host_comm. And lwd_base_engine 156/174/176 for topology watch. Let me read lwd_async_client around 340-360, and base engine 150-180.

## 结论

对 `data_plane_comm: device`（边云用 HCCL 传 hidden）来说，**`api_bind_base_port` 配置不报错，但实际不生效——端口带不会被绑定、也不会被使用**。

## 为什么

先把两个平面分清楚，`api_bind_base_port` 和 `data_plane_comm` 管的不是同一件事：

- `data_plane_comm`（`device` / `host`）决定的是**边↔云之间** hidden/KV 走哪条链路：`device` = 共享 HCCL world；`host` = 各侧独立 world、走 host UCX（UHC1）。
- `api_bind_base_port` 是 `edge_share` 场景下**预留的引擎 IO 端口带基址**（默认 61000，`[base, base+15]`，每个 attach 前端占 2 个端口：`base+2*idx` 输入 ROUTER、`+1` 输出 PULL），服务的是**API 前端 ↔ 引擎**这条 attach 平面——让前端重启后能 rebind 到同一个确定性端口，运行中引擎的 DEALER/PUSH 透明重连，以及弹性扩缩时前端用扩容后的 yaml 仍能 attach。见 [topology.py#L155-L166](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L155-L166) 和 [frontend_attach.py#L300-L340](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L300-L340)。

也就是说，`api_bind_base_port` 是否生效，取决于 **attach 前端流水线**是否运行，而不是边云数据面用 HCCL 还是 host UCX。但在本仓里，整条 attach/自愈/弹性流水线被明确收敛到 host comm：

1. **配置层不拦**：[`_validate_api_bind_base_port`](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L168-L179) 只校验 `scene == edge_share`，不看 `data_plane_comm`。所以 device comm 的 edge_share yaml 里填 `api_bind_base_port` 不会报配置错误，值会被解析进 `deployment.api_bind_base_port`。

2. **真正消费端口带的代码只在 host comm 路径跑**：
   - [`lwd_dispatch_attach_frontend`](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L966-L987) 里 `args.lwd_attach_primary = plan.host_comm`。device comm 主节点该标志为 False，单前端时走原生单进程路径，不进 `run_multi_api_server`，也就不建 registration server——而注册服务器（[`prepare_attach_mode`](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L990-L997) → [`create_frontend_registration_server`](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L752-L780)）才是绑定/推导这个端口带、做 pre-seed 和冲突检查的地方。
   - attach 前端侧的 LWD 自愈客户端（READY 去重、ROUTER 交接、引擎复活）在 [`lwd_async_client.py#L341-L351`](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/lwd_async_client.py#L341-L351) 被显式 gate：`device-comm (HCCL) deployments keep the native AsyncMPClient untouched`，只有 host comm 才替换成 `LwdAsyncMPClient`。
   - 弹性扩缩的拓扑 watch 在 [`_lwd_topology_watch_enabled`](file:///c:/mainmain/vllm_host_rdma/vllm/v1/lwd_control/control_scheduler/lwd_base_engine.py#L163-L176) 也 gated 在 `is_host_comm`，device comm 永不 watch。

所以 device comm 下，即使 `--api-server-rpc-port` / `--api-server-attach` 被配上，依赖这个端口带的注册、预播种、自愈重连都不会激活，`api_bind_base_port` 只是拓扑里一个被解析但没人用的字段。这也符合 AGENTS.md 的定位：attach / 前端自愈 / host 数据面弹性扩缩是同一套 host-comm 机制，device comm（HCCL）路径保持不动。

---

**User**

对比C:\github_main_host_rdma_old\vllm_host_rdma\vllm\config\lwd\topology.py，讲解一下from_dict中的代码修改

**Assistant**

The user asks: compare old topology.py, explain the code modifications in from_dict. I already have both versions partly read. Let me read both from_dict fully and surrounding fields (digest, frontend_digest fields, imports at top). Let me read new file top imports and full class around 340-420, old file 280-350. Also check imports for json, sha256 in both.

Let me read the new file 1-60 and 340-420, old 1-60 and 290-350.

## `from_dict` 的差异总览

对比旧版 [topology.py#L317-L347](file:///C:/github_main_host_rdma_old/vllm_host_rdma/vllm/config/lwd/topology.py#L317-L347) 和新版 [topology.py#L372-L413](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L372-L413)，函数骨架没变（validate → 补全 instance_links → `validate_layout()` → 算 digest），改动集中在一件事：**从单一 `digest` 拆出第二个 `frontend_digest`**。具体分四处。

### 1. 数据类新增派生字段（函数外，但直接影响 from_dict）

```python
digest: str = Field(default="", init=False)
frontend_digest: str = Field(default="", init=False)   # 新增
```

见 [topology.py#L354-L361](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L354-L361)。`init=False` 表示不接受用户输入，只能在 `from_dict` 里用 `object.__setattr__` 回填。它的 docstring 就是设计意图：**对"除实例清单外的一切"取摘要**，弹性扩容只动清单，所以这个摘要跨增量保持稳定。

### 2. canonical 快照排除两个派生字段

旧版只排除自己：

```python
canonical = asdict(topology)
canonical.pop("digest", None)
```

新版多一行（[L388-L390](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L388-L390)）：

```python
canonical.pop("digest", None)
canonical.pop("frontend_digest", None)
```

因为 `asdict(topology)` 会把 `frontend_digest` 的默认空串一起拷进去。虽然空串的哈希是常量、不会造成跨次不稳定，但它属于派生值，参与自身计算会让"摘要里包含摘要"的语义混乱，所以两个派生字段一并剔除——注释也改成了复数 "the derived digest fields themselves"。

### 3. 内联编码抽成 `_canonical_json`

旧版在函数体里直接写 `json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")`。新版改为 `sha256(_canonical_json(canonical))`（[L395-L397](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L395-L397)），复用 [L35-L45](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L35-L45) 的函数。原因很简单：现在要编码两次（全量 + 前端子集），不想把四条确定性编码参数复制两遍导致规则漂移。

### 4. 新增 frontend 子集与第二次哈希（核心改动）

```python
frontend = {
    key: value
    for key, value in canonical.items()
    if key not in ("edges", "clouds", "instance_links")
}
frontend["deployment"] = {
    key: value
    for key, value in canonical["deployment"].items()
    if key not in ("edges_num", "clouds_num")
}
object.__setattr__(
    topology, "frontend_digest", sha256(_canonical_json(frontend)).hexdigest()[:16]
)
```

见 [L398-L412](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L398-L412)。它做两件事：

- **剔除实例清单**：`edges`、`clouds`、`instance_links` 三个顶层键。
- **剔除计数声明**：`deployment` 里的 `edges_num` / `clouds_num`——它们随清单一起增长，留着就等于没剔除。

剩下的内容恰好是 `deployment`（去掉计数）、`feature_ctrl`、`qos`、`control_plane`，也就是 `diff_topology` 里 R1 冻结块 + scene + qos 的集合。这里有个实现细节：`asdict` 返回的是深拷贝的新 dict，所以直接给 `frontend["deployment"]` 重新赋值是安全的，不会污染后面或前面的 `canonical`。

## 为什么要这么改

对应设计文档 [lwd_topology_update.md#L186-L216](file:///c:/mainmain/vllm_host_rdma/docs/design/lwd_topology_update.md#L186-L216) 的 Item 11（前端对账维度拆分）：

| 摘要 | 覆盖范围 | 消费者 | 变化时机 |
| --- | --- | --- | --- |
| `digest` | 全部语义 | 文件 watcher 去重、`diff_topology` 的 no-op 快速路径 | 任何内容变化都必须感知 |
| `frontend_digest` | 除实例清单外的全部（R1 冻结块 + scene + qos） | attach 前端与 primary 的配置对账（payload key 为 `lwd_frontend_digest`，见 [frontend_attach.py#L230](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L230)） | 只有清单增长时保持不变 |

改之前前端对账用完整 `digest`：primary 用小 yaml 启动、前端用扩容后的大 yaml 启动，清单不同 → digest 必然不同 → 前端 attach 被 `value mismatch` 拒绝，弹性扩缩链条在前端这环死锁。拆分后，清单增长不影响 `frontend_digest`，前端能正常 attach；而 deployment 块、qos、scene 这类"扩缩不该动"的字段若被改，`frontend_digest` 仍会变化，从而**对称地**拒绝 attach——这正是运行中引擎通过 R1 拒绝非法增量的同一组不变量。

注意 watcher 去重和 `diff_topology` 的 no-op 判断仍然用完整 `digest`（[topology.py#L842](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L842)），因为它们必须发现每一次内容变化，只有前端对账这一维收窄了。

---

**User**

讲解_validate_scene修改

**Assistant**

Need to compare `_validate_scene` in both files. Let me grep and read.

Wait — the old file's _validate_scene at line 566, but the new file has _validate_scene at 632. The new one I read 455-585 shows _validate_host_layout, _validate_coordination, _validate_ranks. I need to read new _validate_scene at 632 onward. Let me read new 585-710.

Also old line numbers: old _validate_scene at 566 shown. Hmm, but the old read at 555-640 shows _validate_scene content. New at 632. Let me read new 590-700.

Compare old vs new _validate_scene:

Old (566-608):
- single_instance: (1,1)
- cloud_share: edge_count < 2 or cloud_count != 1 → requires multiple edges and one cloud
- lwd_cluster: cloud_count < 2 → at least two clouds
- coordination requirement for cloud_share/lwd_cluster
- edge_share: edge_count != cloud_count or cloud_count < 2 → equal multiple
- else full mesh

New (632-691):
1. cloud_share: edge_count < 1 or cloud_count != 1 → at least one edge and one cloud (allows single-edge degenerate)
2. lwd_cluster minimum count check removed entirely (only type-level >=1); comments explain 1+1 degenerate form
3. edge_share: dropped `cloud_count < 2`, only `edge_count != cloud_count`; comment explains single pair degenerate
4. Comments expanded explaining rationale.
5. single_instance and coordination blocks unchanged.

So the semantic changes: allow degenerate single-pair edge_share, allow single-edge cloud_share, allow single-cloud (and single-edge) lwd_cluster. Essentially relax count minimums while keeping structural invariants (equal counts, full mesh, coordination enabled).

Need to check whether these degenerate forms are supported by other code: e.g., lwd_cluster full-mesh requirement for 1 edge + 1 cloud is trivially satisfied. cloud_share with 1 edge requires coordination enabled blocks still. edge_share 1 pair ties to attach frontend machinery (capacity 8, client_count etc.).

Let me check tests for these changes to cite. Grep test_topology for degenerate / single pair.

Now I have everything. Write explanation of _validate_scene modifications. Also check whether lwd_cluster minimum cloud count 2 was removed and maybe tests. Let me grep test for lwd_cluster requirements in new tests quickly.

## `_validate_scene` 改了什么

对比旧版 [topology.py#L566-L608](file:///C:/github_main_host_rdma_old/vllm_host_rdma/vllm/config/lwd/topology.py#L566-L608) 和新版 [topology.py#L632-L691](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L632-L691)，改动只有一类：**把三个 scene 的最小实例数门槛降到"退化形态"，同时保留全部结构性不变量**。

### 1. `cloud_share`：从"至少 2 个 edge"放宽到"至少 1 个"

```python
# 旧
if scene == "cloud_share" and (edge_count < 2 or cloud_count != 1):
    raise ValueError("[LWD] cloud_share requires multiple edges and one cloud")
# 新
if scene == "cloud_share" and (edge_count < 1 or cloud_count != 1):
    raise ValueError("[LWD] cloud_share requires at least one edge and one cloud")
```

注释解释了理由（[L638-L643](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L638-L643)）：单 edge 的 `cloud_share` 是共享云流水线的退化形态，它跑的是**同一套** coordination 机制和同一套多 edge 请求路由，所以允许用 1 个实例把整条流水线验证通，再通过加 edge 逐步扩缩。这对应测试里的回归用例 [test_topology.py#L100-L123](file:///c:/mainmain/vllm_host_rdma/tests/config/lwd/test_topology.py#L100-L123)，docstring 明确写了"validator used to demand >= 2 edges and the whole deployment was rejected at config parse time"。

### 2. `lwd_cluster`：删除"至少 2 个 cloud"的检查

```python
# 旧
if scene == "lwd_cluster" and cloud_count < 2:
    raise ValueError("[LWD] lwd_cluster requires at least two clouds")
# 新：整段删除，只留注释
```

见 [L648-L651](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L648-L651)。注释说明：除了类型层的 `edges_num/clouds_num >= 1`，`lwd_cluster` 不再有最小数量检查；1 edge + 1 cloud 是全互联流水线的退化形态，可用于验证和渐进扩容。结构性约束没丢——下面的 full-mesh 检查仍然要求 `instance_links` 覆盖所有 edge×cloud 组合，1×1 时自然满足。

### 3. `edge_share`：去掉"至少 2 对"，只保留等数约束

```python
# 旧
if edge_count != cloud_count or cloud_count < 2:
    raise ValueError("[LWD] edge_share requires equal multiple edge/cloud IDs")
# 新
if edge_count != cloud_count:
    raise ValueError("[LWD] edge_share requires equal edge/cloud instance IDs")
```

见 [L667-L678](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L667-L678)。注释是这段改动里信息量最大的：单对是共享 edge 流水线的退化形态，它和 N 对形态跑**完全一样**的 attach 前端机制和基于 `client_index` 的请求路由，所以先用一个实例验证整条流水线、再扩 plan。真正的不变量是"edge 数 = cloud 数"，数量本身不是，且所有消费方都是 count-generic 的。

这正好接上前面讨论过的这条链：单对 `edge_share` + host comm → `lwd_dispatch_attach_frontend` 把 `api_server_count` 设成预留容量 8、`lwd_attach_primary=True` → 走 `run_multi_api_server` 建注册服务器。

### 4. 没改的部分

- `single_instance` 仍要求恰好 `(1, 1)`；
- `cloud_share` / `lwd_cluster` 仍要求每个实例都有 `coordination.enabled` 的配置块（`edge_share`、`single_instance` 不强制）；
- `edge_share` 的 one-to-one link 检查原样保留：link 数 == edge 数，且每个 edge/cloud 恰好出现一次；
- 非 `edge_share` 的 full-mesh 检查原样保留。

## 为什么这样改

改动的共同逻辑是：**把"多实例"从场景的合法性条件降级为流水线的可选规模**。

1. **可验证性**：单机 1 edge + 1 cloud 就能把 `cloud_share`/`lwd_cluster`/`edge_share` 的完整流水线跑通，不必先凑齐多实例环境。测试用例就是这么用的（[test_topology.py#L83-L89](file:///c:/mainmain/vllm_host_rdma/tests/config/lwd/test_topology.py#L83-L89) 的单对 edge_share）。
2. **与弹性扩容配套**：运行中拓扑更新契约 `diff_topology` 只接受"纯追加"，且已有实例必须逐字节不变。从 1 对起步、再追加实例对，是最自然的演进路径；如果基线必须 ≥2，那"从 1 扩到 2"这条最简单的路径就永远走不到。
3. **不牺牲正确性**：放宽的只是数量下限，形状约束（等数、一对一、全互联、coordination 完整、DP 数一致、端口不冲突）一条都没动；0 个实例仍会被类型层 `LwdCount ge=1` 和 tuple `min_length=1` 挡掉。

一句话总结：`_validate_scene` 的修改是"允许 1 实例的退化形态"，用最少的拓扑验证完整流水线并作为弹性扩缩的起点，而不是放宽场景的结构语义。

---

**User**

讲解`c:\mainmain\vllm_host_rdma\vllm\config\lwd\topology.py#L729-947`这些新增的方法

**Assistant**

## 这段代码的整体定位

[topology.py#L725-L947](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L725-L947) 是**运行中拓扑热更新的契约层**，全部是纯配置代码：不建链、不读环境、不碰设备。它回答一个问题——"运维把 yaml 改成新版本后，这个正在跑的引擎能不能安全地把它当作纯追加来应用？如果能，要新建哪些连接？"

两个 dataclass 是结论（要么增量、要么拒绝），三个下划线函数是辅助，`diff_topology` 是主判定流程。调用方在 [lwd_base_engine.py#L216-L271](file:///c:/mainmain/vllm_host_rdma/vllm/v1/lwd_control/control_scheduler/lwd_base_engine.py#L216-L271)（检测）和 edge/cloud 引擎的 `_lwd_apply_topology_increment`（应用）。

## 1. `LwdTopologyIncrement`：校验通过的增量

```python
new_edge_ids: tuple[int, ...]
new_cloud_ids: tuple[int, ...]
new_connections: tuple[LwdDPConnection, ...]
new_topology: LwdTopology
```

见 [L730-L743](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L730-L743)。四个字段的语义是递进的：

- `new_edge_ids` / `new_cloud_ids`：本次新出现的实例 id。由于布局校验强制 id 从 0 连续且唯一（[order_instances](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L420-L428)），合法增量只可能是"在尾部追加"，所以这两个元组必然是后缀。
- `new_connections`：Phase B/C 要建的具体 DP 连接（控制面 DEALER + 数据面 UCX 都按它建），已展开到 `(edge_id, cloud_id, dp_idx)` 粒度。
- `new_topology`：增量**全部应用成功后**才应成为的"已实现快照"。这个推进动作由调用方负责，`diff_topology` 本身不改任何运行时状态。
- **空 `new_connections` = no-op**：文件被 touch 了但语义没变，或只改了引擎无法热应用的字段（例如 `qos`，见后文 R1）。

调用方据此分流：[base engine](file:///c:/mainmain/vllm_host_rdma/vllm/v1/lwd_control/control_scheduler/lwd_base_engine.py#L251-L260) 对空增量只打日志并返回 `None`，不进应用钩子。

## 2. `LwdTopologyUpdateRejected`：违反契约

```python
reason: str
```

见 [L746-L754](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L746-L754)。两个关键设计：

- **拒绝是正常返回值，不是异常**。配置解析失败才抛 `ValueError`；"文件合法但语义上不能热更新"是一种预期内的控制流，用返回值表达让 watcher 可以"记日志 + 继续服务"而不需要 try/except 包住业务逻辑。
- **确定性**：同一份文件重复 diff 必然得到同一个 `reason`，因为判定顺序和 `_first_change` 的遍历顺序都是固定的。`reason` 里带具体字段路径（例如 `global block changed: deployment.api_bind_base_port: 61000 -> 62000`，设计文档 [lwd_topology_update.md#L450](file:///c:/mainmain/vllm_host_rdma/docs/design/lwd_topology_update.md#L450) 就是这个输出）。

## 3. `_first_change`：给出"哪一行不对"

见 [L757-L775](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L757-L775)。对两棵 `asdict` 树做递归深比较，返回**第一个**差异，格式统一为 `path: old -> new`：

- dict：对 `sorted(set(old) | set(new))` 遍历，能同时报告"新增键""缺失键"和"值不同"；排序保证确定性。
- list/tuple：长度不同直接报 `N -> M entries`；否则按下标递归，路径带 `[i]`。
- 标量：直接格式化。

为什么不直接 `old != new`？因为拒绝信息必须能让运维定位到 yaml 里的具体字段。R1（全局块冻结）和 R2（已有实例不可改）都靠它产出可读原因。

## 4. `_frozen_blocks`：R1 的冻结集合

见 [L778-L790](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L778-L790)。抽出三个不允许在运行中变化的块：

- `deployment`（**去掉** `edges_num` / `clouds_num`）——这两个计数本来就该随新增实例增长，留着会把每次合法扩容都误判为违规；剩下的 `scene`、`data_plane_comm`、`data_plane_host`、`hccl_world_size`、`api_bind_base_port` 决定已实现资源的大小（例如 `up_buffer_size` 的推导）和端口带，无法追溯生效。
- `feature_ctrl`：channel scramble 在链路建立时就播种了。
- `control_plane`：控制面拓扑已经建好。
- `qos` **故意不在里面**——docstring 写明是"部署纪律保证不改，而不是代码检查"。注意后果：只改 `qos` 时 `digest` 会变、R1 不拦、最终得到一个空增量，引擎不会应用它，但也不会拒绝整次更新。

## 5. `_relaxed_edge_snapshot`：一个"记录决策的空操作"

见 [L793-L804](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L793-L804)。函数体就是 `return snapshot`，唯一的用途是**用代码位置记录为什么 R2 不需要豁免**：

- 旧分支里，新增 cloud 需要在已有 edge 的 `coordination.control_url_map` 里补一条 URL，所以旧实现必须"放宽"偏移后的 edge 快照，否则 R2 的逐字节比较必然失败。
- 新 schema 把协调 URL 挪到了 cloud 自己的 `LwdDP.coord_url`，并且 `LwdCoordination.reject_legacy_fields` 直接拒绝 `control_url_map`。所以加 cloud 完全不需要动任何已有实例，R2 保持严格相等即可。

参数 `added_clouds` 目前不参与逻辑，保留它是为了把"这里本来该有豁免、但新 schema 下不需要"这个判断显式化，避免后来者以为漏写了。

## 6. `diff_topology`：判定流水线

见 [L807-L946](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L807-L946)。前置条件是两份拓扑都已经过 `from_dict` 的完整布局校验（scene 形状、端口唯一、计数一致、id 连续、link 完整），所以这里只做**增量相关**的判定：

| 步骤 | 判定 | 失败结果 |
| --- | --- | --- |
| 快速路径 | `old.digest == new.digest` | 直接返回空增量（语义完全没变） |
| R0 | scene 不变，且属于有热更新语义的 `edge_share` / `cloud_share` / `lwd_cluster` | `single_instance` 明确拒绝，提示重启 |
| R1 | `_frozen_blocks` 三段逐字段相等 | `global block changed: ...` |
| R2 | 每个旧实例（含 dp 列表和 coordination 块）在新拓扑里逐字节相同 | `instance removed`（禁止缩容）或 `existing instance modified: ...` |
| R3 | `old_links - new_links` 为空 | `link removed` |
| R4 | 按 scene 检查新增形状 | 见下 |

R2 的作用不止"实例没改"——实例不变 + link 只增，等于**冻结了所有已实现的连接**，这是"运行中不扰动既有链路"的充分条件。

R4 分三种：

- **edge_share**：每条新 link 不能"跨新旧"——`(link.edge in old_edges) != (link.cloud in old_clouds)` 为真就拒绝（一边新一边旧）；并且新增 edge 数必须等于新增 cloud 数（整对增加）。两边都旧的 link 理论上会通过这里，但布局校验要求 link 数与实例数一致且一对一，所以实际到不了。
- **cloud_share**：不允许新增 cloud（改 cloud 自己的 plan 要重启 cloud）；每条新 link 必须是"新 edge → 已有 cloud"。
- **lwd_cluster**：不需要额外形状检查，full-mesh 的布局校验已经保证新 edge 连所有 cloud、新 cloud 被所有 edge 连。

最后一步把已通过的 link 展开成连接：

```python
edge_conns = new.connections("edge", link.edge)
connections.extend(
    conn for conn in edge_conns
    if conn.connection_id.cloud_id == link.cloud
)
```

这里必须按 link 的 cloud 过滤，注释解释得很清楚（[L934-L945](file:///c:/mainmain/vllm_host_rdma/vllm/config/lwd/topology.py#L934-L945)）：`lwd_cluster` 加一个新 cloud 时，已有 edge 会同时拥有"老 cloud 的老连接"和"新 cloud 的新连接"，而 `connections("edge", edge)` 会把该 edge 的所有 pair 都返回——只留连这条 link 对应 cloud 的那条。`edge_share` / `cloud_share` 下这个过滤是 no-op，因为新 edge 的每条连接本来就是新的。

## 设计上的几个取舍

1. **纯函数 + 显式结果类型**：没有副作用、没有异常控制流，天然可单测（[test_topology_update.py](file:///c:/mainmain/vllm_host_rdma/tests/config/lwd/test_topology_update.py)），也让 watcher 的重试逻辑保持简单。
2. **检测与应用分离**：`diff_topology` 只回答"能不能、要建什么"，真正建链在角色钩子里——edge 建控制面 + 数据面，cloud 扩 allowlist 和 listener。配置层因此不依赖任何执行钩子。
3. **与 `frontend_digest` 的分工**：watcher 去重和 no-op 快速路径用的是**完整** `digest`（必须感知任何内容变化）；前端 attach 对账用的是收窄的 `frontend_digest`（跨扩容稳定）。两者都在 `from_dict` 里算好，`diff_topology` 只消费。
4. **v1 明确不支持缩容**：R2/R3 直接拒绝移除，缩容继续走原有的 kill + heartbeat 静默 + 链路拆除 + gate 冻结语义，避免两套机制互相干扰。

完整的设计论证（Phase A 检测 / B 控制面加链 / C 数据面装配、扩到 `cloud_share` 与 `lwd_cluster` 的镜像分析）在 [lwd_topology_update.md](file:///c:/mainmain/vllm_host_rdma/docs/design/lwd_topology_update.md)。

---

**User**

对比C:\github_main_host_rdma_old\vllm_host_rdma\vllm\entrypoints\openai\lwd\frontend_attach.py，新增的watchdog是干什么的

**Assistant**

The user asks about the watchdog added in frontend_attach.py compared to old version. I saw `_start_reregister_watchdog` earlier (lines 920-963). Need to compare with old file: check if old has watchdog. Let me grep old file for watchdog/re-register and read new relevant sections. Also check design doc lwd_peer_restart_recovery.md maybe. Let me gather.

So old file has no watchdog. New adds: constants at 56-61, globals 68-71, capture at 841-843, start at 914, implementation 920-963, plus handling in the registration server (lines ~421, ~548) for repeated registration from same pid. Need to read the new file around those areas to explain precisely: how the server handles routine re-registration, idempotency, same pid handling; and the initial registration flow (run_attach_api_server).

Let me read:
- 40-80 (constants/globals)
- 400-440 (handle attach, same pid no-op?)
- 520-570 (report_endpoints handling)
- 830-920 (run_attach_api_server)
- 920-965 already seen.

Also check old run_attach_api_server for comparison (old around 600-640).

Let me read these in parallel.

Now read the FrontendAttachClient.attach to see if it sends pid. New code in _handle_attach uses message.get("pid"); check client.attach implementation to confirm pid included. Also old client method compare. Let me read new around 300-400 maybe (client class), and old same area.

Good. Old client already sends pid? Yes old at line 522 sends pid. So pid existed before; new uses it for distinguishing restarts vs watchdog rounds. Check old _handle_attach around 390-470 to confirm old behavior on re-attach of already accepted index. Let me read old 370-470.

Now check new `_handle_endpoints` around 600-660 to see idempotency for repeated watchdog report. And new registration authority `serve_background`, `wait_for_registrations`. Read 590-710.

Now check new FrontendAttachClient (709-800) for recv_timeout_s param introduced for watchdog, compare old client (no timeout param, uses env VLLM_ENGINE_READY_TIMEOUT_S which could be minutes). Read new 709-800.

Now I have complete picture. Also check old run_attach_api_server for comparison (old lines ~600-640) to see old flow: with client ... report_endpoints then wait_for_completion, no watchdog. Read old 580-640.

## 旧版缺什么

旧版 attach 是**一次性**的握手流程（[旧 frontend_attach.py#L575-L629](file:///C:/github_main_host_rdma_old/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L575-L629)）：

```
attach → 拿 client_count/frontend_args → 起本地 server → report_endpoints → wait
```

注册服务器这边也是严格一次性的（旧 [#L365-L459](file:///C:/github_main_host_rdma_old/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L365-L459)）：同一个 `client_index` 再次 ATTACH 直接回 `"already attached"`，endpoints 第二次上报直接回 `"endpoints already supplied"`，而且这个 server 只在启动阶段存活，启动 barrier 一结束就被关掉。

于是有一个故障场景无解：**primary 重启**。primary 的启动 barrier 要等计划内的远程前端把 endpoints 报上来才放行引擎启动；但 attach 前端还活得好好的，它已经报过了，不会再报第二遍。新 primary 的 barrier 只能干等——旧代码没有 grace window，等于永久挂住。

## watchdog 是什么

[#L920-L955](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L920-L955) 的 `_start_reregister_watchdog` 是 attach 前端进程里的一个**守护线程**，在首次 attach + report_endpoints 成功之后启动（[#L914](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L914)），之后每 10 秒做一轮"重新报名"：

```python
while True:
    time.sleep(_REREGISTER_INTERVAL_S)          # 10s
    with FrontendAttachClient(plan.endpoint, recv_timeout_s=_REREGISTER_TIMEOUT_S) as client:
        client.attach(plan.client_index, <同一份 config payload>, <同一 listen>)
        client.report_endpoints(plan.client_index, input_address, output_address)
```

语义分两种状态：

- **primary 还活着**：新一轮 ATTACH 命中"同一 index + 同一 pid"，primary 静默返回 ACCEPT，不改任何状态——这就是注释里的 "silent no-op"，10 秒一次的成本只是一次本地 ZMQ 往返。
- **primary 重启过**：新 primary 的 `_accepted` / `_reported` 是空的，这一轮 ATTACH + ENDPOINTS 正好把它的启动 barrier 填满，引擎立即可以启动，不用等 grace window 超时。

常量含义见 [#L56-L71](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L56-L71)：10 秒间隔、5 秒单轮超时。注释点明了不变式：**barrier 只管启动，常驻轮次管运行时**。

## 为什么实现成"每轮新建独立短连接"

几个细节都是有意为之：

1. **不复用首次 attach 的 client**。那个 client 在 `with FrontendAttachClient(...)` 作用域里，函数体结束时会被 `close()`；而且它的 `RCVTIMEO` 是 `VLLM_ENGINE_READY_TIMEOUT_S`（可能几分钟，模型加载用），拿来做 10 秒一轮的探活会把线程卡死。新版给 client 加了 `recv_timeout_s` 参数专门服务 watchdog（[#L712-L723](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L712-L723)）。
2. **每轮新建 ZMQ context/socket**，primary 不可达时异常被 `logger.debug` 吞掉，5 秒后下一轮重试，不会拖垮主线程，也不会因为 DEALER 状态残留影响下一轮。
3. **payload 通过模块级全局捕获**（`_WATCHDOG_ARGS`、`_WATCHDOG_LISTEN`，[#L68-L71](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L68-L71) 和 [#L838-L843](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L838-L843)）。watchdog 的生命周期超出 `run_attach_api_server` 的作用域，而 args namespace 和 listen 地址在整个进程内是不变的，直接捕获最省事。
4. **端点用确定性地址**（[#L890-L899](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L890-L899)）。第 3 轮、第 100 轮报的都是同一对 `input/output` 地址，引擎端 DEALER/PUSH 可以透明重连；这也是 primary 侧能"替换旧记录"而不是报冲突的前提。

## 配套的 primary 侧改动

只有前端单向重报还不够，注册服务器必须能接受重复报名。这部分的配套修改是：

| 位置 | 改动 | 作用 |
| --- | --- | --- |
| [#L419-L422](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L419-L422) | 新增 `_accepted_pids: dict[index, pid]` | 区分"watchdog 重报"和"前端重启" |
| [#L546-L558](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L546-L558) | 已 accepted 且 pid 相同 → 静默 ACCEPT | watchdog 轮次幂等，不刷状态、不重复打日志 |
| [#L559-L572](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L559-L572) | pid 不同 → 清掉该 index 的 `_accepted`/`_reported`/`_listen_addresses` 再按新 attach 处理 | attach 前端重启后能重新登记，而不是被旧的"已经 attach 过"挡住 |
| [#L648-L665](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L648-L665) | endpoints 重报从"报错"改为"替换" | 新 primary 的空 `_reported` 由 watchdog 轮次重新填满 |
| [#L667-L691](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L667-L691) | 新增 `serve_background()` 常驻服务线程 | 启动 barrier 之外仍然响应重报；配合 `attach_scope` 不再在启动结束时关闭 server |

注意 pid 字段旧版 client 就在发了（旧的 [#L509-L523](file:///C:/github_main_host_rdma_old/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L509-L523)），只是当时 primary 没用它——watchdog 引入后这个字段才真正承担"区分重启与例行重报"的职责。

## 和另一个方向的故障恢复区分开

这套机制是**双向自愈**的一半，别混：

- **watchdog（前端侧）**：解决 primary 重启。前端主动每 10 秒重新报名，把新 primary 的启动 barrier 解开。
- **`serve_background`（primary 侧）**：解决单个 attach 前端重启。服务器常驻监听，重启的前端带着新 pid 重新 ATTACH，primary 清掉旧状态后接纳它。

两者都建立在"引擎 IO 端口由 `api_bind_base_port` 确定性推导"这个前提上，否则重启方会绑到新端口，对端无法透明重连。回归测试就是 [test_frontend_attach_reregister.py](file:///c:/mainmain/vllm_host_rdma/tests/entrypoints/openai/lwd/test_frontend_attach_reregister.py)，文件名本身即说明用途。设计背景见 [lwd_peer_restart_recovery.md](file:///c:/mainmain/vllm_host_rdma/docs/design/lwd_peer_restart_recovery.md)（拓扑热更新文档称其为"the already-verified half this feature builds on"）。

---

**User**

为什么要用全局_WATCHDOG_LISTEN和_WATCHDOG_ARGS

**Assistant**

The user asks why use globals `_WATCHDOG_LISTEN` and `_WATCHDOG_ARGS` instead of closures/parameters. Need to explain design rationale, referencing code.

Let me re-read relevant pieces to be precise: run_attach_api_server and the watchdog functions. Already have:

- globals at L68-71
- capture at L841-843 in run_attach_api_server
- `_start_reregister_watchdog(plan, input_address, output_address)` at L920-955 calls `_resolve_watchdog_args()` and `_resolve_watchdog_listen()` inside the loop (each round), not captured at start.
- `_resolve_watchdog_args` returns `_WATCHDOG_ARGS`; `_resolve_watchdog_listen` returns `_WATCHDOG_LISTEN`.
- `compute_attach_config_payload(_resolve_watchdog_args(), plan)`.

So why globals? Possible reasons:
1. `_start_reregister_watchdog` signature only takes 3 params — team rule "函数入参尽量控制在 5 个". Passing args & listen would make 5 (still ok) but they deliberately used accessors.
2. Indirection via `_resolve_watchdog_*` allows tests to patch/capture (monkeypatch) — check tests use these? Let me grep tests for _WATCHDOG / _resolve_watchdog usage. In test_frontend_attach_reregister.py, maybe they set globals. Let me check.
3. Thread lifetime: watchdog thread outlives function; closure would also work (thread target=loop closes over locals). So not forced.
4. Testability: tests can set module globals to fake args without running full run_attach_api_server. And `_start_reregister_watchdog` can be called in isolation.
5. Keep the function signature small / avoid passing a huge args namespace through? args is one object; listen is tuple.

Honest answer: it's a pragmatic choice; closure would be equivalent. The explicit reasons from code/comments: "Captured by run_attach_api_server for the re-register watchdog: the args namespace and listen address are stable for the process lifetime." Plus the accessor indirection (`_resolve_watchdog_args` / `_resolve_watchdog_listen`) suggests test injection points.

Let me check tests to see whether they monkeypatch these globals, which would be a concrete justification.

## 直接答案

这是为了把一个**进程级稳定、且只初始化一次**的配置（argparse Namespace + 监听地址）带进一个模块级线程函数，同时不把 `_start_reregister_watchdog` 的参数表撑大。它依赖一个前提：**一个 attach 进程只跑一个 attach plan**。

## 具体理由

### 1. watchdog 是模块级函数，不在 `run_attach_api_server` 的调用栈里

watchdog 线程的入口是模块级的 `_start_reregister_watchdog(plan, input_address, output_address)`（[#L920-L955](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L920-L955)），而 `args` 和 listen 是在 [`run_attach_api_server`](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L834-L917) 的局部作用域里拿到的。跨函数传递只有三条路：加参数、做成闭包、或放到模块级状态。代码选了第三条。

### 2. 控制函数入参数量

如果直接传，签名会变成 `_start_reregister_watchdog(plan, input_address, output_address, args, listen)`，正好 5 个参数，踩到团队规范"函数入参尽量控制在 5 个"的上限。而 `args` 是个巨大的 Namespace，传进去会让人误以为 watchdog 依赖其中任意字段；实际它只被用来重算 `compute_attach_config_payload(args, plan)` 和 `_resolve_listen(args)`。

于是用两个只有一行的解析函数把依赖收窄：

```python
def _resolve_watchdog_args() -> Any:      # L958-L959
    return _WATCHDOG_ARGS

def _resolve_watchdog_listen() -> tuple[str, int]:   # L962-L963
    return _WATCHDOG_LISTEN
```

watchdog 循环里每轮通过它们取用（[#L941-L943](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L941-L943)），而不是直接引用全局名。

### 3. 值在进程生命周期内不变，且"只写一次"

代码注释写明了这个前提（[#L68-L71](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L68-L71)）：

> Captured by run_attach_api_server for the re-register watchdog: the args namespace and listen address are stable for the process lifetime.

`args` 是 CLI 解析结果，listen 由 `_resolve_listen(args)` 推导，二者在整个进程里不会变。写入发生在 [#L841-L843](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L841-L843)，**在守护线程启动之前**；之后线程只读。所以虽然是全局可变状态，实际是"写一次、之后只读"，不存在并发竞争。

### 4. 生命周期与失败路径更简单

watchdog 是 daemon 线程，且 `wait_for_completion_or_failure` 会一直阻塞到进程结束。用闭包当然也能捕获 `args`/listen，但闭包会把整个 `run_attach_api_server` 帧里的大对象（`sock`、`manager` 等）延长存活；用模块级只存两个明确的小值，意图更清楚，也避免线程与函数帧纠缠。

### 5. 提供了"每轮重新解析"的接缝

注意循环里是每轮调用 `_resolve_watchdog_args()` / `_resolve_watchdog_listen()`，而不是在 `_start_reregister_watchdog` 开头取一次。当前实现下二者等价，但如果以后需要"重新从某个实时来源推导"（比如运行中 refresh 配置），只要改这两个解析函数，循环体不用动。

## 代价和边界

这不是没有成本的：

- **全局可变状态**，正确性依赖"一个进程只有一个 attach plan"这一不变式。`each `vllm serve --api-server-attach ...` 是一个独立进程，`lwd_dispatch_attach_frontend` 里 `run_attach_api_server` 只会被调用一次，所以成立。若未来支持一个进程内跑多个 attach 前端，这两个全局就会被覆盖、串味，届时应改成显式参数或封装成对象。
- **默认值掩盖错误**：`_WATCHDOG_ARGS = None`、`_WATCHDOG_LISTEN = ("0.0.0.0", 0)` 保证解析函数永远不会 AttributeError，但如果有人在捕获之前启动 watchdog，会得到一个看似合法实则错误的 listen（端口 0），而不是立刻报错。当前调用顺序保证了不会发生。

## 为什么不用闭包

如果要更"干净"，可以把 watchdog 改成 `run_attach_api_server` 内部的嵌套函数，直接闭包捕获 `args` 和 `_resolve_listen(args)`，彻底去掉全局。当前写法更像是在"函数保持扁平、可单独调用"和"避免传大对象"之间做的折中；从可测性角度，模块级函数配合 `_resolve_watchdog_*` 这个接缝，也让测试可以单独驱动一轮重报逻辑，不必走完整的 server 启动流程。两种写法语义等价，选哪种属于风格取舍，不是功能必需。

---

**User**

讲解一下resolve_attach_plan的修改

**Assistant**

Need to compare resolve_attach_plan old vs new. I have new at lines 171-207. Old? Let me read old resolve_attach_plan. Grep.

I have both. New resolve_attach_plan (171-207) changes:

1. `capacity_value = lwd.topology.is_host_comm and requested_count == LWD_API_MAX_FRONTENDS` (L186-188)
2. validation condition: `if requested_count is not None and requested_count != count and not capacity_value:` raising message with parentheses (L189-193)
3. Return plan with two new fields: `api_bind_base_port=lwd.topology.deployment.api_bind_base_port`, `frontend_digest=lwd.topology.frontend_digest`, `host_comm=lwd.topology.is_host_comm` (L202/L204/L206).

FrontendAttachPlan also changed: added api_bind_base_port, frontend_digest, host_comm and docstrings for digests.

Also the error message text changed slightly: "must equal the edge_share frontend count" (removed "the ... " formatting difference: old f-string spans "must equal the " + "edge_share frontend count", new single line "must equal " + "the edge_share frontend count"). Minor.

Explain why each:

A. capacity_value: `lwd_dispatch_attach_frontend` sets args.api_server_count = LWD_API_MAX_FRONTENDS (8) for host-comm primary before later re-resolves happen: create_frontend_registration_server calls resolve_attach_plan again, watchdog's? Actually the watchdog calls compute_attach_config_payload(_resolve_watchdog_args(), plan) and FrontendAttachClient attach — not resolve. The comment says "the later re-resolves (the registration authority, the attach watchdog)". On the primary, prepare_attach_mode → create_frontend_registration_server → resolve_attach_plan(args) with args.api_server_count now 8, while plan.client_count is the yaml plan (e.g. 1). Without exemption, the mismatch check `8 != 1` raises — regression test says exactly: "--api-server-count (8) must equal the edge_share frontend count (1)". So capacity_value allows that programmatic value.

But then a genuine user passing 3 still fails (test L385-387). Why only equality to capacity, not any value? Because only the programmatic value is exactly LWD_API_MAX_FRONTENDS. It's a narrow whitelist.

B. plan carries api_bind_base_port: consumers need the band base: engine_io_addresses computation on attach side (run_attach_api_server), pre-seed on primary (wait_for_attach_registrations), conflict checks. Previously? Old run_attach_api_server computed `bind_ip = get_route_ip(...)` and ephemeral ports 0; new uses deterministic ports from band. So plan field exposes topology value.

C. frontend_digest: attach config payload key switches lwd_topology_digest → lwd_frontend_digest (L230), so elastic scale-out doesn't dead-end. The plan carries both: topology_digest for authority log, frontend_digest for comparison.

D. host_comm: drives serve dispatch (lwd_attach_primary) and capacity count; also plan.mode? Used at L983-986.

Also mention count semantics: unchanged — `count = len(lwd.topology.edges)` is the yaml plan count; it's not the capacity. New code keeps plan.client_count as plan count; capacity enters via api_server_count override. The server gets client_count = capacity via prepare_attach_mode(args, num_api_servers...) where num_api_servers = args.api_server_count = 8. So resolve returns plan count (1) and later create_frontend_registration_server expects `plan.client_count <= client_count <= LWD_API_MAX_FRONTENDS`.

Potential subtlety: `resolve_attach_plan` uses `api.requested` (requires --api-server-rpc-port or --api-server-attach). Fine.

Also note `capacity_value` is computed before the check and depends on `is_host_comm`; for device comm, user passing 8 would still be rejected (good, since capacity is a host-comm attach concept).

Also note LWD_API_MAX_FRONTENDS import added (L43) and topology import.

Let me write answer with diff-like table and reasons, referencing lines.

Also mention validation still fail-fast for genuine user override; tests. Mention the interplay with `_validate_scene` relaxed single-pair: plan count can be 1, capacity 8.

Also check whether `frontend_digest` is used in compute_attach_config_payload at L220-240. Let me recall L220-240: payload["lwd_frontend_digest"] = plan.frontend_digest. Yes from earlier grep line 230. Let me read 220-240 to quote exactly? I saw grep: frontend_attach.py:230: payload["lwd_frontend_digest"] = plan.frontend_digest. And 204 frontend_digest=...; 202 api_bind_base_port. Also diff_config_payloads compares. Reading 219-245 would give precision. Also check old payload key at old file ~ 190-210. Let me read both quickly.

## 一句话概括

`resolve_attach_plan` 的职责从"按 yaml 计划算出 index/count/endpoint/digest"扩展为"同时暴露 attach 平面需要的三个新维度"（端口带基址、前端对账摘要、通信模式），并额外加了一条**只放行程序化容量值的校验豁免**。

对比旧版 [frontend_attach.py#L136-L162](file:///C:/github_main_host_rdma_old/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L136-L162) 和新版 [#L171-L207](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L171-L207)，共五处改动。

## 1. 新增 `capacity_value` 豁免（唯一的逻辑分支改动）

```python
capacity_value = (
    lwd.topology.is_host_comm and requested_count == LWD_API_MAX_FRONTENDS
)
if requested_count is not None and requested_count != count and not capacity_value:
    raise ValueError(...)
```

见 [#L182-L193](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L182-L193)。旧版是严格的 `requested_count != count` 就报错。

**为什么必须豁免**：这个函数不是只被调用一次。primary 的流程是

1. `serve.py` 的 `lwd_dispatch_attach_frontend` 先 resolve 一次，然后（host comm 下）把 `args.api_server_count` **程序化地**改成 `LWD_API_MAX_FRONTENDS`（8）；
2. 进入 `run_multi_api_server` 后，`prepare_attach_mode` → `create_frontend_registration_server` **再次**调用 `resolve_attach_plan(args)`（[#L796](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L793-L797)）。

第二次调用时 `requested_count` 已经是 8，而 `count` 仍是 yaml 计划里的前端数（可能只有 1）——如果沿用严格相等，就会冒出回归测试里那条报错：`--api-server-count (8) must equal the edge_share frontend count (1)`，整个 primary 起不来。

**为什么只认 `LWD_API_MAX_FRONTENDS` 而不是"任意不等都放过"**：这是精确白名单，只有 dispatch 程序化写入的那一个值能被豁免。真实用户手敲 `--api-server-count 3` 仍然会 fail-fast（测试 [#L385-L387](file:///c:/mainmain/vllm_host_rdma/tests/entrypoints/openai/lwd/test_frontend_attach_reregister.py#L385-L387) 专门覆盖）；device comm 下也不豁免，因为容量化前端接入本来就只属于 host comm 的 attach 平面。

注意 `count` 的语义没变：它仍是 **yaml 计划数**（`len(lwd.topology.edges)`），不是容量。容量只通过 `args.api_server_count` 传下去，二者在 `create_frontend_registration_server` 里以 `plan.client_count <= client_count <= LWD_API_MAX_FRONTENDS` 的形式汇合。

## 2. `FrontendAttachPlan` 增加 `api_bind_base_port`

字段定义在 [#L149](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L143-L161)，赋值在 [#L202](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L202)。

旧版 attach 前端用 `get_tcp_uri(bind_ip, 0)` 绑**临时端口**；新版改成从预留端口带推导确定性地址。这需要端口带基址在两侧都能拿到：

- attach 前端：`engine_io_addresses(plan.endpoint, plan.api_bind_base_port, plan.client_index)`（[#L897-L899](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L897-L899)）；
- primary：预播种未上线前端的地址（[#L1034-L1041](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L1034-L1041)）；
- 双向冲突检查：前端的 HTTP listen 和 primary 自己的端口都不能落进这条带（[#L880-L889](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L880-L889)、[#L594-L610](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L594-L610)）。

把 `deployment.api_bind_base_port` 放进 plan，等于让"计划"成为端口带的唯一来源，避免各调用点各自 `resolve_lwd_config` 再取一次。

## 3. 增加 `frontend_digest`

字段见 [#L152-L155](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L152-L155)，赋值 [#L204](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L204)。

同时 `compute_attach_config_payload` 的对账 key 从旧版的

```python
payload["lwd_topology_digest"] = plan.topology_digest
```

改成新版的

```python
payload["lwd_frontend_digest"] = plan.frontend_digest
```

（旧 [#L183](file:///C:/github_main_host_rdma_old/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L183) → 新 [#L230](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L230)）。

原因是弹性扩容：primary 用小 yaml 启动、前端用扩容后的大 yaml 启动，实例清单不同导致完整 `digest` 必然不同，前端会被 `common configuration mismatch` 拒绝，扩缩链路在前端这一环死锁。`frontend_digest` 排除了实例清单和 `edges_num`/`clouds_num`，跨增量稳定；而它覆盖的 deployment 冻结块、qos、scene 若被改仍会拒绝 attach。

`topology_digest` 没有删，docstring 写明它现在"observability only"——注册权威启动时会把两个摘要一起打进日志（[#L818-L822](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L818-L822)），排障时能看到完整拓扑版本。

## 4. 增加 `host_comm`

字段 [#L157-L161](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L157-L161)，赋值 [#L206](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L206)。

它把"这个 attach 主节点是不是 host comm"从拓扑判断变成 plan 的显式属性，供 `lwd_dispatch_attach_frontend` 决定两件事（[#L983-L986](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L983-L986)）：

- `args.api_server_count = LWD_API_MAX_FRONTENDS if plan.host_comm else plan.client_count`；
- `args.lwd_attach_primary = plan.host_comm`，从而即使单前端也强行走 `run_multi_api_server`，保证注册服务器和 `--api-server-rpc-port` 被绑定。

也就是说，上面第 1 点的 `capacity_value` 豁免和第 4 点的 `host_comm` 是同一套机制的两端：一端写容量值，一端在再次解析时认它。

## 5. 报错文案的格式微调

旧版把 f-string 拆成 `"must equal the "` + `"edge_share frontend count"`，新版合并为 `"must equal "` + `"the edge_share frontend count"`。纯格式化，语义不变。

## 没变的部分

- `api.requested` 门槛、`VLLM_USE_RUST_FRONTEND` 拒绝、`lwd is not None` 断言；
- `count = len(lwd.topology.edges)` 仍是 yaml 计划数；
- `mode`（有 `--api-server-rpc-port` 即 primary）和 `endpoint` 的推导；
- `client_index = lwd.instance_id`。

## 归纳

这次修改的核心不是"多返回几个字段"，而是把 attach 平面从**一次性、计划内、临时端口**的模型，改成**容量化、可对账、确定性端口**的模型：`api_bind_base_port` 提供确定性地址，`host_comm` 决定是否启用这套流水线，`frontend_digest` 提供跨扩容稳定的对账口径，`capacity_value` 则保证容量值和计划数这两个不同层次的数字不会在二次解析时互相打架。

---

**User**

讲解compute_attach_config_payload的变更

**Assistant**

Compare compute_attach_config_payload old vs new. Already read new L220-232 and old L175-185. Changes:
1. `plan: FrontendAttachPlan | None = None` parameter added (new signature; old only args).
2. `plan = plan or resolve_attach_plan(args)` instead of always resolving.
3. payload key `lwd_topology_digest` → `lwd_frontend_digest`.
4. formatting: exclude=set(...) same, line wrapping.

Why the optional plan param? Because the watchdog calls `compute_attach_config_payload(_resolve_watchdog_args(), plan)` (L940-942) — passing the already-resolved plan avoids re-running resolve_attach_plan from inside the watchdog round with `_WATCHDOG_ARGS`. Wait, would re-resolve be wrong? With `_WATCHDOG_ARGS` on attach side, `args.api_server_count` is... In run_attach_api_server, after initial attach, `args.api_server_count = client_count` (which is capacity on host comm, e.g. 8). Then resolve_attach_plan(args) would see requested_count=8, is_host_comm maybe true, count = plan count (1) → capacity_value True → accepted. On device comm, args.api_server_count = client_count = plan count → fine. So re-resolve might work, but it would re-parse the YAML file each round! That's the real reason: avoid re-reading/re-validating the topology config every 10 seconds, and pin the plan resolved at startup (identity stable). Also avoids any dependency on args mutation order. Hmm—also `resolve_attach_plan` on attach side: mode would be "attach", same; endpoint same. So correctness-wise probably fine but wasteful and fragile. Let me verify the watchdog call: `compute_attach_config_payload(_resolve_watchdog_args(), plan)` — yes plan passed. Why capture _WATCHDOG_ARGS at all then? Because payload derives from args (jsonify_non_default_args over all CLI args, excluding some). So it needs args but not a re-resolve.

Also maybe `plan or resolve...` used elsewhere: initial attach in run_attach_api_server calls `compute_attach_config_payload(args, plan)`? Let me check: L846-848 `compute_attach_config_payload(args, plan)` — yes it passes plan. And create_frontend_registration_server L826 passes args only (no plan) → resolves. So new signature serves both.

Wait, L846: `client.attach(plan.client_index, compute_attach_config_payload(args, plan), _resolve_listen(args))`. Yes.

Why avoid re-resolve in attach side? Suppose the yaml file changed between startup and watchdog round; re-resolving would produce a different digest/payload and the primary would reject ("configuration mismatch") — the running frontend should keep using the config it started with. That's a strong correctness argument: the plan is the process's frozen view; watchdog must replay the same payload. Since payload includes lwd_frontend_digest from plan, passing the startup plan pins it.

Also note: `plan = plan or resolve_attach_plan(args)` — if plan None, resolve; assert not None.

Key point: digest key rename is synchronized by construction (both sides same build) per design doc.

Also mention the payload is compared by `diff_config_payloads` at primary `_handle_attach` (L577), and the key name appears in the reason text (test asserts "lwd_frontend_digest" in reason).

Also mention exclude set unchanged: `_ATTACH_CONFIG_EXCLUDE_KEYS` excludes local-only args (host/port/uds/api_server_count/api_server_rpc_port/api_server_attach/additional_config/master_addr...). Let me read the full set maybe relevant. Not necessary but can mention.

Let me read new L73-100 for the exclude set to be accurate.

Now note that new exclude set appears identical to old (same entries). The old list continued beyond line 72; probably identical. Good.

Now, check where compute_attach_config_payload is called:
- L477? earlier grep showed 421, 548 comments; let me grep call sites in new file.
- create_frontend_registration_server L826
- run_attach_api_server L847
- watchdog L940-942
- maybe server _handle_attach calls it? Let me grep.

## 实际改动只有一行

先纠正一个容易误读的点：`plan: FrontendAttachPlan | None = None` 这个可选参数和 `plan = plan or resolve_attach_plan(args)` 在旧版里**已经存在**（旧 [#L175-L181](file:///C:/github_main_host_rdma_old/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L175-L181)），不属于本次变更。

真正的功能改动只有一处（旧 [#L183](file:///C:/github_main_host_rdma_old/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L183) → 新 [#L230](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L230)）：

```diff
-    payload["lwd_topology_digest"] = plan.topology_digest
+    payload["lwd_frontend_digest"] = plan.frontend_digest
```

外加一处纯格式调整：`jsonify_non_default_args(args, exclude=set(...))` 被折成多行（新 [#L227-L229](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L227-L229)）。其余全部不变：

- 函数内延迟 import `jsonify_non_default_args`；
- `_ATTACH_CONFIG_EXCLUDE_KEYS` 排除集（逐项比对过，host/port/uds、`api_server_count`、`api_server_rpc_port`、`additional_config`、SSL 相关、docs 相关等，两边一致）；
- `payload["lwd_additional_config"] = _canonical_additional_config(plan)`；
- 返回值结构。

## 这个 payload 是干什么的

`compute_attach_config_payload` 生成的是 **attach 前端的"公共配置契约"**：

1. 取 args 中所有非默认值（`jsonify_non_default_args`），但排除 `_ATTACH_CONFIG_EXCLUDE_KEYS`——这些是每个命令自己的本地设置（监听地址、HTTP 传输、attach 管道参数、不被使用的 rendezvous 位置），本来就应该允许各前端不同；
2. 把 LWD 语义锚点放进去（`lwd_*_digest`）和规范化后的 additional_config；
3. 这份 payload 在 ATTACH 消息里发给 primary，primary 用 `diff_config_payloads` 与自己启动时的 payload 逐项比较，不一致就拒绝（[#L577-L584](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L577-L584)），拒绝原因里会带上不匹配的 key 名。

调用点有三处，共用同一份逻辑：

- primary 侧的注册权威：[`create_frontend_registration_server`](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L823-L831)；
- attach 前端首次报名：[`run_attach_api_server`](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L844-L849)；
- watchdog 每轮重报：[#L935-L944](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L935-L944)。

三者都传入已解析好的 `plan`，所以 payload 里的 digest 在整个进程生命周期内是冻结的——watchdog 第 N 轮发出的内容和第 1 轮完全一致，primary 才能把它识别为"同一配置的例行重报"。

## 为什么必须换 digest

这正是设计文档 [lwd_topology_update.md#L186-L216](file:///c:/mainmain/vllm_host_rdma/docs/design/lwd_topology_update.md#L186-L216) 的 Item 11：

- **旧行为**：用完整 `topology_digest` 对账。弹性扩容时 primary 用小 yaml 启动、前端用扩容后的大 yaml 启动，实例清单不同 → 完整 digest 必然不同 → primary 回 `common configuration mismatch: ... lwd_topology_digest ...`，前端 attach 失败，扩缩链条在最后一环断掉。
- **新行为**：用收窄的 `frontend_digest`。它排除了实例清单（`edges`/`clouds`/`instance_links`）和 `edges_num`/`clouds_num`，只覆盖 deployment 冻结块、`feature_ctrl`、`qos`、`control_plane`。清单增长时它不变，前端能正常接入；而这些"扩容本来就不该动"的字段若被改，它仍会变化，从而正确拒绝。

key 名也一并从 `lwd_topology_digest` 改成 `lwd_frontend_digest`，原因是：不匹配时 primary 会把 key 名写进拒绝原因，如果仍叫 topology，日志会误导排障的人以为比的是完整拓扑。测试里也直接断言了这一点（[test_frontend_attach_reregister.py#L406-L421](file:///c:/mainmain/vllm_host_rdma/tests/entrypoints/openai/lwd/test_frontend_attach_reregister.py#L406-L421)，`assert "lwd_frontend_digest" in response["reason"]`）。

## 一个需要注意的兼容性后果

这是一次**不向后兼容的协议变更**：新旧版本混跑时，双方 payload 的 key 不同，`diff_config_payloads` 会直接判定不匹配、拒绝 attach。设计文档对此的立场是"两侧运行同一个 build，rename 由构造保证同步"（both sides run the same build, so the rename is synchronized by construction），所以没有做双 key 兼容或版本协商。

## 归纳

`compute_attach_config_payload` 的变更本质上是**对账口径的切换**，不是新功能：把"完整拓扑摘要"换成"前端维度摘要"，并让 key 名与语义一致。函数签名、排除集、additional_config 规范化、三处调用方式都没有变化——这也说明收窄摘要的设计刻意把影响面压在了 `topology.py` 的 `frontend_digest` 计算和这一个赋值点上。

---

**User**

讲解_assert_ports_outside_engine_io_band

**Assistant**

This is a new function presumably. Let me read it and its callers, and compare with old (old may not have it). We saw new lines 343-360 earlier partially. Let me read 300-380 and grep old.

## 它是干什么的

`_assert_ports_outside_engine_io_band`（[frontend_attach.py#L343-L359](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L343-L359)）是一个**启动期的端口冲突断言**：检查"本命令自己配置的端口"有没有落进 attach 前端要用的预留引擎 IO 端口带。旧版没有这个函数，也没有端口带的概念，属于随确定性端口方案一起新增的防护。

签名是 `(base_port: int, ports: dict[str, int], owner: str)`：

- `base_port`：拓扑里的 `deployment.api_bind_base_port`（默认 61000）；
- `ports`：端口名 → 端口号，便于一次检查多个端口并给出可读报错；
- `owner`：报错里标明是谁的端口，例如 `"primary"` 或 `"attach frontend 1"`。

实现很直接：

```python
band = _engine_io_port_band(base_port)          # range(base, base + 2*8)
hits = {name: port for name, port in ports.items() if port in band}
if hits:
    raise ValueError(...)                        # 列出全部越界端口
```

[`_engine_io_port_band`](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L336-L340) 返回 `range(base, base + 2 * LWD_API_MAX_FRONTENDS)`，即 `[61000, 61015]` 共 16 个端口——**每个前端占 2 个**（input ROUTER = `base + 2*idx`，output PULL = `+1`），容量 8 个前端。

## 为什么需要它

这条带是 attach 平面预先划走的。引擎侧和前端侧的 ZMQ 连接在固定的 `base + 2*idx` / `+1` 上建立，好处是前端重启能 rebind 同一地址、运行中引擎的 DEALER/PUSH 透明重连，弹性扩容时 primary 也能预播种。

代价是这个端口空间不能再被别的服务占用。如果 `--port`（HTTP 监听）、`--api-server-rpc-port`（注册端点）、`master_port`、`data_parallel_rpc_port` 落进带内，后果是：

- 最好情况：绑定时 `EADDRINUSE` 直接崩；
- 最坏情况：某个端口抢先绑上了，对端连接到**错误的服务**，表现为数据错乱或长时间挂起，比崩溃难查得多。

所以选择在启动时 fail-fast，并且注释写明了"a collision is at best an EADDRINUSE crash, at worst a wrong-service connection"（[#L346-L350](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L343-L350)）。

## 为什么检查整条带，而不是"已配置的前端切片"

`_engine_io_port_band` 的 docstring 点明了这一点：**spec 预留的是整条带，不是当前配置的那几个**。假设 yaml 只计划 1 个前端，理论上只需要 `61000/61001`，但 `61002..61015` 仍属于未来可能 attach 的前端。如果 primary 的 HTTP 端口配成 `61003`，今天能跑，等运维加第二个前端时就会撞车——而且是在运行时才暴露。所以在启动时按整条带检查，把问题前移。

## 它在哪被调用：双向检查的两半

这个函数是"**本命令自己的端口**"这一半，两个启动点都调：

**primary 侧**（[#L808-L817](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L808-L817)）：检查自己的 HTTP 监听、注册端点，以及显式配置了的 `master_port` / `data_parallel_rpc_port`。

**attach 前端侧**（[#L877-L889](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L877-L889)）：检查自己的 HTTP 监听和注册端点。

另一半在注册权威那里：`_handle_attach` 收到每个前端的 `listen_port` 时会用 `_engine_io_port_band` **再查一遍**（[#L594-L610](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L594-L610)），越界就回 `LWD_FRONTEND_ERROR` 并给出"move --port or deployment.api_bind_base_port"的提示。

两边合起来就是注释所说的 **bidirectional conflict check**：

- 启动侧管"我不会占用别人的门牌号"；
- 权威侧管"来的前端不许占引擎的门牌号"。

只有启动侧检查是不够的：一个前端可能用不同的 yaml/参数启动，primary 无法预知它的 `--port`，只能在实际注册时校验。

## 几个实现细节

1. **一次收集全部冲突端口**再报错，而不是遇到第一个就抛。`hits` 是字典，报错里直接打印 `{'api listen': 61003, 'registration': 61010}`，运维一次就能改完。
2. **只比较端口号，不比较地址**。带是按端口区间定义的，检查是保守的：即便某个地址上绑定 61003 其实不冲突，也会被拒。这是有意的 fail-fast 取舍——允许误报，不允许漏报。
3. **与 `engine_io_addresses` 共用同一来源**。容量检查（`client_index >= LWD_API_MAX_FRONTENDS`）、地址推导、冲突检查全部走 `_engine_io_port_band` / `base + 2*idx`，避免三处各写一套魔数而漂移。
4. **报错信息自带修复指引**：给出端口、带区间、base 值，并提示"move the conflicting port or the base"，不用再翻代码。

一句话总结：它是确定性端口带方案的配套守卫，把"用户端口与引擎预留端口撞车"从运行时的难查故障，变成启动时一条带修复建议的明确报错。

---

**User**

FrontendAttachServer新增的raw_endpoint、expected_attaches成员是干什么的

**Assistant**

The user asks about new members raw_endpoint and expected_attaches in FrontendAttachServer. I have context: constructor at L394-428, raw_endpoint assigned from endpoint (L401), expected_attaches param added in constructor, default client_count-1 (L411-413), used in wait_for_registrations L455 (while len(self._reported) < self.expected_attaches), and raw_endpoint used in wait_for_attach_registrations L1035-1039, and _handle_attach? Let me grep usages to be accurate.

Let me grep raw_endpoint and expected_attaches in new file.

Need to see the exact context around constructor 390-415, wait_for_registrations 446-490, and 1030-1045. I have most already. Let me read 446-490 to quote the elastic partial start logic accurately (we saw part). Read 446-500 and 1025-1050.

Now explain:

1. `raw_endpoint`:
- In __init__, the `endpoint` argument is parsed into host/port then normalized into `self.endpoint = get_tcp_uri(host, port)` (a "tcp://ip:port" ZMQ URI) used to bind the ROUTER socket. `self.raw_endpoint = endpoint` keeps the original `IP:port` spelling.
- Why needed: `engine_io_addresses(registration_server.raw_endpoint, ...)` calls `parse_endpoint(endpoint, "--api-server-attach")`, which expects raw "IP:port", not "tcp://". Also `get_route_ip(parse_endpoint(...)[0])`. So raw is the input for deterministic address derivation on the primary side (pre-seed + missing-frontend warning).
- Also used at L469 in grace-window warning to compute what each missing frontend will bind.
- So: one is the bound ZMQ URI (`endpoint`), the other is the original endpoint spec for derivation. Could alternatively re-parse self.endpoint, but keeping raw avoids stripping the scheme and double parsing; also keeps original host form (IPv6 bracket handling).

Let me confirm __init__ lines 396-406 to quote exact: earlier read shows:
```
        host, port = parse_endpoint(endpoint, "--api-server-rpc-port")? 
```
Let me check lines 390-406 precisely. Earlier I read 400-405: `self.raw_endpoint = endpoint`, ... but the top of __init__ (L394-399) I haven't seen fully. Let me read 388-402.

Actually from earlier read at 401: `self.raw_endpoint = endpoint`; L402-405 comments. Let me quickly read 388-405 to get the endpoint parse line.

## `raw_endpoint`：原始 `host:port` 写法

构造函数里对同一个端点保留了两份（[#L397-L401](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L397-L401)）：

```python
host, port = parse_endpoint(endpoint, "--api-server-rpc-port")
self.endpoint = get_tcp_uri(host, port)   # "tcp://host:port"，用来 bind ROUTER
self.raw_endpoint = endpoint              # 原始 "host:port"
```

**为什么需要两份**：`engine_io_addresses(endpoint, base_port, client_index)`（[#L301-L333](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L301-L333)）内部要再调 `parse_endpoint(endpoint, ...)` 拿 host，再用 `get_route_ip` 算路由 IP。它接受的是 `host:port`，不是 `tcp://host:port`。如果不留原始值，就得先剥掉 `tcp://` 前缀再解析，既脆弱又丢掉了 IPv6 的括号处理等细节。

**用在哪**：primary 侧的弹性预播种和缺失前端提示。

- [wait_for_attach_registrations#L1034-L1041](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L1034-L1041)：对 `range(1, client_count)` 的每个槽位推导确定性引擎 IO 地址，填进 `addresses.inputs/outputs`，让引擎带着完整容量列表启动；
- [wait_for_registrations#L467-L472](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L467-L472)：grace window 到期时，打印每个未上线前端"将来会绑哪个地址"。

注释写得很清楚：它是"the elastic pre-seed derives every attach frontend's deterministic engine IO addresses from it"的输入。

## `expected_attaches`：启动 barrier 真正等待的数量

```python
self.client_count = client_count          # 容量
self.expected_attaches = (
    client_count - 1 if expected_attaches is None else expected_attaches
)
```

见 [#L406-L413](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L406-L413)。

旧版是一个只读属性 `client_count - 1`。新版把它拆成独立的构造参数，原因就是 `client_count` 的语义变了：

| 成员 | 含义 | 谁在用 |
| --- | --- | --- |
| `client_count` | 预留容量（host comm 下 = `LWD_API_MAX_FRONTENDS` = 8） | `_handle_attach` 的 index 上界 `1 <= index < client_count`；预播种循环范围 |
| `expected_attaches` | yaml 计划内的远程前端数（计划数 - 1，index 0 是 primary 本地前端） | `wait_for_registrations` 的 barrier 目标 |

关键在 [#L455](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L455)：

```python
while len(self._reported) < self.expected_attaches:
```

**为什么不能继续用 `client_count - 1`**：容量化之后 primary 会预留 8 个前端槽位，但 yaml 计划里可能只有 1 对。如果 barrier 等 8 个，每次启动都要干等 60 秒 grace window（[`_ATTACH_GRACE_S`](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L62-L67)）才继续，或者干脆永远等不到。拆开之后：

- **启动 barrier 只等计划内的前端**（`expected_attaches`），该等几个等几个，按计划收敛；
- **容量决定能接几个**，超出计划的前端可以在引擎跑起来之后随时 attach，靠确定性地址 + 透明重连接入，这正是弹性扩缩的前提。

供给方在 [create_frontend_registration_server#L823-L831](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L823-L831) 显式传 `expected_attaches=plan.client_count - 1`，即 plan 是 `resolve_attach_plan` 返回的 **yaml 计划数**（`FrontendAttachPlan.client_count`），而不是构造函数收到的容量值。这里两个同名 `client_count` 属于不同层次，读代码时容易混淆，但注释和字段 docstring 都做了区分：

> Capacity, not the yaml plan: the number of frontend slots the primary provisions ... The yaml plan governs which instances the engine routes; the barrier waits only for the planned remote frontends.

**默认值 `client_count - 1` 的作用**：让直接构造 server 的测试（例如 `_server()` 辅助函数）不传该参数也能保持旧语义，同时生产路径总是显式传值，不依赖这个兜底。

## 两者共同的线索

`raw_endpoint` 和 `expected_attaches` 都服务于同一套"容量化 + 弹性部分启动"模型：端口地址必须能从原始端点确定性地推导出来（`raw_endpoint`），而启动放行条件必须与容量解耦（`expected_attaches`）。缺任何一个，要么前端重启后对不上地址，要么 primary 每次启动都卡在等 8 个前端。

---

**User**

讲解wait_for_registrations变更

**Assistant**

Need old wait_for_registrations for comparison. Let me read old around 300-365 maybe. Grep function.

## 一句话概括

`wait_for_registrations` 从"**硬 barrier**（计划内前端不齐就超时报错）"变成了"**软 barrier**（宽限窗口到期后带着已注册的子集继续启动）"，这是弹性部分启动的落点。

对比旧版 [frontend_attach.py#L306-L338](file:///C:/github_main_host_rdma_old/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L306-L338) 和新版 [#L430-L489](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L430-L489)，共五处差异。

## 1. 超时默认值：引擎就绪超时 → 固定宽限窗口

```diff
- timeout = envs.VLLM_ENGINE_READY_TIMEOUT_S if timeout is None else timeout
+ timeout = _ATTACH_GRACE_S if timeout is None else timeout
```

旧版用引擎就绪超时（`VLLM_ENGINE_READY_TIMEOUT_S`，通常几分钟），新版用 [`_ATTACH_GRACE_S = 60.0`](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L62-L67)。

原因在常量注释里：这个窗口和引擎加载模型是**并行**的，模型加载本来就要几分钟，所以等待前端这 60 秒通常不额外增加启动时间；反过来，如果沿用引擎就绪超时，一个迟迟不来的前端会把 primary 卡上几分钟。60 秒是一个"足够覆盖同机前端进程启动，又不至于拖垮主节点"的经验值。

## 2. 超时后的行为：抛异常 → 日志 + 返回部分结果

这是最核心的语义变化：

```diff
  if remaining <= 0:
      missing = sorted(set(range(1, self.client_count)) - set(self._reported))
-     raise RuntimeError(
-         f"Timed out waiting for LWD attach frontends; missing "
-         f"client indices: {missing}"
-     )
+     logger.warning(
+         "... grace window elapsed; proceeding with the registered "
+         "frontends (elastic partial start). Missing client indices %s "
+         "will serve once they start and bind their deterministic engine "
+         "IO addresses: %s",
+         missing,
+         {index: engine_io_addresses(self.raw_endpoint, self.api_bind_base_port, index)
+          for index in missing},
+     )
+     return dict(self._reported)
```

旧逻辑：计划内前端没报齐 = 启动失败。
新逻辑：报齐多少算多少，引擎照常启动；缺的前端之后启动时，绑住确定性地址、由引擎的 DEALER/PUSH 透明重连接入（见 [#L437-L445](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L437-L445) 的 docstring）。

这条日志还顺带打印了每个缺失前端"将来会绑哪个 `input/output` 地址"，排障时可以直接对照。

## 3. 引擎进程死亡仍然是致命错误

```python
if engine_manager is not None and (dead := engine_manager.finished_procs()):
    raise RuntimeError(f"Engine process(es) {dead} exited while waiting ...")
```

这段和旧版完全一致，位置也没变（[#L480-L484](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L480-L484)）。软化的只是"前端没来"，"引擎自己挂了"依旧 fail-fast——否则会带着一个死引擎空转。

## 4. barrier 目标改为 `expected_attaches` 而不是容量

循环条件本身没变（`while len(self._reported) < self.expected_attaches`），但 `expected_attaches` 的含义变了：它现在由构造参数显式传入 `plan.client_count - 1`（yaml 计划数），而不是从 `client_count - 1`（容量）推导。

后果：host comm 下 primary 预留 8 个槽位，但 barrier 只等计划内的那几个；否则每次启动都会白等 60 秒甚至永远等不齐。这一点和上一个问题里的成员拆分是同一件事。

## 5. 日志措辞与 docstring

- 启动日志从 `"waiting for %d frontend(s)"` 改为 `"waiting up to %.0fs for %d frontend(s)"`，把超时时间也打出来；
- 新增完整 docstring，明确"soft startup barrier"的语义，以及为什么一个从未启动的索引"costs nothing"。

## 与外围代码的配合

软化 barrier 之所以成立，靠的是两处配套：

1. **预播种**：[`wait_for_attach_registrations`](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L1015-L1041) 在调用本函数之前，已把 `range(1, client_count)` 的每个槽位填上确定性地址。所以引擎启动时拿到的前端列表是**满容量**的，缺谁不影响引擎启动；注册成功的槽位再在 [#L1042-L1045](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L1042-L1045) 被真实上报地址覆盖。
2. **常驻注册服务**：[`serve_background()`](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L667-L680) 在 barrier 结束后继续响应运行时的重报，晚启动的前端、重启的前端都不需要 primary 重启。

## 一个需要注意的语义边界

- **计划的远程前端**（index 1..plan-1）：软等待，缺了就让引擎先跑。
- **primary 本地前端**（index 0）：不参与这个 barrier，`_reported` 从 index 1 开始统计。
- **容量之外的前端**（index >= `LWD_API_MAX_FRONTENDS`）：直接拒绝，不进入等待。

所以"弹性部分启动"只放宽了"计划内迟到"，没有放宽"超出容量"或"引擎故障"这两个真正的错误条件。

---

**User**

讲解_handle_message变更

**Assistant**

Need old `_handle_message` full body vs new. Old at 340-363. New at 491-526. Let me read both fully.

The diff is minimal:

1. Added optional `reply` parameter to `_handle_message` signature: `reply: Callable[[dict[str, Any]], None] | None = None`.
2. If None, build the default reply closure that sends over the ROUTER socket to `identity`.
3. Docstring added explaining injectability for tests (server built with `object.__new__`, no real socket).
4. Everything else identical: decode, type dispatch, unknown type error, exception catch, suppress ZMQError on reply.

Why: testability. Tests can call `_handle_message(identity, payload, reply=capture)` against a server created via `object.__new__(FrontendAttachServer)` without constructing a real ZMQ socket/context. Indeed tests earlier use `_handle(server, _attach_message(...))` helper. Let me verify test helper `_handle` to state precisely. Grep in test file.

## 改动只有一处：`reply` 变成可注入

对比旧版 [frontend_attach.py#L340-L363](file:///C:/github_main_host_rdma_old/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L340-L363) 和新版 [#L491-L526](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L491-L526)，逻辑分支、错误处理、消息分发完全一致，唯一变化是签名和 `reply` 的构造方式：

```diff
-    def _handle_message(self, identity: bytes, payload: bytes) -> None:
-        def reply(message: dict[str, Any]) -> None:
-            self._socket.send_multipart([identity, msgspec.msgpack.encode(message)])
+    def _handle_message(
+        self,
+        identity: bytes,
+        payload: bytes,
+        reply: Callable[[dict[str, Any]], None] | None = None,
+    ) -> None:
+        """... ``reply`` is injectable so tests can drive the handler against
+        a server built with ``object.__new__`` (no real socket)."""
+        if reply is None:
+
+            def reply(message: dict[str, Any]) -> None:
+                self._socket.send_multipart(
+                    [identity, msgspec.msgpack.encode(message)]
+                )
```

也就是：

1. 新增可选参数 `reply`；
2. 只有 `reply is None` 时才构造原来那个"通过 ROUTER socket 发回给 `identity`"的闭包；
3. 加了 docstring 说明用途。

其余代码逐行相同——`msgspec.msgpack.decode`、非 dict 校验、按 `_MSG_ATTACH` / `_MSG_ENDPOINTS` 分发、未知类型回 `_MSG_ERROR`、最外层 `except Exception` 打日志并 `suppress(zmq.ZMQError)` 回错误。

## 为什么要这样改

这是一个**测试接缝**，不是生产逻辑。原来的 `reply` 闭包硬编码依赖 `self._socket.send_multipart`，而注册服务器的行为测试（同 pid 静默重报、新 pid 重置状态、endpoints 重报替换）本质是纯状态机逻辑，没必要为每条用例起一个真实 ZMQ ROUTER socket。

新版让测试可以这样驱动：

```python
server = object.__new__(FrontendAttachServer)   # 不跑 __init__，没有 socket
...
server._handle_message(b"identity", payload, reply=capture)   # 注入收集器
```

见 [test_frontend_attach_reregister.py#L38-L50](file:///c:/mainmain/vllm_host_rdma/tests/entrypoints/openai/lwd/test_frontend_attach_reregister.py#L38-L50)（用 `object.__new__` 手动填状态）和 [#L75-L82](file:///c:/mainmain/vllm_host_rdma/tests/entrypoints/openai/lwd/test_frontend_attach_reregister.py#L75-L82)（`_handle` 辅助函数注入 reply）。

如果没有这个参数，测试要么必须构造真实 socket，要么得 monkeypatch `self._socket`，两种方式都更重、更容易干扰被测状态。

## 对生产行为的影响

**没有影响**。生产路径的调用方（`_serve_forever`、`wait_for_registrations`）都不传 `reply`，走 `reply is None` 分支，构造出的闭包与旧版逐字等价。唯一细微差别是闭包现在定义在 `if` 块内，仅在需要时才创建；对每条消息的逻辑没有可观察变化。

## 顺带暴露的设计味道

`identity` 参数在注入模式下其实没用（测试传 `b"identity"` 占位）。之所以保留，是因为它是协议的一部分：生产回调必须知道往哪个 DEALER 身份回包。用"可选 reply + 必填 identity"而不是把两者都拆出去，算是侵入最小的做法——只增加了一个带默认值的参数，没有改动 `_handle_attach` / `_handle_endpoints` 的签名。

---

**User**

讲解_handle_attach变更

**Assistant**

## 变更总览

对比旧版 [_handle_attach#L365-L437](file:///C:/github_main_host_rdma_old/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L365-L437) 和新版 [#L528-L637](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L528-L637)，校验逻辑（index 范围、config payload 比对、listen 地址合法性、监听冲突）基本没动，改动集中在"重复 ATTACH 怎么处理"和"新增一项权威侧端口检查"。

## 1. 重复 ATTACH：从"一律拒绝"变成"按 pid 分流"

旧版：

```python
if index in self._accepted:
    reply({"type": _MSG_ERROR,
           "reason": f"client_index {index} is already attached"})
    return
```

新版（[#L546-L572](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L546-L572)）：

```python
if index in self._accepted:
    if message.get("pid") == self._accepted_pids.get(index):
        # 同一活进程的例行重报：静默 ACCEPT
        reply({...ACCEPT...}); return
    # 新 pid：前端重启，清掉死进程的陈旧状态
    self._accepted.discard(index)
    self._reported.pop(index, None)
    self._listen_addresses.pop(index, None)
```

`_accepted_pids` 是新增的状态（[#L419-L422](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L419-L422)），在最终接受时写入（[#L624](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L624)）。

这一条是为 watchdog 服务的。attach 前端每 10 秒重报一次（[#L920-L955](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L920-L955)）：

- **pid 相同** → 就是那个活着的前端在例行重报，primary 静默回 ACCEPT，**不刷状态、不重复打接受日志**，保证 10 秒一轮是零副作用；
- **pid 不同** → 原进程已死，新进程复用同一个 `client_index`。此时必须清掉 `_accepted`、`_reported`、`_listen_addresses` 三份陈旧记录：旧进程上报的 endpoints 已经失效，旧监听地址也不能参与后续冲突判断（否则会跟自己冲突）。清完后继续走完整的首次接入流程。

旧版没有这个分流，watchdog 第一轮就会收到 "already attached" 而失败，重启的前端也永远接不回来。

## 2. 新增权威侧端口带检查

```python
if listen_port in _engine_io_port_band(self.api_bind_base_port):
    reply({"type": _MSG_ERROR,
           "reason": "... falls inside the reserved engine IO band ...; move --port or deployment.api_bind_base_port"})
    return
```

见 [#L594-L610](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L594-L610)。这是"双向冲突检查"的另一半：

- 启动侧（primary 自己、attach 前端自己）用 `_assert_ports_outside_engine_io_band` 检查自己的端口；
- 权威侧在这里检查**每一个来注册的前端的 `listen_port`**。

primary 无法预知前端用什么 `--port` 启动，只能在注册时校验；如果允许它落进 `[api_bind_base_port, +15]`，会和引擎 IO 端口抢绑定，轻则 `EADDRINUSE`，重则连到错误服务。

## 3. index 上界从"计划数"变为"容量"

条件本身没变：

```python
not (1 <= index < self.client_count)
```

但 `client_count` 的语义变了——旧版等于 yaml 计划的前端数，新版是预留容量（host comm 下 `LWD_API_MAX_FRONTENDS` = 8）。所以新版允许 `index` 超出计划、但在容量内的前端接入，这正是"支持超出计划但在容量范围内的前端接入"那部分能力。真正的容量硬边界仍在 `engine_io_addresses` 里（`client_index >= LWD_API_MAX_FRONTENDS` 抛错），`_handle_attach` 这里是第一道门。

## 4. 状态写入多了一项

```python
self._listen_addresses[index] = (listen_host, listen_port)
self._accepted.add(index)
self._accepted_pids[index] = message.get("pid")   # 新增
```

见 [#L622-L629](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L622-L629)。pid 的写入使得下一次同 index 的 ATTACH 能区分"重报"与"重启"。

## 两个典型时序

**场景 A：primary 存活，watchdog 例行重报**

```
attach(index=1, pid=1234, payload) → 已在 _accepted，pid 相同 → ACCEPT（静默）
report_endpoints(index=1, 同地址) → _handle_endpoints 替换同值 → BARRIER
```

全程无状态变化、无日志噪音。

**场景 B：attach 前端崩溃重启**

```
新进程 attach(index=1, pid=5678, payload)
→ 已在 _accepted，pid 不同 → 清 _accepted/_reported/_listen_addresses
→ 重新校验 payload/listen/端口带/冲突
→ 记录新 pid → ACCEPT
→ 新进程 report_endpoints(自己的新地址) → BARRIER
```

新进程必须重新上报 endpoints，因为清了 `_reported`；同时它的监听端口要重新过一遍冲突检查（原 index 的旧地址已被移除，不会自冲突）。

## 没变的部分

- `config_payload` 必填校验与 `diff_config_payloads` 比对；
- `listen_host` / `listen_port` 类型校验；
- `_listen_addresses_conflict` 的遍历式冲突检查；
- ACCEPT 响应体（`client_count`、`stats_update_address`、`frontend_args`）；
- 最外层 `_handle_message` 的异常兜底，保证一个坏请求不会带崩整个权威。

这几块保持不变很重要：重复注册的宽容只发生在"同一个已认证前端"身上，配置一致性、监听冲突这些安全约束对任何一次 ATTACH（包括重报和重启）都仍然生效。

---

**User**

进程重启pid肯定会变是吗

**Assistant**

## 结论：通常变，但不是保证

"重启后 PID 一定不同"在工程上是个**概率性假设**，不是硬保证。代码里 `_accepted_pids` 的判据（[#L546-L558](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L546-L558)）正是建立在这个假设上，方向不对称：

- **pid 不同 ⇒ 一定发生过重启**：可靠。一个活着的进程不可能改变自己的 PID。
- **pid 相同 ⇒ 一定是同一进程**：**不可靠**。这是本设计接受的近似。

## PID 什么时候会"没变"

Linux 的 PID 是递增分配、到 `pid_max` 后回绕复用的（x86_64 常见 `4194304`，部分系统 `32768`）。所以以下情况都可能让新进程拿到旧进程的 PID：

1. **回绕复用**：PID 空间耗尽后，新进程可能正好分到刚死掉的那个号。高频创建进程的系统（CI、容器平台）比想象中容易发生。
2. **PID namespace / 容器**：如果每个 attach 前端跑在独立容器里，容器重启后新进程在新的 namespace 里很可能是 **PID 1**——第一次和第二次完全"相同"。这是最现实的重合场景。
3. **不同主机**：注释里说明该部署模型假定 same-host（[#L1029-L1033](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L1029-L1033)），但跨主机时 PID 空间本来就独立，`1234` 在 A 机和 B 机可以同时存在。不过 `client_index` 唯一标识逻辑前端，只有"同 index 换机重启"才会触发误判。

## 误判为"同进程"会漏掉什么

如果重启后的前端恰好复用旧 PID，primary 会走静默 ACCEPT 分支，跳过三件本该做的事：

1. 不清理 `_listen_addresses[index]` 里的旧监听地址；
2. 不清理 `_reported[index]` 里的旧 endpoints；
3. 不做新一轮的监听冲突检查和端口带检查。

实际影响有限，因为：

- 新版 `_handle_endpoints` 是"重报即替换"（[#L648-L665](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L648-L665)），新进程上报自己的地址后会覆盖旧值；
- 引擎 IO 地址本来就是由 `client_index` 确定性推导的，重启后绑同一地址，引擎侧透明重连不依赖 primary 的记录；
- 对"primary 重启"这个主要场景无影响——新 primary 的 `_accepted`/`_reported` 本来就是空的，走的是全新 ATTACH 分支。

真正的残留风险是边缘情况：比如新进程换了 `--port`，旧监听地址留在 `_listen_addresses` 里，之后另一个前端恰好用这个旧端口启动，会被误判为冲突而拒绝；或者新进程在 report 之前又崩了，primary 仍认为该 index 的 endpoints 已齐。

## 为什么仍然用 pid

这是一个"足够便宜且覆盖主要故障"的取舍：

- watchdog 每 10 秒一次，用 PID 比较零成本；
- 不引入额外协议字段、不依赖持久化状态；
- 主要矛盾是"把例行重报和真重启分开"，PID 已经能覆盖绝大多数情况。

如果要做到严格正确，得换成不可复用的身份标识，常见做法：

- 进程启动时间（`/proc/<pid>/stat` 的 starttime，或 `psutil.Process().create_time()`）；
- 机器/容器 boot id + 随机 nonce（前端启动时生成 UUID 随 ATTACH 发送）；
- 前端进程自己生成的 instance token，重报时保持不变、重启时必然变化。

代价是要多维护一个字段和它的语义。当前实现选择了"pid 足够用"，并在注释里明确写成"pid of the live attached process"，等于承认了它基于 liveness 的近似。

## 一个附带的小坑

`_accepted_pids[index] = message.get("pid")` 用的是 `.get()`，如果某个客户端在 ATTACH 里**不带 pid**，会存成 `None`。此时任何同样不带 pid 的重报都会命中 `None == None`，被当作例行重报。当前两侧同 build、新版 client 固定发送 `os.getpid()`（[#L751-L756](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L749-L758)），所以不会触发；但如果将来接第三方前端或做协议兼容，这是个需要显式校验的边界。

---

**User**

讲解一下_serve_forever

**Assistant**

## 它是谁、在哪启动

`_serve_forever`（[frontend_attach.py#L682-L691](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L682-L691)）是注册权威的**常驻消息循环**，跑在 `serve_background()` 启动的守护线程里（[#L667-L680](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L667-L680)，线程名 `lwd-frontend-attach-server`）。

旧版没有它：注册服务器只在 `wait_for_registrations` 里同步 poll，启动 barrier 一收齐，服务器就在 `with` 块结束时被关掉。新版的启动流程变成：

```
预播种容量槽位 → wait_for_registrations()（启动期同步收注册）
→ serve_background()（把 socket 交给常驻线程，直到进程退出）
```

调用点在 [wait_for_attach_registrations#L1046-L1049](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L1046-L1049)，注释就是它的职责：**barrier 只管启动，运行时重报由常驻线程服务**。

## 循环结构

```python
def _serve_forever(self) -> None:
    while not self._closed:
        try:
            events = self._socket.poll(_POLL_INTERVAL_S * 1000)
        except zmq.ZMQError:
            return  # closed at shutdown
        while events:
            identity, payload = self._socket.recv_multipart()
            self._handle_message(identity, payload)
            events = self._socket.poll(0)
```

拆开看是三层：

1. **外层循环**：`while not self._closed`，进程活多久跑多久（daemon 线程，进程退出即终止）。
2. **带超时的 poll**：每次最多阻塞 `_POLL_INTERVAL_S`（1 秒，[#L55](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L55)）。超时返回 0 事件，回到外层重新检查 `_closed`。这保证关停最多延迟 1 秒，而不是永久阻塞在 socket 上。
3. **排空队列**：一旦有事件，用 `poll(0)` 连续处理，直到队列为空。这样一轮醒来能处理 burst 的多个请求，不用每条等一次 1 秒。

每条消息交给 `_handle_message`。它内部用 `identity` 构造 reply 闭包，通过同一个 ROUTER socket 发回给对应的前端 DEALER（[#L500-L505](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L500-L505)）。

## 它实际处理哪几类请求

线程启动时机在启动 barrier 之后，所以它面对的都是"运行时事件"：

| 请求 | 处理结果 |
| --- | --- |
| watchdog 的例行 ATTACH（同 index 同 pid） | 静默回 ACCEPT，不改状态 |
| 重启前端的 ATTACH（同 index 新 pid） | 清掉旧 `_accepted`/`_reported`/`_listen_addresses`，按新前端重新接入 |
| 重启 primary 后各前端的重新上报 | 填充新 primary 的空状态，解开其启动 barrier |
| endpoints 重报 | 替换 `_reported` 中该 index 的地址 |

没有这个线程，上面四类请求全部没人应答——前端会在 ATTACH 上等到 5 秒超时（watchdog 的 `_REREGISTER_TIMEOUT_S`）或更久，重启的前端则永久挂起。这正是 `attach_scope` docstring 里记录的旧版缺陷（[#L1052-L1059](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L1052-L1061)）：

> closing there left nothing listening on the registration endpoint — an attach frontend restarted at runtime hung forever on its ATTACH reply.

## 关停路径

```python
def close(self) -> None:
    self._closed = True
    self._socket.close(linger=0)
    self._ctx.term()
```

见 [#L693-L700](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L693-L700)。`_serve_forever` 对 `poll` 的 `zmq.ZMQError` 采取"直接 return"，就是为这个竞态准备的：`close()` 可能在另一线程关掉 socket，正在 poll 的线程会抛 `ZMQError`，此时静默退出即可，不需要打错误日志。

```python
while not self._closed:          # 先置位
    try:
        events = self._socket.poll(...)   # 再可能抛 ZMQError
    except zmq.ZMQError:
        return                   # 正常关停路径，不是故障
```

`linger=0` 保证 `close()` 不会为了发送残留消息而阻塞。

## 并发模型

- **消息处理是单线程的**：启动期由 `wait_for_registrations` 在主线程同步 drain，barrier 结束后主线程不再碰 socket，改由 `_serve_forever` 独占。两个阶段没有重叠，所以没有锁。
- **例外是 `close()`**：它可以在任意线程（进程退出路径）触发，与 poll 存在竞态，靠上面的 `ZMQError` 分支兜底。
- `_handle_message` 自己捕获所有业务异常（[#L523-L526](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L523-L526)），保证一条坏消息不会让常驻线程退出——否则整个权威就"失聪"了，后续所有重报都会超时。

## 一个可以注意到的边界

内层 `recv_multipart()` 本身没有再包 `try/except ZMQError`。如果在 poll 返回事件之后、recv 之前 socket 被关闭，理论上会抛异常并终止线程。实际很难触发（poll 与 recv 之间窗口极小，且此时进程正在退出），而且线程是 daemon、异常也发生在关停阶段，所以代码选择不为此增加噪声。若将来要更严格，可以把整个内层 while 也包进 `ZMQError` 保护。

---

**User**

讲解FrontendAttachClient的__init__

**Assistant**

## 唯一的变更：新增 `recv_timeout_s`

对比旧版 [frontend_attach.py#L477-L490](file:///C:/github_main_host_rdma_old/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L477-L490) 和新版 [#L709-L724](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L709-L724)：

```diff
-    def __init__(self, endpoint: str) -> None:
+    def __init__(self, endpoint: str, recv_timeout_s: float | None = None) -> None:
         host, port = parse_endpoint(endpoint, "--api-server-attach")
         self.main_address = get_tcp_uri(host, port)
         self._ctx = zmq.Context()
         self._socket = make_zmq_socket(
             self._ctx, self.main_address, zmq.DEALER, bind=False, linger=0
         )
-        self._socket.setsockopt(zmq.RCVTIMEO, int(envs.VLLM_ENGINE_READY_TIMEOUT_S * 1000))
+        if recv_timeout_s is None:
+            timeout = envs.VLLM_ENGINE_READY_TIMEOUT_S
+        else:
+            timeout = recv_timeout_s
+        self._socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
         self._closed = False
```

其余逐行不变。下面按执行顺序说明每一行在做什么，再讲这个参数为什么出现。

## 逐行说明

**`parse_endpoint(endpoint, "--api-server-attach")`**
校验并拆解 `host:port`：必须是具体单播 IP、IPv6 必须带方括号、端口在 `[1, 65535]`。错误文案用 `--api-server-attach` 是因为这个 client 只在 attach 命令里用，报错要指向用户实际敲的 flag。

**`self.main_address = get_tcp_uri(host, port)`**
转成 ZMQ 能直接用的 `tcp://ip:port`，存成实例属性主要是因为 `_roundtrip` 超时时要把它写进错误信息（[#L731-L734](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L726-L734)）。

**`self._ctx = zmq.Context()` + `make_zmq_socket(..., zmq.DEALER, bind=False, linger=0)`**
每个 client 自建一个 context 和一个 DEALER socket，主动连主节点的 ROUTER：

- `bind=False`：client 是连接方，不监听；
- DEALER/ROUTER 组合让主节点能按身份把回复路由回具体的请求方；
- `linger=0`：关闭时不为未发出的消息阻塞，配合 `__exit__` → `close()` 里的 `socket.close()` + `ctx.term()`，保证 watchdog 每 10 秒建/销一个 client 不会卡住。

**`RCVTIMEO`**
设置 recv 的硬超时。这是 `_roundtrip` 唯一的阻塞点，超时后 ZMQ 抛 `zmq.Again`，被 `_roundtrip` 捕获并转成 `RuntimeError`，从而让"primary 不响应"表现为一次可处理的失败，而不是线程永久挂死。

**`self._closed = False`**
`close()` 的幂等标志，避免 context 被 terminate 两次。

**`__enter__` / `__exit__`**
让调用方用 `with FrontendAttachClient(...) as client:`，任何分支（包括异常）都能保证 socket 和 context 被释放。

## 为什么加 `recv_timeout_s`

初始 attach 和 watchdog 重报对超时的诉求完全相反：

| 调用方 | 场景 | 合适超时 |
| --- | --- | --- |
| `run_attach_api_server` 首次 attach | primary 可能正在加载模型、启动较慢，ATTACH 等待是合理的 | `VLLM_ENGINE_READY_TIMEOUT_S`（通常几分钟） |
| `_start_reregister_watchdog` 每轮重报 | primary 可能已死/正在重启，探活必须快速失败 | `_REREGISTER_TIMEOUT_S = 5.0`（[#L60-L61](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L56-L61)） |

如果沿用引擎就绪超时，watchdog 线程在 primary 挂掉时每轮会被卡住几分钟，10 秒的巡检节奏完全失效，而且 `service_background` 之外的重连语义也会被拖慢。参数化后，默认值保持旧行为，watchdog 显式传短超时（[#L935-L937](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L935-L944)），互不影响。

同时这也解释了 watchdog 为什么"每轮新建 client"而不是复用：短超时和长超时是 socket 级配置，且 `with` 作用域决定了 client 的生命周期只覆盖一轮，复用反而要处理 socket 状态和超时的重新设置。

## 一个值得注意的小瑕疵

`_roundtrip` 的超时错误信息仍然写死了环境变量：

```python
except zmq.Again:
    raise RuntimeError(
        f"No response from LWD primary frontend at {self.main_address} "
        f"within {envs.VLLM_ENGINE_READY_TIMEOUT_S:.0f}s"
    )
```

见 [#L730-L734](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L726-L734)。当 watchdog 用 5 秒超时失败时，日志里会显示"within 600s"之类的环境变量值，与实际等待时间不符。对排障有轻微误导，但不影响行为——watchdog 的异常本来只打 `logger.debug`（[#L948-L953](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L948-L953)）。如果要修，把生效的 `timeout` 存成 `self._recv_timeout_s` 再在错误信息里引用即可。

## 归纳

`__init__` 的职责没变：**校验端点、建独立 DEALER 连接、配置可中断的接收超时**。变更是把超时从"写死取环境变量"变成"可注入，默认保持旧值"，服务两个生命周期和失败语义完全不同的调用方：首次接入愿意等，运行期探活必须快。

---

**User**

讲解create_frontend_registration_server变更

**Assistant**

Need old create_frontend_registration_server (old ~L550-572) and new (L793-831). I have new fully. Old at 550-572 partially read:

```
    plan = resolve_attach_plan(args)
    assert plan is not None and plan.mode == "primary"
    if client_count != plan.client_count:
        raise ValueError("[LWD] API server count does not match edge_share topology")
    ...
    return FrontendAttachServer(
        endpoint=plan.endpoint,
        client_count=client_count,
        config_payload=compute_attach_config_payload(args, plan),
        main_listen=_resolve_listen(args),
        frontend_args=compute_frontend_args_payload(args),
    )
```

Need full old block to see conflict checks. Let me read old 540-575.

## 变更总览

对比旧版 [frontend_attach.py#L559-L572](file:///C:/github_main_host_rdma_old/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L559-L572) 和新版 [#L793-L831](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L793-L831)，函数仍是"解析 plan → 校验 → 构造 `FrontendAttachServer`"四步，但有四处实质改动。

## 1. 校验从"严格相等"改为"计划数 ≤ 容量 ≤ 端口带上限"

旧版：

```python
if client_count != plan.client_count:
    raise ValueError("[LWD] API server count does not match edge_share topology")
```

新版（[#L798-L807](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L798-L807)）：

```python
if not plan.client_count <= client_count <= LWD_API_MAX_FRONTENDS:
    raise ValueError(
        f"[LWD] API server count ({client_count}) must cover the "
        f"edge_share frontend plan ({plan.client_count}) and stay "
        f"within the reserved engine IO band ({LWD_API_MAX_FRONTENDS})"
    )
```

这是"支持超出计划但在容量范围内的前端接入"的核心落点。参数 `client_count` 来自 `run_multi_api_server` 的 `num_api_servers`，host comm 下已被 dispatch 设为 `LWD_API_MAX_FRONTENDS`（8），而 `plan.client_count` 是 yaml 计划数（可能为 1）。两边关系是：

- **下界** `plan.client_count`：容量必须覆盖计划，否则计划内的前端会因 index 越界而接不进来；
- **上界** `LWD_API_MAX_FRONTENDS`：不能超出预留端口带（每个前端占 2 个端口，共 16 个）。

旧版要求严格相等，等于把"容量"和"计划"绑死；新版允许中间地带，前端可以在计划之外、容量之内随时 attach，无需重启 primary。

## 2. 新增 primary 自身端口与端口带的冲突检查

```python
ports = {
    "api listen": _resolve_listen(args)[1],
    "registration": parse_endpoint(plan.endpoint, "--api-server-rpc-port")[1],
}
for name in ("master_port", "data_parallel_rpc_port"):
    if (value := getattr(args, name, None)) is not None:
        ports[name] = int(value)
_assert_ports_outside_engine_io_band(plan.api_bind_base_port, ports, "primary")
```

见 [#L808-L817](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L808-L817)。

这利用了一个事实：能走到这里说明 primary 已经确定要启用 attach 端口带，所以此刻就能做**启动侧**的冲突检查。检查四个端口：

- HTTP 监听 `--port`；
- 注册端点 `--api-server-rpc-port`；
- `master_port`、`data_parallel_rpc_port`（仅当显式设置）。

后两个虽然被排除在配置对账 payload 之外（见 `_ATTACH_CONFIG_EXCLUDE_KEYS`），但在这台机器上仍可能被绑定，所以端口带冲突必须查。任何一个落进 `[base, base+15]` 就 fail-fast，避免运行时 `EADDRINUSE` 或连错服务。这是双向检查的一半，另一半在 `_handle_attach` 里校验每个前端的 listen 端口（[#L594-L610](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L594-L610)）。

## 3. 新增启动日志：同时打印两个摘要

```python
logger.info(
    "LWD registration authority: frontend digest %s (topology %s)",
    plan.frontend_digest,
    plan.topology_digest,
)
```

见 [#L818-L822](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L818-L822)。这是一条纯观测性日志：`frontend_digest` 是 attach 对账实际使用的收窄摘要，`topology_digest` 是完整拓扑摘要。排障时如果前端 attach 被拒，可以直接拿 primary 日志里的 `frontend_digest` 和前端侧比较，同时用完整 digest 确认两侧 yaml 是否同一版本。

## 4. 构造 server 时补两个新参数

```python
return FrontendAttachServer(
    ...,
    api_bind_base_port=plan.api_bind_base_port,
    expected_attaches=plan.client_count - 1,
)
```

见 [#L823-L831](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L823-L831)。

- **`api_bind_base_port`**：server 需要它做两件事——给 `engine_io_addresses` 推导确定性地址（预播种、grace window 提示），以及 `_handle_attach` 里的端口带校验。旧版 server 完全没有端口带概念。
- **`expected_attaches = plan.client_count - 1`**：明确告诉 barrier 只等**计划内**的远程前端，而不是等容量 8 个。这就是上一轮讨论的"计划数 vs 容量"拆分在构造点的体现：`client_count` 传容量，`expected_attaches` 传计划。

## 与调用链的关系

这个函数由 `prepare_attach_mode` 调用（[#L990-L997](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L990-L997)），而它只在 `args.api_server_rpc_port` 非空（即 primary）时触发：

```
serve.py dispatch → run_multi_api_server
  → prepare_attach_mode(args, num_api_servers, is_ray_dp)
      → create_frontend_registration_server(args, num_api_servers)
```

注意 `num_api_servers` 在 host comm 下是容量（8），device comm 下是计划数（1）。新校验同时兼容这两种：device comm 时 `client_count == plan.client_count`，落在区间端点上，行为与旧版等价。

## 归纳

这次修改把注册权威从"**按计划精确起一个服务器**"升级为"**按容量铺一条端口带，并在启动时把冲突挡在门外**"：校验放宽到区间、新增 primary 侧端口带检查、把端口带基址和真实 barrier 目标注入 server、加上可对账的双摘要日志。四条改动都服务于同一个目标——让计划外的前端能在容量内随时接入，同时不牺牲 fail-fast。

---

**User**

讲解run_attach_api_server变更

**Assistant**

## 变更总览

对比旧版 [frontend_attach.py#L575-L629](file:///C:/github_main_host_rdma_old/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L575-L629) 和新版 [#L834-L917](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L834-L917)，整体流程（首次 ATTACH → 继承参数 → 本地起 HTTP server → report endpoints → 等进程结束）没变，改动了四处，全部围绕"确定性端口 + 运行时自愈"。

## 1. 函数开头捕获 watchdog 所需的两个全局值

```python
global _WATCHDOG_ARGS, _WATCHDOG_LISTEN
_WATCHDOG_ARGS = args
_WATCHDOG_LISTEN = _resolve_listen(args)
```

见 [#L838-L843](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L838-L843)。原因前面讨论过：watchdog 线程的生命周期超出本函数作用域，而 args 和 listen 在整个进程里不变——payload 由 CLI args 推导，listen 也由 args 决定。捕获发生在首次 ATTACH 之前，保证 watchdog 第一轮就有正确输入。

## 2. 新增 bind 侧端口带冲突检查

旧版 `setup_server` 之后直接 `gather_actual_addresses`。新版在两者之间插入（[#L876-L889](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L876-L889)）：

```python
listen_address, sock = setup_server(args, reuse_port=False)
_assert_ports_outside_engine_io_band(
    plan.api_bind_base_port,
    {
        "api listen": _resolve_listen(args)[1],
        "registration": parse_endpoint(plan.endpoint, "--api-server-attach")[1],
    },
    f"attach frontend {plan.client_index}",
)
```

检查的是这个 attach 命令自己会占用的两个端口：HTTP 监听和注册端点。它们如果落进即将绑定的引擎端口带 `[base+2*idx, +1]` 所在区间，就会与引擎 IO 端口抢绑定。放在 `setup_server` 之后、worker 进程创建之前，是为了在真正付出进程启动成本前 fail-fast；此时只有父进程的 socket，异常抛出后进程退出，没有子进程需要清理。

这是双向检查的 bind 侧一半：primary 在 `create_frontend_registration_server` 里查自己的端口，权威在 `_handle_attach` 里查每个上报前端的 listen 端口，attach 前端在这里查自己的端口。

## 3. 引擎 IO 地址从临时端口改为确定性推导

旧版：

```python
bind_ip = get_route_ip(parse_endpoint(plan.endpoint, "--api-server-attach")[0])
input_addresses=[get_tcp_uri(bind_ip, 0)],   # 端口 0 = 内核分配
output_addresses=[get_tcp_uri(bind_ip, 0)],
```

新版（[#L890-L899](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L890-L899)）：

```python
input_address, output_address = engine_io_addresses(
    plan.endpoint, plan.api_bind_base_port, plan.client_index
)
```

这是本次变更里对行为影响最大的一处。旧版用临时端口，前端一旦重启就会绑到新地址，而运行中的引擎还连着死进程的旧地址，重连无从谈起。新版用"拓扑端口带 + client_index"确定性推导：

- 前端重启 → rebind **同一对端口** → 引擎的 DEALER/PUSH 透明重连；
- primary 预播种（`wait_for_attach_registrations`）用**同一个函数**推导，所以两侧地址不可能漂移；
- 端口带容量检查也内聚在 `engine_io_addresses` 里（`client_index >= LWD_API_MAX_FRONTENDS` 直接抛错）。

注释里把这个不变式写得很明确（[#L890-L896](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L890-L896)）："ephemeral ports would leave the engine connected to the dead process's address"。

## 4. report 成功后启动 watchdog

```python
inputs, outputs = manager.gather_actual_addresses()
client.report_endpoints(plan.client_index, inputs[0], outputs[0])
_start_reregister_watchdog(plan, inputs[0], outputs[0])
wait_for_completion_or_failure(api_server_manager=manager)
```

见 [#L911-L915](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L911-L915)。

顺序是有讲究的：

1. 先 `report_endpoints` 完成首次报名，保证启动 barrier 在当前 primary 上可解；
2. 再启动 watchdog。它拿着**实际绑定成功**的 `inputs[0]` / `outputs[0]` 每 10 秒重放 ATTACH + ENDPOINTS；
3. primary 存活时这些轮次是静默 no-op；primary 重启后，新 primary 的空 barrier 被下一轮填满，引擎无需等待 grace window 即可启动。

失败路径也合理：如果首次 ATTACH 或 report 就失败（primary 不在），异常会在 watchdog 启动前抛出，进程直接退出，不会留下一个"永远连不上"的后台线程。

## 没变的部分

- 首次 ATTACH 与 `accepted` 响应处理（`client_count`、`frontend_args` 继承）；
- `args.api_server_count = client_count`（这里的 `client_count` 是容量，用于本地 server 的 client 配置）；
- 信号处理；
- `manager` 的创建参数（除地址来源外）与 `finally: manager.shutdown()`；
- `with FrontendAttachClient(...)` 的作用域：首次用的长超时 client 在函数结束时关闭，watchdog 之后用自己每轮新建的短超时 client。

## 归纳

`run_attach_api_server` 的修改把 attach 前端从"**一次性报名 + 临时端口**"改造成"**确定性端口 + 持续自愈**"：

- 第 1、4 点让前端在 primary 重启后能重新报名；
- 第 3 点让前端自己重启后能重新绑同一地址，使引擎侧重连成立；
- 第 2 点把端口带冲突挡在启动阶段。

四者缺一不可：只有 watchdog 没有确定性端口，重启后引擎连不上；只有确定性端口没有 watchdog，primary 重启后 barrier 永远等不到人。

---

**User**

讲解 _start_reregister_watchdog 变更

**Assistant**

## 旧版没有这个函数

旧版 attach 前端只做一次握手（[#L575-L629](file:///C:/github_main_host_rdma_old/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L575-L629)）：ATTACH → report → 等进程结束。primary 重启后没有任何机制重新报名，新 primary 的启动 barrier 只能干等。

`_start_reregister_watchdog`（[#L920-L955](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L920-L955)）是纯新增，解决的就是"primary 重启后前端不重新报名"。

## 逐段说明

**签名与生命周期**

```python
def _start_reregister_watchdog(
    plan: FrontendAttachPlan,
    input_address: str,
    output_address: str,
) -> None:
```

只接收三样东西：解析好的 plan、实际绑定成功的两个引擎 IO 地址。args 和 listen 不传，而是通过 `_resolve_watchdog_args()` / `_resolve_watchdog_listen()` 从模块级捕获值读取（[#L958-L963](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L958-L963)），目的前面讨论过：不撑大参数表、不延长函数帧、值在进程内稳定。

docstring 明确了它的定位：

> Primary-restart self-healing ... Runs until the process exits (daemon).

**循环：先 sleep 再干活**

```python
def loop() -> None:
    while True:
        time.sleep(_REREGISTER_INTERVAL_S)   # 10s
```

先睡后做是一个有意的顺序：调用方刚刚完成首次 ATTACH + report，如果立刻再跑一轮就是纯冗余。睡 10 秒后，第一次真正重报发生时，状态已经稳定。

**每轮独立短连接**

```python
with FrontendAttachClient(
    plan.endpoint, recv_timeout_s=_REREGISTER_TIMEOUT_S   # 5s
) as client:
```

不复用调用方那个 client，原因有三：

1. 调用方的 client 在 `with` 作用域结束时已 close；
2. 它的 `RCVTIMEO` 是 `VLLM_ENGINE_READY_TIMEOUT_S`（可能几分钟），primary 挂掉时会把 watchdog 线程卡死；
3. 每轮独立 context/socket 能隔离失败，上一轮的残留状态不影响下一轮。

5 秒超时保证"primary 不可达"最多占用线程 5 秒，10 秒巡检节奏不会被破坏。

**重放 ATTACH + ENDPOINTS**

```python
client.attach(
    plan.client_index,
    compute_attach_config_payload(_resolve_watchdog_args(), plan),
    _resolve_watchdog_listen(),
)
client.report_endpoints(plan.client_index, input_address, output_address)
```

两个关键点：

- **payload 用捕获的 args + 启动时的 plan 计算**，不重新 `resolve_attach_plan`。这保证每轮发出的 `lwd_frontend_digest` 与首次完全一致——primary 才能把同 pid 的轮次识别为例行重报而不是配置漂移。如果重新解析，期间 yaml 被改过就会导致 payload 变化，反而被拒绝。
- **endpoints 用启动时实际绑定的地址**，与 `engine_io_addresses` 的确定性推导一致，因此 primary 侧重报是"替换同值"，不会引发状态变化。

两条消息在同一轮内串行：attach 失败会直接抛异常，report 不会执行，符合协议顺序（未 accepted 不能报 endpoints）。

**异常吞掉，下一轮重试**

```python
except Exception:
    logger.debug(
        "LWD attach re-registration round failed (primary down?); "
        "retrying next tick",
        exc_info=True,
    )
```

这是 watchdog 最重要的防御：primary 重启期间 `_roundtrip` 会抛 `RuntimeError`（接收超时）或 ZMQ 错误，配置不一致时 primary 会回 `LWD_FRONTEND_ERROR` 也被转成异常——这些在运行期都是**预期的**，不能把线程带崩，更不能影响本进程正在服务的 HTTP 前端。

用 `debug` 级别而不是 `error`，是因为 primary 重启的几十秒里每 10 秒就会失败一次，打 error 会污染日志；真正需要排障时 `exc_info=True` 能给出完整栈。

**守护线程**

```python
threading.Thread(target=loop, name="lwd-attach-reregister", daemon=True).start()
```

daemon 表示进程退出时线程随之终止，不需要 stop event、不需要 join。代价是关闭阶段不会做"最后一轮重报"，但进程都退出了，前端本来也不该再报名。

## 设计取舍与边界

1. **不检测 primary 是否已经恢复**：每轮都盲目重试。实现简单，且 primary 存活时一轮的代价只是一次本地 ZMQ 往返，没有轮询状态的分支。
2. **plan 是启动快照**：如果运行中 yaml 改了（比如实例清单增长），前端不会感知——它本来就不需要感知，`frontend_digest` 的设计就是让清单增长不影响对账。
3. **配置漂移时 fail-closed**：如果新 primary 的 `frontend_digest` 变了（deployment/qos/scene 被改），重报会一直被拒，日志里能看到原因。这是有意的：不匹配的配置不应该被静默接纳。
4. **不覆盖首次失败**：如果启动时 primary 就不在，首次 ATTACH 会直接抛异常、进程退出，watchdog 根本不会启动。它只负责"启动成功之后 primary 再挂掉"的场景。

## 与配套改动的关系

这个函数单看只是一个重试循环，成立依赖三项配套：

| 配套 | 作用 |
| --- | --- |
| `FrontendAttachClient(recv_timeout_s=...)` | 让每轮快速失败（[#L712-L723](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L712-L723)） |
| primary 侧同 pid 静默 ACCEPT / 新 pid 重置状态 | 让重报幂等（[#L546-L572](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L546-L572)） |
| `engine_io_addresses` + `serve_background` | 让重报的地址可重连、且始终有服务端在听 |

缺任何一项，watchdog 都只会变成一个每 10 秒报错的无用线程。

---

**User**

讲解attach_scope变更

**Assistant**

Need old attach_scope and new attach_scope. New at L1052-..., I read up to 1059. Let me read both fully.

## 改动本身很小，语义是反的

旧版（[#L680-L683](file:///C:/github_main_host_rdma_old/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L680-L683)）：

```python
def attach_scope(registration_server):
    if registration_server is None:
        return contextlib.nullcontext()
    return registration_server          # 把 server 本身当上下文管理器
```

新版（[#L1052-L1062](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L1052-L1062)）：

```python
def attach_scope(registration_server):
    """...注册服务器必须活过启动 with 块..."""
    del registration_server
    return contextlib.nullcontext()     # 永远不关 server
```

`FrontendAttachServer` 实现了 `__enter__` / `__exit__`，所以旧版把它直接返回，意味着 `with attach_scope(server):` 在**退出时**会调用 `server.__exit__` → `close()`，关闭 ROUTER socket 并 `ctx.term()`。新版改为永远返回 `nullcontext()`，`with` 退出时什么也不做。

## 为什么必须反过来

调用点在 [serve.py#L359-L364](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/cli/serve.py#L359-L364)：

```python
with (
    launch_core_engines(vllm_config, executor_class, log_stats, addresses) as engine_launch,
    attach_scope(registration_server),
):
    ...
    if registration_server is not None:
        wait_for_attach_registrations(registration_server, addresses, local_engine_manager)
```

这个 `with` 块只覆盖**启动阶段**：引擎拉起、预播种、等前端注册。`wait_for_attach_registrations` 末尾会调用 [`serve_background()`](file:///c:/mainmain/vllm_host_rdma/vllm/entrypoints/openai/lwd/frontend_attach.py#L1046-L1049)，把 socket 的后续服务交给常驻线程。

如果沿用旧版，`with` 一退出就执行 `close()`：

1. `_closed = True`；
2. `self._socket.close(linger=0)`；
3. `self._ctx.term()`；
4. 刚启动的 `_serve_forever` 线程在 `poll` 上收到 `ZMQError`，直接 return。

结果就是启动 barrier 结束后注册端点**不再有人监听**。此后：

- watchdog 每 10 秒的重报全部超时失败；
- 重启的 attach 前端在 ATTACH 上永久等待；
- primary 自身重启后，前端也无法帮它填满 barrier。

这等于把前面几轮讨论的所有自愈能力全部废掉。docstring 记录的正是这个缺陷：

> closing there left nothing listening on the registration endpoint — an attach frontend restarted at runtime hung forever on its ATTACH reply.

新版把生命周期的分界线定清楚：**启动期用 barrier，运行期用常驻线程**（old-branch invariant: barrier for startup, resident service for runtime）。socket 的所有权从"with 块"转给 `serve_background` 的守护线程，直到进程退出才释放。

## 几个实现细节

**`del registration_server`**
参数现在确实不用了。`del` 是显式声明"我故意不碰它"，避免读者误以为漏了逻辑，也防止静态检查提示未使用参数。

**为什么保留函数而不直接从 `serve.py` 删掉 `attach_scope(...)`**
- 调用点保持原样，改动面最小；
- `with` 的并列结构（引擎启动 + 注册作用域）保持不变，可读性好；
- 生命周期约定集中在这一个函数的 docstring 里，将来若要改（例如需要优雅关闭）只改一处。

代价是这个函数现在是个"带说明的空操作"，看代码的人容易困惑——所以 docstring 写得比较重。

**资源释放改为依赖进程退出**
`close()` 仍然存在，但不再由启动作用域调用。现在的释放路径是：守护线程 daemon + 进程退出 + ZMQ context 随进程回收。如果将来出现"不退出进程但要停掉注册服务"的需求（比如热重载 primary），就必须显式调用 `registration_server.close()`，否则线程会一直活着。

## 归纳

`attach_scope` 从"**启动作用域结束时关闭注册服务器**"改成"**注册服务器与进程同生命周期**"。它不是可有可无的清理调整，而是运行时自愈链路的总开关：watchdog 重报、前端重启重连、primary 重启后重新填 barrier，全都依赖这个 socket 在启动完成后继续有人服务。

---

