# LWD 多实例移植修改清单（数据面 / 控制面 / 前端）

> 以实际代码提交为准的完整盘点：老仓（`host_rdma_self_process` 分支）UCX 多实例
> 动态建链/对端重启恢复功能向本双仓基线的全部已提交移植改动，按数据面、控制面、
> 前端三个平面归组。机制设计、缺陷→修复对照、分阶段验收与移植教训见
> `lwd_peer_restart_recovery.md`；本文只回答"改了哪些文件、改了什么"。
>
> * vllm 仓范围：`b6f50d4726..HEAD`（19 个提交，23 文件，+2882/−131）
> * vllm-ascend 仓范围：`8fc5890cc..HEAD`（7 个提交，12 文件，+869/−58）
> * 两仓工作区当时均无未提交改动；老仓五场景行（云重启 / 边重启 / 边实例1重启 /
>   主命令重启 / 弹性部分启动）与下述改动逐一闭环对应。

---

## 一、数据面（UCX/UHC1 host RoCE）

### vllm 仓（传输库）

| 文件 | 修改/新增内容 |
| --- | --- |
| `v1/lwd_comm/data_comm/ucx_channel.py` (+8) | `on_peer_restart` 回调异常不再被 suppress 吞掉，改为 error 留痕（被吞的复位 = 静默悬挂，老仓"日志诚实性"教训） |

### vllm-ascend 仓

| 文件 | 修改/新增内容 |
| --- | --- |
| `distributed/lwd_data_plane/host/lwd_host_channels.py` (+175) | 每 (cloud_id, dp_idx) 共享 listener；`_register_peer_restart` per-IO 复位闭包（seqno 基线复位）；`_teardown_cloud_dp` 控制面重启判定的联动强拆；edge_share 修复（共享引擎持有全部边连接）；host 内存预算按 BLOCKS_PER_CHANNEL×(up+down) 校验；弹性部分启动——warm-up 分段窗口（首链路长窗 + 其余 60s 追赶宽限 + 子集放行，零链路仍 fail-fast） |
| `host/ucx_io.py` (+66) | `on_peer_restart` 转发；提交期有界等待 `_wait_reconnect_window`（submit_wait_ready_s，吸收重连窗口）；`force_teardown`（在途 op 全 ERROR + 立即重建）；op.error 包装为 `LwdLinkError`；超 block 帧提交期快速失败 |
| `lwd_data_plane.py` (+27) | `reset_channel_seqno`：精确 ConnectionId 键复位，MISS error 留痕 |
| `lwd_transfer_queue.py` (+29) | 惰性基线（复位后首达 seqno 即基线，复位不存在"复位到几"的选择题）+ `reset_peer_baseline` |
| `lwd_types.py` (+7) | `LwdLinkError(ConnectionError)`：链路级错误分类（瞬态）区别于应用级错误（永久） |
| `worker/lwd_cloud_model/lwd_cloud_worker.py` (+116) | `_ENDPOINT_WORKER` 单例；提前收台账复位 `drop_endpoint/connection_early_recv`（边重启后残留高水位静默拒收新边 hint 的根治）；常驻 `_lwd_early_recv_loop`（只做轻量四件事、从不 wait future）；link_error 瞬态路由；丢弃 abort 的 error 留痕 |
| `worker/lwd_edge_model/lwd_edge_model_runner.py` (+60) | 4 个传输包围点抛 `LwdTransferFailedError(connection, seqnos, req_ids)`——UP 失败从"炸引擎"变"请求级隔离" |

### 数据面测试

`tests/ut/distributed/lwd_data_plane/test_lwd_peer_restart_queue`（惰性基线/复位）、
`tests/ut/worker/test_lwd_cloud_worker_restart`（台账复位/link_error）、
`tests/ut/worker/test_lwd_edge_runner_containment`（包围点）、
`tests/ut/worker/test_lwd_link_error_routing`（错误分类路由）、
`tests/ut/distributed/lwd_data_plane/test_lwd_host_channels`（+124：listener 共享/预算/弹性 warm-up）。

---

## 二、控制面（ZMQ ROUTER/DEALER）

均为 vllm 仓。

| 文件 | 修改/新增内容 |
| --- | --- |
| `lwd_comm/lwd_control_plane.py` (+97) | `wait_until_ready` 弹性分段窗口（首接受长窗 + 60s 追赶宽限 + 子集放行；零接受仍 fail-fast；含"轮询片≠窗口到期"修复）；`_on_tick` 完成门移除——周期 HELLO 不因握手完成而停（重注册载体，云重启恢复的关键）；迟接受运行期注册 |
| `lwd_comm/lwd_messages.py` (+18) | 新增 `LwdCloudHeartbeat(boot_id)`（云→边心跳，epoch 即进程随机 id）并入消息联合 |
| `lwd_comm/lwd_restart_hooks.py`（新增 48） | `register_cloud_restart_hook` / `notify_cloud_restarted(cloud_id, dp_idx, reason)`——分层注册表（vllm 不反向 import 插件仓） |
| `lwd_exceptions.py`（新增 37） | `LwdTransferFailedError(connection, seqnos, req_ids)` |
| `control_edge_scheduler/lwd_edge_engine.py` (+296) | 活性层：三账 + `_lwd_beat`（boot_id 变化=重启判定）+ `_lwd_sweep_liveness`（静默→联动强拆+门禁冻结）+ `_lwd_gate_reason`（快照∪实时 staleness；弹性扩展：来源边无已收编连接即拒）；心跳宿主（常驻线程 WAKEUP tick，引擎线程发送）；账户先于控制面初始化；containment `_lwd_contain_transfer_failure`；发送路径未注册/未收编丢弃（防 KeyError 炸共享引擎）；弹性收编 `_lwd_adopt_cloud`（init/运行期共用：参数校验、可发送 peer、活性种子、coord client、hash 尺寸）+ 路由后收编校验 |
| `control_edge_scheduler/lwd_edge_scheduler.py` (+14) | `lwd_edge_up_batch_request_ids`：在途 UP 失败的 seqno→请求反查（containment 输入） |
| `control_cloud_scheduler/lwd_cloud_engine.py` (+150) | `_lwd_boot_id` + 心跳常驻线程 + 引擎线程按拍发送（不可达对端剪除）；`_lwd_handle_peer_registered`（边重注册→seqno 账本复位 + 调度参数补发，即迟加入边的收编入口）；`_lwd_retire_connection_requests`（link_error 瞬态隔离，退役不标永久）；done-MQ link_error 路由 |
| `config/lwd/topology.py` (+17) | `hello_interval_s=1.0` / `hello_silent_s=10.0` / `submit_wait_ready_s`（配置文件优先，对齐老仓 env 语义） |

> **后续重构（不在本清单的提交内）**：活性层已从 `lwd_edge_engine.py` 抽出为独立的
> `control_edge_scheduler/lwd_heartbeat.py`（`LwdHeartbeatMonitor`）。它独占三账与静默
> 判定、在单一事件出口发布不可变 `LwdHeartbeatVerdict` 并触发拆链；引擎只装配与订阅，
> 门禁/路由/abort 走只读接口，健康 tracker 只缓存 verdict。上表描述的是移植当时的形态。

### 控制面测试

`tests/v1/lwd_control/test_lwd_liveness`（424 行：心跳/静默/门禁/弹性未收编拒绝/路由与发送防护）、
`test_lwd_transfer_containment`、`test_lwd_cloud_engine_retire`、
`test_lwd_edge_hello_reregister`（周期 HELLO + 屏障分段 + 延迟接受回归）、
`test_lwd_control_plane_multi_link`（真 ZMQ 子集放行）、
`test_lwd_edge_cloud_adoption`（迟云收编）、`tests/config/lwd/test_topology`（+11）。

---

## 三、前端（引擎 ↔ 多 API attach）

均为 vllm 仓。

| 文件 | 修改/新增内容 |
| --- | --- |
| `entrypoints/openai/lwd/frontend_attach.py` (+268) | 注册服务常驻化（`serve_background` 守护线程；屏障管启动、常驻管运行期）；pid-diff 幂等重注册；endpoints 重上报替换；10s 重注册看门狗（主命令重启自愈）；确定性引擎 IO 端口 + `engine_io_addresses` 统一推导源；注册屏障宽限化（60s，宽限内实际地址覆盖预填）；`attach_scope`→nullcontext |
| `v1/engine/core.py`（引擎侧，+182） | READY 重发（DEALER monitor 连接事件触发，前端重启/迟加入自愈）；三态连接探针（CONNECTED 补发留痕 / DISCONNECTED / RETRIED 限频告警）；输入线程地址自证日志；输入线程死亡兜底（未捕获异常→EXECUTOR_FAILED 升级，不再静默死）；ADD 到达探针 |
| `v1/engine/core_client.py`（客户端，+182） | `ROUTER_HANDOVER` 无条件开启；客户端引擎复活（同步/异步读循环 DEAD 存活、双复活判据：正常输出帧 / 输入 socket READY 探活）；init 期重复 READY 容忍；客户端 ROUTER 连接事件 monitor；request-SENT / output-received 限频探针 |

### 前端测试

`tests/entrypoints/openai/lwd/test_frontend_attach_reregister`（257 行：重注册/常驻/attach_scope 回归/弹性宽限与地址预填）、
`tests/v1/engine/test_lwd_client_engine_revival`（154 行：复活三件套）。

---

## 四、文档

* `docs/design/lwd_peer_restart_recovery.md`（新增 362 行：机制设计、缺陷→修复对照表、分阶段验收矩阵、§7.1 四条移植教训）
* `docs/design/lwd_host_data_plane.md`（同步 8 行）

---

## 附：提交清单

### vllm 仓（`b6f50d4726..HEAD`）

```text
6cfb7ff63f fix: 修复未采用的云连接导致引擎崩溃的问题
531b2a60f7 test: 添加测试用例验证延迟接受时持续轮询
6c837381db feat: 为LWD控制平面添加弹性部分启动语义
c3710ffdbf feat(lwd): 实现弹性部分启动的前端注册机制
b3fd796a0c feat: 实现引擎重启恢复的核心逻辑与追踪日志
a8195befef fix: 处理引擎READY消息重复接收的问题
1015dd1f62 fix: 修复注册服务器在attach_scope中被意外关闭的问题
8895bef54f fix(lwd): 修复心跳在控制平面建立前到达时引发的错误
3c175a3cc3 feat: 实现LWD前端注册服务器的自修复机制
4eed9086d7 fix: 修复边缘节点重新注册时的序列号账本问题
3cd330e19e fix(lwd): 处理未注册对等节点错误，避免引擎崩溃并添加相关测试
cbe3082df9 docs: 更新 lwd 对等节点重启恢复设计文档的不变量规则说明
542a83de69 fix(lwd_control): 修复握手完成后停止发送HELLO的问题
66845a6cd2 refactor(lwd_edge_engine): 优化请求拒绝逻辑并添加回归测试
713b99fd02 fix(lwd): 修复边缘引擎的静默路由判断和心跳线程问题
b88c5057ac feat(lwd): 实现云边心跳与重启检测机制
826a7858ee feat: 添加 LWD 传输失败 containment 机制和相关测试
7ad3a236d2 feat(lwd): 添加提交等待时间配置并修复回调异常处理
5a2f1faa8e docs: 添加 LWD 对等节点重启恢复与动态建链设计文档
```

### vllm-ascend 仓（`8fc5890cc..HEAD`）

```text
ba4bef8a2 feat(lwd_data_plane): 实现弹性部分启动的主机通道就绪等待逻辑
75772150b feat(lwd): 实现云数据平面链路强制销毁功能
f5f85b650 feat(lwd): 实现链路错误分类和失败请求的隔离处理
45abf53dc feat: 实现LWD通道对等端重启恢复机制
fec16e0b8 fix: add error logs for dropped abort requests in cloud worker
27d4e8021 fix(lwd): 修复边缘共享模式下主机数据通道连接获取错误
357381c1f fix(lwd): map host comm UP connections to the TP0 endpoint rank
```

---

## 后记：实现形态重构（原生文件零改动）

前端三节（`v1/engine/core.py`、`v1/engine/core_client.py`）中的自愈改动此后迁移至
LWD 侧子类实现，两个 vllm 原生文件已还原为移植前基线：

* **引擎侧**：`vllm/v1/lwd_control/control_scheduler/lwd_base_engine.py` 的
  `LwdBaseEngineCore.process_input_sockets` 全方法重写（READY 重发、DEALER
  三态连接探针、地址自证日志、输入线程死亡兜底——线程 target 绑定 self，子类
  版本自然生效）。
* **客户端侧**：`vllm/entrypoints/openai/lwd/lwd_async_client.py` 的
  `LwdAsyncMPClient(AsyncMPClient)`（ROUTER_HANDOVER 后置开启、init 期 READY
  去重、读循环 DEAD 存活 + 双复活判据、探针与 ROUTER monitor）与工厂包装
  `apply_lwd_async_client_patch`；由 vllm-ascend `platform.py` 的 LWD 配置段
  应用（前端 worker 不 import 任何 LWD 模块，平台配置口是其唯一必经点；非
  LWD 与 DP>1 自动回退原生工厂）。
* 测试随之迁移：`tests/v1/engine/test_lwd_client_engine_revival.py` →
  `tests/entrypoints/openai/lwd/test_lwd_async_client.py`。
