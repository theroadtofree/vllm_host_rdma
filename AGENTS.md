# LWD 边云分布式推理系统

> [!NOTE]
> **本文件定位**: 团队隐形知识的载体. 团队隐性知识(代码位置、设计意图、约定、边界)必须落入本文件, 而不是留在成员脑中口口相传.
> 录入重点是隐形知识, 不是具体操作步骤 —— agent 通用具备的技能(如 git 命令用法)不记录.

LWD (Layerwise Disaggregated) 是基于 vLLM v1 引擎的边云分布式推理系统: 边端 (LwdEdge) 与云端 (LwdCloud) 以独立角色进程协同, 支持 `prefill_only` 协同模式. 控制面在 `vllm/v1/lwd_control`, 观测在 `vllm/v1/lwd_debug`, 配置入口 `vllm/config/lwd.py` (`additional_config` 的 `lwd_config`); NPU 适配依赖 vllm-ascend. 存量代码经整改后逐步合入本仓 main, 不直接沿用旧目录结构.

注: `head_tail` / `embedding_only` 协同模式已不再支持.

## 存量代码与整改

- **老代码基线**: 远端分支 `prefill_only_chunked_prefill_mm`, 相对 main 约 50 文件 / +6.8k 行; 整改后逐步合入 main, 不直接沿用旧目录结构.
- **老代码模块地图**:

  | 位置 | 职责 |
  | --- | --- |
  | `vllm/config/lwd.py` | `LwdConfig` (role/mode/层切分), 入口为 `additional_config` 的 `lwd_config` |
  | `vllm/v1/lwd_control/control_cloud_scheduler/` | 云侧引擎与 arrival 驱动调度 (`LwdCloudEngineCore`, hint/W1/W2, mixed prefill/decode) |
  | `vllm/v1/lwd_control/control_edge_scheduler/` | 边侧引擎/调度/assemble (`LwdEdgeEngineCore`, RangeNotify, UP send watermark, abort 路由) |
  | `vllm/v1/lwd_control/control_communication/` | 边云 pub/sub 控制面通信 |
  | `vllm/v1/lwd_debug/` | 观测日志 |
  | `vllm/v1/core/sched/phase_scheduler.py` | prefill_only 对照实验臂, 非 LWD 部署使用 |
  | `tools/lwd_check_budget.py`, `log_analyze_tools/lwd_*` | kernel 预算检查 / 日志分析工具 |

- **控制面通信已替换**: 老的 pub/sub `control_communication/` 已由 `vllm/v1/lwd_control/lwd_comm/` (ROUTER/DEALER) 取代, 引擎接入在 `control_scheduler/lwd_base_engine.py`; 设计见 `docs/design/lwd_control_plane.md`. `docs/refactor/lwd_control_*.md` 描述的是替换前的结构.
- **数据面与两段 PP**: 边云共享一个 HCCL world, rank 即拓扑里的全局 rank (`hccl_world_size`、`dp[].ranks`); 建组由 `vllm/distributed/lwd/` 的 `LwdParallelGroupsSpec` 从拓扑推导, ascend 侧通道与 MC2 组在 `vllm_ascend/distributed/lwd_data_plane/`; 设计见 `docs/design/lwd_data_plane.md`. 老分支借 `[边0, 云0]` 两段 PP 组表达边云 (pp=2 + `_LWD_FULL_HEAD_TAIL` 改写首/尾 rank + 各处 `not enable_lwd` 的 PP 守卫), 已去除: 各侧按 pp=1 集中式运行, 代码里不再有首/中/尾 rank 的概念, 迁移老代码时不要把这些写法带回来.
- **UP host staging 与 arrival**: `RangeNotify` 只产生 early-recv hint; D2H 完成后的 cumulative arrival 才推进 per-request W1, 云调度器以 W2 限制 native prefill 窗口. 每请求全 prompt pinned host buffer 由 ascend 侧 stager 持有到请求结束, 无总容量 cap; 固定 pending=10、early-recv/slots=2. 云侧 preemption 采用最小重算: 保留 W1, runner 在 native update 前按新 computed rewind host consumed 或对 self-hit 前跳, visible-set 将 preempted 置 waiting 队头. 全 prompt M-RoPE、DSV4 tail、abort/tombstone 及 event 顺序见 `docs/design/lwd_host_staging.md`.
- **Prefix caching/refill**: nonce/READY 协商 H, reservation 固定并 pin 快照且 fail-closed; 重准入以 arrival 对齐后的 H′ 对比 staging floor S, H′<S 时按预算切 refill 且 dispatch 时才公告, self-hit 直接前跳. hash/cache 仅覆盖 prompt; MM 仅处理 chunk 相交 feature, prefix MTP guard 要求 continued prefill 使用权威 embeddings. LWD lookup 的 `shared_prefix_boundary=0`; native Mamba fine-grained junction 机制默认关闭. 详见 `docs/design/lwd_host_staging.md`.
- **接入接缝**: 老代码经 `lwd_resolve_engine_cls` (engine 类按角色分流) 与 `lwd_serve_guard` (serve 入口注入云侧调度器) 两个钩子接入 vLLM; 另对 `v1/engine/core.py`、`v1/core/sched/scheduler.py`、`v1/worker/gpu_model_runner.py`、`distributed/parallel_state.py` 有侵入式改动.
- **已知缺口**: 老代码 docstring 引用的设计文档 `docs/refactor/prefill_only_migration.md` 分支上并不存在, 整改时需补齐.
- **迁移/替换要一次做到位**: 替换一个模块后, 全仓 grep 旧设计的名字与术语 (类名、函数名、注释里的旧概念, 如 publish/HELLO/接收线程/`wire_version`), 残留的转发层、没人覆写的空默认实现、只为旧结构存在的辅助函数、过时注释, 与替换放在同一个 commit 里清掉, 不留到以后. 例: 控制面替换后删掉了 `_lwd_shutdown_comm` (失败清理已由基类 `shutdown` 统一承担) 与转发层 `_lwd_on_message`.

## LWD 命名约定 (团队规范 7.5 的细化)

- **一个概念一个词, 跨类名/字段/方法/异常/日志/报错文案保持一致.** 例: 控制面里"对端完成握手、可以发送"统一叫 registered (`_registered_peers`, `LwdPeerNotRegisteredError`), 不混用 ready/connected/up; 只有成对的回调 `on_peer_up`/`on_peer_down` 例外.
- **复用配置框架的词汇, 不自造近义词.** `vllm/config/lwd/topology.py` 里 link = 一对实例 (`instance_links`), connection = 同序号的一对 DP (`LwdConfig.connections`); 参数名与配置字段同名 (`send_queue_max`, 不缩成 `queue_max`).
- **字段名/参数名跟着类型名走 (vLLM 惯例).** `SchedulerConfig` -> `scheduler_config`; `LwdConfig` -> `lwd_config`; 同一作用域里不让两个不同对象都叫 `config`.
- **Spec 与 Config 分工.** Config = 用户可写的配置 (YAML/CLI); Spec = 由配置推出、交给运行时组件的描述 (如 `LwdControlPlaneSpec`), 用户不直接写.
- **名字描述对象本身, 不描述当前实现的局限.** 例: 控制面参数叫 `LwdControlPlaneSpec` 而非 `LwdLinkSpec` (一边多云后一个控制面含多条链路); 值对象的 docstring 不写"远端"之类只在部分调用处成立的限定.
- **改名后全仓 grep 旧名, 设计文档和报错文案一起改.**

## 分支模型

- **正式分支仅两条, 均 protected**: `main` (集成分支) 与 `prefill_only_chunked_prefill_mm` (存量老代码基线, 只读参照). 新增正式分支需 committer 评审.
- **PR 工作分支**: `<type>/<topic>`, type ∈ feat/fix/refactor/docs/ci/test/chore. 合入后自动删除 (vllm 与 vllm-ascend 均已开启 delete_branch_on_merge).
- **调试分支**: `debug/<topic>`, 可直推远端供机器 checkout, 不提 PR. 每周四评审会过一遍分支清单, 创建者负责清理: 任务已结束 (修复已整改成 PR 进 main, 或调试放弃) 的删除; 仍在用的当场说明保留理由.
- **演示分支**: `demo/<topic>`, 短期存在; 需长期保留的在评审会说明.

> [!CAUTION]
> **团队规范区**: 下方由 `TEAM STANDARDS (BEGIN)` 与 `TEAM STANDARDS (END)` 标记圈出的区域是团队规范原文, 仅由 committer 通过评审维护.
> 在日常编码/文档任务中, **严禁修改区内任何内容**, 包括格式化、重排、翻译、增删条目.
> 若任务确实要求变更团队规范, 先停下来向用户确认, 并提示修改需经 committer 评审后才能合入.

<!-- ================ TEAM STANDARDS (BEGIN) ================ -->

## (分布式推理) 代码规范与团队规范

### 1. 基本信息

- **PR 登记表**: [PR登记表](https://onebox.huawei.com/v/5df49d10f60ca9174a522758f023c6c7?type=0&sheet=PR%E7%99%BB%E8%AE%B0%E8%A1%A8)
- **检视会议**: 每周四下午 16:00~17:00
- **会议内容**: 逐条过登记表中待评审的 PR; 有分歧的当场讨论定论; 不能到场的提前在登记表留言

### 2. 合入规范

1. 必须committer评审后再合入。
2. 提pr需要使用pr模板
   - 验证报告
   - 通过CI
   - 至少 1 名 committer Approve

### 3. Commit 风格

示例:

```text
feat(lwd): add support for xxx

feat(npu): add flash attention support for Ascend CANN
perf: avoid CPU-NPU sync in attention computation
```

常用标签: `feat` / `fix` / `perf` / `refactor` / `docs` / `test` / `ci` / `chore`

规则:

- 我们使用`(lwd)`
- ***建议让编写代码的那个agent session起草commit message***
- 用祈使句, 不限制长度, 有意放宽 conventional commits 的 50/72 字符惯例
- commit按照功能划分; 一个 commit 只做一件事
- 分散的commit要合并
- commit 语言: 英文

### 4. PR 标签

```text
[Bugfix][Lwd] fix xxx sync problem

[Bugfix][KV Offload] fix kernel budget
[Perf][ROCm][Kernel] improve MoE
```

**类型标签**:

- `[Bugfix]`、`[Feature]`、`[Perf]`、`[Refactor]`: 代码变更
- `[CI]` 或 `[CI/Build]`、`[Test]`、`[Doc]` 或 `[Docs]`: 项目基础设施、测试和文档
- `[Misc]`: 其他类型标签都不合适时

**范围标签**:

- 代码库主要区域:`[Model]`、`[Frontend]`、`[Rust Frontend]`、`[Core]`、`[Kernel]`
- 高频变更子系统: `[MRV2]`、`[Attention]`、`[Multimodal]`、`[Quantization]`、`[MoE]`、`[Spec Decode]`、`[LoRA]`、`[KV Connector]`、`[KV Offload]`

规则:

- 我们使用`[Lwd]`
- 建议使用gh工具提pr
- **紧急合入且没有跑 CI 的, 必须打 `no-test` label**

### 5. PR 模板 (可以保留中文; commit仍然要英文)

```markdown
# 方案 (复杂的需要方案架构图, 不复杂的不需要)

# 评审截图 (涉及外部的需要评审, 不涉及的不需要)

# 修改影响 (类似代码注释的功能, 建议补充, 方便回忆意图)

# 测试结果

## 命令
<!-- 运行命令 -->

## 修改前
<!-- 截图 -->

## 修改后
<!-- 截图 -->

# Checklist
- [ ] 实现全面, 功能完备 -- 需要人看!
- [ ] 避免 if-else 堆叠: 公共流程抽象为基类, 差异流程用子类继承承载 -- 需要人看!
- [ ] 模块间隔离: 模块间依赖要少, 要高内聚, 避免改动一个模块影响无关模块 -- 需要人看!
- [ ] 性能敏感路径要评估性能影响 -- 需要人看!
- [ ] 命名要清晰 -- 需要人看!

- [ ] PR描述说明要清晰, 完备
- [ ] PR要有自验证报告, 和测试截图
- [ ] vllm 和 vllm-ascend 的 PR 描述中要互相贴关联链接

- [ ] bugfix需要补充回归测试
- [ ] CI 已通过, 否则打`no-test`标签

- [ ] 避免侵入式修改, 不要使用patch
- [ ] fail-fast, 避免往下传递异常情况
- [ ] 类型标注要全面
- [ ] 函数职责要单一
- [ ] 注释不要冗余
- [ ] 公共函数要提取
- [ ] 检查魔法数字
- [ ] 没做完的地方是否有TODO注释
- [ ] 使用英文, 不要使用中文
- [ ] 测试是否覆盖失败分支
- [ ] 逻辑性比较强的有业务逻辑需要ut. 初始化等代码不需要测试.

```

### 6. RFC

大的架构修改一般都有设计文档; 这种修改要新建一个 issue(RFC) , 在issue里附上设计文档, 并说明思路:

- **触发条件**: 架构性修改超过 500 LOC(不含 kernel/data/config/test)
- RFC 内容建议: 背景与动机、方案设计(含图)、备选方案对比、影响面
- 对应 PR 必须关联 RFC issue

参考:
<https://github.com/vllm-project/vllm-ascend/issues/2470>
`[RFC]: Proposal for Optimizing KVCache Transfer via Layer-wise Strategy in Disaggregation
 #2470`

### 7. 代码规范

#### 7.1 工具链

- 统一使用 uv 管理与运行
- 统一使用 ruff 格式化与 lint, 必须安装 VSCode ruff插件
- ruff 配置需要与上游 vllm 保持一致. (效果待测试)
- 提交前需要运行precommit hook
- 测试和分发使用docker

```bash
uv tool install pre-commit
pre-commit install

ruff check
ruff format
```

#### 7.2 类型标注

- 函数签名必须标注参数与返回类型
- 签名需要细化, 避免只写一个Callable, Dict, 过于宽泛的
- 有能力的可以使用@overload增加对于None等边界条件的静态检查

```python
# wrong
def register_failure_callback(self, callback: Callable):

# right
FailureCallback = Callable[[], None]
def register_failure_callback(self, callback: FailureCallback):

# @overload
@overload
def convert_tokens_to_ids(self, tokens: str) -> int: ...

@overload
def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...

def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:

```

#### 7.3 import

- import 放文件开头; 例外情况(循环依赖、可选依赖、延迟导入优化)不需要
- 注意: ruff 格式化无法将 import 放到文件开头

#### 7.4 魔法数字

- 避免魔法数字, 提取为常量
- 例外: 魔法数字比变量名更清晰时可以保留 (按照品味来)

#### 7.5 命名 / 文件组织 / 函数设计

- 遵循 Python 官方规范(PEP 8)
- 描述性命名: 名字描述**功能**, 不描述实现细节
- 保持代码好味道. 注意编码风格, 类型注释, 公共函数要提取
- 开发时要同时进行代码微重构
- 命名风格要一致: 类、函数、变量命名的前缀统一: Lwd（layerwise_disaggregated）, LwdEdge、LwdCloud.
- 类名使用lwd开头, 新增文件包在lwd路径下面
- 尽量保持非侵入式修改
- 函数行数尽量不要超100行, 函数职责要单一
- 函数、变量命名要能清晰表达其意思, 函数入参尽量控制在5个; 函数名尽量不要太复杂, 也不要太简单导致无法表达其意思

```python
# Bad
is_deepseek_v3_r1
flag1
tmp_var

# Good
supports_dynamic_temperature
uses_speculative_decoding
```

#### 7.6 注释与 TODO

- 代码能自注释的, 不要写注释.
- 特殊场景, 关键算法、绕坑逻辑必须要有解释意图的注释, 其他简单场景不要有注释 -- 必须让用户人工审查
- 未完成项必须打 TODO 注释:`# TODO(name): 描述,关联 issue #xxx`
- 注释用英文

#### 7.7 默认不打patch

- 我们主要是在自己的代码里修改.
- 例外: 昇腾对vllm打了patch, 我们改动很小, 改动的是patch的内容, 不得不在patch内部改动, 否则没有必要.

#### 7.8 主要在 vllm 改代码, 除非是 vllm-ascend 覆写的部分(比如调度器/底层接口等)

#### 7.9 使用lwd的配置文件; 尽量不要使用环境变量

- 使用lwd的配置文件
- 特殊情况, 配置里无法重载的, 经过评审后可以使用
- vllm已存在可用的环境变量时, 尽量和社区保持一致.
- 环境变量前缀统一 `VLLM_LWD_*`
- **需要合理默认值**
- **严禁在性能关键分支读取**: 初始化时读取一次并缓存, 热路径禁止反复 `os.getenv`
- 修改配置文件时要同步维护全景图. 人工维护/ai维护. 需要示例.

#### 7.10 ***模块高内聚, 用抽象降低流程复杂度***

公共流程抽象为基类, 差异流程用子类继承承载; 主流程避免 if-else 分支堆叠, 降低流程复杂度
模块间依赖要少, 要高内聚, 避免改动一个模块影响无关模块

需要补充例子.

#### 7.11 fail-fast编码, 避免将异常情况往下传递

- 提前校验, fail-fast快速失败
- 减少错误抛出运行时异常或error日志, 仅在必要时使用

典型例子:

- lwd的类和函数内部不需要再判断是否是lwd

#### 7.12 关键分支和关键点需要info日志

- 日志使用`[LWD]`
- 后续等日志模块实现后, 要使用自己的日志模块和自己的级别

#### 7.13 要有设计文档并归档到代码仓

- 使用markdown + mermaid.
- 没有markdown要贴链接. 缺少时要提醒用户补充.

### 8. 提交流程

#### 8.1 pre-commit hook

```bash
uv tool install pre-commit
pre-commit install
```

- 运行过 pre-commit 后, commit 会自动带 signoff (DCO,上游合入必需)

#### 8.2 commit 整理

- commit 按功能分割, 同一功能的分散 commit 合并, 不同功能不得混并.
- 不要将没有整理的分散的commit提成pr合入仓库.
- 可以让 AI 辅助整理
- **注意先备份分支**:`git branch backup/<name>` 或 push 到个人远端, 防止整理翻车丢代码

#### 8.3 分支

- fork 后提 PR、不要直接 push
- 建议使用gh的官方命令行工具, 可以减少人工操作

### 9. 测试(pytest)

- 逻辑性比较强的有业务逻辑需要ut. 初始化等代码不需要测试.
- ai编写的测试大部分都很冗余.
- ***ai编写的测试必须列列表让用户审查.***
- 能不能写一个skill来写符合要求的测试

- **UT**(单元测试): 快速、聚焦单个模块
- **ST**(系统测试): 集成/端到端, 需要真实 NPU 环境
- **Nightly**(夜航): 全量回归 + 长稳

要求:

- 必须覆盖失败分支(异常、边界条件), 不能只测 happy path
- 修复回归问题时, 必须补充能复现该 bug 的测试
- 性能关键代码需要独立的性能测试(附基线对比数据)

运行示例:

```bash
# Run specific unit test file
pytest -sv tests/ut/ops/test_prepare_finalize.py

# Run specific unit test
pytest -sv tests/ut/ops/test_prepare_finalize.py::test_prepare_inputs
```

### 10. CI

主要是在线ci. 本地跑test即可.

- 目标: CI 保持在 5min 以内
- CI 如何调用 claude-bot 或其他 bot 进行代码审查？
- CI 不绿不允许合入; 紧急情况打 `no-test` 并事后补跑?

急需补充的功能:

- 检查函数行数是否超过100loc
- 检查类名使用lwd开头
- 检查新增文件在lwd路径下面
- 检查函数入参数量在5个及以下, 不符合的要列出表格写回pr评论里, 让committer确认, 确认ok即可放行
- 检查是否使用了patch
- ai-bot检查Checklist
- 超过500loc检查是否关联 RFC issue和设计文档

### 11. Committer 评审

评审重点是了解代码的修改影响, 评估功能实现是否全面, 审查代码可维护性.

- committer要负责维护pr模板和checklist. checklist要保证语言精练, 可以快速确认.
- committer要负责维护pre-commit hook和ai-bot, 将可自动审查的落地成可重复运行的.
- committer要负责将团队的隐性知识落地成agent可阅读的文档.
- committer要负责维护`.github`里的自动化工具和pr的label
- 评审测试用例对于committer的负担会比较重. 可以将重心放在代码功能上

保持合理性, 不要一刀切

### 12. 维护全景图

方便定位问题, 方便新员工上手, 方便快速定位, 方便ai定位
将隐性知识落地.

skill?
ai维护?

<!-- ================ TEAM STANDARDS (END) ================ -->

---

## LWD CI

- CI 检查与 workflow 集中在 [`vllm-data-comm/ci-infra`](https://github.com/vllm-data-comm/ci-infra)（设计文档 `docs/design/lwd_ci.md`），本仓只放调用文件；改检查规则去 ci-infra 提 PR，两个仓同时生效
- 每项检查两个属性：跑在哪条线 × 必须通过/纯粹检查。必须通过（PR 标题、commit+signoff、pre-commit）卡合入；纯粹检查（入参数、函数长度、类名前缀、新文件路径、patch 使用、RFC 关联、bugfix 无测试）只写进 PR 的 checklist 评论，由 committer 评审时判断
- 本轮只接快线；慢线（LWD 单测、mypy）等 main 完成 mm 分支迁移、测试能代表代码后再接，跟踪在 [ci-infra#4](https://github.com/vllm-data-comm/ci-infra/issues/4)
- 每次 merge 上游后，重跑 `ci-infra` 仓的 `scripts/disable_inherited_workflows.sh vllm-data-comm/<repo>`，停用上游新带来的 workflow
- 本地跑检查：clone ci-infra 后 `uv run python -m lwd_checks.<module> --help`

---

## vLLM 开发工作流

> 本节译自上游 vLLM 的 AGENTS.md 并按本仓语境裁剪; merge 上游时注意核对上游变化, 勿直接用上游原文覆盖本节.

- **严禁使用系统 `python3` 或裸 `pip`/`pip install`.** 所有 Python 命令必须经 `uv` 与 `.venv/bin/python`.

### 环境搭建

```bash
# 安装 uv(如尚未安装):
curl -LsSf https://astral.sh/uv/install.sh | sh

# 统一用 uv 管理 Python 环境:
uv venv --python 3.12
source .venv/bin/activate

# 确保 pre-commit 及其 hooks 已安装:
uv pip install -r requirements/lint.txt
pre-commit install
```

### 安装依赖

```bash
# 优先用预编译产物做可编辑安装:
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# 昇腾上实际编译
# 按需设置VLLM_VERSION_OVERRIDE
cd /vllm-workspace/vllm
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ../vllm-ascend
pip install -v -e . --no-build-isolation --no-deps

```

C/C++ 或 CUDA 改动请遵循[增量编译工作流](docs/contributing/incremental_build.md).

### 测试

> 前置: 完成环境搭建与依赖安装.

```bash
# 安装测试依赖(非 x86_64 平台用 cuda.in):
uv pip install -r requirements/test/cuda.in

# 运行单个测试文件:
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

新增测试时:

- **先设计后动手.** 先回答四个问题: 模块是干什么的, I/O 契约是什么, 要防住哪个失效, 哪个层级抓住它最便宜(单测优先于集成, 集成优先于 e2e).
- **先复用后新建.** 优先扩展现有测试文件、`conftest.py` fixture 与 helpers; 附近没有合适 suite 时才新建文件.
- **测行为, 带意图.** 通过公开 API 断言可观察结果, 测试名或 docstring 要能说明动机; 跳过琐碎接线; flaky 测试比没有测试更糟.
- **保持最小.** 一个测试只测一个行为, 用能触发它的最小 setup; 测试 diff 淹没代码改动时砍范围.
- **`tests/` 下不放一次性 kernel 基准.** kernel 性能工作放 `benchmarks/kernels/`, 正确性在现有 pytest suite 里证明.
- **影响模型的变更要跑模型评估.** 用 `tests/evals/` 或 `vllm bench`, 结果附进 PR, 不要等评审人来要.

模型相关的具体要求见 [`docs/contributing/model/tests.md`](docs/contributing/model/tests.md).

### 运行 lint

> 前置: 完成环境搭建.

```bash
# 对暂存文件跑全部 pre-commit hooks:
pre-commit run

# 对所有文件:
pre-commit run --all-files

# 跑单个 hook:
pre-commit run ruff-check --all-files

# 按 CI 的方式跑 mypy:
pre-commit run mypy-3.12 --all-files --hook-stage manual
```

首次运行会联网下载并构建

外网正常时, 用上面的命令; 外网不通时, 用下面的命令:

```bash
SKIP=actionlint,update-dockerfile-graph \
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
pre-commit run --files <files>
```

`uv tool install pre-commit` 同理加 `UV_DEFAULT_INDEX`

Python 代码行宽上限 88 字符, 拿不准就交给 pre-commit 检查.

docstring 用 [Google 风格](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)(`Args:`/`Returns:`/`Raises:` 段), 不用 reStructuredText/Sphinx 字段(`:param:`、`:return:`、`:rtype:`).

### 编码风格

- 与现有代码风格保持一致
- 注释克制: 冗余注释一律删除, 优先可读的自解释代码; 必须写时保持简短直接.
- 默认读者熟悉 vLLM.
