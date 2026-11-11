# F2: LWD 权重划分 (checkpoint split by role)

> RFC issue: #TBD. 上游总览: [lwd_migration_overview.md](lwd_migration_overview.md).
> 前置: [lwd_role_module_skip.md](lwd_role_module_skip.md) (归属表同源, 方案 A).

## 1. 目标

提供离线工具 `scripts/lwd/split_lwd_checkpoint.py`: 把一个完整 HF checkpoint 物理切分为 edge 与 cloud 两个自洽目录, 两侧各自只含本方运行所需权重与元数据. 作用:

1. 边侧目录小 (embed + lm_head + visual), 适配边端存储;
2. 云侧目录不含 visual/processor, 部署面收窄;
3. 与 F1 镜像: 两侧从本方目录启动时, 模型图 (裁剪后) 与目录内容恰好一致.

老代码脚本**整体迁入**, 纯 Python 标准库、零 vLLM 依赖, 无适配成本; 仅规则表按本仓决策 (D2/D3) 校订.

## 2. 为什么必须按张量切 (而不是按文件)

HF 分片 (shard) 是按张量名交错打包的: 同一个 shard 里既有 `layers.0.*` 也有 `layers.57.*`、`norm.*`、`visual.*`. 按文件拷贝无法得到角色纯净的目录, 必须以**张量名为最小单位**筛选重排.

## 3. 流水线 (迁入后保持不变)

```mermaid
flowchart LR
    S["源 checkpoint\n(只读, stat 快照 guard)"] --> P1["1 解析 index json\n(single/multi/无索引)"]
    P1 --> P2["2 汇总张量清单\n(safetensors 头手解,\nstruct.unpack '<Q')"]
    P2 --> P3["3 逐张量判归属\nROLE_RULES 正则"]
    P3 --> P4["4 逐 shard 规划\n(伴随 .safetensors 一致)"]
    P4 --> P5["5 写临时文件\n(copy_file_range 零拷贝,\n--jobs 多进程)"]
    P5 --> P6["6 写 index json\n(凡写入必重建索引)"]
    P6 --> P7["7 拷 METADATA_RULES\n命中的非权重文件"]
    P7 --> P8["8 原子 rename 落盘"]
    P8 --> E["edge 切片目录"]
    P8 --> C["cloud 切片目录"]
```

工程纪律 (老代码已具备, 迁入时保留):

- **确定性**: 全程 sorted, 两次切分逐字节一致;
- **原子性**: 临时名 + `os.replace`, 失败不留半成品;
- **只读 guard**: 记录源目录 stat 快照, 切分中源被改即中止;
- **索引陷阱**: 单 shard + 伴随 safetensors (`xxx.rot.safetensors` 之类) 的目录, 只要写入了张量就必须重写 index, 否则 loader 看到旧索引;
- 退出码 0-6 区分失败类别, 供 CI 消费.

## 4. ROLE_RULES (按本仓决策校订)

| 张量名模式 | edge | cloud | 说明 |
| --- | --- | --- | --- |
| `embed` / `embed_tokens` | 保留 | 保留 | 边嵌入入口; 云 decode 嵌新 token |
| `lm_head` | 保留 | 保留 | **rank-replay 前提**: 边 unembed 回放、云采样+秩编码各需一份 |
| `layers.*` | 丢弃 | 保留 | 与 `make_layers` LWD 分支镜像 |
| `norm.*` | 丢弃 | 保留 | D2 定案: 边裁 norm (云 forward 末 norm, 下行 hidden 已 post-norm); 证据链见 F1 §3.5 |
| `visual.*` | 保留 | 丢弃 | D3 定案: 多模态编码只在边 (F1 归属表 `visual → edge`) |
| `mtp.*` / draft | 丢弃 | 保留 | drafter 在云 |
| `tid2eid` | 保留 (正则特例) | layers 规则覆盖 | 边虽不建层, vllm-ascend 按路径直接读 (Hash MoE 路由表) |
| `hc_head_{fn,base,scale}` | 保留 (正则特例) | 保留 (正则特例) | DeepSeek-V4 hyper-connection 头: 模型上的**扁平参数** (无点边界, 归属表无法表达; 加载过滤器两侧均不跳过), 两侧都构造参数, 双侧各留一份; 仅云在 trunk forward 末尾真正执行 kernel |
| 其余 | **报错** | **报错** | fail-fast: 规则未覆盖的张量必须显式归类后才能放行 |

**镜像纪律 (方案 A 定案)**: ROLE_RULES 不再手抄——由 F1 的 `LWD_MODULE_OWNERSHIP` 表 (PR1 落 `models/utils.py`) 推导: 表内每行 (后缀, 归属集) 展开为两侧的张量名规则, 加上张量级特例 `tid2eid` (edge 保留) 与 `hc_head_{fn,base,scale}` (双角色保留) (正则, 张量名≠模块点路径, 特例显式登记). 脚本 `import vllm.model_executor.models.utils` 拿到同一对象 (允许脚本依赖 vllm 包; 无 vllm 环境的离线主机可退回 vendored 副本, 但单测断言两份等价, 禁止静默漂移).

## 5. METADATA_RULES

| 角色 | 规则 |
| --- | --- |
| cloud 最小集 | `config.json` + **tokenizer 全家桶** (`tokenizer.json`/`tokenizer_config.json`/`special_tokens_map.json`/`vocab.*`/`merges.*`) + `chat_template` 相关 |
| cloud 不拷 | processor 文件 (`preprocessor_config.json` 等) — 云不建 processor (F1 D3) |
| edge | `("*", "optional")` 全拷 — 边是对外服务前端, tokenizer/processor 都要 |

**tokenizer 全家桶进云侧切片是 F3 的硬前提**: `StructuredOutputManager` 仅在带 tokenizer 时创建编译线程池, 云侧 grammar_init 依赖它. 本规则先落 (PR3), F3 (PR4) 才能跑通.

## 6. 与部署拓扑的关系

本仓 `tools/gen_lwd_topology.py` 生成拓扑配置, 但不含切片目录概念. 部署文档需补充对应关系:

```text
topology (gen_lwd_topology.py)          checkpoint 切片 (split_lwd_checkpoint.py)
  edge 实例  ──────────────────────────  edge_dir   (embed+lm_head+visual+tid2eid+hc_head)
  cloud 实例 ──────────────────────────  cloud_dir  (layers+norm+mtp+lm_head+tokenizer+hc_head)
```

两侧启动时 `--model <本方切片目录>`, 拓扑照常指向对端.

## 7. 单测与验证

- 规则覆盖: 对 Qwen3-VL / DeepSeek-V4 风格的合成 index, 断言每个张量恰好归属一侧 (或显式报错), 输出覆盖报告;
- 确定性: 同源两次切分, 目录 diff 为空;
- 只读 guard: 切分中 touch 源目录 → 中止并退出码非 0;
- 索引陷阱: 构造单 shard + 伴随 safetensors 的目录, 断言 index 被重写且张量集正确;
- 端到端 (ST): edge 从 edge_dir、cloud 从 cloud_dir 各自启动成功; 交换目录 → 加载期失败.
