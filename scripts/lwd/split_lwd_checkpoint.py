#!/usr/bin/env python3
"""把一个完整的 HF safetensors checkpoint 按 LWD 角色(edge/cloud)切成两个子集。

背景:边云 LWD 的每一侧只运行一部分模块(配套 model_load_update 分支里的
``LwdConfig.skips_module`` + ``lwd_skips_module()``(模型侧不构建本角色永不执行的
模块)与 ``lwd_weight_filter``(装载期丢弃占位模块的权重)),所以每侧的模型目录
只需要放自己会用到的张量。但 HF 的分片是**按张量交错打包**的,不能"整片给某一
侧",必须按张量名拆分。

实现要点(为大体量 checkpoint 优化):

* **零拷贝字节直写**:张量内容原样搬运,不做 dtype/device 转换,也不 materialize
  torch 张量。只解析 safetensors 头部(纯标准库),重算新头部,再按字节区间拷贝数据
  (同文件系统优先 ``os.copy_file_range``,否则大 buffer 读写)。
* **多进程**:``--jobs N`` 按"输出分片"并行;父进程规划分片并汇总进度,worker 之间
  没有共享状态。``--jobs 1`` 时完全不走多进程。
* **索引名自动识别**:源目录的索引可能叫 ``quant_model_weights.safetensors.index.json``
  之类的非标准名;没有索引时逐分片读头部建映射(不依赖 safetensors 包)。
* **伴随 safetensors**:``rot.safetensors`` / ``optional/quarot.safetensors`` 这类
  不是权重分片、但运行时会被**按路径直接读取**的文件(QuaRot/EAGLE3 旋转矩阵),按
  角色规则拷给对应的一侧;并且只要目标目录里存在这种文件,**即使只有一个分片也一定
  写索引** —— 否则 vLLM 会把它们当权重文件读入。
* **源目录只读校验(默认开启)**:运行前后对 ``--src`` 下所有文件做 stat 快照比对,
  一旦有任何变化立即报错退出。脚本从不写 ``--src``,也拒绝 ``--dst`` 等于/包含
  ``--src``。

用法(先 dry-run 看清单,再真拆)::

    # 只看规则会保留什么(不写文件)
    python split_lwd_checkpoint.py --src /weight/GLM-5.2-w4a8c8 --role edge --dry-run

    # 真正拆分(默认按 CPU 数小并行)
    python split_lwd_checkpoint.py --src /weight/GLM-5.2-w4a8c8 \
        --dst /weight/GLM-5.2-w4a8c8-cloud --role cloud --max-shard-gb 8 --jobs 8

规则(可用 --keep-prefix / --keep-name-regex 追加)。默认同时覆盖三种命名布局:
VL 包装(model.language_model.*)、model.* 布局(Qwen3.5/GLM-5.x 等)、以及 DeepSeek
原生布局(layers./norm./embed./head./hc_head/mtp.,无 model. 前缀)。

  edge  : head.* / lm_head.* / model.head.*        (unembed:rank-replay)
          + embed.* / embed_tokens.* / model.embed*.(embed:PP first rank)
          + hc_head*(V4 顶层无条件创建的真参数,边侧也要)
          + model.visual.*(多模态塔)
          + 按名匹配 <*>.layers.<i>.mlp|ffn.gate.tid2eid
            (DeepSeek-V4 的 Hash MoE 路由表:边侧不构建层,但会直接从
             checkpoint 分片里读这些表,漏了会报 "Missing tid2eid tables")
          (边侧只做 embed / unembed;transformer 层与 final norm 都不建)
  cloud : layers.* / model.layers.* / model.language_model.*(全部层)
          + norm.* + embed* + head.*/lm_head.* + hc_head*
          + mtp.* / model.mtp.*(开启投机解码时;MTP 若表现为层号续接的额外层,
            同样被 layers./model.layers. 覆盖)
          (云侧不建视觉塔 visual)

非权重条目按角色拷贝(--metadata auto,默认;规则支持 * 通配)。目录默认不拷,只有
规则里显式点名(例如 "optional")或 --metadata all 才整目录拷贝。边侧对外服务,
tokenizer/processor/chat_template 等文件全拷;云侧配套 vllm 的 LWD 改动(强制
skip_tokenizer_init + mm_encoder_remote),只拷 config.json、configuration.json、
量化描述(*quant*description*.json)与 chat_template.jinja;QuaRot 旋转矩阵
(rot.safetensors / optional)两侧都拷。

输出:按 --max-shard-gb 切成若干 model-XXXXX-of-YYYYY.safetensors;多于一个分片、
或目录里存在伴随 safetensors 时,写 model.safetensors.index.json。

写盘期间显示进度(百分比/已写 GiB/速率/ETA + 已完成分片数):终端上原地刷新一行,
重定向到日志文件时每 10 秒一行;用 --no-progress 关闭。进度只写 stderr,stdout
仍然只有报告本身,方便 grep 与重定向。
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import multiprocessing as mp
import os
import re
import shutil
import struct
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

# vLLM 只认这个索引名(vllm/model_executor/model_loader/default_loader.py:
# SAFE_WEIGHTS_INDEX_NAME);本地目录会用它过滤掉不在索引里的 *.safetensors。
INDEX_NAME = "model.safetensors.index.json"
# 源目录里的索引可能是任意前缀(例如 quant_model_weights.safetensors.index.json)。
INDEX_SUFFIX = ".safetensors.index.json"

# 每侧必须保留的 checkpoint 名前缀。默认同时覆盖三种常见命名布局:
#   1) VL 包装布局      model.language_model.*      (Qwen3-VL / GLM-4.1V 等)
#   2) model.* 布局     model.layers./norm./embed_tokens. (Qwen3.5 / GLM-5.x 等)
#   3) DeepSeek 原生布局 layers./norm./embed./head./hc_head/mtp.
#      (见 vllm 的 _make_deepseek_v4_weights_mapper:
#       layers.->model.layers.、embed.->model.embed.、norm.->model.norm.、
#       head.weight->lm_head.weight、hc_head->model.hc_head、mtp.->model.mtp.)
ROLE_RULES: dict[str, tuple[str, ...]] = {
    "edge": (
        # --- 文本嵌入:边侧是 PP first rank,必须自己建 ---
        "model.language_model.embed_tokens.",
        "model.embed_tokens.",
        "model.embed.",
        "embed_tokens.",
        "embed.",  # DeepSeek 原生:embed.weight -> embed_tokens.weight
        # --- 输出头:边侧 rank-replay 还原 token,必须自己建 ---
        "lm_head.",
        "model.head.",
        "head.",  # DeepSeek 原生:head.weight -> lm_head.weight
        # --- DeepSeek-V4 顶层的 hc_head 参数:模型里**无条件**创建
        # (vllm/models/deepseek_v4/*/model.py 与 vllm_ascend/models/deepseek_v4.py),
        # 边侧也必须拿到,否则报缺权重 ---
        "hc_head",
        "model.hc_head",
        # --- 视觉塔:多模态必须(边侧自己跑编码) ---
        "model.visual.",
        # 其它命名的塔(vision_tower / vision_model / mm_projector 等)用
        # --keep-prefix 追加
    ),
    "cloud": (
        # --- 全部 transformer 层 + 尾部 norm ---
        "model.language_model.",  # VL 包装布局(含 embed/layers/norm)
        "model.layers.",
        "layers.",  # DeepSeek 原生布局
        "model.norm.",
        "norm.",
        # --- 文本嵌入:decode 行需要本地嵌入 ---
        "model.embed_tokens.",
        "model.embed.",
        "embed_tokens.",
        "embed.",
        # --- 采样头 + hc_head ---
        "lm_head.",  # 云侧采样 + 算秩
        "model.head.",
        "head.",
        "hc_head",
        "model.hc_head",
        # --- MTP(若开投机解码) ---
        "mtp.",  # DeepSeek 原生布局
        "model.mtp.",
    ),
}

# DeepSeek-V4 的 Hash MoE 路由表。边侧虽然不构建任何层,但会直接从 checkpoint
# 分片里读这些表(vllm-ascend 的 worker/lwd_edge_worker.py:load_model ->
# worker/lwd_hash/lwd_hash_tables.py),因此必须随边侧一起保留,否则边侧启动会报
# "Missing tid2eid tables for Hash MoE layers [...]"。
# 云侧不需要该特例:它构建全部层,这些表就是普通参数,由 layers./model.layers. 覆盖。
_TID2EID_PATTERN = re.compile(
    r"^(?:model\.)?layers\.\d+\.(?:ffn|mlp)\.gate\.tid2eid(?:\.weight)?$"
)

# 前缀规则表达不了的"按整个张量名匹配"的保留项。
ROLE_NAME_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "edge": (_TID2EID_PATTERN,),
    "cloud": (),
}

# 已知的"伴随 safetensors":不是模型权重分片,但运行时会被按路径直接读取。
#   rot.safetensors / optional/quarot.safetensors —— QuaRot 旋转矩阵,由
#   vllm-ascend 的 patch_draft_quarot.py 从 quant_model_description.json 的
#   optional.quarot.rotation_map.global_rotation 取相对路径后 load_file() 读取
#   (只在带 EAGLE3 drafter 时用到)。有索引时"不在索引里的 *.safetensors"会被自动
#   识别为伴随文件;没有索引时用下面这些模式兜底。
COMPANION_SAFETENSORS = ("rot.safetensors", "*quarot*.safetensors", "*rotation*.safetensors")

SKIP_FILE_EXACT = {"README.md", "LICENSE", ".gitattributes", "gitattributes"}
SKIP_FILE_PREFIX = (".git", "__pycache__", INDEX_NAME)

# 每侧需要拷贝的非权重条目("auto" 策略),支持 * 通配(用 fnmatch 匹配);
# 文件与"伴随 safetensors"按规则拷;**目录默认不拷**,只有规则里显式点名某个目录
# (例如 "optional")时才整目录递归拷贝,或者用 --metadata all 全拷。
METADATA_RULES: dict[str, tuple[str, ...]] = {
    # 边侧对外服务:文件全拷;目录默认不拷,所以 QuaRot 的 optional/ 需要显式点名。
    "edge": ("*", "optional"),
    "cloud": (
        "config.json",
        "configuration.json",
        "*quant*description*.json",
        # 仅为兼容启动脚本里 "--chat-template <模型目录>/chat_template.jinja" 的写法;
        # 云侧不渲染模板(请求都在边侧),启动命令里没传 --chat-template 时可以去掉。
        "chat_template.jinja",
        # QuaRot 旋转矩阵(EAGLE3 drafter 用;drafter 在云侧)。
        "rot.safetensors",
        "optional",
    ),
}

COPY_CHUNK = 8 * 2**20  # 兜底拷贝的 buffer
PROGRESS_STEP = 32 * 2**20  # worker 每搬这么多字节汇报一次进度


class TensorInfo:
    """safetensors 头部里一个张量的元信息(offsets 相对数据区起点)。"""

    __slots__ = ("dtype", "shape", "start", "end")

    def __init__(self, dtype: str, shape, start: int, end: int) -> None:
        self.dtype = dtype
        self.shape = list(shape)
        self.start = start
        self.end = end

    @property
    def nbytes(self) -> int:
        return self.end - self.start


def read_shard_header(path: str) -> tuple[dict[str, TensorInfo], dict | None, int]:
    """解析 safetensors 头部(纯标准库,不读权重数据)。

    布局:``[8 字节小端 u64: 头长度 N][N 字节 JSON 头][数据区]``。

    Returns:
        (张量名 -> TensorInfo, ``__metadata__`` 或 None, 数据区起始的绝对偏移)
    """
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: 不是合法的 safetensors(文件太短)")
        (header_len,) = struct.unpack("<Q", raw)
        header = json.loads(handle.read(header_len))
    tensors: dict[str, TensorInfo] = {}
    metadata = header.get("__metadata__")
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        tensors[name] = TensorInfo(info["dtype"], info["shape"], start, end)
    return tensors, metadata, 8 + header_len


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", required=True, help="完整模型目录(含 *.safetensors)")
    parser.add_argument("--dst", help="输出目录(与 --src 不同;--dry-run 时可省略)")
    parser.add_argument("--role", required=True, choices=sorted(ROLE_RULES))
    parser.add_argument(
        "--max-shard-gb",
        type=float,
        default=5.0,
        help="输出分片上限(默认 5GB;越大分片越少、索引越小)",
    )
    parser.add_argument("--keep-prefix", action="append", default=[], help="追加保留前缀(可重复)")
    parser.add_argument(
        "--keep-name-regex",
        action="append",
        default=[],
        help="追加按张量名匹配的正则(可重复;前缀规则表达不了时用,如仅保留某些层内参数)",
    )
    parser.add_argument(
        "--metadata",
        choices=("auto", "all"),
        default="auto",
        help="非权重条目策略:auto=按角色;all=全拷(含目录;云侧仍跑未打补丁的 vllm 时用)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="写分片的并行进程数;0=自动(min(4, CPU 数)),1=单进程(便于排查)",
    )
    parser.add_argument(
        "--no-verify-src-readonly",
        action="store_true",
        help="关闭运行前后对 --src 的只读快照校验(默认开启)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭写盘进度显示(默认:终端原地刷新一行;重定向到文件时每 10 秒输出一行)",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印将保留/丢弃的清单,不写文件")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 源目录盘点
# ---------------------------------------------------------------------------
def find_index_file(src: str) -> str | None:
    """找源目录里的索引文件(名字可能是 quant_model_weights.safetensors.index.json 等)。"""
    if os.path.isfile(os.path.join(src, INDEX_NAME)):
        return INDEX_NAME
    candidates = sorted(
        name
        for name in os.listdir(src)
        if name.endswith(INDEX_SUFFIX) and os.path.isfile(os.path.join(src, name))
    )
    return candidates[0] if candidates else None


def list_safetensors(src: str) -> list[str]:
    return sorted(name for name in os.listdir(src) if name.endswith(".safetensors"))


def load_weight_map(src: str) -> tuple[dict[str, str], str | None]:
    """返回 (张量名 -> 分片文件名, 索引文件名或 None)。

    有索引(任意 ``*.safetensors.index.json``)时直接用;没有索引时逐分片读头部枚举,
    不依赖 safetensors 包。
    """
    index_name = find_index_file(src)
    if index_name is not None:
        with open(os.path.join(src, index_name), encoding="utf-8") as handle:
            return json.load(handle)["weight_map"], index_name
    weight_map: dict[str, str] = {}
    for name in list_safetensors(src):
        tensors, _, _ = read_shard_header(os.path.join(src, name))
        for key in tensors:
            weight_map[key] = name
    return weight_map, None


def classify_safetensors(
    src: str, weight_map: dict[str, str], index_name: str | None
) -> tuple[set[str], list[str]]:
    """把 ``*.safetensors`` 分成"权重分片"与"伴随文件"。

    Returns:
        (权重分片文件名集合, 伴随 safetensors 文件名列表)
    """
    files = list_safetensors(src)
    if index_name is not None:
        weights = set(weight_map.values())
        return weights, [name for name in files if name not in weights]
    # 没有索引:用已知模式兜底,避免把 rot.safetensors 当权重分片
    companions = [
        name
        for name in files
        if any(fnmatch.fnmatch(name, pattern) for pattern in COMPANION_SAFETENSORS)
    ]
    return {name for name in files if name not in companions}, companions


def select_keys(
    weight_map: dict[str, str],
    prefixes: tuple[str, ...],
    patterns: tuple[tuple[str, "re.Pattern[str]"], ...] = (),
) -> tuple[dict[str, list[str]], int]:
    """归组要保留的张量;返回 {规则标签: [张量名]} 与命中总数。

    先按前缀匹配,未命中的再用正则按整个张量名匹配(用于 DeepSeek-V4 的 Hash MoE
    路由表这类"藏在 layers.<i> 之下但边侧必须保留"的参数)。
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    matched = 0
    for key in weight_map:
        for prefix in prefixes:
            if key.startswith(prefix):
                grouped[prefix].append(key)
                matched += 1
                break
        else:
            for label, pattern in patterns:
                if pattern.match(key):
                    grouped[label].append(key)
                    matched += 1
                    break
    return grouped, matched


def collect_tensor_info(
    src: str, weight_files: set[str], keep: set[str], weight_map: dict[str, str]
) -> tuple[dict[str, tuple[str, TensorInfo]], dict | None, dict[str, int]]:
    """读需要的源分片头部。

    Returns:
        ({张量名: (分片, TensorInfo)}, 首个分片的 __metadata__, {分片: 数据区起始绝对偏移})

    注意:TensorInfo 里的 offsets 是相对**该分片数据区**的,搬运时要加上数据区起点;
    这里把每个分片的数据区起点一并返回,交给 plan_shards() 换算成绝对偏移。
    """
    wanted_per_shard: dict[str, list[str]] = defaultdict(list)
    # 排序保证结果可复现:keep 是 set,遍历顺序受 PYTHONHASHSEED 影响,
    # 不排序会导致同样的输入两次运行出现不同的分片打包(边界/顺序漂移)。
    for key in sorted(keep):
        wanted_per_shard[weight_map[key]].append(key)

    info: dict[str, tuple[str, TensorInfo]] = {}
    metadata: dict | None = None
    data_start: dict[str, int] = {}
    for shard in sorted(wanted_per_shard):
        if shard not in weight_files:
            raise ValueError(f"张量所属分片不在索引里: {shard}")
        tensors, shard_meta, shard_data_start = read_shard_header(os.path.join(src, shard))
        data_start[shard] = shard_data_start
        if metadata is None:
            metadata = shard_meta
        missing = [key for key in wanted_per_shard[shard] if key not in tensors]
        if missing:
            raise ValueError(f"{shard} 里找不到 {len(missing)} 个张量,例如 {missing[:3]}")
        for key in wanted_per_shard[shard]:
            info[key] = (shard, tensors[key])
    return info, metadata, data_start


def gib(num_bytes: int) -> float:
    """字节 -> GiB。"""
    return num_bytes / 2**30


def fmt_duration(seconds: float) -> str:
    """秒 -> "m:ss" / "h:mm:ss"。"""
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class Progress:
    """写盘进度显示(百分比 + 已写体积 + 速率 + ETA)。

    模式:
      bar   -- 终端:回车原地刷新一行(在 tty 上默认用它)
      lines -- 非 tty(重定向/日志):每 interval 秒输出一行,避免把日志刷满
      off   -- 不输出

    进度只写 stderr,stdout 依旧只有报告,方便 grep / 重定向。
    """

    def __init__(
        self,
        total_bytes: int,
        label: str,
        mode: str = "auto",
        stream=None,
        interval: float | None = None,
    ) -> None:
        self.stream = stream if stream is not None else sys.stderr
        if mode == "auto":
            is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
            mode = "bar" if is_tty else "lines"
        self.mode = mode
        self.total = max(int(total_bytes), 1)
        self.label = label
        self.interval = interval if interval is not None else (0.3 if mode == "bar" else 10.0)
        self.start = time.monotonic()
        self.last_update = 0.0
        self.done = 0
        self.extra = ""
        self.printed = False

    def update(self, delta_bytes: int, **extra: object) -> None:
        self.done += max(int(delta_bytes), 0)
        if extra:
            self.extra = " ".join(f"{key}={value}" for key, value in extra.items())
        if self.mode == "off":
            return
        now = time.monotonic()
        finished = self.done >= self.total
        if not finished and now - self.last_update < self.interval:
            return
        self.last_update = now
        elapsed = max(now - self.start, 1e-6)
        ratio = min(self.done / self.total, 1.0)
        speed = self.done / elapsed
        eta = (self.total - self.done) / speed if speed > 0 else 0.0
        text = (
            f"{self.label} {ratio * 100:5.1f}% "
            f"({gib(self.done):.1f}/{gib(self.total):.1f} GiB, "
            f"{gib(speed):.2f} GiB/s, ETA {fmt_duration(eta)})"
        )
        if self.extra:
            text += f" {self.extra}"
        if self.mode == "bar":
            print(f"\r{text}", end="\n" if finished else "", file=self.stream, flush=True)
        else:
            print(text, file=self.stream, flush=True)
        self.printed = True

    def close(self) -> None:
        """收尾(bar 模式下补一个换行,免得后续输出接在同一行)。"""
        if self.mode == "bar" and self.printed and self.done < self.total:
            print(file=self.stream, flush=True)
        self.printed = False


# ---------------------------------------------------------------------------
# 规划与写盘
# ---------------------------------------------------------------------------
def plan_shards(
    info: dict[str, tuple[str, TensorInfo]],
    data_start: dict[str, int],
    max_shard_bytes: int,
) -> list[dict]:
    """把要保留的张量按源分片顺序打包成若干输出分片。

    同一个源分片内的张量按头部顺序取,相邻区间在写盘时可以合并成连续拷贝。
    entry = (张量名, 源分片, dtype, shape, **源文件绝对偏移**, 字节数)。
    """
    entries_by_shard: dict[str, list[str]] = defaultdict(list)
    order: dict[str, int] = {}
    for key, (shard, _) in info.items():
        if shard not in order:
            order[shard] = len(order)
        entries_by_shard[shard].append(key)

    shards: list[dict] = []
    current: list[tuple] = []
    current_bytes = 0
    for shard in sorted(entries_by_shard, key=lambda name: order[name]):
        base = data_start[shard]
        for key in entries_by_shard[shard]:
            _, tensor = info[key]
            if current and current_bytes + tensor.nbytes > max_shard_bytes:
                shards.append({"entries": current})
                current, current_bytes = [], 0
            # 纯 tuple(不用自定义类),便于跨平台 pickle 给 worker 进程
            current.append(
                (key, shard, tensor.dtype, list(tensor.shape), base + tensor.start, tensor.nbytes)
            )
            current_bytes += tensor.nbytes
    if current:
        shards.append({"entries": current})
    return shards


def _write_all(fd: int, data: bytes, at: int) -> None:
    """从绝对偏移 at 起写完整段数据(os.write 可能部分写)。"""
    os.lseek(fd, at, os.SEEK_SET)
    view = memoryview(data)
    written = 0
    while written < len(view):
        written += os.write(fd, view[written:])


def _copy_range(src_fd: int, dst_fd: int, count: int, src_off: int, dst_off: int) -> None:
    """把源 fd 的 [src_off, src_off+count) 拷到目标 fd 的 dst_off。

    优先 ``os.copy_file_range``(同文件系统近似零拷贝);跨文件系统等场景退化为
    大 buffer 读写。两条路径都用显式偏移,不依赖 fd 的文件位置。
    """
    remaining = count
    src_pos, dst_pos = src_off, dst_off
    if hasattr(os, "copy_file_range"):
        try:
            while remaining > 0:
                copied = os.copy_file_range(src_fd, dst_fd, remaining, src_pos, dst_pos)
                if copied <= 0:
                    break
                remaining -= copied
                src_pos += copied
                dst_pos += copied
            if remaining == 0:
                return
        except OSError:
            pass  # 跨文件系统等场景:回退到普通读写
    # 兜底路径用 lseek + read/write(Windows 上没有 preadv/pwritev)
    buf = bytearray(min(COPY_CHUNK, max(count, 1)))
    view = memoryview(buf)
    remaining = count
    while remaining > 0:
        want = min(remaining, len(buf))
        os.lseek(src_fd, src_pos, os.SEEK_SET)
        got = os.read(src_fd, want)
        if not got:
            raise OSError("源文件提前结束(文件被截断?)")
        _write_all(dst_fd, bytes(got), dst_pos)
        src_pos += len(got)
        dst_pos += len(got)
        remaining -= len(got)


def write_one_shard(job: dict) -> tuple[str, int, int]:
    """worker:写一个输出分片(临时名),返回 (临时名, 含头的字节数, 张量数)。"""
    entries: list[tuple] = job["entries"]
    src_dir: str = job["src_dir"]
    temp_path: str = job["temp_path"]

    # 1) 重算头部(数据区偏移从 0 开始)
    header: dict[str, dict] = {}
    layout: list[tuple[str, int, int, int]] = []  # (源分片, 源偏移, 长度, 目标偏移)
    offset = 0
    for key, shard, dtype, shape, start, nbytes in entries:
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        layout.append((shard, start, nbytes, offset))
        offset += nbytes
    if job.get("metadata"):
        header["__metadata__"] = job["metadata"]
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)  # 数据区必须 8 字节对齐
    data_offset = 8 + len(raw)

    # 2) 写头 + 按区间拷数据(相邻区间连续,天然合并)
    dst_fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    src_fd = -1
    current_src = None
    queue = job.get("queue")
    pending_report = 0
    try:
        _write_all(dst_fd, struct.pack("<Q", len(raw)) + raw, 0)
        for shard, src_off, length, dst_off in layout:
            if shard != current_src:
                if src_fd >= 0:
                    os.close(src_fd)
                # 只读打开:源目录绝不被写入
                src_fd = os.open(os.path.join(src_dir, shard), os.O_RDONLY)
                current_src = shard
            _copy_range(src_fd, dst_fd, length, src_off, data_offset + dst_off)
            pending_report += length
            if queue is not None and pending_report >= PROGRESS_STEP:
                queue.put(pending_report)
                pending_report = 0
    except BaseException:
        if src_fd >= 0:
            os.close(src_fd)
        os.close(dst_fd)
        if os.path.isfile(temp_path):
            os.remove(temp_path)  # 半成品必须删掉,名字还带 -of-XXXXX
        raise
    if src_fd >= 0:
        os.close(src_fd)
    os.close(dst_fd)
    if queue is not None and pending_report:
        queue.put(pending_report)
    return os.path.basename(temp_path), data_offset + offset, len(entries)


def write_slices(
    src: str,
    dst: str,
    shards: list[dict],
    metadata: dict | None,
    jobs: int,
    progress: "Progress | None" = None,
) -> list[str]:
    """并行(或单进程)写出所有输出分片,全部成功后重命名为最终名字。"""
    for index, shard in enumerate(shards, start=1):
        shard["temp_path"] = os.path.join(dst, f"model-{index:05d}-of-XXXXX.safetensors")
        shard["src_dir"] = src
        shard["metadata"] = metadata

    done_count = 0
    manager = None
    queue = None
    try:
        if jobs > 1:
            try:
                # mp.Queue 只能在进程创建时继承,ProcessPoolExecutor 传参会报错;
                # 这里用 Manager 的代理队列(只用于进度上报,量很小)。
                manager = mp.Manager()
                queue = manager.Queue()
                for shard in shards:
                    shard["queue"] = queue
                with ProcessPoolExecutor(max_workers=jobs) as pool:
                    futures = {pool.submit(write_one_shard, shard): shard for shard in shards}
                    pending = set(futures)
                    while pending:
                        done, pending = wait(pending, timeout=0.3, return_when=FIRST_COMPLETED)
                        while True:
                            try:
                                delta = queue.get_nowait()
                            except Exception:
                                break
                            if progress is not None:
                                progress.update(delta, shards=done_count)
                        for future in sorted(done, key=lambda f: futures[f]["temp_path"]):
                            future.result()
                            done_count += 1
                            if progress is not None:
                                progress.update(0, shards=done_count)
            except (OSError, PermissionError, RuntimeError, ImportError) as exc:
                # 受限环境(容器/seccomp/沙箱禁止匿名管道)起不了多进程:回退单进程,
                # 语义与结果完全一致,只是慢一些。
                print(
                    f"[warn] 无法启用多进程({type(exc).__name__}: {exc});回退为单进程写盘。",
                    file=sys.stderr,
                )
                if manager is not None:
                    manager.shutdown()
                manager, queue = None, None
                done_count = 0
                for shard in shards:
                    shard.pop("queue", None)
                    write_one_shard(shard)
                    done_count += 1
                    if progress is not None:
                        progress.update(0, shards=done_count)
        else:
            for shard in shards:
                write_one_shard(shard)
                done_count += 1
                if progress is not None:
                    progress.update(0, shards=done_count)
    except BaseException:
        if progress is not None:
            progress.close()
        # 本次已写出的分片不完整,整套都不可用;清理掉,只保留 metadata 拷贝
        removed = 0
        for shard in shards:
            path = shard["temp_path"]
            if os.path.isfile(path):
                os.remove(path)
                removed += 1
        print(
            f"[clean] 写出失败,已删除本次生成的 {removed} 个分片"
            f"(目标目录只留 metadata 文件,可直接重跑)",
            file=sys.stderr,
        )
        raise
    finally:
        if manager is not None:
            manager.shutdown()

    total = len(shards)
    final_names: list[str] = []
    for index in range(1, total + 1):
        temp_name = f"model-{index:05d}-of-XXXXX.safetensors"
        final_name = f"model-{index:05d}-of-{total:05d}.safetensors"
        os.replace(os.path.join(dst, temp_name), os.path.join(dst, final_name))
        final_names.append(final_name)
    if progress is not None:
        progress.update(max(progress.total - progress.done, 0), shards=total)
        progress.close()
    return final_names


def write_index(dst: str, shards: list[dict], shard_names: list[str], companion_present: bool) -> None:
    """写 ``model.safetensors.index.json``。

    * 分片数 <= 1 且目标目录没有伴随 safetensors:删掉旧索引(避免旧索引把自建分片
      过滤掉);
    * 其它情况(含"只有一个分片但有 rot.safetensors"):**必须写索引**,否则 vLLM
      会把伴随 safetensors 当权重文件读入。
    """
    index_path = os.path.join(dst, INDEX_NAME)
    if len(shard_names) <= 1 and not companion_present:
        if os.path.isfile(index_path):
            os.remove(index_path)
        return

    weight_map: dict[str, str] = {}
    total_size = 0
    for shard, name in zip(shards, shard_names):
        total_size += os.path.getsize(os.path.join(dst, name))
        for entry in shard["entries"]:
            weight_map[entry[0]] = name
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            handle,
            indent=2,
        )
    expected = {entry[0] for shard in shards for entry in shard["entries"]}
    missing = expected - set(weight_map)
    if missing:
        raise RuntimeError(f"{len(missing)} 个张量没有写进新分片,例如 {sorted(missing)[:5]}")


def verify_output(dst: str, shard_names: list[str], info: dict[str, tuple[str, TensorInfo]]) -> int:
    """回读输出分片头部,要求键集合与每个张量的 dtype/shape/字节数都一致。"""
    seen: dict[str, tuple[str, list[int], int]] = {}
    for name in shard_names:
        tensors, _, _ = read_shard_header(os.path.join(dst, name))
        for key, tensor in tensors.items():
            seen[key] = (tensor.dtype, tensor.shape, tensor.nbytes)
    expected = {key: (t.dtype, t.shape, t.nbytes) for key, (_, t) in info.items()}
    if set(seen) != set(expected):
        print(
            f"[verify] FAIL: 计划 {len(expected)} 个张量 / 实际 {len(seen)} 个",
            file=sys.stderr,
        )
        return 4
    for key, want in expected.items():
        if seen[key] != want:
            print(f"[verify] FAIL: {key} 元信息不一致 {seen[key]} != {want}", file=sys.stderr)
            return 4
    return 0


# ---------------------------------------------------------------------------
# 非权重条目 / 源目录只读校验
# ---------------------------------------------------------------------------
def _metadata_candidates(
    src: str, weight_files: set[str], companions: list[str]
) -> tuple[list[str], list[str]]:
    """候选非权重条目 -> (文件, 目录)。

    文件 = 普通文件 + 伴随 safetensors;目录单独返回(默认不拷,规则点名才拷)。
    """
    files = list(companions)
    dirs: list[str] = []
    for name in sorted(os.listdir(src)):
        if name in weight_files or name in companions:
            continue
        if name.endswith(".safetensors") or name.startswith(SKIP_FILE_PREFIX):
            continue
        if name in SKIP_FILE_EXACT:
            continue
        (dirs if os.path.isdir(os.path.join(src, name)) else files).append(name)
    return sorted(files), dirs


def plan_metadata(
    src: str, role: str, mode: str, weight_files: set[str], companions: list[str]
) -> tuple[list[str], list[str]]:
    """返回 (要拷贝的条目, 主动跳过的条目)。

    文件(含伴随 safetensors)按角色规则(支持 ``*`` 通配)拷贝;目录默认不拷,除非规则
    里**显式点名**它(写 ``"*"`` 不算点名,免得 ``--metadata auto`` 顺手带走整个目录)
    或 mode 是 ``all``。
    """
    allowed = ("*",) if mode == "all" else METADATA_RULES[role]
    files, dirs = _metadata_candidates(src, weight_files, companions)

    def keep_file(name: str) -> bool:
        return any(fnmatch.fnmatch(name, rule) for rule in allowed)

    def keep_dir(name: str) -> bool:
        if mode == "all":
            return True
        return any(rule != "*" and fnmatch.fnmatch(name, rule) for rule in allowed)

    wanted = [name for name in files if keep_file(name)]
    wanted += [name for name in dirs if keep_dir(name)]
    skipped = [name for name in files + dirs if name not in wanted]
    return wanted, skipped


def copy_metadata(src: str, dst: str, names: list[str]) -> list[str]:
    """拷贝非权重条目;目录会整目录递归拷贝(例如 optional/)。"""
    copied = []
    for name in names:
        source_path = os.path.join(src, name)
        if os.path.isdir(source_path):
            shutil.copytree(source_path, os.path.join(dst, name), dirs_exist_ok=True)
            copied.append(name + "/")
            continue
        if not os.path.isfile(source_path):
            continue
        shutil.copy2(source_path, os.path.join(dst, name))
        copied.append(name)
    return copied


def snapshot_tree(root: str) -> dict[str, tuple[int, int, int]]:
    """(相对路径 -> (size, mtime_ns, inode)),用于证明源目录没被改动。"""
    snapshot: dict[str, tuple[int, int, int]] = {}
    for current, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(files):
            path = os.path.join(current, name)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            snapshot[os.path.relpath(path, root)] = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
    return snapshot


def main() -> int:
    args = parse_args()
    prefixes = tuple(args.keep_prefix) + ROLE_RULES[args.role]
    patterns: list[tuple[str, "re.Pattern[str]"]] = [
        (f"re:{pattern.pattern}", pattern) for pattern in ROLE_NAME_PATTERNS[args.role]
    ]
    for raw in args.keep_name_regex:
        try:
            patterns.append((f"re:{raw}", re.compile(raw)))
        except re.error as exc:
            print(f"--keep-name-regex 非法: {raw} ({exc})", file=sys.stderr)
            return 2
    patterns_tuple = tuple(patterns)

    # 相对路径按当前工作目录解析;顺带展开 ~,并消掉符号链接,
    # 这样下面"dst 不能等于/包含 src"的校验不会被软链接绕过。
    src = os.path.realpath(os.path.expanduser(args.src))
    if not os.path.isdir(src):
        print(f"源目录不存在: {src}", file=sys.stderr)
        return 2

    # 源目录只读校验(默认开启):先拍快照,收尾时比对
    guard_snapshot = None if args.no_verify_src_readonly else snapshot_tree(src)

    weight_map, index_name = load_weight_map(src)
    weight_files, companions = classify_safetensors(src, weight_map, index_name)
    grouped, matched = select_keys(weight_map, prefixes, patterns_tuple)

    total_bytes = sum(os.path.getsize(os.path.join(src, name)) for name in weight_files)
    print(f"[in ] {len(weight_map)} tensor(s), {gib(total_bytes):.2f} GiB on disk")
    print(f"[plan] source index: {index_name or '(无索引,逐分片读头部)'}")
    if companions:
        print(f"[plan] companion safetensors(非权重分片): {companions}")
    print(f"[rule] keep prefixes: {list(prefixes)}")
    if patterns_tuple:
        print(f"[rule] keep name patterns: {[label for label, _ in patterns_tuple]}")

    labels = prefixes + tuple(label for label, _ in patterns_tuple)
    width = max([40, *(len(label) for label in labels)])
    print("\n  matched tensors per rule:")
    kept = 0
    for label in labels:
        keys = grouped.get(label, [])
        if not keys:
            print(f"    {label:{width}s} 0 tensor(s)  <-- 没命中,检查命名布局/是否该保留")
            continue
        kept += len(keys)
        print(f"    {label:{width}s} {len(keys)} tensor(s)")
    print(f"\n  => keep {kept} tensor(s) of {len(weight_map)}; drop {len(weight_map) - kept}")

    dropped_groups: dict[str, int] = defaultdict(int)
    keep = {key for keys in grouped.values() for key in keys}
    for key in weight_map:
        if key not in keep:
            dropped_groups[".".join(key.split(".")[:3])] += 1
    print("\n  dropped tensors grouped by leading components (top 15):")
    for name, count in sorted(dropped_groups.items(), key=lambda item: -item[1])[:15]:
        print(f"    {name:48s} {count}")

    if matched == 0:
        print(
            "\n没有任何张量命中保留规则,请用 --keep-prefix/--keep-name-regex 指定正确规则。",
            file=sys.stderr,
        )
        return 3

    # 读头部 -> 精确体积 + 分片规划(dry-run 也做,让 [size] 是准确值)
    try:
        info, metadata, data_start = collect_tensor_info(src, weight_files, keep, weight_map)
    except (OSError, ValueError) as exc:
        print(f"\n  [size] 无法统计保留体积: {exc}", file=sys.stderr)
        return 2
    keep_size = sum(tensor.nbytes for _, tensor in info.values())
    print(
        f"\n  [size] 将写出 {gib(keep_size):.2f} GiB / 源 {gib(total_bytes):.2f} GiB "
        f"({keep_size / max(total_bytes, 1) * 100:.1f}%)"
    )

    max_shard_bytes = int(args.max_shard_gb * 2**30)
    shards = plan_shards(info, data_start, max_shard_bytes)
    print(f"  [plan] {len(shards)} output shard(s), <= {args.max_shard_gb} GiB each")

    meta_copy, meta_skip = plan_metadata(src, args.role, args.metadata, weight_files, companions)
    print(f"\n[meta] 将拷贝 {len(meta_copy)} 个非权重条目: {meta_copy}")
    if meta_skip:
        print(f"[meta] 主动跳过 {len(meta_skip)} 个: {meta_skip}")
    if args.metadata == "auto" and args.role == "cloud":
        print(
            "[meta] 云侧最小化依赖 vllm 的 LWD 改动(skip_tokenizer_init + mm_encoder_remote);"
            "若云侧仍跑未打补丁的 vllm,请加 --metadata all"
        )
    if args.dry_run:
        return 0

    if not args.dst:
        print("非 dry-run 必须给 --dst", file=sys.stderr)
        return 2
    dst = os.path.realpath(os.path.expanduser(args.dst))
    if dst == src or src.startswith(dst + os.sep):
        print("为安全起见,--dst 不能等于/包含 --src", file=sys.stderr)
        return 2

    # 空间预检:宁可在这里退出,也不要在写第 N 个分片时把盘写满、留下半成品
    if keep_size:
        probe = dst
        while not os.path.exists(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        free = shutil.disk_usage(probe).free
        need = keep_size + 256 * 2**20  # 留 256MiB 余量
        print(f"\n[space] 目标 {probe} 可用 {free / 2**30:.2f} GiB,需要 {need / 2**30:.2f} GiB")
        if free < need:
            print(
                "空间不足:换一个有空间的 --dst(另一块盘/挂载点),或先清理目标盘。",
                file=sys.stderr,
            )
            return 5

    jobs = args.jobs if args.jobs > 0 else min(4, os.cpu_count() or 1)
    os.makedirs(dst, exist_ok=True)
    copied = copy_metadata(src, dst, meta_copy)
    print(f"\n[copy] {len(copied)} metadata entrie(s): {copied}")
    print(f"[write] {len(shards)} shard(s), jobs={jobs}, 字节直写(不 materialize 张量)")

    progress = Progress(
        keep_size,
        "[write]",
        mode="off" if (args.no_progress or not keep_size) else "auto",
    )
    shard_names = write_slices(src, dst, shards, metadata, jobs, progress=progress)
    companion_in_dst = any(name in set(companions) for name in meta_copy)
    write_index(dst, shards, shard_names, companion_in_dst)

    code = verify_output(dst, shard_names, info)
    if code != 0:
        return code
    total_written = sum(os.path.getsize(os.path.join(dst, name)) for name in shard_names)
    print(f"[verify] OK: {len(info)} tensor(s), {gib(total_written):.2f} GiB in {dst}")

    if guard_snapshot is not None:
        after = snapshot_tree(src)
        if after != guard_snapshot:
            changed = sorted(set(after) ^ set(guard_snapshot))[:5]
            details = [f"{name}: {guard_snapshot.get(name)} -> {after.get(name)}" for name in changed]
            print(
                f"[guard] FAIL: 源目录 {src} 在本次运行中发生了变化(绝不允许): {details}",
                file=sys.stderr,
            )
            return 6
        print(f"[guard] 源目录只读校验通过({len(after)} 个文件快照一致)")

    print("[next ] 用 split 出来的目录启动该侧;若仍报 'Following weights were not initialized',按报错名单补张量")
    return 0


if __name__ == "__main__":
    sys.exit(main())
