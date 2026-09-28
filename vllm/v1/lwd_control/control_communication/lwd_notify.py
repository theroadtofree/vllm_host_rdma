"""Lwd 控制面通知:仅依赖 msgspec,不携带 tensor,只承载调度决策预告。
上游输出类型经本模块 re-export,其余内核文件不得直接 import vllm.v1.engine。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import msgspec

if TYPE_CHECKING:
    # 上游输出类型 re-export:step 载体的返回注解共用,避免各文件直连上游
    from vllm.v1.engine import EngineCoreOutputs  # noqa: F401


class LwdRangeItem(msgspec.Struct, gc=False):
    """批量范围预告条目(mixed 组批);offset = 本 chunk 在 prompt 中的
    起点(chunk 支持已启用,lwd_chunked_prefill_design.md),云侧按
    offset == num_computed 对账。chunk 为 min(剩余 prompt, chunk 单元)
    的自限块;cap 截断块独占一批。

    has_mrope(pdmix 多模态):逐请求标记本 chunk 是否携带 mrope
    positions 行(UP aux 帧,[n,3] int64 与主帧同 seqno)。混批可含
    文本+图像混合,批级标记不够;纯文本恒 False,零 aux 流量。"""

    request_id: str
    offset: int
    num_tokens: int
    has_mrope: bool = False


class LwdRangeNotify(msgspec.Struct, gc=False, tag=True):
    """边->云调度范围预告(PRE_OUT);offset/num_tokens 取自原生调度决策,
    重复预告按 (request_id, offset) 幂等登记(边侧队满重试天然产生重复)。

    items(mixed 组批):本批全部条目,一个批一个 seqno;空 = 旧版单请求
    语义(顶层 request_id/offset/num_tokens)。新版边侧恒填 items 且顶层
    字段 = 首条目;新云侧优先读 items。"""

    request_id: str
    offset: int
    num_tokens: int
    seqno: int
    edge_id: int = 0
    """来源边 id(云侧据此做 req_id 命名空间包装);缺省 0 单边兼容。"""
    has_mrope: bool = False
    items: list[LwdRangeItem] = []


class LwdRequestNotify(msgspec.Struct, gc=False, tag=True):
    """边->云请求预告(PRE_OUT),线上只带调度决策所需字段;block_hashes 为
    边侧算好的 prompt 满块链,供云侧前缀缓存命中,缺省空回退本地哈希。

    采样参数透传(additive,缺省 = SamplingParams 原生缺省):只带影响
    云侧 token 选择的字段(采样核/惩罚/EOS 策略/min_tokens);stop 字符串
    等 detokenizer 层参数留在边侧前端原生处理,不上 wire。"""

    request_id: str
    num_prompt_tokens: int
    max_tokens: int = 16
    cloud_id: Optional[int] = None
    """路由结果回填:边据此选数据面通道与 PRE_OUT 目标;None = 回退原
    1:1 行为(单云)。"""
    edge_id: int = 0
    """来源边 id(云侧据此做 req_id 命名空间包装);缺省 0 单边兼容。"""
    block_hashes: list[bytes] = []
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: Optional[int] = None
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    ignore_eos: bool = False
    stop_token_ids: list[int] = []
    min_tokens: int = 0
    eos_token_id: Optional[int] = None
    """结束符 id:边侧前端自 tokenizer 解析,云侧占位 prompt 无从
    得知,必须随预告透传;缺省 None = 云侧 EOS 判定落空,仅
    max_tokens 兜底(旧版边侧,additive 兼容)。"""
    prefix_hit_tokens: int = 0
    """边侧 probe 命中并 trim 后的续算边界(= 边侧 chunk 起点):云侧
    resume 不得超过此值,否则「边未裁/裁得少、云块池却命中更多」会注入
    错位。缺省 0 = 未命中/探测失败(fail-open)= 不续算,与边侧恒发
    整段一致;additive 兼容旧版边侧。"""


class LwdAbortNotify(msgspec.Struct, gc=False, tag=True):
    """边->云 abort 预告(PRE_OUT);云侧清理请求登记。"""

    request_id: str
    edge_id: int = 0
    """来源边 id(云侧据此定位被包装的 req_id);缺省 0 单边兼容。"""


class LwdFinishNotify(msgspec.Struct, gc=False, tag=True):
    """边->云收尾记账(PRE_OUT):请求在边侧终结时上报**全量块哈希链**
    (prompt + 生成段,HMAC 域;原始 prompt 与生成内容均不过网)。

    云侧据此:① 校验 prompt 前缀链未变(与 probe 预留的链比对);② 发布
    生成段块摘要(生成段哈希只有持租户密钥的边能算,云不再持密钥)。
    ``full_block_hashes`` 长度须 == (prompt_tokens + completion_tokens) //
    block_size,不符即 fail-closed(只释放与记账、不发布)。"""

    request_id: str
    prompt_tokens: int
    completion_tokens: int
    full_block_hashes: list[bytes] = []
    publish_cache: bool = True
    """False = 只做释放与记账、不发布任何块(fail-closed,如媒体身份无法
    重建);缺省 True。"""
    edge_id: int = 0
    """来源边 id(云侧据此定位被包装的 req_id);缺省 0 单边兼容。"""


class LwdHelloNotify(msgspec.Struct, gc=False, tag=True):
    """云->边发现通告(POST_OUT,首拍一次);pre_out_* 是边侧连接云端点的
    唯一事实源,pre_out_host 须为边可路由真实 IP(0.0.0.0 不可作通告值)。"""

    pre_out_host: str
    pre_out_port: int
    cloud_id: int = 0
    """多云发现:边据此建 cloud_id -> 端点本地路由表;缺省 0 单云兼容。"""


# 完成码哨兵:finish_reasons 中的"本步未终结"值
LWD_NOT_FINISHED = -1


class LwdC2eNotify(msgspec.Struct, gc=False, tag=True):
    """云->边步元数据通告(POST_OUT),先于隐藏张量到达,边侧据此预挂精确
    尺寸 recv;hidden_num_elements=0 为纯终结通告,边侧本地终结不派发 unembed。"""

    hidden_num_elements: int
    top_id_ths: list[list[int]]
    num_accepted_tokens: list[int]
    req_ids: list[str]
    finish_reasons: list[int] = []
    """逐请求完成码,与 req_ids 按位对齐:LWD_NOT_FINISHED(-1) = 本步未
    终结;否则为 FinishReason IntEnum 值(STOP=0/LENGTH=1/ABORT=2/ERROR=3/
    REPETITION=4),边侧原样透传 finish_reason(LENGTH 不再伪装成 STOP)。
    空列表 = 云侧未携带,错配即 IndexError fail-fast。"""
    down_seqno: int = -1
    """本步 DOWN 隐藏张量的通道序号,自 LwdC2eMeta.down_seqno 原样透传:
    云 worker 发送时分配(通道级单调),边侧按此值预挂配对 irecv;
    缺省 -1 = 旧版云侧未携带(msgspec 带默认字段,线上 additive 兼容)。"""
    cloud_id: int = 0
    """来源云 id:边据此选 DOWN 通道接收;缺省 0 单云兼容。"""
    usages: list[dict] = []
    """逐请求 usage 回边(与 req_ids 按位对齐;空列表 = 未携带):
    云侧结算产出的 OpenAI 形状 usage(含 prompt_tokens_details.
    cached_tokens),边侧据此把"复用省了多少 token"落到自身记账/前端。
    参考分支用探针 SSE 流末 chunk 承载;prefill_only 边侧探针是同步、
    不消费 body,故改挂步元数据通道(C2e),语义不变、零新管道。"""
    token_ids: list[list[int]] = []
    """诊断旁路载荷:云侧逐请求 accepted token ids(仅
    VLLM_ASCEND_LWD_EDGE_SKIP_SAMPLE=1 时填充),与 req_ids 按位对齐。
    空列表 = 未携带,边侧走原生 rank-replay(lm_head + top_id_ths)。"""


class LwdGenChainNotify(msgspec.Struct, gc=False, tag=True):
    """边->云生成段哈希链(PRE_OUT,按步增量上报)。

    生成段 token 的 HMAC 只有持租户密钥的边能算;云侧把这些块按**边侧链**
    键入块池,后续"含生成段前缀"的请求(多轮续写)才能命中。语义为
    **全量替换**:每次都带上「prompt + 已生成」的完整满块链,丢包由下一
    条自愈(云侧覆盖写)。仅在满块数增加时发送(每 block_size 个生成
    token 一次),流量极小。

    云侧消费:块登记窗口封顶到本链覆盖范围(`_lwd_capped_cache_tokens`),
    避免链到达前按 local_hasher 误登记(块一旦登记无法重登记)。"""

    request_id: str
    block_hashes: list[bytes] = []
    block_size: int = 0
    """边侧 manifest 块粒度(链的粒度)。云侧仅在它与本地 hash 块粒度**相等**
    (ratio=1)时采用本链:否则链索引与云内 `block_hashes`(hash 粒度)错位,
    会按错键登记块。0 = 旧版边侧未携带 → 同样不采用(保守)。"""
    edge_id: int = 0
    """来源边 id(云侧据此定位被包装的 req_id);缺省 0 单边兼容。"""


# typing.Union 而非 PEP 604 `|`:msgspec 解码器的全版本支持路径
LwdNotify = Union[  # noqa: UP007
    LwdRangeNotify,
    LwdRequestNotify,
    LwdAbortNotify,
    LwdFinishNotify,
    LwdGenChainNotify,
]
# 云->边方向(POST_OUT):HELLO 发现 + 步元数据(唯一载荷,兼结果回传
# 驱动);重同步消息在此 union 上 additive 扩展
LwdCloudNotify = Union[LwdHelloNotify, LwdC2eNotify]  # noqa: UP007
# 双向全集(ROUTER-ROUTER 单通道收两个方向的载荷,按 tag 区分):
# 边->云五类通告 + 云->边步元数据;HELLO/WELCOME 为通道层帧不进解码器。
# **本 union 是通道解码的唯一依据**,任何新增 notify 必须同时加进这里,
# 否则解码失败按坏包丢弃(实测:_lwd_dispatch 里 isinstance 永远不成立,
# 两端日志都看不到该消息)。
LwdWireNotify = Union[  # noqa: UP007
    LwdRangeNotify,
    LwdRequestNotify,
    LwdAbortNotify,
    LwdFinishNotify,
    LwdGenChainNotify,
    LwdC2eNotify,
]
# 数据面批型定义归 vllm/v1/core/sched/output.py(LwdBatch/LwdBatchType/
# LwdEmbedBatch/LwdUnembedBatch,f8182fd5 定稿),协议层不重复声明

_NOTIFY_DECODER = msgspec.msgpack.Decoder(LwdNotify)
_CLOUD_NOTIFY_DECODER = msgspec.msgpack.Decoder(LwdCloudNotify)
_WIRE_NOTIFY_DECODER = msgspec.msgpack.Decoder(LwdWireNotify)


def lwd_encode_notify(message: LwdNotify) -> bytes:
    """通知编码为传输字节(发布端唯一入口)。"""
    return msgspec.msgpack.encode(message)


def lwd_decode_notify(data: bytes) -> LwdNotify:
    """传输字节解码为通知(订阅端唯一入口);坏包抛异常,由接收线程捕获丢弃。"""
    return _NOTIFY_DECODER.decode(data)


def lwd_encode_cloud_notify(message: LwdCloudNotify) -> bytes:
    """云->边通知编码(POST_OUT 发布端唯一入口)。"""
    return msgspec.msgpack.encode(message)


def lwd_decode_cloud_notify(data: bytes) -> LwdCloudNotify:
    """云->边通知解码(POST_OUT 订阅端唯一入口);坏包抛异常由接收方丢弃。"""
    return _CLOUD_NOTIFY_DECODER.decode(data)


def lwd_decode_wire_notify(data: bytes) -> LwdWireNotify:
    """双向全集解码(ROUTER-ROUTER 通道 IO 线程唯一入口);坏包抛异常
    由通道捕获丢弃,不中断 IO 线程。"""
    return _WIRE_NOTIFY_DECODER.decode(data)
