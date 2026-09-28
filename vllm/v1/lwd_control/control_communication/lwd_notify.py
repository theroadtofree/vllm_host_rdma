"""Lwd 控制面通知:仅依赖 msgspec,不携带 tensor,只承载调度决策预告。
上游输出类型经本模块 re-export,其余内核文件不得直接 import vllm.v1.engine。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import msgspec

if TYPE_CHECKING:
    # 上游输出类型 re-export:step 载体的返回注解共用,避免各文件直连上游
    from vllm.v1.engine import EngineCoreOutputs  # noqa: F401


class LwdRangeNotify(msgspec.Struct, gc=False, tag=True):
    """边->云调度范围预告(PRE_OUT);offset/num_tokens 取自原生调度决策,
    重复预告按 (request_id, offset) 幂等登记(边侧队满重试天然产生重复)。"""

    request_id: str
    offset: int
    num_tokens: int
    seqno: int


class LwdRequestNotify(msgspec.Struct, gc=False, tag=True):
    """边->云请求预告(PRE_OUT),线上只带调度决策所需字段;block_hashes 为
    边侧算好的 prompt 满块链,供云侧前缀缓存命中,缺省空回退本地哈希。

    采样参数透传(additive,缺省 = SamplingParams 原生缺省):只带影响
    云侧 token 选择的字段(采样核/惩罚/EOS 策略/min_tokens);stop 字符串
    等 detokenizer 层参数留在边侧前端原生处理,不上 wire。"""

    request_id: str
    num_prompt_tokens: int
    max_tokens: int = 16
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


class LwdAbortNotify(msgspec.Struct, gc=False, tag=True):
    """边->云 abort 预告(PRE_OUT);云侧清理请求登记。"""

    request_id: str


class LwdHelloNotify(msgspec.Struct, gc=False, tag=True):
    """云->边发现通告(POST_OUT,首拍一次);pre_out_* 是边侧连接云端点的
    唯一事实源,pre_out_host 须为边可路由真实 IP(0.0.0.0 不可作通告值)。

    instance_id: 报告方云实例索引(边侧 N-HELLO 集合屏障按此收齐 N 个
    不同实例);缺省 0 = 旧版云侧(additive 兼容,单实例语义不变)。"""

    pre_out_host: str
    pre_out_port: int
    instance_id: int = 0


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
    instance_id: int = 0
    """报告方云实例索引(与 DEALER identity 同源):边侧据此把本通告
    分发到对应实例的 UNEMBED/数据面路径;缺省 0 = 旧版云侧(additive
    兼容,单实例语义不变)。"""
    token_ids: list[list[int]] = []
    """诊断旁路载荷:云侧逐请求 accepted token ids(仅
    VLLM_ASCEND_LWD_EDGE_SKIP_SAMPLE=1 时填充),与 req_ids 按位对齐。
    空列表 = 未携带,边侧走原生 rank-replay(lm_head + top_id_ths)。"""


# typing.Union 而非 PEP 604 `|`:msgspec 解码器的全版本支持路径
LwdNotify = Union[LwdRangeNotify, LwdRequestNotify, LwdAbortNotify]  # noqa: UP007
# 云->边方向(POST_OUT):HELLO 发现 + 步元数据(唯一载荷,兼结果回传
# 驱动);重同步消息在此 union 上 additive 扩展
LwdCloudNotify = Union[LwdHelloNotify, LwdC2eNotify]  # noqa: UP007
# 数据面批型定义归 vllm/v1/core/sched/output.py(LwdBatch/LwdBatchType/
# LwdEmbedBatch/LwdUnembedBatch,f8182fd5 定稿),协议层不重复声明

_NOTIFY_DECODER = msgspec.msgpack.Decoder(LwdNotify)
_CLOUD_NOTIFY_DECODER = msgspec.msgpack.Decoder(LwdCloudNotify)


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
