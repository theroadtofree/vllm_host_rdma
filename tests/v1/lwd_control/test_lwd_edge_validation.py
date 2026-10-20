# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""边侧模式边界校验契约(lwd_edge_scheduler._lwd_validate_request)。

这里专门钉一条容易被反复误判的事:**本版本没有 ``use_beam_search`` 可查**,
但 beam search 也绝不会"静默按普通采样出结果"——它必经的形态已被现有规则拦下。

依据(本仓当前代码):
  - ``vllm/sampling_params.py`` 里 ``SamplingParams`` 的 37 个注解字段中没有任何
    beam 字段,全文也不存在 ``use_beam_search``;beam 只以 ``BeamSearchParams``
    (``sampling_params.py:1025``)存在。
  - v1 的在线 beam search 完全在入口层实现:
    ``vllm/entrypoints/generate/beam_search/online.py:60-64`` 每一步都用
    ``SamplingParams(logprobs=2 * beam_width, max_tokens=1, ...)`` 发普通请求,
    并给每束另起 request id(:83)。所以它到边侧时**必然带 logprobs≠None**,
    而 ``logprobs`` 已在拒列表里。
  - 客户端显式传 ``use_beam_search`` 的 OpenAI 请求在
    ``entrypoints/openai/chat_completion/serving.py:295`` 被换成 BeamSearchParams
    后走同一条入口层循环,不会伪装成普通采样参数。
"""

from types import SimpleNamespace

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdEdgeScheduler,
)

pytestmark = pytest.mark.cpu_test

_BEAM_WIDTH = 4


def _self_stub() -> SimpleNamespace:
    """校验函数只在多模态分支里读 vllm_config,文本路径给 None 足够。"""
    return SimpleNamespace(vllm_config=None)


def _request(sampling_params: SamplingParams) -> SimpleNamespace:
    return SimpleNamespace(
        request_id="req-001",
        prompt_embeds=None,
        mm_features=[],
        pooling_params=None,
        use_structured_output=False,
        sampling_params=sampling_params,
    )


def test_plain_text_request_passes() -> None:
    """基线:正常单路采样请求不该被模式边界拦。"""
    LwdEdgeScheduler._lwd_validate_request(
        _self_stub(), _request(SamplingParams(temperature=0.7))
    )


def test_beam_search_step_request_is_rejected() -> None:
    """beam search 的每一步都带 logprobs=2*beam_width ⇒ 当场拒,不会静默出结果。"""
    beam_step = SamplingParams(
        logprobs=2 * _BEAM_WIDTH, max_tokens=1, temperature=0.7
    )
    with pytest.raises(ValueError, match="logprobs"):
        LwdEdgeScheduler._lwd_validate_request(_self_stub(), _request(beam_step))


def test_multi_path_sampling_is_rejected() -> None:
    """n>1 同样不上 wire:多路会静默缺路,必须在入队前拒。"""
    with pytest.raises(ValueError, match="n=1"):
        LwdEdgeScheduler._lwd_validate_request(
            _self_stub(), _request(SamplingParams(n=_BEAM_WIDTH))
        )


def test_prompt_logprobs_request_is_rejected() -> None:
    with pytest.raises(ValueError, match="prompt_logprobs"):
        LwdEdgeScheduler._lwd_validate_request(
            _self_stub(), _request(SamplingParams(prompt_logprobs=1))
        )
