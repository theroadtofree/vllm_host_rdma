# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""云侧 HELLO 语义采纳的记账键契约(lwd_edge_engine._lwd_adopt_cloud_semantics)。

这里钉的是多云(全互联)下一处**不报错的失效**:每台云都发 HELLO,若对齐
网格与前缀摘要粒度统一写到默认云名下,后到者覆盖前者 —— 其余云在读者
(``_lwd_hash_bs_by_cloud.get(绑定云)``)眼里恒为空,首笔 probe 便按边本地
粒度建链落在错域,那批块对后续请求永久不可命中。真机 2E1C 下默认云就是
唯一云,所以这条只能靠用例守住,现场看不出来。
"""

from types import SimpleNamespace

import pytest

from vllm.v1.lwd_control.control_communication.lwd_notify import LwdHelloNotify
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine import (
    LwdEdgeEngineCore,
)

pytestmark = pytest.mark.cpu_test

_HASH_BS = {0: 128, 1: 768}
_ALIGNMENT = {0: 512, 1: 256}


def _hello(cloud_id: int) -> LwdHelloNotify:
    """只给采纳路径要读的字段;三元组留 0 = 跳过互校,非本用例关注点。"""
    return LwdHelloNotify(
        pre_out_host="10.1.0.1",
        pre_out_port=5700,
        cloud_id=cloud_id,
        prefix_alignment_tokens=_ALIGNMENT[cloud_id],
        prefix_hash_block_size=_HASH_BS[cloud_id],
    )


def _engine_stub() -> SimpleNamespace:
    """采纳路径只读写这几个属性,不起引擎。"""
    scheduler = SimpleNamespace(
        _lwd_spec_factor=1,
        _lwd_prefix_use_eagle=False,
        need_mamba_block_aligned_split=False,
        _lwd_align_grid_by_cloud={},
        _lwd_hash_bs_by_cloud={},
        max_num_scheduled_tokens=20480,
        max_num_running_reqs=230,
        _lwd_mixed=False,
        # 保留:改错形态(统一写默认云)会走到这里,用例才能红在断言上
        # 而不是红在 AttributeError 上。
        _lwd_default_cloud_id=lambda: 0,
    )
    return SimpleNamespace(scheduler=scheduler, vllm_config=None)


def test_envelope_cloud_id_is_the_bookkeeping_key() -> None:
    """两台云各自记账:以信封 id 为键,不得都落到默认云名下。"""
    engine = _engine_stub()
    for cloud_id in (0, 1):
        LwdEdgeEngineCore._lwd_adopt_cloud_semantics(
            engine, _hello(cloud_id), cloud_id
        )

    assert engine.scheduler._lwd_hash_bs_by_cloud == _HASH_BS
    assert engine.scheduler._lwd_align_grid_by_cloud == _ALIGNMENT


def test_payload_cloud_id_used_when_envelope_id_absent() -> None:
    """1E1C 发现路径不传信封 id:回退载荷 cloud_id,单云行为不变。"""
    engine = _engine_stub()

    LwdEdgeEngineCore._lwd_adopt_cloud_semantics(engine, _hello(0))

    assert engine.scheduler._lwd_hash_bs_by_cloud == {0: _HASH_BS[0]}
    assert engine.scheduler._lwd_align_grid_by_cloud == {0: _ALIGNMENT[0]}
