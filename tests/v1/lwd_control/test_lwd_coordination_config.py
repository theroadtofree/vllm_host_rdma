# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""协调配置契约:probe_timeout 单点与它的三处派生关系。

超时口径必须是"云先作答(200/503)→ 边侧放弃 → 看门狗兜线程死",所以
`probe_timeout` 只能是**一个**配置项,余量在代码里派生;三处各写一个数字
正是历史上那个 15s/60s/30s 互相矛盾的形态。
"""

import pytest

from vllm.config.lwd_coordination import LwdCoordinationConfig
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_control import (
    _CLOUD_PROBE_HEADROOM_S,
)

pytestmark = pytest.mark.cpu_test


def _cfg(**overrides) -> LwdCoordinationConfig:
    base = {
        "enabled": True,
        "control_url": "https://gateway.example.com/v1/chat/completions",
        "tenant_key_file": "/keys/tenant.bin",
    }
    base.update(overrides)
    return LwdCoordinationConfig(**base)


def _validate(cfg: LwdCoordinationConfig, is_edge: bool = True) -> None:
    cfg.validate("prefill_only", is_edge, True)


def test_probe_timeout_default_and_cloud_headroom() -> None:
    cfg = _cfg()
    assert cfg.probe_timeout == 25.0
    _validate(cfg)
    # 云侧上限由它减余量派生,必然先于边侧放弃
    assert cfg.probe_timeout - _CLOUD_PROBE_HEADROOM_S < cfg.probe_timeout


@pytest.mark.parametrize("value", [0.0, -1.0, 5.0])
def test_probe_timeout_not_above_the_headroom_is_rejected(value: float) -> None:
    """<=余量即拒:否则云上限会被 max(…,1.0) 钳到 1s,探针根本跑不完。"""
    with pytest.raises(ValueError, match="probe_timeout"):
        _validate(_cfg(probe_timeout=value))


def test_from_dict_reads_probe_timeout_and_ignores_unknown_keys() -> None:
    cfg = LwdCoordinationConfig.from_dict({
        "enabled": True,
        "probe_timeout": 12.5,
        "not_a_field": 1,
    })
    assert cfg.probe_timeout == 12.5
    assert not hasattr(cfg, "not_a_field")
    assert LwdCoordinationConfig.from_dict(None).probe_timeout == 25.0


def test_cloud_cap_floor_keeps_the_timeout_positive() -> None:
    """配置被压到极小也要给出正的上限(start_lwd_cloud_control_server 的 max)。"""
    for edge_wait in (5.5, 6.0, 30.0):
        assert max(edge_wait - _CLOUD_PROBE_HEADROOM_S, 1.0) > 0
