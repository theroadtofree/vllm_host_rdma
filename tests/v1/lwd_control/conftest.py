# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""lwd_control 单测装载垫片。

真实环境(Python >=3.10 且 torch 在)一切原生 import;无 torch 的
轻量环境(如本机 3.9)以 stub 顶替 vllm/__init__(其 env_override
依赖 torch)、vllm/config/__init__ 与 vllm/utils/__init__(后两者
经子模块拉 torch),使纯逻辑模块(picker/ledger/dp_account、包入
口、公共面)与 lwd_comm 传输/控制面在轻量环境可测。依赖
vllm.logger/envs 完整链的用例(test_lwd_config)在此类环境模块级
跳过,见各文件头部。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VLLM_PKG_DIR = REPO_ROOT / "vllm"

if importlib.util.find_spec("torch") is None and "vllm" not in sys.modules:
    stub = types.ModuleType("vllm")
    stub.__path__ = [str(VLLM_PKG_DIR)]
    sys.modules["vllm"] = stub
    # vllm/config/__init__ 拉 torch(经各子模块);vllm.config.lwd 本身只依赖
    # pydantic+yaml。把包入口垫成命名空间,使 dp_account 等纯逻辑模块
    # (只 import vllm.config.lwd)在轻量环境可测。
    config_stub = types.ModuleType("vllm.config")
    config_stub.__path__ = [str(VLLM_PKG_DIR / "config")]
    sys.modules["vllm.config"] = config_stub
    # vllm/utils/__init__ 也拉 torch;network_utils 本身是纯 stdlib。垫成
    # 命名空间后 lwd_comm 的传输/控制面模块(只 import vllm.utils.
    # network_utils)在轻量环境可测。
    utils_stub = types.ModuleType("vllm.utils")
    utils_stub.__path__ = [str(VLLM_PKG_DIR / "utils")]
    sys.modules["vllm.utils"] = utils_stub
