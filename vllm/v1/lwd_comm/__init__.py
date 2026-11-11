# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data-plane transport for LWD.

``data_comm`` holds the device-agnostic host-host transport (currently UCX over
host RoCE). Device copies (D2H/H2D) belong to the caller: the NPU assembly lives
in vllm-ascend (``lwd_data_plane/host``).
"""
