# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-to-host data transport library (vLLM repo): device-agnostic transport
implementation.

The current implementation is UCX over host RoCE: ucx_channel.py (the UHC1
protocol family, a tagged byte pipe); ucx_link.py carries the UCP link
lifecycle, ucx_ucp.py/ucx_pool.py/sock.py are shared primitives, and
ucx_port.py provides the embedded/daemon dual-mode port, with
ucx_daemon.py/ucx_shm.py covering daemon mode. Device copy (D2H/H2D)
orchestration belongs to the caller: the NPU implementation lives in
vllm_ascend (lwd_data_plane/host package), while the GPU/CPU edge sides
implement the same interface in their own plugin repos.

The data plane shutdown registry lives in ``vllm/v1/lwd_comm``.
"""
