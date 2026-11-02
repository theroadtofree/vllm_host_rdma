# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LWD model-family helpers.

Derive from the model config what the edge computes on behalf of the cloud,
so that only transformed values, never token ids, cross the edge-cloud link.
"""
