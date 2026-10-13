# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD 临时调试日志(独立包,便于整体下线)。

接入方式:功能代码里只保留单行 ``LwdDebug.xxx(...)`` 调用(带
``# [lwd-debug]`` 注释标记);下线时删除本目录并移除带标记的调用行
即可,功能逻辑零残留。``LwdDebug.ENABLED = False`` 可整体静默。

日志标签:
  [lwd-finish-dbg]  云侧:请求准入停止参数;每步 token 进度与 finish
  [lwd-token-dbg]   边侧:每拍实际交付的 token(解码为文字)
"""

from __future__ import annotations


class LwdDebug:
    """无状态静态调试器;tokenizer 懒加载后缓存于类属性。"""

    ENABLED = True
    _tokenizer = None

    @classmethod
    def _log(cls, msg: str, *args) -> None:
        if not cls.ENABLED:
            return
        from vllm.logger import init_logger

        # debug 级:诊断日志(逐 token/逐 chunk 量级)不污染默认 info 输出;
        # 排查时提级即可,ENABLED 总开关语义不变
        init_logger(__name__).debug(msg, *args)

    # ------------------------------------------------------------------ #
    # cloud                                                              #
    # ------------------------------------------------------------------ #
    @classmethod
    def cloud_embeds_injected(cls, req_id, idx, start, n, buf) -> None:
        """UP embeds 注入点:写入窗口、首行数值(与 prepared 对照)。"""
        cls._log(
            "[lwd-input-dbg] inject req=%s idx=%s start=%s n=%s buf=%s "
            "row0[:6]=%s",
            req_id,
            idx,
            start,
            n,
            tuple(buf.shape),
            [round(v, 4) for v in buf[0, :6].tolist()] if buf.shape[0] else [],
        )

    @classmethod
    def cloud_fill_window(cls, req_id, start_pos, num_sched, window) -> None:
        """fill loop 实际拷贝点:逐请求打读自 req_prompt_embeds 的窗口摘要
        (sum/mean/head6,与 inject/边侧 SEND UP 的 DUMP 对照)。"""
        try:
            flat = window.float().reshape(-1)
            cls._log(
                "[lwd-input-dbg] fill req=%s start=%s n=%s head6=%s "
                "sum=%.4f mean=%.6f",
                req_id,
                start_pos,
                num_sched,
                [round(v, 4) for v in flat[:6].tolist()],
                float(flat.sum()),
                float(flat.mean()),
            )
        except Exception:  # noqa: BLE001
            pass

    @classmethod
    def cloud_request_admitted(cls, wire, sampling_params) -> None:
        """请求准入时打停止参数(验证 max_tokens / eos / stop_ids)。"""
        cls._log(
            "[lwd-finish-dbg] req=%s admitted: max_tokens=%s eos=%s "
            "stop_ids=%s min_tokens=%s ignore_eos=%s",
            wire.request_id,
            sampling_params.max_tokens,
            sampling_params.eos_token_id,
            sampling_params.stop_token_ids,
            sampling_params.min_tokens,
            sampling_params.ignore_eos,
        )

    @classmethod
    def cloud_step(cls, scheduler, meta, engine_core_outputs) -> None:
        """每步打 token 进度、最后一个 token 与推导出的 finish 码。"""
        from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine import (
            LwdCloudEngineCore,
        )

        finish_reasons = LwdCloudEngineCore._lwd_c2e_finish_reasons(
            meta, engine_core_outputs
        )
        for request_id, finish in zip(meta.req_ids, finish_reasons):
            req = scheduler.requests.get(request_id)
            if req is None:
                cls._log(
                    "[lwd-finish-dbg] req=%s finish=%s (not in scheduler)",
                    request_id,
                    finish,
                )
                continue
            cls._log(
                "[lwd-finish-dbg] req=%s out_len=%d last_tok=%s finish=%s",
                request_id,
                req.num_output_tokens,
                req.output_token_ids[-1] if req.output_token_ids else None,
                finish,
            )

    # ------------------------------------------------------------------ #
    # edge                                                               #
    # ------------------------------------------------------------------ #
    @classmethod
    def edge_tokens_delivered(
        cls, request_id, sampled_token_ids, finish_reason, vllm_config
    ) -> None:
        """边侧每拍交付的 token 解码为文字(判断乱码/通顺)。"""
        text = ""
        if sampled_token_ids:
            try:
                if cls._tokenizer is None:
                    from vllm.tokenizers.registry import get_tokenizer

                    mc = vllm_config.model_config
                    cls._tokenizer = get_tokenizer(
                        mc.tokenizer, trust_remote_code=mc.trust_remote_code
                    )
                text = cls._tokenizer.decode(sampled_token_ids)
            except Exception as e:  # noqa: BLE001
                text = f"<decode failed: {e}>"
        cls._log(
            "[lwd-token-dbg] req=%s tokens=%s text=%r finish=%s",
            request_id,
            sampled_token_ids,
            text,
            finish_reason,
        )
