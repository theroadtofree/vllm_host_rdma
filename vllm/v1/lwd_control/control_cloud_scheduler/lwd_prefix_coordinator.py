# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""云侧前缀命中协调器(HMAC 摘要域,prefill_only 裁剪版)。

从参考分支 ``edge_cloud/cloud_kv.py`` 的 ``CloudKVRequestManager`` 裁剪而来,
只保留「摘要域」的跨边前缀命中/预留/记账语义,去掉与 PD 分离耦合的
``rewrite_scheduler_output``(云改写边侧块表)与 MTP draft 段——prefill_only
边不产生块表,云侧 KV 由原生调度器独占管理。

记账口径:``probe`` 对原生块池(``find_longest_cache_hit``,
只读)求最长命中并落预留,命中值即边侧可声明的续算上界——边侧按它
trim 后随 notify 回传,云侧 resume 再被该 declared 封顶
(``kv_cache_manager.get_computed_blocks``),故边侧 chunk offset 与
云侧 computed 恒等;``complete`` 幂等回填摘要集(无块池时降级记账);
``finish`` 结算生成段摘要并产出 usage。命中块的 ref_cnt 记账归原生
(请求 admit 时 ``cache_blocks`` touch),协调器不做额外 pin。
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.lwd_control.control_communication.lwd_prefix import (
    LwdPrefixManifest,
    LwdProbeResult,
    LwdUsage,
)

logger = init_logger(__name__)


@dataclass
class LwdReservation:
    manifest: LwdPrefixManifest
    hit_tokens: int


class LwdPrefixCoordinator:
    """HMAC 摘要域的前缀缓存协调(跨边命中/预留/记账)。"""

    def __init__(
        self,
        hash_block_size: int,
        instance_id: str = "cloud-0",
        kv_cache_manager=None,
    ) -> None:
        if hash_block_size <= 0:
            raise ValueError("hash_block_size must be positive")
        self._hash_block_size = hash_block_size
        self._instance_id = instance_id
        self._completed_hashes: set[bytes] = set()
        self._reservations: dict[str, LwdReservation] = {}
        # 收尾校验用:request_id -> (manifest 块粒度, prompt 满块数, 协调链)。
        # probe 时落,边侧 LwdFinishNotify 到达时取用(校验 prompt 前缀链未变)。
        # 只由 apply_finish_chain 摘除;超上限按插入序淘汰最旧,防边侧不上报
        # (旧版边/中途 abort)时无界增长。
        self._prompt_chains: dict[str, tuple[int, int, list[bytes]]] = {}
        # 完成数交叉校验(参考 _sync_output_length 语义):云侧结算时记下自己
        # 的 (prompt, completion),边侧 LwdFinishNotify 到达时比对——不一致即
        # fail-closed(说明边云对同一条请求的记账已漂移)。
        self._expected_counts: dict[str, tuple[int, int]] = {}
        # 原生 KV 缓存管理器(用于命中块 touch 防逐出);None = 单测/降级,仅摘要域
        self._kv_cache_manager = kv_cache_manager

    _PROMPT_CHAIN_LIMIT = 4096

    def coordination_hashes(self, request_id: str) -> list[bytes] | None:
        """返回该请求预留 manifest 的协调摘要链(续算/块哈希同源);无预留 None。"""
        reservation = self._reservations.get(request_id)
        if reservation is None:
            return None
        return self._coordination_block_hashes(reservation.manifest)

    def _digest_hit(self, chain: list[bytes]) -> int:
        """摘要域最长命中块数(``complete_request`` 回填的集合)。

        仅用于无块池(单测/降级)时作命中输入;摘要域只说明「该请求算完
        并回填过」,不代表这些块此刻真在池里可复用。"""
        hit = 0
        for digest in chain:
            if digest in self._completed_hashes:
                hit += 1
            else:
                break
        return hit

    def _pool_hit(
        self, chain: list[bytes], max_cache_hit_length: int
    ) -> tuple[list, int, int]:
        """原生块池最长命中(单真源):返回 (命中 KV 块列表, 命中块数, 命中 token 数)。

        max_cache_hit_length 与云侧请求 resume(get_computed_blocks)同值
        (prompt_tokens-1,末 token 必重算以产 logits),同一
        find_longest_cache_hit、同一链;且云侧该次查找另有边侧 declared
        封顶(见 kv_cache_manager),本函数只作「边侧可声明的上界」。
        只读查找(get_cached_block 不改 ref_cnt),不产生池副作用。
        无块池(单测/降级)回退摘要域记账。
        """
        km = self._kv_cache_manager
        if km is None:
            hit = self._digest_hit(chain)
            return [], hit, hit * self._hash_block_size
        blocks_by_group, num_computed_tokens = km.coordinator.find_longest_cache_hit(
            chain, max_cache_hit_length=max(max_cache_hit_length, 0),
        )
        if num_computed_tokens <= 0:
            return [], 0, 0
        # 块池命中的 token 数按 hash_block_size 整除对齐(块池按满块记录),
        # 商即命中块数,与 LwdProbeResult 的 hit_tokens == hit_blocks*bs 一致。
        hit_blocks = num_computed_tokens // self._hash_block_size
        # 收集各组命中的 KVCacheBlock,供 probe 预约时 touch 防驱逐。
        hit_kv_blocks = [blk for group in blocks_by_group for blk in group]
        return hit_kv_blocks, hit_blocks, hit_blocks * self._hash_block_size

    def _coordination_block_hashes(
        self, manifest: LwdPrefixManifest
    ) -> list[bytes]:
        """把 manifest 粒度摘要复制扩展进云 hash 域(整除链)。

        对称部署下 manifest.block_size == hash_block_size,ratio=1,
        链 == manifest.full_block_hashes == 边侧 notify 携带的链。"""
        if manifest.block_size % self._hash_block_size != 0:
            raise ValueError(
                "manifest block size must be a multiple of the hash block "
                f"size ({manifest.block_size} % {self._hash_block_size})"
            )
        ratio = manifest.block_size // self._hash_block_size
        return [d for d in manifest.full_block_hashes for _ in range(ratio)]

    def probe(self, manifest: LwdPrefixManifest) -> LwdProbeResult:
        """对原生块池求最长命中(单真源)并落预留(等待数据面请求认领)。

        命中边界即边侧可声明的续算上界;数据面按此值 trim 后,云侧
        resume 又被该 declared 封顶(kv_cache_manager.get_computed_blocks),
        故边侧 chunk offset 与云侧 computed 恒等。预约命中块时 touch 一次
        (ref_cnt+1) 防驱逐:否则锚点请求 finish 后共享前缀块 ref_cnt 归零
        被回收,probe 命中 vs prefill 命中不一致导致 offset 错位、队首死循环。
        注:此 touch 暂不释放(泄漏),先保证正确性。
        无块池(单测/降级)回退摘要域记账。"""
        chain = self._coordination_block_hashes(manifest)
        hit_kv_blocks, hit_blocks, hit_tokens = self._pool_hit(
            chain, max_cache_hit_length=manifest.prompt_tokens - 1
        )
        if hit_kv_blocks and self._kv_cache_manager is not None:
            self._kv_cache_manager.block_pool.touch(hit_kv_blocks)
        self._reservations[manifest.request_id] = LwdReservation(
            manifest=manifest, hit_tokens=hit_tokens
        )
        # 留一份 prompt 链供收尾校验(边侧 LwdFinishNotify 到达时比对前缀未变)
        if len(self._prompt_chains) >= self._PROMPT_CHAIN_LIMIT:
            for stale in list(self._prompt_chains)[: self._PROMPT_CHAIN_LIMIT // 8]:
                self._prompt_chains.pop(stale, None)
            logger.warning(
                "[Lwd][coord] prompt-chain table full; evicted oldest entries"
            )
        self._prompt_chains[manifest.request_id] = (
            manifest.block_size,
            len(manifest.full_block_hashes),
            chain,
        )
        logger.info(
            "[Lwd][coord] probe req=%s hit_blocks=%d hit_tokens=%d",
            manifest.request_id, hit_blocks, hit_tokens,
        )
        return LwdProbeResult(
            request_id=manifest.request_id,
            instance_id=self._instance_id,
            block_size=self._hash_block_size,
            hit_blocks=hit_blocks,
            hit_tokens=hit_tokens,
        )

    def complete_request(self, request_id: str) -> None:
        """数据面请求 prompt 算完后,把其 manifest 的全部满块协调摘要
        幂等发布进 ``_completed_hashes``(无块池降级路径的命中输入;
        正常路径命中以块池单真源为准,续算闭环不依赖本回填)。

        由云引擎在步后调用(worker 执行已由本步返回隐含);无预留则跳过。
        """
        reservation = self._reservations.get(request_id)
        if reservation is None:
            return
        chain = self._coordination_block_hashes(reservation.manifest)
        before = len(self._completed_hashes)
        self._completed_hashes.update(chain)
        # 幂等回填:prompt 算完后每个 decode 步都会被调一次,只有真新增
        # 摘要时才打日志(否则长生成把日志刷满,掩盖 probe 行)
        if len(self._completed_hashes) != before:
            logger.info(
                "[Lwd][coord] complete req=%s blocks=%d new=%d",
                request_id, len(chain), len(self._completed_hashes) - before,
            )

    def claim(self, request_id: str) -> int | None:
        """数据面请求认领预留,返回命中续算起点(token 数);无预留返回 None。

        续算边界已由块池单真源闭环(probe 命中 == 块池 resume),本方法
        仅保留 admit 期可观测(见 engine)。"""
        reservation = self._reservations.get(request_id)
        return None if reservation is None else reservation.hit_tokens

    def apply_finish_chain(
        self,
        request_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        full_block_hashes: list[bytes],
        publish_cache: bool = True,
    ) -> bool:
        """边侧收尾上报:校验 prompt 前缀链未变,并按 ``publish_cache`` 发布。

        生成段哈希只有持租户密钥的边能算(云侧不再持密钥),故由边在
        ``LwdFinishNotify`` 里上报全量链;云侧只做校验与记账发布,不接触
        明文。返回 True = 校验通过(已按 publish_cache 发布摘要);False =
        fail-closed(链长/前缀链不符),调用方仍应完成释放与记账。
        """
        recorded = self._prompt_chains.pop(request_id, None)
        if recorded is None:
            logger.warning(
                "[Lwd][coord] finish chain has no recorded prompt chain req=%s "
                "(probe 未落或已淘汰); 跳前缀校验",
                request_id,
            )
            manifest_block_size, prompt_blocks, prompt_chain = 0, 0, []
        else:
            manifest_block_size, prompt_blocks, prompt_chain = recorded
        block_size = manifest_block_size or self._hash_block_size
        expected_counts = self._expected_counts.pop(request_id, None)
        if expected_counts is not None and expected_counts != (
            prompt_tokens, completion_tokens
        ):
            logger.warning(
                "[Lwd][coord] finish chain token counts drifted req=%s "
                "edge=(%d,%d) cloud=(%d,%d); fail-closed",
                request_id, prompt_tokens, completion_tokens, *expected_counts,
            )
            return False
        expected = (prompt_tokens + completion_tokens) // block_size
        if len(full_block_hashes) != expected:
            logger.warning(
                "[Lwd][coord] finish chain length mismatch req=%s got=%d "
                "expected=%d (prompt=%d completion=%d block=%d); fail-closed",
                request_id, len(full_block_hashes), expected,
                prompt_tokens, completion_tokens, block_size,
            )
            return False
        if prompt_chain:  # 非空 ⇒ prompt_blocks > 0,整除安全
            ratio = max(len(prompt_chain) // prompt_blocks, 1)
            reported = [
                digest
                for digest in full_block_hashes[:prompt_blocks]
                for _ in range(ratio)
            ]
            if reported != prompt_chain:
                logger.warning(
                    "[Lwd][coord] finish chain prompt prefix changed req=%s; "
                    "fail-closed",
                    request_id,
                )
                return False
        if publish_cache:
            self._completed_hashes.update(full_block_hashes)
        logger.info(
            "[Lwd][coord] finish-chain req=%s prompt=%d completion=%d "
            "blocks=%d publish=%s",
            request_id, prompt_tokens, completion_tokens,
            len(full_block_hashes), publish_cache,
        )
        return True

    def finish(
        self,
        request_id: str,
        generated_hashes: list[bytes],
        prompt_tokens: int,
        completion_tokens: int,
    ) -> LwdUsage:
        """结算:摘除预留、发布生成段摘要、产出 usage。

        同时记下云侧自己的 (prompt, completion) 供边侧 LwdFinishNotify 到达
        时交叉校验(apply_finish_chain);并顺带校验 prompt 长度与预留 manifest
        一致(参考 `_finish_requests` 的同款检查)。"""
        reservation = self._reservations.pop(request_id, None)
        if reservation is not None and reservation.manifest.prompt_tokens != prompt_tokens:
            logger.warning(
                "[Lwd][coord] finish prompt length drifted req=%s "
                "reservation=%d cloud=%d",
                request_id, reservation.manifest.prompt_tokens, prompt_tokens,
            )
        if len(self._expected_counts) >= 4096:
            for stale in list(self._expected_counts)[:512]:
                self._expected_counts.pop(stale, None)
        self._expected_counts[request_id] = (prompt_tokens, completion_tokens)
        cached = reservation.hit_tokens if reservation is not None else 0
        self._completed_hashes.update(generated_hashes)
        logger.info(
            "[Lwd][coord] finish req=%s cached=%d generated=%d",
            request_id, cached, len(generated_hashes),
        )
        return LwdUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached,
        )