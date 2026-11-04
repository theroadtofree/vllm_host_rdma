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

import time
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.lwd_control.lwd_comm.lwd_prefix import (
    LwdPrefixManifest,
    LwdProbeResult,
    LwdUsage,
)

logger = init_logger(__name__)


@dataclass
class LwdReservation:
    manifest: LwdPrefixManifest
    hit_tokens: int
    # 预约时 touch 命中的 KV 块(probe 防驱逐的 ref_cnt+1 记账),prefill 复用
    # (create_kv_cache_blocks 已持有)后由 complete_request 释放,防块池泄漏。
    touched_blocks: list | None = None
    # probe 时算好的协调链缓存(E5):coordination_hashes/complete_request
    # 直接复用,消除 admit 期与每个 decode 步的重复 O(块数) 重建。
    chain: list[bytes] | None = None
    # TTL 清扫维度:落预留时刻(单调钟)与是否已被数据面认领。只有
    # 「预约了但请求永不到达」(notify 丢失/边 crash)才需 sweep 兜底;
    # 已认领预留的 pin 由 complete/finish 正常释放,不在此列。
    created_at: float = 0.0
    claimed: bool = False
    # complete_request 幂等短路:首发布后置 True,后续 decode 步零开销
    completed_published: bool = False
    # probe 时原生分组命中布局的只读快照(每 KV 组一个 KVCacheBlock 元组)。
    # 准入(get_computed_blocks)直接复用这份快照,不再重查——probe 已按
    # EAGLE/MTP 尾块回退查过一次,再拿命中值当上限重查会二次回退,云侧续算
    # 起点比边侧声明少一个对齐单元,边侧已按旧命中裁剪的 embeddings 会被零
    # 向量顶替。冻结成元组:准入与分配路径可能多次消费同一份快照。
    # None = 未命中 / 无块池(降级),准入须 fail-closed。
    pool_blocks: tuple[tuple, ...] | None = None


class LwdPrefixCoordinator:
    """HMAC 摘要域的前缀缓存协调(跨边命中/预留/记账)。"""

    def __init__(
        self,
        hash_block_size: int,
        instance_id: str = "cloud-0",
        kv_cache_manager=None,
        cloud_id: int | None = None,
    ) -> None:
        if hash_block_size <= 0:
            raise ValueError("hash_block_size must be positive")
        self._hash_block_size = hash_block_size
        self._instance_id = instance_id
        # 本云 id:随 probe 回执公告给边侧(多云选路的回执通道)。None = 不
        # 公告(单边/未接选路),边侧沿用默认绑定云,行为与旧版一致。
        self._cloud_id = cloud_id
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
        # 反挂回块池管理器的准入路径:get_computed_blocks 要按 request_id 取回
        # probe 时 pin 住的分组命中布局(不得拿声明命中值当上限重查,见
        # pinned_computed_blocks)。
        if kv_cache_manager is not None:
            kv_cache_manager.lwd_prefix_coordinator = self

    _PROMPT_CHAIN_LIMIT = 4096
    # 未认领预留的 TTL:probe 落预留后数据面请求应秒级到达(notify 与
    # probe 同请求同通道);超时未到视为孤儿(notify 丢失/边 crash),
    # 由 sweep_expired 释放 pin 并摘除
    _RESERVATION_TTL_S = 60.0

    def coordination_hashes(self, request_id: str) -> list[bytes] | None:
        """返回该请求预留 manifest 的协调摘要链(续算/块哈希同源);无预留 None。"""
        reservation = self._reservations.get(request_id)
        if reservation is None:
            return None
        # E5:优先复用 probe 时缓存的链(免重建);旧预留无缓存现算补齐
        if reservation.chain is None:
            reservation.chain = self._coordination_block_hashes(reservation.manifest)
        return reservation.chain

    def has_reservation(self, request_id: str) -> bool:
        """Return whether this request has a frozen probe decision."""
        return request_id in self._reservations

    def has_pending_snapshot(self, request_id: str) -> bool:
        reservation = self._reservations.get(request_id)
        return reservation is not None and reservation.pool_blocks is not None

    def hit_of(self, request_id: str) -> int:
        reservation = self._reservations.get(request_id)
        return reservation.hit_tokens if reservation is not None else 0

    def discard(self, request_id: str) -> None:
        """Release an unused reservation without publishing usage or hashes."""
        reservation = self._reservations.pop(request_id, None)
        if (
            reservation is not None
            and reservation.touched_blocks
            and self._kv_cache_manager is not None
        ):
            self._kv_cache_manager.block_pool.free_blocks(reservation.touched_blocks)
        self._prompt_chains.pop(request_id, None)

    def reserve_hit(self, request, groups, hit_tokens: int) -> None:
        """Install one fresh readmission snapshot after native preemption."""
        old = self._reservations.get(request.request_id)
        if old is None:
            raise RuntimeError(f"missing initial LWD reservation: {request.request_id}")
        manifest = old.manifest
        prompt_chain = self._prompt_chains.get(request.request_id)
        self.discard(request.request_id)
        hit_kv_blocks = [block for group in groups for block in group]
        if hit_kv_blocks and self._kv_cache_manager is not None:
            self._kv_cache_manager.block_pool.touch(hit_kv_blocks)
        self._reservations[request.request_id] = LwdReservation(
            manifest=manifest,
            hit_tokens=hit_tokens,
            touched_blocks=hit_kv_blocks or None,
            chain=list(request.block_hashes),
            pool_blocks=tuple(tuple(group) for group in groups),
            created_at=time.monotonic(),
            claimed=True,
        )
        if prompt_chain is not None:
            self._prompt_chains[request.request_id] = prompt_chain

    def truncate(self, groups, hit_tokens: int):
        managers = self._kv_cache_manager.coordinator.single_type_managers
        return tuple(
            list(group[: hit_tokens // manager.block_size])
            for group, manager in zip(groups, managers, strict=True)
        )

    def pinned_computed_blocks(
        self, request_id: str
    ) -> tuple[tuple[tuple, ...], int] | None:
        """准入用:取回 probe 时 pin 住的原生分组命中布局快照。

        返回 (分组块布局, 命中 token 数);无预留或该次命中为 0/无块池(降级)
        返回 None,调用方须 fail-closed,不得退回重查。

        probe 已按 EAGLE/MTP 尾块回退查过一次,准入必须复用这一次的结果:拿
        命中值当上限再查一次会二次回退(每查一次退一个对齐单元,H 恰为一个
        对齐单元时直接退成 0),而边侧已按旧命中裁剪 embeddings——续算起点
        前移一个单元即用零 embeddings 重算,静默错算。
        """
        reservation = self._reservations.get(request_id)
        if reservation is None or reservation.pool_blocks is None:
            return None
        # Read-only: the snapshot is settled (not consumed) by the allocation
        # outcome via commit_allocation / rollback_allocation below. A failed
        # admission therefore never leaves a consumed snapshot with pins held.
        return reservation.pool_blocks, reservation.hit_tokens

    def commit_allocation(self, request_id: str) -> None:
        """First successful admission: release pins and retire the snapshot.

        The native KV chain now owns the blocks, so the reservation's pin is
        no longer needed. pool_blocks=None makes a later preemption go through
        reserve_hit again (no stale snapshot is ever re-allocated).
        """
        self._settle_snapshot(request_id)

    def rollback_allocation(self, request_id: str) -> None:
        """Failed admission of a preempted request: release pins and drop the
        snapshot so the next readmission re-queries a fresh hit instead of
        wedging the pool with a pin that can never be settled."""
        self._settle_snapshot(request_id)

    def _settle_snapshot(self, request_id: str) -> None:
        reservation = self._reservations.get(request_id)
        if reservation is None:
            return
        if reservation.touched_blocks and self._kv_cache_manager is not None:
            self._kv_cache_manager.block_pool.free_blocks(reservation.touched_blocks)
            reservation.touched_blocks = None
        reservation.pool_blocks = None

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
    ) -> tuple[list, tuple[list, ...] | None, int, int]:
        """原生块池最长命中(单真源):返回
        (命中 KV 块扁平列表, 分组命中布局, 命中块数, 命中 token 数)。

        max_cache_hit_length 取 prompt_tokens-1(末 token 必重算以产 logits)。
        该次查找的内部块尺寸决定了命中粒度,故命中值即「边侧可声明的上界」，
        分组布局原样透出供准入复用(见 pinned_computed_blocks)。
        只读查找(get_cached_block 不改 ref_cnt),不产生池副作用。
        无块池(单测/降级)回退摘要域记账。
        """
        km = self._kv_cache_manager
        if km is None:
            hit = self._digest_hit(chain)
            return [], None, hit, hit * self._hash_block_size
        blocks_by_group, num_computed_tokens, _ = km.coordinator.find_longest_cache_hit(
            chain, max_cache_hit_length=max(max_cache_hit_length, 0),
        )
        if num_computed_tokens <= 0:
            return [], None, 0, 0
        # 块池命中的 token 数按 hash_block_size 整除对齐(块池按满块记录),
        # 商即命中块数,与 LwdProbeResult 的 hit_tokens == hit_blocks*bs 一致。
        hit_blocks = num_computed_tokens // self._hash_block_size
        # 收集各组命中的 KVCacheBlock,供 probe 预约时 touch 防驱逐。
        hit_kv_blocks = [blk for group in blocks_by_group for blk in group]
        # 原始分组布局一并透出:准入复用它即 probe 当时的命中,免二次回退。
        return (
            hit_kv_blocks,
            blocks_by_group,
            hit_blocks,
            hit_blocks * self._hash_block_size,
        )

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
        if ratio != 1:
            raise ValueError(
                "LWD manifest block size must match the cloud hash block size "
                f"({manifest.block_size} != {self._hash_block_size})"
            )
        return list(manifest.full_block_hashes)

    def peek(self, manifest: LwdPrefixManifest) -> LwdProbeResult:
        """只读求命中(不种预留、不 touch 块池):云内 DP 扇出 probe 的
        逐引擎命中采集用。

        每引擎 KV 池/前缀缓存独立,命中值只能由本引擎自报;选定赢家后由
        Bridge 再向赢家发正式 probe(reserve)落预留——跨引擎预留是虚假
        命中,预留必须与后续数据面同引擎。_pool_hit 本身即只读
        (get_cached_block 不改 ref_cnt),此处跳过 touch/预留/链台账。"""
        chain = self._coordination_block_hashes(manifest)
        _, _, hit_blocks, hit_tokens = self._pool_hit(
            chain, max_cache_hit_length=manifest.prompt_tokens - 1
        )
        return LwdProbeResult(
            request_id=manifest.request_id,
            instance_id=self._instance_id,
            block_size=self._hash_block_size,
            hit_blocks=hit_blocks,
            hit_tokens=hit_tokens,
            alignment_tokens=self._resolve_alignment_tokens(),
        )

    def probe(self, manifest: LwdPrefixManifest) -> LwdProbeResult:
        """对原生块池求最长命中(单真源)并落预留(等待数据面请求认领)。

        命中边界即边侧可声明的续算上界;数据面按此值 trim 后,云侧
        resume 复用本次 pin 住的分组布局(不再重查,见 kv_cache_manager),
        故边侧 chunk offset 与云侧 computed 恒等。预约命中块时 touch 一次
        (ref_cnt+1) 防驱逐:否则锚点请求 finish 后共享前缀块 ref_cnt 归零
        被回收,probe 命中 vs prefill 命中不一致导致 offset 错位、队首死循环。
        该预约 touch 在 complete_request(prefill 复用后)释放,抵消 ref_cnt。
        无块池(单测/降级)回退摘要域记账。"""
        chain = self._coordination_block_hashes(manifest)
        hit_kv_blocks, pool_blocks, hit_blocks, hit_tokens = self._pool_hit(
            chain, max_cache_hit_length=manifest.prompt_tokens - 1
        )
        if hit_kv_blocks and self._kv_cache_manager is not None:
            self._kv_cache_manager.block_pool.touch(hit_kv_blocks)
        # 同键重复 probe(边侧重试):覆写前释放旧预留的 pin,防孤儿泄漏
        old = self._reservations.get(manifest.request_id)
        if (
            old is not None
            and old.touched_blocks
            and self._kv_cache_manager is not None
        ):
            logger.warning(
                "[Lwd][coord] duplicate probe req=%s: releasing stale pin",
                manifest.request_id,
            )
            self._kv_cache_manager.block_pool.free_blocks(old.touched_blocks)
        self._reservations[manifest.request_id] = LwdReservation(
            manifest=manifest,
            hit_tokens=hit_tokens,
            touched_blocks=hit_kv_blocks or None,
            chain=chain,
            pool_blocks=(
                tuple(tuple(group) for group in pool_blocks)
                if pool_blocks is not None
                else None
            ),
            created_at=time.monotonic(),
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
        logger.debug(
            "[Lwd][coord] probe req=%s hit_blocks=%d hit_tokens=%d",
            manifest.request_id, hit_blocks, hit_tokens,
        )
        return LwdProbeResult(
            request_id=manifest.request_id,
            instance_id=self._instance_id,
            block_size=self._hash_block_size,
            hit_blocks=hit_blocks,
            hit_tokens=hit_tokens,
            alignment_tokens=self._resolve_alignment_tokens(),
            cloud_id=self._cloud_id,
        )

    def _resolve_alignment_tokens(self) -> int:
        """云侧对齐网格(scheduler_block_size = 各 KV 组块大小 LCM),随
        probe 响应公告给边侧做 mamba 对齐重切;无块池(降级)返回 0。"""
        km = self._kv_cache_manager
        if km is None:
            return 0
        return int(getattr(km.coordinator, "scheduler_block_size", 0))

    def complete_request(self, request_id: str) -> None:
        """数据面请求 prompt 算完后,把其 manifest 的全部满块协调摘要
        幂等发布进 ``_completed_hashes``(无块池降级路径的命中输入;
        正常路径命中以块池单真源为准,续算闭环不依赖本回填)。

        由云引擎在步后调用(worker 执行已由本步返回隐含);无预留则跳过。
        """
        reservation = self._reservations.get(request_id)
        if reservation is None:
            return
        # E5 幂等短路:首发布后每个 decode 步零开销(不再重建链、不再
        # 扫 O(块数) 的 set update)
        if reservation.completed_published:
            return
        # 释放预约 touch:prefill 已复用(create_kv_cache_blocks 已对命中块
        # ref_cnt+1),抵消 probe 时的预约 pin,避免块永久泄漏。幂等:释放后清空。
        if reservation.touched_blocks and self._kv_cache_manager is not None:
            self._kv_cache_manager.block_pool.free_blocks(reservation.touched_blocks)
            reservation.touched_blocks = None
        if reservation.chain is None:
            reservation.chain = self._coordination_block_hashes(reservation.manifest)
        chain = reservation.chain
        before = len(self._completed_hashes)
        self._completed_hashes.update(chain)
        reservation.completed_published = True
        # 幂等回填:prompt 算完后每个 decode 步都会被调一次,只有真新增
        # 摘要时才打日志(否则长生成把日志刷满,掩盖 probe 行)
        if len(self._completed_hashes) != before:
            logger.debug(
                "[Lwd][coord] complete req=%s blocks=%d new=%d",
                request_id, len(chain), len(self._completed_hashes) - before,
            )

    def claim(self, request_id: str) -> int | None:
        """数据面请求认领预留,返回命中续算起点(token 数);无预留返回 None。

        续算边界已由块池单真源闭环(probe 命中 == 块池 resume),本方法
        仅保留 admit 期可观测(见 engine)。认领即标记:已认领预留不
        参与 TTL 清扫(pin 由 complete/finish 正常释放)。"""
        reservation = self._reservations.get(request_id)
        if reservation is None:
            return None
        reservation.claimed = True
        return reservation.hit_tokens

    def sweep_expired(self, now: float) -> int:
        """清扫超 TTL 仍未认领的预留:释放 pin 并摘除,返回清扫数。

        调用点:_lwd_poll_coordination(引擎步循环头部/WAKEUP 后,主循环
        线程,与 probe/complete/finish 同线程,无竞态)。只清未认领的:
        已认领预留可能处于长 decode,其 pin 早已在 complete 释放、
        reservation 本体由 finish 摘除,与 TTL 无关。"""
        expired = [
            rid
            for rid, res in self._reservations.items()
            if not res.claimed
            and res.created_at > 0
            and now - res.created_at > self._RESERVATION_TTL_S
        ]
        for rid in expired:
            res = self._reservations.pop(rid, None)
            if res is None:
                continue
            if res.touched_blocks and self._kv_cache_manager is not None:
                self._kv_cache_manager.block_pool.free_blocks(res.touched_blocks)
            self._prompt_chains.pop(rid, None)
            logger.warning(
                "[Lwd][coord] reservation expired unclaimed req=%s: "
                "pin released (notify lost or edge gone?)", rid,
            )
        return len(expired)

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
                "(probe 未落或已淘汰); fail-closed",
                request_id,
            )
            return False
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
        # 兜底释放预约 touch:abort/提前终止时 complete_request 未跑,预约 pin
        # 仍未释放;此处是 reservation 的最终摘除点,统一收口防泄漏。正常路径
        # complete_request 已释放并置 None,此处幂等跳过。
        if (
            reservation is not None
            and reservation.touched_blocks
            and self._kv_cache_manager is not None
        ):
            self._kv_cache_manager.block_pool.free_blocks(reservation.touched_blocks)
            reservation.touched_blocks = None
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