"""云侧复用(cloud reuse,多边多云)角色注册表。

静态角色表回答三件事:我是谁(``validate_self``/``edge_id``/``cloud_id``)、
我的对端是谁(``edges``/``clouds``)、某个 ``(edge, cloud)`` 数据面通信域
的端点 rank 是什么。云侧复用场景全员挂载同一份 YAML,由
``--role-registry`` 指定;未配置时退化单边一云 ``(edge0, cloud0)`` 域,
行为与原 prefill_only 完全一致(非侵入式)。

控制面端点规划(ROUTER-ROUTER):只有云侧 bind,每云单值 ``zmq_port``
服务全部边;边侧 0 端口 connect 出去。端点公式唯一事实源在
``endpoint``/``bind_endpoint``;数据面只消费 rank 映射
(``edge_rank``/``cloud_rank``)。

registry YAML 示例(2 边 2 云,全场共享同一份)::

    world: { master_addr: 10.1.0.1, master_port: 29500 }
    edges:
      - { id: 0, addr: 10.0.0.1, ranks: [0, 1] }           # 不 bind,无端口字段
      - { id: 1, addr: 10.0.0.2, ranks: [2, 3] }           # addr 仅诊断用,可省
    clouds:
      - { id: 0, addr: 10.1.0.1, ranks: [4, 5, 6, 7], zmq_port: 5700 }
      - { id: 1, addr: 10.1.0.2, ranks: [8, 9, 10, 11], zmq_port: 5700 }

云内 DP(GLM MoE)示例(1 云 dp=2,每引擎 TP=2,共 4 卡)::

    clouds:
      - id: 0
        addr: 10.1.0.1
        ranks: [4, 5, 6, 7]        # engine0=[4,5], engine1=[6,7]
        zmq_port: 5700
        tp_size: 2
        dp_size: 2
        engine_ports: [5710, 5711] # 每引擎 PRE_OUT
        coord_port: 5720           # DPCoordinator front 发布口
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import yaml
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class LwdPeerInfo:
    """单个边/云实例的静态身份。

    ``ranks`` 为该实例的全局 rank 列表(HCCL 组按全局 rank 建);
    ``zmq_port`` 仅云侧有值(边不 bind);``addr`` 边侧仅诊断用。
    """

    peer_id: int
    addr: str
    ranks: list[int]
    zmq_port: int = 0
    # ---- 云内 DP(GLM MoE 云侧 data_parallel)----
    # 仅云条目有意义;边条目忽略。云 c 的 ranks 约定为连续段布局:
    # engine i 占 ranks[i*tp_size : (i+1)*tp_size],len(ranks) ==
    # tp_size * dp_size。dp_size=1 时全部退化为现状(单引擎)。
    tp_size: int = 0
    """每引擎 TP 卡数;0 = 未声明,取 len(ranks)//dp_size 推出。"""
    dp_size: int = 1
    """云内 DP 引擎数(= 该云进程的 --data-parallel-size)。"""
    engine_ports: list[int] = None  # type: ignore[assignment]
    """每引擎 PRE_OUT 端口(边→该引擎的定向控制通道),长度 ==
    dp_size;dp_size=1 时可为空(沿用 zmq_port 的 ROUTER 通道)。"""
    coord_port: int = 0
    """该云 DPCoordinator 的 front 发布端口(边侧 FIRST_REQ 唤醒 /
    stats 对账用);dp_size>1 时必填,dp_size=1 无 coordinator 留 0。"""

    def __post_init__(self) -> None:
        if self.engine_ports is None:
            object.__setattr__(self, "engine_ports", [])


class LwdRoleRegistry:
    """边云静态角色表。

    ``edges``/``clouds`` 以 id 为键;``pairs()`` 给出全部 ``(edge_id,
    cloud_id)`` 通信域;``edge_rank``/``cloud_rank`` 返回该通信域的两端
    端点 rank(边入口 rank / 云 TP0 rank),供数据面建组与选通道;
    ``config_digest`` 供全场配置一致性校验。
    """

    def __init__(self, edges: dict[int, LwdPeerInfo],
                 clouds: dict[int, LwdPeerInfo],
                 config_digest: str = "") -> None:
        self._edges = edges
        self._clouds = clouds
        self._config_digest = config_digest

    @property
    def edges(self) -> dict[int, LwdPeerInfo]:
        return self._edges

    @property
    def clouds(self) -> dict[int, LwdPeerInfo]:
        return self._clouds

    @property
    def edge_ids(self) -> list[int]:
        return sorted(self._edges)

    @property
    def cloud_ids(self) -> list[int]:
        return sorted(self._clouds)

    @property
    def config_digest(self) -> str:
        """全场配置摘要;全员必须算出同一值(启动一致性校验)。"""
        return self._config_digest

    @property
    def is_cloud_reuse(self) -> bool:
        """云侧复用拓扑(多于一边或多于一云)为 True。"""
        return len(self._edges) > 1 or len(self._clouds) > 1

    # ------------------------------------------------------------------ #
    # 身份 / 在场校验                                                     #
    # ------------------------------------------------------------------ #
    def validate_self(self, role: str, instance_id: int) -> None:
        """校验本进程声明的 (role, id) 在表中在场;缺位 fail-fast。"""
        table = self._edges if role == "edge" else self._clouds
        if instance_id not in table:
            raise ValueError(
                f"[lwd] role={role} id={instance_id} not found in registry; "
                f"known ids={sorted(table)}"
            )

    def assert_rank_membership(self, role: str, instance_id: int,
                               global_rank: int) -> None:
        """校验实际全局 rank 与 registry 声明一致(防 HCCL 组错进程)。"""
        table = self._edges if role == "edge" else self._clouds
        expected = table[instance_id].ranks
        if global_rank not in expected:
            raise ValueError(
                f"[lwd] rank/registry mismatch: role={role} id={instance_id} "
                f"process has global_rank={global_rank}, but registry says "
                f"ranks={expected}"
            )

    def self_ids(self, my_rank: int) -> tuple[str | None, int, int]:
        """由全局 rank 判定本进程身份:(role, edge_id, cloud_id)。"""
        for edge_id, peer in self._edges.items():
            if my_rank in peer.ranks:
                return "edge", edge_id, 0
        for cloud_id, peer in self._clouds.items():
            if my_rank in peer.ranks:
                return "cloud", 0, cloud_id
        return None, 0, 0

    # ------------------------------------------------------------------ #
    # 数据面端点(rank 映射)                                              #
    # ------------------------------------------------------------------ #
    def edge_ranks(self, edge_id: int) -> list[int]:
        return list(self._edges[edge_id].ranks)

    def cloud_ranks(self, cloud_id: int) -> list[int]:
        return list(self._clouds[cloud_id].ranks)

    def edge_rank(self, edge_id: int) -> int:
        """边入口 rank(= 边 ranks 首卡)。"""
        return self._edges[edge_id].ranks[0]

    def cloud_rank(self, cloud_id: int) -> int:
        """云端点 rank(= 云 TP0 首卡)。"""
        return self._clouds[cloud_id].ranks[0]

    def cloud_endpoint_rank(self, edge_id: int, cloud_id: int) -> int:
        """云端点 rank:随 edge_id 在该云各 rank 间轮转。

        edge 0 -> ranks[0], edge 1 -> ranks[1], ... 取模回到 ranks[0],
        把不同边的跨机 P2P 分摊到云内不同卡,避免全部压在 TP0 首卡。
        单卡云(单边一云退化域) len(ranks)==1 恒取 ranks[0],与原行为一致。
        """
        ranks = self._clouds[cloud_id].ranks
        return ranks[edge_id % len(ranks)]

    def pairs(self) -> list[tuple[int, int]]:
        return [(e, c) for e in self.edge_ids for c in self.cloud_ids]

    # ------------------------------------------------------------------ #
    # 云内 DP(engine 维,pair 的第三维)                                  #
    # ------------------------------------------------------------------ #
    def cloud_tp_size(self, cloud_id: int) -> int:
        """云 c 的每引擎 TP 卡数 C;未声明 tp_size 时按 ranks/dp 推出。"""
        cloud = self._clouds[cloud_id]
        if cloud.tp_size > 0:
            return cloud.tp_size
        return len(cloud.ranks) // cloud.dp_size

    def cloud_dp_size(self, cloud_id: int) -> int:
        return self._clouds[cloud_id].dp_size

    def engine_ids(self, cloud_id: int) -> list[int]:
        return list(range(self._clouds[cloud_id].dp_size))

    def cloud_engine_ranks(self, cloud_id: int, engine_idx: int) -> list[int]:
        """云 c 引擎 i 的全局 rank 段:ranks[i*C : (i+1)*C]。"""
        cloud = self._clouds[cloud_id]
        if not 0 <= engine_idx < cloud.dp_size:
            raise ValueError(
                f"[lwd] engine_idx={engine_idx} out of range for cloud "
                f"{cloud_id} (dp_size={cloud.dp_size})"
            )
        c = self.cloud_tp_size(cloud_id)
        return list(cloud.ranks[engine_idx * c:(engine_idx + 1) * c])

    def engine_endpoint_rank(self, edge_id: int, cloud_id: int,
                             engine_idx: int) -> int:
        """(edge, cloud, engine) 通信域的云端点 rank:随 edge_id 在该
        引擎的 C 个 rank 内轮转(端点必须属于目标引擎的 TP 组,否则
        UP 端点不在组内)。单引擎单卡退化与 cloud_endpoint_rank 一致。"""
        ranks = self.cloud_engine_ranks(cloud_id, engine_idx)
        return ranks[edge_id % len(ranks)]

    def engine_index_of_rank(self, cloud_id: int, global_rank: int) -> int:
        """云 c 内某全局 rank 所属的引擎下标;不在该云则 fail-fast。"""
        ranks = self._clouds[cloud_id].ranks
        if global_rank not in ranks:
            raise ValueError(
                f"[lwd] rank {global_rank} not in cloud {cloud_id} "
                f"ranks={ranks}"
            )
        return ranks.index(global_rank) // self.cloud_tp_size(cloud_id)

    def triples(self) -> list[tuple[int, int, int]]:
        """全部 (edge_id, cloud_id, engine_idx) 通信域,字典序。"""
        return [
            (e, c, i) for e in self.edge_ids for c in self.cloud_ids
            for i in self.engine_ids(c)
        ]

    def coord_endpoint(self, cloud_id: int) -> str | None:
        """云 c 的 DPCoordinator front 发布端点(边侧 FIRST_REQ/stats);
        dp_size==1 无 coordinator 进程,恒返回 None(即使误配
        coord_port——dp=1 下该端口无人 bind,建客户端只会误导排障)。"""
        cloud = self._clouds[cloud_id]
        if cloud.dp_size <= 1 or cloud.coord_port <= 0:
            return None
        return f"tcp://{cloud.addr}:{cloud.coord_port}"

    def engine_endpoint(self, cloud_id: int, engine_idx: int) -> str:
        """云 c 引擎 i 的 PRE_OUT 端点(边侧定向发布):engine_ports
        声明时按表;未声明(dp_size==1 旧式部署)回落 ROUTER 通道端点。"""
        cloud = self._clouds[cloud_id]
        if cloud.engine_ports:
            return f"tcp://{cloud.addr}:{cloud.engine_ports[engine_idx]}"
        return self.endpoint(cloud_id)

    def engine_bind_endpoint(self, cloud_id: int, engine_idx: int) -> str:
        """云 c 引擎 i 的 PRE_OUT bind 端点(* 形式)。"""
        cloud = self._clouds[cloud_id]
        if cloud.engine_ports:
            return f"tcp://*:{cloud.engine_ports[engine_idx]}"
        return self.bind_endpoint(cloud_id)

    # ------------------------------------------------------------------ #
    # 控制面端点(唯一事实源)                                             #
    # ------------------------------------------------------------------ #
    def endpoint(self, cloud_id: int) -> str:
        """云控制面端点(边侧 connect 用):``tcp://{cloud.addr}:{zmq_port}``。"""
        cloud = self._clouds[cloud_id]
        return f"tcp://{cloud.addr}:{cloud.zmq_port}"

    def bind_endpoint(self, cloud_id: int) -> str:
        """云控制面 bind 端点:addr 换为 ``*``,每云单端口服务全部边。"""
        return f"tcp://*:{self._clouds[cloud_id].zmq_port}"

    # ------------------------------------------------------------------ #
    # 构造                                                               #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LwdRoleRegistry":
        edges = {
            int(e["id"]): LwdPeerInfo(
                peer_id=int(e["id"]),
                addr=str(e.get("addr", "")),
                ranks=[int(r) for r in e.get("ranks", [])],
            )
            for e in raw.get("edges", [])
        }
        clouds = {
            int(c["id"]): LwdPeerInfo(
                peer_id=int(c["id"]),
                addr=str(c.get("addr", "")),
                ranks=[int(r) for r in c.get("ranks", [])],
                # ROUTER-ROUTER:每云单端口;兼容旧 zmq_base_port 字段名
                zmq_port=int(c.get("zmq_port", c.get("zmq_base_port", 0))),
                tp_size=int(c.get("tp_size", 0)),
                dp_size=int(c.get("dp_size", 1)),
                engine_ports=[int(p) for p in c.get("engine_ports", [])],
                coord_port=int(c.get("coord_port", 0)),
            )
            for c in raw.get("clouds", [])
        }
        if not edges or not clouds:
            raise ValueError(
                "[lwd] role registry requires at least one edge and one cloud"
            )
        empty_ranks = [
            f"{role}.{peer_id}"
            for role, peers in (("edge", edges), ("cloud", clouds))
            for peer_id, peer in peers.items()
            if not peer.ranks
        ]
        if empty_ranks:
            raise ValueError(
                "role registry peers must declare non-empty ranks: "
                + ", ".join(empty_ranks)
            )
        for cloud_id, cloud in clouds.items():
            if cloud.zmq_port <= 0:
                raise ValueError(
                    f"[lwd] cloud {cloud_id} requires a positive zmq_port "
                    f"in the role registry (ROUTER bind)"
                )
            # 云内 DP 拓扑校验:ranks 必须恰好切成 dp_size 段;dp>1 时
            # 每引擎 PRE_OUT 端口与 coordinator 端口必须显式声明。
            if cloud.dp_size < 1:
                raise ValueError(
                    f"[lwd] cloud {cloud_id} dp_size must be >= 1, got "
                    f"{cloud.dp_size}"
                )
            c_size = cloud.tp_size or (len(cloud.ranks) // cloud.dp_size)
            if c_size <= 0 or c_size * cloud.dp_size != len(cloud.ranks):
                raise ValueError(
                    f"[lwd] cloud {cloud_id} ranks({len(cloud.ranks)}) cannot "
                    f"be evenly split into dp_size={cloud.dp_size} engines "
                    f"of tp_size={c_size}; ranks must be the contiguous "
                    f"[engine0 | engine1 | ...] layout"
                )
            if cloud.dp_size > 1:
                if len(cloud.engine_ports) != cloud.dp_size:
                    raise ValueError(
                        f"[lwd] cloud {cloud_id} dp_size={cloud.dp_size} "
                        f"requires engine_ports with exactly dp_size entries, "
                        f"got {cloud.engine_ports}"
                    )
                if cloud.coord_port <= 0:
                    raise ValueError(
                        f"[lwd] cloud {cloud_id} dp_size={cloud.dp_size} "
                        f"requires a positive coord_port (DPCoordinator "
                        f"front publish, for edge FIRST_REQ/stats)"
                    )
        # 全场 ranks 重叠检查(跨实例撞 rank 要到 HCCL 建组才爆,
        # 提前到装载期 fail-fast)
        seen_ranks: dict[int, str] = {}
        for role, peers in (("edge", edges), ("cloud", clouds)):
            for peer_id, peer in peers.items():
                for r in peer.ranks:
                    if r in seen_ranks:
                        raise ValueError(
                            f"[lwd] registry rank {r} claimed by both "
                            f"{seen_ranks[r]} and {role}.{peer_id}"
                        )
                    seen_ranks[r] = f"{role}.{peer_id}"
        # 同机端口冲突检查:同 addr 的云的 zmq_port/engine_ports/
        # coord_port 不得互撞(撞了要运行期 zmq bind 才报错)
        port_owner: dict[tuple[str, int], str] = {}
        for cloud_id, cloud in clouds.items():
            ports = [("zmq_port", cloud.zmq_port)]
            ports += [
                (f"engine_ports[{i}]", p)
                for i, p in enumerate(cloud.engine_ports)
            ]
            if cloud.coord_port > 0:
                ports.append(("coord_port", cloud.coord_port))
            for name, port in ports:
                key = (cloud.addr, port)
                if key in port_owner:
                    raise ValueError(
                        f"[lwd] registry port conflict on {cloud.addr}:"
                        f"{port}: cloud {cloud_id} {name} vs "
                        f"{port_owner[key]}"
                    )
                port_owner[key] = f"cloud {cloud_id} {name}"
        digest = hashlib.sha256(
            json.dumps(raw, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        return cls(edges, clouds, digest)

    @classmethod
    def single_pair(cls, edge_rank: int, cloud_rank: int) -> "LwdRoleRegistry":
        """单边一云退化域(未配置 role_registry 时)。"""
        return cls(
            {0: LwdPeerInfo(0, "", [edge_rank], 0)},
            {0: LwdPeerInfo(0, "", [cloud_rank], 0)},
        )


def load_role_registry(role_registry_path: str,
                       fallback: LwdRoleRegistry | None = None
                       ) -> LwdRoleRegistry:
    """按路径加载角色表;路径为空/加载失败回退 fallback。"""
    if not role_registry_path:
        if fallback is None:
            raise ValueError("role_registry_path is empty and no fallback registry")
        return fallback
    try:
        with open(role_registry_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return LwdRoleRegistry.from_dict(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[lwd] failed to load role registry %s: %s",
                       role_registry_path, exc)
        if fallback is not None:
            return fallback
        raise


# ---------------------------------------------------------------------- #
# 进程级单例(控制面装配期加载,运行期只读)                                #
# ---------------------------------------------------------------------- #
_REGISTRY: LwdRoleRegistry | None = None


def init_role_registry(path: str) -> LwdRoleRegistry:
    """进程级单例:首次加载并缓存 registry,重复调用返回同一份。"""
    global _REGISTRY
    if _REGISTRY is None:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        _REGISTRY = LwdRoleRegistry.from_dict(raw)
        logger.info(
            "[lwd] role registry loaded from %s: edges=%s clouds=%s digest=%s",
            path, _REGISTRY.edge_ids, _REGISTRY.cloud_ids,
            _REGISTRY.config_digest,
        )
    return _REGISTRY


def get_role_registry() -> LwdRoleRegistry | None:
    """取已加载 registry;未启用云侧复用时为 None。"""
    return _REGISTRY
