#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""边/云多实例控制面注册表。

边侧 EngineCore 持有逐实例调度状态,并承担云实例的启动注册屏障。本模块刻意
保持纯 stdlib(无 torch、无 zmq),可独立单测,并可从边侧与云侧同时导入。

启动交换(每个边 DP rank 各一次,走 ``master_port + 1 + dp_rank`` 上的一次性
TCPStore)::

    cloud instance i -> edge:  set("cloud_instance_{i}", JSON payload)
    edge              -> self: blocking get() of every key i in [0, N)

payload 携带云实例的实例索引(``--instance-parallel-start-rank``)与实例总数
(``--instance-parallel-size``);边侧把总数与自身 ``instance_parallel_size``
交叉校验,使实例数不匹配的云命令 fail-fast 而非半接线。

请求路径:每个 ApiServer 在其 ``EngineCoreRequest`` 上打 ``client_index``;
endpoint i == instance i == client_index i,故请求的 ``client_index`` 即控制面
消费的实例 pin(在调度器入口由 ``validate_request_instance_pin`` 校验)。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

# 云实例注册 payload 的 TCPStore key 前缀。实例索引编码进 key,使写同一 store
# 的 N 个实例互不覆盖(单 key "cloud_ip" 正是 last-writer-wins 导致实例串线的
# 根因)。
_REGISTRATION_KEY_PREFIX = "cloud_instance_"


def registration_key(instance_id: int) -> str:
    """云实例 ``instance_id`` 注册所用的 TCPStore key。"""
    return f"{_REGISTRATION_KEY_PREFIX}{instance_id}"


@dataclass
class CloudInstanceRegistration:
    """单个云实例向边侧控制面报告的内容。"""

    # 报告方云命令的实例索引
    # (``--instance-parallel-start-rank``,None 归一化为 0)。
    instance_id: int
    # 报告方云命令配置的实例总数
    # (``--instance-parallel-size``)。收集时与边侧自身 N 交叉校验。
    instance_count: int
    # 云节点可达 IP(供边侧 POST_OUT 连接)。
    cloud_ip: str
    # 报告方 PassiveEngineCore 的 DP rank(每个边 DP rank 有自己的 store;
    # 各云实例匹配的 DP rank 在那里注册)。
    dp_rank: int
    # 报告进程的 OS pid(仅诊断;pid 跨主机不唯一,故不做唯一性约束)。
    pid: int

    def encode(self) -> str:
        return json.dumps(
            {
                "instance_id": self.instance_id,
                "instance_count": self.instance_count,
                "cloud_ip": self.cloud_ip,
                "dp_rank": self.dp_rank,
                "pid": self.pid,
            }
        )

    @classmethod
    def decode(cls, raw: bytes | str) -> "CloudInstanceRegistration":
        if isinstance(raw, bytes):
            raw = raw.decode()
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Malformed cloud instance registration payload: {raw!r} ({e})"
            ) from e
        if not isinstance(payload, dict):
            raise ValueError(
                f"Malformed cloud instance registration payload: {raw!r} "
                "(expected a JSON object)"
            )
        try:
            return cls(
                instance_id=int(payload["instance_id"]),
                instance_count=int(payload["instance_count"]),
                cloud_ip=str(payload["cloud_ip"]),
                dp_rank=int(payload["dp_rank"]),
                pid=int(payload["pid"]),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                "Cloud instance registration payload is missing or has "
                f"invalid fields: {payload!r} ({e})"
            ) from e


class InstanceControlPlane:
    """边侧边/云多实例控制面状态。

    持有实例总数(由边命令自身 ``--instance-parallel-size`` 给出)与启动时收集
    到的逐实例云注册信息。由边 EngineCore 在所有 N 个云实例注册完成后创建。
    """

    def __init__(
        self,
        instance_count: int,
        cloud_instances: dict[int, CloudInstanceRegistration],
        dp_rank: int,
    ):
        self.instance_count = instance_count
        self.cloud_instances = cloud_instances
        self.dp_rank = dp_rank

    @classmethod
    def collect_from_store(
        cls,
        store,
        instance_count: int,
        dp_rank: int,
    ) -> "InstanceControlPlane":
        """从 TCPStore 收集 N 个云实例注册信息。

        按索引顺序阻塞等待每个注册 key(上限为 store 自身超时)——这是云实例的
        "数到 N 再放行"屏障。始终未出现的云命令会报出被等待的实例索引,便于
        诊断。

        Args:
            store: 边侧的一次性 ``dist.TCPStore``(master)。
            instance_count: 边命令配置的 N
                (``parallel_config.instance_parallel_size``)。
            dp_rank: 本边 EngineCore 的 DP rank(store 按 DP rank 隔离)。
        """
        cloud_instances: dict[int, CloudInstanceRegistration] = {}
        for instance_id in range(instance_count):
            key = registration_key(instance_id)
            try:
                raw = store.get(key)
            except Exception as e:  # noqa: BLE001 - re-raise with context
                raise RuntimeError(
                    f"Timed out or failed waiting for cloud instance "
                    f"{instance_id} to register on key {key!r} "
                    f"(dp_rank {dp_rank}). The edge command requires all "
                    f"{instance_count} cloud instance(s) "
                    f"(--instance-parallel-size {instance_count}) to "
                    f"register before startup can proceed. Underlying "
                    f"error: {e}"
                ) from e
            try:
                registration = CloudInstanceRegistration.decode(raw)
            except ValueError as e:
                raise RuntimeError(
                    f"Cloud instance {instance_id} registered an unreadable "
                    f"payload on key {key!r}: {e}"
                ) from e
            if registration.instance_id != instance_id:
                raise RuntimeError(
                    f"Cloud instance registration under key {key!r} reports "
                    f"instance_id {registration.instance_id} (payload/key "
                    "mismatch); the cloud command likely has a corrupted "
                    "--instance-parallel-start-rank."
                )
            if registration.instance_count != instance_count:
                raise RuntimeError(
                    f"Cloud instance {instance_id} reports "
                    f"instance_count={registration.instance_count} but the "
                    f"edge command is configured for "
                    f"instance_count={instance_count}. Check that every "
                    "cloud command's --instance-parallel-size matches the "
                    "edge command's."
                )
            if not registration.cloud_ip:
                raise RuntimeError(
                    f"Cloud instance {instance_id} registered an empty "
                    "cloud_ip; the edge cannot connect POST_OUT to it."
                )
            cloud_instances[instance_id] = registration
        return cls(instance_count, cloud_instances, dp_rank)

    def describe(self) -> str:
        """启动日志用单行摘要。"""
        instances = ", ".join(
            f"instance {i} -> {reg.cloud_ip} (pid {reg.pid})"
            for i, reg in sorted(self.cloud_instances.items())
        )
        return (
            f"{self.instance_count} cloud instance(s) registered "
            f"(dp_rank {self.dp_rank}): {instances}"
        )


def validate_request_instance_pin(client_index: int, instance_count: int) -> int:
    """校验请求的实例 pin。

    接收请求的前端把它 pin 到自身实例:``request.client_index == instance index``
    (endpoint i == instance i == client_index i)。在调度器入口调用,使与
    ``--instance-parallel-size`` 不一致的前端/gateway 数量 fail-fast 而非在
    实例间静默串路。

    Returns:
        校验通过的实例索引。
    """
    if not 0 <= client_index < instance_count:
        raise ValueError(
            f"Request carries instance pin {client_index!r} outside "
            f"[0, {instance_count}). The number of frontends/gateway "
            "endpoints disagrees with --instance-parallel-size; every "
            "frontend must map 1:1 onto an instance."
        )
    return client_index
