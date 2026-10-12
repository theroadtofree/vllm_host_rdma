"""边侧请求-云绑定表(云侧复用)。

请求生命周期内与目标云实例绑定/解绑的唯一台账:
  * 请求入口(add 预告前)``bind`` 登记;
  * 范围预告/abort 按 ``cloud_of`` 定向控制面发布;
  * finish/abort 时 ``unbind`` 解绑(释放台账,迟到结果不再认领)。

选路由 probe 回执 cloud_id 驱动(``_lwd_routed_cloud_id``):多云时
按回执绑定;单云/未启用选路时回退 ``registry.cloud_ids[0]``。本表
接口不变。
"""

from __future__ import annotations


class LwdEdgeBindingTable:
    """req_id -> cloud_id 绑定表;dict 语义薄封装,生命周期随调度器。"""

    def __init__(self) -> None:
        self._req_cloud: dict[str, int] = {}

    def bind(self, request_id: str, cloud_id: int) -> None:
        """登记请求与云的绑定(重复绑定按后写覆盖)。"""
        self._req_cloud[request_id] = cloud_id

    def cloud_of(self, request_id: str, default: int | None = None) -> int | None:
        """查请求绑定云;未登记返回 default(未传为 None)。"""
        return self._req_cloud.get(request_id, default)

    def unbind(self, request_id: str) -> int | None:
        """解绑并返回原绑定云;未登记返回 None(幂等)。"""
        return self._req_cloud.pop(request_id, None)

    def bound_cloud_ids(self) -> set[int]:
        """当前有在途请求的云 id 集合。"""
        return set(self._req_cloud.values())

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._req_cloud

    def __len__(self) -> int:
        return len(self._req_cloud)
