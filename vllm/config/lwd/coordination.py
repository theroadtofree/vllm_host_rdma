# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache coordination metadata only; no key reads, clients or server startup."""

from typing import Annotated, Any
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, StrictBool, field_validator
from pydantic.dataclasses import dataclass

from .topology import LwdPort, LwdString


def _validate_http_url(value: str | None) -> None:
    if value is not None:
        url = urlsplit(value)
        if url.scheme not in ("http", "https") or not url.hostname:
            raise ValueError("[LWD] control_url must be an HTTP(S) endpoint")


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdCoordinationConfig:
    enabled: StrictBool = False
    control_url: LwdString | None = None
    control_url_map: dict[int, LwdString] = Field(default_factory=dict)
    placement_only: StrictBool = False
    tenant_key_file: LwdString | None = None
    consumer_id: LwdString | None = None
    listen_host: LwdString = "0.0.0.0"
    listen_port: LwdPort = 8100
    instance_id: LwdString = "cloud-0"
    connect_timeout: Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)] = (
        10.0
    )

    probe_timeout: Annotated[float, Field(strict=True, gt=5.0, allow_inf_nan=False)] = (
        25.0
    )
    enforce_mm_abi_match: StrictBool = True

    @field_validator("control_url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        _validate_http_url(value)
        return value

    @field_validator("control_url_map")
    @classmethod
    def validate_url_map(cls, value: dict[int, str]) -> dict[int, str]:
        for cloud_id, url in value.items():
            if cloud_id < 0:
                raise ValueError("[LWD] control_url_map keys must be non-negative")
            _validate_http_url(url)
        return value

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], *, cloud_id: int | None = None
    ) -> "LwdCoordinationConfig":
        raw = dict(raw)
        raw_url = raw.get("control_url")
        if isinstance(raw_url, dict):
            if raw.get("control_url_map"):
                raise ValueError(
                    "[LWD] control_url map and control_url_map are mutually exclusive"
                )
            raw["control_url"] = None
            raw["control_url_map"] = raw_url
        defaults = {} if cloud_id is None else {"instance_id": f"cloud-{cloud_id}"}
        return cls(**(defaults | raw))

    def validate_role(self, role: str) -> None:
        if (
            self.enabled
            and role == "edge"
            and (
                not (self.control_url or self.control_url_map)
                or (not self.tenant_key_file and not self.placement_only)
            )
        ):
            raise ValueError(
                "[LWD] edge coordination requires control_url/map and, unless "
                "placement_only, tenant_key_file"
            )
