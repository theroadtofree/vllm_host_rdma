# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from lwd_config_under_test import LwdAPIConfig, LwdConfig, log_lwd_config
from sample_topologies import topology_payload


@pytest.mark.parametrize("role,identity", [("edge", 1), ("cloud", 0)])
def test_log_contains_complete_selected_config_without_reading_key_contents(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, role: str, identity: int
) -> None:
    raw = topology_payload("s3_cloud_share_1dp.yaml")
    raw["feature_ctrl"] = {
        "enable_early_recv": True,
        "scramble": {
            "enable": True,
            "device_id": "ECPAIR-SN-8A3F91",
            "custom_factor": "biz-2024-q3",
        },
    }
    path = tmp_path / "topology.yaml"
    key = tmp_path / "tenant.key"
    key.write_text("local-key-content-must-not-be-logged")
    coordination: dict[str, Any] = {"enabled": True}
    if role == "edge":
        coordination.update(
            tenant_key_file=str(key),
            consumer_id="enterprise-a-edge1",
            connect_timeout=5.0,
        )
        raw["edges"][identity]["coordination"] = coordination
    else:
        coordination.update(listen_host="0.0.0.0")
        raw["clouds"][identity]["coordination"] = coordination
    path.write_text(yaml.safe_dump(raw))
    config = LwdConfig.from_dict(
        {
            "path": str(path),
            "role": role,
            "instance_id": identity,
        }
    )
    path.unlink()
    with caplog.at_level(logging.INFO):
        log_lwd_config(logging.getLogger("test.lwd-config"), config)
    line = caplog.records[-1].getMessage()
    assert line.startswith("[LWD][config] resolved=")
    result = json.loads(line.split("resolved=", 1)[1])
    assert result["metadata_only"] is True
    actual = result["lwd_config"]
    assert (actual["path"], actual["role"], actual["instance_id"]) == (
        str(path),
        role,
        identity,
    )
    topology = actual["topology"]
    assert topology.pop("digest") == config.topology.digest
    assert topology.pop("frontend_digest") == config.topology.frontend_digest
    assert [(link["edge"], link["cloud"]) for link in topology["instance_links"]] == [
        (0, 0),
        (1, 0),
    ]
    assert topology["clouds"][0]["dp"][0]["coord_port"] == 8100
    assert topology["clouds"][0]["dp"][0]["coord_url"] == "http://10.1.0.1:8100"
    parsed = actual["coordination"]
    assert all(parsed[name] == value for name, value in coordination.items())
    assert set(parsed) == {
        "enabled",
        "placement_only",
        "tenant_key_file",
        "consumer_id",
        "listen_host",
        "instance_id",
        "connect_timeout",
        "probe_timeout",
        "enforce_mm_abi_match",
    }
    assert parsed["instance_id"] == "cloud-0"
    assert result["cloud_cards"] == {"0": 8}
    assert len(result["connections"]) == (1 if role == "edge" else 2)
    assert all(link["cloud_leader_rank"] == 2 for link in result["connections"])
    assert result["selected_instance"]["id"] == identity
    assert "lwd_api_config" not in result
    assert actual["api"] == {
        "api_server_rpc_port": None,
        "api_server_attach": None,
    }
    assert "local-key-content-must-not-be-logged" not in line


@pytest.mark.parametrize(
    "identity,field", [(0, "api_server_rpc_port"), (1, "api_server_attach")]
)
def test_log_includes_multi_instance_api_endpoint(
    lwd_yaml_dir: Path, caplog: pytest.LogCaptureFixture, identity: int, field: str
) -> None:
    config = LwdConfig.from_dict(
        {
            "path": str(lwd_yaml_dir / "s2_edge_share_2dp.yaml"),
            "role": "edge",
            "instance_id": identity,
        },
        api=LwdAPIConfig(**{field: "10.0.0.1:25100"}),
    )
    with caplog.at_level(logging.INFO):
        log_lwd_config(logging.getLogger("test.lwd-config"), config)
    result = json.loads(caplog.records[-1].getMessage().split("resolved=", 1)[1])
    assert "lwd_api_config" not in result
    assert result["lwd_config"]["api"][field] == "10.0.0.1:25100"
    assert result["lwd_config"]["coordination"] is None
    assert [link["connection_id"]["dp_idx"] for link in result["connections"]] == [
        0,
        1,
        0,
        1,
    ]


def _host_topology_file(directory: Path) -> Path:
    """Write a host-comm layout: no ranks, no hccl_world_size, data_port per DP."""
    raw = topology_payload("s2_edge_share_2dp.yaml")
    deployment = raw["deployment"]
    deployment["data_plane_comm"] = "host"
    deployment["data_plane_host"] = {"up_buffer_size": 64, "down_buffer_size": 32}
    deployment.pop("hccl_world_size", None)
    for instance in raw["edges"] + raw["clouds"]:
        for dp in instance["dp"]:
            dp.pop("ranks", None)
    port = 7000
    for cloud in raw["clouds"]:
        for dp in cloud["dp"]:
            dp["data_port"] = port
            port += 1
    path = directory / "host_topology.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def _qos_line(caplog: pytest.LogCaptureFixture, config: LwdConfig) -> str:
    with caplog.at_level(logging.INFO):
        log_lwd_config(logging.getLogger("test.lwd-qos"), config)
    return next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("[LWD][qos]")
    )


def test_log_shows_the_effective_host_qos(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No qos block: the host default policy is what the run actually carries."""
    path = _host_topology_file(tmp_path)
    config = LwdConfig.from_dict({"path": str(path), "role": "edge", "instance_id": 1})

    line = _qos_line(caplog, config)

    assert line == (
        "[LWD][qos] comm=host data_plane traffic_class=138 (dscp=34 ecn=2) "
        "sl=4 | ctrl_plane dscp=48 (tos=192)"
    )


def test_log_shows_device_comm_as_unmarked(
    lwd_yaml_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Device (HCCL) comm keeps the historical behaviour: nothing is marked."""
    config = LwdConfig.from_dict(
        {
            "path": str(lwd_yaml_dir / "s2_edge_share_2dp.yaml"),
            "role": "edge",
            "instance_id": 1,
        }
    )

    line = _qos_line(caplog, config)

    assert line == (
        "[LWD][qos] comm=device unset (lwd marks nothing; HCCL keeps its own TC/SL)"
    )
