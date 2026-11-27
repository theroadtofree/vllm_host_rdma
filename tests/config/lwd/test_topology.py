# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from typing import Any

import pytest
from lwd_config_under_test import LwdConnectionId, LwdTopology
from sample_topologies import PAYLOADS, topology_payload


def cloud_reuse(dp: int = 2) -> dict:
    return topology_payload(f"s3_cloud_share_{dp}dp.yaml")


@pytest.mark.parametrize("name", sorted(PAYLOADS))
def test_original_eight_layouts_keep_rank_and_link_contract(name: str) -> None:
    topology = LwdTopology.from_dict(topology_payload(name))
    expected = {
        "single_instance": (9, 1),
        "edge_share": (17, 2),
        "cloud_share": (10, 2),
        "lwd_cluster": (18, 4),
    }
    world, links = expected[topology.deployment.scene]
    assert topology.deployment.hccl_world_size == world
    assert len(topology.instance_links) == links
    dp_count = len(topology.clouds[0].dp)
    assert (
        sum(len(topology.connections("edge", e.id)) for e in topology.edges)
        == links * dp_count
    )


@pytest.mark.parametrize("dp_count", [1, 2])
def test_colocated_edges_keep_distinct_ranks_and_share_cloud(dp_count: int) -> None:
    raw = cloud_reuse(dp_count)
    for dp in raw["edges"][1]["dp"]:
        dp["addr"] = raw["edges"][0]["dp"][0]["addr"]
    topology = LwdTopology.from_dict(raw)
    assert [e.dp[0].ranks for e in topology.edges] == [(0,), (1,)]
    assert topology.deployment.hccl_world_size == 10
    for edge in topology.edges:
        assert {d.ranks for d in edge.dp} == {(edge.id,)}
        assert [
            c.connection_id.cloud_id for c in topology.connections("edge", edge.id)
        ] == [0] * dp_count


def minimal_edge_share_payload() -> dict:
    """One-pair edge_share in host comm: the degenerate form of the
    shared-edge pipeline, so a deployment can verify the whole attach
    machinery with a single instance and grow by extending the plan."""
    return {
        "deployment": {
            "mode": 0,
            "scene": "edge_share",
            "edges_num": 1,
            "clouds_num": 1,
            "data_plane_comm": "host",
        },
        "feature_ctrl": {
            "enable_early_recv": False,
            "scramble": {"enable": False, "device_id": None, "custom_factor": None},
        },
        "edges": [{"id": 0, "dp": [{"dp_idx": 0, "addr": "135.150.2.2"}]}],
        "clouds": [
            {
                "id": 0,
                "dp": [
                    {
                        "dp_idx": 0,
                        "addr": "16.32.2.2",
                        "ctrl_port": 5550,
                        "data_port": 5560,
                    }
                ],
            }
        ],
        "instance_links": [{"edge": 0, "cloud": 0}],
    }


def test_single_pair_edge_share_is_the_degenerate_pipeline_form() -> None:
    topology = LwdTopology.from_dict(minimal_edge_share_payload())
    assert topology.deployment.scene == "edge_share"
    (connection,) = topology.connections("edge", 0)
    assert connection.connection_id == LwdConnectionId(0, 0, 0)
    # The cloud sees the same single connection from its own side.
    assert topology.connections("cloud", 0) == (connection,)


def test_edge_share_still_requires_equal_instance_counts() -> None:
    raw = minimal_edge_share_payload()
    raw["deployment"]["edges_num"] = 2
    raw["edges"].append({"id": 1, "dp": [{"dp_idx": 0, "addr": "135.150.2.2"}]})
    with pytest.raises(ValueError, match="equal edge/cloud instance IDs"):
        LwdTopology.from_dict(raw)


def test_single_edge_cloud_share_is_the_degenerate_pipeline_form() -> None:
    """Regression (1-edge cloud_share drill): a one-edge, one-cloud
    cloud_share deployment is the degenerate form of the shared-cloud
    pipeline and must load — the validator used to demand >= 2 edges and
    the whole deployment was rejected at config parse time."""
    raw = minimal_edge_share_payload()
    raw["deployment"]["scene"] = "cloud_share"
    # cloud_share still requires an enabled coordination block per instance.
    for instance in ("edges", "clouds"):
        for entry in raw[instance]:
            entry["coordination"] = {"enabled": True}
    raw["edges"][0]["coordination"].update(
        tenant_key_file="/home/w00505434/edge-cloud-tenant-key",
        consumer_id="enterprise-a-edge0",
        connect_timeout=5.0,
    )
    raw["clouds"][0]["coordination"]["listen_host"] = "16.32.2.2"
    raw["clouds"][0]["dp"][0].update(coord_port=9133, coord_url="http://16.32.2.2:9133")
    topology = LwdTopology.from_dict(raw)
    assert topology.deployment.scene == "cloud_share"
    (connection,) = topology.connections("edge", 0)
    assert connection.connection_id == LwdConnectionId(0, 0, 0)
    # The cloud sees the same single connection from its own side.
    assert topology.connections("cloud", 0) == (connection,)


@pytest.mark.parametrize(
    "fault,match",
    [
        ("duplicate_id", "unique and contiguous"),
        ("rank_alias", "distinct rank"),
        ("missing_link", "isolated instance"),
        ("port_collision", "different ctrl_port"),
        ("world_gap", "without gaps"),
        ("dp_mismatch", "matching DP"),
    ],
)
def test_invalid_topology_fails_before_runtime(fault: str, match: str) -> None:
    raw = cloud_reuse()
    if fault == "duplicate_id":
        raw["edges"][1]["id"] = 0
    elif fault == "rank_alias":
        for dp in raw["edges"][1]["dp"]:
            dp.update(addr=raw["edges"][0]["dp"][0]["addr"], ranks=[0])
        raw["deployment"]["hccl_world_size"] = 9
        for dp in raw["clouds"][0]["dp"]:
            dp["ranks"] = [r - 1 for r in dp["ranks"]]
    elif fault == "missing_link":
        raw["instance_links"] = [{"edge": 0, "cloud": 0}]
    elif fault == "port_collision":
        raw["clouds"][0]["dp"][1]["ctrl_port"] = 5550
    elif fault == "world_gap":
        raw["deployment"]["hccl_world_size"] += 1
    elif fault == "dp_mismatch":
        raw["edges"][1]["dp"].pop()
    with pytest.raises(ValueError, match=match):
        LwdTopology.from_dict(raw)


@pytest.mark.parametrize(
    "path,value",
    [
        (("deployment", "mode"), False),
        (("deployment", "hccl_world_size"), "10"),
        (("feature_ctrl", "scramble", "enable"), "false"),
        (("edges", 0, "dp", 0, "dp_idx"), True),
        (("edges", 0, "dp", 0, "ranks"), ["0"]),
        (("edges", 0, "dp", 0, "ctrl_port"), 6020),
        (("clouds", 0, "dp", 0, "ctrl_port"), None),
        (("clouds", 0, "dp", 0, "ctrl_port"), 65536),
        (("clouds", 0, "dp", 0, "unknown"), 1),
        (("digest",), "user-supplied"),
    ],
)
def test_schema_rejects_coercion_unknown_fields_and_wrong_port_roles(
    path: tuple[str | int, ...], value: object
) -> None:
    """Library parsing must retain strict YAML types and edge/cloud port rules."""
    raw = cloud_reuse()
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        LwdTopology.from_dict(raw)


@pytest.mark.parametrize("name", [name for name in PAYLOADS if "2dp" in name])
@pytest.mark.parametrize("fault", ["edge_overlap", "duplicate", "dp_overlap", "gap"])
def test_cloud_rank_validation_keeps_rejecting_overlap_and_gaps(
    name: str, fault: str
) -> None:
    """All scenes must reject invalid ranks after consolidating layout checks."""
    raw = topology_payload(name)
    first, second = raw["clouds"][0]["dp"]
    invalid_rank = {
        "edge_overlap": raw["edges"][0]["dp"][0]["ranks"][0],
        "duplicate": second["ranks"][1],
        "dp_overlap": first["ranks"][0],
        "gap": raw["deployment"]["hccl_world_size"],
    }
    second["ranks"][0] = invalid_rank[fault]
    match = "without gaps" if fault == "gap" else "distinct ranks"
    with pytest.raises(ValueError, match=match):
        LwdTopology.from_dict(raw)


def test_duplicate_yaml_keys_are_not_silently_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("deployment: {}\ndeployment: {}\n")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        LwdTopology.from_file(str(path))


def test_role_and_instance_lookup_does_not_fall_back() -> None:
    topology = LwdTopology.from_dict(cloud_reuse())
    with pytest.raises(ValueError, match="Instance not found"):
        topology.instance("edge", 2)
    with pytest.raises(ValueError, match="role"):
        topology.instance("unknown", 0)


@pytest.mark.parametrize(
    "fault,match",
    [
        ("different_channel", "same ctrl_port"),
        ("dp_port_alias", "different ctrl_port"),
        ("unequal_partition", "split its ranks equally"),
        ("mixed_dp_counts", "same DP count"),
    ],
)
def test_edge_share_rejects_ambiguous_channels_and_card_layouts(
    fault: str, match: str
) -> None:
    raw = topology_payload("s2_edge_share_2dp.yaml")
    if fault == "different_channel":
        raw["clouds"][1]["dp"][0]["ctrl_port"] += 10
    elif fault == "dp_port_alias":
        for cloud in raw["clouds"]:
            cloud["dp"][1]["ctrl_port"] = cloud["dp"][0]["ctrl_port"]
    elif fault == "unequal_partition":
        first, second = raw["clouds"][0]["dp"]
        second["ranks"].insert(0, first["ranks"].pop())
    elif fault == "mixed_dp_counts":
        cloud_dps = raw["clouds"][1]["dp"]
        cloud_dps[0]["ranks"].extend(cloud_dps.pop()["ranks"])
        raw["edges"][1]["dp"].pop()
    with pytest.raises(ValueError, match=match):
        LwdTopology.from_dict(raw)


@pytest.mark.parametrize("name", [name for name in PAYLOADS if "2dp" in name])
@pytest.mark.parametrize("layout", ["renumber", "multi_rank", "dp_hosts", "ports"])
def test_explicit_layout_does_not_have_to_follow_generator_defaults(
    name: str, layout: str
) -> None:
    """Validate rank ownership and endpoints, not the example's numbering/hosts."""
    raw = topology_payload(name)
    world = raw["deployment"]["hccl_world_size"]
    for instance in raw["edges"] + raw["clouds"]:
        for dp in instance["dp"]:
            if layout == "renumber":
                dp["ranks"] = [world - rank - 1 for rank in dp["ranks"]]
            elif layout == "multi_rank":
                dp["ranks"] = [
                    2 * rank + offset for rank in dp["ranks"] for offset in (0, 1)
                ]
    for cloud in raw["clouds"]:
        for dp in cloud["dp"]:
            if layout == "dp_hosts":
                dp["addr"] = f"10.1.{cloud['id']}.{dp['dp_idx'] + 1}"
            elif layout == "ports":
                dp["ctrl_port"] = 6100 + dp["dp_idx"] * 17
    if layout == "multi_rank":
        raw["deployment"]["hccl_world_size"] *= 2
    raw["clouds"].reverse()
    for edge in raw["edges"]:
        edge["dp"].reverse()
    topology = LwdTopology.from_dict(raw)
    for cloud in raw["clouds"]:
        for dp in cloud["dp"]:
            parsed = topology.instance("cloud", cloud["id"]).dp[dp["dp_idx"]]
            assert (parsed.addr, parsed.ranks, parsed.ctrl_port) == (
                dp["addr"],
                tuple(dp["ranks"]),
                dp["ctrl_port"],
            )
    for connection in topology.connections("edge", 0):
        assert (
            connection.connection_id.dp_idx
            == connection.edge.dp_idx
            == connection.cloud.dp_idx
        )


def test_edge_share_uses_explicit_one_to_one_links() -> None:
    raw = topology_payload("s2_edge_share_1dp.yaml")
    raw["instance_links"] = [{"edge": 0, "cloud": 1}, {"edge": 1, "cloud": 0}]
    topology = LwdTopology.from_dict(raw)
    assert [c.connection_id.cloud_id for c in topology.connections("edge", 0)] == [1]
    assert [c.connection_id.edge_id for c in topology.connections("cloud", 0)] == [1]


HOST_SAMPLES = [name for name in sorted(PAYLOADS) if not name.startswith("s3")]


def host_payload(name: str = "s2_edge_share_2dp.yaml") -> dict:
    """Adapt a device-comm layout to the host data plane: no ranks, no
    hccl_world_size, and one data_port per cloud DP (clouds listen, edges
    connect out)."""
    raw = topology_payload(name)
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
    return raw


@pytest.mark.parametrize("name", HOST_SAMPLES)
def test_host_layout_accepts_rank_free_topologies(name: str) -> None:
    """Host comm has no global world: every DP drops ranks and clouds listen."""
    topology = LwdTopology.from_dict(host_payload(name))
    assert topology.deployment.data_plane_comm == "host"
    for instance in (*topology.edges, *topology.clouds):
        assert all(dp.ranks == () for dp in instance.dp)
    for cloud in topology.clouds:
        assert all(dp.data_port is not None for dp in cloud.dp)
    for edge in topology.edges:
        assert all(dp.data_port is None for dp in edge.dp)


def test_host_comm_fills_the_default_qos() -> None:
    """Host comm ships one RoCE policy: no qos block means 138/4 + CS6."""
    topology = LwdTopology.from_dict(host_payload())

    assert topology.qos is not None
    assert topology.qos.data_plane.traffic_class == 138
    assert topology.qos.data_plane.sl == 4
    assert topology.qos.ctrl_plane.dscp == 48


def test_device_comm_leaves_qos_unset() -> None:
    """Device (HCCL) comm keeps the historical behaviour: nothing marked."""
    topology = LwdTopology.from_dict(topology_payload("s2_edge_share_2dp.yaml"))

    assert topology.qos is None


def test_host_comm_keeps_a_configured_plane_and_fills_the_other() -> None:
    raw = host_payload()
    raw["qos"] = {"data_plane": {"traffic_class": 106, "sl": 5}}

    topology = LwdTopology.from_dict(raw)

    assert topology.qos.data_plane.traffic_class == 106
    assert topology.qos.data_plane.sl == 5
    assert topology.qos.ctrl_plane.dscp == 48  # the omitted plane is filled


def test_host_comm_keeps_explicit_nulls_inside_a_configured_plane() -> None:
    """A written plane belongs to the operator, null fields included."""
    raw = host_payload()
    raw["qos"] = {"data_plane": {"traffic_class": 106}}

    topology = LwdTopology.from_dict(raw)

    assert topology.qos.data_plane.sl is None


def test_device_comm_keeps_a_configured_qos() -> None:
    raw = topology_payload("s2_edge_share_2dp.yaml")
    raw["qos"] = {"data_plane": {"traffic_class": 138, "sl": 4}}

    topology = LwdTopology.from_dict(raw)

    assert topology.qos.data_plane.traffic_class == 138
    assert topology.qos.ctrl_plane is None


@pytest.mark.parametrize(
    "fault,match",
    [
        ("edge_ranks", "ranks must be omitted"),
        ("cloud_ranks", "ranks must be omitted"),
        ("world_size", "hccl_world_size must be omitted"),
        ("edge_port", "data_port is cloud-only"),
        ("missing_port", "data_port is required"),
        ("duplicate_port", "duplicate data_port"),
    ],
)
def test_host_layout_rejects_ranks_world_size_and_port_violations(
    fault: str, match: str
) -> None:
    """Host layouts keep device-only fields out and need one port per cloud DP."""
    raw = host_payload()
    if fault in {"edge_ranks", "cloud_ranks"}:
        instances = raw["edges"] if fault == "edge_ranks" else raw["clouds"]
        instances[0]["dp"][0]["ranks"] = [0]
    elif fault == "world_size":
        raw["deployment"]["hccl_world_size"] = 8
    elif fault == "edge_port":
        raw["edges"][0]["dp"][0]["data_port"] = 7099
    elif fault == "missing_port":
        raw["clouds"][0]["dp"][0].pop("data_port")
    else:
        reusing = raw["clouds"][0]["dp"]
        reusing[1]["addr"] = reusing[0]["addr"]
        reusing[1]["data_port"] = reusing[0]["data_port"]
    with pytest.raises(ValueError, match=match):
        LwdTopology.from_dict(raw)


@pytest.mark.parametrize(
    "fault,match",
    [
        ("host_block", "data_plane_host is host-comm-only"),
        ("no_ranks", "ranks are required"),
        ("no_world_size", "hccl_world_size is required"),
    ],
)
def test_device_layout_requires_ranks_and_world_size(fault: str, match: str) -> None:
    """Device comm must not be left half-configured after switching planes."""
    raw = topology_payload("s2_edge_share_2dp.yaml")
    if fault == "host_block":
        raw["deployment"]["data_plane_host"] = {"up_buffer_size": 32}
    elif fault == "no_ranks":
        for instance in raw["edges"] + raw["clouds"]:
            for dp in instance["dp"]:
                dp.pop("ranks")
    else:
        raw["deployment"].pop("hccl_world_size")
    with pytest.raises(ValueError, match=match):
        LwdTopology.from_dict(raw)


def test_omitted_links_derive_deterministic_full_mesh_and_digest() -> None:
    implicit = LwdTopology.from_dict(cloud_reuse(1))
    explicit_raw = cloud_reuse(1)
    explicit_raw["instance_links"] = [
        {"edge": 1, "cloud": 0},
        {"edge": 0, "cloud": 0},
    ]
    explicit = LwdTopology.from_dict(explicit_raw)
    assert [(link.edge, link.cloud) for link in implicit.instance_links] == [
        (0, 0),
        (1, 0),
    ]
    assert set(implicit.instance_links) == set(explicit.instance_links)
    assert implicit.digest == explicit.digest


def test_edge_share_omitted_links_derive_full_mesh_then_fail_scene() -> None:
    raw = topology_payload("s2_edge_share_2dp.yaml")
    del raw["instance_links"]
    with pytest.raises(ValueError, match="one-to-one instance links"):
        LwdTopology.from_dict(raw)


@pytest.mark.parametrize(
    "links,match",
    [
        ([], "at least 1"),
        (
            [{"edge": 0, "cloud": 0}, {"edge": 0, "cloud": 0}],
            "duplicate instance link",
        ),
        ([{"edge": 0, "cloud": 2}], "unknown edge/cloud"),
    ],
)
def test_explicit_links_keep_strict_validation(links: list, match: str) -> None:
    raw = cloud_reuse(1)
    raw["instance_links"] = links
    with pytest.raises(ValueError, match=match):
        LwdTopology.from_dict(raw)


def test_coord_url_is_derived_and_available_on_every_connection() -> None:
    raw = cloud_reuse()
    expected_digest = LwdTopology.from_dict(raw).digest
    for cloud in raw["clouds"]:
        for dp in cloud["dp"]:
            del dp["coord_url"]
    topology = LwdTopology.from_dict(raw)
    assert topology.digest == expected_digest
    for connection in topology.connections("edge", 0):
        cloud = connection.cloud
        assert cloud.coord_url == f"http://{cloud.addr}:{cloud.coord_port}"
        assert cloud.coord_port == 8100 + cloud.dp_idx


def test_explicit_gateway_uses_complete_url_and_shared_topology_mode() -> None:
    raw = cloud_reuse(1)
    raw["gateway"] = {
        "enabled": True,
        "url": "https://gateway.example.com/custom/cloud?route=blue",
        "connect_timeout": 3.0,
    }
    topology = LwdTopology.from_dict(raw)
    assert topology.gateway.enabled
    assert topology.gateway.url == raw["gateway"]["url"]
    assert topology.gateway.connect_timeout == 3.0


def test_disabled_or_omitted_gateway_keeps_direct_configuration() -> None:
    omitted = cloud_reuse(1)
    disabled = cloud_reuse(1)
    disabled["gateway"] = {
        "enabled": False,
        "url": "https://gateway.example.com/custom/v1/chat/completions",
    }
    assert not LwdTopology.from_dict(omitted).gateway.enabled
    parsed = LwdTopology.from_dict(disabled)
    assert not parsed.gateway.enabled
    assert parsed.clouds[0].dp[0].coord_url == "http://10.1.0.1:8100"


def test_enabled_gateway_requires_coordination_on_every_instance() -> None:
    raw = topology_payload("s1_single_instance_1dp.yaml")
    raw["gateway"] = {
        "enabled": True,
        "url": "https://gateway.example.com/v1/chat/completions",
    }
    with pytest.raises(
        ValueError, match="enabled gateway requires enabled coordination"
    ):
        LwdTopology.from_dict(raw)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://gateway.example.com/v1/chat/completions",
        "https://gateway.example.com",
        "https://gateway.example.com/",
        "https://user:secret@gateway.example.com/v1/chat/completions",
        "https://gateway.example.com/v1/chat/completions#fragment",
        "https://gateway.example.com/v1/chat completions",
    ],
)
def test_gateway_rejects_incomplete_or_invalid_url(url: str) -> None:
    raw = cloud_reuse(1)
    raw["gateway"] = {"enabled": True, "url": url}
    with pytest.raises(ValueError, match="gateway"):
        LwdTopology.from_dict(raw)


def test_gateway_requires_url_and_rejects_legacy_urls_and_path() -> None:
    raw = cloud_reuse(1)
    url = "https://gateway.example.com/v1/chat/completions"
    raw["gateway"] = {"enabled": True}
    with pytest.raises(ValueError, match="requires a URL"):
        LwdTopology.from_dict(raw)
    raw["gateway"] = {"enabled": True, "urls": [url]}
    with pytest.raises(ValueError):
        LwdTopology.from_dict(raw)
    raw["gateway"] = {"enabled": True, "url": url, "path": "/ignored"}
    with pytest.raises(ValueError):
        LwdTopology.from_dict(raw)


def test_probe_poll_budget_yaml_config() -> None:
    raw = cloud_reuse(1)
    coordination = raw["clouds"][0]["coordination"]
    assert LwdTopology.from_dict(raw).clouds[0].coordination.probe_poll_budget == 32
    coordination["probe_poll_budget"] = 2
    assert LwdTopology.from_dict(raw).clouds[0].coordination.probe_poll_budget == 2
    coordination["probe_poll_budget"] = 0
    with pytest.raises(ValueError):
        LwdTopology.from_dict(raw)


def test_explicit_gateway_coord_url_and_ipv6_derivation() -> None:
    raw = topology_payload("s1_single_instance_1dp.yaml")
    dp = raw["clouds"][0]["dp"][0]
    dp["coord_url"] = "https://gateway.example.com:443"
    topology = LwdTopology.from_dict(raw)
    assert topology.clouds[0].dp[0].coord_url == "https://gateway.example.com:443"
    assert topology.clouds[0].dp[0].coord_port == 8100

    dp["addr"] = "2001:db8::1"
    dp["ctrl_port"] = 5550
    dp["coord_port"] = 8100
    del dp["coord_url"]
    topology = LwdTopology.from_dict(raw)
    assert topology.clouds[0].dp[0].coord_url == "http://[2001:db8::1]:8100"


def test_gateway_coord_url_can_be_shared_by_all_dps() -> None:
    raw = topology_payload("s4_lwd_cluster_2dp.yaml")
    for cloud in raw["clouds"]:
        for dp in cloud["dp"]:
            dp["coord_url"] = "https://gateway.example.com:443"
    topology = LwdTopology.from_dict(raw)
    assert {dp.coord_url for cloud in topology.clouds for dp in cloud.dp} == {
        "https://gateway.example.com:443"
    }
    assert [dp.coord_port for cloud in topology.clouds for dp in cloud.dp] == [
        dp["coord_port"] for cloud in raw["clouds"] for dp in cloud["dp"]
    ]


@pytest.mark.parametrize(
    "url,match",
    [
        ("ftp://10.1.0.1:8100", "HTTP"),
        ("http://0.0.0.0:8100", "routable"),
        ("http://bad host:8100", "whitespace"),
        ("http://10.1.0.1", "explicit port"),
        ("http://10.1.0.1:0", "explicit port"),
    ],
)
def test_coord_url_rejects_invalid_endpoint(url: str, match: str) -> None:
    raw = topology_payload("s1_single_instance_1dp.yaml")
    raw["clouds"][0]["dp"][0]["coord_url"] = url
    with pytest.raises(ValueError, match=match):
        LwdTopology.from_dict(raw)


def test_coord_url_requires_coord_port() -> None:
    raw = topology_payload("s1_single_instance_1dp.yaml")
    del raw["clouds"][0]["dp"][0]["coord_port"]
    with pytest.raises(ValueError, match="coord_url requires coord_port"):
        LwdTopology.from_dict(raw)


def test_enabled_cloud_requires_coord_port_on_every_dp() -> None:
    raw = cloud_reuse()
    dp = raw["clouds"][0]["dp"][1]
    del dp["coord_port"]
    del dp["coord_url"]
    with pytest.raises(ValueError, match="enabled cloud coordination"):
        LwdTopology.from_dict(raw)


def test_enabled_edge_requires_complete_linked_cloud_endpoints() -> None:
    raw = topology_payload("s1_single_instance_1dp.yaml")
    raw["edges"][0]["coordination"] = {
        "enabled": True,
        "tenant_key_file": "/run/secrets/edge-cloud-tenant-key",
    }
    dp = raw["clouds"][0]["dp"][0]
    del dp["coord_port"]
    del dp["coord_url"]
    with pytest.raises(ValueError, match="enabled edge coordination"):
        LwdTopology.from_dict(raw)


@pytest.mark.parametrize("field", ["control_url", "control_url_map", "listen_port"])
def test_retired_instance_coordination_fields_are_rejected(field: str) -> None:
    raw = cloud_reuse(1)
    value: Any = "http://10.1.0.1:8100" if field == "control_url" else 8100
    if field == "control_url_map":
        value = {0: "http://10.1.0.1:8100"}
    raw["edges"][0]["coordination"][field] = value
    with pytest.raises(ValueError, match="coord_port/coord_url"):
        LwdTopology.from_dict(raw)


@pytest.mark.parametrize("conflict", ["coord", "ctrl_same_dp", "ctrl_other_dp"])
def test_coord_port_collisions_are_rejected(conflict: str) -> None:
    raw = cloud_reuse()
    first, second = raw["clouds"][0]["dp"]
    if conflict == "coord":
        second["coord_port"] = first["coord_port"]
    elif conflict == "ctrl_same_dp":
        first["coord_port"] = first["ctrl_port"]
    else:
        second["coord_port"] = first["ctrl_port"]
    for dp in (first, second):
        dp["coord_url"] = f"http://{dp['addr']}:{dp['coord_port']}"
    match = "different coord_port" if conflict == "coord" else "coord_port conflicts"
    with pytest.raises(ValueError, match=match):
        LwdTopology.from_dict(raw)


def test_coord_port_cannot_collide_with_host_data_port() -> None:
    raw = host_payload("s1_single_instance_1dp.yaml")
    dp = raw["clouds"][0]["dp"][0]
    dp["coord_port"] = dp["data_port"]
    dp["coord_url"] = f"http://{dp['addr']}:{dp['coord_port']}"
    with pytest.raises(ValueError, match="coord_port conflicts with data_port"):
        LwdTopology.from_dict(raw)


def test_api_bind_base_port_defaults_to_the_reserved_band() -> None:
    """The edge_share engine IO band base defaults to 61000 (16 reserved
    ports, two per frontend, for up to 8 frontends)."""
    topology = LwdTopology.from_dict(topology_payload("s2_edge_share_2dp.yaml"))
    assert topology.deployment.api_bind_base_port == 61000


def test_api_bind_base_port_is_configurable_in_edge_share() -> None:
    raw = topology_payload("s2_edge_share_2dp.yaml")
    raw["deployment"]["api_bind_base_port"] = 62000
    topology = LwdTopology.from_dict(raw)
    assert topology.deployment.api_bind_base_port == 62000


def test_api_bind_base_port_rejected_outside_edge_share() -> None:
    """Only edge_share may configure the band base: any other scene gets a
    loud config error instead of silently ignoring it."""
    raw = topology_payload("s3_cloud_share_2dp.yaml")
    raw["deployment"]["api_bind_base_port"] = 62000
    with pytest.raises(ValueError, match="edge_share setting"):
        LwdTopology.from_dict(raw)
