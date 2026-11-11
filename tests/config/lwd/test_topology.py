# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
from lwd_config_under_test import LwdTopology
from sample_topologies import PAYLOADS, topology_payload


def cloud_reuse(dp: int = 2) -> dict:
    return topology_payload(f"s3_cloud_share_{dp}dp.yaml")


@pytest.mark.parametrize("name", sorted(PAYLOADS))
def test_original_eight_layouts_keep_rank_and_link_contract(
    name: str, lwd_yaml_dir: Path
) -> None:
    topology = LwdTopology.from_file(str(lwd_yaml_dir / name))
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
        assert [c.cloud_id for c in topology.connections("edge", edge.id)] == [
            0
        ] * dp_count


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
        raw["instance_links"].pop()
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
        assert connection.dp_idx == connection.edge.dp_idx == connection.cloud.dp_idx


def test_edge_share_uses_explicit_one_to_one_links() -> None:
    raw = topology_payload("s2_edge_share_1dp.yaml")
    raw["instance_links"] = [{"edge": 0, "cloud": 1}, {"edge": 1, "cloud": 0}]
    topology = LwdTopology.from_dict(raw)
    assert [c.cloud_id for c in topology.connections("edge", 0)] == [1]
    assert [c.edge_id for c in topology.connections("cloud", 0)] == [1]


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
