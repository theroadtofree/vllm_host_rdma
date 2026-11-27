# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Increment-contract unit tests of the runtime topology update.

Every accepted update must be a pure addition in its scene's shape —
edge_share grows by instance pairs, cloud_share by edges; everything
else is rejected with a reason that names the broken rule and the exact
field. The engine never sees a partial update: the caller parses with
the standard validation first, so count consistency, port uniqueness
and id contiguity failures surface at parse time, not here.
"""

from copy import deepcopy
from typing import Any

import pytest
from lwd_config_under_test import (
    LwdConnectionId,
    LwdTopology,
    LwdTopologyIncrement,
    LwdTopologyUpdateRejected,
    diff_topology,
)


def _payload(instances: int) -> dict[str, Any]:
    """An N-pair edge_share host-comm topology; instance i carries its own
    cloud ports so modifications are easy to spot in rejection reasons."""
    return {
        "deployment": {
            "mode": 0,
            "scene": "edge_share",
            "edges_num": instances,
            "clouds_num": instances,
            "data_plane_comm": "host",
        },
        "feature_ctrl": {
            "enable_early_recv": False,
            "scramble": {"enable": False, "device_id": None, "custom_factor": None},
        },
        "edges": [
            {"id": i, "dp": [{"dp_idx": 0, "addr": "135.150.2.2"}]}
            for i in range(instances)
        ],
        "clouds": [
            {
                "id": i,
                "dp": [
                    {
                        "dp_idx": 0,
                        "addr": "16.32.2.2",
                        "ctrl_port": 5550 + i,
                        "data_port": 5560 + i,
                    }
                ],
            }
            for i in range(instances)
        ],
        "instance_links": [{"edge": i, "cloud": i} for i in range(instances)],
    }


def _rejected(old: dict, new: dict) -> str:
    verdict = diff_topology(LwdTopology.from_dict(old), LwdTopology.from_dict(new))
    assert isinstance(verdict, LwdTopologyUpdateRejected)
    return verdict.reason


def test_an_unchanged_file_is_a_no_op() -> None:
    verdict = diff_topology(
        LwdTopology.from_dict(_payload(2)), LwdTopology.from_dict(_payload(2))
    )
    assert not isinstance(verdict, LwdTopologyUpdateRejected)
    assert verdict.new_edge_ids == ()
    assert verdict.new_cloud_ids == ()
    assert verdict.new_connections == ()


def test_adding_one_pair_yields_the_new_connections() -> None:
    new = LwdTopology.from_dict(_payload(2))
    verdict = diff_topology(LwdTopology.from_dict(_payload(1)), new)
    assert verdict is not None
    assert not isinstance(verdict, LwdTopologyUpdateRejected)
    assert verdict.new_edge_ids == (1,)
    assert verdict.new_cloud_ids == (1,)
    (connection,) = verdict.new_connections
    assert connection.connection_id == LwdConnectionId(1, 1, 0)
    assert connection == new.connections("edge", 1)[0]


def test_adding_two_pairs_at_once_is_accepted() -> None:
    verdict = diff_topology(
        LwdTopology.from_dict(_payload(1)), LwdTopology.from_dict(_payload(3))
    )
    assert not isinstance(verdict, LwdTopologyUpdateRejected)
    assert verdict.new_edge_ids == (1, 2)
    assert len(verdict.new_connections) == 2


def test_scene_change_is_rejected() -> None:
    new = _payload(1)
    new["deployment"]["scene"] = "single_instance"
    assert "scene changed" in _rejected(_payload(1), new)


def test_frozen_deployment_field_is_rejected() -> None:
    new = _payload(2)
    new["deployment"]["api_bind_base_port"] = 62000
    assert "deployment.api_bind_base_port" in _rejected(_payload(1), new)


def test_frozen_feature_ctrl_is_rejected() -> None:
    new = _payload(2)
    new["feature_ctrl"]["enable_early_recv"] = True
    assert "feature_ctrl.enable_early_recv" in _rejected(_payload(1), new)


def test_qos_change_is_exempt_by_deployment_guarantee() -> None:
    """2026-11-18 review decision: qos is guaranteed unmodified by the
    deployment discipline, so the contract does not compare it. The test
    locks that exemption; adding the comparison later must flip it."""
    old = _payload(1)
    old["qos"] = {"data_plane": {"traffic_class": 138, "sl": 3}}
    new = _payload(2)
    new["qos"] = {"data_plane": {"traffic_class": 100, "sl": 1}}
    verdict = diff_topology(LwdTopology.from_dict(old), LwdTopology.from_dict(new))
    assert not isinstance(verdict, LwdTopologyUpdateRejected)
    assert verdict.new_edge_ids == (1,)


def test_existing_instance_change_is_rejected() -> None:
    new = _payload(2)
    new["clouds"][0]["dp"][0]["ctrl_port"] = 5599
    assert "cloud 0.dp[0].ctrl_port" in _rejected(_payload(1), new)


def test_existing_coordination_change_is_rejected() -> None:
    old = _payload(1)
    old["edges"][0]["coordination"] = {"connect_timeout": 5.0}
    new = _payload(2)
    new["edges"][0]["coordination"] = {"connect_timeout": 6.0}
    assert "edge 0.coordination.connect_timeout" in _rejected(old, new)


def test_adding_a_cloud_needs_no_existing_edge_edit() -> None:
    """No R2 exemption on this schema: a new cloud's coordination URL
    lives in the cloud's own DP block (``coord_url``/``coord_port``), so
    growing the plan never touches an existing edge — the strict
    byte-identical R2 is naturally satisfiable. (The old-branch
    ``control_url_map`` exemption targeted a field this schema retired;
    ``LwdCoordination.reject_legacy_fields`` rejects it at parse time.)"""
    verdict = diff_topology(
        LwdTopology.from_dict(_payload(1)), LwdTopology.from_dict(_payload(2))
    )
    assert isinstance(verdict, LwdTopologyIncrement)
    assert verdict.new_cloud_ids == (1,)


def test_shrink_is_rejected() -> None:
    # R2 walks edges before clouds, so the first missing instance names the
    # rule deterministically.
    assert "instance removed: edge 1" in _rejected(_payload(2), _payload(1))


def test_link_swap_is_rejected() -> None:
    new = _payload(2)
    new["instance_links"] = [{"edge": 0, "cloud": 1}, {"edge": 1, "cloud": 0}]
    assert "link removed: edge 0 - cloud 0" in _rejected(_payload(2), new)


def test_id_gap_fails_at_parse_not_in_the_diff() -> None:
    """Count consistency and id contiguity are standard validation: the
    engine never reaches the diff with such a file."""
    raw = _payload(1)
    raw["deployment"]["edges_num"] = 2
    raw["deployment"]["clouds_num"] = 2
    raw["edges"].append({"id": 3, "dp": [{"dp_idx": 0, "addr": "135.150.2.2"}]})
    raw["clouds"].append(
        {
            "id": 3,
            "dp": [
                {
                    "dp_idx": 0,
                    "addr": "16.32.2.2",
                    "ctrl_port": 5553,
                    "data_port": 5563,
                }
            ],
        }
    )
    raw["instance_links"].append({"edge": 3, "cloud": 3})
    with pytest.raises(ValueError, match="unique and contiguous"):
        LwdTopology.from_dict(raw)


def test_a_deepcopied_update_is_still_a_no_op() -> None:
    """The digest fast path compares semantics, not object identity."""
    old = LwdTopology.from_dict(_payload(1))
    new = LwdTopology.from_dict(deepcopy(_payload(1)))
    verdict = diff_topology(old, new)
    assert not isinstance(verdict, LwdTopologyUpdateRejected)
    assert verdict.new_connections == ()


def test_frontend_digest_is_stable_across_additions() -> None:
    """Item 11: the compared digest covers everything except the
    instance inventory — an elastic addition must NOT flip it (a
    frontend launched with the grown yaml still attaches to the primary
    that started with the smaller one), while the full digest moves."""
    one = LwdTopology.from_dict(_payload(1))
    two = LwdTopology.from_dict(_payload(2))
    three = LwdTopology.from_dict(_payload(3))
    assert one.frontend_digest == two.frontend_digest == three.frontend_digest
    assert len({one.digest, two.digest, three.digest}) == 3


def test_frontend_digest_tracks_the_frozen_blocks() -> None:
    """A change outside the instance inventory still flips the compared
    digest: the attach refuses a genuinely different deployment."""
    base = LwdTopology.from_dict(_payload(2))
    moved_port = _payload(2)
    moved_port["deployment"]["api_bind_base_port"] = 62000
    assert LwdTopology.from_dict(moved_port).frontend_digest != base.frontend_digest
    qos_moved = _payload(2)
    qos_moved["qos"] = {"data_plane": {"traffic_class": 100, "sl": 1}}
    assert LwdTopology.from_dict(qos_moved).frontend_digest != base.frontend_digest


# --------------------------------------------------------------------- #
# cloud_share: additions are edges only, linked to the existing cloud      #
# --------------------------------------------------------------------- #


def _cloud_share_payload(edges: int) -> dict[str, Any]:
    """An N-edge, one-cloud cloud_share host topology (coordination is a
    cloud_share requirement on every instance)."""
    return {
        "deployment": {
            "mode": 0,
            "scene": "cloud_share",
            "edges_num": edges,
            "clouds_num": 1,
            "data_plane_comm": "host",
        },
        "feature_ctrl": {
            "enable_early_recv": False,
            "scramble": {"enable": False, "device_id": None, "custom_factor": None},
        },
        "edges": [
            {
                "id": i,
                "coordination": {
                    "enabled": True,
                    "consumer_id": f"enterprise-a-edge{i}",
                },
                "dp": [{"dp_idx": 0, "addr": "135.150.2.2"}],
            }
            for i in range(edges)
        ],
        "clouds": [
            {
                "id": 0,
                "coordination": {"enabled": True},
                "dp": [
                    {
                        "dp_idx": 0,
                        "addr": "16.32.2.2",
                        "ctrl_port": 5550,
                        "data_port": 5560,
                        "coord_port": 9133,
                        "coord_url": "http://16.32.2.2:9133",
                    }
                ],
            }
        ],
        # Full mesh: every edge links to the single cloud.
        "instance_links": [{"edge": i, "cloud": 0} for i in range(edges)],
    }


def test_cloud_share_adds_edges_to_the_existing_cloud() -> None:
    """The cloud_share shape: a new edge only, linked to the SAME existing
    cloud — exactly the crossing edge_share forbids. The increment's
    connections are the new edge's zipped DP pairs against cloud 0."""
    new = LwdTopology.from_dict(_cloud_share_payload(2))
    verdict = diff_topology(LwdTopology.from_dict(_cloud_share_payload(1)), new)
    assert not isinstance(verdict, LwdTopologyUpdateRejected)
    assert verdict.new_edge_ids == (1,)
    assert verdict.new_cloud_ids == ()
    (connection,) = verdict.new_connections
    assert connection == new.connections("edge", 1)[0]
    assert connection.connection_id.cloud_id == 0


def test_cloud_share_rejects_a_new_cloud() -> None:
    """The shared cloud's own plan never grows at runtime: a second cloud
    is a scene-shape change (cloud_share demands one cloud), so it fails
    the standard validation before the diff."""
    raw = _cloud_share_payload(1)
    raw["clouds"].append(
        {
            "id": 1,
            "coordination": {"enabled": True},
            "dp": [
                {
                    "dp_idx": 0,
                    "addr": "16.32.2.3",
                    "ctrl_port": 5551,
                    "data_port": 5561,
                    "coord_port": 9133,
                    "coord_url": "http://16.32.2.3:9133",
                }
            ],
        }
    )
    # Keep the count declarations consistent so the count-consistency
    # validation stays out of the way and the SCENE rule answers.
    raw["deployment"]["clouds_num"] = 2
    with pytest.raises(ValueError, match="one cloud"):
        LwdTopology.from_dict(raw)


def test_cloud_share_rejects_a_scene_flip() -> None:
    assert "scene changed" in _rejected(_cloud_share_payload(1), _payload(1))


def test_single_instance_has_no_runtime_update_semantics() -> None:
    """A single_instance deployment has no runtime-update semantics even
    when both files carry the scene and the content moved (a qos change,
    say) — restarting is the only way to change that plan. A scene FLIP
    is a different rule, already covered by R0's scene-change rejection
    in test_scene_change_is_rejected."""
    single = _payload(1)
    single["deployment"]["scene"] = "single_instance"
    changed = deepcopy(single)
    changed["qos"] = {
        "data_plane": {"traffic_class": 139, "sl": 3},
        "ctrl_plane": {"dscp": 46},
    }
    assert "no runtime-update semantics" in _rejected(single, changed)
