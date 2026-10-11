# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent target distances and provenance controls for saved Dex3 evidence."""
import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/retarget"))
from track3_review_dex3_targets import (  # noqa: E402
    DEFAULT_TARGETS,
    array_content_digest,
    minimum_long_finger_separation,
    observation_contract,
    position_residuals,
)


def test_numeric_fingerprint_is_portable_and_detects_changed_results() -> None:
    """Storage order/endianness differ across machines; numeric changes must fail."""
    data = np.arange(6, dtype="<f8").reshape(2, 3)
    expected = array_content_digest({"values": data, "mask": np.array([True, False])})
    alternative = {
        "mask": np.array([True, False]),
        "values": np.asfortranarray(data.astype(">f8")),
    }
    assert array_content_digest(alternative) == expected
    alternative["values"][1, 2] += 1
    assert array_content_digest(alternative) != expected


def test_numeric_fingerprint_refuses_text_arrays() -> None:
    """The numeric comparison cannot silently accept arbitrary object/text arrays."""
    with pytest.raises(ValueError, match="Numeric"):
        array_content_digest({"values": np.array(["held", "observed"])})


def test_euclidean_distances_use_scaled_offsets_from_wrist() -> None:
    """Translation of the whole scene cannot change site residuals."""
    joints = np.zeros((1, 21, 3))
    joints[0, 4] = [3, 4, 0]
    joints[0, 8] = [0, 5, 12]
    joints[0, 12] = [8, 0, 15]
    robot = np.zeros((1, 4, 3))
    expected = np.array([[0, 10, 26, 34]])
    for translation in (np.zeros(3), np.array([12, -30, 2])):
        errors, targets = position_residuals(
            joints + translation, robot + translation, DEFAULT_TARGETS, 2
        )
        np.testing.assert_allclose(errors, expected)
        np.testing.assert_allclose(targets[:, 0], translation[None])


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("mutation", ["axis", "parent"])
def test_urdf_bound_refuses_changed_kinematics(
    side: str, mutation: str, tmp_path: Path
) -> None:
    """A 57 mm bound must not survive altered long-finger kinematics."""
    assets = (
        Path(__file__).resolve().parents[1]
        / "source/robotic_grounding/robotic_grounding/assets"
    )
    original = assets / f"urdfs/g1/dex3_{side}.urdf"
    assert minimum_long_finger_separation(original, side) == pytest.approx(0.057)
    xml = ET.parse(original)
    joint = xml.find(f"joint[@name='{side}_hand_middle_0_joint']")
    assert joint is not None
    if mutation == "axis":
        axis = joint.find("axis")
        assert axis is not None
        axis.attrib["xyz"] = "0 1 0"
    else:
        parent = joint.find("parent")
        assert parent is not None
        parent.attrib["link"] = f"{side}_hand_index_1_link"
    changed = tmp_path / "changed.urdf"
    xml.write(changed)
    with pytest.raises(ValueError, match="Parallel-axis|chain"):
        minimum_long_finger_separation(changed, side)


@pytest.mark.parametrize(
    "mutation", ["held_as_observed", "coverage", "finite_gap", "frame", "integer_mask"]
)
def test_observation_changes_are_refused(mutation: str) -> None:
    """A displayed pose cannot become a supported or direct observation."""
    valid = np.array([[True, True], [False, True]])
    direct = np.array([[True, True], [False, False]])
    source = {
        "frame_indices": np.array([0, 1]),
        "timestamps_s": np.array([0, 0.05]),
        "hand_valid": valid,
        "hand_directly_observed": direct,
    }
    qpos = np.zeros((2, 2, 14))
    qpos[~valid] = np.nan
    robot = {
        "frame_indices": source["frame_indices"].copy(),
        "timestamps_s": source["timestamps_s"].copy(),
        "source_hand_valid": valid.copy(),
        "source_hand_directly_observed": direct.copy(),
        "robot_valid": valid.copy(),
        "robot_qpos": qpos,
    }
    observation_contract(source, robot)
    altered = copy.deepcopy(robot)
    if mutation == "held_as_observed":
        altered["source_hand_directly_observed"][1, 1] = True
    elif mutation == "coverage":
        altered["robot_valid"][1, 0] = True
    elif mutation == "finite_gap":
        altered["robot_qpos"][1, 0] = 0
    elif mutation == "frame":
        altered["frame_indices"][1] = 2
    else:
        altered["robot_valid"] = altered["robot_valid"].astype(int)
    with pytest.raises(ValueError, match="mismatch|NaN"):
        observation_contract(source, altered)
