"""Robot proposals are explicit and bounded, and cannot change source objects."""

import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _dex3_static_load import acquisition_pairs, acquisition_support, apply_grasp_seeds
from scipy.spatial.transform import Rotation
from itertools import product


def example():
    sample = dict(
        prefix="right-object1-grasp0",
        side="right",
        object=1,
        source_frame=47,
        hand_position_stage_m=[0, 0, 0],
        hand_quaternion_stage_xyzw=[0, 0, 0, 1],
        object_position_stage_m=[0, 0, 0.1],
        object_quaternion_stage_xyzw=[0, 0, 0, 1],
        joint_names=[f"joint{i}" for i in range(7)],
        finger_target_rad=[0.0] * 7,
    )
    candidate = dict(
        hand_position_stage_m=[0.002, 0, 0],
        hand_quaternion_stage_xyzw=[0, 0, 0, 1],
        joint_names=sample["joint_names"],
        finger_target_rad=[0.1] * 7,
        pad_escape_direction_dot=-0.95,
        maximum_piece_penetration_lower_bound_m=0.001,
    )
    proposal = dict(
        schema="v2d.dex3-antipodal-seed.v1",
        physical_task_accepted=False,
        results=[
            dict(
                sample=sample["prefix"],
                source_sample=copy.deepcopy(sample),
                candidate=candidate,
            )
        ],
    )
    return [sample], proposal


def test_seed_changes_robot_only_and_never_mutates_original():
    samples, proposal = example()
    before = copy.deepcopy(samples)
    result, changed = apply_grasp_seeds(samples, proposal)
    assert samples == before
    assert changed == [samples[0]["prefix"]]
    np.testing.assert_array_equal(
        result[0]["object_position_stage_m"], before[0]["object_position_stage_m"]
    )
    assert result[0]["robot_grasp_seed_is_experimental"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("hand_position_stage_m", [0.04, 0, 0]),
        ("finger_target_rad", [0.61] * 7),
        ("pad_escape_direction_dot", 0.8),
        ("maximum_piece_penetration_lower_bound_m", float("nan")),
    ],
)
def test_bad_seeds_fail_before_mutation(key, value):
    samples, proposal = example()
    proposal["results"][0]["candidate"][key] = value
    before = copy.deepcopy(samples)
    with pytest.raises(ValueError):
        apply_grasp_seeds(samples, proposal)
    assert samples == before


def test_source_pose_mismatch_is_rejected():
    samples, proposal = example()
    proposal["results"][0]["source_sample"]["object_position_stage_m"][0] += 0.001
    with pytest.raises(ValueError, match="source pose"):
        apply_grasp_seeds(samples, proposal)


def test_acquisition_compares_matching_sources_with_same_initial_fingers():
    base, _ = example()
    originals = []
    for i in range(4):
        row = copy.deepcopy(base[0])
        row["prefix"] = f"sample{i}"
        row["hand_position_stage_m"][0] = 3.0 * i
        row["object_position_stage_m"][0] = 3.0 * i
        originals.append(row)
    seeded = copy.deepcopy(originals)
    for i in (2, 3):
        seeded[i]["hand_position_stage_m"][0] += 0.002
        seeded[i]["finger_target_rad"] = [0.1] * 7
        seeded[i]["robot_grasp_seed_is_experimental"] = True
    result, changed = acquisition_pairs(originals, seeded)
    assert len(result) == 4 and len(changed) == 2
    assert [s["variant"] for s in result] == ["baseline", "antipodal"] * 2
    for i, sample in enumerate(result):
        assert sample["finger_initial_rad"] == [0.0] * 7
        assert sample["object_position_stage_m"][0] == 3.0 * i
        relative_x = (
            sample["hand_position_stage_m"][0] - sample["object_position_stage_m"][0]
        )
        assert relative_x == pytest.approx(0.002 if i % 2 else 0)


def test_tilted_tray_is_below_every_source_mesh_vertex():
    vertices = np.array(list(product([-0.1, 0.1], [-0.08, 0.08], [-0.005, 0.005])))
    rotation = Rotation.from_euler("xyz", [0.2, 0.1, 0.3])
    position = np.array([0.3, 0.2, -0.5])
    tray = acquisition_support(vertices, position, rotation.as_quat())
    orientation = Rotation.from_quat(tray["quaternion_stage_xyzw"])
    world = rotation.apply(vertices) + position
    tray_local = orientation.inv().apply(world - tray["position_stage_m"])
    dimensions = np.array(tray["dimensions_m"])
    assert (np.abs(tray_local[:, :2]) <= dimensions[:2] / 2).all()
    assert (tray_local[:, 2] - dimensions[2] / 2).min() == pytest.approx(
        0.0001, abs=1e-10
    )
