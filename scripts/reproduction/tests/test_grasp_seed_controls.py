"""Robot proposals are explicit and bounded, and cannot change source objects."""

import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _dex3_static_load import (
    acquisition_pairs,
    acquisition_support,
    apply_grasp_seeds,
    cooked_acquisition_support,
)
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


def test_larger_robot_posture_correction_requires_declared_bounds():
    samples, proposal = example()
    proposal["results"][0]["candidate"]["finger_target_rad"] = [1.0] * 7
    proposal["results"][0]["candidate"]["hand_quaternion_stage_xyzw"] = (
        Rotation.from_rotvec([0, 0, 1.0]).as_quat().tolist()
    )
    with pytest.raises(ValueError, match="bounds"):
        apply_grasp_seeds(samples, proposal)
    proposal.update(finger_delta_cap_rad=1.2, wrist_rotation_cap_rad=1.2)
    result, _ = apply_grasp_seeds(samples, proposal)
    assert result[0]["finger_target_rad"] == [1.0] * 7
    proposal["finger_delta_cap_rad"] = 1.3
    with pytest.raises(ValueError, match="finger correction"):
        apply_grasp_seeds(samples, proposal)


def test_squeeze_effort_is_bounded_named_and_never_added_to_the_baseline():
    samples, proposal = example()
    proposal["grip_effort"] = {"schema": "v2d.dex3-jacobian-grip-effort.v1"}
    proposal["results"][0]["candidate"]["finger_feedforward_Nm"] = [0.4] * 7
    result, _ = apply_grasp_seeds(samples, proposal)
    assert result[0]["finger_feedforward_Nm"] == [0.4] * 7
    assert "finger_feedforward_Nm" not in samples[0]
    proposal["results"][0]["candidate"]["finger_feedforward_Nm"][0] = 0.56
    with pytest.raises(ValueError, match="squeeze feedforward"):
        apply_grasp_seeds(samples, proposal)


def test_declared_finger_stiffness_override_is_bounded_and_preserves_effort():
    samples, proposal = example()
    proposal["results"][0]["candidate"]["finger_stiffness_Nm_per_rad"] = 8.0
    result, _ = apply_grasp_seeds(samples, proposal)
    assert result[0]["finger_stiffness_Nm_per_rad"] == 8.0
    assert "finger_stiffness_Nm_per_rad" not in samples[0]
    assert "finger_feedforward_Nm" not in result[0]
    proposal["results"][0]["candidate"]["finger_stiffness_Nm_per_rad"] = 8.01
    with pytest.raises(ValueError, match="finger stiffness"):
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


def test_grip_comparison_has_identical_geometry_and_only_changes_motor_effort():
    base, _ = example()
    originals = []
    for i in range(4):
        row = copy.deepcopy(base[0])
        row["prefix"] = f"sample{i}"
        originals.append(row)
    seeded = copy.deepcopy(originals)
    for i in (2, 3):
        seeded[i].update(
            robot_grasp_seed_is_experimental=True,
            finger_pregrasp_rad=[-0.1] * 7,
            finger_feedforward_Nm=[0.4] * 7,
        )
    result, _ = acquisition_pairs(originals, seeded)
    assert [s["variant"] for s in result] == ["position-only", "squeeze"] * 2
    for baseline, squeeze in [(result[0], result[1]), (result[2], result[3])]:
        for sample in [baseline, squeeze]:
            assert sample["finger_initial_rad"] == [-0.1] * 7
            assert sample["finger_target_rad"] == [0.0] * 7
        np.testing.assert_allclose(
            np.asarray(baseline["hand_position_stage_m"])
            - baseline["object_position_stage_m"],
            np.asarray(squeeze["hand_position_stage_m"])
            - squeeze["object_position_stage_m"],
        )
        assert (
            baseline["hand_quaternion_stage_xyzw"]
            == squeeze["hand_quaternion_stage_xyzw"]
        )
        assert baseline["finger_feedforward_Nm"] == [0.0] * 7
        assert squeeze["finger_feedforward_Nm"] == [0.4] * 7


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


def test_tray_matches_cooked_surface_when_cooking_changes_the_bottom():
    raw = np.array(list(product([-0.1, 0.1], [-0.08, 0.08], [-0.005, 0.005])))
    rotation = Rotation.from_euler("xyz", [0.2, 0.1, 0.3])
    position = np.array([0.3, 0.2, -0.5])
    tray = acquisition_support(raw, position, rotation.as_quat())
    cooked = raw.copy()
    cooked[cooked[:, 2] < 0, 2] += 0.0015
    world = rotation.apply(cooked) + position
    adjusted = cooked_acquisition_support(world, tray)
    local = (
        Rotation.from_quat(adjusted["quaternion_stage_xyzw"])
        .inv()
        .apply(world - adjusted["position_stage_m"])
    )
    assert (local[:, 2] - adjusted["dimensions_m"][2] / 2).min() == pytest.approx(
        0.0001, abs=1e-10
    )
    assert np.linalg.norm(
        np.array(adjusted["position_stage_m"]) - tray["position_stage_m"]
    ) == pytest.approx(0.0015, abs=1e-10)


def test_long_handle_does_not_force_a_steep_global_hull_tray():
    body = np.array(list(product([-0.1, 0.1], [-0.1, 0.1], [-0.005, 0.005])))
    handle = np.array(list(product([-0.01, 0.01], [-0.4, -0.1], [0.10, 0.12])))
    vertices = np.r_[body, handle]
    original = dict(
        position_stage_m=[0, -0.15, -0.0151],
        quaternion_stage_xyzw=[0, 0, 0, 1],
        dimensions_m=[0.25, 0.55, 0.02],
    )
    tray = cooked_acquisition_support(vertices, original)
    np.testing.assert_allclose(tray["quaternion_stage_xyzw"], [0, 0, 0, 1], atol=1e-10)
    np.testing.assert_allclose(
        tray["position_stage_m"], original["position_stage_m"], atol=1e-10
    )
    assert (vertices[:, 2] - tray["position_stage_m"][2] - 0.01).min() == pytest.approx(
        0.0001
    )
