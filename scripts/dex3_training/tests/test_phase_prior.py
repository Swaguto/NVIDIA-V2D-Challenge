"""Policy fitting preserves source axes, actuator ordering and residual bounds."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from v2d_dex3_flashchord.phase_prior import project, schedule, model_signature
from freeze_phase_policy import freeze


def fixture():
    names = {s: [f"{s}{i}" for i in range(7)] for s in ("left", "right")}
    phase = dict(sample="grasp", approach_start_s=1.0, close_s=2.0, release_start_s=4.0)
    spec = dict(phases={"left": [phase], "right": []})
    raw = [0.05, 0, 0, 0, 0, 0] + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65]
    plan = dict(
        schema="v2d.native-phase-policy-prior.v1",
        finger_joint_names={s: list(reversed(n)) for s, n in names.items()},
        phase_actions={"grasp": raw},
    )
    return spec, plan, names


def test_prior_is_zero_outside_its_phase_and_maps_named_fingers():
    spec, plan, names = fixture()
    original = deepcopy((spec, plan))
    times = np.array([0, 1, 2, 3, 4.25, 5.0])
    actions = schedule(spec, plan, ["right", "left"], names, times)
    assert actions.shape == (len(times), 26)
    assert not actions[:, :13].any()
    assert not actions[[0, 1, 4, 5]].any()
    np.testing.assert_allclose(actions[2, 19:], [0.65, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    assert (spec, plan) == original


def test_position_and_orientation_bounds_are_vector_bounds():
    values = np.ones((3, 13)) * 10
    bounded = project(values)
    assert np.linalg.norm(bounded[:, :3], axis=1).max() <= 0.100000001
    assert np.linalg.norm(bounded[:, 3:6], axis=1).max() <= 0.350000001
    assert abs(bounded[:, 6:]).max() <= 0.65
    spec, plan, names = fixture()
    plan["phase_actions"]["grasp"][:3] = [0.1, 0.1, 0.1]
    with pytest.raises(ValueError, match="physical residual bounds"):
        schedule(spec, plan, ["right", "left"], names, np.arange(6.0))


def test_an_untested_parameter_file_cannot_be_promoted(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    path = inputs / "manifest.json"
    path.write_text(json.dumps({"files": {}}))
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            dict(
                input_manifest_sha256=source_sha,
                phase_actions={"grasp": [0.0] * 13},
                fit_reports={"grasp": {"robust_phase_pass": True}},
            )
        )
    )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            dict(
                source_manifest_sha256=source_sha,
                fitted_phase_policy_sha256="other-policy",
                native_motor_caps_verified=True,
            )
        )
    )
    with pytest.raises(ValueError, match="exact fitted policy"):
        freeze(inputs, plan, audit, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_refit_bounds_remain_physical_and_explicit():
    bounded = project(np.ones(13) * 10, profile="grasp_refit")
    assert np.linalg.norm(bounded[:3]) * 0.05 <= 0.020000001
    assert np.linalg.norm(bounded[3:6]) * 0.15 <= 0.210000001
    assert abs(bounded[6:]).max() * 0.15 <= 0.300000001
    spec, plan, names = fixture()
    plan["phase_actions"]["grasp"][6] = 1.0
    with pytest.raises(ValueError):
        schedule(spec, plan, ["right", "left"], names, np.arange(6.0))
    plan["bounds_profile"] = "grasp_refit"
    schedule(spec, plan, ["right", "left"], names, np.arange(6.0))


def test_physics_signature_survives_cohort_size_but_detects_friction_change():
    from types import SimpleNamespace

    def model(worlds, friction):
        return SimpleNamespace(
            body_mass=np.tile([0.3, 0.02], worlds),
            shape_material_mu=np.tile([friction, friction], worlds),
            gravity=np.array([0.0, 0.0, -9.81]),
            mujoco=SimpleNamespace(condim=np.tile([3, 3], worlds)),
        )

    assert model_signature(model(2, 1.25), 2) == model_signature(model(64, 1.25), 64)
    assert model_signature(model(2, 1.25), 2) != model_signature(model(2, 0.5), 2)


def test_object_frame_position_prior_rotates_with_the_grasped_object():
    spec, plan, names = fixture()
    spec["phases"]["left"][0]["object"] = 0
    spec["source_times_s"] = [0.0, 5.0]
    q = [0.0, 0.0, 2**-0.5, 2**-0.5]
    spec["smooth_robot_object_reference"] = {"0": [[0.0, 0.0, 0.0] + q] * 2}
    plan["position_frame"] = "object_reference"
    actions = schedule(spec, plan, ["right", "left"], names, np.array([0.0, 2.0, 5.0]))
    np.testing.assert_allclose(actions[1, 13:16], [0.0, 0.05, 0.0], atol=1e-8)
