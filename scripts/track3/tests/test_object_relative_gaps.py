import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1]))
from object_relative_gaps import complete_object_relative_gaps


def scene():
    t = np.arange(5) * 0.05
    poses = np.tile(np.eye(4), (5, 1, 1))
    poses[:, :3, :3] = Rotation.from_euler(
        "z", [0, 45, 90, 45, 0], degrees=True
    ).as_matrix()
    poses[:, :3, 3] = [[0, 0, 0], [0, 0.2, 0], [0, 0.3, 0], [0, 0.2, 0], [0, 0, 0]]
    q = np.zeros((5, 2, 14))
    q[:, :, 6] = 1
    q[:, 0, 0] = 0.05
    q[:, 1, :3] = [1, 1, 1]
    q[:, :, 7:] = np.arange(5)[:, None, None] * 0.1
    anchors = np.ones((5, 2), bool)
    anchors[1:4, 0] = False
    obj = dict(
        vertices=np.array([[0, 0, 0], [0, 0, 0.01]]),
        world=poses,
        observed=np.ones(5, bool),
    )
    return q, anchors, t, [obj]


def test_rotating_moving_object_gap_preserves_source_endpoints_and_fingers():
    q, a, t, objects = scene()
    old_q, old_a = q.copy(), a.copy()
    out, slots, report = complete_object_relative_gaps(q, a, t, objects)
    np.testing.assert_allclose(out[2, 0, :3], [0, 0.35, 0], atol=1e-12)
    assert Rotation.from_quat(out[2, 0, 3:7]).as_euler("xyz", degrees=True)[
        2
    ] == pytest.approx(90)
    np.testing.assert_array_equal(out[a], q[a])
    np.testing.assert_array_equal(out[:, :, 7:], q[:, :, 7:])
    np.testing.assert_array_equal(q, old_q)
    np.testing.assert_array_equal(a, old_a)
    assert slots[:, 0].tolist() == [-1, 0, 0, 0, -1]
    assert report["inferred_hand_slots"] == [3, 0]
    assert not report["contact_assumed_confirmed"]


@pytest.mark.parametrize(
    "case",
    [
        "unobserved_object",
        "ambiguous_object",
        "released",
        "long_gap",
        "different_object",
    ],
)
def test_uncertain_gaps_keep_original_interpolation(case):
    q, a, t, objects = scene()
    if case == "unobserved_object":
        objects[0]["observed"][2] = False
    elif case == "ambiguous_object":
        objects.append({k: v.copy() for k, v in objects[0].items()})
    elif case == "released":
        q[-1, 0, 0] = -0.05  # Close to the surface, inconsistent relative wrist.
    elif case == "long_gap":
        t *= 100
    else:
        other = {k: v.copy() for k, v in objects[0].items()}
        other["world"][:, 0, 3] = 0.5
        objects.append(other)
        q[-1, 0, 0] = 0.55
    out, slots, report = complete_object_relative_gaps(q, a, t, objects)
    np.testing.assert_array_equal(out, q)
    assert (slots == -1).all()
    assert not report["decisions"][0]["applied"]


def test_global_rigid_frame_change_gives_same_physical_commands():
    q, a, t, objects = scene()
    expected, slots, _ = complete_object_relative_gaps(q, a, t, objects)
    g = np.eye(4)
    r = Rotation.from_euler("xyz", [0.4, -0.3, 0.8])
    g[:3, :3], g[:3, 3] = r.as_matrix(), [0.2, -0.4, 0.7]
    changed = q.copy()
    changed[:, :, :3] = np.einsum("ij,nsj->nsi", g[:3, :3], q[:, :, :3]) + g[:3, 3]
    changed[:, :, 3:7] = (
        (r * Rotation.from_quat(q[:, :, 3:7].reshape(-1, 4))).as_quat().reshape(5, 2, 4)
    )
    objects[0]["world"] = np.einsum("ij,njk->nik", g, objects[0]["world"])
    out, new_slots, _ = complete_object_relative_gaps(changed, a, t, objects)
    np.testing.assert_allclose(
        out[:, :, :3],
        np.einsum("ij,nsj->nsi", g[:3, :3], expected[:, :, :3]) + g[:3, 3],
        atol=1e-12,
    )
    np.testing.assert_array_equal(new_slots, slots)
    delta = (
        Rotation.from_quat(out[:, :, 3:7].reshape(-1, 4)).inv()
        * r
        * Rotation.from_quat(expected[:, :, 3:7].reshape(-1, 4))
    )
    np.testing.assert_allclose(delta.magnitude(), 0, atol=1e-12)


def test_bad_rigid_object_pose_fails_before_command_changes():
    q, a, t, objects = scene()
    objects[0]["world"][1, 0, 0] = 2
    with pytest.raises(ValueError, match="finite and rigid"):
        complete_object_relative_gaps(q, a, t, objects)


def test_shared_primitive_preserves_existing_rigid_roundoff_tolerance():
    q, a, t, objects = scene()
    objects[0]["world"][2, :3, :3] += np.eye(3) * 1e-7
    out, slots, report = complete_object_relative_gaps(q, a, t, objects)
    assert report["decisions"][0]["applied"]
    assert (slots[1:4, 0] == 0).all()
    np.testing.assert_array_equal(out[a], q[a])


def test_unknown_object_poses_outside_selected_window_remain_unknown():
    q, a, _, objects = scene()
    q = np.concatenate([q[:1], q, q[-1:]])
    a = np.concatenate([a[:1], a, a[-1:]])
    poses = objects[0]["world"]
    objects[0]["world"] = np.concatenate([poses[:1], poses, poses[-1:]])
    objects[0]["world"][[0, -1]] = np.nan
    objects[0]["observed"] = np.array([False, True, True, True, True, True, False])
    before = objects[0]["world"].copy()
    out, slots, report = complete_object_relative_gaps(q, a, np.arange(7) * .05, objects)
    assert report["decisions"][0]["applied"]
    assert (slots[2:5, 0] == 0).all()
    np.testing.assert_array_equal(out[a], q[a])
    np.testing.assert_array_equal(objects[0]["world"], before)
