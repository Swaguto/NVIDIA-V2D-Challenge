"""Landmark adapters preserve mesh/skeleton data and reject ambiguous declarations."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from source_hand_landmarks import TIP_JOINT_IDS, TIP_VERTEX_IDS, convert_mano_landmarks


def fixture(layout, mirrored=False):
    rng = np.random.default_rng(42)
    vertices = rng.normal(0, .03, (2, 778, 3))
    joints = rng.normal(0, .03, (2, 21, 3))
    joints[:, TIP_JOINT_IDS] = vertices[:, TIP_VERTEX_IDS[layout]]
    if mirrored:
        vertices[..., 0] *= -1
        joints[..., 0] *= -1
    return vertices, joints


@pytest.mark.parametrize("source,target", [("manotorch21", "hamer_smplx21"), ("hamer_smplx21", "manotorch21")])
@pytest.mark.parametrize("mirrored", [False, True])
def test_explicit_conversion_roundtrip_preserves_skeleton_mesh_and_batch_axes(source, target, mirrored):
    vertices, joints = fixture(source, mirrored)
    old_vertices, old_joints = vertices.copy(), joints.copy()
    converted = convert_mano_landmarks(vertices, joints, source_layout=source, target_layout=target)
    non_tips = [i for i in range(21) if i not in TIP_JOINT_IDS]
    np.testing.assert_array_equal(converted[:, non_tips], joints[:, non_tips])
    np.testing.assert_array_equal(converted[:, TIP_JOINT_IDS], vertices[:, TIP_VERTEX_IDS[target]])
    restored = convert_mano_landmarks(vertices, converted, source_layout=target, target_layout=source)
    np.testing.assert_array_equal(restored, old_joints)
    np.testing.assert_array_equal(vertices, old_vertices)
    np.testing.assert_array_equal(joints, old_joints)
    assert not np.shares_memory(converted, joints)


def test_identity_layout_is_an_exact_copy_including_unbatched_inputs():
    vertices, joints = fixture("manotorch21")
    result = convert_mano_landmarks(vertices[0], joints[0], source_layout="manotorch21", target_layout="manotorch21")
    np.testing.assert_array_equal(result, joints[0])
    assert not np.shares_memory(result, joints)


def test_a_mislabeled_legacy_array_is_rejected_instead_of_silently_reinterpreted():
    vertices, joints = fixture("manotorch21")
    with pytest.raises(ValueError, match="Declared source layout"):
        convert_mano_landmarks(vertices, joints, source_layout="hamer_smplx21", target_layout="manotorch21")


@pytest.mark.parametrize("failure", ["unknown", "batch_axes", "nonfinite"])
def test_rejects_unknown_layout_and_invalid_geometry(failure):
    vertices, joints = fixture("manotorch21")
    source = "manotorch21"
    if failure == "unknown": source = "unspecified"
    elif failure == "batch_axes": joints = joints[0]
    elif failure == "nonfinite": vertices[0, 5, 1] = np.nan
    with pytest.raises(ValueError):
        convert_mano_landmarks(vertices, joints, source_layout=source, target_layout="hamer_smplx21")
