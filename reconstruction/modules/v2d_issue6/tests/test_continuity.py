import numpy as np
import pytest

from issue6.continuity import bridge_visibility, depth_occlusion, sequence_corrections


def test_cad_does_not_exclude_foreground_gripping_hand():
    # A hand in front of a cup shares its projected silhouette but is visible.
    occ = depth_occlusion(np.array([[False, False, True]]), np.array([[0.5, 0.5, 0.5]]), np.array([[0.45, 0.5, 0.45]]))
    assert occ.tolist() == [[False, True, True]]


def test_missing_depth_uses_sequence_prior_and_rejects_outlier():
    t = np.arange(41) / 20
    offsets = np.zeros((41, 3))
    offsets[:, 2] = 0.03
    weights = np.ones(41)
    weights[15:25] = 0
    offsets[15:25] = 0
    offsets[10, 2] = 0.7
    corrected = sequence_corrections(t, offsets, weights, np.tile(np.eye(3), (41, 1, 1)))
    assert np.max(corrected[:, 2]) < 0.06
    assert np.min(corrected[15:25, 2]) > 0.015
    assert np.max(np.abs(np.diff(corrected[:, 2]))) < 0.01


def test_correction_is_smoothed_in_world_despite_rotating_camera():
    t = np.arange(31) / 20
    angle = np.linspace(0, np.pi, 31)
    R = np.zeros((31, 3, 3))
    R[:, 0, 0] = R[:, 1, 1] = np.cos(angle)
    R[:, 0, 1] = -np.sin(angle)
    R[:, 1, 0] = np.sin(angle)
    R[:, 2, 2] = 1
    world = np.tile([0.02, 0, 0], (31, 1))
    measured = np.einsum("nij,nj->ni", R, world)
    corrected = sequence_corrections(t, measured, np.ones(31), R)
    recovered = np.einsum("nji,nj->ni", R, corrected)
    np.testing.assert_allclose(recovered, world / 1.05, atol=1e-7)


def test_no_depth_does_not_fabricate_metric_correction():
    result = sequence_corrections([0, 0.1, 0.2], np.zeros((3, 3)), np.zeros(3), np.tile(np.eye(3), (3, 1, 1)))
    np.testing.assert_array_equal(result, 0)
    with pytest.raises(ValueError):
        sequence_corrections([0, 0, 1], np.zeros((3, 3)), np.ones(3), np.tile(np.eye(3), (3, 1, 1)))


def test_visibility_uses_existing_poses_and_never_extrapolates():
    observed = [False, True, False, False, True, False]
    projected = [True] * 6
    assert bridge_visibility(np.arange(6) / 20, observed, projected).tolist() == [False, True, True, True, True, False]
    projected[2] = False
    assert bridge_visibility(np.arange(6) / 20, observed, projected).tolist() == observed
    assert bridge_visibility(np.arange(6), observed, [True] * 6, 1.5).tolist() == observed
