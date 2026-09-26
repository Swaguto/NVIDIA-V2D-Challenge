"""Synthetic correctness tests for v2d.world_calib (numpy + opencv only).

Run as ``python tests/test_synthetic.py`` from the module dir, or with pytest.
"""

from __future__ import annotations

import numpy as np

from v2d.world_calib.calibration.camera_geometry import CameraIntrinsics, RigCalibration
from v2d.world_calib.calibration.correspondences import (
    quaternion_to_matrix,
    synthetic_frontal_world_cam_pose,
    synthetic_keypoint_correspondences,
)
from v2d.world_calib.calibration.silhouette import (
    silhouette_iou,
    silhouette_overlap_verify,
    silhouette_render_mask,
)
from v2d.world_calib.calibration.solve import project_points, refine_reprojection, solve_camera, solve_pnp_ransac
from v2d.world_calib.calibration.verify import (
    correspondence_reprojection_errors,
    error_stats_pixels,
    pixel_error_grid_map,
    rig_consistency_rows,
)

_HEADLESS_K = CameraIntrinsics(
    fx=1024.0, fy=1024.0, cx=1014.0, cy=760.0, width=2028, height=1520
)


def _local_points(bodies: int = 3, views: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.stack([np.zeros((views, 3)) for _ in range(bodies)], axis=0)
    out[..., :2] = rng.uniform(-0.12, 0.12, size=(bodies, views, 2))
    out[..., 2] = rng.uniform(0.05, 0.25, size=(bodies, views))
    return out


def _world_poses(bodies: int = 3, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = rng.uniform([-0.25, -0.25, 0.45], [0.25, 0.25, 0.75], size=(bodies, 3))
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (bodies, 1))
    return np.concatenate([pos, quat], axis=-1)


def _world_points(local: np.ndarray, poses: np.ndarray) -> np.ndarray:
    out = []
    for b in range(poses.shape[0]):
        q = poses[b, 3:]
        out.append(local[b] @ quaternion_to_matrix(q / np.linalg.norm(q)) + poses[b, :3])
    return np.concatenate(out, axis=0)


def test_pnp_ransac_recovers_transform() -> None:
    rot, trans = synthetic_frontal_world_cam_pose(
        _HEADLESS_K, _world_points(_local_points(), _world_poses()), np.random.default_rng(2)
    )
    corr = synthetic_keypoint_correspondences(
        "camA", _HEADLESS_K, rot, trans, _local_points(), _world_poses(), frames=8, noise_px=0.5, seed=0
    )
    rot_est, trans_est, inliers = solve_pnp_ransac("camA", corr, _HEADLESS_K)
    rel = rot_est.T @ rot
    ang = float(np.degrees(np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))))
    trans_err = float(np.linalg.norm(trans_est - trans)) * 1e3
    assert ang < 1.0, f"rotation recovery failed: {ang} deg"
    assert trans_err < 100.0, f"translation recovery failed: {trans_err} mm"
    assert inliers >= corr.count * 0.90


def test_refine_reprojection_reduces_noise() -> None:
    rot, trans = synthetic_frontal_world_cam_pose(
        _HEADLESS_K, _world_points(_local_points(), _world_poses()), np.random.default_rng(3)
    )
    corr = synthetic_keypoint_correspondences(
        "camB", _HEADLESS_K, rot, trans, _local_points(), _world_poses(), frames=12, noise_px=0.8, seed=3
    )
    rot0, trans0, _ = solve_pnp_ransac("camB", corr, _HEADLESS_K)
    fit_rms = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (corr.points_image - project_points(corr.points_world, rot0, trans0, _HEADLESS_K)) ** 2,
                    axis=-1,
                )
            )
        )
    )
    rot_r, trans_r, rms, converged, iters = refine_reprojection("camB", corr, _HEADLESS_K, rot0, trans0)
    assert rms <= fit_rms + 1e-9, f"refinement made RMS worse: {fit_rms:.3f} -> {rms:.3f}"
    assert converged and iters > 0


def test_full_solve_quality_and_metrics() -> None:
    rot, trans = synthetic_frontal_world_cam_pose(
        _HEADLESS_K, _world_points(_local_points(), _world_poses()), np.random.default_rng(5)
    )
    fit = synthetic_keypoint_correspondences(
        "camA", _HEADLESS_K, rot, trans, _local_points(), _world_poses(), frames=10, noise_px=1.0, seed=5
    )
    held = synthetic_keypoint_correspondences(
        "camA", _HEADLESS_K, rot, trans, _local_points(), _world_poses(), frames=4, noise_px=1.0, seed=6
    )
    s = solve_camera("camA", fit, _HEADLESS_K)
    assert s.refine_rms_px < 2.5

    fit_errs = correspondence_reprojection_errors(fit, s.rot, s.trans, _HEADLESS_K)
    held_errs = correspondence_reprojection_errors(held, s.rot, s.trans, _HEADLESS_K)
    stats = error_stats_pixels(held_errs)
    assert stats["p95"] < 3.0 and stats["max"] < 5.0
    grid = pixel_error_grid_map(fit, fit_errs, _HEADLESS_K.width, _HEADLESS_K.height, grid=(4, 3))
    assert grid.shape == (4, 3)
    assert np.nanmean(grid) < 2.5
    quat = s.quat_xyzw
    assert abs(np.linalg.norm(quat) - 1.0) < 1e-6


def _matrix4(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = trans
    return m


def _invert4(m: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = m[:3, :3].T
    out[:3, 3] = -m[:3, :3].T @ m[:3, 3]
    return out


def test_cam_cam_consistency() -> None:
    pairs = {}
    cams = {}
    intrinsics = {"a": _HEADLESS_K, "b": _HEADLESS_K}
    local = _local_points(2)
    poses = _world_poses(2)
    world_ref = _world_points(local, poses)
    rng = np.random.default_rng(9)
    rot_a, t_a = synthetic_frontal_world_cam_pose(_HEADLESS_K, world_ref, rng)
    rot_b, t_b = synthetic_frontal_world_cam_pose(_HEADLESS_K, world_ref, rng)
    for name, rot, trans in (("a", rot_a, t_a), ("b", rot_b, t_b)):
        corr = synthetic_keypoint_correspondences(
            name, _HEADLESS_K, rot, trans, local, poses, frames=8, noise_px=0.5, seed=int(name == "b")
        )
        pairs[name] = solve_camera(name, corr, _HEADLESS_K)
    # Provided extrinsics_to_stereo_left = T_left->cam, exactly as the real rig
    # would declare them: identity for the left camera, T_b @ inv(T_a) for 'b'.
    cams["a"] = {"extrinsics_to_stereo_left": np.eye(3, 4).tolist()}
    provided_b = _matrix4(rot_b, t_b) @ _invert4(_matrix4(rot_a, t_a))
    cams["b"] = {"extrinsics_to_stereo_left": provided_b[:3].tolist()}
    rig = RigCalibration(
        intrinsics=intrinsics,
        cameras=cams,
        stereo_pairs=[{"camera": "ego", "left": "a", "right": "b", "baseline_m": 0.075}],
        raw={},
    )
    rows = rig_consistency_rows(rig, pairs)
    assert any(r.status == "consistent" for r in rows)


def test_silhouette_iou_verify() -> None:
    rot, trans = synthetic_frontal_world_cam_pose(
        _HEADLESS_K, _world_points(_local_points(1), _world_poses(1)), np.random.default_rng(11)
    )
    local = _local_points(1)[0]  # (V,3)
    pose = _world_poses(1)[0]  # (7,)
    exact_mask = silhouette_render_mask(local, pose, rot, trans, _HEADLESS_K, _HEADLESS_K.width, _HEADLESS_K.height)
    assert float(np.count_nonzero(exact_mask)) > 50
    iou_exact = silhouette_iou(exact_mask, exact_mask)
    assert iou_exact > 0.999
    rotated = rot @ np.asarray([[0.998, -0.060, 0.0], [0.060, 0.998, 0.0], [0.0, 0.0, 1.0]])
    off_mask = silhouette_render_mask(local, pose, rotated, trans, _HEADLESS_K, _HEADLESS_K.width, _HEADLESS_K.height)
    iou_off = silhouette_iou(exact_mask, off_mask)
    assert iou_off < 0.95, f"perturbed projection should not match: IoU {iou_off:.3f}"


def _main() -> None:
    tests = [
        test_pnp_ransac_recovers_transform,
        test_refine_reprojection_reduces_noise,
        test_full_solve_quality_and_metrics,
        test_cam_cam_consistency,
        test_silhouette_iou_verify,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    _main()