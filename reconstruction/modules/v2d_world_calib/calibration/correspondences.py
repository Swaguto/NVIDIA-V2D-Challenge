"""Stage 1: build ``world-3D <-> image-2D`` correspondences for the rig.

The challenge provides GT object poses (mocap world) and object meshes, but no
camera-to-world extrinsic.  Correspondence sources:

* **keypoint_2d** — the primary path: per-camera observations pairing object
  mesh points with detected image keypoints (``{camera}_ep_%06d.npy``,
  ``(T, B, K, 5)`` = [obj-x, obj-y, obj-z, u, v]), turned into world points by
  applying the GT pose.  This is what Issue #5/#7 (FoundationPose/SAM) will
  populate.
* **silhouette_2d** — 2D object masks + the mesh vertex cloud; used by
  mask-based refinement/verification (no keypoint pairing needed).
* **synthetic** — generated ground-truth pairs for self-tests.

All providers return the same typed containers; nothing here depends on the
challenge parquet schema.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_W2XYZW = np.array([0, 1, 2, 4, 5, 6, 3], dtype=np.int64)


@dataclass(frozen=True)
class WorldCameraCorrespondences:
    """Paired 3D (world/metres) <-> 2D (image/pixels) observations for one camera."""

    camera: str
    points_world: np.ndarray  # (N, 3) metres
    points_image: np.ndarray  # (N, 2) pixels
    frame_ids: np.ndarray  # (N,) int provenance
    object_ids: np.ndarray  # (N,) int provenance

    @property
    def count(self) -> int:
        return int(self.points_world.shape[0])

    def __len__(self) -> int:
        return self.count


@dataclass(frozen=True)
class SilhouetteObservation:
    """A single mask observation for one object in one frame."""

    camera: str
    episode: int
    frame_id: int
    object_id: int
    mask: np.ndarray  # (H, W) bool object pixels
    points_object: np.ndarray  # (V, 3) object-frame vertex cloud (metres)
    pose_world_object: np.ndarray  # (7,) xyzw quaternion + translation


def correspondences_from_pairs(
    camera: str,
    points_world: np.ndarray,
    points_image: np.ndarray,
    frame_ids: np.ndarray | None = None,
    object_ids: np.ndarray | None = None,
) -> WorldCameraCorrespondences:
    """Typed wrapper around raw paired arrays (no validation beyond shape)."""

    pw = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    pi = np.asarray(points_image, dtype=np.float64).reshape(-1, 2)
    if pw.shape[0] != pi.shape[0]:
        raise ValueError(f"paired arrays must match: {pw.shape} vs {pi.shape}")
    n = pw.shape[0]
    frames = (
        np.zeros(n, dtype=np.int64)
        if frame_ids is None
        else np.asarray(frame_ids, dtype=np.int64).reshape(-1)
    )
    objs = (
        np.zeros(n, dtype=np.int64)
        if object_ids is None
        else np.asarray(object_ids, dtype=np.int64).reshape(-1)
    )
    return WorldCameraCorrespondences(
        camera=camera, points_world=pw, points_image=pi, frame_ids=frames, object_ids=objs
    )


def correspondences_concatenate(
    parts: list[WorldCameraCorrespondences],
    camera: str,
) -> WorldCameraCorrespondences:
    """Merge per-episode correspondence batches into one container."""

    if not parts:
        return WorldCameraCorrespondences(
            camera=camera,
            points_world=np.zeros((0, 3)),
            points_image=np.zeros((0, 2)),
            frame_ids=np.zeros(0, dtype=np.int64),
            object_ids=np.zeros(0, dtype=np.int64),
        )
    world = np.concatenate([p.points_world for p in parts], axis=0)
    image = np.concatenate([p.points_image for p in parts], axis=0)
    frames = np.concatenate([p.frame_ids for p in parts], axis=0)
    objects = np.concatenate([p.object_ids for p in parts], axis=0)
    return WorldCameraCorrespondences(
        camera=camera, points_world=world, points_image=image, frame_ids=frames, object_ids=objects
    )


# --------------------------------------------------------------------------- #
# Synthetic provider (self-tests / no challenge data)
# --------------------------------------------------------------------------- #


def synthetic_keypoint_correspondences(
    camera: str,
    intrinsics,
    rot: np.ndarray,
    trans: np.ndarray,
    object_local_points: np.ndarray,  # (B, V, 3) object-frame mesh points
    poses_world_object: np.ndarray,  # (B, 7) xyzw + t, world frame
    frames: int = 12,
    noise_px: float = 1.0,
    seed: int = 0,
) -> WorldCameraCorrespondences:
    """Project GT object points through a *known* T world->cam -> paired pixels.

    Used to prove the solver recovers the transform (self-test / tests).
    """

    rng = np.random.default_rng(seed)
    bodies = object_local_points.shape[0]
    views = object_local_points.shape[1]
    world_arr: list[np.ndarray] = []
    image_arr: list[np.ndarray] = []
    frame_arr: list[np.ndarray] = []
    obj_arr: list[np.ndarray] = []

    for t in range(frames):
        for b in range(bodies):
            q = poses_world_object[b][3:]  # xyzw
            qn = q / (np.linalg.norm(q) + 1e-12)
            rot_b = _quat_to_matrix(qn)
            pos_b = poses_world_object[b][:3]
            local = object_local_points[b]  # (V, 3)
            world = local @ rot_b.T + pos_b
            pixels = intrinsics.project(world, rot, trans)
            pixels = pixels + rng.normal(
                0.0, noise_px, size=pixels.shape
            )
            world_arr.append(world)
            image_arr.append(pixels)
            frame_arr.append(np.full(views, t, dtype=np.int64))
            obj_arr.append(np.full(views, b, dtype=np.int64))

    return correspondences_from_pairs(
        camera=camera,
        points_world=np.concatenate(world_arr, axis=0),
        points_image=np.concatenate(image_arr, axis=0),
        frame_ids=np.concatenate(frame_arr, axis=0),
        object_ids=np.concatenate(obj_arr, axis=0),
    )


def synthetic_frontal_world_cam_pose(
    intrinsics,
    points_world: np.ndarray,
    rng: np.random.Generator,
    max_tries: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """(rot, trans) of a geo-valid camera framing ``points_world`` in front of it.

    Builds a true look-at camera (z-forward toward the scene centroid) with a
    small pose perturbation, then rejects any sample that leaves the object
    behind the lens (z > margin) or outside the image.  This is what a real rig
    guarantees; invalid synthetic scenes (camera behind the object) silently
    break PnP and are rejected here.
    """

    pts = np.asarray(points_world, dtype=np.float64)
    center = pts.mean(axis=0)
    margin_z = 0.15
    for _ in range(max_tries):
        cam_pos = center + rng.uniform([-0.05, -0.05, 0.30], [0.05, 0.05, 0.60])
        base = _look_at_rot(center, cam_pos)
        aa = rng.uniform(-0.10, 0.10, size=3)
        rot = _axis_angle_to_matrix(aa) @ base
        trans = -rot @ cam_pos
        cam_pts = pts @ rot.T + trans
        if np.median(cam_pts[:, 2]) <= margin_z or np.mean(cam_pts[:, 2] > 0.0) < 0.99:
            continue
        pixels = intrinsics.project(pts, rot, trans)
        inside = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < intrinsics.width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < intrinsics.height)
        )
        if np.mean(inside) < 0.99:
            continue
        return rot, trans
    raise RuntimeError("could not synthesize a valid frontal camera pose (bad intrinsics/scene?)")


def _look_at_rot(center: np.ndarray, cam_pos: np.ndarray) -> np.ndarray:
    forward = center - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / right_norm
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=0)  # rows = camera x, y, z in world


def _axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(aa))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = np.asarray(aa, dtype=np.float64) / theta
    kx, ky, kz = k
    kmat = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64
    )
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(theta) * kmat
        + (1.0 - np.cos(theta)) * (kmat @ kmat)
    )


# --------------------------------------------------------------------------- #
# Real providers (require challenge data on the shared box)
# --------------------------------------------------------------------------- #


def load_keypoint_correspondences(
    camera: str,
    episodes: list[int],
    data_root: str | os.PathLike,
    keypoints_dir: str | os.PathLike,
    load_reference_pose=None,
) -> WorldCameraCorrespondences:
    """Read per-(camera,episode) keypoint files and wear them with GT poses.

    File layout: ``{keypoints_dir}/{camera}_ep_{episode:06d}.npy`` of shape
    ``(T, B, K, 5)`` = [obj-x, obj-y, obj-z, img-u, img-v].  GT poses are any
    callable mapping episode -> (pose (T,B,7) xyzw, visible (T,B) bool); by
    default the V2D ``reference_loader`` is imported lazily from
    ``scripts/eval``.  Visible==False frames are dropped from both sides.
    """

    if load_reference_pose is None:
        load_reference_pose = _reference_pose_loader

    base = Path(data_root)
    kp_dir = Path(keypoints_dir)
    parts: list[WorldCameraCorrespondences] = []
    for ep in episodes:
        fname = kp_dir / f"{camera}_ep_{ep:06d}.npy"
        if not fname.exists():
            raise FileNotFoundError(f"keypoint cache not found: {fname}")
        arr = np.load(fname)  # (T, B, K, 5)
        pose, visible = load_reference_pose(ep, base)
        t_count, b_count, k_count, _ = arr.shape
        world, image, frames, objs = [], [], [], []
        for t in range(t_count):
            for b in range(b_count):
                if not bool(visible[t, b]):
                    continue
                obj_pts = arr[t, b, :, :3]
                img_pts = arr[t, b, :, 3:5]
                q = pose[t, b, 3:]  # xyzw
                qn = q / (np.linalg.norm(q) + 1e-12)
                rot_b = _quat_to_matrix(qn)
                pos_b = pose[t, b, :3]
                world_pts = obj_pts @ rot_b.T + pos_b
                world.append(world_pts)
                image.append(img_pts)
                frames.append(np.full(k_count, t, dtype=np.int64))
                objs.append(np.full(k_count, b, dtype=np.int64))
        if world:
            parts.append(
                correspondences_from_pairs(
                    camera=camera,
                    points_world=np.concatenate(world, axis=0),
                    points_image=np.concatenate(image, axis=0),
                    frame_ids=np.concatenate(frames, axis=0),
                    object_ids=np.concatenate(objs, axis=0),
                )
            )
    return correspondences_concatenate(parts, camera)


def _reference_pose_loader(episode: int, data_root: str | os.PathLike):
    import sys

    repo_scripts = Path(__file__).resolve().parents[4] / "scripts" / "eval"
    if str(repo_scripts) not in sys.path:
        sys.path.insert(0, str(repo_scripts))
    from reference_loader import load_reference
    from pyarrow import parquet as _pq  # noqa: F401  (assert availability)

    ref = load_reference(episode, data_root)
    return ref.pose_xyzw, ref.visible


# --------------------------------------------------------------------------- #
# Quaternion helpers
# --------------------------------------------------------------------------- #


def _quat_to_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_to_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    return _quat_to_matrix(q_xyzw)


def quaternion_from_matrix(rot: np.ndarray) -> np.ndarray:
    """Rotation matrix -> xyzw unit quaternion."""

    r = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    m00, m01, m02 = r[0]
    m10, m11, m12 = r[1]
    m20, m21, m22 = r[2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w, x, y, z = (m12 - m21) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)