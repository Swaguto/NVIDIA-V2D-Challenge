"""Reconstruct an object from its video masks and metric stereo depth.

A convex depth seed bootstraps mesh-based tracking without released CAD. Once
that seed has been tracked, masked multi-view TSDF fusion refines its surface in
the same fixed body frame. Seeds are explicitly incomplete geometry. Neither
stage establishes final submission or physical acceptance.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reconstruction/modules/v2d_stereo_depth/lib"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_from_points(points, voxel_m=0.002):
    points = np.asarray(points, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 32
        or not np.isfinite(points).all()
    ):
        raise ValueError("Need at least 32 finite measured 3D points")
    if not 0.001 <= voxel_m <= 0.01:
        raise ValueError("Unsupported metric voxel size")
    _, indices = np.unique(
        np.floor(points / voxel_m).astype(np.int64), axis=0, return_index=True
    )
    points = points[np.sort(indices)]
    if len(points) < 32:
        raise ValueError("Too few independent measured surface samples")
    # Preserve thin, locally supported handles; a global density threshold
    # would preferentially discard them beside a densely sampled bowl.
    neighbors = cKDTree(points).query_ball_point(
        points, r=5 * voxel_m, return_length=True
    )
    points = points[neighbors >= 3]
    if (
        len(points) < 32
        or np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-6) < 3
    ):
        raise ValueError("Measured surface is insufficient for a volumetric seed")
    center = np.median(points, axis=0)
    mesh = trimesh.convex.convex_hull(points - center)
    if (
        not mesh.is_watertight
        or not np.isfinite(mesh.vertices).all()
        or mesh.volume <= 1e-9
        or mesh.extents.max() > 2
    ):
        raise ValueError("Invalid or implausible metric seed")
    pose = np.eye(4)
    pose[:3, 3] = center
    return mesh, pose


def points_from_depth(depth, valid, mask, K):
    selected = valid & (mask > 127) & np.isfinite(depth) & (depth > 0.1) & (depth < 2.5)
    y, x = np.nonzero(selected)
    z = depth[y, x].astype(np.float64)
    if len(z) < 200:
        return np.empty((0, 3))
    median = np.median(z)
    mad = np.median(abs(z - median))
    keep = abs(z - median) <= max(0.03, 4 * 1.4826 * mad)
    x, y, z = x[keep], y[keep], z[keep]
    return np.stack(
        [(x - K[0, 2]) * z / K[0, 0], (y - K[1, 2]) * z / K[1, 1], z], axis=1
    )


def frames(data, episode, masks, stride=1):
    import cv2
    from stereo_depth import load_rig, stereo_depth

    rig = load_rig(str(data / "meta/camera_calibration.json"))
    base = data / "videos/chunk-000"
    paths = [
        base / f"observation.images.{cam}" / f"episode_{episode:06d}.mp4"
        for cam in (rig.left, rig.right)
    ]
    captures = [cv2.VideoCapture(str(p)) for p in paths]
    if not all(c.isOpened() for c in captures):
        raise FileNotFoundError("Scored stereo videos are missing")
    try:
        index = 0
        while True:
            reads = [c.read() for c in captures]
            if reads[0][0] != reads[1][0]:
                raise ValueError("Stereo frame coverage differs")
            if not reads[0][0]:
                break
            if index % stride:
                index += 1
                continue
            path = masks / f"{index:06d}.png"
            if path.is_file():
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape != (rig.size[1], rig.size[0]):
                    raise ValueError("Mask does not match rectified stereo")
                if (mask > 127).sum() >= 200:
                    left, right = reads[0][1], reads[1][1]
                    color, _ = rig.rectify_pair(left, right)
                    depth, valid = stereo_depth(rig, left, right)
                    yield index, color, depth, valid, mask, rig
            index += 1
    finally:
        for capture in captures:
            capture.release()


def seed(data, masks, episode, name, out, stride=5):
    if out.exists():
        raise FileExistsError(out)
    candidates = []
    for index, color, depth, valid, mask, rig in frames(
        data, episode, masks / name, stride
    ):
        points = points_from_depth(depth, valid, mask, rig.P1[:3, :3])
        if len(points) >= 200:
            candidates.append((len(points), index, points))
            candidates.sort(key=lambda x: (-x[0], x[1]))
            candidates = candidates[:3]
    if not candidates:
        raise ValueError(
            "No observed mask has sufficient stereo evidence for reconstruction"
        )
    failures = []
    for count, index, points in candidates:
        try:
            mesh, pose = seed_from_points(points)
            break
        except ValueError as error:
            failures.append(str(error))
    else:
        raise ValueError("All depth seeds failed: " + repr(failures))
    out.mkdir(parents=True)
    mesh.export(out / "mesh.glb")
    report = {
        "schema": "v2d.video-object-reconstruction.v1",
        "stage": "depth_seed",
        "episode": episode,
        "object": name,
        "provided_meshes_read": False,
        "reference_pose_labels_read": False,
        "geometry_final": False,
        "submission_ready": False,
        "units": "metres",
        "body_origin_frozen": True,
        "source": "Observed object mask and calibrated stereo video; convex seed is incomplete",
        "seed_frame": index,
        "measured_points": count,
        "seed_camera_from_object": pose.tolist(),
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.faces),
        "extents_m": mesh.extents.tolist(),
        "mesh_sha256": sha(out / "mesh.glb"),
        "source_code_sha256": sha(Path(__file__)),
        "calibration_sha256": sha(data / "meta/camera_calibration.json"),
        "mask_sha256": sha(masks / name / f"{index:06d}.png"),
        "mask_report_sha256": sha(masks / "report.json"),
    }
    report["stereo_videos_sha256"] = {
        cam: sha(
            data
            / "videos/chunk-000"
            / f"observation.images.{cam}"
            / f"episode_{episode:06d}.mp4"
        )
        for cam in (rig.left, rig.right)
    }
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def refine(data, masks, poses, seed_root, episode, name, out, stride=5):
    import cv2
    import open3d as o3d

    if out.exists():
        raise FileExistsError(out)
    original = json.loads((seed_root / "manifest.json").read_text())
    tracking = json.loads((poses / f"e{episode:03d}/{name}_fp_report.json").read_text())
    if (
        original["episode"] != episode
        or original["object"] != name
        or original["provided_meshes_read"] is not False
        or original["mesh_sha256"] != sha(seed_root / "mesh.glb")
        or tracking.get("mesh_sha256") != original["mesh_sha256"]
    ):
        raise ValueError("Refinement needs tracking of this exact video-derived seed")
    voxel = float(np.clip(max(original["extents_m"]) / 128, 0.0015, 0.004))
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=voxel * 6,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    used = []
    for index, color, depth, valid, mask, rig in frames(
        data, episode, masks / name, stride
    ):
        pose_path = (
            poses
            / f"e{episode:03d}/ego_object_poses/e{episode:03d}/{name}/object_to_cam/{rig.left}/{index:06d}.json"
        )
        if not pose_path.exists():
            continue
        record = json.loads(pose_path.read_text())
        if (
            record.get("status") != "observed"
            or record.get("gt_use") != "none"
            or record.get("episode") != episode
            or record.get("frame_index") != index
            or record.get("object") != name
            or record.get("mesh_sha256") != original["mesh_sha256"]
        ):
            raise ValueError("Unbound object pose in multi-view reconstruction")
        selected = (
            valid & (mask > 127) & np.isfinite(depth) & (depth > 0.1) & (depth < 2.5)
        )
        if selected.sum() < 200:
            continue
        filtered = np.where(selected, depth, 0).astype(np.float32)
        q = np.asarray(record["rotation"], float)
        if not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1, atol=1e-4):
            raise ValueError("Invalid tracked rotation")
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        transform[:3, 3] = record["translation"]
        if not np.isfinite(transform).all() or record.get("scale") != 1:
            raise ValueError("Invalid rigid object pose")
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)),
            o3d.geometry.Image(filtered),
            depth_scale=1.0,
            depth_trunc=2.5,
            convert_rgb_to_intensity=False,
        )
        K = rig.P1[:3, :3]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            rig.size[0], rig.size[1], K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        )
        # World of this fusion is the fixed object body frame; extrinsic is camera-from-object.
        volume.integrate(rgbd, intrinsic, transform)
        used.append(
            {
                "frame": index,
                "pose_sha256": sha(pose_path),
                "mask_sha256": sha(masks / name / f"{index:06d}.png"),
            }
        )
    if len(used) < 3:
        raise ValueError("Need at least three observed masked views for refinement")
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > 12000:
        mesh = mesh.simplify_quadric_decimation(12000)
    result = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        process=False,
    )
    if (
        len(result.vertices) < 32
        or len(result.faces) < 32
        or not np.isfinite(result.vertices).all()
        or result.area <= 0
    ):
        raise ValueError("Multi-view reconstruction produced invalid geometry")
    out.mkdir(parents=True)
    result.export(out / "mesh.glb")
    report = {
        "schema": original["schema"],
        "stage": "multi_view_tsdf",
        "episode": episode,
        "object": name,
        "provided_meshes_read": False,
        "reference_pose_labels_read": False,
        "units": "metres",
        "body_origin_frozen": True,
        "seed_manifest_sha256": sha(seed_root / "manifest.json"),
        "seed_mesh_sha256": original["mesh_sha256"],
        "seed_camera_from_object": original["seed_camera_from_object"],
        "views": used,
        "voxel_m": voxel,
        "geometry_final": False,
        "submission_ready": False,
        "physical_collision_accepted": False,
        "mesh_sha256": sha(out / "mesh.glb"),
        "vertices": len(result.vertices),
        "triangles": len(result.faces),
        "extents_m": result.extents.tolist(),
        "source_code_sha256": sha(Path(__file__)),
        "tracking_report_sha256": sha(poses / f"e{episode:03d}/{name}_fp_report.json"),
        "acceptance": "Estimated multi-view surface; inspect and retrack in the unchanged body frame before final bundles",
    }
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["seed", "refine"], required=True)
    for option in ("data", "masks", "out"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--seed-root", type=Path)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("Stride must be positive")
    if args.stage == "seed":
        report = seed(
            args.data, args.masks, args.episode, args.object, args.out, args.stride
        )
    else:
        if args.poses is None or args.seed_root is None:
            parser.error("Refinement needs --poses and --seed-root")
        report = refine(
            args.data,
            args.masks,
            args.poses,
            args.seed_root,
            args.episode,
            args.object,
            args.out,
            args.stride,
        )
    print(json.dumps(report))
