"""Estimate a finite support-plane hypothesis from independent stereo background."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reconstruction/modules/v2d_issue6"))
from issue6.manifest import load, pose


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_bundle(m, bundle_manifest):
    """Bind support to the bundle's actual camera gauge and calibrated pixels."""
    bundle = json.loads(bundle_manifest.read_text())
    payload = bundle_manifest.parent / bundle["payload"]["path"]
    if (
        bundle.get("schema") != "v2d.track3.episode.v1"
        or bundle["episode"] != m["episode"]
        or bundle["frames"] != [f["id"] for f in m["frames"]]
        or bundle.get("input_contract", {}).get("calibration_sha256")
        != m["input_contract"]["calibration_sha256"]
        or sha(payload) != bundle["payload"]["sha256"]
    ):
        raise ValueError(
            "Support observations and bundle have different source contracts"
        )
    c = m["color_camera"]
    intrinsics = [[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]]
    with np.load(payload, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["frame_indices"], bundle["frames"])
        np.testing.assert_allclose(data["intrinsics"], intrinsics, atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(data["image_size"], [c["width"], c["height"]])
        expected = np.stack(
            [
                np.linalg.inv(pose({"pose_path": f["camera_pose_path"]}))
                for f in m["frames"]
            ]
        )
        np.testing.assert_allclose(data["camera_c2w"], expected, atol=1e-5, rtol=1e-5)


def estimate(points, source_rows, cameras, threshold=0.008, seed=20261017):
    points = np.asarray(points, float)
    source_rows = np.asarray(source_rows)
    cameras = np.asarray(cameras, float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 1000
        or source_rows.shape != (len(points),)
        or not np.isfinite(source_rows).all()
        or not np.array_equal(source_rows, np.rint(source_rows))
        or len(np.unique(source_rows)) < 3
        or not np.isfinite(points).all()
        or not np.isfinite(cameras).all()
        or cameras.ndim != 2
        or cameras.shape[1] != 3
    ):
        raise ValueError(
            "Need finite background samples from at least three independent source frames"
        )
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(800):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        inside = np.abs(np.einsum("ni,i->n", points, normal) - normal @ a) < threshold
        if best is None or inside.sum() > best.sum():
            best = inside
    if best is None or best.sum() < 500:
        raise ValueError("No adequately measured background plane")
    center = points[best].mean(0)
    _, _, vectors = np.linalg.svd(points[best] - center, full_matrices=False)
    normal = vectors[-1]
    for _ in range(3):
        # Limit leverage from accidental distant near-plane outliers, then
        # report support using the full original measurement tolerance.
        core = abs(np.einsum("ni,i->n", points - center, normal)) < threshold / 2
        if core.sum() < 500:
            break
        center = points[core].mean(0)
        _, _, vectors = np.linalg.svd(points[core] - center, full_matrices=False)
        normal = vectors[-1]
    if normal @ (cameras.mean(0) - center) < 0:
        normal = -normal
    plane = np.r_[normal, -normal @ center]
    distances = np.einsum("ni,i->n", points, normal) + plane[3]
    inside = abs(distances) < threshold
    if len(np.unique(source_rows[inside])) < 3:
        raise ValueError("Plane support does not repeat across frames")
    axis = np.eye(3)[np.argmin(abs(normal))]
    u = np.cross(normal, axis)
    u /= np.linalg.norm(u)
    basis = np.stack([u, np.cross(normal, u)], axis=1)
    xy = np.einsum("ni,ij->nj", points[inside] - center, basis)
    cells = np.floor(xy / 0.01).astype(np.int64)
    distinct = np.unique(np.c_[cells, source_rows[inside]], axis=0)
    coordinates, counts = np.unique(distinct[:, :2], axis=0, return_counts=True)
    persistent = {tuple(cell) for cell in coordinates[counts >= 3]}
    if len(persistent) < 100:
        raise ValueError("Too little independently repeated support footprint")
    supported = np.asarray([tuple(cell) in persistent for cell in cells])
    hull = ConvexHull(xy[supported])
    polygon = xy[supported][hull.vertices]
    vertices = center + np.einsum("ni,ji->nj", polygon, basis)
    faces = np.array(
        [[0, i, i + 1] for i in range(1, len(vertices) - 1)], dtype=np.int32
    )
    summary = dict(
        plane_world=plane.tolist(),
        points=len(points),
        inliers=int(inside.sum()),
        inlier_fraction=float(inside.mean()),
        source_frames=int(len(np.unique(source_rows[inside]))),
        persistent_1cm_cells=len(persistent),
        footprint_area_m2=float(hull.volume),
        median_inlier_residual_m=float(np.median(abs(distances[inside]))),
        threshold_m=threshold,
        seed=seed,
    )
    return summary, dict(
        plane_world=plane,
        background_points_world_m=points,
        background_source_rows=source_rows,
        background_signed_plane_distance_m=distances,
        vertices_world_m=vertices,
        faces=faces,
    )


def infer(manifest, masks, out, bundle_manifest=None):
    if out.exists():
        raise FileExistsError(out)
    m = load(manifest)
    contract = m.get("input_contract", {})
    if (
        contract.get("reference_pose_tables_read") is not False
        or m["depth"]["encoding"] != "metres"
    ):
        raise ValueError(
            "Support inference requires independently reconstructed metric scored observations"
        )
    names = contract.get("official_object_slot_names", [o["id"] for o in m["objects"]])
    if not names or any(not name.isidentifier() for name in names):
        raise ValueError("Need safe official object names for background exclusion")
    c = m["color_camera"]
    rows = []
    cameras = []
    sources = []
    for frame in m["frames"][::10]:
        depth = np.load(frame["depth_path"], allow_pickle=False)
        height, width = depth.shape
        union = np.zeros(depth.shape, np.uint8)
        frame_masks = {}
        for name in names:
            path = masks / name / f"{frame['id']:06d}.png"
            if path.exists():
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape != depth.shape:
                    raise ValueError("Object mask/depth dimensions differ")
                union = np.maximum(union, mask)
                frame_masks[name] = sha(path)
        y, x = np.nonzero(union)
        if len(x) < 100:
            continue
        region = np.zeros_like(union)
        region[
            max(0, y.min() - 160) : min(height, y.max() + 161),
            max(0, x.min() - 160) : min(width, x.max() + 161),
        ] = 1
        exclusion = cv2.dilate(union, np.ones((51, 51), np.uint8)) > 0
        valid = (
            np.isfinite(depth)
            & (depth > 0.15)
            & (depth < 1.4)
            & (region > 0)
            & ~exclusion
        )
        grid = np.zeros_like(valid)
        grid[::8, ::8] = True
        valid &= grid
        y, x = np.nonzero(valid)
        z = depth[y, x]
        points = np.c_[(x - c["cx"]) * z / c["fx"], (y - c["cy"]) * z / c["fy"], z]
        camera_path = Path(frame["camera_pose_path"])
        record = json.loads(camera_path.read_text())
        if (
            record.get("gt_use") != "independent"
            or record.get("status") != "observed"
            or record.get("coordinate_frame") != "rectified_camera"
            or record.get("episode") != m["episode"]
            or record.get("frame_index") != frame["id"]
        ):
            raise ValueError(
                "Support source camera is not an independent observed frame"
            )
        camera = pose({"pose_path": str(camera_path)})
        c2w = np.linalg.inv(camera)
        world = np.einsum("ni,ji->nj", points, c2w[:3, :3]) + c2w[:3, 3]
        rows.append(np.c_[world, np.full(len(world), frame["id"])])
        cameras.append(c2w[:3, 3])
        sources.append(
            dict(
                frame=frame["id"],
                depth_sha256=sha(Path(frame["depth_path"])),
                camera_sha256=sha(camera_path),
                masks_sha256=frame_masks,
            )
        )
    if not rows:
        raise ValueError("No observed stereo background outside object masks")
    points = np.concatenate(rows)
    _, indices = np.unique(
        np.floor(points[:, :3] / 0.005).astype(np.int64), axis=0, return_index=True
    )
    points = points[np.sort(indices)]
    if len(points) > 50000:
        points = points[
            np.random.default_rng(20261017).choice(len(points), 50000, replace=False)
        ]
    report, payload = estimate(points[:, :3], points[:, 3].astype(np.int64), cameras)
    if bundle_manifest:
        verify_bundle(m, bundle_manifest)
    out.mkdir(parents=True)
    np.savez_compressed(out / "support-plane-hypothesis.npz", **payload)
    np.savez_compressed(
        out / "inferred-support-sheet.npz",
        vertices_world_m=payload["vertices_world_m"],
        faces=payload["faces"],
    )
    report.update(
        schema="v2d.stereo-background-support-hypothesis.v1",
        episode=m["episode"],
        input_manifest_sha256=sha(manifest),
        sources=sources,
        source_code_sha256=sha(Path(__file__)),
        payload_sha256=sha(out / "support-plane-hypothesis.npz"),
        provided_meshes_read=False,
        reference_pose_labels_read=False,
        gravity_is_independently_measured=False,
        physical_support_accepted=False,
        scope="Estimated finite support hypothesis from repeat background observations; physical/collision acceptance pending",
    )
    report["parameters"] = dict(
        frame_stride=10,
        pixel_stride=8,
        minimum_depth_m=0.15,
        maximum_depth_m=1.4,
        object_exclusion_kernel_px=51,
        object_roi_padding_px=160,
        voxel_m=0.005,
        maximum_points=50000,
        ransac_trials=800,
        persistent_cell_m=0.01,
        minimum_cell_source_frames=3,
        core_refit_iterations=3,
        core_refit_threshold_m=0.004,
    )
    if bundle_manifest:
        report["bundle_manifest_sha256"] = sha(bundle_manifest)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["manifest", "masks", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(infer(args.manifest, args.masks, args.out, args.bundle_manifest)))
