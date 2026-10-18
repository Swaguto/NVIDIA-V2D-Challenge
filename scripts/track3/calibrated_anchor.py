"""Verify a measured-surface calibrated SAM seed before retaining its pose.

This chooses an initialization, not an acceptance exception. Tracking still
checks the original measured depth support and temporal pose-jump gates.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def verified_model_seed(manifest, proof):
    """Verify an original SAM model hypothesis without claiming calibrated fit.

    The caller must opt in and still apply the usual measured-depth/jump gates.
    Refined seeds use their separate, stricter calibrated-seed contract.
    """
    root = Path(manifest).resolve().parent
    if (
        proof.get("stage") != "sam3d_metric_depth"
        or proof.get("metric_refinement") is not None
        or proof.get("body_origin_frozen") is not True
        or proof.get("provided_meshes_read") is not False
        or proof.get("reference_pose_labels_read") is not False
    ):
        raise ValueError("Need an original label-free SAM model seed")
    for name in (
        "input.png",
        "mask.png",
        "pointmap.npy",
        "input-intrinsics.json",
        "raw.glb",
        "raw-transform.json",
        "mesh.glb",
    ):
        path = (root / name).resolve()
        expected = (
            proof.get("mesh_sha256")
            if name == "mesh.glb"
            else proof.get("source_files_sha256", {}).get(name)
        )
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("SAM model seed lost its measured-source closure")
    raw = json.loads((root / "raw-transform.json").read_text())
    pose = np.asarray(proof["seed_camera_from_object"], float)
    q, scale, t = (
        np.asarray(raw[k], float) for k in ("rotation", "scale", "translation")
    )
    if (
        q.shape != (4,)
        or scale.shape != (3,)
        or t.shape != (3,)
        or not np.isfinite(np.r_[q, scale, t]).all()
        or not np.isclose(np.linalg.norm(q), 1, atol=1e-5)
        or not ((scale > 0) & (scale < 10)).all()
        or pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(pose[:3, 3], t, atol=1e-6, rtol=0)
        or not np.allclose(
            pose[:3, :3],
            Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix(),
            atol=1e-5,
            rtol=0,
        )
    ):
        raise ValueError("SAM model seed changed its original body pose")
    points = np.load(root / "pointmap.npy", allow_pickle=False)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("SAM seed needs a calibrated measured pointmap")
    count = int(
        (
            np.isfinite(points).all(-1)
            & (points[..., 2] > 0.1)
            & (points[..., 2] < 2.5)
        ).sum()
    )
    if count < 200 or count != proof.get("measured_pointmap_pixels"):
        raise ValueError("SAM seed lost its original measured foreground support")
    return hashlib.sha256(Path(manifest).read_bytes()).hexdigest()


def calibrated_seed(manifest, proof):
    metadata = proof.get("metric_refinement")
    if metadata is None:
        return False, None
    root = Path(manifest).resolve().parent
    files = proof.get("source_files_sha256", {})
    for name in (
        "metric-refinement.json",
        "metric-refinement-source.py",
        "pointmap.npy",
        "raw-transform.json",
    ):
        path = (root / name).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != files.get(name)
        ):
            raise ValueError("Calibrated seed lost its measured-source closure")
    report = json.loads((root / "metric-refinement.json").read_text())
    raw = json.loads((root / "raw-transform.json").read_text())
    q = np.asarray(raw["rotation"], float)
    pose = np.asarray(proof["seed_camera_from_object"], float)
    before, after = report.get("before", {}), report.get("after", {})
    height = report.get("seed_minimum_support_height_after_m")
    values = [
        before.get("quality"),
        after.get("quality"),
        before.get("silhouette_iou"),
        after.get("silhouette_iou"),
        before.get("median_measured_vertex_distance_m"),
        after.get("median_measured_vertex_distance_m"),
        height,
    ]
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
        for v in values
    ):
        raise ValueError("Calibrated seed needs finite measured-fit diagnostics")
    if (
        proof.get("stage") != "sam3d_metric_depth"
        or report.get("episode") != proof.get("episode")
        or report.get("object") != proof.get("object")
        or report.get("schema") != "v2d.measured-surface-metric-candidate.v1"
        or metadata.get("report_sha256") != files["metric-refinement.json"]
        or report.get("source_manifest_sha256")
        != metadata.get("parent_manifest_sha256")
        or report.get("source_mesh_sha256") != metadata.get("parent_mesh_sha256")
        or report.get("source_pointmap_sha256") != files["pointmap.npy"]
        or report.get("producer_sha256") != files["metric-refinement-source.py"]
        or report.get("candidate_mesh_sha256") != proof["mesh_sha256"]
        or report.get("provided_meshes_read") is not False
        or report.get("reference_pose_labels_read") is not False
        or report.get("source_files_modified") is not False
        or report.get("body_origin_changed") is not False
        or report.get("body_rotation_changed") is not False
        or report.get("optimizer_success") is not True
        or report.get("support_plane_constraint", {}).get("source_table_plane_changed")
        is not False
        or height < 0
        or not 0 <= before["quality"] < after["quality"] <= 1
        or not 0 <= before["silhouette_iou"] <= after["silhouette_iou"] <= 1
        or not 0
        <= after["median_measured_vertex_distance_m"]
        < before["median_measured_vertex_distance_m"]
        or not np.array_equal(
            pose, np.asarray(report.get("seed_camera_from_object"), float)
        )
        or q.shape != (4,)
        or not np.isfinite(q).all()
        or not np.isclose(np.linalg.norm(q), 1, atol=1e-5)
        or not np.allclose(
            pose[:3, :3],
            Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix(),
            atol=1e-5,
            rtol=0,
        )
    ):
        raise ValueError(
            "Calibrated seed is not bound to a better measured fit in the original body frame"
        )
    return True, files["metric-refinement.json"]
