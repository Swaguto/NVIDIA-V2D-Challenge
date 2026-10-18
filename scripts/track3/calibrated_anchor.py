"""Verify a measured-surface calibrated SAM seed before retaining its pose.

This chooses an initialization, not an acceptance exception. Tracking still
checks the original measured depth support and temporal pose-jump gates.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


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
