import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from calibrated_anchor import calibrated_seed


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate(tmp_path):
    (tmp_path / "pointmap.npy").write_bytes(b"measured stereo pointmap")
    (tmp_path / "metric-refinement-source.py").write_bytes(b"# pinned refiner")
    (tmp_path / "raw-transform.json").write_text(json.dumps({"rotation": [1, 0, 0, 0]}))
    report = {
        "schema": "v2d.measured-surface-metric-candidate.v1",
        "episode": 3,
        "object": "pot",
        "source_manifest_sha256": "a" * 64,
        "source_mesh_sha256": "b" * 64,
        "source_pointmap_sha256": sha(tmp_path / "pointmap.npy"),
        "producer_sha256": sha(tmp_path / "metric-refinement-source.py"),
        "candidate_mesh_sha256": "c" * 64,
        "before": {
            "quality": 0.03,
            "silhouette_iou": 0.50,
            "median_measured_vertex_distance_m": 0.05,
        },
        "after": {
            "quality": 0.43,
            "silhouette_iou": 0.74,
            "median_measured_vertex_distance_m": 0.01,
        },
        "seed_minimum_support_height_after_m": 0.002,
        "seed_camera_from_object": np.eye(4).tolist(),
        "provided_meshes_read": False,
        "reference_pose_labels_read": False,
        "source_files_modified": False,
        "body_origin_changed": False,
        "body_rotation_changed": False,
        "optimizer_success": True,
        "support_plane_constraint": {"source_table_plane_changed": False},
    }
    path = tmp_path / "metric-refinement.json"
    path.write_text(json.dumps(report))
    proof = {
        "stage": "sam3d_metric_depth",
        "episode": 3,
        "object": "pot",
        "mesh_sha256": "c" * 64,
        "seed_camera_from_object": np.eye(4).tolist(),
        "source_files_sha256": {p.name: sha(p) for p in tmp_path.iterdir()},
        "metric_refinement": {
            "report_sha256": sha(path),
            "parent_manifest_sha256": "a" * 64,
            "parent_mesh_sha256": "b" * 64,
        },
    }
    return proof, report


def test_original_sam_and_tsdf_keep_their_existing_initialization(tmp_path):
    assert calibrated_seed(
        tmp_path / "manifest.json", {"stage": "sam3d_metric_depth"}
    ) == (False, None)
    assert calibrated_seed(tmp_path / "manifest.json", {"stage": "tsdf_refined"}) == (
        False,
        None,
    )


def test_bound_better_fit_retains_its_exact_body_pose(tmp_path):
    proof, _ = candidate(tmp_path)
    assert calibrated_seed(tmp_path / "manifest.json", proof) == (
        True,
        proof["metric_refinement"]["report_sha256"],
    )
    (tmp_path / "pointmap.npy").write_bytes(b"different measurements")
    with pytest.raises(ValueError, match="source closure"):
        calibrated_seed(tmp_path / "manifest.json", proof)


@pytest.mark.parametrize(
    "change", ["below_table", "worse_fit", "different_body_rotation", "other_episode"]
)
def test_unqualified_or_reframed_seed_cannot_bypass_refinement(tmp_path, change):
    proof, report = candidate(tmp_path)
    if change == "below_table":
        report["seed_minimum_support_height_after_m"] = -0.01
    if change == "worse_fit":
        report["after"]["median_measured_vertex_distance_m"] = 0.06
    if change == "different_body_rotation":
        pose = np.diag([-1, -1, 1, 1]).tolist()
        proof["seed_camera_from_object"] = pose
        report["seed_camera_from_object"] = pose
    if change == "other_episode":
        report["episode"] = 8
    path = tmp_path / "metric-refinement.json"
    path.write_text(json.dumps(report))
    proof["source_files_sha256"][path.name] = sha(path)
    proof["metric_refinement"]["report_sha256"] = sha(path)
    with pytest.raises(ValueError, match="original body frame"):
        calibrated_seed(tmp_path / "manifest.json", proof)
