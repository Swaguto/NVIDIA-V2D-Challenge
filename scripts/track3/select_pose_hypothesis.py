"""Select a complete object track by its agreement with observed pixels/depth.

Every hypothesis must already pass the existing tracker gates in the same own
mesh body frame. Selection copies one complete hypothesis unchanged; it never
stitches poses, fills gaps, or uses reference pose labels. Missing frames score
zero. This diagnostic score is not a pose-accuracy or contact certificate.
"""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import trimesh
from batch_public import require_automatic_mask_prompts
from reconstructed_view import FINAL_STAGES, verify_geometry
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

DEPTH_DISTANCE_SCALE_M = 0.02


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed_fit(vertices, faces, tree, pose, depth, valid, mask, K):
    """Full triangle silhouette and a robust measured-point/mesh-vertex fit."""
    if depth.shape != valid.shape or depth.shape != mask.shape or depth.ndim != 2:
        raise ValueError("Depth and foreground must share a calibrated grid")
    measured = valid & mask & np.isfinite(depth) & (depth > 0.1) & (depth < 2.5)
    y, x = np.nonzero(measured)
    if len(x) < 100:
        return {
            "quality": 0.0,
            "measured_pixels": len(x),
            "reason": "insufficient measured foreground",
        }
    # Deterministic raster-order sampling bounds cost and supplies no pose labels.
    if len(x) > 4096:
        indices = np.linspace(0, len(x) - 1, 4096, dtype=int)
        y, x = y[indices], x[indices]
    z = depth[y, x]
    points = np.c_[(x - K[0, 2]) * z / K[0, 0], (y - K[1, 2]) * z / K[1, 1], z]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        camera = vertices @ pose[:3, :3].T + pose[:3, 3]
        local = (points - pose[:3, 3]) @ pose[:3, :3]
    if not np.isfinite(camera).all() or not np.isfinite(local).all():
        raise ValueError("Nonfinite mesh/pose in observed-fit score")
    if (camera[:, 2] <= 0.01).any():
        return {
            "quality": 0.0,
            "measured_pixels": len(x),
            "reason": "mesh crosses camera near plane",
        }
    uv = np.stack(
        [
            K[0, 0] * camera[:, 0] / camera[:, 2] + K[0, 2],
            K[1, 1] * camera[:, 1] / camera[:, 2] + K[1, 2],
        ],
        axis=-1,
    )
    uv = np.clip(uv, -20000, 20000).astype(np.int32)
    silhouette = np.zeros(mask.shape, np.uint8)
    for polygon in uv[faces]:
        cv2.fillConvexPoly(silhouette, polygon, 255)
    predicted = silhouette > 127
    iou = float((predicted & mask).sum() / max(1, (predicted | mask).sum()))
    distances = tree.query(local, workers=1)[0]
    median = float(np.median(distances))
    return {
        "quality": float(iou * np.exp(-median / DEPTH_DISTANCE_SCALE_M)),
        "silhouette_iou": iou,
        "median_measured_vertex_distance_m": median,
        "measured_pixels": len(x),
    }


def choose(scores, frame_count):
    if frame_count < 1 or not scores:
        raise ValueError("Need a nonempty complete source axis and hypotheses")
    means = {}
    for name, values in scores.items():
        a = np.asarray(values)
        if (
            a.shape != (frame_count,)
            or not np.isfinite(a).all()
            or (a < 0).any()
            or (a > 1).any()
        ):
            raise ValueError("Need a finite score on every source frame, unknowns zero")
        means[name] = float(a.mean())
    winner = max(means, key=means.get)  # Stable caller order resolves exact ties.
    if means[winner] <= 0:
        raise ValueError("No measured evidence supports any track")
    return winner, means


def select(episode, name, data, observations, masks, candidates, out):
    if (
        not name.isidentifier()
        or not candidates
        or any(not re.fullmatch(r"[a-z0-9_-]+", key) for key in candidates)
    ):
        raise ValueError("Use safe, distinct object and hypothesis names")
    proof = json.loads((data / "geometry-provenance.json").read_text())
    names = proof["object_slot_names"]
    verify_geometry(data, episode, names)
    if name not in names or proof["objects"][name]["stage"] not in FINAL_STAGES:
        raise ValueError("Need final own reconstructed geometry for this object")
    mesh_path = data / "mesh" / name / (name + ".glb")
    mesh_hash = sha(mesh_path)
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    tree = cKDTree(vertices)
    m = json.loads(observations.read_text())
    if (
        m["episode"] != episode
        or m["units"] != "metres"
        or m["depth"]["encoding"] != "metres"
        or m["input_contract"].get("reference_pose_tables_read") is not False
    ):
        raise ValueError("Need matching label-free measured-metre observations")
    frame_axis = [row["id"] for row in m["frames"]]
    if frame_axis != list(range(len(frame_axis))):
        raise ValueError("Hypothesis selection requires every original source frame")
    mask_report = json.loads((masks / "report.json").read_text())
    require_automatic_mask_prompts(mask_report)
    if (
        mask_report["episode"] != episode
        or mask_report["frames"] != len(frame_axis)
        or mask_report["objects"] != names
    ):
        raise ValueError("Mask frame/object axes differ")
    cam = m["color_camera"]
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
    label = f"e{episode:03d}"
    reports, directories, scores, hashes = {}, {}, {}, {}
    for arm, root in candidates.items():
        report_path = root / label / (name + "_fp_report.json")
        report = json.loads(report_path.read_text())
        if (
            report["episode"] != episode
            or report["object"] != name
            or report["frames"] != len(frame_axis)
            or report["mesh_sha256"] != mesh_hash
            or report.get("no_arbitrary_bootstrap_rotation_registration") is not True
            or report["bootstrap_manifest_sha256"]
            != proof["objects"][name]["manifest_sha256"]
        ):
            raise ValueError(
                "Hypotheses must retain identical reconstructed mesh/body gauge and source axis"
            )
        reports[arm] = report
        directories[arm] = (
            root
            / label
            / "ego_object_poses"
            / label
            / name
            / "object_to_cam"
            / cam["id"]
        )
        scores[arm] = np.zeros(len(frame_axis))
        hashes[arm] = {"report_sha256": sha(report_path), "poses": {}}
    per_frame, input_hashes = [], {}
    for row in m["frames"]:
        frame = row["id"]
        filename = f"{frame:06d}.json"
        paths = {arm: folder / filename for arm, folder in directories.items()}
        if not any(p.is_file() for p in paths.values()):
            continue
        dp = observations.parent / row["depth_path"]
        vp = observations.parent / row["depth_validity"]
        mp = masks / name / f"{frame:06d}.png"
        depth = np.load(dp, allow_pickle=False)
        valid = cv2.imread(str(vp), 0)
        mask = cv2.imread(str(mp), 0)
        if valid is None or mask is None:
            raise ValueError("Accepted poses lost their original measured foreground")
        input_hashes[str(frame)] = {
            "depth_sha256": sha(dp),
            "validity_sha256": sha(vp),
            "mask_sha256": sha(mp),
        }
        for arm, path in paths.items():
            if not path.is_file():
                continue
            rec = json.loads(path.read_text())
            q = np.asarray(rec["rotation"], float)
            t = np.asarray(rec["translation"], float)
            if (
                rec["episode"] != episode
                or rec["frame_index"] != frame
                or rec["object"] != name
                or rec["mesh_sha256"] != mesh_hash
                or rec["status"] != "observed"
                or rec["gt_use"] != "none"
                or rec["coordinate_frame"] != "rectified_camera"
                or rec["camera_frame"] != cam["id"]
                or rec["scale"] != 1.0
                or q.shape != (4,)
                or t.shape != (3,)
                or not np.isfinite(q).all()
                or not np.isfinite(t).all()
                or not np.isclose(np.linalg.norm(q), 1, atol=1e-5)
            ):
                raise ValueError("Unbound or invalid candidate pose")
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            pose[:3, 3] = t
            fit = observed_fit(
                vertices, faces, tree, pose, depth, valid > 127, mask > 127, K
            )
            scores[arm][frame] = fit["quality"]
            per_frame.append(dict(hypothesis=arm, frame=frame, **fit))
            hashes[arm]["poses"][filename] = sha(path)
    if any(len(hashes[arm]["poses"]) != reports[arm]["observed"] for arm in candidates):
        raise ValueError("Candidate observed-pose file closure differs from its report")
    winner, means = choose(scores, len(frame_axis))
    target = (
        out / label / "ego_object_poses" / label / name / "object_to_cam" / cam["id"]
    )
    target.mkdir(parents=True, exist_ok=False)
    for filename, digest in hashes[winner]["poses"].items():
        shutil.copyfile(directories[winner] / filename, target / filename)
        if sha(target / filename) != digest:
            raise ValueError("Selected pose changed during copy")
    receipt = {
        "schema": "v2d.whole-track-observed-hypothesis-selection.v1",
        "episode": episode,
        "object": name,
        "selected_hypothesis": winner,
        "source_frame_count": len(frame_axis),
        "mean_observed_fit": means,
        "observed_pose_counts": {
            arm: len(value["poses"]) for arm, value in hashes.items()
        },
        "depth_distance_scale_m": DEPTH_DISTANCE_SCALE_M,
        "depth_scale_reason": "Existing measured-depth envelope margin; fixed before held-out validation",
        "measured_vertex_distance_scope": "Unsigned nearest raw mesh vertex; approximate surface diagnostic, not contact",
        "silhouette_scope": "Union of projected triangles; hand/other-object occlusion not modeled",
        "entire_track_copied_unchanged": True,
        "missing_observations_not_filled": True,
        "original_tracker_gates_unchanged": True,
        "reference_pose_labels_read": False,
        "provided_meshes_read": False,
        "physical_task_accepted": False,
        "official_score_claimed": False,
        "mesh_sha256": mesh_hash,
        "observations_manifest_sha256": sha(observations),
        "mask_report_sha256": sha(masks / "report.json"),
        "source_pose_hashes": hashes,
        "source_pixel_hashes": input_hashes,
        "per_frame": per_frame,
        "producer_sha256": sha(Path(__file__)),
    }
    rp = out / label / (name + "_hypothesis_selection.json")
    rp.write_text(json.dumps(receipt, indent=2) + "\n")
    report = dict(
        reports[winner],
        whole_track_hypothesis_selection_sha256=sha(rp),
        selected_hypothesis=winner,
    )
    (out / label / (name + "_fp_report.json")).write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return {"selected": winner, "mean_observed_fit": means}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--object", required=True)
    for key in ["data", "observations", "masks", "candidates", "out"]:
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    candidates = {
        k: Path(v) for k, v in json.loads(args.candidates.read_text()).items()
    }
    print(
        json.dumps(
            select(
                args.episode,
                args.object,
                args.data,
                args.observations,
                args.masks,
                candidates,
                args.out,
            ),
            indent=2,
        )
    )
