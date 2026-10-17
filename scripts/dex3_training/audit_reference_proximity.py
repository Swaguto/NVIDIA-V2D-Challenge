"""Screen preserved human meshes for possible contact-reference preparation.

Nearest mesh-vertex distance is an upper bound on surface distance. This audit
does not infer contact labels, surface normals, penetration or contact forces.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def audit(bundle, radius_m=0.005):
    if not np.isfinite(radius_m) or not 0 < radius_m <= 0.01:
        raise ValueError("Need a finite proximity-screen radius within 10 mm")
    manifest = json.loads((bundle.parent / "manifest.json").read_text())
    if manifest["units"] != "metres":
        raise ValueError("Reference mesh units must be metres")
    with np.load(bundle, allow_pickle=False) as data:
        hands = data["hand_vertices_world"].astype(np.float64)
        valid = data["hand_valid"].astype(bool)
        observed = data["hand_directly_observed"].astype(bool)
        frames = data["frame_indices"].copy()
        trees, poses, scales = [], [], []
        object_id = 0
        while f"object_{object_id}_vertices" in data:
            vertices = data[f"object_{object_id}_vertices"].astype(np.float64)
            pose = data[f"object_{object_id}_world"].astype(np.float64)
            scale = data[f"object_{object_id}_scale"].astype(np.float64)
            if not np.isfinite(vertices).all() or not np.isfinite(pose).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("Invalid source object geometry/pose/scale")
            if not np.allclose(pose[:, :3, :3].swapaxes(1, 2) @ pose[:, :3, :3], np.eye(3), atol=1e-5) or not np.allclose(np.linalg.det(pose[:, :3, :3]), 1, atol=1e-5):
                raise ValueError("Source object poses must contain proper rotations")
            trees.append(cKDTree(vertices))
            poses.append(pose)
            scales.append(scale)
            object_id += 1
        measured_contacts = int(np.count_nonzero(data["contact_known"]))
    if hands.shape[:2] != valid.shape or valid.shape != observed.shape or len(frames) != len(hands):
        raise ValueError("Source hand validity and mesh clocks differ")
    series = []
    for frame in range(len(hands)):
        row = dict(frame=int(frames[frame]), hands={})
        for side_index, side in enumerate(manifest["hand_sides"]):
            vertices = hands[frame, side_index]
            usable = bool(valid[frame, side_index] and np.isfinite(vertices).all())
            samples = []
            if usable:
                for obj, tree in enumerate(trees):
                    pose, scale = poses[obj][frame], scales[obj][frame]
                    # Explicit small transforms avoid platform BLAS FP-status
                    # warnings; source NaNs remain excluded by the validity mask.
                    local = np.einsum("vj,jk->vk", vertices - pose[:3, 3], pose[:3, :3]) / scale
                    distance, _ = tree.query(local)
                    distance *= scale
                    samples.append(dict(object=obj, minimum_mesh_vertex_distance_mm=float(distance.min() * 1000), vertices_within_screen_radius=int(np.count_nonzero(distance <= radius_m))))
            row["hands"][side] = dict(valid_source_geometry=usable, directly_observed=bool(observed[frame, side_index]), proximity=samples)
        series.append(row)
    summary = {}
    for side in manifest["hand_sides"]:
        rows = [r["hands"][side] for r in series]
        summary[side] = dict(
            valid_geometry_frames=sum(r["valid_source_geometry"] for r in rows),
            directly_observed_valid_frames=sum(r["valid_source_geometry"] and r["directly_observed"] for r in rows),
            observed_frames_with_mesh_vertices_near_any_object=sum(r["directly_observed"] and any(p["vertices_within_screen_radius"] > 0 for p in r["proximity"]) for r in rows),
        )
    return dict(
        schema="v2d.human-reference-mesh-proximity-screen.v1",
        source_bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
        source_manifest_sha256=hashlib.sha256((bundle.parent / "manifest.json").read_bytes()).hexdigest(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        radius_m=radius_m,
        source_measured_contact_entries=measured_contacts,
        nearest_vertex_distance_is_surface_distance_upper_bound=True,
        source_arrays_unchanged=True,
        contact_labels_generated=False,
        surface_normals_or_forces_estimated=False,
        invalid_source_frames_remain_unknown=True,
        summary=summary,
        frames=series,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.bundle)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))
