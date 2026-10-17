"""Keep estimated hints only on exposed frozen native convex surfaces.

Robot physics, source arrays and measured labels do not change. This filters
and slightly reprojects geometric estimates, not measured contacts or forces.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
import trimesh
from prepare_estimated_contacts import near_surface


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exposed_projection(points, force_directions, meshes, radius=0.002, maximum_normal_angle_deg=30):
    equations = [ConvexHull(mesh.vertices).equations for mesh in meshes]
    def margin(points, equation):
        return (np.einsum("pi,fi->pf", points, equation[:, :3]) + equation[:, 3]).max(axis=-1)
    depth = np.zeros(len(points))
    for equation in equations:
        depth = np.maximum(depth, np.maximum(0, -margin(points, equation)))
    best = np.full(len(points), np.inf)
    selected_points, selected_force = np.zeros_like(points), np.zeros_like(force_directions)
    source_outward = -force_directions
    for mesh, equation in zip(meshes, equations, strict=True):
        bounds = mesh.bounds
        selected = np.flatnonzero((depth <= radius) & np.all(points >= bounds[0] - radius, axis=-1) & np.all(points <= bounds[1] + radius, axis=-1))
        if not len(selected):
            continue
        closest, distance, face = near_surface(mesh, points[selected], radius)
        normals = mesh.face_normals[np.maximum(face, 0)]
        alignment = np.einsum("pi,pi->p", normals, source_outward[selected])
        valid = (face >= 0) & (distance <= radius) & (alignment >= np.cos(np.deg2rad(maximum_normal_angle_deg))) & (distance < best[selected])
        indices = np.flatnonzero(valid)
        if not len(indices):
            continue
        # Faces buried inside another collider are not an exposed union surface.
        outside = np.ones(len(indices), bool)
        outward_probe = closest[indices] + 5e-5 * normals[indices]
        for other in equations:
            outside &= margin(closest[indices], other) >= -1e-7
            outside &= margin(outward_probe, other) >= -1e-7
        indices = indices[outside]
        chosen = selected[indices]
        best[chosen] = distance[indices]
        selected_points[chosen], selected_force[chosen] = closest[indices], -normals[indices]
    return selected_points, selected_force, np.isfinite(best), best, depth


def filter_contacts(native, estimate, out, maximum_reference_projection_m=0.002):
    if not np.isfinite(maximum_reference_projection_m) or not 0 < maximum_reference_projection_m <= 0.012:
        raise ValueError("Estimated-reference projection must be within 12 mm")
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((native / "manifest.json").read_text())
    geometry_path = native / "collision-geometry.json"
    if sha(geometry_path) != manifest["files"]["collision-geometry.json"]:
        raise ValueError("Frozen collision contract changed")
    geometry = json.loads(geometry_path.read_text())
    if not geometry["collision_vertices_in_body_coordinates"]:
        raise ValueError("Native collision vertices need body coordinates")
    receipt = json.loads((estimate / "receipt.json").read_text())
    source = estimate / "estimated-contact-candidates.npz"
    if sha(source) != receipt["files"][source.name]:
        raise ValueError("Estimated candidate data changed")
    with np.load(source, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    original_parts = arrays["part_ids"].copy()
    statistics = []
    for obj in range(len(manifest["objects"])):
        meshes = []
        for part in geometry["hulls"][f"object/{obj}"]:
            path = native / "objects" / part["file"]
            if sha(path) != part["sha256"]:
                raise ValueError("Frozen collision part changed")
            mesh = trimesh.load(path, force="mesh", process=True)
            if not mesh.is_convex or not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
                raise ValueError("Need outward, closed native convex pieces")
            meshes.append(mesh)
        selected = original_parts == obj + 1
        points = arrays["points_object_m"][selected].astype(float)
        forces = arrays["normal_force_directions_object"][selected].astype(float)
        projected, normals, kept, distance, depth = exposed_projection(points, forces, meshes, radius=maximum_reference_projection_m)
        maximum_inward_residual = 0.0
        for mesh in meshes:
            eq = ConvexHull(mesh.vertices).equations
            q = projected[kept].astype(np.float32).astype(float)
            if len(q):
                margins = (np.einsum("pi,fi->pf", q, eq[:, :3]) + eq[:, 3]).max(-1)
                maximum_inward_residual = max(maximum_inward_residual, float(max(0, -margins.min())))
                probe = q - 5e-5 * normals[kept].astype(np.float32).astype(float)
                probe_margin = (np.einsum("pi,fi->pf", probe, eq[:, :3]) + eq[:, 3]).max(-1)
                if probe_margin.min() < -1e-7:
                    raise ValueError("Estimated normal points into a neighbouring collider")
        if maximum_inward_residual > 1e-7:
            raise ValueError("Projected estimate is buried inside a native collider")
        arrays["points_object_m"][selected] = projected
        arrays["normal_force_directions_object"][selected] = normals
        ids = arrays["part_ids"][selected]
        ids[~kept] = 0
        arrays["part_ids"][selected] = ids
        statistics.append(dict(object=obj,source_candidate_points=len(points),kept_exposed_surface_points=int(kept.sum()),source_points_guaranteed_deeper_than2mm=int((depth > 0.002).sum()),rejected_guaranteed_deeper_than_projection_bound=int((depth > maximum_reference_projection_m).sum()),maximum_reprojection_m=float(distance[kept].max()) if kept.any() else None,maximum_kept_point_inward_roundoff_m=maximum_inward_residual))
    counts = np.stack([(arrays["part_ids"] == obj + 1).sum(-1) for obj in range(len(manifest["objects"]))], axis=-1).astype(np.int32)
    arrays["contact_count_per_object"] = counts
    out.mkdir(parents=True)
    np.savez_compressed(out / source.name, **arrays)
    receipt["source_candidate_receipt_sha256"] = sha(estimate / "receipt.json")
    receipt["source_candidate_data_sha256"] = sha(source)
    receipt["collision_filter_source_sha256"] = sha(Path(__file__))
    receipt["native_collision_surface_validated"] = True
    receipt["native_collision_contract_sha256"] = sha(geometry_path)
    receipt["native_surface_filter"] = dict(maximum_reprojection_m=maximum_reference_projection_m, maximum_normal_change_deg=30, exposed_outside_all_parts_tolerance_m=1e-7, outward_free_space_probe_m=5e-5, all_kept_points_rechecked_after_float32_storage=True, reference_projection_is_not_robot_penetration_tolerance=True,robot_grasp_gates_changed=False,objects=statistics)
    receipt["raw_surface_candidate_summary"] = receipt["summary"]
    receipt["summary"] = {}
    for hand, side in enumerate(["left", "right"]):
        number = counts[:, hand].sum(-1)
        receipt["summary"][side] = dict(valid_observed_frames=int(arrays["source_geometry_valid"][:, hand].sum()), frames_with_any_estimated_patch=int((number > 0).sum()), median_selected_vertices_per_active_frame=float(np.median(number[number > 0])) if (number > 0).any() else None)
    receipt["force_normal_convention"] = "Inward exposed native convex-face normal, estimated by source surface proximity; not measured force"
    receipt["files"] = {source.name: sha(out / source.name)}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--maximum-reference-projection-m", type=float, default=0.002)
    args = parser.parse_args()
    print(json.dumps(filter_contacts(args.native, args.estimate, args.out, args.maximum_reference_projection_m), indent=2))
