"""Prepare private geometric contact candidates for review, never measured labels.

Requires trimesh and rtree on the CPU preparation host. This does not activate
any training reward or change the source bundle. Open meshes use local face
normals only; no global signed-distance or force claim is made.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
import importlib.metadata


def near_surface(mesh, points, radius):
    """Exact nearest triangle whenever surface distance <= radius, bounded query.

    A triangle within the radius must intersect the point's expanded AABB. Far
    points need no global closest-surface search over a large reconstruction.
    """
    closest = np.zeros_like(points)
    distance = np.full(len(points), np.inf)
    faces = np.full(len(points), -1, np.int32)
    tree, triangles = mesh.triangles_tree, mesh.triangles
    for index, point in enumerate(points):
        candidates = np.array(list(tree.intersection(np.r_[point - radius, point + radius])), dtype=np.int64)
        if not len(candidates):
            continue
        projection = trimesh.triangles.closest_point(triangles[candidates], np.broadcast_to(point, (len(candidates), 3)))
        squared = np.einsum("ij,ij->i", projection - point, projection - point)
        winner = int(np.argmin(squared))
        closest[index], distance[index], faces[index] = projection[winner], np.sqrt(squared[winner]), candidates[winner]
    return closest, distance, faces


def prepare(bundle, out, radius=0.005, local_inside_margin=0.002, ambiguity_margin=0.001):
    if not 0 < radius <= 0.01 or not 0 <= local_inside_margin <= radius or not 0 <= ambiguity_margin <= radius:
        raise ValueError("Invalid bounded contact-candidate geometry margins")
    if out.exists():
        raise FileExistsError(out)
    source_hash = hashlib.sha256(bundle.read_bytes()).hexdigest()
    manifest = json.loads((bundle.parent / "manifest.json").read_text())
    if manifest["units"] != "metres":
        raise ValueError("Source geometry must use metres")
    with np.load(bundle, allow_pickle=False) as data:
        hands = data["hand_vertices_world"].astype(float)
        valid = data["hand_valid"].astype(bool) & data["hand_directly_observed"].astype(bool)
        frames, times = data["frame_indices"].copy(), data["timestamps_s"].copy()
        known_count = int(np.count_nonzero(data["contact_known"]))
        objects = []
        index = 0
        while f"object_{index}_vertices" in data:
            mesh = trimesh.Trimesh(vertices=data[f"object_{index}_vertices"], faces=data[f"object_{index}_faces"], process=True)
            pose, scale = data[f"object_{index}_world"].astype(float), data[f"object_{index}_scale"].astype(float)
            if not mesh.is_winding_consistent or not np.isfinite(mesh.volume) or mesh.volume <= 0:
                raise ValueError("Cannot orient estimated normals from this source mesh")
            if not np.isfinite(pose).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("Invalid source object pose or scale")
            rotations = pose[:, :3, :3]
            if not np.allclose(np.einsum("tji,tjk->tik", rotations, rotations), np.eye(3), atol=1e-5) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5):
                raise ValueError("Source object poses need proper rotations")
            degree = np.bincount(mesh.edges_unique_inverse)
            boundary = (degree[mesh.faces_unique_edges] != 2).any(axis=1)
            objects.append((mesh, pose, scale, boundary))
            index += 1
    if not objects or valid.shape != hands.shape[:2] or len(manifest["hand_sides"]) != hands.shape[1] or len(frames) != len(hands) or len(times) != len(hands):
        raise ValueError("Source hand/object clocks or side layouts differ")
    points = np.zeros_like(hands, dtype=np.float32)
    directions = np.zeros_like(points)
    parts = np.zeros(hands.shape[:-1], np.int32)
    counts = np.zeros((*valid.shape, len(objects)), np.int32)
    for frame in range(len(hands)):
        for hand in range(hands.shape[1]):
            if not valid[frame, hand] or not np.isfinite(hands[frame, hand]).all():
                valid[frame, hand] = False
                continue
            candidates = []
            for obj, (mesh, poses, scales, boundary) in enumerate(objects):
                pose, scale = poses[frame], scales[frame]
                local = np.einsum("vj,jk->vk", hands[frame, hand] - pose[:3, 3], pose[:3, :3]) / scale
                closest, distance, face = near_surface(mesh, local, radius / scale)
                present = face >= 0
                safe_face = np.maximum(face, 0)
                normals = mesh.face_normals[safe_face]
                gap = np.einsum("vj,vj->v", local - closest, normals) * scale
                usable = present & (distance * scale <= radius) & (gap >= -local_inside_margin) & (~boundary[safe_face]) & (mesh.area_faces[safe_face] > 1e-12)
                candidates.append((closest * scale, -normals, distance * scale, usable))
            distances = np.stack([np.where(c[3], c[2], np.inf) for c in candidates])
            order = np.argsort(distances, axis=0)
            columns = np.arange(hands.shape[2])
            first = distances[order[0], columns]
            second = distances[order[1], columns] if len(objects) > 1 else np.full(len(columns), np.inf)
            both = np.isfinite(first) & np.isfinite(second)
            difference = np.full_like(first, np.inf)
            np.subtract(second, first, out=difference, where=both)
            ambiguous = both & (difference < ambiguity_margin)
            for obj, (position, normal, distance, usable) in enumerate(candidates):
                selected = usable & (order[0] == obj) & (~ambiguous)
                points[frame, hand, selected] = position[selected]
                directions[frame, hand, selected] = normal[selected]
                parts[frame, hand, selected] = obj + 1
                counts[frame, hand, obj] = int(selected.sum())
        if frame % 20 == 0:
            print("source frame", frame, "estimated patch vertices", counts[frame].tolist(), flush=True)
    if not np.isfinite(points).all() or not np.isfinite(directions).all() or not np.allclose(np.linalg.norm(directions[parts > 0], axis=-1), 1, atol=1e-5):
        raise ValueError("Invalid estimated surface directions")
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != source_hash:
        raise ValueError("Source bundle changed during preparation")
    out.mkdir(parents=True)
    np.savez_compressed(out / "estimated-contact-candidates.npz", frame_indices=frames, timestamps_s=times, points_object_m=points, normal_force_directions_object=directions, part_ids=parts, source_geometry_valid=valid, contact_count_per_object=counts)
    summary = {}
    for hand, side in enumerate(manifest["hand_sides"]):
        number = counts[:, hand].sum(-1)
        summary[side] = dict(valid_observed_frames=int(valid[:, hand].sum()), frames_with_any_estimated_patch=int((number > 0).sum()), median_selected_vertices_per_active_frame=float(np.median(number[number > 0])) if (number > 0).any() else None)
    receipt = dict(
        schema="v2d.estimated-surface-contact-probe.v1", source_bundle_sha256=source_hash,
        source_object_names=[item["name"] for item in manifest["objects"]],
        preparation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_arrays_changed=False, source_measured_contact_entries=known_count,
        measured_contact_labels_created=False, inferred_surface_hints_only=True,
        source_objects_watertight=[x[0].is_watertight for x in objects],
        source_objects_winding_consistent=[x[0].is_winding_consistent for x in objects],
        screen_distance_m=radius, minimum_local_outward_gap_m=-local_inside_margin,
        exclude_boundary_faces=True, exclude_ambiguous_objects_within_m=ambiguity_margin,
        invalid_source_frames_unknown=True, project_to_object_surface=True,
        force_normal_convention="Inward locally consistent object face normal; estimated, not measured force",
        normals_on_open_meshes_are_not_globally_certified=True,
        training_guidance_activated=False, summary=summary,
        preparation_library_versions={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "trimesh", "rtree")},
        files={"estimated-contact-candidates.npz": hashlib.sha256((out / "estimated-contact-candidates.npz").read_bytes()).hexdigest()},
    )
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.bundle, args.out), indent=2))
