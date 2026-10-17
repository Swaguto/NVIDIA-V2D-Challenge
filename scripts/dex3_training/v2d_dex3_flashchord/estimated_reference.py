"""Hash-bound, explicitly estimated surface hints; unknown frames stay inactive."""

import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

KIND = "observed_mesh_surface_estimate.v1"
# Surface hints augment the contact arrays, not the parquet or scene schema.
# The runtime uses this schema to resolve the colocated physical support table.
SCHEMA = "dex3_dual_hand_v1"


def contacts(root, manifest, data):
    spec = manifest["estimated_contact_reference"]
    if spec["kind"] != KIND or manifest.get("contacts_known") is not False:
        raise ValueError("Estimated hints must not become measured contact labels")
    def checked(name):
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or name not in manifest["files"] or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][name]:
            raise ValueError("Estimated contact file hash/path mismatch")
        return path
    receipt = json.loads(checked(spec["receipt"]).read_text())
    collision_path = checked("collision-geometry.json")
    surface = receipt.get("native_surface_filter", {})
    if (
        receipt["schema"] != "v2d.estimated-surface-contact-probe.v1"
        or receipt["source_bundle_sha256"] != manifest["source_files_sha256"]["bundle/episode.npz"]
        or receipt["source_arrays_changed"]
        or receipt["measured_contact_labels_created"]
        or not receipt["inferred_surface_hints_only"]
        or not receipt["invalid_source_frames_unknown"]
        or not receipt["exclude_boundary_faces"]
        or not receipt.get("native_collision_surface_validated", False)
        or receipt.get("native_collision_contract_sha256") != hashlib.sha256(collision_path.read_bytes()).hexdigest()
        or not 0 < surface.get("maximum_reprojection_m", 0) <= 0.012
        or not 0 <= surface.get("maximum_normal_change_deg", -1) <= 30
        or surface.get("all_kept_points_rechecked_after_float32_storage") is not True
        or surface.get("outward_free_space_probe_m") != 5e-5
        or surface.get("robot_grasp_gates_changed") is not False
        or receipt["source_object_names"] != [o["name"] for o in manifest["objects"]]
    ):
        raise ValueError("Invalid estimated surface-contact provenance")
    file = checked(spec["data"])
    if hashlib.sha256(file.read_bytes()).hexdigest() != receipt["files"][file.name]:
        raise ValueError("Estimated contact receipt differs from data")
    with np.load(file, allow_pickle=False) as source:
        points = source["points_object_m"].copy()
        normals = source["normal_force_directions_object"].copy()
        parts = source["part_ids"].copy()
        valid = source["source_geometry_valid"].copy()
        if not np.array_equal(source["frame_indices"], data["frame_indices"]) or not np.allclose(source["timestamps_s"], data["timestamps_s"], atol=1e-8):
            raise ValueError("Estimated contacts must keep the exact source clock")
    frames, hands = len(data["frame_indices"]), 2
    if points.ndim != 4 or points.shape[:2] != (frames, hands) or points.shape[-1] != 3 or points.shape != normals.shape or parts.shape != points.shape[:-1] or valid.shape != (frames, hands):
        raise ValueError("Estimated contact layout differs from dual-hand source")
    active = parts > 0
    if (
        not np.issubdtype(parts.dtype, np.integer)
        or valid.dtype != np.bool_
        or not np.isfinite(points).all() or not np.isfinite(normals).all()
        or parts.min() < 0 or parts.max() > len(manifest["objects"])
        or np.any(active[~valid])
        or np.any(normals[~active] != 0)
        or np.any(points[~active] != 0)
        or not np.allclose(np.linalg.norm(normals[active], axis=-1), 1, atol=1e-5)
        or not active.any()
    ):
        raise ValueError("Invalid or unmasked estimated contact geometry")
    world_points, world_normals = np.zeros_like(points), np.zeros_like(normals)
    for obj in range(len(manifest["objects"])):
        rotation = Rotation.from_quat(data["object_poses"][:, obj, 3:]).as_matrix()
        positions = np.einsum("tij,thkj->thki", rotation, points) + data["object_poses"][:, obj, None, None, :3]
        directions = np.einsum("tij,thkj->thki", rotation, normals)
        selected = parts == obj + 1
        world_points[selected], world_normals[selected] = positions[selected], directions[selected]
    return world_points, world_normals, parts
