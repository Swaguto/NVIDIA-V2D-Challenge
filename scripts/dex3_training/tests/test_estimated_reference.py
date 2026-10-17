from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1]))
from v2d_dex3_flashchord.estimated_reference import contacts, KIND


def fixture(root, change=None):
    folder = root / "contacts"
    folder.mkdir()
    points = np.zeros((2, 2, 1, 3), np.float32)
    normals = np.zeros_like(points)
    parts = np.zeros((2, 2, 1), np.int32)
    points[0, :, 0, 0], normals[0, :, 0, 0], parts[0, :, 0] = 0.1, -1, [1, 2]
    valid = np.array([[True, True], [False, False]])
    frames = np.array([0, 1])
    if change == "unmasked":
        parts[1, 0, 0] = 1
        normals[1, 0, 0, 0] = -1
    if change == "normal":
        normals[0, 0, 0, 0] = -0.5
    if change == "clock":
        frames = frames + 1
    file = folder / "estimated-contact-candidates.npz"
    np.savez(file, points_object_m=points, normal_force_directions_object=normals,
             part_ids=parts, source_geometry_valid=valid, frame_indices=frames, timestamps_s=[0, 0.05])
    receipt = dict(schema="v2d.estimated-surface-contact-probe.v1", source_bundle_sha256="fixture",
                   source_arrays_changed=False, measured_contact_labels_created=change == "measured",
                   inferred_surface_hints_only=True, invalid_source_frames_unknown=True,
                   exclude_boundary_faces=True, source_object_names=["pot", "lid"],
                   files={file.name: hashlib.sha256(file.read_bytes()).hexdigest()})
    collision = root / "collision-geometry.json"
    collision.write_text('{}')
    receipt["native_collision_surface_validated"] = change != "unvalidated_collision"
    receipt["native_collision_contract_sha256"] = hashlib.sha256(collision.read_bytes()).hexdigest()
    receipt["native_surface_filter"] = dict(maximum_reprojection_m=0.012, maximum_normal_change_deg=30, all_kept_points_rechecked_after_float32_storage=True, robot_grasp_gates_changed=False)
    if change == "object_order":
        receipt["source_object_names"].reverse()
    receipt_path = folder / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    manifest = dict(contacts_known=False, estimated_contact_reference=dict(kind=KIND,
                    receipt="contacts/receipt.json", data="contacts/estimated-contact-candidates.npz"),
                    files={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in [file, receipt_path, collision]},
                    source_files_sha256={"bundle/episode.npz": "fixture"}, objects=[dict(name="pot"), dict(name="lid")])
    pose = np.zeros((2, 2, 7))
    pose[:, 0] = [1, 2, 3, 0, 0, 0, 1]
    pose[:, 1, :3] = [-1, 2, 1]
    pose[:, 1, 3:] = Rotation.from_euler("z", 90, degrees=True).as_quat()
    data = dict(frame_indices=np.array([0, 1]), timestamps_s=np.array([0, 0.05]), object_poses=pose)
    return manifest, data, file


def test_inward_normals_and_points_follow_each_assigned_object_pose(tmp_path):
    manifest, data, _ = fixture(tmp_path)
    points, normals, parts = contacts(tmp_path, manifest, data)
    np.testing.assert_allclose(points[0, 0, 0], [1.1, 2, 3], atol=1e-6)
    np.testing.assert_allclose(points[0, 1, 0], [-1, 2.1, 1], atol=1e-6)
    np.testing.assert_allclose(normals[0, 0, 0], [-1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(normals[0, 1, 0], [0, -1, 0], atol=1e-6)
    assert not parts[1].any() and not points[1].any() and not normals[1].any()


@pytest.mark.parametrize("change", ["unmasked", "normal", "clock", "measured", "object_order", "unvalidated_collision"])
def test_bad_geometry_or_provenance_cannot_enable_guidance(tmp_path, change):
    manifest, data, _ = fixture(tmp_path, change)
    with pytest.raises(ValueError):
        contacts(tmp_path, manifest, data)


def test_changed_contact_data_requires_new_bound_receipts(tmp_path):
    manifest, data, file = fixture(tmp_path)
    with file.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        contacts(tmp_path, manifest, data)
