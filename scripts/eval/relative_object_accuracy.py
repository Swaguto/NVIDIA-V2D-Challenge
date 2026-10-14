"""Measure object agreement without fitting the reconstruction world to MoCap.

Used by score_candidate.py --mode relative. This is a diagnostic, not an
official competition score. A common rigid world transform cancels in each
ordered object pair. Single-object episodes cannot establish this agreement.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import scipy
import trimesh
from reference_loader import load_reference
from scipy.spatial.transform import Rotation


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_reference(data: Path, reference_lock: Path) -> None:
    """Restore immutable organizer inputs into a fresh root; verify before rename."""
    if data.exists():
        raise FileExistsError("Reference download needs a fresh --data directory")
    lock = json.loads(reference_lock.read_text(encoding="utf-8-sig"))
    revision = lock["revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Reference revision must be a full immutable commit")
    data.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=data.parent) as temporary:
        root = Path(temporary)
        for item in lock["files"]:
            relative = Path(item["relative_path"])
            path = (root / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(root.resolve()):
                raise ValueError("Reference lock path escapes its root")
            url = (
                f"https://huggingface.co/datasets/{lock['dataset']}/resolve/"
                f"{revision}/track_3/public/{relative.as_posix()}"
            )
            if item["source_url"] != url:
                raise ValueError(
                    "Source URL differs from immutable public reference path"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            with (
                urllib.request.urlopen(url, timeout=45) as response,
                path.open("wb") as output,
            ):
                while block := response.read(1024 * 1024):
                    output.write(block)
            if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise ValueError(f"Downloaded reference differs from lock: {relative}")
        root.rename(data)


def summary(values: np.ndarray) -> dict:
    if not len(values):
        return {"count": 0, "median": None, "p90": None, "maximum": None}
    return {
        "count": len(values),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "maximum": float(values.max()),
    }


def measure(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, ...]:
    """Compare two-body rigid poses; translations are in the first scan frame."""
    predicted = np.linalg.inv(first[:, 0]) @ first[:, 1]
    reference = np.linalg.inv(second[:, 0]) @ second[:, 1]
    translation = np.linalg.norm(predicted[:, :3, 3] - reference[:, :3, 3], axis=1)
    rotation = (
        Rotation.from_matrix(predicted[:, :3, :3]).inv()
        * Rotation.from_matrix(reference[:, :3, :3])
    ).magnitude()
    separation = np.abs(
        np.linalg.norm(first[:, 1, :3, 3] - first[:, 0, :3, 3], axis=1)
        - np.linalg.norm(second[:, 1, :3, 3] - second[:, 0, :3, 3], axis=1)
    )
    return translation, np.degrees(rotation), separation


def verify_controls() -> dict:
    """Check gauge cancellation and known errors using nondegenerate poses."""
    poses = np.tile(np.eye(4), (3, 2, 1, 1))
    poses[:, 0, :3, :3] = Rotation.from_rotvec(
        [[0.1, 0.2, -0.1], [0.2, -0.1, 0.1], [-0.1, 0.1, 0.3]]
    ).as_matrix()
    poses[:, 1, :3, 3] = [[0.2, 0.1, 0.3], [0.1, 0.2, 0.4], [0.3, 0.2, 0.1]]
    common = np.eye(4)
    common[:3, :3] = Rotation.from_rotvec([0.3, -0.2, 0.1]).as_matrix()
    common[:3, 3] = [0.2, -0.3, 0.7]
    for actual in measure(common @ poses, poses):
        np.testing.assert_allclose(actual, 0.0, rtol=0, atol=1e-12)
    translated = poses.copy()
    translated[:, 1, :3, 3] += np.einsum(
        "nij,j->ni", translated[:, 0, :3, :3], [0.015, 0, 0]
    )
    np.testing.assert_allclose(measure(translated, poses)[0], 0.015, rtol=0, atol=1e-12)
    rotated = poses.copy()
    rotated[:, 1, :3, :3] = (
        rotated[:, 1, :3, :3] @ Rotation.from_euler("z", 10, degrees=True).as_matrix()
    )
    np.testing.assert_allclose(measure(rotated, poses)[1], 10, rtol=0, atol=1e-10)
    return {
        "common_rigid_world_transform_cancels": True,
        "known_fifteen_mm_translation_recovers_fifteen_mm": True,
        "known_ten_degree_rotation_recovers_ten_degree": True,
    }


def audit(
    bundle: Path, data: Path, reference_lock: Path, proposal: Path | None
) -> dict:
    """Verify immutable inputs, record all visible errors and reference repeats."""
    lock = json.loads(reference_lock.read_text(encoding="utf-8-sig"))
    if not lock.get("revision"):
        raise ValueError("Reference lock must name an immutable dataset revision")
    checked = {}
    for item in lock["files"]:
        relative = Path(item["relative_path"])
        path = (data / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(data.resolve()):
            raise ValueError("Reference lock path escapes its root")
        if digest(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Reference input differs from lock: {relative}")
        checked[relative.as_posix()] = item["sha256"]
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    payload = bundle / "episode.npz"
    if digest(payload) != manifest["payload"]["sha256"]:
        raise ValueError("Bundle payload differs from its manifest")
    episode = int(manifest["episode"])
    required = {f"data/chunk-000/episode_{episode:06d}.parquet", "meta/info.json"}
    names = tuple(item["name"] for item in manifest["objects"])
    required.update(f"mesh/{name}/{name}.glb" for name in names)
    if not required.issubset(checked):
        raise ValueError("Lock does not cover every reference input used by the audit")
    reference = load_reference(episode, data)
    if reference.object_names != names or reference.bodies < 2:
        raise ValueError("Need at least two objects in identical scan-name order")
    source = np.load(payload, allow_pickle=False)
    np.testing.assert_array_equal(reference.frame_index, source["frame_indices"])
    if not np.isfinite(reference.capture_time).all() or np.any(
        np.diff(reference.capture_time) <= 0
    ):
        raise ValueError("Reference capture times must be finite and increasing")
    np.testing.assert_allclose(
        np.linalg.norm(reference.pose_xyzw[reference.visible, 3:], axis=-1),
        1.0,
        rtol=0,
        atol=1e-6,
    )
    predicted = np.stack(
        [source[f"object_{i}_world"] for i in range(reference.bodies)], axis=1
    ).copy()
    if proposal:
        with np.load(proposal, allow_pickle=False) as candidate:
            np.testing.assert_array_equal(
                candidate["frame_indices"], reference.frame_index
            )
            if candidate["object_world"].shape != predicted.shape:
                raise ValueError("Proposal shape differs from source object poses")
            predicted = candidate["object_world"].copy()
    if not np.isfinite(predicted).all():
        raise ValueError("Nonfinite candidate pose")
    np.testing.assert_allclose(
        predicted[:, :, 3, :],
        np.broadcast_to([0, 0, 0, 1], predicted[:, :, 3, :].shape),
        rtol=0,
        atol=1e-10,
    )
    observed = np.array(
        [
            [
                manifest["object_observation_provenance"][name][str(int(frame))]
                == "observed"
                for name in names
            ]
            for frame in reference.frame_index
        ]
    )
    meshes, affine = [], []
    for i, name in enumerate(names):
        mesh = trimesh.load(
            data / f"mesh/{name}/{name}.glb", force="mesh", process=False
        )
        vertices = np.unique(source[f"object_{i}_vertices"].astype(np.float32), axis=0)
        np.testing.assert_array_equal(
            np.unique(np.asarray(mesh.vertices).astype(np.float32), axis=0), vertices
        )
        meshes.append(
            {
                "object": name,
                "all_unique_scan_vertices_exact": True,
                "unique_vertices": len(vertices),
            }
        )
        raw = predicted[:, i, :3, :3].copy()
        u, _, vt = np.linalg.svd(raw)
        correction = np.tile(np.eye(3), (len(raw), 1, 1))
        correction[:, 2, 2] = np.linalg.det(u @ vt)
        proper = u @ correction @ vt
        residual = float(np.linalg.norm(raw - proper, axis=(1, 2)).max())
        scale_error = float(np.abs(source[f"object_{i}_scale"] - 1).max())
        if residual >= 1e-5 or scale_error >= 1e-5:
            raise ValueError(
                "Nonrigid pose/scale cannot be treated as the canonical scan frame"
            )
        affine.append(
            {
                "object": name,
                "maximum_rotation_projection_frobenius_residual": residual,
                "maximum_geometry_shift_bound_m": float(
                    residual * np.linalg.norm(vertices, axis=1).max()
                ),
                "maximum_recorded_scale_deviation_from_one": scale_error,
            }
        )
        predicted[:, i, :3, :3] = proper
    truth = np.tile(np.eye(4), (reference.steps, reference.bodies, 1, 1))
    truth[:, :, :3, 3] = reference.pose_xyzw[:, :, :3]
    truth[reference.visible, :3, :3] = Rotation.from_quat(
        reference.pose_xyzw[reference.visible, 3:]
    ).as_matrix()
    pairs = []
    keys = (
        "relative_scan_origin_translation_error_m",
        "relative_scan_orientation_error_deg",
        "inter_scan_origin_distance_error_m",
    )
    for first, second in itertools.combinations(range(reference.bodies), 2):
        visible = reference.visible[:, first] & reference.visible[:, second]
        rows = np.flatnonzero(visible)
        errors = (
            measure(
                predicted[rows][:, [first, second]], truth[rows][:, [first, second]]
            )
            if len(rows)
            else (np.array([]),) * 3
        )
        both_observed = observed[rows, first] & observed[rows, second]
        pairs.append(
            {
                "objects": [names[first], names[second]],
                "summaries": {
                    "all_visible_GT": dict(zip(keys, map(summary, errors))),
                    "both_reconstruction_objects_observed": dict(
                        zip(keys, [summary(value[both_observed]) for value in errors])
                    ),
                },
                "records": [
                    {
                        "source_frame": int(reference.frame_index[row]),
                        "reconstruction_observed_both": bool(both_observed[j]),
                        **{key: float(value[j]) for key, value in zip(keys, errors)},
                    }
                    for j, row in enumerate(rows)
                ],
            }
        )
    repeats = np.concatenate(
        (
            [False],
            (reference.pose_xyzw[1:] == reference.pose_xyzw[:-1]).all(axis=(1, 2))
            & reference.visible[1:].all(axis=1)
            & reference.visible[:-1].all(axis=1),
        )
    )
    runs = []
    for row in range(1, len(repeats)):
        if repeats[row] and not repeats[row - 1]:
            end = row
            while end + 1 < len(repeats) and repeats[end + 1]:
                end += 1
            runs.append(
                {
                    "first_pose_frame": int(reference.frame_index[row - 1]),
                    "identical_copy_frames": reference.frame_index[
                        row : end + 1
                    ].tolist(),
                }
            )
    return {
        "schema": "v2d.public-object-relative-accuracy.v1",
        "episode": episode,
        "organizer_revision": lock["revision"],
        "reference_lock_sha256": digest(reference_lock),
        "reference_inputs_verified": checked,
        "reference_loader_sha256": digest(
            Path(__file__).with_name("reference_loader.py")
        ),
        "audit_source_sha256": digest(Path(__file__)),
        "bundle_manifest_sha256": digest(bundle / "manifest.json"),
        "bundle_payload_sha256": digest(payload),
        "proposal_sha256": digest(proposal) if proposal else None,
        "runtime_versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "trimesh": trimesh.__version__,
        },
        "source_frames": reference.steps,
        "frame_indices_exact": True,
        "canonical_scan_frames": meshes,
        "declared_affine_projection_and_scale_residuals": affine,
        "controls": verify_controls(),
        "pairs": pairs,
        "reference_freshness": {
            "consecutive_exact_duplicate_all_object_pose_frames": int(repeats.sum()),
            "exact_duplicate_runs": runs,
            "capture_times_increase": True,
            "rows_removed_or_repaired": False,
            "repeated_values_prove_stale_tracking": False,
        },
        "global_coordinate_alignment_estimated": False,
        "GT_used_for_source_fitting": False,
        "absolute_world_pose_or_official_score_measured": False,
        "hand_or_table_accuracy_accepted": False,
        "physical_task_accepted": False,
    }
