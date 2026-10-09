# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auditable Dex3 capsule gate for a frozen Track 3 trajectory.

Mesh collisions are enclosed by fitted capsules rather than silently ignored.
Highly concave object hulls are explicitly excluded under CHORD Appendix D.
Missing observations and excluded checks remain unknown in the output arrays.
"""
from __future__ import annotations

import json
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import trimesh
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from .track3_bundle import SIDES, digest, validate_bundle, write_json


def segment_distance(
    p1: ArrayLike, p2: ArrayLike, q1: ArrayLike, q2: ArrayLike
) -> float:
    """Distance between closed segments, including zero-length spheres."""
    d1, d2, r = p2 - p1, q2 - q1, p1 - q1
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a <= 1e-20 and e <= 1e-20:
        return float(np.linalg.norm(r))
    if a <= 1e-20:
        s, t = 0.0, float(np.clip(f / e, 0, 1))
    else:
        c = float(d1 @ r)
        if e <= 1e-20:
            t, s = 0.0, float(np.clip(-c / a, 0, 1))
        else:
            b = float(d1 @ d2)
            denominator = a * e - b * b
            s = (
                float(np.clip((b * f - c * e) / denominator, 0, 1))
                if denominator > 1e-20
                else 0.0
            )
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, float(np.clip(-c / a, 0, 1))
            elif t > 1:
                t, s = 1.0, float(np.clip((b - c) / a, 0, 1))
    return float(np.linalg.norm(p1 + s * d1 - q1 - t * d2))


def point_segment_distances(
    vertices: ArrayLike, a: ArrayLike, b: ArrayLike
) -> np.ndarray:
    """Measure each point against the complete closed capsule axis."""
    direction = b - a
    denominator = float(direction @ direction)
    t = (
        np.zeros(len(vertices))
        if denominator == 0
        else np.clip((vertices - a) @ direction / denominator, 0, 1)
    )
    return np.linalg.norm(vertices - a - t[:, None] * direction, axis=1)


@dataclass
class Capsule:
    """Link-local capsule enclosing a collision mesh and its possible interior."""

    link: str
    a: np.ndarray
    b: np.ndarray
    radius: float
    provenance: dict

    def transformed(self, pose: ArrayLike) -> tuple[np.ndarray, np.ndarray, float]:
        # Retarget FK stores XYZ followed by scalar-first WXYZ, unlike qpos.
        """Map a link-local capsule using an XYZ and quaternion-WXYZ FK pose."""
        rotation = Rotation.from_quat(pose[3:], scalar_first=True).as_matrix()
        return (rotation @ self.a + pose[:3], rotation @ self.b + pose[:3], self.radius)

    def record(self) -> dict[str, Any]:
        """Serialize the link-local capsule and its geometry provenance."""
        return {
            "link": self.link,
            "a_link_m": self.a.tolist(),
            "b_link_m": self.b.tolist(),
            "radius_m": self.radius,
            **self.provenance,
        }


def fit_capsule(
    vertices: ArrayLike, link: str = "test", provenance: dict[str, Any] | None = None
) -> Capsule:
    """Fit an enclosing capsule from PCA and link-axis candidates."""
    vertices = np.asarray(vertices, dtype=np.float64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or len(vertices) < 4
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("Collision geometry requires finite 3D vertices")
    centered = vertices - vertices.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    candidates = []
    for basis in (axes, np.eye(3)):
        coordinates = vertices @ basis.T
        center = (coordinates.min(axis=0) + coordinates.max(axis=0)) / 2
        for axis in range(3):
            a, b = center.copy(), center.copy()
            a[axis], b[axis] = coordinates[:, axis].min(), coordinates[:, axis].max()
            a, b = a @ basis, b @ basis
            radius = float(point_segment_distances(vertices, a, b).max()) + 1e-9
            volume = (
                math.pi * radius**2 * np.linalg.norm(b - a)
                + 4 * math.pi * radius**3 / 3
            )
            candidates.append((volume, a, b, radius))
    _, a, b, radius = min(candidates, key=lambda item: item[0])
    outside = float(np.max(point_segment_distances(vertices, a, b) - radius))
    if radius <= 0 or outside > 1e-8:
        raise ValueError("Capsule failed collision-mesh enclosure")
    return Capsule(
        link,
        a,
        b,
        radius,
        {
            **(provenance or {}),
            "mesh_vertices": len(vertices),
            "maximum_vertex_outside_m": max(0.0, outside),
            "fit": "minimum-volume enclosing candidate among PCA and link axes; convex capsule encloses every mesh triangle",
        },
    )


def collision_capsules(urdf: Path, mesh_root: Path) -> list[Capsule]:
    """Load and enclose URDF collision meshes, rejecting missing LFS assets."""
    result = []
    for link in ET.parse(urdf).getroot().findall("link"):
        for number, collision in enumerate(link.findall("collision")):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                raise ValueError(
                    f"Unsupported collision geometry: {link.attrib['name']}"
                )
            filename = mesh.attrib["filename"]
            if not filename.startswith("package://"):
                raise ValueError(f"Expected package-relative mesh: {filename}")
            path = (mesh_root / filename.removeprefix("package://")).resolve()
            if not path.is_relative_to(mesh_root.resolve()) or not path.is_file():
                raise ValueError(f"Missing or invalid collision mesh: {path}")
            with path.open("rb") as handle:
                if handle.read(80).startswith(b"version https://git-lfs"):
                    raise ValueError(
                        f"Collision mesh is an unretrieved LFS pointer: {path}"
                    )
            loaded = trimesh.load(path, force="mesh", process=False)
            vertices = np.asarray(loaded.vertices) * np.fromstring(
                mesh.attrib.get("scale", "1 1 1"), sep=" "
            )
            origin = collision.find("origin")
            xyz = (
                np.zeros(3)
                if origin is None
                else np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
            )
            rpy = (
                np.zeros(3)
                if origin is None
                else np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
            )
            vertices = vertices @ Rotation.from_euler("xyz", rpy).as_matrix().T + xyz
            result.append(
                fit_capsule(
                    vertices,
                    link.attrib["name"],
                    {
                        "collision_index": number,
                        "mesh": filename,
                        "mesh_sha256": digest(path),
                        "urdf_sha256": digest(urdf),
                    },
                )
            )
    if not result:
        raise ValueError("No collision capsules: an empty collision model cannot pass")
    return result


def hand_hand_depth(
    left: Sequence[Any], right: Sequence[Any]
) -> tuple[float, tuple[int, int] | None]:
    """Find the largest inter-hand capsule overlap and its pair of indices."""
    if not left or not right:
        raise ValueError("Both hand capsule sets must be nonempty")
    best = 0.0
    pair: tuple[int, int] | None = None
    for i, (a, b, radius) in enumerate(left):
        for j, (c, d, other_radius) in enumerate(right):
            depth = max(0.0, radius + other_radius - segment_distance(a, b, c, d))
            if depth > best:
                best, pair = depth, (i, j)
    return best, pair


def hand_object_depth(
    capsules: Sequence[Any],
    hull: trimesh.Trimesh,
    pose: ArrayLike,
    scale: float,
    spacing_m: float,
) -> tuple[float, float]:
    """Signed-distance samples along full axes, with a Lipschitz upper bound.

    Positive signed distance is inside a watertight convex hull. Radius plus
    its axis maximum is the CHORD capsule-depth proxy. Maximum gap/2 bounds
    the unsampled distance maximum in world metres; endpoints alone miss
    capsules that cross the object through their midshaft.
    """
    if not capsules or not np.isfinite(scale) or scale <= 0 or spacing_m <= 0:
        raise ValueError("Invalid capsule, object scale or sample spacing")
    points, radius, margins = [], [], []
    for a, b, r in capsules:
        length = float(np.linalg.norm(b - a))
        intervals = max(1, math.ceil(length / spacing_m))
        points.append(np.linspace(a, b, intervals + 1))
        radius.extend([r] * (intervals + 1))
        margins.extend([length / intervals / 2] * (intervals + 1))
    points = np.concatenate(points)
    local = (points - pose[:3, 3]) @ pose[:3, :3] / scale
    signed = trimesh.proximity.signed_distance(hull, local) * scale
    if not np.isfinite(signed).all():
        raise ValueError("Nonfinite signed object distances")
    values = signed + np.asarray(radius)
    return max(0.0, float(values.max())), max(0.0, float((values + margins).max()))


def distribution(values: ArrayLike) -> dict[str, Any]:
    """Summarize finite samples while retaining an empty result as unknown."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return {
        "count": len(values),
        "median_mm": float(np.median(values) * 1000) if len(values) else None,
        "p95_mm": float(np.percentile(values, 95) * 1000) if len(values) else None,
        "max_mm": float(values.max() * 1000) if len(values) else None,
    }


def intervals(frames: ArrayLike, mask: ArrayLike) -> list[dict[str, int]]:
    """Group selected observations into contiguous source-frame intervals."""
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return [
        {
            "first_source_frame": int(frames[group[0]]),
            "last_source_frame": int(frames[group[-1]]),
            "frames": len(group),
        }
        for group in groups
    ]


def classify_checks(
    valid: np.ndarray,
    hand_hand: np.ndarray,
    hand_object_upper: np.ndarray,
    threshold_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep rejected, supported and fully checked observation masks separate."""
    object_bad = np.any(hand_object_upper > threshold_m, axis=(1, 2))
    hand_bad = hand_hand > threshold_m
    rejected = hand_bad | object_bad
    supported = valid.all(axis=1)
    fully_checked = (
        supported
        & np.isfinite(hand_hand)
        & np.isfinite(hand_object_upper).all(axis=(1, 2))
    )
    return (
        hand_bad,
        object_bad,
        rejected,
        supported,
        supported & ~rejected,
        fully_checked,
    )


def check_penetrations(  # noqa: PLR0917 - Preserve the existing gate entry-point contract.
    bundle: Path,
    retarget: Path,
    assets: Path,
    out: Path,
    threshold_m: float = 0.02,
    concavity_limit: float = 3.0,
    spacing_m: float = 0.002,
) -> dict[str, Any]:
    """Validate input hashes and audit mesh-derived capsule proxy penetration."""
    if out.exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    if (
        not np.isfinite([threshold_m, concavity_limit, spacing_m]).all()
        or threshold_m <= 0
        or concavity_limit <= 1
        or spacing_m <= 0
    ):
        raise ValueError("Invalid gate parameters")
    verified = validate_bundle(bundle)
    contract = json.loads((retarget / "manifest.json").read_text())
    if (
        contract["episode_bundle_manifest_sha256"] != verified["manifest_sha256"]
        or contract["episode_bundle_payload_sha256"] != verified["payload_sha256"]
        or contract["payload"]["sha256"] != digest(retarget / "retarget.npz")
    ):
        raise ValueError("Retarget trajectory does not match the frozen bundle")
    data, trajectory = np.load(bundle / "episode.npz"), np.load(
        retarget / "retarget.npz"
    )
    frames, valid = data["frame_indices"], trajectory["robot_valid"]
    if (
        contract["processed_frames"] != len(frames)
        or not np.array_equal(frames, trajectory["frame_indices"])
        or not np.array_equal(valid, data["hand_valid"])
    ):
        raise ValueError("Gate requires the complete source-preserving trajectory")
    n = len(frames)
    objects = json.loads((bundle / "manifest.json").read_text())["objects"]
    if not objects:
        raise ValueError("No objects declared for hand-object checks")
    capsules, fk = {}, {}
    for side in SIDES:
        capsules[side] = collision_capsules(
            assets / f"urdfs/g1/dex3_{side}.urdf", assets / "meshes"
        )
        names = contract["joints"][side]["frame_names"]
        fk[side] = trajectory[f"robot_{side}_frame_poses"]
        if any(capsule.link not in names for capsule in capsules[side]):
            raise ValueError(f"Collision link missing from {side} forward kinematics")
        if fk[side].shape != (n, len(names), 7):
            raise ValueError("Invalid forward-kinematic frame array")
        support = valid[:, SIDES.index(side)]
        if (
            not np.isfinite(fk[side][support]).all()
            or not np.isnan(fk[side][~support]).all()
            or not np.allclose(
                np.linalg.norm(fk[side][support, :, 3:], axis=-1), 1, atol=1e-6
            )
        ):
            raise ValueError("Invalid or fabricated collision-link poses")
    hulls, geometry = {}, []
    for obj in objects:
        index = obj["index"]
        mesh = trimesh.Trimesh(
            vertices=data[f"object_{index}_vertices"],
            faces=data[f"object_{index}_faces"],
            process=True,
        )
        hull = mesh.convex_hull
        volume = abs(float(mesh.volume))
        ratio = float(hull.volume / volume) if volume > 1e-15 else None
        reason = (
            "high_concavity_hull_volume_ratio"
            if ratio is not None and ratio > concavity_limit
            else None
        )
        if reason is None and not mesh.is_watertight:
            reason = "nonwatertight_mesh_volume_not_validated"
        if reason is None and (ratio is None or not hull.is_watertight):
            reason = "invalid_mesh_or_hull_volume"
        if reason is None:
            hulls[index] = hull
        geometry.append(
            {
                **obj,
                "mesh_watertight": bool(mesh.is_watertight),
                "mesh_signed_volume_m3": float(mesh.volume),
                "hull_volume_m3": float(hull.volume),
                "hull_to_absolute_mesh_volume_ratio": ratio,
                "volume_ratio_is_estimate": not bool(mesh.is_watertight),
                "hull_check_excluded": reason is not None,
                "exclusion_reason": reason,
                "capsule_signed_distance_checked_frames": 0,
            }
        )
    hand_hand = np.full(n, np.nan)
    hand_object_lower = np.full((n, 2, len(objects)), np.nan)
    hand_object_upper = hand_object_lower.copy()
    worst = None
    started = time.monotonic()
    out.mkdir(parents=True)
    for row, frame in enumerate(frames):
        world = {}
        for side_id, side in enumerate(SIDES):
            if not valid[row, side_id]:
                continue
            names = contract["joints"][side]["frame_names"]
            world[side] = [
                capsule.transformed(fk[side][row, names.index(capsule.link)])
                for capsule in capsules[side]
            ]
            for obj_id, obj in enumerate(objects):
                index = obj["index"]
                if index not in hulls:
                    continue
                lower, upper = hand_object_depth(
                    world[side],
                    hulls[index],
                    data[f"object_{index}_world"][row],
                    float(data[f"object_{index}_scale"][row]),
                    spacing_m,
                )
                (
                    hand_object_lower[row, side_id, obj_id],
                    hand_object_upper[row, side_id, obj_id],
                ) = (lower, upper)
                geometry[obj_id]["capsule_signed_distance_checked_frames"] += 1
        if len(world) == 2:
            hand_hand[row], pair = hand_hand_depth(world["left"], world["right"])
            if worst is None or hand_hand[row] > worst["depth_m"]:
                worst = {
                    "source_frame": int(frame),
                    "depth_m": float(hand_hand[row]),
                    "left_link": capsules["left"][pair[0]].link if pair else None,
                    "right_link": capsules["right"][pair[1]].link if pair else None,
                }
        if (row + 1) % 50 == 0 or row + 1 == n:
            message = {
                "frames_checked": row + 1,
                "elapsed_seconds": time.monotonic() - started,
            }
            with (out / "progress.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(message) + "\n")
            print(json.dumps(message), flush=True)
    # An exclusion is allowed by CHORD's hull proxy, but does not establish
    # measured hand-object depth. Never present those rows as fully checked.
    hand_bad, object_bad, rejected, proxy_supported, proxy_pass, fully_checked = (
        classify_checks(valid, hand_hand, hand_object_upper, threshold_m)
    )
    np.savez_compressed(
        out / "penetration.npz",
        frame_indices=frames,
        source_hand_valid=valid,
        hand_hand_depth_m=hand_hand,
        hand_object_sampled_depth_m=hand_object_lower,
        hand_object_upper_depth_m=hand_object_upper,
        chord_proxy_pass=proxy_pass,
        rejected=rejected,
        fully_checked=fully_checked,
        fully_checked_pass=fully_checked & ~rejected,
    )
    report = {
        "status": (
            "rejected"
            if rejected.any()
            else "proxy thresholds pass; exclusions and unknowns require review"
        ),
        "threshold_m": threshold_m,
        "source_frames": n,
        "hand_hand_checked_frames": int(np.isfinite(hand_hand).sum()),
        "hand_hand_depth": distribution(hand_hand),
        "worst_hand_hand": worst,
        "hand_hand_rejected_frames": int(hand_bad.sum()),
        "hand_object_rejected_frames": int(object_bad.sum()),
        "chord_proxy_pass_frames": int(proxy_pass.sum()),
        "fully_checked_pass_frames": int((fully_checked & ~rejected).sum()),
        "unknown_hand_frames": int((~proxy_supported).sum()),
        "rejected_frames": int(rejected.sum()),
        "whole_clip_chord_rejected": bool(rejected.any()),
        "objects": geometry,
        "rejected_intervals": intervals(frames, rejected),
        "chord_proxy_pass_intervals": intervals(frames, proxy_pass),
        "unknown_hand_intervals": intervals(frames, ~proxy_supported),
        "elapsed_seconds": time.monotonic() - started,
        "limitations": [
            "Capsules conservatively enclose collision meshes; proxy rejection can exceed mesh-only overlap.",
            "Concave or invalid-volume object exclusions are unknown checks, not zero penetration.",
            "Kinematic penetration results are not physics, manipulation or learned-policy success.",
        ],
    }
    manifest = {
        "schema": "v2d.track3.dex3.capsule_gate.v1",
        "bundle_manifest_sha256": verified["manifest_sha256"],
        "retarget_payload_sha256": contract["payload"]["sha256"],
        "retarget_manifest_sha256": digest(retarget / "manifest.json"),
        "source_sha256": digest(Path(__file__)),
        "capsules": {side: [c.record() for c in capsules[side]] for side in SIDES},
        "threshold_m": threshold_m,
        "hull_concavity_limit": concavity_limit,
        "axis_sample_spacing_m": spacing_m,
        "signed_distance": "positive inside convex hull; world radius + signed axis distance; upper bound adds half actual sample gap",
        "hand_hand": "analytic segment distance and sum of capsule radii, all inter-hand pairs",
        "unknown_policy": "NaN for missing hands and excluded object checks",
        "payload": {
            "path": "penetration.npz",
            "sha256": digest(out / "penetration.npz"),
        },
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "quality.json", report)
    return report
