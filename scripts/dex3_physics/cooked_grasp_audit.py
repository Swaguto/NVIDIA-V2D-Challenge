"""Measure separation between actual cooked convexes at recorded static grasp starts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def convex_escape(first, second):
    """Minimum translation separating intersecting convex polyhedra (SAT).

    Return penetration depth and the direction translating FIRST out of SECOND.
    A negative depth denotes a separating axis; it is not Euclidean distance.
    Include edge cross products: face normals alone miss edge/edge separation.
    """
    points = [np.asarray(v, dtype=np.float64) for v in (first, second)]
    # A valid separating world axis is sufficient for the negative-depth case.
    # Avoid rebuilding hulls and edge products for distant scene pieces during
    # repeated whole-hand planning. Intersections still use the complete SAT.
    gaps = np.r_[
        points[0].min(0) - points[1].max(0), points[1].min(0) - points[0].max(0)
    ]
    index = int(np.argmax(gaps))
    if gaps[index] > 0:
        return -float(gaps[index]), np.eye(3)[index % 3] * (1 if index < 3 else -1)
    hulls = [ConvexHull(v) for v in points]
    edges = []
    for v, h in zip(points, hulls, strict=True):
        pairs = np.unique(
            np.sort(
                np.concatenate(
                    [
                        h.simplices[:, [0, 1]],
                        h.simplices[:, [1, 2]],
                        h.simplices[:, [0, 2]],
                    ]
                ),
                axis=1,
            ),
            axis=0,
        )
        edges.append(v[pairs[:, 1]] - v[pairs[:, 0]])
    cross = np.cross(edges[0][:, None, :], edges[1][None, :, :]).reshape(-1, 3)
    axes = np.concatenate([h.equations[:, :3] for h in hulls] + [cross])
    norms = np.linalg.norm(axes, axis=1)
    axes = axes[norms > 1e-10] / norms[norms > 1e-10, None]
    # Work near the origin to avoid cancellation for separated test fixtures.
    origin = points[1].mean(0)
    a, b = [np.einsum("ij,kj->ik", v - origin, axes) for v in points]
    translations = np.column_stack((b.max(0) - a.min(0), a.max(0) - b.min(0)))
    axis, sign = np.unravel_index(np.argmin(translations), translations.shape)
    return float(translations[axis, sign]), axes[axis] * (1 if sign == 0 else -1)


def convex_distance(first, second):
    """Convex solid distance via a constrained quadratic program; no signed-depth claim."""
    first, second = np.asarray(first, float), np.asarray(second, float)
    origin = second.mean(axis=0)
    first, second = first - origin, second - origin
    hulls = [ConvexHull(points) for points in [first, second]]
    a, b = hulls[0].equations, hulls[1].equations
    matrix = np.zeros((len(a) + len(b), 6))
    matrix[: len(a), :3], matrix[len(a) :, 3:] = a[:, :3], b[:, :3]
    bound = -np.r_[a[:, 3], b[:, 3]]
    result = minimize(
        lambda x: 0.5 * np.sum((x[:3] - x[3:]) ** 2),
        np.r_[first.mean(axis=0), second.mean(axis=0)],
        jac=lambda x: np.r_[x[:3] - x[3:], x[3:] - x[:3]],
        constraints=LinearConstraint(matrix, -np.inf, bound),
        method="SLSQP",
        options={"ftol": 1e-13, "maxiter": 250},
    )
    if not result.success or np.max(matrix @ result.x - bound) > 1e-7:
        raise ValueError(f"Convex separation solve failed: {result.message}")
    return float(np.linalg.norm(result.x[:3] - result.x[3:])), result.x.reshape(
        2, 3
    ) + origin


def world_convexes(collider, pose):
    transform = np.asarray(collider["mesh_to_body_matrix_row_convention"])
    rotation = Rotation.from_quat(pose[3:7])
    result = []
    for convex in collider["convexes"]:
        points = np.asarray(convex["vertices_mesh_local_m"])
        body = (np.c_[points, np.ones(len(points))] @ transform)[:, :3]
        result.append(rotation.apply(body) + pose[:3])
    return result


def minimum_distance(first, second):
    pairs = []
    for i, a in enumerate(first):
        for j, b in enumerate(second):
            lower = np.maximum(np.maximum(a.min(0) - b.max(0), b.min(0) - a.max(0)), 0)
            pairs.append((float(np.linalg.norm(lower)), i, j))
    best = float("inf")
    witness = None
    for lower, i, j in sorted(pairs):
        if lower > best:
            break
        value, points = convex_distance(first[i], second[j])
        if value < best:
            best, witness = (
                value,
                {
                    "first_convex": i,
                    "second_convex": j,
                    "points_world_m": points.tolist(),
                },
            )
    return best, witness


def audit(run):
    report = json.loads((run / "report.json").read_text())
    states_file = run / "states.json"
    if hashlib.sha256(states_file.read_bytes()).hexdigest() != report["states_sha256"]:
        raise ValueError("Recorded static states changed")
    states = json.loads(states_file.read_text())
    cooked = json.loads((run / "cooked-colliders.json").read_text())["colliders"]
    samples = []
    for i, initial in enumerate(report["initial_pose_readback"]):
        obj = next(
            r for r in states["records"] if r["sample_index"] == i and r["step"] == 0
        )
        object_shapes = [
            points
            for c in cooked
            if c["asset"] == f"object-{i}"
            for points in world_convexes(c, obj["object_root_state_stage_xyzw"][:7])
        ]
        links = []
        for collider in cooked:
            if collider["asset"] != f"hand-{i}":
                continue
            pose = initial["hand_body_link_pose_stage_xyzw"][collider["body_name"]]
            hand_shapes = world_convexes(collider, pose)
            gap, witness = minimum_distance(hand_shapes, object_shapes)
            links.append(
                {
                    "link": collider["body_name"],
                    "unsigned_convex_solid_gap_m": gap,
                    "witness": witness,
                }
            )
        samples.append(
            {
                "sample": initial["sample"],
                "minimum_gap_m": min(r["unsigned_convex_solid_gap_m"] for r in links),
                "links": links,
            }
        )
    return {
        "schema": "v2d.cooked-grasp-gap.v1",
        "states_sha256": report["states_sha256"],
        "cooked_sha256": hashlib.sha256(
            (run / "cooked-colliders.json").read_bytes()
        ).hexdigest(),
        "samples": samples,
        "scope": "Actual cooked convex geometry at recorded t=0 link poses; unsigned solid separation only, not a grasp or penetration-depth certificate",
        "physical_task_accepted": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(
        json.dumps(
            [{k: v for k, v in s.items() if k != "links"} for s in result["samples"]],
            indent=2,
        )
    )
