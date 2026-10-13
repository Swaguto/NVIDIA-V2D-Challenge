"""Reproduce one original-triangle violation of the declared Dex3 material limit.

This verifies a saved model counterexample. It does not certify a trajectory,
establish true-object penetration, or promote recovered observations.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import _frozen_material_boundary_v1 as boundary_method


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not __debug__:
        raise RuntimeError("Run this verifier without Python -O; its invariant checks must remain enabled.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "docs/team/reproduction/dex3-motion-replay/continuous-counterexample-v1.json",
    )
    parser.add_argument("--material", type=Path)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    material = args.material or args.fixture.parent / fixture["original_material_filename"]
    assert digest(material) == fixture["material_payload_sha256"]
    assert digest(Path(boundary_method.__file__)) == fixture["independent_boundary_method_sha256"]
    assert not fixture["physical_task_accepted"]
    assert not fixture["physical_object_geometry_ground_truth"]
    triangle = np.asarray(fixture["original_triangle_object_local_m"], dtype=float)
    point = np.asarray(fixture["witness_object_local_m"], dtype=float)
    assert triangle.shape == (3, 3) and point.shape == (3,)
    assert np.isfinite(triangle).all() and np.isfinite(point).all()
    coefficients = np.linalg.lstsq((triangle[1:] - triangle[0]).T, point - triangle[0], rcond=None)[0]
    barycentric = np.r_[1 - coefficients.sum(), coefficients]
    residual = float(np.linalg.norm(barycentric @ triangle - point))
    assert residual < 1e-12 and barycentric.min() >= -1e-10 and barycentric.max() <= 1 + 1e-10
    matrix = np.asarray(fixture["source_object_affine_matrix"], dtype=float)
    assert matrix.shape == (3, 3) and np.isfinite(matrix).all()
    singular = np.linalg.svd(matrix, compute_uv=False)
    assert singular.min() > 0
    with np.load(material, allow_pickle=False) as archive:
        boundary = boundary_method.Boundary(archive["solid_cells"], archive["origin_m"], float(archive["pitch_m"]))
    assert boundary.inside(point[None])[0]
    distance, _, _ = boundary.unsigned(point[None])
    lower = float(distance[0] * singular.min())
    limit = fixture["world_depth_limit_m"]
    assert limit == 0.002 and lower > limit
    assert abs(lower - fixture["independent_depth_lower_m"]) < 1e-12
    result = {
        "time_s": fixture["time_s"],
        "side": fixture["side"],
        "link": fixture["link"],
        "material_payload_sha256": digest(material),
        "depth_lower_m": lower,
        "world_depth_limit_m": limit,
        "point_inside_original_triangle": True,
        "declared_material_limit_exceeded": True,
        "entire_trajectory_checked": False,
        "physical_object_geometry_ground_truth": False,
        "physical_task_accepted": False,
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
