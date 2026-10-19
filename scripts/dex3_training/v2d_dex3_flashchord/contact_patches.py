"""Object-local surface patches and explicit Dex3 contact-mode hypotheses.

Patches describe geometry, not active contacts or measured source contact truth.
They never add friction, forces, collision geometry or contact dimensions.
Actual native contacts must be audited separately, including every unmatched
contact; proximity to a patch alone cannot qualify a grasp.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from trimesh.triangles import closest_point


def _finite(value, shape, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {name}") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Need finite {name} with shape {shape}")
    array.setflags(write=False)
    return array


def _identity(object_id, geometry_sha256):
    if not isinstance(object_id, str) or not object_id.strip():
        raise ValueError("Need an explicit object id")
    if (not isinstance(geometry_sha256, str) or len(geometry_sha256) != 64
            or any(c not in '0123456789abcdef' for c in geometry_sha256)):
        raise ValueError("Need a lowercase SHA-256 binding for geometry")


def _pose(value):
    pose = _finite(value, (4, 4), "object-to-world transform")
    rotation = pose[:3, :3]
    if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
            or abs(np.linalg.det(rotation) - 1) > 1e-7):
        raise ValueError("Use a proper rigid object-to-world transform, without scale")
    return pose


class PatchProjection(NamedTuple):
    points: np.ndarray
    outward_normals: np.ndarray
    distances_m: np.ndarray
    triangle_indices: np.ndarray


@dataclass(frozen=True, eq=False)
class ObjectContactPatch:
    patch_id: str
    object_id: str
    geometry_sha256: str
    triangles_local_m: np.ndarray
    outward_normals_local: np.ndarray

    def __post_init__(self):
        _identity(self.object_id, self.geometry_sha256)
        if not isinstance(self.patch_id, str) or not self.patch_id.strip():
            raise ValueError("Need an explicit patch id")
        raw = np.asarray(self.triangles_local_m)
        if raw.ndim != 3 or raw.shape[1:] != (3, 3) or not len(raw):
            raise ValueError("Need nonempty triangles[N,3,3] in object-local metres")
        tri = _finite(raw, raw.shape, "patch triangles")
        normal = _finite(self.outward_normals_local, (len(tri), 3), "patch normals")
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = np.linalg.norm(cross, axis=1)
        if (area < 1e-14).any():
            raise ValueError("Patch triangles must be nondegenerate")
        if (not np.allclose(np.linalg.norm(normal, axis=1), 1, atol=1e-7, rtol=0)
                or (np.abs(np.einsum('ni,ni->n', cross / area[:, None], normal)) < 1 - 1e-7).any()):
            raise ValueError("Patch normals must be unit and perpendicular to their faces")
        # Triangle winding may be unordered in a native convex hull. The caller
        # must establish OUTWARD signs and external-union membership from assets.
        object.__setattr__(self, 'triangles_local_m', tri)
        object.__setattr__(self, 'outward_normals_local', normal)

    def closest_local(self, points_local_m):
        raw = np.asarray(points_local_m)
        if raw.ndim != 2 or raw.shape[1:] != (3,):
            raise ValueError("Need query points[N,3]")
        points = _finite(raw, raw.shape, "query points")
        result, distance, ids = [], [], []
        # One query at a time avoids unbounded query-by-triangle temporaries.
        for point in points:
            projected = closest_point(self.triangles_local_m,
                np.broadcast_to(point, (len(self.triangles_local_m), 3)))
            lengths = np.linalg.norm(projected - point, axis=1)
            i = int(np.argmin(lengths))
            result.append(projected[i]); distance.append(lengths[i]); ids.append(i)
        indices = np.asarray(ids, dtype=int)
        return PatchProjection(np.asarray(result).reshape(-1, 3),
            self.outward_normals_local[indices].copy(), np.asarray(distance), indices)

    def closest_world(self, points_world_m, object_to_world):
        transform = _pose(object_to_world)
        raw = np.asarray(points_world_m)
        if raw.ndim != 2 or raw.shape[1:] != (3,):
            raise ValueError("Need world query points[N,3]")
        points = _finite(raw, raw.shape, "world query points")
        local = np.einsum('ni,ij->nj', points - transform[:3, 3],
                          transform[:3, :3], optimize=False)
        result = self.closest_local(local)
        return PatchProjection(
            np.einsum('ni,ji->nj', result.points, transform[:3, :3], optimize=False) + transform[:3, 3],
            np.einsum('ni,ji->nj', result.outward_normals, transform[:3, :3], optimize=False),
            result.distances_m, result.triangle_indices)


@dataclass(frozen=True)
class RobotContactMode:
    """Required digit-to-patch assignments, not observed contact labels."""
    mode_id: str
    hand: str
    object_id: str
    geometry_sha256: str
    digit_patches: tuple[tuple[str, str], ...]

    def __post_init__(self):
        _identity(self.object_id, self.geometry_sha256)
        if not isinstance(self.mode_id, str) or not self.mode_id.strip() or self.hand not in ('left', 'right'):
            raise ValueError("Need mode id and explicit left/right hand")
        try:
            bindings = tuple(tuple(pair) for pair in self.digit_patches)
        except TypeError as error:
            raise ValueError("Need digit/patch bindings") from error
        if (len(bindings) not in (2, 3) or any(len(p) != 2 for p in bindings)
                or any(p[0] not in ('thumb', 'index', 'middle') or not isinstance(p[1], str) or not p[1] for p in bindings)
                or len({p[0] for p in bindings}) != len(bindings)):
            raise ValueError("Need two or three distinct Dex3 digits and named patches")
        object.__setattr__(self, 'digit_patches', bindings)

    @property
    def required_digits(self):
        return tuple(digit for digit, _ in self.digit_patches)

    def validate_patch_bindings(self, patches):
        for _, patch_id in self.digit_patches:
            patch = patches.get(patch_id)
            if (not isinstance(patch, ObjectContactPatch) or patch.patch_id != patch_id
                    or patch.object_id != self.object_id
                    or patch.geometry_sha256 != self.geometry_sha256):
                raise ValueError("Missing patch or stale/mismatched object geometry binding")
        return dict(mode_is_hypothesis=True, contacts_observed=False,
                    forces_measured=False, physical_success_certified=False)
