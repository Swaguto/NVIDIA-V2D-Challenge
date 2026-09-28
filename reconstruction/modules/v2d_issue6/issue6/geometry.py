"""Metric geometry; no learned models or GPU imports."""

from __future__ import annotations

import numpy as np
import trimesh


def rigid(value):
    a = np.asarray(value, dtype=float)
    if (
        a.shape != (4, 4)
        or not np.isfinite(a).all()
        or not np.allclose(a[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(a[:3, :3]), 1, atol=1e-5)
    ):
        raise ValueError("Expected rigid 4x4 transform (metres, column vectors)")
    return a


def transform(points, matrix):
    t = rigid(matrix)
    return np.asarray(points) @ t[:3, :3].T + t[:3, 3]


def camera(c):
    if c.get("convention") != "rectified_pinhole":
        raise ValueError("Supply rectified images/intrinsics; raw distorted images are unsupported")
    vals = np.array([c[k] for k in ("fx", "fy", "cx", "cy", "width", "height")], float)
    if not np.isfinite(vals).all() or min(vals[:2]) <= 0 or min(vals[4:]) <= 0:
        raise ValueError("Invalid camera intrinsics")
    if any(float(c[k]) != int(c[k]) for k in ("width", "height")):
        raise ValueError("Camera dimensions must be integers")
    return c


def decode_depth(pixels, encoding, valid=None):
    x = np.asarray(pixels, dtype=float)
    if x.ndim != 2:
        raise ValueError("Depth must be a 2D image")
    with np.errstate(divide="ignore", invalid="ignore"):
        if encoding == "inverse_u16":
            z = 65535.0 / x - 1.0
            ok = (x > 0) & (x < 65535)
        elif encoding == "millimetres_u16":
            z = x / 1000.0
            ok = x > 0
        elif encoding == "metres":
            z = x.copy()
            ok = x > 0
        else:
            raise ValueError(f"Unknown depth encoding: {encoding}")
    ok &= np.isfinite(z) & (z > 0)
    if valid is not None:
        if np.shape(valid) != x.shape:
            raise ValueError("Depth validity mask has wrong shape")
        ok &= np.asarray(valid, bool)
    return np.where(ok, z, np.nan)


def reproject_depth(z, source, target, target_from_source):
    """Nearest-pixel forward splat with z-buffer; holes remain NaN."""
    camera(source)
    camera(target)
    if z.shape != (source["height"], source["width"]):
        raise ValueError("Depth dimensions differ from its source camera")
    ys, xs = np.nonzero(np.isfinite(z) & (z > 0))
    depths = z[ys, xs]
    pts = np.column_stack(
        (
            (xs - source["cx"]) * depths / source["fx"],
            (ys - source["cy"]) * depths / source["fy"],
            depths,
        )
    )
    pts = transform(pts, target_from_source)
    pts = pts[np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)]
    u = np.rint(target["fx"] * pts[:, 0] / pts[:, 2] + target["cx"])
    v = np.rint(target["fy"] * pts[:, 1] / pts[:, 2] + target["cy"])
    w, h = target["width"], target["height"]
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v = u[inside].astype(int), v[inside].astype(int)
    out = np.full(h * w, np.inf)
    np.minimum.at(out, v * w + u, pts[inside, 2])
    out[~np.isfinite(out)] = np.nan
    return out.reshape(h, w)


def hand_geometry(vertices, joints, is_right, scale, translation):
    """Match the upstream renderer: mirror, scale about MESH centroid, translate."""
    vertices, joints = np.array(vertices, float), np.array(joints, float)
    if joints.shape != (21, 3) or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Expected MANO vertices and 21 ordered joints")
    if not np.isfinite(vertices).all() or not np.isfinite(joints).all():
        raise ValueError("Nonfinite hand geometry")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid hand scale")
    if not is_right:
        vertices[:, 0] *= -1
        joints[:, 0] *= -1
    center = vertices.mean(axis=0)
    t = np.asarray(translation, float)
    if t.shape != (3,) or not np.isfinite(t).all():
        raise ValueError("Invalid hand translation")
    return (vertices - center) * scale + center + t, (joints - center) * scale + center + t


def alignment_offset(rendered, measured, hand_mask, occluder, centroid, minimum=256):
    if not (rendered.shape == measured.shape == hand_mask.shape == occluder.shape):
        raise ValueError("Alignment images/masks must share a pixel grid")
    keep = (
        np.asarray(hand_mask, bool)
        & ~np.asarray(occluder, bool)
        & np.isfinite(rendered)
        & (rendered > 0)
        & np.isfinite(measured)
        & (measured > 0)
    )
    n = int(keep.sum())
    if n < minimum:
        return np.zeros(3), {
            "metric_valid": False,
            "n_pixels": n,
            "reason": "insufficient_depth",
        }
    dz = float(np.median(measured[keep] - rendered[keep]))
    c = np.asarray(centroid, float)
    if c[2] <= 0 or c[2] + dz <= 0:
        return np.zeros(3), {
            "metric_valid": False,
            "n_pixels": n,
            "reason": "behind_camera",
        }
    offset = c * dz / c[2]
    return offset, {
        "metric_valid": True,
        "n_pixels": n,
        "dz_m": dz,
        "median_abs_residual_m": float(np.median(np.abs(measured[keep] - rendered[keep] - dz))),
    }


def surface_contacts(points_camera, mesh, camera_from_object):
    points = transform(points_camera, np.linalg.inv(rigid(camera_from_object)))
    # Bounded batches; naive triangle distance avoids an rtree system dependency.
    nearest, distance, face_id = trimesh.proximity.closest_point_naive(mesh, points)
    normals = mesh.face_normals[face_id]
    inside = [None] * len(points)
    if mesh.is_watertight and mesh.is_winding_consistent:
        # Generalized winding number; works for concave closed triangle surfaces.
        inside = []
        for point in points:
            a, b, c = np.moveaxis(mesh.triangles - point, 1, 0)
            la, lb, lc = (np.linalg.norm(q, axis=1) for q in (a, b, c))
            numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
            denominator = (
                la * lb * lc
                + np.einsum("ij,ij->i", a, b) * lc
                + np.einsum("ij,ij->i", b, c) * la
                + np.einsum("ij,ij->i", c, a) * lb
            )
            winding = np.sum(2 * np.arctan2(numerator, denominator)) / (4 * np.pi)
            inside.append(bool(abs(winding) > 0.5))
    return nearest, distance, normals, inside


def bridge_contacts(raw, frame_ids, maximum=15):
    """None is unknown. Never bridge missing frames, unknowns or different objects."""
    result = list(raw)
    i = 0
    while i < len(raw):
        if raw[i] is not False:
            i += 1
            continue
        start = i
        while i < len(raw) and raw[i] is False:
            i += 1
        if (
            start > 0
            and i < len(raw)
            and i - start <= maximum
            and raw[start - 1] is True
            and raw[i] is True
            and np.all(np.diff(frame_ids[start - 1 : i + 1]) == 1)
        ):
            result[start:i] = [True] * (i - start)
    return result
