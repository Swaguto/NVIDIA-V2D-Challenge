"""Stage 0: load the challenge camera rig geometry from ``meta/camera_calibration.json``.

The exact on-disk schema is not yet verified against real data, so loading is
deliberately defensive: it accepts intrinsics declared either flat on the
camera entry or under an ``intrinsics`` sub-object, and read fs distortion from
an optional ``dist`` / ``distortion`` list (opencv order) or plumb-bob keyword
fields.  Missing anything required produces a descriptive error instead of a
silent mis-calibration.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover - projections degrade gracefully
    _HAS_CV2 = False


# --------------------------------------------------------------------------- #
# Intrinsics + distortion
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics plus optional opencv (plumb-bob) distortion."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist_coeffs: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.float64)
    )

    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def project(
        self,
        points_world: np.ndarray,
        rot: np.ndarray,
        trans: np.ndarray,
    ) -> np.ndarray:
        """Project ``(N,3)`` world points -> ``(N,2)`` pixels (float), with distortion."""

        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        if _HAS_CV2:
            pixels, _ = cv2.projectPoints(
                pts,
                cv2.Rodrigues(rot)[0],
                np.asarray(trans, dtype=np.float64).reshape(3),
                self.matrix(),
                self.dist_coeffs,
            )
            return pixels.reshape(-1, 2)
        rt = np.asarray(rot, dtype=np.float64)
        tt = np.asarray(trans, dtype=np.float64)
        cam = pts @ rt.T + tt
        p = np.stack(
            [self.fx * cam[:, 0] / cam[:, 2] + self.cx, self.fy * cam[:, 1] / cam[:, 2] + self.cy],
            axis=-1,
        )
        return p

    def project_world_to_camera(self, points_world: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
        """Return ``(N,3)`` camera-frame coordinates (no projection)."""

        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        return pts @ np.asarray(rot, dtype=np.float64).T + np.asarray(trans, dtype=np.float64).reshape(3)


# --------------------------------------------------------------------------- #
# Rig calibration container + parser
# --------------------------------------------------------------------------- #


def _expect_number(container: dict, key: str, fallback: float | None = None) -> float:
    if key in container:
        try:
            return float(container[key])
        except (TypeError, ValueError):
            pass
    if fallback is not None:
        return fallback
    raise KeyError(f"missing required number field {key!r}")


def _parse_intrinsics(camera_entry: dict, name: str) -> CameraIntrinsics:
    src = camera_entry.get("intrinsics", camera_entry)
    src = {k: v for k, v in src.items()} | {
        k: v
        for k, v in camera_entry.items()
        if k
        in (
            "fx",
            "fy",
            "cx",
            "cy",
            "width",
            "height",
            "resolution",
            "dist",
            "distortion",
            "distortion_coefficients",
        )
        and k not in src
    }

    try:
        fx = _expect_number(src, "fx")
        fy = _expect_number(src, "fy")
        cx = _expect_number(src, "cx")
        cy = _expect_number(src, "cy")
    except KeyError as exc:
        raise KeyError(f"camera {name!r}: {exc}") from exc

    resolution = src.get("resolution") or src.get("size") or None
    if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
        width, height = int(resolution[0]), int(resolution[1])
    else:
        width = int(src.get("width", 0) or 0)
        height = int(src.get("height", 0) or 0)

    dist = _parse_distortion(src)
    return CameraIntrinsics(
        fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height, dist_coeffs=dist
    )


def _parse_distortion(src: dict) -> np.ndarray:
    vec = None
    for key in ("dist", "distortion", "dist_coeffs", "distortion_coefficients"):
        value = src.get(key)
        if isinstance(value, dict):
            pip = dict(value)
            vec = np.array(
                [
                    pip.get("k1", 0.0),
                    pip.get("k2", 0.0),
                    pip.get("p1", 0.0),
                    pip.get("p2", 0.0),
                    pip.get("k3", 0.0),
                ],
                dtype=np.float64,
            )
            break
        if isinstance(value, (list, tuple)) and len(value) > 0:
            vec = np.asarray(list(value), dtype=np.float64)
            if vec.size < 5:
                vec = np.pad(vec, (0, 5 - vec.size))
            else:
                vec = vec[:14]
            break
    if vec is None:
        vec = np.zeros(5, dtype=np.float64)
    return np.asarray(vec, dtype=np.float64).reshape(-1)


@dataclass(frozen=True)
class RigCalibration:
    """Parsed rig geometry: per-camera intrinsics + stereo pairs + raw JSON."""

    intrinsics: dict[str, CameraIntrinsics]
    cameras: dict[str, dict]
    stereo_pairs: list[dict]
    raw: dict

    @classmethod
    def load(cls, path: str | os.PathLike) -> "RigCalibration":
        raw = json.loads(Path(path).read_text())
        cams = raw.get("cameras")
        if not isinstance(cams, dict) or not cams:
            raise ValueError(f"{path}: expected a non-empty dict under 'cameras'")
        intrinsics = {name: _parse_intrinsics(entry, name) for name, entry in cams.items()}
        pairs = raw.get("stereo_pairs")
        if not isinstance(pairs, list):
            pairs = []
        return cls(intrinsics=intrinsics, cameras=cams, stereo_pairs=pairs, raw=raw)

    def camera_names(self) -> list[str]:
        return list(self.intrinsics.keys())

    def extrinsic_matrix(self, camera: str) -> np.ndarray | None:
        """Return the 4x4 ``world?->camera`` extrinsic if declared, else None.

        The challenge declares cam<->cam transforms as
        ``extrinsics_to_stereo_left`` (4x4, ``T_left_cam`` of that camera).  Any
        key whose name suggests a pose (``extrinsics_*``) is accepted.
        """
        entry = self.cameras[camera]
        for key in ("extrinsics_to_stereo_left", "extrinsics", "extrinsic", "cam_T_world", "T_world_cam"):
            value = entry.get(key)
            if isinstance(value, (list, tuple)):
                arr = np.asarray(value, dtype=np.float64)
                if arr.size in (12, 16):
                    return arr.reshape(3, 4) if arr.size == 12 else arr.reshape(4, 4)[:3]
            if isinstance(value, dict) and "rotation" in value and "translation" in value:
                rot = np.asarray(value["rotation"], dtype=np.float64).reshape(3, 3)
                trans = np.asarray(value["translation"], dtype=np.float64).reshape(3)
                return np.column_stack([rot, trans])
        return None

    def camera_center(self, camera: str) -> np.ndarray | None:
        """Optic centre in the frame the extrinsic is relative to (world, metres)."""

        ex = self.extrinsic_matrix(camera)
        if ex is None:
            return None
        return self.pose_inverse_transform(ex)[:3, 3]

    @staticmethod
    def pose_inverse_transform(t: np.ndarray) -> np.ndarray:
        """Invert a ``3x4`` (R|t) rigid transform -> ``3x4``."""

        t = np.asarray(t, dtype=np.float64).reshape(3, 4)
        r, tt = t[:, :3], t[:, 3]
        return np.column_stack([r.T, -r.T @ tt])

    def _pair_extrinsic_translations(
        self, pair: dict
    ) -> list[np.ndarray]:
        """Stereo-left-frame translations of a pair's cameras.

        In the challenge rig only the camera physically left of the reference
        declares ``extrinsics_to_stereo_left`` (4x4 pose in the stereo-left
        frame); the reference camera itself carries none and sits at the
        origin.  Return the declared translations and, when exactly one is
        declared, treat the other as the origin reference.
        """
        ts: list[np.ndarray] = []
        for side in ("left", "right"):
            camera = pair.get(side)
            if camera not in self.cameras:
                continue
            ex = self.extrinsic_matrix(camera)
            if ex is not None:
                ts.append(self.pose_inverse_transform(ex)[:3, 3])
        return ts

    def ego_baseline_mm(self) -> tuple[float, float]:
        """(declared, measured) ego stereo baseline in millimetres.

        Declared: ``baseline_m`` on the stereo_pairs entry with ``camera=='ego'``
        (accepted aliases: ``baseline``, ``baseline_mm``).  Measured: the
        distance between the pair's optic centres in the stereo-left reference
        frame (a camera with no ``extrinsics_to_stereo_left`` is that
        reference, i.e. the origin).  Falls back to the declared value when the
        pair geometry cannot be resolved.
        """
        declared = float("nan")
        measured = float("nan")
        ts: list[np.ndarray] = []
        for pair in self.stereo_pairs:
            cam = pair.get("camera") or pair.get("cam") or pair.get("name")
            if cam not in (None, "ego"):
                continue
            for key in ("baseline_m", "baseline"):
                if key in pair:
                    declared = float(pair[key]) * 1e3
                    break
            if "baseline_mm" in pair:
                declared = float(pair["baseline_mm"])
            ts = self._pair_extrinsic_translations(pair)
            break
        if len(ts) == 1:
            measured = float(np.linalg.norm(ts[0])) * 1e3
        elif len(ts) >= 2:
            measured = float(np.linalg.norm(ts[1] - ts[0])) * 1e3
        if not np.isfinite(measured) and np.isfinite(declared):
            measured = declared
        return declared, measured