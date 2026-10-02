#!/usr/bin/env python3
"""Metric depth for Track 3 from the ego stereo pair, RGB only.

Track 3's evaluation split releases **no depth**, but the rig is a stereo camera and the
organisers publish everything needed to recover it. From
``track_3/public/meta/camera_calibration.json``:

    "Each stereo pair can be rectified offline with cv2.stereoRectify + StereoSGBM;
     depth_m = fx * baseline_m / disparity_px."

The ego pair is ``ego_cam_c`` (left) <-> ``ego_cam_b`` (right), baseline ~0.0755 m, both
1280x800. Both carry an OAK ``Perspective`` distortion model with **14** coefficients
(rational polynomial + thin prism), which OpenCV's ``undistortPointsIter`` accepts. The
factory ``intrinsic_matrix`` is therefore valid only for the raw distorted image.

Per frame: undistort both (14-coeff) -> rectify to a common rectified pair -> StereoSGBM
-> metric depth in the **left** camera's rectified frame. Because depth is metric, object
scale derived from it is metric too, which matters: the official frame-0 alignment is a
rigid SE(3), not Sim(3), so scale error is scored.

Poses may be reported directly in the rectified-left frame; the metric's frame-0 SE(3)
alignment means any single consistent frame is acceptable.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

CALIB_PATH = os.environ.get(
    "V2D_CALIB",
    "/home/ubuntu/v2d_track3/data/hf/track_3/public/meta/camera_calibration.json",
)


@dataclass
class StereoRig:
    """Raw-camera calibration plus the derived common rectified pair."""
    left: str
    right: str
    K_l: np.ndarray            # factory intrinsics, RAW distorted image
    K_r: np.ndarray
    D_l: np.ndarray            # 14 OAK Perspective coefficients
    D_r: np.ndarray
    R_lr: np.ndarray           # left_cam -> right_cam rotation
    T_lr: np.ndarray           # left_cam -> right_cam translation, metres
    baseline_m: float
    size: tuple                # (width, height)
    R1: np.ndarray = field(default=None, repr=False)
    R2: np.ndarray = field(default=None, repr=False)
    P1: np.ndarray = field(default=None, repr=False)
    P2: np.ndarray = field(default=None, repr=False)
    Q: np.ndarray = field(default=None, repr=False)
    Ks: np.ndarray = field(default=None, repr=False)   # common undistorted pinhole

    @property
    def fx_rect(self) -> float:
        return float(self.P1[0, 0])

    @property
    def fy_rect(self) -> float:
        return float(self.P1[1, 1])

    # -- raw -> rectified ----------------------------------------------------
    def rectify_pair(self, left_img, right_img):
        """Canonical OpenCV stereo rectification: undistort+rectify in one remap."""
        pl = cv2.remap(left_img, *self._maps_l, cv2.INTER_LINEAR)
        pr = cv2.remap(right_img, *self._maps_r, cv2.INTER_LINEAR)
        return pl, pr


def load_rig(calib_path: str = CALIB_PATH, device: str = "ego") -> StereoRig:
    """Build the rectified stereo pair straight from the published factory calibration.

    Uses the canonical OpenCV stereo path rather than hand-rolling the distortion model:
    ``stereoRectify`` + ``initUndistortRectifyMap``, which invert the 14-coefficient OAK
    Perspective model properly. (Feeding those coefficients to ``undistortPointsIter`` is
    wrong twice over: it maps ideal->distorted, the opposite direction, and yields maps
    ~1e12 in magnitude.)
    """
    cal = json.loads(Path(calib_path).read_text())
    pair = next(p for p in cal["stereo_pairs"] if p["camera"] == device)
    left, right = pair["left"], pair["right"]
    cl, cr = cal["cameras"][left], cal["cameras"][right]
    l2r = np.array(pair["left_to_right"], np.float64)
    K_l = np.array(cl["intrinsic_matrix"], np.float64)
    K_r = np.array(cr["intrinsic_matrix"], np.float64)
    D_l = np.array(cl["distortion_coefficients"], np.float64)
    D_r = np.array(cr["distortion_coefficients"], np.float64)
    size = tuple(cl["resolution"])

    rig = StereoRig(left=left, right=right, K_l=K_l, K_r=K_r, D_l=D_l, D_r=D_r,
                    R_lr=l2r[:3, :3], T_lr=l2r[:3, 3],
                    baseline_m=float(abs(l2r[0, 3])), size=size)

    w, h = size
    # OpenCV convention: x_right = R*x_left + T, so T_lr is passed unchanged.
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_l, D_l.reshape(1, -1), K_r, D_r.reshape(1, -1), (w, h),
        rig.R_lr, rig.T_lr, flags=cv2.CALIB_ZERO_DISPARITY)
    rig.R1, rig.R2, rig.P1, rig.P2, rig.Q = R1, R2, P1, P2, Q
    rig._maps_l = cv2.initUndistortRectifyMap(K_l, D_l.reshape(1, -1), R1, P1, (w, h), cv2.CV_32FC1)
    rig._maps_r = cv2.initUndistortRectifyMap(K_r, D_r.reshape(1, -1), R2, P2, (w, h), cv2.CV_32FC1)
    return rig


def stereo_depth(rig: StereoRig, left_img, right_img, num_disparities: int = 128,
                 block_size: int = 5, min_valid_px: int = 400):
    """Return (depth_m, valid) in the rectified left-camera frame."""
    pl, pr = rig.rectify_pair(left_img, right_img)
    g_l = cv2.cvtColor(pl, cv2.COLOR_BGR2GRAY)
    g_r = cv2.cvtColor(pr, cv2.COLOR_BGR2GRAY)

    depth = np.zeros(g_l.shape, np.float32)
    valid = np.zeros(g_l.shape, bool)
    for nd, bs in ((num_disparities, block_size),
                   (num_disparities - 64, 7),
                   (num_disparities - 96, 9)):
        if nd < 16:
            break
        matcher = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=nd, blockSize=bs,
            P1=8 * 3 * bs, P2=32 * 3 * bs,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        disp = matcher.compute(g_l, g_r).astype(np.float32) / 16.0
        depth = np.zeros_like(disp)
        m = disp > 0
        depth[m] = rig.fx_rect * rig.baseline_m / disp[m]
        depth[~np.isfinite(depth)] = 0.0
        valid = m & (depth > 0.05) & (depth < 3.0)
        if valid.sum() >= min_valid_px:
            break
    return depth, valid


def backproject(depth, valid, P):
    """Depth map -> (points (N,3) in the rectified left frame, pixels (N,2) uv)."""
    ys, xs = np.nonzero(valid)
    z = depth[ys, xs].astype(np.float64)
    x = (xs - P[0, 2]) * z / P[0, 0]
    y = (ys - P[1, 2]) * z / P[1, 1]
    return np.stack([x, y, z], 1), np.stack([xs, ys], 1).astype(np.float64)

def backproject_full(depth, valid, P):
    """Dense (H*W, 3) point array over *every* pixel, zeros where invalid.

    Lets callers index with a full-resolution mask (``mask.ravel() & valid.ravel()``), which
    a valid-only point list cannot accept. Back-project with the rectified left P.
    """
    h, w = depth.shape
    xs = np.tile(np.arange(w, dtype=np.float64), h)
    ys = np.repeat(np.arange(h, dtype=np.float64), w)
    v = valid.ravel()
    z = np.where(v, depth.ravel().astype(np.float64), 0.0)
    x = np.where(v, (xs - P[0, 2]) * z / P[0, 0], 0.0)
    y = np.where(v, (ys - P[1, 2]) * z / P[1, 1], 0.0)
    return np.stack([x, y, z], 1)
