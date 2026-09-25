"""Mocap-to-camera world calibration package.

Solves the rigid ``world -> camera`` transform (``T_CW``) for the V2D Track 3
rig using ground-truth object geometry as the calibration target:

    Stage 0  Load camera intrinsics / distortion / cam-to-cam extrinsics.
    Stage 1  Build 3D<->2D correspondences (GT mesh points <-> image points).
    Stage 2  (skipped) 3D geometric refinement - no camera-space 3D refs.
    Stage 3  PnP-RANSAC init + robust multi-frame reprojection refinement.
    Verify   Held-out transfer error, spatial error map, cam-to-cam check.
"""

from v2d.world_calib.calibration import camera_geometry as camera_geometry
from v2d.world_calib.calibration.correspondences import (
    WorldCameraCorrespondences as WorldCameraCorrespondences,
)
from v2d.world_calib.calibration.solve import (
    CameraCalibrationSolve as CameraCalibrationSolve,
)

__all__ = [
    "camera_geometry",
    "WorldCameraCorrespondences",
    "CameraCalibrationSolve",
]