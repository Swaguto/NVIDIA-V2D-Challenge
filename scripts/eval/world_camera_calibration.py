#!/usr/bin/env python3
"""V2D Track 3 - camera <-> mocap-world calibration (Issue #3).

Thin wrapper around the reusable engine in
``reconstruction/modules/v2d_world_calib``.  Solves ``T_world->cam`` per
camera from GT object geometry + image keypoint correspondences (RANSAC-PnP +
Huber reprojection refinement) and verifies transfer on held-out frames.

    # synthetic correctness proof (no challenge data):
    python scripts/eval/world_camera_calibration.py --self-test

    # real calibration (shared box, keypoint cache from Issue #5/#7):
    python scripts/eval/world_camera_calibration.py \
        --data-root ~/v2d_track3/data/hf/track_3/public \
        --camera ego_cam_a \
        --fit-episodes 3,7,15 \
        --test-episodes 27,43 \
        --keypoints-dir results/keypoints \
        --out-dir results/calib
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parents[1]
_MODULE_DIR = _REPO_ROOT / "reconstruction" / "modules" / "v2d_world_calib"
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from v2d.world_calib.cli.calibrate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())