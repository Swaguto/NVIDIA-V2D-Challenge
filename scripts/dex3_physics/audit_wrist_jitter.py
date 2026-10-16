"""Measure physical wrist and commanded motion without trimming failed frames.

The high pass residual is a motion diagnostic, not task acceptance. A stationary
aborted hand can have zero jitter; abort and contact coverage must accompany it.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation


def motion_residual(times_physical_s, poses_xyzw, cutoff_hz=3.0):
    times, poses = np.asarray(times_physical_s, float), np.asarray(poses_xyzw, float)
    if times.ndim != 1 or poses.shape != (len(times), 7) or len(times) < 40:
        raise ValueError("Need aligned dense wrist samples")
    dt = float(np.median(np.diff(times)))
    if (
        not np.isfinite(np.r_[times, poses.ravel()]).all()
        or dt <= 0
        or not np.allclose(np.diff(times), dt, atol=1e-8)
        or not 0 < cutoff_hz < 0.5 / dt
        or not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1, atol=1e-4)
    ):
        raise ValueError("Invalid physical clock, pose or cutoff")
    rotation = Rotation.from_quat(poses[:, 3:])
    tangent = (rotation * rotation[0].inv()).as_rotvec()
    if np.linalg.norm(tangent, axis=1).max() >= np.pi - 0.1:
        raise ValueError("Rotation window crosses the tangent chart boundary")
    signal = np.c_[poses[:, :3], tangent]
    high = sosfiltfilt(
        butter(3, cutoff_hz, btype="highpass", fs=1 / dt, output="sos"), signal, axis=0
    )
    # Remove only filter boundary transients, identically for every comparison.
    trim = max(1, round(0.1 / dt))
    if len(high) <= 2 * trim:
        raise ValueError("Window too short for boundary trimming")
    high = high[trim:-trim]
    return dict(
        position_residual_rms_mm=float(
            1000 * np.sqrt(np.mean(np.sum(high[:, :3] ** 2, axis=1)))
        ),
        rotation_residual_rms_deg=float(
            np.rad2deg(np.sqrt(np.mean(np.sum(high[:, 3:] ** 2, axis=1))))
        ),
        physical_cutoff_hz=cutoff_hz,
        boundary_trim_physical_s=trim * dt,
        samples_before_filter=len(times),
    )


def audit(run, start_s=5.5, end_s=7.5):
    report = json.loads((run / "report.json").read_text())
    state_file = run / "states.json"
    digest = hashlib.sha256(state_file.read_bytes()).hexdigest()
    if digest != report["state_sha256"]:
        raise ValueError("Simulation readback hash mismatch")
    states = json.loads(state_file.read_text())
    result = dict(
        schema="v2d.wrist-jitter-audit.v1",
        states_sha256=digest,
        source_interval_s=[start_s, end_s],
        physical_task_accepted=False,
        hands={},
    )
    for side in ["left", "right"]:
        rows = [
            h
            for h in states["hands"]
            if h["side"] == side and start_s <= h["source_time_s"] < end_s
        ]
        times = np.array([r["step"] for r in rows]) * report["physics_dt_s"]
        result["hands"][side] = dict(
            target=motion_residual(
                times, [r["target_root_pose_stage_xyzw"] for r in rows]
            ),
            actual=motion_residual(
                times, [r["actual_root_pose_stage_xyzw"] for r in rows]
            ),
            aborted_samples=sum(bool(r.get("reacquisition_failure")) for r in rows),
            required_digit_support_fraction=float(
                np.mean(
                    [
                        (r.get("controller_details") or {}).get(
                            "contact_hold_remaining_s", 0
                        )
                        > 0
                        for r in rows
                    ]
                )
            ),
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-s", type=float, default=5.5)
    parser.add_argument("--end-s", type=float, default=7.5)
    args = parser.parse_args()
    result = audit(args.run, args.start_s, args.end_s)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
