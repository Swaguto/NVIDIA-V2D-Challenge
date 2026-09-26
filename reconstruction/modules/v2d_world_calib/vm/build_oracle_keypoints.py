"""Build an *oracle* keypoint cache by projecting GT scene points with the
challenge-published camera extrinsics — no tracking needed.

The cache files that result are byte-for-byte the same format the calibrator
consumes (``{camera}_ep_{ep:06d}.npy``, ``(T,B,K,5)``), so this both (a) proves
the real-data path of the engine end to end, and (b) provides a published-
extrinsics ground truth to score the recovered ``world->cam`` against.

Semantics: the advertised ``extrinsics_to_stereo_left`` is a camera pose in the
stereo-left frame.  For the exo rig the stereo-left camera is ``exo_cam_b`` (no
extrinsic declared -> origin of the rig frame), so the rig frame is treated as
the "world" here.

Run locally (no GPU / Docker):

  python -m v2d.world_calib.vm.build_oracle_keypoints \
      --data-root ~/v2d_track3/data/hf/track_3/public \
      --keypoints-dir results/oracle_kp \
      --episodes 2 \
      --cameras exo_cam_a exo_cam_c
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))  # v2d.world_calib importable when run as a file

from v2d.world_calib.calibration.camera_geometry import RigCalibration


def _qmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _mesh_points(mesh_path: Path, k: int, seed: int = 0) -> np.ndarray:
    import trimesh

    m = trimesh.load(str(mesh_path), force=None)
    if isinstance(m, trimesh.Scene):
        m = [g for g in m.geometry.values()][0]
    v = np.asarray(m.vertices, dtype=np.float64)
    idx = np.random.default_rng(seed).choice(len(v), k, replace=False)
    return v[idx]


def world_to_cam_for(rig: RigCalibration, camera: str) -> np.ndarray:
    ex = rig.extrinsic_matrix(camera)
    if ex is None:
        return np.eye(4)
    r, t = ex[:, :3], ex[:, 3]
    m = np.eye(4)
    m[:3, :3] = r.T
    m[:3, 3] = -r.T @ t
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--keypoints-dir", required=True)
    ap.add_argument("--episodes", default="2")
    ap.add_argument("--cameras", nargs="+", default=["exo_cam_a", "exo_cam_c"])
    ap.add_argument("--points", type=int, default=64)
    ap.add_argument(
        "--distort",
        action="store_true",
        help="project with the full raw (distorted) camera model; default is "
        "pinhole-space pixels (matches the undistorted tracking domain).",
    )
    args = ap.parse_args()

    base = Path(args.data_root)
    rig = RigCalibration.load(base / "meta" / "camera_calibration.json")
    out_dir = Path(args.keypoints_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = [int(e) for e in args.episodes.split(",") if e.strip()]
    for ep in episodes:
        # object order + GT poses come from the reference loader (parquet)
        import importlib.util  # noqa: E402

        spec = importlib.util.spec_from_file_location(
            "reference_loader",
            Path(__file__).resolve().parents[4] / "scripts" / "eval" / "reference_loader.py",
        )
        refmod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("reference_loader", refmod)
        spec.loader.exec_module(refmod)
        ref = refmod.load_reference(ep, base)
        pose = np.asarray(ref.pose_xyzw, dtype=np.float64)  # (T,B,7) xyzw
        vis = np.asarray(ref.visible, dtype=bool)  # (T,B)
        objects = list(ref.object_names)
        T, B, _ = pose.shape

        pts = [_mesh_points(base / "mesh" / o / f"{o}.glb", args.points) for o in objects]

        for camera in args.cameras:
            intr = rig.intrinsics[camera]
            K = intr.matrix()
            w, h = intr.width, intr.height
            mwc = world_to_cam_for(rig, camera)
            cache = np.full((T, B, args.points, 5), 1e9, dtype=np.float64)
            for t in range(T):
                for b in range(B):
                    if not vis[t, b]:
                        continue
                    pw = pts[b] @ _qmat(pose[t, b, 3:]).T + pose[t, b, :3]
                    ph = np.column_stack([pw, np.ones(args.points)])
                    pc = (mwc @ ph.T).T[:, :3]
                    z = pc[:, 2]
                    front = z > 0
                    if args.distort:
                        px = intr.project(pw, mwc[:3, :3], mwc[:3, 3])
                        u, v = px[:, 0], px[:, 1]
                    else:
                        safe = np.where(front, z, 1.0)
                        u = K[0, 0] * pc[:, 0] / safe + K[0, 2]
                        v = K[1, 1] * pc[:, 1] / safe + K[1, 2]
                    cache[t, b, :, :3] = pts[b]
                    cache[t, b, :, 3] = u
                    cache[t, b, :, 4] = v
                    cache[t, b, ~front, 3:] = 1e9
                    bad = (u < 0) | (u >= w) | (v < 0) | (v >= h)
                    cache[t, b, bad & front, 3:] = 1e9
            fname = out_dir / f"{camera}_ep_{ep:06d}.npy"
            np.save(fname, cache)
            frac = (cache[:, :, :, 3] < 1e8).mean()
            mwc_out = mwc[:3]
            r_out = mwc_out[:3, :3]
            t_out = mwc_out[:, 3]
            print(f"{fname.name}: {cache.shape}  in-view {frac:.1%}  "
                  f"|t|={np.linalg.norm(t_out):.4f} m")

    pub = {}
    for camera in args.cameras:
        m = world_to_cam_for(rig, camera)
        pub[camera] = {"R": m[:3, :3].tolist(), "t": m[:3, 3].tolist()}
    (out_dir / "oracle_extrinsics.json").write_text(json.dumps(pub, indent=2))
    print("oracle extrinsics (rig frame) written to", out_dir / "oracle_extrinsics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())