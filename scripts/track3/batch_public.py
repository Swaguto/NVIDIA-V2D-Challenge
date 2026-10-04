#!/usr/bin/env python3
"""Run Track 3 public episodes with resumable, per-stage logs and provenance.

The perception and camera stages use video, factory calibration, public CAD,
and object names only. Public object poses are read only by the final scorer.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import cv2
import pyarrow.parquet as pq

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
DATA_DEFAULT = Path(os.environ.get("V2D_PUBLIC", "/home/ubuntu/v2d_track3/data/hf/track_3/public"))


def roster(data: Path):
    for path in sorted((data / "data/chunk-000").glob("episode_*.parquet")):
        episode = int(path.stem.removeprefix("episode_"))
        metadata = pq.ParquetFile(path).schema_arrow.metadata
        names = json.loads(metadata[b"objects"])
        yield episode, names


def run_command(command: list[str], log: Path, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(command), flush=True)
    with log.open("w") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env=env, cwd=ROOT, check=False)
    if completed.returncode:
        raise RuntimeError(f"exit {completed.returncode}: {log}")


def frame_count(data: Path, episode: int) -> int:
    path = data / "videos/chunk-000/observation.images.ego_cam_c" / f"episode_{episode:06d}.mp4"
    video = cv2.VideoCapture(str(path))
    count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    if count <= 0:
        raise ValueError(f"empty video: {path}")
    return count


def stage(phase: str, episode: int, names: list[str], data: Path, root: Path,
          python: str, fp_python: str):
    item = root / f"e{episode:03d}"
    item.mkdir(parents=True, exist_ok=True)
    logs = item / "logs"
    count = frame_count(data, episode)
    names_arg = ",".join(names)
    if phase == "cameras":
        out = item / "camera"
        if (out / "report.json").is_file():
            return "cached"
        run_command([python, str(HERE.parent / "stereo_odometry.py"),
                     "--episode", str(episode), "--data", str(data), "--out", str(out)],
                    logs / "cameras.log")
    elif phase == "masks":
        out = item / "masks"
        if (out / "report.json").is_file():
            return "cached"
        command = [python, str(HERE.parent / "temporal_masks.py"),
                   "--episode", str(episode), "--objects", names_arg,
                   "--data", str(data), "--out", str(out)]
        seeds = HERE.parent / "seed_overrides" / f"e{episode:03d}.json"
        if seeds.is_file():
            command += ["--seed-json", str(seeds)]
        run_command(command, logs / "masks.log")
    elif phase == "poses":
        masks = item / "masks"
        if not (masks / "report.json").is_file():
            raise FileNotFoundError(f"masks missing: {masks}")
        env = dict(os.environ)
        env["PYTHONPATH"] = ":".join(filter(None, [
            env.get("V2D_FP_SHIM", "/home/ubuntu/p3d_shim"),
            env.get("V2D_FOUNDATIONPOSE", "/home/ubuntu/FoundationPose"),
            env.get("PYTHONPATH")]))
        env["V2D_RUNTIME"] = str(ROOT / "reconstruction/modules/v2d_stereo_depth/lib")
        for name in names:
            out = item / "poses"
            if (out / f"e{episode:03d}" / f"{name}_fp_report.json").is_file():
                continue
            run_command([fp_python, str(HERE.parent / "foundationpose_track.py"),
                         "--episode", str(episode), "--object", name,
                         "--data", str(data), "--masks", str(masks), "--out", str(out)],
                        logs / f"pose_{name}.log", env)
    elif phase == "scores":
        camera, poses = item / "camera", item / "poses"
        if not (camera / "report.json").is_file():
            raise FileNotFoundError(f"camera missing: {camera}")
        if any(not (poses / f"e{episode:03d}" / f"{name}_fp_report.json").is_file()
               for name in names):
            raise FileNotFoundError(f"object poses incomplete: {poses}")
        rectification = item / "rectification.json"
        if not rectification.exists():
            sys.path.insert(0, str(ROOT / "reconstruction/modules/v2d_stereo_depth/lib"))
            from stereo_depth import load_rig
            import numpy as np
            rig = load_rig()
            rectification.write_text(json.dumps({"episode": episode,
                "camera_frame": rig.left, "coordinate_frame": "rectified_camera",
                "R1": np.asarray(rig.R1).tolist()}))
        world = item / "world"
        if not (world / f"e{episode:03d}" / "ego_object_poses" /
                f"e{episode:03d}" / "conversion_report.json").is_file():
            run_command([python, str(ROOT / "scripts/eval/convert_camera_poses.py"),
                         "--episode", str(episode), "--predictions", str(poses),
                         "--camera-root", str(camera), "--rectification", str(rectification),
                         "--require-independent", "--out", str(world)], logs / "convert.log")
        inferred = item / "inferred"
        if not (inferred / f"e{episode:03d}" / "inference_report.json").is_file():
            run_command([python, str(HERE.parent / "infer_occlusion.py"),
                         "--episode", str(episode), "--objects", names_arg,
                         "--frames", str(count), "--predictions", str(world),
                         "--out", str(inferred)], logs / "infer.log")
        for label, candidate in (("observed", world), ("inferred", inferred)):
            report = item / f"scores_{label}.json"
            if report.is_file():
                continue
            run_command([python, str(ROOT / "scripts/eval/score_track3_official.py"),
                         "--episodes", str(episode), "--predictions", str(candidate),
                         "--data", str(data), "--report", str(report)],
                        logs / f"score_{label}.log")
    elif phase == "overlays":
        out = item / "overlay.mp4"
        if out.is_file():
            return "cached"
        run_command([python, str(HERE.parent / "render_overlay.py"),
                     "--episode", str(episode), "--predictions", str(item / "poses"),
                     "--data", str(data), "--out", str(out), "--frames", str(count),
                     "--stride", "1", "--fps", "15"], logs / "overlay.log")
    else:
        raise ValueError(phase)
    return "done"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["cameras", "masks", "poses", "scores", "overlays"],
                        required=True)
    parser.add_argument("--data", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episodes", default="", help="comma list; default is all public episodes")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--fp-python", default=os.environ.get(
        "V2D_FP_PYTHON", "/home/ubuntu/fp_venv/bin/python"))
    args = parser.parse_args()
    requested = {int(x) for x in args.episodes.split(",") if x.strip()} or None
    results = {}
    for episode, names in roster(args.data):
        if requested is not None and episode not in requested:
            continue
        try:
            results[str(episode)] = stage(args.phase, episode, names, args.data,
                                          args.root, args.python, args.fp_python)
        except Exception as error:
            results[str(episode)] = f"failed: {error}"
            traceback.print_exc()
        manifest = args.root / f"batch_{args.phase}.json"
        manifest.write_text(json.dumps(results, indent=2))
        print(f"{args.phase} episode {episode}: {results[str(episode)]}", flush=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
