"""VM-side driver: build the calibration keypoint cache with tracked 6D poses.

Runs on the GPU VM (Docker + the reconstruction host packages installed).
For selected public episodes and ego cameras, produces, per (camera, object):

  poses/{camera}/{object_name}/{frame_idx:06d}.json   Transform3d object->cam

plus the undistorted frames, SAM2 masks, stereo depth, and a manifest.  The
pose outputs are what gets pushed back; the repo-side converter turns them
into the ``{camera}_ep_{ep:06d}.npy`` keypoint cache for the calibrator.

Pipeline per camera:  video -> extract frames -> undistort (real 14-pt
distortion) -> Grounding-DINO bboxes -> SAM2 masks -> FoundationStereo depth
(stereo pair) -> FoundationPose RGB-D 6D tracking per object.

Usage (run from the reconstruction/ dir on the host):

  python -m v2d.world_calib.vm.track_pipeline \
      --data-root ~/v2d_track3/data/hf/track_3/public \
      --work ~/v2d_calib_work \
      --episode 2 \
      --cameras ego_cam_b ego_cam_c
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()


def _require(pkg_name: str) -> None:
    try:
        __import__(pkg_name)
    except ImportError:
        sys.exit(f"host package missing: {pkg_name}. "
                 "Install host packages first (see reconstruction/scripts/install_packages.sh).")


_require("cv2")
_require("pyarrow")  # read parquet object lists
import cv2  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


def load_rig(data_root: Path) -> dict:
    rig = json.loads((data_root / "meta" / "camera_calibration.json").read_text())
    if not rig.get("cameras"):
        sys.exit(f"{data_root}: camera_calibration.json missing a 'cameras' entry")
    return rig


def load_info(data_root: Path) -> dict:
    info = json.loads((data_root / "meta" / "info.json").read_text())
    for key in ("fps", "episode_indices", "video_path", "data_path"):
        assert key in info, f"info.json missing {key!r}"
    return info


def episode_objects(data_root: Path, episode: int) -> list[str]:
    parquet = data_root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    if not parquet.exists():
        sys.exit(f"{parquet}: episode parquet not found")
    meta = pq.ParquetFile(parquet).schema_arrow.metadata
    return list(json.loads(meta[b"objects"]))


def mesh_path_for(data_root: Path, object_name: str) -> Path:
    mesh_root = data_root / "mesh"
    candidates = (
        mesh_root / object_name / f"{object_name}.glb",
        mesh_root / object_name / f"{object_name}_visual.glb",
        mesh_root / object_name / f"{object_name}_collision.glb",
        mesh_root / object_name / "mesh" / f"{object_name}.glb",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    sys.exit(f"mesh not found for {object_name!r} under {mesh_root}")


def stereo_partner(rig: dict, camera: str) -> str:
    for pair in rig.get("stereo_pairs", []):
        left, right = pair.get("left"), pair.get("right")
        if camera == left:
            return right
        if camera == right:
            return left
    return None


def extract_frames(video: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "%06d.png")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
         "-start_number", "0", "-vsync", "0", pattern],
        check=True,
    )
    return len(list(out_dir.glob("*.png")))


def undistort_to(frames_dir: Path, cam_entry: dict, out_dir: Path, intrinsics_path: Path) -> None:
    K = np.asarray(cam_entry["intrinsic_matrix"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(cam_entry["distortion_coefficients"], dtype=np.float64).reshape(-1)
    W, H = int(cam_entry["resolution"][0]), int(cam_entry["resolution"][1])
    out_dir.mkdir(parents=True, exist_ok=True)
    intrinsics_path.parent.mkdir(parents=True, exist_ok=True)
    intrinsics_path.write_text(json.dumps({
        "fx": float(K[0, 0]), "fy": float(K[1, 1]),
        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "width": W, "height": H,
    }, indent=2))
    for png in sorted(frames_dir.glob("*.png")):
        img = cv2.imread(str(png))
        if img is None:
            continue
        dst = cv2.undistort(img, K, dist)
        cv2.imwrite(str(out_dir / png.name), dst)


def encode_mp4(frames_dir: Path, mp4: Path, fps: float) -> None:
    mp4.parent.mkdir(parents=True, exist_ok=True)
    first = sorted(frames_dir.glob("*.png"))[0]
    h, w = cv2.imread(str(first)).shape[:2]
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for png in sorted(frames_dir.glob("*.png")):
        writer.write(cv2.imread(str(png)))
    writer.release()


def make_sam2_prompts(box_pixels: list[float], width: int, height: int,
                      frame_index: int, object_id: int, path: Path) -> None:
    x0, y0, x1, y1 = box_pixels
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"prompts": [{
        "frame_index": int(frame_index),
        "object_id": int(object_id),
        "box": {
            "x0": round(x0 / width, 6),
            "y0": round(y0 / height, 6),
            "x1": round(x1 / width, 6),
            "y1": round(y1 / height, 6),
        },
    }]}, indent=2))


def pick_detection(detections_json: dict) -> tuple[int, list[float]] | None:
    best = None  # (confidence, frame_idx, box)
    for frame_str, dets in detections_json.items():
        if not dets:
            continue
        det = max(dets, key=lambda d: d["confidence"])
        if best is None or det["confidence"] > best[0]:
            box = det["box"]
            best = (det["confidence"], int(frame_str), [box["x0"], box["y0"], box["x1"], box["y1"]])
    return None if best is None else (best[1], best[2])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, help="track_3/public directory")
    ap.add_argument("--work", required=True, help="scratch work dir (created)")
    ap.add_argument("--episode", type=int, default=2)
    ap.add_argument("--cameras", nargs="+", default=["ego_cam_b", "ego_cam_c"])
    ap.add_argument("--gdino-model-dir", required=True)
    ap.add_argument("--sam2-weights", required=True)
    ap.add_argument("--fp-weights", required=True)
    ap.add_argument("--fs-model-dir", required=True)
    ap.add_argument("--gdino-prompt", default=None,
                    help="text prompt (grounding dino); defaults to object names")
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    work = Path(args.work).resolve()
    frames_root = work / "frames"
    video_root = work / "video"
    depth_root = work / "depth"
    masks_root = work / "masks"
    poses_root = work / "poses"
    intrinsics_dir = work / "intrinsics"
    detections_dir = work / "detections"
    prompts_dir = work / "prompts"

    rig = load_rig(data_root)
    info = load_info(data_root)
    episode = args.episode
    objects = episode_objects(data_root, episode)
    pair = {c: stereo_partner(rig, c) for c in args.cameras}
    missing_pair = [c for c, p in pair.items() if p is None]
    if missing_pair:
        print(f"WARNING: no stereo partner for {missing_pair}; depth will be skipped "
              "(tracking may be less accurate), or add mono-depth first.", file=sys.stderr)

    video_tmpl = info["video_path"].format(episode_chunk=0, video_key="observation.images.{cam}",
                                           episode_index=episode)
    print(f"episode {episode} objects: {objects}")
    print(f"parquet row count will be matched by frame order (via video).")

    # 1. Per-camera undistortion + pinhole intrinsics + mp4.
    for cam in args.cameras:
        cam_entry = rig["cameras"][cam]
        video = data_root / video_tmpl.format(cam=cam)
        raw = frames_root / "raw" / cam
        und = frames_root / cam
        print(f"[extract+undistort] {cam} <- {video}")
        n = extract_frames(video, raw)
        undistort_to(raw, cam_entry, und, intrinsics_dir / f"{cam}.json")
        encode_mp4(und, video_root / f"{cam}.mp4", float(info["fps"]))
        print(f"  {n} frames undistorted -> {und}")

    # 2. Stereo depth (FoundationStereo) for cameras that have a pair.
    from v2d.foundation_stereo.docker.run_image_list_to_depth import run_image_list_to_depth

    ego_baseline = None
    for sp in rig.get("stereo_pairs", []):
        if sp.get("camera") == "ego":
            ego_baseline = float(sp["baseline_m"])
            break
    for cam in args.cameras:
        partner = pair.get(cam)
        if partner is None:
            continue
        if ego_baseline is None:
            print(f"WARNING: no ego baseline declared for {cam}; skipping depth", file=sys.stderr)
            continue
        cam_entry = rig["cameras"][cam]
        print(f"[fs] depth for {cam} (partner {partner}, baseline {ego_baseline:.4f} m)")
        run_image_list_to_depth(
            left_dir=str(frames_root / cam),
            right_dir=str(frames_root / partner),
            depth_folder=str(depth_root / cam),
            intrinsics_folder=str(depth_root / cam / "_intr"),
            model_dir=args.fs_model_dir,
            fx=float(cam_entry["fx"]), fy=float(cam_entry["fy"]),
            cx=float(cam_entry["cx"]), cy=float(cam_entry["cy"]),
            baseline=ego_baseline,
            dev=args.dev,
        )

    # 3. Per object: detection -> SAM2 masks -> FoundationPose.
    from v2d.grounding_dino.docker.run_video_to_object_bboxes import run_video_to_object_bboxes
    from v2d.sam2.docker.run_video_to_masks import run_video_to_masks
    from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses

    for cam in args.cameras:
        cam_entry = rig["cameras"][cam]
        W, H = int(cam_entry["resolution"][0]), int(cam_entry["resolution"][1])
        for obj in objects:
            prompt = args.gdino_prompt or obj
            det_json = detections_dir / f"{cam}_{obj}.json"
            detections_dir.mkdir(parents=True, exist_ok=True)
            print(f"[gdino] {cam} prompt={prompt!r}")
            run_video_to_object_bboxes(
                video_path=str(video_root / f"{cam}.mp4"),
                output_path=str(det_json),
                prompt=prompt,
                model_dir=args.gdino_model_dir,
                dev=args.dev,
            )
            picked = pick_detection(json.loads(det_json.read_text()))
            if picked is None:
                print(f"  !! no detection for {obj} on {cam}; skipping", file=sys.stderr)
                continue
            ref_frame, box_px = picked
            print(f"  {obj}: reference frame {ref_frame}, box {box_px}")

            prompts_path = prompts_dir / f"{cam}_{obj}.json"
            make_sam2_prompts(box_px, W, H, ref_frame, 0, prompts_path)

            masks_dir = masks_root / cam / obj
            print(f"[sam2] {cam}/{obj} -> masks")
            run_video_to_masks(
                video_path=str(video_root / f"{cam}.mp4"),
                prompts_path=str(prompts_path),
                masks_dir=str(masks_dir),
                weights_dir=args.sam2_weights,
                dev=args.dev,
            )

            mesh = mesh_path_for(data_root, obj)
            poses_dir = poses_root / cam / obj
            print(f"[fp] {cam}/{obj} mesh={mesh} ref={ref_frame}")
            run_video_to_poses(
                video_path=str(video_root / f"{cam}.mp4"),
                depth_folder=str(depth_root / cam),
                masks_folder=str(masks_dir / "0"),
                camera_intrinsics_path=str(intrinsics_dir / f"{cam}.json"),
                mesh_path=str(mesh),
                poses_dir=str(poses_dir),
                weights_dir=args.fp_weights,
                reference_frame=ref_frame,
                dev=args.dev,
            )

    manifest = {
        "episode": episode,
        "objects": objects,
        "cameras": args.cameras,
        "fps": info["fps"],
        "intrinsics": {c: str(intrinsics_dir / f"{c}.json") for c in args.cameras},
        "poses": {c: {o: str(poses_root / c / o) for o in objects} for c in args.cameras},
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDONE. Outputs under {work}.")
    print("Push back: the full 'poses' tree (and 'intrinsics') from manifest.json")


if __name__ == "__main__":
    main()