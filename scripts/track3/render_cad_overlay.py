#!/usr/bin/env python3
"""Render textured CAD meshes into rectified Track 3 video with stereo occlusion."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pyrender
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reconstruction/modules/v2d_stereo_depth/lib"))
from stereo_depth import load_rig, stereo_depth  # noqa: E402


def mesh_path(data: Path, name: str) -> Path:
    folder = data / "mesh" / name
    for stem in (name, f"{name}_visual", f"{name}_collision"):
        candidate = folder / f"{stem}.glb"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"CAD mesh missing for {name}")


def pose_matrix(record: dict) -> np.ndarray:
    quaternion = np.asarray(record["rotation"], dtype=float)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    pose[:3, 3] = record["translation"]
    scale = float(np.atleast_1d(record.get("scale", 1.0))[0])
    pose[:3, :3] *= scale
    return pose


def video_path(data: Path, camera: str, episode: int) -> Path:
    return (data / "videos/chunk-000" / f"observation.images.{camera}" /
            f"episode_{episode:06d}.mp4")


def run(episode: int, data: Path, predictions: Path, out: Path,
        start: int, frames: int, alpha: float, occlude: bool,
        colorize: bool) -> dict:
    rig = load_rig()
    prefix = predictions / f"e{episode:03d}/ego_object_poses/e{episode:03d}"
    names = sorted(path.name for path in prefix.iterdir() if path.is_dir())
    if not names:
        raise ValueError(f"no object predictions: {prefix}")
    left = cv2.VideoCapture(str(video_path(data, rig.left, episode)))
    right = cv2.VideoCapture(str(video_path(data, rig.right, episode)))
    if not left.isOpened() or not right.isOpened():
        raise FileNotFoundError(f"stereo videos for episode {episode}")
    total = int(left.get(cv2.CAP_PROP_FRAME_COUNT))
    if total != int(right.get(cv2.CAP_PROP_FRAME_COUNT)):
        raise ValueError("stereo frame count mismatch")
    first = max(0, start)
    last = min(total, first + frames if frames else total)
    if first >= last:
        raise ValueError("empty requested frame range")
    width = int(left.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K = np.asarray(rig.P1[:3, :3], dtype=float)
    scene = pyrender.Scene(ambient_light=np.array([0.45, 0.45, 0.45]),
                            bg_color=np.array([0, 0, 0, 0]))
    camera = pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                                      znear=0.02, zfar=4.0)
    camera_pose = np.diag([1.0, -1.0, -1.0, 1.0])
    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=camera_pose)
    nodes = {}
    render_meshes = {}
    directories = {}
    tints = ([1.0, 0.45, 0.70, 1.0], [0.45, 0.85, 1.0, 1.0],
             [1.0, 0.80, 0.35, 1.0])
    for index, name in enumerate(names):
        mesh = trimesh.load(str(mesh_path(data, name)), force="mesh", process=False)
        render_meshes[name] = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        if colorize:
            for primitive in render_meshes[name].primitives:
                primitive.material.baseColorFactor = np.asarray(
                    tints[index % len(tints)], dtype=np.float32)
        nodes[name] = scene.add(render_meshes[name], pose=np.eye(4))
        directories[name] = prefix / name / "object_to_cam" / rig.left
    renderer = pyrender.OffscreenRenderer(width, height)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             15.0, (width, height))
    if not writer.isOpened():
        raise OSError(f"cannot write video: {out}")
    for _ in range(first):
        left.read()
        right.read()
    shown = {name: 0 for name in names}
    try:
        for frame in range(first, last):
            ok_l, raw_l = left.read()
            ok_r, raw_r = right.read()
            if not ok_l or not ok_r:
                break
            rect_l, _ = rig.rectify_pair(raw_l, raw_r)
            present = []
            for name in names:
                path = directories[name] / f"{frame:06d}.json"
                if not path.is_file():
                    if nodes[name] is not None:
                        scene.remove_node(nodes[name])
                        nodes[name] = None
                    continue
                pose = pose_matrix(json.loads(path.read_text()))
                if nodes[name] is None:
                    nodes[name] = scene.add(render_meshes[name], pose=pose)
                else:
                    scene.set_pose(nodes[name], pose)
                present.append(name)
                shown[name] += 1
            overlay = rect_l.copy()
            if present:
                color, model_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
                visible = (color[:, :, 3] > 0) & (model_depth > 0)
                if occlude:
                    measured, valid = stereo_depth(rig, raw_l, raw_r)
                    occluder = (visible & valid &
                                (model_depth > measured + 0.04)).astype(np.uint8)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                    occluder = cv2.morphologyEx(occluder, cv2.MORPH_OPEN, kernel)
                    occluder = cv2.dilate(occluder, kernel, iterations=1)
                    visible &= occluder == 0
                model_bgr = cv2.cvtColor(color[:, :, :3], cv2.COLOR_RGB2BGR)
                overlay[visible] = (alpha * model_bgr[visible] +
                                    (1 - alpha) * rect_l[visible]).astype(np.uint8)
            label = f"ep{episode} frame {frame}/{total - 1}  CAD: "
            label += ", ".join(present) if present else "no observed pose"
            cv2.rectangle(overlay, (8, 8), (min(width - 8, 20 + 13 * len(label)), 49),
                          (0, 0, 0), -1)
            cv2.putText(overlay, label, (15, 37), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(overlay)
            if frame % 50 == 0:
                print(f"CAD overlay {frame}/{last - 1}: {present}", flush=True)
    finally:
        writer.release()
        renderer.delete()
        left.release()
        right.release()
    return {"episode": episode, "start": first, "end_exclusive": last,
            "objects": names, "rendered_frames_by_object": shown,
            "occlusion": "stereo_depth" if occlude else "none",
            "alpha": alpha, "colorize": colorize, "output": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=0,
                        help="number of frames; 0 uses the rest of the episode")
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--no-occlusion", action="store_true")
    parser.add_argument("--natural-material", action="store_true",
                        help="retain the original CAD material colors")
    args = parser.parse_args()
    if not 0 < args.alpha <= 1:
        parser.error("--alpha must be in (0, 1]")
    print(json.dumps(run(args.episode, args.data, args.predictions, args.out,
                         args.start, args.frames, args.alpha,
                         not args.no_occlusion, not args.natural_material),
          indent=2), flush=True)


if __name__ == "__main__":
    main()
