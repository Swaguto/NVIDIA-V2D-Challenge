"""Render observed CAD and final temporal MANO together in the calibrated camera."""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import pyrender
import torch
import trimesh
from issue6.geometry import transform
from issue6.manifest import camera_from_object, digest, file_hash, load
from issue6.pipeline import load_mesh
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument("--episode", type=int, required=True)
parser.add_argument(
    "--run-root",
    type=Path,
    default=Path(
        os.environ.get(
            "V2D_ISSUE6_RUN_ROOT", "/workspace/issue6-full-runs/calibrated-temporal"
        )
    ),
)
parser.add_argument(
    "--review-root",
    type=Path,
    default=Path(
        os.environ.get("V2D_ISSUE6_REVIEW_ROOT", "/workspace/pr32-evidence/review")
    ),
)
parser.add_argument("--manifest", type=Path)
parser.add_argument(
    "--hands-mode",
    choices=("metric", "continuous"),
    default="metric",
    help="Continuous shows explicitly marked visual estimates from sequence alignment.",
)
parser.add_argument(
    "--modules-dir", type=Path, default=Path(__file__).resolve().parents[2] / "modules"
)
parser.add_argument(
    "--mano-assets-root",
    type=Path,
    default=Path(os.environ.get("V2D_ISSUE6_MANO_ROOT", "/workspace/mano")),
)
args = parser.parse_args()
label = f"e{args.episode:03d}"
root = args.run_root / label
out = args.review_root / label
out.mkdir(parents=True, exist_ok=True)
config = json.loads((root / "align/job.json").read_text())
camera = config["camera"]
frames = config["frames"]
manifest = load(
    args.manifest
    or (
        Path(os.environ.get("V2D_ISSUE6_INPUT_ROOT", "/workspace/issue6-full-inputs"))
        / label
        / "manifest.temporal.json"
    )
)
source_frames = {f["id"]: f for f in manifest["frames"]}
meshes = {o["id"]: load_mesh(o["mesh"]) for o in manifest["objects"]}
for frame in frames:
    frame["projected_objects"] = []
    for obj in manifest["objects"]:
        pose = camera_from_object(source_frames[frame["id"]], obj)
        if pose is not None:
            frame["projected_objects"].append(
                {"id": obj["id"], "camera_from_object": pose}
            )
times = np.array([f["timestamp_s"] for f in frames])
assert np.allclose(np.diff(times), np.median(np.diff(times)), rtol=0.01, atol=1e-6)
fps = float(1 / np.median(np.diff(times)))
spec = importlib.util.spec_from_file_location(
    "combined_upstream",
    args.modules_dir / "v2d_hamer/lib/render_hands_aligned_video.py",
)
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)
torch.set_num_threads(4)
mano = upstream._mano_layer(str(args.mano_assets_root))
records = upstream._frame_records(str(root / "align/records"))
renderer = pyrender.OffscreenRenderer(camera["width"], camera["height"])
cam = pyrender.IntrinsicsCamera(camera["fx"], camera["fy"], camera["cx"], camera["cy"])
cv_to_gl = np.array([1, -1, -1])
object_colors = [(45, 220, 100), (240, 190, 50), (200, 100, 220)]
object_ids = sorted({o["id"] for f in frames for o in f["projected_objects"]})
colors = {
    key: object_colors[i % len(object_colors)] for i, key in enumerate(object_ids)
}
cache = {}
video = out / "cad-temporal-hands.mp4"
process = subprocess.Popen(
    [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{camera['width']}x{camera['height']}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(video),
    ],
    stdin=subprocess.PIPE,
)
selected = np.unique(np.linspace(0, len(frames) - 1, 12).round().astype(int)).tolist()
montage = Image.new("RGB", (1920, 1712), (12, 17, 24))
counts = []
try:
    for index, frame in enumerate(frames):
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.35, 0.35, 0.35])
        scene.add(cam, pose=np.eye(4))
        upstream._add_lights(scene, np.eye(4))
        hand_rows = records.get(frame["id"], [])
        for record in hand_rows:
            if (
                args.hands_mode == "metric"
                and not record["diagnostics"]["metric_valid"]
            ):
                continue
            if args.hands_mode == "continuous" and not record.get(
                "visual_valid", False
            ):
                continue
            with torch.no_grad():
                vertices, faces = upstream._build_hand_mesh(record, mano)
            mesh = trimesh.Trimesh(vertices * cv_to_gl, faces, process=False)
            color = (235, 70, 70) if record["is_right"] else (60, 150, 235)
            scene.add(
                pyrender.Mesh.from_trimesh(
                    mesh, material=upstream._material(color), smooth=False
                )
            )
        for obj in frame["projected_objects"]:
            mesh_source = meshes[obj["id"]]
            vertices, faces = mesh_source.vertices, mesh_source.faces
            vertices = transform(vertices, obj["camera_from_object"])
            assert np.isfinite(vertices).all()
            mesh = trimesh.Trimesh(vertices * cv_to_gl, faces, process=False)
            scene.add(
                pyrender.Mesh.from_trimesh(
                    mesh, material=upstream._material(colors[obj["id"]]), smooth=False
                )
            )
        rgb, depth = renderer.render(scene)
        source = np.array(
            Image.open(source_frames[frame["id"]]["color"]).convert("RGB")
        )
        mask = depth > 0
        source[mask] = (0.45 * source[mask] + 0.55 * rgb[mask]).astype(np.uint8)
        image = Image.fromarray(source)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, camera["width"], 64), fill=(12, 17, 24))
        draw.text(
            (12, 8),
            f"{label} | frame {frame['id']} | {frame['timestamp_s']:.3f}s | temporal hands + observed CAD",
            font=upstream._font(17),
            fill="white",
        )
        draw.text(
            (12, 36),
            "Hands: left blue / right red  |  CAD: "
            + " / ".join(str(o) for o in object_ids),
            font=upstream._font(15),
            fill="white",
        )
        estimated = [
            "right" if r["is_right"] else "left"
            for r in hand_rows
            if r.get("visual_valid") and not r["diagnostics"]["metric_valid"]
        ]
        if args.hands_mode == "continuous" and estimated:
            draw.text(
                (12, camera["height"] - 28),
                "Estimated hand pose: "
                + ", ".join(estimated)
                + " | metric contact remains unknown",
                font=upstream._font(15),
                fill=(255, 220, 80),
                stroke_width=2,
                stroke_fill=(10, 10, 10),
            )
        elif args.hands_mode == "metric" and any(
            not r["diagnostics"]["metric_valid"] for r in hand_rows
        ):
            draw.text(
                (12, camera["height"] - 28),
                "Unsupported hand meshes omitted: associated contacts remain unknown",
                font=upstream._font(15),
                fill=(255, 220, 80),
                stroke_width=2,
                stroke_fill=(10, 10, 10),
            )
        process.stdin.write(np.asarray(image).tobytes())
        counts.append(
            {
                "frame_id": frame["id"],
                "timestamp_s": frame["timestamp_s"],
                "hands": len(hand_rows),
                "hands_rendered": sum(
                    bool(r.get("visual_valid", False))
                    if args.hands_mode == "continuous"
                    else bool(r["diagnostics"]["metric_valid"])
                    for r in hand_rows
                ),
                "estimated_hands": estimated if args.hands_mode == "continuous" else [],
                "observed_objects": [o["id"] for o in frame["projected_objects"]],
            }
        )
        if index in selected:
            position = selected.index(index)
            x, y = position % 3 * 640, position // 3 * 428
            montage.paste(image.resize((640, 400)), (x, y))
            ImageDraw.Draw(montage).text(
                (x + 8, y + 405),
                f"{label} frame {frame['id']} / {frame['timestamp_s']:.3f}s",
                fill="white",
            )
finally:
    process.stdin.close()
    renderer.delete()
assert process.wait() == 0
montage.save(out / "cad-temporal-hands-montage.jpg", quality=94)
receipt = {
    "episode": args.episode,
    "hands_mode": args.hands_mode,
    "frames": counts,
    "fps": fps,
    "video": str(video),
    "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
    "camera": camera,
    "configuration_sha256": digest(json.loads((root / "inputs.lock.json").read_text())),
    "renderer_source_sha256": file_hash(Path(__file__)),
    "upstream_renderer_sha256": file_hash(
        args.modules_dir / "v2d_hamer/lib/render_hands_aligned_video.py"
    ),
    "mano_model_sha256": file_hash(args.mano_assets_root / "models/MANO_RIGHT.pkl"),
    "basis": (
        "Continuous fitted visual hands with separate per-frame metric support and observed CAD; "
        "joint source-camera depth rendering. Estimates are marked; contacts are not recomputed or accepted. "
        "No ground-truth poses or filled object poses."
        if args.hands_mode == "continuous"
        else "Same observed CAD transforms/meshes as contact calculation and same final aligned temporal MANO records; "
        "joint depth-buffer rendering in source camera. Missing poses are omitted. No ground-truth poses or filled object poses."
    ),
}
(out / "cad-temporal-hands-receipt.json").write_text(
    json.dumps(receipt, indent=2) + "\n"
)
print(
    json.dumps(
        {
            "episode": args.episode,
            "frames": len(counts),
            "video": str(video),
            "sha256": receipt["sha256"],
        }
    )
)
