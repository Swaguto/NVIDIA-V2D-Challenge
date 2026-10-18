"""Render every recorded source knot with a fixed orthographic review camera.

Uses native collision mesh hulls and measured body poses, with CPU triangle
rasterization. This is an offline physics diagnostic, not an Isaac Sim render.
The painter's algorithm provides approximate visibility, not a depth measurement.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from render_native_readback import faces, reference_view
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.sampling import integer_rate_factor


def render(run, inputs, out, world=0):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    report = json.loads((run / "report.json").read_text())
    with np.load(run / "native-state-readback.npz", allow_pickle=False) as f:
        poses = f["body_poses"]
    with np.load(run / "native-model-readback.npz", allow_pickle=False) as f:
        model = {k: f[k].copy() for k in f.files}
    labels = json.loads((run / "body-labels.json").read_text())
    worlds = report["metrics"]["world_count"]
    if not 0 <= world < worlds or len(labels) % worlds:
        raise ValueError("Unsupported world/body axis")
    stride = len(labels) // worlds
    if len(model["body_mass"]) != stride + 1:
        raise ValueError("Unsupported native body mapping")
    source_fps, control_fps = report["source_fps"], report["control_fps"]
    ratio = integer_rate_factor(source_fps, control_fps)
    frames = np.arange(0, len(poses), ratio)
    if len(frames) != report["source_frame_count"] or (len(poses) - 1) % ratio:
        raise ValueError("Readback does not cover the complete original source axis")
    bounds = reference_view(inputs, report)
    center = bounds.mean(axis=1)
    azimuth, elevation = np.deg2rad([-111, 25])
    outward = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    right = np.cross(-outward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, -outward)
    basis = np.stack([right, up, outward], axis=1)
    corners = np.array(np.meshgrid(*bounds)).T.reshape(-1, 3)
    projected = (corners - center) @ basis
    scale = min(900 / np.ptp(projected[:, 0]), 570 / np.ptp(projected[:, 1]))
    triangles, body_ids, colors = [], [], []
    for gi, bid in enumerate(model["geom_bodyid"]):
        bi = int(bid) - 1
        if bi < 0 or model["geom_type"][gi] != 7:
            continue
        mesh = int(model["geom_dataid"][gi])
        start, count = (
            int(model["mesh_vertadr"][mesh]),
            int(model["mesh_vertnum"][mesh]),
        )
        q = model["geom_quat"][gi]
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            vertices = (
                Rotation.from_quat(q[[1, 2, 3, 0]]).apply(
                    model["mesh_vert"][start : start + count]
                )
                + model["geom_pos"][gi]
            )
        if not np.isfinite(vertices).all():
            raise ValueError("Nonfinite native collision mesh")
        polygons = faces(vertices)
        label = labels[bi]
        color = (
            (225, 170, 75)
            if "dex3_left" in label
            else (100, 155, 235)
            if "dex3_right" in label
            else (210, 220, 215)
            if "lid" in label
            else (225, 175, 90)
            if "blue_cup" in label
            else (170, 210, 230)
            if "beige_cup" in label
            else (155, 145, 125)
            if label == "support"
            else (205, 195, 180)
        )
        triangles.extend(vertices[polygons])
        body_ids.extend([bi + world * stride] * len(polygons))
        colors.extend([color] * len(polygons))
    triangles, body_ids, colors = (
        np.asarray(triangles),
        np.asarray(body_ids),
        np.asarray(colors),
    )
    if not len(triangles) or not np.isfinite(triangles).all():
        raise ValueError("No finite native collision surfaces")
    destination = out / "native-readback.mp4"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        "1000x750",
        "-framerate",
        str(source_fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    cv2.setNumThreads(1)
    outside_frames = []
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for si, frame in enumerate(frames):
            state = poses[frame]
            rotations = Rotation.from_quat(state[:, 3:]).as_matrix()
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                points = (
                    np.einsum("tij,tkj->tki", rotations[body_ids], triangles)
                    + state[body_ids, None, :3]
                )
                camera = (points - center) @ basis
            if not np.isfinite(camera).all():
                raise ValueError("Nonfinite measured geometry in readback")
            pixel = np.stack(
                [500 + scale * camera[:, :, 0], 410 - scale * camera[:, :, 1]], axis=-1
            )
            normals = np.cross(camera[:, 1] - camera[:, 0], camera[:, 2] - camera[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
            light = np.array([0.25, 0.4, 1.0])
            light /= np.linalg.norm(light)
            shades = np.clip(
                colors * (0.5 + 0.5 * np.abs(np.sum(normals * light, axis=1)))[:, None],
                0,
                255,
            ).astype(np.uint8)
            # Clip to the viewport before integer conversion, including escaped bodies.
            visible = (
                (pixel[:, :, 0].max(1) >= 0)
                & (pixel[:, :, 0].min(1) < 1000)
                & (pixel[:, :, 1].max(1) >= 100)
                & (pixel[:, :, 1].min(1) < 700)
            )
            canvas = np.full((750, 1000, 3), (34, 21, 13), np.uint8)
            for ti in np.argsort(camera[:, :, 2].mean(1)):
                if visible[ti]:
                    polygon = np.clip(pixel[ti], -10000, 10000).astype(np.int32)
                    cv2.fillConvexPoly(
                        canvas,
                        polygon,
                        tuple(map(int, shades[ti])),
                        lineType=cv2.LINE_AA,
                    )
            outside = [
                labels[i].split("/")[0]
                for i in range(stride)
                if "/object" in labels[i]
                and np.any(
                    (state[i + world * stride, :3] < bounds[:, 0])
                    | (state[i + world * stride, :3] > bounds[:, 1])
                )
            ]
            if outside:
                outside_frames.append(int(si))
            cv2.rectangle(canvas, (0, 0), (1000, 105), (34, 21, 13), -1)
            cv2.rectangle(canvas, (0, 695), (1000, 750), (34, 21, 13), -1)
            cv2.putText(
                canvas,
                f"Measured physics | source frame {si:03d}/{len(frames) - 1:03d} | {si / source_fps:.2f}s | world {world}",
                (28, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (245, 235, 225),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Left: blue   Right: orange | native collision surfaces | fixed view",
                (28, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (215, 205, 190),
                1,
                cv2.LINE_AA,
            )
            message = (
                "OUTSIDE FIXED VIEW: " + ", ".join(outside)
                if outside
                else "Unassisted diagnostic replay; physical task acceptance is pending."
            )
            cv2.putText(
                canvas,
                message,
                (28, 726),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (130, 195, 255),
                1,
                cv2.LINE_AA,
            )
            encoder.stdin.write(canvas.tobytes())
            if si == min(80, len(frames) - 1):
                cv2.imwrite(str(out / "poster.jpg"), canvas)
            if (si + 1) % 50 == 0:
                print(
                    json.dumps(
                        {
                            "rendered_source_frames": si + 1,
                            "total_source_frames": len(frames),
                        }
                    ),
                    flush=True,
                )
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoder failed")
    receipt = {
        "schema": "v2d.native-fast-readback-render.v1",
        "source_frames": len(frames),
        "source_fps": source_fps,
        "world": world,
        "new_simulation": False,
        "isaac_sim_render": False,
        "source_speed": 1.0,
        "fixed_view_bounds_xyz_m": bounds.tolist(),
        "outside_fixed_view_source_frames": outside_frames,
        "visibility": "Triangle painter order; approximate occlusion, no depth claim",
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_sha256": {
            name: hashlib.sha256((run / name).read_bytes()).hexdigest()
            for name in (
                "native-state-readback.npz",
                "native-model-readback.npz",
                "body-labels.json",
                "report.json",
            )
        },
    }
    (out / "render-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "inputs", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--world", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(render(args.run, args.inputs, args.out, args.world), indent=2))
