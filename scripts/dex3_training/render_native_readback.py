"""Render recorded native collision geometry and measured body poses offline.

This is a visualization of readback, not a new simulation or Isaac Sim render.
One world is shown. Every original source knot is included at its source rate.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def faces(vertices):
    if len(vertices) == 3:
        return np.array([[0, 1, 2]])
    centered = vertices - vertices.mean(0)
    if np.linalg.matrix_rank(centered, tol=1e-9) == 2:
        _, _, axes = np.linalg.svd(centered)
        boundary = ConvexHull(centered @ axes[:2].T).vertices
        return np.array(
            [
                [boundary[0], boundary[i], boundary[i + 1]]
                for i in range(1, len(boundary) - 1)
            ]
        )
    return ConvexHull(vertices).simplices


def render(run, out, world=0):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    report = json.loads((run / "report.json").read_text())
    with np.load(run / "native-state-readback.npz") as f:
        poses = f["body_poses"]
    with np.load(run / "native-model-readback.npz") as f:
        model = {k: f[k].copy() for k in f.files}
    labels = json.loads((run / "body-labels.json").read_text())
    worlds = report["metrics"]["world_count"]
    if not 0 <= world < worlds:
        raise ValueError("World index unavailable")
    stride = len(labels) // worlds
    if len(model["body_mass"]) != stride + 1:
        raise ValueError("Unsupported native body mapping")
    source_fps = report.get("source_fps", 20.0)
    fps = report.get("control_fps", 20.0)
    ratio = int(fps / source_fps)
    if ratio < 1 or fps / source_fps != ratio or (len(poses) - 1) % ratio:
        raise ValueError("Readback does not contain every original source knot")
    frames = np.arange(0, len(poses), ratio)
    plt.rcParams.update(
        {
            "text.color": "#e8edf5",
            "axes.labelcolor": "#c4d0df",
            "xtick.color": "#a7b7cb",
            "ytick.color": "#a7b7cb",
        }
    )
    fig = plt.figure(figsize=(10, 7.5), dpi=100, facecolor="#0d1522")
    ax = fig.add_axes([0.02, 0.07, 0.96, 0.82], projection="3d", facecolor="#0d1522")
    ax.set(
        xlim=(-0.16, 0.43),
        ylim=(0.04, 0.52),
        zlim=(-0.65, -0.13),
        xlabel="world X (m)",
        ylabel="world Y (m)",
        zlabel="world Z (m)",
    )
    ax.set_box_aspect([0.59, 0.48, 0.52])
    ax.view_init(elev=25, azim=-111)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.08, 0.12, 0.19, 1))
        axis.pane.set_edgecolor("#41526a")
    meshes = []
    for gi in range(len(model["geom_bodyid"])):
        bi = int(model["geom_bodyid"][gi]) - 1
        if bi < 0 or model["geom_type"][gi] != 7:
            continue
        mesh = model["geom_dataid"][gi]
        start, count = model["mesh_vertadr"][mesh], model["mesh_vertnum"][mesh]
        q = model["geom_quat"][gi]
        vertices = (
            Rotation.from_quat(q[[1, 2, 3, 0]]).apply(
                model["mesh_vert"][start : start + count]
            )
            + model["geom_pos"][gi]
        )
        label = labels[bi]
        color = (
            "#60b7df"
            if "dex3_left" in label
            else "#ed9d74"
            if "dex3_right" in label
            else "#d4e2ce"
            if "pot_lid" in label
            else "#bac8d9"
            if "pot" in label
            else "#546b83"
        )
        polygons = faces(vertices)
        collection = Poly3DCollection(
            vertices[polygons],
            facecolors=color,
            edgecolors="#223147",
            linewidths=0.18,
            alpha=0.85 if label != "support" else 0.3,
        )
        ax.add_collection3d(collection)
        meshes.append((bi + world * stride, vertices, polygons, collection))
    title = fig.text(0.035, 0.958, "", fontsize=14, fontweight="bold")
    fig.text(
        0.035,
        0.918,
        "Measured native physics · collision surfaces · offline readback rendering",
        fontsize=10,
    )
    status = fig.text(0.035, 0.027, "", fontsize=10, color="#ffc68f")
    destination = out / "native-readback.mp4"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
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
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for source_index, frame in enumerate(frames):
            for bi, vertices, polygons, collection in meshes:
                pose = poses[frame, bi]
                points = Rotation.from_quat(pose[3:]).apply(vertices) + pose[:3]
                collection.set_verts(points[polygons])
            title.set_text(
                f"Reference {source_index:03d}/{len(frames) - 1:03d} · {frame / fps:05.2f} s · world {world} · {fps:g} Hz control"
            )
            fallen = [
                labels[i].split("/")[0]
                for i in range(stride)
                if "/object" in labels[i]
                and poses[frame, i + world * stride, 2] < -0.72
            ]
            status.set_text(
                "Below the fixed view: " + ", ".join(fallen)
                if fallen
                else "Diagnostic replay; full physical task has not passed acceptance."
            )
            fig.canvas.draw()
            encoder.stdin.write(
                np.asarray(fig.canvas.buffer_rgba())[:, :, :3].tobytes()
            )
            if source_index == min(140, len(frames) - 1):
                fig.savefig(out / "poster.png", facecolor=fig.get_facecolor())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoding failed")
    plt.close(fig)
    receipt = dict(
        schema="v2d.native-readback-render.v1",
        world=world,
        source_frames=len(frames),
        source_fps=source_fps,
        control_fps=fps,
        data_sha256={
            name: hashlib.sha256((run / name).read_bytes()).hexdigest()
            for name in (
                "native-state-readback.npz",
                "native-model-readback.npz",
                "body-labels.json",
                "report.json",
            )
        },
        rendering_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_simulation=False,
        isaac_sim_render=False,
        source_speed=1.0,
    )
    (out / "render-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--world", type=int, default=0)
    a = p.parse_args()
    print(json.dumps(render(a.run, a.out, a.world), indent=2))
