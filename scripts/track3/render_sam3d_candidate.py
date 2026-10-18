"""Render the declared metric body pose once, without further axis conversions."""

import argparse, json
from pathlib import Path
import cv2, numpy as np, trimesh
from scipy.spatial.transform import Rotation
from reconstruct_stereo_object import sha


def render(candidate, out):
    import pyrender

    if out.exists():
        raise FileExistsError(out)
    report = json.loads((candidate / "manifest.json").read_text())
    if sha(candidate / "mesh.glb") != report["mesh_sha256"]:
        raise ValueError("Candidate geometry changed")
    for name, digest in report["source_files_sha256"].items():
        if sha(candidate / name) != digest:
            raise ValueError("Candidate source changed")
    mesh = trimesh.load(candidate / "mesh.glb", force="mesh", process=False)
    image = cv2.imread(str(candidate / "input.png"))
    observed = cv2.imread(str(candidate / "mask.png"), 0) > 127
    intr = json.loads((candidate / "input-intrinsics.json").read_text())
    height, width = image.shape[:2]
    pose = np.asarray(report["seed_camera_from_object"])
    camera_pose = np.diag([1, -1, -1, 1])
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.7, 0.7, 0.7])
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.1, 0.7, 0.9, 1.0], metallicFactor=0.0, roughnessFactor=0.7
    )
    node = scene.add(
        pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False), pose=pose
    )
    scene.add(
        pyrender.IntrinsicsCamera(
            intr["fx"], intr["fy"], intr["cx"], intr["cy"], znear=0.01, zfar=5
        ),
        pose=camera_pose,
    )
    scene.add(
        pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.0), pose=camera_pose
    )
    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        color, depth = renderer.render(scene)
    finally:
        renderer.delete()
    visible = depth > 0
    union = (visible | observed).sum()
    overlap = visible & observed
    overlay = image.copy()
    overlay[visible] = (image[visible] * 0.45 + color[visible, ::-1] * 0.55).astype(
        np.uint8
    )
    points = np.load(candidate / "pointmap.npy", allow_pickle=False)
    measured = overlap & np.isfinite(points[:, :, 2])
    errors = abs(depth[measured] - points[:, :, 2][measured])
    out.mkdir(parents=True)
    cv2.imwrite(str(out / "source-overlay.jpg"), np.hstack([image, overlay]))
    result = dict(
        schema="v2d.sam3d-metric-render-review.v1",
        mesh_sha256=report["mesh_sha256"],
        camera_from_body_unchanged=True,
        additional_axis_conversion_applied=False,
        rendered_visible_mask_iou=float((overlap).sum() / max(1, union)),
        measured_depth_overlap_pixels=int(measured.sum()),
        median_depth_residual_m=float(np.median(errors)) if len(errors) else None,
        p90_depth_residual_m=float(np.quantile(errors, 0.9)) if len(errors) else None,
        unobserved_surfaces_inferred=True,
        physical_task_accepted=False,
    )
    # Diagnostic turntable uses a separate camera and does not alter the asset.
    half = float(np.linalg.norm(mesh.extents) / 2)
    center = mesh.bounds.mean(0)
    distance = max(0.1, half * 3.5)
    spin = pyrender.Scene(bg_color=[0.07, 0.08, 0.1, 1], ambient_light=[0.8, 0.8, 0.8])
    n = spin.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    spin.add(
        pyrender.PerspectiveCamera(yfov=np.deg2rad(45), znear=0.001, zfar=10),
        pose=np.eye(4),
    )
    spin.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0), pose=np.eye(4))
    renderer = pyrender.OffscreenRenderer(640, 480)
    frames = []
    try:
        for angle in np.linspace(0, 2 * np.pi, 72, endpoint=False):
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_euler("xy", [0.35, angle]).as_matrix()
            matrix[:3, 3] = [0, 0, -distance] - matrix[:3, :3] @ center
            spin.set_pose(n, matrix)
            rgb, _ = renderer.render(spin)
            frames.append(rgb[:, :, ::-1])
    finally:
        renderer.delete()
    import subprocess

    p = subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            "640x480",
            "-r",
            "24",
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out / "turntable.mp4"),
        ],
        stdin=subprocess.PIPE,
    )
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("Turntable encoder failed")
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(render(a.candidate, a.out), indent=2))
