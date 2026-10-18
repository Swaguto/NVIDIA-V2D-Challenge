"""Run the repository SAM3D backend on one calibrated, masked stereo frame.

The output is an inferred complete-shape candidate. Its pose and unobserved
surfaces remain model estimates, grounded by the supplied measured pointmap.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reconstruction/modules/v2d_stereo_depth/lib"))
from stereo_depth import load_rig, stereo_depth
from reconstruct_stereo_object import sha
from rgb_stereo import seed_frames


def bake_metric_scale(mesh, transform):
    scale = np.asarray(transform["scale"], float)
    quat = np.asarray(transform["rotation"], float)
    translation = np.asarray(transform["translation"], float)
    if (
        scale.shape != (3,)
        or quat.shape != (4,)
        or translation.shape != (3,)
        or not np.isfinite(np.r_[scale, quat, translation]).all()
        or not ((scale > 0) & (scale < 10)).all()
        or not np.isclose(np.linalg.norm(quat), 1, atol=1e-4)
    ):
        raise ValueError("SAM3D returned an invalid metric transform")
    mesh = mesh.copy()
    mesh.vertices = np.asarray(mesh.vertices) * scale
    if not np.isfinite(mesh.vertices).all() or not 0.005 < mesh.extents.max() < 1.0:
        raise ValueError("Candidate size is outside the tabletop reconstruction range")
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    pose[:3, 3] = translation
    return mesh, pose


def generate(data, masks, episode, name, frame, weights, out, seed=20261017):
    if out.exists():
        raise FileExistsError(out)
    if not name.isidentifier():
        raise ValueError("Unsafe object identifier")
    receipt = json.loads((weights / "download-receipt.json").read_text())
    if receipt.get("revision") != "2e73555018d2741ccd486e56c24fac41155a1dc6":
        raise ValueError("Use the pinned SAM3D checkpoint revision")
    rig = load_rig(str(data / "meta/camera_calibration.json"))
    image = seed_frames(data, episode, rig, [frame])[frame]
    captures = [
        cv2.VideoCapture(
            str(
                data
                / "videos/chunk-000"
                / f"observation.images.{cam}"
                / f"episode_{episode:06d}.mp4"
            )
        )
        for cam in (rig.left, rig.right)
    ]
    try:
        reads = []
        for c in captures:
            c.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, pixels = c.read()
            if not ok:
                raise ValueError("Requested stereo source frame is unavailable")
            reads.append(pixels)
        depth, valid = stereo_depth(rig, *reads)
    finally:
        for c in captures:
            c.release()
    mask_path = masks / name / f"{frame:06d}.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != depth.shape:
        raise ValueError("Missing calibrated source mask")
    selected = valid & (mask > 127) & np.isfinite(depth) & (depth > 0.1) & (depth < 2.5)
    if selected.sum() < 200:
        raise ValueError(
            "Too little measured stereo surface for metric shape grounding"
        )
    y, x = np.indices(depth.shape)
    K = rig.P1[:3, :3]
    pointmap = np.stack(
        [(x - K[0, 2]) * depth / K[0, 0], (y - K[1, 2]) * depth / K[1, 1], depth],
        axis=-1,
    ).astype(np.float32)
    pointmap[~selected] = np.nan
    out.mkdir(parents=True)
    cv2.imwrite(str(out / "input.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "mask.png"), mask)
    np.save(out / "pointmap.npy", pointmap)
    intrinsics = dict(
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        width=depth.shape[1],
        height=depth.shape[0],
    )
    (out / "input-intrinsics.json").write_text(json.dumps(intrinsics, indent=2) + "\n")
    from v2d.sam3d.lib.image_to_mesh import image_to_mesh

    image_to_mesh(
        str(out / "input.png"),
        str(out / "mask.png"),
        str(out / "raw.glb"),
        str(out / "raw-transform.json"),
        str(out / "output-intrinsics.json"),
        str(weights),
        seed=seed,
        pointmap_path=str(out / "pointmap.npy"),
        pointmap_intrinsics_path=str(out / "input-intrinsics.json"),
    )
    mesh = trimesh.load(out / "raw.glb", force="mesh", process=False)
    mesh, pose = bake_metric_scale(
        mesh, json.loads((out / "raw-transform.json").read_text())
    )
    mesh.export(out / "mesh.glb")
    report = dict(
        schema="v2d.video-object-reconstruction.v1",
        stage="sam3d_metric_depth",
        episode=episode,
        object=name,
        units="metres",
        body_origin_frozen=True,
        provided_meshes_read=False,
        reference_pose_labels_read=False,
        geometry_final=False,
        submission_ready=False,
        physical_task_accepted=False,
        source="Pinned pretrained SAM3D shape/pose with calibrated source mask and measured stereo pointmap",
        unseen_surfaces_are_inferred=True,
        anchor_pose_is_inferred=True,
        seed_frame=frame,
        seed=seed,
        seed_camera_from_object=pose.tolist(),
        mesh_sha256=sha(out / "mesh.glb"),
        vertices=len(mesh.vertices),
        triangles=len(mesh.faces),
        extents_m=mesh.extents.tolist(),
        model_revision=receipt["revision"],
        model_receipt_sha256=sha(weights / "download-receipt.json"),
        source_code_sha256=sha(Path(__file__)),
        calibration_sha256=sha(data / "meta/camera_calibration.json"),
        mask_sha256=sha(mask_path),
        measured_pointmap_pixels=int(selected.sum()),
        coordinate_frame="rectified_camera",
        camera_frame=rig.left,
        stereo_videos_sha256={
            cam: sha(
                data
                / "videos/chunk-000"
                / f"observation.images.{cam}"
                / f"episode_{episode:06d}.mp4"
            )
            for cam in (rig.left, rig.right, "ego_cam_a")
        },
        source_files_sha256={
            name: sha(out / name)
            for name in [
                "input.png",
                "mask.png",
                "pointmap.npy",
                "input-intrinsics.json",
                "raw.glb",
                "raw-transform.json",
            ]
        },
    )
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["data", "masks", "weights", "out"]:
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--object", required=True)
    p.add_argument("--frame", type=int, required=True)
    a = p.parse_args()
    print(
        json.dumps(
            generate(a.data, a.masks, a.episode, a.object, a.frame, a.weights, a.out),
            indent=2,
        )
    )
