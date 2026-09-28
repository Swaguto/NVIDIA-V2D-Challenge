"""Runs inside the existing v2d_hamer image, only when explicitly dispatched."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

from .backend import prepare_render_inputs
from .geometry import alignment_offset, hand_geometry, transform
from .manifest import read_json, write_json


def upstream(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(job):
    import pyrender
    import torch
    import trimesh
    from PIL import Image

    align = upstream("issue6_upstream_align", "/modules/v2d_hamer/lib/align_hands.py")
    render = upstream("issue6_upstream_render", "/modules/v2d_hamer/lib/render_hands_aligned_video.py")
    config = read_json(job / "align" / "job.json")
    c = config["camera"]
    by_id = {f["id"]: f for f in config["frames"]}
    mano = align._make_mano("/mano")
    renderer = pyrender.OffscreenRenderer(c["width"], c["height"])
    cam = pyrender.IntrinsicsCamera(c["fx"], c["fy"], c["cx"], c["cy"])
    out = job / "align"
    records = []
    samples = {}
    occlusions = {}

    def materialize(r):
        aa = np.array(r["mano"]["global_orient"] + r["mano"]["hand_pose"], np.float32)
        betas = np.array(r["mano"]["betas"], np.float32)
        if aa.shape != (48,) or betas.shape != (10,) or not np.isfinite(aa).all() or not np.isfinite(betas).all():
            raise ValueError("Invalid MANO parameters")
        result = mano(torch.from_numpy(aa)[None], torch.from_numpy(betas)[None])
        return result.verts[0].detach().numpy(), result.joints[0].detach().numpy()

    def masks(r, f):
        depth = np.load(job / "depth" / f"{r['frame_idx']:06d}.npy", allow_pickle=False)
        mp = job / "hands" / "masks" / ("right" if r["is_right"] else "left") / f"{r['frame_idx']:06d}.png"
        hm = np.asarray(Image.open(mp)) > 0 if mp.exists() else np.zeros_like(depth, bool)
        if r["frame_idx"] not in occlusions:
            occ = np.load(out / f"occlusion_{r['frame_idx']:06d}.npy", allow_pickle=False)
            for obj in f["projected_objects"]:
                with np.load(out / obj["mesh"], allow_pickle=False) as data:
                    mesh = trimesh.Trimesh(
                        transform(data["vertices"], obj["camera_from_object"]),
                        data["faces"],
                        process=False,
                    )
                occ |= align._render_depth(renderer, cam, mesh, np.zeros(3)) > 0
            if not f["occlusion_known"]:
                occ[:] = True
            occlusions[r["frame_idx"]] = occ
        occ = occlusions[r["frame_idx"]]
        if hm.shape != depth.shape:
            raise ValueError("Hand mask is not in the color-camera grid")
        return depth, hm, occ

    try:
        # Pass 1 matches upstream's per-track global-scale heuristic, but only
        # finite, image-supported, unoccluded stereo samples can contribute.
        for path in sorted((job / "hands" / "records").glob("*/*.json")):
            r = read_json(path)
            if r["frame_idx"] not in by_id:
                raise ValueError("Unexpected hand frame")
            if r["image_size"] != [c["width"], c["height"]]:
                raise ValueError("Hand image_size disagrees with calibration")
            vertices, joints = materialize(r)
            p = np.array(r["camera"]["pred_cam_t_full"], float)
            focal = float(r["camera"]["scaled_focal_length"])
            if p.shape != (3,) or not np.isfinite(p).all() or p[2] <= 0 or not np.isfinite(focal) or focal <= 0:
                raise ValueError("Invalid hand virtual-camera parameters")
            uv = focal * p[:2] / p[2] + [c["width"] / 2, c["height"] / 2]
            z = p[2] * c["fx"] / focal
            translation = np.array([(uv[0] - c["cx"]) * z / c["fx"], (uv[1] - c["cy"]) * z / c["fy"], z])
            verts, _ = hand_geometry(vertices, joints, r["is_right"], 1.0, np.zeros(3))
            faces = mano.th_faces.numpy()
            if not r["is_right"]:
                faces = faces[:, [0, 2, 1]]
            mesh = trimesh.Trimesh(verts, faces, process=False)
            rendered = align._render_depth(renderer, cam, mesh, translation)
            measured, hm, occ = masks(r, by_id[r["frame_idx"]])
            keep = hm & ~occ & np.isfinite(measured) & (measured > 0) & (rendered > 0)
            if keep.sum() >= config["minimum_pixels"]:
                ratio = float(np.median(measured[keep] / rendered[keep]))
                if 0.2 < ratio < 5:
                    samples.setdefault(r["track_id"], []).append((ratio, int(keep.sum())))
            records.append((r, vertices, joints, faces, translation))
        scales = {}
        for track, values in samples.items():
            values.sort()
            weight = sum(v[1] for v in values)
            total = 0
            for ratio, n in values:
                total += n
                if total >= weight / 2:
                    scales[track] = ratio
                    break
        exported = []
        for r, vertices, joints, faces, translation in records:
            f = by_id[r["frame_idx"]]
            scale = scales.get(r["track_id"], 1.0)
            verts, js = hand_geometry(vertices, joints, r["is_right"], scale, np.zeros(3))
            mesh = trimesh.Trimesh(verts, faces, process=False)
            rendered = align._render_depth(renderer, cam, mesh, translation)
            measured, hm, occ = masks(r, f)
            offset, diag = alignment_offset(
                rendered,
                measured,
                hm,
                occ,
                verts.mean(axis=0) + translation,
                config["minimum_pixels"],
            )
            final_t = translation + offset
            if diag["metric_valid"]:
                # Rerender to measure residual in the actual final silhouette.
                final_depth = align._render_depth(renderer, cam, mesh, final_t)
                valid = hm & ~occ & np.isfinite(measured) & (measured > 0) & (final_depth > 0)
                diag["n_pixels"] = int(valid.sum())
                diag["metric_valid"] = bool(valid.sum() >= config["minimum_pixels"])
                if diag["metric_valid"]:
                    diag["median_abs_residual_m"] = float(np.median(np.abs(measured[valid] - final_depth[valid])))
                else:
                    diag["reason"] = "insufficient_final_depth"
            aligned = {
                **r,
                "cam_t": final_t.tolist(),
                "hand_scale": scale,
                "intrinsics": c,
                "diagnostics": {**diag, "cam_t_pre_dz": translation.tolist()},
            }
            write_json(
                out / "records" / str(r["track_id"]) / f"{r['frame_idx']:06d}.json",
                aligned,
            )
            exported.append(
                {
                    "frame_id": r["frame_idx"],
                    "timestamp_s": f["timestamp_s"],
                    "mode": config["mode"],
                    "camera_provenance": config["camera_provenance"],
                    "hand_provenance": config["hand_provenance"],
                    "track_id": r["track_id"],
                    "is_right": r["is_right"],
                    "mano": r["mano"],
                    "hand_scale": scale,
                    "metric_valid": diag["metric_valid"],
                    "joints_camera_m": (js + final_t).tolist(),
                    "diagnostics": diag,
                }
            )
        write_json(out / "hands.json", exported)
        np.savez_compressed(
            out / "hands.npz",
            mode=np.array(config["mode"]),
            camera_provenance=np.array(config["camera_provenance"]),
            hand_provenance=np.array(config["hand_provenance"]),
            frame_ids=np.array([r["frame_id"] for r in exported], dtype=np.int64),
            timestamp_s=np.array([r["timestamp_s"] for r in exported]),
            track_ids=np.array([r["track_id"] for r in exported], dtype=np.int64),
            is_right=np.array([r["is_right"] for r in exported], bool),
            metric_valid=np.array([r["metric_valid"] for r in exported], bool),
            joints_camera_m=np.array([r["joints_camera_m"] for r in exported]),
            mano_betas=np.array([r["mano"]["betas"] for r in exported]),
            mano_global_orient=np.array([r["mano"]["global_orient"] for r in exported]),
            mano_hand_pose=np.array([r["mano"]["hand_pose"] for r in exported]),
        )
    finally:
        renderer.delete()
    # Native renderer is reused for mesh overlays. Its side panels are
    # camera-relative views, not a verified world trajectory.
    if len(config["frames"]) >= 2:
        dt = np.diff([f["timestamp_s"] for f in config["frames"]])
        if np.allclose(dt, np.median(dt), rtol=0.01, atol=1e-6):
            image_dir, hand_dir = prepare_render_inputs(
                out / "frames", out / "records", config["frames"], out / "render_inputs"
            )
            render.render_hands_aligned_video(
                str(image_dir),
                str(hand_dir),
                "/mano",
                str(out / "hand_mesh_overlay.mp4"),
                fps=float(1 / np.median(dt)),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    run(parser.parse_args().job)
