"""Repair sequence alignment from preserved fits and stereo, in a new output.

No detector observations, metric thresholds, original fits, or contacts are
overwritten. Visual continuity and metric support are distinct output fields.
"""

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import pyrender
import trimesh
from issue6.continuity import (
    bridge_visibility,
    depth_occlusion,
    sequence_corrections,
    visibility_summary,
)
from issue6.geometry import (
    alignment_offset,
    final_alignment_support,
    hand_translation,
    transform,
)
from issue6.worker import upstream
from PIL import Image


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def repair(
    input_root,
    episode,
    modules,
    output,
    image_fit=False,
    prior_records=None,
    smooth_seconds=0.15,
    prior_weight=0.05,
    image_flow=False,
):
    label = f"e{episode:03d}"
    cache = input_root / "issue6-quality-cache/episodes" / label
    original = cache / "issue6-full-runs/all20-final-v3" / label
    temporal = cache / "temporal-hands" / label
    if output.exists():
        raise FileExistsError(
            f"Preserve the previous run; choose a new output: {output}"
        )
    output.mkdir(parents=True)
    config = read(original / "align/job.json")
    frames = config["frames"]
    frame_index = {f["id"]: i for i, f in enumerate(frames)}
    times = np.array([f["timestamp_s"] for f in frames])
    c = config["camera"]
    source_manifest = read(original / "inputs.lock.json")["manifest"]

    def relocate(x):
        if isinstance(x, dict):
            return {k: relocate(v) for k, v in x.items()}
        if isinstance(x, list):
            return [relocate(v) for v in x]
        if isinstance(x, str):
            for name in (
                "issue6-full-inputs",
                "issue6-final-inputs",
                "cad-final",
                "cad-validation",
                "cad-baseline-e012",
                "temporal-hands",
            ):
                prefix = f"/workspace/{name}/{label}"
                if x == prefix or x.startswith(prefix + "/"):
                    return str(cache / name / label) + x[len(prefix) :]
            if x.startswith("/workspace/"):
                return str(input_root / x[len("/workspace/") :])
        return x

    manifest = relocate(source_manifest)
    write(output / "manifest.json", manifest)
    write(output / "align/job.json", config)
    (output / "inputs.lock.json").write_bytes(
        (original / "inputs.lock.json").read_bytes()
    )
    for f in manifest["frames"]:
        p = output / "align/frames" / f"{f['id']:06d}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.symlink_to(f["color"])
    render = upstream(
        "continuity_render", modules / "v2d_hamer/lib/render_hands_aligned_video.py"
    )
    align = upstream("continuity_align", modules / "v2d_hamer/lib/align_hands.py")
    mano_root = cache / "local-assets/mano"
    mano = render._mano_layer(str(mano_root))
    renderer = pyrender.OffscreenRenderer(c["width"], c["height"])
    cam = pyrender.IntrinsicsCamera(c["fx"], c["fy"], c["cx"], c["cy"])
    old = {}
    scales = {"left": [], "right": []}
    for p in sorted((original / "align/records").glob("*/*.json")):
        r = read(p)
        side = "right" if r["is_right"] else "left"
        old[(side, r["frame_idx"])] = r
        scales[side].append(r["hand_scale"])
    scales = {s: float(np.median(v)) for s, v in scales.items()}
    gate = read(temporal / "observation_gate.json")["direct_observation"]
    recorded_checkpoint = Path(read(temporal / "fit_receipt.json")["final_checkpoint"])
    fits = [temporal / "fit/smooth_fit" / recorded_checkpoint.name]
    if not fits[0].is_file():
        raise ValueError("Recorded final temporal fit is unavailable")
    with np.load(fits[0], allow_pickle=False) as data:
        rotations = np.array(data["cam_R"], float)
        camera_translations = np.array(data["cam_t"], float)
        handedness = np.array(data["is_right"])
    raw = {"left": {}, "right": {}}
    for p in sorted((prior_records or (temporal / "converted")).rglob("*.json")):
        r = read(p)
        if "mano" not in r or "frame_idx" not in r:
            continue
        side = "right" if r["is_right"] else "left"
        if r["frame_idx"] in raw[side]:
            raise ValueError("Ambiguous same-side native hand")
        raw[side][r["frame_idx"]] = r
    objects_cache = {}
    measurements = {s: np.zeros((len(frames), 3)) for s in raw}
    weights = {s: np.zeros(len(frames)) for s in raw}
    records = {s: {} for s in raw}
    failures = Counter()
    prior_count = Counter()

    def measured_inputs(side, f):
        fid = f["id"]
        depth = np.load(original / "depth" / f"{fid:06d}.npy", allow_pickle=False)
        p = original / "hands/masks" / side / f"{fid:06d}.png"
        mask = (
            np.asarray(Image.open(p)) > 0 if p.exists() else np.zeros_like(depth, bool)
        )
        explicit = np.load(
            original / "align" / f"occlusion_{fid:06d}.npy", allow_pickle=False
        )
        cad_depth = np.zeros_like(depth)
        for obj in f["projected_objects"]:
            key = obj["mesh"]
            if key not in objects_cache:
                with np.load(original / "align" / key, allow_pickle=False) as z:
                    objects_cache[key] = (z["vertices"].copy(), z["faces"].copy())
            v, faces = objects_cache[key]
            mesh = trimesh.Trimesh(
                transform(v, obj["camera_from_object"]), faces, process=False
            )
            z = align._render_depth(renderer, cam, mesh, np.zeros(3))
            use = (z > 0) & ((cad_depth == 0) | (z < cad_depth))
            cad_depth[use] = z[use]
        occ = depth_occlusion(explicit, cad_depth, depth)
        if not f["occlusion_known"]:
            occ[:] = True
        return depth, mask, occ

    try:
        # Observe depth shifts without altering the fitted prior or visibility.
        for side, side_records in raw.items():
            for f in frames:
                fid = f["id"]
                r = side_records.get(fid)
                if r is None:
                    failures[(side, "missing_native_pose")] += 1
                    continue
                r = copy.deepcopy(r)
                try:
                    translation = (
                        np.array(r["cam_t"], float)
                        if prior_records
                        else hand_translation(r["camera"], c)
                    )
                except ValueError:
                    failures[(side, "invalid_native_camera")] += 1
                    continue
                r.update(
                    cam_t=translation.tolist(), hand_scale=scales[side], intrinsics=c
                )
                r["diagnostics"] = {
                    "metric_valid": False,
                    "cam_t_pre_dz": translation.tolist(),
                }
                vertices, faces = render._build_hand_mesh(r, mano)
                mesh = trimesh.Trimesh(vertices - translation, faces, process=False)
                rendered = align._render_depth(renderer, cam, mesh, translation)
                depth, mask, occ = measured_inputs(side, f)
                offset, diagnostic = alignment_offset(
                    rendered,
                    depth,
                    mask,
                    occ,
                    vertices.mean(0),
                    config["minimum_pixels"],
                )
                # The confidence controls depth fitting, never hand existence.
                if diagnostic["metric_valid"]:
                    shifted = align._render_depth(
                        renderer, cam, mesh, translation + offset
                    )
                    diagnostic.update(
                        final_alignment_support(
                            shifted,
                            depth,
                            mask,
                            occ,
                            config["minimum_pixels"],
                            config["maximum_residual_m"],
                        )
                    )
                i = frame_index[fid]
                if diagnostic["metric_valid"]:
                    measurements[side][i] = offset
                    weights[side][i] = min(
                        1.0, diagnostic["n_pixels"] / (4 * config["minimum_pixels"])
                    )
                r["visual_valid"] = bool(
                    (rendered > 0).sum() >= 256
                    and (gate[side][fid] or mask.sum() >= 64)
                )
                r["directly_observed"] = bool(gate[side][fid])
                r["depth_measurement"] = diagnostic
                records[side][fid] = r
                prior_count[side] += 1
                if i % 50 == 0:
                    print(label, side, "depth measurement", i, flush=True)
        summaries = {}
        for side in raw:
            (h,) = [
                i
                for i in range(len(handedness))
                if bool(handedness[i].mean() > 0.5) == (side == "right")
            ]
            R = rotations[h]
            if len(R) != len(frames):
                raise ValueError("Fitted camera axis differs from source")
            # Project tiny float32 rotation errors onto SO(3), like native conversion.
            u, _, vh = np.linalg.svd(R)
            R = u @ vh
            correction = sequence_corrections(
                times,
                measurements[side],
                weights[side],
                R,
                smooth_seconds=smooth_seconds,
                prior_weight=prior_weight,
            )
            for fid, record in records[side].items():
                record["cam_t"] = (
                    np.array(record["cam_t"]) + correction[frame_index[fid]]
                ).tolist()
            image_receipt = None
            if image_fit:
                import torch
                from issue6.image_fit import refine_sequence

                if any(f["id"] not in records[side] for f in frames):
                    raise ValueError(
                        "Image refinement requires the complete finite native fit"
                    )
                kp = np.zeros((len(frames), 21, 3))
                targets = np.zeros((len(frames), 21))
                confidence = np.zeros_like(targets)
                observation_rows = []
                mask_paths = []
                image_paths = []
                for f in frames:
                    p = (
                        temporal
                        / "tracks"
                        / f"{h:03d}"
                        / f"{f['id']:06d}_keypoints.json"
                    )
                    observation_rows.append(
                        read(p)["people"][0]["pose_keypoints_2d"]
                        if p.exists() and gate[side][f["id"]]
                        else None
                    )
                    mask = original / "hands/masks" / side / f"{f['id']:06d}.png"
                    mask_paths.append(mask if mask.exists() else None)
                    image_paths.append(output / "align/frames" / f"{f['id']:06d}.png")
                if image_flow:
                    from issue6.image_tracks import propagate_landmarks

                    kp = propagate_landmarks(
                        image_paths, mask_paths, observation_rows, times
                    )
                else:
                    for i, observed in enumerate(observation_rows):
                        if observed is not None:
                            kp[i] = observed
                reference_depth = None
                for f in frames:
                    i = frame_index[f["id"]]
                    records[side][f["id"]]["flow_supported"] = bool(
                        not gate[side][f["id"]] and np.count_nonzero(kp[i, :, 2]) >= 4
                    )
                    write(
                        output / "image-observations" / side / f"{f['id']:06d}.json",
                        {
                            "keypoints": kp[i].tolist(),
                            "directly_observed": bool(gate[side][f["id"]]),
                            "flow_supported": records[side][f["id"]]["flow_supported"],
                        },
                    )
                    depth, mask, occ = measured_inputs(side, f)
                    if records[side][f["id"]]["flow_supported"]:
                        # Only sample at checked image tracks, and reject distant
                        # background using the last measured hand surface depth.
                        mask = (
                            (np.abs(depth - reference_depth) < 0.1)
                            if reference_depth is not None
                            else np.zeros_like(mask)
                        )
                    valid_depth = mask & ~occ & np.isfinite(depth) & (depth > 0)
                    for k, (x, y, score) in enumerate(kp[i]):
                        if score <= 0:
                            continue
                        x, y = round(x), round(y)
                        if not (0 <= x < c["width"] and 0 <= y < c["height"]):
                            continue
                        ys, xs = (
                            slice(max(0, y - 4), y + 5),
                            slice(max(0, x - 4), x + 5),
                        )
                        values = depth[ys, xs][valid_depth[ys, xs]]
                        if len(values) >= 5:
                            # Visible skin is in front of a joint centre. This
                            # small anatomical prior is not contact ground truth.
                            targets[i, k] = float(np.median(values)) + 0.005
                            confidence[i, k] = score
                    if np.count_nonzero(confidence[i]) >= 4:
                        reference_depth = (
                            float(np.median(targets[i, confidence[i] > 0])) - 0.005
                        )
                    records[side][f["id"]]["flow_depth_supported"] = bool(
                        records[side][f["id"]]["flow_supported"]
                        and np.count_nonzero(confidence[i]) >= 4
                    )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                fitted_mano = render._mano_layer(str(mano_root)).to(device)
                ordered = [records[side][f["id"]] for f in frames]
                poses, translations, image_receipt = refine_sequence(
                    fitted_mano,
                    ordered,
                    kp,
                    targets,
                    confidence,
                    R,
                    camera_translations[h],
                    c,
                    side == "right",
                )
                image_receipt["image_flow"] = image_flow
                image_receipt["flow_supported_frames"] = [
                    f["id"]
                    for f in frames
                    if records[side][f["id"]].get("flow_supported", False)
                ]
                del fitted_mano
                for i, record in enumerate(ordered):
                    record["mano"]["global_orient"] = poses[i, :3].tolist()
                    record["mano"]["hand_pose"] = poses[i, 3:].tolist()
                    record["cam_t"] = translations[i].tolist()
            metric_frames, visual_frames = [], []
            observed_visibility = np.zeros(len(frames), bool)
            projected_visibility = np.zeros(len(frames), bool)
            for f in frames:
                fid = f["id"]
                r = records[side].get(fid)
                if r is None:
                    continue
                i = frame_index[fid]
                vertices, faces = render._build_hand_mesh(r, mano)
                mesh = trimesh.Trimesh(vertices - r["cam_t"], faces, process=False)
                rendered = align._render_depth(
                    renderer, cam, mesh, np.array(r["cam_t"])
                )
                depth, mask, occ = measured_inputs(side, f)
                diagnostic = final_alignment_support(
                    rendered,
                    depth,
                    mask,
                    occ,
                    config["minimum_pixels"],
                    config["maximum_residual_m"],
                )
                # No inferred frame is promoted to a directly measured contact.
                if not r["directly_observed"]:
                    diagnostic.update(
                        metric_valid=False, reason="not_directly_observed"
                    )
                r["diagnostics"].update(diagnostic)
                projected_visibility[i] = bool((rendered > 0).sum() >= 256)
                observed_visibility[i] = r["visual_valid"] or r.get(
                    "flow_depth_supported", False
                )
                r["continuity"] = {
                    "method": "robust_sequence_depth_in_fitted_world",
                    "correction_camera_m": correction[i].tolist(),
                    "estimated": not diagnostic["metric_valid"],
                    "original_metric_valid": old.get((side, fid), {})
                    .get("diagnostics", {})
                    .get("metric_valid", False),
                }
                if diagnostic["metric_valid"]:
                    metric_frames.append(fid)
            visible = bridge_visibility(
                times, observed_visibility, projected_visibility
            )
            for fid, r in records[side].items():
                i = frame_index[fid]
                r["visual_valid"] = bool(visible[i])
                r["continuity"]["temporally_inferred_visibility"] = bool(
                    visible[i] and not observed_visibility[i]
                )
                write(output / "align/records" / side / f"{fid:06d}.json", r)
                if r["visual_valid"]:
                    visual_frames.append(fid)
                    target = output / "visual-records" / side / f"{fid:06d}.json"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(
                        output / "align/records" / side / f"{fid:06d}.json"
                    )
            baseline = [
                fid
                for (s, fid), r in old.items()
                if s == side and r["diagnostics"]["metric_valid"]
            ]
            summaries[side] = {
                "baseline_visual": visibility_summary(baseline),
                "repaired_visual": visibility_summary(visual_frames),
                "revalidated_metric_frames": len(metric_frames),
                "visual_frame_ids": visual_frames,
                "metric_frame_ids": metric_frames,
                "depth_measurements": int(np.count_nonzero(weights[side])),
                "image_refinement": image_receipt,
                "maximum_correction_m": float(np.linalg.norm(correction, axis=1).max()),
            }
    finally:
        renderer.delete()
    receipt = {
        "episode": episode,
        "source_frames": len(frames),
        "hands": summaries,
        "unusable_prior": {str(k): v for k, v in failures.items()},
        "source_fit_sha256": hashlib.sha256(fits[0].read_bytes()).hexdigest(),
        "source_input_lock_sha256": hashlib.sha256(
            (original / "inputs.lock.json").read_bytes()
        ).hexdigest(),
        "original_contact_outputs_unchanged": True,
        "configuration": {
            "smooth_seconds": smooth_seconds,
            "prior_weight": prior_weight,
            "huber_m": 0.015,
            "image_fit": image_fit,
            "maximum_visual_gap_seconds": 1.5,
            "image_flow": image_flow,
            "prior_records": str(prior_records) if prior_records else None,
            "cad_foreground_margin_m": 0.005,
            "minimum_pixels": config["minimum_pixels"],
            "maximum_residual_m": config["maximum_residual_m"],
        },
        "scope": "Continuous visual estimates from frozen native fits; new stereo checks; no contact acceptance inferred from visual continuity.",
    }
    write(output / "continuity-receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--modules", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--image-fit", action="store_true")
    p.add_argument("--image-flow", action="store_true")
    p.add_argument("--prior-records", type=Path)
    p.add_argument("--smooth-seconds", type=float, default=0.15)
    p.add_argument("--prior-weight", type=float, default=0.05)
    a = p.parse_args()
    repair(
        a.input_root,
        a.episode,
        a.modules,
        a.output,
        a.image_fit,
        a.prior_records,
        a.smooth_seconds,
        a.prior_weight,
        a.image_flow,
    )
