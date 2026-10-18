"""Recover fresh, explicitly estimated hand priors when tracking masks disappear.

Inputs are hash-bound source RGB, image tracks and canonical identity/flags.
SAM2 supplies a new image mask and HaMeR supplies a fresh pose. Neither model
creates a direct observation, metric label, contact label or accepted reference.
Run --validate-only on CPU before a bounded GPU probe in the perception image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

SCHEMA = "v2d.source-hand-recovery-request.v1"
KINDS = {"identity_associated_source_model_prediction", "checked_inferred_image_flow"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_request(path):
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get("schema") != SCHEMA or not data.get("frames"):
        raise ValueError("Need a nonempty source hand recovery request")
    if data.get("source_hand") not in {"left", "right"}:
        raise ValueError("Need one identified source hand for these image arrays")
    camera = data["camera"]
    c = np.array([camera[k] for k in ("fx", "fy", "cx", "cy", "width", "height")], float)
    if not np.isfinite(c).all() or min(c[:2]) <= 0 or min(c[4:]) <= 0:
        raise ValueError("Invalid source calibration")
    evidence = Path(data["evidence_receipt"])
    if not evidence.is_absolute() or digest(evidence) != data["evidence_sha256"]:
        raise ValueError("Changed or unbound image evidence")
    evidence_arrays = Path(data["evidence_arrays"])
    bundle = Path(data["canonical_bundle"])
    for source, expected in [(evidence_arrays, data["evidence_arrays_sha256"]), (bundle, data["canonical_bundle_sha256"])]:
        if not source.is_absolute() or digest(source) != expected:
            raise ValueError("Changed or unbound canonical source arrays")
    with np.load(evidence_arrays, allow_pickle=False) as z:
        image_points = z["landmarks_px"].copy()
        image_frames = z["frame_indices"].copy()
    with np.load(bundle, allow_pickle=False) as z:
        frames, times = z["frame_indices"].copy(), z["timestamps_s"].copy()
        direct = z["hand_directly_observed"].copy()
        intrinsics = z["intrinsics"].copy()
    expected_k = np.array([[c[0], 0, c[2]], [0, c[1], c[3]], [0, 0, 1]])
    if intrinsics.shape != (3, 3) or not np.allclose(intrinsics, expected_k, rtol=0, atol=1e-6):
        raise ValueError("Source camera calibration cannot change")
    if direct.shape != (len(frames), 2) or direct.dtype != bool or len(np.unique(frames)) != len(frames):
        raise ValueError("Invalid canonical observation axis")
    seen, last_time = set(), {}
    for row in data["frames"]:
        key = (row["frame_idx"], row["track_id"])
        if key in seen or type(row["frame_idx"]) is not int or row["frame_idx"] < 0:
            raise ValueError("Duplicate or invalid source frame/track")
        seen.add(key)
        if type(row["is_right"]) is not bool or type(row["original_direct_observation"]) is not bool:
            raise ValueError("Need explicit canonical identity and observation flags")
        if row["is_right"] != (data["source_hand"] == "right"):
            raise ValueError("Image evidence cannot be reassigned to the other hand")
        matches = np.flatnonzero(frames == row["frame_idx"])
        if len(matches) != 1 or row["original_direct_observation"] != bool(direct[matches[0], int(row["is_right"])]):
            raise ValueError("Canonical direct observation flags cannot change")
        if row["evidence_kind"] not in KINDS or not row["identity_anchor_frames"]:
            raise ValueError("Source identity needs an anchor association receipt")
        t = float(row["timestamp_s"])
        if not np.isfinite(t) or t != float(times[matches[0]]) or t <= last_time.get(row["track_id"], -np.inf):
            raise ValueError("Source clock must increase within each hand track")
        last_time[row["track_id"]] = t
        points = np.array(row["keypoints"], float)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            raise ValueError("Need finite 21x3 image points; unknown points use zero confidence")
        if ((points[:, 2] < 0) | (points[:, 2] > 1)).any():
            raise ValueError("Invalid image confidence")
        p = points[points[:, 2] > 0, :2]
        observed = np.flatnonzero(image_frames == row["frame_idx"])
        if len(observed) != 1:
            raise ValueError("Missing source image evidence frame")
        expected = image_points[observed[0]]
        if not np.array_equal(np.isfinite(expected).all(-1), points[:, 2] > 0) or not np.array_equal(expected[points[:, 2] > 0], p):
            raise ValueError("Source image points cannot be fabricated or relabelled")
        if len(p) < 4 or (p < 0).any() or (p[:, 0] >= c[4]).any() or (p[:, 1] >= c[5]).any():
            raise ValueError("Need four in-view supported image points")
        crop = np.array([row["crop"][k] for k in ("cx", "cy", "size")], float)
        if not np.isfinite(crop).all() or crop[2] <= 0 or not (0 <= crop[0] < c[4] and 0 <= crop[1] < c[5]):
            raise ValueError("Invalid image crop")
        if (np.abs(p - crop[:2]) > crop[2] / 2).any():
            raise ValueError("Supported points must lie in the requested crop")
        shape = np.array(row["fixed_betas"], float)
        if shape.shape != (10,) or not np.isfinite(shape).all() or not np.isfinite(row["fixed_hand_scale"]) or row["fixed_hand_scale"] <= 0:
            raise ValueError("Need fixed subject shape and positive hand scale")
        image = Path(row["image"])
        if not image.is_absolute() or digest(image) != row["image_sha256"]:
            raise ValueError("Changed or unbound source image")
        with Image.open(image) as im:
            if im.size != (int(c[4]), int(c[5])):
                raise ValueError("Source image differs from calibrated pixel grid")
    return data


def associated_keypoints(points, mask):
    """Keep source pixels on the new mask; do not synthesize missing fingers."""
    points = np.array(points, float, copy=True)
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or mask.sum() < 64:
        raise ValueError("No usable fresh hand mask")
    for k, (x, y, score) in enumerate(points):
        if score <= 0:
            continue
        x, y = round(x), round(y)
        if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]) or not mask[max(0, y-3):y+4, max(0, x-3):x+4].any():
            points[k, 2] = 0
    if np.count_nonzero(points[:, 2]) < 4:
        raise ValueError("Fresh mask does not support four source image points")
    return points


def make_record(row, prediction, camera, request_sha):
    pose = Rotation.from_matrix(np.asarray(prediction["hand_pose"]).reshape(15, 3, 3)).as_rotvec().ravel()
    orient = Rotation.from_matrix(np.asarray(prediction["global_orient"]).reshape(3, 3)).as_rotvec()
    betas = np.asarray(prediction["betas"]).reshape(-1)
    if betas.shape != (10,) or not np.isfinite(betas).all():
        raise ValueError("Invalid predicted hand shape")
    t = np.array(prediction["pred_cam_t_full"], float)
    focal = float(prediction["scaled_focal_length"])
    if t.shape != (3,) or not np.isfinite(t).all() or t[2] <= 0 or not np.isfinite(focal) or focal <= 0:
        raise ValueError("Invalid fresh model camera")
    image_uv = focal * t[:2] / t[2] + np.array([camera["width"], camera["height"]]) / 2
    z = t[2] * camera["fx"] / focal
    calibrated_t = [(image_uv[0]-camera["cx"])*z/camera["fx"], (image_uv[1]-camera["cy"])*z/camera["fy"], z]
    return dict(track_id=row["track_id"], is_right=row["is_right"], frame_idx=row["frame_idx"], inference_frame_idx=row["frame_idx"], image_size=[camera["width"], camera["height"]], crop=row["crop"],
        mano=dict(betas=row["fixed_betas"], global_orient=orient.tolist(), hand_pose=pose.tolist()),
        camera=dict(pred_cam_t_full=t.tolist(), scaled_focal_length=focal), cam_t=calibrated_t, hand_scale=row["fixed_hand_scale"],
        fresh_image_supported=True, directly_observed=False, original_direct_observation=row["original_direct_observation"], metric_valid=False, estimated=True,
        visual_valid=False, diagnostics=dict(metric_valid=False, reason="fresh_image_prior_requires_independent_source_depth_and_native_validation"),
        source_request_sha256=request_sha, evidence_kind=row["evidence_kind"], raw_predicted_betas=np.asarray(prediction["betas"]).tolist())


def canonical_sources(data):
    """Verify original priors against the bundle's separately bound manifest."""
    bundle = Path(data["canonical_bundle"])
    manifest_path = bundle.parent / "manifest.json"
    expected = data.get("canonical_manifest_sha256")
    if not expected or digest(manifest_path) != expected:
        raise ValueError("Fresh inference needs a hash-bound canonical hand manifest")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "v2d.track3.episode.v1"
        or manifest["payload"] != dict(path=bundle.name, sha256=data["canonical_bundle_sha256"])):
        raise ValueError("Canonical manifest belongs to another source bundle")
    sources = {}
    for entry in manifest["sources"]:
        name = entry["path"]
        if not name.startswith("source/hands/"):
            continue
        parts = Path(name).parts
        if len(parts) != 4 or parts[2] not in {"left", "right"}:
            raise ValueError("Invalid canonical hand path")
        source = bundle.parent / name
        if digest(source) != entry["sha256"]:
            raise ValueError("Canonical hand prior changed")
        record = json.loads(source.read_text())
        fid, side = record["frame_idx"], parts[2]
        if (type(fid) is not int or fid < 0 or parts[3] != f"{fid:06d}.json"
            or type(record["is_right"]) is not bool or record["is_right"] != (side == "right")):
            raise ValueError("Canonical hand identity/frame mismatch")
        key = (side, fid)
        if key in sources:
            raise ValueError("Duplicate canonical hand prior")
        sources[key] = (source, record)
    if not sources:
        raise ValueError("No canonical hand priors are bound to this bundle")
    for row in data["frames"]:
        side = "right" if row["is_right"] else "left"
        prior = sources.get((side, row["frame_idx"]))
        if prior is None:
            raise ValueError("Recovery request lacks its canonical prior")
        old = prior[1]
        if (old["track_id"] != row["track_id"]
            or old.get("directly_observed") is not row["original_direct_observation"]
            or old["mano"]["betas"] != row["fixed_betas"]
            or old["hand_scale"] != row["fixed_hand_scale"]):
            raise ValueError("Recovery request changes canonical identity, observation or subject shape")
    return sources


def infer_frames(data, output, request_sha, predict_masks, predict_hand, receipt):
    """Shared file protocol; CPU tests supply explicitly fake model callbacks."""
    output = Path(output)
    for row in data["frames"]:
        side = "right" if row["is_right"] else "left"
        stem = f"{row['frame_idx']:06d}"
        image = np.array(Image.open(row["image"]).convert("RGB"))
        points = np.array(row["keypoints"], float)
        active = points[:, 2] > 0
        crop = row["crop"]
        box = np.r_[np.array([crop["cx"], crop["cy"]])-crop["size"]/2, np.array([crop["cx"], crop["cy"]])+crop["size"]/2]
        masks, scores = predict_masks(image, points[active, :2], box)
        candidates = []
        for mask, score in zip(masks, scores):
            if np.asarray(mask).shape != image.shape[:2] or not np.isfinite(score):
                continue
            try:
                kp = associated_keypoints(points, mask)
                candidates.append((float(score), mask, kp))
            except ValueError:
                pass
        if not candidates:
            receipt["rows"].append(dict(frame_idx=row["frame_idx"], side=side, status="rejected_mask_support"))
            continue
        score, mask, kp = max(candidates, key=lambda x: x[0])
        prediction = predict_hand(image, crop, row["is_right"])
        record = make_record(row, prediction, data["camera"], request_sha)
        for folder in [output/"records"/side, output/"image-observations"/side]:
            folder.mkdir(parents=True, exist_ok=True)
        maskpath = output/"image-observations"/side/f"{stem}-mask.png"
        Image.fromarray(mask.astype(np.uint8)*255).save(maskpath)
        (output/"records"/side/f"{stem}.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
        estimate = dict(estimated=True, metric_valid=False, directly_observed=False, original_direct_observation=row["original_direct_observation"], keypoints=kp.tolist(), mask=maskpath.name, source=f"fresh_sam2_hamer:{request_sha}")
        (maskpath.parent/f"{stem}.json").write_text(json.dumps(estimate, indent=2, allow_nan=False)+"\n")
        receipt["rows"].append(dict(frame_idx=row["frame_idx"], side=side, status="fresh_prior_requires_validation", mask_score=score, supported_points=int(np.count_nonzero(kp[:, 2]))))


def output_files(output):
    return {str(p.relative_to(output)): digest(p) for p in sorted(Path(output).rglob("*"))
            if p.is_file() and p != Path(output)/"receipt.json"}


def run(path, output, sam_weights, sam_config, hamer_weights):
    data = validate_request(path)
    canonical_sources(data)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve earlier results; choose a new output directory")
    output.mkdir(parents=True)
    request_sha = digest(path)
    receipt = dict(schema="v2d.source-hand-recovery-result.v1", request_sha256=request_sha,
        canonical_manifest_sha256=data["canonical_manifest_sha256"],
        producer_sha256=digest(Path(__file__)), rows=[], files={}, complete=False,
        original_observation_flags_changed=False, reference_accepted=False, training_approved=False)
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from v2d.hamer.lib._hamer import get_model, run_hamer

        if not torch.cuda.is_available():
            raise RuntimeError("Fresh inference requires the bounded CUDA perception runtime")
        receipt.update(sam_weights_sha256=digest(sam_weights), sam_config=sam_config,
            hamer_checkpoint_sha256=digest(Path(hamer_weights)/"_DATA/hamer_ckpts/checkpoints/hamer.ckpt"))
        predictor = SAM2ImagePredictor(build_sam2(sam_config, str(sam_weights), device="cuda", apply_postprocessing=False))
        model, cfg = get_model(str(hamer_weights))

        def predict_masks(image, points, box):
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(point_coords=points, point_labels=np.ones(len(points), int), box=box, multimask_output=True)
            return masks, scores

        def predict_hand(image, crop, is_right):
            return run_hamer(model, cfg, image, crop["cx"], crop["cy"], crop["size"], is_right=is_right)

        infer_frames(data, output, request_sha, predict_masks, predict_hand, receipt)
        validate_request(path)
        canonical_sources(data)
        receipt["complete"] = True
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        receipt["files"] = output_files(output)
        (output/"receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--out", type=Path)
    p.add_argument("--sam-weights", type=Path)
    p.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--hamer-weights", type=Path)
    a = p.parse_args()
    if a.validate_only:
        print(json.dumps(dict(validated_requests=len(validate_request(a.request)["frames"]), gpu_inference=False)))
    else:
        if any(x is None for x in (a.out, a.sam_weights, a.hamer_weights)):
            p.error("Inference requires --out, --sam-weights and --hamer-weights")
        print(json.dumps(run(a.request, a.out, a.sam_weights, a.sam_config, a.hamer_weights)))
