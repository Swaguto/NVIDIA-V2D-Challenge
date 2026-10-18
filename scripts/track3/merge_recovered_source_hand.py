"""Stage fresh source hand priors into a new, unqualified reconstruction input.

This merges only explicitly requested missing observations. Original priors,
bundle arrays and confidence flags remain immutable. It performs no fitting,
retargeting, contact acceptance, policy training or physics simulation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

from recover_source_hand import canonical_sources, digest, validate_request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"reconstruction/modules/v2d_issue6"))
from issue6.image_observations import load_image_estimate


def validate_recovery(request_path, recovery):
    data = validate_request(request_path)
    sources = canonical_sources(data)
    recovery = Path(recovery).resolve()
    receipt_path = recovery/"receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != "v2d.source-hand-recovery-result.v1"
        or receipt.get("complete") is not True
        or receipt.get("request_sha256") != digest(request_path)
        or receipt.get("canonical_manifest_sha256") != data["canonical_manifest_sha256"]
        or receipt.get("original_observation_flags_changed") is not False
        or receipt.get("reference_accepted") is not False
        or receipt.get("training_approved") is not False):
        raise ValueError("Need a complete, matching, unqualified recovery receipt")
    files = receipt["files"]
    actual = {str(p.relative_to(recovery)) for p in recovery.rglob("*")
              if p.is_file() and p != receipt_path}
    if actual != set(files):
        raise ValueError("Unbound or missing fresh recovery outputs")
    for name, expected in files.items():
        p = Path(name)
        if p.is_absolute() or ".." in p.parts or (recovery/p).is_symlink():
            raise ValueError("Invalid fresh recovery output path")
        if digest(recovery/p) != expected:
            raise ValueError("Fresh recovery output changed")
    requested = {("right" if r["is_right"] else "left", r["frame_idx"]):r for r in data["frames"]}
    results = {}
    for row in receipt["rows"]:
        key = (row["side"], row["frame_idx"])
        if key in results or key not in requested:
            raise ValueError("Duplicate or unrequested recovery result")
        results[key] = row
    if set(results) != set(requested):
        raise ValueError("Recovery did not account for every requested source frame")
    accepted, expected_files = [], set()
    with np.load(data["canonical_bundle"], allow_pickle=False) as z:
        axes = {k:z[k].copy() for k in ["frame_indices", "timestamps_s", "hand_valid", "hand_directly_observed", "hand_metric_valid"]}
    for key, result in results.items():
        side, fid = key
        row = requested[key]
        if result["status"] == "rejected_mask_support":
            continue
        if result["status"] != "fresh_prior_requires_validation":
            raise ValueError("Unknown fresh recovery status")
        index = np.flatnonzero(axes["frame_indices"] == fid)
        si = int(row["is_right"])
        if len(index) != 1 or axes["hand_directly_observed"][index[0],si] or axes["hand_metric_valid"][index[0],si]:
            raise ValueError("Fresh recovery cannot replace supported canonical observations")
        record_name = f"records/{side}/{fid:06d}.json"
        estimate_name = f"image-observations/{side}/{fid:06d}.json"
        mask_name = f"image-observations/{side}/{fid:06d}-mask.png"
        expected_files.update([record_name, estimate_name, mask_name])
        if not {record_name, estimate_name, mask_name}.issubset(files):
            raise ValueError("Fresh record, image observations and mask must be bound together")
        record = json.loads((recovery/record_name).read_text())
        if (record["frame_idx"] != fid or record["inference_frame_idx"] != fid
            or record["is_right"] is not row["is_right"] or record["track_id"] != row["track_id"]
            or record.get("source_request_sha256") != receipt["request_sha256"]
            or record.get("estimated") is not True or record.get("fresh_image_supported") is not True
            or record.get("directly_observed") is not False or record.get("metric_valid") is not False
            or record.get("original_direct_observation") is not False
            or record.get("visual_valid") is not False or record["diagnostics"].get("metric_valid") is not False
            or record["mano"]["betas"] != row["fixed_betas"] or record["hand_scale"] != row["fixed_hand_scale"]):
            raise ValueError("Fresh prior changes identity, shape or observation authority")
        pose = np.r_[record["mano"]["global_orient"], record["mano"]["hand_pose"]]
        translation = np.asarray(record["cam_t"], float)
        if pose.shape != (48,) or not np.isfinite(pose).all() or translation.shape != (3,) or not np.isfinite(translation).all() or translation[2] <= 0:
            raise ValueError("Fresh hand geometry must be finite and in front of the camera")
        # A source-frame landmark cannot acquire confidence merely because a new
        # model supplies a full hand. The existing issue6 mask checker also runs.
        estimate = json.loads((recovery/estimate_name).read_text())
        if estimate["mask"] != Path(mask_name).name or estimate["source"] != f"fresh_sam2_hamer:{receipt['request_sha256']}":
            raise ValueError("Fresh image evidence has a different source")
        checked = load_image_estimate(recovery/"image-observations", side, fid, data["camera"], False)
        points, original = np.asarray(checked["keypoints"]), np.asarray(row["keypoints"])
        keep = points[:,2] > 0
        if (points[:,2] > original[:,2]).any() or not np.array_equal(points[keep,:2], original[keep,:2]):
            raise ValueError("Fresh reconstruction fabricated or moved source image landmarks")
        accepted.append((key, record_name, estimate_name, mask_name))
    if set(files) != expected_files:
        raise ValueError("Recovery files do not match the declared frame outcomes")
    if not accepted:
        raise ValueError("No recovered priors to stage; preserve the rejected result")
    return data, sources, receipt, accepted, axes


def merge(request_path, recovery, output):
    output, recovery = Path(output).resolve(), Path(recovery).resolve()
    data, sources, receipt, accepted, axes = validate_recovery(request_path, recovery)
    bundle_root = Path(data["canonical_bundle"]).parent.resolve()
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output) for p in (bundle_root, recovery)):
        raise ValueError("Use a fresh output outside the immutable source and recovery roots")
    before = {str(p):digest(p) for p,_ in sources.values()}
    output.mkdir(parents=True)
    for (side,fid),(source,_) in sources.items():
        p = output/f"priors/{side}/{fid:06d}.json"
        p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,p)
    for (side,fid),record,estimate,mask in accepted:
        shutil.copyfile(recovery/record, output/f"priors/{side}/{fid:06d}.json")
        for name in (estimate,mask):
            p = output/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(recovery/name,p)
    np.savez_compressed(output/"original-source-confidence.npz", **axes)
    for source,_ in sources.values():
        if digest(source) != before[str(source)]:
            raise ValueError("Canonical source changed while staging fresh priors")
    validate_request(request_path)
    canonical_sources(data)
    validate_recovery(request_path,recovery)
    result = dict(schema="v2d.staged-source-hand-recovery.v1", request_sha256=digest(request_path),
        recovery_receipt_sha256=digest(recovery/"receipt.json"), canonical_manifest_sha256=data["canonical_manifest_sha256"],
        canonical_hand_records=len(sources), replaced_missing_source_priors=[dict(side=k[0],frame_idx=k[1]) for k,*_ in accepted],
        original_observation_flags_changed=False, canonical_source_files_changed=False,
        geometry_fitted=False, retargeting_performed=False, reference_contact_gate_tolerance_m=.001,
        reference_contact_gate_passed=None, source_fit_gate_passed=None, motion_gate_passed=None,
        carry_reference_accepted=False, training_approved=False, physical_task_accepted=False,
        required_next_stages=["source image/depth fit", "source-fit nonregression and missing-observation audit", "robot retargeting under original morphology and actual URDF limits", "1mm native reference-contact and swept checks", "grasp proximity and motion checks before any PPO"])
    result["files"] = {str(p.relative_to(output)):digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
    (output/"merge-receipt.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request",type=Path,required=True)
    p.add_argument("--validate-sources-only",action="store_true")
    p.add_argument("--recovery",type=Path)
    p.add_argument("--out",type=Path)
    a = p.parse_args()
    if a.validate_sources_only:
        sources=canonical_sources(validate_request(a.request))
        print(json.dumps(dict(canonical_hand_sources_verified=len(sources),GPU_inference=False)))
    else:
        if a.recovery is None or a.out is None:p.error("Merge requires --recovery and --out")
        print(json.dumps(merge(a.request,a.recovery,a.out),indent=2))
