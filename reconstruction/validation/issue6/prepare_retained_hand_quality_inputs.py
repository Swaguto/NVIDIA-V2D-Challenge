"""Adapt retained directly observed Dyn-HaMR proposals to GitHub quality checks.

No model inference or human observation is fabricated. Retained global scales and independent current-frame
detector boxes preserve the source geometry and define SAM search boxes. This reruns consistency checks, not full
source reconstruction. All outputs remain unaccepted development proposals.
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "original-run",
        "source",
        "temporal-source",
        "modules",
        "quality-source",
        "mano-assets",
        "sam-checkpoint",
        "out",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists()
    sys.path.insert(0, str(a.modules / "v2d_issue6"))
    sys.path.insert(0, str(a.quality_source))
    from automatic_hands import method_hash, sha, write
    from issue6.geometry import hand_geometry, hand_translation

    config = a.quality_source / "automatic_hands.json"
    manifest = json.loads((a.source / "manifest.json").read_text())
    policy = json.loads(config.read_text())
    assert manifest["episode"] not in policy["holdout_episodes"], (
        "Development adapter must not silently run a holdout"
    )
    torch.manual_seed(policy["seed"])
    module_path = a.modules / "v2d_hamer/lib/align_hands.py"
    spec = importlib.util.spec_from_file_location("retained_quality_mano", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mano = module._make_mano(str(a.mano_assets))
    a.out.mkdir()
    np.save(a.out / "faces.npy", mano.th_faces.numpy())
    hashes, rows, refused, counts = [], [], [], {"left": 0, "right": 0}
    c = manifest["color_camera"]
    wearer_path = a.temporal_source / "wearer_candidates.json"
    wearer = json.loads(wearer_path.read_text())
    assert wearer["episode"] == manifest["episode"] and not wearer.get(
        "manual_annotations", False
    )
    detector_rows = {int(row["frame_index"]): row for row in wearer["frames"]}
    assert len(detector_rows) == len(manifest["frames"])
    scales = {}
    for side, track in (("left", 2), ("right", 3)):
        values = [
            float(json.loads(path.read_text())["hand_scale"])
            for path in (a.original_run / f"align/records/{track}").glob("*.json")
        ]
        assert values and np.isfinite(values).all() and min(values) > 0
        np.testing.assert_allclose(values, values[0], atol=1e-12, rtol=0)
        scales[side] = values[0]
    for frame in manifest["frames"]:
        index = frame["id"]
        image = a.source / f"color/{index:06d}.png"
        depth = a.original_run / f"depth/{index:06d}.npy"
        row = {
            "frame": index,
            "timestamp_s": frame["timestamp_s"],
            "source_image_sha256": sha(image),
            "depth_sha256": sha(depth),
        }
        rows.append(row)
        for side, track in (("left", 2), ("right", 3)):
            record_path = a.original_run / f"hands/records/{track}/{index:06d}.json"
            if not record_path.exists():
                continue
            record = json.loads(record_path.read_text())
            assert record["frame_idx"] == index and record["track_id"] == track
            if not record.get("directly_observed", False):
                refused.append(
                    {"frame": index, "side": side, "reason": "not_directly_observed"}
                )
                continue
            detection = detector_rows[index]
            chosen = detection["selected"][side]["candidate_index"]
            if chosen is None:
                refused.append(
                    {
                        "frame": index,
                        "side": side,
                        "reason": "no_independent_current_frame_detection",
                    }
                )
                continue
            candidate = detection["candidates"][chosen]
            if (
                candidate["confidence"] < policy["detector_confidence"]
                or candidate.get("owner_rejection") is not None
            ):
                refused.append(
                    {
                        "frame": index,
                        "side": side,
                        "reason": "unsupported_current_frame_detection",
                    }
                )
                continue
            assert candidate["class"] == int(side == "right")
            box = np.asarray(candidate["box_xyxy"], float)
            assert (
                box.shape == (4,)
                and np.isfinite(box).all()
                and box[0] < box[2]
                and box[1] < box[3]
            )
            assert max(box[0], 0) < min(box[2], c["width"]) and max(box[1], 0) < min(
                box[3], c["height"]
            )
            alignment_path = a.original_run / f"align/records/{track}/{index:06d}.json"
            alignment = json.loads(alignment_path.read_text())
            assert (
                alignment["mano"] == record["mano"]
                and alignment["hand_scale"] == scales[side]
            )
            angles = np.asarray(
                record["mano"]["global_orient"] + record["mano"]["hand_pose"],
                np.float32,
            )
            betas = np.asarray(record["mano"]["betas"], np.float32)
            with torch.inference_mode():
                result = mano(
                    torch.from_numpy(angles)[None], torch.from_numpy(betas)[None]
                )
            translation = hand_translation(record["camera"], c)
            # The retained converter already stores real-camera intrinsics.
            # Refuse the native-record export if its camera needs another map.
            np.testing.assert_allclose(
                translation, record["camera"]["pred_cam_t_full"], atol=1e-7, rtol=0
            )
            vertices, joints = hand_geometry(
                result.verts[0].numpy(),
                result.joints[0].numpy(),
                side == "right",
                scales[side],
                translation,
            )
            raw = a.out / f"raw/{side}/{index:06d}.npz"
            raw.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(raw, vertices=vertices, joints=joints, box=box)
            dest = a.out / f"records/{track}/{index:06d}.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            record["hand_scale"] = scales[side]
            write(dest, record)
            hashes.append(
                {
                    "frame": index,
                    "side": side,
                    "record_sha256": sha(record_path),
                    "alignment_record_sha256": sha(alignment_path),
                    "hand_scale": scales[side],
                    "detector_candidate_index": chosen,
                    "detector_box": box.tolist(),
                    "export_record_sha256": sha(dest),
                    "raw_geometry_sha256": sha(raw),
                }
            )
            counts[side] += 1
    write(a.out / "detections.json", rows)
    receipt = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "adapter_sha256": sha(Path(__file__)),
        "source_manifest_sha256": sha(a.source / "manifest.json"),
        "method_files": method_hash(config),
        "sam_checkpoint_sha256": sha(a.sam_checkpoint),
        "geometry_helper_sha256": sha(a.modules / "v2d_issue6/issue6/geometry.py"),
        "upstream_mano_helper_sha256": sha(module_path),
        "mano_right_sha256": sha(a.mano_assets / "models/MANO_RIGHT.pkl"),
        "temporal_fit_receipt_sha256": sha(a.temporal_source / "fit_receipt.json"),
        "detections_sha256": sha(a.out / "detections.json"),
        "faces_sha256": sha(a.out / "faces.npy"),
        "records": hashes,
        "refused": refused,
        "counts": counts,
        "frames": len(rows),
        "hand_geometry_source": "Retained Dyn-HaMR temporal estimates tagged directly observed; no HaMeR or Dyn-HaMR inference rerun",
        "crop_source": "Independent current-frame YOLO wearer detection; no hand pose supplied by optical flow",
        "detector_source_sha256": sha(wearer_path),
        "retained_global_scales": scales,
        "hand_scale_fit_rerun": False,
        "geometry_scale_preserved": True,
        "manual_frame_or_crop_choices": False,
        "contact_annotations_or_GT_read": False,
        "original_source_or_observation_masks_modified": False,
        "full_source_reconstruction_reproduced": False,
        "freeze_sha256": None,
        "quality_accepted": False,
        "physical_task_accepted": False,
    }
    write(a.out / "receipt.json", receipt)
    print(
        json.dumps(
            {
                "frames": len(rows),
                "proposals": counts,
                "refused": len(refused),
                "original_source_unchanged": True,
                "quality_accepted": False,
            }
        )
    )


if __name__ == "__main__":
    main()
