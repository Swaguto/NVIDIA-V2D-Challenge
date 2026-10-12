"""Development quality checks with detector-bounded SAM and retained MANO scale.

This is a recheck of retained estimates, not a complete reconstruction. The native writer retains fitted scale. Each consistency-passing native
export is independently reconstructed and verified before it is marked consistent.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "quality-source",
        "source",
        "proposals",
        "out",
        "depth-root",
        "sam-checkpoint",
        "modules",
        "mano-assets",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists()
    sys.path.insert(0, str(a.quality_source))
    sys.path.insert(0, str(a.modules / "v2d_issue6"))
    import automatic_hand_quality
    from automatic_hands import method_hash, sha, write
    from issue6.geometry import hand_geometry
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    a.config = a.quality_source / "automatic_hands.json"
    policy = json.loads(a.config.read_text())
    manifest = json.loads((a.source / "manifest.json").read_text())
    assert manifest["episode"] not in policy["holdout_episodes"], (
        "Freeze complete effective method before holdouts"
    )
    proposals = json.loads((a.proposals / "receipt.json").read_text())
    assert proposals["geometry_scale_preserved"] and not proposals["quality_accepted"]
    hashes = method_hash(a.config)
    assert proposals["method_files"] == hashes
    assert sha(a.sam_checkpoint) == proposals["sam_checkpoint_sha256"]
    assert sha(a.proposals / "detections.json") == proposals["detections_sha256"]
    assert sha(a.proposals / "faces.npy") == proposals["faces_sha256"]
    assert (
        sha(a.modules / "v2d_issue6/issue6/geometry.py")
        == proposals["geometry_helper_sha256"]
    )
    assert (
        sha(a.modules / "v2d_hamer/lib/align_hands.py")
        == proposals["upstream_mano_helper_sha256"]
    )
    assert (
        sha(a.mano_assets / "models/MANO_RIGHT.pkl") == proposals["mano_right_sha256"]
    )
    for row in proposals["records"]:
        frame = row["frame"]
        side = row["side"]
        track = 3 if side == "right" else 2
        assert (
            sha(a.proposals / f"raw/{side}/{frame:06d}.npz")
            == row["raw_geometry_sha256"]
        )
        assert (
            sha(a.proposals / f"records/{track}/{frame:06d}.json")
            == row["export_record_sha256"]
        )
    assert proposals["adapter_sha256"] == sha(
        Path(__file__).with_name("prepare_retained_hand_quality_inputs.py")
    )

    random.seed(policy["seed"])
    np.random.seed(policy["seed"])
    torch.manual_seed(policy["seed"])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False

    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    receipt = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": sha(Path(__file__)),
        "github_method_files": hashes,
        "proposal_receipt_sha256": sha(a.proposals / "receipt.json"),
        "runtime": {
            "python": sys.version,
            "gpu": torch.cuda.get_device_name(0),
            "packages": {
                name: version(name)
                for name in (
                    "torch",
                    "numpy",
                    "opencv-python",
                    "pyrender",
                    "trimesh",
                    "SAM-2",
                    "manotorch",
                )
            },
        },
        "upstream_inference_rerun": False,
        "quality_accepted": False,
        "physical_task_accepted": False,
        "status": "running",
    }
    try:
        predictor = SAM2ImagePredictor(
            build_sam2(
                "configs/sam2.1/sam2.1_hiera_l.yaml",
                str(a.sam_checkpoint),
                device="cuda",
            )
        )
        automatic_hand_quality.run(a, predictor=predictor)
        assert method_hash(a.config) == hashes

        module_path = a.modules / "v2d_hamer/lib/align_hands.py"
        spec = importlib.util.spec_from_file_location(
            "retained_quality_export_mano", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        mano = module._make_mano(str(a.mano_assets))
        exports = []
        for path in sorted((a.out / "accepted/records").glob("*/*.json")):
            native = json.loads(path.read_text())
            track = int(native["track_id"])
            side = "right" if native["is_right"] else "left"
            frame = int(native["frame_idx"])
            original = json.loads(
                (a.proposals / f"records/{track}/{frame:06d}.json").read_text()
            )
            assert native["mano"] == original["mano"]
            assert native["hand_scale"] == original["hand_scale"], (
                "Native export changed checked scale"
            )
            assert native["hand_scale"] == proposals["retained_global_scales"][side]
            angles = np.asarray(
                native["mano"]["global_orient"] + native["mano"]["hand_pose"],
                np.float32,
            )
            betas = np.asarray(native["mano"]["betas"], np.float32)
            with torch.inference_mode():
                result = mano(
                    torch.from_numpy(angles)[None], torch.from_numpy(betas)[None]
                )
            vertices, joints = hand_geometry(
                result.verts[0].numpy(),
                result.joints[0].numpy(),
                native["is_right"],
                native["hand_scale"],
                native["cam_t"],
            )
            geometry = a.out / f"geometry/{side}/{frame:06d}.npz"
            with np.load(geometry, allow_pickle=False) as data:
                assert bool(data["accepted"])
                np.testing.assert_allclose(
                    vertices, data["vertices"], atol=1e-7, rtol=0
                )
                np.testing.assert_allclose(joints, data["joints"], atol=1e-7, rtol=0)
                error = max(
                    float(np.max(np.abs(vertices - data["vertices"]))),
                    float(np.max(np.abs(joints - data["joints"]))),
                )
            native["quality_accepted"] = False
            native["geometry_export_verified"] = True
            native["physical_task_accepted"] = False
            destination = path
            write(destination, native)
            exports.append(
                {
                    "frame": frame,
                    "side": side,
                    "hand_scale": native["hand_scale"],
                    "geometry_sha256": sha(geometry),
                    "record_sha256": sha(destination),
                    "maximum_geometry_error_m": error,
                }
            )
        write(
            a.out / "geometry-export-verification.json",
            {
                "exports": exports,
                "all_consistency_passing_exports_verified": True,
                "native_writer_geometry_verified": True,
                "native_scale_modified_by_verifier": False,
                "geometry_helper_sha256": sha(
                    a.modules / "v2d_issue6/issue6/geometry.py"
                ),
                "mano_helper_sha256": sha(module_path),
                "mano_right_sha256": sha(a.mano_assets / "models/MANO_RIGHT.pkl"),
                "quality_accepted": False,
                "physical_task_accepted": False,
            },
        )
        summary = json.loads((a.out / "summary.json").read_text())
        assert len(exports) == sum(summary["hand_counts"].values())
        receipt.update(
            status="completed_consistency_checks_proposals_unaccepted",
            effective_method_unchanged_during_run=True,
            summary_sha256=sha(a.out / "summary.json"),
            frame_evidence_sha256=sha(a.out / "frame-evidence.json"),
            geometry_export_verification_sha256=sha(
                a.out / "geometry-export-verification.json"
            ),
            verified_exports=len(exports),
        )
    except Exception as error:
        receipt.update(status="failed", error=str(error))
        raise
    finally:
        receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        path = (
            a.out / "run-receipt.json"
            if a.out.exists()
            else a.out.with_suffix(".failed.json")
        )
        write(path, receipt)


if __name__ == "__main__":
    main()
