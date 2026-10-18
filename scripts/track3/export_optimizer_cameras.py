"""Export verified independent stereo cameras for the existing temporal optimizer.

This adapts already inferred camera poses; it does not rerun ViPE or change the
camera gauge. Unknown/held camera frames remain explicitly recorded as such.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reconstruction/modules/v2d_issue6"))
from issue6.manifest import load, pose


def export(manifest_path, out):
    m = load(manifest_path)
    if m.get("input_contract", {}).get("reference_pose_tables_read") is not False:
        raise ValueError("Expected label-free scored input contract")
    if out.exists():
        raise FileExistsError(out)
    matrices, known = [], []
    for frame in m["frames"]:
        path = Path(frame["camera_pose_path"])
        record = json.loads(path.read_text())
        if (
            record.get("gt_use") != "independent"
            or record.get("frame_index") != frame["id"]
            or record.get("episode") != m["episode"]
        ):
            raise ValueError("Camera evidence has a different frame or provenance")
        matrices.append(pose({"pose_path": str(path)}))
        known.append(bool(frame["camera_valid"]))
    w2c = np.asarray(matrices, dtype=np.float32)
    c = m["color_camera"]
    intrinsics = np.tile(
        [c[k] for k in ("fx", "fy", "cx", "cy")], (len(w2c), 1)
    ).astype(np.float32)
    out.mkdir(parents=True)
    np.savez_compressed(
        out / "cameras.npz",
        height=c["height"],
        width=c["width"],
        focal=c["fx"],
        intrins=intrinsics,
        w2c=w2c,
    )
    np.savez_compressed(
        out / "observation-validity.npz",
        camera_observed=np.asarray(known),
        frame_indices=np.asarray([f["id"] for f in m["frames"]]),
    )
    receipt = {
        "schema": "v2d.stereo-to-optimizer-cameras.v1",
        "episode": m["episode"],
        "frames": len(w2c),
        "observed_frames": sum(known),
        "method": "existing stereo PnP cameras",
        "video_inference_rerun": False,
        "vipe_executed": False,
        "reference_pose_labels_read": False,
        "input_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "cameras_sha256": hashlib.sha256(
            (out / "cameras.npz").read_bytes()
        ).hexdigest(),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.manifest, args.out)))
