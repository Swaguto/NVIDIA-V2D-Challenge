"""Run native pot registration with the hash-pinned local model data cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--models", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--episode", type=int, choices=[0, 2], default=2)
parser.add_argument("--frames", type=int, default=1, help="one-frame diagnostic by default; 0 means full episode")
args = parser.parse_args()
REPO = Path(__file__).resolve().parents[2]
if args.out.exists() and any(args.out.iterdir()):
    parser.error("output must be fresh")
if args.frames < 0:
    parser.error("frames must be nonnegative")
expected = json.loads((REPO / "docs/team/issue5/evidence/episode39_models.json").read_text())
for row in expected:
    path = args.models / row["model"].replace("/", "--") / row["revision"] / row["file"]
    if path.stat().st_size != row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError(f"pinned model changed: {path}")
os.environ["V2D_PUBLIC"] = str(args.data)
os.environ["V2D_CALIB"] = str(args.data / "meta/camera_calibration.json")
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, str(REPO / "scripts/track3"))
import register_poses as native

native.GDINO = str(args.models / "IDEA-Research--grounding-dino-base/12bdfa3120f3e7ec7b434d90674b3396eccf88eb")
native.SAM2 = str(args.models / "facebook--sam2.1-hiera-large/665f8e2ad61cf5f53d65644ff27c8ee525124610")
native.torch.set_num_threads(4)
sys.argv = ["register_poses", "--episode", str(args.episode), "--no-gt",
            "--objects", "white_pot,white_pot_lid", "--frames", str(args.frames),
            "--out", str(args.out), "--log", "1", "--seed", "0"]
native.main()
