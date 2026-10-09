"""Execute a reviewed upstream training script with one in-memory config hook.

Usage (inside the upstream Isaac Lab environment):
python scripts/train_reliability.py --upstream upstream/video_to_data --sidecar PATH -- [upstream arguments]
Omit --sidecar for unchanged baseline execution. No upstream files are edited.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("--upstream", type=Path, required=True)
parser.add_argument("--sidecar", type=Path)
parser.add_argument("--variant", choices=["baseline", "force_closure", "reliability", "reliability_force_closure"])
parser.add_argument("--revision-lock", type=Path, help="Reviewed revision lock (default: configs/upstream.json)")
parser.add_argument("args", nargs=argparse.REMAINDER)
args = parser.parse_args()
upstream = args.upstream.resolve()
lock = json.loads((args.revision_lock or Path(__file__).resolve().parents[1] / "configs/upstream.json").read_text())
commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
if commit != lock["commit"]:
    raise SystemExit("Upstream revision differs from the validated adapter revision")
if subprocess.check_output(["git", "-C", str(upstream), "diff", "HEAD", "--name-only"], text=True).strip():
    raise SystemExit("Upstream tracked files are modified; use a clean baseline checkout")
script = upstream / "robotic_grounding/scripts/rsl_rl/train.py"
source = script.read_text(encoding="utf-8")
variant = args.variant or ("reliability" if args.sidecar else "baseline")
if variant == "baseline" and args.sidecar:
    raise SystemExit("Baseline must not receive a sidecar")
if variant.startswith("reliability") and not args.sidecar:
    raise SystemExit("Confidence variant requires --sidecar")
if variant != "baseline":
    sidecar = str(args.sidecar.resolve(strict=True)) if args.sidecar else None
    marker = "    # create isaac environment\n"
    if source.count(marker) != 1:
        raise SystemExit("Upstream integration point changed; refusing an unverified injection")
    source = source.replace(marker, "    from v2d_reliability.isaac_adapter import configure_variant\n"
                            f"    configure_variant(env_cfg, variant={variant!r}, sidecar_path={sidecar!r})\n" + marker)
sys.path.insert(0, str(script.parent))
sys.argv = [str(script), *(args.args[1:] if args.args[:1] == ["--"] else args.args)]
os.chdir(upstream / "robotic_grounding")
exec(compile(source, str(script), "exec"), {"__name__": "__main__", "__file__": str(script)})
