"""Clone source at the reviewed commit without downloading licensed model assets."""
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
lock = json.loads((root / "configs/upstream.json").read_text())
target = root / "upstream/video_to_data"
env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
if not target.exists():
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1",
                    "--branch", lock["ref"], lock["repository"], str(target)], env=env, check=True)
subprocess.run(["git", "-C", str(target), "config", "core.longpaths", "true"], check=True)
actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
if actual != lock["commit"]:
    subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", lock["commit"]], env=env, check=True)
if subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True).strip() and (target / "README.md").exists():
    raise SystemExit("Upstream has local changes; refusing to overwrite them")
subprocess.run(["git", "-C", str(target), "checkout", "--detach", lock["commit"]], env=env, check=True)
if subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True).strip():
    raise SystemExit("Checkout is incomplete or modified; inspect git status before using baseline")
print(f"Pinned source ready: {target}. LFS assets and licensed weights are not downloaded.")
