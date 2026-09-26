#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One-shot VM bootstrap + calibration-data production for Issue #3 (world-camera
# calibration).  Run from anywhere on a fresh GPU VM; everything is created
# under ~ (repo clone, venv, docker images, weights, data, outputs).
#
#   bash <(curl -fsSL <url>)          # or scp this file and: bash setup_and_run.sh
#
# Env overrides (all optional):
#   HF_TOKEN   huggingface token for the gated dataset (prompted if unset)
#   VM_REPO    repo URL / SSH-path         (default https://github.com/Swaguto/NVIDIA-V2D-Challenge.git)
#   VM_DATASET_ROOT  where track_3/public lands (default ~/v2d_track3/data/hf/track_3/public)
#   VM_WORK          pipeline scratch      (default ~/v2d_calib_work)
#   VM_EPISODE       episode                (default 2)
#   VM_CAMERAS       space-separated cams   (default "ego_cam_b ego_cam_c")
#   VM_WEIGHTS       weights dir            (default ~/v2d_weights)
#   VM_SKIP_PIPELINE set to 1 to stop after setup (data+weights+images ready)
set -euo pipefail

REPO_URL="${VM_REPO:-https://github.com/Swaguto/NVIDIA-V2D-Challenge.git}"
REPO_DIR="${VM_REPO_DIR:-$HOME/NVIDIA-V2D-Challenge}"
VENV_DIR="${VM_VENV:-$HOME/v2d_venv}"
WEIGHTS="${VM_WEIGHTS:-$HOME/v2d_weights}"
DATA_LOCAL="${VM_DATA_LOCAL:-$HOME/v2d_track3/data/hf}"
DATASET_ROOT="${VM_DATASET_ROOT:-$DATA_LOCAL/track_3/public}"
WORK="${VM_WORK:-$HOME/v2d_calib_work}"
EPISODE="${VM_EPISODE:-2}"
CAMERAS="${VM_CAMERAS:-ego_cam_b ego_cam_c}"
SPLIT_INCLUDE="track_3/public/**"

say()  { echo -e "\n=== $* ==="; }
die()  { echo "FATAL: $*" >&2; exit 1; }

say "1/10 system checks"
command -v docker >/dev/null || die "docker not installed"
docker info >/dev/null 2>&1 || die "docker daemon not running"
command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1 && echo "GPU OK ($(nvidia-smi -L | head -1))" || echo "WARN: no GPU visible"
command -v ffmpeg >/dev/null || { echo "installing ffmpeg..."; sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg; }
command -v pip3 >/dev/null || sudo apt-get install -y -qq python3-pip python3-venv

say "2/10 python via venv ($VENV_DIR)"
if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv "$VENV_DIR" || die "venv failed (need python3-venv)"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python --version

say "3/10 repo checkout ($REPO_DIR)"
BRANCH="${VM_BRANCH:-issue3/world-camera-calibration}"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR" || die "git clone failed"
fi
cd "$REPO_DIR"
git config pull.rebase false
git fetch --all --quiet || true
git checkout -B "$BRANCH" "origin/$BRANCH" 2>/dev/null || git checkout -B "$BRANCH"
git pull --ff-only || true
if command -v git-lfs >/dev/null 2>&1; then git lfs pull || true; fi

say "4/10 host packages (editable docker wrappers)"
cd "$REPO_DIR/reconstruction"
python -m pip install --quiet --upgrade pip
./scripts/install_packages.sh || die "install_packages.sh failed"
python -m pip install --quiet opencv-python-headless pyarrow
python -m pip install --quiet -e "$REPO_DIR/reconstruction/modules/v2d_world_calib"

say "5/10 docker images (skip if present)"
build_if_missing() {
  local image="$1" mod="$2"
  if docker image inspect "$image" >/dev/null 2>&1; then
    echo "present: $image"
  else
    echo "building $image..."
    python -m "v2d.${mod}.docker.build" || die "build of $image failed"
  fi
}
build_if_missing v2d_grounding_dino:latest grounding_dino
build_if_missing v2d_sam2:latest sam2
build_if_missing v2d_foundation_stereo:latest foundation_stereo
build_if_missing v2d_foundation_pose:latest foundation_pose

say "6/10 weights"
mkdir -p "$WEIGHTS"
download_if_empty() {
  local dir="$1" cmd="$2"
  if [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
    echo "weights present: $dir"
  else
    eval "$cmd" || echo "WARN: weight download failed for $dir (rerun separately)"
  fi
}
cd "$REPO_DIR/reconstruction"
download_if_empty "$WEIGHTS/gdino" "python -m v2d.grounding_dino.docker.run_download_weights --output_dir $WEIGHTS/gdino"
download_if_empty "$WEIGHTS/sam2"  "python -m v2d.sam2.docker.run_download_weights --target_dir $WEIGHTS/sam2"
download_if_empty "$WEIGHTS/fs"    "python -m v2d.foundation_stereo.docker.run_download_weights --output_dir $WEIGHTS/fs"
download_if_empty "$WEIGHTS/fp"    "python -m v2d.foundation_pose.docker.run_download_weights --output_dir $WEIGHTS/fp"

say "7/10 dataset (gated; needs HF token)"
TOKEN="${HF_TOKEN:-}"
if [ -z "$TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
if [ -z "$TOKEN" ]; then
  read -r -p "HuggingFace token (hf_...): " -s TOKEN
  echo
fi
test -n "$TOKEN" || die "no HF token; set HF_TOKEN or create one at https://huggingface.co/settings/tokens"
huggingface-cli login --token "$TOKEN" >/dev/null 2>&1 || true

if [ ! -f "$DATASET_ROOT/meta/camera_calibration.json" ]; then
  if ! command -v hf >/dev/null 2>&1; then python -m pip install --quiet "huggingface_hub[cli]" --upgrade; fi
  mkdir -p "$DATA_LOCAL"
  hf download nvidia/video_to_data_challenge --repo-type dataset \
    --include "$SPLIT_INCLUDE" --local-dir "$DATA_LOCAL" || die "dataset download failed"
fi
[ -f "$DATASET_ROOT/meta/camera_calibration.json" ] || die "dataset present but camera_calibration.json missing"
du -sh "$DATASET_ROOT"

say "8/10 sanity"
MANIFEST_OK=0
python - "$WORK" <<'PY' || true
import os, sys
work = sys.argv[1]
man = os.path.join(work, "manifest.json")
print("existing run manifest:", man, "->", os.path.exists(man))
PY

if [ "${VM_SKIP_PIPELINE:-0}" = "1" ]; then
  say "9/10 SKIPPED pipeline (VM_SKIP_PIPELINE=1)"
else
  say "9/10 track pipeline (episode $EPISODE, cams: $CAMERAS)"
  mkdir -p "$WORK"
  python -m v2d.world_calib.vm.track_pipeline \
    --data-root "$DATASET_ROOT" \
    --work "$WORK" \
    --episode "$EPISODE" \
    --cameras $CAMERAS \
    --gdino-model-dir "$WEIGHTS/gdino" \
    --sam2-weights "$WEIGHTS/sam2" \
    --fp-weights "$WEIGHTS/fp" \
    --fs-model-dir "$WEIGHTS/fs" 2>&1 | tee "$WORK/pipeline.log"
fi

say "10/10 pack outputs"
if [ -f "$WORK/manifest.json" ]; then
  tar czf "$HOME/v2d_poses_ep${EPISODE}.tar.gz" -C "$WORK" poses intrinsics manifest.json 2>/dev/null \
    && echo "OUTPUTS: $HOME/v2d_poses_ep${EPISODE}.tar.gz" \
    || echo "OUTPUTS: tarball skipped (manifest present; tar the $WORK/poses tree)"
else
  echo "no manifest yet - after the pipeline finishes it will appear in $WORK"
fi

say "DONE"
echo "push back:  $HOME/v2d_poses_ep${EPISODE}.tar.gz   (or scp -r $WORK/poses ...)"