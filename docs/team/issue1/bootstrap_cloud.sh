#!/bin/bash
set -euo pipefail

# =============================================================================
# V2D Track 3 — Issue #1: Cloud GPU bootstrap
#
# Prepares a fresh cloud GPU instance (Ubuntu 22.04/24.04, x86_64) to run the
# V2D pipeline. Run as a non-root user with sudo. Requires:
#   - NVIDIA GPU: A100 / H100 / L40S (NOT Blackwell sm_120)
#   - NGC API key (https://ngc.nvidia.com/signin) for the Isaac Lab base image
#
# Usage:
#   ./bootstrap_cloud.sh [--with-reconstruction] [--with-robo-grounding]
#
# Default (no flags) = GPU + Docker + toolkit + repo clone only.
# Recommended first run on a fresh box  →  ./bootstrap_cloud.sh
# Full pipeline box                    →  ./bootstrap_cloud.sh --with-reconstruction
# RL box (robotic_grounding only)      →  ./bootstrap_cloud.sh --with-robo-grounding
# =============================================================================

WITH_RECON=false
WITH_RG=false
for arg in "$@"; do
  case "$arg" in
    --with-reconstruction)   WITH_RECON=true ;;
    --with-robo-grounding)   WITH_RG=true ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
say()  { echo -e "${GREEN}[bootstrap]${NC} $*"; }
die()  { echo -e "${RED}[bootstrap] ERROR:${NC} $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

sudo -v || die "needs sudo"

# ------------------------------------------------------------------------------
# 0. Reject unsupported GPUs early (Blackwell sm_120 is NOT supported by V2D)
# ------------------------------------------------------------------------------
if have nvidia-smi; then
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ') "
  case " $(nvidia-smi --query-gpu=name --format=csv,noheader) " in
    *"RTX PRO 6000 Blackwell"*|*"B200"*|*"B100"*|*"GB200"*|*"RTX 5090"*) die "Blackwell GPU not supported (sm_120). Use A100/H100/L40S/A6000." ;;
  esac
  say "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader) (cc=$CC)"
else
  say "nvidia-smi not found yet — installing driver via ubuntu-drivers below."
fi

# ------------------------------------------------------------------------------
# 1. NVIDIA driver (>= 580) + CUDA
# ------------------------------------------------------------------------------
if ! have nvidia-smi || ! nvidia-smi >/dev/null 2>&1; then
  say "Installing NVIDIA driver (open kernel modules, >=580 recommended)…"
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends ubuntu-drivers-common
  DRIVER="$(ubuntu-drivers devices | awk '/nvidia-driver-5[0-9][0-9]/ {print $2; exit}')"
  : "${DRIVER:=nvidia-driver-580}"
  sudo apt-get install -y --no-install-recommends "${DRIVER}"
  say "Driver installed: $DRIVER — REBOOT REQUIRED before continuing."
  echo "Run:  sudo reboot"
  exit 0
fi
DRV_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
say "Driver $DRV_VER present."

# ------------------------------------------------------------------------------
# 2. Docker Engine + NVIDIA Container Toolkit
# ------------------------------------------------------------------------------
if ! have docker; then
  say "Installing Docker Engine…"
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl gnupg
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
sudo usermod -aG docker "$USER" || true

if ! docker info >/dev/null 2>&1; then
  sudo systemctl enable --now docker
  sudo systemctl restart docker
  say "Waiting for docker daemon…"; sleep 3
fi

if ! have nvidia-ctk; then
  say "Installing NVIDIA Container Toolkit…"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null || true
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1 \
  || say "GPU-in-docker check skipped (image pull may be slow) — run 'docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi' to confirm."

say "GPU + Docker + toolkit ready."

# ------------------------------------------------------------------------------
# 3. System tools + repo clone (release/0.2.0)
# ------------------------------------------------------------------------------
sudo apt-get install -y git git-lfs python3-venv ffmpeg libgl1 libglib2.0-0 >/dev/null
git lfs install
WORK="$(pwd)"
if [ ! -d "$WORK/video_to_data" ]; then
  say "Cloning video_to_data (release/0.2.0)…"
  git clone -b release/0.2.0 --depth 1 https://github.com/nvidia-isaac/video_to_data.git
  cd "$WORK/video_to_data"
else
  cd "$WORK/video_to_data" && git checkout release/0.2.0
fi
say "Pull LFS assets (robot URDFs, meshes, demo video)…"
git lfs pull
say "Repo ready at $WORK/video_to_data"

# ------------------------------------------------------------------------------
# 4. Optional: build the robotic-grounding images (loader + rg), needs NGC key
# ------------------------------------------------------------------------------
if $WITH_RG || $WITH_RECON; then
  if [ -z "${NGC_API_KEY:-}" ]; then
    say "NGC_API_KEY not set — skipping container build. Set it and rerun with --build.",
    say "   Get your key at https://ngc.nvidia.com/signin  →  'docker login nvcr.io' (user: \$oauthtoken)."
  else
    echo "$NGC_API_KEY" | sudo docker login nvcr.io -u '$oauthtoken' --password-stdin
    cd "$WORK/video_to_data"
    python3 -m venv "$HOME/venvs/v2d" || true
    # shellcheck disable=SC1091
    source "$HOME/venvs/v2d/bin/activate"
    cd reconstruction
    pip install -e modules/v2d_common -e modules/v2d_docker -e modules/v2d_task_library_loader/docker >/dev/null
    if $WITH_RECON; then
      say "Building reconstruction images…"
      ./scripts/build_containers.sh
    fi
    cd ../robotic_grounding
    say "Building loader + robotic-grounding images (large; may take a while)…"
    python scripts/run_pipeline_docker.py --build-only
    say "Containers built."
  fi
fi

# ------------------------------------------------------------------------------
# 5. Sync the track_3 dataset from the local dev box (optional helper)
#    On the cloud box, simply:  rsync -av --info=progress2 $USER@<dev-box>:~/track3_data/ ./data/
# or re-download from HF (needs `hf download`):
#    pip install -U "huggingface_hub[cli]" && hf download nvidia/video_to_data_challenge --repo-type dataset --include 'track_3/*' --local-dir ./data/hf/track_3
# ------------------------------------------------------------------------------

say "Done. Next:  docker login nvcr.io  →  build images per docs/SETUP.md  →  run smoke test (see README_ISSUE1.md)."