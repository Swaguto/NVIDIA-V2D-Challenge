#!/bin/bash
set -euo pipefail
say() { echo -e "\n### $*"; }

say "Install git-lfs + helper tools"
sudo apt-get update -qq
sudo apt-get install -y -qq git-lfs ffmpeg tmux htop >/dev/null 2>&1 || sudo apt-get install -y git-lfs ffmpeg
git lfs version

say "Clone video_to_data (release/0.2.0)"
if [ ! -d "$HOME/video_to_data" ]; then
  git clone -b release/0.2.0 --depth 1 https://github.com/nvidia-isaac/video_to_data.git "$HOME/video_to_data"
fi
cd "$HOME/video_to_data"

say "Pull LFS assets (~1 GB)"
git lfs pull
du -sh .git/lfs

say "Create host venv for the orchestrator"
python3 -m venv "$HOME/venvs/v2d" || true
# shellcheck disable=SC1091
source "$HOME/venvs/v2d/bin/activate"
cd "$HOME/video_to_data/reconstruction"
pip install -qq --upgrade pip
pip install -qq -e modules/v2d_common -e modules/v2d_docker -e modules/v2d_task_library_loader/docker 2>&1 | tail -3

say "Reserve disk scratch for build"
# Docker build cache goes under /var/lib/docker (103 GB free / 124 G total — fine)

echo "HOUSEKEEPING DONE"