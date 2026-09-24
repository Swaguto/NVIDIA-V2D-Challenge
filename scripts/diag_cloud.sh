#!/bin/bash
echo "=== OS ==="
. /etc/os-release && echo "$PRETTY_NAME"
echo "=== USER ==="; whoami; id
echo "=== DRIVER/CUDA ==="; nvidia-smi | head -12
echo "=== DOCKER GPU TEST ==="
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi 2>&1 | head -8
echo "=== DOCKER GPU TEST (13.0 base for isaac 2.3.2) ==="
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi 2>&1 | head -8
echo "=== DISK ==="; df -h / | tail -1
echo "=== RAM ==="; free -g | head -2
echo "=== CORES ==="; nproc
echo "=== BREV GPU === (instance type from label)"
echo "=== git-lfs ==="; command -v git-lfs || echo "missing"
echo "=== NGC login saved? ==="; grep -c nvcr.io ~/.docker/config.json 2>/dev/null || echo "no"
echo "=== HF token? ==="; ls -la ~/.cache/huggingface/ 2>/dev/null | head -3 || echo "no hf cache"
echo "=== tmux/screen ==="; command -v tmux || echo "no tmux"
echo "=== container root filesystem check ==="; docker info --format '{{.DockerRootDir}}' 2>/dev/null