#!/usr/bin/env bash
# Replay preparation for the pinned PR33 A40 image; models are recovered separately.
set -euo pipefail
export PATH=/opt/conda/bin:$PATH
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYOPENGL_PLATFORM=egl
helpers=$(cd "$(dirname "$0")" && pwd)
project=$(cd "$helpers/../../.." && pwd)
runtime_root=${ISSUE6_RUNTIME_ROOT:-/workspace}
evidence=${ISSUE6_RUNTIME_EVIDENCE:-"$runtime_root/pr32-evidence"}
mkdir -p "$evidence"
python - <<'PY'
import torch
assert torch.__version__ == '2.5.1+cu124', torch.__version__
assert torch.cuda.is_available()
print(torch.__version__, torch.cuda.get_device_name(0))
PY

clone_pinned() {
  local url=$1 root=$2 commit=$3
  if [ ! -d "$root/.git" ]; then
    if [ -e "$root" ] && { [ ! -d "$root" ] || [ -n "$(ls -A "$root")" ]; }; then
      echo "Existing nonempty non-Git runtime path: $root" >&2; return 1
    fi
    git clone --filter=blob:none --no-checkout "$url" "$root"
    git -C "$root" checkout --detach "$commit"
  fi
  test "$(git -C "$root" rev-parse HEAD)" = "$commit"
}

clone_pinned https://github.com/geopavlakos/hamer.git "$runtime_root/hamer" 3a01849f4148352e9260b69bf28b65d1671a4905
clone_pinned https://github.com/ZhengdiYu/Dyn-HaMR.git "$runtime_root/Dyn-HaMR" fa9cd7412c205fd15ee4139c8caacf79bf6167e6
clone_pinned https://github.com/ZhengdiYu/hamer.git "$runtime_root/Dyn-HaMR/third-party/hamer" 2a5b8ee852d87fdb56d65df7fb0e473c0d4f7648
clone_pinned https://github.com/nv-tlabs/vipe.git "$runtime_root/ViPE" fdc79359ec6d063c86b9d203ca3d7054a9636823
patch="$helpers/dynhamr-native-fixed-shape.patch"
if git -C "$runtime_root/Dyn-HaMR" apply --ignore-space-change --check "$patch"; then
  git -C "$runtime_root/Dyn-HaMR" apply --ignore-space-change "$patch"
else
  # Git archives use LF; a previous Windows transfer may contain CRLF context.
  git -C "$runtime_root/Dyn-HaMR" apply --ignore-space-change --reverse --check "$patch"
fi

python -m venv --system-site-packages "$runtime_root/hand-runtime"
hand="$runtime_root/hand-runtime/bin/python"
"$hand" -m pip install numpy==1.26.4 opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86 \
  smplx==0.1.28 yacs==0.1.8 hydra-core==1.3.7 pytorch-lightning==2.6.6 timm==1.0.30 \
  webdataset==1.0.2 braceexpand==0.1.7 dill==0.4.1 av==18.1.0 loguru==0.7.3 \
  mediapipe==0.10.18 ultralytics==8.1.34 decord==0.6.0 matplotlib==3.8.4 \
  torchgeometry==0.1.2 tensorboard==2.18.0 \
  git+https://github.com/mattloper/chumpy.git@580566eafc9ac68b2614b64d6f7aaa84eebb70da \
  git+https://github.com/hassony2/manopth.git@4f1dcad1201ff1bfca6e065a85f0e3456e1aa32b
SAM2_BUILD_CUDA=0 "$hand" -m pip install --no-deps \
  git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4 \
  git+https://github.com/lixiny/manotorch.git@a2a70c591f91551078b7bb2af9b5d9f275b626e0 \
  -e "$runtime_root/hamer" -e "$runtime_root/Dyn-HaMR" \
  -e "$project/reconstruction/modules/v2d_common" \
  -e "$project/reconstruction/modules/v2d_hamer/lib" \
  -e "$project/reconstruction/modules/v2d_issue6" \
  -e "$project/reconstruction/modules/v2d_sam2/lib" \
  -e "$project/reconstruction/modules/v2d_hand_alignment/lib"
"$hand" -m pip freeze > "$evidence/hand-runtime-packages.txt"

python -m venv --system-site-packages "$runtime_root/vipe-runtime"
vipe="$runtime_root/vipe-runtime/bin/python"
"$vipe" -m pip install numpy==1.26.4 opencv-python==4.11.0.86 transformers==4.48.3 \
  huggingface-hub==0.25.1 rerun-sdk==0.22.0 python-pycg==0.10.1 \
  OpenEXR==3.2.12 kornia==0.8.0 viser==1.0.1 ninja==1.11.1.4 gdown==5.2.0 \
  timm==1.0.30 hydra-core==1.3.7
export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/include/eigen3:${CPATH:-}
export TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=${ISSUE6_BUILD_JOBS:-6}
"$vipe" -m pip install --no-build-isolation --no-deps -e "$runtime_root/ViPE"
"$vipe" -m pip freeze > "$evidence/vipe-runtime-packages.txt"
"$hand" -c 'import hamer.models, manopth, manotorch, sam2, mediapipe, issue6.worker; print("Native hand imports verified")'
"$vipe" -c 'import torch; import vipe_ext; from vipe.slam.system import SLAMSystem; print("ViPE extension verified")'
sha256sum "$patch" "$runtime_root/Dyn-HaMR/dyn-hamr/optim/optimizers.py" > "$evidence/native-runtime-source.sha256"
echo 'Runtime imports verified; inference and model-input verification still required'
