#!/usr/bin/env bash
set -euo pipefail

repo="${TRACK3_REPO:-/workspace/NVIDIA-V2D-Challenge}"
data_root="${TRACK3_DATA_ROOT:-/workspace/data}"
run_root="${TRACK3_RUN_ROOT:-/workspace/track3-validation}"
weights="${FOUNDATIONPOSE_WEIGHTS_DIR:-/workspace/weights}"
public="$data_root/track_3/public"

if [[ ! -d "$repo/.git" ]]; then
  git clone --branch feat/track3-runpod-validation \
    https://github.com/Swaguto/NVIDIA-V2D-Challenge.git "$repo"
fi

mkdir -p "$data_root" "$run_root" "$weights"
hf download nvidia/video_to_data_challenge --repo-type dataset \
  --include 'track_3/public/meta/*' \
  --include 'track_3/public/data/chunk-000/episode_000000.parquet' \
  --include 'track_3/public/mesh/white_pot/*' \
  --include 'track_3/public/mesh/white_pot_lid/*' \
  --include 'track_3/public/videos/chunk-000/observation.images.ego_cam_c/episode_000000.mp4' \
  --include 'track_3/public/videos/chunk-000/observation.images.ego_cam_b/episode_000000.mp4' \
  --local-dir "$data_root"

if [[ ! -f "$weights/2023-10-28-18-33-37/model_best.pth" || \
      ! -f "$weights/2024-01-11-20-02-45/model_best.pth" ]]; then
  gdown --continue \
    'https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i' \
    -O "$weights/"
fi
for stamp in 2023-10-28-18-33-37 2024-01-11-20-02-45; do
  test -s "$weights/$stamp/model_best.pth"
  test -s "$weights/$stamp/config.yml"
done

export V2D_PUBLIC="$public"
export V2D_CALIB="$public/meta/camera_calibration.json"
export V2D_FOUNDATIONPOSE=/opt/FoundationPose
export V2D_FP_SHIM=''
export V2D_FP_PYTHON=/opt/conda/bin/python
export FOUNDATIONPOSE_WEIGHTS_DIR="$weights"

cd "$repo"
for phase in cameras masks poses scores overlays; do
  python scripts/track3/batch_public.py --phase "$phase" \
    --data "$public" --root "$run_root" --episodes 0
  python - "$run_root/batch_$phase.json" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
assert manifest.get('0') in ('done', 'cached'), manifest
PY
done

python scripts/track3/render_cad_overlay.py --episode 0 --data "$public" \
  --predictions "$run_root/e000/poses" \
  --out "$run_root/e000/cad_mesh_overlay_color.mp4"
python scripts/track3/report_public.py --data "$public" --root "$run_root"
python - "$run_root/e000" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for name in ('scores_observed.json', 'scores_inferred.json'):
    data = json.loads((root / name).read_text())
    print(name, data['mean_metrics'])
for name in ('overlay.mp4', 'cad_mesh_overlay_color.mp4'):
    path = root / name
    assert path.stat().st_size > 0, path
    print(name, path.stat().st_size, 'bytes')
PY
