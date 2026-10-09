#!/usr/bin/env bash
# One GPU preparation queue; review completed fits concurrently in a second queue.
set -euo pipefail
export PATH=/opt/conda/bin:$PATH PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 V2D_OPT_HAND_SHAPE=0
export PYTHONPATH=/workspace/pr32/reconstruction/modules/v2d_issue6:${PYTHONPATH:-}
helpers=/workspace/pr32/reconstruction/validation/issue6
queue=${1:?Queue ID required}
episodes=${2:?Comma-separated episode IDs required}
exec 9>/workspace/pr32-evidence/all20-hand-preparation-$queue.lock
flock -n 9
for episode in ${episodes//,/ }; do
  printf -v item 'e%03d' "$episode"
  root="/workspace/temporal-hands/$item"
  until /opt/conda/bin/python - "$episode" 2>/dev/null <<'PY'
import json,sys
from pathlib import Path
import pyarrow.parquet as pq
episode=int(sys.argv[1]); root=Path(f'/workspace/cad-final/e{episode:03d}/poses/e{episode:03d}')
names=json.loads(pq.ParquetFile(f'/workspace/track3/public/data/chunk-000/episode_{episode:06d}.parquet').schema_arrow.metadata[b'objects'])
assert all((root/f'{name}_fp_report.json').is_file() for name in names)
PY
  do
    kill -0 "$(cat /workspace/pr32-evidence/remaining-cad.pid)" 2>/dev/null || exit 1
    sleep 5
  done
  /workspace/hand-runtime/bin/python "$helpers/prepare_all_episode_manifest.py" --episodes "$episode" --protocol-receipt "/workspace/pr32-evidence/fit-queue-$queue-protocol.json"
  if [ ! -f "$root/fit_receipt.json" ]; then
    /workspace/hand-runtime/bin/python -u "$helpers/enhanced_hands.py" --episode "$episode" --use-saved-detections > "/workspace/pr32-evidence/$item-final-hand-predictions.log" 2>&1
    /workspace/hand-runtime/bin/python -u "$helpers/full_hand_masks.py" --episode "$episode" > "/workspace/pr32-evidence/$item-final-hand-masks.log" 2>&1
    /workspace/vipe-runtime/bin/python -u "$helpers/vipe_stereo_cameras.py" --episode "$episode" > "/workspace/pr32-evidence/$item-final-vipe.log" 2>&1
    /workspace/hand-runtime/bin/python -u "$helpers/run_temporal_fit.py" --episode "$episode" > "/workspace/pr32-evidence/$item-final-fit.log" 2>&1
  fi
  /opt/conda/bin/python - "$root/fit_receipt.json" <<'PY'
import json,sys
receipt=json.load(open(sys.argv[1]))
assert receipt['fixed_shape_exactly_preserved'] and receipt['root_iterations']==50 and receipt['smooth_iterations']==300
PY
  echo "Native fit episode $episode complete; six-stage reconstruction and review still required"
done
echo "Requested native fits $episodes completed"
