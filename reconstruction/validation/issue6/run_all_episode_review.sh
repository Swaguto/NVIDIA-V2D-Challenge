#!/usr/bin/env bash
# The pinned runtime and calibrated full inputs must already be recovered.
set -euo pipefail
export PATH=/opt/conda/bin:$PATH PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 V2D_OPT_HAND_SHAPE=0
export PYTHONPATH=/workspace/pr32/reconstruction/modules/v2d_issue6:${PYTHONPATH:-}
export V2D_ISSUE6_INPUT_ROOT=/workspace/issue6-final-inputs
export V2D_ISSUE6_RUN_ROOT=/workspace/issue6-full-runs/all20-final
export V2D_ISSUE6_REVIEW_ROOT=/workspace/pr32-evidence/all20-final-review
helpers=/workspace/pr32/reconstruction/validation/issue6
episodes=${1:-12,22,0,2,11,13,18,21,23,24,26,27,31,39,40,41,42,43,44,45}
# Keep the earlier quality controls and their manifests stable until completion.
if [[ ",$episodes," == *,12,* || ",$episodes," == *,22,* ]] && [ -f /workspace/pr32-evidence/quality-hands.pid ]; then
  control_pid=$(cat /workspace/pr32-evidence/quality-hands.pid)
  while kill -0 "$control_pid" 2>/dev/null; do sleep 5; done
fi
for episode in ${episodes//,/ }; do
  printf -v item 'e%03d' "$episode"
  root="/workspace/temporal-hands/$item"
  until /opt/conda/bin/python - "$episode" 2>/dev/null <<'PY'
import json,sys
from pathlib import Path
import pyarrow.parquet as pq
episode=int(sys.argv[1]); root=Path(f'/workspace/cad-final/e{episode:03d}/poses/e{episode:03d}')
names=json.loads(pq.ParquetFile(f'/workspace/track3/public/data/chunk-000/episode_{episode:06d}.parquet').schema_arrow.metadata[b'objects'])
if not all((root/f'{name}_fp_report.json').is_file() for name in names):
    raise RuntimeError('Required CAD receipts are missing')
PY
  do
    cad_pid=$(cat /workspace/pr32-evidence/remaining-cad.pid)
    if ! kill -0 "$cad_pid" 2>/dev/null; then echo 'CAD job ended before required object receipts'; exit 1; fi
    sleep 5
  done
  /workspace/hand-runtime/bin/python "$helpers/prepare_all_episode_manifest.py" --episodes "$episode"
  if [ ! -f "$root/fit_receipt.json" ]; then
    /workspace/hand-runtime/bin/python -u "$helpers/enhanced_hands.py" --episode "$episode" --use-saved-detections > "/workspace/pr32-evidence/$item-final-hand-predictions.log" 2>&1
    /workspace/hand-runtime/bin/python -u "$helpers/full_hand_masks.py" --episode "$episode" > "/workspace/pr32-evidence/$item-final-hand-masks.log" 2>&1
    /workspace/vipe-runtime/bin/python -u "$helpers/vipe_stereo_cameras.py" --episode "$episode" > "/workspace/pr32-evidence/$item-final-vipe.log" 2>&1
    /workspace/hand-runtime/bin/python -u "$helpers/run_temporal_fit.py" --episode "$episode" > "/workspace/pr32-evidence/$item-final-fit.log" 2>&1
  fi
  /opt/conda/bin/python - "$root/fit_receipt.json" <<'PY'
import json,sys
receipt=json.load(open(sys.argv[1]))
if not (receipt['fixed_shape_exactly_preserved'] and receipt['root_iterations']==50 and receipt['smooth_iterations']==300):
    raise RuntimeError('Expected a verified fixed-shape native fit')
PY
  /workspace/hand-runtime/bin/python -u "$helpers/convert_and_validate.py" --episode "$episode" > "/workspace/pr32-evidence/$item-final-pipeline.log" 2>&1
  /workspace/hand-runtime/bin/python -u "$helpers/probe_dynhamr_geometry.py" --episode "$episode" --all-frames --converted-records "$root/converted"
  /workspace/hand-runtime/bin/python -u "$helpers/measure_temporal_consistency.py" --episode "$episode"
  /workspace/hand-runtime/bin/python -u "$helpers/measure_aligned_consistency.py" --episode "$episode"
  /workspace/hand-runtime/bin/python -u "$helpers/make_review_media.py" --episode "$episode"
  /workspace/hand-runtime/bin/python -u "$helpers/review_motion_extremes.py" --episode "$episode"
  /workspace/hand-runtime/bin/python -u "$helpers/render_combined_overlay.py" --episode "$episode"
  /workspace/hand-runtime/bin/python -u "$helpers/audit_completed_contacts.py" --root "$V2D_ISSUE6_RUN_ROOT" --episodes "$episode" --direct-run-root --require-complete --output "$V2D_ISSUE6_REVIEW_ROOT/$item/contact-artifact-audit.json"
  /opt/conda/bin/python "$helpers/collect_all20_progress.py"
  echo "Final native episode $episode completed; visual acceptance still pending"
done
/workspace/hand-runtime/bin/python -u "$helpers/audit_completed_contacts.py" --root "$V2D_ISSUE6_RUN_ROOT" --episodes "$episodes" --direct-run-root --require-complete --output "$V2D_ISSUE6_REVIEW_ROOT/subset-$episodes-contact-audit.json"
echo "Requested native episodes $episodes and audits completed; visual acceptance still required"
