#!/usr/bin/env bash
# Consume completed immutable fits; preserve an already running first pipeline.
set -euo pipefail
export PATH=/opt/conda/bin:$PATH PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 V2D_OPT_HAND_SHAPE=0
export PYTHONPATH=/workspace/pr32/reconstruction/modules/v2d_issue6:${PYTHONPATH:-}
export V2D_ISSUE6_INPUT_ROOT=/workspace/issue6-final-inputs
export V2D_ISSUE6_RUN_ROOT=/workspace/issue6-full-runs/all20-final
export V2D_ISSUE6_REVIEW_ROOT=/workspace/pr32-evidence/all20-final-review
helpers=/workspace/pr32/reconstruction/validation/issue6
episodes=${1:?Comma-separated episode IDs required}
for episode in ${episodes//,/ }; do
  printf -v item 'e%03d' "$episode"
  root="/workspace/temporal-hands/$item"
  until /opt/conda/bin/python - "$root/fit_receipt.json" 2>/dev/null <<'PY'
import json,sys
receipt=json.load(open(sys.argv[1]))
assert receipt['fixed_shape_exactly_preserved'] and receipt['root_iterations']==50 and receipt['smooth_iterations']==300
PY
  do
    kill -0 "$(cat /workspace/pr32-evidence/all20-hand-preparation.pid)" 2>/dev/null || exit 1
    sleep 5
  done
  if [ ! -f "$V2D_ISSUE6_RUN_ROOT/$item/report/receipt.json" ]; then
    /workspace/hand-runtime/bin/python -u "$helpers/convert_and_validate.py" --episode "$episode" > "/workspace/pr32-evidence/$item-final-pipeline.log" 2>&1
  fi
  /opt/conda/bin/python "$helpers/audit_stage_receipts.py" --root "$V2D_ISSUE6_RUN_ROOT" --episodes "$episode" --modules-dir /workspace/pr32/reconstruction/modules --require-complete --output "$V2D_ISSUE6_REVIEW_ROOT/$item/stage-integrity-audit.json"
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
