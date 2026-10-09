#!/usr/bin/env bash
# Issue #7: joint Gaussian-splat refinement of one Track 3 episode, end to end.
#
#   1. gsplat_inputs.py      Track 3 + issue-6 outputs -> refinement input layout
#   2. refine_simple         photometric + segmentation + metric stereo depth +
#                            world-frame temporal smoothness; objects, hands and
#                            camera optimized jointly
#   3. refinement_report.py  jitter / drift / contact report, 3D viewer, and the
#                            official diagnostic score before vs after
#
# Usage:
#   EPISODE=2 DATA=... RESULTS=... BUNDLE=... HANDS_RUN=... MANO_ROOT=... OUT=... \
#     scripts/track3/refine_episode.sh
#
# Required:  EPISODE DATA RESULTS BUNDLE OUT, plus documented SLAM camera poses
#            in RESULTS or SLAM_CAMERA_DIR (camera_to_world in ego_cam_c).
#            RUN_SLAM=1 runs local DROID-SLAM first; requires DROID_WEIGHTS and
#            DROID_REPO, with its CUDA dependencies available to SLAM_PYTHON.
# Optional:  HANDS_RUN (issue-6 run dir; omit to refine objects + camera only)
#            MANO_ROOT (dir with models/MANO_RIGHT.pkl; required with HANDS_RUN)
#            REFINE_MODE=docker|native (default docker; native runs
#              $REFINE_PYTHON -m v2d.gsplat_refinement.lib.refine_simple in an
#              env with gsplat>=1.4 + manotorch, e.g. a RunPod pod without docker)
#            SCORE=0 to skip the official before/after diagnostic score
#            START / FRAMES (sub-window), EPOCHS, W_METRIC_DEPTH, W_SMOOTH,
#            LR_POSE, LR_CAMERA, W_MASK, REST_ANNOTATIONS, CONTACT_TRUTH,
#            W_CAMERA_ACCELERATION, EXTRA_REFINE_ARGS
#            REQUIRE_SLAM=0 permits an explicitly diagnostic odometry pilot.
# Defaults reproduce the bounded episode-2 pilot; see docs/team/issue7/RESULTS.md.
# Pose limits are initialization-relative engineering priors, not accuracy bounds.
set -euo pipefail

: "${EPISODE:?}" "${DATA:?}" "${RESULTS:?}" "${BUNDLE:?}" "${OUT:?}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-python}"
REFINE_MODE="${REFINE_MODE:-docker}"
REFINE_PYTHON="${REFINE_PYTHON:-$PY}"
EPOCHS="${EPOCHS:-10}"
W_METRIC_DEPTH="${W_METRIC_DEPTH:-100}"
W_SMOOTH="${W_SMOOTH:-1000}"
LR_POSE="${LR_POSE:-5e-5}"
LR_CAMERA="${LR_CAMERA:-1e-5}"
W_MASK="${W_MASK:-50}"
W_CAMERA_ACCELERATION="${W_CAMERA_ACCELERATION:-1000000}"

IN="$OUT/inputs"
REF="$OUT/refined"
REPORT="$OUT/report"
if [[ -e "$OUT" || -L "$OUT" ]]; then
  echo "Preserve the existing experiment; choose a new OUT: $OUT" >&2
  exit 1
fi
# Reserve the experiment before any stage runs; mkdir also rejects concurrent reuse.
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"

window=()
[[ -n "${START:-}" ]] && window+=(--start "$START")
[[ -n "${FRAMES:-}" ]] && window+=(--frames "$FRAMES")
hands_in=()
[[ -n "${HANDS_RUN:-}" ]] && hands_in+=(--hands-run "$HANDS_RUN")
slam_in=()
if [[ "${RUN_SLAM:-0}" == 1 ]]; then
  : "${DROID_WEIGHTS:?}" "${DROID_REPO:?}"
  SLAM_CAMERA_DIR="$OUT/slam/camera_c2w"
  "$PY" "$REPO/scripts/track3/gsplat_inputs.py" --episode "$EPISODE" --data "$DATA" \
    --results "$RESULTS" --bundle "$BUNDLE" --out "$OUT/slam-inputs" \
    ${hands_in[@]+"${hands_in[@]}"} ${window[@]+"${window[@]}"}
  "${SLAM_PYTHON:-$REFINE_PYTHON}" "$REPO/scripts/track3/slam_camera.py" \
    --inputs "$OUT/slam-inputs" --weights "$DROID_WEIGHTS" --droid-repo "$DROID_REPO" --out "$OUT/slam"
fi
[[ -n "${SLAM_CAMERA_DIR:-}" ]] && slam_in+=(--slam-camera-dir "$SLAM_CAMERA_DIR")
[[ "${REQUIRE_SLAM:-1}" != 0 ]] && slam_in+=(--require-slam)

echo "== 1/3 assembling inputs -> $IN"
"$PY" "$REPO/scripts/track3/gsplat_inputs.py" --episode "$EPISODE" --data "$DATA" \
  --results "$RESULTS" --bundle "$BUNDLE" --out "$IN" ${hands_in[@]+"${hands_in[@]}"} \
  ${window[@]+"${window[@]}"} ${slam_in[@]+"${slam_in[@]}"}
FPS="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['fps'] or 20)" "$IN/inputs.json")"

args=(
  --frames_dir "$IN/frames" --depth_dir "$IN/depth" --intrinsics_path "$IN/intrinsics.json"
  --objects_root "$IN/objects" --refined_objects_root "$REF/objects" --missing_object_mask_unknown
  --background_pose_init_dir "$IN/camera_c2w" --refined_camera_poses_dir "$REF/camera"
  --overlay_path "$REF/overlay.mp4" --progress_dir "$REF/progress" --scene_export_path "$REF/scene.npz"
  --n_epochs "$EPOCHS" --batch_size 4 --train_resolution_scale 0.5
  --lr_gaussians 1e-3 --lr_object_pose "$LR_POSE" --lr_object_scale 0
  --lr_hand_pose "$LR_POSE" --lr_hand_articulation "$LR_POSE" --lr_hand_shape 0 --lr_hand_scale 0
  --lr_camera_pose "$LR_CAMERA" --lr_schedule cosine --lr_cosine_min_factor 0.1
  --init_opacity_obj 0.5 --init_opacity_hand 0.5 --init_opacity_bg 0.5
  --w_mask "$W_MASK" --w_relative_depth 1.0 --w_metric_depth "$W_METRIC_DEPTH"
  --max_object_translation_change "${OBJECT_TRANSLATION_LIMIT:-0.003}"
  --max_object_rotation_change_deg "${OBJECT_ROTATION_LIMIT_DEG:-0.5}"
  --max_hand_translation_change "${HAND_TRANSLATION_LIMIT:-0.002}"
  --max_hand_rotation_change_deg "${HAND_ROTATION_LIMIT_DEG:-0.5}"
  --max_hand_articulation_change_deg "${FINGER_ROTATION_LIMIT_DEG:-1}"
  --max_camera_translation_change "${CAMERA_TRANSLATION_LIMIT:-0.002}"
  --max_camera_rotation_change_deg "${CAMERA_ROTATION_LIMIT_DEG:-0.25}"
  --smooth_in_world
  --w_smooth_object_rot "$W_SMOOTH" --w_smooth_object_trans "$W_SMOOTH"
  --w_smooth_hand_rot "$W_SMOOTH" --w_smooth_hand_articulation "$W_SMOOTH" --w_smooth_hand_trans "$W_SMOOTH"
  --w_smooth_camera_rot "$W_SMOOTH" --w_smooth_camera_trans "$W_SMOOTH"
  --w_camera_acceleration_trans "$W_CAMERA_ACCELERATION" --w_camera_acceleration_rot "$W_CAMERA_ACCELERATION"
  --render_every "${RENDER_EVERY:-100}" --fps "$FPS" --seed "${SEED:-42}"
)
for side in left right; do
  if [[ -d "$IN/hands/$side/poses" ]]; then
    : "${MANO_ROOT:?MANO_ROOT (containing models/MANO_RIGHT.pkl) is required to refine hands}"
    args+=(--"${side}_hand_pose_dir" "$IN/hands/$side/poses" --"${side}_hand_mask_dir" "$IN/hands/$side/masks"
           --refined_"${side}"_hand_pose_dir "$REF/hands/$side")
  fi
done
[[ -n "${MANO_ROOT:-}" ]] && args+=(--mano_assets_root "$MANO_ROOT")
# shellcheck disable=SC2206
args+=(${EXTRA_REFINE_ARGS:-})

echo "== 2/3 refining ($REFINE_MODE) -> $REF"
mkdir -p "$REF"
printf '%q ' "${args[@]}" > "$REF/command.txt"
if [[ "$REFINE_MODE" == docker ]]; then
  # Host needs: pip install -e reconstruction/modules/v2d_docker \
  #   -e reconstruction/modules/v2d_gsplat_refinement/docker, and the
  # v2d_gsplat_refinement image (python -m v2d.gsplat_refinement.docker.build).
  "$PY" -m v2d.gsplat_refinement.docker.run_refine_simple "${args[@]}" 2>&1 | tee "$REF/refine.log"
else
  "$REFINE_PYTHON" -m v2d.gsplat_refinement.lib.refine_simple "${args[@]}" 2>&1 | tee "$REF/refine.log"
fi

echo "== 3/3 report -> $REPORT"
contacts=()
score=()
rest=()
truth=()
# SCORE=0 skips the official scorer (it needs the public parquet reference).
[[ "${SCORE:-1}" != 0 ]] && score+=(--data "$DATA")
[[ -n "${HANDS_RUN:-}" && -f "$HANDS_RUN/contacts/contacts.json" ]] && contacts+=(--contacts "$HANDS_RUN/contacts/contacts.json")
[[ -n "${REST_ANNOTATIONS:-}" ]] && rest+=(--rest-annotations "$REST_ANNOTATIONS")
[[ -n "${CONTACT_TRUTH:-}" ]] && truth+=(--contact-truth "$CONTACT_TRUTH")
"$PY" "$REPO/scripts/track3/refinement_report.py" --scene "$REF/scene.npz" --episode "$EPISODE" \
  --out "$REPORT" --inputs "$IN/inputs.json" ${score[@]+"${score[@]}"} ${contacts[@]+"${contacts[@]}"} \
  ${rest[@]+"${rest[@]}"} ${truth[@]+"${truth[@]}"}
echo "Viewer: $REPORT/viewer.html"
