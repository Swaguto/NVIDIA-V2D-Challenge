#!/usr/bin/env bash
# Preserve each experiment. INPUT_ROOT contains the recovered workspace layout.
set -euo pipefail
exec </dev/null
: "${INPUT_ROOT:?Set INPUT_ROOT to the preserved input workspace}"
: "${OUT:?Set OUT to a new experiment directory}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PYTHON:-python}"
export PYTHONPATH="$REPO/reconstruction/modules/v2d_issue6:${PYTHONPATH:-}"
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
[[ ! -e "$OUT" ]] || { echo "Preserve existing experiment: $OUT" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
FLOW_ARGS=(--image-flow)
if [[ "${BIDIRECTIONAL_FLOW:-0}" == "1" ]]; then
    FLOW_ARGS+=(--bidirectional-image-flow)
fi
[[ $# -gt 0 ]] || set -- 12 40
for episode in "$@"; do
    [[ "$episode" =~ ^[0-9]+$ ]] || { echo "Invalid episode: $episode" >&2; exit 1; }
    label="$(printf 'e%03d' "$episode")"
    modules="$REPO/reconstruction/modules"
    runner="$REPO/reconstruction/validation/issue6/repair_hand_continuity.py"
    "$PY" "$runner" --input-root "$INPUT_ROOT" --episode "$episode" --modules "$modules" \
        --output "$OUT/image-prior/$label" --image-fit
    "$PY" "$runner" --input-root "$INPUT_ROOT" --episode "$episode" --modules "$modules" \
        --output "$OUT/surface-prior/$label" --prior-records "$OUT/image-prior/$label/align/records" \
        --smooth-seconds 0.08 --prior-weight 0.01
    "$PY" "$runner" --input-root "$INPUT_ROOT" --episode "$episode" --modules "$modules" \
        --output "$OUT/gap-recovery/$label" --prior-records "$OUT/surface-prior/$label/align/records" \
        --image-fit "${FLOW_ARGS[@]}"
    "$PY" "$runner" --input-root "$INPUT_ROOT" --episode "$episode" --modules "$modules" \
        --output "$OUT/runs/$label" --prior-records "$OUT/gap-recovery/$label/align/records" \
        --smooth-seconds 0.08 --prior-weight 0.01
    mano="$INPUT_ROOT/issue6-quality-cache/episodes/$label/local-assets/mano"
    "$PY" "$REPO/reconstruction/validation/issue6/render_combined_overlay.py" \
        --episode "$episode" --run-root "$OUT/runs" --review-root "$OUT/videos" \
        --manifest "$OUT/runs/$label/manifest.json" --modules-dir "$modules" \
        --mano-assets-root "$mano" --hands-mode continuous
    "$PY" "$REPO/reconstruction/validation/issue6/render_combined_overlay.py" \
        --episode "$episode" --run-root "$INPUT_ROOT/issue6-quality-cache/episodes/$label/issue6-full-runs/all20-final-v3" \
        --review-root "$OUT/baseline-videos" --manifest "$OUT/runs/$label/manifest.json" \
        --modules-dir "$modules" --mano-assets-root "$mano" --hands-mode metric
    fps="$("$PY" -c 'import json,sys; f=json.load(open(sys.argv[1]))["frames"]; print((len(f)-1)/(f[-1]["timestamp_s"]-f[0]["timestamp_s"]))' "$OUT/runs/$label/manifest.json")"
    "$PY" "$modules/v2d_hamer/lib/render_hands_aligned_video.py" \
        --frames_dir "$OUT/runs/$label/align/frames" --aligned_dir "$OUT/runs/$label/visual-records" \
        --mano_assets_root "$mano" --output_path "$OUT/videos/$label/hand-tracking-raw.mp4" --fps "$fps"
    "$PY" "$REPO/reconstruction/validation/issue6/label_hand_video.py" \
        "$OUT/videos/$label/hand-tracking-raw.mp4" "$OUT/videos/$label/hand-tracking.mp4"
done
