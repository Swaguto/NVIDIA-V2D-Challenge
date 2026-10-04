#!/bin/bash
# Render shaded CAD overlays for every public episode that has world poses.
# Skips episodes whose output already exists, so it is safe to re-run.
set -u
REPO=/home/ubuntu/pr30
R=/home/ubuntu/track3_all_20261004
export PYTHONPATH="/home/ubuntu/p3d_shim:/home/ubuntu/FoundationPose:${PYTHONPATH:-}"
export V2D_RUNTIME="$REPO/reconstruction/modules/v2d_stereo_depth/lib"
SKIP="${1:-}"

cd "$REPO"
for EP in 0 2 11 12 13 18 21 22 23 24 26 27 31 39 40 41 42 43 44 45; do
  E=$(printf %03d "$EP")
  OUT="$R/e$E/cad_mesh_overlay_color.mp4"
  if [ "$E" = "$SKIP" ]; then echo "skip e$E (held back)"; continue; fi
  if [ -s "$OUT" ]; then echo "e$E cached ($(stat -c%s "$OUT") bytes)"; continue; fi
  if [ ! -d "$R/e$E/poses/e$E/ego_object_poses/e$E" ]; then
    echo "e$E NO WORLD POSES"; continue
  fi
  echo "=== rendering e$E ==="
  /home/ubuntu/v2d_venv/bin/python scripts/track3/render_cad_overlay.py \
    --episode "$EP" \
    --data /home/ubuntu/v2d_track3/data/hf/track_3/public \
    --predictions "$R/e$E/poses" \
    --out "$OUT" --natural-material \
    > "$R/e$E/logs/cad.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "e$E ok $(stat -c%s "$OUT") bytes"
  else
    echo "e$E FAILED rc=$rc"
    tr -cd '\11\12\15\40-\176' < "$R/e$E/logs/cad.log" | tail -6
  fi
done
echo "=== cad pass complete ==="
