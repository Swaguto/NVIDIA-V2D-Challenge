# Track 3 public CAD tracking

For the scored episode input contract and current end-to-end baseline work, see
[scored baseline](../../docs/team/dex3-isaac-review/SCORED_BASELINE.md).
Use `--roster /kit/data/track_3_evaluation_objects.json` to process scored video
inputs without public pose tables. Scored world conversion is a separate
`--phase world`; hidden ground-truth scoring is unavailable locally.

This pipeline tracks the public CAD objects in rectified left camera video and
estimates the camera trajectory independently from stereo video. It uses the
reference object poses only in the final scoring step.

## Stages

1. `stereo_odometry.py` estimates a metric camera trajectory using stereo depth,
   KLT correspondences, and PnP RANSAC. Each camera pose records whether it was
   observed or inferred.
2. `temporal_masks.py` finds a box for each named object with Grounding DINO and
   propagates masks in both directions with SAM2 video memory. Pot masks that
   switch to the lid are suppressed and reported as missing observations.
3. `foundationpose_track.py` fits and tracks each public GLB CAD model against
   rectified RGB and stereo depth. It writes camera frame object poses with
   per-frame observation provenance.
4. `convert_camera_poses.py` converts object poses to the independent world
   frame. `infer_occlusion.py` fills gaps with separately marked inferred poses.
5. `score_track3_official.py` writes reconstruction diagnostic scores for both
   observed and inferred pose sets. `render_overlay.py` draws CAD outlines on
   the exact corresponding video frames. `render_cad_overlay.py` renders the
   textured CAD surfaces with a depth test against stereo video.

On the prepared GPU host, run:

```bash
python scripts/track3/batch_public.py --phase cameras --root /path/to/results
python scripts/track3/batch_public.py --phase masks --root /path/to/results
python scripts/track3/orchestrate_public.py --root /path/to/results
python scripts/track3/report_public.py --data /path/to/public --root /path/to/results
```

The batch runner accepts `--episodes 0,2` for a subset. The prepared host needs
the video and CAD data, the stereo calibration, the Grounding DINO and SAM2
weights, and a FoundationPose environment. The `--fp-python` option selects
that environment; `--data` selects the public dataset.

`seed_overrides/eNNN.json` stores box prompts chosen from public video when
Grounding DINO misses an object or assigns the same mask to two objects. Each
entry has a frame index and an `xyxy` box in rectified left camera pixels.
The override is used only by the mask stage. No reference object pose is used.

To render a textured CAD video after object tracking:

```bash
python scripts/track3/render_cad_overlay.py \
  --episode 0 --data /path/to/public \
  --predictions /path/to/results/e000/poses \
  --out /path/to/results/e000/cad_mesh_overlay.mp4
```

The report has one row and one overlay for each episode. A missing score is
shown as pending, never as zero. AUC, MP-SR, and SP-SR are reconstruction
diagnostics under one constant alignment per episode. They do not measure
robot policy rollout success.
