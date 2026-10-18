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

## Scored own-geometry reconstruction runner

`run_scored_reconstruction.py` runs a single scored episode through object masks,
visibility-based source selection, pinned SAM3D metric geometry, FoundationPose,
independent world conversion, native hand alignment/continuity, bundle export,
and support inference. It uses the prepared stereo observations and calibrated
hand fits. Its completion receipt is an export result, not a physical task pass.
The host requires Python 3.10+, Docker, an NVIDIA GPU, and the separately licensed
private model cache. Model weights and episode data are not baked into the images.

Create the committed code overlay with `docker/track3-scored/make_context.py`.
Verify the private model cache once before the batch with
`download_sam3d_models.py --root "$MODELS/sam3d" --verify-only`.
Use a source view containing only `videos` and `meta`: relative links must remain
inside their common mounted parent. The runner rejects views exposing object CAD,
URDFs or pose tables. The older general-purpose scored view includes diagnostic
assets and is intentionally rejected by this runner.

Example private configuration (replace directory paths, retain immutable images):

```json
{
  "episode": 3,
  "source_root": "/private/inputs",
  "data_subdir": "scored-video-view",
  "models": "/private/models",
  "observations": "/private/hand-inputs",
  "temporal_fits": "/private/temporal-hands",
  "cameras": "/private/perception",
  "overlay": "/private/committed-code-overlay",
  "roster": "/private/official-kit/data/track_3_evaluation_objects.json",
  "out": "/private/results/e003-run1",
  "max_seconds": 2400,
  "foreground_depth_only": false,
  "perception_image": "ghcr.io/swaguto/nvidia-v2d-challenge-dex3-review@sha256:6e2039d0a54e14a188a1798efe57c3465d002f063c782c827b791e4418d2dce6",
  "sam3d_image": "ghcr.io/swaguto/nvidia-v2d-challenge-dex3-review@sha256:dc9d0aedd2c6d50f53ce2d0677eca6f96e9631806d2bc12a7da4396629ce5357"
}
```

Run `python3 scripts/track3/run_scored_reconstruction.py --config "$CONFIG"`.
An optional `docker_host` selects an existing Docker daemon. Each run requires a
fresh output directory, bounded stages, read-only source inputs, source hashes,
and digest-pinned images. A surrounding operator must enforce VM shutdown and
verify result backups. Brev provisioning and billing controls are separate.
`foreground_depth_only` remains an experimental opt-in until source-video
validation passes; it does not relax pose acceptance gates.

### Isolated development validation

`make_public_validation_roster.py` reads only the public Parquet schema's object
names and probes the three source-video clocks. Keep those reference Parquets in
a separate directory **outside** the common inference mount. The generated JSON
contains names, clocks and hashes; pose values are not decoded or copied. It
records `method_frozen: false`: acquisition is preparation, not validation.

```sh
python scripts/track3/make_public_validation_roster.py \
  --source "$VIDEO_ONLY_VIEW" --metadata-source "$REFERENCE_ONLY_METADATA" \
  --episodes 11 22 42 --out "$PUBLIC_VIDEO_ROSTER"
python -m issue6.track3_adapter --episode 11 --data "$VIDEO_ONLY_VIEW" \
  --results "$INDEPENDENT_CAMERAS" --camera-root "$INDEPENDENT_CAMERAS" \
  --video-roster "$PUBLIC_VIDEO_ROSTER" --hands-only --output "$HAND_OBSERVATIONS"
```

This explicit public video mode uses the same fixed-camera, own-geometry and
bundle gates as scored inference. It retains `split: public`; it does not create
a fake scored kit or change source confidence. Freeze the method and training
budget before evaluating held-out episodes. These tests assess transfer of the
pipeline to new episodes; per-episode policies are not a universal zero-shot
policy, and recurring object categories are not unseen-category validation.
### Complete reconstructed episode to a short policy baseline

`docker/track3-scored/make_policy_context.py --out <fresh-context>` exports the
committed CPU preparation code. It excludes robot assets and private inputs.
`run_scored_policy.py --config <config.json>` then performs IK, prepares the
complete source frame axis, freezes collision geometry, compiles the native
scene on CPU, trains at most 100 PPO iterations, and runs the untouched official
kit evaluator on both initial and final checkpoints. A separate final native
readback verifies zero object assistance and records the task audit.

The JSON config requires `episode`, `bundle`, `support`, `context`,
`robot_assets`, `robot_asset_lock`, `kit`, `out`, `max_seconds`, `iterations`, and
an immutable `training_image` digest. The lock uses the existing
`v2d.dex3.robot-assets.v1` contract and its `files_sha256` closure. Real retrieved
robot meshes are required. `anchor_policy` defaults to `visual`; `metric` remains
an explicit experimental alternative. Keep the same choice across comparisons.

The runner uses `/opt/ik/bin/python` from the existing training image for IK and
case preparation. Portable preparation and PPO use that image's pinned training
environment. It records the preparation source commit separately from the
training image. No image download or rebuild happens inside this command.

Use a surrounding timed lease with verified result backup and provider shutdown.
Each episode needs a new output directory. A complete baseline can still fail
every grasp: inspect evaluator metrics, full native motion and the task audit.
These scores compare against reconstructed references and are not leaderboard
ground truth. The command does not upload or select a submission automatically.

### Start from videos, without saved hand fits

`run_scored_hands.py --config <config.json>` estimates independent stereo camera
motion, builds video/depth observations, exports calibrated optimizer cameras,
runs hand inference and masks, and fits the complete episode. The config uses
`episode`, `split` (`evaluation` or `public`), `source_root`, `data_subdir`,
`models`, `overlay`, `roster`, `out`, `max_seconds`, and `perception_image`.
Scored inference additionally requires `kit`; public validation requires the
explicit roster produced by `make_public_validation_roster.py`. An optional
`cameras` root reuses an existing independent camera sequence and records its
report hash. `docker_host` optionally selects a separate Docker daemon.

The entire `source_root` mount must contain only videos and calibration metadata,
including files outside `data_subdir`. Keep reference pose tables and provided
CAD elsewhere. The hand runtime weights and licensed MANO files remain separate
private mounts. The resulting `observations`, `temporal-hands`, and `perception`
directories supply the reconstruction runner's corresponding config paths.
All three runners need a surrounding timed compute lease and result backup.

For an offline native replay, run
`scripts/dex3_training/render_native_readback_fast.py --run <readback> --inputs <frozen-pack> --out <new-preview>`.
It retains every source knot at source speed, draws the actual recorded collision
surfaces with a fixed view, and warns when an object leaves that view. Triangle
visibility is approximate. It does not run new physics or render Isaac Sim.

Two separate optional tracking experiments are available in `batch_public.py`:
`--foreground-depth-only` removes non-object depth from refinement;
`--foreground-translation-prior` initializes a moving object's displacement from
the measured foreground surface median. The latter retains the reconstructed
body gauge and does not treat that median as the body origin. Both still use the
original measured-depth support and pose-jump acceptance checks. Use fresh result
roots and review source overlays before promoting either experiment. Neither is
enabled by default.

`--foreground-translation-fallback` retries a rejected normal track once using
that measured displacement. Successful normal frames do not use it. Both the
normal and recovery candidates pass the same original depth-support and
pose-jump checks. This conservative alternative also remains experimental.

Rostered video inference always uses automatic object prompts. Saved per-episode
box overrides are retained for legacy public CAD diagnostics only; a rostered run
rejects an annotated mask cache. Historically reviewed public episodes are useful
regression cases and must not be presented as untouched end-to-end holdouts.
