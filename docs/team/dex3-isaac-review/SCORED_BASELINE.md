# Scored Track 3 baseline work

## Scope and current evidence

The current official kit names 20 scored episodes and 36 episode/object pairs.
The 2026-10-17 acquisition is pinned to dataset revision
`5f68335f3acc802033d1e80728c1633197521de8`. All 60 evaluation camera videos
were downloaded, checked against upstream content hashes, and completely decoded.
Their frame counts match the kit's sample row IDs for every episode and object
slot. Public pose-label Parquets were excluded from this acquisition.

The shared released assets cover 26 pairs. The other ten need `wooden_spoon`,
which is still absent from this revision. Released CAD is recorded as released
CAD; acquisition does not establish participant reconstruction provenance or
submission eligibility. The [official FAQ](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf)
explicitly requires participants to reconstruct their own objects and use
reconstructed geometry to generate trajectories, including objects with public
meshes. All 36 pairs therefore need our own reconstruction; the ten spoon pairs
additionally lack a released diagnostic asset. The current kit requires a mesh for every object in
the combined submission, including for registration of the five pose metrics.

The first camera pilots are e001 (pot/lid), e010 (two cups), and e038 (one dust
pan). Their stereo camera trajectories cover 335, 161 and 263 frames respectively,
with no held camera frames under the existing PnP checks. Camera checks do not
establish correct object tracks, reconstructed geometry, retargeting or policies.
All 20 camera runs are now complete: 5,181 frames, with zero held frames under
the existing PnP checks. The [coverage receipt](scored-baseline-coverage-v1.json)
records per-episode counts, diagnostics and hashes. These checks are independent
of hidden trajectory accuracy.

## Reproduce the scored input contract

Use Python 3.12. The private working directory preserves a complete Hugging Face
tree inventory, the acquisition receipt, coverage audit and raw stage outputs.
No input footage or MANO weights are published with this code.

```bash
python scripts/track3/acquire_scored_inputs.py \
  --inventory /private/dataset-inventory.json \
  --roster /kit/data/track_3_evaluation_objects.json \
  --root /private/scored/dataset --receipt /private/scored/acquisition.json
python scripts/track3/prepare_scored_view.py \
  --root /private/scored/dataset --receipt /private/scored/acquisition.json \
  --out /private/scored/inputs
python scripts/track3/audit_scored_coverage.py \
  --kit /kit --data /private/scored/inputs --work /private/scored/perception \
  --out /private/scored/coverage.json
python scripts/track3/batch_public.py --phase cameras \
  --roster /kit/data/track_3_evaluation_objects.json \
  --data /private/scored/inputs --root /private/scored/perception \
  --episodes 1,10,38
```

The scored view has portable relative links. Preserve the dataset/view layout
and mount their common parent read-only in a container. It contains no pose
tables. The batch runner preserves official object-slot order, selects the
view's calibration, rejects requests outside the selected roster and exits
nonzero on failed stages. `--phase world` performs camera-to-world conversion
and explicitly labeled occlusion inference without invoking a ground-truth
scorer. `--roster ... --phase scores` is rejected because scored ground truth
is unavailable. The official policy evaluator and complete packer remain
separate final gates.

### Fresh hand inputs from scored video

The existing Issue 6 stereo adapter now accepts `--scored-kit` and obtains
object ordering and frame coverage from the kit. It derives timestamps from
the actual video clock and checks both decoded stereo timestamps. It reads no
source pose tables. Only complete original scored episodes are accepted.
`--camera-root` attaches the independently inferred stereo cameras. The manifest
binds raw video/calibration/kit/adapter hashes and uses portable relative paths
for source observations and geometry. Missing object observations remain invalid.

```bash
PYTHONPATH=reconstruction/modules/v2d_issue6 python -m issue6.track3_adapter \
  --episode 1 --data /private/scored/inputs --results /private/scored/perception \
  --output /private/scored/hand-inputs/e001 --scored-kit /kit \
  --camera-root /private/scored/perception
python scripts/track3/export_optimizer_cameras.py \
  --manifest /private/scored/hand-inputs/e001/manifest.json \
  --out /private/scored/optimizer-cameras/e001
```

This prepares observations for reconstruction; it does not claim completed hand
inference. The camera adapter preserves the world-to-camera convention and
explicit observation-validity sidecar. It does not run ViPE. The new
`docker/track3-scored/` code-only runtime reuses the pinned FoundationPose image
and original pinned HaMeR/Dyn-HaMR/SAM2 preparation dependencies. Model weights,
licensed MANO files and episode inputs remain separate private mounts. Build and
GPU inference validation are recorded separately; no fresh end-to-end success
is established by its Dockerfile.

### Geometry path without released CAD

`reconstruct_stereo_object.py --stage seed` makes an explicitly incomplete
convex seed from a source object mask and calibrated stereo measurements. It
does not read released object meshes. `reconstructed_view.py --allow-seed`
binds that seed to one episode and the official object-slot order for initial
tracking. Scored `--phase poses` rejects the released-asset view and any changed
mesh/cache binding. Every new tracked pose records its actual mesh digest.

After initial tracking, `--stage refine` integrates observed masked depth views
with TSDF in the same fixed object body frame. It requires poses generated from
that exact seed, preserves the body origin and records the used observations.
Inspect and retrack refined geometry before creating final robot bundles. A
depth seed, TSDF output or passing format check alone does not establish geometry
quality, correct collision behavior or submission readiness.

The first GPU perception pilot exposed incomplete handle coverage and an
inherited Open3D 0.20.0 scalable TSDF that returned empty geometry. An initial
pot/lid identity-swap diagnosis was incorrect: the preview's colour legend was
reversed. The original detector's seed coordinates and mask directories assigned
the pot and lid correctly. The empty-geometry failure also reproduces with an
isolated 0.5-metre synthetic plane, independently of episode poses, matching
[Open3D #7592](https://github.com/isl-org/Open3D/issues/7592). The repaired image
pins Open3D 0.19.0 and must pass `check_geometry_runtime.py` before reconstruction.
Its incremental `Dockerfile.geometry` reuses the already compiled GPU libraries.
Scored semantic seeding now uses the calibrated colour camera projected through
measured stereo depth, with distinct saucepan/lid prompts including the long
handle. Colour is needed to distinguish the blue/beige cup names. Colour-only depth
hole filling changes appearance for seeding; metric observations remain unchanged.
The stereo propagation remains at the original frame rate. Identity and geometry
quality still require actual pilot inspection; this code change alone is not a pass.

The scored bundle exporter requires refined own-geometry and complete world
targets bound to its mesh hashes. Conversion and occlusion inference preserve
those bindings. Inferred frames keep false observation masks and explicit methods;
gaps cannot silently become identity-pose targets. All camera estimates are checked
against the fixed native hand-fit cameras. The legacy development exporter retains
its separate interpretation. Input footage and MANO remain private mounts.

The next tracking revision initializes from the measured seed-frame transform
that defines the reconstructed body frame. It visits every decoded frame once,
forwards and backwards from that anchor, without re-encoding cached pixels.
FoundationPose's internal mesh-centering transform is accounted for explicitly.
This avoids another arbitrary initial orientation search for our partial seeds.
Every accepted frame must have at least 200 masked stereo points, with at least
half inside the fixed seed envelope (20 mm or 10% extent margin). Fusion uses
the same bound, records rejected views, and does not claim unseen-surface
completion. These gates reject bad evidence; they do not establish physical
acceptance. GPU validation of this anchored revision is a separate run from the
RGB/pinned-runtime pilot.

The same FAQ explicitly permits floating hands and fixed-base arm/hand setups
for Track 3. Our selected policy runtime remains the official pinned
FlashCHORD/Newton/JAX path with floating Dex3 hands. The Isaac renderer remains
a separate review runtime; a whole-body G1 port is not required by that FAQ.

### General full-episode retargeted training references

`scripts/dex3_training/prepare_window.py --full-episode` keeps the complete
retargeted command sequence instead of selecting a dual-hand confidence window.
It checks that all original frames and finite, normalized inferred robot poses
are present. A one-hand task may have no frame where both human hands were
directly observed; that does not promote any source confidence to observed.
Missing/nonfinite robot or object targets are rejected and require explicit
upstream inference. The original full source masks are saved separately.

This path writes `v2d.dex3-training-retargeted-task.v4` with
`inferred_retargeted_robot_commands.v1`. It does not require a phase plan assigning
one object per hand. The native reference loader accepts it with the ordinary
residual action and tracking objective; the grasp-feedforward action retains its
separate v3 contract. Contact labels remain unknown and their rewards stay
disabled in the generic baseline. This adapter has CPU contract checks; complete
scored runtime training and physical validation are still pending.

## Short controlled training forks

The retained e002 checkpoint is a development control outside the scored roster.
The preparation tool creates three 100-iteration trials (819,200 new environment
steps each at its saved 256-world, 32-step configuration):

1. Same recipe and complete learner restoration.
2. Only contact-wrench objective weight 10 → 5, including resolved curriculum aliases.
3. Only training reset stabilization 20 → 5 control steps.

All three restore the same actor, critic, optimizer, exploration state, RNG,
normalization and counters. These are explicitly labeled controlled forks;
they do not restore the prior simulator trajectory. Inputs, collision geometry,
action/observation dimensions, optimizer settings, curriculum reset probabilities
and final evaluation composition are unchanged. A separate relative-object reward
experiment follows these tests; it is not included in them.

```bash
python scripts/dex3_training/prepare_controlled_experiments.py \
  --checkpoint /private/parent/ppo/model_2000.safetensors \
  --out /private/experiments --iterations 100
```

Use the new committed extension image, the original input manifest and parent
build context, and a fresh output directory per trial. Mount generated configs
read-only, then run `train.py --config-path /experiment-configs --config-name control`
(or `half-cws`, `short-settling`). Restore the parent at `/initial-policy` and
its context at `/initial-context`. No longer continuation is authorized by a
rise in training reward alone.

### First controlled trial results

All three 100-iteration trials completed with no reported numerical failures:
2,457,600 new environment steps in total. The image digest is
`ghcr.io/swaguto/nvidia-v2d-challenge-dex3-review@sha256:f73bfdcfec5244757633fbdf24a86e4ff15ebe37ee41f7de8a36ebe69c19984f`,
with entry-point source `68d7d8ad5c1c5c35ee6639e5d7aab85390b44aad` and original
runtime payload source `31c0d89282e157f413d3dc6a0286734b19f474ea`.

Each saved policy received three full, unassisted eight-world readbacks:

| Policy | Mean AUC | Mean MP-SR | Mean RPE, cm |
| --- | ---: | ---: | ---: |
| Retained parent | 0.86549 | 0.87500 | 5.0241 |
| Unchanged 100-iteration continuation | 0.85844 | 0.87500 | 4.9866 |
| Contact-wrench reward halved | 0.85690 | 0.83333 | 5.1323 |
| Reset settling shortened to 5 steps | 0.87444 | 0.95833 | 4.6377 |

These are e002 inferred-reference diagnostics, not scored-episode or leaderboard
results. The half-contact candidate regresses. Shorter settling is promising,
but its 0.00895 AUC gain is below the observed parent/control AUC range of
0.01995. Neither treatment qualifies for a longer run under the empirical
variability guard. No candidate passes all physical grasp checks. Keep the
retained parent selected while independently confirming the reset result and
building scored pilots. Three readbacks of one trained policy are not three
independent training seeds.

[Complete repeated report](controlled-training-results-v1.json). All 166 output
files, including checkpoints and failed grasp evidence, were hash-verified on
the producing Mac. Brev was confirmed STOPPED after the 1,000.4-second session.
This duration is not a verified invoice charge.

## Evaluation variability and compute

Two new four-world, unassisted readbacks of the same archived checkpoint in its
original immutable image produced AUC 0.86284 / 0.84516 and MP-SR 0.75 / 0.50.
No optimizer steps occurred. Initial state differences were approximately
1.8e-7 in pose components; later trajectories diverged. The cause is not yet
isolated. This is evidence that one four-world score is insufficient to qualify
a small improvement. Final candidate screening needs repeated readbacks and a
matched control in the same runtime; local values remain distinct from hidden
ground-truth leaderboard scores.

The readbacks took 54.6 / 40.4 seconds. This does not isolate a cold cache or
prove billed savings. All 19 output files were hash-verified locally and the
provider confirmed STOPPED after the 186.6-second lifecycle. Subsequent GPU
work requires an explicit authorization receipt and a timed backup/stop wrapper.
Historical billing remains unreconciled; do not relabel local runtime estimates
as verified account usage or an enforced spending limit.

## Remaining acceptance gates

- Valid object geometry and trajectories, hand reconstruction, complete training
  bundles and usable grasps for the three scored pilots, then the whole roster.
- Repeated unassisted policy evaluations and rejection of regressed candidates.
- Mesh/pose body-frame consistency, complete actual rollout frame axes and the
  untouched official combined packer; captain approval before any upload.
- Held-out development checks with frozen tuning choices.
- A new immutable release image, private input manifest, fresh teammate
  reproduction and report. Current workflow images do not yet provide a complete
  fresh video-to-policy implementation.
