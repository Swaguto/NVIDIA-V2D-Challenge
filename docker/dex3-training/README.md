# Dex3 development training

This is a short unassisted training path for two floating Dex3 hands. It uses the
submission kit's FlashCHORD runtime at
`8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`, Newton/MuJoCo-Warp physics,
JAX PPO, and upstream safetensors checkpoint/evaluation APIs. The existing
Isaac Sim image remains the separate physics diagnostic runtime.

## Scope

- Seven real finger joints and three digits per hand; no Sharpa contact mapping.
- Gravity, hand/object collision, object/object collision, finite inferred support.
- No virtual object assistance, settling assistance, or external service logging.
- Select the longest consecutive source interval where both hands and all objects
  are observed. Preserve excluded rows and confidence masks in a sidecar. This
  conservative pilot does not train through missing or uncertain source intervals.
- Human observations and unknown contact/wrench labels are preserved. Label-based
  contact rewards fail closed. Robot targets are inferred, not ground truth.
- Explicit upstream dispatch extensions load dual-hand data and the finite support
  triangles. Their original file hashes must match the pinned runtime before patching.
- A checkpoint binds the private target/asset manifest. The installed extension is
  required for evaluation; an unmodified stock runtime cannot load this schema.

These are development diagnostics. A learned checkpoint alone does not establish
grasp acquisition, full-episode success, hidden-set performance, or submission acceptance.

## Build

Use Linux amd64, an NVIDIA GPU, Container Toolkit and driver 580 or newer. The
initial recipe uses 64 worlds on a 48 GB L40S. The build uses a separate Python
3.11.15 environment and the upstream `uv.lock`; Isaac's Python remains separate.

Prepare a Git checkout at the exact upstream commit above, with the sparse path
`robotic_grounding/flash_chord`. Upstream LFS assets are unnecessary for this
extension: its robot/object inputs are mounted privately.

From this repository, with the training source committed:

```bash
python docker/dex3-training/make_context.py \
  --upstream /path/to/pinned/video_to_data \
  --out /path/to/new/build-context
docker build --build-arg VCS_REF="$(git rev-parse HEAD)" \
  -t v2d-dex3-training:local /path/to/new/build-context
```

The context is generated from committed upstream Git objects and an allowlist of
our extension source. It excludes private episode inputs, MANO weights and credentials.
The image records both source revisions and rejects a dirty/mismatched build label.

## Private inputs

Keep the input pack local or transfer it to a teammate through an approved private
channel. It is not inside the image or a GitHub release.

`scripts/dex3_training/prepare_window.py` takes a verified frozen case, its prepared
static-load inputs, and the private robot assets. It verifies source checksums,
retains a contiguous observed interval, converts the inferred support normal to
stage up, preserves the finite support footprint, and writes portable relative
robot mesh paths. Object mass/inertia and support orientation remain nominal/inferred.

```bash
python scripts/dex3_training/prepare_window.py \
  --case /private/frozen-case \
  --static-inputs /private/static-load-inputs \
  --robot-assets /private/portable-inputs/robot-assets \
  --out /private/new-pilot
```

This preparation command can also run in the training image at
`/opt/dex3-extension/prepare_window.py`, with the three inputs mounted read-only
and the output directory mounted writable. It requires no GPU.

## Train and evaluate

```bash
export DEX3_TRAINING_INPUTS=/private/new-pilot
export DEX3_TRAINING_OUTPUTS=/private/new-results
mkdir -p "$DEX3_TRAINING_OUTPUTS"
docker compose -f docker/dex3-training/compose.yaml run --rm pilot
```

The default pilot saves `model_0.safetensors` before learning, then checkpoints at
25, 50, 75 and 100 iterations. Local outputs include every iteration's metrics,
definitions, scene physics arrays and the resolved checkpoint configuration.

Evaluate the initial and learned checkpoints with the same unmodified upstream
evaluator entry point, inside the extension image:

```bash
docker compose -f docker/dex3-training/compose.yaml run --rm pilot \
  /opt/flash-chord/scripts/evaluate_policy.py \
  evaluation.checkpoint=/outputs/ppo/model_100.safetensors \
  evaluation.world_count=4 \
  evaluation.metrics_output=/outputs/eval-100.json \
  evaluation.object_trajectories_output=/outputs/eval-100.parquet
```

Repeat for `model_0` with fresh output names. The evaluator uses explicit resets,
VOC zero and schema checks. Its metrics cover the selected development window;
they are not a score for the official evaluation roster.

Installed runtime contract controls:

```bash
docker compose -f docker/dex3-training/compose.yaml run --rm pilot \
  /opt/dex3-extension/runtime_controls.py \
  --reference /data/reference.parquet --out /outputs/runtime-controls.json
```

## Grasp geometry diagnostics

`scripts/dex3_physics/antipodal_seed.py` searches opposing contact surfaces in
actual Isaac cooked collision geometry. It changes only experimental robot
targets, caps wrist and finger motion, checks FK against SDK link poses, and
rejects large object penetrations. This is a proposal generator, not force closure
or a grasp-success certificate. Use `run_container.py --stage grasp-seed
--grasp-seed /path/to/proposal.json` to run its separate static gravity diagnostic.
The unchanged left-hand samples remain controls; the changed seeds have their
own retention result. Full movement and physical acquisition still need validation.

`--stage grasp-acquisition --grasp-seed /path/to/proposal.json` adds a distinct
paired test for the two selected grasps. Both baseline and candidate begin with
the same source finger angles. The candidate closes smoothly for 0.25 seconds,
settles until 0.5 seconds, then all temporary tray collisions and visuals are
removed. Retention is measured over the following unsupported second, with
collision-disable readbacks and independent gravity/contact controls. The tray
is a diagnostic fixture and must not be substituted for the evaluation scene.
