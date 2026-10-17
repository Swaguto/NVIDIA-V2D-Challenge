# Dex3 development training

For the latest complete-state contact-guided continuation, use
[the exact PPO resume recipe](../../docs/team/dex3-isaac-review/EXACT_PPO_RESUME.md).
The actor-only continuation and retention-weight-8 experiments regressed; their
recipes remain available as recorded experiments. Full physical grasp is unresolved.

For measured-contact force observations, lift/load rewards and the staged reset
experiment, see [grip learning](../../docs/team/dex3-isaac-review/GRIP_LEARNING.md).
`compose.grip.yaml` defaults to the GPU reward/observation contract check; training
is an explicit command. Complete physical task acceptance is still required.

For the newer **343-frame full-task experiment**, measured initial/trained
results, corrected force-plan clock and local reproduction pack, start with
[the ordered fixes checkpoint](../../docs/team/dex3-isaac-review/ORDERED_FIXES.md).
The 75-frame experiment below is retained as historical evidence. Neither
experiment has achieved successful complete-task retention.

For the latest failed multi-contact diagnostic, use `compose.current.yaml` and
`registry.experimental.env.example` with the local corrected-clock v8 inputs.
It runs four diagnostic worlds by default. Archived PPO checkpoints require
their matching image and inputs; see the ordered-fixes guide for both paths.

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
- Collision hulls are frozen into the private pack and hash-checked before every
  load. CoACD can generate different hulls across processes; an unbound cache is
  refused during training/evaluation.

These are development diagnostics. A learned checkpoint alone does not establish
grasp acquisition, full-episode success, hidden-set performance, or submission acceptance.

## Tested image and current result

The private GHCR image is pinned in `registry.env.example`. Team members need
repository/package read access and the local frozen input pack. Authenticate to
GHCR, then load the image reference below. The image contains no episode pack.

```bash
set -a
. docker/dex3-training/registry.env.example
set +a
docker pull "$DEX3_TRAINING_IMAGE"
```

The tested 1,000-iteration e002 experiment completed 1,536,000 environment steps.
The unchanged upstream evaluator loaded both checkpoints inside this extension
image, using identical frozen collision geometry. Both achieved **0% success**
on the 75-frame development window. The training path runs, but the learned
controller has not solved retention. See
`docs/team/dex3-isaac-review/TRAINING_VALIDATION.md` for measured results.
The later Isaac grip-controller correction passes two lid acquisition/lift
diagnostics; its source overlay and private input recipe are documented in
`docs/team/dex3-isaac-review/GRIP_CONTROLLER_FIX.md`. It has not been integrated
into this recorded PPO experiment.

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
python scripts/dex3_training/freeze_collision_cache.py \
  --inputs /private/new-pilot --out /private/new-pilot-locked
```

Both preparation commands run in the training image under `/opt/dex3-extension/`,
with inputs mounted read-only and the output parent mounted writable. Neither
requires a GPU. Freezing uses the actual upstream scene builder and records every
collision cache file it reads or creates. An optional `--seed-cache` reuses a
previous verified run's geometry. The original source pack stays unchanged.

## Train and evaluate

```bash
export DEX3_TRAINING_INPUTS=/private/new-pilot-locked
export DEX3_TRAINING_OUTPUTS=/private/new-results
mkdir -p "$DEX3_TRAINING_OUTPUTS"
docker compose -f docker/dex3-training/compose.yaml run --rm pilot
```

The default pilot saves `model_0.safetensors` before learning, then checkpoints at
25, 50, 75 and 100 iterations. Local outputs include every iteration's metrics,
definitions, scene physics arrays and the resolved checkpoint configuration.
Set `DEX3_TRAINING_IMAGE` to the verified registry digest to use the published
image. `experiment=dex3_exploration` is a separate 1,000-iteration experiment with
higher exploration and a simulation force penalty above 10 N per contact slot.
Its force penalty uses simulated contacts and requires no human contact labels.

To reproduce that longer experiment in a fresh output directory:

```bash
docker compose -f docker/dex3-training/compose.yaml run --rm pilot \
  /opt/dex3-extension/train.py --config-dir /opt/dex3-extension/configs \
  experiment=dex3_exploration output_dir=/outputs/exploration
```

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

For the longer experiment, evaluate `exploration/model_0.safetensors` and
`exploration/model_1000.safetensors` with fresh report/trajectory names. After
downloading results, use this checkout's `scripts/dex3_training/compare_policies.py`
under Python 3.11 or newer to compare them. It verifies all input bytes, frozen
collision hashes, matching policy configurations/schema/image receipts, checkpoint
hashes, installed controls, evaluation coverage and zero assistance. It rejects
the comparison if these contracts differ. This comparison helper was added after
the published image; invoke it from this checkout.

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
