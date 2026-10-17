# Ordered fixes: October 16 development checkpoint

## Current result

The evaluation adapter, native controller extensions, controlled PPO experiment
and local reproduction package are implemented. **The physical task is still
unsolved.** Later phases remain gated on that result; no failed grasp or policy
is promoted as ready for the scored or hidden episodes. CPU verification:
**161 tests pass**. See [the complete measured report](ordered-fixes-results-v1.json).

The final three multi-contact variants each cover 343 original source frames
in four worlds at 20 Hz control, 400 Hz physics and source speed 1.0. All fail
full-task acceptance. Solver readback verifies multiple convex contact points
are enabled, actual GPU timestep is 0.0025 seconds, and native drive caps hold.
The point-contact control has 39.09 mm pot and 53.05 mm lid mean position error,
with zero success. The fitted patch variant becomes unstable in one world;
that divergence remains in the report and is not clipped from the metrics.

Brev is **STOPPED** after the final session55 backup. Its 30 new result files
were hash verified. Conservative cumulative compute bound is **$19.694 of $20**,
not an invoice. All private inputs remain local.

Full physical task acceptance remains pending. Reproduction begins with frozen
derived inputs; new-footage reconstruction and the complete scored roster still
require independent validation. These changes implement the
validation and reproduction path; they do not establish a successful policy or
hidden-episode generalization. The previously reviewed Isaac replay remains
available and has not been replaced by a failed candidate.

## Evaluation compatibility

The actual unmodified kit wrapper and included checkpoint evaluator loaded our
extension runtime and checkpoint and produced trajectories. The kit's runtime pin
is `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`. See
[the measured contract](official-runtime-contract-v1.json) and
[compatibility notes](OFFICIAL_EVALUATION_COMPATIBILITY.md).

e002 is a development episode outside the 20 scored episodes. A compatible
checkpoint is insufficient for submission: the complete scored roster, required
frame grids and 36 reconstructed meshes are still needed.

## Physical repair results

The third-pad lid candidate reduces lid relative rotation to 2.62° and 1.66°,
but its right palm intersects the other hand's nominal grasp and the first pot
slips 119 mm. It is rejected. Raising the right finger stiffness reduces lid
rotation to 8.95° and 10.56°, still above the separate 5° engineering target;
the left wrist position p95 also exceeds the existing 20 mm gate. It is rejected.

The new native training adapter preserves all 343 source knots, including 271
confident and 72 inferred source rows. Original confidence masks remain in the
private pack; inferred robot commands do not supply unknown human contact labels.
It transfers bounded finger feedforward and measured-object wrist acquisition.
Reference velocity feedforward is optional and was evaluated separately.
The native engine runs at source speed 1.0; the reviewed Isaac replay used 0.5.

The fitted Isaac convex surfaces can now be copied into native URDF collision
elements while retaining authored visual meshes, joints, masses and inertias.
The imported native geometry audit matches every convex part on 18 hand/object
bodies within 1 μm; the largest measured vertex difference was 3.7 nm. This does
not make the engines' contact response or corrected inertias identical.

Native controls compare original/firmer wrist gains, bounded pose feedback,
recorded collision hulls, 20/100 Hz motor commands, normal-only preload and
reference velocity feedforward. The 100 Hz option preserves source knots and
motion speed, with 400 Hz physics unchanged. It needs a verified submission
frame-grid contract before use on a scored submission. None of these controls
has passed complete grasp retention. Preload alternatives remain rejected.

Readback now includes actual hand/object poses, native joint motor forces,
motor targets, imported collision geometry and measured contacts. The latest
velocity trials verify all 14 finger joint limits and six root position limits.
Contacts are sampled at the final physics substep per control step; they do not
certify continuous contact. The independent grasp audit rejects missing required
pad support, excessive relative rotation and a stationary object when the source
requires a lift. Existing gates remain unchanged.

## Source-speed force-plan correction

The archived Isaac force plan was computed at source speed 0.5 while native
playback is 1.0. Its inertial feedforward terms were therefore too small.
`recompute_native_force_plan.py` recomputes those terms at 1.0 and preserves
every other target array exactly. Maximum motor-plan change is 0.069271 Nm;
maximum corrected feedforward is 0.400269 Nm, below the 0.55 Nm planning bound
and 0.76 Nm native limit. Original poses, timestamps and confidence masks remain.
Newly generated native packs reject a declared clock mismatch. Legacy packs
remain frozen so earlier results can be reproduced honestly.

The corrected native pack has manifest SHA-256
`c4d95304f28b1c484059a10c2f5bd19412f5f468b003769f9ef1f1882bb82da4`.
Both 20 and 100 Hz corrected-clock diagnostics still fail full retention. A
fallen object leaving the finite support remains visible in the error metrics.

## Bounded grasp fitting and contact checks

The phase-policy fitter optimizes robot wrist/finger residuals with full-sequence
physics. It preserves reference trajectories, hardware effort limits and unknown
source contact labels. Parameters are bound to input, physics and controller
hashes; position offsets can rotate with the object reference. A candidate must
pass a separate replay before it can be frozen into training inputs.

The local and widened object-frame searches each ran eight iterations,
32 candidates repeated in two worlds, over 343 source frames: **175,616 simulated
environment steps per search; zero PPO steps**. Both failed the first pot phase.
The widened candidate lifts only 43.7–51.5 mm against 99.0 mm required reference
motion, with 73.6–129.4 mm relative slip and 35.9–45.6° rotation. Sampled
hand/object penetration reaches 2.64–3.73 mm, above the separate 2 mm check.
No phase-policy prior was accepted. Other phases are not marked solved.

Contact experiments separate a faster 5 ms time constant from higher impedance
at the original 20 ms time constant, and point versus torsional contact. None
has passed complete retention. The original physical parameters are retained
as a selectable control; failed contact variants are not promoted.

## Controlled learning

The new full-episode PPO experiment compares the untouched initial checkpoint
with 1,000 iterations / 1,536,000 environment steps. It uses 64 worlds, a
24-step rollout, zero virtual object assistance and no source-time slowdown.
Initial and final policies use identical inputs, geometry and saved config.
The official kit supplies separate complete-episode evaluation trajectories;
read-only checkpoint diagnostics additionally preserve hand and contact state.

Both checkpoints have **0% full-task success** in the actual kit evaluator.

| Policy | Pot mean position error | Lid mean position error | Mean ADD |
| --- | ---: | ---: | ---: |
| Initial, iteration 0 | 42.24 mm | 91.18 mm | 69.46 mm |
| Trained, iteration 1000 | 34.69 mm | 78.70 mm | 62.37 mm |

This experiment used the frozen legacy force plan described above. Its inputs,
image and checkpoints are retained together; it is not a test of the later
clock correction or payload compensation. See
[the controlled comparison receipt](full-task-policy-comparison-v1.json).
Further PPO and multi-episode validation remain gated on physical retention.

## Reproduce the recorded experiment

The **local v7 pack** includes both `native/e002-legacy-v5` for the recorded PPO
comparison and `native/e002-speed1-v8` for corrected-clock diagnostics, plus
checkpoints, evaluator outputs, receipts and source archives. Transfer it directly
to teammates through an approved private channel. It is not in GHCR or a release.
Do not mix checkpoints and manifests from the two input directories.

The archive is 319,661,713 bytes with SHA-256
`d38dff1330ecf03c587ef0583d39ccfaec14c8feeaf61fafb3788d13af000f5b`.
All 8,888 file members were verified after packaging. On the Linux GPU host,
verify the transferred archive before extracting into a fresh private directory:

```bash
export DEX3_ARCHIVE=/private/dex3-pot-task-inputs-local-v7.tar.gz
printf '%s  %s\n' \
  d38dff1330ecf03c587ef0583d39ccfaec14c8feeaf61fafb3788d13af000f5b \
  "$DEX3_ARCHIVE" | sha256sum -c -
mkdir /private/extracted-v7
tar -xzf "$DEX3_ARCHIVE" -C /private/extracted-v7
```

These procedures have been exercised on the validation VM with the same input
bytes. A separate teammate's fresh-machine run has not yet been witnessed.
Package access and direct private transfer of this local archive are both needed.

For experimental phase fitting use the separately pinned image in
`docker/dex3-training/registry.phase-fit.env.example` and the corrected-clock
pack. Fitted priors remain rejected until their independent replay passes.

### Historical full-task PPO comparison

Use the private experimental image digest recorded in
`docker/dex3-training/registry.repaired.env.example` and a
separately transferred local input pack. The image contains code and dependency
layers only. Episode footage, MANO weights, checkpoints and credentials are
excluded. Pulling the image alone does not provide the inputs.

The input directory must have manifest SHA-256
`28e7d07a7d63c5b4a3b8b80e15640463eda17caf353592016f6452bc66d6f3cc`.
The loader verifies all declared file hashes before running. Use an NVIDIA Linux
host with the Container Toolkit and the same tested runtime; an L40S was used.

```bash
set -a
. docker/dex3-training/registry.repaired.env.example
set +a
export DEX3_TRAINING_INPUTS=/private/native/e002-legacy-v5
export DEX3_TRAINING_OUTPUTS=/private/fresh-full-task-results
mkdir -p "$DEX3_TRAINING_OUTPUTS"
docker pull "$DEX3_REPAIRED_IMAGE"

# Installed loader and manifest checks.
docker compose -f docker/dex3-training/compose.repaired.yaml run --rm full-task \
  /opt/dex3-extension/runtime_controls.py \
  --reference /data/reference.parquet --out /outputs/runtime-controls.json

# Complete zero-residual diagnostic. This records a failed baseline as well.
docker compose -f docker/dex3-training/compose.repaired.yaml run --rm full-task \
  /opt/dex3-extension/diagnose_task.py \
  --config-dir /opt/dex3-extension/configs \
  experiment=dex3_repaired_velocity scene.world_count=2 \
  output_dir=/outputs/diagnostic

# Controlled PPO; preserve model_0 before learning.
docker compose -f docker/dex3-training/compose.repaired.yaml run --rm full-task
```

Use the downloaded official kit's `examples/run_policy_evaluation.py` with
separate episode-to-checkpoint maps for `model_0.safetensors` and
`model_1000.safetensors`, four worlds each. The official wrapper lives in the
downloaded kit, not in the image; mount the verified kit read-only and follow
[the wrapper contract](OFFICIAL_EVALUATION_COMPATIBILITY.md).

To reevaluate the **archived** checkpoints from the local pack, extract its kit
ZIP into a fresh local directory and mount it read-only. The included maps use
the fixed `/repro` mount below. Run each policy into a fresh result directory:

```bash
export DEX3_LOCAL_PACK=/private/extracted-v7
export DEX3_KIT=/private/extracted-official-kit
# DEX3_KIT must directly contain examples/run_policy_evaluation.py.
for policy in initial trained; do
  docker compose -f docker/dex3-training/compose.repaired.yaml run --rm \
    -v "$DEX3_LOCAL_PACK:/repro:ro" -v "$DEX3_KIT:/kit:ro" full-task \
    /kit/examples/run_policy_evaluation.py \
    --checkpoints "/repro/official-kit/checkpoints-$policy.json" \
    --results "/outputs/official-$policy" \
    --source-root /data --runtime-root /opt/flash-chord --world-count 4
done
```

This invokes the kit's included evaluator and preserves its provenance. The
recorded outputs in `results/official-velocity-{initial,trained}-v1/eval` provide
the comparison reference. GPU contact trajectories can diverge after a failed
grasp, so compare coverage, contracts and measured outcomes without expecting
bit-identical numbers.

For additional hand/contact readback:

```bash
docker compose -f docker/dex3-training/compose.repaired.yaml run --rm full-task \
  /opt/dex3-extension/record_policy_readback.py \
  --checkpoint /outputs/ppo/model_1000.safetensors \
  --out /outputs/trained-readback --world-count 2
```

New output directories fail closed if already present. Keep the initial and
trained policies separate and compare achieved trajectories, retention and
source coverage. GPU physics and training are not guaranteed bit-exact; repeated
native rollouts have diverged after losing contact. The team can reproduce the
recorded experiment and its evaluation procedure; successful task reproduction
has not been established. Stop Brev after backing up each compute session.

## Optional wrist payload compensation

The native action now has optional robot wrist payload feedforward in addition
to finger and velocity terms. It uses the loaded nominal payload mass, COM and
inertia at source speed 1.0. Force shifts only the robot's prismatic motor
setpoints; moments are shifted from object COM to the wrist and rotated into the
ball joint's child frame. The orientation PD and payload term share the existing
wrist torque cap. It never applies a wrench or a pose write to an object.

The term fades after losing the grasp. Reset worlds receive no stale command
until their next processed action; substep setpoint offsets do not accumulate.
The actual GPU reset/limit contract passes. Four-world complete replay verifies
all 14 native finger limits, six root position limits and the shared orientation
cap. The reset-safe replay reaches at most 0.75335 Nm finger motor torque and
1.32163 Nm direct wrist effort.

This remains an **experimental, failed full-task controller**. Its fresh
four-world run has 39.68 mm pot and 84.30 mm lid mean position error, with zero
full-task success. First-pot lift is only 8.75–43.31 mm versus 99.0 mm reference
motion. Required-pad support is 33.9–41.5% of sampled phase frames. The other
half/full-gain and point/torsional-contact comparisons also fail retention.

## Grasp-search acceptance

The fitter now includes measured assigned-hand/object penetration in its
objective and feasibility checks. A GPU regression verifies signed separation,
contact ownership and world isolation. Its separate replay audit remains the
promotion gate; no training policy or input prior is accepted from reward alone.

An optional curriculum first optimizes a stable initial hold, then extends the
objective toward the complete carry phase. Every candidate still simulates all
343 source frames. Previously accepted phases remain fully checked. A partial
hold is explicitly marked incomplete and cannot set `robust_phase_pass` or be
frozen as a successful complete grasp. The legacy full-window mode remains
selectable for reproducing earlier searches.

## Later controller comparisons and remaining failure

The original contact curriculum passes its initial supported hold through
5.4 source seconds, but fails the next 6.4-second window. Lift is only
51.9–54.5 mm against 69.3 mm required, and rotation exceeds 5°. The higher
impedance variant fails earlier. A subsequent 32-iteration refinement with
64 worlds simulates 702,464 steps but still fails the 6.4-second prefix and
independent full replay. These are bounded action searches, with zero PPO steps.

Torsional contact with original compliance, lift-dependent robot payload gain,
elliptic cones and native multi-point convex contact were compared separately.
No complete variant passes. Mass, friction coefficients, geometry, gravity,
source timing and motor caps were retained. These are explicit experimental
configurations, with the original profile still available.

Readback of the refined grasp shows roughly 0.7° / 2.5 mm wrist command error
around 6.4 seconds while pot orientation error grows to 15° and later 38°.
Measured pad forces are already about 14/12/7 N at that time. This supports
investigating grasp wrench balance and geometry before adding pressure or
running more blind PPO. It does not prove which contact or geometry assumption
causes the rolling. No successful full-episode solution has been found.

The verified CPU command is:

```bash
python -m pytest -q scripts/dex3_training/tests scripts/dex3_physics/tests \
  scripts/reproduction/tests scripts/dex3_review/tests scripts/reproduce/tests
```

The next physical acceptance work is to establish a stable loaded handle grasp
in the native engine, then carry and lid manipulation, before repeating learning
and expanding to independently prepared episodes and the scored roster.
The existing one-object-per-hand phase planner does not automatically cover
arbitrary hidden scenes.

## Reproduce the latest diagnostic

This command reproduces a **failed diagnostic**, not a successful task. Extract
the separately transferred local v7 archive. Set paths to its corrected-clock
inputs and a fresh output directory. The full pack receipt records every member
hash: [local v7 receipt](local-pot-task-input-receipt-v7.json).

```bash
set -a
. docker/dex3-training/registry.experimental.env.example
set +a
export DEX3_TRAINING_INPUTS=/private/extracted-v7/native/e002-speed1-v8
export DEX3_TRAINING_OUTPUTS=/private/fresh-multicontact-results
mkdir -p "$DEX3_TRAINING_OUTPUTS"
docker pull "$DEX3_EXPERIMENTAL_IMAGE"
docker compose -f docker/dex3-training/compose.current.yaml run --rm diagnostic

docker compose -f docker/dex3-training/compose.current.yaml run --rm diagnostic \
  /opt/dex3-extension/audit_native_task.py \
  --run /outputs/diagnostic --inputs /data --out /outputs/audit.json
```

The latest image was built from source
`4407d699e3f4c6e93c699aa33fde7e5bf9b7ef8e`, privately published and pulled back
by digest. It adds an explicit native solver configuration and a fourth guarded
runtime dispatch hook. Installed loader checks and GPU diagnostics pass; a fresh
checkpoint from this four-hook configuration has **not** been evaluated through
the official kit wrapper. The archived PPO comparison uses its matching older
three-hook image and legacy input pack. Keep those reproduction paths separate.

The recorded results are under `results/native-multiccd-point-v1` in the local
pack. Further experiments can choose the patch or fitted configuration, but none
has passed the independent physical gate. New outputs must use fresh paths.
