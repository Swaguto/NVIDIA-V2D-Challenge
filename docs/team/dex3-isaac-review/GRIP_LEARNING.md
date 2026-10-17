# Measured-contact grip learning

**Experimental: full physical grasp has not passed.** Current contact-time wrench recipe:
3,000 iterations / 24,576,000 logged environment steps for the fresh-run control. Actor-initialized training is recorded separately below.
Use the private, pushed-and-pulled digest in
[registry.grip.env.example](../../../docker/dex3-training/registry.grip.env.example).

This experiment addresses the pot rotating within a contacting hand. Contact
geometry readback places the failed contacts on the saucepan handle shaft near
its base. Reaching the handle alone does not establish a load-bearing grasp.
The offline wrench audit uses measured normal-force budgets and an inscribed
eight-edge friction pyramid. Its gravity-load capacity is conditional on those
forces; it does not certify motor feasibility, dynamic stability or task success.

## Active matched comparison — October17

Three fresh experiments compare the12mm wrist residual clip, a35mm clip, and the35mm clip with decaying training VOC and20-step reset settling. Each completed 2,000 iterations / 16,384,000 new environment steps. All three failed the full grasp gates. The assisted arm reached the required lift but still slipped and rotated. [Measured comparison](TRAINING_DIAGNOSIS.md). Final evaluation starts at frame0 with zero assistance and no settling. [Reproduction commands](CURRICULUM_REPRODUCTION.md).

A separate estimated CWS path addresses the newly measured mismatch between reconstructed contact surfaces and native convex colliders. GPU contracts and a fresh 2,000-iteration / 16,384,000-step trial passed. The first carry improved from 74.9 mm to 33.2 mm worst slip, with 99.5% required-pad support; full retention still fails. [Measured comparison](ESTIMATED_CWS_RESULTS.md). The zero-assistance actor-only continuation and its matched retention-weight-8 variant both regressed and were rejected. A complete PPO-state continuation is now running; it preserves the parent critic, optimizer, exploration, normalization, learning rate and counters. [Exact-resume diagnosis and reproduction](EXACT_PPO_RESUME.md). [Method and limits](ESTIMATED_CONTACT_GUIDANCE.md). Inputs, contact estimates and checkpoints remain private.

## Earlier actor-initialized PPO run — October 17

The actor-initialized native-wrench run logged **3,000 new iterations / 24,576,000 new environment steps**, separately from its parent actor’s 16,384,000 steps. Logged steps include explicitly failed numerical transitions where reported. Full frame-zero readbacks and fixed grasp gates are in [the measured report](grip-learning-results-v1.json). **Full physical grasp remains unresolved**; no hidden-set readiness is claimed.

Session 87 was **STOPPED** after hash-verified backups at a conservative cumulative compute bound of **$31.2011 of the approved $40**, not an invoice. Session 88 completed and stopped after hash-verified backups at a conservative cumulative bound of $32.0532. Sessions 89–91 are stopped with verified backups. Session 92 is testing a full-state continuation under a 90-minute stop guard; its prior conservative bound is $33.5414 and its guard bound is $36.1514 of $40. Session 86 completed the matched controlled comparison. Inputs and checkpoints stay in the local v8 pack for private transfer. The image contains code only. GPU training and contact physics are not bit-exact.

## Earlier unassisted training changes

The `dex3_grip_curriculum` experiment adds measured object-frame contact forces,
object velocity and reference phase to the policy observations. Rewards use the
simulator's robot contacts for required-pad support, relative motion, resultant
load and moment balance, achieved lift and excess force. Unknown human contact
labels remain disabled. A stationary object that misses the required lift gets
zero achieved-lift reward.

The observation wrapper clears stale contact features after reset. Its configured
magnitude bound applies only to finite observation values. NaNs and infinities
are preserved and abort training with metric names and a physics-state snapshot;
they are never silently converted into successful steps.

All 343 source frames, original confidence masks, collision geometry, nominal
mass/friction and hardware effort limits are retained. Object assistance is zero.
Wrist residual bounds are 12 mm per position axis and 0.35 rad per Euler axis;
finger residuals are bounded to 0.5 rad per joint. These are component limits,
not vector-magnitude limits. Finger drive effort remains capped at 0.76 Nm.

The earlier staged pilot used 256 worlds, 32 rollout steps and 2,000 PPO iterations:
**16,384,000 logged environment steps**. Checkpoints are saved every 100
iterations. Training reset probabilities are:

| Iterations | Starts from frame zero | Other starts |
| --- | --- | --- |
| 0–499 | 25% | Sampled reference states |
| 500–999 | 60% | Sampled reference states |
| 1,000–1,999 | 100% | None |

Reference-state initialization is a training curriculum. It does not demonstrate
grasp acquisition. Separate evaluation starts at frame zero, has no auto resets
or settling assistance, and covers the entire source sequence. Training metrics,
including reference-end fractions from sampled starts, are not evaluation success.

## Local input and checkpoint archive

The owner has the verified `dex3-pot-task-inputs-local-v8.tar.gz` archive (1,062,953,182 bytes). [Its receipt](local-pot-task-input-receipt-v8.json) records the archive SHA256 and every member hash. Inputs, checkpoint weights and derived footage remain local; obtain the archive by private transfer. Repository and GHCR access alone do not provide these inputs.

On the Linux GPU host, compare the archive SHA256 with the receipt, extract into a private directory, and use its `native/e002-speed1-v8` directory for `DEX3_TRAINING_INPUTS`. Parent results, recorded fit candidates, code contexts and validation receipts are included under the paths used by the recipes. The archive preserves rejected experiments as evidence; use the matching pinned image and recipe, not an arbitrary historical context.

## Reproduce the experiment

Use a Linux NVIDIA GPU host, the verified grip image digest, and the privately
transferred corrected-clock `native/e002-speed1-v8` directory. Its manifest SHA256
is `c4d95304f28b1c484059a10c2f5bd19412f5f468b003769f9ef1f1882bb82da4`.
The image contains no episode pack, footage, checkpoints, MANO weights or credentials.
These observations differ from the older PPO policies: use the matching image
and saved experiment for each checkpoint.

Copy the example environment file, set its two absolute paths, authenticate to
private GHCR with your own GitHub account, and load its three variables into
your shell. The tested host is one L40S 48 GB; RAM is capped at 24 GB with
8 GB shared memory. Check the actual reset as well as rewards and loader:

```bash
docker compose -f docker/dex3-training/compose.grip.yaml run --rm grip \
  /opt/dex3-extension/check_numerical_failure.py \
  --config-dir /opt/dex3-extension/configs \
  experiment=dex3_grip_wrench_learning scene.world_count=2 training.world_count=2
```

Set `DEX3_GRIP_IMAGE`, `DEX3_TRAINING_INPUTS` and `DEX3_TRAINING_OUTPUTS`, then run:

```bash
docker compose -f docker/dex3-training/compose.grip.yaml run --rm grip
docker compose -f docker/dex3-training/compose.grip.yaml run --rm grip \
  /opt/dex3-extension/runtime_controls.py \
  --reference /data/reference.parquet --out /outputs/runtime-controls.json
docker compose -f docker/dex3-training/compose.grip.yaml run --rm grip \
  /opt/dex3-extension/train.py --config-dir /opt/dex3-extension/configs \
  experiment=dex3_grip_wrench_learning scene.world_count=8 training.world_count=8 \
  training.max_iterations=8 training.save_interval=4 output_dir=/outputs/smoke
```

For the full pilot, use a fresh output directory:

```bash
docker compose -f docker/dex3-training/compose.grip.yaml run --rm grip \
  /opt/dex3-extension/train.py --config-dir /opt/dex3-extension/configs \
  experiment=dex3_grip_wrench_learning scene.world_count=256 training.world_count=256 \
  training.max_iterations=3000 training.save_interval=250 output_dir=/outputs/ppo
```

Record achieved hand/object poses, contacts, motor readback and reward terms:

```bash
docker compose -f docker/dex3-training/compose.grip.yaml run --rm grip \
  /opt/dex3-extension/record_policy_readback.py \
  --checkpoint /outputs/ppo/model_3000.safetensors \
  --out /outputs/trained-readback --world-count 4
docker compose -f docker/dex3-training/compose.grip.yaml run --rm grip \
  /opt/dex3-extension/audit_native_task.py --run /outputs/trained-readback \
  --inputs /data --out /outputs/trained-audit.json
```

Repeat the readback and audit for `model_0.safetensors`. Also run both checkpoints
through the unchanged submission kit wrapper as described in
[official compatibility](OFFICIAL_EVALUATION_COMPATIBILITY.md). That separate run
records evaluator provenance; the instrumented readback does not claim an official
score. No bit-exact GPU training or physics reproducibility is claimed.

## Acceptance

The previous full-start force-aware pilot stopped after 716 healthy iterations
(2,932,736 environment steps) when a metric became nonfinite. Its last saved
checkpoint at iteration 500 independently failed the full replay and was rejected.
Later staged and actor-initialized results are measured separately in the linked report. Passing CPU/GPU contracts
or loading a checkpoint through the official wrapper does not pass physical grip.
e002 is an unscored development episode; scored-roster and hidden-episode performance
remain unverified. See [ordered results](ORDERED_FIXES.md) for measured outcomes.

### Staged pilot: numerical failure retained

The staged run recorded **1,601 healthy iterations / 13,115,392 steps**, then
aborted on nonfinite metrics at attempted step 13,123,584. Its last saved model is
iteration 1,600 / 13,107,200 steps. End-of-rollout arrays were finite after automatic
resets; that snapshot alone cannot identify the failed transition. A training-only,
read-only observer now retains the first nonfinite state, reward, contact and
tracking arrays **before reset**. It aborts learning when it observes a failure;
it does not repair physics, clamp NaNs, rewrite rewards or affect evaluation.

The 500-iteration independent replay still fails. First-pot lift is 42.8–53.1 mm
versus approximately 99.0 mm in the reference; required-pad support is 84.2–86.9%.
The second lid lifts in all four worlds but twists by 22.4–35.8 degrees. These
intermediate observations do not establish the final saved checkpoint's quality.

The optional `dex3_grip_pose_tracking` configuration is a separate hypothesis,
not a validated repair. It broadens the existing unit-axis keypoint Laplacian
scale from 0.02 to 0.15 and raises its training weight from 2 to 8. Offline reward
sensitivity on checkpoint 500 increases, but new physical evaluation is required.
It does not change the grip, penetration, tracking or evaluation thresholds.

The separate `dex3_grip_carry_learning` candidate also requires actual lift
progress before paying the support and carried-load rewards during a requested
carry. Acquisition before lift retains its contact reward. A 50 mm rational
height scale gives partial lifts a useful training signal, and the optional
stability term includes relative hand/object pose error as well as relative
velocity. A motionless grip in the wrong relative pose can no longer receive
the maximum stability reward. All reward terms remain bounded.

The carry candidate starts with Gaussian action noise 0.1. At the fixed left
finger stiffness of 8 Nm/rad and position scale 0.25 rad/action, noise 0.3
corresponds to 0.6 Nm of target-torque variation before native saturation; noise
0.1 reduces that component to 0.2 Nm. These are unsaturated target contributions,
not measured actuator efforts. The hardware limit remains 0.76 Nm.

This candidate starts a fresh training comparison; it changes the objective and
must not be presented as an unchanged resumption of an older policy. Its GPU reward and actual native-reset contracts, eight-iteration smoke and
fresh-checkpoint official-wrapper checks passed. Full physical performance
requires independent validation. The
10 mm / 5 degree retention and other independent acceptance checks are unchanged.

### Continue an unchanged experiment

`training.resume=true` requires a fresh output directory and the parent image's
context manifest, mounted read-only with the parent checkpoint. The validator binds
input bytes, observation schema, resolved experiment, upstream patches and runtime
payload. It allows a longer horizon and different output/checkpoint save locations;
changes to the action, objective, physics or selected experiment are rejected.

The upstream restore retains optimizer, normalization and RNG state, then resets
the simulated worlds. It is not bit-exact continuation. The GPU smoke continuation
from iteration 8 to 10 completed 512 new steps, retained the 2,560 cumulative-step
counter, emitted no false `model_0`, and loaded through the actual official wrapper.
That verifies checkpoint mechanics, not physical task success.

## Contact-model sensitivity

`prepare_physics_probe.py` can fork a checkpoint for evaluation with a finer
timestep or the existing `default_patch` hand profile. It preserves every tensor
byte and the original training configuration/counter in provenance, and marks
the result evaluation-only. Training resumption rejects these probes.

The patch profile changes `condim` from 3 to 4. MuJoCo documents the additional
degree as friction torque about the contact normal, representing finite contact
patches. Its torsional coefficient has units of length.
[MuJoCo contact documentation](https://mujoco.readthedocs.io/en/stable/computation/#contact).
The existing model coefficient is nominal, not a hardware measurement. Runtime
readback must verify the actual profile, other material parameters and motor caps;
passing with this profile does not validate a policy under point contacts.

### Native warm-start reset correction

The terminate-on-numerical-failure experiment encountered 9,244 invalid world
transitions. After the first affected rollout, one world failed every subsequent
control step. The run was stopped by the operator; exit 137 was the stop timeout,
not evidence of an out-of-memory failure. This run is not accepted training.

Pinned MuJoCo-Warp 3.8.1 initializes its constraint solve from
`Data.qacc_warmstart`. Newton 1.2.0 updates reset positions and velocities but does
not clear that acceleration cache. The training-only environment now clears the
cache for selected reset worlds, including automatic resets inside the captured
step. It does not call `reset_data`, alter packed contact membership, repair a
failed transition, or change stock evaluation. It rejects stateful actuators,
which would require a separate reset contract.

The first raw failure is written at the next metrics boundary. Two consecutive
invalid transitions in any world abort training at that boundary, rather than
allowing a poisoned world to consume the remaining budget. The GPU regression passed: it injected NaNs into the actual native warm-start buffer, observed a failed
transition and reset, then verified six healthy captured steps and unchanged
unselected-world cache/contact membership before further training. All four
inspected installed vendor sources match verified PyPI wheel source hashes.

### Distributed-contact wrench candidate

The slot-only load reward approximates the moment by crossing each link's mean
contact position with its summed force. This loses contact couples. An offline
reconstruction from the recorded 2,000-iteration policy found a median pot moment
difference of 0.094 Nm and a 95th percentile of 0.222 Nm over its grasp window
(four worlds). These differences exceed the reward's 0.08 Nm moment scale.
The reconstruction uses recorded contact points; small differences between the
two bodies' contact points are not included in that comparison.

The separate fresh experiment `dex3_grip_wrench_learning` sums every actual native
hand/object contact force and moment about the assigned object's COM. It includes
native torsional/rolling contact moments, handles either shape ordering and
excludes the table and other objects. Required-pad observations remain unchanged.
The existing slot approximation remains the default for archived configurations.
This objective change is a new experiment. Its GPU couple/sign/isolation/NaN, actual reset, fresh smoke and official wrapper checks passed. Independent physical evaluation remains required.

The vendor conversion also distinguishes two surface points from the force
application point. Newton offsets those surfaces by half the signed gap on
either side of MuJoCo's contact midpoint. There is also a timing difference:
native contact positions and inertial COM positions precede integration, while
the returned Newton state follows integration. Reconstructing either quantity
from that returned state mixes times. The candidate now uses an explicit read-only
binding to native `contact.pos` and `xipos`, with a validated native/body mapping.
Its real-scene check decodes the wrench independently on the CPU and checks that
the readout leaves native inputs unchanged. A nonzero resultant with a displaced
COM exercises the moment origin; the different-time COM displacement is reported
as a diagnostic. The direct-buffer GPU check passed: 48 assigned contacts over six captured steps matched an independent CPU decode within 7.16e-7; native input buffers remained unchanged. Synthetic couple/sign/COM/world/NaN, reward, actual numerical-reset and fresh eight-iteration official-wrapper checks also passed. See [the mechanics receipt](grip-learning-mechanics-v1.json).
The early surface-point pilot was stopped and preserved before longer training.
New diagnostics retain both the archived surface-point convention and the native
force application positions, together with measured assigned-hand wrenches.

### Actor initialization for a new reward experiment

`dex3_grip_wrench_finetune` is an explicit new experiment. It can initialize the
actor mean and observation normalization from the measured 2,000-iteration
staged policy. It retains a fresh critic, optimizer, RNG, exploration noise,
learning rate, simulator state and zero training counters. This avoids discarding
the earlier actor's stronger contact behavior when correcting the reward.

Initialization rejects changed input hashes, policy schemas, action/observation
semantics, embodiment, geometry, physics, reset behavior or network structure.
The native read-only solver config with `enable_multiccd=false` is equivalent to
the pinned upstream solver default for this check. Reward changes are allowed
and recorded as a new experiment. Parent and current build contexts must be
clean, hash bound and use the same pinned runtime.

The parent checkpoint and its context manifest must be mounted privately at
`/initial-policy/ppo/model_2000.safetensors` and
`/initial-context/context-manifest.json`. Initialization lineage records the
parent training count separately. Its steps are never added to the new run's
counter. `check_actor_initialization.py` compares a real new `model_0` with the
parent and a fresh upstream learner. That GPU check, an eight-iteration smoke,
the actual official wrapper and complete four-world initial/smoke replays passed
their mechanics checks. Initial pot lift was 27.1–28.2 mm; after eight updates it
was 26.1–27.8 mm with at least 97.8% required-pad support. Both still fail the
99 mm lift and fixed retention criteria. The longer run completed 3,000 new iterations / 24,576,000 new steps with zero numerical failures. Broad reference tracking reached 100%, but pot lift was only 31.9–33.3 mm, with up to 67.8 mm slip and 31.5° relative rotation; strict grasp checks failed.

The actor image is pinned separately in
[registry.actor.env.example](../../../docker/dex3-training/registry.actor.env.example).
The parent paths refer to files retained for private transfer in the local v8
pack. The earlier v7 pack does not contain these later training artifacts.
Copy the example to a private environment file and set all five paths/variables.
The following commands use that file and the two Compose files from this branch:

```bash
docker compose --env-file /private/actor.env \
  -f docker/dex3-training/compose.grip.yaml \
  -f docker/dex3-training/compose.actor-init.yaml run --rm grip

docker compose --env-file /private/actor.env \
  -f docker/dex3-training/compose.grip.yaml \
  -f docker/dex3-training/compose.actor-init.yaml run --rm grip \
  /opt/dex3-extension/check_actor_initialization.py \
  --parent /initial-policy/ppo/model_2000.safetensors \
  --child /outputs/actor-smoke/model_0.safetensors \
  --parent-context /initial-context/context-manifest.json

docker compose --env-file /private/actor.env \
  -f docker/dex3-training/compose.grip.yaml \
  -f docker/dex3-training/compose.actor-init.yaml run --rm grip \
  /opt/dex3-extension/train.py --config-dir /opt/dex3-extension/configs \
  experiment=dex3_grip_wrench_finetune scene.world_count=256 training.world_count=256 \
  training.max_iterations=3000 training.save_interval=250 output_dir=/outputs/actor-ppo
```

Readbacks label the measured wrench's side order explicitly. The pinned contact
tracker orders hands right then left; this differs from the target file's hand
axis. Use the saved labels and assigned object IDs when interpreting those arrays.

### Bounded carry motor calibration

The native grasp fitter also supports `fit.mode=carry_height_sweep`. It retains
an existing acquisition pose and tests wrist-height command offsets from 0 to
25 mm, only after closure and while the source asks the object to rise. The gate
is smooth on the original source clock; it fades out with the grasp. Object
poses, gravity, geometry, mass, friction and actuator caps remain unchanged.

This is a robot motor program, scored through actual physics. It is not an
object trajectory edit or PPO training. Every candidate runs all source frames;
the selected grasp must cover its complete window. The fitter reports every
paired grid result and retains raw non-finite or motor-bound failures. A fitting
pass still needs fresh independent replay before `freeze_phase_policy.py` can
promote the exact program.

`fit.prior_journal` can seed the sweep from an incomplete recorded search. It
checks input hashes and matching action, embodiment, task, physics and geometry
configuration, records the journal/configuration hashes, and marks the recovered
candidate unaccepted. Incomplete search results do not become accepted priors.
GPU sweeps and joint fits ran; all current candidates remain rejected by the independent grasp checks.

## Carry-transition parameter fitting

The current bounded native-physics searches preserve acquisition, source arrays, masses, friction, gravity, and native torque limits. A motor-only height correction can reach the requested pot lift but loses the grip during the carry reversal. This candidate is rejected. Pose refitting retains that separate carry program. The new `pose_and_carry_height` mode jointly fits thirteen motor-pose parameters and a separate 0–25 mm carry-height parameter; exported phase actions remain thirteen-dimensional. Original source motion and hardware limits remain fixed. The opt-in `constraint_violation` ranking prioritizes failed physical constraints so an already-safe rotation cannot buy worse slip. The pass/fail auditor remains unchanged. Native fitting clears acceleration warm starts after resetting all worlds. With explicit retry enabled, an invalid proposal is retained as raw evidence and excluded; every remaining proposal is replayed from frame zero in an entirely finite batch before scoring. Only one replay is allowed per iteration, and more than 10% invalid candidate pairs aborts the search. Evaluation is unchanged. Each candidate executes all 343 frames from frame zero; growing-prefix results cannot accept a complete grasp.

The measured JSON lists parameter-search iterations and simulated steps separately from PPO. Promotion requires the full fixed grasp window and independent four-world replay; a visually higher pot alone cannot pass. See `native_parameter_searches` and `motor_program_readbacks` in [the measured report](grip-learning-results-v1.json).

## Docker Compose compatibility

Training Compose files request GPUs through device reservations, compatible with the Brev host’s Compose 2.29.7. The actor overlay retains read-only parent/input mounts. Use the supplied Compose files and the corresponding pinned image; the GPU contract check must pass before training. The device reservation follows [Docker’s GPU Compose documentation](https://docs.docker.com/compose/how-tos/gpu-support/).

## Issue 10 and current physical status

[Issue 10 remains incomplete](ISSUE_10_STATUS.md). Recent controller/contact searches are physics diagnostics, with **zero new PPO steps**. The later session 86 PPO comparison implements decaying assistance and has a local accuracy report; nothing has been posted to Scorekeeper. FlashCHORD/Newton/MuJoCo-Warp with JAX PPO is the backend of these readbacks. The earlier Isaac render is a separate artifact.

The joint pose/height fit reached the requested lift, but its independent carry still failed. A four-world contact-model comparison also failed: its best multi-contact/angular case reached approximately 12.3 mm slip, 5.4 degrees rotation and 86.9% required-pad support. Acceptance still requires 10 mm, 5 degrees and 95% support, along with the unchanged lift, penetration and source-tracking checks. Other variants include severe finite trajectory failures; all worlds remain in the measured report. None is promoted.

The optional angular-reference wrist servo is separately capped and requires zero native passive wrist damping. It changes controller semantics and is not an unchanged policy resume. Contact-model experiments retain the recorded material coefficients; neither soft-finger contact nor multiple collision contacts has established a secure full-task grasp.

## Verified repeated-world fitting recipe

The latest fitting image supports `fit.worlds_per_candidate=2`, `4` or `8`; each proposal is ranked by its worst world. A failed world excludes the complete proposal group. Raw failed states are preserved, and the entire clean cohort must replay from frame zero before scoring. This improves candidate screening; it does not guarantee robustness to physical perturbations or hidden episodes.

Native fitting now requires `NativeContactSimConfig`. Its plan binds all resolved solver and timing settings, as well as model and action contracts. Changing collision-manifold settings, solver iterations, friction cone or timing invalidates the bound plan. The GPU check exercised four-world grouping, injected a native failure, completed the clean replay, matched a fresh independent replay's solver hash, and verified that changed solver iterations are rejected. Physical acceptance was not asserted by that mechanics check. The remote receipt-writing job lacked permission to write into a Docker-owned output folder; a local verifier recovered the successful checks from hash-verified reports and logs. [Mechanics evidence](grip-fitting-mechanics-v1.json) preserves that distinction.

For the exact short recovery check, privately extract local v8, copy `docker/dex3-training/registry.fit.env.example` to `registry.fit.env`, and set its absolute paths. Use a fresh output directory. From `docker/dex3-training`:

```bash
docker compose --env-file registry.fit.env -f compose.grip.yaml -f compose.fit.yaml run --rm grip
docker compose --env-file registry.fit.env -f compose.grip.yaml -f compose.fit.yaml run --rm grip \
  /opt/dex3-extension/check_fit_recovery.py --config-dir /opt/dex3-extension/configs \
  experiment=dex3_repaired_payload action.wrist_payload_gain=0.5 \
  sim._target_=v2d_dex3_flashchord.sim.NativeContactSimConfig ++sim.enable_multiccd=false \
  scene.world_count=16 +fit.worlds_per_candidate=4 +fit.side=left +fit.phase=0 \
  +fit.iterations=1 +fit.mode=pose_and_carry_height +fit.retry_failed_cohorts=true \
  +fit.prior=/fit-prior/phase-policy.json +fit.initial_sigma_scale=0.05 +fit.objective=constraint_violation \
  +fit.bounds_profile=grasp_refit +fit.position_frame=object_reference output_dir=/outputs/fit
```

The default command checks reward mechanics. The second command deliberately injects a numerical failure to test rejection and recovery. It is not a trained policy or a successful task. Its `fit/recovery-contract.json` records the check. Normal parameter searches call `fit_phase_policy.py`, retain full-window independent acceptance, and must have their own output directory and compute allowance. Do not use the injected-failure checker for learning.

For fresh replay, point `DEX3_FIT_PRIOR_DIRECTORY` at the resulting output `fit` directory and run:

```bash
docker compose --env-file registry.fit.env -f compose.grip.yaml -f compose.fit.yaml run --rm grip \
  /opt/dex3-extension/diagnose_task.py --config-dir /opt/dex3-extension/configs \
  experiment=dex3_repaired_payload action.wrist_payload_gain=0.5 \
  sim._target_=v2d_dex3_flashchord.sim.NativeContactSimConfig ++sim.enable_multiccd=false \
  scene.world_count=4 +residual_plan=/fit-prior/phase-policy.json output_dir=/outputs/replay/diagnostic
docker compose --env-file registry.fit.env -f compose.grip.yaml -f compose.fit.yaml run --rm grip \
  /opt/dex3-extension/audit_native_task.py --run /outputs/replay/diagnostic --inputs /data --out /outputs/replay/audit.json
```

GPU physics and training are not bit-exact across runs. The team can reproduce the pinned procedure, contracts and failure diagnosis; a secure full grasp has not been produced. The archived actor fine-tuning recipe uses its separate matching actor image and parent checkpoint. Episode inputs and checkpoint weights remain local for private transfer, never in GHCR.
