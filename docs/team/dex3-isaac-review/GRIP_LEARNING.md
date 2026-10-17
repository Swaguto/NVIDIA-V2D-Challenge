# Measured-contact grip learning

This experiment addresses the pot rotating within a contacting hand. Contact
geometry readback places the failed contacts on the saucepan handle shaft near
its base. Reaching the handle alone does not establish a load-bearing grasp.
The offline wrench audit uses measured normal-force budgets and an inscribed
eight-edge friction pyramid. Its gravity-load capacity is conditional on those
forces; it does not certify motor feasibility, dynamic stability or task success.

## Training changes

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

The staged pilot uses 256 worlds, 32 rollout steps and 2,000 PPO iterations:
**16,384,000 planned environment steps**. Checkpoints are saved every 100
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

## Reproduce the experiment

Use a Linux NVIDIA GPU host, the verified grip image digest, and the privately
transferred corrected-clock `native/e002-speed1-v8` directory. Its manifest SHA256
is `c4d95304f28b1c484059a10c2f5bd19412f5f468b003769f9ef1f1882bb82da4`.
The image contains no episode pack, footage, checkpoints, MANO weights or credentials.
These observations differ from the older PPO policies: use the matching image
and saved experiment for each checkpoint.

Set `DEX3_GRIP_IMAGE`, `DEX3_TRAINING_INPUTS` and `DEX3_TRAINING_OUTPUTS`, then run:

```bash
docker compose -f docker/dex3-training/compose.grip.yaml run --rm grip
docker compose -f docker/dex3-training/compose.grip.yaml run --rm grip \
  /opt/dex3-extension/runtime_controls.py \
  --reference /data/reference.parquet --out /outputs/runtime-controls.json
docker compose -f docker/dex3-training/compose.grip.yaml run --rm grip \
  /opt/dex3-extension/train.py --config-dir /opt/dex3-extension/configs \
  experiment=dex3_grip_curriculum scene.world_count=8 training.world_count=8 \
  training.max_iterations=8 training.save_interval=4 output_dir=/outputs/smoke
```

For the full pilot, use a fresh output directory:

```bash
docker compose -f docker/dex3-training/compose.grip.yaml run --rm grip \
  /opt/dex3-extension/train.py --config-dir /opt/dex3-extension/configs \
  experiment=dex3_grip_curriculum scene.world_count=256 training.world_count=256 \
  training.max_iterations=2000 training.save_interval=100 output_dir=/outputs/ppo
```

Record achieved hand/object poses, contacts, motor readback and reward terms:

```bash
docker compose -f docker/dex3-training/compose.grip.yaml run --rm grip \
  /opt/dex3-extension/record_policy_readback.py \
  --checkpoint /outputs/ppo/model_2000.safetensors \
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
The staged pilot's results must be measured separately. Passing CPU/GPU contracts
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
must not be presented as an unchanged resumption of an older policy. Its GPU
contracts and full physical performance require separate validation. The
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
