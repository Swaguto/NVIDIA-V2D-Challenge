# Full-task training transfer

Status: the native transfer and full-episode evaluation run. The controlled
1,000-iteration / 1,536,000-step full-343-frame experiment has zero success,
before and after learning. See [the current measured checkpoint](ORDERED_FIXES.md).
Complete physical grasp retention is still required for policy acceptance.
The newer [measured-contact grip curriculum](GRIP_LEARNING.md) is an experimental
attempt to learn load-bearing retention; it has a separate observation schema,
image, checkpoints and independent full-start evaluations.

## Private input preparation

Use a verified frozen base training pack, its source reconstruction case, and a
hash-bound `task-grips.json` produced from that case. Do not use a successful
simulator rollout as a training reference. The full-task adapter derives commands
from the frozen planner and preserves the source confidence masks, including
unobserved source frames. Human contact/wrench labels remain unknown.

```bash
python scripts/dex3_training/prepare_task_reference.py \
  --base /private/frozen-base \
  --case /private/source-case \
  --task-spec /private/task-grips.json \
  --out /private/new-full-task
```

This copies the verified robot/object assets and frozen Newton collision cache.
It replaces the previous 75-frame pilot targets with the full source timeline,
including approach, closure, transport and release. It records motor feedforward
separately from geometric finger targets and checks the 0.55 Nm planned bound.
The native finger drive retains the 0.76 Nm limit. Actual runtime physics and
actuator readback still require validation; file generation alone is insufficient.

Current scope: two Dex3 hands, one assigned object per hand, complete inferred
phase plans. Unsupported assignments are rejected. This is not yet an automatic
planner for every hidden scene.

The force plan must explicitly use physical source speed **1.0**. The archived
Isaac recipe used 0.5; reusing its inertial terms at native speed underestimates
acceleration terms by a factor of four. The adapter now rejects that mismatch.
For the archived full-task pack, use `recompute_native_force_plan.py`; it
recomputes only finger feedforward and preserves every other NPZ target array
exactly. Its receipt records alignment checks and source hashes. The local
pack includes the verified result as `native/e002-speed1-v8`.

## Build the code layer

The repaired Dockerfile reuses the verified dependency/runtime image and replaces
only our committed extension. All image contexts exclude episode inputs,
footage, MANO weights and credentials.

```bash
python docker/dex3-training/make_context.py \
  --upstream /path/to/pinned/video_to_data \
  --out /path/to/new-context --repair-layer
docker build --build-arg VCS_REF="$(git rev-parse HEAD)" \
  -t v2d-dex3-training:repaired /path/to/new-context
```

## Validate before learning

Mount the input pack read-only at `/data`, a fresh result parent at `/outputs`,
and use the repaired image with an NVIDIA GPU. First run the unassisted
zero-residual diagnostic:

```bash
/opt/dex3-extension/diagnose_task.py \
  --config-dir /opt/dex3-extension/configs experiment=dex3_repaired_velocity \
  scene.world_count=2 output_dir=/outputs/diagnostic
```

The diagnostic records every source frame using the upstream full-sequence
rollout and scoring implementation. It does not load or train a learned policy.
Newton runs at source speed 1.0; the archived Isaac diagnostic used speed 0.5.
The contact engine and wrist controller differ. Verify this baseline independently
before expecting the Isaac result to transfer.

After a viable baseline, run `train.py` in a fresh output directory. The recorded
full-task recipe in `compose.repaired.yaml` preserves checkpoints at iterations
0/250/500/750/1000. That recipe reproduces the legacy experiment with its own
matching image and inputs; do not mix it with the corrected force-plan pack.
Evaluate initial and trained policies using the same exact inputs,
geometry and official kit runner; compare task success and achieved object poses.
Reward improvement alone does not accept the policy. No new successful training
result is claimed by these commands.

The private input packs remain local. Team members need a separately transferred
copy and matching manifest hashes; pulling the image and cloning the repository
alone does not provide episode data.
