# Bounded GPU experiments

## Current status

The owner explicitly resumed as-needed project validation on October 17 after
the billing audit. The original historical billing discrepancy is unresolved.
Each current session records that renewed authorization separately, keeps a
wall-clock deadline and independent provider-stop guard, verifies private
backups, and confirms provider STOPPED. This is not a provider spending cap or
evidence of reconciled billing. Account credits alone do not authorize work.

The original profile controls have CPU and saved-artifact verification. New
scored runs retain stage timings and use the existing pinned images and caches;
there is no controlled estimate of actual billed savings from those caches.
The original physical grasp is still unresolved. These controls govern allocation
of compute; they do not establish policy quality or hidden-episode performance.

## Changes

- All four standalone training Compose services set
  `JAX_COMPILATION_CACHE_DIR=/root/.cache/jax` inside their existing named cache
  volume. Services extending the grip configuration inherit it. This works with
  the existing immutable images without rebuilding their dependencies or changing
  the training, physics, checkpoint or evaluator source.
- `scripts/compute/run_batch.py` defaults to offline preflight. It verifies every
  private manifest member, collision cache binding, declared baseline/checkpoint
  files, distinct read-only mounts, fresh outputs, image digest and bounded stage
  schedule. It invokes no Docker, GPU or cloud commands in that mode.
- Execution reuses one named cache across sequential containers, refuses implicit
  image pulls, records stage wall times and sampled GPU memory/utilization, and
  stops on failure, timeout, a new spending freeze or a logged numerical failure.
  There is no automatic retry. Numerical evidence and earlier checkpoints remain.
- Every training chunk requires an evaluation and a passing allocation gate before
  another chunk. The gate checks the input/checkpoint/runtime contracts, identical
  full-episode coverage with at least four worlds, frame-zero resets, zero object
  assistance, and unchanged physics rates/thresholds. It requires an AUC increase
  of at least 0.005 with no regression in SP-SR, MP-SR, CHORD SR, RPE or MPPE.
  These are conservative local allocation rules, not leaderboard pass thresholds.
  They can reject a slow-learning candidate; review that tradeoff before changing
  a new task's plan rather than silently extending failed runs.
- Budget authorization requires a fresh actual usage snapshot, verified billing
  rules, a recorded explicit approval, sufficient allowance for the whole bounded
  batch, and the existing backup-and-stop lifecycle wrapper. A runtime estimate
  is recorded separately from actual usage. It is not a provider-enforced cap.

JAX keys persistent compiled programs by computation, runtime version, flags and
GPU topology. Changing shapes, code or GPU may require compilation again. Keep
the image, world count and shapes stable when comparing repeated runs. See
[JAX persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html).

## 1. Prepare one profiling batch while Brev is stopped

Use Python 3.10 or newer in the checkout. Preparation requires the local private
input pack, retained policy and matching build context described in
[local reproduction](LOCAL_TRAINING_REPRODUCTION.md). Paths below are placeholders.

```bash
python scripts/compute/prepare_profile_batch.py \
  --inputs /private/training-task-estimated-cws-v4 \
  --checkpoint /private/grip-estimated-cws-ppo-v2/ppo/model_2000.safetensors \
  --context /private/build-context-grip-estimated-cws-v4/context-manifest.json \
  --outputs /private/fresh-profile-results \
  --image ghcr.io/swaguto/nvidia-v2d-challenge-dex3-review@sha256:04688b4d2f087d462a4a8fd1428e77523021d908ccb27b637bc1306a6f3a9220 \
  --out /private/profile-plan.json

python scripts/compute/run_batch.py \
  --plan /private/profile-plan.json --report /private/profile-preflight.json
```

This prepares two identical four-world readbacks, with a 330-second batch limit
and **zero training steps**. The first readback may already find a warm cache;
compare measured runs rather than assuming a cold start. Plans contain absolute
host paths. Regenerate and preflight them for the GPU host's restored paths before
execution. Do not copy a Mac plan unchanged to Linux.

## 2. Resolve billing and run the bounded profile

Keep private invoices, balances, approval records and credentials outside Git.
`scripts/compute/budget.example.json` is deliberately non-executable: its approval
and reconciliation flags are false and its unknown metering fields are null.
Fill a private copy only from verified provider usage/rules and actual owner
authorization. Include all account instances and storage in the usage snapshot.
It must be at most 15 minutes old and the approval must not have expired. Account
usage ceiling means an approved cumulative usage amount, not a credit balance.

Run on the already prepared Linux GPU host through the operator's existing
verified backup-and-stop wrapper:

```bash
python scripts/compute/run_batch.py \
  --plan /private/profile-plan.json --budget /private/approved-budget.json --execute
```

The runner removes its active container on timeout/interruption. It **does not
provision or stop the Brev VM**, and its receipt explicitly records that VM
shutdown is unverified. The outer lifecycle owner must bound boot/deployment,
back up and hash-verify new outputs, call `brev stop`, and confirm `STOPPED` from
the control plane, including on a gate decline or exception. Keep the independent
shutdown guard active during that lifecycle. A guard running on a sleeping laptop
cannot guarantee an account spending cap. Do not use this standalone runner as a
replacement for the lifecycle wrapper.

Budget reserve includes the usage snapshot's age, the batch time limit, declared
remaining backup/shutdown time (the `stop_latency_reserve_seconds` field), and
the verified provider minimum/rounding/fixed charges. Reserve the full remaining
lifecycle overhead, with a minimum of 30 seconds for container cleanup. It does
not predict metering delays or another user's concurrent spending. Reconcile
actual provider usage after the batch before allowing a further paid session.

Use `batch-receipt.json`, individual raw logs, sampled memory and evaluation
artifacts to compare cost per identical valid evaluation. Once training is
justified, compare actual billed cost per million valid new environment steps.
Hourly GPU price alone is insufficient. A smaller GPU needs a memory and
throughput check; sampled memory can miss a brief peak, so retain headroom.

## 3. Require progress before a longer continuation

Only prepare another e002 continuation when there is a specific justified change
or experiment to test. The saved longer continuation regressed and is rejected
by the new gate. Prioritize a complete official scored-episode baseline;
e002 readbacks are development evidence, not the official roster result.

For a compatible experiment with an existing full-episode readback:

```bash
python scripts/compute/prepare_training_batch.py \
  --inputs /private/training-task-estimated-cws-v4 \
  --checkpoint /private/grip-estimated-cws-ppo-v2/ppo/model_2000.safetensors \
  --context /private/build-context-grip-estimated-cws-v4/context-manifest.json \
  --baseline /private/policy-grip-estimated-cws-2000-v2/diagnostic \
  --outputs /private/fresh-staged-results \
  --image ghcr.io/swaguto/nvidia-v2d-challenge-dex3-review@sha256:04688b4d2f087d462a4a8fd1428e77523021d908ccb27b637bc1306a6f3a9220 \
  --experiment dex3_grip_estimated_cws \
  --short-iterations 100 --longer-iterations 400 \
  --out /private/staged-plan.json
```

At the recorded 256 worlds × 32 rollout steps, the first chunk is 819,200 new
steps. Another 3,276,800 steps are conditional on the gate. Each chunk saves every
25 global iterations; the total batch limit is 1,170 seconds. Preparation is
offline and does not grant spending approval. The upstream exact-state resume
checks still run inside the immutable training image. Changed experiments need
a new controlled baseline instead of relabeling an incompatible checkpoint as a
continuation.

Passing the allocation gate still requires an independent confirmation and the
official evaluation/packing path. See
[official evaluation compatibility](OFFICIAL_EVALUATION_COMPATIBILITY.md).

## Verification

```bash
python -m pytest scripts/compute/tests -q
python scripts/compute/check_compose_cache.py
```

Tests exercise denied/stale/insufficient budgets before process launch, input
corruption, coverage mismatch, conditional training, regression rejection, a real
child-process timeout, final nonfinite metrics with a zero exit code, and cache
mount construction. Existing saved e002 artifacts also confirm that the 3500
continuation is rejected: AUC 0.87015 → 0.77326, MP-SR 1 → 0, RPE and MPPE worse.
No new GPU work is needed for those checks.

For multi-day idle periods, compare the cost of preserved disks with rebuilding
from verified backups. Stopping avoids compute charges but preserves billed
storage; deletion removes the data and needs separate owner approval.
[Brev lifecycle and billing](https://docs.nvidia.com/brev/concepts/gpu-instances).
