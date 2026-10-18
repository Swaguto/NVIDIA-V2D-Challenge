# Bounded scored-policy continuation

The first scored baseline uses 100 PPO iterations (153,600 environment steps).
More training should be checked in short chunks against the retained control.

`scripts/track3/continue_scored_policy.py --config PRIVATE_CONFIG.json` resumes
the complete learner in the **same immutable training image** and mounts the
original frozen inputs read-only. Configuration fields:

```json
{
  "parent": "/private/policy-e003",
  "kit": "/private/v2d_submission_kit",
  "training_image": "THE_IDENTICAL_IMAGE_DIGEST_FROM_PARENT_RECEIPT",
  "out": "/private/e003-continuation-200",
  "chunk_iterations": 100,
  "max_seconds": 1200
}
```

The runner checks checkpoint counters, input/collision hashes and the image's
build-context hash before training. The existing strict resume contract checks
the experiment and restores actor, critic, optimizer, normalization and RNG.
The simulator resets on continuation; this is not exact simulator replay.

Each chunk retains the parent checkpoint, records only **new** steps, runs three
complete frame-zero eight-world native readbacks per checkpoint, and runs the
untouched kit evaluator for both. It reports overlapping AUC ranges and MP-SR
without promoting a candidate. These repeats measure evaluation variability;
an independent training repeat is still required. Higher tracking metrics alone
do not establish a physical grasp.

Run inside the existing timed compute controller with verified private backups
and provider shutdown. This command does not provision or stop a VM itself.
GPU validation of this continuation path is pending; its local preflight and
comparison gates must not be represented as a successful training run.
