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
submission eligibility. The current kit requires a mesh for every object in
the combined submission, including for registration of the five pose metrics.

The first camera pilots are e001 (pot/lid), e010 (two cups), and e038 (one dust
pan). Their stereo camera trajectories cover 335, 161 and 263 frames respectively,
with no held camera frames under the existing PnP checks. Camera checks do not
establish correct object tracks, reconstructed geometry, retargeting or policies.
All-episode camera processing is being recorded separately.

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
