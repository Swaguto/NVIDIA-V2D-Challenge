# Scored baseline development — 18 October 2026

The complete Track 3 release is **not ready**. Development runs use reconstructed reference trajectories; their local evaluator results are not leaderboard scores. No successful physical task or unseen-episode generalization has been established.

## Current coverage

- All 20 scored episodes' 60 source videos and the shared camera calibration were downloaded at a pinned dataset revision and hash-verified. No scored object pose tables were read.
- Independent stereo camera trajectories and hand observation inputs are prepared for all 20 episodes.
- Three pilot episodes (e001 pot/lid, e010 cups, e038 dustpan) have complete hand fits, own SAM3D meshes, object tracks, native hand alignment, full-episode retargeting, portable inputs, and six short trained policies.
- Each of those six policies ran 100 PPO iterations, 153,600 environment steps, seed 42. Visual/metric anchor variants preserve identical object targets, clock, confidence, geometry and robot assets. The robot commands and their derived FK targets are the changed factor.
- All 20 neural hand fits now exist, covering all 5,181 source frames with fixed stereo cameras and fixed sequence hand shape verified. The remaining batch's 35,317 retained files were hash-verified on the operator's Mac before provider STOPPED was confirmed. Observation gaps remain unknown; finishing a fit does not establish tracking accuracy. See `all-scored-hand-fit-coverage-v1.json`.
- The remaining 31 own mesh slots and 17 policy baselines are still in progress. A complete 20-episode/36-mesh submission has not been packed.

## First failures found

The old partial-depth meshes could produce invalid native collision geometry. The clean SAM3D dustpan now passes native compilation and short PPO training. This fixes that execution failure, not grasp success. The beige cup's wire handle remains incomplete.

The clean e010 beige-cup track loses measured support at frame 45. Its report contains 45 observed and 116 inferred world-hold frames. Synchronized source overlays show that this held target misses the demonstrated lift. Longer policy training against that target cannot repair the missing demonstrated motion. e001 also contains substantial inferred object pose spans; e038 object tracking covers all 263 frames but its short policies still fail the physical task.

Restricting robot command anchors to metric hand observations changes initial poses and parks unsupported hands, but is not a universal improvement. It remains an opt-in experiment. Source observations and contact unknowns stay unchanged.

`foundationpose_track.py --foreground-depth-only` is a new **unvalidated** single-factor experiment: retain only the measured target-mask depth for pose refinement. The independent depth-support and pose-jump acceptance gates still use the original data. The default behavior is unchanged. Tracking caches record and bind this choice so experimental results cannot silently reuse the old trajectory.

`--foreground-translation-prior` is a separate **unvalidated** initialization experiment. It uses changes in measured foreground surface position to help the tracker follow object motion and reacquire it after a rejection. The median surface point is not declared to be the body origin. It retains the original jump-check reference, the original measured-depth acceptance data, and the reconstructed body gauge. Neither experiment is promoted merely by increasing the count of accepted poses; source overlays must improve.

The recovered e001 visual/metric 100-iteration policies both have local MP-SR zero. Their AUC values are approximately 0.0213 and 0.1047 respectively against their reconstructed targets. These do not establish successful pot handling or hidden-ground-truth improvement.

## Reproduction work and controlled repeats

The committed host runners are `run_scored_hands.py`, `run_scored_reconstruction.py`, and `run_scored_policy.py`. They cover video observations through complete hand fitting, own object reconstruction, calibrated alignment, retargeting, portable collision assets, a short PPO baseline, official evaluation, and separate unassisted native readback. They use immutable images and explicit private mounts; operational compute guards remain outside these runners. See the Track 3 scripts README for configuration.

Fresh CPU cloud IK matches all three Mac pilots numerically (maximum robot-position difference approximately 1.24e-12). Integer/confidence arrays are exact. Cloud case and portable-pack preparation also passed. The new complete GPU driver integration is queued in the active bounded validation session, not yet established by these CPU tests.

The e002 checkpoint-2000 control is retained. Two distinct exploration-key forks (101 and 202) preserve every other learner array and counter exactly, and pair control versus short training-reset settling. Four 100-iteration continuations with repeated unassisted evaluation are queued. The earlier settling improvement remains unqualified; no seed-confirmed improvement is claimed yet.

Public development episodes 11, 22 and 42 were selected before inspecting new metadata or outcomes. A subsequent history audit found that all three were reviewed in the earlier continuity work, and e022 has a legacy manual object-box override. They are therefore retrospective regression cases, not untouched end-to-end holdouts. Their reference pose tables remain isolated outside inference mounts. Rostered video runs now exclude per-episode manual object prompts and reject annotated mask caches.

Scored episodes e014 (cup/dish rack), e032 (cup/spoon) and e036 (spoon) are reserved for validation after the current tuning choices are frozen. Their camera/hand preparations already ran with unchanged settings; no object reconstruction or PPO results from these three have been used for tuning. No such validation has run yet. It will test the reconstruction/training procedure across new task results, not a single zero-shot policy across arbitrary episodes.

## Diagnostics and interpretation

`audit_reference_geometry.py` verifies bundle/input hashes, complete source frame and timestamp axes, object ordering, and named robot joint order. It compares supported human and robot fingertips with the same own mesh. Human summaries require joint hand-metric and object-pose observations. Inferred object frame counts are explicit.

Unsigned nearest-vertex distance does not prove contact, penetration clearance or force closure. Full-episode summaries include intentional release/idle frames. XML joint traversal order is not the payload's declared joint order; diagnostics and commands must use the manifest's names.

Example (private paths provided by the operator):

```sh
python scripts/dex3_training/audit_reference_geometry.py \
  --bundle "$BUNDLE" --inputs "$FROZEN_TASK" --out "$REPORT"
```

Next gates: verify the tracking experiment against source pixels; complete own geometry and policies for the roster; run the untouched official packer; independently repeat any proposed training improvement; freeze held-out development validation; finish a clean-setup team reproduction. A short successful process exit is never a physical acceptance certificate.

## Compute and privacy

The owner renewed as-needed Brev authorization. Each paid session still has an independent stop guard, bounded stages, verified private result backup, and provider STOPPED verification. Stop compute while idle. Images exclude credentials, footage, MANO and gated model weights. Derived private input packs are not published. A captain-approved leaderboard upload and another teammate's reproduction are separate external gates.
