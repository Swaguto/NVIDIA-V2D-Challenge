# Earlier scored baseline development — 18 October 2026

This is the preserved development log. Counts and pending gates below describe
earlier checkpoints. See [current status](SCORED_BASELINE_PROGRESS.md) for the
completed full-roster artifact and current failures.

The complete Track 3 release is **not ready**. Development runs use reconstructed reference trajectories; their local evaluator results are not leaderboard scores. No successful physical task or unseen-episode generalization has been established.

## Current coverage

The latest [slot-by-slot inventory](scored-coverage-progress-20261018.json)
accounts for all 20 episodes, 36 mesh slots and 5,181 source frames. Seven
episodes (e001/e003/e005/e006/e008/e010/e038) have completed short baselines and
13 own candidate mesh slots have verified source closures, metric units, fixed
body origins, object ordering and full bundle frame axes. The other 23 mesh
slots remain pending with an explicit implementation owner. **No physical task
has passed.** e008 contains both a beige cup and a pot.

The latest [measured-geometry experiments](MEASURED_GEOMETRY_20261018.md) repair
the calibrated pot seed but expose later native reference collisions. Future
policy preparation audits every reference knot before PPO. A first-frame spoon
diagnostic and an e008 pot-transfer comparison remain experiments; they are not
accepted geometry or generalization results.

- All 20 scored episodes' 60 source videos and the shared camera calibration were downloaded at a pinned dataset revision and hash-verified. No scored object pose tables were read.
- Independent stereo camera trajectories and hand observation inputs are prepared for all 20 episodes.
- Three pilot episodes (e001 pot/lid, e010 cups, e038 dustpan) have complete hand fits, own SAM3D meshes, object tracks, native hand alignment, full-episode retargeting, portable inputs, and six short trained policies.
- Each of those six policies ran 100 PPO iterations, 153,600 environment steps, seed 42. Visual/metric anchor variants preserve identical object targets, clock, confidence, geometry and robot assets. The robot commands and their derived FK targets are the changed factor.
- All 20 neural hand fits now exist, covering all 5,181 source frames with fixed stereo cameras and fixed sequence hand shape verified. The remaining batch's 35,317 retained files were hash-verified on the operator's Mac before provider STOPPED was confirmed. Observation gaps remain unknown; finishing a fit does not establish tracking accuracy. See `all-scored-hand-fit-coverage-v1.json`.
- e003 additionally has its own pot and wooden-spoon mesh candidates. Export and hand-prior recovery, short training and full native evaluation are complete; these are not accepted geometry or grasp results. e005/e006/e008 also completed the complete automatic driver. A complete 20-episode/36-mesh submission has not been packed.

## First failures found

The old partial-depth meshes could produce invalid native collision geometry. The clean SAM3D dustpan now passes native compilation and short PPO training. This fixes that execution failure, not grasp success. The beige cup's wire handle remains incomplete.

The clean e010 beige-cup track loses measured support at frame 45. Its report contains 45 observed and 116 inferred world-hold frames. Synchronized source overlays show that this held target misses the demonstrated lift. Longer policy training against that target cannot repair the missing demonstrated motion. e001 also contains substantial inferred object pose spans; e038 object tracking covers all 263 frames but its short policies still fail the physical task.

Restricting robot command anchors to metric hand observations changes initial poses and parks unsupported hands, but is not a universal improvement. It remains an opt-in experiment. Source observations and contact unknowns stay unchanged.

`foundationpose_track.py --foreground-depth-only` is a new **unvalidated** single-factor experiment: retain only the measured target-mask depth for pose refinement. The independent depth-support and pose-jump acceptance gates still use the original data. The default behavior is unchanged. Tracking caches record and bind this choice so experimental results cannot silently reuse the old trajectory.

`--foreground-translation-prior` is a separate **unvalidated** initialization experiment. It uses changes in measured foreground surface position to help the tracker follow object motion and reacquire it after a rejection. The median surface point is not declared to be the body origin. It retains the original jump-check reference, the original measured-depth acceptance data, and the reconstructed body gauge. Neither experiment is promoted merely by increasing the count of accepted poses; source overlays must improve.

The recovered e001 visual/metric 100-iteration policies both have local MP-SR zero. Their AUC values are approximately 0.0213 and 0.1047 respectively against their reconstructed targets. These do not establish successful pot handling or hidden-ground-truth improvement.

## Reproduction work and controlled repeats

The committed host runners are `run_scored_hands.py`, `run_scored_reconstruction.py`, and `run_scored_policy.py`. They cover video observations through complete hand fitting, own object reconstruction, calibrated alignment, retargeting, portable collision assets, a short PPO baseline, official evaluation, and separate unassisted native readback. They use immutable images and explicit private mounts; operational compute guards remain outside these runners. See the Track 3 scripts README for configuration.

Fresh CPU cloud IK matches all three Mac pilots numerically (maximum robot-position difference approximately 1.24e-12). Integer/confidence arrays are exact. Cloud case and portable-pack preparation also passed. Subsequent complete GPU driver runs on e005/e006/e008 validated execution of reconstruction through short policy evaluation, while all three physical tasks failed.

The e002 checkpoint-2000 control is retained. Two distinct exploration-key forks (101 and 202) preserve every other learner array and counter exactly, and pair control versus short training-reset settling. All four 100-iteration continuations and three eight-world unassisted readbacks per case are complete: 3,276,800 new environment steps. Short settling shows a modest paired MP-SR benefit and avoids one large-error repeat, but its AUC improvement does not exceed observed repeat variability or consistently beat the retained parent. No candidate is promoted. These are distinct exploration streams from the same learner state, not independent actor initializations. [Results](RESET_CONFIRMATION_20261018.md).

The complete hand runner passed a fresh e010 GPU integration. The reconstruction runner's first e003 run completed masks, two own meshes, object tracking and world conversion, then failed because relative camera paths escaped the mounted observation namespace. A copied manifest now relocates only those paths after proving each camera JSON byte-identical. The formerly failing bind stage and complete exports subsequently passed. All 12,521 retained recovery files were hash-verified before provider STOPPED.

The e003 end-to-end audit then found a second integration defect: native alignment had 252/125 metric hand frames, but continuity repair looked in a missing default conversion directory. It silently loaded no priors, erased both hands, and produced a parked-hand policy. That policy is disqualified. The fix passes the actual raw conversion directory explicitly and preserves the cropped frame mapping (310 fitted indices map to source frames 3–312). Missing/empty priors now fail immediately; policy training refuses zero command anchors.

The GPU repair completed: 310 left and 309 right visual frames and command anchors, complete exports, 100 PPO iterations / 153,600 environment steps, initial/final official evaluations and an eight-world native readback. Its 5,795 retained files were hash-verified before STOPPED at 07:48:58 UTC. The next non-reserved cohort (e005/e006/e008) started only after that closure. It completed the three-hypothesis tracking driver with source `848f408178e4d7e3b1bcbdcc1404a64325303307`; all 35,383 retained files were verified before STOPPED at 08:36:51 UTC. All three MP-SR results are zero.

The repaired e003 policy still fails: initial/final AUC is 0.01505/0.01493, MP-SR remains zero, and training repeatedly terminates for object error after one step. The native convex hull envelope lies 28.4 mm (pot) and 34.6 mm (spoon) below the inferred horizontal support top at the reference start. This is an envelope diagnostic, not an exact intersection test. The first recorded native spoon pose is 97.7 mm and 66.7 degrees from its reference. Its complete readback retains nonfinite late objective terms/scores, although body/contact arrays, checkpoint tensors and training metrics are finite. The reward-balance audit correctly refuses to summarize that nonfinite rollout. Source geometry/contact initialization needs investigation before a longer run; the checkpoint is not promoted.

A subsequent CPU MuJoCo contact probe explicitly resets both free objects to
their exact reference start and verifies their positions/quaternions before
contact detection. It confirms 21.75 mm pot/support and 14.77 mm spoon/support
penetration. No physics step or robot-reference reset is performed; robot
contacts are excluded. `audit_initial_contacts.py` records this separate geometry
diagnostic before future short policy runs. It does not relax native mesh checks
or declare a physical pass. An experimental measured-surface pot candidate
improves source silhouette overlap from 0.497 to 0.744 and median measured
point-to-mesh-vertex distance from 53.2 to 10.7 mm, with 2 mm seed clearance above
the unchanged support plane. It has not passed full-sequence tracking or physics;
its above-table constraint currently applies to this diagnosed pilot only.

An experimental whole-track selector compares the control, foreground-depth and foreground-motion trajectories against the same measured source masks/depth. It selects one entire unchanged trajectory per object; missing observations score zero. On the two inspected pilots it selected the original pot control, the foreground-motion lid/beige cup, and the foreground-depth blue cup. Both cups now retain 161 observed frames and the lid 335. Video review still shows silhouette/orientation errors. Its complete driver integration passed on the next three episodes, but this establishes execution rather than physical acceptance or hidden accuracy. Tuning choices remain unfrozen for the reserved episodes.

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

The [current scored setup](../../../docker/track3-scored/README.md) now includes
immutable runtime digests, public upstream input metadata, hand/robot asset
checksums, video-only acquisition and the final official-packer staging command.
A clean input-directory rehearsal verified all 61 selected files without CAD or
pose tables. It reused verified source bytes and does not establish a fresh
policy or teammate reproduction.
