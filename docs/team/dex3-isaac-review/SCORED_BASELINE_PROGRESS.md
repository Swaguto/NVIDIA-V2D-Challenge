# Scored baseline development — 18 October 2026

The complete Track 3 release is **not ready**. Development runs use reconstructed reference trajectories; their local evaluator results are not leaderboard scores. No successful physical task or unseen-episode generalization has been established.

## Current coverage

- All 20 scored episodes' 60 source videos and the shared camera calibration were downloaded at a pinned dataset revision and hash-verified. No scored object pose tables were read.
- Independent stereo camera trajectories and hand observation inputs are prepared for all 20 episodes.
- Three pilot episodes (e001 pot/lid, e010 cups, e038 dustpan) have complete hand fits, own SAM3D meshes, object tracks, native hand alignment, full-episode retargeting, portable inputs, and six short trained policies.
- Each of those six policies ran 100 PPO iterations, 153,600 environment steps, seed 42. Visual/metric anchor variants preserve identical object targets, clock, confidence, geometry and robot assets. The robot commands and their derived FK targets are the changed factor.
- The other 17 neural hand fits and the remaining 31 own mesh slots are still in progress. A complete 20-episode/36-mesh submission has not been packed.

## First failures found

The old partial-depth meshes could produce invalid native collision geometry. The clean SAM3D dustpan now passes native compilation and short PPO training. This fixes that execution failure, not grasp success. The beige cup's wire handle remains incomplete.

The clean e010 beige-cup track loses measured support at frame 45. Its report contains 45 observed and 116 inferred world-hold frames. Synchronized source overlays show that this held target misses the demonstrated lift. Longer policy training against that target cannot repair the missing demonstrated motion. e001 also contains substantial inferred object pose spans; e038 object tracking covers all 263 frames but its short policies still fail the physical task.

Restricting robot command anchors to metric hand observations changes initial poses and parks unsupported hands, but is not a universal improvement. It remains an opt-in experiment. Source observations and contact unknowns stay unchanged.

`foundationpose_track.py --foreground-depth-only` is a new **unvalidated** single-factor experiment: retain only the measured target-mask depth for pose refinement. The independent depth-support and pose-jump acceptance gates still use the original data. The default behavior is unchanged. Tracking caches record and bind this choice so experimental results cannot silently reuse the old trajectory.

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
