# Issue 6 reconstruction and contact quality acceptance

Issue 6 was reopened on October 7, 2026 at the user's request. PR32 is draft.
The completed v3 runs, integrity checks and backups remain useful baseline
evidence. Their previous visual passes do **not** establish the quality criteria
below. All 20 episodes must pass individually before issue 6 can close.

## Frozen engineering criteria

These are project acceptance targets, not measured results or a claim of
ground-truth contact accuracy. Freeze them before evaluating repairs on holdouts
0, 11, 23, 31 and 39. Do not adjust thresholds to make an episode pass.

1. Annotate each visible holding interval from source video with its hand,
   object, original frame IDs and visibility. Annotate lift, sustained hold,
   transfer/pour/rotation and release where present. An entire episode or task
   cannot be omitted because reconstruction is difficult. Fully hidden or
   out-of-view periods require explicit source-video evidence and remain unknown.
2. At least **90%** of the visible holding frames in **each hand/object interval**
   must have usable, correctly identified, stereo-supported reconstructed hand
   geometry and an observed CAD pose. No unsupported gap may exceed **0.50 s**.
   Report source-axis coverage separately; an aggregate cannot hide a bad grasp.
3. At least **80%** of frames in each visible holding interval must contain a
   supported **raw** fingertip contact with that object under the unchanged
   strict 2 cm rule. Bridged contacts are reported separately and do not satisfy
   this evidence target. Every interval needs actual evidence, including intervals
   shorter than 0.50 s. If a visible grasp is palm-only or does not support this
   fingertip criterion, document it and implement an appropriate measured contact
   representation; do not silently redefine the interval or fabricate fingertips.
4. Inspect normal-speed complete videos and the start, middle and end of every
   holding interval, every long gap and every reacquisition. Verify hand identity,
   visible finger/mesh correspondence, object silhouette/pose, plausible contact
   locations and normals, and absence of unexplained drift, jumps or penetration.
   Use additional source-frame checks whenever these samples expose a failure.
   Known gross geometry or identity failures prevent acceptance even with good
   coverage. Record findings and the exact reviewed media/configuration hashes.
5. Verify contact onset and release against observable source-video events. A
   visibly released hand must not retain contact through a bridge; ambiguous
   events remain unknown. This is independent image-event evidence, not precise
   3D contact ground truth. Report remaining uncertainty explicitly.
6. Keep the final stereo residual strictly below **2 cm**, preserve track identity
   across bridging, and never bridge invalid/unknown observations. Preserve raw
   fits and failures. Each accepted output needs six current stage receipts,
   geometry and contact audits, input hashes, three complete videos and a verified
   private backup. Earlier configuration receipts cannot accept a repaired run.

## Required per-episode record

Record hand/object holding intervals, eligible and supported frame counts,
usable coverage, raw-contact coverage, longest unsupported gap in seconds,
dropout reasons, geometry/identity findings, onset/release findings, media/config
hashes, remaining limitations and final pass/fail. Missing annotations or review
evidence mean **pending**, never pass. Overall completion requires 20/20 passes.

## Repair order and constraints

Start with episode 22 and other sustained visible-holding gaps. Diagnose missing
detections/association, mask overlap, usable stereo pixels, CAD occlusion,
translation/scale alignment and final depth residuals separately. In particular,
test whether projecting the entire CAD silhouette as an occluder incorrectly
removes visible foreground hand pixels. That is currently a code-level hypothesis,
not a confirmed measured cause. Reuse immutable fits and rerun affected stages.

Use local private evidence first. Cumulative compute/storage authorization remains
US$30, increased by the user on October 7; refresh billing and enforce a stop
deadline before paid compute. This supersedes the previous US$15 authorization. Do not
delete preserved volumes or publish footage, model weights or licensed MANO assets.
Do not claim robot-training success or exact contact precision from this work.
