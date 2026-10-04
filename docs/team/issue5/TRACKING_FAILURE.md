# Tracking failure reported after draft PR #30

**Real tracking remains unresolved.** The user reports that the team lead tested
the draft and obtained the same incorrect tracking. The supplied evidence is
`pan_cad_diagnostic.mp4`, a CAD-ICP overlay labeled PAN, episode 39. The generating
command, source commit, detector outputs, masks, depth and pose JSONs are unavailable.
This prevents attributing the rendered failure to a particular version of the pipeline
or reproducing that exact run. The initial 109 CPU tests did not establish real tracking.

## What the clip establishes

The clip contains 319 frames at 20 FPS, 1280 x 440 pixels, lasting 15.95 seconds.
SHA-256: `ccd46524af5db070d108f3d612a6a89dcc9d1127f69f2008b09f7eb9c9885964`.
The original video and extracted frames are retained locally; this PR records the
observations and hash without adding the video binary.

The caption explicitly labels green as the mask. At frame 0 that outline surrounds
the round tabletop, rather than isolating the small dustpan resting on it. The yellow
CAD is displaced across the tabletop. Sampled later frames retain the large tabletop
outline as the actual dustpan moves/lifts. The displayed IoU at sampled video frames
0, 39, 79, 119, 159, 199, 239 and 318 is respectively
0.15, 0.16, 0.21, 0.16, 0.17, 0.17, 0.15 and 0.13.
These are readings of the rendered caption, not independently computed metrics;
the original IoU definition and inputs are unknown.

The visible mask/target mismatch is strong evidence to inspect observation selection
before trying further quaternion or trajectory filtering changes. If that displayed
mask is the mask used to backproject depth, registration receives mostly table points.
A mathematically correct coordinate conversion cannot turn those into dustpan points.
The clip alone does not determine whether the mismatch originates in detection,
SAM segmentation, mask/frame/axis association, resizing, or the diagnostic renderer.

## Verified code defects addressed in the follow-up

`register_poses.Segmenter._assign` previously assigned **every unrecognized detection
to object 0 when only one object was requested**. A `table` detection could therefore
become the `mini_dust_pan` mask. That fallback is removed. Unmatched labels are retained
in diagnostic records with null object assignment. A generated detector/SAM control
checks that a table-sized mask is excluded while a correctly labeled small pan mask
is retained. This verifies the software behavior; actual DINO/SAM inference was not run.
Whether this fallback caused the supplied clip remains unconfirmed.

The native tracker now writes `registration_report.json` with detector labels/scores,
mask/depth pixel counts, filtered-cloud counts, residuals, rejection reasons, accepted
counts and input-read failures. Zero-solve objects or input-read failures produce a
nonzero exit status. Missing observations remain missing, with reasons recorded.
These changes expose failures; they do not repair an incorrectly labeled pan mask,
establish tracking accuracy, or create an independent world camera trajectory.

The nine verified PR review findings are also addressed: fresh output checks before
right-scale estimation and explicit `--out`; visualizer `--poses`; checks retained under
`python -O`; complete repository-pinned asset/calibration/reference hashes; bounded
local kit reads; portable diagnostic paths; source-ID-to-GT-row matching; stereo probe
imports; and smoothing only declared fixed-world poses after camera conversion.
World smoothing stays inferred, leaves gaps, and does not claim validated residuals.

## Next diagnostic receipt for the run owner

Obtain the original command and commit, episode/camera/frame identifiers, calibration,
rectification and image sizes, detector boxes/labels/scores, raw masks, depth with its
encoding/validity, CAD identity, pose JSONs and the overlay's IoU calculation. Keep the
original failed run immutable. Compare the table-inclusive mask with an independently
reviewed dustpan-only mask on the **same** frozen depth/CAD inputs, beginning at frame 0.
Inspect target-only mask overlap and cloud extent before accepting an ICP pose. Rejecting
bad masks is not a tracking-success result; report coverage and all zero-solve cases.

An operator with an existing compatible CUDA/model environment can use the reviewed
checkout to collect the first-frame receipt:

```bash
export V2D_PUBLIC=/absolute/path/to/track_3/public
export V2D_CALIB="$V2D_PUBLIC/meta/camera_calibration.json"
export V2D_OUT=/absolute/fresh/episode39-mask-probe
python scripts/track3/probe_segmentation.py 39 0
python scripts/track3/register_poses.py --episode 39 --frames 1 --log 1 \
  --out /absolute/fresh/episode39-registration
```

These commands were not run on the shared VM. Archive the probe overlay, console output,
`rectification.json` and `registration_report.json`; one frame is a diagnostic, not an
episode benchmark. Independent cameras and fixed-world scoring are separate subsequent
gates described in [HANDOFF.md](HANDOFF.md). Original issue-5 criteria remain open.
Do not merge this draft to main or close issue 5 on the new CPU controls.
