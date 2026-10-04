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

## Real episode-39 mask control, October 4

Downloaded the public episode-39 stereo videos at dataset revision
`5f68335f3acc802033d1e80728c1633197521de8` and reused the verified dustpan CAD.
On rectified `ego_cam_c` frame 0, manually outlined the pan and a table-inclusive
region. Both solves use the same calibrated SGBM depth, uncentered CAD, scale 1,
trim 0.6 and deterministic sampling. No GT object pose or camera trajectory is read.
These annotations approximate the observation failure; they are not the original
team masks. The exact failed team run remains unreproduced.

| Input mask | Mask pixels | Valid depth pixels | ICP residual | CAD silhouette / manual pan IoU | Passes native 50 mm residual gate |
|---|---:|---:|---:|---:|---|
| Table-inclusive | 154,754 | 122,084 | 40.55 mm | 0.164 | Yes |
| Pan-only | 11,358 | 10,028 | 8.12 mm | 0.662 | Yes |

The table-inclusive cloud spans approximately 45 x 36 x 39 cm; the pan-only cloud
spans 15 x 13 x 14 cm after filtering/capping. The wrong observation can therefore
produce an accepted, visibly displaced pose under the current residual gate.
Fixing the observation improves this frame's fit and overlap. This is evidence for
prioritizing target selection and checking silhouette/cloud extent alongside residual,
not proof of full tracking recovery. No new rejection threshold is tuned on this frame.

IoU here is intersection/union of the full projected CAD silhouette and the manually
outlined pan mask, including its handle hole. It is not the clip's unknown IoU definition
or an occlusion-aware visible-surface score. SGBM has no independent depth truth.
The CPU control uses 1,200 CAD samples, a 1,600-point cloud cap, coarse 600/refine 1,200
and three seeds; it is not the native GPU/default-sample episode benchmark.
Rotation/translation accuracy, world-camera motion, temporal behavior and raw coverage
are unmeasured. The 133-test result remains tied to software commit `ea38fff`.

Replay from this checkout with Python, NumPy, CPU PyTorch, OpenCV and trimesh:

```bash
python scripts/fetch_track3_assets.py --episodes 39 --output /absolute/episode39-public
mkdir -p /absolute/episode39-public/meta
cp tests/fixtures/track3_camera_calibration.json /absolute/episode39-public/meta/camera_calibration.json
python scripts/eval/replay_issue5_pan_mask_control.py \
  --data /absolute/episode39-public --out /absolute/fresh/pan-mask-control
```

The replay verifies the actual bytes/hashes of both videos, CAD and calibration
against [evidence/episode39_input_manifest.json](evidence/episode39_input_manifest.json)
before processing. Annotations are in
[evidence/episode39_mask_annotations.json](evidence/episode39_mask_annotations.json).
The measured [evidence/episode39_mask_control.json](evidence/episode39_mask_control.json)
records source/input/output hashes, rectification matrices, environment, parameters,
poses and limits. The command emits both mask PNGs and overlay PNGs in a fresh directory.
The asset fetcher now records its selected episode IDs instead of labeling every
download as episode 0/2; the original episode-0/2 evidence remains unchanged.

## Is 6D pose estimation useful here?

Yes, as a controlled alternative for initialization and recovery. The current ICP
already estimates a 6-DoF object pose: three position and three orientation degrees
of freedom. Adding the term "6D" does not itself change the estimator.

[FoundationPose](https://arxiv.org/abs/2312.08344) supplies learned pose refinement
and scoring for novel objects using a CAD model or reference images. Its
[official demo](https://github.com/NVlabs/FoundationPose/blob/main/run_demo.py)
registers from RGB, depth, intrinsics and an initial object mask, then tracks
subsequent frames from RGB/depth/intrinsics. It may improve initialization or
avoid some purely geometric local minima; that is a method-based expectation,
not a measured improvement on our video. It still needs correct initial target
selection and compatible metric depth/calibration. A table mask remains a faulty input.

This repository already wraps the method in
`reconstruction/modules/v2d_foundation_pose/lib/foundation_pose_tracker.py`.
`track_pipeline.py` invokes that wrapper. The GT-assisted camera diagnostic's
`fit_ego_rig.py --reuse-fp` consumes saved FoundationPose poses as extra camera-fit
initializations; it is not an independent camera producer. `fit_object_pose.py`
does not currently expose that flag. No new object-fitter integration is claimed.

For the next bounded test, freeze corrected masks, rectified RGB/depth/intrinsics,
CAD identity and source frames. Compare current ICP with the existing FoundationPose
camera-pose baseline on the same observations, keeping raw poses and smoothing off.
Start with a small visible/lifting/occluded sample, then run the sequence if it passes.
Report silhouette overlap, visible-depth residual, flips, gaps and raw coverage;
report world translation/rotation against GT only after a shared independent `C(t)`
exists. Preserve both failures and runtime/model/checkpoint hashes. Evaluate optional
FoundationPose initialization of ICP only with an explicit implemented pose-frame
contract; do not silently replace the accepted tracker or experiment.

FoundationPose inference was **not run** here: the local runtime is CPU-only and
compatible CUDA/model checkpoints and shared-VM access are unavailable. Object
camera pose also does not solve egomotion; fixed-world output still needs
`X(t)=inv(C(t))@P(t)` with the appropriate rectification conversion.

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
original failed run immutable. Repeat the mask comparison above on the **original**
frozen depth/CAD inputs and additional moving/occluded frames, beginning at frame 0.
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
