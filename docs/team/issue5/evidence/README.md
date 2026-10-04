# Recorded repair evidence

These are small October 3-4, 2026 CPU validation and actual local GPU control
records. They include sampled GT-relative diagnostics, without establishing full
fixed-world object-pose accuracy or a policy rollout.
The follow-up records below also document a user-reported tracking failure;
that run has not been reproduced or verified fixed.
The JSON/XML records and audit snapshot preserve their original bytes across
checkouts so the published evidence SHA-256s remain valid on Windows and Linux.

- `tests-pr.xml`: full portable-input test run, 109 passed, zero skipped.
- `validation_snapshot.json`: tested environment, commands, kit and evidence hashes.
- `asset_download_manifest.json`: pinned public dataset, advertised sizes and verified
  SHA-256s for ten meshes and episode-0/2 stereo videos; binaries remain external.
- `local_results.json`: actual full-mesh, decoded-frame, rectification and eight-frame
  whole-image stereo statistics with explicit evidence limits.
- `COMPLETION_AUDIT.md`: original criteria, proposed gates and missing artifacts.

Follow-up at tested software commit `ea38fff4c70445343ad5e5cfcb8acd8508d872d1`:

- `tests-review.xml`: 133 passed, zero skipped, including 24 additional controls
  relative to the original run. The detector/model outputs in these controls are
  generated fixtures; actual GPU perception was not run in that earlier test run.
- `review_validation.json`: source Git-blob hashes, environment, commands,
  results, evidence hashes and remaining work.
- `review-input-results.json`: full expected fourteen-asset verification under
  `python -O`, with pinned calibration/GT hashes, ten meshes and eight stereo
  frames checked. No episode-39 tracking accuracy is measured.
- `tracking_video_review.json`: provided clip hash/metadata and sampled caption
  readings; original command/commit/inputs remain unknown. The clip itself is
  retained locally. See [../TRACKING_FAILURE.md](../TRACKING_FAILURE.md).

October 4 public episode-39 frame-0 observation control:

- `episode39_input_manifest.json`: pinned hashes/bytes for the two public stereo
  videos, dustpan CAD and calibration actually used in this control.
- `episode39_mask_annotations.json`: manual rectified pan/table outlines and
  handle-hole annotation; neither GT poses nor model-generated segmentation.
- `episode39_mask_control.json`: measured bounded CPU registrations on the same
  SGBM depth/CAD, full projected silhouette IoU against the manual pan, source/input/
  output hashes, calibration/rectification, environment and explicit limits. The
  table-inclusive fit passes the native residual gate despite low target overlap.
  Replay with `scripts/eval/replay_issue5_pan_mask_control.py`; masks/overlays remain
  local and are regenerated into a fresh directory. No FoundationPose inference,
  independent camera estimation, official pose accuracy or full tracking is measured.

Subsequent October 4 actual local GPU control, tested native software `45a9fc5`:

- `episode39_models.json`: fixed model revisions and verified hashes/bytes for
  DINO/SAM2 weights and processor data, 1,832,251,936 bytes; binaries remain external.
- `episode39_native_probe.json` and `episode39_noun_probe.json`: actual fixed-frame
  detector/SAM results. The requested pan label selects a tabletop; the standard
  noun control has zero detections at unchanged thresholds.
- `episode39_sam2_tracking.json` and `episode39_prepared_masks.json`: all 319 mask
  hashes and explicit one-box RGB initialization, model/producer provenance,
  calibration/rectification and source-frame records. No pose GT is used.
- `episode39_registration_report.json`: the actual native frozen-mask run,
  accepted/rejected observations, source/runtime and input provenance.
- `episode39_gpu_validation.json`: exact software/commands, 319 raw pose hashes,
  residual/increment summaries, evidence hashes, verification and limitations.
- `episode39_pose_overlays.json`: eight z-buffer/depth/mask comparisons and the
  locally retained control-video hash/metadata. Silhouette overlap is not official
  pose error; camera motion and independent depth truth are absent.
- `tests-gpu-control.xml`: 145 targeted CPU tests passed, zero skips; generated
  fixtures cover mask integrity/provenance and native gap behavior. Actual model
  execution is evidenced separately above.

See [../GPU_CONTROL.md](../GPU_CONTROL.md) for portable replay and remaining gates.
The semi-automatic observation control does not close all-20 world-pose acceptance.

Subsequent October 4 pot/lid control, software `f8bc24f`:

- `pot_native_observations.json`: actual DINO/SAM2 both-view boxes/masks and native
  scale-1 left-camera fits at eight fixed frames per episode 2 and 0; all 64 mask
  hashes and all 32 fit records reproduce the initial control exactly.
- `pot_language_controls.json`: 30 actual predetermined single-phrase controls;
  none is promoted as a production prompt fix.
- `pot_relative_evaluation.json`: evaluator-only public GT sources/visibility,
  frozen inference binding, both pair directions and direct participant RPE
  kernel results; no policy-rollout score or full-episode accuracy claim.
- `pot_registration_report.json`: actual native one-frame CLI report with
  persisted individual detector boxes, scores, assignments and mask counts.
- `pot_validation.json`: source/commands/runtime/receipt hashes, current tests,
  exact-replay checks and explicit remaining gates.
- `pot_masks_preview.jpg`: sampled both-view masks and left-view CAD contours;
  green pot / magenta lid / yellow CAD. No right-camera pose fit is shown.
- `tests-pot-control.xml`: 145 CPU tests passed, zero skips at this source;
  strengthened candidate capture/reset/report controls use generated fixtures.

See [../POT_CONTROL.md](../POT_CONTROL.md). Sparse relative measurements are
conditional on common visible accepted observations; missing cases are retained.

October 4 episode-2 RGB-prompted / external-lid-exclusion follow-up:

- `pot_prompted_mask_observations.json` and `pot_prompted_mask_relative.json`:
  the first box/point control before lid exclusion; all eight accepted fits still
  fail canonical relative orientation. This original frozen record retains
  inherited native baseline metadata; its `producer`/`annotations` identify the
  actual RGB-prompted acquisition. The published portable control clears the text
  prompt and records camera axes explicitly.
- `pot_visible_mask_observations.json`: eight frozen no-GT camera fits, exact
  RGB-only point/box annotations, unchanged native depth hashes, SAM2/model/source
  provenance, mask hashes and pre-exclusion overlap counts.
- `pot_visible_mask_relative.json`: separate frozen-prediction GT evaluation;
  eight common visible accepted samples, positional RPE kernel 29.00 cm and
  separate median canonical relative rotation error 165.95 degrees.
- `pot_visible_mask_validation.json`: executed helper/native source hashes,
  unchanged depth/geometry verification, all sixteen prompted-mask replay checks,
  runtime/commands, matched-six-frame comparisons and explicit limitations.
- `pot_visible_mask_preview.jpg`: four sampled left-view masks/CAD contours;
  green pot / magenta lid / yellow CAD. The CAD handle remains misaligned in
  several uncovered-pot frames. All eight local overlays were reviewed.

The masks are Codex-reviewed RGB prompts, not independent human mask truth.
FoundationPose is proposed initialization/recovery and has not run. These
observations do not establish continuous tracking, world pose or all-20 acceptance.

The original local snapshots and previews remain archived in the author's workspace.

October 4 handle-preservation / competing-pose follow-up:

- `pot_local_density_observations.json` / `pot_local_density_relative.json`:
  both objects refit on eight frozen episode-2 inputs; RPE 14.36 cm and canonical
  relative rotation median 122.37 degrees.
- `pot_pose_support_observations.json` / `pot_pose_support_relative.json`:
  all 25 pot candidates retained per frame, native geometric replay, selected
  visible-support hypothesis and frozen lid; RPE 7.09 cm, relative translation
  median 2.55 cm, canonical relative rotation median 46.81 degrees. Failures stay.
- `pot_pose_support_render.json` / `pot_pose_support_preview.jpg`: eight full-mesh
  overlays against the frozen masks, with prediction and output hashes.
- `pan_density_registration.json`, `pan_density_poses.json`,
  `pan_density_overlays.json`, `pan_density_preview.jpg`: actual 319-frame density
  regression and sampled camera-fit overlays, with no GT/world-pose accuracy claim.
- `cloud_pose_validation.json`: source/input/prediction/test bindings, exact replay
  receipts and comparison limits. `tests-pose-support.xml`: 150 passed, zero skips.

Portable commands and limits: [../CLOUD_AND_POSE_SUPPORT.md](../CLOUD_AND_POSE_SUPPORT.md).
Use the portable commands in [../HANDOFF.md](../HANDOFF.md) to regenerate evidence
on a teammate's checkout. Source revisions are identified there; these snapshots
are not an assertion that later source or data revisions have been validated.
