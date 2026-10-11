# Automatic hand recovery: development and held-out evaluation

The candidate is frozen in `automatic-hand-method-freeze-v1.json`, after clean
development runs on episodes 12 and 22. Held-out evaluation is running. It has
not fixed flickering across all episodes or passed pose-accuracy acceptance.
Enlarged development review found visible fingertip errors among consistency
passes. This frozen candidate is not approved for generating contact targets.
The freeze also pins `automatic-hand-runtime-lock.json` and
`automatic-hand-input-lock.json`; preserve their exact bytes.

## Split and order

Public episodes: 0, 2, 11, 12, 13, 18, 21, 22, 23, 24, 26, 27, 31, 39, 40, 41,
42, 43, 44, 45. Episodes **0, 11, 23, 31, 39** are the previously reserved repair
holdouts. Their historical baseline processing does not make them never-seen
data; the valid claim is no tuning of this recovery method on their results.
Do not read their recovery results until the method is frozen.

Use development episodes to debug automatic detection, fitting, camera conversion,
image/depth consistency and motion checks. Preserve failed trials. A lower
visibility-transition count with fewer usable hands is not a successful fix.

Before the holdout run, freeze all algorithm files, configuration, model assets,
source revision and runtime versions. Record the actual development episodes and
all settings. Run the same method on all five holdouts, without frame-specific
overrides, changing confidence thresholds, or selective retries. Infrastructure
repairs must be recorded separately. Any result-informed method change consumes
the holdout: subsequent runs must be called reused-holdout diagnostics.

## Current candidate

The candidate performs current-image detection at full view and overlapping
upright tiles. A development trial using rotated detection produced a duplicate
left/right assignment to the same visible hand and was rejected. Current code
rejects conflicting cross-side assignments instead of counting both as observed.
The revised search also checks horizontal reflections, mapping both boxes and
handedness back to the original image. Each view is inferred independently so
letterboxing cannot change with the shapes of other crops in a batch.
Image motion can propose a new search crop, but the detector must independently
detect a hand in the current crop. Wearer selection uses the existing automatic
depth and motion association. No episode ID, hand-picked frame, annotated crop,
contact score or object ground truth enters detection or fitting.

Fresh HaMeR inference fits current images. Native virtual-camera projections are
converted to measured camera intrinsics with rigid PnP. The candidate does not
copy the e012 repaired fits. It retains raw proposals before quality filtering.

The quality stage compares each mesh with a separately predicted current-image
SAM2 mask, valid stereo depth and forward/backward optical flow. Incorrect fits
remain explicitly rejected. No hidden hand pose is fabricated to hide a dropout.
These are consistency proxies, not 3D hand ground truth. Mask agreement alone can
approve the wrong shape, and segmentation itself can fail.
Only foreground occlusion supported by measured scene depth can exclude rendered
mesh pixels from silhouette precision. Fully occluded fits cannot pass as observed.

## Accuracy and flicker evidence

For every episode and each hand, report proposals, accepted consistency checks,
supported frame IDs, missing intervals, longest unsupported gap, visibility
transitions and reasons for rejection. Report mask precision/recall, depth
residuals, and pose changes inconsistent with image motion. Compare against the
same source frames in the historical baseline; distinguish changes to coverage
from changes to placement.

Review source-video overlays at normal speed and inspect every reacquisition,
long gap, large motion discrepancy, identity ambiguity and visible holding
interval. Record whether the wrist, fingers and handedness agree with the image.
Occluded fingers remain unverified; a depth residual cannot establish their
accuracy. Keep ambiguous contacts unknown. The existing
`reconstruction/validation/issue6/QUALITY_ACCEPTANCE.md` criteria still apply.

## Reproduction contract

Teammates must use matching source images/calibration/depth, model weights,
method freeze and runtime configuration. Preserve raw detections, selected crop
provenance, raw fitted geometry, quality measurements and rejected observations.
Compare frame-level support masks and numerical geometry, not only total counts.
The prior frozen e012 reproduction pack remains immutable and separate from this
new automatic-inference experiment.

## Running the candidate

Use Linux with a CUDA GPU and EGL rendering. The development machine used WSL2
and an 8 GB RTX 5060 Laptop GPU. `automatic-hands-runtime.json` records selected
installed versions; this is an environment inventory, not a validated clean
installation recipe. HaMeR source is ZhengdiYu/hamer at
`2a5b8ee852d87fdb56d65df7fb0e473c0d4f7648`. Obtain licensed MANO assets yourself.
Do not substitute different model weights while calling a run a reproduction.

The input cache for each `eNNN` contains `issue6-full-inputs/eNNN/manifest.json`,
`color/000000.png`, optional `valid/000000.png`, and metric floating-point depth
in `issue6-full-runs/all20-final-v3/eNNN/depth/000000.npy`. Depth must share the
color image pixel grid. PNG depth from the older replay kit is not this input
format. This experiment reuses existing calibrated stereo outputs; it does not
claim a fresh stereo reconstruction or clean video-to-simulation reproduction.
To regenerate upstream inputs, follow the rectified stereo-left adapter and
depth-stage contract in
[`v2d_issue6/README.md`](../../../reconstruction/modules/v2d_issue6/README.md).
Use full episodes and the original calibration/timestamps, then compare the
resulting image/depth/validity files to the checked-in input lock. A mismatch is
a different input experiment, even when the source video filename is the same.

Set paths for your machine, then from the repository root run:

```bash
python reconstruction/validation/issue6/run_automatic_hand_suite.py \
  --episodes 12 22 --cache-root "$CACHE" --out "$DEV" \
  --detector "$DETECTOR" --hamer-source "$HAMER_SOURCE" \
  --hamer-weights "$HAMER_WEIGHTS_ROOT" --sam-checkpoint "$SAM"
```

The HaMeR weights root contains `_DATA/hamer_ckpts/checkpoints/hamer.ckpt`,
`_DATA/hamer_ckpts/model_config.yaml`, `_DATA/data/mano/MANO_RIGHT.pkl`, and
`_DATA/data/mano_mean_params.npz`. Set `PYOPENGL_PLATFORM=egl` and use matching
runtime dependencies. The existing trusted HaMeR checkpoint needs
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` with the recorded PyTorch version; do not use
that setting with untrusted checkpoints. Output directories must be new.

After reviewing the development evidence, freeze before running the holdouts:

```bash
python reconstruction/validation/issue6/freeze_automatic_hands.py \
  --receipts "$DEV/e012/proposals/receipt.json" "$DEV/e022/proposals/receipt.json" \
  --quality-reports "$DEV/e012/quality/summary.json" "$DEV/e022/quality/summary.json" \
  --runtime-inventory "$RUNTIME" --input-inventory "$INPUTS" \
  --out "$FREEZE"
python reconstruction/validation/issue6/run_automatic_hand_suite.py \
  --episodes 0 11 23 31 39 --cache-root "$CACHE" --out "$HOLDOUT" \
  --detector "$DETECTOR" --hamer-source "$HAMER_SOURCE" \
  --hamer-weights "$HAMER_WEIGHTS_ROOT" --sam-checkpoint "$SAM" --freeze "$FREEZE"
```

The runner rejects held-out episodes without a freeze and rejects method/model
hash mismatches. Keep the completed suite receipt, per-episode receipts, freeze,
runtime inventory, input hashes, and model hashes with the results. A freeze
identifies a candidate for evaluation; it does not certify accuracy.

Run `inventory_automatic_hand_inputs.py --cache-root "$CACHE" --episodes ...
--out "$INPUTS"` before evaluation. Repeat with a new output and
`--compare "$INPUTS"` afterward and on the reproducing machine. This additional
inventory covers validity-mask contents and absent/extra input files; proposal
receipts alone do not pin optional validity masks. Both checks are required for
the reproduction claim.

Use a normal Git checkout so `.gitattributes` restores the byte-hashed source
line endings. A raw `git archive` may contain normalized blob line endings;
when using an archive, restore the exact method files from the run's `source/`
snapshot before checking the freeze. Do not bypass a method-hash mismatch.
`inventory_automatic_hand_runtime.py --out "$RUNTIME"` records installed package
versions and hashes of SAM2, detector and MANO implementation source files.

```bash
python reconstruction/validation/issue6/report_automatic_hands.py \
  --episode-roots "$HOLDOUT/e000" "$HOLDOUT/e011" "$HOLDOUT/e023" \
                  "$HOLDOUT/e031" "$HOLDOUT/e039" --out "$REPORT"
python reconstruction/validation/issue6/review_automatic_hands.py \
  --source "$CACHE/e000/issue6-full-inputs/e000" \
  --proposals "$HOLDOUT/e000/proposals" --quality "$HOLDOUT/e000/quality" \
  --out "$REVIEW"
```

Review videos show source and proposals side by side. Yellow means rejected;
blue/red mean left/right consistency pass, not verified 3D accuracy. Raw geometry
and every rejection are retained. Only consistency-passing native MANO records
are written to `quality/accepted/records`, with `quality_accepted` still false.
They require downstream reconstruction and contact validation before becoming
robot motion targets. No automatic method in this experiment has policy-training
or physical-grasp acceptance.

For an independent rerun of the same episode, use a new suite output directory
with the same freeze. Run `compare_automatic_reproduction.py --reference
"$DEV/e012" --candidate "$REPEAT/e012" --out "$COMPARISON"`. It requires identical
frame-level proposal/support decisions and maximum geometry difference no larger
than 0.1 mm. This is a numerical reproduction tolerance, not a pose-accuracy
threshold. Input inventories must also match. Use
`check_automatic_hand_export.py --quality "$REPEAT/e012/quality" --mano-assets
"$MANO_ASSETS" --out "$EXPORT_CHECK"` to verify native-record conversion with the
existing reconstruction renderer.

For enlarged accuracy-review crops, run `audit_automatic_hand_crops.py` with the
same `--source`, `--quality`, `--proposals`, and a new `--out` directory. It records
its deterministic frame selection. Inspect fingertips and handedness against the
source, including numerically accepted fits. Record visible mismatches separately
from unobservable fingers; a consistency pass must not overrule a visible error.
