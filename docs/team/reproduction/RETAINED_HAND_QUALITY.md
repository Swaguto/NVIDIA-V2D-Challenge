# Retained hand recovery: reproducible development checks

This method checks current-image support for directly observed temporal hand
estimates that the earlier global object-occlusion gate discarded. It preserves
the retained MANO scale and uses independent current-frame wearer detections to
bound segmentation. It supplies consistency-passing **proposals**, not accepted
hand accuracy, robot contact targets, or training data.

This is separate from the e002 Dex3 baseline in PR #35 and from the older e012
display-continuity replay. Observed human poses, display-held human poses and
inferred robot poses must remain separate. A display hold is never an observation.

## Changes to the existing GitHub method

- Bound SAM masks by the independently detected hand box. Preserve SAM confidence
  and logits. Keep the complete rendered hand in silhouette/depth checks, so an
  incorrectly placed or oversized mesh still fails.
- Preserve a retained nonunit `hand_scale` when writing native records. The old
  writer reset it to 1.0, changing the geometry after it had passed the checks.
- Reconstruct each exported MANO hand independently and compare both vertices
  and joints with the actual checked geometry. Refuse discrepancies above
  `1e-7` metres; do not repair mismatches by changing the exported scale.
- Hash the input frames, depth, detections, native records, raw geometry, models,
  geometry helpers and complete effective method. Refuse changed inputs or sources.

The existing policy thresholds are unchanged. No episode-specific crop, frame
list, contact label or object ground-truth pose enters the recovery method.

## Preconditions

Use a Linux GPU environment with the project reconstruction dependencies,
including Torch, SAM-2, MANO/manotorch and an EGL-capable renderer. Supply the
licensed MANO assets and model checkpoints through the project's existing setup.
Do not commit those assets. The current development environment uses Python
3.11.17, Torch 2.9.1+cu128, NumPy 1.26.4, OpenCV 4.11.0.86, pyrender 0.1.45,
trimesh 4.12.2, SAM-2 1.0 and manotorch 0.0.2. Each run saves actual package
versions and GPU identity; this list is not a complete environment lock.

The required upstream outputs are:

- Rectified source `manifest.json`, `color/` and optional `valid/` images.
- Original metric `depth/`, temporal `hands/records/{2,3}/` and
  `align/records/{2,3}/` from the same episode.
- Automatic `wearer_candidates.json` and `fit_receipt.json` from that episode.

Tracks 2 and 3 are the native left/right convention. The adapter refuses
nonobserved estimates, missing independent detections, unsupported boxes,
inconsistent MANO parameters and changing per-track scales. A changing-scale
episode requires an explicitly supported adapter; it must not silently reuse
one development episode's scale.

## Run the retained-estimate checks

Set these paths to independently reproduced upstream outputs. Use fresh output
directories. All commands run from the repository root; no machine-specific
absolute path is required by the method.

```bash
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
QUALITY="$PWD/reconstruction/validation/issue6"
MODULES="$PWD/reconstruction/modules"

python "$QUALITY/prepare_retained_hand_quality_inputs.py" \
  --source "$RECTIFIED_INPUTS" --original-run "$ORIGINAL_RUN" \
  --temporal-source "$TEMPORAL_HANDS" --modules "$MODULES" \
  --quality-source "$QUALITY" --mano-assets "$MANO_ASSETS" \
  --sam-checkpoint "$SAM_CHECKPOINT" --out "$FRESH_PROPOSALS"

python "$QUALITY/run_retained_hand_quality.py" \
  --source "$RECTIFIED_INPUTS" --proposals "$FRESH_PROPOSALS" \
  --depth-root "$ORIGINAL_RUN/depth" --modules "$MODULES" \
  --quality-source "$QUALITY" --mano-assets "$MANO_ASSETS" \
  --sam-checkpoint "$SAM_CHECKPOINT" --out "$FRESH_QUALITY"
```

The runner fixes seed, Torch thread count, cuDNN benchmarking and TF32 behavior.
It verifies the source/model/method hashes before checking any proposals and
checks that the effective method stayed unchanged at completion. On a failure,
preserve the failure receipt and use a new output directory for a corrected run.

Check `run-receipt.json` for terminal status, `summary.json` for coverage,
`frame-evidence.json` for every refusal, `detector-mask-evidence.json` for mask
bounds and unchanged scores, and `geometry-export-verification.json` for every
native export. The existing `accepted/records` directory name means only that
the **consistency proxies** passed. `quality_accepted` and
`physical_task_accepted` remain false. Export verification is geometry
consistency, not hand-placement ground truth.

## Verification controls

```bash
python -m pip install numpy==1.26.4 pytest==8.4.2 opencv-python-headless==4.11.0.86
python -m pytest \
  reconstruction/modules/v2d_issue6/tests/test_hand_quality_support.py \
  reconstruction/modules/v2d_issue6/tests/test_automatic_recovery.py \
  reconstruction/modules/v2d_issue6/tests/test_automatic_reproduction.py -q
python -m unittest discover -s scripts/reproduction -p 'test_*.py'
```

The controls cover complete displaced meshes, high-confidence background masks,
wrong depth, fully occluded/no-stereo hands, malformed detector boxes, nonunit
native scale, display holds masquerading as observations, changed reproduction
support, changed geometry and false accuracy claims. The Windows/Linux workflow
also exercises these checks with NumPy 2.2.6 and Python 3.11/3.12.

## Development evidence and limits

Before GitHub integration, the completed e002 development recheck covered all
343 frames: 321 left / 332 right consistency-passing proposals, including
51 left / 48 right hand-frame candidates missing from the frozen baseline.
All 653 retained-scale exports matched measured geometry within
`1.849e-8` metres. Six synthetic refusal controls passed. Seventeen diagnostic
source frames were reviewed, including all four largest left-hand shifts.

The fair comparison uses identical fresh masks, depth, camera and policy for old
and new geometry. Among baseline-valid frames, the left pass count changes from
268/272 to 270/272 and median depth residual from 7.648 mm to 6.996 mm. The right
pass count remains 284/284; its median residual changes from 5.167 mm to 5.218 mm.
The geometry/baseline comparison agrees on Windows/Linux within `1.2e-16`.
This verifies those calculations, not complete source reconstruction or contact
accuracy. The integrated GitHub runner completed on October 12 using source
commit `fcb5ad8a811743a6ed62229bdca29f5a8c49b60b`: 321 left / 332 right passing
proposals and 653 geometry-verified exports. All 3,966 saved geometry arrays
match the previous measured run exactly; all 661 detector-mask PNGs match by
SHA-256. The maximum independently rematerialized native export error is
`1.8489534375776273e-8` metres. This reruns SAM and the quality checker from
retained estimates; it does not rerun upstream detector/MANO inference.

The full 175-test issue6 suite passes on the native inference environment;
32 recovery/geometry controls pass on Windows (NumPy 1.26.4) and native Linux
(NumPy 2.2.6). The 14 reproduction controls also pass on Windows. These are
local results; the cross-platform CI matrix must still report its own results.
The compact measured receipt is in `evidence/retained-hand-quality-e002.json`.

Use a Git checkout with the repository's `.gitattributes` applied. Several
method files use LF, while the existing frozen `wearer_selection.py` uses CRLF.
ZIP/archive downloads or editors that rewrite line endings can change method
byte hashes even without a logic change. A hash mismatch must be investigated;
do not edit a receipt to make it pass.

Do not overwrite existing observation flags from these counts. Original and
retained source shapes can differ; a 16.033 mm change from one existing left-hand
estimate is not proof of improved 3D accuracy. Image and depth checks can reject
bad fits but do not establish precise fingertips behind an object or torque.

The retained adapter intentionally refuses held-out episodes until the complete
effective method has a suitable, verified freeze. Untouched-episode validation,
independent full upstream inference/fitting, production adoption, contact/support
acceptance and an updated actual Isaac replay remain unfinished. The older
rejected automatic-hand holdout results are historical evidence for a different
method and must not be reported as results of this candidate.
