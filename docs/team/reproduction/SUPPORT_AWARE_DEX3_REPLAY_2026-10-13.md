# Support-aware Dex3 development replay

This is the e002 development candidate following the immutable
[original CAD replay](ORIGINAL_CAD_DEX3_REPLAY_2026-10-13.md). It adds automatic
free-motion repairs and checks an estimated finite support sheet. The latest
candidate passes all 343 source-pose audits and both complete continuous
checks: original CAD/estimated support and material/self/cross surfaces. Its
new Isaac run completed 343 frames; state/camera/video audits pass on the A40
and independently on Windows. All 343 actual rendered frames have been inspected
in chronological sheets, with additional full-resolution checks. Both hands
remain displayed; no artificial observation-mask on/off flicker was found.

Use the immutable
[support replay release](https://github.com/Swaguto/NVIDIA-V2D-Challenge/releases/tag/dex3-support-replay-20261013)
and its [asset registry](dex3-motion-replay/support-artifacts-v1.json).
Everyone with repository read access can download these artifacts. The earlier
baseline remains available. Physical motion-target approval is still withheld;
no training or issue #9 work has started.

Clone the repository and check out `dex3-support-replay-20261013` in detached
mode. Download the nine named release assets; use the registry's exact filenames,
sizes and hashes. `GIT_LFS_SKIP_SMUDGE=1` can be used while cloning because the
replay input archive includes its locked hand assets. A moving branch or `main`
is not the immutable reproduction reference.

## Method, in order

1. Preserve the source bundle, camera/object estimates, original Dex3 assets,
   observation masks and 377 fitted closed grasp anchors from the reviewed
   baseline. Source observations, display-only poses and inferred robot poses
   remain separate. Increasing display continuity adds no observations.
2. Estimate support from source depth using the frozen plane/cell method.
   The finite sheet is the convex hull of supported cells. It fills unmeasured
   holes and has no measured thickness. The 8–10 mm support-height uncertainty
   remains; this is not a measured table or its physical collision mesh.
3. Select free phases automatically from actual continuous-check failures or
   the fixed 80 mm wrist-distance search guard. The guard compares an inferred
   robot wrist with an inferred human wrist; it is a morphology search bound,
   not a measured tracking-accuracy tolerance. No manual episode frame, crop or
   contact annotation selects these repairs.
4. Fit directed paths with deterministic guide/edge/height candidates, bounded
   orientation and finger changes, original CAD/support checks, self/cross
   checks and actual URDF finger limits. Keep a checked observed free waypoint
   so a release cannot be replaced by an artificial continuing grasp. Freeze
   four initial paths and two later paths as inferred fitted inputs.
5. Test the full trajectory continuously. Sampled search checks missed three
   brief support crossings. Each refusal automatically selects its adjacent
   source-time bracket for bounded joint/root repair on a 1 ms screening grid.
   These three repairs add 151 knots, move roots at most 0.577, 0.289 and
   0.474 mm, and preserve every source-time robot pose.
6. Resolve the unobserved opening overlap by automatically selecting the last
   checked opening pair before observations/contact. Here that anchor is
   frame 45. Both hands are held only for display, with explicit provenance;
   no observed pose or closed anchor changes. This is a general selection rule,
   not a manually chosen frame in the fitting method.
7. Retain prior refinement masks and accumulated corrections when applying a
   later path. Reuse a completed proof prefix only after exact structural
   verification of its knots, source poses/provenance, all closed anchors,
   geometry, object/camera inputs and checker bindings. Missing coverage,
   altered observations, changed grasp anchors and altered input bindings are
   refused. A search success never substitutes for a continuous certificate.
8. Reconstruct from a clean directory using saved paths and correction
   coefficients. Check each generated motion file and case manifest against
   its frozen hash before carrying forward original producer metadata.
   Reconstruction receipts identify the actual new run separately from the
   original optimizer's reports.
9. Run the existing frozen Isaac renderer, audit actual state/camera readback,
   decode all 343 frames and inspect them in order against the source. The new
   replay and independent Windows audit pass. The owned A40 was confirmed
   `EXITED` at 23:42:21 UTC on October 13 after downloading and verifying evidence.

## Measured evidence and limits

| Check | Current result |
| --- | --- |
| Source observation coverage | 272 left / 284 right; original masks unchanged |
| Closed source anchors | All 377 retained exactly |
| Source FK, fingers and hand intersections | All 343 frames pass |
| Source wrist search guard | Observed free-state maxima 79.631 mm left / 71.975 mm right against inferred human wrists |
| Finger speed | Maximum URDF interval speed ratios 0.912626 left / 0.999698 right |
| Original CAD + inferred finite sheet | All 3,955 segments; 4,698 certified intervals; no failed/unresolved interval |
| Material/self/cross continuous check | All 3,955 segments; 7,985 certified intervals; unchanged 2 mm declared voxel-model depth limit; no failed/unresolved interval |
| Stored proof verification | Complete coverage/bindings pass independently on Linux and Windows |
| Source inspection | 25 earlier left, 11 first-right and all 24 late-path full images reviewed |
| Saved-fit reproduction | All 15 stages reproduced byte exactly through the consolidated runner; Windows input verification and legacy CAD command pass |
| New Isaac replay and flicker inspection | 343 frames / 686 explicit hand states; all frames visually inspected; no artificial on/off flicker observed |
| Independent Isaac position/joint errors | Maximum root position 6.711e-8 m; joint angle 5.768e-8 rad; root rotation 2.997e-7 rad |
| Independent camera errors | Maximum position 1.028e-9 m; rotation 4.032e-7 rad; intrinsics 0 px |
| Visibility indicators | Minimum left/right hand-colored pixels 9,239 / 3,112; no below-20 or isolated low-color frame; color is not segmentation truth |

The late source overlays match the actual candidate source poses to 1e-14.
They are CPU mesh outlines without occlusion, not Isaac renders. Left return
placement changes are modest; finger posture and morphology differ from the
estimated human skeleton. The right final release is at the camera edge, where
the fitted finger extent/posture differs and complete placement cannot be
verified. No estimated skeleton is drawn for invalid left 265/266 or right
332–337. A valid reconstruction mask is not proof that every finger is visible
or accurate in the recording.

Object calibration alternatives were rejected on held-out source-image guards;
the original object tracks remain unchanged. Contact timing remains an
evaluation-only source annotation, never a fitting input. Six previously
unsupported core contact boundaries remain documented in the original guide.
Hidden thumbs, source-depth residuals and support identity/height cannot be
resolved into independent ground truth from the available recording.

All robot poses are inferred and all motion-target validation flags remain
false. Unsigned CAD separation does not certify a true watertight solid. The
voxel check concerns the declared material model. These checks do not prove
real-table clearance, correct finger identity, contact forces, dynamic grasp
stability, arm/whole-body feasibility or evaluation scores. The renderer is a
kinematic replay with gravity and collision resolution disabled. No untouched
public or private evaluation episode has been validated with this fitted method.

The minimum certified CAD/support separation lower bound is only
0.000188 mm; the minimum self/cross-hand lower bound is 0.291 mm. A positive
mathematical bound is not a robust physical clearance margin against an
8–10 mm support-height error. The finite sheet is checked but is not rendered
as a table in this frozen Isaac renderer. Its `display_holds_used: false`
means the renderer does not create holds from missing observations; the input
motion still includes the explicitly marked, unobserved opening display pair.
Read pose provenance rather than treating this renderer field as an accuracy flag.

## One reproduction entry point

The active codebase uses the existing
`scripts/reproduction/reproduce_CAD_dex3_motion.py`. Its original CAD archive
arguments remain supported. New method-kit support shares the same hash and
runtime checks; no separate support-reproduction entry point is installed.
Immutable historical method snapshots remain evidence, not competing live
pipeline implementations. The Isaac renderer and runtime setup are reused.

The new kit contains a locked starting case, source/assets, material/sheet
inputs, saved paths/corrections, original reports, stage recipe, runtime lock and
a frozen copy of the existing entry point. It verifies 1,828 input files and
regenerates 15 stages into fresh directories. It does **not** repeat raw-video
inference or optimizer searches. Anyone with repository/artifact access and
the stated compute environment can repeat this fitted-input reconstruction
and Isaac replay; private MANO weights are not required for this boundary.
Original source images are needed only to regenerate the source comparison.

For this frozen kit, the manifest hash is
`bec54ca77b60cdf6013c075b3e3be6265eab4ac644a04e75cf114276493fd61e`.
Use Linux Python 3.11.17 and the included `fitted-motion-runtime-lock.txt`;
the packaged OMPL 2.0.1 library is separately hash-locked. Windows users can
verify inputs with ordinary Python and use WSL for numerical reconstruction.

The method archive is `CAD-support-fitted-inputs-and-methods-v3.tar.gz`,
109,868,963 bytes, SHA256
`b238cb32ea136e123b6ecec0a922c833ccc7fe097746b05abd4553815a2878d7`.
Its external `archive-registry.json` lists all 1,829 extracted files, including
the kit manifest. Restore into a new directory with the existing verified
extractor, then run the commands below:

```sh
python -c "import sys; from pathlib import Path; sys.path.insert(0, 'scripts/reproduction'); from reproduce_frozen_dex3_replay import restore_methods; restore_methods(Path('/data/v2d/CAD-support-fitted-inputs-and-methods-v3.tar.gz'), Path('/data/v2d/archive-registry.json'), Path('/data/v2d/support-method-kit'))"
```

The archive has been restored and every extracted file hash verified. Changed
input, wrong manifest hash and escaping input paths are refused before execution.
These controls verify bookkeeping; the independent continuous proof verification
separately accounts for every segment and all declared geometry pairs.

```sh
python scripts/reproduction/reproduce_CAD_dex3_motion.py \
  --method-kit /data/v2d/support-method-kit \
  --method-manifest-sha256 bec54ca77b60cdf6013c075b3e3be6265eab4ac644a04e75cf114276493fd61e \
  --out /data/v2d/support-input-check --check-only

/data/v2d/geometry-runtime/bin/python scripts/reproduction/reproduce_CAD_dex3_motion.py \
  --method-kit /data/v2d/support-method-kit \
  --method-manifest-sha256 bec54ca77b60cdf6013c075b3e3be6265eab4ac644a04e75cf114276493fd61e \
  --out /data/v2d/support-reconstruction
```

Require all 15 stage checks and `all_recorded_motion_bytes_exact: true` in the
new receipt. The final case is `15-case`, with manifest
`dfe06ac14d976e48ffc58ec07cad74f8d49573fc4a080bd6f57f79668ddf4ebb`.
Its source motion hash is
`0f30f3d7baea4359160c7b243c05f3bda4bb28c3494999b2e975c960b3702bd8`;
its knot hash is
`783dd6e5afa0cad47f40489728d0c767fe757771c3c958f65a9c42acfe6cc3d4`.

Input verification alone is not reconstruction. Exact saved-fit reconstruction
is not fresh geometry or an Isaac run. The release separates fitted methods,
simulator inputs, actual simulator evidence and source/visual evidence. Stored
receipts record their own scope; do not promote a successful hash check into
a claim of raw-video inference, new geometry calculation or physical accuracy.

## Verify the saved continuous proofs

After reconstructing the method kit and verifying the simulator input archive,
use the existing frozen verifier. The `case` below is the extracted simulator
case; `support-reconstruction` is the fresh 15-stage result. This command
verifies saved proof coverage and source/geometry bindings, not new geometry:

```sh
/data/v2d/geometry-runtime/bin/python /data/v2d/support-method-kit/helpers/verify_CAD_and_support_replay_geometry_v4.py \
  --case /data/v2d/support-inputs/case \
  --sheet /data/v2d/support-method-kit/sheet \
  --methods /data/v2d/support-method-kit/methods \
  --material /data/v2d/support-method-kit/material \
  --proof /data/v2d/support-inputs/case/validation/model-continuous/report.json \
  --cad-proof /data/v2d/support-inputs/case/validation/original-CAD-continuous/report.json \
  --source-fk /data/v2d/support-inputs/case/scene-audit/report.json \
  --closed-parent-case /data/v2d/support-method-kit/base-case \
  --original-case /data/v2d/support-method-kit/base-case \
  --pre-repair-proposal /data/v2d/support-reconstruction/12-finalized \
  --out /data/v2d/new-stored-proof-verification.json
```

Require `complete_continuous_replay_verified: true`, 3,955 segments, 4,698 CAD
intervals, 7,985 material/self/cross intervals, 343 source frames and 377 retained
closed anchors. The extracted simulator manifest is
`0954068550b897b5cae5465e24e8dc7edeae025223e5e4786068bffb22628fb2`;
it additionally includes proof/support evidence and differs from the trajectory
case manifest. Both contain the exact same motion files.

The simulator input archive is `CAD-support-dex3-simulator-inputs-v2.tar.gz`,
83,764,720 bytes, SHA256
`759b6e9fa642b47ec24306969688cc1ce15ead4cb0b80e4f8d086b727b93c680`.
Verify/extract it using the existing replay command:

```sh
python scripts/reproduction/reproduce_frozen_dex3_replay.py \
  --archive /data/v2d/CAD-support-dex3-simulator-inputs-v2.tar.gz \
  --sha256 759b6e9fa642b47ec24306969688cc1ce15ead4cb0b80e4f8d086b727b93c680 \
  --out /data/v2d/support-inputs --check-only
```

Windows input extraction and the full stored-proof command above have both
passed. Use fresh output paths, preserve refusals and report the first mismatch;
never replace hashes or relax limits to make a reproduction pass.

## Numerical and simulator environments

For fitted reconstruction, provide Linux x86-64 Python **3.11.17**, create a
fresh environment and install the kit's recorded closure:

```sh
python3.11 -m venv /data/v2d/geometry-runtime
/data/v2d/geometry-runtime/bin/python -m pip install --no-deps \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r /data/v2d/support-method-kit/fitted-motion-runtime-lock.txt
```

The command refuses a different Python patch version or any mismatch in the
eight core geometry packages. The separately packaged OMPL library is hash
checked. Fresh-directory repetition of all 15 stages is measured on the pinned
Linux runtime. Windows verifies inputs, stored proof coverage and actual Isaac
readback; Windows numerical reconstruction is not claimed.

For Isaac, provide a supported NVIDIA GPU/driver, Linux Python **3.12.15** and
the NVIDIA license/graphics prerequisites described in the
[original runtime guide](ORIGINAL_CAD_DEX3_REPLAY_2026-10-13.md). The measured run
used the pinned Track3 container, A40 and driver 580.159.04. Install on local disk
using the existing verified setup helper:

```sh
python3.12 /data/v2d/support-inputs/methods/setup_sim6.py \
  --lock /data/v2d/support-inputs/methods/sim6-runtime-lock.txt \
  --prefix /data/v2d/runtime6 --lab /data/v2d/IsaacLab3

python scripts/reproduction/reproduce_frozen_dex3_replay.py \
  --archive /data/v2d/CAD-support-dex3-simulator-inputs-v2.tar.gz \
  --sha256 759b6e9fa642b47ec24306969688cc1ce15ead4cb0b80e4f8d086b727b93c680 \
  --out /data/v2d/new-support-Isaac-run \
  --runtime-python /data/v2d/runtime6/bin/python --lab /data/v2d/IsaacLab3
```

This exact existing wrapper was used for the measured new run. It requires all
178 runtime pins and clean Isaac Lab commit
`28a37cecdd433c22d9eabd6a5954add9f13a8951`, runs the renderer and performs the
independent state/camera/video audit. Require `simulator/complete.json`, no
`failure.json`, 343 source and decoded frames, 686 displayed inferred hand
states, original observation masks and false motion-target approval flags.

## Inspect and compare results

The release contains the actual `replay.mp4` and
`source-observations-dex3.mp4` (source, valid recovered observations, actual Isaac).
`CAD-support-simulator-evidence-v1.tar.gz` includes all 343 rendered PNGs,
readback, runtime and numerical audits. `support-source-visual-evidence-v1.tar.gz`
includes all-frame sheets, source comparison pages, selected source diagnostics
and a separate visual-review receipt. Restore either using `restore_methods`
and its matching release registry into a new directory; the same hash-checked
extractor handles both archives.

All 343 actual Isaac frames were visually inspected in chronological sheets.
Source comparisons were inspected at selected contact/release and late-path
windows, including the six named source comparison pages in the visual receipt;
not every source comparison page was manually inspected. Camera-edge clipping
and fast orientation changes remain visible. A hand leaving the camera view
is different from an observation mask hiding its model. No artificial on/off
flicker was found in this rendered sequence; this is not an all-episode guarantee.

Regenerated motion NPZ bytes and case manifests must match exactly. Actual
simulator positions/rotations/joints must meet the unchanged 1e-4 m / 1e-4 rad
tolerances; camera intrinsics must meet 1e-3 px. Video pixels and compressed
bytes may differ across GPU drivers and encoders. Published video hashes verify
download integrity; they do not require another person's renderer to produce
identical compressed bytes.

To regenerate the source comparison, use the kit's existing
`methods/make_generated_physical_review_v4.py` with the verified case, new
simulator and its audit. Original e002 rectified `color/000000.png` through
`000342.png` must match `bundle/source/provenance/inputs-lock.json`. The measured
comparison used geometry Python 3.11.17, ImageIO 2.38.1, Pillow 11.3.0 and the
additional `imageio-ffmpeg==0.6.0` plugin in a separate dependency directory.
Missing observations deliberately have no recovered-hand wireframe. Neither
the wireframe nor its visibility is independent accuracy evidence.

```sh
/data/v2d/geometry-runtime/bin/python -m pip install --no-deps --no-compile \
  --target /data/v2d/video-review-deps imageio-ffmpeg==0.6.0

PYTHONPATH=/data/v2d/video-review-deps /data/v2d/geometry-runtime/bin/python \
  /data/v2d/support-method-kit/methods/make_generated_physical_review_v4.py \
  --bundle /data/v2d/support-inputs/case/bundle \
  --retarget /data/v2d/support-inputs/case/retarget \
  --proposal /data/v2d/support-inputs/case/proposal \
  --replay /data/v2d/new-support-Isaac-run/simulator \
  --color-dir /data/v2d/original-e002/color \
  --independent-audit /data/v2d/new-support-Isaac-run/independent-audit/report.json \
  --out /data/v2d/new-source-comparison
```

PR #35 targets `main`; PR #37 remains a separate recovery review. PR #38 contains
this baseline and documentation. Technical self-review and reproduction checks
are complete; human review and merge are outstanding. Physical hand accuracy,
real support clearance, force/dynamic success and untouched-episode validation
remain separate gates. Do not treat this replay as permission to start issue #9.
