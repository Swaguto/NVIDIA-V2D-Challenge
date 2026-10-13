# Original CAD Dex3 development replay: method and reproduction

The e002 development replay now has 343 actual Isaac Sim frames, complete
continuous checks against the original CAD triangle surfaces and the declared
object material model, and byte-exact regeneration from saved fitted inputs.
All 343 rendered frames were inspected in order: no artificial hand on/off
flicker was seen. Camera-edge cropping and object occlusion remain.

This is a kinematic diagnostic replay. Source pose accuracy and physical motion
targets remain unaccepted. Six source contact boundaries remain unsupported;
table support, forces, grasp stability and true solid penetration are not
certified. Issue #9 and training remain deferred. This document supersedes the
current-result status of the earlier corrected replay, whose release is preserved.

## What was done, step by step

1. **Freeze the source and assets.** Preserve the 343-frame e002 bundle, object
   tracks, cameras and original Dex3 meshes. Keep observed hands, directly
   observed hands, display holds and inferred robot poses distinct. Source
   valid-hand counts remain 272 left and 284 right; visibility adds no observations.
2. **Check the original contact geometry.** The earlier selected witness faces
   did not meet the 2 mm facing-face contact proxy. Fit four bounded object-local
   thumb/index grasp anchors against original CAD geometry. The selected static
   fit is `original-CAD-contact-candidates-v10-pair03`; this is a development fit,
   not an untouched-episode result. All 377 inferred closed source hand states
   now pass the selected original-face gap and normal checks.
3. **Fit automatic approaches and releases.** Infer eight transition windows
   from existing motion intent. Use bounded envelopes and joint trajectory
   repair against original geometry and finger limits. The winning frozen
   producer is `plan_CAD_contact_transition_envelopes_v12.py` with
   `repair_CAD_joint_trajectory_v2.py`, measured search output V13. Do not confuse
   other prototypes included in the method archive with this selected result.
4. **Check between samples and repair refusals.** Three original-CAD continuous
   counterexamples select their adjacent source brackets automatically. Add
   1 ms repair knots there without changing any of the 343 source robot poses
   or 377 new closed anchors. The repairs use no manually selected episode
   frames, crops or source contact labels. The result is motion V10, with 3,737
   knots and 3,736 interpolation segments. The initial CAD fitting in steps 2–3
   does change inferred robot poses relative to the earlier replay.
5. **Verify the complete trajectory.** Check every segment against unsigned
   original triangle surfaces, the unchanged 2 mm modeled material-depth limit,
   self separation and cross-hand separation. Reuse only an exactly verified
   unchanged prefix. Independently verify full coverage, hashes, source masks,
   closed anchors, source-time FK and finger position/speed constraints.
6. **Run actual Isaac Sim.** Install and verify the frozen 178-package runtime
   on an A40 worker. Render actual Dex3 articulations at all 343 source times,
   save state readback and camera metadata, independently audit them on the
   worker and Windows, and inspect all frames in chronological montages.
   Intermediate knots are certified offline; the video renders source times.
   Download and verify evidence, then stop the owned worker.
7. **Reproduce from a clean directory.** Restore saved static/transition fits
   and three fitted counterexample repairs. Regenerate all 27 motion arrays and
   four knot arrays. A fresh Linux run reproduced both compressed NPZ files
   byte for byte and repeated the finite repair-window CAD/hardware checks.

The reproducible boundary is **saved source bundle and fitted inputs to final
motion and Isaac replay**. The command below does not rerun raw-video
reconstruction, optimizer searches, manual source annotation or a new continuous
geometry calculation. Those stages have separate measured records and limits.
This distinction applies to everyone with repository access, not only teammates.

## Measured results and limits

| Check | Result |
|---|---|
| Original CAD surfaces | All 3,736 segments; 4,229 certified intervals and 135,328 link/object records; no failed/unresolved interval |
| Material/self/cross checks | 7,387 certified intervals and 236,384 link/object records; unchanged 2 mm modeled depth limit |
| Closed selected-face proxy | 377/377 inferred closed states; facing selected thumb/index triangles, gap 0.3–2 mm; not grasp force |
| Source timing | 375/381 labeled core hand states supported; left 46, 272, 332, 333 and right 46, 309 remain unsupported |
| Source-time FK and fingers | All 343 source times pass; maximum finger speed ratios 0.900544 left / 0.999698 right |
| Actual Isaac replay | 343 rendered/decoded frames; 686 inferred robot hand states; Isaac Sim 6.0.0.1 |
| Independent readback | Root position error 8.78e-8 m; joints 5.81e-8 rad; camera position 1.03e-9 m; intrinsic error 0 px |
| Visual inspection | All 343 simulator frames inspected, plus source/observation/Isaac comparisons at five contact events; no artificial mesh disappearance seen |
| Fresh fitted reproduction | 27 motion + 4 knot arrays exact; both NPZ hashes exact |

The original-CAD proof concerns **unsigned surface separation**. It is not a
signed solid-volume or true-object penetration guarantee. The material proof
concerns the declared voxel model. Table/environment clearance, root speed and
acceleration, dynamic contact and whole-body feasibility are not certified.
Gravity and collision resolution are disabled in this kinematic renderer.

Accuracy is mixed. New pad targets move 2.28–3.08 mm relative to the old targets.
Proximity to the estimated human mesh improves some fingers and worsens others:
the second left-thumb median changes from 2.412 to 5.125 mm. The estimated MANO
mesh is not independent ground truth or a finger-identity test. Hidden fingertips
and limited left distal-thumb support remain uncertain. Object source-depth
median absolute residuals are 12.923 mm for the pot and 6.538 mm for the lid;
an 8–10 mm support disagreement remains. The pot has 285 observed, 47 head
backfill and 11 inferred tail frames; the lid has 343 observed frames.

All robot poses remain inferred and motion-target validation flags remain false.
Source annotations are read-only evaluation after the motion is frozen; they
do not choose transitions or repairs. No untouched public episodes or private
evaluation episodes were tested with this CAD method. There is no score claim.

Machine-readable measurements: [CAD-replay-evidence-v1.json](dex3-motion-replay/CAD-replay-evidence-v1.json).

## Obtain the frozen version

Use the private [CAD prerelease](https://github.com/Swaguto/NVIDIA-V2D-Challenge/releases/tag/dex3-cad-replay-20261013)
and [asset registry](dex3-motion-replay/CAD-artifacts-v1.json). Anyone with
repository access can download the derived artifacts. Original datasets and
licensed upstream models retain their existing access requirements.

```sh
git clone https://github.com/Swaguto/NVIDIA-V2D-Challenge.git
cd NVIDIA-V2D-Challenge
git checkout --detach dex3-cad-replay-20261013
```

Download `CAD-dex3-simulator-inputs-v1.tar.gz` and
`CAD-fitted-inputs-and-methods-v1.tar.gz` from this release. Download
`fitted-motion-inputs-v1.tar.gz` from the preserved
[October 12 release](https://github.com/Swaguto/NVIDIA-V2D-Challenge/releases/tag/dex3-development-replay-20261012).
The registry locks download sizes and SHA256 hashes. Keep archives compressed.
Use a new output directory for each command. Examples use `/data/v2d`; replace
it with an absolute directory on your machine. Windows check commands accept
the same arguments on one line with absolute Windows paths.

## Verify the package and stored certificates

The standard-library restoration command works on Windows and Linux with
Python 3.11.8 or newer and requires no GPU:

```sh
python scripts/reproduction/reproduce_frozen_dex3_replay.py \
  --archive /data/v2d/CAD-dex3-simulator-inputs-v1.tar.gz \
  --sha256 26466ebd7a7ae5eacb32f8410e2b1e418438ee443bcaaf8c4b56b1cbbbbd20e8 \
  --out /data/v2d/verified-inputs --check-only
```

Require **1,660 verified listed files**, plus the manifest itself. The manifest
SHA256 is `6ec5735d60d50ccd38fb77e97111b9fba7d184ab915c47fb2d2eb75f01578e30`.
This command passed on both Windows and Linux for this package.

With NumPy installed, verify the complete stored proof:

```sh
python scripts/reproduction/verify_CAD_dex3_certificate.py \
  --case /data/v2d/verified-inputs/case \
  --out /data/v2d/stored-proof-verification.json
```

Require 3,736 segments, 4,229 CAD intervals, 7,387 model intervals and
`all_segment_boundaries_and_saved_margins_verified: true`. Windows and Linux
receipts are identical. This verifies stored proof coverage and input/method
bindings; **`geometry_recomputed` must be false**.

## Regenerate the final motion from saved fits

Use **Linux Python 3.11.17**, including the exact scientific versions in
[fitted-motion-runtime-lock.txt](dex3-motion-replay/fitted-motion-runtime-lock.txt).
Windows users can use WSL. Input restoration alone is not proof of an equivalent
scientific runtime. The runner verifies eight core package versions and sets
BLAS/OMP numerical threads to one.

```sh
python3.11 -m venv /data/v2d/geometry-runtime
/data/v2d/geometry-runtime/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r docs/team/reproduction/dex3-motion-replay/fitted-motion-runtime-lock.txt

/data/v2d/geometry-runtime/bin/python scripts/reproduction/reproduce_CAD_dex3_motion.py \
  --case /data/v2d/verified-inputs/case \
  --fitted-archive /data/v2d/fitted-motion-inputs-v1.tar.gz \
  --CAD-archive /data/v2d/CAD-fitted-inputs-and-methods-v1.tar.gz \
  --out /data/v2d/regenerated-motion
```

Require `all_payload_and_knot_arrays_identical: true`, counts 27 and 4, payload
SHA256 `1ac9fa5dbfcd778d2d62728c8cf3bf7e21110909f04b186f49034766b6f4026b`
and knot SHA256
`7fa56d17fa8caa53e1db2bede3e420e1fa9d6093fcafa3b67db715560e513f6a`.
The fresh run passed. New materializer reports identify the actual reproducer
and are retained separately; immutable measured fit reports are input metadata,
not falsely presented as newly regenerated reports. Output paths are in the
receipt and under `generated/v10`.

For input-only restoration on Windows, add `--check-only` to the command and use
ordinary Python. That check does not regenerate scientific arrays.

## Run actual Isaac Sim

Use Linux, an RTX-capable NVIDIA GPU, graphics libraries and accepted NVIDIA
licenses. The measured environment is Python 3.12.15, Isaac Sim 6.0.0.1 and
Isaac Lab commit `28a37cecdd433c22d9eabd6a5954add9f13a8951`
(tracked VERSION file 3.0.0). Pin the commit: the installed `isaaclab` package
metadata reports 6.1.11 and is not the Lab source version. The recorded A40
driver was 580.159.04. This runtime is separate from the geometry environment.

```sh
python3.12 /data/v2d/verified-inputs/methods/setup_sim6.py \
  --lock /data/v2d/verified-inputs/methods/sim6-runtime-lock.txt \
  --prefix /data/v2d/runtime6 --lab /data/v2d/IsaacLab3

python scripts/reproduction/reproduce_frozen_dex3_replay.py \
  --archive /data/v2d/CAD-dex3-simulator-inputs-v1.tar.gz \
  --sha256 26466ebd7a7ae5eacb32f8410e2b1e418438ee443bcaaf8c4b56b1cbbbbd20e8 \
  --out /data/v2d/new-isaac-replay \
  --runtime-python /data/v2d/runtime6/bin/python --lab /data/v2d/IsaacLab3
```

The measured new run used this same frozen setup, renderer and independent
auditor directly on the verified package; the convenience wrapper itself was
tested in input-only mode on both operating systems. Require passing 178 runtime
pins, `simulator/complete.json`, `source_frames: 343`, `decoded_video_frames: 343`,
no `failure.json`, a passing independent audit and exact source-observation masks.
Inspect all saved frames for flicker, contact timing, cropping and placement.
Stop your owned GPU when evidence has been downloaded and verified.

The release contains `replay.mp4`, `source-observations-dex3.mp4` and the complete
simulator evidence archive with all rendered frames, state readback and audits.
The three-panel video shows source, source observations and actual Isaac replay.

## Optional fresh continuous geometry calculation

The original-CAD checker can rerun from the restored methods and verified case:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/data/v2d/geometry-runtime/bin/python \
  /data/v2d/regenerated-motion/kit/methods/certify_original_CAD_replay_v1.py \
  --case /data/v2d/verified-inputs/case \
  --methods /data/v2d/regenerated-motion/kit/methods \
  --out /data/v2d/fresh-original-CAD-proof
```

This optional calculation was not rerun by the saved-fit reproducer. It needs
the pinned geometry runtime and takes additional CPU time. Require
`complete_original_CAD_triangle_surface_interpolation_certified: true` and no
failed/unresolved intervals. For a fresh material/self/cross check, download
`corrected-motion-inputs-v1.tar.gz` from the preserved
[corrected release](https://github.com/Swaguto/NVIDIA-V2D-Challenge/releases/tag/dex3-corrected-replay-20261013).
Restore its frozen material snapshot with the existing hash-checked extractor,
then run the new checker without a resumed prefix:

```sh
python -c "import sys; from pathlib import Path; sys.path.insert(0, 'scripts/reproduction'); from reproduce_frozen_dex3_replay import restore_methods; restore_methods(Path('/data/v2d/corrected-motion-inputs-v1.tar.gz'), Path('docs/team/reproduction/dex3-motion-replay/corrected-inputs-manifest-v1.json'), Path('/data/v2d/material-supplement'))"

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/data/v2d/geometry-runtime/bin/python \
  /data/v2d/regenerated-motion/kit/methods/certify_CAD_contact_replay_scene_v3.py \
  --source /data/v2d/verified-inputs/case/source/robotic_grounding \
  --bundle /data/v2d/verified-inputs/case/bundle \
  --baseline /data/v2d/verified-inputs/case/retarget \
  --parent-proposal /data/v2d/regenerated-motion/cases/original/proposal \
  --interval-inputs /data/v2d/regenerated-motion/inputs/contact-interval-dense-full-chain-inputs-v1 \
  --atlas /data/v2d/regenerated-motion/inputs/per-grasp-motion-atlas-reserve-v2.json \
  --proposal /data/v2d/verified-inputs/case/proposal \
  --material /data/v2d/material-supplement/material \
  --out /data/v2d/fresh-model-proof
```

Require `complete_replay_object_self_and_cross_hand_surfaces_certified: true`
and no failed/unresolved intervals. This optional fresh calculation was not
performed by the fitted-input reproduction command. Do not substitute the
stored-proof verifier when claiming a fresh geometry calculation.

## Compare another person's result

Source, asset, method, fitted input and trajectory bytes must match their locks.
All regenerated arrays and both NPZ hashes must match exactly in the declared
geometry runtime. Fresh simulator output must meet the unchanged state/camera
tolerances: 1e-4 m positions, 1e-4 rad rotations/joints and 1e-3 px intrinsics.
Video pixels and compressed bytes can vary across GPU drivers and encoders;
published video hashes check download integrity, not universal pixel equality.

Preserve a failed directory and report the first mismatch, runtime versions and
command. Do not relax limits, edit expected hashes or turn missing observations
into tracked hands. PR #35 targets `main`; PR #37 remains a separate recovery
review; PR #38 publishes this development work. Human review/merge remains
outstanding. See [the baseline review](DEX3_BASELINE_REVIEW_2026-10-13.md).
