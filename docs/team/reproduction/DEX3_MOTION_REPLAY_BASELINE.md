# Dex3 development replay: frozen method and verification

This records the e002 development replay over PR #35, separately from the
retained hand-quality recovery review in PR #37. PR #35 targets `main`; neither
draft has human approval. This is a motion replay, not trained grasping.

Recorded October 12, 2026. The final replay completed on a replacement A40,
and its independent audit passed in both the simulator environment and Windows.
All 343 rendered frames were reviewed in order: no artificial on/off model
flicker was observed. Camera-edge cropping and object occlusion still occur.
The worker was stopped after the downloaded evidence passed its hash check.
Measured results are in [e002-development-replay.json](dex3-motion-replay/e002-development-replay.json).

## What the method does

1. Retain the original video timestamps, camera, object tracks and observation
   masks. A missing human observation remains missing.
2. Retarget and fit four automatically inferred, object-relative Dex3 grasp
   corridors. Preserve all 377 closed-grasp hand-frame anchors.
3. Use source-following release alignment, screened against the declared voxel
   material. Apply bounded smooth corrections (6.419 mm left, 4.910 mm right,
   each below the unchanged 10 mm correction budget).
4. Generate current-image-supported robot entry/exit priors from the verified
   retained hand geometry. Use an automatic 0.5 s handoff. These are inferred
   robot poses, not accepted human observations or physical targets.
5. Reject a complete recovered robot edge when the finite geometry screen finds
   object or cross-hand failures. In e002, both recovered entries fail and use
   the existing checked approaches; both improved exits remain. There are no
   e002 frame numbers, crop choices or contact labels in this selection rule.
6. Check the selected trajectory against full original robot triangle surfaces
   at every source frame, probe its procedural timeline, and independently
   check contact timing against source labels after geometry calculations.
7. Freeze the replay inputs, verify transferred bytes, replay actual Dex3
   articulations in Isaac Sim, and check state/camera/video readback.

Human hand recovery, source observation flags, inferred robot poses and display
continuity are separate outputs. A consistently visible robot is not evidence
that every reconstructed human hand is correct.

## Measured development checks

The selected method is `clearance_gated_edge_motion_v2.py` in the byte-preserving
snapshot [dex3-motion-replay/methods.tar.gz](dex3-motion-replay/methods.tar.gz).
The snapshot includes 42 sources/runtime files with a
[manifest](dex3-motion-replay/methods-manifest.json). Unpack it to review sources;
the archive preserves the exact source bytes used by existing hash locks.

- All 343 frames and original source flags are unchanged.
- All 377 closed-grasp anchors and all finger arrays are unchanged.
- All 1,372 full source-time hand/object surface checks pass the declared 2 mm
  material-depth limit. The material model has 1 mm voxels and a 3.464 mm
  uncertainty band; this is not a true-object penetration guarantee.
- The 3,566-time geometry screen has zero object or cross-hand failures. This
  is a finite screen, not a continuous-path certificate.
- Source-frame joint positions and finger speeds pass their declared limits;
  there are no source-frame self/cross-hand mesh intersections. Root velocity
  bounds were not declared (maximum root speed 1.494 m/s, angular speed
  13.592 rad/s); whole-body feasibility has not been checked.
- Contact proxy coverage is 375/381 labeled core hand-frames. Gaps are left 46,
  272, 332, 333 and right 46, 309; all are grasp boundaries. The proxy measures
  two nearby, facing original pad faces, not force or grasp stability. The left
  thumb proximal pad remains a qualification; do not call it distal-tip contact.
- The retained quality runner reproduces all 3,966 geometry arrays and 661 mask
  images from its earlier measured run exactly. It exports 653 verified native
  records (321 left, 332 right), still unaccepted as physical targets. See PR #37
  for the adapter, independent geometry check and measured recovery protocol.

Object tracking and placement were checked against source image/depth evidence,
not ground truth: median object depth residuals are 12.923 mm for the pot and
6.538 mm for the lid. Pot poses include 285 observations, 47 head backfills and
11 inferred tail frames; lid poses are observed at all 343 frames. The source
support-plane discrepancy is about 8–10 mm. Hidden fingertips, support contact
and recovered wrist accuracy remain uncertain. These checks qualify this
development replay; they do not approve the trajectory as evaluation targets.

## Reproduce the frozen replay

Prerequisites: authorized access to the same source data/frozen case; Linux with
an RTX-capable NVIDIA GPU for rendering; Python 3.12.15; Isaac Sim 6.0.0.1;
Isaac Lab commit `28a37cecdd433c22d9eabd6a5954add9f13a8951`. The measured runtime
lock and its installer are inside the method snapshot. The worker uses container
`ghcr.io/swaguto/track3-runpod@sha256:46cc00a62d1076595b71401a96d17fbe53488afe1fe4d7a9784bf3512ac3e908`.
Install large runtime files on local disk; keep evidence on persistent storage.
NVIDIA license acceptance and system graphics libraries are prerequisites.

The private replay archive is `generated-dex3-simulator-inputs-v5.tar.gz`, SHA256
`0acb47d1aa8fdcdabc3023f8564297ed7a996f4a4a9ea235c8507bda553bd50d`.
It contains 1,650 manifest-listed files plus the manifest, including the exact
robot assets, source bundle, saved retarget, selected motion and FK audit. Raw
development data is not embedded in this repository's method snapshot.

First verify the case and method bytes (Python 3.11.8+; Windows or Linux):

Clone the repository and check out tag `dex3-development-replay-20261012` before
running these commands. Download the case and fitted-input archives listed in
[artifacts.json](dex3-motion-replay/artifacts.json). Keep the archives compressed;
the runners verify and extract them. The frozen derived inputs are assets of a
private repository prerelease, available to people with repository access. The
separate project Drive folder retains its existing permissions. Fresh upstream
inference still requires authorized access to the original data and models.

```sh
git clone https://github.com/Swaguto/NVIDIA-V2D-Challenge.git
cd NVIDIA-V2D-Challenge
git checkout --detach dex3-development-replay-20261012
```

[Open the private prerelease](https://github.com/Swaguto/NVIDIA-V2D-Challenge/releases/tag/dex3-development-replay-20261012)
to download the two archives and the actual Isaac/comparison videos.

```sh
python scripts/reproduction/reproduce_frozen_dex3_replay.py \
  --archive /absolute/path/generated-dex3-simulator-inputs-v5.tar.gz \
  --sha256 0acb47d1aa8fdcdabc3023f8564297ed7a996f4a4a9ea235c8507bda553bd50d \
  --out /absolute/path/fresh-verification --check-only
```

For an actual replay using the installed pinned runtime:

If the runtime is absent, install the recorded replay closure with Linux
Python 3.12.15. Use new environment and Lab directories on local disk:

```sh
python3.12 /absolute/path/fresh-verification/methods/setup_sim6.py \
  --lock /absolute/path/fresh-verification/methods/sim6-runtime-lock.txt \
  --prefix /absolute/path/runtime6 \
  --lab /absolute/path/IsaacLab3
```

This installer and the 178-package runtime check were used for the completed
replay. It installs the replay dependency closure; optional training features
are outside this environment. Then run:

```sh
python scripts/reproduction/reproduce_frozen_dex3_replay.py \
  --archive /absolute/path/generated-dex3-simulator-inputs-v5.tar.gz \
  --sha256 0acb47d1aa8fdcdabc3023f8564297ed7a996f4a4a9ea235c8507bda553bd50d \
  --out /absolute/path/fresh-replay \
  --runtime-python /absolute/path/runtime6/bin/python \
  --lab /absolute/path/IsaacLab3
```

The runner refuses reused outputs and altered hashes, checks installed versions
and Lab source, runs the source camera replay and independently audits readback.
Require `simulator/complete.json`, no `failure.json`, 343 decoded frames and all
343 saved images. Review every frame and contact boundary for visible flicker.
The numerical audit alone does not establish visual continuity. Stop the GPU
worker after evidence is downloaded and verified.

## What must match on another machine

The frozen source inputs, robot asset bytes, observation masks, timestamps and
inferred trajectory must match exactly. The method source snapshot is locked by
SHA256. Root/joint readback, camera poses and intrinsics use the tolerances in the
independent auditor (root/camera position 1e-4 m, rotations/joints 1e-4 rad,
intrinsics 1e-3 px). Raster pixels and compressed video bytes can differ with GPU
drivers; compare the declared state/camera metrics and inspect the video.

## Regenerate the procedural motion from fitted inputs

The private fitted-input kit is `fitted-motion-inputs-v1.tar.gz`, SHA256
`3daaa90efd1487f42019a14b5d783c9580e2b3b5103bf55b3c5b63b7ace21fe0`,
with its `manifest.json`. It includes 2,882 files in the fitted-input ancestry,
including quality records, grasp corridors, calibration and geometry screens.
Use the same authorized dataset access as for the replay archive.

After verifying the frozen case, use the measured Linux CPU environment
(Python 3.11.17, NumPy 2.2.6, SciPy 1.15.3, Pinocchio 3.7.0, Pink 4.2.0 and
their native libraries). The captured package pins are in
[fitted-motion-runtime-lock.txt](dex3-motion-replay/fitted-motion-runtime-lock.txt).
For a new CPU environment:

```sh
python3.11 -m venv /absolute/path/fitted-motion-runtime
/absolute/path/fitted-motion-runtime/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r docs/team/reproduction/dex3-motion-replay/fitted-motion-runtime-lock.txt
```

The lock captures the measured environment; installation of that complete CPU
lock into a second environment has not been tested. The clean procedural run
below used the measured environment with newly restored inputs and method
sources. Use the checked-in fitted-input
[manifest](dex3-motion-replay/fitted-input-manifest.json) and run:

```sh
python scripts/reproduction/regenerate_fitted_dex3_motion.py \
  --fitted-archive /absolute/path/fitted-motion-inputs-v1.tar.gz \
  --fitted-manifest docs/team/reproduction/dex3-motion-replay/fitted-input-manifest.json \
  --case /absolute/path/fresh-verification/case \
  --out /absolute/path/fresh-motion-regeneration
```

The public command was run in a fresh verified tree: all 22 payload arrays and
the generated NPZ file match exactly, SHA256
`3541cc177e14ebae927c8026d7693dd89e96b0438503db0bc68010a975688ef4`.
See [the reproduction receipt](dex3-motion-replay/procedural-reproduction-receipt.json).
This repeats the selected motion method from fitted inputs, not the upstream
hand/object reconstruction or the grasp-fitting search.

Verification of frozen replay is distinct from regeneration of its procedural
motion, refitting its grasp corridors, and reconstruction from the source video.
The method snapshot exposes the exact procedural entry points and dependencies;
refitting additionally requires the fitted-input ancestry, material models,
quality records and their receipts. It does not substitute for those inputs.
The complete upstream reconstruction/fitting pipeline has not been rerun
independently for this final snapshot. Untouched public episodes and the 20
private evaluation episodes have not been validated with this physical method.

The Isaac playback uses CPU PhysX transforms with RTX rendering, gravity and
collision resolution disabled. It verifies a prescribed kinematic motion and
model display, not a force-driven grasp. Continuous clearance, support contact,
whole-body dynamics and trained policy success remain separate later checks.
Issue #9 and training were not started by this work.

## Review the video and the remaining physical gate

The shared artifacts include the actual Isaac video and a synchronized
source / original-observation overlay / actual-Isaac comparison. The comparison
verified all 343 source image hashes. Its observation wireframe draws through
the source image, so it is an alignment overlay rather than an occlusion-aware
surface render. Missing original observations remain missing, including the
final source frames where the robot remains visible using inferred poses.

The final source comparison was inspected at frames 42–47, 270–275, 309,
330–335 and 342, covering every failed contact boundary. This is separate from
the all-frame Isaac flicker review. These views support the recorded event order
but do not resolve hidden pad contact, the 8–10 mm support discrepancy, or
millimetre accuracy of recovered wrists.

The reproducible replay and its display checks are complete. Physical-target
approval remains withheld: resolve the six contact-boundary gaps and proximal
thumb qualification, establish support/placement accuracy within declared
tolerances, and check the path between source frames before treating this as a
physically approved demonstration. Human PR review is also still pending.
Do not advance the Issue #9 gate on the basis of flicker removal alone.
