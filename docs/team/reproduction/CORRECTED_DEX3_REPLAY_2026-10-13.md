# Corrected Dex3 development replay and reproduction

The e002 replay is complete: 343 actual Isaac Sim frames, no artificial mesh
on/off flicker observed in the chronological frame review, and a complete
continuous check against the declared object material and original hand meshes.
The corrected fitted motion regenerates exactly in a clean Linux directory.
This is a kinematic motion replay. Its source accuracy, support and physical
motion-target approval remain unaccepted.

This supersedes the **current-result** status of the October 12 replay and
between-sample findings; those files and the old release remain historical
evidence. The development episode is e002. These measurements do not establish
coverage or accuracy on every public/private episode.

## Review and provenance

- [PR #35](https://github.com/Swaguto/NVIDIA-V2D-Challenge/pull/35) targets `main`
  at `ae8fa33f8f48a005cb22726ee96686fa30dda20b`. Its source masks, retarget and
  replay contracts are separate from the generated development motion here.
  CPU controls, reproduction and pre-commit checks pass; Build & Test is queued.
- [PR #37](https://github.com/Swaguto/NVIDIA-V2D-Challenge/pull/37) independently
  reviews retained hand recovery over `recent-reproducibility-code`, at
  `bc1a60132f30f7307ade59623d8e035ef0e20e4f`. Windows/Linux controls on Python
  3.11/3.12 pass. Its corrected exporter preserves the checked hand scale and
  independently rematerializes native exports. Its retained-input adapter uses
  the measured development track mapping; it is not a general upstream runner.
- [PR #38](https://github.com/Swaguto/NVIDIA-V2D-Challenge/pull/38) publishes this
  replay method and evidence over PR #35. These are draft reviews; no human
  approval or merge is implied by the technical checks.

Original observed/supported/missing hand flags stay exact. Display holds never
become observations. All generated robot poses remain inferred, and all robot
motion-target validation bits remain false. The new
`robot_noncontact_surface_repair_inferred` flag identifies the bounded root
corrections; it does not identify newly observed human hands.

## What changed, step by step

1. Preserve the source bundle, cameras, object tracks and original Dex3 assets.
   Reuse the fitted grasp corridors and prior source-following releases from
   [the original method](DEX3_MOTION_REPLAY_BASELINE.md). Preserve all 377 closed
   grasp anchors, finger arrays and source observations.
2. Declare explicit interpolation knots. Free motion uses linear translation
   and fingers with shortest quaternion interpolation; closed holds follow the
   unchanged rigid object and local grasp anchors.
3. Locate geometry refusals between saved frames. Choose noncontact repair
   windows from those refusals and clamp them outside inferred grasps. The
   solver reads no contact labels or ground truth. Keep the 2 mm modeled-depth
   limit and 10 mm added root-correction budget unchanged. Rejected fits remain
   rejected and retained as diagnostic evidence.
4. Apply the two successful smooth translation profiles. Fourteen inferred
   source hand states change. The maximum additional source-time translation
   relative to the original locked motion is **2.292396 mm**. First/second
   continuous profile amplitudes are 2.363669/1.626697 mm; these are different
   from the maximum sampled source-time correction.
5. Check all original hand triangle surfaces against both object materials,
   self separation and cross-hand separation through the full interpolation.
   Reuse a prior proof only for an exactly unchanged, structurally verified
   prefix. Rerun both modified windows and the entire remaining withdrawal.
6. Verify all coverage intervals and input hashes independently on Windows.
   Recheck source-time FK, asset geometry, finger positions and speeds.
7. Transfer the verified package, install the pinned simulator runtime, replay
   actual Dex3 articulations, and independently audit state, camera and video.
   Download and hash-check the evidence, then stop the GPU. Inspect all 343
   saved frames in order. Camera-edge cropping and object occlusion remain;
   no artificial model on/off flicker was observed.

## Measurements and remaining physical limits

| Check | Result and scope |
|---|---|
| Continuous modeled geometry | All 3,588 segments, 7,306 certified subintervals and 233,792 complete hand-surface/material records pass; no unresolved intervals |
| Material definition | Unchanged 2 mm depth limit; 1 mm voxels, 3.464 mm uncertainty band. This is not a true-object penetration guarantee |
| Original inputs | Observation arrays, object poses, orientations/fingers and all 377 grasp anchors unchanged |
| Isaac replay | Isaac Sim 6.0.0.1; 343 rendered and decoded frames, 686 inferred hand states |
| Independent readback | Maximum root position error 8.78e-8 m, joint error 5.85e-8 rad, link error 1.12e-7 m; camera position 1.03e-9 m and intrinsics 0 px |
| Visibility | All frames inspected; minimum review-color pixels 1,026 left / 2,604 right; no isolated empty-hand frames |
| Exact corrected regeneration | All 23 payload arrays, compressed payload bytes and interpolation-knot bytes match on clean Linux restoration |
| Source contact timing | Unchanged proxy coverage 375/381 labeled core hand-frames; left 46, 272, 332, 333 and right 46, 309 remain unsupported boundaries |
| Source hand placement | Visible-mask/stereo consistency is measured, not pose ground truth; limited left distal-thumb support and hidden fingertips remain uncertain |
| Object/support placement | Pot/lid median source-depth residuals 12.923/6.538 mm; pot includes inferred head/tail poses. An 8–10 mm support-plane disagreement remains qualified |
| Dynamics and environment | Kinematic replay; gravity/contact resolution disabled. Table/environment clearance, grasp forces, stability and whole-body feasibility are not certified |

The contact proxy checks nearby facing original distal thumb/index faces, not
force or observed human fingertip contact. Estimated image-mask agreement is
not semantic ground truth. Removing 1,451 wrong-owner contact candidates did
not justify extending any of the six unsupported boundaries. Do not invent
contacts, shift the whole object to hide support uncertainty, or use visibility
as an accuracy acceptance test. These limitations withhold full physical-target
approval and the move to Issue #9/training.

The compact machine-readable results and measured receipts are in
[corrected-replay-evidence-v1.json](dex3-motion-replay/corrected-replay-evidence-v1.json).

## Obtain the exact version and inputs

Use the private [corrected prerelease](https://github.com/Swaguto/NVIDIA-V2D-Challenge/releases/tag/dex3-corrected-replay-20261013)
and [artifact hashes](dex3-motion-replay/corrected-artifacts-v1.json). Anyone with
repository access can obtain these derived replay inputs. Original data and
licensed models retain their existing access requirements for upstream inference.
Do not change repository or data permissions as part of reproduction.

```sh
git clone https://github.com/Swaguto/NVIDIA-V2D-Challenge.git
cd NVIDIA-V2D-Challenge
git checkout --detach dex3-corrected-replay-20261013
```

Download `generated-dex3-simulator-inputs-v6.tar.gz` and
`corrected-motion-inputs-v1.tar.gz` from this release. Download
`fitted-motion-inputs-v1.tar.gz` from the preserved
[October 12 release](https://github.com/Swaguto/NVIDIA-V2D-Challenge/releases/tag/dex3-development-replay-20261012).
Keep archives compressed. Every output directory must be new; failed or partial
runs are preserved rather than overwritten. Examples below use `/data/v2d`;
replace that absolute directory with your own.

## 1. Verify and restore inputs without a GPU

Python 3.11.8 or newer; this check works on Windows and Linux and needs no
scientific or simulator packages. On Windows, use the same arguments on one
line with absolute Windows paths.

```sh
python scripts/reproduction/reproduce_frozen_dex3_replay.py \
  --archive /data/v2d/generated-dex3-simulator-inputs-v6.tar.gz \
  --sha256 25dcd06a5b141a6f6742be51669b4e5038ab6600f6228416c409baf8a59d1124 \
  --out /data/v2d/verified-inputs --check-only
```

Require `verified_files: 1654`, `all transferred input bytes verified`, and
case manifest SHA256
`87b027f05ca58259fa355a635ce6c1fed90a90a64ffcc84b9a2bed46d50880c0`.
This public command passed on Windows for the corrected package.

With NumPy available, independently check the stored certificate coverage and
numerical margins. This command passes with identical results on Windows and
Linux; it verifies the recorded proof, without running new geometry checks:

```sh
python scripts/reproduction/verify_corrected_dex3_certificate.py \
  --case /data/v2d/verified-inputs/case
```

Require 3,588 segments, 7,306 subintervals and 233,792 complete surface/material
records. The 132,907 explicitly stored margin bounds stay below 2 mm; their
maximum is `0.001999990496590006` m. The remaining records are swept AABB or
other declared complete-surface/rigid-hold proofs bound by the frozen report
hash, not newly recomputed geometry in this verifier.

## 2. Regenerate the corrected motion from fitted inputs

Use the recorded **Linux Python 3.11.17 geometry runtime**, including NumPy
2.2.6, SciPy 1.15.3, Pinocchio 3.7.0, Coal 3.0.1 and pin-pink 4.2.0. Windows
users can use Linux/WSL; ordinary Windows input verification does not install
this runtime. A Windows NumPy/SciPy-only environment was tested and refused
regeneration because `pink` was missing. Do not treat that environment as an
equivalent reproduction runtime.

Create a fresh environment with Python 3.11.17, and install the recorded lock:

```sh
python3.11 -m venv /data/v2d/geometry-runtime
/data/v2d/geometry-runtime/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r docs/team/reproduction/dex3-motion-replay/fitted-motion-runtime-lock.txt

/data/v2d/geometry-runtime/bin/python scripts/reproduction/reproduce_corrected_dex3_motion.py \
  --case /data/v2d/verified-inputs/case \
  --fitted-archive /data/v2d/fitted-motion-inputs-v1.tar.gz \
  --supplement-archive /data/v2d/corrected-motion-inputs-v1.tar.gz \
  --out /data/v2d/regenerated-motion
```

Require `all_payload_arrays_identical: true`, `array_count: 23`, payload SHA256
`91e41d5f44eef43f910b18ab8d24f64c9d4b8ad2b938ede7ec7af46a65bbab19`, and
knot SHA256
`2b60a5671f7a32f5f2db4954c8aaa822fae3112c7f9c786904c876e450baf6c3`.
The runner validates the case and all archived inputs, checks fit-parent
bindings and preserves every original observation. Its receipt identifies the
actual reproducer. Generation reports retain measured report templates with
new producer/parent hashes; report files are not expected to be byte-identical.

This step regenerates corrected explicit motion from saved fitted geometry and
profiles. It **does not rerun upstream reconstruction or geometric fit searches**.
The 51-file byte-preserving continuous method snapshot and its hash manifest
are committed beside these instructions. For a new full continuous check,
repeat the command with a fresh `--out` and `--check-continuous`. That check uses
no resumed prefix, takes additional CPU time and needs the full pinned geometry
runtime. Only require `new_continuous_check_complete: true` if you ran it.
The published complete certificate is the measured V11 check described above.

## 3. Run actual Isaac Sim

Use Linux with an RTX-capable NVIDIA GPU and required graphics libraries.
The measured simulator environment is Linux Python 3.12.15, Isaac Sim 6.0.0.1,
Isaac Lab commit `28a37cecdd433c22d9eabd6a5954add9f13a8951` (Lab VERSION 3.0.0).
It is separate from the CPU geometry environment. Accept the required NVIDIA
license. The unchanged published installer and 178-package version check were
used for the corrected run; install large runtime files on local disk.

```sh
python3.12 /data/v2d/verified-inputs/methods/setup_sim6.py \
  --lock /data/v2d/verified-inputs/methods/sim6-runtime-lock.txt \
  --prefix /data/v2d/runtime6 --lab /data/v2d/IsaacLab3

python scripts/reproduction/reproduce_frozen_dex3_replay.py \
  --archive /data/v2d/generated-dex3-simulator-inputs-v6.tar.gz \
  --sha256 25dcd06a5b141a6f6742be51669b4e5038ab6600f6228416c409baf8a59d1124 \
  --out /data/v2d/new-isaac-replay \
  --runtime-python /data/v2d/runtime6/bin/python --lab /data/v2d/IsaacLab3
```

Require `simulator/complete.json`, no `failure.json`, all 343 saved/decoded
frames, a passing independent audit, exact source observation masks and all
178 runtime pins. Review the video and saved frames for artificial flicker,
contact timing and cropping; the numerical audit alone is insufficient.
Stop your GPU after downloaded evidence passes its hash check.

## What another person must compare

Source, asset, method and trajectory bytes must match their SHA256 locks.
Compare all payload arrays, timestamps, observation/inference flags and the
receipt's runtime versions. Simulator state/camera tolerances are unchanged:
positions 1e-4 m, rotations/joints 1e-4 rad, intrinsics 1e-3 px. Video pixels and
compressed bytes can differ by GPU/driver; those bytes are provenance for the
recorded run, not a universal pixel-equality requirement.

If a hash, dependency, count or declared tolerance differs, preserve the failed
output and report the first mismatch and runtime versions. Do not alter masks,
relax depth limits or edit the expected evidence until it passes. Untouched
public episodes and full upstream reconstruction require a separately frozen
method and new measured accuracy evidence; this development replay cannot
substitute for that experiment.
