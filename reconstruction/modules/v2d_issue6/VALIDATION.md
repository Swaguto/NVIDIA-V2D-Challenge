# Validation and remaining gates

## October 6, 2026: repaired source and paused real-run checkpoint

The repaired CPU suite passes **71 tests with zero skips**; Ruff lint/format and
compilation checks pass. Actual enhanced YOLO/HaMeR predictions are saved for
all 722 frames of episodes 2, 13 and 22. Episode 2 completed ViPE camera
reconstruction, SAM2 masks, Dyn-HaMR root/smooth fitting (50/300 iterations),
and validate/hands/depth stages. Alignment was paused in progress; contacts and
report remain absent. Episodes 13/22 still need camera/mask/temporal/pipeline work.

The original PR's real 64-frame baseline failed visual acceptance (missing hands
and incorrect identities). The repaired three-episode acceptance gate remains
open. Both task GPU pods were verified stopped, and a complete local checkpoint
archive passed all 12,914 member hashes. See
[the checkpoint handoff](CHECKPOINT_2026-10-06.md) for exact evidence, runtime,
remaining checks and safe resume commands. The earlier milestones below are
historical; their asset blockers and unrun-inference statements describe those
earlier runs.

## Track 3 integration on 2026-10-05

The `feat/issue6-stereo-contact` branch adds a rectified `ego_cam_c` adapter on
top of the issue-6 CPU integration. It was exercised on the team's Thunder
A6000 host, with Track 3 outputs from the completed public run. Three 64-frame
bundles passed the manifest and calibrated depth stages:

| Episode | Original frames | Objects with directly observed CAD poses | Initial stereo-valid pixels |
| --- | --- | --- | ---: |
| 2 | 60–123 | white pot 64/64; lid 64/64 | 50.8% |
| 13 | 60–123 | blue cup 64/64; dish rack 64/64 | 52.3% |
| 22 | 30–93 | blue cup 64/64; water pitcher 64/64 | 57.0% |

Full input bundles also passed the manifest and depth stages: episode 2 has
343/343 depth files, 285/343 observed pot poses and 343/343 observed lid poses;
episode 13 has 190/190 depth files and observed poses for both objects; episode
22 has 189/189 depth files and observed poses for both objects. The 58 missing
episode-2 pot poses remain unknown contact inputs, with no inferred pose silently
substituted.

The frame windows were chosen from the public rectified video to include
visible hand-object activity and directly observed object poses. The adapter
uses public parquet rows only for object names, frame IDs, and timestamps. It
does not read reference poses. The VM bundles and CPU stage receipts are under
`/home/ubuntu/issue6_runs/`; the input root is
`/home/ubuntu/track3_all_20261004`. The current local CPU suite passes 55
tests, including `ego_cam_c` and rejection of inferred/wrong-frame object poses.

**The three-episode acceptance gate remains open.** The VM has no
`MANO_RIGHT.pkl` or HaMeR checkpoint at the checked locations, and its Docker
daemon refused the `v2d_mediapipe` build with `unshare: operation not permitted`.
After installing Docker Buildx, even a two-line Alpine smoke build failed to
mount a BuildKit work directory with `operation not permitted`; this confirms
the current daemon cannot build the required images.
No real MANO hand inference, contact event or hand/contact overlay was produced
by this run. Stereo-valid pixel fraction across the full image does not establish
depth support on the hands; that still needs checking after hand masks exist.

Implementation session: September 28, 2026.

## September 28, 2026: local CPU evidence

**53 tests passed** in the local CPU environment. Python compilation and the
installed command-line entry point also passed. This validates the CPU path and
adapter contracts; it does not validate the GPU execution path.

PR preparation base: `ca3f64a` (current `origin/main` when fetched September 28).
Local runtime: Windows, Python 3.14; NumPy 2.5.3, Pillow 12.3.0 and trimesh 5.1.0.
Ruff lint and formatting, the repository's root-file hygiene check and its
agent-context check passed. No root/reconstruction pre-commit configuration exists;
the unrelated ingestion and grounding hook configurations were not run.

`.github/workflows/issue6_cpu_ci.yml` adds Linux Python 3.10/3.12 CPU checks.
**Those hosted jobs have not run yet**; they run when the PR is opened or manually
dispatched. No GPU CI, cloud provisioning, model download or external handoff
message was performed.

The package includes CPU tests for:

- Three depth encodings, invalid values and explicit validity masks.
- Calibrated depth reprojection, transform direction, z-buffer collisions,
  camera bounds, missing depth and refusal of distorted-camera inputs.
- Left-hand mirroring and identical mesh-centroid scaling for joints/vertices.
- Finite masked depth alignment and insufficient-depth invalidation.
- Nearest-triangle contact distances, strict 2 cm threshold, closed-surface
  penetration, and unknown penetration for open geometry.
- Gap boundaries, 15 versus 16 frames, missing observations and frame gaps.
- Pose conventions, units, timestamp/frame axes, holdout exclusions and
  predicted-object outputs retaining GT-assisted camera provenance.
- Conflicting camera pose declarations and invalid contact settings fail early;
  inward-facing mesh normals are marked ambiguous.
- LFS pointer rejection, stale inputs/outputs, failure receipts, and duplicate
  hands remaining unknown.
- Original-frame preservation through the upstream renderer's zero-based indexing.
- Preparing CAD occlusion inputs when no color-view object mask is available.
- A complete CPU synthetic integration test through import, projection,
  contact extraction, report and per-frame overlays. **That test replaces the
  GPU worker with synthetic hand coordinates. It is not an inference test.**

Run from the repository root:

```sh
python -m pytest reconstruction/modules/v2d_issue6/tests -q
python -m compileall -q reconstruction/modules/v2d_issue6/issue6
python -m issue6.cli --help
python -m ruff check reconstruction/modules/v2d_issue6
python -m ruff format --check reconstruction/modules/v2d_issue6
```

## Unverified at the original CPU checkpoint (historical)

- Actual HaMeR / ViPE / Dyn-HaMR execution or rendering in the team GPU images.
- Actual MANO assets, joint ordering from the installed manotorch version, and
  end-to-end agreement between the installed renderer and exporter on real fits.
- Episode-2 inputs from the shared VM, including current pose-quality reports,
  color-camera rectification and stereo-left-to-color calibration.
- Contact accuracy, hand-pose accuracy, policy training, official evaluation,
  or any competition improvement.
- Three completed, visually reviewed episodes.

The first GPU validation is the 64-frame episode-2 run in README.md. Check hand
identity and joint order, compare the exported joints against the rendered mesh,
inspect depth residuals and grasp/release timing, then run the complete episode.
Use separate HaMeR and Dyn-HaMR runs for a matched diagnostic comparison. Do not
replace unknown observations with bridged contacts or label GT-assisted inputs
as an independent predicted reconstruction.

## Required handoff

The producer-to-consumer contract and review record are in [HANDOFF.md](HANDOFF.md).

1. Shared-host connection and runnable existing reconstruction images.
2. Authorized MANO asset plus pretrained model paths.
3. Episode-2 color frames with their actual timestamps and rectified intrinsics.
4. Stereo-left depth, encoding/validity, and depth-to-color transform.
5. Object meshes and per-frame poses with quality flags and provenance; color
   masks if available (valid CAD poses are the occlusion-mask fallback).
6. The analogous episode-12 and episode-13 bundles for subsequent validation.

Large generated files remain outside source control. This contribution does not
alter the existing contact-reliability package or the team training pipeline.
