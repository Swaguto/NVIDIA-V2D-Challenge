# Validation and remaining gates

## October 6, 2026: three-episode acceptance passed

Validated source: `af6ed6fdd46b55e5ca0254246a1ad18275bfebd4`, native upstream
runtime on one A40 using PR #33's pinned image. All six pipeline stages completed
on 722 real rectified `ego_cam_c` frames. Actual HaMeR, SAM2, ViPE and Dyn-HaMR
outputs feed calibrated stereo ray alignment and observed-CAD contact distances.

| Episode | Source frames | Persisted MANO conversion checks | Maximum vertex error (m) | Valid / unknown contact rows | Raw / bridged contacts |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 343 | 676 | 1.84e-7 | 5560 / 1300 | 1503 / 127 |
| 13 | 190 | 380 | 1.72e-7 | 2290 / 1510 | 309 / 48 |
| 22 | 189 | 378 | 1.67e-7 | 1500 / 2280 | 224 / 17 |

All 1,434 persisted hand-frame comparisons pass the unchanged `1e-5 m`
conversion tolerance. Episode 2's fitted sequence starts after the first five
source frames; the output restores the original axis and leaves missing
observations unknown. The contact audit covers every row: timestamps/frame IDs,
predicted provenance, strict distance below 2 cm, null unknown observations,
and at most 15 consecutive valid same-track gap frames between raw-contact
endpoints. Counts are hand/fingertip/object/frame rows, not independent events.

Visual acceptance inspected 12 evenly spaced hand/contact frames per episode,
plus four consecutive-frame triples selected at measured motion extremes.
Reviewed meshes follow the wearer through grasp, transfer or pouring and release,
without the previously observed gross shape distortion or bystander identity
switch. Occluded hands are omitted; insufficient stereo support leaves contacts
unknown. This is sampled review, not independent annotation of every frame.
Full-length hand and contact videos were generated for all three episodes.
The separately requested combined CAD-plus-temporal-hand videos cover episodes
2 (343 frames) and 13 (190 frames). Episode 2 was fully decoded locally at
1280×800, and both videos passed sampled visual review and SHA256 transfer checks:

- Episode 2: `a6c8c243b61c751b017cc7a5b066353e2be6f33e18ca3bc00a2896ec87f85a5c`.
- Episode 13: `8564c2cd0ca14d9dd50c91d9e3565f9b726d1b7025b41c4dda2629aed67cc6cc`.

Episode 22 required fixed initial MANO shape and a manually reviewed background
region exclusion before detector selection. All candidates and failed runs are
retained. Its median joint acceleration before/after temporal fitting was
19.43/5.25 m/s² (left) and 21.97/6.32 m/s² (right), using directly observed
consecutive triples in estimated ViPE world coordinates. These are consistency
measurements, not ground-truth pose accuracy. Sparse stereo support remains a
limitation: episode 22 has 2,280 unknown contact rows out of 3,780.

The CPU suite passes **77 tests, zero skips**; Ruff lint/format checks pass.
The Docker-default backend was not exercised by this native run. No independent
contact precision, reference-pose accuracy or successful robot training is
claimed. No heavy optimization was added.

Machine-readable measurements and review scope are in
[acceptance.json](../../validation/issue6/acceptance.json); pinned runtime and
replay instructions are in the [validation helpers](../../validation/issue6/README.md).
At the user's pause checkpoint, all three final review bundles (including full
contact JSONs and videos) and both combined videos are downloaded and hash-checked.
All three task pods were verified `EXITED` with no live runtime; the temporary
account SSH key was removed. The full reproducibility archive was interrupted
before completion. RunPod then refused GPU and CPU-only recovery because the
host had no available GPU capacity. Retained volumes preserve the remaining
runtime files and incur storage charges. **Issue closure remains pending full
backup recovery and final PR integration.** Do not delete the retained volumes
or mistake the partial archive for a verified backup. Earlier entries below
describe historical states, not current acceptance.

## October 6, 2026: full-episode validation in progress

All three episodes have completed the six pipeline stages at least once; this
does **not** establish final acceptance. Episodes 2 and 13 passed sampled mesh
and contact review. Episode 22 exposed two failures: unconstrained MANO shape
drift and selection of a background bystander's hand. Failed outputs are retained.
Its replacement run fixes shape to the initial sequence-average HaMeR estimate
and excludes the manually reviewed background region before detector confidence
selection, retaining every candidate. This episode-specific review is not an
automatic general-purpose wearer identity method.

The fixed-shape checkpoint exposed rotation-conversion error near a half-turn.
Projecting rounded camera rotations onto SO(3), extracting the angle with atan2,
and preserving mixed-axis signs reduced the failing checkpoint's maximum vertex
error from 4.13e-5 m to 1.43e-7 m, below the unchanged 1e-5 m tolerance. Six new
regressions cover identity, small angles and float32 half-turns. **77 CPU tests
pass, with zero skips.** A broader check found additional affected frames in
episodes 2 and 13, so their downstream outputs are being regenerated too.

Final full-frame geometry checks, three-episode visual/contact acceptance,
verified evidence backup and compute shutdown remain pending. Earlier paused
entries below are historical. No independent ground-truth contact accuracy or
successful robot training is claimed.

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
