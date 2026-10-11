# Issue 6 / PR 32: paused validation checkpoint

Validation paused at the user's request on October 6, 2026, after episode 2's
depth stage. **Issue 6 remains unresolved and PR 32 has not passed acceptance.**
The user subsequently authorized publishing the repairs and this handoff.
No further GPU validation is authorized by that publication request.

## Repairs and measured checks

The repair baseline is PR 32 commit
`5b417857e9e64820e2e9d6dd8c3db18b2618334e`. Changes cover:

- Contact bridging only within an uninterrupted hand-track identity.
- Absolute CAD, observed-pose and mask paths in the Track 3 adapter.
- Selecting the numbered final Dyn-HaMR smooth-fit checkpoint and preserving
  its cropped original-frame axis.
- Explicit native alignment in PR 33's prepared CUDA environment; Docker remains
  the default and its execution path has not been validated by this run.
- Moving-camera MANO conversion about the shape-dependent wrist pivot.
- Preserving calibrated principal points and both focal lengths through
  temporal conversion and stereo alignment, with legacy HaMeR compatibility.

The repaired CPU suite passes **71 tests with zero skips**; Ruff lint/format and
compilation checks passed. The unchanged original PR passed 55 CPU tests but
failed actual visual acceptance: its three 64-frame clips each missed a visible
hand, and episodes 2/22 had incorrect hand identities. CPU tests alone do not
establish real hand/contact acceptance.

An earlier actual licensed-MANO vertex comparison agreed within approximately
`2.05e-7 m` after the wrist-pivot repair. A fresh geometry probe on the final
fit saved below is still pending. Actual fingertip ordering was checked on both
mirrored hands and two poses/shapes: thumb/index/middle/ring/little are joints
4/8/12/16/20, matching vertices 745/317/444/556/673 exactly in that probe.

## Saved stage boundary

| Episode | Full frames | Actual per-frame hand predictions | Temporal / pipeline status |
| --- | ---: | --- | --- |
| 2 | 343 | Both hands saved | ViPE cameras, SAM2 masks, Dyn-HaMR root 50 / smooth 300 complete; validate/hands/depth complete; alignment partial |
| 13 | 190 | Both hands saved | Cameras, masks, temporal fit and six pipeline stages pending |
| 22 | 189 | Both hands saved; wearer review recorded | Cameras, masks, temporal fit and six pipeline stages pending |

All 722 calibrated full-episode inputs and observed CAD poses/masks are prepared.
Depth comes from calibrated OpenCV StereoSGBM disparity, not FoundationStereo.
Public parquet is used only for names, timestamps and frame counts; reference
object poses do not enter these inputs. Public data revision:
`5f68335f3acc802033d1e80728c1633197521de8` of
`nvidia/video_to_data_challenge`.

Episode 2 has 343 SAM2 mask files per side, with 325 directly observed left and
336 directly observed right samples. Its optimized interval is `[5, 343)`:
338 frames, 676 converted records, 661 retained directly observed records.
Occluded optimizer estimates remain in the raw NPZ but do not become contact
observations. Missing hand, stereo support or object pose observations remain
unknown. Episode 22's wearer review excludes 15 bystander right-hand detections
(frames 0–13 and 15); originals and the review receipt are preserved. This manual
identity exclusion is not independent contact labeling.

Saved final fit:
`/workspace/temporal-hands/e002/fit/smooth_fit/e002_000300_world_results.npz`.
Run root: `/workspace/issue6-full-runs/calibrated-temporal/e002`.
The alignment receipt says `running` because the worker was frozen at the pause;
it is **partial**, not a successful result. Contacts/report are absent. The full
task process group was verified stopped before the backup was made.

## Evidence and runtime

The operator has a locally verified `pause-checkpoint-20261006/evidence.tar.gz`:
1,306,600,317 bytes, 12,914 data members, SHA-256
`e850bf2fcf58a1d6aa11482c042ce56dcdfee905e7efd92c4fa77e4e346f2862`.
Every member hash matches; 2,439 identical copies use tar hardlinks. The archive
contains final/initial fits, cameras/masks, calibrated inputs, observed CAD
outputs, reusable JSON MANO predictions, completed and partial stage receipts,
source/runtime patches, logs and model-source hashes. Intermediate fit files,
model files and credentials are excluded. Footage, evidence archives and licensed
MANO assets remain outside Git. Obtain the operator's checkpoint and licensed
assets through authorized channels before attempting recovery.

The model-free orchestration/probe helpers are saved with the local handoff in
`runtime-helpers.tar.gz` (29 files), SHA-256
`72f5b03d6b6d9dd9afb7e5cec8eec1696f8a723e00831638cdb6ad77058d8ffd`.
The complete detailed operator handoff and verification receipts remain alongside
the local archive. This document records their state; it does not claim the
bulk artifacts are downloadable from this repository.

PR 33 image used:
`ghcr.io/swaguto/track3-runpod@sha256:46cc00a62d1076595b71401a96d17fbe53488afe1fe4d7a9784bf3512ac3e908`.
Torch 2.5.1/cu124, Python 3.11, NumPy 1.26.4. Runtime revisions:

- HaMeR: `3a01849f4148352e9260b69bf28b65d1671a4905`.
- Dyn-HaMR: `fa9cd7412c205fd15ee4139c8caacf79bf6167e6`.
- Enhanced HaMeR: `2a5b8ee852d87fdb56d65df7fb0e473c0d4f7648`.
- SAM2: `2b90b9f5ceec907a1c18123530e92e794ad901a4`.
- ViPE: `fdc79359ec6d063c86b9d203ca3d7054a9636823`.

ViPE's actual SM86 CUDA extension import passed. HMP/BMC remain disabled by
upstream zero-weight defaults, without placeholder assets or new objectives.
The native compatibility patch is preserved in the archive, SHA-256
`469b5244f717b397635b697ce4c0878ed884562a90e3134d9177e33b70fa6e17`.
A MediaPipe/protobuf mismatch remains in an unused runtime path; global dependency
consistency has not been established.

## Resume after explicit user instruction

Use one A40 at a time, verify source/model hashes, and install a fresh budget/time
cutoff before starting. The October 6 cutoff cannot be reused for a later run.
The original US$10 total cap includes previous compute and ongoing storage.
At the pause, estimated compute was US$5.6096 before storage; provider invoicing
was not independently verified. Both task pods were verified `EXITED` with no
live runtime at 17:52:55 UTC (13:52:55 America/Toronto); the temporary account
SSH key was removed. Two retained task volumes total 120 GB and still bill storage.

Restore inputs at the same `/workspace` paths. Do not blindly rerun the e2
converter or the full launcher: completed immutable stage directories already
exist. Preserve the paused run and copy only its completed stages into a fresh
active run, keeping the original partial alignment untouched:

```bash
export PYOPENGL_PLATFORM=egl
saved=/workspace/issue6-full-runs/calibrated-temporal-pause-20261006/e002
active=/workspace/issue6-full-runs/calibrated-temporal/e002
test ! -e "$saved"
mkdir -p "$(dirname "$saved")"
mv "$active" "$saved"
mkdir "$active"
for name in inputs.lock.json validate hands depth; do
  cp -a "$saved/$name" "$active/$name"
done
for stage in align contacts report; do
  /workspace/hand-runtime/bin/python -m issue6.cli "$stage" \
    --manifest /workspace/issue6-full-inputs/e002/manifest.temporal.json \
    --output "$active"
done
```

Existing manifest/input/implementation hashes must pass. If they differ, retain
the evidence and use a fully fresh run; do not weaken the receipt checks. Reuse
the completed e2 fit. Upstream partial optimizer-state resume has stale reference
and gradient-flag problems; restart incomplete fits in fresh directories.

Remaining acceptance work:

1. Finish e2 alignment/contacts/report and write its final validation receipt,
   explicitly recording reuse of the saved fit and completed stages.
2. Run camera reconstruction, SAM2 masks, temporal fitting and the pipeline for
   episodes 13/22 with the saved actual predictions and matching model hashes.
3. Run geometry and temporal/aligned consistency probes on all three episodes.
   Estimated-world consistency does not establish ground-truth accuracy.
4. Generate and review full-episode mesh/contact videos and timestamped panels
   for all three, especially identity, grasps/releases, gaps and stereo support.
   Contact precision remains unknown without independent contact labels.
5. Save final evidence and report acceptance or remaining defects. **Turn off
   compute after finishing or failure**, verify shutdown, and remove task storage
   only after needed data are safely backed up.

These experiments have not demonstrated robot training, policy performance,
independent world/contact accuracy or competition improvement.
