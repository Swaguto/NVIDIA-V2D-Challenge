# Issue 6 real-run validation helpers

## October 7, 2026: all 20 native episodes validated and preserved

All 20 required episodes completed all six stages and passed final visual review.
All 20 complete private checkpoints were transferred and independently verified
locally before stopping the A40 at 08:18 UTC. Earlier failed and partial results
remain separate historical evidence. The [final summary](all20-progress/quality-v3/final-summary.json)
and per-episode receipts bind acceptance to each actual executed configuration.

The final evidence covers 120 stage receipts, 10,300 native geometry checks,
51,673 input hashes, 84,920 contact rows and all 60 complete final videos.
Each video retains its original source axis: 5,246 source frames and 15,738
decoded review-video frames in total. Visual review combines normal-speed full
playback with uniform samples, consecutive motion extremes and large image changes.
It does not provide independent per-frame hand or contact labels.

Episode 41's approximately 20 cm depth failure now remains unknown. Final median
absolute stereo residual must stay strictly below the existing 2 cm threshold.
Episode 43's invalid negative virtual-camera estimate remains raw and unknown.
Episode 42 exposed a detector label temporarily assigning the missing left watch
hand to the right track. A uniform continuity guard rejects that identity conflict
without relabeling fits, adding detections, using manual regions or reading reference
poses. All 15 right fingertip rows at frames 45–47 are unknown and unbridged.
The full derived alignment was rerun; its global stereo scale and support were
rechecked, including changed support at right 133 and 152. The rejected predecessor's
2,910 files were separately archived and locally verified. The same guard audited
all 20 frozen candidate traces; other flags in 43 and 45 were already unknown.

Episode 22 uses automatic stereo-depth and image continuity association without
its historical manual exclusion. Episode 12's earlier large CAD drift was not
reproduced with the recovered real inputs and continuous tracking. Matched periodic
re-registration worsened its diagnostic results; continuous CAD remains fixed.
Residual CAD offsets, unknown holding intervals and imperfect hand coverage remain.
No sub-2-cm CAD accuracy, contact precision, complete recall or robot-training result
is claimed. Holdouts 0,11,23,31,39 were not used for episode-specific tuning.

The consolidated native installer was replayed in isolated directories on the pinned
CUDA image. Its repaired dependency pins, patch application and ViPE extension loader
passed imports. Bounded real replay then passed HaMeR inference on two frames/four
hands, ViPE camera reconstruction on all 181 episode-12 frames, and 362 native MANO
geometry comparisons. Actual sources, model hashes, package inventories and failed
attempt logs are preserved. This does not rerun all 20 fits from zero or validate the
Docker-host orchestration. Both Windows and the actual Linux worker passed all 105
CPU tests with zero skips; five identity regressions also passed in the fresh runtime.

Review speed improves by reusing immutable 50/300 fixed-shape fits, preparing episodes
in four independent slots, rendering the original source camera from existing meshes,
and overlapping review preparation with serial verified backup transfer. Unchanged
integrity configurations are reused; the changed episode is rehashed separately.
On the actual 2.69 GB model, an initial hash scan took 3.2 seconds and guarded repeats
took about 1–2 milliseconds. This is a repeated-check measurement, not an end-to-end
pipeline speedup. File changes invalidate the cache; recent POSIX files and Windows
files require byte scans. Acceptance never follows automatically from prefetching.

Private inputs and all result videos remain available in the locally verified
checkpoint archives. Licensed MANO assets and binary weights are not distributed
through GitHub. The compute pod is stopped and all existing volumes are retained.
The user-authorized cumulative compute/storage cap is US$15.

## Earlier all-20 pass: October 7, 2026

This pass is **in progress**, with final episodes **0,2,11,12,13,18,21,22,23,24,26,39 accepted (12/20)**
as of October7 04:50UTC. Source starts
from PR #32 `620ec548953d7ccd76890066512f6b364482e14d` plus the recorded source
repairs in this checkout. It replaces the historical manual episode-22
exclusion with automatic measured-depth/continuity selection. All 20 actual
footage episodes and detector passes are complete; full native reconstruction,
audits and final visual reviews remain required for the other8 episodes.
Accepted episodes have complete six-stage receipts, independent stage/contact
audits, all-frame persisted geometry checks and reviewed full-length videos.
Machine checks and video decoding are batched; uniform, motion-extreme and
large-image-change samples guide visual review without automatic acceptance.
The current [public receipts](all20-progress/progress.json) record all20 status,
twelve actual visual passes, stage/contact audits and twelve verified private
checkpoint summaries. Their execution configuration hashes identify the
preserved inputs and sources; a later documentation/scheduling commit is not
claimed to be the byte-for-byte source of already completed native runs.

The public split is pinned to dataset revision
`5f68335f3acc802033d1e80728c1633197521de8`: 106 files, 60 videos and 20 episode
tables. Every public file was checked against its upstream LFS SHA256 or Git
blob identity; every video was fully decoded. Licensed MANO files must be
supplied privately by a license holder. Neither the image nor GitHub supplies
those files. Use an authorized SSH channel for commands and hash-checked file
transfer; pod-management/logging MCP access alone cannot reach `/workspace`.
Keep API and SSH private credentials on the client, with an external stop
watchdog and an explicit cumulative compute/storage budget.

Recover the pinned public split using this directory's checked-in inventory:

```sh
python recover_public_inputs.py --root /workspace/track3/public \
  --inventory public-input-inventory.json \
  --receipt /workspace/pr32-evidence/real-input-receipt.json
```

This standard-library downloader verifies every byte count and upstream content
identity, including existing files, and refuses an incomplete inventory. It
downloads reference tables as part of the public split; inference adapters read
only frame axes, timestamps and object-name metadata, not reference object poses.

Prepare the two native environments on the pinned A40 image with
`bash recover_native_runtime.sh`. It records installed packages and source
hashes, pins the actual upstream code and direct dependency versions recovered
in this pass, applies the native compatibility/fixed-shape patch, and checks
imports. It preserves existing runtime checkouts and refuses a different commit.
This consolidated recipe has syntax checks; the underlying preparation and
native inference ran on the worker. A fresh run of the consolidated installer
is not yet independently validated.

Download FoundationPose's official weights folder with the ViPE runtime's
`gdown` (5.2.0), supply licensed MANO files privately, and recover/check the
remaining model inputs:

```sh
/workspace/vipe-runtime/bin/gdown --continue \
  https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i \
  -O /workspace/foundationpose-weights/
python recover_runtime_assets.py --root /workspace/pr32-models \
  --mano-dir /workspace/mano/models \
  --foundationpose-dir /workspace/foundationpose-weights \
  --receipt /workspace/pr32-evidence/runtime-assets.json
mkdir -p /workspace/hamer-weights
python -m tarfile -e /workspace/pr32-models/hamer_demo_data.tar.gz /workspace/hamer-weights
python recover_runtime_assets.py --verify-only --root /workspace/pr32-models \
  --mano-dir /workspace/mano/models --hamer-dir /workspace/hamer-weights \
  --foundationpose-dir /workspace/foundationpose-weights \
  --receipt /workspace/pr32-evidence/runtime-assets-extracted.json
```

The checked-in `runtime-asset-inventory.json` verifies the detector, hand
landmarker, SAM2 and HaMeR archive, both licensed MANO files, all four required
FoundationPose files and the extracted HaMeR checkpoint. Existing files must
also match. Public downloads are promoted from temporary files only after
verification. MANO is never downloaded by the helper or distributed in Git.
Copy the authorized MANO files into `_DATA/data/mano` in the extracted HaMeR
directory; symlink `/workspace/Dyn-HaMR/_DATA` to `/workspace/hamer-weights/_DATA`.
Keep the licensed distribution's license/readme with the private assets.

After preparing full calibrated inputs, automatic saved detections, the pinned
native runtimes and observed CAD at `/workspace/cad-final`, run
`bash run_all_episode_review.sh`. Continuous CAD tracking (`--reinit-every 0`),
fixed shape and 50/300 optimizer iterations apply to every episode. The script
preserves the earlier quality controls, uses `/workspace/issue6-final-inputs`,
`/workspace/issue6-full-runs/all20-final` and
`/workspace/pr32-evidence/all20-final-review`, and stops on any failed required
stage. `prepare_all_episode_manifest.py` attaches observed CAD and records the
explicit frozen evaluation purpose; development holdouts still reject tuning
and GT-assisted inference. Scoring against reference poses is a separate
diagnostic operation after inference.

The sequential split replay overlaps preparation and review on the same A40:
`prepare_episode_hands.sh EPISODES` prepares each verified fixed-shape fit, while
`review_fitted_episodes.sh EPISODES` waits for those fits and runs the six stages,
geometry checks, media and contact audits. The preparation queue has an
exclusive process lock. Use matching episode lists and do not run the serial
entry point on the same episodes concurrently. The one-time transition preserved
the active episode-0 pipeline before stopping its paused serial scheduling
parent. Quality episodes 12/22 retain their separate control/final queue.
The live preparation queue was subsequently partitioned into three disjoint
queues on that same A40 after preserving its active episode11 fit. Source-frame
loads1416/1374/1370 balance the remaining work. An observed GPU snapshot rose
from12% in serial fitting to100% while concurrent preparation ran, using24.6GB
of46.1GB. This is not yet a measured end-to-end speedup. No optimizer
iterations, models or acceptance gates were reduced.

The subsequent review scheduler consumes whichever verified fit is ready,
instead of waiting for episode-number order. It preserves the active episode2
pipeline before switching schedules. Automatic client-side evidence transfer,
independent audits and full video decoding prepare completed results for the
reviewer; visual acceptance remains a separate recorded review.

The live review queue now overlaps two independent reconstruction/media jobs
after preserving episode11's active pipeline. CPU rendering was observed
using one core for several minutes while other fits were ready. Episode13
started in the second slot without rerunning episode11 or reducing checks.
For a fresh replay, the self-contained entry point is:

```sh
python run_parallel_episode_review.py --workers 3 --review-workers 2
```

It balances disjoint episode queues by source-frame count, rejects live
competing schedulers, and reconstructs whichever verified fits are ready.
Every queue uses the same fixed-shape50/300 configuration and independent
episode paths. This consolidated entry point is syntax-checked; its underlying
native preparation and review scripts run on the worker, while the one-time
transition is separately recorded. A fresh consolidated parallel replay has
not yet been independently run. The progress collector is checked in here;
no private `/workspace/pr32-evidence` helper is needed for replay.

Private checkpoint transfers use authenticated SSH/SFTP with pinned host keys.
Each downloaded compressed archive is fully read, every member is checked,
all six stages match their configuration and outputs, and all locked inputs
are verified either inside the checkpoint or in separately recovered footage
and model assets. Early12/22 archives include verified locked-CAD supplements.
Archives0/2/11/12/13/18/21/22/23/24/26/39 are verified locally. Approximately
0.9GB transferred in70seconds for each of12/22. Transfer speed does not reduce
any verification step. Models, licensed assets, raw footage and full private
archives are excluded from Git.

The separate [frozen CAD diagnostics](all20-progress/frozen-cad-diagnostics.json)
cover all20 episodes. They are prediction/reference pose scores, not final
hand/contact acceptance. Single-object RPE is identically zero and cannot assess
pose quality. Episodes39/42 have MPPE about4.4cm and ADD AUC about0.56 despite
no gross tracking loss in the reviewed CAD samples. Episode39's final review
also records depth-unsupported, very small or omitted right-hand meshes during
parts of the grasp. Those contact observations remain unknown. The scorer's
missing-pose fill is used only for diagnostics; contact inference uses observed
poses and does not convert missing observations into measured contacts.

Episode43's first alignment failed on a single fitted left-hand estimate behind
the camera at original frame158. The native adapter now records a gate receipt
and omits invalid virtual-camera predictions from observed input records; raw
fits and converted geometry remain preserved, and affected contacts stay unknown.
Invalid calibration remains fatal. The gate's source hashes and rejected
observations are embedded in the locked manifest. The failed run is preserved;
its fresh reconstruction and visual review are still pending. Five new CPU
regressions include the actual negative-depth parameters. The parallel replay
scheduler records failed reviews separately, completes other independent jobs,
and still exits unsuccessfully if any failed episode remains unaccepted.

An [eleven-episode input audit](all20-progress/eleven-accepted-stage-input-integrity.json)
also rehashed actual locked inputs, implementation files and every stage output
for0/2/11/12/13/18/21/22/23/26/39. The required all20 audit remains pending and
does not substitute for the individual contact/video reviews or complete backups.

Review all full-length hand, contact and combined CAD/hand videos, timestamped
samples and consecutive motion-extreme samples before writing any visual pass.
Complete six-stage receipts, exact fixed-shape preservation, every persisted
MANO conversion check and independent contact audits are required as well.
Audit all six stages against their locked configuration, source and actual
output files; rehash original inputs before final acceptance:

```sh
python audit_stage_receipts.py --root /workspace/issue6-full-runs/all20-final \
  --episodes 0,2,11,12,13,18,21,22,23,24,26,27,31,39,40,41,42,43,44,45 \
  --modules-dir /workspace/pr32/reconstruction/modules --verify-inputs \
  --require-complete --output /workspace/pr32-evidence/all20-stage-integrity-audit.json
```

This standard-library audit independently recomputes the lock and file hashes.
Its completed-stage checks do not substitute for the contact artifact audit or
visual review. Pending or modified required stages produce a failing exit.
Unknown observations remain unknown. Detector-only review does not count as
episode acceptance. The earlier three-episode instructions below document the
native procedure; their acceptance does not satisfy the new all-20 gate.

These scripts record the native A40 validation procedure for PR #32. They are
experiment replay helpers with explicit `/workspace` paths, not a replacement
for the package CLI. Final acceptance is recorded in the module's
[validation record](../../modules/v2d_issue6/VALIDATION.md); a completed process
alone does not establish visual or contact accuracy.

## Runtime and inputs

- PR #33 image: `ghcr.io/swaguto/track3-runpod@sha256:46cc00a62d1076595b71401a96d17fbe53488afe1fe4d7a9784bf3512ac3e908`.
- Historical three-episode validation source: `/workspace/pr32`, commit `af6ed6fdd46b55e5ca0254246a1ad18275bfebd4`.
- Dyn-HaMR: `fa9cd7412c205fd15ee4139c8caacf79bf6167e6`, at `/workspace/Dyn-HaMR`.
- Enhanced HaMeR fork: `2a5b8ee852d87fdb56d65df7fb0e473c0d4f7648`.
- ViPE: `fdc79359ec6d063c86b9d203ca3d7054a9636823`, at `/workspace/ViPE`.
- SAM2: `2b90b9f5ceec907a1c18123530e92e794ad901a4`.
- Runtime: Python 3.11, PyTorch 2.5.1+cu124, NumPy 1.26.4. The hand runtime
  is `/workspace/hand-runtime/bin/python`; current ViPE uses
  `/workspace/vipe-runtime/bin/python`.
- Full rectified `ego_cam_c` bundles at `/workspace/issue6-full-inputs/e002`,
  `e013`, `e022`, with 343, 190 and 189 source frames. Use the Track 3 adapter
  to prepare metric stereo depth, calibrated intrinsics, observed CAD poses and
  masks. Depth here is calibrated SGBM stereo disparity. Reference object poses
  are not inputs; 58 absent episode-2 pot poses remain unknown.
- User-licensed MANO at `/workspace/mano` and
  `/workspace/hamer-weights/_DATA/data/mano`; actual HaMeR weights at
  `/workspace/hamer-weights`, detector and SAM2 weights at
  `/workspace/pr32-models`. Models, private credentials and raw data are not
  distributed here. The image alone does not supply the licensed assets.

Apply `dynhamr-native-fixed-shape.patch` to the pinned Dyn-HaMR checkout. Its
optional-import and zero-weight BMC changes permit the upstream defaults to run
without fake assets. HMP and BMC remain disabled by those defaults. Setting
`V2D_OPT_HAND_SHAPE=0` also fixes shape during smoothing. It applies to all 20
episodes in the current pass, following the historical episode-22 correction.

## Replay sequence

With the prepared inputs and runtime packages installed, copy these helpers to
`/workspace/pr32-evidence`. Use new output directories; preserve old results
before reruns. Keep the source checkout fixed while pipeline stages execute.

1. Run `enhanced_hands.py --episode N` with the hand runtime. Its current default
   saves every candidate, measured depth and automatic selection/rejection
   reason. `--detections-only` separates detector review from reconstruction;
   `--use-saved-detections` reuses those audited detector observations. Manual
   exclusion regions from the historical three-episode pass are superseded.
2. Run `vipe_stereo_cameras.py --episode N` with ViPE installed and the image's
   GPU runtime. Previously measured cameras were reused during targeted hand
   corrections, so an end-to-end fresh run need not reproduce bit-identical
   camera estimates.
3. Run `full_hand_masks.py --episode N` with the hand runtime. Actual SAM2 masks
   are prompted by detector boxes. Unsupported observations stay unknown.
4. Run `run_temporal_fit.py --episode N` with the hand runtime (50 root and 300
   smoothing iterations). Set `V2D_OPT_HAND_SHAPE=0` for episode 22. Verify the
   final checkpoint's betas equal the initial checkpoint before accepting it.
5. Run `convert_and_validate.py --episode N` with the hand runtime. It converts
   the final checkpoint, restores the original frame axis, applies observation
   gates, and executes validate/hands/depth/align/contacts/report.
6. Run `probe_dynhamr_geometry.py --episode N --all-frames --converted-records
   /workspace/temporal-hands/eNNN/converted`. Every persisted converted record
   must agree with upstream MANO vertices within `1e-5` metres. This checks
   conventions, not independent ground-truth accuracy.
7. Run `measure_temporal_consistency.py`, `measure_aligned_consistency.py`,
   `make_review_media.py` and `review_motion_extremes.py`, each with `--episode N`.
   Review sampled overlays plus consecutive frames around measured motion
   extremes. Measurements use estimated ViPE world coordinates.
8. Audit downloaded contact artifacts with `audit_completed_contacts.py --root
   REVIEW_ROOT`. Each `eNNN` directory contains the original relative
   `issue6-full-runs/calibrated-temporal/eNNN` paths. Unknown contacts must stay
   unknown, and bridging requires uninterrupted valid same-track observations.
9. After acceptance, `render_combined_overlay.py --episode N` renders observed
   CAD and final temporal MANO in one calibrated source-camera view. It uses the
   same CAD transforms as contacts, depth-tests hands and objects together, and
   omits missing poses. Inspect the video and its montage before delivery.

Save and verify all required evidence before stopping the GPU. Keep failed runs
as failure evidence, never count them toward the three-episode acceptance gate.
No independent contact precision or successful robot training is claimed.
