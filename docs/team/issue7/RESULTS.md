# Issue 7: SLAM-backed episode-2 refinement

The final candidate is **trial 4**, completed October 9, 2026. It satisfies
issue 7's one-episode requirements: real metric SLAM initialization, joint
refinement, a browser-verified 3D viewer, and written jitter/drift/contact
precision measurements. [PR #34](https://github.com/Swaguto/NVIDIA-V2D-Challenge/pull/34)
is ready for review and issue closure on merge. Issue 7 remains open; no merge
or issue closure has been performed. This supersedes the older odometry pilot.
Current software checks and later repairs are in [REVIEW.md](REVIEW.md);
`RESULTS.json` retains the timestamped experiment assessment.

Baseline: Thanosan's `2e8efa668c9485fbd99f961ff88284a905f2b021`, with repairs on
`Evc-r/issue7-refinement-validation`. All six live team-document tabs were
refreshed October 9 before the consequential scope decision. This result covers
one development episode; upstream issues 4–6 and robot training have separate
acceptance criteria.

## Requirement assessment

| Issue-7 requirement | Completed evidence |
|---|---|
| Gaussian scene anchored to hands, objects and background; SLAM camera | Optimized anchored Gaussian state, both meshes/hand tracks/background, 343 actual DROID-SLAM camera poses |
| Joint photometric + stereo depth + segmentation + temporal objectives | All four loss families active; 860 successful updates; object, hand root/finger and camera poses optimized together |
| Per-episode jitter, drift and contact precision | Matched report; source-reviewed stationary interval; independent hand-object precision with label coverage and missing geometry |
| One full episode viewable in 3D, jitter/drift written | All 343 frames, browser control/loop checks, complete source/render review, numeric Markdown/JSON |

## Actual run and integrity

| Evidence | Result |
|---|---|
| Episode / frame IDs | 2 / 0–342, nominal 20 fps |
| Successful / skipped optimizer updates | **860 / 0** |
| Successful visits per frame | **10 for every frame** |
| Camera and each object export | 343, finite |
| Pot observed / inferred initialization | 285 / 58; provenance retained |
| Lid observed initialization | 343 |
| Metric-valid left / right hand frames | 272 / 284; missing observations stay missing |
| Original source/input files rechecked | 4,673; zero changed |
| SLAM-staged input files rechecked | 2,961; zero changed |
| Runtime | Python 3.11.17, PyTorch 2.9.1+cu128, gsplat 1.4.0, CUDA 12.8 |
| Hardware | RTX 5060 Laptop GPU; native WSL execution |

Scene SHA-256:
`3e3f51aa85c60f0d68384de4896e2f6c8ca0fc1cf3891d83d77c023676a9bc6a`.
The exact configuration, source hashes, complete numerical measurements and
requirement assessment are in [RESULTS.json](RESULTS.json). The frozen optimizer
source hashes identify the actual executed GPU run. The report's later viewer-only
fix leaves scene bytes and measurements unchanged. Subsequent frame-gap and
launcher/provenance fixes in the current source are CPU-tested separately; see
[REVIEW.md](REVIEW.md). They have not been rerun on the GPU.

## Real metric SLAM and numerical repairs

The previous 343 stereo-PnP odometry poses did not satisfy the SLAM requirement.
The local stage now runs the repository's DROID integration on rectified
`ego_cam_c` RGB and calibrated stereo depth, with no public trajectory GT.
Its frontend initialized; eight keyframes underwent local and global bundle
adjustment, and 343 finite camera-to-world poses were exported in metres.
Global BA passes used 7/12 iterations. Peak SLAM tensor allocation was 1.24 GB.
The eight median map-versus-stereo depth errors were 0.04–0.43 mm. This checks
agreement with the sensor prior; it is not independent physical ground truth.
The earlier four-keyframe attempt never initialized and is rejected/preserved.

The adapter requires declared SLAM provenance by default, rejects GT-assisted
and ambiguous transforms, and retains original camera-relative object/hand
geometry when changing the world gauge. All 343 scene initial cameras match the
SLAM source after reanchoring (maximum matrix error 1.35e-6). This supplies the
camera input needed for issue 7; it does not certify every deliverable in issue 4.

The numerical failure was reproduced on CUDA: gsplat 1.4 padded seven feature
channels with uninitialized memory; a NaN in the unused channel caused nonfinite
scale gradients. Explicit zero padding removes that path. The controlled backward
comparison and the full zero-skip run are preserved. Inverse-depth resize uses
nearest-neighbor measurements and keeps invalid pixels unknown. Camera
translation/rotation acceleration priors reduce jitter without penalizing constant
velocity. Temporal hand/object terms avoid bridging missing observations.

Ten epochs, batch size 4, half-resolution training, seed 42. Gaussian learning
rate 0.001; object/hand/finger rates 0.00005; camera rate 0.00001; cosine minimum
factor 0.1. Mask weight 50, metric stereo depth 100, relative depth 1, first-order
temporal weights 1,000, camera acceleration weights 1e6. CAD scale, shared hand
shape and hand scale were frozen. The shared beta starts at the accepted-frame
mean, rather than retaining different per-frame shapes.

Initialization-relative limits were objects 3 mm/0.5°, hand roots 2 mm/0.5°,
each finger axis-angle block 1°, cameras 2 mm/0.25°. Exports respect these
limits within float32 tolerance. They are engineering priors, not accuracy
bounds. Reference poses and contact validation labels did not enter optimization.

## Matched measurements

All derivatives use nominal 20 fps (source container average approximately
20.254 fps). Object rows below use the same observed initialization frames in
both states; temporal stencils exclude gaps. Jitter alone does not establish
accuracy.

| Track | High-frequency residual, mm | Acceleration RMS, m/s² |
|---|---:|---:|
| Camera | **0.48 → 0.20** | **0.34 → 0.08** |
| Pot, 285 observed frames | 3.06 → 2.93 | 1.67 → 1.40 |
| Lid, 343 observed frames | 1.55 → 1.36 | 1.11 → 0.70 |
| Left wrist | 3.05 → 2.73 | 2.18 → 1.72 |
| Right wrist | 3.18 → 2.91 | 2.40 → 1.93 |
| Left fingertips | 3.48 → 3.28 | 2.53 → 2.22 |
| Right fingertips | 4.80 → 4.71 | 4.31 → 4.12 |

Camera angular acceleration improves 64 → 34 deg/s²; observed pot/lid angular
acceleration improves 560 → 362 and 755 → 406 deg/s². Actual mean camera
correction is 0.72 mm/0.073°, maximum 2.00 mm/0.244°.

For source-reviewed stationary lid frames **112–228**, selected before results,
endpoint drift improves **1.9 → 1.1 mm** and angular drift 1.01 → 0.98°.
Maximum deviation slightly rises 2.8 → 3.0 mm. The separate unreviewed
far-from-hands pot-rest proxy regresses in endpoint drift 5.1 → 6.7 mm; it
does not establish actual stationary motion. These regressions remain reported.

For 44 fixed predicted contact runs, all 1,494 expected rows have geometry.
Within-2-cm geometric consistency is 95.52% → 95.58%, median surface distance
7.0 → 6.7 mm, touch-point spread 6.8 → 6.3 mm and normal spread 26.6 → 26.5°.
The oriented-surface penetration proxy is 8.40% → 8.20%; its p95 depth remains
large (66.9 → 66.3 mm) and is not solid-volume collision truth.

## Independent contact precision

Source-video annotations reviewed October 7, before the fresh reconstruction
metrics, supply definite grasp cores and explicit separated intervals. The
reference has 1,372 hand/object/frame labels: 381 positive, 182 negative and
809 unknown. Grasp boundaries stay unknown. These labels were frozen before
trial 3/4 metrics and never supplied to refinement.

At **hand-object granularity**, conditional precision is **100% → 100%**
(367 true positives, zero labelled false positives), with positive-prediction
label coverage **94.34% → 94.83%**. Refined predictions include 20 positives
without independent truth; these are not counted as correct. Recall is 96.33%
and known-label geometry coverage 93.96%. The 34 missing geometry rows include
14 positive labels, retained in the recall denominator.

The report supports independent fingertip labels when available. This episode's
source annotations establish coarse hand-object contact only; **per-fingertip
and metric 3D contact truth remain unavailable**. The conditional 100% result
is not a claim of perfect contact accuracy across all frames or fingers.

Public-reference diagnostics ran only after freezing the output: ADD AUC
0.972 → 0.973, RPE 0.817 → 0.765 cm, MPPE 0.939 → 0.925 cm; SP/MP success
values remain 1.0. These are reconstruction diagnostics, not learned-policy
rollouts or final challenge scores. Holdouts 0/11/23/31/39 were not tuned.

## Visual and software verification

All 343 source/render pairs were inspected in 29 contact sheets, plus 12
full-size event pairs. Pot/lid lift, motion and replacement remain visible;
no gross mesh collapse or teleport was observed. Sparse background holes,
Gaussian appearance speckle, absent forearms and missing upstream hand frames
remain. This is not a photorealistic scene.

The HTML viewer shows optimized mesh geometry and background points;
`overlay.mp4` renders the actual Gaussians. All optimized Gaussian state is
retained in the scene archive. Keeping complete CAD meshes up to 100,000 faces
fixes viewer decimation speckle without changing optimization. Browser checks
covered frames 0/171/342, both geometry states, background/camera toggles,
orbit/zoom/reset, play/pause and wrap-around. The comparison MP4 reached its
17.15-second end without media or console errors. Complete still-frame review
plus playback verification does not claim a continuous visual audit at speed.

**65 targeted CPU tests pass**, together with targeted Ruff, Python syntax and
Bash syntax checks. Tests cover SLAM/odometry provenance, different world gauges,
independent contact labels/unknowns, zero-padding gradients, inverse-depth holes,
gap-aware acceleration, bounds and Docker argument forwarding. The actual GPU
run is separate evidence. Docker image execution was not tested; rebuild with
this source before use. Inherited unrelated SDK style warnings are outside the
targeted lint scope.

## Reproduction and preserved history

Use [REPRODUCE.md](REPRODUCE.md). Current private evidence is under
`runs/issue7-validation/trial4` relative to the parent workspace;
complete native outputs are `/var/tmp/v2d-issue7-e002/trial4`. SLAM evidence
is separately preserved in `slam-trial2`. Review server:
`http://127.0.0.1:8138/`, with the full comparison MP4 available for download.
Licensed MANO, footage, models and scene assets remain local.

Rejected trial 1, historical trial 2 (840 updates/20 skips, odometry cameras),
repaired odometry control trial 3, and rejected SLAM trial 1 remain preserved.
Their original archives are unchanged. Trial 4's original archive retains its
initial numerical checkpoint; the additive completion supplement contains the
final viewer, visual receipt and software/docs evidence. The old requested
trial-2 viewer recording remains explicitly historical.

The issue-7 one-episode milestone is complete and can be closed with this PR
after review and merge. Background completion, upstream missing-hand recovery
and stronger independent contact truth remain follow-up quality work. They do
not change this issue's stated one-episode milestone. Issue 6's all-20 acceptance,
runtime and budget are unchanged by this work.
