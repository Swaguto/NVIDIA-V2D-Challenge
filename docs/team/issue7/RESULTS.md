# Issue 7: completed episode-2 refinement pilot

Episode 2 now meets issue 7's local milestone: all 343 frames were refined on
the laptop GPU, a working 3D viewer was checked in the browser, and before/after
jitter, drift and contact-consistency measurements are written. The candidate is
trial 2. GitHub issue 7 remains open for draft-PR review; publication was
authorized by the user on October 8, 2026.
The unconstrained first trial is preserved and rejected for geometry/contact
regressions. This result does not complete upstream issues 4–6 or robot training.

Baseline: Thanosan's `2e8efa668c9485fbd99f961ff88284a905f2b021`, with the repairs
in `Evc-r/issue7-refinement-validation`. Run: October 8, 2026,
23:10:24–23:27:31 UTC. Live team context was refreshed across all six Google Doc
tabs that day. The issue's acceptance milestone is one episode, not all twenty.

## Actual run and preserved coverage

| Evidence | Result |
|---|---|
| Episode / frame IDs | 2 / 0–342, nominal 20 fps |
| Successful optimizer updates | 840 of 860 scheduled batches |
| Guarded skipped updates | 20 nonfinite Gaussian-scale gradients; no update applied |
| Successful visits per frame | 7–10; histogram: 7:1, 8:5, 9:67, 10:270 |
| Camera and both object exports | 343 each, finite |
| Pot initialization | 285 observed, 58 inferred; provenance retained |
| Lid initialization | 343 observed |
| Accepted left / right hand exports | 272 / 284; missing observations stay missing |
| Original input/source hashes rechecked | 4,673; zero changed files |
| Runtime | Python 3.11.17, PyTorch 2.9.1+cu128, gsplat 1.4.0, CUDA 12.8 |
| Hardware | NVIDIA GeForce RTX 5060 Laptop GPU, native WSL runtime |

The isolated CUDA extension passed an actual 32×32 render/backward preflight.
The optimizer completed with skips, rather than an assertion of zero numerical
failures. The underlying Gaussian-scale backward instability remains a follow-up;
the finite-gradient guard prevents those batches from corrupting parameters.
Every frame received successful updates, and all exported scene arrays are finite.

The scene retains the optimized Gaussian state, initial/refined geometry,
cameras, joints and frame indices. Gaussian anchor offsets remained zero in this
run. The HTML viewer displays the optimized meshes and background point cloud;
`overlay.mp4` renders the actual Gaussian appearance. The viewer is not a splat
shader. Display-only fixes after the run did not change scene bytes or metrics.

Scene SHA-256:
`e8e58a896f05f786b39d01bdc1983e9a9eeb579f0cf0d11c7a0d99fdbc2df8e2`.
Input-lock SHA-256:
`1c85ea4a6a64c5bd0a38fcce0cd72975fac76f8e39e7a0df700b7d13787d0fac`.
Exact run-source hashes and configuration are in [RESULTS.json](RESULTS.json),
with the private frozen-source snapshot retained beside the actual run receipt.

## Configuration and integrity repairs

Ten epochs, batch size 4, half-resolution training, seed 42. Gaussian learning
rate 0.001; object/hand root and finger learning rates 0.00005; camera rate
0.00001; cosine schedule with minimum factor 0.1. Photometric, segmentation,
calibrated metric/relative stereo-depth and world-frame temporal losses were
active. Mask weight 50, metric-depth weight 100, relative-depth weight 1 and
each temporal weight 1,000. CAD scale, hand shape and hand scale were frozen.
Each hand's shared shape starts at the mean beta of its accepted input frames;
this does not preserve different raw per-frame betas as separate shapes.

Initialization-relative correction limits were fixed before trial 2:
objects 3 mm / 0.5°, hand roots 2 mm / 0.5°, each of 15 finger axis-angle
blocks 1°, and cameras 2 mm / 0.25°. Zero disables a limit. These are engineering
priors, not measured uncertainty or accuracy bounds. Actual exported corrections
respect them within float32 tolerance; the maximum camera translation was
1.738 mm and physical rotation 0.134°.

The adapter rejects absent independent cameras, GT-assisted initialization,
nonfinite/degenerate transforms and mixed hand sides, and defaults to metric-valid
hand observations. Cameras here come from independent stereo-PnP initialization;
this is not evidence that issue 4's SLAM work is complete. Temporal losses avoid
smoothing across missing hand frames. Nonfinite depth cannot contaminate masked
loss gradients. Zero-update runs and nonfinite scene exports are rejected.

Report writing uses explicit UTF-8. Fixed predicted contact runs and matched
before/after coverage prevent post-refinement relabeling or denominator inflation.
Rest candidates require observed hands and remain explicitly labeled as proxies.
The viewer frames transformed world geometry correctly, removes duplicate display
faces while preserving winding, and prevents the initial ghost from occluding
the refined scene. The native and Docker launchers expose the same pose limits;
the actual GPU experiment used native execution.

## Matched measurements

All derivatives use nominal dataset 20 fps, not the container average of about
20.254 fps. Object rows use matching observed initialization frames; temporal
stencils exclude gaps. Lower jitter is not an independent accuracy measurement.

| Track | High-frequency translation residual, mm | Acceleration RMS, m/s² |
|---|---:|---:|
| Pot, 285 observed frames | 3.04 → 2.81 | 1.66 → 1.38 |
| Lid, 343 observed frames | 1.54 → 1.30 | 1.08 → 0.71 |
| Left wrist | 3.02 → 2.59 | 2.15 → 1.64 |
| Right wrist | 3.20 → 2.74 | 2.42 → 1.79 |
| Left fingertips | 3.44 → 3.17 | 2.51 → 2.24 |
| Right fingertips | 4.83 → 4.70 | 4.29 → 4.16 |
| Camera | **0.51 → 0.61** | **0.33 → 0.50** |

Object angular acceleration RMS improves from 559 → 366 deg/s² for the pot and
755 → 402 deg/s² for the lid. Camera angular acceleration regresses from
63 → 81 deg/s²; right fingertip acceleration p95 also regresses, despite lower
RMS. These regressions remain in the full numerical report.

The source-reviewed stationary lid interval 112–228 was fixed before the runs:
endpoint drift 1.18 → 1.05 cm, maximum deviation 1.34 → 1.24 cm, angular endpoint
drift approximately 0.60 → 0.60°. A separate inferred far-from-hands rest proxy
regresses for the lid (0.51 → 0.64 cm); it does not override the source-reviewed
interval or establish true rest.

For 44 fixed predicted contact runs, all 1,494 expected rows have geometry in
both scenes. Within-2-cm consistency improves 95.5% → 96.5%, median surface
distance 7.0 → 6.6 mm, and contact-point spread 6.8 → 6.5 mm. Normal spread
regresses 26.6 → 26.9°. The oriented-surface penetration proxy regresses slightly
(8.40% → 8.53%); it is not a collision ground truth metric. Independent per-finger
3D contact precision is **unavailable**, not inferred from predicted labels.

Public-reference reconstruction diagnostics ran only after the output was frozen:
RPE 0.817 → 0.776 cm, MPPE 2.010 → 1.986 cm, ADD AUC 0.860 → 0.859, with
MP-SR/SP-SR unchanged at 1.0. These are reconstruction diagnostics, not learned
policy rollout or final challenge scores. GT did not enter optimization. Holdouts
0/11/23/31/39 were neither inspected nor tuned for this pilot.

The rejected trial 1 had 851 successful updates and 9 skips, with within-2-cm
contact consistency falling to 66.4%, worse object/hand/camera jitter and worse
public-reference diagnostics. Its complete outputs and receipts remain preserved.
Trial 2 was designed using development reconstruction/geometry findings, not
holdout tuning. It demonstrates a bounded pilot, not an all-episode recipe.

## Visual and software acceptance

All 343 source/render pairs were inspected in 29 contact sheets, together with
12 full-size onset/lift/release/end event pairs. Pot/lid lift, movement and
replacement remain present; no gross mesh collapse or teleport was observed.
Sparse background holes/speckle, absent forearms and missing upstream hand
observations remain visible. The result is not a photorealistic reconstruction.

The source/render MP4 played to its 17.15-second end without a media error.
Browser checks covered refined/initial/background/camera toggles, frames
0/171/342, orbit, zoom, reset and animation wrap-around. Console checks found no
warnings/errors during review. This review combines complete still-frame inspection
and playback/control verification; it does not claim a continuous visual audit
of every video frame at playback speed.

The requested downloadable 3D viewer loop contains all 343 actual browser frame
captures, assembled as a 1080p, 20 fps, 17.15-second H.264 MP4. Decoding confirms
343 frames. It has no audio and uses refined geometry with background off and
camera on. It is a frame-by-frame viewer capture, not a real-time screencast.

37 targeted CPU tests pass, including invalid input rejection, missing-contact
denominators, gap-aware/finite depth gradients, pose bounds, viewer face
deduplication and Docker CLI bound forwarding. Targeted Ruff checks pass. Native
optimizer/Docker entrypoint syntax and Bash launcher syntax were checked. Actual
Docker image execution was not tested; rebuild the image with this source before
using it. CPU tests alone do not prove GPU refinement; the real run provides that
separate evidence.

## Review handoff

Private laptop artifacts are under
`C:/Users/evanc/OneDrive/Desktop/Nvidia V2D/runs/issue7-validation/trial2`;
native full outputs are under `/var/tmp/v2d-issue7-e002/trial2`. The original
`trial2-artifacts.tar.gz` checkpoint, its manifest and its pre-display-fix viewer
are retained. The completion supplement adds current display HTML, video,
visual review and software receipts without replacing the frozen run.
Licensed MANO, footage and scene assets have not been published.

Start the laptop review server with
`./runs/issue7-validation/open_review.ps1 -Trial 2 -Port 8137` from the parent
workspace, or open the existing `http://127.0.0.1:8137/` server. See
[REPRODUCE.md](REPRODUCE.md) for commands and evidence checks. The repository
contains numeric documentation and software repairs; private assets stay local.

Suggested closure statement after review and merge: “Episode 2
demonstrates issue 7's one-episode refinement/viewer/report milestone, with all
343 frames covered and reported numerical/visual limitations.” Further numerical
stability, camera smoothing, background completion and upstream missing-hand
coverage are explicit follow-up work. The user authorized pushing this branch
and opening a draft PR. Issue closure remains tied to review and merge.
