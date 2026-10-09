# Reproducing and reviewing the issue-7 pilot

Read [RESULTS.md](RESULTS.md) and [RESULTS.json](RESULTS.json) first. The frozen
experiment used native WSL CUDA execution; the Docker CLI forwarding test does
not validate an existing image. Rebuild `v2d_gsplat_refinement` with the updated
module before choosing Docker execution. Use a new output directory for every
experiment and preserve previous runs.

## Existing laptop evidence

Parent workspace:
`C:/Users/evanc/OneDrive/Desktop/Nvidia V2D`.
Source worktree: `issue7-refinement`, branch
`Evc-r/issue7-refinement-validation`.

Private helper directory: `runs/issue7-validation`. `runtime.sh` selects the
isolated issue-7 package/toolchain/extension directories and the verified local
Python runtime. It reads the existing licensed MANO/CAD assets on this laptop;
it neither provisions a worker nor changes issue-6 runtime files.

`run_episode_slam.py` is the exact actual call, with every argument captured
in `trial4/run-receipt.json`; its frozen source copy is retained. It refuses an
existing `/var/tmp/v2d-issue7-e002/trial4`. Do not rerun over the accepted trial.

| Private artifact | Purpose |
|---|---|
| `trial4/run-receipt.json`, `frozen-source/` | Exact runtime, arguments and source hashes |
| `inputs-lock.json` / native `inputs-slam1-lock.json` | 4,673 original plus 2,961 SLAM-staged hashes |
| `trial4/progress/optimization_trace.jsonl` | Every update, skip, frame batch and loss |
| `trial4/verification.json` | Finite arrays, coverage, source preservation; historical numerical checkpoint |
| `trial4/pose-limits-verification.json` | Actual exported corrections and frozen shape/scale checks |
| `trial4/quality/report.json`, `report.md` | Full matched numerical measurements |
| `trial4/scene.npz`, `objects/`, `hands/`, `camera/` | Full scene and pose exports; pose directories also in native archive |
| `trial4/overlay.mp4`, `playback.mp4` | Actual Gaussian rendering and source/render comparison |
| `trial4/quality/viewer.html` | Current interactive mesh/point-cloud viewer |
| `trial4/review/`, `visual-review.json` | Complete contact sheets/events and final visual review receipt |
| `trial4/trial4-artifacts.tar.gz`, `backup-sha256.json` | Original portable native-output checkpoint, preserved |
| `trial4/completion-supplement.tar.gz`, `completion-manifest.json` | Current display/video/review/document supplement |
| `cpu-tests-quality-completion.xml`, `quality-completion-checks.json` | Final software check receipts |

`verification.json` deliberately retains its original “visual review pending”
checkpoint. `visual-review.json` and `quality-completed-checkpoint-20261009.json` supersede
that review status. Display fixes retain `quality/viewer-before-display-fix.html`
and a separate `viewer-display-receipt.json`; they do not alter the scene/metrics.

From the parent workspace in PowerShell, if the review server is not running:

```powershell
.\runs\issue7-validation\open_review.ps1 -Trial 4 -Port 8138
```

Open `http://127.0.0.1:8138/` for source/render playback, report and comparison download.
The server binds only to loopback. The viewer can require network access for its
Three.js module. Check frames 0/171/342, both geometry states, background/camera,
orbit/zoom/reset and looping. The saved MP4 is self-contained.

## Frozen final evidence

The current candidate is trial4, using initialized SLAM trial2 cameras. Original
odometry trials1/2/3 and rejected SLAM trial1 remain historical. The optimizer
source hashes exactly match the executed run; only the final report/viewer
display code changed later. Current viewer repairs retain the full pot/lid CAD
meshes (97,803/99,231 faces) and preserve original scene/metric hashes.
`trial4/slam-initialization-verification.json` checks all343 source poses and
unchanged camera-relative object/hand geometry. SLAM source/model/build pins,
trajectory, 343 poses and receipt are in `slam-trial2`, with its own archive.

## Local metric SLAM stage

`scripts/track3/slam_camera.py` runs the existing DROID integration on the same
rectified `ego_cam_c` RGB, intrinsics and calibrated stereo depth used by
refinement. It uses DROID's RGB-D sensor-prior path; native stereo mode hardcodes
a different baseline. The wrapper reads no public reference camera trajectory.
It requires an initialized frontend, a multi-view map, complete finite poses,
and unchanged source hashes before declaring success. Per-frame output is
explicitly **camera-to-world**, in metres, with the run evidence identifier.

The verified local runtime uses Python 3.11.17, PyTorch 2.9.1+cu128,
`torch_scatter` 2.1.2+pt29cu128, CUDA 12.8 and DROID commit
`2dfd39f0dcad44012ca7bbb8aa70b55edbfa9c99`, including its pinned submodules.
The upstream `lietorch` setup hardcodes old CUDA architectures: remove those
`-gencode` entries and set `TORCH_CUDA_ARCH_LIST` for the actual GPU before
building. This laptop used `12.0` and two compilation jobs. Compile/install
`thirdparty/lietorch`, the matching `torch_scatter` wheel and `droid_backends`
in an isolated environment. Add both the checkout and its `droid_slam/`
subdirectory to `PYTHONPATH`, plus the project's `v2d.common` and
`v2d.droid_slam` modules. The existing module's `download_weights.py` fetches
the official checkpoint; its verified SHA256 is
`46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526`.

```bash
python scripts/track3/slam_camera.py --inputs /path/to/staged-inputs \
  --weights /path/to/droid.pth --droid-repo /path/to/DROID-SLAM \
  --out /path/to/new-slam-run
```

Defaults use all frames at 240×384, valid stereo depth up to 3 m, motion
threshold 0.5, keyframe threshold 1 and warmup 8. Default upstream motion
threshold 2.4 selected only four keyframes in episode 2 and never initialized
the frontend; that attempt is preserved and rejected. The corrected run
initialized, retained eight keyframes, performed local BA and global passes of
7/12 iterations, and exported 343 finite poses. Peak GPU tensor allocation was
1.24 GB. Its eight keyframe median map-versus-stereo depth errors were
0.04–0.43 mm. This is agreement with the sensor prior, not independent physical
ground truth or a camera accuracy score. Exact private source, runtime, input,
model and output hashes are in `runs/issue7-validation/slam-trial4/`.

`gsplat_inputs.py --slam-camera-dir /path/to/new-slam-run/camera_c2w
--require-slam` stages these poses. Gap-filled objects are first converted from
their original world through their original camera; replacing the world gauge
therefore leaves all object/hand camera-relative geometry unchanged. Unit tests
exercise different world gauges and reject ambiguous camera directions.

## A new experiment with the repaired launcher

The shell defaults match the bounded pilot's optimizer configuration. Stage
independent cameras, rectified images/intrinsics, metric stereo, object poses/CAD
and metric-valid hand fits via the adapter. Keep provenance and hash the inputs
before running. Supply your own authorized licensed assets and a fresh `OUT`:

```bash
EPISODE=2 DATA=/path/to/public RESULTS=/path/to/reconstruction-results \
  BUNDLE=/path/to/episode2-bundle HANDS_RUN=/path/to/episode2-hand-run \
  MANO_ROOT=/path/to/mano OUT=/path/to/new-issue7-trial \
  SLAM_CAMERA_DIR=/path/to/new-slam-run/camera_c2w \
  REFINE_MODE=native REFINE_PYTHON=/path/to/cuda-python \
  PYTHON=/path/to/report-python REST_ANNOTATIONS=/path/to/frozen-rest.json \
  CONTACT_TRUTH=/path/to/independent-contact-reference.json \
  bash scripts/track3/refine_episode.sh
```

The launcher now requires declared independent SLAM inputs by default, including
when reusing cached inputs. `RUN_SLAM=1 DROID_WEIGHTS=... DROID_REPO=...` runs the
local stage first; `SLAM_PYTHON` selects its runtime. `REQUIRE_SLAM=0` explicitly
permits an odometry diagnostic. Camera acceleration weights default to 1e6;
all pose/shape bounds remain initialization-relative engineering priors.

The actual pilot staged inputs once, then called the native function directly
with a complete run receipt. The generic wrapper is an equivalent configuration
path, not a claim that this full wrapper or Docker container was GPU-tested.
Record exact versions, source/input hashes, full arguments and seed for any
reproduction. Exported contact labels remain fixed; reviewed rest intervals must
be chosen from source evidence before results. Run GT diagnostics only after
freezing outputs. Do not tune the reserved holdouts to reproduce this pilot.

Accept only finite exports with at least one successful visit per frame. Inspect
the whole source/render sequence and viewer; record skips, missing/inferred
coverage and regressions. A low training loss or low jitter alone is insufficient.

## Software checks

From the source worktree on this laptop:

```powershell
..\.venv\Scripts\python.exe -m pytest scripts/track3/tests/test_gsplat_inputs.py scripts/track3/tests/test_refinement_report.py scripts/track3/tests/test_refinement_losses.py scripts/track3/tests/test_refinement_constraints.py scripts/track3/tests/test_refinement_docker.py scripts/track3/tests/test_contact_validation.py scripts/track3/tests/test_refinement_depth.py scripts/track3/tests/test_refinement_render_features.py -q
```

The 65-test suite includes actual CLI parsing/forwarding with a stub container;
it does not launch Docker. Targeted lint excludes inherited unrelated style
warnings in existing optimizer modules. Those modules receive syntax checks and
the loss/constraint behavior tests, together with separate actual GPU evidence.

## Independent contact reference

Pass `CONTACT_TRUTH` to the launcher or `--contact-truth` to the report. The
schema uses `schema_version: 1`, `episode`, `granularity` (`hand_object` or
`fingertip`), `independent_of_reconstruction: true`, a documented `source`, and
`evidence.source_sha256`. Each label contains `frame`, `hand`, `object_id`, and
`contact: true/false/null`; fingertip
granularity also requires `fingertip`. Null means unknown. Record source review
before looking at refinement metrics; never derive truth from predicted contacts.
The private episode2 reference is `source-hand-object-contact-reference.json`,
SHA256 `af2836ab4efac8666904a76a0cbd82ad2e697000314f8910b85139c6faf129b3`.
It uses the previously reviewed source grasp cores/separated intervals, with
ambiguous boundaries unknown. Keep this validation file out of optimization.
The report writes conditional precision, positive-prediction label coverage,
recall and missing geometry separately. Coarse hand-object labels cannot establish
per-fingertip or metric3D contact truth.
