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

`run_episode_bounded.py` is the exact actual call, with every argument captured
in `trial2/run-receipt.json`; its frozen source copy is retained. It refuses an
existing `/var/tmp/v2d-issue7-e002/trial2`. Do not rerun over the accepted trial.

| Private artifact | Purpose |
|---|---|
| `trial2/run-receipt.json`, `frozen-source/` | Exact runtime, arguments and source hashes |
| `inputs-lock.json` | 4,673 frozen input/source hashes |
| `trial2/progress/optimization_trace.jsonl` | Every update, skip, frame batch and loss |
| `trial2/verification.json` | Finite arrays, coverage, source preservation; historical numerical checkpoint |
| `trial2/pose-limits-verification.json` | Actual exported corrections and frozen shape/scale checks |
| `trial2/quality/report.json`, `report.md` | Full matched numerical measurements |
| `trial2/scene.npz`, `objects/`, `hands/`, `camera/` | Full scene and pose exports; pose directories also in native archive |
| `trial2/overlay.mp4`, `playback.mp4` | Actual Gaussian rendering and source/render comparison |
| `trial2/quality/viewer.html` | Current interactive mesh/point-cloud viewer |
| `trial2/review/`, `visual-review.json` | Complete contact sheets/events and final visual review receipt |
| `trial2/episode002-refined-viewer-loop.mp4` | Requested complete viewer capture |
| `trial2/viewer-recording-receipt.json` | Frame/video hashes and decoded media verification |
| `trial2/trial2-artifacts.tar.gz`, `backup-sha256.json` | Original portable native-output checkpoint, preserved |
| `trial2/completion-supplement.tar.gz`, `completion-manifest.json` | Current display/video/review/document supplement |
| `cpu-tests-completion.xml`, `completion-checks.json` | Final software check receipts |

`verification.json` deliberately retains its original “visual review pending”
checkpoint. `visual-review.json` and `completed-checkpoint-20261008.json` supersede
that review status. Display fixes retain `quality/viewer-before-display-fix.html`
and a separate `viewer-display-receipt.json`; they do not alter the scene/metrics.

From the parent workspace in PowerShell, if the review server is not running:

```powershell
.\runs\issue7-validation\open_review.ps1 -Trial 2 -Port 8137
```

Open `http://127.0.0.1:8137/` for source/render playback, report and download.
The server binds only to loopback. The viewer can require network access for its
Three.js module. Check frames 0/171/342, both geometry states, background/camera,
orbit/zoom/reset and looping. The saved MP4 is self-contained.

## A new experiment with the repaired launcher

The shell defaults match the bounded pilot's optimizer configuration. Stage
independent cameras, rectified images/intrinsics, metric stereo, object poses/CAD
and metric-valid hand fits via the adapter. Keep provenance and hash the inputs
before running. Supply your own authorized licensed assets and a fresh `OUT`:

```bash
EPISODE=2 DATA=/path/to/public RESULTS=/path/to/reconstruction-results \
  BUNDLE=/path/to/episode2-bundle HANDS_RUN=/path/to/episode2-hand-run \
  MANO_ROOT=/path/to/mano OUT=/path/to/new-issue7-trial \
  REFINE_MODE=native REFINE_PYTHON=/path/to/cuda-python \
  PYTHON=/path/to/report-python REST_ANNOTATIONS=/path/to/frozen-rest.json \
  bash scripts/track3/refine_episode.sh
```

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
..\.venv\Scripts\python.exe -m pytest scripts/track3/tests/test_gsplat_inputs.py scripts/track3/tests/test_refinement_report.py scripts/track3/tests/test_refinement_losses.py scripts/track3/tests/test_refinement_constraints.py scripts/track3/tests/test_refinement_docker.py -q
```

The 37-test suite includes actual CLI parsing/forwarding with a stub container;
it does not launch Docker. Targeted lint excludes inherited unrelated style
warnings in existing optimizer modules. Those modules receive syntax checks and
the loss/constraint behavior tests, together with separate actual GPU evidence.
