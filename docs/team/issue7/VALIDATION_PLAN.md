# Issue 7: one-episode refinement validation

Reviewed October 8, 2026. Baseline: Thanosan's commit
`2e8efa668c9485fbd99f961ff88284a905f2b021` on
`feat/track7-gaussian-splat-refinement`. Issue 7 remains open.

The live issue requires a Gaussian scene anchored to hand/object meshes and
background, independently initialized cameras, joint photometric/stereo-depth/
segmentation/temporal optimization, and a per-episode quality report. Its explicit
completion milestone is **one episode fully refined, viewable in a 3D viewer,
with written jitter/drift measurements**. Twenty accepted episodes are not an
issue-7 closure requirement. Issue 6's separate quality repair remains independent.

## Findings and repairs

The commit supplies the input adapter, optimizer, report and viewer, but does not
contain a completed real-episode refinement receipt. Its two adapter/report test
files originally produced 13 passes and one Windows Unicode-encoding failure.
Existing archived review probes also expose unsupported hand fits and an inflated
contact denominator.

This validation branch fixes explicit UTF-8 report writing, defaults to
metric-valid hand observations, rejects missing independent cameras and declared
GT-assisted initialization, and rejects nonfinite/degenerate transforms. Temporal
losses do not join hand observations across missing frames. Missing contact
geometry remains in the denominator and is reported separately. Rest candidates
require observation coverage and are described as a proxy, not established rest.
Contact runs are fixed before refinement; the report describes geometric
consistency and leaves independent contact precision unavailable.

The completed evidence is in [RESULTS.md](RESULTS.md), [RESULTS.json](RESULTS.json)
and [REPRODUCE.md](REPRODUCE.md). After the user resumed, the isolated toolchain
aliases and spaced-prefix linker configuration were repaired; an actual CUDA
render/backward check passed. The bounded second real trial completed all 343
frames with 840 successful updates and 20 guarded skips, finite exports, full
visual/control review and matched numerical reports. The unconstrained first
trial was rejected and preserved. 37 targeted CPU tests and targeted lint pass;
these are separate from the real GPU evidence and do not prove robot training.
The optimizer now records successful/skipped updates and loss components, rejects
zero-update runs and nonfinite scene exports, and retains the optimized Gaussian
state alongside geometry. Source review was extended to all 343 images. A lid
stationary interval (112–228) is frozen before observing results; the report
can compare that exact interval. The initial short 120–131 annotation is preserved.

Issues 4, 5 and 6 are still open. Their selected episode-2 outputs permit a
development refinement trial; this does not certify the upstream issues or their
all-episode acceptance criteria. Issue 7 can demonstrate its one-episode milestone
independently while the separate upstream repairs continue.

## First controlled experiment

Use development episode 2, all 343 frames. Keep holdouts 0/11/23/31/39 untouched.
Inputs include rectified ego_cam_c images, completed native calibrated stereo
depth, independent stereo-PnP cameras, pot/lid CAD and observed/inferred poses,
and the verified frozen-mask-repair hand outputs. Current staged counts:

| Input | Count |
|---|---:|
| Camera initializations | 343 |
| Pot observed / inferred | 285 / 58 |
| Lid observed | 343 |
| Metric-valid left hand | 272 |
| Metric-valid right hand | 284 |

Inferred object initialization stays inferred. Missing hands stay unknown.
Licensed MANO and all private footage remain on the same laptop. Freeze hand
shape, hand scale and CAD scale at initialization. Jointly optimize root/finger
poses, independent cameras and Gaussian appearance with the requested four loss
families. Do not read public pose GT during optimization.

## Closure evidence

1. Save exact source/input hashes, runtime versions, command/configuration, seed,
   full frame coverage, completion receipt and finite outputs for the actual run.
2. Produce initial/refined scene, object/hand/camera outputs, complete overlay,
   3D viewer and written jitter/drift numbers. Record retained unknown/inferred
   coverage. Validate the viewer in a browser and inspect complete source/output
   playback plus onset, lift and release events.
3. Use identical before/after frames and fixed contact runs. Independently
   reviewed stationary intervals support drift; inferred rest candidates alone
   do not. Grasp timing labels are not independent per-fingertip 3D contact truth.
4. Run public-reference reconstruction diagnostics only after outputs are frozen.
   Lower jitter alone does not establish accuracy; record any geometry, contact,
   motion or diagnostic regression. These are not policy-rollout scores.
5. Prepare a reviewable closure handoff. Keep issue 7 open until the real run,
   viewer and report meet its milestone. Commit/push/PR, merge, issue changes and
   external messages require the user's applicable authorization.

Local experiment scripts and private evidence are in
`C:/Users/evanc/OneDrive/Desktop/Nvidia V2D/runs/issue7-validation` and
`/var/tmp/v2d-issue7-e002`. Source checkout:
`C:/Users/evanc/OneDrive/Desktop/Nvidia V2D/issue7-refinement`.

## Historical paused checkpoint — October 8, 2026

The user initially requested a pause at the next nearest major checkpoint.
The saved checkpoint stopped after input/code checks and a CUDA preflight
failure. The user subsequently resumed with “go ahead and finish up issue 7”.
The completed run/viewer/report evidence above supersedes that paused status.
Issue 7 remains open for review; the user authorized push and draft-PR publication
on October 8, 2026. The independent
issue-6 worker and its budget controls were not changed.

Private receipt: `runs/issue7-validation/pause-checkpoint-20261008.json` in the
parent workspace; `inputs-lock.json`, source hashes, CPU receipt and build logs
are saved beside it. The worktree remains on
`Evc-r/issue7-refinement-validation` with uncommitted repairs.

The historical resume sequence was completed; retain its original failure logs:

1. Diagnose and repair the isolated WSL toolchain's C++ linker/library path
   references. The latest failure contains the truncated path
   `V2D/runs/pr32-validation/local-runtime/env/lib`; references to the spaced
   original prefix need verification. Verify `toolchain_probe.cpp` links and executes.
2. Source `runs/issue7-validation/runtime.sh`, run incremental `ninja -v -j 2`
   inside `/var/tmp/v2d-issue7-runtime/extensions/gsplat_cuda`, then run
   `gpu_preflight.py`. **Do not import gsplat's backend before a shared library
   exists:** its fresh-build path deletes the directory and the retained objects.
3. The frozen first real trial was run once and rejected for quality regressions.
   Its complete outputs remain preserved. A separately frozen bounded trial 2
   supplies the accepted pilot; neither trial was overwritten.
4. Generate the report with the locked inputs, fixed source rest interval and
   contact labels; inspect source/output playback and the browser 3D viewer.
   Continue with the closure-evidence checklist above.

The frozen trial uses nominal dataset 20 fps; the source video's container average
is about 20.254 fps. Timing-derived measurements must state this convention.
