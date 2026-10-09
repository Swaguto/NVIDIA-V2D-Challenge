# Issue 7: completion assessment

Reviewed October 9, 2026 against the live issue. The goal is to satisfy the
one-episode Gaussian refinement, SLAM camera, quality-report and 3D-viewer
milestone on draft PR #34, without pushing to main.

All stated requirements now have actual episode-2 evidence in
[RESULTS.md](RESULTS.md) and [RESULTS.json](RESULTS.json):

| Requirement | Final trial-4 evidence |
|---|---|
| Anchored hand/object/background Gaussian scene; SLAM camera | Initialized metric DROID frontend, local/global BA, 343 finite camera poses and retained Gaussian state |
| Four objective families, jointly clean object/hand/camera | Photometric, calibrated stereo depth, segmentation and temporal losses; 860/860 updates, zero skips, 10 visits per frame |
| Jitter, drift, contact precision | Matched before/after report, frozen reviewed stationary interval, independent hand-object precision with label/geometry coverage |
| Full episode, 3D viewer and written numbers | All343 frames, complete source/render still review, 12 events, browser controls/playback, numeric handoff |

The audit found three material gaps in the historical pilot: odometry was not
SLAM, uninitialized feature padding produced nonfinite backward gradients, and
geometric consistency was not independent contact precision. The final change
implements and runs local metric SLAM, zero-pads unsupported feature channels,
preserves inverse-depth unknowns, adds gap-aware camera acceleration and scores
independent labels at their declared granularity. Actual GPU and 65-test CPU
evidence are preserved separately.

Source-reviewed labels remain coarse hand-object truth. Unknown boundaries and
missing geometry stay explicit; the 100% conditional precision has 94.83%
positive-prediction label coverage. Per-fingertip/metric3D truth is unavailable.
Small nonuniform drift regressions, sparse background and missing upstream hands
are reported, rather than hidden by aggregate jitter improvements.

The explicit completion milestone is one development episode, not twenty.
Episode2 uses 343 SLAM poses, pot285 observed/58 inferred frames, lid343 observed,
left272 and right284 metric-valid hand frames. Inferred provenance remains
inferred and missing hands remain unknown. CAD/hand scale and shared hand shape
are frozen. No public reference trajectory enters SLAM or refinement. Holdouts
0/11/23/31/39 remain untouched by this pilot.

All trials, failures, source/input hashes and full outputs are retained. Current
optimizer source hashes match the executed run. Viewer-only display changes
have a separate receipt and do not change scene/metrics. Reproduction commands
and limitations are in [REPRODUCE.md](REPRODUCE.md).

The final course of action is feature-branch publication, draft-PR review and
merge; issue7 can close with this PR. User authorization covers branch push and
draft publication, not merge or a direct issue-state change. Upstream issues4–6
and all20 quality acceptance remain separate. The issue6 worker/budget was not
changed. Historical pause/run checkpoints are preserved in the private evidence
directory and are superseded by the current completion assessment.
