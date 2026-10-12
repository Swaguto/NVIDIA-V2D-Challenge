# Separate reproduction-branch review — October 12, 2026

This review uses `recent-reproducibility-code` at
`4b3dfef26b8493139205253c3909beb3d825187c`. Dex3 PR #35 remains a separate
draft based on `main`; its parent has merged. Neither PR has been merged or
human-approved by this review.

Two correctness problems affected the newer recovery path. SAM masks could
extend over a forearm/background beyond the independent current-frame detector
box, contaminating image/depth checks. More seriously, the native pose writer
exported a unit hand scale after checking geometry at a different fitted scale.
A reproduced native pose could therefore differ from the pose that passed.

The change bounds the segmentation mask to the detector box, keeps the entire
rendered hand in the geometry precision checks, and preserves the fitted native
scale. The retained-input adapter and runner verify source/model hashes and
independently rematerialize every native export. Display holds and false
accuracy flags are refused. CPU test dependencies and the existing Windows/
Linux workflow now include the necessary pinned OpenCV dependency and controls.

All 175 issue6 tests pass in the native inference environment. Thirty-two
recovery/export controls pass locally on Windows and Linux with different NumPy
versions; fourteen reproduction controls pass on Windows. Lint/format checks
pass. The integrated 343-frame e002 run agrees exactly with the measured
experimental method's geometry arrays and mask PNGs. See
[the measured method](RETAINED_HAND_QUALITY.md) for commands and limits.

Remaining review gates: remote CI completion, human review, untouched public
episodes under a fully frozen upstream method, and independent full inference/
fitting reproduction. This patch is a development consistency checker, not a
claim of accurate physical contact or a production recovery release. In
particular, a valid-looking mesh may still have an inaccurate hidden fingertip.

Observed source hands, display-held source poses and inferred robot motion must
retain separate flags. The physical Dex3 trajectory work and updated Isaac
replay are separate evidence, not implied by these tests. Issue #9 and training
remain deferred under the active goal.
