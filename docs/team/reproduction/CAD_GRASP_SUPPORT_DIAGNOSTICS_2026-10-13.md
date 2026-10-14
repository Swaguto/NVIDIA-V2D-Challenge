# e002 source accuracy and estimated support diagnostics

This extends the [published original CAD replay](ORIGINAL_CAD_DEX3_REPLAY_2026-10-13.md).
The published replay and its immutable release remain unchanged. The new
estimated-plane candidate is a development proposal, not an accepted physical
training target. All human observations, object poses and observed/display-held
provenance remain unchanged. Issue #9 and training remain deferred.

## What was measured

The episode contains 343 source frames. The existing replay contains 3,737
interpolation knots and 377 inferred closed hand states in four grasp phases.
The new work reuses the GitHub CAD query, FK, continuous collision check,
transition search, joint repair and source rasterizer. It reads no manual frame
ranges or crops when fitting the new proposal. Original source geometry and
images are measurements or estimates with their recorded provenance; inferred
joint anatomy, contact intentions and robot poses are not ground truth.

### Object placement

A bounded object calibration was fitted on even source frames and tested on odd
frames. It reused the repository's rigid depth-alignment solver and exact
original CAD. Maximum search bounds were 10 mm translation and 0.1 rad rotation.
The candidate was rejected for both objects under the predeclared image and
support guards. No object pose was changed.

| Held-out measure | Pot, original → proposed | Lid, original → proposed |
| --- | --- | --- |
| Median frame depth error | 12.856 → 12.722 mm | 6.497 → 6.510 mm |
| Visible mask precision | 0.82986 → 0.81518 | 0.92756 → 0.93265 |
| Visible mask recall | 0.96058 → 0.95026 | 0.97550 → 0.96165 |
| Median frame depth-error p90 | 37.235 → 39.261 mm | 23.884 → 22.945 mm |

The depth metric compares qualified stereo depth with rendered original CAD.
Estimated masks, occlusion, depth noise and CAD/pose disagreement affect it; it
is not a measured ground-truth pose error. The signed median depth discrepancy
is only about +1.59 mm for the pot and -0.54 mm for the lid. Applying a uniform
12 mm shift would misinterpret the absolute-error statistic.

### Hand contact accuracy

Actual selected Dex3 surface points were compared with unchanged, finger-labelled
stereo points, using the existing 15 mm nearest-bone and 2 mm other-digit margin.
The 32-point/10-frame support guard remained unchanged. A separate sensitivity
test relabelled the same stereo points with the reviewed retained-scale hand
joint proposals; 2,303 inferred labels changed, without adding observations.

The apparent worst second-left-thumb discrepancy is 58.38 mm. Direct projection
shows why nearest *visible* thumb points are insufficient to measure the contact
point: in frames 319, 326 and 331, the few labelled points lie near the thumb
base, while the inferred pad projects near the visible tip. At frame 326, the
pad is 9.35 mm behind measured skin depth; frame 331 is 11.03 mm behind it, and
frame 319 has an unqualified depth neighborhood. This explains a sparse-point
failure, but does not prove accurate hidden contact or upgrade a robot target.

All closed pad points were independently projected into locked source depth and
hand masks. Occluded, unavailable and unqualified pixels remained unmeasured.
Four first-left-thumb states project ahead of measured source depth by more
than the unchanged 10 mm diagnostic tolerance: frames 48, 49, 151 and 154.
Their residuals are -15.61, -19.33, -14.07 and -10.15 mm. Frames 48/49 are
outside the own-hand mask. These exceptions remain explicit accuracy/timing
concerns; neither a good median nor a visible mesh clears them.

### Estimated support and new grasp proposal

The frozen source plane is an inferred hypothesis with unresolved semantic
identity and uncertainty. Its old convex footprint fills unobserved gaps.
A second diagnostic uses only 10 mm cells supported by at least three distinct
source frames, with a 5 mm plane neighborhood. Gaps remain unknown. This finite
vertex diagnostic is not a continuous table or full-surface certificate.

In the published replay, the worst closed-state plane distances are -7.06 mm
left and -0.98 mm right. A bounded fit adds a 0.5 mm estimated-plane margin to
every original hand vertex at all 377 closed source states. Original CAD gap,
opposing normal, self-clearance, material and hardware gates stay unchanged.
The four fits pass the original 8 mm translation, 0.2 rad rotation and 0.25 rad
finger-change budgets relative to their recorded parent templates.

The second right fit required a separate tighter 7.99999 mm search radius.
Its solver exceeded that *local search* radius by a fraction of a nanometre;
that failure is retained in its producer report. Independent assembly passes
the unchanged original 8 mm acceptance budget. It does not claim that the
tighter local search report passed, and it changes no fit parameters.

The first replay of these fits failed the between-frame CAD check during the
first left approach. Automatic transition search then tested fixed envelope
fractions and repaired the left release with the existing joint solver. All
eight finite approach/release windows passed, preserving all new closed anchors.
The resulting candidate passes full finger position and speed checks; maximum
speed/limit ratios are 0.961017 left and 0.999698 right. The exact source arrays
and source observation flags are preserved.

Both fresh continuous calculations completed successfully for this proposal:
4,373 original-CAD proof intervals and 7,444 material/self/cross-hand intervals
cover all 3,736 trajectory segments. Independent verification checked exact
interval boundaries, input bindings, all link/object records, hardware limits,
fresh FK and preservation of the 377 closed source anchors. These calculations
do not clear the separate free-hand/support or source-accuracy concerns.

A fresh reconstruction from the saved fitted inputs reproduced both stages'
motion NPZ, knot NPZ and report bytes exactly in the declared Linux geometry
runtime. The final stage compares all 27 motion arrays and four knot arrays.
Optimizer searches and raw-video inference were not rerun by this reproduction.

The new grasp proposal reduces source states with a below-2 mm vertex in the
reliably observed support-cell union from 107 to 43 for the left hand; the right
count remains 92. Remaining violations occur outside closed grasps. The worst
free-hand witness is still about 58.5 mm below the plane within that inferred
footprint on the right. This below-plane statistic alone does not prove a table
intersection: a hand can be entirely below or beside a finite tabletop.

A subsequent original-triangle test distinguishes that case from actual
crossing of an estimated finite support sheet. The sheet uses the convex hull
of reliable-cell corners; it is conservative about gaps and does not model
table thickness. It found crossings in 32 source hand states and 959 knot hand
states, all outside closed grasps. Some crossings occur in observed support
cells and some in unknown gaps of the convex hull. The free-hand/support
question therefore remains unresolved even though the grasp states pass.
These are intersections with an estimated model, not confirmed real-table
collisions; no source or robot pose was changed by this diagnostic.

Simply replacing neutral free fingers with saved GitHub retarget finger values
was tested and refused: only 17 of 155 eligible source states passed finite
self/CAD checks. At frame 179, the candidate introduced an index/thumb
self-intersection and worsened the infinite-plane minimum. These values were
not adopted into the motion or any published release.

## Free wrist and support follow-up

A rigid root search around the generated pose found safe static proposals for
21 of the 32 source states that crossed the inferred finite sheet. These are
unadopted static proposals: speed, cross-hand motion and continuous clearance
were not checked. Increasing visibility does not make them source observations.

The generated left release wrist at source frames 232–234 was 63.3, 72.1 and
71.0 mm from the estimated human wrist. Matching the saved source-retarget root
reduced those distances to 3.0, 2.3 and 2.0 mm, but caused original-CAD surface
intersections. Across all 189 eligible observed free states, resetting roots
while keeping the generated fingers introduced ten CAD failures and left 22
finite-sheet crossings. This shortcut was rejected.

Wrist distance alone is not a hand-accuracy certificate: the Dex3 and human
hands have different geometry, and their grasp wrists may need different
placements. The follow-up therefore fits free robot wrist and finger posture
together. Its existing 80 mm source-wrist translation guard is a morphology
search bound, not a measured tracking-error tolerance. This differs from the
earlier 8 mm search around an already generated robot root. Every successful
static proposal still requires a trajectory and source-image review before use.

The measured source-orientation search V2 passed 43 of 51 automatically
selected static states. A separate V3 fit passed 47, but changed the orientation
reference to the generated robot. Some new V3 poses remained 48–66 degrees from
the saved source-retarget orientation. Those geometric passes are not accepted
as accurate recovered motion. The stricter source-orientation fallback remains
a separate experiment, and no free-motion proposal is adopted by this document.

The automatically selected 46 full source images compare actual projected robot
link outlines with estimated source joints. They are CPU geometry diagnostics;
they do not render occlusion and are not Isaac output. For example, moving a
wrist toward its source estimate can leave a different finger posture. Geometry
and wrist-distance improvements must therefore be reviewed together.

## Frozen local evidence

The Linux evidence root for this measured development run is
`/var/tmp/v2d-source-contact-20261011`. Preserve these directories. Outputs with
the same name must never be replaced by a later trial.

| Directory | Purpose |
| --- | --- |
| `object-calibration-image-validation-v2` | Paired full-image object test and rejected guards |
| `CAD-target-full-chain-stereo-diagnostic-v1` | Original inferred-digit diagnostic |
| `retained-prior-stereo-finger-diagnostic-v1` | Alternative inferred joint-prior sensitivity |
| `plane-constrained-pad-retained-prior-stereo-v3` | Actual old/new pad comparison with retained labels |
| `plane-constrained-pad-source-depth-v1` | All closed-pad source depth classifications and automatically selected images |
| `plane-constrained-CAD-grasp-fit-v1` / `v2` | Separate measured fits, including budget refusals |
| `plane-constrained-CAD-grasp-candidates-v3` | Independent original-budget assembly |
| `plane-constrained-CAD-transition-search-v1` | Eight automatic approach/release trials and joint repair |
| `plane-constrained-CAD-motion-v2` | New inferred source poses and interpolation knots |
| `plane-constrained-CAD-case-v2` | Locked new case; historical scene/proof files inside are not new evidence |
| `plane-constrained-CAD-source-FK-scene-v2` | Fresh original-asset FK/self/cross/material source-state audit |
| `plane-constrained-CAD-original-continuous-v2` | Fresh original-CAD surface proof |
| `plane-constrained-CAD-material-continuous-v2` | Fresh material/self/cross proof; require terminal report |
| `plane-constrained-CAD-reliable-support-cells-v2` | Fresh finite support diagnostic |
| `source-retarget-free-finger-diagnostic-v1` | Rejected shortcut, not an adopted motion |
| `plane-CAD-finite-support-crossings-v1` | Actual original-triangle crossings of an estimated finite support sheet; below-plane vertices alone are not collisions |
| `per-link-free-support-escape-diagnostic-v2` | 21/32 unadopted bounded rigid static proposals |
| `source-free-wrist-reset-diagnostic-v1` | Rejected root-only source matching; ten CAD failures |
| `source-aware-free-static-search-v2` | 43/51 unadopted source-orientation static passes |
| `source-aware-free-static-search-v3` | Orientation-reference sensitivity; new poses with large source-orientation differences not adopted |
| `source-aware-free-source-review-v2` | 46 automatic full-image CPU projection comparisons; no rendered occlusion |
| `plane-constrained-CAD-fitted-reproduction-v1` | Byte-exact saved-fit reconstruction and comparison receipt |
| `plane-constrained-CAD-independent-geometry-verification-v1.json` | Exact independent coverage and binding verification |
| `plane-constrained-CAD-simulator-input-package-v1` | Frozen new simulator case with fresh FK and proof inputs; no new simulator run claimed |

The new case manifest is
`679fbb264ec516853b69e9abb59272f4e9233888757550d0b67772e5f47e89ed`.
The motion payload is
`4b4d6fdc6c51315ce3e45b9754cb6fe35b2eb82a79ed33922303c203e45578a7`;
the knot payload is
`2946e162568f6e68c733a7ca957e28472d872757f0e7196f1d0f1a3588488085`.

## Reproduction and acceptance boundary

Use the existing published guide for the independently reproduced, actual
Isaac replay. It provides the pinned code revision, private release assets,
hash-checked extraction, geometry runtime, saved-fit reconstruction, simulator
runtime and comparison tolerances. That baseline can be reproduced by anyone
with the required repository and asset access.

The new development diagnostics above are not yet a portable published kit.
Their frozen local inputs, scripts, reports and failures must be packaged and
verified before a teammate is asked to reproduce them. A new simulator run must
bind to fresh FK and continuous reports for the new motion. The old successful
video cannot stand in for that run. Do not overwrite the immutable baseline.

The newly frozen simulator-input archive has SHA-256
`9c37184f8b14a1dba7e2c7cf9bc274609ef6035164a21efb65aa98d1f9a241de`
and contains 1,661 locked files. Its distinct case manifest has SHA-256
`3cad5bdcbeff98570d8d46bc3c209d05b3292da56713c246f59d050331220ffd`.
Neither those hashes nor packaging an input archive is evidence that the
new candidate has run in Isaac or passed source/physical acceptance.

Source observation coverage, display continuity, CAD/model clearance, source
accuracy and task acceptance are separate conclusions. The support-plane fit
and pad-depth diagnostics do not certify true solid geometry, the real table,
hidden fingertips, contact force or training targets. Raw-video inference and
untouched/private-episode generalization have not been rerun here.
