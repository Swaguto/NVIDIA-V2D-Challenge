# Visual inspection and continuous robot-reference candidate

The reviewed full e002 video is `coupled-task-live-v3`, report SHA-256
`99b1e2af831f8886f398da0029de07421bb7ca96881566aa4e5fb7466a79bdab`.
Frame samples show the pot separate from the blue hand near source frames
122–124 (6.1–6.2 s), leave the finite support sheet, then disappear below it.
The blue hand continues the reference and later closes in empty space. The lid
is released separately. No frame is removed to hide these outcomes.

The frozen object evaluator uses linear translation and separate SLERP segments.
Its velocity can jump at source knots. The inferred hand reference inherits
those jumps during carrying, while the old force planner balances only gravity.
These are concrete reference/controller limitations; they do not establish one
exclusive cause for the observed loss.

## Proposed changes, disabled by default

`task_reference_interpolation=cubic` constructs an inferred robot object path
with a clamped-endpoint cubic position spline and a rotation spline. Every
recorded pose knot and timestamp is preserved. Source arrays, source masks,
meshes, object state and nominal friction/motor limits are unchanged. Only robot
commands between observations change. Angular velocity/acceleration are continuous
inside the source interval; endpoint settling holds are explicitly separate.
The nominal scene-height corrections are included before constructing the spline.

The e002 dense interpolation audit (6,900 times per object) finds maximum
position changes of 1.213/1.053 mm and rotation changes of 0.375/0.519 degrees
relative to the earlier piecewise path, below the fixed 6 mm / 3 degree bounds.
This finite audit does not certify collision clearance between samples.

`live_contact_load_model=reference_acceleration` includes nominal payload
linear acceleration, angular acceleration and gyroscopic moment in the contact
allocation. Physical acceleration scales with source speed squared. Body-axis
angular derivatives and nominal inertia are converted to world axes correctly.
The inferred load is supplied to both planned fallback torque and live contact
allocation. Real measured contacts, original force and motor limits, and the
frozen posture target remain required. No object wrench, attachment or pose
playback is introduced. The existing gravity mode remains available.

The complete nominal pot load plan is feasible at every source contact row,
with maximum planned feedforward 0.54310 Nm under the existing 0.55 Nm bound.
This is an analytic check, not simulated retention. Actual live allocation may
still be infeasible or lose contact; activation/fallback and requested wrench
are recorded.

74 CPU tests cover continuity, exact knot preservation, immutable inputs, angular
frame conversion, physical speed scaling, Newton/Euler load balance, contact
normal signs and original effort caps. A fresh free-space gate and full physics
comparison are required. No policy or physical task is accepted yet.

## Comparison protocol

1. Continuous reference with the frozen 2 N/gravity feedforward baseline.
2. The same continuous reference with planned/live acceleration-aware allocation
   and the frozen posture target.

Both run at the existing half source speed, on the same source data and simulator
image. Use a fresh output root per variant. Back up and verify results, then stop
Brev. Inputs remain local under the user's existing publishing choice.

Implementation references: [SciPy CubicSpline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html)
and [SciPy RotationSpline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RotationSpline.html).
