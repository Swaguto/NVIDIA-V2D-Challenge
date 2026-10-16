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

## Reacquisition and weakening-contact repair

Later in the inspected failed replay, unbounded measured-object feedback pulls
the hand reference toward the fallen pot: at source frame 273, the recorded
hand target z is -1024.92 m and the hand itself is already at -174.35 m. This
explains its later disappearance from the camera, rather than missing rendered
frames.

The new bounded-reacquisition option fades settling feedback between 3 and 6 cm
(or 10 and 20 degrees). Beyond the outer bounds it latches a failed phase,
opens the fingers and retreats to the source-relative parked reference over
0.25 source seconds. A prediction query cannot mutate this latch. The video
labels aborted reacquisition; original object errors and source frames remain
recorded. This prevents a lost object from commanding an out-of-scene hand; it
does not turn a failed grasp into success.

The recovery comparison also separates measured-reaction compensation from the
strict three-pad diagnostic. `measured_primary_reaction` uses only actual normal
and friction reactions against the intended object, and can remain active while
some intended fingers weaken. The all-declared-digit 0.5 N retention gate is
still recorded independently. The same 30 ms dropout hold, 20 ms filter, 10 N /
2 Nm compensation bounds and 60 N / 60 Nm total wrist caps are retained.

`minimum_live_patch_force_N=0.05` permits reconstruction of a current contact
patch at lower pressure so that the finger allocator can try to recover it.
The prior 0.5 N threshold was also shutting off recovery at a weak pad. This
changes geometric eligibility only; it does not lower the 2 N planned minimum
or the 0.5 N measured support diagnostic. Missing, invalid or infeasible contacts
still fall back explicitly. Every new option is bound in the private input and
requires a fresh GPU check. 77 CPU tests pass.

## Force/posture isolation and integration check

A further comparison projects posture spring torque into the nullspace of the
current measured normal-contact Jacobians. This preserves finger configuration
in unconstrained directions while releasing the position spring's opposition to
normal loading. It retains joint damping, original joint limits and real effort
caps. Rank-deficient or out-of-limit targets reject the allocation. Position and
effort commands are committed together only after all candidate calculations
succeed, so fallback restores both frozen commands. The older pure-torque and
full frozen-posture modes remain reproducible. This hybrid mode is experimental.

The container runner also accepts an explicit supported-task `--physics-dt` in
[0.0005, 0.01]. The full-stage gate must match that timestep exactly. This permits
a 400 Hz integration check against the previous 100 Hz contact simulation while
preserving the source clock and video frames. No motor, material, mesh or source
scale is increased. Static fixtures retain their original 0.01 s timestep.

78 CPU tests pass, including hybrid posture torque orthogonality, retained
configuration, rank rejection and joint-bound enforcement. Fresh simulator
results, including failed variants, determine whether either change is useful.

The bounded-reacquisition simulator comparison confirms that a pot loss no
longer pulls the blue hand out of the scene: its actual root z stays between
-0.3494 and -0.2173 m while the failed object continues falling. The failed
phase is recorded. The final fault response additionally holds the measured
wrist pose and opens from the measured finger configuration, instead of
continuing a source-relative park path that could cross the displaced object.
This affects robot targets only. 79 CPU tests pass; the final fault hold still
requires its new simulator comparison.
