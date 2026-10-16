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

## Torque telemetry correction

The legacy `actual_combined_finger_effort_Nm` name overstates its measurement
scope. Its values come from `hand.data.applied_torque`, which the pinned
ImplicitActuator computes as a clipped model estimate. New runs use
`estimated_implicit_actuator_torque_Nm`; the auditor also interprets the old
field as an estimate. Historical raw files and hashes are preserved. Reported
old maxima do not independently certify actual motor torque.

An isolated fixed-base hinge check measures average net torque from known
constant inertia and angular acceleration, with gravity, friction and collisions
absent. It tests drive-only, aligned feedforward and opposing feedforward cases
in the pinned backend. A valid drive-only calibration is required before
interpreting the combined 0.76 Nm limit result. This probe loads no source data
and must pass before trusting a combined-budget claim in the next comparison.

Primary source for estimate semantics: [pinned IsaacLab ImplicitActuator](https://github.com/isaac-sim/IsaacLab/blob/28a37cecdd433c22d9eabd6a5954add9f13a8951/source/isaaclab/isaaclab/actuators/actuator_pd.py).
The [Omni Physics drive model API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html)
documents combined internal-drive and user-force effort constraints. The probe
checks the active runtime rather than assuming those semantics from a label.

## Shared motor-limit repair

The isolated runtime probe measured 0.7600 Nm with the drive alone, 1.3100 Nm
with an additional aligned 0.55 Nm joint effort, and 0.2100 Nm with opposing
effort. In this pinned runtime, the separate joint effort bypasses the drive's
0.76 Nm cap. Historical clipped torque estimates therefore cannot certify the
combined motor budget, including for the earlier isolated lifts.

Feedforward is now encoded as a bias of the implicit drive target:
`q_drive = q_posture + torque_feedforward / stiffness`. The separate joint
effort command is zero. Below saturation, this preserves
`K*(q_posture-q) - D*velocity + torque_feedforward`; the complete command now
passes through the native implicit drive's motor cap. Physical joint limits
remain unchanged. The bias is a force-control command rather than a desired
reachable finger configuration. Task targets also retain the existing current
state torque clamp. Static acquisition and lift use the same drive encoding.

83 CPU tests pass, including the unsaturated control law and simultaneous
posture/feedforward saturation in both directions. Before running an episode,
the isolated inertia-based probe must confirm both signed limits and the
unsaturated implicit dynamics.

The fresh `motor-budget-v3` simulator probe confirms signed saturation at
±0.7600 Nm and verifies unsaturated positive/negative feedforward scaling
against a calibrated drive-only inertial response. The measured unsaturated
gain is 0.998540. PhysX's iterative integration does not equal one full-timestep
implicit Euler update, so the probe does not assume that formula.

The repaired 100 Hz pot lift fails both fixtures despite passing gravity,
support and release controls. The old lift result cannot validate this repaired
motor command. Full carries were skipped at that gate. The next comparison
increases integration to 400 Hz while preserving the fixture clock: 0.25 s
closure, 0.25 s squeeze ramp, 0.5 s support release, 1.5 s duration and 20 fps
video. Lift metrics select the physical load interval, and force averaging
keeps its 0.2 s window. No motor or friction limit changes. 86 CPU tests pass.

The 400 Hz fixed-posture lift also fails both pot fixtures. Maximum wrist
rotation improves from about 6.0 to 2.6 degrees, but relative object drift is
still 2.62 / 1.08 m. Its gravity, support and support-removal controls pass.
The initial postprocessor rejected the increased raw record counts; a local
re-audit confirms all 2,404 sample records, 1,403 control records and timestamps
cover the complete 1.5 s fixture clock. Count checks now derive from that clock.
The original failed postprocessing receipt and raw hashes remain preserved.

The next unaccepted full-episode diagnostic compares fixed posture against the
live force/posture controller at 400 Hz, each with a fresh free-space gate.
The fixture uses frozen finger efforts and nominal payload compensation;
the full candidate uses current contact allocation and measured wrist reaction.
Neither fixture failure nor a passing wrist gate certifies the full controller.

## Audit rejection of aborted phases and grip solver profile

An aborted reacquisition can leave both hand and object stationary. Zero
relative drift in that case is not retention. The updated auditor marks an
aborted phase as failed and requires measured support to have been exercised.
A hash-bound synthetic simulator record reproduces this false-positive case.
Historical raw reports remain unchanged; new summaries apply the corrected rule.

The runtime warns that default TGS force integration can produce noisy velocities.
An optional `--contact-solver grip` profile enables external force updates every
position iteration, resolves articulation contact later, and requests at least
four velocity iterations. It changes solver accuracy and ordering; motor caps,
mass, friction, source observations and geometry are unchanged. Preparation,
invocation and reports bind the profile, and the full-stage gate must match it.
USD scene attributes are read back and checked against the requested values.
The default profile remains available for the recorded comparisons. The motor
probe accepts the same option. 87 CPU tests pass; simulator checks are required.

Sources: [pinned PhysxCfg](https://github.com/isaac-sim/IsaacLab/blob/28a37cecdd433c22d9eabd6a5954add9f13a8951/source/isaaclab_physx/isaaclab_physx/physics/physx_manager_cfg.py)
and [NVIDIA articulation stability guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html#articulation-solver-order).

## Measured motor/contact checkpoint

The optional grip solver passes the repeated direct motor probe and both new
pot lift fixtures. Their source readback confirms the requested scene flags.
The motor cap, nominal mass, friction, geometry and source data are preserved.
The profile differs from the default diagnostic solver; official evaluation
settings have not been matched. These are local experimental results.

| Full comparison | Pot 1 drift | Pot 2 drift | Lid 1 drift | Lid 2 drift | Wrist gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Shared motor cap + hybrid at 400 Hz | 129.82 mm | aborted | 29.98 mm | 208.85 mm | pass |
| Shared motor cap + frozen posture at 400 Hz | 126.78 mm | aborted | 88.71 mm | 291.32 mm | pass |
| Shared motor cap + verified grip solver + hybrid at 400 Hz | 115.08 mm | 13.62 mm | 7.01 mm | 15.42 mm | pass |
| Grip solver + fixed posture + 4 N squeeze floor | 434352.22 mm | aborted | 7.42 mm | 6.20 mm | pass |

Position drift alone does not establish rotational retention or full task success.
The JSON receipt includes rotation, contact coverage, aborted phases, source and
state/spec hashes. Historical raw audits remain preserved; the current derived
summaries apply the aborted/no-support rejection. The v4 input receipt binds
all recipes, including rejected comparisons, and remains local.
