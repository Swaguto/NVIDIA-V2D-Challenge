# Bounded grasp reference repair

## Observed failure

In `coupled-task-grip-solver-v1`, the first pot grasp has all three measured
pad contacts during source time 2.4–2.8 s. Its median thumb–index normal dot
product is −0.952, consistent with opposing contacts. No complete three-pad
contact is present during 2.9–3.3 s. Contacts subsequently return, but their
median normal dot is +0.075 during 3.75–5.25 s. The physical grasp no longer
matches the planned opposing contact geometry.

The robot reference contains reconstruction jitter and fades from measured
settled object pose to the absolute reconstructed pose after closure. The
resulting command offset reaches approximately 13 mm in object coordinates
while the pot is still supported. These are measured contributors to inspect;
they do not prove that reference repair alone solves transport.

## Implementation

`experimental_reference_denoising` adds bounded acceleration regularization
to the inferred robot reference. The horizon is 0.1 source seconds. Translation
knots move at most 3 mm; rotation knots move at most 2°. Quaternion signs are
unwrapped and fitted as normalized chordal coordinates. A constant upward
correction restores table clearance using recorded cooked geometry. A dense
audit rejects total changes above 6 mm / 3° against the previous cubic path.
The dense check is not a continuous collision certificate. Nominal finger load
planning uses the resulting reference orientation.

`experimental_register_settled_grasp` captures settled translation and body
rotation offsets once at
closure, requiring a settled pose within 3 cm / 10° of the inferred reference.
It preserves subsequent world translation and world angular increments. The offsets stay fixed;
an object that later falls is not chased. Lookahead calls cannot mutate the
registration. Previously aborted phases cannot register. Settling tilt cannot
rotate the transport path relative to the table. With the declared isotropic
nominal inertia, its world load wrench is unchanged.

The historical source37 trial used a spatial rigid alignment that rotated the
whole transport path. That rejected trial and its original bytes remain intact.

Both flags require cubic interpolation. Registration also requires bounded
reacquisition. Both default to false. The settings are recorded in the generated
grasp specification, report and hand readback, with every captured alignment.

Original source trajectories, frame numbers, timestamps, observation masks and
geometry remain unchanged. Object errors continue to use the original source.
Motor limits, materials, mass, source speed and gravity are unchanged. Object
poses are never assigned after initialization; no attachments or external object
forces are introduced.

## Verification protocol

The CPU suite has 95 passing tests, including bounded jitter rejection, retained
linear motion, quaternion sign equivalence, immutable inputs, fixed registration,
read-only lookahead, and rejection of late or aborted registration.

An additional optional `experimental_carry_compliance` candidate requires settled
registration and the same bounded failure guard. During carry, 90% measured-pose
feedback attenuates the reference error imposed across the grip. The full pose
error still drives the 6 cm / 20° failure boundary. A one-step prediction uses
world actor-frame twist and physical time, preserving the measured component of
target velocity. This prediction is a robot control input, never an object state
write. The [pinned runtime implementation](https://github.com/isaac-sim/IsaacLab/blob/28a37cecdd433c22d9eabd6a5954add9f13a8951/source/isaaclab_physx/isaaclab_physx/assets/rigid_object/rigid_object_data.py)
documents the actor pose/twist frame and xyzw convention. CPU tests check
attenuation, world-axis prediction and latched rejection of a dropped object.
This candidate also defaults to false.

The first compliant trial reduced world stiffness along with its pose error and
failed transport. The repaired controller scales wrist stiffness and damping by
`1 / (1 - feedback_weight)` during registered carry, at most tenfold. In the
rigid-grasp continuous limit this retains the original world tracking law while
adding impedance against relative grasp error. Its discrete coupled prediction
remains approximate. Approach gains and original force/torque limits are
unchanged. The regression test checks this limit and unchanged force saturation.
Nominal acceleration-aware load planning is independent of whether live contact
allocation is enabled, allowing a fixed-posture comparison with the same plan.

## Executed checkpoint

Five new complete e002 comparisons ran with fresh wrist gates and passing
gravity/contact controls. Every replay retains 343 original frames and 20 labelled
settling frames. None completes reliable pot transport. Fixed-posture compensated
carry maintains a tight grasp through source time 7 s, then loses contact near
7.9 s. Its unblended registered reference clears the table by approximately
55–69 mm during the later loss; the actual pot also clears it. Late loss is not
explained by table penetration. The early reference can still cross the table
by 4.2 mm after settling alignment, so collision-feasible reference planning is
not fully solved.

The smoothing-only comparison has the best new lid position diagnostics: 5.26 /
6.07 mm relative drift and 100% required-digit contact-gate coverage. Rotation
drift remains 10.31 / 11.24°. These are separate diagnostics, not task acceptance.
See [complete measurements](grasp-reference-repair-results-v1.json).

The optional underside search rolls the wrist about the two pinch contacts,
searches downward-facing object surfaces and verifies the resulting middle
contact pushes upward. Reconfiguration stays within original source-relative
joint and wrist bounds. The final search includes fixture-floor clearance in
the optimization and independent postcheck. All original 1.5 mm penetration
limits remain. Of 44 fitted trials across nine configurations, none passes
every contact, support-direction and clearance check. No rejected proposal was
sent to a GPU grasp trial. This bounded search does not establish infeasibility.
See [search receipts](geometry-search-results-v1.json).

The remaining priority is a physically robust grasp and collision-feasible
approach/transport, followed by fresh load and complete-episode validation.
These failures are not accepted demonstrations for policy training.

Compare two fresh complete e002 trials with the same hybrid controller and
400 Hz grip solver: smoothing alone, then smoothing plus closure alignment.
Each requires a fresh matching free-space wrist gate. Preserve every original
frame and labelled settling frame, all failed phases and unchanged retention
thresholds. Inspect contact normals, contact coverage, relative position and
rotation drift, and original source-object errors independently.

The optional grip solver differs from default settings. Official evaluation
compatibility and hidden-episode performance remain unverified. No complete
physical policy or policy training is accepted by these diagnostics.
