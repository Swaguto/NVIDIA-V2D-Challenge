# Object-relative command gap experiment

The e020 control loses right-hand observations between source frames 35 and
110. Its world-space wrist interpolation passes about 20 cm from the watering
can's nearest mesh vertex at frame 80. At the two observed endpoints the wrist is only
23.8 and 20.5 mm from their nearest mesh vertices. The endpoint poses in the can's frame differ
by 20.0 mm and 14.4 degrees. All intervening object poses are observed estimates.
These measurements concern reconstructed references, not hidden ground truth.

`prepare_scored_case.py --gap-policy object-relative` adds an optional command
hypothesis. The default remains `world`. The policy runner accepts the same
choice as `gap_policy` in its configuration. Export a fresh committed policy
context before using it.

For each interior hand gap, the rule requires:

- The same nearest object at both selected IK anchors, with a 1 cm margin over
  other objects and endpoint wrist-to-mesh-vertex distance below 12 cm.
- Directly observed, finite rigid object poses throughout the interval.
- Endpoint relative wrist translation within 3 cm and rotation within 30 degrees.
- Endpoint duration no longer than five seconds.

It interpolates wrist position and orientation in that object's frame. It keeps
the original finger interpolation, every observed command, all object poses,
the source confidence flags and the source clock. It never extrapolates before
the first anchor or after the last. Ambiguous cases keep world interpolation.

The proposal records the inferred object slot and every applied/rejected gap.
Endpoint consistency does not establish continuous contact: release and
regrasp within an occlusion remain possible. These are inferred robot commands,
with no fabricated human observation or contact label.

## Validation status

Eight CPU mechanics tests cover rotating/moving objects, uncertainty rejection,
frame invariance and rigid-transform validation. The actual e020 case applies
the rule to 74 right-hand frames and leaves all observed commands unchanged.
The CPU native comparison reuses byte-identical object, robot, support and
convex collision geometry. Object/clock/confidence arrays, observed robot
commands and finger commands are checked against the control.

At frame 80, right fingertip-to-can signed gaps change from 124/110/157 mm
to 17/21/38 mm. This still does not establish contact. At frame 40, the worst
hand/object penetration increases from 21.75 to 24.45 mm. The observed initial
left-hand penetration remains 20.81 mm. A zero from `mj_geomDistance` alone is
not a contact or clearance certificate; actual native contact samples are also
retained.

**Do not promote this gap rule or train longer on this candidate yet.** Native
collision-aware retargeting and full-task validation remain necessary. A
bounded CPU feasibility probe can reduce some intersections below 1 mm, but
one sampled left-hand frame remains above 6 mm and other fingers remain far
from the object. Neither a favorable optimizer status nor small penetration
alone qualifies as a grip.

The revised analytic native-Jacobian projection clears all five sampled poses.
Its full-trajectory check still fails **44 of 404 hand/frame slots**, with a
worst remaining intersection of 23.02 mm. All 44 failures lie in source slots
without metric hand support. Source metric support covers only the last 32
left-hand frames and right-hand frames 0–16 and 165–201 in this episode.
This is a source-depth/retargeting problem during the interaction; it cannot be
declared fixed from favorable sampled poses. No projected trajectory is promoted
and no additional PPO is run on this candidate.

The complete native audit of the original references finds left cup
penetrations above 1 mm in 156 frames and right can penetrations in 73 frames.
The object-relative candidate increases the right can count to 130 frames.
Its native geometry and source confidence remain identical to the control.
