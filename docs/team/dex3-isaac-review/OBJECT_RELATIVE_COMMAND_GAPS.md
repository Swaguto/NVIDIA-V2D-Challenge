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
Native collision and physical rollout checks are still required. This change
does not fix the original 12–21 mm finger/object intersections at observed
anchors. More training is deferred until those targets are addressed.
