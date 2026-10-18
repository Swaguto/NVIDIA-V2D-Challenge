# Partial table overlap experiment

The earlier translation experiment applies its support constraint only when the
whole projected body fits inside the finite measured footprint. That conservative
rule skips an overhanging rack, including frames where native physics detects
rack/table intersections.

`finite_support_overlap` now clips the actual visual mesh triangles against the
convex support footprint and returns exact height extrema over that overlap.
It checks triangle interiors as well as vertices. Low vertices outside the table
do not impose an infinite-plane constraint. The calculation is invariant under
a common rigid coordinate change. Native convex collision shapes can differ
from this visual mesh, so native validation remains necessary.

`refine_translation(..., partial_support=True)` optionally adds a conservative
overlap witness constraint. The default remains off. It rechecks the complete
finite triangle overlap after fitting and rejects a new intersecting solution.
A wholly below-table overhanging body is left unclassified. The original rigid
body, measured pixels, source confidence, fit-improvement checks and displacement
bound are preserved.

Thirteen focused CPU tests pass, covering partial overlap, triangle interiors,
off-table geometry, rigid frame changes, malformed inputs and contradictory
depth. In the four-frame e015 rack probe, this option enables the previously
skipped support constraint. Only the final-frame candidate passes the source-fit
checks; the difficult frame 53 is rejected. This is not a complete repair and is
not enabled in the frozen baseline.

Two related CPU screens also remain unpromoted. A constant-world rack hypothesis
passes the original measured-depth and jump gates on all 166 frames, but its mean
source fit is 0.5391 versus 0.5602 for the selected dynamic track. A seed-height,
support-normal motion hypothesis scores 0.5164. The moving spoon rejects both
priors strongly. These scores do not establish hidden pose accuracy; none of
these diagnostics justifies replacing the complete track or extending PPO.
