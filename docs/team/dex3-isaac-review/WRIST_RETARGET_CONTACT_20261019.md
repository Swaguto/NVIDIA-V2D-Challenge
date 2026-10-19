# Wrist retargeting and contact checkpoint

These are CPU experiments on non-reserved e037. No policy was trained, no
completed baseline or replay was rerun, and no complete physical task passed.
Source footage, model weights and result arrays remain private.

## Motion-constrained wrist reconstruction

The previous object-relative proposal changed only frames 36–69. That window
cannot satisfy the original acceleration cap: the repaired control already
exceeds it at acceleration row 70, whose three position knots are all fixed
under that restriction. This is a feasibility issue at the interval boundary.

The new proposal permits changes to right wrist commands at frames 25–99,
where source hand observations are neither direct nor metric. Direct and metric
commands, all source validity flags, the left hand, finger angles, object poses,
meshes and URDFs remain unchanged. A valid but inferred source hand flag is not
promoted to a direct observation.

The first SLSQP translation fit reaches its iteration limit and fails motion
checks. A convex OSQP projection then passes exact motion checks, but reports
inaccurate convergence; that control remains retained. A final SLSQP refinement
from the convex initializer converges in eight iterations. The completed
angular fit is reused, rather than rerun. Its analytical constraint Jacobian
agrees with central differences within 6.25e-10 in the sampled coordinates.

| Full reference motion summary | Original cap | Final proposal |
| --- | ---: | ---: |
| Maximum linear speed, m/s | 0.6223014934 | 0.5737553493 |
| Maximum angular speed, rad/s | 8.6099547292 | 8.5320580159 |
| Maximum linear acceleration, m/s² | 13.5397961244 | 13.5397947704 |
| Maximum angular acceleration, rad/s² | 111.3218987085 | 111.3218875764 |
| p95 linear speed, m/s | 0.5272371386 | 0.5272370859 |
| p95 angular speed, rad/s | 5.0782837183 | 5.0170609885 |

All six original caps pass. The five retained inferred-image checks also
improve over the previous repaired command: median errors at frames
40/45/50/55/67 change from 147.19/260.07/290.65/266.41/67.34 pixels to
46.77/58.73/79.01/79.81/43.68 pixels. Absolute errors remain substantial.
These are inferred source landmarks, not direct measurements or an independent
episode generalization test. No source image is used in this trajectory solve.

## Fresh native result: rejected

Right command frames are regenerated from the actual URDF before separately
freezing and compiling the new reference. The unchanged full 151-knot native
gate finds **five hand slots over 1 mm**, with worst penetration **6.302 mm**.
Failures occur at frames 35, 36, 37, 47 and 55, on the right index links.
Object/support penetration remains zero. The motion pass therefore does not
qualify this reference for training.

At frame 50, whole-digit separation lower bounds improve substantially, but
the thumb still sits 27.46 mm from the spoon. The index can overlap it. A zero
separation lower bound is not evidence of a stable grip.

## Finger reachability and the missing pinch

One-frame follow-ups keep actual URDF joint limits and use geometry-only
objectives. They never fit to held-out source image/depth values. Subsequent
image comparisons retain their inferred and nonmetric status.

1. A fixed-wrist search reduces the thumb distance upper bound to about
   2.6 mm, while the middle finger's bound remains over 4 mm. It is not
   accepted. The first global
   search exhausts its iteration budget; that failure is retained even though
   its subsequent Powell stage returns successfully.
2. A coupled proposal allows at most 8 mm per root translation component and
   0.08 rad per rotation component. Its selected translation changes by
   approximately 4.95 mm. All three distal surface-distance upper bounds fall
   below 0.006 mm at frame 50, with no object penetration. This narrow geometric
   improvement does **not** establish contact or force closure.
3. A separate witness audit verifies external object surfaces, then finds
   thumb/index outward-normal dot product **+0.8202**: they approach nearly
   the same side. There are **zero native hand/object contact pairs** at this
   pose. A palm/thumb self-contact of about 0.250 mm is retained in the report.
4. An explicit opposed-site control selects a thumb site with normal dot
   product **−0.7581** relative to the index. Under the same wrist bounds, the
   two local searches leave the thumb **21.36 mm** from that facing site. The
   proposal is rejected. This limited search is not proof of global
   unreachability.

The reachability producers save the lowest finite objective seen during their
searches, which can be an intermediate point rather than the optimizer's
returned stationary point. Optimizer return success is reported separately
and does not qualify the saved candidate. No one-knot candidate has a full
trajectory, velocity, swept-contact or force certificate.

## Next work

Compare the retained single-frame RGB hand's inferred palm orientation with
the current object-relative wrist and the actual Dex3 palm/hinge convention.
Audit chirality, coordinate transforms and contact-site association before a
new source-supported orientation proposal. Reuse the saved geometry; do not
rerun completed inference or expand the RGB probe.

Then solve wrist orientation and object-local opposing contact sites together
over time, with actual URDF joint/velocity limits. Every changed reference
needs fresh source and motion checks, the **unchanged full 1 mm native gate**,
swept checks and a grasp assessment before PPO. Establish a reference that
forms an opposing grasp before investing in additional PPO.

All earlier source-fit failures, including 31/32 held-out skin-ray coverage,
remain unchanged. No whole source hand or complete physical task is qualified.
All selected baseline policies still have MP-SR zero. Independent teammate
reproduction and a hidden leaderboard score remain unverified.

Exact experimental code and restoration instructions are archived under
`experiments/scored_baseline_20261019/wrist_retarget/`. It is not enabled as a
default preprocessing policy. Brev stayed stopped; these studies used no GPU
job or PPO step.

## Saved palm convention and pose audit

The follow-up audits reuse saved native RGB geometry without another inference
or source fit. The right MANO canonical finger direction maps to the Dex3
forward direction with dot product **0.99868** under the existing correction.
An independently constructed anatomical MCP basis differs by 5.31 degrees;
this is a descriptive model comparison, not a new adapter calibration.
There is no evidence here for an erroneous 180-degree right-hand flip.

The retained raw, multiview-fit and failed hard-skin palm orientations differ
from the current inferred wrist by **25.77, 29.23 and 26.97 degrees**. All are
model estimates; none is a measured or qualified source pose. Their wrist
positions disagree too: the raw full pose shifts 129.30 mm from the current
robot root, while the two fitted poses shift about 38.4 mm. Copying one of
these transforms is not a validated repair.

A necessary thumb-reach bound uses actual joint-origin lengths and the
native distal mesh radius. Its relaxed radius is 117.35 mm. At the exact
current root, the frozen proposed thumb site is at least 3.62 mm beyond
the distal workspace; at the raw-orientation-only root the bound is 3.29 mm.
This ignores joint limits, facing constraints and collisions. It excludes
2 mm proximity only for those exact fixed roots and that one proposed site.
It says nothing about the coupled root boxes or other grasp sites. Zero
lower bounds at the other roots do not prove reachability. Native FK agrees
with the URDF, and sampled actual thumb poses lie inside the analytic sphere.

## Retained source-oriented controls

The first new orientation-only control has an invalid numerical warm start:
the old robot rotation lies outside the residual box around the new source
orientation. Powell warns, and the selected intermediate is out of bounds.
It is rejected and retained. Version 2 clips only the warm start to the
declared box and checks every objective evaluation and selected candidate.
Its thumb still misses the facing site by **23.76 mm**. Bounds and gates are
unchanged.

Three predeclared full source-pose centers are then tested under identical
residual bounds and actual URDF limits. All source-fit failures and confidence
flags remain unchanged; no source pose is selected as a winner. Two local
starts fail. A bounded differential-evolution follow-up uses seed 37, population
multiplier 6 and 80 iterations, followed by Powell. Every global search reaches
its iteration cap; the hard-skin case also exhausts the Powell evaluation cap.
Those solver failures remain explicit. The saved lowest-cost intermediate is
not claimed to be a converged stationary point.

| Retained source center | Thumb / index / middle facing-site distance, mm | Native distal object contacts |
| --- | --- | --- |
| Raw RGB | 25.42 / 69.42 / 89.45 | None |
| Unqualified multiview fit | 4.28 / 1.87 / 1.07 | Two thumb contacts |
| Failed hard-skin fit | 4.00 / 2.21 / 1.46 | One thumb and one index contact |

The last pair has native-normal dot product −0.4166. None passes the planned
site checks. Their maximum native penetration is below 0.263 mm at this one
knot, and inferred-image errors improve. Neither fact qualifies a trajectory
or repairs the source hand's retained held-out failures.

## Contact-load audit

A separate CPU audit tests every saved global-search contact set against
gravity and the estimated frame-50 reference motion load. It reads native
contact positions, dimensions and friction, including torsional or rolling
components when present. These cases have dimension-3 point contacts.
Support and hand self-contacts are excluded from object support.

The LP uses an **outer friction box** containing the native cone, with
unlimited nonnegative normal force. It invents no measured force budget and
omits motor limits, making this an optimistic feasibility test. Three analytic
controls verify opposed support, one-sided failure, and axial-torque failure
of a two-point pair.

**All three frozen contact sets fail both gravity and estimated-motion load
feasibility.** The raw case has no hand/object contacts. The other two fail
even in the optimistic outer model. The model uses a nominal 0.300 kg object
and its saved COM/inertia; these are not measurements of the source spoon.
The motion load comes from inferred object poses. No simulator step or
measured contact-force claim is made.

This is evidence against these particular contact configurations, not proof
that the task is impossible or that a policy could never change contacts.
More gain or reward tuning on the same frozen grasp would miss this failure.

Exact producers, declared plans and lossless restoration are under
`experiments/scored_baseline_20261019/palm_diagnosis/`.

## Reusable load check and new source-conditioned sites

`scripts/dex3_training/v2d_dex3_flashchord/contact_load_feasibility.py`
now implements the explicit outer-model diagnostic. It validates finite points,
orthonormal force bases, dimensions 1/3/4/6, friction order, COM and wrench units.
Its native geometry adapter binds the object side before reversing contact axes.
An independent equilibrium and friction check follows an optimal LP result;
unresolved numerical/solver errors raise instead of qualifying a candidate.
Empty geometry supports only an exactly zero load.

The result names `feasible_in_outer_model` and `force_origin` distinguish the
optimistic solved forces from measured forces. There are no invented force caps,
motor limits or source contact labels. The existing measured-budget
`wrench_capacity.capacity` API is unchanged. **34 focused tests pass**, including
the existing capacity tests. Analytic cases cover gravity, unsupported moments,
spin versus rolling, anisotropic friction, COM/world transforms, native object
binding and a friction-box corner outside the exact elliptic cone. All six
saved native contact/load outcomes match without repeating a native job.

A new finite screen samples three object-local surface sites near each retained
inferred hand's projected thumb/index/middle locations. It uses an explicitly
exploratory 8 mm surface patch, at most six samples per digit, fixed native
geometry/friction, opposing normals and necessary actual-morphology bounds.
The wrist seed is each saved source pose; its residual boxes remain 8 mm and
0.08 rad per component. No wrist or finger optimization runs. Both saved gravity
and inferred-motion wrenches enter the feasibility screen before retargeting.

| Source hypothesis | Triples tested | Both modeled loads feasible | Selected seed gravity / motion total normal force, N |
| --- | ---: | ---: | ---: |
| Raw RGB | 216 | 36 | 20.25 / 27.57 |
| Unqualified multiview fit | 216 | 79 | 254.05 / 232.77 |
| Failed hard-skin fit | 216 | 69 | 251.11 / 295.47 |

These are **potential surface contacts**, not contacts realized by robot fingers.
Within each source hypothesis, the recorded seed minimizes source-point distance
among the sampled feasible triples. There is no source-pose winner, no claim of
a minimum-force grasp across the family, and no motor feasibility certificate.
The raw hand is 104–110 mm from its nearest object surface before projection;
the 8 mm patch does not erase that discrepancy or bound source error. The other
two hypotheses retain their source-fit/held-out failures. Their selected sites
also require hundreds of newtons even in the optimistic model. A finite search
does not establish global feasibility or infeasibility.

Next verify the source skin/visibility association of these sites and actual
robot realization and motor loads before constructing a trajectory. Preserve
source uncertainty and all failures. Any changed commands need original
root/finger motion limits, fresh full 1 mm native, swept and grasp checks before
GPU/PPO. No new policy, physical task or source hand is qualified. Brev remains
stopped. New code-only producers and plans are under
`experiments/scored_baseline_20261019/contact_load_sites/`.
