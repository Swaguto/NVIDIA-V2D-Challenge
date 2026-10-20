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

An additional saved-geometry audit checks the selected sites against whole-hand
and finger-associated distal skin triangles, using the licensed MANO topology
privately. No source fit or skinning inference is repeated. The raw sites remain
70–97 mm from their associated distal skin. Multiview-fit sites are 4.35–5.89 mm
away; failed-skin sites are 2.17–5.84 mm away. In the failed-skin case the index
skin/object outward-normal dot is only −0.135, so source proximity alone does
not give a facing contact. The other fitted associations have more opposed
normals, but are still inferred and unqualified.

All selected index sites project outside the inferred hand mask and inside the
inferred object mask. Several projected sites miss the inferred hand model's
front surface. These are model/mask indicators, not direct visibility, depth or
contact observations. Every missing result stays explicit; the original 31/32
held-out failure and all source flags remain unchanged. No pose winner is chosen.

Next construct a source-skin-associated contact/robot seed that includes the
required wrench and actual motor loads before constructing a trajectory.
Preserve source uncertainty and all failures. Any changed commands need original
root/finger motion limits, fresh full 1 mm native, swept and grasp checks before
GPU/PPO. No new policy, physical task or source hand is qualified. Brev remains
stopped. New code-only producers and plans are under
`experiments/scored_baseline_20261019/contact_load_sites/`.

## Source skin and actuator-constrained pose seed

The reusable load diagnostic now has a separate actuator-aware entry point.
At a concrete pose it combines contact Jacobians, free joint effort and supplied
joint effort bounds while solving for force redistribution. It preserves the
original outer-only entry point and measured-budget capacity API. **51 focused
tests pass**, including virtual work, gravity consuming effort, native versus
asset ratings, spin/rolling Jacobians and redistribution across contacts.

The saved compiled model confirms every finger's runtime joint clamp is
approximately **±0.76 Nm**, with actuator gravity compensation inside the clamp.
The pinned upstream G1/Dex3 controller uses the same 0.76 Nm configuration.
Larger URDF asset ratings are not used to increase the limits. Gravity
compensation consumes this joint budget; it is not free passive support.
[MuJoCo documents this clamp convention](https://mujoco.readthedocs.io/en/3.4.0/XMLreference.html#body-joint-actuatorgravcomp).

A new source-conditioned site generator starts at finger-associated distal
**skin triangles**, without projecting a distant joint onto the object. It
requires at most 8 mm inferred skin separation, opposing skin/object normals,
and at most 1 mm inferred skin-model penetration. These are exploratory proposal
criteria, not measured uncertainty or a source qualification. Four nearest
distinct sites per finger are retained before load screening. Raw RGB has no
nearby sites. The two fitted cases yield 24 and 44 opposed triples; **all 68
fail gravity balance in the outer model**. This finite pruning does not prove
anything about the complete skin patch or globally possible grasps.

The generator retains NumPy matrix-multiplication warnings. A separate explicit
tensor calculation verifies its finite object vertex set within `5.73e-17` m
and selected source centroids within `1.12e-16` m. The warning cause remains
unproven; no failed controls are discarded.

The previously recorded load-balanced potential sites remain a positive control
for site selection. A new centroid/facing/penetration check finds only the
unqualified multiview case satisfies all three exploratory skin associations.
Raw separation and the failed-skin index normal reject the other cases. This
does not select the multiview pose as source truth or erase its held-out failure.
The old load solve, site family and visibility audit are not rerun.

One new coupled solve tests that eligible potential-site control. Variables
include the wrist within the unchanged 8 mm/0.08 rad component boxes, seven
actual URDF finger ranges, barycentric points on fixed native distal triangles,
and forces for both saved loads. Contact coincidence, facing, friction,
equilibrium and native joint effort bounds enter together. Hand gravity and
recorded native stiction are explicit. No invented normal-force budgets or
larger asset effort ratings enter. Only the optimizer's returned candidate is
recorded, not its best finite intermediate.

**The solve is rejected.** SLSQP reports a positive directional derivative after
34 iterations. Returned hand-site distances are **2.85 / 9.82 / 13.05 mm** for
thumb/index/middle; contact/force/effort constraints fail. Its conditional
potential-site effort estimate reaches **5.64 Nm**, but the assumed contacts
are not realized, so this is not a measured torque or a global motor
infeasibility proof. A separate native position/velocity check finds **zero
hand/object contacts** and no penetration at this knot. The actual-contact load
checks fail for both required loads. No physics step runs.

Analytic free-gravity effort and point Jacobians agree with this new native pose
within `1.84e-9` Nm and `2.83e-9` m/rad. This validates the calculation and joint
binding; it does not qualify a grasp. Source flags, the original 31/32 held-out
failure, motion limits and the full 1 mm gate remain unchanged. No whole
trajectory, physical task or new policy is qualified.

Next diagnose the grip-region geometry and discrete robot contact-patch
assignment using the saved failure, rather than resweeping this fixed-triangle
solve or increasing bounds/caps. Any new proposal requires source support,
original motion limits, fresh full native/swept checks and actual motor/grasp
validation before GPU/PPO. Brev remains stopped. Exact code-only producers and
declared plans are under
`experiments/scored_baseline_20261019/actuated_skin_seed/`.

### Object-independent palm-origin control

A subsequent diagnostic derives a translation from the saved index/middle MCP
midpoint and actual robot joint origins, keeping every source orientation fixed.
Object sites, forces and image errors do not enter that derivation. The same
robot-local offset `[4.82, 4.63, 7.75]` mm appears in all three saved source
models, within `8.88e-17` m. This is a morphology-based hypothesis, not a new fit.

**It is also rejected.** It worsens the inferred thumb image error in every
case, leaves about 18.0 mm error at each MCP anchor and a 20.1 mm unused thumb-base
error, and falls outside the existing world-component translation box. No box
is enlarged, default mapping changed or new center adopted. The actual robot
index/middle base separation is **57.0 mm**, versus **22.7 mm** in the inferred
source hand. Half the span difference alone gives a **17.15 mm** necessary
worst-anchor error for any rigid alignment of those two bases. This concerns
MCP alignment, not global grasp reachability or task impossibility.

The next representation should distinguish inferred human hand anchors from
robot contact assignment. In particular, an inferred or missing human middle
finger is not evidence that all three robot fingertips must contact its three
chosen points. Develop explicit object-local robot contact-patch/mode hypotheses
under the actual morphology and unchanged effort, motion and source checks;
test native contact manifolds and the required wrench rather than only isolated
surface proximity. Preserve the rejected three-point/fixed-triangle and MCP
controls. Do not resweep them or adopt the failed MCP center. No source
confidence, contact labels or qualification gates change.
## Object-local contact patches and compiled-support correction

The next CPU investigation separates inferred human anatomical anchors from
robot contact-region hypotheses. New reusable `contact_patches` contracts bind
object-local external triangles to the exact object/model hash and support
two-digit as well as three-digit modes. Twenty-five focused contract tests pass.
A new analytic two-box control also verifies signed native distance and its
point-Jacobian gradient in separation and penetration. No prior native job was
recompiled or physical replay repeated.

The first regional bank clips external convex faces to a frozen, inferred grip
region formed from both retained fitted source hypotheses. This union is not a
measured uncertainty interval or an observed contact label. Potential eight-point
opposed patch sets need about 14.26 N total normal force for the worse of the two
saved modeled loads, versus hundreds for the old source-closest isolated sites.
This potential-contact comparison does not apply motor limits or establish that
the robot realizes those contacts. The raw distant source case stays rejected.

Three bounded dynamic whole-finger trials allow proximal or distal surfaces to
change during the solve. All three fail actual contact/load prerequisites.
The two-digit thumb/index trial improves the retained inferred image errors but
has only one unmatched native index contact; the other modes also regress an
image observation. Native contacts, including unmatched contacts, are retained
in their load checks. Single-knot penetrations stay below 1 mm, which is not a
full-trajectory certificate. These are failed controls, not stationary solutions
or evidence of global unreachability.

## A concrete representation error

A separate frozen-pose audit finds that near-zero raw-hull thumb witnesses are
still 2.18–4.80 mm from the actual native object geometry. Broadphase masks allow
the pairs; stale witness updates and collision filtering do not explain this
gap. The new contact search had rebuilt hulls from every raw `mesh_vert`, while
the pinned native support function follows a compiled graph with fewer vertices.
For example, the distal thumb has 381 raw vertices but 64 native support vertices.
Its old witnesses lie 2.34–2.60 mm outside the compiled thumb hull. The object
witnesses also lie about 0.34 mm outside their compiled object hull.

New `native_mesh.compiled_mesh_vertices` follows the pinned MuJoCo 3.8.1
vertex-global-ID table and fails closed on unknown layouts or invalid spans.
Eighteen new focused tests include a discarded protrusion that must not become a
collision surface. Independent support-plane checks at all seven saved witness
pairs match native signed distances within **9.4e-16 m**. This isolates a bug in
the new geometric contact proposal, not the authoritative native collision
audits. The original failures and native model remain unchanged. The first audit
producer passed its tests/numerical assertions but failed JSON serialization of
a NumPy boolean; its failed log is retained. Version 2 fixes serialization only,
reuses the 18 passing tests and writes the verified result.

## Corrected bounded control still rejected

A separately frozen bank now uses compiled support vertices for both object
patches and robot surfaces, with unchanged source region and model parameters.
Its potential patch-pair loads are 16.54 N (X), 13.16 N (Y) and 49.70 N (Z).
These are still solved forces on assumed geometry, not motor or grasp success.
Only one predeclared thumb/index control is executed with this correction.

That trial stops after five iterations without an acceptable bounded step.
Surface target errors are **1.19 mm / 3.92 mm**. It has one actual index contact
outside the assigned patch and **no thumb contact**; both gravity and inferred
motion load checks remain infeasible. Single-knot penetration is 0.272 mm.
Inferred image errors improve from 71.19/86.83 to 39.36/78.39 pixels (landmark
IDs 4 and 12; neither is direct metric truth). No complete source hand, trajectory,
grasp or policy is qualified, and no training or GPU work runs.

Next, contact-region choice must account for actual compiled-hand reachability
and a common bounded root before choosing the lowest potential-force patch
pair. The finite failed controls do not prove that all patch modes are
unreachable. Preserve every old bound, failure and source flag; do not resweep
these raw/corrected local controls or promote proximity to contact truth.

## Finite reachability atlas and common-root facet LP

The next method samples actual URDF joint configurations at the unchanged
current root orientation: 344 thumb, 50 index and 50 middle configurations,
including each original command. Compiled whole-digit facets are compared with
the saved inferred object-local patches. Four nearest object-facet centroids
and a finite retained bank limit the search. These pruning choices and fixed
orientation mean the atlas cannot establish global unreachability.

Necessary translation-box tests find 34 thumb/index facet pairs and no
thumb/middle pairs in this finite bank. The predeclared top 32 are frozen.
The new reusable `common_translation_contact_seed` chooses barycentric points
on both facets and **one common root translation**, with original 8 mm component
bounds. It does not run a local wrist or finger optimizer or change a source
pose/uncertainty center. Nineteen focused tests cover common-root closure,
incompatible independently reachable contacts, two/three facet cases, unchanged
bounds and conservative whole-hull halfspaces. The first test fixture failed
NumPy broadcasting in three tests; its exact fixture/log are retained. The
production solver was not changed to accommodate that fixture.

Nine of the 32 facet LPs are geometrically feasible. All nine produce actual
native thumb contacts, but **none passes the physical prerequisites**. Eight
have two actual hand/object contacts; one has three. All nine improve the two
retained inferred image observations but fail the 1 mm full single-knot native
check and both actuator-aware load checks. Their maximum native penetrations
range from 2.591 to 15.184 mm. Facet closure is not a collision-free hand pose.
All actual contacts, including unmatched and third-digit contacts, are retained.

The same frozen queue is then checked with conservative support-plane
constraints from all compiled right-hand/object hull pairs, allowing at most
0.5 mm support overlap inside the unchanged 1 mm native gate. Fixed-configuration
self-penetration is also retained. No new joint configurations are generated.
All 32 whole-hull LPs are infeasible. This concerns these fixed configurations,
facets and separating directions; it is not proof of global task infeasibility.

## Required wrench rank and a distinct self-contact gap

A new analysis reads the saved native contact records without rerunning native
geometry or any load LP. Four analytical mechanics controls pass. For eight
two-point, dimension-3 contact sets the unrestricted wrench matrix has rank 5;
both required modeled loads lie outside its span even before friction or effort
bounds. Relative scaled residuals are 2.11–10.57% for gravity and 28.45–38.06%
for estimated motion. Increasing force or reward cannot make those frozen
contact sets balance those loads. This is not an impossibility result for a
different contact manifold or moving policy.

Candidate 4 has three native point contacts and rank 6, with load-span residuals
below 1.3e-15, yet its saved constrained load checks still fail. The responsible
friction/effort component is not isolated by this rank check. Source mass,
friction and motion remain nominal or inferred, not measured truth.

That same candidate's object contacts are only 0.278–0.559 mm deep, but its
right palm/thumb-1 self-contact is **2.591 mm**. The unchanged left reference has
a separate **2.098 mm** palm/thumb-1 self-contact at this knot. The existing
hand/object/support auditor explicitly records `hand_self_contacts_assessed=False`;
its previous certificates do not cover this failure. No existing certificate is
rewritten and the full single-knot check is not relaxed to ignore it.

Next audit hand self-contacts over the cached reference, as a new scope rather
than rerunning the completed hand/object audit. Then develop source-constrained
collision-free finger retargeting and contact modes with a sufficient actual
wrench span. Three point contacts need not mean three mandatory human or robot
digits: multiple legitimate native contacts on a finger may contribute, but
must be realized and verified rather than invented. Do not resweep these finite
atlas/LP controls, enlarge bounds, alter condim/morphology/effort or retry PPO.
All source, original motion, fresh full 1 mm native, swept, self-contact, motor
and grasp checks remain necessary. No complete task or policy is qualified.

## Full self-contact audit and left-thumb correction

A new cached-model auditor covers the previously unassessed same-hand scope.
It verifies frozen hashes, both complete body/joint registries and independent
URDF FK across all 151 knots, with no scene recompilation or physics step.
Fourteen focused tests cover both hands, empty knots, clock completeness,
duplicate contacts, body ambiguity and invalid gaps. The original reference
fails self-contact tolerance on **151/151 left** and **46/151 right** knots.
The left maximum is 2.291 mm; the right maximum is 7.248 mm, including
thumb/index and thumb/palm pairs. Native FK closes within 3.21e-9.

All failing slots carry direct source-frame flags; 24 left and 26 right failing
slots also carry a metric flag. These frame flags do not make an occluded
thumb's inferred depth or derived robot angles measured truth. Human landmarks
and all confidence/authority data remain fixed during the new retargeting.
Both neutral zero-angle hands have clear palm/thumb-1 gaps around 18.3 mm.
Reference-pose gap Jacobians agree with two central-difference steps within
4.4e-11 m/rad. Thus the measured reference collisions are not inevitable in the
neutral model; their effect on prior physical jitter or scores remains unisolated.

A new left frame-50 thumb-only constrained probe converges in seven iterations.
It reduces native self-penetration from 2.098 to 0.500 mm while thumb image error
improves 11.775 to 11.353 px. Root poses, other fingers, right hand, objects and
human observations stay unchanged. The old 0.25 rad component cap is measured
against the original aligned reference, preventing accumulated bound expansion.

The full per-frame SLSQP continuation preserves all source/image, position and
motion checks in its returned path, but 75 optimizer returns report positive
directional-derivative failures. That path is retained and rejected. A distinct
one-joint bracketed construction converges geometrically at every knot but
regresses image fit at eight knots, maximum 0.207 px; it is also rejected.
A source-compensated version fixes four of those constraints, but the remaining
four cannot move thumb-0 in the needed direction because its original joint
bound is active. No bound or source gate is enlarged.

The final constructor selects one available source-compensation direction from
thumb-0 or thumb-2 using the native/image Jacobian and actual bound authority.
It uses one analytically directed bracket per active source constraint, not
restarts or a parameter sweep. Thumb-1 solves the native self-gap equation by
bracketed root finding. Eight image constraints require compensation: thumb-0
at frames 54–57 and thumb-2 at 120–123. All original controls remain retained,
including all failed optimizer statuses. Original held-out skin failures are
not used, relabelled or removed.

### Independent changed-reference verification

The final candidate is frozen separately; original source/authority files are
byte-identical, and both wrist roots, right-hand commands and left non-thumb
fingers remain exact. Fresh cached-model verification on the **new commands**
finds **zero left self-contact knots over 1 mm**, maximum 0.500 mm. Independent
URDF projection checks all **604 left source image anchors** without regression;
median error improves 15.364 to 15.332 px and p95 remains 24.723 px. No new
per-landmark metric-depth truth is claimed. All original actual URDF position,
velocity, 0.25 rad component and 0.05 rad adjacent/second-added finger limits pass.
Exact root equality preserves the previous root-motion summaries.

The 450 quarter/half/three-quarter self-contact samples have maximum left
penetration **0.500445 mm**. This is a sampled left-self check, not a continuous
or full-task swept certificate. A fresh hand/object/support check still rejects
the same five right-index knots, maximum 6.302 mm; the unchanged right hand also
retains all 46 self-contact failures. The complete native gate remains rejected.
No physics, PPO or new policy ran, and no MP-SR or leaderboard gain is claimed.

Next retain this verified left component and address the right thumb/index/palm
self-contacts under the same source, actual URDF and original motion bounds.
The old finite grasp controls remain rejected. Right-hand/object penetration,
full swept/motor/grasp checks and whole-source qualification are still required
before any further learning.

## Right-hand contact modes and original temporal limits

Eight new guarded CPU jobs start from the verified left-hand freeze, preserving
its commands exactly. Worst right self-contact gap Jacobians at frames 35, 116,
119 and 121 agree with two central differences within 6.9e-12 m/rad. Native
right thumb/palm and thumb/index pairs require different constraints; no source
or joint authority is inferred from a derived robot angle.

The first right bracket constructor stops when it reaches missing source hand
observations. Its failed log is retained. Corrected contexts keep all 136 missing
right image anchors explicit and freeze those 34 frames' right commands. There
are 468 available right anchors and 604 left anchors. Frame direct/metric flags
remain unchanged and do not establish measured occluded-joint depth.

The full source-compensated right bracket path still fails at six frames and
exceeds original temporal/velocity limits. A new whole-clock sequential-convex
method stops on a numerical QP failure before accepting a new step. Its numerical
repair uses shared image-frame slacks and linear programs, but stops without an
acceptable merit step: maximum penetration 1.972 mm and source regression
3.005 px remain. These are rejected controls, not successful optimization or
physical success. Finite image-direction approximations never replace the exact
Euclidean source gate in candidate verification.

A separately declared late contact-mode component uses source-eligible commands
from frame 116 onward; the earlier failed transitions stay unchanged and visible.
A fixed cubic bridge first exceeds the original 0.05 rad second-added bound
(0.0584 rad). That failure is retained. One convex projection onto the unchanged
original component, adjacent/second-added, actual position and velocity polytope
converges. It does not relax source, collision or motion gates.

### Independently verified combined component

Fresh 151-knot native/FK verification on the new private freeze finds **left
0/151** and **right 13/151** self-contact failures; the right count falls from
46, and its maximum penetration falls 7.248 to 2.646 mm. Every right knot's
maximum self-depth is nonregressing. Remaining failures are frames 0–4 and
110–117. All 1,072 available original source image anchors are nonregressing;
combined median improves 15.083 to 14.978 px and p95 stays 30.012 px. All 136
missing anchors remain missing, with unchanged commands at those frames.

Actual URDF position/velocity, original 0.25 rad finger component bounds and
0.05 rad adjacent/second-added bounds pass. The right second-added bound is
active at 0.05 rad; no new envelope is introduced. Both roots, the verified left
component, right nonthumb fingers, objects and human/source files remain exact.

The five right-index/object penetration knots remain, maximum 6.302 mm at frame
47. Another 450 new quarter/half/three-quarter samples interpolate complete rigid
wrist and object poses plus fingers. Sampled maxima are 0.500445 mm left-self,
2.618326 mm right-self and 4.553011 mm hand/object. These samples retain failures
and are not a continuous/full-task swept certificate. No physical task, full
source hand/reference, motor/grasp success, new policy or score gain is claimed.
No physics, GPU or PPO ran; Brev remained stopped. Exact producers/plans are in
`experiments/scored_baseline_20261019/right_self_contact`; private results stay
private and old failed controls are not overwritten.

Next preserve this **combined** freeze, including both left and right repairs.
Diagnose the remaining observed transitions using actual thumb kinematics,
source image rays and active original joint/velocity/temporal authority before
another mutation. The failed local brackets and whole-clock controls do not
prove global infeasibility. Do not repeat their sweeps or enlarge bounds. Any
new command still needs fresh source/motion, full native/self, swept, actual
motor and grasp checks before learning; the existing object failures remain.

## Analytic thumb source rays and finite temporal branches

A new explicitly bound Y/Z/Z thumb-chain module enumerates all nonsingular
point-IK branches within supplied joint bounds. Empty branches and an axial
singularity are distinct outcomes. Twenty-two focused tests cover analytic
positions, radial/elbow branches, angle wrapping, tangencies, limits and invalid
inputs. Nine retained native configurations close to analytic geometry within
9.6e-10 m. The first test import failure is retained; the test repair does not
repeat native binding. Numerical closure does not establish source accuracy.

At the 13 remaining right-self failures, a new finite study samples 1,025 depths
on each of two declared image rays: the source-image center and the current
robot tip pixel. A whole-chain workspace sphere bounds possible depths. All
analytic branches are checked against original component/URDF position limits,
source-image nonregression and actual cached native self-contact geometry.
These are hypothesized robot tip depths, not measured human or contact depth.
No source inference, old optimization or completed native audit is repeated.

The finite bank contains source/native 1 mm seeds for frames 0–4, 110 and
115–117; the two sampled rays supply none for 111–114. This is a constructive
finite bank, not an exclusion of other allowed image rays or continuous poses.
An exact dynamic program over consecutive node pairs finds a path for frames
0–4 with unchanged frame-5/6 boundaries. It enforces actual velocity and the
original first/second differences of corrections against the original aligned
reference. Four analytical controls include a nongreedy predecessor example;
the first faulty synthetic fixture is retained and corrected separately.

### Independent new-reference verification

Fresh 151-knot native/FK/source/motion checks on the new private freeze find
**left 0** and **right 8** self-contact failures, down from right 13. Remaining
right frames are 110–117, maximum 2.613 mm at 111. The verified left and late-right
components, all roots, nonthumb fingers, objects, source/human arrays and flags
remain unchanged. All **1,072 available image anchors** are nonregressing;
136 missing right anchors remain missing with unchanged commands. Original
URDF position/velocity, 0.25 rad component and 0.05 rad adjacent/second-added
bounds pass. The right second-added bound remains active at 0.05 rad.

The early reference knots approach the existing 1 mm gate: maximum 0.993366 mm.
Their 15 new quarter/half/three-quarter samples have maximum 0.990957 mm. Across
all 450 sampled complete rigid-root/object/finger states, right-self still reaches
2.608131 mm and hand/object 4.553011 mm. Sampling and narrow nominal margins do
not qualify continuous swept motion or physical execution. All five right-index
object failures persist, maximum 6.302 mm. No whole task, full source/reference,
motor/grasp success, policy or MP-SR gain is qualified. Brev stayed stopped.

### Retained fixed-root compatibility diagnosis

A distinct necessary-geometry study at frames 111 and 113 uses conservative
trigonometric/source-image intervals and common inscribed balls in actual
compiled palm/thumb-1 hulls. A common ball of radius greater than 0.5 mm in both
solids is a lower bound exceeding 1 mm on translation needed to separate them.
The finite numerical enumeration logs rejection of every box under its fixed
root, original joint box and image-error cap, then fails in report serialization:
a path variable was overwritten by a rotation matrix. Its failed status remains.

A separate recovery reads the saved logs without repeating subdivision, sphere
LPs or earlier controls. Binary-tree counts close, 512 independent URDF projection
samples fit the declared image enclosures, and 16 new native frozen-center checks
respect the saved common-ball lower bounds. Per-leaf certificates and original
native center-control values were not saved. This supports an incompatibility
diagnosis under the stated numerical model; **formal global infeasibility is not
certified** and source accuracy remains unqualified. No candidate is promoted
from this diagnosis. Exact executed sources and historical dependencies are
archived code-only under `experiments/scored_baseline_20261019/thumb_source_rays`.

Next preserve all three verified components in the new combined freeze. Audit
source authority for the derived robot wrist orientation and its coupling to
thumb geometry at the remaining transition before any root mutation. Direct
human wrist observations are not measured robot joint/root angles. Do not adopt
rejected centers, enlarge the 8 mm/0.08 rad residual boxes, relax source/motion/
1 mm gates, or repeat the completed branch/path/bound studies. Any justified new
proposal needs fresh whole-clock source/native/self/motion, full swept, actual
motor and grasp checks before GPU/PPO. No root change or paid job is queued.

## Wrist authority and source-constrained coupling

The frozen exporter and retarget code hashes match the bundle manifests. The
exported wrist orientation comes from the MANO model's global orientation,
transformed through the calibrated camera and existing Dex3 palm convention.
The retargeter assigns it a soft orientation cost of 0.1; tip positions have
position costs and no tip-orientation cost. The actual robot wrist and joint
angles remain derived IK results. Direct/metric flags at the remaining observed
transition describe frame-level source/depth support. They do not establish
measured robot orientation, per-joint ground truth or an orientation covariance.
The original human observations and flags are unchanged.

A declared half geodesic step toward the retained source-model orientation at
111/113 fails analytic finger-branch prerequisites and is retained as a failed
control. A new coupled root-orientation/actual-URDF-finger solve at each of those
two knots converges within the unchanged 0.08 rad root and 0.25 rad finger
component boxes, with root translation fixed. Both optimizer returns pass
source-image and model-orientation nonregression and native right-self depth
of 0.5 mm. Independent URDF/native/source verification closes point positions
within 2.7e-9 m. Neither pose has a hand/object contact. This is a constructive
isolated-pose result, not a measured wrist correction or a realized grasp.

Joining the poses remains unresolved. A dense 104:118 trajectory reaches its
120-iteration cap. Explicit SO(3) root/pixel derivatives subsequently pass new
numerical controls (maximum pixel derivative difference 1.85e-6 px/rad); the
old norm-error finite difference is sensitive near an almost exact image
anchor. This does not isolate every cause of the optimizer failure. An
algebraically equivalent squared-source-constraint formulation with those
derivatives also reaches its unchanged 120-iteration cap and fails source
nonregression. Both returned trajectories remain rejected; no best intermediate
or unconverged return is promoted.

A distinct constructive path uses only the converged isolated seeds and fixed
103/118 boundaries, plus exact remaining scalar finger-motion LP intervals.
It stops at 111 on a failed conditional nonlinear solve and source regression.
The scalar LPs certify only their stated linear motion intervals. They do not
certify a future collision/source-feasible pose. Independent sparse insertion
of the two verified poses also fails finger velocity and original temporal
caps: maximum adjacent correction 0.1731 rad and second-added correction
0.3459 rad, versus unchanged 0.05 rad caps. The original root motion summaries
pass that sparse check; it still cannot qualify a trajectory.

Eight new guarded CPU jobs and exact sources are archived code-only under
`experiments/scored_baseline_20261019/source_wrist_authority`. The original
three-component freeze remains byte unchanged. Eight right-self and five
hand/object failures persist; no new full reference, motor/grasp success,
physical task, policy or score gain is qualified. No source fitting, model
compilation, physics stepping, GPU or PPO was performed. Brev remained stopped.

Next assess temporal reachability of coupled source-supported pose sets before
another path solve, including actual joint speed and first/second correction
differences. The failed greedy construction and dense local solves do not prove
global infeasibility. Preserve all source/held-out failures, existing root
centers, original boxes and source/motion/native gates. Any future changed
reference needs fresh full-clock and swept/motor/grasp qualification.

## Coupled finite-pose reachability

A new saved-control linear audit finds a joint-only bridge between the retained
111/113 poses under actual URDF limits, original component corrections and both
temporal bounds. Its independent scalar witnesses satisfy their linear
constraints. This establishes neither source/root feasibility nor collision
clearance between those poses. The isolated angles are hypotheses, not measured
robot joint values that future paths must reproduce.

Two reusable geometric tools now have **34 focused tests**: sphere/ray root
mapping with an explicit minimum-swing convention (18), and bidirectional
pair-state joint temporal reachability (16). The latter enforces actual speed
and original-reference adjacent/second-added differences, keeps empty banks,
and agrees with independent exhaustive small graphs. Neither tool certifies
source accuracy, native contact, root motion, swept motion or dynamics.

A conditional finite bank using thumb configurations, ray-bound common roots
and planar other-finger inverses has 58 eligible nodes, with no nodes at six
rows. Its fixed-pixel convention excludes some allowed source motion: saved
isolated middle-tip errors imply at least 7.434/3.882 px of pixel motion at
111/113, whereas that generator specifies a 0.0001 px inward shift. These
isolated controls were imported separately, not generated by the locked rays.

The first graph adapter fails on the retained native thumb bound's 1.197e-9 rad
difference from the rounded URDF value. This failure remains archived. The
corrected adapter explicitly binds the original reference check's 1e-8 rad
numerical position tolerance; it does not change component/temporal caps or
physical gates. The finite bidirectional graph has no complete path.

A new source-disk projection representation permits the **original** image
inequalities instead of locking an arbitrary pixel. It reuses earlier eligible
nodes and produces 2,003 eligible finite hypotheses. Rows 109/112/114 remain
empty, so its graph also has no complete path. At 109, generated roots or
other-finger inverses fail before full native evaluation. At 112/114, all
140/1,155 new full-pose evaluations fail self-contact. Finite grids, two twist
conventions and conditional control nodes do not prove global impossibility.
Per-state rejected commands were not saved; aggregate counts are not formal
per-state exclusion certificates. Do not repeat these banks to reconstruct
those missing records or claim their failures cover unrepresented poses.

Eight guarded CPU jobs are archived code-only under
`experiments/scored_baseline_20261019/coupled_pose_reachability`. Both library
test runs and the initial graph adapter failure are retained. Source/human
arrays, missing/held-out failures, native parameters and the three-component
freeze remain unchanged. No whole reference, full task, motor/grasp success,
policy or MP-SR improvement is qualified. No GPU/PPO or physics stepping ran.

Next treat the remaining free common-root twist as a source-constrained
kinematic degree of freedom, rather than only two selected swing conventions.
Derive allowed intervals from original source-model orientation and root bounds,
then account for original finger/image and temporal constraints. Use the saved
isolated poses as controls, not measured mandatory path nodes. Do not resweep
these completed finite banks, old local solvers or source/gain/reward families.
Keep all original bounds and source/native gates, and save any new interval
witnesses and unresolved boxes explicitly. Full-clock and swept/motor/grasp
qualification remain required before learning.

## Explicit common-root twist and source/finger intervals

A new one-dimensional twist module preserves an already mapped robot thumb
point and solves the source-model orientation cap as a quaternion sinusoid.
Numerical principal-log envelopes partition the original world-component root
box. All splits, rejected/inside/unresolved leaves and witnesses are retained.
Its **22 focused tests** include independent rotation grids, envelope containment,
leaf cover and cut-locus/budget handling. Neither numerical enclosures nor a
source-model orientation establishes measured hand accuracy or a global proof.

Real-case controls produce finite witnesses at 109/111/113. A construction
using the original image disk adds 25 witnesses at 112, while the particular
tested 114 ray families do not produce an accepted pose. The thumb geometry
borrowed for these tests is explicitly a component hypothesis from a rejected
whole path; no old path or certificate is promoted. Root centers, source flags,
human arrays, actual model parameters and original bounds remain unchanged.

Independent source/URDF/native verification of four selected new witnesses
closes point positions within 2.6e-9 m and checks self, external and interhand
contacts at those knots. Sparse insertion still fails actual finger speed and
the original temporal caps: 0.1731 rad adjacent and 0.3459 rad second-added,
versus 0.05 rad. Original root motion passes the sparse diagnostic. This is
neither a whole-clock reference nor a swept, motor, grasp or task certificate.

Separate necessary image bounds use enclosing orientation-cap balls and
cylinders, with four analytical controls each and saved support/tangent records.
They do **not** exclude the tested frozen 114 thumb geometry: the stronger
finite directional bound is 20.813 px versus the original 24.201 px allowance.
No global infeasibility or constructive feasibility follows from that overlap.

A distinct three-variable convex robot-point construction fails an analytical
control before reaching the episode case. The first return was not saved;
its assertion failure remains. A start-only numerical repair saves returns
before assertions but still returns status 8, with a near-feasible failed
control. Both methods remain rejected; no more solver restarts, tolerance
tuning or intermediate promotion is justified by those returns.

A new finite 32-ray image-cone boundary study at 114 produces 40 root witnesses.
Two fail exact root/orientation checks; 38 fail the bounded index inverse.
A subsequent actual planar-index/source-cone inequality is represented by a
tan-half-angle quartic, with four analytical controls. Five declared distal
samples at each saved root yield no accepted index interval. No new native
state follows from either family. Those finite samples do not exclude other
continuous distal angles, thumb articulations or source-cone/root configurations.

Eleven guarded CPU jobs and exact sources are archived code-only under
`experiments/scored_baseline_20261019/root_twist_intervals`. No physics/GPU/PPO
ran. The three-component freeze, eight right-self failures and five object
failures remain; no full reference, motor/grasp success, task, policy or score
gain is qualified. All original source/held-out failures remain explicit.

Next investigate the coupled thumb-articulation/index-workspace domain under
the original source-cone and root bounds. The distal thumb joint can change the
tip vector while the palm/thumb-1 geometry is fixed, but new thumb-2/finger and
object contacts still require native checks. Treat the existing isolated poses
as controls, not measured mandatory joints. Use saved interval evidence before
declaring any new bounded study; do not resweep these completed ray families,
five distal samples, failed convex controls, old pose solvers or source fits.
Keep unresolved continuous domains explicit and save all new certificates and
state rejections. A complete new source/native/motion/swept/motor/grasp result
is still required before learning.

## Continuous coupled articulation domain

Actual URDF ancestry and three new native distal-thumb states confirm that
thumb-2 changes the tip while leaving palm/thumb-1 geometry fixed. Analytic
tip/native closure is below 7.16e-10 m. Seven new controls also include 384
independent SO(3), principal-log and continuous finger enclosure samples.
These controls validate the stated representation, not source accuracy.

A new eight-variable numerical interval study at 114 varies the common-root
Rodrigues ball, thumb-2 and both index/middle joint pairs continuously under
the original source disks, model-orientation cap, root box and joint bounds.
The fixed upstream thumb pair is a retained robot hypothesis from a rejected
whole path, not a measured human angle or new source-pose winner. All 12,001
nodes and proposed states are saved. The declared budget leaves 150 unresolved
boxes; no evaluated center passes every source/root prerequisite, so there
is no new native candidate. This is not global infeasibility.
The final binary split exceeds the nominal 12,000-node budget by one; the
actual 12,001-node count is retained, with no change to the CPU resource caps.

An independent saved-record audit closes the complete binary partition and
checks 364 actual-URDF camera projections against its numerical enclosures.
All samples fit. The best diagnostic proposal still has 24.826 px thumb error
against the unchanged 24.201 px allowance and remains rejected. The numerical
padding is not a formal directed-rounding or global exclusion certificate.

A distinct conditional thumb-1/thumb-2 slice reaches its output writer but
fails on NumPy Boolean JSON serialization. Its proposed states and native
outcomes were not saved. The failure/log are retained; no physical outcome
or qualified point is inferred and the slice must not repeat to reconstruct
its missing results. Future proposals and native outcomes must be converted
to plain JSON scalars and journaled immediately, before assertions or further
work. This failure does not justify changing any source or native gate.

Four guarded CPU jobs and 11 exact sources/plans are archived code-only under
`experiments/scored_baseline_20261019/coupled_articulation`. No production
runtime changed and no physics/GPU/PPO ran. The three-component freeze and
all missing/held-out source failures remain unchanged; eight right-self and
five index/object failures persist. No whole reference, task, motor/grasp,
policy or score improvement is qualified.

Next derive the original source-image conic's preimage in the actual planar
thumb workspace together with the proximal thumb/palm clearance relationship.
Use saved coupled-domain records, explicit robot-hypothesis authority and
actual compiled geometry. Distinguish a necessary bound from a constructive
pose, and do not resweep either the eight-variable bank or failed slice.
Preserve original boxes, temporal authority and all source/native gates;
full-clock, swept, motor/grasp and source qualification still precede learning.

## Source-conic preimage and parent-clearance witnesses

A new representation pulls the original image-error disk back to a conic in
the actual thumb plane. Its numerical ellipse/circle quartics partition the
negative distal 2R branch by original joint and clearance limits. Constructive
points use a 0.02 px inward margin without changing the original source gate.
The conditional root/index/middle command remains a robot hypothesis, not a
qualified human/source pose or a failed optimizer promoted to truth.

Actual compiled palm/thumb-1 hulls supply shared-ball penetration lower bounds
and support-plane overlap bounds over the original proximal range. All 47
interval records and LP returns are saved; leaves comprise seven clear,
12 over-gate and five unresolved intervals. Seven analytical/durable-output
controls pass. Independent samples of every certificate fit its stated bounds.
This numerical geometric scope is distinct from actual active native contacts.

The original and strict conics each have 51 boundary arcs. A frozen analytic
source queue leads to ten new native position checks, six of which pass.
Independent URDF/source/native checks confirm all six frame-114 poses, with
tip closure below 2.56e-9 m. The closest control has 0.9365 mm self-penetration
and zero external/interhand contacts. Margins remain narrow and absolute
source errors remain large. No grip or whole reference is qualified.

Seven necessary motion-only LPs can connect that one control in `[104,119)`;
their paths omit source, native and root feasibility. Sparse insertion fails
the original added-motion limits at 0.2482 rad adjacent and 0.4961 rad second,
versus 0.05 rad. Original sparse root-motion statistics pass separately.

A new saved-node temporal compatibility calculation uses six controls, four
distinct 113 nodes and 125 distinct 115 nodes. All 3,000 conditional pairs
fail: 1,128 full joint boxes are empty and 1,872 reject the fixed controls.
Index-0, thumb-0 and thumb-2 are outside the allowed interval for every saved
pair. Freeing only the middle finger therefore cannot connect these particular
finite nodes. This does not exclude other source-constrained poses or prove
global task infeasibility. No earlier bank, graph, source fit or failed slice
was repeated; all new states and native outcomes were durably saved.

Four guarded jobs and exact code are archived under
`experiments/scored_baseline_20261019/thumb_conic`. The existing three-component
freeze, missing/held-out source failures and original model/gates remain
unchanged. Eight right-self and five object failures remain. No physics/GPU/PPO,
full trajectory, motor/grasp, task, policy or score success is established.

Next derive the allowed axial-thumb/index branch around 113–115 from the saved
temporal boxes, then account for the source conic plane's dependence on axial
thumb/root orientation. Do not fix the isolated controls as measured mandatory
nodes, promote a motion-only LP path, resweep completed banks/slices or loosen
source and native gates. A new construction still needs independent original
source/URDF/native/motion checks and full-clock/swept/motor/grasp qualification.

## Correlated axial/index temporal domains

New work retains the correlation of each saved 113/115 neighbor pair rather
than replacing the union by independent per-joint ranges. Of 500 distinct
pairs, 188 nominal joint boxes are empty, 36 are numerical-boundary cases and
276 are nominally nonempty. Local maximum root-motion checks pass; global
p95 and outside boundary motion remain unassessed.

Full seven-joint source enclosures exclude 103 nominal boxes through index
source fit. All 128 new actual-URDF enclosure samples pass. A single declared
source/kinematic-selected box has two axial-thumb sections and two middle
sections but no index section, so it yields no native candidate. Five new
harmonic algebra/contract controls pass; no old solver or bank is repeated.

A new continuous positive-distal 2R workspace/conic intersection predicate
tests corners, source-ellipse-center containment and all numerical boundary
intersections. Four analytical controls pass and all 173 remaining domains
are numerically empty. An independent linear-support calculation, using the
four circular workspace boundaries instead of quartics, supplies separating
planes for all 173. Three controls and 1,384 new actual-URDF checks pass; the
minimum plane separation is 2.2907 mm after a 1e-8 m binding guard.

The 36 near-tolerance boxes are not discarded. A separate necessary-bound
resolution uses the original 1e-7 rad motion tolerance explicitly, preserves
the 173 prior certificates by hash and examines 139 previously unverified
cases. All 500 finite conditional pairs are accounted for: 188 outer joint
boxes stay empty and 312 have index-source separators. The 417 additional
URDF checks pass. Candidate gates remain unchanged: outward source/motion
allowances apply only to conservative geometric exclusion, never acceptance.

This identifies a conditional wrist/index-source conflict: the particular
114 wrist control cannot connect these retained finite neighbors while
meeting the original source and finger-motion constraints. It is not global
unreachability, measured source truth, or a reason to increase joint/root
bounds. Other continuous neighbor articulations and allowed wrist orientations
remain unexcluded. Numerical certificates are not formal directed-rounding
proofs. A journal key-name typo is retained with an explicit private correction;
no numeric source gate changed and no scientific result was recomputed.

Six guarded jobs are archived code-only under
`experiments/scored_baseline_20261019/axial_temporal_domains`. Production runtime
is unchanged. No new native position, physics, GPU/PPO, complete reference,
motor/grasp, task, policy or score gain is qualified. The verified three-component
freeze still has eight right-self and five object failures.

Next derive camera-cone support witnesses and necessary common-wrist changes
from the saved separators. Preserve the original source-model orientation and
world-component root bounds, actual morphology and temporal authority. Keep
neighbors uncertain and do not fix isolated controls as measured mandatory
nodes or resweep completed source/solver/bank/slice families. New commands still
require independent full original source/native/self/motion and swept/motor/grasp
qualification before learning.

## Global camera-cone duals and necessary wrist domains

A new derivation lifts the saved planar source separators into genuine 3-D
camera-cone halfspaces. For positive depth, the original image disk has
`||B C|| <= radius * Cz`; any unit disk dual `u` supplies
`(B.T u - radius * ez).C <= 0`. At an ellipse support point this is a tangent
plane. Its pullback agrees with the saved planar separator by a positive
scale. The old 312 planar minima are reused, without rerunning their solves.

Eight new analytic controls pass. Independent checks cover 1,248 source-cone
points and 936 new actual-URDF states; tip closure is below 1.67e-16 m. This
validates the geometric lift and a rigid-rotation necessary bound. It does not
validate source accuracy, native contacts or a trajectory.

With translation fixed, each particular retained finite-neighbor domain needs
at least 0.00383–0.01215 rad rotation from the isolated frame-114 conic control
to remove its source-plane separation. The minimum over that finite family is
about 0.219 degrees. This loose lower bound is below the conservative allowed
upper bound of 0.17994 rad; it neither excludes allowed wrist changes nor
specifies a constructive change. The center is the isolated robot control,
not a measured wrist orientation or the current verified whole reference.

A separate continuous root-box study uses the original source-model
orientation cap, original world-component residual box and actual planar
index workspace. A 2,047-node binary partition has 1,024 leaves: 141 exclude
source-model orientation, four exclude even the original index authority,
37 exclude the conditional finite-neighbor union, and 842 remain unresolved.
The latter regions are necessary outer regions, not feasible robot poses.
Continuous neighbors, root temporal statistics, other fingers, self contacts
and motors remain unqualified. The source-model estimate is not measured truth.

The first writer fails on a duplicate `kind` keyword after saving all six new
support controls and its first proposed box. That failure is retained. Version
2 repairs serialization and reuses those controls by hash, without rerunning
them. A separate preparation NameError/missing-plan attempt changed no guard
state. Independent verification checks the complete binary cover, 5,718 scalar
boundary supports, 512 SO(3) box samples and 64 new URDF states. Scalar/batch
support differences stay below 6.94e-18 m. These are numerical certificates,
not formal directed-rounding or global exclusion proofs.

Six guarded attempts are archived code-only under
`experiments/scored_baseline_20261019/wrist_cone_domains`. Production runtime
and tests are unchanged. No new native position, physics, GPU/PPO or command
mutation ran. The verified three-component freeze still has eight right-self
and five object failures; no whole reference, grasp, task, policy or score
improvement is qualified.

Next develop a constructive coupled wrist/index source-cone branch and thumb
conic/clearance chart under the original temporal authority. Preserve continuous
neighbor uncertainty. Unresolved root boxes are not accepted seeds, and the
isolated controls are not mandatory measured trajectory nodes. Do not expand
the partition budget, resweep completed controls or relax source/native gates.

## Coupled source wrist mode and compiled-face charts

A new common wrist mode starts from the verified three-component reference.
Its axis uses only source-index differential geometry and source-model
orientation constraints over a smooth time profile. The first linear program
selects a tangent orientation direction: exact caps leave only 1.27e-5 rad
amplitude. It is retained and is not a useful physical-reference candidate.
Three analytic and three actual source-derivative controls pass.

A distinct strict-interior source-cone program maximizes common inward
orientation margin; four analytic controls pass. Its one declared midpoint
amplitude is 0.033644 rad. All 19 changed roots pass the original model
orientation/component constraints and all six original full-clock root motion
summaries. Collision, force and physical outcomes do not enter this selection.
Source-model orientation remains an estimate, and this root-only result does
not qualify source images, fingers, contacts or a whole reference.

The new compiled palm-face chart intersects vertex sinusoid inequalities to
obtain continuous proximal-angle domains at finite axial-thumb sections.
Distal source-image domains are continuous within each section. A 0.75 mm
constructor overlap and 0.02 px inward source margin construct subsets; the
original 1 mm and image acceptance gates are unchanged. Actual compiled
support vertices are used throughout. Independent world-URDF surface checks
on 32 saved points close the support gaps within 7.45e-10 m. This does not
certify native active contacts or all possible separating axes.

The newly constructed 151-row source/finger family has 68 actual-URDF chart
points but 13 empty rows and no path under original position, velocity and
component/adjacent/second-added limits. The old isolated neighbor controls are
not mandatory nodes. No native position is evaluated because the full source
and temporal prerequisites fail. Finite axial samples, fixed other-finger
distal sections, sufficient palm faces and node pruning are explicit scope
limits; this does not establish continuous or global infeasibility.

A separate new full 2R source-domain diagnostic uses this wrist hypothesis,
without changing its amplitude. Of 38 index/middle domains, 28 have numerical
constructive source witnesses and ten remain unresolved at zero-distal/chart
boundaries. Three of five failed fixed-distal sections gain witnesses when
both actual joints are free: middle at 103/104 and index at 109. These are
isolated mathematical source witnesses, not a temporal or native certificate.

Five serial CPU jobs are archived code-only under
`experiments/scored_baseline_20261019/coupled_source_branches`. Ten analytic
controls, three derivative controls and 100 new chart/independent URDF states
pass their stated scopes. Production runtime/tests, human/source flags and the
current verified freeze are unchanged. Eight right-self and five object
failures persist; no native position, physics, GPU/PPO or score gain occurred.

Next handle the zero-distal/conic-chart domains explicitly and develop source
branches allowing both index/middle joints before reconsidering thumb/source
continuation. Retain unresolved domains and original temporal authority.
Do not resweep the two wrist modes, amplitudes, old fixed sections or failed
finite graph, and do not promote the root-only proposal to a reference.

## Closed source charts and temporal source components

The ten previously unresolved source cases comprise nine zero-distal middle
domains and one nonelliptic index conic. A new constructive closed 2R chart
uses camera harmonics on all four workspace boundaries and checks finite
conic stationary points. It supports straight-joint boundaries and retains
unresolved results when no witness is verified. Eight new analytic contracts
pass. All ten domains now have constructive source witnesses; all 64 new
witnesses pass independent actual-URDF/source checks. The earlier 28 solved
domains are reused without recomputing their conics.

Independent checks assess 221 reused/new source points. Twenty-four points
fail their actual gates and remain rejected; every one of the 38 source
domains retains verified points. The maximum analytic/URDF closure is
1.12e-16 m. These are robot image-fit witnesses, not measured contact/depth.

A new source-cone workspace homotopy constructs intermediate robot points
and inverts them through actual 2R kinematics; joint poses are not naively
interpolated. It retains 25 invalid workspace inverses and verifies 452 new
URDF/source points. Its finite four-joint path fails the first transition.
Three new analytic controls pass; no global exclusion follows from pruning.

A separate one-direction source-gradient/bracket construction verifies 34
source points but leaves four branches unresolved: index 106/108/109 and
middle 110. Its full proposal fails original adjacent/second-added caps at
0.1195/0.2296 rad versus 0.05 rad and remains rejected. Three analytic controls
pass; no restart, wrist-mode change or physical outcome enters the method.

Exact per-digit finite reachability removes Cartesian node pruning and
retains all saved points. Two independently exhaustive analytic controls pass.
Integrating 20 distinct, independently passing source points from the scalar
constructor produces a complete middle-digit path. The rejected whole scalar
proposal is not promoted. Saved-evidence extraction checks all 151 rows and
direct original bounds: component 0.13692 rad, adjacent 0.04879 rad, second
0.04842 rad and maximum velocity 11.4717 rad/s. The index still has no path
in the retained finite corpus; its first empty forward pair is 108→109.

The private middle source/motion component is an intermediate robot hypothesis
under the fixed wrist mode. It is unqualified for thumb source fit, native
collision/self contact, swept motion, motors, grasp or a whole task. The current
verified three-component reference remains unchanged with eight right-self and
five object failures. No native position, physics, GPU/PPO or score gain ran.

Eight serial guarded CPU jobs and exact code are archived under
`experiments/scored_baseline_20261019/closed_source_continuation`. Production
runtime/tests are unchanged. Sixteen new analytic controls and all 707 URDF
kinematic bindings pass; 24 source/position points remain explicitly rejected.
The integrated factorization reuses its two controls without rerunning them.

Next derive necessary source/temporal compatibility around the index 108→109
transition using saved correlated pair boxes and the continuous two-joint
source chart. Keep the middle source/motion component as an intermediate
control, retain continuous neighbor uncertainty, and do not rerun the completed
source constructors or finite graphs or increase their sample budgets.

## Continuous index source and temporal domains

The saved 107/108 index banks contain 14 and 13 nodes. All 182 consecutive
pairs are retained as a conservative superset; missing forward-pair identities
are not reconstructed. Original actual velocity and added-motion checks reject
151 pairs locally. Of the remaining 31 correlated next-angle boxes, 26 have
constructive source-chart witnesses. Independent actual-URDF/source/triple
checks accept 155 of 173 proposed triples and retain all 18 failures. These
local triples do not establish a feasible prefix or future trajectory.

The five unresolved charts receive new global camera-cone dual support checks
over their entire planar workspace rectangles. All five separate numerically
from the original source cone; 160 independent URDF samples respect the saved
support witnesses. This is a numerical fixed-pair restriction, not a formal
directed-rounding or global reachability proof.

A new formulation creates source-qualified continuous angle boxes around
passing saved points, preserving correlated per-clock regions. Its 550-box
mixed linear path model is infeasible under original temporal limits. This
inner-family failure is retained and does not establish continuous exclusion.
Its two path controls and two independent cone-enclosure controls pass.

A separate new method considers the full continuous original index domain at
the unchanged wrist mode. Camera-cone tangents and sinusoid Taylor remainders
give necessary linear relaxations; an inscribed polygon and opposite remainder
sign give sufficient constructive programs. All four new analytic controls
pass. The declared 255-domain study ends with 122 numerical linear exclusions
and twelve unresolved leaves. No returned point passes every source row. No
budget expansion, failed-intermediate promotion or old graph rerun occurs.

Independent checks verify the complete binary partition cover, 256 new actual
URDF/Taylor states and 152 source-inner-box states. Maximum URDF/camera closure
is 2.23e-16 m. A distinct dual calculation supplies eight residual-corrected,
finite-box contradiction witnesses for the first eight saved empty leaves;
three new analytical controls pass. Active dual rows concern source and box
bounds, so these witnesses do not isolate a temporal cause or require a wrist
change. Twelve outer regions remain unresolved. Dual weights are mathematical
coefficients, not physical forces or causal importance scores.

Six serial CPU jobs, exact code and dependency snapshots are archived code-only
under `experiments/scored_baseline_20261019/index_temporal_domains`. Fifteen new
analytic controls and 741 actual-URDF assessments pass their stated scopes.
All rejected states, solver returns and unresolved domains are retained
privately. Production runtime/tests, source authority, missing/held-out failures,
the separate middle source/motion component and current verified reference are
unchanged. Eight right-self and five object failures persist. No native
position, physics, GPU/PPO, qualified new reference or score gain occurred.

Next derive a continuous index source-boundary corridor jointly with original
temporal limits, using the saved unresolved domains and source-only dual
patterns as restrictions. The source cone's exact tan-half-angle 2R preimage
can expose correlated proximal/distal branches without another finite point
bank. Retain continuous-neighbor uncertainty; do not rerun these completed
charts, MILP or 255-domain study, increase budgets, change the wrist mode or
infer a necessary wrist mutation from source-only exclusions. A concrete new
path still needs independent whole-clock source/URDF/original motion before
native/self, swept, motor/grasp and whole-source gates.

## Algebraic index source corridors

A new tan-half-angle representation derives bidegree-(2,2) camera numerators
and an at-most bidegree-(4,4) source-disk polynomial. Its common denominator
is positive for finite chart coordinates. Positive camera depth is a separate
inequality. Half-angle poles and the straight-distal kinematic singularity
retain explicit scopes. No root or complex branch is silently selected.

The first producer fails on Python list division before its first synthetic
sample; source, guard and log are retained. V2 changes only that expression.
Eight new analytic controls pass, covering independent complex-exponential
kinematics, the disk identity, a known bivariate-square range, negative depth,
straight-distal behavior, half-angle poles, fixed points and camera contracts.

Nineteen actual index source preimages and 228 numerical Bernstein charts
reuse the twelve saved outer regions without a new partition or old LP/source
rerun. Five regions have numerical source exclusions; seven remain unresolved.
Independent actual-URDF camera binding checks cover 304 new states, closing
within 4.45e-16 m. A separate tensor-Bernstein evaluation checks 7,752 basis
values against the saved power coefficients; all stay inside the saved
padding, with maximum basis discrepancy 8.89e-16. Source/depth polynomial
identity errors remain below 1.12e-15 after normalization.

These bounds concern the fixed robot/source hypothesis. Floating coefficients,
tan endpoints and numerical Bernstein padding are not a formal directed-rounding
or global task proof. They do not qualify source accuracy, a temporal path,
native contacts or a wrist mutation. The current verified reference and
separate middle component remain unchanged. No native position, physics,
GPU/PPO, new full reference or score gain occurred. Three bounded attempts
and exact code are archived under
`experiments/scored_baseline_20261019/index_source_polynomial`.

Next derive continuous proximal-root branches and their distal critical events
from the saved polynomials, preserving real/complex roots, leading-coefficient
drops, singularities, poles and unresolved cases. Use seven remaining regions
as outer restrictions, not accepted poses. This precedes any new path solve;
do not repeat polynomial generation, the 228 charts, old partition/LP/graph
families or enlarge their budgets. A constructive path still needs independent
whole-clock source, actual URDF and original motion before physical gates.

## Continuous root charts at the first index transition

Eleven new exact-arithmetic controls cover Sylvester determinants, degree
changes, repeated and complex roots, endpoint roots, fixed intervals and
unresolved root-isolation budgets. Arithmetic concerns the saved floating
coefficient model; it does not validate upstream source or camera accuracy.

The frame-109 critical-family calculation hits its declared 32,768-bit cap.
Intermediate coefficients were not durably saved before subsequent work;
the precise failing stage is not established. This failure stays unresolved.
No rerun reconstructs those missing values, and no cap increases. Four other
leading/proximal-boundary families have saved exact zero real-root counts on
the scoped distal interval, with complex roots retained.

A distinct method compares saved Bernstein grids with the frozen power
polynomial through exact rational coefficient identities. It bounds source
error and the proximal derivative without repeating the old transforms.
Four new controls pass. Rational common-zero charts assess three local
sections, retaining all seven full-region identities. Their node counts are
127/7/7. Six unresolved event boxes remain in the first section; the other
two have no source-boundary tangency. No failed resultant is reconstructed.

Five regular distal cells yield implicit proximal source strips. Nine new
proximal specializations retain real/complex roots, endpoint behavior and
unknown event bands. Independent verification checks 135 exact coefficient
identity values, 6,210 subdivision identities, all 141 chart nodes and 251
saved Sturm root-count certificate nodes. Two new actual-URDF source witnesses
close within 5.56e-17 m and pass the original image and joint-position gates.

Both sparse insertions fail original temporal authority: adjacent corrections
reach 0.07732/0.08074 rad and second-added corrections 0.15331/0.16016 rad,
against unchanged 0.05 rad caps. Neither is a full source/path or native seed.
The two empty local sections reject six conditional outer regions in the
saved coefficient model; region 2 remains unresolved. This is not global
task impossibility or measured source/contact truth.

Five serial guarded jobs and exact code are archived code-only under
`experiments/scored_baseline_20261019/index_root_charts`. All fifteen new
analytic controls pass, while the critical-family cap failure remains explicit.
No native position, physics, GPU/PPO, new qualified reference or score gain
occurred. Current verified hand components, the separate middle component,
source authority, missing/held-out failures and original parameters/gates are
unchanged. The next construction must couple adjacent clock rows through
the original temporal inequalities; do not select more isolated frame-109
points or treat the current witnesses as mandatory measured nodes.

## Five-clock continuous source/time equations

A new research-only half-angle formulation couples index source branches over
107–111. It retains the full correlated outer region 2, copies five saved
source polynomials by hash, and preserves the three unknown frame-109 event
bands. It does not regenerate source models, root charts or old time studies.
The saved isolated witnesses are not mandatory clock nodes.

For `Q = sum(k_j q_j)` and `q_j = 2 atan(u_j)`, a polynomial complex product
represents `Q/2`. Its squared modulus is a positive product of `1 + u_j^2`.
Once angle bounds establish the sine-sign branch, lower/upper signs give
polynomial inequalities for each original motion gate. No possibly negative
`1 + u_t*u_previous` denominator is multiplied through without authority.
The actual 22 gates comprise eight adjacent-added, six second-added and eight
actual-velocity constraints, giving 44 polynomial inequalities in ten
continuous variables. Actual timestamp intervals remain explicit and positive.

Rational Taylor enclosures bind coefficient constants to saved binary angle
and timestamp inputs. A 1e-14-rad arithmetic conversion guard and a 1e-14
half-angle endpoint guard are conservative necessary restrictions only.
Original candidate motion 1e-7, position 1e-8, image 1e-6 and root/orientation
1e-9 tolerances remain unchanged. These equations do not validate upstream
camera, source, trig or measured robot/contact truth.

Three serial guarded CPU jobs pass twelve new analytic contracts. Independent
checks validate 272 coefficient enclosures, 22 angle-branch vertex domains
and 702 direct angle/timestamp vectors, retaining 348 intentional rejections.
Phase closure is below 2.50e-16. Every test proposal and intermediate return
is durably saved. Test vectors are not source/URDF/native/path candidates.

This is a verified representation for a future coupled construction, not a
source/time feasible path or exclusion. No native position, physics, GPU/PPO,
new qualified reference or score gain occurred. The three verified components,
separate unqualified middle component, source/missing/held-out evidence and
all original parameters/gates remain unchanged. Exact code/guards/plans are
archived under `experiments/scored_baseline_20261019/index_temporal_algebra`.

## Correlated filled-source/time domain clipping

The next study uses the filled source-disk sublevels and shared temporal
polynomials across 107–111, leaving all ten joint variables continuous. A
Bernstein lower grid gives a convex-hull enclosure of each coordinate and
polynomial value. Projecting below zero gives a necessary coordinate interval;
shared factors propagate those bounds to neighboring frames. Disconnected
source components and interiors remain covered. No isolated source witness is
fixed as a mandatory node and no outer center is accepted as a pose.

Five old source grids are reused. Four new exact tensor interpolation defect
bounds certify their lower representations; the frame-109 coefficient error is
reused by hash from its saved audit. The 44 new temporal lower grids come from
the already verified half-angle equations. Original candidate gates remain
unchanged. Exact rational arithmetic binds saved binary coefficient models;
upstream trig/camera/source uncertainty and old numerical endpoint scope remain.

The declared 24-pass contraction has no splits and stops UNRESOLVED, retaining
all 1,176 factor steps. Some normalized half-angle widths fall to 14.7–34.0%;
others remain 49.6–95.9%. Intervals are necessary restrictions, not source/time
feasible sets, accepted seeds, an exclusion or a new reference. Do not extend
or repeat this pass budget. Three unknown frame-109 source-event bands remain
within the filled-domain evidence and are not relabeled as known branches.

Eight new analytic controls pass. An independent checker uses blossom formulas
and scalar supporting-line certificates instead of repeating the contractor.
It verifies 397 input lower coefficients, 9,528 restriction coefficients, all
1,176 steps, 5,280 supporting lines and 100 exact interpolation defects. All
returns and rejected/unresolved domains are durably retained. Supporting-line
multipliers are algebraic certificates, not contact forces or importance scores.

Three guarded CPU jobs and exact sources/plans are archived code-only under
`experiments/scored_baseline_20261019/source_time_clipping`. Production runtime
and its tests are unchanged. No native position, physics, GPU/PPO, command/root
mutation, qualified path or score gain occurred. Current verified components,
the separate unqualified middle component, all source/missing/held-out evidence,
model parameters and original gates remain unchanged. The next method must
preserve source/time correlations that interval projections leave unresolved;
these interval hulls cannot establish a constructive common path.

## Common-marginal source/time model and realizability failure

A new degree-four Bernstein model couples proximal/distal and neighboring-clock
tables through common variable/pair marginals. Every deterministic assignment
embeds, including filled source interiors; conditioned square expectations must
be nonnegative. Eight analytic controls cover embedding, elevation, conditioned
positivity, finite-box duals and an odd-cycle false positive. Consistent local
marginals can still lack any globally realizable pose or common measure.

The first new bounded phase-I LP has 1,125 entries, 549 equalities and 5,359
inequalities. It returns objective zero, with no positive exclusion margin.
Independent checks verify 2,425 transferred/elevated coefficients and every
model row. Its saved table has 247 exact negative square expectations, including
six negative variances. These reject that table, not the source/time family.

A single corrective model adds all 247 independently proven inequalities in
advance, yielding 5,606 rows. Source/model/physical gates and solver tolerances
stay unchanged. Its phase-I objective is also zero. The strict audit passes
four new PSD controls and verifies all changed rows, then FAILS on two actual
negative solver weights: maximum bound violation 3.61446e-13. It remains failed.
No normalization, clipping or convergence/tolerance repair hides this result.

A new saved-return audit reuses only the completed controls and changed-row
checks by hash. It performs no LP or old control rerun; the unsaved prior dual
check is not claimed independently verified. All actual weights remain intact.
Exact equality and inequality residuals reach 6.030e-13 and 4.217e-13.

For conditioned moments y0..y4, a real measure on [0,1] requires the Hankel
matrix `[y(i+j)](i,j=0..2)` and localizer
`[[y1-y2,y2-y3],[y2-y3,y3-y4]]` to be positive semidefinite. Of 590 groups,
329 fail this necessity; 642 matrices have negative exact principal minors.
All 642 yield rational polynomial-square witnesses with negative expectation.
For 640, negativity survives a bound on the contribution of the two offending
weights alone. This diagnostic does not modify actual entries or establish
that bound correction would produce a normalized common measure.

Finite square-anchor constraints are insufficient. The next method must impose
continuous moment positivity before seeking a realizable source/time candidate.
Neither relaxed solution, its barycenter nor a matrix eigenvector is a robot
pose, observed source truth or a mandatory trajectory node. The inherited
fixed wrist/region-2 family remains unresolved; no global/task exclusion follows.

Six guarded jobs, including the strict audit failure, are archived code-only in
`experiments/scored_baseline_20261019/common_moment_viability` (16 exact sources/
plans). Twelve new analytic controls pass their stated scope, with the failed
process and both invalid tables retained. Production runtime/tests and the
verified reference/middle component/source authority/parameters/gates remain
unchanged. No native position, physics, GPU/PPO or qualified score gain occurred.

## Bounded continuous moment-cone separation

Ten new analytical contracts derive unnormalized conditional moment maps,
Hankel/localizer polynomial-square identities, signed/zero mass handling,
exact scale-invariant separation and a new tiny conic phase-I dual control.
The method keeps actual weights intact. Polynomial coefficient vectors are
neither robot poses nor measured source orientation/contact/force evidence.

One declared conic outer-approximation uses the saved initial LP return rather
than replaying that solve. Limits are fixed at six updates, 64 unique planes
per update, seven cone audits and 20 seconds per subproblem. The same source,
marginal and original authority constraints remain, with at most 5,990 rows.
Each plane has a rational nonnegative-square identity and a strictly negative
expectation at the current saved table. Selection uses exact normalized
mathematical violation and hash order; no physical outcome selects source data.

All six LP subproblems return phase-I objective zero. Six independently checked
finite-box dual gaps are zero, so they establish no exclusion. The declared
budget ends UNRESOLVED. The last table has 698 matrices failing exact PSD checks;
no table, barycenter or matrix eigenvector is promoted to a physical candidate.
No budget increase, added iteration or old-solver/tolerance repair is queued.

The independent checker validates 4,130 conditioned groups, 8,260 matrices,
41,300 principal-minor certificates, 5,254 rational square planes, all 384
selected planes, all six exact top-64 selections and six duals. The initial
saved table is rechecked under the new maps; it is not new source/physical
progress. The last table has no unit-box violations, but exact binary equality
and inequality residuals are 1.553e-14 and 1.895e-8. These normalized LP diagnostics
do not alter any original image, root, finger, motion or native gate.

After durably saving all completed certificates, the independent report writer
FAILS on a duplicate schema keyword. Its source, guard and log remain failed.
A separate report-only producer consumes the saved certificates by hash; no
numerical test, eigenproblem, LP or conic algorithm is rerun. Four serial guarded
jobs (three successful processes and that retained writer failure), 12 exact
sources/plans and all private evidence are archived code-only under
`experiments/scored_baseline_20261019/continuous_moment_cones`.

The conditional fixed-wrist/region-2 source-time family remains unresolved.
Production runtime/tests, the verified reference, separate middle component,
source authority, missing/held-out evidence and all original gates/parameters
are unchanged. No native position, physics, GPU/PPO, qualified trajectory or
score gain occurred. A direct semidefinite formulation is a distinct next
representation to assess; do not extend or repeat this six-update algorithm.

## Direct conditional moment SDP, 2026-10-19

A distinct direct semidefinite model now binds all saved conditional Hankel and
localizer constraints simultaneously, rather than extending the completed
six-update plane algorithm. It reuses the final saved 1,125 table entries, 549
shared equalities and 5,990 necessary inequalities. Exact original unit boxes
and 1,180 PSD blocks produce 14,099 canonical rows. Column-major svec uses
sqrt(2) off-diagonal scaling and the matching Frobenius adjoint, following the
[official solver specification](https://clarabel.org/stable/api_cone_types/).
Rational sqrt(2) brackets and finite-box coefficient bounds provide outward
linear RHS and matrix-diagonal guards. These are necessary model arithmetic
bounds, never changes to source/robot acceptance tolerances.

The private Python has no pip; its first installation process fails and remains
failed. Already SHA-verified binary wheels are then independently RECORD-checked
and extracted into a private target: Clarabel 0.11.1, cffi 2.1.1, pycparser 3.11.
Global/production environments and cached assets remain unchanged. The first
contract process fails on a map/list fixture after four completed contracts.
V2 reuses those four results by hash and completes six others without replaying
them. All ten new contracts pass, including two tiny analytical SDP controls,
packing adjoints, original equality/box signs, outward errors and exact conic
finite-box dual signs. Both process failures are retained.

One declared episode feasibility solve, with 100-iteration/60-second/one-thread
limits and default solver tolerances, returns Solved in ten iterations, about
0.138 seconds. No restart, continuation or numerical tolerance repair follows.
All matrices, maps, errors, proposals and actual primal/dual returns are saved
before assertions or further work. Six serial guarded processes (four successful,
two retained failures), 17 exact source/plan files and private receipts are
archived code-only in `experiments/scored_baseline_20261019/direct_moment_SDP`.

Independent saved-evidence checks validate all 14,099 rows, 25,690 linear and
26,550 PSD map coefficients, and all 5,900 exact principal minors. Every one of
the 1,180 returned original moment matrices passes exact PSD; actual weights
have no unit-box failure and original inequalities have no positive residual.
The exact shared-equality residual is still 2.467644e-16, so this is not an exact
realization of every frozen rational constraint. Actual weights are never
clipped, normalized or repaired. A separate rational-Gram dual construction,
with all 1,180 Gram identities independently checked, gives residual-corrected
finite-box gap -3.120771e-8: no exclusion certificate. Numerical Solved is not a
formal feasible common measure or a deterministic trajectory.

The inherited conditional region-2/fixed-unqualified-wrist source-time family
remains unresolved. A useful next distinction is positivity of squares involving
multiple joint variables within the existing degree-four tensor tables; passing
one-dimensional conditional PSD does not establish this joint positivity or
global consistency. No new solver call or degree sweep is queued from this
result. F and the separate unqualified middle component, all source/missing/
held-out evidence, original motion/physical gates and parameters are unchanged.
No native position, physics, GPU/PPO, full reference, grasp, motor, task or score
success occurred. Production runtime/tests are unchanged.

## Mixed joint-variable consistency, 2026-10-19

A new saved-return diagnosis tests polynomial squares involving multiple joints
and clocks. The existing degree-four tensor tables contain thirteen pair scopes
and six triples. Total-degree-two joint moment bases and total-degree-one
interval localizers t_i(1-t_i) require only powers already available. No source
polynomial, moment degree, anchor, optimizer or original gate changes. The
necessity follows the [moment/localizer identities in Laurent, Section 4.1](https://homepages.cwi.nl/~monique/files/moment-ima-update.pdf).

Nine new analytical contracts cover exact tensor-to-cross-moment maps, signed/
zero mass, mixed square/localizer identities, boundary localizers, positive/
rank-deficient/negative/zero-cross pivots, axis order, residual bounds and strict
scope checks. Three additional contracts bind projected moments to marginal
equalities. All twelve pass. Exact rational congruence replaces numerical
eigenvalue thresholds; any negative square would be retained explicitly.

All 63 new matrices pass exact PSD and are full rank: thirteen moment matrices
have rank six, six have rank ten; 26 interval localizers have rank three and
18 have rank four. Independent falling-factorial evaluation verifies all
405 monomial maps, 14,818 coefficients and 1,590 congruence entry identities.
This is additional local mixed positivity, not a qualified common measure or
deterministic source/time path. No solver runs and no old matrix audit repeats.

A genuinely new overlap calculation derives exact linear-combination
certificates from the ORIGINAL shared-marginal equations. The earlier exact
equality residuals are reused by hash rather than recomputed. Independent
checks pass all 405 projection identities and 226 shared-moment comparisons.
Of these comparisons, 223 have nonzero exact differences, maximum
2.2941718e-16. Every difference equals and is bounded by its saved equality
residual combination. These are arithmetic diagnostics, not new acceptance
tolerances, missing source or a permission to repair the returned table.
Actual weights are never clipped, normalized or changed.

All ten joint variables have strictly positive unnormalized variance
numerators. Thus the saved table is not a single deterministic assignment;
local PSD and small residuals do not produce commands or establish source
truth. Pointwise support on all source/time constraints, global common-measure
consistency and a concrete whole-clock candidate remain unqualified. A future
necessary study should first determine which temporal/depth support localizers
fit the EXISTING moment powers; unavailable moments must remain unresolved,
not inferred or supplied by a degree sweep. No new solve is queued from this
result before that derivation and meaningful new contracts.

Four serial guarded CPU jobs succeed; all twelve exact sources/plans and
private evidence are retained in the code-only `mixed_joint_consistency`
archive. Production runtime/tests, F, separate unqualified components, source/
missing/held-out flags, original bounds/gates and physical parameters remain
unchanged. No native position, physics, GPU/PPO, new reference, grasp, motor,
task, policy or score success occurred.

## Pointwise motion/depth support, 2026-10-19

A genuinely new necessity now tests whether the saved table could describe a
measure supported on each original motion condition. Scalar expected gates
and joint PSD alone do not establish this support. For each SAVED temporal
polynomial g>=0, exact saved coefficient intervals and affine domains produce
an upper polynomial h with h>=g throughout that domain. Thus h*q^2 is
nonnegative on valid support. The [moment/localizer identity](https://homepages.cwi.nl/~monique/files/moment-ima-update.pdf)
binds these weighted-square expectations to PSD matrices. No temporal/source
polynomial, trigonometric enclosure or old algebra job is regenerated.

One declared 96-bit dyadic representation keeps a rational uniform upward
error bound. Tiny nonzero coefficients retain their structural degree even if
a rounded center vanishes. These necessary arithmetic guards never alter
original robot/source acceptance tolerances. A fixed maximal multilinear
multiplier basis uses only EXISTING per-coordinate degree-four tensor powers.
The temporal pair/triple matrices have dimensions four/eight; no moment degree,
source box, motion cap, wrist mode or candidate authority changes.

Eight new analytical contracts pass, including a positive-mean measure whose
weighted square violates pointwise support, supported positive measures,
signed/zero mass, affine order, tiny structural terms, degree availability and
separate depth. The fixed saved-return diagnosis includes 44 temporal
boundaries and five saved binary depth models. All five source-image factors
retain unavailable nonconstant polynomial support localizers explicitly;
required powers exceed the existing tensor. Their old scalar tests are not
rerun. Finite-chart denominators, minimum depth, unknown zero-distal/event
cases, region-2 correlations and all source/missing/held-out flags remain.

Exactly 25 temporal support matrices fail, with a nonnegative scalar mean in
every case. The five depth and nineteen other temporal matrices pass. Independent
checks validate 317 coefficient/error bindings, 49 affine maps, 657 shift maps,
26,064 coefficients, 1,360 entry identities, 384 positive congruence entries
and all 25 exact negative weighted-square certificates. Independent affine
composition uses repeated polynomial products and moment maps use falling
factorials. No floating PSD threshold, returned-weight repair or old audit runs.

This rejects THAT saved numerical table as a measure supported on valid motion
states. It does not exclude the source/time family, establish a compulsory root
change or qualify a concrete trajectory. Witness coefficients and changes in
moment weights are algebraic quantities, not joint commands or forces. Exact
coefficient statements concern frozen numerical models; upstream camera/source
accuracy and old endpoint uncertainty remain unqualified.

Three serial guarded CPU jobs succeed; eleven exact sources/plans and all
private states, matrices, error bounds, failed/unavailable cases and witnesses
are retained in the code-only `pointwise_support` archive. No source/physical
candidate, native position, physics, GPU/PPO, motor/grasp/task/score success or
production runtime/test change occurs. F and separate unqualified components
stay unchanged. A distinct future support-aware conic formulation requires new
packing/dual contracts and a declared single bounded evaluation, not another
anchor/cut loop, restart or extension of earlier solver budgets.

## Complete support-aware conic formulation, 2026-10-19

A new formulation appends the complete saved 63 joint and 49 motion/depth
support maps to the original canonical conic base. The original 14,099 rows
are reused by hash, not regenerated or solved as a preliminary subproblem.
The 112 new cones add 1,741 rows: total 15,840 rows, 1,125 original columns and
1,292 PSD cones. All 549 shared equalities, 5,990 inequalities, explicit unit
boxes and 1,180 conditional cones remain. No finite-anchor loop, new moment
power, source polynomial, wrist mode or physical gate is introduced.

The first new synthetic contract fails a dictionary roundtrip comparison
before packing or a solver call: exact-zero entries are represented differently.
Its actual roundtrip objects were not saved and are not reconstructed. The
failed source/log/guard remain. V2 uses a canonical sparse fixture and durably
journals its new returns before assertions. Ten NEW analytical contracts pass
weighted packing/adjoints/error bounds in dimensions 4/6/8/10, exact rational
Gram certificates and conic dual signs, including two new tiny SDP controls.
Old dimension-three contracts are referenced by hash without reexecution.

Independent pre-solve checks bind all 112 new cones, 1,741 new canonical rows
and 67,123 upper-entry coefficients. Necessary outward spectral/packing guards
are explicit and do not change original candidate tolerances. The reused base
prefix and its box/equality/inequality bindings are unchanged. All model maps,
constructors, solver returns and exact certificate data are durable.

ONE new bounded evaluation (100 iterations, 60 seconds, one thread, default
solver tolerances) returns **AlmostSolved**, reduced accuracy, at 14 iterations
and about 0.432 seconds. There is no restart, tolerance repair or continuation.
The independent NEW-return audit finds all 1,292 original rational matrices PSD,
with zero actual unit-box or original-inequality failures, but shared equalities
still have nonzero exact residual 3.215744e-16. No actual weight or dual is
clipped, normalized or repaired. This is not an exact supported common measure
or deterministic robot trajectory, and the solver status remains AlmostSolved.

All 1,292 separate rational-Gram dual constructions and 10,620 entry identities
verify independently. The exact residual-corrected finite-box gap is negative,
-6.385598e-7; there is no exclusion certificate. This does not prove infeasibility,
stationarity, a mandatory root change, source accuracy or task success. The
conditional frozen region-2/fixed-wrist family remains unresolved. Five scalar
source-image factors still require unavailable moments for nonconstant scalar
localizers; no missing powers are fabricated.

Six serial guarded CPU jobs (five successful processes and the retained first
contract failure), eighteen exact sources/plans and all private evidence are
preserved in `experiments/scored_baseline_20261019/supported_conic_model`.
Production runtime/tests, F, separate unqualified components, source/missing/
held-out evidence, original bounds, parameters and physical gates remain.
No native position, physics, GPU/PPO, motor/grasp/task/policy or score success
occurs. A useful new direction is to derive a degree-two matrix source-cone
representation from SAVED camera numerators, with an independent necessity/error
proof against the SAVED scalar source/depth models before any diagnosis. This
would be a distinct representation; the old unavailable scalar localizers and
all earlier failures must remain, without a degree increase or old solver retry.

## Matrix-valued image support within the existing tensor, 2026-10-19

A genuinely new representation addresses the remaining source-image support
necessity without adding moments. For positive depth, the original camera cone
has arrow matrix `[[t,x,y],[x,t,0],[y,0,t]]`, PSD exactly when
`t >= sqrt(x*x+y*y)`. Its entries use SAVED degree-(2,2) camera numerators.
A four-element multilinear polynomial vector gives a 12x12 weighted matrix
localizer within EXISTING per-coordinate degree-four tensors. The earlier
nonconstant SCALAR quartic localizers remain unavailable; no powers are invented.

Saved floating numerator and normalized scalar source/depth models are not
assumed identical. Exact coefficient discrepancy bounds and the saved positive
depth gate imply explicit necessary diagonal guards, 2.043574e-12 to 7.465079e-12.
The original cap, 1e-6px image tolerance and .01m depth model are unchanged.
These guards bind saved numerical coefficient models, not upstream trig/camera
accuracy, measured human depth or accepted robot candidates. Source-model/root
uncertainty, all region-2 correlations and three unknown 109 event bands remain.

Twelve NEW analytical controls pass cone/PSD signs, positive/zero/signed cases,
weighted-support necessity, existing-power capacity, tiny terms, axis order and
error guards. A single NEW diagnosis reuses the SAVED support-aware AlmostSolved
return without another solver or the old 1,292-matrix audit. Three new source
support matrices fail at 107/109/110; 108/111 are exact PSD at rank 12.
The earlier positivity passes and nonzero shared-equality residual remain.
Actual weights and duals are never clipped, normalized, repaired or promoted.

Independent exact checks verify 30 discrepancy/depth/guard identities, 170
uniform-error terms, 15 affine polynomials, 135 falling-factorial shift maps,
2,160 coefficients, 720 matrix entries, 288 positive congruence entries and
all three negative rational weighted-square witnesses. These reject THAT saved
numerical table as a source-supported measure. They do not exclude the source/
time family, establish a required wrist change, qualify a deterministic path,
source accuracy, native mechanics, motor/grasp/task or a policy/score gain.

Three serial guarded CPU jobs and twelve exact source/plan files are retained
in `experiments/scored_baseline_20261019/matrix_image_support`. All private maps,
matrices, returns, errors and witnesses are durable. Production runtime/tests,
verified F, separate middle/wrist components, missing/held-out observations and
all original parameters/gates stay unchanged. No native position, physics or
GPU/PPO runs. A future direct formulation may reuse the saved five new maps
only after new dimension-12 packing/adjoint/dual contracts and a declared single
bounded solve; it must not retry/tune the completed support-aware solve or repeat
these producers, controls or audits.

## Direct image-support-aware conic formulation, 2026-10-19

A genuinely new formulation appends all five SAVED 12x12 image-support maps
to the SAVED supported canonical base by hash, with its 15,840-row prefix
unchanged. Actual constructor counts are 16,230 rows, 1,125 original columns
and 1,297 PSD cones. All original 549 equalities, 5,990 inequalities, 2,250 unit
box rows, 1,180 conditional, 63 joint and 49 motion/depth cones remain. Existing
necessary source-coefficient guards are reused by hash. Five nonconstant SCALAR
image localizers remain unavailable; there is no degree/source/wrist/gate change.

Eight NEW dimension-12 analytical packing, spectral-error, Frobenius-adjoint,
signed/zero and exact Gram/finite-box dual controls pass, including two new tiny
SDPs. Old dimension 3/4/6/8/10 controls are not rerun. Independent pre-solve checks
verify the five saved maps, all 390 new rows, 4,960 exact/packed coefficients,
necessary packing guards and unmodified base prefix. All intermediate returns,
actual constructor/solver outputs and certificates are durable before further work.

ONE new declared evaluation at 100 iterations/60 seconds/one thread/default
tolerances returns AlmostSolved at reduced accuracy, 15 iterations and about
0.511564 seconds. There is no restart, warmstart, continuation or tolerance
repair. Independent NEW-return checks find all 1,297 original rational matrices
PSD, with no actual unit-box or original-inequality failures. Exact shared
equality residual is still nonzero, 3.053113e-16. Actual weights and duals are
not clipped, normalized, repaired or promoted to a pose.

All 1,297 rational-Gram certificate matrices and 11,340 entry identities verify
independently. The residual-corrected original-map finite-box gap is negative,
-1.459083e-7; no exclusion certificate follows. This is not an exact common
supported measure, deterministic source/time path, source accuracy or physical
candidate. The frozen coefficient/region-2/fixed-wrist family remains unresolved.
Old invalid tables, failed/reduced statuses, unknown event bands and numerical
endpoint qualifications remain. Global compatibility across local factor scopes
is unassessed and is a useful distinct question before another solver.

Five serial guarded CPU jobs and seventeen exact code/plan files are in
`experiments/scored_baseline_20261019/image_support_conic`; private arrays,
coefficients, returns and certificates stay private. Production runtime/tests,
verified F, separate unqualified components, original source/missing/held-out
observations, parameters and gates are unchanged. No native position, physics,
GPU/PPO, motor/grasp/task/policy/score gain occurs. Do not repeat or extend this
model, solver, controls or audits, or promote local means/eigenvectors/Gram
factors to commands or mandatory measured source joints.

## Global factor-scope and four-variable cycle necessities, 2026-10-19

A new study addresses compatibility ACROSS local factors without rerunning
positivity within old scopes. The 29 saved scopes have 11 maximal scopes.
Their maximum intersection-weight tree has weight 14 against the running-
intersection target 18. Independent tree cycle-property checks verify its
optimality, so these maximal scopes admit no standard junction tree. This is
not an exclusion of a global common measure; automatic clique gluing is absent.
The partial global first/second-moment Gram matrix has 40 known upper entries
and 26 absent cross moments, retained free rather than filled with products
of means. All moments are normalized tensor coordinates, not robot commands.

Eleven NEW analytical contracts pass continuous/even four-cycle Schur bounds,
zero/signed mass, fixed 96-bit outward square roots, tiny overlap residuals,
running intersection and product availability. The old odd-cycle control,
source polynomials, solvers and mixed-matrix/projection audits are not rerun.
A single SAVED newest-return diagnosis maps 39 moment groups and 139 copies
to saved marginal residuals by hash. There are 100 nonzero copy differences,
maximum 1.162265e-16; 27 of 29 actual masses differ from one. Actual weights,
masses and residuals are unchanged, never normalized or repaired.

Diagnostic moment intervals follow exact residual-map bounds and contain
reported copies. They are not all-correction/family-feasibility bounds or new
candidate tolerances. All seven induced four-variable cycle envelopes pass
necessary missing-diagonal Schur interval overlap tests. Both missing diagonals
remain free; no convenient covariance completion or mean becomes source truth.
Passing these seven checks does not establish full global PSD completion,
compatibility on longer cycles, a common supported measure or a trajectory.

Independent checks verify 55 maximum-tree cycle properties, 29 mass bindings,
139 copy identities, 5,778 map coefficients, the generic exact symbolic Schur
identity, 14 triangle certificates and 14 rational square-root enclosures.
Three serial guarded CPU jobs, ten exact sources/plans and every private map,
proposal, intermediate return, unavailable entry and certificate are retained
in `experiments/scored_baseline_20261019/global_cycle_consistency`.

Production runtime/tests, verified F, separate unqualified components, original
source/missing/held-out authority, region-2 correlations, unknown event bands,
parameters and physical gates remain. No new solver, native position, physics,
GPU/PPO, motor/grasp/task/policy or score qualification occurs. A future global
Gram-completion diagnostic requires separate necessity/packing/dual contracts
and a single declared bounded evaluation. Missing entries may be existential
auxiliary variables only, not new measured moments or a change to the original
source/time model. No repeats, budget extensions, means/eigenvectors/Gram
factors promoted to commands, or repairs to raw solver weights are authorized.

## Exact separate global first/second Gram completion, 2026-10-19

A genuinely new, SEPARATE diagnostic uses 66 auxiliary upper-triangle entries
for the uncentered 11x11 Gram matrix of `(1,t_1,...,t_10)`. The saved 40 known
entries include original unit mass one and 39 frozen copy/residual intervals;
26 cross entries remain existential unknowns. Necessary normalized-coordinate
support/Fréchet bounds yield 1 equality, 233 inequalities, 132 unit-box rows
and 66 PSD rows, 432 canonical rows total. This does not alter the original
1,125 source/time columns, moment authority, raw weights or human observations.

Ten NEW dimension-11 partial-Gram, support, packing/adjoint/error, interval,
zero/signed and dual-sign contracts pass, including two tiny new analytical
SDPs. Old packing controls are referenced by hash. Independent constructor,
frozen-interval, free-entry, support and packing bindings pass before ONE new
100-iteration/60-second/one-thread/default-tolerance completion evaluation.
It returns Solved at 7 iterations and about 0.003445 seconds. No retry,
warmstart, tolerance repair, continuation or alternative proposal occurs.

The actual auxiliary return is approximate: 40 scalar constraints fail exactly,
with equality residual 2.220446e-15 and maximum inequality violation 8.263608e-11.
Its actual Gram matrix is PSD/rank 11, with no box failure. Neither these values
nor the original 1,125 returned weights are clipped, normalized or repaired.
The old AlmostSolved status and all original mass/overlap residuals remain.

A SINGLE predeclared SEPARATE certificate uses the saved rational known anchors,
original unit mass and the sole return's 26 free entries. Its 40 specified
entries differ from the actual auxiliary return; it is an existential diagnostic
matrix, not a repaired return or original moment table. Independent exact checks
verify all 39 interval bindings, 26 free-entry provenance, 1 equality, 233
inequalities, 66 boxes and 121 PSD-congruence identities. The separate matrix
is exact PSD/rank 11 and completes the diagnostic first/second Gram envelope.

Independent 121 rational-Gram dual products and the original-diagnostic finite-
box gap -9.674558e-9 verify; there is no exclusion. Generic helper scope text is
not claimed: this certificate concerns only the separate copy/residual envelope,
not source/time family infeasibility. First/second Gram completion is only a
necessary common-measure condition. A common distribution with joint support,
deterministic source/time trajectory, source accuracy or physical candidate
remains unqualified. Existential cross moments never become observations.

Five serial guarded CPU jobs and seventeen exact sources/plans are retained in
`experiments/scored_baseline_20261019/global_Gram_completion`, with private models,
actual returns, proposed matrices, errors and certificates durable before further
work. Production runtime/tests, F, separate components, source/missing/held-out
flags, region-2 correlations/unknown bands, parameters and original gates remain.
No native position, physics, GPU/PPO, motor/grasp/task/policy/score gain occurs.
Do not rerun/tune these producers, solvers, controls or audits, or construct
alternative proposals. Joint-support necessities of the separate completion
remain a distinct future question; no mean/eigenvector/Gram factor is a command.

## Complementary-corner joint unit-cube support, 2026-10-19

A new exact support necessity uses three normalized tensor variables `(x,y,z)`.
For each complementary pair of Boolean corners, adding their multilinear
Bernstein products cancels the cubic term. The resulting four quadratics are
`1-x-y-z+xy+xz+yz`, `z-xz-yz+xy`, `y-xy-yz+xz` and
`x-xy-xz+yz`. Each is nonnegative throughout the unit cube; their sum is one.
Their expectations use only mass, first and cross moments. No absent moments,
products of means, normalized weights or repaired matrix entries are introduced.
The Boolean corners are algebraic proof controls, not source/robot anchors.

Ten NEW analytic contracts pass exact product/cancellation/partition identities,
a correlated continuous measure, unnormalized positive/zero/signed mass,
permutations, exact tiny negative signs, Frobenius maps and explicit absent
moment handling. Old odd/even-cycle, Schur, PSD, mixed-projection and source
controls are not replayed. Only after these contracts, ONE fixed diagnosis
checks all 120 triples and four polynomials each against the SAVED separate
exact first/second Gram completion. All 480 expectations pass exactly, with
minimum 0.15996792242118518, and all 120 sums equal the original unit mass.
The 26 completed cross entries remain existential values of this ONE matrix.

Independent Boolean-lattice interpolation verifies 3,840 coefficient identities,
3,840 corner/nonnegativity identities, 2,280 expectation-map coefficients,
480 direct rational expectation formulas, 120 partitions and 480 durable
BEFORE/AFTER journal pairs. All proposed polynomials, maps, intermediate returns,
results and empty negative/unresolved sets are private and hash bound.
Three serial CPU guards succeed within 240 seconds/2 GiB/1.5 GiB reserve;
ten exact sources/plans are archived under `cube_joint_support`.

This passes a NECESSARY triple support test only. A positive common measure on
all ten cube variables, support on the original source/time inequalities,
deterministic robot trajectory and physical candidate remain unqualified.
No new completion, solver, anchor/cut loop, old scalar/PSD/marginal audit,
source/time polynomial, native position, physics or GPU/PPO job runs. All actual
original/auxiliary returns and separate certificate entries remain unchanged.
The old AlmostSolved/equality/overlap residuals and failed statuses remain.
F, separate unqualified middle/wrist components, source/missing/held-out flags,
region-2/full-clock correlations/unknown109bands, parameters and original gates
are unchanged. Production runtime/tests and motor/grasp/task/policy/score stay
unchanged. Do not rerun this fixed diagnosis, controls or verification.

The synthetic continuous contract stores only the requested mass/first/distinct-cross moments; unused diagonal zeros are storage placeholders, not second moments or a PSD/measure claim. That control checks only the new paired-corner formulas. Exact executed code/returns remain retained unchanged.

## Fixed continuous product-Beta density attempt, 2026-10-19

A distinct constructive question tests ONE explicit continuous distribution for
the SAME separate exact first/second Gram completion. For saved unit mass,
`mu_i=M0i`, `v_i=Mii-mu_i^2` and `c_ij=Mij-mu_i*mu_j`, the fixed inverse-Beta
parameters use `kappa_i=mu_i*(1-mu_i)/v_i-1`, `a_i=mu_i*kappa_i` and
`b_i=(1-mu_i)*kappa_i`. All ten satisfy the strict positive interior conditions.
Relative to their product law, the proposed ratio is
`D(t)=1+sum c_ij*(t_i-mu_i)*(t_j-mu_j)/(v_i*v_j)`.
Orthogonality fixes its normalization and all requested first/second moments.
This does not normalize, repair or alter any original/auxiliary/certificate entry.
The 26 completed cross entries remain existential values of that ONE matrix.

Twelve NEW analytic controls pass Beta integral recurrences, inverse parameters,
positive correlated continuous examples, normalization/orthogonality/moment maps,
multiaffine interpolation, missing/zero/signed/boundary handling, exact tiny
negative signs and axis order. Old cube/cycle/Schur/PSD/mixed/source controls and
solvers are not repeated. Independent pre-evaluation bindings verify all ten
parameters, 45 fixed pair corrections and 56 multiaffine coefficients.

ONE bounded fixed evaluation of all 1,024 algebraic cube corners finds 524
strictly negative values and 500 nonnegative values. Minimum ratio is about
-12.687057659 and maximum 27.729057868. These corners are proof evaluations,
not robot/source anchors or a search over poses. Continuity carries a negative
corner into a negative interior neighborhood, where the positive Beta base
law has positive density. THIS density ansatz is REJECTED. All 66 formal
normalization/first/second/cross identities still match exactly, demonstrating
why moment closure alone cannot certify a positive measure. No coefficient
adjustment, retry, alternative law, optimizer, solver or completion follows.

Independent verification uses factored corner formulas and rising-product Beta
moments rather than the producer's expanded polynomial/recurrence. All ten
parameter bindings, 45 pair bindings, 56 coefficients, 1,024 corner signs,
40 finite Beta moments, 66 centered-factor moment identities and 1,090 ordered
durable BEFORE/AFTER pairs pass. Four serial guarded CPU jobs succeed; twelve
exact code/plan files are archived under `product_Beta_density`. Parameters,
actual returns, maps, signs, rejected/unresolved cases and certificates stay
private and hash bound. Failed construction is a scientific result, not a
failed worker or a proof that the matrix lacks any supported common measure.

The rejection applies ONLY to this declared density. Other measures/completions,
the copy/residual envelope, original source/time family and task remain
unresolved. Even a positive law for this diagnostic first/second matrix would
not realize the original 1,125 higher tensor weights or source/time support.
No measured source truth, deterministic path or physical candidate is qualified.
All actual original/auxiliary returns, separate exact certificate, F/separate
middle/wrist components, source/missing/held-out flags, full region-2 correlations
and unknown bands, parameters and original gates remain unchanged. Production
runtime/tests, native/physics/GPU/PPO and motor/grasp/task/policy/score do not
change. Do not rerun this construction, controls, evaluation or verification,
or treat density coefficients/corners as robot commands or observations.

## Existing pair-product global moment inventory, 2026-10-19

A distinct higher-order consistency question uses the 13 existing pair tensor
scopes as observables `Y_s=t_i*t_j` in the SAME saved normalized coordinates.
Squares use powers `(2,2)` and overlapping pair products use `(1,2,1)`; these
powers already exist in the saved per-coordinate degree-four tables. Products
whose union is absent from existing pair/triple scopes remain free/unavailable.
The input is the newest ORIGINAL 1,125-weight image-conic AlmostSolved return,
not the separate exact Gram certificate or a replacement distribution.

Thirteen NEW analytic controls pass product/alias/box identities, a continuous
correlated example, exact tensor/falling-factorial maps, axis order, weighted
residual identities/envelopes, zero/signed mass/variance, domain aliases and
unavailable scopes. No old scope/cycle/PSD/mixed/source control or audit repeats.
Only after these controls and exact provenance/domain bindings, ONE fixed
inventory checks all 105 upper entries of partial `Gram(1,13Y)`, dimension 14.
There are 33 specified entries and 72 absent products, all retained FREE.
The 32 nonconstant groups have 56 actual scope copies and 2,113 coefficients.

Twenty-four copy differences are exactly nonzero, maximum 4.0515279933e-17.
Each equals and is bounded by a NEW exact linear combination of ORIGINAL
marginal-equality residual values. All 549 old residual values, saved link paths,
old raw mass envelope and domain provenance are reused by hash; their original
audit is not rerun. Actual raw weights, masses and differences stay unchanged.
Diagnostic envelopes contain reported copies only. They are not bounds on all
possible corrections/family feasibility or new candidate tolerances.

Independent falling-factorial maps verify all 105 availability decisions,
56 copy maps, 2,113 coefficients, original-row identities, exact residual
predictions/bounds, domain aliases and 105 durable entry BEFORE/AFTER pairs.
Every monomial, proposal, map, copy/path, intermediate return, unavailable product
and exact certificate is private and hash bound. Three serial CPU guards
succeed within 240 seconds/2 GiB/1.5 GiB reserve; eleven exact code/plan files are
archived under `pair_product_moments`.

No missing product is zeroed, replaced by means or supplied by a convenient
completion. No original or auxiliary returned value, separate exact certificate,
source observation or command/root changes. This inventory does not test global
PSD completion or establish a common supported measure, deterministic source/time
path, source truth, physical candidate or family/task exclusion. No new solver,
PSD audit, source polynomial, native position, physics or GPU/PPO runs. The old
AlmostSolved/equality/overlap statuses and failed controls remain. F/separate
middle/wrist components, source/missing/held-out authority, full region-2 and
unknown-band uncertainty, parameters, original gates and production runtime/tests
stay unchanged. Do not rerun this fixed inventory, controls or verification.

## Mixed original-variable / pair-observable aliases, 2026-10-19

A distinct mixed block links the original ten normalized tensor variables with
13 derived pair observables `Y_ij=t_i*t_j`. The 130 requests `E[t_a*Y_ij]`
need powers `(2,1)` inside the pair and `(1,1,1)` outside. Existing scopes/powers
and exact normalized-domain identities are required before evaluating any term.
Eleven NEW contracts pass pointwise multiplicative aliases, these new tensor
maps, continuous correlated moments, exact marginal rows, zero/signed mass,
zero-variance mixed identities, domain mismatch handling and pointer assembly.
The completed pair-power contracts are reused by hash, not repeated.

ONE fixed inventory of the original 1,125-weight image-conic AlmostSolved return
has 38 available mixed entries and 92 unavailable entries. The 130 requests
form 108 distinct monomials; 32 are available. Six duplicate available entries
share one exact monomial map. All 22 duplicate groups, including unknown groups,
remain explicit. A shared map is an algebraic identity, not an averaging or
repair of actual raw copies, new observation or accepted source state.

The 32 new available groups have 56 actual scope copies and 2,136 coefficients.
Twenty-four differences are exactly nonzero, maximum 3.9835394821e-17. Each equals
and is bounded by a NEW exact linear combination of ORIGINAL saved marginal
residual values. All 549 old values are reused by hash; no old residual audit
runs. Domains bind identically. Actual raw weights/masses/residuals are unchanged.
Copy envelopes cover reported values only, not all possible corrections/family
feasibility and not any new original candidate tolerance.

The SAVED old variable-11Gram and pair-14Gram inventories contribute 170 upper
entries BY HASH, with only the constant shared. No old numerical block is
reexecuted. The pointer-only partial `Gram(1,10t,13Y)`, dimension 24, has 300
upper entries: 110 specified and 190 free/unknown. All 26 old variable unknowns,
72 old pair unknowns and 92 new mixed unknowns remain free. The separate exact
completed 11Gram supplies no new known values or source truth here.

Independent verification checks all 130 mixed availability/alias decisions,
56 new copies, 2,136 falling-factorial coefficients, original-row identities,
exact residual predictions/bounds, domain bindings, 144 old pointer hashes and
130 ordered durable entry BEFORE/AFTER pairs. All requests/maps/paths/copies,
intermediate and constructor returns, unavailable entries and exact certificates
are private/hash bound. Three serial guarded CPU jobs succeed; twelve exact
code/plan files are archived under `mixed_product_aliases`.

This is an algebraic inventory, not a global PSD completion, common supported
measure, deterministic source/time trajectory or source accuracy certificate.
No family/task exclusion, new solver, PSD audit, source generation, alternative
distribution, command/root mutation, native position, physics or GPU/PPO occurs.
Original/auxiliary raw returns and separate exact certificate stay unchanged.
F/separate middle/wrist components, source/missing/held-out authority, region-2
and unknown-band uncertainty, parameters, original gates and production
runtime/tests remain. Do not rerun these contracts, inventory or verification,
or promote a product-moment pointer, mean or Gram factor to a robot command.

## Symbolic common monomial functional, 2026-10-19

A new representation groups entries of `Gram(1,10t,13Y)` by polynomial exponents
modulo the pointwise identities `Y_ij=t_i*t_j`. A common positive measure must
use ONE linear-functional value for the same monomial wherever it occurs.
Twelve NEW contracts pass quotient classes, shared symbols, original-column
maps modulo exact original rows, missing-row failure, positive/zero/signed/nonunit
mass bounds, deterministic and continuous embeddings, adjoints and namespaces.
No old numerical inventory, copy audit, packing control or source job repeats.

ONE symbolic construction groups 300 upper entries into 261 monomials. There
are 91 known classes covering 110 cells and 170 FREE existential classes covering
190 cells. Missing cells stay unobserved and have no assigned numerical value;
identical monomials share one symbol. The separate symbolic layout has 1,295
columns with the entire original 1,125-column prefix unchanged. The saved
canonical model, arrays and returned values are not mutated or repaired.

Known classes use existing original-column LINEAR maps rather than frozen
numeric anchors/means from a diagnostic interval. All 110 known aliases have
identical selected anchor coefficients. Their new differences are zero, requiring
no additional row combination; internal copy equivalences and independently
verified original-row certificates are reused by hash. Exact original unit mass
comes from a saved normalization row, not normalization of the approximate raw
table. Under a positive exact unit-mass cube measure, every nonnegative monomial
has expectation in [0,1]. Nonunit positive mass has its own upper bound, zero
positive mass is zero, and signed mass is not normalized or granted unit boxes.
The contract requirement flag records whether unit mass is met; mass two has
upper bound two, not one. Candidate tolerances are unchanged.

Independent checks verify all 261 classes, 300 entry-map identities, 110 known
coefficient bindings, 899 exact Frobenius-adjoint coefficients and 300 new
analytic positive-measure embedding entries. Synthetic tensor points and
Bernstein columns are algebraic proof controls, not episode poses, source
anchors, returned weights or physical candidates. All class proposals, maps,
constructors/intermediate returns, FREE declarations and empty unresolved cases
are durable/private/hash bound. Three serial CPU guards succeed; ten exact
code/plan files are archived under `global_monomial_functional`.

This establishes only a NECESSARY symbolic representation. It does not establish
PSD feasibility, a common measure supported on original source/time constraints,
source accuracy, a deterministic path, task/global exclusion or physical success.
The original 1,125 columns, 549 equalities, 5,990 necessary inequalities, unit
boxes, 1,297 cones and 16,230-row canonical prefix are retained by hash. Five
unavailable nonconstant SCALAR image localizers remain unavailable. No numerical
moment proposal, PSD audit/completion, packing, solver, source generation,
command/root mutation, native position, physics or GPU/PPO occurs. F/separate
components, missing/held-out authority, full region-2 and unknown-band uncertainty,
parameters, original gates and production runtime/tests stay unchanged. Do not
rerun these contracts/construction/verification. New dimension-24 packing/dual
contracts and one separately declared global-conic evaluation remain future work.

## Global shared-monomial conic consistency, 2026-10-19

A new direct conic formulation couples repeated monomials across the saved
original-variable and pair-product scopes. It preserves the entire original
16,230-row/1,125-column prefix byte for byte, adding 170 UNOBSERVED shared
auxiliary monomials, 340 necessary unit-box rows and one 24x24 Gram cone.
The actual model has 16,870 rows, 1,295 columns and 1,298 PSD cones. ORIGINAL
549 equalities/5,990 inequalities/2,250 boxes and 1,297 cones are unchanged.
Auxiliary upper bounds require a positive exact unit-mass cube measure, not
normalization of the approximate raw table. All source/packing guards remain
NECESSARY bounds; original candidate tolerances do not change.

Twelve new 24D packing/adjoint/spectral/auxiliary/pivot/dual controls include two
tiny analytic SDPs. A manual fixture's wrong SIGN COMMENT and limited hardcoded
residual check are retained in the executed source; three distinct new exact
row/sign/saved-certificate checks supply the missing derivation, including the
wrong-sign countercheck and 576 Gram identities. Correct alias beta is +1.
No completed control or solver repeats. Independent bindings check every new row,
the unchanged prefix and 2,646 exact/packed coefficients. The outward diagonal
guard is 1.2914127466105907e-14, not an original acceptance tolerance.

ONE bounded 100-iteration/60-second/one-thread/default-tolerance evaluation returns
Solved,15 iterations,0.614900835 solver seconds. No restart/warmstart/continuation
or degree/anchor/tolerance/profile sweep occurs. All actual returns are durable.
A NEW-return audit reader fails before numerical checking because row metadata
lives in a separate packing record. Its failed source/plan/log/guard stay intact.
V2 binds that saved SHA-verified metadata and completes the independent audit,
without producer or solver replay. Seven CPU guards: six succeed, one fails.
Twenty-one exact code/plan files are archived under `global_monomial_conic`;
private maps/arrays/weights/geometry/results remain off Git.

All 1,298 original rational matrices are exact PSD, including the global block.
All 1,295 actual values pass exact unit boxes; original inequalities have no
positive residual. However 534 of 549 equalities differ exactly, maximum
4.2327252813834093e-16. The actual table is NOT an exact realization. Independent
1,298 rational-Gram identities / 11,916 entry products and the residual-corrected
finite-box certificate over ALL original and auxiliary variables verify gap
-2.6418807394451264e-8: NO exclusion. No raw x/z/s value is clipped, normalized,
repaired or promoted. Gram factors remain certificate objects, not commands.

Passing global moment necessities does not establish a common measure supported
on original source/time constraints, source accuracy, deterministic trajectory,
native/motor/grasp/task/policy qualification or score gain. Five nonconstant
SCALAR image localizers stay unavailable. F, separate unqualified components,
all continuous107:111/region2/unknown109 bands, missing136/heldout31/32 failures,
source confidence, original boxes/caps/tolerances/model/parameters/gates remain.
Production runtime/tests are unchanged. No native position/physics/GPU/PPO or
reference/root mutation occurs. Do not repeat these completed jobs or extend the
budget. The owner's actual trajectory-qualification goal remains active.

## Exact real five-clock assignment and whole-clock rejection, 2026-10-19

A new exact-real QF_NRA construction encodes SAVED region2 domains,44saved temporal
coefficient bounds and five source/depth polynomials. Ten variables stay continuous.
The sufficient temporal lower polynomial is h-2G=P-G<=g for saved h=P+G and
|g-P|<=E<=G; this is an INNER numerical construction, not a changed candidate gate
or global exclusion model. All source/time polynomials and original authority
are reused.74constraints/482coefficients bind exact signs,scopes and domains.
Ten new encoding/RETURN contracts complete; a missing-SymPy failure is retained,
with five finished controls/four CLI returns reused by hash and NEW algebraic
model roots parsed using existing exact polynomial utilities. No old root chart,
producer, contract or audit repeats; no dependency installation occurs.

The first episode input has22unsupported ^ expressions; its exit1/errors/later
partial sat are REJECTED and preserved. A distinct actual CLI multivariate-power
contract verifies multiplication-only syntax.222new identities and independent
74row/482coefficient expansions prove the repaired query mathematically identical.
Two subprocess attempts are recorded: one invalid input and ONE complete query;
no completed valid solver result is retried or tolerances/degrees/limits swept.
The complete query returns sat in14.992872458seconds under180sec/one-thread bounds.
All10returned values are exact dyadic rationals, not moment means or fitted data.

Independent74exact query values,22ORIGINAL direct angle/timestamp gates and five
actualURDF/source-image poses at107:111 pass. Camera polynomial closure is
<1.67e-16m, a representation check only. ONE sparse whole-clock embedding is still
REJECTED: source failures103/104/105/106/116/117/118/119/120/121, added adjacent
.21679386324802863rad and second .24088276464938163rad versus ORIGINAL.05rad.
117available index anchors are assessed;34missing index anchors keep commands.
Fresh both-hand151clock actualURDF positions/components/velocities pass and left
motion passes. Five local poses are NOT mandatory measured joints, a full source
path, a replacement F or a native/self/swept/motor/grasp-qualified reference.
No full root/source/physical certificates are inherited by the changed proposal.

Nine guards retain eight successful workers/one missing-package process failure;
26exact code/plans and2repository snapshots are code-only in `exact_real_source_time`.
All numerical queries, raw returns, failures, source/geometry/arrays and rejected
proposals remain private. F/separate middle/wrist, missing136/heldout31of32 failure,
fullregion2/unknown109bands, source authority/model/parameters and ALL original
caps/tolerances/gates stay unchanged. No source polynomial generation,old completed
study replay,native position,physics,GPU/PPO,score or trajectory qualification.
Production runtime/tests unchanged. Do not repeat or extend these studies. A new
surrounding-clock construction must free these five values, retain all19saved
source frames and impose original full-clock temporal boundaries. Owner goal
remains ACTIVE and cannot complete for this research milestone.

## Surrounding-clock continuous source/time construction, 2026-10-19

A NEW19clock formulation releases all38index half-angle variables at103:121;
the five earlier local points are not fixed. All19source polynomial records and
filled image disks are reused, with strict separate .01m depth, inherited region2
correlations and3unknown109bands. F commands outside the band are fixed as a
restricted construction choice, not observations or global feasibility bounds.
All outside commands fit the inherited boxes. Root mode/source authority/model/
physics/parameters/gates are unchanged. Eight NEW boundary contracts pass affected
incidence, exact fixed-term folding, second-difference signs, future joins,
actual timestamp/tolerance units, invalid clocks, shifted phase and poles.

122affected original temporal requests contain22old local requests/44polynomials
reused by hash and100NEWsurrounding/join requests/200boundaries. All important
phasor,trig,coefficient,boundary intermediate returns are durable before further
work. The sufficient lower h-2G=P-G<=g remains a numerical construction bound,
NEVER a changed candidate tolerance. Query:38continuous variables/358constraints.
No source polynomial regeneration,old702tests,charts,graphs,banks or audit replay.

Independent100NEWoriginal gate bindings,400trig enclosures and1216exact coefficient
identities pass. A first reader then fails while looking up pre-repair ^ syntax
for an unchanged old row. Its failed source/guard/log/proof prefix stay intact.
V2 uses the saved valid multiplication query, reuses the completed prefix and132
NEW row certificates by hash,then expands152remaining NEW rows. All284NEW rows/
1804coefficients bind;74old rows/old482coefficient proof are reused,not replayed.
Five guarded processes retain four successes and one failed reader;20exact
code/plan files are code-only in `surrounding_clock_source_time`.

The ONE180sec/1thread evaluation returns actual timeout,exit0,180.087737208seconds,
118194176bytes peak combined solver/worker RSS. No assignment or candidate is
returned. Worker completion is successful retention of an UNRESOLVED scientific
return, not infeasibility or a trajectory. No restart,continuation,tactic/order/
degree/anchor/tolerance/profile/limit sweep or budget increase follows. All
returns,queries and intermediate/certificates remain private and hash-bound.
Original F,separate components,source missing136/heldout31of32 failures,all original
caps/centers/tolerances/model/gates stay unchanged. No native position/physics/
GPU/PPO/reference mutation/physical or score success. The actual trajectory goal
remains ACTIVE. Do not rerun or enlarge this completed bounded study.

## Direct primal inequality restoration, 2026-10-19

A NEWfixed-start restoration uses the unchanged SAVED358inequalities/38variables.
Ten new derivative/merit/strict-depth/normalization/damping/alias/finite/boundary/
Armijo/coupled-continuous contracts pass. A cached backend is independently bound
by684exact derivative-stencil identities,3080derivative index terms,358numeric
values and684numeric derivatives;2286value/3080derivative pack entries are
structurally verified. Numeric representation diagnostics do not alter any gate.
No source/time polynomial,old query,control,optimizer or source audit is replayed.

Initial F index commands are transformed to half angles with four domain
violations retained,not clipped. The old five SATposes are UNUSED. Static row
conditioning,damping1e-6,trust step.02,merit target1e-10and Armijo rules are
algorithmic,not source confidence,effort or acceptance gates. Only unchanged
exact polynomial inequalities and strictdepth>0can yield a proposed-success
return. One80iteration/120sec/one-thread call ends at80iterations in1.184087417sec,
611trials,zero exact-feasibility checks,NOpoint. Its terminal return and every
intermediate remain REJECTED. Six model inequalities remain negative near
source119/120and temporal joins;this is not original-family/task infeasibility.
No restart,alternate start/weight/damping/tolerance/budget/profile sweep follows.

Six NumPy matrix-product warnings remain with cause UNPROVEN. A NEWsaved-evidence
audit,without method/derivative/linear-solve reruns,checks80normal systems,
115520matrix entries and611saved decisions. Actual rounded normals are exactly
symmetric,with positive quadratic-form margin>9.99999985e-7. Matrix/rhs errors
<1.500e-14/2.013e-17and raw linear residual<4.081e-17fit diagnostic rounding bounds.
Finite agreement does not erase warnings,prove their cause or qualify a path.
The audit's terminal-labeled residual field uses the last pre-step iterate;
a separate private clarification records actual final-trial values and their
binding to the terminal vector. Low bits differ,both have six violations.
All executed sources/returns remain unchanged;no repeat to reconstruct evidence.

Four guarded workers succeed;scientific restoration remains UNRESOLVED.
13exact sources/plans are code-only in `primal_inequality_restoration`;all numeric
iterates/models/source/geometry/results remain private. F,separate components,
missing136/heldout31of32 failures,all original source/model/cap/center/tolerance/
gates and production runtime/tests stay unchanged. No native/physics/GPU/PPO,
root/reference replacement,motor/grasp/task/policy/score qualification occurs.
The owner's actualtrajectory goal remains ACTIVE. Do not repeat/extend this
completed restoration,contracts,binding or saved-return audit.

## Saved-root/preserved-branch source prerequisite, 2026-10-19

A NEW pure-URDF source prerequisite checks the saved unqualified 19-clock root
mode with F's unchanged palm/thumb commands. No source fit, polynomial, old
source audit, native position, physics, optimizer or alternate pose is run.
The actual URDF ancestor chains prove index/middle joints cannot alter these
palm/thumb points at a fixed root and fixed thumb commands. F differs from the
previous late freeze only at early thumb frames0:4; the late repaired thumb
frames116:150 remain unchanged. The mode changes roots103:121, overlapping late
frames116:121, so no prior whole-component certificate is inherited.

ONE19-state/38-anchor check retains all original source caps, .01m camera depth,
1e-6px image tolerance, human flags and missing evidence. Twelve thumb anchors
at104:115 fail original image limits; maximum excess3.443327293px. All palm
anchors and the other thumb checks pass this limited scope. There are no missing
anchors in this selected window; no missing source label is filled or promoted.
Every proposal, FK/source return, rejection and empty unknown set is durable.

A separate independent implementation uses explicit quaternion matrices,
URDF Euler origins, Rodrigues joints and scalar compensated products. All38
chain/projection/gate identities agree: world closure<8.327e-17m, camera closure
<1.111e-16m and source-error closure<3.251e-13px. Diagnostic closure thresholds
do not change original source or candidate acceptance tolerances.

This rejects ONLY the saved-root combination with these F palm/thumb commands
held fixed. It does NOT exclude adjustable unfrozen thumb joints, other allowed
continuous roots, the source/time family or task. The isolated root mode's old
orientation/motion checks remain valid in their stated scope; they never proved
whole-source accuracy. A broader index-only construction would leave this source
failure unresolved. Any next coupled source construction must include unfrozen
thumb source/motion constraints while preserving all early/late repairs exactly.
No new wrist axis, amplitude, time profile or source-confidence change follows.

Two serial guarded workers succeed at retaining this scientific rejection.
Six exact sources/plans plus one generic FK source snapshot are code-only here.
All numeric states, human/source/geometry and error results remain private.
F, separate components, missing136, heldout31of32 failures, original parameters,
caps, centers and tolerances remain unchanged. No new candidate, native, motor,
grasp, trajectory, physical task, GPU/PPO or score result is qualified.
Do not rerun these new producers/checks to reconstruct or tune their outcome.

## Whole-clock index/unfrozen-thumb angle source model, 2026-10-19

A NEW direct angle formulation covers the original151clock and actualURDF.
Source availability is derived separately for every original palm/thumb/index/
middle landmark, preserving all604right mask decisions and136missing anchors.
Both index/thumb have117available anchors. Earlythumb0:4 andlate116:150 stay
exactly F;34missing-clock commands stayF. Variables:234index+231thumb=465,
representing117index and77unfrozen-thumb clocks. The same one saved unqualified
root mode and separate unqualified middle component form a model hypothesis;
no reference pack/component/source array is replaced or qualified.

Twelve NEW analytic contracts pass actual chain/root/origin ordering, explicit
joint axes and separate branches, smooth source-cone/Jacobian identities, strict
positive depth, original image tolerance/tiny terms, exact original angle/time
rows, fixed-boundary folding, minus-two second differences, actual nonuniform
timestamp/velocity units, infeasible immutable rows and malformed inputs.
Analytic coordinates are proof controls, not observed/robot/source anchors.

ONE model has8114exact linear inequalities: actualURDFpositions, original.25
components/.05addedadjacent+second/actual velocities plus the entire inherited
region2index box. All2948constant-only rows remain and pass. Original position
1e-8rad,motion1e-7 and velocity1e-7rad/s tolerances are explicit. Exact expressions
bind saved binary inputs; they do not reproduce every floating intermediate.
No rounding/Jacobian diagnostic changes original candidate acceptance checks.
All38saved half-angle domains, fullregion2correlations andthree109unknownbands
are copied byhash, with no endpoint or source/time polynomial regeneration.

194unexpanded actualURDF/camera source-cone factors retain filled original image
disks and SEPARATE strict.01mdepth. Radius is originalcap+original1e-6px tolerance.
Source derivatives use actual joint world axes and tip offsets. Fixed palm,
immutable thumb and separate middle source evidence is reused byhash in its
stated scope. The194initial-F factor returns are constructor diagnostics only,
not a candidate, new source fit/measurement or whole-source accuracy evidence.
Source/motion blocks couple clocks within each digit; cross-digit native contact
is not represented or claimed at this stage. The fixed-thumb source failures
remain retained; the newly variable thumb clocks are not promoted as feasible.

Independent availability/column/preservation bindings pass604mask decisions,
465columns, all8114exact angle rows/14094term identities and2948fixed rows.
A separate complex analytic URDF chain verifies194source factors,1358joint
derivatives/4074world-Jacobian coefficients. World/camera closures<1.111e-16/
2.221e-16m; worldJ<8.327e-17, sourceJ<9.550e-12, depthJ<8.327e-17. All38domain
copy identities pass, without old model/source/derivative/solver audit replay.
Every variable/factor/linear-row constructor, derivative and rejected/unavailable
state is durable before further work. Numeric diagnostic bounds are NOT gates.

Three guarded workers succeed. Nine exact sources/plans are code-only here;
all human/source/geometry,465angle-model arrays and numerical results are private.
No solver, new point/trajectory, native position, physics, GPU/PPO, motor/grasp,
task, policy or score qualification occurs. F, separate components, source
authority/missing136/heldout31of32 failures, original roots/centers/model/physical
parameters/caps/tolerances/gates stay unchanged. Do not rerun these producers,
contracts or independent bindings. A NEW bounded continuous construction can
consume this saved model only after its algorithm contracts and bindings.

## Bounded sequential camera-cone construction, 2026-10-19

A NEW sequential SOCP method consumes the SAVED465angle/8114linear/194source/
38half-domain model. It retains one fixed F start, all original source/model/
temporal authority, unqualified root/middle components and all immutable/missing
commands. Local affine camera SOCs and one common elastic variable are step
models, not global outer source/task proofs. All actual nonlinear source/motion
constraints must pass independently of merit, slack or solver status.

Twelve NEW analytic controls pass cone order/signs, original linear prefix,
positive static conditioning, separate depth, trust bounds, finite half-angle
derivatives, strict-zero rejection despite zero merit, local-versus-actual
geometry, positive quadratic objective, one tiny SOC, rational SOC-dual/finite-
box signs, Armijo rules and one NEWcoupled continuous nonlinear construction.
Controls are not robot/source anchors. Two NEWcache readback/dataflow contracts
pass. A wrapper differs from the contracted method only by reusing the initial
view and first matrices; executable AST lineage verifies every other statement
unchanged. Twelve controls are reused byhash, not rerun. Own private SHA-bound
pickle caches are never external or published.

ONEpreflight constructor has466columns/9897rows:9315nonnegative rows+194SOC3cones.
Independent exact scalar/rational maps bind20243coefficients and9897RHSvalues,
all objective/cone/column/source/unit bindings. Arithmetic packing guards concern
the LOCAL numerical step only, never original candidate tolerances. The initial
source/Jacobian returns are reused byhash; no old model/source producer is replayed.
Every subsequent actual canonical constructor is independently bound BEFORE its
solver constructor and call. Actual primal/dual/slack/status returns stay raw.

ONE40major/120sec/1thread method uses stepLinfty.1rad, objective
rho+.5*1e-3*||step||^2, positive STATICconditioning and at most12Armijo trials per
major iteration. Subproblems have100iterations/atmost10sec/defaulttolerances.
No old SOCP/SDP/LP/model/optimizer restart, warmstart, profile/degree/anchor/
weight/trust/tolerance/limit sweep or continuation occurs. Failed/intermediate
angles cannot become commands or a best candidate.

The method ends UNRESOLVED_no_predicted_descent after7subproblems/6actualtrials
in8.240607625seconds; all7local solver returns areSolved. Zero exact-acceptance
checks and NOproposed point result. Actual normalized merit goes2.8739982166to
.08896329569; it is not a source/pixel acceptance tolerance. Last rawrho
.0889632957457 gives predicted reduction-5.39805e-11. This numerical stopping
decision does not prove nonlinear stationarity, fixed-root infeasibility or a
necessary wrist change. No solver tolerance repair or restart follows.

A distinct saved-evidence audit checks7canonical certificates/141701coefficient
residuals/69279RHSresiduals,6unlinearized views/6Armijo decisions/7rawdirection
identities, and retains1164NEWactualURDF source returns without FK/Jacobian/
solver/method replay. Merit/linear/pixel homogeneous closure errors<1.111e-16/
1.777e-15/2.061e-13px. The raw terminal vector matches the last accepted trial.
There are240positive model residuals,44above1e-5 normalized units:25source,
212linear and3retained-half-domain. Important failures include thumb109 image
.09142313px versus originalcap.00235290px+original1e-6px, thumb110/111 source,
thumb0speed/adjacent, thumb2adjacent/second andthumb1lower position. These
diagnose only this REJECTEDreturn, not global source-family/task impossibility.
No RuntimeWarning occurs in this episode; earlier warning failures remain.

Four serial guarded workers succeed at retaining scientific failure. Fifteen
exact sources/plans/ASTmetadata are code-only here. All numerical matrices,
angles, human/source/geometry, private caches and results stay private. F,
separate components, source missing136/heldout31of32 failures, all original
parameters/centers/caps/tolerances/gates and MP-SR0remain unchanged. No new
candidate/reference, native position, physics, GPU/PPO, motor/grasp/task/policy/
score or trajectory qualifies. Do not rerun these controls, preflight, method,
saved-return audit, or extend/tune any completed budget. Owner goal stays ACTIVE.

## Frozen-scene/native-gate incidence prerequisite, 2026-10-19

A NEW exact incidence/provenance proof reuses the SAVED F native receipts byhash,
without native position, source, cohort or physics/replay reruns. The465angle
robot-only construction has no free robot columns at34missing-anchor clocks;
both complete hand states there are exactlyF. Its object poses and pinned native
geometry/parameters are also F. Four already failing1mm scenes at36/37/47/55
therefore remain identical for EVERYmember of that restricted family. Reported
penetrations are old measurements, not new samples or a new baseline milestone.
All full robot-state bytes, object-pose bytes, frame/timestamps, source masks,
native-model hash, cached native contacts and original gate bindings are durable.

This proves that a robot-only candidate with frozenFobjects cannot pass the
full native gate under those immutable-command constraints, even if its new
source/motion solve passes. It does NOT exclude the task, other source-qualified
object poses, or a family with broader permitted controls. Native static scene
invariance does not diagnose the old physics-startup/replay variability.

Before another full candidate solve, inspect the original object-pose SOURCE
and motion authority and saved observations for a genuinely NEW joint object/
source-native construction. Object changes are not yet source/motion qualified,
are not observations and invalidate old reference certificates. No old object
fit/depth/calibration/probe/verification job may be repeated. Keep all misses,
original gates, object geometry/mass/inertia/friction/stiction/effort, missing
robot commands and source authority unchanged. A new candidate still needs all
original fullclock source/motion and fresh full native/self/swept/motor/grasp
qualification beforeGPU/PPO. New object/source feasibility is not established.

One guarded worker succeeds. Four exact code/plans are retained code-only; all
source/geometry/commands/receipt rows and results remain private. No new native
measurement, physics/GPU/PPO, candidate, trajectory, score or task qualifies.
The verified F and separate unqualified components remain unchanged. Do not
rerun this incidence proof or the reused native audit.

## Original object source/motion authority and translation maps, 2026-10-19

Three NEW serial guarded CPU jobs inspect saved object authority and derive a
translation-only mathematical prerequisite. No old source/depth/native/fit,
contract, solver, cohort or physics producer is replayed. Original source data,
F, all repaired robot components/missing commands, heldout failures, physical
parameters and gates remain unchanged. No pose or trajectory is proposed.

The saved119-file object checkpoint integrity chain passes. All151current source
thresholds are bound to their original receipt/JSON path:132from the selected
track,17from the accepted smooth component and2from its retained translation
repairs. All302cached depth/mask hashes bind. Old threshold values are copied,
not rescored. Four native-obstructed immutable-robot frames36/37/47/55 have
object observation flags, but those are inferred-source provenance, not measured
pose/depth/contact truth. New source/world/native coordinate bindings close
below7.772e-16; that is representation closure, not source accuracy.

Original OBJECT full-track caps are .8029814731m/s,5.0437560801rad/s,
15.4760445311m/s^2,152.5148785861rad/s^2 and p95 speeds .4925047000m/s and
3.5699946858rad/s. Retained object producers compare with1e-9, distinct from
original robot motion1e-7. Original source-comparison tolerance1e-10 is retained;
robot position/image/root-orientation1e-8rad/1e-6px/1e-9rad stay unchanged.
The tracker retains center camera depth>.05m, vertices>.01m, transported prior
jump<=.15m/40degrees and unchanged measured-depth support counts/fractions/body
margin. Object estimates are not assigned robot8mm/.08rad root boxes. Old50mm
and5mm fit limits were construction bounds, not original object acceptance
boxes. Absence of a component box in the inspected gate code is not blanket
permission for arbitrary object movement or a feasibility claim.

Original full-pixel scoring includes both checkerboard parities and is a
POSTCONSTRUCTION gate, not a new optimizer objective. The inventory's even/odd
metadata describes a FUTURE declared training-only protocol; it does not assert
151historical independent partitions. Saved fixed first-surface split artifacts
exist for50and47/49. No new rays, labels or partitions are constructed here.
All original validity, missing rays/infinite residuals, components, source flags
and heldout evidence remain. Any new objective must use training evidence only.

Fourteen NEW analytic controls pass object/world/body/camera translation signs,
nonuniform source-clock velocity and midpoint acceleration, constant-translation
nullspace, fixed-stage norms/fixed-rotation angular independence, separate p95
necessity, object-translation pixel derivatives, positive-depth/parallel-ray
unknowns, translated triangle-plane roots and sufficient support-plane signs.
A plane root is not a visible face/first surface; algebraic support multipliers
are not measured contacts/forces. Controls are synthetic, not robot/source poses.

Independent NEW bindings check453threshold/flag copies,453strict-floor
components and4exact original gate ASTs. Exact full151clock affine maps have
453UNASSIGNED world-translation symbols,150velocity+149acceleration maps,
2241rational sparse scalar coefficients and897rational constants. All map
coefficients and constants agree with independent nonuniform-clock formulas;
no translation is assigned/evaluated. Maximum SOC bounds do not replace the
original p95 constraint. Fixed rotations keep angular motion constant only
within a prospective translation-only construction.

Nine exact code/plans are retained code-only; all source, geometry, thresholds,
clock maps, matrices and results remain private. Production runtime/tests are
unchanged. No native position/physics/GPU/PPO, new candidate, motor/grasp/task,
policy/score or trajectory qualifies. The owner goal stays ACTIVE. Do not rerun
these completed inventories,14controls or independent bindings. Next derive a
genuinely NEW training-only object/source construction using the saved maps and
unchanged authority, preserving all robot missing commands and the original
accepted object component as the starting hypothesis. Before any new bounded
fit, independently bind its training split, exact model/coordinate maps and
all original source/motion tests. No alternate wrist/profile, source confidence,
geometry/physical parameter or tolerance changes. Old native gaps lack normals;
no missing witness may be reconstructed by rerunning an old native audit.
Any changed object/reference invalidates prior certificates and still requires
independent whole-clock source/actualURDF/ALLroot+finger+object motion, fresh full
1mm native hand/object/support ANDself, fullswept,actualmotor/grasp and whole-source
qualification beforeGPU/PPO. Source and geometric feasibility remain unknown.

## Bounded object visibility/support conic construction, 2026-10-19

Ten NEW serial guarded CPU processes finish: six succeed and four fail. All
statuses, sources, returns, unknowns and warnings remain. Thirty-two exact code/
plans are retained code-only; source/geometry/training pixels/models/results stay
private. No original source fit, old source/native audit, optimizer or solver
is replayed. Production runtime/tests and F/robot missing commands/source/physical
parameters/original gates stay unchanged. No object or full trajectory qualifies.

The prospective model has453continuous WORLD object translations at151clocks,
fixed F object rotations, a DECLARED20mm component construction box,24fixed EVEN
training rays per clock, NEW analytic compiled-geometry support planes and the
saved299original velocity/acceleration maps. That box is not original object
acceptance authority or a robot root box. The p95 speed envelope is a sufficient
monotone-quantile construction restriction; the original p95 gate is unchanged.
Selected-face barycentric maps are LOCAL visibility cells, not a first-surface/
full-mask source certificate.1662of3624training rays are baseline misses; all are
explicit/UNRESOLVED and remain infinite, with no invented hit or heldout fitting.

Eleven NEW visibility/support/quantile/QP controls pass, including one tiny
analytical QP. First model preflight FAILS its unsupported1e-10m equality
assumption between decimal URDF origins and compiled origins. The particular
failing body/matrix return was not journaled; it remains UNKNOWN, not recreated.
Read-only static records expose float32-origin differences, not a joint-reference
shift (all14hinge refs are0). Four NEW exact origin-defect/path/support controls
pass. A distinct model binds exact per-path origin error budgets, uses actual
COMPILED FK for support geometry and reuses saved compiled vertices by hash.
It FAILS because support was assumed a fixed body. The actual support is a free
body whose unchanged default qpos0 is used by the native reference auditor.
Its failed support return was not produced/reconstructed. All completed frame0
kinematic returns/static bounds are reused by hash in the next producer.

Final support-default-bound preflight succeeds without native positioning or
physics.3020NEW pure URDF/compiled frame bindings have component difference
<=3.206e-9m within exact path budgets<=4.103e-9m plus a declared1e-12m arithmetic
bound. This accounts for distinct represented input coordinates; it does not
relax any original angle, image, native penetration or source tolerance. Actual
MuJoCo3.8.1 support vertices come from compiled_mesh_vertices, never raw mesh_vert.
All58compiled geoms/66,591eligible fixed-hand-or-support/object pair incidences
remain:17,520NEW selected-axis sufficient planes and49,071NEW independently
certified broadphase omissions under the declared box. These are construction
constraints, not measured contacts/forces or recovered old gap normals.

First independent ray binding FAILS at74/ray22: a NumPy inverse derivative
comparison differs1.920852810e-8 versus its1e-8 representation check; its actual
return is saved. A genuinely distinct exact SAVED-binary-input3x3 method proves
5886affine constants/17,658derivative identities, retaining rational discrepancy
bounds uniformly over the declared box (maximum1.603e-9). It does not recreate
unsaved floating intermediates or prove upstream camera/trig/source truth.
Three NEW exact inverse/singular/signed-box contracts pass and are reused by
hash. The exact-input audit then FAILS a float64 expected-objective comparison:
P matches, q differs by.0416152261. Actual coefficients remain unchanged.

Cached depth isfloat32. NumPy2.2.6 promotes the Python-float-minus-depth scalar
and its division tofloat32. Six NEW independent integer/rational IEEE-binary32
rounding/promotion controls and all1962objective scalar bindings reproduce the
ACTUAL P/q exactly. No coefficient or solver-return repair occurs; the previous
float64-expectation audit remains failed. Old matrix-multiplication warnings
remain in logs with cause unproven. Completed3020frame/1962exact-ray/control
prefixes are reused by hash, never rerun to hide failures. Independent final
binding checks26,576nonnegative rows,897SOC scalar rows, allsupport/omitted pairs
and all453columns. Row/support closure<=1.111e-16/2.221e-16. Source cells remain
locally approximate and all original actual source/motion gates still rule.

ONEnew27772row/453column construction has299SOC4cones,100iterations/60sec/1thread/
default tolerances, no warmstart/restart/continuation/repair. It returns
PrimalInfeasible after17iterations/.151607667sec. No pose is formed; no original
full-source/motion/native gate is run on a failed point. All actual x/z/s and
constructor returns are durable and unaltered. Four NEW SOC finite-box/dual-sign
controls precede independent exact raw-return checks.80,157raw model coefficients
bind, with0negative nonnegative-dual entries and0invalid SOC dual blocks. A raw-
dual residual-corrected finite-box certificate has strict margin35.2573630435
in algebraic certificate units. It excludes ONLY THIS saved numeric local-face/
selected-support construction inside its20mm box, not all object translations,
other visibility/support branches, original source/time family or task. No
mandatory wrist change follows. An exact opposite-row lookup finds0pairs;
there is no two-row native-plane contradiction or established causal diagnosis.
Dual coefficients are not measured forces or physical importance scores.

All28NEWanalytic controls pass their stated scopes, including prefixes in failed
processes. No failed status is erased and no budget is increased or replayed.
No new candidate, full reference, native position/physics/GPU/PPO, motor/grasp,
policy/score or trajectory qualifies. F,separate components,136missing/31of32
heldout/source-confidence failures,original caps/centers/tolerances/parameters and
reserved episodes remain unchanged. Keep Brev stopped during CPU/idle.

NEXT use saved row/certificate/source-visibility records to choose a genuinely
NEW continuous source/native construction that can handle visibility-face and/or
support-choice changes. The certificate does not establish which restriction
causes failure, so do not infer a causal source/native/motion claim from dual
magnitudes. Do not retry this SOCP, drop gates, change tolerances/limits/anchors,
replay samples/maps/audits or promote a failed vector. A new method must retain
training-only authority, every missing/unknown/occlusion/parallel case, all151
clocks and original motion/heldout/source gates. Existing completed prefixes
stay reusable by hash. Any real object/robot proposal still needs independently
checked whole-clock source/actualURDF/all original object+root+finger motion,
fresh FULL1mm native hand/object/support ANDself, fullswept,actualmotor/grasp and
whole-source qualification beforeGPU/PPO. Owner goal remains ACTIVE/unachieved.

## Dynamic source visibility and support construction, 2026-10-19

Seven NEW serial guarded CPU processes complete: six succeed and one synthetic
binding fixture fails. A separate syntax/absent-plan preparation error happens
BEFORE guard state mutation; its exact failed text is retained as.py.txt.
Twenty-four exact source/plan/metadata files are code-only. Numerical records,
source/training pixels, geometry, arrays and results remain private. Production
runtime/tests,F,allsource authority/missing/heldout/physical parameters/gates stay
unchanged. No native position/physics/GPU/PPO or trajectory qualification occurs.

A NEW subsystem-functional analysis binds all27,772saved conic rows to box,
selected geometry, local visibility or complete motion-cone groups. Three NEW
scope/whole-SOC/additivity contracts pass. Six predeclared restricted algebraic
functionals all have NONNEGATIVE finite-box upper bounds: all are INCONCLUSIVE.
Independent27,772row classifications and6exact functional identities verify.
The prior strict joint-construction certificate remains valid; no individual
source/native/motion cause follows from raw dual magnitudes. Actual duals stay
unaltered; separate functionals are certificate objects, not forces/commands.

A genuinely NEW nonlinear construction allows first-surface FACE CHANGES and
MAXIMUM over all retained face-separation directions. It uses the same453world
translation variables at151clocks/fixedobjectrotations/20mmalgorithmicbox and
fixed24EVENsamples per clock. It retains every one of4,832,184directions across
17,520nearby pairs and all49,071prior uniform-box omissions, never sampling a
finite pose bank. NEW all-direction support bindings close<2.221e-16m; prior
selected-direction alias<3.331e-16. Old compiled geometry,3020frame bindings,
3624baseline ray returns/1662infinite misses and299motion maps are REUSED BYHASH:
no FK/baseline ray/native audit replay. Full18,000visual geometry edges support
projected-union distance for CURRENT missing training rays. Infinite model depth
is retained, with an explicitly finite OPTIMIZER-ONLY missing loss; no ray,
mask component, source flag or heldout label is deleted/filled/relabelled.

Ten NEW analytic controls pass dynamic support/face switching, projection/edge
and robust-depth derivatives, zero edges/misses, actual p95 sorting pieces and
one NEW tiny sparse nonlinear solve. The actual method uses ORIGINAL maximum
speed/acceleration and actual p95 norms; it does not substitute the old inner
speed envelope. Robust depth/coverage losses and100native/motion penalty scales
are optimizer units, not source confidence or candidate gates. All actual
original source/motion/native/self/swept/motor/grasp tests still rule.

The first full synthetic runtime check FAILS one centered-FD row805: its
synthetic velocity50is exactly.65, at the positive-part kink. Actual chosen
zero generalized derivative is valid while centered FD is.795526273; all other
row errors<1.627e-8. Its exact source/returns remain failed. A distinct saved-
fixture scope control and smooth synthetic fixture (.600123+.001*i, source/
method/gates unchanged) pass two full-runtime directional controls and exact
fun/Jac cache binding. Relative directional errors<1.656e-11. All4,832,184actual
axis norms,897scalar motion aliases and3624EVENsamples bind independently.
The scalar-label399in one bool key is a documented KEY typo; actual matrix/report
have897rows. Likewise131in a declaration key is a KEY typo, not a numeric gate:
original136missing right anchors/commands are unchanged. Executed sources and
returns remain exact, with corrections in separate private metadata.

ONEpredeclared SciPy1.16.3sparse TRF/LSMR method starts at Fzero translations:
40function-evaluation/170sec/one-thread bounds, default tolerances, no restart,
warmstart,tolerance/weight/profile/budget sweep or old failed-point promotion.
It returns optimizer success (`ftol`,status2) after31actual model evaluations/
14callbacks. The recorded run takes155.060sec. Callbacks and every candidate/
source/geometry/residual return are durable before decisions. Solver success is
NOT qualification. Its actual final vector is evaluation30, not an arbitrarily
chosen latest/best trial. Source faces change727times; numerical union proxies
retain2violations (max1.208e-7m above the model's1mm threshold). These are NOT
actual native contacts or penetration certificates.

The ONEnew raw object proposal is checked against all six ORIGINAL full-clock
object motion limits and all151ORIGINAL source gates. Motion passes at the
unchanged1e-9 object comparison tolerance. SOURCE FAILS60of151frames: the saved
named checks include21quality,20silhouetteIoU,54vertex-median regressions and
2old strict-floor failures, with overlaps. No tracker failure occurs. Original
source comparison1e-10 remains; no physical/reference score is promoted. All
raw and intermediate vectors are REJECTED as references. No new native audit
or GPU/PPO follows this source failure. Do not tune/restart/reweight against
these full-pixel/heldout postchecks or selectively repair the failed frames.

A distinct saved-return audit verifies31complete states/112,344typed source
entries/777,108residual entries,14callbacks/15cache reuses and actual optimizer
vector matching. One first-time independent final-return union check covers
all4,832,184directions/17,520pairs, closing<2.776e-17m. All151source decisions,
raw-vector-to-object transforms, unchanged robot/rotation bytes and original
motion identities verify from saved returns without scorer/FK/ray/solver replay.
All17NEWanalytic controls pass their stated scopes, with the failed fixture and
preparation failure retained. No object source/motion component, full reference,
native/grasp/motor/task/policy/AUC or trajectory qualifies. Fstill has8right-self/
5objectfailures,missing136/heldout31of32 andsource/model/gates unchanged.

NEXT derive a genuinely NEW TRAINING-ONLY source-constraint construction whose
semantics match original silhouette/mesh-vertex-median nonregression, rather
than only soft24ray depth/projected-edge losses. Derive it from ORIGINAL metric
and observation definitions, never the60postcheck failures or heldout weights.
Allclock correlations/native support/source uncertainty remain. No rerun of this
method/31states/3624baseline rays/runtime controls/subsystem analysis or extension
of40eval/170sec/penalties/20mmbox is queued. No old fit, interpolation/native/
source audit, graph/bank/profile/tolerance sweep or lost-record reconstruction.
Any new candidate must start from F's accepted object component, preserve all
missing robot commands and repaired components, retain all original gates and
separately pass whole-clock source/URDF/root+finger+object motion, fresh FULL1mm
native hand/object/support ANDself,fullswept,actualmotor/grasp andwhole-source
qualification beforeGPU/PPO. Keep Brev stopped CPU/idle; owner goal ACTIVE.

## Training-only source constraints, 2026-10-20

Four NEW serial guarded CPU processes succeed. Fourteen exact source/plan/helper
files are archived code-only under `training_source_constraints`; all source
pixels, geometry, arrays, numerical returns and certificates remain private.
Production runtime/tests are unchanged. No actual nonzero episode pose or solver
is selected/evaluated, and no trajectory/source accuracy/physical score qualifies.

The NEW source construction uses ORIGINAL functional definitions on a declared
EVEN training domain: eligibility precedes deterministic raster-order sampling
(cap4096), full source-visual triangle rasterization preserves ORIGINAL clip then
integer truncation, silhouette counts use only training pixels, and the median
is distance to the SOURCE VISUAL MESH VERTICES with original even-rank averaging.
These source vertices are the original source operator; they are not a substitute
for compiled MuJoCo support vertices in native collision checks. Quality uses
IoU*exp(-median/.02). The NEW training statistics/nominal hard nonregression
predicate are construction restrictions, not original full-pixel gate changes.
Original source comparison1e-10 and object motion comparison1e-9 stay explicit.
The full-pixel/heldout gates remain postconstruction verification. The rejected
old point, its60postcheck failures and heldout values are not construction inputs.

Nineteen NEW controls pass:12functional,4independent BVH,3cache/provenance/unknown.
ONE151clock training dataset has572,956depth points and no insufficient-support
or zero-central-distance unknown frame. Nearest-vertex/rank ties in42frames
remain explicit. Current nearest vertices/ranks may change; none becomes a
measured pose/mandatory anchor. Positive-depth/projection/zero-distance/empty
support cases fail or remain UNKNOWN explicitly. No missing observation is filled.

Independent bindings verify1,818,644quantizer coordinates,302intersection/union
counts,572,956point lifts and nearest pairs using an independent source-vertex
BVH (39,343,817leaf point/vertex pairs). Maximum projection/point/nearest/gradient
identity errors are2.274e-13px/2.843e-14px/4.441e-16m/2.221e-16. These are numerical
identity diagnostics, not source-accuracy proofs or candidate tolerance changes.

A NEW hard-source oracle retains real raster discontinuities (no invented IoU
gradient) and local median derivatives with explicit rank/NN uncertainty. Its151
actual F queries reuse the saved training baseline BY HASH, without re-rendering
or rescoring it; no actual nonzero episode query ran. One synthetic zero-cache
payload is arbitrary and not a consistent source measurement: its scope is cache
transparency only. Private clarification preserves executed source/returns and
does not rerun controls. Actual151quality bindings are independently verified.

The latest read-only provider attempt times out, so CURRENT Brev state is
UNVERIFIED. Last actual STOPPED observation remains2026-10-20T03:42:30.871525UTC;
the raw attempt's older fallback timestamp is retained but superseded by that
later actual receipt. No VM mutation or paid job is queued. No future lifecycle
reliability is inferred. F/separate components/missing136/heldout31of32/original
source authority/geometry/physics/boxes/caps/centers/tolerances/gates stay unchanged.

NEXT derive one NEW construction that uses the saved training hard-source
predicate during geometric/motion restoration, with explicit discrete-raster
acceptance and meaningful new sign/model/acceptance contracts and independent
canonical bindings before any bounded evaluation. No method is yet selected or
queued. Reuse all completed training datasets/19controls/BVH/oracle evidence BY
HASH; never replay them or the rejected soft method. No heldout/full-pixel
feedback, failed-point warmstart, framewise tuning, budget extension or gate
change is allowed. Object changes alone do not repair8right-self failures.
Actual whole151source/URDF/root+finger+object motion, fresh FULL1mm native
hand/object/support ANDself, full swept, actual motor/grasp and whole-source
qualification still precede GPU/PPO. Owner trajectory goal remains unachieved;
this research checkpoint is not goal completion.

## Sufficient training-median order-statistic support, 2026-10-20

ONE further serial guarded CPU worker succeeds with seven NEW analytic contracts
and ONE new saved-data map inventory. Six exact source/plan/helper files are
code-only under `training_orderstat_support`; all numerical/source/geometry
records stay private. No old training/source/nearest-neighbor/control audit is
rerun. No solver or actual nonzero episode pose is selected/evaluated. Production
runtime/tests/F/source/parameters/original gates remain unchanged.

For n training distances, nested sets of floor((n-1)/2)+1 and floor(n/2)+1
training points suffice to bound the two median ranks. Distance to ANY fixed
source visual vertex upper-bounds actual nearest-vertex distance. Thus fixed
proxy-vertex norm bounds on those sets can ensure a training median upper bound
even when actual nearest vertices/ranks change. Proxy choices are conservative
construction witnesses, not observed correspondences, mandatory source poses or
necessary source-family restrictions. Original fullpixel/heldout gates and the
actual hard training raster/median/quality oracle remain separate and mandatory.

Seven NEW contracts cover even/odd nested-rank cardinalities, nearest-vertex
switches under a valid proxy bound, SOC4 world-map/adjoint signs, exact zero-radius
semantics, all-frame hard-filter completeness and a continuous interpolation
counterexample for p95 feasibility. Original actual p95 must be checked on each
trial; interpolation of feasible endpoints does not ensure p95 feasibility.
Raster IoU remains discrete, with no invented classical gradient.

ONE fixed151clock inventory proposes286,608SOC4maps/1,146,432canonical rows on
453UNASSIGNED world-translation columns. It reuses the saved training points,
nearest-vertex returns and original transform domains BY HASH. New independent
cardinality/row-sign/exact affine-map-error bindings are durable. The original
source statistic is not recomputed and no candidate value is assigned.

Exact squared-norm comparison retains124nominal F proxy-radius defects, maximum
10524878282246375/1329227995784915872903807060280344576m^2
(about7.92e-21m^2). These result from distinct saved numerical distance/point
representations; they are not clipped, silently repaired or used to change any
original candidate tolerance. Initial F feasibility for this exact proxy model
is NOT established. No arithmetic/packing allowance or radius repair is selected.
Any future allowance needs a justified numerical-operator binding and explicit
scope before a solve, while actual source acceptance remains unchanged.

NEXT design one bounded hard-source-filtered geometric/motion construction using
these saved proxy maps only if its numerical semantics are justified. Preserve
all124defects and retain the distinction between this sufficient restriction,
the nominal training oracle and original fullpixel source gates. First establish
new packing/arithmetic/actual-acceptance contracts and independent complete-model
bindings; no solver/pose evaluation is yet selected or queued. Never replay these
seven controls/maps or the completed19training controls/data/BVH/oracle cache.
No old rejected point/60postchecks/heldout values may enter fitting, frame weights
or parameter selection. No old budget continuation/restart/tolerance repair.
Current Brev remains UNVERIFIED after the saved read-only timeout, last actual
STOPPED03:42:30.871525UTC; no VM mutation or paid work. All source/missing136/
heldout31of32/F/separate components/gates remain unchanged. Object changes alone
cannot clear the8right-self failures. No source/native/self/swept/motor/grasp/
trajectory/policy/AUC qualification follows; the owner objective is unachieved.

## Hard training-source conic construction, 2026-10-20

Seven NEW serial guarded CPU processes complete: six succeed and one fails at
Python parsing BEFORE any module execution/constructor/solver. Twenty-two exact
source/plan/helper/metadata files are code-only under `hard_training_source_conic`.
The failed source is retained verbatim as.py.txt, with its private failed log,
guard and plan. V2 inserts one missing whitespace and changes only its output
directory; the syntax-only diff is retained. There is ONE completed episode
solver call, with no restart/warmstart/tolerance/weight/profile/budget sweep.
Production runtime/tests and all original source/physical authority stay unchanged.

Five NEW binary64 contracts derive uniform source-translation/point-local and
outward-proposal bounds. Independent exact operation-by-operation recurrences
cover all six three-term dot orders on all151frames. Maximum distance allowance
is9.984e-15m, covering the inherited124nominal F proxy defects without modifying
old radii/arrays or candidate tolerances. These OUTWARD proposal guards do NOT
certify KD-tree/raster source operators, source accuracy or actual acceptance.
All124defects and original1e-10source/1e-9object-motion comparisons remain.

The NEW canonical model has454columns/1,166,056rows/286,907SOC4cones and18,428
nonnegative rows. It combines ALL286,608saved sufficient median proxies, ALL299
original max-speed/acceleration maps and17,520selected geometry directions from
the saved INITIAL F union return. All453world translations are continuous in a
declared20mm algorithmic box; one bounded clearance slack is optimized. Frobot
commands/object rotations remain fixed. No old rejected point is a warmstart,
and no fullpixel/heldout/old60failure value enters fitting or framewise tuning.
The phase objective is rho+.5*1e-3*sum(delta^2), an algorithmic choice, not physics.
Original actual p95 and all source/native/self/swept/motor/grasp gates still rule.

Independent complete binding verifies all151new source blocks/3,439,296copied
coefficient cells,17,520geometry rows,299motion cones,908box rows and objective.
One NEW tiny analytic callback control passes, saving all six actual info returns;
actual cone dimensions and callback API are checked without dependency installs.
All old source/NN/control/packing/model audits are reused BY HASH, not replayed.

ONE100iteration/60sec/one-thread/default-tolerance solve returns Solved in30
iterations/17.088963sec, with31durable info callbacks. Raw x/z/s and constructor
returns are saved before further work, never clipped/repaired/normalized. The
complete guarded evaluation takes26.120sec/peakRSS853,426,176bytes. Solver success
does NOT qualify a point. The ONEraw proposal fails training silhouette IoU at
19clocks and training quality at18 (overlap), while training median has0failures.
All151actual hard-training decisions are retained. ALLsix ORIGINAL full-clock
object motion checks pass, at the original1e-9 comparison. The point is REJECTED.
No fullpixel/heldout source rescue, native-position/physics/GPU/PPO or new source-
motion component/reference follows. These19training failures are NOT comparable
to the old60fullpixel failures and do not establish a physical improvement.

Four NEW analytic dual-defect/dot-bound contracts precede the independent saved-
return audit. It binds2,652,701canonical coefficients,31callbacks,151training
decisions, all six saved motion decisions and raw transform/unchanged robot and
rotation bytes, without source/NN/FK/native/solver replay. All286,907new rational
root certificates pass; raw dual nonnegative/SOC failures are0/0. A finite-box,
residual/dual-defect-corrected certificate gives rho>=.005355725241031122m for
THIS frozen numerical median-proxy/selected-direction/20mm-box construction.
Thus zero clearance slack is excluded only for that restricted numeric model.
This is NOT native penetration, a source/time-family/task exclusion, physical
force/importance or a required wrist change. Raw rho=.005355750741394552m and
all rejected returns/statuses remain unchanged. Numerical primal residuals are
retained separately; they are not original candidate acceptance tolerances.

NEXT derive a genuinely NEW training-source feasibility representation that can
handle changing median-rank/nearest-vertex subsets AND discrete silhouette
coverage, using original source definitions and saved training authority. Do not
swap a prefix/axis or reweight/restart this completed conic model to repair its
return. First derive meaningful new pointwise/cardinality/raster/unknown/error
contracts and independent maps; no new solver, pose or physical job is queued.
No old data/controls/budgets/audits or fullpixel/heldout failures may be replayed
or used for fitting/selection. A sufficient construction exclusion does not
exclude original source-valid paths. Object changes alone cannot fix8right-self
failures. F/separate components/missing136/heldout31of32/parameters/boxes/centers/
caps/tolerances/gates stay unchanged; actual whole trajectory remains unqualified.

## Free-rank, nearest-union and discrete training raster, 2026-10-20

Three NEW serial guarded CPU workers succeed. Nine exact sources/plans/public
reference metadata files are code-only under `free_rank_raster_source`. Private
source arrays/results remain private. All18NEW analytic contracts pass their
stated scopes; no completed old control/producer/audit/solver is replayed.
Production runtime/tests and all original gates/parameters remain unchanged.

The NEW median representation leaves thresholds, rank subsets and nearest
vertices FREE. For n=2k ideal distances, count(d<=a)>=k and count(d<=b)>=k+1,
0<=a<=b and a+b<=2*m0 are equivalent to median<=m0. Odd n uses one threshold.
General forward/reverse proofs, tie/zero/missing behavior and a changing-nearest
vertex union are retained. These are algebraic proof controls, not measured
source poses, fixed human anchors or robot commands. All original visual6022
vertices are alternatives; no fixed nearest/prefix map is substituted.

Numerical semantics remain explicit. The original binary64 two-rank average
can differ from the rational average at round-to-even and subnormal boundaries;
NEW controls retain both examples. The numerical branch uses the ORIGINAL
rounded average on numerical distances. Exact Euclidean ball unions bind the
ideal saved-binary-input model, not an unproved KD-tree/norm implementation or
upstream camera/source truth. Missing/uncertain events retain separate certain
and possible cardinalities; they are never dropped or set false.

The NEW raster branch program follows original clip-then-truncate, LINE_8,
shift0, full visual-triangle OR and EVEN-domain I/U. It includes integer line
ties, double-intersection clipping, degenerate triangles and inclusive scan
spans. Fifteen initial contracts include fixed NEW synthetic triangles/lines;
no episode baseline/rejected scene is rendered. Public OpenCV4.11.0 source
hashes and the installed4.11.0binary hash are retained. Matching a version tag
and synthetic cases is NOT a proof of full installed binary equivalence.
Source-informed code retains the original Intel license and attribution.
See official source: https://github.com/opencv/opencv/blob/4.11.0/modules/imgproc/src/drawing.cpp

All integer projected-vertex cells remain FREE in[-20000,20000], with correct
positive/negative/zero/saturation preimages under strict positive depth. No
classical IoU derivative, fixed face/visibility branch or front-surface/depth
buffer surrogate is invented. Crossed quantizer boundaries remain UNRESOLVED
unless original numerical execution or a justified outward bound resolves them.
Certain/possible full-triangle coverage yields exact outward integer-count
determinants; empty unions and missing/parallel/strict-depth cases stay explicit.
Fullpixel/heldout source gates remain POSTconstruction only.

ONE factored151clock formulation uses453continuous UNASSIGNED world translations
and260FREE median thresholds. It represents572,956saved training points and
3,450,341,032point/vertex distance predicates (6,015,628,724threshold instances),
1,818,644FREE integer projection coordinates,1,812,000triangle programs and
77,312,000EVEN pixel expressions without expanding billions of clauses or
assigning any value. Original fixed F rotations are a declared scope, not new
source truth. No new20mm box/radius/source cap/tolerance or solver is selected.
All six original object caps/actual timestamps and tracker/full-source gates
remain mandatory; p95 is not replaced by a convex maximum-speed envelope.

Three further NEW inverse-cell/shared-variable-polynomial/unknown-cardinality
contracts precede independent binding of4,983NEW exact factor coefficients,
all151rank/domain/incidence records,302input point/partition hashes and151
ordered durable BEFORE/AFTER frame pairs. The compiler and its independent
matrix embedding agree exactly. Old training statistics,6022vertex geometry,
domains, motion maps and old19+7+10contracts are reused BY HASH, not rerun.
The new representation is FORMAL IDEAL equivalence with explicit numerical
operator boundary obligations. No numeric episode pose, source/NN/raster
evaluation, solver, native position, physics, GPU/PPO or trajectory qualifies.

NEXT derive only NEW numerical-operator/error and actual-acceptance contracts
needed to evaluate this free-rank/full-raster construction, or retain unknowns
where no justified bound exists. Keep the actual original numerical operator
separate from rational Euclidean geometry; a proposal allowance is not a source
gate. Before any ONE bounded new evaluation, declare the complete construction,
independently bind all NEW maps/returns and make every numerical/unknown boundary
decision durable. No solver/pose method is currently selected or queued. Do not
swap/tune/restart the completed fixed-prefix conic model, repair its rejected
point or use any old fullpixel/heldout failure as fitting feedback. No clause,
vertex/rank/axis bank, finite pose search, partition/cut loop or budget sweep is
authorized by this symbolic representation.

F/separate unqualified components/136missing robot anchors and commands/31of32
heldout failures/model/source authority/parameters/original caps and tolerances
remain unchanged. Object changes alone cannot clear8right-self failures. Any
actual new reference requires fresh WHOLE151 source/URDF/all object/root/finger
motion, FULL1mm native hand/object/support AND self, full swept, actual motor/
grasp and whole-source qualification before GPU/PPO. The owner trajectory goal
remains unachieved; this research checkpoint does not complete it.

## Conditional numerical source operators, 2026-10-20

Three NEW serial guarded CPU workers succeed; nine exact source/plan/public
reference metadata files are code-only under `source_numerical_operators`.
Ten NEW analytic contracts pass. Production runtime/tests and all original
source/physical authority remain unchanged. No episode pose, source/NN/raster,
solver, native position, physics, GPU/PPO or trajectory qualification occurs.

NEW exact rational interval recurrences cover the original multiply/divide/add
projection graph under stated finite round-to-nearest binary64 assumptions.
Strict near-plane/singular/overflow/missing cases stay UNRESOLVED; a quantizer
cell crossed by an outward interval is not assigned a convenient integer. NEW
selected-vertex leaf-norm bounds retain zero and underflow behavior, signed
coordinates and a fixed128bit outward root certificate grid. The grid is not
a candidate tolerance. No returned norm/radius/weight is clipped or repaired.
A separate control shows a selected-vertex norm check can pass even when that
vertex is not globally nearest. Leaf arithmetic does NOT prove KD search.

Public SciPy1.16.3 source and the installed binary are SHA-pinned privately,
without installation/build/global environment changes. Matching a source tag
or synthetic returns is not a compiler/libm/KD-search binary equivalence proof.
The source norm accumulates squared differences before a sqrt; tree pruning
has its own floating operations and remains outside the leaf enclosure claim.
Official source: https://github.com/scipy/scipy/blob/v1.16.3/scipy/spatial/ckdtree/src/query.cxx

NEW operator-scope bindings record a distinction already present in the saved
code: original fullpixel observed_fit calls query(k=1 default) and NumPy matmul;
the declared EVEN-training statistic calls k=2 for first/second-neighbor and
tie diagnostics and uses einsum. Both use the conceptual nearest-vertex median,
but exact numeric k1/k2/search/matmul/einsum identity is NOT asserted. All actual
old sources/returns remain unchanged. The prior free-rank rational identities
and rounded two-rank controls retain their scopes; they do not replace the
original fullpixel scorer or its original1e-10 comparison.

The NEW cache alias is C0-(V*R^T+t) in exact saved-binary-input arithmetic.
All2,727,966signed source-visual vertex components are retained losslessly as
integer numerators/common dyadic denominators. All are exactly nonzero; maximum
absolute difference is1324166867436149043/
20769187434139310514121985316880384m, approximately6.375632e-17m. This is a
representation difference, not camera/source accuracy, physical improvement,
data repair or a changed tolerance. Cached C0 and the ideal affine formula
remain distinct. No old source projection/NN/raster/FK/native audit is replayed.

ONE151clock constructor derives1,812NEW nonnegative conditional camera error
coefficients: e_i(H)=alpha_i0+sum_j alpha_ij H_j. All H halfwidth symbols remain
UNASSIGNED; no20mm/8mm or other object box is introduced. The bound includes
the exact cache alias, inherited BY-HASH dot-order error constants and the new
cache-add rounding. Finite-RN operator assumptions remain explicit. Independent
per-operand dyadic exponent arithmetic verifies every2,727,966component, and
independent affine-polynomial propagation verifies all1,812coefficients exactly.
Both complete numerator returns,2distinct operator bindings and151ordered
BEFORE/constructor/AFTER pairs are durable/private. Old source data/controls/
roundoff proofs/free-rank maps/motion authority are reused BY HASH, not rerun.

The NEW acceptance protocol seals actual NEW proposals/constructor/solver and
intermediate returns before queries. F zero queries reuse saved cache. Actual
nonzero training k2/full-raster/median/quality outputs stay separate from optional
conditional enclosures. All151hard training restrictions and ALL6ORIGINAL object
motion gates/actual p95 must pass. Original fullpixel k1/matmul source gates
remain POSTconstruction, with original tolerances; no training certificate or
interval substitutes for their actual returns. A rejected fullpixel/heldout
point is not refitted or promoted. Global KD/compiler/sqrt/raster binary
equivalence and source accuracy remain unresolved outside the stated controls.
The protocol does not require claiming those global identities to run a NEW
declared candidate through the actual unmodified operators.

NEXT declare ONE genuinely NEW bounded hard-training-source-filtered full-clock
motion/geometry construction, after NEW complete-construction/conditioning and
actual-return acceptance contracts plus independent NEW matrix binding. Use
actual changed-rank/nearest returns and discrete full-triangle training source
tests as the hard filter; no invented classical IoU derivative or fixed-rank/
vertex/source-face truth. Conditional enclosures may retain unknowns and may
not falsely accept a reference or exclude the family. No method/solver/pose is
yet selected or queued by this phase. Derive from original definitions and
saved training authority, never old rejected vectors/fullpixel/heldout failures.
No fixed-prefix/axis swap, old algorithm retry/warmstart/tolerance/weight/profile/
limit sweep, finite pose/vertex-rank/axis bank/cut/partition loop or budget
continuation. F/separate components/all missing commands/source uncertainty/
parameters/gates stay unchanged. Any real changed reference needs whole151
source/URDF/all object/root/finger motion, FULL1mm native hand/object/support AND
self, full swept, actual motor/grasp and whole-source qualification before
GPU/PPO. Object changes alone cannot resolve8right-self failures. The owner
trajectory goal remains unachieved; this checkpoint does not complete it.

## Bounded actual-source-filtered object construction, 2026-10-20

Five NEW serial guarded CPU workers succeed; fourteen exact code/plan/helper
files are code-only under `hard_source_filtered_motion`. Six NEW complete-
construction contracts pass, including two NEW analytical tiny QPs. They bind
the coupled step/clearance phase, source search plane, whole-SOC conditioning,
raw-step versus line-trial identities and sealed proposal/nonzero-cache scopes.
One actual synthetic raster/median example improves the median while worsening
IoU; the actual hard filter rejects the solved QP direction. It does not invent
a classical IoU derivative. Old controls/data are reused BY HASH, not rerun.
Production runtime/tests, original source/physics authority and gates stay fixed.

The ONE declared method starts only from F zero object translations, with F
robot commands/object rotations fixed. Median chosen-gradient rows are SEARCH
predictors, including branches at ties; they are not source certificates or
measured correspondences. Actual training NN/ranks/full-triangle silhouette/
quality are trial-filtered. All six ORIGINAL object-motion gates/actual p95 and
timestamps are mandatory. The model has454columns/19,926rows:18,730nonnegative
rows and299SOC4cones, including17,520initial geometry search branches,151median
predictor and151strict-depth predictor rows. Positive row/whole-cone scaling is
conditioning only, not source confidence, a score weight or a candidate tolerance.
Independent complete preflight binding passes all raw/conditioned rows/objective,
151source-stat/gradient/depth copies,17,520cached initial branches and299motion
cones. Later models would require independent binding before EACH constructor.
The first model and zero-F source/geometry are reused BY HASH, never re-queried.

The declared limits are8major steps/8trials/3fixed internal line fractions
1,.5,.25/160seconds, one thread; subproblems use100iterations/10seconds/default
tolerances. A20mm world-translation box,1mm step box, bounded phase slack and
rho+.5*1e-3*||step||^2 are algorithmic choices, not original object/robot boxes,
source caps, physical parameters or gate changes. Raw QP x/z/s are never
clipped/normalized/repaired. A line fraction creates a separately sealed NEW
trial. Source returns must match that proposal identity, every clock and the
correct zero-cache/nonzero actual-query scope. Actual source and motion pass,
algorithmic box and strict union-clearance-proxy descent are all required to
accept a method step. No best/intermediate or failed-point selection is allowed.
No fullpixel/heldout/old60failure value enters fitting, weights or selection.

The SOLE bounded episode returns
UNRESOLVED_no_hard_source_motion_feasible_descent after ONE Solved QP,
20actual info callbacks and3NEW trial vectors;0steps are accepted. Method wall
is21.280263seconds, complete guard23.119333seconds/peakRSS603,422,720bytes.
All raw constructors/matrices/QP x/z/s/callbacks/proposals/poses/union-axis values/
source and motion returns are durable before further work. Every one of the
three trials passes ALL6ORIGINAL151clock object-motion gates at original1e-9,
and reduces the numerical clearance proxy, but fails the actual training source
filter at78/60/37frames. Those are NEW TRAINING failures, not the old60FULLPIXEL
failures, and they are not a physical improvement or an accepted reference.
The method stops as declared, without restart/tuning/extra line fractions.

Terminal world translations and state are EXACTLY initial F. Every nonzero
trial remains REJECTED; no source-motion component/reference, original fullpixel/
heldout rescue, native-position/physics/GPU/PPO/task/policy/AUC result follows.
The numerical proxy phases (.005360372 initial; .004423318/.004891845/.005126108
for the rejected trials) are model values, not native penetration or forces.
No unaccepted point replaces F, and no source confidence/missing label/geometry/
stiction/effort/physical parameter/root box/motion tolerance/gate is modified.

Independent SAVED-return verification binds the raw QP and20callbacks, all3
sealed trial/raw-step/pose/motion/acceptance identities,14,496,552NEW returned
axis values and453actual returned-mask/distance/source decisions. Scalar axis
closures are<=5.552e-17m within explicit arithmetic guards, not native gates.
Raw returned masks provide exact I/U counts, and raw returned distance arrays
provide the actual rank/median/quality decision maps; no raster/KD/source/FK/
native/method/model/solver is replayed. All6motion decisions per trial and the
zero-accepted-step terminal identity independently verify. All failures/raw
returns stay unchanged. This rejects only this finite method's trials; it does
not exclude the translation family/task, establish stationarity or prove that
rotation/a new wrist profile is necessary. Existing8right-self/5object failures
and all F/separate component/source/heldout/missing/authority limitations remain.

NEXT investigate a genuinely NEW full object-pose construction with continuous
object rotation, not a smaller step/reweight/restart of this completed method.
FIRST reuse original object authority/6caps/clock/source receipts BY HASH and
derive only NEW changed-object-rotation source/motion/compiled-geometry maps and
meaningful contracts. Consider all3object rotation components at every151clock
about ORIGINAL F object poses; no selected axis/amplitude/time-profile/pose bank
or borrowed robot8mm/.08object bound. Object rotations would be derived PRIVATE
reference variables, never new measured observations or altered human/source
arrays. Robot wrist mode/centers/commands/repairs and all missing commands remain
fixed as authorized. Original angular speed/acceleration/p95 must be evaluated
for changed rotations; old fixed-rotation invariance is NOT inherited.

Rotating object geometry invalidates the old translation-only affine planes,
fixed axes and49071uniform-box omission certificates. Bind proper compiled
MuJoCo3.8.1object/body/geom/support-qpos0 transforms, full eligible pair coverage
and any NEW conservative broadphase before a future complete model or solve.
Do not regenerate old FK/native/source/ray/cohort audits or reconstruct unknown
returns; reuse cached input evidence BY HASH within its scope. Preserve source
uncertainty, ORIGINAL fullpixel/heldout POST-only verification, all151clocks,
all6object caps and original object1e-9/source1e-10 comparisons. No new solve/
pose/physics/GPU is queued. Source/native/motion/conditioning/actual-acceptance
contracts and independent NEW maps must precede ONE declared bounded future
evaluation. No axis/anchor/partition/cut/pathgraph/optimizer/degree/budget/profile/
tolerance sweep, old failed-point warmstart or physical-score source-truth choice.
This possible extension is not proved necessary or feasible by the rejection.
Any actual new reference still needs whole151source/actualURDF/all object/root/
finger motion, FULL1mm native hand/object/support AND self, full swept, actual
motor/grasp and whole-source qualification before GPU/PPO. The owner objective
remains unachieved; a completed research/method checkpoint never completes it.


## NEW full object-pose factors — 2026-10-20 16:32 UTC

Four serial guarded CPU workers succeeded. Thirteen NEW synthetic composition,
source-local/rotation and original angular-clock derivative, zero-cache,
near-plane, compiled-body order, full-pair and actual-return contracts pass.
No completed producer/control/audit/solve/source cohort was repeated.

ONE private factor construction declares 906 UNASSIGNED continuous variables
at all 151 clocks: three WORLD translations and three ORIGINAL-object-frame
rotation coordinates, `R_new = R_F Exp(theta)`, `t_new = t_F + delta`.
Rotation is about the object origin; all three components remain free. No new
robot wrist axis/amplitude/profile, object halfwidth, method, optimizer, solve or
episode pose is selected. These variables are derived reference variables, not
measured object/human/source observations. Original arrays/flags remain intact.

Independent NEW binding checks 8,183,898 source rotational coefficients,
782,784 compiled-body rotational coefficients, 6,723 sparse angular/midclock
coefficients, 453 training-median rotation SEARCH coefficients and all 906
column incidences. Maximum source/native coefficient discrepancy is
1.388e-17, angular coefficient discrepancy 1.677e-12 and principal-rotation
representation difference 6.037e-16. These are arithmetic diagnostics, NOT
candidate tolerances or source-accuracy proofs. Old camera/native/cache-alias
limitations remain; search derivatives never replace actual numeric operators.

All 66,591 hand/object/support cross-pair instances are retained, with ZERO
broadphase omissions. The 58 saved compiled geoms comprise nine object geoms
and 49 hand/support geoms. Cached vertices already include proper geom-to-body
placement; the NEW rotation map acts on BODY vertices/normals once. Proper
ORIGINAL free-support qpos0 is bound from static pinned MuJoCo3.8.1 MJB metadata.
Twelve body-exclusion signatures touching the object are recorded as static
metadata; all cross pairs remain retained, including excluded cases. This is
construction coverage, not a rewritten native gate or a contact/force claim.
No MjData/FK/fwdPosition/native gap/physics or old source/NN/raster query ran.
Old fixed axes/translation planes and 49,071 omission certificates are invalid
for changed rotations and are NOT inherited. Full fresh native gates still rule.

The NEW angular map uses original WORLD-relative principal rotation logs,
actual timestamps/midpoint acceleration intervals and all six original caps.
All relative principal-log pi branches and nonfinite/unavailable cases must stay
explicit; zero pi branches occur only in these initial derivative inputs.
Actual changed-rotation angular speed/acceleration/p95 must be evaluated;
fixed-rotation angular invariance is NOT inherited. Point/NN/rank gradients are
SEARCH predictors only. Actual changed-rank/NN/rounded median/quality and discrete
FULL-triangle EVEN training hard tests are mandatory. Zero F uses saved cache;
pure rotation is a NEW nonzero input. No classical IoU derivative is invented.
Original fullpixel k1/matmul/heldout source checks remain POST-only, with original
1e-10 source and 1e-9 object-motion comparisons; robot tolerances stay separate.

NEXT build ONE genuinely NEW complete full-pose motion/geometry construction
using these frozen maps. FIRST derive/bind current rotating compiled support
and any conservative omission logic, actual hard-training return provenance,
conditioning and ALL new canonical matrices. Only then declare ONE bounded new
evaluation. No old method retry, smaller-step/line-fraction extension, failed
point warmstart, axis/prefix/pose/rank/partition/cut/optimizer/weight/tolerance/
profile/budget sweep or fullpixel/heldout fitting/selection feedback. No further
generic representation study is needed without a concrete construction need.

All F repairs, both hands' missing commands, separate unqualified middle/ONE
wrist component, 136 missing anchors, heldout31/32/source accuracy limitations,
8 right-self/5 object failures, geometry/physical parameters/reserved episodes
and original gates remain. Object changes alone cannot clear self failures.
No actual candidate, source-motion reference, full trajectory/task/policy/AUC
qualifies. Any NEW actual reference needs whole151source/actualURDF/all object+
root+finger motion, FULL1mm native hand/object/support AND self, full swept,
actual motor/grasp and whole-source qualification before GPU/PPO. Owner intent
remains active and unachieved; historical goal-tool blocked observation remains
separate. Archive `object_full_pose_factors` is code-only; production runtime and
tests are unchanged. New checkpoint: `compute/object-full-pose-checkpoint-v1.json`.


## NEW complete full-pose construction — 2026-10-20 17:04 UTC

Seven serial guarded CPU workers complete: five succeed, two fail. Twenty exact
source/plan/preparation-metadata files are archived code-only under
`full_pose_hard_filter`. Production runtime/tests and private source data stay
unchanged. No completed old producer/control/solve/source/NN/raster/native audit
was repeated. The previous13 full-pose contracts/maps are reused BY HASH.

Five NEW contracts pass their scopes: moving-normal total derivative, support
ties/both-sign union, exact saved-row roundoff, actual pure-rotation synthetic
training-wrapper and zero-cache provenance. One earlier synthetic canonical
bit-identity expectation FAILS at one coefficient (5.551e-17 difference); it
remains failed. Its actual row is unchanged, and a NEW exact rational SAVED-row
two-operator normal-RN proof covers the discrepancy. No model/control replay,
coefficient/radius repair or candidate tolerance change occurs. A V1 independent
reader completes a durable all-return finiteness check, then FAILS TypeError
`abs(list)` before saving support residuals. Those unsaved values remain
UNAVAILABLE and are not reconstructed. V2 supplies a DISTINCT world-infinitesimal
coefficient proof with441 group returns saved immediately before checks, reuses
the finiteness receipt by hash and executes only the unreached canonical binding.
Separate AST preparation fails missing whitespace BEFORE module/guard/solver;
its exact raw source is retained as `.py.txt`. No valid solve is repeated.

The NEW canonical has907columns/70495rows/66895nonnegative/900SOC4cones and
477821coefficients. It couples ALL151clocks/906translation+object-local-rotation
coordinates, current rotating compiled FACE-support union predictors, source
median/min-depth SEARCH branches and linear AND angular motion. All66591cross
pairs and22428936retained face-direction values are represented, ZERO broadphase
omissions. This face-union proxy is NOT exact native penetration/complete edge-SAT,
measured contact, force, grasp or task. Current object normals rotate, including
their derivative; tied supports remain explicit. No old fixed axes/49071omission
certificates are inherited. Compiled BODY vertices/normals/proper body/geom/F
frames and ORIGINAL free-support qpos0 stay bound to the saved pinned model.
No MjData/FK/native position/gap/physics call ran.

Construction-only SI conditioning uses original object compiled radius r:
variables(delta/r,theta,rho/r), translation norm<=r/8, rotation norm<=pi/12,
rho<=r and objective rho/r+.5*1e-3*||[delta/r,theta]||². These choices are NOT
original object bounds, source confidence, physical authority or borrowed robot
8mm/.08boxes. ALL3rotation components at EVERYclock remain continuous. Original
six motion caps/actual clocks/actual angular p95 and original1e-9comparison stay.
Median gradients are predictors, no fixed NN/rank/face observations and no IoU
derivative. Actual unchanged trainingk2/einsum local median/rounded average,
NEW changed-pose matmul camera vertices, full triangle raster and IoU/quality
hard tests rule; baseline F uses saved cache only. Numerical cache-alias/build/
source-accuracy uncertainties remain. Fullpixelk1/matmul/heldout original source
checks stay POST-only, never optimizer inputs or rescue/selection feedback.

Independent NEW binding covers ALL477821canonicalcoefficients/70495RHS entries,
907columns/900whole-cone scales/objective and66591selected support/rotation rows.
Max selected-gap/rotation-coefficient discrepancy2.221e-16/2.776e-16, canonical
and RHS errors0. These are arithmetic diagnostics, NOT candidate tolerances.
Six preflight NumPy matmul RuntimeWarnings remain; all returned arrays are finite,
and warning cause is UNPROVEN. Agreement never erases warnings or proves future
reliability. No original source or native gate is certified by those bindings.

SOLE100iteration/60second/default/one-thread conic solve returnsSolved at
25iterations/1.078648835sec,26durable actual-info callbacks. ONE raw full-pose
trial only, NO line fractions/step repair/restart/best-intermediate selection.
Method22.742347667sec; guard23.755016459sec/peakRSS751878144bytes. The raw proposal
passes ALL6original151clock object-motion gates and constructor norm bounds,
but FAILS actual training source at58frames:43IoU/28median/52quality failures
(overlapping). It is REJECTED and not a reference. These training counts are NOT
comparable to earlier60FULLPIXEL or other training counts as physical progress.
The numerical union proxy still violates its1mm threshold by1.919003596e-5m;
this is NOT actual native penetration/clearance or grasp. Its decrease and solver
success never qualify source accuracy or physical success. No original fullpixel/
heldout source rescue, native/physics/GPU/PPO or score evaluation occurs.

NEW saved-return audit binds26callbacks/one raw-sealed-pose identity/151actual
training decisions/572956returned point-distance entries/all6SAVED motion-vector
decisions/66591SAVED face-union maxima across22428936returned axis values,
WITHOUTmethod/solver/source/NN/raster/FK/native/geometry runtime replay. Raw linear
slack residual4.441e-15 is a diagnostic only. Nine actual evaluation matmul
warnings are retained; source/geometry returns are finite, cause UNPROVEN.
The trial's unused geometric Jacobian is a CURRENT RIGHT-tangent derivative,
not an absolute-theta derivative; no second QP uses it. All raw returns remain.

NEXT construct a genuinely NEW hard TRAINING-silhouette-feasible full-pose model
from ORIGINAL source functional/quantizer/full-triangle definitions and SAVED F
training authority, not this rejected point, its58failures, old60fullpixel failures
or heldout values. A declared projected-integer-cell preservation construction
may provide a sufficient raster restriction; it is NOT an original gate,
measured anchor, necessary source-family condition or global exclusion. Keep
NN/rank/threshold choices free and require actual rounded median/quality hard
returns; no fixed median prefix or repaired radius. Retain all zero/boundary/
positive-depth/missing/alias uncertainties, and do NOT assume F's cached raster
is an exact freshly recomputed numeric pose/operator identity. Derive the NEW
concrete complete constraints/resource conditioning/actual-acceptance contracts,
then independent ALL-matrix bindings before ONE declared bounded evaluation.
No another soft-source-QP with tuned bounds/weights/steps/linefractions/limits,
no old model/control/audit/source replay or failed-point warmstart/repair.
No fullpixel/heldout fitting/selection, axis/pose/rank/partition/cut/optimizer/
profile/budget sweep, new source labels/confidence/geometry/physical parameters
or gates. No new method/solve/pose is currently queued. Finite failure does NOT
exclude full object poses/task, prove stationarity or require a wrist change.

F and all previous failures remain:8rightself/5object failures,136missing source
anchors/commands, source accuracy and31/32heldout failures, separate middle/ONE
wrist component and all authorized repairs. No actual reference/component/task/
policy/AUC/trajectory qualifies. Object changes alone cannot clear self contacts.
Any NEW reference still needs original whole151source/actualURDF/ALLobject+root+
finger motion, FULL1mm native hand/object/support AND self, full swept, actual
motor/grasp/whole-source qualification before GPU/PPO. Owner intent remains
active/unachieved, historical blocked goal-tool observation separate. Checkpoint:
`compute/full-pose-construction-checkpoint-v1.json` with its private SHA chain.


## NEW all-cell hard raster model — 2026-10-20 20:14 UTC

Five serial CPU guards complete: four succeed; the sole materialized constructor
attempt is interrupted by its240second CPU wall guard. Five NEW concrete Cayley,
lifted-translation/quadratic-cell/curvature/packing/actual-acceptance contracts
pass. Old source/quantizer/NN/raster/control/model/solver/native producers are
not replayed. A reader SyntaxError and absent-plan preparation occur BEFORE
worker/module/guard-state mutation; raw source is retained verbatim `.py.txt`.

The NEW variables retain all3 continuous ORIGINAL-object-local rotations at
EVERY151clock. Cayley `Q=((1-t.t)I+2tt^T+2[t]x)/(1+t.t)`, lifted translation
`eta=(1+t.t)delta`, and fixed F camera factors give a quadratic homogeneous
camera expression. This finite chart excludes pi rotations as a construction
choice, not original authority/source truth; it is unrelated to the ONE robot
wrist mode. No old robot source polynomials/charts or camera calibrations were
regenerated, no human/source observations or flags changed.

Every6022 original VISUAL vertex's saved training integer projection cells is
retained at all151clocks: four cell slots plus every strict vertex-depth slot,
and strict tracker center depth. Saturated inactive sides remain typed tautology
rows, never deleted source data. These coordinates come from SAVED training
raster geometry, NOT heldout mask labels/new measurements or the58rejected
training failures. All NN/ranks/thresholds remain free; actual rounded median,
quality, fulltriangle EVEN raster and all6 original motion gates still rule.
A same-integer-input restriction is sufficient to preserve triangle raster
inputs, NOT a necessary original source-family condition, source accuracy or
fullbinary-equivalence claim. Open/strict faces are CLOSED ONLY for the proposal
model; actual exact integer-cell equality/strict depth/source hardfilters are
mandatory before any acceptance. Numeric camera/cache aliases and unknowns stay.

A common conditional saved-binary curvature bound Lambda524288 and scale
s2118.302057564509 give `r*s*s>=Lambda` EXACTLY for stored r=.11684069766135205.
The lower-bound cone uses u=s*t and sigma>=u.u. Source rows are linear in
(eta/r,u,sigma), with a151SOC5 epigraph; no fake classical IoU derivative.
All source coefficients are proposal restrictions, not candidate tolerances.
Old allpair/linear+angular SEARCH blocks are reused BY HASH under the NEW
variable map; old nominal geometry measurements or source queries are not rerun.
All original caps/timestamps/object1e-9motion/source1e-10POSTcomparisons remain.

Complete NEW canonical:4618162rows/1058columns/4613807nonnegative/900SOC4+151SOC5,
32306054stored entries. It has4546761NEW source rows/31827327coefficient cells,
477821prefix coefficient copies, all integer targets/strict flags intact.
Direct CSC construction avoids Python row lists and saves everyframe RETURN
before further checks. Independent ALL-matrix binding verifies all source rows,
columns, RHS, prefix, lift cones/objective, exact curvature/scale certificate.
Max new coefficient/RHS differences1.137e-13/3.948e-13 lie within PREDECLARED
operationwise gamma32 bounds on operands. These are LOCAL conditional numerical
packing diagnostics, NOT candidate tolerances/source accuracy/global identities.
Nominal negative source constants0 is only an inventory; F strict/numeric/source
feasibility is NOT assumed. All2728117open/strict faces remain explicit.
Model and binding NumPy warnings remain retained with cause UNPROVEN; the checked
arrays are finite, no warnings are erased or future reliability inferred.

The SOLE declared100iter/60second/default/one-thread attempt performs lossless
zero-entry packing:32306054to32305601entries, only453explicitzeros, ALLrows/RHS/
nonzero values retained and1058exact column digests independently compared.
The240second outer guard stops before a durable constructor or solver RETURN:
wall240.418392875sec, peak observedRSS1195458560bytes<2GiB. This is a CPU WALL
cap interruption, NOT an observed RSS breach. Last durable journal boundary is
BEFOREconstructor. Constructor/internal solver state is UNAVAILABLE/UNKNOWN;
no returned vector, sealed proposal, actual source query, or solver result is
produced/observed. Do NOT reconstruct unreturned state, claim a solver status,
feasibility/infeasibility/global exclusion or extend/retry the attempt.

A separate receipt/program-order boundary audit binds guard/source/plan hashes,
all1058lossless-packing column identities, saved settings/cones and return-file
absence WITHOUT model/constructor/solver/source/old audit replay. No actual pose,
source/motion/reference/native/self/swept/motor/grasp/physics/GPU/PPO/task/policy/
AUCqualification occurs. F/separate components/136missing/31of32heldout failures/
source accuracy/physics/gates/reserved episodes and F8self+5object failures remain.
Research/model/binding/checkpoint completion never completes the owner objective.

NEXT derive ONE genuinely NEW structured all-row interface for THIS frozen hard
model so constructor startup fits the SAME limits. An exact per-frame minimum
of ALL affine residuals can express all hard cell rows without a materialized
large cone interface. Derive/prove full-row equivalence, all cone/objective/
variable/strict-flag incidences, meaningful nonsmooth-tie/search-J/conditioning/
actual-return contracts and independent NEW maps for a concrete bounded method.
EVERYoriginal row must be evaluated/retained, no finite cut/active-row/pose/vertex/
axis/rank bank, heuristic row drop or source partition change. This is a concrete
resource need; no unrelated generic arithmetic inventories. Do NOT rerun the
materialized constructor/attempt, raise240sec/2GiB, warmstart from unavailable
state, retry old source/native/solver/audits, tune source caps/weights/bounds/
profiles/linefractions/limits, fit rejected58/old60fullpixel/heldout failures or
borrow robot boxes. Keep actual integer-cell/source/rounded median/quality and
all6motion hard gates plus original whole151source/URDF/root+finger motion/FULL
1mmnative hand/object/support AND self/full swept/motor/grasp before GPU/PPO.
Object-only changes cannot clear8self; no new wrist axis/profile/repairs/missing
commands/geometry/physics/source authority/gates. Owner intent active/unachieved;
historical blocked goal-tool observation remains separate. No new solver/pose
is queued. Code-only archive `hard_raster_cayley`; production runtime/tests unchanged.

Latest read-only provider attempt20:15:40–20:15:44UTC returnsCLI1: currentBrev
status UNVERIFIED. Last ACTUAL confirmed STOPPED remains2026-10-20T17:05:28.709292UTC.
Raw attempt and reconciliation retained; attempt/update timestamps never replace
actual provider observations. No VM mutation/paidqueue/auth-cause/future lifecycle
claim. Successful current provider read is required before any future GPU work;
never start for a recovery test. Checkpoint `compute/hard-raster-checkpoint-v1.json`.


## 2026-10-20: frozen ALL-row structured method (unresolved)

Code/plans: `experiments/scored_baseline_20261019/structured_all_row_minima/`.
Private checkpoint: `compute/structured-all-row-checkpoint-v1.json`; private
SHA integrity chain: `compute/private-integrity-structured-all-row-v1.json`.
Five serial CPU guards succeed. Ten NEW contracts support a complete structured
interface for the frozen hard-raster canonical: all 4,618,162 rows / 32,306,054
coefficients, 1,058 columns, 900 SOC4 and 151 SOC5 cones. Independent NEW routing
bindings are exact. No old constructor/model/source/NN/raster/native audit or
completed controls are replayed; the interrupted materialized state stays UNKNOWN.

Per-frame minima use EVERY affine row and retain every tie/strict flag/unknown.
Averaged tie slopes are SEARCH supergradients, not universal directional
derivatives or stationarity certificates. Objective/conditioning/variables are
unchanged. The SINGLE new bounded SLSQP method starts F, evaluates ALL rows at
each input, and ends at its declared 48-query limit after 16 callbacks / 11.0593s.
No OptimizeResult/selected candidate/source query is returned. No restart, cap
extension, best/intermediate promotion, gate change or feasibility/exclusion claim.
The NEW saved-return audit verifies 221,671,776 row values, all minima/ties/cones
and SEARCH identities without solver/matvec/source replay. Three objective-matmul
RuntimeWarnings remain with cause unproven; all saved objective values are finite.
No source/native/physics/GPU/PPO or physical qualification occurred. Object changes
alone cannot clear the eight right-self failures. F/separate components/missing
136/heldout 31-of-32/source authority/parameters/original gates are unchanged.
Owner continuation remains active and unachieved; historical blocked goal-tool
observation is separate. Research or optimizer completion is not goal completion.


## 2026-10-20: direct whole-row conic splitting (unresolved)

Exact code/plans/provenance: `experiments/scored_baseline_20261019/whole_row_conic_splitting/`.
Private checkpoint/integrity: `compute/whole-row-conic-checkpoint-v1.json` /
`compute/private-integrity-whole-row-conic-v1.json`. Six serial CPU guards succeed;
14 NEW controls. ALL frozen 4,618,162 rows / 32,306,054 coefficients / 1,058
variables, 900 SOC4 + 151 SOC5, objective, conditioning and strict/open flags
remain. A direct P+A'A normal map and coordinate-row SPD proof use all rows;
proof witnesses do not drop constraints. New independent complete map and
normal/factor bindings pass conditional arithmetic checks. No old completed
control/model/source/NN/raster/native/solver/audit is replayed. The unavailable
materialized constructor state remains UNKNOWN; SLSQP is not continued.

ONE new fixed direct ADMM method starts F (rho=1/relaxation=1/32 updates/120s);
unchanged 240s/2GiB outer guard. It finishes in 16.0187s (guard17.0325s, peak
856,506,368 bytes). Sole raw step32 FAILS closed-model screening: minimum affine
-3.60345463e-9 and cone -5.02237085e-9 in model units. No tolerance repair, best
point, restart, extension, tuning, source query or physical claim. The .9995194688
objective is not a source/physical/score metric or a proof of optimality/exclusion.

NEW saved-return audit checks 443,343,552 row-state equations, all 1,051 cone
norms/branches per step, raw linear guards and exact terminal identity without
method/solver/projector/Ax/adjoint/source replay. Across the 32 saved intermediate
slacks/duals, 870/1,764 exact dyadic SOC defects are RETAINED; zero exact branch
condition failures. Three binding and three evaluation matmul warnings remain;
checked returns are finite, cause unproven. No raw array repair or force claim.
No actual pose/source/NN/raster/native/physics/GPU/PPO or trajectory qualification.
F/separate components/missing136/heldout31-of-32/source authority/physical
parameters/original gates remain. Object-only changes cannot clear eight self
failures. Owner continuation remains active/unachieved; historical blocked
goal-tool observation stays separate. Research checkpoints never complete it.


## 2026-10-20: exact raw terminal-dual phase bound (inconclusive)

Exact code/plans: `experiments/scored_baseline_20261019/exact_terminal_dual_bound/`.
Private checkpoint/integrity: `compute/exact-terminal-dual-checkpoint-v1.json` /
`compute/private-integrity-exact-terminal-dual-v1.json`. Five serial CPU guards
succeed; 14 NEW bound/sign/defect/dyadic controls plus 2 NEW independent bit-decoder
controls. No old producer/control/model/solver/source/NN/raster/native audit replay.

A NEW exact phase-support functional uses only the original actual terminal-step32
dual unchanged. Every negative orthant coordinate and whole SOC defect is covered,
without raw dual projection/clipping/normalization/rescaling/alternate selection.
Exact rational necessary coordinate bounds derive from actual frozen cone/phase/
sigma rows, not unused prior float diagnostics. Independent maps bind all 1,058
columns, 32,306,054 stored coefficient positions, 4,618,162 b entries and all
900 SOC4 + 151 SOC5 cones. Fixed96bit outward root bounds are certificate guards,
not candidate tolerances/source accuracy. Objective/conditioning/rows stay fixed.

The SINGLE exact evaluation returns normalized phase lower bound
-0.07744113078428044: INCONCLUSIVE. It does not exclude zero phase even for this
frozen sufficient construction. The scaled bound is not penetration/clearance/
physical force/task-family evidence. No optimization, new pose, source/NN/raster/
native/physics/GPU/PPO query. A NEW independent IEEE754 bit-integer audit verifies
ALL coefficient/b products (254,396 nonzero A products), 1,058 support terms, all
18,286 negative orthant entries and 1,051 whole-cone corrections exactly, without
calling the producer's denominator-bin algorithm or any solver/source operator.

F/separate components/missing136/heldout31-of-32/source authority/physics/gates and
8self+5object failures remain. No source/reference/trajectory/task/policy qualifies.
Owner continuation is active/unachieved; historical blocked goal observation is
separate. Count-only clarification: prior whole-row conic reports actually contain
11 base + 2 range controls = 13; older 12/14 labels were incorrect. Executed
sources/returns remain unchanged; no old contracts are repeated or invented.


## 2026-10-20 — joint robot/object symbolic collision construction

Code-only archive: `experiments/scored_baseline_20261019/joint_robot_object_collision`. Private checkpoint: `compute/joint-collision-checkpoint-v1.json`; integrity chain: `compute/private-integrity-joint-collision-v1.json`. Production runtime/tests unchanged.

The 465 saved robot-angle variables and 1058 saved object coordinates now share an unassigned 1523-coordinate namespace. All 31 native bodies/24 joints/58 compiled geometries and all 249603 pair-clock instances are represented, including all right-self pairs. New articulated body/support/normal ancestry and native-object gauge maps replace invalid fixed-hand geometry; all original robot/source/object clock-motion-cone restrictions remain by hash. Continuous all-vertex unit-sphere support predicates are construction geometry, not numerical native contact, force or task certificates.

Eight serial guards finish with six successes and two retained failures/interruption. Eighteen new controls pass. Native/URDF metadata identity fails (origins up to 3.469e-9m; ranges up to 1.312e-7rad); both maps remain separate and unchanged, cause unproven, no tolerance repair. A redundant compressed-I/O verifier is stopped before its fixed resource limit; unfinished return stays unavailable. New one-load bindings verify every new coefficient routing and support/native-frame incidence without replaying source, FK/native, model or solver cohorts. A preparation SyntaxError is retained separately.

No optimizer, episode pose/source/NN/raster/native position/distance/physics/GPU/PPO evaluation occurred. Wrist/middle components remain unqualified hypotheses; all F repairs/missing commands remain fixed. Original 8 right-self/5 hand-object failures persist. Whole source/URDF/motion, actual training/fullpixel-heldout POST, native+self/swept/motor/grasp gates still precede training. Owner intent remains active and unachieved; historical blocked goal-tool observation stays separate.


## 2026-10-20 — executable joint constraint/support derivative interface

Code-only archive: `experiments/scored_baseline_20261019/joint_constraint_interface`; private checkpoint/integrity: `compute/joint-interface-checkpoint-v1.json` / `compute/private-integrity-joint-interface-v1.json`. Production runtime/tests unchanged.

The new executable interface keeps all 1523 unassigned coordinates, 151 clocks, 8114 robot rows/194 original URDF source factors/38 domains/fullregion2/unknown bands, 4551571 retained object rows and all cones. Independent bindings verify all 31839917 retained A coefficients and every original coefficient routing, RHS/P/q/conditioning/source route. All compiled support vertices, moving body/direction normals and correlated ties remain; mean SEARCH slopes are not universal directional derivatives. Actual six-motion/angular-p95 and all original training/fullpixel-heldout POST/native+self/swept/motor/grasp gates remain.

Six serial guards complete5success/1failure;19 new controls pass. A tuple-key metadata writer fails after loader invocation before a saved constructor receipt; raw object state/receipt remain UNAVAILABLE, never rerun/reconstructed. New typed return controls cover only new synthetic mappings/IEEE bits/sparse pointers and explicit zeros. Three API matmul warnings are retained with finite saved returns; cause/future reliability unproven. Nested Brev fallback bookkeeping is reconciled to the latest real receipt without changing raw receipts or observation timestamps.

No optimizer, episode pose/source/NN/raster/FK/native/physics/GPU/PPO evaluation occurs. Coordinates/directions remain unassigned; no physical/reference/trajectory/task/policy/AUC qualification. F, separate components and all original failures/gates remain unchanged. Owner intent remains active/unachieved; research completion is not goal completion.


## 2026-10-20 — bounded whole-joint candidate construction stops before update

Code-only archive: `experiments/scored_baseline_20261019/joint_primal_dual_construction`. Private checkpoint/integrity: `compute/joint-primal-dual-checkpoint-v1.json` / `compute/private-integrity-joint-primal-dual-v1.json`. Production runtime/tests unchanged.

One fixed nonlinear primal–dual construction declares two updates/terminal2, three complete whole-model queries maximum, and 150 seconds under the unchanged 240-second/2-GiB guard. All 1523 joint/object coordinates plus continuous geometric direction charts, 4810140 rows, all 1245 whole cones and all 249603 pair scopes remain. Ten method controls, one exact two-update control and one lossless pair-frame roundtrip pass; a wrong 17-field label (actual16) is retained as a failure, followed only by unreached independent pointer/field bindings.

The sole method finishes one whole151 query in 37.7493 seconds then stops BEFORE metrics, dual projection, adjoint or any update: 30 finite saved SEARCH coefficients exceed the declared numerical tie envelope by at most exactly one ULP (max5.551115123125783e-17 derivative algebra units). These are not candidate tolerances. No envelope repair, gate relaxation, restart, warmstart or intermediate/terminal candidate selection occurs. Saved-return audits bind all249603 pair instances/478078 command incidences/1623929 tied slopes without geometry/source/matrix/projector/method replay. The cause beyond the saved arithmetic scope remains unproven.

No new terminal reference or trajectory exists, and no actual training NN/raster/integer-cell/fullpixel-heldout POST/native+self/swept/motor-grasp/physics/GPU/PPO gate is claimed. F/separate components/missing136/heldout31-of-32/source authority/physical parameters and original8self+5object failures remain. Owner intent stays active/unachieved; historical blocked goal-tool observation is separate.


## 2026-10-20 — saved all-tie numerical envelope proof

Code-only archive: `experiments/scored_baseline_20261019/joint_saved_mean_proof`. Private checkpoint/integrity: `compute/joint-mean-proof-checkpoint-v1.json` / `compute/private-integrity-joint-mean-proof-v1.json`. Production runtime/tests unchanged.

Seven serial CPU guards succeed; eleven NEW concrete controls pass. Independent new proof maps bind every one of 933560 saved dynamic coefficient positions and all249603 pair scopes before the sole fixed certificate evaluation. Exact saved-binary mean/product expressions retain every correlated tie and current continuous-direction differential. A conditional finite-RN operation-count bound includes absolute subnormal allowances and final-addition error; a96-bit outward dyadic encoding encloses the resulting rational envelope. Raw SEARCH J/old E, all original4810140 rows/objective/conditioning/cones/strict flags remain unchanged. This is no global installed-binary, source-accuracy or native-contact identity.

The sole certificate covers all151 frames/933560 entries with zero failures (101.7850sec, observedpeak164134912bytes). A NEW independent IEEE-bit decoder and fixed2^-1074/2^-2148 integer accumulator audit verifies all3734240 exact terms (84.0167sec, observedpeak158089216bytes), without producer proof, old method, dot/mean, source, geometry, matrix/solver or native query replay. Unsaved numeric intermediates remain unavailable; exact ideal proof products are not their reconstruction. The max saved mean/product gap4.573361891211582e-16 is DERIVATIVE ALGEBRA, not a candidate tolerance. It differs from the retained30 old-envelope defects/maxoneULP5.551115123125783e-17. No old J/E or method check is repaired and the finished method is not restarted.

No new candidate, reference, trajectory, source/training/NN/raster/integercell/fullpixel-heldout POST/native+self/swept/motor-grasp/physics/GPU/PPO evaluation occurs. F/separate unqualified components/missing136/heldout31-of-32/source authority/physical parameters and original8self+5object failures remain. Whole original qualification gates still precede training. Owner continuation stays active/unachieved, separate from the historical blocked goal-tool observation.
