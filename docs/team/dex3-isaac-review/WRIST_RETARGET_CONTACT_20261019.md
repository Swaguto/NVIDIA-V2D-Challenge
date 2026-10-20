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
