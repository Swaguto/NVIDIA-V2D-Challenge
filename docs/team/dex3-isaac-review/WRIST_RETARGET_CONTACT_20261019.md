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
