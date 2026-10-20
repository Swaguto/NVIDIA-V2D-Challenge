# Source object pose and depth checkpoint

## Compute state

The owner switched the preferred provider back to Brev. Its CLI health check,
SSH and actual STOP command succeeded. The existing L40S was idle when inspected
and was confirmed STOPPED afterward. Watcloud has no active jobs. These CPU
studies launched no GPU job or PPO step. Future GPU work still requires a
concrete candidate, bounded runtime, an independent stop guard, private verified
backup and provider STOPPED confirmation.

## Retained source poses

The new frame-50 comparison uses the participant's fixed reconstructed spoon
mesh and the pose pool whose SHA matches the frozen temporal path: **pool v7**.
The first audit selected the older v6 pool, and its hash assertion rejected it
before any comparison. The failed producer, guard and binding diagnosis remain
available. No saved pool, temporal path, mesh or hand target was rewritten.

All four retained poses leave a source mismatch. Their projected masks cover
46–55% of the inferred source object mask. Their median inferred fingertip
distances to the visual mesh range from 18.23 to 26.92 mm. These distances are
between estimates; the fingertips are not contact labels. The source hand
remains unqualified. This limited four-pose comparison cannot exclude other
poses or distinguish source hand error from object shape error.

## Separate pose experiments

Two new one-frame experiments keep the mesh and uniform scale fixed. Neither
uses hand points in its objective. Their CPU guards enforce 240 seconds and
2 GiB RSS. All source data and historical failures remain unchanged.

| Control | Image agreement | Independent depth diagnostic | Outcome |
| --- | --- | --- | --- |
| Frozen selected pose | mask IoU 0.4817; 54.71% coverage | 310/512 held-out rays hit; hit-only median error 8.97 mm | Retained control; no physical qualification |
| Estimated visible cloud to nearest mesh vertices, selected seed | IoU falls to 0.3939 | 258/512 rays hit; hit-only median error 7.87 mm | Rejected: source image/coverage regression |
| Image-only pose, selected seed, fixed camera z translation | IoU rises to 0.6860; 72.79% coverage | 422/512 rays hit; hit-only median error rises to 27.14 mm | Rejected: depth regression and incomplete coverage |

The visible-cloud experiment used a declared even/odd checkerboard split with
512 training and 512 held-out depth pixels, Huber loss at 10 mm, and four saved
pose seeds. The image-only control used even image pixels for mask IoU and
tested odd pixels afterward; odd-pixel IoU improved from 0.4820 to 0.6847. This
checkerboard test is not an independent episode or a generalization result.
Neither control used the held-out depth values in its solver. The image-only
control used no depth values at all. Every missing ray remains in the report;
hit-only depth summaries do not excuse missing coverage.

The image-only experiment also increases median inferred tip separation from
23.24 to 31.08 mm. A visually improved outline therefore does not establish a
grip. Conversely, the cloud fit can bring an inferred fingertip within 0.5 mm
of the mesh while worsening the source silhouette. Such proximity alone is not
a valid success criterion.

## Depth and mask authority

An audit retains all 9,729 mask pixels with finite, positive, left/right
consistent cached depth. Their median is 0.4082 m, but **46 pixels exceed 1 m**,
up to 2.398 m. All 46 lie within 2 pixels of an inferred mask boundary. The
512-pixel training and held-out subsets each contain three such estimates.
The source overlay also shows a disconnected mask component on the distant
round marker. The model input was 512×320; the cached output is 1280×800.

Left/right consistency does not certify semantic foreground identity or metric
accuracy. Boundary interpolation and segmentation errors are hypotheses; this
audit does not prove either cause. Its distance cohorts are descriptive, not
new rejection rules. No source sample, validity flag, mask component or held-out
failure was deleted or relabeled.

## Next experiment and gates

Test a joint source observation model using projected visibility and first
surface depth, with fixed geometry first. A nearest-vertex cloud loss can match
the wrong surface; an image-only loss can match the outline at the wrong depth.
Treat uncertain mask/depth evidence explicitly and retain the old reports.
Do not repeat these completed fits or tune against their held-out failures.

Any changed pose or geometry invalidates prior collision certificates. A new
full reference must pass source checks, actual morphology/URDF/motion limits,
the committed **1 mm full native reference-contact gate**, swept checks and
grasp checks before PPO. There is no newly qualified reference, physical task,
policy improvement or hidden leaderboard score.

Exact code is archived under
`experiments/scored_baseline_20261018/source_pose/`, with SHA bindings and
restoration instructions. Footage, model weights, private arrays and images
remain outside Git. Independent teammate reproduction is unverified.

## Joint source fit and native validation

The translation-only follow-up reuses the image-only candidate. A training-only
first-surface median correction reduces held-out depth error to 14.94 mm, but
leaves 120/512 rays missing and increases median inferred human-tip separation
to 54.73 mm. It remains rejected.

A new joint fixed-mesh pose fit uses even-pixel mask IoU and a Cauchy
first-surface loss on 128 fixed training depth pixels. No held-out depth or
hand point enters its objective. Its finite missing-ray cost is only an
optimizer device: reports preserve infinite residuals for missing coverage.
The solver converges, and an independent check reproduces these measurements:

| Frame-50 source diagnostic | Frozen object pose | Joint object pose |
| --- | --- | --- |
| Odd-pixel mask IoU | 0.4820 | 0.6594 |
| Held-out front-surface hits | 310/512 | 417/512 |
| Hit-only median depth error | 8.97 mm | 3.17 mm |
| Hit-only p95 depth error | 28.82 mm | 8.58 mm |
| Median **inferred human-tip** separation | 23.24 mm | 6.68 mm |
| Original source-fit score, unchanged scorer | 0.3754 | 0.5814 |

The unchanged tracker gate accepts this single frame. Of the 95 missing rays,
72 are in the primary spoon component and 23 in the disconnected round-marker
component. Every sample remains in the reports. This is a source improvement,
not a complete geometry, grasp or policy qualification.

NumPy matrix multiplication emitted warnings despite finite results. Explicit
tensor contractions and independently expanded coordinate sums agree within
`5.56e-17` m and reproduce the mask/depth results. The warnings remain recorded;
their cause is unproven. Subsequent producers use explicit contractions.

Inserting only frame 50 violates the original motion limits. A declared
one-second cosine object-body correction passes all original object motion
limits and the source checks at 17 of 19 changed knots. It initially regresses
the mesh-vertex depth statistic at frames 47 and 49; that control is rejected.
Training-only constrained translations of at most 5 mm repair those two knots.
The resulting hypothesis passes unchanged per-frame source nonregression at
all 19 changed knots and all six original full-track motion summaries.
Rotations and other knots stay fixed during the two repairs.

The **object-only** reference was frozen separately and compiled fresh. The
unchanged native audits verify all 151 knots, robot FK and geometry buffers.
The committed **1 mm gate passes**: zero object knots and hand slots exceed
tolerance. Robot commands, confidence, meshes, URDFs and support stay unchanged.
No physics step, force test or PPO was run.

The actual Dex3 reference still has no carry grip. At frame 50, conservative
whole-digit separation lower bounds are **247 mm thumb, 213 mm index and
197 mm middle**. These robot distances differ from the inferred human-tip
distances above. Collision-free empty space is not a grasp. No new swept audit
was run after this grasp failure.

## Inferred object-relative command experiment

The existing optional automatic object-relative path already has conservative
association and endpoint checks. Those checks and the default gap policy
remain unchanged. A shared primitive now makes a single-object command
hypothesis explicit and validates its clock, transforms, unknown interval and
bracketing source flags. The automatic selector calls it after its existing
checks. Fingers, outside-gap commands and source validity stay unchanged. Its
output mask means inferred robot commands, not observed contact.

The explicit development hypothesis for right-hand frames `[36,70)` follows
the object between estimated anchors 35 and 70. It moves the wrist by up to
234.56 mm and reduces frame-50 inferred tip-image median error from 290.65 to
79.01 px. All five inspected inferred-image snapshots improve. However, the
hypothesis fails the original root motion limits:

| Root motion statistic | Original source cap | New hypothesis |
| --- | --- | --- |
| Maximum linear speed | 0.622 m/s | 0.955 m/s |
| Maximum linear acceleration | 13.54 m/s² | 30.95 m/s² |
| Maximum angular acceleration | 111.32 rad/s² | 137.97 rad/s² |
| p95 linear speed | 0.527 m/s | 0.651 m/s |

This command hypothesis is **rejected** and has no new native, swept or grasp
certificate. Human observations remain invalid/unmeasured in the original gap.
The first private producer's missing import path is retained separately; the
corrected producer generated the measured hypothesis. Its exact helper
snapshot is archived, and its commands match the current shared primitive
exactly. **24 focused CPU tests pass** for the primitive, automatic selector
and scored command completion. Tests preserve unknown object poses outside
the selected window and the selector's existing rigid-input tolerances.

Next fit the inferred wrist trajectory under the original motion caps and
available source evidence, then retarget actual Dex3 fingers and check native
contact, swept motion and grasp proximity. Do not enable the rejected path by
default or relax association, source or motion gates. Preserve the improved
object-only reference and failed command controls. Brev stays stopped until a
concrete candidate qualifies for bounded GPU validation.

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
