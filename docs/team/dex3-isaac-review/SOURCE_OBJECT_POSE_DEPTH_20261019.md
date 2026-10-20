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
