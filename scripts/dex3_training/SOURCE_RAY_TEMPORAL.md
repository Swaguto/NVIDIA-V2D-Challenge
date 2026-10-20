# Source-ray geometry and finite temporal pose graphs

These helpers construct and assess hypotheses before an expensive physical
rollout. Neither helper changes the original source observations or their
missing, inferred, direct and metric flags.

## Root rotations from a point and ray

`source_ray_kinematics.ray_bound_root_rotations` intersects the sphere traced
by a fixed local point about a fixed root translation with a declared image
ray. Each intersection uses the minimum rotation from the supplied reference
that maps the local vector onto the ray. This is an explicit twist convention,
not enumeration of the full rotation family. Antiparallel ambiguity is separate
from an empty nonsingular result. Ray parameters retain the caller's convention;
the helper does not silently normalize the direction or infer measured depth.

Check original root bounds, source-image/model-orientation nonregression,
actual other fingers, native collision geometry and trajectory limits after
construction. Source-model orientation remains an estimate, not a measured
robot wrist pose. Eighteen focused tests cover sphere branches, tangency,
transforms, minimum swing, ambiguity and strict input contracts.

## Finite temporal reachability

`temporal_pose_graph.finite_temporal_reachability` accepts one finite joint-pose
bank per clock row and explicit actual joint position/velocity limits. The
component and adjacent/second-added caps apply against the original reference.
Velocity applies to actual joint angles and actual timestamp differences.

A consecutive-node pair is the state, allowing exact second-difference checks
and bidirectional trimming of the supplied graph. The result includes a
minimum-cost complete path when one exists, forward/backward pair counts,
complete-path viable nodes and every empty bank. Missing clock rows are not
skipped. Invalid positions fail instead of silently being filtered.

The graph does not certify unsampled solutions, root motion, source accuracy,
continuous swept collision, native motors, contacts, dynamics or task success.
Sixteen focused tests include an independent exhaustive enumeration, a greedy
dead end, actual clock velocities, original-reference corrections and empty
banks. Preserve a caller's original numerical tolerances explicitly.

## Retained e037 study

A joint-only linear bridge can connect the two saved isolated source/native
poses under the original finger bounds. This does not supply a source- or
collision-feasible trajectory. The first finite ray-based bank has 58 nodes
and six missing rows; its original fixed-pixel convention excludes some known
allowed source-image motion. A new construction projects into the original
image-error disks, retaining the same source thresholds. It reuses the old
nodes and produces 2,003 eligible finite nodes, but still lacks 109/112/114
and has no complete path. These finite failures are not global infeasibility.

The isolated poses are controls, not mandatory measured robot angles. No new
reference replaces the three verified components. Full original source/motion,
1 mm native/self, swept, actual motor and grasp checks remain before training.

## Common-root twist intervals

`source_root_twist.source_orientation_twist_domain` keeps the remaining
one-dimensional rotation around an already mapped local vector. Its closed-form
quaternion sinusoid describes the supplied source-model orientation error cap.
The twist preserves that robot point, not measured human depth or contact.

`partition_root_twist_domain` intersects that domain with the caller's original
world-component rotation box using numerical principal-log envelopes. It saves
every split and rejected/inside/unresolved leaf, including cut-locus or budget
uncertainty. The envelopes use outward numerical padding and are **not** a
formal libm directed-rounding certificate. Source images, actual fingers,
native collisions and full motion remain separate checks.

Twenty-two focused tests cover closed-form domains and seams, independent
rotation grids, point preservation, component enclosures, complete leaf cover,
cut-locus/budget uncertainty and strict contracts. Independent private checks
on actual e037 geometry validate selected isolated poses; their sparse insertion
still fails original joint speed and temporal caps. No new reference replaces
the three verified components, and no physical or policy success is implied.

## Research camera-cone duals

The private wrist study now uses a global positive-depth source-cone halfspace
instead of treating a fixed-plane conic separator as valid after wrist motion.
For image disk matrix `B`, radius `r` and unit 2-vector `u`, every source-cone
point satisfies `(B.T u - r * ez).C <= 0`. The tangent at a planar ellipse
support point has a positive-scale pullback to the saved 2-D separator.

With fixed translation and local point radius bound `rho`, a wrist rotation
of angle `theta` changes any unit-normal support by at most
`2 * rho * sin(theta / 2)`. This yields necessary rotation bounds, not a pose,
source accuracy, native collision or motor/grasp certificate. The numerical
root partition keeps its source-model/component constraints and every
unresolved leaf. See the code-only `wrist_cone_domains` research archive and
the wrist-contact review for exact scopes, retained failures and checks.

## Coupled continuous time algebra in research

The private five-clock index study represents actual angles as `q = 2 atan(u)`.
For an integer angle combination `Q = sum(k_j q_j)`, the polynomial phasor
`P = product(1 + sign(k_j) i u_j)^abs(k_j)` has squared modulus at least one.
After certifying the angle branch, the two signs of
`Im(exp(-i L/2) P)` and `-Im(exp(-i U/2) P)` enforce `L <= Q <= U`.
This avoids assuming a positive tangent-difference denominator.

The study binds original correction first/second differences and actual
clock velocities to the saved source-disk polynomials. Rational Taylor
intervals enclose coefficient constants for saved binary inputs. Numerical
endpoint/arithmetic guards produce necessary outer restrictions only;
original candidate gates and tolerances remain unchanged. Source interiors,
unknown critical bands and full correlated clock boxes remain explicit.

Twelve new controls and independent coefficient/angle/timestamp checks pass.
No source/time solution or exclusion is asserted; the equations precede a
new coupled construction. The exact code-only research archive is
`experiments/scored_baseline_20261019/index_temporal_algebra`. Existing
runtime helpers/tests and the verified reference remain unchanged.

## Filled-domain source/time clipping in research

A subsequent private method reuses source Bernstein grids and the continuous
time polynomials. Convex-hull projection of each filled sublevel gives necessary
coordinate bounds; shared factors contract neighboring clocks. Outward domain
and lower coefficient rounding retain continuous source interiors and unknown
critical bands. A projected interval is not a feasible source branch or a path.

The declared 24-pass scope remains unresolved. Independent blossom identities
and scalar supporting lines verify every contraction without rerunning the
method. No isolated pose is a mandatory node or accepted center. The exact
code-only `source_time_clipping` archive and wrist review record all scopes,
controls and remaining uncertainty. Original runtime/tests, reference, source
observations and candidate acceptance gates remain unchanged.

## Common-marginal relaxation and continuous positivity

A subsequent private model couples degree-four Bernstein tables through shared
joint and clock marginals. Deterministic assignments embed, but relaxed local
tables are not robot poses, source truth or a globally realizable measure.
Conditioned finite square tests leave invalid tables even after 247 proven
inequalities are added. A strict bound-audit failure is retained, not repaired
by clipping the actual returned weights.

The saved-return diagnosis finds negative exact principal minors in the
conditioned moment/localizer matrices, with exact negative polynomial-square
witnesses. Continuous positivity is the next missing necessary condition;
local positivity alone would still not certify a deterministic whole path.
See the code-only `common_moment_viability` archive and wrist review for all
solver returns, failed status, reused-control scopes and independent checks.
No reference, source authority, runtime, physical gate or score is promoted.

## Continuous cone separation in research

A new bounded method separates conditional Hankel/localizer cones using exact
rational polynomial-square planes. Ten new contracts pass. Six declared LP
updates add 384 valid planes, with no original tolerance change or pose promotion.
The fixed budget remains unresolved; continuous positivity still fails at the
last table. All matrices, vectors and unresolved cases remain private.

Independent recorded-certificate checks pass before a report serialization
failure; a report-only repair reuses those durable results without numerical
reruns. The code-only `continuous_moment_cones` archive and wrist review retain
all statuses and scopes. Local moment positivity would remain necessary only;
whole-clock source/URDF/motion and physical gates still precede learning.

## Direct moment SDP: numerical local positivity is not a path

The saved source/time/common-marginal model now has one direct semidefinite
feasibility evaluation with 1,180 conditioned PSD blocks, 1,125 entries and
14,099 canonical rows. Ten new packing/adjoint/rounding/dual-sign controls pass;
four completed controls were reused by hash after a retained map fixture failure.
A private SHA-pinned binary runtime avoids global or production dependency changes.
The earlier missing-pip failure is also retained. No old source, graph, LP, cone
iteration or completed contract is repeated. Exact code/plans are archived in
`experiments/scored_baseline_20261019/direct_moment_SDP`.

The single bounded call reports Solved in ten iterations. Independent exact
principal-minor checks find all 1,180 original moment matrices PSD, no weight-box
failure and no violated original inequality. Shared equalities retain a nonzero
2.467644e-16 residual, so the actual table is not an exact realization. No actual
weights are altered. The independently checked Gram-dual finite-box gap is
negative, -3.120771e-8; there is no exclusion certificate. Svec packing/error
bounds are necessary model arithmetic guards, not source/trajectory tolerances.

Local conditional PSD is only necessary. Mixed-variable polynomial-square
positivity, global common-measure consistency and deterministic source/time
realization remain separate. Do not promote a mean, eigenvector, Gram factor,
local table or numerical solver flag to a robot command or source truth. Any
new mixed-variable investigation must reuse the existing saved degree-four
coefficients, declare its scope, save all intermediate returns and preserve all
unknown event bands/correlations; it is not permission for a degree/anchor/solver
sweep. Verified F, unqualified middle and wrist components and all original
source/motion/native/motor/grasp gates remain unchanged. No native/physics/GPU
or whole-clock reference qualification occurred.

## Mixed squares and overlapping joint/time scopes

Existing degree-four tensor coefficients now support a separate mixed-variable
consistency audit: nineteen pair/triple scopes, sixty-three joint moment and
interval-localizer matrices. Twelve new exact analytical contracts pass. All
sixty-three actual matrices pass exact PSD with full rank. Independent checks
verify 405 cross-moment maps, 14,818 coefficients and 1,590 congruence entries.
No solver, old univariate audit or source constructor is repeated.

A new cross-clique projection audit reuses saved marginal-equality residuals by
hash. All 405 linear identities and 226 overlap comparisons verify. There are
223 exact mismatches, maximum 2.294172e-16, each accounted for by an exact
combination of saved residuals. No actual weights, tolerances or source flags
are repaired. Positive unnormalized variances in all ten variables also prevent
calling this table a deterministic assignment.

These are necessary local consistency checks. A global common measure with
pointwise support on every source/motion condition and a concrete trajectory
remain unqualified. Derive the availability of temporal/depth support-localizer
maps within the unchanged saved tensor powers before another formulation;
unknown or unavailable moments stay explicit. Do not sweep degrees or rerun
these contracts, matrices, overlap audit or earlier solvers. Exact code/plans
are in `experiments/scored_baseline_20261019/mixed_joint_consistency`; all private
results and the verified reference/source/physical gates remain unchanged.

## Pointwise temporal support is stronger than scalar averages

New weighted-square matrices use SAVED temporal coefficient enclosures and
positive-depth models. A guarded dyadic upper polynomial preserves the
necessary implication from each original temporal gate. Multilinear multipliers
fit the EXISTING per-coordinate degree-four tensors; five source-image factors
retain unavailable nonconstant localizers instead of inventing moments. All
original finite-chart, branch, depth, source, motion and candidate gates remain.

Eight new analytical contracts pass. The fixed diagnosis finds 25 failing
temporal support matrices despite positive scalar means. Five depth and
nineteen other temporal matrices pass. Independent exact error, affine,
coefficient-map and weighted-square certificates verify all results. No old
polynomial generation, algebra/PSD audit or solver repeats. No actual weight is
clipped, normalized or repaired.

This rejects the saved table as a supported motion measure, not the conditional
source/time family or task. No new reference/trajectory/physical candidate or
GPU/PPO qualifies. Exact code/plans are in
`experiments/scored_baseline_20261019/pointwise_support`; all private results and
the verified reference stay private and unchanged. Any support-aware conic
formulation is a new bounded scope requiring new packing/dual contracts first;
it is not permission to rerun or extend completed solvers or cut algorithms.

## Complete support-aware conic return remains unqualified

A single NEW direct model reuses the canonical base by hash and appends all
63 joint and 49 temporal/depth PSD maps. It has 15,840 rows, 1,125 columns and
1,292 PSD cones, with all original linear/equality/unit-box constraints intact.
Ten new weighted packing/adjoint/dual contracts pass; a first sparse-fixture
dictionary failure is retained and its unsaved return objects are not recreated.
Independent pre-solve packing checks pass. No old producer, contract or solver
runs as a preliminary step.

The one bounded call returns AlmostSolved at reduced accuracy after 14 iterations.
Independent NEW-return checks find every matrix PSD and no box/inequality failure,
but a nonzero 3.215744e-16 shared-equality residual remains. The independently
checked rational-Gram finite-box gap is negative, so no exclusion follows.
Weights are not repaired, and no exact common measure, source truth or path is
promoted. Five nonconstant scalar image-support localizers remain unavailable.

All exact code/plans are in `supported_conic_model`; private evidence, failed
statuses, F and all original source/motion/native/physical gates stay unchanged.
Do not restart, tune or extend this model or repeat its contracts/audits. A new
matrix source-cone representation would need separate equivalence, coefficient
error and tensor-capacity proofs from saved data before use; it cannot silently
tighten the original image gate or regenerate completed source fits/polynomials.

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
