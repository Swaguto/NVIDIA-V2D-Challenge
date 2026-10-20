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
