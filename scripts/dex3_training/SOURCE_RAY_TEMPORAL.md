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
