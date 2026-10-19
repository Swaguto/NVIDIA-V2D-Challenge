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
