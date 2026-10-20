# Global shared-monomial conic model

Seven serial guarded CPU processes are retained: six succeed and one audit reader
fails before numerical return checks. Twenty-one exact sources/plans are archived.
Production runtime/tests are unchanged. Numerical maps, arrays, returned weights,
coefficients, geometry and certificates remain private.

Twelve new dimension-24 controls cover weighted svec, Frobenius adjoints,
spectral rounding, auxiliary boxes, exact zero/negative pivots and two tiny
analytic SDPs. One manual finite-box fixture had a wrong sign in its COMMENT
and hardcoded the expected residual instead of deriving it. The executed source
and successful process status remain unchanged; that check had limited coverage.
Three distinct new independent exact checks derive the actual rows and dual
signs, reject the wrong-sign alternative and verify the saved tiny solver's 576
rational Gram identities. Correct alias beta is +1, not -1. No old control or
solver is rerun. Generic helper scope text is not claimed for synthetic controls.

The new formulation reuses the saved 16,230-row image-conic prefix byte for byte,
with 170 zero auxiliary columns in that prefix. It adds 170 UNOBSERVED shared
monomial symbols with necessary [0,1] bounds and one 24x24 global Gram cone for
(1,10t,13Y). Bounds require the ORIGINAL exact positive unit-mass cube assumptions;
raw table masses are not normalized. Actual totals: 1,295 columns, 16,870 rows,
1,298 PSD cones. Original 549 equalities, 5,990 inequalities, 2,250 unit-box rows,
and all 1,297 old cones remain. New 340 box rows follow the old cones; then 300
new PSD rows. Independent binding verifies the whole prefix, all new rows and
2,646 exact/packed coefficients. The outward diagonal guard 1.2914127466105907e-14
is a necessary packing bound, NEVER an original candidate tolerance.

ONE 100-iteration/60-second/one-thread/default-tolerance evaluation returns Solved
in 15 iterations / 0.614900835 solver seconds. No warmstart, restart, continuation,
cut loop, degree/anchor/profile/limit/tolerance sweep or old solve occurs. Actual
x/z/s and every constructor/solver/certificate return are saved before further
work. The first independent audit reader fails with KeyError(row_start): the
rational map intentionally stores no packing row metadata. No numerical return
check occurred. V2 binds row_start/end from the saved SHA-verified packing return;
V1 source, plan, log and failed guard remain. No producer or solver is repeated.

Independent NEW-return checks find all 1,298 ORIGINAL exact rational matrices PSD
(1,180 conditional,63 joint,49 motion/depth,5 image,1 global). All 1,295 returned
values pass their exact unit boxes; original inequalities have no positive
residual. The 549 shared equalities have 534 nonzero residuals, maximum
4.2327252813834093e-16, so this is NOT an exact realization. Independent 1,298
rational-Gram dual identities / 11,916 entry products and a finite-box residual
bound over ALL 1,295 variables verify gap -2.6418807394451264e-8: NO exclusion.
Raw primal/dual/slack values are never clipped, normalized, repaired or promoted.
Separate Gram factors are proof objects, not corrected solver returns or commands.

This is only necessary global moment consistency, NOT a common supported measure,
source accuracy, deterministic robot trajectory or physical candidate. Five
nonconstant SCALAR image localizers remain unavailable. F, separate unqualified
middle/wrist components, missing136, heldout31/32 failures, all source/model/gate
and original tolerance authority stay unchanged. No native position, physics,
GPU/PPO, motor/grasp/task/policy/score success occurs. Do not rerun this model,
contracts, solver or audits; do not extend its budget. The owner trajectory goal
remains ACTIVE and unachieved.
