# Direct conditional moment SDP (research archive)

Exact code/plans for six serial guarded CPU processes. Four succeed; the private
pip absence and a contract map/list fixture failure remain failed. Four completed
contracts are reused by hash in v2 without reexecution. Ten new contracts total
include two tiny analytical SDP solves. There is exactly one new episode SDP.

The private runtime uses SHA-pinned Clarabel 0.11.1, cffi 2.1.1 and pycparser 3.11
binary wheels. Wheel RECORD checks and private import inventories are saved
privately. Production/global environments and cached assets are unchanged.
No wheels, binaries, model coefficients, arrays, solver returns or private
results are present here. Restore paths and exact hashes are in the manifest.

The single direct feasibility model reuses the saved 1,125-column, 549-equality,
5,990-inequality model, including all existing square planes. Original unit
boxes and 1,180 conditional Hankel/localizer cones give 14,099 canonical rows.
Column-major upper-triangle packing scales off-diagonals by sqrt(2), preserving
Frobenius products, per the [official Clarabel specification](https://clarabel.org/stable/api_cone_types/).
Exact rational brackets for sqrt(2) and finite unit-box coefficient errors yield
outward RHS/diagonal guards for a NECESSARY numerical representation. They do
not alter robot/source candidate tolerances. Settings are declared before work:
one evaluation, 100 iterations maximum, 60 seconds, one thread, default solver
tolerances, no restart or continuation.

Clarabel reports Solved in 10 iterations (about 0.138 seconds). Independent saved
evidence verifies every row, 25,690 linear coefficients, 26,550 PSD map
coefficients, and 5,900 exact principal minors. All 1,180 actual original moment
matrices are PSD; no actual unit-box or original inequality violation exists.
The exact shared-equality residual is 2.467644e-16. The table is therefore NOT an
exact realization of every frozen rational equality. No clipping, normalization
or table repair occurs. A separate rational-Gram dual construction and independent
finite-box residual audit have gap -3.120771e-8, hence no exclusion certificate.

A numerically feasible local PSD table is not a globally consistent common
measure, deterministic robot path, source truth or physical candidate. The
conditional region-2/fixed-unqualified-wrist family remains unresolved. Verified
F, separate unqualified middle component, human/source evidence and all original
bounds/parameters/gates are unchanged. No native position, physics, GPU/PPO,
reference or task success is claimed. Do not rerun these producers, contracts,
packing audit or solver, or sweep tolerances/degrees/limits/anchors.
