# Direct whole-row conic splitting (2026-10-20)

Exact code/plans and pointer provenance only. No private array/result upload.
Six serial CPU guards succeed. Fourteen NEW controls cover split signs, whole
product projections/Moreau signs, adjoints, positive definiteness, actual-return
ordering, resource limits and two NEW finite-norm stop-precondition checks.
Previous controls and model/source/NN/raster/native cohorts are not replayed.

The frozen 4,618,162-row / 32,306,054-coefficient / 1,058-variable canonical,
900 SOC4 + 151 SOC5 cones, objective, conditioning and strict/open pointers
are unchanged. The direct normal system P+A'A uses EVERY row. Existing
coordinate rows give an exact factored positive-definiteness bound. They are
proof witnesses only; every other row remains used. No diagonal loading,
constraint drop, bank, cut, parameter adaptation or source gate change.
Independent new coefficient maps bind every source frame and nonsource row.
Normal/factor roundoff checks are conditional diagnostics, not candidate gates.

ONE NEW direct ADMM method starts F with fixed rho=1, relaxation=1, 32 updates,
120-second method limit and the existing 240-second / 2 GiB outer guard. All
intermediate adjoint inputs/returns, raw linear solves, Ax, projection inputs,
slacks, duals and decisions are durable before checks. The whole-cone projection
is an algorithm operator; raw x/slack/dual returns are never repaired/promoted.
The final actual step32 only is retained, with no restart, extension or tuning.

The method finishes 32 steps in 16.0187 seconds, guard 17.0325 seconds and
856,506,368-byte peak observed RSS. Its raw closed-model screen FAILS:
minimum affine -3.60345463e-9 and whole-cone -5.02237085e-9 (model units).
These are failures, not tolerance-repaired passes. Objective .9995194688 and
primal residual are numerical diagnostics, not source accuracy/penetration/task
or hidden score. No object pose/source/NN/raster/native/physics/GPU query follows.
The cap is not convergence, optimality, stationarity or infeasibility evidence.

A NEW independent saved-return audit checks 443,343,552 row-state equations,
all 1,051 cone norm/branch cases at all 32 steps, linear guard returns and final
raw identity without method, solver, projector, Ax/adjoint or source replay.
Across all intermediate returns, 870 slack and 1,764 dual SOC exact dyadic
sign defects remain retained; there are zero exact branch-condition failures.
No raw state is clipped or corrected. Small state-identity error bounds remain
local numerical diagnostics. Three binding and three evaluation matmul warnings
remain; checked saved arrays are finite, but cause/future reliability is unproven.

No reference/component/trajectory/task/policy qualifies. F/separate components,
missing commands, source data/confidence, heldout failures, physics parameters
and all original gates are unchanged. Object changes alone cannot clear eight
right-self failures. Research completion is never owner-goal completion.
