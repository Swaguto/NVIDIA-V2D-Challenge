# Scored baseline progress — October 18

## Verified coverage

- All 20 source episodes and all 5,181 source frames are inventoried.
- 25 of 36 own mesh candidates have verified source and mesh closures.
- 12 complete short baselines pass the standard submission episode contract and
  untouched kit trajectory reader, including the complete eight-world grid.
- These episodes are 1, 3, 5, 6, 8, 10, 16, 19, 20, 25, 28 and 38.
- e015 has exported geometry and a full reconstruction bundle, but training
  stopped on nonfinite metrics after 119,808 steps. It is not a complete policy.
- All 12 complete scored baselines have MP-SR zero. No physical task is accepted.

The stage-separated machine-readable inventory is
`scored-coverage-progress-20261018.json`. Geometry availability is counted
independently from policy completion. A valid trajectory file does not mean a
successful physical task.

## Latest complete pilots

| Episode | Native local ADD AUC | MP-SR | Object position error | Main observed limitation |
| --- | ---: | ---: | ---: | --- |
| 16 | 0.58797 | 0 | 0.10376 m | Rack/table reference intersection; failed spoon handling |
| 28 | 0.51083 | 0 | 0.09170 m | Rack/table reference intersection; failed pot handling |

Each used 100 PPO iterations, 153,600 environment steps and seed 42. The native
readback evaluates eight worlds over the full source clock without object
assistance. The scores use our inferred references and are not leaderboard
ground-truth scores. A nearly stationary rack can score well while the handled
object fails, so the average AUC must not be treated as a grasp pass.

## Current repairs

1. Investigate seed-frame refinement moving good source rack poses into the
   table. The optional verified SAM seed arm retains source provenance and the
   original depth/jump gates. No promotion before native/full-task validation.
2. Correct missing-hand command drift with the optional object-relative gap
   experiment. Observed finger/object intersections need a separate repair.
3. Keep the better e002 checkpoint 2000. Independent reward-weight repeats did
   not establish a reproducible improvement. The exact paired metric-only
   anchor screen regressed and was rejected.

Remaining nonreserved baseline episodes are 30, 33, 35 and 37, plus e015
recovery. Reserve scored episodes 14, 32 and 36 for a method frozen before their
task outcomes are inspected. The all-episode packer and independent teammate
reproduction are still pending. Private inputs and gated assets remain private.
