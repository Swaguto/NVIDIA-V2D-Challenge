# Identical collision geometry for policy comparisons

The first e003 metric-anchor screen completed training/evaluation but failed its
post-run comparison. Object trajectories, source confidence and clocks were
exactly equal to the visual-anchor control. One convex decomposition differed:
the control contained 32 hulls; rebuilding the same surface produced 31.
The result cannot establish the effect of the anchor policy alone. It remains a
failed comparison with full private outputs preserved; physical MP-SR is zero.

Image and code pins alone do not make a regenerated decomposition identical.
Team reproduction should reuse the complete frozen pack, including its
SHA256-bound collision cache.

## Paired experiments

`run_scored_policy.py` now accepts an optional private `collision_seed_pack`
configuration path pointing to a previously frozen control pack. Before reuse:

1. Verify the complete seed file closure and reject extra/unbound hull files.
2. Require exactly matching ordered object metadata and byte-identical object,
   robot and support assets in the newly prepared candidate.
3. Mount the seed read-only and use the pinned freezer's existing `--seed-cache`.
4. Require identical resulting cache paths, runtime commit and cache bytes
   **before training**. Any regenerated or changed hull fails this comparison.
5. Run the normal native preflight and initial/full-reference contact audits.

Robot commands may differ for an anchor experiment. Neither reuse nor passing
the cache comparison establishes a valid grasp or physically valid reference.
This does not change the existing default for a fresh reconstructed task.

The next controlled e003 comparison must use this path. Do not promote the
earlier metric-anchor screen or interpret its AUC difference as causal evidence.
