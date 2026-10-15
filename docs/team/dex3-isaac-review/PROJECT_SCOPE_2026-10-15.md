# Project review — October 15, 2026

Read all 22 items and 18 fields of the private
[V2D tasks project](https://github.com/users/Swaguto/projects/1).
This is an assessment of the inspected evidence, not an update to the board.

## Current position

| Board status | Issues |
| --- | --- |
| Done | #1, #2, #3, #4, #7 |
| In review | #5, #6 |
| In progress | #8, #9, #10, #11 |
| Ready | #12, #13, #14, #15, #16 |
| Backlog | #21, #22, #23, #24, #25, #29 |

The perception → reference bundle → retargeting → policy → evaluation sequence
is reasonable. Board labels alone do not establish acceptance. #7's original
done criterion covers one refined episode; #6 has reopened, stricter quality
criteria. No accepted learned policy or official score was found in the
inspected evidence. #9, #11 and #12 have no assigned owner in this snapshot.

The current Docker work preserves continuity and Dex3 replay, with three
private images and complete GPU pilots for e011/e040/e041. Those renders assign
hand/object poses directly and disable gravity/collision response. They establish
replay reproducibility, not physical manipulation or policy performance.
See [container verification](CONTAINER_VERIFICATION.md).

## Physics is the next concrete gate

The existing [full e002 physics diagnostic](../reproduction/SUPPORTED_DEX3_TASK_DIAGNOSTIC_2026-10-14.md)
already enabled gravity and contact. Its controls passed, but the complete task
failed: median wrist position errors were about 0.835/0.864 m and both objects
left the inferred support. No positive normal force was recorded at 381 sampled
source-labelled grasp-core hand/frame observations. The cause is unresolved;
the historical executor/SDK closure is incomplete and a clean repeat remains open.

Start by reproducing and isolating wrist tracking in free space: frame
conventions, target timing, force/torque application and controller stability.
Then test the complete physical task with objects moving under forces, preserving
unknown/inferred labels and existing acceptance limits. Reference object poses
must remain measurement targets rather than simulated predictions.

## Recommended order

1. **#8: stable physical control.** Reproduce the failure, fix its measured cause,
   then verify contact and object retention over a complete episode.
2. **#12: one training-ready bundle.** Define the adapter against the actual
   CHORD motion/task schema and chosen embodiment. This can proceed alongside
   #5/#6. Resolve the board's G1/Dex3 versus Sharpa task assumptions before training.
3. **#10/#11: one complete policy pilot.** Test the checkpoint through the official
   evaluator before long training. The inspected kit defaults to `flash_chord`,
   while the board discusses RSL-RL/PPO; verify compatibility. Evaluate an
   unassisted physics task, not the collision-disabled reconstruction stage.
4. **#5/#6/#15/#29: cover the scored roster and meshes.** Maintain per-episode
   quality reports, metric scale, object-slot order and mesh/body-frame consistency.
5. **#13/#14: robustness and stretch experiments.** Start after a working baseline.
   Keep #16 provenance, report and release work running throughout.

Assign owners to #9/#11/#12 and use explicit evidence for each acceptance gate.
#23's implicit right-depth scale remains a pre-leaderboard risk to audit in actual
callers; this does not establish that every current run used the wrong scale.

## Official coverage differs from the preview gallery

The [official submission kit](https://raw.githubusercontent.com/nvidia-isaac/video_to_data/main/docs/v2d_challenge/assets/v2d_submission_kit.zip)
inspected October 15 lists **20 scored episodes and 36 episode/object mesh pairs**:
16 two-object episodes and four single-object episodes. `white_pot` appears in
six, and `wooden_spoon` in ten. The older counts in #5 need reconciliation;
#29's wooden-spoon geometry gap affects ten scored pairs.

Scored episode IDs are 1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30,
32, 33, 35, 36, 37 and 38. The twelve Dex3 gallery episodes are development
episodes and do not overlap this roster. A one-episode policy pilot is useful
locally; a complete official submission needs the full required artifact roster,
including reconstructed meshes and policy rollout provenance. Missing e039
development caches do not directly remove an episode from this scored roster.

Track 3 scores object rollout and reconstructed geometry. It has no independent
hand-flicker score; hand/contact errors matter through the resulting policy and
object motion. See the existing [scoring scope review](../reproduction/SCORING_SCOPE_2026-10-13.md)
and [official challenge](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/).

No board status, issue, PR merge or official submission was changed by this review.
