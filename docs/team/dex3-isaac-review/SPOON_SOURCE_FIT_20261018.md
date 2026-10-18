# Spoon source-fit investigation — 18 October 2026

CPU-only candidates were tested on non-reserved e033/e035/e037. Existing scored
meshes, targets, checkpoints and the packed baseline were preserved. No new GPU
lease or PPO run was started for these candidates.

The archived measured-surface fitter used identical bounds and losses for each
seed: uniform shape scale plus bounded translation, with rotation and the mesh
body origin fixed. Scale estimates were 1.30, 1.1011 and 1.30 respectively. Two
estimates hit the scale bound; they are not evidence that the optimum is known.

## Other source frames

Seven uniform indices per sequence were selected. Missing source poses were
reported, not filled for scoring: e033 had four usable samples, e035 five, e037
seven. Shape scale was fixed from the seed; it was not refit at these frames.

| Episode | Mean silhouette IoU, control → scale only | Mean measured vertex distance | Original depth/jump gates |
| --- | --- | --- | --- |
| e033 | 0.2330 → 0.3378 | 20.38 → 15.01 mm | 4 / 4 retained |
| e035 | 0.4233 → 0.4798 | 16.16 → 14.54 mm | 5 / 5 retained |
| e037 | 0.3607 → 0.3832 | 15.51 → 14.31 mm | 7 / 7 retained |

Mean source-fit quality also increased in all three. Propagating the seed's
translation as a fixed body offset performed worse than scale alone and lost
two original gates on e035. That offset proposal is rejected. These are sampled
source-fit comparisons, not independent hidden-episode tests or physics passes.

## Finite support check

The scaled e037 spoon still intersects the measured support plane at some
reference frames. Translation refinement with a finite-footprint constraint
and an above-plane fitting band can improve the first frame's silhouette while
removing its visual-mesh/plane intersection. However, its full-source quality
falls from 0.1983 to 0.1355 and measured vertex distance worsens from 12.76 to
21.72 mm. It does not qualify as a complete first-frame repair or repair the sequence:
frame 100 remains intersecting after the source-fit gate rejects its proposal,
and the accepted filtered-fit proposal at frame 125 fails the original depth-support
gate. No acceptance criteria were relaxed and no trajectory was promoted.

The next geometry experiment must satisfy full-sequence source evidence and
native collision checks together, then repair hand/object alignment and motion
limits before training. A seed-only fit or a visually clear first frame cannot
qualify the entire task.
