# Paired hand-anchor screen — retain the visual default

The second e003 screen reuses the visual control's exact frozen collision cache.
Object trajectories, source confidence, frame clock, object/robot/support assets
and collision bytes are verified equal. Only robot command anchors and their
derived FK change. Both runs use 100 PPO iterations / 153,600 environment steps,
seed 42 and the same immutable image.

| Native readback | Visual control | Metric-only candidate |
| --- | ---: | ---: |
| AUC | 0.16048 | 0.11817 |
| MP-SR | 0 | 0 |
| Mean object position error | 0.1056 m | 60.9214 m |

The candidate contains a large finite object-motion outlier. It is not promoted;
the visual-anchor default remains. A single screen is not an independent
training-seed confirmation. The earlier screen that rebuilt different collision
hulls is an invalid single-factor comparison and remains preserved separately.

The paired screen's 2,283 files are privately hash-verified. Brev stopped at
11:32:27 UTC after 353.5 seconds. These local metrics use inferred references and
must not be described as leaderboard or hidden ground-truth scores.

## e020 reference hand/object incompatibility

A separate CPU diagnostic resets both hands and objects to six exact reference
knots. All native hand body FK is verified against the URDF and declared joint
order. At frame zero, index and middle finger collision shapes intersect the cup
by approximately 12.3/20.8 mm. At frame 80, during a missing right-hand observation,
pad-to-can gaps exceed 10 cm. These are native geometry queries with no physics
step; contact forces and grasp stability are not certified.

The earlier object/support audit deliberately excluded robot contacts. Its empty
contact list did not establish an initial grasp. A longer e020 continuation is
prepared but deferred while these incompatible references are repaired. No new
GPU continuation steps have been run.
