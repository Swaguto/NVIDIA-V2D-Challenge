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

## Thickness sensitivity

A subsequent CPU grid separated size along the table from thickness along its
normal, with and without a fixed seed clearance offset. None of the 30
hypotheses cleared the visual finite-support intersections across all seven
sampled e037 frames. The best source-fit variant still intersected at three
samples. A fixed size/thickness correction does not resolve this trajectory.

The original grid emitted NumPy floating-point warnings despite finite values.
A second run used explicit finite-checked `einsum` for its matrix products and
matched aggregate scores within 1e-12 and all intersection counts, without those
warnings. The warning's cause is not isolated and is not evidence about GPU
replay variability. [Comparison receipt](spoon-thickness-grid-20261018.json).
No candidate was promoted or sent to PPO; Brev remained stopped.

## Rigid pose and depth consistency

A bounded six-degree-of-freedom probe tested five deterministic starts per
shape/frame on e037 frames 0, 100 and 125, using the control and uniformly scaled
meshes. Only the scaled frame-100 proposal qualified. Clearing the start/return
intersections worsened the full measured-depth fit. No full trajectory qualified.

A seven-frame source audit found nearby background median height varying from
-6.96 to +8.90 mm against the global support plane. Local plane hypotheses had
approximately 2.66–3.22 mm median held-out background residual in six frames;
the lifted frame had insufficient nearby table support. At frame 125, 68.0% of
eroded-mask foreground depth points were below the locally fitted surface by
more than 1 mm. These are stereo-estimate consistency checks, without ground
truth depth; neither plane nor source observations were replaced.

The next depth comparison freezes seven original rectified pairs, masks and
control depth maps. A versioned NVIDIA FoundationStereo ONNX worker includes
bidirectional consistency checks and calibrated disparity scaling. **It has not
run:** Brev launch attempts failed, and its stop preflight still returns an
invalid-security-token error after a successful fresh CLI login. All attempted launches ended with provider
STOPPED; no depth inference or PPO occurred. Inputs remain private.

See the [rigid/depth receipt](spoon-rigid-depth-diagnostics-20261018.json) and
[native CPU contact investigation](NATIVE_CPU_CONTACT_REPAIR_20261018.md).
