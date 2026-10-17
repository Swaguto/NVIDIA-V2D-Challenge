# Current grip diagnosis

Useful learning gains are demonstrated on e002; a secure full-task physical grasp is not.

## What the comparisons establish

- Increasing wrist residual range from 12 mm to 35 mm helped tracking, but did not alone teach lifting. Training-only assistance enabled full-height lift.
- Estimated human surface hints disagreed with native convex collision faces. The explicitly estimated, collision-compatible CWS path reduced worst first-carry slip from **74.9 mm to 33.2 mm**, rotation from **42.8° to 25.2°**, and raised required-pad support from **85.2% to 99.5%** in the matched four-world readbacks. It still exceeded the 10 mm / 5° retention and 2 mm sampled-depth gates.
- Actor-only continuation reset the critic, optimizer, learning rate and exploration. Its first PPO mean KL was about **2.1**, versus a desired **0.005**. That run and its matched stability-weight-8 variant regressed. Their changed reset regime confounds attribution to the initialization alone.
- Full-state continuation preserved the complete learner and the original training recipe. Its initial KL fell to **0.02075**, but another **12,288,000 steps** still produced **42.1 mm / 26.1°** first-carry slip/rotation. Intermediate checkpoints also failed. A smaller initial policy jump did not solve retention.

[Measured controlled policies](ESTIMATED_CWS_RESULTS.md) · [retained checkpoints](RETAINED_PPO_CHECKPOINTS.md) · [learning-state diagnostics](ppo-learning-state-regression-audit-v1.json).

## What remains unresolved

The fingers can stay loaded while the pot rotates and slides within the grasp. The best policy's contact close-ups show loaded handle contacts; force continuity alone does not establish a stable carry. In an offline inscribed point-contact friction model, recorded contacts could balance gravity at **97.8–100%** of first-carry samples where table support did not bear weight. This conditional calculation excludes motor feasibility, dynamics, compliance and torsional friction; it does not certify that the commanded grasp can succeed.

The saved first-carry reward breakdown assigns about **37%** of positive mean reward to contact guidance and **2%** to relative-grip stability. A trade-off that tolerates excessive rotation is a hypothesis supported by that breakdown, not isolated causal proof. A relative-pose stability term already exists; it is not missing. The failed weight-8 comparison shows that increasing one weight is insufficient under the tested actor-only protocol.

The original training recipe also uses 20-step reset assistance while independent evaluation uses none. Geometry remains an approximation: hints are estimated from incomplete human observations, and their projection onto coarse convex surfaces does not establish accurate handle geometry. These are remaining distribution and modeling uncertainties.

[Static-capacity diagnostic and limits](grip-estimated-cws-static-capacity-summary-v1.json) · [saved reward balance](grip-estimated-cws-reward-balance-v1.json) · [contact method and limits](ESTIMATED_CONTACT_GUIDANCE.md).

## Next experiment

Keep the 2,000-iteration CWS parent. Isolate carry retention and rotation under the existing motor and friction limits, then test a deliberate gradual removal of training reset assistance with preserved learner state. Validate any objective change against the existing full-task gates before spending on another long run. A new recipe must record its changed contract; it must not be called an unchanged exact resume.

After a secure e002 grasp, repeat independently and on separately prepared episodes before training the scored roster. e002 is outside that roster. No hidden-episode success is established. The backend here is FlashCHORD/Newton/MuJoCo-Warp plus JAX PPO; the offline readback videos are not Isaac Sim renders.

Sessions 86–93 are stopped with hash-verified backups. Latest confirmation: 2026-10-17T22:41:27.756047+00:00. Conservative cumulative compute bound **$34.3920 of $40**, not an invoice. No restart is scheduled. Inputs, videos and checkpoints remain local for private transfer.
