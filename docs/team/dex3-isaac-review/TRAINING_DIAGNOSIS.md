# Training diagnosis and controlled comparison

Session86 ended **STOPPED**; all comparison results are hash-verified. Conservative compute bound **$31.1413 / $40**, not an invoice.

This is one development episode (e002), outside the scored roster. Each arm starts from the same fresh actor, seed42,256worlds,32-step rollouts and2000iterations: **16,384,000 new steps per arm**,49,152,000 total, plus65,536 smoke-test steps. One seed does not establish a statistically robust training improvement.

## What the experiment isolates

A changes nothing from the current12mm action control. B changes only the wrist position clip to35mm per axis. C adds training-only VOC target scales0.75→0.5→0.2→0.05→0 at iterations0/250/500/750/1000, with20-step linear reset settling. The zero-target training stage still includes reset assistance. All four-world evaluations start at frame0 and verify all343steps have zero VOC, zero virtual actuator caps, no settling and no automatic resets. Geometry, gravity, mass, friction,0.76Nm finger caps, action scales, observations, objective weights and network/optimizer settings remain matched.

The GPU gate measured a30mm request clipping to12mm under A and reaching30mm under B/C. This proves an action-space restriction; it does not prove that restriction is the sole cause of failed learning.

## Accuracy baseline

These are the pinned evaluator’s development-cohort numbers. SP-SR uses the uncentered metric. The report is prepared locally and has not been sent to Scorekeeper or posted to the project board.

| Run | ADD AUC | SP-SR | MP-SR | MPPE (cm) | Numerical failures | Full grasp gates |
|---|---:|---:|---:|---:|---:|---|
| 12 mm control | 0.6776 | 100.0% | 0.0% | 9.947 | 1 | FAIL |
| 35 mm residual | 0.7883 | 100.0% | 0.0% | 4.504 | 1 | FAIL |
| 35 mm + decaying training VOC | 0.7443 | 100.0% | 0.0% | 4.722 | 4 | FAIL |

## First pot carry: all four worlds

| Run | Lift range / reference (mm) | Worst slip (mm) | Worst rotation | Minimum pad support |
|---|---:|---:|---:|---:|
| 12 mm control | 3.3–8.6 / 99.0 | 139.4 | 16.8° | 37.2% |
| 35 mm residual | 1.5–3.1 / 99.0 | 126.8 | 14.1° | 37.7% |
| 35 mm + decaying training VOC | 101.6–105.9 / 99.0 | 74.9 | 42.8° | 85.2% |

Existing acceptance still requires retention≤10mm/5°, every required pad≥0.5N for95%of sampled frames, required lift within10mm, sampled penetration≤2mm, source tracking≤30mm/15°. Contacts are sampled at20Hz, not at every physics step.

## What remains unresolved

Earlier contact-normal audits show a weak third-finger opposition and twist about the thumb/index pinch. Broad tracking rewards and reported success rates can coexist with inadequate lift or slip. Current robot-wrench/support rewards do not reproduce the paper’s human contact-wrench-space guidance; human contact labels remain unknown. Multi-object relative-pose reward and the issue’s literal IsaacLab/RSL-RL entry point are also missing. A reward curve cannot certify physical grasping or hidden-episode generalization.

The completed [estimated-contact follow-up](ESTIMATED_CWS_RESULTS.md) improves the grip. The next trials remove reset assistance and, if needed, isolate the retention reward weight.

[Complete per-world measurements and hashes](grip-training-curriculum-comparison-v1.json) · [issue10 requirements](ISSUE_10_STATUS.md) · [earlier training and inputs](GRIP_LEARNING.md)

## Training scale

These are bounded development experiments. The pinned upstream Sharpa reference recipe uses 4,096 worlds, 24-step rollouts and 20,000 PPO iterations (1,966,080,000 environment steps), with a much longer assistance schedule. Our fresh Dex3 comparison uses 256 worlds, 32-step rollouts and 2,000 iterations (16,384,000 steps). The robots, network settings and playback rate also differ. These totals are not equivalent, and the comparison does not establish that more steps alone will solve the grip. Reference/collider compatibility, contact guidance and removal of reset assistance must be checked before scaling training.
