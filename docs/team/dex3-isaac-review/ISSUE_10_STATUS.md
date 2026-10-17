# Issue 10: implementation status

Compared with [CHORD RL baseline on the first task, issue #10](https://github.com/Swaguto/NVIDIA-V2D-Challenge/issues/10), read October 17, 2026.

The project board currently marks the issue **In progress**; the issue is open. **The issue is incomplete.** Current Dex3 grip experiments support its training work, but do not establish a successful CHORD baseline or official accuracy report.

A baseline can have poor scores and still be a useful, complete baseline report. The issue's stated completion criterion is a trained policy plus the first accuracy numbers on the Scorekeeper board. Our stricter physical-grasp checks are a separate quality gate; they must not indefinitely replace implementing and reporting that baseline.

| Requirement | Current evidence | Remaining work |
|---|---|---|
| Reference bundle in Isaac Lab | e002 inferred Dex3 targets, object trajectories and frozen collision geometry load in the pinned official FlashCHORD runtime. | This is Newton/MuJoCo-Warp with JAX PPO. The Isaac Lab/RSL-RL entry point named in the issue has not been implemented by this work. Record which runtime the team will use for the baseline. |
| Task, relative-object, imitation and contact-wrench rewards | Object and hand tracking, joint imitation, measured robot pad support, distributed contact wrench and carry rewards are implemented. | No dedicated multi-object relative-pose reward is in the current experiment. Unknown human contact labels remain disabled; measured robot wrench shaping is not evidence of an exact reproduction of the paper's full reward stack. |
| Decaying virtual object controller, trajectory resets, stabilization | Staged mid-trajectory reset probabilities are implemented. Recent experiments use zero virtual object assistance throughout. | The requested assisted-to-unassisted curriculum and stabilization schedule are not validated. Current full evaluations use frame-zero starts and zero settling steps. |
| PPO policy | The latest actor-initialized run completed 3,000 new iterations, 24,576,000 new environment steps, separately from its parent's 16,384,000 steps. Checkpoints and measured readbacks are backed up privately. | The full physical pot/lid grasp still fails. Broad tracking success is insufficient to establish retention or correct contact. |
| Accuracy numbers for the Scorekeeper | The actual kit wrapper loads and evaluates the extension checkpoints; diagnostic tracking and contact audits are recorded. | AUC, SP-SR, MP-SR and MPPE have not been delivered as the requested Scorekeeper baseline report. e002 is a development episode outside the scored roster. |

## Current activity

Recent contact-model, wrist-controller and bounded motor-program fitting runs are **physics diagnostics, not additional PPO training**. Parameter-search simulation steps must not be reported as learned-policy training steps. Four/eight-world candidate fitting and full solver-configuration binding are new safeguards; their GPU checks are recorded separately from physical acceptance.

## Next baseline milestone

1. Freeze a documented runtime, action/observation contract and reference bundle. Keep the existing Isaac render and FlashCHORD physics evidence labeled with their actual backend.
2. Implement and verify the missing curriculum and reward/metric requirements against that runtime. Keep unknown contact labels explicit and distinguish any robot-contact surrogate.
3. Train a controlled PPO comparison with assistance decaying to zero, retain the unassisted evaluation and grasp gates, and preserve checkpoint/step lineage.
4. Produce the requested evaluator metrics and an independent physical readback. Use the results to decide whether the first baseline satisfies the issue; no completion or hidden-performance claim follows from a reward curve alone.

[Measured training and reproduction](GRIP_LEARNING.md) · [all measured results](grip-learning-results-v1.json) · [official wrapper compatibility](OFFICIAL_EVALUATION_COMPATIBILITY.md).
