# Issue 10: implementation status

Compared with [CHORD RL baseline on the first task, issue #10](https://github.com/Swaguto/NVIDIA-V2D-Challenge/issues/10), read October 17, 2026.

The project board currently marks the issue **In progress**; the issue is open. **The issue is incomplete.** Current Dex3 grip experiments support its training work, but do not establish a successful CHORD baseline or official accuracy report.

A baseline can have poor scores and still be a useful, complete baseline report. The issue's stated completion criterion is a trained policy plus the first accuracy numbers on the Scorekeeper board. Our stricter physical-grasp checks are a separate quality gate; they must not indefinitely replace implementing and reporting that baseline.

| Requirement | Current evidence | Remaining work |
|---|---|---|
| Reference bundle in Isaac Lab | e002 inferred Dex3 targets, object trajectories and frozen collision geometry load in the pinned official FlashCHORD runtime. | This is Newton/MuJoCo-Warp with JAX PPO. The Isaac Lab/RSL-RL entry point named in the issue has not been implemented by this work. Record which runtime the team will use for the baseline. |
| Task, relative-object, imitation and contact-wrench rewards | Object and hand tracking, joint imitation, measured robot pad support, distributed contact wrench and carry rewards are implemented. | No dedicated multi-object relative-pose reward is in the current experiment. Measured human labels remain unknown. An explicit estimated-surface CWS path now has CPU mechanics checks and collision-compatible hints; GPU validation and a 16,384,000-step controlled PPO trial are complete; it improves contact and tracking but still fails retention. The actor-only zero-assistance and retention-weight-8 continuations regressed. A complete PPO-state continuation is running with the original training recipe; every evaluation remains unassisted. This does not establish an exact reproduction of the full reward stack. |
| Decaying virtual object controller, trajectory resets, stabilization | Staged trajectory resets, decaying training VOC and20-step linear reset stabilization are implemented. GPU checks passed; the matched three-arm PPO comparison completed. | All three full-task physical readbacks failed. Assistance enabled full-height lift but retention remains poor. The zero-target training stage still includes reset assistance; evaluation disables all assistance and settling. |
| PPO policy | The earlier actor-initialized run completed3,000new iterations/24,576,000new steps, separately from its parent. Fresh matched12mm and35mm action controls have each completed2,000iterations/16,384,000steps; both failed the first pot carry. The assisted comparison also completed 2,000 iterations / 16,384,000 steps; it reached full-height lift but failed retention. | The estimated-CWS trial also completed 2,000 iterations / 16,384,000 steps, with improved tracking and required-pad contact; the full physical pot/lid grasp still fails. Broad tracking success is insufficient to establish retention or correct contact. |
| Accuracy numbers for the Scorekeeper | The actual kit wrapper loads and evaluates the extension checkpoints; diagnostic tracking and contact audits are recorded. | AUC, SP-SR, MP-SR and MPPE exist in evaluator outputs; the [consolidated controlled report](TRAINING_DIAGNOSIS.md) is prepared locally. Nothing has been posted to Scorekeeper. e002 is a development episode outside the scored roster. |

## Current activity

The completed matched action-limit/VOC trials contain 49,152,000 new PPO steps across three arms. Earlier contact-model, wrist-controller and bounded motor-program fitting runs are **physics diagnostics, not additional PPO training**. Parameter-search simulation steps must not be reported as learned-policy training steps. Four/eight-world candidate fitting and full solver-configuration binding are new safeguards; their GPU checks are recorded separately from physical acceptance.

## Next baseline milestone

1. Freeze a documented runtime, action/observation contract and reference bundle. Keep the existing Isaac render and FlashCHORD physics evidence labeled with their actual backend.
2. Preserve the distinction between estimated human surface hints, measured robot forces and unknown measured human labels. The CWS GPU checks and controlled trial are complete; retain their measured failures as well as improvements.
3. Evaluate the complete-state PPO continuation against the unchanged unassisted grasp gates. Actor-only continuation reset the critic, optimizer, adaptive learning rate and noise; its first-update KL was excessive and its grasp regressed. [Diagnosis and reproduction](EXACT_PPO_RESUME.md).
4. Produce the requested evaluator metrics and an independent physical readback. Use the results to decide whether the first baseline satisfies the issue; no completion or hidden-performance claim follows from a reward curve alone.

[Measured training and reproduction](GRIP_LEARNING.md) · [all measured results](grip-learning-results-v1.json) · [official wrapper compatibility](OFFICIAL_EVALUATION_COMPATIBILITY.md).

[Estimated contact guidance and collider mismatch](ESTIMATED_CONTACT_GUIDANCE.md) · [matched trial reproduction](CURRICULUM_REPRODUCTION.md).
