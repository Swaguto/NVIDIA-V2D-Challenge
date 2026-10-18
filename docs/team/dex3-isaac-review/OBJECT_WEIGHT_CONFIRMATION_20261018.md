# Object-tracking reward confirmation — 18 October 2026

**Retain checkpoint 2000.** Doubling `object_keypoints` from 8 to 16 improved
paired mean AUC in both exploration forks, but does not yet establish a
repeatable improvement over the retained parent's approximately 0.86549 mean.
No candidate is promoted or claimed to solve physical grasping.

| Fork | Control mean AUC | Weight-16 mean AUC | Control MP-SR | Weight-16 MP-SR | Weight-16 position error | Weight-16 relative position error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 0.86067 | 0.86749 | 91.67% | 95.83% | 17.21 mm | 4.834 cm |
| 202 | 0.85168 | 0.86804 | 87.50% | 87.50% | 16.99 mm | 4.761 cm |

Each candidate restores the full matched fork at iteration 2000 and continues
100 iterations / 819,200 new environment steps. Inputs, runtime, resets,
observations, actions, optimizer and evaluator stay unchanged. Every curriculum
and resolved objective alias uses the same single weight change. The strict
resume lineage records the exact factor. Environment reset seed remains 42;
forks 101/202 change only exploration RNG state from a common parent, not actor
initialization. These are two exploration streams, not two fresh policies.

Each checkpoint received three complete unassisted eight-world readbacks. The
candidate AUC ranges are 0.86033–0.87181 and 0.86037–0.87544. Paired mean gains
are +0.00683 and +0.01636; the latter control contains a large-error repeat.
The candidate means exceed the retained parent by only about 0.002–0.003, within
observed evaluation variability. MP-SR improves in only one fork. All native
physical task audits remain unaccepted. Results use the development task's
inferred references and are not hidden leaderboard scores or evidence of a
universal policy.

Total new training: **1,638,400 environment steps**. Both trials and all six
readbacks completed. All 95 retained result files were SHA256 verified before
provider STOPPED at 10:18:58 UTC, after a 623.5-second guarded lease. This run
contained only the two weight experiments; a subsequently prepared scored
metric-anchor screening trial was not deployed and did not run.

Parent SHA256:
`4d91fd4164b043a41a390ee7fbb904a3f3aa6a0bb724e19fc1f2342e3f108a40`.

Candidate checkpoint SHA256 values and full lineage are retained in the private
`tracking-weight-confirmation-summary-v1.json`. No new relative-object reward,
reset change, input change or automatic longer continuation is part of this
comparison. Validate a candidate on physically feasible scored references and
repeat it independently before changing the baseline recipe.
