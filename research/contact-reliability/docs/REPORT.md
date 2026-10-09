# Reliability aware contact guidance for egocentric skill transfer

Draft structure only. No measured robot results are available yet.

## Research question

Can reliability signals from egocentric reconstruction improve downstream robotic
contact guidance under the same environment-step budget?

## Method

Describe the pinned NVIDIA baseline, diagnostic export, development-only error
normalization, bounded contact weights, and the conservative hand-level treatment
of objectives that merge object bodies. Compare against CHORD; do not present
contact-wrench guidance itself as a new contribution.

## Experimental protocol

Record official task/metric versions, permission and split policy, leakage groups,
development and holdout membership, seeds, environment steps, hardware, runtime,
memory measurement method, human intervention, and failure accounting. Explain
any deviation from the planned three-seed comparison.

## Results to populate

| Variant | Official metric | Completion rate | GPU-hours | Peak process memory | Intervention |
|---|---|---|---|---|---|
| Baseline | Unmeasured | Unmeasured | Unmeasured | Unmeasured | Unmeasured |
| Reliability | Unmeasured | Unmeasured | Unmeasured | Unmeasured | Unmeasured |
| Constant mean weight | Unmeasured | Unmeasured | Unmeasured | Unmeasured | Unmeasured |
| Temporally shuffled weight | Unmeasured | Unmeasured | Unmeasured | Unmeasured | Unmeasured |

Include per-sequence and per-seed results. Give uncertainty with its sample size
and aggregation unit, not a false impression that every frame is independent.
Report negative findings, occlusion/contact failures, and compute spent extracting
reliability. Do not equate improved reward or reconstruction appearance with skill success.

## Reproducibility and limitations

Link frozen configs, commit and source hashes, dependency versions, input checksums,
checkpoints, official evaluator logs, and reproduction instructions. Explain absent
signals, calibration sensitivity, conservative multi-body weighting, hardware limits,
and why no measured force recovery or arbitrary-robot transfer is claimed.

## Release and submission

Confirm required research license and third-party redistribution terms with the team.
Record official artifact checksum and acceptance receipt. Code readiness alone does
not establish leaderboard eligibility.
