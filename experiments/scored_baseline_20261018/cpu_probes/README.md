# Preserved CPU investigation producers

These are exact copies of experimental producers, bound by
`source-manifest.json`. Executed producers and prepared, unexecuted next steps
are explicitly distinguished in that manifest. They preserve negative results and the source of the
reported geometry/contact measurements. They are not production stages or
accepted methods, and require their original private input layout and pinned
runtime. Absolute paths describe the historical environment. No input assets,
model weights, results or credentials are included here.

The full 151-frame CPU depth pass and independent array/clock/closure
verification completed. The first complete object fit is rejected; later
source-derived alignment and temporal selection qualify all 151 inferred
object poses under unchanged motion caps. A fresh articulated repair passes
full static source/native/motion checks. Its between-knot clipping control is
retained; a later local refinement clears all 3,805 independent interpolation
samples. No physics, force-bearing grasp, policy or score improvement is
established. See the [complete reference investigation](../../../docs/team/dex3-isaac-review/E037_FULL_REFERENCE_REPAIR_20261018.md).

The manifest contains 137 exact sources: 135 executed and two prepared,
unexecuted old unaligned validators. Executed does not mean successful: failed
and partial workers remain recorded. `later_job_closed` distinguishes a closed
job from a historically true `full_job_in_progress_at_archive` field.

- Contact projections v1–v6 never established a full contact-consistent,
  velocity-feasible grasp trajectory. V5 cleared the earlier sampled poses but
  failed 44 of 404 full-sequence hand/frame slots and introduced velocity
  violations. V6's view-ray prior worsened that count to 89. No proposal was
  promoted or used for PPO.
- The three spoon scale probes improve sampled source fit. The fixed body
  translation and finite-support probes expose regressions; no complete new
  object/reference trajectory was accepted.
- The bounded rigid-pose and source-depth audits also reject a complete spoon
  repair. The full native compiler's Mac CPU control matches archived static
  contacts; the flat-body surrogate does not. A trajectory-wide root correction
  clears e037 hand contacts but fails the independent source projection screen;
  the conservative source-constrained variant is infeasible. No candidate was
  promoted or trained. See the [native CPU investigation](../../../docs/team/dex3-isaac-review/NATIVE_CPU_CONTACT_REPAIR_20261018.md).

See [current results](../../../docs/team/dex3-isaac-review/SCORED_BASELINE_PROGRESS.md)
and [spoon diagnostics](../../../docs/team/dex3-isaac-review/SPOON_SOURCE_FIT_20261018.md).

A later [carry-source diagnosis](../../../docs/team/dex3-isaac-review/E037_CARRY_SOURCE_GAP_20261018.md) rejects the static/sweep-qualified candidate as a grasp reference. It preserves the incomplete full separation audit, completed11-frame bounds, unqualified object-relative proposals, failed full-frame detector and complete151-frame CPU detector/stereo screens. No physics or PPO was launched.
