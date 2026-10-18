# Preserved CPU investigation producers

These are exact copies of executed experimental producers, bound by
`source-manifest.json`. They preserve negative results and the source of the
reported geometry/contact measurements. They are not production stages or
accepted methods, and require their original private input layout and pinned
runtime. Absolute paths describe the historical environment. No input assets,
model weights, results or credentials are included here.

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
