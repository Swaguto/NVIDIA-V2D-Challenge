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

See [current results](../../../docs/team/dex3-isaac-review/SCORED_BASELINE_PROGRESS.md)
and [spoon diagnostics](../../../docs/team/dex3-isaac-review/SPOON_SOURCE_FIT_20261018.md).
