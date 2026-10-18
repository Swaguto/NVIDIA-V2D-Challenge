# Submission contract progress — 2026-10-18

Eight preserved scored baselines have been audited with the untouched kit's
trajectory reader: **e001, e003, e005, e006, e008, e010, e019, e038**. Each has the
complete eight-world frame/object grid and all environment-valid flags true.
All eight also pass the episode-level own-geometry/checkpoint contract.

These facts establish artifact coverage, not grasp quality. Current scored
physical MP-SR remains zero. The complete 20-episode/36-mesh pack has not run and
is not ready for submission.

## Original pilot layout

The first e001/e010/e038 runs store frozen packs and their comparison receipt
separately from policy directories. e001's recovery evaluation is in another
directory. Their bytes remain unchanged.

The submission catalog can now explicitly provide `legacy_sources` containing
`comparison_receipt`, `frozen`, `evaluation` and `readback_report` paths. The
stager verifies:

* Original pinned image, source commit and exact learner iteration/step counts.
* Both initial/final checkpoint bindings to the same frozen manifest.
* The frozen pack's original episode-array hash against the recovered bundle.
* A complete native readback with zero object assistance/reset stabilization,
  the original source speed, matching checkpoint and input hashes.
* The same object order, mesh closure, source-frame coverage and final evaluator
  checks applied to the newer layout.

No normalized producer receipt is written. The final preparation receipt records
the original comparison/readback hashes and provenance explicitly. This avoids
retraining valid old artifacts just to change their directory structure.

The full pack still invokes the untouched official packer without skipping its
crosschecks. Episode-level audit is not a replacement for that required gate.
