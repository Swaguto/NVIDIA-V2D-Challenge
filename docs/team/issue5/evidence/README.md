# Recorded repair evidence

These are small October 3, 2026 CPU validation records. They do not establish
real object-pose accuracy or a policy rollout.
The follow-up records below also document a user-reported tracking failure;
that run has not been reproduced or verified fixed.
The JSON/XML records and audit snapshot preserve their original bytes across
checkouts so the published evidence SHA-256s remain valid on Windows and Linux.

- `tests-pr.xml`: full portable-input test run, 109 passed, zero skipped.
- `validation_snapshot.json`: tested environment, commands, kit and evidence hashes.
- `asset_download_manifest.json`: pinned public dataset, advertised sizes and verified
  SHA-256s for ten meshes and episode-0/2 stereo videos; binaries remain external.
- `local_results.json`: actual full-mesh, decoded-frame, rectification and eight-frame
  whole-image stereo statistics with explicit evidence limits.
- `COMPLETION_AUDIT.md`: original criteria, proposed gates and missing artifacts.

Follow-up at tested software commit `ea38fff4c70445343ad5e5cfcb8acd8508d872d1`:

- `tests-review.xml`: 133 passed, zero skipped, including 24 additional controls
  relative to the original run. The detector/model outputs in these controls are
  generated fixtures; actual GPU perception was not run.
- `review_validation.json`: source Git-blob hashes, environment, commands,
  results, evidence hashes and remaining work.
- `review-input-results.json`: full expected fourteen-asset verification under
  `python -O`, with pinned calibration/GT hashes, ten meshes and eight stereo
  frames checked. No episode-39 tracking accuracy is measured.
- `tracking_video_review.json`: provided clip hash/metadata and sampled caption
  readings; original command/commit/inputs remain unknown. The clip itself is
  retained locally. See [../TRACKING_FAILURE.md](../TRACKING_FAILURE.md).

The original local snapshots and previews remain archived in the author's workspace.
Use the portable commands in [../HANDOFF.md](../HANDOFF.md) to regenerate evidence
on a teammate's checkout. Source revisions are identified there; these snapshots
are not an assertion that later source or data revisions have been validated.
