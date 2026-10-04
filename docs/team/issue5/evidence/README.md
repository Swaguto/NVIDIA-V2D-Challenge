# Recorded repair evidence

These are small October 3, 2026 CPU validation records. They do not establish
real object-pose accuracy or a policy rollout.
The JSON/XML records and audit snapshot preserve their original bytes across
checkouts so the published evidence SHA-256s remain valid on Windows and Linux.

- `tests-pr.xml`: full portable-input test run, 109 passed, zero skipped.
- `validation_snapshot.json`: tested environment, commands, kit and evidence hashes.
- `asset_download_manifest.json`: pinned public dataset, advertised sizes and verified
  SHA-256s for ten meshes and episode-0/2 stereo videos; binaries remain external.
- `local_results.json`: actual full-mesh, decoded-frame, rectification and eight-frame
  whole-image stereo statistics with explicit evidence limits.
- `COMPLETION_AUDIT.md`: original criteria, proposed gates and missing artifacts.

The original local snapshots and previews remain archived in the author's workspace.
Use the portable commands in [../HANDOFF.md](../HANDOFF.md) to regenerate evidence
on a teammate's checkout. Source revisions are identified there; these snapshots
are not an assertion that later source or data revisions have been validated.
