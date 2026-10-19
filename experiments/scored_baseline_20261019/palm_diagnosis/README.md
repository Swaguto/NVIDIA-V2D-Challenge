# Saved-palm and contact-load diagnosis

This contains code and declared experiment plans only. Source recordings,
derived input arrays, native models, gated weights and private reports are
excluded. All rejected controls are retained privately.

The manifest records the executed producer SHA separately from the archive
SHA. Whitespace on blank lines is normalized for code review and restored
losslessly by `restore_sources.py` before checking the original SHA.

Verify provenance without writing or running an experiment:

```bash
python restore_sources.py
```

Restore the exact producers under an existing private workspace:

```bash
python restore_sources.py --private-workspace "$V2D_PRIVATE_WORKSPACE" --write
```

The helper refuses to replace different existing producers. Restore and
verify the matching private input tree separately. The producers refer to
the owner's repository and private artifact layout; they are research
provenance, not a portable policy reproduction. Do not rerun completed
experiments or overwrite their result directories.

Results and limitations are recorded in
`docs/team/dex3-isaac-review/WRIST_RETARGET_CONTACT_20261019.md`.
