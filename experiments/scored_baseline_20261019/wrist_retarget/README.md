# Private wrist and grasp experiment code archive

This preserves the exact CPU producers and independent resource guards used
for the October 19 wrist/contact investigation. The SHA manifest binds each
copy to its original private producer. Failed and rejected controls are kept.
No footage, result arrays, MANO, gated weights or credentials are included.

Restore scripts under `PRIVATE_WORKSPACE/compute/`, alongside the matching
private input tree. They resolve that workspace from their file location and
refer to the owner's existing repository path. Do not execute them from this
archive directory. Several guards intentionally refuse an existing output
directory. Restore private inputs and inspect the recorded source/result hashes
before any new experiment; do not overwrite or rerun completed controls.

The code was executed with the private CPU environment, whose package versions
are recorded in `source-manifest.json`. The environment and private inputs are
not distributed here. The native model uses the existing pinned runtime from
the project. This archive is research provenance, not a portable full-policy
reproduction or a default preprocessing change.

See `docs/team/dex3-isaac-review/WRIST_RETARGET_CONTACT_20261019.md` for results
and limits. A successful process or optimizer return is not physical success.
