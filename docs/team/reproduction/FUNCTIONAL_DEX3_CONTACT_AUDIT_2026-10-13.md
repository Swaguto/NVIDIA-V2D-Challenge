# Final-fit Dex3 contact validation

The current frozen e002 motion has **377/377 closed-grasp states** with two
selected pad faces within 2 mm of their intended CAD triangles and opposing
normals. It covers **375/381 source-labeled definite grasp states**. Six missed
transitions remain; this is a geometric contact proxy, not force-supported
manipulation, measured hand accuracy or an official policy score.

The [summary](dex3-motion-replay/support-final-CAD-pad-audit-summary-v1.json)
and [complete compressed report](dex3-motion-replay/support-final-CAD-pad-audit-v1.json.gz)
record all 343 source frames, 439 phase-bound hand states, 878 pad-to-CAD
measurements and 878 additional intended-target triangle checks. Source
annotations enter validation only after every geometric measurement. No source
pose, validity flag, contact annotation, robot motion or release asset changes.

## Correct audit binding

The earlier template and final CAD fit select different finger triangles.
Applying the legacy template-face audit to the final motion falsely suggests
that the left index misses contact throughout its first grasp: its old face is
approximately 3.463 mm from the object. The final fit's selected index face is
approximately 1.501 mm away and has a verified witness on its intended CAD
triangle. This is an audit-input mismatch, not a corrected robot pose.

The existing `audit_frozen_dex3_original_contacts.py` now accepts
`--cad-candidates`. It requires the candidate file's hash to match the current
motion report, checks its grasp/template/seed bindings, and uses the final
fit's saved selected triangle indices. It independently projects each positive
gap witness onto both original triangles and checks opposing normals. The
normal comparison retains the fitter's recorded 1e-7 numerical tolerance;
the 2 mm gap limit is unchanged. Without the new argument it retains the
legacy template-face mode. The immutable old audit archive remains unchanged.
All checks concern the original URDF collision meshes; cosmetic visual-surface
work remains deferred under the [current scoring scope](SCORING_SCOPE_2026-10-13.md).

The [final candidate report](dex3-motion-replay/support-final-CAD-candidates-v1.json)
is the original 48,369-byte measured input, SHA256
`a208dc2aa9625b479244cc7afb5352f06bed84fa2ac3eb2085156d6270b18a0e`.
Its source metadata and refused historical search flags are retained; this
audit does not upgrade them to physical acceptance.

## Repeat with the existing pipeline

Use this documentation revision of PR #38 and the pinned Linux Python 3.11.17
geometry runtime from the [support replay guide](SUPPORT_AWARE_DEX3_REPLAY_2026-10-13.md).
Windows users can run that runtime in WSL. First regenerate `15-case` with all
15 stage checks passing. Restore the original CAD fitted inputs using the
[CAD reproduction instructions](ORIGINAL_CAD_DEX3_REPLAY_2026-10-13.md).
That runner's `inputs` directory supplies the atlas and its measured templates.
The support kit's `methods` directory supplies the unchanged geometry helpers.

Restore the independent source annotation from its existing archive:

```sh
python scripts/reproduction/restore_original_CAD_audits.py \
  --archive docs/team/reproduction/dex3-motion-replay/original-CAD-baseline-audits-v1.tar.gz \
  --manifest docs/team/reproduction/dex3-motion-replay/original-CAD-baseline-audits-manifest-v1.json \
  --out /data/v2d/original-CAD-audits
```

Require all ten file checks. The restored
`evidence/e002-source-grasp-annotation.json` SHA256 is
`f4de747c96da804158b063b86d5683dd28c618ae5c24558cb4f8038399d4b8bf`.
Use fresh output directories. Replace the example absolute roots consistently:

```sh
/data/v2d/geometry-runtime/bin/python scripts/reproduction/audit_frozen_dex3_original_contacts.py \
  --case /data/v2d/support-reconstruction/15-case \
  --case-manifest-sha256 dfe06ac14d976e48ffc58ec07cad74f8d49573fc4a080bd6f57f79668ddf4ebb \
  --methods /data/v2d/support-method-kit/methods \
  --fitted-inputs /data/v2d/regenerated-motion/inputs \
  --cad-candidates docs/team/reproduction/dex3-motion-replay/support-final-CAD-candidates-v1.json \
  --annotation /data/v2d/original-CAD-audits/evidence/e002-source-grasp-annotation.json \
  --out /data/v2d/final-CAD-contact-audit
```

Compare input/module hashes, the four selected-face bindings, all 439 records,
pad gaps, contact witnesses and normal dots against the compressed report.
Numerical distances should agree within 1e-8 m; elapsed time varies. Require
343 processed frames, 377/377 closed states passing both current-target checks
and exactly the six missed source states below. Restoring stored reports or
comparing hashes alone is not a fresh geometry execution.

A deliberately different candidate report was refused before an output
directory was created. The ten-file source annotation archive was restored and
verified on Windows. No GPU is needed for this audit.
After the code change, the legacy command was also rerun on all 343 frames:
its 439 records, numerical measurements, labels and phase bindings match the
pre-change execution exactly. Only new scope metadata and execution timing differ.

## What remains physically unverified

These are **false-negative geometric contact states against positive source
grasp labels**, not false-positive contacts that can be fixed by suppressing
predictions. The measured intended-surface gaps are:

| Hand / source frame | Recovered hand pose available | Thumb gap | Index gap |
| --- | --- | --- | --- |
| Left / 46 | No | 3.698 mm | 16.576 mm |
| Right / 46 | No | 9.431 mm | 21.478 mm |
| Left / 272 | Yes | 19.142 mm | 20.929 mm |
| Right / 309 | Yes | 25.022 mm | 47.456 mm |
| Left / 332 | No | 20.054 mm | 38.495 mm |
| Left / 333 | No | 37.617 mm | 77.624 mm |

The source labels establish coarse visible grasp timing, not fingertip identity
or 3D contact truth. Four missed states have no recovered hand pose. Changing
their display continuity cannot create an observation. The two observed misses
need source-supported timing/placement refinement if they affect subsequent
physical object tracking. Hand target approvals remain false.

The current Isaac video still disables gravity and collision resolution.
These contacts do not prove that replayed hands can carry the objects with
physics enabled. That functional question, object tracking and reproducibility
remain relevant to the development goal; further cosmetic mesh and internal
assembly-overlap polishing is deferred. Training and Issue #9 are not started.
