# Issue 5 software repairs

Updated October 3, 2026 (America/Toronto). Repairs are on
`Evc-r/issue5-repairs`, prepared for a draft PR targeting PR #26's branch.
No merge, GPU VM run, robot training or policy submission was performed.

Follow-up: the team reports continued tracking failure. The episode-39 overlay shows
a tabletop mask instead of an isolated dustpan. [TRACKING_FAILURE.md](TRACKING_FAILURE.md)
separates this visible evidence from unconfirmed causes and records the additional
mask-assignment/tool repairs. Real tracking is still unresolved; initial CPU tests
were not an end-to-end tracking validation.

Base: PR26 `ef8a7ac4615aaf9b499387fa68b76b27246d9d55`. The stereo/registration
and official pose-metric utilities were imported from PR28
`a6e0cb2e87299278fd610a9f3afc7fa754732668` and repaired in this checkout.
The original `team-repo` checkout and historical diagnostic artifacts were preserved.
The baseline project document was refreshed across all six tabs on October 3;
Research Specifics was empty. Scope remains CHORD/stereo/contact reliability.
See [HANDOFF.md](HANDOFF.md) for dependency reconciliation, portable test setup and
requested artifacts, and [evidence/README.md](evidence/README.md) for committed records.

## Implemented and verified

| Finding | Repair and verification |
|---|---|
| F1 | GT xyzw is explicitly reordered to wxyz at `_w_obj`. Quaternion boundaries normalize and reject zero/nonfinite values. Identity, random rotations, near-half-turns and q/-q roundtrips pass. |
| F2 | Registration restores the original cloud and CAD centers before returning camera-space translation. Warm starts convert back into centered coordinates and retain free scale. Similarity composition also scales the previous translation. Perfect asymmetric CPU clouds recover rotation, scale and translation, including nonzero scan origins. |
| F3 | The calibration exporter writes `C @ W`, not `C @ inv(W)`. A CLI fixture with nonidentity moving camera/object transforms and reordered GT membership verifies actual JSON exports. |
| F4 | Every object uses the same selected camera artifact at each source frame. No per-object/fallback switching. The file path, SHA-256, axes and GT-use tag are persisted in transforms and reports. `--require-independent` rejects GT-assisted/unknown cameras; `--no-gt` alone makes no independence claim. |
| F5 | Raw moving-camera observations are rejected by the fixed-world scorer. `convert_camera_poses.py` applies `inv(C(t)) @ P(t)`, or `inv(H1 @ C(t)) @ P_rect(t)` using the recorded rectification. Tests cover moving cameras, shared multi-object trajectories and noncontiguous source IDs. |
| F6 | Rejected fits/jumps are excluded from propagation and observation exports. Gaps trigger reinitialization; temporal gates record elapsed frames/time. Smoothing defaults off, uses an immutable snapshot, and never fills missing observations. Residuals are recomputed on the final pose. Camera coverage uses each frame's mesh, not the last solved mesh. Existing outputs are preserved by requiring fresh destinations. |
| F7 | Overlays use the full scene mesh, including node transforms; faces retain their original vertex indices. A perspective-correct triangle z-buffer produces frontmost mesh depth. Camera lookup uses the tracker's hashed source and derives the right view from the rig. Repeated invocation excludes visualization directories. |
| F9 | The metric CLI inserts world after time, validates axes/finite values, and checks optional frame IDs. Episode loading supports NPY and both native JSON layouts, matches source IDs, and rejects duplicates, invalid rotations and incompatible scales. Structured reports preserve raw/inferred/gap-filled masks and GT-use labels. Missing episodes fail rather than disappearing from aggregation. |

Additional repairs: rectified baseline is derived from P2; the SGBM smoothness
penalties use squared block size and one grayscale channel; calibration arrays
are contiguous with a column translation for OpenCV. The segmentation probe matches
detector labels to object names rather than assuming box order. Rotation smoothing
uses linear weights and refuses existing, nested, or parent output destinations.
The evaluator GLB reader now includes all active scene nodes and triangle primitives,
retains node transforms, and supports unsigned byte/short/int indices and strided
vertex buffers. The stereo diagnostic excludes the reverse matcher's invalid sentinel
and does not call a camera-frame plane a horizontal world support surface.

## Local validation

Run from this checkout with the workspace Python environment:

```powershell
../.venv/Scripts/python.exe -m pytest tests/test_issue5_official_wrapper.py tests/test_issue5_repairs.py tests/test_issue5_review_regressions.py tests/test_issue5_stereo_diagnostic.py reconstruction/modules/v2d_world_calib/tests -q
../.venv/Scripts/python.exe scripts/eval/official_metric.py --selfcheck
```

The initial 109 tests passed; results are recorded in
`evidence/tests-pr.xml` (portable-input rerun); earlier records remain archived locally.
Five downloaded official pose kernels match within 1e-9 on oracle, constant
gauge, moving camera, body-offset, noise, scale and multiworld controls. Oracle
and constant-gauge controls produce AUC/SP-SR/MP-SR=1 and RPE/MPPE=0.
The full downloaded participant-kit `score()` wrappers also match the adapter on
oracle, moving-camera, scale and multiworld fixtures. All five reject missing/duplicate
rows, inconsistent provenance, invalid commit URLs, nonfinite positions and zero
quaternions. Fixture provenance explicitly says `synthetic-test-only`; no rollout is
claimed. The JSON-to-episode CLI is tested with noncontiguous IDs, missing rows,
inferred rows and a structured report. CD-O and real policy-rollout scoring were not run.

Follow-up results and exact tested source hashes are in `evidence/review_validation.json`
and `evidence/tests-review.xml`. The real-input checks were rerun with `python -O`
against the full expected fourteen-asset set, plus pinned calibration/reference hashes;
see `evidence/review-input-results.json`. These checks do not use the episode-39
tracking inputs, which are unavailable.

## Real input checks

The repo's public dataset reference was usable despite the absent VM artifacts.
Downloaded all ten public visual meshes and both ego-camera RGB videos for public
episodes 0 and 2, pinned to dataset revision
`5f68335f3acc802033d1e80728c1633197521de8`: 14 files, 105,239,822 bytes.
Advertised sizes and LFS SHA-256 hashes were verified. The bounded downloader is
`scripts/fetch_track3_assets.py`; inputs and its manifest are in the enclosing
workspace's `data/track3/public`.

The repaired GLB reader matches complete trimesh geometry for all ten actual assets.
Both videos decode to exactly the GT source frame count: episode 0 has 373 frames,
episode 2 has 343, all 1280x800. Timestamps are finite and strictly increasing.
Eight predetermined frames were rectified and processed by repaired SGBM. Valid depth
covered 46.5–55.4% of the whole image. Disagreement above 1.5 pixels was 12.5–17.4%
among valid textured reverse matches, versus 23.5–30.0% for vertical-shift controls.
This supports calibration sensitivity, while substantial missing depth remains.
It does not measure masked pot depth accuracy, object poses, or the historical
FoundationStereo right-depth bias. No parameters were tuned against public GT.

Portable reproduction uses `scripts/eval/validate_issue5_inputs.py`; the raw result
and pinned asset manifest are committed under `evidence/`. Earlier commands and
previews are preserved in the author's workspace:
`runs/issue5_repairs_20261003/validate_public_assets.py` and
`runs/issue5_repairs_20261003/real_assets/local_results.json`, with rectified stereo
and depth previews. Software source hashes and the review patch are refreshed in
the same run directory. The original diagnostic and earlier test records are retained.

## Coordinate contract and operation

Column vectors; metres; JSON wxyz; metric arrays xyzw. `W=T_world_object`,
`C=T_camera_world`, `P=C@W`. Original stereo axes satisfy `C_right=M@C_left`.
Rectified observations require the corresponding R1 rotation. Scan origins remain
unchanged by point sampling and registration. A free scale is a diagnostic option;
fixed-world conversion and scoring require CAD scale 1.
The machine-readable boundary/provenance specification is `contracts.json` beside
this report. `evidence/COMPLETION_AUDIT.md` checks original
criteria, proposed gates and named deliverables separately; missing real evidence
prevents completion despite the passing software tests.

The camera fitter remains a GT-assisted diagnostic producer. Independent tracking
requires an externally produced metric camera trajectory, shared across objects,
with `gt_use: independent` on its frame records. A constant factory rig transform
does not supply world egomotion. The converter does not estimate missing cameras.

Public prepared-input diagnostic:

```bash
python -m v2d.world_calib.vm.fit_ego_rig \
  --data-root "$DATA" --work "$WORK" --episode 2 --out "$NEW_CAMERA_ROOT" --no-smooth
python -m v2d.world_calib.vm.fit_object_pose \
  --data-root "$DATA" --work "$WORK" --episode 2 --score-gt \
  --camera-root "$NEW_CAMERA_ROOT/white_pot_lid/world_to_cam" --out "$NEW_OBJECT_EPISODE" --no-smooth
```

Native camera observations, then independent world conversion:

```bash
python scripts/track3/register_poses.py --episode 2 --no-gt \
  --objects white_pot,white_pot_lid --out "$NEW_NATIVE_ROOT"
python scripts/eval/convert_camera_poses.py --episode 2 \
  --predictions "$NEW_NATIVE_ROOT" --camera-root "$INDEPENDENT_CAMERA_ROOT" \
  --rectification "$NEW_NATIVE_ROOT/rectification.json" --require-independent --out "$NEW_FIXED_ROOT"
python scripts/eval/score_track3_official.py --episodes 2 \
  --predictions "$NEW_FIXED_ROOT" --data "$DATA" --oracle --report "$NEW_REPORT_JSON"
```

Set `V2D_PUBLIC` to DATA and `V2D_CALIB` to its calibration JSON for the native
pipeline. Models/checkpoints, challenge meshes and RGB must exist in the prepared
operator environment. These are conditional commands; they were not run on real
episodes in this repair session. Camera/frame timestamps must match. Missing poses
are preserved as gaps, then explicitly flagged when the diagnostic scorer fills them.

## Outstanding real-data evidence

F8 remains unresolved: there is no validated diagnosis of the pot/depth discrepancy
or its spin ambiguity. Previous rotation/separation claims used faulty exports and
must be regenerated before attributing error to CAD, depth, masks or symmetry.
The stale PR28 150-degree diagnosis is historical, not a current measurement.

The repo's TEAM_ACCESS.md names a Brev L40S VM and paths under
`/home/ubuntu/v2d_track3/data/hf/track_3/public`; it provides no usable host/IP or
authorized login. Source inspection found no prepared masks/depth or independent
camera artifacts in Git. The issue-4 branch has no source diff from the fetched main.
The remote-device connector lists only an offline desktop. The local PyTorch build
is CPU-only and perception checkpoints are absent. RGB/CAD/GT/calibration now exist
locally, but original VM observations and lineage cannot yet be compared.

Original issue acceptance remains unverified: all 20 public-episode error reports;
general median <5 cm/<5 degrees; pot median <10 cm/<10 degrees in both ego views;
and real cross-object RPE. Kit audit gives 20 evaluation episodes, 16 two-object
configurations, 36 episode-object pairs, 5,181 frames and 9,120 rows. The issue's
17th configuration is unidentified; pot occurs in six kit episodes including 8.
No tolerance or episode was silently removed. The asset-free fallback remains a
separate dependency when a required mesh is unavailable.

Software regression success is not measured real tracking, robot training, or a
policy-rollout submission. Do not close issue 5 on this evidence alone.
