# Issue 5 repair review and validation handoff

This draft makes the software repairs reviewable while issue #5 remains open.
It does not establish improved real tracking accuracy, successful robot training,
or a policy-rollout submission. The accepted CHORD/stereo/contact experiment is
unchanged. FoundationPose remains optional initialization, not a replacement tracker.

The team lead subsequently reported that tracking still fails. The provided episode-39
clip shows a tabletop-sized green mask and a displaced dustpan CAD. See
[TRACKING_FAILURE.md](TRACKING_FAILURE.md) for observed evidence, the removed
single-object unmatched-label fallback, new native rejection reports, and the exact
run artifacts needed. The video generator and inputs are not available, so this
follow-up does not claim to have reproduced or fixed the team's run.

## Background and branch dependencies

The repair branch starts at PR #26 (`issue5/object-world-icp`,
`ef8a7ac4615aaf9b499387fa68b76b27246d9d55`). The draft targets that branch so its
diff isolates these repairs instead of repeating #26's object-tracker implementation.

Native stereo/registration and scoring tools were copied from PR #28
(`track3/official-metric-and-submission`,
`a6e0cb2e87299278fd610a9f3afc7fa754732668`) and then repaired here. Those copied
files appear as additions against #26. This is not a Git merge of #28. Before any
integration, reconcile overlapping files with #28 and review the resulting diff.
If #26 merges or changes, rebase/reconcile this branch and explicitly review any
base change; neither PR is merged by this handoff.

PR #18/#20 supply the GT-assisted calibration baseline on main `ca3f64a`.
Historical small ICP residuals do not prove correct poses: a fitted camera can
absorb a faulty GT rotation. Likewise, the official frame-0 SE(3) alignment removes
one constant gauge, not worn-camera motion or metric-scale error. A shared metric
camera path is needed for fixed-world reconstruction.

The older PR #26 “official harness” result used `score_candidate.py`; it is not
proof of agreement with the five official pose scorers. PR #28's older 150-degree
diagnosis was retracted after writer/convention fixes. Neither historical result
is a regenerated measurement of the repaired pipeline. The ~48 mm pot / ~11 mm
lid discrepancy and FoundationStereo right-depth bias remain hypotheses to retest.

## Review map

| Area | Files and key review question |
|---|---|
| Pose boundaries | `fit_ego_rig.py`: xyzw→wxyz, selected GT name/index, original scan origin, export `P=C@W` |
| Object fitting | `fit_object_pose.py`: one hashed camera path per frame across objects; rejection controls propagation/export; gaps/timestamps; raw versus final residuals |
| Native registration | `register_poses.py`: restore centroids including warm-start scale; rectified axes and rectification manifest; accepted observation tags |
| Stereo | `stereo_depth.py`, `validate_stereo.py`: P2 baseline, grayscale SGBM penalties, invalid reverse matches, camera-frame plane interpretation |
| Mesh/overlays | `glb_mesh.py`, `visualize_object_pose.py`: every active scene primitive/node transform; correct face indices and frontmost z-buffer; hashed camera source |
| Evaluation | `official_metric.py`, `score_track3_official.py`, `convert_camera_poses.py`: time/world axes, original/rectified conversion, frame IDs, no silent missing episodes, explicit raw/fill/provenance |
| Supporting tools | Segmentation probe label matching; safe output destinations and linear rotation weights; bounded asset/test-kit downloads |

The machine-readable contract is [contracts.json](contracts.json). JSON rotations
are wxyz; metric/loader rotations are xyzw; all geometry is in metres. The
original-frame fitter expects compatible masks/depth/intrinsics. Do not feed it
rectified clouds with original intrinsics or original rig axes. Native rectified
camera observations use the separate world-conversion CLI.

## Reproduce CPU validation

From this repository root, use a dedicated Python environment with NumPy, PyTorch,
SciPy, OpenCV, trimesh, Pillow, pyarrow, pandas and pytest. Install the calibration
module from this branch:

```bash
python -m pip install -e 'reconstruction/modules/v2d_world_calib[data]' scipy trimesh Pillow pytest torch pandas opencv-python
python scripts/fetch_issue5_test_inputs.py --output /absolute/fresh/issue5-test-inputs
export V2D_SUBMISSION_KIT=/absolute/fresh/issue5-test-inputs/v2d_submission_kit
python -m pytest tests/test_issue5_official_wrapper.py tests/test_issue5_repairs.py tests/test_issue5_review_regressions.py tests/test_issue5_stereo_diagnostic.py reconstruction/modules/v2d_world_calib/tests -q
python scripts/eval/official_metric.py --selfcheck
```

On PowerShell, use `$env:V2D_SUBMISSION_KIT = 'C:\absolute\issue5-test-inputs\v2d_submission_kit'`
instead of `export`. The public calibration is a small checked-in test fixture;
`V2D_CALIB` may override it. The kit download verifies SHA-256
`b1f8f703c2e772684a57d048cf28f40c7b9b54cf7a61aff6917b46d826915e1f` before extraction.
`--kit-zip` can reuse an existing archive after the same hash check.

The initial recorded run has 109 passed and zero skipped. The follow-up adds CLI,
input integrity, fixed-world smoothing and single-object segmentation controls;
its exact source revision/results are in `evidence/review_validation.json`.
Missing-kit parity tests skip
explicitly; such a reduced run is not equivalent to the recorded check. Whole-repo
IsaacLab/GPU/training tests were not run. Native DINO/SAM2/registration entrypoints
still need the operator's compatible CUDA/perception environment and checkpoints;
CPU registration unit tests do not demonstrate native inference performance.
See [evidence/validation_snapshot.json](evidence/validation_snapshot.json) for the
tested environment, commands, hashes and check boundaries.

For fresh-output camera scale estimates, `--est-right-scale` needs existing saved
poses under the input work directory and an explicit fresh `--out`. Render fresh
object-fitting output with `visualize_object_pose --poses /path/from/fit_object_pose/--out`;
`--work` still supplies RGB, depth, masks and intrinsics. The rotation filter accepts
declared fixed-world `object_to_world` records, after `convert_camera_poses.py`, and
writes inferred rotations without validated residuals. `--window` is measured in
source frame IDs; episode averaging can erase real object motion.

The real-input validator requires all fourteen pinned assets in the supplied manifest
and checks their actual sizes/hashes against the committed expected manifest. It also
checks the pinned calibration and episode-0/2 reference hashes before opening them.
An episode-subset download manifest is insufficient, even if older files remain on disk.
Operational checks remain enabled under `python -O`.

## Real input evidence

The public dataset was pinned at `5f68335f3acc802033d1e80728c1633197521de8`.
All ten visual meshes and both ego videos for public episodes 0 and 2 were downloaded:
14 files, 105,239,822 bytes, checked against advertised lengths and LFS SHA-256s.
All full GLB geometries match trimesh. Both views decode to 373 and 343 frames,
respectively, exactly matching GT source frame IDs.

Eight predetermined whole-image SGBM checks produced 46.5–55.4% valid depth.
Disagreement above 1.5 pixels was 12.5–17.4% among valid textured reverse matches,
versus 23.5–30.0% with vertical-shift controls. This supports rectification
sensitivity but does not measure pot depth, masks, poses or the FoundationStereo
bias. Substantial missing depth remains. No parameters were tuned against GT.

The small evidence JSONs are committed under [evidence/](evidence/README.md).
RGB, CAD, checkpoints and generated pose binaries are not in this PR. Recreate
RGB/CAD with `scripts/fetch_track3_assets.py --output /absolute/public` and supply
matching public `meta` and episode-0/2 `data/chunk-000` files. Then run:

```bash
python scripts/eval/validate_issue5_inputs.py --data /absolute/public --out /absolute/fresh/input-diagnostic
```

## Needed handoffs

These are workstream handoffs, not new team assignments or sent messages.

| Workstream | Requested input / receipt |
|---|---|
| Compute / VM owner | Usable SSH host/IP, port, username and authorized public-key access, or artifact download link; actual repo/data/cache/environment paths. TEAM_ACCESS.md names a VM but gives an IP placeholder. |
| Issue #3 calibration / prepared perception | Original episode-0/2 masks, raw depth/disparity validity, matching intrinsics/rectification, original `make_masks.py`/`make_depth.py` or actual producer versions, logs/model/input hashes and frozen old outputs. These scripts/observations were not found in the reviewed Git branches. |
| Issue #4 camera/depth | One metric time-varying camera trajectory per source frame in one fixed gauge, shared across objects. Include episode/frame/capture time, axes, rectification/calibration hash, scale check, producer and GT-use classification. Independent lane must not use object-trajectory GT. The reviewed issue-4 branch equals main and supplies no such implementation/artifact. |
| Issue #5 tracking | Run bug-only corrected episode 2 and 0 with frozen observations, scale 1 and smoothing off; compare one lid-derived shared GT-assisted camera versus pot-derived camera separately, then the independent path. Keep lanes/output roots distinct and retain failures. |
| Evaluation / integration | Score frozen candidates with the five kit pose metrics and raw/fill coverage; reconcile kit roster with issue wording; inspect an asymmetric control, then expand to all 20 public IDs. Coordinate overlapping PR #28 files before integration. |

Detailed prepared-input and conversion commands are in
[REPAIR_STATUS.md](REPAIR_STATUS.md). `--no-gt` only controls the object fitter's
GT reads; it does not make an upstream GT-assisted camera independent.
`--require-independent` enforces the declared camera/observation provenance.
The converter consumes a camera path; it does not estimate a missing one.

For every real run preserve source commit, command, environment/model hashes,
input hashes, source frame/capture-time map, calibration/rectification and mesh
identity, shared camera path/hash, raw/final/inferred status, coverage/gaps and
report/overlay hashes. Outputs must go to fresh directories.

## Acceptance still open

- All 20 public per-episode/object reports: visible median translation <5 cm and
  rotation <5 degrees. Keep per-episode medians and pooled medians, rotation,
  translation, Chamfer, pass@5cm/10deg, zero-solve cases and raw coverage.
- Pot on both `ego_cam_b` and `ego_cam_c`: medians <10 cm/<10 degrees and a measured
  residual explanation. Recheck masks/depth/CAD with frontmost visible surfaces,
  then actual-axis spin/tilt and temporal observability; F8 is unresolved.
- Real cross-object RPE and both-direction relative geometry. The kit has 16
  two-object configurations, not the issue's 17: report all 16 and mark the
  seventeenth unidentified. Pot appears in six kit episodes including 8, omitted
  from the issue's five-episode list. Do not silently remove episode 8.
- Preserve the asset-free fallback note for missing geometry. `wooden_spoon` has
  no released mesh/URDF in the audited snapshot; its separate fallback/shape
  dependency is not solved here. CD-O is separate shape evaluation.

Reconstructed trajectories are diagnostic references. An official policy-rollout
submission needs real policy/checkpoint/evaluator provenance and is outside this
repair evidence. Keep this PR in draft and issue #5 open until the real gates are
assessed. No merge or automatic integration is performed.
