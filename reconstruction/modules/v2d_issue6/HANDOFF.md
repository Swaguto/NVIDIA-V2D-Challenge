# Issue 6 handoff

This is the integration/CPU milestone, not completion of issue #6. No recipient
has been assigned or notified by this contribution. Responsibilities below are
handoff roles to confirm with the team. Keep issue #6 open until three episodes
have usable inputs and complete visual review.

## 1. Calibration producer (#4) → reconstruction operator

Deliver an episode-2 bundle with a manifest derived from
`examples/episode2.template.json`, original frame IDs and actual timestamps:

- Rectified `ego_cam_a` color frames and the intrinsics for that exact pixel grid.
- Stereo-left depth files, declared encoding and validity masks; identify the
  actual source camera. Supply its rectified intrinsics and `color_from_depth`.
- Per-frame `camera_from_world` (world-to-color), validity and calibration report.
  The rigid stereo-to-color extrinsic is not a substitute for this moving pose.
- Revision, settings, quality reports and ground-truth dependencies used during
  calibration. The episode-2 calibration work is GT-assisted development input.

Acceptance: frame/time axes match, metres are confirmed, and a depth-to-color
projection overlay is reviewed. Do not resize depth or assume camera b is left.
Keep right-camera bias correction in a separate, explicitly labeled experiment.

## 2. Object-pose producer (#5) → reconstruction operator

Deliver metric meshes with object IDs and the matching original-frame poses,
plus the fitter's quality report and the producer's rule for accepting a frame.
Map rejected/untracked frames to `valid:false`; file existence is not quality.

The `issue5/object-world-icp` exporter interface maps as follows:

| Producer artifact | Manifest field | Direction |
| --- | --- | --- |
| `ego_object_poses/<object>/object_to_world/<frame>.json` | `frames[].objects[id].pose_path` | `world_from_object` |
| `ego_object_poses/<object>/object_to_cam/ego_cam_a/<frame>.json` | Same field, alternative representation | `camera_from_object` |
| Calibration `world_to_cam/ego_cam_a/<frame>.json` | `frames[].camera_pose_path`, with `camera_valid` | World → color camera |
| Original object CAD asset | `objects[].mesh` | Object-local metres |

Pose JSON uses normalized `[w,x,y,z]` rotation, metre translation and unit scale.
Use exactly one pose representation. World poses require the matching camera
trajectory: `camera_from_object = camera_from_world @ world_from_object`.
Stereo-camera object poses must first be composed into `ego_cam_a`; renaming a
directory does not transform them. Filenames must match original frame IDs.

Supply color-view object masks when available. Otherwise valid CAD poses provide
conservative occlusion masks. Unknown occlusion prevents metric hand alignment;
`occlusion_clear:true` is an explicit producer assertion, not a default workaround.

Use `predicted_object_integration` for predicted object poses. Keep
`camera_provenance:ground_truth_assisted` when inherited from the existing
calibration, even if the fitter ran with `--no-gt`. Hash or archive quality reports
alongside the run; record acceptance rules in the manifest's descriptive metadata.
The adapter does not invent an ICP residual cutoff or validate unseen #5 outputs.

## 3. GPU host/asset custodian → reconstruction operator

Provide host access, the prepared image names/IDs, upstream host packages, ffmpeg,
authorized MANO paths and materialized pretrained weights. Do not include model
files or footage in Git. LFS pointers are rejected. Prefer the existing L40S host;
no paid machine is provisioned by this PR.

Use an isolated Python environment and an editable install from this repository;
the GPU adapters resolve sibling reconstruction modules from the checkout.
Do not upgrade the team's training environment to satisfy CPU development tools.

## 4. Reconstruction operator → episode reviewer

Follow README's six-stage commands with `--limit 64` for episode 2, using the same
manifest and run directory throughout. Inspect `validate/readiness.json` first;
successful validation means a readable contract, not complete assets or a GPU pass.

1. Run HaMeR; inspect detection/mask and native hand overlays. Check both hands,
   original-frame mapping, MANO joint ordering, left mirroring and mesh/joint
   agreement. A missed single-reference detection requires a new reference/run.
2. Run calibrated depth projection, alignment, contacts and reporting. Review all
   64 frames, especially occlusions, grasp, release and missing observations.
3. Record image IDs, runtime/driver, elapsed time, observed peak VRAM, failures and
   source revisions. Keep `inputs.lock.json`, all stage receipts and execution logs.
4. Once the smoke run is accepted, run the complete episode in a fresh directory.
   Run Dyn-HaMR separately and compare matching inputs; record it as pending if
   unavailable. A HaMeR baseline does not establish temporal reconstruction.
5. Repeat with episodes 12 and 13 when their validated bundles arrive. Preserve
   holdout episodes 0, 11, 23, 31 and 39.

For a corrected calibration/pose bundle, start a fresh run importing prior hand
records and original-ID masks as described in README. Never edit a completed
stage or remove a failure receipt to make it look successful.

Reviewer handback (one record per run):

```text
Episode / run path / manifest hash:
Reviewer / date / reviewed original-frame range:
Hand backend / camera, hand and object provenance:
Coverage / residuals / switches / temporal jumps / contact transitions:
Occlusion, release and mesh/joint agreement findings (frame IDs):
Independent contact labels, if any (otherwise precision remains unknown):
Accepted for downstream experimentation? / remaining defects:
```

## 5. Reviewed artifacts → refinement, retargeting and reliability experiments

Deliver `align/hands.json`, `align/hands.npz`, `align/records/`,
`contacts/contacts.json`, `contacts/settings.json`, `report/report.json`, both
overlays/galleries, immutable receipts, input lock and the signed review record.
Archive the source bundle separately so its hashed paths remain recoverable.

- Joints are 21 MANO-ordered points in the color-camera frame, in metres. Use
  `metric_valid` and explicit frame/time IDs; absent hand records are unknown.
- Contact rows retain every hand/fingertip/object. `raw_contact` is strict surface
  proximity below 2 cm; `window_contact` bridges at most 15 observed gaps.
  Null values remain unknown. The dominant-object summary is optional.
- Surface points/normals are object-local. Transform points with the full pose
  and normals with its rotation. `geometry_ambiguous` includes open, inconsistent
  or inward-oriented geometry. Penetration is a separate flag.
- Metric validity indicates depth support, not verified accuracy. No measured
  force, wrench reward, robot retargeting or learned-policy result is supplied.
- Keep provenance and raw/bridged variants separate in downstream comparisons.
  Do not report precision without independently reviewed labels.

See [VALIDATION.md](VALIDATION.md) for measured local evidence and unrun gates.
