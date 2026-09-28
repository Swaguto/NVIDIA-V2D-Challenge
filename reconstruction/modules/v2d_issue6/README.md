# Issue 6: hands and stereo contact diagnostics

Local integration for [issue #6](https://github.com/Swaguto/NVIDIA-V2D-Challenge/issues/6).
This supplies independent stages, strict camera/frame contracts and CPU-tested
geometry. **No real episode, GPU inference, MANO fit or contact precision result
has been validated by this contribution.** The GPU adapters need a prepared host.

Start with [HANDOFF.md](HANDOFF.md) for producer contracts, operator steps and
reviewer handback; [VALIDATION.md](VALIDATION.md) separates completed checks from
the remaining GPU and three-episode gates.

## What can run now

Install in a CPU environment from the repository root:

```sh
python -m pip install -e 'reconstruction/modules/v2d_issue6[test]'
python -m pytest reconstruction/modules/v2d_issue6/tests -q
python -m issue6.cli --help
```

`validate`, `depth`, imported `hands`, `contacts`, `report`, and `compare` require
no CUDA. `contacts` consumes the alignment stage's outputs; the CPU integration
test explicitly substitutes **synthetic** hands for the unavailable GPU worker.
Synthetic tests do not validate the worker or count toward the three episodes.

## Input contract and handoff from #4/#5

Copy `examples/episode2.template.json` outside the source tree. It is intentionally
incomplete: null calibration values and empty frames must fail validation, rather
than silently using invented calibration. Paths resolve relative to the manifest.
All geometry is in metres. IDs and timestamps are original video IDs/times.

Provide the actual color-camera intrinsics and **rectified** frames. Rectifying
or cropping changes the intrinsics; record the resulting pixel grid. The v1
adapter rejects raw distorted images. The depth-to-color transform must likewise
include rectification rotations, not just the factory raw-camera extrinsic.

Add a `depth` object:

```json
{
  "camera": {
    "id": "ego_cam_b",
    "convention": "rectified_pinhole",
    "width": 640, "height": 480,
    "fx": 400, "fy": 400, "cx": 320, "cy": 240
  },
  "encoding": "inverse_u16",
  "color_from_depth": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
  "provenance": "predicted"
}
```

**The numbers above illustrate syntax only; replace every calibration value.**
Determine the actual stereo-left camera from calibration, rather than inferring
it from `b`/`c`. Use the validated left depth first. A right-depth scale experiment
must use separately prepared inputs and a separate run.

Supported depth files: PNG or non-pickled 2D `.npy`. Encodings:
`metres`, `millimetres_u16`, or the upstream `inverse_u16` formula
`z = 65535 / pixel - 1`. Invalid/zero/nonfinite depth is masked. Optional
`depth_validity` is a separate source-depth-grid mask. Reprojection forward-splats
points into the color view with a nearest-depth z-buffer. Holes remain unknown;
there is no resizing or fabricated filling.

Each `frames` entry follows this shape:

```json
{
  "id": 0,
  "timestamp_s": 0.0,
  "color": "color/000000.png",
  "depth_path": "left_depth/000000.png",
  "depth_validity": "left_valid/000000.png",
  "objects": {
    "white_pot": {
      "valid": true,
      "direction": "camera_from_object",
      "pose_path": "pot_in_color/000000.json",
      "mask": "pot_masks_color/000000.png"
    },
    "white_pot_lid": {"valid": false}
  }
}
```

- Replace IDs/times with the episode frame table; include every selected frame,
  including frames with absent hands/depth/objects. Frames are contiguous, times
  strictly increasing. `--limit 64` selects the first 64 without renumbering outputs.
- Null/omitted depth means unknown depth. An absent object or `valid:false` means
  unknown object pose. A named but missing file is an input error, not permission
  to substitute another frame.
- A pose is either `matrix` (4x4) or `pose_path` (#3/#5 Transform3d JSON:
  `translation:[x,y,z]`, `rotation:[w,x,y,z]`). Pose filenames must match frame IDs.
- Directions are explicit. `camera_from_object` MUST refer to the color camera.
  For `world_from_object`, supply `camera_valid:true` and `camera_from_world`
  (4x4), or `camera_pose_path` (world-to-color-camera Transform3d JSON), per frame.
  Compose `color_from_world = color_from_left @ left_from_world`; never treat
  a moving wearer's camera-to-world transform as fixed.
- #5's `ego_object_poses/{object}/object_to_world/*.json` can feed the
  world-from-object route. Its `object_to_cam` files need conversion to the color
  camera if they describe either stereo view. Apply frame-quality flags from
  the actual report; file existence does not establish a valid pose.
- Object masks must be in the color grid. If a color mask is missing but the
  object pose is valid, the worker rasterizes its CAD mesh in the color view to
  conservatively exclude object pixels. Otherwise alignment remains invalid
  unless that frame explicitly has `occlusion_clear:true` (reviewed absence of
  object occlusion). Missing geometry/masks are not evidence of clear hand pixels.
- Use original object mesh coordinates and units; do not center, normalize, or
  replace a concave mesh with a convex hull. Materialize LFS assets first.

Modes: `gt_assisted_development`, `predicted_object_integration`, `synthetic`.
Every object and the depth source has `provenance`; camera and hand provenance
are recorded separately. Values are `predicted`, `ground_truth_assisted`, or
`synthetic`. Predicted-object mode requires predicted object poses but may still
have GT-assisted camera calibration. **#5's `--no-gt` does not remove inherited
GT dependence from its camera trajectory.** Synthetic inputs require synthetic mode.
Proxy episodes 0, 11, 23, 31, 39 are excluded from real development runs.

## Staged execution

Use the same manifest, output directory and frame limit at every stage:

```sh
python -m issue6.cli validate --manifest episode2.json --output runs/ep2-hamer-64 --limit 64
python -m issue6.cli hands    --manifest episode2.json --output runs/ep2-hamer-64 --limit 64
python -m issue6.cli depth    --manifest episode2.json --output runs/ep2-hamer-64 --limit 64
python -m issue6.cli align    --manifest episode2.json --output runs/ep2-hamer-64 --limit 64
python -m issue6.cli contacts --manifest episode2.json --output runs/ep2-hamer-64 --limit 64
python -m issue6.cli report   --manifest episode2.json --output runs/ep2-hamer-64 --limit 64
```

The first stage records input hashes, settings, implementation hashes and Git
state. Every stage records elapsed time, completion/failure and output hashes.
Commands/logs are retained for GPU subprocesses. Existing stage directories are
never silently reused. To use changed depth/poses/settings, create a new run and
set `hand_source.backend` to `import` with the previous `hands/records` directory:
this reuses the expensive inference without trusting stale contact outputs.
For imported hand masks provide `masks_dir` and `tracks_path` from the original
mask pipeline, with filenames remapped to original IDs when the clip starts
after frame zero. Track metadata is upstream `hand_tracks.json`.

### GPU prerequisites

Use a prepared Linux host with Docker, GPU support, ffmpeg and installed upstream
host orchestration packages (HaMeR, hand-alignment, MediaPipe, SAM2, and the
pipeline dependencies). The existing `v2d_hamer` image must be built. Do not
overwrite the team's Isaac Lab environment with the CPU environment's packages.

- `mano_assets_root/models/MANO_RIGHT.pkl`: obtained by an authorized team member
  from [MANO](https://mano.is.tue.mpg.de/). No asset download is automated here.
- `weights_dir/_DATA/hamer_ckpts/checkpoints/hamer.ckpt`, associated upstream
  configuration/mean parameters, and `sam2_weights`: actual pretrained assets.
- `model_files`: explicit paths to model checkpoint files, for presence/LFS/hash checks.
- `backend:hamer` runs the shipped mask pipeline without `--run_hamer` (thus no
  MoGe), then invokes the shipped HaMeR and baseline-overlay runners separately.
- `backend:dynhamr` additionally needs the built ViPE/Dyn-HaMR images, synced
  vendor sources and `dynhamr_weights` containing `models/MANO_RIGHT.pkl` and
  `BMC/*.npy`. Its native runner produces `world_results.npz`; the existing
  converter feeds the same alignment stage. Multiple candidate results fail
  explicitly rather than selecting an arbitrary optimization run.

The smoke video uses original frames in sequence, internally reindexed for
upstream runners; exported records are mapped back to the original IDs. Video
inference requires regular timestamps. HaMeR and Dyn-HaMR use different run
directories. If Dyn-HaMR fails, its failure receipt remains and the HaMeR result
is still available. Neither backend is assumed validated on the 8 GB laptop.
Detection defaults to the first selected frame. Set `settings.reference_frame_id`
to another selected original frame with visible hands if necessary. Inspect the
mask overlay for missed hands; the upstream single-reference mask tracker cannot
guarantee recovery of a hand that first appears later. Image IDs and GPU/driver
details are retained in the execution log.

The alignment worker runs inside the existing HaMeR image. It reuses upstream
MANO construction and rendering, estimates global per-track scale from valid
stereo samples, and shifts the scaled hand along its centroid ray. It applies
the same mesh-centroid scale pivot to joints and vertices, including left-hand
mirroring. Missing depth/masks produce a retained visual estimate with
`metric_valid:false`. Large residuals remain visible in diagnostics; metric
validity is a support flag, not an accuracy guarantee.

## Outputs and review

- `align/records`: native-style aligned MANO JSON with explicit validity.
- `align/hands.json` and `hands.npz`: original frame/time axes, tracks, handedness,
  MANO parameters, 21 joints in color-camera metres and metric-valid masks.
  Missing hand observations have no record; contacts expands these to unknowns.
- `align/hand_mesh_overlay.mp4`: existing renderer's mesh overlay when timing is
  regular. Its side panels are camera-relative views, not world-frame validation.
- `contacts/contacts.json`: per fingertip/object raw proximity, bridged windows,
  distance, object-frame nearest triangle point and normal, unknown reason,
  penetration flag (closed consistent meshes only), and geometry ambiguity.
- `contacts/dominant_objects.json`: raw-contact majority summary; ties and
  incomplete observation are explicit. It never deletes per-object results.
- `report/review.html` and `overlays/*.png`: frame-exact contact annotations.
- `report/contact_overlay.mp4`: also encoded if host ffmpeg is available and
  timestamps are regular; otherwise the timestamped gallery remains usable.
- `report/report.json`: coverage, residuals, track handedness changes,
  camera-frame wrist motion, contact transitions and review status.

The default contact threshold is strictly less than 0.02 m. Bridging covers at
most 15 **observed non-contact** frames bounded by contact for the same
hand/fingertip/object. Unknown frames and track loss are never bridged. The
reported duration uses the actual timestamp interval, not assumed FPS.
Penetration is flagged separately from the proximity heuristic. Surface normals
on open, inconsistently wound or inward-oriented meshes are marked ambiguous.

```sh
python -m issue6.cli compare runs/ep2-hamer-64/report/report.json runs/ep2-dynhamr-64/report/report.json
```

Comparison requires matching frame/time axes, calibration, object inputs and
settings. Review all frames, especially grasp/release and occlusions. Contact
precision and reprojection error remain null without independent labels.
Camera-frame velocity includes wearer motion and is not pure hand jitter.
No report automatically marks issue #6 complete. Episode 2 is the pilot;
episodes 12 and 13 need their own validated bundles before three-episode review.

The local preparation milestone does not claim robot training, a submission,
measured forces, or a leaderboard improvement. No commits, pushes, PRs, cloud
provisioning or external messages are performed by these tools.

## Research basis

- [HaMeR](https://github.com/geopavlakos/hamer): per-frame MANO estimation.
- [ViPE](https://github.com/nv-tlabs/vipe): camera estimation for temporal reconstruction.
- [Dyn-HaMR](https://github.com/ZhengdiYu/Dyn-HaMR): temporally consistent hand reconstruction.
- [CHORD Appendices C/D](https://arxiv.org/html/2607.00033v2): ray-based alignment;
  HOT3D's 2 cm / 15-frame contact-segmentation heuristic. Its applicability here
  remains an experiment, not established contact ground truth.
