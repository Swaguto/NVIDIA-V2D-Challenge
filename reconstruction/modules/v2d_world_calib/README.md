# v2d_world_calib

Mocap-to-camera coordinate bridge for V2D Track 3.

Solves the missing **`world -> camera`** rigid transform (`T_CW`) of the fixed
rig, once, from public GT object geometry paired with image keypoints:

- Object mesh points (object frame) + **GT pose** -> **world** 3D points
- Object image keypoints -> **camera** 2D points
- **RANSAC-PnP** initialization + **Huber reprojection refinement** (SE(3)
  Levenberg-Marquardt over many frames) -> `T_world->cam` per camera
- Verification on **held-out** frames + a cam-to-cam consistency check against
  the provided intrinsics/extrinsics

## Layout

```
v2d_world_calib/
├── calibration/
│   ├── camera_geometry.py   Stage 0: parse camera_calibration.json (intrinsics,
│   │                                 distortion, cam<->cam, ego baseline check)
│   ├── correspondences.py   Stage 1: 3D<->2D pair builders (keypoint_2d,
│   │                                 silhouette_2d, synthetic providers)
│   ├── solve.py             PnP-RANSAC + Huber LM refinement (numpy, no scipy)
│   ├── silhouette.py        Option-B: mesh-mask IoU for a candidate transform
│   └── verify.py            transfer stats, spatial error map, cam-cam check
├── cli/calibrate.py         CLI + --self-test
└── tests/test_synthetic.py  correctness suite (numpy + opencv only)
```

## Run

```bash
# install (editable)
python -m pip install -e reconstruction/modules/v2d_world_calib

# synthetic correctness proof (no challenge data)
python -m v2d.world_calib.cli.calibrate --self-test --out-dir results/calib

# thin Issue #3 wrapper from scripts/eval also works:
python scripts/eval/world_camera_calibration.py --self-test

# real calibration (shared box: GT parquets + keypoint cache from Issue #5/#7)
python -m v2d.world_calib.cli.calibrate \
    --data-root ~/v2d_track3/data/hf/track_3/public \
    --camera ego_cam_a --camera ego_cam_b --camera ego_cam_c \
    --fit-episodes 3,7,15 \
    --test-episodes 27,43 \
    --keypoints-dir results/keypoints \
    --out-dir results/calib
```

Outputs: `REPORT.md` (stages, recovery/transfer metrics, spatial error map,
consistency) and `calibration.json` (`T_world->cam` as `R`/`t` + `quat_xyzw`
per camera).

## Correspondence input format (real path)

`--keypoints-dir/{camera}_ep_{episode:06d}.npy`, shape `(T, B, K, 5)`:
row `[obj-x, obj-y, obj-z, img-u, img-v]` per object mesh point / image
keypoint pair, `visible == False` frames ignored.  Filled by the object-tracking
track (FoundationPose/SAM, Issue #5/#7).

## Self-test results

With ~1 px observation noise the engine recovers each camera's transform to
~0.1 mm translation / 0.01 deg rotation; the Huber-refined reprojection RMS
matches the injected noise and held-out transfer estimates generalize (see
`results/calib_selftest/REPORT.md`).

## Real data (verified)

The Track 3 **public** split is available at
`nvidia/video_to_data_challenge` (HF), expected at
`~/v2d_track3/data/hf/track_3/public` (matches `reference_loader.py` and the
CLI default):

```
meta/camera_calibration.json   cameras{exo,ego}_cam_{a,b,c}: resolution=[W,H],
                               fx/fy/cx/cy, distortion_coefficients[14],
                               extrinsics_to_stereo_left (4x4, only the camera
                               physically offset from the stereo-left reference)
meta/info.json + episodes_metadata.jsonl
data/chunk-000/episode_*.parquet    observation.objects{name,pose[7] w-first,
                                    visible} + frame_index/capture_time
mesh/<name>/{name}.glb (or _visual/_collision)
urdf/
videos/chunk-000/observation.images.ego_cam_*/episode_*.mp4
```

Verified on the real file: all six cameras parse (14-pt distortion), GT world
poses load via `reference_loader`, and the ego stereo baseline measures ~75.5 mm
against a declared 75.00 mm (factory residual, 14x expected).

The one missing input for a real solve is the per-camera **2D keypoint cache**
(`{camera}_ep_{ep:06d}.npy`, shape `(T,B,K,5)`) produced by the object-tracking
track (FoundationPose/SAM, Issue #5/#7).

## Notes

- Stage 2 (3D geometric refinement) from the design is skipped deliberately:
  the challenge provides no camera-space 3D references.
- Proxy episodes `(0, 11, 23, 31, 39)` are never tuned on; a guard warns if a
  proxy episode is passed as a fit frame.