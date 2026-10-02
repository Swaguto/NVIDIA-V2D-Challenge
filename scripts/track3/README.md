# Track 3 pose pipeline — status and open problems

RGB-only evaluation is solvable: the ego rig is a stereo camera and the organisers publish
the factory calibration, so **metric** depth is recoverable without any depth sensor stream.
This document records what is validated, what is measured, and what is still broken.

## Pipeline

```
ego_cam_c (left) ┐
                 ├─ stereoRectify + initUndistortRectifyMap ─ SGBM ─► metric depth
ego_cam_b (right)┘
rectified RGB ─ Grounding DINO (name-only prompts) ─ SAM2 ─► per-object mask
mask ∩ valid depth ─► object point cloud ─ trimmed symmetric ICP vs public CAD ─► pose
```

`scripts/track3/register_poses.py` implements this and writes the per-frame JSON layout
`scripts/eval/score_track3_official.py` reads.

## Validated

**Stereo depth** (`scripts/track3/validate_stereo.py`, episodes 0 and 11):

- Left–right disparity consistency beats deliberately mis-rectified controls at every
  tolerance (23.6% violations at 1.5 px vs 34.6% / 40.9% when the right image is shifted by
  2 px / 4 px).
- Temporal depth std is 1.7–3.2% of the depth level.
- RANSAC recovers the support plane to ~5 mm RMS.

**Segmentation** (`scripts/track3/probe_segmentation.py`, episode 0 frame 120): Grounding DINO
separates the known objects from name-only prompts (`white pot` 0.637, `white pot lid` 0.631)
and SAM2 yields masks that back-project to ~0.19 × 0.17 × 0.27 m clouds at 0.535 m — plausible
for a pot and lid.

**CAD scale**: the public meshes are at true physical scale. The pot lid fits identically at
fixed scale 1.0 and at a free scale of 0.955, so scale is pinned to 1.0. This matters because
the official frame-0 alignment is SE(3), not Sim(3), so scale error is scored.

## Measured baseline

Official metric, public episode 0, `object_to_cam`, 97.6% coverage:

| AUC | MP-SR | SP-SR | RPE | MPPE |
|---|---|---|---|---|
| 0.0505 | 0.00 | 0.00 | 17.33 cm | 13.69 cm |

Reference points: oracle 1.0; the earlier depth/mask pipeline averaged 0.0688 AUC on five
proxy episodes. Target is AUC ≥ 0.90.

## Open problem: rotation, and it dominates

`scripts/track3/diag_relative_geometry.py` compares the *relative* transform between the two
objects. This is frame-invariant, so it is comparable to GT without knowing the camera→world
extrinsic, and it isolates the failure:

- relative rotation error: **median 150°** (near-random)
- relative translation error: median 19.8 cm
- yet predicted object separation is close to GT (4–7 cm vs 5.68 cm)

Two candidate explanations were tested and rejected:

1. *A fixed per-object frame offset* (e.g. GLB vs URDF convention). Rejected:
   `diag_rotation_offset.py` finds a mean offset of 168°, but removing it leaves a 95° median
   residual, so the error is not constant.
2. *Canonical, snap-able orientations.* Rejected: `diag_gt_orientation.py` shows objects
   genuinely rotate within episodes (within-episode spread median 6–97°, p95 up to 179°), so
   orientation must be tracked rather than snapped.

The residual signature points at the real cause: the pot lid registers to a **7 mm** residual
while its in-plane angle is near-random. That is the classic symmetric-object failure — a pot
and a lid are close to solids of revolution, so shape alone cannot observe rotation about the
vertical axis. Because the relative translation is expressed in the other object's frame, an
erroneous relative rotation also inflates the translation error, which is why both RPE and
MPPE are poor.

Note the metric's frame-0 SE(3) alignment uses object slot 0 only, so object 0's in-plane angle
is absorbed at frame 0. Object 1's in-plane angle relative to object 0 is not, and is an
irreducible ambiguity for near-symmetric shapes under shape-only registration.

### What is needed to fix it

A cue beyond geometry, or an explicit ambiguity model:

- appearance-based (silhouette/photometric) alignment, which can resolve pose for shapes with
  texture even when geometry is symmetric;
- explicit symmetry-aware fitting (estimate the symmetry axis, constrain the in-plane angle
  with a temporal smoothness prior instead of letting ICP choose freely per frame);
- temporal filtering that penalises angular acceleration, so a smooth ground truth is tracked
  as smooth rather than as per-frame independent local minima.

## Separate blocker: CD-O spoon

`wooden_spoon` has no mesh or URDF anywhere in the released data but appears in 10 of the 36
scored `(episode, object)` pairs, and `pack_reconstruction.py` hard-fails on any missing mesh.
Tracked in issue #29.

## Reproducing

```bash
# depth + calibration validation
python scripts/track3/validate_stereo.py 0 120

# masks + point clouds for one frame
python scripts/track3/probe_segmentation.py 0 120

# poses for one episode
python scripts/track3/register_poses.py --episode 0 --stride 1 --out /tmp/pred

# official score
python scripts/eval/score_track3_official.py \
    --predictions /tmp/pred --kind object_to_cam --episodes 0 \
    --data <track_3 public root> --oracle
```

Defaults point at the remote working copy; override with `V2D_PUBLIC`, `V2D_VIDEOS`,
`V2D_REPO`, and `V2D_CALIB`.