# Repeating the source accuracy diagnostic

This diagnostic compares frozen reconstructed hands and objects with the original
rectified RGB, hand masks and calibrated stereo depth. It helps locate alignment
errors before retargeting. It never certifies physical grasp success or converts
an occluded contact into a directly observed pose. The script accepts any episode
bundle; it contains no e002/e012 frame or crop choices.

## Inputs and command

Use the validated episode bundle and the frozen prepared adapter inputs from
the reconstruction run. Keep its `source/provenance/inputs-lock.json`. Acquire
the source rasters under their original dataset permissions; code access alone
does not supply licensed video, MANO assets or private episode artifacts.

Install Python 3.11 and the pinned CPU dependencies from the same Git revision:

```bash
python -m pip install -r scripts/reproduce/requirements-cpu.txt
python robotic_grounding/scripts/retarget/track3_review_source.py --bundle /data/episode-bundle --prepared-inputs /data/frozen-inputs --out /data/new-source-review
```

The command reviews every source frame by default. Output directories must be
new. `report.json` retains each frame's measurements, the source hashes, direct
observation counts and supported counts separately. `images/` contains source
RGB, visible reconstructed surfaces and a stereo residual panel every tenth
frame. `--image-every 1` saves every frame. Missing hands remain absent, and
missing object masks remain unknown. No held display pose is used.

If the locked source paths differ on your machine, pass `--path-map paths.json`.
Mappings change where files are read, not their required hashes. For example:

```json
[
  {
    "source_prefix": "/var/tmp/v2d-issue6-quality-fast/episodes/e002",
    "local_prefix": "D:/v2d/cache/e002"
  },
  {
    "source_prefix": "/var/tmp/v2d-issue7-e002",
    "local_prefix": "D:/v2d/issue7-e002"
  }
]
```

The longest matching source root wins. Parent traversal is refused. Every RGB,
depth and validity file must match its original source SHA256. Prepared masks
must match the correct frame and hand/object identity in the frozen source lock.
A PNG from another frame or hand cannot satisfy that requirement. The depth inputs are metric `.npy`
rasters and separate validity PNGs; the tool does not guess another encoding.

An optional `--witnesses collision-witnesses.json` includes collision diagnostic
points in the review and saves their frames automatically. That report must
carry the same `bundle_manifest_sha256` and a `witnesses` list with `frame`,
`kind`, `side` and `point_world_m`. The output reports the front surface and
whether each point lies behind it. Supplying `--frames` reviews a declared
subset only; it cannot establish an all-frame result.

## What the measurements mean

The CPU renderer rasterizes original object and MANO triangles at integer
calibrated pixel coordinates. It uses perspective-correct depth, both triangle
windings and the nearest visible surface. The diagnostic requires rectified
zero-skew intrinsics and all rendered geometry in front of its 10 mm near plane;
unsupported inputs fail instead of silently dropping a crossing triangle.

`rendered_surface_mask_precision` is the fraction of rendered visible hand
pixels inside its source mask. The masks can include forearms that MANO does
not model, so mask IoU alone is not a complete hand accuracy score. Depth errors
use only pixels where the rendered hand, source hand mask and valid stereo
depth overlap. Report summaries are medians/percentiles over frame-level
measurements, not pooled pixel errors. Masks and stereo depth are estimated
inputs, and visible-surface fit does not resolve hidden finger positions.

Six CPU controls cover ray/plane depth, occlusion, path relocation and mask
frame/identity integrity, and pass on Windows and Linux. For the e002
diagnostic it also matched 512 seeded Trimesh ray intersections across four
collision frames, with maximum depth difference 5e-16 m in the reviewed
renderer run. Four actual-frame measurements also agree exactly between
Windows and native Linux. This verifies rasterization and measurement
reproduction, not reconstruction accuracy.

## Current e002 evidence and rejected repairs

The fixed e002 bundle keeps all 343 source frames and 272 left / 284 right
supported hands. The source review gives these medians over supported frames:

| Measurement | Left | Right |
|---|---:|---:|
| Rendered visible hand pixels inside source mask | 96.07% | 97.22% |
| Median absolute stereo depth residual | 8.03 mm | 6.54 mm |

Object mask files are missing from this frozen reconstruction input. They remain
unknown; this diagnostic does not invent object silhouette measurements. Source
video review and the separate object tracking report are still required.

The recovered human mesh has derived-material lower witnesses of 9.55 / 6.66 mm
before retargeting. Its worst left/right points lie 8.67 / 28.77 mm behind the
front reconstructed surface in the source view. The robot's two worst points
are also behind reconstructed front surfaces. This does not prove physical
occlusion in the source video. In particular, the left-hand source mask and RGB
show a finger at the worst left point's pixel, while the reconstruction's front
surface belongs to the pot. That is a visible depth-ordering mismatch. The right
human point lies behind another surface of its own reconstructed hand. These
pixels do not establish the exact identity or placement of a hidden vertex.
Nearest original CAD surfaces remain 8.31 / 5.35 mm from the two human points;
neither nearest face is an added closure. Local face-normal signs are not a
global solid classification for these nonwatertight meshes. Actual material
interiors and contact geometry remain uncertain.

Two isolated MANO repair experiments retained cameras, object poses, shape,
scale, source records and all support masks. They optimized both hands over
every supported timestamp, using no grasp annotations, reference trajectories,
display holds, frame selections or crops. They used the explicit derived voxel
material, a collision penalty, near-contact retention, source vertex/projection
anchors and smoothing of changes over adjacent supported frames.

| Human-source experiment | Frames above 2 mm, L / R | Worst lower witness, L / R | Maximum projected vertex change, L / R |
|---|---:|---:|---:|
| Original frozen hands | 203 / 113 | 9.55 / 6.66 mm | 0 / 0 px |
| 8 mm translation, 2 degree root, 8 degree finger bounds | 135 / 48 | 9.01 / 6.04 mm | 19.13 / 15.31 px |
| 12 mm translation, 6 degree root, 25 degree finger bounds | 120 / 36 | 8.89 / 5.99 mm | 34.77 / 22.21 px |

Both repairs are **rejected**. Fewer collision witnesses are insufficient while
substantial intrusion remains and projected geometry moves away from its
original fit. The first repair's full-frame depth/mask check does not show an
accuracy improvement. None of these objectives enters the baseline or replaces
direct observations. Private receipts preserve the failed attempts and frozen
experiment code; these are development diagnostics, not a released recovery
method or untouched-episode validation result.

The next physical repair must enforce observed hand/object depth ordering and
address contact geometry and morphology while checking source placement and
timing together. Increasing visibility or further
loosening a collision penalty cannot establish an accepted task. Issue9 and
training remain deferred.
