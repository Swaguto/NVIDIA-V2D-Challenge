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

Four isolated MANO repair experiments retained cameras, object poses, shape,
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
| Fixed initial self-visible vertex ordering and stereo targets | 232 / 18 | 9.31 / 3.79 mm | 95.78 / 29.01 px |
| Refreshed opaque front triangles at source pixels | 174 / 43 | 9.04 / 6.06 mm | 41.55 / 26.68 px |

All four repairs are **rejected**. Fewer collision witnesses are insufficient while
substantial intrusion remains and projected geometry moves away from its
original fit. The first repair's full-frame depth/mask check does not show an
accuracy improvement. None of these objectives enters the baseline or replaces
direct observations. Private receipts preserve the failed attempts and frozen
experiment code; these are development diagnostics, not a released recovery
method or untouched-episode validation result.

The third experiment exposed a correspondence failure: keeping an initial
vertex's depth target after articulation moves can pull that vertex toward a
different part of the source hand. Its full 343-frame check reduces left mask
precision to 84.51% and increases the stereo residual to 15.11 mm. It is not a
recovery improvement.

The fourth experiment refreshes front-triangle visibility every 25 iterations
at calibrated source pixels, fits confidence-filtered stereo, and penalizes
projected hand vertices outside the source hand mask. It uses a fixed four-pixel
sampling stride and at most 1,024 pixels per supported frame, with no episode
frame or crop choices. Independent ray/plane and nine finite-difference
coordinate controls pass. Every visibility refresh agrees with the CPU raster
depth within 1e-8 m. Its full-frame median mask precision is 95.57% / 96.55%, and
stereo residual is 7.82 / 5.45 mm. The modest stereo improvement does not resolve
the remaining material witnesses or establish contact accuracy. Neither fit was
retargeted or promoted to Isaac Sim.

An additional all-frame self-visibility check confirms that the original worst
left and right material vertices are hidden behind their own reconstructed
hand surfaces. Their positive source masks therefore do not identify those
particular vertices. Visible source pixels can constrain front surfaces;
hidden vertices remain uncertain. New fitted poses are inferred development
geometry, and never become new direct observations merely because the original
support mask is unchanged.

## Additional object placement diagnosis

The frozen input's missing object masks are still missing. A separate private
development run produces **fresh estimated masks**, using the original recorded
automatic Grounding DINO seeds, native SAM2 and all 343 exact source PNGs.
It does not substitute these masks into the original source lock or claim
byte-identical recovery. The checkpoint, model code, source images and runtime
are recorded. The executed A40 run uses Torch 2.11.0+cu130 and a cache of eight
normalized PNG tensors; both preceding incomplete local attempts are retained.

All 686 new masks were byte-verified after transfer. The original source object
pose statuses remain unchanged. A pot/lid overlap rule marks 61 pot masks
ambiguous; the source check measures 282 observed pot poses with valid fresh
masks and all 343 observed lid poses. It excludes source hand pixels from the
object measurements and keeps inferred object timestamps separate.

| Median over observed, mask-valid frames | Pot | Lid |
|---|---:|---:|
| Rendered visible pixels inside fresh object mask | 82.95% | 92.76% |
| Fresh visible object mask covered by rendering | 96.06% | 97.54% |
| Median absolute camera-depth residual | 12.92 mm | 6.54 mm |

These are estimated-mask/stereo diagnostics, not object-pose ground truth. The
source-only comparison at frame 200 shows that the fresh pot mask omits much of
the handle. Its precision therefore cannot establish whole-object shape or
handle-placement accuracy. Source masks, depth, object pose and contact geometry
must be assessed together before attributing the remaining hand/material conflict
to either hand tracking or object tracking. No object pose, scale or hand target
is changed by this diagnosis.

A separate rigid-pose proposal test uses the earlier conservative CAD diagnostic
protocol: fixed scale/shape, at most 20 iterations, a 6 cm translation bound and
a 10 degree rotation bound. Every fifth selected stereo point is withheld from
the fit. This is a spatial split of the same estimated depth, not independent
video validation. Grasp labels, contact scores and reference poses are unused.

It fits 282 pot and 282 lid timestamps; 61 rows lack sufficient stereo support,
and 61 have uncertain source pose/mask identity. **None of the 564 proposals meets
the predeclared development-review conditions.** Median nearest-surface distances
over fitted frames change from 5.59 to 4.99 mm for the pot and from 4.29 to 3.42 mm
for the lid. Some proposals also reduce visible-mask recall substantially.
Camera-axis depth residual and nearest-surface distance are different measurements;
the former cannot be treated as an object translation error. These results do not
support moving objects globally to make a collision diagnostic pass. All proposed
poses are stored separately, remain inferred and unaccepted, and enter neither
the baseline nor the Dex3 replay.

The next physical repair must enforce observed hand/object depth ordering and
address contact geometry and morphology while checking source placement and
timing together. Increasing visibility or further
loosening a collision penalty cannot establish an accepted task. Issue9 and
training remain deferred.
