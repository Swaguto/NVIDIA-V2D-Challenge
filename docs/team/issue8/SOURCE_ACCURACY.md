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

## Dex3 mapping and error semantics

An independent audit checks the mirrored left-palm convention and the source
joint ordering against actual native robot FK. It confirms that the current
targets are wrist, thumb tip, index tip and middle tip. The human MCP directions
point along the expected robot forward axis, and the thumb is on the expected
side. This diagnosis finds no additional global wrist flip or joint-name swap.

The source and robot still have different morphology. Both Dex3 long fingers
rotate about parallel palm-Z axes; their tip Z coordinates stay 57 mm apart,
so their Euclidean tip distance is at least 57 mm for any root pose. Median
source index-to-middle tip distances are only 33.60 mm left and 26.84 mm right.
The triangle inequality therefore gives median lower bounds of 23.40 and
30.16 mm on the **sum** of those two site errors, regardless of wrist placement.
This is a kinematic limitation of the current targets, not proof that either
reconstructed human finger is inaccurate.

There is a separate objective/reporting issue: the current Pink FrameTask uses
the SE(3) logarithm, so even with zero tip-orientation cost, its position objective
can depend on the supplied, unused target rotation. Changing only that rotation
changes the native QP Hessian and linear term. A development-only world-position
task makes both invariant; central finite differences check all 13 tangent
coordinates on each hand to less than 3.5e-11. This alternative does not infer or
observe a fingertip rotation.

Consequently, the archived left median tip translation-log norm of 26.76 mm
must not be described as Euclidean position error. Independent saved-FK distances
for the current baseline are 17.45 mm left and 17.59 mm right at the median.
The read-only `track3_review_dex3_targets.py` command reproduces those measurements
for any complete baseline episode, with hashes and explicit observation masks.
It does not change the IK solver or the replay, and it refuses development target
overrides. Twelve CPU controls check scene-translation invariance, scaled wrist
offsets, changed robot axes/chains, provenance/gap rejection and portable numeric
fingerprints that detect changed results.

Three predeclared e002 alternatives isolate position-only tasks, an outer-finger
mapping (human pinky to the second long robot finger), and both together. They
process all 343 frames, preserve 272/284 supported hands and all direct-observation
masks, and have zero finger position/velocity violations. No source hand, object,
camera, shape or scale is edited; no grasp annotation or reference pose enters
the optimization. Alternative robot targets remain inferred.

| Development variant | Median world tip distance L/R | Worst material vertex witness L/R | Frames with vertex witness >2 mm L/R |
|---|---:|---:|---:|
| Current baseline | 17.45 / 17.59 mm | 10.88 / 7.02 mm | 238 / 133 |
| Current mapping, position-only tasks | 18.57 / 17.59 mm | 10.91 / 7.03 mm | 238 / 133 |
| Outer fingers, original FrameTask | 9.84 / 11.42 mm | 10.97 / 6.56 mm | 238 / 134 |
| Outer fingers, position-only tasks | 10.59 / 11.48 mm | 10.96 / 6.56 mm | 238 / 134 |

Distances for the outer-finger variants use their different declared source
targets; these are not improvements in recovered human accuracy. Neither
position-only variant passes the combined sampled-pad-opposition and whole-hand
vertex check for any of the four source-labelled grasp intervals. Better fit
to joint targets alone has not made this task physically credible.

A fourth alternative adds the earlier bounded collision QP to the outer-finger,
position-only targets without loosening its guards or changing geometry. All
supported frames run 100 iterations. It reduces timestamps with vertex witnesses
above 2 mm to 164 left and 33 right, but still witnesses 10.86 / 6.10 mm intrusion.
Median wrist distances rise to about 20.5 / 10.1 mm; fingertip maxima reach
119.36 / 83.97 mm. Partial sampled contact checks pass at 28, 19, 33 and 0
timestamps in the four source-labelled intervals, respectively. These do not
establish any complete task or continuous collision-free motion. **All four
alternatives are rejected for physical acceptance and none replaces the baseline
or current Isaac replay.**

Independent native FK recomputation verifies 12,788 saved link poses for each of
the four alternatives, with zero position difference and rotation differences
below 2.1e-16 radians. This confirms the geometry being audited; it does not resolve
the penetration or placement failures.

The evidence calls for contact-aware retargeting that accounts for Dex3 morphology
and source-supported contact regions, rather than forcing all robot sites onto
human fingertips. Contact-region object geometry and source depth remain uncertain.
Any proposed grasp or robot offset must be stored as inferred, checked against
source placement and timing, and tested on untouched episodes after its method is
frozen. Issue9 and training remain deferred.

## Source-supported regions and static grasp feasibility

A new private diagnostic processes every source frame with the original direct
hand observations and observed object poses. It uses actual stereo samples in
exclusive hand masks, requiring all nine pixels in a 3 by 3 square to have valid
depth and consistent semantics, at most 10 mm local depth spread, and at most
10 mm disagreement with the visible reconstructed front surface. Samples must
be within 5 mm of a point on the original CAD surface, unambiguous between the
objects, and within the declared material-depth limit. All qualifying points are
retained; there are no selected optimization frames or crops. These are potential
contact regions, **not observed tactile contacts**.

This neighborhood check also found that an earlier private stereo-fit cache used
SciPy's default cross-shaped erosion, not a square. Corner-invalidity controls
demonstrate the difference. The stricter square filter rejects an additional
2,561 left and 4,590 right sampled pixels in this diagnostic. Earlier rejected
fit artifacts retain their original bytes. The published generic source review
uses individual valid pixels and never claims square-neighborhood confidence;
its measurements and the baseline trajectory are unchanged.

The region pass finds 4,244 left/pot points across 207 frames and 195 right/lid
points across 80 frames. The right side has at most five qualifying points in any
frame, so this pass does not establish complete contact timing. Anatomical labels
from the original joint chains remain inferred, and ambiguous labels stay
unknown. No human pose, camera, object pose, scale, support or display hold changes.

Separate experimental robot grasp templates test geometric feasibility near
these regions. They are static inferred Dex3 configurations, with explicitly
recorded robot root offsets; they do not repair human tracking. Two different
variants supply a left pot-handle and right lid-handle diagnostic with opposed
pad proximity. At automatically selected maximum-support source frames, an
independent check of every robot mesh vertex against both scene objects gives
maximum material penetration upper bounds of 1.70 mm left and 1.65 mm right.
These checks concern declared development geometry, not physical ground truth or
complete surface/volume collision. Source-camera ghost overlays are labelled
as static diagnostics, not Isaac Sim output.

The selected robot roots differ from the source human wrists by 69.1 mm left and
46.7 mm right at those diagnostic frames. Such morphology offsets require their
own source-placement justification. The left diagnostic also fails the solver's
0.5 mm sampled-clearance requirement; it is not an accepted solution. All 13
template records reproduce exactly in a second native run after protocol-text
corrections, including poses, metrics, iteration counts and failure statuses.
That is a same-machine repeat, not independent team validation.

**No complete task is accepted.** The two variants are not combined into a
trajectory or promoted to the baseline. A single frozen method still needs
source-backed approach and release timing, source placement, transitions,
self-collision and full-sequence scene checks before a new Isaac replay. The
original baseline replay, hand observation provenance and issue9 deferral remain
unchanged.

## Contact timing and self-collision follow-up

An automatic development diagnostic combines the strict source regions with
short windows of hand motion relative to each observed object. It requires source
support and proximity evidence; two stationary bodies alone cannot establish a
grasp. Its inferred batch segments preserve missing observations and source-frame
gaps. No grasp annotation, reference pose or display hold enters inference.

The method runs on all 343 frames before comparison with the pre-existing
source-video annotations. These coarse visible-grasp labels are validation data,
not tactile ground truth. The complete source denominators remain visible:

| Source-labelled grasp | Predicted core / all core frames | Predicted core / source-supported core frames |
|---|---:|---:|
| Left pot lift/lower |166 / 185|166 / 174|
| Right lid remove/place |59 / 62|59 / 61|
| Right lid regrasp/replace |69 / 72|69 / 72|
| Left pot regrasp/release |59 / 62|59 / 60|

No hypothesis falls outside a labelled grasp interval; two fall in ambiguous
right-hand boundary frames. Fourteen source-supported core frames are missed,
and another fourteen lack the required original observations. Strong source
proximity sometimes remains despite estimated wrist motion. The thresholds are
not adjusted to these labels, and complete contact timing remains unaccepted.

An independent Coal/Pinocchio audit now checks all 21 nonadjacent collision-mesh
pairs on each hand. Only directly joint-adjacent pairs are excluded. It detects
thumb–palm surface intersections in the earlier left diagnostic and rejects that
pose. The earlier right diagnostic has no detected nonadjacent surface
intersection. This check does not certify volume containment or continuous motion.

A new private fitter adds conservative convex-hull self-clearance constraints.
Every original mesh vertex is verified inside its link hull. Independent box
overlap/separation and finite-difference controls check the distance sign and all
13 tangent coordinates on twelve configurations. Constraints saturate above
5 mm; the required 0.5 mm clearance and its feasible set are unchanged. The raw
distance derivative discrepancy at a far, inactive neutral feature is retained
in the control report rather than presented as a passed raw derivative check.

The fitter produces nineteen static candidates. All ten left candidates have
zero detected nonadjacent triangle-surface intersections, but **none of the
nineteen passes the complete intended-pad contact proxy**. Several candidates
still fail solver or geometric constraints. An additional audit computes exact
vertex depths against exposed square faces of the same declared voxel-cell
union, with exhaustive and analytic controls. It distinguishes actual depth in
that model from the earlier conservative upper bounds; it is not a new physical
object model or a complete robot-surface certificate.

These are reproducible diagnostic methods and preserved failures, not accepted
motion targets. No new full trajectory or Isaac result is produced by this
follow-up. Source placement, contact surfaces, complete timing, transitions and
whole-scene collision checks remain required before physical-task acceptance.
