# Automatic hand recovery evidence

This is an experiment log, not acceptance of hand accuracy or robot targets.
Development episodes are 12 and 22. Reserved recovery holdouts are 0, 11, 23,
31 and 39. No private evaluation episodes were used.

## Frozen held-out evaluation: candidate rejected

The frozen method ran on e000, e011, e023, e031 and e039, totaling 1,444 source
frames. They were untouched by tuning this recovery method, but had previously
been processed by the baseline; they are not claimed to be never-seen data.
No episode-specific recovery frame or crop overrides were used.

| Episode | Frames | Baseline supported L / R | Automatic consistency passes L / R |
|---|---:|---:|---:|
| e000 | 373 | 231 / 124 | 188 / 250 |
| e011 | 137 | 133 / 122 | 125 / 73 |
| e023 | 325 | 316 / 99 | 275 / 276 |
| e031 | 290 | 196 / 133 | 288 / 54 |
| e039 | 319 | 318 / 113 | 291 / 222 |

These columns use different support criteria: the baseline records carry their
original depth support, while the candidate additionally applies image/motion
checks. Neither is a count of ground-truth accurate hands. On 1,603 shared
observations evaluated with the same current masks/depth criteria, the baseline
passes 1,550 and the candidate passes 1,394. Candidate-prompted masks and different
alignment histories remain limitations of this comparison. It is not independent
3D ground truth, but it does not support a claim of improved placement consistency.

Total supported observations rise from 1,785 to 2,042 under those different
criteria, while present/missing transitions rise from 191 to 289. Longest gaps
improve for five episode/hand pairs and worsen for five. For example, e031 left
improves from a 25-frame longest gap to one frame, but e031 right worsens from
63 to 219 frames. Seven of ten pairs have more transitions, two have fewer,
and one is unchanged. Counts include absent/out-of-view hands and cannot by
themselves identify tracking flicker. Full per-hand counts, shared-frame checks,
gaps and camera-conversion residuals are in the
[comparison CSV](automatic-hand-heldout-comparison.csv) and
[machine-readable report](automatic-hand-final-comparison.json).

Qualitative enlarged review found passing-but-visibly-mismatched examples in
all five holdouts: e000 left36, e011 right112, e023 left30, e031 right247, and
e039 left120. See [recorded findings](automatic-hand-review-findings.json).
Review was risk-focused, not exhaustive annotation; no false-acceptance rate is
claimed. Occluded fingers and precise 3D contact accuracy remain unmeasured.

**Do not promote this candidate to production reconstruction/contact targets.**
Recovery is automatic and frozen, but improved accuracy and an all-episode
flicker fix were not achieved. More coverage must not be presented as a safer
motion target. The next candidate needs camera-aware fitting and independent
finger/keypoint validation on development data, followed by a newly reserved
evaluation set. These five evaluated episodes must not be reused as untouched
evidence after tuning on their results.

## Preserved development results

| Candidate | Episode | Frames | Left consistency passes | Right consistency passes | Left/right support transitions |
|---|---:|---:|---:|---:|---:|
| Upright tiles, independent current-image fits (`dev-suite-v1`) | 12 | 181 | 145 | 95 | 27 / 38 |
| Same method | 22 | 189 | 100 | 21 | 22 / 4 |
| Independent-view preprocessing plus mirrored search (`dev-suite-v2`) | 12 | 181 | 153 | 122 | 32 / 28 |
| Same revised method | 22 | 189 | 163 | 59 | 28 / 17 |

These runs completed with unchanged method hashes. They are not equivalent to
visible-hand recall: their denominators include absent and occluded hands, and
the checks are proxies. Fewer support transitions can mean a hand is missing
for most of the video; e022's right hand is an example. Longest unsupported runs
were 9/16 frames on e012 and 20/151 frames on e022.

An earlier rotated-image detector trial was rejected after visual review found
the same visible left hand assigned to both sides at e012 frame 25. It is not
part of the candidate. Upright views and exclusive cross-side association remove
that observed duplicate in the reviewed sequence. This does not establish that
all identity errors are eliminated.

The e012 v5 diagnostic (before the clean v1 run) compared the old refined
reconstruction and new proposals against the same current-image masks and valid
stereo depth on common frames. Image/depth checks passed 75/89 versus 71/89 left,
and 86/109 versus 63/109 right (new versus reference). The new fit had not received
joint refinement, while the reference had. This is neither ground-truth pose
accuracy nor a fully matched reconstruction-stage ablation.

The clean v1 e022 comparison is negative: image/depth checks pass 56/89 versus
57/89 left and 17/34 versus 19/34 right (new versus saved aligned baseline, on
common frames). The baseline has 89 left and 62 right depth-supported records.
The candidate therefore cannot be promoted based on its e012 gains alone.

The revised v2 e012 comparison against saved aligned baseline records passes
80/88 versus 83/88 left, and 104/114 versus 72/114 right (new versus baseline,
same mask/depth checks on common frames). That is a substantial right-hand
consistency gain with a small left-hand regression on shared frames. These
numbers use the original aligned baseline, whereas the earlier v5 diagnostic
used the later refined baseline; do not merge those comparisons.
The revised e012 longest gaps are 4 left and 18 right frames. The right gap is
frames 20–37, around the mug grasp with substantial occlusion and rejected fits.
Original baseline longest gaps were 30/25 frames (left/right), with 18/23 support
transitions. The revision reduces long gaps and increases coverage, but has more
short transitions than that baseline. It is not a complete flicker fix.

For revised e022, matched image/depth checks pass 84/86 versus 83/86 left and
52/62 versus 53/62 right (new versus baseline). Placement consistency is similar
on shared frames, while more left-hand observations survive. Longest gaps are
6/106 frames. The method was frozen only after both clean development reports
completed, before running any reserved-episode inference. Freeze SHA256:
`81d13102c669e271a8c4e0cc107e50f9e8887eeb8c325cbda8f4b1410690a758`.

## Visual accuracy findings

The e012 reviews show plausible palm placement in several lifting intervals,
but missing visible open hands at the start/end and rejected fits around grasp
changes remain. For example, the right-hand fit is repeatedly rejected around
frames 48–60. In e022, the right hand holding the large object is mostly absent
in the initial candidate. Left-hand rejections also persist during the holding
sequence. These prevent a claim that flicker is fixed across episodes.

Contact-sheet review checks gross position, visible shape and handedness; it is
not fingertip-level annotation or normal-speed temporal acceptance. Hidden
fingers behind the objects cannot be validated from these views. Masks, depth
and optical flow can reject inconsistencies but cannot certify contact accuracy.
In particular, depth translation is fitted using stereo, so its residual is an
in-sample consistency measure, not an independent accuracy measurement.

Enlarged, automatically selected crops reveal false acceptance by the consistency
checks: e012 left frames **22 and 148** have visible fingertip/outline mismatch
despite passing. Left frame 151 covers only part of the visible finger silhouette
while also passing. These are visual accuracy failures for contact-target use;
the counts above must not be described as accurate recovered-hand counts. The
frozen candidate is **not approved for production contact or motion targets**.
This finding is recorded without changing the frozen thresholds or adding
episode-specific exclusions. Stronger independent finger/keypoint evidence is
needed before a future candidate can be accepted.

A further development audit identifies a camera-conversion limitation. Rigid
PnP approximates HaMeR's virtual-camera projections under the measured camera;
it does not preserve them exactly. Across e012 proposals, maximum conversion
error is 38.33 pixels, with median per-pose mean error 4.46 pixels. For e022,
those values are 34.74 and 3.37 pixels. The visually flagged left frames 22,
148 and 151 have maximum conversion errors of 16.01, 19.84 and 19.92 pixels.
This can contribute to image mismatch, but does not isolate all pose errors.
It is distinct from the near-exact serialization/native-export check: a saved
calibrated mesh can reproduce perfectly while the camera conversion that created
it is inaccurate. A future method needs camera-aware fitting and a projection
error gate, without deforming hands merely to make an overlay look correct.

The crop review selects regular 30-frame samples, both sides of support changes,
and the five lowest accepted precision/recall and highest depth-residual cases
per side. It is a reproducible risk-focused sample, not exhaustive annotation.

## Validation status

Five reproduction-check controls pass after hardening the checker to reject
missing geometry files and non-finite coordinates, in addition to dropped
support and deliberate displacement. The earlier 20-control run included three
of these five cases plus 17 recovery/selection controls. The two added controls
do not change frozen inference.

The recorded evaluation suffered a CUDA unknown error during e039 fitting.
Completed detections and completed e000/e011/e023/e031 fits were retained;
the interrupted e039 fitting stage was preserved separately and restarted in
full with unchanged frozen functions and seeds. The original failed receipt and
separate recovery log are retained. This is an infrastructure recovery, not a
change to the evaluated method.

The post-inference input inventory matches all 15,758 recorded files across the
20 cached public episodes exactly. No source image, depth map, validity mask or
manifest changed during fitting.

Twenty automated controls pass, including missing depth, wrong depth,
silhouette mismatch, occlusion, cross-side ambiguity and freeze rejection.
The reproduction controls reject missing support and a deliberately displaced
hand. All 275 revised e012 exports were checked with the existing native MANO
renderer: maximum vertex difference is 0.00013 mm, below the 0.01 mm conversion
tolerance. This is format preservation, not source-pose accuracy.
The fresh e012 detection repeat exactly matches all 181 complete detection
records, including boxes, confidence values and selection decisions, on the
same GPU. The full geometry/support comparison now passes: all 324 proposed
hands have identical raw and adjusted joints/vertices (maximum measured
difference zero), and proposal/support decisions match exactly. All 2,317
consistency-passing native exports across the five holdouts plus the e012 repeat
also pass the geometry-preservation check. These are same-GPU reproduction and
serialization results, not source-pose accuracy.
The previous saved-fit reproduction result does not validate this fresh-inference
method on another machine. Clean installation, cross-GPU numerical agreement,
downstream joint refinement and new Dex3 replay remain separate checks.
