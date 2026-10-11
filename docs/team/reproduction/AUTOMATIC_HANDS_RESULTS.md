# Automatic hand recovery evidence

This is an experiment log, not acceptance of hand accuracy or robot targets.
Development episodes are 12 and 22. Reserved recovery holdouts are 0, 11, 23,
31 and 39. No private evaluation episodes were used.

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

## Validation status

Twenty automated controls pass, including missing depth, wrong depth,
silhouette mismatch, occlusion, cross-side ambiguity and freeze rejection.
The reproduction controls reject missing support and a deliberately displaced
hand. All 275 revised e012 exports were checked with the existing native MANO
renderer: maximum vertex difference is 0.00013 mm, below the 0.01 mm conversion
tolerance. This is format preservation, not source-pose accuracy.
The previous saved-fit reproduction result does not validate this fresh-inference
method on another machine. Clean installation, cross-GPU numerical agreement,
downstream joint refinement and new Dex3 replay remain separate checks.
