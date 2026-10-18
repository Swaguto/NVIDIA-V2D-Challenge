# e037 carry reference: source gap diagnosis

## Current decision

**Do not train the current e037 candidate.** It still passes the independently
verified 151-knot source/contact/motion screens and 3,805 discrete interpolation
samples, but a new check demonstrates that it cannot represent the source grasp
through the lift. Those geometric checks establish nonpenetration, not contact
or a usable grasp. No physical or leaderboard improvement has been established.
The previous complete baseline and every failed experiment remain intact.

## Source evidence and traced cause

The source video shows the right hand carrying the spoon at frame 50. Original
right-hand records are not directly observed or metric-supported for **25–99**
(75 frames, 3.75 seconds). Their validity also fails for **36–69** (34 frames,
1.70 seconds). The original robot wrist during 36–69 reproduces straight world
interpolation between 35 and 70 to **5.55e-17 m** component error. It therefore
misses the spoon's approximately **301 mm** upward/away excursion, which peaks
at frame 50. Later small morphology/contact repairs preserve this upstream gap.
This is evidence of an unsuitable target, not a proof explaining every episode
or policy failure.

An independent separation calculation uses the actual native convex geometry.
Unit-axis separating distances bound true Euclidean separation from below;
vertex-pair distances bound it from above. Minima over all nine spoon hulls
bound distance to their union. Positive native distance queries are not used.
The analytic box controls pass. At frame 50:

| Right-hand surface | Distance lower bound | Distance upper bound |
| --- | ---: | ---: |
| Distal thumb | 271.583 mm | 271.835 mm |
| Distal index | 218.939 mm | 219.005 mm |
| Distal middle | 208.516 mm | 208.548 mm |
| Entire thumb | 254.656 mm | 259.166 mm |
| Entire index | 211.830 mm | 211.848 mm |
| Entire middle | 201.598 mm | 201.757 mm |

The first full separation audit exceeded its 240-second worker limit. It is
retained as incomplete. A new bounded 11-frame audit completes, includes frame
50 and neighbouring transition phases, and independently reproduces these
bounds. No claim of full 151-frame separation coverage is made.

## Tested gap hypotheses

The existing optional object-relative algorithm proposes 34 inferred frames
using visual anchors 35/70 and 75 using direct/metric anchors 24/100. Both pass
that algorithm's endpoint association/drift checks. They require respectively
**254.457 mm** and **314.023 mm** maximum root changes. Both therefore fail the
original 25 mm morphology bound and remain unqualified hypotheses. Neither was
passed off as a small morph correction, used for training, or granted fabricated
source confidence/contact labels. A new upstream source-supported hand recovery
must establish its own evidence before a new command reference is evaluated.

## Independent CPU hand detector investigation

A separate pinned MediaPipe model runs locally on the Mac CPU; footage never
leaves the machine. Model SHA256:
`fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`.
MediaPipe is pinned to 1.1.1 in a separate dependency directory. Its hand-relative
world coordinates are **not camera metric coordinates** and are not used as
absolute depth. [Official output-coordinate documentation](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python).

- Independent whole-image inference detects no hands on the original eleven
  samples at the default 0.5 thresholds. That failed control is retained.
- A fixed full/left-half/right-half crop and four-rotation diagnostic produces
  raw 2D candidates on some samples. The public documentation-image control
  still gives zero detections; it is not reported as a successful control.
  Source overlays at frames 0 and 75 visually place some candidates on the
  right hand. Anatomical identity and metric accuracy are not certified.
- Full-clock independent image inference completes on all **151 frames**,
  reusing all five prior frame results exactly. During the original 34-frame
  invalid interval, both cameras have raw hand candidates at **22 frames**.
  Frames **48–51** still have none in either camera. Candidates are not counted
  as recovered metric hand observations.
- A separate strict stereo screen checks all 151 frames with fixed thresholds:
  median epipolar error ≤1 px, p95 ≤3 px; finite positive range, per-point range
  0.15–1.4 m, estimated stereo-depth agreement ≤10 mm, wrist/knuckle support.
  The synthetic projection/triangulation control passes. **Zero complete hand
  pairs qualify**: 115 pairs fail the epipolar screen, 67 disagree in predicted
  handedness. None reach the depth screen. No threshold was lowered afterward.
  Per-view detection differences and ambiguity remain explicit.

## Next work

Recover source-supported partial wrist/knuckle evidence or repair the original
hand-detection/fit stage during the carry. A partial observation must retain
unknown points and cannot be promoted into a complete hand. Preserve the failed
whole-hand stereo screen and original missing flags. A separately validated
object-relative completion may then connect genuinely supported anchors; it
remains inferred during missing frames. New commands require unchanged native
1 mm, source-fit, global morphology and URDF/motion checks before physics/PPO.

The private GPU plan v2 is marked `BLOCKED_SOURCE_HAND_CARRY_REFERENCE` and
`do_not_launch=true`. The old plan and private static candidate are preserved.
Brev was independently confirmed **STOPPED at 21:00:39 UTC**; no start, new paid
session or training occurred. Its separate shutdown API issue remains unresolved.
All 20 selected policies still have local MP-SR zero. No submission was uploaded.
