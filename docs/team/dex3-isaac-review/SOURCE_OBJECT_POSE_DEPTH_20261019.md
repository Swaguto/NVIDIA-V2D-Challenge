# Source object pose and depth checkpoint

## Compute state

The owner switched the preferred provider back to Brev. Its CLI health check,
SSH and actual STOP command succeeded. The existing L40S was idle when inspected
and was confirmed STOPPED afterward. Watcloud has no active jobs. These CPU
studies launched no GPU job or PPO step. Future GPU work still requires a
concrete candidate, bounded runtime, an independent stop guard, private verified
backup and provider STOPPED confirmation.

## Retained source poses

The new frame-50 comparison uses the participant's fixed reconstructed spoon
mesh and the pose pool whose SHA matches the frozen temporal path: **pool v7**.
The first audit selected the older v6 pool, and its hash assertion rejected it
before any comparison. The failed producer, guard and binding diagnosis remain
available. No saved pool, temporal path, mesh or hand target was rewritten.

All four retained poses leave a source mismatch. Their projected masks cover
46–55% of the inferred source object mask. Their median inferred fingertip
distances to the visual mesh range from 18.23 to 26.92 mm. These distances are
between estimates; the fingertips are not contact labels. The source hand
remains unqualified. This limited four-pose comparison cannot exclude other
poses or distinguish source hand error from object shape error.

## Separate pose experiments

Two new one-frame experiments keep the mesh and uniform scale fixed. Neither
uses hand points in its objective. Their CPU guards enforce 240 seconds and
2 GiB RSS. All source data and historical failures remain unchanged.

| Control | Image agreement | Independent depth diagnostic | Outcome |
| --- | --- | --- | --- |
| Frozen selected pose | mask IoU 0.4817; 54.71% coverage | 310/512 held-out rays hit; hit-only median error 8.97 mm | Retained control; no physical qualification |
| Estimated visible cloud to nearest mesh vertices, selected seed | IoU falls to 0.3939 | 258/512 rays hit; hit-only median error 7.87 mm | Rejected: source image/coverage regression |
| Image-only pose, selected seed, fixed camera z translation | IoU rises to 0.6860; 72.79% coverage | 422/512 rays hit; hit-only median error rises to 27.14 mm | Rejected: depth regression and incomplete coverage |

The visible-cloud experiment used a declared even/odd checkerboard split with
512 training and 512 held-out depth pixels, Huber loss at 10 mm, and four saved
pose seeds. The image-only control used even image pixels for mask IoU and
tested odd pixels afterward; odd-pixel IoU improved from 0.4820 to 0.6847. This
checkerboard test is not an independent episode or a generalization result.
Neither control used the held-out depth values in its solver. The image-only
control used no depth values at all. Every missing ray remains in the report;
hit-only depth summaries do not excuse missing coverage.

The image-only experiment also increases median inferred tip separation from
23.24 to 31.08 mm. A visually improved outline therefore does not establish a
grip. Conversely, the cloud fit can bring an inferred fingertip within 0.5 mm
of the mesh while worsening the source silhouette. Such proximity alone is not
a valid success criterion.

## Depth and mask authority

An audit retains all 9,729 mask pixels with finite, positive, left/right
consistent cached depth. Their median is 0.4082 m, but **46 pixels exceed 1 m**,
up to 2.398 m. All 46 lie within 2 pixels of an inferred mask boundary. The
512-pixel training and held-out subsets each contain three such estimates.
The source overlay also shows a disconnected mask component on the distant
round marker. The model input was 512×320; the cached output is 1280×800.

Left/right consistency does not certify semantic foreground identity or metric
accuracy. Boundary interpolation and segmentation errors are hypotheses; this
audit does not prove either cause. Its distance cohorts are descriptive, not
new rejection rules. No source sample, validity flag, mask component or held-out
failure was deleted or relabeled.

## Next experiment and gates

Test a joint source observation model using projected visibility and first
surface depth, with fixed geometry first. A nearest-vertex cloud loss can match
the wrong surface; an image-only loss can match the outline at the wrong depth.
Treat uncertain mask/depth evidence explicitly and retain the old reports.
Do not repeat these completed fits or tune against their held-out failures.

Any changed pose or geometry invalidates prior collision certificates. A new
full reference must pass source checks, actual morphology/URDF/motion limits,
the committed **1 mm full native reference-contact gate**, swept checks and
grasp checks before PPO. There is no newly qualified reference, physical task,
policy improvement or hidden leaderboard score.

Exact code is archived under
`experiments/scored_baseline_20261018/source_pose/`, with SHA bindings and
restoration instructions. Footage, model weights, private arrays and images
remain outside Git. Independent teammate reproduction is unverified.
