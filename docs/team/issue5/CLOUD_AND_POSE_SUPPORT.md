# Handle preservation and competing 6D poses

October 4, 2026. These repairs and experiments improve sampled relative geometry;
**automatic acquisition, canonical rotation and issue #5 acceptance remain open**.
Keep PR #30 draft. Do not merge to main.

The subsequent [lid appearance and episode-0 replication](LID_SPIN.md) keeps this
baseline frozen, adds guarded spin diagnostics, and tests the same selection
method on eight episode-0 frames. Canonical rotation still fails. Its latest
suite has 156 tests; the 150-test results below are historical cloud/support
validation.

## What failed

The radial point-cloud filter used distance from a coordinatewise median. It can
discard valid elongated geometry and is not invariant to rigid rotation. In the
RGB-reviewed episode-2 handle regions, it retained 0 of 197 points at frame 147,
0 of 259 at frame 195, and 39 of 644 at frame 244. This removes a visible cue that
could distinguish pot orientations. These regions were reviewed from RGB, not GT.

The native filter now uses mean spacing to eleven neighbours, one pass, with
threshold `max(median + 3.5 * MAD, 2 * median)`. The floor protects uniformly
sampled boundaries. `--cloud-filter radial` retains the old calculation exactly
for baseline replay; historical probe helpers select it explicitly. New CPU
regressions verify connected-handle preservation, isolated-noise rejection and
rigid-transform invariance. The native pot wrapper defaults to radial to preserve
its historical baseline, and exposes explicit filter selection.

Preserving the handle alone is insufficient. Native global registration refines
only the three best coarse 3D seeds and chooses the lowest trimmed 3D residual.
At frame 0, the visible-support hypothesis has coarse rank 14, outside that set.
At frame 195, a near-equal-residual alternative within the three seeds aligns the
handle, while geometric ranking selects the sideways handle. The competing-pose
diagnostic replays the native three-seed result exactly on every sampled frame.

`probe_issue5_pose_candidates.py` refines all 25 PCA/identity seeds for the pot,
retains them all, and ranks accepted hypotheses using projected CAD-sample
coverage of the target mask plus projected exterior distance. The lid fits stay
frozen. Measured nearer depth can explain occlusion; labelled other-object pixels
are excluded only when the CAD is not clearly in front of their measured depth.
The margin is 3 cm; image distances are clipped at 20 pixels. Tests ensure a
visible handle distinguishes two bowl orientations and a farther object cannot
hide an erroneous foreground CAD projection. No pose GT enters this selection.
This is a diagnostic CLI, not a change to native temporal pose selection.

## Matched eight-frame evidence

All rows below use episode-2 development source IDs
`[0,49,98,147,195,244,293,342]`, reviewed SAM2 prompts with lid exclusion, and
frozen SGBM depth, metric CAD and calibration. Every pair is GT-visible and
accepted: 8/8, with no failed sample removed. Predictions are frozen before the
separate public-GT evaluator runs.

| Configuration | Ordered positional RPE kernel | Median relative translation | Median canonical relative rotation |
| --- | --- | --- | --- |
| Reviewed-mask / radial baseline | 29.00 cm | 18.34 cm | 165.95 degrees |
| Density filter; both objects refit | 14.36 cm | 16.57 cm | 122.37 degrees |
| Density baseline + projected-support pot selection; lid frozen | **7.09 cm** | **2.55 cm** | **46.81 degrees** |

The RPE kernel averages ordered first-slot pot-to-lid positional errors. Both
directions and individual frames are recorded separately. Relative transforms
cancel common camera motion; these are not fixed-world absolute errors or full
policy-rollout scores. The third row isolates pot hypothesis selection against
the second row; the first comparison changes both objects' cloud filters.

Earlier exploratory variants are not the published portable result: a four-frame
pot-only density pilot without the threshold floor gave 1.65 cm / 18.71 degrees;
a two-frame projected-support pilot gave 2.43 cm / 15.19 degrees; an eight-frame
variant that unconditionally excluded other-object pixels gave 5.60 cm RPE. The
portable version corrects that exclusion and gives **7.09 cm**, retaining its
worse frame-293 result. Those variants have different configurations/denominators
and must not be reported as current full-episode accuracy.

The 319-frame episode-39 reviewed-box SAM2 control also ran with the new density
filter: 319/319 accepted observations, no input errors, median/max residual
6.24/15.34 mm, and no rotation increments above 90 degrees. The maximum increment
is 14.22 degrees (radial baseline 15.58); P95 is 7.66 degrees (baseline 8.10).
Eight sampled full-mesh overlays give CAD/SAM IoU 0.658–0.904 and were reviewed.
This retains semi-automatic acquisition and lacks canonical/world-pose truth.
An explicit radial rerun after the filter change reproduces all 319 historical
pose JSON files byte-for-byte, including every translation, quaternion and residual.

150 CPU tests pass, zero failures/errors/skips. Actual GPU receipts are separate
from tests using generated inputs. Full predictions, competing candidates,
evaluations, pan regression and source/input bindings are in `evidence/`.

## Reproduce

Use the pinned GPU environment, assets and models from [GPU_CONTROL.md](GPU_CONTROL.md)
and the native/RGB-reviewed mask producer commands from [POT_CONTROL.md](POT_CONTROL.md).
`NATIVE` below is the eight-frame native observation probe directory containing
the frozen `e002/FRAME/depth.npz` files; `VISIBLE` is the reviewed-mask control.
All output directories/files must be fresh.

```bash
python scripts/eval/probe_issue5_cloud_filter.py \
  --data "$DATA" --observations "$VISIBLE/probe_results.json" \
  --depth-root "$NATIVE" --filter local_density --out "$RUN/density"
python scripts/eval/probe_issue5_pose_candidates.py \
  --data "$DATA" --observations "$RUN/density/probe_results.json" \
  --masks "$VISIBLE" --depth "$NATIVE" --out "$RUN/support"
python scripts/eval/render_issue5_pot_control.py \
  --data "$DATA" --observations "$RUN/support/probe_results.json" \
  --masks "$VISIBLE" --out "$RUN/support-overlays"
python scripts/eval/evaluate_issue5_pot_relative.py \
  --data "$DATA" --predictions "$RUN/support/probe_results.json" \
  --kit "$KIT" --out "$RUN/support-relative.json"
```

The diagnostic validates input/calibration/mask/depth hashes and requires the
specified density-baseline configuration. It is currently limited to pot/lid
membership and independent sampled global initialization, with no warm starts
or smoothing. The renderer verifies prediction/input bindings and uses full
mesh faces with a frontmost z-buffer. Its raw silhouette IoU includes external
occlusion and is not an independently labelled visible-surface score.

## Remaining work and 6D-pose interpretation

The current tracker already estimates three position and three orientation
degrees of freedom. This evidence supports **retaining and comparing competing
6D hypotheses using visible asymmetric cues**, not declaring that 6D pose itself
was missing. FoundationPose remains a useful unrun initialization/recovery
comparison; no runtime migration or benefit is claimed.

Canonical rotation still fails. Review lid appearance/symmetry, stacked-pot
partial visibility and the retained frame-293/342 errors. Then validate episode 0,
both pot cameras and continuous pot tracking using independently reviewed target
and occluder masks. The new filter has not been validated across all objects.
Automatic DINO acquisition still selects table/empty/mixed masks. The original
team command and prepared FS-depth/mask artifacts are unknown. Independent shared
metric C(t), all-20 reports, both-camera absolute pot errors and original issue
acceptance remain incomplete. No robot training or official submission ran.
