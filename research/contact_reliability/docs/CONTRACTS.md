# Data and integration contracts

## Reconstruction diagnostics

`v2d-research extract` consumes a diagnostic NPZ with no pickle payloads and a JSON
metadata file. This is an explicit adapter boundary: official reconstruction
artifacts are preserved, and their producer must export these aligned diagnostics.
The release does not expose one universally compatible diagnostic file; this
implementation does not fabricate missing visibility or reprojection measurements.

Axes are time T, hand H, object body B, and keypoint K. `object_ids` means exact
motion `object_body_names`, including articulated parts, not informal object labels.

| NPZ field | Shape | Meaning |
|---|---|---|
| visibility_mask | T,H,B,K boolean | Valid observed hand keypoints |
| observed_masks, rendered_masks | T,H,B,height,width boolean | Aligned object silhouettes |
| observed_keypoints, projected_keypoints | T,H,B,K,2 | Pixel coordinates in the same image |
| relative_positions | T,H,B,3 | Hand minus object position in meters, shared world axes |
| timestamps | T | Strictly increasing seconds on the motion frame grid |
| image_size | 2 | Width and height in pixels |

JSON metadata supplies `sequence_id`, `timestamps`, `hand_ids` (left/right), and
`object_ids`. Missing geometric measurements may be NaN in NPZ. Extraction exports
zero placeholders plus a false validity mask. Empty mask unions are invalid, not
perfect overlap. The first and last acceleration samples are invalid. No frame
interpolation is performed. Resampling from reconstruction to the retargeted motion
grid is a producer responsibility and must be explicit and recorded.

Observation JSON `observations_v1` adds `signal_names`, `signals[T,H,B,4]`, and
boolean `valid[T,H,B,4]`. Signal order is visibility, mask IoU, reprojection error
normalized by image diagonal, and magnitude of temporal acceleration in m/s².

## Normalization and weights

`normalizer_v1` records development sequence IDs and input hashes. Fitting rejects
sequences outside the supplied development split. Visibility and IoU are already
bounded quality measures. Error measures become `exp(-error / development_p95)`;
the scale has a numerical floor of 1e-8. A signal absent from calibration is ignored.

Equal-weight the available quality signals. Weight is `floor + (1-floor)*quality`,
where floor is restricted to 0, 0.25, or 0.5. A cell without any calibrated valid
signal gets baseline weight 1 and a warning. These are reliability heuristics, not
calibrated confidence probabilities or measured physical forces.

`reliability_v1` retains raw signal values and validity, adds effective validity,
weights[T,H,B], the normalizer hash, input hash, configuration, and cache identity.
The identity includes input contents, settings, calibration, and implementation
version. Bump the implementation version when changing score semantics.

Constant ablation uses the mean of proposed weights within the same sequence.
Shuffled ablation applies one seeded time permutation to all hands/bodies together.
These ablations intentionally change the location of baseline fallback weights.
Do not tune normalizer parameters on the local holdout or hidden challenge truth.

## Robot training integration

The motion parquet stays `motion_v1`: meters, world-space poses, wxyz quaternion
order, explicit hand/body names. The optional adapter checks sequence ID, timestamps,
dimensions and exact hand/body order, then indexes weights using each environment's
current reference frame. Random resets therefore select the corresponding confidence
frame. There is no clamping, silent reordering, or nearest-frame matching.

Wrench support, missed-contact, and unintended-contact terms multiply each body
contribution before original body/hand denominators. Denominators never sum weights,
which would cancel attenuation. Weights also affect reference *absence* penalties.

Upstream contact Chamfer combines bodies into a point cloud, and force closure takes
maximum support over bodies. For these two objectives, use minimum reliability over
bodies per hand and multiply that hand's contribution before hand aggregation. This
conservative rule preserves the original objective at all-one confidence. It may
over-attenuate an otherwise good hand when one object body is unreliable; report that
limitation. Do not reinterpret the simulator-only force regularizer as a reference
force target; leave it unchanged.

Configuration replaces supported terms even when initial coefficients are zero,
because the upstream curriculum can activate them later. Object tracking, joint
tracking, action limits, and unrelated rewards remain unchanged. No arbitrary-robot
or floating-hand compatibility is claimed.

## Evaluation

The official evaluator contract remains unknown. Set the path, SHA-256, exact argv,
output directory, and required artifact names in `configs/submission.json` only from
official instructions. Every expected output must be new and nonempty. Paths escaping
the declared output directory are rejected. A zero exit code without all artifacts
is failure. Successful local verification never implies an accepted submission.
