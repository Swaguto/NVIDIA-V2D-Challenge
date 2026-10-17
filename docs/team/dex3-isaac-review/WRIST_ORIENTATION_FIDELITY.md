# Wrist orientation fidelity: e002

The source-to-grasp reference repair is now the first priority before another long training run. The existing fitted wrist reference is substantially different from the source-aligned retargeting.

## Measured decomposition

The audit compares the original calibrated Dex3 retargeting, the frozen grasp commands used for training, and the actual measured palm orientations of the selected CWS2000 policy. It uses matching source frames and a common stage coordinate frame. It does not compare raw MANO and robot quaternion axes directly. Unknown source frames are excluded.

During the first pot carry, source time 2.4–11.5 seconds:

| Wrist | Fitted target versus source-aligned Dex3 baseline, median | Actual wrist versus fitted target, median range across four worlds |
|---|---:|---:|
| Left | 109.84° | 2.98–3.25° |
| Right | 25.20° | 3.54–3.79° |

The original calibrated retargeting itself has a median reported source-orientation residual of 7.84° left / 3.73° right across observed frames. It is an inferred baseline, not ground truth. The stage rotation is bound to the same source-bundle hash and independently checked against both object references: their maximum orientation differences are 1.84° / 1.78°, within the existing 2° smoothing bound.

The current training target generator obtains wrist poses from `TaskGripReference.command`. During a grasp this composes the object pose with a fitted `hand_in_object_xyzw` transform. It does not preserve the demonstrated time-varying wrist orientation. The measured wrist follows that altered target fairly closely. This identifies a reference-fidelity problem; it does not establish that every visible difference has a single cause.

## Required repair

1. Verify the calibrated human-to-Dex3 palm frame against the source-camera view for both hands.
2. Use the source-aligned wrist orientation as the trajectory prior, preserving observed roll/pitch/yaw changes through grasp phases. Keep unknown intervals explicitly marked.
3. Refit finger contacts and wrist translation around that prior, with an explicit bounded angular deviation. If the desired orientation is infeasible under robot limits or collision geometry, report the conflict rather than silently substituting a very different grasp.
4. Compare source-aligned baseline, repaired target and actual simulated wrist separately, using the source camera as well as numerical rotation errors.
5. Run the short hold/lift/carry tests and the complete unassisted physical replay before retraining. Orientation fidelity and stable contact both remain required.

The audit is complete. A repaired trajectory has not yet been generated or physically validated. The large fitted rotation should not be treated as an unavoidable consequence of different human/robot hand anatomy.

[Hash-bound orientation measurements](wrist-orientation-audit-v1.json) · [current grip diagnosis](CURRENT_GRIP_DIAGNOSIS.md).
