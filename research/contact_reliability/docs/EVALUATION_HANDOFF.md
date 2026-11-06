# Evaluation-owner handoff (draft, not sent)

Review target: PR #17, team main b6dd0d9f9ddabf7f1955119d797d9ccfe8107104.
Owners recorded on issue #2: CreativeLapse and ForceDrift.
This document proposes tests and records source findings; it does not modify the
team scorer or assert official evaluator equivalence.

| Finding in scripts/eval | Proposed regression / resolution |
|---|---|
| score_candidate passes GT zero/invisible poses without a visibility mask to metrics | Change invisible values while keeping mask false; scores must be unchanged. Define all-invalid episode behavior explicitly. |
| Candidate loading does not align by frame_index/capture_time | Reject a shifted, duplicated, missing, reordered, or extra frame. Do not silently truncate. |
| rigid_align transforms positions only | Apply a known 90-degree rigid rotation and translation; both orientations and positions must recover. Fit only valid correspondences. |
| NPY carries no object/frame identity | Require an accompanying manifest with object order, exact time map, frame, units, and asset hashes. |
| Completion object is synthetic | Report actual termination and sequence coverage separately. Never infer completion from pose metrics or the wrapper's completion object. |
| Local output omits official RPE | Trace the official definition; do not rename obj_pos_err_m into RPE without proof. |

Our local `track3.validate_pair` rejects mismatches and preserves GT visibility.
`transform_poses` rotates both position and orientation. These are boundary helpers,
not a replacement scorer. The synthetic oracle test verifies input pairing only;
it does not prove official AUC/SP-SR/MP-SR/MPPE values.

Primary reported scores should use the declared episode coordinate frame.
Ground-truth-fitted alignment belongs in a separately labeled diagnostic column.
Do not describe corresponding-point ADD as symmetry-invariant ADD-S.

Calibration issue #3 / draft PR #18 also needs a real-data test: fixed stereo
extrinsics do not establish a constant world-to-camera pose. Account for egomotion
and the independently recentered episode origin; verify on held-out frames and
episodes. Synthetic PnP recovery cannot establish this transfer.

Question still pending with the group: do we have NVIDIA's official eval_e2e.py,
or only scripts/eval/score_candidate.py? If official, record location, checksum,
version, command, and required submission outputs before invoking it.
