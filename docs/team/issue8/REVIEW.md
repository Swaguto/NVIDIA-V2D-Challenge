# Dex3 baseline integration review — October 11, 2026

PR35 targets `main` after PR34 merged. The branch combines the merged refinement
fixes and PR36 reproduction tools without changing the frozen e002 observations,
trajectory or the historical 72/199/72 gate receipt. This is a software and
reproducibility review; issue8 remains open and independent team review is pending.

## Findings and repairs

New IK outputs previously recorded only the adapter and solver source hashes.
That did not establish that another machine used the same robot geometry.
`track3_model_lock.py` now hashes both Dex3 URDFs and every referenced visual and
collision mesh. New gate/playback runs refuse missing locks, unretrieved LFS
files, incomplete dependency records and changed model assets. An explicit
`--allow-unlocked-legacy-model` option preserves historical diagnostic access;
receipts identify that exception and do not claim asset verification.

The adapter additionally records mapping/setup/helper source hashes and carries
`source_hand_directly_observed` through IK, independently of validity. Unknown
qpos and FK remain NaN. Display holds are absent from this branch's trajectories.

A fresh native Linux IK run processes all 343 e002 frames, supports 272 left and
284 right observations and locks 24 robot asset files. Independent Linux and
Windows checks reproduce all 10 previous trajectory arrays exactly, including
NaN masks, qpos, FK, errors and iteration counts. The extra direct-observation
mask matches the exported source. Finger position/velocity violations remain 0.
This checks numerical reproduction, not anatomical accuracy or manipulation.

The combined final checkout passed 149 CPU controls. Five new asset-change,
LFS, legacy and portable-path controls passed with the 27 prior bundle/collision
controls. Validation receipts are retained in the goal evidence directory.
Windows Application Control blocked the mypy hook executable; Linux ruff and
mypy checks passed for all 6 changed source/test files. No Windows policy was
changed. Historical simulator renders remain the simulator evidence; no fresh
simulation or training is claimed by this integration patch.

## Separate review: recent-reproducibility-code

Reviewed source: `a70d2efbad12100b341057535f4699058d2b6364`. Its automatic recovery
candidate remains rejected by the documented five-episode accuracy evaluation.
It must not replace the accepted reconstruction method or supply accepted
contact/training observations merely because coverage increased.

The presentation adapter preserves the original retarget NaNs/masks in a
separate `presentation.npz`. It records held rows and their source frame IDs;
leading holds explicitly use a future supported frame. Both-hand presence in
the e012 render is presentation evidence. It does not establish continuous
observed motion, eliminate reacquisition jumps or validate finger placement.

Review found that the verifier checked arrays and hashes but did not enforce
its accuracy/training flags or support-count metadata. A separate repair on
that branch rejects accuracy/contact/training acceptance claims and counts that
contradict the source masks. All14 reproduction/presentation controls pass,
including explicit metadata tampering cases. This repair is kept separate from
PR35; frozen inference settings and the original evidence packs are unchanged.

## Development task: e002 remains under repair

Source annotations identify four grasp intervals, including their ambiguous
onset/release boundaries. A new unsigned original-CAD fingertip-distance audit
retains every interval and all missing samples in the denominators. Supported
hand plus observed-object coverage is 93.12%, 92.65%, 100%, 94.03%. Reconstructed
human raw proximity fractions are 92.59%, 91.18%, 94.87%, 94.03%, respectively.
These calculations use no temporal bridge or display hold.

The first left pot grasp exposes a retargeting defect: the closest robot tip
is a median 12.54 mm from the surface versus 1.75 mm for the reconstructed human.
The loose 2 cm proximity criterion cannot establish an actual grasp. Contact
placement must improve before the task is accepted as physically credible.

A separate development diagnostic reduces hand enclosure radii from 12 mm to 6 mm,
retaining the same robot meshes, trajectory, object material grids and 2 cm
threshold. It gives 271/271 supported two-hand frames below the derived-model
bound, with maximum 18.384 mm; 72 frames remain unsupported. The historical
72-frame receipt is unchanged. The finer enclosure covers 737,636 source
vertices, 246,836 faces and 2,336 cell corners. Independent full-axis verification
is required before promoting this diagnostic. Inferred object closures/material
remain model assumptions, not physical ground truth; between-frame collision
and dynamic grasp stability remain unverified.

Next work is contact-aware retargeting checked against the original video and
object geometry, followed by a new audited simulator replay. Issue9 and policy
training are outside this goal until one development reference passes review.
