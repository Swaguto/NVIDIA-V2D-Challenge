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
Base qpos uses XYZW quaternions; per-link FK poses use WXYZ, as returned by
`HandKinematics.compute`. Consumers must not assume these two arrays use the
same rotation order. Native Pinocchio verifies all 12,788 supported FK poses:
zero position error and maximum rotation difference 2.01e-16 radians.

Final checkout review found that Windows CRLF conversion changed the hashed
Python and URDF bytes relative to Linux Git checkouts. `.gitattributes` now pins
LF for the retarget sources and Dex3 URDFs. All 43 affected working files match
their existing Git blobs; no Python statements or parsed robot geometry change.
A new native LF run reproduces all 11 locked trajectory arrays exactly, and the
32 Windows model/bundle/collision controls pass again. Older archives retain
their original bytes and locks; they are not relabelled as this new revision.

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
contradict the source masks. All 14 reproduction/presentation controls pass,
including explicit metadata tampering cases. This repair is kept separate from
PR35; frozen inference settings and the original evidence packs are unchanged.

## Development task: e002 remains under repair

Source annotations identify four grasp intervals, including their ambiguous
onset/release boundaries. A new unsigned original-CAD fingertip-distance audit
retains every interval and all missing samples in the denominators. Supported
hand plus observed-object coverage is 93.12%, 92.65%, 100%, 94.03%. Reconstructed
human raw proximity fractions are 92.59%, 91.18%, 94.87%, 94.03%, respectively.
These calculations use no temporal bridge or display hold.

The first left pot grasp has a closest robot tip-site median distance of
12.54 mm from the surface versus 1.75 mm for the reconstructed human. Tip sites
are not complete contact pads: a proximal segment can contact the handle while
its tip does not. These distances alone cannot establish absence of grasp.
The loose 2 cm proximity criterion cannot establish a credible grasp either;
pad placement, opposition and penetration require separate checks.

A separate development diagnostic reduces hand enclosure radii from 12 mm to 6 mm,
retaining the same robot meshes, trajectory, object material grids and 2 cm
threshold. It gives 271/271 supported two-hand frames below the derived-model
bound, with maximum 18.384 mm; 72 frames remain unsupported. The historical
72-frame receipt is unchanged. The finer enclosure covers 737,636 source
vertices, 246,836 faces and 2,336 cell corners. Independent verification checked
162,304 full-axis bounds and all material depth witnesses; maximum independent
upper-bound discrepancy is 5.56e-17 m. Inferred object closures/material
remain model assumptions, not physical ground truth; between-frame collision
and dynamic grasp stability remain unverified.

A development-only position-cost experiment selects source thumb/index/middle
tips within 5 mm of observed CAD and increases their position cost from 1 to 3.
It retains source geometry, missing masks and joint/velocity constraints. First
left-grasp closest-tip median improves to 5.32 mm, but the thumb stays about
30 mm from CAD and two-tip proximity is rare. Right-hand regrasp two-tip
proximity below 5 mm decreases from 60.26% to 55.13%. This candidate is not
promoted: nearest-tip improvement alone does not prove opposing contacts or
safe geometry. Contact-pad surfaces, source-video alignment and penetration
still require review. Historical frozen receipts remain unchanged.

A corrected full collision-mesh vertex audit queries 68,630,152 vertex/object
pairs at supported timestamps. Under the documented inferred material,
baseline depth lower witnesses reach 10.88 mm left / 7.02 mm right, at frames
54 / 303. There are 238 left and 133 right supported timestamps above 2 mm.
Version 1 of this additional development audit incorrectly interpreted FK
WXYZ rotations as XYZW; its numbers and worst-frame claims are invalid. Version
2 corrects that conversion and retains the original reports without relabelling
them. The pipeline's capsule/material gate already used WXYZ correctly and its
historical and independent enclosure receipts are unchanged. These vertex
witnesses concern the declared derived material, not physical ground truth or
a complete surface certificate. The 2 cm bound can pass while defects remain.

A second development candidate adds sampled material repulsion to the existing
QP. Finite-difference point Jacobians agree within 8.1e-9. It retains every
source support mask and satisfies joint/velocity limits, but the corrected
vertex audit still has 6.95 mm left / 6.55 mm right lower witnesses. Left median wrist residual rises from
10.48 to 22.11 mm and orientation from 7.84 to 18.20 degrees. It is rejected;
moving the hand away or improving one proximity count is not task acceptance.

A separate actual Sim 6.0.0.1 / Lab 3.0.0 beta2 A40 source-camera diagnostic
renders all 343 frames. Independent state/decoded-video checks pass: maximum
link error is 9.11e-8 m. Moving camera USD readback matches the reconstructed
camera within 1.03e-9 m / 4.04e-7 rad; output intrinsics match exactly. This
pinned renderer ignores aperture offsets, so a declared automatic overscan and
bilinear principal-point translation supplies the source view. The first failed
camera-audit attempt is retained separately. Source RGB, recovered-observation
wireframes and actual Sim form a complete comparison video; all 343 source RGB
hashes are verified. Spot review includes frames 55 and 184. Whole-video pose
accuracy acceptance is pending. No display holds, collision resolution, dynamics
or training are used. This diagnostic's archived source/model bytes predate the
LF checkout fix; numerical IK arrays are identical, but its receipts remain
bound to that original private source package. A second actual Sim run uses
the canonical LF source/assets. Its independent 343-frame state and video audit
also passes. Every raw PNG pixel array and shared simulator readback array is
identical across the two runs on this same A40 and pinned runtime. This is not
independent-machine rendering or physical manipulation validation.

Next work is contact-aware retargeting checked against the original video and
object geometry, followed by a new audited simulator replay. Issue9 and policy
training are outside this goal until one development reference passes review.
