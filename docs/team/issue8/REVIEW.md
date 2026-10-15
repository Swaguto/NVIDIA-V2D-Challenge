# Dex3 baseline integration review — October 11, 2026

PR35 targets `main` after PR34 merged. The branch combines the merged refinement
fixes and PR36 reproduction tools without changing the frozen e002 observations
or historical trajectory/gate archives. The initialization fix below creates a
distinct current trajectory. This is a software and
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

A subsequent initialization review found a real solver defect: assigning
`Configuration.q` alone leaves Pink's cached FK and Jacobians at the preceding
pose. Cold starts and observation-gap resets therefore supplied inconsistent
kinematics to the first QP. The pinned-runtime probe shows roughly 0.45 m stale
root-position error on each first supported e002 hand. `HandKinematics.compute`
now calls `configuration.update(initial)` before solving. Two native regression
controls compare every FK frame and Jacobian with independent recomputation on
cold/custom/default resets; both fail before the fix and pass after it.

The synchronized full e002 run preserves all source geometry, masks, limits and
272/284 supported hands. Relative to the preceding LF reference, maximum wrist
change is 0.035 mm left / 0.107 mm right, and maximum finger-angle change is
0.00467 / 0.01221 rad. Nine left and one right iteration counts change. This
revision intentionally does not claim equality with the old trajectory; its
new payload SHA256 is
`ca011c10673425ad25cc5badc445fe953905bbb84585ea5a3cbcb2d6fe9def09`.
An isolated synchronization-only diagnostic reproduces all its motion arrays.
Joint/velocity violations remain zero, and the corrected vertex audit still
has 10.88 / 7.02 mm worst lower witnesses. The initialization fix does not
repair the grasp. Earlier byte-exact results remain bound to their archives.
The synchronized trajectory was also actually rendered in the pinned A40
source-camera simulator. All 343 frames and 556 supported states pass the
independent decoded-video/state audit; maximum link error is 1.26e-7 m.
Its source/recovered-hand/Sim video retains the UNACCEPTED label and original
missing observations. This is a new result, not a relabelled earlier render.

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

Further isolated development trials confirm that repulsion alone is inadequate.
Strict local material inequalities fail feasibility. Adding explicit slack
reduces penetrating timestamps but allows a 541 mm right-wrist source residual;
it is rejected. Adding source-wrist guards and initialization synchronization
prevents that escape, but still witnesses 9.69 / 5.38 mm intrusion (164 left /
36 right timestamps above 2 mm), with left three-tip residuals reaching 149 mm.
It is also rejected. None of these collision objectives is in the baseline code.

Source accuracy is now checked separately: all recovered-human mesh vertices
are queried against the same declared material (865,136 queries). Source lower
witnesses reach 9.55 mm left at frame161 and 6.66 mm right at frame93; 203 / 113
supported timestamps exceed 2 mm. This shows a source-hand/object-model
inconsistency already exists before retargeting. It does not distinguish
inaccurate reconstruction from incorrect inferred CAD material or provide
physical ground truth. Source-camera sheets90-95 and156-161 were inspected;
hidden finger surfaces are not directly observable. Source geometry/contact
alignment and material uncertainty must be resolved before another robot cost
change can establish a physically credible reference. The original observations
and every unknown grasp onset remain unchanged.

The source-view diagnostic is now reviewable code:
`robotic_grounding/scripts/retarget/track3_review_source.py`. It verifies locked
RGB, metric stereo depth, validity and mask bytes, rasterizes original object
and MANO triangles, and reports visible-surface ownership and depth residuals.
It accepts relocated inputs using explicit root mappings and reviews all frames
by default, with no episode-specific crop/frame choices or display holds.
Missing object masks remain unknown. It never sets hand accuracy, contact-target
or physical-task acceptance. See [SOURCE_ACCURACY.md](SOURCE_ACCURACY.md).

The immutable staged e002 run reproduces every per-frame measurement from the
private diagnostic exactly across all 343 frames. Median visible-hand mask
precision is 96.07% / 97.22%, while median absolute stereo residual is 8.03 /
6.54 mm. The two worst human-source collision witnesses lie 8.67 / 28.77 mm
behind the front reconstructed surface. Their original CAD nearest faces are
not added closures. Those observations do not resolve hidden contacts or prove
the inferred material is correct. Independent triangle-ray checks match 512
actual mesh queries within 5e-16 m; six new occlusion/perspective/path/identity controls
pass on both Windows and Linux. The current scoped suite passes 140 Linux
refinement/bundle/source-review controls and 15 archival controls; 38
bundle/collision/model/source-review controls pass on Windows. Four actual
source-frame measurements match exactly between Windows and native Linux.

Reconstructed occlusion is not proof of physical source occlusion. At the worst
left source point, the source hand mask and RGB show a finger, but the front
reconstructed surface belongs to the pot. This identifies a visible depth-ordering
error that a future source fit must address. It does not identify the exact
placement of an occluded finger vertex. Source/view depth ordering is a stronger
next constraint than another blind robot-only repulsion weight.

Two complete source-MANO contact repair trials retain source masks, cameras,
objects, hand shape and scale, and use no grasp annotations in optimization.
The tighter bounds reduce human-source timestamps over 2 mm from 203/113 to
135/48, but leave 9.01/6.04 mm lower witnesses. The more permissive bounds leave
8.89/5.99 mm with 120/36 timestamps, and move projected vertices by up to
34.77/22.21 pixels. Both are rejected. The tighter trial's full-frame mask/depth
check does not establish better source accuracy. No MANO repair objective or
new targets enter PR35. These trials are not untouched-episode validation.
