# Original CAD contact checks for the e002 development replay

The [published corrected replay](CORRECTED_DEX3_REPLAY_2026-10-13.md) is
reproducible and visually stable. Its 2 mm voxel-material certificate does not
establish that the intended contact faces reach the original object surfaces.
This follow-up measures that distinction and preserves the unsuccessful results.
The existing release and its video remain unchanged.

## Measured results on the published replay

| Check | Result | Limit |
|---|---|---|
| Original hand/object triangle intersections | Zero across all 343 source frames and 10,976 surface pairs | Finite source states; containment and motion between states are not certified |
| Selected original thumb/index face gaps | All 878 positive-distance witnesses independently verified | Only the declared faces are tested; other parts of a pad may be closer |
| Two declared faces within 2 mm during inferred closed grasps | 0/377 hand states | This is a contact-distance test, distinct from the previous voxel proximity score |
| Contact review labels | 0/381 labeled core states meet that two-face CAD-distance criterion | Labels are source-video review, not measured pose ground truth; they never enter fitting |
| Original CAD topology | Both objects remain nonwatertight after exact vertex welding | No trustworthy signed solid depth is assigned to these meshes |
| Estimated support plane | Original source states and 3,589 motion knots checked | The estimated plane and convex footprint are not a measured table boundary |

The declared face gaps are approximately constant within each rigid hold:

| Grasp | Thumb gap | Index gap |
|---|---:|---:|
| Left first pot grasp | 2.725 mm | 3.035 mm |
| Left second pot grasp | 3.364 mm | 3.369 mm |
| Right first/second lid grasp | 3.859 mm | 3.301 mm |

The support diagnostic finds minima below the *infinite estimated plane* of
190.3 mm on the left and 87.0 mm on the right. Source-image inspection places
these witnesses outside the visible table edge or image. A convex footprint
fills unobserved gaps and includes some of that exterior region. These numbers
must not be presented as verified table penetration or corrected by shifting
the observed objects. The remaining 8–10 mm object/support discrepancy and
source depth residuals remain qualified in the published replay document.

## Reproduce the audits

Use the same pinned Linux scientific runtime and verified source case/fitted
inputs from the corrected replay instructions. Ordinary Windows NumPy/SciPy
is not the geometry runtime. The archive below includes raw measured reports,
the source-only annotation, the plane hypothesis and all three audit entrypoints.
It excludes reconstructed robot inputs already supplied by the existing release.

Restore the raw bytes with the existing safe, hash-checked archive implementation:

```bash
python scripts/reproduction/restore_original_CAD_audits.py \
  --archive docs/team/reproduction/dex3-motion-replay/original-CAD-baseline-audits-v1.tar.gz \
  --manifest docs/team/reproduction/dex3-motion-replay/original-CAD-baseline-audits-manifest-v1.json \
  --out /var/tmp/original-CAD-audits
```

Run the checks with the pinned scientific Python. `CASE`, `METHODS` and `INPUTS`
below denote the already verified corrected case, the 51-file continuous method
snapshot and fitted inputs, respectively. `AUDITS` denotes the restored folder.
Use fresh output directories for each command.

```bash
python "$AUDITS/methods/audit_frozen_dex3_original_mesh_intersections.py" \
  --case "$CASE" --methods "$METHODS" \
  --case-manifest-sha256 87b027f05ca58259fa355a635ce6c1fed90a90a64ffcc84b9a2bed46d50880c0 \
  --out /var/tmp/new-original-CAD-source-check

python "$AUDITS/methods/audit_frozen_dex3_original_contacts.py" \
  --case "$CASE" --methods "$METHODS" --fitted-inputs "$INPUTS" \
  --annotation "$AUDITS/evidence/e002-source-grasp-annotation.json" \
  --case-manifest-sha256 87b027f05ca58259fa355a635ce6c1fed90a90a64ffcc84b9a2bed46d50880c0 \
  --out /var/tmp/new-original-CAD-pad-check

python "$AUDITS/methods/audit_frozen_dex3_support_plane.py" \
  --case "$CASE" --plane "$AUDITS/evidence/source-support-plane-v1" \
  --case-manifest-sha256 87b027f05ca58259fa355a635ce6c1fed90a90a64ffcc84b9a2bed46d50880c0 \
  --plane-report-sha256 38a6bf97df7def496bed7773e265c894e77da6bb5521d3077bf43330cbd3166f \
  --out /var/tmp/new-original-support-check
```

The case manifest binds every source file. Original object affine transforms,
scale, robot meshes and observation flags are preserved. Every reported positive
pad-gap witness is projected independently onto the original triangles using
the frozen small-triangle projector. A containment control demonstrates why
zero triangle intersections cannot certify signed solid clearance.

Compare numerical fields, input hashes, counts and pass/refusal results.
Elapsed time and output locations can differ. Restoring the archive verifies
existing evidence bytes; it does not rerun a geometry check or Isaac Sim.

## Development corrections remain candidates

The follow-up searches for nearby original-CAD contact faces automatically,
starting from the existing source-supported grasp templates. It reads no
episode frame selections, crops, contact annotations or ground truth. It
requires opposing selected object-face normals, opposing robot-face normals,
0.3–2 mm selected triangle gaps, URDF finger limits, self separation, and the
unchanged complete original-hand-surface 2 mm voxel-material test.

A four-grasp static fit passed, but blending it into the full replay introduced
18 original-mesh intersection pairs at source states. The candidate was rejected
and its continuous checker stopped after that independent counterexample. A
second fit includes verified contact witnesses against the other object; the
resulting source check removes that closed-grasp crossing but still finds 16
transition intersection pairs. Neither candidate is a new accepted replay.

The current transition search derives its windows from the grasp schedule and
checks hardware speeds and original CAD at the stored knots. A successful
finite search must still pass a fresh continuous scene check, an actual Isaac
replay, independent simulator readback and chronological frame review before
publication as a corrected result. Static success or a solver convergence flag
alone does not pass those gates.

Human observations stay observed/supported/missing exactly as before. Changed
robot poses are explicitly inferred. Display holds never become measurements,
and all source-accuracy and motion-target approval flags remain false until the
corresponding evidence exists. This development audit does not establish private
episode performance or trained physical grasp success.
