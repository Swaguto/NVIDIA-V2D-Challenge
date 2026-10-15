# Between-sample Dex3 clearance findings

The corrected combined candidate now passes the complete modeled continuous
check and has a new audited Isaac replay. See [the October 13 result and exact
reproduction instructions](CORRECTED_DEX3_REPLAY_2026-10-13.md). The original
counterexamples and release below remain preserved; physical-target approval
is still withheld.

This development check follows the frozen October 12 replay. It changes neither
the release tag nor the published videos. The saved replay is reproducible and
its display review is complete; its physical-target gate remains unapproved.

## Why the finite screen was insufficient

The original mesh check found a counterexample inside an interval whose saved
endpoints pass. At 11.3703125 seconds, an interior point of an original right
Dex3 `thumb_2` triangle has a material-depth **lower bound of
2.169149631216985 mm**, exceeding the declared 2 mm model limit. The existing
procedural parent also fails at this time: its lower witness is
2.125480897854542 mm. A new interpolation alone would not repair the parent.

The earlier 3,566 sampled-time screen did not query this time. Its zero-failure
result is retained as a finite measurement, not a continuous clearance claim.
This is a counterexample against the declared voxel material, which has a
3.464 mm uncertainty band. It does not establish penetration of the true
physical lid or the accuracy of the recovered human hands.

The initial continuous refusal was unresolved near the limit. Finer spatial
and temporal subdivisions kept the 2 mm limit and 50,000-query spatial cap
unchanged. An analytic safe-corner control becomes certifiable with finer
bounds, while an analytic unsafe triangle remains rejected. Subsequent actual
time samples expose a material-depth witness, rather than treating an
unresolved numerical proof as a collision.

## Reproduce the counterexample from the repository

The repository includes the original declared lid-material payload, a numerical
triangle/witness fixture and a byte-preserving boundary-distance method. No video,
GPU, Isaac installation, public contact annotations or ground-truth poses are
needed for this focused check. NumPy and SciPy are required; use the project's
CPU environment and record its versions.

The command was run on Windows (Python 3.14.0, NumPy 2.5.3, SciPy 1.18.1) and
Linux (Python 3.11.17, NumPy 2.2.6, SciPy 1.15.3). Both return the same depth
lower bound and material hash. This checks the numerical witness on different
environments; it does not reproduce the full new repair or upstream inference.

```bash
python scripts/reproduction/verify_dex3_clearance_counterexample.py
```

Require `declared_material_limit_exceeded: true`, the material SHA256
`d4f9296232622a31deee1b179997f7692e4597890d7072c529c75b93e26da5ad`, and
`depth_lower_m` equal to `0.002169149631216985` within `1e-12` metres. The script
verifies the material/method bytes, recomputes the point's barycentric position
inside the original transformed triangle, and uses the minimum singular value
of the source object affine matrix to produce a conservative world-depth bound.
Its exit status verifies the saved counterexample, not trajectory acceptance.

The fixture also retains an automatically extracted material subset. Its depth
is conservative for the full declared union and independently agrees here, but
the command above uses the complete original material payload by default.

The fixture is numerical evidence derived from the locked motion and original
mesh; it is not a reconstruction/refitting run. Its source motion, diagnostic,
mesh and full-material hashes identify that provenance. The original point was
also independently rematerialized from the pinned Dex3 geometry and FK, with a
point-to-triangle residual of `1.39e-17` metres.

## Bounded repair candidate

An automatic window around the first refused source-time bracket is clamped to
exclude inferred grasps. The existing translation solver operates on complete
original hand surfaces against both objects; its limit remains 2 mm and its
additional root-correction norm is capped at 10 mm. It reads no contact labels
or ground truth. The first fit is rejected and preserved. A subsequent fit
reuses 36 rejected original-triangle witnesses as constraints; rejected fits
are never used as acceptance evidence.

The revised fit passes **486 static surface checks at 243 sampled times**. Its
maximum root correction is **2.363669 mm** over an automatically selected
0.45-second noncontact window. The explicit candidate has 3,578 knots and changes
eight inferred free-motion hand states at source times. Original human/object
observation arrays, all grasp anchors, orientations and finger arrays remain
unchanged. `robot_noncontact_surface_repair_inferred` marks these generated
corrections; it never marks a new observed hand.

The raw original knots are preserved outside the repair window. Re-evaluating
closed-hold knots can cause subnanometre quaternion-rounding differences, so
the prefix guard refused proof reuse until the original raw knots were retained
exactly. The corrected proposal's payload SHA256 is
`86fde5562e6c7b0353010c902115d720d98d1ec8565003a128f519219f611a19`.

Only the exactly unchanged prefix reuses earlier complete interval proofs.
The entire modified window and all later motion require new continuous checks.
Static fit passes do not certify interpolation, self/cross-hand motion, support
contact, source accuracy or manipulation dynamics. This candidate has not yet
received a new Isaac replay and is not the frozen release.

## Remaining physical gate

Complete continuous checking of the repaired candidate, then run an audited
Isaac replay. The six source/contact-boundary gaps and hand-mask ownership,
limited human distal-thumb support, support-plane discrepancy and recovered
placement uncertainty remain separate unresolved accuracy questions. Human PR
review is still pending. Do not advance Issue #9 or train from this reference
on the basis of visibility or finite surface passes alone.
