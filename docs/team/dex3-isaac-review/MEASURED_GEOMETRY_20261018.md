# Measured geometry and reference contacts — 18 October 2026

The physical tasks still fail. Longer training is not qualified as the next fix
for the current e003 reference. Keep the recorded controls and initial policies.

## Completed comparisons

- e005, e006 and e008 completed automatic reconstruction, retargeting, 100 PPO
  iterations / 153,600 environment steps, initial/final official evaluation and
  complete eight-world unassisted native readbacks. All MP-SR values are zero.
  e005/e006 training episodes terminate after one step. e008 averages 6.93 steps.
  This brings the inspected short baseline to seven scored episodes and **13 of
  36 episode/object mesh slots**, not a complete submission.
- e003 hand motion is restored: 310 left and 309 right command anchors. The
  older policy with both hands parked is disqualified.
- A private source-only pot fit improves seed silhouette IoU from 0.497 to 0.744
  and median measured-point/vertex distance from 53.2 to 10.7 mm. It changes
  uniform scale and translation, preserving the raw body origin and rotation.
  A pilot-specific constraint places the seed 2 mm above the independently
  inferred, unchanged table. This is not a general off-table classifier.
- The unchanged FoundationPose initialization undid that calibrated seed and
  caused 19.5 mm initial native pot/support penetration. The calibrated-seed fix
  preserves the exact seed in all three GPU tracking hypotheses, while retaining
  the original measured-depth and temporal jump gates. The subsequent native
  check finds no initial pot/support contact. The unchanged old spoon still
  penetrates the support by 22.9 mm.
- That repaired-seed e003 trial still fails: native readback AUC 0.08313, MP-SR
  zero, mean object position error 0.16785 m. Training episodes average 1.90
  steps. AUC is measured against our changed inferred reference, not hidden
  ground truth; it cannot establish a leaderboard improvement.

Both geometry trials were fully backed up and SHA256 verified before the VM
was stopped: 14,536 retained files for the control and 14,550 for the seed fix.

## Full-reference native collision diagnosis

The new CPU audit compiles the actual native finite support and convex object
shapes, resets each rigid object to each reference knot, verifies its world pose,
and calls collision detection without advancing physics. It excludes the robot.
The production audit completed all 313 e003 knots in 7.85 seconds:

| Native body pair | Knots with penetration over 1 mm | Maximum | Worst reference knot |
| --- | ---: | ---: | ---: |
| Pot / support | 109 | 19.78 mm | 6 |
| Spoon / support | 87 | 23.81 mm | 49 |
| Pot / spoon | 19 | 19.70 mm | 122 |

This establishes a reference/collision inconsistency after the corrected first
frame. Whole-body minimum heights alone are insufficient because handles may
extend beyond the table. These figures use native finite collision shapes.
They are static reference-knot checks, not swept contacts or measured grip force.

Frozen input manifest SHA256:
`2d3c32b6c67d8539a352bb7cd09c29cea7b6aae0920d8b464a178b010d6d6d3d`.
Runtime: `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`.

Future policy preparation records both `initial-contacts.json` and
`reference-contacts.json` before PPO. The audit reports failures; it does not
alter targets, geometry, reward weights, evaluator gates or native mesh checks.

## Rejected shortcut and next experiments

Simply projecting early pot poses upward reduces silhouette IoU and worsens
their measured depth fit. It passes the tracker's coarse depth gate but has
worse source evidence, so it is **not promoted**. The experimental
`measured_pose_refinement.py` translation helper is also not integrated into the
default pipeline: the full-body footprint condition does not apply to the
sampled pot, and translation alone does not fix its later intersections. Its
tests cover missing depth, raw-body preservation, finite support and rejection
of a table correction that contradicts measured pixels.

A separate first-frame spoon diagnostic has better source support than the old
later-seeded shape: IoU 0.523→0.604, measured distance 14.4→7.49 mm, and 2 mm
seed clearance. The next e003 trial tests both corrected objects. The manual
first-frame diagnostic is not a frozen hidden-episode selection rule.

The source-only selector audit subsequently found why that clean view was
excluded: e003 frame 0 has 9,247 foreground pixels, below the old 10,440-pixel
cutoff, while the selected closer frame 136 has 20,600 pixels and **58.5% direct
hand overlap**. Their measured projected metric areas are 0.00962 and 0.00920
square metres respectively. Camera distance, rather than visible completeness,
was driving the area rejection. The generic runner now supplies its calibrated
observations to the selector, ranks area using median measured depth squared,
and checks direct hand overlap. It keeps measured-depth sufficiency checks,
source hashes, unknown other-object exclusions and automatic prompts. This
removes the need to specify e003 frame 0 as an episode override. New-object GPU
integration and physical validation remain necessary.

An e008 transfer trial changes only the pot; its required beige cup is retained
unchanged. **e008 has two objects.** Both slots must be present in every bundle
and submission. Neither experiment is accepted as a physical task yet.

The remaining work includes general reconstruction/pose repair, full task
validation, the remaining roster, frozen reserved-episode testing, the untouched
official packer and fresh team reproduction. Inputs, derived videos, MANO and
gated model weights remain private. No leaderboard artifact has been submitted.

## Coverage failures at the detector boundary

The next cohort completed e019 (sweeper/dustpan), while e020 and e025 stopped
before geometry inference. e020 had 202 blue-cup masks and zero pitcher masks;
its actual source object is a watering can. e025 had zero first-piece masks and
356 second-piece masks: the detector returned the combined text
`wooden piece 1 wooden piece 2`, and the old longest-match tie silently chose
the last slot. Neither failure is evidence about learned grasp quality.

The generic prompt map now covers all canonical classes, including watering-can
aliases. Ambiguous detector labels remain ambiguous. Numbered same-category
instances require distinct boxes in a joint source view; a fused box cannot
populate two slots. Specific detected identities take precedence, otherwise the
earliest joint view's left-to-right association is explicitly **inferred**.
Printed canonical identity is not certified by that convention. Source inspection
of e025 shows the marked first board on the left and second board on the right,
but this does not establish a general hidden-episode identity guarantee.
The driver now reports an absent object mask at the detector boundary. A fresh
bounded retry must validate these changes through actual segmentation/geometry.

An additional source-only camera-compensated stillness experiment passes its
synthetic tests, but supports only 11/313 pot and 26/313 spoon frames in the
inspected clip. It is not connected to the tracker or promoted as a repair.
