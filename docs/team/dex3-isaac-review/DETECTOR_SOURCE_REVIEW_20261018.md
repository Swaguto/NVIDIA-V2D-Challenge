# Scored detector source review — 2026-10-18

The e020/e025 attempts stopped at automatic masks. They did not reach geometry
or policy training. Failed outputs remain private and preserved.

## Source evidence and repair

Pinned Grounding DINO was checked on the calibrated source RGB frames. A CPU
diagnostic recorded detections down to 0.10 for inspection; production detection
confidence stays at 0.30.

* **Watering can:** the correct source box includes the long spout and covers
  16.13% of the image, with confidence 0.852. The detector returns the truncated
  label `white watering`. Add `watering` as an alias for `water_pitcher`. The
  scored runner opts into a 20% detector box area prior for that class only.
  The previous uniform 10% prior wrongly rejected this correct box.
* **Two boards:** the canonical prompts `wooden piece 1. wooden piece 2.` give
  two distinct, correctly located boxes, with confidence 0.418/0.432 and image
  area 3.28%/4.23%. Both have the combined label
  `wooden piece 1 wooden piece 2`. Restore these prompts and retain the joint
  numbered-instance association added in `1368a5d`. The replacement phrase
  `wooden puzzle board` instead selected an oversized fused region.

No episode-specific box or frame override is used. The 20% prior is a deliberate
change to the watering-can detector size rule. Other classes retain the 10% box
limit; the SAM2 mask area limit remains 15%. Depth, mesh, pose and physics gates
are unchanged. The new rule is opt-in for legacy callers and enabled by the
rostered scored runner. Reports record the effective limits; detector logs now
include box coordinates and the applied limit.

Numbered identity inferred from an earliest joint left-to-right view is still
marked uncertified. This is not a general guarantee that printed canonical
identities are recovered on arbitrary hidden tasks.

## Validation status

Source overlays were visually inspected. Focused CPU tests cover truncated
labels, class-specific limits, rejection of fused boxes and recovery of two
distinct combined-label detections. A fresh GPU cohort must still validate
mask propagation, reconstruction and complete unassisted policy rollouts.

Private diagnostics and footage are excluded from this repository.
