# e037: recover the missing source hand before more PPO

Status: **incomplete; no new accepted grasp or policy.**

The right hand's original direct/metric observation gap is frames 25–99. The
right-hand tracking mask is empty through that interval in the recovered cache.
`masks_to_hands.py` skips empty masks, so that path cannot generate fresh HaMeR
priors for the carry. The robot reference later fills an interior gap with world
interpolation, leaving its fingers over 20 cm from the lifted spoon.

## Changes and CPU evidence

- Cached detector handedness labels flip during the carry. Identity is associated
  with original source wrist anchors and checked image motion; the label string
  alone cannot identify the wearer hand.
- Per-joint stereo screening retains 61 pairs with some supported points, but
  **zero complete supported palm tuples**. The prior full-hand stereo rejection
  remains unchanged.
- One-sided image flow uses the existing 1.5 px forward/backward and 1.5 second
  gap limits. It supplies **inferred image evidence at 72/75 gap frames**; 37–39
  remain unknown. This is neither 72 metric hand poses nor contact ground truth.
  A more restrictive bidirectional identity experiment recovered none of those
  frames and is preserved as a failed control.
- A rigid fit at frame 50 reduces held-out image median error from **292.71 px
  to 12.60 px**, but its held-out estimated depth error is **26.71 mm**. It fails
  the source gate and is not promoted.
- A CPU MANO implementation closes against all **117 finite native source hand
  frames**, with maximum vertex/joint difference **5.69e-8 m**. This checks the
  geometry convention; it does not validate missing poses.
- Eight articulated fits, with the existing root/finger pose delta bounds, still
  reject the carry peak. Only frame 89 passes that experiment's image/joint-depth
  checks. Replacing bone-centre depth residuals with visible-skin residuals does
  not solve the interval. All failures remain saved. Sparse skin pixels cannot
  identify the pose of every occluded finger.

## Implemented recovery path

[`recover_source_hand.py`](../../../scripts/track3/recover_source_hand.py) accepts
hash-bound source RGB and explicitly inferred image points, then runs SAM2 point
prompts and fresh HaMeR crops. It can request a new pose when the old mask is
empty. CPU validation imports neither Torch nor the GPU models.

The adapter checks image hashes/grid, canonical clock and direct-observation
flags, and equality with the saved supported image points. Missing fingers keep
zero confidence. New masks must support four source points; no synthetic mask
is used. New priors keep fixed subject shape/scale and remain estimated,
non-metric and not directly observed. They are **not** accepted robot references.
The existing observation gate is not edited.

Ten focused CPU contract tests pass. The private eight-frame request validates
locally. Visual review confirms that the eight crops include the right hand and
handle region; forearm crop boundaries and occluded fingers remain limitations. **SAM2/HaMeR CUDA execution and downstream integration are unverified.**
Fresh records and image observations are separate outputs. Any later integration
must preserve the original records, merge only the explicitly selected new priors
into a new reconstruction run, and pass source-fit, motion, actual URDF limits,
the unchanged **1 mm native reference-contact gate**, swept checks and grasp
proximity review before PPO. No baseline reset may disguise a large upstream
source repair as a small robot morphology correction.

The private request, footage, geometry arrays and MANO/model weights remain local.
The producer archive contains code only. The model geometry convention was
cross-checked with the [upstream MANO layer](https://github.com/lixiny/manotorch/blob/master/manotorch/manolayer.py).

## Next bounded experiment

Use the existing perception runtime for source frames 25, 35, 40, 45, 50, 55,
67 and 99, inspect fresh masks/poses against the source, and stop if source checks
fail. This is a source-recovery probe, not additional PPO.

Brev is **STOPPED**. The shutdown-only preflight at 21:41 UTC still returned
`UnrecognizedClientException` / invalid provider security token. No START or paid
job was attempted. Resolve that control failure before a GPU probe; then install
an independent stop guard, bind the private mounts/code revision, enforce a short
lease, hash-verify backups and confirm provider STOPPED after the run.

The owner confirmed Brev is down. Keep all GPU work unqueued and avoid further
provider retries during the reported outage. The recovery contract tests are now
included in the existing CPU CI workflow.

## CPU handoff integration

The next adapter now hashes fresh output files and can merge selected missing
priors into a new reconstruction input while preserving the other hand, original
records, source indices and confidence arrays. The revision-2 private request
binds all 302 canonical hand records. Twenty focused tests pass, including
synthetic-backend execution through the real downstream loaders; real model
inference remains unverified. See
[SOURCE_HAND_RECOVERY_HANDOFF.md](SOURCE_HAND_RECOVERY_HANDOFF.md).
