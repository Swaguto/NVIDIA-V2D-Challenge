# Issue 6 real-run validation helpers

These scripts record the native A40 validation procedure for PR #32. They are
experiment replay helpers with explicit `/workspace` paths, not a replacement
for the package CLI. Final acceptance is recorded in the module's
[validation record](../../modules/v2d_issue6/VALIDATION.md); a completed process
alone does not establish visual or contact accuracy.

## Runtime and inputs

- PR #33 image: `ghcr.io/swaguto/track3-runpod@sha256:46cc00a62d1076595b71401a96d17fbe53488afe1fe4d7a9784bf3512ac3e908`.
- Validation source: `/workspace/pr32`, commit `af6ed6fdd46b55e5ca0254246a1ad18275bfebd4`.
- Dyn-HaMR: `fa9cd7412c205fd15ee4139c8caacf79bf6167e6`, at `/workspace/Dyn-HaMR`.
- Enhanced HaMeR fork: `2a5b8ee852d87fdb56d65df7fb0e473c0d4f7648`.
- ViPE: `fdc79359ec6d063c86b9d203ca3d7054a9636823`, at `/workspace/ViPE`.
- SAM2: `2b90b9f5ceec907a1c18123530e92e794ad901a4`.
- Runtime: Python 3.11, PyTorch 2.5.1+cu124, NumPy 1.26.4. The hand runtime
  is `/workspace/hand-runtime/bin/python`; ViPE uses the image's environment.
- Full rectified `ego_cam_c` bundles at `/workspace/issue6-full-inputs/e002`,
  `e013`, `e022`, with 343, 190 and 189 source frames. Use the Track 3 adapter
  to prepare metric stereo depth, calibrated intrinsics, observed CAD poses and
  masks. Depth here is calibrated SGBM stereo disparity. Reference object poses
  are not inputs; 58 absent episode-2 pot poses remain unknown.
- User-licensed MANO at `/workspace/mano` and
  `/workspace/hamer-weights/_DATA/data/mano`; actual HaMeR weights at
  `/workspace/hamer-weights`, detector and SAM2 weights at
  `/workspace/pr32-models`. Models, private credentials and raw data are not
  distributed here. The image alone does not supply the licensed assets.

Apply `dynhamr-native-fixed-shape.patch` to the pinned Dyn-HaMR checkout. Its
optional-import and zero-weight BMC changes permit the upstream defaults to run
without fake assets. HMP and BMC remain disabled by those defaults. Setting
`V2D_OPT_HAND_SHAPE=0` also fixes shape during smoothing, used for episode 22
after free-shape fitting visibly distorted its left hand.

## Replay sequence

With the prepared inputs and runtime packages installed, copy these helpers to
`/workspace/pr32-evidence`. Use new output directories; preserve old results
before reruns. Keep the source checkout fixed while pipeline stages execute.

1. Run `enhanced_hands.py --episode N` with the hand runtime. For episode 22,
   add `--exclude-reviewed-bystander-region`. It rejects upper-left background
   candidates before selecting by confidence and saves all candidates. This
   constraint came from RGB review of this episode; do not apply it to other
   episodes as a general wearer-identification method.
2. Run `vipe_stereo_cameras.py --episode N` with ViPE installed and the image's
   GPU runtime. Previously measured cameras were reused during targeted hand
   corrections, so an end-to-end fresh run need not reproduce bit-identical
   camera estimates.
3. Run `full_hand_masks.py --episode N` with the hand runtime. Actual SAM2 masks
   are prompted by detector boxes. Unsupported observations stay unknown.
4. Run `run_temporal_fit.py --episode N` with the hand runtime (50 root and 300
   smoothing iterations). Set `V2D_OPT_HAND_SHAPE=0` for episode 22. Verify the
   final checkpoint's betas equal the initial checkpoint before accepting it.
5. Run `convert_and_validate.py --episode N` with the hand runtime. It converts
   the final checkpoint, restores the original frame axis, applies observation
   gates, and executes validate/hands/depth/align/contacts/report.
6. Run `probe_dynhamr_geometry.py --episode N --all-frames --converted-records
   /workspace/temporal-hands/eNNN/converted`. Every persisted converted record
   must agree with upstream MANO vertices within `1e-5` metres. This checks
   conventions, not independent ground-truth accuracy.
7. Run `measure_temporal_consistency.py`, `measure_aligned_consistency.py`,
   `make_review_media.py` and `review_motion_extremes.py`, each with `--episode N`.
   Review sampled overlays plus consecutive frames around measured motion
   extremes. Measurements use estimated ViPE world coordinates.
8. Audit downloaded contact artifacts with `audit_completed_contacts.py --root
   REVIEW_ROOT`. Each `eNNN` directory contains the original relative
   `issue6-full-runs/calibrated-temporal/eNNN` paths. Unknown contacts must stay
   unknown, and bridging requires uninterrupted valid same-track observations.
9. After acceptance, `render_combined_overlay.py --episode N` renders observed
   CAD and final temporal MANO in one calibrated source-camera view. It uses the
   same CAD transforms as contacts, depth-tests hands and objects together, and
   omits missing poses. Inspect the video and its montage before delivery.

Save and verify all required evidence before stopping the GPU. Keep failed runs
as failure evidence, never count them toward the three-episode acceptance gate.
No independent contact precision or successful robot training is claimed.
