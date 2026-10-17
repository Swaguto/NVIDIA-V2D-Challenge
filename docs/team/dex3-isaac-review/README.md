# Hand continuity and twelve-episode Dex3 checkpoint

## Latest ordered implementation and evaluation

[Issue 10 scope and remaining work](ISSUE_10_STATUS.md): the CHORD baseline is incomplete. Current grip searches are separate from PPO training.

See [measured grip learning and reproduction](GRIP_LEARNING.md) and [all measurements](grip-learning-results-v1.json). The contact-time wrench readout, numerical reset recovery, fresh PPO run and actual official kit wrapper were checked. The latest run logged 3,000 iterations / 24,576,000 environment steps. **The full physical grasp remains unresolved.** e002 is development data outside the scored roster; hidden performance is unverified.

Brev is **STOPPED**, backups are hash verified, and the conservative cumulative compute bound is $28.9614 of $40 (not an invoice). The local v8 input pack must be transferred privately. Earlier Isaac replays and rejected comparisons below remain archived.

This branch combines `codex/hand-continuity` at `9f93a50` with
`Evc-r/dex3-motion-replay-baseline` at
`3db10766bf09bca10fe1b452afe5c78e1649c870`. Both histories are retained.


## Archived e002 left-hand repair at source speed 0.5

The new side-bracing handle grasp completes both pot grasps with **2.040 / 3.042 mm**
maximum relative slip, **1.034 / 1.656°** rotation drift, full required-digit
support and no aborts. All four grasp position-retention diagnostics pass in the
final hand-scoped run. Lid rotation remains **16.404 / 11.851°**; a full physical
policy and official evaluation are not accepted.

Left wrist motion above 3 Hz drops from **2.103° to 0.0233° RMS** and from
**1.008 to 0.0445 mm RMS**, over the same source interval 5.5–7.5 s. All three
actual distal pads stay loaded there, with an opposed thumb–middle pair. The
previous quiet-control/old-grip failure is preserved; aborted stationary hands
are not counted as low-jitter successes.

See [the repair and remaining limits](HANDLE_GRIP_REPAIR.md),
[full measured outcomes](handle-repair-results-v1.json),
[the exact geometric search receipt](handle-grip-search-v1.json), and the local
[v6 input-pack receipt](local-pot-task-input-receipt-v6.json). Inputs remain local.
The final recipe uses source43 (`437ef44`), the unchanged pinned runtime image,
seed22, 0.0025 s steps and the optional `grip` solver. A fresh matching free-space
gate and complete replay both pass. 100 CPU tests pass. The source42 isolated
side-grip fixtures also pass both gravity lifts after support removal.

All 109,567 result/log files (34,302,839,389 bytes) are backed
up and hash-verified through the immutable inventory chain. Brev is confirmed
**STOPPED**. Conservative cumulative compute bound: **$15.0380 of $20**;
not an invoice. Original observations, masks, geometry and source timestamps are
preserved. Official solver compatibility and hidden-episode performance remain
unverified.

## Previous e002 reference-repair checkpoint

Complete e002 transport remains unresolved; no full physical task or official
policy is accepted. Reference jitter is bounded, settled-grasp registration
preserves the table frame, and compliant carry retains its world tracking gains.
Original observations, geometry, timestamps, masks and source-error references
are preserved. Motor and wrist force limits, mass, friction and gravity remain.

Five new full comparisons preserve all 343 original frames and 20 labelled
settling frames, with fresh passing wrist gates and gravity/contact controls.
The compensated fixed-posture grasp remains tight through source time 7 s, then
loses the pot near 7.9 s. The smoothing-only comparison improves lid position
drift to 5.26 / 6.07 mm with full required-digit contact coverage; rotation still
drifts 10.31 / 11.24°. These partial diagnostics do not accept the full task.
A bounded underside search fitted 44 proposals; none passed all contact,
upward-support, collision and floor checks. No rejected pose was GPU-tested.

The shared native motor cap repair and both isolated lift successes remain:
49.6 / 51.2 mm lift with 1.86 / 1.42 mm relative slip at 400 Hz. The isolated
probe verifies signed 0.76 Nm saturation and unsaturated command scaling.
Historical clipped torque telemetry does not certify older separate-effort runs.

See [new measurements](grasp-reference-repair-results-v1.json),
[implementation and remaining work](GRASP_REFERENCE_REPAIR.md),
[grasp search](geometry-search-results-v1.json), and
[earlier motor/lift checks](motor-contact-repair-results-v1.json).
95 CPU tests pass. The optional grip solver differs from default settings;
official evaluation compatibility and hidden-episode performance are unverified.
No policy training is accepted from these failures.

The private [v5 reproduction receipt](local-pot-task-input-receipt-v5.json)
binds the 205,469,974-byte local input pack, exact source overlays, seeds and
run receipts. Nothing in that pack has been published. New physics comparisons
use source37, source39 or source40 as specified in each recipe, the pinned
simulator image, `--physics-dt 0.0025`, and `--contact-solver grip`.
Source41 contains the final CPU grasp-search tools and rejection records.
The [URDF frame check](geometry-urdf-equivalence-v1.json) verifies the archived
search asset against 22 recorded SDK link poses. A complete episode requires
a fresh matching free-space gate. Team members must receive the private pack
separately; reproducing a diagnostic does not establish a successful submission.

All 97,002 result/log files (30,048,003,316 bytes) are backed up and hash-verified
through the immutable inventory chain. Brev is confirmed stopped. Conservative
cumulative compute bound: $14.1958 of the approved $20; not an invoice.

## Executed result

Twelve episodes: e002, e011, e012, e013, e018, e021, e023, e024, e031,
e040, e041 and e042. Total: 3,096 source frames, 24 actual Isaac camera
videos, 12 comparisons and 12 human-input videos. Simulator: Isaac Sim
6.0.0.1, Isaac Lab release 3.0.0 at
`28a37cecdd433c22d9eabd6a5954add9f13a8951` (distribution metadata 6.1.11).

The numerical audit verifies frame axes, visual and metric masks separately,
robot assets, joint limits and root/joint/link/camera readback. All twelve
comparison videos were played through at normal speed and eight aligned
samples per episode were inspected. See `receipts/` for exact hashes and counts.

**Kinematic preview only. Gravity and collision response were disabled.** Object
poses are assigned from reconstruction each frame; the robot does not cause their
motion. No physical grasp, learned policy, hidden-evaluation success, or issue 6
quality acceptance is established. Fingertip fit remains approximate. Unknown
hand geometry is absent. e021 retains a long missing right-hand interval;
e002 has five leading frames without a native camera; e042 input articulation
remains approximate. e039 could not be recovered; the user confirmed no backup.

## Sources and private inputs

`scripts/dex3_review/frozen/` preserves the executed scripts byte for byte,
including original machine paths, expired deadlines and superseded preparation
helpers. They are historical records, **not the portable entry point**. The final
executed renderer is `replay_continuity_dex3_sim6.py`, SHA256
`83cfb67e3b88a65db092b6f677f7ec05b58ae8d53114034bc47efff30d2d3f37`.
`prepare_renderer.py` predates later edits and must not regenerate that file.
The orchestration snapshot preserves the nineteen-episode runs and source-image
anchor/mask recovery helpers. Core repair changes remain in the reconstruction
package and their tests are retained from the original branch.

Receipts and source are committed; footage, model weights, native caches and
licensed MANO assets remain private. Full outputs are preserved on the existing
pod's `/workspace/hand-continuity-repair/dex3-isaac-review` and the private review
backup. `gallery.html` is the exact gallery source; its relative media paths
require the private media pack. The local review is
`http://127.0.0.1:8766/continuity_fix/dex3_isaac/` on the producing Mac only.

## Next steps

The portable image and Brev run instructions are in
[`docker/dex3-review/README.md`](../../../docker/dex3-review/README.md).
Physics validation should use new outputs and retain this kinematic baseline.
Use the current official Track 3 submission kit for policy evaluation: its six
metrics include AUC, SP-SR, MP-SR, MPPE, RPE and CD-O. The preview's provided
CAD and four legacy proxy metrics do not establish a valid final submission.
