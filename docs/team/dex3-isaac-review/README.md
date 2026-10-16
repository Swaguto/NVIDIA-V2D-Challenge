# Hand continuity and twelve-episode Dex3 checkpoint

This branch combines `codex/hand-continuity` at `9f93a50` with
`Evc-r/dex3-motion-replay-baseline` at
`3db10766bf09bca10fe1b452afe5c78e1649c870`. Both histories are retained.


## Current e002 physics checkpoint

Complete e002 transport remains unresolved; no full physical task or official policy is accepted.
The motor feedforward bug is repaired: posture and feedforward now share the
native implicit drive cap. The inertia-based simulator probe verifies signed
0.76 Nm saturation and unsaturated command scaling. Historical clipped torque
telemetry cannot certify the combined budget of the older separate-effort runs.

Both fresh pot lift fixtures pass at 400 Hz with the optional grip solver profile:
49.6 / 51.2 mm lift and 1.86 / 1.42 mm relative slip. The hand runaway after object
loss is repaired. Aborted or unsupported stationary phases fail the corrected
retention audit. Full replays preserve all 343 source frames and 20 labelled
settling frames; their losses remain in the measurements.

See [latest measured results](motor-contact-repair-results-v1.json),
[video findings and implementation](VISUAL_REFERENCE_REPAIR.md), and
[earlier campaign](COUPLED_TASK_DIAGNOSTIC.md). 87 CPU tests pass. The grip solver
profile differs from the default diagnostic solver; compatibility with official
evaluation settings has not been established. No hidden-episode result or PPO
training is claimed.

The private [v4 reproduction receipt](local-pot-task-input-receipt-v4.json)
binds exact inputs, source overlays and run receipts. The 180,579,458-byte
archive is retained locally and has not been published. The newest recipes use
source33, the pinned simulator image, `--physics-dt 0.0025` and
`--contact-solver grip`. A full episode requires a fresh matching free-space gate.
An input recipient must receive the private pack separately.

All 79,097 result/log files (23,795,980,176 bytes)
are backed up and hash-verified through the immutable inventory chain. Brev is
confirmed stopped. Conservative cumulative compute bound:
$12.5618 of the approved $20; not an invoice.
Reproducing a diagnostic does not establish an accepted submission.

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
