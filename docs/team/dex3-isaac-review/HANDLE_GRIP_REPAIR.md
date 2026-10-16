# Left wrist jitter and pot handle grip

## Verified left-hand result

The final e002 `coupled-task-side-left-quiet-v1` replay preserves all 343 source frames
and 20 labelled settling frames. Both pot grasps complete without aborts,
with required-digit support throughout their measured loaded intervals:

| Pot grasp | Maximum relative translation drift | Maximum relative rotation drift |
| --- | ---: | ---: |
| First | 2.040 mm | 1.034° |
| Second | 3.042 mm | 1.656° |

These are simulated grasp diagnostics, not an official task score. Restoring the
earlier right-hand controller also retains both lid grasps: 7.977 / 6.013 mm
relative translation drift and full required-digit support. All four grasps
complete without aborts and pass the existing position-retention diagnostic.
The lid still rotates by 16.404 / 11.851° relative to its hand. Full task and
official evaluation acceptance remain separate, unresolved gates.

The preceding `coupled-task-side-quiet-v1` combination also retained both pot
grasps (4.164 / 0.833 mm drift), but failed the lid's second grasp. That complete
failed comparison is retained. The earlier quiet controller with the old grasp
failed acquisition; low jitter in its aborted frames is not success evidence.

## Measured jitter

The comparison uses source time 5.5–7.5 s, the same physical clock, a third-order
3 Hz high-pass filter and 0.1 physical second boundary trimming. It measures
actual SDK wrist poses, not rendered pixels or finger target changes.

| Left wrist | Earlier compensated frozen grasp | Source42 side-bracing quiet grasp | Final source43 hand-scoped run |
| --- | ---: | ---: | ---: |
| Position residual RMS | 1.008 mm | 0.0376 mm | 0.0445 mm |
| Rotation residual RMS | 2.103° | 0.0143° | 0.0233° |
| Aborted readback samples | 0 | 0 | 0 |
| Required-digit support coverage | 100% | 100% | 100% |

In the final run, all three distal pads exceed 0.5 N throughout this interval.
Thumb and middle normals are opposed throughout (median dot −0.950); the
thumb–index pair is oblique (median −0.418). The preceding source42 run instead
has thumb–index opposition in 99.875% of these samples. The auditor reports all
three pairs as well as the original thumb–index metric. No retention gate was
changed. The final window's maximum recorded solver penetration is 2.66 µm;
this readback does not certify continuous mesh nonpenetration everywhere.
Low jitter after an abort is explicitly reported with the abort count and
must never be counted as successful stabilization.

The earlier wrist target itself has a 1.963° high-frequency rotation residual;
its actual wrist has 2.103°. Its finger posture targets are constant in this
interval. The measured-pose loop is therefore an observed contributor to wrist
motion. The successful candidate changes geometry and control together; it
does not isolate each setting's causal contribution.

## Changes

- Fit the middle pad to a real cooked-mesh face at the handle side. The prior
  middle pad pushed down from above. Surface point coincidence alone had
  allowed edge contacts with the wrong supporting normal.
- Independently check all three distal contacts, opposed thumb/index normals,
  unchanged joint limits, fixture clearance and ≤1.5 mm cooked-piece SAT depth.
  Transfer the donor to the other phase and recompute anchors, motor efforts
  and a 4 mm pregrasp opening. Recheck the complete nominal load plan.
- Reduce measured-object carry weight from 0.9 to 0.5, reducing the gain
  compensation from 10× to 2×. The configured weight is validated in [0, 0.9].
- Use nominal payload gravity compensation for the quiet carry instead of
  cancelling the instantaneous measured primary-object reaction. Keep the
  frozen finger posture; live allocation remains off in this comparison.
- Allow four explicit settings to be scoped by hand through
  `hand_control_overrides`: settled registration, carry compliance, feedback
  weight and payload compensation. All wrist consumers use the resolved hand
  settings, and the simulator report records them. Historical global defaults
  remain supported.

The new nominal carry feedforward maximum is 0.33334 Nm, versus 0.51166 Nm for
the earlier grasp. The existing 0.55 Nm feedforward check and shared native
0.76 Nm motor cap remain. These feedforward values are plans, not measured
solver motor torques.

## Validation and reproduction

Both new isolated squeeze fixtures pass the existing lift diagnostic after
support removal: 51.185 / 49.500 mm lift and 3.447 / 3.378 mm relative slip.
The fixture is 1.5 s and does not by itself certify full transport.

Source42 (`973d371`) ran the new geometry and global quiet comparison. Source43
(`437ef44`) adds scoped controller settings and the physical-clock jitter
auditor. Each source archive has a file manifest and archive SHA. The pinned
runtime image is unchanged. Use the exact source overlay and private seed for
the selected recipe, a fresh output directory, `implicit_pd_ff`, 0.0025 s
physics steps and the optional `grip` solver profile. A complete replay requires
a fresh matching `grip-task-free` gate before `grip-task`.

The input pack stays local. It contains the original source geometry, exact
source overlays, proposals, transferred commands, recalculated efforts and run
receipts. The extended post-run contact-pair auditor is separately hash-bound as
`analysis-tools/audit_wrist_jitter.py` in v6; the archived simulator source43 is
unchanged. No source image, pose observation, timestamp or mask is changed.
No object attachment, external object wrench or object pose write after
initialization is introduced. No policy training or hidden-evaluation success
is claimed. Solver compatibility with official evaluation remains unverified.

100 CPU tests cover the current implementation, including pad-face selection,
hand isolation, fixed registration, physical-frequency noise measurement,
quaternion sign equivalence and rejection of missing readback times.
