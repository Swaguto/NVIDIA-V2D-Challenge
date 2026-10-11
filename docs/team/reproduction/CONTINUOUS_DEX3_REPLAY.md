# Continuous Dex3 replay: e012

This adds explicitly labelled **presentation holds** to the frozen automatic e012
reconstruction. Missing observations remain missing in every measurement and
contact check. A hold copies the last supported robot pose and its full FK link
poses. A leading gap uses the first supported pose and records that future source
frame. A hand with no supported pose fails instead of inventing a robot target.
The algorithm has no episode-specific frames, crops or thresholds.

The automatic reconstruction is still rejected for hand-placement accuracy. See
[recovery findings](AUTOMATIC_HANDS_RESULTS.md). Continuous visibility does not
establish accurate fingers, contact, physical manipulation or learned control.
Reacquisition can still produce a motion jump. This replay removes disappearing
models; it does not establish jitter-free or accurate recovered motion.

## Inputs and runtime

Use the immutable **e012 replay kit v3**, plus the frozen automatic-recovery
evidence pack and this repository revision. The delivery handoff lists their
checksums and how to restore the Git bundle. Everyone with repository access
also needs the source data, model weights, calibration and runtime; code access
alone cannot reproduce a licensed model or unavailable video.

The automatic method freeze SHA256 is
`81d13102c669e271a8c4e0cc107e50f9e8887eeb8c325cbda8f4b1410690a758`.
Staging uses saved fresh-inference observations from that run. It does not rerun
perception. To reproduce inference from the source video first, follow
[the automatic recovery protocol](AUTOMATIC_HANDS_PROTOCOL.md), preserving its
input, source, model and runtime locks and comparing frame-level geometry.

Keep three environments separate: refinement (Python 3.11.17, torch 2.9.1+cu128,
gsplat 1.4.0), IK (Python 3.11.17, NumPy 2.2.6, pin 3.7.0, pin-pink 4.2.0,
DAQP 0.10.3), and simulator (Python 3.12.15, Isaac Sim 6.0.0.1, Isaac Lab commit
`28a37cecdd433c22d9eabd6a5954add9f13a8951`). Full measured environment locks
are in the kit and this run's evidence. Refinement used an RTX 5060 Laptop GPU;
simulation uses CPU physics and actual RTX rendering on an A40, driver 595.91.07.
The kit's `setup_sim6.py` and README describe graphics libraries and the pinned
installation. A second clean-machine installation is not yet validated.
Supply licensed MANO assets separately with the kit's recorded model hashes.

## Reproduce step by step

Set `CODE` to this repository's absolute path, `KIT` to the extracted v3 kit,
`AUTO` to the automatic evidence episode directory ending in `e012`, `OUT` to
a new directory, and `MANO` to your licensed MANO asset root. Use absolute paths.
Never overwrite the reference kit or a previous run. Run from Linux.

In the refinement environment:

```bash
python "$CODE/scripts/reproduction/stage_automatic_hands.py" \
  --kit "$KIT" --episode-root "$AUTO" --out "$OUT/inputs"

python "$CODE/scripts/reproduction/reproduce_refinement.py" \
  --kit "$KIT" --inputs "$OUT/inputs" \
  --source "$KIT/refinement-source" \
  --source-manifest "$KIT/evidence/receipts/refinement-source-hashes.json" \
  --asset-manifest "$KIT/evidence/receipts/mano-required-hashes.json" \
  --mano-assets "$MANO" --out "$OUT/refinement"
```

Staging verifies the original kit, same episode and timeline, frozen method and
all selected pose/mask identities. It records a SHA256 lock for every input.
Refinement verifies that lock, optimizer source and MANO assets, then retains the
recorded seed 42 and ten epochs. Completion requires `run-receipt.json` status
`completed`, a scene hash and the full 181-frame timeline.

In the IK environment:

```bash
python "$CODE/scripts/reproduction/rebuild_motion.py" \
  --kit "$KIT" --inputs "$OUT/inputs" \
  --refinement "$OUT/refinement" --out "$OUT/motion"

python "$CODE/scripts/reproduction/continuous_dex3.py" \
  --retarget "$OUT/motion/retarget" --out "$OUT/presentation"
```

This exports the refined scene, runs the pinned CHORD/Pink/DAQP IK with 100
iterations and scale 1, and runs penetration diagnostics. The original retarget
payload keeps NaNs for unobserved hands. `presentation.npz` separately records
finite display targets, `source_robot_valid`, `held`, `pose_source_rows` and
`pose_source_frames`. Observed robot targets are bitwise unchanged. Held poses
are never written back into the original trajectory or passed off as contact
evidence. The previous kit's policy of hiding unobserved hands remains unchanged;
this new adapter is a separate presentation mode.

In the pinned simulator environment, after normal NVIDIA license acceptance:

```bash
python "$CODE/scripts/reproduction/replay_continuous_dex3_sim6.py" \
  --source "$KIT/source/robotic_grounding" \
  --bundle "$OUT/motion/bundle" --retarget "$OUT/motion/retarget" \
  --gate "$OUT/motion/gate" --presentation "$OUT/presentation" \
  --out "$OUT/simulator" --width 1280 --height 960 \
  --headless --enable_cameras --device cpu --rendering_mode performance

python "$CODE/scripts/reproduction/audit_continuous_replay.py" \
  --bundle "$OUT/motion/bundle" --retarget "$OUT/motion/retarget" \
  --presentation "$OUT/presentation" \
  --replay "$OUT/simulator"
```

Each imported Dex3 is an actual floating articulation with seven finger joints
and the pinned URDF/STL meshes. Both remain instantiated and visible throughout.
Gravity and collision response are disabled for this kinematic demonstration.
The overview camera includes full-clip object and hand bounds. CPU PhysX link
transforms are explicitly synchronized to USD for the renderer. Blue identifies
left Dex3, orange right Dex3. The video labels missing observations `HELD`.

## Acceptance and independent reproduction

Require `complete.json`, no `failure.json`, all 181 PNGs and 181 decoded video
frames, and a passing `independent-audit.json`. Every frame must contain at least
20 raw rendered review-color pixels for **each** hand. Every imported visual
mesh must retain visible USD state on every frame. All root/joint rotations,
root positions and physical/rendered link positions must match display targets
within `1e-4` rad/m.
Every object mesh must retain visible USD state, and its renderer transform
must match the original exported object pose/scale within `1e-6` per matrix entry.
The independent audit reopens PNGs and video, verifies artifact hashes and
recomputes errors from recorded simulator states.
Review contact sheets and low-pixel frames too: color counts verify model
presence, not visibility of every finger or absence of natural occlusion.

For another machine, compare original input hashes and frozen method/model
hashes first, then frame IDs, side identities, masks and hold provenance exactly.
Compare refined camera/object/hand numeric arrays and IK results separately;
do not require equal NPZ ZIP bytes or MP4 bytes as a numeric-equivalence test.
Require the simulator tolerances above. Different GPU kernels/renderers can
change optimization floats or pixels; any discrepancy must be measured and
reported. Exact cross-machine inference/optimization has not been demonstrated.
Do not relax tolerances or claim a successful reproduction from matching counts.

The presentation and audit tools are episode agnostic, but only e012 is run here.
Use a matching episode kit and full timeline for other episodes. This result
does not certify all public episodes or the twenty private evaluation episodes.

## Measured result

The staged and retargeted e012 trajectory contains 153 left and 122 right source
fits out of 181 frames. Display continuity holds 28 left and 59 right frames;
275 original observations remain unchanged. Median three-tip IK residuals are
18.38 mm left and 12.63 mm right relative to reconstructed targets, not human
ground truth. There are zero finger joint limit violations. The conservative
hand/hand proxy checks 103 frames, maximum overlap 7.72 mm. Both object collision
checks are excluded for mesh/concavity reasons: zero fully checked contact passes.
These limitations remain even if the renderer continuity audit passes.

The final handoff includes renderer results, independent audit, actual replay
and source/reconstruction comparison video, plus the full artifact inventory.

Both hands appear in **181/181 actual simulator frames**. Minimum raw rendered
review-color pixels are 1,492 left and 2,671 right; all eight visual meshes per
hand retain visible USD state. Both object meshes remain visible and match their
exported object transforms exactly. Maximum hand link error is `8.67e-8` m;
maximum root orientation error is `2.81e-7` rad. The independent disk/video/state
audit passes. Exhaustive contact sheets cover every frame; fresh refined overlays
at frames 22, 75, 148 and 151 retain the hand-accuracy warning.

A second export/IK/gate run from the same refined scene matches all 43 arrays
exactly. A second real Isaac Sim run on the same A40 matches all shared readback
arrays and all 181 raw PNG pixel arrays exactly. These are same-runtime checks,
not a demonstration of reproducibility on another person's GPU.
The receipt is [continuous-dex3-e012-result.json](continuous-dex3-e012-result.json).

To compare an independent motion rebuild and a second simulator run:

```bash
python "$CODE/scripts/reproduction/compare_dex3_payloads.py" \
  --reference "$REFERENCE/motion" --candidate "$OUT/motion" \
  --out "$OUT/motion-comparison.json"
python "$CODE/scripts/reproduction/compare_renderer_outputs.py" \
  --reference "$REFERENCE/simulator" --candidate "$OUT/simulator" \
  --out "$OUT/renderer-comparison.json"
```

The simulator comparison reports pixel differences separately; cross-GPU pixels
are not required to be identical. Both runs must pass their independent audits.
For the deterministic presentation layer, run
`python scripts/reproduction/test_continuous_dex3.py`. Four controls cover hold
provenance, missing-all-hand rejection, target/FK preservation and tampering,
and refusal to export a changed refined scene.
