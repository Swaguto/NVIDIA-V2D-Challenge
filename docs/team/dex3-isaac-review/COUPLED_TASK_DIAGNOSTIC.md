# Coupled e002 diagnostic: approach, grasp, transport and release

**Current status:** seven complete e002 trials fail carrying. The signed-force
handling bug is fixed; live allocation remains disabled. See the final validation
section below and `coupled-physics-campaign-v2.json`. All 70 CPU tests pass.

This is experimental robot-reference/controller code. No complete physical task,
policy or official score is accepted by this document. The isolated pot/lid
successes remain documented separately.

## Execution stages

`grip-task-free` checks the floating wrist controller before the coupled
`grip-task` stage. Both use `--candidate-controller implicit_pd_ff` and the same
exact private `--grasp-seed`. The full stage requires a fresh passing free-space
gate bound to the executor, wrist controller, grasp controller and seed hashes.
Use one output root for both stages; earlier runs cannot be overwritten.

`task_grasp.py` binds every existing source contact phase to an object-relative
robot grasp. It preserves source frames, time axis, masks and unaccepted target
flags. Approach starts above the object, fingers close before squeeze ramps,
then the wrist follows the object reference. Fingers open before retreat.
A final one-source-second (two physical seconds) postroll records release and
settling separately from the 343 original frames. No learned policy is loaded.

The robot remains physical during human occlusion: all robot collisions stay
active outside the explicitly isolated free-space stage. Human observation and
display flags remain recorded, without reclassifying them as observations.
Measured object pose assists approach only, then blends to the source command.
It is not an approved learned actor observation contract.

## Force and material contract

Three-pad force plans are recomputed at every original pot orientation. The
original extra-conservative coefficient 1 model becomes infeasible for the
second phase at source row 274, despite the static lift passing. The task model
uses coefficient 1.25, matching the nominal average of unchanged hand friction
2.0 and object friction 0.5. The inscribed friction octagon remains inside that
cone. The runtime explicitly authors and reads back the average combine mode
and coefficients. Motor feedforward remains capped at 0.55 Nm and the real
combined motor limit remains 0.76 Nm. Pot stiffness is 8 Nm/rad; lid stiffness
is 2 Nm/rad, with original damping and wrist caps.

Wrist feedback sums measured normal and friction contact forces and moments
about the wrist COM, filters their opposite at 20 ms, and bounds compensation to
10 N / 2 Nm per component within the unchanged 60 N / 60 Nm total wrist caps.
It requires the declared measured contact digits. Contact buffers are copied
before another getter runs and checked against the aggregate normal forces.
A mismatch preserves raw diagnostic data and stops execution. Initial cached
contact data is excluded. Objects receive no external wrench or attachment.
The API contract is from NVIDIA's [RigidContactView documentation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html).

## Initial scene problem and bounded projection

The first coupled attempt passed free-space tracking but stopped on a contact
buffer consistency check before grasping. Its raw initial cooked scene also
reveals approximately 8.5 mm penetration into the inferred table and up to
21 mm pot/lid convex-piece overlap. A cross-section of the original triangle
surfaces confirms that source placement and collision approximation both
contribute. Those conditions must be resolved before interpreting a grasp loss.

`scene_projection.py` proposes upward corrections using the recorded cooked
shapes and inferred support plane. It uses conservative face-axis overlap
intervals, preserves clearance while bounding correction rate, and caps all
corrections at 6 cm. It does not alter meshes, scale, orientation or source arrays.
The current proposal starts at approximately +8.7 mm for the pot and +34.4 mm
for the lid, reaching at most 36.5 mm. These are inferred corrections, not ground
truth. Initial object pose is written once; subsequent object state is dynamic.
Fresh cooked shapes are compared with the planning shapes, and full SAT
independently checks initial object clearance before execution.

The correction also affects robot commands. Object errors are always measured
against the **original uncorrected source trajectory**, so the 3–4 cm discrepancy
remains visible. It is not a reconstruction or evaluation fix. Physical scene
and reference quality still need validation before PPO retraining or a team
release.

## Local inputs and limitations

`local-pot-task-input-receipt-v1.json` binds the locally retained input pack,
verified pot/lid proposals and source overlays. The first task proposal is
experimental. New projection proposals and their source snapshot stay local
until a new verified reproduction bundle is prepared. No videos, geometry,
MANO weights or credentials are published in this change.

CPU tests cover phase continuity, closure/loading/release order, contact buffer
offsets and unused capacity, wrench moment/sign, compensation limits/decay,
vertical clearance, and bounded smoothing. They do not substitute for simulator
validation. Finite reference collision checks are not continuous clearance proofs.

## Complete coupled attempt, before the 4 N minimum

Source `950d98d` completed all 343 original frames plus 20 inferred postroll
frames. Initial scene clearance and the fresh wrist tracking gate passed.
The scene no longer ejects objects before grasping. The lid position error was
7.6 mm median / 21.8 mm p90; the pot was 18.5 mm median / 82.9 mm p90, peaking at
112.2 mm. These are local raw-source tracking measurements, not official scores.

The pot loses a required measured pad contact around source frame 105 and then
slips away, while wrist error stays only a few millimetres. No contact with the
other object is recorded at the sampled failure times. The 2 N minimum nominal
force plan therefore remains unsuitable as a training reference. The follow-up
raises the planned minimum to 4 N for every pot pad, with all source orientations
feasible and at most 0.511 Nm feedforward, below the unchanged 0.55 Nm cap.
Per-body/per-object normal forces and actual combined finger effort are now
recorded. Compensation contact eligibility uses the primary object only.
The 4 N version still requires a new simulator run; no policy is retrained yet.

## Tactile seating follow-up

The 4 N open-loop minimum delayed the first pot loss but did not prevent it.
Per-pad readback identifies the middle pad: its actual load falls below 0.5 N
while the opposing pads remain strongly loaded. Planned force is insufficient
as a contact guarantee. The 4 N run is retained as a failed candidate, and its
wrist tracking gate also fails.

The next candidate adds bounded unilateral pressure feedback for the three-pad
pot grasp. A weak distal pad receives a normal-direction finger target correction
using its recorded Jacobian, seeking a 4 N floor. The integral unwinds above 6 N.
Predicted extension is capped at 4 mm and each joint correction at 0.08 rad,
within the original URDF limits. Strong pads are not deliberately opened. State
resets outside the squeeze interval and between phases. All original motor and
wrist effort caps remain unchanged. This is diagnostic measured-contact feedback;
its compatibility with the learned actor contract is not established.

`audit_coupled_grip.py` verifies state/spec hashes, excludes inferred postroll from
source-object metrics, reports each grasp's relative drift and contact coverage,
and records actual combined finger effort when available. It does not certify
an official score or automatically accept a full task.

## Controller isolation after the tactile trial

The tactile candidate also failed during the first carry. It is retained for
inspection and is **disabled by default**. Measured reaction cancellation is now
an explicit experimental mode. The nominal payload comparison uses the nominal
0.3 kg gravity force and lever-arm moment already checked in isolated lifts,
with measured contact eligibility and filtered enable/disable. The current
comparison explicitly selects a 4 N planned minimum and nominal compensation,
without tactile target changes. The outcome is not yet accepted.

Controller choices are bound in the private proposal: `task_minimum_pad_normal_N`
(default 2), explicit `wrist_payload_compensation` (`nominal_gravity` or
`measured_reaction`), and boolean `experimental_tactile_seating` (default false).
All source orientations are checked again for the selected force floor.

Completed `grip-task` runs now produce a local `review/index.html`, browser-compatible
video, state/spec-bound grasp audit and video receipt before the wrist gate is
interpreted. Failed candidates remain reviewable. This adds no publication or
input upload. For an existing run:

```bash
python scripts/dex3_physics/package_coupled_preview.py \
  --run /private/results/grip-task/run/simulator \
  --out /private/results/new-review
```

## Earlier four-run checkpoint and then-prepared candidate

All four full carry candidates are rejected. `coupled-physics-campaign-v1.json`
preserves their hashes and readback, including the failed pressure feedback and
nominal-gravity variants. The 2 N measured-reaction baseline has the best complete
local tracking among these attempts; it still loses the first pot grasp and is
not a training reference. The nominal comparison also loses the lid. No PPO
checkpoint is retrained, no hidden score is claimed, and no failed candidate is
promoted to an accepted controller.

The next disabled candidate is `experimental_live_contact_wrench`. It recomputes
patch centroids and inward normals from current copied PhysX contact data for
the primary object. Current finger FK supplies point Jacobians. A bounded nominal
load allocation produces squeeze torques; actual simulator joint gravity
compensation is added, with the original 0.76 Nm total command limit enforced.
During valid allocation, finger position targets follow current joint positions,
retaining drive damping while avoiding a competing geometric posture spring.
Missing contacts, infeasible forces or invalid data fall back to the frozen
position/effort candidate and are recorded. This mode cannot be combined with
experimental tactile seating. It is **CPU prepared only; no GPU success is claimed**.

The source now requires an explicit wrist payload mode for full task preparation.
`combine()` writes an explicit 2 N measured-reaction diagnostic baseline. This
prevents legacy inputs from silently selecting a different, failed comparison.
The new live allocation is off unless explicitly enabled in the private proposal.

69 CPU tests pass, including current contact buffer selection, normal sign,
unused capacity, translation invariance, finger gravity addition and motor-cap
rejection. A complete hash-checked e002 candidate is prepared locally for a fresh
free-space gate and coupled rollout. It still needs simulator verification.

All 34,242 saved files (9,881,583,857 bytes) are SHA-256 verified on the owner's
Mac. Brev `dngrtbldz` is confirmed STOPPED. The conservative cumulative compute
bound is $9.7755 of the approved $10, not an invoice. That earlier checkpoint required a renewed compute allowance, since approved
and used for the three trials recorded below. Inputs remain local.

## Historical v22 reproduction (failed activation check)

Obtain the owner's private `dex3-pot-task-inputs-local-v2.tar.gz` through an
approved private channel and verify it against
`local-pot-task-input-receipt-v2.json`. Its 119,886,849 bytes have SHA-256
`bba120e71b1abe71734d3567ac2b065698d1f03f847f902abea031737d9df869`.
All members were individually hash-verified. The manifest distinguishes the
verified isolated pot fixtures, rejected historical baseline, and next untested
candidate. Input videos/geometry remain local; no package upload is performed.

For the next candidate, extract `sources/physics-source-v22.tar.gz` into a fresh
code directory and use `combined-task-grasps-live-contact-v6.json`. This overlay
is based on commit `674910c`; archive SHA-256 is
`43e62050e28713e534881062f058476001789cc18738825bddbec433efc46497`.
Use the pinned Isaac digest and GPU/driver/EULA requirements in the existing
container documentation. Set absolute paths for the extracted code and inputs:

```bash
export V2D_IMAGE=ghcr.io/swaguto/nvidia-v2d-challenge-dex3-review@sha256:8585588357d898929bf6f80f0011441cbcab07a3d95e2f5f4d0e24bc5c177190
export V2D_CODE_DIR=/private/extracted-code-v22
export V2D_DATA_DIR=/private/extracted-inputs-v2
export V2D_PHYSICS_ARCHIVE="$V2D_DATA_DIR/input-pack.tar.gz"
export V2D_GRIP_SEED="$V2D_DATA_DIR/combined-task-grasps-live-contact-v6.json"
export V2D_OUTPUT_DIR=/private/new-contact-candidate-results
mkdir -p "$V2D_OUTPUT_DIR"
# Set ACCEPT_EULA=Y and OMNI_KIT_ACCEPT_EULA=YES after accepting NVIDIA's license.
cd "$V2D_CODE_DIR"
docker compose -f docker/dex3-review/compose.yaml \
  -f docker/dex3-review/physics.compose.yaml \
  -f docker/dex3-review/grip.compose.yaml run --rm physics \
  --stage grip-task-free --candidate-controller implicit_pd_ff \
  --grasp-seed /physics-inputs/grasp-seeds.json
# Run only if that fresh gate passes; use the same output directory and sources.
docker compose -f docker/dex3-review/compose.yaml \
  -f docker/dex3-review/physics.compose.yaml \
  -f docker/dex3-review/grip.compose.yaml run --rm physics \
  --stage grip-task --candidate-controller implicit_pd_ff \
  --grasp-seed /physics-inputs/grasp-seeds.json
```

The runtime enforces the free-space prerequisite again. Inspect
`grip-task/review/index.html`, the grasp audit and wrist gate. Full task and
learned policy acceptance remain false. Back up outputs and stop Brev after the
run and whenever it is idle.

## First live-contact validation

The user approved another $10 (cumulative cap $20) for the existing stoppable
L40S. Overlay v22 passed the fresh free-space and full wrist checks, but the pot
still slipped. The live allocator was inactive for every loaded pot step: it
reported insufficient or missing patch pressure and therefore repeated the
frozen baseline. This is a failed activation check, not evidence that the live
torque controller was physically exercised.

The follow-up preserves the signed scalar multiplied by its reported normal
when forming force. Absolute scalar magnitude weights contact centroids; negative
scalars are no longer clipped to zero. Reversing both scalar and normal must
preserve the physical patch, covered by the regression test. New per-patch
telemetry records signed ranges, magnitude sums and resultant force, including
fallback cases, so the simulator's actual convention can be checked. The change
still needs a fresh GPU run. All 69 CPU tests pass. Input publication and PPO
training remain pending physical validation.

## Signed-force readback and posture comparison

Overlay v23 confirms negative contact-force scalars in this PhysX backend; their
product with reported normals matches the independently checked aggregate force.
The corrected allocator activates for 70/1830 first-pot load steps (3.83%), then
contact weakens and fallback dominates. It never activates in the second pot
phase. The full carry and wrist gate fail. Finger gravity compensation is at
most 0.02684 Nm and the live commanded motor effort at most 0.54234 Nm; actual
combined effort reaches the unchanged 0.76 Nm limit during the run. This variant
is rejected.

The next comparison keeps the frozen grasp position target when live allocation
succeeds. This retains the existing posture spring instead of replacing its
target with the current joint position. It isolates that change under the same
2 N nominal force floor, measured-reaction wrist controller and live allocation.
`live_contact_posture_mode` must now be explicit whenever live allocation is
requested: `current_position` reproduces the removed-spring comparison;
`frozen_reference` selects the new comparison. Old overlays remain reproducible.
No new carry success is claimed. All 70 CPU tests pass.

The measured grasp audit and preview now show the fraction of loaded steps with
live allocation active, fallback reasons and signed-force evidence. A passing
wrist gate or selected flag alone cannot demonstrate that a candidate ran.


## Completed renewed-budget validation

The same existing Brev L40S was started for each gated comparison, backed up and
stopped after each run. All three fresh free-space gates passed. Every complete
replay retains 343 source frames and 20 labelled inferred postroll frames.

| Trial | Exact source overlay | First-pot live activation | Full wrist gate | Carry |
| --- | --- | ---: | --- | --- |
| `coupled-task-live-v1` | v22 / `674910c` | 0 / 1830 | Pass | Failed; allocator never exercised |
| `coupled-task-live-v2` | v23 / `97d8e5b` | 70 / 1830 (3.83%) | Fail | Failed; posture spring removed |
| `coupled-task-live-v3` | v24 / `02c94bb` | 627 / 1830 (34.26%) | Fail | Failed; posture spring retained |

Both exercised variants have zero live activation in the second pot phase after
losing the object. The latest trial keeps all three pads loaded at source frame
120, but the middle pad falls to approximately 0.45 N at frame 122 and all three
contacts are lost by frame 124. Wrist position error is approximately 4.6 and
7.3 mm at those latter samples. The pot leaves the finite table and continues
falling; the complete metre-scale errors remain in the receipt and video. The
latest lid relative-position drifts remain approximately 6.4/6.8 mm, separately
from its source errors. No failed variant is promoted over the frozen baseline.

The allocator currently balances gravity only. Finite differences of the
recorded original reference before the first loss show linear acceleration up
to 4.42 m/s² at the already reduced source speed, corresponding to another
1.33 N at nominal 0.3 kg, with nonzero angular acceleration as well. This suggests
the next development target: replan and validate a grasp/reference with dynamic
wrench reserve, clearance and stable measured pad loading. The evidence does
not identify one exclusive cause. Increasing training steps cannot be assumed
to repair this unvalidated reference. PPO remains deferred until physical
reference validation passes.

`coupled-physics-campaign-v2.json` binds all seven trials to source, report,
video, state and grasp-spec hashes. `live-contact-failure-diagnosis-v1.json`
records the sampled loss and reference-demand evidence. The grasp audit now
reports actual live activation/fallback, even when the selected flag is true.

All 44,985 result/log files (13,029,303,475 bytes) are SHA-256 verified locally.
Brev `dngrtbldz` was confirmed **STOPPED** at
`2026-10-16T04:55:08.973340+00:00`. The conservative cost of these three sessions
is $0.6006 from the new $10 allowance; cumulative bound $10.3761 of the approved
$20, not an invoice. No new VM, policy, public input upload or image publication
was created in this validation.

### Exact local v24 reproduction of the failed posture trial

The private v3 archive is `dex3-pot-task-inputs-local-v3.tar.gz`, 131,936,310 bytes,
SHA-256 `7fc0c15ddfc77ce28e82698ede9b062ad385c39fa08b36f17124eb851fb53cd7`.
All members are individually verified by `local-pot-task-input-receipt-v3.json`.
It remains on the owner's Mac. It contains unchanged frozen input data and exact
historical source overlays; footage, geometry and inputs have not been uploaded.

Use the earlier pinned-image command recipe with these substitutions:

- Extract `sources/physics-source-v24.tar.gz` into a fresh `V2D_CODE_DIR`.
- Its exact source commit is `02c94bb4b49c1ed627c1411120588e9fd37bdfa2` and archive
  SHA-256 `1a1617041b5ddced08557e4cee74073a4f1ae6f89c65beea5aca717e51c78e50`.
- Select `combined-task-grasps-live-posture-v7.json` as `V2D_GRIP_SEED`.
- Run `grip-task-free`, then `grip-task` in the same new output directory.
- Expect a rejected full wrist/carry result; inspect the complete preview,
  activation audit and source-error measurements.
- Back up results and stop the compute instance after the run.

This is an exact diagnostic reproduction recipe, not a successful training or
submission release. The verified isolated-lift recipe remains unchanged.
