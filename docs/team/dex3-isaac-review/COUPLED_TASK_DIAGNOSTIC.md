# Coupled e002 diagnostic: approach, grasp, transport and release

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
an explicit experimental mode. The default payload controller uses the nominal
0.3 kg gravity force and lever-arm moment already checked in isolated lifts,
with measured contact eligibility and filtered enable/disable. The current
comparison explicitly selects a 4 N planned minimum and nominal compensation,
without tactile target changes. The outcome is not yet accepted.

Controller choices are bound in the private proposal: `task_minimum_pad_normal_N`
(default 2), `wrist_payload_compensation` (`nominal_gravity`, or the experimental
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
