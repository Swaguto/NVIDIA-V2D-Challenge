# Native CPU contact repair investigation — 18 October 2026

The full native reference-contact compiler now runs on the Mac CPU. This lets
the contact investigation continue while Brev's VM start/stop calls fail.
There is **no new trained policy or accepted physical task** in this experiment.

## Control validation

The pinned FlashCHORD source at `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`
was copied to a private local runtime; the existing four checked extension
patches were applied to that copy. Frozen e037 inputs and collision cache were
verified before and after the run. Versions were Newton 1.2.0, Warp 1.14.0,
MuJoCo/MuJoCo-Warp 3.8.1 and USD 25.11, using Python 3.12.13.

The unchanged native hand and object auditors reproduced all 151 source knots'
archived penetration values within **3.32e-16 m**, with identical frame/pair
classifications at the existing 1 mm threshold. Every native hand body passed
the existing FK checks. The warmed compile plus full audits took about five
seconds. [Control receipt](local-native-cpu-contact-control-20261018.json).

This validates static contact checks. It does not establish identical dynamics:
inertia buffers have small platform differences, and evaluation's large wrist
armatures are not present in this compile-only configuration. No physics was
stepped and no score equivalence is claimed. A preliminary flat-body surrogate
matched surfaces but differed in contact depth by up to 1.45 mm and had 14 false
penetrating classifications. That surrogate was rejected.

## Trajectory-wide wrist correction

The CPU prototype solves both hands' root translation corrections together over
all 151 knots. It holds fingers, wrist orientations, objects and observation
flags fixed. Bounds limit correction norm to 25 mm and each correction component
to 14 mm; added adjacent and second-difference components are limited to 1.5 mm.
Actual URDF finger position and velocity limits are checked. Native contact
normal derivatives were checked against finite differences before optimization.

- The first producer stopped before measurement on a sanitized native-name
  mismatch. The corrected producer with fixed endpoints was infeasible.
- Allowing the endpoints the same bounded inferred correction as other knots
  cleared **27 → 0** hand/frame slots above 1 mm, with maximum correction
  **11.025 mm**. Finger limits and velocities passed. This is a contact-stage
  result, not a grasp or force-closure certificate.
- An independent source check rejected that candidate: median projected errors
  improved slightly, but left/right p95 errors worsened from 24.74/35.47 px to
  25.27/38.35 px. These use fitted MANO wrist and three tip projections on
  directly observed frames, not hidden human pixel ground truth.
- A fourth variant added conservative per-point projection cones, represented
  by 16-sided inscribed polygons over 904 observed points. Its QP was infeasible.
  This rejects that restricted root-only optimization, not every possible
  articulated correction. Neither source nor contact acceptance was relaxed.

Receipts: [contact-only candidate](temporal-root-contact-candidate-20261018.json),
[independent source check](temporal-root-source-check-20261018.json),
[source-constrained variant](temporal-root-source-constrained-20261018.json).

**No candidate was promoted or trained.** The original object/support reference
still has 78 failing knots, so the full pre-training contact gate remains blocked.
The next coupled retarget should include finger articulation and source pose
uncertainty; source depth and object geometry also need repair. Exact executed
producers and failed controls are retained under the experiment archive.

## Articulated source-constrained repair

A new CPU candidate optimizes root translation, wrist rotation and all seven
finger angles together over the same 151 knots. It preserves the original
object/support geometry and poses, source observations, clock and confidence
flags. It changes inferred robot commands in a **separate frozen candidate**;
the selected baseline and its replay pack are preserved.

The first numerical approach failed. Arbitrary distance queries between
separated native meshes sometimes jumped from centimetres of separation to
zero after a one-micrometre perturbation. Across the three predetermined worst
knots, 7–24 separated pairs exhibited this issue. The ten actual native contact
derivatives were stable and matched their analytic Jacobians within 5e-12.
These measurements concern a CPU optimization query; they do not explain the
earlier GPU rollout variability.

The corrected optimizer uses conservative separating planes from native contact
normals and exact native convex vertices. Full surface checks retain the
original **+0.5 mm optimization clearance**. Worst-vertex cuts reduce the first
linear system from about 157,000 to 28,000 constraints. Source constraints use
the same conservative projection cones, with a small interior margin. Finger
position and velocity constraints come from the actual URDFs. Bounds also limit
added root/rotation/finger changes between adjacent knots and second differences.
All earlier failed controls are retained.

The three difficult samples passed independent URDF/FK, source and native
contact checks first. The full trajectory then converged in six CPU QP solves,
taking about 26.5 seconds. A fresh native compile of a separate hash-bound
candidate and the **unchanged original full native auditors** confirmed:

| Check | Result |
| --- | --- |
| Hand/frame slots above 1 mm | **27 → 0 / 302** |
| Maximum native hand/object or hand/support penetration | **0 m** at the 151 sampled knots |
| Independently checked source projections | 904 points; none worsened |
| Left projected median / p95 | 15.506 / 24.740 → 15.483 / 24.739 px |
| Right projected median / p95 | 15.037 / 35.466 → 14.902 / 35.262 px |
| Seven finger position/velocity checks per hand | All pass |
| Maximum root correction | 17.084 mm |
| Maximum wrist correction | 0.150 rad, within numerical tolerance |
| Added adjacent root component | At most 1.5 mm |
| Checked native geometry buffers | Match the original native static control |

The **complete task gate still rejects the candidate**: object/support
penetration remains at 78 knots, with maximum depth 13.664 mm. No physics was
stepped, contact force/force closure was not established, and swept contacts
were not assessed. No candidate was trained or substituted into the selected
baseline. These source projections use fitted MANO estimates, not hidden labels;
missing metric observations remain missing.

The next required repair is source-consistent object geometry and pose. A
seven-frame image-only stereo cycle check retained approximately 69–81% of
foreground estimates. At the problematic return frame 125, the fraction of
eroded foreground estimates below the local inferred table increased from
68.0% to 69.2% after the fixed one-pixel cycle filter. This filter does **not**
resolve the source/support inconsistency, so it was not promoted. It used the
unchanged saved control depths after verifying exact 1/16-pixel disparities and
validity masks. A learned-stereo comparison remains prepared but unexecuted.

The later [CPU learned-stereo investigation](FAST_STEREO_CPU_20261018.md) supplies
a new depth hypothesis and seven fitted spoon poses with zero native
object/support penetration. One hand/spoon sample still fails; the full new
trajectory is not qualified.

Receipts: [articulated stage](e037-articulated-contact-stage-20261018.json),
[stereo cycle diagnostic](e037-stereo-cycle-diagnostic-20261018.json).

## Replay variability investigation

The saved native model options leave warmstart enabled in all three replay
cases. Inspection of the pinned local runtime shows that the reset path clears
Newton coordinates/velocities, while the native adapter copies qpos/qvel and the
solver can reuse `qacc_warmstart`. The actual GPU startup/warmstart buffers were
not recorded. This is a hypothesis to test, **not an isolated cause**.

The next diagnostic should passively snapshot both Newton state buffers,
native qpos/qvel/qacc/warmstart/time/control, and reset/action counters before
and after initialization, capture and explicit reset, then immediately before
the first interval. The existing frame-zero readback is after that interval.
Only if those snapshots match should a separate labeled debug run inspect the
first solver substep. The untouched official evaluator must remain unchanged.
[Source-inspection receipt](replay-reset-source-inspection-20261018.json).

## Cloud state

Three guarded Brev launch paths never reached RUNNING. A successful refresh of
the CLI login did not resolve the stop preflight's `UnrecognizedClientException:
The security token included in the request is invalid`. Starting paid work is
unsafe while its stop control fails, so a fresh start was not attempted.
Twelve subsequent checks
confirmed STOPPED. No GPU inference or new PPO steps ran in these attempts;
controller waiting time is not an assertion of billed GPU time. The owner has
the exact support message. Do not retry blindly until the cloud error is fixed.
