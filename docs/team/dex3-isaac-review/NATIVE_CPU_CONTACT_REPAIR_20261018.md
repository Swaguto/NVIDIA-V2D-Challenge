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

## Cloud state

Three guarded Brev launch paths never reached RUNNING. A successful refresh of
the CLI login did not resolve the stop preflight's `UnrecognizedClientException:
The security token included in the request is invalid`. Starting paid work is
unsafe while its stop control fails, so a fresh start was not attempted.
Twelve subsequent checks
confirmed STOPPED. No GPU inference or new PPO steps ran in these attempts;
controller waiting time is not an assertion of billed GPU time. The owner has
the exact support message. Do not retry blindly until the cloud error is fixed.
