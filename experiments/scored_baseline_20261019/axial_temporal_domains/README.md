# Correlated temporal domains and the index-source bottleneck

Six serial guarded CPU jobs are preserved as 19 exact private research
sources/plans and two repository dependency snapshots. Private source data,
geometry, model, reports and results are excluded. Production runtime and
tests are unchanged. No native position/physics/GPU/PPO work ran.

## Scope and results

1. Reuse 500 distinct saved 113/115 neighbor pairs without regenerating their
   banks or recomputing their joint boxes. There are 188 nominal empty boxes,
   36 numerical-boundary boxes and 276 nominal nonempty correlated boxes.
   Do not replace this union by the hull of its coordinate ranges. Local
   maximum root-motion checks add no rejection; global p95/outside motion is
   not assessed.
2. Continuous seven-joint source-image enclosures exclude 103 of the 276
   nominal boxes through the index anchor. All 128 new independent actual-URDF
   samples fit their enclosures; 173 boxes remain unresolved at that stage.
3. One source/kinematic-selected box yields two continuous axial-thumb
   sections and two middle sections, but no index section. Five new harmonic
   contract/closure controls pass. There is no whole-pose queue or native
   candidate. This is not a failed optimizer promoted to a source pose.
4. A new positive-distal 2R workspace/conic predicate accounts for the whole
   index joint rectangle, including corners, ellipse-center containment and
   all numerical circle-boundary roots. Four analytical controls pass; all
   173 domains are numerically empty. This is not a formal floating-point or
   free-root/global exclusion proof.
5. An independent method uses extrema of linear support on the four circular
   workspace boundaries, without re-solving quartics. Three controls pass;
   all 173 domains have saved separating planes. The minimum plane margin is
   **2.2907 mm**, with a conservative 1e-8 m binding guard. All 1,384 new
   actual-URDF checks fit the model to below 1.12e-16 m.
6. Resolve the remaining 103 interval-only and 36 numerical-boundary cases
   using necessary outer boxes with the **original** 1e-7 rad motion tolerance.
   Reuse the 173 certificates by hash without recomputing them. All 500 pairs
   are accounted for: 188 outer boxes remain empty; 312 have source separators
   (173 reused, 139 new). All 417 additional URDF checks pass. No original
   candidate position, source, motion or collision gate is changed.

The geometric exclusion uses an outward source allowance of 1e-4 px and
outward motion boxes only to make a **necessary bound** conservative. These
are not thresholds for accepting candidate commands. One journal Boolean key
mistypes `1e6` in its name; no numeric 1e6 threshold is used. The original
source-point tolerance remains 1e-6 px, and a private metadata correction
records this naming error without changing or repeating scientific results.

## What this establishes

With this particular frame-114 wrist orientation and the retained finite
113/115 neighbor controls, index source fit conflicts with the original joint
motion constraints. Thumb adjustment alone cannot resolve that conditional
conflict. This does **not** prove that continuously varying neighbors, other
allowed wrist orientations or the whole task are impossible.

The six previously verified isolated 114 poses remain isolated controls.
The verified three-component reference remains unchanged, with eight right-self
and five object failures. No complete reference, grip, motor/grasp, policy,
task or score improvement is qualified.

Next map the saved source-conic support witnesses to camera-cone halfspaces
and derive necessary common-wrist changes under the original source-model
orientation and world-component root bounds. Keep neighbor uncertainty and
source authority explicit before constructing a new trajectory.
