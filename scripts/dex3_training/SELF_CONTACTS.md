# Native hand self-contact audit

The hand/object/support audit does not assess collisions between links of the
same hand. Before physical validation, assess both scopes on the changed
reference. Keep the original 1 mm gate and native collision filters.

```bash
python scripts/dex3_training/audit_hand_self_contacts.py \
  --inputs /private/frozen-reference \
  --native-model /private/native-model.mjb \
  --native-model-sha256 <verified-model-sha256> \
  --out /private/hand-self-contacts.json
```

Use a model from the same pinned native runtime and geometry. The audit loads
the cached model without recompiling. It verifies frozen input hashes, explicit
joint/body bindings and independent URDF FK at every reference knot. It reports
both hands, including empty contact knots. Unknown or malformed body bindings,
gaps and clocks fail rather than silently removing failures.

`hand_self_contact_gate_passed` concerns native active collision geometry at
the reference knots. It does not certify measured forces, source accuracy,
continuous swept motion, motor feasibility, a grasp or task success. Assess
those separately. Never reinterpret an old hand/object certificate as covering
self-contact or a newly changed command sequence.

## e037 retained results

The original cached reference has self-penetration over 1 mm in 151 left and
46 right knots. A new derived left-thumb correction passes an independent
151-knot self/source/motion check: maximum left self-penetration is 0.50 mm,
all 604 original image-anchor errors are nonregressing, and actual joint limits
and original motion bounds pass. The human source data, authority flags, wrist
roots, right hand and non-thumb fingers are unchanged. Robot thumb angles are
derived retargets, not measured human joint angles.

There are also 450 sampled intermediate left-self states, maximum 0.501 mm;
these samples are not a continuous or full-task swept certificate. Right-hand
self-collisions and five hand/object penetration failures remain. No complete
physical task, policy or score improvement is qualified, and no PPO was run.

A subsequent right late-contact-mode component preserves that left fix and
reduces right failing knots from 46 to 13, with every right knot nonregressing.
Fresh verification checks all 1,072 available left/right image anchors and all
original motion bounds; 136 missing right anchors remain explicit and unchanged.
The first full right paths and bridge failures are retained. Five object
penetration failures persist, and 450 new complete-pose interpolation samples
also retain failures. This combined component is not a whole-task, continuous
swept, motor, grasp or policy certificate. No GPU/PPO ran.

Analytic source-ray branches and a finite temporal path subsequently clear the
first five right-hand failures. Fresh full-clock checks preserve both earlier
components, all 1,072 available image anchors and original motion limits. Right
self failures are now eight (110–117); all five object failures remain. Early
reference penetration reaches 0.9934 mm and sampled intermediate penetration
0.9910 mm, so nominal margins are narrow. No continuous swept, physical or
policy success is established. Fixed-root compatibility diagnostics remain
qualified by missing per-leaf evidence; audit source/root authority before a new
wrist hypothesis. Human observations, missing flags and held-out failures remain.

The subsequent authority audit confirms that wrist orientation is model-derived
and softly fitted during retargeting. Two new coupled wrist/finger poses clear
the remaining hard knots in isolation while preserving image/model-orientation
nonregression and original component bounds. Independent native/URDF checks
agree, but their sparse insertion violates actual finger speed and original
temporal bounds; two dense paths and a conditional construction also fail their
acceptance checks. No new reference replaces the three-component freeze. The
eight right-self and five object failures remain; no GPU/PPO or task success
is qualified. See the retained source-wrist authority archive and wrist report.


## 2026-10-20 — joint robot/object symbolic collision construction

Code-only archive: `experiments/scored_baseline_20261019/joint_robot_object_collision`. Private checkpoint: `compute/joint-collision-checkpoint-v1.json`; integrity chain: `compute/private-integrity-joint-collision-v1.json`. Production runtime/tests unchanged.

The 465 saved robot-angle variables and 1058 saved object coordinates now share an unassigned 1523-coordinate namespace. All 31 native bodies/24 joints/58 compiled geometries and all 249603 pair-clock instances are represented, including all right-self pairs. New articulated body/support/normal ancestry and native-object gauge maps replace invalid fixed-hand geometry; all original robot/source/object clock-motion-cone restrictions remain by hash. Continuous all-vertex unit-sphere support predicates are construction geometry, not numerical native contact, force or task certificates.

Eight serial guards finish with six successes and two retained failures/interruption. Eighteen new controls pass. Native/URDF metadata identity fails (origins up to 3.469e-9m; ranges up to 1.312e-7rad); both maps remain separate and unchanged, cause unproven, no tolerance repair. A redundant compressed-I/O verifier is stopped before its fixed resource limit; unfinished return stays unavailable. New one-load bindings verify every new coefficient routing and support/native-frame incidence without replaying source, FK/native, model or solver cohorts. A preparation SyntaxError is retained separately.

No optimizer, episode pose/source/NN/raster/native position/distance/physics/GPU/PPO evaluation occurred. Wrist/middle components remain unqualified hypotheses; all F repairs/missing commands remain fixed. Original 8 right-self/5 hand-object failures persist. Whole source/URDF/motion, actual training/fullpixel-heldout POST, native+self/swept/motor/grasp gates still precede training. Owner intent remains active and unachieved; historical blocked goal-tool observation stays separate.


## 2026-10-20 — executable joint constraint/support derivative interface

Code-only archive: `experiments/scored_baseline_20261019/joint_constraint_interface`; private checkpoint/integrity: `compute/joint-interface-checkpoint-v1.json` / `compute/private-integrity-joint-interface-v1.json`. Production runtime/tests unchanged.

The new executable interface keeps all 1523 unassigned coordinates, 151 clocks, 8114 robot rows/194 original URDF source factors/38 domains/fullregion2/unknown bands, 4551571 retained object rows and all cones. Independent bindings verify all 31839917 retained A coefficients and every original coefficient routing, RHS/P/q/conditioning/source route. All compiled support vertices, moving body/direction normals and correlated ties remain; mean SEARCH slopes are not universal directional derivatives. Actual six-motion/angular-p95 and all original training/fullpixel-heldout POST/native+self/swept/motor/grasp gates remain.

Six serial guards complete5success/1failure;19 new controls pass. A tuple-key metadata writer fails after loader invocation before a saved constructor receipt; raw object state/receipt remain UNAVAILABLE, never rerun/reconstructed. New typed return controls cover only new synthetic mappings/IEEE bits/sparse pointers and explicit zeros. Three API matmul warnings are retained with finite saved returns; cause/future reliability unproven. Nested Brev fallback bookkeeping is reconciled to the latest real receipt without changing raw receipts or observation timestamps.

No optimizer, episode pose/source/NN/raster/FK/native/physics/GPU/PPO evaluation occurs. Coordinates/directions remain unassigned; no physical/reference/trajectory/task/policy/AUC qualification. F, separate components and all original failures/gates remain unchanged. Owner intent remains active/unachieved; research completion is not goal completion.


## 2026-10-20 — bounded whole-joint candidate construction stops before update

Code-only archive: `experiments/scored_baseline_20261019/joint_primal_dual_construction`. Private checkpoint/integrity: `compute/joint-primal-dual-checkpoint-v1.json` / `compute/private-integrity-joint-primal-dual-v1.json`. Production runtime/tests unchanged.

One fixed nonlinear primal–dual construction declares two updates/terminal2, three complete whole-model queries maximum, and 150 seconds under the unchanged 240-second/2-GiB guard. All 1523 joint/object coordinates plus continuous geometric direction charts, 4810140 rows, all 1245 whole cones and all 249603 pair scopes remain. Ten method controls, one exact two-update control and one lossless pair-frame roundtrip pass; a wrong 17-field label (actual16) is retained as a failure, followed only by unreached independent pointer/field bindings.

The sole method finishes one whole151 query in 37.7493 seconds then stops BEFORE metrics, dual projection, adjoint or any update: 30 finite saved SEARCH coefficients exceed the declared numerical tie envelope by at most exactly one ULP (max5.551115123125783e-17 derivative algebra units). These are not candidate tolerances. No envelope repair, gate relaxation, restart, warmstart or intermediate/terminal candidate selection occurs. Saved-return audits bind all249603 pair instances/478078 command incidences/1623929 tied slopes without geometry/source/matrix/projector/method replay. The cause beyond the saved arithmetic scope remains unproven.

No new terminal reference or trajectory exists, and no actual training NN/raster/integer-cell/fullpixel-heldout POST/native+self/swept/motor-grasp/physics/GPU/PPO gate is claimed. F/separate components/missing136/heldout31-of-32/source authority/physical parameters and original8self+5object failures remain. Owner intent stays active/unachieved; historical blocked goal-tool observation is separate.


## 2026-10-20 — saved all-tie numerical envelope proof

Code-only archive: `experiments/scored_baseline_20261019/joint_saved_mean_proof`. Private checkpoint/integrity: `compute/joint-mean-proof-checkpoint-v1.json` / `compute/private-integrity-joint-mean-proof-v1.json`. Production runtime/tests unchanged.

Seven serial CPU guards succeed; eleven NEW concrete controls pass. Independent new proof maps bind every one of 933560 saved dynamic coefficient positions and all249603 pair scopes before the sole fixed certificate evaluation. Exact saved-binary mean/product expressions retain every correlated tie and current continuous-direction differential. A conditional finite-RN operation-count bound includes absolute subnormal allowances and final-addition error; a96-bit outward dyadic encoding encloses the resulting rational envelope. Raw SEARCH J/old E, all original4810140 rows/objective/conditioning/cones/strict flags remain unchanged. This is no global installed-binary, source-accuracy or native-contact identity.

The sole certificate covers all151 frames/933560 entries with zero failures (101.7850sec, observedpeak164134912bytes). A NEW independent IEEE-bit decoder and fixed2^-1074/2^-2148 integer accumulator audit verifies all3734240 exact terms (84.0167sec, observedpeak158089216bytes), without producer proof, old method, dot/mean, source, geometry, matrix/solver or native query replay. Unsaved numeric intermediates remain unavailable; exact ideal proof products are not their reconstruction. The max saved mean/product gap4.573361891211582e-16 is DERIVATIVE ALGEBRA, not a candidate tolerance. It differs from the retained30 old-envelope defects/maxoneULP5.551115123125783e-17. No old J/E or method check is repaired and the finished method is not restarted.

No new candidate, reference, trajectory, source/training/NN/raster/integercell/fullpixel-heldout POST/native+self/swept/motor-grasp/physics/GPU/PPO evaluation occurs. F/separate unqualified components/missing136/heldout31-of-32/source authority/physical parameters and original8self+5object failures remain. Whole original qualification gates still precede training. Owner continuation stays active/unachieved, separate from the historical blocked goal-tool observation.
