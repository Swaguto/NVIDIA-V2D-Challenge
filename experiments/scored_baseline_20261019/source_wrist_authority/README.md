# Wrist authority and coupled source constraints

Code-only archive of eight serial, guarded CPU jobs. `source-manifest.json`
binds the exact 19 executed or declared source/guard/plan files; restore paths
are relative to the owner's private experiment directory. Source arrays,
records, model binaries, proposed commands and result reports remain private.
The scripts require those private inputs and the recorded repository runtime.
They are research producers, not a new production training pipeline.

## Retained outcomes

1. Exporter, bundle and retarget provenance: the right wrist orientation is a
   MANO model estimate passed to a soft retarget orientation cost. A frame's
   direct/metric flags do not establish measured robot orientation or joint
   angles. Source record and code hashes close to the frozen manifests.
2. A declared half-step toward the source orientation at 111/113 fails to
   provide all required analytic finger branches. No half-step seed qualifies.
3. One coupled native/source solve at each of 111/113 converges. Both returned
   poses pass isolated source-image/model-orientation nonregression, original
   component/URDF bounds and 0.5 mm native right-self penetration. There are no
   hand/object contacts and no trajectory or motor/grasp qualification.
4. The first dense 104:118 trajectory exhausts its 120-iteration cap. The
   returned proposal is rejected; a finite feasible-looking return does not
   replace optimizer convergence.
5. Independent numerical controls verify explicit SO(3) root/pixel derivatives.
   The old norm-error finite difference is sensitive near a nearly exact image
   anchor. This does not establish the sole cause of trajectory stagnation.
6. The equivalent squared-constraint/analytic-derivative formulation also
   exhausts 120 iterations and fails source nonregression. It remains rejected.
7. A distinct one-pass construction uses only the converged isolated seeds,
   with exact remaining scalar finger-motion viability intervals. It stops at
   frame 111: the conditional nonlinear solve fails source/convergence checks.
   Scalar interval feasibility does not imply future source/native feasibility.
8. Independent URDF/source/native checks confirm both isolated poses, with
   point closure below 2.7e-9 m. Sparse insertion fails original finger velocity
   and temporal limits (0.1731 rad adjacent and 0.3459 rad second-added changes
   versus 0.05 rad caps). It is rejected as a complete trajectory.

All eight worker processes completed under the existing CPU guard; several
candidate methods failed their scientific acceptance checks as recorded above.
No completed old source fit, native audit, optimization family or replay was
repeated. No physics step, GPU job or PPO ran. Native geometry, effort, source
authority, missing observations and gates are unchanged. The verified combined
left/early-right/late-right reference remains the current reference, with eight
right-self and five hand/object failures. No full task or score gain is claimed.

Next investigate temporal reachability of source-supported coupled poses before
another path optimizer. Preserve the unchanged component and motion bounds;
failure of these finite/local controls is not a global impossibility proof.
