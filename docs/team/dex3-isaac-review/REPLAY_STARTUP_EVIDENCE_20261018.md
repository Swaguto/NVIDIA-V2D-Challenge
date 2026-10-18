# Replay startup evidence: recorded differences remain unexplained

Status: CPU analysis and diagnostic preparation only. No replay, renderer,
training job or GPU session was run for this investigation.

## What the saved files establish

All three saved replay pairs (e003, e015 and e038) use matching checkpoint and
source-manifest hashes. At the **first saved sample**, raw motor targets,
applied motor position targets and motor velocity targets are each bit exact.
Reactive efforts, measured forces and physics states differ at that sample.

That sample is taken **after a complete control interval**. It cannot reveal
whether reset state, native solver caches, the first policy observation or the
first physics substep differed. Later reactive effort differences cannot by
themselves establish different initial actions.

The new CPU audit separates translation and orientation instead of treating
mixed joint/quaternion coordinates as distances:

| Episode | Maximum first-sample object position difference, all 8 worlds | Maximum first-sample object orientation difference |
| --- | ---: | ---: |
| e003 | 0.08555 mm | 0.005322 rad |
| e015 | 0.34366 mm | 0.002030 rad |
| e038 | 0.13810 mm | 0.007508 rad |

Previously reported larger generic coordinate differences also included joint
angles and quaternion components. They were not measurements of centimetres of
object translation. Later trajectories can diverge substantially; this audit
does not prove an amplification mechanism or explain the score variation.

The previously recorded native model arrays match. They are a saved subset,
not a complete capture of every GPU model, integration and solver state.
All input readbacks were hashed before and after this read-only audit.

Private result: `saved-replay-first-interval-audit-v1/report.json` under the scored
workspace. Raw saved trajectories remain private and unchanged.

## Prepared diagnostic changes

The optional passive startup trace now also records native actuator activation,
external spatial forces, mocap/equality inputs, force terms, solver counters,
action filter history and actuation gain/damping arrays where available. Root
action history and nested `action.buffers` history have separate names; absent
fields stay explicit. This is a diagnostic extension with tracing off by default.

[`compare_startup_traces.py`](../../../scripts/dex3_training/compare_startup_traces.py)
validates NPZ and per-field hashes, numeric types, shapes, finite flags and exact
lifecycle ordering before comparing traces. It reports missing required buffers
and different trace producers. Matching partial or nonfinite snapshots cannot
become a successful required-buffer comparison. A first recorded difference is
reported with its stage; **causality is always left unproven**. Even a complete
match of the required fields is not a complete model/solver-state capture or a
reproducibility certificate.

Eleven focused CPU tests pass, including passive copying, distinct action
histories, missing state, nonfinite state, altered evidence, stage mismatch,
unsigned differences and readback size limits. CUDA integration is unverified.

Use this only during a separately labelled, justified **new candidate** validation
once source, motion and the unchanged 1 mm native reference-contact gate pass.
Do not rerun the completed replay cohort or change the official evaluator. Keep
Brev unqueued during the owner-confirmed outage. Host readbacks synchronize the
GPU; no trace run should claim timing or score equivalence to an uninstrumented
run without evidence.
