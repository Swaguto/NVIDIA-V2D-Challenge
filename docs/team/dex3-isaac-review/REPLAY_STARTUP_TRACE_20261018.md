# Passive replay startup diagnostic

The completed three-case replay has different first recorded physics states
despite identical recorded native model arrays and first motor targets. The
cause remains unresolved. No completed replay or training cohort was rerun for
this change.

## Reset path verified in source

The pinned evaluator constructs the stock `RLEnv`, calls reset and capture,
constructs the PPO inference template (which may reset again), then explicitly
resets to frame zero. The training-only `NumericalFailureRLEnv` warm-start
cleanup is a separate class and is not used by this evaluator. Stock reset and
the native adapter do not explicitly clear `qacc_warmstart` in the inspected
path. This is a hypothesis to measure, not proof that warm starts caused the
observed variability.

## Optional diagnostic recording

`record_policy_readback.py --trace-startup` adds host copies at construction,
each existing reset, before/after the existing capture, and before/after the
first control interval. It records both Newton states, native solver position,
velocity, acceleration and warm-start buffers, counters, action/observation
buffers and native options. Missing buffers and nonfinite values remain
explicit. The pending first policy action is recorded before stepping.

This is disabled by default, restricted to 1–16 diagnostic worlds, and uses a
separate output directory. The optional wrapper inherits the same environment
methods and restores the original harness bindings on exit. It adds no reset,
physics step, warm-start clear or change to the official evaluator's source.
Host readbacks synchronize execution and may change timing; diagnostic scores
must remain separately labelled. Startup evidence is also saved if evaluation
raises an exception.

Outputs are `startup-state-trace.npz` and `startup-state-trace.json`, with per
buffer shape, dtype and SHA256 plus a hash of the complete archive. Preserve
them privately. Compare stages in order before considering a separately labelled
single-substep experiment. Do not infer causation from matching model arrays.

Two CPU tests verify independent snapshot storage, no buffer writes, missing
fields, nonfinite evidence and archive integrity. GPU lifecycle integration is
**unverified**. Existing images do not yet contain this new source; no new image
or GPU replay was built or queued. Brev remains stopped while its start/stop
control error is unresolved.
