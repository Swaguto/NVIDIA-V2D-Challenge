# Evan: robot learning and contact robustness

Updated September 25, 2026. This is the local implementation and handoff index.
Proposed ownership: team issues #10 (CHORD baseline) and #13 (contact robustness).
Team assignment is not confirmed. No commits, pushes, PRs, messages, or submissions
are authorized by this document. The user's explicit instructions govern those actions.

## Implemented locally

- Fixed team proxy split: 0, 11, 23, 31, 39; development IDs discovered from files.
- Four matched variants and seeds 17/29/43, requiring an explicit step budget.
- Trajectory boundary checks for units, quaternion convention, visibility, asset
  identity, frame order, sequence length, timestamps, and coordinate provenance.
- Explicit source/motion/simulation time mapping with warm-up accounting.
- Sidecar reference SHA256 validation plus exact hand/object/time alignment.
- Rigid transforms rotate positions and orientations together.
- Baseline, force-closure, confidence, and confidence-plus-force-closure launcher modes.
- Command logs, manifests, input/artifact hashes, optional sampled GPU-device memory,
  and a rebuildable ledger retaining failures. Command completion is not learning success.
- Pinned download of public pose tables/metadata; videos and robot references are separate.

The audit/export envelope is not a substitute for the upstream motion schema.
Both the upstream motion loader and simulation must still accept every training bundle.
See [TRAINING_CONTRACT.md](TRAINING_CONTRACT.md) for exact fields and axis/time semantics.

## Current evidence and blockers

| Item | Evidence | Status |
|---|---|---|
| Team source | main b6dd0d9f9ddabf7f1955119d797d9ccfe8107104 | Read via Evc-r connection |
| Adapter compatibility | 15 inspected Git blobs match pinned upstream | Static evidence only |
| Local tests | Full suite passing; see VALIDATION.md | CPU, including reward parity |
| Public data | Download manifest in data/track3 | Pose tables + metadata only |
| L40S 48 GB and containers | Team issue #1 and access guide | Team-reported, no remote run here |
| SSH/Brev access | No local SSH config or supplied connection command | Required for GPU work |
| Training-ready team reference | Not present locally | Required for Track 3 policy training |
| Official evaluator | PR #17 describes a proxy harness | Location/version still unconfirmed |
| Trained checkpoint / rollout / official score | None produced | Do not claim completion |

## Commands

From this workspace, use `.venv/Scripts/python.exe` on Windows. On the GPU, use
the existing container's Python after installing this toolkit there. Do not use
the laptop Python installation as evidence of Isaac Lab compatibility.

```text
python scripts/fetch_track3_public.py
python -m v2d_reliability.cli track3-split --public-root data/track3/public --output configs/track3_split.json
python -m v2d_reliability.cli audit-public --public-root data/track3/public --output runs/public-audit.json
python -m v2d_reliability.cli track3-matrix --split configs/track3_split.json --steps MEASURED_COMMON_STEPS --phase pilot --output runs/pilot-matrix.json
python -m v2d_reliability.cli validate-trajectory --reference reference-audit.json --candidate rollout-audit.json --time-mapping time-map.json --output runs/boundary-check.json
python -m v2d_reliability.cli ledger --runs runs --output runs/ledger.json
python -m v2d_reliability.cli compare-track3 --results results.json --split configs/track3_split.json --phase pilot --metric add_auc --output runs/comparison.json
python -m pytest -q
```

`MEASURED_COMMON_STEPS` deliberately has no invented value. Profile the shipped
baseline, determine available shared-GPU time, and freeze one common budget.
Matrix generation plans runs; it does not launch jobs or prove results.
The observed development IDs are 2, 12, 13, 18, 21, 22, 24, 26, 27, 40, 41,
42, 43, 44, 45; the deterministic pilot starts with 2, 12, 13, 18.
`compare-track3` requires every planned seed/episode/variant cell, matched actual
step budgets and reference/physics/evaluation identities. Record failed runs
explicitly; missing metrics suppress the aggregate rather than averaging survivors.

For recorded execution:

```text
python -m v2d_reliability.cli run --config configs/baseline_run.json --output runs/UNIQUE_RUN --cwd ISOLATED_CHECKOUT --input REFERENCE_FILE --artifact CHECKPOINT_PATH -- python TRAINING_SCRIPT_AND_ARGUMENTS
```

Replace every uppercase argument with a verified path/value. `--artifact` is
repeatable and resolves relative to `--cwd`; required artifacts must be nonempty.
The run configuration records intended budget. Actual completed environment steps
must be taken from training telemetry, not copied from that intended budget.
GPU memory sampling is whole-device usage including teammates' processes, not a
per-process VRAM measurement. GPU billing hours are not inferred from wall time.

## Remote baseline handoff

1. Obtain the existing VM connection, availability window, and reference location.
   Do not start a new paid VM. Do not change the shared user's global Git identity.
2. Inspect GPU/process state and container image digest. Make an isolated checkout
   at the pinned team revision with separate outputs; never pull shared main during jobs.
3. Materialize required LFS assets in that isolated environment and hash actual
   asset bytes. A pointer file is not an asset. Record dataset and motion hashes.
4. Run the shipped example for one optimizer iteration and verify checkpoint reload.
   This is an infrastructure check, not a successful task.
5. Validate a team motion bundle through the upstream loader, inspect retargeted
   hand/body motion, and train the intended G1/Dex3 task. Preserve upstream staged
   training/resume configuration; do not evaluate the no-collision Stage 1 task.
6. Evaluate with training assistance disabled and an explicit frame/time mapping.
   Record termination, timeout, sequence coverage, video, and object trajectories.
7. Give the evaluation owners the manifest and aligned rollout. Use the official
   evaluator only after its identity, invocation, and required artifacts are verified.

Launcher inside the compatible environment:

```text
python scripts/train_reliability.py --upstream ISOLATED_TEAM_CHECKOUT --revision-lock configs/team_revision.json --variant baseline -- --task VERIFIED_G1_TASK --motion_file ABSOLUTE_MOTION_FILE --seed 17 --headless
```

Confidence modes also require `--sidecar ABSOLUTE_SIDECAR` before `--`.
The launcher injects an in-memory hook; it does not edit upstream source.
Use `--variant force_closure`, `reliability`, or `reliability_force_closure` only
after baseline validation. Static source matching does not verify container assets,
dependencies, physics, task suitability, or learning behavior.

## Experimental meaning

The baseline retains the shipped rewards. Force-closure mode replaces the reviewed
ReconHand CWS term with upstream force closure, retains its weight and curriculum,
and disables CWS missed/unintended penalties and their schedules. Confidence mode
weights the supported contact rewards. The combination applies confidence weighting
to the force-closure replacement. This is an explicit per-task ablation, not a
learned or dynamic contact switch. It refuses an already-active force-closure term
to avoid double counting. These configurations remain GPU-unvalidated.

Keep motion, robot, physics, seeds, step budget, initial states, and evaluation
settings matched. Fit confidence scales only on development IDs; freeze the floor
before proxy evaluation. Use 0/.25/.5 on development. Keep real reconstruction error,
position jitter, contact dropout, and timing shifts as separately named conditions.
Do not claim contact precision without labels or human-force recovery from wrench cones.

Report every seed/episode and failures. Compare against force closure, not just CWS.
Retain baseline for submission unless improvements repeat without unacceptable failures.
No paper benchmark number is a target guaranteed by the provided stereo cameras.

## Deliverable sequence

| Target | Deliverable | Acceptance evidence |
|---|---|---|
| Sep 25–28 | Contract, validation suite, source/data locks | Passing corruption checks and explicit unresolved fields |
| Sep 28–Oct 4 | Baseline package | Reloaded checkpoint, rollout, exact config, telemetry |
| Oct 5–18 | Contact ablations | Matched budgets, three seeds, per-episode metrics and failures |
| Oct 19–Nov 2 | Integration and report | Reproducible evaluation artifacts and claim/evidence table |

Allocate 10 h implementation, 7 h experiments, 4 h targeted study, 4 h handoffs per
week. Study coordinate frames/time, CHORD rewards/curriculum, then PPO and evaluation.
The dates are planning targets; blocked prerequisites do not become completed milestones.

## Sources

- Living team context: https://docs.google.com/document/d/1QhZwCoRwiCLJyEz5xhOKyFkx1qLhnBljTmOwKyjzvAM/edit
- Team work: https://github.com/Swaguto/NVIDIA-V2D-Challenge/issues/10 and /issues/13
- Proxy evaluator: https://github.com/Swaguto/NVIDIA-V2D-Challenge/pull/17
- Dataset: https://huggingface.co/datasets/nvidia/video_to_data_challenge/blob/main/track_3/README.md
- Method: https://nvidia-isaac.github.io/video_to_data/chord/
- Rules: https://nvidia-isaac.github.io/video_to_data/v2d_challenge/
