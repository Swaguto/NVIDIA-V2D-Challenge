# Execution runbook

## 1. Resolve external dependencies

Review DEPENDENCIES.md with the project lead. Confirm official task, permitted data,
evaluator, access rules, and compute. Use a compatible Linux host; the inspected
laptop is suitable for this toolkit's CPU checks, not a verified full V2D run.
Do not accept model/data licenses for other people or assume compute sponsorship.

Run `python scripts/bootstrap_upstream.py`. The source checkout is pinned and clean;
LFS files are pointers until separately fetched. Follow the **pinned** upstream
`reconstruction/docs/ego_e2e_setup.md` and `robotic_grounding/docs/SETUP.md` to provision
approved containers and assets. Use a short path on Windows; bootstrap enables long
paths in the clone's Git configuration. Baseline smoke tests need real simulator
execution, not only a replay video.

## 2. Profile baseline before expansion

Run one short and one representative permitted sequence through reconstruction,
retargeting, and physics-based training. Preserve baseline configs and checkpoints.
Wrap expensive commands with:

```text
v2d-research run --config configs/experiment.json --output runs/baseline_001 -- python <approved_entrypoint> <arguments>
```

Each output directory must be new. Manifests record command, source hashes, upstream
pin, config, input hashes, runtime, hardware, exit status, and available artifact
checksums. Process GPU peak memory is explicitly unmeasured by the portable runner;
record it from the approved simulator/profiler, with units and measurement scope.
Never substitute whole-device memory for process peak memory.

Estimate total GPU-hours from measured reconstruction cost per sequence and training
cost per paired run. Include the four variants and three seeds: a four-sequence pilot
has 48 runs; use one seed for screening if the approved budget requires it, then record
that decision. Reserve 20% for final evaluation/reruns. No paid jobs start merely
because a matrix has been generated.

## 3. Freeze sequence membership

Create `data/catalog.json` as an array of records with `sequence_id`, `group_id`,
`permitted_for_development`, `task_type`, `occlusion`, `hand_count`, and
`duration_seconds`. Use one group for connected shared recordings/objects. Permission
is a team-confirmed fact, not inferred by code.

```text
v2d-research split --catalog data/catalog.json --output data/split.json
v2d-research matrix --split data/split.json --steps <profiled_positive_integer> --output runs/pilot_matrix.json
```

The deterministic splitter searches grouped allocations with task/occlusion/hand-count
coverage, never splits leakage groups, and fails when it cannot reserve four holdout
plus four pilot sequences. Inspect duration coverage manually before freezing the
catalog and split. A generated matrix schedules nothing and contains no invented tasks.

## 4. Extract, calibrate, score

Export the diagnostic contract described in CONTRACTS.md on the **motion frame grid**.

```text
v2d-research extract --diagnostics data/dev_001.npz --metadata data/dev_001.metadata.json --output data/dev_001.observations.json
v2d-research fit --split data/split.json --observations data/dev_001.observations.json data/dev_002.observations.json --output data/normalizer.json
v2d-research score --observations data/dev_001.observations.json --normalizer data/normalizer.json --variant reliability --floor 0.25 --output data/dev_001.reliability.json
```

Fit on all selected development diagnostics, never holdout inputs. Repeat scoring
with baseline, constant, and shuffled variants. Keep original inputs immutable.
The synthetic demonstration exercises this logic but is not a real-data bridge test.

## 5. Train without modifying the baseline source

Inside the pinned Isaac Lab environment, install this package without replacing its
dependencies. From this workspace, after actual task and motion paths are confirmed:

```text
python scripts/train_reliability.py --upstream upstream/video_to_data --sidecar data/dev_001.reliability.json -- --headless --task <confirmed_whole_body_task> --motion_file <motion_partition> --logger tensorboard
```

Omit `--sidecar` for baseline. The launcher requires a clean checkout at the reviewed
commit and refuses changed injection points. First run a one-iteration smoke and
verify timestamp/body mapping. This adapter has CPU tensor tests; full Isaac Lab
integration still needs validation on the approved host. It is not appropriate to
claim successful training from the CPU tests.

Compare all variants using identical sequences, seeds, environment-step budgets,
upstream curricula, optimizer settings, and reconstruction caches. Tune only the
three permitted confidence floors. Freeze the selected configuration and hashes
before running the four local holdout sequences.

## 6. Pilot decision and reporting

Write result rows with `sequence_id`, `seed`, `variant`, `environment_steps`,
`completed` boolean and `metric` numeric (null is allowed for failed runs). Choose
one verified official metric and use it consistently for every row.

```text
v2d-research decide --results runs/results.json --split data/split.json --output runs/pilot_decision.json
```

Use `--lower-is-better` only for a lower-is-better metric. Missing pairs, duplicate
results, and unequal budgets fail validation. Advance when a majority of sequences
improve on mean paired differences and method failures do not increase. Keep per-seed
results and total runtime. An unsuccessful hypothesis remains a valid research finding.

## 7. Evaluate and submit

Populate the official submission configuration only after obtaining organizer instructions.
Verify its evaluator checksum, exact arguments, and required artifacts.

```text
v2d-research evaluate --config configs/submission.json --output runs/evaluation_001
```

Use a fresh output directory for the evaluator's artifacts as well. Have a second
member independently reproduce the evaluation. The team lead uploads through the
official channel, records the acceptance receipt, and archives the artifact hash.
This toolkit never uploads or marks leaderboard acceptance automatically.
