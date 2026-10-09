# Add Track 3 contact reliability research toolkit

Target repository: Swaguto/NVIDIA-V2D-Challenge
Base: main
Head: codex/contact-reliability
Local commit: 14a8ce32e1f2de88785875e079557894811f01e6
Status: Prepared for review; head branch is not yet on GitHub. No PR created.

## Summary

Adds a self-contained research package under `research/contact_reliability` for
Track 3 CHORD baseline and contact-robustness work (related to #10 and #13).
It validates reference/rollout boundaries, configures controlled reward ablations,
and records reproducible experiments. It does not establish a trained policy,
complete either issue, or replace the team's scoring harness.

Before this addition, the standalone research tools were outside the team repo.
The package can now be reviewed together with the NVIDIA source, and its reward
parity tests discover that source directly within the monorepo.

## Implementation

- **Public data and splits:** bounded downloader for public pose tables/metadata,
  pinned dataset revision and content hashes; actual episode discovery; fixed
  proxy IDs 0, 11, 23, 31, 39; deterministic pilot IDs 2, 12, 13, 18.
- **Input boundaries:** explicit meters, wxyz quaternions, episode-world frame,
  asset identities, object names/order, visibility, contiguous frames, and matching
  timestamps/lengths. Invisible zero poses remain masked. Known rigid transforms
  rotate positions and orientations together. An explicit source/motion/simulation
  time map accounts for resampling and warm-up.
- **Confidence:** existing versioned sidecar format, bound to the motion file's
  SHA256 and exact time/hand/object axes. Visibility, mask agreement, reprojection
  and temporal consistency feed development-only normalization. Missing-signal
  validity is preserved; all-missing cells use the documented baseline-weight fallback.
- **Training integration:** opt-in in-memory configuration hook with revision
  and clean-checkout guards; no NVIDIA source rewriting. Four variants are
  `baseline`, `force_closure`, `reliability`, and `reliability_force_closure`.
- **Fallback semantics:** replace the reviewed ReconHand CWS term with upstream
  force closure while preserving its weight/schedule; disable missed/unintended
  CWS penalties and their schedules. Reject an already-active force-closure term
  to avoid double counting. The combination weights the fallback by confidence;
  it is not a learned or dynamic switching policy.
- **Experiments:** seeds 17/29/43, common explicit step budgets, development-only
  contact dropout/position jitter/normal rotation and separate trajectory corruption.
  Comparison requires all planned cells and matched actual steps and reference,
  physics, and evaluation identities. Failures remain explicit; missing metrics
  suppress aggregate means instead of averaging only survivors.
- **Records:** commands, working directory, revision/dirty state, configuration,
  input/artifact hashes, logs, runtime, exit status, missing artifacts, optional
  GPU device-memory sampling, and a rebuildable failure-preserving ledger.
  Required artifact paths must be fresh to prevent stale-checkpoint false positives.

## Documentation

All links below are relative to the repository:

- [Package setup and monorepo usage](research/contact_reliability/README.md)
- [Deliverables, runbook, and outstanding prerequisites](research/contact_reliability/docs/EVAN_DELIVERABLES.md)
- [Exact training, sidecar, trajectory, time-map, and result contracts](research/contact_reliability/docs/TRAINING_CONTRACT.md)
- [Evaluation-owner findings and proposed regressions](research/contact_reliability/docs/EVALUATION_HANDOFF.md)
- [Validation evidence and its limits](research/contact_reliability/docs/VALIDATION.md)
- [Dependencies](research/contact_reliability/docs/DEPENDENCIES.md)
- [Research report scaffold](research/contact_reliability/docs/REPORT.md)
- [Source attribution](research/contact_reliability/NOTICE)

Additional configs document source revisions, public metadata, the actual split,
and unfilled run/evaluator prerequisites. Unknown container, task, budget, and
evaluator values are deliberately not presented as measured facts.

## Validation

- **45 tests passed in the cloned monorepo**, with no skipped tests in that run.
  Coverage includes CPU reward parity against NVIDIA function bodies, confidence
  attenuation, variant configuration, frame/time/asset mismatch rejection,
  invisible poses, fixed-holdout leakage prevention, corruption reproducibility,
  matched comparisons, stale-artifact rejection, and telemetry labeling.
- `git diff --cached --check` passed before the local commit.
- In the original workspace, real public-data audit passed for **20 episodes and
  5,246 frames** at dataset revision `5f68335f3acc802033d1e80728c1633197521de8`.
  Download scope was 23 pose-table/metadata files totaling 550,001 bytes.
- Episodes 11 and 18 have row object ordering different from sorted schema metadata;
  the audit explicitly records a name-based permutation. The public tables had
  no invisible entries; synthetic tests exercise invisible zero poses.
- Fifteen inspected training/configuration/command/reward Git blobs matched the
  team's base revision and the pinned upstream source. This is static evidence only.

Reproduce the local suite from the repository root in a suitable CPU research environment:

```bash
cd research/contact_reliability
python -m pip install -e '.[test,training,motion]'
python -m pytest -q
```

Use the existing managed dependencies inside Isaac Lab; do not install the Windows
requirements lock over its environment. See the runbook for data-audit and launcher commands.

## Limitations and review focus

- No GPU training, checkpoint reload, recorded policy evaluation, three-seed GPU
  ablation, or official evaluator invocation has run. This PR supplies tooling
  and test evidence, not a benchmark improvement or submission artifact.
- Actual runtime access and a training-ready team reference bundle remain prerequisites.
- The launcher lock points to the reviewed base revision. This commit changes HEAD;
  revalidate and update the lock before training on the integrated branch. Do not
  bypass the guard or treat static source parity as complete runtime compatibility.
- The portable trajectory envelope supplements, rather than replaces, NVIDIA's
  native motion format. The native loader and simulator must validate real bundles.
- GPU sampling is whole-device usage including other processes; sampled peaks
  are not attributable process peaks, and runtime is not a billing measurement.
- Official `eval_e2e.py` availability and RPE semantics remain unresolved. Scores
  are explicitly proxy-only. The evaluation handoff records review findings;
  it does not modify PR #17's scorer or claim official metric equivalence.
- Downloaded data, generated outputs, models, virtual environments and private
  access credentials are excluded. Existing team training/scoring source is unchanged.

Review priority: reward/fallback semantics, reference and timing contracts, split
discipline, and reproducibility. Team issue ownership remains proposed; this PR
does not assign teammates or close #10/#13.

## Repository checklist

- [ ] Commit sign-off: the existing local commit has no Signed-off-by trailer;
  it has not been amended because the current instruction prohibits commits.
- [ ] Full pre-commit suite: not run; do not equate pytest with all repository hooks.
- [x] Tests added and updated; 45 package tests passed in the clone.
- [x] Implementation, contracts, validation and operational documentation included.
