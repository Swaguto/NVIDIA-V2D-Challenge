# Issue 7 review handoff

The accepted 343-frame episode 2 experiment is documented in [RESULTS.md](RESULTS.md)
and [REPRODUCE.md](REPRODUCE.md). The reviewed baseline is PR 34 commit
`6a2e24b0e09a8361d589be4d542e410a9d331e30`; the PR remains draft and unmerged.

This follow-up fixes all four reported blockers and both additional review
concerns:

- Published source-review paths use `runs/...` relative to the parent workspace.
  All 11 source-image hashes and every recorded metric remain unchanged.
- Reproduction names `slam-trial2`, the initialized SLAM run used by refinement
  `trial4`.
- Camera smoothness and acceleration use source frame indices ordered by the
  optimizer's frame-to-row map. Gaps contribute no cross-gap loss or gradient.
  Consecutive-frame behavior retains the original weighting.
- The launcher reserves a previously nonexistent `OUT` before executing any
  stage. Existing directories, dangling links and partial failures require a new
  location; cached inputs are never silently reused.
- Scoring exports require explicit observation provenance. Missing, null or
  unrecognized provenance is `unknown`; only explicit inferred labels produce
  `inferred`. The existing scorer excludes unknown rows.
- `.github/workflows/track3_refinement_cpu.yml` runs the complete expanded CPU
  suite on relevant pull requests, pushes to main, or manual dispatch, including
  documentation changes. It uses Python 3.11, CPU PyTorch 2.9.1, NumPy 1.26.4,
  SciPy 1.15.3, Pillow 11.3.0 and pytest 8.4.2, and preserves its JUnit report.
  No GPU, licensed MANO model or private footage is required.

On October 9, 2026, all **92 tests passed on Linux**, using the CI dependency pins
above. Windows passed **85 tests**, with the seven Bash launcher controls skipped.
The new controls execute the actual optimizer temporal objective on CPU and the
real launcher with a failing interpreter stub. Before the repairs, 15 new cases
failed. Both scoring-export states and absent input manifests are covered.
Receipts are `runs/issue7-validation/review-fixes-linux.xml` and
`review-fixes-windows.xml` relative to the parent workspace. Targeted Ruff,
Python syntax, Bash syntax and whitespace checks pass.

A 60-frame CPU comparison with the frozen PR source produces exactly equal
camera losses and gradients for contiguous frames. The accepted scene's 343
frame indices were verified contiguous. The independent audit in
`runs/issue7-validation/review-fixes-verification.json` also confirms that the
public results differ only in the 11 evidence locations. The CI workflow YAML,
event filters, test command, read-only permissions and result upload were checked.

These fixes and the workflow are included in this PR follow-up. Inspect GitHub
Actions on the current PR head for the published workflow result; the CPU
results above were measured locally. The frozen GPU experiment and private archives remain
historical evidence for their recorded source hashes. The current temporal fix
has not been rerun on the GPU, and these CPU checks establish no new refinement
quality result.

For review, inspect the camera gauge contract and missing-observation controls,
the repaired nonfinite-depth handling and temporal losses, and the separation
between measured improvements and source-label coverage. Preserve the stated
drift regressions, sparse appearance and upstream hand gaps. These results do
not establish learned robot performance or acceptance of every Issue 6 episode.

The episode 2 original and staged source locks were reverified for the Issue 8
handoff: 4,673/4,673 original and 2,961/2,961 staged files match. The scene SHA256
remains `3e3f51aa85c60f0d68384de4896e2f6c8ca0fc1cf3891d83d77c023676a9bc6a`.
The local Issue 8 checkout is separate from this branch.
