# Dex3 baseline and recovery review

October 14 continuation: the [full supported task diagnostic](SUPPORTED_DEX3_TASK_DIAGNOSTIC_2026-10-14.md)
records the Sim6/Lab3 compatibility self-review, actual cooked decomposition,
complete physical failure and missing repeat/SDK closure. Human approval remains
outstanding; the historical review boundaries below are preserved.

The newer original-CAD development replay and its exact saved-fit reproduction
are documented in [the current CAD method](ORIGINAL_CAD_DEX3_REPLAY_2026-10-13.md).
The corrected replay measurements below describe the preserved earlier result.

The [support-aware development method](SUPPORT_AWARE_DEX3_REPLAY_2026-10-13.md)
now passes complete original-CAD/estimated-sheet and material/self/cross checks.
All 15 saved-fit stages reproduce byte exactly. The active change extends the
existing CAD reproducer, reuses its shared hash/extraction/runtime checks and
preserves its legacy command. No second support-reproduction entry point is
installed; versioned methods remain in immutable evidence archives. Changed
inputs, incorrect manifest hashes and escaping input paths are refused.
Its new Isaac run completes all 343 frames and passes independent A40/Windows
state, camera and video audits. All actual frames were inspected for artificial
on/off flicker. This technical self-review does not imply human approval;
source accuracy and physical motion-target acceptance remain withheld.

This refresh reads the exact GitHub diffs at the heads below and preserves the
separate review boundaries. It is a technical review, not a human approval,
merge or acceptance of reconstructed poses as training targets.

| Review | Head and base | Findings and verification |
|---|---|---|
| PR #35: source bundle, Dex3 retarget and replay | `ae8fa33f8f48a005cb22726ee96686fa30dda20b`, base `main` | Source valid/direct-observation masks are separately retained and checked; retarget skips unsupported hands. The baseline renderer preserves the source visibility contract. Model/asset locks, original triangle bounds and CPU controls remain present. CPU, reproduction, docs and pre-commit checks pass; GPU Build & Test completed as cancelled, not passed |
| PR #37: retained recovery consistency | `bc1a60132f30f7307ade59623d8e035ef0e20e4f`, base `recent-reproducibility-code` | Detector-bounded segmentation leaves the entire rendered mesh in precision checks. Export preserves fitted MANO scale, pose and measured translation. Display holds and false accuracy-acceptance flags are refused; every exported native pose is independently rematerialized. Windows/Linux 3.11/3.12 controls pass |
| PR #38: fitted development motion reproduction | Separate branch over PR #35 | Corrected explicit motion retains every original observation and grasp anchor. Fresh Linux regeneration matches all 23 arrays and both compressed motion hashes. Windows verifies the input package and independent simulator readback; stored full-path certificate checks agree on Windows/Linux |

The newer recovery branch does not replace the PR #35 source observations.
Its retained Dyn-HaMR adapter is measured on e002 and uses that retained
development track mapping. General upstream reconstruction, automatic mapping
for arbitrary retained runs and untouched-episode accuracy are separate gates.
The source and robot flags distinguish direct/supported observations,
display continuity and inferred robot poses; none becomes a validated target
because a rendered mesh is visible.

The original finite-screen counterexample is preserved. The corrected replay
now passes the declared continuous modeled-depth limit, with all 343 rendered
frames reviewed for artificial mesh flicker. See
[the corrected replay instructions and measured qualifications](CORRECTED_DEX3_REPLAY_2026-10-13.md).
Source contact boundaries, hidden-finger accuracy and object/support placement
remain unaccepted. No new unconditional physical-task or all-episode accuracy
claim is justified by these reviews. Human review remains outstanding.

The later [physical-acceptance audit](DEX3_PHYSICAL_ACCEPTANCE_AUDIT_2026-10-13.md)
now includes inspection of all 343 source comparison rows on 58 sheets and
audits each goal requirement. This expands source review without altering the
frozen motion or source observations. Technical review and reproducible replay
are complete; physical hand/contact/support acceptance is still unproven.
