# Dex3 baseline and recovery review

The newer original-CAD development replay and its exact saved-fit reproduction
are documented in [the current CAD method](ORIGINAL_CAD_DEX3_REPLAY_2026-10-13.md).
The corrected replay measurements below describe the preserved earlier result.

This refresh reads the exact GitHub diffs at the heads below and preserves the
separate review boundaries. It is a technical review, not a human approval,
merge or acceptance of reconstructed poses as training targets.

| Review | Head and base | Findings and verification |
|---|---|---|
| PR #35: source bundle, Dex3 retarget and replay | `ae8fa33f8f48a005cb22726ee96686fa30dda20b`, base `main` | Source valid/direct-observation masks are separately retained and checked; retarget skips unsupported hands. The baseline renderer preserves the source visibility contract. Model/asset locks, original triangle bounds and CPU controls remain present. CPU, reproduction, docs and pre-commit checks pass; Build & Test is queued |
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
