# Prepared PR — awaiting Evan's affirmation

Title: `[reconstruction] Add staged hand and stereo contact integration for issue 6`

Base: `main` (`ca3f64a` at preparation). Local branch:
`Evc-r/issue6-hand-contacts`. The unrelated `evan/contact-reliability` branch is
preserved and its commit is not included. No commit, push or PR has been made.

## Summary

Issue #6 needs a way to consume evolving stereo/calibration and object-pose
outputs without repeating hand inference. Add an isolated stage runner with
explicit frame, calibration, metric-unit and provenance contracts, plus adapters
for the existing HaMeR and ViPE/Dyn-HaMR paths.

- Project calibrated stereo depth into the color camera; export aligned MANO
  joints and diagnostics with unknown observations preserved.
- Extract nearest-triangle fingertip proximity with separate raw and bridged
  annotations, surface points/normals, penetration and geometry ambiguity flags.
- Add immutable input/output receipts, comparison reports, overlays, CPU tests
  and CI, and producer/operator/reviewer handoffs.

Related to #6; **does not close #6**. This is the local integration milestone.
GPU adapters and real episodes are not yet validated. No model assets, footage,
generated results, paid compute, training or retargeting changes are included.

## Test plan

- [x] 53 CPU tests passed locally, including a synthetic pipeline integration
  test whose alignment worker is explicitly replaced with test coordinates.
- [x] Ruff lint/format, compile checks and installed CLI help passed.
- [x] Repository root-file hygiene and agent-context checks passed.
- [ ] Added Linux Python 3.10/3.12 CI must pass after publication; not yet run.
- [ ] Prepared GPU host: episode-2 64-frame HaMeR run, inspect original-frame
  mapping, MANO joint ordering, mesh/joint agreement and depth/contact overlays.
- [ ] Full episode 2, separate Dyn-HaMR comparison where available, and reviewed
  episodes 12/13 before declaring issue #6 complete.

## Handoffs and review focus

See `reconstruction/modules/v2d_issue6/HANDOFF.md` for the exact #4/#5 input
mapping, compute/asset handoff, reproducible first run, reviewer record and
downstream export semantics. Team documentation links it from
`docs/team/issue6-handoff.md`.

Calibration and object producers must supply matching frames, rectified camera
calibration, metric meshes, transforms, validity/quality reports and provenance.
The operator needs existing GPU images and authorized MANO/pretrained assets.
No team member has been notified or assigned by this change.

Review transform direction, left mirroring, unknown handling, occlusion support
and GT provenance first. `metric_valid` means adequate depth support, not proven
accuracy. Contact precision remains unknown without independent labels.

## Checklist

- [ ] All commits signed off per DCO — commit creation awaits affirmation.
- [x] Applicable local checks pass. No root/reconstruction pre-commit config is
  present; unrelated package hooks are not applicable to this contribution.
- [x] Tests added and local CPU pytest passes.
- [x] User-facing module documentation and team handoff entry added.

## Publication handback (local only; exclude this draft from the PR)

After explicit affirmation, recheck remote main and the scoped staged diff;
confirm Git identity, create a signed-off commit, push the dedicated branch and
open this PR against main. Attach the PR to this chat and inspect hosted CI.
Keep GPU gates visible and avoid a closing keyword for issue #6. Do not merge
or send separate team messages without authorization.
