# Issue 6: hand reconstruction and stereo contact integration

## Team reproduction entry point — October 9, 2026

Use the [team reproduction guide](reproduction/README.md) and shared Drive pack
to recompute episode 2's frozen v3 contact outputs from real aligned observations.
It checks all contact, unknown and bridge decisions across the complete timeline.
PR32 is now merged; issue 6's stricter all-20 quality acceptance remains 0/20.
Fresh native processing follows the runbook below; older PR-status notes are historical.

The integration provides separate validation, hand, depth, alignment, contact
and report stages. The historical all-20 native runs, audits, reviewed videos
and private backups are preserved. Issue 6 was reopened after visible grasp
failures; those earlier visual passes do not establish the current acceptance
criteria. Final quality acceptance remains **0/20**, and PR #32 remains draft.

- [Setup, manifest contract and commands](../../reconstruction/modules/v2d_issue6/README.md)
- [Calibration/object/compute handoffs and reviewer handback](../../reconstruction/modules/v2d_issue6/HANDOFF.md)
- [Validation evidence and remaining gates](../../reconstruction/modules/v2d_issue6/VALIDATION.md)
- [Current all-20 quality criteria](../../reconstruction/validation/issue6/QUALITY_ACCEPTANCE.md)
- [October 9 finalization review](../../reconstruction/validation/issue6/FINALIZATION_REVIEW_2026-10-09.md)
- [Incomplete episode-2 manifest template](../../reconstruction/modules/v2d_issue6/examples/episode2.template.json)

The October 8 episode 22 trial31 still fails its right-hand visible grasp gates:
87.50% usable geometry and 73.96% strict raw contact, below the required 90% and
80%. Episode 12 finger geometry, episode 41, remaining event/geometry reviews,
production configuration freeze and holdouts remain unfinished. Missing or
ambiguous evidence stays unknown. Preserve holdouts 0/11/23/31/39 from adaptive
tuning. No independent metric contact accuracy, robot-training success or final
challenge improvement is established by these runs or CPU tests.
