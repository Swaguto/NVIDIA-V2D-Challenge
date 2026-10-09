# Evan's code comparison

This branch collects Evan's existing source, tests, configurations, and public
project documentation for comparison with main as fetched on October 9, 2026
(`0e182aa`). It is a review snapshot, not a new experiment or acceptance result.

| Work | Included version | Location |
| --- | --- | --- |
| Issue 5 repairs and diagnostics | `955f2f1`, with its prerequisite history | `reconstruction/modules/v2d_world_calib`, `scripts/eval`, `scripts/track3`, `tests`, `docs/team/issue5` |
| Issue 6 hand/contact validation | `3668729`, already merged into main | `reconstruction/modules/v2d_issue6`, `reconstruction/validation/issue6` |
| Issue 7 SLAM/refinement | `2d97b01`, already merged into main | `reconstruction/modules/v2d_gsplat_refinement`, `scripts/track3`, `docs/team/issue7` |
| Issue 8 episode 2 pilot | `43fc832` | `robotic_grounding`, `docs/team/issue8` |
| Local contact reliability toolkit | 52 files copied without changing their bytes | [`research/contact-reliability`](../../../research/contact-reliability/README.md) |

The original issue 6 branch commit `84016ca` has the same stable patch ID as
`048d37a` in main. Its later accepted updates remain in the main version.

Four issue 5/main merge conflicts were reconciled. The camera converter and
official diagnostic scorer retain main's explicit inferred-observation handling.
Pose registration retains the issue 5 repairs and main's `--mask-dir` support
alongside the hash-pinned `--mask-manifest` option. Track 3's current README remains
the entry point; the older notes are preserved as
[`POSE_PIPELINE_HISTORY.md`](../issue5/POSE_PIPELINE_HISTORY.md).

[`comparison-sources.json`](comparison-sources.json) records the exact source heads
and SHA-256 hashes of all copied toolkit files. Local agent instructions and
private workspace memory are excluded, as are Git-ignored environments,
datasets, licensed assets, model weights, caches, and run artifacts. Existing
branch histories are integrated rather than copying nested checkouts.

The source workstreams retain their documented evidence limits. Combining them
does not establish completed issue 5 tracking, all-20 issue 6 quality acceptance,
physical collision truth, or robot-training success.

The combined checkout passed 416 CPU tests on Windows, with nine platform/video
skips and two passing subtests. The copied contact reliability toolkit passed
45 additional CPU tests, including reward parity against the existing pinned
upstream source. All 52 copied toolkit files match their original bytes.
