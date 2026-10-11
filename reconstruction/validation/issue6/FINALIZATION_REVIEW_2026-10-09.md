# Issue 6 finalization review — October 9, 2026

PR #32 cannot be recommended for merge under the accepted full scope. Its
software checks pass, but final episode quality acceptance remains **0/20**.
The original all-20 execution and visual-review records are historical baseline
evidence. They do not waive [QUALITY_ACCEPTANCE.md](QUALITY_ACCEPTANCE.md).

## Current quality blockers

The [October 8 checkpoint](quality-repair/PAUSED_CHECKPOINT_2026-10-08.md)
records episode 22 trial31's right pitcher grasp at 87.50% usable geometry and
73.96% strict raw contact, below the 90% and 80% thresholds. Its archive has not
completed independent local verification and its new videos have not completed
full visual review. Episode 12 finger geometry, episode 41, other geometry and
release reviews, production freeze and the five holdouts remain unfinished.

Every episode requires source holding/event annotations, current geometry and
contact evidence, exact media/configuration reviews, six-stage integrity checks
and verified private backup. Pending evidence cannot be promoted to pass.

## Software review repairs

- The source-view encoder pads only the encoded output to even dimensions for
  H.264; calibrated input images and reconstruction grids retain their dimensions.
- Contact artifact checks raise explicit exceptions and remain active under
  optimized Python. Strict and legacy bridge policies retain their semantics.
- Native fit validation checks finite arrays and exact fixed-shape preservation
  before issuing a fit receipt, including when Python optimization is enabled.
- The all-episode replay calls the checked-in progress collector; its inline
  prerequisite checks also remain active under optimized Python.
- CPU CI now covers hand alignment, validation helpers and their supporting
  Track 3 scripts, and installs the tools for real video-encoding controls.
- Team handoff status reflects the reopened gate. The historical episode 0
  image-change sample documents its confirmed last-frame clamp rather than
  inventing an additional reviewed frame. Historical aggregate review counts
  retain their original assessment scope.

The reviewed source before these follow-ups was `b7c884d`. Current CPU checks
include the actual stage/integrity contracts and real source-video encoding
with synthetic renderer output. These checks establish software behavior, not
new native fitting, episode quality, physical contact truth or robot training.

All 141 issue-6 controls passed on Linux with the real encoder available.
Windows passed 139, with its two encoder controls skipped. Both final branch
suites together passed **243 tests with no skips** in an isolated combined
source snapshot. Package lint/formatting, all six changed shell scripts and
workflow parsing pass.

Private test and compatibility receipts are in `runs/pr-finalization-review/`
relative to the parent workspace. Consult current GitHub Actions for the
published head's lint, formatting, Python 3.10/3.12 tests and CLI checks.
The revised PR description retains the full quality gate and leaves issue 6 open.
