# Full supported e002 Dex3 task diagnostic — October 14, 2026

**The complete physical task fails. Physical acceptance and a clean repeat remain
open.** The measured v5 run completes 3,420 physics steps and 343 source frames;
gravity/contact controls pass, but floating wrists do not track their targets
and both objects leave the inferred support. This is a development diagnostic,
not a policy, successful manipulation demonstration or official Track 3 score.

The code continues the existing frozen replay wrapper. The original uncommitted
prototype was preserved before compatibility repairs. No reconstruction stage,
isolated grasp optimization, cosmetic hand work, Issue #9 or training was run.
The next score-relevant gate is stable physical wrist tracking and object/contact
motion over this complete task. No cause of the instability is yet established.

## Measured result

The A40 run used Linux Python 3.12.15, Isaac Sim 6.0.0.1, Isaac Lab commit
`28a37cecdd433c22d9eabd6a5954add9f13a8951`, driver 595.91.07 and all 178 checked
package pins. Physics ran on CPU; RTX rendering used the GPU. The frozen
`LockedReplayMotion` retains 3,956 knots, source time 0–17.1 s, simulation step
0.01 s and motion speed 0.5. The existing wrist controller has zero policy
residual, linear gains 50/10, angular gains 12/0.1 and force/torque caps 60/60.
Objects are initialized before the task and thereafter move through gravity
and contact; reference object poses are used only to measure errors.

| Readback | Left / pot | Right / lid |
| --- | ---: | ---: |
| Median wrist position error against inferred target | 0.834885 m | 0.864078 m |
| p90 wrist position error | 1.061729 m | 1.041459 m |
| Median wrist rotation error | 114.746° | 113.695° |
| Median observed object position error against inferred target | 93.731141 m | 29.866584 m |
| Final object position error | 1776.999939 m | 1763.322867 m |
| Maximum reported hand/object solver penetration | 12.805 mm | 0 mm (no reported contact) |

The very large late object errors describe objects falling/moving away from the
finite support, not reconstruction accuracy or an official metric. The controls
measure a 4.954055 m free-fall drop, a supported sphere at z=0.05 m and mean
vertical normal force 2.943 N. Raw records contain 6,842 hand rows, 6,842 object
rows and 202 control rows.

The independent frame-indexed source annotation describes four definite grasp
cores totaling 381 hand/frame samples. None has positive normal contact force
with its intended object at the sampled source frame. This does not rule out
transient contact between frames or outside those cores. Normal contact force
is not a frictional wrench or force-closure certificate. Solver separations
cover reported hand/object contacts, not complete geometric penetration or
support/self-intersection certification.

All recorded source observation, direct-observation and display-only flags
match the frozen inputs. All 686 input motion-target approvals remain false.
Display-only hand rows have physical collisions disabled. No organizer reference
pose is applied to the simulation or copied into its predictions. The inferred
support's existing 8–10 mm uncertainty and nominal 0.3 kg object properties
remain assumptions; no additional hand/support measurements are available.

## Actual collision decomposition

The SDK's synchronous `request_convex_collision_representation` returned actual
vertices, indices, polygon planes and mesh-to-body transforms for 18 colliders
in v3. Each object returns **32 cooked convex pieces** with `convexDecomposition`;
the 16 hand colliders each return one piece with `convexHull`. This establishes
actual decomposition, beyond an importer request or authored token. It does not
certify cavity fidelity or watertightness. The geometry readback is from v3,
separate from the full v5 task, using the same hash-locked object meshes/URDFs
and collision setup. Historical static results still describe their actual
convex-hull geometry; old releases and measurements are preserved.

## Evidence and review limits

Use the private
[diagnostic release](https://github.com/Swaguto/NVIDIA-V2D-Challenge/releases/tag/dex3-supported-task-diagnostic-20261014)
and code tag `dex3-supported-task-diagnostic-20261014`.

- `supported-task-v35.mp4`: actual 343-frame simulation video, SHA256
  `e20b57df37b098b4bd6f01c3cc10d0932db332197665c4c504bf27606091a927`.
- `source-vs-dynamic-task-failure.mp4`: original source beside actual simulation,
  prominently labeled failed task, SHA256
  `59738fa5ef2af14f1ca707624e1af7a1ced61c8293df62d345c947ca3baed42b`.
- [Measurement archive](dex3-motion-replay/supported-task-measurements-v35.tar.gz),
  3,686,016 bytes, SHA256
  `c9cad3911a1a29a35d1bfa1a2fab36a40a2e7e737b57040d4dcc4e89111232df`:
  original states, cooked collider readback, runtime check, independent audit,
  first startup failure log and unchanged motion evaluator.
- [Fixture registry](dex3-motion-replay/supported-task-fixtures-v35.json) binds
  every archive member and video. [Readback summary](dex3-motion-replay/supported-task-readback-summary-v35.json)
  contains the complete measurement summary and four source-contact intervals.

All 343 original source PNGs were hash-checked and the complete comparison video
decoded as 343 frames at 20 fps. Ten chronological source/simulation sample rows
were visually inspected; this is not a manual review of every comparison frame.

The full remote SDK report and all failed-attempt artifacts were not recovered
before shutdown. The exact remote v5 executor digest was not recovered. The
committed executor restores the logical v5 implementation after reverting a
later experimental timestep/startup change; its digest is in the fixture
registry. No byte-identical code closure or clean simulator repeat is claimed.
The later v6 attempt is unvalidated: its source is preserved locally, its remote
final result was not recovered, and it is not the published result.

The compatibility repairs address measured Sim6/Lab3 startup failures: activate
contact reporting on the actual rigid bodies, bind exact hand/object body paths,
and explicitly convert Lab3 proxies/Warp arrays to Torch at the scripted
controller boundary. Cooked geometry and contact separation readback are added.
The published physical failure remains unresolved; passing startup and controls
does not approve targets. Ruff, format, diff checks and the independent full
stored-state/source audit pass. No simulated mock test substitutes for that run.

This technical self-review preserves PR #35 against `main`, PR #37 against
`recent-reproducibility-code`, and PR #38 against `Evc-r/issue8-episode2-pilot`.
All three remain draft with zero human reviews at the October 14 check. No merge
or human approval has occurred. The previously reviewed #35/#37 heads are
unchanged; this continuation is published on #38.

## Reproduce inputs and execution

Obtain `CAD-support-dex3-simulator-inputs-v2.tar.gz` from the immutable
[support input release](https://github.com/Swaguto/NVIDIA-V2D-Challenge/releases/tag/dex3-support-replay-20261013).
Its SHA256 is
`759b6e9fa642b47ec24306969688cc1ce15ead4cb0b80e4f8d086b727b93c680`.
The existing [runtime guide](SUPPORT_AWARE_DEX3_REPLAY_2026-10-13.md#numerical-and-simulator-environments)
provides `setup_sim6.py`, the 178-pin lock and GPU/graphics prerequisites.
Use Linux Python 3.12.15 and a clean pinned Isaac Lab checkout. On the measured
container, missing native graphics libraries were supplied by `libxt6`,
`libglu1-mesa`, `libopengl0`, `libice6`, `libsm6` and `libvulkan1`.

From the repository root, first restore the small evidence archive. The existing
extractor checks the archive hash, every member hash and unsafe/duplicate paths:

```sh
git checkout dex3-supported-task-diagnostic-20261014
python -c "import sys; from pathlib import Path; sys.path.insert(0, 'scripts/reproduction'); from reproduce_frozen_dex3_replay import restore_methods; d=Path('docs/team/reproduction/dex3-motion-replay'); restore_methods(d/'supported-task-measurements-v35.tar.gz', d/'supported-task-fixtures-v35.json', Path('runs/supported-task-evidence'))"
```

Run the existing wrapper with a new output directory:

```sh
python scripts/reproduction/reproduce_frozen_dex3_replay.py \
  --archive downloads/CAD-support-dex3-simulator-inputs-v2.tar.gz \
  --sha256 759b6e9fa642b47ec24306969688cc1ce15ead4cb0b80e4f8d086b727b93c680 \
  --out runs/supported-task-first \
  --supported-task-check \
  --task-motion-evaluator runs/supported-task-evidence/motion-evaluator.py \
  --support-plane docs/team/reproduction/dex3-motion-replay/support-inferred-support-plane-report-v1.json \
  --support-plane-sha256 38a6bf97df7def496bed7773e265c894e77da6bb5521d3077bf43330cbd3166f \
  --runtime-python /data/v2d/runtime6/bin/python \
  --lab /data/v2d/IsaacLab3
```

For input-only preparation, omit the final two arguments and add `--check-only`.
An already verified case can instead use `--verified-case PATH` and
`--case-manifest-sha256 0954068550b897b5cae5465e24e8dc7edeae025223e5e4786068bffb22628fb2`.
Every declared case file is verified. Windows/Linux prepared object meshes,
URDFs, sheet and evaluator match byte exactly; platform numeric preparation
differences are at most 3.33e-16. The worker pack and handoff companion were
verified before execution (1,683 members total); no upstream inference or saved
fit stage was rerun.

Inspect `simulator/report.json`, `states.json`, `cooked-colliders.json` and video.
Require complete source coverage and passing controls, and reject missing
results or failure sidecars even when SDK shutdown returns zero. Those process
checks are necessary but do not establish physical task success.

## Independently audit the downloaded result

Download the actual `supported-task-v35.mp4` from the diagnostic release. The
case below is the already verified extracted case, not a new reconstruction.
The audit uses NumPy/SciPy; optional video composition additionally uses OpenCV,
Pillow and imageio-ffmpeg. The source annotation is read only after the state
and provenance checks, and is never a control/fitting input.

```sh
python scripts/reproduction/audit_supported_dex3_task.py \
  --case runs/supported-task-first/case \
  --states runs/supported-task-evidence/states.json \
  --states-sha256 8d170d0fb415c7491d2d4160f7775514b191d4200c0f449ed907f8e4b630c1f7 \
  --video downloads/supported-task-v35.mp4 \
  --video-sha256 e20b57df37b098b4bd6f01c3cc10d0932db332197665c4c504bf27606091a927 \
  --annotation-archive docs/team/reproduction/dex3-motion-replay/original-CAD-baseline-audits-v1.tar.gz \
  --out runs/supported-task-readback
```

Use `--source-color-dir /absolute/path/to/e002/color` to rebuild the comparison
from the original `000000.png`–`000342.png`; each is verified against the case's
`bundle/source/provenance/inputs-lock.json`. This camera comparison is qualitative,
not an absolute world-alignment or independent hidden-finger measurement.

## Compute closure and remaining gate

The owned A40 `tiwcn7kqpjt2zd` ran at $0.59/hour, under the existing $0.75/hour
cap. A local watchdog was armed before startup and renewed while running.
The renewed cutoff was 16:30:28 UTC, but its observed stop occurred at
17:09:27 UTC (13:09 Toronto), about 39 minutes late; the reason is unresolved.
Do not call that safeguard timely or imply zero cost. The usable states,
geometry, runtime check and videos were downloaded and hash-verified before
stopping; the missing SDK closure remains on the preserved worker volume.

The original worker's restart requests report no available host GPU; guarded
A40/A6000/A5000 and RTX 3090 Ti secure replacements report insufficient supply.
An L4 replacement `2zeqloztm0oilw` was allocated at $0.59/hour at 20:24 UTC with
a provider stop deadline and local watchdog. The user then requested publication
of current progress and a checkpoint pause. No package upload or experiment
started on that replacement; it was stopped, verified EXITED and removed at
20:26:48 UTC to avoid further compute/storage charges. This does not establish
an L4 experiment or a validated provider timeout. GPU-free recovery was blocked
by automatic approval review with the reason "blocked by policy"; no such
mutation executed. The other worker and shared network volume remain preserved.
The 20:27 UTC account check confirms both preserved pods EXITED, no running GPU,
balance $28.63695 and reported storage spend $0.077/hour. The original workers
and shared network volume remain preserved; stopped storage still has charges.

To finish, recover the outstanding SDK closure, correct the measured floating
wrist/object instability in the pinned environment, assess full-task source
contact/motion again, then freeze a credible implementation and perform one
clean repeat. Work is paused at the user's requested publication checkpoint;
no further GPU success or repeat is claimed. Human PR approval remains a
separate outstanding gate.
