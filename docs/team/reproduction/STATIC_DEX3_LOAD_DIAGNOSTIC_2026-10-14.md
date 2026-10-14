# Dex3 static load diagnostic — October 14, 2026

**The test runs reproducibly; it does not demonstrate a successful manipulation task.**
Four automatically selected e002 grasp starts fail an unsupported, one-second
hold test under the stated assumptions. Both gravity/contact controls pass.
Two fresh runs on the same A40 produce byte-identical state files and videos.
All 44 imported initial link frames agree with the existing Pinocchio hand
model: maximum translation error 0.000770 mm and rotation error 4.426e-7 rad.
This checks simulator/model consistency, not recovered human finger accuracy.

This is a diagnostic mode of the existing
`scripts/reproduction/reproduce_frozen_dex3_replay.py`, not a reconstruction
pipeline, policy, training run or official challenge score. The original
343-frame replay, source observations, inferred motion, display holds and
all motion-target approval flags remain unchanged. Cosmetic visual mesh
overlap work remains deferred under the [scoring scope](SCORING_SCOPE_2026-10-13.md).

## Reproduce the test

Use the code tag `dex3-static-load-diagnostic-20261014`. Use the immutable
[support replay input release](https://github.com/Swaguto/NVIDIA-V2D-Challenge/releases/tag/dex3-support-replay-20261013)
for `CAD-support-dex3-simulator-inputs-v2.tar.gz`; its SHA256 is
`759b6e9fa642b47ec24306969688cc1ce15ead4cb0b80e4f8d086b727b93c680`.
Everyone with repository/release read access can obtain the same inputs.
The old release tag and nine assets have not been replaced.

```bash
git clone https://github.com/Swaguto/NVIDIA-V2D-Challenge.git
cd NVIDIA-V2D-Challenge
git checkout dex3-static-load-diagnostic-20261014
```

Follow the [existing runtime setup](SUPPORT_AWARE_DEX3_REPLAY_2026-10-13.md).
The measured runtime is Python 3.12.15, Isaac Sim 6.0.0.1, Isaac Lab commit
`28a37cecdd433c22d9eabd6a5954add9f13a8951`, with all 178 package pins checked.
The measured hardware is an NVIDIA A40 with driver 595.91.07. Physics uses CPU;
RTX rendering still requires the GPU. Windows supports input preparation, but
the measured simulator command requires Linux.

From the repository root, with the pinned environment's Python active:

```bash
python scripts/reproduction/reproduce_frozen_dex3_replay.py \
  --archive downloads/CAD-support-dex3-simulator-inputs-v2.tar.gz \
  --sha256 759b6e9fa642b47ec24306969688cc1ce15ead4cb0b80e4f8d086b727b93c680 \
  --out runs/static-load-first \
  --static-load-check \
  --support-plane docs/team/reproduction/dex3-motion-replay/support-inferred-support-plane-report-v1.json \
  --support-plane-sha256 38a6bf97df7def496bed7773e265c894e77da6bb5521d3077bf43330cbd3166f \
  --runtime-python "$VIRTUAL_ENV/bin/python" \
  --lab /absolute/path/to/IsaacLab3
```

For preparation only, omit the final two runtime arguments and add
`--check-only`; install NumPy, SciPy and Trimesh in that preparation environment.
Use a new output directory every time. The command verifies the frozen source
and method bytes, plane/bundle binding, observation masks and inferred/unapproved
target flags before running physics. An already extracted case can instead use
`--verified-case PATH --case-manifest-sha256 HASH`; every declared file is checked.

Repeat the command with `--out runs/static-load-repeat`. Inspect both
`simulator/report.json` files, their control results and `simulator/states.json`.
Do not accept a missing result, a `.failure.json`, or failed controls, even if
the SDK exits with status zero. No pass result approves motion targets.

## What the method does

1. Select the first closed-intent row of each existing inferred hand/object
   grasp phase. There are no episode-name, frame-number, crop or source-label
   overrides. For these frozen inputs the selected source frames are 47L,
   273L, 47R and 236R.
2. Preserve each selected frame's own object scale. Export portable meshes
   without changing vertices/faces, and reuse the existing rigid-object URDF
   generator. Windows/Linux output meshes and object URDFs match byte for byte;
   pose/preparation numeric differences are at most 1.78e-15. Source/report
   hashes containing platform line endings or absolute paths can differ;
   those are not physical state differences.
3. Reuse the frozen Dex3 importer and project object physics preset. Adapt the
   renamed Sim6 collision-import parameter to **Convex Decomposition**, and
   use the Lab3 PhysX configuration. After removing importer instances,
   reapply and read back zero contact/rest margins on every collider. Earlier
   attempts that omitted this step are superseded.
4. Fix the wrists, initialize fingers once, and apply bounded position drives:
   stiffness 2.0, damping 0.2, armature 0.00149; effort/velocity caps are the
   smaller of the existing G1 Dex3 constants and each URDF joint limit. The
   hand material uses friction 2.0, the lower bound of the project's startup
   range. This is not the randomized training environment.
5. Rotate the inferred support normal to stage +Z and enable gravity
   `(0, 0, -9.81)`. Initialize dynamic objects once; no object teleportation
   or external object forces occur during the load. Run 100 steps at 0.01 s.
   Independent free-fall and sphere-on-platform controls verify gravity and
   actual contact response.
6. Log the initial imported link poses, object trajectory, finger positions,
   object contact-sensor resultant and estimated implicit-actuator torques.
   The [contact sensor reports net normal contact force](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#isaaclab.sensors.ContactSensorData.net_forces_w), not a complete grasp
   wrench or measured hardware torque. Initial link readback is compared
   independently with the existing URDF/Pinocchio model in the committed fixture.

## Results and interpretation

| Automatic sample | Final object displacement | Maximum net normal contact force | Held within 10 mm for 1 s |
| --- | ---: | ---: | --- |
| Left, pot, first phase | 2.997 m | 258.49 N | No |
| Left, pot, second phase | 5.977 m | 457.60 N | No |
| Right, lid, first phase | 5.016 m | 3.190 N | No |
| Right, lid, second phase | 5.018 m | 3.284 N | No |

The unsupported control drops 4.954055 m. The supported sphere settles at
0.050000 m with mean vertical normal force 2.943000 N. Each run contains
404 grasp records and 202 control records. Both corrected runs have identical
numerical records and the same video SHA256
`7b80e43e2eb165cf39badea301855fefd822e264191a5145843b7e17a90b937a`.
This exact repeat is measured on the same worker; an independent teammate or
different GPU/driver has not yet been validated. Compare source bindings and
numerical states before comparing rendered bytes on another machine.

Mass 0.3 kg and diagonal inertia 0.001 kg m² are existing generator defaults,
not measured object properties. Gravity direction is inferred. Each grasp
fixture omits the table, other object and other hand; a first-touch pose may
still rely on those supports in the real recording. There is no squeeze
preload, learned residual, moving wrist or whole-body controller. Consequently,
these failures do not prove human-hand inaccuracy, failure throughout the
episode, a private evaluation failure, or an official success rate of zero.
They show that the two-pad geometric certificate alone is insufficient to
approve an unsupported physical load target.

The raw diagnostic video's t=0 image can retain the importer rest display
until the first physics/render update. Use initial link readback for starting
pose consistency; this video is not a source-camera accuracy overlay. Further
cosmetic rendering work is deferred. This limitation does not affect the logged
physical states or the independent initial FK check.

## Evidence for independent review

[Fixture registry](dex3-motion-replay/support-static-load-fixtures-v1.json),
[repeat receipt](dex3-motion-replay/support-static-load-repeat-v1.json),
[initial link verification](dex3-motion-replay/support-static-load-initial-FK-v1.json)
and the complete compressed measurements
`dex3-motion-replay/support-static-load-measurements-v1.json.gz`
are committed with the review code. The gzip contains both full reports/state
traces, initial FK comparison, Windows/Linux preparation receipt and wrong-plane
refusal. Verify their SHA256 values against the registry before comparing.
The complete 301-file local SDK evidence archive has SHA256
`54fa2378f1d380e0b3ec5dbc99777f8fc3b793ba7605ccc61a5fb937e29c2d9f`.

The owned worker is stopped. Physical task acceptance, independent human-finger
accuracy and successful source-aware object manipulation remain open. Issue #9
and training have not started; this diagnostic is not completion of the goal.
