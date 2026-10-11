# Reproduce the improved e012 hand coverage

Use this profile when comparing the **same e012 video** with the improved
reconstruction. The older e012 v3 baseline has different hand inputs and therefore
lower coverage. Do not mix the two profiles.

The exact target is 181 frames, with 169 left-hand and 155 right-hand supported
frames. Acceptance requires the **same frame IDs for each hand**, not merely the
same totals. The source videos, calibration and reconstructed inputs are pinned
by SHA256. No missing frames are filled merely to pass this check.

This profile reproduces from frozen recovered hand inputs. It does **not** prove
that independently rerunning hand inference from only the video produces the same
result. The recovery experiment includes episode-specific development choices.
Fresh video-only inference and generalization to other episodes remain unverified.
Coverage acceptance is also separate from pose/contact quality acceptance.

## Release and access

At preparation on October 10, 2026, the code and new coverage archive are local
handoff artifacts, not yet published on GitHub or in the shared Drive. The older
shared issue6/7/8 packs do not contain this improved e012 profile. Repository
access alone does not supply the private videos, packs or licensed MANO models.
Obtain both archives through the project's authorized distribution channel.

| Archive | SHA256 |
|---|---|
| `e012-replay-kit-v3.tar.gz` | `02c5878ec9a3a7445466094314eaf329ef2a4d21f6843e5b056cd0381af87d38` |
| `e012-coverage-team-v1.tar.gz` | `9b66563274419ed3387761e7796a87c5b59537d436485afa22420f9b4e1cd96e` |

The coverage extension is 326,771,881 bytes and contains 3,373 catalogued files:
recovered inputs, reference reconstruction/motion and the helper snapshot.
It requires the v3 pack for the frozen optimizer, robot source and raw videos.
It does not include separately licensed MANO weights.

## 1. Restore and verify both packages

Use Python 3.11 or 3.12. Run from the repository root with new output folders:

```text
python scripts/reproduction/restore_pack.py --archive downloads/e012-replay-kit-v3.tar.gz --sha256 02c5878ec9a3a7445466094314eaf329ef2a4d21f6843e5b056cd0381af87d38 --out e012-base
python scripts/reproduction/restore_pack.py --archive downloads/e012-coverage-team-v1.tar.gz --sha256 9b66563274419ed3387761e7796a87c5b59537d436485afa22420f9b4e1cd96e --out e012-coverage
```

The restore checks the archive and every member before extracting. A mismatch is
a failure; do not update the expected hashes to match an unexplained download.

## 2. Verify the received reference

In a separate CPU environment, install `numpy==2.2.6`, then run:

```text
python scripts/reproduction/verify_coverage.py --reference e012-coverage/coverage-reference.json --inputs e012-coverage/inputs --raw e012-base/raw-stereo --bundle e012-coverage/reference-motion/bundle/episode.npz --retarget e012-coverage/reference-motion/retarget/retarget.npz --out received-reference-check.json
```

This verifies input identity and the supplied reference; it does not count as
recomputing the reconstruction. To verify your own source copy, replace `--raw`
with its dataset root, retaining the relative `meta/` and `videos/` layout.

## 3. Recompute reconstruction and Dex3 motion

Use the refinement environment and licensed MANO assets described in
[the environment guide](../../EPISODE_REPRODUCTION.md). The measured refinement
profile uses Python 3.11.17, PyTorch 2.9.1+cu128 and gsplat 1.4.0 on Linux/WSL.
A full clean GPU environment bootstrap on another person's machine is not yet
validated. The package records source/model hashes and runtime evidence.

```text
python e012-coverage/scripts/reproduce_refinement.py --kit e012-base --inputs e012-coverage/inputs --source e012-base/refinement-source --source-manifest e012-base/evidence/receipts/refinement-source-hashes.json --asset-manifest e012-base/evidence/receipts/mano-required-hashes.json --mano-assets MANO_ROOT --out my-e012-refinement
```

Switch to the IK environment from the same guide, then run:

```text
python e012-coverage/scripts/rebuild_motion.py --kit e012-base --inputs e012-coverage/inputs --refinement my-e012-refinement --out my-e012-motion
```

Always supply the coverage extension's inputs to both commands. The motion
helper refuses experimental refinement without its matching input lock. The
original baseline remains immutable. Use new output folders after a failed run.

## 4. Accept or reject your reproduction

In the CPU checking environment:

```text
python scripts/reproduction/verify_coverage.py --reference e012-coverage/coverage-reference.json --inputs e012-coverage/inputs --raw e012-base/raw-stereo --bundle my-e012-motion/bundle/episode.npz --retarget my-e012-motion/retarget/retarget.npz --out my-e012-coverage-check.json
python scripts/reproduction/verify_motion_match.py --reference e012-coverage/reference-motion --candidate my-e012-motion --out my-e012-motion-check.json
```

Both commands must return exit code zero. Coverage must match exactly. Numerical
comparison uses fixed limits declared before comparison: maximum 0.1 mm
Euclidean position error for hand joints/vertices and robot roots, and 0.001 rad
for robot root orientation and finger angles. These are reproduction tolerances,
not accuracy against real-world ground truth. A hardware-specific failure must
be investigated, not hidden by widening the tolerance or overwriting references.

Retain both reports, the refinement receipt, `my-e012-motion/rebuild.json`, all
stage logs, runtime versions, GPU/driver details and the code revision. Inspect
the overlay and `gate/quality.json` as well: matching a saved result does not
approve its hand/object alignment. This candidate still has unknown frames and
excluded object-overlap checks. No new candidate simulator replay is included.

## What has been tested

The extension was restored and all 3,373 files verified under Windows and Linux
in distinct locations. Frame-level checking matches the received reference.
Failure controls cover swapped supported frames despite equal counts, changed
videos, rewritten input locks, nonfinite supported joints and extra pose files.
Fresh refinement from the independently restored Linux package completed all
460 updates and 181 rendered frames, followed by fresh export and Dex3 IK. The
exact left/right support masks matched. Maximum differences from the original
improved result were 0.04493 mm for hand joints, 0.04522 mm for hand vertices,
0.04084 mm for robot root positions, 0.00005984 rad for robot root orientation
and 0.00002266 rad for finger angles. All were inside the declared limits.
See [coverage receipt](receipts/e012-coverage-repeat.json) and
[motion receipt](receipts/e012-coverage-motion-repeat.json).

Ten checker tests passed on Windows Python 3.12.14 and Linux Python 3.11.17,
both using NumPy 2.2.6. The Windows checker environment was created fresh.
The GPU rerun used the same local RTX 5060 Laptop GPU and existing refinement
environment, not a second physical machine. A run by another teammate on another
machine remains the final independent validation step. The added GitHub workflow
has not yet run remotely because this change is not published.
