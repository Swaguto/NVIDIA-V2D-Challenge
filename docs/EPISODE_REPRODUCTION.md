# Episode reproduction for repository users

This workflow is intended for anyone with access to the repository and required
data. e012 is the first complete pipeline validation target. Accepting an episode
ID is not evidence that every episode has passed. Training is a separate stage.

## Current validation scope (October 10, 2026)

| Stage | e012 evidence | Other episodes |
|---|---|---|
| Raw stereo | All 181 color/depth arrays exactly match preserved baseline; portable receipt comparison also passes | Generic runner completed e002's 343 frames; no reference comparison claimed for that run |
| Human-hand inference, camera motion, CAD tracking | Reused hash-frozen baseline | No fresh claim from this experiment |
| Joint refinement | Two full ten-epoch executions, including relocated-kit run; floating-point scene arrays differ slightly | Not rerun by this experiment |
| Dex3 IK | Independent rerun: all ten arrays exactly equal | Not rerun by this experiment |
| Overlap diagnostics | Restored-kit rerun: all nine arrays exactly equal | Not rerun by this experiment |
| Isaac Sim replay | Sim 6.0.0.1 on A40 rendered all 181 frames; independent root, joint and link-state audit passes at 1e-4 m/rad | Not validated by this experiment |

Historical runs elsewhere in this repository have their own evidence and scope.
The versioned private replay pack contains reference outputs, the source snapshot,
raw stereo inputs, refinement helpers and source/model manifests. Its archive
hash and restore results are reported separately alongside the delivery.
No successful full fresh raw-video-to-simulator reproduction is claimed here.

The independent refinement completed 460 updates and 181 overlay frames. Frame IDs
and missing-value masks match. The largest per-coordinate hand-joint difference
is 0.000025480985641479492 metres (about 0.0255 mm); this is a repeat-run measurement,
not a guaranteed cross-hardware bound. Appearance parameters have separate
differences recorded in the full comparison receipt. Fresh GPU optimization is
not bitwise reproducible in this test. Frozen-bundle IK/gate reproduction remains
exact; do not conflate those two results.

The fresh refinement was also re-exported and passed through Dex3 IK and overlap
diagnostics. All frame IDs, missing-hand masks, IK iteration counts and diagnostic
decisions match the first run. Hand-hand overlap depth differs by at most
0.0000022011667676138935 metres (about 0.0022 mm). Robot states have small numerical
differences documented per array; joint angles and translations have different
units and must not be combined into a single positional error claim.

## Raw stereo: select an episode

Use Python 3.11 with NumPy 1.26.4 and opencv-python 4.11.0.86.
The measured runtime reports OpenCV 4.11.0. Run from the repository root:

```text
python3.11 -m venv .venv-stereo
.venv-stereo/bin/python -m pip install -r scripts/reproduction/requirements-stereo.txt
```

Activate that environment before the commands below (on Windows use the
environment's `Scripts/python.exe`). A new Linux Python 3.11 virtual environment
with only these two installed packages reproduced all 181 e012 color/depth arrays
exactly. This is a fresh package environment on the same host, not a second
independent person's machine.

```text
python scripts/reproduction/rebuild_stereo.py --episode 12 --expected-frames 181 --data DATA_ROOT --out OUTPUT_ROOT
```

`DATA_ROOT` contains `meta/camera_calibration.json` and
`videos/chunk-000/observation.images.ego_cam_{b,c}/episode_000012.mp4`.
The calibration selects the stereo pair. `OUTPUT_ROOT` must not exist.
For another episode, change `--episode`; use `--chunk` for its dataset chunk.
Supply its documented frame count with `--expected-frames` to catch truncation.
Without that option the tool reads the full synchronized pair and rejects
unequal lengths, but cannot detect equally truncated input videos.

To reproduce a previously frozen run, additionally supply
`--reference-receipt PATH_TO_REFERENCE_JSON`. This checks relative input hashes,
the episode, decoded frame count, per-frame color/depth hashes and depth schema.
Reference receipts must come from a trusted, hash-pinned package. They are not
ground truth. A mismatch exits with failure and preserves the output evidence.

The output contains rectified color PNGs, floating-point depth NPYs in metres,
validity PNGs and `receipt.json`. Invalid depth remains NaN. The algorithm source
is frozen locally in `scripts/reproduction/stereo_depth.py`; the receipt hashes
that source. Changes to stereo settings require a distinct reference experiment.

## Refinement and motion commands

The reusable entry points are in `scripts/reproduction/`. Their paths do not
depend on a person's username. They consume an episode-specific frozen pack;
the original e012 pack is preliminary; use the versioned v3 pack for the source
and model manifests and Sim 6 adapter. The packaged replay entry point completed
an independent second 181-frame replay with matching state-audit errors.
Do not interpret these commands as an already published downloadable release.

In the refinement environment, with the reference pack and its source/model hash
manifests available, first run the verification-only preflight:

```text
python scripts/reproduction/reproduce_refinement.py --kit PACK --source REFINEMENT_SOURCE --source-manifest SOURCE_HASHES.json --asset-manifest MODEL_HASHES.json --mano-assets MANO_ROOT --out NEW_REFINEMENT --check-only
```

Remove `--check-only` to execute the frozen optimization. The measured e012
environment used Python 3.11.17, PyTorch 2.9.1+cu128, gsplat 1.4.0 and an RTX 5060
Laptop GPU. Full installed-package evidence and source hashes are recorded;
installation from a clean environment is still to be validated. MANO_ROOT must
contain the licensed `models/MANO_LEFT.pkl` and `models/MANO_RIGHT.pkl` matching
the reference hashes. The preflight does not require GPU initialization.

In the IK environment, export the newly refined scene and rebuild robot motion:

```text
python scripts/reproduction/rebuild_motion.py --kit PACK --refinement NEW_REFINEMENT --out NEW_MOTION
```

This performs bundle export, full-sequence Dex3 IK and overlap diagnostics in
order. It infers the episode from the matching input/refinement receipts and
rejects an episode mismatch. It preserves per-stage logs and stops on errors.
Use a new output directory every time. It does not train a policy or run Isaac
Sim. The verified simulator profile is Linux, Isaac Sim 6.0.0.1 and Isaac Lab
commit `28a37cecdd433c22d9eabd6a5954add9f13a8951` (3.0.0 beta2), on an A40 with
driver 595.91.07. Sim 5.1 failed on that driver and is not the reproduction profile.

For a quantitative comparison of any corresponding NPZ output pair:

```text
python scripts/reproduction/compare_arrays.py --reference REFERENCE.npz --candidate CANDIDATE.npz --out NEW_COMPARISON.json
```

The report requires matching keys, shapes and dtypes, reports exact equality and
missing-value masks, and measures floating-point differences. It reports evidence;
it does not silently assign a new acceptance tolerance. The frozen-bundle replay
helper has separate strict tolerances and decision checks.

## Access and assets

Repository access does not automatically grant access to the video dataset,
private reference packs or separately licensed MANO/model weights. Obtain those
through their existing authorized distribution channels. Do not put credentials
or private model weights into source control. Downstream IK/replay should require
the frozen reconstruction bundle and Dex3 assets, not the original inference
accounts or this chat history.

## How to report reproduction

Record the code revision, input/package hashes, OS, GPU driver, environment lock,
episode, commands and all generated receipts. Distinguish matching supplied
files, matching computed arrays, matching simulator states and fresh perception.
Keep missing observations missing and keep failed quality checks visible.
Reproducing a failure is reproducible execution, not successful reconstruction.

Exact checks apply to frame IDs, input hashes, validity masks and decisions.
Floating-point tolerances must be declared before comparison; pixel equality
across different rendering GPUs is not the simulator-motion criterion. Never
alter reference outputs or loosen thresholds just to obtain a pass.
