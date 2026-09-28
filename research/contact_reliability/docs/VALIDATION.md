# Local validation evidence

## September 25 implementation

The full local suite passes: **45 tests**, including CPU Torch parity against upstream rewards.
Real-data audit: 20 public parquet episodes, 5,246 frames; visible quaternion,
episode identity, timestamps, membership and frame-contiguity checks passed.
No invisible poses occurred in these public tables; synthetic tests exercise
invisible zero poses. Episodes 11 and 18 store objects in a different order from
sorted metadata; the audit reports an explicit name-based permutation.

Dataset revision: `5f68335f3acc802033d1e80728c1633197521de8`.
Download: 23 pose-table/metadata files, 550,001 bytes. No video, mesh or training
reference was downloaded. See `runs/public-audit.json`,
`data/track3/download_manifest.json`, and `configs/track3_split.json`.

The 15 Git blobs recorded in `configs/team_revision.json` match the pinned local
upstream source. Upstream checkout remains clean. This is static compatibility
evidence, not a simulator run. The GPU launcher, checkpoint reload, actual rollout
export, three-seed training matrix, and official evaluator remain unrun.

## Historical September 21 validation

Validated September 21, 2026, on Windows, Python 3.14, CPU PyTorch. Full local
dependency versions are recorded in `requirements-local-windows.txt`. That lock is
for these CPU checks, not for installing over Isaac Lab's managed environment.

## Results

- Project tests: **18 passed**.
- Pinned upstream `test_motion_schema.py`: **17 passed**.
- Synthetic extraction → normalization → scoring → four variants → alignment: passed.
- Pinned upstream source checkout: clean; LFS payloads deliberately not downloaded.

Project tests cover lower weights for corruption, missing-signal fallback, rejection
of holdout calibration, timestamp/identifier mismatches, deterministic ablations,
input/config cache identities, group-safe splits, paired-budget pilot decisions,
failed evaluator/artifact handling, reference-frame lookup after resets, contact
configuration replacement, and development-only perturbations.

CPU reward tests execute function bodies from the actual pinned source. They check
all-one parity for wrench support, missed contacts, unintended contacts, force closure,
and contact Chamfer; empty expected contacts; and attenuation before aggregation.
Isaac Lab is not mocked into a claimed simulation run: these are tensor-level checks.

Upstream motion-schema tests cover parquet round trips, required fields, dual-hand
alignment, schema version rejection, pose shapes and its quaternion convention guard.
They do not prove reconstructed geometry has correct metric scale in real scenes.

## Repeat

```powershell
.\.venv\Scripts\python -m pytest -q
$env:PYTHONPATH = (Resolve-Path upstream/video_to_data/robotic_grounding/source/robotic_grounding).Path
.\.venv\Scripts\python -m pytest -q upstream/video_to_data/robotic_grounding/tests/test_motion_schema.py
.\.venv\Scripts\python scripts/smoke_demo.py
```

Install `.[motion]` if reproducing upstream parquet tests in a fresh environment.
Upstream parity tests explicitly skip when the source checkout is absent; a skip
must not be reported as a verified upstream comparison.

## Not yet validated

Real reconstruction diagnostic export, simulator initialization, policy learning,
robot rollout performance, peak training GPU memory, official evaluator compatibility,
complete challenge task coverage, and leaderboard submission acceptance. These need
the outstanding data, assets and compatible machine described in DEPENDENCIES.md.
