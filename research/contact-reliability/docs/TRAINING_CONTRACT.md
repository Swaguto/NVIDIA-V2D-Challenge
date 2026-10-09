# Training and evaluation boundary contract

This supplements the existing upstream motion schema. It does not change that
schema or claim that public object-pose tables contain human hands or robot motion.

## Required training bundle

Keep the original motion parquet plus robot and object assets. Record the episode,
source dataset revision, motion SHA256, asset SHA256s, robot task/config, source
frame, transform provenance, quaternion convention and units. Validate the native
motion file with the pinned upstream loader before simulator use.

Confidence uses the existing `reliability_v1` sidecar, with one additional required
field for the new launcher: `reference_sha256` (hash of the actual motion parquet).
Its timestamps must be motion sample times, not unconverted source capture times.
The sidecar retains raw signal validity and normalization provenance. Its arrays
have `[time, hand, object]` axes; order must exactly equal declared `hand_ids` and
`object_ids`. `left`/`right` are explicit. No silent reorder or interpolation.

Use development sequence IDs such as `episode_000002` for observation and normalizer
documents. The CLI `fit` converts the fixed split's integer episode IDs to
`episode_%06d`; do the same when calling `fit_normalizer` directly. Never include
the proxy IDs 0, 11, 23, 31, 39. All-missing confidence cells retain the existing
baseline-weight fallback and produce a warning; missing signals are not high-quality
observations. The sidecar's effective validity records the distinction.

## Trajectory audit/export envelope

`validate-trajectory` consumes JSON with the following required fields:

| Field | Meaning |
|---|---|
| schema_version | `track3_trajectory_v1` |
| episode_id | Nonnegative integer |
| units / quaternion_order | `m` / `wxyz`, explicitly converted by the producer |
| coordinate_frame | `episode_world`; camera poses must be transformed first |
| frame_provenance | Human-readable source of this episode-frame transformation |
| object_ids | Unique names in exact pose-array order |
| asset_sha256 | Map of each object name to actual asset-content hash |
| frame_index | Contiguous source frame indices |
| timestamps | Finite, strictly increasing source times in seconds |
| poses | `[T,B,7]`, position then unit quaternion |
| visible | Boolean `[T,B]`; invisible zero poses are never interpreted as rotations |

Candidate and reference must have identical episode, assets, objects, frames and
timestamps. A candidate cannot mark a required visible pose missing to evade errors.
No silent truncation, sorting, scale fitting, or quaternion normalization occurs.
Wrong quaternion convention cannot always be detected numerically; explicit producer
declarations and known-transform tests are required in addition to norm checks.
File hashes establish identity, not geometric correctness of an asset.

Public schema metadata stores sorted membership; per-row storage order is separately
defined. The real-data audit maps by object name and records the permutation.
Episodes 11 and 18 currently require nonidentity permutations.

## Time map

Supply `source_timestamps` (original capture times), `motion_timestamps` (resampled
zero-origin motion times), `source_time_at_motion` (one source time per motion sample),
`simulation_timestamps` (one simulation time per motion sample), and
`warmup_seconds`. Simulation time equals motion time plus warm-up. Source mapping
is monotone and bounded by source capture times. Repeated source times can describe
held samples. This map must come from the actual resampler/planner; nominal FPS
alone cannot reconstruct it. A rollout exporter must remove/identify warm-up and
resample to source frames before pairing with public GT.

## Result rows for compare-track3

Each planned episode × seed × variant needs an explicit row containing:
`episode_id`, `seed`, `variant`, actual `environment_steps`, `status` (`completed`
or `failed`), `rollout_complete` (actual boolean), `score_kind` (`proxy`),
`reference_sha256`, `physics_config_hash`, `evaluation_config_hash`, and `metrics`
(a dictionary such as `{"add_auc": 0.5}`). The number shown here is a format example,
not a measured result. Include failure explanation and runtime/compute fields as
additional columns when available; they are retained in `per_run` output.

Failure records are mandatory. Do not mark a command's exit code zero as successful
learning. Do not replace missing metrics with zero or average only surviving runs.
The comparator requires matching reference, physics, evaluation and actual step
budgets across variants for each episode/seed; a failed early run with fewer steps
must be rerun or separately reported, not presented as a matched comparison.

## GPU evidence still required

Record container digest, actual task and dependency versions, asset hashes,
training step count, samples/second, elapsed time, checkpoint reload, evaluation
settings, actual completion/termination, and aligned object trajectories. The
portable runner samples whole-device memory, which includes other users' processes.
It cannot establish attributable process peak VRAM or billed GPU hours.

Only the official evaluator's verified interface can establish submission artifacts.
Do not rename local proxy fields to official metrics without semantic verification.
