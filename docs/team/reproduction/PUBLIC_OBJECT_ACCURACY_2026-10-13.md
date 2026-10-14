# Independent public-object accuracy check for the Dex3 development replay

The organizer supplies public e002 object motion-capture poses. The earlier
physical audit's statement that no independently verified object reference was
available is superseded by this check. Independent hand/finger and support-height
measurements remain unavailable. The frozen Dex3 motion, Isaac video and release
are unchanged; physical task acceptance remains false.

The reference is NVIDIA's `video_to_data_challenge`, immutable revision
`5f68335f3acc802033d1e80728c1633197521de8`. The
[organizer README](https://huggingface.co/datasets/nvidia/video_to_data_challenge/blob/5f68335f3acc802033d1e80728c1633197521de8/track_3/README.md)
declares frame-exact video/parquet indexing, metre-valued scan-frame object poses,
w-first quaternions, visibility filtering and no camera-to-MoCap-world extrinsic.
These data and scanned meshes are attributed to NVIDIA under CC BY 4.0.
e002 is a development episode, outside this repository's five held-out proxy
episodes. No proxy episode was downloaded or used to select this method.

## What was measured

The existing `scripts/eval/score_candidate.py` now has `--mode relative`.
Its shared diagnostic module uses the existing `reference_loader.py`; it does
not add a motion generator or a second reconstruction/replay pipeline.
For each pair it compares `inverse(world_T_first) @ world_T_second` between
the reconstruction and public reference. A common rigid world transform
cancels. This measures relative scan-origin translation, relative orientation
and the difference in inter-origin distance. It does not measure absolute
world/camera placement, hand accuracy, table height or an official score.
Single-object episodes cannot provide this pairwise check.

Every file used by the check is covered by the
[immutable input lock](dex3-motion-replay/support-public-e002-object-reference-lock-v1.json).
The pot's 49,227 and lid's 49,738 unique float32 mesh vertices match the original
bundle exactly, establishing the shared scan frames. All 343 frame IDs match;
both reference objects are visible on all 343 frames. Only 285 frames have both
reconstruction object poses marked observed. The other frames remain reported
separately; inferred backfill does not become an observation.

The tiny affine rotation residuals are projected explicitly to proper rotations
and recorded, with a geometry displacement bound. Maximum bounds are below
0.0001 mm. Nonrigid input beyond the declared 1e-5 residual/scale limits is refused.
Known 15 mm translation, 10 degree rotation and common-world-transform controls
pass. The audit neither aligns the worlds to GT nor fits motion to it.

| Sample | Relative translation median / p90 / maximum | Relative orientation median / p90 / maximum |
| --- | --- | --- |
| All 343 visible reference frames | 6.598 / 13.063 / 62.936 mm | 3.746 / 6.470 / 8.253 degrees |
| Both reconstructed objects observed, 285 frames | 6.774 / 13.827 / 62.936 mm | 3.582 / 4.737 / 8.253 degrees |

Observed inter-scan-origin distance error is median 3.847 mm, p90 8.300 mm,
maximum 46.026 mm. These are descriptive measurements; no acceptance threshold
or competition score is inferred from them.

## Why the largest discrepancy is not yet a justified pose correction

The largest relative discrepancy is at source frame 161. In the raw public
parquet, both complete object poses at frames 158,159,160 are exact copies of
frame 157, including the quaternions, although capture time increases. Source
RGB frames 154–168 were inspected: images change and the visible pot moves
relative to the lid. The reference pot's position then moves 64.766 mm from
161 to 162, while the reconstruction changes 16.733 mm in that step.

This establishes repeated reference values and a disagreement with the source
sequence. Stale reference tracking or timing remains a hypothesis; exact
equality alone does not prove its cause or establish a corrected trajectory.
The source camera and depth also have uncertainty. Six all-object exact-copy
frames are detected automatically: 158,159,160,255,257,277. **None are removed,
interpolated or excluded from the table above.** No reference pose is copied into
the reconstruction to improve a metric.

The existing FoundationPose ESKF/RTS core was also tried, unchanged, with its
function defaults, uniform IoU weights and contiguous observed camera-frame
runs. It reduced the maximum relative error to 52.174 mm but worsened the
observed median from 6.774 to 12.751 mm and p90 from 13.827 to 37.686 mm.
It was rejected; no production pose changed. The existing world-pose velocity
failure detector's defaults also rejected sustained real manipulation (209/285
pot and 285/343 lid frames). It is not adopted as a general correction for this
multi-movement sequence. There was no GT-driven parameter search.

## Repeat the check

Use the PR #38 revision that includes this diagnostic. The immutable experiment
tag `dex3-support-replay-20261013` predates it and supplies the original frozen
motion/inputs. Reconstruct `15-case` using the
[support-aware replay guide](SUPPORT_AWARE_DEX3_REPLAY_2026-10-13.md), or use the
verified simulator archive's `case/bundle`. Their episode payload must be
`4de1b9507e42ed66e4341b93cc38563885fc2eda43db72935a546967671b6277`.

Create a separate CPU environment. The measured clean Linux runtime was Python
3.11.17, with exactly these five packages:

```sh
python3.11 -m venv /data/v2d/object-audit-runtime
/data/v2d/object-audit-runtime/bin/python -m pip install --no-deps \
  -r scripts/eval/relative_object_accuracy.requirements.txt
```

Run the existing evaluation entrypoint. Both output paths below must be fresh.
The command downloads only the four locked public files, verifies their byte
sizes and SHA256 values before publishing the reference directory, and then
records every visible pair and its observation status:

```sh
/data/v2d/object-audit-runtime/bin/python scripts/eval/score_candidate.py \
  --mode relative \
  --candidate /data/v2d/support-reconstruction/15-case/bundle \
  --data /data/v2d/new-public-object-reference \
  --fetch-reference \
  --reference-lock docs/team/reproduction/dex3-motion-replay/support-public-e002-object-reference-lock-v1.json \
  --json /data/v2d/new-public-object-report.json
```

For already restored, verified inputs, omit `--fetch-reference`. The episode
comes from the bundle manifest; this mode does not run the legacy proxy scorecard
or `--align-rigid`. An optional `--pose-proposal` accepts a separate NPZ containing
`object_world` and exactly matching `frame_indices` for read-only comparisons.
It never rewrites the bundle. Check candidate producer provenance separately;
the audit does not certify how an arbitrary proposal was generated.

The actual clean Linux execution used the fresh 15-stage reconstruction and
fresh HTTPS reference download. A separate Windows run used Python 3.14,
NumPy 2.5.3, SciPy 1.18.1, Trimesh 5.1.0 and PyArrow 25.0.1. All 343 records
agree numerically within an absolute 1e-12 tolerance in the reported metre/degree
units. Cross-platform JSON bytes and runtime-version fields are not expected
to match. The published
[canonical Linux report](dex3-motion-replay/support-public-e002-object-accuracy-v1.json)
and [repeat/refusal receipt](dex3-motion-replay/support-public-object-accuracy-repeat-v1.json)
give the measured hashes and differences.

Wrong reference bytes, an escaping reference path, wrong proposal frame order
and a nonrigid proposal were each refused with no result written. Source motion,
reference rows, observation masks and existing release assets remain unchanged.

To repeat the rejected smoother experiment, use the same clean CPU environment
and [diagnostic probe](dex3-motion-replay/support-existing-object-smoother-probe-v1.py).
It extracts the two original GitHub core functions verbatim; it implements no
replacement smoother and accepts no reference input:

```sh
/data/v2d/object-audit-runtime/bin/python \
  docs/team/reproduction/dex3-motion-replay/support-existing-object-smoother-probe-v1.py \
  --bundle /data/v2d/support-reconstruction/15-case/bundle \
  --smoother-source reconstruction/modules/v2d_foundation_pose/lib/run_ekf_smoothing.py \
  --out /data/v2d/new-rejected-smoother-proposal
/data/v2d/object-audit-runtime/bin/python scripts/eval/score_candidate.py \
  --mode relative \
  --candidate /data/v2d/support-reconstruction/15-case/bundle \
  --data /data/v2d/new-public-object-reference \
  --reference-lock docs/team/reproduction/dex3-motion-replay/support-public-e002-object-reference-lock-v1.json \
  --pose-proposal /data/v2d/new-rejected-smoother-proposal/object-pose-proposal.npz \
  --json /data/v2d/new-rejected-smoother-report.json
```

The existing smoother source SHA256 is
`b06fff42a2fa7835e5461ab6e77fa59507c67e2c8239e30fbf1df4f40453487d`.
The measured proposal payload SHA256 is
`b1622c3110d894d6eeb384ee62824e70a256b5d13e5c4a4e8a9dea28e9353b4f`.
The clean pinned runtime regenerated this payload exactly. Use the
[probe receipt](dex3-motion-replay/support-existing-object-smoother-proposal-v1.json)
for defaults, original-core AST hash and observation-run scope. This is evidence
for rejecting a proposed change, not a production motion stage.

## Remaining physical work

The next corrections must improve source-supported hand placement/contact timing
and clearance under the support-height uncertainty, while retaining reference
anomalies and occluded-hand uncertainty in the record. This object check narrows
the diagnosis but does not resolve the six unsupported contact boundaries,
hidden fingertips, estimated table extent/thickness or sensitivity to 8–10 mm
support shifts. Another identical Isaac replay would only repeat those inputs.
Issue #9, training, private-episode generalization and scoring claims remain
deferred until the development task is physically credible.
