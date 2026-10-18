# Scored video → Dex3 policy development setup

This is the current scored pipeline. The earlier `docker/dex3-review` setup
replays historical development episodes in Isaac Sim. The policy pipeline here
uses FlashCHORD/Newton/MuJoCo-Warp/JAX. A complete scored release and successful
physical grasp are still pending; see [current evidence](../../docs/team/dex3-isaac-review/SCORED_BASELINE_PROGRESS.md).

## Runtime and source pins

The measured host is Linux amd64, L40S 48 GB, driver 580.178.04. Host runners need
Python 3.10+ and Docker with NVIDIA Container Toolkit. The separate official CPU
packer environment uses Python 3.12 and the kit's `requirements.txt`. Keep model
and input storage outside the repository and outside Docker build contexts.

| Runtime | Private GHCR digest |
| --- | --- |
| Perception / calibrated neural hand fitting | `ghcr.io/swaguto/nvidia-v2d-challenge-dex3-review@sha256:6e2039d0a54e14a188a1798efe57c3465d002f063c782c827b791e4418d2dce6` |
| SAM3D own object reconstruction | `ghcr.io/swaguto/nvidia-v2d-challenge-dex3-review@sha256:dc9d0aedd2c6d50f53ce2d0677eca6f96e9631806d2bc12a7da4396629ce5357` |
| CPU IK / portable assets / PPO / native policy evaluation | `ghcr.io/swaguto/nvidia-v2d-challenge-dex3-review@sha256:bcde4a54c47743564d65f95725c4891e814077962b3f1b0a4a522ecb7d576922` |

Authenticate to GHCR using your own approved package-read identity and pull these
three digests. The current helpers are mounted from **committed code contexts**;
an older image alone does not contain the current fixes. Record the context
manifest's source commit alongside every image digest. The validated hand-prior
repair source is `848f408178e4d7e3b1bcbdcc1404a64325303307`; later input-acquisition
documentation does not change that measured source snapshot.

```sh
git rev-parse HEAD
python docker/track3-scored/make_context.py --out "$PRIVATE/perception-code"
python docker/track3-scored/make_policy_context.py --out "$PRIVATE/policy-code"
nvidia-smi
docker info
```

Both context builders reject uncommitted selected source. They exclude episode
footage, MANO, gated model weights, credentials and actual robot mesh files.

## Acquire reproducible video-only inputs

`scored-inputs.lock.json` is public upstream **metadata and hashes only**, pinned
to `nvidia/video_to_data_challenge@5f68335f3acc802033d1e80728c1633197521de8`.
The following mode selects exactly 60 scored camera videos and the shared camera
calibration. It excludes pose tables, provided CAD and unrelated public metadata.

Use a fresh `$INPUTS` directory that will contain only `dataset`, `raw` and JSON
receipts. Keep code, model files, diagnostic CAD and reference pose tables outside
this common mount. The runner checks the entire mount, including files outside
the `raw` view.

```sh
python scripts/track3/acquire_scored_inputs.py \
  --inventory docker/track3-scored/scored-inputs.lock.json \
  --roster "$KIT/data/track_3_evaluation_objects.json" \
  --root "$INPUTS/dataset" --receipt "$INPUTS/acquisition.json" --videos-only
python scripts/track3/prepare_scored_view.py \
  --root "$INPUTS/dataset" --receipt "$INPUTS/acquisition.json" \
  --out "$INPUTS/raw" --videos-only
```

All selected files are checked against their upstream content identities, even
when files already exist. The local input rehearsal verified all 61 files and
the common-mount contract without reading CAD or pose tables. It reused verified
source bytes; this was not a second internet download or a fresh policy replay.

## Private model and robot prerequisites

- `$MODELS/hand-runtime-v1` contains the separately licensed HaMeR/MANO files,
  model configs, detector and SAM2 weights. Its exact eight-file closure is
  `hand-assets.sha256`. From that directory, run `sha256sum --check` against the
  absolute path to that checksum file. Obtain weights under your own access or
  the team's authorized private storage; no weights are attached here.
- `$MODELS/sam3d` is prepared by
  `scripts/track3/download_sam3d_models.py --root "$MODELS/sam3d"`, after local
  `hf auth login` and approval of the gated model's access terms. Use
  `--verify-only` before running offline. The lock pins SAM3D, MoGe and DINO
  content. Credentials are not accepted as command arguments or copied to images.
- Keep the existing FoundationPose weight cache at `$MODELS/foundationpose`.
  Do not replace the compiled pinned runtime during a checkpoint comparison.
- Retrieve the actual robot asset closure rather than Git LFS pointer text.
  `robot-assets.lock.json` pins the 24 files used by this measured runtime. The
  policy runner verifies them before Docker. Robot assets remain a separate
  read-only mount.
- Obtain the official submission kit unchanged. Preserve its archive hash,
  evaluator and source revision in the experiment record. The kit used here has
  archive SHA256 `8d2bce64ecab6da4eee938fe2859bd60e767535af30068b386e2dc20a72cf43e`.

## Run stages and inspect failures

The configuration keys and pinned examples are in the
[Track 3 runner guide](../../scripts/track3/README.md). Use a fresh output
directory for every run.

```sh
python scripts/track3/run_scored_hands.py --config "$PRIVATE/hands-config.json"
python scripts/track3/run_scored_reconstruction.py --config "$PRIVATE/reconstruction-config.json"
python scripts/track3/run_scored_policy.py --config "$PRIVATE/policy-config.json"
```

For a fresh hand run, use `source_root: "$INPUTS"`, `data_subdir: "raw"`,
`split: "evaluation"`, and the official roster/kit. Its `observations`,
`temporal-hands` and `perception` directories supply the reconstruction config.
The reconstruction `bundles/eNNN` and `support/eNNN` directories supply policy
preparation. Use the committed `robot-assets.lock.json` with the actual asset
directory. The optional three-hypothesis object-track selector remains
experimental; freeze its choice before reserved-episode evaluation.

The generic baseline runs 100 PPO iterations: 153,600 environment steps with
64 worlds × 24 rollout steps, training seed 42. Preserve both the initial and
final checkpoint evaluations and the separate unassisted native readback.
Finishing a fit or a training process does not certify reconstruction accuracy,
grasp retention or leaderboard performance. Re-running PPO/physics need not be
bit-identical; compare pinned inputs, contracts, full coverage and repeated
metrics. The earlier cross-host IK comparison agreed within 1.24e-12 in robot
positions, with exact integer/confidence arrays. This is not full policy
reproduction or another teammate's reproduction.

## Assemble the complete roster

Create an explicit private catalog with schema
`v2d.scored-submission-catalog.v1` and an `episodes` list. Every row supplies
`episode`, `iteration`, `policy_root` and `bundle`. A policy root is the complete
output of `run_scored_policy.py`. All 20 official episodes and their exact
36 object slots are required. Checkpoint choice is explicit; the assembler does
not select by local score.

```sh
python scripts/track3/prepare_scored_submission.py --kit "$KIT" \
  --catalog "$PRIVATE/submission-catalog.json" --out "$PRIVATE/captain-review" \
  --code-commit "https://github.com/Swaguto/NVIDIA-V2D-Challenge/commit/<40-character-source-SHA>"
```

The assembler checks own-geometry provenance, command anchors, source frame
coverage, frozen-file hashes and checkpoint identity. It copies rollout and mesh
bytes unchanged, then runs the untouched official packer. No trajectory repair,
mesh recentering, disabled cross-check or upload is performed. A passing packer
result goes to the captain for approval; it does not establish a good score.

Run GPU work inside a separately bounded compute lease. Back up results with
hash verification, preserve failures and stop Brev when idle. The runners do not
provision or stop VMs themselves. Keep a provider-confirmed shutdown receipt.
