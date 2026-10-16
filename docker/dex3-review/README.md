# Brev containers for the continuity / Dex3 checkpoint

This setup preserves three separate runtimes:

| Image | Python / scope |
| --- | --- |
| `v2d-hand-continuity:checkpoint` | Python 3.11.10, native Torch 2.5.1 CUDA 12.4 closure; repaired hand fitting from the preserved native cache |
| `v2d-dex3-ik:checkpoint` | Python 3.12.15, CPU Torch 2.9.1 / Pin 3.7 / Pink 4.2; exact recorded IK dependencies |
| `v2d-dex3-review:checkpoint` | The IK environment plus a separate Sim 6.0.0.1 / Lab release 3.0.0 environment; renderer and media packaging |

Do not merge these NumPy/Torch environments. Base Python images are pinned by
OCI digest; Python distributions and the Lab source commit are locked. System
APT packages are resolved at build time. For an identical system closure, transfer
the built image archive or use the recorded image ID, rather than rebuilding an
APT layer later. The package installer checks all 178 simulator pins and the
tracked Lab source. These are workflow images, not the official challenge policy
runtime and not a complete fresh video-to-policy implementation.

## Use the published private images

The three images are published privately on GHCR, linked to this repository.
Their exact digests, Linux amd64 platform, source revision and privacy checks are
in [the registry receipt](../../docs/team/dex3-isaac-review/receipts/container/brev-5128f63/ghcr-images.json).
The release source is `5128f63e250ca5626e7b093ccb6d26cdfad304b6`; later documentation
commits do not imply a different image build. The first Brev pilot rendered
three complete development episodes (696 frames), and the final registry image
replayed e011 with four byte-identical videos. This does not validate the RL stack.

Team members need access to the private repository/packages. Authenticate with
`docker login ghcr.io` using your own GitHub identity and a package-read credential
([GitHub authentication guide](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-to-the-container-registry)).
Obtain and restore the separate private inputs as described in section 2. Then:

```bash
cp docker/dex3-review/registry.env.example docker/dex3-review/.env
# Edit absolute host paths; Brev providers may use ubuntu, shadeform, or another user.
# Read the NVIDIA license before enabling the two EULA values for a render.
docker compose --env-file docker/dex3-review/.env \
  -f docker/dex3-review/compose.yaml --profile native-cache pull
docker compose --env-file docker/dex3-review/.env \
  -f docker/dex3-review/compose.yaml run --rm review doctor --runtime all --gpu
python3 scripts/dex3_review/run_batch.py \
  --env-file docker/dex3-review/.env --episodes 11,40,41
```

`registry.env.example` pins registry digests for **pull/run**. For rebuilding from
source, start from `.env.example` with local image tags and follow section 3.
No episode video, private MANO model or credential is in a published image.

## 1. Brev host and code

Use an **x86_64 Linux GPU VM** with Docker/Compose and NVIDIA Container Toolkit.
Brev documents these as preinstalled. The rendering GPU needs RT cores; NVIDIA
lists A100/H100 as unsupported for Isaac rendering. Allow at least 32 GB host RAM,
16 GB VRAM and ample disk space (100 GB free is a practical starting allowance
for building three images and keeping caches; this is not a measured minimum).
The source replay worked on A40 / driver 595.91.07. CUDA 13 is included in the
Sim closure: verify the chosen host with the doctor and first real render.

- [Brev custom containers](https://docs.nvidia.com/brev/guides/development-tools/custom-containers)
- [Isaac Sim requirements and driver table](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html)

```bash
git clone --branch codex/continuity-dex3-brev https://github.com/Swaguto/NVIDIA-V2D-Challenge.git
cd NVIDIA-V2D-Challenge
git rev-parse HEAD
nvidia-smi
docker compose version
```

No instance is provisioned by these files. Registry credentials, source footage,
MANO weights and private inputs are not baked into the images. Dockerfile-specific
ignore files allow only needed source into each build context. Default Compose
exposes no ports, mounts input data read-only and keeps outputs separate.

## 2. Obtain and restore the private replay inputs

The archive `dex3-review-private-inputs-v1.tar.gz` is preserved in the producing
pod at `/workspace/hand-continuity-repair/dex3-isaac-review/` and in the producing
Mac's private task directory. Its exact SHA256, size and episode roster are in
[`private-input-pack.json`](../../docs/team/dex3-isaac-review/private-input-pack.json).
Transfer it to the authorized Brev VM with SCP/rsync. Do not put it in Git or an
image registry. It contains repaired geometry, camera poses, original development
CAD, frozen IK, exact robot mesh bytes and comparison input videos. It excludes
MANO model weights and authentication material.

Run from the repo root, with the downloaded archive in `downloads/`:

```bash
PACK_SHA=$(python3 -c 'import json; print(json.load(open("docs/team/dex3-isaac-review/private-input-pack.json"))["sha256"])')
python3 scripts/dex3_review/restore_pack.py \
  --archive downloads/dex3-review-private-inputs-v1.tar.gz \
  --sha256 "$PACK_SHA" --out "$HOME/v2d-data/dex3-review"
mkdir -p "$HOME/v2d-outputs/review-001"
cp docker/dex3-review/.env.example docker/dex3-review/.env
```

Edit `.env` to use **absolute host paths** for `V2D_DATA_DIR` and
`V2D_OUTPUT_DIR`. Set `V2D_CODE_COMMIT` to the result of `git rev-parse HEAD`.
Read the [Isaac Sim license](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/common/licenses.html)
before setting `ACCEPT_EULA=Y` and `OMNI_KIT_ACCEPT_EULA=YES` for rendering.
The restore command verifies the archive, complete inventory and every member
before extraction; it rejects links, path escapes and existing output directories.

## 3. Build and check

```bash
docker compose --env-file docker/dex3-review/.env \
  -f docker/dex3-review/compose.yaml build ik review
docker compose --env-file docker/dex3-review/.env \
  -f docker/dex3-review/compose.yaml run --rm ik doctor --runtime ik
docker compose --env-file docker/dex3-review/.env \
  -f docker/dex3-review/compose.yaml run --rm review doctor --runtime all --gpu
```

The review image copies the already installed simulator prefix into payload layers
of at most 8 GiB, then verifies every installed file's bytes/mode and each symlink
against the builder. This stays below GHCR's 10 GB layer limit without changing
runtime package versions. The builder cache itself is not published.

The image build needs network access for package registries and the pinned public
Isaac Lab checkout. Cold extension/shader startup can take several minutes.
Version/import checks and `nvidia-smi` do not establish RTX rendering success.
The first complete render and its simulator receipt are the GPU acceptance gate.
macOS can build the Linux images with emulation; it cannot execute their NVIDIA
GPU render. See [verification status](../../docs/team/dex3-isaac-review/CONTAINER_VERIFICATION.md).

## 4. Reproduce the previews

Start with e011, e040 and e041. Use a fresh host output directory per batch.
The default uses the exact recorded IK targets, renders the full sequence and
composes/verifies four videos. This isolates the renderer/runtime migration from
a new optimization run.
There is no hard-coded cloud host, SSH key, expired deadline or pilot approval file.

```bash
python3 scripts/dex3_review/run_batch.py \
  --env-file docker/dex3-review/.env --episodes 11,40,41
# All twelve, after choosing another fresh V2D_OUTPUT_DIR in .env:
python3 scripts/dex3_review/run_batch.py \
  --env-file docker/dex3-review/.env --episodes 2,11,12,13,18,21,23,24,31,40,41,42
```

`--reuse-frozen-ik` explicitly selects the default. `--recompute-ik` reruns the
existing solver with 100 maximum iterations and checks its full frame axis,
unknown mask and qpos against frozen IK (fixed absolute tolerance 1e-6).
The first Mac-emulated Linux reruns of e011/e021 failed this strict numerical
comparison; joint limits and masks passed. On the native x86 Brev L40S host,
both episodes passed at the unchanged 1e-6 tolerance (maximum differences
4.22e-14 and 1.77e-12 respectively). The Mac failures remain in the receipts.
The strict comparison still runs for every requested fresh IK episode; two passes
do not establish numerical identity for every episode or host.
Fresh retarget manifest bytes include timing/runtime details; a new render binds
to the new manifest. Video bytes may differ across GPU/driver/FFmpeg versions;
compare frame axes, masks, numerical receipts and sampled/full playback, not only
video SHA256. Original checksums identify the historical media and do not promise
cross-GPU pixel identity.

Results are under `V2D_OUTPUT_DIR/eNNN/{retarget,sim,review}`. The review contains
`comparison.mp4`, `isaac-source.mp4`, `isaac-oblique.mp4`, `human-input.mp4`,
eight sample images, timestamps/frame checks and hash-bound receipts. Composition
uses explicit frame indices, avoiding the previously rejected FFmpeg framesync
timebase mismatch. A package refuses mixed input/retarget/simulator manifests.
Existing outputs cause failure; inspect partial failures and choose another output
rather than silently overwriting them. A completed package still needs visual review.

To verify a teammate's frozen-target delivery against the preserved inputs and
original readback tolerances, run `scripts/dex3_review/verify_reproduction.py`.
The separate [physics validation stages](../../docs/team/dex3-isaac-review/PHYSICS_VALIDATION.md)
use an explicit overlay and candidate source mount; the preview commands above
remain kinematic.

## 5. Native continuity repair from the original cache

The native image can check its 35 installed runtime pins without loading MANO weights:

```bash
docker run --rm --platform linux/amd64 --entrypoint python \
  v2d-hand-continuity:checkpoint scripts/dex3_review/check_runtime.py \
  --runtime continuity --lock /opt/locks/continuity-runtime-lock.txt
```

The small replay pack starts **after** human fitting. To rerun that fitting, also
migrate the original `/workspace/hand-continuity-repair` workspace with its native
input cache, licensed local assets, source-image mask/anchor observations and
`runtime-additions`. Preserve absolute symlink targets by mounting the migrated
workspace at the same container path. The source audit found 3,336 symlinks and no external or broken targets in the
scanned native input/runtime/ROI roots. This is a link audit, not proof that all
episode inputs exist (e039 remains missing). Run
`python3 scripts/dex3_review/audit_workspace_links.py --root YOUR_WORKSPACE`
after transfer. Keep the existing pod/volume until a verified copy and the new
results have been checked. Exclude disposable installed virtual
environments from transfer if desired; preserve inputs, outputs, receipts and models.

`continuity-runtime-sources-v2.tar.gz` and its committed hash inventory preserve
the runtime source files (including the required `mano` support module) if `runtime-additions` must be restored separately. They
exclude MANO weights. The user/team must already be licensed for the mounted
MANO assets; the containers do not download or redistribute them.

If restoring those source files separately, verify and extract them into a fresh directory:

```bash
python3 scripts/dex3_review/restore_runtime_sources.py \
  --archive downloads/continuity-runtime-sources-v2.tar.gz \
  --receipt docs/team/dex3-isaac-review/continuity-runtime-sources.json \
  --out "$HOME/v2d-preserved/hand-continuity-repair/runtime-additions"
```

Set `V2D_WORKSPACE_DIR` in `.env`, then:

```bash
docker compose --env-file docker/dex3-review/.env \
  -f docker/dex3-review/compose.yaml --profile native-cache build continuity
# Re-run the preserved multi-stage hand repair (start with a single episode):
docker compose --env-file docker/dex3-review/.env \
  -f docker/dex3-review/compose.yaml --profile native-cache run --rm \
  --entrypoint bash -e INPUT_ROOT=/workspace/hand-continuity-repair/inputs \
  -e OUT=/outputs/continuity-rerun -e PYTHON=python continuity \
  reconstruction/validation/issue6/run_hand_continuity_review.sh 12
```

For e041/e042, retain the reviewed source-image recovery inputs and exact staged
commands in the frozen orchestration snapshot. Running the generic script alone
does not reproduce those additional manual recovery decisions or establish hidden
set generalization. Fresh HaMeR/DynHAMR/SAM inference and policy training require
additional upstream environments/assets and are not automated by these images.

To export a newly repaired run into the same Dex3 bundle schema, use
`scripts/dex3_review/export_bundle.py` in the continuity container with explicit
`--run`, `--checkpoint`, `--mano`, `--episode`, `--out` paths (override entrypoint
with `python`). The checkpoint must be the recorded native fit, not a guessed
latest file. This adapter preserves source frame indices and separate visual and
metric support; it checks MANO geometry against the original renderer.

## 6. Transfer built images rather than rebuilding

Use the tags/IDs in `CONTAINER_VERIFICATION.md` for the produced images. An image
archive can be loaded on Brev without installing Python dependencies again:

```bash
docker image save -o v2d-checkpoint.docker-image.tar \
  v2d-dex3-ik:checkpoint v2d-dex3-review:checkpoint v2d-hand-continuity:checkpoint
sha256sum v2d-checkpoint.docker-image.tar
# Transfer privately to Brev, verify the recorded hash, then:
docker image load -i v2d-checkpoint.docker-image.tar
```

To create a compressed archive with checked image IDs and SHA256, install `zstd`
on the host and use:

```bash
python3 scripts/dex3_review/save_images.py --out "$HOME/v2d-checkpoint.docker-image.tar.zst"
# After private transfer and hash verification on Brev:
zstd -dc "$HOME/v2d-checkpoint.docker-image.tar.zst" | docker image load
```

The Docker cache/log volumes and bind-mounted outputs survive container removal.
Export data, outputs and image hashes before retiring the old machine. Physics
validation belongs in new output directories and a new mode: these replay commands
explicitly keep gravity/collision response off and directly assign object poses.
Turning on gravity alone does not turn this replay into a learned policy or an
unassisted grasp test. The existing
[full-task physics diagnostic](../../docs/team/reproduction/SUPPORTED_DEX3_TASK_DIAGNOSTIC_2026-10-14.md)
already passed gravity/contact controls but failed wrist tracking and object
retention. Controller stability and a complete physical task remain the next gates. The current official Track 3 policy runtime and six-metric
submission kit must be validated separately before a final evaluation claim.

The [October 15 physics comparison](../../docs/team/dex3-isaac-review/PHYSICS_VALIDATION.md)
now contains actual free-space, full-task and decomposed static-load results.
All three controller candidates pass the short free-space test; full manipulation
still fails. Those commands reproduce diagnostics and must not be used as a
training acceptance claim. Build a local comparison page from completed stages:

```bash
python3 scripts/dex3_physics/build_review.py \
  --experiment 'implicit PD=/absolute/path/to/implicit-run/candidate-full' \
  --experiment 'velocity feedforward=/absolute/path/to/feedforward-run/candidate-full' \
  --out /absolute/path/to/new-physics-review
```

The builder checks source state/video hashes, recomputes the wrist decisions and
binds independent contact audits before copying the videos and receipts. It
allows 1e-12 rounding in recomputed scalar summaries across CPUs; thresholds,
hashes and pass/fail decisions must match exactly.

## 7. End each Brev session

Stop compute when it is no longer in use. Prefer a provider marked `stoppable`
when the team needs to preserve a VM between sessions. After copying the results,
use the exact instance name and verify the resulting state:

```bash
brev stop YOUR_INSTANCE_NAME
brev ls --json
```

Some providers, including the temporary `massedcompute_L40S` build VM, are listed
as non-stoppable. A paused container or disconnected SSH session does not release
that VM. Copy and verify source, image archives and results first, then delete the
temporary VM with the owner's confirmation and verify it is absent from `brev ls`.
Do not delete the original RunPod workspace as part of this procedure.

See [Brev instance lifecycle commands](https://docs.nvidia.com/brev/cli/instance-management).


## Current motor and contact diagnostics

Use the locally retained v4 input pack and its source33 overlay for the newest
motor/contact checks. See the [measured checkpoint](../../docs/team/dex3-isaac-review/motor-contact-repair-results-v1.json)
and [private input receipt](../../docs/team/dex3-isaac-review/local-pot-task-input-receipt-v4.json).
The existing pinned image is unchanged; these commands require the source overlay.
The optional `--contact-solver grip` profile and `--physics-dt 0.0025` must match
between `grip-task-free` and `grip-task`. Bind the recipe's exact seed through
`V2D_GRIP_SEED`; use a fresh `V2D_OUTPUT_DIR` per recipe. The fixed 4 N comparison
uses `combined-task-grasps-grip4N-v11.json`; the live comparison uses
`combined-task-grasps-hybrid-v10.json`. Both remain diagnostic references.

The isolated probe uses `/opt/sim/bin/python` as its container entrypoint and:

```bash
/code/scripts/dex3_physics/check_implicit_torque_budget.py \
  --out /outputs/probe.json --encode-in-drive --contact-solver grip
```

A passing probe checks the actuator implementation; it does not certify a grasp.
A full-stage exit code of zero verifies execution, physics controls and wrist
tracking; inspect the separate grasp diagnostics for object retention.
The optional grip solver differs
from the default diagnostic settings; official evaluation compatibility remains
unverified. Inputs are local and are not included in GHCR images or Git.
