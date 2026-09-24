# Issue #1 — GPU box, Docker, Isaac Lab + CHORD smoke test

Owner: **Role 2 (Cloud & Build)** · GitHub: [Issue #1](https://github.com/Swaguto/NVIDIA-V2D-Challenge-/issues/1) · Labels: `infra`, `isaac-sim`

**Goal:** the box every teammate will log into. When this is done, anyone can
start a simulator container and run a policy — and the Scorekeeper (Role 3) can
start scoring outputs.

---

## What "done" looks like

1. A teammate can **log in** (SSH / their own shell) and run `nvidia-smi`.
2. `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` works.
3. `video_to_data` (release/0.2.0) is checked out with LFS assets pulled.
4. **robotic-grounding** + **v2d_task_library_loader** images are built.
5. The **smoke test** below produces a working MP4 of the Sharpa robot replaying a
   recorded demo (or completes 1 training iteration) — proving Isaac Lab task
   registration + asset loading work.
6. The track_3 dataset is available on the box (mount or rsync).

---

## Step 0 — Provision the instance (on the cloud provider)

| Requirement | Value |
|---|---|
| GPU | **A100 40/80GB, H100, or L40S/A6000** — *NOT* Blackwell (sm_120 unsupported by V2D TensorRT/cuVSLAM) |
| VRAM | ≥ 24 GB (48 GB recommended) |
| Driver | ≥ 580.126 (containerized Isaac Lab 2.3.2 requires recent driver + CUDA 13 runtime) |
| OS | Ubuntu 22.04 or 24.04, x86_64 |
| Disk | ≥ 200 GB (reconstruction images + weights + dataset are large) |
| RAM | ≥ 64 GB |
| Network | outbound to `nvcr.io`, `huggingface.co`, `github.com` |

Log in over SSH once provisioned.

## Step 1 — Run the bootstrap

The pipeline now lives **in our repo** (`robotic_grounding/` etc. at the repo
root). On a fresh box, clone **this** repo and pull LFS objects from the public
upstream via the committed `.lfsconfig`:

```bash
git clone https://github.com/Swaguto/NVIDIA-V2D-Challenge-.git
mv NVIDIA-V2D-Challenge- video_to_data      # keep the same dir name the runbook expects
cd video_to_data
git lfs install && git lfs pull             # ~1GB assets, fetched from public upstream
```

Simplest path: copy this repo's helper locally and run it.

```bash
bash bootstrap_cloud.sh            # GPU + Docker + toolkit + clone (does not reboot)
sudo reboot                        # only if it installed a driver
bash bootstrap_cloud.sh --with-reconstruction
```

On a box where the driver is already present, this completes in a single pass.

## Step 2 — NGC access (for Isaac Lab base image)

```bash
docker login nvcr.io    # username: $oauthtoken  password: <NGC API key from https://ngc.nvidia.com/signin>
```

## Step 3 — Build the two pipeline images

```bash
python3 -m venv ~/venvs/v2d && source ~/venvs/v2d/bin/activate
cd video_to_data/reconstruction
pip install -e modules/v2d_common -e modules/v2d_docker -e modules/v2d_task_library_loader/docker
cd ../robotic_grounding
python scripts/run_pipeline_docker.py --build-only   # builds loader + robotic-grounding
```

Reference: [`robotic_grounding/docs/SETUP.md`](../../../robotic_grounding/docs/SETUP.md) (in-repo upstream docs)

## Step 4 — Smoke test (no external dataset needed)

Track 3's own videos live in `track_3/`, but the in-repo **synthbox** sample is a
zero-download way to prove the stack. Start the simulator container:

```bash
cd video_to_data/robotic_grounding
./workflow/run.sh start latest 0
```

Inside the container (working dir `/workspace/video_to_data/robotic_grounding`):

```bash
# Asset-less dummy-agent replay (no object URDFs needed) → proves task + motion load
python scripts/rsl_rl/dummy_agent.py \
  --headless \
  --task Sharpa-V2D-v0-Play \
  --motion_file synthbox/synthbox_processed/sequence_id=synthbox_box_open_000/sharpa_wave \
  --num_envs 1 \
  --use_primitive_urdfs \
  --record_video \
  --output_dir /tmp/rg_dummy_agent_video \
  --video_length 300
```

Success = no missing-asset exception + simulation advances + MP4 written to
`/tmp/rg_dummy_agent_video`. (GUI variant: drop `--headless --record_video`.)

**Or** a real 1-iteration training run:

```bash
python scripts/rsl_rl/train.py --headless --task Sharpa-V2D-v0 \
  --motion_file synthbox/synthbox_processed/sequence_id=synthbox_box_open_000/sharpa_wave \
  --num_envs 1 --max_iterations 1 --logger tensorboard --run_name issue1_smoke \
  --use_primitive_urdfs
```

Whole-body (G1+Dex3 — our final embodiment) smoke test, also in-repo:

```bash
python scripts/rsl_rl/dummy_agent.py \
  --headless --task SonicG1-ReconBody-v0 \
  --motion_file whole_body/soma/sequence_id=2026-03-06_10-24-18_snack_box_pick_and_place_01/g1 \
  --num_envs 1 --record_video --output_dir /tmp/rg_g1_reconbody --video_length 300
```

## Step 5 — Get the track_3 dataset onto the box

Efficient path (dev box already has it):

```bash
rsync -av --info=progress2 $USER@<dev-machine>:~/track3_data/ ./data/
```

Or from Hugging Face:

```bash
pip install -U "huggingface_hub[cli]"
hf download nvidia/video_to_data_challenge --repo-type dataset --include "track_3/*" --local-dir ./data/hf/track_3
```

---

## Outputs / handoff

- [ ] Instance details (IP, GPU, RAM) + login notes in team channel
- [ ] Datasets: `track_3/` on the box
- [ ] Smoke-test MP4s posted in the GitHub issue
- [ ] A `~/README.md` on the box with the daily-use commands above

## Mistakes to avoid

- **Blackwell GPU** (`sm_120`) — the build checks will fail; use Ampere/Hopper.
- **Missing NGC login** — the `robotic-grounding` build pulls `nvcr.io/nvidia/isaac-lab:2.3.2`; it will fail with a pull error if not logged in.
- **Missing LFS pull** — robot URDFs/meshes come via Git LFS; broken pointers → asset errors in the smoke test. Fix: `git lfs install && git lfs pull` (objects come from the **public upstream** repo, not our private repo's LFS).
- **Driver too old** — Isaac Lab 2.3.2 needs driver ≥ 580; check with `nvidia-smi`.

## Related docs

- Infra/team workflow: [`TEAM_ACCESS.md`](TEAM_ACCESS.md)
- Upstream setup (in-repo): [`robotic_grounding/docs/SETUP.md`](../../../robotic_grounding/docs/SETUP.md)
- Upstream README: [`docs/team/upstream-README.md`](../upstream-README.md)