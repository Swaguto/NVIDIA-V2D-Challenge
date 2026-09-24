# V2D Challenge — Track 3 (Egocentric Video → Policy)

**Track 3:  team setup, and the cloud GPU + CHORD smoke test.**

We teach a **Unitree G1 + Dex3** humanoid (in Isaac Lab) to reproduce kitchen
manipulation from a head-worn **stereo camera** video. We use NVIDIA's CHORD
pipeline as the scaffold and improve the vision half (stereo depth + CAD
tracking) where the baseline loses ~27% on real video.

- Freeze: **Nov 4, 2026, 5:00 p.m. ET** · ≤5 submissions/week (unlimited last 3 days)
- Metrics: AUC (ADD-AUC), SP-SR, MP-SR, MPPE
- Embodiment: **Unitree G1 + Dex3** · Sim: Isaac Lab 2.3.x + RSL-RL PPO

---

## Quickstart (for the team / new box)

The full runbook is in [`docs/issue1/README_ISSUE1.md`](docs/issue1/README_ISSUE1.md).
5-minute version:

```bash
# 1. Provision a GPU box (A100 / H100 / L40S — NOT Blackwell). Then:
bash docs/issue1/bootstrap_cloud.sh              # GPU + Docker + toolkit + repo clone
sudo reboot                                      # only if it installed a driver
bash docs/issue1/bootstrap_cloud.sh --with-reconstruction

# 2. NGC access (pulls nvcr.io/nvidia/isaac-lab:2.3.2)
docker login nvcr.io   # username: $oauthtoken, password: your NGC API key

# 3. Smoke test (zero-download, in-repo synthbox motion)
cd ~/video_to_data/robotic_grounding
./workflow/run.sh start latest 0
# inside the container:
python scripts/rsl_rl/dummy_agent.py --headless --task Sharpa-V2D-v0-Play \
  --motion_file synthbox/synthbox_processed/synthbox_box_open_000/sharpa_wave \
  --num_envs 1 --use_primitive_urdfs --record_video --output_dir /tmp/smoke
```

Verified smoke-test MP4s live in [`docs/issue1/`](docs/issue1/):
[`sharpa_smoke.mp4`](docs/issue1/sharpa_smoke.mp4) (Sharpa floating hand)
and [`g1_smoke.mp4`](docs/issue1/g1_smoke.mp4) (G1 whole body).

---

## Repo contents

| Path | What it is |
|---|---|
| [`docs/plan.md`](docs/plan.md) | Winning architecture (challenge decode, CHORD-vs-DexMachina, pipeline stages A–E, timeline, risks) |
| [`docs/team_brief.md`](docs/team_brief.md) | Plain-language 9-role team brief + hand-offs |
| [`docs/tasks/github_issues.md`](docs/tasks/github_issues.md) | The 16 issue cards (also live in the GitHub Issues tab, with labels + milestones) |
| [`docs/issue1/README_ISSUE1.md`](docs/issue1/README_ISSUE1.md) | Full Issue #1 runbook: provision, build, smoke test, dataset sync |
| [`docs/issue1/bootstrap_cloud.sh`](docs/issue1/bootstrap_cloud.sh) | Provider-agnostic bootstrap script (blocks Blackwell, driver build, Docker + toolkit, clone) |
| [`scripts/cloud_setup_housekeeping.sh`](scripts/cloud_setup_housekeeping.sh) | Box housekeeping: git-lfs, clone, LFS pull, host venv |
| [`scripts/diag_cloud.sh`](scripts/diag_cloud.sh) | One-shot health check for any team GPU box |

## GitHub project state

- Discussion / tracking: all work lives in the **16 issues** (labels: `infra`,
  `ml`, `isaac-sim`, `perception`, `mapping`, `retargeting`, `rl`, `eval`,
  `integration`, `data`, `stretch`, `blocker`; milestones W1–Freeze).
- Issue #1 (GPU box + Isaac Lab + CHORD smoke test) is done and verified.

---

## Security / house rules (important)

- This repo is **public**. Never commit: NGC API keys, cloud instance IDs/names,
  SSH keys, `.docker/config.json`, or personal local paths.
- Cloud box snapshots / READMEs with credentials stay on the box, not in this repo.
- If you need a private space: use GitHub Secrets / a private repo for secrets,
  and keep only placeholders here.

## References

- Starter toolkit: https://github.com/nvidia-isaac/video_to_data (branch `release/0.2.0`)
- Challenge page: https://nvidia-isaac.github.io/video_to_data/v2d_challenge/
- CHORD paper: https://arxiv.org/abs/2607.00033
