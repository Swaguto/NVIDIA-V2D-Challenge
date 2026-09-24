# Team access & contribution workflow

This is the playbook every teammate needs to contribute to
`Swaguto/NVIDIA-V2D-Challenge-`. Two things are unusual, read both sections.

---

## 1. The repo: merged monorepo, LFS pointers only

The repo root **is** the `video_to_data` pipeline tree
(`robotic_grounding/`, `reconstruction/`, `video_ingestion_agent/`,
upstream `docs/`, `.github/`, `LICENSE`, …) plus our team docs in
`docs/team/`.

**Critical: binary assets are stored as git-lfs POINTERS, not the objects.**
The ~956MB of meshes/ONNX/USD/motion files are *not* in this repo — they are
downloaded on demand from the **public** upstream repo
(`nvidia-isaac/video_to_data`) via a committed `.lfsconfig`.

Why: GitHub's free LFS is capped at **1GB storage + 1GB bandwidth/month**.
Uploading the assets here would exhaust it in one shot; pointing fetches at the
public upstream keeps our repo tiny (~1MB) and free.

### Cloning fresh (laptop, Windows/macOS/Linux)

```bash
# 1. Install git + git-lfs  (macOS: brew install git-lfs; Windows: git-lfs installer)
git lfs install

# 2. Clone (you must have been added as a collaborator)
git clone https://github.com/Swaguto/NVIDIA-V2D-Challenge-.git
cd NVIDIA-V2D-Challenge-

# 3. Pull the binary objects from the public upstream (one-time, ~1GB)
git lfs pull
```

> If `git lfs pull` errors, confirm `git-lfs` is on PATH and run
> `git lfs install` again — the `.lfsconfig` in the repo root is what routes
> fetches to the public upstream.

### The shared VM already has everything

The Brev GPU box (`rising-gold-junglefowl`, Ubuntu, L40S) already has the full
working tree incl. all LFS objects under `/home/ubuntu/video_to_data`. You
**do not** need to re-download 1GB there — just make sure your checkout points
at our repo and `git lfs pull` is a no-op (objects already present).

---

## 2. Day-to-day: branch → PR → merge → pull on the box

### GitHub flow (all teammates)

1. Make sure you're on a **feature branch** for your issue, never push to `main` directly:
   ```bash
   git checkout -b issue/42-my-feature
   ```
2. Commit with a clear message (`closes #42` when it fixes the issue).
3. Push and open a **PR** (GitHub UI, or `gh pr create`):
   ```bash
   git push -u origin issue/42-my-feature
   ```
4. Request review (pick 1 reviewer), keep the PR small and focused.
5. Merge via the GitHub UI when green. Keep `main` always runnable.

### Running on the shared VM

Everyone shares one box (only the owner can start it). To use the latest merged
code on the box, **pull on the VM** — the mounted volume in the running
container picks it up instantly (no rebuild):

```bash
# on the box, as ubuntu
cd ~/video_to_data
git fetch origin && git checkout main && git pull --ff-only
```

Long training runs: use `tmux` (or `nohup`) so they survive your SSH drop.

### If you added a NEW binary asset

The pipeline will happily generate new `.onnx`/`.parquet`/`.usd` files. Do **not**
commit a new large asset to `main` without thinking about LFS quota:

- If it's a small derived artifact, check with the team before adding it to git.
- Prefer regenerating assets in place (they get ignored / stay out of git) over
  committing new binaries.
- If a real new binary asset is needed by the rest of the team, upload its
  object to **our** repo LFS too: `git lfs track <file> && git add <file> &&
  git commit && git push` then `git lfs push --all origin`. Flag any >100MB file
  first — GitHub free LFS has a 1GB cap.

---

## 3. One-time setup for each teammate

1. **Get GitHub access**: tell the repo owner your GitHub username → they add you
   as a **collaborator** on `Swaguto/NVIDIA-V2D-Challenge-` (private repo).
2. **SSH to the shared VM** (optional but recommended for compute): add your SSH
   public key to `/home/ubuntu/.ssh/authorized_keys` on the box (ask the owner
   to append it), then `ssh ubuntu@<public-ip>`.
3. **Set your git identity on the VM** (the box is shared; never commit as
   `ubuntu`'s default identity):
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```
   Per-member identity + PR-based workflow keeps attribution correct even though
   the VM account is shared.
4. **Read the runbook** [`README_ISSUE1.md`](README_ISSUE1.md) and the plan
   [`docs/team/plan.md`](../plan.md).

---

## 4. Gotchas

| Gotcha | What to do |
|---|---|
| `git push` rejected "fetch first" | `git pull --rebase` then push |
| `git lfs pull` slow / errors | Check `.lfsconfig` is present; run `git lfs install`; check network |
| Push failed with LFS lock error | Locks are disabled; `git config lfs.locksverify false` |
| Huge file staged by accident | `git reset <file>` before committing (never force-push) |
| `main` accidentally broken | Fix forward in a PR — never rebase/force-push `main` |