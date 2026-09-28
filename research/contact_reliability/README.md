# NVIDIA V2D contact reliability research

## Monorepo usage

This self-contained package lives at `research/contact_reliability`. Run its
commands from this directory. Install with `python -m pip install -e '.[test,training,motion]'`
in a suitable research environment; use the existing managed Isaac Lab dependencies
on the GPU instead of replacing them with the Windows requirements file.

Run `python -m pytest -q` here. Reward-parity tests automatically use this
monorepo's NVIDIA source. For training, set `--upstream ../..` and
`--revision-lock configs/team_revision.json` in the launcher command. The revision
guard intentionally refuses a changed HEAD: revalidate and update the revision
lock after integrating this package before launching training.

Downloaded data, virtual environments, generated run outputs and model assets are
excluded from this contribution. The original validation reports describe the
standalone workspace where those artifacts were generated; use the documented
download/audit commands to reproduce them. No GPU training result is included.

This workspace implements the software foundation for a Track 3 entry studying
whether reconstruction reliability improves contact-guided robot learning at a
fixed compute budget. It is not a trained policy or an accepted submission.

## Current status

September 25: see [Evan's implementation and handoff](docs/EVAN_DELIVERABLES.md).
Public pose tables and metadata are now downloaded and audited (20 episodes,
5,246 frames). Use `track3-split` and `track3-matrix` for the fixed team proxy split.

- NVIDIA source is pinned to `43a3f5b4d479581f2e8bdf5e6ed7ca361d39ec5a` from `release/0.2.0`.
- Signal extraction, development-only normalization, sidecars, ablations,
  contact-reward integration, experiment matrices, run records, and a guarded
  evaluator runner are implemented.
- CPU tests compare weighted kernels with the actual pinned NVIDIA functions.
  Synthetic smoke artifacts explicitly identify themselves as synthetic.
- No challenge videos, official `eval_e2e.py`, licensed MANO assets, or trained
  checkpoint is present. No robotics success claim has been measured.
- Local hardware: RTX 5060 Laptop GPU, 8 GB, compute capability 12.0; Docker is
  absent in Windows and the inspected Ubuntu WSL environment. The upstream
  cuVSLAM/TensorRT reconstruction path excludes this architecture. Full execution
  needs a validated environment and adequate storage/compute.

## Start here

The local `.venv` is already installed. On another machine:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test,training]"
.\.venv\Scripts\python scripts/bootstrap_upstream.py
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python scripts/smoke_demo.py
.\.venv\Scripts\v2d-research doctor --output runs/environment.json
```

On Linux use `.venv/bin/python` and `.venv/bin/v2d-research`. In an existing Isaac
Lab environment install `pip install -e . --no-deps` to preserve its approved
PyTorch/NumPy stack; do not install this Windows environment lock there.

Read [the execution runbook](docs/RUNBOOK.md), [data contracts](docs/CONTRACTS.md),
[the team backlog](docs/BACKLOG.md), and [external dependencies](docs/DEPENDENCIES.md).
The [technical report outline](docs/REPORT.md) separates planned claims from evidence.

## Reproducibility boundaries

The bootstrap leaves upstream tracked source unchanged, downloads no LFS payloads,
and does not accept dataset or model license terms. The training launcher injects
one configuration call into memory after upstream scene configuration. Omitting
the sidecar runs the original script unchanged. Upstream source and assets retain
their own licenses; see NOTICE for reward formula provenance.

The integration targets the pinned **whole-body motion command**, not an assumed
official Track 3 embodiment. It must be smoke-tested in Isaac Lab once the official
task is confirmed. Both hands must have explicit motion/sidecar entries. Contact
point and force-closure rewards use conservative per-hand reliability because
their upstream objectives merge object bodies; wrench rewards use per-body weights.

All training configs default to baseline. The project neither rents compute nor
uploads artifacts automatically. The evaluator adapter checks supplied files and
exit status, not leaderboard acceptance or unspecified official schema semantics.
# September 25 implementation

Start with [Evan's deliverables and runbook](docs/EVAN_DELIVERABLES.md) and the
[evaluation handoff](docs/EVALUATION_HANDOFF.md). Track 3 uses the fixed team
proxy split, not the older generic randomized split described below.
