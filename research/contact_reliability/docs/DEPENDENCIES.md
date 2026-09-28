# External dependencies and current evidence

Status checked September 21, 2026. The challenge's public description is not a
substitute for the official Track 3 package and rules.

| Dependency | Status | Required action and owner |
|---|---|---|
| Pinned baseline source | Downloaded, commit locked | Infrastructure lead preserves clean source |
| CPU toolkit | Implemented; see test evidence | Reliability and evaluation leads review |
| Challenge dataset | Not supplied | Project lead obtains permitted access and records split rules |
| Official eval_e2e.py | Absent from reviewed release tree | Evaluation lead obtains official package and SHA-256 |
| Track 3 robot/observations and metric semantics | Not verified | Project lead confirms; do not inherit Track 2 rules |
| Compatible GPU | Not available in inspected environment | Infrastructure lead obtains approved access |
| Docker | Not found in Windows or inspected WSL | Infrastructure lead provisions approved Linux host |
| Licensed MANO and other model assets | Not obtained | Team member with authority completes licenses/downloads |
| Named team owners | Ten names/workstreams in live Doc; individual leads incomplete | See PROJECT_CONTEXT.md; confirm integration and evaluation leads |
| Paid budget | Unconfirmed; spending not enabled | Team lead confirms currency, cap, and authorization |
| Baseline rollout / official artifact | Not produced | Depends on data, assets, evaluator and compute |

The local GPU is RTX 5060 Laptop, 8151 MiB, compute capability 12.0. The release
documents that its cuSFM reconstruction dependencies do not support sm_120. This
does not prove every module fails, but it prevents treating this laptop as a
validated end-to-end host. Disk inventory was about 104 GB free before setup;
the reconstruction documentation describes a substantial container footprint.

Questions for the project lead to consolidate (not sent automatically):

1. Where are Track 3 data, `eval_e2e.py`, schema, example submission and metric definitions?
2. Which robot, simulator version, observations and pretrained assets are permitted?
3. May teams train policies per test demonstration, and what data/labels are forbidden?
4. Are external datasets, manual corrections and human intervention allowed and reported?
5. Is compute provided, and which hardware/container configurations are supported?
6. What code-release license and report format satisfy award eligibility?

Sources: https://nvidia-isaac.github.io/video_to_data/v2d_challenge/ and the pinned
upstream READMEs. Do not silently update the upstream branch during experiments.

## Follow-up repository check

The user confirmed they have no separately obtained dataset or known GPU server.
A recursive check of the current `main` tree also found no `eval_e2e.py`.
The public leaderboard metadata does identify Track 3 metrics: AUC, SP-SR,
MP-SR (higher is better), RPE and MPPE in cm (lower is better). The exact payload
is saved in `configs/track3_public_metadata.json`; definitions, aggregate ranking
rules and evaluator behavior still need verification.

That metadata links to Kaggle competitions, including
https://www.kaggle.com/competitions/v2d-challenge-track3-auc/data . Automated browsing
could not retrieve this page, so no claim is made that its data are downloadable
without registration. Check that official entry point through the team's account.
Repository example assets are not assumed to be the held-out challenge dataset.
