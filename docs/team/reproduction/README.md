# Reproduce issues 6, 7 and 8

For the corrected October 13 Dex3 motion replay, start with
[the current replay instructions](CORRECTED_DEX3_REPLAY_2026-10-13.md).
They cover the completed 343-frame Isaac replay, complete modeled continuous
check, exact fitted-profile regeneration,
observed/inferred provenance, and remaining physical-target qualifications.
The older measurements below retain their original scope and acceptance state.

The team bundles are in [Google Drive](https://drive.google.com/drive/folders/1aie0RV1jJ_XxF5ddJbGVRpJskdi4IMmA).
Sign in with the account listed on the shared project document. The nine named
teammates have read access; the folder is restricted. Download the four archives
without extracting them. Exact filenames, sizes, download links and SHA256 values
are in [artifacts.json](artifacts.json). Each command checks the archive and every
member before writing to a new directory, then recomputes the stated measurements.

| Issue | Reproducible result | Scope |
|---|---|---|
| 6 | Episode 2 v3 contact replay with reproducible edge/vertex normals | Full 343-frame baseline, original contact/unknown/bridge decisions unchanged; all-20 current quality acceptance remains 0/20 |
| 7 | Episode 2 quality report from the unchanged GPU scene | All non-official report fields; camera jitter 0.4809198709547093 → 0.20234134418580074 mm; includes independent contact precision and coverage |
| 8 | Episode 2 derived collision geometry and gate | All 343 frames: 72 pass, 199 rejected, 72 unsupported; exact disposition masks; physical manipulation and between-frame collision remain unverified |

These commands recompute results from frozen experiment inputs. They do not rerun
upstream hand inference, the GPU optimizer, or robot training. CPU tests alone do
not establish those experiments. Fresh inference/SLAM/optimization and simulator
rendering have separate instructions below.

## Setup once

Use **Python 3.11** and a fresh virtual environment. Run from the repository root.
Before this handoff PR is merged, check out `Evc-r/team-reproducibility`.
No GPU, Docker, private MANO model, model download or submodule checkout is needed
for the three measurement replays.

```bash
git clone https://github.com/Swaguto/NVIDIA-V2D-Challenge.git
cd NVIDIA-V2D-Challenge
# Before the handoff PR merges:
git checkout Evc-r/team-reproducibility
python3.11 -m venv .venv-reproduce
source .venv-reproduce/bin/activate
python -m pip install -r scripts/reproduce/requirements-cpu.txt
```

Windows PowerShell, after cloning and entering the repository:

```powershell
py -3.11 -m venv .venv-reproduce
.\.venv-reproduce\Scripts\Activate.ps1
python -m pip install -r scripts/reproduce/requirements-cpu.txt
```

Place downloaded archives in a directory called `downloads` at the repository
root. The following commands work unchanged in Bash and PowerShell. Keep every
`--out` directory new; a partial or completed run is preserved and cannot be reused.
Floating measurements use an absolute tolerance of 1e-8 in report units for
issues 6/7, while counts, masks, nulls, booleans and text must match exactly.
Issue 8 uses 1e-12 metres for floating gate arrays and exact disposition masks.

## Issue 6: replay real contact calculations

```text
python scripts/reproduce/checkpoint.py --issue 6 --archive downloads/issue6-episode2-contacts-stable-20261009.tar.gz --out reproduce-issue6
```

The command recomputes every fingertip/object contact on the full source timeline
and compares `contacts.json`, `dominant_objects.json` and `settings.json` against
the archived native outputs. Read `reproduce-issue6/reproduction.json` and
`reproduce-issue6/recomputed-contacts/`. The pack contains original aligned hand
observations, object poses and CAD geometry, with only referenced paths relocated.
It includes original manifest bytes and the parent full-archive SHA256.

The old nearest-face argmin could choose different normals at mesh edges across
runtimes: 51 normals differed in the initial Linux replay while every other
contact field matched. The repair selects the lowest face index within 1e-12 m
of the minimum and retains the actual minimum distance for the strict 2 cm gate.
The corrected pack preserves `historical-contacts/` and compares all non-normal
fields against them; 37 canonical normals differ from the original frozen output.
The current `expected-contacts/` includes the deterministic normals and policy.
The `HISTORICAL-` archive in Drive preserves the earlier package; use the
`contacts-stable` archive for the supported replay.

This is the October 7 v3 **baseline**, preserved before the stricter quality
scope. It does not accept all 20 episodes or repair the remaining visible-grasp
failures. Follow [QUALITY_ACCEPTANCE.md](../../../reconstruction/validation/issue6/QUALITY_ACCEPTANCE.md).
For new six-stage processing and authorized native models, use
[HANDOFF.md](../../../reconstruction/modules/v2d_issue6/HANDOFF.md) and
[issue6-handoff.md](../issue6-handoff.md). The complete historical all-20 archives
are retained by Evan locally; the lightweight team pack covers episode 2.

## Issue 7: regenerate the quality measurements

```text
python scripts/reproduce/checkpoint.py --issue 7 --archive downloads/issue7-episode2-metrics-20261009.tar.gz --out reproduce-issue7
```

Read `reproduce-issue7/recomputed-report.json` and `reproduction.json`. The command
uses the exact scene, contact rows, independent contact truth, source-reviewed
rest interval and input provenance from trial4. Every non-official field is
compared with the frozen report, including unknown coverage and drift regressions.
Public-reference scoring (`official`) is excluded explicitly: it requires the
separate reference dataset and scorer and is not needed for jitter/contact replay.
Coarse hand-object precision does not establish fingertip or metric 3D truth.

The [source/render comparison video](https://drive.google.com/file/d/1ZckS_94bd3z3Cce8p_u1l6iInDISymgb/view?usp=drivesdk)
is also available directly in Drive for team review.

For new calibrated SLAM and GPU refinement, use
[the issue 7 experiment runbook](../issue7/REPRODUCE.md). It documents DROID/model
pins, native CUDA setup, input provenance, optimizer settings, fresh outputs and
full visual review. The accepted GPU scene predates subsequent software repairs;
the measurement replay preserves that scene rather than claiming new optimization.

## Issue 8: rebuild the derived collision model and gate

Download both `episode2-pilot-20261009.tar.gz` and
`episode2-derived-collision-20261009.tar.gz` from Drive, then run:

```text
python scripts/reproduce/issue8.py --baseline downloads/episode2-pilot-20261009.tar.gz --supplement downloads/episode2-derived-collision-20261009.tar.gz --out reproduce-issue8
```

This restores a portable workspace containing the exact executed sources/assets,
rebuilds both occupied-cell object solids and all hand-volume covers, reruns the
full timeline, and compares every model/gate array. The 72/199/72 result and exact
passing mask are checked. Completion requires
the newly written `reproduce-issue8/reproduction.json`, which is created only
after both subprocesses succeed. The detailed array comparisons are in
`reproduce-issue8/runs/issue8-validation/derived-portable-reproduction.json`.
Logs for each recomputation are in the same directory. Initial restored logs and
old receipts are historical; this new receipt is written only after recomputation.

The archives contain `simulator/replay11/replay.mp4`, 343 rendered frames and the
review receipts. Open the MP4 for the already reviewed 17.15-second result.
The [latest replay video](https://drive.google.com/file/d/1okzO6KBaQWDx8CJC1dK2ny3L1U_WwG3O/view?usp=drivesdk)
can also be viewed directly in Drive.
For fresh IK or rendering, see the [PR35 runbook](https://github.com/Swaguto/NVIDIA-V2D-Challenge/blob/43fc832b6c6ceae536d1f89f952126e3c2c6413d/docs/team/issue8/REPRODUCE.md).
Use its separate native IK and Isaac Lab 2.3.2 / Sim 5.1 environments; the NumPy
2.2.6 IK closure must not be installed into the NumPy 1.26 simulator environment.
The CPU environment above cannot launch Isaac Sim.

The model infers closure, thickness and handle interiors. Original CAD and IK
are unchanged. Passing timestamps establish bounds for the declared model;
they do not certify real solids, continuous motion, physics or learned-policy success.

## Software controls and verification

```text
python -m pytest scripts/reproduce/tests -q --strict-markers
```

These controls reject corrupt hashes, unsafe paths, links, duplicate members,
existing output directories and changed missing-data/pass decisions. The GitHub
`Team reproduction controls` workflow runs them plus the issue 6/7 contracts.
Private archives stay in Drive; CI cannot reproduce their measurements without
authorized access. See [VERIFIED.md](VERIFIED.md) for executed reproduction receipts.

If an archive check fails, download the exact catalogued filename again. If a
measurement comparison fails, preserve the new output and logs with the error;
do not loosen thresholds or overwrite the frozen result. Use the pinned packages
and Python 3.11. Teammates can attach the generated JSON receipt when reporting results.
