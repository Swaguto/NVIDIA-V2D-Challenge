# Executed verification — October 9, 2026

The commands in [README.md](README.md) were run against the exact four archives
listed in [artifacts.json](artifacts.json), using fresh output directories and
Python 3.11.17 with the pinned CPU packages. These are measured replays of frozen
episode 2 inputs. Private experiment data remains in the restricted team Drive.

| Issue | Runtime | Executed result | Receipt |
|---|---|---|---|
| 6 | Windows and Linux | All 343 frames, 6,860 contact rows; 1,442 raw positives and 1,310 unknowns. All current output fields match. Historical non-normal fields match; 37 canonical normals changed. | [Windows](receipts/issue6-windows.json), [Linux](receipts/issue6-linux.json) |
| 7 | Windows and Linux | All non-official report fields match; camera high-frequency translation RMS 0.4809198709547093 → 0.20234134418580074 mm. | [Windows](receipts/issue7-windows.json), [Linux](receipts/issue7-linux.json) |
| 8 | Linux | Both object occupied-cell models and all hand-volume covers rebuilt, all gate arrays compared, exact passing mask; 72 accepted / 199 rejected / 72 unsupported across 343 frames. | [Linux](receipts/issue8-linux.json) |

For issues 6/7, floating values use absolute tolerance 1e-8 in the report's units;
counts, contact decisions, unknowns, booleans and text match exactly. Issue 8 uses
1e-12 m for floating gate arrays and exact disposition masks. A successful
`reproduction.json` is written only after recomputation and comparison succeed.
The receipts record package versions, input hashes and executed source byte hashes.
Source byte hashes describe the tested local checkout; Git line-ending conversion
can change byte hashes without changing Python code.

## Repair discovered during reproduction

The first issue 6 Linux replay differed from the historical Windows result in
51 surface normals while all other contact fields matched. At tied mesh edges
or vertices, roundoff could change the face selected by raw argmin. The repaired
calculation picks the lowest face index within 1e-12 m of the actual minimum,
and retains the actual minimum distance for strict contact classification.
Two regression controls cover tied-face selection and preservation of the
minimum distance. The corrected full replay passed on both runtimes. Original
contact outputs remain in the stable archive's `historical-contacts/` directory;
the earlier archive is retained in Drive with a `HISTORICAL-` filename.

## Software checks

The same final source passed **260 Linux tests with zero skips**: 143 issue 6
contracts, 102 issue 7 contracts and 15 archive/comparison controls. The 15 helper
controls also passed in the fresh native Windows environment. The pinned Linux
test environment additionally includes CPU PyTorch 2.9.1 for the issue 7 contracts.
Private-archive measurements are local verification; the public GitHub workflow
runs software controls without private data.

## Limits of these results

Issue 6 reproduces the October 7 v3 episode 2 baseline. The stricter all-20
quality objective remains **0/20 accepted**. Historical contact decisions are
preserved; reproduction does not repair the remaining visible-grasp failures.

Issue 7 recomputes measurements from the previously executed GPU scene. The
upstream hand model, DROID-SLAM, GPU optimization and official reference scorer
were not rerun here. Contact truth is coarse hand-object source review, not
independent per-fingertip or metric 3D truth.

Issue 8 recomputes collision bounds for the declared derived geometry model at
sampled timestamps. Original CAD and IK are unchanged. Fresh IK, Isaac Sim
rendering, physics, continuous motion and robot training were not run here.
The shared videos are the existing reviewed outputs. Windows issue 8 computation
has not been verified; Linux is the verified runtime for that replay.
