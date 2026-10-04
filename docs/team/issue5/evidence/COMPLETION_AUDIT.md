# Issue 5 completion audit

Pre-publication snapshot, October 3, 2026. **Issue 5 not complete.** The preceding goal turn made progress:
109 passing tests, full official pose-wrapper checks, complete real CAD loading,
decoded stereo-frame correspondence and eight-frame depth diagnostics. Those
results do not prove the requested real pose-accuracy targets.

Revalidated remote state with a successful Git fetch. Main is still ca3f64a;
the issue-4 camera/depth branch is the same commit, with no implementation diff.
PR26 remains ef8a7ac and PR28 remains a6e0cb2. Live issue 4 remains open and has
no comments or delivered camera artifact. TEAM_ACCESS.md names
`rising-gold-junglefowl` but supplies `ssh ubuntu@<public-ip>`, not a host.
The remote-device connector again returned only one offline desktop. Asked the
user for the SSH host or artifact link; the user replied that they are not aware
of one. This dependency has persisted through the original software repair turn,
the real-input validation continuation, and this refreshed completion audit.

| Requirement / experiment | Current authoritative evidence | Verdict |
|---|---|---|
| F1–F3/L1 quaternion, direction, centers, scale, selected GT identity | Production changes and tests-final.xml: actual CLI/export plus asymmetric registration fixtures | Software repair verified; real impact unmeasured |
| F4–F5 shared camera and original/rectified fixed-frame conversion | Hashed-source producer/consumer tests; contracts.json; require-independent rejection | Consumer contract verified; independent producer/input missing |
| F6 temporal rejection, immutable filtering, final residuals | Rejected-jump propagation/export and smoothing/residual fixtures | Software repair verified; real temporal quality unmeasured |
| F7 overlay geometry | Triangle z-buffer, complete-scene and repeated-overlay fixtures; all ten real GLBs match trimesh | Rendering boundary verified; both-camera pot overlays missing |
| F9/L2 five pose kernels and full wrappers | 109-test JUnit includes five-kernel/seven-control parity and full wrapper/four-control parity with invalid-row/provenance checks | Local scorer fidelity verified; no real reconstructed candidate |
| C1 shared/per-object/independent camera comparison, camera disagreement | No frozen VM observations or independent C(t); no regenerated camera/object tracks | Missing |
| O1 original/reviewed masks, FS/SGBM, visible surface bias and silhouette/depth errors | real_assets/local_results.json checks whole-image SGBM only | Partial input diagnostics; masked comparison missing |
| S1 actual-axis spin/tilt, appearance, both-direction relative transforms, flips and lag | No corrected O1 camera/mask/object outputs | Missing |
| E1 all 20 public episode/object reports, median visible <5 cm/<5 degrees | Public GT exists; stereo videos checked for episodes 0 and 2 only; no frozen independent predictions | Missing; target unverified |
| Pot <10 cm/<10 degrees on both ego views and residual explanation | No real accepted pot trajectory or reviewed masks/camera control | Missing; F8 mechanism unresolved |
| Cross-object RPE over issue-stated 17 configurations | Kit audit: 16 two-object configurations, 36 pairs, 5,181 frames, 9,120 rows; actual RPE absent | Roster verified, score missing, seventeenth configuration unidentified |
| Per-episode rotation/translation/Chamfer/pass@5cm10deg and coverage | Report fields preserved in tested fitter; no real report.json for each public episode | Software retained, measured deliverables missing |
| Asset-free fallback note/E2 spoon geometry | Contract and existing plan retain asset-free route; missing spoon geometry recorded | Note retained; separate implementation/readiness dependency |
| Proposed observation, coverage, geometry and temporal gates | Whole-image LR sensitivity supports stereo rectification only | Not proven; no substituted easier thresholds |
| Official artifact completeness and six separate metrics | Kit roster/packer source inspected; no policy/checkpoint/rollout provenance or CD-O result | Not achieved; reconstruction diagnostics cannot establish rollout submission |

Named artifact audit:

- Present: contracts.json; source_manifest.json; software_repairs.patch; tests-final.xml;
  input download manifests; calibration/rectification and GT/source hashes; real-input
  diagnostics and stereo/depth previews; kit roster source; conditional operator commands.
- Missing: original VM depth/mask producer scripts and model/observation lineage;
  immutable regenerated raw/corrected object predictions; shared independent C(t);
  real raw/fill/occlusion status and per-episode/object report.json; both-view pot
  overlays; visible-depth residual plots; axis/spin/relative-motion report;
  real five-pose-metric coverage reports; complete shape/fallback receipt.
- Proposed gates remain proposed. Actual missing measurements have no numerical
  results; they must stay null with a reason when reports are eventually produced.

Remaining dependency: a usable shared-VM login or downloadable prepared observations
and independent camera trajectory with provenance. Replacing the planned issue-4
camera producer, fabricating segmentation, using GT-derived cameras as independent,
or treating whole-image depth as pot accuracy would not satisfy this goal.
At this pre-publication snapshot, no live job was waiting and no remote run,
policy training, commit, push, PR, external message or submission had occurred.
The user subsequently authorized publishing the repairs as a draft PR, with no
merge to main. The draft's Git commits and validation_snapshot.json identify the
published software. Remaining real tracking evidence still requires prepared
inputs or compute access; publishing the draft does not satisfy those gates.
