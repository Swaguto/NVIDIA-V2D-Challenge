# Episode 2 Dex3 pilot results

The complete 343-frame episode 2 motion pipeline runs from PR 34's frozen
refinement output through CHORD differential IK, an audited capsule gate, and
actual Isaac Lab camera rendering. All renderer frames were reviewed and the
17.15-second video played to completion. The result is a reproducible kinematic
pilot with a documented derived collision model. **72 source frames pass the
2 cm capsule gate; 199 are rejected and 72 are unsupported. Issue 8 remains
open.** Passes concern this model at sampled timestamps, not physical ground
truth, continuous motion or successful manipulation.

The source revision is `6a2e24b0e09a8361d589be4d542e410a9d331e30`.
The implementation branch is `Evc-r/issue8-episode2-pilot`, based on PR 34.
This draft includes source, tests, documentation and CPU CI. Private episode
data, derived geometry arrays, videos and archives remain outside Git. Team
review and merge remain the integration gate.

## Complete inputs and trajectory

The bundle retains frames 0–342 at 20 Hz, calibrated camera transforms,
metres, proper object/wrist transforms, separate object scales, hand identity,
source provenance and unknown-data masks. All 1,591 declared source records
and payload hashes validate. Original and SLAM-staged input locks independently
match all 4,673 and 2,961 files respectively.

All 272 left and 284 right supported hand observations have retargeted poses.
The 71 left and 59 right missing observations remain NaN and are hidden in the
simulator. No contact probabilities, forces or torques were invented. Both
supplied object meshes are development CAD and remain submission-ineligible.

| Robot-to-reconstruction error | Left median / p95 / max | Right median / p95 / max |
|---|---|---|
| Wrist position | 10.5 / 17.9 / 18.4 mm | 7.2 / 10.6 / 15.2 mm |
| Three fingertips | 26.8 / 36.2 / 52.1 mm | 17.6 / 26.4 / 36.6 mm |
| Wrist orientation | 7.8 / 9.5 / 14.0 degrees | 3.7 / 6.8 / 7.3 degrees |

These residuals compare the robot with frozen reconstructed observations;
they are not independent real-hand ground truth. A dataset-specific mirrored
left-palm conversion fixed the earlier alignment failure. A separate source-frame
velocity bound inside the QP fixed violations that internal iteration limits
alone had missed. The rejected diagnostics remain preserved.

Independent URDF checks find zero finger position-limit violations and zero
velocity-limit violations across 262 left and 282 right adjacent supported pairs.
Median/p95 wrist speeds are 0.051/0.350 m/s left and 0.064/0.522 m/s right.
Missing intervals reset the warm start and have no asserted continuity bridge.
The largest left fingertip residual occurs at frame 261, while the left hand is
released; the closed three-finger pose does not faithfully reproduce the human's
open free-hand posture there.

## Documented derived collision model and final gate

The user authorized a documented derived model rather than waiting for the
team's watertight assets. The original CAD, source data and IK trajectory are
unchanged. All closure assumptions and intermediate failures are preserved.

The hand-object proxy uses 46 left and 44 right capsules covering tiled
oriented-box volumes of each collision link. This encloses every possible
material point within the original collision-mesh hull, rather than only its
surface. An independent audit checks 737,636 vertices, 246,836 faces and 720
box-cell corners. Palm radii are approximately 10.03 mm. The stricter original
eight-capsule inter-hand check remains unchanged, with zero overlap in all
271 both-hand frames.

Closing CAD boundaries added 667 pot and 333 lid triangles without moving an
original triangle. An independent intersection audit found 101 pot and 24 lid
crossing pairs, so this patched triangle soup was **not accepted as a signed
solid**. It supplies explicit inferred surfaces to a separately defined union
of closed occupied 1 mm cubes. Triangle-box separating-axis tests rasterize
every source and patch triangle, a one-cell envelope thickens the surface, and
enclosed voids are filled using six-neighbour connectivity. The pot opening
remains connected to air; 101 cavity probes and six slice views per object
were reviewed. Enclosed handle interiors may be filled.

The pot has 732,730 material cells and the lid 227,594. Surface-band extension
is bounded by 3.464 mm relative to its input surfaces. This bound does **not**
cover inferred patches or filling relative to the physical object. Mass,
friction, inertia, contact forces and true solid interiors remain unknown.
These development-CAD derivatives are not submission-eligible reconstructions.

The gate score is capsule radius plus signed material depth along its axis.
Axes are sampled at at most 2 mm spacing. Conservative signed-depth bounds
use verified empty-cell witnesses, exact object-transform inverses,
singular-value metric bounds, half the actual sample gap and a 1 micrometre
numerical world margin. This is a conservative capsule proxy, not an exact
robot-mesh penetration depth or minimum separating translation.

| Final disposition, all 343 source frames retained | Frames |
|---|---:|
| Fully checked pass against the documented derived model | 72 |
| Rejected: sampled lower-bound capsule score above 20 mm | 1 |
| Rejected: bound does not certify the 20 mm gate | 198 |
| Unsupported: at least one hand missing | 72 |

Passing frames form 25 source intervals. The longest is frames 238–264,
27 frames or 1.35 seconds at 20 Hz. Frame 72 is the single proxy-score
violation, at 20.00319 mm. The maximum full-axis upper bound is 23.05683 mm;
the maximum accepted upper bound is 19.99738 mm after the numerical margin.
This margin is computational, not an allowance for physical-geometry error.
Collision between timestamps is unverified even inside a passing interval.
Rejected and unsupported frames are excluded from accepted pilot data.

The independent certificate audit verifies 891,101 geometry samples, an
outside witness for each of 960,324 material cells, and all 50,016
capsule-object-frame bounds. Recomputed upper bounds agree within
5.04 × 10⁻¹⁷ m and reproduce exactly the 72-frame acceptance mask.
Four analytic voxel controls cover triangle interiors, irregular cell unions
against an independent empty-cube oracle, a hollow cavity and an axis whose
endpoints are outside but middle is inside. Eight volume-bound controls and
five patch controls accompany the existing 15 bundle/capsule controls.
All **32 controls pass in the actual simulator environment**.

## Historical penetration evidence and exclusions

These original broad-capsule and surface-audit results remain reproducible
historical evidence. Their exclusions and rejection counts do not describe
the final derived-model gate above.

Eight enclosing capsules per hand use the actual URDF collision meshes,
including mesh scales and collision origins. Every mesh vertex is enclosed.
All 271 frames with both hands observed have zero inter-hand capsule overlap.
An independent bounded least-squares segment-distance oracle checks all 17,344
pairs and finds minimum capsule surface clearance of 9.53 mm.

The pot and lid are both nonwatertight. Estimated convex-hull/mesh volume ratios
are 7.72 and 7.98, above the concavity cutoff of 3 in
[CHORD Appendix D](https://arxiv.org/html/2607.00033v2). Both object hull checks
are excluded. All hand-object depth entries remain NaN. There are 271 proxy
pass frames with these exclusions, 72 frames lacking at least one hand, zero
measured threshold violations in that original hull-only gate, and **zero fully checked pass frames**.
Excluded checks cannot establish zero hand-object penetration.

The 11 supported proxy intervals and 12 missing-hand intervals are recorded
in the gate's `quality.json`. No interval is promoted to collision-safe training
data. Grasp contact is partly occluded in the simulator view, so visual inspection
also cannot certify shell/handle clearance, measured forces or stable grasping.

A supplementary enclosing-hull bound was evaluated on every supported frame.
It certified no complete interval; the maximum upper bound was 66.1 mm. Because
these hulls fill concave empty space, this value is not measured CAD penetration
and does not override the exclusions or reject the clip. The diagnostic is
preserved separately as `excluded-hull-upper-bounds.json` and `.npz`.

A subsequent full original-triangle surface audit supersedes the absence of
rejections in the original gate. Native BVH queries cover all supported hands
against both objects. An independent audit verifies 1,112 reported points on
their original triangles and actual capsule axes. It confirms **146 rejected
frames in 17 intervals**, with maximum surface incursion **29.72 mm** and the
closest violating point still 10.47 micrometres above the 20 mm threshold.
All violations involve the left palm capsule. Its 44.94 mm enclosing radius
is conservative; these values are not exact robot-mesh penetration depths.
The added gate retains every excluded signed-depth entry as NaN.

There are 126 remaining both-hand proxy candidates, 72 frames missing at least
one hand, and **zero fully checked accepted frames**. One rejected frame also
has a missing hand. Low unsigned surface incursion cannot rule out complete
containment inside an object solid. Six native analytic controls include this
containment limitation and a sub-millimetre triangle regression. Float32
transform round-trip error peaks at 0.194 micrometres; the independent audit
projects each witness onto the exact world capsule segment before measuring
the rejection. Original geometry and the IK trajectory are unchanged.

Topology inspection finds 11 boundary loops in the pot, with the largest
spanning 237 mm, and 21 in the lid, all spanning at most 8.13 mm. The exact left
palm mesh is also nonwatertight. Closing these holes requires an explicit
derived collision model; it cannot silently establish original solid truth.
The final model above resolves the pilot gate with explicit material geometry
and whole-volume hand enclosure. It does not infer physical truth from a
repaired surface or silently erase earlier conservative rejections.

## Actual simulator evidence

The tested environment uses official Isaac Lab release 2.3.2, commit
`37ddf626871758333d6ed89cf64ad702aef127d0`, and Isaac Sim 5.1.0.0 with Python
3.11.17. The internal Lab core SDK package version is 0.54.2. CPU physics and
Direct3D 12 RTX rendering completed locally on the 8 GB RTX 5060 laptop without
paid compute. This workload does not establish that the below-minimum laptop
can run larger scenes or policy training.

The renderer uses real seven-joint floating Dex3 articulations, full canonical
object triangle meshes and explicitly declared world-to-stage coordinates.
Gravity and collision resolution are configured off for kinematic playback.
The camera is a review viewpoint rather than the calibrated source view, and
the presentation transform is not measured gravity alignment.

Independent checks verify all 556 supported simulator states and all 343 PNGs:

| Simulator fidelity check | Maximum difference |
|---|---|
| Root position | 1.31 × 10⁻⁷ m |
| Finger joint position | 5.96 × 10⁻⁸ rad |
| Root orientation | 3.54 × 10⁻⁷ rad |
| Actual PhysX link position | 1.40 × 10⁻⁷ m |
| Displayed USD link position | 1.40 × 10⁻⁷ m |

Side-color controls detect both hands and zero assigned-color pixels on unknown
hand frames. The independent audit checks the absence of a failure marker as
well as the completion receipt. Simulator fidelity to the IK output is distinct
from tracking fidelity to the reconstructed human.

Earlier diagnostic renders exposed a CPU transform synchronization problem and
native mesh-instancing/material problem. Explicit PhysX-to-USD writeback and
expanding the mesh instances repaired both. Earlier numerical-only replays are
not the accepted visual result. The tested batch shutdown exits after artifacts
are flushed; normal Replicator cleanup had stalled. Importer tip-reference and
performance warnings remain in the log; the final eight mesh-bearing links per
hand have nonempty geometry and matching transforms. Runtime metadata retains a
Starlette discrepancy between Lab and Sim; headless replay works, livestreaming
is untested.

All 343 frames were inspected in seven ordered sheets, with full-resolution
zooms at 60, 97, 112, 227, 261, 264 and 330, and source-event comparisons at 60
and 264. Pot/lid opening, release, return and replacement remain coherent; no
identity swaps or gross finger collapse were observed. Missing observations
produce explicit disappearances at their source boundaries. Performance-mode
surface noise, absent arms/background and imperfect free-hand posture remain.
Browser playback reached 17.15 seconds with no media error or console warnings.
This is still-frame visual inspection plus playback verification, not a claim
of continuous audit at playback speed or metric contact truth from one view.

Replay 10 reruns the same frozen trajectory with the verified supplementary
gate, using the same pinned simulator and source entry point. All 343 frames
were reviewed again in seven ordered sheets and the same seven event zooms.
Browser playback again reached 17.15 seconds without media or console errors.
Independent decoding verifies 146 `REJECTED` and 197 excluded/review labels.
Raw PNGs retain unlabelled renderer pixels. The earlier source-event comparison
remains the reference for the identical trajectory; source video was not reread
for this label update. All simulator fidelity maxima remain unchanged.

Replay 11 is the final pilot replay. It renders all 343 frames with the derived
gate and the unchanged trajectory in the pinned runtime. All seven ordered
sheets were reviewed afresh, with zooms at 60, 72, 97, 112, 227, 261, 264 and
330. Browser playback ended at 17.15 seconds without media or console errors.
Independent decoding verifies all 343 labels: 72 derived-model passes, one
proxy-score rejection, 198 uncertain-bound rejections and 72 missing-hand
frames. Numerical and visual fidelity checks pass. The simulator displays
original CAD with collision resolution disabled; the derived material gate
is calculated separately. Frame 72 contact is occluded in the review view.

## Reproduction and next action

Use [REPRODUCE.md](REPRODUCE.md) and [BUNDLE_CONTRACT.md](BUNDLE_CONTRACT.md).
The parent workspace stores the following private artifacts:

| Artifact | Location |
|---|---|
| Frozen bundle and provenance | `runs/issue8-validation/episode2-bundle-v2` |
| Full trajectory and IK report | `runs/issue8-validation/retarget-trial3` |
| Capsule definitions, masks and intervals | `runs/issue8-validation/penetration-trial2` |
| Verified surface rejections and supplementary gate | `runs/issue8-validation/surface-rejection-checkpoint.json` and `penetration-surface-trial1` |
| Whole-volume hand cover and independent audit | `runs/issue8-validation/volume-capsule-cover-trial1` |
| Derived material model and rendered slices | `runs/issue8-validation/derived-voxel-model-trial1` |
| Full-axis bounds and independent audit | `runs/issue8-validation/derived-voxel-penetration-trial1` |
| Final replay gate | `runs/issue8-validation/penetration-derived-trial1` |
| Latest simulator video, readback, raw frames and review | `C:/v2d8/replay11` |
| Original reviewed simulator baseline | `C:/v2d8/replay9` |
| Independent consolidated report | `runs/issue8-validation/pilot-results.json` |
| Exact runtime closures | `runs/issue8-validation/*runtime-lock.txt` |
| Portable archive and verification | `runs/issue8-validation/episode2-pilot-20261009.tar.gz` and accompanying receipt |
| Surface-audit supplement | `runs/issue8-validation/episode2-surface-audit-20261009.tar.gz` and accompanying receipt |
| Derived-model supplement | `runs/issue8-validation/episode2-derived-collision-20261009.tar.gz` and accompanying receipt |
| Fresh unpacked model and gate reproduction | `runs/issue8-validation/derived-portable-reproduction.json` |

All 32 bundle, capsule, volume-bound, voxel and patch controls pass in
the actual simulator environment. The existing 65 refinement controls pass on
Windows and in clean Linux with the proposed CI pins. The `track3_pilot_cpu.yml`
workflow runs those 65 refinement controls and the 27 bundle/capsule/bounds/voxel
controls included in this PR. The five patch controls are retained in the
private reproduction package. GitHub CI status is tracked separately from
the completed local experiment.

First review PR 34 and the Issue 7 CI package, and agree the proposed Issue 12
contract. Review the derived-model assumptions and rejected intervals. Before
accepting training data or closing Issue 8, validate continuous motion, dynamics
and actual hand-object/grasp behavior. Keep
Issue 6's 20/20 quality requirement and holdouts independent, then run a separate
learned-policy experiment. Preserve the accepted Issue 13 contact-reliability
work and the official requirement for participants' own reconstructed meshes.

The community RunPod repository can inform remote desktop hosting if needed;
its default Sim 6.1 setup is not the tested project runtime. A future Pod should
use a deliberately pinned Lab 2.3.2 / Sim 5.1 image, clone the project and retrieve
the robot LFS assets. Private/licensed inputs need separate transfer and a new
approved cost/shutdown limit. No new cloud spending, GitHub publication, PR state
change, issue closure or team message occurred during the local experiment.

## Artifact hashes

These hashes identify the immutable executed experiment in the private
packages. The draft source includes repository license/formatting preparation
and places all four test modules under `robotic_grounding/tests`; the frozen
executed source and original test layout remain in the private archives.

```text
bundle manifest c6284bb75b9d4cc38d35e625ad52d35e28da1b2cc89723291f25d64462df76df
bundle payload  4de1b9507e42ed66e4341b93cc38563885fc2eda43db72935a546967671b6277
IK payload      47981eac0f6310f6efe5e136242983ae36477b6084911be61f6d569f3b6d48e3
gate manifest   b030bee0e6910e0e001bb5e42bc1fd0a0fc67ff42e49676f3d5c0359fab486d2
baseline video  fe01619d95309c8a5725306a156887e5ff70d911cce4c1e2edb6eda73e74a63d
surface gate    42de8995f748d9c32410b1d88afbd631ff48a96b23a508c8f270f4962373ad02
surface video   9787244ae04d5ce2903da03b991ac5aaaa6b3740feeb4effedfdcbaf1b54a81b
derived model   21819a8b65470202596a74997059f43f5e0925aaf21809802b1877ce1ea61b6c
volume hand     50ebe4366da4c1fec75b508d9846e0fde4bcb9e2e4f106e365ce324b3f5ab2b6
derived gate    87d6e2188bb994e741a1d5ba0fb050dc99589def50af502364e1d4bc6e757879
latest video    dd215b923a5ec5d2ad7eec474c96c2a9b264a9999b08b27a01a16ec46bc25a7f
```
