# Reproducing the episode 2 retargeting checkpoint

The episode 2 pilot now includes a complete real-data IK trajectory, audited
Dex3 capsule gate, and a fully reviewed actual Isaac Lab replay of all 343
frames. It establishes a reproducible kinematic motion pipeline. The final
documented derived-model gate accepts 72 source frames, rejects 199 and marks
72 unsupported. Earlier hull exclusions remain historical evidence. Acceptance
applies to the declared model at sampled timestamps. Issue 8 remains open;
no physics manipulation or
trained-policy success is claimed. See [RESULTS.md](RESULTS.md) for the review
findings, hashes, and remaining integration gates.

Before accepting contact targets, run the locked RGB/mask/stereo comparison in
[SOURCE_ACCURACY.md](SOURCE_ACCURACY.md). It is a CPU diagnostic with pinned
dependencies and explicit path relocation. Its acceptance flags remain false;
source visibility and pose accuracy require separate evidence.

## Export and validate

From `robotic_grounding/`, use a Python environment with NumPy and SciPy. Supply
the accepted Issue 7 scene, native trial directory and its input manifest. Keep
original licensed assets and footage private and use a fresh output directory.

```bash
python source/robotic_grounding/robotic_grounding/retarget/track3_bundle.py \
  --scene /private/trial4/scene.npz --trial /private/trial4 \
  --inputs /private/inputs-slam1/inputs.json --out /private/episode2-bundle \
  --source-commit 6a2e24b0e09a8361d589be4d542e410a9d331e30 --episode 2 \
  --provenance /private/inputs-lock.json \
  --provenance /private/inputs-slam1-lock.json

python source/robotic_grounding/robotic_grounding/retarget/track3_bundle.py \
  --validate /private/episode2-bundle
```

Read [BUNDLE_CONTRACT.md](BUNDLE_CONTRACT.md) for transforms, unknowns, geometry
provenance and the proposed Issue 12 producer/consumer interface.

## Retarget with CHORD

The October 11 integration revision binds new IK outputs to both Dex3 URDFs
and every referenced visual/collision mesh using portable SHA256 records.
Collision checks and playback refuse a different asset set before running.
New trajectories also retain `source_hand_directly_observed` separately from
`source_hand_valid` and `robot_valid`; none of these means pose accuracy is accepted.
The merged refinement and reproduction tools are available in the same checkout.
Record `git rev-parse HEAD`, the manifest hashes and the installed environment
lock for each new result. Historical archives and their expected masks stay frozen.

Rotation order is array-specific. `robot_qpos[..., 3:7]` is XYZW;
`robot_left_frame_poses[..., 3:7]` and `robot_right_frame_poses[..., 3:7]`
are WXYZ. For per-link geometry in SciPy, use
`Rotation.from_quat(link_pose[3:], scalar_first=True)`.
The collision gate already follows this convention. Interchanging these
orders rotates collision geometry incorrectly even if FK positions agree.

Git checkout rules now pin LF bytes for the retarget Python sources and both
Dex3 URDFs. This matters even though CRLF and LF XML describe identical geometry:
receipts deliberately verify bytes. Use a fresh checkout of this revision for
a new reference run; the native Linux repeat and Windows asset checks use the
same LF source/URDF hashes. Its 11 trajectory arrays match the preceding locked
run exactly. Archived locked runs must keep their own recorded source/assets;
do not mix a newly checked-out URDF with an older byte lock or edit its receipt.

The later initialization fix explicitly refreshes Pink FK/Jacobians before the
first QP on each source fit. It changes a few trajectory values and iteration
counts, while keeping the observations and constraints fixed. Its reference
payload is `ca011c10673425ad25cc5badc445fe953905bbb84585ea5a3cbcb2d6fe9def09`;
it must not be compared with the preceding LF run as if they used one solver
revision. In the native IK environment, verify the initialization regression:

```bash
PYTHONPATH=source/robotic_grounding python -m pytest \
  tests/test_hand_initialization.py -q
```

The older archived retarget outputs have no IK asset lock. To inspect those
outputs with current gate/replay code, explicitly add
`--allow-unlocked-legacy-model`. Receipts then record
`robot_asset_lock_verified: false` and `unlocked_legacy_diagnostic: true`.
This mode cannot establish that the robot model matches the original IK model.
For a new baseline, rerun IK and use the default checks. The frozen archive
measurement-reproduction command uses its own original source and needs no flag.

Retrieve the Dex3 hand STL assets **and** their collision OBJ assets through Git
LFS. An unretrieved collision mesh prevents the native Pinocchio model loading.

```bash
python scripts/retarget/track3_to_dex3.py \
  --bundle /private/episode2-bundle --out /private/new-retarget-run
```

The native retargeting environment used Python 3.11, Pinocchio 3.7.0,
pin-pink 4.2.0, qpsolvers 4.9.0, DAQP and CPU PyTorch 2.9.1. Its cmeel native
closure is pinned to the repository's versions. A fresh install requires
NumPy 2.2.6 for that cmeel-boost dependency; this isolated IK environment is
separate from the project's pinned Isaac Lab simulator environment. The complete
installed closure is retained in the private `retarget-runtime-lock.txt`.

Preserve `manifest.json`, `retarget.npz`, `quality.json` and `progress.jsonl`.
The completed candidate is private `runs/issue8-validation/retarget-trial3` in
the parent workspace. The earlier missing-LFS attempt and the rejected alignment
and velocity diagnostic remain preserved.

## Current measurements

All 343 frames were processed; all 272 left and 284 right supported observations
were retargeted. The original 71 left and 59 right missing frames remain unknown.
An independent audit of the output against the URDF confirms zero finger
position-limit violations and zero finger-speed violations across the 262 left
and 282 right adjacent supported frame pairs.

| Error | Left median / p95 | Right median / p95 |
|---|---|---|
| Wrist position | 10.5 / 17.9 mm | 7.2 / 10.6 mm |
| Three fingertip positions | 26.8 / 36.2 mm | 17.6 / 26.4 mm |
| Wrist orientation | 7.8 / 9.5 degrees | 3.7 / 6.8 degrees |

These are robot-to-frozen-reconstruction residuals, not independent hand truth.
The left fingertip maximum is 52.1 mm at frame 261, during the released-hand
interval. It was inspected in the simulator; the three-finger closed pose is
an imperfect match to the human free-hand pose. Do not treat
joint feasibility as a penetration or manipulation success check.

The checkpoint audit is `runs/issue8-validation/current-checkpoint.json`.
Bundle controls pass nine tests. Capsule controls pass six tests. Existing refinement controls pass 65 tests on
Windows and 65 in a clean Linux environment with the proposed CI pins.

## Original capsule gate (historical baseline)

```bash
python scripts/retarget/check_track3_penetration.py \
  --bundle /private/episode2-bundle --retarget /private/new-retarget-run \
  --out /private/new-penetration-run
```

The adapter fits eight capsules per hand from actual collision meshes and
verifies enclosure of every vertex. Capsule convexity also encloses every mesh
triangle. Hand-object checks sample the full capsule axis at no more than 2 mm
spacing and include half the actual sample gap as a conservative distance bound.
Unknown hand frames and excluded object checks remain NaN.

The completed private candidate is `penetration-trial2`. All 271 frames with
both hands observed have zero inter-hand capsule depth. A separate bounded
least-squares distance calculation independently checked all 17,344 capsule
pairs and found minimum surface clearance of 9.53 mm.

Both supplied object meshes are nonwatertight. Their estimated hull-to-mesh
volume ratios are 7.72 for the pot and 7.98 for its lid, so both hull checks are
explicitly excluded. This gives **zero fully checked acceptance frames**, not
zero measured hand-object penetration. The [CHORD Appendix D](https://arxiv.org/html/2607.00033v2)
uses a 2 cm rejection threshold and a hull-to-mesh volume ratio above 3 to
exclude highly concave objects from the hull check.

The gate preserves all 343 frames, the 72 frames lacking at least one hand,
and each supported/unknown interval. Simulator review is complete, with
occluded grasp contact and shell clearance still uncertain. This proxy report
does not establish collision-safe manipulation.

## Actual simulator replay

The pinned Isaac Lab 2.3.2 / Isaac Sim 5.1 workload completed locally on an
8 GB RTX 5060 laptop, driver 596.36, with 16 GB RAM. This small headless replay
does not establish suitability for larger scenes or training. The hardware is
below NVIDIA's listed minima; see the [Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).
No paid RunPod execution was needed. Any later cloud workload requires a
specific image, RTX-capable GPU, cost limit and shutdown proposal before new
spending authorization. Issue 6's spending cap covers its separate workstream.

`scripts/retarget/replay_track3_dex3.py` completed the reviewed trial 9. It
requires matching bundle, IK and gate hashes, checks simulator versions, drives
actual seven-joint articulations, hides unknown hands and validates simulator
root, joint and every visible link position against the frozen targets. It saves
all renderer frames and writes a completion receipt
only after the full 343-frame video decodes at the source rate. Gravity and
collision resolution are disabled for kinematic review; this cannot establish
physics or learned-policy success.

```bash
python scripts/retarget/replay_track3_dex3.py \
  --bundle /private/episode2-bundle --retarget /private/new-retarget-run \
  --gate /private/new-penetration-run --out /private/new-simulator-run \
  --headless --enable_cameras --device cpu --width 1280 --height 960 \
  --rendering_mode performance
```

On October 9, the isolated Windows Python 3.11.17 environment successfully
rendered a nonblank 640×480 cube through Isaac Lab's real camera. The official
Lab checkout is release `2.3.2`, commit
`37ddf626871758333d6ed89cf64ad702aef127d0`; its internal core package version
is `0.54.2`. Simulator distribution version is `5.1.0.0`. A release check must
read Lab's `VERSION` file rather than compare its internal package version to
the release number.

For this Windows laptop, append `--kit_args=--/app/vulkan=false` to use Direct3D
12. The initial Vulkan probe crashed in `rtx.scenedb.plugin.dll`; preserve its
diagnostics rather than describe Vulkan as tested successfully.

The CPU camera path disables Fabric transform rendering, refreshes articulation
FK, and explicitly writes actual PhysX transforms to USD before rendering.
Native-importer mesh instances are expanded to permit the side review materials.
These controls fixed earlier diagnostic replays that had correct root/joint
readback but incorrect displayed link transforms or missing side colors.

The tested batch exit uses the SDK's `SimulationApp.close(skip_cleanup=True)`
after closing the writer, decoding all video frames, and saving readback and
receipts. Ordinary cleanup stalled in unused Replicator orchestration. This
SDK exit can return code zero after an exception: **require `complete.json`
and the absence of `failure.json`; process status alone is insufficient**.
The bounded local controller retained all logs and used a 900-second timeout;
trial 9 exited in 87.9 seconds.

The reviewed replay is private `C:/v2d8/replay9`: `replay.mp4`, all 343 raw PNGs,
`simulator-readback.npz`, frozen source, asset hashes and separate visual review
receipt. Maximum root and displayed-link position differences are below
0.00014 mm. This checks simulator fidelity to the IK target, not target fidelity
to real hands. The full video is 17.15 seconds at 20 fps.

The separate simulator environment uses NumPy 1.26.0 and CPU PyTorch 2.7.0;
physics runs on CPU and RTX rendering uses the laptop GPU. Its installed
closure is recorded in `local-isaac-runtime-lock.txt`. All 15 bundle/capsule
controls also pass in this environment. Core Lab metadata asks for Starlette
0.49.1 while Sim pins 0.45.3; the headless probe retains Sim's pin and does not
exercise livestream services. This known dependency discrepancy prevents
claiming a fully consistent general-purpose runtime.

## Local reproduction and review

Use separate pinned IK and simulator environments; do not install the NumPy
2.2.6 native IK closure into the NumPy 1.26.0 simulator environment. The official
Lab tag is `v2.3.2`, and Sim's official pip packages are
`isaacsim[all,extscache]==5.1.0`. Follow the vendor's installation and license
requirements. Runtime locks are measurements of these tested environments,
not a claim that all pins resolve identically on other operating systems.

From the parent workspace, the independent audits are:

```powershell
& '.venv/Scripts/python.exe' 'runs/issue8-validation/audit_simulator_replay.py' 9
& '.venv/Scripts/python.exe' 'runs/issue8-validation/audit_pilot_checkpoint.py'
```

The final private artifact manifest, portable archive and archive verification
receipt are in `runs/issue8-validation`. The archive preserves the bundle,
trajectory, gate, source/assets, runtime locks, video, raw frames and review
records. It contains development geometry and private derived hand data; it
is not a public release. Team review of PR 34 and the local Issue 7 CI addition
remains the merge gate. Issue 12's all-episode production loader, Issue 6's
20/20 quality acceptance and the accepted Issue 13 experiment remain separate.

## Supplementary original-surface rejection audit

The original portable archive and replay9 are preserved as immutable baselines.
The separate `episode2-surface-audit-20261009.tar.gz` contains the full original
triangle audit, independently verified world witnesses, augmented gate, replay10
and its review. It adds 146 conservative surface-proxy rejections and preserves
all excluded solid-volume checks. Zero frames have fully checked acceptance.

From the parent workspace, reproduce the original-triangle audit in the native
IK runtime with Coal3.0.1. Its output path must be new:

```powershell
wsl --distribution Ubuntu --exec env LD_LIBRARY_PATH=/var/tmp/v2d-issue6-quality-fast/lib /var/tmp/v2d-issue8-runtime/bin/python '/mnt/c/Users/evanc/OneDrive/Desktop/Nvidia V2D/runs/issue8-validation/coal_surface_probe.py' --out '/private/new-surface-audit'
```

The script preserves original geometry, runs six analytic controls and checks
the complete timeline. The independently frozen audit is in
`surface-incursion-full5`; its `frozen-source.py` hash matches the run receipt.
The additional Windows checks are:

```powershell
& 'C:/v2d8/sim/Scripts/python.exe' 'runs/issue8-validation/audit_surface_witnesses.py' --out 'runs/issue8-validation/surface-incursion-full5'
& 'C:/v2d8/sim/Scripts/python.exe' 'runs/issue8-validation/audit_simulator_replay.py' 10
& 'C:/v2d8/sim/Scripts/python.exe' 'runs/issue8-validation/audit_replay_labels.py'
```

`merge_surface_rejections.py` produced `penetration-surface-trial1` only after
the independent witness audit. It requires a fresh output directory and retains
the original hull proxy results separately. Pass that gate to the existing
replay entry point to reproduce rejection labels. The bounded local launcher
is `launch_local_replay_trial10.py`; replay10 completed in 90.3 seconds and its
receipt has no failure marker. Use new paths for reruns; preserve prior receipts.

Unsigned surface overlap can reject a triangle deeply inside an enclosing hand
capsule. It does not certify complete solid containment, exact robot-mesh
penetration or a stable physical grasp. The independent witness audit projects
small float32-transform round-trip errors onto the exact capsule segment before
deciding the threshold. A complete collision model was still the acceptance
gate for that historical audit.

## Final documented derived material model

The user authorized this geometry derivation. Read the definition, uncertainty
scope, independent checks and 72/199/72 disposition in [RESULTS.md](RESULTS.md)
before reusing passing intervals. Original CAD and IK remain unchanged;
physics and collision between timestamps are unverified.

The private `episode2-derived-collision-20261009.tar.gz` supplements the original
`episode2-pilot-20261009.tar.gz`. Verify both against their artifact manifests.
The earlier surface-audit supplement is not needed to recompute the final gate.
The new supplement contains exact source, model arrays, audits, latest video,
all raw frames and review records. It preserves the patched-surface prototype
and its failed self-intersection check; this is not an accepted signed solid.

Use the supplied helpers with NumPy, SciPy and trimesh:

```powershell
& 'C:/v2d8/sim/Scripts/python.exe' 'runs/issue8-validation/restore_derived_pilot.py' `
  --baseline 'runs/issue8-validation/episode2-pilot-20261009.tar.gz' `
  --supplement 'runs/issue8-validation/episode2-derived-collision-20261009.tar.gz' `
  --out 'C:/private/fresh-derived-pilot'

& 'C:/v2d8/sim/Scripts/python.exe' 'runs/issue8-validation/reproduce_derived_pilot.py' `
  --workspace 'C:/private/fresh-derived-pilot'
```

Restore refuses an existing target, verifies every archive member before
writing it and rejects unsafe paths. It maps the baseline into the documented
workspace layout, then applies the supplement. Reproduction refuses existing
rerun directories, regenerates both solid grids from frozen inferred surfaces,
rebuilds the whole-volume hand cover and reruns all 343-frame capsule bounds.
It compares every model and gate array with the frozen result and records
the exact passing mask. The original IK has a separate fresh native-Linux
reproduction receipt: all ten arrays and payload bytes match exactly.

Individual entry points, run from the restored workspace, are:

```powershell
& 'C:/v2d8/sim/Scripts/python.exe' 'runs/issue8-validation/derive_voxel_collision_models.py' `
  --source 'runs/issue8-validation/derived-object-model-trial5' `
  --out 'runs/issue8-validation/new-voxel-model' --pitch 0.001

& 'C:/v2d8/sim/Scripts/python.exe' 'runs/issue8-validation/fit_volume_capsule_cover.py' `
  --out 'runs/issue8-validation/new-hand-cover' --target-radius 0.012

& 'C:/v2d8/sim/Scripts/python.exe' 'runs/issue8-validation/check_derived_voxel_penetration.py' `
  --objects 'runs/issue8-validation/new-voxel-model' `
  --hands 'runs/issue8-validation/new-hand-cover' `
  --out 'runs/issue8-validation/new-derived-bounds'
```

To rerun the simulator, pass the saved `penetration-derived-trial1` gate to
`replay_track3_dex3.py` with the earlier replay command and a fresh output path.
This gate matches the frozen bundle and IK. For a new gate, rebuild its
manifest rather than reusing frozen hashes. The latest reviewed result is
restored under `simulator/replay11`: inspect `complete.json`, `visual-review.json`,
`frame-hashes.json` and `replay.mp4`. Replay 11 completed in 91.98 seconds;
all 343 labels, seven sheets, eight event zooms and 17.15-second browser playback
were verified. Require the completion receipt and absence of a failure marker.

The 32-control command from the restored workspace is:

This command uses the immutable archive's original test layout. In a fresh
checkout of this PR, run the 27 repository controls with the command below it;
the five patch controls require the private reproduction package.

```powershell
$env:PYTHONPATH = (Resolve-Path 'issue8-pilot/robotic_grounding/source/robotic_grounding').Path
& 'C:/v2d8/sim/Scripts/python.exe' -m pytest `
  'issue8-pilot/robotic_grounding/tests/test_track3_bundle.py' `
  'issue8-pilot/robotic_grounding/tests/test_track3_penetration.py' `
  'issue8-pilot/robotic_grounding/tests/test_track3_collision_bounds.py' `
  'issue8-pilot/robotic_grounding/source/robotic_grounding/tests/test_track3_collision_voxels.py' `
  'runs/issue8-validation/test_derived_collision_geometry.py' -q
```

Repository-only CPU controls require NumPy, SciPy, trimesh, rtree and pytest, without
Isaac or private data. From the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path 'robotic_grounding/source/robotic_grounding').Path
python -m pytest 'robotic_grounding/tests/test_track3_bundle.py' `
  'robotic_grounding/tests/test_track3_penetration.py' `
  'robotic_grounding/tests/test_track3_collision_bounds.py' `
  'robotic_grounding/tests/test_track3_collision_voxels.py' -q
```

`.github/workflows/track3_pilot_cpu.yml` runs these 27 controls and the existing
65 Issue 7 refinement controls on the draft's dependency branch and main.
