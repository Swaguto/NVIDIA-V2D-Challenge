# Dex3 controller and physics validation

## Scope and status

The October 15 implementation adds a controller candidate and explicit validation
stages. Local mechanics/schema controls pass; a new GPU comparison is in progress.
No physical task or learned policy has been accepted. Existing preview images,
frozen renderer and historical failed-task receipts remain available.

The saved failed e002 run loses tracking before object contact: the left wrist
first exceeds 10 cm error at source time 0.075 s, versus its first sampled object
contact at 7.09 s. The right exceeds 10 cm at 0.015 s and has no sampled positive
object contact. Both wrists' local X torques change sign ten times in the first
0.05 source seconds and grow toward the original 60 Nm component cap. This is
evidence for controller isolation, not proof of a unique cause.

## Changes under test

- Preserve the existing `project_pd` controller for comparison. Its gains and
  force/torque caps are unchanged.
- Add optional `implicit_pd`: a backward-Euler PD prediction using the actual
  world-frame root 6×6 mass block, root COM velocity and gravity compensation.
  Position error is also measured at the root COM, with explicit xyzw rotations.
  The mass block approximates locked fingers; this is not full inverse dynamics.
- Add separately selected `implicit_pd_ff`: use the next reference pose to
  predict the endpoint and damp velocity error against the reference twist.
  This addresses the moving-target lag measured in the first full-task run.
  Gains, force/torque caps and acceptance limits are unchanged. It supplies no
  external object forces and does not alter the reference trajectory.
- Add separately selected `implicit_pid_ff` for the remaining contact-load error:
  linear integral gain 50 N/(m·s), with each integral component capped at 10 N.
  Updates that push farther into force saturation are frozen, and the state is
  reset when the observation-derived collision mask changes. PD gains and the
  total 60 N / 60 Nm component limits remain unchanged. This remains a controller
  experiment; it cannot establish that the target grasp is physically feasible.
- Initialize floating wrists once after Kit startup with zero velocity. Both
  comparison controllers use this initialization. No wrist pose is teleported
  during execution, and objects receive no motion playback or external forces.
- Free-space tests disable hand collisions and apply a declared 1 cm / 5° initial
  perturbation to test recovery. Full-task tests retain the original display-only
  collision masks and inferred support, without that diagnostic perturbation.
- Record the candidate/executor hashes, root mass eigenvalues, velocities, wrench
  frame, saturation and source provenance. Failures remain visible in new outputs.
  New runs also check and record every actual USD hand collider flag at each mask
  transition; a requested mask without matching readback aborts execution.

Controller-only limits, fixed before the GPU comparison, are p95 position error
≤2 cm, maximum ≤10 cm, p95 orientation error ≤10° and maximum ≤45°, for **each**
hand. No startup interval is excluded. Gravity/contact controls and complete
requested physics-step coverage must also pass. These are diagnostic limits,
not official score thresholds or a physical grasp certificate.

## Run with the published runtime

Follow the [container setup](../../../docker/dex3-review/README.md) first. Use
the digest-pinned registry environment and a fresh `V2D_OUTPUT_DIR`. The overlay
mounts the checked-out candidate code read-only; it does not rebuild or relabel
the published image. The producing source checkout and image digest must both
be recorded for the experiment.

The host must supply an NVIDIA driver compatible with CUDA 13 (580 or newer).
Some Brev VM templates arrive with older drivers even when the GPU is an L40S.
Each physics stage checks the actual host driver and executes a CUDA tensor
calculation in the simulator environment before launching Isaac. An RTX render
is still required after that compute preflight.

Obtain the separate, verified physics inputs:

```bash
gh release download dex3-support-replay-20261013 \
  --repo Swaguto/NVIDIA-V2D-Challenge \
  --pattern CAD-support-dex3-simulator-inputs-v2.tar.gz --dir downloads
printf '\nV2D_CODE_DIR=%s\nV2D_PHYSICS_ARCHIVE=%s\n' \
  "$PWD" "$PWD/downloads/CAD-support-dex3-simulator-inputs-v2.tar.gz" \
  >> docker/dex3-review/.env
```

The input archive SHA256 is
`759b6e9fa642b47ec24306969688cc1ce15ead4cb0b80e4f8d086b727b93c680`.
Every one of the 1,663 case files is verified before execution. The source motion
evaluator and independent source grasp labels are restored from their committed,
hash-bound evidence archive.

```bash
# Separate invocations preserve each result, including an expected baseline failure.
docker compose --env-file docker/dex3-review/.env \
  -f docker/dex3-review/compose.yaml -f docker/dex3-review/physics.compose.yaml \
  run --rm physics --stage baseline-free
docker compose --env-file docker/dex3-review/.env \
  -f docker/dex3-review/compose.yaml -f docker/dex3-review/physics.compose.yaml \
  run --rm physics --stage candidate-free
docker compose --env-file docker/dex3-review/.env \
  -f docker/dex3-review/compose.yaml -f docker/dex3-review/physics.compose.yaml \
  run --rm physics --stage candidate-full
```

The two free-space tests cover the first two source seconds at the original
0.01 s physics timestep and 0.5 motion speed. Full-task execution is refused
unless the saved free-space gate still passes with the same candidate/executor
bytes and selected controller mode. To compare the velocity-feedforward candidate,
add `--candidate-controller implicit_pd_ff` to **both** candidate invocations in a
fresh output root. Use `implicit_pid_ff` for the bounded integral candidate.
Each stage verifies actual video decoding and frame count. The full task
also runs the independent source observation/contact audit before its wrist gate.
Source annotations are evaluation inputs only; they do not fit the controller.

Add `--prepare-only` for input checks in a separate fresh output root. Existing
stage directories are never overwritten. A gate failure returns nonzero and
preserves its video, states and report. Passing wrist tracking does not approve
object retention, inferred targets, collision geometry or policy performance.

## Team reproduction and reference handoff

After the frozen-target container pilots, a teammate can check their deliveries:

```bash
python3 scripts/dex3_review/verify_reproduction.py \
  --inputs /absolute/path/to/dex3-review \
  --outputs /absolute/path/to/review-001 --episodes 11,40,41 \
  --out /absolute/path/to/team-reproduction-check.json
```

This checks reconstruction/IK hashes, exact renderer source, original numerical
tolerances, observation provenance, all four videos and their frame/PTS receipts.
It does not require pixel identity across GPUs. Full manual playback remains a
separate review.

`scripts/dex3_physics/export_reference.py` converts the preserved target contract
to **diagnostic** `motion_v1` dual-hand Parquet and a hash-bound uncertainty sidecar.
It preserves frame indices, gaps, observations and unknown contacts/wrenches;
xyzw target poses are explicitly converted to the schema's wxyz convention.
It does not invent MANO contact geometry or mark a reference training-ready.

The current dual-hand trainer still reads the legacy `ManoSharpaData` contract;
its robot registry includes Sharpa and whole-body G1, not floating Dex3.
Registering the task, integrating confidence masks/contact geometry and proving
checkpoint compatibility are required after the physical controller gate.
Do not run the existing Sharpa command against a file labelled Dex3.

The offline `submission_readiness.py` command reads the supplied official kit's
roster and inventories per-episode checkpoints, object meshes and rollout files.
It distinguishes development previews from the scored roster. Presence alone
does not establish checkpoint compatibility or replace the official packer.
